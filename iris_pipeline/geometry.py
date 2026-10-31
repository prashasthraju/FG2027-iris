"""S1: lid-masked elliptical boundary fitting.

The limbus is a circle on the eyeball, so its image is an **ellipse** whenever
the eye is off-axis. Fitting a circle there costs accuracy exactly where our
targets fail (research/02 §1). The fit here is deliberately classical and
training-free:

  1. cast rays from a prior centre and find the strongest radial intensity step
     on each ray (dark iris -> bright sclera), sub-pixel by parabola;
  2. drop rays that fall under an eyelid, on a specular highlight, or whose
     edge is weak, so the lids cannot drag the ellipse;
  3. fit a conic by RANSAC + Fitzgibbon least squares (cv2.fitEllipse);
  4. reject implausible solutions (axis ratio, size, centre drift) and fall
     back to the caller's circle.

Alternatives considered: Hough ellipse (5-D vote, far too slow and noisy at our
edge quality); snakes/active contours (needs initialisation as good as this and
can leak into lids); a learned regressor (research/02 step 3, needs labels).
"""
from __future__ import annotations

import cv2
import numpy as np

from .normalize import as_ellipse, mean_radius


def radial_edge_points(gray, centre, r_lo, r_hi, valid=None, n_rays=180,
                       polarity=1, samples_per_px=2.0, min_strength=0.0):
    """Strongest radial step along each ray. Returns (points Nx2, strengths N).

    polarity +1: dark inside -> bright outside (the limbus in most images).
    `valid` is a uint8/bool image mask; rays whose edge lands on an invalid
    pixel are discarded (this is how eyelids and speculars are excluded).
    """
    cx, cy = float(centre[0]), float(centre[1])
    n_r = max(int((r_hi - r_lo) * samples_per_px), 8)
    radii = np.linspace(r_lo, r_hi, n_r)
    theta = 2 * np.pi * np.arange(n_rays) / n_rays
    xs = (cx + radii[:, None] * np.cos(theta)[None]).astype(np.float32)
    ys = (cy + radii[:, None] * np.sin(theta)[None]).astype(np.float32)
    prof = cv2.remap(gray.astype(np.float32), xs, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.GaussianBlur(prof, (1, 0), sigmaX=0.0, sigmaY=1.2)  # smooth along r only (ksize width 1)
    d = np.gradient(prof, radii, axis=0) * polarity

    ok = np.ones_like(d, bool)
    if valid is not None:
        vm = cv2.remap(valid.astype(np.uint8), xs, ys, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        ok &= vm
    d_masked = np.where(ok, d, -np.inf)
    idx = np.argmax(d_masked, axis=0)
    strength = d_masked[idx, np.arange(d.shape[1])]

    # sub-pixel refinement: parabola through the three samples around the peak
    i0 = np.clip(idx, 1, len(radii) - 2)
    y0, y1, y2 = d[i0 - 1, np.arange(d.shape[1])], d[i0, np.arange(d.shape[1])], d[i0 + 1, np.arange(d.shape[1])]
    denom = (y0 - 2 * y1 + y2)
    delta = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(denom) > 1e-9, denom, 1), 0.0)
    delta = np.clip(delta, -1, 1)
    step = radii[1] - radii[0]
    r_edge = radii[i0] + delta * step

    keep = np.isfinite(strength) & (strength > min_strength)
    th, re = theta[keep], r_edge[keep]
    pts = np.stack([cx + re * np.cos(th), cy + re * np.sin(th)], 1)
    return pts, strength[keep]


def fit_circle_lsq(points):
    """Algebraic (Kasa) circle fit -> (cx, cy, r). The null model for D14's test."""
    x, y = points[:, 0], points[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9))
    return (float(cx), float(cy), float(r))


def fit_ellipse_ransac(points, iters=200, inlier_px=1.5, min_points=12, seed=0):
    """RANSAC over cv2.fitEllipse. Returns (ellipse, inlier_mask) or (None, None)."""
    n = len(points)
    if n < min_points:
        return None, None
    rng = np.random.default_rng(seed)
    best = (0, None, None)
    pts32 = points.astype(np.float32)
    for _ in range(iters):
        sel = rng.choice(n, size=min(8, n), replace=False)
        try:
            e = cv2.fitEllipse(pts32[sel])
        except cv2.error:
            continue
        res = ellipse_residuals(points, _cv2_to_ours(e))
        inl = res < inlier_px
        if inl.sum() > best[0]:
            best = (int(inl.sum()), e, inl)
    if best[1] is None or best[0] < min_points:
        return None, None
    try:                                    # refit on all inliers
        e = cv2.fitEllipse(pts32[best[2]])
    except cv2.error:
        e = best[1]
    ours = _cv2_to_ours(e)
    return ours, ellipse_residuals(points, ours) < inlier_px


def _cv2_to_ours(e):
    """cv2.fitEllipse ((cx,cy),(MA,ma),angle_deg) -> (cx, cy, a, b, phi_rad)."""
    (cx, cy), (d1, d2), ang = e
    return (float(cx), float(cy), float(d1) / 2, float(d2) / 2, float(np.deg2rad(ang)))


def ellipse_residuals(points, ell):
    """Approximate point-to-ellipse distance (algebraic distance scaled to px)."""
    cx, cy, a, b, phi = as_ellipse(ell)
    a, b = max(a, 1e-6), max(b, 1e-6)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    cp, sp = np.cos(phi), np.sin(phi)
    u, v = (dx * cp + dy * sp) / a, (-dx * sp + dy * cp) / b
    rho = np.hypot(u, v)
    # radial distance along the ray, in pixels
    scale = np.hypot(u * a, v * b) / np.maximum(rho, 1e-9)
    return np.abs(rho - 1.0) * scale


def fit_limbus_constrained(points, R, centre0, k_range=(0.55, 1.0), n_k=19, n_phi=24,
                           huber_px=None):
    """Fit a limbus ellipse with the **semi-major axis locked to R**.

    A circle of radius R on the eyeball projects (weak perspective) to an ellipse
    whose semi-major axis is still R and whose semi-minor axis is R*cos(tilt).
    Locking a removes the degeneracy that makes a free 5-parameter conic chase a
    short, low-curvature arc into an enormous ellipse (observed: b = 1567 px from
    2 angular sectors). Free parameters: centre, axis ratio k, orientation phi.

    Returns (ellipse, median_residual_px).
    """
    from scipy.optimize import least_squares
    huber_px = huber_px or max(1.5, 0.02 * R)

    def resid(params):
        cx, cy, k, phi = params
        return ellipse_residuals(points, (cx, cy, R, R * np.clip(k, 0.3, 1.0), phi))

    best = (np.inf, None)
    for k in np.linspace(*k_range, n_k):
        for phi in np.linspace(0, np.pi, n_phi, endpoint=False):
            r = resid((centre0[0], centre0[1], k, phi))
            cost = float(np.median(r))
            if cost < best[0]:
                best = (cost, (centre0[0], centre0[1], k, phi))
    if best[1] is None:
        return None, np.inf
    out = least_squares(resid, best[1], loss="soft_l1", f_scale=huber_px,
                        bounds=([centre0[0] - 0.3 * R, centre0[1] - 0.3 * R, k_range[0], -np.pi],
                                [centre0[0] + 0.3 * R, centre0[1] + 0.3 * R, 1.0, 2 * np.pi]))
    cx, cy, k, phi = out.x
    ell = (float(cx), float(cy), float(R), float(R * k), float(phi))
    return ell, float(np.median(ellipse_residuals(points, ell)))


def fit_limbus(gray, prior, valid=None, search=(0.85, 1.18), R_major=None,
               max_axis_ratio=1.45, min_axis_ratio=1.15, max_centre_shift=0.15,
               polarity=1, n_rays=180, min_inlier_frac=0.55):
    """Fit the limbus ellipse around a circle/ellipse prior.

    `R_major` is the **true limbus radius in pixels, supplied from outside**, and
    it is what makes the fit well posed. When the eyelids cover the top and
    bottom of the limbus (almost always), only the left and right arcs are
    visible, and those lie on the *minor* axis of a yawed eye. The major axis is
    then unobservable: a circle fitted to those arcs simply returns R*cos(tilt),
    and every ellipse sharing that minor axis fits equally well. Measured on a
    known-tilt sweep: the circle radius tracked R*cos(t) to within 1 px at
    t = 0..35 deg, so deriving the lock from the same image is circular.

    Supply R_major from a scale prior that does not depend on gaze:
      - several captures of the same eye (per-subject median), or
      - the anatomy, iris diameter ~ 0.19 x inter-ocular distance.
    Without it the fit stays free and will usually be rejected as unsupported.

    Returns (ellipse, info). Falls back to `prior` with a reason in info["note"].
    """
    p = as_ellipse(prior)
    R = mean_radius(p)
    pts, strength = radial_edge_points(
        gray, (p[0], p[1]), search[0] * R, search[1] * R, valid=valid,
        n_rays=n_rays, polarity=polarity)
    info = dict(n_edge_points=int(len(pts)), n_rays=n_rays)
    if len(pts) >= 12:
        # keep the stronger half of the edges: weak rays are usually lid shadow
        thr = np.median(strength)
        sel = strength >= thr
        if sel.sum() >= 12:
            pts, strength = pts[sel], strength[sel]
    # robust inliers around the prior circle, then a radius-locked ellipse fit
    circ0 = fit_circle_lsq(pts) if len(pts) >= 12 else p[:3]
    R_obs = float(np.clip(circ0[2], 0.85 * R, 1.18 * R))
    R0 = float(R_major) if R_major else R_obs
    info["R_major"] = round(R0, 2)
    info["R_observed"] = round(R_obs, 2)
    tol = max(1.5, 0.03 * R)
    keep = ellipse_residuals(pts, (circ0[0], circ0[1], R_obs, R_obs, 0.0)) < 3 * tol
    pts_f = pts[keep] if keep.sum() >= 12 else pts
    info["n_fit_points"] = int(len(pts_f))
    ell, res_e = fit_limbus_constrained(pts_f, R0, (circ0[0], circ0[1]))
    if ell is None:
        info["note"] = "ellipse fit failed; using prior"
        return p, info
    inl = ellipse_residuals(pts_f, ell) < tol
    info["n_inliers"] = int(inl.sum())
    inl_pts = pts_f[inl]
    if info["n_inliers"] < max(12, min_inlier_frac * len(pts_f)):
        info["note"] = (f"ellipse unsupported ({info['n_inliers']}/{len(pts_f)} points); using prior")
        return p, info
    # angular coverage: a fit supported only by a short arc is degenerate
    ang = np.arctan2(inl_pts[:, 1] - ell[1], inl_pts[:, 0] - ell[0])
    sectors = len(np.unique(((ang + np.pi) / (np.pi / 6)).astype(int)))
    info["sectors"] = int(sectors)
    if sectors < 7:
        info["note"] = f"ellipse degenerate ({sectors}/12 angular sectors); using prior"
        return p, info
    # model selection: 4 params vs a circle's 3, so require a real improvement
    circ = fit_circle_lsq(inl_pts)
    res_e = float(np.median(ellipse_residuals(inl_pts, ell)))
    res_c = float(np.median(ellipse_residuals(inl_pts, circ)))
    info.update(resid_ellipse_px=round(res_e, 3), resid_circle_px=round(res_c, 3))
    if R_major is None and res_e > 0.75 * res_c:
        info["note"] = (f"ellipse not better than a circle ({res_e:.2f}px vs {res_c:.2f}px); using prior")
        return p, info
    cx, cy, a_, b_, phi = ell
    ratio = max(a_, b_) / max(min(a_, b_), 1e-6)
    shift = np.hypot(cx - p[0], cy - p[1]) / R
    size = mean_radius(ell) / R if R_major is None else 1.0
    # Only adopt an ellipse when the tilt is real. Measured on UTIRIS (near-frontal
    # close-ups): accepting ratio ~1.06 fits -- which are mostly fit noise -- raised
    # EER from 8.90% to 9.33%, because 4 parameters add variance without removing
    # bias. min_axis_ratio 1.15 is a ~31 deg tilt.
    if ratio < min_axis_ratio:
        info["note"] = f"near-circular (ratio {ratio:.3f} < {min_axis_ratio}); using circle"
        return p, info
    if ratio > max_axis_ratio or shift > max_centre_shift or not (0.85 < size < 1.18):
        info["note"] = (f"ellipse rejected (ratio {ratio:.2f}, shift {shift:.2f}R, size {size:.2f}R); using prior")
        return p, info
    info.update(axis_ratio=round(float(ratio), 3), centre_shift_R=round(float(shift), 3),
                size_R=round(float(size), 3))
    return ell, info


def tilt_from_ellipse(ell):
    """Apparent off-axis angle of a circular limbus imaged as this ellipse.

    Under weak perspective the semi-major axis keeps the true radius and the
    minor axis shrinks by cos(tilt), so tilt = acos(b/a). Used by the pose gate
    (we refuse to embed into eyes turned too far) and to report results by gaze.
    """
    _, _, a, b, _ = as_ellipse(ell)
    a, b = max(a, b), min(a, b)
    return float(np.degrees(np.arccos(np.clip(b / max(a, 1e-6), 0.0, 1.0))))
