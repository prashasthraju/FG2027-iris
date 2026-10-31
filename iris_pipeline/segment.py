"""Stage 2 - iris segmentation: refine circles, build occlusion masks, unwrap.

Circle refinement is Daugman's integro-differential operator,
    max_(r, x0, y0) | G_σ(r) * ∂/∂r ∮ I(x, y) ds / (2πr) |,
restricted to a small search window around the MediaPipe prior and to arc
samples that are actually visible (inside the eyelid polygon).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry, normalize
from .detect import EyeLandmarks

PUPIL_RATIO_FALLBACK = 0.4
# Below this iris diameter the integro-differential refinement chases noise
# (geometry test: MediaPipe-only circles 8/8 eyes recovered vs 4/8 refined at
# ~27 px), so "auto" keeps the MediaPipe circle and a fixed pupil ratio.
REFINE_MIN_DIAMETER_PX = 60


@dataclass
class IrisSegmentation:
    side: str
    pupil: tuple          # (cx, cy, r) px
    iris: tuple           # (cx, cy, r) px
    lid_polygon: np.ndarray
    bbox: tuple           # (x0, y0, x1, y1) region holding the iris
    visible: np.ndarray   # bool, bbox-sized: iris annulus ∩ lids ∖ speculars
    specular: np.ndarray  # bool, bbox-sized: highlights inside the iris disk
    strip: np.ndarray     # float32 (H, W, 3) RGB in [0, 1]
    strip_mask: np.ndarray  # bool (H, W) usable texture
    notes: list = field(default_factory=list)

    @property
    def diameter_px(self):
        return 2 * normalize.mean_radius(self.iris)


def _circle_means(gray, valid, cx, cy, radii, angles):
    """Mean intensity on each circle (only visible samples). -> (len(radii),), coverage."""
    xs = cx + radii[:, None] * np.cos(angles)[None]
    ys = cy + radii[:, None] * np.sin(angles)[None]
    vals = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    ok = cv2.remap(valid, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_NEAREST) > 0
    cnt = ok.sum(1)
    return (vals * ok).sum(1) / np.maximum(cnt, 1), cnt / len(angles)


def integro_differential(gray, valid, c0, r_range, shift, angles, step=0.25, sigma=1.0, shift_step=0.5,
                         polarity=1):
    """Best (cx, cy, r, score) maximising the blurred radial derivative of circle means.
    polarity=+1: dark inside -> bright outside; 0: either sign (visible-light pupils
    lit by a flash can be *brighter* than a dark-brown iris)."""
    radii = np.arange(r_range[0], r_range[1], step)
    best = (c0[0], c0[1], float(np.mean(r_range)), -np.inf)
    offs = np.arange(-shift, shift + 1e-9, shift_step)
    for dx in offs:
        for dy in offs:
            m, cov = _circle_means(gray, valid, c0[0] + dx, c0[1] + dy, radii, angles)
            if cov.min() < 0.3:
                continue
            d = np.gradient(m, step)
            d = cv2.GaussianBlur(d.reshape(1, -1), (0, 0), sigma / step).ravel()
            d = np.abs(d) if polarity == 0 else polarity * d
            i = int(np.argmax(d))
            if d[i] > best[3]:
                best = (c0[0] + dx, c0[1] + dy, float(radii[i]), float(d[i]))
    return best


def intensity_outliers(strip, mask, k=3.0):
    """Sclera / residual highlights (too bright) and lashes (too dark) inside the
    nominal iris region, by robust z-score of L*. Catches sclera leaking into
    the outer rows when the circle overshoots (e.g. off-axis gaze -> elliptic iris)."""
    L = cv2.cvtColor(strip.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0]
    if mask.sum() < 50:
        return np.zeros_like(mask)
    med = np.median(L[mask])
    mad = 1.4826 * np.median(np.abs(L[mask] - med)) + 1e-3
    out = (L > med + k * mad) | (L < med - k * mad)
    return cv2.dilate(out.astype(np.uint8), np.ones((3, 9), np.uint8)) > 0


def reference_outliers(strip, mask, k=3.5, lateral_deg=25):
    """Occlusion mask without lid landmarks (close-ups): build an iris colour model
    from the lateral sectors (θ within ±lateral_deg of horizontal, middle rows),
    which eyelids almost never cover, and flag samples far from it in LAB
    (lid skin, sclera, dense lashes)."""
    h, w = mask.shape
    th = np.degrees(2 * np.pi * np.arange(w) / w)
    lat = (np.minimum(th, 360 - th) <= lateral_deg) | (np.abs(th - 180) <= lateral_deg)
    ref = mask.copy()
    ref[:, ~lat] = False
    ref[: h // 5] = ref[-h // 5:] = False
    if ref.sum() < 50:
        return np.zeros_like(mask)
    lab = cv2.cvtColor(strip.astype(np.float32), cv2.COLOR_RGB2LAB)
    med = np.median(lab[ref], 0)
    mad = 1.4826 * np.median(np.abs(lab[ref] - med), 0) + np.array([1.0, 0.5, 0.5])
    z = np.sqrt((((lab - med) / mad) ** 2).mean(-1))
    out = cv2.GaussianBlur(z, (0, 0), 1.5) > k
    return cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def segment_eye(rgb: np.ndarray, eye: EyeLandmarks, refine="auto", pupil=None,
                lid_landmarks=True, geom="circle", r_major=None) -> IrisSegmentation:
    """pupil: optional (cx, cy, r) override in image px (skips the pupil search).
    geom: "circle" or "ellipse"."""
    notes = []
    if refine == "auto":
        refine = 2 * eye.iris_radius >= REFINE_MIN_DIAMETER_PX
        if not refine:
            notes.append(f"iris {2 * eye.iris_radius:.0f}px < {REFINE_MIN_DIAMETER_PX}px: "
                         f"MediaPipe circle, pupil ratio {PUPIL_RATIO_FALLBACK}")
    h_img, w_img = rgb.shape[:2]
    R0 = eye.iris_radius
    m = int(np.ceil(R0 * 1.6)) + 2
    step, sstep = max(0.25, R0 / 100), max(0.5, R0 / 60)  # keep search cost flat for close-ups
    cx0, cy0 = eye.iris_center
    x0, y0 = max(int(cx0) - m, 0), max(int(cy0) - m, 0)
    x1, y1 = min(int(cx0) + m + 1, w_img), min(int(cy0) + m + 1, h_img)
    crop = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lid = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(lid, [np.round(eye.lid_polygon - [x0, y0]).astype(np.int32)], 1)
    c0 = (cx0 - x0, cy0 - y0)

    yy, xx = np.mgrid[: gray.shape[0], : gray.shape[1]]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    def specular_mask(circ):
        disk = normalize.inside(circ, xx, yy)
        inside = disk & (lid > 0)
        v = hsv[..., 2].astype(np.float32)
        thr = max(np.percentile(v[inside], 50) + 2.5 * v[inside].std(), 0.6 * 255) if inside.any() else 255
        spec = inside & (v > thr) & (hsv[..., 1] < 110)
        return disk, inside, cv2.dilate(spec.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    iris_c = (*c0, R0)
    pupil_c = None
    if pupil is not None:
        pupil_c = (pupil[0] - x0, pupil[1] - y0, pupil[2])
    elif refine:
        # --- pupil first (Daugman's order): at >= 60 px it is the strongest, most
        # reliable edge. Wide centre window absorbs MediaPipe centre jitter.
        # Colour: saturation edge of either sign (a flash-lit pupil is achromatic
        # but can be brighter than a dark-brown iris). Grayscale / NIR: dark inside.
        disk0, inside0, spec0 = specular_mask(normalize.scale((*c0, R0), 1.1))
        is_colour = inside0.any() and hsv[..., 1][inside0].mean() > 25
        feat, pol = (hsv[..., 1].astype(np.float32), 0) if is_colour else (gray, 1)
        valid_p = ((lid > 0) & ~spec0).astype(np.uint8)
        lo, hi = 0.12 * R0, 0.68 * R0
        pcx, pcy, pr, ps = integro_differential(
            feat, valid_p, c0, (lo, hi), 0.25 * R0, np.deg2rad(np.arange(0, 360, 6)),
            step=step, shift_step=sstep, polarity=pol)
        noise = feat[inside0 & ~spec0].std() if (inside0 & ~spec0).any() else np.inf
        margin = 0.03 * (hi - lo) + step
        if np.isfinite(ps) and ps > 0.5 * noise / max(0.1 * R0, 1) and lo + margin < pr < hi - margin:
            pupil_c = (pcx, pcy, pr)

        # --- limbus around the pupil centre, lateral arcs only (top/bottom under lids)
        lat = np.deg2rad(np.r_[np.arange(-40, 41, 4), np.arange(140, 221, 4)])
        lo, hi = 0.8 * R0, 1.2 * R0
        lc = pupil_c[:2] if pupil_c is not None else c0
        cx, cy, r, score = integro_differential(gray, lid, lc, (lo, hi), 0.1 * R0 + 1, lat,
                                                step=step, shift_step=sstep)
        if np.isfinite(score) and score > 0 and lo + 0.03 * (hi - lo) + step < r < hi - 0.03 * (hi - lo) - step:
            iris_c = (cx, cy, r)
        else:
            notes.append("limbus refinement inconclusive; using MediaPipe circle")

        # --- D14: elliptical limbus (off-axis). The circle above is the prior.
        if geom == "ellipse":
            # lid-masked rays (research/02): lids, speculars and lashes must not
            # drag the conic. The lash detector is the one already used for mattes.
            from .composite import dark_line_score
            lash_px = int(np.clip(R0 / 12, 3, 41)) | 1
            lash = dark_line_score(gray, (lid > 0) & disk0, lash_px) > 3.0
            lash = cv2.dilate(lash.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            valid_e = ((lid > 0) & ~spec0 & ~lash).astype(np.uint8)
            ell, einfo = geometry.fit_limbus(gray, iris_c, valid=valid_e, R_major=r_major)
            if "note" in einfo:
                notes.append(einfo["note"])
            else:
                iris_c = ell
                notes.append(f"ellipse limbus: axis ratio {einfo['axis_ratio']}, "
                             f"tilt {geometry.tilt_from_ellipse(ell):.0f} deg")
        # pupil must sit inside the iris, roughly centred
        R_iris = normalize.mean_radius(iris_c)
        if pupil_c is not None and (normalize.mean_radius(pupil_c) > 0.75 * R_iris or
                                    np.hypot(pupil_c[0] - iris_c[0], pupil_c[1] - iris_c[1]) > 0.25 * R_iris):
            pupil_c = None
    if pupil_c is None:
        # concentric and, for an elliptical limbus, the same shape (the pupil is
        # also a circle on the same plane, so it projects with the same axes)
        ie = normalize.as_ellipse(iris_c)
        pupil_c = normalize.scale(ie, PUPIL_RATIO_FALLBACK) if geom == "ellipse" else \
            (iris_c[0], iris_c[1], PUPIL_RATIO_FALLBACK * normalize.mean_radius(iris_c))
        if refine:
            notes.append(f"pupil edge not found; concentric pupil at ratio {PUPIL_RATIO_FALLBACK}")

    disk, inside, spec = specular_mask(iris_c)
    pupil_mask = normalize.inside(pupil_c, xx, yy)
    visible = disk & ~pupil_mask & (lid > 0) & ~spec

    to_img = lambda c: normalize.translate(c, x0, y0) if len(c) == 5 else (c[0] + x0, c[1] + y0, c[2])
    pupil_g, iris_g = to_img(pupil_c), to_img(iris_c)
    rgbf = rgb.astype(np.float32) / 255.0
    # large irises (close-ups) are downsampled into 512 columns: low-pass first
    down = 2 * np.pi * normalize.mean_radius(iris_g) / normalize.STRIP_W
    if down > 1.5:
        rgbf = rgbf.copy()
        rgbf[y0:y1, x0:x1] = cv2.GaussianBlur(rgbf[y0:y1, x0:x1], (0, 0), 0.5 * down)
    strip = normalize.unwrap(rgbf, pupil_g, iris_g)
    vis_full = np.zeros((h_img, w_img), np.float32)
    vis_full[y0:y1, x0:x1] = visible
    strip_mask = normalize.unwrap(vis_full, pupil_g, iris_g) > 0.99
    strip_mask &= ~intensity_outliers(strip, strip_mask)
    if not lid_landmarks:
        strip_mask &= ~reference_outliers(strip, strip_mask)

    return IrisSegmentation(eye.side, pupil_g, iris_g, eye.lid_polygon, (x0, y0, x1, y1),
                            visible, spec, strip, strip_mask, notes)


def extract_irises(rgb, detector, refine="auto", geom="circle", r_major=None) -> list[IrisSegmentation]:
    return [segment_eye(rgb, e, refine, geom=geom, r_major=r_major) for e in detector.detect(rgb)]


def _global_limbus_search(rgb, work=120):
    """Daugman-style whole-image search for the limbus on a downscaled copy:
    centres on a grid over the central region, lateral arcs (lids cover the
    top/bottom), max blurred radial derivative. Integrating whole arcs makes it
    tolerant of the soft edges that defeat Canny/Hough on visible-light close-ups."""
    h, w = rgb.shape[:2]
    k = work / min(h, w)
    g = cv2.cvtColor(cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    gh, gw = g.shape
    valid = np.ones_like(g, np.uint8)
    lat = np.deg2rad(np.r_[np.arange(-45, 46, 5), np.arange(135, 226, 5)])
    radii = np.arange(0.12 * work, 0.45 * work, 0.5)
    best = (-np.inf, None)
    for cy in np.arange(0.25 * gh, 0.75 * gh, 2.0):
        for cx in np.arange(0.2 * gw, 0.8 * gw, 2.0):
            m, cov = _circle_means(g, valid, cx, cy, radii, lat)
            d = cv2.GaussianBlur(np.gradient(m, 0.5).reshape(1, -1), (0, 0), 2.0).ravel()
            d[cov < 0.9] = -np.inf  # circle must stay inside the image
            i = int(np.argmax(d))
            if d[i] > best[0]:
                best = (d[i], (cx / k, cy / k, radii[i] / k))
    if best[1] is None:
        raise ValueError("no iris circle found in close-up; pass circles=(cx, cy, r) / --donor-circles")
    return best[1]


def segment_closeup(rgb: np.ndarray, circles=None, refine="auto", geom="circle",
                    r_major=None) -> IrisSegmentation:
    """Donor path for eye/iris close-ups (UBIRIS, UTIRIS, CASIA-style), where
    MediaPipe finds no face. Iris prior = whole-image integro-differential search
    (or `circles` = (cx, cy, r[, pcx, pcy, pr]) given by hand), then the same
    refinement and unwrapping as the face path. There are no eyelid landmarks:
    the lid polygon is a wide ellipse, and lids/lashes are removed by the
    intensity-outlier mask.
    """
    pupil = None
    if circles is not None:
        cx, cy, r = circles[:3]
        if len(circles) >= 6:
            pupil = tuple(circles[3:6])
    else:
        cx, cy, r = _global_limbus_search(rgb)
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    lid = np.stack([cx + 2.5 * r * np.cos(t), cy + 1.6 * r * np.sin(t)], 1)  # generous, no lid landmarks
    eye = EyeLandmarks("closeup", np.array([cx, cy]), float(r), lid)
    seg = segment_eye(rgb, eye, refine, pupil=pupil, lid_landmarks=False, geom=geom,
                      r_major=r_major)
    seg.notes.insert(0, "close-up donor: global integro-differential prior" if circles is None
                     else "close-up donor: manual circles")
    return seg
