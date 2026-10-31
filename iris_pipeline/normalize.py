"""Daugman rubber-sheet model, forward (unwrap) and inverse (wrap back).

    P(r, θ) = (1 - r) · p(θ) + r · s(θ)

where p and s are the pupil and limbus boundary curves. Both directions use this
single map, so an embedded strip lands exactly where an extracted strip came
from, even when the two boundaries are not concentric.

**Boundaries are ellipses**: `(cx, cy, a, b, phi)`, with `phi` in radians.
A circle `(cx, cy, r)` is accepted everywhere and promoted to `a = b = r`,
`phi = 0`, so nothing that passed circles changes.

Why ellipses: the limbus is a circle on the eyeball, so its image is an ellipse
whenever the eye is off-axis. θ is the *parametric* angle of the ellipse, which
under weak perspective is exactly the uniform angular sampling of the underlying
circle. That is what makes the off-axis strip comparable to a frontal one.
"""
from __future__ import annotations

import cv2
import numpy as np

STRIP_H, STRIP_W = 64, 512  # radial x angular samples (typical, cf. open-iris)


# ------------------------------------------------------------------ geometry
def as_ellipse(c):
    """(cx, cy, r) -> (cx, cy, r, r, 0);  (cx, cy, a, b, phi) passes through."""
    c = tuple(float(v) for v in c)
    if len(c) == 3:
        return (c[0], c[1], c[2], c[2], 0.0)
    if len(c) == 5:
        return c
    raise ValueError(f"expected a circle (cx, cy, r) or ellipse (cx, cy, a, b, phi), got {c}")


def is_circle(c, tol=1e-6):
    cx, cy, a, b, phi = as_ellipse(c)
    return abs(a - b) <= tol * max(a, 1.0)


def mean_radius(c):
    """Scalar size of a boundary; for an ellipse the geometric mean of the axes."""
    _, _, a, b, _ = as_ellipse(c)
    return float(np.sqrt(a * b))


def scale(c, k):
    """Scale a boundary about its own centre."""
    cx, cy, a, b, phi = as_ellipse(c)
    return (cx, cy, a * k, b * k, phi)


def translate(c, dx, dy):
    cx, cy, a, b, phi = as_ellipse(c)
    return (cx + dx, cy + dy, a, b, phi)


def inside(c, xs, ys):
    """Boolean membership test for a circle or ellipse."""
    cx, cy, a, b, phi = as_ellipse(c)
    dx, dy = xs - cx, ys - cy
    cp, sp = np.cos(phi), np.sin(phi)
    u, v = (dx * cp + dy * sp) / max(a, 1e-6), (-dx * sp + dy * cp) / max(b, 1e-6)
    return u * u + v * v <= 1.0


# theta is measured in the IMAGE frame, not in the ellipse's own frame: the
# parametric angle used is t = theta - phi. Equivalently, the ellipse is
# rectified to a circle by an affine map that leaves the image x-axis fixed, and
# theta is the angle there ("frontal projection"). This matters because phi is
# nearly unconstrained for a near-circular limbus: measured on UTIRIS, phi varies
# by 36 deg (std) between captures of the SAME eye at axis ratio 1.06. Anchoring
# theta to phi rotated the strip origin by that much and raised EER from 8.9% to
# 40.9%; anchoring it to the image frame removes the dependence entirely, and
# reduces to the circle convention when a == b.
def _boundary(c, theta):
    """Point on the boundary at image-frame angle theta."""
    cx, cy, a, b, phi = as_ellipse(c)
    ct, st = np.cos(theta - phi), np.sin(theta - phi)
    cp, sp = np.cos(phi), np.sin(phi)
    return (cx + a * ct * cp - b * st * sp,
            cy + a * ct * sp + b * st * cp)


def _boundary_dtheta(c, theta):
    """d/dtheta of _boundary (needed by the Newton inverse)."""
    _, _, a, b, phi = as_ellipse(c)
    ct, st = np.cos(theta - phi), np.sin(theta - phi)
    cp, sp = np.cos(phi), np.sin(phi)
    return (-a * st * cp - b * ct * sp,
            -a * st * sp + b * ct * cp)


# ------------------------------------------------------------------ forward
def forward_grid(pupil, iris, h=STRIP_H, w=STRIP_W):
    """Image coords (map_x, map_y) of each strip sample; rows = r, cols = θ."""
    theta = 2 * np.pi * np.arange(w) / w
    r = (np.arange(h) + 0.5) / h
    px, py = _boundary(pupil, theta)
    sx, sy = _boundary(iris, theta)
    mx = (1 - r)[:, None] * px[None] + r[:, None] * sx[None]
    my = (1 - r)[:, None] * py[None] + r[:, None] * sy[None]
    return mx.astype(np.float32), my.astype(np.float32)


def unwrap(img, pupil, iris, h=STRIP_H, w=STRIP_W, interp=cv2.INTER_LINEAR):
    mx, my = forward_grid(pupil, iris, h, w)
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ inverse
def inverse_coords(xs, ys, pupil, iris, iters=12):
    """Solve P(r, θ) = (x, y) for every pixel with Newton's method.

    Returns r (0 = pupil edge, 1 = limbus; outside [0,1] = not iris) and θ ∈ [0, 2π).
    """
    pe, se = as_ellipse(pupil), as_ellipse(iris)
    scx, scy, sa, sb, sphi = se
    # initial guess: parametric angle and radius in the limbus ellipse's own frame
    dx, dy = xs - scx, ys - scy
    cp, sp = np.cos(sphi), np.sin(sphi)
    u, v = (dx * cp + dy * sp) / max(sa, 1e-6), (-dx * sp + dy * cp) / max(sb, 1e-6)
    theta = np.arctan2(v, u) + sphi          # parametric -> image frame
    r = np.clip(np.hypot(u, v), 0, 1.5)
    for _ in range(iters):
        px, py = _boundary(pe, theta)
        qx, qy = _boundary(se, theta)
        dpx, dpy = _boundary_dtheta(pe, theta)
        dqx, dqy = _boundary_dtheta(se, theta)
        fx = (1 - r) * px + r * qx - xs
        fy = (1 - r) * py + r * qy - ys
        # Jacobian of P wrt (r, θ)
        j11, j21 = qx - px, qy - py
        j12 = (1 - r) * dpx + r * dqx
        j22 = (1 - r) * dpy + r * dqy
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        r = r - (j22 * fx - j12 * fy) / det
        theta = theta - (-j21 * fx + j11 * fy) / det
    return r, np.mod(theta, 2 * np.pi)


def wrap(strip, pupil, iris, bbox, interp=cv2.INTER_LINEAR):
    """Render a strip into image space over bbox=(x0, y0, x1, y1).

    Returns (patch, r) where r is the radial coordinate per pixel (for masking).
    The strip is wrap-padded in θ so sampling across 0/2π is seamless.
    """
    x0, y0, x1, y1 = bbox
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    r, theta = inverse_coords(xs, ys, pupil, iris)
    h, w = strip.shape[:2]
    pad = 4
    padded = np.concatenate([strip[:, -pad:], strip, strip[:, :pad]], axis=1)
    mx = (theta / (2 * np.pi) * w + pad).astype(np.float32)
    my = np.clip(r * h - 0.5, 0, h - 1).astype(np.float32)
    patch = cv2.remap(padded, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
    return patch, r
