"""Target-side gate: is this eye fit to receive a donor iris?

D16 showed transfer is limited by the *target* side, so the cheapest gain is to
stop embedding where the result cannot be right. Three independent failures seen
in real data, each with its own check:

  tilt      the eye is turned so far that a circular/elliptical limbus no longer
            describes it. Observed on profile frames of the user's clips, where
            donor texture was painted onto the sclera.
  size      below ~60 px of iris diameter nothing identity-related survives
            (D11 P4), and below ~120 px an identity *source* is unreliable.
  occlusion a mostly closed or lash-covered eye leaves too little iris to carry
            texture; what is left is dominated by lid shadow.

The gate returns a verdict rather than raising, so callers can choose: skip the
eye, fall back to leaving it untouched, or embed anyway and record the risk.
Scores are reported even when the verdict passes, so results can be stratified
by gaze angle and iris size (research/03's requirement for honest reporting).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry, normalize

# Thresholds. `tilt` comes from the measured failure mode, not from theory:
# fit_limbus only adopts an ellipse above ~31 deg, and beyond ~35 deg the
# planar-iris map starts to disagree visibly with the eyeball's real curvature.
MAX_TILT_DEG = 35.0
# Gaze offset: how far the iris sits from the centre of the eye opening, as a
# fraction of the half-width. It is the one tilt cue that needs no ellipse fit,
# so it works with plain circles. Looking straight ahead gives ~0; at the limit
# of comfortable gaze the iris reaches the corner and it approaches 1.
# Calibrated, not guessed: on 48 known-frontal CASIA eyes the offset is
# mean 0.098, p90 0.179, p99 0.249, max 0.276; on a clip of a subject turning to
# profile it runs 0.287 (near-frontal) to 0.473 (turned). 0.30 sits just above
# the frontal maximum, so frontal faces are never refused and turning is caught.
MAX_GAZE_OFFSET = 0.30
MIN_DIAMETER_PX = 60.0
MIN_SOURCE_DIAMETER_PX = 120.0
MIN_VISIBLE_FRAC = 0.35


@dataclass
class EyeVerdict:
    ok: bool
    role: str                       # "target" or "source"
    reasons: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    def __bool__(self):
        return self.ok

    def __str__(self):
        s = ", ".join(f"{k}={v}" for k, v in self.scores.items())
        return ("pass" if self.ok else "REFUSE: " + "; ".join(self.reasons)) + f"  [{s}]"


def gaze_offset(seg):
    """Iris displacement from the centre of the eye opening, in half-widths.

    The eye corners are the two most distant points of the lid contour; their
    midpoint is the centre of the opening. When the eye turns, the iris slides
    towards a corner while the opening stays put, so this rises even though the
    limbus still looks circular. Returns nan for close-ups, whose lid polygon is
    a synthetic ellipse and carries no such information.
    """
    poly = np.asarray(seg.lid_polygon, float)
    if seg.side == "closeup" or len(poly) < 4:
        return float("nan")
    d2 = ((poly[:, None, :] - poly[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    c1, c2 = poly[i], poly[j]
    half = np.hypot(*(c2 - c1)) / 2
    if half < 1e-6:
        return float("nan")
    axis = (c2 - c1) / (2 * half)
    mid = 0.5 * (c1 + c2)
    # project the displacement onto the corner-to-corner axis: vertical lid
    # movement (blinks) must not read as gaze
    return float(abs(np.dot(np.asarray(seg.iris[:2], float) - mid, axis)) / half)


def assess(seg, role="target", max_tilt_deg=MAX_TILT_DEG, min_visible_frac=MIN_VISIBLE_FRAC,
           max_gaze_offset=MAX_GAZE_OFFSET):
    """Judge one segmented eye. `role` picks the size threshold."""
    tilt = geometry.tilt_from_ellipse(seg.iris)
    offset = gaze_offset(seg)
    diam = 2 * normalize.mean_radius(seg.iris)
    visible = float(seg.strip_mask.mean()) if seg.strip_mask is not None else float("nan")
    min_d = MIN_SOURCE_DIAMETER_PX if role == "source" else MIN_DIAMETER_PX

    reasons = []
    if tilt > max_tilt_deg:
        reasons.append(f"eye turned {tilt:.0f} deg (> {max_tilt_deg:.0f}); "
                       "a planar iris model does not hold here")
    if diam < min_d:
        reasons.append(f"iris {diam:.0f} px (< {min_d:.0f} for a {role})")
    if np.isfinite(offset) and offset > max_gaze_offset:
        reasons.append(f"iris sits {offset:.2f} of the way to the eye corner "
                       f"(> {max_gaze_offset}); the eye is turned away")
    if np.isfinite(visible) and visible < min_visible_frac:
        reasons.append(f"only {visible:.0%} of the iris is visible (< {min_visible_frac:.0%})")

    return EyeVerdict(ok=not reasons, role=role, reasons=reasons,
                      scores=dict(tilt_deg=round(float(tilt), 1),
                                  diameter_px=round(float(diam), 1),
                                  gaze_offset=(round(offset, 3) if np.isfinite(offset) else None),
                                  visible_frac=(round(visible, 3) if np.isfinite(visible) else None)))


def gaze_bin(seg, edges=(10.0, 20.0, 35.0)):
    """Coarse gaze bucket for stratified reporting ("0-10", "10-20", ...)."""
    t = geometry.tilt_from_ellipse(seg.iris)
    lo = 0.0
    for e in edges:
        if t < e:
            return f"{lo:.0f}-{e:.0f}"
        lo = e
    return f">{edges[-1]:.0f}"


def size_bin(seg, edges=(60.0, 120.0, 200.0)):
    d = 2 * normalize.mean_radius(seg.iris)
    lo = 0.0
    for e in edges:
        if d < e:
            return f"{lo:.0f}-{e:.0f}px"
        lo = e
    return f">{edges[-1]:.0f}px"
