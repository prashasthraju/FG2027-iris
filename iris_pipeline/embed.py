"""Stage 3 - embed a donor iris texture into a target eye.

Donor input is an `IrisTexture` (normalized strip + mask). Today it comes from
a real face/strip; later a synthetic generator just has to return one.

Steps (see README.md D5):
  1. fill donor holes (eyelids/speculars) in polar space
  2. appearance: donor high-frequency texture x target low-frequency shading;
     colour either kept from donor or matched to target (LAB stats)
  3. anti-alias to the target's native iris resolution
  4. inverse rubber-sheet into the target's circles
  5. feathered alpha = target visible iris region (lids stay on top),
     target speculars re-applied by excluding them from alpha
No Poisson blending: it re-lights the patch from the boundary and washes out
the transplanted texture.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import normalize
from .segment import IrisSegmentation


@dataclass
class IrisTexture:
    """Normalized iris texture - the donor / synthetic-generator interface.

    strip: (64, 512, 3) float RGB in [0, 1]  (PNG files on disk are BGR, as OpenCV writes them)
      rows: row 0 = pupil boundary ... row 63 = limbus (outer iris edge)
      cols: col 0 = θ=0 pointing to image +x (right); θ increases clockwise on
            screen (image y points down), so col 128 = straight down, 256 = left.
    mask: (64, 512) bool, True = usable texture (PNG: white = usable). Holes are inpainted.
    """
    strip: np.ndarray   # float32 (H, W, 3) RGB [0, 1]
    mask: np.ndarray    # bool (H, W)
    source: str = ""

    @classmethod
    def from_segmentation(cls, seg: IrisSegmentation, source=""):
        return cls(seg.strip, seg.strip_mask, source)

    @classmethod
    def from_files(cls, strip_png, mask_png=None):
        s = cv2.imread(str(strip_png), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        s = cv2.resize(s, (normalize.STRIP_W, normalize.STRIP_H), interpolation=cv2.INTER_AREA)
        m = np.ones(s.shape[:2], bool) if mask_png is None else (
            cv2.resize(cv2.imread(str(mask_png), 0), s.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127)
        return cls(s, m, str(strip_png))

    def save(self, prefix):
        prefix = Path(prefix)
        cv2.imwrite(str(prefix) + "_strip.png", np.uint8(np.clip(self.strip, 0, 1) * 255)[:, :, ::-1])
        cv2.imwrite(str(prefix) + "_mask.png", np.uint8(self.mask) * 255)


def _wrap_pad(a, p):
    return np.concatenate([a[:, -p:], a, a[:, :p]], axis=1)


def fill_holes(tex: IrisTexture) -> np.ndarray:
    """Inpaint masked strip samples (Telea), θ-periodic via wrap padding."""
    p = 32
    s8 = np.uint8(np.clip(_wrap_pad(tex.strip, p), 0, 1) * 255)
    holes = np.uint8(~_wrap_pad(tex.mask[..., None], p)[..., 0]) * 255
    if holes.any():
        s8 = cv2.inpaint(s8, holes, 5, cv2.INPAINT_TELEA)
    return s8[:, p:-p].astype(np.float32) / 255


def _masked_lowpass(img, mask, sigma):
    """Normalized convolution: blur only over valid samples (θ-periodic)."""
    p = int(3 * sigma[1]) + 1
    m = _wrap_pad(mask.astype(np.float32)[..., None], p)[..., 0]
    x = _wrap_pad(img, p)
    num = cv2.GaussianBlur(x * m, (0, 0), sigmaX=sigma[1], sigmaY=sigma[0], borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma[1], sigmaY=sigma[0], borderType=cv2.BORDER_REFLECT)
    return (num / np.maximum(den, 1e-6))[:, p:-p], den[:, p:-p]


def match_appearance(donor_filled, donor_mask, target: IrisSegmentation, color="donor", sig=(12.0, 48.0)):
    lab_d = cv2.cvtColor(donor_filled, cv2.COLOR_RGB2LAB)       # L in [0,100]
    lab_t = cv2.cvtColor(target.strip, cv2.COLOR_RGB2LAB)
    tm = target.strip_mask
    # sig = strip px (rows, cols) of the low-pass that defines 'shading': keeps only shading-scale variation
    Ld_low, _ = _masked_lowpass(lab_d[..., 0], np.ones_like(donor_mask), sig)  # holes already filled
    out = lab_d.copy()
    if tm.sum() > 50:
        Lt_low, cov = _masked_lowpass(lab_t[..., 0], tm, sig)
        Lt_mean = lab_t[..., 0][tm].mean()
        shade_t = np.where(cov > 0.05, Lt_low / max(Lt_mean, 1e-3), 1.0)
        shade_d = Ld_low / max(Ld_low.mean(), 1e-3)
        out[..., 0] = lab_d[..., 0] / np.maximum(shade_d, 1e-3) * shade_t
        if color == "target":
            for c in range(3):
                d, t = out[..., c], lab_t[..., c][tm]
                out[..., c] = (d - d.mean()) / (d.std() + 1e-6) * t.std() + t.mean()
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2RGB), 0, 1)


def antialias(strip, seg: IrisSegmentation):
    """Resample to roughly the target's native polar resolution and back, so the
    remap below does not alias at ~20-30 px iris diameters."""
    h, w = strip.shape[:2]
    ring = max(seg.iris[2] - seg.pupil[2], 2.0)
    nw = int(np.clip(2 * np.pi * seg.iris[2], 16, w))
    nh = int(np.clip(ring, 4, h))
    p = 8
    small = cv2.resize(_wrap_pad(strip, p * w // nw), (nw + 2 * p, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w + 2 * (p * w // nw), h), interpolation=cv2.INTER_LINEAR)[:, p * w // nw: p * w // nw + w]


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _is_grayscale(rgb):
    s = rgb[:: max(1, rgb.shape[0] // 64), :: max(1, rgb.shape[1] // 64)].astype(np.int16)
    return np.abs(s[..., 0] - s[..., 1]).mean() < 2 and np.abs(s[..., 1] - s[..., 2]).mean() < 2


def embed_into_eye(target_rgb: np.ndarray, seg: IrisSegmentation, donor: IrisTexture,
                   color="donor", feather_px=0.7, antialias_on=True, method="simple",
                   blur_match=True, add_noise=True, gate=False):
    """Returns (new_rgb uint8, alpha (bbox-sized), donor strip actually rendered).

    method="simple": v1. method="layered": v2.
    color: donor | target | nir (donor red channel as NIR proxy) | auto (nir for
    grayscale targets, else donor).
    gate: refuse eyes a planar iris model cannot describe. On refusal the
    image is returned unchanged with alpha all-zero, and the verdict in `render`."""
    if gate:
        from .quality import assess
        verdict = assess(seg, role="target")
        if not verdict:
            h, w = seg.bbox[3] - seg.bbox[1], seg.bbox[2] - seg.bbox[0]
            return target_rgb, np.zeros((h, w), np.float32), verdict
    if color == "auto":
        color = "nir" if _is_grayscale(target_rgb) else "donor"
    if method == "layered":
        return _embed_layered(target_rgb, seg, donor, color, blur_match, add_noise)
    filled = fill_holes(donor)
    if color == "nir":
        filled, color = np.repeat(filled[..., :1], 3, -1), "target"
    styled = match_appearance(filled, donor.mask, seg, color)
    render = antialias(styled, seg) if antialias_on else styled

    x0, y0, x1, y1 = seg.bbox
    patch, r = normalize.wrap(render, seg.pupil, seg.iris, seg.bbox)

    ring_px = max(seg.iris[2] - seg.pupil[2], 1.0)
    dr = feather_px / ring_px
    radial = _smoothstep(-dr, dr, r) * (1 - _smoothstep(1 - dr, 1 + dr, r))

    h, w = y1 - y0, x1 - x0
    lid = np.zeros((h, w), np.float32)
    cv2.fillPoly(lid, [np.round(seg.lid_polygon - [x0, y0]).astype(np.int32)], 1.0)
    lid = cv2.GaussianBlur(lid, (0, 0), feather_px)
    spec = cv2.GaussianBlur(seg.specular.astype(np.float32), (0, 0), feather_px)
    alpha = np.clip(radial * lid * (1 - spec), 0, 1)

    out = target_rgb.astype(np.float32) / 255
    roi = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = alpha[..., None] * patch + (1 - alpha[..., None]) * roi
    return np.uint8(np.clip(out * 255 + 0.5, 0, 255)), alpha, render


def _embed_layered(target_rgb, seg, donor, color, blur_match=True, add_noise=True):
    from . import composite as C
    # L2 donor cleaning (+ L5 cross-spectral)
    cleaned, _ = C.clean_donor(donor.strip, donor.mask)
    if color == "nir":
        cleaned, color = np.repeat(cleaned[..., :1], 3, -1), "target"
    soft0 = os.environ.get("IRIS_SOFT_MODE", "1") != "0" and 2 * seg.iris[2] >= 60 and C.is_soft_render(target_rgb, seg)
    sig = (float(os.environ.get("IRIS_RX_SIGR", "12")) if soft0 else 12.0, 48.0)   # D35 experiment: finer radial shading on soft renders
    styled = match_appearance(cleaned, np.ones(cleaned.shape[:2], bool), seg, color, sig=sig)
    render = antialias(styled, seg)

    x0, y0, x1, y1 = seg.bbox
    patch, r = normalize.wrap(render, seg.pupil, seg.iris, seg.bbox)
    roi = target_rgb[y0:y1, x0:x1].astype(np.float32) / 255

    # L1 target occluders stay on top
    is_face = seg.side != "closeup"
    # D30: rendered (soft) targets get the lid-only matte and relative speculars;
    # IRIS_SOFT_MODE=0 disables it for before/after experiments
    soft = os.environ.get("IRIS_SOFT_MODE", "1") != "0" and 2 * seg.iris[2] >= 60 and C.is_soft_render(target_rgb, seg)
    matte, _ = C.target_matte(target_rgb, seg, lid_polygon=is_face, occluders="lids" if soft else "pixel")
    ring_px = max(seg.iris[2] - seg.pupil[2], 1.0)
    # keep the target's own limbal transition (its ring + blur) outside 1 - δ
    edge = C.limbus_edge_px(target_rgb, seg)
    # ...but cap it: on small irises it would otherwise hand half the ring (and
    # the target's identity texture) back to the target
    d_out = min(max(1.5 * edge, 1.0) / ring_px, 0.12)
    d_in = min(max(1.0, 0.03 * ring_px) / ring_px, 0.08)
    if soft and os.environ.get("IRIS_RX_DIN"):        # D34 experiment: soften the pupil-side join on soft renders
        d_in = float(os.environ["IRIS_RX_DIN"])
    radial = _smoothstep(0, d_in, r) * (1 - _smoothstep(1 - 2 * d_out, 1 - 0.5 * d_out, r))
    spec_mask = seg.specular | C.relative_highlights(target_rgb, seg) if soft else seg.specular
    spec = cv2.GaussianBlur(spec_mask.astype(np.float32), (0, 0), max(0.7, 0.002 * ring_px * 3))
    alpha = np.clip(matte * radial * (1 - spec), 0, 1)

    # L4 blur + noise match over the replaced region
    m = alpha > 0.5
    # L4 needs pixels: below 60 px the blur / noise estimates come from a few
    # hundred pixels and destroyed the donor texture (RAVDESS ablation, D9)
    hi_res = 2 * seg.iris[2] >= 60
    blur_match, add_noise = blur_match and hi_res, add_noise and hi_res
    if blur_match:
        patch, _ = C.match_blur(patch, roi, m)
    sn = C.pupil_noise(roi, seg) if add_noise else 0.0
    if sn > 0:
        rng = np.random.default_rng(0)
        patch = patch + rng.normal(0, sn, patch.shape[:2]).astype(np.float32)[..., None]

    # D34 experiment (IRIS_RX_HF_GAIN, default 1 = unchanged): tame the donor's fine-band contrast toward the
    # render's. Phase (identity) is untouched by a gain; the cue is the residual-noise ratio (realism_bench).
    g = float(os.environ.get("IRIS_RX_HF_GAIN", "0.6")) if soft else 1.0     # soft renders only: real captures unchanged
    if g != 1.0 and hi_res:
        sg = max(1.0, 0.012 * float(np.mean(seg.iris[2:3])))
        low = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), sg)
        patch = low + g * (patch - low)

    # L3 multi-band blend; outside the region the donor is replaced by the target
    # so coarse pyramid levels cannot pull donor colour into lids / sclera
    hard = (alpha > 0.01).astype(np.float32)[..., None]
    patch = hard * patch + (1 - hard) * roi
    blended = C.laplacian_blend(roi, np.clip(patch, 0, 1).astype(np.float32), alpha.astype(np.float32))

    pb = float(os.environ.get("IRIS_RX_PBLUR", "0.04")) if soft else 0.0
    if pb > 0:      # D34 experiment: soften only the pupil boundary (a thin annulus), keep the donor texture beyond it
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - seg.pupil[0], yy - seg.pupil[1])
        w = 0.25 * ring_px
        band = (1 - _smoothstep(0.5 * w, w, np.abs(dist - float(np.mean(seg.pupil[2:3])))))[..., None].astype(np.float32)
        blurred = cv2.GaussianBlur(blended.astype(np.float32), (0, 0), max(0.5, pb * ring_px))
        blended = band * blurred + (1 - band) * blended

    out = target_rgb.astype(np.float32) / 255
    out[y0:y1, x0:x1] = np.clip(blended, 0, 1)
    return np.uint8(np.clip(out * 255 + 0.5, 0, 255)), alpha, render


def embed_face(target_rgb, segs, donors, **kw):
    """Embed one donor per eye (donors: list aligned with segs, or a single texture)."""
    if isinstance(donors, IrisTexture):
        donors = [donors] * len(segs)
    out = target_rgb
    for seg, d in zip(segs, donors):
        out, _, _ = embed_into_eye(out, seg, d, **kw)
    return out
