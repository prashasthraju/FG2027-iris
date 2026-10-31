"""Layered eye compositing: occluders / reflections / iris albedo × shading.

Used by embed_into_eye(method="layered"). Layers:
  L1 target occlusion matte: colour-model iris likelihood + lash black-hat +
     regularised lid boundary, computed at native polar resolution
  L2 donor cleaning: same occlusion model on the donor strip + FSR inpainting
  L3 multi-band (Laplacian pyramid) blending + target limbal transition kept
  L4 blur / noise matching to the target
  L5 cross-spectral: donor red channel as NIR proxy for grayscale targets
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import normalize

LATERAL_DEG = 25


# ---------------------------------------------------------------- polar helpers
def native_polar_size(seg):
    ring = max(seg.iris[2] - seg.pupil[2], 2.0)
    return int(np.clip(round(ring), 8, 256)), int(np.clip(round(2 * np.pi * seg.iris[2]), 64, 2048))


def _lateral_cols(w, deg=LATERAL_DEG):
    th = np.degrees(2 * np.pi * np.arange(w) / w)
    return (np.minimum(th, 360 - th) <= deg) | (np.abs(th - 180) <= deg)


def _wrap_pad(a, p):
    return np.concatenate([a[:, -p:], a, a[:, :p]], axis=1)


def colour_zscore(strip, valid):
    """Robust LAB distance of every sample to the iris colour model built from the
    lateral sectors (rarely covered by lids), middle rows only."""
    h, w = valid.shape
    ref = valid.copy()
    ref[:, ~_lateral_cols(w)] = False
    ref[: h // 5] = ref[h - h // 5:] = False
    if ref.sum() < 30:
        ref = valid
    lab = cv2.cvtColor(strip.astype(np.float32), cv2.COLOR_RGB2LAB)
    med = np.median(lab[ref], 0)
    mad = 1.4826 * np.median(np.abs(lab[ref] - med), 0) + np.array([1.0, 0.5, 0.5])
    return np.sqrt((((lab - med) / mad) ** 2).mean(-1))


def dark_line_score(gray, valid, size):
    """Black-hat (closing - image) normalised by its robust spread: thin dark
    structures such as eyelashes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    v = bh[valid] if valid.any() else bh.ravel()
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-3
    return (bh - med) / mad


def regularise_lids(vis, min_run=None, win=None):
    """Eyelids occlude from the limbus inwards along smooth curves. For each θ,
    find how deep the occlusion reaches from the outer row (until a run of
    visible rows), smooth that depth along θ, and occlude everything beyond it.
    Removes jagged lid edges and fills holes inside the lid region."""
    h, w = vis.shape
    min_run = min_run or max(2, h // 12)
    win = win or max(5, (w // 40) | 1)
    depth = np.zeros(w, np.float32)
    for c in range(w):
        col = vis[::-1, c]  # outer row first
        run, d = 0, h
        for i, v in enumerate(col):
            run = run + 1 if v else 0
            if run >= min_run:
                d = i - min_run + 1
                break
        depth[c] = d
    pad = win
    dp = np.concatenate([depth[-pad:], depth, depth[:pad]])
    from scipy.ndimage import median_filter
    dp = median_filter(dp, size=win, mode="wrap")
    dp = cv2.GaussianBlur(dp.reshape(1, -1), (0, 0), win / 3).ravel()[pad:-pad]
    rows = np.arange(h)[:, None]
    return rows < (h - dp[None, :])  # True = not under a lid


def visibility(base, z, lash, k_colour=3.0, k_strict=2.0):
    """Combine the cues into a visible-iris mask.
    - lid boundary from a *strict* iris test (z < k_strict), plus a small margin,
      so the dark lash/shadow band at the lid edge is not taken for iris;
    - pixel-level occluders (z >= k_colour or lash) count only if connected to
      a lid region or the outer (limbus) row: lashes hang from lids, whereas
      dark crypts float inside the iris and must stay as texture."""
    h, w = base.shape
    lidfree = regularise_lids(base & (z < k_strict) & ~lash)
    margin = max(1, h // 32)
    lidfree &= np.vstack([lidfree[margin:], np.zeros((margin, w), bool)])  # pull boundary inwards
    occ = base & ((z >= k_colour) | lash) & lidfree
    if occ.any():
        pad = 16
        occ_p = _wrap_pad(occ[..., None], pad)[..., 0].astype(np.uint8)
        seed = _wrap_pad((~lidfree)[..., None], pad)[..., 0]
        seed = cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        seed[-1, :] = True  # outer row
        n, lab = cv2.connectedComponents(occ_p, connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab[seed & (occ_p > 0)])] = True
        keep[0] = False
        occ = keep[lab][:, pad:-pad]
    return base & lidfree & ~occ


# ---------------------------------------------------------------- L1 target matte
def target_matte(rgb, seg, lid_polygon=True, k_colour=3.0, k_lash=4.0, soft_px=1.0, occluders="pixel"):
    """Soft visibility of the target's iris (1 = iris we may replace, 0 = lid,
    lash, sclera, specular), returned in image space over seg.bbox.

    occluders="pixel": lid boundary + pixel occluders (lashes, off-colour blobs connected
    to a lid or the limbus), the D9 matte for real captures. occluders="lids": smooth lid
    boundary only. Rendered faces (reenactor output) have no resolvable lashes over the
    iris; there the pixel cues fire on blur and on the darker limbal ring and punch holes
    into the inserted iris."""
    H, W = native_polar_size(seg)
    rgbf = rgb.astype(np.float32) / 255
    strip = normalize.unwrap(rgbf, seg.pupil, seg.iris, H, W)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    x0, y0, x1, y1 = seg.bbox
    spec_full = np.zeros(rgb.shape[:2], np.float32)
    spec_full[y0:y1, x0:x1] = seg.specular
    spec = normalize.unwrap(spec_full, seg.pupil, seg.iris, H, W) > 0.5
    lid = np.ones((H, W), bool)
    if lid_polygon:
        lid_full = np.zeros(rgb.shape[:2], np.float32)
        cv2.fillPoly(lid_full, [np.round(seg.lid_polygon).astype(np.int32)], 1.0)
        lid = normalize.unwrap(lid_full, seg.pupil, seg.iris, H, W) > 0.5

    base = lid & ~spec
    if 2 * seg.iris[2] < 60:  # too few pixels for colour / lash cues (cf. D2a):
        # lid polygon x non-specular, built directly in image space (a polar grid
        # only ~9 rows tall aliases when mapped back)
        h, w = y1 - y0, x1 - x0
        m = np.zeros((h, w), np.float32)
        if lid_polygon:
            cv2.fillPoly(m, [np.round(seg.lid_polygon - [x0, y0]).astype(np.int32)], 1.0)
        else:
            m[:] = 1.0
        m *= 1 - seg.specular.astype(np.float32)
        return np.clip(cv2.GaussianBlur(m, (0, 0), 0.7), 0, 1), base
    z = colour_zscore(strip, base)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lash_size = int(np.clip(seg.iris[2] / 12, 3, 41)) | 1
    disk = np.zeros(rgb.shape[:2], bool)
    cv2.circle(disk.view(np.uint8), (int(seg.iris[0]), int(seg.iris[1])), int(seg.iris[2]), 1, -1)
    lash_full = dark_line_score(gray, disk, lash_size)
    lash = normalize.unwrap(lash_full, seg.pupil, seg.iris, H, W) > k_lash

    if occluders == "lids":
        vis = base & regularise_lids(base & (z < 2.0))
        margin = max(1, int(round(H * float(os.environ.get("IRIS_RX_LIDMARGIN", 1 / 32)))))   # D34 experiment
        vis &= np.vstack([vis[margin:], np.zeros((margin, W), bool)])
    else:
        vis = visibility(base, z, lash, k_colour)
    soft = cv2.GaussianBlur(vis.astype(np.float32), (0, 0), soft_px)
    matte, _ = normalize.wrap(soft, seg.pupil, seg.iris, seg.bbox)
    return np.clip(matte, 0, 1), vis


SOFT_RATIO = 0.48   # D30: fine/mid band ratio inside the target iris; CASIA stills >= 0.51, LivePortrait <= 0.45, PersonaLive <= 0.29


def is_soft_render(rgb, seg):
    """True when the target iris has too little fine detail for pixel-level occluder cues:
    the signature of reenactor output (soft, upsampled eyes), measured, not per model."""
    x0, y0, x1, y1 = seg.bbox
    L = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    r = band_ratio(L, seg.visible)
    return bool(np.isfinite(r) and r < SOFT_RATIO)


def relative_highlights(rgb, seg, k=4.0, max_frac=0.03):
    """Specular highlights found *relative to the iris's own brightness* (white top-hat,
    robust z > k, small blobs only). Soft renders blur speculars into dim blobs that the
    absolute high-V / low-S test misses, so the donor would paint over them."""
    x0, y0, x1, y1 = seg.bbox
    g = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = float(seg.iris[2]) if np.isscalar(seg.iris[2]) else float(np.mean(seg.iris[2:4]))
    size = int(np.clip(r / 4, 3, 61)) | 1
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    disk = np.zeros(g.shape, np.uint8)
    cv2.circle(disk, (int(seg.iris[0] - x0), int(seg.iris[1] - y0)), int(r), 1, -1)
    ref = seg.visible if seg.visible.any() else disk > 0   # statistics from iris pixels, not lid skin / sclera
    v = th[ref]
    med, mad = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-3
    # a highlight must stand out *locally* by a clear absolute step (>= 30 grey levels over its
    # surroundings) as well as statistically; a global brightness rule fails on soft renders,
    # whose over-wide limbus circle takes in bright sclera (LivePortrait catchlight: 145 vs 100)
    cand = ((th - med) / mad > k) & (th > 0.12 * 255) & (disk > 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] < max_frac * disk.sum()
    out = keep[lab]
    return cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


# ---------------------------------------------------------------- L2 donor cleaning
def clean_donor(strip, mask, k_colour=3.0, k_lash=4.0):
    """Occlusion model on the donor strip itself, then FSR inpainting
    (frequency-selective reconstruction: texture-preserving, unlike Telea)."""
    z = colour_zscore(strip, mask)
    gray = cv2.cvtColor(strip.astype(np.float32), cv2.COLOR_RGB2GRAY)
    lash = dark_line_score(_wrap_pad(gray, 8), _wrap_pad(mask[..., None], 8)[..., 0], 5)[:, 8:-8] > k_lash
    vis = visibility(mask, z, lash, k_colour)
    if vis.all():
        return strip.copy(), vis
    p = 32
    src = np.uint8(np.clip(_wrap_pad(strip, p), 0, 1) * 255)
    valid = np.uint8(_wrap_pad(vis[..., None], p)[..., 0]) * 255  # xphoto: non-zero = known
    dst = np.zeros_like(src)
    cv2.xphoto.inpaint(src, valid, dst, cv2.xphoto.INPAINT_FSR_FAST)
    return dst[:, p:-p].astype(np.float32) / 255, vis


# ---------------------------------------------------------------- L4 blur / noise
def _bands(L):
    g1 = cv2.GaussianBlur(L, (0, 0), 1.0)
    g3 = cv2.GaussianBlur(L, (0, 0), 3.0)
    return L - g1, g1 - g3


def band_ratio(L, m):
    b1, b2 = _bands(L)
    if m.sum() < 30:
        return np.nan
    return float(np.std(b1[m]) / (np.std(b2[m]) + 1e-6))


def match_blur(patch, target, m, sigmas=(0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)):
    """Blur the donor patch until its fine/mid band-energy ratio (contrast
    invariant) matches the target's iris region."""
    Lt = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    rt = band_ratio(Lt, m)
    if not np.isfinite(rt):
        return patch, 0.0
    best = (np.inf, 0.0, patch)
    for s in sigmas:
        p = cv2.GaussianBlur(patch, (0, 0), s) if s > 0 else patch
        rp = band_ratio(cv2.cvtColor(p, cv2.COLOR_RGB2GRAY), m)
        if abs(rp - rt) < best[0]:
            best = (abs(rp - rt), s, p)
    return best[2], best[1]


def pupil_noise(target, seg):
    """Sensor noise σ from the (flat, dark) pupil: MAD of the finest band."""
    x0, y0 = seg.bbox[:2]
    h, w = target.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    pm = (xx + x0 - seg.pupil[0]) ** 2 + (yy + y0 - seg.pupil[1]) ** 2 <= (0.7 * seg.pupil[2]) ** 2
    pm &= ~seg.specular
    if pm.sum() < 20:
        return 0.0
    b1, _ = _bands(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY))
    return float(1.4826 * np.median(np.abs(b1[pm] - np.median(b1[pm]))))


# ---------------------------------------------------------------- L3 multi-band blend
def laplacian_blend(a, b, alpha, levels=None):
    """Burt & Adelson multi-band blending of b into a with weight alpha."""
    h, w = alpha.shape
    levels = levels or int(np.clip(np.log2(min(h, w) / 8), 1, 6))
    ga, gb, gm = [a], [b], [alpha[..., None] if a.ndim == 3 else alpha]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        m = cv2.pyrDown(gm[-1])
        gm.append(m[..., None] if a.ndim == 3 and m.ndim == 2 else m)
    out = gm[-1] * gb[-1] + (1 - gm[-1]) * ga[-1]
    for i in range(levels - 1, -1, -1):
        size = (ga[i].shape[1], ga[i].shape[0])
        la = ga[i] - cv2.pyrUp(ga[i + 1], dstsize=size)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        out = cv2.pyrUp(out, dstsize=size) + gm[i] * lb + (1 - gm[i]) * la
    return out


def limbus_edge_px(rgb, seg):
    """Width of the target's own limbus transition (σ of the radial edge on the
    lateral arcs), so the blend keeps the target's limbal ring and its blur."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cx, cy, R = seg.iris
    radii = np.linspace(0.8 * R, 1.2 * R, 81)
    th = np.deg2rad(np.r_[np.arange(-LATERAL_DEG, LATERAL_DEG + 1, 3), np.arange(180 - LATERAL_DEG, 181 + LATERAL_DEG, 3)])
    xs = (cx + radii[:, None] * np.cos(th)[None]).astype(np.float32)
    ys = (cy + radii[:, None] * np.sin(th)[None]).astype(np.float32)
    prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR).mean(1)
    d = np.clip(np.gradient(prof), 0, None)
    if d.sum() <= 0:
        return 2.0
    mu = (d * radii).sum() / d.sum()
    return float(max(np.sqrt((d * (radii - mu) ** 2).sum() / d.sum()), 1.0))
