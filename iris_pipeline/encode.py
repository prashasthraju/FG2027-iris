"""Verification encoder: 1-D log-Gabor IrisCode + masked fractional Hamming distance.

Masek-style: each strip row is filtered along θ with a log-Gabor filter in the
frequency domain; the phase is quantised to 2 bits (sign of real / imag).
Bits with near-zero magnitude or masked texture are excluded from matching.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IrisCode:
    bits: np.ndarray   # bool (H, W, 2)
    mask: np.ndarray   # bool (H, W, 2) True = usable


def log_gabor_1d(w, wavelength, sigma_on_f=0.5):
    f = np.fft.fftfreq(w)
    f0 = 1.0 / wavelength
    g = np.zeros(w)
    pos = f > 0  # one-sided -> analytic (complex) response
    g[pos] = np.exp(-(np.log(f[pos] / f0) ** 2) / (2 * np.log(sigma_on_f) ** 2))
    return g


def encode(strip_rgb, strip_mask, wavelength=24.0, rows=16, mag_frac=0.1) -> IrisCode:
    """strip_rgb (H, W, 3) float [0,1]. Rows are averaged down to `rows` bands."""
    gray = cv2.cvtColor(strip_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    band = cv2.resize(gray, (w, rows), interpolation=cv2.INTER_AREA)
    bmask = cv2.resize(strip_mask.astype(np.float32), (w, rows), interpolation=cv2.INTER_AREA) > 0.9
    # zero-mean per band over usable samples so masked regions don't bias the phase
    mu = np.array([band[i][bmask[i]].mean() if bmask[i].any() else 0 for i in range(rows)])
    band = np.where(bmask, band - mu[:, None], 0.0)
    resp = np.fft.ifft(np.fft.fft(band, axis=1) * log_gabor_1d(w, wavelength)[None], axis=1)
    mag = np.abs(resp)
    good = bmask & (mag > mag_frac * (np.median(mag[bmask]) if bmask.any() else 0))
    bits = np.stack([resp.real > 0, resp.imag > 0], -1)
    return IrisCode(bits, np.stack([good, good], -1))


def hamming(a: IrisCode, b: IrisCode, max_shift=16, min_bits=200):
    """Min masked fractional HD over circular θ shifts (head-roll compensation).
    Returns (hd, best_shift, n_bits); hd = nan if too few common bits."""
    best = (np.nan, 0, 0)
    for s in range(-max_shift, max_shift + 1):
        bb, bm = np.roll(b.bits, s, axis=1), np.roll(b.mask, s, axis=1)
        m = a.mask & bm
        n = int(m.sum())
        if n < min_bits:
            continue
        hd = float(((a.bits ^ bb) & m).sum() / n)
        if not hd >= best[0]:  # also replaces nan
            best = (hd, s, n)
    return best
