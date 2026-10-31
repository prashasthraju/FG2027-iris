"""Stage 4 - video: track iris geometry over a clip, embed one donor consistently.

Three passes over the clip (README.md V1-V6):
  1. measure  - per frame: MediaPipe eyes + the still-image segmentation
                (photometric circles), eyelid metrics, a running masked mean
                of the target's own strip (for shading/colour reference)
  2. smooth   - offline, zero-lag: Hampel outlier rejection, blink / failed-
                refinement frames treated as missing and filled from the eye
                corners, forward-backward One Euro filter on the centre,
                iris radius tied to inter-ocular distance (iris ~11.7 mm)
  3. render   - per frame: live eyelids + speculars, smoothed circles, one
                donor render prepared per clip, rotated with head roll.

Everything that should be constant for a clip (hole filling, colour/shading
reference, anti-alias resolution) is computed once: re-computing it per frame
is what makes video edits flicker.
"""
from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import normalize
from .detect import EyeLandmarks
from .embed import IrisTexture, antialias, embed_into_eye, fill_holes
from .segment import segment_eye

SIDES = ("image_left", "image_right")
BLINK_EAR_FRAC = 0.55   # eye counts as closing below this fraction of the clip's median EAR
# The eyeball counter-rotates against head roll (static ocular counter-roll gain
# ~0.21, Chandrakumar et al. 2011), so the iris turns by (1 - gain) x head roll.
OCULAR_COUNTER_ROLL = 0.2


# ---------------------------------------------------------------- video I/O
def read_frames(path):
    """Yields RGB frames sequentially (seeking is not frame-accurate on H.264)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield np.ascontiguousarray(bgr[:, :, ::-1])
    finally:
        cap.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0, n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


class VideoWriter:
    """RGB frames -> ffmpeg (libx264), audio copied from `audio_from` if it has any."""

    def __init__(self, path, w, h, fps, audio_from=None, crf=16):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{fps}", "-i", "-"]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb):
        self._p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self):
        self._p.stdin.close()
        if self._p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


# ---------------------------------------------------------------- filters
def one_euro(x, fps, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """1€ filter (Casiez et al., CHI 2012) along axis 0. The cutoff rises with
    speed, so slow jitter is smoothed hard while fast moves (saccades) pass."""
    x = np.asarray(x, np.float64)
    te = 1.0 / fps
    a = lambda fc: 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * te))
    out, dx = np.empty_like(x), np.zeros_like(x[0])
    out[0] = x[0]
    for i in range(1, len(x)):
        dx = dx + a(d_cutoff) * ((x[i] - out[i - 1]) / te - dx)
        fc = min_cutoff + beta * np.linalg.norm(np.atleast_1d(dx))
        out[i] = out[i - 1] + a(fc) * (x[i] - out[i - 1])
    return out


def zero_phase(x, fps, **kw):
    """Forward-backward 1€: offline, so cancel the filter's lag."""
    return 0.5 * (one_euro(x, fps, **kw) + one_euro(x[::-1], fps, **kw)[::-1])


def fixation_smooth(x, fps, min_step=0.12, k=4.0, w=3, **kw):
    """Zero-phase 1€ within fixations only. Forward-backward filtering turns a
    saccade (a 1-2 frame step) into a symmetric ramp that starts *before* the
    eye moves, so the track is cut at saccades first. Step detector: distance
    between the means of the w samples after and before each frame boundary
    (peaks exactly at the step; medians plateau and misplace the cut; outliers
    are already removed by hampel()), above max(min_step, k robust sigmas),
    non-maximum suppressed (+-w).
    Units of x (iris radii). Step test, 105 px iris: README.md V3."""
    x = np.asarray(x, np.float64)
    n = len(x)
    D = np.zeros(n)
    for i in range(w, n - w + 1):
        D[i] = np.linalg.norm(x[i:i + w].mean(0) - x[i - w:i].mean(0))
    thr = max(min_step, k * 1.4826 * np.median(D[w:n - w + 1])) if n > 2 * w else np.inf
    peaks = [i for i in range(w, n - w + 1) if D[i] > thr and D[i] == D[max(0, i - w):i + w + 1].max()]
    cuts = set(peaks)
    for i in peaks:  # samples caught mid-saccade (2+ frame saccades) stay raw
        mb, S = x[i - w:i].mean(0), x[i:i + w].mean(0) - x[i - w:i].mean(0)
        for j in (i - 2, i - 1, i, i + 1):
            if 0 <= j < n and 0.2 < np.dot(x[j] - mb, S) / np.dot(S, S) < 0.8:
                cuts |= {j, j + 1}
    cuts = sorted(c for c in cuts if 0 < c < n)
    out = x.copy()
    for a, b in zip([0] + cuts, cuts + [n]):
        if b - a >= 3:
            out[a:b] = zero_phase(x[a:b], fps, **kw)
    return out, len(peaks)


def hampel(x, valid, win=4, k=3.0):
    """Mark samples further than k robust sigmas from the local median as invalid."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    keep = valid.copy()
    for i in np.flatnonzero(valid):
        lo, hi = max(0, i - win), min(len(x), i + win + 1)
        nb = x[lo:hi][valid[lo:hi]]
        if len(nb) < 3:
            continue
        med = np.median(nb, 0)
        mad = 1.4826 * np.median(np.abs(nb - med), 0) + 1e-6
        if np.any(np.abs(x[i] - med) > k * mad + 0.5):  # +0.5: ignore sub-pixel "outliers"
            keep[i] = False
    return keep


def fill_missing(x, valid):
    """Linear interpolation over invalid samples (hold at the ends), per column."""
    x = np.asarray(x, np.float64).reshape(len(x), -1).copy()
    t, v = np.arange(len(x)), np.flatnonzero(valid)
    if len(v) == 0:
        return x
    for c in range(x.shape[1]):
        x[:, c] = np.interp(t, v, x[v, c])
    return x


# ---------------------------------------------------------------- per-frame measurement
def lid_metrics(lid):
    """Eye corners are contour points 0 and 8 (EYE_CONTOURS order). EAR as in
    Soukupova & Cech 2016: mean lid aperture / eye width."""
    c0, c1 = lid[0], lid[8]
    width = float(np.linalg.norm(c1 - c0)) + 1e-6
    aperture = np.mean([np.linalg.norm(lid[i] - lid[16 - i]) for i in (3, 4, 5)])
    return (c0 + c1) / 2, width, aperture / width


@dataclass
class Track:
    """Per-frame raw measurements + smoothed geometry for one clip."""
    fps: float
    n: int = 0
    face: list = field(default_factory=list)                     # bool per frame
    raw: dict = field(default_factory=lambda: {s: [] for s in SIDES})
    ref_sum: dict = field(default_factory=dict)                   # side -> (strip*mask sum, mask sum)
    geom: dict = field(default_factory=dict)                      # filled by smooth()
    roll: np.ndarray | None = None

    def summary(self):
        out = dict(frames=self.n, face_found=int(np.sum(self.face)))
        for s in SIDES:
            g = self.geom.get(s)
            if g is None:
                continue
            out[s] = dict(iris_diam_px_median=round(float(2 * np.median(g["iris"][:, 2])), 1),
                          measured_frames=int(g["measured"].sum()), blink_frames=int(g["closed"].sum()),
                          rejected_outliers=int(g["outliers"]),
                          pupil_measured_frac=round(g["pupil_measured"], 3),
                          saccade_cuts=g["saccade_cuts"], skin_track_fallback_frames=g["skin_fallback"],
                          # 2nd-difference noise of the centre relative to the skin (motion-free)
                          centre_noise_px_raw=round(float(g["jitter_raw"]), 3),
                          centre_noise_px_smoothed=round(float(g["jitter_smooth"]), 3))
        return out


def skin_motion(prev_gray, gray, lid, mid, width):
    """Similarity transform (2x3, prev -> cur) of the skin around one eye:
    Lucas-Kanade on corners in a periocular box, the eye opening (iris, lids,
    lashes) excluded, RANSAC. None if too few features track."""
    h, w = gray.shape
    x0, x1 = int(max(mid[0] - 1.4 * width, 0)), int(min(mid[0] + 1.4 * width, w))
    y0, y1 = int(max(mid[1] - 1.1 * width, 0)), int(min(mid[1] + 1.1 * width, h))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    m = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    cv2.fillPoly(m, [np.round(lid - [x0, y0]).astype(np.int32)], 0)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(width / 12)))
    p0 = cv2.goodFeaturesToTrack(prev_gray[y0:y1, x0:x1], 80, 0.01, max(2.0, width / 15), mask=m)
    if p0 is None or len(p0) < 8:
        return None
    p0 = p0 + np.float32([x0, y0])
    win = int(np.clip(width / 4, 9, 31)) | 1
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(win, win), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return None
    A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                        ransacReprojThreshold=max(0.5, width / 60))
    return A if A is not None and inl.sum() >= 6 else None


def measure(frames, detector, fps, refine="auto"):
    tr = Track(fps)
    prev = None
    for rgb in frames:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        eyes = {e.side: e for e in detector.detect(rgb)}
        tr.face.append(bool(eyes))
        for s in SIDES:
            e = eyes.get(s)
            if e is None:
                tr.raw[s].append(None)
                continue
            mid, width, ear = lid_metrics(e.lid_polygon)
            seg = segment_eye(rgb, e, refine)
            failed = any("inconclusive" in n or "not found" in n for n in seg.notes)
            pupil_ok = not any("not found" in n or "MediaPipe circle" in n for n in seg.notes)
            A = skin_motion(prev, gray, e.lid_polygon, mid, width) if prev is not None else None
            tr.raw[s].append(dict(lid=e.lid_polygon, mid=mid, width=width, ear=ear, mp_centre=e.iris_center,
                                  iris=np.array(seg.iris), pupil=np.array(seg.pupil), failed=failed, pupil_ok=pupil_ok, skin=A))
            # target appearance reference: masked running mean of the strip (open, refined frames)
            if not failed and seg.strip_mask.mean() > 0.3:
                acc, cnt = tr.ref_sum.get(s, (0.0, 0.0))
                m = seg.strip_mask.astype(np.float32)
                tr.ref_sum[s] = (acc + seg.strip * m[..., None], cnt + m)
        prev = gray
        tr.n += 1
    return tr


# ---------------------------------------------------------------- smoothing
def smooth(tr: Track, min_cutoff=1.0, beta=4.0):
    """Turn raw measurements into per-frame circles. The centre filter works in
    skin coordinates and iris radii, so (min_cutoff, beta) depend neither on head
    motion nor on resolution."""
    n, fps = tr.n, tr.fps
    # inter-ocular distance and head roll from the two eye-corner midpoints
    mids = {s: fill_missing([r["mid"] if r else (np.nan, np.nan) for r in tr.raw[s]],
                            np.array([r is not None for r in tr.raw[s]])) for s in SIDES}
    both = np.array([tr.raw["image_left"][i] is not None and tr.raw["image_right"][i] is not None for i in range(n)])
    if not both.any():
        raise RuntimeError("no frame with both eyes detected")
    d = mids["image_right"] - mids["image_left"]
    iod = zero_phase(fill_missing(np.hypot(d[:, 0], d[:, 1]), both), fps, min_cutoff=0.5, beta=0.0)[:, 0]
    roll = zero_phase(fill_missing(np.arctan2(d[:, 1], d[:, 0]), both), fps, min_cutoff=1.0, beta=0.5)[:, 0]
    tr.roll = roll

    for s in SIDES:
        raw = tr.raw[s]
        seen = np.array([r is not None for r in raw])
        if not seen.any():
            continue
        ear = np.array([r["ear"] if r else 0.0 for r in raw])
        open_ear = np.median(ear[seen])
        closed = seen & (ear < BLINK_EAR_FRAC * open_ear)
        meas = seen & ~closed & np.array([bool(r) and not r["failed"] for r in raw])
        if meas.sum() < 2:  # nothing photometric: fall back to MediaPipe circles
            meas = seen & ~closed

        # iris radius = k * IOD (physically constant ratio), k robust over the clip
        ir = np.array([r["iris"][2] if r else np.nan for r in raw])
        k = np.median(ir[meas] / iod[meas])
        r_iris = k * iod
        # pupil ratio: changes slowly (pupil response ~0.3-1 s)
        pr = np.array([r["pupil"][2] / r["iris"][2] if r else np.nan for r in raw])
        pr = zero_phase(fill_missing(pr, meas), fps, min_cutoff=0.5, beta=0.0)[:, 0]

        # centre in skin coordinates: chain the per-frame periocular transforms
        # (T_t maps frame-0 skin coords -> frame t), so head motion drops out and only
        # gaze moves remain for the filter. Where skin tracking fails, fall back to the
        # translation of the MediaPipe eye-corner midpoint. Gaps (blinks, failed fits,
        # outliers) are filled in skin coords: the iris stays put on the face.
        T = np.empty((n, 3, 3))
        T[0] = np.eye(3)
        mid = mids[s]
        for i in range(1, n):
            A = raw[i]["skin"] if raw[i] else None
            step = np.eye(3)
            if A is not None:
                step[:2] = A
            else:
                step[:2, 2] = mid[i] - mid[i - 1]
            T[i] = step @ T[i - 1]
        Tinv = np.linalg.inv(T)
        to = lambda M, p: np.einsum("nij,nj->ni", M[:, :2, :2], p) + M[:, :2, 2]
        c_abs = np.array([r["iris"][:2] if r else (np.nan, np.nan) for r in raw])
        skin_scale = np.sqrt(np.abs(np.linalg.det(T[:, :2, :2])))
        o = to(Tinv, np.nan_to_num(c_abs))
        keep = hampel(o * skin_scale[:, None], meas)
        o = fill_missing(o, keep)
        r_skin = r_iris / skin_scale  # iris radius in skin coords (filter units)
        o_s, n_sacc = fixation_smooth(o / r_skin[:, None], fps, min_cutoff=min_cutoff, beta=beta)
        o_s = o_s * r_skin[:, None]
        c_s = to(T, o_s)

        # pupil centre: small, slowly varying offset from the iris centre (in iris radii)
        po = np.array([(r["pupil"][:2] - r["iris"][:2]) / r["iris"][2] if r else (np.nan, np.nan) for r in raw])
        po = zero_phase(fill_missing(po, keep), fps, min_cutoff=0.5, beta=0.0)

        hf = lambda x: np.linalg.norm(x[keep][2:] - 2 * x[keep][1:-1] + x[keep][:-2], axis=1).mean() / np.sqrt(6) \
            if keep.sum() > 3 else np.nan
        tr.geom[s] = dict(iris=np.column_stack([c_s, r_iris]),
                          pupil=np.column_stack([c_s + po * r_iris[:, None], pr * r_iris]),
                          saccade_cuts=n_sacc,
                          skin_fallback=int(sum(1 for r in raw[1:] if r is not None and r["skin"] is None)),
                          pupil_measured=float(np.mean([r["pupil_ok"] for r, m in zip(raw, meas) if m])),
                          seen=seen, closed=closed, measured=meas, outliers=int(meas.sum() - keep.sum()),
                          jitter_raw=hf(o * skin_scale[:, None]), jitter_smooth=hf(o_s * skin_scale[:, None]))
    return tr


# ---------------------------------------------------------------- rendering
def roll_strip(strip, cols):
    """Fractional circular shift along θ (strip columns), linear interpolation."""
    w = strip.shape[1]
    x = (np.arange(w) - cols) % w
    x0 = np.floor(x).astype(int)
    f = (x - x0)[None, :, None] if strip.ndim == 3 else (x - x0)[None, :]
    return strip[:, x0] * (1 - f) + strip[:, (x0 + 1) % w] * f


def render_eye(frame, seg, donor: IrisTexture, ref_strip, ref_mask, color, **embed_kw):
    """The single call into the embedder (swap point for a better compositor).
    seg carries the clip's time-averaged target strip, so shading/colour
    matching is identical in every frame."""
    seg_ref = dataclasses.replace(seg, strip=ref_strip, strip_mask=ref_mask)
    out, alpha, _ = embed_into_eye(frame, seg_ref, donor, color=color, antialias_on=False, **embed_kw)
    return out, alpha


class ClipRenderer:
    """Holds the per-clip donor preparation; call .frame(i, rgb) in order."""

    def __init__(self, tr: Track, donors: dict, color="donor", follow_roll=True, **embed_kw):
        self.tr, self.color, self.kw = tr, color, embed_kw
        ref_roll = float(np.median(tr.roll))
        self.roll_cols = (1 - OCULAR_COUNTER_ROLL) * (tr.roll - ref_roll) * normalize.STRIP_W / (2 * np.pi) \
            if follow_roll else np.zeros(tr.n)
        self.prep, self.ref = {}, {}
        for s, g in tr.geom.items():
            if s not in donors:
                continue
            acc, cnt = tr.ref_sum.get(s, (None, None))
            if acc is None:
                raise RuntimeError(f"{s}: no usable target frame for the appearance reference")
            self.ref[s] = (acc / np.maximum(cnt, 1e-6)[..., None], cnt > 0.5 * cnt.max())
            # hole filling + anti-aliasing once, at the clip's median iris size
            med = np.median(g["iris"], 0), np.median(g["pupil"], 0)
            dummy = _GeomOnly(tuple(med[1]), tuple(med[0]))
            filled = fill_holes(donors[s])
            self.prep[s] = IrisTexture(antialias(filled, dummy), np.ones(filled.shape[:2], bool), donors[s].source)

    def frame(self, i, rgb):
        out = rgb
        for s, g in self.tr.geom.items():
            r = self.tr.raw[s][i]
            if s not in self.prep or r is None:
                continue  # face/eye not found in this frame: pass through
            iris, pupil = g["iris"][i], g["pupil"][i]
            eye = EyeLandmarks(s, iris[:2], float(iris[2]), r["lid"])  # live eyelids
            seg = segment_eye(out, eye, refine=False, pupil=tuple(pupil))
            tex = self.prep[s]
            if self.roll_cols[i]:
                tex = IrisTexture(roll_strip(tex.strip, self.roll_cols[i]), tex.mask, tex.source)
            src = out
            out, _ = render_eye(out, seg, tex, *self.ref[s], self.color, **self.kw)
            if g["pupil_measured"] < 0.5:
                out = paint_pupil(out, src, seg)
        return out


def paint_pupil(out, src, seg, feather_px=0.7):
    """Assumed pupil (fixed ratio, small irises) is often larger than the real one,
    leaving a ring of the *original* iris visible inside it. Fill the pupil disk
    with the frame's own pupil colour (10th percentile of the disk), lids and
    speculars kept."""
    x0, y0, x1, y1 = seg.bbox
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - seg.pupil[0], yy - seg.pupil[1])
    disk = np.clip((seg.pupil[2] + feather_px - d) / (2 * feather_px), 0, 1)
    lid = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(lid, [np.round(seg.lid_polygon - [x0, y0]).astype(np.int32)], 1.0)
    a = disk * cv2.GaussianBlur(lid, (0, 0), feather_px) * (1 - seg.specular)
    roi = src[y0:y1, x0:x1].astype(np.float32)
    sel = (disk > 0.5) & (lid > 0) & ~seg.specular
    if sel.sum() < 3:
        return out
    dark = np.percentile(roi[sel], 10, axis=0)
    res = out.copy()
    o = res[y0:y1, x0:x1].astype(np.float32)
    res[y0:y1, x0:x1] = np.uint8(np.clip(a[..., None] * dark + (1 - a[..., None]) * o + 0.5, 0, 255))
    return res


@dataclass
class _GeomOnly:
    """Just the circles antialias() reads."""
    pupil: tuple
    iris: tuple
