"""Building blocks of the proxy-iris attribution test (README.md):
iris-code extraction from a video, consensus code by per-bit majority vote, proxy templates, Hamming matching.

The extractor is the project's own log-Gabor IrisCode (iris_pipeline.encode) on strips that are RE-DETECTED from the
video frames (D7: MediaPipe mesh + circle, never the embedder's geometry). Strips live in the image theta frame,
so frames need no alignment before the vote.
"""
import base64
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIDES = ("image_left", "image_right")
MIN_VISIBLE = 0.35        # per-observation floor: usable strip fraction (closed eyes, motion blur and occlusion fall below)
MIN_OBS = 5               # observations needed per side for a consensus code
MIN_VOTES = 3             # votes needed per bit position
EXTRACTOR = "irislg-0.1"  # our log-Gabor IrisCode; named in the ledger parameters record


def read_frames(path, every=1):
    import cv2
    cap, out, i = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % every == 0:
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return out


def frame_codes(frames, det):
    """[(frame_index, side, IrisCode, visible_fraction)] for every detected eye of every frame."""
    from iris_pipeline.encode import encode
    from iris_pipeline.segment import extract_irises
    obs = []
    for i, rgb in enumerate(frames):
        for seg in extract_irises(rgb, det):
            if seg.side not in SIDES:
                continue
            vis = float(np.mean(seg.strip_mask))
            obs.append((i, seg.side, encode(seg.strip, seg.strip_mask), vis))
    return obs


def consensus(obs):
    """Per side: per-bit majority vote over the observations above the quality floor.
    Returns ({side: IrisCode}, stats). A side with fewer than MIN_OBS good observations has no code."""
    from iris_pipeline.encode import IrisCode
    codes, stats = {}, dict(n_obs={}, n_good={})
    for side in SIDES:
        allo = [o for o in obs if o[1] == side]
        good = [o for o in allo if o[3] >= MIN_VISIBLE]
        stats["n_obs"][side], stats["n_good"][side] = len(allo), len(good)
        if len(good) < MIN_OBS:
            continue
        bits = np.stack([o[2].bits for o in good]).astype(np.float32)
        mask = np.stack([o[2].mask for o in good])
        votes = (bits * mask).sum(0)
        cnt = mask.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = votes / cnt
        codes[side] = IrisCode(frac > 0.5, cnt >= MIN_VOTES)
    return codes, stats


def hd_to(codes, template):
    """Mean over the sides both have of the masked, rotation-searched fractional HD (encode.hamming)."""
    from iris_pipeline.encode import hamming
    ds = []
    for side, c in codes.items():
        if side in template:
            hd = hamming(c, template[side])[0]
            if not np.isnan(hd):
                ds.append(hd)
    return float(np.mean(ds)) if ds else float("nan")


def pack_template(tpl):
    """{side: IrisCode} -> JSON-able dict with base64 bit and mask planes (the published proxy code and its mask)."""
    out = {}
    for side, c in tpl.items():
        out[side] = dict(shape=list(c.bits.shape), bits=base64.b64encode(np.packbits(c.bits.ravel())).decode(),
                         mask=base64.b64encode(np.packbits(c.mask.ravel())).decode())
    return out


def unpack_template(d):
    from iris_pipeline.encode import IrisCode
    out = {}
    for side, v in d.items():
        n = int(np.prod(v["shape"]))
        bits = np.unpackbits(np.frombuffer(base64.b64decode(v["bits"]), np.uint8))[:n].reshape(v["shape"]).astype(bool)
        mask = np.unpackbits(np.frombuffer(base64.b64decode(v["mask"]), np.uint8))[:n].reshape(v["shape"]).astype(bool)
        out[side] = IrisCode(bits, mask)
    return out
