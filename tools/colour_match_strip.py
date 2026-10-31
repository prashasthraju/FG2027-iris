"""LOCAL ONLY (licensed faces): before/after eye strip, rows = first 3 sessions of the pre-registered order, columns = raw twin | donor colour | target colour.
    CUDA_VISIBLE_DEVICES='' python tools/colour_match_strip.py   # -> outputs/colour_match/strip_LOCAL_ONLY.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "colour_match"
FRAME = 20


def frame(p, k):
    c = cv2.VideoCapture(str(ROOT / p))
    c.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = c.read()
    c.release()
    return f


rows = []
for sid in json.loads((OUT / "subset.json").read_text())["order"][:3]:
    m = json.loads((OUT / "sessions" / f"{sid}.json").read_text())
    for k in (20, 19, 21, 18, 22, 16, 24, 12, 28):          # first frame where the hook changed both the donor-colour and the target-colour clip
        raw, don, tgt = frame(m["raw_twin_new"], k), frame(m["orig_video"], k), frame(m["video"], k)
        ys, xs = np.where(np.abs(don.astype(int) - frame(m["orig_raw_twin"], k).astype(int)).max(-1) > 0)
        if len(ys) and (np.abs(tgt.astype(int) - raw.astype(int)).max(-1) > 0).any():
            break
    y0, y1, x0, x1 = ys.min() - 14, ys.max() + 14, xs.min() - 14, xs.max() + 14
    tiles = []
    for name, f in (("raw twin (no hook)", raw), ("donor colour (current)", don), ("target colour (colour-matched)", tgt)):
        t = cv2.resize(f[y0:y1, x0:x1], (700, int(700 * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(t, (0, 0), (700, 26), (0, 0, 0), -1)
        cv2.putText(t, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(str(OUT / "strip_LOCAL_ONLY.png"), np.vstack(rows))
print("wrote", OUT / "strip_LOCAL_ONLY.png")
