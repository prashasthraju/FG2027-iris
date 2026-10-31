"""Frame-embedding worker for generators that run in another Python env.

PersonaLive needs its own env (py3.10, mediapipe 0.10, numpy 1.26), while the iris pipeline
needs the conda env (mediapipe 1.0, numpy 2). The generator's frame hook sends each frame
here the moment it is generated and waits for the embedded frame, so embedding stays inline
and causal.

Protocol on stdin/stdout, one frame at a time:
    request:  int32 h, int32 w, then h*w*3 bytes RGB uint8
    reply:    int32 H, int32 W, then H*W*3 bytes RGB uint8 (the frame with the pool iris, upscaled
              when the iris is small: cancelable.inline.FrameEmbedder), then one JSON log line on stderr
A request with h = 0 ends the worker.

    python tools/embed_worker.py --entry 034
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--sharp", action="store_true")
    ap.add_argument("--target-iris-px", type=float, default=150)
    a = ap.parse_args()
    from cancelable.inline import FrameEmbedder
    emb = FrameEmbedder(a.entry, a.sharp, target_iris_px=a.target_iris_px)
    rd, wr = sys.stdin.buffer, sys.stdout.buffer
    while True:
        head = rd.read(8)
        if len(head) < 8:
            break
        h, w = struct.unpack("<ii", head)
        if h == 0:
            break
        buf = rd.read(h * w * 3)
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        out = emb(frame)
        wr.write(struct.pack("<ii", *out.shape[:2]) + np.ascontiguousarray(out, np.uint8).tobytes())
        wr.flush()
        sys.stderr.write(json.dumps(emb.log[-1]) + "\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
