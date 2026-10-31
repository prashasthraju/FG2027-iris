"""Generic adapter: embed a pool iris into frames *as they arrive* from any generator.

For a reenactor we have not hooked, pipe its frames through this filter while it runs. Each
frame is embedded the moment it is read, before the next one arrives (causal, per frame),
and written straight to the output, so nothing waits for the full clip:

    <generator writing a video to stdout> | ffmpeg -loglevel error -i pipe: -f rawvideo -pix_fmt rgb24 - \\
      | python tools/embed_stream.py --size 512x512 --fps 25 --entry 034 -o outputs/stream

or read a video file that is still being written / already finished:

    python tools/embed_stream.py --input clip.mp4 --entry 034 -o outputs/stream

Writes <name>__stream_<entry>.mp4 (lossless) plus the matching raw frames at the same scale,
and a per-frame JSON log. Upscaling, padding and the soft-render matte are chosen from the
frames themselves (cancelable.inline.FrameEmbedder), so no per-model settings are needed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class LosslessWriter:
    def __init__(self, path, fps):
        self.path, self.fps, self.p = path, fps, None

    def write(self, frame):
        if self.p is None:
            h, w = frame.shape[:2]
            self.size = (w - w % 2, h - h % 2)
            self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(self.fps), "-i", "-",
                                       "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                                       str(self.path)], stdin=subprocess.PIPE)
        self.p.stdin.write(np.ascontiguousarray(frame[:self.size[1], :self.size[0]]).tobytes())

    def close(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()


def frames_from(a):
    if a.input:
        cap = cv2.VideoCapture(a.input)
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    w, h = map(int, a.size.split("x"))
    rd = sys.stdin.buffer
    while True:
        buf = rd.read(w * h * 3)
        if len(buf) < w * h * 3:
            return
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--input", default="", help="video file instead of rawvideo on stdin")
    ap.add_argument("--size", default="", help="WxH of rawvideo frames on stdin")
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--name", default="stream")
    ap.add_argument("--sharp", action="store_true")
    ap.add_argument("--target-iris-px", type=float, default=150)
    ap.add_argument("-o", "--out", default="outputs/stream")
    a = ap.parse_args()
    from cancelable.inline import FrameEmbedder
    fps = a.fps or (cv2.VideoCapture(a.input).get(cv2.CAP_PROP_FPS) if a.input else 25) or 25
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.input).stem if a.input else a.name
    emb = FrameEmbedder(a.entry, a.sharp, target_iris_px=a.target_iris_px)
    w_emb = LosslessWriter(out / f"{name}__stream_{a.entry}.mp4", fps)
    w_raw = LosslessWriter(out / f"{name}__stream_raw.mp4", fps)
    for frame in frames_from(a):
        result = emb(frame)                       # embedded the moment the frame arrives
        w_emb.write(result)
        w_raw.write(emb.last_raw)
    w_emb.close()
    w_raw.close()
    log = emb.log
    (out / f"{name}__stream_{a.entry}.json").write_text(json.dumps(dict(
        entry=a.entry, frames=len(log), scale=log[0]["scale"] if log else None,
        embed_ms_median=float(np.median([r["ms"] for r in log])) if log else None,
        embedded_eyes=sum(r["embedded_eyes"] for r in log), gated_eyes=sum(r["gated_eyes"] for r in log),
        per_frame=log), indent=1))
    print(f"wrote {w_emb.path} and {w_raw.path}: {len(log)} frames, scale {log[0]['scale'] if log else '-'}, "
          f"{sum(r['embedded_eyes'] for r in log)} eyes embedded, {sum(r['gated_eyes'] for r in log)} gated")


if __name__ == "__main__":
    main()
