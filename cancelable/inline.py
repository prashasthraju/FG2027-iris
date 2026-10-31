"""Model-agnostic inline embedding: one frame in, the same frame with the pool iris out.

Everything that used to differ per reenactor is now measured on the frames themselves:
  - upscale:   chosen once per clip so the iris reaches `target_iris_px` (LivePortrait output
               already has ~170 px -> 1x; PersonaLive's 512-px frames -> ~4x);
  - detection: if no face is found, retry on a padded copy (close-up framing, CASIA);
  - matte:     soft-render mode switches on per eye from the target's own sharpness.
What remains per model is only *where* the frame is intercepted (an adapter, a few lines):

  LivePortrait   live_portrait_pipeline.paste_back      (in-process)        tools/reenact_video.py
  PersonaLive    Pose2VideoPipeline_Stream.decode_latents (other env -> worker) tools/personalive_lowmem.py
  any generator  frames on a pipe / in a stream          (generic)           tools/embed_stream.py

Every adapter calls an embedder with the same interface: `frame_out = embedder(frame_in)`.
"""
import os
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONDA_PY = Path(os.environ.get("EMBED_WORKER_PYTHON", sys.executable))


class FrameEmbedder:
    """In-process embedder. Causal: each call sees only the frame it is given."""

    def __init__(self, entry, sharp=False, target_iris_px=150, max_upscale=6.0, pad=0.25, preview=False, color=None):
        from cancelable.pool import IrisPool
        from iris_pipeline.detect import FaceEyeDetector
        self.entry, self.sharp, self.preview = entry, sharp, preview
        self.tex = IrisPool().textures(entry)
        self.det = FaceEyeDetector()
        self.target_iris_px, self.max_upscale, self.pad = target_iris_px, max_upscale, pad
        self.scale = None          # fixed on the first frame with a detectable eye, then kept for the clip
        self.color = None          # colour mode fixed once per clip (see _choose_color)
        # optional override of the colour mode for COLOUR clips (donor | target); grey clips keep the automatic 'nir'. Default (None, env unset) = unchanged behaviour.
        self.color_override = color or os.environ.get("IRIS_COLOR_MODE") or None
        assert self.color_override in (None, "donor", "target"), self.color_override
        self.log = []

    def _segs(self, frame):
        """Eye segmentations, retrying on a padded copy when the face touches the frame edge."""
        from iris_pipeline.segment import extract_irises
        segs = extract_irises(frame, self.det)
        if segs or not self.pad:
            return segs, 0, 0
        h, w = frame.shape[:2]
        py, px = int(h * self.pad), int(w * self.pad)
        padded = cv2.copyMakeBorder(frame, py, py, px, px, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return extract_irises(padded, self.det), py, px

    def _choose_scale(self, frame):
        from iris_pipeline.normalize import mean_radius
        segs, _, _ = self._segs(frame)
        if not segs:
            return None
        d = max(2 * mean_radius(s.iris) for s in segs)
        return float(np.clip(self.target_iris_px / d, 1.0, self.max_upscale))

    @staticmethod
    def _choose_color(frame, tol=6.0):
        """'nir' for a (near-)grayscale clip, else 'donor'. Decided once per clip: deciding per frame
        let a generator's faint colour cast flip single frames to the donor's colour (a brown iris on
        ~5 PersonaLive frames of a grey NIR face), and the per-frame test tolerates only 2 grey levels."""
        s = frame[:: max(1, frame.shape[0] // 64), :: max(1, frame.shape[1] // 64)].astype(np.int16)
        d = max(np.abs(s[..., 0] - s[..., 1]).mean(), np.abs(s[..., 1] - s[..., 2]).mean())
        return "nir" if d < tol else "donor"

    def __call__(self, frame):
        from iris_pipeline.embed import embed_into_eye
        t0 = time.perf_counter()
        if self.color is None:
            self.color = self._choose_color(frame)
            if self.color == "donor" and self.color_override:
                self.color = self.color_override
        if self.scale is None:
            self.scale = self._choose_scale(frame)
        k = self.scale or 1.0
        big = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1 else frame
        segs, py, px = self._segs(big)
        work = big if not py else cv2.copyMakeBorder(big, py, py, px, px, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        out, done, gated = work, 0, 0
        for seg in segs:
            if seg.side in self.tex:
                g, _, _ = embed_into_eye(out, seg, self.tex[seg.side], color=self.color, method="layered",
                                         gate=True, blur_match=not self.sharp)
                done, gated = done + (g is not out), gated + (g is out)
                out = g
        if py:
            out = out[py:py + big.shape[0], px:px + big.shape[1]]
        self.log.append(dict(frame=len(self.log), scale=round(k, 3), embedded_eyes=done, gated_eyes=gated,
                             ms=round(1000 * (time.perf_counter() - t0), 1)))
        self.last_raw, self.last_segs = big, segs
        return out


class WorkerEmbedder:
    """Same interface, for generators running in another Python env: frames go to
    tools/embed_worker.py (conda env) and come back embedded before generation continues."""

    def __init__(self, entry, sharp=False, python=CONDA_PY, **kw):
        args = [str(python), str(ROOT / "tools" / "embed_worker.py"), "--entry", entry]
        args += ["--sharp"] if sharp else []
        for key, v in kw.items():
            args += [f"--{key.replace('_', '-')}", str(v)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.log = []

    def __call__(self, frame):
        h, w = frame.shape[:2]
        self.proc.stdin.write(struct.pack("<ii", h, w) + np.ascontiguousarray(frame, np.uint8).tobytes())
        self.proc.stdin.flush()
        oh, ow = struct.unpack("<ii", self.proc.stdout.read(8))
        out = np.frombuffer(self.proc.stdout.read(oh * ow * 3), np.uint8).reshape(oh, ow, 3)
        while True:                                     # skip MediaPipe's own log lines
            line = self.proc.stderr.readline().decode(errors="replace")
            if not line or line.startswith('{"frame"'):
                break
        if line:
            self.log.append(json.loads(line))
        return out

    def close(self):
        self.proc.stdin.write(struct.pack("<ii", 0, 0))
        self.proc.stdin.close()
        self.proc.wait()


def make_embedder(entry, same_env=True, **kw):
    """FrameEmbedder when the generator shares our env, else a WorkerEmbedder bridge."""
    return FrameEmbedder(entry, **kw) if same_env else WorkerEmbedder(entry, **kw)
