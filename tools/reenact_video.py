"""Cross-reenact a face with a locally installed model and write the result losslessly.

    third_party/LivePortrait/.venv_lp/bin/python tools/reenact_video.py liveportrait --source FACE.jpg
    python tools/reenact_video.py personalive --source FACE.jpg

The source is the person's *real* face; the cancelable iris is applied afterwards to
the model's output frames (option O1, research/06), because no installed reenactor
carries an embedded iris. Output: an x264 `-qp 0` mp4 (lossless luma), so the
later embed + verification is not confounded by compression.

liveportrait  CASIA faces are padded for its detector and the output keeps the source
              resolution (liveportrait_eval.LivePortrait); default --scale 0.5 halves the
              2352x1728 frame so the output iris is ~86 px (>= the 60 px gate).
--embed-entry SID  (liveportrait, D27) embeds that pool entry INSIDE the generation loop:
              each frame gets the pool iris the moment LivePortrait generates it (per-frame
              re-detection, layered compositor, D18 gate), before the next frame exists.
              The source face is never edited. Writes the raw and the embedded clip plus a
              per-frame log (gated eyes, embed time).

personalive   needs a separate PersonaLive environment (PERSONALIVE_PYTHON) and its weights; the source is a
              square face crop resized to 512 (iris ~50 px). Its own writer is CRF 18 (not
              lossless); frames are upscaled x2 (output iris ~100 px) before embedding.
"""
import os
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LP = ROOT / "third_party" / "LivePortrait"
PL = Path(os.environ.get("PERSONALIVE_DIR", "external/PersonaLive"))
PL_PY = Path(os.environ.get("PERSONALIVE_PYTHON", "python"))


def write_lossless(frames, path, fps):
    """RGB uint8 frames -> x264 qp 0 mp4 (even dimensions)."""
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w2}x{h2}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-qp", "0",
                          "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    p.stdin.close()
    p.wait()
    return path


def read_frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out, cap.get(cv2.CAP_PROP_FPS) or 25


def driving_clip(n, out, start=0, driver="d6"):
    """n frames (from `start`) of one of LivePortrait's example driving clips (default d6, the tuning clip)."""
    path = Path(driver) if ("/" in str(driver) or str(driver).endswith(".mp4")) else LP / "assets" / "examples" / "driving" / f"{driver}.mp4"
    frames, fps = read_frames(path)
    clip = frames[start:start + n]
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, clip[0].shape[1::-1])
    for f in clip:
        w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    w.release()
    return out, fps


sys.path.insert(0, str(ROOT))
from cancelable.inline import FrameEmbedder  # noqa: E402  (model-agnostic embedder, D31)


class InlineEmbedder(FrameEmbedder):
    """FrameEmbedder plus an optional live preview window (demo)."""

    def __init__(self, sid, sharp=False, preview=False):
        # inside LivePortrait's loop the frame size must not change: no upscaling here
        super().__init__(sid, sharp, target_iris_px=150, max_upscale=1.0, pad=0.0)
        self.preview, self.sid = preview, sid

    def __call__(self, frame):
        out = super().__call__(frame)
        if self.preview:
            self.show(self.last_raw, out, self.last_segs)
        return out

    def show(self, raw, emb, segs):
        """Live window: full frame as generated | with pool iris, and an eye zoom row."""
        from iris_pipeline.normalize import mean_radius

        def lab(img, text, sub=""):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 46 if sub else 26), (0, 0, 0), -1)
            cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if sub:
                cv2.putText(img, sub, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 230, 120), 1, cv2.LINE_AA)
            return img
        W = 560
        k = W / raw.shape[1]
        small = lambda x: cv2.resize(x, (W, int(raw.shape[0] * k)), interpolation=cv2.INTER_AREA)
        last = self.log[-1]
        top = np.hstack([lab(small(raw), f"frame {last['frame'] + 1}: as generated"),
                         lab(small(emb), f"+ pool iris {self.sid} (inline)",
                             f"{last['embedded_eyes']} eye(s) embedded, {last['gated_eyes']} gated, {last['ms']:.0f} ms")])
        eye = next((s for s in segs if s.side == "image_left"), None)
        if eye is not None:
            cx, cy, R = int(eye.iris[0]), int(eye.iris[1]), int(2.6 * mean_radius(eye.iris))
            crop = lambda x: cv2.resize(x[max(cy - R, 0):cy + R, max(cx - R, 0):cx + R], (W, W // 2),
                                        interpolation=cv2.INTER_AREA)
            top = np.vstack([top, np.hstack([lab(crop(raw), "eye zoom"), lab(crop(emb), "eye zoom")])])
        cv2.imshow("iris-gen live demo: pool iris embedded as each frame is generated",
                   cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


def run_liveportrait(src_rgb, a, tmp):
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT))
    from liveportrait_eval import LivePortrait
    drv, fps = driving_clip(a.frames, tmp / "drive.mp4", a.driver_start, a.driver)
    lp = LivePortrait()
    if a.embed_entry:
        lp.frame_hook = InlineEmbedder(a.embed_entry, a.sharp, preview=a.preview)
    frames, _, _ = lp(src_rgb, drv, a.lp_crop)
    if a.embed_entry:
        return frames, fps, lp.raw, lp.frame_hook.log
    return frames, fps, None, None


def run_personalive(src_rgb, a, tmp):
    if not PL_PY.exists():
        raise SystemExit(f"PersonaLive env missing: {PL_PY}")
    h, w = src_rgb.shape[:2]
    if a.embed_entry:
        raise SystemExit("--embed-entry is implemented for liveportrait only (PersonaLive does not fit on 8 GB, D26)")
    s = min(h, w)
    sq = src_rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    ref = tmp / "ref.png"
    cv2.imwrite(str(ref), cv2.cvtColor(cv2.resize(sq, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    n = a.frames - a.frames % 4                       # -L must be a multiple of 4
    drv, fps = driving_clip(n, tmp / "drive.mp4", a.driver_start, a.driver)
    # -s: ignore user site-packages, which can shadow the environment's libraries
    subprocess.run([str(PL_PY), "-s", "inference_offline.py", "--reference_image", str(ref),
                    "--driving_video", str(drv), "-W", "512", "-H", "512", "-L", str(n)], cwd=PL, check=True)
    outs = sorted((PL / "results").glob("*personalive_offline*/split_vid/*.mp4"), key=lambda p: p.stat().st_mtime)
    frames, fps = read_frames(outs[-1])
    return [cv2.resize(f, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for f in frames], fps, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["liveportrait", "personalive"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--driver", default="d6", help="LivePortrait example driving clip (d0..d20); d6 is the tuning clip")
    ap.add_argument("--driver-start", type=int, default=0, help="first driving frame")
    ap.add_argument("--scale", type=float, default=0.5, help="source resize before reenactment")
    ap.add_argument("--lp-crop", type=float, default=2.3, help="LivePortrait crop scale (default = its own)")
    ap.add_argument("--pad", type=float, default=0.25,
                    help="grey border added to every output frame: CASIA close-ups put the face at the frame "
                         "edge, where MediaPipe loses it once the head moves (13/30 -> 30/30 frames)")
    ap.add_argument("--embed-entry", default="", help="liveportrait: embed this pool entry inline, frame by frame")
    ap.add_argument("--sharp", action="store_true", help="with --embed-entry: no blur matching")
    ap.add_argument("--preview", action="store_true", help="with --embed-entry: live window while generating")
    ap.add_argument("-o", "--out", default="outputs/reenact")
    a = ap.parse_args()
    src = cv2.cvtColor(cv2.imread(a.source), cv2.COLOR_BGR2RGB)
    if a.scale != 1:
        src = cv2.resize(src, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reenact_") as tmp:
        tmp = Path(tmp)
        frames, fps, raw, log = (run_liveportrait if a.model == "liveportrait" else run_personalive)(src, a, tmp)

    def pad(fs):
        if not a.pad:
            return fs
        h, w = fs[0].shape[:2]
        py, px = int(h * a.pad), int(w * a.pad)
        return [cv2.copyMakeBorder(f, py, py, px, px, cv2.BORDER_CONSTANT, value=(128, 128, 128)) for f in fs]
    stem = f"{a.model}__{Path(a.source).stem}"
    if raw is None:
        dst = write_lossless(pad(frames), out / f"{stem}.mp4", fps)
        print(f"wrote {dst}  ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, {fps:.2f} fps, lossless)")
        return
    dst = write_lossless(pad(frames), out / f"{stem}__inline_{a.embed_entry}.mp4", fps)
    raw_dst = write_lossless(pad(raw), out / f"{stem}__inline_raw.mp4", fps)
    import json
    ms = [r["ms"] for r in log]
    (out / f"{stem}__inline_{a.embed_entry}.json").write_text(json.dumps(dict(
        entry=a.embed_entry, sharp=a.sharp, frames=len(log), embed_ms_median=float(np.median(ms)),
        embed_ms_p95=float(np.percentile(ms, 95)), gated_eyes=sum(r["gated_eyes"] for r in log),
        embedded_eyes=sum(r["embedded_eyes"] for r in log), per_frame=log), indent=1))
    print(f"wrote {dst} (embedded inline, frame by frame) and {raw_dst} (same frames, un-embedded); "
          f"embed {np.median(ms):.0f} ms/frame median, {sum(r['embedded_eyes'] for r in log)} eyes embedded, "
          f"{sum(r['gated_eyes'] for r in log)} gated")


if __name__ == "__main__":
    main()
