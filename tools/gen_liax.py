"""LIA-X cross-reenactment with the pool iris embedded INSIDE the generation loop (D31 adapter).

Run with the LIA-X venv (it layers on the conda env, so cancelable/ and iris_pipeline/ import in-process):

    third_party/LIA-X/.venv/bin/python tools/gen_liax.py \
        --source dbs/ravdess_video/Actor_01/source_01-01-01-01-01-01-01_f024.png \
        --driver dbs/ravdess_video/Actor_02/driver_01-01-02-01-02-01-02.mp4 \
        --frames 20 --scale 3 --embed-entry 034 -o outputs/liax

LIA-X (https://github.com/AGI-Research-Lab/LIA-X, code CC BY-NC 4.0) maps a 512x512 face crop and a 512x512 driving
face to a 512x512 frame; the architecture is fixed at 512 px. Its output is therefore upscaled by --scale (bicubic,
the same thing LivePortrait's paste-back onto a large canvas amounts to) and handed to cancelable.inline.FrameEmbedder,
which re-detects the eyes on that frame and writes the pool iris (layered compositor, D18 gate) before the next frame
is generated. Nothing here composites an iris itself.

Outputs (same conventions as tools/reenact_video.py liveportrait --embed-entry), lossless x264 qp 0:
    liax__<source stem>__inline_<entry>.mp4   embedded
    liax__<source stem>__inline_raw.mp4       raw twin: the same upscaled frames without the iris
    liax__<source stem>__inline_<entry>.json  entry, frames, embed_ms_median/p95, gated_eyes, embedded_eyes,
                                               per_frame, plus gen_ms_median, vram and resolution fields

GPU: the model is 910 M parameters; fp16 weights take 1.8 GB and the run peaks at ~2.65 GB (measured). The script refuses to
start on the GPU when less than --min-free-gb is free (the GPU is shared) and caps its own allocation (--vram-cap-gb).
--cpu runs fp32 on the CPU (~16 s/frame).

--crop auto: sources/drivers that are not square are cropped to a square face box with LIA-X's own detector
(face_alignment, run on the CPU) the way its utils/crop.py does; square inputs (RAVDESS faces) are used as they are.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIAX = ROOT / "third_party" / "LIA-X"
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(LIAX / "torch_extension"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LIAX))

SIZE = 512  # LIA-X's fixed resolution


def free_vram_gb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return float(out.decode().split()[0]) / 1024
    except Exception:
        return float("nan")


def face_box(frame_rgb, fa, increase_area=0.25, increase_top=0.5):
    """LIA-X utils/crop.py: S3FD box, grown, as an (x, y, side) square. None if no face."""
    scale = max(frame_rgb.shape[:2]) / 640.0
    small = cv2.resize(frame_rgb, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA) if scale > 1 else frame_rgb
    boxes = fa.face_detector.detect_from_image(small[..., ::-1].copy())
    if len(boxes) == 0:
        return None
    l, t, r, b = (np.array(boxes[0][:4]) * max(scale, 1.0)).tolist()
    w, h = r - l, b - t
    wi = max(increase_area, ((1 + 2 * increase_area) * h - w) / (2 * w))
    hi = max(increase_area, ((1 + 2 * increase_area) * w - h) / (2 * h))
    l, t, r, b = l - wi * w, t - hi * h * (1 + increase_top), r + wi * w, b + hi * h
    side = min(b - t, r - l)
    cx, cy = (l + r) / 2, (t + b) / 2
    H, W = frame_rgb.shape[:2]
    x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
    y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
    return x0, y0, int(min(side, W, H))


def to_512(frame_rgb, box):
    if box is not None:
        x, y, s = box
        frame_rgb = frame_rgb[y:y + s, x:x + s]
    return cv2.resize(frame_rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA if frame_rgb.shape[0] > SIZE else cv2.INTER_CUBIC)


def read_driver(path, n, start):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    i = 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i >= start:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    return frames, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source face (the owner's real photo; never edited)")
    ap.add_argument("--driver", required=True, help="driving video")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--driver-start", type=int, default=0)
    ap.add_argument("--scale", type=float, default=3.0,
                    help="upscale of the 512-px LIA-X frame before embedding (RAVDESS iris 26 px -> 78 px at 3; the 60 px gate)")
    ap.add_argument("--embed-entry", default="", help="pool entry id (cancelable.pool.IrisPool().ids); empty = raw clip only")
    ap.add_argument("--sharp", action="store_true", help="with --embed-entry: no blur matching")
    ap.add_argument("--crop", choices=["auto", "none", "always"], default="auto")
    ap.add_argument("--pad", type=float, default=0.0, help="grey border added to every output frame (as reenact_video --pad)")
    ap.add_argument("--cpu", action="store_true", help="fp32 on CPU (~16 s/frame)")
    ap.add_argument("--min-free-gb", type=float, default=3.4, help="refuse the GPU below this much free VRAM")
    ap.add_argument("--vram-cap-gb", type=float, default=2.8, help="hard cap on this process's CUDA allocation")
    ap.add_argument("-o", "--out", default="outputs/liax")
    a = ap.parse_args()

    import torch
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":
        free = free_vram_gb()
        if not free >= a.min_free_gb:
            raise SystemExit(f"only {free:.2f} GB VRAM free (< --min-free-gb {a.min_free_gb}); wait for the other job or use --cpu")
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(a.vram_cap_gb / total, 1.0))
    dt = torch.float16 if dev == "cuda" else torch.float32

    # ---- data
    src = cv2.cvtColor(cv2.imread(a.source), cv2.COLOR_BGR2RGB)
    drv, fps = read_driver(a.driver, a.frames, a.driver_start)
    if not drv:
        raise SystemExit("no driving frames read")
    boxes = [None, None]
    need = [src.shape[0] != src.shape[1], drv[0].shape[0] != drv[0].shape[1]]
    if a.crop == "always" or (a.crop == "auto" and any(need)):
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")
        if a.crop == "always" or need[0]:
            boxes[0] = face_box(src, fa)
        if a.crop == "always" or need[1]:
            boxes[1] = face_box(drv[0], fa)   # box fixed on the first frame, as utils/crop.py
        del fa

    def tens(rgb):
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        return ((x - 0.5) * 2)[None].to(dev).to(dt)

    # ---- model
    from networks.generator import Generator
    t0 = time.perf_counter()
    gen = Generator(size=SIZE, motion_dim=40, scale=2)
    gen.load_state_dict(torch.load(LIAX / "model" / "lia-x.pt", map_location="cpu", weights_only=False))
    gen.eval()
    gen = gen.half().cuda() if dev == "cuda" else gen
    load_s = time.perf_counter() - t0

    emb = None
    if a.embed_entry:
        from cancelable.inline import FrameEmbedder
        # frames are already upscaled by --scale below, so the embedder itself must not rescale (max_upscale=1)
        emb = FrameEmbedder(a.embed_entry, a.sharp, target_iris_px=150, max_upscale=1.0, pad=0.25)

    emb_frames, raw_frames, gen_ms = [], [], []
    with torch.no_grad():
        img_s = tens(to_512(src, boxes[0]))
        z_s2r, feat = gen.enc.enc_2r(img_s)
        alpha_r2s = gen.enc.enc_r2t(z_s2r)
        alpha_start = gen.get_alpha(tens(to_512(drv[0], boxes[1])))
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for f in drv:
            t0 = time.perf_counter()
            alpha = gen.enc.enc_transfer_vid(alpha_r2s, tens(to_512(f, boxes[1])), alpha_start)
            out = gen.dec(z_s2r, alpha, feat)
            frame = ((out.clamp(-1, 1)[0].float().cpu() + 1) / 2 * 255).round().permute(1, 2, 0).numpy().astype(np.uint8)
            gen_ms.append(1000 * (time.perf_counter() - t0))
            if a.scale != 1:
                frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
            if emb is not None:
                emb_frames.append(emb(frame))      # embedded before the next frame is generated
                raw_frames.append(emb.last_raw)
            else:
                raw_frames.append(frame)
    peak = torch.cuda.max_memory_reserved() / 2**30 if dev == "cuda" else None

    # ---- write
    from reenact_video import write_lossless
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pad(fs):
        if not a.pad:
            return fs
        h, w = fs[0].shape[:2]
        py, px = int(h * a.pad), int(w * a.pad)
        return [cv2.copyMakeBorder(f, py, py, px, px, cv2.BORDER_CONSTANT, value=(128, 128, 128)) for f in fs]
    stem = f"liax__{Path(a.source).stem}"
    h, w = raw_frames[0].shape[:2]
    info = dict(model="LIA-X", source=a.source, driver=a.driver, driver_start=a.driver_start, scale=a.scale,
                native_size=SIZE, out_size=[w, h], fps=fps, device=dev, precision="fp16" if dev == "cuda" else "fp32",
                crop_boxes=boxes, gen_ms_median=float(np.median(gen_ms)), gen_ms_p95=float(np.percentile(gen_ms, 95)),
                vram_peak_reserved_gb=peak, model_load_s=load_s)
    if emb is None:
        dst = write_lossless(pad(raw_frames), out_dir / f"{stem}__raw.mp4", fps)
        (out_dir / f"{stem}__raw.json").write_text(json.dumps(info, indent=1))
        print(f"wrote {dst} ({len(raw_frames)} frames, {w}x{h}); gen {info['gen_ms_median']:.0f} ms/frame; peak VRAM {peak}")
        return
    dst = write_lossless(pad(emb_frames), out_dir / f"{stem}__inline_{a.embed_entry}.mp4", fps)
    raw_dst = write_lossless(pad(raw_frames), out_dir / f"{stem}__inline_raw.mp4", fps)
    log = emb.log
    ms = [r["ms"] for r in log]
    (out_dir / f"{stem}__inline_{a.embed_entry}.json").write_text(json.dumps(dict(
        entry=a.embed_entry, sharp=a.sharp, frames=len(log), embed_ms_median=float(np.median(ms)),
        embed_ms_p95=float(np.percentile(ms, 95)), gated_eyes=sum(r["gated_eyes"] for r in log),
        embedded_eyes=sum(r["embedded_eyes"] for r in log), **info, per_frame=log), indent=1))
    print(f"wrote {dst} and {raw_dst}: {len(log)} frames {w}x{h}; eyes embedded {sum(r['embedded_eyes'] for r in log)}, "
          f"gated {sum(r['gated_eyes'] for r in log)}; gen {info['gen_ms_median']:.0f} ms/frame, embed {np.median(ms):.0f} ms/frame; "
          f"peak VRAM {peak if peak is None else round(peak, 2)} GB")


if __name__ == "__main__":
    main()
