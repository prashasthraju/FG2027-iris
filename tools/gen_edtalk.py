"""EDTalk-V cross-reenactment with the pool iris embedded inline (generator #2 after LivePortrait).

    third_party/EDTalk/.venv_edtalk/bin/python tools/gen_edtalk.py \\
        --source dbs/ravdess_video/Actor_01/source_01-01-01-01-01-01-01_f024.png \\
        --driver dbs/ravdess_video/Actor_02/driver_01-01-02-01-02-01-02.mp4 \\
        --frames 20 --scale 2.5 --embed-entry 034 -o outputs/edtalk

What it does, per frame (causal, nothing waits for the clip):
  1. EDTalk-V (video-driven; the SAME driving video feeds lip, pose and expression = plain
     cross-reenactment) renders a 256x256 face crop from the source crop.
  2. The 256 face is pasted back onto the SOURCE PHOTO at `--scale` x its size (bicubic warp with the
     inverse of the crop transform and a feathered face-hull matte; hair/background stay the source's).
     EDTalk itself only ever produces 256 px, so this canvas is plain resampling, never super-resolution.
  3. The canvas frame goes to cancelable.inline.WorkerEmbedder (the project's embedder, run in the
     conda env through tools/embed_worker.py): the pool iris is written into the frame the moment it exists.
Outputs (lossless x264 qp 0), names mirror tools/reenact_video.py:
  edtalk__<stem>__inline_<entry>.mp4   embedded video
  edtalk__<stem>__inline_raw.mp4       the same frames without the iris (raw twin)
  edtalk__<stem>__inline_<entry>.json  embedded_eyes / gated_eyes / embed_ms_median ... + gen_ms, VRAM, iris px

The canvas is NOT upscaled further by the embedder (target_iris_px=1 -> factor 1): choose --scale so that the
iris on the canvas is >= 60 px (the hook's resolution gate). The run reports the measured iris diameter and
warns when it is below the gate. Without --embed-entry only the raw video is written.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ED = ROOT / "third_party" / "EDTalk"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def landmarks(img_bgr, detector, predictor):
    """dlib 68 landmarks of the (single, else largest) face; None if there is none."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    r = max(rects, key=lambda r: r.width() * r.height())
    sh = predictor(gray, r)
    return np.array([(sh.part(i).x, sh.part(i).y) for i in range(68)], np.float64)


def crop_matrix(shape68):
    """2x3 affine mapping EDTalk 256-crop coordinates -> source-image coordinates
    (the similarity fit of data_preprocess/crop_image2.py: first 47 landmarks onto M003_template)."""
    from skimage import transform as tf
    tpl = np.load(ED / "data_preprocess" / "M003_template.npy").astype(np.float64)
    t = tf.SimilarityTransform()
    t.estimate(tpl[:47], shape68[:47])
    return t.params[:2].copy(), tpl


def crop_to_256(img_bgr, M):
    """Source -> 256 crop (area-aware: linear warp after the inverse matrix; BORDER_CONSTANT 0 like the official script)."""
    return cv2.warpAffine(img_bgr, M, (256, 256), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def face_matte(tpl, dilate=14, blur=21):
    """Feathered face-hull matte in 256-crop coordinates (landmarks 0-67 of the template)."""
    m = np.zeros((256, 256), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(tpl.astype(np.int32)), 255)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    m = cv2.GaussianBlur(m, (0, 0), blur / 3)
    return m.astype(np.float32) / 255.0


def to_tensor(rgb_u8, dev):
    import torch
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ((t - 0.5) * 2.0).to(dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source photo (identity)")
    ap.add_argument("--driver", required=True, help="driving video (lip + pose + expression)")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--driver-start", type=int, default=0, help="first driving frame (after 25 fps resampling)")
    ap.add_argument("--scale", type=float, default=2.5,
                    help="canvas = source photo x scale (bicubic). 2.5 gives a ~60+ px iris on 512 px RAVDESS faces")
    ap.add_argument("--embed-entry", default="", help="pool entry id (cancelable.pool.IrisPool().ids) to embed inline")
    ap.add_argument("--blur-match", action="store_true",
                    help="with --embed-entry: let the embedder blur the pool iris to the target's sharpness. OFF by default: "
                         "EDTalk's eyes hold ~12 px of real detail, and blur matching erased the identity (0/2 entries identified; "
                         "sharp: correct, ratio 0.87), docs/18")
    ap.add_argument("--fp16", action="store_true", help="run EDTalk in autocast fp16")
    ap.add_argument("--model-path", default=str(ED / "ckpts" / "EDTalk.pt"))
    ap.add_argument("--pad", type=float, default=0.0, help="grey border fraction around every output frame")
    ap.add_argument("-o", "--out", default="outputs/edtalk")
    a = ap.parse_args()

    import torch
    import dlib
    sys.path.insert(0, str(ED))
    from networks.generator import Generator
    from embed_stream import LosslessWriter

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(ED / "data_preprocess" / "shape_predictor_68_face_landmarks.dat"))

    src = cv2.imread(a.source)
    if src is None:
        raise SystemExit(f"cannot read {a.source}")
    sh = landmarks(src, detector, predictor)
    if sh is None:
        raise SystemExit("dlib found no face in the source photo")
    Ms, tpl = crop_matrix(sh)                                   # crop256 -> source
    src_crop = cv2.cvtColor(crop_to_256(src, Ms), cv2.COLOR_BGR2RGB)
    canvas0 = cv2.resize(src, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC) if a.scale != 1 else src
    S = np.diag([a.scale, a.scale, 1.0])
    Mc = (S @ np.vstack([Ms, [0, 0, 1]]))[:2]                   # crop256 -> canvas
    canvas0 = cv2.cvtColor(canvas0, cv2.COLOR_BGR2RGB).astype(np.float32)
    matte = face_matte(tpl)
    # never paste generated pixels where the crop lies outside the source photo (the crop is black there)
    valid = crop_to_256(np.full(src.shape[:2], 255, np.uint8), Ms)
    valid = cv2.GaussianBlur(cv2.erode(valid, np.ones((9, 9), np.uint8)), (0, 0), 3).astype(np.float32) / 255.0
    matte = matte * valid
    up = float(np.sqrt(abs(np.linalg.det(Mc[:, :2]))))          # canvas pixels per EDTalk output pixel
    H, W = canvas0.shape[:2]
    big_matte = cv2.warpAffine(matte, Mc, (W, H), flags=cv2.INTER_LINEAR)[..., None]

    with tempfile.TemporaryDirectory(prefix="edtalk_") as tmp:
        tmp = Path(tmp)
        d25 = tmp / "driver25.mp4"                              # EDTalk wants 25 fps driving video
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.driver, "-r", "25", "-an",
                        "-vf", "scale=iw:ih", str(d25)], check=True)
        cap = cv2.VideoCapture(str(d25))
        drv = []
        while len(drv) < a.driver_start + a.frames:
            ok, f = cap.read()
            if not ok:
                break
            drv.append(f)
        drv = drv[a.driver_start:]
        if not drv:
            raise SystemExit("no driving frames")
    dh = landmarks(drv[0], detector, predictor)
    if dh is None:
        raise SystemExit("dlib found no face in the first driving frame")
    Md, _ = crop_matrix(dh)                                     # one fixed crop for the whole clip (as crop_image_tem)

    model = Generator(256, 512, 20, 6, 10, 1).to(dev)
    model.load_state_dict(torch.load(a.model_path, map_location="cpu")["gen"])
    model.eval()

    emb = None
    if a.embed_entry:
        from cancelable.inline import WorkerEmbedder
        emb = WorkerEmbedder(a.embed_entry, sharp=not a.blur_match, target_iris_px=1)   # factor 1: canvas size is kept

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"edtalk__{Path(a.source).stem}"
    pad = (lambda f: f) if not a.pad else (lambda f: cv2.copyMakeBorder(
        f, int(f.shape[0] * a.pad), int(f.shape[0] * a.pad), int(f.shape[1] * a.pad), int(f.shape[1] * a.pad),
        cv2.BORDER_CONSTANT, value=(128, 128, 128)))
    w_raw = LosslessWriter(out / f"{stem}__inline_raw.mp4" if emb else out / f"{stem}.mp4", 25)
    w_emb = LosslessWriter(out / f"{stem}__inline_{a.embed_entry}.mp4", 25) if emb else None

    img_src = to_tensor(src_crop, dev)
    gen_ms, log = [], []
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=a.fp16 and dev == "cuda"):
        for i, fr in enumerate(drv):
            t0 = time.perf_counter()
            d = to_tensor(cv2.cvtColor(crop_to_256(fr, Md), cv2.COLOR_BGR2RGB), dev)
            rec = model.test_EDTalk_V(img_src, d, d, d, None)    # same video drives lip, pose and expression
            rec = ((rec.float().clamp(-1, 1) + 1) * 127.5)[0].permute(1, 2, 0).cpu().numpy()
            if dev == "cuda":
                torch.cuda.synchronize()
            gen_ms.append(1000 * (time.perf_counter() - t0))
            gface = cv2.warpAffine(rec, Mc, (W, H), flags=cv2.INTER_CUBIC)       # 256 face on the canvas
            canvas = np.clip(canvas0 * (1 - big_matte) + gface * big_matte, 0, 255).astype(np.uint8)
            w_raw.write(pad(canvas))
            if emb is not None:
                res = emb(canvas)
                if res.shape != canvas.shape:
                    raise SystemExit(f"embedder changed the frame size {canvas.shape} -> {res.shape}; "
                                     "this script keeps the canvas (raw twin must match)")
                w_emb.write(pad(res))
            print(f"\rframe {i + 1}/{len(drv)}", end="", flush=True)
    print()
    w_raw.close()
    if emb is not None:
        w_emb.close()
        emb.close()
        log = emb.log
    vram = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0.0
    info = dict(model="EDTalk-V", source=a.source, driver=a.driver, frames=len(drv), scale=a.scale,
                canvas=[W, H], native_output=[256, 256], resample_factor=round(up, 3), fp16=a.fp16, gen_ms_median=float(np.median(gen_ms)),
                torch_peak_vram_mib=round(vram, 1))
    if emb is not None:
        ms = [r["ms"] for r in log]
        info.update(entry=a.embed_entry, sharp=not a.blur_match, embed_ms_median=float(np.median(ms)),
                    embed_ms_p95=float(np.percentile(ms, 95)), gated_eyes=sum(r["gated_eyes"] for r in log),
                    embedded_eyes=sum(r["embedded_eyes"] for r in log), per_frame=log)
        (out / f"{stem}__inline_{a.embed_entry}.json").write_text(json.dumps(info, indent=1))
        print(f"wrote {w_emb.path} and {w_raw.path}: {len(log)} frames, canvas {W}x{H}, "
              f"EDTalk {np.median(gen_ms):.0f} ms/frame, torch peak {vram:.0f} MiB, embed {np.median(ms):.0f} ms/frame, "
              f"{info['embedded_eyes']} eyes embedded, {info['gated_eyes']} gated")
    else:
        (out / f"{stem}.json").write_text(json.dumps(info, indent=1))
        print(f"wrote {w_raw.path}: {len(drv)} frames, canvas {W}x{H}, EDTalk {np.median(gen_ms):.0f} ms/frame, "
              f"torch peak {vram:.0f} MiB")


if __name__ == "__main__":
    main()
