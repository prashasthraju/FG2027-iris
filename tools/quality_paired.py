"""Visual cost of the hook, measured on paired clips (paper_needed.tex Table quality).

Every held-out clip has an un-embedded twin made from the same generation (same frames, no iris written): the twin is
"LivePortrait" and the embedded clip is "+ hook". Per frame pair (every k-th frame):
  PSNR (over frames the hook changed; identical frames are counted separately), SSIM (full frame, 2x down-scaled), LPIPS (AlexNet, full frame 2x down-scaled), eye-LPIPS (2.6 x iris radius crop of the
  first eye), AKD (mean MediaPipe landmark displacement, in % of the inter-ocular distance), AED (mean absolute change of the
  52 MediaPipe blendshape scores x 100), and an eye-crop FID between the two frame sets.
Means are over clips (frame means first), with a bootstrap 95 % CI over clips.  CSIM (ArcFace) is tools/csim_paired.py.

    third_party/.venv_q/bin/python tools/quality_paired.py --set nir_v1
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402

SETS = {"nir_v1": ("outputs/heldout", "outputs/heldout_plan.json"),
        "nir_v2": ("outputs/heldout_v2", "outputs/heldout_plan.json"),
        "ravdess": ("outputs/heldout_ravdess", "outputs/heldout_plan_ravdess.json"),
        "idmap": ("outputs/heldout_idmap", "outputs/heldout_plan_idmap.json"),
        "proxy_ravdess": ("outputs/proxy_attr", "outputs/scratch/proxy_quality_ravdess.json"),
        "proxy_nir": ("outputs/proxy_attr", "outputs/scratch/proxy_quality_nir.json")}


def read(path, every):
    cap, out, i = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import lpips
    import mediapipe as mp
    import torch
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.segment import extract_irises
    base, plan = SETS[a.set]
    plan = json.loads((ROOT / plan).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models" / "face_landmarker.task"), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def landmarks(rgb):
        h, w = rgb.shape[:2]
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not r.face_landmarks:
            return None, None
        pts = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
        bs = np.array([c.score for c in r.face_blendshapes[0]]) if r.face_blendshapes else None
        return pts, bs

    def lp(x, y):
        t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
        with torch.no_grad():
            return float(L(t(x), t(y)))

    rows, crops_raw, crops_emb = {}, [], []
    with FaceEyeDetector() as det:
        for r in plan["rows"]:
            if a.limit and len(rows) >= a.limit:
                break
            d = ROOT / base / "clips" / r["id"]
            stem = Path(r["source"]).stem
            emb, raw = d / f"liveportrait__{stem}__inline_{r['entry']}.mp4", d / f"liveportrait__{stem}__inline_raw.mp4"
            if not (emb.exists() and raw.exists()):
                continue
            fr, fe = read(raw, a.every), read(emb, a.every)
            acc = {k: [] for k in ("psnr", "ssim", "lpips", "eye_lpips", "akd_pct_iod", "aed", "identical_frame")}
            for x, y in zip(fr, fe):
                xs, ys = cv2.resize(x, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), cv2.resize(y, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
                pv = psnr(xs, ys, data_range=255)
                acc["identical_frame"].append(1.0 if not np.isfinite(pv) else 0.0)      # frames the hook left untouched (gated)
                if np.isfinite(pv):
                    acc["psnr"].append(pv)
                acc["ssim"].append(ssim(xs, ys, channel_axis=2, data_range=255))
                acc["lpips"].append(lp(xs, ys))
                segs = extract_irises(x, det)
                if segs:
                    s = segs[0]; cx, cy, R = int(s.iris[0]), int(s.iris[1]), int(1.3 * 2 * np.mean(s.iris[2:3]))
                    sl = (slice(max(cy - R, 0), cy + R), slice(max(cx - R, 0), cx + R))
                    cx_, cy_ = [cv2.resize(im[sl], (224, 224), interpolation=cv2.INTER_AREA) for im in (x, y)]
                    acc["eye_lpips"].append(lp(cx_, cy_))
                    crops_raw.append(cx_); crops_emb.append(cy_)
                p1, b1 = landmarks(x); p2, b2 = landmarks(y)
                if p1 is not None and p2 is not None:
                    iod = np.linalg.norm(p1[33] - p1[263]) + 1e-6
                    acc["akd_pct_iod"].append(100 * float(np.linalg.norm(p1 - p2, axis=1).mean() / iod))
                    if b1 is not None and b2 is not None:
                        acc["aed"].append(100 * float(np.abs(b1 - b2).mean()))
            rows[r["id"]] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()} | dict(frames=len(fr))
            print(r["id"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[r["id"]].items()}, flush=True)
    # eye-crop FID between the raw and the embedded crop sets
    fid = None
    try:
        from pytorch_fid.inception import InceptionV3
        from scipy import linalg
        net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()

        def feats(crops):
            out = []
            for i in range(0, len(crops), 64):
                b = torch.from_numpy(np.stack([cv2.resize(c, (299, 299)) for c in crops[i:i + 64]])).permute(0, 3, 1, 2).float().div(255).to(dev)
                with torch.no_grad():
                    out.append(net(b)[0].squeeze(-1).squeeze(-1).cpu().numpy())
            return np.concatenate(out)
        f1, f2 = feats(crops_raw), feats(crops_emb)
        m1, m2, s1, s2 = f1.mean(0), f2.mean(0), np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
        cs, _ = linalg.sqrtm(s1.dot(s2), disp=False)
        fid = float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(cs.real))
    except Exception as e:  # noqa: BLE001
        print("FID skipped:", type(e).__name__, e)
    rng = np.random.default_rng(0)
    summ = {}
    for k in ("psnr", "ssim", "lpips", "eye_lpips", "akd_pct_iod", "aed", "identical_frame"):
        v = np.array([x[k] for x in rows.values() if x[k] is not None])
        bs = [float(np.mean(rng.choice(v, size=len(v)))) for _ in range(1000)]
        summ[k] = dict(mean=round(float(v.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], n_clips=len(v))
    out = dict(set=a.set, every=a.every, n_pairs_eye=len(crops_raw), eye_crop_fid=fid, summary=summ, per_clip=rows,
               note="paired: raw twin = LivePortrait without the hook, embedded = with the hook; same generation; full-frame metrics on 2x down-scaled frames")
    (ROOT / "outputs" / f"quality_paired_{a.set}.json").write_text(json.dumps(stamp(out), indent=1))
    print(json.dumps({"set": a.set, "eye_crop_fid": fid, **summ}, indent=1))
    lm.close()


if __name__ == "__main__":
    main()
