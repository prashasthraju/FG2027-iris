#!/usr/bin/env python
"""AVAPrintDB facial-motion spatio-temporal GCN ("graph") baseline on our avatar videos.

Pipeline (faithful to AVAPrintDB, deviations documented in outputs/behavioural/graph/run.md):
  video --(MediaPipe FaceLandmarker 'face_landmarker.task', VIDEO mode, 109 selected
          landmarks, nose-centred, scaled by inter-ocular distance; official
          database/preprocessing/landmarks_extraction/landmark_extraction.py functions)--> (T,109,3)
  --(70-frame windows, stride 35, pad-if-short by repeating the last frame)--> windows
  --(SpatialTemporalTripletModel.predict, official checkpoint)--> (W,256) L2-normalised
  --> mean over windows, L2-normalised --> one 256-d embedding per video.

Run with the main environment (it has mediapipe and torch_geometric):
  python tools/behav_graph.py \
      --videos outputs/behavioural/videos.json --out-dir outputs/behavioural/graph
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

AVA = Path("external/AVAPrintDB")
LM_PY = AVA / "database/preprocessing/landmarks_extraction/landmark_extraction.py"
LM_TASK = AVA / "database/preprocessing/landmarks_extraction/face_landmarker.task"
CODE = AVA / "benchmark/code/graph_model"
CKPT = {
    "ravdess_liveportrait": AVA / "benchmark/checkpoints/graph-based/checkpoint_graph_trained_ravdess_liveportrait.pt",
    "cremad_liveportrait": AVA / "benchmark/checkpoints/graph-based/checkpoint_graph_trained_cremad_liveportrait.pt",
}
WIN, STRIDE = 70, 35  # official graph evaluation: --num-frames 70 --window-stride 35


def _load_lm_module():
    spec = importlib.util.spec_from_file_location("ava_landmark_extraction", LM_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def content_box(f, tol=6):
    """Bounding box of the non-background area (our NIR faces sit on a flat grey canvas)."""
    g = f if f.ndim == 2 else f.astype(np.float32).mean(2)
    edge = np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])
    bg = np.median(edge)
    ys, xs = np.where(np.abs(g - bg) > tol)
    if len(xs) == 0:
        return 0, 0, g.shape[1], g.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def extract_landmarks(video, crop="auto", maxside=1024):
    """Return (T_ok, 109, 3) normalised landmarks and a stats dict.

    crop: 'none'  -> full frame (what the benchmark does: its videos are already face frames)
          'auto'  -> crop to the content box of frame 0 (+35% replicate border) so the face fills the
                     frame like a benchmark avatar; frames are then downscaled to <= maxside.
    """
    import cv2
    import mediapipe as mp
    lm = _load_lm_module()
    ids = lm.LANDMARK_IDS
    nose_i, left_i, right_i = ids.index(4), ids.index(133), ids.index(362)
    landmarker = lm.create_landmarker(str(LM_TASK), prefer_gpu=False)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    anchors = {"nose": None, "left": None, "right": None}
    out, n, det, box = [], 0, 0, None
    import contextlib
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if crop == "auto":
            if box is None:
                box = content_box(fr)
            x0, y0, x1, y1 = box
            fr = fr[y0:y1, x0:x1]
            p = int(0.35 * max(fr.shape[:2]))
            fr = cv2.copyMakeBorder(fr, p, p, p, p, cv2.BORDER_REPLICATE)
            s = maxside / max(fr.shape[:2])
            if s < 1:
                fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(img, int(n / fps * 1000))
        n += 1
        if res.face_landmarks:
            det += 1
            l = res.face_landmarks[0]
            pts = np.full((len(ids), 3), np.nan, np.float32)
            for i, lid in enumerate(ids):
                if lid < len(l):
                    pts[i] = (l[lid].x, l[lid].y, l[lid].z)
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                nrm, anchors = lm.normalize_landmarks(pts, nose_i, left_i, right_i, last_valid_anchors=anchors)
            if nrm is not None:
                out.append(nrm)
    cap.release()
    landmarker.close()
    arr = np.stack(out).astype(np.float32) if out else np.zeros((0, len(ids), 3), np.float32)
    return arr, dict(frames=n, detected=det, kept=len(out))


def _job(a):
    vid, out_npy, crop = a
    t = time.time()
    try:
        arr, st = extract_landmarks(vid, crop)
        np.save(out_npy, arr)
        st["err"] = None
    except Exception as e:  # noqa
        st = dict(frames=0, detected=0, kept=0, err=repr(e))
    st["sec"] = time.time() - t
    return st


def windows(x, size=WIN, stride=STRIDE):
    """Official sliding windows (stride 35), with the benchmark's --pad-if-short behaviour:
    a sequence shorter than `size` is padded by repeating its last frame into ONE window."""
    T = x.shape[0]
    if T < size:
        x = np.concatenate([x, np.repeat(x[-1:], size - T, axis=0)], 0)
        return x[None]
    return np.stack([x[s:s + size] for s in range(0, T - size + 1, stride)])


def load_model(ckpt, device):
    import pandas as pd
    import torch
    sys.path.insert(0, str(CODE))
    spec = importlib.util.spec_from_file_location("ava_graph_models", CODE / "models.py")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    e = pd.read_csv(CODE / "delaunay_edges.csv")
    edges = list(zip(e["i"], e["j"])) + list(zip(e["j"], e["i"]))
    ei = torch.tensor(edges, dtype=torch.long).T.contiguous().to(device)
    m = models.SpatialTemporalTripletModel(edge_index=ei, num_nodes=109, in_dim=3,
                                           hidden_dim=64, embed_dim=256).to(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="outputs/behavioural/videos.json")
    ap.add_argument("--out-dir", default="outputs/behavioural/graph")
    ap.add_argument("--ckpt", default="ravdess_liveportrait", choices=list(CKPT))
    ap.add_argument("--crop", default="auto", choices=["auto", "none"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--emb-name", default="emb.npz")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    vids = json.load(open(a.videos))
    if a.limit:
        vids = vids[:a.limit]
    od = Path(a.out_dir)
    ld = od / f"landmarks_{a.crop}"
    ld.mkdir(parents=True, exist_ok=True)
    jobs = [(v["video"], str(ld / f"{v['id']}.npy"), a.crop) for v in vids]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    t0 = time.time()
    stats = {}
    if todo:
        with ProcessPoolExecutor(a.workers) as ex:
            for k, (j, st) in enumerate(zip(todo, ex.map(_job, todo))):
                stats[j[0]] = st
                if st["err"]:
                    print("ERR", j[0], st["err"], flush=True)
                if k % 25 == 0:
                    print(f"landmarks {k+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    t_lm = time.time() - t0

    model = load_model(CKPT[a.ckpt], a.device)
    embs, rows = [], []
    t1 = time.time()
    for v in vids:
        x = np.load(ld / f"{v['id']}.npy")
        if x.shape[0] == 0:
            raise RuntimeError(f"no usable face in {v['video']}")
        w = windows(x)
        with torch.no_grad():
            e = model.predict(torch.from_numpy(w).to(a.device))[0].float().cpu().numpy()
        m = e.mean(0)
        embs.append(m / np.linalg.norm(m))
        rows.append((v["id"], x.shape[0], w.shape[0]))
    emb = np.stack(embs).astype(np.float32)
    ids = np.array([v["id"] for v in vids], dtype=str)
    np.savez(od / a.emb_name, ids=ids, emb=emb)
    json.dump(dict(rows=rows, landmark_sec=t_lm, embed_sec=time.time() - t1, ckpt=str(CKPT[a.ckpt]),
                   crop=a.crop), open(od / (a.emb_name + ".meta.json"), "w"))
    print("wrote", od / a.emb_name, emb.shape, f"landmarks {t_lm:.0f}s embed {time.time()-t1:.0f}s")


if __name__ == "__main__":
    main()
