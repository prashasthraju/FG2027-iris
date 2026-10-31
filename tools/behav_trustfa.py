#!/usr/bin/env python
"""Behavioural baseline: TrustFA inter-frame feature differencing (FG 2026), one embedding per video.

Stage 1 (face crop, CPU, multiprocess): uses TrustFA's own avafp.preprocess_faces routines
  (MediaPipe FaceLandmarker on 8 probe frames -> one fixed square box, median centre, 1.6 x face
  extent; every frame cropped with that box, grayscale, 128x128 INTER_AREA). Videos are decoded
  twice (probe pass, crop pass) so a 2304 px clip is never held in RAM. Result cached as
  <out>/faces128/<id>.npy (uint8, n_frames x 128 x 128).
Stage 2 (embed, CPU by default): clip -> T=64 frames, per-clip z-score (avafp.feat_diff.normalize_clip),
  FeatDiffModel of the retrained NVFAIR checkpoint -> 256-d L2-normalised embedding.

Clip length handling (--pad):
  last      (default, as requested / paper) clips < T: repeat the last frame; clips > T: centre crop.
  pingpong  (what the retrained model saw in training/eval: avafp.feat_diff.pad_to) mirrored playback.
Nothing in the TrustFA repo is modified; it is only imported.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

TRUSTFA = Path("external/TrustFA-FG2026")
CKPT_DIR = TRUSTFA / "avafp/models/trustfa_nvfair"   # symlink -> external/data/nvfair_models/trustfa_nvfair
sys.path.insert(0, str(TRUSTFA))


# ------------------------------------------------------------------ stage 1
def crop_one(args):
    vid_id, path, out_dir = args
    out = Path(out_dir) / f"{vid_id}.npy"
    if out.exists():
        return vid_id, "skip", 0
    cv2.setNumThreads(1)
    from avafp import preprocess_faces as pf
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        return vid_id, "unreadable", 0
    probe = set(np.linspace(0, n - 1, min(pf.N_PROBE, n)).astype(int).tolist())
    probes, k = [], 0
    while True:                      # pass 1: keep only the probe frames
        if not cap.grab():
            break
        if k in probe:
            ok, f = cap.retrieve()
            if ok:
                probes.append(f)
        k += 1
    cap.release()
    if k == 0:
        return vid_id, "unreadable", 0
    box = pf.estimate_box(probes)    # same estimator as the repo (landmark extent, median centre)
    del probes
    cap = cv2.VideoCapture(str(path))
    clip = []
    while True:                      # pass 2: crop every frame with the fixed box
        ok, f = cap.read()
        if not ok:
            break
        clip.append(pf.crop_frames([f], box)[0])
    cap.release()
    np.save(out, np.stack(clip))
    return vid_id, "ok" if box is not None else "noface", len(clip)


# ------------------------------------------------------------------ stage 2
def make_window(frames, T, pad):
    n = len(frames)
    if n < T:
        if pad == "last":
            return np.concatenate([frames, np.repeat(frames[-1:], T - n, 0)])
        from avafp.feat_diff import pad_to
        return pad_to(frames, T)[:T]
    s = (n - T) // 2                 # centre crop
    return frames[s:s + T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="outputs/behavioural/videos.json")
    ap.add_argument("--out", default="outputs/behavioural/trustfa")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--pad", choices=["last", "pingpong"], default="last")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-name", default="emb.npz")
    ap.add_argument("--skip-crop", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    faces = out / "faces128"
    faces.mkdir(parents=True, exist_ok=True)
    vids = json.load(open(a.videos))[: a.limit]
    t0 = time.time()

    if not a.skip_crop:
        todo = [(v["id"], v["video"], str(faces)) for v in vids]
        stats, bad = {}, []
        with Pool(a.workers) as pool:
            for i, (vid, st, n) in enumerate(pool.imap_unordered(crop_one, todo)):
                stats[st] = stats.get(st, 0) + 1
                if st in ("noface", "unreadable"):
                    bad.append((vid, st))
                if (i + 1) % 50 == 0:
                    print(f"crop {i + 1}/{len(todo)} {stats} {time.time() - t0:.0f}s", flush=True)
        print("crop done", stats, f"{time.time() - t0:.0f}s", flush=True)
        (out / "crop_problems.json").write_text(json.dumps(bad))
    t_crop = time.time() - t0

    import torch
    from avafp.feat_diff import _load, normalize_clip
    torch.set_num_threads(a.workers)
    ck = Path(a.ckpt_dir)
    model = _load(ck, a.device)
    T = a.T
    assert T == model.avafp_cfg["clip_length"], "T differs from the checkpoint's clip_length"
    clips = []
    for v in vids:
        fr = np.load(faces / f"{v['id']}.npy")
        if len(fr) != v["frames"]:   # videos.json 'frames' can be off by one vs the real stream (ffprobe agrees with cv2)
            print(f"note: {v['id']} has {len(fr)} decoded frames, videos.json says {v['frames']}")
        clips.append(make_window(fr, T, a.pad))
    t1 = time.time()
    embs = []
    with torch.no_grad():
        for s in range(0, len(clips), a.batch):
            x = torch.stack([normalize_clip(c) for c in clips[s:s + a.batch]]).to(a.device)
            embs.append(model(x).float().cpu())
    emb = torch.nn.functional.normalize(torch.cat(embs), dim=1).numpy().astype(np.float32)
    t_emb = time.time() - t1
    ids = np.array([v["id"] for v in vids])
    np.savez(out / a.out_name, ids=ids, emb=emb)
    print(f"wrote {out / a.out_name} {emb.shape} crop {t_crop:.0f}s embed {t_emb:.0f}s "
          f"(ckpt {ck.resolve()}/checkpoint.pt, pad={a.pad}, T={T}, bs={a.batch}, dev={a.device})")


if __name__ == "__main__":
    main()
