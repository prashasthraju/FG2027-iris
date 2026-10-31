"""Behavioural baselines 'DINOv2 features' / 'CLIP features' of AVAPrintDB on our avatar videos.

Follows AVAPrintDB exactly (database/preprocessing/foundational_extraction + benchmark/code/foundational_models):
  frames -> RGB -> Resize((224,224)) (torchvision v2, squashing, no centre-crop) -> ImageNet/CLIP normalisation
  -> frozen backbone (DINO: torch.hub dinov2_vitl14_reg CLS token, fp32, 1024-d;
                      CLIP: ViT-L/14 encode_image, fp16 weights as clip.load does on CUDA, 768-d)
  -> sliding windows of 60 frames, stride 30 (videos shorter than 60 are skipped by the benchmark)
  -> official trained TemporalAttentionPoolingModelNew head (checkpoint *_trained_ravdess_liveportrait.pt)
  -> per-window embedding, L2-normalised; pair score = mean cosine over window pairs
     = <mean of unit window embeddings (a), mean of unit window embeddings (b)>.
Our videos have 48 or 80 frames: 80 -> exactly one window [0:60] (the benchmark's own windowing),
48 -> ONE window over all 48 frames (deviation: benchmark would skip the video).
Our only extra step is geometry: the released LivePortrait canvases pad the 512 avatar frame with
grey (content = centre 2/3 of the canvas), so the content box is cropped before the 224x224 resize
(AVAPrintDB frames are the bare generator output, no padding). Other generators: whole frame.

Outputs: emb.npz {ids, emb (L2-normalised), emb_raw (mean of unit window embeddings, un-normalised;
score = emb_raw_a . emb_raw_b reproduces the benchmark score exactly), emb_w48 (head over the first
48 frames only; identical to emb for 48-frame videos), n_windows} and feats.npz (frame features, fp16).

    python tools/behav_foundation.py --model dino   # or clip
    python tools/behav_foundation.py --model clip --validate   # check against official AVAPrintDB features/scores
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True   # never write __pycache__ into AVAPrintDB
import cv2
import numpy as np
import torch

REPO = Path(".")
AVA = Path("external/AVAPrintDB")
BENCH = AVA / "benchmark"
CKPT = {"dino": BENCH / "checkpoints/dino-based/checkpoint_dino_trained_ravdess_liveportrait.pt",
        "clip": BENCH / "checkpoints/clip-based/checkpoint_clip_trained_ravdess_liveportrait.pt"}
OUTDIR = REPO / "outputs/behavioural"
WIN, STRIDE = 60, 30


# ----------------------------------------------------------------------------- backbones
def load_backbone(kind, device):
    import torchvision.transforms.v2 as T
    if kind == "dino":
        hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        net = torch.hub.load(str(hub), "dinov2_vitl14_reg", source="local").eval().to(device)  # fp32, as official
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        enc = lambda x: net(x)                                    # CLS token (B, 1024)
    else:
        import clip
        net, _ = clip.load("ViT-L/14", device=device)             # fp16 on CUDA, like the official extraction
        net.eval()
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        enc = lambda x: net.encode_image(x)                       # (B, 768), casts to model dtype internally
    for p in net.parameters():
        p.requires_grad = False
    tf = T.Compose([T.ToImage(), T.Resize((224, 224)), T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=mean, std=std)])             # == benchmark get_*_transform
    return enc, tf


def load_head(kind, device):
    sys.path.insert(0, str(BENCH / "code/foundational_models"))
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("avaprint_fm_models", BENCH / "code/foundational_models/models.py")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    ck = torch.load(CKPT[kind], map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m = models.TemporalAttentionPoolingModelNew(
        embedding_dim=sd["temporal_pooling.norm.weight"].shape[0], project_dim=sd["projection.0.weight"].shape[0],
        output_dim=sd["projection.3.weight"].shape[0], num_heads=2, save_model_to_dir=tempfile.mkdtemp(prefix="bf_"))
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


# ----------------------------------------------------------------------------- frames
def content_box(generator, H, W):
    """LivePortrait canvases pad the avatar frame with 128-grey: content = centre 2/3 (offset canvas/6)."""
    if generator == "liveportrait":
        return H // 6, 5 * H // 6, W // 6, 5 * W // 6
    return 0, H, 0, W


def read_frames(path, box_fn=None):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if box_fn is not None:
            y0, y1, x0, x1 = box_fn(*f.shape[:2])
            f = f[y0:y1, x0:x1]
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return fr


class VidSet(torch.utils.data.Dataset):
    def __init__(self, items, tf, crop):
        self.items, self.tf, self.crop = items, tf, crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, gen = self.items[i]
        fr = read_frames(path, (lambda H, W, g=gen: content_box(g, H, W)) if self.crop else None)
        return i, torch.stack([self.tf(f) for f in fr])


@torch.no_grad()
def frame_feats(enc, x, device, bs):
    out = []
    for j in range(0, len(x), bs):
        out.append(enc(x[j:j + bs].to(device)).float().cpu())
    return torch.cat(out)


# ----------------------------------------------------------------------------- windows / head
def windows(T, size=WIN, stride=STRIDE):
    if T < size:                       # benchmark skips; we use one window over everything we have
        return [(0, T)]
    return [(s, s + size) for s in range(0, T - size + 1, stride)]


@torch.no_grad()
def video_embedding(head, feats, device, limit=None):
    """-> (raw, nwin): raw = mean of L2-normalised window embeddings (benchmark pair score = raw_a . raw_b)."""
    f = feats if limit is None else feats[:limit]
    ws = windows(len(f))
    e = torch.cat([head.predict(f[a:b].unsqueeze(0).to(device)) for a, b in ws]).float()
    e = torch.nn.functional.normalize(e, dim=-1)
    return e.mean(0).cpu().numpy(), len(ws)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------- main
def run(args):
    device = torch.device(args.device)
    t0 = time.time()
    vids = json.load(open(args.videos))
    enc, tf = load_backbone(args.model, device)
    head = load_head(args.model, device)
    items = [(v["video"], v["generator"]) for v in vids]
    dl = torch.utils.data.DataLoader(VidSet(items, tf, crop=True), batch_size=None, num_workers=args.workers,
                                     prefetch_factor=2, shuffle=False)
    feats, raw, nwin, raw48 = [None] * len(vids), [None] * len(vids), [0] * len(vids), [None] * len(vids)
    for k, (i, x) in enumerate(dl):
        f = frame_feats(enc, x, device, args.bs)
        feats[i] = f
        raw[i], nwin[i] = video_embedding(head, f, device)
        raw48[i], _ = video_embedding(head, f, device, limit=48)
        if k % 25 == 0:
            mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
            print(f"{k + 1}/{len(vids)} {time.time() - t0:.0f}s peak_gpu={mem:.0f}MiB", flush=True)
    out = OUTDIR / args.model
    out.mkdir(parents=True, exist_ok=True)
    ids = np.array([v["id"] for v in vids])
    raw = np.stack(raw).astype(np.float32)
    np.savez(out / "emb.npz", ids=ids, emb=l2(raw).astype(np.float32), emb_raw=raw,
             emb_w48=l2(np.stack(raw48)).astype(np.float32), n_windows=np.array(nwin))
    lens = np.array([len(f) for f in feats])
    np.savez(out / "feats.npz", ids=ids, lens=lens,
             feats=torch.cat(feats).numpy().astype(np.float16))
    print(f"done {len(vids)} videos in {time.time() - t0:.0f}s -> {out}", flush=True)
    sanity(out / "emb.npz", vids)


def sanity(npz, vids):
    d = np.load(npz)
    for key in ("emb", "emb_w48"):
        e = d[key]
        S = e @ e.T
        for label, sel in (("liveportrait", lambda v: v["generator"] == "liveportrait"),
                           ("liveportrait 48f", lambda v: v["generator"] == "liveportrait" and v["frames"] == 48),
                           ("liveportrait 80f", lambda v: v["generator"] == "liveportrait" and v["frames"] == 80)):
            idx = [i for i, v in enumerate(vids) if sel(v)]
            drv = np.array([vids[i]["driver_actor"] for i in idx])
            s = S[np.ix_(idx, idx)]
            same = (drv[:, None] == drv[None, :]) & ~np.eye(len(idx), dtype=bool)
            diff = drv[:, None] != drv[None, :]
            print(f"[{key}] {label}: n={len(idx)} same-driver cos={s[same].mean():.4f} (pairs {same.sum() // 2}) "
                  f"diff-driver cos={s[diff].mean():.4f} (pairs {diff.sum() // 2})")


@torch.no_grad()
def validate(args):
    """Re-derive official features/scores from the AVAPrintDB mp4 (no crop) and compare."""
    import pandas as pd
    device = torch.device(args.device)
    enc, tf = load_backbone(args.model, device)
    head = load_head(args.model, device)
    sc = pd.read_csv(Path("external/TrustFA-FG2026/avafp/results") / f"{args.model}_ravdess_liveportrait/scores/scores_RAVDESS_LIVE.csv")
    names = list(dict.fromkeys(list(sc.enrolment_sample[:3000]) + list(sc.test_sample[:3000])))[:8]
    cache = {}
    emb_dir = AVA / "database/data/embeddings" / ("DINOFeats" if args.model == "dino" else "CLIPFeats") / "TEST/LIVE"
    for n in names:
        fr = read_frames(AVA / "database/data/videos/TEST/LIVE" / (n + ".mp4"))
        x = torch.stack([tf(f) for f in fr])
        mine = frame_feats(enc, x, device, args.bs)
        off = torch.load(emb_dir / (n + ".pt"))
        cs = torch.nn.functional.cosine_similarity(mine, off, dim=1)
        print(f"{n[:30]} frames mine={len(mine)} official={len(off)}  frame-feat cos min={cs.min():.4f} mean={cs.mean():.5f}")
        cache[n] = (mine, off)

    def win_emb(f):
        ws = windows(len(f))
        e = torch.cat([head.predict(f[a:b].unsqueeze(0).to(device)) for a, b in ws])
        return torch.nn.functional.normalize(e, dim=-1)
    sub = sc[sc.enrolment_sample.isin(names) & sc.test_sample.isin(names)].head(40)
    d = []
    for _, r in sub.iterrows():
        ea, eb = win_emb(cache[r.enrolment_sample][0]), win_emb(cache[r.test_sample][0])
        d.append((float((ea @ eb.T).mean()), r.score))
    d = np.array(d)
    print(f"pair-score check on {len(d)} official pairs: max |mine-official| = {np.abs(d[:, 0] - d[:, 1]).max():.5f}, "
          f"corr = {np.corrcoef(d.T)[0, 1]:.5f}")
    # same through the *official* pre-extracted features (checks head/windowing only)
    d2 = []
    for _, r in sub.iterrows():
        ea, eb = win_emb(cache[r.enrolment_sample][1]), win_emb(cache[r.test_sample][1])
        d2.append((float((ea @ eb.T).mean()), r.score))
    d2 = np.array(d2)
    print(f"head-only check (official features): max |diff| = {np.abs(d2[:, 0] - d2[:, 1]).max():.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dino", "clip"])
    ap.add_argument("--videos", default=str(OUTDIR / "videos.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    validate(a) if a.validate else run(a)
