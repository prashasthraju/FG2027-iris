"""Reviewer request A: driver attribution as a function of the iris size IN THE VIDEO (README.md).

Existing embedded LivePortrait videos (the ones the paper evaluates; iris about 88 px on the 3x RAVDESS canvas, about 162 px on the NIR canvas, as measured) are DOWNSCALED
with ffmpeg and the unchanged verifier front end is re-run on them. This is downscaling of existing outputs, NOT native generation at that size: the proxy
was written at the native size and then lost resolution, a native low-resolution run would write it with the hook's own resampling at that size.

  select     seeded subset of honest LivePortrait test sessions (self + cross, not tuning), half RAVDESS, half NIR     -> outputs/resolution_sweep/subset.json
  measure    native iris diameters on a few sessions (project detector), fixes the factor per target diameter           -> outputs/resolution_sweep/native_diam.json
  extract    ffmpeg downscale + ArcIris observations for every (session, target); resumable, one pickle each            -> outputs/resolution_sweep/obs/
  analyse    acceptance at the frozen T1, usable share, EER / AUC vs the other enrolled proxies, per dataset and factor  -> outputs/resolution_sweep/results.json
  plot       PNG curve                                                                                                -> outputs/resolution_sweep/curve.png

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python tools/resolution_sweep.py all --workers 6
    python tools/resolution_sweep.py analyse        # recompute the table from the stored observations

Video encode of every point (also the 1.0x one): scale=trunc(iw*f/2)*2:trunc(ih*f/2)*2:flags=area, libx264 -crf 18 -pix_fmt yuv420p -an (as the blur / alter variants of
tools/proxy_verify.py). The native row of the table is read from the existing extraction caches (original files, no re-encode); the 1.0x re-encode row shows what ffmpeg alone costs.
Observations are extracted exactly as proxy_verify.extract_key (every 2nd frame, extract_irises, ArcIrisJudge) minus the IrisCode, which the ArcIris decision does not use.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "outputs" / "resolution_sweep"
OBS = OUT / "obs"
SEED = 20261030
N_PER_DATASET = 60
TARGETS = {"ravdess": [88, 80, 60, 40, 30, 20, 15], "nir": [162, 100, 80, 60, 40, 30, 20, 15]}     # target iris diameter in px; 88 / 162 = measured native medians, re-encoded at f = 1.0
RAW_TAG = "raw"
LL_TARGETS = [60, 40, 30, 20]       # second arm, labels "L<t>": the same downscale written with libx264 -crf 0 (lossless, like the released originals) to separate resolution from the crf-18 re-encode


# ------------------------------------------------------------------------------------------------ subset
def load_state():
    import proxy_verify as V
    plan = json.loads(V.PLAN.read_text())
    return V, plan, plan["enrolment"]


def cmd_select(a):
    V, plan, enrol = load_state()
    rng = np.random.default_rng(SEED)
    pool = [m for m in V.sessions() if m["kind"] in ("self", "cross") and not m["tuning"] and V.load_cache(m["id"]) is not None]       # LivePortrait only
    subset = []
    for ds in ("ravdess", "nir"):
        c = sorted((m for m in pool if m["dataset"] == ds), key=lambda m: m["id"])
        for i in rng.permutation(len(c))[:N_PER_DATASET]:
            subset.append(dict(id=c[i]["id"], dataset=ds, kind=c[i]["kind"], driver_actor=c[i]["driver_actor"], source_actor=c[i]["source_actor"],
                               driver_did=enrol[c[i]["driver_actor"]]["did"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subset.json").write_text(json.dumps(dict(seed=SEED, n_pool={ds: sum(m["dataset"] == ds for m in pool) for ds in ("ravdess", "nir")}, sessions=subset), indent=1))
    print("subset", len(subset), {ds: sum(s["dataset"] == ds for s in subset) for ds in ("ravdess", "nir")})


def subset():
    return json.loads((OUT / "subset.json").read_text())["sessions"]


# ------------------------------------------------------------------------------------------------ worker side
_W = {}


def _init():
    if not os.environ.get("RS_GPU"):          # CPU unless asked (shared GPU: at most 2 workers, ~0.8 GB each)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    _W["det"] = FaceEyeDetector().__enter__()
    _W["J"] = ArcIrisJudge()


def encode_scaled(src, dst, f, crf=18):
    vf = f"scale=trunc(iw*{f:.6f}/2)*2:trunc(ih*{f:.6f}/2)*2:flags=area"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def extract_arc(path):
    """Observations of one mp4, as proxy_verify.extract_key but ArcIris only; also the iris diameter of every detected eye."""
    import proxy_attr_lib as L
    from cancelable.identify import geometry_agrees
    from iris_pipeline.segment import extract_irises
    det, J = _W["det"], _W["J"]
    frames = L.read_frames(path, every=2)
    obs, diam = [], []
    for i, rgb in enumerate(frames):
        for seg in extract_irises(rgb, det):
            if seg.side not in L.SIDES:
                continue
            diam.append(float(seg.diameter_px))
            v, info = J.encode_image(J.eye_crop(rgb, seg.iris))
            rej = bool(info["rejected"])
            obs.append(dict(frame=i, side=seg.side, vis=float(np.mean(seg.strip_mask)), diam=float(seg.diameter_px),
                            arc=None if rej else np.asarray(v, np.float16), geo_ok=False if rej else geometry_agrees(info["iris"])))
    return dict(n_frames=len(frames), obs=obs, shape=tuple(frames[0].shape[:2]) if frames else None)


def work(task):
    """task = (session id, dataset, label, factor or None). label 'raw' = the un-embedded twin at native size."""
    sid, ds, label, f = task
    out = OBS / str(label) / ds / f"{sid}.pkl"
    if out.exists():
        return task, 0.0, "cached"
    import proxy_verify as V
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    if label == RAW_TAG:
        d = extract_arc(V.clip_path(f"{sid}__raw"))
    else:
        with tempfile.TemporaryDirectory(prefix="ressweep_") as td:
            p = Path(td) / "v.mp4"
            encode_scaled(V.clip_path(sid), p, f, crf=0 if str(label).startswith("L") else 18)
            d = extract_arc(p)
    d.update(id=sid, label=label, factor=f)
    out.write_bytes(pickle.dumps(d))
    return task, time.time() - t0, f"{len(d['obs'])} obs"


# ------------------------------------------------------------------------------------------------ measure native diameters
def cmd_measure(a):
    import proxy_verify as V
    import proxy_attr_lib as L
    from iris_pipeline.segment import extract_irises
    _init()
    sub = subset()
    res = {}
    for ds in ("ravdess", "nir"):
        ids = [s["id"] for s in sub if s["dataset"] == ds][:12]
        d = []
        for sid in ids:
            frames = L.read_frames(V.clip_path(sid), every=2)
            for rgb in frames[::4][:5]:
                d += [float(s.diameter_px) for s in extract_irises(rgb, _W["det"]) if s.side in L.SIDES]
        res[ds] = dict(n_eyes=len(d), median=float(np.median(d)), p10=float(np.quantile(d, .1)), p90=float(np.quantile(d, .9)), n_sessions=len(ids))
        print(ds, res[ds])
    (OUT / "native_diam.json").write_text(json.dumps(res, indent=1))


def factor_of(ds, target):
    nd = json.loads((OUT / "native_diam.json").read_text())[ds]["median"]
    return 1.0 if target == TARGETS[ds][0] else round(target / nd, 5)


def tasks_all(include_raw=True):
    sub = subset()
    T = [(s["id"], s["dataset"], t, factor_of(s["dataset"], t)) for s in sub for t in TARGETS[s["dataset"]]]
    T += [(s["id"], s["dataset"], f"L{t}", factor_of(s["dataset"], t)) for s in sub for t in LL_TARGETS]
    # smallest first would hide slow big ones; interleave by shuffling with a fixed seed
    T = [T[i] for i in np.random.default_rng(1).permutation(len(T))]
    if include_raw:
        import proxy_verify as V
        raw = [(s["id"], s["dataset"], RAW_TAG, None) for s in sub if V.load_cache(f"{s['id']}__raw") is None]       # raw twins already in the extraction cache are reused
        T = raw + T
    return T


def cmd_extract(a):
    from multiprocessing import Pool
    T = tasks_all()
    todo = [t for t in T if not (OBS / str(t[2]) / t[1] / f"{t[0]}.pkl").exists()]
    if a.reverse:          # a second pool (GPU) works from the other end of the same resumable list
        todo = todo[::-1]
    print(f"{len(T)} tasks, {len(todo)} to do, {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for i, (t, dt, msg) in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {time.time() - t0:.0f}s last {t[0][:30]} {t[2]} {dt:.0f}s {msg}", flush=True)
    print(f"extract done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ analysis
def arc_vector(G, obs, min_obs):
    """Mean ArcIris distance of the usable observations to every proxy; copy of proxy_verify.Gallery.vectors for the arciris matcher. -> (vector, n usable)."""
    from iris_pipeline.judge import ArcIrisJudge
    use = [o for o in obs if o["arc"] is not None]
    if len(use) < min_obs:
        return np.full(len(G.names), np.nan), len(use)
    D = np.array([[ArcIrisJudge.distance(o["arc"].astype(np.float32), g[o["side"]]) if o["side"] in g else np.nan for g in G.arc] for o in use])
    return np.nanmean(D, 0), len(use)


def block(rows, T1, didx_k, roc_stats, rate):
    """rows = [(d vector or None, k)] -> metrics block."""
    n = len(rows)
    usable = [r for r in rows if r[0] is not None and np.isfinite(r[0]).any()]
    gen = [float(d[k]) for d, k in usable]
    imp = [float(x) for d, k in usable for x in np.delete(d, k)]
    acc = sum(bool(np.isfinite(d[k]) and d[k] < T1) for d, k in usable)
    imp_f = [x for x in imp if np.isfinite(x)]
    out = dict(n_sessions=n, usable=rate(sum(np.isfinite(d[k]) for d, k in usable), n),
               genuine_accepted_at_T1=rate(acc, n), impostor_accepted_at_T1=rate(sum(x < T1 for x in imp_f), len(imp_f)) if imp_f else None,
               genuine_distance_median=float(np.nanmedian(gen)) if gen and np.isfinite(gen).any() else None)
    if sum(np.isfinite(gen)) >= 5 and imp_f:
        out["roc"] = roc_stats(gen, imp)
    return out


def cmd_analyse(a):
    import proxy_verify as V
    import proxy_attr_lib as L
    from proxy_paper_numbers import roc_stats
    plan = json.loads(V.PLAN.read_text())
    G = V.Gallery()
    didx = {n[0]: i for i, n in enumerate(G.names)}
    T1 = json.loads(V.CALIB.read_text())["claim"]["arciris"]["T1"]
    sub = subset()
    nd = json.loads((OUT / "native_diam.json").read_text())
    kof = {s["id"]: didx[s["driver_did"]] for s in sub}
    res = dict(T1=T1, n_proxies=len(G.names), seed=SEED, encode="area downscale, libx264 crf 18 yuv420p; 1.0x row = re-encode only", native_diameter_px=nd, by_dataset={}, check={})

    def vec_native(sid):
        c = V.load_cache(sid)
        return None if c is None else G.vectors(c)[0]["arciris"]

    # consistency of my observation copy with the cache: the 1.0x re-encode is not bit-identical, so compare the arc code path on the CACHED observations
    chk = []
    for s in sub[:5] + sub[-5:]:
        c = V.load_cache(s["id"])
        v1 = G.vectors(c)[0]["arciris"]
        v2, _ = arc_vector(G, c["obs"], L.MIN_OBS)
        chk.append(float(np.nanmax(np.abs(v1 - v2))) if np.isfinite(v1).any() else 0.0)
    res["check"]["arc_vector_vs_gallery_vectors_max_abs_diff"] = max(chk)

    for ds in ("ravdess", "nir"):
        S = [s for s in sub if s["dataset"] == ds]
        table = []
        # native, from the existing caches
        rows = [(vec_native(s["id"]), kof[s["id"]]) for s in S]
        nat = block(rows, T1, kof, roc_stats, V.rate)
        nat.update(label="native (original files)", arm="native", target=nd[ds]["median"], achieved_diameter_px=nd[ds]["median"], factor=1.0, n_obs_median=None)
        table.append(nat)
        # raw twin at native size
        rows, nobs = [], []
        anyacc = 0
        for s in S:
            c = V.load_cache(f"{s['id']}__raw")
            if c is not None:
                d, n = G.vectors(c)[0]["arciris"], None
            else:
                p = OBS / RAW_TAG / ds / f"{s['id']}.pkl"
                if not p.exists():
                    rows.append((None, kof[s["id"]]))
                    continue
                d, _ = arc_vector(G, pickle.loads(p.read_bytes())["obs"], L.MIN_OBS)
            rows.append((d, kof[s["id"]]))
            anyacc += int(np.nanmin(d) < T1) if np.isfinite(d).any() else 0
        raw = block(rows, T1, kof, roc_stats, V.rate)
        raw.update(label="raw twin (no proxy), native size", any_proxy_accepted_at_T1=V.rate(anyacc, len(S)))
        res.setdefault("raw_twin_control", {})[ds] = raw
        arms = [(t, str(t), "crf18") for t in TARGETS[ds]] + [(t, f"L{t}", "lossless") for t in LL_TARGETS]
        for t, lab_, arm in arms:
            rows, nobs, diam = [], [], []
            miss = 0
            for s in S:
                p = OBS / lab_ / ds / f"{s['id']}.pkl"
                if not p.exists():
                    miss += 1
                    continue
                d = pickle.loads(p.read_bytes())
                v, n = arc_vector(G, d["obs"], L.MIN_OBS)
                rows.append((v, kof[s["id"]]))
                nobs.append(n)
                diam += [o["diam"] for o in d["obs"]]
            if not rows:
                continue
            b = block(rows, T1, kof, roc_stats, V.rate)
            b.update(label=f"{t} px target" + (" lossless" if arm == "lossless" else ""), arm=arm, target=t, factor=factor_of(ds, t), achieved_diameter_px=float(np.median(diam)) if diam else None,
                     achieved_diameter_p10_p90=[float(np.quantile(diam, .1)), float(np.quantile(diam, .9))] if diam else None,
                     n_obs_median=float(np.median(nobs)), n_missing=miss, n_eyes_detected=len(diam))
            table.append(b)
        res["by_dataset"][ds] = table
    res["_provenance"] = {"command": " ".join(sys.argv), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print_table(res)


def print_table(res):
    for ds, tab in res["by_dataset"].items():
        print(f"\n{ds}")
        print("label                              diam   usable  accept@T1   EER    AUC   n_gen  medObs")
        for b in tab:
            r = b.get("roc", {})
            print(f"{b['label']:34s} {b.get('achieved_diameter_px') or 0:6.1f} {b['usable']['rate']:7.2f} {b['genuine_accepted_at_T1']['rate']:8.2f}   "
                  f"{r.get('eer', float('nan')):5.1f} {r.get('auc', float('nan')):6.1f} {r.get('n_genuine', 0):5d} {b.get('n_obs_median') or 0:7.1f}")
        rc = res["raw_twin_control"][ds]
        print(f"raw twin control: usable {rc['usable']['rate']}, genuine accepted {rc['genuine_accepted_at_T1']['k']}/{rc['n_sessions']}, any proxy accepted {rc['any_proxy_accepted_at_T1']['k']}")


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.loads((OUT / "results.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for ds, col in (("ravdess", "#1b6ca8"), ("nir", "#c4511a")):
        tab = res["by_dataset"][ds]
        for arm, ls, lab_ in (("crf18", "o-", f"{ds}, crf-18 re-encode"), ("lossless", "s--", f"{ds}, lossless downscale")):
            rows = [b for b in tab if b.get("arm") == arm and b.get("achieved_diameter_px")]
            if arm == "lossless":
                rows = [tab[0]] + rows          # the native original is the lossless 1.0x point
            rows = sorted(rows, key=lambda b: b["achieved_diameter_px"])
            x = [b["achieved_diameter_px"] for b in rows]
            ax[0].plot(x, [100 * b["genuine_accepted_at_T1"]["rate"] for b in rows], ls, color=col, label=lab_, alpha=1 if arm == "crf18" else .6)
            ax[1].plot(x, [b["genuine_distance_median"] for b in rows], ls, color=col, alpha=1 if arm == "crf18" else .6)
            ok = [b for b in rows if "roc" in b]
            ax[2].plot([b["achieved_diameter_px"] for b in ok], [b["roc"]["eer"] for b in ok], ls, color=col, alpha=1 if arm == "crf18" else .6)
    ax[1].axhline(res["T1"], color="k", lw=.8, ls=":")
    ax[1].text(15, res["T1"] + .005, "T1", fontsize=8)
    for a_, t in zip(ax, ("genuine acceptance at T1 (%)", "median genuine distance (ArcIris)", "EER vs other enrolled proxies (%)")):
        a_.set_xscale("log")
        a_.set_xlabel("iris diameter in the video (px, detector median)")
        a_.set_title(t, fontsize=10)
        a_.grid(alpha=.3)
        a_.set_xticks([15, 20, 30, 40, 60, 90, 160])
        a_.set_xticklabels([15, 20, 30, 40, 60, 90, 160])
    ax[0].legend(fontsize=7)
    fig.suptitle("Downscaled existing embedded videos (not native generation at that size); 60 sessions per dataset", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "curve.png", dpi=130)
    print("wrote", OUT / "curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "measure", "extract", "analyse", "plot", "all"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for c in (["select", "measure", "extract", "analyse", "plot"] if a.cmd == "all" else [a.cmd]):
        if c == "select" and not (OUT / "subset.json").exists() or (c == "select" and a.cmd == "select"):
            cmd_select(a)
        elif c == "measure" and (not (OUT / "native_diam.json").exists() or a.cmd == "measure"):
            cmd_measure(a)
        elif c == "extract":
            cmd_extract(a)
        elif c == "analyse":
            cmd_analyse(a)
        elif c == "plot":
            cmd_plot(a)


if __name__ == "__main__":
    main()
