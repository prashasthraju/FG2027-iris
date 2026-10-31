"""Reviewer request B: how visible is the pasted proxy iris? Colour shift of the avatar's iris, embedded video vs its raw twin (README.md, part B).

For every sampled session the embedded video and its raw twin (identical frames, no hook) are read in lockstep; on every 4th frame the irises are detected on the RAW twin with
the project detector (iris_pipeline.segment.extract_irises). The mask is seg.visible (iris annulus between pupil and limbus, inside the lids, without specular highlights; bbox-sized)
eroded by 1-2 px, so eyelid skin and highlights the hook never touched do not dilute the numbers. Both crops go to CIELAB (skimage.color.rgb2lab, sRGB D65, float RGB in [0,1]).
Per eye and frame:
  dE_pix   mean CIEDE2000 between co-located pixels of the annulus (also counts the texture change, large even if the average colour is equal)
  dE_mean  CIEDE2000 between the MEAN Lab colour of the raw annulus and of the embedded annulus (what "the iris colour class changed" is about; the 5 / 10 thresholds are applied to it)
  dL       signed L* shift of the mean colour (embedded minus raw);  dh  circular hue-angle shift of the mean (a*, b*) in degrees, NaN if chroma < 2 in either
An eye counts as `changed` if the maximum |dRGB| inside the mask is > 3 (8-bit); eyes the hook skipped (visibility gate) are unchanged and are reported separately.
Per session = mean over the changed eyes of the sampled frames.

Reference scales (all in dE2000 of mean annulus colour unless said otherwise):
  twin_vs_twin_next   raw twin eye at frame t vs t+4 (natural temporal variation: gaze, blur, compression)
  source_vs_twin      iris of the SOURCE photo (detected on the photo) vs the raw twin iris of the avatar (what the generator alone does to the colour)
  between_people      raw twin iris colour of two sessions with different source persons (random pairs, same dataset, same side)
  two_proxies         imitation session vs its orig_session: same photo, same driving clip, a different driver and so a different embedded proxy; frame-aligned embedded-vs-embedded

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python tools/appearance_change.py --workers 6     # -> outputs/appearance_change/results.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "outputs" / "appearance_change"
SEED = 20261031
EVERY = 4
N_LP = {"ravdess": 50, "nir": 50}
CHANGED_RGB = 3
_W = {}


def _init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    cv2.setNumThreads(2)
    from iris_pipeline.detect import FaceEyeDetector
    _W["det"] = FaceEyeDetector().__enter__()


def lockstep(pa, pb, every):
    import cv2
    ca, cb = cv2.VideoCapture(str(pa)), cv2.VideoCapture(str(pb))
    i = 0
    while True:
        oka, a = ca.read()
        okb, b = cb.read()
        if oka != okb:
            raise RuntimeError(f"frame counts differ: {pa} vs {pb}")
        if not oka:
            break
        if i % every == 0:
            yield i, cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        i += 1


def eroded_mask(seg, shape):
    import cv2
    x0, y0, x1, y1 = [int(v) for v in seg.bbox]
    vis = np.asarray(seg.visible, bool)
    if vis.shape != (y1 - y0, x1 - x0):
        return None
    r = float(np.mean(seg.iris[2:3]))
    k = 1 if r < 15 else 2
    m = cv2.erode(vis.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    return (x0, y0, x1, y1), m


def lab(rgb_u8):
    from skimage.color import rgb2lab
    return rgb2lab(rgb_u8.astype(np.float64) / 255.0)


def de00(a, b):
    from skimage.color import deltaE_ciede2000
    return deltaE_ciede2000(a, b)


def mean_lab(L, m):
    return L[m].mean(0)


def hue_chroma(c):
    return np.degrees(np.arctan2(c[2], c[1])) % 360.0, float(np.hypot(c[1], c[2]))


def dhue(c1, c2):
    h1, k1 = hue_chroma(c1)
    h2, k2 = hue_chroma(c2)
    if min(k1, k2) < 2.0:
        return np.nan
    return float((h2 - h1 + 180.0) % 360.0 - 180.0)


def eye_metrics(ra, ea, bbox, m):
    x0, y0, x1, y1 = bbox
    H, W = ra.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    ra_c, ea_c = ra[y0:y1, x0:x1], ea[y0:y1, x0:x1]
    if ra_c.shape[:2] != m.shape or m.sum() < 30:
        return None
    Lr, Le = lab(ra_c), lab(ea_c)
    pix = de00(Lr[m], Le[m])
    mr, me = mean_lab(Lr, m), mean_lab(Le, m)
    maxd = int(np.abs(ra_c[m].astype(int) - ea_c[m].astype(int)).max())
    return dict(dE_pix=float(pix.mean()), dE_mean=float(de00(mr[None], me[None])[0]), dL=float(me[0] - mr[0]), dh=dhue(mr, me),
                chroma_raw=hue_chroma(mr)[1], chroma_emb=hue_chroma(me)[1], L_raw=float(mr[0]), max_drgb=maxd,
                frac_changed=float((np.abs(ra_c[m].astype(int) - ea_c[m].astype(int)).max(-1) > CHANGED_RGB).mean()), changed=maxd > CHANGED_RGB,
                mean_raw=mr.tolist())


def session_task(m):
    from iris_pipeline.segment import extract_irises
    import cv2
    import proxy_verify as V
    t0 = time.time()
    det = _W["det"]
    eyes, prev, nxt = [], {}, []
    raw_means = defaultdict(list)
    diam = []
    for i, ea, ra in lockstep(ROOT / m["video"], ROOT / m["raw_twin"], EVERY):
        for seg in extract_irises(ra, det):
            if seg.side not in ("image_left", "image_right"):
                continue
            r = eroded_mask(seg, ra.shape)
            if r is None:
                continue
            em = eye_metrics(ra, ea, *r)
            if em is None:
                continue
            em.update(frame=i, side=seg.side, diam=float(seg.diameter_px))
            eyes.append(em)
            diam.append(em["diam"])
            raw_means[seg.side].append(np.array(em["mean_raw"]))
            if seg.side in prev:
                nxt.append(float(de00(prev[seg.side][None], np.array(em["mean_raw"])[None])[0]))
            prev[seg.side] = np.array(em["mean_raw"])
    out = dict(id=m["id"], dataset=m["dataset"], generator=m.get("generator") or "liveportrait", kind=m["kind"], source_actor=m["source_actor"], eyes=eyes,
               twin_vs_twin_next=nxt, raw_mean={s: np.mean(v, 0).tolist() for s, v in raw_means.items()})
    # source photo iris colour (geometry differs, so mean annulus colour only)
    src = {}
    try:
        photo = cv2.cvtColor(cv2.imread(str(m["source"])), cv2.COLOR_BGR2RGB)
        for seg in extract_irises(photo, det):
            r = eroded_mask(seg, photo.shape)
            if seg.side in ("image_left", "image_right") and r is not None:
                (x0, y0, x1, y1), mk = r
                if mk.sum() >= 30:
                    src[seg.side] = mean_lab(lab(photo[y0:y1, x0:x1]), mk).tolist()
    except Exception as e:      # noqa: BLE001
        out["source_error"] = repr(e)
    out["source_mean"] = src
    out["seconds"] = time.time() - t0
    return out


def pair_task(args):
    """Two embedded videos of the same photo and driving clip, different driver proxy: embedded vs embedded on the shared frames."""
    from iris_pipeline.segment import extract_irises
    mi, mo = args
    det = _W["det"]
    # raw twins must be (near) identical: measured, not assumed
    ev, rows, twin_diff = [], [], []
    import cv2
    cs = [cv2.VideoCapture(str(ROOT / p)) for p in (mi["video"], mo["video"], mi["raw_twin"], mo["raw_twin"])]
    i = 0
    while True:
        fr = [c.read() for c in cs]
        if not all(o for o, _ in fr):
            break
        if i % EVERY == 0:
            a, b, ra, rb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in fr]
            twin_diff.append(float(np.abs(ra.astype(np.int16) - rb.astype(np.int16)).mean()))
            for seg in extract_irises(ra, det):
                r = eroded_mask(seg, ra.shape)
                if seg.side not in ("image_left", "image_right") or r is None:
                    continue
                (x0, y0, x1, y1), mk = r
                if mk.sum() < 30:
                    continue
                La, Lb = lab(a[y0:y1, x0:x1]), lab(b[y0:y1, x0:x1])
                ma, mb = mean_lab(La, mk), mean_lab(Lb, mk)
                rows.append(dict(dE_pix=float(de00(La[mk], Lb[mk]).mean()), dE_mean=float(de00(ma[None], mb[None])[0]), dL=float(mb[0] - ma[0]), dh=dhue(ma, mb)))
        i += 1
    return dict(id=mi["id"], orig=mo["id"], dataset=mi["dataset"], entries=(mi.get("embed_entry"), mo.get("embed_entry")), n_eyes=len(rows), eyes=rows,
                raw_twin_mean_abs_diff=float(np.mean(twin_diff)) if twin_diff else None)


def run(task):
    return (pair_task if isinstance(task, tuple) else session_task)(task)


# --------------------------------------------------------------------------------------------- selection and statistics
def select():
    import proxy_verify as V
    rng = np.random.default_rng(SEED)
    allm = [m for m in V.sessions(all_generators=True) if m["kind"] in ("self", "cross") and not m["tuning"] and m.get("raw_twin") and (ROOT / m["raw_twin"]).exists()]
    lp = [m for m in allm if (m.get("generator") or "liveportrait") == "liveportrait"]
    oth = [m for m in allm if (m.get("generator") or "liveportrait") != "liveportrait"]
    pick = []
    for ds, n in N_LP.items():
        c = sorted((m for m in lp if m["dataset"] == ds), key=lambda m: m["id"])
        pick += [c[i] for i in rng.permutation(len(c))[:n]]
    return pick + sorted(oth, key=lambda m: m["id"])


def pairs():
    import proxy_verify as V
    ms = {m["id"]: m for m in V.sessions(all_generators=True)}
    out = []
    for m in ms.values():
        if m["kind"] == "imitation" and not m["tuning"] and m.get("orig_session") in ms and ms[m["orig_session"]].get("embed_entry") != m.get("embed_entry"):
            out.append((m, ms[m["orig_session"]]))
    return out


def q(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), mean=round(float(x.mean()), 2), median=round(float(np.median(x)), 2), p90=round(float(np.quantile(x, .9)), 2), max=round(float(x.max()), 2))


def summarise(rows, tag):
    """Per-session means over the changed eyes; shares of sessions above 5 / 10."""
    sess = []
    for r in rows:
        ch = [e for e in r["eyes"] if e["changed"]]
        if not ch:
            sess.append(None)
            continue
        f = lambda k: float(np.nanmean([e[k] for e in ch])) if np.isfinite([e[k] for e in ch]).any() else np.nan
        sess.append(dict(dE_pix=f("dE_pix"), dE_mean=f("dE_mean"), dL=f("dL"), dL_abs=float(np.mean([abs(e["dL"]) for e in ch])), dh=f("dh"),
                         dh_abs=float(np.nanmean([abs(e["dh"]) for e in ch])) if np.isfinite([e["dh"] for e in ch]).any() else np.nan,
                         chroma_raw=f("chroma_raw"), diam=float(np.mean([e["diam"] for e in ch]))))
    ok = [s for s in sess if s]
    eyes = [e for r in rows for e in r["eyes"]]
    ch = [e for e in eyes if e["changed"]]
    out = dict(n_sessions=len(rows), n_sessions_with_changed_eye=len(ok), n_eyes_detected=len(eyes), n_eyes_changed=len(ch),
               share_eyes_unchanged=round(1 - len(ch) / max(len(eyes), 1), 3),
               session_dE_pix=q([s["dE_pix"] for s in ok]), session_dE_mean=q([s["dE_mean"] for s in ok]),
               session_dL_signed=q([s["dL"] for s in ok]), session_dL_abs=q([s["dL_abs"] for s in ok]), session_dh_abs=q([s["dh_abs"] for s in ok]),
               raw_chroma=q([s["chroma_raw"] for s in ok]), iris_diameter_px=q([s["diam"] for s in ok]),
               eye_dE_pix_changed=q([e["dE_pix"] for e in ch]), eye_dE_mean_changed=q([e["dE_mean"] for e in ch]),
               eye_dE_mean_all_eyes_incl_unchanged=q([e["dE_mean"] for e in eyes]),
               frac_annulus_pixels_changed_median=round(float(np.median([e["frac_changed"] for e in ch])), 3) if ch else None)
    for thr in (5, 10):
        out[f"share_sessions_dE_mean_gt{thr}"] = dict(k=sum(s["dE_mean"] > thr for s in ok), n=len(ok), rate=round(sum(s["dE_mean"] > thr for s in ok) / max(len(ok), 1), 3))
        out[f"share_sessions_dE_pix_gt{thr}"] = dict(k=sum(s["dE_pix"] > thr for s in ok), n=len(ok), rate=round(sum(s["dE_pix"] > thr for s in ok) / max(len(ok), 1), 3))
        out[f"share_eyes_dE_mean_gt{thr}"] = round(float(np.mean([e["dE_mean"] > thr for e in ch])), 3) if ch else None
    out["tag"] = tag
    return out


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select()
    pr = pairs()
    if a.limit:
        sel, pr = sel[:a.limit], pr[:a.limit]
    t0 = time.time()
    tasks = [m for m in sel] + [p for p in pr]
    res = []
    with Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(run, tasks, chunksize=1), 1):
            res.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    sess = [r for r in res if "eyes" in r and "orig" not in r]
    prs = [r for r in res if "orig" in r]
    (OUT / "per_session.json").write_text(json.dumps(sess, default=float))
    out = dict(note=("embedded video vs raw twin, mask = seg.visible eroded 1-2 px on every 4th frame, CIELAB via skimage.color.rgb2lab, CIEDE2000 via skimage.color.deltaE_ciede2000; "
                     "dE_mean (mean colour of the annulus) carries the 5 / 10 visibility thresholds, dE_pix also counts texture change"),
               seed=SEED, every=EVERY, n_sampled={ds: sum(m["dataset"] == ds and (m.get("generator") or "liveportrait") == "liveportrait" for m in sel) for ds in ("ravdess", "nir")},
               by_group={}, by_dataset={}, by_generator={})
    key = lambda r: (r["generator"], r["dataset"])
    for k in sorted({key(r) for r in sess}):
        out["by_group"][f"{k[0]}/{k[1]}"] = summarise([r for r in sess if key(r) == k], f"{k[0]}/{k[1]}")
    for ds in ("ravdess", "nir"):
        out["by_dataset"][ds] = summarise([r for r in sess if r["dataset"] == ds and r["generator"] == "liveportrait"], f"liveportrait/{ds}")
    for g in sorted({r["generator"] for r in sess}):
        out["by_generator"][g] = summarise([r for r in sess if r["generator"] == g], g)
    out["all_sessions"] = summarise(sess, "all")
    # reference scales
    ref = {}
    ref["twin_vs_twin_next"] = {ds: q([np.median(r["twin_vs_twin_next"]) for r in sess if r["dataset"] == ds and r["generator"] == "liveportrait" and r["twin_vs_twin_next"]]) for ds in ("ravdess", "nir")}
    sv = defaultdict(list)
    for r in sess:
        for s, c in r["source_mean"].items():
            if s in r["raw_mean"]:
                sv[(r["generator"], r["dataset"])].append(float(de00(np.array(c)[None], np.array(r["raw_mean"][s])[None])[0]))
    ref["source_vs_twin"] = {f"{k[0]}/{k[1]}": q(v) for k, v in sorted(sv.items())}
    rng = np.random.default_rng(SEED)
    bp = {}
    for ds in ("ravdess", "nir"):
        c = [r for r in sess if r["dataset"] == ds and r["generator"] == "liveportrait" and r["raw_mean"]]
        d = []
        for _ in range(3000 if c else 0):
            i, j = rng.integers(0, len(c), 2)
            if c[i]["source_actor"] == c[j]["source_actor"]:
                continue
            for s in set(c[i]["raw_mean"]) & set(c[j]["raw_mean"]):
                d.append(float(de00(np.array(c[i]["raw_mean"][s])[None], np.array(c[j]["raw_mean"][s])[None])[0]))
        bp[ds] = q(d)
    ref["between_people"] = bp
    ref["two_proxies_same_frames"] = dict(
        n_pairs=len(prs), raw_twin_mean_abs_diff_rgb=q([r["raw_twin_mean_abs_diff"] for r in prs if r["raw_twin_mean_abs_diff"] is not None]),
        pair_dE_pix=q([np.mean([e["dE_pix"] for e in r["eyes"]]) for r in prs if r["eyes"]]), pair_dE_mean=q([np.mean([e["dE_mean"] for e in r["eyes"]]) for r in prs if r["eyes"]]),
        pair_dL_abs=q([np.mean([abs(e["dL"]) for e in r["eyes"]]) for r in prs if r["eyes"]]),
        share_pairs_dE_mean_gt5=round(float(np.mean([np.mean([e["dE_mean"] for e in r["eyes"]]) > 5 for r in prs if r["eyes"]])), 3) if prs else None)
    out["reference_scales"] = ref
    out["seconds"] = round(time.time() - t0)
    out["_provenance"] = {"command": " ".join(sys.argv), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["by_group"].items():
        print(k, "n", v["n_sessions"], "dE_mean", v["session_dE_mean"], "dE_pix", v["session_dE_pix"], ">5", v["share_sessions_dE_mean_gt5"]["rate"], ">10", v["share_sessions_dE_mean_gt10"]["rate"])
    print(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
