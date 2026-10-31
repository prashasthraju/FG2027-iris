# LEGACY: part of an earlier design (public proxy codes / 1:N identification of the proxy in the video / local Anvil chain). The submitted protocol
# (commit-and-reveal, 1:1 check against the opened proxy) is implemented in tools/proxy_issuer_eval.py (verify_session) and tools/proxy_ledger.py; see README.md.
"""Cross-capture identity transfer test (the real embedding test).

Donor  = UTIRIS capture 1 of an eye class.   Gallery = captures 2..5 of all classes.
Target = a face (CASIA-Iris-Distance, iris ~170 px; or RAVDESS, ~29 px).
For each (target, donor): embed -> re-detect edited face from scratch ->
re-extract -> identify against the gallery (158 classes).

Reported:
  rank-1 / EER of embedded results   vs   the same donors matched directly
  (no embedding: the upper bound the transplant can reach),
  and target removal: HD(edited face, *another capture of the target person*)
  vs HD(original face, that capture).

Needs outputs/cache_utiris from tools/baseline_utiris.py.
    python tools/transfer_eval.py --targets casia --n-targets 10 --n-donors 40
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from provenance import stamp
from iris_pipeline.embed import IrisTexture, embed_into_eye
from iris_pipeline.encode import encode, hamming
from iris_pipeline.segment import extract_irises, segment_closeup
from iris_pipeline.composite import band_ratio
from baseline import eer
import baseline_utiris
from baseline_utiris import load_cache

CASIA = "dbs/archive (3)/CASIA-Iris-Distance"
SIDE = "image_left"
WL = 16.0
_det = None


def detector():
    global _det
    if _det is None:
        from iris_pipeline.detect import FaceEyeDetector
        _det = FaceEyeDetector()
    return _det


def load_rgb(p):
    return np.ascontiguousarray(cv2.imread(str(p))[:, :, ::-1])


UTIRIS = "dbs/UTIRIS_V.1/RGB Images"


def target_list(kind, n):
    if kind == "utiris":  # high-res visible-light close-ups as targets (capture 2; capture 3 = other photo)
        subs = sorted(glob.glob(UTIRIS + "/*"))[1::7][:n]
        return [(f"{s}/IMG_{Path(s).name}_L_2.JPG", f"{s}/IMG_{Path(s).name}_L_3.JPG") for s in subs]
    if kind == "casia":
        subs = sorted(glob.glob(CASIA + "/*"))[:: max(1, len(glob.glob(CASIA + "/*")) // n)][:n]
        return [(sorted(glob.glob(s + "/*.jpg"))[0], sorted(glob.glob(s + "/*.jpg"))[1]) for s in subs]
    frames = sorted(glob.glob("samples/baseline/*_1.png"))[:n]
    return [(f, f.replace("_1.png", "_2.png")) for f in frames]


def seg_of(img, closeup):
    if closeup:
        try:
            return segment_closeup(img)
        except ValueError:
            return None
    return {s.side: s for s in extract_irises(img, detector())}.get(SIDE)


def realism(orig, res, seg):
    """seam: limbus-band gradient energy (lateral arcs), result / original
    (1 = same edge strength; >1 = harder seam). sharp: fine/mid band ratio inside
    the visible iris, result / original (1 = matched blur)."""
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    def grad(im):
        x = g(im)
        return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    cx, cy, R = seg.iris
    th = np.deg2rad(np.r_[np.arange(-25, 26, 2), np.arange(155, 206, 2)])
    rr = np.linspace(0.95 * R, 1.05 * R, 11)
    xs = (cx + rr[:, None] * np.cos(th)[None]).astype(np.float32)
    ys = (cy + rr[:, None] * np.sin(th)[None]).astype(np.float32)
    band = lambda im: float(cv2.remap(grad(im), xs, ys, cv2.INTER_LINEAR).mean())
    x0, y0, x1, y1 = seg.bbox
    m = seg.visible
    br = lambda im: band_ratio(g(im)[y0:y1, x0:x1] / 255.0, m)
    return band(res) / max(band(orig), 1e-6), br(res) / max(br(orig), 1e-6)


def run_target(job):
    """All donors for one target face. Returns list of per-pair dicts."""
    (tpath, tpath2), donors, color, method, closeup, opts = job
    rgb = load_rgb(tpath)
    tseg = seg_of(rgb, closeup)
    t2 = seg_of(load_rgb(tpath2), closeup) if Path(tpath2).exists() else None
    if tseg is None:
        return []
    c_t2 = encode(t2.strip, t2.strip_mask, wavelength=WL) if t2 is not None else None
    hd_orig_vs_t2 = hamming(encode(tseg.strip, tseg.strip_mask, wavelength=WL), c_t2)[0] if c_t2 else np.nan
    rows = []
    tcls = Path(tpath).parent.name
    for cls, strip, mask in donors:
        if closeup and cls.startswith(tcls + "_"):
            continue  # donor must be a different person
        res, _, _ = embed_into_eye(rgb, tseg, IrisTexture(strip, mask), color=color, method=method, **opts)
        r = seg_of(res, closeup)
        if r is None:
            rows.append(dict(target=tpath, cls=cls, code=None))
            continue
        code = encode(r.strip, r.strip_mask, wavelength=WL)
        seam, sharp = realism(rgb, res, tseg)
        rows.append(dict(target=tpath, cls=cls, code=(code.bits, code.mask), diam=tseg.diameter_px,
                         seam=seam, sharp=sharp,
                         hd_res_vs_t2=hamming(code, c_t2)[0] if c_t2 else np.nan, hd_orig_vs_t2=hd_orig_vs_t2))
    return rows


def identify(code, gallery):
    """min HD per class -> (sorted classes, dict)."""
    best = {}
    for cls, gc in gallery:
        hd = hamming(code, gc)[0]
        if not np.isnan(hd) and hd < best.get(cls, 1.0):
            best[cls] = hd
    return sorted(best, key=best.get), best


def summarize(name, probes, gallery):
    from iris_pipeline.encode import IrisCode
    ranks, gen, imp = [], [], []
    for cls, code in probes:
        if not isinstance(code, IrisCode):
            code = IrisCode(*code)
        order, best = identify(code, gallery)
        ranks.append(order.index(cls) + 1 if cls in order else len(order) + 1)
        gen.append(best.get(cls, np.nan))
        imp += [v for k, v in best.items() if k != cls]
    ranks, gen, imp = np.array(ranks), np.array(gen), np.array(imp)
    e, t = eer(gen[~np.isnan(gen)], imp)
    return dict(set=name, probes=len(ranks), rank1=round(float((ranks == 1).mean()), 3),
                rank5=round(float((ranks <= 5).mean()), 3), EER=round(e, 3),
                genuine_HD=round(float(np.nanmean(gen)), 3), impostor_HD=round(float(imp.mean()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", choices=["casia", "ravdess", "utiris"], default="casia")
    ap.add_argument("--method", choices=["simple", "layered"], default="simple")
    ap.add_argument("--n-targets", type=int, default=10)
    ap.add_argument("--n-donors", type=int, default=40)
    ap.add_argument("--color", default="target", choices=["donor", "target", "nir", "auto"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-blur-match", action="store_true")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output JSON")
    ap.add_argument("--donor-cache", default="", help="strip cache for donors (D15 geometry A/B)")
    a = ap.parse_args()

    if a.donor_cache:
        baseline_utiris.CACHE = Path(a.donor_cache)
    items = load_cache()
    classes = sorted({it["cls"] for it in items})
    gallery = [(it["cls"], encode(it["strip"], it["mask"], wavelength=WL)) for it in items if it["cap"] != 1]
    rng = np.random.default_rng(0)
    pick = set(rng.choice(classes, size=min(a.n_donors, len(classes)), replace=False))
    donors = [(it["cls"], it["strip"], it["mask"]) for it in items if it["cap"] == 1 and it["cls"] in pick]
    print(f"gallery {len(gallery)} codes / {len(classes)} classes; {len(donors)} donors; targets={a.targets}")

    direct = summarize("direct (no embedding)", [(c, encode(s, m, wavelength=WL)) for c, s, m in donors], gallery)
    opts = dict(blur_match=not a.no_blur_match, add_noise=not a.no_noise) if a.method == "layered" else {}
    jobs = [(t, donors, a.color, a.method, a.targets == "utiris", opts) for t in target_list(a.targets, a.n_targets)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run_target, jobs) for r in rs]
    ok = [r for r in rows if r["code"] is not None]
    emb = summarize(f"embedded into {a.targets}", [(r["cls"], r["code"]) for r in ok], gallery)
    emb["redetected"] = f"{len(ok)}/{len(rows)}"
    emb["target_iris_diam_px"] = round(float(np.mean([r["diam"] for r in ok])), 1)
    rem = dict(HD_original_face_vs_target_other_capture=round(float(np.nanmean([r["hd_orig_vs_t2"] for r in ok])), 3),
               HD_edited_face_vs_target_other_capture=round(float(np.nanmean([r["hd_res_vs_t2"] for r in ok])), 3))
    emb["seam_ratio_median"] = round(float(np.median([r["seam"] for r in ok])), 2)
    emb["sharpness_ratio_median"] = round(float(np.median([r["sharp"] for r in ok])), 2)
    per_target = {}
    for t in sorted({r["target"] for r in ok}):
        sub = [(r["cls"], r["code"]) for r in ok if r["target"] == t]
        per_target[Path(t).parent.name + "/" + Path(t).name] = summarize("t", sub, gallery)["rank1"]
    print("per-target rank-1:", per_target)
    rep = dict(direct=direct, embedded=emb, target_removal=rem, per_target_rank1=per_target,
               wavelength=WL, color=a.color, method=a.method)
    print(json.dumps(rep, indent=2))
    suffix = ("" if a.method == "simple" else f"_{a.method}") + (f"_{a.tag}" if a.tag else "")
    Path(f"outputs/transfer_{a.targets}_{a.color}{suffix}.json").write_text(json.dumps(stamp(rep), indent=2))


if __name__ == "__main__":
    main()
