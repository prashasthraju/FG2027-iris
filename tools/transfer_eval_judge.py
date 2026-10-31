# LEGACY: part of an earlier design (public proxy codes / 1:N identification of the proxy in the video / local Anvil chain). The submitted protocol
# (commit-and-reveal, 1:1 check against the opened proxy) is implemented in tools/proxy_issuer_eval.py (verify_session) and tools/proxy_ledger.py; see README.md.
"""S0 (D11 P3): re-score identity transfer with the HELD-OUT judge.

`transfer_eval.py` embeds a donor and then identifies the edited face with our
own IrisCode, which unwraps using the same geometry model we used to wrap. That
flatters the result (research/03 ss0). Here the edited image is handed to ArcIris,
which runs its own segmentation, its own normalisation and its own encoder, and
shares nothing with our pipeline.

Three scores per (target, donor) pair, all from ArcIris:
  - rank-1 against a gallery of the donors' OTHER captures (identity transfer)
  - distance to the donor's other captures (genuine)
  - distance to the target's other capture (removal)
Plus the matcher-disjointness gap: our rank-1 minus ArcIris rank-1 on the same
pairs. research/03 calls a gap above ~5 pp evidence of matcher-specific gains.

    python tools/transfer_eval_judge.py --targets casia --method layered --n-targets 10 --n-donors 40
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import stamp
from baseline import eer
from iris_pipeline.detect import FaceEyeDetector
from iris_pipeline.embed import IrisTexture, embed_into_eye
from iris_pipeline.encode import encode, hamming
from iris_pipeline.segment import extract_irises, segment_closeup
from transfer_eval import CASIA, SIDE, UTIRIS, WL, target_list

VIS_ROOT = "dbs/UTIRIS_V.1/RGB Images"


def load_rgb(p):
    return np.ascontiguousarray(cv2.imread(str(p))[:, :, ::-1])


def donor_files(n_donors, seed=0):
    """Capture 1 = the donor we embed; captures 2..5 = the ArcIris gallery."""
    by_cls = {}
    for p in sorted(Path(VIS_ROOT).rglob("*")):
        if p.suffix.lower() != ".jpg" or p.name.startswith("._"):
            continue
        _, sid, eye, cap = p.stem.split("_")
        by_cls.setdefault(f"{sid}_{eye}", {})[int(cap)] = p
    classes = sorted(k for k, v in by_cls.items() if 1 in v and len(v) > 1)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(classes, size=min(n_donors, len(classes)), replace=False))
    return by_cls, pick, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", choices=["casia", "utiris", "ravdess"], default="casia")
    ap.add_argument("--method", choices=["simple", "layered"], default="layered")
    ap.add_argument("--color", default="auto")
    ap.add_argument("--n-targets", type=int, default=10)
    ap.add_argument("--n-donors", type=int, default=40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--crop-from", choices=["redetect", "embed"], default="redetect",
                    help="centre the judge crop on the re-detected iris or on the pre-edit circle")
    a = ap.parse_args()

    from iris_pipeline.judge import ArcIrisJudge
    J = ArcIrisJudge()
    det = FaceEyeDetector()
    by_cls, pick, classes = donor_files(a.n_donors)

    # ---- ArcIris gallery: every class's captures 2..5, ArcIris segmentation
    gallery = []
    for cls in classes:
        for cap, p in sorted(by_cls[cls].items()):
            if cap == 1:
                continue
            v, info = J.encode_image(load_rgb(p))
            if not info["rejected"]:
                gallery.append((cls, v))
    print(f"ArcIris gallery: {len(gallery)} vectors / {len({c for c, _ in gallery})} classes")

    # ---- donors: our segmentation (that is what we embed), plus ArcIris on the same file
    donors = []
    for cls in pick:
        p = by_cls[cls][1]
        rgb = load_rgb(p)
        try:
            seg = segment_closeup(rgb)
        except ValueError:
            continue
        donors.append(dict(cls=cls, tex=IrisTexture.from_segmentation(seg),
                           our_code=encode(seg.strip, seg.strip_mask, wavelength=WL)))
    print(f"donors: {len(donors)}")

    closeup = a.targets == "utiris"
    rows = []
    for tpath, tpath2 in target_list(a.targets, a.n_targets):
        rgb = load_rgb(tpath)
        tseg = (segment_closeup(rgb) if closeup
                else {s.side: s for s in extract_irises(rgb, det)}.get(SIDE))
        if tseg is None:
            continue
        tcls = Path(tpath).parent.name
        # crop the eye before handing a FACE to ArcIris (see judge.eye_crop)
        v_t2 = None
        if Path(tpath2).exists():
            rgb2 = load_rgb(tpath2)
            s2 = (segment_closeup(rgb2) if closeup
                  else {s.side: s for s in extract_irises(rgb2, det)}.get(SIDE))
            if s2 is not None:
                v_t2, _ = J.encode_image(J.eye_crop(rgb2, s2.iris) if not closeup else rgb2)
        for d in donors:
            if closeup and d["cls"].startswith(tcls + "_"):
                continue
            res, _, _ = embed_into_eye(rgb, tseg, d["tex"], color=a.color, method=a.method)
            # --- re-detect the edited image from scratch: never reuse the embed geometry
            r = (segment_closeup(res) if closeup
                 else {s.side: s for s in extract_irises(res, det)}.get(SIDE))
            # --- held-out judge on the edited image. The eye crop is centred on the RE-DETECTED iris;
            #     `--crop-from embed` centres it on the pre-edit circle (the old behaviour, kept only
            #     to reproduce outputs/transfer_judge_casia_layered_crop.json).
            if closeup:
                probe = res
            else:
                centre = tseg.iris if a.crop_from == "embed" else (r.iris if r is not None else None)
                probe = J.eye_crop(res, centre) if centre is not None else None
            v, info = J.encode_image(probe) if probe is not None else (None, dict(rejected=True))
            # --- our own matcher on the same edited image, for the gap
            our_hd = (hamming(encode(r.strip, r.strip_mask, wavelength=WL), d["our_code"])[0]
                      if r is not None else np.nan)
            rows.append(dict(target=str(tpath), cls=d["cls"], v=v, rejected=info["rejected"],
                             our_hd=our_hd, v_t2=v_t2))
        print(f"  {Path(tpath).name}: {len(rows)} pairs", flush=True)

    ok = [r for r in rows if r["v"] is not None]
    ranks, gen, imp, removal = [], [], [], []
    for r in ok:
        best = {}
        for cls, gv in gallery:
            dist = J.distance(r["v"], gv)
            if dist < best.get(cls, np.inf):
                best[cls] = dist
        order = sorted(best, key=best.get)
        ranks.append(order.index(r["cls"]) + 1 if r["cls"] in order else len(order) + 1)
        gen.append(best.get(r["cls"], np.nan))
        imp += [v for k, v in best.items() if k != r["cls"]]
        if r["v_t2"] is not None:
            removal.append(J.distance(r["v"], r["v_t2"]))
    ranks, gen, imp = np.array(ranks), np.array(gen, float), np.array(imp, float)
    e, _ = eer(gen[~np.isnan(gen)], np.array(imp))

    our_hd = np.array([r["our_hd"] for r in ok], float)
    rep = dict(
        targets=a.targets, method=a.method, color=a.color, crop_from=a.crop_from,
        pairs=len(rows), judged=len(ok), rejected_by_judge=len(rows) - len(ok),
        gallery_classes=len({c for c, _ in gallery}),
        judge_rank1=round(float((ranks == 1).mean()), 3),
        judge_rank5=round(float((ranks <= 5).mean()), 3),
        judge_EER=round(float(e), 3),
        judge_genuine_dist=round(float(np.nanmean(gen)), 3),
        judge_impostor_dist=round(float(imp.mean()), 3),
        judge_removal_dist=(round(float(np.nanmean(removal)), 3) if removal else None),
        our_HD_result_vs_donor=round(float(np.nanmean(our_hd)), 3),
        chance_rank1=round(1 / max(len({c for c, _ in gallery}), 1), 4),
    )
    print(json.dumps(rep, indent=2))
    out = Path(f"outputs/transfer_judge_{a.targets}_{a.method}{'_' + a.tag if a.tag else ''}.json")
    out.write_text(json.dumps(stamp(rep), indent=2))
    print("wrote", out)
    det.close()


if __name__ == "__main__":
    main()
