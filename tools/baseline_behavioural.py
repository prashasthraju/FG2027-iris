"""Behavioural baselines for paper_needed.tex: AUC / EER / TAR@FAR from the TrustFA-FG2026 `avafp` score files.

The scores are produced by `python -m avafp.evaluate <model> --datasets RAVDESS --gens LIVE` in
external/TrustFA-FG2026 (official AVAPrintDB TEST pairs, genuine = same driver, impostor = other driver
wearing the same face). This tool only re-reads those CSVs, so the numbers can be traced to one file each.

    python tools/baseline_behavioural.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402

RESULTS = Path("external/TrustFA-FG2026/avafp/results")
DATASET, GEN = "RAVDESS", "LIVE"
MODELS = {   # name -> role in the paper's tables
    "trustfa_nvfair": "Avatar fingerprinting family (TrustFA FEAT_DIFF retrained on NVFAIR; original weights unreleased)",
    "graph_ravdess_liveportrait": "Motion GCN, official AVAPrintDB checkpoint, trained on RAVDESS/LivePortrait (in-domain)",
    "graph_cremad_liveportrait": "Motion GCN, official checkpoint, trained on CREMA-D/LivePortrait (cross-dataset)",
    "dino_ravdess_liveportrait": "DINOv2 features, official checkpoint (in-domain)",
    "clip_ravdess_liveportrait": "CLIP features, official checkpoint (in-domain)",
    "fusion_ravdess_liveportrait": "Score fusion (z-normalised mean) of the in-domain graph, DINOv2 and CLIP checkpoints",
}


def tar_at(fpr, tpr, far):
    return float(np.interp(far, fpr, tpr))


def row(model):
    f = RESULTS / model / "scores" / f"scores_{DATASET}_{GEN}.csv"
    d = pd.read_csv(f)
    y, s = d.label.values, d.score.values
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    ci = json.loads((RESULTS / model / "results.json").read_text())["results"][DATASET][GEN]["auc_ci"]
    return dict(auc=round(100 * roc_auc_score(y, s), 2), auc_ci95=[round(c, 2) for c in ci], eer=round(100 * (fpr[i] + fnr[i]) / 2, 2),
                tar_far1pct=round(100 * tar_at(fpr, tpr, 1e-2), 2), tar_far0p1pct=round(100 * tar_at(fpr, tpr, 1e-3), 2),
                n_pairs=int(len(d)), n_genuine=int((y == 1).sum()), score_file=str(f))


def main():
    out = {m: dict(role=r, **row(m)) for m, r in MODELS.items()}
    note = (f"AVAPrintDB official TEST pairs, {DATASET} source, {GEN} (LivePortrait) avatars; higher AUC / lower EER is better. "
            "Pair protocol: two videos, same driver or not. Not directly comparable with our 1:N claim-verification protocol.")
    (ROOT / "outputs" / "baseline_behavioural.json").write_text(json.dumps(stamp(dict(note=note, rows=out)), indent=1))
    for m, r in out.items():
        print(f"{m:30s} AUC {r['auc']:6.2f} {r['auc_ci95']}  EER {r['eer']:6.2f}  TAR@1% {r['tar_far1pct']:6.2f}  TAR@0.1% {r['tar_far0p1pct']:6.2f}  n={r['n_pairs']}")


if __name__ == "__main__":
    main()
