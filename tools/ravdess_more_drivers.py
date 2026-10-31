"""More driving clips per RAVDESS actor for the large session (batch 4, README.md).

Same preparation as tools/ravdess_prep.py (face-crop mp4 cut to 120 frames, eyes open in >= 60 % of the cut); the candidates are all other
audio-video clips of the actor with emotion neutral / calm / happy (both statements, both repetitions), taken in a seeded random order, so no
clip is picked by how well it later verifies. Existing drivers and the source clip are excluded.

    python tools/ravdess_more_drivers.py [--per-actor 8]   # -> outputs/ravdess_drivers_more.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import ravdess_prep as RP  # noqa: E402
from iris_pipeline.detect import FaceEyeDetector  # noqa: E402
from provenance import stamp  # noqa: E402

OUT = ROOT / "outputs" / "ravdess_drivers_more.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-actor", type=int, default=8)
    a = ap.parse_args()
    idx = json.loads(RP.INDEX.read_text())["targets"]
    old = json.loads(OUT.read_text()) if OUT.exists() else {"targets": []}
    done = {t["actor"]: t for t in old["targets"]}
    with FaceEyeDetector() as det:
        for t in idx:
            actor = t["actor"].split("_")[1]
            if not t["drivers"] or t["actor"] in done:
                continue
            rows = RP.rows_for(actor)
            base = lambda r: Path(r["filepath"]).name
            used = {Path(d["clip"]).name for d in t["drivers"]} | {Path(t["source"]["clip"]).name}
            cand = [r for r in rows if r["emotion_code"] in ("1", "2", "3") and base(r) not in used]
            rng = np.random.default_rng(1000 + int(actor))
            cand = [cand[i] for i in rng.permutation(len(cand))]
            zp = RP.ensure_zip(actor, False)
            d = RP.extract_av(actor, zp, [base(r) for r in cand])
            out, tried = [], []
            for r in cand:
                if len(out) >= a.per_actor:
                    break
                dv = RP.pick_driver(d / base(r), det)
                tried.append(dict(clip=r["filepath"], open_frac=None if dv is None else round(dv["open_frac"], 3)))
                if dv is None or dv["open_frac"] < RP.MIN_OPEN_FRAC:
                    continue
                mp4 = RP.DATA / f"Actor_{actor}" / f"driver_{Path(base(r)).stem}.mp4"
                RP.write_mp4(dv.pop("frames"), mp4, dv["fps"])
                out.append(dict(mp4=str(mp4.relative_to(ROOT)), clip=r["filepath"], emotion=r["emotion"], intensity=r["intensity"],
                                statement=r["statement"], repetition=int(r["repetition"]), **dv))
            done[t["actor"]] = dict(actor=t["actor"], drivers=out, drivers_tried=tried)
            print(t["actor"], len(out), "new drivers (tried", len(tried), ")", flush=True)
            OUT.write_text(json.dumps(stamp(dict(note="extra driving clips for batch 4; see the module docstring", targets=[done[k] for k in sorted(done)]), seed=None), indent=1))


if __name__ == "__main__":
    main()
