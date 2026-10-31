"""Numbers that answer the referee report (docs/Referee Report ...): what the pixels add over the signed record, how the three distance
figures of the paper relate, intervals, and how far apart the pool codes are.

  record_only   verdicts of a verifier that stops after steps 1-3 (record, hashes, claimant key + live iris) and never looks at the eyes,
                derived from the stored verdicts of the full verifier (a row that reached step 4 passed steps 1-3)
  operating     distance-only and end-to-end acceptance at the frozen T1, the ROC-interpolated TAR@1 % of Table II and the re-read threshold of the ablation
  bootstrap     session-level bootstrap intervals for AUC, EER, TAR@1 %
  pool          pairwise ArcIris distances of the 79 pool codes, and genuine / impostor distances after generation and after CRF 23
  false_claims  the 623 false claims by the step that refused them

    python tools/proxy_referee_numbers.py     # -> outputs/proxy_attr_referee_numbers.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402
import proxy_attr_lib as L  # noqa: E402
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402
from proxy_paper_numbers import roc_stats  # noqa: E402

OUT = ROOT / "outputs" / "proxy_attr_referee_numbers.json"
PASSED_STEPS_1_3 = ("SELF", "CROSS", "FLAGGED", "INCONCLUSIVE")


def rate(k, n):
    return V.rate(int(k), int(n))


def main():
    R = json.loads((ROOT / "outputs" / "proxy_attr_issuer_results.json").read_text())
    rows = [r for r in R["rows"] if not r["tuning"] and r.get("generator", "liveportrait") == "liveportrait"]      # the other generators are reported in their own table
    out = dict(note="derived from the stored verdicts and the extraction caches; see the module docstring")

    # ------------------------------------------------------------------ record-only verifier
    def ro(sel):
        sel = list(sel)
        # key + record: the claimant holds the right key but no live iris is checked, so a failed live-iris match (CLAIM_FAILED of an honest claimant) does not stop it
        return dict(n=len(sel), key_record_attributed=rate(sum(r["outcome"] in PASSED_STEPS_1_3 or r["outcome"] == "CLAIM_FAILED" for r in sel), len(sel)),
                    record_only_attributed=rate(sum(r["outcome"] in PASSED_STEPS_1_3 for r in sel), len(sel)),
                    full_attributed=rate(sum(r["outcome"] in ("SELF", "CROSS") for r in sel), len(sel)))
    orig = [r for r in rows if r["variant"] == "original"]
    out["record_only"] = {
        "honest": ro(r for r in orig if r["kind"] in ("self", "cross")),
        "twin_without_hook": ro(r for r in rows if r["variant"] == "raw_twin" and r["kind"] in ("self", "cross")),
        "wrong_proxy_embedded": ro(r for r in orig if r["kind"] == "wrong_proxy"),
        "unenrolled_proxy_embedded": ro(r for r in orig if r["kind"] == "unenrolled_proxy"),
        "crf23": ro(r for r in rows if r["variant"] == "crf23"), "crf30": ro(r for r in rows if r["variant"] == "crf30"),
        "blur": ro(r for r in rows if r["variant"] == "blur"), "alter": ro(r for r in rows if r["variant"] == "alter"),
    }
    # false claims and imitation: refusals by step
    steps = Counter()
    for c in R["claims"]:
        if c["tuning"]:
            continue
        step = {"the claimant is not the record's driver": "record names another driver (step 3)",
                "the signature does not match the key published in the DID": "key check (step 3)",
                "the live iris does not match the iris enrolled for this DID": "live-iris check at the issuer (step 3)"}.get(c["reason"], c["reason"])
        steps[(c["claim"], step)] += 1
    # what each verifier arm accepts among the false claims: only the claim made with the driver's key and ANOTHER PERSON'S IRIS passes a key-only check
    kinds = Counter(c["claim"] for c in R["claims"] if not c["tuning"])
    stolen = "driver's key, another person's iris"
    out["claim_arms"] = dict(stolen_key_n=kinds.get(stolen, 0), other_false_claims_n=sum(v for k, v in kinds.items() if k != stolen),
                             stolen_key_accepted=dict(key_record=kinds.get(stolen, 0),
                                                      live_iris=sum(1 for c in R["claims"] if not c["tuning"] and c["claim"] == stolen and c["outcome"] in ("SELF", "CROSS")),
                                                      full=sum(1 for c in R["claims"] if not c["tuning"] and c["claim"] == stolen and c["outcome"] in ("SELF", "CROSS"))),
                             other_false_accepted=dict(key_record=0, live_iris=0, full=sum(1 for c in R["claims"] if not c["tuning"] and c["claim"] != stolen and c["outcome"] in ("SELF", "CROSS"))))
    out["false_claims_by_step"] = [dict(claim=k[0], refused_at=k[1], n=v) for k, v in sorted(steps.items())]
    out["false_claims_total"] = sum(steps.values())
    out["false_claims_refused_by_distance_step"] = 0

    # ------------------------------------------------------------------ operating points
    plan = json.loads(V.PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(V.LEDGER)
    G = V.Gallery(led)
    didx = {n[0]: i for i, n in enumerate(G.names)}
    calib = json.loads(V.CALIB.read_text())
    T1 = calib["claim"]["arciris"]["T1"]
    metas = [m for m in V.sessions() if m["kind"] in ("self", "cross") and not m["tuning"]]
    vec = {}
    for m in metas:
        c = V.load_cache(m["id"])
        if c is not None:
            vec[m["id"]] = G.vectors(c)[0]["arciris"]
    gen, imp, per, d_only, e2e = [], [], [], 0, 0
    for m in metas:
        d = vec.get(m["id"])
        if d is None:
            continue
        k = didx[enrol[m["driver_actor"]]["did"]]
        s_idx = didx.get(enrol.get(m["source_actor"], {}).get("did"))
        ok1 = bool(np.isfinite(d[k]) and d[k] < T1)
        ok2 = ok1 and (m["kind"] == "self" or s_idx is None or not np.isfinite(d[s_idx]) or d[k] < d[s_idx])
        d_only += ok1
        e2e += ok2
        others = np.delete(d, k)
        others = others[np.isfinite(others)]
        gen.append(float(d[k]))
        imp += [float(x) for x in others]
        per.append((float(d[k]), others))
    n = len(per)
    out["operating"] = dict(
        T1_frozen=T1, n_sessions=n,
        distance_only_accepted_at_T1=rate(d_only, n), distance_and_second_condition_at_T1=rate(e2e, n),
        impostor_accepted_at_T1=rate(sum(int((o < T1).sum()) for _, o in per), sum(len(o) for _, o in per)),
        roc_interpolated=roc_stats(gen, imp),
        genuine_distance=dict(median=float(np.median(gen)), p90=float(np.quantile(gen, 0.9))),
        impostor_distance=dict(median=float(np.median(imp)), p1=float(np.quantile(imp, 0.01)), min=float(np.min(imp))),
        explanation=("Table II reads TAR@1% from the ROC at the threshold that gives 1% false accepts on the test impostors themselves (an oracle operating point); "
                     "Table VI re-reads the threshold on the tuning impostors for each setting and uses MIN_OBS=1; the end-to-end verdict uses the frozen T1 and MIN_OBS=5"))

    # ------------------------------------------------------------------ bootstrap over sessions
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(1000):
        pick = rng.integers(0, n, n)
        g = [per[i][0] for i in pick]
        im = [x for i in pick for x in per[i][1]]
        st = roc_stats(g, im)
        stats.append((st["auc"], st["eer"], st["tar_far1"]))
    a = np.array(stats)
    out["bootstrap_arciris_1000"] = {k: [round(float(np.quantile(a[:, i], 0.025)), 2), round(float(np.quantile(a[:, i], 0.975)), 2)] for i, k in enumerate(("auc", "eer", "tar_far1"))}

    # ------------------------------------------------------------------ pool separation
    from cancelable.pool import IrisPool, SIDE_OF
    from iris_pipeline.judge import ArcIrisJudge
    import cv2
    pool = IrisPool()
    J = ArcIrisJudge()
    vecs = {}
    for sid in pool.ids:
        for eye, side in SIDE_OF.items():
            rgb = cv2.cvtColor(cv2.imread(str(pool.files[sid][eye][1])), cv2.COLOR_BGR2RGB)
            v, info = J.encode_image(rgb)
            if not info["rejected"]:
                vecs.setdefault(sid, {})[side] = np.asarray(v, np.float32)
    ids = [s for s in pool.ids if s in vecs]
    pw = []
    for i, a_ in enumerate(ids):
        for b_ in ids[i + 1:]:
            ds = [J.distance(vecs[a_][s], vecs[b_][s]) for s in vecs[a_] if s in vecs[b_]]
            if ds:
                pw.append(float(np.mean(ds)))
    pw = np.array(pw)
    out["pool_pairwise_arciris"] = dict(n_codes=len(ids), n_pairs=int(len(pw)), min=round(float(pw.min()), 3), p1=round(float(np.quantile(pw, 0.01)), 3),
                                        median=round(float(np.median(pw)), 3),
                                        note="clean image to clean image, different people; not comparable with T1, which is calibrated on distances from generated video")
    # matched units: same eye, another capture of it (clean images), against the pairwise distances above
    same = []
    for sid in ids:
        for eye, side in SIDE_OF.items():
            caps = pool.files[sid][eye]
            for cap, path in sorted(caps.items()):
                if cap == 1 or side not in vecs[sid]:
                    continue
                v, info = J.encode_image(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
                if not info["rejected"]:
                    same.append(J.distance(np.asarray(v, np.float32), vecs[sid][side]))
    same = np.array(same)
    out["pool_clean_same_eye_other_capture"] = dict(n=int(len(same)), median=round(float(np.median(same)), 3), p90=round(float(np.quantile(same, 0.9)), 3),
                                                    max=round(float(same.max()), 3),
                                                    clean_pairs_closer_than_p90_of_same_eye=int((pw < np.quantile(same, 0.9)).sum()))
    vp = json.loads((V.WORK / "variant_plan.json").read_text())
    allowed = {"__crf23": set(vp["crf23"]), "__crf30": set(vp["crf30"])}

    def sep(key_suffix):
        g2, i2 = [], []
        for m in metas:
            if key_suffix in allowed and m["id"] not in allowed[key_suffix]:        # stale variant files of an earlier draw are not part of the plan
                continue
            c = V.load_cache(m["id"] + key_suffix)
            if c is None:
                continue
            d = G.vectors(c)[0]["arciris"]
            k = didx[enrol[m["driver_actor"]]["did"]]
            if np.isfinite(d[k]):
                g2.append(float(d[k]))
            i2 += [float(x) for x in np.delete(d, k) if np.isfinite(x)]
        return (roc_stats(g2, i2) if g2 else None), dict(n_sessions_with_code=len(g2), genuine_median=float(np.median(g2)) if g2 else None,
                                                         impostor_p1=float(np.quantile(i2, 0.01)) if i2 else None)

    # rank-1 identification accuracy of the embedded proxy among all enrolled proxies (ArcIris mean angular distance); an analysis from the issuer's registry,
    # the verifier itself does a 1:1 match. Sessions without a usable iris code count as misses.
    k_ok = 0
    for m in metas:
        c = V.load_cache(m["id"])
        if c is None:
            continue
        d = G.vectors(c)[0]["arciris"]
        if np.isfinite(d).any():
            k_ok += int(np.nanargmin(d) == didx[enrol[m["driver_actor"]]["did"]])
    out["rank1"] = rate(k_ok, len(metas))
    out["after_generation"] = dict(zip(("roc", "dist"), sep("")))
    out["after_crf23"] = dict(zip(("roc", "dist"), sep("__crf23")))
    OUT.write_text(json.dumps(stamp(out), indent=1))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
