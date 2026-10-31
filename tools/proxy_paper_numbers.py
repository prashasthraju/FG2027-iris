"""Numbers for the pending tables of paper_needed.tex, computed from the caches and ledger of the proxy-iris test (docs/16).

  separation   AUC, EER, TAR at 1 % and 0.1 % FAR of the 1:1 distance (true driver's proxy = genuine, every other enrolled proxy = impostor), per extractor
  ablation     frames fused (first k analysed frames), extractor, false-accept target; thresholds re-read on the tuning sessions for every setting
  overhead     embedding ms per frame, seconds per session, verification time, record size, PDQ time
  imitation    the motion-matched sessions of batch 3 (same photo, same driving clip, a different authenticated driver)

    python tools/proxy_paper_numbers.py     # -> outputs/proxy_attr_paper_numbers.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402
import proxy_attr_lib as L  # noqa: E402
import proxy_issuer_eval as IE  # noqa: E402
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402

OUT = ROOT / "outputs" / "proxy_attr_paper_numbers.json"


def roc_stats(gen, imp):
    from sklearn.metrics import roc_auc_score, roc_curve
    gen, imp = np.asarray([g for g in gen if np.isfinite(g)]), np.asarray([x for x in imp if np.isfinite(x)])
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    s = -np.r_[gen, imp]                                            # smaller distance = more genuine
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs((1 - tpr) - fpr)))
    return dict(auc=round(100 * float(roc_auc_score(y, s)), 2), eer=round(100 * float((fpr[i] + 1 - tpr[i]) / 2), 2),
                tar_far1=round(100 * float(np.interp(1e-2, fpr, tpr)), 2), tar_far01=round(100 * float(np.interp(1e-3, fpr, tpr)), 2),
                n_genuine=int(len(gen)), n_impostor=int(len(imp)))


def main():
    plan = json.loads(V.PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(V.LEDGER)
    G = V.Gallery(led)
    didx = {n[0]: i for i, n in enumerate(G.names)}
    metas = [m for m in V.sessions() if m["kind"] in ("self", "cross")]
    caches = {m["id"]: V.load_cache(m["id"]) for m in metas}
    out = dict(note="computed from the extraction caches; thresholds are always read on the tuning sessions only")
    # ---------------------------------------------------------------- separation
    sep = {}
    vecs = {m["id"]: G.vectors(caches[m["id"]])[0] for m in metas if caches[m["id"]] is not None}
    for mt in V.MATCHERS:
        gen, imp = [], []
        for m in metas:
            if m["tuning"] or m["id"] not in vecs:
                continue
            d = vecs[m["id"]][mt]
            k = didx[enrol[m["driver_actor"]]["did"]]
            gen.append(float(d[k]))
            imp += [float(x) for x in np.delete(d, k) if np.isfinite(x)]
        sep[mt] = roc_stats(gen, imp)
    out["separation_test"] = sep

    # ---------------------------------------------------------------- ablation
    def accept_rates(mt, k_frames, far):
        old = L.MIN_OBS
        L.MIN_OBS = 1
        try:
            def dvec(m):
                c = caches[m["id"]]
                if k_frames:
                    c = dict(c, obs=[o for o in c["obs"] if o["frame"] < k_frames])
                return G.vectors(c)[0][mt]
            tun_imp = []
            for m in metas:
                if not m["tuning"] or caches[m["id"]] is None:
                    continue
                d = dvec(m)
                k = didx[enrol[m["driver_actor"]]["did"]]
                tun_imp += [float(x) for x in np.delete(d, k) if np.isfinite(x)]
            T = float(np.quantile(tun_imp, far))
            gen_ok = n = imp_ok = n_imp = 0
            for m in metas:
                if m["tuning"] or caches[m["id"]] is None:
                    continue
                d = dvec(m)
                k = didx[enrol[m["driver_actor"]]["did"]]
                n += 1
                gen_ok += int(np.isfinite(d[k]) and d[k] < T)
                o = np.delete(d, k)
                o = o[np.isfinite(o)]
                imp_ok += int((o < T).sum())
                n_imp += len(o)
            return dict(T=round(T, 4), genuine=V.rate(gen_ok, n), impostor=V.rate(imp_ok, n_imp))
        finally:
            L.MIN_OBS = old
    abl = {"frames_fused_arciris": {}, "extractor": {}, "far_target": {}}
    for kf in (1, 4, 16, 0):
        abl["frames_fused_arciris"]["all" if kf == 0 else str(kf)] = accept_rates("arciris", kf, 1e-2)
        print("frames", kf, abl["frames_fused_arciris"]["all" if kf == 0 else str(kf)]["genuine"], flush=True)
    for mt in V.MATCHERS:
        abl["extractor"][mt] = accept_rates(mt, 0, 1e-2)
    for far in (1e-2, 1e-3):
        abl["far_target"][f"{far:g}"] = accept_rates("arciris", 0, far)
    out["ablation"] = abl

    # ---------------------------------------------------------------- overhead
    allm = [m for m in V.sessions() if m["kind"] in ("self", "cross")]
    ms = {ds: [m["embed_ms_median"] for m in allm if m["dataset"] == ds] for ds in ("ravdess", "nir")}
    sec = {ds: [m["seconds"] / m["frames"] for m in allm if m["dataset"] == ds] for ds in ("ravdess", "nir")}
    ov = dict(embed_ms_per_frame_median={ds: round(float(np.median(v)), 0) for ds, v in ms.items()},
              session_seconds_per_frame_median={ds: round(float(np.median(v)), 2) for ds, v in sec.items()},
              frames_per_session_median=float(np.median([m["frames"] for m in allm])))
    rec = led.session(allm[0]["session_id"])["rec"]
    ov["record_bytes"] = len(PL.canon(rec).encode())
    t0 = time.time()
    from proxy_attr import pdq_hashes
    pdq_hashes(V.clip_path(allm[0]["id"]))
    ov["pdq_seconds_per_video"] = round(time.time() - t0, 1)
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    J = ArcIrisJudge()
    ts = []
    with FaceEyeDetector() as det:
        for m in allm[:3]:
            t0 = time.time()
            V.extract_key(m["id"], det, J)
            ts.append(time.time() - t0)
    ov["extraction_seconds_per_video_median"] = round(float(np.median(ts)), 1)
    issuer = IE.Issuer(plan)
    calib = json.loads(V.CALIB.read_text())
    cm, T1 = calib["claim_primary"], calib["claim"][calib["claim_primary"]]["T1"]
    ts = []
    for m in allm[:5]:
        t0 = time.time()
        IE.verify_session(led, issuer, G, caches[m["id"]], V.clip_path(m["id"]), m["session_id"], m["driver_actor"], issuer.real[m["driver_actor"]]["probes"][0], IE.acct(m["driver_actor"]), cm, T1)
        ts.append(time.time() - t0)
    ov["verify_seconds_after_extraction_median"] = round(float(np.median(ts)), 1)
    out["overhead"] = ov

    # ---------------------------------------------------------------- imitation (batch 3)
    imit = []
    for f in sorted(V.SESS.glob("I_*.json")):
        m = json.loads(f.read_text())
        if m.get("status") != "ok":
            continue
        c = V.load_cache(m["id"])
        if c is None:
            continue
        path = V.clip_path(m["id"])
        x, d_ = m["driver_actor"], m["imitates"]
        v_x = IE.verify_session(led, issuer, G, c, path, m["session_id"], x, issuer.real[x]["probes"][1], IE.acct(x), cm, T1)
        v_d = IE.verify_session(led, issuer, G, c, path, m["session_id"], d_, issuer.real[d_]["probes"][1], IE.acct(d_), cm, T1)
        o = json.loads((V.SESS / f"{m['orig_session']}.json").read_text())
        v_o = IE.verify_session(led, issuer, G, V.load_cache(o["id"]), V.clip_path(o["id"]), o["session_id"], d_, issuer.real[d_]["probes"][1], IE.acct(d_), cm, T1)
        imit.append(dict(id=m["id"], attacker=x, imitated=d_, original_session=o["id"], same_photo_and_motion=bool(m["source"] == o["source"] and m["driver"] == o["driver"]),
                         attacker_claim=v_x["outcome"], attacker_attributed_correctly=bool(v_x["outcome"] == "CROSS" and v_x.get("driver") == enrol[x]["did"]),
                         imitated_persons_claim=v_d["outcome"], original_video_imitated_person=v_o["outcome"]))
    from collections import Counter
    out["imitation"] = dict(n=len(imit), attacker_attributed=V.rate(sum(r["attacker_attributed_correctly"] for r in imit), len(imit)),
                            imitated_person_claim_outcomes=dict(Counter(r["imitated_persons_claim"] for r in imit)),
                            imitated_person_accepted=V.rate(sum(r["imitated_persons_claim"] in ("SELF", "CROSS") for r in imit), len(imit)),
                            original_sessions_outcomes=dict(Counter(r["original_video_imitated_person"] for r in imit)), rows=imit)
    OUT.write_text(json.dumps(stamp(out), indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "imitation"}, indent=1)[:5000])
    print(json.dumps({k: v for k, v in out["imitation"].items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
