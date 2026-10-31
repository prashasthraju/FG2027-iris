"""Self vs cross reenactment through the DID-mapped pool iris, following paper_needed.tex (the only paper).

Chain :
  ENROL   the DRIVING person D's real iris is captured and tied to ONE random pool iris E(D); the mapping lives in the session
          record of D's DID (here: outputs/idmap_registry.json, fake DIDs; on a ledger once the wallet arrives).
  GENERATE the platform writes the pool iris E(D) inline into every frame of the clip that shows the TARGET T's face.
  VERIFY  from the clip alone: (1) retrieve the DID (here: identify the carried pool entry from the video, ArcIris, z < zeta;
          a VideoSeal payload is the second channel for the same DID), (2) use the DID to fetch the driver's ORIGINAL iris
          (D's enrolled captures), (3) compare it with the TARGET's iris (the iris of T's source face image):
          same person -> SELF-reenactment, different -> CROSS-reenactment. A clip whose DID cannot be retrieved is UNKNOWN.
Real irises = CASIA-Iris-Distance faces (iris ~170 px). Source image of T = capture D00; D's enrolled template = captures D05..D09;
the real-iris decision threshold theta is read on tuning subjects 000-039 at 1 % false 'self' on impostor pairs and frozen.

    python tools/idmap_eval.py plan                 # -> outputs/heldout_plan_idmap.json + outputs/idmap_registry.json
    python tools/heldout_eval.py run --plan outputs/heldout_plan_idmap.json --tag idmap --keep-video
    python tools/idmap_eval.py score               # -> outputs/idmap_scores.json
    python tools/idmap_eval.py report
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402

CASIA = ROOT / "dbs" / "archive (3)" / "CASIA-Iris-Distance"
PLAN = ROOT / "outputs" / "heldout_plan_idmap.json"
REGISTRY = ROOT / "outputs" / "idmap_registry.json"
SCORES = ROOT / "outputs" / "idmap_scores.json"
CLIPS = ROOT / "outputs" / "heldout_idmap" / "clips"
TEMPLATE_CAPS = (5, 6, 7, 8, 9)          # D05..D09 = the driver's enrolled iris captures
SOURCE_CAP = 0                            # D00 = the face image the avatar is made from (the target's iris is read here)
DRIVERS = ("d10", "d14", "d9")
TUNE_SUBJECTS = [f"{i:03d}" for i in range(40)]


def img_path(subject, cap):
    return CASIA / subject / f"S4{subject}D{cap:02d}.jpg"


def plan(a):
    from cancelable.pool import IrisPool
    rng = np.random.default_rng(a.seed)
    pool_ids = IrisPool().ids
    subjects = sorted(p.name for p in CASIA.iterdir() if p.is_dir() and int(p.name) >= 93)
    pop = sorted(rng.choice(subjects, size=a.n_people, replace=False).tolist())
    entries = [str(e) for e in rng.choice(pool_ids, size=len(pop), replace=False)]        # injective random mapping driver -> pool iris
    registry = {}
    for s, e in zip(pop, entries):
        salt = hashlib.sha256(f"{a.seed}|{s}|salt".encode()).hexdigest()[:32]
        did = "did:exp:" + hashlib.sha256(f"{a.seed}|{s}|did".encode()).hexdigest()[:40]       # fake DID, no keys
        registry[did] = dict(driver_identity=s, pool_entry=e, session=f"sess-{s}-{a.seed}",
                             commitment="keccak256('iris-pool/v1|'||entry||salt) with salt " + salt[:6] + "...",
                             template=[str(img_path(s, c).relative_to(ROOT)) for c in TEMPLATE_CAPS])
    did_of = {v["driver_identity"]: d for d, v in registry.items()}
    rows, k = [], 0
    for i, t in enumerate(pop):
        others = [x for x in pop if x != t]
        picks = [(t, "self")] + [(str(x), "cross") for x in rng.choice(others, size=2, replace=False)]
        for d, cond in picks:
            rows.append(dict(id=f"{t}_by_{d}", subject=t, source=str(img_path(t, SOURCE_CAP).relative_to(ROOT)),
                             driver=DRIVERS[k % len(DRIVERS)], driver_label=cond, owner=t, driver_identity=d, condition=cond,
                             entry=registry[did_of[d]]["pool_entry"], did=did_of[d], group="idmap", frames=a.frames, scale=1,
                             model="liveportrait", dataset="casia_idmap"))
            k += 1
    out = dict(seed=a.seed, tau=0.97, every=2, frames=a.frames, dataset="casia_idmap", rows=rows,
               rule="enrol driver -> random pool iris + DID session record; inline-embed the pool iris into the target's clip; "
                    "verify: clip -> DID -> driver's enrolled iris vs the target's source iris -> self / cross / unknown")
    PLAN.write_text(json.dumps(stamp(out, seed=a.seed), indent=1))
    REGISTRY.write_text(json.dumps(dict(note="synthetic DIDs and salts for the experiment; no keys; not a ledger", registry=registry), indent=1))
    print(f"planned {len(rows)} clips ({sum(r['condition'] == 'self' for r in rows)} self, {sum(r['condition'] == 'cross' for r in rows)} cross), "
          f"{len(pop)} people -> {PLAN}")


# ------------------------------------------------------------------ real-iris comparison
def eye_vectors(img_file, det, J):
    """{side: ArcIris vector} of one CASIA face image (MediaPipe eye crop -> ArcIris, as identify.py does)."""
    import cv2
    from iris_pipeline.segment import extract_irises
    rgb = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)
    out = {}
    for seg in extract_irises(rgb, det):
        if seg.side in ("image_left", "image_right"):
            v, info = J.encode_image(J.eye_crop(rgb, seg.iris))
            if v is not None:
                out[seg.side] = v
    return out


def iris_distance(probe, template_list, J):
    """Mean over sides of the min distance to the template captures (lower = same person)."""
    ds = []
    for side, v in probe.items():
        cand = [J.distance(v, t[side]) for t in template_list if side in t]
        cand = [c for c in cand if np.isfinite(c)]
        if cand:
            ds.append(min(cand))
    return float(np.mean(ds)) if ds else float("nan")


def tune_theta(det, J, far=0.01):
    """theta such that `far` of impostor (different person) pairs would be called self; read on tuning subjects 000-039."""
    probes, temps = {}, {}
    for s in TUNE_SUBJECTS:
        if not img_path(s, SOURCE_CAP).exists():
            continue
        probes[s] = eye_vectors(img_path(s, SOURCE_CAP), det, J)
        temps[s] = [eye_vectors(img_path(s, c), det, J) for c in TEMPLATE_CAPS if img_path(s, c).exists()]
    gen = [iris_distance(probes[s], temps[s], J) for s in probes]
    imp = [iris_distance(probes[s], temps[o], J) for s in probes for o in temps if o != s]
    gen, imp = np.array([g for g in gen if np.isfinite(g)]), np.array([x for x in imp if np.isfinite(x)])
    theta = float(np.quantile(imp, far))
    return theta, dict(n_genuine=len(gen), n_impostor=len(imp), genuine_accepted=float((gen < theta).mean()),
                       genuine_median=float(np.median(gen)), impostor_median=float(np.median(imp)))


def score(a):
    from cancelable.pool import IrisPool
    from cancelable_demo import read_frames
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    from verify_1to1 import fused_vector
    plan_ = json.loads(PLAN.read_text())
    reg = json.loads(REGISTRY.read_text())["registry"]
    J, pool = ArcIrisJudge(), IrisPool()
    gallery, ids = pool.gallery(J), pool.ids
    done = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    with FaceEyeDetector() as det:
        if "theta" not in done:
            theta, info = tune_theta(det, J, a.far)
            done.update(theta=theta, theta_info=info, far=a.far)
            SCORES.write_text(json.dumps(done))
            print("theta", theta, info, flush=True)
        rows = done.setdefault("rows", {})
        tmpl = {}
        for did, v in reg.items():
            tmpl[did] = [eye_vectors(ROOT / f, det, J) for f in v["template"]]
        src = {}
        for r in plan_["rows"]:
            if r["id"] in rows:
                continue
            d = CLIPS / r["id"]
            stem = Path(r["source"]).stem
            emb, raw = d / f"liveportrait__{stem}__inline_{r['entry']}.mp4", d / f"liveportrait__{stem}__inline_raw.mp4"
            if not (emb.exists() and raw.exists()):
                continue
            fe, _ = read_frames(emb, every=plan_["every"])
            fr, _ = read_frames(raw, every=plan_["every"])
            if r["owner"] not in src:
                src[r["owner"]] = eye_vectors(ROOT / r["source"], det, J)
            import hashlib as _h
            raw_hash = _h.sha256(b''.join(f.tobytes() for f in fr)).hexdigest()[:16]
            rows[r["id"]] = dict(raw_hash=raw_hash, owner=r["owner"], driver_identity=r["driver_identity"], condition=r["condition"], did=r["did"],
                                 entry=r["entry"], embedded=fused_vector(fe, det, J, gallery, ids),
                                 control=fused_vector(fr, det, J, gallery, ids),
                                 gated_embedded=fused_vector(fe, det, J, gallery, ids, True),
                                 iris_true_did=iris_distance(src[r["owner"]], tmpl[r["did"]], J),
                                 iris_all=[iris_distance(src[r["owner"]], tmpl[dd], J) for dd in reg])
            done["ids"], done["dids"] = ids, list(reg)
            print(r["id"], flush=True)
            SCORES.write_text(json.dumps(done))
    SCORES.write_text(json.dumps(stamp(done)))


def zvec(d):
    d = np.asarray(d, float)
    med = np.nanmedian(d)
    return (d - med) / (1.4826 * np.nanmedian(np.abs(d - med)) + 1e-9)


def cp(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def rate(k, n):
    return dict(k=int(k), n=int(n), rate=round(k / n, 3) if n else None, ci95=cp(int(k), int(n)))


def eer_from_scores(pos, neg):
    """pos = scores that should be LOW (self), neg = scores that should be HIGH (cross). EER by threshold sweep."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ths = np.sort(np.unique(np.concatenate([pos, neg])))
    best = (1.0, 0.0)
    for t in ths:
        frr = float((pos > t).mean())         # self rejected
        far = float((neg <= t).mean())        # cross accepted as self
        if abs(frr - far) < abs(best[0] - best[1]) or (abs(frr - far) == abs(best[0] - best[1]) and max(frr, far) < max(best)):
            best = (frr, far)
    return 100 * (best[0] + best[1]) / 2


def auc_from_scores(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(p < n) + 0.5 * (p == n) for p in pos for n in neg]))


def report(a):
    zeta = json.loads((ROOT / "outputs" / "verify1to1_report.json").read_text())["sets"]["nir_v1"]["far_0.01"]["zeta"]
    d = json.loads(SCORES.read_text())
    reg = json.loads(REGISTRY.read_text())["registry"]
    dids = list(reg)
    entry_to_did = {v["pool_entry"]: k for k, v in reg.items()}
    enrolled_idx = [d["ids"].index(v["pool_entry"]) for v in reg.values()]
    owner_entry_map = {v["driver_identity"]: v["pool_entry"] for v in reg.values()}
    theta, ids = d["theta"], d["ids"]
    rng = np.random.default_rng(0)
    out = dict(zeta=zeta, theta=theta, theta_info=d["theta_info"], n_clips=len(d["rows"]),
               note="pre-registered ungated ArcIris retrieval; theta read on tuning subjects 000-039 at 1 % false self; EER/AUC over self (pos) vs cross (neg) clips")
    for arm in ("embedded", "embedded_gated"):
        recs = []
        for cid, r in d["rows"].items():
            z = zvec(r["embedded"]) if arm == "embedded" else np.asarray(r["gated_embedded"], float)
            zr = z[enrolled_idx]
            k_enr = int(np.nanargmin(zr))
            did_top = dids[k_enr]                                   # closed-set top-1 among ENROLLED drivers
            k79 = int(np.nanargmin(z))
            recs.append(dict(id=cid, cond=r["condition"], owner=r["owner"], did=r["did"],
                             rank1_enrolled=(did_top == r["did"]), rank1_79=(ids[k79] == r["entry"]),
                             accepted=bool(z[k79] < zeta and ids[k79] in entry_to_did),
                             did_hat=did_top, z_top=float(zr[k_enr]), z_owner=float(z[ids.index(owner_entry_map[r['owner']])]),
                             s_ret=r["iris_all"][k_enr], s_true=r["iris_true_did"]))
        n = len(recs)
        selfs = [x for x in recs if x["cond"] == "self"]; cross = [x for x in recs if x["cond"] == "cross"]
        def boot(f, B=1000):
            owners = sorted({x["owner"] for x in recs}); vals = []
            for _ in range(B):
                pick = rng.choice(owners, size=len(owners), replace=True)
                rs = [x for o in pick for x in recs if x["owner"] == o]
                S = [x for x in rs if x["cond"] == "self"]; C = [x for x in rs if x["cond"] == "cross"]
                vals.append(f(S, C))
            return [round(float(np.percentile(vals, 2.5)), 1), round(float(np.percentile(vals, 97.5)), 1)]
        eer_ret = eer_from_scores([x["s_ret"] for x in selfs], [x["s_ret"] for x in cross])
        eer_orc = eer_from_scores([x["s_true"] for x in selfs], [x["s_true"] for x in cross])
        out[arm] = dict(
            rank1_driver_among_enrolled=rate(sum(x["rank1_enrolled"] for x in recs), n),
            rank1_driver_among_79=rate(sum(x["rank1_79"] for x in recs), n),
            retrieved_and_accepted=rate(sum(x["accepted"] and x["rank1_79"] for x in recs), n),
            eer_self_vs_cross_retrieved_pct=dict(eer=round(eer_ret, 1), ci95=boot(lambda S, C: eer_from_scores([x["s_ret"] for x in S], [x["s_ret"] for x in C])),
                                                 auc=round(auc_from_scores([x["s_ret"] for x in selfs], [x["s_ret"] for x in cross]), 3)),
            eer_self_vs_cross_ruleP_owner_claim_pct=dict(eer=round(eer_from_scores([x['z_owner'] for x in selfs], [x['z_owner'] for x in cross]), 1),
                                                          ci95=boot(lambda S, C: eer_from_scores([x['z_owner'] for x in S], [x['z_owner'] for x in C])),
                                                          auc=round(auc_from_scores([x['z_owner'] for x in selfs], [x['z_owner'] for x in cross]), 3)),
            eer_self_vs_cross_oracle_did_pct=dict(eer=round(eer_orc, 1), ci95=boot(lambda S, C: eer_from_scores([x["s_true"] for x in S], [x["s_true"] for x in C])),
                                                  auc=round(auc_from_scores([x["s_true"] for x in selfs], [x["s_true"] for x in cross]), 3)),
            self_called_self_at_theta=rate(sum(x["s_ret"] < theta for x in selfs), len(selfs)),
            cross_called_cross_at_theta=rate(sum(x["s_ret"] >= theta for x in cross), len(cross)))
    # Rule P = the paper's (step 6): self iff the identified entry is committed under the DID that owns the source face (A3).
    # Rule U = supplementary: real-iris comparison of the driver's enrolled iris with the target's (the platform-side A2/A3 step).
    owner_entry = {v["driver_identity"]: v["pool_entry"] for v in reg.values()}
    for arm in ("embedded", "embedded_gated"):
        P_self_ok, P_cross_ok, n_s, n_c = 0, 0, 0, 0
        for cid, r in d["rows"].items():
            z = zvec(r["embedded"]) if arm == "embedded" else np.asarray(r["gated_embedded"], float)
            k = int(np.nanargmin(z))
            e_hat = ids[k] if (z[k] < zeta and ids[k] in entry_to_did) else None
            says_self = (e_hat is not None) and (e_hat == owner_entry[r["owner"]])
            if r["condition"] == "self":
                n_s += 1; P_self_ok += int(says_self)
            else:
                n_c += 1; P_cross_ok += int(not says_self)
        out[arm]["rule_P_did_equality"] = dict(self_called_self=rate(P_self_ok, n_s), cross_called_cross=rate(P_cross_ok, n_c))
    # motion-matched check: raw twins of one target's three clips must be identical (same source + same driving video)
    by_owner = {}
    for cid, r in d["rows"].items():
        by_owner.setdefault(r["owner"], set()).add(r["raw_hash"] if "raw_hash" in r else cid)
    out["motion_matched"] = dict(owners=len(by_owner), owners_with_identical_raw_twins=sum(1 for h in by_owner.values() if len(h) == 1),
                                 note="identical raw frames => identical blinks, mouth motion and head pose: a behavioural verifier is at chance (AUC 0.5) on these self/cross pairs by construction")
    beh = json.loads((ROOT / "outputs" / "baseline_behavioural.json").read_text())["rows"]
    out["behavioural_baselines_other_protocol"] = {k: dict(auc=v.get("auc"), eer=v.get("eer")) for k, v in beh.items()}
    ctrl_unknown = sum(1 for cid, r in d["rows"].items() if not (np.nanmin(zvec(r["control"])) < zeta and ids[int(np.nanargmin(zvec(r["control"])))] in entry_to_did))
    out["controls_no_did_retrieved"] = rate(ctrl_unknown, len(d["rows"]))
    (ROOT / "outputs" / "idmap_report.json").write_text(json.dumps(stamp(out), indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "score", "report"])
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--n-people", type=int, default=24)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--far", type=float, default=0.01)
    a = ap.parse_args()
    {"plan": plan, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
