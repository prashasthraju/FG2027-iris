"""Batch 4: the large session .

Everything here is a TEST session at the frozen threshold T1 of batch 1 (no new calibration):
  * more people: 36 more CASIA-Iris-Distance NIR subjects are enrolled as photo owners (DID, key, proxy); their own CASIA captures are their
    real irises (template D05-D09, probes D00-D04), exactly as the NIR people of batch 2;
  * more self and cross scenarios: every RAVDESS actor gets 3 more self sessions and 8 more cross sessions with driving clips taken from
    outputs/ravdess_drivers_more.json (seeded random clips, no quality ranking), every NIR person gets 4-5 more cross sessions;
  * the rows are shuffled with a fixed seed, so any prefix of the plan is a random sample of it.
`--personalive N` adds N sessions of a second generator (PersonaLive, 8 GB low-memory path, D29) drawn from the same scenarios.

    python tools/proxy_batch4.py                  # -> outputs/proxy_attr_plan4.json (COMMIT before generating); enrols the new NIR people
    python tools/proxy_batch4.py --personalive 60 # adds the PersonaLive rows to a plan that already exists (no new enrolment)
"""
import argparse
import hashlib
import hmac
import json
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402
import proxy_attr_lib as L  # noqa: E402
import proxy_ledger as PL  # noqa: E402
from idmap_eval import TEMPLATE_CAPS, eye_vectors, img_path  # noqa: E402
from proxy_attr import FRAMES, ISSUER_DID, ISSUER_SEED, LEDGER, WORK, n_frames  # noqa: E402

PLAN3 = ROOT / "outputs" / "proxy_attr_plan3.json"
PLAN4 = ROOT / "outputs" / "proxy_attr_plan4.json"
REAL = WORK / "real_iris.pkl"
REAL2 = WORK / "real_iris2.pkl"
REAL3 = WORK / "real_iris3.pkl"
MORE = ROOT / "outputs" / "ravdess_drivers_more.json"
N_NIR = 36
SELF_PER_ACTOR, CROSS_PER_ACTOR = 3, 8
NIR_DRIVERS_OLD, NIR_DRIVERS_NEW = 4, 5


def clip_tag(path):
    stem = Path(path).stem.replace("driver_", "")
    return "".join(stem.split("-")[2:6])                 # emotion, intensity, statement, repetition


def enrol_nir(plan, rng):
    """Enrol N_NIR more NIR people (same procedure as tools/proxy_batch2.py). Returns (enrolment additions, real-iris dict)."""
    from cancelable.pool import IrisPool, SIDE_OF
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.encode import encode
    from iris_pipeline.judge import ArcIrisJudge
    import cv2
    enrol = plan["enrolment"]
    pool = IrisPool()
    pool_ids = pool.ids
    M = len(pool_ids)
    used = {v["subject"] for v in pickle.loads(REAL.read_bytes())["real"].values()} | {v["subject"] for v in pickle.loads(REAL2.read_bytes())["real"].values()}
    subs = [f"{i:03d}" for i in range(40, 142) if f"{i:03d}" not in used and all(img_path(f"{i:03d}", c).exists() for c in range(10))]
    pick = [str(x) for x in rng.choice(subs, size=N_NIR, replace=False)]
    taken = {e["pool_index"] for e in enrol.values()}
    plan1 = json.loads((ROOT / "outputs" / "proxy_attr_plan.json").read_text())
    taken |= {pool_ids.index(r["embed_entry"]) for r in plan1["rows"] if r["kind"] == "unenrolled_proxy"}       # attack proxies stay unenrolled
    k_idx = hmac.new(ISSUER_SEED.encode(), b"idx-v1", hashlib.sha256).digest()
    led = PL.Ledger.load(LEDGER)
    issuer = PL.account_from(ISSUER_SEED, "issuer")
    J = ArcIrisJudge()
    arc_cache, real, add = {}, {}, {}
    with FaceEyeDetector() as det:
        for sub in pick:
            name = f"N{sub}"
            acct = PL.account_from(ISSUER_SEED, "person|" + name)
            R = hmac.new(ISSUER_SEED.encode(), ("R|" + name).encode(), hashlib.sha256).digest()
            subject_id = hashlib.sha256(R + acct.address.encode()).hexdigest()
            for k in range(64):
                i = PL.hmac_index(k_idx, subject_id, 1, k, M)
                if i not in taken:
                    taken.add(i)
                    break
            else:
                raise SystemExit("pool too full")
            sid = pool_ids[i]
            tex = pool.textures(sid)
            iris_tpl = {side: encode(tex[side].strip, tex[side].mask) for side in L.SIDES}
            arc = {}
            for eye, side in SIDE_OF.items():
                if (sid, eye) not in arc_cache:
                    rgb = cv2.cvtColor(cv2.imread(str(pool.files[sid][eye][1])), cv2.COLOR_BGR2RGB)
                    v, info = J.encode_image(rgb)
                    arc_cache[(sid, eye)] = None if info["rejected"] else v
                if arc_cache[(sid, eye)] is not None:
                    arc[side] = PL.b64(np.asarray(arc_cache[(sid, eye)], np.float16).tobytes())
            templates = {L.EXTRACTOR: L.pack_template(iris_tpl), "arciris-1": arc}
            did = f"did:example:n-{name.lower()}"
            doc = dict(id=did, controller=ISSUER_DID,
                       verificationMethod=[dict(id=did + "#key-1", type="EcdsaSecp256k1RecoveryMethod2020", blockchainAccountId=acct.address)],
                       proxyIris=dict(scheme="pool-v1", epoch=1, templates=templates, extractorVersion=L.EXTRACTOR), created="epoch-1")
            doc["proof"] = dict(type="EcdsaSecp256k1Signature", verificationMethod=ISSUER_DID + "#key-1", proofValue=PL.sign(issuer, doc))
            h = led.register_did_version(issuer.address, did, 1, doc)
            add[name] = dict(dataset="nir", subject_id=subject_id[:16], pool_index=i, probe=k, entry=sid, epoch=1, did=did, block=h, address=acct.address,
                             templateSha256=PL.sha256_hex(PL.canon(templates)))
            real[name] = dict(subject=sub, template=[eye_vectors(img_path(sub, c), det, J) for c in TEMPLATE_CAPS],
                              probes=[eye_vectors(img_path(sub, c), det, J) for c in range(5)])
            print("enrolled", name, "proxy", sid, "block", h, flush=True)
    led.save(LEDGER)
    REAL3.write_bytes(pickle.dumps(dict(real=real)))
    return add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imitation", type=int, default=0, help="add N motion-matched imitation rows built from generated batch-4 cross sessions")
    ap.add_argument("--extra", default="", help="add rows of other generators to the existing plan 4, e.g. personalive:24,liax:36,edtalk:36 "
                                                  "(paired: the same RAVDESS photo, driving clip and proxy as a LivePortrait row)")
    a = ap.parse_args()
    if a.imitation:
        plan4 = json.loads(PLAN4.read_text())
        enrol = plan4["enrolment"]
        rng = np.random.default_rng(plan4["seed"] + 55)
        sess = {f.stem: json.loads(f.read_text()) for f in sorted((WORK / "sessions").glob("*cross4_*.json"))}
        cand = [m for m in sess.values() if m.get("status") == "ok" and m["dataset"] == "ravdess" and m.get("generator", "liveportrait") == "liveportrait"]
        ravdess = [n for n, e in enrol.items() if e["dataset"] == "ravdess"]
        have = {r["id"] for r in plan4["rows"]}
        for i in rng.permutation(len(cand))[:a.imitation]:
            m = cand[int(i)]
            x = str(rng.choice([n for n in ravdess if n not in (m["source_actor"], m["driver_actor"])]))
            r = dict(id=f"I_{m['source_actor']}_imitates_{m['driver_actor']}_by_{x}_{clip_tag(m['driver'])}", dataset="ravdess", kind="imitation", source_actor=m["source_actor"], driver_actor=x,
                     imitates=m["driver_actor"], orig_session=m["id"], source=m["source"], driver=m["driver"], frames=m["frames"], scale=m["scale"], embed_entry=enrol[x]["entry"],
                     tuning=False, generator="liveportrait", batch=4)
            r["session_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "proxy-attr/" + r["id"]))
            if r["id"] not in have:
                plan4["rows"].append(r)
        PLAN4.write_text(json.dumps(stamp(plan4, seed=plan4["seed"]), indent=1))
        print("plan 4 now has", len(plan4["rows"]), "rows")
        return
    if a.extra:
        plan4 = json.loads(PLAN4.read_text())
        rng = np.random.default_rng(plan4["seed"] + 44)
        base = [r for r in plan4["rows"] if r["generator"] == "liveportrait" and r["dataset"] == "ravdess"]
        order = [base[i] for i in rng.permutation(len(base))]
        have = {r["id"] for r in plan4["rows"]}
        code = dict(personalive="RP", liax="RL", edtalk="RE")
        for spec in a.extra.split(","):
            gen_, n_ = spec.split(":")
            for r in order[:int(n_)]:
                fr = min(r["frames"], 80)
                n = dict(r, id=r["id"].replace("R_", code[gen_] + "_", 1), generator=gen_, scale=dict(personalive=4, liax=3, edtalk=2.5)[gen_],
                         frames=fr - fr % 4 if gen_ == "personalive" else fr, paired_with=r["id"])
                n["session_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "proxy-attr/" + n["id"]))
                if n["id"] not in have:
                    plan4["rows"].append(n)
        PLAN4.write_text(json.dumps(stamp(plan4, seed=plan4["seed"]), indent=1))
        from collections import Counter
        print("plan 4 now has", len(plan4["rows"]), "rows", Counter(r["generator"] for r in plan4["rows"]))
        return
    plan3 = json.loads(PLAN3.read_text())
    enrol = {k: dict(v) for k, v in plan3["enrolment"].items()}
    rng = np.random.default_rng(plan3["seed"] + 4)
    more = {t["actor"]: [str(ROOT / d["mp4"]) for d in t["drivers"]] for t in json.loads(MORE.read_text())["targets"]}
    idx = json.loads((ROOT / "outputs" / "ravdess_targets_index.json").read_text())["targets"]
    sources = {t["actor"]: str(ROOT / t["source"]["png"]) for t in idx if t["drivers"]}
    old_rav = [n for n, e in enrol.items() if e["dataset"] == "ravdess"]
    old_nir = [n for n, e in enrol.items() if e["dataset"] == "nir"]
    add = enrol_nir(dict(plan3, enrolment=enrol), rng)
    enrol.update(add)
    new_nir = list(add)
    rows = []

    def row(kind, ds, src, drv, src_path, drv_path, scale):
        fr = min(FRAMES, n_frames(drv_path))
        r = dict(id=f"{'R' if ds == 'ravdess' else 'N'}_{src}_{kind}4_{drv}_{clip_tag(drv_path)}", dataset=ds, kind=kind, source_actor=src, driver_actor=drv, source=src_path,
                 driver=drv_path, frames=fr, scale=scale, embed_entry=enrol[drv]["entry"], tuning=False, generator="liveportrait", batch=4)
        r["session_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "proxy-attr/" + r["id"]))
        return r
    for p in old_rav:
        for k in rng.permutation(len(more[p]))[:SELF_PER_ACTOR]:
            rows.append(row("self", "ravdess", p, p, sources[p], more[p][k], 3))
        others = [q for q in old_rav if q != p]
        for q in rng.choice(others, size=CROSS_PER_ACTOR, replace=False):
            rows.append(row("cross", "ravdess", p, str(q), sources[p], more[str(q)][int(rng.integers(len(more[str(q)])))], 3))
    for n in old_nir + new_nir:
        sub = (enrol[n].get("subject") or n[1:])
        photo = str(img_path(sub, 0))
        for q in rng.choice(old_rav, size=NIR_DRIVERS_NEW if n in new_nir else NIR_DRIVERS_OLD, replace=False):
            rows.append(row("cross", "nir", n, str(q), photo, more[str(q)][int(rng.integers(len(more[str(q)])))], 1))
    rows = [rows[i] for i in rng.permutation(len(rows))]
    from collections import Counter
    out = dict(seed=plan3["seed"], frames=FRAMES, scale=dict(ravdess=3, nir=1), pdq=plan3["pdq"], extractor=L.EXTRACTOR, pool_size=plan3["pool_size"],
               frozen=plan3["frozen"], note="batch 4 (large session): every session is a test session at the frozen T1; rows are shuffled with a fixed seed",
               n_people=dict(ravdess=len(old_rav), nir=len(old_nir) + len(new_nir)), enrolment=enrol, rows=rows)
    PLAN4.write_text(json.dumps(stamp(out, seed=plan3["seed"]), indent=1))
    print("batch 4:", len(rows), "sessions", Counter((r["dataset"], r["kind"]) for r in rows), "people", out["n_people"], "->", PLAN4)


if __name__ == "__main__":
    main()
