"""Verification through the issuer , evaluated on the generated sessions.

  * The session id travels WITH the video (a sidecar or tag: how it is carried robustly is future work). It tells the verifier which
    ledger record describes the video: source DID, driver DID, their proxies (template hashes), video hashes, source binding.
  * The issuer's keys are published in the issuer's own DID version on the ledger; every record's proof is checked against it.
  * To verify, the claimant brings their KEY and a LIVE IRIS to the same issuer that assigned the proxy. The issuer checks the key
    (signature on a fresh nonce), matches the live iris against the protected template kept at enrolment, and re-derives the proxy for
    the record's epoch with the same keyed-hash assignment used at generation time. The re-derived proxy must be the one the ledger
    holds for that DID, and the code extracted from the video's eyes must be within T1 of it. If so, the claimant was the driver:
    SELF when the record's source is the claimant too, CROSS otherwise.

Simulation choices (stated in the report): the real irises are CASIA-Iris-Distance NIR subjects paired arbitrarily with the enrolled
people (enrolled template = captures D05-D09, live probes = D00-D04); the fuzzy extractor is replaced by "biometric match at the issuer,
then the keyed derivation"; ledger, keys and issuer are simulated in one process.

    python tools/proxy_issuer_eval.py setup       # pair people with real irises, issuer threshold, issuer DID on the ledger
    python tools/proxy_issuer_eval.py evaluate    # -> outputs/proxy_attr_issuer_results.json
"""
import argparse
import hashlib
import hmac
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402
from idmap_eval import TEMPLATE_CAPS, eye_vectors, img_path, iris_distance, tune_theta  # noqa: E402
from proxy_attr import ISSUER_DID, ISSUER_SEED, LEDGER, PLAN, WORK  # noqa: E402

REAL = WORK / "real_iris.pkl"
REAL2 = WORK / "real_iris2.pkl"
REAL3 = WORK / "real_iris3.pkl"
RESULTS = ROOT / "outputs" / "proxy_attr_issuer_results.json"
FAR_REAL = 1e-3             # issuer's live-iris threshold: false-accept rate on the tuning subjects 000-039 (never used as people)


def acct(name):
    return PL.account_from(ISSUER_SEED, "person|" + name)


# ------------------------------------------------------------------------------------------------------- setup
def cmd_setup(a):
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    plan = json.loads(PLAN.read_text())
    people = list(plan["enrolment"])
    rng = np.random.default_rng(plan["seed"] + 7)
    subs = [f"{i:03d}" for i in range(40, 142) if all(img_path(f"{i:03d}", c).exists() for c in range(10))]
    pick = rng.choice(subs, size=len(people), replace=False)
    J = ArcIrisJudge()
    real = {}
    with FaceEyeDetector() as det:
        theta, info = tune_theta(det, J, far=FAR_REAL)
        for name, sub in zip(people, pick):
            real[name] = dict(subject=str(sub), template=[eye_vectors(img_path(sub, c), det, J) for c in TEMPLATE_CAPS],
                              probes=[eye_vectors(img_path(sub, c), det, J) for c in range(5)])
            print(name, "<- CASIA", sub, flush=True)
    REAL.write_bytes(pickle.dumps(dict(real=real, theta=theta, theta_info=info, far=FAR_REAL)))
    print("issuer live-iris threshold theta", round(theta, 4), info)
    led = PL.Ledger.load(LEDGER)
    if led.did_version(ISSUER_DID, 1) is None:
        issuer = PL.account_from(ISSUER_SEED, "issuer")
        doc = dict(id=ISSUER_DID, controller=ISSUER_DID,
                   verificationMethod=[dict(id=ISSUER_DID + "#key-1", type="EcdsaSecp256k1RecoveryMethod2020", blockchainAccountId=issuer.address)],
                   note="in a deployment this is the genesis block; appended here to the already populated simulation ledger")
        doc["proof"] = dict(type="EcdsaSecp256k1Signature", verificationMethod=ISSUER_DID + "#key-1", proofValue=PL.sign(issuer, doc))
        h = led.register_did_version(issuer.address, ISSUER_DID, 1, doc)
        led.save(LEDGER)
        print("issuer DID registered in block", h)


# ------------------------------------------------------------------------------------------------------- the issuer
class Issuer:
    def __init__(self, plan):
        from iris_pipeline.judge import ArcIrisJudge
        d = pickle.loads(REAL.read_bytes())
        self.real, self.theta, self.J = d["real"], d["theta"], ArcIrisJudge
        if REAL2.exists():                                 # batch 2: the NIR photo owners, whose real irises are their own CASIA images
            self.real.update(pickle.loads(REAL2.read_bytes())["real"])
        if REAL3.exists():                                 # batch 4: 36 more NIR photo owners
            self.real.update(pickle.loads(REAL3.read_bytes())["real"])
        self.enrol = plan["enrolment"]
        self.registry = json.loads(V.REGISTRY.read_text())           # the issuer's private registry: proxy templates and salts (never on the ledger)
        self.M = plan["pool_size"]
        self.k_idx = hmac.new(ISSUER_SEED.encode(), b"idx-v1", hashlib.sha256).digest()
        from cancelable.pool import IrisPool
        self.pool_ids = IrisPool().ids

    def rederive(self, led, name, did, nonce, sig, probe, epoch=1):
        """The claimant's key and live iris through the issuer. Returns dict(ok, reason, entry, template_sha256, d_iris, steps)."""
        steps = []
        dv = led.did_version(did, epoch)
        if dv is None or name not in self.enrol:
            return dict(ok=False, reason="unknown DID", steps=steps)
        addr = dv["doc"]["verificationMethod"][0]["blockchainAccountId"]
        sig_ok = PL.signer_of(dict(nonce=nonce, purpose="rederive-proxy", did=did), sig) == addr
        steps.append(("signature on the fresh nonce matches the key in the DID", sig_ok, ""))
        if not sig_ok:
            return dict(ok=False, reason="the signature does not match the key published in the DID", steps=steps)
        d = iris_distance(probe, self.real[name]["template"], self.J)
        iris_ok = bool(np.isfinite(d) and d < self.theta)
        steps.append(("live iris matches the template enrolled for this DID", iris_ok, f"d={d:.3f}, theta={self.theta:.3f}"))
        if not iris_ok:
            return dict(ok=False, reason="the live iris does not match the iris enrolled for this DID", d_iris=d, steps=steps)
        R = hmac.new(ISSUER_SEED.encode(), ("R|" + name).encode(), hashlib.sha256).digest()      # stand-in for the fuzzy-extractor output
        subject_id = hashlib.sha256(R + addr.encode()).hexdigest()
        e = self.enrol[name]
        idx = PL.hmac_index(self.k_idx, subject_id, epoch, e["probe"], self.M)
        entry = self.pool_ids[idx]
        steps.append(("keyed derivation (subject id, epoch, stored probe) gives the registered pool entry", entry == e["entry"], f"pool entry {entry}"))
        return dict(ok=entry == e["entry"], reason="" if entry == e["entry"] else "derivation mismatch", entry=entry,
                    commitment=self.registry[f"{did}|{epoch}"]["commitment"] if entry == e["entry"] else None, d_iris=d, steps=steps)

    def open_commitment(self, did, epoch):
        """After a successful claim the issuer opens the commitment to the verifier: the proxy template and the salt."""
        r = self.registry[f"{did}|{epoch}"]
        return dict(templates=r["templates"], salt=r["salt"])


# ------------------------------------------------------------------------------------------------------- verification
def verify_session(led, issuer, G, cache, video_path, session_id, claimant, probe, signer, matcher, T1, pdq_cache=None, opener=None):
    """The commit-and-reveal flow. claimant = enrolled person's name; probe = live-iris vectors; signer = the account signing the issuer's nonce;
    opener = what the issuer hands the verifier as the proxy (default: the registry's opening; attacks pass another one). G is kept for the callers' signature."""
    tr = []
    res = led.session(session_id)
    if res is None:
        tr.append(dict(step="1 read the session record", detail=f"session {session_id[:8]}... is not on the ledger"))
        return dict(outcome="UNREGISTERED", trace=tr)
    rec, height = res["rec"], res["height"]
    sha = PL.sha256_hex(Path(video_path).read_bytes())
    pq = pdq_cache if pdq_cache is not None else V.pdq_hashes(video_path)["hashes"]
    if rec["video"]["sha256"] == sha:
        assur, info = "EXACT", None
    else:
        ok, info = V.pdq_match(pq, rec["video"]["perceptual"])
        assur = "REENCODED" if ok else None
    tr.append(dict(step="1 read the session record from the ledger",
                   detail=f"block {height}: source {rec['source'].get('did') or 'unenrolled'}, driver {rec['driver']['did']}; video hash match: "
                          + (assur or "NONE (the record does not describe this video)")
                          + (f" (PDQ median {info['median']:.0f}, max {info['max']} of 256 bits)" if info else ""), block=height, ok=assur is not None,
                   data=dict(file_sha256=sha, record_sha256=rec["video"]["sha256"], assurance=assur, pdq=info, pdq_bound=rec["video"]["perceptual"]["maxDistance"],
                             source=rec["source"].get("did"), driver=rec["driver"]["did"], session_id=session_id)))
    if assur is None:
        return dict(outcome="REJECT", reason="the video does not match the record's hashes", trace=tr)
    ok, checks = V.check_record(led, rec, height)
    tr.append(dict(step="2 check the record (issuer key from the issuer's DID on the ledger)", detail="all checks pass" if ok else "a check FAILED", checks=checks, ok=ok))
    if not ok:
        return dict(outcome="REJECT", reason="a record check failed", trace=tr)
    did = issuer.enrol[claimant]["did"]
    nonce = hashlib.sha256(("nonce|" + session_id + claimant).encode()).hexdigest()[:24]
    sig = PL.sign(signer, dict(nonce=nonce, purpose="rederive-proxy", did=did))
    rd = issuer.rederive(led, claimant, did, nonce, sig, probe, epoch=rec["driver"]["epoch"])
    tr.append(dict(step=f"3 the claimant ({did}) brings key and live iris to the issuer", detail=("proxy re-derived: pool entry " + rd["entry"]) if rd["ok"] else "CLAIM FAILED: " + rd["reason"],
                   checks=rd["steps"], ok=rd["ok"], data=dict(entry=rd.get("entry"), d_iris=rd.get("d_iris"), theta=issuer.theta)))
    if not rd["ok"]:
        return dict(outcome="CLAIM_FAILED", reason=rd["reason"], trace=tr)
    same = rd["commitment"] == rec["driver"]["commitment"] and did == rec["driver"]["did"]
    tr.append(dict(step="4 same proxy as in the record", detail=("the re-derived proxy is the driver's proxy named by the record" if same
                                                               else f"the re-derived proxy is NOT the driver's proxy: the record names {rec['driver']['did']} as driver"), ok=same,
                   data=dict(rederived_commitment=rd["commitment"], record_driver_commitment=rec["driver"]["commitment"], claimant=did, record_driver=rec["driver"]["did"])))
    if not same:
        return dict(outcome="REJECT", reason="the claimant is not the record's driver", trace=tr)
    opened = (opener or issuer.open_commitment)(did, rec["driver"]["epoch"])
    c_open = PL.commit(opened["templates"], opened["salt"])
    c_ledger = led.did_version(rec["driver"]["did"], rec["driver"]["epoch"])["doc"]["proxyCommitment"]["value"]
    opened_ok = c_open == c_ledger == rec["driver"]["commitment"]
    tr.append(dict(step="5 the issuer opens the commitment (proxy template and salt) to the verifier",
                   detail="keccak256(template || salt) equals the commitment on the ledger" if opened_ok else "the opened proxy does NOT match the commitment on the ledger", ok=opened_ok,
                   data=dict(commitment=c_ledger, recomputed=c_open)))
    if not opened_ok:
        return dict(outcome="REJECT", reason="the opened proxy does not match the commitment", trace=tr)
    G1 = V.Gallery.single(did, rec["driver"]["epoch"], opened["templates"])
    vec, nobs = G1.vectors(cache)
    d_drv = float(vec[matcher][0])
    src_did = rec["source"].get("did")
    tr.append(dict(step="6 distance between the video's code and the opened proxy",
                   detail=f"{matcher}: {nobs[matcher]} usable observations; distance {d_drv:.4f}; T1 = {T1:.4f}", top=[(rec["driver"]["did"], round(d_drv, 4))],
                   ok=bool(np.isfinite(d_drv) and d_drv < T1), data=dict(d_driver=d_drv, d_source=None, T1=T1, matcher=matcher, n_obs=nobs[matcher])))
    if not np.isfinite(d_drv):
        return dict(outcome="INCONCLUSIVE", reason="no usable code in the video", trace=tr)
    if d_drv < T1:
        out = "SELF" if src_did == did else "CROSS"
        tr.append(dict(step="7 decide", detail=f"distance satisfied: the claimant drove this video; the record's source is {src_did or 'unenrolled'} -> {out} ({assur})", ok=True,
                       data=dict(source=src_did, claimant=did)))
        return dict(outcome=out, driver=rec["driver"]["did"], source=src_did, assurance=assur, session=session_id, block=height, trace=tr)
    tr.append(dict(step="7 decide", detail="distance NOT satisfied: the video's code is not within T1 of the opened proxy"))
    return dict(outcome="INCONCLUSIVE", reason="distance not satisfied", trace=tr)


# ------------------------------------------------------------------------------------------------------- evaluation
def cmd_evaluate(a):
    plan = json.loads(PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(LEDGER)
    calib = json.loads(V.CALIB.read_text())
    cm = calib["claim_primary"]
    T1 = calib["claim"][cm]["T1"]
    issuer = Issuer(plan)
    real = issuer.real
    G = V.Gallery(led)
    pdq_dir = WORK / "pdq"

    def pq_of(key, path):
        f = pdq_dir / f"{key}.json"
        if not f.exists():
            f.write_text(json.dumps(V.pdq_hashes(path)["hashes"]))
        return json.loads(f.read_text())
    metas = V.sessions(all_generators=True)
    by_id = {m["id"]: m for m in metas}
    vp = json.loads((WORK / "variant_plan.json").read_text()) if (WORK / "variant_plan.json").exists() else {}
    names = list(enrol)
    rng = np.random.default_rng(plan["seed"] + 11)
    keys = []
    for m in metas:
        keys.append((m["id"], m, "original"))
        keys.append((m["id"] + "__raw", m, "raw_twin"))
    for kind, ids in vp.items():
        for i in ids:
            if i in by_id:
                keys.append((f"{i}__{kind}", by_id[i], kind))

    def base(m, variant, v):
        return dict(id=m["id"], kind=m["kind"], dataset=m["dataset"], tuning=m["tuning"], generator=m.get("generator", "liveportrait"), source_actor=m["source_actor"], driver_actor=m["driver_actor"],
                    truth_driver=enrol[m["driver_actor"]]["did"], variant=variant, **{k: v.get(k) for k in ("outcome", "assurance", "driver", "source", "reason")})
    rows, claims = [], []
    cnt = 0
    for key, m, variant in keys:
        c = V.load_cache(key)
        path = V.clip_path(key)
        if c is None or not path.exists():
            continue
        drv = m["driver_actor"]
        probe = real[drv]["probes"][cnt % 5]
        cnt += 1
        v = verify_session(led, issuer, G, c, path, m["session_id"], drv, probe, acct(drv), cm, T1, pq_of(key, path))
        rows.append(base(m, variant, v))
        if variant == "original" and m["kind"] in ("self", "cross"):
            others = [n for n in names if n != drv]
            for n in rng.choice(others, size=3, replace=False):                       # (a) another enrolled person claims to be the driver (own key, own iris)
                v2 = verify_session(led, issuer, G, c, path, m["session_id"], str(n), real[str(n)]["probes"][cnt % 5], acct(str(n)), cm, T1, pq_of(key, path))
                claims.append(dict(id=m["id"], tuning=m["tuning"], claim="other person, own key and iris", outcome=v2["outcome"], reason=v2.get("reason")))
            if m["kind"] == "cross":                                                    # (a') the source's owner claims to have driven his own photo
                sn = m["source_actor"]
                v2 = verify_session(led, issuer, G, c, path, m["session_id"], sn, real[sn]["probes"][cnt % 5], acct(sn), cm, T1, pq_of(key, path))
                claims.append(dict(id=m["id"], tuning=m["tuning"], claim="the photo's owner claims to be the driver", outcome=v2["outcome"], reason=v2.get("reason")))
            n = str(rng.choice(others))                                                 # (b) the driver's key, somebody else's iris
            v2 = verify_session(led, issuer, G, c, path, m["session_id"], drv, real[n]["probes"][cnt % 5], acct(drv), cm, T1, pq_of(key, path))
            claims.append(dict(id=m["id"], tuning=m["tuning"], claim="driver's key, another person's iris", outcome=v2["outcome"], reason=v2.get("reason")))
            v2 = verify_session(led, issuer, G, c, path, m["session_id"], drv, real[drv]["probes"][cnt % 5], acct(n), cm, T1, pq_of(key, path))   # (c) driver's iris, another key
            claims.append(dict(id=m["id"], tuning=m["tuning"], claim="driver's iris, another person's key", outcome=v2["outcome"], reason=v2.get("reason")))
    lp_rows = [r for r in rows if r["generator"] == "liveportrait"]
    sm = V.summarise(lp_rows, T1, 0.0)
    sm["by_generator"] = {}
    for g in sorted({r["generator"] for r in rows}):
        hh = [r for r in rows if r["generator"] == g and not r["tuning"] and r["variant"] == "original" and r["kind"] in ("self", "cross")]
        tw = [r for r in rows if r["generator"] == g and not r["tuning"] and r["variant"] == "raw_twin" and r["kind"] in ("self", "cross")]
        ok = lambda r: r["outcome"] == ("SELF" if r["kind"] == "self" else "CROSS") and r["driver"] == r["truth_driver"]
        sm["by_generator"][g] = dict(n=len(hh), correct=V.rate(sum(ok(r) for r in hh), len(hh)), wrong=V.rate(sum(r["outcome"] in ("SELF", "CROSS") and not ok(r) for r in hh), len(hh)),
                                     refused=V.rate(sum(r["outcome"] == "INCONCLUSIVE" for r in hh), len(hh)), twin_attributed=V.rate(sum(r["outcome"] in ("SELF", "CROSS") for r in tw), len(tw)),
                                     outcomes=dict(Counter(r["outcome"] for r in hh)))
    test_claims = [r for r in claims if not r["tuning"]]
    sm["false_claims"] = {}
    for kind in sorted({r["claim"] for r in test_claims}):
        sel = [r for r in test_claims if r["claim"] == kind]
        sm["false_claims"][kind] = dict(n=len(sel), accepted_as_self_or_cross=V.rate(sum(r["outcome"] in ("SELF", "CROSS") for r in sel), len(sel)),
                                        outcomes=dict(Counter(r["outcome"] for r in sel)))
    gen = [iris_distance(p, real[n]["template"], issuer.J) for n in names for p in real[n]["probes"]]
    imp = [iris_distance(p, real[o]["template"], issuer.J) for n in names for p in real[n]["probes"] for o in names if o != n]
    gen, imp = np.array([g for g in gen if np.isfinite(g)]), np.array([x for x in imp if np.isfinite(x)])
    sm["issuer_live_iris_check"] = dict(theta=round(issuer.theta, 4), far_target=FAR_REAL, genuine=V.rate(int((gen < issuer.theta).sum()), len(gen)),
                                        impostor=V.rate(int((imp < issuer.theta).sum()), len(imp)))
    atk = ledger_attacks(plan, led, issuer, G, cm, T1, [m for m in metas if m.get("generator", "liveportrait") == "liveportrait"], pq_of, real)
    out = dict(matcher=cm, T1=T1, results=sm, record_attacks=atk, rows=rows, claims=claims,
               setup=dict(real_irises="CASIA-Iris-Distance NIR subjects paired arbitrarily with the enrolled people (enrolled = D05-D09, live probes = D00-D04)",
                          session_id="travels with the video (assumed intact; robust carriage is future work)"))
    RESULTS.write_text(json.dumps(stamp(out), indent=1))
    pk = {k: v for k, v in sm.items() if k in ("honest_all", "honest_self", "honest_cross", "honest_ravdess", "honest_crema", "control_raw_twin", "issuer_live_iris_check")}
    print(json.dumps(pk, indent=1)[:4000])
    print(json.dumps(sm["false_claims"], indent=1)[:2500])
    print(json.dumps(atk, indent=1))
    for k in [k for k in sm if k.startswith(("variant_", "attack_"))]:
        print(k, sm[k]["outcomes"] if "outcomes" in sm[k] else sm[k])


def ledger_attacks(plan, led, issuer, G, mt, T1, metas, pq_of, real, n=12):
    enrol = plan["enrolment"]
    rng = np.random.default_rng(plan["seed"] + 2)
    cand = [m for m in metas if m["kind"] in ("self", "cross") and not m["tuning"] and V.load_cache(m["id"]) is not None]
    sel = [cand[i] for i in rng.permutation(len(cand))[:n]]
    out = {}

    def blk(L2, m):
        return next(b for b in L2.blocks if b["kind"] == "session" and b["payload"]["sessionId"] == m["session_id"])

    def run(name, mutate, opener=None):
        res = []
        for m in sel:
            L2 = led.fork()
            mutate(L2, m)
            path = V.clip_path(m["id"])
            drv = m["driver_actor"]
            v = verify_session(L2, issuer, None, V.load_cache(m["id"]), path, m["session_id"], drv, real[drv]["probes"][0], acct(drv), mt, T1, pq_of(m["id"], path),
                               opener=opener(m) if opener else None)
            res.append(v["outcome"])
        out[name] = dict(n=len(res), outcomes=dict(Counter(res)))

    def record_missing(L2, m):
        blk(L2, m)["kind"] = "void"

    def forged_binding(L2, m):
        b = blk(L2, m)
        b["payload"]["source"]["binding"]["proofValue"] = PL.sign(PL.account_from(ISSUER_SEED, "person|nobody"), dict(photoSha256=b["payload"]["source"]["photoSha256"]))

    def revoked_before(L2, m):
        b = blk(L2, m)
        L2.blocks.insert(b["height"], dict(height=b["height"], time=0, prev="", kind="revoke", payload=dict(did=b["payload"]["driver"]["did"], epoch=1), hash=""))
        for i, bb in enumerate(L2.blocks):
            bb["height"] = i

    def tampered(L2, m):
        blk(L2, m)["payload"]["nonce"] = "0" * 32

    def wrong_issuer_key(L2, m):                   # the issuer's published key does not match the key that signed the record
        v = next(b for b in L2.blocks if b["kind"] == "did" and b["payload"]["did"] == ISSUER_DID)
        v["payload"]["doc"]["verificationMethod"][0]["blockchainAccountId"] = PL.account_from("someone-else", "issuer").address
    def nothing(L2, m):
        pass

    def other_of(m):                                # a person other than the driver (for a self session the source is the driver, so it cannot serve)
        return next(e["did"] for n, e in enrol.items() if n != m["driver_actor"] and e["dataset"] != "crema")

    def opens_source(m):                            # the issuer hands over another person's proxy (template and salt) as the driver's
        return lambda did, ep: issuer.open_commitment(other_of(m), 1)

    def opens_wrong_salt(m):                        # the right template with another person's salt
        return lambda did, ep: dict(templates=issuer.open_commitment(did, ep)["templates"], salt=issuer.open_commitment(other_of(m), 1)["salt"])
    run("issuer_opens_another_proxy", nothing, opens_source)
    run("issuer_opens_with_a_wrong_salt", nothing, opens_wrong_salt)
    run("record_missing (session id unknown)", record_missing)
    run("forged_source_binding", forged_binding)
    run("driver_revoked_before_record", revoked_before)
    run("record_tampered", tampered)
    run("issuer_key_on_ledger_does_not_match_the_record_proof", wrong_issuer_key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "evaluate"])
    a = ap.parse_args()
    {"setup": cmd_setup, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
