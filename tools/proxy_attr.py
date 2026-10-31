"""Proxy-iris attribution test: plan, enrolment and generation (README.md).

The design under test is new_problem_statement.md adapted to the paper (no 100,000-entry pool, no fixed thresholds):
an issuer enrols people and gives each a proxy iris (a pool entry chosen by HMAC probing, unique per person); every
session embeds the DRIVER's proxy inline into the generated frames; the issuer writes a signed session record (source,
driver, source binding, video SHA-256 and PDQ hashes) to an append-only ledger. A verifier starts from the video alone
(tools/proxy_verify.py). Simulated ledger: tools/proxy_ledger.py.

  plan   outputs/proxy_attr_plan.json  seeded people, enrolment (person -> pool entry), sessions (self / cross / two
         attack-generation kinds), tuning split. COMMIT BEFORE `gen`.   -> also builds the enrolled ledger
  gen    LivePortrait with the driver's proxy embedded inline for every planned session, then PDQ + record + ledger block.
         Resumable (a finished session is skipped).

    python tools/proxy_attr.py plan
    python tools/proxy_attr.py gen [--limit N] [--dataset ravdess|crema]
"""
import argparse
import hashlib
import hmac
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402
import proxy_attr_lib as L  # noqa: E402
import proxy_ledger as PL  # noqa: E402

PLAN1 = ROOT / "outputs" / "proxy_attr_plan.json"
PLAN = next((q for q in (ROOT / "outputs" / "proxy_attr_plan4.json", ROOT / "outputs" / "proxy_attr_plan3.json", ROOT / "outputs" / "proxy_attr_plan2.json") if q.exists()), PLAN1)   # plan 2 = batch 2, enrolment is a superset of plan 1
WORK = ROOT / "outputs" / "proxy_attr"
LEDGER = WORK / "ledger.json"
CLIPS = WORK / "clips"
SESS = WORK / "sessions"
LP_PY = ROOT / "third_party" / "LivePortrait" / ".venv_lp" / "bin" / "python"
PDQ_PY = ROOT / "third_party" / ".venv_pdq" / "bin" / "python"
AVAPRINT_TARGETS = Path("external/AVAPrintDB/database/generation/target_identities")
SEED = 20261028
ISSUER_SEED = "proxy-attr-simulation-2026-10-28"
ISSUER_DID = "did:example:issuer"
PDQ_STRIDE, PDQ_MAX = 10, 10     # 31 (the statement) accepts different sessions of one source photo: re-encodes are within 4 bits, other sessions at 18+ (dry run)
FRAMES = 80
SCALE = dict(ravdess=3, crema=7)
N_CREMA, N_WRONG, N_UNENROLLED = 16, 8, 8


def person_did(dataset, name):
    return f"did:example:{'r' if dataset == 'ravdess' else 'c'}-{name.lower()}"


def n_frames(path):
    import cv2
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n


def collect_people(rng):
    """people[dataset] = [dict(name, source, drivers[list of mp4])] for everybody who can be a target and a driver."""
    idx = json.loads((ROOT / "outputs" / "ravdess_targets_index.json").read_text())["targets"]
    rav = [dict(name=t["actor"], source=str(ROOT / t["source"]["png"]), drivers=[str(ROOT / d["mp4"]) for d in t["drivers"]])
           for t in idx if t["drivers"] and (ROOT / t["source"]["png"]).exists()]
    crema = []
    for p in sorted((ROOT / "dbs" / "crema_d" / "mp4").glob("*_DFA_NEU_XX.mp4")):
        a = p.name[:4]
        src = AVAPRINT_TARGETS / "cremad_identities" / f"C{a}.png"
        drv = [str(ROOT / "dbs" / "crema_d" / "mp4" / f"{a}_DFA_{e}_XX.mp4") for e in ("NEU", "HAP", "SAD", "ANG")]
        if src.exists() and all(Path(d).exists() for d in drv):
            crema.append(dict(name=f"C{a}", source=str(src), drivers=drv))
    pick = sorted(rng.choice(len(crema), size=min(N_CREMA, len(crema)), replace=False))
    return dict(ravdess=rav, crema=[crema[i] for i in pick])


def build_ledger(people, pool_ids, k_idx, enrol):
    """Genesis: parameters v1, then one DID version (epoch 1) per person with the proxy templates (IrisCode and ArcIris)."""
    from cancelable.pool import IrisPool, SIDE_OF
    from iris_pipeline.encode import encode
    from iris_pipeline.judge import ArcIrisJudge
    import cv2
    pool, J = IrisPool(), ArcIrisJudge()
    issuer = PL.account_from(ISSUER_SEED, "issuer")
    led = PL.Ledger(issuer.address)
    led.set_parameters(issuer.address, dict(version=1, issuer=ISSUER_DID, extractorVersion=L.EXTRACTOR, matchThreshold=None,
                                            matchMargin=None, perceptual=dict(algorithm="pdq-per-keyframe", stride=PDQ_STRIDE, maxDistance=PDQ_MAX),
                                            note="thresholds are calibrated on the tuning sessions and published as version 2"))
    arc_cache = {}
    for dataset, plist in people.items():
        for p in plist:
            e = enrol[p["name"]]
            sid = e["entry"]
            tex = pool.textures(sid)
            iris_tpl = {side: encode(tex[side].strip, tex[side].mask) for side in L.SIDES}
            arc = {}
            for eye, side in SIDE_OF.items():
                if (sid, eye) not in arc_cache:
                    rgb = cv2.cvtColor(cv2.imread(str(pool.files[sid][eye][1])), cv2.COLOR_BGR2RGB)
                    v, info = J.encode_image(rgb)
                    arc_cache[(sid, eye)] = None if info["rejected"] else v
                v = arc_cache[(sid, eye)]
                if v is not None:
                    arc[side] = PL.b64(np.asarray(v, np.float16).tobytes())
            templates = {L.EXTRACTOR: L.pack_template(iris_tpl), "arciris-1": arc}
            acct = PL.account_from(ISSUER_SEED, "person|" + p["name"])
            did = person_did(dataset, p["name"])
            doc = dict(id=did, controller=ISSUER_DID,
                       verificationMethod=[dict(id=did + "#key-1", type="EcdsaSecp256k1RecoveryMethod2020", blockchainAccountId=acct.address)],
                       proxyIris=dict(scheme="pool-v1", epoch=1, templates=templates, extractorVersion=L.EXTRACTOR),
                       created=f"epoch-1")
            doc["proof"] = dict(type="EcdsaSecp256k1Signature", verificationMethod=ISSUER_DID + "#key-1", proofValue=PL.sign(issuer, doc))
            h = led.register_did_version(issuer.address, did, 1, doc)
            enrol[p["name"]].update(did=did, block=h, address=acct.address, templateSha256=PL.sha256_hex(PL.canon(templates)))
    return led, enrol


def plan(a):
    from cancelable.pool import IrisPool
    rng = np.random.default_rng(SEED)
    people = collect_people(rng)
    allp = [(ds, p) for ds, pl in people.items() for p in pl]
    pool_ids = IrisPool().ids
    M = len(pool_ids)
    assert len(allp) <= M * 0.8, f"{len(allp)} people for a pool of {M}"
    k_idx = hmac.new(ISSUER_SEED.encode(), b"idx-v1", hashlib.sha256).digest()
    taken, enrol = {}, {}
    for ds, p in allp:
        # stand-in for the fuzzy-extractor output R of the person's real iris (the extractor itself is not built here)
        R = hmac.new(ISSUER_SEED.encode(), ("R|" + p["name"]).encode(), hashlib.sha256).digest()
        acct = PL.account_from(ISSUER_SEED, "person|" + p["name"])
        subject_id = hashlib.sha256(R + acct.address.encode()).hexdigest()
        for k in range(64):
            idx = PL.hmac_index(k_idx, subject_id, 1, k, M)
            if idx not in taken:
                taken[idx] = p["name"]
                break
        else:
            raise SystemExit("pool too full")
        enrol[p["name"]] = dict(dataset=ds, subject_id=subject_id[:16], pool_index=idx, probe=k, entry=pool_ids[idx], epoch=1)
    free = [pool_ids[i] for i in range(M) if i not in taken]
    names = [p["name"] for _, p in allp]
    tuning = set(rng.choice(names, size=max(4, len(names) // 4), replace=False).tolist())
    byname = {p["name"]: (ds, p) for ds, p in allp}
    rows = []

    def row(kind, ds, src, drv, emb_entry, frames, tag=""):
        return dict(id=f"{'R' if ds == 'ravdess' else 'C'}_{src['name']}_{kind}{tag}_{drv['name']}", dataset=ds, kind=kind,
                    source_actor=src["name"], driver_actor=drv["name"], source=src["source"], driver=drv["drivers"][0],
                    frames=frames, scale=SCALE[ds], embed_entry=emb_entry, tuning=src["name"] in tuning)
    for ds, plist in people.items():
        for i, t in enumerate(plist):
            others = [p for p in plist if p["name"] != t["name"]]
            sel = rng.choice(len(others), size=2, replace=False)
            for kind, drv in (("self", t), ("cross", others[sel[0]]), ("cross", others[sel[1]])):
                fr = min(FRAMES, n_frames(drv["drivers"][0]))
                rows.append(row(kind, ds, t, drv, enrol[drv["name"]]["entry"], fr))
    cross_rows = [r for r in rows if r["kind"] == "cross"]
    for j in rng.choice(len(cross_rows), size=N_WRONG, replace=False):
        r = dict(cross_rows[j])
        r.update(kind="wrong_proxy", id=r["id"].replace("_cross_", "_wrongproxy_"), embed_entry=enrol[r["source_actor"]]["entry"])
        rows.append(r)
    for n, j in enumerate(rng.choice(len(cross_rows), size=N_UNENROLLED, replace=False)):
        r = dict(cross_rows[j])
        r.update(kind="unenrolled_proxy", id=r["id"].replace("_cross_", "_unenrolled_"), embed_entry=free[n % len(free)])
        rows.append(r)
    for r in rows:
        r["session_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "proxy-attr/" + r["id"]))
    WORK.mkdir(parents=True, exist_ok=True)
    led, enrol = build_ledger(people, pool_ids, k_idx, enrol)
    led.save(LEDGER)
    out = dict(seed=SEED, frames=FRAMES, scale=SCALE, pdq=dict(stride=PDQ_STRIDE, max_distance=PDQ_MAX), extractor=L.EXTRACTOR,
               pool_size=M, n_people=dict(ravdess=len(people["ravdess"]), crema=len(people["crema"])), tuning_people=sorted(tuning),
               calibration_rule=("per matcher, on the tuning sessions only: choose (T, margin) from a grid that gives ZERO wrong attributions on "
                                 "tuning sessions and ZERO accepts on tuning control twins, then maximise correct attributions; "
                                 "primary matcher = the one with the highest tuning correct rate at zero false attributions"),
               enrolment=enrol, rows=rows, free_entries=free)
    PLAN.write_text(json.dumps(stamp(out, seed=SEED), indent=1))
    from collections import Counter
    print("people", out["n_people"], "sessions", len(rows), Counter(r["kind"] for r in rows), "tuning people", len(tuning))
    print("ledger blocks", len(led.blocks), "chain ok", led.chain_ok(), "->", PLAN)


# ----------------------------------------------------------------------------------------------------- gen
def pdq_hashes(path):
    p = subprocess.run([str(PDQ_PY), str(ROOT / "tools" / "pdq_worker.py"), str(path), str(PDQ_STRIDE)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-300:])
    return json.loads(p.stdout)


def make_record(led, row, enrol, emb_mp4, fps, n_fr):
    issuer = PL.account_from(ISSUER_SEED, "issuer")
    drv, src = enrol[row["driver_actor"]], enrol[row["source_actor"]]
    photo_sha = hashlib.sha256(Path(row["source"]).read_bytes()).hexdigest()
    pq = pdq_hashes(emb_mp4)
    rec = dict(sessionId=row["session_id"], timestamp="block", nonce=hashlib.sha256(("nonce|" + row["id"]).encode()).hexdigest()[:32],
               issuer=ISSUER_DID,
               source=dict(did=src["did"], epoch=1, templateSha256=src["templateSha256"], photoSha256=photo_sha,
                           binding=dict(method="owner-signature",
                                        proofValue=PL.sign(PL.account_from(ISSUER_SEED, "person|" + row["source_actor"]), dict(photoSha256=photo_sha)))),
               driver=dict(did=drv["did"], epoch=1, templateSha256=drv["templateSha256"]),
               video=dict(sha256=hashlib.sha256(Path(emb_mp4).read_bytes()).hexdigest(),
                          perceptual=dict(algorithm="pdq-per-keyframe", stride=PDQ_STRIDE, hashes=pq["hashes"], maxDistance=PDQ_MAX),
                          frames=pq["n_frames"], durationMs=int(1000 * pq["n_frames"] / fps)),
               embedding=dict(region="eyes", compositor="layered-v2 (soft-render matte, D30/D34)", frameInterval=1, extractorVersion=L.EXTRACTOR))
    body = PL.without(rec, "proofs")
    rec["proofs"] = [dict(by="issuer", proofValue=PL.sign(issuer, body)),
                     dict(by="driver", proofValue=PL.sign(PL.account_from(ISSUER_SEED, "person|" + row["driver_actor"]), body))]
    return rec


def gen(a):
    plan_ = json.loads(PLAN.read_text())
    enrol = plan_["enrolment"]
    led = PL.Ledger.load(LEDGER)
    CLIPS.mkdir(parents=True, exist_ok=True)
    SESS.mkdir(parents=True, exist_ok=True)
    issuer = PL.account_from(ISSUER_SEED, "issuer")
    done = 0
    for row in plan_["rows"]:
        if a.dataset and row["dataset"] != a.dataset:
            continue
        if a.kinds and row["kind"] not in a.kinds.split(","):
            continue
        meta_f = SESS / f"{row['id']}.json"
        if meta_f.exists():
            continue
        if a.limit and done >= a.limit:
            break
        t0 = time.time()
        out = CLIPS / row["id"]
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(LP_PY), str(ROOT / "tools" / "reenact_video.py"), "liveportrait", "--source", row["source"], "--frames", str(row["frames"]),
               "--scale", str(row["scale"]), "--driver", row["driver"], "--embed-entry", row["embed_entry"], "-o", str(out)]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        stem = Path(row["source"]).stem
        emb = out / f"liveportrait__{stem}__inline_{row['embed_entry']}.mp4"
        raw = out / f"liveportrait__{stem}__inline_raw.mp4"
        info_f = emb.with_suffix(".json")
        if p.returncode != 0 or not emb.exists():
            meta_f.write_text(json.dumps(dict(row, status="gen_failed", stderr_tail=p.stderr[-500:]), indent=1))
            print(row["id"], "generation failed", flush=True)
            continue
        info = json.loads(info_f.read_text())
        import cv2
        c = cv2.VideoCapture(str(emb)); fps = c.get(cv2.CAP_PROP_FPS) or 25.0; c.release()
        rec = make_record(led, row, enrol, emb, fps, row["frames"])
        h = led.record_session(issuer.address, rec)
        led.save(LEDGER)
        meta = dict(row, status="ok", video=str(emb.relative_to(ROOT)), raw_twin=str(raw.relative_to(ROOT)), ledger_block=h,
                    video_sha256=rec["video"]["sha256"], embedded_eyes=info["embedded_eyes"], gated_eyes=info["gated_eyes"],
                    embed_ms_median=info["embed_ms_median"], seconds=round(time.time() - t0, 1))
        meta_f.write_text(json.dumps(meta, indent=1))
        done += 1
        print(f"{row['id']} kind {row['kind']} embedded {info['embedded_eyes']}/{info['embedded_eyes'] + info['gated_eyes']} "
              f"block {h} {meta['seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "gen"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", choices=["ravdess", "crema"], default="")
    ap.add_argument("--kinds", default="", help="comma list of session kinds to generate (self,cross,wrong_proxy,unenrolled_proxy)")
    a = ap.parse_args()
    {"plan": plan, "gen": gen}[a.cmd](a)


if __name__ == "__main__":
    main()
