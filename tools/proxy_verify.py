# NOTE: the module docstring below describes the earlier 1:N verifier. Still used by the paper: Gallery, check_record, pdq_match, load_cache, extract/variants/calibrate.
"""Verification test for the proxy-iris attribution design (README.md).

A verifier starts from the video alone, exactly as in new_problem_statement.md:

  1 extract   a code from the video's eyes (re-detected from scratch, D7)
  2 find DID  1:N over EVERY DID version on the ledger: closest proxy X, d1, second closest d2; accept iff d1 < T and d2 - d1 >= margin
  3 find the session record: the records of X as driver whose video hash matches the file (exact SHA-256, else PDQ)
  4 check the record: issuer proof, driver countersignature, template hashes, revocation by block order, source binding
  5 decide    SELF / CROSS / UNREGISTERED / FLAGGED / REJECT / INCONCLUSIVE, with the assurance EXACT or REENCODED

Distances come from three matchers whose per-observation data are cached once per clip:
  irislg-cons  our IrisCode, per-bit majority vote over frames (the statement's consensus code), HD to the published template
  arciris      ArcIris vectors, mean angular distance over observations (template = ArcIris vector of the proxy image)
  arciris-geo  the same, keeping only observations whose ArcIris circle agrees with MediaPipe's
T and margin per matcher are calibrated on the TUNING sessions only (rule in the plan file) and published to the ledger.

    python tools/proxy_verify.py variants        # re-encodes, blur, alteration (ffmpeg)
    python tools/proxy_verify.py extract         # per-clip caches (slow, CPU + a little GPU)
    python tools/proxy_verify.py calibrate
    python tools/proxy_verify.py evaluate        # -> outputs/proxy_attr_results.json
"""
import argparse
import base64
import json
import os
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from provenance import stamp  # noqa: E402
import proxy_attr_lib as L  # noqa: E402
import proxy_ledger as PL  # noqa: E402
from proxy_attr import (CLIPS, ISSUER_DID, ISSUER_SEED, LEDGER, PDQ_MAX, PDQ_STRIDE, PLAN, SESS, WORK, pdq_hashes)  # noqa: E402

CACHE = WORK / "cache"
VARIANTS = WORK / "variants"
CALIB = ROOT / "outputs" / "proxy_attr_calibration.json"
REGISTRY = WORK / "registry.json"            # the issuer's private registry (proxy templates and salts); NOT on the ledger
RESULTS = ROOT / "outputs" / "proxy_attr_results.json"
MATCHERS = ("irislg-cons", "arciris", "arciris-geo")
EVERY = 2                      # every second frame is analysed (the embedding is in every frame, so this only halves the cost)
PDQ_FRAC = 0.9                 # share of the record's keyframes that must find a copy keyframe within the PDQ distance


# ----------------------------------------------------------------------------------------------- variants
def sessions(all_generators=False):
    """Evaluated sessions: LivePortrait only unless all_generators (the paired PersonaLive / LIA-X / EDTalk sessions of batch 4 are reported separately)."""
    out = []
    for f in sorted(SESS.glob("*.json")):
        m = json.loads(f.read_text())
        if m.get("status") == "ok" and m.get("dataset") != "crema" and (all_generators or m.get("generator", "liveportrait") == "liveportrait"):       # CREMA-D was dropped from the analysis 
            out.append(m)
    if os.environ.get("PROXY_DEBUG"):          # debugging the tooling on a handful of sessions: alternate tuning / test, never for results
        for i, m in enumerate(out):
            m["tuning"] = i % 2 == 0
    return out


def variant_plan(plan, metas):
    rng = np.random.default_rng(plan["seed"] + 1)
    honest = [m for m in metas if m["kind"] in ("self", "cross") and not m["tuning"] and m.get("batch") != 4]      # batch 4 has its own draw (cmd_variants)
    order = rng.permutation(len(honest))
    pick = lambda a, b: [honest[i]["id"] for i in order[a:b]]
    return dict(crf23=pick(0, 30), crf30=pick(0, 30), blur=pick(30, 45), alter=pick(45, 60))


FFMPEG = dict(
    crf23=["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p", "-an"],
    crf30=["-c:v", "libx264", "-crf", "30", "-preset", "medium", "-pix_fmt", "yuv420p", "-an"],
    blur=["-vf", "gblur=sigma=14", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an"],
    alter=["-vf", "drawbox=x=0:y=ih*0.55:w=iw:h=ih*0.45:color=black:t=fill", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an"])


def cmd_variants(a):
    plan = json.loads(PLAN.read_text())
    metas = {m["id"]: m for m in sessions()}
    vp = variant_plan(plan, list(metas.values()))
    rows4 = [r["id"] for r in plan["rows"] if r.get("batch") == 4 and r.get("generator", "liveportrait") == "liveportrait"]        # plan order is a seeded shuffle
    vp4 = dict(crf23=rows4[:60], crf30=rows4[:60], blur=rows4[60:90], alter=rows4[90:120])
    vp = {k: vp[k] + vp4[k] for k in vp}
    VARIANTS.mkdir(parents=True, exist_ok=True)
    (WORK / "variant_plan.json").write_text(json.dumps(vp, indent=1))
    for kind, ids in vp.items():
        for i in ids:
            out = VARIANTS / f"{i}__{kind}.mp4"
            if out.exists() or i not in metas:
                continue
            src = ROOT / metas[i]["video"]
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *FFMPEG[kind], str(out)], check=True)
            print("variant", out.name, flush=True)


# ----------------------------------------------------------------------------------------------- extraction cache
def clip_path(key):
    """key = session id | id__raw | id__crf23 | ..."""
    if "__" in key:
        sid, v = key.split("__", 1)
        if v == "raw":
            return ROOT / json.loads((SESS / f"{sid}.json").read_text())["raw_twin"]
        if v == "wm":
            return WORK / "wm" / f"{sid}.mp4"
        return VARIANTS / f"{key}.mp4"
    return ROOT / json.loads((SESS / f"{key}.json").read_text())["video"]


def extract_key(key, det, J):
    from cancelable.identify import geometry_agrees
    from iris_pipeline.encode import encode
    from iris_pipeline.segment import extract_irises
    frames = L.read_frames(clip_path(key), every=EVERY)
    obs = []
    for i, rgb in enumerate(frames):
        for seg in extract_irises(rgb, det):
            if seg.side not in L.SIDES:
                continue
            code = encode(seg.strip, seg.strip_mask)
            v, info = J.encode_image(J.eye_crop(rgb, seg.iris))
            rej = bool(info["rejected"])
            obs.append(dict(frame=i, side=seg.side, bits=np.packbits(code.bits.ravel()), mask=np.packbits(code.mask.ravel()),
                            vis=float(np.mean(seg.strip_mask)), arc=None if rej else np.asarray(v, np.float16),
                            geo_ok=False if rej else geometry_agrees(info["iris"])))
    return dict(key=key, n_frames=len(frames), obs=obs)


def cmd_extract(a):
    from iris_pipeline.detect import FaceEyeDetector
    from iris_pipeline.judge import ArcIrisJudge
    plan = json.loads(PLAN.read_text())
    metas = sessions(all_generators=True)
    raw_ok = {r["id"] for r in plan["rows"][:a.raw_first]} if a.raw_first else None      # un-embedded twins for the first rows of the (shuffled) plan only
    keys = [m["id"] for m in metas] + [f"{m['id']}__raw" for m in metas if raw_ok is None or m.get("batch") != 4 or m["id"] in raw_ok or m.get("generator", "liveportrait") != "liveportrait"]
    vp = json.loads((WORK / "variant_plan.json").read_text()) if (WORK / "variant_plan.json").exists() else {}
    keys += [f"{i}__{k}" for k, ids in vp.items() for i in ids]
    keys += [f"{q.stem}__wm" for q in sorted((WORK / "wm").glob("*.mp4"))]      # videos that carry the DID tag (tools/proxy_wm.py)
    CACHE.mkdir(parents=True, exist_ok=True)
    J = ArcIrisJudge()
    done = 0
    with FaceEyeDetector() as det:
        for key in keys:
            f = CACHE / f"{key}.pkl"
            if zlib.crc32(key.encode()) % a.of != a.shard or f.exists() or not clip_path(key).exists():
                continue
            if a.limit and done >= a.limit:
                break
            t0 = time.time()
            d = extract_key(key, det, J)
            f.write_bytes(pickle.dumps(d))
            done += 1
            print(f"{key}: {len(d['obs'])} observations, {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- distances
def unpack(o):
    from iris_pipeline.encode import IrisCode
    shape = (16, 512, 2)
    n = int(np.prod(shape))
    return IrisCode(np.unpackbits(o["bits"])[:n].reshape(shape).astype(bool), np.unpackbits(o["mask"])[:n].reshape(shape).astype(bool))


class Gallery:
    """Every DID version on the ledger as a 1:N gallery for the three matchers."""

    def __init__(self, led=None, params=None, templates=None):
        """The proxies of the analysis: the issuer's PRIVATE registry (ledger v2 carries only commitments, never a proxy code); the ledger itself is used only for
        a v1 ledger that still holds public proxies. `templates` = [(did, epoch, template)] builds an ad hoc gallery (the single proxy a verifier was given)."""
        if templates is not None:
            items = list(templates)
        elif REGISTRY.exists():
            reg = json.loads(REGISTRY.read_text())
            items = [(r["did"], r["epoch"], r["templates"]) for r in reg.values() if not r["did"].startswith("did:example:c-")]      # CREMA-D DIDs dropped
        else:
            items = [(v["did"], v["epoch"], v["doc"]["proxyIris"]["templates"]) for v in led.all_did_versions()
                     if "proxyIris" in v["doc"] and not v["did"].startswith("did:example:c-")]
        self.names = [(d, e) for d, e, _ in items]
        self.iris = []
        self.arc = []
        for _, _, t in items:
            self.iris.append(L.unpack_template(t[L.EXTRACTOR]))
            self.arc.append({s: np.frombuffer(base64.b64decode(b), np.float16).astype(np.float32) for s, b in t.get("arciris-1", {}).items()})

    @classmethod
    def single(cls, did, epoch, templates):
        return cls(templates=[(did, epoch, templates)])

    def vectors(self, cache):
        """{matcher: distance vector over the DID versions (nan = not computable)} and the usable-observation counts."""
        from iris_pipeline.judge import ArcIrisJudge
        obs = cache["obs"]
        out, n = {}, {}
        cons, st = L.consensus([(o["frame"], o["side"], unpack(o), o["vis"]) for o in obs if o["vis"] >= L.MIN_VISIBLE])
        n["irislg-cons"] = sum(st["n_good"].values())
        out["irislg-cons"] = np.array([L.hd_to(cons, t) if cons else np.nan for t in self.iris])
        for name, need_geo in (("arciris", False), ("arciris-geo", True)):
            use = [o for o in obs if o["arc"] is not None and (o["geo_ok"] or not need_geo)]
            n[name] = len(use)
            if len(use) < L.MIN_OBS:
                out[name] = np.full(len(self.names), np.nan)
                continue
            D = np.array([[ArcIrisJudge.distance(o["arc"].astype(np.float32), g[o["side"]]) if o["side"] in g else np.nan for g in self.arc]
                          for o in use])
            out[name] = np.nanmean(D, 0)
        return out, n


# LEGACY (1:N decision rule with a margin); kept because calibrate() still tabulates it.
def decide(d, T, margin):
    """The statement's step 2 on a distance vector: (index of X or None, d1, d2, reason)."""
    ok = np.where(~np.isnan(d))[0]
    if len(ok) < 2:
        return None, np.nan, np.nan, "no usable code"
    o = ok[np.argsort(d[ok])]
    d1, d2 = float(d[o[0]]), float(d[o[1]])
    if d1 >= T:
        return None, d1, d2, f"d1 {d1:.3f} >= T"
    if d2 - d1 < margin:
        return None, d1, d2, f"margin {d2 - d1:.3f} < {margin:.3f}"
    return int(o[0]), d1, d2, "match"


# ----------------------------------------------------------------------------------------------- calibration
def load_cache(key):
    f = CACHE / f"{key}.pkl"
    return pickle.loads(f.read_bytes()) if f.exists() else None


def cmd_calibrate(a):
    plan = json.loads(PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(LEDGER)
    G = Gallery(led)
    metas = [m for m in sessions() if m["tuning"]]
    rows = []
    for m in metas:
        c = load_cache(m["id"])
        if c is None:
            continue
        vec, _ = G.vectors(c)
        raw = load_cache(m["id"] + "__raw")
        rvec = G.vectors(raw)[0] if raw else None
        rows.append((m, vec, rvec))
    print(len(rows), "tuning sessions with caches")
    res = {}
    for mt in MATCHERS:
        d1s = [decide(v[mt], 9, 0)[1] for m, v, r in rows if np.isfinite(decide(v[mt], 9, 0)[1])]
        gaps = [decide(v[mt], 9, 0)[2] - decide(v[mt], 9, 0)[1] for m, v, r in rows if np.isfinite(decide(v[mt], 9, 0)[1])]
        if not d1s:
            continue
        Ts = np.unique(np.round(np.quantile(d1s, np.linspace(0.02, 1.0, 50)) + 1e-6, 4))
        Ms = np.unique(np.round(np.concatenate([[0.0], np.quantile(gaps, np.linspace(0.05, 0.95, 40))]), 4))
        best = None
        for T in Ts:
            for M in Ms:
                ok = wrong = ctl = 0
                for m, v, r in rows:
                    x, _, _, _ = decide(v[mt], T, M)
                    if m["kind"] in ("self", "cross"):
                        if x is not None:
                            if G.names[x][0] == enrol[m["driver_actor"]]["did"]:
                                ok += 1
                            else:
                                wrong += 1
                    elif m["kind"] == "unenrolled_proxy" and x is not None:
                        ctl += 1
                    if r is not None and decide(r[mt], T, M)[0] is not None:
                        ctl += 1
                if wrong == 0 and ctl == 0:
                    key = (ok, M, -T)
                    if best is None or key > best[0]:
                        best = (key, float(T), float(M), ok)
        n_hon = sum(1 for m, _, _ in rows if m["kind"] in ("self", "cross"))
        res[mt] = dict(T=best[1], margin=best[2], tuning_correct=best[3], tuning_honest_sessions=n_hon,
                       tuning_correct_rate=round(best[3] / max(n_hon, 1), 3)) if best else dict(T=None, margin=None, tuning_correct=0,
                                                                                         tuning_honest_sessions=n_hon, tuning_correct_rate=0.0)
        print(mt, res[mt])
    primary = max((m for m in res if res[m]["T"] is not None), key=lambda m: (res[m]["tuning_correct_rate"], m == "arciris-geo", m == "arciris"), default=None)
    # claim mode (verification of a named DID): T1 = the tuning impostor-distance quantile at the target FAR; impostors are the other
    # DIDs' proxies against the clip. Primary claim matcher = the best tuning genuine acceptance at that T1.
    claim = {}
    didx = {n_[0]: i for i, n_ in enumerate(G.names)}
    for mt in MATCHERS:
        gen, imp = [], []
        for m, v, r in rows:
            if m["kind"] not in ("self", "cross"):
                continue
            d = v[mt]
            k = didx[enrol[m["driver_actor"]]["did"]]
            gen.append(float(d[k]))
            o = np.delete(d, k)
            imp += [float(x) for x in o[np.isfinite(o)]]
        if not gen or not imp:
            continue
        for far in (1e-2,):
            T1 = float(np.quantile(imp, far))
            claim[mt] = dict(T1=round(T1, 4), far=far, tuning_genuine_accepted=int(sum(np.isfinite(g) and g < T1 for g in gen)), tuning_genuine=len(gen))
    claim_primary = max(claim, key=lambda m: (claim[m]["tuning_genuine_accepted"], m == "arciris-geo", m == "arciris"), default=None)
    out = dict(rule=plan["calibration_rule"], matchers=res, primary=primary, n_tuning_sessions=len(rows), claim=claim, claim_primary=claim_primary)
    CALIB.write_text(json.dumps(stamp(out), indent=1))
    if primary:
        issuer = PL.account_from(ISSUER_SEED, "issuer")
        led.set_parameters(issuer.address, dict(version=2, issuer=ISSUER_DID, extractorVersion=L.EXTRACTOR, matchers=res, primary=primary, claim=claim, claim_primary=claim_primary,
                                                perceptual=dict(algorithm="pdq-per-keyframe", stride=PDQ_STRIDE, maxDistance=PDQ_MAX)))
        led.save(LEDGER)
    print("primary (search)", primary, "| claim", claim, "primary (claim)", claim_primary)


# ----------------------------------------------------------------------------------------------- the verifier
def hexdist(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def pdq_match(copy_hashes, rec_p):
    """PDQ at lower assurance: at least PDQ_FRAC of the record's keyframes find a keyframe of the copy within maxDistance."""
    if not copy_hashes:
        return False, None
    best = [min(hexdist(h, c) for c in copy_hashes) for h in rec_p["hashes"]]
    hit = np.mean([b <= rec_p["maxDistance"] for b in best])
    return bool(hit >= PDQ_FRAC), dict(median=float(np.median(best)), max=int(max(best)), hit=float(hit))


def issuer_key(led):
    """The issuer's signing key as published in the issuer's own DID version on the ledger (falls back to the genesis account)."""
    v = led.did_version(ISSUER_DID, 1)
    return v["doc"]["verificationMethod"][0]["blockchainAccountId"] if v else led.issuer


def check_record(led, rec, height):
    """Step 4. Returns (ok, [(check, passed, detail)])."""
    checks = []
    body = PL.without(rec, "proofs")
    pr = {p["by"]: p for p in rec["proofs"]}
    checks.append(("issuer is trusted", rec["issuer"] == ISSUER_DID and led.did_version(ISSUER_DID, 1) is not None, rec["issuer"]))
    checks.append(("issuer proof valid (key from the issuer's DID on the ledger)", PL.signer_of(body, pr["issuer"]["proofValue"]) == issuer_key(led), ""))
    drv = led.did_version(rec["driver"]["did"], rec["driver"]["epoch"])
    src = led.did_version(rec["source"]["did"], rec["source"]["epoch"]) if rec["source"].get("did") else None
    checks.append(("driver DID version exists", drv is not None, rec["driver"]["did"]))
    if drv is None:
        return False, checks
    checks.append(("driver countersignature valid",
                   PL.signer_of(body, pr["driver"]["proofValue"]) == drv["doc"]["verificationMethod"][0]["blockchainAccountId"], ""))
    for name, v, party in (("driver", drv, rec["driver"]), ("source", src, rec["source"])):
        if v is None:
            continue
        c = v["doc"]["proxyCommitment"]["value"]
        checks.append((f"{name} commitment in the record = commitment in the DID version", c == party["commitment"], c[:12]))
        checks.append((f"{name} controller = issuer", v["doc"]["controller"] == rec["issuer"], ""))
        checks.append((f"{name} version not revoked before the record", v["revoked_block"] is None or height < v["revoked_block"],
                       f"block {height} vs revoked {v['revoked_block']}"))
    if src is not None:
        b = rec["source"]["binding"]
        ok = b["method"] == "owner-signature" and PL.signer_of(dict(photoSha256=rec["source"]["photoSha256"]), b["proofValue"]) == \
            src["doc"]["verificationMethod"][0]["blockchainAccountId"]
        checks.append(("source binding (owner signature over the photo hash)", bool(ok), b["method"]))
    return all(c[1] for c in checks), checks


# LEGACY 1:N verifier (identify the proxy in the video first, then find its record); the paper's 1:1 verifier is proxy_issuer_eval.verify_session.
def verify_video(led, G, cache, video_path, matcher, T, margin, pdq_cache=None):
    """The statement's verify(video). Returns dict(outcome, assurance, driver, source, trace[])."""
    tr = []
    vec, nobs = G.vectors(cache)
    d = vec[matcher]
    tr.append(dict(step="1 extract", detail=f"{matcher}: {nobs[matcher]} usable observations"))
    x, d1, d2, why = decide(d, T, margin)
    if np.isnan(d1):
        return dict(outcome="INCONCLUSIVE", reason="no usable code", trace=tr)
    order = np.argsort(np.where(np.isnan(d), 9, d))[:5]
    tr.append(dict(step="2 find the DID (1:N)", detail=f"closest {G.names[order[0]][0]} d1={d1:.4f}, second {G.names[order[1]][0]} d2={d2:.4f}; T={T}, margin={margin} -> {why}",
                   top=[(G.names[j][0], round(float(d[j]), 4)) for j in order]))
    if x is None:
        return dict(outcome="INCONCLUSIVE", reason=why, trace=tr)
    X = G.names[x][0]
    sha = PL.sha256_hex(Path(video_path).read_bytes())
    pq = pdq_cache if pdq_cache is not None else pdq_hashes(video_path)["hashes"]

    def find(sessions_):
        for s in sessions_:
            r = s["rec"]
            if r["video"]["sha256"] == sha:
                return s, "EXACT", None
            ok, info = pdq_match(pq, r["video"]["perceptual"])
            if ok:
                return s, "REENCODED", info
        return None, None, None
    s, assur, info = find(led.sessions_by_driver(X))
    if s is None:
        s2, assur2, info2 = find(led.sessions_by_source(X))
        if s2 is not None:
            tr.append(dict(step="3 find the session record", detail=f"matches a record whose SOURCE is {X}, not its driver {s2['rec']['driver']['did']}"))
            return dict(outcome="FLAGGED", driver=s2["rec"]["driver"]["did"], source=X, assurance=assur2, trace=tr)
        tr.append(dict(step="3 find the session record", detail=f"{len(led.sessions_by_driver(X))} records with {X} as driver; none matches this video (exact or PDQ)"))
        return dict(outcome="UNREGISTERED", proxy=X, trace=tr)
    tr.append(dict(step="3 find the session record", detail=f"record {s['rec']['sessionId'][:8]} in block {s['height']}: {assur} match" +
                   (f" (PDQ median {info['median']:.0f}, max {info['max']} of 256 bits)" if info else ""), block=s["height"]))
    ok, checks = check_record(led, s["rec"], s["height"])
    tr.append(dict(step="4 check the record", detail="all checks pass" if ok else "a check FAILED", checks=checks))
    if not ok:
        return dict(outcome="REJECT", driver=X, trace=tr)
    r = s["rec"]
    src_did = r["source"].get("did")
    out = "SELF" if src_did == r["driver"]["did"] else "CROSS"
    tr.append(dict(step="5 decide", detail=f"source {src_did or 'unenrolled'}, driver {r['driver']['did']} -> {out} ({assur})"))
    return dict(outcome=out, driver=r["driver"]["did"], source=src_did, assurance=assur, session=r["sessionId"], block=s["height"], trace=tr)


def _find_record(candidates, sha, pq):
    for s_ in candidates:
        r = s_["rec"]
        if r["video"]["sha256"] == sha:
            return s_, "EXACT", None
        ok, info = pdq_match(pq, r["video"]["perceptual"])
        if ok:
            return s_, "REENCODED", info
    return None, None, None


def verify_claim(led, G, cache, video_path, claimed_did, matcher, T1, pdq_cache=None):
    """VERIFICATION of a named DID (the claimed driver): no search over the ledger.
    1 find the claimed DID's session record whose video hash matches the file (exact SHA-256, else PDQ)
    2 check the record (issuer proof, driver countersignature, template hashes, revocation by block order, source binding)
    3 the iris check, 1:1 against the proxies the record names: the video's code must be within T1 of the DRIVER's proxy and closer
      to it than to the source's proxy (SELF: source = driver). Closer to the source's proxy instead: FLAGGED."""
    tr = []
    sha = PL.sha256_hex(Path(video_path).read_bytes())
    pq = pdq_cache if pdq_cache is not None else pdq_hashes(video_path)["hashes"]
    cands = led.sessions_by_driver(claimed_did)
    s_, assur, info = _find_record(cands, sha, pq)
    if s_ is None:
        tr.append(dict(step="1 find the record", detail=f"{len(cands)} records name {claimed_did} as driver; none matches this video (exact or PDQ)"))
        return dict(outcome="UNREGISTERED", claimed=claimed_did, trace=tr)
    tr.append(dict(step="1 find the record", detail=f"record {s_['rec']['sessionId'][:8]} in block {s_['height']}: {assur} match" +
                   (f" (PDQ median {info['median']:.0f}, max {info['max']} of 256 bits)" if info else ""), block=s_["height"]))
    r = s_["rec"]
    ok, checks = check_record(led, r, s_["height"])
    tr.append(dict(step="2 check the record", detail="all checks pass" if ok else "a check FAILED", checks=checks))
    if not ok:
        return dict(outcome="REJECT", claimed=claimed_did, trace=tr)
    G_idx = {n: i for i, n in enumerate(G.names)}
    vec, nobs = G.vectors(cache)
    d = vec[matcher]
    drv_k = G_idx[(r["driver"]["did"], r["driver"]["epoch"])]
    src_did = r["source"].get("did")
    src_k = G_idx.get((src_did, r["source"].get("epoch"))) if src_did and src_did != r["driver"]["did"] else None
    d_drv = float(d[drv_k])
    d_src = float(d[src_k]) if src_k is not None else float("nan")
    tr.append(dict(step="3 iris check (1:1)", detail=f"{matcher}: {nobs[matcher]} usable observations; distance to the driver's proxy {d_drv:.4f}" +
                   (f", to the source's proxy {d_src:.4f}" if src_k is not None else "") + f"; T1={T1:.4f}",
                   top=[(r["driver"]["did"], round(d_drv, 4))] + ([(src_did, round(d_src, 4))] if src_k is not None else [])))
    if not np.isfinite(d_drv):
        return dict(outcome="INCONCLUSIVE", claimed=claimed_did, reason="no usable code", trace=tr)
    if d_drv < T1 and (src_k is None or not np.isfinite(d_src) or d_drv < d_src):
        out = "SELF" if src_did == r["driver"]["did"] else "CROSS"
        tr.append(dict(step="4 decide", detail=f"the video carries the driver's proxy; source {src_did or 'unenrolled'}, driver {r['driver']['did']} -> {out} ({assur})"))
        return dict(outcome=out, driver=r["driver"]["did"], source=src_did, assurance=assur, claimed=claimed_did, session=r["sessionId"], block=s_["height"], trace=tr)
    if src_k is not None and np.isfinite(d_src) and d_src < T1 and d_src < d_drv:
        tr.append(dict(step="4 decide", detail="the video carries the SOURCE's proxy, not the driver's, although the record names the driver"))
        return dict(outcome="FLAGGED", driver=r["driver"]["did"], source=src_did, assurance=assur, claimed=claimed_did, trace=tr)
    tr.append(dict(step="4 decide", detail="the video's code is not within T1 of the driver's proxy"))
    return dict(outcome="INCONCLUSIVE", claimed=claimed_did, reason="code matches no named proxy", trace=tr)


# ----------------------------------------------------------------------------------------------- evaluation
def cp(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def rate(k, n):
    return dict(k=int(k), n=int(n), rate=round(k / n, 3) if n else None, ci95=cp(k, n) if n else None)


def cmd_evaluate(a):
    plan = json.loads(PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(LEDGER)
    calib = json.loads(CALIB.read_text())
    G = Gallery(led)
    pdq_dir = WORK / "pdq"
    pdq_dir.mkdir(exist_ok=True)

    def pq_of(key, path):
        f = pdq_dir / f"{key}.json"
        if not f.exists():
            f.write_text(json.dumps(pdq_hashes(path)["hashes"]))
        return json.loads(f.read_text())

    metas = sessions()
    by_id = {m["id"]: m for m in metas}
    vp = json.loads((WORK / "variant_plan.json").read_text()) if (WORK / "variant_plan.json").exists() else {}
    rng = np.random.default_rng(plan["seed"] + 3)
    dids = [e["did"] for e in enrol.values()]
    keys = []                                       # (key, session meta, variant)
    for m in metas:
        keys.append((m["id"], m, "original"))
        keys.append((m["id"] + "__raw", m, "raw_twin"))
    for kind, ids in vp.items():
        for i in ids:
            if i in by_id:
                keys.append((f"{i}__{kind}", by_id[i], kind))
    out = dict(calibration=calib, results={}, rows={})

    def base(m, key, variant, v):
        return dict(id=m["id"], kind=m["kind"], dataset=m["dataset"], tuning=m["tuning"], source_actor=m["source_actor"], driver_actor=m["driver_actor"],
                    truth_driver=enrol[m["driver_actor"]]["did"], variant=variant, **{k: v.get(k) for k in ("outcome", "assurance", "driver", "source", "reason")})
    # ---------------- claim mode: verification of a named DID (primary)
    cm = calib.get("claim_primary")
    if cm:
        T1 = calib["claim"][cm]["T1"]
        rows, imp_rows = [], []
        for key, m, variant in keys:
            c = load_cache(key)
            if c is None or not clip_path(key).exists():
                continue
            truth = enrol[m["driver_actor"]]["did"]
            v = verify_claim(led, G, c, clip_path(key), truth, cm, T1, pq_of(key, clip_path(key)))
            rows.append(base(m, key, variant, v))
            if variant == "original" and m["kind"] in ("self", "cross"):
                others = [d_ for d_ in dids if d_ != truth]
                for d_ in rng.choice(others, size=3, replace=False):
                    v2 = verify_claim(led, G, c, clip_path(key), str(d_), cm, T1, pq_of(key, clip_path(key)))
                    imp_rows.append(dict(id=m["id"], tuning=m["tuning"], kind=m["kind"], claimed=str(d_), outcome=v2["outcome"]))
        sm = summarise(rows, T1, 0.0)
        test_imp = [r for r in imp_rows if not r["tuning"]]
        sm["impostor_claims"] = dict(n=len(test_imp), accepted_as_self_or_cross=rate(sum(r["outcome"] in ("SELF", "CROSS") for r in test_imp), len(test_imp)),
                                     outcomes=dict(__import__("collections").Counter(r["outcome"] for r in test_imp)))
        out["results"]["claim"] = dict(matcher=cm, T1=T1, **sm)
        out["rows"]["claim"] = rows
        out["claim_attacks"] = ledger_attacks_claim(plan, led, G, calib, cm, T1, metas, pq_of)
    # ---------------- search mode: no DID named (the fallback, a 1:N search over every DID version)
    mt = calib.get("primary")
    if mt:
        T, M = calib["matchers"][mt]["T"], calib["matchers"][mt]["margin"]
        rows = []
        for key, m, variant in keys:
            c = load_cache(key)
            if c is None or not clip_path(key).exists():
                continue
            v = verify_video(led, G, c, clip_path(key), mt, T, M, pq_of(key, clip_path(key)))
            rows.append(base(m, key, variant, v))
        out["results"]["search"] = dict(matcher=mt, **summarise(rows, T, M))
        out["rows"]["search"] = rows
        out["search_attacks"] = ledger_attacks(plan, led, G, calib, mt, metas, pq_of)
    RESULTS.write_text(json.dumps(stamp(out), indent=1))
    for mode, r in out["results"].items():
        print("==", mode, r["matcher"], {k: v for k, v in r.items() if k in ("honest_all", "honest_self", "honest_cross", "control_raw_twin", "impostor_claims")})
    print(json.dumps(out.get("claim_attacks", {}), indent=1))
    print(json.dumps(out.get("search_attacks", {}), indent=1))


def ledger_attacks_claim(plan, led, G, calib, mt, T1, metas, pq_of, n=12):
    """Record-level attacks, verified in claim mode on forks of the ledger (honest test videos)."""
    enrol = plan["enrolment"]
    rng = np.random.default_rng(plan["seed"] + 2)
    cand = [m for m in metas if m["kind"] in ("self", "cross") and not m["tuning"] and load_cache(m["id"]) is not None]
    sel = [cand[i] for i in rng.permutation(len(cand))[:n]]
    out = {}

    def run(name, mutate):
        res = []
        for m in sel:
            L2 = led.fork()
            vid = mutate(L2, m) or clip_path(m["id"])
            cache = load_cache(m["id"]) if vid == clip_path(m["id"]) else load_cache(m["id"] + "__raw")
            v = verify_claim(L2, Gallery(L2), cache, vid, enrol[m["driver_actor"]]["did"], mt, T1, pq_of(m["id"] + ("" if vid == clip_path(m["id"]) else "__raw"), vid))
            res.append(v["outcome"])
        from collections import Counter
        out[name] = dict(n=len(res), outcomes=dict(Counter(res)))

    def blk(L2, m):
        return next(b for b in L2.blocks if b["kind"] == "session" and b["payload"]["sessionId"] == m["session_id"])

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

    def record_for_proxyless_video(L2, m):
        """The issuer recorded a session whose video carries no proxy (the generator failed to embed): a valid record for the raw twin."""
        raw = clip_path(m["id"] + "__raw")
        from proxy_attr import make_record
        row = dict(m)
        rec = make_record(L2, row, enrol, raw, 25.0, row["frames"])
        rec["sessionId"] = rec["sessionId"][:-4] + "ffff"
        body = PL.without(rec, "proofs")
        issuer = PL.account_from(ISSUER_SEED, "issuer")
        rec["proofs"] = [dict(by="issuer", proofValue=PL.sign(issuer, body)), dict(by="driver", proofValue=PL.sign(PL.account_from(ISSUER_SEED, "person|" + m["driver_actor"]), body))]
        L2.record_session(issuer.address, rec)
        return raw
    run("record_missing", record_missing)
    run("forged_source_binding", forged_binding)
    run("driver_revoked_before_record", revoked_before)
    run("record_tampered", tampered)
    run("valid_record_but_no_proxy_in_video", record_for_proxyless_video)
    return out


def summarise(rows, T, M):
    s = dict(T=T, margin=M)
    test = [r for r in rows if not r["tuning"]]

    def sub(sel, name):
        return [r for r in test if sel(r)]
    honest = sub(lambda r: r["variant"] == "original" and r["kind"] in ("self", "cross"), "honest")

    def correct(r):
        want = "SELF" if r["kind"] == "self" else "CROSS"
        return r["outcome"] == want and r["driver"] == r["truth_driver"]

    def wrong(r):
        return r["outcome"] in ("SELF", "CROSS") and not correct(r)
    for name, sel in (("all", lambda r: True), ("self", lambda r: r["kind"] == "self"), ("cross", lambda r: r["kind"] == "cross"),
                      ("ravdess", lambda r: r["dataset"] == "ravdess"), ("nir", lambda r: r["dataset"] == "nir")):
        h = [r for r in honest if sel(r)]
        s[f"honest_{name}"] = dict(n=len(h), correct=rate(sum(correct(r) for r in h), len(h)), wrong=rate(sum(wrong(r) for r in h), len(h)),
                                   inconclusive=rate(sum(r["outcome"] == "INCONCLUSIVE" for r in h), len(h)),
                                   other=rate(sum(r["outcome"] in ("UNREGISTERED", "FLAGGED", "REJECT") for r in h), len(h)))
    from collections import Counter
    s["honest_outcomes"] = dict(Counter(r["outcome"] for r in honest))
    s["by_dataset_and_kind"] = {f"{ds}/{k}": rate(sum(correct(r) for r in honest if r["dataset"] == ds and r["kind"] == k),
                                                  sum(1 for r in honest if r["dataset"] == ds and r["kind"] == k))
                                for ds in ("ravdess", "nir") for k in ("self", "cross")}
    ctl = sub(lambda r: r["variant"] == "raw_twin" and r["kind"] in ("self", "cross"), "ctl")
    s["control_raw_twin"] = dict(n=len(ctl), accepted=rate(sum(r["outcome"] in ("SELF", "CROSS") for r in ctl), len(ctl)),
                                 outcomes=dict(Counter(r["outcome"] for r in ctl)))
    for v, want in (("crf23", None), ("crf30", None), ("blur", "INCONCLUSIVE"), ("alter", None)):
        vr = sub(lambda r: r["variant"] == v, v)
        s[f"variant_{v}"] = dict(n=len(vr), outcomes=dict(Counter(r["outcome"] for r in vr)),
                                 assurance=dict(Counter(r["assurance"] for r in vr if r["assurance"])),
                                 correct=rate(sum(correct(r) for r in vr), len(vr)), wrong=rate(sum(wrong(r) for r in vr), len(vr)))
    for k, want in (("wrong_proxy", "INCONCLUSIVE"), ("unenrolled_proxy", "INCONCLUSIVE")):
        vr = sub(lambda r: r["variant"] == "original" and r["kind"] == k, k)
        s[f"attack_{k}"] = dict(n=len(vr), expected=want, outcomes=dict(Counter(r["outcome"] for r in vr)),
                                as_expected=rate(sum(r["outcome"] == want for r in vr), len(vr)),
                                false_attribution=rate(sum(r["outcome"] in ("SELF", "CROSS") for r in vr), len(vr)))
    tun = [r for r in rows if r["tuning"] and r["variant"] == "original" and r["kind"] in ("self", "cross")]
    s["tuning_honest"] = dict(n=len(tun), correct=rate(sum(correct(r) for r in tun), len(tun)))
    return s


def ledger_attacks(plan, led, G, calib, mt, metas, pq_of, n=12):
    """Record-level attacks on a fork of the ledger, checked by the same verifier (original, honest, test videos)."""
    if mt is None:
        return {}
    T, M = calib["matchers"][mt]["T"], calib["matchers"][mt]["margin"]
    rng = np.random.default_rng(plan["seed"] + 2)
    cand = [m for m in metas if m["kind"] in ("self", "cross") and not m["tuning"] and load_cache(m["id"]) is not None]
    sel = [cand[i] for i in rng.permutation(len(cand))[:n]]
    out = {}

    def run(name, mutate):
        res = []
        for m in sel:
            L2 = led.fork()
            mutate(L2, m)
            G2 = Gallery(L2)
            v = verify_video(L2, G2, load_cache(m["id"]), clip_path(m["id"]), mt, T, M, pq_of(m["id"], clip_path(m["id"])))
            res.append(v["outcome"])
        from collections import Counter
        out[name] = dict(n=len(res), outcomes=dict(Counter(res)))

    def idx_of(L2, m):
        return next(b for b in L2.blocks if b["kind"] == "session" and b["payload"]["sessionId"] == m["session_id"])

    def no_record(L2, m):                # the video was made outside the system: its record never existed
        b = idx_of(L2, m)
        b["kind"] = "void"

    def forged_binding(L2, m):           # the source's owner never signed this photo
        b = idx_of(L2, m)
        b["payload"]["source"]["binding"]["proofValue"] = PL.sign(PL.account_from(ISSUER_SEED, "person|nobody"), dict(photoSha256=b["payload"]["source"]["photoSha256"]))

    def revoked_before(L2, m):           # the driver's key version was revoked before the record's block
        b = idx_of(L2, m)
        L2.blocks.insert(b["height"], dict(height=b["height"], time=0, prev="", kind="revoke", payload=dict(did=b["payload"]["driver"]["did"], epoch=1), hash=""))
        for i, bb in enumerate(L2.blocks):
            bb["height"] = i

    def tampered(L2, m):                 # the session record was edited after signing
        b = idx_of(L2, m)
        b["payload"]["nonce"] = "0" * 32
    run("record_missing", no_record)
    run("forged_source_binding", forged_binding)
    run("driver_revoked_before_record", revoked_before)
    run("record_tampered", tampered)
    return out


# ----------------------------------------------------------------------------------------------- 1:1 when the DID is carried
def cmd_onetoone(a):
    """If the session's DID is carried in the video (a watermark payload) the verifier knows WHICH proxy to check: no 1:N search.
    Assuming a correct decode, this scores the 1:1 check "does the video carry the proxy of the claimed DID?" from the same caches.
    Threshold per matcher = the tuning impostor distance quantile at the target FAR (impostors: every OTHER DID's proxy against the
    clip); applied unchanged to the test sessions. Genuine = the true driver's DID; controls = the un-embedded twin claiming it."""
    plan = json.loads(PLAN.read_text())
    enrol = plan["enrolment"]
    led = PL.Ledger.load(LEDGER)
    G = Gallery(led)
    didx = {n[0]: i for i, n in enumerate(G.names)}
    metas = [m for m in sessions() if m["kind"] in ("self", "cross")]
    out = dict(note="assumes the DID was decoded correctly from the video; the decode itself (watermark) is measured separately",
               matchers={})
    for mt in MATCHERS:
        recs = []
        for m in metas:
            c = load_cache(m["id"])
            if c is None:
                continue
            d = G.vectors(c)[0][mt]
            k = didx[enrol[m["driver_actor"]]["did"]]
            raw = load_cache(m["id"] + "__raw")
            dr = G.vectors(raw)[0][mt][k] if raw is not None else np.nan
            imp = np.delete(d, k)
            recs.append(dict(id=m["id"], tuning=m["tuning"], dataset=m["dataset"], kind=m["kind"], gen=float(d[k]), imp=imp[np.isfinite(imp)], ctl=float(dr)))
        tun = [r for r in recs if r["tuning"]]
        test = [r for r in recs if not r["tuning"]]
        if not tun or not test:
            continue
        timp = np.concatenate([r["imp"] for r in tun])
        res = {}
        for far in (1e-2, 1e-3):
            T = float(np.quantile(timp, far))
            sel = lambda rs, f=lambda r: True: [r for r in rs if f(r)]
            gen = [r for r in test]
            tar = sum(1 for r in gen if np.isfinite(r["gen"]) and r["gen"] < T)
            imp_all = np.concatenate([r["imp"] for r in test])
            far_k = int((imp_all < T).sum())
            ctl = [r for r in test if np.isfinite(r["ctl"])]
            ctl_k = sum(1 for r in ctl if r["ctl"] < T)
            res[f"far_{far:g}"] = dict(T=round(T, 4), genuine=rate(tar, len(gen)),
                                       by_dataset={ds: rate(sum(1 for r in gen if r["dataset"] == ds and np.isfinite(r["gen"]) and r["gen"] < T),
                                                           sum(1 for r in gen if r["dataset"] == ds)) for ds in ("ravdess", "crema")},
                                       by_kind={k_: rate(sum(1 for r in gen if r["kind"] == k_ and np.isfinite(r["gen"]) and r["gen"] < T),
                                                         sum(1 for r in gen if r["kind"] == k_)) for k_ in ("self", "cross")},
                                       impostor_other_did=rate(far_k, len(imp_all)), control_raw_twin=rate(ctl_k, len(ctl)))
        out["matchers"][mt] = res
    (ROOT / "outputs" / "proxy_attr_1to1.json").write_text(json.dumps(stamp(out), indent=1))
    for mt, r in out["matchers"].items():
        for f, v in r.items():
            print(mt, f, "T", v["T"], "genuine", pct1(v["genuine"]), "| impostor", pct1(v["impostor_other_did"]), "| control", pct1(v["control_raw_twin"]))


def pct1(d):
    return f"{d['k']}/{d['n']}={100 * d['rate']:.1f}% {d['ci95']}" if d and d.get("n") else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["variants", "extract", "calibrate", "evaluate", "onetoone"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--raw-first", type=int, default=0, help="extract: un-embedded twins of batch-4 sessions only for the first N plan rows")
    a = ap.parse_args()
    {"variants": cmd_variants, "extract": cmd_extract, "calibrate": cmd_calibrate, "evaluate": cmd_evaluate, "onetoone": cmd_onetoone}[a.cmd](a)


if __name__ == "__main__":
    main()
