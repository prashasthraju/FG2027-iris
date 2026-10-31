"""Ledger v2: commit-and-reveal .

The public-proxy ledger of the first sessions (outputs/proxy_attr/ledger.json, kept as ledger_v1_public_proxy.json) carries every person's proxy template in
their DID version. This script re-issues the SAME ledger without any proxy code:

  * every DID version holds `proxyCommitment` = keccak256(canonical(template) || salt) instead of `proxyIris`; the salt is a 32-byte secret per (person, epoch);
  * the template and the salt move to the issuer's PRIVATE registry (outputs/proxy_attr/registry.json, git-ignored; the analysis harness reads it as "the issuer's registry");
  * every session record cites `commitment` for driver and source instead of the unsalted `templateSha256` (a hash over a 79-entry public pool would be brute-forceable);
    the issuer and driver proofs are re-signed over the new body; video hashes, session ids, nonces and the source bindings are unchanged;
  * the block order and heights are unchanged, so every `ledger_block` of the session files stays valid; a final parameter block publishes the commit-and-reveal scheme and T1.

    python tools/proxy_commit_migrate.py
"""
import hashlib
import hmac
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import proxy_ledger as PL  # noqa: E402
from proxy_attr import ISSUER_DID, ISSUER_SEED, LEDGER, WORK  # noqa: E402

V1 = WORK / "ledger_v1_public_proxy.json"
REGISTRY = WORK / "registry.json"


def salt_for(did, epoch):
    """A 32-byte secret per (person, epoch); derived from the issuer secret so that the simulation is reproducible (a deployment draws it at random)."""
    return hmac.new(ISSUER_SEED.encode(), f"salt|{did}|{epoch}".encode(), hashlib.sha256).hexdigest()


def main():
    if not V1.exists():
        shutil.copy(LEDGER, V1)
    old = PL.Ledger.load(V1)
    names = {}
    for f in ("proxy_attr_plan.json", "proxy_attr_plan4.json"):
        for n, e in json.loads((ROOT / "outputs" / f).read_text())["enrolment"].items():
            names[e["did"]] = n
    issuer = PL.account_from(ISSUER_SEED, "issuer")
    new = PL.Ledger(issuer.address)
    registry, commit_of = {}, {}
    for b in old.blocks:
        k, pl = b["kind"], b["payload"]
        if k == "did" and "proxyIris" in pl["doc"]:
            did, ep = pl["did"], pl["epoch"]
            tpl = pl["doc"]["proxyIris"]["templates"]
            salt = salt_for(did, ep)
            c = PL.commit(tpl, salt)
            doc = {kk: vv for kk, vv in pl["doc"].items() if kk not in ("proxyIris", "proof")}
            doc["proxyCommitment"] = dict(scheme="keccak256(canonical template || salt)", epoch=ep, value=c)
            doc["proof"] = dict(type="EcdsaSecp256k1Signature", verificationMethod=ISSUER_DID + "#key-1", proofValue=PL.sign(issuer, doc))
            new._append("did", dict(did=did, epoch=ep, doc=doc))
            registry[f"{did}|{ep}"] = dict(did=did, epoch=ep, name=names.get(did), templates=tpl, salt=salt, commitment=c, extractor=pl["doc"]["proxyIris"].get("extractorVersion"))
            commit_of[(did, ep)] = c
        elif k == "session":
            rec = json.loads(json.dumps(pl))
            for party in ("source", "driver"):
                r = rec[party]
                if r.get("did"):
                    r["commitment"] = commit_of[(r["did"], r["epoch"])]
                r.pop("templateSha256", None)
            body = PL.without(rec, "proofs")
            dname = names[rec["driver"]["did"]]
            rec["proofs"] = [dict(by="issuer", proofValue=PL.sign(issuer, body)), dict(by="driver", proofValue=PL.sign(PL.account_from(ISSUER_SEED, "person|" + dname), body))]
            new.record_session(issuer.address, rec)
        else:
            new._append(k, pl)
    calib = json.loads((ROOT / "outputs" / "proxy_attr_calibration.json").read_text())
    cm = calib["claim_primary"]
    new.set_parameters(issuer.address, dict(version=3, issuer=ISSUER_DID, scheme="commit-and-reveal (proxy codes are never on the ledger)", extractorVersion="irislg-0.1",
                                            primaryMatcher=cm, T1=calib["claim"][cm]["T1"], perceptual=dict(algorithm="pdq-per-keyframe", stride=10, maxDistance=10),
                                            note="T1 frozen from the batch-1 tuning sessions before the later sessions were generated; published here when the ledger was migrated"))
    assert len(new.blocks) == len(old.blocks) + 1 and new.chain_ok()
    new.save(LEDGER)
    REGISTRY.write_text(json.dumps(registry))
    n_s = sum(1 for b in new.blocks if b["kind"] == "session")
    print("ledger v2:", len(new.blocks), "blocks,", n_s, "session records, chain ok", new.chain_ok(), "| registry:", len(registry), "proxies ->", REGISTRY)
    leaked = any("proxyIris" in json.dumps(b["payload"]) or "templates" in json.dumps(b["payload"]) for b in new.blocks)
    print("proxy code anywhere on the new ledger:", leaked)


if __name__ == "__main__":
    main()
