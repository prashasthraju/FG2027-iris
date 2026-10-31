"""Tier 0: the protocol rules on synthetic data (no images, no GPU, seconds).

    python tests/test_protocol.py        # or: pytest -q tests

Checks, against the ledger and record logic that produced the paper's results (tools/proxy_ledger.py, tools/proxy_verify.py):
  commit/open with a salt, wrong template or salt rejected, a bare hash is not the commitment, only the issuer writes, a record needs a valid issuer proof and a
  valid driver countersignature, a tampered record fails, a commitment that differs from the DID version fails, a forged source binding fails, a DID version revoked
  in an earlier block fails, the chain detects a rewritten block.
"""
import copy
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402
from proxy_attr import ISSUER_DID  # noqa: E402

SEED = "synthetic-test-seed"
TEMPLATE = {"image_left": {"arciris": [0.1, 0.2, 0.3]}, "image_right": {"arciris": [0.3, 0.2, 0.1]}}     # a stand-in template: any JSON value
OTHER_TEMPLATE = {"image_left": {"arciris": [0.9, 0.2, 0.3]}, "image_right": {"arciris": [0.3, 0.2, 0.1]}}


def person_doc(acct, template, salt):
    return dict(controller=ISSUER_DID, verificationMethod=[dict(blockchainAccountId=acct.address)],
                proxyCommitment=dict(scheme="keccak256(canonical template || salt)", epoch=1, value=PL.commit(template, salt)))


def world():
    issuer = PL.account_from(SEED, "issuer")
    drv, src = PL.account_from(SEED, "person|driver"), PL.account_from(SEED, "person|source")
    salts = {"driver": os.urandom(32).hex(), "source": os.urandom(32).hex()}
    led = PL.Ledger(issuer.address)
    led.register_did_version(issuer.address, ISSUER_DID, 1, dict(verificationMethod=[dict(blockchainAccountId=issuer.address)]))
    led.register_did_version(issuer.address, "did:example:driver", 1, person_doc(drv, TEMPLATE, salts["driver"]))
    led.register_did_version(issuer.address, "did:example:source", 1, person_doc(src, OTHER_TEMPLATE, salts["source"]))
    photo = hashlib.sha256(b"photo").hexdigest()
    rec = dict(sessionId="s-1", timestamp="block", nonce="n", issuer=ISSUER_DID,
               source=dict(did="did:example:source", epoch=1, commitment=PL.commit(OTHER_TEMPLATE, salts["source"]), photoSha256=photo,
                           binding=dict(method="owner-signature", proofValue=PL.sign(src, dict(photoSha256=photo)))),
               driver=dict(did="did:example:driver", epoch=1, commitment=PL.commit(TEMPLATE, salts["driver"])),
               video=dict(sha256=hashlib.sha256(b"video").hexdigest(), perceptual=dict(hashes=[], stride=10, maxDistance=10)))
    body = PL.without(rec, "proofs")
    rec["proofs"] = [dict(by="issuer", proofValue=PL.sign(issuer, body)), dict(by="driver", proofValue=PL.sign(drv, body))]
    return dict(issuer=issuer, drv=drv, src=src, salts=salts, led=led, rec=rec, photo=photo)


def test_commitment_opens_only_with_template_and_salt():
    w = world()
    c = w["led"].did_version("did:example:driver", 1)["doc"]["proxyCommitment"]["value"]
    assert PL.commit(TEMPLATE, w["salts"]["driver"]) == c
    assert PL.commit(OTHER_TEMPLATE, w["salts"]["driver"]) != c            # wrong proxy opened
    assert PL.commit(TEMPLATE, os.urandom(32).hex()) != c                   # wrong salt
    assert PL.keccak_hex(PL.canon(TEMPLATE)) != c                           # a bare hash of the template is not the commitment (it would be guessable over a small pool)


def test_only_the_issuer_writes():
    w = world()
    try:
        w["led"].record_session(w["drv"].address, w["rec"])
    except PermissionError:
        return
    raise AssertionError("a non-issuer wrote to the ledger")


def test_honest_record_passes_all_checks():
    w = world()
    h = w["led"].record_session(w["issuer"].address, w["rec"])
    ok, checks = V.check_record(w["led"], w["rec"], h)
    assert ok, [c for c in checks if not c[1]]


def test_record_needs_valid_proofs():
    w = world()
    bad = copy.deepcopy(w["rec"])
    bad["proofs"][1] = dict(by="driver", proofValue=PL.sign(w["src"], PL.without(bad, "proofs")))            # countersigned by the wrong key
    try:
        w["led"].record_session(w["issuer"].address, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a record with a wrong driver countersignature was accepted")
    bad = copy.deepcopy(w["rec"])
    bad["proofs"][0] = dict(by="issuer", proofValue=PL.sign(w["drv"], PL.without(bad, "proofs")))             # not signed by the issuer
    try:
        w["led"].record_session(w["issuer"].address, bad)
    except ValueError:
        return
    raise AssertionError("a record without a valid issuer proof was accepted")


def test_tampered_record_is_rejected():
    w = world()
    h = w["led"].record_session(w["issuer"].address, w["rec"])
    t = copy.deepcopy(w["rec"])
    t["video"]["sha256"] = hashlib.sha256(b"another video").hexdigest()                                     # an edit after signing
    ok, _ = V.check_record(w["led"], t, h)
    assert not ok


def test_commitment_must_match_the_did_version():
    w = world()
    t = copy.deepcopy(w["rec"])
    t["driver"]["commitment"] = PL.commit(OTHER_TEMPLATE, w["salts"]["driver"])
    body = PL.without(t, "proofs")
    t["proofs"] = [dict(by="issuer", proofValue=PL.sign(w["issuer"], body)), dict(by="driver", proofValue=PL.sign(w["drv"], body))]
    h = w["led"].record_session(w["issuer"].address, t)
    ok, checks = V.check_record(w["led"], t, h)
    assert not ok and any("commitment in the record" in c[0] and not c[1] for c in checks)


def test_forged_source_binding_is_rejected():
    w = world()
    t = copy.deepcopy(w["rec"])
    t["source"]["binding"]["proofValue"] = PL.sign(w["drv"], dict(photoSha256=w["photo"]))                  # signed by someone who does not own the photo
    body = PL.without(t, "proofs")
    t["proofs"] = [dict(by="issuer", proofValue=PL.sign(w["issuer"], body)), dict(by="driver", proofValue=PL.sign(w["drv"], body))]
    h = w["led"].record_session(w["issuer"].address, t)
    ok, _ = V.check_record(w["led"], t, h)
    assert not ok


def test_revocation_is_ordered_by_block():
    w = world()
    before = w["led"].record_session(w["issuer"].address, w["rec"])
    w["led"].revoke_did_version(w["issuer"].address, "did:example:driver", 1)
    ok, _ = V.check_record(w["led"], w["rec"], before)
    assert ok                                                                                               # recorded before the revocation: old videos still verify
    after = copy.deepcopy(w["rec"])
    after["sessionId"] = "s-2"
    body = PL.without(after, "proofs")
    after["proofs"] = [dict(by="issuer", proofValue=PL.sign(w["issuer"], body)), dict(by="driver", proofValue=PL.sign(w["drv"], body))]
    h2 = w["led"].record_session(w["issuer"].address, after)
    ok, _ = V.check_record(w["led"], after, h2)
    assert not ok                                                                                           # recorded after the revocation: rejected


def test_chain_detects_a_rewritten_block():
    w = world()
    w["led"].record_session(w["issuer"].address, w["rec"])
    assert w["led"].chain_ok()
    w["led"].blocks[2]["payload"]["doc"]["proxyCommitment"]["value"] = "0x" + "00" * 32
    assert not w["led"].chain_ok()


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in tests:
        f()
        print("ok  ", f.__name__)
    print(len(tests), "protocol checks passed")
