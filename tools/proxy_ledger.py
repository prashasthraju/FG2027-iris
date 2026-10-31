"""Simulated append-only ledger plus the issuer, DID-version and session-record logic of the proxy-iris design
(new_problem_statement.md, README.md). A simulation: one process, one JSON file, secp256k1 keys derived
from a seed (no real funds, no real people), but the rules are the contract's:

  * only the issuer account can write;
  * `record_session` refuses a record without a valid issuer proof and a valid driver countersignature;
  * every write is one block (height, synthetic block time, previous hash, own hash), so revocation order and "which
    block holds which record" are decided by the chain, not by a timestamp in the record.

Run with python (eth_account).
"""
import base64
import hashlib
import hmac
import json
from pathlib import Path

from eth_account import Account
from eth_account.messages import encode_defunct

T0 = 1793000000          # synthetic block clock: 2026-10-26, 5 s per block
BLOCK_S = 5


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def keccak_hex(b):
    """Keccak-256 (the Ethereum hash) as 0x-hex."""
    from eth_utils import keccak
    return "0x" + keccak(b if isinstance(b, bytes) else b.encode()).hex()


def commit(templates, salt_hex):
    """Salted commitment to a proxy template: keccak256(canonical(template) || salt). Only this goes on the ledger, never the template."""
    return keccak_hex(canon(templates).encode() + bytes.fromhex(salt_hex))


def account_from(seed, name):
    return Account.from_key(hashlib.sha256(f"{seed}|{name}".encode()).digest())


def sign(acct, obj):
    return Account.sign_message(encode_defunct(text=canon(obj)), acct.key).signature.hex()


def signer_of(obj, sig):
    try:
        return Account.recover_message(encode_defunct(text=canon(obj)), signature=sig)
    except Exception:  # noqa: BLE001 - malformed signature
        return None


def without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def hmac_index(k_idx, subject_id, epoch, probe, M):
    """The statement's assignment: idx = HMAC(K_idx, subject_id || epoch || k) mod M, probing k until the index is free."""
    h = hmac.new(k_idx, f"{subject_id}|{epoch}|{probe}".encode(), hashlib.sha256).digest()
    return int.from_bytes(h, "big") % M


class Ledger:
    def __init__(self, issuer_addr):
        self.issuer = issuer_addr
        self.blocks = []

    # ------------------------------------------------------------------ chain
    def _append(self, kind, payload):
        h = len(self.blocks)
        b = dict(height=h, time=T0 + BLOCK_S * h, prev=self.blocks[-1]["hash"] if self.blocks else "0" * 64, kind=kind,
                 payload=payload)
        b["hash"] = sha256_hex(canon(b))
        self.blocks.append(b)
        return h

    def chain_ok(self):
        prev = "0" * 64
        for b in self.blocks:
            if b["prev"] != prev or b["hash"] != sha256_hex(canon(without(b, "hash"))):
                return False
            prev = b["hash"]
        return True

    def _require_issuer(self, sender):
        if sender != self.issuer:
            raise PermissionError("only the trusted issuer account may write")

    # ------------------------------------------------------------------ writes
    def set_parameters(self, sender, params):
        self._require_issuer(sender)
        return self._append("params", params)

    def register_did_version(self, sender, did, epoch, doc):
        self._require_issuer(sender)
        return self._append("did", dict(did=did, epoch=epoch, doc=doc))

    def revoke_did_version(self, sender, did, epoch):
        self._require_issuer(sender)
        return self._append("revoke", dict(did=did, epoch=epoch))

    def record_session(self, sender, rec):
        self._require_issuer(sender)
        proofs = {p["by"]: p for p in rec["proofs"]}
        body = without(rec, "proofs")
        if signer_of(body, proofs["issuer"]["proofValue"]) != self.issuer:
            raise ValueError("invalid issuer proof")
        dv = self.did_version(rec["driver"]["did"], rec["driver"]["epoch"])
        if dv is None or signer_of(body, proofs["driver"]["proofValue"]) != dv["doc"]["verificationMethod"][0]["blockchainAccountId"]:
            raise ValueError("invalid driver countersignature")
        return self._append("session", rec)

    # ------------------------------------------------------------------ reads
    def parameters_at(self, height=None):
        ps = [b for b in self.blocks if b["kind"] == "params" and (height is None or b["height"] <= height)]
        return ps[-1]["payload"] if ps else None

    def all_did_versions(self):
        rev = {(b["payload"]["did"], b["payload"]["epoch"]): b["height"] for b in self.blocks if b["kind"] == "revoke"}
        return [dict(did=b["payload"]["did"], epoch=b["payload"]["epoch"], doc=b["payload"]["doc"], height=b["height"],
                     revoked_block=rev.get((b["payload"]["did"], b["payload"]["epoch"]))) for b in self.blocks if b["kind"] == "did"]

    def did_version(self, did, epoch):
        for v in self.all_did_versions():
            if v["did"] == did and v["epoch"] == epoch:
                return v
        return None

    def sessions_by_driver(self, did):
        return [dict(height=b["height"], rec=b["payload"]) for b in self.blocks if b["kind"] == "session" and b["payload"]["driver"]["did"] == did]

    def sessions_by_source(self, did):
        return [dict(height=b["height"], rec=b["payload"]) for b in self.blocks if b["kind"] == "session" and b["payload"]["source"].get("did") == did]

    def session(self, session_id):
        for b in self.blocks:
            if b["kind"] == "session" and b["payload"]["sessionId"] == session_id:
                return dict(height=b["height"], rec=b["payload"])
        return None

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        Path(path).write_text(json.dumps(dict(issuer=self.issuer, blocks=self.blocks)))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        L = cls(d["issuer"])
        L.blocks = d["blocks"]
        return L

    def fork(self):
        L = Ledger(self.issuer)
        L.blocks = json.loads(json.dumps(self.blocks))
        return L


def b64(b):
    return base64.b64encode(b).decode()
