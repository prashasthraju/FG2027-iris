# LEGACY: part of an earlier design (public proxy codes / 1:N identification of the proxy in the video / local Anvil chain). The submitted protocol
# (commit-and-reveal, 1:1 check against the opened proxy) is implemented in tools/proxy_issuer_eval.py (verify_session) and tools/proxy_ledger.py; see README.md.
"""Cancelable iris identities: enrol, revoke, verify.

Roles, all simulated locally for the research prototype:
  platform  owns the iris pool and the allocation table (off-chain, private);
  holder    owns a fresh did:ethr key and a salt (an encrypted wallet file here);
  ledger    the local Anvil chain: one salted commitment per active identity.

Enrol:   new secp256k1 key -> did:ethr; a random *unused* pool entry (CSPRNG);
         salt = 128 random bits; commitment c = keccak256("iris-pool/v1" | sid | salt);
         the holder signs setAttribute("iris/commit/v1", c) and the platform relays it.
Revoke:  the holder signs revokeAttribute(c); the entry is RETIRED, never reassigned,
         so an old video can never verify to a new holder (R1). Re-enrolment = new key,
         new entry, new salt: nothing links the two identities on-chain (R2).
Verify:  1. identify the pool entry carried by the media (identify.py, D25);
         2. challenge: the claimant signs a fresh verifier nonce with the DID key;
         3. the claimant reveals the salt; the verifier recomputes c from the
            *identified* entry and checks it is an ACTIVE attribute of that DID.
Nothing biometric, and no pool index, is ever written on-chain: with ~79 entries an
unsalted hash of the index would be inverted in 79 guesses.
"""
import json
import secrets
import time
from pathlib import Path

from eth_account import Account
from web3 import Web3

from .ledger import STATE, Ledger

ATTR = "iris/commit/v1"
ALLOC = STATE / "allocations.json"
WALLETS = STATE / "wallets"
PASSPHRASE = STATE / ".passphrase"
LOG = STATE / "events.jsonl"


def commitment(sid: str, salt: bytes) -> bytes:
    return Web3.solidity_keccak(["string", "string", "bytes16"], ["iris-pool/v1|", sid, salt])


SCREEN = Path(__file__).resolve().parent.parent / "outputs" / "pool_screen.json"
SCREEN_RATIO, HUB_MIN_WRONG = 0.94, 2


def screened_entries(max_ratio=None):
    """Entries that survive reenactor output: fused ratio < max_ratio (default 0.94) on the
    screening clip, minus "hub" entries that attract wrong matches (fused top for >= 2 other entries).
    D29: PersonaLive's softer output needs the strict screen, max_ratio 0.90."""
    max_ratio = SCREEN_RATIO if max_ratio is None else max_ratio
    if not SCREEN.exists():
        return set()
    rows = json.loads(SCREEN.read_text())["rows"]
    wrong_tops = [r["fused_top"] for r in rows if r["fused_top"] != r["entry"]]
    hubs = {e for e in wrong_tops if wrong_tops.count(e) >= HUB_MIN_WRONG}
    return {r["entry"] for r in rows if r["ratio"] < max_ratio and r["fused_top"] == r["entry"]} - hubs


def _passphrase():
    if not PASSPHRASE.exists():
        STATE.mkdir(parents=True, exist_ok=True)
        PASSPHRASE.write_text(secrets.token_urlsafe(32))
        PASSPHRASE.chmod(0o600)
    return PASSPHRASE.read_text()


class Platform:
    def __init__(self, pool, ledger=None, screened_only=False, screen_ratio=None):
        self.pool = pool
        self.screened_only = screened_only
        self.screen_ratio = screen_ratio
        self.ledger = ledger or Ledger()
        self.alloc = json.loads(ALLOC.read_text()) if ALLOC.exists() else {}

    # ---- bookkeeping ------------------------------------------------------
    def _save(self):
        ALLOC.write_text(json.dumps(self.alloc, indent=1))

    def _log(self, **kw):
        with LOG.open("a") as f:
            f.write(json.dumps(dict(t=round(time.time(), 3), **kw)) + "\n")

    def free_entries(self):
        ids = self.pool.ids
        if self.screened_only:
            ids = [s for s in ids if s in screened_entries(self.screen_ratio)]
        return [sid for sid in ids if sid not in self.alloc]

    # ---- holder wallet (stands in for the user's own device) --------------
    @staticmethod
    def save_wallet(did, key, salt, label):
        WALLETS.mkdir(parents=True, exist_ok=True)
        f = WALLETS / f"{did.rsplit(':', 1)[1]}.json"
        f.write_text(json.dumps(dict(did=did, label=label, salt=salt.hex(),
                                     keystore=Account.encrypt(key, _passphrase()))))
        f.chmod(0o600)
        return f

    @staticmethod
    def load_wallet(did):
        w = json.loads((WALLETS / f"{did.rsplit(':', 1)[1]}.json").read_text())
        return w, Account.decrypt(w["keystore"], _passphrase()), bytes.fromhex(w["salt"])

    # ---- enrol / revoke ---------------------------------------------------
    def enroll(self, label):
        free = self.free_entries()
        if not free:
            raise RuntimeError("iris pool exhausted: every entry is assigned or retired")
        sid = secrets.SystemRandom().choice(free)
        acct, salt = Account.create(), secrets.token_bytes(16)
        did = self.ledger.did(acct.address)
        tx = self.ledger.set_attribute(acct.key, ATTR, commitment(sid, salt))
        self.alloc[sid] = dict(status="assigned", did=did, label=label, t=time.time())
        self._save()
        self.save_wallet(did, acct.key, salt, label)
        self._log(op="enroll", did=did, label=label, tx=tx)
        return did, sid

    def revoke(self, did):
        w, key, salt = self.load_wallet(did)
        sid = next(s for s, a in self.alloc.items() if a["did"] == did)
        tx = self.ledger.revoke_attribute(key, ATTR, commitment(sid, salt))
        self.alloc[sid]["status"] = "retired"
        self._save()
        self._log(op="revoke", did=did, tx=tx)
        return tx

    def entry_of(self, did):
        """Platform-side lookup used only to *render*: which strips to embed for this DID."""
        return next(s for s, a in self.alloc.items() if a["did"] == did and a["status"] == "assigned")

    # ---- verification -----------------------------------------------------
    def holder_response(self, did, nonce):
        """What the claimant's device returns to a verifier: a signature over the nonce and the salt."""
        w, key, salt = self.load_wallet(did)
        return dict(signature=self.ledger.sign_challenge(key, nonce), salt=salt.hex())

    def verify(self, identified_sid, did, nonce, response):
        """Verifier side. Uses only public ledger state plus what the claimant revealed."""
        steps = {}
        steps["iris_matched_pool_entry"] = identified_sid is not None
        steps["controls_did_key"] = self.ledger.check_challenge(did, nonce, response["signature"])
        active = self.ledger.attributes(did, ATTR)
        steps["did_has_active_commitment"] = bool(active)
        ok_c = identified_sid is not None and bytes(commitment(identified_sid, bytes.fromhex(response["salt"]))) in active
        steps["commitment_matches_identified_iris"] = ok_c
        verdict = all(steps.values())
        self._log(op="verify", did=did, sid_seen=identified_sid, verdict=verdict, steps=steps)
        return verdict, steps
