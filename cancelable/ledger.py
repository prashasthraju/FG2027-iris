# LEGACY: part of an earlier design (public proxy codes / 1:N identification of the proxy in the video / local Anvil chain). The submitted protocol
# (commit-and-reveal, 1:1 check against the opened proxy) is implemented in tools/proxy_issuer_eval.py (verify_session) and tools/proxy_ledger.py; see README.md.
"""did:ethr on a local Anvil chain: deploy the ERC-1056 registry, write signed
attributes, and resolve a DID by replaying its event log.

Nothing biometric goes on-chain. The only attribute we write is a salted commitment
to a pool entry; the registry itself stores owners, delegates, a per-identity
`changed` block pointer and a signature nonce, and emits attributes as events.

Identity keys are fresh secp256k1 keys (never Anvil's public dev accounts). They sign
each change; a separate *relayer* account (an Anvil dev account) submits and pays gas
through the registry's `*Signed` functions, so identities never hold ETH.
"""
import json
import time
from pathlib import Path

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

CHAIN = Path(__file__).resolve().parent.parent / "chain"
ARTIFACT = CHAIN / "out" / "EthereumDIDRegistry.sol" / "EthereumDIDRegistry.json"
STATE = CHAIN / "state"
DEPLOYMENT = STATE / "deployment.json"
RPC = "http://127.0.0.1:8545"
CHAIN_ID = 31337
# Anvil's first dev account: a well-known public key, used ONLY as the gas-paying relayer
RELAYER_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
YEAR = 365 * 24 * 3600


def b32(name: str) -> bytes:
    raw = name.encode()
    assert len(raw) <= 32, name
    return raw.ljust(32, b"\0")


class Ledger:
    def __init__(self, rpc=RPC):
        self.w3 = Web3(Web3.HTTPProvider(rpc))
        if not self.w3.is_connected():
            raise ConnectionError(f"no chain at {rpc}; start it with chain/anvil.sh")
        self.relayer = Account.from_key(RELAYER_KEY)
        art = json.loads(ARTIFACT.read_text())
        self.abi, self.bytecode = art["abi"], art["bytecode"]["object"]
        self.registry = self._registry()

    # ---- deployment -------------------------------------------------------
    def _registry(self):
        if DEPLOYMENT.exists():
            dep = json.loads(DEPLOYMENT.read_text())
            if self.w3.eth.get_code(dep["address"]):        # still there (state persisted)
                return self.w3.eth.contract(address=dep["address"], abi=self.abi)
        c = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        rcpt = self._send(c.constructor())
        STATE.mkdir(parents=True, exist_ok=True)
        DEPLOYMENT.write_text(json.dumps(dict(address=rcpt.contractAddress, tx=rcpt.transactionHash.hex(),
                                              block=rcpt.blockNumber, chain_id=CHAIN_ID), indent=1))
        return self.w3.eth.contract(address=rcpt.contractAddress, abi=self.abi)

    def _send(self, fn):
        base = {"from": self.relayer.address, "chainId": CHAIN_ID,
                "nonce": self.w3.eth.get_transaction_count(self.relayer.address)}
        # exact estimates can fall short around the ecrecover precompile (63/64 rule): +25 %
        tx = fn.build_transaction({**base, "gas": int(fn.estimate_gas({"from": base["from"]}) * 1.25)})
        signed = self.relayer.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h)
        if rcpt.status != 1:
            raise RuntimeError(f"transaction reverted: {h.hex()}")
        return rcpt

    @staticmethod
    def did(address: str) -> str:
        return f"did:ethr:0x{CHAIN_ID:x}:{Web3.to_checksum_address(address)}"

    @staticmethod
    def address_of(did: str) -> str:
        return Web3.to_checksum_address(did.rsplit(":", 1)[1])

    # ---- signed attribute writes (ERC-1056 *Signed variants) --------------
    def _sig(self, identity_key, payload: bytes):
        """The registry's signing scheme: keccak(0x19 0x00 registry nonce identity payload),
        signed raw (no Ethereum message prefix)."""
        ident = Account.from_key(identity_key).address
        owner = self.registry.functions.identityOwner(ident).call()
        n = self.registry.functions.nonce(owner).call()
        h = Web3.solidity_keccak(["bytes1", "bytes1", "address", "uint256", "address", "bytes"],
                                 [b"\x19", b"\x00", self.registry.address, n, ident, payload])
        s = Account.unsafe_sign_hash(h, identity_key)
        return ident, s.v, s.r.to_bytes(32, "big"), s.s.to_bytes(32, "big")

    def set_attribute(self, identity_key, name, value: bytes, validity=YEAR):
        payload = (b"setAttribute" + b32(name) + value + validity.to_bytes(32, "big"))
        ident, v, r, s = self._sig(identity_key, payload)
        rcpt = self._send(self.registry.functions.setAttributeSigned(ident, v, r, s, b32(name), value, validity))
        return rcpt.transactionHash.hex()

    def revoke_attribute(self, identity_key, name, value: bytes):
        payload = b"revokeAttribute" + b32(name) + value
        ident, v, r, s = self._sig(identity_key, payload)
        rcpt = self._send(self.registry.functions.revokeAttributeSigned(ident, v, r, s, b32(name), value))
        return rcpt.transactionHash.hex()

    # ---- resolution -------------------------------------------------------
    def attributes(self, did: str, name: str):
        """Active values of attribute `name`: replay DIDAttributeChanged backwards from
        changed[identity] via previousChange, then apply events oldest-first so a later
        revocation (validTo = 0) or expiry cancels an earlier set."""
        ident = self.address_of(did)
        blk, events = self.registry.functions.changed(ident).call(), []
        while blk:
            logs = self.registry.events.DIDAttributeChanged().get_logs(
                from_block=blk, to_block=blk, argument_filters={"identity": ident})
            if not logs:
                break
            events += logs
            blk = logs[0].args.previousChange
        now = self.w3.eth.get_block("latest").timestamp
        active = {}
        for e in sorted(events, key=lambda e: (e.blockNumber, e.logIndex)):
            if e.args.name.rstrip(b"\0").decode() != name:
                continue
            key = bytes(e.args.value)
            if e.args.validTo > now:
                active[key] = e.args.validTo
            else:
                active.pop(key, None)
        return active

    def owner(self, did: str) -> str:
        return self.registry.functions.identityOwner(self.address_of(did)).call()

    @staticmethod
    def sign_challenge(identity_key, nonce: str):
        return Account.sign_message(encode_defunct(text=nonce), identity_key).signature.hex()

    def check_challenge(self, did: str, nonce: str, signature: str) -> bool:
        signer = Account.recover_message(encode_defunct(text=nonce), signature=bytes.fromhex(signature.removeprefix("0x")))
        return signer == self.owner(did)
