# From the paper to the code

| Paper | Code |
|---|---|
| Rubber-sheet map and strip (Eq. 1) | `iris_pipeline/normalize.py` (`unwrap`, `wrap`; ellipse boundaries) |
| Proxy template: ArcIris embedding and log-Gabor code (Eq. 2) | `iris_pipeline/judge.py` (ArcIris wrapper), `iris_pipeline/encode.py` (log-Gabor), `cancelable/pool.py` |
| Subject id and keyed-hash assignment with probing (Eqs. 3-4) | `tools/proxy_ledger.py` (`hmac_index`), `tools/proxy_attr.py` (enrolment, `build_ledger`) |
| Salted commitment, DID version (Eq. 5) | `tools/proxy_ledger.py` (`commit`, `keccak_hex`, `Ledger.register_did_version`), `tools/proxy_commit_migrate.py` |
| Visibility gate (Eq. 6) | `iris_pipeline/quality.py` (`assess`) |
| Inverse map, blur and noise match, matte, Laplacian blend (Eqs. 7-9) | `iris_pipeline/composite.py` (`target_matte`, `match_blur`, `pupil_noise`, `laplacian_blend`), `iris_pipeline/embed.py` (`embed_into_eye`) |
| Per-frame hook for any generator | `cancelable/inline.py` (`FrameEmbedder`, `WorkerEmbedder`); adapters `tools/reenact_video.py`, `tools/gen_liax.py`, `tools/gen_edtalk.py`, `tools/personalive_lowmem.py` |
| Session record and signatures (Eq. 10) | `tools/proxy_attr.py` (`make_record`), `tools/proxy_ledger.py` (`sign`, `Ledger.record_session`) |
| Verification step 1 (video matches the record; PDQ tolerance, Eq. 11) | `tools/proxy_verify.py` (`pdq_match`), `tools/pdq_worker.py` |
| Verification step 2 (record checks, revocation by block height) | `tools/proxy_verify.py` (`check_record`) |
| Steps 3-4 (key, live iris, re-derivation, opening the commitment; Eq. 12) | `tools/proxy_issuer_eval.py` (`Issuer`, `Issuer.open_commitment`, `verify_session`), `tools/idmap_eval.py` (live-iris distance and threshold) |
| Step 5 (mean angular distance, T1; Eqs. 13-14) | `tools/proxy_verify.py` (`Gallery.vectors`, `cmd_calibrate`) |
| One-time proxy per session (Eq. 15) | design only, not implemented |
