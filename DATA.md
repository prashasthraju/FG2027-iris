# Data and weights needed for the full pipeline (Tier 2)

The paper's numbers can be re-checked without any of this (Tier 0 and Tier 1 in `README.md`). To regenerate the sessions yourself you need the datasets below, obtained from
their official pages and used under their licences. Keep them outside version control (`data/` and `external/` are git-ignored).

| Dataset | Official source | Expected location | Used by |
|---|---|---|---|
| RAVDESS video speech, actors 01-24 | https://zenodo.org/records/1188976 | `dbs/ravdess_video/` (made by `tools/ravdess_prep.py`) | sources and driving clips of the 21 RAVDESS people |
| CASIA-Iris-Distance (CASIA-IrisV4) | http://biometrics.idealtest.org/ | `dbs/CASIA-Iris-Distance/` | 46 near-infrared photo owners; their own captures D05-D09 / D00-D04 stand in for real irises |
| UTIRIS V.1 (RGB images) | https://utiris.wordpress.com/ | `dbs/UTIRIS_V.1/RGB Images/<subject>/IMG_<subject>_<L|R>_<capture>.JPG` | the 79-entry proxy pool (`cancelable/pool.py`) |
| AVAPrintDB and TrustFA-FG2026 score files | AVAPrintDB (Zenodo, record 20548979) and the TrustFA-FG2026 repository | `external/AVAPrintDB`, `external/TrustFA-FG2026` | behavioural baselines (`tools/baseline_behavioural.py`, `tools/behav_*.py`) |

Weights and code, from their official pages (see `THIRD_PARTY.md` for terms), each in its own folder under `third_party/` (git-ignored): LivePortrait (HuggingFace `KwaiVGI/LivePortrait`) in `third_party/LivePortrait` with its virtual environment at `third_party/LivePortrait/.venv_lp`; ArcIris from the CVRL release (`khan2026irex` in the paper) in `third_party/arciris` (the code expects `third_party/arciris/models`); LIA-X and EDTalk in `third_party/LIA-X` and `third_party/EDTalk`; PDQ in the environment `third_party/.venv_pdq`. PersonaLive and VideoSeal live outside the repository (`PERSONALIVE_DIR`, default `external/PersonaLive`; the watermark baseline reads `external/Watemark/baseline`).
Paths are read from environment variables where they are machine specific: `PERSONALIVE_DIR`, `PERSONALIVE_PYTHON`, `EMBED_WORKER_PYTHON`, `RAVDESS_METADATA`.

Hardware used for the paper: one shared 8 GB GPU (RTX 4060); the main run took days of wall time, the paired PersonaLive sessions need the GPU exclusively.
