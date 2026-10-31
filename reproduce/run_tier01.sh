#!/usr/bin/env bash
# Tier 0 + Tier 1: protocol tests, independent recomputation of the headline numbers, regeneration of the paper's macros and tables. About 10 seconds, CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/test_protocol.py
python reproduce/recompute_metrics.py
bash reproduce/regenerate_paper_numbers.sh
