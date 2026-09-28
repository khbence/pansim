#!/bin/bash
# 20 PanSim runs of 52 weeks (1 year), default parameters.
# Output: results/sim/{run_XX.tsv, all_runs.tsv, config.txt, meta.txt}
# Run on the HPDC machine after building panSim (see ../../SETUP_HPC.md).
#
# Simulation day 0 is treated as 2020-09-23 (see config.yaml). panSim's default
# --startDate 267 / --startDay 2 is left untouched (267 = 24 Sep with Jan 1 = 0 in 2020,
# so there may be a one-day offset; see README.md).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

N_RUNS=20 WEEKS=52 OUT_DIR="$here/results/sim" "$here/../../scripts/run_20_sims.sh"

echo "Done. Next: python validate.py   (in $here)"
