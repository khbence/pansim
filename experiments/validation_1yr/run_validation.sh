#!/bin/bash
# 20 PanSim runs of 52 weeks (1 year), using the validated best-fit parameters
# (PARAM_SET=validated, see scripts/run_20_sims.sh and ../../viz/README.md).
# Output: results/sim/{run_XX.tsv, all_runs.tsv, config.txt, meta.txt}
# Run on the HPDC machine after building panSim (see ../../SETUP_HPC.md).
#
# Needs inputRealExample/agents1.json and locations0.json. Simulation day 0 is
# 2020-09-23 (see README.md; panSim's --startDate 267/--startDay 2 default).
#
# To reproduce the advisor's exact validated window (first two waves only) instead:
#   WEEKS=40 ./run_validation.sh
# To run the compiled defaults instead of the validated fit:
#   PARAM_SET=defaults ./run_validation.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

N_RUNS=20 WEEKS="${WEEKS:-52}" OUT_DIR="$here/results/sim" "$here/../../scripts/run_20_sims.sh"

echo "Done. Next: python aggregate_compartments.py && python plot_key_metrics.py   (in $here)"
