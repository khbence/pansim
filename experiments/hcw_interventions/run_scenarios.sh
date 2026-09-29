#!/bin/bash
# Run run_20_sims.sh once per scenario in scenarios.yaml -> $RESULTS_ROOT/<scenario>/
#
# Usage: ./experiments/hcw_interventions/run_scenarios.sh
#   RESULTS_ROOT=/scratch/results N_RUNS=2 WEEKS=2 ./run_scenarios.sh   # cheap smoke test first
#   SCENARIOS_FILE=some_other_scenarios.yaml ./run_scenarios.sh        # use a different scenario file
#   FORCE=1 ./run_scenarios.sh                           # re-run scenarios that already have results
#
# A scenario whose $RESULTS_ROOT/<name>/all_runs.tsv already exists is skipped (unless FORCE=1),
# so this script is safe to re-run after an interruption, and "baseline" - provisioned once from
# experiments/validation_1yr - is never re-simulated.
#
# Each scenario is 20 runs x 52 weeks by default (~1h observed previously); scenarios run
# sequentially, not in parallel, since the target HPC allocation is a fixed core count
# (SETUP_HPC.md) that concurrent panSim processes would oversubscribe.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
results_root="${RESULTS_ROOT:-$here/results}"
scenarios_file="${SCENARIOS_FILE:-$here/scenarios.yaml}"

python3 - "$scenarios_file" <<'PY' | while IFS=$'\t' read -r name flags; do
# Pure-stdlib parser for scenarios.yaml's flat `name: "flags"` mapping (no pyyaml dependency -
# HPC nodes often have no pip/network access to install it). Every active line is
# `name: "flags"` or `name: ""`; blank lines and lines starting with # are skipped.
import sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#") or ":" not in line:
        continue
    name, _, value = line.partition(":")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    print(f"{name.strip()}\t{value}")
PY
    out_dir="$results_root/$name"
    if [ -f "$out_dir/all_runs.tsv" ] && [ -z "${FORCE:-}" ]; then
        echo "=== scenario: $name - already has results at $out_dir, skipping (FORCE=1 to re-run)"
        continue
    fi
    echo "=== scenario: $name ($flags)"
    start=$(date +%s)
    EXTRA_ARGS="$flags" OUT_DIR="$out_dir" bash "$root/scripts/run_20_sims.sh"
    echo "=== scenario: $name took $(( $(date +%s) - start ))s"
done
