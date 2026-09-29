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
import sys, yaml
for k, v in yaml.safe_load(open(sys.argv[1])).items():
    print(f"{k}\t{v}")
PY
    out_dir="$results_root/$name"
    if [ -f "$out_dir/all_runs.tsv" ] && [ -z "${FORCE:-}" ]; then
        echo "=== scenario: $name - already has results at $out_dir, skipping (FORCE=1 to re-run)"
        continue
    fi
    echo "=== scenario: $name ($flags)"
    start=$(date +%s)
    EXTRA_ARGS="$flags" OUT_DIR="$out_dir" "$root/scripts/run_20_sims.sh"
    echo "=== scenario: $name took $(( $(date +%s) - start ))s"
done
