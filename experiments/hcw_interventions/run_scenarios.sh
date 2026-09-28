#!/bin/bash
# Run run_20_sims.sh once per scenario in scenarios.yaml -> results/<scenario>/
# Usage: ./experiments/hcw_interventions/run_scenarios.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
python3 - "$here/scenarios.yaml" <<'PY' | while IFS=$'\t' read -r name flags; do
import sys, yaml
for k, v in yaml.safe_load(open(sys.argv[1])).items():
    print(f"{k}\t{v}")
PY
    echo "=== scenario: $name ($flags)"
    EXTRA_ARGS="$flags" OUT_DIR="$here/results/$name" "$root/scripts/run_20_sims.sh"
done
