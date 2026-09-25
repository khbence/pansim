#!/usr/bin/env bash
# Stochastic panSim ensemble. One GPU run at a time; stdout is the daily table.
#
#   viz/run_ensemble.sh
#   viz/run_ensemble.sh -w 8 -n 6 -o viz/runs/weeks8
#
# Defaults match the 8-week set: 6 repeats, written to viz/runs/weeks8.
set -euo pipefail

WEEKS=8
COUNT=6
OUT=viz/runs/weeks8

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--weeks) WEEKS="$2"; shift 2 ;;
    -n|--count) COUNT="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"

if [[ ! -x ./build_gpu/panSim ]]; then
  echo "Missing ./build_gpu/panSim" >&2
  exit 1
fi

for n in $(seq 1 "$COUNT"); do
  i="$(printf '%02d' "$n")"
  dest="$OUT/run_${i}.stdout"
  echo "Run ${i}/${COUNT} -> ${dest} (-w ${WEEKS})"
  ./build_gpu/panSim -r --quarantinePolicy 0 -k 0.00041 \
    --progression inputConfigFiles/progressions_Jun17_tune/transition_config.json \
    -A inputConfigFiles/agentTypes_3.json \
    -a inputRealExample/agents1.json \
    -l inputRealExample/locations0.json \
    --infectiousnessMultiplier 0.98,1.81,2.11,2.58,4.32,6.8,6.8 \
    --diseaseProgressionScaling 0.94,1.03,0.813,0.72,0.57,0.463,0.45 \
    --closures inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json \
    -w "$WEEKS" \
    > "$dest"
done

echo "Wrote ${COUNT} runs to ${OUT}"
