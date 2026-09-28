#!/bin/bash
# Run PanSim N_RUNS times with the default parameters (defaultargs.json)
# and collect the results.
#
# Usage (from anywhere, on the HPDC machine, after building - see SETUP_HPC.md):
#   ./scripts/run_20_sims.sh
#   N_RUNS=20 WEEKS=24 OUT_DIR=results_$(date +%F) ./scripts/run_20_sims.sh
#   EXTRA_ARGS="--quarantinePolicy 3" OUT_DIR=results/q3 ./scripts/run_20_sims.sh
#
# Output:
#   $OUT_DIR/run_XX.tsv     stdout of run XX (tab-separated time series)
#   $OUT_DIR/run_XX.log     stderr of run XX
#   $OUT_DIR/all_runs.tsv   all runs concatenated, with a leading "run" column
#   $OUT_DIR/config.txt     exact command line + start date + git commit
#   $OUT_DIR/meta.txt       wall-clock start/end time of the experiment

set -euo pipefail

cd "$(dirname "$0")/.."          # pansim/ - all paths below are relative to it

N_RUNS=${N_RUNS:-20}
WEEKS=${WEEKS:-12}               # default from defaultargs.json
BIN=${BIN:-./build/panSim}
OUT_DIR=${OUT_DIR:-results_$(date +%Y-%m-%d_%H%M%S)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}

[ -x "$BIN" ] || { echo "Binary $BIN not found - build first (SETUP_HPC.md)"; exit 1; }
[ -f inputRealExample/agents.json ] || { echo "inputRealExample/agents.json missing - see SETUP_HPC.md step 3"; exit 1; }

# Everything except the run length comes from defaultargs.json
# (start date 267 = Sep 24, start weekday 2 = Wednesday).
ARGS=(-w "$WEEKS")
# Optional extra panSim flags, e.g. EXTRA_ARGS="--quarantinePolicy 3 --closures x.json"
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -ra EXTRA <<< "$EXTRA_ARGS"
    ARGS+=("${EXTRA[@]}")
fi

mkdir -p "$OUT_DIR"
{
    echo "date:   $(date -Is)"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "cmd:    $BIN ${ARGS[*]}"
} > "$OUT_DIR/config.txt"

echo "experiment start: $(date -Is)" | tee "$OUT_DIR/meta.txt"

for i in $(seq 1 "$N_RUNS"); do
    id=$(printf '%02d' "$i")
    echo "[$(date +%T)] run $id/$N_RUNS"
    "$BIN" "${ARGS[@]}" > "$OUT_DIR/run_$id.tsv" 2> "$OUT_DIR/run_$id.log"
done

# Combine: header once (prefixed with "run"), then every row prefixed with its run id
first="$OUT_DIR/run_01.tsv"
{ printf 'run\t'; head -n 1 "$first"; } > "$OUT_DIR/all_runs.tsv"
for i in $(seq 1 "$N_RUNS"); do
    id=$(printf '%02d' "$i")
    tail -n +2 "$OUT_DIR/run_$id.tsv" | awk -v r="$i" 'BEGIN{OFS="\t"} {print r, $0}' >> "$OUT_DIR/all_runs.tsv"
done

echo "experiment end:   $(date -Is)" | tee -a "$OUT_DIR/meta.txt"
echo "Results in $OUT_DIR/all_runs.tsv"
