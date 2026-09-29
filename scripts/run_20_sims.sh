#!/bin/bash
# Run PanSim N_RUNS times and collect the results.
#
# Usage (from anywhere, on the HPDC machine, after building - see SETUP_HPC.md):
#   ./scripts/run_20_sims.sh
#   N_RUNS=20 WEEKS=52 OUT_DIR=results_$(date +%F) ./scripts/run_20_sims.sh
#   PARAM_SET=defaults ./scripts/run_20_sims.sh          # compiled defaults only, no validated fit
#   EXTRA_ARGS="--quarantinePolicy 0" ./scripts/run_20_sims.sh   # appended on top of PARAM_SET
#   SEED_BASE=1000 ./scripts/run_20_sims.sh              # reproducible per-run seeds (see note below)
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
WEEKS=${WEEKS:-52}
BIN=${BIN:-./build/panSim}
OUT_DIR=${OUT_DIR:-results_$(date +%Y-%m-%d_%H%M%S)}
PARAM_SET=${PARAM_SET:-validated}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}

[ -x "$BIN" ] || { echo "Binary $BIN not found - build first (SETUP_HPC.md)"; exit 1; }

ARGS=(-w "$WEEKS")

case "$PARAM_SET" in
    validated)
        # Best-known fit against real Hungarian data (waves 1-3), from viz/README.md /
        # viz/run_ensemble.sh and the advisor's Sep 2026 message. Day 0 is 2020-09-23
        # (--startDate 267, the compiled default). Requires inputRealExample/agents1.json
        # and locations0.json (SETUP_HPC.md step 3, or ask the advisor for the "users" tarball).
        ARGS+=(
            -r
            --quarantinePolicy 3
            -k 0.00041
            --progression inputConfigFiles/progressions_Jun17_tune/transition_config.json
            -A inputConfigFiles/agentTypes_3.json
            -a inputRealExample/agents1.json
            -l inputRealExample/locations0.json
            --infectiousnessMultiplier 0.98,1.85,2.05,2.58,4.32,6.8,6.8
            --diseaseProgressionScaling 0.94,1.03,1.40,0.72,0.57,0.463,0.45
            --closures inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json
        )
        [ -f inputRealExample/agents1.json ] || { echo "inputRealExample/agents1.json missing (PARAM_SET=validated needs it, see SETUP_HPC.md / ask advisor)"; exit 1; }
        ;;
    defaults)
        # Compiled defaults only (include/simulation.h etc.), plus -w.
        [ -f inputRealExample/agents.json ] || { echo "inputRealExample/agents.json missing - see SETUP_HPC.md step 3"; exit 1; }
        ;;
    *)
        echo "Unknown PARAM_SET '$PARAM_SET' (use 'validated' or 'defaults')"; exit 1 ;;
esac

# Optional extra panSim flags, applied after PARAM_SET so they can override it,
# e.g. EXTRA_ARGS="--quarantinePolicy 0 --closures x.json"
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -ra EXTRA <<< "$EXTRA_ARGS"
    ARGS+=("${EXTRA[@]}")
fi

# By default no --seed is passed: each run is left stochastic, which is what gives an
# ensemble its spread (viz/README.md: "omitting it keeps the stochastic spread the bands
# are for"). Set SEED_BASE to instead run reproducibly - run i uses seed SEED_BASE+i - but
# note deterministic behaviour is only guaranteed single-threaded (ARCHITECTURE.md), so this
# forces --threads 1 and will be much slower than the default multi-threaded stochastic runs.
USE_SEED=${SEED_BASE:+1}

mkdir -p "$OUT_DIR"
{
    echo "date:      $(date -Is)"
    echo "commit:    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "PARAM_SET: $PARAM_SET"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    if [ -n "${USE_SEED:-}" ]; then
        echo "seeds:     $((SEED_BASE+1))..$((SEED_BASE+N_RUNS)) (--threads 1, deterministic per run)"
    else
        echo "seeds:     none (stochastic, default)"
    fi
    echo "cmd:       $BIN ${ARGS[*]}"
} > "$OUT_DIR/config.txt"

echo "experiment start: $(date -Is)" | tee "$OUT_DIR/meta.txt"

for i in $(seq 1 "$N_RUNS"); do
    id=$(printf '%02d' "$i")
    echo "[$(date +%T)] run $id/$N_RUNS"
    RUN_ARGS=("${ARGS[@]}")
    if [ -n "${USE_SEED:-}" ]; then
        RUN_ARGS+=(--seed "$((SEED_BASE+i))" --threads 1)
    fi
    "$BIN" "${RUN_ARGS[@]}" > "$OUT_DIR/run_$id.tsv" 2> "$OUT_DIR/run_$id.log"
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
