#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build_cpu}"
BIN="${BUILD_DIR}/panSim"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ ! -x "${BIN}" ]]; then
    echo "Missing executable: ${BIN}" >&2
    exit 1
fi

COMMON_ARGS=(
    --seed 1234
    --threads 1
    -w 1
    -n 256
    -N 64
    --outAgentStat
)

echo "Running deterministic CPU smoke test with ${BIN}"
"${BIN}" "${COMMON_ARGS[@]}" "${TMP_DIR}/stats_run_1.json" >"${TMP_DIR}/run1.stdout" 2>"${TMP_DIR}/run1.stderr"
"${BIN}" "${COMMON_ARGS[@]}" "${TMP_DIR}/stats_run_2.json" >"${TMP_DIR}/run2.stdout" 2>"${TMP_DIR}/run2.stderr"

cmp -s "${TMP_DIR}/stats_run_1.json" "${TMP_DIR}/stats_run_2.json"

echo "Deterministic output verified"
