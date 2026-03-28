#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <panSim-binary> <reference-dir>" >&2
    exit 1
fi

BIN="$1"
REF_DIR="$2"
REPO_ROOT="$(cd "${REF_DIR}/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

STDOUT_REF="${REF_DIR}/determinism_cpu_seed1234_threads1.stdout"
JSON_REF="${REF_DIR}/determinism_cpu_seed1234_threads1.json"

compare_text_file() {
    local expected="$1"
    local actual="$2"
    local label="$3"

    if ! diff -u <(printf '%s\n' "$(cat "${expected}")") <(printf '%s\n' "$(cat "${actual}")") >&2; then
        echo "${label} mismatch against deterministic reference" >&2
        exit 1
    fi
}

(
    cd "${REPO_ROOT}"
    "${BIN}" \
    --seed 1234 \
    --threads 1 \
    -w 1 \
    -n 8 \
    -N 4 \
    --outAgentStat "${TMP_DIR}/actual.json" \
    >"${TMP_DIR}/actual.stdout" \
    2>"${TMP_DIR}/actual.stderr"
)

compare_text_file "${STDOUT_REF}" "${TMP_DIR}/actual.stdout" "stdout"
compare_text_file "${JSON_REF}" "${TMP_DIR}/actual.json" "agent stats JSON"

echo "deterministic reference matched"
