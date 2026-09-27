#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <panSim-binary>" >&2
    exit 1
fi

BIN="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ ! -x "${BIN}" ]]; then
    echo "Missing executable: ${BIN}" >&2
    exit 1
fi

compare_file() {
    local lhs="$1"
    local rhs="$2"
    local label="$3"

    if ! cmp -s "${lhs}" "${rhs}"; then
        echo "${label} differed between seeded runs" >&2
        diff -u <(sed -e '$a\' "${lhs}") <(sed -e '$a\' "${rhs}") >&2 || true
        exit 1
    fi
}

run_once() {
    local tag="$1"

    (
        cd "${REPO_ROOT}"
        "${BIN}" \
        --seed 1234 \
        --threads 1 \
        -w 1 \
        -n 256 \
        -N 64 \
        --outAgentStat "${TMP_DIR}/${tag}.json" \
        >"${TMP_DIR}/${tag}.stdout" \
        2>"${TMP_DIR}/${tag}.stderr"
    )
}

run_once run1
run_once run2

compare_file "${TMP_DIR}/run1.json" "${TMP_DIR}/run2.json" "agent stats JSON"
compare_file "${TMP_DIR}/run1.stdout" "${TMP_DIR}/run2.stdout" "stdout"

echo "deterministic smoke output verified"
