#!/usr/bin/env bash
# Run the parameters in a fitted_parameters.json file as a stochastic ensemble.
# One GPU run at a time. Each stdout is run_01.stdout, run_02.stdout, ...
#
#   viz/run_fitted.sh viz/fitted_parameters.json -n 20 -o viz/runs/refit
#   viz/run_fitted.sh viz/fitted_parameters.json -n 8 -o viz/runs/check -w 68
#
# Weeks default to the latest wave end date in the JSON. Day 0 is 2020-09-23.
# Seeds in the JSON are applied to the standard closure file. viz/run_ensemble.sh
# is not used and is not modified.
set -euo pipefail

PARAMS=""
COUNT=""
OUT=""
WEEKS=""
DRY=0

usage() {
  echo "Usage: viz/run_fitted.sh fitted_parameters.json -n COUNT -o DIRECTORY [-w WEEKS]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--count) COUNT="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -w|--weeks) WEEKS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ -z "$PARAMS" ]]; then
        PARAMS="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$PARAMS" || -z "$COUNT" || -z "$OUT" ]]; then
  usage
  exit 2
fi
if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "COUNT must be a positive integer" >&2
  exit 2
fi
if [[ -n "$WEEKS" && ! "$WEEKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WEEKS must be a positive integer" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$PARAMS" != /* ]]; then
  PARAMS="$(pwd)/$PARAMS"
fi
if [[ "$OUT" != /* ]]; then
  OUT="$(pwd)/$OUT"
fi
if [[ ! -f "$PARAMS" ]]; then
  echo "Missing parameters file: $PARAMS" >&2
  exit 1
fi

cd "$ROOT"
if [[ ! -x ./build_gpu/panSim ]]; then
  echo "Missing ./build_gpu/panSim" >&2
  exit 1
fi

CLOSURE_SRC="inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
closure="$workdir/closures.json"
cfg="$workdir/run.env"

python3 - "$PARAMS" "$WEEKS" "$CLOSURE_SRC" "$closure" "$cfg" << 'PY'
import json
import shlex
import sys
from datetime import date
from pathlib import Path

params_path, weeks_arg, closure_src, closure_path, cfg_path = sys.argv[1:]
params = json.loads(Path(params_path).read_text())
start = date(2020, 9, 23)

def weeks_for(end: str) -> int:
    end_day = (date.fromisoformat(end) - start).days
    return (end_day + 1 + 6) // 7

if weeks_arg:
    weeks = int(weeks_arg)
else:
    ends = [wave["end"] for wave in params.get("waves", []) if wave.get("end")]
    if not ends:
        raise SystemExit("JSON has no wave end date; pass -w WEEKS")
    weeks = weeks_for(max(ends))

def num(value) -> str:
    return format(float(value), ".8g")

seeds = {int(item["variant"]): item for item in params["seeds"]}
rules = json.loads(Path(closure_src).read_text())
seen = set()
for rule in rules["rules"]:
    if rule.get("name") != "ExposeToMutation":
        continue
    variant = int(rule.get("threshold2", -1))
    if variant not in seeds or variant in seen or int(rule["threshold"]) > 1000:
        continue
    rule["threshold"] = int(seeds[variant]["day"])
    rule["parameter"] = float(seeds[variant]["fraction"])
    seen.add(variant)
Path(closure_path).write_text(json.dumps(rules, indent=2) + "\n")

lines = [
    f"WEEKS={weeks}",
    "QUARANTINE=" + shlex.quote(str(int(params.get("quarantinePolicy", 3)))),
    "K=" + shlex.quote(num(params["k"])),
    "INF=" + shlex.quote(",".join(num(x) for x in params["infectiousnessMultiplier"])),
    "PROG=" + shlex.quote(",".join(num(x) for x in params["diseaseProgressionScaling"])),
]
Path(cfg_path).write_text("\n".join(lines) + "\n")
PY

# shellcheck disable=SC1090
source "$cfg"
mkdir -p "$OUT"

echo "Parameters ${PARAMS}"
echo "  -w ${WEEKS} -n ${COUNT} -o ${OUT}"
echo "  -k ${K} --quarantinePolicy ${QUARANTINE}"
echo "  --infectiousnessMultiplier ${INF}"
echo "  --diseaseProgressionScaling ${PROG}"

if [[ "$DRY" -eq 1 ]]; then
  echo "Dry run; no simulations started."
  exit 0
fi

for n in $(seq 1 "$COUNT"); do
  i="$(printf '%02d' "$n")"
  dest="$OUT/run_${i}.stdout"
  partial="${dest}.partial"
  echo "Run ${i}/${COUNT} -> ${dest} (-w ${WEEKS})"
  if ! ./build_gpu/panSim -r --quarantinePolicy "$QUARANTINE" -k "$K" \
    --progression inputConfigFiles/progressions_Jun17_tune/transition_config.json \
    -A inputConfigFiles/agentTypes_3.json \
    -a inputRealExample/agents1.json \
    -l inputRealExample/locations0.json \
    --infectiousnessMultiplier "$INF" \
    --diseaseProgressionScaling "$PROG" \
    --closures "$closure" \
    -w "$WEEKS" \
    > "$partial"; then
    rm -f "$partial"
    exit 1
  fi
  mv "$partial" "$dest"
done

echo "Wrote ${COUNT} runs to ${OUT}"
