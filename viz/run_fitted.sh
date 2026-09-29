#!/usr/bin/env bash
# Run the parameters in a fitted_parameters.json file as a stochastic ensemble.
# One GPU run at a time. Each stdout is run_01.stdout, run_02.stdout, ...
#
#   viz/run_fitted.sh viz/fitted_parameters.json -n 20 -o viz/runs/refit
#   viz/run_fitted.sh viz/fitted_parameters.json -n 8 -o viz/runs/check -w 68
#
# Weeks default to the latest wave end date in the JSON. Day 0 is 2020-09-23.
# Seeds in the JSON are applied to the standard closure file. The output
# directory gets reproducibility.json (configuration, the panSim argument list,
# and git status) and the closures.json that the runs actually read.
# viz/run_ensemble.sh is not used and is not modified.
set -euo pipefail

ORIGINAL_ARGS=("$@")
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

write_manifest() {
  local state="$1"
  local failed_run="${2:-}"
  PARAMS="$PARAMS" OUT="$OUT" COUNT="$COUNT" WEEKS="$WEEKS" \
    QUARANTINE="$QUARANTINE" K="$K" INF="$INF" PROG="$PROG" \
    CLOSURE_SRC="$CLOSURE_SRC" CLOSURE="$closure" ROOT="$ROOT" \
    STATE="$state" FAILED_RUN="$failed_run" DRY="$DRY" \
    python3 - "${ORIGINAL_ARGS[@]}" << 'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["ROOT"])
out = Path(os.environ["OUT"])
closure_path = out / "closures.json"
manifest_path = out / "reproducibility.json"
now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def git(*args):
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()

def num_list(text):
    return [float(part) for part in text.split(",") if part]

arguments = [
    "./build_gpu/panSim",
    "-r",
    "--quarantinePolicy",
    os.environ["QUARANTINE"],
    "-k",
    os.environ["K"],
    "--progression",
    "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
    "-A",
    "inputConfigFiles/agentTypes_3.json",
    "-a",
    "inputRealExample/agents1.json",
    "-l",
    "inputRealExample/locations0.json",
    "--infectiousnessMultiplier",
    os.environ["INF"],
    "--diseaseProgressionScaling",
    os.environ["PROG"],
    "--closures",
    str(closure_path),
    "-w",
    os.environ["WEEKS"],
]
binary = root / "build_gpu" / "panSim"
created = now
if manifest_path.exists():
    previous = json.loads(manifest_path.read_text())
    created = previous.get("created", now)
    code = previous.get("code", {})
    binary_hash = previous.get("panSim_sha256")
else:
    status = git("status", "--porcelain")
    code = {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None
closure_path.write_text(Path(os.environ["CLOSURE"]).read_text())
runs = sorted(path.name for path in out.glob("run_*.stdout"))
payload = {
    "created": created,
    "updated": now,
    "state": os.environ["STATE"],
    "dry_run": os.environ["DRY"] == "1",
    "invocation": ["viz/run_fitted.sh", *sys.argv[1:]],
    "parameters_file": os.environ["PARAMS"],
    "output_directory": str(out),
    "count": int(os.environ["COUNT"]),
    "weeks": int(os.environ["WEEKS"]),
    "failed_run": os.environ["FAILED_RUN"] or None,
    "configuration": {
        "parameters": json.loads(Path(os.environ["PARAMS"]).read_text()),
        "k": float(os.environ["K"]),
        "quarantinePolicy": int(os.environ["QUARANTINE"]),
        "infectiousnessMultiplier": num_list(os.environ["INF"]),
        "diseaseProgressionScaling": num_list(os.environ["PROG"]),
        "closure_source": os.environ["CLOSURE_SRC"],
        "closure_file": str(closure_path),
        "progression": "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "agentTypes": "inputConfigFiles/agentTypes_3.json",
        "agents": "inputRealExample/agents1.json",
        "locations": "inputRealExample/locations0.json",
        "startDate": "2020-09-23",
    },
    "arguments": arguments,
    "code": code,
    "panSim_sha256": binary_hash,
    "runs": runs,
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote {manifest_path}", flush=True)
PY
}

echo "Parameters ${PARAMS}"
echo "  -w ${WEEKS} -n ${COUNT} -o ${OUT}"
echo "  -k ${K} --quarantinePolicy ${QUARANTINE}"
echo "  --infectiousnessMultiplier ${INF}"
echo "  --diseaseProgressionScaling ${PROG}"

write_manifest started

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
    --closures "$OUT/closures.json" \
    -w "$WEEKS" \
    > "$partial"; then
    rm -f "$partial"
    write_manifest failed "$i"
    exit 1
  fi
  mv "$partial" "$dest"
done

write_manifest finished
echo "Wrote ${COUNT} runs to ${OUT}"
