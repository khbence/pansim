#!/usr/bin/env python3
"""Shared helpers for the PS1 / PS2 experiments (ACT-049, instructions 16)."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))

from run_study import CLOSURE_NONE, INF, PROG, ROOT, BASE_K  # noqa: E402

BINARY = Path(os.environ.get("PANSIM_BIN", ROOT / "build_gpu" / "panSim"))
RESULTS = STUDY / "results" / "ps"
CAL_LOG = RESULTS / "calibration_log.json"
STEPS_PER_DAY = 144


def fmt(value):
    return format(value, ".6g")


def log_event(entry):
    CAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CAL_LOG.read_text()) if CAL_LOG.exists() else []
    data.append(entry)
    CAL_LOG.write_text(json.dumps(data, indent=1) + "\n")


def run_job(name, args, binary=None, extra_outputs=(), results=None):
    """Run one PanSim job. Outputs: stdout.txt, trace.tsv, plus any extra_outputs (name -> flag)."""
    out = (results or RESULTS) / name
    stdout_path = out / "stdout.txt"
    trace_path = out / "trace.tsv"
    if stdout_path.exists() and trace_path.exists() and trace_path.stat().st_size > 0:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    extra = []
    for filename, flag in extra_outputs:
        extra += [flag, str(out / filename)]
    cmd = [
        str(binary or BINARY),
        "--progression", "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "-A", "inputConfigFiles/agentTypes_3.json",
        "-a", "inputRealExample/agents1.json",
        "-l", "inputRealExample/locations0.json",
        "--infectiousnessMultiplier", INF,
        "--diseaseProgressionScaling", PROG,
        "--infectionTrace", str(trace_path),
        *extra,
        *args,
    ]
    started = time.time()
    partial = out / "stdout.partial"
    with partial.open("w") as handle:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    if completed.returncode != 0 or not trace_path.exists() or trace_path.stat().st_size == 0:
        print(f"FAILED {name} exit {completed.returncode}", flush=True)
        partial.rename(out / "stdout.failed")
        return completed.returncode or 1
    partial.rename(stdout_path)
    (out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (out / "elapsed_seconds.txt").write_text(f"{elapsed:.1f}\n")
    print(f"ran {name} in {elapsed:.0f}s", flush=True)
    return 0


def base_args(k, seed, weeks, quarantine=0, closures=None):
    return [
        "-r",
        "--quarantinePolicy", str(quarantine),
        "--closures", str(closures or CLOSURE_NONE),
        "-k", fmt(k),
        "-w", str(weeks),
        "--seed", str(seed),
        "--densityExponent", "1",
        "--venueProfile", "off",
    ]
