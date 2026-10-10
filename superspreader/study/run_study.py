#!/usr/bin/env python3
"""Run the ACT-049 PanSim experiments on the first two Hungarian waves.

Day 0 is 2020-09-23. Week 37 ends just after 2021-06-02, the end of the
alpha wave in viz/fitted_parameters2.json. One GPU run at a time.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build_gpu" / "panSim"
RESULTS = Path(__file__).resolve().parent / "results"
PARAMS = json.loads((ROOT / "viz" / "fitted_parameters2.json").read_text())
CLOSURE_FITTED = ROOT / "inputConfigFiles" / "closureJun6_real_later3_delta_omicronBA2_earlier8.json"
CLOSURE_NONE = Path(__file__).resolve().parent / "closures_none.json"
CLOSURE_SEED = Path(__file__).resolve().parent / "closures_seed.json"

INF = ",".join(format(float(value), ".8g") for value in PARAMS["infectiousnessMultiplier"])
PROG = ",".join(format(float(value), ".8g") for value in PARAMS["diseaseProgressionScaling"])
BASE_K = float(PARAMS["k"])


def run_job(name, args):
    out = RESULTS / name
    stdout_path = out / "stdout.txt"
    trace_path = out / "trace.tsv"
    if stdout_path.exists() and trace_path.exists() and trace_path.stat().st_size > 0:
        print(f"skip {name}", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BINARY),
        "--progression",
        "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "-A",
        "inputConfigFiles/agentTypes_3.json",
        "-a",
        "inputRealExample/agents1.json",
        "-l",
        "inputRealExample/locations0.json",
        "--infectiousnessMultiplier",
        INF,
        "--diseaseProgressionScaling",
        PROG,
        "--infectionTrace",
        str(trace_path),
        *args,
    ]
    print("run " + name, flush=True)
    partial = out / "stdout.partial"
    with partial.open("w") as handle:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not trace_path.exists() or trace_path.stat().st_size == 0:
        print(f"FAILED {name} exit {completed.returncode}", flush=True)
        partial.rename(out / "stdout.failed")
        return completed.returncode or 1
    partial.rename(stdout_path)
    (out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    return 0


def exp1():
    failures = 0
    shapes = [("off", 0.0), ("1", 1.0), ("0.3", 0.3), ("0.1", 0.1)]
    for shed_name, shed in shapes:
        for shuffle in (0, 1):
            for venue in ("off", "covid"):
                for rep in range(1, 5):
                    name = f"exp1/shed{shed_name}_shuf{shuffle}_venue{venue}/rep{rep:02d}"
                    failures += run_job(
                        name,
                        [
                            "-r",
                            "--quarantinePolicy",
                            "0",
                            "--closures",
                            str(CLOSURE_NONE),
                            "-k",
                            format(BASE_K, ".8g"),
                            "-w",
                            "8",
                            "--seed",
                            str(rep),
                            "--sheddingShape",
                            format(shed, ".8g"),
                            "--shuffleSchedules",
                            str(shuffle),
                            "--venueProfile",
                            venue,
                            "--venueStrength",
                            "1",
                        ],
                    )
    for rep in range(1, 5):
        name = f"exp1/events_r0.02_f15/rep{rep:02d}"
        failures += run_job(
            name,
            [
                "-r",
                "--quarantinePolicy",
                "0",
                "--closures",
                str(CLOSURE_NONE),
                "-k",
                format(BASE_K, ".8g"),
                "-w",
                "8",
                "--seed",
                str(rep),
                "--sheddingShape",
                "0",
                "--shuffleSchedules",
                "0",
                "--venueProfile",
                "off",
                "--eventRate",
                "0.02",
                "--eventFactor",
                "15",
            ],
        )
    return failures


def exp1_waves():
    """Fitted closures and quarantine through the alpha wave, four mechanism corners."""
    failures = 0
    corners = [
        ("baseline", 0.0, 0, "off", 0.0, 1.0),
        ("biology", 0.1, 1, "off", 0.0, 1.0),
        ("setting", 0.0, 0, "covid", 0.0, 1.0),
        ("stack", 0.3, 0, "covid", 0.0, 1.0),
    ]
    for label, shed, shuffle, venue, event_rate, strength in corners:
        for rep in range(1, 3):
            name = f"waves/{label}/rep{rep:02d}"
            failures += run_job(
                name,
                [
                    "-r",
                    "--quarantinePolicy",
                    str(int(PARAMS["quarantinePolicy"])),
                    "--closures",
                    str(CLOSURE_FITTED),
                    "-k",
                    format(BASE_K, ".8g"),
                    "-w",
                    "37",
                    "--seed",
                    str(100 + rep),
                    "--sheddingShape",
                    format(shed, ".8g"),
                    "--shuffleSchedules",
                    str(shuffle),
                    "--venueProfile",
                    venue,
                    "--venueStrength",
                    format(strength, ".8g"),
                    "--eventRate",
                    format(event_rate, ".8g"),
                ],
            )
    return failures


def exp2(set_i, set_ii):
    """Earlier draft of the intervention grid. The reported runs are run_followup.py."""
    failures = 0
    interventions = {
        "none": [],
        "community_half": ["--communityScale", "0.5"],
        "contacts_75": [],  # k is passed inside the set, replaced below
        "isolate": ["--quarantinePolicy", "1"],
        "trace_household": ["--quarantinePolicy", "2"],
    }
    for label, spec in (("I", set_i), ("II", set_ii)):
        for intervention, extra in interventions.items():
            for rep in range(1, 4):
                args = [
                    "-r",
                    "--closures",
                    str(CLOSURE_NONE),
                    "-w",
                    "8",
                    "--seed",
                    str(200 + rep),
                    *spec,
                ]
                if intervention == "contacts_75":
                    # spec contains -k VALUE; scale that value.
                    scaled = []
                    skip = False
                    for item in args:
                        if skip:
                            scaled.append(format(float(item) * 0.75, ".8g"))
                            skip = False
                        else:
                            scaled.append(item)
                            if item == "-k":
                                skip = True
                    args = scaled
                # quarantine in spec is 0; intervention overrides by appearing later.
                # cxxopts uses the last value only if the parser is configured that way.
                # Pass quarantine once: strip a leading policy from spec and set it here.
                args = [item for item in args]
                if intervention in ("isolate", "trace_household"):
                    cleaned = []
                    skip = False
                    for item in args:
                        if skip:
                            skip = False
                            continue
                        if item == "--quarantinePolicy":
                            skip = True
                            continue
                        cleaned.append(item)
                    args = cleaned + extra
                elif intervention == "community_half":
                    args = args + extra
                elif "--quarantinePolicy" not in args:
                    args = ["--quarantinePolicy", "0", *args]
                name = f"exp2/{label}_{intervention}/rep{rep:02d}"
                failures += run_job(name, args)
    return failures


def exp4(k_by_arm):
    """Earlier draft of the chain grid. The reported runs are run_followup.py."""
    failures = 0
    for (profile, shed, shuffle, target), k_value in k_by_arm.items():
        for quarantine, qlabel in ((0, "none"), (1, "isolate")):
            if quarantine == 1 and not (profile == "caregiving" and target == "0.30" and shed == 0.0):
                continue
            reps = 16 if quarantine == 0 else 12
            for rep in range(1, reps + 1):
                name = f"exp4/{profile}_shed{shed}_shuf{shuffle}_R{target}_{qlabel}/rep{rep:02d}"
                failures += run_job(
                    name,
                    [
                        "--quarantinePolicy",
                        str(quarantine),
                        "--closures",
                        str(CLOSURE_SEED),
                        "-k",
                        format(k_value, ".8g"),
                        "-w",
                        "6",
                        "--seed",
                        str(300 + rep),
                        "--sheddingShape",
                        format(shed, ".8g"),
                        "--shuffleSchedules",
                        str(shuffle),
                        "--venueProfile",
                        profile if profile != "flat" else "off",
                        "--venueStrength",
                        "1",
                    ],
                )
    return failures


def sweep():
    """Stronger venue amplification and harsher explicit events. Two replicates."""
    failures = 0
    jobs = [
        ("sweep/venue_s2", ["--venueProfile", "covid", "--venueStrength", "2"]),
        ("sweep/venue_s4", ["--venueProfile", "covid", "--venueStrength", "4"]),
        ("sweep/venue_s8", ["--venueProfile", "covid", "--venueStrength", "8"]),
        ("sweep/event_r0.05_f40", ["--venueProfile", "off", "--eventRate", "0.05", "--eventFactor", "40"]),
        ("sweep/event_r0.1_f30", ["--venueProfile", "off", "--eventRate", "0.1", "--eventFactor", "30"]),
    ]
    for label, extra in jobs:
        for rep in range(1, 3):
            failures += run_job(
                f"{label}/rep{rep:02d}",
                [
                    "-r",
                    "--quarantinePolicy",
                    "0",
                    "--closures",
                    str(CLOSURE_NONE),
                    "-k",
                    format(BASE_K, ".8g"),
                    "-w",
                    "8",
                    "--seed",
                    str(rep),
                    "--sheddingShape",
                    "0",
                    "--shuffleSchedules",
                    "0",
                    *extra,
                ],
            )
    return failures


def exp4_pilot():
    failures = 0
    arms = [
        ("flat", 0.0, 1),
        ("caregiving", 0.0, 0),
        ("caregiving", 1.0, 0),
    ]
    for profile, shed, shuffle in arms:
        for rep in range(1, 3):
            name = f"exp4pilot/{profile}_shed{shed}_shuf{shuffle}/rep{rep:02d}"
            failures += run_job(
                name,
                [
                    "--quarantinePolicy",
                    "0",
                    "--closures",
                    str(CLOSURE_SEED),
                    "-k",
                    format(BASE_K, ".8g"),
                    "-w",
                    "4",
                    "--seed",
                    str(400 + rep),
                    "--sheddingShape",
                    format(shed, ".8g"),
                    "--shuffleSchedules",
                    str(shuffle),
                    "--venueProfile",
                    "off" if profile == "flat" else profile,
                    "--venueStrength",
                    "1",
                ],
            )
    return failures


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "exp1"
    if phase == "exp1":
        sys.exit(min(exp1(), 1))
    if phase == "waves":
        sys.exit(min(exp1_waves(), 1))
    if phase == "sweep":
        sys.exit(min(sweep(), 1))
    if phase == "exp4pilot":
        sys.exit(min(exp4_pilot(), 1))
    print("unknown phase", phase, file=sys.stderr)
    sys.exit(2)
