#!/usr/bin/env python3
"""Experiment 2 (matched k) and experiment 4 (subcritical chains)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_study import BASE_K, CLOSURE_NONE, CLOSURE_SEED, run_job

# Early-window R was 1.307 for shedding shape 1 and 0.998 for venue strength 4.
# Scale the biological arm so the two mechanisms start at the same mean offspring.
BIO_K = BASE_K * (0.998 / 1.307)
SETTING_K = BASE_K


def exp2():
    failures = 0
    sets = {
        "I": {
            "k": BIO_K,
            "extra": [
                "--sheddingShape",
                "1",
                "--shuffleSchedules",
                "1",
                "--venueProfile",
                "off",
            ],
        },
        "II": {
            "k": SETTING_K,
            "extra": [
                "--sheddingShape",
                "0",
                "--shuffleSchedules",
                "0",
                "--venueProfile",
                "covid",
                "--venueStrength",
                "4",
            ],
        },
    }
    interventions = ("none", "community_half", "contacts_75", "isolate", "trace_household")
    for label, spec in sets.items():
        for intervention in interventions:
            k_value = spec["k"] * (0.75 if intervention == "contacts_75" else 1.0)
            if intervention == "isolate":
                quarantine = "1"
            elif intervention == "trace_household":
                quarantine = "2"
            else:
                quarantine = "0"
            extra = list(spec["extra"])
            if intervention == "community_half":
                extra += ["--communityScale", "0.5"]
            for rep in range(1, 4):
                name = f"exp2/{label}_{intervention}/rep{rep:02d}"
                failures += run_job(
                    name,
                    [
                        "-r",
                        "--quarantinePolicy",
                        quarantine,
                        "--closures",
                        str(CLOSURE_NONE),
                        "-k",
                        format(k_value, ".8g"),
                        "-w",
                        "8",
                        "--seed",
                        str(500 + rep),
                        *extra,
                    ],
                )
    return failures


def exp4():
    # Generation-1 R at the fitted infection coefficient, from the 4-week pilot.
    pilot_r = {
        ("flat", 0.0, 1): 1.25,
        ("caregiving", 0.0, 0): 1.29,
        ("caregiving", 1.0, 0): 1.79,
    }
    targets = {"0.30": 0.30, "0.20": 0.20}
    failures = 0
    for (profile, shed, shuffle), r_pilot in pilot_r.items():
        for target_name, target_r in targets.items():
            k_value = BASE_K * (target_r / r_pilot)
            for rep in range(1, 17):
                name = (
                    f"exp4/{profile}_shed{shed:g}_shuf{shuffle}_R{target_name}_none/rep{rep:02d}"
                )
                failures += run_job(
                    name,
                    [
                        "--quarantinePolicy",
                        "0",
                        "--closures",
                        str(CLOSURE_SEED),
                        "-k",
                        format(k_value, ".8g"),
                        "-w",
                        "6",
                        "--seed",
                        str(600 + rep),
                        "--sheddingShape",
                        format(shed, ".8g"),
                        "--shuffleSchedules",
                        str(shuffle),
                        "--venueProfile",
                        "off" if profile == "flat" else "caregiving",
                        "--venueStrength",
                        "1",
                    ],
                )
    # Early isolation of detected index cases, Nipah-like caregiving arm only.
    k_iso = BASE_K * (0.30 / pilot_r[("caregiving", 0.0, 0)])
    for rep in range(1, 13):
        name = f"exp4/caregiving_shed0_shuf0_R0.30_isolate/rep{rep:02d}"
        failures += run_job(
            name,
            [
                "--quarantinePolicy",
                "1",
                "--closures",
                str(CLOSURE_SEED),
                "-k",
                format(k_iso, ".8g"),
                "-w",
                "6",
                "--seed",
                str(700 + rep),
                "--sheddingShape",
                "0",
                "--shuffleSchedules",
                "0",
                "--venueProfile",
                "caregiving",
                "--venueStrength",
                "1",
            ],
        )
    return failures


def exp4_calibrated():
    """Second pass. The first pass overshot generation-1 R, so these infection
    coefficients are the first-pass values times target/achieved."""
    specs = [
        ("flat", 0.0, 1, "0.30", 6.946e-05),
        ("flat", 0.0, 1, "0.20", 4.388e-05),
        ("caregiving", 0.0, 0, "0.30", 4.622e-05),
        ("caregiving", 0.0, 0, "0.20", 3.081e-05),
        ("caregiving", 1.0, 0, "0.30", 3.336e-05),
        ("caregiving", 1.0, 0, "0.20", 2.224e-05),
    ]
    failures = 0
    for profile, shed, shuffle, target, k_value in specs:
        for rep in range(1, 13):
            name = f"exp4cal/{profile}_shed{shed:g}_shuf{shuffle}_R{target}/rep{rep:02d}"
            failures += run_job(
                name,
                [
                    "--quarantinePolicy",
                    "0",
                    "--closures",
                    str(CLOSURE_SEED),
                    "-k",
                    format(k_value, ".8g"),
                    "-w",
                    "6",
                    "--seed",
                    str(800 + rep),
                    "--sheddingShape",
                    format(shed, ".8g"),
                    "--shuffleSchedules",
                    str(shuffle),
                    "--venueProfile",
                    "off" if profile == "flat" else "caregiving",
                    "--venueStrength",
                    "1",
                ],
            )
    for rep in range(1, 13):
        name = f"exp4cal/caregiving_shed0_shuf0_R0.30_isolate/rep{rep:02d}"
        failures += run_job(
            name,
            [
                "--quarantinePolicy",
                "1",
                "--closures",
                str(CLOSURE_SEED),
                "-k",
                format(4.622e-05, ".8g"),
                "-w",
                "6",
                "--seed",
                str(900 + rep),
                "--sheddingShape",
                "0",
                "--shuffleSchedules",
                "0",
                "--venueProfile",
                "caregiving",
                "--venueStrength",
                "1",
            ],
        )
    return failures


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "cal"
    failures = 0
    if phase in ("exp4", "all"):
        failures += exp4()
    if phase in ("cal", "all"):
        failures += exp4_calibrated()
    if phase in ("exp2", "cal", "all"):
        failures += exp2()
    print("failures", failures, flush=True)
    sys.exit(1 if failures else 0)
