#!/usr/bin/env python3
"""Capacity audit and volume-model experiments (12_capacity_work_package). Needs the phase-3 binary."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_phase2 import BASE_K, RESULTS, early_r, fmt, mech_args, open_epidemic_args, stored_k  # noqa: E402
from run_study import run_job  # noqa: E402


def phase_reference():
    mech = mech_args(0, 0, "off", 1, 1.0)
    args = open_epidemic_args(mech)(BASE_K, 7001, 3)
    run_job("p3/reference_uncontrolled", args + ["--occupancyStats", str(RESULTS / "p3" / "reference_uncontrolled" / "occupancy.tsv")])


def phase_reference2():
    mech = mech_args(0, 0, "off", 1, 1.0)
    args = open_epidemic_args(mech)(BASE_K, 7002, 3)
    run_job("p3/reference_uncontrolled_b", args + ["--occupancyStats", str(RESULTS / "p3" / "reference_uncontrolled_b" / "occupancy.tsv")])


INPUTS = RESULTS / "p3" / "inputs"
FACTORS = json.loads((INPUTS / "volume_type_factors.json").read_text())
ORIGINAL = "inputRealExample/locations0.json"


def volume_flags(locations, mode, scenario="mid", extra=()):
    return [
        "-l", str(locations),
        "--volumeModel", mode,
        "--volumeTypeFactors", FACTORS[scenario],
        *extra,
    ]


def volume_cells():
    peak = INPUTS / "locations_vol_peak.json"
    tmed = INPUTS / "locations_vol_typemedian.json"
    return [
        ("vol_nonres_peak_mid", volume_flags(peak, "nonresidential"), BASE_K),
        ("vol_all_peak_mid", volume_flags(peak, "all"), BASE_K),
        ("vol_nonres_raw_mid", volume_flags(ORIGINAL, "nonresidential"), BASE_K / 10),
        ("vol_nonres_peak_low", volume_flags(peak, "nonresidential", "low"), BASE_K),
        ("vol_nonres_peak_high", volume_flags(peak, "nonresidential", "high"), BASE_K),
        ("vol_nonres_fillone_mid", volume_flags(ORIGINAL, "fillone"), BASE_K),
        ("vol_nonres_tmed_mid", volume_flags(tmed, "nonresidential"), BASE_K),
        ("vol_events_r003_f1000", volume_flags(peak, "nonresidential", "mid", ["--eventRate", "0.003", "--eventFactor", "1000"]), BASE_K * 0.7),
        ("vol_events_r01_f200", volume_flags(peak, "nonresidential", "mid", ["--eventRate", "0.01", "--eventFactor", "200"]), BASE_K * 0.7),
    ]


def phase_volume(names=None):
    from run_phase2 import BASELINE_EARLY_R
    mech = mech_args(0, 0, "off", 1, 1.0)
    for label, flags, k0 in volume_cells():
        if names and label not in names:
            continue
        def make(k, seed, weeks, flags=flags):
            return open_epidemic_args(mech)(k, seed, weeks) + flags
        k = stored_k(label, k0, (11, 12), make, early_r, BASELINE_EARLY_R, 4)
        for rep in range(1, 5):
            run_job(f"p2/{label}/rep{rep:02d}", make(k, 1000 + rep, 8))


def phase_vol_waves(labels=("vol_nonres_peak_mid",)):
    from run_phase2 import hospital_area, wave_args
    cells = {c[0]: c for c in volume_cells()}
    mech = mech_args(0, 0, "off", 1, 1.0)
    target = 8565
    for label in labels:
        flags = cells[label][1]

        def make(k, seed, weeks, flags=flags):
            return wave_args(mech)(k, seed, weeks) + flags
        k = stored_k(f"waves_{label}", BASE_K * 0.8, (101, 102), make, hospital_area, target, 15)
        for seed in (101, 102, 103, 104):
            run_job(f"p2/waves_{label}/rep{seed - 100:02d}", make(k, seed, 37))


def write_ventilation_lever():
    path = HERE / "levers" / "ventilation.json"
    rule = {
        "name": "Ventilation",
        "conditionType": "afterDays",
        "threshold": 21,
        "threshold2": 21,
        "openAfter": 400,
        "closeAfter": -1,
        "parameter": "0.5",
        "locationTypesToClose": [5, 6, 8, 9],
    }
    path.write_text(json.dumps({"rules": [rule]}, indent=2) + "\n")


def phase_vol_levers(label="vol_nonres_peak_mid"):
    from run_phase2 import LEVERS, write_lever_files
    levers = write_lever_files()
    write_ventilation_lever()
    levers.append("ventilation")
    cells = {c[0]: c for c in volume_cells()}
    flags = cells[label][1]
    mech = mech_args(0, 0, "off", 1, 1.0)

    def make(closures):
        def build(k, seed, weeks):
            return open_epidemic_args(mech, closures=closures)(k, seed, weeks) + flags
        return build
    k = stored_k(f"lever_III_{label}", BASE_K, (21, 22), make(None), early_r, 1.3, 4)
    for lever in levers:
        closures = LEVERS / f"{lever}.json"
        for rep in range(1, 5):
            run_job(f"p2/lever_III_{label}_{lever}/rep{rep:02d}", make(closures)(k, 3000 + rep, 12))


PHASES = {
    "reference": phase_reference,
    "reference2": phase_reference2,
    "volume": phase_volume,
    "volwaves": phase_vol_waves,
    "vollevers": phase_vol_levers,
}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        print("== phase", name, flush=True)
        PHASES[name]()
