#!/usr/bin/env python3
"""Phase 2: experiments re-run at matched early growth, per 07/08.

Every comparison is made after calibrating the infection coefficient of each arm to the
same early reproduction number, measured on index cases infected on days 1-10 (children
within 14 days). Calibration is deterministic given the seeds, and run_job skips finished
jobs, so an interrupted run resumes where it stopped.

Phases: kernel, grid, audit, lever, chains, waves
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import hospital, load_daily, load_trace  # noqa: E402
from early_window import early_counts  # noqa: E402
from run_study import BASE_K, CLOSURE_FITTED, CLOSURE_NONE, RESULTS, run_job  # noqa: E402

CLOSURE_SEED1000 = HERE / "closures_seed1000.json"
LEVERS = HERE / "levers"
LOG = RESULTS / "p2_calibration_log.json"
BASELINE_EARLY_R = 1.39


def fmt(value):
    return format(value, ".6g")


def log_event(entry):
    data = json.loads(LOG.read_text()) if LOG.exists() else []
    data.append(entry)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(data, indent=1) + "\n")


SATURATED = 30000


def early_r(run_dir):
    trace = load_trace(run_dir / "trace.tsv")
    if len(trace) > SATURATED:
        return float("inf")
    counts = early_counts(trace)
    return sum(counts) / len(counts) if counts else 0.0


def all_node_mean(run_dir):
    trace = load_trace(run_dir / "trace.tsv")
    nodes = len(trace)
    seeds = sum(1 for row in trace if row["infector"] is None)
    return (nodes - seeds) / nodes if nodes else float("nan")


def hospital_area(run_dir, days=105):
    rows = load_daily(run_dir / "stdout.txt")[:days]
    return sum(hospital(row) for row in rows)


def calibrate(label, make_args, measure, target, k0, seeds, weeks, tol=0.05, max_iter=9):
    """Secant search in log-log. make_args(k, seed, weeks) -> args after the binary prefix."""
    points = []
    k = k0
    for it in range(max_iter):
        values = []
        for seed in seeds:
            name = f"p2cal/{label}/it{it:02d}_s{seed}"
            run_job(name, make_args(k, seed, weeks))
            values.append(measure(RESULTS / name))
        value = sum(values) / len(values)
        log_event({"label": label, "iteration": it, "k": k, "measured": value, "target": target})
        print(f"cal {label} it{it} k={k:.6g} measured={value:.4f} target={target}", flush=True)
        if math.isinf(value):
            k *= 0.15
            continue
        if math.isnan(value):
            k *= 0.5
            continue
        if value <= 0:
            k *= 3.0
            continue
        points.append((k, value))
        if abs(value / target - 1.0) <= tol:
            return k, value
        slope = 1.0
        if len(points) >= 2:
            (k1, v1), (k2, v2) = points[-2], points[-1]
            if k1 != k2 and v1 > 0 and v2 > 0:
                slope = math.log(v2 / v1) / math.log(k2 / k1)
                slope = min(3.0, max(0.4, slope))
        k = k * (target / value) ** (1.0 / slope)
    if not points:
        raise RuntimeError(f"calibration {label} never produced a usable measurement")
    best = min(points, key=lambda p: abs(p[1] / target - 1.0))
    print(f"cal {label} did not reach tolerance; using k={best[0]:.6g}", flush=True)
    return best


def stored_k(label, fallback_k0, seeds, make_args, measure, target, weeks):
    """Cache the calibrated k next to the log so reruns do not repeat the search."""
    cache = RESULTS / "p2cal" / label / "k.json"
    if cache.exists():
        return json.loads(cache.read_text())["k"]
    k, value = calibrate(label, make_args, measure, target, fallback_k0, seeds, weeks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"k": k, "measured": value, "target": target}) + "\n")
    return k


def mech_args(shed, shuffle, venue, strength, alpha):
    return [
        "--sheddingShape", fmt(shed),
        "--shuffleSchedules", str(shuffle),
        "--venueProfile", venue,
        "--venueStrength", fmt(strength),
        "--densityExponent", fmt(alpha),
    ]


def open_epidemic_args(mech, closures=None, quarantine=0):
    def build(k, seed, weeks):
        return [
            "-r",
            "--quarantinePolicy", str(quarantine),
            "--closures", str(closures or CLOSURE_NONE),
            "-k", fmt(k),
            "-w", str(weeks),
            "--seed", str(seed),
            *mech,
        ]
    return build


def kernel_and_grid_cells():
    cells = []
    for alpha, k0 in ((1.0, BASE_K), (0.5, BASE_K / 25), (0.0, BASE_K / 1500)):
        cells.append((f"kernel_alpha{alpha:g}", mech_args(0, 0, "off", 1, alpha), k0))
    for shed_name, shed, k_shed in (("off", 0, 1.0), ("1", 1, 1.05), ("0.3", 0.3, 1.15), ("0.1", 0.1, 1.3)):
        for venue, strength, k_venue in (("off", 1, 1.0), ("covid", 1, 0.7)):
            cells.append(
                (f"grid_shed{shed_name}_venue{venue}", mech_args(shed, 0, venue, strength, 1.0), BASE_K * k_shed * k_venue)
            )
    return cells


def run_matched_cells(cells, target, reps=4, weeks=8, seed_base=1000):
    for label, mech, k0 in cells:
        k = stored_k(label, k0, (11, 12), open_epidemic_args(mech), early_r, target, 4)
        for rep in range(1, reps + 1):
            run_job(f"p2/{label}/rep{rep:02d}", open_epidemic_args(mech)(k, seed_base + rep, weeks))


def phase_kernel():
    cells = [c for c in kernel_and_grid_cells() if c[0].startswith("kernel_")]
    run_matched_cells(cells, BASELINE_EARLY_R)


def phase_grid():
    cells = [c for c in kernel_and_grid_cells() if c[0].startswith("grid_")]
    run_matched_cells(cells, BASELINE_EARLY_R)


def phase_audit():
    cells = [
        ("audit_flat_R2.5", mech_args(0, 0, "off", 1, 1.0), BASE_K * 1.8),
        ("audit_shed0.1_R2.5", mech_args(0.1, 0, "off", 1, 1.0), BASE_K * 4.0),
    ]
    run_matched_cells(cells, 2.5, reps=3, weeks=6, seed_base=2000)


def write_lever_files():
    LEVERS.mkdir(exist_ok=True)

    def rule(name, parameter="0", types=(), day=21):
        return {
            "name": name,
            "conditionType": "afterDays",
            "threshold": day,
            "threshold2": day,
            "openAfter": 400,
            "closeAfter": -1,
            "parameter": parameter,
            "locationTypesToClose": list(types),
        }

    files = {
        "none": [],
        "masks": [rule("Masks", "0.6")],
        "venue_close": [rule("sept_venue_close", "0", (5, 6, 8, 9))],
        "isolate": [rule("QuarantinePolicy", "1")],
        "household": [rule("QuarantinePolicy", "2")],
    }
    for name, rules in files.items():
        (LEVERS / f"{name}.json").write_text(json.dumps({"rules": rules}, indent=2) + "\n")
    return list(files)


def phase_lever():
    levers = write_lever_files()
    arms = {
        "I_bio": (mech_args(1.0, 0, "off", 1, 1.0), BASE_K * 0.75),
        "II_set": (mech_args(0, 0, "covid", 8, 1.0), BASE_K * 0.12),
    }
    for arm, (mech, k0) in arms.items():
        k = stored_k(f"lever_{arm}", k0, (21, 22), open_epidemic_args(mech), early_r, 1.3, 4)
        for lever in levers:
            closures = LEVERS / f"{lever}.json"
            for rep in range(1, 5):
                run_job(
                    f"p2/lever_{arm}_{lever}/rep{rep:02d}",
                    open_epidemic_args(mech, closures=closures)(k, 3000 + rep, 12),
                )


def write_seed_file():
    payload = {
        "rules": [
            {
                "name": "ExposeToMutation",
                "conditionType": "afterDays",
                "threshold": 0,
                "threshold2": 0,
                "openAfter": 1,
                "closeAfter": -1,
                "parameter": "0.0054",
                "locationTypesToClose": [],
            }
        ]
    }
    CLOSURE_SEED1000.write_text(json.dumps(payload, indent=2) + "\n")


def chain_args(mech, quarantine=0):
    def build(k, seed, weeks):
        return [
            "--quarantinePolicy", str(quarantine),
            "--closures", str(CLOSURE_SEED1000),
            "-k", fmt(k),
            "-w", str(weeks),
            "--seed", str(seed),
            *mech,
        ]
    return build


def phase_chains():
    write_seed_file()
    arms = [
        ("flat", mech_args(0, 1, "off", 1, 1.0), 6.9e-5, 0.33),
        ("caregiving", mech_args(0, 0, "caregiving", 1, 1.0), 4.7e-5, 0.33),
        ("flat_shed0.125", mech_args(0.125, 1, "off", 1, 1.0), 9e-5, 0.33),
        ("flat_shed0.04", mech_args(0.04, 1, "off", 1, 1.0), 2e-4, 0.33),
        ("caregiving_shed0.09", mech_args(0.09, 0, "caregiving", 1, 1.0), 1.5e-4, 0.33),
    ]
    for label, mech, k0, target in arms:
        k = stored_k(f"chain_{label}", k0, (31, 32), chain_args(mech), all_node_mean, target, 7)
        for rep in range(1, 5):
            run_job(f"p2/chain_{label}/rep{rep:02d}", chain_args(mech)(k, 4000 + rep, 7))
        if label in ("flat", "flat_shed0.04", "caregiving_shed0.09"):
            for rep in range(1, 5):
                run_job(f"p2/chain_{label}_isolate/rep{rep:02d}", chain_args(mech, quarantine=1)(k, 5000 + rep, 7))


def wave_args(mech):
    def build(k, seed, weeks):
        return [
            "-r",
            "--quarantinePolicy", "3",
            "--closures", str(CLOSURE_FITTED),
            "-k", fmt(k),
            "-w", str(weeks),
            "--seed", str(seed),
            *mech,
        ]
    return build


def phase_waves():
    base_area = []
    for seed in (101, 102):
        base_area.append(hospital_area(RESULTS / f"waves/baseline/rep{seed - 100:02d}"))
    target = sum(base_area) / len(base_area)
    print("baseline wave-1 hospital area", base_area, flush=True)
    for seed in (103, 104):
        run_job(f"p2/waves_baseline/rep{seed - 100:02d}", wave_args(mech_args(0, 0, "off", 1, 1.0))(BASE_K, seed, 37))
    corners = [
        ("shed0.3", mech_args(0.3, 0, "off", 1, 1.0), BASE_K * 1.6),
        ("shed0.1", mech_args(0.1, 0, "off", 1, 1.0), BASE_K * 3.5),
        ("venue1", mech_args(0, 0, "covid", 1, 1.0), BASE_K * 0.55),
        ("stack", mech_args(0.3, 0, "covid", 1, 1.0), BASE_K * 0.8),
        ("venue4", mech_args(0, 0, "covid", 4, 1.0), BASE_K * 0.15),
    ]
    for label, mech, k0 in corners:
        k = stored_k(f"waves_{label}", k0, (101, 102), wave_args(mech), hospital_area, target, 15)
        for seed in (101, 102, 103, 104):
            run_job(f"p2/waves_{label}/rep{seed - 100:02d}", wave_args(mech)(k, seed, 37))


PHASES = {
    "kernel": phase_kernel,
    "chains": phase_chains,
    "grid": phase_grid,
    "audit": phase_audit,
    "lever": phase_lever,
    "waves": phase_waves,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or ["kernel"]:
        print("== phase", name, flush=True)
        PHASES[name]()
    print("done", flush=True)
