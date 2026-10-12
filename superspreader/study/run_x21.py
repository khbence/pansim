#!/usr/bin/env python3
"""Experiments from 21_further_experiments.md.

  python3 run_x21.py x1     ward-split subcritical chains (flat shape 0.04, caregiving shape 0.09)
  python3 run_x21.py x5     recalibrate the lognormal full-coupling arm onto P100's early R
  python3 run_x21.py x2     matched-size levers, then the household-share gradient
  python3 run_x21.py x3     Nipah-like chain-size versus dispersion calibration
  python3 run_x21.py all    x1, x5, x2, x3 in that order

X4 is not run: note 21 says to run it only if the paper claims the lever ranking is
general, and the draft does not. X6 needs a new simulator output and is not a run.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import load_daily, load_trace, mom_k, venue_metrics  # noqa: E402
from early_window import early_counts  # noqa: E402
from run_phase2 import (  # noqa: E402
    LEVERS,
    all_node_mean,
    chain_args,
    early_r,
    fmt,
    mech_args,
    open_epidemic_args,
    write_lever_files,
    write_seed_file,
)
from run_study import BINARY, CLOSURE_NONE, INF, PROG, ROOT  # noqa: E402

RESULTS = HERE / "results" / "x21"
WARDS = HERE / "inputs" / "x21"
AGENTS = WARDS / "agents_wards.json"
LOCATIONS = WARDS / "locations_wards.json"
LOG = RESULTS / "calibration_log.json"
CHAIN_REPS = 40
LEVER_REPS = 4


def log_event(entry):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(LOG.read_text()) if LOG.exists() else []
    data.append(entry)
    LOG.write_text(json.dumps(data, indent=1) + "\n")


def run_job(name, args, agents=None, locations=None):
    out = RESULTS / name
    stdout_path = out / "stdout.txt"
    trace_path = out / "trace.tsv"
    if stdout_path.exists() and trace_path.exists() and trace_path.stat().st_size > 0:
        print(f"skip {name}", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BINARY),
        "--progression", "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "-A", "inputConfigFiles/agentTypes_3.json",
        "-a", str(agents or (ROOT / "inputRealExample" / "agents1.json")),
        "-l", str(locations or (ROOT / "inputRealExample" / "locations0.json")),
        "--infectiousnessMultiplier", INF,
        "--diseaseProgressionScaling", PROG,
        "--infectionTrace", str(trace_path),
        *args,
    ]
    print("run " + name, flush=True)
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


def early_window_r(run_dir):
    """Early R without the 30,000-infection saturation guard used during short calibrations."""
    trace = load_trace(run_dir / "trace.tsv")
    counts = early_counts(trace)
    return sum(counts) / len(counts) if counts else 0.0


def final_size(run_dir):
    rows = load_daily(run_dir / "stdout.txt")
    return int(rows[-1].get("INF", 0)) if rows else 0


def household_share(run_dir):
    return venue_metrics(load_trace(run_dir / "trace.tsv"))["household_fraction"]


def calibrate(label, make_args, measure, target, k0, seeds, tol=0.05, max_iter=8, **job):
    cache = RESULTS / "cal" / label / "k.json"
    if cache.exists():
        stored = json.loads(cache.read_text())
        print(f"cal {label} cached k={stored['k']:.6g} measured={stored['measured']:.4f}", flush=True)
        return stored["k"], stored["measured"]
    points = []
    k = k0
    for it in range(max_iter):
        values = []
        for seed in seeds:
            name = f"cal/{label}/it{it:02d}_s{seed}"
            code = run_job(name, make_args(k, seed), **job)
            if code:
                raise SystemExit(code)
            values.append(measure(RESULTS / name))
        value = sum(values) / len(values)
        log_event({"label": label, "iteration": it, "k": k, "measured": value, "target": target})
        print(f"cal {label} it{it} k={k:.6g} measured={value:.4f} target={target}", flush=True)
        if math.isinf(value):
            k *= 0.15
            continue
        if value <= 0 or math.isnan(value):
            k *= 3.0 if value <= 0 else 0.5
            continue
        points.append((k, value, it))
        if abs(value / target - 1.0) <= tol:
            break
        slope = 1.0
        if len(points) >= 2:
            (k1, v1, _), (k2, v2, _) = points[-2], points[-1]
            if k1 != k2 and v1 > 0 and v2 > 0:
                slope = min(3.0, max(0.4, math.log(v2 / v1) / math.log(k2 / k1)))
        k = k * (target / value) ** (1.0 / slope)
    else:
        pass
    if not points:
        raise RuntimeError(f"calibration {label} produced no measurement")
    best_k, best_value, _ = min(points, key=lambda item: abs(item[1] / target - 1.0))
    # Prefer the last accepted point when it is inside tolerance; otherwise the closest.
    if abs(points[-1][1] / target - 1.0) <= tol:
        best_k, best_value = points[-1][0], points[-1][1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"k": best_k, "measured": best_value, "target": target}) + "\n")
    return best_k, best_value


def repeat(prefix, make_args, k, seeds, **job):
    for seed in seeds:
        code = run_job(f"{prefix}/rep{seed:02d}" if seed < 100 else f"{prefix}/rep{seed}", make_args(k, seed), **job)
        if code:
            raise SystemExit(code)


def x1():
    if not AGENTS.exists() or not LOCATIONS.exists():
        raise SystemExit("ward inputs missing; run build_wards.py first")
    write_seed_file()
    job = {"agents": AGENTS, "locations": LOCATIONS}
    arms = [
        ("flat_shed0.04", mech_args(0.04, 1, "off", 1, 1.0), 9.97e-5),
        ("caregiving_shed0.09", mech_args(0.09, 0, "caregiving", 1, 1.0), 6.69e-5),
    ]
    for label, mech, k0 in arms:
        make = chain_args(mech)
        k, _ = calibrate(f"x1_{label}", lambda k, seed, make=make: make(k, seed, 7), all_node_mean, 0.33, k0, (41, 42), **job)
        for rep in range(1, CHAIN_REPS + 1):
            code = run_job(f"x1/{label}/rep{rep:02d}", make(k, 8000 + rep, 7), **job)
            if code:
                raise SystemExit(code)


def p100_early_r():
    summary = json.loads((HERE / "results" / "ps" / "ps1" / "summary.json").read_text())
    for arm in summary["arms"]:
        if arm["arm"] == "P100":
            return arm["early_R"]
    raise KeyError("P100")


def trait_flags():
    return ["--traitDist", "lognormal", "--traitShape", "0.3", "--traitPersist", "1", "--traitSusceptibilityExp", "1"]


def ps1_args(flags):
    def build(k, seed, weeks):
        return [
            "-r", "--quarantinePolicy", "0", "--closures", str(CLOSURE_NONE),
            "-k", fmt(k), "-w", str(weeks), "--seed", str(seed),
            "--densityExponent", "1", "--venueProfile", "off", *flags,
        ]
    return build


def x5():
    target = p100_early_r()
    flags = trait_flags()
    make4 = ps1_args(flags)
    k, measured = calibrate(
        "x5_P100ln", lambda k, seed, make4=make4: make4(k, seed, 4), early_r, target, 0.000184097, (1101, 1102), tol=0.04,
    )
    for seed in range(1101, 1113):
        code = run_job(f"x5/P100ln/rep{seed}", make4(k, seed, 10))
        if code:
            raise SystemExit(code)
    values = [early_window_r(RESULTS / f"x5/P100ln/rep{seed}") for seed in range(1101, 1113)]
    realised = sum(values) / len(values)
    print(f"x5 twelve-replicate early R {realised:.4f} target {target:.4f}", flush=True)
    (RESULTS / "x5" / "first_pass.json").write_text(json.dumps({"k": k, "two_seed_R": measured, "twelve_rep_R": realised, "target": target}) + "\n")
    if abs(realised / target - 1.0) > 0.08:
        k2 = k * (target / realised)
        print(f"x5 rescale k {k:.6g} -> {k2:.6g}", flush=True)
        for seed in range(1101, 1113):
            code = run_job(f"x5/P100ln_scaled/rep{seed}", make4(k2, seed, 10))
            if code:
                raise SystemExit(code)
        values2 = [early_window_r(RESULTS / f"x5/P100ln_scaled/rep{seed}") for seed in range(1101, 1113)]
        (RESULTS / "x5" / "scaled_pass.json").write_text(json.dumps({"k": k2, "twelve_rep_R": sum(values2) / len(values2), "target": target}) + "\n")


def score_pair(left, right):
    if min(left["early_R"], right["early_R"], left["final_size"], right["final_size"]) <= 0:
        return 1e9
    return abs(math.log(left["final_size"] / right["final_size"])) * 2.0 + abs(math.log(left["early_R"] / right["early_R"]))


def x2():
    """Match Set I and Set II on final size and early R as far as one coefficient allows, then repeat the five levers.

    Separately, pilot mechanisms and keep the three whose household shares land nearest 0.25, 0.37 and 0.45.
    The sign prediction is fixed here, before those lever runs: venue-closure reduction exceeds household
    isolation below a household share of 0.37, and household isolation exceeds venue closure above it.
    """
    write_lever_files()
    levers = ["none", "masks", "venue_close", "isolate", "household"]
    arms = {
        "I_bio": (mech_args(1.0, 0, "off", 1, 1.0), 0.000403805),
        "II_set": (mech_args(0, 0, "covid", 8, 1.0), 5.65578e-5),
    }
    factors = (0.45, 0.7, 1.0, 1.4, 2.0)
    frontier = {arm: [] for arm in arms}
    for arm, (mech, k0) in arms.items():
        make = open_epidemic_args(mech)
        for factor in factors:
            k = k0 * factor
            points = []
            for seed in (71, 72):
                name = f"x2/frontier/{arm}/k{factor:.2f}/s{seed}"
                code = run_job(name, make(k, seed, 12))
                if code:
                    raise SystemExit(code)
                run_dir = RESULTS / name
                points.append({"early_R": early_window_r(run_dir), "final_size": final_size(run_dir)})
            frontier[arm].append({
                "k": k,
                "factor": factor,
                "early_R": sum(item["early_R"] for item in points) / 2,
                "final_size": sum(item["final_size"] for item in points) / 2,
            })
            print(f"frontier {arm} k={k:.6g} R={frontier[arm][-1]['early_R']:.3f} size={frontier[arm][-1]['final_size']:.0f}", flush=True)
    best = None
    for left in frontier["I_bio"]:
        for right in frontier["II_set"]:
            scored = score_pair(left, right)
            if best is None or scored < best[0]:
                best = (scored, left, right)
    chosen = {"I_bio": best[1], "II_set": best[2], "score": best[0], "prediction": "venue_close_minus_household_isolation changes sign near household share 0.37"}
    (RESULTS / "x2").mkdir(parents=True, exist_ok=True)
    (RESULTS / "x2" / "match.json").write_text(json.dumps({"frontier": frontier, "chosen": chosen}, indent=1) + "\n")
    for arm, (mech, _) in arms.items():
        make = open_epidemic_args(mech)
        k = chosen[arm]["k"]
        for lever in levers:
            closures = LEVERS / f"{lever}.json"
            build = open_epidemic_args(mech, closures=closures)
            for rep in range(1, LEVER_REPS + 1):
                code = run_job(f"x2/matched_{arm}_{lever}/rep{rep:02d}", build(k, 9000 + rep, 12))
                if code:
                    raise SystemExit(code)
    # Household-share pilots. Shares are measured on the uncontrolled runs of the same arm.
    candidates = [
        ("venue_s3", mech_args(0, 0, "covid", 3, 1.0), 8e-5),
        ("venue_s1.5", mech_args(0, 0, "covid", 1.5, 1.0), 1.8e-4),
        ("venue_s1", mech_args(0, 0, "covid", 1, 1.0), 3e-4),
        ("shape0.5", mech_args(0.5, 0, "off", 1, 1.0), 4.5e-4),
        ("shape0.3", mech_args(0.3, 0, "off", 1, 1.0), 5.5e-4),
        ("shape1_venue1", mech_args(1.0, 0, "covid", 1, 1.0), 2.5e-4),
    ]
    pilot = []
    for label, mech, k0 in candidates:
        make = open_epidemic_args(mech)
        k, _ = calibrate(f"x2_{label}", lambda k, seed, make=make: make(k, seed, 4), early_r, 1.3, k0, (81, 82))
        shares = []
        for seed in (83, 84):
            name = f"x2/pilot_{label}/s{seed}"
            code = run_job(name, make(k, seed, 12))
            if code:
                raise SystemExit(code)
            shares.append(household_share(RESULTS / name))
        share = sum(shares) / len(shares)
        pilot.append({"label": label, "mech": mech, "k": k, "household_share": share})
        print(f"pilot {label} share {share:.3f}", flush=True)
    targets = (0.25, 0.37, 0.45)
    chosen_pilots = []
    used = set()
    for target_share in targets:
        ranked = sorted(pilot, key=lambda item: abs(item["household_share"] - target_share))
        pick = next(item for item in ranked if item["label"] not in used)
        used.add(pick["label"])
        chosen_pilots.append({**pick, "target_share": target_share, "mech": None})
    # mech is not JSON serialisable; store labels and rebuild.
    mech_by_label = {label: mech for label, mech, _ in candidates}
    (RESULTS / "x2" / "pilot.json").write_text(json.dumps({
        "pilot": [{k: v for k, v in item.items() if k != "mech"} for item in pilot],
        "chosen": [{k: v for k, v in item.items() if k != "mech"} for item in chosen_pilots],
        "sign_prediction": "venue closure reduces more than household isolation when household share is below 0.37; household isolation reduces more above 0.37",
    }, indent=1) + "\n")
    k_by_label = {item["label"]: item["k"] for item in pilot}
    for item in chosen_pilots:
        mech = mech_by_label[item["label"]]
        k = k_by_label[item["label"]]
        for lever in levers:
            closures = LEVERS / f"{lever}.json"
            build = open_epidemic_args(mech, closures=closures)
            for rep in range(1, LEVER_REPS + 1):
                code = run_job(f"x2/share_{item['label']}_{lever}/rep{rep:02d}", build(k, 9100 + rep, 12))
                if code:
                    raise SystemExit(code)


def chain_moments(run_dir):
    trace = load_trace(run_dir / "trace.tsv")
    # All-node offspring: every infected agent, children anywhere in the trace.
    children = {}
    for row in trace:
        if row["infector"] is not None:
            children[row["infector"]] = children.get(row["infector"], 0) + 1
    counts = [children.get(row["agent"], 0) for row in trace]
    zeros = sum(1 for value in counts if value == 0)
    return {
        "nodes": len(counts),
        "mean": sum(counts) / len(counts) if counts else None,
        "k": mom_k(counts) if len(counts) > 2 else None,
        "zero_share": zeros / len(counts) if counts else None,
    }


def x3():
    write_seed_file()
    shapes = (0.02, 0.025, 0.03, 0.035)
    grid = []
    reference = json.loads((HERE / "summaries" / "p2_chains.json").read_text())["chain_flat_shed0.04"]
    grid.append({
        "shape": 0.04,
        "k_coeff": reference["calibrated"]["k"],
        "mean": reference["all_node_mean_offspring"],
        "dispersion_k": reference["all_node_k_mom"],
        "zero_share": reference["zero_share"],
        "source": "existing 40-replicate flat_shed0.04",
    })
    for shape in shapes:
        mech = mech_args(shape, 1, "off", 1, 1.0)
        make = chain_args(mech)
        k0 = 9.97e-5 * (0.04 / shape)
        k, measured = calibrate(
            f"x3_shape{shape:g}", lambda k, seed, make=make: make(k, seed, 7), all_node_mean, 0.33, k0, (51, 52),
        )
        samples = []
        for seed in (53, 54):
            name = f"x3/probe_shape{shape:g}/s{seed}"
            code = run_job(name, make(k, seed, 7))
            if code:
                raise SystemExit(code)
            samples.append(chain_moments(RESULTS / name))
        grid.append({
            "shape": shape,
            "k_coeff": k,
            "mean": measured,
            "dispersion_k": sum(item["k"] for item in samples) / 2,
            "zero_share": sum(item["zero_share"] for item in samples) / 2,
            "source": "two probe runs",
        })
        print(f"x3 shape {shape:g} k_disp {grid[-1]['dispersion_k']:.4f} zero {grid[-1]['zero_share']:.3f}", flush=True)
    finite = [item for item in grid if item["dispersion_k"] and math.isfinite(item["dispersion_k"])]
    shape_k = min(finite, key=lambda item: abs(item["dispersion_k"] - 0.036))
    shape_z = min(grid, key=lambda item: abs(item["zero_share"] - 0.92))
    (RESULTS / "x3").mkdir(parents=True, exist_ok=True)
    (RESULTS / "x3" / "grid.json").write_text(json.dumps({"grid": grid, "dispersion_arm": shape_k, "zero_arm": shape_z}, indent=1) + "\n")
    chosen = []
    for item in (shape_k, shape_z):
        if item["shape"] == 0.04 or item["shape"] in chosen:
            continue
        chosen.append(item["shape"])
        mech = mech_args(item["shape"], 1, "off", 1, 1.0)
        make = chain_args(mech)
        for rep in range(1, CHAIN_REPS + 1):
            code = run_job(f"x3/shape{item['shape']:g}/rep{rep:02d}", make(item["k_coeff"], 8200 + rep, 7))
            if code:
                raise SystemExit(code)


def main():
    phases = {"x1": x1, "x5": x5, "x2": x2, "x3": x3}
    names = sys.argv[1:] or ["x1"]
    if names == ["all"]:
        names = ["x1", "x5", "x2", "x3"]
    for name in names:
        print("==", name, flush=True)
        phases[name]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
