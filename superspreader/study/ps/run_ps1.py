#!/usr/bin/env python3
"""PS1: persistent trait against the per-episode multiplier (ACT-049 instructions, section 4).

Stages: python3 run_ps1.py [core|rest|all]
  core  calibrate A0, E, P0, P100, P100ln; pilot seeds 1101-1104; stop if H1 fails
  rest  remaining arms, the shape solve for P25k, seeds 1101-1112
  all   core, then rest only if H1 passes

One job at a time. Finished jobs are skipped.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import load_daily, load_trace, mom_k  # noqa: E402
from early_window import early_counts  # noqa: E402
from ps_analysis import json_ready, load_trace as load_extended_trace, replicate_dirs, t_interval, window_counts  # noqa: E402
from ps_common import BASE_K, RESULTS, base_args, fmt, log_event, run_job  # noqa: E402

CAL_SEEDS = (9001, 9002)
PILOT_SEEDS = (1101, 1102, 1103, 1104)
FINAL_SEEDS = tuple(range(1101, 1113))
WEEKS = 10
WINDOWS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50))
SATURATED = 30000

CORE = ("A0", "E", "P0", "P100", "P100ln")


def trait_flags(shape, rho, dist="gamma"):
    flags = ["--traitShape", fmt(shape), "--traitPersist", "1", "--traitSusceptibilityExp", fmt(rho)]
    if dist != "gamma":
        flags = ["--traitDist", dist, *flags]
    return flags


def fixed_arms():
    """Arms whose trait shape is specified in advance. k0 is the secant start."""
    return [
        {"id": "A0", "flags": [], "k0": 0.00041, "core": True},
        {"id": "E", "flags": ["--sheddingShape", "0.3"], "k0": 0.000483661, "core": True},
        {"id": "P0", "flags": trait_flags(0.3, 0), "k0": 0.000483661, "core": True},
        {"id": "P25", "flags": trait_flags(0.3, 0.25), "k0": 0.00065, "core": False},
        {"id": "P50", "flags": trait_flags(0.3, 0.5), "k0": 0.0009, "core": False},
        {"id": "P100", "flags": trait_flags(0.3, 1), "k0": 0.0014, "core": True},
        {"id": "P100ln", "flags": trait_flags(0.3, 1, "lognormal"), "k0": 0.0018, "core": True},
        {"id": "P50a005", "flags": trait_flags(0.05, 0.5), "k0": 0.0012, "core": False},
        {"id": "P100a005", "flags": trait_flags(0.05, 1), "k0": 0.002, "core": False},
    ]


def arm_by_id(arm_id):
    for arm in fixed_arms():
        if arm["id"] == arm_id:
            return arm
    raise KeyError(arm_id)


def cache_path(label):
    return RESULTS / "cal" / label / "k.json"


def early_r(run_dir):
    trace = load_trace(run_dir / "trace.tsv")
    if len(trace) > SATURATED:
        return float("inf")
    counts = early_counts(trace)
    return sum(counts) / len(counts) if counts else 0.0


def pooled_early_k(run_dirs):
    counts = []
    for run_dir in run_dirs:
        trace_path = Path(run_dir) / "trace.tsv"
        if not trace_path.exists():
            continue
        counts.extend(early_counts(load_trace(trace_path)))
    if len(counts) < 2:
        return None, 0
    return mom_k(counts), len(counts)


def calibrate(label, flags, k0, target=1.39, tol=0.05, max_iter=9):
    """Log-log secant on the infection coefficient. Returns (k, measured early R, iteration)."""
    cache = cache_path(label)
    if cache.exists():
        stored = json.loads(cache.read_text())
        print(f"cal {label} cached k={stored['k']:.6g} R={stored['measured']:.4f}", flush=True)
        return stored["k"], stored["measured"], stored.get("iteration")
    points = []
    k = k0
    accepted = None
    for it in range(max_iter):
        values = []
        for seed in CAL_SEEDS:
            name = f"cal/{label}/it{it:02d}_s{seed}"
            code = run_job(name, base_args(k, seed, 4) + flags)
            if code:
                raise SystemExit(code)
            values.append(early_r(RESULTS / name))
        value = sum(values) / len(values)
        log_event({"experiment": "PS1", "label": label, "iteration": it, "k": k, "measured_R": value, "target": target})
        print(f"cal {label} it{it} k={k:.6g} R={value:.4f} target={target}", flush=True)
        if math.isinf(value):
            k *= 0.15
            continue
        if value <= 0 or math.isnan(value):
            k *= 3.0
            continue
        points.append((k, value, it))
        if abs(value / target - 1.0) <= tol:
            accepted = points[-1]
            break
        slope = 1.0
        if len(points) >= 2:
            (k1, v1, _), (k2, v2, _) = points[-2], points[-1]
            if k1 != k2 and v1 > 0 and v2 > 0:
                slope = math.log(v2 / v1) / math.log(k2 / k1)
                slope = min(3.0, max(0.4, slope))
        k = k * (target / value) ** (1.0 / slope)
    if accepted is None:
        if not points:
            raise RuntimeError(f"calibration {label} never produced a usable measurement")
        accepted = min(points, key=lambda p: abs(p[1] / target - 1.0))
        print(f"cal {label} STOPPED short of tolerance; best k={accepted[0]:.6g} R={accepted[1]:.4f}", flush=True)
    k, value, it = accepted
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"k": k, "measured": value, "target": target, "iteration": it, "reached": abs(value / target - 1) <= tol}) + "\n")
    return k, value, it


def calibration_dirs(label, iteration):
    return [RESULTS / f"cal/{label}/it{iteration:02d}_s{seed}" for seed in CAL_SEEDS]


def run_replicates(arm_id, flags, k, seeds):
    for seed in seeds:
        extra = []
        if seed == FINAL_SEEDS[0] and any(flag == "--traitShape" for flag in flags):
            extra = [("agents_trait.csv", "--traitDump")]
        code = run_job(f"ps1/{arm_id}/rep{seed}", base_args(k, seed, WEEKS) + list(flags), extra_outputs=extra)
        if code:
            raise SystemExit(code)


def cell_early_k(arm_id):
    counts = []
    for run in replicate_dirs(RESULTS / "ps1" / arm_id):
        counts.extend(early_counts(load_trace(run / "trace.tsv")))
    if len(counts) < 2:
        return None, 0
    return mom_k(counts), len(counts)


def final_sizes(arm_id):
    sizes = []
    for run in replicate_dirs(RESULTS / "ps1" / arm_id):
        rows = load_daily(run / "stdout.txt")
        if rows:
            sizes.append(int(rows[-1].get("INF", 0)))
    return sizes


def cohort_k_intervals(arm_id):
    """Replicate-level t interval of cohort k in each infection-day window."""
    out = []
    runs = replicate_dirs(RESULTS / "ps1" / arm_id)
    for first, last in WINDOWS:
        ks = []
        for run in runs:
            counts = [item[1] for item in window_counts(load_extended_trace(run / "trace.tsv"), first, last)]
            ks.append(mom_k(counts) if len(counts) >= 2 else None)
        out.append({"window": [first, last], "per_replicate": ks, "t": t_interval(ks)})
    return out


def intervals_overlap(a, b):
    if not a or not b or a.get("low") is None or b.get("low") is None:
        return None
    return a["low"] <= b["high"] and b["low"] <= a["high"]


def check_h1():
    k_e, n_e = cell_early_k("E")
    k_p, n_p = cell_early_k("P0")
    size_e, size_p = final_sizes("E"), final_sizes("P0")
    ratio_k = k_p / k_e if k_e else None
    ratio_size = (sum(size_p) / len(size_p)) / (sum(size_e) / len(size_e)) if size_e and size_p else None
    cohorts = []
    overlap_fail = False
    for left, right in zip(cohort_k_intervals("P0"), cohort_k_intervals("E")):
        overlap = intervals_overlap(left["t"], right["t"])
        if overlap is False:
            overlap_fail = True
        cohorts.append({"window": left["window"], "P0": left["t"], "E": right["t"], "overlap": overlap})
    k_ok = ratio_k is not None and 0.75 <= ratio_k <= 1.33
    size_ok = ratio_size is not None and 0.90 <= ratio_size <= 1.10
    report = {
        "early_k_E": k_e, "early_k_P0": k_p, "infectors_E": n_e, "infectors_P0": n_p,
        "k_ratio_P0_over_E": ratio_k, "k_ratio_in_band": k_ok,
        "final_size_E": size_e, "final_size_P0": size_p, "size_ratio": ratio_size, "size_ratio_in_band": size_ok,
        "cohort_k": cohorts, "cohort_intervals_fail": overlap_fail,
        "pass": bool(k_ok and size_ok and not overlap_fail),
    }
    (RESULTS / "ps1" / "h1_pilot.json").write_text(json.dumps(json_ready(report), indent=1) + "\n")
    print(
        f"H1 pilot k P0/E={ratio_k:.3f} (band 0.75-1.33, {'pass' if k_ok else 'FAIL'}) "
        f"size ratio={ratio_size:.3f} (band 0.90-1.10, {'pass' if size_ok else 'FAIL'}) "
        f"cohort overlap fail={overlap_fail}",
        flush=True,
    )
    return report["pass"]


def calibrate_ids(ids):
    solved = {}
    for arm in fixed_arms():
        if arm["id"] not in ids:
            continue
        k, measured, it = calibrate(arm["id"], arm["flags"], arm["k0"])
        k_early, n = pooled_early_k(calibration_dirs(arm["id"], it))
        solved[arm["id"]] = {"k": k, "early_R": measured, "early_k": k_early, "infectors": n}
        print(f"arm {arm['id']} coefficient={k:.6g} early R={measured:.3f} early k={k_early}", flush=True)
    return solved


def solve_p25k(target_k):
    """Alternate coefficient and shape search, at most six passes. Target is arm E's early k."""
    cache = RESULTS / "ps1" / "P25k_solution.json"
    if cache.exists():
        return json.loads(cache.read_text())
    shape = 0.08
    k0 = 0.0007
    points = []
    best = None
    for attempt in range(6):
        label = f"P25k_a{fmt(shape)}"
        flags = trait_flags(shape, 0.25)
        k, measured_r, it = calibrate(label, flags, k0)
        k0 = k
        measured_k, n = pooled_early_k(calibration_dirs(label, it))
        row = {"attempt": attempt, "shape": shape, "coefficient": k, "early_R": measured_r, "early_k": measured_k, "infectors": n}
        log_event({"experiment": "PS1", "label": "P25k_shape", **row, "target_k": target_k})
        print(f"P25k attempt {attempt} shape={shape:.4g} k_coef={k:.6g} early k={measured_k} target={target_k:.4g}", flush=True)
        if measured_k is None or math.isinf(measured_k):
            shape = min(2.0, shape * 1.5) if measured_k is not None and math.isinf(measured_k) else shape * 0.7
            continue
        points.append((shape, measured_k))
        best = row
        if abs(measured_k / target_k - 1.0) <= 0.15:
            best["reached"] = True
            break
        slope = 1.0
        if len(points) >= 2:
            (a1, k1), (a2, k2) = points[-2], points[-1]
            if a1 != a2 and k1 > 0 and k2 > 0:
                slope = math.log(k2 / k1) / math.log(a2 / a1)
                slope = min(3.0, max(0.4, abs(slope))) * (1 if slope > 0 else -1)
                if slope < 0:
                    slope = 0.4
        shape = min(2.0, max(0.02, shape * (target_k / measured_k) ** (1.0 / slope)))
    if best is None:
        raise RuntimeError("P25k shape search produced no measurement")
    best["reached"] = best.get("reached", False)
    best["target_k"] = target_k
    best["flags"] = trait_flags(best["shape"], 0.25)
    cache.write_text(json.dumps(best) + "\n")
    return best


def stage_core():
    print("PS1 core: calibrate A0, E, P0, P100, P100ln, then pilot seeds 1101-1104", flush=True)
    solved = calibrate_ids(CORE)
    (RESULTS / "ps1").mkdir(parents=True, exist_ok=True)
    (RESULTS / "ps1" / "core_calibration.json").write_text(json.dumps(solved, indent=1) + "\n")
    for arm_id in CORE:
        arm = arm_by_id(arm_id)
        run_replicates(arm_id, arm["flags"], solved[arm_id]["k"], PILOT_SEEDS)
    passed = check_h1()
    if not passed:
        print("H1 failed on the pilot. Stopping before the rest of PS1, as specified.", flush=True)
        raise SystemExit(2)
    print("H1 passed on the pilot.", flush=True)


def stage_rest():
    remaining = [arm["id"] for arm in fixed_arms() if arm["id"] not in CORE]
    print(f"PS1 rest: calibrate {remaining}, solve P25k, run seeds 1101-1112", flush=True)
    solved = calibrate_ids(remaining)
    k_e, n_e = cell_early_k("E")
    if not k_e:
        raise RuntimeError("arm E has no pilot replicates; run the core stage first")
    print(f"P25k target is arm E early k {k_e:.4g} from {n_e} infectors", flush=True)
    solution = solve_p25k(k_e)
    solved["P25k"] = solution
    (RESULTS / "ps1" / "rest_calibration.json").write_text(json.dumps(solved, indent=1) + "\n")
    for arm in fixed_arms():
        k = json.loads(cache_path(arm["id"]).read_text())["k"]
        # Pilot seeds of the core arms are already done; run_job skips them.
        run_replicates(arm["id"], arm["flags"], k, FINAL_SEEDS)
    run_replicates("P25k", solution["flags"], solution["coefficient"], FINAL_SEEDS)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(
        f"Budget gate: a 4-week calibration job is about 15s and a 10-week job about 40s. "
        f"Core calibration plus the 4-replicate pilot is about half an hour; the full PS1 grid "
        f"(10 arms, 12 seeds, plus the shape search) is about 2 to 3 hours on one GPU. "
        f"That is within reach, so the full grid is being run. Baseline coefficient {BASE_K:.6g}.",
        flush=True,
    )
    if stage in ("core", "all"):
        stage_core()
    if stage in ("rest", "all"):
        stage_rest()
    print("PS1 runs complete", flush=True)


if __name__ == "__main__":
    main()
