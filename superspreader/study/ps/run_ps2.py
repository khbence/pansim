#!/usr/bin/env python3
"""PS2: constant against peak-window infectiousness (ACT-049 instructions, section 5).

Stages: python3 run_ps2.py [profile|calibrate|runs|all]
One job at a time. Finished jobs are skipped. Run this only after PS1 has released the GPU.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import mom_k  # noqa: E402
from early_window import early_counts  # noqa: E402
from ps_analysis import load_trace  # noqa: E402
from ps_common import RESULTS, base_args, fmt, log_event, run_job  # noqa: E402

HERE = Path(__file__).resolve().parent
LEVERS = HERE.parent / "levers"
CAL_SEEDS = (9001, 9002)
SEEDS = tuple(range(2101, 2109))
WEEKS = 12
TARGET_R = 1.3
SATURATED = 80000


def early_r(run_dir):
    trace = load_trace(run_dir / "trace.tsv")
    if len(trace) > SATURATED:
        return float("inf")
    counts = early_counts(trace)
    return sum(counts) / len(counts) if counts else 0.0


def pooled_k(run_dirs):
    counts = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "trace.tsv"
        if path.exists():
            counts.extend(early_counts(load_trace(path)))
    if len(counts) < 2:
        return None, 0
    return mom_k(counts), len(counts)


def cache_path(label):
    return RESULTS / "cal" / label / "k.json"


def calibrate(label, flags, k0, target=TARGET_R, tol=0.05, max_iter=9):
    cache = cache_path(label)
    if cache.exists():
        stored = json.loads(cache.read_text())
        if abs(stored.get("target", target) - target) < 1e-6:
            print(f"cal {label} cached k={stored['k']:.6g}", flush=True)
            return stored["k"], stored["measured"], stored.get("iteration")
        print(f"cal {label} cache target {stored.get('target')} differs from {target}; recalibrating", flush=True)
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
        log_event({"experiment": "PS2", "label": label, "iteration": it, "k": k, "measured_R": value, "target": target})
        print(f"cal {label} it{it} k={k:.6g} R={value:.4f}", flush=True)
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
                slope = min(3.0, max(0.4, math.log(v2 / v1) / math.log(k2 / k1)))
        k = k * (target / value) ** (1.0 / slope)
    if accepted is None:
        if not points:
            raise RuntimeError(f"calibration {label} produced no measurement")
        accepted = min(points, key=lambda p: abs(p[1] / target - 1.0))
        print(f"cal {label} short of tolerance; best k={accepted[0]:.6g} R={accepted[1]:.4f}", flush=True)
    k, value, it = accepted
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"k": k, "measured": value, "target": target, "iteration": it, "reached": abs(value / target - 1) <= tol}) + "\n")
    return k, value, it


def profile_from_csv(path):
    days = []
    with open(path) as handle:
        next(handle)
        for line in handle:
            day, total, count = line.strip().split(",")
            days.append((int(day), float(total), float(count)))
    usable = [(d, total / count) for d, total, count in days if count >= 30 and d <= 21]
    if not usable:
        raise RuntimeError(f"no usable profile days in {path}")
    peak = max(mean for _, mean in usable)
    return usable, peak


def window_for(usable, peak, fraction):
    hot = [day for day, mean in usable if mean >= fraction * peak]
    if not hot:
        raise RuntimeError(f"no days at {fraction} of the peak")
    return min(hot), max(hot), hot


def area_share(usable, start, end):
    total = sum(mean for _, mean in usable)
    inside = sum(mean for day, mean in usable if start <= day <= end)
    return inside / total if total else None


def stage_profile():
    """One uncontrolled run, no multiplier, to define the peak window and f_pre."""
    name = "ps2/profile"
    code = run_job(
        name,
        base_args(0.00041, 7, 8),
        extra_outputs=[("profile_by_day.csv", "--profileOut")],
    )
    if code:
        raise SystemExit(code)
    usable, peak = profile_from_csv(RESULTS / name / "profile_by_day.csv")
    t1, t2, hot = window_for(usable, peak, 0.50)
    u1, u2, hot75 = window_for(usable, peak, 0.75)
    trace = load_trace(RESULTS / name / "trace.tsv")
    delays = []
    for row in trace:
        if row["diag"] > 0:
            delays.append(row["diag"] // 144 - row["day"])
    delays.sort()
    median_delay = delays[len(delays) // 2] if delays else None
    before = sum(mean for day, mean in usable if median_delay is not None and day < median_delay)
    total = sum(mean for _, mean in usable)
    report = {
        "peak_value": peak,
        "days": [{"day": day, "mean": mean} for day, mean in usable],
        "window_50": [t1, t2],
        "days_at_least_50": hot,
        "window_75": [u1, u2],
        "days_at_least_75": hot75,
        "f_peak_50": area_share(usable, t1, t2),
        "f_peak_75": area_share(usable, u1, u2),
        "median_infection_to_diagnosis_days": median_delay,
        "diagnosed_agents": len(delays),
        "f_pre": before / total if total else None,
        "profile_definition": "mean of isInfectious() at the first timestep of each day since infection, days with at least 30 infected agents",
    }
    (RESULTS / "ps2").mkdir(parents=True, exist_ok=True)
    (RESULTS / "ps2" / "profile.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"profile window50={t1}-{t2} f_peak={report['f_peak_50']:.3f} "
        f"window75={u1}-{u2} f_pre={report['f_pre']} median diagnosis delay={median_delay}",
        flush=True,
    )
    return report


def shedding_flags(shape, mode, window):
    flags = ["--sheddingShape", fmt(shape), "--sheddingMode", mode]
    if mode == "peak":
        flags += ["--sheddingPeakWindow", f"{window[0]},{window[1]}"]
    return flags


def solve_shape(label, window, target_k, f_peak):
    cache = RESULTS / "ps2" / f"{label}_solution.json"
    if cache.exists():
        return json.loads(cache.read_text())
    # Only the fraction f_peak of the area varies, so the shape must be smaller than 0.3.
    shape = max(0.02, 0.3 * (f_peak ** 2))
    k0 = 0.0005
    points = []
    best = None
    for attempt in range(6):
        flags = shedding_flags(shape, "peak", window)
        k, measured_r, it = calibrate(f"{label}_a{fmt(shape)}", flags, k0)
        k0 = k
        measured_k, n = pooled_k([RESULTS / f"cal/{label}_a{fmt(shape)}/it{it:02d}_s{seed}" for seed in CAL_SEEDS])
        row = {"attempt": attempt, "shape": shape, "coefficient": k, "early_R": measured_r, "early_k": measured_k, "infectors": n}
        print(f"{label} attempt {attempt} shape={shape:.4g} early k={measured_k} target={target_k:.4g}", flush=True)
        log_event({"experiment": "PS2", "label": label, **row, "target_k": target_k})
        if measured_k is None or (isinstance(measured_k, float) and math.isinf(measured_k)):
            shape = min(1.0, shape * 1.4)
            continue
        points.append((shape, measured_k))
        best = row
        if abs(measured_k / target_k - 1.0) <= 0.15:
            best["reached"] = True
            break
        slope = 1.0
        if len(points) >= 2:
            (a1, k1), (a2, k2) = points[-2], points[-1]
            if a1 != a2 and k1 > 0 and k2 > 0 and k2 != k1:
                slope = math.log(k2 / k1) / math.log(a2 / a1)
                slope = 0.4 if slope <= 0 else min(3.0, max(0.4, slope))
        shape = min(1.0, max(0.02, shape * (target_k / measured_k) ** (1.0 / slope)))
    if best is None:
        raise RuntimeError(f"{label} shape search produced nothing")
    best["reached"] = best.get("reached", False)
    best["target_k"] = target_k
    best["window"] = list(window)
    best["flags"] = shedding_flags(best["shape"], "peak", window)
    cache.write_text(json.dumps(best) + "\n")
    return best


def stage_calibrate():
    profile = json.loads((RESULTS / "ps2" / "profile.json").read_text())
    k_c, r_c, it_c = calibrate("C", shedding_flags(0.3, "constant", None), 0.00048)
    # Distinct from the PS1 cache, which was calibrated to early R 1.39.
    k_a, r_a, _ = calibrate("ps2_A0", [], 0.00038)
    target_k, n = pooled_k([RESULTS / f"cal/C/it{it_c:02d}_s{seed}" for seed in CAL_SEEDS])
    print(f"arm C early k target {target_k} from {n} infectors", flush=True)
    pk50 = solve_shape("Pk50", profile["window_50"], target_k, profile["f_peak_50"])
    pk75 = solve_shape("Pk75", profile["window_75"], target_k, profile["f_peak_75"])
    solved = {
        "C": {"k": k_c, "early_R": r_c, "flags": shedding_flags(0.3, "constant", None)},
        "A0": {"k": k_a, "early_R": r_a, "flags": []},
        "Pk50": pk50,
        "Pk75": pk75,
    }
    (RESULTS / "ps2" / "calibration.json").write_text(json.dumps(solved, indent=1) + "\n")
    return solved


def stage_runs():
    solved = json.loads((RESULTS / "ps2" / "calibration.json").read_text())
    levers = {"none": None, "isolate": LEVERS / "isolate.json", "household": LEVERS / "household.json"}
    for arm, info in solved.items():
        flags = info["flags"]
        for lever, path in levers.items():
            for seed in SEEDS:
                args = base_args(info["coefficient"] if "coefficient" in info else info["k"], seed, WEEKS)
                if path is not None:
                    args = base_args(info["coefficient"] if "coefficient" in info else info["k"], seed, WEEKS, closures=path)
                code = run_job(f"ps2/{arm}_{lever}/rep{seed}", args + flags)
                if code:
                    raise SystemExit(code)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("profile", "all"):
        stage_profile()
    if stage in ("calibrate", "all"):
        stage_calibrate()
    if stage in ("runs", "all"):
        stage_runs()
    print("PS2 stage complete", flush=True)


if __name__ == "__main__":
    main()
