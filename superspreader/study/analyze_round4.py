#!/usr/bin/env python3
"""Round-4 tables that do not need new PanSim runs: R4-0, R4-4, R4-5, R4-6."""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "superspreader"
RESULTS = HERE / "results" / "round2"
sys.path.insert(0, str(HERE))

from analyze import load_daily  # noqa: E402
from analyze_round2 import index_cases, load_arm, mom_k, percentile  # noqa: E402

TYPE_NAME = {
    1: "public",
    2: "home",
    3: "school",
    4: "work",
    5: "restaurant",
    6: "cinema",
    7: "short_shop",
    8: "long_shop",
    9: "church",
    10: "park",
    11: "prison",
    12: "hospital",
    13: "night_work",
    14: "health",
    15: "commuter",
    22: "nursing",
    33: "classroom",
    44: "university",
}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({path.stat().st_size} bytes)", flush=True)


def r40():
    print("R4-0 capacity", flush=True)
    places = json.loads((ROOT / "inputRealExample" / "locations3.json").read_text())["places"]
    by_type = {}
    for index, place in enumerate(places):
        by_type.setdefault(int(place.get("type", -1)), []).append((index, place))
    type_rows = []
    large_rows = []
    named = None
    for loc_type in sorted(by_type):
        items = by_type[loc_type]
        caps = [float(place.get("capacity") or 0) for _i, place in items]
        type_rows.append({
            "type": loc_type,
            "type_name": TYPE_NAME.get(loc_type, ""),
            "locations": len(items),
            "total_capacity": sum(caps),
            "capacity_min": min(caps) if caps else None,
            "capacity_median": percentile(caps, 0.5),
            "capacity_p90": percentile(caps, 0.9),
            "capacity_p99": percentile(caps, 0.99),
            "capacity_max": max(caps) if caps else None,
            "n_above_100": sum(c > 100 for c in caps),
            "n_above_300": sum(c > 300 for c in caps),
            "n_above_1000": sum(c > 1000 for c in caps),
            "n_at_most_100": sum(c <= 100 for c in caps),
            "n_at_most_300": sum(c <= 300 for c in caps),
        })
        ranked = sorted(items, key=lambda item: float(item[1].get("capacity") or 0), reverse=True)[:20]
        for rank, (index, place) in enumerate(ranked, start=1):
            row = {
                "type": loc_type,
                "type_name": TYPE_NAME.get(loc_type, ""),
                "rank": rank,
                "index": index,
                "id": place.get("ID", ""),
                "capacity": place.get("capacity") or 0,
            }
            large_rows.append(row)
            if place.get("ID") == "121284_1803":
                named = row
    if named is None:
        for index, place in enumerate(places):
            if place.get("ID") == "121284_1803":
                named = {"index": index, "id": place["ID"], "type": place.get("type"), "capacity": place.get("capacity")}
                break
    write_csv(OUT / "35_capacity_by_type.csv", type_rows)
    write_csv(OUT / "35_largest_locations.csv", large_rows)
    print("121284_1803", named, flush=True)
    (RESULTS / "r40_named.json").write_text(json.dumps(named, indent=1) + "\n")
    return type_rows


def chain_histograms():
    print("R4-4 chain histograms", flush=True)
    arms = [
        ("old_map_shape0.03", HERE / "results" / "x21" / "x3" / "shape0.03"),
        ("new_map_hospitals_unsplit", HERE / "results" / "v25" / "v4" / "shape0.03"),
        ("hospitals_split", RESULTS / "r34" / "shape0.03"),
        ("w3_hospitals_cinemas_churches", RESULTS / "w3" / "shape0.03"),
        ("w10_repaired_and_split", RESULTS / "w10" / "tail"),
        ("w8_home035", RESULTS / "w8" / "home035"),
    ]
    rows = []
    for label, directory in arms:
        print(f"  {label}", flush=True)
        _off, chains = load_arm(directory)
        counts = Counter(chain["size"] for chain in chains)
        maximum = max(counts) if counts else 1
        for size in range(1, maximum + 1):
            rows.append({"arm": label, "size": size, "chains": counts.get(size, 0)})
        print(f"    chains {sum(counts.values())} longest {maximum}", flush=True)
    write_csv(OUT / "35_chain_size_histograms.csv", rows)


def offspring_histograms():
    print("R4-5 offspring histograms", flush=True)
    arms = [
        ("A0", RESULTS / "r35" / "A0"),
        ("E", RESULTS / "r3cal" / "r35_E" / "round0"),
        ("P0", RESULTS / "r3cal" / "r35_P0" / "round2"),
    ]
    rows = []
    for label, directory in arms:
        parts = [index_cases(path)[1] for path in sorted(directory.glob("rep*/trace.tsv"))]
        pooled = np.concatenate(parts) if parts else np.asarray([], dtype=float)
        counts = Counter(int(value) for value in pooled)
        maximum = max(counts) if counts else 0
        for offspring in range(0, maximum + 1):
            rows.append({"arm": label, "offspring": offspring, "index_cases": counts.get(offspring, 0)})
        print(f"  {label} n {len(pooled)} mean {float(pooled.mean()) if len(pooled) else None} k {mom_k(pooled)} max {maximum}", flush=True)
    write_csv(OUT / "35_r45_offspring_hist.csv", rows)


def volume_replicates():
    print("R4-6 volume replicates", flush=True)
    arms = [
        ("full_mid", RESULTS / "r33" / "vol_full_mid"),
        ("full_low", RESULTS / "r3cal" / "r33_full_low" / "round3"),
        ("full_high", RESULTS / "r3cal" / "r33_full_high" / "round2"),
        ("fill_pinned", RESULTS / "r3cal" / "r33_fill_pinned" / "round3"),
    ]
    rows = []
    for label, directory in arms:
        for path in sorted(directory.glob("rep*/stdout.txt")):
            daily = load_daily(path)
            _shed, offs = index_cases(path.parent / "trace.tsv")
            rows.append({
                "cell": label,
                "rep": path.parent.name,
                "day28": int(daily[27]["INF"]) if len(daily) >= 28 else None,
                "early_k": mom_k(offs),
                "early_R": float(offs.mean()) if len(offs) else None,
                "n_index": int(len(offs)),
            })
        print(f"  {label} reps {sum(1 for row in rows if row['cell'] == label)}", flush=True)
    write_csv(OUT / "35_r46_volume_replicates.csv", rows)


def existing_lever_replicates():
    """Per-replicate finals for the five lever arms already run. New arms are appended after R4-3."""
    print("R4-3 existing lever replicates", flush=True)
    from analyze import load_trace, venue_metrics
    specs = [
        ("venue_s3", RESULTS / "w9" / "venue_s3_{condition}"),
        ("venue_s1", RESULTS / "w9" / "venue_s1_{condition}"),
        ("shape1", RESULTS / "w9" / "shape1_{condition}"),
        ("shape0.5", RESULTS / "r37" / "shape0.5_{condition}"),
        ("shape0.35", RESULTS / "r37" / "share375_{condition}"),
    ]
    rows = []
    for arm, pattern in specs:
        for condition in ("none", "venue_close", "household", "ventilation"):
            directory = Path(str(pattern).format(condition=condition))
            for path in sorted(directory.glob("rep*/stdout.txt")):
                daily = load_daily(path)
                share = venue_metrics(load_trace(path.parent / "trace.tsv"))["household_fraction"]
                rows.append({
                    "arm": arm,
                    "condition": condition,
                    "rep": path.parent.name,
                    "final_size": int(daily[-1]["INF"]) if daily else None,
                    "household_share": share,
                })
        print(f"  {arm} rows {sum(1 for row in rows if row['arm'] == arm)}", flush=True)
    write_csv(OUT / "35_r43_lever_replicates.csv", rows)


def r41_tables():
    """Daily series and the day-28 adjustment for the 16-week A0, E and P0 runs."""
    print("R4-1 tables", flush=True)
    from analyze import daily_summary, prevalence
    daily_rows = []
    summary = []
    for label in ("A0", "E", "P0"):
        for path in sorted((RESULTS / "r41" / label).glob("rep*/stdout.txt")):
            rows = load_daily(path)
            summary_row = daily_summary(rows)
            day28 = int(rows[27]["INF"]) if len(rows) >= 28 else None
            final = int(rows[-1]["INF"]) if rows else None
            summary.append({
                "arm": label,
                "rep": path.parent.name,
                "day28": day28,
                "final_size": final,
                "peak_prevalence": summary_row.get("peak_prevalence"),
                "peak_day": None if summary_row.get("peak_prevalence_day") is None else summary_row["peak_prevalence_day"] + 1,
                "days": summary_row.get("days"),
            })
            for index, row in enumerate(rows):
                daily_rows.append({
                    "arm": label,
                    "rep": path.parent.name,
                    "day": index + 1,
                    "new_infections": int(row.get("NI", 0)),
                    "prevalence": prevalence(row),
                    "cumulative": int(row.get("INF", 0)),
                })
    write_csv(OUT / "35_r41_daily.csv", daily_rows)
    write_csv(OUT / "35_r41_summary.csv", summary)
    # Regression: final = a + b * day28 + arm effects, A0 as the reference.
    arms = ("A0", "E", "P0")
    y = np.asarray([row["final_size"] for row in summary], dtype=float)
    day = np.asarray([row["day28"] for row in summary], dtype=float)
    labels = [row["arm"] for row in summary]
    design = [np.ones(len(y)), day]
    for arm in ("E", "P0"):
        design.append(np.asarray([1.0 if label == arm else 0.0 for label in labels]))
    coef, *_ = np.linalg.lstsq(np.column_stack(design), y, rcond=None)
    anchor = float(np.mean([row["day28"] for row in summary if row["arm"] == "A0"]))
    adjusted = {
        "A0": float(coef[0] + coef[1] * anchor),
        "E": float(coef[0] + coef[1] * anchor + coef[2]),
        "P0": float(coef[0] + coef[1] * anchor + coef[3]),
    }
    raw = {arm: float(np.mean([row["final_size"] for row in summary if row["arm"] == arm])) for arm in arms}
    day_means = {arm: float(np.mean([row["day28"] for row in summary if row["arm"] == arm])) for arm in arms}
    peaks = {arm: float(np.mean([row["peak_prevalence"] for row in summary if row["arm"] == arm])) for arm in arms}
    peak_days = {arm: [row["peak_day"] for row in summary if row["arm"] == arm] for arm in arms}
    # Replicate bootstrap of the raw P0/E ratio and of the adjusted ratio.
    rng = np.random.default_rng(41)
    by_arm = {arm: [row for row in summary if row["arm"] == arm] for arm in arms}

    def fit_ratio(draw):
        y_b = np.asarray([row["final_size"] for row in draw], dtype=float)
        d_b = np.asarray([row["day28"] for row in draw], dtype=float)
        lab_b = [row["arm"] for row in draw]
        columns = [np.ones(len(y_b)), d_b]
        for arm in ("E", "P0"):
            columns.append(np.asarray([1.0 if label == arm else 0.0 for label in lab_b]))
        beta, *_ = np.linalg.lstsq(np.column_stack(columns), y_b, rcond=None)
        ref = float(np.mean([row["day28"] for row in draw if row["arm"] == "A0"]))
        adj = {
            "A0": beta[0] + beta[1] * ref,
            "E": beta[0] + beta[1] * ref + beta[2],
            "P0": beta[0] + beta[1] * ref + beta[3],
        }
        raw_now = {arm: float(np.mean([row["final_size"] for row in draw if row["arm"] == arm])) for arm in arms}
        return raw_now["P0"] / raw_now["E"], adj["P0"] / adj["E"], adj["E"] / adj["A0"], adj["P0"] / adj["A0"]

    samples = []
    n = 12
    for _ in range(2000):
        draw = []
        for arm in arms:
            pick = rng.integers(0, n, size=n)
            draw.extend(by_arm[arm][i] for i in pick)
        samples.append(fit_ratio(draw))
    samples = np.asarray(samples, dtype=float)
    def band(column):
        return [float(np.quantile(samples[:, column], q)) for q in (0.05, 0.95)]
    payload = {
        "anchor_day28": anchor,
        "slope_per_day28_infection": float(coef[1]),
        "raw_mean_final": raw,
        "mean_day28": day_means,
        "mean_peak": peaks,
        "peak_days": peak_days,
        "adjusted_final_at_A0_day28": adjusted,
        "raw_P0_over_E": raw["P0"] / raw["E"],
        "raw_P0_over_E_90": band(0),
        "adjusted_P0_over_E": adjusted["P0"] / adjusted["E"],
        "adjusted_P0_over_E_90": band(1),
        "adjusted_E_over_A0": adjusted["E"] / adjusted["A0"],
        "adjusted_E_over_A0_90": band(2),
        "adjusted_P0_over_A0": adjusted["P0"] / adjusted["A0"],
        "adjusted_P0_over_A0_90": band(3),
        "raw_E_over_A0": raw["E"] / raw["A0"],
        "raw_P0_over_A0": raw["P0"] / raw["A0"],
    }
    (RESULTS / "r41_adjustment.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1), flush=True)


def main():
    which = sys.argv[1:] or ["r40", "r44", "r45", "r46", "r43existing"]
    if "r41" in which:
        r41_tables()
    if "r40" in which:
        r40()
    if "r44" in which:
        chain_histograms()
    if "r45" in which:
        offspring_histograms()
    if "r46" in which:
        volume_replicates()
    if "r43existing" in which:
        existing_lever_replicates()
    print("no-run tables done", flush=True)


if __name__ == "__main__":
    main()
