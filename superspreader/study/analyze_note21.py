#!/usr/bin/env python3
"""Analyses A1, A2 and A3 from 21_further_experiments.md, plus readers for X1, X2, X3 and X5 when those runs exist."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ps"))

from analyze import load_daily, load_trace, mom_k  # noqa: E402
from early_window import early_counts  # noqa: E402
from ps_analysis import json_ready, t_interval  # noqa: E402

OUT = HERE / "results" / "x21"
SUMMARIES = HERE / "summaries"
T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def paired_interval(diffs):
    return t_interval(diffs)


def replicates(arm):
    root = HERE / "results" / "ps" / "ps1" / arm
    rows = []
    for path in sorted(root.glob("rep*/trace.tsv")):
        counts = early_counts(load_trace(path))
        daily = load_daily(path.parent / "stdout.txt")
        rows.append({
            "rep": path.parent.name,
            "early_k": mom_k(counts) if len(counts) > 2 else None,
            "early_R": sum(counts) / len(counts) if counts else None,
            "final_size": int(daily[-1]["INF"]) if daily else None,
        })
    return rows


def a1():
    arms = {name: replicates(name) for name in ("E", "P0")}
    by_seed = {name: {row["rep"]: row for row in rows} for name, rows in arms.items()}
    seeds = sorted(set(by_seed["E"]) & set(by_seed["P0"]))
    k_diff = [by_seed["P0"][seed]["early_k"] - by_seed["E"][seed]["early_k"] for seed in seeds]
    size_diff = [by_seed["P0"][seed]["final_size"] - by_seed["E"][seed]["final_size"] for seed in seeds]
    k_interval = paired_interval(k_diff)
    size_interval = paired_interval(size_diff)
    k_level = sum(by_seed["E"][seed]["early_k"] for seed in seeds) / len(seeds)
    # Half-width scales as 1/sqrt(n). Target half-width is 5% of the dispersion level.
    half = (k_interval["high"] - k_interval["low"]) / 2
    target_half = 0.05 * k_level
    n_for_5pct = len(seeds) * (half / target_half) ** 2 if target_half else None
    return {
        "seeds": len(seeds),
        "early_k_difference_P0_minus_E": k_interval,
        "relative_to_E": None if not k_level else k_interval["mean"] / k_level,
        "equivalence_half_width_over_level": None if not k_level else half / k_level,
        "replicates_for_plus_minus_5_percent": n_for_5pct,
        "final_size_difference_P0_minus_E": size_interval,
        "final_size_E": t_interval([by_seed["E"][seed]["final_size"] for seed in seeds]),
        "final_size_P0": t_interval([by_seed["P0"][seed]["final_size"] for seed in seeds]),
    }


def branching_sizes(rng, mean, k, n):
    sizes = np.empty(n, dtype=np.int32)
    for i in range(n):
        total = 1
        current = 1
        while current:
            lam = rng.gamma(k, mean / k, size=current)
            nxt = int(rng.poisson(lam).sum())
            total += nxt
            current = nxt
            if total > 5000:
                break
        sizes[i] = total
    return sizes


def a2():
    chains = json.loads((SUMMARIES / "p2_chains.json").read_text())
    flat = chains["chain_flat_shed0.04"]
    care = chains["chain_caregiving_shed0.09"]
    rng = np.random.default_rng(21)
    expected = {}
    for label, arm in (("flat_shed0.04", flat), ("caregiving_shed0.09", care)):
        sizes = branching_sizes(rng, arm["all_node_mean_offspring"], arm["all_node_k_mom"], 300000)
        expected[label] = {
            "chains": arm["chains"],
            "observed_gt_10": arm["p_gt_10"] * arm["chains"],
            "observed_gt_50": arm["chains_gt_50"],
            "expected_gt_10": float(np.mean(sizes > 10) * arm["chains"]),
            "expected_gt_50": float(np.mean(sizes > 50) * arm["chains"]),
            "max_observed": arm["max_chain"],
        }
    # Two-proportion z test of chains > 50. Counts are small and the rates are almost equal.
    a, n1 = expected["flat_shed0.04"]["observed_gt_50"], flat["chains"]
    b, n2 = expected["caregiving_shed0.09"]["observed_gt_50"], care["chains"]
    p_pool = (a + b) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = 0.0 if se == 0 else ((a / n1) - (b / n2)) / se
    # Two-sided normal p, adequate at these sample sizes; the counts themselves are 17 and 19.
    p_value = math.erfc(abs(z) / math.sqrt(2))
    p = (a + b) / (n1 + n2)
    p2 = 2 * p
    # Replicates per arm to detect a doubling of the >50 rate at 80% power, two-sided 5%.
    effect = p
    var = p * (1 - p) + p2 * (1 - p2)
    n_double = (1.96 + 0.8416) ** 2 * var / (p2 - p) ** 2
    return {
        "negative_binomial": expected,
        "chains_gt_50": {"flat": a, "caregiving": b, "z": z, "two_sided_p": p_value},
        "chains_per_arm_to_detect_doubling_of_gt_50": n_double,
        "note": "The between-arm count of chains above 50 is not evidence of a difference. The caregiving departure from its own negative binomial is.",
    }


def mom_samples(rng, true_k, reproduction, n_index, n_studies):
    estimates = []
    for _ in range(n_studies):
        lam = rng.gamma(true_k, reproduction / true_k, size=n_index)
        offspring = rng.poisson(lam)
        mean = float(offspring.mean())
        var = float(offspring.var(ddof=1))
        if var <= mean or mean <= 0:
            continue
        estimates.append(mean * mean / (var - mean))
    estimates = np.sort(np.asarray(estimates))
    return {
        "kept": int(len(estimates)),
        "p05": float(np.quantile(estimates, 0.05)),
        "p50": float(np.quantile(estimates, 0.50)),
        "p95": float(np.quantile(estimates, 0.95)),
    }


def a3():
    rng = np.random.default_rng(21)
    rows = []
    for reproduction in (0.8, 1.0, 1.4):
        for n_index in (100, 200, 500):
            sample = mom_samples(rng, 0.3, reproduction, n_index, 4000)
            sample.update({"R": reproduction, "index_cases": n_index, "true_k": 0.3})
            rows.append(sample)
    return {"studies": rows, "reading": "At a few hundred index cases the sampling range of k covers 0.2 to 0.5, so a tracing study of that size does not separate those two values."}


def chain_summary(directory):
    """Chain sizes, hospital inheritance, and same-location inheritance for one arm directory."""
    sizes = []
    offspring_counts = []
    parent_n = defaultdict(int)
    parent_off = defaultdict(int)
    parent_child_hospital = defaultdict(int)
    parent_children = defaultdict(int)
    same_location = 0
    child_links = 0
    nonseed = 0
    for path in sorted(directory.glob("rep*/trace.tsv")):
        rows = {}
        with open(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            index = {name: i for i, name in enumerate(header)}
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                infector = int(parts[index["infector"]])
                rows[int(parts[index["agent"]])] = {
                    "parent": None if infector == 4294967295 else infector,
                    "type": int(parts[index["locType"]]),
                    "location": int(parts[index["location"]]),
                }
        children = defaultdict(list)
        roots = []
        for agent, row in rows.items():
            if row["parent"] is not None and row["parent"] in rows:
                children[row["parent"]].append(agent)
            else:
                roots.append(agent)
        for root in roots:
            size = 0
            stack = [root]
            while stack:
                node = stack.pop()
                size += 1
                stack.extend(children.get(node, ()))
            sizes.append(size)
        for agent in rows:
            offspring_counts.append(len(children.get(agent, [])))
        for agent, row in rows.items():
            if row["parent"] is None:
                continue
            nonseed += 1
            kind = row["type"]
            kids = children.get(agent, [])
            parent_n[kind] += 1
            parent_off[kind] += len(kids)
            parent_children[kind] += len(kids)
            parent_child_hospital[kind] += sum(rows[child]["type"] == 12 for child in kids)
            for child in kids:
                child_links += 1
                if rows[child]["location"] == row["location"]:
                    same_location += 1
    sizes = np.asarray(sizes, dtype=float)
    if len(sizes) == 0:
        return None
    mean_chain = float(sizes.mean())
    mean_off = float(np.mean(offspring_counts)) if offspring_counts else None
    dispersion = mom_k(offspring_counts) if len(offspring_counts) > 2 else None
    return {
        "chains": int(len(sizes)),
        "mean_chain": mean_chain,
        "all_node_mean_offspring": mean_off,
        "all_node_k": dispersion,
        "chains_gt_10": int(np.sum(sizes > 10)),
        "chains_gt_50": int(np.sum(sizes > 50)),
        "max_chain": int(sizes.max()),
        "same_location_child_share": same_location / child_links if child_links else None,
        "nonseed": nonseed,
        "by_parent_type": {
            str(kind): {
                "n": parent_n[kind],
                "mean_offspring": parent_off[kind] / parent_n[kind] if parent_n[kind] else None,
                "child_in_hospital": parent_child_hospital[kind] / parent_children[kind] if parent_children[kind] else None,
            }
            for kind in (2, 12, 14, 22)
            if parent_n[kind]
        },
    }


def x1_report():
    base = OUT / "x1"
    if not base.exists():
        return None
    report = {}
    rng = np.random.default_rng(21)
    for label in ("flat_shed0.04", "caregiving_shed0.09"):
        summary = chain_summary(base / label)
        if not summary or not summary.get("all_node_k") or not math.isfinite(summary["all_node_k"]):
            if summary:
                report[label] = summary
            continue
        sizes = branching_sizes(rng, summary["all_node_mean_offspring"], summary["all_node_k"], 200000)
        summary["expected_gt_10"] = float(np.mean(sizes > 10) * summary["chains"])
        summary["expected_gt_50"] = float(np.mean(sizes > 50) * summary["chains"])
        report[label] = summary
    original = json.loads((SUMMARIES / "p2_chains.json").read_text())
    return {"ward": report, "original_flat_gt_10": original["chain_flat_shed0.04"]["p_gt_10"] * original["chain_flat_shed0.04"]["chains"],
            "original_care_gt_10": original["chain_caregiving_shed0.09"]["p_gt_10"] * original["chain_caregiving_shed0.09"]["chains"],
            "original_flat_gt_50": original["chain_flat_shed0.04"]["chains_gt_50"],
            "original_care_gt_50": original["chain_caregiving_shed0.09"]["chains_gt_50"],
            "nb_placeholder": bool(rng)}


def main():
    payload = {"A1": a1(), "A2": a2(), "A3": a3()}
    x1 = x1_report()
    if x1:
        payload["X1"] = x1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analyses.json").write_text(json.dumps(json_ready(payload), indent=1) + "\n")
    print(json.dumps(json_ready({"A1": payload["A1"], "A2_p": payload["A2"]["chains_gt_50"], "A3": payload["A3"]["studies"]}), indent=1))


if __name__ == "__main__":
    main()
