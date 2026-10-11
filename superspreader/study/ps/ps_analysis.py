#!/usr/bin/env python3
"""Analysis helpers for PS1 / PS2: trace loading, early-window k, cohort windows, bootstraps."""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from ps_common import STEPS_PER_DAY

UNKNOWN = 0xFFFFFFFF
T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def load_trace(path):
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: pos for pos, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            infector = int(parts[index["infector"]])
            ts = int(parts[index["timestamp"]])
            row = {
                "agent": int(parts[index["agent"]]),
                "infector": None if infector == UNKNOWN else infector,
                "locType": int(parts[index["locType"]]),
                "day": ts // STEPS_PER_DAY,
                "ts": ts,
                "shed": float(parts[index["shed"]]),
                "diag": int(parts[index["diagnosed"]]),
                "hosp": int(parts[index["hospitalized"]]),
                "count": int(parts[index["infectedCount"]]),
            }
            if "infectorShed" in index:
                its = int(parts[index["infectorTimestamp"]])
                so = int(parts[index["symptomOnset"]])
                row["infector_shed"] = float(parts[index["infectorShed"]])
                row["infector_day"] = None if its == UNKNOWN else its // STEPS_PER_DAY
                row["symptom_ts"] = None if so == UNKNOWN else so
                row["q_from"] = int(parts[index["quarantinedFrom"]])
                row["q_until"] = int(parts[index["quarantinedUntil"]])
            rows.append(row)
    return rows


def offspring_by_agent(trace, horizon=14):
    """Return {agent: list of child infection days}, using known infectors only."""
    children = defaultdict(list)
    for row in trace:
        if row["infector"] is not None:
            children[row["infector"]].append(row["day"])
    return children


def window_counts(trace, first, last, horizon=14, children=None):
    """Offspring counts (within horizon days) of index cases infected on days first..last with a known infector.

    Also returns the infected agents' ids and trait values for the same index cases.
    """
    children = children if children is not None else offspring_by_agent(trace)
    out = []
    for row in trace:
        if row["infector"] is None:
            continue
        if first <= row["day"] <= last:
            day = row["day"]
            n = sum(1 for d in children.get(row["agent"], ()) if d <= day + horizon)
            out.append((row["agent"], n, row["shed"], row["diag"]))
    return out


def mom_k(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if mean <= 0:
        return None
    if var <= mean:
        return math.inf
    return mean * mean / (var - mean)


def moments_k(n, s, ss):
    """k from pooled count, sum, sum of squares (sample variance, ddof=1)."""
    if n < 2 or s <= 0:
        return None
    mean = s / n
    var = (ss - n * mean * mean) / (n - 1)
    if var <= mean:
        return math.inf
    return mean * mean / (var - mean)


def hist_stats(hist):
    n = sum(hist.values())
    s = sum(v * c for v, c in hist.items())
    ss = sum(v * v * c for v, c in hist.items())
    return n, s, ss


def top_share(hist, fraction=0.2):
    n = sum(hist.values())
    total = sum(v * c for v, c in hist.items())
    if n == 0 or total == 0:
        return None
    need = fraction * n
    taken = 0.0
    acc = 0.0
    for v in sorted(hist, reverse=True):
        c = hist[v]
        use = min(c, need - taken)
        if use <= 0:
            break
        acc += use * v
        taken += use
    return acc / total


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cluster_bootstrap(hists, statistic, draws=2000, seed=11):
    """Resample replicates; statistic(pooled_hist) -> number or None. Returns (low, high, finite_fraction)."""
    rng = random.Random(seed)
    m = len(hists)
    values = []
    for _ in range(draws):
        pooled = Counter()
        for _i in range(m):
            pooled.update(hists[rng.randrange(m)])
        v = statistic(pooled)
        if v is not None:
            values.append(v)
    finite = [v for v in values if not math.isinf(v)]
    values.sort()
    if not values:
        return None
    return {
        "low": percentile(values, 0.025),
        "high": percentile(values, 0.975),
        "finite_fraction": len(finite) / len(values),
    }


def infector_bootstrap(counts, draws=1000, seed=7):
    n = len(counts)
    if n < 30:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [counts[rng.randrange(n)] for _ in range(n)]
        k = mom_k(sample)
        if k is not None:
            values.append(k)
    values.sort()
    return {"low": percentile(values, 0.025), "high": percentile(values, 0.975),
            "finite_fraction": sum(1 for v in values if not math.isinf(v)) / len(values)}


def t_interval(values):
    values = [v for v in values if v is not None and not math.isinf(v) and not math.isnan(v)]
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "low": None, "high": None, "n": n}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    t = T975.get(n - 1, 1.96)
    half = t * sd / math.sqrt(n)
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": n, "sd": sd}


def paired_t_interval(a, b):
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    return t_interval(diffs)


def replicate_dirs(cell_dir):
    return sorted(p.parent for p in Path(cell_dir).glob("rep*/trace.tsv"))


def early_window_summary(cell_dir, first=1, last=10, draws=2000):
    """Pooled and per-replicate early-window statistics with both interval types."""
    hists, per_rep, pooled = [], [], []
    for run in replicate_dirs(cell_dir):
        trace = load_trace(run / "trace.tsv")
        counts = [c[1] for c in window_counts(trace, first, last)]
        hist = Counter(counts)
        hists.append(hist)
        pooled.extend(counts)
        n, s, ss = hist_stats(hist)
        per_rep.append({"run": run.name, "n": n, "mean": s / n if n else None, "k": moments_k(n, s, ss),
                        "zero": hist.get(0, 0) / n if n else None, "top20": top_share(hist)})
    pooled_hist = Counter()
    for h in hists:
        pooled_hist.update(h)
    n, s, ss = hist_stats(pooled_hist)
    result = {
        "replicates": len(hists),
        "infectors": n,
        "mean": s / n if n else None,
        "k": moments_k(n, s, ss),
        "zero": pooled_hist.get(0, 0) / n if n else None,
        "top20": top_share(pooled_hist),
        "k_cluster_ci": cluster_bootstrap(hists, lambda h: moments_k(*hist_stats(h)), draws),
        "mean_cluster_ci": cluster_bootstrap(hists, lambda h: (lambda t: t[1] / t[0] if t[0] else None)(hist_stats(h)), draws),
        "zero_cluster_ci": cluster_bootstrap(hists, lambda h: h.get(0, 0) / sum(h.values()) if h else None, draws),
        "top20_cluster_ci": cluster_bootstrap(hists, top_share, draws),
        "k_infector_ci": infector_bootstrap(pooled),
        "per_replicate": per_rep,
        "k_replicate_t": t_interval([r["k"] for r in per_rep]),
        "mean_replicate_t": t_interval([r["mean"] for r in per_rep]),
    }
    return result


def json_ready(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "infinity"
        if math.isnan(value):
            return None
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value
