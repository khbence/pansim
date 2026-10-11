#!/usr/bin/env python3
"""Summaries for PS1. Reads whatever replicates are already on disk."""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import load_daily, hospital, prevalence  # noqa: E402
from ps_analysis import (  # noqa: E402
    cluster_bootstrap,
    hist_stats,
    json_ready,
    load_trace,
    moments_k,
    replicate_dirs,
    t_interval,
    top_share,
    window_counts,
)
from ps_common import RESULTS  # noqa: E402

WINDOWS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50))
KB = 5.25
# shape that enters the composition rule: a + rho for a gamma trait, None if not gamma
COMPOSITION = {
    "E": 0.30, "P0": 0.30, "P25": 0.55, "P50": 0.80, "P100": 1.30, "P25k": None,
}
EXPECTED_TRAIT = {"E": 1.0, "P0": 1.0, "P25": 1.83, "P50": 2.67, "P100": 4.33}


def composition_k(shape):
    return 1.0 / (1.0 / KB + 1.0 / shape + 1.0 / (KB * shape))


def arm_dirs():
    root = RESULTS / "ps1"
    return sorted(p for p in root.iterdir() if p.is_dir() and replicate_dirs(p))


def per_run(run):
    trace = load_trace(run / "trace.tsv")
    children_ready = None
    windows = {}
    for first, last in WINDOWS:
        items = window_counts(trace, first, last)
        hist = Counter(item[1] for item in items)
        n, s, ss = hist_stats(hist)
        traits = [item[2] for item in items]
        windows[(first, last)] = {
            "hist": hist,
            "k": moments_k(n, s, ss),
            "mean": s / n if n else None,
            "trait": sum(traits) / len(traits) if traits else None,
            "n": n,
        }
    rows = load_daily(run / "stdout.txt")
    prev = [prevalence(row) for row in rows]
    return {
        "windows": windows,
        "final_size": int(rows[-1].get("INF", 0)) if rows else None,
        "peak_prevalence": max(prev) if prev else None,
        "peak_hospital": max(hospital(row) for row in rows) if rows else None,
        "trace": trace,
    }


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = 0.5 * (i + j) + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def summarise(arm_dir):
    runs = [per_run(run) for run in replicate_dirs(arm_dir)]
    early = [run["windows"][(1, 10)] for run in runs]
    hists = [item["hist"] for item in early]
    pooled = Counter()
    for hist in hists:
        pooled.update(hist)
    n, s, ss = hist_stats(pooled)
    days, traits = [], []
    for run in runs:
        for row in run["trace"]:
            if row["infector"] is None:
                continue
            days.append(row["day"])
            traits.append(row["shed"])
    cohorts = []
    for window in WINDOWS:
        ks = [run["windows"][window]["k"] for run in runs]
        traits_w = [run["windows"][window]["trait"] for run in runs]
        hists_w = [run["windows"][window]["hist"] for run in runs]
        pooled_w = Counter()
        for hist in hists_w:
            pooled_w.update(hist)
        nw, sw, ssw = hist_stats(pooled_w)
        cohorts.append({
            "window": list(window),
            "infectors": nw,
            "k": moments_k(nw, sw, ssw),
            "k_cluster_ci": cluster_bootstrap(hists_w, lambda h: moments_k(*hist_stats(h))),
            "k_replicate_t": t_interval(ks),
            "mean_trait": sum(v for v in traits_w if v is not None) / len([v for v in traits_w if v is not None]) if any(v is not None for v in traits_w) else None,
            "trait_replicate_t": t_interval(traits_w),
        })
    return {
        "arm": arm_dir.name,
        "replicates": len(runs),
        "infectors": n,
        "early_R": s / n if n else None,
        "early_k": moments_k(n, s, ss),
        "early_k_cluster_ci": cluster_bootstrap(hists, lambda h: moments_k(*hist_stats(h))),
        "early_k_replicate_t": t_interval([item["k"] for item in early]),
        "zero": pooled.get(0, 0) / n if n else None,
        "zero_cluster_ci": cluster_bootstrap(hists, lambda h: h.get(0, 0) / sum(h.values()) if h else None),
        "top20": top_share(pooled),
        "top20_cluster_ci": cluster_bootstrap(hists, top_share),
        "final_size": t_interval([run["final_size"] for run in runs]),
        "peak_prevalence": t_interval([run["peak_prevalence"] for run in runs]),
        "peak_hospital": t_interval([run["peak_hospital"] for run in runs]),
        "trait_day_spearman": spearman(traits, days),
        "cohorts": cohorts,
    }


def judge(summaries):
    by = {item["arm"]: item for item in summaries}
    out = {}
    if "E" in by and "P0" in by:
        ratio_k = by["P0"]["early_k"] / by["E"]["early_k"]
        ratio_size = by["P0"]["final_size"]["mean"] / by["E"]["final_size"]["mean"]
        out["H1"] = {
            "k_ratio": ratio_k, "size_ratio": ratio_size,
            "pass": 0.75 <= ratio_k <= 1.33 and 0.90 <= ratio_size <= 1.10,
        }
    h2 = []
    for arm, shape in COMPOSITION.items():
        if arm not in by or shape is None or by[arm]["early_k"] is None:
            continue
        base = composition_k(shape)
        band = (1.2 * base, 1.5 * base)
        measured = by[arm]["early_k"]
        h2.append({"arm": arm, "composition_k": base, "band": band, "measured": measured, "inside": band[0] <= measured <= band[1]})
    if h2:
        out["H2"] = {"rows": h2, "pass": all(row["inside"] for row in h2)}
    if "P0" in by and "P25" in by and "P50" in by and "P100" in by:
        sizes = [by[arm]["final_size"]["mean"] for arm in ("P0", "P25", "P50", "P100")]
        out["H4"] = {"final_size": dict(zip(("P0", "P25", "P50", "P100"), sizes)), "pass": sizes[0] > sizes[1] > sizes[2] > sizes[3]}
    if "P100" in by and "P100ln" in by and by["P100"]["early_k"]:
        ratio = by["P100ln"]["early_k"] / by["P100"]["early_k"]
        out["H6"] = {"ratio": ratio, "pass": ratio < 0.7}
    return out


def main():
    summaries = [summarise(arm) for arm in arm_dirs()]
    payload = {"arms": summaries, "predictions": judge(summaries)}
    out = RESULTS / "ps1" / "summary.json"
    out.write_text(json.dumps(json_ready(payload), indent=1) + "\n")
    print(f"wrote {out}")
    for item in summaries:
        print(f"{item['arm']:8} reps {item['replicates']:2} n {item['infectors']:6} R {item['early_R']:.3f} k {item['early_k']:.3f} size {item['final_size']['mean']:.0f}")
    print(json.dumps(json_ready(payload["predictions"]), indent=1))


if __name__ == "__main__":
    main()
