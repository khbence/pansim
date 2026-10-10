#!/usr/bin/env python3
"""Tables for phase 2. Writes summaries/p2_<group>.json and prints compact tables."""

import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import (  # noqa: E402
    chain_sizes_fixed,
    hospital,
    json_ready,
    load_daily,
    mean_var,
    mom_k,
    prevalence,
    share_for_fraction_of_mass,
    top_share,
)
from early_window import bootstrap_k, early_counts, stats  # noqa: E402

RESULTS = HERE / "results"
OUT = HERE / "summaries"
STEPS = 144


def load_trace_full(path):
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: pos for pos, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            infector = int(parts[index["infector"]])
            rows.append(
                {
                    "agent": int(parts[index["agent"]]),
                    "infector": None if infector == 4294967295 else infector,
                    "location": int(parts[index["location"]]),
                    "locType": int(parts[index["locType"]]),
                    "day": int(parts[index["timestamp"]]) // STEPS,
                    "shed": float(parts[index["shed"]]),
                    "diagnosed": int(parts[index["diagnosed"]]) > 0,
                    "occupancy": int(parts[index["occupancy"]]) if "occupancy" in index else 0,
                    "present": float(parts[index["infectiousPresent"]]) if "infectiousPresent" in index else 0.0,
                }
            )
    return rows


def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def complete_counts(trace):
    nodes = set()
    children = defaultdict(int)
    for row in trace:
        nodes.add(row["agent"])
        if row["infector"] is not None:
            nodes.add(row["infector"])
            children[row["infector"]] += 1
    return [children.get(a, 0) for a in nodes]


def occupancy_profile(trace, first=1, last=14):
    occ = [r["occupancy"] for r in trace if r["infector"] is not None and first <= r["day"] <= last and r["occupancy"] > 0]
    if not occ:
        return {}
    occ.sort()
    n = len(occ)
    return {
        "median_occupancy": occ[n // 2],
        "p90_occupancy": occ[int(0.9 * n)],
        "share_above_50": sum(o > 50 for o in occ) / n,
        "share_above_200": sum(o > 200 for o in occ) / n,
    }


def venue_top10(trace, first=1, last=14):
    counts = defaultdict(int)
    for r in trace:
        if r["infector"] is not None and first <= r["day"] <= last:
            counts[r["location"]] += 1
    return top_share(list(counts.values()), 0.10) if counts else None


def run_dirs(cell):
    return sorted(p.parent for p in cell.glob("rep*/trace.tsv"))


def cell_open(cell, calibrated_k=None):
    pooled = []
    occ = []
    top10 = []
    finals = []
    peaks = []
    for run in run_dirs(cell):
        trace = load_trace_full(run / "trace.tsv")
        pooled.extend(early_counts(trace))
        occ.append(occupancy_profile(trace))
        top10.append(venue_top10(trace))
        daily = load_daily(run / "stdout.txt")
        finals.append(len(trace))
        peaks.append(max(prevalence(r) for r in daily))
    out = stats(pooled)
    out["k_ci"] = bootstrap_k(pooled)
    out["replicates"] = len(finals)
    out["mean_infections"] = sum(finals) / len(finals) if finals else None
    out["mean_peak_prevalence"] = sum(peaks) / len(peaks) if peaks else None
    out["top10_venues_early"] = sum(x for x in top10 if x is not None) / max(1, sum(x is not None for x in top10))
    keys = ("median_occupancy", "p90_occupancy", "share_above_50", "share_above_200")
    out["occupancy"] = {
        key: sum(o.get(key, 0) for o in occ) / len(occ) for key in keys
    } if occ else {}
    kfile = RESULTS / "p2cal" / cell.name / "k.json"
    if kfile.exists():
        out["calibrated"] = json.loads(kfile.read_text())
    return out


def group_open(prefix, name):
    payload = {}
    for cell in sorted((RESULTS / "p2").glob(prefix + "*")):
        print(name, cell.name, flush=True)
        payload[cell.name] = cell_open(cell)
    return payload


def observation_models_early(trace, first=1, last=10, horizon=14):
    children = defaultdict(list)
    for row in trace:
        if row["infector"] is not None:
            children[row["infector"]].append(row)
    by_agent = {r["agent"]: r for r in trace}
    index = [r for r in trace if r["infector"] is not None and first <= r["day"] <= last]
    true = [sum(1 for c in children.get(r["agent"], ()) if c["day"] <= r["day"] + horizon) for r in index]
    forward = []
    backward = []
    pure_backward = []
    for r in index:
        kids = [c for c in children.get(r["agent"], ()) if c["day"] <= r["day"] + horizon]
        dk = sum(1 for c in kids if c["diagnosed"])
        if r["diagnosed"]:
            forward.append(dk)
            backward.append(len(kids))
        elif dk > 0:
            backward.append(dk)
        if dk > 0:
            pure_backward.append(len(kids))
    venue_n = defaultdict(int)
    venue_d = defaultdict(int)
    for r in trace:
        if r["infector"] is None:
            continue
        venue_n[r["location"]] += 1
        venue_d[r["location"]] += 1 if r["diagnosed"] else 0
    detected = {v for v, n in venue_n.items() if n >= 3 and venue_d[v] >= 1}
    cluster = [
        sum(1 for c in children.get(r["agent"], ()) if c["day"] <= r["day"] + horizon and c["location"] in detected)
        for r in index
        if r["location"] in detected
    ]
    return {
        "true": stats(true),
        "symptom_forward": stats(forward),
        "mixed_backward": stats(backward),
        "pure_backward_true_offspring": stats(pure_backward),
        "cluster": stats(cluster),
    }


def group_audit():
    payload = {}
    for cell in sorted((RESULTS / "p2").glob("audit_*")):
        pooled = defaultdict(list)
        results = []
        for run in run_dirs(cell):
            print("audit", cell.name, run.name, flush=True)
            results.append(observation_models_early(load_trace_full(run / "trace.tsv")))
        # pool by concatenating per-run summaries weighted by n via recomputation
        merged = {}
        for model in results[0]:
            ns = [r[model].get("n", 0) for r in results]
            total = sum(ns)
            merged[model] = {
                "n": total,
                "mean": sum(r[model].get("mean", 0) * r[model].get("n", 0) for r in results) / total if total else None,
                "k_mom_per_rep": [r[model].get("k_mom") for r in results],
                "zero": sum(r[model].get("zero", 0) * r[model].get("n", 0) for r in results) / total if total else None,
            }
        payload[cell.name] = merged
    return payload


def group_lever():
    payload = {}
    for cell in sorted((RESULTS / "p2").glob("lever_*")):
        finals, peaks, hosp, q, r_pre, r_post, d21, d35 = [], [], [], [], [], [], [], []
        for run in run_dirs(cell):
            trace = load_trace_full(run / "trace.tsv")
            daily = load_daily(run / "stdout.txt")
            finals.append(len(trace))
            peaks.append(max(prevalence(r) for r in daily))
            hosp.append(max(hospital(r) for r in daily))
            q.append(sum(int(r.get("Q", 0)) for r in daily))
            pre = early_counts(trace, 1, 10, 14)
            post = early_counts(trace, 24, 33, 14)
            r_pre.append(sum(pre) / len(pre) if pre else None)
            r_post.append(sum(post) / len(post) if post else None)
            d21.append(prevalence(daily[21]))
            d35.append(prevalence(daily[35]))

        def avg(x):
            v = [i for i in x if i is not None]
            return sum(v) / len(v) if v else None

        payload[cell.name] = {
            "replicates": len(finals),
            "infections": avg(finals),
            "infections_per_rep": finals,
            "peak_hospital_per_rep": hosp,
            "peak_prevalence": avg(peaks),
            "peak_hospital": avg(hosp),
            "quarantine_person_days": avg(q),
            "early_R_days1_10": avg(r_pre),
            "R_days24_33": avg(r_post),
            "prevalence_day21": avg(d21),
            "prevalence_day35": avg(d35),
        }
    return payload


def group_chains():
    payload = {}
    for cell in sorted((RESULTS / "p2").glob("chain_*")):
        sizes = []
        counts = []
        gen1 = []
        reps = 0
        for run in run_dirs(cell):
            trace = load_trace_full(run / "trace.tsv")
            chain = chain_sizes_fixed(trace)
            sizes.extend(chain["sizes"])
            counts.extend(complete_counts(trace))
            seeds = [r["agent"] for r in trace if r["infector"] is None]
            kids = defaultdict(int)
            for r in trace:
                if r["infector"] is not None:
                    kids[r["infector"]] += 1
            gen1.append(sum(kids.get(s, 0) for s in seeds) / max(1, len(seeds)))
            reps += 1
        mean, var = mean_var(counts)
        k = mom_k(counts)
        n = len(sizes)
        payload[cell.name] = {
            "replicates": reps,
            "chains": n,
            "nodes": len(counts),
            "all_node_mean_offspring": mean,
            "all_node_k_mom": "infinity" if (k is not None and math.isinf(k)) else k,
            "zero_share": sum(c == 0 for c in counts) / len(counts),
            "zero_share_ci": wilson(sum(c == 0 for c in counts), len(counts)),
            "generation1_R": sum(gen1) / len(gen1),
            "generation1_R_per_rep": gen1,
            "mean_chain": sum(sizes) / n,
            "max_chain": max(sizes),
            "p_gt_5": sum(s > 5 for s in sizes) / n,
            "p_gt_10": sum(s > 10 for s in sizes) / n,
            "p_gt_10_ci": wilson(sum(s > 10 for s in sizes), n),
            "p_gt_20": sum(s > 20 for s in sizes) / n,
            "p_gt_50": sum(s > 50 for s in sizes) / n,
            "p_gt_50_ci": wilson(sum(s > 50 for s in sizes), n),
            "chains_gt_50": sum(s > 50 for s in sizes),
        }
        kfile = RESULTS / "p2cal" / cell.name / "k.json"
        if kfile.exists():
            payload[cell.name]["calibrated"] = json.loads(kfile.read_text())
    return payload


START = date(2020, 9, 23)


def group_waves():
    payload = {}
    cells = sorted((RESULTS / "p2").glob("waves_*"))
    cells.append(RESULTS / "waves" / "baseline")
    for cell in cells:
        name = cell.name if cell.parent.name == "p2" else "waves_baseline_orig"
        runs = []
        pooled = []
        for run in run_dirs(cell):
            print("waves", name, run.name, flush=True)
            daily = load_daily(run / "stdout.txt")
            hosp = [hospital(r) for r in daily]
            prev = [prevalence(r) for r in daily]
            split = 140
            h1 = max(range(split), key=lambda i: hosp[i])
            h2 = max(range(split, len(hosp)), key=lambda i: hosp[i])
            trace = load_trace_full(run / "trace.tsv")
            counts = early_counts(trace)
            pooled.extend(counts)
            runs.append(
                {
                    "run": run.name,
                    "h1": hosp[h1],
                    "h1_date": (START + timedelta(days=h1)).isoformat(),
                    "h2": hosp[h2],
                    "h2_date": (START + timedelta(days=h2)).isoformat(),
                    "p1": max(prev[:split]),
                    "p2": max(prev[split:]),
                    "area_105": sum(hosp[:105]),
                    "infections": len(trace),
                }
            )
        entry = {"runs": runs, "early": stats(pooled)}
        entry["early"]["k_ci"] = bootstrap_k(pooled)
        kfile = RESULTS / "p2cal" / cell.name / "k.json"
        if kfile.exists():
            entry["calibrated"] = json.loads(kfile.read_text())
        payload[name] = entry
    return payload


GROUPS = {
    "kernel": lambda: group_open("kernel_", "kernel"),
    "grid": lambda: group_open("grid_", "grid"),
    "audit": group_audit,
    "lever": group_lever,
    "chains": group_chains,
    "waves": group_waves,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or list(GROUPS):
        payload = GROUPS[name]()
        (OUT / f"p2_{name}.json").write_text(json.dumps(json_ready(payload), indent=2) + "\n")
        print("wrote", name, flush=True)
