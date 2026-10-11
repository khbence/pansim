#!/usr/bin/env python3
"""Structure of subcritical chains: generation depth, generation interval, setting inheritance.

Tests two explanations for why the caregiving arm has a thinner middle and heavier extreme
tail than a negative binomial with the same mean and k (10_reading_of_09_phase2.md, section 4).
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import load_trace  # noqa: E402

STEPS_PER_DAY = 144
RESULTS = HERE / "results" / "p2"
OUT = HERE / "summaries"


def load(path):
    rows = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: pos for pos, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            infector = int(parts[index["infector"]])
            rows[int(parts[index["agent"]])] = {
                "parent": None if infector == 4294967295 else infector,
                "type": int(parts[index["locType"]]),
                "time": int(parts[index["timestamp"]]),
                "shed": float(parts[index["shed"]]),
            }
    return rows


def analyse_run(rows):
    children = defaultdict(list)
    roots = []
    for agent, row in rows.items():
        if row["parent"] is not None and row["parent"] in rows:
            children[row["parent"]].append(agent)
        else:
            roots.append(agent)
    chains = []
    for root in roots:
        size, depth, stack = 0, 0, [(root, 0)]
        members = []
        while stack:
            node, d = stack.pop()
            size += 1
            depth = max(depth, d)
            members.append(node)
            for child in children.get(node, ()):
                stack.append((child, d + 1))
        chains.append((root, size, depth, members))
    return children, chains


def summarise(arm):
    pair_same = pair_total = 0
    gaps = []
    type_counts = Counter()
    big = {"chains": 0, "nodes": 0, "types": Counter(), "depths": [], "sizes": [], "root_types": Counter(), "shed_sum": 0.0}
    all_sizes, all_depths = [], []
    runs = sorted(RESULTS.glob(f"{arm}/rep*"))
    for run in runs:
        rows = load(run / "trace.tsv")
        children, chains = analyse_run(rows)
        for agent, row in rows.items():
            type_counts[row["type"]] += 1
            parent = row["parent"]
            if parent is not None and parent in rows:
                pair_total += 1
                pair_same += rows[parent]["type"] == row["type"]
                gaps.append((row["time"] - rows[parent]["time"]) / STEPS_PER_DAY)
        for root, size, depth, members in chains:
            all_sizes.append(size)
            all_depths.append(depth)
            if size > 10:
                big["chains"] += 1
                big["nodes"] += size
                big["depths"].append(depth)
                big["sizes"].append(size)
                big["root_types"][rows[root]["type"]] += 1
                for m in members:
                    big["types"][rows[m]["type"]] += 1
                    big["shed_sum"] += rows[m]["shed"]
    total = sum(type_counts.values())
    shares = {t: c / total for t, c in type_counts.items()}
    expected_same = sum(s * s for s in shares.values())
    gaps.sort()
    n = len(gaps)
    nonroot = [d for d in all_depths]
    return {
        "runs": len(runs),
        "chains": len(all_sizes),
        "infections": total,
        "mean_generation_interval_days": sum(gaps) / n if n else None,
        "median_generation_interval_days": gaps[n // 2] if n else None,
        "mean_chain_depth": sum(nonroot) / len(nonroot),
        "max_chain_depth": max(nonroot),
        "chains_depth_ge_3": sum(d >= 3 for d in all_depths),
        "chains_depth_ge_5": sum(d >= 5 for d in all_depths),
        "parent_child_same_type": pair_same / pair_total if pair_total else None,
        "expected_same_type_if_independent": expected_same,
        "type_share_all": {str(t): round(s, 4) for t, s in sorted(shares.items())},
        "chains_gt_10": big["chains"],
        "big_chain_mean_depth": sum(big["depths"]) / len(big["depths"]) if big["depths"] else None,
        "big_chain_type_share": {
            str(t): round(c / big["nodes"], 4) for t, c in sorted(big["types"].items())
        } if big["nodes"] else {},
        "big_chain_root_types": {str(t): c for t, c in sorted(big["root_types"].items())},
        "big_chain_mean_shed": big["shed_sum"] / big["nodes"] if big["nodes"] else None,
        "top_chain_sizes": sorted(all_sizes, reverse=True)[:10],
    }


if __name__ == "__main__":
    arms = sys.argv[1:] or [
        "chain_flat_shed0.04",
        "chain_caregiving_shed0.09",
        "chain_flat_shed0.125",
        "chain_flat",
        "chain_caregiving",
    ]
    payload = {arm: summarise(arm) for arm in arms}
    OUT.mkdir(exist_ok=True)
    (OUT / "p3_chain_structure.json").write_text(json.dumps(payload, indent=1) + "\n")
    for arm, value in payload.items():
        print(arm, json.dumps({k: v for k, v in value.items() if k not in ("type_share_all", "big_chain_type_share")}))
        print("  all types", value["type_share_all"])
        print("  big chain types", value["big_chain_type_share"])


def by_parent_type(arm, types=(2, 12, 7, 8, 14)):
    offspring, count, child_hospital, children = Counter(), Counter(), Counter(), Counter()
    for run in sorted(RESULTS.glob(f"{arm}/rep*")):
        rows = load(run / "trace.tsv")
        kids = defaultdict(list)
        for agent, row in rows.items():
            if row["parent"] is not None and row["parent"] in rows:
                kids[row["parent"]].append(agent)
        for agent, row in rows.items():
            if row["parent"] is None:
                continue
            kind = row["type"]
            count[kind] += 1
            offspring[kind] += len(kids.get(agent, []))
            children[kind] += len(kids.get(agent, []))
            child_hospital[kind] += sum(rows[c]["type"] == 12 for c in kids.get(agent, []))
    total = sum(count.values())
    return {
        "non_seed_infections": total,
        "hospital_share": count[12] / total,
        "by_parent_type": {
            str(t): {
                "n": count[t],
                "mean_offspring": offspring[t] / count[t] if count[t] else None,
                "child_in_hospital": child_hospital[t] / children[t] if children[t] else None,
            }
            for t in types
        },
    }
