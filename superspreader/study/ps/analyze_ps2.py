#!/usr/bin/env python3
"""PS2 summaries: generation intervals, pre-diagnosis share, lever reductions, T6."""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import hospital, load_daily, prevalence  # noqa: E402
from early_window import early_counts  # noqa: E402
from ps_analysis import json_ready, load_trace, mom_k, paired_t_interval, t_interval  # noqa: E402
from ps_common import RESULTS  # noqa: E402

ARMS = ("A0", "C", "Pk50", "Pk75")
LEVERS = ("none", "isolate", "household")
WINDOW = (1, 19)  # Pk50 half-maximum window
F_PEAK = 0.9493656009796918


def runs(arm, lever):
    root = RESULTS / "ps2" / f"{arm}_{lever}"
    return sorted(p.parent for p in root.glob("rep*/trace.tsv"))


def quantiles(values):
    if not values:
        return {}
    s = sorted(values)
    def q(p):
        pos = p * (len(s) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    mean = sum(s) / len(s)
    var = sum((v - mean) ** 2 for v in s) / (len(s) - 1) if len(s) > 1 else 0
    return {"n": len(s), "mean": mean, "sd": math.sqrt(var), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}


def index_agents(trace):
    return {row["agent"]: row for row in trace}


def offspring(trace):
    kids = defaultdict(list)
    for row in trace:
        if row["infector"] is not None:
            kids[row["infector"]].append(row)
    return kids


def generation_intervals(trace):
    out = []
    for row in trace:
        if row["infector"] is None or row.get("infector_day") is None:
            continue
        out.append(row["day"] - row["infector_day"])
    return out


def stage_shares(trace):
    """Share of transmissions by the infector's stage and by diagnosis."""
    by_agent = index_agents(trace)
    kids = offspring(trace)
    buckets = {"before_symptom": 0, "within_2d_of_onset": 0, "later": 0, "never_symptomatic": 0}
    before_dx = 0
    after_dx = 0
    total = 0
    for infector, children in kids.items():
        parent = by_agent.get(infector)
        if parent is None:
            continue
        onset = parent.get("symptom_ts")
        diagnosed = parent["diag"]
        for child in children:
            total += 1
            if onset is None:
                buckets["never_symptomatic"] += 1
            else:
                delta_days = child["day"] - onset // 144
                if child["ts"] < onset:
                    buckets["before_symptom"] += 1
                elif delta_days <= 2:
                    buckets["within_2d_of_onset"] += 1
                else:
                    buckets["later"] += 1
            if diagnosed <= 0 or child["ts"] < diagnosed:
                before_dx += 1
            else:
                after_dx += 1
    if total == 0:
        return {}
    return {**{k: v / total for k, v in buckets.items()}, "before_diagnosis": before_dx / total,
            "after_diagnosis": after_dx / total, "transmissions": total}


def top_share_before_diagnosis(trace, fraction=0.10):
    by_agent = index_agents(trace)
    kids = offspring(trace)
    # Offspring of every infected agent, zeros included.
    counts = [(row["agent"], len(kids.get(row["agent"], []))) for row in trace]
    if not counts:
        return None
    counts.sort(key=lambda item: item[1], reverse=True)
    cutoff = max(1, int(round(fraction * len(counts))))
    top = {agent for agent, _ in counts[:cutoff]}
    def share(agents):
        before = 0
        total = 0
        for agent in agents:
            parent = by_agent.get(agent)
            if parent is None:
                continue
            for child in kids.get(agent, []):
                total += 1
                if parent["diag"] <= 0 or child["ts"] < parent["diag"]:
                    before += 1
        return (before / total) if total else None, total
    top_share, top_n = share(top)
    rest = {agent for agent, _ in counts[cutoff:]}
    rest_share, rest_n = share(rest)
    return {"top_share_before_diagnosis": top_share, "top_transmissions": top_n,
            "rest_share_before_diagnosis": rest_share, "rest_transmissions": rest_n, "cutoff_agents": cutoff}


def final_infected(run):
    rows = load_daily(run / "stdout.txt")
    if not rows:
        return None, None, None
    return int(rows[-1].get("INF", 0)), max(hospital(row) for row in rows), sum(int(row.get("Q", 0)) for row in rows)


def early(run):
    trace = load_trace(run / "trace.tsv")
    counts = early_counts(trace)
    if len(counts) < 2:
        return None, None, len(counts)
    return sum(counts) / len(counts), mom_k(counts), len(counts)


def arm_summary(arm, lever):
    per = []
    intervals = []
    stages = []
    tops = []
    for run in runs(arm, lever):
        trace = load_trace(run / "trace.tsv")
        r, k, n = early(run)
        infected, peak_h, qdays = final_infected(run)
        gi = generation_intervals(trace)
        per.append({"run": run.name, "early_R": r, "early_k": k, "infectors": n, "infected": infected,
                    "peak_hospital": peak_h, "quarantine_days": qdays, "gi": quantiles(gi)})
        intervals.extend(gi)
        stages.append(stage_shares(trace))
        tops.append(top_share_before_diagnosis(trace))
    def col(key):
        return [row[key] for row in per]
    return {
        "replicates": len(per),
        "early_R": t_interval(col("early_R")),
        "early_k": t_interval([row["early_k"] for row in per if row["early_k"] is not None and not math.isinf(row["early_k"])]),
        "infected": t_interval(col("infected")),
        "peak_hospital": t_interval(col("peak_hospital")),
        "quarantine_days": t_interval(col("quarantine_days")),
        "generation_interval": quantiles(intervals),
        "stage": {key: t_interval([s.get(key) for s in stages if s]) for key in
                  ("before_symptom", "within_2d_of_onset", "later", "never_symptomatic", "before_diagnosis")},
        "top10_before_diagnosis": t_interval([t["top_share_before_diagnosis"] for t in tops if t]),
        "rest_before_diagnosis": t_interval([t["rest_share_before_diagnosis"] for t in tops if t]),
        "per_replicate": [{k: v for k, v in row.items() if k != "gi"} | {"gi_mean": row["gi"].get("mean")} for row in per],
    }


def reductions(arm):
    none = {row["run"]: row for row in arm_summary(arm, "none")["per_replicate"]}
    # recompute from the stored per-replicate by re-reading would duplicate. Use the summary's per_replicate.
    out = {}
    base = {row["run"]: row for row in json_ready(arm_summary(arm, "none"))["per_replicate"]}
    for lever in ("isolate", "household"):
        other = {row["run"]: row for row in arm_summary(arm, lever)["per_replicate"]}
        seeds = sorted(set(base) & set(other))
        base_inf = [base[s]["infected"] for s in seeds]
        lev_inf = [other[s]["infected"] for s in seeds]
        reduction = [(b - l) / b if b else None for b, l in zip(base_inf, lev_inf)]
        averted = [b - l for b, l in zip(base_inf, lev_inf)]
        qdays = [other[s]["quarantine_days"] for s in seeds]
        per_1000 = [1000 * a / q if q else None for a, q in zip(averted, qdays)]
        out[lever] = {
            "reduction": t_interval(reduction),
            "averted": t_interval(averted),
            "per_1000_quarantine_days": t_interval(per_1000),
            "paired_infected_difference": paired_t_interval(base_inf, lev_inf),
            "per_seed_reduction": reduction,
        }
    return out


def t6():
    def load(path):
        rows = []
        with open(path) as handle:
            next(handle)
            for line in handle:
                day, total, count = line.strip().split(",")
                count = float(count)
                if count >= 30 and int(day) <= 21:
                    rows.append((int(day), float(total) / count))
        return rows
    base = dict(load(RESULTS / "ps2" / "t6_constant" / "profile_by_day.csv"))
    peak = dict(load(RESULTS / "ps2" / "t6_peak" / "profile_by_day.csv"))
    days = sorted(set(base) & set(peak))
    base_area = sum(base[d] for d in days)
    peak_area = sum(peak[d] for d in days)
    outside = [d for d in days if not (WINDOW[0] <= d <= WINDOW[1])]
    outside_ratio = [peak[d] / base[d] for d in outside if base[d] > 0]
    # Multiplier moments on the peak run, and the implied episode-total CV2.
    trace = load_trace(RESULTS / "ps2" / "t6_peak" / "trace.tsv")
    sheds = [row["shed"] for row in trace if row["infector"] is not None]
    mean = sum(sheds) / len(sheds)
    var = sum((s - mean) ** 2 for s in sheds) / (len(sheds) - 1)
    cv2 = var / mean ** 2
    shape = 0.270389
    implied = (F_PEAK ** 2) * cv2
    target = (F_PEAK ** 2) / shape
    return {
        "days": days,
        "area_ratio_peak_over_constant": peak_area / base_area,
        "area_within_2pct": abs(peak_area / base_area - 1) <= 0.02,
        "outside_day_ratios": {str(d): peak[d] / base[d] for d in outside if base[d] > 0},
        "outside_all_within_15pct": all(abs(r - 1) <= 0.15 for r in outside_ratio),
        "multiplier_mean": mean,
        "multiplier_cv2": cv2,
        "cv2_target_1_over_shape": 1 / shape,
        "implied_episode_cv2": implied,
        "formula_episode_cv2": target,
        "implied_within_15pct_of_formula": abs(implied / target - 1) <= 0.15,
    }


def main():
    payload = {arm: {lever: arm_summary(arm, lever) for lever in LEVERS} for arm in ARMS}
    payload["reductions"] = {arm: reductions(arm) for arm in ARMS}
    payload["t6"] = t6()
    # Ranking: household reduction versus isolate reduction, point estimates.
    rank = {}
    for arm in ("C", "Pk50"):
        iso = payload["reductions"][arm]["isolate"]["reduction"]["mean"]
        hh = payload["reductions"][arm]["household"]["reduction"]["mean"]
        rank[arm] = {"isolate": iso, "household": hh, "household_beats_isolate": hh > iso}
    payload["ranking_unchanged"] = rank["C"]["household_beats_isolate"] == rank["Pk50"]["household_beats_isolate"]
    payload["rank"] = rank
    out = RESULTS / "ps2" / "summary.json"
    # per_replicate makes this large but is the raw material the note cites
    out.write_text(json.dumps(json_ready(payload), indent=1) + "\n")
    print("T6", json.dumps(json_ready(payload["t6"]), indent=1)[:1500])
    print("ranking", json.dumps(json_ready(rank), indent=1), "unchanged", payload["ranking_unchanged"])
    for arm in ARMS:
        none = payload[arm]["none"]
        print(f"{arm} early R {none['early_R']['mean']:.3f} k {none['early_k']['mean']:.3f} "
              f"infected {none['infected']['mean']:.0f} GI {none['generation_interval']['mean']:.2f} "
              f"before dx {none['stage']['before_diagnosis']['mean']:.3f} "
              f"top10 before dx {none['top10_before_diagnosis']['mean']:.3f}")
        for lever in ("isolate", "household"):
            red = payload["reductions"][arm][lever]["reduction"]
            print(f"  {lever} reduction {red['mean']:.3f} ({red['low']:.3f} to {red['high']:.3f})")


if __name__ == "__main__":
    main()
