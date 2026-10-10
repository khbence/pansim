#!/usr/bin/env python3
"""Estimate offspring dispersion and the experiment metrics from panSim traces."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

STEPS_PER_DAY = 144
CENSOR_DAYS = 14
UNKNOWN = 4294967295
HOME = 2

PREVALENCE_COLUMNS = ("E", "I1", "I2", "I3", "I4", "I5_h", "I6_h")
HOSPITAL_COLUMNS = ("I5_h", "I6_h", "R_h")


def load_trace(path):
    rows = []
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: pos for pos, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            timestamp = int(parts[index["timestamp"]])
            infector = int(parts[index["infector"]])
            rows.append(
                {
                    "agent": int(parts[index["agent"]]),
                    "infector": None if infector == UNKNOWN else infector,
                    "location": int(parts[index["location"]]),
                    "locType": int(parts[index["locType"]]),
                    "day": timestamp // STEPS_PER_DAY,
                    "variant": int(parts[index["variant"]]),
                    "shed": float(parts[index["shed"]]),
                    "diagnosed": int(parts[index["diagnosed"]]) > 0,
                    "hospitalized": int(parts[index["hospitalized"]]) > 0,
                    "infectedCount": int(parts[index["infectedCount"]]),
                }
            )
    return rows


def load_daily(path):
    with open(path) as handle:
        header = None
        for line in handle:
            if line.startswith("S\t"):
                header = line.rstrip("\n").split("\t")
                break
        if header is None:
            return []
        rows = []
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header) or not parts[0].lstrip("-").isdigit():
                continue
            row = {}
            for name, value in zip(header, parts):
                if "," in value:
                    row[name] = value
                else:
                    try:
                        row[name] = int(value)
                    except ValueError:
                        row[name] = value
            rows.append(row)
    return rows


def prevalence(row):
    return sum(int(row.get(name, 0)) for name in PREVALENCE_COLUMNS)


def hospital(row):
    return sum(int(row.get(name, 0)) for name in HOSPITAL_COLUMNS)


def daily_summary(rows):
    if not rows:
        return {}
    prev = [prevalence(row) for row in rows]
    hosp = [hospital(row) for row in rows]
    incidence = [int(row.get("NI", 0)) for row in rows]
    peak_i = max(range(len(prev)), key=lambda i: prev[i])
    peak_h = max(range(len(hosp)), key=lambda i: hosp[i])

    def growth(start):
        end = start + 7
        if end >= len(prev) or prev[start] <= 0 or prev[end] <= 0:
            return None
        return math.log(prev[end] / prev[start])

    return {
        "days": len(rows),
        "final_prevalence": prev[-1],
        "final_infected_ever": int(rows[-1].get("INF", 0)),
        "peak_prevalence": prev[peak_i],
        "peak_prevalence_day": peak_i,
        "peak_hospital": hosp[peak_h],
        "peak_hospital_day": peak_h,
        "final_hospital": hosp[-1],
        "growth_day7": growth(7),
        "growth_day14": growth(14),
        "growth_day21": growth(21),
        "quarantine_person_days": sum(int(row.get("Q", 0)) for row in rows),
    }


def case_table(trace, censor_days=CENSOR_DAYS):
    by_agent = {}
    for row in trace:
        by_agent[row["agent"]] = row
    for row in trace:
        infector = row["infector"]
        if infector is not None and infector not in by_agent:
            by_agent[infector] = {
                "agent": infector,
                "infector": None,
                "location": -1,
                "locType": -1,
                "day": 0,
                "variant": 0,
                "shed": 1.0,
                "diagnosed": False,
                "hospitalized": False,
                "infectedCount": 0,
                "synthetic": True,
            }
    children = defaultdict(list)
    for row in trace:
        if row["infector"] is not None:
            children[row["infector"]].append(row)
    last_day = max((row["day"] for row in trace), default=0)
    cutoff = last_day - censor_days
    index = []
    for agent, row in by_agent.items():
        if row["day"] <= cutoff:
            index.append(agent)
    return by_agent, children, index, last_day, cutoff


def offspring_counts(children, index):
    return [len(children.get(agent, [])) for agent in index]


def mean_var(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, var


def mom_k(values):
    mean, var = mean_var(values)
    if mean is None or mean <= 0:
        return None
    if var <= mean * 1.0000001:
        return math.inf
    return mean * mean / (var - mean)


def nb_loglik(values, k, mean):
    if k <= 0 or mean < 0:
        return -math.inf
    if mean == 0:
        return 0.0 if all(value == 0 for value in values) else -math.inf
    log_p = k * math.log(k / (k + mean))
    log_q = math.log(mean / (k + mean))
    total = 0.0
    for value in values:
        total += (
            math.lgamma(value + k)
            - math.lgamma(k)
            - math.lgamma(value + 1)
            + log_p
            + value * log_q
        )
    return total


def fit_nb(values):
    mean, _var = mean_var(values)
    if mean is None:
        return {"n": 0}
    k_mom = mom_k(values)
    if not values:
        return {"n": 0, "mean": mean}
    if k_mom is None:
        k_hat = None
        loglik = None
    elif math.isinf(k_mom):
        k_hat = math.inf
        loglik = sum(
            -mean + value * math.log(mean) - math.lgamma(value + 1) for value in values
        ) if mean > 0 else 0.0
    else:
        grid = [k_mom * (10 ** (i / 40)) for i in range(-80, 81)]
        grid = [k for k in grid if 0.005 <= k <= 200]
        best_k = max(grid, key=lambda k: nb_loglik(values, k, mean))
        for _ in range(12):
            neighbors = [best_k * factor for factor in (0.9, 0.97, 1.0, 1.03, 1.1)]
            best_k = max(neighbors, key=lambda k: nb_loglik(values, k, mean))
        k_hat = best_k
        loglik = nb_loglik(values, k_hat, mean)
    return {
        "n": len(values),
        "mean": mean,
        "variance": mean_var(values)[1],
        "k_mom": k_mom,
        "k": k_hat,
        "loglik": loglik,
        "zero_fraction": sum(value == 0 for value in values) / len(values),
    }


def poisson_lognormal_loglik(values, sigma, mean, nodes=24):
    if mean <= 0 or sigma < 0:
        return -math.inf
    if sigma == 0:
        return sum(-mean + value * math.log(mean) - math.lgamma(value + 1) for value in values)
    mu = math.log(mean) - 0.5 * sigma * sigma
    # Gauss-Hermite style grid on a standard normal, unnormalized trapezoid.
    zs = []
    weights = []
    span = 4.0
    for i in range(nodes):
        z = -span + (2 * span) * i / (nodes - 1)
        w = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        zs.append(z)
        weights.append(w)
    norm = sum(weights) * (zs[1] - zs[0])
    total = 0.0
    for value in values:
        prob = 0.0
        for z, weight in zip(zs, weights):
            lam = math.exp(mu + sigma * z)
            prob += weight * math.exp(-lam + value * math.log(lam) - math.lgamma(value + 1))
        prob *= (zs[1] - zs[0]) / norm
        if prob <= 0:
            return -math.inf
        total += math.log(prob)
    return total


def fit_lognormal(values, mean):
    if mean is None or mean <= 0 or not values:
        return None
    grid = [i / 20 for i in range(0, 41)]
    best = max(grid, key=lambda sigma: poisson_lognormal_loglik(values, sigma, mean))
    return {
        "sigma": best,
        "loglik": poisson_lognormal_loglik(values, best, mean),
    }


def top_share(weights, fraction):
    if not weights or sum(weights) <= 0:
        return None
    ordered = sorted(weights, reverse=True)
    count = max(1, int(math.ceil(fraction * len(ordered))))
    return sum(ordered[:count]) / sum(ordered)


def share_for_fraction_of_mass(weights, mass=0.8):
    if not weights or sum(weights) <= 0:
        return None
    ordered = sorted(weights, reverse=True)
    total = sum(ordered)
    running = 0
    for i, value in enumerate(ordered, start=1):
        running += value
        if running >= mass * total:
            return i / len(ordered)
    return 1.0


def venue_metrics(trace):
    counts = defaultdict(int)
    by_type = defaultdict(int)
    for row in trace:
        if row["infector"] is None and row["day"] == 0 and row["infectedCount"] == 0:
            continue
        counts[row["location"]] += 1
        by_type[row["locType"]] += 1
    values = list(counts.values())
    total = sum(by_type.values()) or 1
    community = sum(count for loc_type, count in by_type.items() if loc_type not in (HOME, -1))
    return {
        "venues_with_infection": len(values),
        "top10_venues": top_share(values, 0.10),
        "top20_venues": top_share(values, 0.20),
        "venues_for_80": share_for_fraction_of_mass(values, 0.8),
        "household_fraction": by_type.get(HOME, 0) / total,
        "community_fraction": community / total,
        "by_type": {str(key): value for key, value in sorted(by_type.items())},
    }


def generation1_r(trace):
    seeds = [row["agent"] for row in trace if row["infector"] is None]
    children = defaultdict(int)
    for row in trace:
        if row["infector"] is not None:
            children[row["infector"]] += 1
    if not seeds:
        return None
    return sum(children.get(seed, 0) for seed in seeds) / len(seeds)


def chain_sizes(trace):
    parent = {}
    nodes = set()
    for row in trace:
        nodes.add(row["agent"])
        if row["infector"] is not None:
            nodes.add(row["infector"])
            parent[row["agent"]] = row["infector"]

    def find(agent):
        while agent in parent and parent[agent] != agent:
            agent = parent[agent]
        return agent

    clusters = defaultdict(int)
    for agent in nodes:
        clusters[find(agent)] += 1
    sizes = list(clusters.values())
    if not sizes:
        return {"chains": 0}
    return {
        "chains": len(sizes),
        "max_chain": max(sizes),
        "mean_chain": sum(sizes) / len(sizes),
        "p_gt_10": sum(size > 10 for size in sizes) / len(sizes),
        "p_gt_50": sum(size > 50 for size in sizes) / len(sizes),
        "size_histogram": dict(sorted(defaultdict(int, {size: sizes.count(size) for size in set(sizes)}).items())),
    }


def component_histogram(sizes):
    hist = defaultdict(int)
    for size in sizes:
        hist[size] += 1
    return dict(sorted(hist.items()))


def chain_sizes_fixed(trace):
    parent = {}
    nodes = set()
    for row in trace:
        nodes.add(row["agent"])
        if row["infector"] is not None:
            nodes.add(row["infector"])
            parent[row["agent"]] = row["infector"]

    def root(agent):
        seen = set()
        while agent in parent and agent not in seen:
            seen.add(agent)
            agent = parent[agent]
        return agent

    clusters = defaultdict(int)
    for agent in nodes:
        clusters[root(agent)] += 1
    sizes = list(clusters.values())
    if not sizes:
        return {"chains": 0, "sizes": []}
    return {
        "chains": len(sizes),
        "max_chain": max(sizes),
        "mean_chain": sum(sizes) / len(sizes),
        "p_gt_10": sum(size > 10 for size in sizes) / len(sizes),
        "p_gt_50": sum(size > 50 for size in sizes) / len(sizes),
        "sizes": sizes,
    }


def ascertainment(trace, censor_days=CENSOR_DAYS):
    by_agent, children, index, _last, _cutoff = case_table(trace, censor_days)
    true_counts = offspring_counts(children, index)
    diagnosed_index = [agent for agent in index if by_agent[agent].get("diagnosed")]
    forward_counts = []
    for agent in diagnosed_index:
        forward_counts.append(sum(1 for child in children.get(agent, []) if child["diagnosed"]))
    # Backward tracing keeps an infector when at least one offspring is diagnosed.
    backward_index = []
    backward_counts = []
    for agent in index:
        diagnosed_children = sum(1 for child in children.get(agent, []) if child["diagnosed"])
        if by_agent[agent].get("diagnosed") or diagnosed_children > 0:
            backward_index.append(agent)
            backward_counts.append(diagnosed_children if not by_agent[agent].get("diagnosed") else len(children.get(agent, [])))
    # Cluster-triggered: a venue is investigated when it has at least one diagnosis
    # and at least three infections. Observed offspring stay inside observed cases.
    venue_counts = defaultdict(int)
    venue_diagnosed = defaultdict(int)
    for row in trace:
        venue_counts[row["location"]] += 1
        if row["diagnosed"]:
            venue_diagnosed[row["location"]] += 1
    detected_venues = {
        venue
        for venue, count in venue_counts.items()
        if count >= 3 and venue_diagnosed[venue] >= 1
    }
    observed = {row["agent"] for row in trace if row["location"] in detected_venues}
    cluster_index = [agent for agent in index if agent in observed or by_agent[agent].get("synthetic")]
    cluster_counts = [
        sum(1 for child in children.get(agent, []) if child["agent"] in observed) for agent in cluster_index
    ]
    true_fit = fit_nb(true_counts)
    return {
        "true": true_fit,
        "symptom_forward": fit_nb(forward_counts),
        "backward": fit_nb(backward_counts),
        "cluster": fit_nb(cluster_counts),
        "diagnosed_fraction": (
            sum(1 for agent in index if by_agent[agent].get("diagnosed")) / len(index) if index else None
        ),
        "detected_venue_fraction": (
            sum(1 for row in trace if row["location"] in detected_venues) / len(trace) if trace else None
        ),
    }


def bootstrap_k(values, draws=200, seed=1):
    if len(values) < 20:
        return None
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        fit = fit_nb(sample)
        k_hat = fit.get("k")
        if k_hat is not None and not math.isinf(k_hat):
            estimates.append(k_hat)
    if len(estimates) < 20:
        return None
    estimates.sort()
    return {
        "low": estimates[int(0.025 * len(estimates))],
        "high": estimates[int(0.975 * len(estimates)) - 1],
    }


def summarize_run(stdout_path, trace_path, censor_days=CENSOR_DAYS, bootstrap=False, lognormal=False):
    daily = daily_summary(load_daily(stdout_path))
    trace = load_trace(trace_path)
    by_agent, children, index, last_day, cutoff = case_table(trace, censor_days)
    counts = offspring_counts(children, index)
    fit = fit_nb(counts)
    alt = fit_lognormal(counts, fit.get("mean")) if lognormal else None
    venues = venue_metrics(trace)
    result = {
        "daily": daily,
        "last_day": last_day,
        "censor_day": cutoff,
        "index_cases": len(index),
        "infections": len(trace),
        "offspring": fit,
        "lognormal": alt,
        "top10_infectors": top_share(counts, 0.10),
        "top20_infectors": top_share(counts, 0.20),
        "infectors_for_80": share_for_fraction_of_mass(counts, 0.8),
        "venues": venues,
        "generation1_r": generation1_r(trace),
        "shed_mean": (sum(row["shed"] for row in trace) / len(trace)) if trace else None,
        "k_bootstrap": bootstrap_k(counts) if bootstrap else None,
    }
    if alt and fit.get("loglik") is not None and not math.isinf(fit.get("k", 0) or 0):
        result["lognormal_better"] = alt["loglik"] > fit["loglik"] + 2
    return result, counts


def pool_runs(run_dirs, censor_days=CENSOR_DAYS):
    pooled = []
    per_run = []
    for run_dir in run_dirs:
        summary, counts = summarize_run(
            run_dir / "stdout.txt", run_dir / "trace.tsv", censor_days, lognormal=False
        )
        summary["run"] = run_dir.name
        per_run.append(summary)
        pooled.extend(counts)
    fit = fit_nb(pooled)
    lognormal_sample = pooled
    if len(lognormal_sample) > 20000:
        rng = random.Random(0)
        lognormal_sample = rng.sample(lognormal_sample, 20000)
    alt = fit_lognormal(lognormal_sample, fit.get("mean"))
    growths = [item["daily"].get("growth_day14") for item in per_run if item["daily"].get("growth_day14") is not None]
    finals = [item["infections"] for item in per_run]
    extinct = sum(1 for item in per_run if item["infections"] < 80)
    return {
        "replicates": len(per_run),
        "pooled": fit,
        "lognormal": alt,
        "k_bootstrap": bootstrap_k(pooled),
        "top10_infectors": top_share(pooled, 0.10),
        "top20_infectors": top_share(pooled, 0.20),
        "infectors_for_80": share_for_fraction_of_mass(pooled, 0.8),
        "mean_infections": sum(finals) / len(finals) if finals else None,
        "extinction_fraction": extinct / len(per_run) if per_run else None,
        "growth_day14_mean": sum(growths) / len(growths) if growths else None,
        "growth_day14_variance": mean_var(growths)[1] if len(growths) > 1 else None,
        "runs": per_run,
    }


def json_ready(value):
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None if math.isnan(value) else "infinity"
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1])
    summary, _counts = summarize_run(target / "stdout.txt", target / "trace.tsv")
    print(json.dumps(json_ready(summary), indent=2))
