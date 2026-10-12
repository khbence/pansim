#!/usr/bin/env python3
"""D1: test the absorption / ceiling account on the PS1 traces already on disk.

Index cases are infections acquired on days 1-10 with a known infector. Offspring are
children infected within 14 days. That is the same window as the published early k.

trace.tsv has one row per agent (the latest episode) and no uninfected contacts, so
"distinct contacts" cannot be measured. The script counts distinct transmission
locations among the 14-day offspring, which is at most the offspring count, and it
tests the ceiling on the offspring-versus-multiplier curve.

Pre-registered decision, fixed before the pooled estimates were used:

- The multiplier is uncompressed when pooled CV^2 of shed among index cases is
  within 20% of the generative expectation, or the replicate t interval covers it.
  Expectation is 1/shape for episode draws, uncoupled traits and lognormal traits
  (lognormal CV^2 is invariant to size biasing), and 1/(shape+rho) for a coupled
  gamma trait.
- A mean ceiling is confirmed for an arm with expected CV^2 >= 1 only when the
  top 1% mean offspring is below 60% of the lower-half proportional prediction,
  the replicate-bootstrap 97.5% of that ratio is below 0.80, and the elasticity
  of mean offspring to mean shed from the [0.5, 1) bin to the shed>=8 bin is
  below 0.60.
- Truncation is confirmed only when every such arm passes. Otherwise the
  composition arithmetic is dropped rather than given a ceiling explanation.

The multiplier claim in the verdict uses the six arms in note 18 section 3.
Arms outside that table are still measured and still counted in the ceiling
conjunction when their expected CV^2 is at least 1. A CV^2 above the target
is recorded as "above", not as compression.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps_analysis import json_ready, t_interval  # noqa: E402
from ps_common import RESULTS, STEPS_PER_DAY  # noqa: E402

UNKNOWN = 4294967295
CV_TOL = 0.20
RATIO_CEILING = 0.60
BOOT_HIGH_CEILING = 0.80
ELASTICITY_CEILING = 0.60
BOOT_DRAWS = 1000
NB_DRAWS = 400

ARMS = {
    "A0": {"family": "none", "shape": None, "rho": 0.0, "dist": None},
    "E": {"family": "episode", "shape": 0.3, "rho": 0.0, "dist": "gamma"},
    "P0": {"family": "persistent", "shape": 0.3, "rho": 0.0, "dist": "gamma"},
    "P25": {"family": "persistent", "shape": 0.3, "rho": 0.25, "dist": "gamma"},
    "P50": {"family": "persistent", "shape": 0.3, "rho": 0.5, "dist": "gamma"},
    "P100": {"family": "persistent", "shape": 0.3, "rho": 1.0, "dist": "gamma"},
    "P100ln": {"family": "persistent", "shape": 0.3, "rho": 1.0, "dist": "lognormal"},
    "P25k": {"family": "persistent", "shape": 0.08, "rho": 0.25, "dist": "gamma"},
    "P50a005": {"family": "persistent", "shape": 0.05, "rho": 0.5, "dist": "gamma"},
    "P100a005": {"family": "persistent", "shape": 0.05, "rho": 1.0, "dist": "gamma"},
}

ABS_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]


def expected_infected_cv2(meta):
    if meta["family"] == "none":
        return 0.0
    if meta["dist"] == "lognormal" or meta["family"] == "episode" or meta["rho"] == 0.0:
        return 1.0 / meta["shape"]
    return 1.0 / (meta["shape"] + meta["rho"])


def expected_mean(meta):
    if meta["family"] == "none" or meta["family"] == "episode" or meta["rho"] == 0.0:
        return 1.0
    if meta["dist"] == "lognormal":
        return (1.0 + 1.0 / meta["shape"]) ** meta["rho"]
    return (meta["shape"] + meta["rho"]) / meta["shape"]


def drawn_cv2(meta):
    if meta["family"] == "none":
        return 0.0
    return 1.0 / meta["shape"]


def cv2(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    mean = float(values.mean())
    if mean == 0.0:
        return None
    var = float(values.var(ddof=1))
    return var / (mean * mean)


def mom_k(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    mean = float(values.mean())
    var = float(values.var(ddof=1))
    if mean <= 0.0:
        return None
    if var <= mean:
        return float("inf")
    return mean * mean / (var - mean)


def implied_residual(total_cv2, trait_cv2):
    if total_cv2 is None or trait_cv2 is None or not np.isfinite(total_cv2):
        return None
    return (total_cv2 - trait_cv2) / (1.0 + trait_cv2)


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = average
        i = j + 1
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom == 0.0:
        return None
    return float(np.dot(x, y) / denom)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def scan_trace(path):
    """Return shed, 14-day offspring, distinct child locations, and reinfection counts for days 1-10."""
    day = {}
    shed = {}
    known = {}
    infected_count = {}
    children = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: pos for pos, name in enumerate(header)}
        i_agent = index["agent"]
        i_inf = index["infector"]
        i_loc = index["location"]
        i_ts = index["timestamp"]
        i_shed = index["shed"]
        i_count = index["infectedCount"]
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            agent = int(parts[i_agent])
            infector = int(parts[i_inf])
            day[agent] = int(parts[i_ts]) // STEPS_PER_DAY
            shed[agent] = float(parts[i_shed])
            infected_count[agent] = int(parts[i_count])
            if infector != UNKNOWN:
                known[agent] = True
                bucket = children.get(infector)
                if bucket is None:
                    children[infector] = [(day[agent], int(parts[i_loc]))]
                else:
                    bucket.append((day[agent], int(parts[i_loc])))
            else:
                known[agent] = False
    sheds, offs, locs = [], [], []
    reinfected = 0
    for agent, infected_day in day.items():
        if not known[agent] or infected_day < 1 or infected_day > 10:
            continue
        limit = infected_day + 14
        n = 0
        places = set()
        for child_day, place in children.get(agent, ()):
            if child_day <= limit:
                n += 1
                places.add(place)
        sheds.append(shed[agent])
        offs.append(n)
        locs.append(len(places))
        if infected_count[agent] > 1:
            reinfected += 1
    return (
        np.asarray(sheds, dtype=float),
        np.asarray(offs, dtype=float),
        np.asarray(locs, dtype=float),
        reinfected,
    )


def trait_population(path):
    values = []
    with open(path) as handle:
        handle.readline()
        for line in handle:
            values.append(float(line.split(",", 2)[1]))
    values = np.asarray(values, dtype=float)
    return {"n": int(len(values)), "mean": float(values.mean()), "cv2": cv2(values), "replicate": "rep1101"}


def bin_rows(arm, kind, shed, off, loc, edges, labels=None):
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (shed >= lo) & (shed <= hi)
        else:
            mask = (shed >= lo) & (shed < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "arm": arm,
            "bin_type": kind,
            "bin_lo": float(lo) if np.isfinite(lo) else None,
            "bin_hi": float(hi) if np.isfinite(hi) else None,
            "label": labels[i] if labels else "",
            "n": n,
            "mean_shed": float(shed[mask].mean()),
            "mean_offspring": float(off[mask].mean()),
            "mean_locations": float(loc[mask].mean()),
            "var_offspring": float(off[mask].var(ddof=1)) if n > 1 else None,
        })
    return rows


def decile_edges(shed):
    quantiles = np.quantile(shed, np.linspace(0.0, 1.0, 11))
    # Keep the bins non-empty when the lower tail piles up at zero.
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = np.nextafter(quantiles[i - 1], np.inf)
    quantiles[0] = min(quantiles[0], float(shed.min()))
    quantiles[-1] = max(quantiles[-1], float(shed.max()))
    return quantiles


def group_mean_ratio(shed, off, high_mask, low_mask):
    if high_mask.sum() < 8 or low_mask.sum() < 20:
        return None
    low_shed = float(shed[low_mask].mean())
    if low_shed <= 0.0:
        return None
    slope = float(off[low_mask].mean()) / low_shed
    if slope <= 0.0:
        return None
    predicted = slope * float(shed[high_mask].mean())
    if predicted <= 0.0:
        return None
    return float(off[high_mask].mean()) / predicted


def elasticity(shed, off):
    if len(shed) == 0 or float(np.max(shed)) <= float(np.min(shed)):
        return None, "no_multiplier_variation", 0, 0
    low = (shed >= 0.5) & (shed < 1.0)
    high = shed >= 8.0
    definition = "absolute_0.5_1_vs_ge_8"
    if low.sum() < 40 or high.sum() < 40:
        positive = shed > 0
        if positive.sum() < 80:
            return None, "undefined", 0, 0
        lo_cut = np.quantile(shed[positive], 0.25)
        hi_cut = np.quantile(shed, 0.90)
        low = positive & (shed <= lo_cut)
        high = shed >= hi_cut
        definition = "quartile_fallback"
        if low.sum() < 40 or high.sum() < 40:
            return None, definition, int(low.sum()), int(high.sum())
    mean_shed_low = float(shed[low].mean())
    mean_off_low = float(off[low].mean())
    if mean_shed_low <= 0.0 or mean_off_low <= 0.0:
        return None, definition, int(low.sum()), int(high.sum())
    value = (float(off[high].mean()) / mean_off_low) / (float(shed[high].mean()) / mean_shed_low)
    return float(value), definition, int(low.sum()), int(high.sum())


def top_mask(shed, fraction):
    threshold = float(np.quantile(shed, 1.0 - fraction))
    mask = shed >= threshold
    if mask.mean() > max(0.05, 3 * fraction) or float(shed.max()) <= float(shed.min()):
        return None, threshold
    return mask, threshold


def ratio_from_reps(reps, fraction):
    shed = np.concatenate([item[0] for item in reps])
    off = np.concatenate([item[1] for item in reps])
    if float(shed.max()) <= float(shed.min()):
        return None
    median = float(np.median(shed))
    low = shed <= median
    high, _threshold = top_mask(shed, fraction)
    if high is None:
        return None
    return group_mean_ratio(shed, off, high, low)


def bootstrap_ratio(reps, fraction, rng):
    point = ratio_from_reps(reps, fraction)
    if point is None:
        return None
    draws = []
    n = len(reps)
    for _ in range(BOOT_DRAWS):
        choice = rng.integers(0, n, n)
        value = ratio_from_reps([reps[i] for i in choice], fraction)
        if value is not None and np.isfinite(value):
            draws.append(value)
    draws = np.sort(np.asarray(draws, dtype=float))
    if len(draws) < BOOT_DRAWS // 2:
        return {"point": point, "low": None, "high": None, "n": int(len(draws))}
    return {
        "point": point,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n": int(len(draws)),
    }


def dist_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "var": float(values.var(ddof=1)) if len(values) > 1 else None,
        "zero": float(np.mean(values == 0)),
        "p90": float(np.quantile(values, 0.90)),
        "n": int(len(values)),
    }


def nb_stats(rng, means, k, draws=NB_DRAWS):
    means = np.asarray(means, dtype=float)
    collected = []
    for _ in range(draws):
        if k is None or not np.isfinite(k):
            sample = rng.poisson(means)
        else:
            lam = rng.gamma(shape=k, scale=np.maximum(means, 0.0) / k)
            sample = rng.poisson(lam)
        collected.append(dist_stats(sample))
    def col(name):
        vals = np.asarray([item[name] for item in collected], dtype=float)
        return {"mean": float(vals.mean()), "low": float(np.quantile(vals, 0.025)), "high": float(np.quantile(vals, 0.975))}
    return {"mean": col("mean"), "var": col("var"), "zero": col("zero"), "p90": col("p90")}


def fit_saturation(shed, off):
    """Grid search of offspring = a (1 - exp(-b shed)), against a straight line through the origin."""
    if len(shed) < 50 or float(np.dot(shed, shed)) <= 0.0:
        return None
    linear_slope = float(np.dot(off, shed) / np.dot(shed, shed))
    linear_sse = float(np.sum((off - linear_slope * shed) ** 2))
    best = None
    for b in np.logspace(-3, 1, 100):
        curve = 1.0 - np.exp(-b * shed)
        denom = float(np.dot(curve, curve))
        if denom <= 0.0:
            continue
        a = float(np.dot(off, curve) / denom)
        sse = float(np.sum((off - a * curve) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, float(b))
    half = float(np.log(2.0) / best[2]) if best and best[2] > 0 else None
    return {
        "linear_slope": linear_slope,
        "linear_sse": linear_sse,
        "saturating_a": best[1],
        "saturating_b": best[2],
        "saturating_sse": best[0],
        "sse_ratio_saturating_over_linear": best[0] / linear_sse if linear_sse > 0 else None,
        "half_saturation_shed": half,
    }


def analyse_arm(arm, meta, rng):
    root = RESULTS / "ps1" / arm
    reps = []
    reinfected = 0
    indexed = 0
    for run in sorted(root.glob("rep*/trace.tsv")):
        shed, off, loc, rein = scan_trace(run)
        if len(shed) == 0:
            continue
        reps.append((shed, off, loc, run.parent.name))
        reinfected += rein
        indexed += len(shed)
    if not reps:
        raise SystemExit(f"no traces for {arm}")
    shed = np.concatenate([item[0] for item in reps])
    off = np.concatenate([item[1] for item in reps])
    loc = np.concatenate([item[2] for item in reps])
    cv2_reps = [cv2(item[0]) for item in reps]
    k_value = mom_k(off)
    total_cv2 = None if k_value is None or not np.isfinite(k_value) or k_value <= 0 else 1.0 / k_value
    expected = expected_infected_cv2(meta)
    cv_interval = t_interval(cv2_reps)
    pooled_cv2 = cv2(shed)
    matches = False
    if expected == 0.0:
        matches = pooled_cv2 is not None and pooled_cv2 < 1e-6
    elif pooled_cv2 is not None:
        relative = abs(pooled_cv2 / expected - 1.0) <= CV_TOL
        covers = cv_interval["low"] is not None and cv_interval["low"] <= expected <= cv_interval["high"]
        matches = bool(relative or covers)
    ratio_top1 = bootstrap_ratio([(item[0], item[1]) for item in reps], 0.01, rng)
    ratio_decile = bootstrap_ratio([(item[0], item[1]) for item in reps], 0.10, rng)
    elast, elast_def, elast_n_low, elast_n_high = elasticity(shed, off)
    positive = off > 0
    cv2_positive = cv2(shed[positive]) if positive.sum() > 2 else None
    trait_path = root / "rep1101" / "agents_trait.csv"
    population = trait_population(trait_path) if trait_path.exists() else None
    top_mask_1, top_threshold = top_mask(shed, 0.01)
    saturation = fit_saturation(shed, off) if expected > 0 else None
    summary = {
        "arm": arm,
        "family": meta["family"],
        "shape": meta["shape"],
        "rho": meta["rho"],
        "dist": meta["dist"],
        "replicates": len(reps),
        "infectors": int(len(shed)),
        "reinfected_share": reinfected / indexed if indexed else None,
        "drawn_cv2": drawn_cv2(meta),
        "expected_cv2": expected,
        "expected_mean_shed": expected_mean(meta),
        "mean_shed": float(shed.mean()),
        "cv2_pooled": pooled_cv2,
        "cv2_replicate_t": cv_interval,
        "cv2_among_those_with_offspring": cv2_positive,
        "multiplier_matches_draw": matches,
        "early_R": float(off.mean()),
        "early_k": k_value,
        "total_cv2": total_cv2,
        "residual_vs_expected_trait": implied_residual(total_cv2, expected),
        "residual_vs_realised_shed": implied_residual(total_cv2, pooled_cv2),
        "pearson_shed_offspring": pearson(shed, off),
        "spearman_shed_offspring": spearman(shed, off),
        "pearson_shed_locations": pearson(shed, loc),
        "spearman_shed_locations": spearman(shed, loc),
        "mean_offspring": float(off.mean()),
        "mean_locations": float(loc.mean()),
        "p90_offspring": float(np.quantile(off, 0.90)),
        "p90_locations": float(np.quantile(loc, 0.90)),
        "p99_shed": float(np.quantile(shed, 0.99)),
        "max_shed": float(shed.max()),
        "top1_threshold_shed": top_threshold,
        "top1_ratio": ratio_top1,
        "top_decile_ratio": ratio_decile,
        "elasticity": elast,
        "elasticity_definition": elast_def,
        "elasticity_n_low": elast_n_low,
        "elasticity_n_high": elast_n_high,
        "saturation_fit": saturation,
        "population_trait": population,
        "contact_limitation": (
            "trace.tsv records infected agents only, so uninfected contacts are absent. "
            "mean_locations is the number of distinct places among 14-day offspring and cannot exceed offspring."
        ),
    }
    labels = [f"d{i}" for i in range(1, 11)]
    edges = decile_edges(shed)
    bins = bin_rows(arm, "decile", shed, off, loc, edges, labels)
    bins += bin_rows(arm, "absolute", shed, off, loc, ABS_EDGES)
    held = None
    if top_mask_1 is not None:
        held = {
            "shed": shed[top_mask_1],
            "off": off[top_mask_1],
            "low_shed_mean": float(shed[shed <= np.median(shed)].mean()),
            "low_off_mean": float(off[shed <= np.median(shed)].mean()),
        }
    return summary, bins, held, shed, off


def attach_top1_nulls(summary, held, k_a0, rng):
    if held is None:
        summary["top1"] = None
        return
    low_shed = held["low_shed_mean"]
    slope = held["low_off_mean"] / low_shed if low_shed > 0 else None
    means = slope * held["shed"] if slope and slope > 0 else None
    observed = dist_stats(held["off"])
    block = {
        "observed": observed,
        "mean_shed": float(held["shed"].mean()),
        "proportional_mean": None if means is None else float(means.mean()),
        "mean_ratio_vs_proportional": None if means is None or means.mean() <= 0 else observed["mean"] / float(means.mean()),
    }
    if means is not None:
        block["poisson_null"] = nb_stats(rng, means, None)
        if k_a0 and np.isfinite(k_a0):
            block["nb_a0_null"] = nb_stats(rng, means, k_a0)
        k_arm = summary["early_k"]
        if k_arm and np.isfinite(k_arm):
            block["nb_arm_k_at_proportional_mean"] = nb_stats(rng, np.full(len(means), float(means.mean())), k_arm)
    summary["top1"] = block


def within_bin_dispersion(bins, k_a0):
    out = []
    for row in bins:
        if row["var_offspring"] is None or row["mean_offspring"] <= 0:
            ratio = None
        else:
            ratio = row["var_offspring"] / row["mean_offspring"]
        benchmark = None if not k_a0 or not np.isfinite(k_a0) else 1.0 + row["mean_offspring"] / k_a0
        row = dict(row)
        row["var_over_mean"] = ratio
        row["nb_a0_var_over_mean"] = benchmark
        out.append(row)
    return out


SECTION3 = ("E", "P0", "P25", "P50", "P100", "P25k")
NARROW_BINS = ((0.8, 1.0), (1.0, 1.2), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, 20.0))


def k_from(mean, var):
    if mean <= 0 or var <= mean:
        return None
    return float(mean * mean / (var - mean))


def marginal_models(shed, off, k_a0):
    """Marginal k predicted from the realised shed values under two conditional stories."""
    if k_a0 is None or not np.isfinite(k_a0) or len(shed) < 50 or float(shed.max()) <= float(shed.min()):
        return None
    low = shed <= float(np.median(shed))
    if float(shed[low].mean()) <= 0.0:
        return None
    slope = float(off[low].mean()) / float(shed[low].mean())
    proportional = slope * shed
    edges = np.quantile(shed, np.linspace(0.0, 1.0, 21)).astype(float)
    edges[0] = float(shed.min()) - 1e-9
    edges[-1] = float(shed.max()) + 1e-9
    empirical = np.empty_like(shed)
    for i in range(len(edges) - 1):
        mask = (shed >= edges[i]) & (shed < edges[i + 1])
        if np.any(mask):
            empirical[mask] = off[mask].mean()

    def pack(means):
        var_pois = float(means.mean() + means.var(ddof=1))
        var_nb = float((means + means ** 2 / k_a0).mean() + means.var(ddof=1))
        return {
            "k_if_poisson_given_shed": k_from(float(means.mean()), var_pois),
            "k_if_a0_noise_given_shed": k_from(float(means.mean()), var_nb),
            "cv2_of_conditional_mean": cv2(means),
        }

    return {"proportional_lower_half": pack(proportional), "empirical_ventile_mean": pack(empirical), "observed_k": mom_k(off)}


def narrow_dispersion(arm, shed, off, k_a0):
    rows = []
    for lo, hi in NARROW_BINS:
        mask = (shed >= lo) & (shed < hi)
        n = int(mask.sum())
        if n < 30:
            continue
        mean_off = float(off[mask].mean())
        var_off = float(off[mask].var(ddof=1))
        rows.append({
            "arm": arm,
            "bin_lo": lo,
            "bin_hi": hi,
            "n": n,
            "mean_shed": float(shed[mask].mean()),
            "shed_cv2": cv2(shed[mask]),
            "mean_offspring": mean_off,
            "var_over_mean": var_off / mean_off if mean_off > 0 else None,
            "nb_a0_var_over_mean": None if mean_off <= 0 or not np.isfinite(k_a0) else 1.0 + mean_off / k_a0,
        })
    return rows


def judge(summaries):
    for item in summaries:
        expected = item["expected_cv2"]
        pooled = item["cv2_pooled"]
        if expected <= 0 or pooled is None:
            item["cv2_direction"] = "none"
        elif pooled < expected * (1.0 - CV_TOL):
            item["cv2_direction"] = "below"
        elif pooled > expected * (1.0 + CV_TOL):
            item["cv2_direction"] = "above"
        else:
            item["cv2_direction"] = "match"
        ratio = item["top1_ratio"] or {}
        point = ratio.get("point")
        high = ratio.get("high")
        item["mean_ceiling"] = bool(
            item["expected_cv2"] >= 1.0
            and point is not None and point < RATIO_CEILING
            and high is not None and high < BOOT_HIGH_CEILING
            and item["elasticity"] is not None and item["elasticity"] < ELASTICITY_CEILING
        )
    section = [item for item in summaries if item["arm"] in SECTION3]
    section_match = all(item["multiplier_matches_draw"] for item in section)
    ceiling_arms = [item for item in summaries if item["expected_cv2"] >= 1.0]
    ceiling_ok = bool(ceiling_arms) and all(item["mean_ceiling"] for item in ceiling_arms)
    if section_match and ceiling_ok:
        verdict = "confirmed"
        claim = "Mean offspring of the top 1 percent flattens below the proportional line"
    else:
        verdict = "not_confirmed"
        claim = "Mean offspring keeps rising with the multiplier, so a contact ceiling is not confirmed"
    return {
        "verdict": verdict,
        "figure_claim": claim,
        "section3_multiplier_matches": section_match,
        "section3_arms": list(SECTION3),
        "mean_ceiling_on_every_wide_arm": ceiling_ok,
        "arms_below_expected_cv2": [item["arm"] for item in summaries if item["cv2_direction"] == "below"],
        "arms_above_expected_cv2": [item["arm"] for item in summaries if item["cv2_direction"] == "above"],
        "rule": {
            "cv2_relative_tolerance": CV_TOL,
            "top1_ratio_below": RATIO_CEILING,
            "top1_bootstrap_high_below": BOOT_HIGH_CEILING,
            "elasticity_below": ELASTICITY_CEILING,
            "ceiling_required_on_arms_with_expected_cv2_at_least": 1.0,
            "multiplier_claim_uses": "the six arms in note 18 section 3; other arms are reported beside that claim",
        },
    }


def main():
    rng = np.random.default_rng(18)
    summaries = []
    bins = []
    held = {}
    arrays = {}
    for arm, meta in ARMS.items():
        print(f"scan {arm}", flush=True)
        summary, arm_bins, arm_held, shed, off = analyse_arm(arm, meta, rng)
        summaries.append(summary)
        bins.extend(arm_bins)
        held[arm] = arm_held
        arrays[arm] = (shed, off)
        print(
            f"  n={summary['infectors']} cv2={summary['cv2_pooled']:.3f} expected={summary['expected_cv2']:.3f} "
            f"k={summary['early_k']:.3f} top1={None if not summary['top1_ratio'] else round(summary['top1_ratio']['point'], 3)}",
            flush=True,
        )
    by = {item["arm"]: item for item in summaries}
    published = json.loads((RESULTS / "ps1" / "summary.json").read_text())
    mismatches = []
    for item in published["arms"]:
        got = by[item["arm"]]["early_k"]
        ref = item["early_k"]
        if abs(got - ref) > max(0.02, 0.02 * ref):
            mismatches.append({"arm": item["arm"], "d1_k": got, "summary_k": ref})
    k_a0 = by["A0"]["early_k"]
    narrow = []
    for summary in summaries:
        shed, off = arrays[summary["arm"]]
        summary["marginal_models"] = marginal_models(shed, off, k_a0)
        narrow.extend(narrow_dispersion(summary["arm"], shed, off, k_a0))
        attach_top1_nulls(summary, held[summary["arm"]], k_a0, rng)
    bins = within_bin_dispersion(bins, k_a0)
    decision = judge(summaries)
    payload = {
        "question": "Does a contact ceiling truncate offspring at high multipliers, absorbing the model's own heterogeneity?",
        "window": {"first_day": 1, "last_day": 10, "horizon_days": 14},
        "a0_k": k_a0,
        "a0_residual_cv2": None if not k_a0 or not np.isfinite(k_a0) else 1.0 / k_a0,
        "decision": decision,
        "early_k_mismatches_against_ps1_summary": mismatches,
        "arms": summaries,
    }
    out_dir = RESULTS / "ps1"
    (out_dir / "d1_absorption.json").write_text(json.dumps(json_ready(payload), indent=1) + "\n")
    with (out_dir / "d1_bins.csv").open("w", newline="") as handle:
        fields = ["arm", "bin_type", "label", "bin_lo", "bin_hi", "n", "mean_shed", "mean_offspring",
                  "mean_locations", "var_offspring", "var_over_mean", "nb_a0_var_over_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in bins:
            writer.writerow({key: row.get(key) for key in fields})
    with (out_dir / "d1_narrow.csv").open("w", newline="") as handle:
        fields = ["arm", "bin_lo", "bin_hi", "n", "mean_shed", "shed_cv2", "mean_offspring", "var_over_mean", "nb_a0_var_over_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(narrow)
    with (out_dir / "d1_arms.csv").open("w", newline="") as handle:
        fields = ["arm", "infectors", "expected_cv2", "cv2_pooled", "cv2_low", "cv2_high", "multiplier_matches_draw",
                  "early_R", "early_k", "residual_vs_expected_trait", "residual_vs_realised_shed",
                  "pearson_shed_offspring", "spearman_shed_offspring", "top1_ratio", "top1_low", "top1_high",
                  "elasticity", "mean_ceiling", "p99_shed", "max_shed", "half_saturation_shed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            ratio = item["top1_ratio"] or {}
            fit = item["saturation_fit"] or {}
            interval = item["cv2_replicate_t"] or {}
            writer.writerow({
                "arm": item["arm"],
                "infectors": item["infectors"],
                "expected_cv2": item["expected_cv2"],
                "cv2_pooled": item["cv2_pooled"],
                "cv2_low": interval.get("low"),
                "cv2_high": interval.get("high"),
                "multiplier_matches_draw": item["multiplier_matches_draw"],
                "early_R": item["early_R"],
                "early_k": item["early_k"],
                "residual_vs_expected_trait": item["residual_vs_expected_trait"],
                "residual_vs_realised_shed": item["residual_vs_realised_shed"],
                "pearson_shed_offspring": item["pearson_shed_offspring"],
                "spearman_shed_offspring": item["spearman_shed_offspring"],
                "top1_ratio": ratio.get("point"),
                "top1_low": ratio.get("low"),
                "top1_high": ratio.get("high"),
                "elasticity": item["elasticity"],
                "mean_ceiling": item["mean_ceiling"],
                "p99_shed": item["p99_shed"],
                "max_shed": item["max_shed"],
                "half_saturation_shed": fit.get("half_saturation_shed"),
            })
    print(json.dumps(json_ready(decision), indent=1))
    if mismatches:
        print("early k mismatches", json.dumps(json_ready(mismatches)))
        sys.exit(1)


if __name__ == "__main__":
    main()
