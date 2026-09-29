"""Per-run HCW metrics, and significance testing of each scenario against baseline.

HCI = % of HCWs infected or quarantined on that day (prevalence, integer-truncated).
HCE = HCWs newly infected in the last 24h (daily incidence).
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def per_run(df: pd.DataFrame, hci_threshold: float = 5.0) -> pd.DataFrame:
    g = df.groupby(["scenario", "run"])
    out = g.agg(
        peak_hci=("HCI", "max"),
        peak_day=("HCI", "idxmax"),
        hcw_infections=("HCE", "sum"),
        days_above_thr=("HCI", lambda s: int((s > hci_threshold).sum())),
        mean_hci=("HCI", "mean"),
        mean_hce=("HCE", "mean"),
        n_days=("HCI", "size"),  # surfaces a short/crashed run directly (e.g. < 364)
    )
    out["peak_day"] = df.loc[out["peak_day"], "day"].to_numpy()
    # Run-level "incidence per unit of average prevalence": total new HCW infections over the
    # run, divided by that run's mean HCI. NOT a day-by-day HCE_t/HCI_t ratio - HCI is an
    # integer-truncated percentage that's frequently exactly 0 while HCE_t>0, which would make a
    # per-day ratio undefined/blow up on most days. mean_hci below 1e-9 (near-zero prevalence
    # throughout the run) is treated as undefined (NaN) rather than dividing by ~0.
    out["hce_hci_ratio"] = out["hcw_infections"] / out["mean_hci"].where(out["mean_hci"] > 1e-9)
    return out.reset_index()


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["peak_hci", "peak_day", "hcw_infections", "days_above_thr", "mean_hci", "mean_hce",
            "hce_hci_ratio"]
    s = runs.groupby("scenario")[cols].agg(["mean", "std"])
    base = runs[runs.scenario == "baseline"][cols].mean()
    for c in cols:
        s[(c, "delta_vs_baseline")] = s[(c, "mean")] - base[c]
    return s.sort_index(axis=1)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """BH-FDR correction. NaN in, NaN out; family size = count of non-NaN p-values."""
    valid = pvalues.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=pvalues.index)
    order = valid.sort_values().index
    ranks = np.arange(1, len(order) + 1)
    n = len(order)
    q_sorted = (valid.loc[order].to_numpy() * n) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]  # enforce monotonicity
    q_sorted = np.clip(q_sorted, 0, 1)
    q = pd.Series(q_sorted, index=order).reindex(pvalues.index)
    return q


def mann_whitney_vs_baseline(runs: pd.DataFrame, metric: str, baseline_name: str = "baseline") -> pd.DataFrame:
    """Two-sided Mann-Whitney U of `metric`'s per-run values, every scenario vs baseline.

    effect_size_rank_biserial: 2*U/(n_scenario*n_baseline) - 1, where U is scipy's statistic for
    the scenario sample (the first argument to mannwhitneyu); +1 means every scenario run outranks
    every baseline run (scenario stochastically larger), -1 the reverse. Sign verified empirically
    against a synthetic pair with an obviously-smaller group (see experiments/hcw_interventions
    development notes) - scipy's U for the first argument shrinks toward 0 when that group is
    stochastically smaller, so the naive `1 - 2U/(n1*n2)` form has the opposite sign of what
    "positive = scenario larger" requires. n_baseline/n_scenario are counts AFTER dropping NaN
    (e.g. a scenario where hce_hci_ratio is undefined in every run), so a degenerate comparison is
    visible in the output rather than silently wrong.
    """
    base_vals = runs.loc[runs.scenario == baseline_name, metric].dropna().to_numpy()
    rows = []
    for name, d in runs.groupby("scenario"):
        if name == baseline_name:
            continue
        scen_vals = d[metric].dropna().to_numpy()
        n_b, n_s = len(base_vals), len(scen_vals)
        if n_b < 2 or n_s < 2:
            rows.append({"metric": metric, "scenario": name, "n_baseline": n_b, "n_scenario": n_s,
                         "median_baseline": np.median(base_vals) if n_b else np.nan,
                         "median_scenario": np.median(scen_vals) if n_s else np.nan,
                         "median_diff": np.nan, "U_statistic": np.nan, "p_value": np.nan,
                         "effect_size_rank_biserial": np.nan})
            continue
        U, p = mannwhitneyu(scen_vals, base_vals, alternative="two-sided")
        rows.append({
            "metric": metric, "scenario": name, "n_baseline": n_b, "n_scenario": n_s,
            "median_baseline": np.median(base_vals), "median_scenario": np.median(scen_vals),
            "median_diff": np.median(scen_vals) - np.median(base_vals),
            "U_statistic": U, "p_value": p,
            "effect_size_rank_biserial": (2 * U) / (n_s * n_b) - 1,
        })
    return pd.DataFrame(rows)


def significance_table(runs: pd.DataFrame, metrics=None, baseline_name: str = "baseline") -> pd.DataFrame:
    """Mann-Whitney U + BH-FDR of each scenario vs baseline, for each metric.

    q_within_metric: BH-FDR applied within each metric's scenario comparisons (the primary
    column - "which scenarios differ significantly in this metric"). q_global: BH-FDR across
    every row at once (stricter, supplementary). significant_q05: q_within_metric < 0.05.
    """
    if metrics is None:
        metrics = ["mean_hci", "mean_hce", "hce_hci_ratio", "peak_hci", "hcw_infections",
                   "days_above_thr"]
    tables = [mann_whitney_vs_baseline(runs, m, baseline_name) for m in metrics]
    out = pd.concat(tables, ignore_index=True)
    out["q_within_metric"] = out.groupby("metric")["p_value"].transform(
        lambda p: benjamini_hochberg(p))
    out["q_global"] = benjamini_hochberg(out["p_value"])
    out["significant_q05"] = out["q_within_metric"] < 0.05
    return out
