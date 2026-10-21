"""Run-to-run (stochastic) uncertainty of the HCW outcomes, from outputs/per_run.csv.

Writes to outputs/:
  uncertainty_ci.csv           per scenario x metric: mean, std, 95% bootstrap CI of the mean,
                               5-95% spread across runs
  uncertainty_effects.csv      scenario vs baseline: bootstrap 95% CI of the difference in means
                               and of the ratio of means (CI excluding 0 / 1 = effect beyond noise)
  uncertainty_convergence.csv  baseline: CI half-width vs number of runs
  uncertainty_convergence.png  the same, as % of the mean

Runs are treated as independent draws (unpaired). Per-run scalars are used, not daily values,
because days within a run are autocorrelated.

Usage: python uncertainty.py [per_run.csv] [out_dir]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = Path(__file__).parent
per_run_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "outputs" / "per_run.csv"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "outputs"

METRICS = ["hcw_infections", "peak_hci", "mean_hci", "mean_hce", "incidence_RR"]
N_BOOT = 10_000
rng = np.random.default_rng(0)

runs = pd.read_csv(per_run_csv)


def boot_means(x: np.ndarray) -> np.ndarray:
    return x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)


def ci(b: np.ndarray) -> tuple:
    return tuple(np.percentile(b, [2.5, 97.5]))


rows = []
for metric in METRICS:
    for scen, g in runs.groupby("scenario"):
        x = g[metric].to_numpy(float)
        lo, hi = ci(boot_means(x))
        rows.append({"scenario": scen, "metric": metric, "n_runs": len(x), "mean": x.mean(),
                     "std": x.std(ddof=1), "cv": x.std(ddof=1) / x.mean() if x.mean() else np.nan,
                     "mean_ci95_lo": lo, "mean_ci95_hi": hi,
                     "run_q05": np.percentile(x, 5), "run_q95": np.percentile(x, 95)})
pd.DataFrame(rows).to_csv(out / "uncertainty_ci.csv", index=False)

eff = []
for metric in METRICS:
    base = runs.loc[runs.scenario == "baseline", metric].to_numpy(float)
    bb = boot_means(base)
    for scen in sorted(set(runs.scenario) - {"baseline"}):
        x = runs.loc[runs.scenario == scen, metric].to_numpy(float)
        bs = boot_means(x)
        d_lo, d_hi = ci(bs - bb)
        r_lo, r_hi = ci(bs / bb)
        eff.append({"scenario": scen, "metric": metric, "mean_baseline": base.mean(),
                    "mean_scenario": x.mean(), "diff": x.mean() - base.mean(),
                    "diff_ci95_lo": d_lo, "diff_ci95_hi": d_hi,
                    "ratio": x.mean() / base.mean(), "ratio_ci95_lo": r_lo, "ratio_ci95_hi": r_hi,
                    "beyond_noise": bool(d_lo > 0 or d_hi < 0)})
eff = pd.DataFrame(eff)
eff.to_csv(out / "uncertainty_effects.csv", index=False)

# Convergence: is 20 runs enough? CI half-width of the baseline mean using a random subset of n runs.
conv = []
for metric in METRICS:
    base = runs.loc[runs.scenario == "baseline", metric].to_numpy(float)
    for n in range(3, len(base) + 1):
        halfw = []
        for _ in range(300):
            sub = rng.choice(base, n, replace=False)
            halfw.append(1.96 * sub.std(ddof=1) / np.sqrt(n))
        conv.append({"metric": metric, "n_runs": n, "ci95_halfwidth": np.mean(halfw),
                     "halfwidth_pct_of_mean": 100 * np.mean(halfw) / base.mean()})
conv = pd.DataFrame(conv)
conv.to_csv(out / "uncertainty_convergence.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
for metric, g in conv.groupby("metric"):
    ax.plot(g.n_runs, g.halfwidth_pct_of_mean, label=metric)
ax.set_xlabel("number of runs"); ax.set_ylabel("95% CI half-width of baseline mean [% of mean]")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(out / "uncertainty_convergence.png", dpi=150)

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).query("scenario == 'baseline'").round(3).to_string(index=False))
print("\nscenario effects on hcw_infections (vs baseline):")
print(eff[eff.metric == "hcw_infections"].round(3).to_string(index=False))
print("\nbaseline CI half-width at n=5/10/20 (% of mean):")
print(conv[conv.n_runs.isin([5, 10, 20])].pivot(index="metric", columns="n_runs",
      values="halfwidth_pct_of_mean").round(1).to_string())
