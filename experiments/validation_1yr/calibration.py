"""Is the simulation's run-to-run spread realistic? Calibration of the ensemble against real data.

For each day with a real observation, the real value is ranked within the N simulated runs
(PIT value). If the ensemble spread is right, the real value falls inside the central x% interval
about x% of the time. Reports empirical coverage at 50/80/90/95%, a 95% CI on each (block
bootstrap over 7-day blocks, because consecutive days are autocorrelated), the share of days the
real value is above/below the whole ensemble, and the PIT histogram (png).

Series: hospitalized (I5_h+I6_h vs korona_hun.xlsx, rescaled) and HCE (vs Surjan/EMMI HCW
infections, rescaled; only ~21 overlapping days, so very weak evidence).

NOTE the ensemble only contains stochastic variance. Real data adds reporting noise and model
misspecification, so under-coverage means "ensemble too narrow for reality", not necessarily
"stochasticity is mis-modelled".

Usage: python calibration.py [sim_dir] [start_date]
Defaults: ../hcw_interventions/results/baseline (same baseline config), 2020-09-23
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation.aggregate import target_per_run
from validation.load_real import load_real_xlsx
from validation.load_sim import load_sim_runs
from validation.normalize import rescale_real

here = Path(__file__).parent
sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "hcw_interventions" / "results" / "baseline"
start_date = sys.argv[2] if len(sys.argv) > 2 else "2020-09-23"
HUNGARY_POPULATION = 9_700_000
LEVELS = [0.5, 0.8, 0.9, 0.95]
rng = np.random.default_rng(0)

sim = load_sim_runs(sim_dir, start_date)
pop_cols = ["S", "E", "I1", "I2", "I3", "I4", "I5_h", "I6_h", "R_h", "R", "D1", "D2"]
sim_pop = sim[sim["day"] == 0].groupby("run")[pop_cols].sum().sum(axis=1).mean()

wide_h = target_per_run(sim, ["I5_h", "I6_h"], "level")
real_h = rescale_real(load_real_xlsx(here.parent.parent / "korona_hun.xlsx", start_date, len(wide_h),
                                     {"hospitalized": 19})["hospitalized"], sim_pop, HUNGARY_POPULATION)
series = {"hospitalized": (wide_h, real_h)}

hcw_csv = here / "data" / "hungarian_data" / "Gmail-Surjan" / "2022-01-18_eg.dolg_utf8.csv"
if hcw_csv.exists():
    raw = pd.read_csv(hcw_csv, sep=";", index_col=0, parse_dates=True)["1"].astype(float)
    series["HCE"] = (target_per_run(sim, ["HCE"], "level"),
                     rescale_real(raw, sim_pop, HUNGARY_POPULATION))


def block_bootstrap_ci(hit: np.ndarray, block: int = 7, n_boot: int = 2000):
    n = len(hit)
    starts = np.arange(0, n, block)
    est = []
    for _ in range(n_boot):
        pick = rng.choice(starts, len(starts))
        est.append(np.concatenate([hit[s:s + block] for s in pick]).mean())
    return np.percentile(est, [2.5, 97.5])


rows, pits = [], {}
for name, (wide, real) in series.items():
    real = real.reindex(wide.index)
    ok = real.notna()
    w, r = wide[ok], real[ok]
    n_runs = w.notna().sum(axis=1)
    # randomised PIT: fraction of runs below the real value, ties split uniformly
    below = (w.lt(r, axis=0)).sum(axis=1)
    equal = (w.eq(r, axis=0)).sum(axis=1)
    pit = (below + rng.uniform(size=len(r)) * equal) / n_runs
    pits[name] = pit
    for lv in LEVELS:
        lo = w.quantile((1 - lv) / 2, axis=1); hi = w.quantile(1 - (1 - lv) / 2, axis=1)
        hit = ((r >= lo) & (r <= hi)).to_numpy(float)
        ci_lo, ci_hi = block_bootstrap_ci(hit)
        rows.append({"series": name, "nominal": lv, "n_days": len(r), "coverage": hit.mean(),
                     "ci95_lo": ci_lo, "ci95_hi": ci_hi})
    rows.append({"series": name, "nominal": "above_all_runs", "n_days": len(r),
                 "coverage": float((r > w.max(axis=1)).mean())})
    rows.append({"series": name, "nominal": "below_all_runs", "n_days": len(r),
                 "coverage": float((r < w.min(axis=1)).mean())})

res = pd.DataFrame(rows)
out_dir = here / "outputs"; out_dir.mkdir(exist_ok=True)
res.to_csv(out_dir / "calibration.csv", index=False)
print(f"{wide_h.shape[1]} runs from {sim_dir}")
print(res.round(3).to_string(index=False))

fig, axes = plt.subplots(1, len(pits), figsize=(4.5 * len(pits), 3.5), squeeze=False)
for ax, (name, pit) in zip(axes[0], pits.items()):
    ax.hist(pit, bins=10, range=(0, 1), color="#2a78d6", alpha=0.8)
    ax.axhline(len(pit) / 10, color="k", ls="--", lw=1, label="perfectly calibrated")
    ax.set_title(f"PIT: {name} (n={len(pit)} days)"); ax.set_xlabel("rank of real value in ensemble")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(out_dir / "calibration_pit.png", dpi=150)
