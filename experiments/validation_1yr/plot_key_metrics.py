"""Plot COVID-hospitalized, HCI and HCE from aggregate_compartments.py's output, with the real
Hungarian hospitalized series (korona_hun.xlsx) overlaid for comparison.

- Hospitalized = I5_h + I6_h (COVID-hospitalized compartments; NOT the printed H column, which
  is non-COVID hospitalisation - see README.md). Real: "Korhazi apoltak szama" (koronahun sheet,
  column 19), rescaled from all-of-Hungary to the simulated population (see config.yaml).
- HCI = % of healthcare workers currently infected or quarantined.
- HCE = healthcare workers newly exposed/infected in the preceding 24h (count).
  (see src/runtime/simulationOrchestration.cpp)
  korona_hun.xlsx has no healthcare-worker-specific series (checked all 8 sheets). HCE is
  compared against the daily new confirmed health-worker infections (Surjan/EMMI, eg.dolg.csv,
  column "1"; confidential, git-ignored), rescaled like the hospitalized series. That file starts
  2021-09-01, so only the last days of the 1-year run overlap. HCI has no real-data series.

Each panel shows the mean over runs with a 5-95% band; a run that ends early (crash/short run)
drops out of the average for the days after it ends, shaded past that point since fewer runs
back the average there.

Usage: python plot_key_metrics.py [aggregated_csv] [out_png] [real_xlsx] [start_date]
Defaults: outputs/aggregated_compartments.csv, outputs/key_metrics.png, ../../korona_hun.xlsx,
2020-09-23 (see config.yaml)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from validation.load_real import load_real_xlsx
from validation.metrics import compare
from validation.normalize import rescale_real

here = Path(__file__).parent
in_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "outputs" / "aggregated_compartments.csv"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "outputs" / "key_metrics.png"
real_xlsx = Path(sys.argv[3]) if len(sys.argv) > 3 else here.parent.parent / "korona_hun.xlsx"
start_date = sys.argv[4] if len(sys.argv) > 4 else "2020-09-23"

HUNGARY_POPULATION = 9_700_000  # config.yaml
HCW_CSV = here / "data" / "hungarian_data" / "Gmail-Surjan" / "2022-01-18_eg.dolg_utf8.csv"

df = pd.read_csv(in_csv, index_col=0, parse_dates=True)
n_full = int(df["n_runs"].max())

population_columns = ["S", "E", "I1", "I2", "I3", "I4", "I5_h", "I6_h", "R_h", "R", "D1", "D2"]
sim_population = df.loc[df.index[0], [f"{c}_mean" for c in population_columns]].sum()
print(f"simulated population: {sim_population:.0f}")

real = load_real_xlsx(real_xlsx, start_date, len(df), {"hospitalized": 19})["hospitalized"]
real_rescaled = rescale_real(real, sim_population, HUNGARY_POPULATION)

m = compare(df[["hospitalized_covid_mean", "hospitalized_covid_q05", "hospitalized_covid_q95"]]
            .rename(columns=lambda c: c.replace("hospitalized_covid_", "")), real_rescaled)
print("hospitalized vs. real (rescaled):", m)

# Real new confirmed HCW infections (national, column "1" = health workers), same rescaling.
hcw_real = None
if HCW_CSV.exists():
    raw = pd.read_csv(HCW_CSV, sep=";", index_col=0, parse_dates=True)
    hcw_real = rescale_real(raw["1"].astype(float), sim_population, HUNGARY_POPULATION)
    hcw_real = hcw_real.reindex(df.index)  # NaN outside the overlap
    m_hce = compare(df[["HCE_mean", "HCE_q05", "HCE_q95"]].rename(columns=lambda c: c.replace("HCE_", "")),
                    hcw_real)
    print("HCE vs. real HCW infections (rescaled):", m_hce)
else:
    m_hce = None
    print(f"no HCW data at {HCW_CSV}; HCE panel is simulation-only")

metrics_json = out_png.with_name("validation_metrics.json")
metrics_json.write_text(json.dumps(
    {"hospitalized_covid": m, "HCE": m_hce, "sim_population": float(sim_population), "hungary_population": HUNGARY_POPULATION},
    indent=2))

# Validated categorical slots (dataviz skill palette): blue, orange, aqua.
panels = [
    ("hospitalized_covid", "Hospitalized (COVID: I5_h + I6_h)", "count", "#2a78d6"),
    ("HCI", "HCI - healthcare workers infected/quarantined", "%", "#eb6834"),
    ("HCE", "HCE - healthcare workers newly exposed (24h)", "count", "#1baf7a"),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(10, 9.5), sharex=True)
for ax, (prefix, title, ylabel, color) in zip(axes, panels):
    mean, q05, q95 = df[f"{prefix}_mean"], df[f"{prefix}_q05"], df[f"{prefix}_q95"]
    ax.fill_between(df.index, q05, q95, color=color, alpha=0.2, linewidth=0,
                     label=f"5-95% ({n_full} runs)")
    ax.plot(df.index, mean, color=color, linewidth=2, label="mean")

    short = df.index[df["n_runs"] < n_full]
    if len(short):
        ax.axvspan(short.min(), short.max(), color="grey", alpha=0.08,
                   label=f"< {n_full} runs averaged here")

    if prefix == "hospitalized_covid":
        ax.plot(real_rescaled.index, real_rescaled, "k.", ms=4,
                 label="real, rescaled (koronahun.xlsx)")
        ax.text(0.99, 0.97, f"RMSE={m['rmse']:.0f}  r={m['pearson_r']:.2f}  "
                f"peak offset={m['peak_day_offset']:+d}d  5-95% coverage={m['coverage_real_in_5_95_band']:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")
    elif prefix == "HCE" and hcw_real is not None:
        ax.plot(hcw_real.index, hcw_real, "k.", ms=4,
                 label="real HCW infections, rescaled (Surjan/EMMI)")
        if "rmse" in m_hce:
            ax.text(0.99, 0.97, f"overlap {m_hce['n_days']}d  RMSE={m_hce['rmse']:.2f}  "
                    f"mean sim/real={df['HCE_mean'][hcw_real.notna()].mean() / hcw_real.mean():.1f}x  "
                    f"5-95% coverage={m_hce['coverage_real_in_5_95_band']:.0%}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")
    else:
        ax.text(0.5, 0.5, "no public Hungarian data for this metric - simulation only",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.5",
                style="italic")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

axes[-1].set_xlabel("date")
axes[-1].tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"-> {out_png}")
