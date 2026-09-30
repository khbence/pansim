"""results/<scenario>/all_runs.tsv -> outputs/{per_run,summary,significance,relative_risk_table}.csv,
hci.png, hce.png, vulnerability.png

Usage: python analyze.py [results_dir] [out_dir]
Defaults: results/, outputs/ (relative to this file).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from hcw import load, metrics, plots

here = Path(__file__).parent
results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "outputs"
out.mkdir(parents=True, exist_ok=True)

df = load.load_all(results_dir)
runs = metrics.per_run(df)
runs.to_csv(out / "per_run.csv", index=False)
metrics.summarize(runs).to_csv(out / "summary.csv")

sig = metrics.significance_table(runs)
sig.to_csv(out / "significance.csv", index=False)
n_sig = int(sig["significant_q05"].sum())
print(f"{n_sig}/{len(sig)} scenario x metric comparisons significant at q_within_metric < 0.05")
if n_sig:
    print(sig[sig.significant_q05][
        ["scenario", "metric", "median_diff", "effect_size_rank_biserial", "q_within_metric"]
    ].to_string(index=False))

rr_table = metrics.relative_risk_table(runs, sig)
rr_table.to_csv(out / "relative_risk_table.csv", index=False)
print("\nrelative risk table (HCW vs rest of population):")
print(rr_table.round(4).to_string(index=False))

for col, title in [("HCI", "HCWs infected or quarantined [%]"), ("HCE", "new HCW infections per day")]:
    plt.figure(figsize=(8, 4))
    plots.band(df, col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out / f"{col.lower()}.png", dpi=150)
    plt.close()

# Is the HCW group a vulnerable subgroup? HCI vs general-population infection prevalence,
# per scenario - see hci_pi_infected_ratio in significance.csv for the statistical test.
df = metrics.add_population_columns(df)
fig = plots.hcw_vs_population(df)
fig.savefig(out / "vulnerability.png", dpi=150)
plt.close(fig)
