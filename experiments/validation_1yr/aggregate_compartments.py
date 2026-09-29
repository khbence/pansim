"""Average every PanSim compartment over the N runs in results/sim, per simulated day.

Reads run_XX.tsv directly (rebuilding all_runs.tsv alongside, in case it's missing), then for
every scalar compartment column writes mean/std/q05/q95/n_runs per day to one wide CSV.

List-valued columns (MUT, IMM, INFV: comma-separated per-variant/age-group breakdowns) are
skipped - see validation/load_sim.py.

A short/crashed run contributes to the average only on the days it actually reached (see
validation/load_sim.py:load_sim_runs); n_runs in the output shows how many runs covered each day.

Usage: python aggregate_compartments.py [sim_dir] [start_date] [out_csv]
Defaults: results/sim, 2020-09-23 (see config.yaml), outputs/aggregated_compartments.csv
"""
import sys
from pathlib import Path

import pandas as pd

from validation.aggregate import aggregate_runs, target_per_run
from validation.load_sim import load_sim_runs

here = Path(__file__).parent
sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results" / "sim"
start_date = sys.argv[2] if len(sys.argv) > 2 else "2020-09-23"
out_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else here / "outputs" / "aggregated_compartments.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

sim = load_sim_runs(sim_dir, start_date)

skip = {"run", "day", "date", "MUT", "IMM", "INFV"}
compartments = [c for c in sim.columns if c not in skip]

# Derived series worth having pre-aggregated alongside the raw columns.
derived = {"hospitalized_covid": ["I5_h", "I6_h"]}  # COVID-hospitalized; NOT the printed H column
                                                     # (H is non-COVID hospitalisation, see README)

pieces = {}
n_runs = None
for name in compartments:
    wide = target_per_run(sim, [name], "level")
    agg = aggregate_runs(wide)
    if n_runs is None:
        n_runs = wide.notna().sum(axis=1).rename("n_runs")
    pieces[name] = agg

for name, cols in derived.items():
    wide = target_per_run(sim, cols, "level")
    pieces[name] = aggregate_runs(wide)

out = pd.concat({name: df for name, df in pieces.items()}, axis=1)
out.columns = [f"{name}_{stat}" for name, stat in out.columns]
out.insert(0, "n_runs", n_runs)
out.insert(0, "day", range(len(out)))
out.to_csv(out_csv, index_label="date")

print(f"{len(compartments)} compartments + {len(derived)} derived series, "
      f"{len(out)} days, {int(n_runs.max())} runs -> {out_csv}")
if (n_runs < n_runs.max()).any():
    last_full = n_runs[n_runs == n_runs.max()].index.max()
    print(f"NOTE: fewer than {int(n_runs.max())} runs contribute after {last_full.date()} "
          f"(n_runs column shows exactly how many per day)")
