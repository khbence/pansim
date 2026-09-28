"""results/sim/all_runs.tsv + korona_hun csv -> outputs/{aggregated.csv,metrics.csv,<target>.png}

Usage: python validate.py [config.yaml]
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

from validation.aggregate import aggregate_runs, scalar_summary, target_per_run
from validation.load_real import load_real
from validation.load_sim import load_sim, require_columns
from validation.metrics import compare
from validation.normalize import real_transform, rescale_real
from validation.plots import plot_target

here = Path(__file__).parent
cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "config.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
base = cfg_path.parent

start, n_days = cfg["start_date"], cfg["n_days"]
out = here / "outputs"
out.mkdir(exist_ok=True)

sim = load_sim(base / cfg["sim_tsv"], start, n_days)
targets = cfg["targets"]
for t in targets.values():
    require_columns(sim, t["sim"])
require_columns(sim, cfg["population_columns"])

real = load_real(
    (base / cfg["real_csv"]).resolve(),
    start,
    n_days,
    {name: t["real_col"] for name, t in targets.items()},
    encoding=cfg["real_encoding"],
    sep=cfg["real_sep"],
)

sim_pop = cfg["sim_population"]
if sim_pop is None:
    day0 = sim[sim["day"] == 0]
    sim_pop = float(day0[cfg["population_columns"]].sum(axis=1).mean())
print(f"simulated population: {sim_pop:.0f}")

aggregated, metrics, scalars = [], {}, {}
for name, t in targets.items():
    wide = target_per_run(sim, t["sim"], t["transform"])
    agg = aggregate_runs(wide)
    r = real_transform(real[name], t["transform"])
    if cfg["rescale_real"]:
        r = rescale_real(r, sim_pop, cfg["hungary_population"])

    metrics[name] = compare(agg, r)
    scalars[name] = scalar_summary(agg)
    aggregated.append(agg.add_prefix(f"{name}_").join(r.rename(f"{name}_real")))
    plot_target(name, agg, r, out / f"{name}.png", start)

pd.concat(aggregated, axis=1).to_csv(out / "aggregated.csv", index_label="date")
m = pd.DataFrame(metrics).T
m["sim_mean_over_year"] = pd.Series(scalars)
m.to_csv(out / "metrics.csv", index_label="target")
print(m.to_string())
