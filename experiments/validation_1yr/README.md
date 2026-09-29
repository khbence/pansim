# 1-year validation against real data

1. **Simulate** (HPDC, after building `build/panSim`, see `../../SETUP_HPC.md`):
   `./run_validation.sh` - 20 runs x 52 weeks, default parameters, into `results/sim/`.
2. **Aggregate** (any machine with the results copied over): `python aggregate_compartments.py`
3. **Plot + validate**: `python plot_key_metrics.py`

`aggregate_compartments.py [sim_dir] [start_date] [out_csv]` averages *every* scalar compartment
over the runs in `results/sim/`, per simulated day (mean/std/5%/95%/`n_runs`), into
`outputs/aggregated_compartments.csv`. It rebuilds `all_runs.tsv` from `run_XX.tsv` if missing
(e.g. because a run crashed before `run_20_sims.sh` reached its combine step - check `n_runs` in
the output for exactly which days that affects). It also adds a derived `hospitalized_covid`
column (`I5_h + I6_h`) since the printed `H` column is non-COVID hospitalisation (see below).

`plot_key_metrics.py [aggregated_csv] [out_png] [real_xlsx] [start_date]` plots
`hospitalized_covid`, `HCI` and `HCE` (mean + 5-95% band per panel) from that CSV into
`outputs/key_metrics.png`. The hospitalized panel also overlays the real Hungarian series read
directly from `../../korona_hun.xlsx` (sheet `koronahun`, column 19, "Korhazi apoltak szama"),
rescaled from all-of-Hungary to the simulated population - and writes agreement metrics (RMSE,
Pearson r, peak-day offset, 5-95% band coverage) to `outputs/validation_metrics.json`.
**HCI/HCE have no real-data overlay**: `korona_hun.xlsx` (checked all 8 sheets) has no
healthcare-worker-specific series to compare against, so those two panels are simulation-only.

Needs `pandas` and `matplotlib` (`pip install pandas matplotlib`).

## Things to check
- **Compartment mapping**: hospitalized = I5_h+I6_h is an interpretation - please review. The
  printed `H` column is non-COVID hospitalisation and is not used.
- **Population:** real counts are multiplied by sim_pop / 9.7M. The default `agents.json` has
  only 22,364 agents, so simulated counts are small and noisy; `agents1.json` has 184,745.
- **Start date:** panSim default `--startDate 267` is 24 Sep (Jan 1 = 0, 2020); this experiment
  labels day 0 as 2020-09-23 (Wednesday, matching `--startDay 2`). Possible one-day offset.
- **Variation between runs:** panSim has no seed flag; check that the 20 runs actually differ.
- `korona_hun.xlsx`'s `koronahun` sheet has blank cells on some days (e.g. weekends); those stay
  NaN and are dropped before computing agreement metrics.

## History
This experiment previously also had a broader per-target pipeline (`validate.py` + `config.yaml`,
comparing cases/deaths/active-infected/hospitalized against a `korona_hun(koronahun).csv` export)
producing `aggregated.csv` + `metrics.csv` + one PNG per target. It was removed for being unused
and out of date (the `.csv` export no longer exists, only `korona_hun.xlsx`; `config.yaml`'s
`population_columns` also had a stale `I5h`/`I6h` typo for the actual `I5_h`/`I6_h` columns) -
see git history if you want to revive it.
