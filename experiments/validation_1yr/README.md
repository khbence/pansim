# 1-year validation against real data

1. **Simulate** (HPDC, after building `build/panSim`, see `../../SETUP_HPC.md`):
   `./run_validation.sh` - 20 runs x 52 weeks, default parameters, into `results/sim/`.
2. **Validate** (any machine with the results copied over):
   `pip install -r requirements.txt && python validate.py`

`validate.py` builds each target from `config.yaml` (sum of simulator compartments), takes the
mean over the 20 runs per day (plus std and 5/95% quantiles), and compares it with the real
series from `../../korona_hun(koronahun).csv` from **2020-09-23** onwards. Outputs go to
`outputs/`: `aggregated.csv`, `metrics.csv` (RMSE, MAE, Pearson r, peak ratio, peak-day
offset, coverage of the real data by the 5-95% band, and the yearly mean of the aggregated
series as the single summary number), and one PNG per target.

## Things to check
- **Compartment mapping** in `config.yaml` is my interpretation (e.g. hospitalized = I5h+I6h,
  deaths = D1+D2). The printed `H` column is non-COVID hospitalisation and is not used.
- **Population:** real counts are multiplied by sim_pop / 9.7M. The default `agents.json` has
  only 22,364 agents, so simulated counts are small and noisy; `agents1.json` has 184,745.
- **Detected vs. true cases:** real cases are detections, the simulator's NI is all infections,
  so peak-normalized shape metrics are more meaningful than absolute ones for cases/active.
- **Start date:** panSim default `--startDate 267` is 24 Sep (Jan 1 = 0, 2020); this experiment
  labels day 0 as 2020-09-23 (Wednesday, matching `--startDay 2`). Possible one-day offset.
- **Variation between runs:** panSim has no seed flag; check that the 20 runs actually differ.
- The real csv is cp1250, `;`-separated, comma-decimal, with blank days; `load_real.py` handles
  this and selects columns by position, not by (Hungarian, repeated) header names.
