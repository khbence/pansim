# panSim output viewer

Interactive line plots for the daily table panSim prints to the console. A run is one stdout capture. An ensemble is a directory of those files (typically about 20, one per stochastic seed). Each selected series is drawn as the ensemble mean with a transparent ±1 standard deviation band.

The table is tab-separated. `MUT`, `IMM` and `INFV` are comma-separated per-variant lists inside one field; the viewer expands them to `MUT1`…, `IMM1`… and `INFV1`…. Column meanings follow `matlab/Readme/Compute_beta.txt`.

## Run an ensemble

`panSim` prints one statistics row per simulated day. Save stdout, and do not interleave several runs into one file:

```bash
mkdir -p viz/runs/baseline
./build_gpu/panSim -r --quarantinePolicy 0 -k 0.00041 \
  --progression inputConfigFiles/progressions_Jun17_tune/transition_config.json \
  -A inputConfigFiles/agentTypes_3.json \
  -a inputRealExample/agents1.json \
  -l inputRealExample/locations0.json \
  --infectiousnessMultiplier 0.98,1.81,2.11,2.58,4.32,6.8,6.8 \
  --diseaseProgressionScaling 0.94,1.03,0.813,0.72,0.57,0.463,0.45 \
  --closures inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json \
  > viz/runs/baseline/run_01.stdout
```

Repeat with the GPU free between runs. `--seed` fixes a run; omitting it keeps the stochastic spread the bands are for. The default length is 12 weeks (`-w`). Day 0 is 2020-09-23 (`--startDate 267`, the 23 September used in the seasonality code and in `matlab/Study1/simout2table.m`).

`viz/runs/baseline/` already contains six stdout captures from that command (84 days each, the default 12 weeks).

An 8-week ensemble is `viz/run_ensemble.sh` (defaults: `-w 8 -n 6 -o viz/runs/weeks8`). The captures from that script are in `viz/runs/weeks8/`. Compare both with `--group baseline=viz/runs/baseline --group weeks8=viz/runs/weeks8`.

## View

```bash
python3 -m venv viz/.venv
viz/.venv/bin/pip install -r viz/requirements.txt
PYTHONPATH=viz viz/.venv/bin/python -m pansimviz \
  --group baseline=viz/runs/baseline \
  --group alternative=viz/runs/alternative \
  --ground-truth korona_hun.xlsx
```

Open http://127.0.0.1:8050. The page can reload paths without a restart.

Layout controls:

- plots per row, add and remove panels, metric and title on each panel
- trailing average (default 7 days), applied to each run before the mean and standard deviation
- individual runs, standard-deviation band, Hungary overlay
- scale: simulated agents (default), per 100 000, or full national population
- save the layout as JSON and pass it back with `--layout`

## Hungary ground truth

`korona_hun.xlsx` (sheet `koronahun`) starts on 2020-03-04 and ends on 2022-12-28. The simulation starts later, so only the overlapping dates are drawn. Empty cells stay gaps; the national line is not interpolated across them.

Counts are for about 9.6 million people. On the simulation scale they are multiplied by `N_sim / 9_600_000`. `N_sim` is the median of `S+E+I1+I2+I3+I4+I5_h+I6_h+R_h+R+D1+D2`. With `agents1.json` that sum is about 178 750, a little under the 179 500 agents in the file; type 179500 in Sim population to force the file size. Percentages, beta and the positive rate are not rescaled. The overlay is the reported national series, not a reconstructed epidemic:

| Plot | Workbook column |
| --- | --- |
| NI | Új esetek száma |
| I_all | Aktív fertőzöttek száma |
| H_covid | Kórházi ápoltak száma |
| I6_h | Lélegeztetőgépen lévők száma |
| D1 | Elhunytak |
| D_new | Az új elhunytak száma naponta |
| R_all / R_new | Gyógyultak / Új gyógyultak naponta |
| Q | Hatósági házi karantén |
| T | Új mintavételek száma |
| VAC / VAC_cum | Új beoltottak / Beoltottak száma |
| INF | Regisztrált esetek száma |
| pos_rate | Pozitív tesztek aránya |

Reported cases and hospital counts are not the same measurement as the simulator compartments; the overlay is there so the shape and timing can be compared after population scaling.
