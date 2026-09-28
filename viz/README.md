# panSim output viewer

Interactive line plots for the daily table panSim prints to the console. A run is one stdout capture. An ensemble is a directory of those files (typically about 20, one per stochastic seed). Each selected series is drawn as the ensemble mean with a transparent ±1 standard deviation band.

The table is tab-separated. `MUT`, `IMM` and `INFV` are comma-separated per-variant lists inside one field; the viewer expands them to `MUT1`…, `IMM1`… and `INFV1`…. Column meanings follow `matlab/Readme/Compute_beta.txt`.

## Run an ensemble

`panSim` prints one statistics row per simulated day. Save stdout, and do not interleave several runs into one file:

```bash
mkdir -p viz/runs/baseline
./build_gpu/panSim -r --quarantinePolicy 3 -k 0.00041 \
  --progression inputConfigFiles/progressions_Jun17_tune/transition_config.json \
  -A inputConfigFiles/agentTypes_3.json \
  -a inputRealExample/agents1.json \
  -l inputRealExample/locations0.json \
  --infectiousnessMultiplier 0.98,1.85,1.98,2.42,4.10,4.55,6.8 \
  --diseaseProgressionScaling 0.94,1.03,0.84,0.72,0.57,0.463,0.45 \
  --closures inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json \
  > viz/runs/baseline/run_01.stdout
```

Repeat with the GPU free between runs. `--seed` fixes a run; omitting it keeps the stochastic spread the bands are for. The default length is 12 weeks (`-w`). Day 0 is 2020-09-23 (`--startDate 267`, the 23 September used in the seasonality code and in `matlab/Study1/simout2table.m`).

`viz/runs/baseline/` holds six stdout captures of the same command run earlier with `--quarantinePolicy 0` (84 days each, the default 12 weeks). The example and `viz/run_ensemble.sh` now pass `--quarantinePolicy 3`.

An 8-week ensemble is `viz/run_ensemble.sh` (defaults: `-w 8 -n 6 -o viz/runs/weeks8`). The captures from that script are in `viz/runs/weeks8/`. Compare both with `--group baseline=viz/runs/baseline --group weeks8=viz/runs/weeks8`.

`viz/runs/cmp_head/` and `viz/runs/cmp_main/` are twenty 8-week runs each of this branch and git `main`, same command, used to compare `I5_h+I6_h+R_h`.

`viz/runs/hosp_jan2021/` is eight 19-week runs (`-w 19`) that fit official hospital occupancy through 2021-01-31. The same command is used, with only the first multiplier changed:

```bash
--infectiousnessMultiplier 0.82,1.81,2.11,2.58,4.32,6.8,6.8
--diseaseProgressionScaling 0.88,1.03,0.813,0.72,0.57,0.463,0.45
```

That pass scored national-scale `I5_h+I6_h` only (`9_600_000 / N_sim` against `Kórházi ápoltak száma`). RMSE was 643. The mean was about 1100–1500 low from 4–18 November.

A later pass scores hospital occupancy as `I5_h+I6_h+R_h`, scaled by the reconstruction population `9_967_304 / N_sim`, and uses the `Infected` column of `Full_reconstruction_2026-09-26.xlsx` as a secondary check against `E+I1+I2+I3+I4+I5_h+I6_h` on the same scale. Raising `-k` from 0.00041 to 0.000418 closes most of the early–mid November hospital gap. Eight runs of that setting are in `viz/runs/hosp_k418/`:

```bash
-k 0.000418
--infectiousnessMultiplier 0.81,1.81,2.11,2.58,4.32,6.8,6.8
--diseaseProgressionScaling 0.90,1.03,0.813,0.72,0.57,0.463,0.45
```

That higher `-k` matches 4–18 November and then overshoots December (hospital RMSE 1273, mean squared error 1,619,389). The minimum mean squared error against hospital counts is the eight-run ensemble in `viz/runs/hosp_k41/`, same multipliers at `-k 0.00041`. Hospital mean squared error is 304,931 (RMSE 552, correlation 0.978): peak 8420 on 3 Dec versus 8045 on 8 Dec, and 3973 versus 3562 on 31 Jan. Against the reconstruction `Infected` column the same mean has mean squared error 482,888,634 (RMSE 21,975, correlation 0.890). The infected peak is 164,903 on 21 Nov versus 166,157 on 10 Nov, and 31 Jan is 92,614 versus 64,535. Nearby infectiousness values 0.80–0.812 and progression values 0.89–0.91 had higher hospital mean squared error once repeated. Those runs all used `--quarantinePolicy 0`.

With `--quarantinePolicy 3` the hospital mean squared error is lowest back at the original multipliers. Eight runs are in `viz/runs/hosp_q3/`:

```bash
--quarantinePolicy 3
-k 0.00041
--infectiousnessMultiplier 0.98,1.81,2.11,2.58,4.32,6.8,6.8
--diseaseProgressionScaling 0.94,1.03,0.813,0.72,0.57,0.463,0.45
```

Hospital mean squared error is 285,121 (RMSE 534, correlation 0.977). The mean peaks at 8741 on 30 Nov versus 8045 on 8 Dec, and is 3227 versus 3562 on 31 Jan. Against reconstruction `Infected` the mean squared error is 243,168,234 (RMSE 15,594, correlation 0.956): peak 155,256 on 18 Nov versus 166,157 on 10 Nov, and 72,972 versus 64,535 on 31 Jan. Progression 0.92 instead of 0.94 raises the hospital mean squared error to 332,893 and lowers the infected RMSE to 12,650.

The next hospital wave, 26 Jan through 2 Jun 2021, is the second strain. It is seeded by the first `ExposeToMutation` in the closure file: variant 1, daily fraction `0.00024`, for 6 days. The second progression value stays `1.03`. Seeding that variant on day 125 (26 Jan) at infectiousness `1.72` put the hospital peak on 23 Mar, a week before the official 30 Mar peak (hospital mean squared error 999,962). The closure file now starts the same seeding on day 132 (2 Feb), and the second infectiousness is `1.85`. Eight runs of that setting are in `viz/runs/wave2/`. Hospital mean squared error on the window is 246,795 (RMSE 497, correlation 0.992). The ensemble mean is highest at 12,839 on 30 Mar versus 12,553 that day, and stays within 2% of that height from 26 Mar through 1 Apr. Separate runs peak on day 186±4. On 2 Jun the mean is 1,039 versus 837. Reconstruction `Infected` has mean squared error 1,559,048,162 (RMSE 39,485, correlation 0.926): peak 236,593 on 20 Mar versus 220,698 on 19 Mar, with a slow decline (19,830 versus 4,337 on 2 Jun). Starting the seeding on day 134 or later, or raising the daily fraction, either missed the hospital peak or placed it after the 8 March closures ended. By the hospital peak about 96% of infections are variant 1 (`MUT1`).

The delta wave is variant 2, scored from 5 Aug 2021 through 11 Jan 2022. By 11 Jan variant 3 (BA.1) is about 73% of infections. The second `ExposeToMutation` starts on day 339 (28 Aug), daily fraction `0.00035`, for 6 days. Infectiousness is `1.98` and progression is `0.84`. Five runs of that setting are in `viz/runs/wave3/`. The search (`viz/optimize_delta.py`) varied the seed day, the daily fraction, and those two multipliers. Progression near 1.2 or higher, which would have been needed to cut prevalence down to the reconstruction, kept the hospital wave from turning inside the window. Settings that do turn down still peak near 450,000–520,000 infected.

The five-run mean peaks at 7,361 hospital on 1 Dec versus 7,596 on 30 Nov (hospital mean squared error 85,483, RMSE 292, correlation 0.995). On 25 Nov it is 6,691 versus 6,858, on 30 Nov 7,349 versus 7,596, and on 11 Jan 3,056 versus 2,932. Reconstruction `Infected` has mean squared error 19,498,458,129 (RMSE 139,637, correlation 0.952): the crest is 516,605 on 26 Nov versus 261,654 on 25 Nov, and 11 Jan is 340,597 versus 256,529. At the hospital peak every infection is variant 2.

BA.1 and BA.2 are variants 3 and 4, scored from 12 Jan through 31 Aug 2022. BA.1 is seeded on day 431 (28 Nov 2021), daily fraction `0.00025`, for 6 days, with infectiousness `2.42` and progression `0.72`. BA.2 is seeded on day 474 (10 Jan 2022), daily fraction `0.00020`, for 6 days, with infectiousness `4.10` and progression `0.57`. The summer rise is variant 5, seeded from day 576 (22 Apr) at daily fraction `0.00010`, infectiousness `4.55` and progression `0.463`. Five runs are in `viz/runs/ba/`. `viz/run_ba.py` is the trial runner used for that search.

Hospital mean squared error on the window is 287,837 (RMSE 537, correlation 0.956). The mean crests at 5,297 on 7 Feb, against 4,919 that day; the official crest is 5,291 on 15 Feb, when the mean is 4,907. On 16 Mar it is 1,941 versus 2,259, on 15 Jun 56 versus 232, on 10 Aug 625 versus 1,501, and on 31 Aug 1,204 versus 978. On 15 Feb about 65% of infections are BA.1 and 34% are BA.2; on 16 Mar about 90% are BA.2; from August they are variant 5. Reconstruction `Infected` has mean squared error 74,619,259,356 (RMSE 273,165, correlation 0.839): the crest is 1,190,505 on 5 Feb versus 522,752 on 4 Feb, and 31 Aug is 353,779 versus 302,324. The summer hospital wave is still rising at the end of August, later than the official early-August bump.

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
