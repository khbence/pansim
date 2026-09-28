# HCW interventions experiment

Question: how do interventions change healthcare-worker (HCW) infection and absence?

- `HCI` column: % of HCWs infected or quarantined that day (prevalence).
- `HCE` column: HCWs newly infected in the last 24 h (incidence).
- HCW = agent with a work location of type hospital (12) or doctor (14)
  (`flagHealthcareWorkers` in `include/simulation.h`).

Run: `./run_scenarios.sh` (needs a built `build/panSim`, see `SETUP_HPC.md`), then
`python analyze.py`. Edit `scenarios.yaml` to define interventions; keep `baseline`.
Requires pandas, matplotlib, pyyaml.
