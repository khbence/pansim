Deterministic CPU reference fixture for:

`./panSim --seed 1234 --threads 1 -w 1 -n 8 -N 4 --outAgentStat <file>`

Files:

- `determinism_cpu_seed1234_threads1.stdout`: expected stdout table.
- `determinism_cpu_seed1234_threads1.json`: expected `--outAgentStat` JSON.

Refresh workflow:

1. Build the CPU binary.
2. Run the command above from the repository root.
3. Replace both reference files together only when the deterministic behavior change is intentional.
