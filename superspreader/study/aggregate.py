#!/usr/bin/env python3
"""Pool replicates for every finished experiment directory."""

import sys
from pathlib import Path

from analyze import ascertainment, chain_sizes_fixed, json_ready, load_trace, pool_runs, write_json

RESULTS = Path(__file__).resolve().parent / "results"


def cells(experiment):
    root = RESULTS / experiment
    if not root.exists():
        return
    for cell in sorted(path for path in root.iterdir() if path.is_dir()):
        runs = sorted(
            path
            for path in cell.iterdir()
            if path.is_dir() and (path / "trace.tsv").exists() and (path / "stdout.txt").exists()
        )
        if runs:
            yield cell, runs


def main():
    experiment = sys.argv[1]
    summaries = {}
    for cell, runs in cells(experiment):
        print(cell.name, len(runs), flush=True)
        pooled = pool_runs(runs)
        # Drop per-run offspring-heavy fields that are already summarized.
        pooled["runs"] = [
            {
                "run": item["run"],
                "infections": item["infections"],
                "k": item["offspring"].get("k"),
                "mean": item["offspring"].get("mean"),
                "zero_fraction": item["offspring"].get("zero_fraction"),
                "top20_infectors": item["top20_infectors"],
                "household_fraction": item["venues"]["household_fraction"],
                "top10_venues": item["venues"]["top10_venues"],
                "peak_prevalence": item["daily"].get("peak_prevalence"),
                "final_prevalence": item["daily"].get("final_prevalence"),
                "peak_hospital": item["daily"].get("peak_hospital"),
                "growth_day14": item["daily"].get("growth_day14"),
            }
            for item in pooled["runs"]
        ]
        if experiment in ("exp1", "waves") and cell.name in (
            "shedoff_shuf0_venueoff",
            "baseline",
            "shed0.1_shuf0_venueoff",
            "biology",
        ):
            trace = load_trace(runs[0] / "trace.tsv")
            pooled["ascertainment_rep01"] = ascertainment(trace)
        if experiment == "exp4":
            sizes = []
            zeros = []
            for run in runs:
                chains = chain_sizes_fixed(load_trace(run / "trace.tsv"))
                sizes.extend(chains.get("sizes", []))
                fit_zero = pooled["pooled"].get("zero_fraction")
                zeros.append(fit_zero)
            if sizes:
                pooled["chains"] = {
                    "n": len(sizes),
                    "max": max(sizes),
                    "mean": sum(sizes) / len(sizes),
                    "p_gt_10": sum(size > 10 for size in sizes) / len(sizes),
                    "p_gt_50": sum(size > 50 for size in sizes) / len(sizes),
                }
        summaries[cell.name] = pooled
    write_json(RESULTS / f"{experiment}_summary.json", summaries)
    print("wrote", RESULTS / f"{experiment}_summary.json", flush=True)


if __name__ == "__main__":
    main()
