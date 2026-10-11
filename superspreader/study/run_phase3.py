#!/usr/bin/env python3
"""Phase 3 follow-ups from 10_reading_of_09_phase2.md, run with the phase-2 binary.

Phases: tail (more subcritical chains), waves (more replicates of the wave-2 hold-out),
levers (Set II lever comparison at other venue strengths).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_phase2 import (  # noqa: E402
    BASE_K,
    LEVERS,
    RESULTS,
    chain_args,
    early_r,
    fmt,
    mech_args,
    open_epidemic_args,
    stored_k,
    wave_args,
    write_lever_files,
    write_seed_file,
)
from run_study import run_job  # noqa: E402


def phase_tail(total_reps=40):
    write_seed_file()
    arms = [
        ("flat_shed0.04", mech_args(0.04, 1, "off", 1, 1.0), 2e-4),
        ("caregiving_shed0.09", mech_args(0.09, 0, "caregiving", 1, 1.0), 1.5e-4),
    ]
    for label, mech, k0 in arms:
        k = stored_k(f"chain_{label}", k0, (31, 32), chain_args(mech), None, 0.33, 7)
        for rep in range(5, total_reps + 1):
            run_job(f"p2/chain_{label}/rep{rep:02d}", chain_args(mech)(k, 4000 + rep, 7))


def phase_waves():
    baseline = mech_args(0, 0, "off", 1, 1.0)
    for seed in range(105, 109):
        run_job(f"p2/waves_baseline/rep{seed - 100:02d}", wave_args(baseline)(BASE_K, seed, 37))
    mech = mech_args(0.3, 0, "off", 1, 1.0)
    k = stored_k("waves_shed0.3", BASE_K * 1.6, (101, 102), wave_args(mech), None, 8565, 15)
    for seed in range(105, 113):
        run_job(f"p2/waves_shed0.3/rep{seed - 100:02d}", wave_args(mech)(k, seed, 37))


def phase_levers():
    levers = write_lever_files()
    for strength, k0 in ((4, BASE_K * 0.3), (2, BASE_K * 0.55)):
        arm = f"II_set{strength}"
        mech = mech_args(0, 0, "covid", strength, 1.0)
        k = stored_k(f"lever_{arm}", k0, (21, 22), open_epidemic_args(mech), early_r, 1.3, 4)
        for lever in levers:
            closures = LEVERS / f"{lever}.json"
            for rep in range(1, 5):
                run_job(
                    f"p2/lever_{arm}_{lever}/rep{rep:02d}",
                    open_epidemic_args(mech, closures=closures)(k, 3000 + rep, 12),
                )


PHASES = {"levers": phase_levers, "waves": phase_waves, "tail": phase_tail}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        print("== phase", name, flush=True)
        PHASES[name]()
    print("done", flush=True)
