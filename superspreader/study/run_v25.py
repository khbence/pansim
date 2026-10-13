#!/usr/bin/env python3
"""Note 25 experiments on locations1.json and agents2.json.

  python3 run_v25.py occ     3-week occupancy reference on the new files
  python3 run_v25.py v1      early-R calibration, open replicates, four wave runs
  python3 run_v25.py v4      shape-0.03 subcritical chains on the new map
  python3 run_v25.py v5      X1 chain arms on the new map with hospitals split into wards
  python3 run_v25.py v2      A0, E, P0 at 12 replicates, venue mechanisms off
  python3 run_v25.py v3      contact-ceiling shapes 0.3, 0.08, 0.03
  python3 run_v25.py all     occ, v1, v4, v5, v2, v3
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_phase2 import (  # noqa: E402
    BASE_K,
    chain_args,
    early_r,
    mech_args,
    open_epidemic_args,
    wave_args,
    write_seed_file,
)
from run_study import CLOSURE_NONE, ROOT  # noqa: E402
from run_x21 import calibrate, run_job as run_job_x21  # noqa: E402

# Rebind the runner's output root by wrapping run_job.
from run_x21 import RESULTS as X21_RESULTS  # noqa: E402
import run_x21  # noqa: E402

RESULTS = HERE / "results" / "v25"
run_x21.RESULTS = RESULTS
run_x21.LOG = RESULTS / "calibration_log.json"

AGENTS = ROOT / "inputRealExample" / "agents2.json"
LOCS = ROOT / "inputRealExample" / "locations1.json"
OLD_AGENTS = ROOT / "inputRealExample" / "agents1.json"
OLD_LOCS = ROOT / "inputRealExample" / "locations0.json"
WARDS = HERE / "inputs" / "v25_wards"
JOB_NEW = {"agents": AGENTS, "locations": LOCS}


def run_job(name, args, **kwargs):
    return run_job_x21(name, args, **kwargs)


def v_occ():
    mech = mech_args(0, 0, "off", 1, 1.0)
    out = RESULTS / "v0" / "occupancy.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    args = open_epidemic_args(mech)(BASE_K, 7001, 3) + ["--occupancyStats", str(out)]
    code = run_job("v0/reference", args, **JOB_NEW)
    if code:
        raise SystemExit(code)


def v1():
    mech = mech_args(0, 0, "off", 1, 1.0)
    make = open_epidemic_args(mech)
    for label, job in (("old", {"agents": OLD_AGENTS, "locations": OLD_LOCS}), ("new", JOB_NEW)):
        k, measured = calibrate(
            f"v1_{label}", lambda k, seed, make=make: make(k, seed, 4), early_r, 1.39, BASE_K, (11, 12), **job,
        )
        for rep in range(1, 9):
            code = run_job(f"v1/{label}/rep{rep:02d}", make(k, 1200 + rep, 8), **job)
            if code:
                raise SystemExit(code)
        if label == "new":
            waves = wave_args(mech)
            for rep in range(1, 5):
                code = run_job(f"v1/waves_new/rep{rep:02d}", waves(k, 1300 + rep, 37), **job)
                if code:
                    raise SystemExit(code)


def v4():
    write_seed_file()
    mech = mech_args(0.03, 1, "off", 1, 1.0)
    make = chain_args(mech)
    k, _ = calibrate(
        "v4_shape0.03", lambda k, seed, make=make: make(k, seed, 7),
        __import__("run_phase2", fromlist=["all_node_mean"]).all_node_mean,
        0.33, 0.000132933, (61, 62), **JOB_NEW,
    )
    for rep in range(1, 41):
        code = run_job(f"v4/shape0.03/rep{rep:02d}", make(k, 8400 + rep, 7), **JOB_NEW)
        if code:
            raise SystemExit(code)


def v5():
    if not (WARDS / "agents_wards.json").exists():
        from build_wards import main as build
        build(LOCS, AGENTS, WARDS)
    write_seed_file()
    job = {"agents": WARDS / "agents_wards.json", "locations": WARDS / "locations_wards.json"}
    from run_phase2 import all_node_mean
    arms = [
        ("flat_shed0.04", mech_args(0.04, 1, "off", 1, 1.0), 1.28e-4),
        ("caregiving_shed0.09", mech_args(0.09, 0, "caregiving", 1, 1.0), 6.7e-5),
    ]
    for label, mech, k0 in arms:
        make = chain_args(mech)
        k, _ = calibrate(
            f"v5_{label}", lambda k, seed, make=make: make(k, seed, 7), all_node_mean, 0.33, k0, (71, 72), **job,
        )
        for rep in range(1, 41):
            code = run_job(f"v5/{label}/rep{rep:02d}", make(k, 8600 + rep, 7), **job)
            if code:
                raise SystemExit(code)


def ps1_args(extra):
    def build(k, seed, weeks):
        return [
            "-r", "--quarantinePolicy", "0", "--closures", str(CLOSURE_NONE),
            "-k", run_x21.fmt(k), "-w", str(weeks), "--seed", str(seed),
            "--densityExponent", "1", "--venueProfile", "off", *extra,
        ]
    return build


def v2():
    arms = [
        ("A0", [], 0.00041),
        ("E", ["--sheddingShape", "0.3"], 0.000483661),
        ("P0", ["--traitShape", "0.3", "--traitPersist", "1", "--traitSusceptibilityExp", "0"], 0.000483661),
    ]
    for label, flags, k0 in arms:
        make = ps1_args(flags)
        k, _ = calibrate(
            f"v2_{label}", lambda k, seed, make=make: make(k, seed, 4), early_r, 1.39, k0, (1101, 1102), tol=0.05, **JOB_NEW,
        )
        for seed in range(1101, 1113):
            code = run_job(f"v2/{label}/rep{seed}", make(k, seed, 10), **JOB_NEW)
            if code:
                raise SystemExit(code)


def v3():
    shapes = ((0.3, 0.00048), (0.08, 0.0007), (0.03, 0.0012))
    for shape, k0 in shapes:
        make = ps1_args(["--sheddingShape", run_x21.fmt(shape)])
        k, _ = calibrate(
            f"v3_shape{shape:g}", lambda k, seed, make=make: make(k, seed, 4), early_r, 1.39, k0, (91, 92), **JOB_NEW,
        )
        for rep in range(1, 9):
            code = run_job(f"v3/shape{shape:g}/rep{rep:02d}", make(k, 1400 + rep, 10), **JOB_NEW)
            if code:
                raise SystemExit(code)


def main():
    phases = {"occ": v_occ, "v1": v1, "v4": v4, "v5": v5, "v2": v2, "v3": v3}
    names = sys.argv[1:] or ["occ"]
    if names == ["all"]:
        names = ["occ", "v1", "v4", "v5", "v2", "v3"]
    for name in names:
        print("==", name, flush=True)
        phases[name]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
