#!/usr/bin/env python3
"""Note 34 round 4. Resumes finished jobs. One GPU job at a time.

  python3 superspreader/study/run_round4.py r41 r42 r43
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_round2  # noqa: E402,F401
from analyze_round2 import index_cases, mom_k  # noqa: E402
from run_phase2 import (  # noqa: E402
    LEVERS,
    fmt,
    mech_args,
    open_epidemic_args,
    write_lever_files,
)
from run_round2 import event_make  # noqa: E402
from run_round3 import JOB_A, RESULTS, continue_match, day28, match_day28, mean, multi  # noqa: E402
from run_x21 import calibrate, early_window_r, household_share, ps1_args, run_job  # noqa: E402

# Exact coefficients from the R3-5 replicates, not the rounded figures in note 34.
K_A0 = 0.00041
K_E = 0.000497841223604006
K_P0 = 0.0004972587500573815
# No-event day-28 mean from the R3-6 baseline (12 replicates).
BASELINE_DAY28 = 6211.25
EVENT_RATE = 0.003


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1) + "\n")


def r41():
    """A0, E and P0 for 16 weeks at the R3-5 coefficients. New seeds."""
    arms = {
        "A0": (ps1_args([]), K_A0, list(range(51001, 51013))),
        "E": (ps1_args(["--sheddingShape", "0.3"]), K_E, list(range(51101, 51113))),
        "P0": (ps1_args(["--traitShape", "0.3", "--traitPersist", "1", "--traitSusceptibilityExp", "0"]), K_P0, list(range(51201, 51213))),
    }
    for label, (make, k, seeds) in arms.items():
        print(f"R4-1 {label} k={k:.8g}", flush=True)
        multi(f"r41/{label}", make, k, seeds, 16, JOB_A)


def early_k_of(prefix):
    parts = []
    for path in sorted((RESULTS / prefix).glob("rep*/trace.tsv")):
        _shed, offs = index_cases(path)
        parts.append(offs)
    if not parts:
        return None
    import numpy as np
    pooled = np.concatenate(parts)
    reps = [mom_k(values) for values in parts]
    return {"pooled_k": mom_k(pooled), "pooled_R": float(pooled.mean()), "rep_k": reps, "n": int(len(pooled))}


def r42():
    """Configuration (a) events restricted by --eventMaxCapacity. Day-28 matched to 6211.25."""
    caps = (100, 300)
    factors = (2500, 10000)
    summary = []
    for cap in caps:
        measured = {}
        for factor in factors:
            label = f"cap{cap}_f{factor}"
            # Higher event factors add infections, so start the background coefficient lower.
            k0 = 0.00025 if factor == 2500 else 0.00012
            seeds = list(range(52001 + cap, 52013 + cap))

            def make(k, seed, weeks, cap=cap, factor=factor, label=label):
                log_path = RESULTS / "r42" / label / f"seed{seed}.events.tsv"
                return event_make("5,8", EVENT_RATE, factor)(k, seed, weeks, log_path) + [
                    "--eventMaxCapacity", str(cap),
                ]

            stored = match_day28(f"r42_{label}", make, k0, seeds, 4, BASELINE_DAY28, JOB_A, rounds=3, tol=0.03)
            if not stored.get("matched"):
                stored = continue_match(f"r42_{label}", make, seeds, 4, BASELINE_DAY28, JOB_A, extra_rounds=2, tol=0.03)
            # Eight-week sample at the accepted coefficient, same seeds, so day 28 is the matched sample.
            for index, seed in enumerate(seeds, start=1):
                log_path = RESULTS / "r42" / label / f"rep{index:02d}.events.tsv"
                code = run_job(
                    f"r42/{label}/rep{index:02d}",
                    event_make("5,8", EVENT_RATE, factor)(stored["k"], seed, 8, log_path) + ["--eventMaxCapacity", str(cap)],
                    **JOB_A,
                )
                if code:
                    raise SystemExit(code)
            stats = early_k_of(f"r42/{label}")
            record = {"cap": cap, "factor": factor, "k_coeff": stored["k"], "day28": stored["day28"], "matched": stored.get("matched"), "early": stats}
            measured[factor] = record
            summary.append(record)
            print(f"R4-2 cap {cap} factor {factor} day28 {stored['day28']:.1f} early k {stats['pooled_k']:.3f}", flush=True)
        k_lo = measured[2500]["early"]["pooled_k"]
        k_hi = measured[10000]["early"]["pooled_k"]
        # k falls as the factor rises. Interpolate if 0.3 is between them.
        if k_hi > 0.6 and k_lo > 0.6:
            print(f"R4-2 cap {cap} stays above 0.6 at factor 10000; stopping", flush=True)
            continue
        if (k_lo - 0.3) * (k_hi - 0.3) < 0 and k_lo > 0 and k_hi > 0:
            frac = math.log(k_lo / 0.3) / math.log(k_lo / k_hi)
            factor = 2500 * (10000 / 2500) ** frac
            print(f"R4-2 cap {cap} interpolating factor {factor:.1f}", flush=True)
            label = f"cap{cap}_f{int(round(factor))}"
            k0 = measured[2500]["k_coeff"]
            seeds = list(range(53001 + cap, 53013 + cap))

            def make_i(k, seed, weeks, cap=cap, factor=factor, label=label):
                log_path = RESULTS / "r42" / label / f"seed{seed}.events.tsv"
                return event_make("5,8", EVENT_RATE, factor)(k, seed, weeks, log_path) + ["--eventMaxCapacity", str(cap)]

            stored = match_day28(f"r42_{label}", make_i, k0, seeds, 4, BASELINE_DAY28, JOB_A, rounds=3, tol=0.03)
            if not stored.get("matched"):
                stored = continue_match(f"r42_{label}", make_i, seeds, 4, BASELINE_DAY28, JOB_A, extra_rounds=2, tol=0.03)
            for index, seed in enumerate(seeds, start=1):
                log_path = RESULTS / "r42" / label / f"rep{index:02d}.events.tsv"
                code = run_job(
                    f"r42/{label}/rep{index:02d}",
                    event_make("5,8", EVENT_RATE, factor)(stored["k"], seed, 8, log_path) + ["--eventMaxCapacity", str(int(cap))],
                    **JOB_A,
                )
                if code:
                    raise SystemExit(code)
            stats = early_k_of(f"r42/{label}")
            summary.append({"cap": cap, "factor": factor, "k_coeff": stored["k"], "day28": stored["day28"], "matched": stored.get("matched"), "early": stats, "interpolated": True})
            print(f"R4-2 interpolated cap {cap} factor {factor:.1f} k {stats['pooled_k']:.3f}", flush=True)
    save_json(RESULTS / "r42_summary.json", summary)


def pilot_share(label, mech, k0):
    make = open_epidemic_args(mech)
    k, _measured = calibrate(
        f"r43_{label}", lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.3,
        k0, (601, 602), tol=0.05, **JOB_A,
    )
    shares = []
    for seed in (603, 604):
        name = f"r43/pilot_{label}/s{seed}"
        code = run_job(name, make(k, seed, 12), **JOB_A)
        if code:
            raise SystemExit(code)
        shares.append(household_share(RESULTS / name))
    return k, mean(shares)


def run_levers(label, mech, k, seed_base):
    write_lever_files()
    from run_phase3b import write_ventilation_lever
    write_ventilation_lever()
    for condition in ("none", "venue_close", "household", "ventilation"):
        build = open_epidemic_args(mech, closures=LEVERS / f"{condition}.json")
        for rep in range(1, 13):
            code = run_job(f"r43/{label}_{condition}/rep{rep:02d}", build(k, seed_base + rep, 12), **JOB_A)
            if code:
                raise SystemExit(code)


def r42_lower(factors=(1200, 600), caps=(100,)):
    """Factors below 2,500. Used when 2,500 already has early k below 0.3, so the prescribed pair does not bracket 0.3."""
    for cap in caps:
        for factor in factors:
            label = f"cap{cap}_f{factor}"
            k0 = 0.00034
            seeds = list(range(54001 + factor + cap, 54013 + factor + cap))

            def make(k, seed, weeks, cap=cap, factor=factor, label=label):
                log_path = RESULTS / "r42" / label / f"seed{seed}.events.tsv"
                return event_make("5,8", EVENT_RATE, factor)(k, seed, weeks, log_path) + ["--eventMaxCapacity", str(cap)]

            stored = match_day28(f"r42_{label}", make, k0, seeds, 4, BASELINE_DAY28, JOB_A, rounds=3, tol=0.03)
            if not stored.get("matched"):
                stored = continue_match(f"r42_{label}", make, seeds, 4, BASELINE_DAY28, JOB_A, extra_rounds=2, tol=0.03)
            for index, seed in enumerate(seeds, start=1):
                log_path = RESULTS / "r42" / label / f"rep{index:02d}.events.tsv"
                code = run_job(
                    f"r42/{label}/rep{index:02d}",
                    event_make("5,8", EVENT_RATE, factor)(stored["k"], seed, 8, log_path) + ["--eventMaxCapacity", str(cap)],
                    **JOB_A,
                )
                if code:
                    raise SystemExit(code)
            stats = early_k_of(f"r42/{label}")
            print(f"R4-2 lower cap {cap} factor {factor} day28 {stored['day28']:.1f} early k {stats['pooled_k']:.3f}", flush=True)
            save_json(RESULTS / f"r42_{label}.json", {"cap": cap, "factor": factor, "k_coeff": stored["k"], "day28": stored["day28"], "matched": stored.get("matched"), "early": stats})


def r43():
    """One shedding-shape arm near share 0.243 and one venue-profile arm near share 0.424.

    Coefficients follow the W9 protocol (early R 1.3), so these arms sit on the same
    table as the five arms already run. Home weight is tuned only if shape alone cannot
    reach 0.243.
    """
    # Venue profile, shedding shape 0. Strength 1 was share 0.365; strength 0 is venues off at 0.424.
    venue_trials = []
    for strength in (0.25, 0.5, 0.12):
        mech = mech_args(0, 0, "covid", strength, 1.0)
        k, share = pilot_share(f"venue_s{strength:g}", mech, 0.0003)
        venue_trials.append({"strength": strength, "k": k, "household_share": share, "mech": "venue"})
        print(f"R4-3 venue strength {strength:g} share {share:.3f}", flush=True)
        if abs(share - 0.424) <= 0.01:
            chosen_venue = venue_trials[-1]
            break
    else:
        chosen_venue = min(venue_trials, key=lambda item: abs(item["household_share"] - 0.424))
    # Shedding shape 0.3, venues off. Shape alone sat near 0.37 at 0.35, so home weight is the search.
    shed_trials = []
    for home in (1.0, 0.35, 0.15):
        mech = mech_args(0.3, 0, "off", 1.0, 1.0) + ["--homeVenueFactor", fmt(home)]
        k, share = pilot_share(f"shape0.3_home{home:g}", mech, 0.00045)
        shed_trials.append({"shape": 0.3, "homeVenueFactor": home, "k": k, "household_share": share})
        print(f"R4-3 shape 0.3 home {home:g} share {share:.3f}", flush=True)
        if abs(share - 0.243) <= 0.01:
            chosen_shed = shed_trials[-1]
            break
    else:
        chosen_shed = min(shed_trials, key=lambda item: abs(item["household_share"] - 0.243))
        # One more step toward 0.243 from the closest home weight if still outside tolerance.
        if abs(chosen_shed["household_share"] - 0.243) > 0.01 and len(shed_trials) >= 2:
            ordered = sorted(shed_trials, key=lambda item: item["homeVenueFactor"])
            below = [item for item in ordered if item["household_share"] <= 0.243]
            above = [item for item in ordered if item["household_share"] >= 0.243]
            if below and above:
                lo, hi = below[-1], above[0]
                span = hi["household_share"] - lo["household_share"]
                frac = (0.243 - lo["household_share"]) / span if span else 0.5
                home = lo["homeVenueFactor"] + frac * (hi["homeVenueFactor"] - lo["homeVenueFactor"])
                mech = mech_args(0.3, 0, "off", 1.0, 1.0) + ["--homeVenueFactor", fmt(home)]
                k, share = pilot_share(f"shape0.3_home{home:.3f}", mech, chosen_shed["k"])
                chosen_shed = {"shape": 0.3, "homeVenueFactor": home, "k": k, "household_share": share}
                print(f"R4-3 interpolated home {home:.3f} share {share:.3f}", flush=True)
    venue_mech = mech_args(0, 0, "covid", chosen_venue["strength"], 1.0)
    shed_mech = mech_args(0.3, 0, "off", 1.0, 1.0) + ["--homeVenueFactor", fmt(chosen_shed["homeVenueFactor"])]
    run_levers("venue_target424", venue_mech, chosen_venue["k"], 61000)
    run_levers("shed_target243", shed_mech, chosen_shed["k"], 62000)
    save_json(RESULTS / "r43_pilot.json", {"venue_trials": venue_trials, "shed_trials": shed_trials, "chosen_venue": chosen_venue, "chosen_shed": chosen_shed})


def main():
    phases = {"r41": r41, "r42": r42, "r42_lower": r42_lower, "r43": r43}
    names = sys.argv[1:] or ["r41"]
    for name in names:
        print("==", name, flush=True)
        phases[name]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
