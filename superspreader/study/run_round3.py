#!/usr/bin/env python3
"""Note 31 round 3. Resumes finished jobs. One GPU job at a time.

  python3 superspreader/study/run_round3.py r32 r33_waves r33_volume r34 r35 r36 r37
"""

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_round2  # noqa: E402,F401  rebinds run_x21.RESULTS
import run_x21  # noqa: E402
from analyze import daily_summary, hospital, load_daily, load_trace, mom_k  # noqa: E402
from early_window import early_counts  # noqa: E402
from run_phase2 import (  # noqa: E402
    LEVERS,
    all_node_mean,
    chain_args,
    fmt,
    mech_args,
    open_epidemic_args,
    wave_args,
    write_lever_files,
    write_seed_file,
)
from run_phase3b import FACTORS  # noqa: E402
from run_study import ROOT  # noqa: E402
from run_x21 import calibrate, early_window_r, household_share, run_job  # noqa: E402

RESULTS = HERE / "results" / "round2"
AGENTS2 = ROOT / "inputRealExample" / "agents2.json"
LOCS1 = ROOT / "inputRealExample" / "locations1.json"
AGENTS4 = ROOT / "inputRealExample" / "agents4.json"
LOCS3 = ROOT / "inputRealExample" / "locations3.json"
JOB_A = {"agents": AGENTS2, "locations": LOCS1}
JOB_C = {"agents": AGENTS4, "locations": LOCS3}
PERM = HERE / "inputs" / "r32_agents_type5perm.json"
PERM_SEED = 20261014
BASE_MECH = mech_args(0, 0, "off", 1, 1.0)


def mean(values):
    return sum(values) / len(values)


def day28(run_dir):
    rows = load_daily(run_dir / "stdout.txt")
    if len(rows) < 28:
        raise RuntimeError(f"{run_dir} has {len(rows)} days")
    return int(rows[27]["INF"])


def early_k(run_dir):
    trace = load_trace(run_dir / "trace.tsv")
    counts = early_counts(trace)
    return mom_k(counts) if counts else None


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


def multi(prefix, make, k, seeds, weeks, job):
    values_day = []
    for index, seed in enumerate(seeds, start=1):
        code = run_job(f"{prefix}/rep{index:02d}", make(k, seed, weeks), **job)
        if code:
            raise SystemExit(code)
        values_day.append(day28(RESULTS / prefix / f"rep{index:02d}"))
    return values_day


def match_day28(label, make, k0, seeds, weeks, target, job, rounds=3, tol=0.03):
    """Move k by the log-log secant until mean day-28 infections are within tol of target."""
    state_path = RESULTS / "r3state" / f"{label}.json"
    if state_path.exists():
        stored = json.loads(state_path.read_text())
        if stored.get("matched"):
            print(f"match {label} cached k={stored['k']:.6g} day28={stored['day28']:.1f}", flush=True)
            return stored
    k = k0
    points = []
    matched = False
    last = None
    for rnd in range(rounds):
        prefix = f"r3cal/{label}/round{rnd}"
        values = multi(prefix, make, k, seeds, weeks, job)
        value = mean(values)
        points.append({"round": rnd, "k": k, "day28": value, "values": values})
        print(f"match {label} round {rnd} k={k:.6g} day28={value:.1f} target={target:.1f}", flush=True)
        last = points[-1]
        if abs(value / target - 1.0) <= tol:
            matched = True
            break
        slope = 1.0
        if len(points) >= 2 and points[-2]["day28"] > 0 and value > 0 and points[-2]["k"] != k:
            slope = math.log(value / points[-2]["day28"]) / math.log(k / points[-2]["k"])
            slope = min(8.0, max(0.2, slope))
        # If earlier rounds bracket the target, interpolate in log k rather than stepping off the last point.
        low = [item for item in points if item["day28"] < target]
        high = [item for item in points if item["day28"] > target]
        if low and high and low[-1]["day28"] > 0 and high[-1]["day28"] > 0:
            lo, hi = low[-1], high[-1]
            span = math.log(hi["day28"] / lo["day28"])
            frac = math.log(target / lo["day28"]) / span if span else 0.5
            k = math.exp(math.log(lo["k"]) + frac * (math.log(hi["k"]) - math.log(lo["k"])))
        else:
            k = k * (target / value) ** (1.0 / slope)
    stored = {"k": last["k"], "day28": last["day28"], "matched": matched, "target": target, "points": points}
    save_json(state_path, stored)
    return stored


def permute_type5():
    if PERM.exists() and (PERM.parent / "r32_perm_seed.json").exists():
        print("permutation cached", PERM, flush=True)
        return
    print("permute type-5 ids", flush=True)
    agents = json.loads(AGENTS2.read_text())
    people = agents["people"]
    slots = []
    ids = []
    for person_i, person in enumerate(people):
        for slot_i, slot in enumerate(person.get("locations", [])):
            if slot.get("typeID") == 5:
                slots.append((person_i, slot_i))
                ids.append(slot["locID"])
    before = Counter(ids)
    rng = random.Random(PERM_SEED)
    rng.shuffle(ids)
    for (person_i, slot_i), loc_id in zip(slots, ids):
        people[person_i]["locations"][slot_i]["locID"] = loc_id
    # Swap until no agent has two slots at the same type-5 venue. Counts are preserved.
    by_person = {}
    for pos, (person_i, slot_i) in enumerate(slots):
        by_person.setdefault(person_i, []).append(pos)
    swaps = 0
    for _pass in range(8):
        dirty = []
        for person_i, positions in by_person.items():
            seen = {}
            for pos in positions:
                loc_id = ids[pos]
                if loc_id in seen:
                    dirty.append(pos)
                else:
                    seen[loc_id] = pos
        if not dirty:
            break
        for pos in dirty:
            person_i = slots[pos][0]
            own = {ids[other] for other in by_person[person_i]}
            for attempt in range(10000):
                other = rng.randrange(len(ids))
                other_person = slots[other][0]
                if other_person == person_i:
                    continue
                if ids[other] in own:
                    continue
                other_ids = {ids[p] for p in by_person[other_person]}
                if ids[pos] in other_ids:
                    continue
                ids[pos], ids[other] = ids[other], ids[pos]
                swaps += 1
                break
        for pos, (person_i, slot_i) in enumerate(slots):
            people[person_i]["locations"][slot_i]["locID"] = ids[pos]
    after = Counter(slot["locID"] for person in people for slot in person.get("locations", []) if slot.get("typeID") == 5)
    if before != after:
        raise SystemExit("permutation changed venue counts")
    duplicate_agents = 0
    for person in people:
        seen = [slot["locID"] for slot in person.get("locations", []) if slot.get("typeID") == 5]
        if len(seen) != len(set(seen)):
            duplicate_agents += 1
    if duplicate_agents:
        raise SystemExit(f"permutation left {duplicate_agents} agents with a repeated type-5 venue")
    PERM.parent.mkdir(parents=True, exist_ok=True)
    PERM.write_text(json.dumps(agents, separators=(",", ":")))
    save_json(PERM.parent / "r32_perm_seed.json", {
        "seed": PERM_SEED,
        "slots": len(slots),
        "swaps_to_remove_duplicates": swaps,
        "agents_with_duplicate_type5": duplicate_agents,
        "definition": "type-5 location ids shuffled across slots; each venue keeps its assigned count; duplicates within an agent were swapped away",
    })
    print(f"wrote {PERM} swaps {swaps} duplicate agents {duplicate_agents}", flush=True)


def r32():
    """Peak-prevalence gate. Day-28 infections matched within 3%. Open epidemic, 16 weeks."""
    permute_type5()
    make = open_epidemic_args(BASE_MECH)
    seeds_a = list(range(41001, 41009))
    print("R3-2 arm a", flush=True)
    values_a = multi("r32/a", make, 0.00041, seeds_a, 16, JOB_A)
    target = mean(values_a)
    print(f"arm a day28 mean {target:.1f}", flush=True)
    arms = {
        "b": (list(range(41101, 41109)), {"agents": PERM, "locations": LOCS1}, 0.00041),
        "c": (list(range(41201, 41209)), JOB_C, 0.00041),
    }
    chosen = {"a": {"k": 0.00041, "day28": target, "matched": True}}
    for label, (seeds, job, k0) in arms.items():
        # Four-week calibration, then the accepted coefficient is rerun for 16 weeks.
        stored = match_day28(f"r32_{label}", make, k0, seeds, 4, target, job, rounds=3, tol=0.03)
        chosen[label] = stored
        print(f"R3-2 arm {label} 16-week k={stored['k']:.6g}", flush=True)
        final_vals = multi(f"r32/{label}", make, stored["k"], seeds, 16, job)
        stored["day28_16w"] = mean(final_vals)
        stored["day28_16w_values"] = final_vals
        print(f"arm {label} 16-week day28 {stored['day28_16w']:.1f}", flush=True)
    save_json(RESULTS / "r32_match.json", chosen)


def continue_match(label, make, seeds, weeks, target, job, extra_rounds=4, tol=0.03):
    """More rounds after a bracketed miss. Does not replay earlier coefficients."""
    state_path = RESULTS / "r3state" / f"{label}.json"
    stored = json.loads(state_path.read_text())
    if stored.get("matched") and abs(stored["day28"] / target - 1.0) <= tol:
        print(f"continue {label} already matched", flush=True)
        return stored
    points = stored["points"]
    for _ in range(extra_rounds):
        low = [item for item in points if 0 < item["day28"] < target]
        high = [item for item in points if item["day28"] > target]
        if low and high:
            lo, hi = low[-1], high[-1]
            span = math.log(hi["day28"] / lo["day28"])
            frac = math.log(target / lo["day28"]) / span if span else 0.5
            k = math.exp(math.log(lo["k"]) + frac * (math.log(hi["k"]) - math.log(lo["k"])))
        else:
            last = points[-1]
            k = last["k"] * (target / last["day28"])
        rnd = len(points)
        values = multi(f"r3cal/{label}/round{rnd}", make, k, seeds, weeks, job)
        value = mean(values)
        points.append({"round": rnd, "k": k, "day28": value, "values": values})
        print(f"continue {label} round {rnd} k={k:.6g} day28={value:.1f} target={target:.1f}", flush=True)
        stored = {"k": k, "day28": value, "matched": abs(value / target - 1.0) <= tol, "target": target, "points": points}
        save_json(state_path, stored)
        if stored["matched"]:
            break
    return stored


def r33_finish():
    """Finish volume cells whose three-round search missed the day-28 target."""
    ref = json.loads((RESULTS / "r33_volume_ref.json").read_text())
    target = ref["day28"]
    seeds = list(range(43001, 43013))
    make_base = open_epidemic_args(BASE_MECH)
    cells = [
        ("full_low", "nonresidential", "low"),
        ("full_high", "nonresidential", "high"),
        ("fill_pinned", "fillone", "mid"),
    ]
    for label, mode, scenario in cells:
        extra = ["--volumeModel", mode, "--volumeTypeFactors", FACTORS[scenario]]

        def make(k, seed, weeks, extra=extra, make_base=make_base):
            return make_base(k, seed, weeks) + extra

        continue_match(f"r33_{label}", make, seeds, 4, target, JOB_C, extra_rounds=4, tol=0.03)


def r33_waves():
    mech = BASE_MECH
    waves = wave_args(mech)
    for rep in range(1, 9):
        code = run_job(f"r33/waves/rep{rep:02d}", waves(0.00041, 1400 + rep, 37), **JOB_C)
        if code:
            raise SystemExit(code)


def r33_volume():
    """Match day-28 infections across the four volume cells to within 3% (max/min)."""
    cells = [
        ("full_mid", "nonresidential", "mid", 0.000342732),
        ("full_low", "nonresidential", "low", 0.0002),
        ("full_high", "nonresidential", "high", 0.000522617),
        ("fill_pinned", "fillone", "mid", 0.000279071),
    ]
    make_base = open_epidemic_args(BASE_MECH)
    seeds = list(range(43001, 43013))
    # Reference is full_mid at its previous coefficient. Others move onto its day-28 mean.
    ref_make = lambda k, seed, weeks, make_base=make_base: make_base(k, seed, weeks) + [
        "--volumeModel", "nonresidential", "--volumeTypeFactors", FACTORS["mid"],
    ]
    ref_vals = multi("r33/vol_full_mid", ref_make, 0.000342732, seeds, 4, JOB_C)
    target = mean(ref_vals)
    print(f"volume reference day28 {target:.1f}", flush=True)
    save_json(RESULTS / "r33_volume_ref.json", {"k": 0.000342732, "day28": target, "values": ref_vals})
    for label, mode, scenario, k0 in cells:
        if label == "full_mid":
            continue
        extra = ["--volumeModel", mode, "--volumeTypeFactors", FACTORS[scenario]]

        def make(k, seed, weeks, extra=extra, make_base=make_base):
            return make_base(k, seed, weeks) + extra

        match_day28(f"r33_{label}", make, k0, seeds, 4, target, JOB_C, rounds=3, tol=0.015)
    # Confirm the accepted coefficients on a fresh measurement of the calibration runs themselves.
    # max/min is checked by the report, not by another full pass, unless a cell missed 1.5%.


def r34():
    write_seed_file()
    job = {
        "agents": HERE / "inputs" / "v25_wards" / "agents_wards.json",
        "locations": HERE / "inputs" / "v25_wards" / "locations_wards.json",
    }
    mech = mech_args(0.03, 1, "off", 1, 1.0)
    make = chain_args(mech)
    k, measured = calibrate(
        "r34_hospital", lambda k, seed, make=make: make(k, seed, 7), all_node_mean, 0.33,
        0.00014, (83, 84, 85, 86), tol=0.05, **job,
    )
    probe = []
    for rep in range(1, 9):
        code = run_job(f"r34/probe/rep{rep:02d}", make(k, 4500 + rep, 7), **job)
        if code:
            raise SystemExit(code)
        probe.append(all_node_mean(RESULTS / f"r34/probe/rep{rep:02d}"))
    probe_mean = mean(probe)
    print(f"R3-4 probe mean {probe_mean:.4f} at k={k:.6g}", flush=True)
    if abs(probe_mean / 0.33 - 1.0) > 0.05 and probe_mean > 0:
        k = k * (0.33 / probe_mean)
        print(f"R3-4 rescale k to {k:.6g}", flush=True)
    save_json(RESULTS / "r34_k.json", {"k": k, "calibration_measured": measured, "probe_mean": probe_mean})
    for rep in range(1, 41):
        code = run_job(f"r34/shape0.03/rep{rep:02d}", make(k, 4600 + rep, 7), **job)
        if code:
            raise SystemExit(code)


def r35():
    from run_x21 import ps1_args
    seeds = list(range(47001, 47013))
    arms = {
        "A0": (ps1_args([]), 0.00041),
        "E": (ps1_args(["--sheddingShape", "0.3"]), 0.000497841223604006),
        "P0": (ps1_args(["--traitShape", "0.3", "--traitPersist", "1", "--traitSusceptibilityExp", "0"]), 0.000510872),
    }
    print("R3-5 A0", flush=True)
    values = multi("r35/A0", arms["A0"][0], arms["A0"][1], seeds, 10, JOB_A)
    target = mean(values)
    print(f"A0 day28 {target:.1f}", flush=True)
    save_json(RESULTS / "r35_A0.json", {"k": 0.00041, "day28": target, "values": values})
    for label in ("E", "P0"):
        make, k0 = arms[label]
        match_day28(f"r35_{label}", make, k0, seeds, 10, target, JOB_A, rounds=3, tol=0.03)


def r36():
    from run_round2 import event_make
    seeds = list(range(48001, 48013))
    base = open_epidemic_args(BASE_MECH)
    values = multi("r36/baseline", base, 0.00041, seeds, 4, JOB_A)
    target = mean(values)
    print(f"R3-6 baseline day28 {target:.1f}", flush=True)
    save_json(RESULTS / "r36_baseline.json", {"k": 0.00041, "day28": target, "values": values})
    specs = [
        ("b_r0.003_f1000", "5,6,8,9", 0.003, 1000, 0.000256024),
        ("b_r0.003_f100", "5,6,8,9", 0.003, 100, 0.000358711),
        ("a_r0.003_f2000", "5,8", 0.003, 2000, 0.0003),
        ("a_r0.003_f1200", "5,8", 0.003, 1200, 0.00032),
        ("a_r0.003_f1500", "5,8", 0.003, 1500, 0.00031),
    ]
    for label, types, rate, factor, k0 in specs:
        build = event_make(types, rate, factor)

        def make(k, seed, weeks, label=label, build=build):
            log_path = RESULTS / "r36" / label / f"rep_seed{seed}.events.tsv"
            return build(k, seed, weeks, log_path)

        stored = match_day28(f"r36_{label}", make, k0, seeds, 4, target, JOB_A, rounds=3, tol=0.03)
        # Eight-week replicates at the accepted coefficient, for cluster sizes.
        # Fresh seeds so the calibration runs are not the reported sample twice.
        report_seeds = list(range(48101, 48113))
        for index, seed in enumerate(report_seeds, start=1):
            log_path = RESULTS / "r36" / label / f"rep{index:02d}.events.tsv"
            code = run_job(
                f"r36/{label}/rep{index:02d}",
                build(stored["k"], seed, 8, log_path),
                **JOB_A,
            )
            if code:
                raise SystemExit(code)


def r36_finish():
    """Extra rounds for the two configuration-(b) arms that missed day 28, then 8-week replicates."""
    from run_round2 import event_make
    target = json.loads((RESULTS / "r36_baseline.json").read_text())["day28"]
    seeds = list(range(48001, 48013))
    specs = [
        ("b_r0.003_f1000", "5,6,8,9", 0.003, 1000),
        ("b_r0.003_f100", "5,6,8,9", 0.003, 100),
    ]
    for label, types, rate, factor in specs:
        build = event_make(types, rate, factor)

        def make(k, seed, weeks, label=label, build=build):
            log_path = RESULTS / "r36" / label / f"finish_seed{seed}.events.tsv"
            return build(k, seed, weeks, log_path)

        stored = continue_match(f"r36_{label}", make, seeds, 4, target, JOB_A, extra_rounds=3, tol=0.03)
        report_seeds = list(range(48201, 48213))
        for index, seed in enumerate(report_seeds, start=1):
            log_path = RESULTS / "r36" / f"matched_{label}" / f"rep{index:02d}.events.tsv"
            code = run_job(
                f"r36/matched_{label}/rep{index:02d}",
                build(stored["k"], seed, 8, log_path),
                **JOB_A,
            )
            if code:
                raise SystemExit(code)


def r37():
    write_lever_files()
    from run_phase3b import write_ventilation_lever
    write_ventilation_lever()
    # shape 0.5 already calibrated in W9.
    shape_k = json.loads((RESULTS / "cal" / "w9_shape0.5" / "k.json").read_text())["k"]
    levers = ["none", "venue_close", "household", "ventilation"]
    mech = mech_args(0.5, 0, "off", 1, 1.0)
    for lever in levers:
        closures = LEVERS / f"{lever}.json"
        build = open_epidemic_args(mech, closures=closures)
        for rep in range(1, 13):
            code = run_job(f"r37/shape0.5_{lever}/rep{rep:02d}", build(shape_k, 9900 + rep, 12), **JOB_A)
            if code:
                raise SystemExit(code)
    # One arm aimed at household share 0.375. Shape is the lever; early R stays at 1.3.
    candidates = (0.35, 0.25, 0.45)
    chosen = None
    for shape in candidates:
        mech_s = mech_args(shape, 0, "off", 1, 1.0)
        make = open_epidemic_args(mech_s)
        k, _measured = calibrate(
            f"r37_shape{shape:g}", lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.3,
            0.0004, (501, 502), tol=0.05, **JOB_A,
        )
        shares = []
        for seed in (503, 504):
            name = f"r37/pilot_shape{shape:g}/s{seed}"
            code = run_job(name, make(k, seed, 12), **JOB_A)
            if code:
                raise SystemExit(code)
            shares.append(household_share(RESULTS / name))
        share = mean(shares)
        print(f"R3-7 shape {shape:g} share {share:.3f}", flush=True)
        record = {"shape": shape, "k": k, "household_share": share}
        if chosen is None or abs(share - 0.375) < abs(chosen["household_share"] - 0.375):
            chosen = record
        if 0.370 <= share <= 0.380:
            break
    save_json(RESULTS / "r37_pilot.json", chosen)
    mech_s = mech_args(chosen["shape"], 0, "off", 1, 1.0)
    for lever in levers:
        closures = LEVERS / f"{lever}.json"
        build = open_epidemic_args(mech_s, closures=closures)
        for rep in range(1, 13):
            code = run_job(
                f"r37/share375_{lever}/rep{rep:02d}",
                build(chosen["k"], 9920 + rep, 12),
                **JOB_A,
            )
            if code:
                raise SystemExit(code)


def main():
    phases = {
        "r32": r32,
        "r33_waves": r33_waves,
        "r33_volume": r33_volume,
        "r33_finish": r33_finish,
        "r34": r34,
        "r35": r35,
        "r36": r36,
        "r36_finish": r36_finish,
        "r37": r37,
    }
    names = sys.argv[1:] or ["r32"]
    for name in names:
        print("==", name, flush=True)
        phases[name]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
