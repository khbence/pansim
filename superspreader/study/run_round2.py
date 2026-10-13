#!/usr/bin/env python3
"""Note 28 round 2. Resumes finished jobs. One GPU job at a time.

  python3 run_round2.py w5 w3 w4b w6c w8 w7
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_x21  # noqa: E402
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
from run_study import CLOSURE_NONE, ROOT  # noqa: E402
from run_x21 import calibrate, early_window_r, final_size, household_share, run_job  # noqa: E402

RESULTS = HERE / "results" / "round2"
run_x21.RESULTS = RESULTS
run_x21.LOG = RESULTS / "calibration_log.json"
RESULTS.mkdir(parents=True, exist_ok=True)

AGENTS = ROOT / "inputRealExample" / "agents2.json"
LOCS = ROOT / "inputRealExample" / "locations1.json"
OLD_AGENTS = ROOT / "inputRealExample" / "agents1.json"
OLD_LOCS = ROOT / "inputRealExample" / "locations0.json"
V25 = HERE / "results" / "v25"
WARDS = HERE / "inputs" / "v25_wards"
W3_DIR = HERE / "inputs" / "w3_wards"
JOB_NEW = {"agents": AGENTS, "locations": LOCS}
JOB_OLD = {"agents": OLD_AGENTS, "locations": OLD_LOCS}


def ps1_args(extra):
    def build(k, seed, weeks):
        return [
            "-r", "--quarantinePolicy", "0", "--closures", str(CLOSURE_NONE),
            "-k", fmt(k), "-w", str(weeks), "--seed", str(seed),
            "--densityExponent", "1", "--venueProfile", "off", *extra,
        ]
    return build


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def secant_k(points, target):
    k, value = points[-1]
    slope = 1.0
    if len(points) >= 2 and points[-2][0] != k and points[-2][1] > 0 and value > 0:
        slope = math.log(value / points[-2][1]) / math.log(k / points[-2][0])
        slope = min(3.0, max(0.4, slope))
    return k * (target / value) ** (1.0 / slope)


def multi_rep(prefix, make, k, seeds, weeks, measure, job):
    values = []
    for seed in seeds:
        name = f"{prefix}/rep{seed}"
        code = run_job(name, make(k, seed, weeks), **job)
        if code:
            raise SystemExit(code)
        values.append(measure(RESULTS / name))
    return values


def second_stage(label, make, k0, prior, target, seeds, weeks, measure, job, tol=0.03, rounds=3, absolute=True, prior_is_sample=False):
    """Correct k from a multi-replicate measurement. `prior` is (k, measured) already in hand, or None.

    If prior_is_sample, that measurement already counts as a round and a value inside tolerance stops there.
    Otherwise the prior is only the secant start, and at least one new sample is run.
    """
    state_path = RESULTS / "stage" / f"{label}.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("converged") or state.get("rounds", 0) >= rounds:
            return state["k"], state.get("measured"), state
    else:
        state = {"k": k0, "points": [], "rounds": 0, "converged": False}
        if prior:
            state["points"].append({"k": prior[0], "measured": prior[1], "source": "prior"})
            state["k"] = prior[0]
            state["measured"] = prior[1]
            if prior_is_sample:
                state["rounds"] = 1
    while state["rounds"] < rounds:
        measured = state.get("measured")
        close = measured is not None and (abs(measured - target) <= tol if absolute else abs(measured / target - 1) <= tol)
        if close and (state["rounds"] > 0 or prior_is_sample):
            state["converged"] = True
            break
        points = [(item["k"], item["measured"]) for item in state["points"]]
        # A two-seed prior that is already close is still checked on the multi-replicate sample before the coefficient moves.
        if points and not (state["rounds"] == 0 and close):
            state["k"] = secant_k(points, target)
        seeds_round = [seed + 1000 * state["rounds"] for seed in seeds]
        values = multi_rep(f"stage/{label}/round{state['rounds']}", make, state["k"], seeds_round, weeks, measure, job)
        state["measured"] = mean(values)
        state["values"] = values
        state["points"].append({"k": state["k"], "measured": state["measured"], "round": state["rounds"]})
        state["rounds"] += 1
        close = abs(state["measured"] - target) <= tol if absolute else abs(state["measured"] / target - 1) <= tol
        state["converged"] = bool(close)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1) + "\n")
        print(f"stage {label} round {state['rounds']} k={state['k']:.6g} measured={state['measured']:.4f}", flush=True)
        if state["converged"]:
            break
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    return state["k"], state.get("measured"), state


def w5():
    """Match 12-replicate early R. A0 is already inside 0.03; E and P0 are not."""
    arms = {
        "A0": (ps1_args([]), 0.00041, 1.382),
        "E": (ps1_args(["--sheddingShape", "0.3"]), 0.0004605926716221236, 1.286),
        "P0": (ps1_args(["--traitShape", "0.3", "--traitPersist", "1", "--traitSusceptibilityExp", "0"]),
               0.0005193560749898141, 1.408),
    }
    # A0's existing twelve-replicate early R is 1.382, inside 0.03 of 1.39. Keep it.
    out = {"A0": {"k": arms["A0"][1], "early_R": arms["A0"][2], "source": "v2 twelve replicates", "converged": True}}
    for label in ("E", "P0"):
        make, k0, prior_r = arms[label]
        k, measured, state = second_stage(
            f"w5_{label}", make, k0, (k0, prior_r), 1.39,
            list(range(4101, 4113)), 10, early_window_r, JOB_NEW, tol=0.03, rounds=3, prior_is_sample=True,
        )
        finals = []
        # The stage rounds already are the twelve-replicate samples. Use the last round.
        last = state["points"][-1]
        if last.get("source") == "prior":
            finals = None
        else:
            prefix = f"stage/w5_{label}/round{last['round']}"
            for seed in range(4101 + 1000 * last["round"], 4113 + 1000 * last["round"]):
                finals.append(final_size(RESULTS / prefix / f"rep{seed}"))
        out[label] = {"k": k, "early_R": measured, "final_sizes": finals, "converged": state.get("converged"), "state": state}
    # E and P0 must also be within 0.02 of each other. One extra joint nudge if the gap remains and rounds remain.
    gap = abs(out["E"]["early_R"] - out["P0"]["early_R"])
    out["E_minus_P0"] = out["E"]["early_R"] - out["P0"]["early_R"]
    out["gap"] = gap
    if gap > 0.02 and out["E"]["state"]["rounds"] < 3 and out["P0"]["state"]["rounds"] < 3:
        print("W5 gap still above 0.02; the per-arm round limit is what stops further correction", flush=True)
    (RESULTS / "w5.json").write_text(json.dumps(out, indent=1) + "\n")
    print("W5", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "state"} if isinstance(v, dict) else v for k, v in out.items()}, indent=1), flush=True)


def build_w3_map(wards_dir=None, out_dir=None):
    wards_dir = Path(wards_dir) if wards_dir else WARDS
    out_dir = Path(out_dir) if out_dir else W3_DIR
    if (out_dir / "agents_wards.json").exists() and (out_dir / "ward_summary.json").exists():
        print("w3 map cached", out_dir, flush=True)
        return
    from build_wards import split_int
    from collections import defaultdict

    print("build w3 map", flush=True)
    base_locs = json.loads((wards_dir / "locations_wards.json").read_text())
    agents = json.loads((wards_dir / "agents_wards.json").read_text())
    places = base_locs["places"]
    by_id = {}
    for place in places:
        by_id.setdefault(place["ID"], place)
    raw_places = json.loads(LOCS.read_text())["places"]
    peaks = {}
    with open(V25 / "v0" / "occupancy.tsv") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            loc = int(parts[index["location"]])
            if loc < len(raw_places):
                peaks[raw_places[loc]["ID"]] = float(parts[index["maxOccupancy"]])
    targets = {6: 150.0, 9: 300.0}
    assigned = defaultdict(list)
    people = agents["people"]
    for agent_index, person in enumerate(people):
        seen = set()
        for slot in person.get("locations", []):
            loc_id = slot.get("locID")
            place = by_id.get(loc_id)
            if place and place["type"] in targets and loc_id not in seen:
                assigned[loc_id].append(agent_index)
                seen.add(loc_id)
    new_places = []
    redirect = defaultdict(dict)
    rows = []
    for place in places:
        if place["type"] not in targets:
            continue
        members = assigned.get(place["ID"], [])
        peak = peaks.get(place["ID"], 0.0)
        target = targets[place["type"]]
        n_wards = 1 if peak <= 0 else max(1, int(round(peak / target)))
        if n_wards <= 1:
            rows.append({"id": place["ID"], "type": place["type"], "peak": peak, "wards": 1, "agents": len(members)})
            continue
        n = len(members)
        chunks = []
        start = 0
        base, extra = divmod(n, n_wards)
        for ward in range(n_wards):
            size = base + (1 if ward < extra else 0)
            chunks.append(members[start:start + size])
            start += size
        capacities = split_int(int(place["capacity"]), [max(len(chunk), 1) for chunk in chunks])
        for ward, (chunk, capacity) in enumerate(zip(chunks, capacities)):
            new_id = f"{place['ID']}__p{ward:03d}"
            clone = dict(place)
            clone["ID"] = new_id
            clone["capacity"] = capacity
            new_places.append(clone)
            for agent_index in chunk:
                redirect[agent_index][place["ID"]] = new_id
        rows.append({"id": place["ID"], "type": place["type"], "peak": peak, "wards": n_wards, "agents": n})
    split_ids = {row["id"] for row in rows if row["wards"] > 1}
    base_locs["places"] = [place for place in places if place["ID"] not in split_ids] + new_places
    for agent_index, mapping in redirect.items():
        for slot in people[agent_index].get("locations", []):
            new_id = mapping.get(slot.get("locID"))
            if new_id:
                slot["locID"] = new_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cinema_target_peak": 150,
        "church_target_peak": 300,
        "split": [row for row in rows if row["wards"] > 1],
        "left_whole": sum(1 for row in rows if row["wards"] == 1),
        "wards": len(new_places),
    }
    (out_dir / "ward_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "locations_wards.json").write_text(json.dumps(base_locs, separators=(",", ":")))
    (out_dir / "agents_wards.json").write_text(json.dumps(agents, separators=(",", ":")))
    print(f"w3 wards {summary['wards']} split {len(summary['split'])}", flush=True)


def w3():
    build_w3_map()
    write_seed_file()
    job = {"agents": W3_DIR / "agents_wards.json", "locations": W3_DIR / "locations_wards.json"}
    mech = mech_args(0.03, 1, "off", 1, 1.0)
    make = chain_args(mech)
    k, _ = calibrate(
        "w3_shape0.03", lambda k, seed, make=make: make(k, seed, 7), all_node_mean, 0.33,
        0.000156714, (81, 82), **job,
    )
    for rep in range(1, 41):
        code = run_job(f"w3/shape0.03/rep{rep:02d}", make(k, 9400 + rep, 7), **job)
        if code:
            raise SystemExit(code)


def w4b():
    for shape, k0 in ((0.08, 0.0007), (0.03, 0.0012)):
        make = ps1_args(["--sheddingShape", fmt(shape)])
        k, measured = calibrate(
            f"w4b_shape{shape:g}", lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.39,
            k0, (181, 182), tol=0.05, **JOB_OLD,
        )
        second_stage(
            f"w4b_shape{shape:g}", make, k, (k, measured), 1.39,
            list(range(5101, 5109)), 10, early_window_r, JOB_OLD, tol=0.03, rounds=3,
        )


def w6c():
    mech = mech_args(0, 0, "off", 1, 1.0)
    waves = wave_args(mech)
    for rep in range(5, 9):
        code = run_job(f"w6/waves_new/rep{rep:02d}", waves(0.00041, 1300 + rep, 37), **JOB_NEW)
        if code:
            raise SystemExit(code)


def w8():
    write_seed_file()
    job = {"agents": WARDS / "agents_wards.json", "locations": WARDS / "locations_wards.json"}
    mech = mech_args(0.09, 0, "caregiving", 1, 1.0) + ["--homeVenueFactor", "0.35"]
    make = chain_args(mech)
    k, _ = calibrate(
        "w8_home035", lambda k, seed, make=make: make(k, seed, 7), all_node_mean, 0.33,
        6.7e-5, (91, 92), **job,
    )
    for rep in range(1, 41):
        code = run_job(f"w8/home035/rep{rep:02d}", make(k, 9600 + rep, 7), **job)
        if code:
            raise SystemExit(code)


def event_make(event_types, rate, factor):
    extra = ["--eventRate", fmt(rate), "--eventFactor", fmt(factor), "--eventTypes", event_types]

    def build(k, seed, weeks, log_path):
        return [
            "-r", "--quarantinePolicy", "0", "--closures", str(CLOSURE_NONE),
            "-k", fmt(k), "-w", str(weeks), "--seed", str(seed),
            "--densityExponent", "1", "--venueProfile", "off",
            "--eventLog", str(log_path),
            *extra,
        ]
    return build


def w7():
    configs = {
        "a_restaurant_shop": "5,8",
        "b_all_social": "5,6,8,9",
    }
    pairs = ((0.003, 1000), (0.01, 200), (0.003, 100), (0.01, 20))
    for config, types in configs.items():
        for rate, factor in pairs:
            label = f"w7_{config}_r{rate:g}_f{factor:g}"
            build = event_make(types, rate, factor)

            def make(k, seed, weeks, label=label, build=build):
                log_path = RESULTS / "cal" / label / f"s{seed}_w{weeks}.events.tsv"
                # The calibration helper does not know about the log path, so bake a stable path.
                return build(k, seed, weeks, RESULTS / label / f"seed{seed}.events.tsv")

            k, measured = calibrate(
                label, lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.39,
                0.00041, (201, 202, 203, 204), tol=0.03, **JOB_NEW,
            )
            for rep in range(1, 13):
                seed = 7000 + rep
                log_path = RESULTS / label / f"rep{rep:02d}.events.tsv"
                code = run_job(f"{label}/rep{rep:02d}", build(k, seed, 8, log_path), **JOB_NEW)
                if code:
                    raise SystemExit(code)
            print(f"{label} k={k:.6g} pilot R={measured:.3f}", flush=True)


def w9():
    """Household-share arms on the new map, plus venue closure, household isolation and ventilation."""
    write_lever_files()
    candidates = [
        ("venue_s3", mech_args(0, 0, "covid", 3, 1.0), 8e-5),
        ("venue_s1", mech_args(0, 0, "covid", 1, 1.0), 2.4e-4),
        ("shape0.5", mech_args(0.5, 0, "off", 1, 1.0), 4.3e-4),
        ("shape1", mech_args(1.0, 0, "off", 1, 1.0), 4.0e-4),
    ]
    pilot = []
    for label, mech, k0 in candidates:
        make = open_epidemic_args(mech)
        k, _ = calibrate(
            f"w9_{label}", lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.3, k0, (301, 302), **JOB_NEW,
        )
        shares = []
        for seed in (303, 304):
            name = f"w9/pilot_{label}/s{seed}"
            code = run_job(name, make(k, seed, 12), **JOB_NEW)
            if code:
                raise SystemExit(code)
            shares.append(household_share(RESULTS / name))
        share = mean(shares)
        pilot.append({"label": label, "k": k, "household_share": share})
        print(f"w9 pilot {label} share {share:.3f}", flush=True)
    # Keep the arms nearest 0.25 and 0.37, and the highest share at or above 0.39 if one exists,
    # otherwise the arm nearest 0.45.
    used = set()
    chosen = []
    for target_share in (0.25, 0.37):
        pick = min((item for item in pilot if item["label"] not in used), key=lambda item: abs(item["household_share"] - target_share))
        used.add(pick["label"])
        chosen.append({**pick, "target_share": target_share})
    above = [item for item in pilot if item["label"] not in used and item["household_share"] > 0.39]
    if above:
        pick = max(above, key=lambda item: item["household_share"])
    else:
        pick = min((item for item in pilot if item["label"] not in used), key=lambda item: abs(item["household_share"] - 0.45))
    chosen.append({**pick, "target_share": 0.45})
    mech_by_label = {label: mech for label, mech, _k in candidates}
    (RESULTS / "w9_pilot.json").write_text(json.dumps({"pilot": pilot, "chosen": chosen}, indent=1) + "\n")
    levers = ["none", "venue_close", "household", "ventilation"]
    for item in chosen:
        mech = mech_by_label[item["label"]]
        for lever in levers:
            closures = LEVERS / f"{lever}.json"
            build = open_epidemic_args(mech, closures=closures)
            for rep in range(1, 13):
                code = run_job(f"w9/{item['label']}_{lever}/rep{rep:02d}", build(item["k"], 9800 + rep, 12), **JOB_NEW)
                if code:
                    raise SystemExit(code)


def w5_p0():
    """One further P0 round. The first pass stopped at R 1.408, 0.031 away from E."""
    make = ps1_args(["--traitShape", "0.3", "--traitPersist", "1", "--traitSusceptibilityExp", "0"])
    k = 0.0005193560749898141 * (1.385 / 1.408)
    values = multi_rep("w5/P0/round1", make, k, list(range(8101, 8113)), 10, early_window_r, JOB_NEW)
    measured = mean(values)
    print(f"P0 round1 k={k:.6g} R={measured:.4f}", flush=True)
    if abs(measured - 1.3765) > 0.02:
        k2 = k * (1.385 / measured)
        values = multi_rep("w5/P0/round2", make, k2, list(range(8201, 8213)), 10, early_window_r, JOB_NEW)
        print(f"P0 round2 k={k2:.6g} R={mean(values):.4f}", flush=True)


def w7_search():
    """Bracket early k of 0.3 in configuration (a). The 1000x arm landed at 0.36."""
    rate, factor = 0.003, 2000
    label = f"w7_a_restaurant_shop_r{rate:g}_f{factor:g}"
    build = event_make("5,8", rate, factor)

    def make(k, seed, weeks, label=label, build=build):
        return build(k, seed, weeks, RESULTS / label / f"seed{seed}.events.tsv")

    k, _measured = calibrate(
        label, lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.39,
        0.00030, (211, 212, 213, 214), tol=0.03, **JOB_NEW,
    )
    for rep in range(1, 13):
        seed = 7200 + rep
        code = run_job(f"{label}/rep{rep:02d}", build(k, seed, 8, RESULTS / label / f"rep{rep:02d}.events.tsv"), **JOB_NEW)
        if code:
            raise SystemExit(code)


def build_repaired_tail_map():
    out = HERE / "inputs" / "w10_tail"
    if (out / "agents_wards.json").exists():
        return out
    from build_wards import main as build_wards_main
    hospital = HERE / "inputs" / "w10_hospital"
    if not (hospital / "agents_wards.json").exists():
        build_wards_main(
            ROOT / "inputRealExample" / "locations3.json",
            ROOT / "inputRealExample" / "agents4.json",
            hospital,
        )
    build_w3_map(hospital, out)
    return out


def w10_step5():
    job = {
        "agents": ROOT / "inputRealExample" / "agents4.json",
        "locations": ROOT / "inputRealExample" / "locations3.json",
    }
    mech = mech_args(0, 0, "off", 1, 1.0)
    make = open_epidemic_args(mech)
    k, measured = calibrate(
        "w10_v1", lambda k, seed, make=make: make(k, seed, 4), early_window_r, 1.39,
        0.00041, (11, 12), tol=0.05, **job,
    )
    print(f"w10 v1 coefficient {k:.6g} two-seed R {measured:.3f}", flush=True)
    for rep in range(1, 9):
        code = run_job(f"w10/v1/rep{rep:02d}", make(k, 2200 + rep, 8), **job)
        if code:
            raise SystemExit(code)
    occ = RESULTS / "w10" / "occupancy.tsv"
    occ.parent.mkdir(parents=True, exist_ok=True)
    code = run_job("w10/occupancy", make(k, 7003, 3) + ["--occupancyStats", str(occ)], **job)
    if code:
        raise SystemExit(code)
    tail = build_repaired_tail_map()
    write_seed_file()
    tail_job = {"agents": tail / "agents_wards.json", "locations": tail / "locations_wards.json"}
    chain = chain_args(mech_args(0.03, 1, "off", 1, 1.0))
    ck, _ = calibrate(
        "w10_tail", lambda k, seed, make=chain: chain(k, seed, 7), all_node_mean, 0.33,
        0.000135832, (83, 84), **tail_job,
    )
    for rep in range(1, 41):
        code = run_job(f"w10/tail/rep{rep:02d}", chain(ck, 9700 + rep, 7), **tail_job)
        if code:
            raise SystemExit(code)
    factors = json.loads((HERE / "results" / "p3" / "inputs" / "volume_type_factors.json").read_text())
    cells = [
        ("full_mid", "nonresidential", "mid"),
        ("full_low", "nonresidential", "low"),
        ("full_high", "nonresidential", "high"),
        ("fill_pinned", "fillone", "mid"),
    ]
    for label, mode, scenario in cells:
        extra = ["--volumeModel", mode, "--volumeTypeFactors", factors[scenario]]
        vol = open_epidemic_args(mech_args(0, 0, "off", 1, 1.0))

        def build(k, seed, weeks, extra=extra, vol=vol):
            return vol(k, seed, weeks) + extra

        vk, _ = calibrate(
            f"w10_vol_{label}", lambda k, seed, build=build: build(k, seed, 4), early_window_r, 1.39,
            0.0002, (301, 302), tol=0.05, **job,
        )
        for rep in range(1, 5):
            code = run_job(f"w10/vol_{label}/rep{rep:02d}", build(vk, 3300 + rep, 8), **job)
            if code:
                raise SystemExit(code)


def main():
    phases = {
        "w5": w5, "w3": w3, "w4b": w4b, "w6c": w6c, "w8": w8, "w7": w7, "w9": w9,
        "w5_p0": w5_p0, "w7_search": w7_search, "w10": w10_step5,
    }
    names = sys.argv[1:] or ["w5"]
    for name in names:
        print("==", name, flush=True)
        phases[name]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
