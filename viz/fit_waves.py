#!/usr/bin/env python3
"""Fit panSim one wave at a time.

Each wave tunes only its own infectiousness, progression, and introduction
(seed day and daily fraction). Earlier waves stay frozen. The loss is a
weighted sum of hospital RMSE and reconstruction-infected RMSE, each divided
by the standard deviation of that target on the wave window so the two series
share a scale. Hospital weight defaults to 0.75.

Search evaluations use one stochastic run. The best candidate and the wave's
starting parameters are then repeated --finalize-runs times, and the better
mean is kept.

Results go to viz/fitted_parameters.json and viz/fitted_closure.json.
viz/run_ensemble.sh is not modified.

    viz/.venv/bin/python viz/fit_waves.py --finalize-runs 3 --hospital-weight 0.75
    viz/.venv/bin/python viz/fit_waves.py --from-wave delta --search-evals 8 --finalize-runs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/home/ireguly/pansim")
sys.path.insert(0, str(ROOT / "viz"))
from pansimviz.ground_truth import load_ground_truth
from pansimviz.parse import parse_text

BIN = ROOT / "build_gpu" / "panSim"
CLOSURE_SRC = ROOT / "inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json"
PARAMS_OUT = ROOT / "viz" / "fitted_parameters.json"
CLOSURE_OUT = ROOT / "viz" / "fitted_closure.json"
RUN_ROOT = Path("/tmp/pansim-fit/incremental")
NP = 9_967_304
START = np.datetime64("2020-09-23")
# Warm start. These match the current closure seeds; the ensemble script is left as it is.
INITIAL_INF = [0.98, 1.85, 1.98, 2.42, 4.10, 4.55, 6.8]
INITIAL_PROG = [0.94, 1.03, 0.84, 0.72, 0.57, 0.463, 0.45]
INITIAL_K = 0.00041
QUARANTINE = 3
STOCK = ("S", "E", "I1", "I2", "I3", "I4", "I5_h", "I6_h", "R_h", "R", "D1", "D2")
PREVALENCE = ("E", "I1", "I2", "I3", "I4", "I5_h", "I6_h")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = datetime(1899, 12, 30)
LAST_DAY = np.datetime64("2022-08-31")

# name, score start, score end, multiplier index, ExposeToMutation variant or None
WAVES = (
    ("wildtype", "2020-09-23", "2021-01-31", 0, None),
    ("alpha", "2021-01-26", "2021-06-02", 1, 1),
    ("delta", "2021-08-05", "2022-01-11", 2, 2),
    ("ba1", "2022-01-12", "2022-03-15", 3, 3),
    ("ba2", "2022-02-15", "2022-06-15", 4, 4),
    ("summer", "2022-06-01", "2022-08-31", 5, 5),
)


def day_index(date: str) -> int:
    return int((np.datetime64(date) - START) / np.timedelta64(1, "D"))


def weeks_for(end_day: int) -> int:
    return (end_day + 1 + 6) // 7


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch) - 64)
    return n


def load_infected(dates: np.ndarray) -> np.ndarray:
    archive = zipfile.ZipFile(ROOT / "Full_reconstruction_2026-09-26.xlsx")
    root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    strings = []
    shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    for si in shared.findall("m:si", NS):
        strings.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    header = {}
    rows = root.findall("m:sheetData/m:row", NS)
    for cell in rows[0].findall("m:c", NS):
        value = cell.find("m:v", NS)
        header[_col_index(cell.attrib["r"])] = strings[int(value.text)]
    date_col = next(i for i, name in header.items() if name == "Date")
    inf_col = next(i for i, name in header.items() if name == "Infected")
    by_date = {}
    for row in rows[1:]:
        vals = {}
        for cell in row.findall("m:c", NS):
            ci = _col_index(cell.attrib["r"])
            if ci not in (date_col, inf_col):
                continue
            value = cell.find("m:v", NS)
            if value is None or value.text is None:
                continue
            vals[ci] = float(value.text)
        if date_col in vals and inf_col in vals:
            day = np.datetime64(EXCEL_EPOCH + timedelta(days=int(vals[date_col])))
            by_date[day] = vals[inf_col]
    return np.array([by_date[day] for day in dates], dtype=float)


def read_seeds(rules: list[dict]) -> dict[int, dict]:
    seeds = {}
    for rule in rules:
        if rule.get("name") != "ExposeToMutation":
            continue
        variant = int(rule.get("threshold2", -1))
        if variant in seeds or int(rule["threshold"]) > 1000:
            continue
        seeds[variant] = {
            "day": int(rule["threshold"]),
            "fraction": float(rule["parameter"]),
            "openAfter": int(rule.get("openAfter", 6)),
        }
    return seeds


def apply_seeds(rules: list[dict], seeds: dict[int, dict]) -> None:
    seen = set()
    for rule in rules:
        if rule.get("name") != "ExposeToMutation":
            continue
        variant = int(rule.get("threshold2", -1))
        if variant not in seeds or variant in seen or int(rule["threshold"]) > 1000:
            continue
        rule["threshold"] = int(seeds[variant]["day"])
        rule["parameter"] = float(seeds[variant]["fraction"])
        seen.add(variant)


def clone(params: dict) -> dict:
    return {
        "inf": [float(x) for x in params["inf"]],
        "prog": [float(x) for x in params["prog"]],
        "k": float(params["k"]),
        "seeds": {int(v): dict(s) for v, s in params["seeds"].items()},
    }


def axes_for(wave: tuple) -> list[str]:
    names = ["inf", "prog"]
    if wave[4] is not None:
        names.extend(["day", "frac"])
    return names


def clip_axis(kind: str, value: float, origin: float) -> float:
    if kind == "inf":
        lo, hi = max(0.4, origin * 0.55), min(8.0, origin * 1.8)
        return float(min(hi, max(lo, value)))
    if kind == "prog":
        lo, hi = max(0.25, origin * 0.55), min(2.2, origin * 1.8)
        return float(min(hi, max(lo, value)))
    if kind == "day":
        lo, hi = max(0, origin - 28), origin + 28
        return float(int(min(hi, max(lo, round(value)))))
    if kind == "frac":
        lo, hi = max(1e-5, origin / 4), min(8e-4, origin * 4)
        return float(min(hi, max(lo, value)))
    raise KeyError(kind)


def step_params(params: dict, wave: tuple, kind: str, sign: int) -> dict | None:
    nxt = clone(params)
    index = wave[3]
    variant = wave[4]
    if kind == "inf":
        origin = params["inf"][index]
        nxt["inf"][index] = clip_axis("inf", origin * (1.20 if sign > 0 else 1 / 1.20), origin)
    elif kind == "prog":
        origin = params["prog"][index]
        nxt["prog"][index] = clip_axis("prog", origin * (1.20 if sign > 0 else 1 / 1.20), origin)
    elif kind == "day":
        origin = params["seeds"][variant]["day"]
        nxt["seeds"][variant]["day"] = int(clip_axis("day", origin + sign * 12, origin))
    elif kind == "frac":
        origin = params["seeds"][variant]["fraction"]
        factor = 1.6 if sign > 0 else 1 / 1.6
        nxt["seeds"][variant]["fraction"] = clip_axis("frac", origin * factor, origin)
    else:
        raise KeyError(kind)
    if cache_token(nxt) == cache_token(params):
        return None
    return nxt


def coordinate_candidates(params: dict, wave: tuple) -> list[dict]:
    found = []
    seen = {cache_token(params)}
    for kind in axes_for(wave):
        for sign in (1, -1):
            nxt = step_params(params, wave, kind, sign)
            if nxt is None:
                continue
            token = cache_token(nxt)
            if token in seen:
                continue
            seen.add(token)
            found.append(nxt)
    return found


def cache_token(params: dict) -> str:
    seeds = {str(v): {"day": int(s["day"]), "fraction": round(float(s["fraction"]), 8)} for v, s in sorted(params["seeds"].items())}
    blob = json.dumps(
        {
            "inf": [round(float(x), 5) for x in params["inf"]],
            "prog": [round(float(x), 5) for x in params["prog"]],
            "k": round(float(params["k"]), 8),
            "seeds": seeds,
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def write_closure(params: dict, dest: Path) -> None:
    rules = json.loads(CLOSURE_SRC.read_text())
    apply_seeds(rules["rules"], params["seeds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rules, indent=2))


def run_once(params: dict, weeks: int, dest: Path, closure: Path) -> None:
    cmd = [
        str(BIN),
        "-r",
        "--quarantinePolicy",
        str(QUARANTINE),
        "-k",
        str(params["k"]),
        "--progression",
        "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "-A",
        "inputConfigFiles/agentTypes_3.json",
        "-a",
        "inputRealExample/agents1.json",
        "-l",
        "inputRealExample/locations0.json",
        "--infectiousnessMultiplier",
        ",".join(str(x) for x in params["inf"]),
        "--diseaseProgressionScaling",
        ",".join(str(x) for x in params["prog"]),
        "--closures",
        str(closure),
        "-w",
        str(weeks),
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with dest.open("w") as stdout:
            subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.DEVNULL, check=True, timeout=max(180, weeks * 4 + 90))
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    print(f"  {dest.parent.name}/{dest.name} {time.time() - t0:.0f}s", flush=True)


def series_hospital_infected(text: str, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    run = parse_text(text)
    if run is None or run.n_days < n_days:
        raise RuntimeError(f"short run: {None if run is None else run.n_days} need {n_days}")
    pop = float(np.median(np.sum([run.data[c][:n_days] for c in STOCK], axis=0)))
    scale = NP / pop
    hosp = (run.data["I5_h"][:n_days] + run.data["I6_h"][:n_days] + run.data["R_h"][:n_days]) * scale
    infected = sum(run.data[c][:n_days] for c in PREVALENCE) * scale
    return hosp, infected


def normalized_loss(h_rmse: float, i_rmse: float, h_scale: float, i_scale: float, hospital_weight: float) -> dict:
    h_norm = h_rmse / h_scale
    i_norm = i_rmse / i_scale
    loss = hospital_weight * h_norm + (1.0 - hospital_weight) * i_norm
    return {"h_norm": float(h_norm), "i_norm": float(i_norm), "loss": float(loss)}


def score_paths(paths: list[Path], window: dict, hospital_weight: float) -> dict:
    n_days = window["end_day"] + 1
    hosp, infected = [], []
    for path in paths:
        h, inf = series_hospital_infected(path.read_text(errors="replace"), n_days)
        hosp.append(h)
        infected.append(inf)
    hm = np.vstack(hosp).mean(axis=0)
    im = np.vstack(infected).mean(axis=0)
    w0, w1 = window["w0"], window["w1"]
    hm_w = hm[w0:w1]
    im_w = im[w0:w1]
    gt_h = window["gt_h"]
    gt_i = window["gt_i"]
    finite = window["finite"]
    he = (hm_w - gt_h)[finite]
    ie = im_w - gt_i
    h_rmse = float(np.sqrt(np.mean(he**2)))
    i_rmse = float(np.sqrt(np.mean(ie**2)))
    scored = {
        "n": len(paths),
        "h_rmse": h_rmse,
        "i_rmse": i_rmse,
        "h_peak": float(hm_w[int(np.argmax(hm_w))]),
        "h_peak_day": w0 + int(np.argmax(hm_w)),
        "i_peak": float(im_w[int(np.argmax(im_w))]),
        "i_peak_day": w0 + int(np.argmax(im_w)),
    }
    scored.update(normalized_loss(h_rmse, i_rmse, window["h_scale"], window["i_scale"], hospital_weight))

    def corr(a, b, mask):
        if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
            return float("nan")
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    scored["h_corr"] = corr(hm_w, gt_h, finite)
    scored["i_corr"] = corr(im_w, gt_i, np.isfinite(gt_i))
    return scored


def ensure_runs(params: dict, wave_name: str, weeks: int, n: int) -> list[Path]:
    folder = RUN_ROOT / wave_name / f"w{weeks}_{cache_token(params)}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(params, indent=2))
    closure = folder / "closures.json"
    write_closure(params, closure)
    have = sorted(folder.glob("run*.stdout"))
    while len(have) < n:
        dest = folder / f"run{len(have):02d}.stdout"
        run_once(params, weeks, dest, closure)
        have = sorted(folder.glob("run*.stdout"))
    return have[:n]


def fmt_trial(wave_name: str, params: dict, index: int, scored: dict) -> str:
    variant = index
    seed = ""
    if variant in params["seeds"]:
        s = params["seeds"][variant]
        seed = f" day {s['day']} frac {s['fraction']:.3g}"
    return (
        f"{wave_name} inf {params['inf'][index]:.3f} prog {params['prog'][index]:.3f}{seed} "
        f"loss {scored['loss']:.3f} (h {scored['h_norm']:.3f} i {scored['i_norm']:.3f}) "
        f"RMSE h {scored['h_rmse']:.0f} i {scored['i_rmse']:.0f} "
        f"peak h {scored['h_peak']:.0f}@{scored['h_peak_day']} i {scored['i_peak']:.0f}@{scored['i_peak_day']} "
        f"n={scored['n']}"
    )


def tuned_vector(params: dict, wave: tuple) -> np.ndarray:
    index = wave[3]
    values = [params["inf"][index], params["prog"][index]]
    if wave[4] is not None:
        seed = params["seeds"][wave[4]]
        values.extend([seed["day"], math.log10(seed["fraction"])])
    return np.array(values, dtype=float)


def propose_ei(trials: list[dict], wave: tuple, rng: np.random.Generator) -> dict:
    base = trials[0]["params"]
    lo = []
    hi = []
    index = wave[3]
    lo.append(clip_axis("inf", base["inf"][index] * 0.55, base["inf"][index]))
    hi.append(clip_axis("inf", base["inf"][index] * 1.8, base["inf"][index]))
    lo.append(clip_axis("prog", base["prog"][index] * 0.55, base["prog"][index]))
    hi.append(clip_axis("prog", base["prog"][index] * 1.8, base["prog"][index]))
    if wave[4] is not None:
        origin_day = base["seeds"][wave[4]]["day"]
        origin_frac = base["seeds"][wave[4]]["fraction"]
        lo.extend([clip_axis("day", origin_day - 28, origin_day), math.log10(clip_axis("frac", origin_frac / 4, origin_frac))])
        hi.extend([clip_axis("day", origin_day + 28, origin_day), math.log10(clip_axis("frac", origin_frac * 4, origin_frac))])
    lo_a = np.array(lo, dtype=float)
    hi_a = np.array(hi, dtype=float)
    span = np.maximum(hi_a - lo_a, 1e-6)
    xs = np.vstack([(tuned_vector(t["params"], wave) - lo_a) / span for t in trials])
    y = np.array([t["scored"]["loss"] for t in trials], dtype=float)
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    yn = (y - y_mean) / y_std
    length = np.full(xs.shape[1], 0.45)
    diff = (xs[:, None, :] - xs[None, :, :]) / length
    kernel = np.exp(-0.5 * np.sum(diff * diff, axis=-1))
    nugget = 0.08**2
    chol = np.linalg.cholesky(kernel + np.eye(len(trials)) * (nugget + 1e-8))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yn))
    draw = lo_a + rng.random((2000, len(lo_a))) * span
    known = {cache_token(t["params"]) for t in trials}
    best_ei = -1.0
    best_raw = draw[0]
    y_best = float(np.min(y))
    for row in draw:
        xn = ((row - lo_a) / span)[None, :]
        diff_s = (xs - xn) / length
        k_s = np.exp(-0.5 * np.sum(diff_s * diff_s, axis=1))
        mu = y_mean + y_std * float(k_s @ alpha)
        solved = np.linalg.solve(chol, k_s)
        std = y_std * math.sqrt(max(1.0 - float(solved @ solved), 1e-8))
        z = (y_best - mu) / max(std, 1e-9)
        ei = std * (z * 0.5 * (1 + math.erf(z / math.sqrt(2))) + math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))
        if ei > best_ei:
            best_ei = ei
            best_raw = row
    nxt = clone(base)
    nxt["inf"][index] = clip_axis("inf", float(best_raw[0]), base["inf"][index])
    nxt["prog"][index] = clip_axis("prog", float(best_raw[1]), base["prog"][index])
    if wave[4] is not None:
        nxt["seeds"][wave[4]]["day"] = int(clip_axis("day", float(best_raw[2]), base["seeds"][wave[4]]["day"]))
        nxt["seeds"][wave[4]]["fraction"] = clip_axis("frac", 10 ** float(best_raw[3]), base["seeds"][wave[4]]["fraction"])
    if cache_token(nxt) in known:
        return coordinate_candidates(base, wave)[len(trials) % max(1, len(coordinate_candidates(base, wave)))]
    return nxt


def build_windows(gt_h: np.ndarray, gt_i: np.ndarray) -> dict[str, dict]:
    windows = {}
    for wave in WAVES:
        name, start, end, _index, _variant = wave
        w0 = day_index(start)
        end_day = day_index(end)
        sl = slice(w0, end_day + 1)
        hosp = gt_h[sl]
        infected = gt_i[sl]
        finite = np.isfinite(hosp)
        h_scale = float(np.std(hosp[finite])) or 1.0
        i_scale = float(np.std(infected[np.isfinite(infected)])) or 1.0
        windows[name] = {
            "w0": w0,
            "w1": end_day + 1,
            "end_day": end_day,
            "weeks": weeks_for(end_day),
            "gt_h": hosp,
            "gt_i": infected,
            "finite": finite,
            "h_scale": h_scale,
            "i_scale": i_scale,
            "gt_h_peak": float(np.nanmax(hosp)),
            "gt_h_peak_day": w0 + int(np.nanargmax(hosp)),
            "gt_i_peak": float(np.nanmax(infected)),
            "gt_i_peak_day": w0 + int(np.nanargmax(infected)),
        }
    return windows


def evaluate(params: dict, wave: tuple, window: dict, n: int, hospital_weight: float) -> dict:
    paths = ensure_runs(params, wave[0], window["weeks"], n)
    scored = score_paths(paths, window, hospital_weight)
    print(" ", fmt_trial(wave[0], params, wave[3], scored), flush=True)
    return {"params": clone(params), "scored": scored}


def fit_wave(params: dict, wave: tuple, window: dict, args, rng: np.random.Generator) -> tuple[dict, dict]:
    print(
        f"WAVE {wave[0]} days {window['w0']}..{window['end_day']} weeks {window['weeks']} "
        f"scales h {window['h_scale']:.0f} i {window['i_scale']:.0f} "
        f"targets peak h {window['gt_h_peak']:.0f}@{window['gt_h_peak_day']} "
        f"i {window['gt_i_peak']:.0f}@{window['gt_i_peak_day']}",
        flush=True,
    )
    trials = [evaluate(params, wave, window, 1, args.hospital_weight)]
    candidates = coordinate_candidates(params, wave)
    n_search = 0
    while n_search < args.search_evals:
        if n_search < len(candidates):
            cand = candidates[n_search]
        else:
            try:
                cand = propose_ei(trials, wave, rng)
            except np.linalg.LinAlgError:
                break
        trials.append(evaluate(cand, wave, window, 1, args.hospital_weight))
        n_search += 1
    best = min(trials, key=lambda t: t["scored"]["loss"])
    finalists = [best]
    if cache_token(best["params"]) != cache_token(trials[0]["params"]):
        finalists.append(trials[0])
    finalized = []
    for trial in finalists:
        print(f"  finalize n={args.finalize_runs}", flush=True)
        finalized.append(evaluate(trial["params"], wave, window, args.finalize_runs, args.hospital_weight))
    winner = min(finalized, key=lambda t: t["scored"]["loss"])
    record = wave_record(wave, window, winner, args)
    print(f"KEEP {fmt_trial(wave[0], winner['params'], wave[3], winner['scored'])}", flush=True)
    return winner["params"], record


def wave_record(wave: tuple, window: dict, winner: dict, args) -> dict:
    index = wave[3]
    variant = wave[4]
    params = winner["params"]
    tuned = {
        "index": index,
        "infectiousness": round(float(params["inf"][index]), 5),
        "progression": round(float(params["prog"][index]), 5),
    }
    if variant is not None:
        tuned["seed"] = params["seeds"][variant]
    scored = {k: v for k, v in winner["scored"].items()}
    return {
        "name": wave[0],
        "start": wave[1],
        "end": wave[2],
        "hospital_weight": args.hospital_weight,
        "h_scale": window["h_scale"],
        "i_scale": window["i_scale"],
        "gt_h_peak": window["gt_h_peak"],
        "gt_h_peak_day": window["gt_h_peak_day"],
        "gt_i_peak": window["gt_i_peak"],
        "gt_i_peak_day": window["gt_i_peak_day"],
        "tuned": tuned,
        "score": scored,
    }


def public_params(params: dict, args, waves: list[dict]) -> dict:
    seeds = [
        {"variant": variant, **seeds}
        for variant, seeds in sorted(params["seeds"].items())
    ]
    return {
        "description": "Incremental wave-by-wave fit. viz/run_ensemble.sh is not updated.",
        "hospital_weight": args.hospital_weight,
        "finalize_runs": args.finalize_runs,
        "search_evals": args.search_evals,
        "k": params["k"],
        "quarantinePolicy": QUARANTINE,
        "infectiousnessMultiplier": [round(float(x), 5) for x in params["inf"]],
        "diseaseProgressionScaling": [round(float(x), 5) for x in params["prog"]],
        "seeds": seeds,
        "waves": waves,
    }


def write_outputs(params: dict, args, waves: list[dict]) -> None:
    payload = public_params(params, args, waves)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    closure_path = args.output.with_name("fitted_closure.json") if args.output.name == "fitted_parameters.json" else args.output.with_suffix(".closure.json")
    if args.output == PARAMS_OUT:
        closure_path = CLOSURE_OUT
    rules = json.loads(CLOSURE_SRC.read_text())
    apply_seeds(rules["rules"], params["seeds"])
    closure_path.write_text(json.dumps(rules, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


def initial_params() -> dict:
    rules = json.loads(CLOSURE_SRC.read_text())["rules"]
    return {
        "inf": list(INITIAL_INF),
        "prog": list(INITIAL_PROG),
        "k": INITIAL_K,
        "seeds": read_seeds(rules),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit panSim incrementally, one wave at a time.")
    parser.add_argument("--finalize-runs", type=int, default=3, help="Stochastic runs used to accept the wave's parameters.")
    parser.add_argument("--search-evals", type=int, default=4, help="New one-run trials per wave before the finalize step. 4 covers both directions of infectiousness and progression; 8 also moves the seed.")
    parser.add_argument("--hospital-weight", type=float, default=0.75, help="Weight on normalized hospital RMSE. Infected gets the rest.")
    parser.add_argument("--from-wave", default=None)
    parser.add_argument("--through-wave", default=None)
    parser.add_argument("--output", type=Path, default=PARAMS_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.hospital_weight < 1.0:
        parser.error("--hospital-weight must be between 0 and 1")
    if args.finalize_runs < 1 or args.search_evals < 0:
        parser.error("run counts must be non-negative, and finalize-runs at least 1")

    gt = load_ground_truth(ROOT / "korona_hun.xlsx")
    mask = (gt["date"] >= START) & (gt["date"] <= LAST_DAY)
    dates = gt["date"][mask]
    if len(dates) != day_index("2022-08-31") + 1:
        raise RuntimeError(f"unexpected date span {len(dates)}")
    gt_i = load_infected(dates)
    windows = build_windows(gt["hospital"][mask], gt_i)
    names = [wave[0] for wave in WAVES]
    start_at = names.index(args.from_wave) if args.from_wave else 0
    stop_at = names.index(args.through_wave) if args.through_wave else len(names) - 1

    if args.dry_run:
        for wave in WAVES[start_at : stop_at + 1]:
            window = windows[wave[0]]
            print(
                f"{wave[0]}: {wave[1]}..{wave[2]} weeks {window['weeks']} "
                f"h_scale {window['h_scale']:.1f} i_scale {window['i_scale']:.1f} "
                f"axes {','.join(axes_for(wave))}"
            )
        return

    params = initial_params()
    records: list[dict] = []
    if args.output.exists() and args.from_wave:
        previous = json.loads(args.output.read_text())
        params["inf"] = [float(x) for x in previous["infectiousnessMultiplier"]]
        params["prog"] = [float(x) for x in previous["diseaseProgressionScaling"]]
        params["k"] = float(previous["k"])
        params["seeds"] = {int(item["variant"]): {k: item[k] for k in ("day", "fraction", "openAfter")} for item in previous["seeds"]}
        records = [row for row in previous["waves"] if names.index(row["name"]) < start_at]

    rng = np.random.default_rng(0)
    for wave in WAVES[start_at : stop_at + 1]:
        params, record = fit_wave(params, wave, windows[wave[0]], args, rng)
        records = [row for row in records if row["name"] != wave[0]]
        records.append(record)
        records.sort(key=lambda row: names.index(row["name"]))
        write_outputs(params, args, records)


if __name__ == "__main__":
    main()
