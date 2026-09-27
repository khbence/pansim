#!/usr/bin/env python3
"""Bayesian optimization of the delta (variant 2) wave.

Search variables are the second ExposeToMutation (start day and daily fraction)
and the third entries of infectiousnessMultiplier and diseaseProgressionScaling.
Wild-type and alpha multipliers, later variants, k, and quarantine policy stay fixed.

The loss is the hospital RMSE and the reconstruction-infected RMSE on 5 Aug 2021
through 11 Jan 2022, plus penalties when the hospital crest is late or either
peak height is far from the official / reconstructed crest. Each new proposal is
one stochastic run while that loss is still large. A proposal that is close on
timing, peak height, and both curves is repeated so the mean uses 3–5 runs.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ireguly/pansim/viz")
from pansimviz.ground_truth import load_ground_truth
from pansimviz.parse import parse_text

ROOT = Path("/home/ireguly/pansim")
BIN = ROOT / "build_gpu" / "panSim"
CLOSURE_SRC = ROOT / "inputConfigFiles/closureJun6_real_later3_delta_omicronBA2_earlier8.json"
OUT = Path("/tmp/pansim-fit/wave3")
NP = 9_967_304
START = np.datetime64("2020-09-23")
WAVE_START = np.datetime64("2021-08-05")
WAVE_END = np.datetime64("2022-01-11")
INF_HEAD = [0.98, 1.85]
PROG_HEAD = [0.94, 1.03]
INF_TAIL = [2.58, 4.32, 6.8, 6.8]
PROG_TAIL = [0.72, 0.57, 0.463, 0.45]
STOCK = ("S", "E", "I1", "I2", "I3", "I4", "I5_h", "I6_h", "R_h", "R", "D1", "D2")
PREVALENCE = ("E", "I1", "I2", "I3", "I4", "I5_h", "I6_h")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = datetime(1899, 12, 30)
WEEKS = 69
VARIANT = 2
OPEN_AFTER = 6
# Tags that set progression but not infectiousness were the 2.35 hospital-matched pair.
KNOWN_INF = {"d330_p066": 2.35, "d330_p068": 2.35}
# Tags that set infectiousness and leave progression unnamed used the config default.
DEFAULT_PROG = 0.813

# Proposal box. Day starts at 330 so the search moves the seed later than 5 Aug.
PROPOSE_LO = np.array([330.0, -5.15, 1.75, 0.55])  # day, log10(frac), inf, prog
PROPOSE_HI = np.array([385.0, -3.45, 2.55, 1.65])
# Normalization box also covers the earlier runs used as training data.
NORM_LO = np.array([290.0, -5.4, 1.30, 0.50])
NORM_HI = np.array([400.0, -3.30, 2.70, 2.10])

MAX_NEW = 12
MAX_GPU = 20

GT = load_ground_truth(ROOT / "korona_hun.xlsx")
MASK = (GT["date"] >= START) & (GT["date"] <= WAVE_END)
DATES = GT["date"][MASK]
GT_H_ALL = GT["hospital"][MASK]
N_ALL = int(MASK.sum())
W0 = int((WAVE_START - START) / np.timedelta64(1, "D"))
W1 = N_ALL
GT_H = GT_H_ALL[W0:W1]
FINITE = np.isfinite(GT_H)


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch) - 64)
    return n


def load_infected() -> np.ndarray:
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
    return np.array([by_date[day] for day in DATES], dtype=float)


GT_I = load_infected()[W0:W1]
H_PEAK_I = int(np.nanargmax(GT_H))
H_PEAK = float(GT_H[H_PEAK_I])
H_PEAK_DAY = W0 + H_PEAK_I
JAN = int((np.datetime64("2022-01-01") - WAVE_START) / np.timedelta64(1, "D"))
DELTA_I = GT_I.copy()
DELTA_I[JAN:] = -np.inf
I_PEAK_I = int(np.argmax(DELTA_I))
I_PEAK = float(GT_I[I_PEAK_I])
I_PEAK_DAY = W0 + I_PEAK_I
H_SCALE = float(np.std(GT_H[FINITE]))
I_SCALE = float(np.std(GT_I))


def write_closure(day: int, frac: float, dest: Path) -> None:
    rules = json.loads(CLOSURE_SRC.read_text())
    found = False
    for rule in rules["rules"]:
        if rule["name"] == "ExposeToMutation" and int(rule["threshold2"]) == VARIANT and not found:
            rule["threshold"] = int(day)
            rule["parameter"] = float(frac)
            rule["openAfter"] = OPEN_AFTER
            found = True
            break
    if not found:
        raise RuntimeError("ExposeToMutation for variant 2 not found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rules, indent=2))


def run_once(inf2: float, prog2: float, dest: Path, closure: Path) -> None:
    inf = ",".join(str(x) for x in [*INF_HEAD, inf2, *INF_TAIL])
    prog = ",".join(str(x) for x in [*PROG_HEAD, prog2, *PROG_TAIL])
    cmd = [
        str(BIN),
        "-r",
        "--quarantinePolicy",
        "3",
        "-k",
        "0.00041",
        "--progression",
        "inputConfigFiles/progressions_Jun17_tune/transition_config.json",
        "-A",
        "inputConfigFiles/agentTypes_3.json",
        "-a",
        "inputRealExample/agents1.json",
        "-l",
        "inputRealExample/locations0.json",
        "--infectiousnessMultiplier",
        inf,
        "--diseaseProgressionScaling",
        prog,
        "--closures",
        str(closure),
        "-w",
        str(WEEKS),
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with dest.open("w") as stdout:
        subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.DEVNULL, check=True, timeout=420)
    print(f"  {dest.parent.name}/{dest.name} {time.time() - t0:.0f}s", flush=True)


def series(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run = parse_text(text)
    if run is None or run.n_days < N_ALL:
        raise RuntimeError(f"short run: {None if run is None else run.n_days} need {N_ALL}")
    pop = float(np.median(np.sum([run.data[c][:N_ALL] for c in STOCK], axis=0)))
    scale = NP / pop
    hosp = (run.data["I5_h"][:N_ALL] + run.data["I6_h"][:N_ALL] + run.data["R_h"][:N_ALL]) * scale
    infected = sum(run.data[c][:N_ALL] for c in PREVALENCE) * scale
    muts = []
    for name in ("MUT1", "MUT2", "MUT3"):
        muts.append(run.data[name][:N_ALL] if name in run.data else np.full(N_ALL, np.nan))
    return hosp[W0:W1], infected[W0:W1], np.vstack(muts)[:, W0:W1]


def score_files(paths: list[Path]) -> dict:
    hosp, infected, mut = [], [], []
    for path in paths:
        h, inf, m = series(path.read_text(errors="replace"))
        hosp.append(h)
        infected.append(inf)
        mut.append(m)
    hm = np.vstack(hosp).mean(axis=0)
    im = np.vstack(infected).mean(axis=0)
    mm = np.mean(np.stack(mut, axis=0), axis=0)
    he = (hm - GT_H)[FINITE]
    ie = im - GT_I
    hp = int(np.argmax(hm))
    ip = int(np.argmax(im[:JAN]))

    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    scored = {
        "n": len(paths),
        "h_rmse": float(np.sqrt(np.mean(he**2))),
        "h_corr": corr(hm[FINITE], GT_H[FINITE]),
        "h_peak": float(hm[hp]),
        "h_peak_day": W0 + hp,
        "h_end": float(hm[-1]),
        "i_rmse": float(np.sqrt(np.mean(ie**2))),
        "i_corr": corr(im, GT_I),
        "i_peak": float(im[ip]),
        "i_peak_day": W0 + ip,
        "i_end": float(im[-1]),
        "mut2_at_hpeak": float(mm[1, hp]),
    }
    scored.update(objective(scored))
    return scored


def objective(s: dict) -> dict:
    """Lower is better. Hospital and infected curve error are on the same scale."""
    h_term = s["h_rmse"] / H_SCALE
    i_term = s["i_rmse"] / I_SCALE
    day_err = abs(s["h_peak_day"] - H_PEAK_DAY)
    t_term = day_err / 15.0 + (max(0, day_err - 14) / 10.0) ** 2
    h_ratio = s["h_peak"] / H_PEAK
    i_ratio = s["i_peak"] / I_PEAK
    h_ex = max(0.0, h_ratio - 1.35) ** 2 + max(0.0, 0.70 - h_ratio) ** 2
    i_ex = max(0.0, i_ratio - 1.50) ** 2 + max(0.0, 0.60 - i_ratio) ** 2
    mut_pen = 0.0 if s["mut2_at_hpeak"] >= 50 else 1.5
    total = h_term + i_term + 0.8 * t_term + h_ex + i_ex + mut_pen
    return {
        "loss": float(total),
        "h_term": float(h_term),
        "i_term": float(i_term),
        "t_term": float(t_term),
        "h_ex": float(h_ex),
        "i_ex": float(i_ex),
    }


def tier(s: dict) -> str:
    """bad: 1 run. closer: 2 runs. close: 3 new runs, and 5 on the leader."""
    h_ratio = s["h_peak"] / H_PEAK
    i_ratio = s["i_peak"] / I_PEAK
    day_err = abs(s["h_peak_day"] - H_PEAK_DAY)
    close = (
        day_err <= 10
        and 0.75 <= h_ratio <= 1.30
        and 0.65 <= i_ratio <= 1.50
        and s["h_corr"] >= 0.90
        and s["i_corr"] >= 0.75
        and s["i_peak_day"] <= I_PEAK_DAY + 20
        and s["mut2_at_hpeak"] >= 60
    )
    closer = (
        day_err <= 18
        and 0.55 <= h_ratio <= 1.60
        and 0.50 <= i_ratio <= 1.80
        and s["h_corr"] >= 0.75
        and s["mut2_at_hpeak"] >= 40
    )
    if close:
        return "close"
    if closer:
        return "closer"
    return "bad"


def params_from_tag(name: str) -> tuple[float, float] | None:
    inf_m = re.search(r"(?:^|_)i(\d{3})(?=_|$)", name)
    prog_m = re.search(r"(?:^|_)p(\d{3})(?=_|$)", name)
    inf = int(inf_m.group(1)) / 100 if inf_m else KNOWN_INF.get(name)
    if prog_m:
        prog = int(prog_m.group(1)) / 100
    elif inf_m and inf is not None:
        prog = DEFAULT_PROG
    else:
        prog = None
    if inf is None or prog is None:
        return None
    return inf, prog


def closure_seed(path: Path) -> tuple[int, float, int] | None:
    rules = json.loads(path.read_text())["rules"]
    for rule in rules:
        if rule.get("name") == "ExposeToMutation" and int(rule.get("threshold2", -1)) == VARIANT:
            return int(rule["threshold"]), float(rule["parameter"]), int(rule.get("openAfter", 6))
    return None


def trial_from_score(tag: str, inf: float, prog: float, day: int, frac: float, scored: dict) -> dict:
    row = {
        "tag": tag,
        "inf": inf,
        "prog": prog,
        "day": day,
        "frac": frac,
        "tier": tier(scored),
    }
    row.update(scored)
    return row


def load_history() -> list[dict]:
    rows = []
    if not OUT.exists():
        return rows
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():
            continue
        paths = sorted(folder.glob("run*.stdout"))
        if not paths:
            continue
        meta_path = folder / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            inf, prog = float(meta["inf"]), float(meta["prog"])
            day, frac = int(meta["day"]), float(meta["frac"])
        else:
            parsed = params_from_tag(folder.name)
            seed = closure_seed(folder / "closures.json") if (folder / "closures.json").exists() else None
            if parsed is None or seed is None:
                continue
            day, frac, open_after = seed
            if open_after != OPEN_AFTER:
                continue
            inf, prog = parsed
        try:
            scored = score_files(paths)
        except Exception as exc:
            print(f"skip {folder.name}: {exc}", flush=True)
            continue
        rows.append(trial_from_score(folder.name, inf, prog, day, frac, scored))
    return rows


def x_of(row: dict) -> np.ndarray:
    return np.array([row["day"], math.log10(row["frac"]), row["inf"], row["prog"]], dtype=float)


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - NORM_LO) / (NORM_HI - NORM_LO)


def se_kernel(a: np.ndarray, b: np.ndarray, length: np.ndarray, sf2: float) -> np.ndarray:
    diff = (a[:, None, :] - b[None, :, :]) / length
    return sf2 * np.exp(-0.5 * np.sum(diff * diff, axis=-1))


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def fit_gp(rows: list[dict]):
    x = np.vstack([normalize(x_of(r)) for r in rows])
    y = np.array([r["loss"] for r in rows], dtype=float)
    nrep = np.array([max(1, r["n"]) for r in rows], dtype=float)
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    yn = (y - y_mean) / y_std
    best = None
    best_lml = -np.inf
    length_grid = (0.18, 0.35, 0.6, 1.0)
    noise_grid = (0.05, 0.12, 0.25, 0.45)
    sf2_grid = (0.6, 1.2, 2.2)
    # Anisotropic grid is 4^4; keep the three least-sensitive axes tied when n is small.
    for ld in length_grid:
        for lo in length_grid:
            for noise in noise_grid:
                for sf2 in sf2_grid:
                    length = np.array([ld, lo, lo, lo])
                    nugget = (noise**2) / nrep
                    k = se_kernel(x, x, length, sf2) + np.diag(nugget) + 1e-8 * np.eye(len(rows))
                    try:
                        chol = np.linalg.cholesky(k)
                    except np.linalg.LinAlgError:
                        continue
                    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yn))
                    lml = -0.5 * float(yn @ alpha) - float(np.sum(np.log(np.diag(chol))))
                    lml -= 0.5 * len(rows) * math.log(2 * math.pi)
                    if lml > best_lml:
                        best_lml = lml
                        best = (length, noise, sf2, chol, alpha)
    if best is None:
        raise RuntimeError("GP fit failed")
    return {
        "x": x,
        "y_mean": y_mean,
        "y_std": y_std,
        "length": best[0],
        "noise": best[1],
        "sf2": best[2],
        "chol": best[3],
        "alpha": best[4],
        "lml": best_lml,
    }


def predict(model: dict, x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = normalize(x_raw)
    k_s = se_kernel(model["x"], xs, model["length"], model["sf2"])
    mu_n = k_s.T @ model["alpha"]
    solved = np.linalg.solve(model["chol"], k_s)
    var_n = np.maximum(model["sf2"] - np.sum(solved * solved, axis=0), 1e-8)
    mu = model["y_mean"] + model["y_std"] * mu_n
    std = model["y_std"] * np.sqrt(var_n)
    return mu, std


def too_close(candidate: np.ndarray, rows: list[dict]) -> bool:
    for row in rows:
        have = x_of(row)
        if (
            abs(candidate[0] - have[0]) < 0.5
            and abs(candidate[1] - have[1]) < 0.04
            and abs(candidate[2] - have[2]) < 0.025
            and abs(candidate[3] - have[3]) < 0.025
        ):
            return True
    return False


def propose(rows: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    model = fit_gp(rows)
    # Reference for improvement is the best noise-free prediction at the data, not one lucky run.
    train = np.vstack([x_of(r) for r in rows])
    mu_t, _std_t = predict(model, train)
    y_best = float(np.min(mu_t))
    span = PROPOSE_HI - PROPOSE_LO
    draw = PROPOSE_LO + rng.random((6000, 4)) * span
    best_rows = sorted(rows, key=lambda r: r["loss"])[:4]
    local = []
    for row in best_rows:
        center = np.clip(x_of(row), PROPOSE_LO, PROPOSE_HI)
        local.append(center + rng.normal(0, [6, 0.12, 0.08, 0.08], size=(400, 4)))
    cand = np.vstack([draw, *local])
    cand = np.clip(cand, PROPOSE_LO, PROPOSE_HI)
    cand[:, 0] = np.rint(cand[:, 0])
    mu, std = predict(model, cand)
    z = (y_best - mu) / np.maximum(std, 1e-9)
    ei = std * (z * _Phi(z) + _phi(z))
    order = np.argsort(-ei)
    for idx in order[:400]:
        picked = cand[idx]
        if not too_close(picked, rows):
            return picked, float(mu[idx]), float(std[idx])
    idx = int(order[0])
    return cand[idx], float(mu[idx]), float(std[idx])


def tag_for(day: int, frac: float, inf: float, prog: float) -> str:
    return f"bo_d{day}_i{int(round(inf * 100)):03d}_p{int(round(prog * 100)):03d}_f{int(round(frac * 1e6)):04d}"


def runs_needed(incumbent: dict | None) -> int:
    if incumbent is None or tier(incumbent) == "bad":
        return 1
    if tier(incumbent) == "closer":
        return 2
    return 3


def ensure(row_params: dict, n_target: int, gpu: dict) -> dict:
    tag = row_params["tag"]
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "inf": row_params["inf"],
        "prog": row_params["prog"],
        "day": row_params["day"],
        "frac": row_params["frac"],
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    closure = folder / "closures.json"
    write_closure(row_params["day"], row_params["frac"], closure)
    have = sorted(folder.glob("run*.stdout"))
    while len(have) < n_target:
        if gpu["n"] >= MAX_GPU:
            break
        dest = folder / f"run{len(have):02d}.stdout"
        run_once(row_params["inf"], row_params["prog"], dest, closure)
        gpu["n"] += 1
        have = sorted(folder.glob("run*.stdout"))
    scored = score_files(have)
    return trial_from_score(tag, row_params["inf"], row_params["prog"], row_params["day"], row_params["frac"], scored)


def fmt(row: dict) -> str:
    return (
        f"loss {row['loss']:.2f}  hosp RMSE {row['h_rmse']:.0f} corr {row['h_corr']:.3f} "
        f"peak {row['h_peak']:.0f}@{row['h_peak_day']}  "
        f"inf RMSE {row['i_rmse']:.0f} corr {row['i_corr']:.3f} "
        f"peak {row['i_peak']:.0f}@{row['i_peak_day']}  "
        f"mut2 {row['mut2_at_hpeak']:.0f}%  n={row['n']} {row['tier']}  "
        f"d{row['day']} f{row['frac']:.3g} inf {row['inf']:.2f} prog {row['prog']:.2f}  {row['tag']}"
    )


def main() -> None:
    dry = "--dry-run" in sys.argv
    print(
        f"targets hosp peak {H_PEAK:.0f} on day {H_PEAK_DAY}, "
        f"infected peak {I_PEAK:.0f} on day {I_PEAK_DAY}, "
        f"scales {H_SCALE:.0f} and {I_SCALE:.0f}",
        flush=True,
    )
    rows = load_history()
    print(f"loaded {len(rows)} scored settings", flush=True)
    rows.sort(key=lambda r: r["loss"])
    for row in rows[:8]:
        print(" ", fmt(row), flush=True)
    if dry:
        if len(rows) >= 4:
            rng = np.random.default_rng(0)
            picked, mu, std = propose(rows, rng)
            print(
                f"next day {int(picked[0])} frac {10 ** picked[1]:.3g} "
                f"inf {picked[2]:.2f} prog {picked[3]:.2f}  pred {mu:.2f}±{std:.2f}",
                flush=True,
            )
        return

    rng = np.random.default_rng(0)
    gpu = {"n": 0}
    new_points = 0
    log_path = OUT / "bo_log.jsonl"
    while new_points < MAX_NEW and gpu["n"] < MAX_GPU:
        incumbent = min(rows, key=lambda r: r["loss"])
        level = tier(incumbent)
        if level == "closer" and incumbent["n"] < 2 and gpu["n"] < MAX_GPU:
            print(f"replicate closer leader to n=2 ({incumbent['tag']})", flush=True)
            updated = ensure(incumbent, 2, gpu)
            rows = [updated if r["tag"] == updated["tag"] else r for r in rows]
            print(" ", fmt(updated), flush=True)
            continue
        if level == "close" and incumbent["n"] < 5 and gpu["n"] < MAX_GPU:
            target_n = min(5, incumbent["n"] + 2)
            print(f"replicate close leader to n={target_n} ({incumbent['tag']})", flush=True)
            updated = ensure(incumbent, target_n, gpu)
            rows = [updated if r["tag"] == updated["tag"] else r for r in rows]
            print(" ", fmt(updated), flush=True)
            continue
        picked, mu, std = propose(rows, rng)
        day = int(picked[0])
        frac = float(10 ** picked[1])
        inf = float(picked[2])
        prog = float(picked[3])
        n_target = runs_needed(incumbent)
        tag = tag_for(day, frac, inf, prog)
        print(
            f"propose {tag} n={n_target} pred {mu:.2f}±{std:.2f} "
            f"(gpu {gpu['n']}/{MAX_GPU}, new {new_points}/{MAX_NEW})",
            flush=True,
        )
        updated = ensure(
            {"tag": tag, "inf": inf, "prog": prog, "day": day, "frac": frac},
            n_target,
            gpu,
        )
        rows = [r for r in rows if r["tag"] != updated["tag"]]
        rows.append(updated)
        new_points += 1
        with log_path.open("a") as log:
            log.write(json.dumps({k: v for k, v in updated.items() if k != "hm"}) + "\n")
        print(" ", fmt(updated), flush=True)

    best = min(rows, key=lambda r: r["loss"])
    print("BEST", fmt(best), flush=True)
    (OUT / "bo_best.json").write_text(json.dumps({k: v for k, v in best.items()}, indent=2))


if __name__ == "__main__":
    main()
