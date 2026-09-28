#!/usr/bin/env python3
"""Score panSim stdout against the BA.1/BA.2 hospital wave through 31 Aug 2022.

Hospital target is I5_h+I6_h+R_h scaled by Np/N_sim, on days with an official count.
Infected target is E+I1+...+I6_h on the same scale, against the reconstruction.
"""

from __future__ import annotations

import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ireguly/pansim/viz")
from pansimviz.ground_truth import load_ground_truth
from pansimviz.parse import parse_text

ROOT = Path("/home/ireguly/pansim")
NP = 9_967_304
START = np.datetime64("2020-09-23")
WAVE_START = np.datetime64("2022-01-12")
WAVE_END = np.datetime64("2022-08-31")
STOCK = ("S", "E", "I1", "I2", "I3", "I4", "I5_h", "I6_h", "R_h", "R", "D1", "D2")
PREVALENCE = ("E", "I1", "I2", "I3", "I4", "I5_h", "I6_h")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = datetime(1899, 12, 30)

GT = load_ground_truth(ROOT / "korona_hun.xlsx")
MASK = (GT["date"] >= START) & (GT["date"] <= WAVE_END)
DATES = GT["date"][MASK]
N_ALL = int(MASK.sum())
W0 = int((WAVE_START - START) / np.timedelta64(1, "D"))
GT_H = GT["hospital"][MASK][W0:]
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


GT_I = load_infected()[W0:]
# The BA.1 crest is the February peak. Later shoulders are scored by the curve, not by this argmax.
FEB_END = int((np.datetime64("2022-03-01") - WAVE_START) / np.timedelta64(1, "D"))
H_PEAK_I = int(np.nanargmax(GT_H[:FEB_END]))
H_PEAK = float(GT_H[H_PEAK_I])
H_PEAK_DAY = W0 + H_PEAK_I
I_PEAK_I = int(np.argmax(GT_I[:FEB_END]))
I_PEAK = float(GT_I[I_PEAK_I])
I_PEAK_DAY = W0 + I_PEAK_I


def series(text: str):
    run = parse_text(text)
    if run is None or run.n_days < N_ALL:
        raise RuntimeError(f"short run: {None if run is None else run.n_days} need {N_ALL}")
    pop = float(np.median(np.sum([run.data[c][:N_ALL] for c in STOCK], axis=0)))
    scale = NP / pop
    hosp = (run.data["I5_h"][:N_ALL] + run.data["I6_h"][:N_ALL] + run.data["R_h"][:N_ALL]) * scale
    infected = sum(run.data[c][:N_ALL] for c in PREVALENCE) * scale
    muts = []
    for name in ("MUT1", "MUT2", "MUT3", "MUT4", "MUT5"):
        muts.append(run.data[name][:N_ALL] if name in run.data else np.full(N_ALL, np.nan))
    return hosp[W0:], infected[W0:], np.vstack(muts)[:, W0:], pop


def score_files(paths: list[Path]) -> dict:
    hosp, infected, mut = [], [], []
    for path in paths:
        h, inf, m, _pop = series(path.read_text(errors="replace"))
        hosp.append(h)
        infected.append(inf)
        mut.append(m)
    hm = np.vstack(hosp).mean(axis=0)
    im = np.vstack(infected).mean(axis=0)
    mm = np.mean(np.stack(mut, axis=0), axis=0)
    he = (hm - GT_H)[FINITE]
    ie = im - GT_I
    # February crest, before the summer rise can take the argmax.
    hp = int(np.argmax(hm[: max(FEB_END, 1)]))
    summer0 = int((np.datetime64("2022-07-01") - WAVE_START) / np.timedelta64(1, "D"))
    sp = summer0 + int(np.argmax(hm[summer0:]))
    ip = int(np.argmax(im[: max(FEB_END, 1)]))

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def at(day: int):
        i = day - W0
        gt_h = float(GT_H[i]) if np.isfinite(GT_H[i]) else float("nan")
        return float(hm[i]), float(im[i]), gt_h, float(GT_I[i]), mm[:, i]

    return {
        "n": len(paths),
        "h_rmse": float(np.sqrt(np.mean(he**2))),
        "h_corr": corr(hm, GT_H),
        "h_peak": float(hm[hp]),
        "h_peak_day": W0 + hp,
        "i_rmse": float(np.sqrt(np.mean(ie**2))),
        "i_corr": corr(im, GT_I),
        "i_peak": float(im[ip]),
        "i_peak_day": W0 + ip,
        "summer_h_peak": float(hm[sp]),
        "summer_h_day": W0 + sp,
        "snapshots": {
            "Feb15": at(510),
            "Mar16": at(539),
            "Jun15": at(630),
            "Aug10": at(686),
            "Aug31": at(707),
        },
        "hm": hm,
        "im": im,
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    print(
        f"window {WAVE_START}..{WAVE_END} days {W0}..{W0 + len(GT_H) - 1} "
        f"hosp peak {H_PEAK:.0f} on day {H_PEAK_DAY} ({START + np.timedelta64(H_PEAK_DAY, 'D')}) "
        f"infected peak {I_PEAK:.0f} on day {I_PEAK_DAY}",
        flush=True,
    )
    if not paths:
        return
    scored = score_files(paths)
    print(
        f"n={scored['n']} hosp RMSE {scored['h_rmse']:.0f} corr {scored['h_corr']:.3f} "
        f"peak {scored['h_peak']:.0f}@{scored['h_peak_day']} "
        f"inf RMSE {scored['i_rmse']:.0f} corr {scored['i_corr']:.3f} "
        f"peak {scored['i_peak']:.0f}@{scored['i_peak_day']} "
        f"summer hosp {scored['summer_h_peak']:.0f}@{scored['summer_h_day']}",
        flush=True,
    )
    for name, (h, inf, gh, gi, mut) in scored["snapshots"].items():
        shares = " ".join(f"M{i+1}:{mut[i]:.0f}" for i in range(len(mut)))
        print(f"  {name}: hosp {h:.0f} vs {gh:.0f}  inf {inf:.0f} vs {gi:.0f}  {shares}", flush=True)


if __name__ == "__main__":
    main()
