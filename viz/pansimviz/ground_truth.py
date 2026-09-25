"""National Hungarian COVID statistics from korona_hun.xlsx.

The sheet starts on 2020-03-04, before the simulation (default start 2020-09-23,
day-of-year index 267). Many cells are empty. Counts are for the whole country
(default 9.6 million) and must be scaled before comparison with a 179500-agent run.
Ratios are left unchanged.
"""

from __future__ import annotations

import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = datetime(1899, 12, 30)

# Header text on the koronahun sheet -> internal series name.
# Duplicate headers (several "Hétnapos csúszóátlag" columns) keep the first match only.
GT_HEADERS = {
    "Dátum": "date",
    "Regisztrált esetek száma": "registered",
    "Új esetek száma": "new_cases",
    "Elhunytak": "deaths",
    "Gyógyultak": "recovered",
    "Aktív fertőzöttek száma": "active",
    "Hatósági házi karantén": "quarantine",
    "Új mintavételek száma": "tests",
    "Az új elhunytak száma naponta": "deaths_new",
    "Új gyógyultak naponta": "recovered_new",
    "Kórházi ápoltak száma": "hospital",
    "Lélegeztetőgépen lévők száma": "ventilated",
    "Beoltottak száma Magyarországon": "vacc_cum",
    "Új beoltottak száma Magyarországon": "vacc_new",
    "Pozitív tesztek aránya": "pos_rate",
}

# Counts are scaled. Ratios are not.
GT_KIND = {
    "registered": "count",
    "new_cases": "count",
    "deaths": "count",
    "recovered": "count",
    "active": "count",
    "quarantine": "count",
    "tests": "count",
    "deaths_new": "count",
    "recovered_new": "count",
    "hospital": "count",
    "ventilated": "count",
    "vacc_cum": "count",
    "vacc_new": "count",
    "pos_rate": "ratio",
    # No separate official series for mild hospital occupancy.
    "hospital_mild": "count",
    "positives": "count",
}


def _col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for si in root.findall("m:si", NS):
        strings.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    return strings


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str = "koronahun") -> str:
    wb = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    for sheet in wb.findall(".//m:sheet", NS):
        if sheet.attrib.get("name") == sheet_name:
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = targets[rid]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            return target
    raise ValueError(f"sheet {sheet_name!r} not found")


def load_ground_truth(path: str | Path) -> dict[str, np.ndarray]:
    """Return date (datetime64[D]) plus one float array per series. Gaps are NaN."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        strings = _load_shared_strings(archive)
        root = ET.fromstring(archive.read(_sheet_path(archive)))

    rows: dict[int, dict[int, str]] = {}
    for cell in root.findall(".//m:c", NS):
        ref = cell.attrib.get("r")
        if not ref:
            continue
        value_node = cell.find("m:v", NS)
        if value_node is None or value_node.text is None:
            continue
        raw = value_node.text
        if cell.attrib.get("t") == "s":
            raw = strings[int(raw)]
        row = int("".join(ch for ch in ref if ch.isdigit()))
        rows.setdefault(row, {})[_col_index(ref)] = raw

    if 1 not in rows:
        raise ValueError("ground-truth sheet has no header row")

    header = rows[1]
    col_of: dict[str, int] = {}
    for idx, text in header.items():
        key = GT_HEADERS.get(text)
        if key and key not in col_of:
            col_of[key] = idx
    if "date" not in col_of:
        raise ValueError("ground-truth sheet has no Dátum column")

    dates: list[np.datetime64] = []
    series: dict[str, list[float]] = {k: [] for k in col_of if k != "date"}
    for row_idx in sorted(r for r in rows if r > 1):
        raw_date = rows[row_idx].get(col_of["date"])
        if raw_date is None:
            continue
        try:
            day = EXCEL_EPOCH + timedelta(days=float(raw_date))
        except ValueError:
            continue
        dates.append(np.datetime64(day.date()))
        for key, col in col_of.items():
            if key == "date":
                continue
            raw = rows[row_idx].get(col)
            if raw is None or raw == "":
                series[key].append(np.nan)
                continue
            try:
                series[key].append(float(raw))
            except ValueError:
                series[key].append(np.nan)

    out: dict[str, np.ndarray] = {"date": np.asarray(dates, dtype="datetime64[D]")}
    for key, values in series.items():
        out[key] = np.asarray(values, dtype=float)
    return out


def align_to_dates(
    ground_truth: dict[str, np.ndarray],
    dates: np.ndarray,
    key: str,
    scale: float,
) -> np.ndarray:
    """Sample one ground-truth series onto simulation dates. Missing days stay NaN."""
    aligned = np.full(len(dates), np.nan)
    if key not in ground_truth:
        return aligned
    gt_dates = ground_truth["date"]
    gt_values = ground_truth[key]
    index = {day: i for i, day in enumerate(gt_dates.tolist())}
    for i, day in enumerate(dates.tolist()):
        j = index.get(day)
        if j is None:
            continue
        value = gt_values[j]
        if np.isfinite(value):
            aligned[i] = value * scale
    return aligned
