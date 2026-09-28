"""Load koronahun.csv (Hungarian COVID-19 statistics)."""
from pathlib import Path

import pandas as pd


def _to_number(s: pd.Series) -> pd.Series:
    """'11 157,43' -> 11157.43, '3,80%' -> 3.8, '' -> NaN."""
    s = (
        s.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")


def load_real(csv: Path, start_date, n_days: int, columns: dict, encoding="cp1250", sep=";") -> pd.DataFrame:
    """Return a daily DataFrame indexed by date with one column per name in `columns`.

    `columns` maps a name to the 0-based position of the csv column (headers repeat, so names
    are not reliable). Days without data (e.g. blank rows) stay NaN.
    """
    raw = pd.read_csv(csv, sep=sep, encoding=encoding, header=0, dtype=str)
    dates = pd.to_datetime(raw.iloc[:, 0], format="%Y-%m-%d", errors="coerce")
    out = pd.DataFrame({"date": dates})
    for name, pos in columns.items():
        out[name] = _to_number(raw.iloc[:, pos])
    out = out.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()

    idx = pd.date_range(pd.Timestamp(start_date), periods=n_days, freq="D")
    out = out.reindex(idx)
    if out.isna().all(axis=None):
        raise ValueError(f"No real data found in {csv} for {idx[0].date()} .. {idx[-1].date()}")
    return out
