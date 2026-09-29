"""Load Hungarian COVID-19 statistics, from either koronahun.csv or korona_hun.xlsx."""
from pathlib import Path

import pandas as pd


def _to_number(s: pd.Series) -> pd.Series:
    """'11 157,43' -> 11157.43, '3,80%' -> 3.8, '' -> NaN."""
    s = (
        s.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")


def _finish(dates: pd.Series, values: dict, start_date, n_days: int, source) -> pd.DataFrame:
    out = pd.DataFrame({"date": dates, **values})
    out = out.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()

    idx = pd.date_range(pd.Timestamp(start_date), periods=n_days, freq="D")
    out = out.reindex(idx)
    if out.isna().all(axis=None):
        raise ValueError(f"No real data found in {source} for {idx[0].date()} .. {idx[-1].date()}")
    return out


def load_real(csv: Path, start_date, n_days: int, columns: dict, encoding="cp1250", sep=";") -> pd.DataFrame:
    """Return a daily DataFrame indexed by date with one column per name in `columns`.

    `columns` maps a name to the 0-based position of the csv column (headers repeat, so names
    are not reliable). Days without data (e.g. blank rows) stay NaN.
    """
    raw = pd.read_csv(csv, sep=sep, encoding=encoding, header=0, dtype=str)
    dates = pd.to_datetime(raw.iloc[:, 0], format="%Y-%m-%d", errors="coerce")
    values = {name: _to_number(raw.iloc[:, pos]) for name, pos in columns.items()}
    return _finish(dates, values, start_date, n_days, csv)


def load_real_xlsx(xlsx: Path, start_date, n_days: int, columns: dict, sheet="koronahun") -> pd.DataFrame:
    """Same as load_real(), but reads korona_hun.xlsx directly (no cp1250/csv export needed).

    `columns` positions are 0-based in the same layout as the csv (position 0 = date), and the
    sheet's cells are already numeric/datetime, so no comma-decimal cleanup is needed.
    """
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=0)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    values = {name: pd.to_numeric(raw.iloc[:, pos], errors="coerce") for name, pos in columns.items()}
    return _finish(dates, values, start_date, n_days, f"{xlsx}[{sheet}]")
