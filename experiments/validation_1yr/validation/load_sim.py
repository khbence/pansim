"""Load the PanSim runs written by scripts/run_20_sims.sh (all_runs.tsv)."""
from pathlib import Path

import pandas as pd


def load_sim(tsv: Path, start_date, n_days: int) -> pd.DataFrame:
    """Return one row per (run, day) with a `date` column.

    panSim prints one row per simulated day, so the day index is the row number within a run.
    Comma-list columns (MUT, IMM, INFV) stay strings and are not used for validation.
    """
    df = pd.read_csv(tsv, sep="\t", index_col=False)
    df["day"] = df.groupby("run").cumcount()
    df = df[df["day"] < n_days].copy()
    df["date"] = pd.Timestamp(start_date) + pd.to_timedelta(df["day"], unit="D")

    counts = df.groupby("run")["day"].max() + 1
    if (counts < n_days).any():
        print(f"WARNING: runs shorter than {n_days} days: {counts[counts < n_days].to_dict()}")
    return df


def require_columns(df: pd.DataFrame, columns) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not in simulator output. Available: {list(df.columns)}")
