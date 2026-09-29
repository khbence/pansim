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


def combine_runs(sim_dir: Path) -> Path:
    """Rebuild all_runs.tsv from run_XX.tsv (mirrors scripts/run_20_sims.sh's combine step).

    Useful when only the per-run files were copied over (e.g. all_runs.tsv was never written
    because a run crashed before the driver script reached its combine step).
    """
    sim_dir = Path(sim_dir)
    run_files = sorted(sim_dir.glob("run_*.tsv"))
    if not run_files:
        raise FileNotFoundError(f"No run_*.tsv files in {sim_dir}")

    out = sim_dir / "all_runs.tsv"
    with out.open("w") as f_out:
        header = run_files[0].read_text().splitlines()[0]
        f_out.write("run\t" + header + "\n")
        for run_file in run_files:
            run_id = int(run_file.stem.split("_")[1])
            lines = run_file.read_text().splitlines()[1:]  # skip header
            for line in lines:
                f_out.write(f"{run_id}\t{line}\n")
    return out


def load_sim_runs(sim_dir: Path, start_date) -> pd.DataFrame:
    """Load run_XX.tsv files directly (no n_days truncation) and rebuild all_runs.tsv alongside.

    Unlike load_sim(), this does not require all runs to reach the same length: a short/crashed
    run (e.g. missing tail days) simply has no rows for its missing days, so it drops out of the
    per-day average for those days instead of raising.
    """
    all_runs = combine_runs(sim_dir)
    df = pd.read_csv(all_runs, sep="\t", index_col=False)
    df["day"] = df.groupby("run").cumcount()
    df["date"] = pd.Timestamp(start_date) + pd.to_timedelta(df["day"], unit="D")

    full_len = df.groupby("run")["day"].max().max() + 1
    counts = df.groupby("run")["day"].max() + 1
    short = counts[counts < full_len]
    if not short.empty:
        print(f"WARNING: runs shorter than {full_len} days (excluded from later days' average): "
              f"{short.to_dict()}")
    return df
