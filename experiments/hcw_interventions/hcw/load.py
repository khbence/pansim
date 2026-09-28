"""Load panSim output written by scripts/run_20_sims.sh."""
from pathlib import Path

import pandas as pd


def load_scenario(results_dir: Path) -> pd.DataFrame:
    """all_runs.tsv -> DataFrame with columns run, day, <panSim columns>.

    panSim prints one row per simulated day (at midnight). Some columns hold
    comma-separated lists (e.g. MUT, INFV), so they stay strings.
    """
    df = pd.read_csv(results_dir / "all_runs.tsv", sep="\t", index_col=False)
    df["day"] = df.groupby("run").cumcount()
    return df


def load_all(results_root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in results_root.iterdir() if (p / "all_runs.tsv").exists()):
        df = load_scenario(d)
        df.insert(0, "scenario", d.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
