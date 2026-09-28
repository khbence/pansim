"""Per-run HCW metrics.

HCI = % of HCWs infected or quarantined on that day (prevalence, integer-truncated).
HCE = HCWs newly infected in the last 24h (daily incidence).
"""
import pandas as pd


def per_run(df: pd.DataFrame, hci_threshold: float = 5.0) -> pd.DataFrame:
    g = df.groupby(["scenario", "run"])
    out = g.agg(
        peak_hci=("HCI", "max"),
        peak_day=("HCI", "idxmax"),
        hcw_infections=("HCE", "sum"),
        days_above_thr=("HCI", lambda s: int((s > hci_threshold).sum())),
    )
    out["peak_day"] = df.loc[out["peak_day"], "day"].to_numpy()
    return out.reset_index()


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["peak_hci", "peak_day", "hcw_infections", "days_above_thr"]
    s = runs.groupby("scenario")[cols].agg(["mean", "std"])
    base = runs[runs.scenario == "baseline"][cols].mean()
    for c in cols:
        s[(c, "delta_vs_baseline")] = s[(c, "mean")] - base[c]
    return s.sort_index(axis=1)
