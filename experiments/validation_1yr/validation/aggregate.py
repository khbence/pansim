"""Collapse the 20 runs into one aggregated series per target."""
import pandas as pd

from .normalize import apply_transform


def target_per_run(sim: pd.DataFrame, cols, transform: str) -> pd.DataFrame:
    """Wide frame: index = date, one column per run, values = the (transformed) target."""
    total = sim[cols].sum(axis=1)
    wide = total.groupby([sim["date"], sim["run"]]).first().unstack("run").sort_index()
    return wide.apply(lambda c: apply_transform(c, transform))


def aggregate_runs(wide: pd.DataFrame) -> pd.DataFrame:
    """Per date: mean over runs (the aggregated number), std, 5% and 95% quantiles."""
    return pd.DataFrame(
        {
            "mean": wide.mean(axis=1),
            "std": wide.std(axis=1),
            "q05": wide.quantile(0.05, axis=1),
            "q95": wide.quantile(0.95, axis=1),
        }
    )


def scalar_summary(agg: pd.DataFrame) -> float:
    """One number per target: time-mean of the run-mean series."""
    return float(agg["mean"].mean())
