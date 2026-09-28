import matplotlib.pyplot as plt
import pandas as pd


def band(df: pd.DataFrame, col: str, ax=None):
    """Median and 5-95% band of `col` over runs, one line per scenario."""
    ax = ax or plt.gca()
    for name, d in df.groupby("scenario"):
        q = d.groupby("day")[col].quantile([0.05, 0.5, 0.95]).unstack()
        ax.plot(q.index, q[0.5], label=name)
        ax.fill_between(q.index, q[0.05], q[0.95], alpha=0.2)
    ax.set_xlabel("day")
    ax.set_ylabel(col)
    ax.legend()
    return ax
