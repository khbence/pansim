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


def hcw_vs_population(df: pd.DataFrame):
    """Grid of scenario (row) x {prevalence, incidence} (column): HCW vs REST-of-population
    (HCWs excluded, so the two groups never overlap), median + 5-95% band each - is the HCW
    group's infection burden disproportionate to everyone else's, and does that gap change
    between scenarios? (needs metrics.add_population_columns() to have added PI_nonHCW/HCE_rate/
    NI_nonHCW_rate to df already).

    Prevalence (HCI vs PI_nonHCW) is "infected OR quarantined" on both sides, so a quarantine-
    policy change can move it for reasons unrelated to actual infection risk (see
    hcw/metrics.py:add_population_columns). Incidence (HCE_rate vs NI_nonHCW_rate) is the clean
    comparison - both are exact new-infection-per-capita rates, no quarantine involved either
    side - prefer it when the two disagree.
    """
    scenarios = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(13, 3.5 * len(scenarios)), sharex=True)
    axes = axes.reshape(len(scenarios), 2)
    panels = [
        ("HCI", "PI_nonHCW", "infected or quarantined, %", "prevalence"),
        ("HCE_rate", "NI_nonHCW_rate", "new infections/day, % of group", "incidence"),
    ]
    for row, scen in enumerate(scenarios):
        d = df[df["scenario"] == scen]
        for col, (hcw_col, pop_col, ylabel, title) in enumerate(panels):
            ax = axes[row, col]
            for series_col, label, color in [(hcw_col, "healthcare workers", "#2a78d6"),
                                              (pop_col, "rest of population", "#eb6834")]:
                q = d.groupby("day")[series_col].quantile([0.05, 0.5, 0.95]).unstack()
                ax.plot(q.index, q[0.5], label=label, color=color, linewidth=2)
                ax.fill_between(q.index, q[0.05], q[0.95], color=color, alpha=0.2, linewidth=0)
            ax.set_title(f"{scen} - {title}")
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("day")
    fig.tight_layout()
    return fig
