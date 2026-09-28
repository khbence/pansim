import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .normalize import peak_normalize


def plot_target(name, agg, real, out_png, start_date):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.fill_between(agg.index, agg["q05"], agg["q95"], alpha=0.25, label="sim 5-95%")
    ax.plot(agg.index, agg["mean"], label="sim mean (20 runs)")
    ax.plot(real.index, real, "k.", ms=3, label="real (rescaled)")
    ax.set_title(f"{name} (counts)")

    ax = axes[1]
    ax.plot(agg.index, peak_normalize(agg["mean"]), label="sim mean")
    ax.plot(real.index, peak_normalize(real), "k.", ms=3, label="real")
    ax.set_title(f"{name} (each series / its own peak)")

    for ax in axes:
        ax.set_xlim(agg.index.min(), agg.index.max())
        ax.set_xlabel(f"date (from {start_date})")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
