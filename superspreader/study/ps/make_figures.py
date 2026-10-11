#!/usr/bin/env python3
"""Figures F2, F5 and F7 for the ACT-049 report.

The figure-style skill named in the instructions is not on this machine, so
apply_figure_style() below is the local stand-in: one style, applied before
every plot. Titles are claims, and the script raises if the claim is no longer
true of the data.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
OUT = HERE.parent / "figures"
SUMMARY = HERE.parent / "results" / "ps" / "ps1" / "summary.json"

ARM_COLOUR = {
    "A0": "#4d4d4d",
    "E": "#1b4f72",
    "P0": "#1a7a4c",
    "P25": "#b86e00",
    "P50": "#c0392b",
    "P100": "#6c3483",
    "P25k": "#1a5276",
    "P100ln": "#117a65",
    "P50a005": "#7b241c",
    "P100a005": "#4a235a",
}


def apply_figure_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def f2():
    apply_figure_style()
    studies = read_csv(INPUTS / "published_k_from_R7.csv")
    settings = [r for r in read_csv(INPUTS / "published_k_by_setting_R6.csv") if r["estimator"] == "cluster_ascertainment" and r["setting"] != "All settings"]
    ks = []
    for row in studies:
        if row["k"]:
            ks.append(float(row["k"]))
    below = sum(k < 1 for k in ks)
    claim = f"{below} of {len(ks)} study-level SARS-CoV-2 k estimates are below 1"
    assert below == sum(k < 1 for k in ks) and below > len(ks) / 2, claim

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    ax = axes[0]
    for i, row in enumerate(studies):
        if not row["k"]:
            continue
        k = float(row["k"])
        lo = float(row["k_lo"]) if row["k_lo"] else k
        hi = float(row["k_hi"]) if row["k_hi"] else k
        ax.errorbar(k, i, xerr=[[max(k - lo, 0)], [max(hi - k, 0)]], fmt="o", color="#1b4f72", ms=3.5, lw=0.8)
        ax.text(-0.02, i, f"{row['author']} {row['year']}", ha="right", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    for j, row in enumerate(settings):
        y = len(studies) + 1 + j
        k, lo, hi = float(row["k"]), float(row["k_lo"]), float(row["k_hi"])
        ax.errorbar(k, y, xerr=[[k - lo], [hi - k]], fmt="D", color="#b86e00", ms=3.5, lw=0.8)
        ax.text(-0.02, y, row["setting"], ha="right", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    ax.axvline(1, color="#888", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("dispersion k")
    ax.set_yticks([])
    ax.set_title(claim, fontsize=8)

    ax = axes[1]
    # Verified reproduction numbers only. Unverified abstract values are not drawn.
    verified = [
        ("Aditama 2012\nH5N1, full text", 0.1, 0.25),
        ("Ward 2025\nH5N1 US, full text", 0.020, 0.090),
    ]
    # Abstract only. Drawn hollow and excluded from the asserted claim.
    unverified = [
        ("Nikolay 2019\nNipah, all cases, abstract", 0.19, 0.59),
        ("Nikolay 2019\nNipah, age 45+, abstract", 0.4, 3.2),
    ]
    for i, (label, lo, hi) in enumerate(verified + unverified):
        colour = "#6c3483" if i < len(verified) else "#888888"
        marker = "o" if i < len(verified) else "o"
        fill = colour if i < len(verified) else "none"
        ax.plot([lo, hi], [i, i], color=colour, lw=2)
        ax.plot([(lo + hi) / 2], [i], marker, color=colour, markerfacecolor=fill)
        ax.text(0.008, i, label, va="center", ha="right", fontsize=6.5)
    ax.axvline(1, color="#888", lw=0.6)
    ax.set_xlim(0.005, 5)
    ax.set_xscale("log")
    ax.set_yticks([])
    both_below = all(hi < 1 for _, _, hi in verified)
    claim_b = "The verified H5N1 reproduction numbers sit below 1"
    assert both_below, claim_b
    ax.set_title(claim_b, fontsize=8)
    ax.set_xlabel("reproduction number")
    fig.suptitle("Published dispersion and subcritical reproduction numbers", fontsize=11)
    save(fig, "F2_published_k")
    with open(OUT / "F2_published_k.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "label", "k_or_R", "lo", "hi"])
        for row in studies:
            if row["k"]:
                writer.writerow(["A", f"{row['author']} {row['year']}", row["k"], row["k_lo"], row["k_hi"]])
        for row in settings:
            writer.writerow(["A setting", row["setting"], row["k"], row["k_lo"], row["k_hi"]])
        writer.writerow(["B verified", "Aditama 2012 range", "", 0.1, 0.25])
        writer.writerow(["B verified", "Ward 2025 US estimate", 0.046, 0.020, 0.090])
        writer.writerow(["B abstract only", "Nikolay 2019 all cases", 0.33, 0.19, 0.59])
        writer.writerow(["B abstract only", "Nikolay 2019 age 45+", 1.1, 0.4, 3.2])


def f5():
    apply_figure_style()
    summary = json.loads(SUMMARY.read_text())
    order = ["P0", "P25", "P50", "P100"]
    by = {item["arm"]: item for item in summary["arms"]}
    sizes = [by[arm]["final_size"]["mean"] for arm in order]
    early = [by[arm]["early_k"] for arm in order]
    claim = "Final size falls, and early k rises, as coupling goes from none to full"
    assert sizes[0] > sizes[1] > sizes[2] > sizes[3], claim
    assert early[0] < early[1] < early[2] < early[3], claim
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    windows = ["1-10", "11-20", "21-30", "31-40", "41-50"]
    for arm in ["E", "P0", "P25", "P50", "P100", "P100ln"]:
        item = by[arm]
        ys = [c["k_replicate_t"]["mean"] for c in item["cohorts"]]
        lo = [c["k_replicate_t"]["low"] for c in item["cohorts"]]
        hi = [c["k_replicate_t"]["high"] for c in item["cohorts"]]
        axes[0].errorbar(range(5), ys, yerr=[[max(a - b, 0) for a, b in zip(ys, lo)], [max(b - a, 0) for a, b in zip(hi, ys)]],
                         color=ARM_COLOUR[arm], label=arm, ms=3, lw=1, fmt="-o")
    axes[0].set_xticks(range(5), windows, rotation=30)
    axes[0].set_ylabel("cohort k")
    axes[0].legend(fontsize=6, frameon=False)
    for arm in ["P0", "P25", "P50", "P100", "P100ln"]:
        item = by[arm]
        ys = [c["trait_replicate_t"]["mean"] for c in item["cohorts"]]
        axes[1].plot(range(5), ys, "-o", color=ARM_COLOUR[arm], label=arm, ms=3, lw=1)
    axes[1].set_xticks(range(5), windows, rotation=30)
    axes[1].set_ylabel("mean trait of infected cohort")
    axes[1].legend(fontsize=6, frameon=False)
    axes[2].bar(range(len(order)), sizes, color=[ARM_COLOUR[a] for a in order])
    axes[2].set_xticks(range(len(order)), order)
    axes[2].set_ylabel("final size, ever infected")
    fig.suptitle(claim, fontsize=11)
    save(fig, "F5_ps1")
    with open(OUT / "F5_ps1.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "window", "k_mean", "k_low", "k_high", "trait_mean", "final_size"])
        for arm, item in by.items():
            for c in item["cohorts"]:
                writer.writerow([arm, f"{c['window'][0]}-{c['window'][1]}", c["k_replicate_t"]["mean"],
                                 c["k_replicate_t"]["low"], c["k_replicate_t"]["high"],
                                 c["trait_replicate_t"]["mean"], item["final_size"]["mean"]])


def f7():
    apply_figure_style()
    # The five arms transcribed in 14 section 6. Kept here so the figure does not depend on Drive.
    rows = [
        ("Set II strength 8", 0.15, 75.3, 34.2, 78.8),
        ("Set II strength 4", 0.21, 75.4, 37.6, 79.5),
        ("Set II strength 2", 0.28, 68.3, 41.5, 75.6),
        ("Set I shape 1", 0.42, 37.2, 53.3, 73.6),
        ("volume model E1", 0.53, 26.8, 40.9, 66.1),
    ]
    shares = [r[1] for r in rows]
    closure = [r[2] for r in rows]
    # One rank inversion: 75.3 at share 0.15 against 75.4 at share 0.21.
    inversion = closure[0] < closure[1] and shares[0] < shares[1]
    rest_falls = closure[1] > closure[2] > closure[3] > closure[4]
    claim = "Venue closure falls with household share, with one rank inversion"
    assert inversion and rest_falls, claim
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(shares, closure, "o-", color="#1b4f72", label="venue closure")
    ax.plot(shares, [r[3] for r in rows], "s-", color="#b86e00", label="household isolation")
    ax.plot(shares, [r[4] for r in rows], "^-", color="#1a7a4c", label="masks")
    for share, close, label in zip(shares, closure, [r[0] for r in rows]):
        ax.annotate(label, (share, close), textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set_xlabel("household share of infections")
    ax.set_ylabel("reduction, percent")
    ax.set_title(claim + "\nn = 5, shares from different experiments", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "F7_lever_household")
    with open(OUT / "F7_lever_household.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "household_share", "venue_closure_pct", "household_isolation_pct", "masks_pct"])
        writer.writerows(rows)


def composition_k(shape, kb=5.25):
    return 1.0 / (1.0 / kb + 1.0 / shape + 1.0 / (kb * shape))


def f3():
    apply_figure_style()
    summary = json.loads(SUMMARY.read_text())
    by = {item["arm"]: item for item in summary["arms"]}
    # Uncoupled ladder from the phase 2 grid, confirmed against summaries/p2_grid.json.
    ladder = [(1.0, 0.884), (0.3, by["E"]["early_k"]), (0.1, 0.127)]
    comp = composition_k(0.3)
    claim = "Uncoupled early k at shape 0.3 sits inside the 20 to 50 percent band above the composition rule"
    assert 1.2 * comp <= by["E"]["early_k"] <= 1.5 * comp, claim
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    shapes = [s / 50 for s in range(2, 150)]
    ax.plot(shapes, [composition_k(s) for s in shapes], color="#888", lw=1, label="composition rule")
    ax.fill_between(shapes, [1.2 * composition_k(s) for s in shapes], [1.5 * composition_k(s) for s in shapes], color="#bbbbbb", alpha=0.4, label="20 to 50 percent above")
    ax.plot([s for s, _ in ladder], [k for _, k in ladder], "o", color="#1b4f72", label="uncoupled")
    for arm in ("P0", "P25", "P50", "P100", "P100ln"):
        ax.plot([0.3], [by[arm]["early_k"]], "D", color=ARM_COLOUR[arm], label=arm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("multiplier or trait shape")
    ax.set_ylabel("early k")
    ax.set_title(claim, fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    save(fig, "F3_early_k")
    with open(OUT / "F3_early_k.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "shape", "early_k"])
        for shape, k in ladder:
            writer.writerow(["uncoupled", shape, k])
        for arm in ("P0", "P25", "P50", "P100", "P100ln"):
            writer.writerow([arm, 0.3, by[arm]["early_k"]])


def f6():
    apply_figure_style()
    summary = json.loads((HERE.parent / "results" / "ps" / "ps2" / "summary.json").read_text())
    claim = "Household isolation reduces more infections than isolating the diagnosed person, under both multipliers"
    for arm in ("C", "Pk50"):
        iso = summary["reductions"][arm]["isolate"]["reduction"]["mean"]
        house = summary["reductions"][arm]["household"]["reduction"]["mean"]
        assert house > iso + 0.2, claim
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    # Generation-interval quantiles, which is what the summary stores. The density is in the CSV of raw means.
    for arm, xpos in (("A0", 0), ("C", 1), ("Pk50", 2), ("Pk75", 3)):
        gi = summary[arm]["none"]["generation_interval"]
        axes[0].errorbar(xpos, gi["mean"], yerr=[[gi["mean"] - gi["p10"]], [gi["p90"] - gi["mean"]]], fmt="o", color="#1b4f72")
    axes[0].set_xticks([0, 1, 2, 3], ["A0", "C", "Pk50", "Pk75"])
    axes[0].set_ylabel("generation interval, days (p10 to p90)")
    labels = ["top 10%", "the rest"]
    for i, arm in enumerate(("C", "Pk50")):
        top = summary[arm]["none"]["top10_before_diagnosis"]["mean"]
        rest = summary[arm]["none"]["rest_before_diagnosis"]["mean"]
        axes[1].bar([i - 0.15, i + 0.15], [top, rest], width=0.3, color=["#6c3483", "#bbbbbb"])
    axes[1].set_xticks([0, 1], ["C", "Pk50"])
    axes[1].set_ylim(0.8, 0.95)
    axes[1].set_ylabel("share of transmissions before diagnosis")
    for i, arm in enumerate(("C", "Pk50")):
        for j, lever in enumerate(("isolate", "household")):
            red = summary["reductions"][arm][lever]["reduction"]
            axes[2].errorbar(i + (j - 0.5) * 0.3, red["mean"], yerr=[[red["mean"] - red["low"]], [red["high"] - red["mean"]]],
                             fmt="o", color="#1b4f72" if lever == "isolate" else "#b86e00")
    axes[2].set_xticks([0, 1], ["constant", "peak window"])
    axes[2].set_ylabel("reduction in cumulative infections")
    fig.suptitle(claim, fontsize=10)
    save(fig, "F6_ps2")
    with open(OUT / "F6_ps2.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "gi_mean", "gi_p10", "gi_p90", "top10_before_dx", "rest_before_dx", "isolate_reduction", "household_reduction"])
        for arm in ("A0", "C", "Pk50", "Pk75"):
            gi = summary[arm]["none"]["generation_interval"]
            writer.writerow([
                arm, gi["mean"], gi["p10"], gi["p90"],
                summary[arm]["none"]["top10_before_diagnosis"]["mean"],
                summary[arm]["none"]["rest_before_diagnosis"]["mean"],
                summary["reductions"][arm]["isolate"]["reduction"]["mean"],
                summary["reductions"][arm]["household"]["reduction"]["mean"],
            ])


if __name__ == "__main__":
    f2()
    f3()
    f5()
    f6()
    f7()
    print("wrote", OUT)
