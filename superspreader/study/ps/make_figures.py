#!/usr/bin/env python3
"""Figures for the ACT-049 report.

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
import numpy as np

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


def _quartile(values, q):
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def f2():
    apply_figure_style()
    studies = read_csv(INPUTS / "published_k_from_R7.csv")
    settings = [r for r in read_csv(INPUTS / "published_k_by_setting_R6.csv") if r["estimator"] == "individual_offspring_sensitivity"]
    assert settings and all(r["estimator"] == "individual_offspring_sensitivity" for r in settings)
    ks = [float(row["k"]) for row in studies if row["k"]]
    below = sum(k < 1 for k in ks)
    q1, med, q3 = _quartile(ks, 0.25), _quartile(ks, 0.5), _quartile(ks, 0.75)
    claim = f"Median study-level k is {med:.2f} (IQR {q1:.2f} to {q3:.2f}); {below} of {len(ks)} are below 1"
    assert len(ks) == 28 and below == 26, claim
    assert round(med, 2) == 0.32 and round(q1, 2) == 0.20 and round(q3, 2) == 0.51, claim

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 8.2))
    ax = axes[0]
    ax.axvspan(q1, q3, color="#d6eaf8", zorder=0)
    ax.axvline(med, color="#1b4f72", lw=0.8, zorder=0)
    for i, row in enumerate(studies):
        if not row["k"]:
            continue
        k = float(row["k"])
        lo = float(row["k_lo"]) if row["k_lo"] else k
        hi = float(row["k_hi"]) if row["k_hi"] else k
        ax.errorbar(k, i, xerr=[[max(k - lo, 0)], [max(hi - k, 0)]], fmt="o", color="#1b4f72", ms=3.5, lw=0.8)
        ax.text(-0.02, i, f"{row['author']} {row['year']}", ha="right", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    for j, row in enumerate(settings):
        y = len(studies) + 1.4 + j
        k, lo, hi = float(row["k"]), float(row["k_lo"]), float(row["k_hi"])
        ax.errorbar(k, y, xerr=[[max(k - lo, 0)], [max(hi - k, 0)]], fmt="D", color="#b86e00", ms=3.5, lw=0.8)
        ax.text(-0.02, y, row["setting"], ha="right", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    ax.axvline(1, color="#888", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("dispersion k")
    ax.set_yticks([])
    ax.set_title(claim, fontsize=8)
    ax.text(0.0, -0.08, "Diamonds are Chen 2025 individual offspring, the quantity PanSim measures. The band is the study-level IQR.",
            transform=ax.transAxes, fontsize=6, color="#444")

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
    band_lo, band_hi = 1.2 * comp, 1.5 * comp
    claim = "The 20 to 50 percent band fits the uncoupled arms and not the coupled arms"
    assert band_lo <= by["E"]["early_k"] <= band_hi, claim
    assert band_lo <= by["P0"]["early_k"] <= band_hi, claim
    for arm in ("P25", "P50", "P100"):
        assert by[arm]["early_k"] > band_hi, claim
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    shapes = [s / 50 for s in range(2, 150)]
    ax.plot(shapes, [composition_k(s) for s in shapes], color="#888", lw=1, label="composition rule")
    ax.fill_between(shapes, [1.2 * composition_k(s) for s in shapes], [1.5 * composition_k(s) for s in shapes], color="#bbbbbb", alpha=0.4, label="20 to 50 percent band")
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


def _top_fraction(values, share):
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    cumulative = np.cumsum(ordered)
    index = int(np.searchsorted(cumulative, share * cumulative[-1], side="left"))
    return (min(index, len(ordered) - 1) + 1) / len(ordered)


def _top_curve(values, points=250):
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    fraction = np.arange(1, len(ordered) + 1) / len(ordered)
    index = np.linspace(0, len(ordered) - 1, points).astype(int)
    return fraction[index], cumulative[index]


def _nb_offspring(rng, k, reproduction, n):
    lam = rng.gamma(k, reproduction / k, size=n)
    return rng.poisson(lam)


def f4():
    """Transmission shares and emission shares on separate axes.

    The left axis is realised offspring. The right axis is a continuous emission.
    Adam's 19 percent share is a realised count, so it is compared with his own
    k of 0.43, not with the withdrawn Lorenz k of 0.228.
    """
    apply_figure_style()
    evidence = {row["row_id"]: row for row in read_csv(INPUTS / "published_evidence_verified.csv")}
    assert evidence["E03"]["derived_equiv_gamma_k"] == ""
    assert evidence["E04"]["derived_equiv_gamma_k"] == ""
    assert evidence["E03"]["value"] == "19.0"
    edwards_k = float(evidence["E17"]["derived_equiv_gamma_k"])
    yang_k = float(evidence["E18"]["derived_equiv_gamma_k"])
    rng = np.random.default_rng(18)
    n = 400000
    corners = ((0.43, 0.58, "k 0.43, R 0.58"), (0.29, 0.45, "k 0.29, R 0.45"), (0.67, 0.72, "k 0.67, R 0.72"))
    realised = {}
    for k, reproduction, label in corners:
        realised[(k, reproduction)] = _nb_offspring(rng, k, reproduction, n)
    fractions = {key: _top_fraction(sample, 0.80) for key, sample in realised.items()}
    assert 0.175 <= fractions[(0.43, 0.58)] <= 0.210
    assert 0.130 <= fractions[(0.29, 0.45)] <= 0.175
    assert 0.215 <= fractions[(0.67, 0.72)] <= 0.275
    lorenz_k = 0.228
    lorenz_share = _top_fraction(rng.gamma(lorenz_k, 1.0 / lorenz_k, size=n), 0.80)
    assert 0.16 <= lorenz_share <= 0.22
    assert 0.43 / lorenz_k > 1.7
    edwards = rng.gamma(edwards_k, 1.0 / edwards_k, size=n)
    yang = rng.gamma(yang_k, 1.0 / yang_k, size=n)
    edwards_fraction = _top_fraction(edwards, 0.80)
    yang_fraction = _top_fraction(yang, 0.90)
    assert abs(edwards_fraction - 0.18) < 0.03
    assert abs(yang_fraction - 0.02) < 0.015
    claim = "Adam's realised 19 percent share is his k of 0.43, not the Lorenz k of 0.23"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax = axes[0]
    colours = ["#1b4f72", "#7f8c8d", "#7f8c8d"]
    for (k, reproduction, label), colour in zip(corners, colours):
        xs, ys = _top_curve(realised[(k, reproduction)])
        ax.plot(xs, ys, color=colour, lw=1.4 if k == 0.43 else 0.8, label=label)
    ax.errorbar([0.19], [0.80], xerr=[[0.04], [0.05]], fmt="o", color="#c0392b", ms=5, label="Adam 19% (15 to 24)")
    ax.axhline(0.80, color="#cccccc", lw=0.6)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("top fraction of cases")
    ax.set_ylabel("share of realised offspring")
    ax.set_title(claim, fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax = axes[1]
    xs, ys = _top_curve(edwards)
    ax.plot(xs, ys, color="#1a7a4c", lw=1.4, label=f"Edwards aerosol, k {edwards_k:.2f}")
    xs, ys = _top_curve(yang)
    ax.plot(xs, ys, color="#6c3483", lw=1.4, label=f"Yang virions, k {yang_k:.3f}")
    ax.plot([0.18], [0.80], "o", color="#1a7a4c")
    ax.plot([0.02], [0.90], "o", color="#6c3483")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("top fraction of people")
    ax.set_ylabel("share of emission")
    emission_claim = "Emission shares stay on their own axis"
    ax.set_title(emission_claim, fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("Transmission shares and emission shares are different quantities", fontsize=11)
    save(fig, "F4_concentration")
    with open(OUT / "F4_concentration.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "label", "k", "R", "target_share", "top_fraction"])
        writer.writerow(["transmission", "Adam central", 0.43, 0.58, 0.80, fractions[(0.43, 0.58)]])
        writer.writerow(["transmission", "Adam low corner", 0.29, 0.45, 0.80, fractions[(0.29, 0.45)]])
        writer.writerow(["transmission", "Adam high corner", 0.67, 0.72, 0.80, fractions[(0.67, 0.72)]])
        writer.writerow(["transmission", "withdrawn Lorenz mapping, not plotted on this axis", lorenz_k, "", 0.80, lorenz_share])
        writer.writerow(["emission", "Edwards exhaled aerosol", edwards_k, "", 0.80, edwards_fraction])
        writer.writerow(["emission", "Yang circulating virions", yang_k, "", 0.90, yang_fraction])


def f8():
    apply_figure_style()
    payload = json.loads((SUMMARY.parent / "d1_absorption.json").read_text())
    bins = [row for row in read_csv(SUMMARY.parent / "d1_bins.csv") if row["bin_type"] == "decile"]
    assert payload["decision"]["verdict"] == "not_confirmed"
    claim = "Mean offspring in each decile rises with the multiplier"
    plotted = ("E", "P0", "P25", "P50", "P100", "P25k")
    by = {item["arm"]: item for item in payload["arms"]}
    for arm in plotted:
        deciles = sorted((row for row in bins if row["arm"] == arm), key=lambda row: float(row["mean_shed"]))
        assert float(deciles[-1]["mean_offspring"]) > float(deciles[4]["mean_offspring"]), claim
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for arm in plotted:
        deciles = sorted((row for row in bins if row["arm"] == arm), key=lambda row: float(row["mean_shed"]))
        xs = [float(row["mean_shed"]) for row in deciles]
        axes[0].plot(xs, [float(row["mean_offspring"]) for row in deciles], "-o", ms=3, lw=1, color=ARM_COLOUR[arm], label=arm)
        axes[1].plot(xs, [float(row["mean_locations"]) for row in deciles], "-o", ms=3, lw=1, color=ARM_COLOUR[arm], label=arm)
        top = by[arm].get("top1") or {}
        if top.get("mean_shed") and top.get("proportional_mean"):
            slope = top["proportional_mean"] / top["mean_shed"]
            line = np.linspace(0, top["mean_shed"], 30)
            axes[0].plot(line, slope * line, color=ARM_COLOUR[arm], lw=0.6, ls="--", alpha=0.7)
            axes[0].plot([top["mean_shed"]], [top["observed"]["mean"]], marker="*", ms=7, color=ARM_COLOUR[arm])
    axes[0].set_xlabel("mean multiplier (stars: top 1 percent; dashed: lower-half slope)")
    axes[0].set_ylabel("mean offspring within 14 days")
    axes[0].set_title(claim, fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].set_xlabel("mean multiplier in the decile")
    axes[1].set_ylabel("distinct transmission locations")
    axes[1].set_title("Locations among offspring, not contacts", fontsize=8)
    axes[1].legend(fontsize=6, frameon=False)
    fig.suptitle("Early-window offspring against the infector's multiplier", fontsize=11)
    save(fig, "F8_offspring_multiplier")
    with open(OUT / "F8_offspring_multiplier.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "decile", "mean_shed", "mean_offspring", "mean_locations", "n"])
        for arm in plotted:
            for row in sorted((item for item in bins if item["arm"] == arm), key=lambda item: float(item["mean_shed"])):
                writer.writerow([arm, row["label"], row["mean_shed"], row["mean_offspring"], row["mean_locations"], row["n"]])


if __name__ == "__main__":
    f2()
    f3()
    f4()
    f5()
    f6()
    f7()
    f8()
    print("wrote", OUT)
