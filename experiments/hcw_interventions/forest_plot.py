"""Forest plot: effect of each scenario on total HCW infections relative to baseline (ratio of
means, 95% bootstrap CI). The baseline is drawn as a reference row at 1 with a CI
(the baseline mean's own bootstrap CI).
Reads outputs/uncertainty_effects.csv and uncertainty_ci.csv (run uncertainty.py first).

Colour = sign of the effect: CI entirely below 1 (protective), entirely above 1 (harmful),
spanning 1 (not distinguishable from baseline). Identity is also carried by the row labels and
the printed ratio, so the plot does not rely on colour alone.

Usage: python forest_plot.py [--all] [--no-table] [out_dir]
--no-table leaves the scenario key out of the figure (use the .tex table instead; much shorter figure).
Default shows the single-lever scenarios; --all adds the combined ones.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
no_table = "--no-table" in sys.argv
show_all = "--all" in sys.argv
out = Path(args[0]) if args else here / "outputs"

LABELS = {
    "hcwPPE": "Hospital PPE: 70% lower infectiousness in hospitals and clinics",
    "hcwDailyTest": "Hospital staff 95% test chance per day, retest after 1 day, until day 258",
    "hiTest": "Hospital staff test chance 50% per day instead of 0.5%, until day 258",
    "longQuarantine": "Quarantine after diagnosis lasts 21 days instead of 10",
    "closuresNone": "All closure rules removed (also no new variants; hospital testing stays low)",
    "noVaccHCW": "HCWs vaccinated as general adults, no longer first",
    "quarantine0": "No quarantine of diagnosed cases, households or workplaces",
    "noVacc": "No vaccination programme: no doses given",
    "closuresNone_hiTest": "No closure rules, hospital staff test chance 50% per day all year",
    "noVaccHCW_hiTest": "HCWs vaccinated as general adults + hospital test chance 50%, to day 258",
    "quarantine0_hiTest": "No quarantine + hospital staff test chance 50% per day, to day 258",
    "quarantine0_noVaccHCW": "No quarantine + HCWs vaccinated as general adults",
}
NUMBER = {name: k + 1 for k, name in enumerate(list(LABELS))}
SINGLE = ["hcwPPE", "hcwDailyTest", "hiTest", "longQuarantine", "closuresNone",
          "noVaccHCW", "quarantine0", "noVacc"]

# Diverging pair from the key-metrics palette + neutral grey for "not distinguishable".
PROTECT, HARM, NEUTRAL = "#2a78d6", "#eb6834", "#8a8f98"
BASE = "#4a4f57"  # baseline reference row

eff = pd.read_csv(out / "uncertainty_effects.csv")
eff = eff[eff.metric == "hcw_infections"].set_index("scenario")
eff = eff.loc[list(LABELS) if show_all else SINGLE].sort_values("ratio")

ci = pd.read_csv(out / "uncertainty_ci.csv")

# Baseline as a reference row: ratio 1 by construction; its CI is the baseline mean's own
# bootstrap CI divided by the baseline mean (the noise the scenario ratios are measured against).
bl = ci[(ci.scenario == "baseline") & (ci.metric == "hcw_infections")].iloc[0]
eff.loc["baseline"] = {"ratio": 1.0, "ratio_ci95_lo": bl.mean_ci95_lo / bl["mean"],
                       "ratio_ci95_hi": bl.mean_ci95_hi / bl["mean"]}
eff = eff.sort_values("ratio")
LABELS["baseline"] = f"Policies used in the real world: {bl['mean']:,.0f} HCW infections (mean of 20 runs)"


def colour(lo, hi):
    return PROTECT if hi < 1 else HARM if lo > 1 else NEUTRAL


POINT = "#1a1a1a"


def draw(a, x, y, lo, hi, c):
    """CI as a coloured bar with end caps; the mean is a small dark point on top of it."""
    a.hlines(y, lo, hi, color=c, lw=3, zorder=2)
    a.vlines([lo, hi], y - 0.12, y + 0.12, color=c, lw=2.5, zorder=2)
    a.plot(x, y, "D", color=POINT, ms=3.5, zorder=3)


plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 13, "axes.edgecolor": "0.4"})
n = len(eff)
# Layout in inches so row spacing stays fixed regardless of the number of scenarios.
TABLE_ROW = 0.27
TABLE_H = 0.0 if no_table else 0.6 + TABLE_ROW * n   # scenario key under the legend
LEGEND_H, XAXIS_H = 0.85, 0.9
W, TOP, ROW_H = 7.5, 0.7, 0.31
BOTTOM = TABLE_H + LEGEND_H + XAXIS_H
axB_h = ROW_H * (n + 1.2)
H = TOP + axB_h + BOTTOM
VALUE_COL = 1.75  # inches reserved right of the axes for the ratio and CI column
LEFT, RIGHT = 0.12, 1 - VALUE_COL / W
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([LEFT, BOTTOM / H, RIGHT - LEFT, axB_h / H])

ax.axvline(1, color="0.35", lw=1, ls="--", zorder=0)   # reference: no change
# Scenario effect on total HCW infections vs baseline
for i, (scen, r) in enumerate(eff.iterrows()):
    c = BASE if scen == "baseline" else colour(r.ratio_ci95_lo, r.ratio_ci95_hi)
    draw(ax, r.ratio, i, r.ratio_ci95_lo, r.ratio_ci95_hi, c)
    txt = f"{r.ratio:.2f}  [{r.ratio_ci95_lo:.2f}, {r.ratio_ci95_hi:.2f}]"
    ax.annotate(txt, xy=(1, i), xycoords=("axes fraction", "data"), xytext=(12, 0),
                textcoords="offset points", va="center", ha="left", fontsize=12.5)
ax.annotate("ratio [95% CI]", xy=(1, n + 0.1), xycoords=("axes fraction", "data"), xytext=(12, 0),
            textcoords="offset points", va="center", ha="left", fontsize=12.5, fontweight="bold")
# One-line row labels: bold scenario id, then the description. Ids are fixed per scenario (order of
# LABELS: single changes first, then the combinations) so the text can refer to them.
def row_label(scen):
    return r"$\mathbf{Baseline}$" if scen == "baseline" else rf"$\mathbf{{S{NUMBER[scen]}}}$"


ax.set_yticks(range(n)); ax.set_yticklabels([row_label(sc) for sc in eff.index], fontsize=13)
ax.set_xlabel("ratio to baseline; 1 means no change")

ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xscale("log")
ax.set_xlim(0.3, 2.3)
ax.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2])
ax.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2"])
ax.minorticks_off()
ytop = n + 0.05
ax.annotate("", xy=(0.34, ytop), xytext=(0.9, ytop), arrowprops=dict(arrowstyle="->", color=PROTECT))
ax.annotate("", xy=(2.1, ytop), xytext=(1.1, ytop), arrowprops=dict(arrowstyle="->", color=HARM))
ax.text(0.58, ytop + 0.3, "fewer infections", ha="center", fontsize=11.5, color=PROTECT)
ax.text(1.55, ytop + 0.3, "more infections", ha="center", fontsize=11.5, color=HARM)
ax.set_ylim(-0.7, n + 0.55)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_w = max(t.get_window_extent(_r).width for t in ax.get_yticklabels()) / fig.dpi
LEFT = min(0.7, (_w + 0.3) / W)
ax.set_position([LEFT, BOTTOM / H, RIGHT - LEFT, axB_h / H])

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_items = [
    Line2D([], [], marker="o", color=PROTECT, lw=2.5, label="Fewer infections: CI below 1"),
    Line2D([], [], marker="o", color=HARM, lw=2.5, label="More infections: CI above 1"),
]
if any(colour(r.ratio_ci95_lo, r.ratio_ci95_hi) == NEUTRAL for sc, r in eff.iterrows() if sc != "baseline"):
    legend_items.append(Line2D([], [], marker="o", color=NEUTRAL, lw=2.5, label="No clear difference: CI spans 1"))
legend_items.append(Line2D([], [], marker="o", color=BASE, lw=2.5, label="Baseline: real-world policies"))
legend_items.append(Line2D([], [], marker="D", color=POINT, ms=4, lw=0, label="Diamond: mean of 20 runs; bar: 95% CI"))
legend_items.append(Line2D([], [], color="0.35", lw=1, ls="--", label="Dashed line: ratio of 1"))
fig.legend(handles=legend_items, loc="lower center", ncol=2, frameon=False, fontsize=11.5, columnspacing=1.2, handlelength=1.8,
           bbox_to_anchor=(0.5, TABLE_H / H))

# Scenario key at the bottom: id, description (also written as LaTeX for the paper).
key = [("baseline", "Baseline", LABELS["baseline"])] + [
    (sc, f"S{NUMBER[sc]}", LABELS[sc]) for sc in sorted((x for x in eff.index if x != "baseline"), key=NUMBER.get)]
if not no_table:
  fig.text(0.015, (TABLE_H - 0.12) / H, "Scenarios", fontsize=13.5, fontweight="bold", va="top")
  for k, (_, sid, desc) in enumerate(key):
      y = (TABLE_H - 0.12 - 0.34 - TABLE_ROW * k) / H
      fig.text(0.015, y, sid, fontsize=12.5, fontweight="bold", va="top")
      fig.text(0.015 + 0.95 / W, y, desc, fontsize=12.5, va="top")
  fig.add_artist(plt.Line2D([0.015, 0.985], [(TABLE_H - 0.12 - 0.3) / H] * 2, color="0.5", lw=0.8))
tex = ["\\begin{tabular}{@{}ll@{}}", "\\hline", "ID & Scenario \\\\ \\hline"]
tex += [f"{sid} & {desc.replace('%', chr(92) + '%')} \\\\" for _, sid, desc in key]
tex += ["\\hline", "\\end{tabular}"]
(out / ("forest_plot_all_scenarios.tex" if show_all else "forest_plot_scenarios.tex")).write_text("\n".join(tex) + "\n")
fig.suptitle("Effect of interventions on healthcare worker infections", fontsize=15, fontweight="bold", y=1 - 0.12 / H)
name = ("forest_plot_all" if show_all else "forest_plot") + ("_notable" if no_table else "")
fig.savefig(out / f"{name}.png", dpi=200)
fig.savefig(out / f"{name}.pdf")
print(f"-> {out / name}.png/.pdf ({n} scenarios)")
