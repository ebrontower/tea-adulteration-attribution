"""Fig. 2 — nine-class tea and four-class cinnamon discrimination, raw versus whitened.

Single numeric source: Table S4 of manuscript.md, parsed by _fig_tables.py (no literal result
values in this file). Criteria: preregistrations/figures_criteria_CEA_v1.md §2 — all eleven model classes appear,
including the two arms that read against us (kNN on cinnamon −0.700, budget-searched CNN −0.151);
the LDA control keeps Δ = 0.000; the mean row carries only +0.116 and −0.089; a dash is never
drawn as a zero.
"""
from __future__ import annotations

import matplotlib.gridspec as gs
from matplotlib.lines import Line2D

import _fig_style as S
from _fig_tables import S4

FID = "Fig2_discrimination"

# Full names only: the manuscript defines PLS-DA and PCA but not SVM/MLP/kNN/LDA/CNN,
# so the figure may not use those acronyms either (memo: abbreviations are defined before use).
LABEL = {
    "Linear discriminant analysis (invariant control)": "Linear discriminant\nanalysis (control)",
    "Logistic regression": "Logistic\nregression",
    "PLS-DA": "PLS-DA",
    "PCA–LDA": "PCA + linear\ndiscriminant",
    "k-nearest neighbours": "k-nearest\nneighbours",
    "Support vector machine, radial basis": "Support vector\nmachine, radial",
    "Random forest": "Random forest",
    "Gradient boosting": "Gradient\nboosting",
    "Multilayer perceptron": "Multilayer\nperceptron",
    "One-dimensional CNN, fixed budget": "Convolutional network,\nfixed budget",
    "One-dimensional CNN, budget searched": "Convolutional network,\nbudget searched",
    "Mean Δ over the nine non-invariant arms": "Mean Δ, nine\nnon-invariant arms",
}

CLS = [k for k in LABEL if not k.startswith("Mean")]
MEAN = "Mean Δ over the nine non-invariant arms"
ORDER = CLS + [MEAN]


def dumbbell(ax, i_raw, i_wht, title, chance, window):
    for yi, key in zip(ys, CLS):
        r, w = data[key][i_raw], data[key][i_wht]
        ax.plot([r, w], [yi, yi], color="#C8C8C8", lw=1.0, zorder=1, solid_capstyle="round")
        ax.scatter([r], [yi], s=24, color=S.RAW, zorder=2, edgecolor="none")
        ax.scatter([w], [yi], s=24, color=S.WHITE, zorder=2, edgecolor="none")
    ax.axvline(chance, color=S.RULE, ls=":", lw=0.9, zorder=1)
    ax.axhline(0.62, color=S.RULE, lw=0.6, zorder=1)
    ax.set_title(title, fontsize=8.4, pad=5)
    ax.set_xlim(*window)
    ax.set_ylim(-0.75, len(ORDER) - 0.25)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Balanced accuracy", fontsize=7.4)
    return ax


fig = S.barefig(S.TEXT_W, 9.8)
spec = gs.GridSpec(1, 3, width_ratios=[1.00, 0.58, 1.00], wspace=0.13,
                   left=0.235, right=0.985, top=0.905, bottom=0.155)
ax_tea = fig.add_subplot(spec[0])
ax_del = fig.add_subplot(spec[1])
ax_cin = fig.add_subplot(spec[2])

data = S4()
ys = list(range(len(ORDER)))[::-1]

dumbbell(ax_tea, 0, 1, "Tea, nine classes", 1.0 / 9, (0.05, 1.04))
dumbbell(ax_cin, 3, 4, "Cinnamon, four classes", 0.25, (0.15, 1.04))

# model classes as y tick labels of the left panel only
ax_tea.set_yticks(ys, [LABEL[k] for k in ORDER], fontsize=7.0)
ax_tea.tick_params(axis="y", length=0, pad=2.0)
for t in ax_tea.get_yticklabels():
    t.set_ha("right")
    t.set_linespacing(1.0)
    t.set_clip_on(False)
for ax in (ax_del, ax_cin):
    ax.set_yticks(ys, [""] * len(ORDER))
    ax.tick_params(axis="y", length=0)

# central panel: whitened minus raw, both datasets as grouped bars
ax_del.axhline(0.62, color=S.RULE, lw=0.6, zorder=1)
ax_del.axvline(0, color=S.INK, lw=0.7, zorder=1)
ax_del.set_xlim(-0.86, 0.72)
ax_del.set_ylim(-0.75, len(ORDER) - 0.25)
ax_del.set_xticks([-0.5, 0.0, 0.5])
ax_del.set_xlabel("Δ balanced accuracy", fontsize=7.4)
ax_del.set_title("Whitened − raw", fontsize=8.4, pad=5)
for yi, key in zip(ys, ORDER):
    for d, off, col in ((data[key][2], 0.20, S.ACCENT), (data[key][5], -0.20, S.WHITE)):
        ax_del.barh(yi + off, d, height=0.34, color=col,
                    hatch="//" if key == MEAN else None,
                    edgecolor=S.INK if key == MEAN else "none",
                    linewidth=0.6 if key == MEAN else 0.0)
        if key == MEAN:                                   # the two summary numbers the table quotes
            ax_del.text(d + (0.045 if d >= 0 else -0.045), yi + off, f"{d:+.3f}",
                        ha="left" if d >= 0 else "right", va="center", fontsize=7.0)

for key in ORDER:
    S.note(FID, f"tea raw/{key}", data[key][0])
    S.note(FID, f"tea whitened/{key}", data[key][1])
    S.note(FID, "tea Δ/" + key, data[key][2])
    S.note(FID, f"cinnamon raw/{key}", data[key][3])
    S.note(FID, f"cinnamon whitened/{key}", data[key][4])
    S.note(FID, f"cinnamon Δ/{key}", data[key][5])

handles = [Line2D([], [], marker="o", ls="", color=S.RAW, ms=6, label="Raw absorbance"),
           Line2D([], [], marker="o", ls="", color=S.WHITE, ms=6, label="Whitened metric"),
           Line2D([], [], ls=":", color=S.RULE, lw=1.0, label="Chance level"),
           Line2D([], [], marker="s", ls="", color=S.ACCENT, ms=6, label="Δ tea"),
           Line2D([], [], marker="s", ls="", color=S.WHITE, ms=6, label="Δ cinnamon")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.585, 0.008), handlelength=1.4, columnspacing=1.3)

problems, bad = S.check_type(fig), S.scan_strings(fig)
tag = S.save(fig, FID)
p, counts = S.dump_drawn()
print(f"written {tag}.png/.pdf | text problems {len(problems)} | forbidden strings {len(bad)}")
for x in problems[:25]:
    print("   -", x)
for x in bad[:10]:
    print("   !", x)
print(f"drawn values -> {p} {counts}")
