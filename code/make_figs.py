"""Publication figures for the R2 manuscript: the geometry, gate-coverage and alpha-map

panels. They are drawn under their original file names (Fig1_geometry, Fig2_gate_coverage,
Fig3_alpha_map) and appear as Figs. 3, 4 and 5 of the CEA manuscript; all values are read
from the exported JSONs in results/, nothing is recomputed or invented.
Canvas sizes are the final print sizes (inches), so every font size below is the
printed point size.  Output: 600 dpi PNG + vector PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import p5_paths  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(p5_paths.FIGURES)
R2 = Path(p5_paths.RESULTS)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RED = "#B3192B"
BLUE = "#2F5F8F"
GEO = json.loads((R2 / "exp35_geometry.json").read_text(encoding="utf-8"))
SEL = json.loads((R2 / "exp34_selective_mondrian.json").read_text(encoding="utf-8"))

NAMES = ["sugar", "talc", "sulfur", "paraffin", "dye", "flavouring"]


def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Fig. 1  increment geometry
# ---------------------------------------------------------------------------
def fig1():
    g = GEO["main"]
    raw = np.array(g["gram_raw"])
    wht = np.array(g["gram_white"])
    cosr = np.array(g["cos_raw"])
    cosw = np.array(g["cos_white"])

    fig = plt.figure(figsize=(6.42, 2.35))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.04, 1.34], wspace=0.80,
                  left=0.085, right=0.985, top=0.86, bottom=0.30)

    for k, (M, ttl, lab) in enumerate([(raw, "Raw absorbance", "(a)"),
                                       (wht, "Whitened metric", "(b)")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(NAMES, rotation=50, ha="right")
        ax.set_yticklabels(NAMES)
        for i in range(6):
            for j in range(6):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                        color="white" if abs(v) > 0.55 else "black")
        ax.set_title(ttl, pad=3)
        ax.tick_params(length=0)
        panel_label(ax, lab, dx=-0.42, dy=1.03)
        for s in ax.spines.values():
            s.set_visible(False)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=5.8, width=0.5, length=1.8)

    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(6)
    ax.bar(x - 0.2, cosr, 0.38, color=RED, label="raw")
    ax.bar(x + 0.2, cosw, 0.38, color=BLUE, label=r"$\Sigma^{-1}$")
    ax.axhline(0, color="0.35", lw=0.6)
    ax.axhline(0.9, color="0.55", lw=0.5, ls=(0, (2, 2)))
    ax.axhline(-0.9, color="0.55", lw=0.5, ls=(0, (2, 2)))
    ax.set_ylim(-1.12, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=50, ha="right")
    ax.set_ylabel("cos with pure tea")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False,
              handlelength=1.1, handletextpad=0.4, borderaxespad=0,
              ncol=1, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "(c)", dx=-0.24, dy=1.03)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig1_geometry.{ext}", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 2  gate acceptance and per-group coverage
# ---------------------------------------------------------------------------
def fig2():
    arms = [("standard", "standard residual", RED),
            ("normalized", "local-scale normalised", BLUE)]
    groups = ["single", "binary", "ternary"]

    fig, axes = plt.subplots(1, 2, figsize=(6.10, 2.55))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.30, wspace=0.30)

    for ax, which, ylab, lab in [(axes[0], "accept", "Acceptance rate", "(a)"),
                                 (axes[1], "coverage", "Empirical coverage", "(b)")]:
        x = np.arange(3)
        for k, (key, name, col) in enumerate(arms):
            vals = [SEL["white|blocked"][key]["groups"][g][which] for g in groups]
            ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=col, label=name)
            for xi, v in zip(x + (k - 0.5) * 0.36, vals):
                ax.text(xi, v + 0.025, "0" if v == 0 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.6, color=col)
        if which == "coverage":
            ax.axhline(0.9, color="0.2", lw=0.6, ls=(0, (3, 2)))
            ax.text(2.42, 0.825, "nominal 0.90", fontsize=5.8, ha="right",
                    color="0.2")
        ax.set_xticks(x)
        ax.set_xticklabels(["single\n($n$=350)", "binary\n($n$=50)", "ternary\n($n$=50)"])
        ax.set_ylim(0, 1.13)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel(ylab)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lab, dx=-0.20, dy=1.03)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.2,
               handletextpad=0.5, columnspacing=2.2)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig2_gate_coverage.{ext}", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 3  alpha map
# ---------------------------------------------------------------------------
def fig3():
    am = SEL["alpha_map|blocked"]
    alphas = [float(a) for a in SEL["alphas"]]
    fig, ax = plt.subplots(figsize=(4.15, 2.65))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.94, bottom=0.185)

    style = {"standard": dict(color=RED, marker="o", label="standard score"),
             "normalized": dict(color=BLUE, marker="s",
                                label="local-scale normalised")}
    series = {k: [am[k][f"{a:g}"]["mean_size"] for a in alphas]
              for k in ("standard", "normalized")}
    for key in ("standard", "normalized"):
        sizes = series[key]
        covs = [am[key][f"{a:g}"]["coverage"] for a in alphas]
        ax.plot(alphas, sizes, lw=1.0, ms=3.4, **style[key])
        for j, (a, s, c) in enumerate(zip(alphas, sizes, covs)):
            other = series["normalized" if key == "standard" else "standard"][j]
            dy = 7.0 if s >= other else -11.5
            ax.annotate(f"{c:.2f}", (a, s), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=5.8,
                        color=style[key]["color"])

    ax.set_xlabel(r"target miscoverage $\alpha$")
    ax.set_ylabel("mean attribution-set size\n(of 64 compositions)")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlim(0.035, 0.215)
    ax.set_ylim(4, 31)
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4,
              handletextpad=0.5, borderaxespad=0.2)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig3_alpha_map.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    from PIL import Image
    for p in sorted(OUT.glob("*.png")):
        im = Image.open(p)
        print(p.name, im.size, im.info.get("dpi"))
