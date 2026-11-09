"""Fig. 1 — the certified-attribution workflow, one block per stage, one flow.

Criteria: preregistrations/figures_criteria_CEA_v1.md §1. Seven stage blocks plus an output block, joined by solid
arrows into a single flow (no (a)(b)(c) collage). The thumbnails are measurements, not sketches:
the acquired spectrum and the six reference increments come from the project loader and are gated
against the recorded experiment (see _fig_data.py), and the two Gram matrices are read verbatim
from the geometry file the author's own figure used. The numbers printed in the blocks are the
numbers the manuscript already states, and the gate greps each of them out of the text.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _fig_data as D
import _fig_style as S

FID = "Fig1_workflow"
H_CM = 13.5

COLS = [0.015, 0.3525, 0.690]
BW = 0.295
ROWS = [0.975, 0.650, 0.325]          # top edge of each row
BH = 0.275
TITLE_H = 0.062                        # one line of 7.8 pt at the top of a block
PAD = 0.030                            # space kept at the bottom of a block

fig = S.barefig(S.TEXT_W, H_CM)
TITLE_PT, BODY_PT, TICK_PT = 7.8, 7.0, 7.0

# the six increments, three dark reds and three dark blues of the author's RdBu_r, no pale middle
DELTA_COLS = [plt.cm.RdBu_r(v) for v in (0.02, 0.10, 0.20, 0.80, 0.90, 0.98)]
GRID = 8                               # 64 compositions = the subsets of six increments


def block(i_col, i_row, title):
    """Draw one flow block and return (left, bottom, thumb_x, thumb_y, thumb_w, thumb_h)."""
    x, ytop = COLS[i_col], ROWS[i_row]
    ybot = ytop - BH
    fig.patches.append(FancyBboxPatch((x, ybot), BW, BH, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      fc="white", ec=S.INK, lw=0.8, transform=fig.transFigure, zorder=1))
    fig.text(x + BW / 2, ytop - 0.012, title, ha="center", va="top", fontsize=TITLE_PT, zorder=6)
    return x, ybot, x + PAD, ybot + PAD, BW - 2 * PAD, BH - TITLE_H - 2 * PAD


def arrow(x0, y0, x1, y1):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                       lw=1.0, color=S.INK, transform=fig.transFigure, zorder=5,
                                       shrinkA=1.5, shrinkB=1.5))


def dots(tx, ty, tw, th, keep=None):
    """The 64 enumerated compositions as an 8x8 grid; `keep` marks how many are retained."""
    cell = th * 0.60 / GRID
    ox = tx + (tw - cell * GRID) / 2
    oy = ty + th * 0.34
    for i in range(GRID):
        for j in range(GRID):
            held = keep is not None and (i * GRID + j) < keep
            fig.patches.append(plt.Rectangle((ox + j * cell, oy + (GRID - 1 - i) * cell),
                                             cell * 0.66, cell * 0.66,
                                             fc=S.WHITE if held else "none",
                                             ec=S.INK if (held or keep is None) else S.RULE, lw=0.5,
                                             transform=fig.transFigure, zorder=3))
    return oy + cell * GRID


def mid(i_row):
    return ROWS[i_row] - BH / 2


t, g = D.tea(), D.gram()
S.note(FID, "window channels", t["channels"])
S.note(FID, "gram coherence raw", g["coherence"])
S.note(FID, "gram coherence whitened", g["coherence_white"])

# ---- 1. acquisition: one measured spectrum, as stored, over the fingerprint window ----------------
bx, by, tx, ty, tw, th = block(0, 0, "ATR-FTIR acquisition")
a1 = fig.add_axes([tx, ty + 0.026, tw, th * 0.70], zorder=3)
a1.plot(t["axis"], t["spectrum"], color=S.RAW, lw=0.8)
a1.set_xlim(t["axis"][0], t["axis"][-1])
a1.set_xticks([1000, 1500])
a1.set_yticks([])
a1.set_xlabel("Wavenumber (cm\u207b\u00b9)", fontsize=BODY_PT)
a1.set_title("Absorbance", fontsize=BODY_PT, pad=2)
a1.tick_params(labelsize=TICK_PT, pad=1.5, length=2)

# ---- 2. the audit funnel: exported rows to unique spectra ----------------------------------------
bx, by, tx, ty, tw, th = block(1, 0, "Duplicate and drift audit")
fig.text(tx + tw / 2, ty + th * 0.74, "850 exported rows", ha="center", va="center", fontsize=BODY_PT)
fig.text(tx + tw / 2, ty + th * 0.16, "450 unique spectra", ha="center", va="center", fontsize=BODY_PT)
arrow(tx + tw / 2, ty + th * 0.62, tx + tw / 2, ty + th * 0.30)

# ---- 3. dictionary: the six measured increments, same window, unscaled ---------------------------
bx, by, tx, ty, tw, th = block(2, 0, "Reference dictionary")
a3 = fig.add_axes([tx, ty, tw, th], zorder=3)
for k in range(6):
    a3.plot(t["axis"], t["deltas"][k], color=DELTA_COLS[k], lw=0.8)
a3.set_xlim(t["axis"][0], t["axis"][-1])
a3.axhline(0, color=S.RULE, lw=0.6)
a3.set_xticks([1000, 1500])
a3.set_yticks([])
a3.tick_params(labelsize=TICK_PT, pad=1.5, length=2)

# ---- 4. metric: the dictionary Gram before and after the noise whitening -------------------------
bx, by, tx, ty, tw, th = block(2, 1, "Noise metric")
gw, gap = tw * 0.42, tw * 0.10
for k, (mat, lab) in enumerate(((g["raw"], "Raw"), (g["white"], "Whitened"))):
    ax = fig.add_axes([tx + k * (gw + gap), ty + th * 0.16, gw, th * 0.70], zorder=3)
    ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(lab, fontsize=BODY_PT, pad=2)
fig.text(tx + tw / 2, ty - 0.002, "933 channels", ha="center", va="top", fontsize=BODY_PT)

# ---- 5. fit: the 64 compositions enumerated over the six increments ------------------------------
bx, by, tx, ty, tw, th = block(1, 1, "Additive fit")
dots(tx, ty, tw, th)
fig.text(tx + tw / 2, ty + th * 0.02, "64 candidate compositions", ha="center", va="bottom",
         fontsize=BODY_PT)

# ---- 6. sets: 16 of the 64 retained at the calibrated level --------------------------------------
bx, by, tx, ty, tw, th = block(0, 1, "Conformal set")
dots(tx, ty, tw, th, keep=16)
fig.text(tx + tw / 2, ty + th * 0.06, "16 of 64 candidates", ha="center", va="bottom", fontsize=BODY_PT)
fig.text(tx + tw / 2, ty - 0.002, "coverage 0.946", ha="center", va="top", fontsize=BODY_PT)

# ---- 7. gate: refusal as an output alongside a name -----------------------------------------------
bx, by, tx, ty, tw, th = block(0, 2, "Abstention gate")
fig.text(tx + tw / 2, ty + th * 0.70, "5.4% false abstention", ha="center", va="center", fontsize=BODY_PT)
fig.text(tx + tw / 2, ty + th * 0.26, "100 of 100 composites refused", ha="center", va="center",
         fontsize=BODY_PT)

# ---- output and its two exits ---------------------------------------------------------------------
bx, by, tx, ty, tw, th = block(1, 2, "Composition set or refusal")
ex, ey = COLS[2], ROWS[2]
for k, (lab, col) in enumerate((("Composition set", S.WHITE), ("Refusal", S.RAW))):
    yy = ey - 0.055 - k * 0.120
    fig.patches.append(FancyBboxPatch((ex + 0.012, yy - 0.070), BW - 0.024, 0.070,
                                      boxstyle="round,pad=0.003,rounding_size=0.008",
                                      fc="white", ec=col, lw=1.0, transform=fig.transFigure, zorder=2))
    fig.text(ex + BW / 2, yy - 0.035, lab, ha="center", va="center", fontsize=BODY_PT, zorder=6)
    arrow(bx + BW * 0.80, by + BH * (0.70 - 0.36 * k), ex + 0.010, yy - 0.035)

# ---- the flow itself -------------------------------------------------------------------------------
for c in (0, 1):
    arrow(COLS[c] + BW, mid(0), COLS[c + 1], mid(0))
arrow(COLS[2] + BW / 2, ROWS[0] - BH, COLS[2] + BW / 2, ROWS[1])
for c in (2, 1):
    arrow(COLS[c], mid(1), COLS[c - 1] + BW, mid(1))
arrow(COLS[0] + BW / 2, ROWS[1] - BH, COLS[0] + BW / 2, ROWS[2])
arrow(COLS[0] + BW, mid(2), COLS[1], mid(2))

problems, bad = S.check_type(fig), S.scan_strings(fig)
tag = S.save(fig, FID)
p, counts = S.dump_drawn()
print(f"written {tag}.png/.pdf | text problems {len(problems)} | forbidden strings {len(bad)}")
for x in problems[:30]:
    print("   -", x)
for x in bad[:10]:
    print("   !", x)
print(f"drawn values -> {p} {counts}")
