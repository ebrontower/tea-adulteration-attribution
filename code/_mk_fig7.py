"""Fig. 7 — the same claim read at a fixed information budget (Table S5).

Numeric source: Table S5 of manuscript.md, parsed by _fig_tables.py; x is the rung printed in the row
label, y the AUROC cells, and the arm names form a column on the right, one row per table row, joined
to the plotted point by a leader. Criteria: preregistrations/figures_criteria_CEA_v1.md §4 — all eleven rows drawn, the
One-class SVM row marked "absent" rather than placed at 0.5, a dash in the published-pool column left
blank and never bridged by a connecting segment, and the refusal gate (rung 6, 0.949) and the whitened
nearest-neighbour rule (rung 2, 0.947) drawn at the same marker size.
"""
from __future__ import annotations

import re

import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

import _fig_style as S
from _fig_tables import S5

FID = "Fig7_information_budget"

# RdBu_r endpoints of the author's own palette; one style per column of Table S5.
ORPH = ("#053061", "o")        # held-out adulterant
SING = ("#2166AC", "s")        # composite held out, singles pool
PUB = ("#B2182B", "^")         # composite held out, published reference pool
ABSENT_Y = -0.075              # the strip below the axis, where no AUROC is claimed

# the table's own row wording, wrapped so the name column stays inside its share of the page
WRAP = {
    "Distance to the authentic mean": "Distance to the authentic mean",
    "Nearest neighbour to pure tea, raw": "Nearest neighbour to pure tea, raw",
    "Hotelling T²": "Hotelling T²",
    "SPE(Q)": "SPE(Q)",
    "Autoencoder residual": "Autoencoder residual",
    "One-class SVM": "One-class SVM",
    "Nearest neighbour, all fit rows, raw": "Nearest neighbour,\nall fit rows, raw",
    "Mahalanobis in whitened coordinates": "Mahalanobis in\nwhitened coordinates",
    "Nearest neighbour, all fit rows, whitened": "Nearest neighbour,\nall fit rows, whitened",
    "Dictionary residual without the local scale": "Dictionary residual\nwithout the local scale",
    "Gate of Algorithm 1": "Gate of Algorithm 1",
}

fig = S.barefig(S.TEXT_W, 12.5)
spec = gs.GridSpec(1, 2, width_ratios=[1.00, 0.78], wspace=0.08,
                   left=0.085, right=0.995, top=0.905, bottom=0.215)
ax = fig.add_subplot(spec[0])
axn = fig.add_subplot(spec[1])

t = S5()
rows = [(k, re.sub(r"\s*\((?:\d+|oracle)\)$", "", k), d) for k, d in t.items()]
assert len(rows) == 11, f"Table S5 has {len(rows)} rows, all eleven must be drawn"


def anchor(d):
    """The point each name row is joined to: the held-out-adulterant AUROC, or the absent strip."""
    return ABSENT_Y - 1.0 if d["absent"] else (d["orphan"] if d["orphan"] is not None else 0.0)


# the table's rung order would make the leaders cross; sorting the name rows by the height of their
# own point keeps every leader monotone, and changes no drawn value or rung number
rows.sort(key=lambda r: anchor(r[2]), reverse=True)

# group by rung so arms sharing a rung can be jittered apart in x
by_rung = {}
for key, name, d in rows:
    by_rung.setdefault(d["rung"], []).append(name)
xpos = {}
for rung, names in by_rung.items():
    n = len(names)
    for i, name in enumerate(names):
        xpos[name] = float(rung) + (i - (n - 1) / 2.0) * 0.18

ax.set_xlim(-0.55, 6.55)
ax.set_ylim(-0.20, 1.06)
ax.set_xticks([0, 1, 2, 5, 6])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Information budget (rung)", fontsize=7.6)
ax.set_ylabel("AUROC for detecting the unseen", fontsize=7.6)
ax.set_title("Refusal readings at a fixed information budget", fontsize=8.4, pad=6)
ax.axhline(ABSENT_Y - 0.038, color=S.RULE, lw=0.5, ls=":")

axn.set_axis_off()
axn.set_xticks([])
axn.set_yticks([])
axn.set_xlim(0, 1)
axn.set_ylim(ax.get_ylim())

# a name sits at the height of the point it labels, nudged up only when the row above would touch it,
# so the leaders stay short instead of fanning across the panel
GAP = 0.046            # one 7 pt line, in AUROC units at this figure height
gap_rows = [GAP * (1 + (WRAP[name].count("\n"))) for _, name, _ in rows]
label_y, top = [], ax.get_ylim()[1] - 0.03
for a, g in zip([anchor(d) for _, _, d in rows], gap_rows):
    y = min(a, top - g / 2.0)
    label_y.append(y)
    top = y - g / 2.0

for (key, name, d), ylab in zip(rows, label_y):
    x = xpos[name]
    for what, (col, mk) in (("orphan", ORPH), ("comp_singles", SING), ("comp_pub", PUB)):
        v = d[what]
        if d["absent"]:
            continue
        if v is None:                                   # a dash stays blank, never a zero
            S.note(FID, f"{what}/{key}", None)
            continue
        ax.scatter([x], [v], s=30, marker=mk, color=col, zorder=3, edgecolor="none")
        S.note(FID, f"{what}/{key}", v)
    if d["absent"]:                                     # explicit empty marker, no value claimed
        ax.scatter([x], [ABSENT_Y], s=30, marker="X", color=S.RULE, zorder=3, linewidths=1.2)
        S.note(FID, f"orphan/{key}", None)

    y0 = ABSENT_Y if d["absent"] else d["orphan"]
    axn.text(0.02, ylab, WRAP[name], ha="left", va="center", fontsize=7.0, linespacing=1.0)
    axn.text(0.985, ylab, f"rung {d['rung']}", ha="right", va="center", fontsize=7.0)
    S.note(FID, f"rung/{key}", d["rung"])
    ax.add_patch(ConnectionPatch(xyA=(x, y0), xyB=(0.005, ylab), coordsA="data", coordsB="data",
                                 axesA=ax, axesB=axn, color=S.RULE, lw=0.5, zorder=1,
                                 shrinkB=1.0))

handles = [Line2D([], [], marker="o", ls="", color=ORPH[0], ms=6, label="Held-out adulterant"),
           Line2D([], [], marker="s", ls="", color=SING[0], ms=6, label="Composite, singles pool"),
           Line2D([], [], marker="^", ls="", color=PUB[0], ms=6, label="Composite, published pool"),
           Line2D([], [], marker="X", ls="", color=S.RULE, ms=7, mfc="none",
                  label="No usable threshold")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.40, 0.006), handlelength=1.3, columnspacing=1.6)

problems, bad = S.check_type(fig), S.scan_strings(fig)
tag = S.save(fig, FID)
p, counts = S.dump_drawn()
print(f"written {tag}.png/.pdf | text problems {len(problems)} | forbidden strings {len(bad)}")
for x in problems[:30]:
    print("   -", x)
for x in bad[:10]:
    print("   !", x)
print(f"drawn values -> {p} {counts}")
