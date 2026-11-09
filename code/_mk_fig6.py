"""Fig. 6 — transfer to the public cinnamon corpus: metric gain (a) and the group collapse (b).

Numeric sources, all parsed from manuscript.md by _fig_tables.py: Table 5a (standard-score rows and
the block-cluster bootstrap intervals quoted in its note) and Table 5b (all eight groups), plus the two
caveats that Section 3.10 states in prose. Criteria: preregistrations/figures_criteria_CEA_v1.md §3 — every unfavourable
group stays visible; the "adulterant x loading" control row stays out of (a); error bars are the
intervals the table prints, never recomputed here.
"""
from __future__ import annotations

import matplotlib.gridspec as gs
from matplotlib.lines import Line2D

import _fig_style as S
from _fig_tables import T5a, T5b, bootstrap_intervals, prose_numbers

FID = "Fig6_cinnamon_transfer"
EU, WH = "Euclidean | standard", "Whitened (by adulterant) | standard"
N_OF_8 = 8.0                      # candidate-dictionary size, stated in Table 5a's caption

GROUPS = ["Walnut shell", "Peanut", "Wood dust", "Authentic cinnamon",
          "Loading 1–4%", "Loading 6–12%", "Loading 15–24%", "All spectra"]
GRP_LABEL = {g: ("All spectra (marginal)" if g == "All spectra" else g) for g in GROUPS}

fig = S.barefig(S.TEXT_W, 12.2)
spec = gs.GridSpec(2, 1, height_ratios=[0.40, 0.60], hspace=0.44,
                   left=0.205, right=0.990, top=0.925, bottom=0.175)
ax_a = fig.add_subplot(spec[0])
ax_b = fig.add_subplot(spec[1])

t5a, t5b = T5a(), T5b()
bs, prov = bootstrap_intervals(), prose_numbers()

# ---------------------------------------------------------------- (a) metric gain
rows = [
    ("Coverage", t5a[EU]["coverage"], t5a[WH]["coverage"], None, bs["coverage"], 3),
    ("Top-1", t5a[EU]["top1"], t5a[WH]["top1"], bs["top1_euclidean"], bs["top1"], 3),
    ("Mean set\n(of 8)", t5a[EU]["size"], t5a[WH]["size"], None, bs["size"], 2),
    ("Empty sets", t5a[EU]["empty"], t5a[WH]["empty"], None, bs["empty"], 3),
]
ya = [3, 2, 1, 0]
for yi, (name, rv, wv, ei, ew, dp) in zip(ya, rows):
    k = N_OF_8 if name.startswith("Mean set") else 1.0      # only the set size is a count of 8
    r, w = rv / k, wv / k
    ax_a.plot([r, w], [yi, yi], color="#C8C8C8", lw=1.0, zorder=1, solid_capstyle="round")
    if ei:
        ax_a.plot([ei[0] / k, ei[1] / k], [yi, yi], color=S.RAW, lw=1.7, zorder=2)
    if ew:
        ax_a.plot([ew[0] / k, ew[1] / k], [yi, yi], color=S.WHITE, lw=1.7, zorder=2)
    ax_a.scatter([r], [yi], s=26, color=S.RAW, zorder=3, edgecolor="none")
    ax_a.scatter([w], [yi], s=26, color=S.WHITE, zorder=3, edgecolor="none")
    ax_a.text(1.12, yi, f"{rv:.{dp}f} → {wv:.{dp}f}", ha="left", va="center", fontsize=7.0)
    S.note(FID, f"a raw/{name}", rv)
    S.note(FID, f"a whitened/{name}", wv)
    S.note(FID, f"a interval whitened/{name}", list(ew) if ew else None)
    S.note(FID, f"a interval raw/{name}", list(ei) if ei else None)

ax_a.set_yticks(ya, [r[0] for r in rows], fontsize=7.0)
ax_a.tick_params(axis="y", length=0, pad=2.0)
ax_a.set_xlim(-0.035, 1.42)
ax_a.set_ylim(-0.55, 3.55)
ax_a.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax_a.set_xlabel("Proportion, or fraction of the 8 candidates", fontsize=7.4)
ax_a.set_title("(a) Standard-score readings, euclidean against whitened", fontsize=8.4, pad=5)

# ---------------------------------------------------------------- (b) group conditional
yb = list(range(len(GROUPS)))[::-1]
X_SET, X_ACC, X_CAV = 1.10, 1.52, 1.98
for yi, key in zip(yb, GROUPS):
    g = t5b[key]
    for off, mk, col in ((0.17, "o", "coverage"), (-0.17, "s", "top1")):
        r, w = g[col]
        ax_b.plot([r, w], [yi + off, yi + off], color="#C8C8C8", lw=1.0, zorder=1,
                  solid_capstyle="round")
        ax_b.scatter([r], [yi + off], s=24, marker=mk, color=S.RAW, zorder=3, edgecolor="none")
        ax_b.scatter([w], [yi + off], s=24, marker=mk, color=S.WHITE, zorder=3, edgecolor="none")
    ax_b.text(X_SET, yi, f"{g['size'][0]:.2f} / {g['size'][1]:.2f}", ha="left", va="center", fontsize=7.0)
    ax_b.text(X_ACC, yi, f"{g['accept'][0]:.2f} / {g['accept'][1]:.2f}", ha="left", va="center", fontsize=7.0)
    S.note(FID, f"b coverage/{key}", [g["coverage"][0], g["coverage"][1]])
    S.note(FID, f"b top-1/{key}", [g["top1"][0], g["top1"][1]])
    S.note(FID, f"b set/{key}", [g["size"][0], g["size"][1]])
    S.note(FID, f"b acceptance/{key}", [g["accept"][0], g["accept"][1]])

# the two caveats Section 3.10 states in prose, pinned to their groups
caveats = {"Wood dust": (f"{prov['wood dust empty sets'] * 100:.1f}% empty sets",
                         prov["wood dust empty sets"]),
           "Authentic cinnamon": (f"{prov['authentic only among accepted'] * 100:.1f}% authentic\n"
                                  "among accepted", prov["authentic only among accepted"])}
for key, (txt, val) in caveats.items():
    ax_b.text(X_CAV, yb[GROUPS.index(key)], txt, ha="left", va="center", fontsize=7.0)
    S.note(FID, f"b caveat/{key}", val)

ax_b.axvline(1.05, color=S.RULE, lw=0.6)
for x, lab in ((X_SET, "Mean set / 8"), (X_ACC, "Acceptance"), (X_CAV, "Group caveat")):
    ax_b.text(x, len(GROUPS) - 0.42, lab, ha="left", va="bottom", fontsize=7.0)

ax_b.set_yticks(yb, [GRP_LABEL[g] for g in GROUPS], fontsize=7.0)
ax_b.tick_params(axis="y", length=0, pad=2.0)
ax_b.set_xlim(-0.045, 2.42)
ax_b.set_ylim(-0.62, len(GROUPS) + 0.22)
ax_b.set_xticks([0.0, 0.5, 1.0])
ax_b.set_xlabel("Coverage and top-1", fontsize=7.4)
ax_b.set_title("(b) Group- and loading-conditional readings, raw against whitened", fontsize=8.4, pad=5)

handles = [Line2D([], [], marker="o", ls="", color=S.RAW, ms=6, label="Euclidean (raw)"),
           Line2D([], [], marker="o", ls="", color=S.WHITE, ms=6, label="Whitened (by adulterant)"),
           Line2D([], [], ls="-", color="#C8C8C8", lw=1.4, label="Raw to whitened"),
           Line2D([], [], ls="-", color=S.RAW, lw=1.7, label="95% interval, euclidean"),
           Line2D([], [], ls="-", color=S.WHITE, lw=1.7, label="95% interval, whitened"),
           Line2D([], [], marker="o", ls="", color=S.INK, ms=6, label="Coverage (circles)"),
           Line2D([], [], marker="s", ls="", color=S.INK, ms=6, label="Top-1 (squares)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.55, 0.004), handlelength=1.5, columnspacing=1.4)

problems, bad = S.check_type(fig), S.scan_strings(fig)
tag = S.save(fig, FID)
p, counts = S.dump_drawn()
print(f"written {tag}.png/.pdf | text problems {len(problems)} | forbidden strings {len(bad)}")
for x in problems[:25]:
    print("   -", x)
for x in bad[:10]:
    print("   !", x)
print(f"drawn values -> {p} {counts}")
