"""exp49c - sample structure of the cinnamon corpus + what the engine returns at each level.

Two things the exp49 curve needs before it can be quoted:

1. the block assumption. exp40 builds five-scan contiguous blocks by counting rows inside each
   (series, level) group, i.e. it assumes 50 rows = 10 samples x 5 scans in name order. If instead
   the 50 rows are 50 distinct samples with one scan each, the block-cluster bootstrap is resampling
   arbitrary name-adjacent quintuplets and the manuscript's "50 independently prepared samples per
   loading" needs re-reading. This prints the real group structure straight from the deposition.

2. the confusion at each level, so the wood-dust 24% dip and the Euclidean 0.667 plateau can be
   described by what the engine actually returns rather than inferred from a rate.
"""
from __future__ import annotations

import json
import os

import numpy as np

import exp40_public_pipeline as P
import p5_data as D
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp49c_structure.json"
LEVELS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24]

out = dict(group_structure={}, best_set_by_cell={})

# ---------------------------------------------------------------- 1. group structure
for code, key, label in P.SERIES:
    d = D.load_public(key)
    nm = np.asarray(d["groups"], dtype=object).astype(str)
    lv = np.asarray(d["y"], float)
    ax = np.asarray(d["wvn"], float)
    m = (ax >= P.WINDOW[0]) & (ax <= P.WINDOW[1])
    rows = []
    for l in sorted(set(lv.tolist())):
        sub = nm[lv == l]
        rows.append(dict(level=l, n_rows=int(len(sub)), n_unique_names=int(len(set(sub))),
                         example=list(sub[:3])))
    out["group_structure"][label] = dict(
        n_rows=int(len(nm)), n_channels_window=int(m.sum()),
        dnu=float(np.median(np.diff(ax[m]))), levels=rows,
        names_all_unique=bool(len(set(nm)) == len(nm)))
    print(f"{label:14s} rows {len(nm):5d} unique names {len(set(nm)):5d} "
          f"channels(win) {m.sum()} dnu {np.median(np.diff(ax[m])):.4f}")
    for r in rows[:4]:
        print("    ", r)

# ---------------------------------------------------------------- 2. confusion per cell
X, comp, level, block, series, n_dup, nch = P.load_cinnamon()
cls_a = comp.copy()
cls_b = np.zeros(len(comp), int)
for s in (1, 2, 3):
    for lv in np.unique(level[series == s]):
        cls_b[(series == s) & (level == lv)] = 100 * s + int(lv)
cand, atoms, bf = P.subsets(3), [1, 2, 3], P.blocked_folds(block)

for metric in ("euclid", "white_a"):
    _, rows = P.run_arm(X, comp, level, block, cls_a, cls_b, cand, atoms, bf, metric)
    tab = {}
    for c, nm in enumerate(P.NAME, start=1):
        for l in LEVELS:
            sub = [r for r in rows if r["comp"] == c and r["level"] == l]
            cnt = {}
            for r in sub:
                cnt[r["best_set"]] = cnt.get(r["best_set"], 0) + 1
            tab[f"{nm} @ {l}%"] = dict(truth=nm, n=len(sub),
                                       returned=dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
    out["best_set_by_cell"][metric] = tab

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print("\nwood dust, whitened, per level (returned set: count):")
for l in LEVELS:
    v = out["best_set_by_cell"]["white_a"][f"wood dust @ {l}%"]
    print(f"  {l:3d}%  n {v['n']}  {v['returned']}")
print("\nall adulterants, euclid, at 24% and at 1%:")
for nm in P.NAME:
    for l in (1, 24):
        v = out["best_set_by_cell"]["euclid"][f"{nm} @ {l}%"]
        print(f"  {nm:14s} {l:3d}%  n {v['n']}  {v['returned']}")
print("saved", OUT)
