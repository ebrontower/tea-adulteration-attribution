"""exp51 fold-level follow-up: WHY do folds 4-5 of the KAN deposits lose coverage on both arms?

For each corpus and each level, every preparation is compared with the mean of the OTHER
preparations at the same level. A preparation whose offset exceeds the dose increment it is
supposed to be carrying is a baseline artefact, not a method failure - and a fold built on it
cannot be exchangeable with a calibration pool that excludes it.
"""
import os
import sys

import numpy as np

HERE = p5_paths.CODE
sys.path.insert(0, HERE)

import _p51_xcorpus as G        # noqa: E402
import ext_data as ED           # noqa: E402
import p5_paths  # noqa: E402

for name in ("D1", "D2", "D3", "D8", "HONEY"):
    spec = [s for s in G.corpora() if s["name"] == name][0]
    d = spec["data"]
    X, lv, g = d["X"], d["level"], np.asarray(d["groups"], object)
    print(f"\n### {name}  n={len(lv)} p={X.shape[1]}")
    rows = []
    for l in np.unique(lv):
        m = np.where(lv == l)[0]
        gs = sorted(set(g[m]))
        if len(gs) < 3:
            continue
        mus = {q: X[m][g[m] == q].mean(0) for q in gs}
        grand = X[m].mean(0)
        for j, q in enumerate(gs):
            others = np.concatenate([X[m][g[m] == o] for o in gs if o != q], 0)
            om = others.mean(0)
            off = float(np.linalg.norm(mus[q] - om))
            spread = float(np.mean(np.linalg.norm(others - om, axis=1)))
            rows.append((name, l, q, j, off, spread, off / max(spread, 1e-12)))
    # the dose increment between adjacent levels, for scale
    lv_pos = [v for v in np.unique(lv) if v > 0]
    if len(lv_pos) > 1:
        inc = float(np.linalg.norm(X[lv == lv_pos[1]].mean(0) - X[lv == lv_pos[0]].mean(0)))
        base = float(np.linalg.norm(X[lv == lv_pos[0]].mean(0) - X[lv == 0].mean(0)))
    else:
        inc = base = float("nan")
    rs = sorted(rows, key=lambda r: -r[6])[:8]
    print(f"   adjacent-level increment norm={inc:.4f}   first-level-vs-authentic={base:.4f}")
    print("   worst preparations (offset from the other preparations of the same level):")
    for nm, l, q, j, off, spread, ratio in rs:
        print(f"     level {l:5.1f}%  prep {str(q):>14s}  fold {j % 5}  "
              f"offset {off:.4f}  within-prep scatter {spread:.4f}  ratio {ratio:.2f}"
              + ("   [>= dose increment]" if off >= inc else ""))
    byfold = {}
    for nm, l, q, j, off, spread, ratio in rows:
        byfold.setdefault(j % 5, []).append(off)
    print("   mean offset by tested fold:", [round(float(np.mean(v)), 4) for k, v in sorted(byfold.items())])
