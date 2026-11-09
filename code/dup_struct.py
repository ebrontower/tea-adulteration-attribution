"""Check whether the two batch files are bit-identical, and whether within-class
PC1 is drift or a concentration gradient."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

FP = (900.0, 1800.0)


def main():
    d = P.load_tea(verbose=False)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    X = np.asarray(d["X"], float)[:, sel]
    y, b = d["y"], d["y_batch"]
    names = P.TEA_RECIPE_NAMES
    out = {"exact_dup": {}, "pc1_vs_index": {}}

    # 1) Exact duplication: same formulation, same row index in both batches.
    for c in sorted(set(y.tolist())):
        m1, m0 = (y == c) & (b == 1), (y == c) & (b == 0)
        if m1.sum() == 0 or m0.sum() == 0:
            continue
        A, B = X[m1], X[m0]
        # Row-by-row difference under the order-preserving assignment.
        if A.shape == B.shape:
            dmax = float(np.abs(A - B).max())
            drel = float(np.abs(A - B).max() / (np.abs(A).max() + 1e-12))
        else:
            dmax = drel = float("nan")
        out["exact_dup"][names[c]] = dict(n1=int(A.shape[0]), n0=int(B.shape[0]),
                                          max_abs_diff=dmax, max_rel_diff=drel)

    # 2) Within-class PC1 against row order (time or concentration ramp).
    for c in sorted(set(y.tolist())):
        m = (y == c) & (b == 1)
        Xc = X[m]
        pc = PCA(n_components=5).fit(Xc)
        t = pc.transform(Xc)[:, 0]
        idx = np.arange(len(Xc))
        r = float(np.corrcoef(t, idx)[0, 1])
        # Projection on the pure-tea mean (dilution axis).
        mp = (y == 0) & (b == 1)
        tea_mean = X[mp].mean(0)
        proj = Xc @ tea_mean / (tea_mean @ tea_mean)
        out["pc1_vs_index"][names[c]] = dict(
            r_pc1_index=r, pc1_var=float(pc.explained_variance_ratio_[0]),
            proj_cv=float(np.std(proj) / abs(np.mean(proj))),
            proj_r_index=float(np.corrcoef(proj, idx)[0, 1]),
            pc1_span=float(t.max() - t.min()))
        print(f"{names[c]:10s} r(PC1,index)={r:+.3f}  "
              f"PC1={pc.explained_variance_ratio_[0]:.3f}  "
              f"dilution-projection CV={np.std(proj)/abs(np.mean(proj)):.4f}  "
              f"r(projection,index)={np.corrcoef(proj, idx)[0,1]:+.3f}  "
              f"row-wise max diff={out['exact_dup'].get(names[c],{}).get('max_abs_diff')}")

    outp = os.path.join(p5_paths.RESULTS, "dup_struct.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
