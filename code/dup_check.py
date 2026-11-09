"""Are the two batch workbooks duplicate exports of the same set of scans?

For every formulation, correlate each spectrum of batch 0 against all spectra of
batch 1 and keep the maximum; repeat against the *other* formulations. If the
same-formulation cross-batch maximum is ~1 and far above the different-formulation
maximum, the workbooks are duplicate exports rather than independent measurements.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

FP = (900.0, 1800.0)


def norm_rows(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main():
    d = P.load_tea(verbose=False)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    X = np.asarray(d["X"], float)[:, sel]
    y, b = d["y"], d["y_batch"]
    names = P.TEA_RECIPE_NAMES
    Xn = norm_rows(X)

    out = {"per_recipe": {}}
    for c in sorted(set(y.tolist())):
        m1, m0 = (y == c) & (b == 1), (y == c) & (b == 0)
        if m1.sum() == 0 or m0.sum() == 0:
            continue
        A, B = Xn[m1], Xn[m0]
        # Same formulation, across batches.
        R_same = A @ B.T
        best_same = R_same.max(1)          # best cross-batch match per batch-1 scan
        # Different formulation (other recipes within batch 1).
        others = [cc for cc in sorted(set(y.tolist())) if cc != c]
        mo = (np.isin(y, others)) & (b == 1)
        R_diff = A @ Xn[mo].T
        best_diff = R_diff.max(1)
        out["per_recipe"][names[c]] = dict(
            n1=int(m1.sum()), n0=int(m0.sum()),
            same_min=float(best_same.min()), same_median=float(np.median(best_same)),
            diff_max=float(best_diff.max()), diff_median=float(np.median(best_diff)),
            # Residual after one-to-one assignment (Hungarian).
            pair_corr=float(np.mean(np.sort(R_same, axis=1)[:, -1])))
        print(f"{names[c]:10s} same-formulation cross-batch best r: "
              f"min {best_same.min():.6f} med {np.median(best_same):.6f} | "
              f"different-formulation max r {best_diff.max():.4f}")

    # Global: each spectrum's maximum correlation outside its own formulation.
    print()
    outp = os.path.join(p5_paths.RESULTS, "dup_check.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    r = out["per_recipe"]
    print("global minimum of the same-formulation cross-batch best r:",
          min(v["same_min"] for v in r.values()),
          "| global maximum of the different-formulation max r:",
          max(v["diff_max"] for v in r.values()))


if __name__ == "__main__":
    main()
