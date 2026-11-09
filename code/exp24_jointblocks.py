"""exp24: can complementary spectral blocks break the attribution degeneracy?

Single-window geometry (exp23) has a coherence floor of ~0.91, and the window that makes
paraffin visible (C-H stretch) makes something else ambiguous. Test whether *jointly*
modelling several blocks - each centred and scaled to unit within-level noise, with one
shared composition vector - lowers the effective coherence and restores set recovery.
"""
import sys
import json
import itertools

import numpy as np
from scipy.optimize import nnls

import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

d = P.load_tea(verbose=False)
X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
X, yr = X[yb == 1], yr[yb == 1]
names = P.TEA_RECIPE["sulfur"]
codes = list(range(1, 7))
comp = {0: set(), 1: {"sugar"}, 2: {"talc"}, 3: {"sulfur"}, 4: {"paraffin"},
        5: {"dye"}, 6: {"flavouring"}, 7: {"sugar", "paraffin"},
        8: {"talc", "paraffin", "flavouring"}}


def blocks_of(spec):
    return [(axis >= lo) & (axis <= hi) for lo, hi in spec]


def stack(Xin, spec):
    """vertical stack of per-block centred, noise-scaled channels"""
    parts = []
    for lo, hi in spec:
        m = (axis >= lo) & (axis <= hi)
        B = Xin[:, m]
        B = B - B.mean(axis=1, keepdims=True)          # per-block baseline
        parts.append(B)
    return parts


def joint_geometry(spec):
    ms = blocks_of(spec)
    parts = []
    sig = []
    for m in ms:
        B = X[:, m]
        Bc = B - B.mean(axis=1, keepdims=True)
        inner = []
        for c in np.unique(yr):
            inner.append(Bc[yr == c] - Bc[yr == c].mean(0))
        sig.append(float(np.sqrt(np.mean(np.concatenate(inner, axis=0) ** 2))))
    for m, s in zip(ms, sig):
        parts.append((X[:, m] - X[:, m].mean(axis=1, keepdims=True)) / s)
    Z = np.hstack(parts)
    mu = {c: Z[yr == c].mean(0) for c in np.unique(yr)}
    R = np.stack([mu[c] - mu[0] for c in codes], axis=1)
    C = np.corrcoef(R.T)
    iu = np.triu_indices(6, 1)
    rpure = np.array([np.corrcoef(mu[c] - mu[0], mu[0])[0, 1] for c in codes])
    return dict(window=spec, n_channel=int(Z.shape[1]),
                coherence=float(np.abs(C[iu]).max()),
                max_abs_r_pure=float(np.abs(rpure).max()),
                q=float(max(np.abs(C[iu]).max(), np.abs(rpure).max())),
                cond=float(np.linalg.cond(R.T)),
                r_pure={names[c]: round(float(v), 3) for c, v in zip(codes, rpure)})


def joint_unmix(spec, train, held, thr):
    ms = [np.where((axis >= lo) & (axis <= hi))[0] for lo, hi in spec]
    sig = []
    for ix in ms:
        B = X[:, ix]
        Bc = B - B.mean(axis=1, keepdims=True)
        inner = [Bc[yr == c] - Bc[yr == c].mean(0) for c in np.unique(yr)]
        sig.append(float(np.sqrt(np.mean(np.concatenate(inner, axis=0) ** 2))))
    Z = np.hstack([(X[:, ix] - X[:, ix].mean(axis=1, keepdims=True)) / s
                   for ix, s in zip(ms, sig)])
    mu = {c: Z[yr == c].mean(0) for c in np.unique(yr)}
    E = np.array([mu[c] for c in train]).T
    W = np.array([nnls(E, x)[0] for x in Z[yr == held]])
    exact = []
    for w in W:
        s = set()
        for k, c in enumerate(train):
            if w[k] > thr:
                s |= comp[c]
        exact.append(s == comp[held])
    return float(np.mean(exact)), W.mean(0)


cands = {
    "fp_only": [(900, 1800)],
    "fp+ch": [(900, 1800), (2800, 3000)],
    "fp+ch+oh": [(900, 1800), (2800, 3000), (3200, 3600)],
    "fp+ch+low": [(900, 1800), (2800, 3000), (400, 900)],
    "fp+ch+oh+low": [(900, 1800), (2800, 3000), (3200, 3600), (400, 900)],
    "ch_bend+fp+ch": [(1300, 1500), (900, 1800), (2800, 3000)],
    "all5": [(400, 900), (900, 1800), (1800, 2800), (2800, 3000), (3200, 3600)],
}
out = {"geometry": {}, "attribution": {}}
for k, spec in cands.items():
    out["geometry"][k] = joint_geometry(spec)
    for held, train in [(7, [0, 1, 2, 3, 4, 5, 6, 8]), (8, [0, 1, 2, 3, 4, 5, 6, 7])]:
        for thr in (0.05, 0.10):
            e, w = joint_unmix(spec, train, held, thr)
            out["attribution"][f"{k}_h{held}_t{thr}"] = dict(
                exact=round(e, 3),
                weights={names[c]: round(float(v), 3) for c, v in zip(train, w)})

json.dump(out, open(p5_paths.RESULTS + "/exp24_jointblocks.json", "w"),
          indent=1, ensure_ascii=False)
print(f"{'block set':16s} {'ch':>5s} {'coh':>6s} {'q':>6s} {'cond':>7s} {'h7@.05':>7s} {'h8@.05':>7s}")
for k in cands:
    g = out["geometry"][k]
    print(f"{k:16s} {g['n_channel']:5d} {g['coherence']:6.3f} {g['q']:6.3f} "
          f"{g['cond']:7.1f} {out['attribution'][k+'_h7_t0.05']['exact']:7.2f} "
          f"{out['attribution'][k+'_h8_t0.05']['exact']:7.2f}")
    print("    r_pure", g["r_pure"])
for k, v in out["attribution"].items():
    if k.endswith("t0.05"):
        print(k, v["exact"], v["weights"])
