"""exp23: does the 'unattributable' verdict survive a change of spectral window?

All geometry in the paper was computed in 900-1800 cm^-1. Two questions:
  (a) is paraffin's increment a scaled image of the pure-tea spectrum in *every* window, or
      only in the fingerprint window?
  (b) can a window chosen *by geometry alone* (no attribution labels) reduce the dictionary
      coherence enough for NNLS set recovery to work?

Selection rule fixed before looking at attribution: pick the window minimising
    q(w) = max( max_{i!=j} |corr(d_i,d_j)| , max_j |corr(d_j, m)| )
over all contiguous 400-cm^-1 windows stepping 100 cm^-1 (keeping >= 100 channels).
Then, once, evaluate NNLS set recovery in that window with exp18's protocol.
"""
import sys
import json

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


def geometry(m):
    Xw = X[:, m]
    mu = {c: Xw[yr == c].mean(0) for c in np.unique(yr)}
    m0 = mu[0]
    R = np.stack([mu[c] - m0 for c in codes], axis=1)      # chan x 6
    C = np.corrcoef(R.T)
    iu = np.triu_indices(6, 1)
    rpure = np.array([np.corrcoef(mu[c] - m0, m0)[0, 1] for c in codes])
    return dict(n_channel=int(m.sum()),
                coherence=float(np.abs(C[iu]).max()),
                r_pure={names[c]: round(float(v), 3) for c, v in zip(codes, rpure)},
                max_abs_r_pure=float(np.abs(rpure).max()),
                q=float(max(np.abs(C[iu]).max(), np.abs(rpure).max())),
                cond=float(np.linalg.cond(R.T)))


def unmix(m, train_codes, test_code, thr):
    Xw = X[:, m]
    mu = {c: Xw[yr == c].mean(0) for c in np.unique(yr)}
    E = np.array([mu[c] for c in train_codes]).T
    W = np.array([nnls(E, x)[0] for x in Xw[yr == test_code]])
    exact = []
    for w in W:
        s = set()
        for k, c in enumerate(train_codes):
            if w[k] > thr:
                s |= comp[c]
        exact.append(s == comp[test_code])
    return float(np.mean(exact))


out = {"named_windows": {}, "scan": []}

named = {
    "fingerprint_900_1800": (900, 1800),
    "full_400_4000": (400, 4000),
    "ch_stretch_2800_3000": (2800, 3000),
    "oh_stretch_3200_3600": (3200, 3600),
    "carbonyl_1650_1800": (1650, 1800),
    "ch_bend_1300_1500": (1300, 1500),
    "low_400_900": (400, 900),
    "mid_1800_2800": (1800, 2800),
}
for k, (lo, hi) in named.items():
    m = (axis >= lo) & (axis <= hi)
    g = geometry(m)
    g["window"] = [lo, hi]
    out["named_windows"][k] = g

# ---- scan: contiguous 400 cm^-1 windows, 100 cm^-1 step, >=100 channels ----------
step, width = 100, 400
lo = 500
while lo + width <= 3900:
    m = (axis >= lo) & (axis <= lo + width)
    if m.sum() >= 100:
        g = geometry(m)
        g["window"] = [lo, lo + width]
        out["scan"].append(g)
    lo += step
out["scan"].sort(key=lambda g: g["q"])
out["scan_top10"] = out["scan"][:10]

# ---- once: attribution in the geometry-selected window --------------------------
best = out["scan"][0]["window"]
out["selected_window"] = best
res = {}
for tag, win in [("selected", best), ("fingerprint_900_1800", [900, 1800])]:
    m = (axis >= win[0]) & (axis <= win[1])
    for held, train in [(7, [0, 1, 2, 3, 4, 5, 6, 8]), (8, [0, 1, 2, 3, 4, 5, 6, 7])]:
        for thr in (0.02, 0.05, 0.10):
            res[f"{tag}_holdout{held}_thr{thr}"] = round(unmix(m, train, held, thr), 3)
out["attribution"] = res

json.dump(out, open(p5_paths.RESULTS + "/exp23_bandsel.json", "w"),
          indent=1, ensure_ascii=False)

print("named windows")
for k, g in out["named_windows"].items():
    print(f"  {k:24s} ch={g['n_channel']:5d} coh={g['coherence']:.3f} "
          f"max|r_pure|={g['max_abs_r_pure']:.3f} q={g['q']:.3f} cond={g['cond']:.1f}")
    print(f"      r_pure {g['r_pure']}")
print("\ntop-10 windows by q")
for g in out["scan_top10"]:
    print(f"  {g['window']} ch={g['n_channel']:4d} coh={g['coherence']:.3f} "
          f"q={g['q']:.3f} paraffin_r={g['r_pure']['石蜡']:.3f} cond={g['cond']:.1f}")
print("\nattribution:")
for k, v in res.items():
    print(f"  {k:34s} {v}")
