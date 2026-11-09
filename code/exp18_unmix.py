"""Linear unmixing floor: NNLS on the nine recipe mean spectra as endmembers.
The question is whether the composite blind spot of Section 3.4 is specific to
PLS or simply what any linear mixture model does on these data."""
import sys, json
import numpy as np
from scipy.optimize import nnls
import p5_data as P
import p5_paths  # noqa: E402

d = P.load_tea(verbose=False)
X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
X, yr = X[yb == 1], yr[yb == 1]
m = (axis >= 900) & (axis <= 1800)
X = X[:, m]
names = P.TEA_RECIPE["sulfur"]
comp = {0: set(), 1: {"sugar"}, 2: {"talc"}, 3: {"sulfur"}, 4: {"paraffin"},
        5: {"dye"}, 6: {"flavouring"}, 7: {"sugar", "paraffin"},
        8: {"talc", "paraffin", "flavouring"}}
mu = {int(c): X[yr == c].mean(0) for c in np.unique(yr)}

def unmix(train_codes, test_code, thr):
    E = np.array([mu[c] for c in train_codes]).T          # chan x k
    Xt = X[yr == test_code]
    W = np.array([nnls(E, x)[0] for x in Xt])
    pred = []
    for w in W:
        s = set()
        for k, c in enumerate(train_codes):
            if w[k] > thr:
                s |= comp[c]
        pred.append(s)
    exact = np.mean([p == comp[test_code] for p in pred])
    return float(exact), W

out = {}
for held, train in [(7, [0,1,2,3,4,5,6,8]), (8, [0,1,2,3,4,5,6,7])]:
    for thr in (0.02, 0.05, 0.10):
        e, W = unmix(train, held, thr)
        out[f"holdout_{held}_thr{thr}"] = {"exact": e,
            "mean_w_true": float(np.mean([W[:, train.index(c)] for c in range(9) if c in comp[held] and c in train], axis=0).mean()) if any(c in train for c in comp[held]) else None}
# in-distribution: all nine endmembers, threshold 0.05, test on every formulation
E = np.array([mu[c] for c in range(9)]).T
W = np.array([nnls(E, x)[0] for x in X])
exact = []
for i, c in enumerate(yr):
    s = set()
    for k in range(9):
        if W[i, k] > 0.05:
            s |= comp[k]
    exact.append(s == comp[int(c)])
out["insample_all9_thr0.05_exact"] = float(np.mean(exact))
out["mean_weights_example_holdout7"] = None
print(json.dumps(out, indent=1))
json.dump(out, open(p5_paths.RESULTS + "/exp18_unmix.json", "w"), indent=1)
