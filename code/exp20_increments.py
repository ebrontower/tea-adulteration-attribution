"""Net-increment structure of the six single adulterants (deduplicated 450-spectrum set).
Increment_i = mean spectrum of adulterant i - mean spectrum of pure tea, fingerprint region.
Reports the pairwise correlation matrix, the correlation of each increment with the pure-tea
spectrum, and the condition number of the 933 x 6 increment matrix."""
import sys, json
import numpy as np
import p5_data as P
import p5_paths  # noqa: E402

d = P.load_tea(verbose=False)
X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
X, yr = X[yb == 1], yr[yb == 1]
m = (axis >= 900) & (axis <= 1800)
X = X[:, m]
names = P.TEA_RECIPE["sulfur"]
codes = list(range(1, 7))
mu = {int(c): X[yr == c].mean(0) for c in np.unique(yr)}
pure = mu[0]
R = np.array([mu[c] - pure for c in codes])
C = np.corrcoef(R)
iu = np.triu_indices(6, 1)
five = [i for i, c in enumerate(codes) if c != 4]
C5 = C[np.ix_(five, five)]
iu5 = np.triu_indices(5, 1)
par = codes.index(4)
others = [i for i in range(6) if i != par]
out = {
    "n": int(X.shape[0]),
    "labels": [names[c] for c in codes],
    "corr_matrix": C.round(4).tolist(),
    "r_with_pure": {names[c]: float(np.corrcoef(mu[c] - pure, pure)[0, 1]) for c in codes},
    "range_five_nonparaffin": [float(C5[iu5].min()), float(C5[iu5].max())],
    "range_paraffin_vs_others": [float(C[par, others].min()), float(C[par, others].max())],
    "range_others_vs_pure": [float(min(np.corrcoef(mu[c] - pure, pure)[0, 1] for c in codes if c != 4)),
                             float(max(np.corrcoef(mu[c] - pure, pure)[0, 1] for c in codes if c != 4))],
    "cond_increment_matrix": float(np.linalg.cond(R.T)),
}
json.dump(out, open(p5_paths.RESULTS + "/exp20_increments.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "corr_matrix"}, indent=1))
