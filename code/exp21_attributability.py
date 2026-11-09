"""Attributability score v2: residual of a component's increment after removing the
baseline (constant), the tea-dilution direction (pure-tea mean spectrum) and every other
single-adulterant increment. Normalised by within-class noise."""
import sys, json
import numpy as np
import p5_data as P

d = P.load_tea(verbose=False)
X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
X, yr = X[yb == 1], yr[yb == 1]
m = (axis >= 900) & (axis <= 1800)
X = X[:, m]
names = P.TEA_RECIPE["sulfur"]
codes = list(range(1, 7))
mu = {int(c): X[yr == c].mean(0) for c in np.unique(yr)}
pure = mu[0]
inc = {c: mu[c] - pure for c in codes}
res = np.concatenate([X[yr == c] - mu[c] for c in np.unique(yr)], axis=0)
sigma = float(np.sqrt(np.mean(res ** 2)))

out = {}
for j in codes:
    cols = [np.ones_like(pure), pure] + [inc[k] for k in codes if k != j]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, inc[j], rcond=None)
    r = inc[j] - A @ coef
    out[names[j]] = {
        "A2": round(float(np.sqrt(np.mean(r ** 2)) / sigma), 3),
        "unique_var_frac": round(float(np.sum(r ** 2) / np.sum(inc[j] ** 2)), 4),
        "resid_rms": round(float(np.sqrt(np.mean(r ** 2))), 4),
    }
# leave-two-out as well: remove only the components that could co-occur? for now full span
print("sigma", round(sigma, 5))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump({"sigma": sigma, "per_component": out}, open("prep_score2.json", "w"), indent=1, ensure_ascii=False)
