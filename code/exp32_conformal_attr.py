"""exp32 - T1: conformal attribution sets. Turning "detectable but not attributable" into a
statement with a guarantee.

The stuck point: on a new composite the strict component set is never right (0.000 in exp18), and
exp21 shows why - the six increment vectors have pairwise correlations 0.73-0.96 and the paraffin
increment is -0.959 x the pure-tea spectrum. No point estimator can fix that; it is a property of
the dictionary, not of the model.

If a point answer is impossible, the honest answer is a set with coverage. For a candidate
composition S subset of {sugar, talc, sulfur, paraffin, dye, flavour} the additive model predicts
    m_S = m0 + sum_{j in S} Delta_j
and split conformal with score s(x,S) = ||x - m_S||_2 gives
    C(x) = {S : s(x,S) <= q_{1-alpha}},   P(S_true in C(x)) >= 1-alpha.
The set size is bounded below by the number of compositions whose predicted spectra are within
2q of the truth, i.e. by the geometry of the dictionary. We measure both the empirical coverage
and that geometric floor.

Protocols
    A  in-distribution: stratified 5-fold over all 9 formulas (exchangeability holds)
    B  zero-shot composite: calibrate on pure + 6 singles, test on formulas 7 and 8
       (exchangeability violated by construction - the coverage breach measures how badly the
        additive dictionary describes a composite, not how badly the method works)
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.path.join(p5_paths.RESULTS, "exp32_conformal_attr.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
WINDOW = (900.0, 1800.0)
NAMES = ["sugar", "talc", "sulfur", "paraffin", "dye", "flavouring"]
ALPHA = 0.10


def load():
    d = P.load_tea(verbose=False)
    X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
    X, yr = X[yb == 1], yr[yb == 1]
    m = (axis >= WINDOW[0]) & (axis <= WINDOW[1])
    return X[:, m], yr


def dictionary(X, yr):
    mu = {int(c): X[yr == c].mean(0) for c in np.unique(yr)}
    m0 = mu[0]
    D = np.stack([mu[j] - m0 for j in range(1, 7)], 1)      # (p, 6)
    res = np.concatenate([X[yr == c] - mu[c] for c in np.unique(yr)], 0)
    sigma = float(np.sqrt(np.mean(res ** 2)))
    return m0, D, sigma, mu


def subsets():
    out = [()]
    for k in (1, 2, 3, 4, 5, 6):
        out += list(itertools.combinations(range(6), k))
    return out


def pred(m0, D, S):
    return m0 + D[:, list(S)].sum(1) if S else m0.copy()


def true_set(recipe):
    """component indices for a recipe code 0..8 (composites included)."""
    return {7: (0, 3), 8: (1, 3, 5)}.get(recipe, () if recipe == 0 else (recipe - 1,))


def fit_dict(X, yr, idx):
    """Additive dictionary fitted on idx only: class means, m0, six increments, residual sigma."""
    mu = {int(c): X[idx][yr[idx] == c].mean(0) for c in np.unique(yr[idx])}
    m0 = mu[0]
    D = np.stack([mu[j] - m0 for j in sorted(mu) if j != 0], 1)
    res = np.concatenate([X[idx][yr[idx] == c] - mu[c] for c in mu], 0)
    sigma = float(np.sqrt(np.mean(res ** 2)))
    return m0, D, sigma


def run_protocol(X, yr, cal_idx, test_idx, cand, alpha=ALPHA):
    m0, D, sigma = fit_dict(X, yr, cal_idx)          # no test data in the dictionary
    cand_m = np.stack([pred(m0, D, S) for S in cand])  # (64, p)
    S_of = {S: k for k, S in enumerate(cand)}

    # calibration: same functional as membership, but evaluated on the TRUE composition
    cal_scores = np.array([float(np.linalg.norm(X[i] - cand_m[S_of[tuple(sorted(true_set(int(yr[i]))))]]))
                           for i in cal_idx])
    n = len(cal_scores)
    q = float(np.quantile(cal_scores, min(1.0, (1 - alpha) * (1 + 1 / n))))

    rows = []
    for i in test_idx:
        sc = np.linalg.norm(cand_m - X[i], axis=1)      # (64,)
        C = [cand[k] for k in np.where(sc <= q)[0]]
        truth = tuple(sorted(true_set(int(yr[i]))))
        ktrue = S_of[truth]
        covered = ktrue in np.where(sc <= q)[0]
        order = np.argsort(sc)
        rank = int(np.where(order == ktrue)[0][0])
        # geometry, independent of the test point: neighbours of the truth prediction
        dt = np.linalg.norm(cand_m - cand_m[ktrue], axis=1)
        near = int(np.sum(dt <= 2 * q)) - 1             # exclude itself
        gap = float(np.min(dt[np.arange(len(cand)) != ktrue])) if len(cand) > 1 else np.inf
        rows.append(dict(covered=bool(covered), size=len(C),
                         point_exact=bool(sc[order[0]] == sc[ktrue]),
                         rank=rank, top3=bool(rank <= 2), top5=bool(rank <= 4),
                         floor=near, gap_over_sigma=round(gap / sigma, 3), q=q))

    cov = float(np.mean([r["covered"] for r in rows]))
    return dict(n_cal=n, n_test=len(rows), q=round(q, 4), coverage=round(cov, 3),
                mean_size=round(float(np.mean([r["size"] for r in rows])), 2),
                median_size=float(np.median([r["size"] for r in rows])),
                max_size=int(np.max([r["size"] for r in rows])),
                point_exact=round(float(np.mean([r["point_exact"] for r in rows])), 3),
                top3=round(float(np.mean([r["top3"] for r in rows])), 3),
                top5=round(float(np.mean([r["top5"] for r in rows])), 3),
                mean_floor=round(float(np.mean([r["floor"] for r in rows])), 2),
                mean_gap_over_sigma=round(float(np.mean([r["gap_over_sigma"] for r in rows])), 3),
                q_over_sigma=round(q / sigma, 3))


def main():
    X, yr = load()
    m0, D, sigma, mu = dictionary(X, yr)
    cand = subsets()
    rng = np.random.default_rng(0)

    # dictionary geometry
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    C = Dn.T @ Dn
    iu = np.triu_indices(6, 1)
    geo = dict(coherence=round(float(C[iu].max()), 4),
               cond=round(float(np.linalg.cond(D)), 2),
               cond_norm=round(float(np.linalg.cond(Dn)), 2),
               sigma=round(sigma, 5))

    # model check: does the additive dictionary describe the composites?
    for code in (7, 8):
        S = tuple(sorted(true_set(code)))
        r = np.linalg.norm(mu[code] - pred(m0, D, S)) / sigma
        geo[f"composite{code}_resid_over_sigma"] = round(float(r), 3)

    # protocol A: stratified 5-fold over all 9 formulas
    folds = [[] for _ in range(5)]
    for c in np.unique(yr):
        idx = np.where(yr == c)[0].copy()
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, 5)):
            folds[k] += list(chunk)
    a = []
    for k in range(5):
        test = np.array(sorted(folds[k]))
        cal = np.array(sorted(set(range(len(X))) - set(test)))
        a.append(run_protocol(X, yr, cal, test, cand))

    # protocol B: calibrate on pure+singles, test on composites
    cal = np.where(np.isin(yr, [0, 1, 2, 3, 4, 5, 6]))[0]
    test = np.where(np.isin(yr, [7, 8]))[0]
    b = run_protocol(X, yr, cal, test, cand)
    # B': also report per composite
    b7 = run_protocol(X, yr, cal, test[yr[test] == 7], cand)
    b8 = run_protocol(X, yr, cal, test[yr[test] == 8], cand)

    out = dict(geometry=geo, protocolA=dict(folds=a,
                                            coverage=round(float(np.mean([x["coverage"] for x in a])), 3),
                                            mean_size=round(float(np.mean([x["mean_size"] for x in a])), 2),
                                            point_exact=round(float(np.mean([x["point_exact"] for x in a])), 3),
                                            top3=round(float(np.mean([x["top3"] for x in a])), 3),
                                            top5=round(float(np.mean([x["top5"] for x in a])), 3),
                                            mean_floor=round(float(np.mean([x["mean_floor"] for x in a])), 2),
                                            mean_gap_over_sigma=round(float(np.mean([x["mean_gap_over_sigma"] for x in a])), 3)),
               protocolB=dict(all=b, comp7=b7, comp8=b8),
               candidates=len(cand), alpha=ALPHA, window=WINDOW)
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "protocolA"}, indent=1, ensure_ascii=False))
    print("A:", out["protocolA"])


if __name__ == "__main__":
    main()
