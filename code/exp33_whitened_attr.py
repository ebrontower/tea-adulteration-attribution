"""exp33 - T1b: is the attribution gap an information problem or a metric problem?

exp32 gives valid 90% conformal attribution sets under plain Euclidean distance, but they hold
30 of 64 compositions - the guarantee is real and useless. The reason is that the six increments
are dominated by the pure-tea spectrum: the Euclidean metric spends its resolution on the
direction in which all samples vary most (tea matrix + baseline drift), not on the directions
that carry composition.

This script repeats the whole protocol in the within-class noise metric (Mahalanobis). If the
sets collapse, attribution was never an information deficit - it was the wrong inner product;
the deciding quantity is the whitened increment SNR ||Delta_j||_{Sigma^-1}. If they do not
collapse, the information genuinely is not in the spectrum.

Also reports the gauge coefficient gamma_j = <Delta_j, m0> / ||m0||^2: how much of each increment
is indistinguishable from a global multiplicative change of the pure spectrum.
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

OUT = os.path.join(p5_paths.RESULTS, "exp33_whitened_attr.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
WINDOW = (900.0, 1800.0)
NAMES = ["sugar", "talc", "sulfur", "paraffin", "dye", "flavouring"]
ALPHA = 0.10
K_WHITE = 60


def load():
    d = P.load_tea(verbose=False)
    X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
    X, yr = X[yb == 1], yr[yb == 1]
    m = (axis >= WINDOW[0]) & (axis <= WINDOW[1])
    return X[:, m], yr


def subsets():
    out = [()]
    for k in (1, 2, 3, 4, 5, 6):
        out += list(itertools.combinations(range(6), k))
    return out


def true_set(recipe):
    return {7: (0, 3), 8: (1, 3, 5)}.get(recipe, () if recipe == 0 else (recipe - 1,))


def pred(m0, D, S):
    return m0 + D[:, list(S)].sum(1) if S else m0.copy()


class Whitener:
    """Metric from pooled within-class residuals, truncated to k components with a tail floor."""

    def __init__(self, X, yr, idx, k=K_WHITE):
        mu = {int(c): X[idx][yr[idx] == c].mean(0) for c in np.unique(yr[idx])}
        R = np.concatenate([X[idx][yr[idx] == c] - mu[c] for c in mu], 0)
        R = R - R.mean(0)
        _, S, Vt = np.linalg.svd(R, full_matrices=False)
        lam = S ** 2 / max(1, len(R) - 1)
        k = int(min(k, len(lam)))
        self.V = Vt[:k].T
        self.inv = 1.0 / np.sqrt(np.maximum(lam[:k], 1e-10))
        self.delta = float(np.mean(lam[k:])) if len(lam) > k else float(lam[-1])
        self.dim = k

    def coords(self, V):
        P = V @ self.V
        tail = V - P @ self.V.T
        return np.hstack([P * self.inv[None, :], tail / np.sqrt(self.delta)])

    def dist(self, V):
        return np.linalg.norm(self.coords(V), axis=1)


def fit_dict(X, yr, idx):
    mu = {int(c): X[idx][yr[idx] == c].mean(0) for c in np.unique(yr[idx])}
    m0 = mu[0]
    elems = [j for j in sorted(mu) if 1 <= j <= 6]      # 7 and 8 are composites, not atoms
    D = np.stack([mu[j] - m0 for j in elems], 1)
    res = np.concatenate([X[idx][yr[idx] == c] - mu[c] for c in mu], 0)
    sigma = float(np.sqrt(np.mean(res ** 2)))
    return m0, D, sigma


def run_protocol(X, yr, fit_idx, cal_idx, test_idx, cand, metric, alpha=ALPHA):
    """Proper split conformal: dictionary and metric from fit_idx, quantile from cal_idx,
    coverage measured on test_idx. The three index sets are disjoint."""
    m0, D, sigma = fit_dict(X, yr, fit_idx)
    wh = Whitener(X, yr, fit_idx) if metric == "white" else None
    cand_m = np.stack([pred(m0, D, S) for S in cand])
    S_of = {S: k for k, S in enumerate(cand)}

    def score(x, k):
        v = (x - cand_m[k])[None, :]
        return float(wh.dist(v)[0]) if wh is not None else float(np.linalg.norm(v))

    cal_scores = np.array([score(X[i], S_of[tuple(sorted(true_set(int(yr[i]))))]) for i in cal_idx])
    n = len(cal_scores)
    q = float(np.quantile(cal_scores, min(1.0, (1 - alpha) * (1 + 1 / n))))

    rows = []
    for i in test_idx:
        V = X[i][None, :] - cand_m                        # (64, p)
        sc = wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)
        C = np.where(sc <= q)[0]
        truth = tuple(sorted(true_set(int(yr[i]))))
        ktrue = S_of[truth]
        order = np.argsort(sc)
        rank = int(np.where(order == ktrue)[0][0])
        dt = (wh.dist(cand_m - cand_m[ktrue]) if wh is not None
              else np.linalg.norm(cand_m - cand_m[ktrue], axis=1))
        near = int(np.sum(dt <= 2 * q)) - 1
        rows.append(dict(covered=bool(ktrue in C), size=int(len(C)),
                         point_exact=bool(rank == 0), rank=rank,
                         top3=bool(rank <= 2), top5=bool(rank <= 4), floor=near))

    cov = float(np.mean([r["covered"] for r in rows]))
    return dict(n_cal=n, n_test=len(rows), q=round(q, 4), coverage=round(cov, 3),
                mean_size=round(float(np.mean([r["size"] for r in rows])), 2),
                median_size=float(np.median([r["size"] for r in rows])),
                max_size=int(np.max([r["size"] for r in rows])),
                point_exact=round(float(np.mean([r["point_exact"] for r in rows])), 3),
                top3=round(float(np.mean([r["top3"] for r in rows])), 3),
                top5=round(float(np.mean([r["top5"] for r in rows])), 3),
                mean_floor=round(float(np.mean([r["floor"] for r in rows])), 2),
                q_over_sigma=(round(q / sigma, 3) if metric == "euclid" else None))


def geometry(X, yr):
    m0, D, sigma = fit_dict(X, yr, np.arange(len(X)))
    wh = Whitener(X, yr, np.arange(len(X)))
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    iu = np.triu_indices(6, 1)
    C = Dn.T @ Dn

    W = wh.coords(D.T)                                  # whitened coordinates of each increment
    wn = np.linalg.norm(W, axis=1)
    Gn = (W @ W.T) / np.outer(wn, wn)                   # whitened cosine Gram
    Wm = wh.coords(m0[None, :])[0]
    cos_white = (W @ Wm) / (wn * np.linalg.norm(Wm))    # increment direction vs pure spectrum

    gamma = (D.T @ m0) / float(m0 @ m0)
    perp = D - m0[:, None] * gamma[None, :]
    frac_perp = np.linalg.norm(perp, axis=0) / np.linalg.norm(D, axis=0)

    return dict(
        coherence=round(float(C[iu].max()), 4),
        coherence_white=round(float(Gn[iu].max()), 4),
        cond=round(float(np.linalg.cond(D)), 2),
        cond_white=round(float(np.linalg.cond(Gn)), 2),
        sigma=round(sigma, 5),
        snr_euclid=round(float(np.mean(np.linalg.norm(D, axis=0)) / sigma), 3),
        snr_white=round(float(np.mean(wn)), 3),
        gauge_gamma=dict(zip(NAMES, [round(float(g), 4) for g in gamma])),
        gauge_cos_white=dict(zip(NAMES, [round(float(c), 4) for c in cos_white])),
        perp_frac=dict(zip(NAMES, [round(float(f), 4) for f in frac_perp])),
        increments_euclid=dict(zip(NAMES, [round(float(v), 3) for v in np.linalg.norm(D, axis=0)])),
        increments_white=dict(zip(NAMES, [round(float(v), 3) for v in wn])),
    )


def stratified_split(yr, idx, frac, rng):
    """Split idx into (fit, calib) preserving every class present."""
    fit, cal = [], []
    for c in np.unique(yr[idx]):
        ii = idx[yr[idx] == c].copy()
        rng.shuffle(ii)
        cut = int(round(frac * len(ii)))
        fit += list(ii[:cut])
        cal += list(ii[cut:])
    return np.array(sorted(fit)), np.array(sorted(cal))


def block_ids(yr, block=5):
    """Contiguous scan blocks per recipe: repeat scans are 0.996-correlated, so scans must not be
    split at random."""
    b = np.zeros(len(yr), int)
    for c in np.unique(yr):
        idx = np.where(yr == c)[0]
        b[idx] = np.arange(len(idx)) // block
    return b


def make_splits(yr, mode, rng, nf=5):
    """Return nf (fit, cal, test) triples. mode='random' is the scan-level (optimistic) split;
    mode='blocked' holds out whole 5-scan blocks and never mixes blocks across roles."""
    triples = []
    if mode == "random":
        folds = [[] for _ in range(nf)]
        for c in np.unique(yr):
            idx = np.where(yr == c)[0].copy()
            rng.shuffle(idx)
            for k, chunk in enumerate(np.array_split(idx, nf)):
                folds[k] += list(chunk)
        for k in range(nf):
            test = np.array(sorted(folds[k]))
            rest = np.array(sorted(set(range(len(yr))) - set(test)))
            fit, cal = stratified_split(yr, rest, 0.7, rng)
            triples.append((fit, cal, test))
    else:
        b = block_ids(yr)
        for k in range(nf):
            test = np.where(b % nf == k)[0]
            rest = np.where(b % nf != k)[0]
            fit = rest[b[rest] < 5]
            cal = rest[b[rest] >= 5]
            triples.append((np.array(sorted(fit)), np.array(sorted(cal)), np.array(sorted(test))))
    return triples


def main():
    X, yr = load()
    cand = subsets()
    rng = np.random.default_rng(0)

    out = dict(geometry=geometry(X, yr), alpha=ALPHA, window=WINDOW, k_white=K_WHITE,
               candidates=len(cand))

    def agg(rs):
        return dict(coverage=round(float(np.mean([x["coverage"] for x in rs])), 3),
                    mean_size=round(float(np.mean([x["mean_size"] for x in rs])), 2),
                    median_size=float(np.median([x["median_size"] for x in rs])),
                    max_size=int(np.max([x["max_size"] for x in rs])),
                    point_exact=round(float(np.mean([x["point_exact"] for x in rs])), 3),
                    top3=round(float(np.mean([x["top3"] for x in rs])), 3),
                    top5=round(float(np.mean([x["top5"] for x in rs])), 3),
                    mean_floor=round(float(np.mean([x["mean_floor"] for x in rs])), 2),
                    q=round(float(np.mean([x["q"] for x in rs])), 3))

    for metric in ("euclid", "white"):
        out[metric] = {}
        for mode in ("random", "blocked"):
            a, a1 = [], []
            for fit, cal, test in make_splits(yr, mode, rng):
                a.append(run_protocol(X, yr, fit, cal, test, cand, metric))
                f1, c1, t1 = fit[yr[fit] <= 6], cal[yr[cal] <= 6], test[yr[test] <= 6]
                a1.append(run_protocol(X, yr, f1, c1, t1, cand, metric))

            if mode == "random":
                calB = np.where(np.isin(yr, [0, 1, 2, 3, 4, 5, 6]))[0]
                fitB, calibB = stratified_split(yr, calB, 0.7, rng)
            else:
                b = block_ids(yr)
                sel = np.where(np.isin(yr, [0, 1, 2, 3, 4, 5, 6]))[0]
                fitB = sel[b[sel] < 5]
                calibB = sel[b[sel] >= 5]
            testB = np.where(np.isin(yr, [7, 8]))[0]
            b_all = run_protocol(X, yr, fitB, calibB, testB, cand, metric)
            b7 = run_protocol(X, yr, fitB, calibB, testB[yr[testB] == 7], cand, metric)
            b8 = run_protocol(X, yr, fitB, calibB, testB[yr[testB] == 8], cand, metric)

            out[metric][mode] = dict(A=dict(folds=a, **agg(a)),
                                     A_single=dict(folds=a1, **agg(a1)),
                                     B=dict(all=b_all, comp7=b7, comp8=b8))
            print(metric, mode, json.dumps(out[metric][mode], ensure_ascii=False), flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("geometry", json.dumps(out["geometry"], ensure_ascii=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
