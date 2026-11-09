"""exp37 - sample-level holdout: is the blocked protocol already at the independence limit?

The 450 spectra are 50 repeat scans of nine preparations, so no split can hold out a *new
sample*. This script does the two things that remain possible without new measurements:

(a) Effective sample size. For each formulation the first principal component of the
    within-formulation residuals is the drift score (98% of variance for most formulations).
    Its integrated autocorrelation time tau = 1 + 2 * sum_k rho_k gives n_eff = 50 / tau, the
    number of effectively independent scans behind each formulation.

(b) Cluster holdout. The 50 scans of each formulation are clustered by raw spectral correlation
    (average linkage on 1 - r, cut into five groups with fcluster), and whole clusters - not
    fixed blocks - are assigned to roles: fold k tests cluster k, calibrates on clusters k+1 and
    k+2, fits on the rest. Cluster boundaries are data-driven, each cluster is used in exactly
    one role, and the sizes (10 test / 20 fit / 20 calibration per class) match the blocked
    protocol, so any difference is due to *where* the boundary falls, not to sample size.

Both protocols are evaluated with the whitened metric at alpha = 0.10: the conformal set uses
the standard whitened score (as in Section 3.5) and the abstention gate uses the locally
normalised residual calibrated on single-adulterant calibration spectra (as in Section 3.7).
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import exp33_whitened_attr as E  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.path.join(p5_paths.RESULTS, "exp37_samp_holdout.json")
ALPHA = 0.10
K_LOCAL = 5
N_CLUST = 5


def group_of(code):
    return {0: "pure", 7: "binary", 8: "ternary"}.get(int(code), "single")


def clusters_by_formula(X, yr, n_clust=N_CLUST):
    """Return a label per scan: cluster id 0..n_clust-1, ordered by mean acquisition index."""
    lab = np.full(len(X), -1, int)
    sizes = {}
    for c in np.unique(yr):
        idx = np.where(yr == c)[0]
        Z = X[idx]
        C = np.corrcoef(Z)
        d = np.clip(1.0 - C, 0.0, None)
        np.fill_diagonal(d, 0.0)
        L = linkage(d[np.triu_indices(len(Z), 1)], method="average")
        cid = fcluster(L, t=n_clust, criterion="maxclust")
        order = [c0 for c0, _ in sorted(((c0, np.mean(np.where(cid == c0)[0]))
                                         for c0 in np.unique(cid)), key=lambda t: t[1])]
        remap = {c0: k for k, c0 in enumerate(order)}
        lab[idx] = [remap[v] for v in cid]
        sizes[int(c)] = [int(np.sum(lab[idx] == k)) for k in range(n_clust)]
    return lab, sizes


def macro_splits(yr, n_blocks=5):
    """Contiguous blocks of n/n_blocks scans; fold k tests block k."""
    order = np.arange(len(yr))
    pos = np.zeros(len(yr), int)
    for c in np.unique(yr):
        idx = np.where(yr == c)[0]
        pos[idx] = np.arange(len(idx)) // (len(idx) // n_blocks)
    out = []
    for k in range(n_blocks):
        test = order[pos == k]
        cal = order[(pos == (k + 1) % n_blocks) | (pos == (k + 2) % n_blocks)]
        fit = order[(pos == (k + 3) % n_blocks) | (pos == (k + 4) % n_blocks)]
        out.append((fit, cal, test))
    return out


def cluster_splits(lab, yr):
    """Fold k: test cluster k, calibration clusters k+1 and k+2, fit the remaining two."""
    out = []
    for k in range(N_CLUST):
        test = np.where(lab == k)[0]
        cal = np.where((lab == (k + 1) % N_CLUST) | (lab == (k + 2) % N_CLUST))[0]
        fit = np.where((lab == (k + 3) % N_CLUST) | (lab == (k + 4) % N_CLUST))[0]
        out.append((fit, cal, test))
    return out


def local_scale(wh, X, fit_idx, i, k=K_LOCAL):
    d = wh.dist(X[i][None, :] - X[fit_idx])
    k = max(1, min(k, len(d)))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def run(X, yr, cand, splits, alpha=ALPHA):
    rows = []
    for fit, cal, test in splits:
        m0, D, sigma = E.fit_dict(X, yr, fit)
        wh = E.Whitener(X, yr, fit)
        cand_m = np.stack([E.pred(m0, D, S) for S in cand])
        S_of = {S: k for k, S in enumerate(cand)}

        cal_sc = np.array([
            wh.dist(X[i][None, :] - cand_m[S_of[tuple(sorted(E.true_set(int(yr[i]))))]])[0]
            for i in cal])
        n = len(cal_sc)
        q = float(np.quantile(cal_sc, min(1.0, (1 - alpha) * (1 + 1 / n))))

        best_cal = np.array([float(np.min(wh.dist(X[i][None, :] - cand_m)))
                             for i in cal if 1 <= int(yr[i]) <= 6])
        tau = float(np.quantile(best_cal, min(1.0, (1 - alpha) * (1 + 1 / len(best_cal)))))

        for i in test:
            dvec = wh.dist(X[i][None, :] - cand_m)
            best = float(dvec.min())
            C = np.where(dvec <= q)[0]
            truth = tuple(sorted(E.true_set(int(yr[i]))))
            kt = S_of[truth]
            rank = int(np.where(np.argsort(dvec) == kt)[0][0])
            rows.append(dict(grp=group_of(int(yr[i])), covered=bool(kt in C),
                             size=int(len(C)), empty=bool(len(C) == 0),
                             top1=bool(rank == 0),
                             abstain=bool(best > tau), best=best))

    out = dict(coverage=round(float(np.mean([r["covered"] for r in rows])), 3),
               mean_size=round(float(np.mean([r["size"] for r in rows])), 2),
               empty=round(float(np.mean([r["empty"] for r in rows])), 3),
               top1=round(float(np.mean([r["top1"] for r in rows])), 3),
               abstain=round(float(np.mean([r["abstain"] for r in rows])), 3),
               q=round(q, 3), tau=round(tau, 3), n_test=len(rows))
    groups = {}
    for g in ("single", "binary", "ternary"):
        rr = [r for r in rows if r["grp"] == g]
        if rr:
            groups[g] = dict(n=len(rr),
                             coverage=round(float(np.mean([r["covered"] for r in rr])), 3),
                             mean_size=round(float(np.mean([r["size"] for r in rr])), 2),
                             top1=round(float(np.mean([r["top1"] for r in rr])), 3),
                             abstain=round(float(np.mean([r["abstain"] for r in rr])), 3))
    out["groups"] = groups
    return out


def n_eff(X, yr):
    rec = {}
    for c in np.unique(yr):
        Z = X[yr == c]
        R = Z - Z.mean(0)
        u, s, _ = np.linalg.svd(R, full_matrices=False)
        z = u[:, 0] * s[0]
        rho, tau = [], 1.0
        for k in range(1, 21):
            r = float(np.corrcoef(z[:-k], z[k:])[0, 1])
            if r <= 0:
                break
            rho.append(r)
            tau += 2 * r
        rec[int(c)] = dict(tau=round(tau, 2), n_eff=round(len(Z) / tau, 1),
                           pc1_var=round(float(s[0] ** 2 / np.sum(s ** 2)), 3),
                           n_scans=len(Z))
    vals = [v["n_eff"] for v in rec.values()]
    rec["mean"] = round(float(np.mean(vals)), 1)
    rec["median"] = round(float(np.median(vals)), 1)
    return rec


def main():
    X, yr = E.load()
    cand = E.subsets()
    rng = np.random.default_rng(0)
    lab, sizes = clusters_by_formula(X, yr)

    out = dict(alpha=ALPHA, k_local=K_LOCAL, n_clusters=N_CLUST,
               window=E.WINDOW, n_eff=n_eff(X, yr),
               cluster_sizes=sizes,
               cluster=run(X, yr, cand, cluster_splits(lab, yr)),
               macro=run(X, yr, cand, macro_splits(yr)),
               blocked=run(X, yr, cand, E.make_splits(yr, "blocked", rng)))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"cluster holdout vs blocked, whitened, alpha={ALPHA}, window={E.WINDOW}")
    for name in ("blocked", "macro", "cluster"):
        r = out[name]
        print(f"{name:8s} coverage={r['coverage']:.3f} size={r['mean_size']:5.1f} "
              f"top1={r['top1']:.3f} empty={r['empty']:.3f} abstain={r['abstain']:.3f}")
    print("\nper group:")
    for name in ("blocked", "macro", "cluster"):
        for g, v in out[name]["groups"].items():
            print(f"  {name:8s} {g:7s} n={v['n']:3d} cov={v['coverage']:.3f} "
                  f"size={v['mean_size']:5.1f} top1={v['top1']:.3f} abst={v['abstain']:.3f}")
    print("\nclusters per formulation:", sizes)
    print("n_eff:", out["n_eff"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
