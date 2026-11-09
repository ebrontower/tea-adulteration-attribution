"""exp39 - robustness of the two hyper-parameters and of the headline numbers.

Two questions a referee asks before believing Section 3, neither of which the manuscript
currently answers:

1. Is k = 60 a knife edge? The whitened metric truncates the residual-covariance inverse at
   sixty components. This script sweeps k over 10 ... 320 (and no whitening at all) under the
   blocked protocol and reports, for each k: the coherence and condition number of the whitened
   dictionary, the single-adulterant top-1, the conformal coverage, the mean set size, the
   abstention rate, and the gate's false-abstention / composite-abstention rates. A plateau is
   what makes k = 60 a choice rather than a tuned constant.

2. How wide are the headline numbers? Coverage, top-1, set size and the two gate rates are
   proportions over spectra that are not independent: the 50 scans of a formulation come in ten
   contiguous blocks (exp37: integrated autocorrelation 1.7-19 scans). Intervals are therefore
   taken by resampling whole blocks within each formulation (cluster bootstrap, 2000 draws),
   plus a ten-seed sweep of the random-split arm to show what the seed contributes. The gate's
   false-abstention rate is also given as a Wilson interval, and the 100/100 composite rejection
   as a one-sided exact bound.

Its results are written to out/exp39_robust.json in this package.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import exp33_whitened_attr as E  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.path.join(p5_paths.RESULTS, "exp39_robust.json")
ALPHA = 0.10
K_LOCAL = 5
KS = (10, 20, 40, 60, 80, 120, 160, 240, 320, 0)   # 0 = no whitening (Euclidean)
NBOOT = 2000
SEEDS = range(10)


def local_scale(wh, X, fit, i, k=K_LOCAL):
    V = X[i][None, :] - X[fit]
    d = wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)
    k = max(1, min(k, len(d)))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def eval_k(X, yr, cand, splits, k, alpha=ALPHA):
    rows = []
    for fit, cal, test in splits:
        m0, D, sigma = E.fit_dict(X, yr, fit)
        wh = E.Whitener(X, yr, fit, k) if k else None
        cand_m = np.stack([E.pred(m0, D, S) for S in cand])
        S_of = {S: j for j, S in enumerate(cand)}
        cal_sc = np.array([wh.dist(X[i][None, :] - cand_m[S_of[tuple(sorted(E.true_set(int(yr[i]))))]])[0]
                           if wh is not None else
                           np.linalg.norm(X[i] - cand_m[S_of[tuple(sorted(E.true_set(int(yr[i]))))]])
                           for i in cal])
        q = float(np.quantile(cal_sc, min(1.0, (1 - alpha) * (1 + 1 / len(cal_sc)))))

        best_cal = []
        for i in cal:
            if int(yr[i]) <= 6:
                v = X[i][None, :] - cand_m
                d = wh.dist(v) if wh is not None else np.linalg.norm(v, axis=1)
                best_cal.append(float(np.min(d)) / local_scale(wh, X, fit, i))
        tau = float(np.quantile(best_cal, min(1.0, (1 - alpha) * (1 + 1 / len(best_cal)))))

        b = E.block_ids(yr)
        for i in test:
            v = X[i][None, :] - cand_m
            sc = wh.dist(v) if wh is not None else np.linalg.norm(v, axis=1)
            kt = S_of[tuple(sorted(E.true_set(int(yr[i]))))]
            best = float(np.min(sc)) / local_scale(wh, X, fit, i)
            rows.append(dict(code=int(yr[i]), block=int(b[i]), fold=int(b[i]),
                             covered=bool(kt in np.where(sc <= q)[0]),
                             top1=bool(int(np.argmin(sc)) == kt),
                             size=int(np.sum(sc <= q)),
                             accepted=bool(best <= tau)))
    return rows


def geo_k(X, yr, k):
    """Same definitions as exp33.geometry, at a given truncation."""
    m0, D, sigma = E.fit_dict(X, yr, np.arange(len(X)))
    iu = np.triu_indices(6, 1)
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    coh_raw = float((Dn.T @ Dn)[iu].max())
    if k:
        wh = E.Whitener(X, yr, np.arange(len(X)), k)
        W = wh.coords(D.T)
        wn = np.linalg.norm(W, axis=1)
        Gn = (W @ W.T) / np.outer(wn, wn)
        Wm = wh.coords(m0[None, :])[0]
        gauge_w = (W @ Wm) / (wn * np.linalg.norm(Wm))
        coh_w = float(Gn[iu].max())
        cond_w = float(np.linalg.cond(Gn))
    else:
        gauge_w = (D.T @ m0) / (np.linalg.norm(D, axis=0) * np.linalg.norm(m0))
        coh_w = coh_raw
        cond_w = float(np.linalg.cond(Dn.T @ Dn))
    gauge_raw = (D.T @ m0) / (np.linalg.norm(D, axis=0) * np.linalg.norm(m0))
    return dict(coh=round(coh_raw, 3), coh_white=round(coh_w, 3), cond_white=round(cond_w, 2),
                gauge_raw_paraffin=round(float(gauge_raw[3]), 3),
                gauge_white_paraffin=round(float(gauge_w[3]), 3),
                gauge_white_absmax=round(float(np.abs(gauge_w).max()), 3))


def summarise(rows):
    return dict(coverage=round(float(np.mean([r["covered"] for r in rows])), 3),
                size=round(float(np.mean([r["size"] for r in rows])), 2),
                top1=round(float(np.mean([r["top1"] for r in rows])), 3),
                abstain=round(float(np.mean([not r["accepted"] for r in rows])), 3),
                single_top1=round(float(np.mean([r["top1"] for r in rows if r["code"] <= 6])), 3),
                single_abstain=round(float(np.mean([not r["accepted"] for r in rows if r["code"] <= 6])), 3),
                comp_abstain=round(float(np.mean([not r["accepted"] for r in rows if r["code"] >= 7])), 3),
                ternary_cover=round(float(np.mean([r["covered"] for r in rows if r["code"] == 8])), 3))


def cluster_boot(rows, nboot=NBOOT, seed=0, stat=None):
    """Resample (formulation, block) cells with replacement."""
    rng = np.random.default_rng(seed)
    cells = {}
    for r in rows:
        cells.setdefault((r["code"], r["block"]), []).append(r)
    keys = sorted(cells)
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for j in pick for r in cells[keys[j]]]
        out.append(stat(sample))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 3), round((c + h) / d, 3)


def main():
    X, yr = E.load()
    cand = E.subsets()
    rng = np.random.default_rng(0)
    splits_blocked = E.make_splits(yr, "blocked", rng)

    out = dict(alpha=ALPHA, window=E.WINDOW, k_local=K_LOCAL, ks=list(KS), sweep={},
               ci={}, seeds={})

    # ---- 1. k sweep ---------------------------------------------------------------
    for k in KS:
        rows = eval_k(X, yr, cand, splits_blocked, k)
        rec = summarise(rows)
        rec.update(geo_k(X, yr, k))
        out["sweep"][str(k)] = rec
        print("k", k, rec, flush=True)

    # ---- 2. intervals on the k = 60 blocked run -----------------------------------
    rows60 = eval_k(X, yr, cand, splits_blocked, 60)
    cov = cluster_boot(rows60, stat=lambda s: float(np.mean([r["covered"] for r in s])))
    top1 = cluster_boot(rows60, stat=lambda s: float(np.mean([r["top1"] for r in s])))
    size = cluster_boot(rows60, stat=lambda s: float(np.mean([r["size"] for r in s])))
    srows = [r for r in rows60 if r["code"] <= 6]
    fa = cluster_boot(srows, stat=lambda s: float(np.mean([not r["accepted"] for r in s])))
    crows = [r for r in rows60 if r["code"] >= 7]
    ca = cluster_boot(crows, stat=lambda s: float(np.mean([not r["accepted"] for r in s])))

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

    n_fa = int(sum(1 for r in srows if not r["accepted"]))
    n_ca = int(sum(1 for r in crows if not r["accepted"]))
    out["ci"] = dict(
        coverage=dict(point=round(float(np.mean([r["covered"] for r in rows60])), 3), ci=ci(cov)),
        top1=dict(point=round(float(np.mean([r["top1"] for r in rows60])), 3), ci=ci(top1)),
        size=dict(point=round(float(np.mean([r["size"] for r in rows60])), 2), ci=ci(size)),
        false_abstention=dict(point=round(n_fa / len(srows), 3), ci=ci(fa),
                              wilson=wilson(n_fa, len(srows)), n=len(srows)),
        composite_abstention=dict(point=round(n_ca / len(crows), 3), ci=ci(ca),
                                  wilson=wilson(n_ca, len(crows)), n=len(crows)),
        n_boot=NBOOT, block_effect="cluster bootstrap over (formulation, block) cells")
    print("CI", json.dumps(out["ci"], ensure_ascii=False), flush=True)

    # ---- 3. seed sweep of the random-split arm ------------------------------------
    seeds = []
    for s in SEEDS:
        r2 = np.random.default_rng(s)
        rows = eval_k(X, yr, cand, E.make_splits(yr, "random", r2), 60)
        seeds.append(summarise(rows))
    out["seeds"] = dict(
        folds=seeds,
        coverage=dict(mean=round(float(np.mean([x["coverage"] for x in seeds])), 3),
                      sd=round(float(np.std([x["coverage"] for x in seeds])), 3),
                      range=[round(min(x["coverage"] for x in seeds), 3),
                             round(max(x["coverage"] for x in seeds), 3)]),
        top1=dict(mean=round(float(np.mean([x["top1"] for x in seeds])), 3),
                  sd=round(float(np.std([x["top1"] for x in seeds])), 3),
                  range=[round(min(x["top1"] for x in seeds), 3),
                         round(max(x["top1"] for x in seeds), 3)]),
        comp_abstain=dict(mean=round(float(np.mean([x["comp_abstain"] for x in seeds])), 3),
                          range=[round(min(x["comp_abstain"] for x in seeds), 3),
                                 round(max(x["comp_abstain"] for x in seeds), 3)]))
    print("seeds", out["seeds"]["coverage"], out["seeds"]["top1"], flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\nk sweep, blocked, alpha={ALPHA}")
    print(f"{'k':>5s} {'cohW':>6s} {'condW':>7s} {'gauW':>7s} {'top1s':>6s} {'cover':>6s} "
          f"{'size':>6s} {'abst':>6s} {'tern':>6s}")
    for k in KS:
        s = out["sweep"][str(k)]
        print(f"{k:5d} {s['coh_white']:6.3f} {s['cond_white']:7.2f} {s['gauge_white_paraffin']:7.3f} "
              f"{s['single_top1']:6.3f} {s['coverage']:6.3f} {s['size']:6.2f} "
              f"{s['abstain']:6.3f} {s['ternary_cover']:6.3f}")
    print("\nintervals (cluster bootstrap, %d draws):" % NBOOT)
    for k, v in out["ci"].items():
        if isinstance(v, dict):
            print(" ", k, v)
    print("saved", OUT)


if __name__ == "__main__":
    main()
