"""exp36 - open-set test: what happens when the adulterant is not in the dictionary?

The manuscript claims that a sample outside the dictionary is returned as "not attributable"
rather than assigned to the nearest composition. Every experiment so far tested that claim on
*composites of dictionary components* (formulas 7 and 8). The deployment case is different: an
adulterant that was never among the reference increments. This script builds, for each of the six
single adulterants in turn, a dictionary from pure tea + the other five, calibrates the conformal
threshold q and the abstention threshold tau exactly as in Sections 2.5-2.6 (singles only,
blocked splits, whitened metric, alpha = 0.10), and then tests the 50 spectra of the held-out
adulterant.

Reported per held-out adulterant, over the five blocked folds (50 test spectra each):
  reject      fraction the gate sends to "not attributable"
  empty       fraction whose conformal set is empty (the coverage-rule form of abstention)
  accepted    fraction that receive a composition anyway (1 - reject)
  top1_wrong  among accepted spectra, the most frequent attributed composition and its share
  orphan      whitened distance of the held-out increment to the span of the five dictionary
              increments, relative to its own whitened norm (1 = fully outside, 0 = inside)
  gauge_cos   cosine of the held-out increment with the pure-tea spectrum
  coh         largest raw cosine between the held-out increment and a dictionary increment

A control row repeats the identical pipeline on the five in-dictionary singles of the same fold,
so the open-set rejection rate can be read against the in-set false-abstention rate.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import exp33_whitened_attr as E  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.path.join(p5_paths.RESULTS, "exp36_open_set.json")
NAMES = ["sugar", "talc", "sulfur", "paraffin", "dye", "flavouring"]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
ALPHA = 0.10
K_LOCAL = 5


def local_scale(wh, X, fit_idx, i, k=K_LOCAL):
    V = X[i][None, :] - X[fit_idx]
    d = wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)
    k = max(1, min(k, len(d)))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def auc(pos, neg):
    """P(score_pos > score_neg), rank-based, for anomaly scores."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        k = i
        while k + 1 < len(sv) and sv[k + 1] == sv[i]:
            k += 1
        ranks[order[i:k + 1]] = 0.5 * (i + k) + 1
        i = k + 1
    rp = ranks[:len(pos)].sum()
    return round(float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def subsets(n):
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def dict_from(X, yr, idx, exclude_code):
    """Means of pure tea and of every single adulterant present except exclude_code."""
    cls = [c for c in sorted(set(yr[idx].tolist())) if 1 <= c <= 6 and c != exclude_code]
    mu = {0: X[idx][yr[idx] == 0].mean(0)}
    for c in cls:
        mu[c] = X[idx][yr[idx] == c].mean(0)
    m0 = mu[0]
    codes = cls                                   # recipe codes, e.g. [1,2,4,5,6]
    D = np.stack([mu[c] - m0 for c in codes], 1)
    res = np.concatenate([X[idx][yr[idx] == c] - mu[c] for c in mu], 0)
    sigma = float(np.sqrt(np.mean(res ** 2)))
    return m0, D, sigma, codes


def orphan_geometry(X, yr, j_code):
    """Whitened distance of increment j to the span of the other five (descriptive, full data)."""
    m0, D, _, codes = dict_from(X, yr, np.arange(len(X)), j_code)
    wh = E.Whitener(X, yr, np.arange(len(X)))
    W = wh.coords(D.T)                            # (5, p_w)
    mu_j = X[yr == j_code].mean(0)
    wj = wh.coords((mu_j - m0)[None, :])[0]
    beta, *_ = np.linalg.lstsq(W.T, wj, rcond=None)
    return dict(
        orphan=round(float(np.linalg.norm(wj - W.T @ beta) / np.linalg.norm(wj)), 4),
        gauge_cos=round(float((mu_j - m0) @ m0 / (np.linalg.norm(mu_j - m0) * np.linalg.norm(m0))), 4),
        coh=round(float(np.max(np.abs((D / np.linalg.norm(D, axis=0)[None, :]).T @
                                      ((mu_j - m0) / np.linalg.norm(mu_j - m0))))), 4),
        codes=[NAMES[c - 1] for c in codes])


def main():
    X, yr = E.load()
    rng = np.random.default_rng(0)
    splits = E.make_splits(yr, "blocked", rng)

    out = dict(alpha=ALPHA, alphas=ALPHAS, k_local=K_LOCAL, window=E.WINDOW,
               held_out={}, control={}, geometry={})
    rows = {j: [] for j in range(1, 7)}
    ctrl = {j: [] for j in range(1, 7)}

    for j in range(1, 7):
        out["geometry"][NAMES[j - 1]] = orphan_geometry(X, yr, j)
        for fit, cal, test in splits:
            fit_j = fit[yr[fit] != j]
            cal_j = np.array([i for i in cal if int(yr[i]) == 0 or (1 <= int(yr[i]) <= 6 and int(yr[i]) != j)])
            if len(cal_j) == 0:
                continue
            m0, D, sigma, codes = dict_from(X, yr, fit_j, j)
            wh = E.Whitener(X, yr, fit_j)
            map_local = {c: k for k, c in enumerate(codes)}          # recipe code -> column
            cand = subsets(len(codes))
            cand_m = np.stack([E.pred(m0, D, S) for S in cand])
            S_of = {S: k for k, S in enumerate(cand)}

            s_cal = np.array([local_scale(wh, X, fit_j, i) for i in cal_j])
            true_local = np.array([map_local[int(yr[i])] for i in cal_j if int(yr[i]) in map_local])
            keep = [k for k, i in enumerate(cal_j) if int(yr[i]) in map_local]
            cal_sc = np.array([
                wh.dist(X[cal_j[k]][None, :] - cand_m[S_of[(true_local[t],)]])[0] / s_cal[k]
                for t, k in enumerate(keep)])
            best_cal = np.array([
                float(np.min(wh.dist(X[i][None, :] - cand_m)) / local_scale(wh, X, fit_j, i))
                for i in cal_j if 1 <= int(yr[i]) <= 6])
            best_abs_cal = np.array([
                float(np.min(wh.dist(X[i][None, :] - cand_m)))
                for i in cal_j if 1 <= int(yr[i]) <= 6])
            n_c, n_b = len(cal_sc), len(best_cal)

            q = {a: float(np.quantile(cal_sc, min(1.0, (1 - a) * (1 + 1 / n_c)))) for a in ALPHAS}
            tau = {a: float(np.quantile(best_cal, min(1.0, (1 - a) * (1 + 1 / n_b)))) for a in ALPHAS}
            tau_abs = {a: float(np.quantile(best_abs_cal, min(1.0, (1 - a) * (1 + 1 / n_b)))) for a in ALPHAS}
            tau_u = {}
            for a in ALPHAS:
                u_cal = np.maximum(best_cal / tau[a], best_abs_cal / tau_abs[a])
                tau_u[a] = float(np.quantile(u_cal, min(1.0, (1 - a) * (1 + 1 / n_b))))

            # ---- held-out adulterant spectra ------------------------------------
            for i in test[yr[test] == j]:
                dvec = wh.dist(X[i][None, :] - cand_m)
                sc = dvec / local_scale(wh, X, fit_j, i)
                best = float(sc.min())
                abs_best = float(dvec.min())
                kbest = int(np.argmin(sc))
                comp = tuple(NAMES[codes[k] - 1] for k in cand[kbest])
                empty = {a: bool(np.sum(sc <= q[a]) == 0) for a in ALPHAS}
                rows[j].append(dict(comp=comp, best=best, abs=abs_best, empty=empty,
                                    reject={a: bool(best > tau[a]) for a in ALPHAS},
                                    reject_abs={a: bool(abs_best > tau_abs[a]) for a in ALPHAS},
                                    reject_u={a: bool(max(best / tau[a], abs_best / tau_abs[a]) > tau_u[a])
                                              for a in ALPHAS}))

            # ---- in-dictionary single controls ----------------------------------
            for i in test[(yr[test] != j) & (yr[test] >= 1) & (yr[test] <= 6)]:
                dvec = wh.dist(X[i][None, :] - cand_m)
                sc = dvec / local_scale(wh, X, fit_j, i)
                best = float(sc.min())
                abs_best = float(dvec.min())
                top1 = bool(int(np.argmin(sc)) == S_of[(map_local[int(yr[i])],)])
                ctrl[j].append(dict(top1=top1, best=best, abs=abs_best,
                                    reject={a: bool(best > tau[a]) for a in ALPHAS},
                                    reject_abs={a: bool(abs_best > tau_abs[a]) for a in ALPHAS},
                                    reject_u={a: bool(max(best / tau[a], abs_best / tau_abs[a]) > tau_u[a])
                                              for a in ALPHAS}))

    for j in range(1, 7):
        rr = rows[j]
        cc = ctrl[j]
        rec = dict(n=len(rr),
                   reject={f"{a:.2f}": round(float(np.mean([r["reject"][a] for r in rr])), 3) for a in ALPHAS},
                   reject_abs={f"{a:.2f}": round(float(np.mean([r["reject_abs"][a] for r in rr])), 3) for a in ALPHAS},
                   reject_u={f"{a:.2f}": round(float(np.mean([r["reject_u"][a] for r in rr])), 3) for a in ALPHAS},
                   empty={f"{a:.2f}": round(float(np.mean([r["empty"][a] for r in rr])), 3) for a in ALPHAS},
                   accept=round(float(np.mean([not r["reject"][ALPHA] for r in rr])), 3),
                   compositions={})
        acc = [r for r in rr if not r["reject"][ALPHA]]
        for r in acc:
            key = "+".join(r["comp"]) if r["comp"] else "pure"
            rec["compositions"][key] = rec["compositions"].get(key, 0) + 1
        rec["compositions"] = dict(sorted(rec["compositions"].items(), key=lambda kv: -kv[1])[:5])
        rec["top1"] = (round(max(rec["compositions"].values()) / len(acc), 3) if acc else None)
        rec["top1_label"] = (max(rec["compositions"], key=rec["compositions"].get) if acc else None)
        rec["auc_norm"] = auc([r["best"] for r in rr], [c["best"] for c in cc])
        rec["auc_abs"] = auc([r["abs"] for r in rr], [c["abs"] for c in cc])
        out["held_out"][NAMES[j - 1]] = rec

        out["control"][NAMES[j - 1]] = dict(
            n=len(cc),
            accept=round(float(np.mean([not r["reject"][ALPHA] for r in cc])), 3),
            top1=round(float(np.mean([r["top1"] for r in cc])), 3),
            reject={f"{a:.2f}": round(float(np.mean([r["reject"][a] for r in cc])), 3) for a in ALPHAS},
            reject_abs={f"{a:.2f}": round(float(np.mean([r["reject_abs"][a] for r in cc])), 3) for a in ALPHAS},
            reject_u={f"{a:.2f}": round(float(np.mean([r["reject_u"][a] for r in cc])), 3) for a in ALPHAS})

    pool_o = [r for j in range(1, 7) for r in rows[j]]
    pool_c = [r for j in range(1, 7) for r in ctrl[j]]
    out["pooled"] = dict(auc_norm=auc([r["best"] for r in pool_o], [c["best"] for c in pool_c]),
                         auc_abs=auc([r["abs"] for r in pool_o], [c["abs"] for c in pool_c]),
                         n_orphan=len(pool_o), n_control=len(pool_c))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---- console table --------------------------------------------------------
    print(f"open-set, blocked, whitened metric, alpha={ALPHA}, window={E.WINDOW}")
    print(f"{'held-out':11s} {'orphan':>7s} {'gauge':>6s} {'coh':>6s} | "
          f"{'rejN':>6s} {'rejA':>6s} {'rejU':>6s} {'empty':>6s} {'accept':>6s} {'top1':>5s} "
          f"{'aucN':>5s} {'aucA':>5s}  attributed to")
    for j in range(1, 7):
        nm = NAMES[j - 1]
        g = out["geometry"][nm]
        h = out["held_out"][nm]
        print(f"{nm:11s} {g['orphan']:7.3f} {g['gauge_cos']:6.3f} {g['coh']:6.3f} | "
              f"{h['reject'][f'{ALPHA:.2f}']:6.3f} {h['reject_abs'][f'{ALPHA:.2f}']:6.3f} "
              f"{h['reject_u'][f'{ALPHA:.2f}']:6.3f} "
              f"{h['empty'][f'{ALPHA:.2f}']:6.3f} "
              f"{h['accept']:6.3f} {h['top1'] if h['top1'] is not None else 0:5.3f} "
              f"{h['auc_norm']:5.3f} {h['auc_abs']:5.3f}  "
              f"{h['top1_label']} ({h['compositions']})")
    p = out["pooled"]
    print(f"\npooled: AUC normalised {p['auc_norm']}  AUC absolute {p['auc_abs']}  "
          f"(orphan n={p['n_orphan']}, in-dictionary n={p['n_control']})")
    print("control (in-dictionary singles, same folds):")
    for j in range(1, 7):
        c = out["control"][NAMES[j - 1]]
        print(f"  without {NAMES[j-1]:11s} n={c['n']:4d} accept={c['accept']:.3f} "
              f"top1={c['top1']:.3f} rejN@0.10={c['reject']['0.10']:.3f} "
              f"rejA@0.10={c['reject_abs']['0.10']:.3f} rejU@0.10={c['reject_u']['0.10']:.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
