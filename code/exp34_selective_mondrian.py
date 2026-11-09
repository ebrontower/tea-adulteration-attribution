"""exp34 - E7: per-group coverage, selective abstention, locally adaptive scores, alpha map.

exp33 showed that one marginal 90% guarantee can hide a 0.00 conditional one (ternary mixture).
This script answers the three questions a referee will ask of that table.

1. Group-conditional coverage. Report coverage separately for singles, the binary mixture and the
   ternary mixture, so the failure is visible instead of averaged away.

2. Selective attribution. A single rule that can be computed at inference time: fit the best
   dictionary composition, and if its residual exceeds the (1-alpha) quantile of that residual on
   singles, refuse to attribute. Report acceptance (1 - false abstention) on singles, acceptance
   on mixtures, abstention rate, and coverage conditional on acceptance. This is the deployable
   form of "detectable but not attributable".

3. Locally adaptive score. Normalise the nonconformity score by a local scale
   sigma_hat(x) = mean whitened distance to the k nearest fit spectra. Standard locally-adaptive
   conformal; the question is whether it absorbs the block-level drift that inflates q under the
   blocked protocol.

4. Alpha map. Set size and abstention as a function of alpha, for the standard and the normalised
   score, so the operating point can be chosen instead of asserted.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp33_whitened_attr as E  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.path.join(p5_paths.RESULTS, "exp34_selective_mondrian.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ALPHAS = (0.05, 0.10, 0.15, 0.20)
K_LOCAL = 5
GROUP = {0: "single", 1: "single", 2: "single", 3: "single", 4: "single", 5: "single",
         6: "single", 7: "binary", 8: "ternary"}


def group_of(code):
    return GROUP[int(code)]


def scores_for(wh, X, cand_m, i, idx_all):
    V = X[i][None, :] - cand_m
    return wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)


def local_scale(wh, X, fit_idx, i, k=K_LOCAL):
    V = X[i][None, :] - X[fit_idx]
    d = wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)
    k = max(1, min(k, len(d)))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def run(X, yr, cand, splits, metric, alpha, local):
    rows = []
    for fit, cal, test in splits:
        m0, D, sigma = E.fit_dict(X, yr, fit)
        wh = E.Whitener(X, yr, fit) if metric == "white" else None
        cand_m = np.stack([E.pred(m0, D, S) for S in cand])
        S_of = {S: k for k, S in enumerate(cand)}

        if local:
            s_cal = np.array([local_scale(wh, X, fit, i) for i in cal])
            s_test = np.array([local_scale(wh, X, fit, i) for i in test])
        else:
            s_cal = np.ones(len(cal))
            s_test = np.ones(len(test))

        cal_sc = np.array([
            scores_for(wh, X, cand_m, i, cal)[S_of[tuple(sorted(E.true_set(int(yr[i]))))]] / s_cal[j]
            for j, i in enumerate(cal)])
        n = len(cal_sc)
        q = float(np.quantile(cal_sc, min(1.0, (1 - alpha) * (1 + 1 / n))))

        # model-class gate: residual of the best candidate, calibrated on singles only
        best_cal = []
        for i in cal:
            if int(yr[i]) <= 6:
                best_cal.append(float(np.min(scores_for(wh, X, cand_m, i, cal)) / s_cal[list(cal).index(i)]))
        tau = float(np.quantile(best_cal, min(1.0, (1 - alpha) * (1 + 1 / len(best_cal)))))

        for j, i in enumerate(test):
            sc = scores_for(wh, X, cand_m, i, test) / s_test[j]
            C = np.where(sc <= q)[0]
            truth = tuple(sorted(E.true_set(int(yr[i]))))
            kt = S_of[truth]
            order = np.argsort(sc)
            rank = int(np.where(order == kt)[0][0])
            best = float(sc[order[0]])
            rows.append(dict(grp=group_of(int(yr[i])), covered=bool(kt in C), size=int(len(C)),
                             empty=bool(len(C) == 0), rank=rank, best=best,
                             accepted=bool(best <= tau)))

    out = dict(coverage=round(float(np.mean([r["covered"] for r in rows])), 3),
               mean_size=round(float(np.mean([r["size"] for r in rows])), 2),
               abstain=round(float(np.mean([r["empty"] for r in rows])), 3),
               top1=round(float(np.mean([r["rank"] == 0 for r in rows])), 3),
               q=round(q, 3), tau=round(tau, 3), n_test=len(rows))
    groups = {}
    for g in ("single", "binary", "ternary"):
        rr = [r for r in rows if r["grp"] == g]
        if rr:
            groups[g] = dict(n=len(rr), coverage=round(float(np.mean([r["covered"] for r in rr])), 3),
                             mean_size=round(float(np.mean([r["size"] for r in rr])), 2),
                             top1=round(float(np.mean([r["rank"] == 0 for r in rr])), 3),
                             accept=round(float(np.mean([r["accepted"] for r in rr])), 3))
    acc = [r for r in rows if r["accepted"]]
    out["groups"] = groups
    out["selective"] = dict(
        accept_rate=round(float(np.mean([r["accepted"] for r in rows])), 3),
        cond_coverage=round(float(np.mean([r["covered"] for r in acc])), 3) if acc else None,
        cond_mean_size=round(float(np.mean([r["size"] for r in acc])), 2) if acc else None,
        cond_top1=round(float(np.mean([r["rank"] == 0 for r in acc])), 3) if acc else None)
    return out


def main():
    X, yr = E.load()
    cand = E.subsets()
    rng = np.random.default_rng(0)
    splits = {m: E.make_splits(yr, m, rng) for m in ("random", "blocked")}

    out = dict(geometry=E.geometry(X, yr), alphas=list(ALPHAS), k_local=K_LOCAL)
    for metric in ("euclid", "white"):
        for mode in ("random", "blocked"):
            key = f"{metric}|{mode}"
            out[key] = {}
            for local in (False, True):
                name = "normalized" if local else "standard"
                out[key][name] = run(X, yr, cand, splits[mode], metric, 0.10, local)
            print(key, json.dumps(out[key], ensure_ascii=False), flush=True)

    # alpha map on the deployable arm (white), blocked and random
    for mode in ("random", "blocked"):
        out[f"alpha_map|{mode}"] = {}
        for local in (False, True):
            name = "normalized" if local else "standard"
            out[f"alpha_map|{mode}"][name] = {
                str(a): run(X, yr, cand, splits[mode], "white", a, local) for a in ALPHAS}
        print("alpha_map", mode, json.dumps(out[f"alpha_map|{mode}"], ensure_ascii=False), flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
