"""exp38 - group-conditional (Mondrian) calibration: can the ternary blind spot be fixed?

Section 3.6 reports a marginal 90% guarantee that hides a 0.28 conditional one on the ternary
mixture. Split conformal has a standard answer to that: calibrate the quantile separately inside
each group (Mondrian / label-conditional conformal). This script asks whether the answer works
here, and where it stops working.

Arms, all under the blocked protocol with the whitened metric:

  marginal      one quantile from all calibration spectra (the number in the manuscript)
  mondrian_true grouping by the *true* recipe (single / binary / ternary) - an oracle, it is
                not available at inference time and is reported only to bound what conditioning
                can buy
  mondrian_pred grouping by the *fitted* composition: the cardinality of the best candidate
                (<=1 -> single, 2 -> binary, >=3 -> ternary). This is computable at inference
                time and is the deployable version.
  mondrian_pred + abstain
                identical, but a test spectrum whose estimated group has no calibration member
                is returned as "not attributable" instead of receiving a fallback quantile.

The zero-shot arm repeats the calibration on single adulterants only and tests on the two
composite formulations, which are absent from fit and calibration. There the ternary group has
no calibration member by construction; the script reports the calibration mass actually available
and how far the ternary scores lie beyond it (the ratio of the smallest ternary score to the
largest calibration score, and the alpha at which the quantile would reach it).

Reported per arm and per alpha: marginal coverage, per-group coverage, mean set size (overall and
per group), empty-set rate (= abstention), coverage conditional on acceptance, and how often the
estimated group equals the true group.
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

OUT = os.path.join(p5_paths.RESULTS, "exp38_mondrian.json")
ALPHAS = (0.05, 0.10, 0.15, 0.20)
ALPHA = 0.10
GROUPS = ("single", "binary", "ternary")


def group_true(code):
    c = int(code)
    if c in (7,):
        return "binary"
    if c in (8,):
        return "ternary"
    return "single"


def group_est(size):
    return "single" if size <= 1 else "binary" if size == 2 else "ternary"


def quant(scores, alpha):
    n = len(scores)
    if n == 0:
        return None
    return float(np.quantile(scores, min(1.0, (1 - alpha) * (1 + 1 / n))))


def fold_arm(X, yr, fit, cal, test, cand, alpha):
    """Return one record per test spectrum for the four arms of a single fold."""
    m0, D, sigma = E.fit_dict(X, yr, fit)
    wh = E.Whitener(X, yr, fit)
    cand_m = np.stack([E.pred(m0, D, S) for S in cand])
    S_of = {S: k for k, S in enumerate(cand)}

    cal_sc = np.array([wh.dist(X[i][None, :] - cand_m[S_of[tuple(sorted(E.true_set(int(yr[i]))))]])[0]
                       for i in cal])
    cal_best = np.argmin(np.stack([wh.dist(X[i][None, :] - cand_m) for i in cal]), 1)
    cal_est = np.array([group_est(len(cand[k])) for k in cal_best])
    cal_true = np.array([group_true(yr[i]) for i in cal])

    q_m = quant(cal_sc, alpha)
    q_true = {g: quant(cal_sc[cal_true == g], alpha) for g in GROUPS}
    q_pred = {g: quant(cal_sc[cal_est == g], alpha) for g in GROUPS}
    mass = {g: int(np.sum(cal_est == g)) for g in GROUPS}

    # deployable group estimate: nearest centroid in the whitened space, trained on fit
    Zf = wh.coords(X[fit])
    lab_f = np.array([group_true(c) for c in yr[fit]])
    cent = {g: Zf[lab_f == g].mean(0) for g in GROUPS if np.any(lab_f == g)}

    def clf_group(idx):
        Z = wh.coords(X[idx])
        dd = np.stack([np.linalg.norm(Z - cent[g][None, :], axis=1) for g in cent], 1)
        names = list(cent)
        return [names[k] for k in np.argmin(dd, 1)]

    cal_clf = np.array(clf_group(cal))
    q_clf = {g: quant(cal_sc[cal_clf == g], alpha) for g in GROUPS}
    mass_clf = {g: int(np.sum(cal_clf == g)) for g in GROUPS}

    recs = []
    gclf_test = clf_group(test)
    for t, i in enumerate(test):
        sc = wh.dist(X[i][None, :] - cand_m)
        ktrue = S_of[tuple(sorted(E.true_set(int(yr[i]))))]
        kb = int(np.argmin(sc))
        ge = group_est(len(cand[kb]))
        gt = group_true(yr[i])
        gc = gclf_test[t]
        row = dict(gt=gt, ge=ge, gc=gc, kbest=kb)
        for arm, q in (("marginal", q_m),
                       ("mondrian_true", q_true[gt]),
                       ("mondrian_pred", q_pred[ge] if q_pred[ge] is not None else q_m),
                       ("mondrian_abstain", q_pred[ge]),
                       ("mondrian_clf", q_clf[gc] if q_clf.get(gc) is not None else q_m)):
            if q is None:
                row[arm] = dict(covered=False, size=0, empty=True, accepted=False)
                continue
            C = np.where(sc <= q)[0]
            row[arm] = dict(covered=bool(ktrue in C), size=int(len(C)),
                            empty=bool(len(C) == 0), accepted=bool(len(C) > 0))
        recs.append(row)
    return recs, dict(q_marginal=round(q_m, 3),
                      q_true={k: (round(v, 3) if v is not None else None) for k, v in q_true.items()},
                      q_pred={k: (round(v, 3) if v is not None else None) for k, v in q_pred.items()},
                      q_clf={k: (round(v, 3) if v is not None else None) for k, v in q_clf.items()},
                      cal_mass={g: mass[g] for g in GROUPS},
                      cal_mass_clf={g: mass_clf[g] for g in GROUPS},
                      clf_acc=round(float(np.mean(cal_clf == cal_true)), 3),
                      n_cal=len(cal))


def summarise(recs, arm):
    rr = [r[arm] for r in recs]
    out = dict(
        coverage=round(float(np.mean([x["covered"] for x in rr])), 3),
        size=round(float(np.mean([x["size"] for x in rr])), 2),
        empty=round(float(np.mean([x["empty"] for x in rr])), 3),
        cond_coverage=(round(float(np.mean([x["covered"] for x in rr if x["accepted"]])), 3)
                       if any(x["accepted"] for x in rr) else None),
        groups={})
    for g in GROUPS:
        gg = [r[arm] for r in recs if r["gt"] == g]
        if gg:
            out["groups"][g] = dict(
                coverage=round(float(np.mean([x["covered"] for x in gg])), 3),
                size=round(float(np.mean([x["size"] for x in gg])), 2),
                empty=round(float(np.mean([x["empty"] for x in gg])), 3),
                cond_coverage=(round(float(np.mean([x["covered"] for x in gg if x["accepted"]])), 3)
                               if any(x["accepted"] for x in gg) else None))
    return out


def main():
    X, yr = E.load()
    cand = E.subsets()
    rng = np.random.default_rng(0)
    splits = E.make_splits(yr, "blocked", rng)

    out = dict(alpha=ALPHA, alphas=list(ALPHAS), window=E.WINDOW, arms={}, folds={},
               group_identifiability={}, zero_shot={})

    per_alpha = {a: [] for a in ALPHAS}
    for a in ALPHAS:
        for fit, cal, test in splits:
            recs, info = fold_arm(X, yr, fit, cal, test, cand, a)
            per_alpha[a].append((recs, info))

    for a in ALPHAS:
        out["arms"][f"{a:.2f}"] = {}
        for arm in ("marginal", "mondrian_true", "mondrian_pred", "mondrian_abstain", "mondrian_clf"):
            pooled = [r for recs, _ in per_alpha[a] for r in recs]
            out["arms"][f"{a:.2f}"][arm] = summarise(pooled, arm)
        if a == ALPHA:
            out["folds"] = {
                "q_marginal": [i["q_marginal"] for _, i in per_alpha[a]],
                "q_true": {g: [i["q_true"][g] for _, i in per_alpha[a]] for g in GROUPS},
                "q_pred": {g: [i["q_pred"][g] for _, i in per_alpha[a]] for g in GROUPS},
                "cal_mass": {g: [i["cal_mass"][g] for _, i in per_alpha[a]] for g in GROUPS},
                "cal_mass_clf": {g: [i["cal_mass_clf"][g] for _, i in per_alpha[a]] for g in GROUPS},
                "q_clf": {g: [i["q_clf"][g] for _, i in per_alpha[a]] for g in GROUPS},
                "clf_acc": [i["clf_acc"] for _, i in per_alpha[a]]}

    pooled = [r for recs, _ in per_alpha[ALPHA] for r in recs]
    out["group_identifiability"] = {
        "accuracy": round(float(np.mean([r["ge"] == r["gt"] for r in pooled])), 3),
        "confusion": {gt: {ge: int(sum(1 for r in pooled if r["gt"] == gt and r["ge"] == ge))
                           for ge in GROUPS} for gt in GROUPS}}

    # ---- zero-shot: fit and calibrate on singles only, test on the two composites ----------
    sel = np.where(yr <= 6)[0]
    b = E.block_ids(yr)
    fitB = sel[b[sel] < 5]
    calB = sel[b[sel] >= 5]
    m0B, DB, _ = E.fit_dict(X, yr, fitB)
    whB = E.Whitener(X, yr, fitB)
    candB = np.stack([E.pred(m0B, DB, S) for S in cand])
    S_of_B = {S: j for j, S in enumerate(cand)}
    calB_sc = np.array([whB.dist(X[i][None, :] - candB[S_of_B[tuple(sorted(E.true_set(int(yr[i]))))]])[0]
                        for i in calB])
    zs = {}
    for a in ALPHAS:
        q = quant(calB_sc, a)
        cov, sizes, empty, cov_g, empty_g = [], [], [], {7: [], 8: []}, {7: [], 8: []}
        for k in range(5):
            test = np.where((b % 5 == k) & (yr >= 7))[0]
            for i in test:
                sc = whB.dist(X[i][None, :] - candB)
                kt = S_of_B[tuple(sorted(E.true_set(int(yr[i]))))]
                C = np.where(sc <= q)[0]
                c7 = int(yr[i]) == 7
                cov.append(bool(kt in C)); sizes.append(len(C)); empty.append(len(C) == 0)
                cov_g[7 if c7 else 8].append(bool(kt in C))
                empty_g[7 if c7 else 8].append(len(C) == 0)
        zs[f"{a:.2f}"] = dict(q=round(q, 3),
                              coverage=round(float(np.mean(cov)), 3),
                              size=round(float(np.mean(sizes)), 2),
                              empty=round(float(np.mean(empty)), 3),
                              binary_coverage=round(float(np.mean(cov_g[7])), 3),
                              ternary_coverage=round(float(np.mean(cov_g[8])), 3),
                              binary_empty=round(float(np.mean(empty_g[7])), 3),
                              ternary_empty=round(float(np.mean(empty_g[8])), 3))
    out["zero_shot"] = zs

    # how far the composite scores lie beyond the calibration support
    t8 = np.where(yr == 8)[0]
    t8_sc = np.array([whB.dist(X[i][None, :] - candB[S_of_B[tuple(sorted(E.true_set(8)))]])[0]
                      for i in t8])
    frac_above = float(np.mean(calB_sc >= t8_sc.min()))
    out["zero_shot_support"] = dict(
        n_cal=len(calB),
        cal_max=round(float(calB_sc.max()), 3),
        ternary_min=round(float(t8_sc.min()), 3),
        ratio=round(float(t8_sc.min() / calB_sc.max()), 3),
        cal_fraction_above=round(frac_above, 3),
        alpha_needed=round(float(1 - frac_above), 4))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"Mondrian calibration, blocked, whitened, alpha={ALPHA}, window={E.WINDOW}")
    print(f"{'arm':18s} {'cover':>6s} {'size':>6s} {'empty':>6s} | "
          f"{'single':>7s} {'binary':>7s} {'ternary':>7s} | {'size s/b/t':>16s}")
    for arm in ("marginal", "mondrian_true", "mondrian_pred", "mondrian_abstain", "mondrian_clf"):
        s = out["arms"][f"{ALPHA:.2f}"][arm]
        g = s["groups"]
        print(f"{arm:18s} {s['coverage']:6.3f} {s['size']:6.2f} {s['empty']:6.3f} | "
              f"{g['single']['coverage']:7.3f} {g['binary']['coverage']:7.3f} "
              f"{g['ternary']['coverage']:7.3f} | "
              f"{g['single']['size']:5.1f}/{g['binary']['size']:4.1f}/{g['ternary']['size']:4.1f}")
    print("\ncalibration mass per group (mean over folds):",
          {k: round(float(np.mean(v)), 1) for k, v in out["folds"]["cal_mass"].items()})
    print("q per fold, marginal:", out["folds"]["q_marginal"])
    print("q true-group  :", out["folds"]["q_true"])
    print("q pred-group  :", out["folds"]["q_pred"])
    print("q clf-group   :", out["folds"]["q_clf"])
    print("clf cal acc   :", out["folds"]["clf_acc"], "mass", out["folds"]["cal_mass_clf"])
    print("group identifiability:", out["group_identifiability"])
    print("\nzero-shot (calibrated on singles, tested on composites):")
    for a, v in zs.items():
        print(f"  a={a} cover={v['coverage']:.3f} size={v['size']:.1f} empty={v['empty']:.3f} "
              f"| binary {v['binary_coverage']:.3f} ternary {v['ternary_coverage']:.3f}")
    print("support:", out["zero_shot_support"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
