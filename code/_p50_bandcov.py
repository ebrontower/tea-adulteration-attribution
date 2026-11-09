"""exp50 - band-conditional coverage + a formal batch-baseline (MSC) control.

Pre-registration (written BEFORE this run; arms / corpora / criteria are fixed there):
    preregistrations/exp50_conditional_coverage_and_baseline_correction_v1.md

Two documented collapses motivate it (both already measured, not new guesses):
  (1) band inhomogeneity - exp49 S L2: the whitened arm's MARGINAL guarantee does not answer per
      dose band (1/2/3/4% coverage 0.813/0.787/0.820/0.773 while 6-12% is over-covered
      0.993-1.000); a single global quantile redistributes coverage along the dose axis.
  (2) fold collapse - exp51 S6: holding out one displaced preparation punches a whole fold through
      the floor (D1 f4 = 0.143, D2 f4 = 0.392, D3 f3 = 0.578, HONEY f3 = 0.333).
Quantile selection can in principle fix (1) and cannot fix (2); an affine baseline correction can in
principle fix (2) and is the formal batch-covariate control exp49 S5 owed. Both are priced here, and
"(2) is not repairable" is a pre-registered legal outcome (C5's second branch).

Arms (same folds, same dictionary, same whitener, same alpha = 0.10):
  quantile arm
    `marg`        one split-conformal quantile over the whole calibration pool - identical to
                  exp51's arm; it must reproduce logs/exp51_xcorpus.json to 1e-9 (criterion C1)
    `band`        Mondrian quantile per band. Band = unlabeled e(x) = d(x, authentic atom), cuts =
                  tertiles of e over the FIT pool only, 3 bands fixed before the run. A band whose
                  calibration pool is < 20 falls back to the global quantile and is flagged.
    `band_oracle` grouping by the tertile band of the DECLARED level - prices conditioning only;
                  never a claim, never quoted as a guarantee.
  correction arm  `abs` (deposited), `snv`, `msc` = per-spectrum affine regression (slope+intercept)
                  on the FIT-POOL mean spectrum with the residual kept; the dictionary, whitener,
                  local scale and quantiles are all refit on the corrected spectra.
  metric arm      `white_a` (main), `euclid` (control).
No new model, no new hyper-parameter, alpha / k_white / K_LOCAL / folds unchanged, zero new
measurement, and no averaging across corpora.

Criteria C1-C7 are evaluated mechanically in `verdicts()` and written into the JSON, kept separate
from any human reading of the tables.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = p5_paths.CODE
sys.path.insert(0, HERE)

import exp40_public_pipeline as P                                    # noqa: E402
import ext_data as ED                                                # noqa: E402
import _p51_xcorpus as Q                                             # noqa: E402
import p5_paths  # noqa: E402

OUT = os.environ.get("P50_OUT") or p5_paths.RESULTS + "/exp50_bandcov.json"
EXP51_JSON = p5_paths.RESULTS + "/exp51_xcorpus.json"

ALPHA, NF = Q.ALPHA, Q.NF
PREPS = ("abs", "snv", "msc")
METRICS = ("euclid", "white_a")
ARMS = ("marg", "band", "band_oracle")
QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
NB = 3                               # bands, fixed before the run (pre-registration I.A)
MIN_CAL_PER_BAND = 20                # exp51 S2's pool floor, reused as the fallback trigger
BAND_COV_FLOOR = 0.85                # C2
GLOBAL_BAND_WINDOW = (0.88, 0.95)    # C2
C3_SIZE_RATIO, C3_ACCEPT_DROP = 1.5, 0.10
C4_RANGE, C4_MIN_BAND_N = 0.10, 20
C5_RECOVERED, C5_NOT = 0.80, 0.60
COLLAPSE_CELLS = {"D1": 4, "D2": 4, "D3": 3, "HONEY": 3}   # exp51 fold vectors (0-based index)
MAIN = "white_a"

# Corpora (pre-registration II): collapse set, clean controls, the multi-candidate pooled cinnamon
# corpus. D7p and COCO_instant are in the run because II names them as MAF-admissible rows.
RUN = ("D1", "D2", "D3", "HONEY", "D8", "D7w", "D7d", "D7p", "COCO_instant")
STAT_OUT = ("n", "top1", "coverage", "size", "empty", "accept", "top1_acc", "n_accepted")
COMPARE = STAT_OUT


# --------------------------------------------------------------------------- corrections
def msc(Z, fit):
    """Per-spectrum affine regression on the FIT-POOL mean spectrum, residual kept (slope+intercept
    estimated from that spectrum alone against the one reference vector). Fit pool only: a test row
    never touches any statistic except through its own spectrum."""
    m = Z[fit].mean(0)
    A = np.column_stack([m, np.ones(len(m))])
    coef = (Z @ A) @ np.linalg.inv(A.T @ A)                 # (n, 2): slope, intercept
    resid = Z - coef @ A.T
    diag = dict(slope_mean=round(float(coef[:, 0].mean()), 4),
                slope_sd=round(float(coef[:, 0].std()), 4),
                intercept_sd=round(float(coef[:, 1].std()), 5),
                resid_norm_ratio=round(float(np.linalg.norm(resid, axis=1).mean()
                                             / np.linalg.norm(Z, axis=1).mean()), 4))
    return resid, diag


def prep_matrix(Z, fit, prep):
    if prep == "abs":
        return Z, {}
    if prep == "snv":
        return ED.snv(Z), {}
    return msc(Z, fit)


# --------------------------------------------------------------------------- one fold, three quantile arms
def fold_v2(X, comp, level, block, cls_a, fit, cal, test, cand, atoms, alpha, metric,
            gate_cal, labels):
    """exp40's fold computation plus the per-band quantiles.

    The marginal path is a line-for-line copy of `P.fold_rows` (same operations in the same order,
    same min{1,(1-alpha)(1+1/n)} quantile) so that C1 can assert agreement with exp51 to 1e-9; a
    row-level self-check against the original `fold_rows` is run alongside it. The additions are the
    band statistic e(x) = d(x, authentic atom) in the fold's own metric, the fit-pool tertile cuts
    and the per-band / per-oracle-band Mondrian quantiles.
    """
    m0 = X[fit][comp[fit] == 0].mean(0)
    mus = {0: m0}
    for c in atoms:
        mus[c] = X[fit][comp[fit] == c].mean(0)
    D = np.stack([mus[c] - m0 for c in atoms], 1)
    wh = P.make_metric(metric, X, cls_a, cls_a, fit)
    cand_m = np.stack([P.pred(m0, D, S) for S in cand])
    S_of = {S: j for j, S in enumerate(cand)}

    Z = X if wh is None else wh.coords(X)
    Zc = cand_m if wh is None else wh.coords(cand_m)
    k = max(1, min(P.K_LOCAL, len(fit)))
    loc = np.sqrt(np.partition(P._sq_dists(Z, Z[fit]), k - 1, axis=1)[:, :k]).mean(1)
    d2 = P._sq_dists(Z, Zc)

    def dists(i):
        return np.sqrt(d2[i])

    def truth(i):
        c = int(comp[i])
        if c == 0:
            return ()
        return (atoms.index(c),) if c in atoms else None

    loc_cal, loc_test = loc[cal], loc[test]
    cal_raw = np.array([dists(i)[S_of[truth(i)]] for i in cal])
    q_std = float(np.quantile(cal_raw, min(1.0, (1 - alpha) * (1 + 1 / len(cal_raw)))))
    cal_sc = cal_raw / loc_cal
    q = float(np.quantile(cal_sc, min(1.0, (1 - alpha) * (1 + 1 / len(cal_sc)))))

    gc = np.asarray(gate_cal)
    pos = {int(i): j for j, i in enumerate(cal)}
    best_cal = np.array([float(np.min(dists(i))) / loc_cal[pos[int(i)]] for i in gc])
    tau = float(np.quantile(best_cal, min(1.0, (1 - alpha) * (1 + 1 / len(best_cal)))))
    abs_cal = np.array([float(np.min(dists(i))) for i in gc])
    tau_abs = float(np.quantile(abs_cal, min(1.0, (1 - alpha) * (1 + 1 / len(abs_cal)))))

    # band statistic: distance to the authentic atom on the same (standard) score the guarantee is
    # carried on. Cuts come from the fit pool only; the declared level is never used here.
    e_fit = np.array([float(dists(i)[0]) for i in fit])
    e_cal = np.array([float(dists(i)[0]) for i in cal])
    e_test = np.array([float(dists(i)[0]) for i in test])
    cuts = [float(v) for v in np.quantile(e_fit, list(QUANTILES))]
    b_cal = np.searchsorted(cuts, e_cal, side="right")
    b_test = np.searchsorted(cuts, e_test, side="right")
    # oracle grouping: tertiles of the declared level on the fit pool (pricing only)
    ocuts = [float(v) for v in np.quantile(np.asarray(level)[fit], list(QUANTILES))]
    ob_cal = np.searchsorted(ocuts, np.asarray(level)[cal], side="right")
    ob_test = np.searchsorted(ocuts, np.asarray(level)[test], side="right")

    def mond(group_cal, ngroups=NB):
        qs, fb = [], []
        for b in range(ngroups):
            g = group_cal == b
            n = int(g.sum())
            if n >= MIN_CAL_PER_BAND:
                qs.append(float(np.quantile(cal_raw[g], min(1.0, (1 - alpha) * (1 + 1.0 / n)))))
                fb.append(False)
            else:
                qs.append(q_std)
                fb.append(True)
        return qs, fb

    q_b, fb_b = mond(b_cal)
    q_o, fb_o = mond(ob_cal)

    rows = {a: [] for a in ARMS}
    for j, i in enumerate(test):
        raw = dists(i)
        sc = raw / loc_test[j]
        tset = truth(i)
        kt = S_of[tset] if tset is not None else -1
        order, order_s = np.argsort(sc), np.argsort(raw)
        rank_std = int(np.where(order_s == kt)[0][0]) if kt >= 0 else len(cand)
        best = float(sc[order[0]])
        base = dict(comp=int(comp[i]), level=float(level[i]),
                    cell=(int(comp[i]), float(level[i]), int(block[i])), rank=rank_std,
                    best=best, best_q=float(raw[order[0]]),
                    best_set=P.set_label(cand[order[0]], labels),
                    accepted=bool(best <= tau),
                    accepted_two=bool(best <= tau and float(np.min(raw)) <= tau_abs),
                    e=float(e_test[j]), band=int(b_test[j]), oracle_band=int(ob_test[j]))
        thr = dict(marg=q_std, band=q_b[int(b_test[j])], band_oracle=q_o[int(ob_test[j])])
        fall = dict(marg=False, band=fb_b[int(b_test[j])], band_oracle=fb_o[int(ob_test[j])])
        for a in ARMS:
            Cs = np.where(raw <= thr[a])[0]
            r = dict(base)
            r.update(covered=bool(kt >= 0 and kt in Cs), size=int(len(Cs)),
                     empty=bool(len(Cs) == 0), thr=float(thr[a]), fallback=bool(fall[a]), arm=a)
            rows[a].append(r)
    meta = dict(q_std=q_std, q_norm=q, tau=tau, tau_abs=tau_abs,
                n_fit=int(len(fit)), n_cal=int(len(cal)), n_test=int(len(test)),
                band_cuts=cuts, band_q=q_b, band_cal_n=[int((b_cal == b).sum()) for b in range(NB)],
                band_test_n=[int((b_test == b).sum()) for b in range(NB)], band_fallback=fb_b,
                oracle_cuts=ocuts, oracle_q=q_o,
                oracle_cal_n=[int((ob_cal == b).sum()) for b in range(NB)],
                oracle_test_n=[int((ob_test == b).sum()) for b in range(NB)],
                oracle_fallback=fb_o,
                e_fit=[float(v) for v in (e_fit.min(), np.median(e_fit), e_fit.max())],
                e_cal=[float(v) for v in (e_cal.min(), np.median(e_cal), e_cal.max())],
                e_test=[float(v) for v in (e_test.min(), np.median(e_test), e_test.max())],
                whitener_dim=(None if wh is None else int(wh.dim)),
                whitener_delta=(None if wh is None else float(wh.delta)))
    return rows, meta


def scaffold_check(Y, comp, level, block, cls_a, fit, cal, test, cand, atoms, metric,
                   gate_cal, labels, rows, meta):
    """Row-level identity of the marginal path against exp40's own fold_rows."""
    rr, q, tau, tau_abs, q_std = P.fold_rows(Y, comp, level, block, cls_a, cls_a, fit, cal, test,
                                            cand, atoms, ALPHA, metric, gate_cal, labels=labels)
    bad = 0
    for a, b in zip(rr, rows["marg"]):
        if (a["covered_std"] != b["covered"] or a["size_std"] != b["size"]
                or a["rank_std"] != b["rank"] or a["comp"] != b["comp"]
                or a["best"] != b["best"] or a["accepted"] != b["accepted"]
                or a["accepted_two"] != b["accepted_two"]):
            bad += 1
    return dict(rows=len(rr), mismatch=bad,
                q_std_delta=abs(float(q_std) - float(meta["q_std"])),
                q_norm_delta=abs(float(q) - float(meta["q_norm"])),
                tau_delta=abs(float(tau) - float(meta["tau"])),
                tau_abs_delta=abs(float(tau_abs) - float(meta["tau_abs"])))


# --------------------------------------------------------------------------- aggregation
def pack(rows, lvls, metas, cand_n):
    """exp51's readout structure, so the marg arm is directly comparable to its JSON."""
    fold_cov, fold_size, fold_n = [], [], []
    for f in range(len(metas)):
        rr = [r for r in rows if r["fold"] == f]
        fold_cov.append(round(float(np.mean([r["covered"] for r in rr])), 3) if rr else None)
        fold_size.append(round(float(np.mean([r["size"] for r in rr])), 2) if rr else None)
        fold_n.append(int(len(rr)))
    bands = {}
    for b in range(NB):
        rb = [r for r in rows if r["band"] == b]
        bands[b] = dict(n=len(rb), cal_n_per_fold=[m["band_cal_n"][b] for m in metas],
                        test_n_per_fold=[m["band_test_n"][b] for m in metas],
                        fallback_any=bool(any(m["band_fallback"][b] for m in metas)),
                        all=Q.stat(rb), adulterated=Q.stat([r for r in rb if r["comp"] > 0]))
    obands = {}
    for b in range(NB):
        rb = [r for r in rows if r["oracle_band"] == b]
        obands[b] = dict(n=len(rb), cal_n_per_fold=[m["oracle_cal_n"][b] for m in metas],
                         fallback_any=bool(any(m["oracle_fallback"][b] for m in metas)),
                         all=Q.stat(rb),
                         levels=sorted(set(float(r["level"]) for r in rb)))
    return dict(all=Q.stat(rows),
                pure=Q.stat([r for r in rows if r["comp"] == 0]),
                adulterated=Q.stat([r for r in rows if r["comp"] > 0]),
                by_level={l: Q.stat([r for r in rows if r["level"] == l and r["comp"] > 0])
                          for l in lvls},
                fold_coverage=fold_cov, fold_size=fold_size, fold_test_n=fold_n,
                by_band=bands, by_oracle_band=obands,
                fallback_test_share=round(float(np.mean([r["fallback"] for r in rows])), 3),
                band_q_per_fold=[[round(m["band_q"][b], 4) for b in range(NB)] for m in metas],
                q_std_per_fold=[round(m["q_std"], 4) for m in metas],
                tau_per_fold=[round(m["tau"], 4) for m in metas],
                n_cal_per_fold=[m["n_cal"] for m in metas],
                whitener_dim_per_fold=[m["whitener_dim"] for m in metas],
                whitener_delta_per_fold=[f"{m['whitener_delta']:.3e}"
                                         if m["whitener_delta"] is not None else None
                                         for m in metas],
                n_candidates=cand_n)


def run_corpus(name, d, splits, atoms, labels, adm, geom_src="X"):
    cand = P.subsets(len(atoms))
    lvls = sorted(set(float(v) for v in np.unique(d["level"]) if v > 0))
    comp, level, block = d["comp"], d["level"], d["block"]
    cls_a = comp.copy()
    X = np.asarray(d[geom_src], float)
    arms, geos, scaffold, msc_diag, band_diag = {}, {}, {}, [], {}
    for prep in PREPS:
        if prep in ("abs", "snv"):
            Yg = X if prep == "abs" else ED.snv(X)
            g = Q.geometry_one(Yg, comp, cls_a, atoms, labels)
            g["per_fold"] = Q.whitener_per_fold(Yg, cls_a, splits)
            g["rank_deficient"] = g["per_fold"]["rank_deficient"]
            geos[prep] = g
        for metric in METRICS:
            rows = {a: [] for a in ARMS}
            metas = []
            for fi, (fit, cal, test) in enumerate(splits):
                Y, diag = prep_matrix(X, fit, prep)
                if prep == "msc" and metric == MAIN:
                    msc_diag.append(diag)
                gate_cal = cal[comp[cal] > 0]
                rr, meta = fold_v2(Y, comp, level, block, cls_a, fit, cal, test, cand, atoms,
                                   ALPHA, metric, gate_cal, labels)
                if metric == MAIN:
                    sc = scaffold_check(Y, comp, level, block, cls_a, fit, cal, test, cand, atoms,
                                        metric, gate_cal, labels, rr, meta)
                    sc["fold"] = fi
                    scaffold.setdefault(prep, []).append(sc)
                for a in ARMS:
                    for r in rr[a]:
                        r["fold"] = fi
                    rows[a] += rr[a]
                metas.append(meta)
            for a in ARMS:
                p = pack(rows[a], lvls, metas, len(cand))
                if metric == MAIN:
                    p["intervals"] = P.cluster_boot(rows[a], ["coverage", "size", "top1",
                                                              "abstain"])
                p["maf"] = {k: Q.maf(lvls, p["by_level"], fn) for k, fn in Q.CRITS.items()}
                p["maf_admissible"] = bool(prep in adm)
                p["levels"] = lvls
                arms[f"{prep}/{metric}/{a}"] = p
            band_diag[f"{prep}/{metric}"] = dict(
                e_median_fit=round(float(np.mean([m["e_fit"][1] for m in metas])), 4),
                e_median_cal=round(float(np.mean([m["e_cal"][1] for m in metas])), 4),
                e_median_test=round(float(np.mean([m["e_test"][1] for m in metas])), 4),
                leverage_ratio=round(float(np.mean([m["e_cal"][1] for m in metas])
                                           / max(1e-12, np.mean([m["e_fit"][1] for m in metas]))), 3),
                test_n_per_band=[int(sum(m["band_test_n"][b] for m in metas)) for b in range(NB)],
                cal_n_per_band=[int(sum(m["band_cal_n"][b] for m in metas)) for b in range(NB)],
                band_cuts_per_fold=[[round(m["band_cuts"][b], 4) for b in range(NB - 1)]
                                    for m in metas])
    msd = ({k: round(float(np.mean([m[k] for m in msc_diag])), 5)
            for k in ("slope_mean", "slope_sd", "intercept_sd", "resid_norm_ratio")}
           if msc_diag else None)
    return arms, geos, lvls, len(cand), dict(
        scaffold_rows=sum(s["rows"] for v in scaffold.values() for s in v),
        scaffold_mismatch=sum(s["mismatch"] for v in scaffold.values() for s in v),
        per_fold=scaffold), msd, band_diag


def pooled_cinnamon(adm):
    X, comp, level, blk, series, n_dup, nch = P.load_cinnamon()
    d = dict(X=X, comp=comp, level=level, block=blk)
    return d, P.blocked_folds(blk), [1, 2, 3], P.NAME, n_dup, nch


# --------------------------------------------------------------------------- main
def main():
    ref = json.load(open(EXP51_JSON, encoding="utf-8"))
    audit = json.load(open(Q.EXP45_JSON, encoding="utf-8"))["corpora"]
    out = dict(prereg=r"preregistrations/exp50_conditional_coverage_and_baseline_correction_v1.md",
               alpha=ALPHA, k_white=P.K_WHITE, k_local=P.K_LOCAL, nf=NF, bands=NB,
               band_quantiles=list(QUANTILES), min_cal_per_band=MIN_CAL_PER_BAND,
               band_statistic="e(x) = d(x, authentic atom), standard (un-normalised) score, "
                              "fold's own metric; tertile cuts on the fit pool only",
               oracle_definition="tertile bands of the declared level on the fit pool; prices "
                                 "conditioning only, never quoted as a guarantee",
               msc_definition="per-spectrum affine regression (slope+intercept) on the fit-pool mean "
                              "spectrum, residual kept; dictionary/whitener/local scale/quantiles "
                              "refit on the corrected spectra",
               window=list(Q.WINDOW), metrics=list(METRICS), preps=list(PREPS), arms=list(ARMS),
               corpora={}, verdicts={}, dose_axis_audit={})

    for spec in Q.corpora():
        nm = spec["name"]
        if nm not in RUN:
            continue
        adm = audit.get(nm, {}).get("usable_in_metrics") or list(
            Q.DOSE_ADMISSIBLE_FALLBACK.get(nm, ()))
        splits = Q.blocked_folds(spec["data"]["stratum"], spec["data"]["block"])
        t0 = time.time()
        arms, geos, lvls, ncand, scaf, msd, bdiag = run_corpus(nm, spec["data"], splits,
                                                              spec["data"]["atoms"],
                                                              spec["data"]["labels"], adm)
        r = dict(matrix=spec["matrix"], adulterant=spec["adulterant"], source=spec["source"],
                 naming_claim=spec["naming"], n_spectra=int(len(spec["data"]["comp"])),
                 n_channels=int(spec["data"]["X"].shape[1]),
                 window=[round(v, 1) for v in spec["data"]["window"]],
                 levels=lvls, n_candidates=ncand, dose_axis=audit.get(nm, {}).get("dose_axis"),
                 dose_admissible_in=list(adm), aliasing_flag=audit.get(nm, {}).get("aliasing_flag"),
                 geometry=geos, msc_fitted_over_folds=msd, band_diagnostic=bdiag,
                 scaffold_check=scaf, arms=arms,
                 minutes=round((time.time() - t0) / 60, 2))
        out["corpora"][nm] = r
        m = r["arms"][f"abs/{MAIN}/marg"]["all"]
        b = r["arms"][f"abs/{MAIN}/band"]["all"]
        c = r["arms"][f"msc/{MAIN}/marg"]["all"]
        print(f"[{nm}] {r['n_spectra']}x{r['n_channels']} lvls={lvls[:4]}.. "
              f"marg abs cov {m['coverage']} -> band cov {b['coverage']} size {m['size']}"
              f"/{b['size']} | msc cov {c['coverage']} fold_cov "
              f"{r['arms'][f'abs/{MAIN}/marg']['fold_coverage']} "
              f"mismatch {scaf['scaffold_mismatch']} ({r['minutes']} min)", flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)

    d, splits, atoms, labels, n_dup, nch = pooled_cinnamon(
        Q.DOSE_ADMISSIBLE_FALLBACK["D7_pooled"])
    adm = Q.DOSE_ADMISSIBLE_FALLBACK["D7_pooled"]
    t0 = time.time()
    arms, geos, lvls, ncand, scaf, msd, bdiag = run_corpus("D7_pooled", d, splits, atoms, labels,
                                                           adm)
    out["corpora"]["D7_pooled"] = dict(
        matrix="cinnamon", adulterant=list(P.NAME),
        source="Mendeley rtbsf2r53x.1, D7w+D7p+D7d pooled", naming_claim="yes",
        n_spectra=int(len(d["comp"])), n_channels=int(nch), window=list(Q.WINDOW),
        levels=lvls, n_candidates=ncand, duplicate_rows=n_dup,
        folds="exp40's own blocked_folds (five name-adjacent records per block)",
        dose_axis="usable (exp49/exp40 dose series)", dose_admissible_in=list(adm),
        aliasing_flag=False, geometry=geos, msc_fitted_over_folds=msd, band_diagnostic=bdiag,
        scaffold_check=scaf,
        arms=arms, minutes=round((time.time() - t0) / 60, 2))
    print(f"[D7_pooled] marg abs cov {arms[f'abs/{MAIN}/marg']['all']['coverage']} "
          f"band {arms[f'abs/{MAIN}/band']['all']['coverage']} "
          f"msc cov {arms[f'msc/{MAIN}/marg']['all']['coverage']} top1 msc "
          f"{arms[f'msc/{MAIN}/marg']['all']['top1']} vs abs "
          f"{arms[f'abs/{MAIN}/marg']['all']['top1']} mismatch "
          f"{scaf['scaffold_mismatch']} ({out['corpora']['D7_pooled']['minutes']} min)", flush=True)

    out["dose_axis_audit"] = {nm: dict(dose_axis=v.get("dose_axis"),
                                       usable_in=v.get("usable_in_metrics"),
                                       aliasing=bool(v.get("aliasing_flag")))
                              for nm, v in audit.items()}
    verdicts(out, ref)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(out["verdicts"], indent=1, ensure_ascii=False)[:12000])
    print("saved", OUT)


# --------------------------------------------------------------------------- C1-C7
def diff_stat(mine, theirs):
    if mine is None or theirs is None:
        return None if (mine is None and theirs is None) else ["presence mismatch"]
    bad = []
    for k in COMPARE:
        if k not in mine or k not in theirs:
            continue
        a, b = mine[k], theirs[k]
        if a is None or b is None:
            if a is not b:
                bad.append(f"{k}:presence")
            continue
        if abs(float(a) - float(b)) > 1e-9:
            bad.append(k)
    return bad or None


def verdicts(out, ref):
    C = out["corpora"]

    # ---- C1 reproduction gate: marg/abs+snv must equal exp51 to 1e-9
    c1 = {}
    for nm, r in C.items():
        if nm not in ref["corpora"]:
            continue
        for prep in ("abs", "snv"):
            for metric in METRICS:
                key = f"{prep}/{metric}"
                mine, theirs = r["arms"][key + "/marg"], ref["corpora"][nm]["arms"][key]
                bad = {}
                for part in ("all", "pure", "adulterated"):
                    d = diff_stat(mine[part], theirs[part])
                    if d:
                        bad[part] = d
                for l, s in mine["by_level"].items():
                    ref_bl = {float(k): v for k, v in (theirs["by_level"] or {}).items()}
                    d = diff_stat(s, ref_bl.get(l))
                    if d:
                        bad[f"by_level[{l}]"] = d
                if bad:
                    c1[f"{nm}/{key}"] = bad
    mism = sum(sc["mismatch"] for r in C.values() for lst in r["scaffold_check"]["per_fold"].values()
               for sc in lst)
    out["verdicts"]["C1"] = dict(
        fired=bool(c1 or mism), offenders=c1, scaffold_row_mismatch=int(mism),
        n_fields_compared=int(sum(len(r["arms"]) for r in C.values())),
        rule="if the marg arm's globals or per-level statistics differ from "
             "logs/exp51_xcorpus.json by >= 1e-9, fix the scaffolding first: no reading from this "
             "run may be used and nothing goes into the manuscript until it agrees")

    admissible = [(nm, prep, MAIN) for nm, r in C.items()
                  for prep in PREPS if r["arms"][f"{prep}/{MAIN}/marg"]["maf_admissible"]
                  and len(r["levels"]) >= 1]

    # ---- C2 band coverage on the three lowest admissible levels + global band coverage
    c2 = {}
    for nm, prep, metric in admissible:
        band, lv = C[nm]["arms"][f"{prep}/{metric}/band"], C[nm]["levels"]
        low = lv[:3]
        per = {}
        for l in low:
            s = band["by_level"].get(l)
            per[l] = None if s is None else s["coverage"]
        cov_ok = all(v is not None and v >= BAND_COV_FLOOR for v in per.values())
        gv = band["all"]["coverage"]
        glob_ok = bool(GLOBAL_BAND_WINDOW[0] <= gv <= GLOBAL_BAND_WINDOW[1])
        c2[f"{nm}/{prep}"] = dict(low_levels=low, band_coverage_at_low=per,
                                  all_low_levels_ge_085=bool(cov_ok),
                                  global_coverage=gv, global_in_088_095=glob_ok,
                                  pass_=bool(cov_ok and glob_ok))
    achieved = bool(c2) and all(v["pass_"] for v in c2.values())
    out["verdicts"]["C2"] = dict(
        achieved=achieved, fired=bool(not achieved), rows=c2, n_rows=len(c2),
        rule="`band` may only be called a coverage fix if every MAF-admissible corpus/preparation "
             "row has coverage >= 0.85 at its three lowest levels AND global coverage in "
             "[0.88, 0.95]; otherwise the word 'fix' is withdrawn and only the reporting duty "
             "(per-band coverage vectors beside the marginal number) survives")

    # ---- C3 cost of the band arm: set size inflation / acceptance drop
    c3 = {}
    for nm in C:
        for prep in PREPS:
            m = C[nm]["arms"][f"{prep}/{MAIN}/marg"]["all"]
            b = C[nm]["arms"][f"{prep}/{MAIN}/band"]["all"]
            ratio = round(float(b["size"] / m["size"]), 3) if m["size"] else None
            drop = round(float(m["accept"] - b["accept"]), 3)
            c3[f"{nm}/{prep}"] = dict(size_marg=m["size"], size_band=b["size"], size_ratio=ratio,
                                      accept_marg=m["accept"], accept_band=b["accept"],
                                      accept_drop=drop,
                                      empty_marg=m["empty"], empty_band=b["empty"],
                                      size_inflation_gt_1_5=bool(ratio and ratio > C3_SIZE_RATIO),
                                      accept_drop_gt_010=bool(drop > C3_ACCEPT_DROP))
    c3_fire = any(v["size_inflation_gt_1_5"] or v["accept_drop_gt_010"] for v in c3.values())
    out["verdicts"]["C3"] = dict(
        fired=bool(c3_fire), applicable=bool(c2 and achieved), rows=c3,
        note="the abstention gate is NOT band-recalibrated (pre-registration I.A changes only the "
             "set quantile), so `accept` is arm-invariant by construction and the acceptance-drop "
             "limb of C3 cannot fire; the operating cost channel of the band arm is the set size "
             "and the empty-set rate, both reported in the same rows",
        rule="if C2 is achieved but the mean set size inflates > 1.5x, or acceptance drops > 0.10, "
             "the coverage was bought with set size / abstention: the method claim retreats to "
             "diagnosis and the word 'guarantee' may not appear in a positive sentence")

    # ---- C4 residual between-band spread after conditioning
    c4 = {}
    for nm in C:
        for prep in PREPS:
            for metric in METRICS:
                band = C[nm]["arms"][f"{prep}/{metric}/band"]
                covs, ns = {}, {}
                for b in range(NB):
                    e = band["by_band"][b]
                    if e["n"] >= C4_MIN_BAND_N and e["all"]:
                        covs[b] = e["all"]["coverage"]
                        ns[b] = e["n"]
                rng = (round(float(max(covs.values()) - min(covs.values())), 3)
                       if len(covs) >= 2 else None)
                if rng is not None:
                    c4[f"{nm}/{prep}/{metric}"] = dict(band_n=ns, band_coverage=covs,
                                                       range=rng, range_gt_010=bool(rng > C4_RANGE))
    c4_fire = any(v["range_gt_010"] for k, v in c4.items() if k.endswith("/" + MAIN))
    out["verdicts"]["C4"] = dict(
        fired=bool(c4_fire), rows=c4,
        rule="if, after conditioning, the coverage spread across bands with n >= 20 is still "
             "> 0.10, the claim downgrades to 'conditioning aligns the band structure but does not "
             "guarantee per-band coverage' and the per-band vectors must be given")

    # ---- C5 can an affine baseline correction pull the collapsed folds back?
    c5 = {}
    for nm, fi in COLLAPSE_CELLS.items():
        if nm not in C:
            continue
        row = {}
        for prep in PREPS:
            arm = C[nm]["arms"][f"{prep}/{MAIN}/marg"]
            row[prep] = dict(fold_coverage=arm["fold_coverage"], n_test=arm["fold_test_n"],
                             collapsed_fold_index=fi, collapsed_fold_coverage=arm["fold_coverage"][fi],
                             n_cal_per_fold=arm["n_cal_per_fold"])
        v = row["msc"]["collapsed_fold_coverage"]
        c5[nm] = dict(**row, msc_fold_coverage=v,
                      recovered_to_080=bool(v is not None and v >= C5_RECOVERED),
                      still_below_060=bool(v is not None and v < C5_NOT))
    all_rec = bool(c5) and all(v["recovered_to_080"] for v in c5.values())
    all_bad = bool(c5) and all(v["still_below_060"] for v in c5.values())
    out["verdicts"]["C5"] = dict(
        rows=c5, all_recovered=bool(all_rec), all_still_below_060=bool(all_bad),
        branch=("msc_writes_itself_into_the_protocol" if all_rec else
                ("collapse_exceeds_any_affine_correction" if all_bad else
                 "partial_between_060_and_080_NOT_pre_registered_read_conservatively")),
        rule="if MSC lifts the collapsed folds (D1 f4 / D2 f4 / D3 f3 / HONEY f3) to >= 0.80, "
             "baseline correction becomes a protocol step and SNV is demoted to 'one of a family'; "
             "if they stay < 0.60 the collapse is declared beyond any affine correction and every "
             "marginal coverage number must be published with its per-fold vector. Values between "
             "0.60 and 0.80 are not a pre-registered branch and are read conservatively (as "
             "not-recovered), never as a success")

    # ---- C6 MSC must only move coverage, not attribution direction
    c6 = {}
    pooled = C.get("D7_pooled")
    if pooled:
        a = pooled["arms"][f"abs/{MAIN}/marg"]["all"]["top1"]
        m = pooled["arms"][f"msc/{MAIN}/marg"]["all"]["top1"]
        c6["D7_pooled_top1"] = dict(abs=a, msc=m, drop=round(float(a - m), 3),
                                    drop_gt_005=bool(a - m > 0.05))
    for nm, prep, metric in admissible:
        if prep != "abs":
            continue
        e = C[nm]["arms"][f"abs/{MAIN}/marg"]["maf"]["naming"][0]
        m = C[nm]["arms"][f"msc/{MAIN}/marg"]["maf"]["naming"][0]
        worse = bool(e is not None and m is not None and m > e) or bool(e is not None and m is None)
        c6[f"{nm}_maf"] = dict(abs=e, snv=C[nm]["arms"][f"snv/{MAIN}/marg"]["maf"]["naming"][0],
                               msc=m, pushed_higher_by_msc=bool(worse),
                               informational_abs_had_no_level_msc_has=bool(e is None
                                                                           and m is not None))
    c6_fire = any(v.get("drop_gt_005") or v.get("pushed_higher_by_msc") for v in c6.values())
    out["verdicts"]["C6"] = dict(
        fired=bool(c6_fire), rows=c6,
        rule="MSC is supposed to change coverage, not attribution direction. If it drops the "
             "pooled-cinnamon whitened top-1 by > 0.05, or pushes the naming MAF to a higher level "
             "on any MAF-admissible corpus, MSC does NOT enter the main arm - it stays a contrast "
             "column, and the paper must state that the band baseline is not a freely removable "
             "nuisance but is aliased with the dose information (same direction as exp45's "
             "aliasing audit)")

    # ---- C7 reading-discipline audit (self-check on the structures the claims are drawn from)
    audit_rows = []
    for k in c2:
        nm, prep = k.split("/")
        b = C[nm]["arms"][f"{prep}/{MAIN}/band"]["all"]
        m = C[nm]["arms"][f"{prep}/{MAIN}/marg"]["all"]
        have = all(x in b and b[x] is not None for x in ("coverage", "size", "empty", "accept")) \
            and all(x in m and m[x] is not None for x in ("coverage", "size", "empty", "accept"))
        audit_rows.append((k, "cost columns present beside coverage", bool(have)))
    pairs = []
    for nm in C:
        for prep in PREPS:
            e = C[nm]["arms"][f"{prep}/euclid/marg"]
            w = C[nm]["arms"][f"{prep}/{MAIN}/marg"]
            pairs.append((f"{nm}/{prep}", "MAF paired euclid/white",
                          bool("maf" in e and "maf" in w)))
    oracle_used = [k for k in ("C2", "C3", "C4", "C5", "C6")
                   if "band_oracle" in json.dumps(out["verdicts"].get(k, {}))]
    out["verdicts"]["C7"] = dict(
        fired=bool(oracle_used) or not all(r[2] for r in audit_rows + pairs),
        cost_column_audit=[list(r) for r in audit_rows], maf_pair_audit=[list(r) for r in pairs],
        band_oracle_appears_in_claim_criteria=oracle_used,
        rule="every 'fix/improvement' statement must (a) appear in the same sentence as its cost, "
             "(b) be given per corpus and per arm, (c) publish MAF as a euclid/whitening pair, "
             "(d) never quote band_oracle in a claim. Rows violating any limb stay out of the "
             "tables; anything read off after the run can only enter the limitations section")

    out["verdicts"]["scaffold_row_level_identity"] = {
        nm: dict(rows=r["scaffold_check"]["scaffold_rows"],
                 mismatch=r["scaffold_check"]["scaffold_mismatch"]) for nm, r in C.items()}
    out["verdicts"]["rank_deficient_whiteners"] = {
        f"{nm}/{p}": dict(dim_per_fold=C[nm]["arms"][f"{p}/{MAIN}/marg"]
                          ["whitener_dim_per_fold"],
                          delta_per_fold=C[nm]["arms"][f"{p}/{MAIN}/marg"]
                          ["whitener_delta_per_fold"],
                          n_cal_per_fold=C[nm]["arms"][f"{p}/{MAIN}/marg"]["n_cal_per_fold"])
        for nm in C for p in PREPS
        if min(C[nm]["arms"][f"{p}/{MAIN}/marg"]["whitener_dim_per_fold"] or [P.K_WHITE])
        < P.K_WHITE}
    out["verdicts"]["fallback_bands"] = {
        f"{nm}/{p}": {b: C[nm]["arms"][f"{p}/{MAIN}/band"]["by_band"][b]["cal_n_per_fold"]
                      for b in range(NB)
                      if any(n < MIN_CAL_PER_BAND for n in
                             C[nm]["arms"][f"{p}/{MAIN}/band"]["by_band"][b]["cal_n_per_fold"])}
        for nm in C for p in PREPS}


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("total minutes", round((time.time() - t0) / 60, 2))
