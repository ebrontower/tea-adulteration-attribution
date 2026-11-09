"""exp45 - pre-flight audit of every dose corpus on disk, and an out-of-sample test of the one
quantity that is supposed to predict extrapolation failure.

Why this exists. exp43/exp44 audited the 7 recorded-dose public corpora and found that 4 of them
(D3, D7w, D7d, D8) have a step-function distance-to-authentic, i.e. the "dose axis" in the
deposited sheet is partly a per-level block, so no graded claim can rest on them. The candidate
predictor of extrapolation failure was slope_cv over *all* declared levels, and the outcome came
from the same corpora - which is descriptive, not validating (the 10-31 audit said exactly this
about the earlier diagnostics). exp45 fixes both halves:

  (1) more corpora. The honey/sucrose panel (16 paired preparations, levels 0/10/20/40 + a pure
      sucrose endmember, %T -> A) and the coconut-milk/water panel (2 milk types x levels
      0/10/20, NIR-region transmission, kept as two separate corpora) are added, so the audit
      covers 10 dose corpora. Neither was used to form the exp44 rule.

  (2) leakage-free predictor. An analyst holding a calibration range cannot compute a statistic
      over levels they do not have. Every predictor is computed inside the FIT band only:
        slope_cv_fit   CV of ||mu_L - mu_auth|| / L over the fit-band levels
        r2_pred_loo    leave-one-level-out predictive R^2 of the rank-1 model o_L = L*v
        r2_in_fit      in-sample version of the same fit (for contrast)
        sv1_fit        share of the fit-band offset matrix carried by one direction
        monotone_fit / snr_fit
      The outcome is PLS extrapolation error on rows whose *preparations* are disjoint from the
      fit preparations, in two metrics (absolute absorbance, SNV) and both directions (train HIGH
      -> test LOW, train LOW -> test HIGH), each with two preparation folds. A paired panel like
      honey has every preparation at every level, so disjointness must be imposed on preparation
      identity, not on row order - that is what the alternating preparation split does.

Rules fixed before looking at any result:
  a level enters the analysis only with >= 2 independent preparations; authentic (0) must be one
  of them. fit band >= 6 rows / >= 2 preparations, test band >= 5 rows / >= 2 preparations,
  preparations disjoint between them. nc = min(6, max(2, n_fit // 3)). r2_pred_loo is undefined
  (null) below 3 fit-band levels, because a 2-point rank-1 fit is exact by construction.
  "usable dose axis" = distance-to-authentic monotone with snr_top >= 1 in at least one metric.
  A corpus whose declared levels also form contiguous, level-ordered row blocks is flagged.
"""
import json
import os
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression

HERE = p5_paths.CODE
sys.path.insert(0, HERE)
import p5_public_data as PP                                    # noqa: E402
import ext_data as ED                                          # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp45_external_validation.json"
MIN_FIT_ROWS, MIN_TEST_ROWS, MIN_FIT_GROUPS, MIN_TEST_GROUPS = 6, 5, 2, 2
PASS_ABS_BIAS = 50.0          # percent; |rel bias| at or below this counts as "extrapolation OK"


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def prep(X, tag):
    return X if tag == "abs" else snv(X)


def in_set(values, s):
    return np.array([v in s for v in values], bool)


def levels_with_preps(y, g, min_groups=2):
    return np.array([L for L in np.unique(y) if len(set(g[y == L])) >= min_groups], float)


def alt_split(groups):
    """deterministic balanced partition of the preparation ids."""
    gs = sorted(set(groups))
    return set(gs[0::2]), set(gs[1::2])


def rank1_predict(O, Lv, rows, all_rows):
    """v estimated on `rows`, offsets reconstructed for every level."""
    v = (Lv[rows, None] * O[rows]).sum(0) / (Lv[rows] ** 2).sum()
    return np.outer(Lv[all_rows], v)


def band_stats(Z, y, levels):
    """closed-form geometry of the dose response, computed inside `levels` only."""
    lev = [L for L in levels if (y == L).any()]
    if 0.0 not in lev or len([L for L in lev if L > 0]) < 2:
        return None
    mus = {L: Z[y == L].mean(0) for L in lev}
    Lv = np.array([L for L in lev if L > 0], float)
    O = np.array([mus[L] - mus[0.0] for L in Lv])
    o = np.argsort(Lv)
    Lv, O = Lv[o], O[o]
    nrm = np.linalg.norm(O, axis=1)
    within = np.array([np.linalg.norm(Z[y == L] - Z[y == L].mean(0), axis=1).mean() for L in Lv])
    slopes = nrm / Lv
    sv = np.linalg.svd(O, compute_uv=False)
    sv1 = float(sv[0] ** 2 / (sv ** 2).sum())
    full = np.arange(len(Lv))
    r2_in = float(1 - ((O - rank1_predict(O, Lv, full, full)) ** 2).sum() / (O ** 2).sum())
    if len(Lv) >= 3:
        se = sum(float(((O[i] - rank1_predict(O, Lv, [j for j in full if j != i], [i])[0]) ** 2).sum())
                 for i in full)
        r2_loo = float(1 - se / (O ** 2).sum())
    else:
        r2_loo = None
    steps = nrm[1:] - nrm[:-1]
    return dict(levels=[float(v) for v in Lv], norms=[round(float(a), 4) for a in nrm],
                slope_cv=round(float(slopes.std() / slopes.mean()), 4),
                r2_in=round(r2_in, 4), r2_pred_loo=(None if r2_loo is None else round(r2_loo, 4)),
                sv1_share=round(sv1, 4), monotone=bool(np.all(steps > 0)),
                worst_rel_step=(round(float(min(steps / np.maximum(nrm[:-1], 1e-12))), 4)
                                if len(steps) else None),
                snr_top=round(float(nrm[-1] / within[-1]), 3))


def one_point(Z, y, g, fitb, testb, fit_preps, test_preps, geo_fit):
    fr = np.where(in_set(y, set(fitb)) & in_set(g, fit_preps))[0]
    tr = np.where(in_set(y, set(testb)) & in_set(g, test_preps) & (y > 0))[0]
    if len(fr) < MIN_FIT_ROWS or len(tr) < MIN_TEST_ROWS:
        return None
    if len(set(g[fr])) < MIN_FIT_GROUPS or len(set(g[tr])) < MIN_TEST_GROUPS:
        return None
    k = int(min(6, max(2, len(fr) // 3), len(fr) - 1, Z.shape[1] - 1))
    p = np.asarray(PLSRegression(n_components=k).fit(Z[fr], y[fr]).predict(Z[tr])).ravel()
    err = p - y[tr]
    pool = set(list(fitb) + list(testb) + [0.0])
    cp, ct = [], []
    for q in sorted(set(g[tr])):
        mtr = np.where(in_set(y, pool) & np.array([v != q for v in g], bool))[0]
        mte = np.where(in_set(y, set(testb)) & in_set(g, {q}))[0]
        if len(mtr) < MIN_FIT_ROWS or len(mte) == 0:
            continue
        kk = int(min(6, max(2, len(mtr) // 3), len(mtr) - 1, Z.shape[1] - 1))
        cp.append(np.asarray(PLSRegression(n_components=kk).fit(Z[mtr], y[mtr]).predict(Z[mte])).ravel())
        ct.append(y[mte])
    ctrl = None
    if cp:
        e2 = np.concatenate(cp) - np.concatenate(ct)
        ctrl = dict(rmse=round(float(np.sqrt(np.mean(e2 ** 2))), 3),
                    rel_bias_pct=round(float(100 * np.mean(e2) / np.mean(np.concatenate(ct))), 2),
                    n_test=int(len(e2)))
    return dict(n_fit=int(len(fr)), n_test=int(len(tr)), groups_fit=int(len(set(g[fr]))),
                groups_test=int(len(set(g[tr]))), nc=k,
                rmse=round(float(np.sqrt(np.mean(err ** 2))), 3),
                rel_bias_pct=round(float(100 * np.mean(err) / np.mean(y[tr])), 2),
                mean_test_level=round(float(np.mean(y[tr])), 3),
                interp_control=ctrl,
                slope_cv_fit=geo_fit["slope_cv"], r2_pred_loo=geo_fit["r2_pred_loo"],
                r2_in_fit=geo_fit["r2_in"], sv1_fit=geo_fit["sv1_share"],
                monotone_fit=geo_fit["monotone"], snr_fit=geo_fit["snr_top"],
                fit_levels=geo_fit["levels"], test_levels=[float(v) for v in testb])


def audit_one(name, X, y, g, matrix, adulterant, instrument, note_extra=None):
    y = np.asarray(y, float)
    g = np.asarray(g, object)
    usable = levels_with_preps(y, g)
    if len(usable) < 3 or usable[0] != 0.0:
        return dict(name=name, matrix=matrix, adulterant=adulterant, instrument=instrument,
                    n=int(len(y)), skipped_by_rule=True,
                    levels=[float(v) for v in np.unique(y)],
                    preps_per_level=[int(len(set(g[y == L]))) for L in np.unique(y)],
                    note="needs authentic + >=3 levels with >=2 preparations: " + str(note_extra or ""))
    bands = usable[1:]
    q = float(np.quantile(bands, 0.5))
    lowb, highb = list(bands[bands <= q]), list(bands[bands > q])
    PA, PB = alt_split(g)
    rec = dict(name=name, matrix=matrix, adulterant=adulterant, instrument=instrument,
               n=int(len(y)), n_channel=int(X.shape[1]), n_groups=int(len(set(g))),
               levels=[float(v) for v in usable],
               spectra_per_level=[int((y == L).sum()) for L in usable],
               preps_per_level=[int(len(set(g[y == L]))) for L in usable],
               band_split=dict(low=[float(v) for v in lowb], high=[float(v) for v in highb]),
               prep_folds=[len(PA), len(PB)], geometry={}, points=[], skips=[])
    pos = {}
    for i, v in enumerate(y):
        pos.setdefault(round(float(v), 6), []).append(i)
    nz = [k for k in sorted(pos) if k > 0]
    contiguous = all(len(v) == v[-1] - v[0] + 1 for v in pos.values())
    ordered = all(pos[nz[i]][0] < pos[nz[i + 1]][0] for i in range(len(nz) - 1)) if len(nz) > 1 else False
    rec["rows"] = dict(each_level_one_interval=bool(contiguous),
                       intervals_ordered_by_level=bool(ordered))
    for tag in ("abs", "snv"):
        Z = prep(X, tag)
        gf = {k: band_stats(Z, y, v) for k, v in
              (("full", list(usable)),
               ("low", [0.0] + lowb), ("high", [0.0] + highb))}
        rec["geometry"][tag] = gf
        for direction, fitb, testb in (("down", highb, lowb), ("up", lowb, highb)):
            side = "high" if direction == "down" else "low"
            if not fitb or not testb:
                rec["skips"].append("%s/%s: empty band" % (tag, direction))
                continue
            if gf[side] is None:
                rec["skips"].append("%s/%s: fit band has <2 levels with preparations, "
                                    "so no fit-band geometry exists" % (tag, direction))
                continue
            for fp, tp in ((PA, PB), (PB, PA)):
                p = one_point(Z, y, g, fitb, testb, fp, tp, gf[side])
                if p:
                    rec["points"].append(dict(metric=tag, direction=direction, **p))
                else:
                    rec["skips"].append("%s/%s: fit/test rows or preparations below the "
                                        "minimum after the disjoint split" % (tag, direction))
    fulls = [rec["geometry"][t]["full"] for t in ("abs", "snv") if rec["geometry"][t]["full"]]
    ok = [t for t in ("abs", "snv") if rec["geometry"][t]["full"]
          and rec["geometry"][t]["full"]["monotone"] and rec["geometry"][t]["full"]["snr_top"] >= 1]
    rec["dose_axis"] = ("usable" if ok else
                        "low_snr" if any(f["monotone"] for f in fulls) else
                        "non_monotone" if fulls else "undefined")
    rec["usable_in_metrics"] = ok
    if not rec["points"]:
        rec["no_point_reason"] = " | ".join(sorted(set(rec["skips"])))[:400]
    rec["aliasing_flag"] = bool(contiguous and ordered and rec["dose_axis"] != "usable")
    if note_extra:
        rec["note"] = note_extra
    return rec


def collect(cache=None):
    recs = {}

    def dump():
        if cache:
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(recs, f, ensure_ascii=False)

    for nm in PP.ALL:
        d = PP.load(nm)
        recs[nm] = audit_one(nm, d["X"], d["y"], d["groups"], d["meta"]["matrix"],
                             d["meta"]["adulterant"], d["meta"].get("instrument", "ATR-FTIR"))
        dump()
        print("done", nm, recs[nm].get("dose_axis") or recs[nm].get("note"),
              "points=%d" % len(recs[nm].get("points", [])), flush=True)
    h = ED.load_honey()
    recs["HONEY"] = audit_one("HONEY", h["X"], h["y"], h["groups"], "honey", "sucrose",
                              "FTIR transmission (%T -> A)",
                              note_extra="level 100 is a 1-row pure sucrose endmember, excluded "
                                         "by the >=2-preparations rule; panel is fully paired so "
                                         "disjointness is imposed on preparations")
    dump()
    print("done HONEY", recs["HONEY"].get("dose_axis") or recs["HONEY"].get("note"),
          "points=%d" % len(recs["HONEY"].get("points", [])), flush=True)
    for tag, d in ED.load_coconut().items():
        recs["COCO_" + tag] = audit_one("COCO_" + tag, d["X"], d["y"], d["groups"],
                                        "coconut milk", "water", d["meta"]["instrument"],
                                        note_extra="rows carry no sample id: every row is its own "
                                                   "preparation")
        dump()
        print("done COCO_" + tag, recs["COCO_" + tag].get("dose_axis") or recs["COCO_" + tag].get("note"),
              "points=%d" % len(recs["COCO_" + tag].get("points", [])), flush=True)
    return recs


def summarise(recs):
    pts = [dict(corpus=nm, **p) for nm, r in recs.items() for p in r.get("points", [])]
    if not pts:
        return dict(note="no extrapolation points survived the rules")
    ab = np.array([abs(p["rel_bias_pct"]) for p in pts], float)
    cand = dict(slope_cv_fit=np.array([p["slope_cv_fit"] for p in pts], float),
                r2_pred_loo=np.array([np.nan if p["r2_pred_loo"] is None else p["r2_pred_loo"]
                                      for p in pts], float),
                r2_in_fit=np.array([p["r2_in_fit"] for p in pts], float),
                sv1_fit=np.array([p["sv1_fit"] for p in pts], float),
                snr_fit=np.array([p["snr_fit"] for p in pts], float))
    out = dict(n_points=len(pts), n_corpora=len({p["corpus"] for p in pts}),
               point_level_spearman_absbias={k: round(float(spearmanr(v, ab).statistic), 4)
                                             for k, v in cand.items()
                                             if np.isfinite(v).sum() > 2})
    out["point_level_spearman_signed"] = {
        k: round(float(spearmanr(v, np.array([p["rel_bias_pct"] for p in pts], float)).statistic), 4)
        for k, v in cand.items() if np.isfinite(v).sum() > 2}
    # corpus level: one observation per corpus = median of its |bias|, median of its predictor
    cs = sorted({p["corpus"] for p in pts})
    med = lambda f, sub: float(np.median([f(p) for p in sub]))
    x = np.array([med(lambda p: p["slope_cv_fit"], [p for p in pts if p["corpus"] == c]) for c in cs])
    yv = np.array([med(lambda p: abs(p["rel_bias_pct"]), [p for p in pts if p["corpus"] == c])
                   for c in cs])
    ri = np.array([med(lambda p: (np.nan if p["r2_pred_loo"] is None else p["r2_pred_loo"]),
                       [p for p in pts if p["corpus"] == c]) for c in cs])
    out["corpora"] = cs
    out["corpus_slope_cv"] = [round(float(a), 4) for a in x]
    out["corpus_abs_bias"] = [round(float(a), 2) for a in yv]
    out["corpus_r2_pred_loo"] = [None if np.isnan(a) else round(float(a), 4) for a in ri]
    out["spearman_corpus_level_slope_cv"] = round(float(spearmanr(x, yv).statistic), 4)
    m = np.isfinite(ri)
    out["spearman_corpus_level_r2_pred_loo"] = (
        round(float(spearmanr(ri[m], yv[m]).statistic), 4) if m.sum() > 2 else None)
    # cluster bootstrap over corpora for the corpus-level rank correlation
    rng = np.random.default_rng(20261101)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(cs), len(cs))
        if len(set(x[idx])) < 3:
            continue
        boot.append(spearmanr(x[idx], yv[idx]).statistic)
    boot = np.array([b for b in boot if np.isfinite(b)])
    out["cluster_bootstrap_rho"] = dict(n=len(boot), median=round(float(np.median(boot)), 4),
                                        lo=round(float(np.percentile(boot, 2.5)), 4),
                                        hi=round(float(np.percentile(boot, 97.5)), 4))
    # leave-one-corpus-out: linear map and threshold rule
    lin_pred = np.zeros(len(cs))
    thr_call, thr_truth, cuts = [], [], []
    lab = yv <= PASS_ABS_BIAS
    for i in range(len(cs)):
        sel = np.ones(len(cs), bool)
        sel[i] = False
        b = np.polyfit(x[sel], yv[sel], 1)
        lin_pred[i] = np.polyval(b, x[i])
        cuts.append(float(np.median(x[sel])))
        thr_call.append(bool(x[i] < cuts[-1]))
        thr_truth.append(bool(lab[i]))
    out["loocv_linear"] = dict(pred=[round(float(a), 2) for a in lin_pred],
                               pearson=round(float(np.corrcoef(lin_pred, yv)[0, 1]), 4),
                               mae=round(float(np.mean(np.abs(lin_pred - yv))), 2),
                               r2=round(float(1 - ((lin_pred - yv) ** 2).sum() /
                                              ((yv - yv.mean()) ** 2).sum()), 4))
    out["loocv_threshold"] = dict(cuts=[round(c, 4) for c in cuts], calls=thr_call, truth=thr_truth,
                                  accuracy=round(float(np.mean([a == t for a, t in zip(thr_call, thr_truth)])), 3),
                                  pass_rule="|bias| <= %d%%" % PASS_ABS_BIAS)
    out["new_corpora_only"] = {}
    for nm in ("HONEY", "COCO_traditional", "COCO_instant"):
        sub = [p for p in pts if p["corpus"] == nm]
        if sub:
            out["new_corpora_only"][nm] = dict(
                n=len(sub), slope_cv_range=[round(min(p["slope_cv_fit"] for p in sub), 4),
                                           round(max(p["slope_cv_fit"] for p in sub), 4)],
                abs_bias_range=[round(min(abs(p["rel_bias_pct"]) for p in sub), 2),
                                round(max(abs(p["rel_bias_pct"]) for p in sub), 2)])
    return out


if __name__ == "__main__":
    cache = p5_paths.RESULTS + "/exp45_corpora.cache.json"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if "--fresh" not in sys.argv and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            recs = json.load(f)
        print("resumed %d corpora from cache" % len(recs), flush=True)
    else:
        recs = collect(cache=cache)
    summ = summarise(recs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(rules=dict(min_fit_rows=MIN_FIT_ROWS, min_test_rows=MIN_TEST_ROWS,
                                  min_fit_groups=MIN_FIT_GROUPS, min_test_groups=MIN_TEST_GROUPS,
                                  pass_abs_bias_pct=PASS_ABS_BIAS),
                       corpora=recs, summary=summ), f, indent=1, ensure_ascii=False)
    print("\n--- summary ---")
    print(json.dumps(summ, indent=1, ensure_ascii=False)[:3000])
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
