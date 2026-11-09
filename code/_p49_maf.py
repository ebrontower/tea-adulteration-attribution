"""exp49 - minimum attributable fraction on the public cinnamon corpus (2026-11-03).

Pre-registration (written before the run, verdicts computed mechanically here):
    preregistrations/exp49_minimum_attributable_fraction_v1.md

Zero new measurement: every number comes from the same folds, dictionary, thresholds and score
arms that produced Table 8 (exp40_public_pipeline). exp40 already writes a record per test spectrum
 carrying comp / level / block / rank / covered / size / accepted; this script only re-buckets those
records by recorded loading level (12 non-zero levels instead of exp40's three coarse buckets),
adds the monotone-tail MAF rule and the per-level block-cluster intervals, and adds one descriptive
dose-response geometry column (increment norm / within-level residual, raw and whitened) to explain
the curve rather than just draw it.

The pooled global statistics are asserted against exp40_public.json so a divergence in the
re-bucketing would fail loudly instead of silently changing Table 8.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr  # noqa: E402

import exp40_public_pipeline as P
import p5_paths  # noqa: E402

OUT = os.environ.get("P49_OUT") or p5_paths.RESULTS + "/exp49_maf.json"
EXP40 = p5_paths.RESULTS + "/exp40_public.json"
LEVELS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24]
TOP1_CRIT, COV_CRIT, ACCEPT_CRIT, ACC_TOP1_CRIT = 0.80, 0.90, 0.50, 0.80
METRICS = ("euclid", "white_a", "white_b")


# --------------------------------------------------------------------------- aggregation
def stat(rr, keys):
    """Per-bucket statistics on one score arm. keys = (covered, size, empty, rank) column names."""
    kc, ks, ke, kr = keys
    if not rr:
        return None
    acc = [r for r in rr if r["accepted"]]
    return dict(
        n=len(rr),
        top1=round(float(np.mean([r[kr] == 0 for r in rr])), 3),
        coverage=round(float(np.mean([r[kc] for r in rr])), 3),
        size=round(float(np.mean([r[ks] for r in rr])), 2),
        empty=round(float(np.mean([r[ke] for r in rr])), 3),
        accept=round(float(np.mean([r["accepted"] for r in rr])), 3),
        top1_acc=round(float(np.mean([r[kr] == 0 for r in acc])), 3) if acc else None,
        n_accepted=len(acc),
        mean_best_norm=round(float(np.mean([r["best"] for r in rr])), 3),
        mean_best_raw=round(float(np.mean([r["best_q"] for r in rr])), 4),
    )


STD_KEYS = ("covered_std", "size_std", "empty_std", "rank_std")


def rekey(rows, arm):
    if arm == "norm":
        return rows
    return [{**r, "covered": r["covered_std"], "size": r["size_std"],
             "empty": r["empty_std"], "rank": r["rank_std"]} for r in rows]


def maf(levels, per_level, crit):
    """Monotone-tail minimum attributable level: smallest l with crit(l..) all true.

    Returns (level or None, note). `note` is never empty when the curve is non-monotone, because a
    non-monotone curve is itself a finding and must not be hidden by taking the best level.
    """
    ls = sorted(levels)
    ok = {l: crit(per_level.get(l)) for l in ls}
    first_fail_above = None
    for i, l in enumerate(ls):
        if ok[l] and all(ok[x] for x in ls[i + 1:]):
            return l, "monotone tail holds"
    fails = [l for l in ls if not ok[l]]
    return None, f"no level satisfies the criterion together with all higher levels; failing: {fails}"


def crit_top1(s):
    return bool(s and s["top1"] >= TOP1_CRIT)


def crit_cov(s):
    return bool(s and s["coverage"] >= COV_CRIT)


def crit_delivery(s):
    return bool(s and s["accept"] >= ACCEPT_CRIT and (s["top1_acc"] or 0.0) >= ACC_TOP1_CRIT)


# --------------------------------------------------------------------------- dose-response geometry
def geometry_curve(X, comp, level, series, cls_a, wh):
    """Per (adulterant, level): how big is the increment, and how big is the noise around it.

    Descriptive statistics of the corpus, computed on all rows of the cell - the same convention
    exp40's geometry() already uses for Table 8's dictionary numbers.
    """
    out = {}
    for s in (1, 2, 3):
        m0 = X[(series == s) & (comp == 0)].mean(0)
        for lv in LEVELS:
            m = (series == s) & (level == lv)
            mu = X[m].mean(0)
            R = X[m] - mu
            incr_raw = float(np.linalg.norm(mu - m0))
            res_raw = float(np.mean(np.linalg.norm(R, axis=1)))
            wmu, w0 = wh.coords(mu[None, :])[0], wh.coords(m0[None, :])[0]
            incr_white = float(np.linalg.norm(wmu - w0))
            res_white = float(np.mean(np.linalg.norm(wh.coords(R) - (wmu - w0), axis=1)))
            out[P.NAME[s - 1]] = out.get(P.NAME[s - 1], {})
            out[P.NAME[s - 1]][lv] = dict(
                incr_raw=round(incr_raw, 4), res_raw=round(res_raw, 4),
                snr_raw=round(incr_raw / res_raw, 3),
                incr_white=round(incr_white, 3), res_white=round(res_white, 3),
                snr_white=round(incr_white / res_white, 3),
                incr_over_level=round(incr_raw / lv, 4))
    return out


def linearity(geom):
    """Is the increment proportional to the recorded loading? (additivity of the spectral model)"""
    out = {}
    for nm, d in geom.items():
        lv = np.array(sorted(d), float)
        for key in ("incr_raw", "incr_white"):
            y = np.array([d[l][key] for l in sorted(d)], float)
            r = float(np.corrcoef(lv, y)[0, 1])
            sl = np.polyfit(lv, y, 1)[0]
            cv = float(np.std(y / lv) / np.mean(y / lv))
            out.setdefault(nm, {})[key] = dict(pearson_r=round(r, 4), slope=round(float(sl), 5),
                                               cv_of_ratio=round(cv, 4))
    return out


# --------------------------------------------------------------------------- main
def main():
    X, comp, level, block, series, n_dup, nch = P.load_cinnamon()
    cls_a = comp.copy()
    cls_b = np.zeros(len(comp), int)
    for s in (1, 2, 3):
        for lv in np.unique(level[series == s]):
            cls_b[(series == s) & (level == lv)] = 100 * s + int(lv)
    cand, atoms = P.subsets(3), [1, 2, 3]
    bf = P.blocked_folds(block)

    rows_by_metric, glob_check = {}, {}
    for metric in METRICS:
        _, rows = P.run_arm(X, comp, level, block, cls_a, cls_b, cand, atoms, bf, metric)
        rows_by_metric[metric] = rows
        std = rekey(rows, "std")
        glob_check[metric] = dict(
            coverage=round(float(np.mean([r["covered"] for r in std])), 3),
            size=round(float(np.mean([r["size"] for r in std])), 2),
            top1=round(float(np.mean([r["rank"] == 0 for r in std])), 3),
            empty=round(float(np.mean([r["empty"] for r in std])), 3),
            accept=round(float(np.mean([r["accepted"] for r in std])), 3))

    # --- zero-new-measurement gate: re-bucketing must reproduce the stored Table 8 global numbers
    ref = json.load(open(EXP40, encoding="utf-8"))["arms"]
    checks = {}
    for metric in ("euclid", "white_a", "white_b"):
        want = ref[f"blocked/{metric}/std"]["all"]
        got = glob_check[metric]
        diffs = {k: [got[k], want[k]] for k in ("coverage", "size", "top1", "abstain", "accept")
                 if abs(got[k if k != "abstain" else "empty"] - want[k]) > 1e-9}
        checks[f"blocked/{metric}/std"] = diffs or "identical to exp40_public.json"
    out = dict(
        prereg=r"preregistrations/exp49_minimum_attributable_fraction_v1.md",
        source="cinnamon public corpus [33], walnut shell / peanut / wood dust, 0-24% w/w",
        levels=LEVELS, n_spectra=int(len(comp)), n_channels=int(nch), duplicate_rows=n_dup,
        per_level_n_expected=dict(adulterant_level=50, level_pooled=150, pure=150),
        criteria=dict(top1=TOP1_CRIT, coverage=COV_CRIT, accept=ACCEPT_CRIT,
                      top1_on_accepted=ACC_TOP1_CRIT, monotone_tail=True),
        alpha=P.ALPHA, k_white=P.K_WHITE, k_local=P.K_LOCAL, window=list(P.WINDOW),
        reproduce_exp40=checks, global_std_arm=glob_check)

    # --- per-level curves (standard score arm = the arm carrying the guarantee, as in Table 8)
    curves = {}
    for metric in METRICS:
        std = rekey(rows_by_metric[metric], "std")
        nrm = rows_by_metric[metric]
        per = {l: stat([r for r in std if r["level"] == l and r["comp"] > 0],
                       ("covered", "size", "empty", "rank")) for l in LEVELS}
        per_norm = {l: stat([r for r in nrm if r["level"] == l and r["comp"] > 0],
                            ("covered", "size", "empty", "rank")) for l in LEVELS}
        by_adm = {nm: {l: stat([r for r in std if r["level"] == l and r["comp"] == c],
                               ("covered", "size", "empty", "rank"))
                       for l in LEVELS}
                  for c, nm in enumerate(P.NAME, start=1)}
        pure = stat([r for r in std if r["comp"] == 0], ("covered", "size", "empty", "rank"))
        pure_norm = stat([r for r in nrm if r["comp"] == 0], ("covered", "size", "empty", "rank"))
        mafs = dict(
            naming=maf(LEVELS, per, crit_top1),
            certified_set=maf(LEVELS, per, crit_cov),
            deliverable=maf(LEVELS, per, crit_delivery))
        cov_fail = [l for l in LEVELS if per[l] and per[l]["coverage"] < 0.85]
        rho = spearmanr(LEVELS, [per[l]["top1"] for l in LEVELS if per[l]])
        curves[metric] = dict(
            per_level=per, per_level_normalised=per_norm, by_adulterant=by_adm,
            pure=pure, pure_normalised_gate=pure_norm, maf=mafs,
            coverage_below_085_levels=cov_fail,
            spearman_top1_vs_level=round(float(rho.correlation), 3))

        # intervals over (adulterant, level, block) cells: 30 clusters per level
        iv = {}
        for l in LEVELS:
            sub = [r for r in std if r["level"] == l and r["comp"] > 0]
            iv[l] = P.cluster_boot(sub, ["coverage", "size", "top1", "abstain"])
        curves[metric]["per_level_intervals"] = iv
    out["curves"] = curves

    # --- mechanism columns
    wh = P.Whitener(X, cls_a, np.arange(len(comp)))
    geom = geometry_curve(X, comp, level, series, cls_a, wh)
    out["dose_response_geometry"] = geom
    out["linearity"] = linearity(geom)
    pooled_snr = {}
    for l in LEVELS:
        vals = [geom[nm][l] for nm in P.NAME]
        pooled_snr[l] = dict(snr_raw=round(float(np.mean([v["snr_raw"] for v in vals])), 3),
                             snr_white=round(float(np.mean([v["snr_white"] for v in vals])), 3))
    out["pooled_snr_by_level"] = pooled_snr

    # --- pre-registered verdicts, computed here so they cannot be re-read after seeing the curve
    ea, wa = curves["euclid"], curves["white_a"]
    d1 = [wa["per_level"][l]["top1"] - ea["per_level"][l]["top1"] for l in (1, 2)]
    verdicts = dict(
        L1=dict(fired=bool(all(abs(v) < 0.10 for v in d1)),
                gap_at_1pct=round(d1[0], 3), gap_at_2pct=round(d1[1], 3),
                rule="whitened minus euclidean top-1 at the two lowest levels, all |gap| < 0.10 "
                     "=> no 'limit pushed lower' claim allowed"),
        L2=dict(fired=bool(len(wa["coverage_below_085_levels"]) > 0),
                levels=wa["coverage_below_085_levels"],
                euclid_levels=ea["coverage_below_085_levels"],
                rule="any level whose whitened standard-score marginal coverage < 0.85"),
        L3=dict(fired=not (0.80 <= wa["pure_normalised_gate"]["accept"] <= 1.00),
                pure_accept=wa["pure_normalised_gate"]["accept"],
                euclid_pure_accept=ea["pure_normalised_gate"]["accept"],
                rule="authentic (0%) acceptance outside 1-alpha +/- 0.10 => the gate is calibrated "
                     "on adulterated spectra only and cannot be described as an alpha false-reject "
                     "rate on authentic samples"),
        L4=dict(fired=bool(ea["spearman_top1_vs_level"] >= 0.7),
                euclid_rho=ea["spearman_top1_vs_level"], white_a_rho=wa["spearman_top1_vs_level"],
                rule="euclidean top-1 also rises monotonically with level => part of the limit "
                     "gain is SNR, not the metric; MAF must then be reported as a pair of curves"))
    mafs = {m: {k: v[0] for k, v in curves[m]["maf"].items()} for m in METRICS}
    verdicts["maf_table"] = mafs
    verdicts["maf_gain_levels"] = {k: (None if mafs["euclid"][k] is None
                                       else mafs["euclid"][k] - (mafs["white_a"][k] or 0))
                                   for k in mafs["euclid"]}
    out["verdicts"] = verdicts

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------ print
    print("reproduce exp40:", json.dumps(out["reproduce_exp40"], ensure_ascii=False))
    for metric in METRICS:
        c = curves[metric]
        print(f"\n=== {metric} (standard score arm)   rho(top1,level)={c['spearman_top1_vs_level']}"
              f"  MAF={ {k: v[0] for k, v in c['maf'].items()} }")
        print(f"{'lv':>4s} {'n':>4s} {'top1':>6s} {'[lo,hi]':>14s} {'cov':>6s} {'[lo,hi]':>14s} "
              f"{'size':>5s} {'empt':>5s} {'acc':>5s} {'t1@acc':>7s} {'snr_raw':>8s} {'snr_w':>7s}")
        for l in LEVELS:
            s, i = c["per_level"][l], c["per_level_intervals"][l]
            sn = pooled_snr[l]
            print(f"{l:4d} {s['n']:4d} {s['top1']:6.3f} {str(i['top1']):>14s} "
                  f"{s['coverage']:6.3f} {str(i['coverage']):>14s} {s['size']:5.2f} "
                  f"{s['empty']:5.3f} {s['accept']:5.3f} {str(s['top1_acc']):>7s} "
                  f"{sn['snr_raw']:8.2f} {sn['snr_white']:7.2f}")
        p = c["pure"]
        print(f"   pure(0%): cov {p['coverage']} size {p['size']} top1 {p['top1']} "
              f"accept {c['pure_normalised_gate']['accept']} n {p['n']}")
    print("\nverdicts:", json.dumps(verdicts, ensure_ascii=False, indent=1))
    print("linearity:", json.dumps(out["linearity"], ensure_ascii=False))
    print("\nby-adulterant top1 (white_a / euclid):")
    for nm in P.NAME:
        w = [curves["white_a"]["by_adulterant"][nm][l]["top1"] for l in LEVELS]
        e = [curves["euclid"]["by_adulterant"][nm][l]["top1"] for l in LEVELS]
        print(f"  {nm:14s} white {w}\n  {'':14s} euclid {e}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
