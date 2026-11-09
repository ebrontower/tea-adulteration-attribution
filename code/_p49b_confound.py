"""exp49b - confound probe for the cinnamon dose-response curve (2026-11-03).

Written after exp49 produced a curve that needs explaining before it can be quoted:
the raw increment norm ||mu(adulterant, level) - m0(series)|| is monotone in the recorded loading for
peanut only (0.0080 -> 0.1462, r = 1.00), while for walnut shell and wood dust it is piecewise
constant and DECREASING (walnut 0.588 for 1-4%, 0.192 for 6-12%, 0.169 for 15-24%). Two of three
series therefore have level-group-to-level-group baseline steps as large as or larger than the
concentration increment they are supposed to carry.

That makes one question decisive before any cinnamon number is used as a headline:
does the whitened metric name the adulterant at 1% because of the adulterant's chemistry, or because
each series' spectra carry a per-group baseline/throughput fingerprint that is series-specific?

Three checks, all on the same folds and the same pipeline code (exp40_public_pipeline):

  1. scalar dose response per series - window-mean absorbance and the pure-referenced increment,
     plus the size of the pure-set-to-pure-set gap compared with the 1% increment (session structure);
  2. SNV ablation - per-spectrum mean-centring and scaling removes offset and multiplicative
     throughput before anything is fitted. Chemistry-based attribution survives; baseline
     fingerprinting does not;
  3. first-derivative ablation - the classical baseline/slope remover, as a second, independent
     confound filter.
"""
from __future__ import annotations

import json
import os

import numpy as np

import exp40_public_pipeline as P
import p5_paths  # noqa: E402

OUT = os.environ.get("P49B_OUT") or p5_paths.RESULTS + "/exp49b_confound.json"
LEVELS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24]


def snv(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True)
    return (X - m) / np.maximum(s, 1e-12)


def deriv1(X):
    """First difference along the (ascending) wavenumber axis, divided by the channel spacing."""
    return np.diff(X, axis=1)


def curve_from_rows(rows):
    out = {}
    for l in LEVELS:
        sub = [r for r in rows if r["level"] == l and r["comp"] > 0]
        out[l] = dict(n=len(sub),
                      top1=round(float(np.mean([r["rank_std"] == 0 for r in sub])), 3),
                      coverage=round(float(np.mean([r["covered_std"] for r in sub])), 3),
                      size=round(float(np.mean([r["size_std"] for r in sub])), 2),
                      empty=round(float(np.mean([r["empty_std"] for r in sub])), 3))
    pure = [r for r in rows if r["comp"] == 0]
    out["pure"] = dict(n=len(pure),
                       top1=round(float(np.mean([r["rank_std"] == 0 for r in pure])), 3),
                       coverage=round(float(np.mean([r["covered_std"] for r in pure])), 3),
                       accept=round(float(np.mean([r["accepted"] for r in pure])), 3))
    return out


def per_adulterant(rows):
    out = {}
    for c, nm in enumerate(P.NAME, start=1):
        out[nm] = {l: round(float(np.mean([r["rank_std"] == 0 for r in rows
                                           if r["level"] == l and r["comp"] == c])), 3)
                   for l in LEVELS}
    return out


def main():
    X, comp, level, block, series, n_dup, nch = P.load_cinnamon()
    cls_a = comp.copy()
    cls_b = np.zeros(len(comp), int)
    for s in (1, 2, 3):
        for lv in np.unique(level[series == s]):
            cls_b[(series == s) & (level == lv)] = 100 * s + int(lv)
    cand, atoms, bf = P.subsets(3), [1, 2, 3], P.blocked_folds(block)

    out = dict(levels=LEVELS, n_channels_original=int(nch),
               note="checks 2 and 3 re-run the identical pipeline on a preprocessed X; "
                    "the dictionary, whitener, folds, alpha and score arms are unchanged")

    # ---------------------------------------------------------------- check 1: scalar dose response
    dose = {}
    for s in (1, 2, 3):
        nm = P.NAME[s - 1]
        m0 = X[(series == s) & (comp == 0)].mean(0)
        cell_mean = {}
        incr = {}
        for lv in LEVELS:
            R = X[(series == s) & (level == lv)]
            mu = R.mean(0)
            cell_mean[lv] = float(np.mean(mu))
            incr[lv] = float(np.linalg.norm(mu - m0))
        # between-group step (what a session/baseline jump costs) vs within-cell row scatter
        mus = np.array([X[(series == s) & (level == lv)].mean(0) for lv in LEVELS])
        steps = np.abs(np.diff(mus, axis=0))
        within = np.array([np.mean(np.abs(X[(series == s) & (level == lv)]
                                          - X[(series == s) & (level == lv)].mean(0)), axis=1)
                           for lv in LEVELS]).mean()
        dose[nm] = dict(window_mean_by_level={k: round(v, 5) for k, v in cell_mean.items()},
                        incr_by_level={k: round(v, 5) for k, v in incr.items()},
                        monotone_incr=bool(np.all(np.diff([incr[l] for l in LEVELS]) > 0)),
                        mean_step_between_levels=round(float(steps.mean()), 5),
                        mean_row_scatter_within_cell=round(float(within), 5),
                        step_over_scatter=round(float(steps.mean() / within), 3))
    # pure-set-to-pure-set gaps: are the three 0% controls the same cinnamon on the same baseline?
    m0s = {P.NAME[s - 1]: X[(series == s) & (comp == 0)].mean(0) for s in (1, 2, 3)}
    wh = P.Whitener(X, cls_a, np.arange(len(comp)))
    gaps = {}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        na, nb = P.NAME[a], P.NAME[b]
        d_raw = float(np.linalg.norm(m0s[na] - m0s[nb]))
        d_white = float(wh.dist((m0s[na] - m0s[nb])[None, :])[0])
        # same-scale references: 1% increment of each series, and one row's scatter
        gaps[f"{na} vs {nb}"] = dict(d_raw=round(d_raw, 5), d_white=round(d_white, 3))
    row_gap_raw, row_gap_white = [], []
    for s in (1, 2, 3):
        idx = np.where((series == s) & (comp == 0))[0]
        for i in idx[:20]:
            for t in (1, 2, 3):
                if t == s:
                    continue
                j = np.where((series == t) & (comp == 0))[0][i % 50]
                row_gap_raw.append(float(np.linalg.norm(X[i] - X[j])))
                row_gap_white.append(float(wh.dist((X[i] - X[j])[None, :])[0]))
    ref = {}
    for s in (1, 2, 3):
        idx = np.where((series == s) & (level == 1))[0][:20]
        d = [float(np.linalg.norm(X[i] - m0s[P.NAME[s - 1]])) for i in idx]
        ref[P.NAME[s - 1]] = dict(one_pct_row_gap_raw=round(float(np.mean(d)), 5),
                                  one_pct_row_gap_white=round(
                                      float(np.mean([wh.dist((X[i] - m0s[P.NAME[s - 1]])[None, :])[0]
                                                     for i in idx])), 3))
    out["session_structure"] = dict(pure_mean_gaps=gaps,
                                    pure_row_gap_raw_mean=round(float(np.mean(row_gap_raw)), 5),
                                    pure_row_gap_white_mean=round(float(np.mean(row_gap_white)), 3),
                                    one_pct_reference=ref)
    out["scalar_dose_response"] = dose

    # ---------------------------------------------------------- checks 2 and 3: confound filters
    for tag, Xt in (("snv", snv(X)), ("deriv1", deriv1(X))):
        arms = {}
        for metric in ("euclid", "white_a"):
            _, rows = P.run_arm(Xt, comp, level, block, cls_a, cls_b, cand, atoms, bf, metric)
            arms[metric] = dict(curve=curve_from_rows(rows), per_adulterant=per_adulterant(rows),
                                global_top1=round(float(np.mean([r["rank_std"] == 0
                                                                 for r in rows if r["comp"] > 0])), 3),
                                global_cov=round(float(np.mean([r["covered_std"] for r in rows
                                                               if r["comp"] > 0])), 3))
        out[tag] = arms
        print(f"\n=== {tag}: global top1 (adulterated) "
              f"euclid {arms['euclid']['global_top1']} white_a {arms['white_a']['global_top1']}")
        for metric in arms:
            c = arms[metric]["curve"]
            print(f"  {metric:8s} top1 by level "
                  + " ".join(f"{c[l]['top1']:.2f}" for l in LEVELS))
            print(f"  {'':8s} cov  by level "
                  + " ".join(f"{c[l]['coverage']:.2f}" for l in LEVELS))
            if metric == "white_a":
                print(f"  {'':8s} pure: cov {c['pure']['coverage']} top1 {c['pure']['top1']} "
                      f"accept {c['pure']['accept']}")

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("\nsession:", json.dumps(out["session_structure"], ensure_ascii=False))
    print("dose:", json.dumps({k: v["monotone_incr"] for k, v in dose.items()}))
    for nm, v in dose.items():
        print(f"  {nm:14s} incr {v['incr_by_level']}")
        print(f"  {'':14s} window mean {v['window_mean_by_level']}")
        print(f"  {'':14s} step/scatter {v['step_over_scatter']}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
