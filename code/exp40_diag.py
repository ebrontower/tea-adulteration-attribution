"""Diagnostics behind exp40 / the public-corpus section.

(1) Public corpus: are the three pure controls (one per adulterant series) one reference or three
    batches, and how far is each from the pooled authentic mean, raw and whitened?
(2) Public corpus: whitened distance to the true candidate, adulterated vs authentic rows.
(3) Tea: the same quantities for the pure-tea rows under the whitened blocked protocol, so the
    authentic-row behaviour can be compared between the two corpora.
"""
from __future__ import annotations

import json

import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp40_public_pipeline as P40
import exp33_whitened_attr as TEA
import p5_paths  # noqa: E402

out = {}
print("== public corpus")
X, comp, level, block, series, n_dup, nch = P40.load_cinnamon()
cls_a = comp.copy()
wh = P40.Whitener(X, cls_a, np.arange(len(comp)))     # whole-corpus metric: descriptive only

pure = np.where(comp == 0)[0]
m0 = X[pure].mean(0)
print("authentic rows:", len(pure), "per series:", [int((series[pure] == s).sum()) for s in (1, 2, 3)])
cent = {s: X[pure[series[pure] == s]].mean(0) for s in (1, 2, 3)}

# scale references: within-series scatter of the authentic rows, and the adulterant increments
scat_raw = {s: float(np.mean([np.linalg.norm(X[i] - cent[s]) for i in pure if series[i] == s]))
            for s in (1, 2, 3)}
scat_wh = {s: float(np.mean([wh.dist((X[i] - cent[s])[None, :])[0]
                             for i in pure if series[i] == s])) for s in (1, 2, 3)}
pairs_raw = {(a, b): float(np.linalg.norm(cent[a] - cent[b])) for a in (1, 2) for b in (a + 1, 3)}
pairs_wh = {(a, b): float(wh.dist((cent[a] - cent[b])[None, :])[0])
            for a in (1, 2) for b in (a + 1, 3)}
to_pooled_raw = {s: float(np.linalg.norm(cent[s] - m0)) for s in (1, 2, 3)}
to_pooled_wh = {s: float(wh.dist((cent[s] - m0)[None, :])[0]) for s in (1, 2, 3)}
_m0, D, sigma = P40.fit_dict(X, comp, np.arange(len(X)))
incr_raw = [round(float(np.linalg.norm(D[:, j])), 3) for j in range(3)]
incr_wh = [round(float(wh.dist(D[:, j][None, :])[0]), 3) for j in range(3)]
out["authentic_batches"] = dict(
    per_series=[int((series[pure] == s).sum()) for s in (1, 2, 3)],
    within_series_scatter_raw={s: round(v, 3) for s, v in scat_raw.items()},
    within_series_scatter_white={s: round(v, 3) for s, v in scat_wh.items()},
    between_series_raw={f"{a}-{b}": round(v, 3) for (a, b), v in pairs_raw.items()},
    between_series_white={f"{a}-{b}": round(v, 3) for (a, b), v in pairs_wh.items()},
    centroid_to_pooled_raw={s: round(v, 3) for s, v in to_pooled_raw.items()},
    centroid_to_pooled_white={s: round(v, 3) for s, v in to_pooled_wh.items()},
    increment_norms_raw=incr_raw, increment_norms_white=incr_wh)
for k, v in out["authentic_batches"].items():
    print(" ", k, v)

# (2) whitened distance of every row to its own true candidate, by role
cand = P40.subsets(3)
Z = wh.coords(X)
Zc = wh.coords(np.stack([P40.pred(m0, D, S) for S in cand]))
d2 = P40._sq_dists(Z, Zc)
idx_true = [cand.index(P40.true_set(c)) for c in comp]
own = np.sqrt(d2[np.arange(len(comp)), idx_true])
out["own_candidate_white"] = {
    "authentic": dict(n=int(len(pure)), mean=round(float(own[pure].mean()), 3),
                      q10=round(float(np.quantile(own[pure], .1)), 3),
                      q90=round(float(np.quantile(own[pure], .9)), 3),
                      by_series={s: round(float(own[pure[series[pure] == s]].mean()), 3)
                                 for s in (1, 2, 3)}),
    "adulterated": dict(n=int((comp > 0).sum()), mean=round(float(own[comp > 0].mean()), 3),
                        q10=round(float(np.quantile(own[comp > 0], .1)), 3),
                        q90=round(float(np.quantile(own[comp > 0], .9)), 3))}
print(" (2) distance to own candidate, whitened:", json.dumps(out["own_candidate_white"]))
# local scale, whitened: how far is each row from its 5 nearest fit-order neighbours
loc = np.sqrt(np.partition(P40._sq_dists(Z, Z), 6, axis=1)[:, 1:6]).mean(1)
out["own_over_local"] = {
    "authentic": round(float(np.mean((own / loc)[pure])), 3),
    "adulterated": round(float(np.mean((own / loc)[comp > 0])), 3)}
print(" (2b) own-candidate distance / local scale:", out["own_over_local"])

print("== tea, pure rows under the whitened blocked protocol")
Xt, yr = TEA.load()
candt = TEA.subsets()
S_of = {S: j for j, S in enumerate(candt)}
recs = []
for fit_i, cal_i, test_i in TEA.make_splits(yr, "blocked", np.random.default_rng(0)):
    m0t, Dt, sigt = TEA.fit_dict(Xt, yr, fit_i)
    wht = TEA.Whitener(Xt, yr, fit_i)
    Zt = wht.coords(Xt)
    Zct = wht.coords(np.stack([TEA.pred(m0t, Dt, S) for S in candt]))
    Zf = Zt[fit_i]
    kloc = max(1, min(5, len(fit_i)))
    loct = np.sqrt(np.partition(P40._sq_dists(Zt, Zf), kloc - 1, axis=1)[:, :kloc]).mean(1)
    d2t = P40._sq_dists(Zt, Zct)
    kt_cal = np.array([S_of[tuple(sorted(TEA.true_set(int(yr[i]))))] for i in cal_i])
    raw_cal = np.sqrt(d2t[np.ix_(cal_i, kt_cal)].diagonal())
    q_std = float(np.quantile(raw_cal, min(1.0, 0.9 * (1 + 1 / len(raw_cal)))))
    sing_cal = cal_i[yr[cal_i] > 0]                      # gate calibrated on single adulterants
    bestn = np.array([np.sqrt(d2t[i]).min() / loct[i] for i in sing_cal])
    tau = float(np.quantile(bestn, min(1.0, 0.9 * (1 + 1 / len(bestn)))))
    for i in test_i:
        raw = np.sqrt(d2t[i])
        sc = raw / loct[i]
        kt = S_of[tuple(sorted(TEA.true_set(int(yr[i]))))]
        recs.append(dict(pure=bool(int(yr[i]) == 0),
                         cov_std=bool(kt in np.where(raw <= q_std)[0]),
                         size_std=int((raw <= q_std).sum()),
                         top1_std=bool(int(np.argmin(raw)) == kt),
                         accept=bool(sc.min() <= tau),
                         own=float(raw[kt]), scale=float(loct[i])))
tea = {}
for nm, sel in (("pure_tea", [r for r in recs if r["pure"]]),
                ("adulterated", [r for r in recs if not r["pure"]]),
                ("all", recs)):
    tea[nm] = dict(n=len(sel),
                   coverage_std=round(float(np.mean([r["cov_std"] for r in sel])), 3),
                   size_std=round(float(np.mean([r["size_std"] for r in sel])), 2),
                   top1_std=round(float(np.mean([r["top1_std"] for r in sel])), 3),
                   accept_gate=round(float(np.mean([r["accept"] for r in sel])), 3),
                   own_over_scale=round(float(np.mean([r["own"] / r["scale"] for r in sel])), 3))
print(json.dumps(tea, indent=1))
out["tea_blocked_white"] = tea

OUT = os.path.join(p5_paths.RESULTS, "exp40_diag_pure.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved exp40_diag_pure.json")
