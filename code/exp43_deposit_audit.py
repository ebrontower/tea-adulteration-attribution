"""exp43 - pre-flight audit of the recorded-dose deposits (route 2, and the honest end of route 1).

exp41 showed the only two deposits that record a true acquisition session (meats RunA/RunB,
olive oils two sessions) transfer at the ceiling of linear models, so cross-session shift leaves
no room for a learned representation. The remaining corpora are organised by *dose*, so before
any criterion can be validated on them the deposit itself has to pass a geometric check: can a
spectral increment be read off the dose at all?

Six closed-form quantities, computed in two metrics (SNV-normalised and absolute absorbance) so
nothing depends on one analyst's preprocessing. All of them are computable by a reviewer on
their own files.

  mono_spearman      rank correlation of level with ||mu_level - mu_authentic||
  monotone           every step up in dose moves the level mean further from authentic
  worst_rel_step     worst fractional change of that distance as the dose rises (negative =
                     the deposit claims a higher dose looks *closer* to authentic than a lower one)
  cos_increments     mean |cos| between consecutive increments mu_{L+1}-mu_L (linearity of mixing)
  snr_top            ||mu_top - mu_auth|| / mean within-level scatter at the top level
  auth_hetero        within-authentic scatter / median within-scatter of the adulterated levels
  auth_split         max centroid separation inside the authentic block after k=3 clustering,
                     in units of its own scatter, and whether the split follows scan order

Verdict rules, fixed before running and stated in the script so they can be re-derived:
  ok        mono_spearman >= 0.9 and monotone and snr_top >= 1 in at least one metric
  low_snr   monotone in some metric, but the top level lies within its own scatter
  broken    non-monotone in both metrics, or auth_hetero >= 2 (the reference class is not a class)
Each metric is judged on its own and the corpus takes the better of the two, because the dose
response can be carried by absolute absorbance and erased by SNV - which is what happens in D7p.
"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import p5_public_data as PP                                   # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp43_deposit_audit.json"


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def scan_index(ids):
    out = []
    for i in ids:
        m = re.search(r"S(\d+)$", str(i))
        out.append(int(m.group(1)) if m else -1)
    return np.array(out)


def curve(X, y, levels):
    """Distance of each level mean from the authentic mean, plus per-level within scatter."""
    mus = np.array([X[y == L].mean(0) for L in levels])
    dist = np.linalg.norm(mus - mus[0], axis=1)
    within = np.array([np.linalg.norm(X[y == L] - mus[i], axis=1).mean()
                       for i, L in enumerate(levels)])
    return mus, dist, within


def audit(name):
    d = PP.load(name)
    X, y = d["X"], np.asarray(d["y"], float)
    g = np.asarray(d["groups"], object)
    ids = np.asarray(d["ids"], object)
    levels = np.unique(y)
    rec = dict(name=name, matrix=d["meta"]["matrix"], adulterant=d["meta"]["adulterant"],
               instrument=d["meta"]["instrument"], n=int(len(y)), n_channel=int(X.shape[1]),
               n_groups=int(len(set(g))), n_levels=int(len(levels)),
               levels=[float(v) for v in levels],
               spectra_per_level=[int((y == L).sum()) for L in levels],
               axis=[float(d["wvn"][0]), float(d["wvn"][-1])])
    if levels[0] != 0 or len(levels) < 3:
        rec["verdict"] = "no authentic level or fewer than three levels"
        return rec
    for tag, Z in [("snv", snv(X)), ("abs", X)]:
        mus, dist, within = curve(Z, y, levels)
        mono = float(spearmanr(levels[1:], dist[1:]).statistic)
        # worst fractional change of the distance when the dose increases (negative = inversion)
        steps = [(dist[i] - dist[i - 1]) / max(dist[i - 1], 1e-9) for i in range(2, len(levels))]
        max_rel_drop = float(min(steps)) if steps else None
        monotone = bool(all(dist[i] >= dist[i - 1] - 1e-9 for i in range(2, len(levels))))
        inc = np.diff(mus, axis=0)
        norms = np.linalg.norm(inc, axis=1) + 1e-12
        cs = [float(np.abs(np.dot(inc[i] / norms[i], inc[i + 1] / norms[i + 1])))
              for i in range(len(inc) - 1)]
        rec[tag] = dict(mono_spearman=round(mono, 4), monotone=monotone,
                        worst_rel_step=(round(max_rel_drop, 4) if max_rel_drop is not None else None),
                        cos_increments=round(float(np.mean(cs)), 4) if cs else None,
                        snr_top=round(float(dist[-1] / within[-1]), 3),
                        snr_low=round(float(dist[1] / within[1]), 3),
                        auth_hetero=round(float(within[0] / np.median(within[1:])), 3),
                        dist_over_within=[round(float(v / w), 3) for v, w in zip(dist, within)])
    # authentic block: is it one cluster?
    A = snv(X[y == levels[0]])
    sc = scan_index(ids)[y == levels[0]]
    if len(A) >= 12:
        pc = PCA(n_components=4, random_state=0).fit(A)
        Z = pc.transform(A)
        base = float(np.linalg.norm(A - A.mean(0), axis=1).mean())
        km = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Z[:, :2])
        cents = [A[km == c].mean(0) for c in range(3) if (km == c).sum() > 1]
        sep = max([float(np.linalg.norm(a - b)) for i, a in enumerate(cents)
                   for b in cents[i + 1:]] or [0.0])
        cont = None
        if sc.min() >= 0:
            cont = [round(float(np.mean(np.diff(np.sort(sc[km == c])) == 1)), 3)
                    for c in range(3) if (km == c).sum() > 2]
        rec["authentic_block"] = dict(n=int(len(A)), within=round(base, 3),
                                      cluster_sizes=[int(v) for v in np.bincount(km)],
                                      centroid_separation_over_within=round(sep / base, 3),
                                      scan_contiguity=cont)
    per = {}
    for tag in ("snv", "abs"):
        m = rec[tag]
        if m["auth_hetero"] >= 2 or not m["monotone"] or m["mono_spearman"] < 0.9:
            per[tag] = "non_monotone"
        elif m["snr_top"] < 1:
            per[tag] = "low_snr"
        else:
            per[tag] = "ok"
    rec["per_metric"] = per
    if "ok" in per.values():
        rec["verdict"] = "ok"
    elif "low_snr" in per.values():
        rec["verdict"] = "low_snr"
    else:
        rec["verdict"] = "broken"
    return rec


if __name__ == "__main__":
    recs = {}
    for nm in PP.ALL:
        try:
            r = audit(nm)
        except Exception as e:
            r = dict(name=nm, error="%s: %s" % (type(e).__name__, e))
        recs[nm] = r
        s = r.get("snv") or {}
        ab = r.get("abs") or {}
        print("%-4s %-14s/%-14s lvls=%2d  snv mono=%s step=%s cos=%s snr=%s het=%s | abs snr=%s | %s %s" %
              (nm, r.get("matrix"), r.get("adulterant"), r.get("n_levels", 0),
               s.get("mono_spearman"), s.get("monotone"), s.get("cos_increments"),
               s.get("snr_top"), s.get("auth_hetero"), ab.get("snr_top"),
               r.get("per_metric"), r.get("verdict")))
        blk = r.get("authentic_block")
        if blk:
            print("      authentic block n=%d sizes=%s sep/within=%s contiguity=%s" %
                  (blk["n"], blk["cluster_sizes"], blk["centroid_separation_over_within"],
                   blk["scan_contiguity"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
