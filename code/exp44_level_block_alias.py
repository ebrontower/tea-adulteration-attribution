"""exp44 - is the recorded dose factor separable from a block factor in these deposits?

exp43 found that in four of the seven recorded-dose corpora the distance from authentic is a
*step function* of the declared percentage: identical to three decimals within a run of levels
and jumping between runs, with the low levels sometimes farthest from authentic. A genuine
additive mixture cannot do that - its mean offset should grow with the amount. The alternative
explanation is that the deposition aliases the declared level with an acquisition/preparation
block, so what looks like a dose axis is partly a batch axis of larger magnitude.

Test, per file and per metric (absolute absorbance and SNV):
  o_L          mean offset of level L from the authentic mean (channels)
  sv1_share    share of the offset matrix [o_L] carried by its first singular direction
               (one nuisance axis -> the offsets are one direction with free magnitudes)
  cos_to_top   cos(o_L, o_top) for every level: a shared direction means high cos
  slope_cv     CV of ||o_L|| / L over levels: 0 would mean exact linearity of mixing
  r2_dose      R^2 of a rank-1 model o_L = L * v fitted through the origin
  rows         structural, model-free: does each declared level occupy one contiguous interval of
               file rows, and are those intervals ordered by level? Contiguous rows only prove the
               sheet is *organised* by level; what makes that fatal is that it comes together with
               step-function distances - one constant offset per level run, which chemistry cannot
               produce, so the deposited sheet carries a per-level-block quantity on top of the dose.
  alias_verdict  r2_dose < 0.5 with sv1_share > 0.8 -> a single direction with dose-free
               magnitudes explains the level means better than the declared numbers do

Also reported: whether the offset magnitude correlates with the level's position in the file
(row order), which is what a preparation/acquisition block would leave behind.
"""
import json
import os
import sys
import numpy as np

import p5_public_data as PP                                   # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp44_level_block_alias.json"


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def analyse(name):
    d = PP.load(name)
    X, y = d["X"], np.asarray(d["y"], float)
    levels = np.unique(y)
    if levels[0] != 0 or len(levels) < 4:
        return dict(name=name, note="needs an authentic level and >=4 levels")
    pos = {}
    for i, v in enumerate(y):
        pos.setdefault(round(float(v), 6), []).append(i)
    spans = {k: (min(v), max(v), len(v)) for k, v in sorted(pos.items())}
    nz = [k for k in sorted(spans) if k > 0]
    contiguous = all(spans[k][2] == spans[k][1] - spans[k][0] + 1 for k in spans)
    ordered = all(spans[nz[i]][0] < spans[nz[i + 1]][0] for i in range(len(nz) - 1))
    rec = dict(name=name, matrix=d["meta"]["matrix"], adulterant=d["meta"]["adulterant"],
               levels=[float(v) for v in levels[1:]], metrics={},
               rows=dict(each_level_one_interval=bool(contiguous),
                         intervals_ordered_by_level=bool(ordered),
                         spans={str(k): spans[k] for k in nz[:6]}))
    order_rank = np.argsort(np.argsort(levels[1:])).astype(float)
    for tag, Z in [("abs", X), ("snv", snv(X))]:
        mus = np.array([Z[y == L].mean(0) for L in levels])
        O = mus[1:] - mus[0]                      # (levels-1) x channels
        nrm = np.linalg.norm(O, axis=1)
        L = levels[1:]
        U, S, Vt = np.linalg.svd(O, full_matrices=False)
        sv1 = float(S[0] ** 2 / (S ** 2).sum())
        # rank-1 dose model through the origin: O ~ L * v
        v = (L[:, None] * O).sum(0) / (L ** 2).sum()
        res_dose = O - np.outer(L, v)
        r2_dose = float(1 - (res_dose ** 2).sum() / (O ** 2).sum())
        slopes = nrm / L
        slope_cv = float(slopes.std() / slopes.mean())
        cos_top = [float(np.dot(o, O[-1]) / (np.linalg.norm(o) * np.linalg.norm(O[-1]) + 1e-12))
                   for o in O]
        # does offset magnitude track position in the file rather than the declared number?
        from scipy.stats import spearmanr
        rho_row = float(spearmanr(order_rank, nrm).statistic)
        rec["metrics"][tag] = dict(
            norm_over_within=[round(float(a), 3) for a in nrm],
            sv1_share=round(sv1, 4), slope_cv=round(slope_cv, 3),
            r2_dose=round(r2_dose, 4),
            cos_to_top_level=[round(c, 3) for c in cos_top],
            min_cos_to_top=round(float(min(cos_top)), 3),
            spearman_rank_vs_norm=round(rho_row, 3),
            alias_flag=bool(sv1 > 0.8 and r2_dose < 0.5))
    sel = rec["metrics"]["abs"]
    rec["verdict"] = ("aliased" if sel["alias_flag"]
                      else "dose_linear" if sel["r2_dose"] >= 0.8 else "partly_aliased")
    return rec


if __name__ == "__main__":
    recs = {}
    for nm in PP.ALL:
        try:
            r = analyse(nm)
        except Exception as e:
            r = dict(name=nm, error="%s: %s" % (type(e).__name__, e))
        recs[nm] = r
        m = (r.get("metrics") or {}).get("abs", {})
        s = (r.get("metrics") or {}).get("snv", {})
        rw = r.get("rows") or {}
        print("%-4s %-13s/%-12s abs r2_dose=%s sv1=%s slope_cv=%s mincos=%s rows_contig=%s ord=%s -> %-13s | snv r2=%s" %
              (nm, r.get("matrix"), r.get("adulterant"), m.get("r2_dose"), m.get("sv1_share"),
               m.get("slope_cv"), m.get("min_cos_to_top"), rw.get("each_level_one_interval"),
               rw.get("intervals_ordered_by_level"), r.get("verdict"), s.get("r2_dose")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
