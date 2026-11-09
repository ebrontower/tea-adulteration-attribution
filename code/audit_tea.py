"""Structure audit of the tea dataset: are the 50 scans per formulation repeated
measurements of one sample, or do they contain several physical samples?

Writes out/audit_tea.json and out/audit_tea.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS
os.makedirs(OUT, exist_ok=True)
FP = (900.0, 1800.0)


def main():
    d = P.load_tea(verbose=True)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    X = np.asarray(d["X"], float)[:, sel]
    y = d["y"]
    b = d["y_batch"]
    names = P.TEA_RECIPE_NAMES
    res = {"n": int(len(y)), "n_chan": int(X.shape[1]), "recipes": {}, "batch_probe": {}}

    # Per formulation and batch: within-group structure of the 50 scans.
    for rc in sorted(set(y.tolist())):
        for bb in (1, 0):
            m = (y == rc) & (b == bb)
            if m.sum() == 0:
                continue
            Xc = X[m]
            # Correlation-distance clustering: reads out an effective sample count.
            Xn = (Xc - Xc.mean(1, keepdims=True))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
            D = pdist(Xn, metric="correlation")
            Dc = np.clip(np.nan_to_num(D, nan=0.0, posinf=2.0), 0, 2)
            Z = linkage(Dc, method="average")
            # Threshold: 3x the median nearest-neighbour distance (adaptive rather
            # than a hand-set absolute value).
            near = squareform(Dc)
            np.fill_diagonal(near, np.nan)
            nn = np.nanmedian(np.nanmin(near, axis=1))
            thr = float(max(1e-4, 3.0 * nn))
            lab = fcluster(Z, t=thr, criterion="distance")
            sizes = sorted(np.bincount(lab)[1:].tolist(), reverse=True)
            pc = PCA(n_components=5).fit(Xc)
            res["recipes"].setdefault(names[rc], {})[str(bb)] = dict(
                n=int(m.sum()), n_clusters=int(len(sizes)), sizes=sizes,
                nn_dist=float(nn), thr=thr,
                pc1_var=float(pc.explained_variance_ratio_[0]),
                pc5_var=float(pc.explained_variance_ratio_[:5].sum()),
                d12=float(np.linalg.norm(pc.transform(Xc)[:, :2].std(0))))

    # Batch detectability: 5-fold logistic probe, batch 1 vs batch 0 within a
    # formulation.
    shared = [c for c in sorted(set(y.tolist())) if ((y == c) & (b == 1)).any() and ((y == c) & (b == 0)).any()]
    accs, ntr = [], []
    for c in shared:
        m = (y == c)
        Xc, bc = X[m], b[m]
        sc = StandardScaler().fit(Xc)
        Xs = sc.transform(Xc)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = np.zeros_like(bc)
        for tr, te in skf.split(Xs, bc):
            mdl = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], bc[tr])
            pred[te] = mdl.predict(Xs[te])
        accs.append(balanced_accuracy_score(bc, pred)); ntr.append(int(m.sum()))
    res["batch_probe"] = dict(recipes=[names[c] for c in shared], bacc=accs,
                              bacc_mean=float(np.mean(accs)), n=ntr)
    # Pooled across formulations (within-formulation centring) to check for
    # residual batch signal.
    Xc_all, b_all = [], []
    for c in shared:
        m = (y == c)
        Xc_all.append(X[m] - X[m].mean(0, keepdims=True))
        b_all.append(b[m])
    Xc_all = np.vstack(Xc_all); b_all = np.concatenate(b_all)
    Xs = StandardScaler().fit_transform(Xc_all)
    pred = np.zeros_like(b_all)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, b_all):
        mdl = LogisticRegression(max_iter=3000).fit(Xs[tr], b_all[tr])
        pred[te] = mdl.predict(Xs[te])
    res["batch_probe"]["pooled_centered_bacc"] = float(balanced_accuracy_score(b_all, pred))

    # Batch centroid distance vs formulation centroid distance, on one scale.
    def centroid(mask):
        return X[mask].mean(0)
    same_batch_dist, diff_recipe_dist = [], []
    for c in shared:
        m1 = (y == c) & (b == 1)
        m0 = (y == c) & (b == 0)
        same_batch_dist.append(float(np.linalg.norm(centroid(m1) - centroid(m0))))
    for c1 in shared:
        for c2 in shared:
            if c2 <= c1:
                continue
            m1 = (y == c1) & (b == 1)
            m2 = (y == c2) & (b == 1)
            diff_recipe_dist.append(float(np.linalg.norm(centroid(m1) - centroid(m2))))
    res["centroid"] = dict(same_recipe_cross_batch=float(np.mean(same_batch_dist)),
                           diff_recipe_same_batch=float(np.mean(diff_recipe_dist)),
                           ratio=float(np.mean(same_batch_dist) / np.mean(diff_recipe_dist)))
    # Pairwise cosine between formulation centroids (within one batch).
    C = []
    for c in sorted(set(y.tolist())):
        m = (y == c) & (b == 1)
        if m.sum():
            C.append(centroid(m))
    C = np.array(C); Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    R = Cn @ Cn.T
    res["recipe_corr_min"] = float(R[np.triu_indices(len(C), 1)].min())
    res["recipe_corr_median"] = float(np.median(R[np.triu_indices(len(C), 1)]))
    res["cond_number"] = float(np.linalg.cond(C))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "audit_tea.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    L = ["# Tea IR data structure audit", "",
         f"- spectra {res['n']}, fingerprint channels {res['n_chan']} (900-1800 cm^-1)", "",
         "## Cluster structure per formulation and batch (correlation distance, "
         "threshold = 3x within-group nearest neighbour)", "",
         "| formulation | batch 1 n/clusters | batch 0 n/clusters | batch 1 sizes "
         "| batch 0 sizes | PC1 var | PC5 var |",
         "|---|---|---|---|---|---|---|"]
    for nm, v in res["recipes"].items():
        a, c = v.get("1"), v.get("0")
        L.append(f"| {nm} | {a['n'] if a else '-'}/{a['n_clusters'] if a else '-'} "
                 f"| {c['n'] if c else '-'}/{c['n_clusters'] if c else '-'} "
                 f"| {a['sizes'][:4] if a else '-'} | {c['sizes'][:4] if c else '-'} "
                 f"| {a['pc1_var'] if a else 0:.2f} | {a['pc5_var'] if a else 0:.3f} |")
    bp = res["batch_probe"]
    L += ["", "## Batch detectability (batch 1 vs batch 0 within a formulation)", "",
          f"- per-formulation bacc: {', '.join(f'{x:.3f}' for x in bp['bacc'])}",
          f"- mean {bp['bacc_mean']:.3f}; pooled after within-formulation centring "
          f"{bp['pooled_centered_bacc']:.3f}", "",
          "## Centroid scale", "",
          f"- same formulation, cross-batch {res['centroid']['same_recipe_cross_batch']:.3f}; "
          f"different formulation, same batch {res['centroid']['diff_recipe_same_batch']:.3f}; "
          f"ratio {res['centroid']['ratio']:.3f}",
          f"- formulation-centroid correlation min {res['recipe_corr_min']:.3f} / "
          f"median {res['recipe_corr_median']:.3f}; condition number "
          f"{res['cond_number']:.1f}"]
    with open(os.path.join(OUT, "audit_tea.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
