"""exp41 - headroom diagnostic for route (1): when the ONLY thing that changes between
training and deployment is the acquisition session, is there representational headroom?

Two deposited corpora carry the sample-level replication the tea set does not:
  qib_meats  60 physical samples x RunA/RunB, 3 meat classes     (Al-Jowder 1997, Food Chem 59:195)
  qib_olive  60 physical oils x two sessions, 4 countries         (Tapp 2003, J.AFC 51:6110)

Protocols (group = physical sample; no sample ever appears on both sides of a split except
in cross_same, where sharing samples is the point - it isolates session shift from novelty):
  cross_same   fit ALL of domain A, test ALL of domain B              -> pure session shift
  cross_new    fit A on half the samples, test B on the other half    -> shift + unseen samples
  in_src       fit/test inside A (sample-level split)                 -> source-domain ceiling
  tgt_sup_full fit 60% of B's samples, test the rest                  -> oracle with target labels
  tgt_sup_k    fit k labels/class inside B's 60%                      -> label-efficient ceiling

Nuisance arms use no target labels (SNV / 1st derivative / MSC / CORAL). If they close the
oracle-minus-transfer gap, the shift is a fixed metric problem and route (1) is dead; if only
target-supervised models close it, a transferable learned representation has a real job.

Geometry, reported in the attribution line's own language:
  shift_over_within      ||mu_A - mu_B|| / mean within-class scatter, whitened space
  abs_cos_shift_label    |cos| between domain-difference and class-difference directions
  subspace_kept          target class-mean separation retained inside source top-k PCs
  pair_residual_class_r2 R^2 of the paired per-sample difference (x_B - x_A) on class dummies
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qib_data as Q
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp41_transfer.json"
SEED = 20261101
K_W = 40
BUDGETS = [5, 10, 20]
MODELS = ["plsda10", "pca20_lda", "pca20_rbf", "linearsvc_full"]


# ------------------------------------------------------------------ loaders

def get(name):
    if name == "qib_meats":
        axis, X, df = Q.meats()
        axis, X = Q.window(axis, X, 1000, 1800)
        dom, lab, unit = df["run"].to_numpy(), df["label"].to_numpy(), "abs"
    elif name == "qib_olive":
        axis, X, df = Q.olive()
        axis, X = Q.window(axis, X, 900, 1800)
        dom = np.where(df["rep"].to_numpy() == 0, "session1", "session2")
        lab, unit = df["label"].to_numpy(), "abs"
    else:
        raise KeyError(name)
    X = Q.to_absorbance(X, unit)
    return dict(name=name, wvn=axis, X=X, dom=dom, y=lab, unit=unit,
                phys=df["phys"].to_numpy(), n_channel=int(len(axis)))


# ------------------------------------------------------------------ nuisance arms

def arm_matrix(Zs, Zt):
    """(label, source, target). Every statistic is estimated on the source except CORAL,
    which is transductive and uses unlabeled target spectra only."""
    sn = Q.snv(Zs), Q.snv(Zt)
    d1 = savgol_filter(sn[0], 15, 3, deriv=1, axis=1), savgol_filter(sn[1], 15, 3, deriv=1, axis=1)
    return [("raw", Zs, Zt),
            ("snv", sn[0], sn[1]),
            ("snv_d1", d1[0], d1[1]),
            ("msc", Q.msc(Zs), Q.msc(Zt, Zs.mean(0))),
            ("coral_snv", sn[0], Q.coral(sn[1], sn[0])),
            ("coral_d1", d1[0], Q.coral(d1[1], d1[0]))]


# ------------------------------------------------------------------ models

def sample_budget(Z, y, budget, rs):
    cls = np.unique(y)
    idx = []
    for c in cls:
        w = np.where(y == c)[0]
        w = w.copy()
        rs.shuffle(w)
        idx.append(w[:budget])
    idx = np.sort(np.concatenate(idx))
    return Z[idx], y[idx]


def fit_predict(Zs, ys, Zt, rs=None):
    cls = np.unique(ys)
    if len(cls) < 2 or len(Zs) <= len(cls):
        return {m: None for m in MODELS}
    res = {}
    Y = np.zeros((len(ys), len(cls)))
    Y[np.arange(len(ys)), [int(np.where(cls == c)[0][0]) for c in ys]] = 1.0
    pls = PLSRegression(n_components=int(min(10, Zs.shape[1], max(2, len(cls) * 3))))
    pls.fit(Zs, Y)
    res["plsda10"] = cls[np.argmax(pls.predict(Zt), axis=1)]
    nc = int(min(20, Zs.shape[1], len(Zs) - 1))
    p = PCA(n_components=nc, random_state=0).fit(Zs)
    a, b = p.transform(Zs), p.transform(Zt)
    res["pca20_lda"] = LinearDiscriminantAnalysis().fit(a, ys).predict(b)
    try:
        res["pca20_rbf"] = SVC(kernel="rbf", C=10.0, gamma="scale").fit(a, ys).predict(b)
    except Exception:
        res["pca20_rbf"] = None
    try:
        res["linearsvc_full"] = LinearSVC(C=1.0, max_iter=40000).fit(Zs, ys).predict(Zt)
    except Exception:
        res["linearsvc_full"] = None
    return res


def acc(pred, truth):
    if pred is None or len(np.unique(truth)) == 0:
        return None
    return float(np.mean(np.asarray(pred) == np.asarray(truth)))


def scores(Zs, ys, Zt, yt, rs, budget=None):
    if budget:
        Zs, ys = sample_budget(Zs, ys, budget, rs)
    p = fit_predict(Zs, ys, Zt, rs=rs)
    return {m: (round(acc(p[m], yt), 4) if acc(p[m], yt) is not None else None) for m in MODELS}


# ------------------------------------------------------------------ splits

def phys_split(y, phys, mask, frac, rs):
    """Sample-level split inside one domain, stratified by class. Returns row-index arrays."""
    idx_dom = np.where(mask)[0]
    rows = pd.DataFrame(dict(row=idx_dom, y=y[idx_dom], phys=phys[idx_dom]))
    keep = []
    for c in rows["y"].unique():
        ps = np.sort(rows.loc[rows["y"] == c, "phys"].unique())
        rs.shuffle(ps)
        k = max(1, int(np.ceil(frac * len(ps))))
        keep += list(ps[:k])
    trm = rows["phys"].isin(keep).to_numpy()
    return rows.loc[trm, "row"].to_numpy(), rows.loc[~trm, "row"].to_numpy()


# ------------------------------------------------------------------ geometry

def geo(d, mA, mB):
    iA, iB = np.where(mA)[0], np.where(mB)[0]
    Zs, Zt = Q.snv(d["X"][iA]), Q.snv(d["X"][iB])
    ys, yt = d["y"][iA], d["y"][iB]
    classes = np.unique(ys)
    W = np.zeros((Zs.shape[1], Zs.shape[1]))
    dof = 0
    for c in classes:
        m = Zs[ys == c]
        W += (m - m.mean(0)).T @ (m - m.mean(0))
        dof += len(m) - 1
    W /= max(dof, 1)
    w, V = np.linalg.eigh(W + np.eye(Zs.shape[1]) * 1e-8)
    T = (V * np.clip(w, 1e-8, None) ** -0.5) @ V.T
    A, B = Zs @ T, Zt @ T
    shift = B.mean(0) - A.mean(0)
    cd = np.stack([A[ys == c].mean(0) - A.mean(0) for c in classes])
    cos = np.abs(cd @ shift / (np.linalg.norm(cd, axis=1) * np.linalg.norm(shift) + 1e-12))
    within = np.mean([np.linalg.norm(A[ys == c] - A[ys == c].mean(0), axis=1).mean() for c in classes])
    nc = int(min(K_W, Zs.shape[1], len(Zs) - 1))
    p = PCA(n_components=nc, random_state=0).fit(Zs)
    full = p.inverse_transform(p.transform(Zt))
    keep = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            raw = Zt[yt == classes[i]].mean(0) - Zt[yt == classes[j]].mean(0)
            proj = full[yt == classes[i]].mean(0) - full[yt == classes[j]].mean(0)
            keep.append(float(np.linalg.norm(proj) / (np.linalg.norm(raw) + 1e-12)))
    left = pd.DataFrame(dict(row=iA, phys=d["phys"][iA]))
    right = pd.DataFrame(dict(row=iB, phys=d["phys"][iB]))
    mg = left.merge(right, on="phys", suffixes=("_s", "_t"))
    r2 = None
    if len(mg) > 2 * len(classes):
        pA = {int(r): j for j, r in enumerate(iA)}
        pB = {int(r): j for j, r in enumerate(iB)}
        rs_rows = np.array([pA[int(r)] for r in mg["row_s"]])
        rt_rows = np.array([pB[int(r)] for r in mg["row_t"]])
        D = Zt[rt_rows] - Zs[rs_rows]
        Y = np.zeros((len(mg), len(classes)))
        Y[np.arange(len(mg)), [int(np.where(classes == c)[0][0]) for c in
                               d["y"][mg["row_s"].to_numpy()]]] = 1.0
        G = np.column_stack([np.ones(len(mg)), Y[:, 1:]])
        beta, *_ = np.linalg.lstsq(G, D, rcond=None)
        ss_res = ((D - G @ beta) ** 2).sum()
        ss_tot = ((D - D.mean(0)) ** 2).sum()
        r2 = float(1 - ss_res / ss_tot)
    return dict(n_pair=int(len(mg)), shift_norm_whit=round(float(np.linalg.norm(shift)), 3),
                within_norm=round(float(within), 3),
                shift_over_within=round(float(np.linalg.norm(shift) / within), 3),
                abs_cos_shift_label=[round(float(x), 4) for x in cos],
                max_abs_cos_shift_label=round(float(cos.max()), 4),
                subspace_kept=round(float(np.mean(keep)), 4),
                pair_residual_class_r2=round(r2, 4) if r2 is not None else None)


# ------------------------------------------------------------------ main

def run_corpus(name, frac=0.6):
    d = get(name)
    doms = sorted(set(d["dom"]))
    A, B = doms
    mA, mB = d["dom"] == A, d["dom"] == B
    iA, iB = np.where(mA)[0], np.where(mB)[0]
    rec = dict(name=name, unit=d["unit"], n_spectra=int(len(d["y"])), n_channel=d["n_channel"],
               window=[float(d["wvn"][0]), float(d["wvn"][-1])],
               classes=[str(c) for c in np.unique(d["y"])],
               class_counts={str(c): int((d["y"] == c).sum()) for c in np.unique(d["y"])},
               domains=[A, B], n_per_domain=[int(len(iA)), int(len(iB))],
               n_phys=int(len(np.unique(d["phys"]))),
               geometry_A_to_B=geo(d, mA, mB), geometry_B_to_A=geo(d, mB, mA),
               arms=[])

    # sample groups shared / disjoint between domains
    pA = set(d["phys"][iA])
    pB = set(d["phys"][iB])
    shared = sorted(pA & pB)
    disjoint_B = sorted(pB - pA)
    rec["overlap"] = dict(shared_phys=len(shared), b_only_phys=len(disjoint_B))

    for lab, Zs_all, Zt_all in arm_matrix(d["X"][iA], d["X"][iB]):
        rs = np.random.RandomState(SEED)
        posA = {int(r): j for j, r in enumerate(iA)}
        posB = {int(r): j for j, r in enumerate(iB)}

        def take(Z, rows, pos):
            return Z[np.array([pos[int(r)] for r in rows], dtype=int)]

        trA, vaA = phys_split(d["y"], d["phys"], mA, 0.6, rs)
        trB, vaB = phys_split(d["y"], d["phys"], mB, 0.6, np.random.RandomState(SEED + 1))
        # cross_new: fit on some physical samples, test the session-shifted spectra of the rest
        src_pool = trA
        out_phys = set(np.unique(d["phys"][iB])) - set(np.unique(d["phys"][src_pool]))
        tgt_idx = np.array([r for r in iB if d["phys"][r] in out_phys])
        if len(tgt_idx) == 0:                      # fully paired corpora: split samples instead
            half = np.unique(d["phys"][src_pool]).copy()
            np.random.RandomState(SEED + 2).shuffle(half)
            keep = set(half[:len(half) // 2])
            src_idx = np.array([r for r in iA if d["phys"][r] in keep])
            tgt_idx = np.array([r for r in iB if d["phys"][r] not in keep])
        else:
            src_idx = src_pool
        row = dict(prep=lab,
                   in_src=scores(take(Zs_all, trA, posA), d["y"][trA],
                                 take(Zs_all, vaA, posA), d["y"][vaA], rs),
                   cross_same=scores(Zs_all, d["y"][iA], Zt_all, d["y"][iB], rs),
                   cross_new=scores(take(Zs_all, src_idx, posA), d["y"][src_idx],
                                    take(Zt_all, tgt_idx, posB), d["y"][tgt_idx], rs)
                   if len(src_idx) > len(np.unique(d["y"])) else None,
                   tgt_sup_full=scores(take(Zt_all, trB, posB), d["y"][trB],
                                       take(Zt_all, vaB, posB), d["y"][vaB], rs),
                   tgt_sup_k={})
        row["cross_new_n"] = [int(len(src_idx)), int(len(tgt_idx))]
        for k in BUDGETS:
            if len(trB) >= 3 * k:
                row["tgt_sup_k"][str(k)] = scores(take(Zt_all, trB, posB), d["y"][trB],
                                                  take(Zt_all, vaB, posB), d["y"][vaB],
                                                  np.random.RandomState(SEED + 7 + k), budget=k)
        rec["arms"].append(row)
    return rec


if __name__ == "__main__":
    out = {}
    for nm in ["qib_meats", "qib_olive"]:
        r = run_corpus(nm)
        out[nm] = r
        print("== %s  n=%d  classes=%s  domains=%s(%d/%d)  shared_phys=%d" %
              (nm, r["n_spectra"], r["classes"], r["domains"],
               r["n_per_domain"][0], r["n_per_domain"][1], r["overlap"]["shared_phys"]))
        g = r["geometry_A_to_B"]
        print("   geom A->B: shift/within=%.2f maxcos=%.3f kept=%.3f pairR2=%s npair=%d" %
              (g["shift_over_within"], g["max_abs_cos_shift_label"], g["subspace_kept"],
               g["pair_residual_class_r2"], g["n_pair"]))
        for a in r["arms"]:
            cs, tf = a["cross_same"], a["tgt_sup_full"]
            gaps = ["%s:%s->%s" % (m, cs[m], tf[m]) for m in MODELS]
            print("   %-9s cross_same %s | tgt_sup %s" % (a["prep"],
                  {m: cs[m] for m in MODELS}, {m: tf[m] for m in MODELS}))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
