"""exp28 - E3: where does the label live? A computable criterion for "SSL cannot help".

E1 showed the MAE embedding is 75-97 % a linear function of 5 PC scores and that a probe on 20
PCs matches or beats it on 7 of 8 datasets. E2 asks whether a different pretext escapes that.
E3 gives the reason one way or the other: if the class information is contained in the smooth
(low-rank) subspace that reconstruction is built to model, then *no* pretext that acts on that
subspace can add information, and the ceiling is set by the label, not by the pretraining.

Per dataset, all at the same split and the same label budget:
    pc20_lin    logistic probe on top-20 PCs              (smooth-subspace reference)
    pc20_svm    RBF-SVM on top-20 PCs                     (nonlinear head, same subspace)
    full_svm    RBF-SVM on all channels                   (upper bound with a flexible head)
    resid_svm   RBF-SVM on the PC-20 residual only        (what lives *outside* the subspace)
    coverage    share of between-class scatter inside the top-20 PC subspace

Criterion: if resid_svm ~ chance and coverage ~ 1, the label is already inside the subspace that
reconstruction models, so pretext choice is irrelevant; any observed SSL gain must come from
regularisation, not from representation.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import exp01_g0_interpolation as E1  # noqa: E402
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

BUDGETS = (10, 20)
SEEDS = (0, 1, 2)
BUDGET_FULL = 10_000
OUT = p5_paths.RESULTS + "/exp28_label_subspace.json"


def load(name):
    if name == "TEA":
        d = P.load_tea(verbose=False)
        X = d["X"][d["y_batch"] == 1]
        y = d["y"][d["y_batch"] == 1]
        idx = np.arange(len(X))
        rng = np.random.default_rng(42)
        tr = []
        for c in np.unique(y):
            ii = idx[y == c].copy()
            rng.shuffle(ii)
            tr += list(ii[: int(0.8 * len(ii))])
        tr = np.array(sorted(tr))
        te = np.setdiff1d(idx, tr)
    else:
        d = E1.prep_dataset(name, atr_correct=True)
        X, y = np.asarray(d["X"], float), np.asarray(d["y_main"])
        if y.dtype.kind == "f":
            y = y.astype(int)
        tr, te = np.asarray(d["idx_tr"]), np.asarray(d["idx_te"])
    mu, sd = X[tr].mean(0), np.clip(X[tr].std(0), 1e-8, None)
    return (X - mu) / sd, y, tr, te


def probe(F, y, tr, te, n_per_class, seed, kind="linear"):
    rng = np.random.default_rng(seed)
    sub = []
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c].copy()
        rng.shuffle(idx)
        sub += list(idx[:min(n_per_class, len(idx))])
    sub = np.array(sub)
    sc = StandardScaler().fit(F[sub])
    clf = SVC(kernel="rbf", C=10.0, gamma="scale") if kind == "svm" else LogisticRegression(max_iter=3000)
    clf.fit(sc.transform(F[sub]), y[sub])
    return balanced_accuracy_score(y[te], clf.predict(sc.transform(F[te])))


def mean_probe(F, y, tr, te, budgets, kind="linear"):
    out = {}
    for b in budgets:
        vals = [probe(F, y, tr, te, b, s, kind) for s in SEEDS]
        out[str(b)] = round(float(np.mean(vals)), 3)
    return out


def coverage(X, y, tr, k):
    """share of between-class scatter (trace Sb) carried by the top-k PC subspace of X[tr]."""
    pc = PCA(n_components=k, random_state=0).fit(X[tr])
    classes = np.unique(y[tr])
    mu = X[tr].mean(0)
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        m = X[tr][y[tr] == c].mean(0) - mu
        Sb += (y[tr] == c).sum() * np.outer(m, m)
    V = pc.components_.T                       # (p, k)
    tot = float(np.trace(Sb))
    sub = float(np.trace(V.T @ Sb @ V))
    return sub / tot if tot > 0 else float("nan")


def run_one(nm):
    t0 = time.time()
    X, y, tr, te = load(nm)
    ncls = int(len(np.unique(y)))
    res = {"n": int(len(X)), "n_channel": int(X.shape[1]), "n_class": ncls,
           "n_test": int(len(te)), "chance": round(1.0 / ncls, 3)}
    k = min(20, len(tr) - ncls - 1, X.shape[1])
    pc = PCA(n_components=k, random_state=0).fit(X[tr])
    Ptr, Pte = pc.transform(X[tr]), pc.transform(X[te])
    Ptr_full = pc.transform(X)
    # residual outside the smooth subspace (recompute from the same fitted PCA)
    Rtr = X[tr] - pc.inverse_transform(Ptr)
    Rte = X[te] - pc.inverse_transform(Pte)

    res["coverage_pc20"] = round(float(coverage(X, y, tr, k)), 4)
    res["pc20_lin"] = mean_probe(Ptr_full, y, tr, te, BUDGETS, "linear")
    res["pc20_svm"] = mean_probe(Ptr_full, y, tr, te, BUDGETS, "svm")
    res["full_svm"] = mean_probe(X, y, tr, te, BUDGETS, "svm")
    res["resid_svm"] = mean_probe(np.vstack([Rtr, Rte]),
                                  np.concatenate([y[tr], y[te]]), np.arange(len(tr)),
                                  np.arange(len(tr), len(tr) + len(te)), BUDGETS, "svm")
    # full-training reference (no label budget): does the residual carry anything at all?
    for tag, Ftr, Fte in (("pc20", Ptr, Pte), ("resid", Rtr, Rte), ("full", X[tr], X[te])):
        sc = StandardScaler().fit(Ftr)
        clf = (LinearDiscriminantAnalysis() if tag == "pc20"
               else SVC(kernel="rbf", C=10.0, gamma="scale"))
        clf.fit(sc.transform(Ftr), y[tr])
        res[f"{tag}_full"] = round(float(balanced_accuracy_score(y[te], clf.predict(sc.transform(Fte)))), 3)

    res["headroom_full"] = round(float(res["full_svm"]["20"] - max(res["pc20_lin"]["20"],
                                                                    res["pc20_svm"]["20"])), 3)
    res["resid_lift"] = round(float(res["resid_svm"]["20"] - res["chance"]), 3)
    res["secs"] = round(time.time() - t0, 1)
    print(nm, json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    names = sys.argv[1:] or ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8", "TEA"]
    try:
        out = json.load(open(OUT))
    except Exception:  # noqa: BLE001
        out = {}
    for nm in names:
        try:
            out[nm] = run_one(nm)
        except Exception as e:  # noqa: BLE001
            out[nm] = {"error": repr(e)}
            print(nm, "FAILED", e, flush=True)
        json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
