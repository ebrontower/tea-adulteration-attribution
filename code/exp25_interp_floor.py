"""exp25 - E1: is the masked-MAE representation anything more than a low-rank projection?

For each dataset: pretrain MAE1D unsupervised on the training split (same arch/epochs as the
G0 gate), embed every spectrum with the encoder, then ask three questions.

1. Linear equivalence: how many principal components are needed to linearly reproduce the
   128-d MAE embedding (ridge, 5-fold CV, R^2 per dimension)?  If k* is small, the encoder
   has discovered nothing beyond a k*-dimensional linear projection.
2. Task equivalence: frozen-feature logistic probe on MAE embedding vs on the top-k PCs.
   If the two agree, the pretraining contributes no task information.
3. Interpolation floor: reconstruction error of the trained MAE at masked positions vs the
   best smooth completion from visible context only (cubic / PCA-k / DCT).  MAE >= floor
   means the objective is satisfied by interpolation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import exp01_g0_interpolation as E1  # noqa: E402
import g0_torch as GT  # noqa: E402
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PATCH, DIM, DEPTH, EPOCHS = 16, 128, 4, 60
OUT = p5_paths.RESULTS + "/exp25_interp_floor.json"


def embed(model, X):
    """mean-pooled encoder output over all tokens (no mask: full-context embedding)."""
    xb = torch.tensor(X, dtype=torch.float32, device=DEV)
    B, N = xb.shape
    p = model.patch
    T = N // p
    with torch.no_grad():
        tok = model.embed(xb[:, :T * p].reshape(B, T, p)) + model.pos[:, :T]
        h = model.encoder(tok)
    return h.mean(1).cpu().numpy()


def ridge_r2(A, B, n_splits=5):
    """mean CV R^2 when predicting each column of B from A (both standardised)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    y = np.zeros(len(A), int)
    preds = np.zeros_like(B)
    for tr, te in cv.split(A, y):
        m = Ridge(alpha=1.0).fit(A[tr], B[tr])
        preds[te] = m.predict(A[te])
    ss_res = ((B - preds) ** 2).sum(0)
    ss_tot = ((B - B.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.clip(ss_tot, 1e-12, None)
    return float(np.mean(r2)), float(np.min(r2))


def probe(F, y, n_splits=5):
    F = StandardScaler().fit_transform(F)
    cv = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())), shuffle=True,
                         random_state=0)
    sc = []
    for tr, te in cv.split(F, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(F[tr], y[tr])
        sc.append(balanced_accuracy_score(y[te], clf.predict(F[te])))
    return float(np.mean(sc))


def run_one(name, verbose=True):
    if name == "TEA":
        d = P.load_tea(verbose=False)
        X = d["X"][d["y_batch"] == 1]
        y = d["y"][d["y_batch"] == 1]
        groups = X.shape[0]
        idx = np.arange(len(X))
        rng = np.random.default_rng(42)
        tr = []
        for c in np.unique(y):
            ii = idx[y == c]
            rng.shuffle(ii)
            tr += list(ii[: int(0.8 * len(ii))])
        te = np.setdiff1d(idx, tr)
    else:
        d = E1.prep_dataset(name, atr_correct=True)
        X = np.asarray(d["X"], float)
        y = np.asarray(d["y_main"])
        tr, te = np.asarray(d["idx_tr"]), np.asarray(d["idx_te"])
        if y.dtype.kind == "f":
            y = y.astype(int)
    mu, sd = X[tr].mean(0), np.clip(X[tr].std(0), 1e-8, None)
    Xs = (X - mu) / sd
    Xtr = Xs[tr]
    X = Xs

    _, model = GT.train_mae1d(Xtr, patch=PATCH, ratio=0.5, seed=0, epochs=EPOCHS,
                              device=DEV, verbose=False, return_model=True)
    Z = embed(model, X)

    # PCA basis on the same (unsupervised) data
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)

    res = {"n": int(len(X)), "n_channel": int(X.shape[1]), "n_class": int(len(np.unique(y))),
           "pc1_share": float(ev[0])}
    # 1. how much of the embedding is a linear function of the top-k PC scores
    Zc = Z - Z.mean(0)
    sst = float(np.sum(Zc ** 2))
    for k in (5, 10, 20, 50, 100):
        if k >= len(X):
            continue
        A = U[:, :k] * S[:k]
        A1 = np.hstack([np.ones((len(A), 1)), A])
        coef, *_ = np.linalg.lstsq(A1, Zc, rcond=None)
        res[f"lin_r2_pc{k}"] = round(1 - float(np.sum((Zc - A1 @ coef) ** 2)) / sst, 4)
    # reverse: how much of the leading PCs the embedding spans
    kbest = 20
    Pk = U[:, :kbest] * S[:kbest]
    m, _ = ridge_r2(Z, Pk)
    res[f"r2_pc{kbest}_from_mae"] = round(m, 4)

    # 2. task probe
    res["probe_mae"] = round(probe(Z, y), 3)
    for k in (5, 10, 20, 50):
        if k >= len(X):
            continue
        res[f"probe_pc{k}"] = round(probe(Pk_all := (U[:, :k] * S[:k]), y), 3)

    # 3. reconstruction at masked channels: trained MAE vs low-rank (PCA-k) completion
    rng = np.random.default_rng(0)
    N = X.shape[1]
    xin = X[te][:, : (N // PATCH) * PATCH]
    mu_c = X.mean(0)[None, : xin.shape[1]]
    mae_errs, pca_errs = {5: [], 10: [], 20: []}, {5: [], 10: [], 20: []}
    for _ in range(10):
        m = GT.make_mask_np(N, PATCH, 0.5, rng)
        mm = m[: (N // PATCH) * PATCH]
        with torch.no_grad():
            pred = model(torch.tensor(xin, dtype=torch.float32, device=DEV),
                         torch.tensor(np.broadcast_to(mm, xin.shape), dtype=torch.bool,
                                      device=DEV))[0]
        pred = pred.cpu().numpy()
        e = float(np.mean((pred - xin)[:, mm] ** 2))
        for k in mae_errs:
            mae_errs[k].append(e)
            Vk = Vt[:k, :xin.shape[1]].T
            Bv = np.hstack([np.ones((int((~mm).sum()), 1)), Vk[~mm]])
            Bm = np.hstack([np.ones((int(mm.sum()), 1)), Vk[mm]])
            coef, *_ = np.linalg.lstsq(Bv, (xin - mu_c)[:, ~mm].T, rcond=None)
            rec = Bm @ coef + mu_c[0, mm][:, None]          # (n_masked, n_spectra)
            pca_errs[k].append(float(np.mean((rec - xin[:, mm].T) ** 2)))
    for k in mae_errs:
        res[f"mae_over_pca{k}"] = round(float(np.median(
            np.array(mae_errs[k]) / np.array(pca_errs[k]))), 3)

    if verbose:
        print(name, json.dumps({k: v for k, v in res.items()}, ensure_ascii=False), flush=True)
    return res


def main():
    names = sys.argv[1:] or ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8", "TEA"]
    out = {}
    for nm in names:
        try:
            out[nm] = run_one(nm)
        except Exception as e:  # noqa: BLE001
            out[nm] = {"error": repr(e)}
            print(nm, "FAILED", e, flush=True)
            json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
