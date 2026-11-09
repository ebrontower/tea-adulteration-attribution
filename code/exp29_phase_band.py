"""exp29 - E4 (redesign): phase diagram of SSL gain vs. label energy outside the smooth subspace.

The 8 real datasets are 8 points and cannot support a law. This builds the map with a controlled
knob that real data does not offer: where, in latent order, the label lives.

Generator
    x_i = sum_{k=1..12} z_ik b_k + noise,   b_k = normalised sin(pi k nu)   (smooth, ordered)
    label s_i uses only latent components k in a band [lo, lo+1]
        linear : s = z . w,  w unit-support on the band      (linearly decodable in principle)
        radial : s = sum_{k in band} z_k^2                   (not linearly decodable)
    4 classes = quantile bins of s.

Sweeping lo pushes the label out of the top-4 PC subspace while leaving the spectrum itself
unchanged. For each setting we measure
    rho_out = 1 - R^2(s ~ top-4 PC scores)     label energy outside the reconstruction subspace
    gain    = bacc(MAE probe) - max(bacc(PC4 linear), bacc(PC4 RBF))   at a fixed label budget

Law under test: gain <= 0 whenever rho_out ~ 0, and gain rises with rho_out only where the
pretext actually reconstructs the band (i.e. the band is inside the model's smoothness prior).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import g0_torch as GT  # noqa: E402
import p5_paths  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, PATCH, EPOCHS, K = 1024, 16, 60, 12
NOISE = 0.005
OUT = p5_paths.RESULTS + "/exp29_phase_band.json"


def basis(N=N, K=K):
    nu = np.linspace(0, 1, N)
    B = np.stack([np.sin(np.pi * (k + 1) * nu) for k in range(K)], 1)
    return B / np.linalg.norm(B, axis=0, keepdims=True)


def make(lo, kind, n=900, seed=0, bandwidth=2):
    rng = np.random.default_rng(seed)
    B = basis()
    z = rng.normal(size=(n, K))
    amp = 1.0 / np.sqrt(np.arange(1, K + 1))
    X = (z * amp) @ B.T + rng.normal(scale=NOISE, size=(n, N))
    band = np.arange(lo, lo + bandwidth)
    zw = z * amp                                         # label uses the amplitude-weighted factor
    if kind == "linear":
        w = np.zeros(K)
        w[band] = rng.normal(size=bandwidth)
        w /= np.linalg.norm(w)
        s = zw @ w
    else:
        s = (zw[:, band] ** 2).sum(1)
    q = np.quantile(s, [0.25, 0.5, 0.75])
    y = np.digitize(s, q)
    return X.astype(float), y, s


def split(y, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    tr = []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        tr += list(idx[: int(frac * len(idx))])
    tr = np.array(sorted(tr))
    return tr, np.setdiff1d(np.arange(len(y)), tr)


def probe(F, y, tr, te, b, kind="linear", seed=0):
    rng = np.random.default_rng(seed)
    sub = []
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c].copy()
        rng.shuffle(idx)
        sub += list(idx[:b])
    sub = np.array(sub)
    sc = StandardScaler().fit(F[sub])
    clf = SVC(kernel="rbf", C=10.0) if kind == "svm" else LogisticRegression(max_iter=3000)
    clf.fit(sc.transform(F[sub]), y[sub])
    return balanced_accuracy_score(y[te], clf.predict(sc.transform(F[te])))


def embed(model, X):
    xb = torch.tensor(X, dtype=torch.float32, device=DEV)
    T = xb.shape[1] // PATCH
    with torch.no_grad():
        tok = model.embed(xb[:, :T * PATCH].reshape(len(xb), T, PATCH)) + model.pos[:, :T]
        return model.encoder(tok).mean(1).cpu().numpy()


def rho_out(s, P):
    """1 - R^2 of a linear model s ~ PCs: the label energy outside the PC subspace."""
    A = np.hstack([np.ones((len(P), 1)), P])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    res = float(np.sum((s - A @ coef) ** 2))
    tot = float(np.sum((s - s.mean()) ** 2))
    return res / tot


def run(lo, kind, kpc=4, budgets=(5, 10, 20), seeds=(0, 1, 2)):
    res = {"lo": lo, "label": kind, "kpc": kpc}
    mae_curves, pc_curves, svm_curves, rhos = [], [], [], []
    for sd in seeds:
        X, y, s = make(lo, kind, seed=sd)
        tr, te = split(y, seed=sd)
        mu, sdv = X[tr].mean(0), float(X[tr].std())
        Xs = (X - mu) / sdv
        _, model = GT.train_mae1d(Xs[tr], patch=PATCH, ratio=0.5, seed=sd, epochs=EPOCHS,
                                  device=DEV, return_model=True)
        Zm = embed(model, Xs)
        pc = PCA(n_components=kpc, random_state=0).fit(Xs[tr])
        P = pc.transform(Xs)
        rhos.append(rho_out(s, P))
        for key, F, kin in (("mae", Zm, "linear"), ("pc", P, "linear"), ("svm", P, "svm")):
            vals = {b: probe(F, y, tr, te, b, kin, sd) for b in budgets}
            {"mae": mae_curves, "pc": pc_curves, "svm": svm_curves}[key].append(vals)
    for b in budgets:
        m = float(np.mean([v[b] for v in mae_curves]))
        p = float(np.mean([v[b] for v in pc_curves]))
        q = float(np.mean([v[b] for v in svm_curves]))
        res[str(b)] = dict(mae=round(m, 3), pc_lin=round(p, 3), pc_svm=round(q, 3),
                           gain=round(m - max(p, q), 3))
    res["rho_out"] = round(float(np.mean(rhos)), 3)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    out = []
    seeds = (0, 1, 2)
    for kind in ("linear", "radial"):
        for lo in (1, 3, 5, 7, 9):
            out.append(run(lo, kind, seeds=seeds))
            json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
