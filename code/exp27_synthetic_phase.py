"""exp27 - E4: phase diagram of SSL gain on synthetic smooth low-dimensional spectra.

Real datasets give one point each; a theory needs the map. Generate spectra from a d-dimensional
latent manifold with a smoothness knob and two label types:

  linear     label = quantile bin of a random projection of z   (decodable from the top PCs)
  radial     label = quantile bin of |z|^2                      (not linearly decodable)

Arms: MAE pretraining (same architecture and epochs as the real-data runs), random init,
PC20 + linear probe, PC20 + RBF-SVM. Report balanced accuracy at fixed label budgets and
delta(MAE - best baseline) over the (d, smoothness, label type) grid.
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
N, PATCH, EPOCHS = 768, 16, 40
OUT = p5_paths.RESULTS + "/exp27_synthetic_phase.json"


def make_data(d, decay, label, n=600, seed=0):
    rng = np.random.default_rng(seed)
    nu = np.linspace(0, 1, N)
    K = 2 * d
    freqs = np.arange(1, K + 1)
    basis = np.stack([np.sin(np.pi * f * nu) for f in freqs], 1)       # (N, K)
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    amp = decay ** (freqs - 1)
    z = rng.normal(size=(n, K))
    X = (z * amp) @ basis.T
    X += rng.normal(scale=0.02, size=X.shape)
    if label == "linear":
        w = rng.normal(size=K) * amp
        s = z @ w
    else:                                                              # radial
        s = np.sum((z * amp) ** 2, 1)
    q = np.quantile(s, [0.25, 0.5, 0.75])
    y = np.digitize(s, q)
    return X.astype(float), y


def split(y, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    tr = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        tr += list(idx[: int(frac * len(idx))])
    tr = np.array(sorted(tr))
    te = np.setdiff1d(np.arange(len(y)), tr)
    return tr, te


def probe(F, y, tr, te, b, kind="linear", seed=0):
    rng = np.random.default_rng(seed)
    sub = []
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c].copy()
        rng.shuffle(idx)
        sub += list(idx[:b])
    sc = StandardScaler().fit(F[sub])
    clf = (SVC(kernel="rbf", C=10.0) if kind == "svm"
           else LogisticRegression(max_iter=2000))
    clf.fit(sc.transform(F[sub]), y[sub])
    return balanced_accuracy_score(y[te], clf.predict(sc.transform(F[te])))


def embed(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X, dtype=torch.float32, device=DEV)).cpu().numpy()


def run(d, decay, label, seed=0):
    X, y = make_data(d, decay, label, seed=seed)
    tr, te = split(y, seed=seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    _, model = GT.train_mae1d(Xs[tr], patch=PATCH, ratio=0.5, seed=seed, epochs=EPOCHS,
                              device=DEV, return_model=True)
    # encoder embedding (mean-pooled tokens)
    xb = torch.tensor(Xs, dtype=torch.float32, device=DEV)
    T = xb.shape[1] // PATCH
    with torch.no_grad():
        tok = model.embed(xb[:, :T * PATCH].reshape(len(xb), T, PATCH)) + model.pos[:, :T]
        Zm = model.encoder(tok).mean(1).cpu().numpy()

    kp = min(20, len(tr) - 4, Xs.shape[1])
    Pc = PCA(n_components=kp, random_state=0).fit(Xs[tr]).transform(Xs)

    res = {"d": d, "decay": decay, "label": label}
    for b in (5, 10, 20):
        mae = np.mean([probe(Zm, y, tr, te, b, seed=s) for s in range(3)])
        pclin = np.mean([probe(Pc, y, tr, te, b, seed=s) for s in range(3)])
        pcsvm = np.mean([probe(Pc, y, tr, te, b, "svm", seed=s) for s in range(3)])
        best = max(pclin, pcsvm)
        res[str(b)] = dict(mae=round(float(mae), 3), pc_lin=round(float(pclin), 3),
                           pc_svm=round(float(pcsvm), 3),
                           delta=round(float(mae - best), 3))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    grid_d = [2, 4, 8, 16]
    grid_decay = [0.9, 0.6, 0.3]
    out = []
    for lab in ("linear", "radial"):
        for d in grid_d:
            for dec in grid_decay:
                out.append(run(d, dec, lab))
                json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
