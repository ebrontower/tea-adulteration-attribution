"""exp26 - E2: non-reconstructive SSL (VICReg / Barlow) under the same protocol as the MAE arm.

The paper currently shows that *masked reconstruction* fails. The question a referee asks next
is whether SSL per se fails. Same splits, same encoder capacity, same label budgets; only the
pretext changes.

Arms
    vicreg    variance-invariance-covariance regularisation on two label-preserving views
    barlow    Barlow-Twins redundancy reduction
    randinit  same architecture, random weights, frozen
    pcalda    PCA(20) + LDA
    plsda     PLS-DA (10 LVs, one-hot)
Views: baseline offset/slope, multiplicative scatter, channel noise, small contiguous band
dropout, moving-average smoothing. No shift, flip or chemistry-changing transform.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import exp01_g0_interpolation as E1  # noqa: E402
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PATCH, DIM, DEPTH = 16, 128, 4
BUDGETS = (2, 5, 10, 20)
SEEDS = (0, 1, 2)
OUT = p5_paths.RESULTS + "/exp26_nonrecon_ssl.json"


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = PATCH
        self.embed = nn.Linear(PATCH, DIM)
        self.pos = nn.Parameter(torch.zeros(1, 4096, DIM))
        layer = nn.TransformerEncoderLayer(DIM, 4, 256, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, DEPTH)

    def forward(self, x):
        B, N = x.shape
        T = N // self.patch
        tok = self.embed(x[:, :T * self.patch].reshape(B, T, self.patch)) + self.pos[:, :T]
        return self.encoder(tok).mean(1)


class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, 256), nn.GELU(), nn.Linear(256, 256))

    def forward(self, z):
        return self.net(z)


def augment(x, rng):
    B, N = x.shape
    out = x.clone()
    dev = x.device
    t = torch.linspace(-1, 1, N, device=dev)
    for i in range(B):
        row = out[i]
        row = row + torch.randn((), device=dev) * 0.15 + t * torch.randn((), device=dev) * 0.10
        row = row * (1.0 + 0.10 * torch.randn((), device=dev))
        row = row + torch.randn(N, device=dev) * 0.03
        if torch.rand(()) < 0.5:
            w = int(N * (0.02 + 0.06 * float(torch.rand(()))))
            s = int(torch.randint(0, max(1, N - w), ()))
            row[s:s + w] = row.mean()
        if torch.rand(()) < 0.5:
            k = int(torch.randint(3, 9, ()))
            row = F.avg_pool1d(F.pad(row[None, None], (k // 2, k // 2), mode="replicate"),
                               k, 1).squeeze()[:N]
        out[i] = row
    return out


def off_diagonal(m):
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_ssl_encoder(X, kind, epochs=80, batch=64, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, proj = Enc().to(DEV), Proj().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()),
                            lr=lr, weight_decay=1e-4)
    xb_all = torch.tensor(X, dtype=torch.float32, device=DEV)
    n = len(X)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            xb = xb_all[perm[i:i + batch]]
            if len(xb) < 8:
                continue
            z1, z2 = proj(enc(augment(xb, rng))), proj(enc(augment(xb, rng)))
            if kind == "vicreg":
                sim = F.mse_loss(z1, z2)
                std1 = torch.sqrt(z1.var(0) + 1e-4)
                std2 = torch.sqrt(z2.var(0) + 1e-4)
                var = torch.relu(1 - std1).mean() + torch.relu(1 - std2).mean()
                c1 = (z1 - z1.mean(0)).T @ (z1 - z1.mean(0)) / (len(z1) - 1)
                c2 = (z2 - z2.mean(0)).T @ (z2 - z2.mean(0)) / (len(z2) - 1)
                cov = (off_diagonal(c1).pow(2).sum() + off_diagonal(c2).pow(2).sum()) / z1.shape[1]
                loss = 25 * sim + 25 * var + cov
            else:
                z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
                z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
                C = z1n.T @ z2n / len(z1)
                loss = ((torch.diagonal(C) - 1) ** 2).mean() \
                    + 0.04 * off_diagonal(C).pow(2).sum() / z1.shape[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    return enc


@torch.no_grad()
def embed(enc, X):
    return enc(torch.tensor(X, dtype=torch.float32, device=DEV)).cpu().numpy()


def probe(F, y, tr, te, n_per_class, seed, clf_kind="linear"):
    rng = np.random.default_rng(seed)
    sub = []
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c].copy()
        rng.shuffle(idx)
        sub += list(idx[:n_per_class])
    sub = np.array(sub)
    sc = StandardScaler().fit(F[sub])
    if clf_kind == "svm":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale")
    else:
        clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(sc.transform(F[sub]), y[sub])
    return balanced_accuracy_score(y[te], clf.predict(sc.transform(F[te])))


def plsda(Xtr, ytr, Xte, yte):
    classes = np.unique(ytr)
    Y = np.zeros((len(ytr), len(classes)))
    Y[np.arange(len(ytr)), np.searchsorted(classes, ytr)] = 1
    m = PLSRegression(n_components=min(10, Xtr.shape[1] - 1))
    m.fit(Xtr, Y)
    return balanced_accuracy_score(yte, classes[np.argmax(m.predict(Xte), 1)])


def pcalda(Xtr, ytr, Xte, yte, k=20):
    k = min(k, Xtr.shape[0] - len(np.unique(ytr)) - 1, Xtr.shape[1])
    pc = PCA(n_components=k, random_state=0).fit(Xtr)
    lda = LinearDiscriminantAnalysis().fit(pc.transform(Xtr), ytr)
    return balanced_accuracy_score(yte, lda.predict(pc.transform(Xte)))


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


def run_one(nm):
    t0 = time.time()
    X, y, tr, te = load(nm)
    res = {"n": int(len(X)), "n_channel": int(X.shape[1]),
           "n_class": int(len(np.unique(y))), "n_test": int(len(te))}
    arms = {}
    for kind in ("vicreg", "barlow"):
        enc = train_ssl_encoder(X[tr], kind, seed=0)
        arms[kind] = embed(enc, X)
    enc_rand = Enc().to(DEV).eval()
    arms["randinit"] = embed(enc_rand, X)
    # linear reference in the same low-rank subspace: 20 PCs, then a shallow nonlinear head
    kp = min(20, X.shape[0] - len(np.unique(y)) - 1, X.shape[1])
    pc = PCA(n_components=kp, random_state=0).fit(X[tr])
    arms["pc20"] = pc.transform(X)

    for arm, F in arms.items():
        per_budget = {}
        for b in BUDGETS:
            vals = [probe(F, y, tr, te, b, s) for s in SEEDS]
            per_budget[str(b)] = round(float(np.mean(vals)), 3)
        res[arm] = per_budget
    res["svm_pc20"] = {str(b): round(float(np.mean(
        [probe(arms["pc20"], y, tr, te, b, s, clf_kind="svm") for s in SEEDS])), 3)
        for b in BUDGETS}
    res["plsda"] = round(float(plsda(X[tr], y[tr], X[te], y[te])), 3)
    res["pcalda20"] = round(float(pcalda(X[tr], y[tr], X[te], y[te])), 3)

    deltas = []
    for b in BUDGETS:
        for arm in ("vicreg", "barlow"):
            d = res[arm][str(b)] - res["randinit"][str(b)]
            res.setdefault("delta", {}).setdefault(arm, {})[str(b)] = round(d, 3)
            deltas.append((arm, d))
    for arm in ("vicreg", "barlow"):
        v = [d for a, d in deltas if a == arm]
        res.setdefault("summary", {})[arm] = dict(mean=round(float(np.mean(v)), 4),
                                                  pos=int(sum(d > 0 for d in v)), n=len(v))
    res["secs"] = round(time.time() - t0, 1)
    print(nm, json.dumps(res, ensure_ascii=False), flush=True)
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
    print("saved", OUT)


if __name__ == "__main__":
    main()
