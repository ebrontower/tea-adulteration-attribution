"""exp30 - E5: is the VICReg/Barlow gain representation learning, or just its augmentations?

E2 compares pretexts against a random-init encoder. But every VICReg view is built from the same
label-preserving augmentations that a spectroscopist already applies as preprocessing (baseline
offset/slope, multiplicative scatter, channel noise, band dropout, smoothing). If plain PLS-DA on
identically augmented *labelled* spectra matches VICReg, then the pretext is not learning a
representation -- it is doing data augmentation, which needs no encoder at all.

Arms, all at the same split and the same label budget b per class:
    pls_b          PLS-DA on the b/class labelled spectra
    pls_aug_b      the same spectra, each with 4 augmented copies (5x rows)
    vicreg_b       frozen VICReg probe (read from exp26 where available)
    randinit_b     frozen random-init probe (read from exp26)
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score

import exp28_label_subspace as E3  # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp30_aug_control.json"
E2_JSON = p5_paths.RESULTS + "/exp26_nonrecon_ssl.json"
BUDGETS = (5, 10, 20)
SEEDS = (0, 1, 2)
NCOPY = 4


def augment_np(X, rng):
    B, N = X.shape
    out = X.copy()
    t = np.linspace(-1, 1, N)
    for i in range(B):
        row = out[i]
        row = row + rng.normal(0, 0.15) + t * rng.normal(0, 0.10)
        row = row * (1.0 + 0.10 * rng.normal())
        row = row + rng.normal(0, 0.03, N)
        if rng.random() < 0.5:
            w = int(N * (0.02 + 0.06 * rng.random()))
            s = int(rng.integers(0, max(1, N - w)))
            row[s:s + w] = row.mean()
        if rng.random() < 0.5:
            k = int(rng.integers(3, 9))
            pad = np.pad(row, (k // 2, k // 2), mode="edge")
            row = np.convolve(pad, np.ones(k) / k, mode="valid")[:N]
        out[i] = row
    return out


def plsda(Xtr, ytr, Xte, yte, ncomp=10):
    classes = np.unique(ytr)
    Y = np.zeros((len(ytr), len(classes)))
    Y[np.arange(len(ytr)), np.searchsorted(classes, ytr)] = 1
    m = PLSRegression(n_components=min(ncomp, Xtr.shape[1] - 1))
    m.fit(Xtr, Y)
    return float(balanced_accuracy_score(yte, classes[np.argmax(m.predict(Xte), 1)]))


def sub_index(y, tr, b, rng):
    sub = []
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c].copy()
        rng.shuffle(idx)
        sub += list(idx[:b])
    return np.array(sub)


def run_one(nm, e2):
    X, y, tr, te = E3.load(nm)
    res = {"n": int(len(X)), "n_channel": int(X.shape[1])}
    res["pls_all"] = round(plsda(X[tr], y[tr], X[te], y[te]), 3)
    for b in BUDGETS:
        base, augd = [], []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            sub = sub_index(y, tr, b, rng)
            base.append(plsda(X[sub], y[sub], X[te], y[te]))
            Xa = [X[sub]] + [augment_np(X[sub], rng) for _ in range(NCOPY)]
            ya = [y[sub]] * (NCOPY + 1)
            augd.append(plsda(np.vstack(Xa), np.concatenate(ya), X[te], y[te]))
        res.setdefault("pls_b", {})[str(b)] = round(float(np.mean(base)), 3)
        res.setdefault("pls_aug_b", {})[str(b)] = round(float(np.mean(augd)), 3)
        for arm in ("vicreg", "barlow", "randinit"):
            if nm in e2 and arm in e2[nm]:
                res.setdefault(arm, {})[str(b)] = e2[nm][arm][str(b)]
    print(nm, json.dumps({k: v for k, v in res.items() if k != "n_channel"}, ensure_ascii=False),
          flush=True)
    return res


def main():
    names = sys.argv[1:] or ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8", "TEA"]
    try:
        e2 = json.load(open(E2_JSON))
    except Exception:  # noqa: BLE001
        e2 = {}
    try:
        out = json.load(open(OUT))
    except Exception:  # noqa: BLE001
        out = {}
    for nm in names:
        try:
            out[nm] = run_one(nm, e2)
        except Exception as e:  # noqa: BLE001
            out[nm] = {"error": repr(e)}
            print(nm, "FAILED", e, flush=True)
        json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
