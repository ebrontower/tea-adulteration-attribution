"""exp31 - E4 control: is the MAE arm losing on the synthetic set because of information or capacity?

In exp29 the MAE probe (128-d features) loses badly to a 4-PC probe at every label budget, while
in the real data the same 128-d probe was competitive. Two explanations: (a) the encoder does not
carry the label information, (b) the probe is capacity-limited (80 labelled spectra, 128 dims).
This reruns the key cells with the MAE embedding compressed to its top 4 PCs, so both arms have
the same feature dimension, plus a PC-12 reference (all latent components) as the ceiling.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

import exp29_phase_band as E  # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp31_synth_control.json"
BUDGETS = (5, 10, 20)
SEEDS = (0, 1, 2)


def run(lo, kind, seeds=SEEDS, k=4):
    res = {"lo": lo, "label": kind, "k": k}
    curves = {a: [] for a in ("mae", "mae_pc4", "pc4", "pc4_svm", "pc12", "rand")}
    for sd in seeds:
        X, y, s = E.make(lo, kind, seed=sd)
        tr, te = E.split(y, seed=sd)
        mu, sdv = X[tr].mean(0), float(X[tr].std())
        Xs = (X - mu) / sdv
        _, model = E.GT.train_mae1d(Xs[tr], patch=E.PATCH, ratio=0.5, seed=sd, epochs=E.EPOCHS,
                                    device=E.DEV, return_model=True)
        Zm = E.embed(model, Xs)
        Zm4 = PCA(n_components=k, random_state=0).fit_transform(Zm - Zm[tr].mean(0))
        pc4 = PCA(n_components=4, random_state=0).fit(Xs[tr]).transform(Xs)
        pc12 = PCA(n_components=min(12, Xs.shape[1]), random_state=0).fit(Xs[tr]).transform(Xs)
        for key, F, kin in (("mae", Zm, "linear"), ("mae_pc4", Zm4, "linear"),
                            ("pc4", pc4, "linear"), ("pc4_svm", pc4, "svm"),
                            ("pc12", pc12, "linear")):
            curves[key].append({b: E.probe(F, y, tr, te, b, kin, sd) for b in BUDGETS})
    for b in BUDGETS:
        res[str(b)] = {a: round(float(np.mean([v[b] for v in curves[a]])), 3)
                       if curves[a] else None for a in ("mae", "mae_pc4", "pc4", "pc4_svm", "pc12")}
        res[str(b)]["gain_mae"] = round(res[str(b)]["mae"] - max(res[str(b)]["pc4"],
                                                                res[str(b)]["pc4_svm"]), 3)
        res[str(b)]["gain_mae_pc4"] = round(res[str(b)]["mae_pc4"] - max(
            res[str(b)]["pc4"], res[str(b)]["pc4_svm"]), 3)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    out = []
    for kind in ("linear", "radial"):
        for lo in (1, 3):
            out.append(run(lo, kind))
            json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
