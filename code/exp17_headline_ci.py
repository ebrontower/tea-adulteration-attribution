"""exp17 —— 关键格的稳健估计：5 个块偏移 × 5 折 = 25 折，给均值、折间 sd 与 bootstrap 95%CI。"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

FP = (900.0, 1800.0)


def folds5(y, off):
    out = []
    for f in range(5):
        te = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            idx = np.roll(idx, -off)
            te.append(np.array_split(idx, 5)[f])
        te = np.concatenate(te)
        out.append((np.setdiff1d(np.arange(len(y)), te), te))
    return out


def main():
    d = P.load_tea(verbose=False)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    keep = d["y_batch"] == 1
    X = np.asarray(d["X"], float)[keep][:, sel]
    y = d["y"][keep]
    imgs = {
        "raw": X,
        "SG1": savgol_filter(X, 11, 4, deriv=1, axis=1),
        "SNV": (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12),
    }
    masks = {"9cls": np.ones(len(y), bool), "8cls": (y != 3),
             "binary": np.ones(len(y), bool)}
    res = {}
    rng = np.random.default_rng(0)
    for tname, m in masks.items():
        for pname in ("raw", "SG1", "SNV"):
            Xp = StandardScaler().fit_transform(imgs[pname][m])
            ys = (y > 0).astype(int)[m] if tname == "binary" else y[m]
            ys = np.unique(ys, return_inverse=True)[1]      # 连续化标签
            accs = []
            for off in range(5):
                for tr, te in folds5(ys, off):
                    Y = np.eye(len(np.unique(ys)))[ys[tr]]
                    k = max(1, min(10, len(tr) - 1))
                    mdl = PLSRegression(n_components=k).fit(Xp[tr], Y)
                    accs.append(balanced_accuracy_score(
                        ys[te], np.argmax(mdl.predict(Xp[te]), axis=1)))
            accs = np.array(accs)
            boot = [float(np.mean(rng.choice(accs, len(accs), replace=True))) for _ in range(2000)]
            res[f"{tname}|{pname}"] = dict(
                mean=float(accs.mean()), sd=float(accs.std()),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                n_folds=len(accs))
            print(f"{tname:7s} {pname:4s} {accs.mean():.3f} ± {accs.std():.3f} "
                  f"CI [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")
    with open(p5_paths.RESULTS + "/exp17_headline_ci.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
