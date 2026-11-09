"""exp16 —— 逐掺假物的单独可判别性（纯茶 vs 单一掺假），块划分 × 5 偏移。

目的：区分两件事——
  (a) 该掺假物单独存在时能否检出（"有没有信号"）
  (b) 在复合配方里能否被归因到它头上（exp14 C 臂的逐组分召回）
石蜡若 (a) 高而 (b)=0，说明它表现为"稀释"而非"独立指纹"，这正是可加性失效的机制。
"""
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
NAMES = {1: "白糖", 2: "滑石粉", 3: "硫磺", 4: "石蜡", 5: "色素", 6: "香精", 7: "白糖+石蜡",
         8: "滑石粉+石蜡+香精"}


def main():
    d = P.load_tea(verbose=False)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    keep = d["y_batch"] == 1
    X = np.asarray(d["X"], float)[keep][:, sel]
    y = d["y"][keep]
    Xs = StandardScaler().fit_transform(savgol_filter(X, 11, 4, deriv=1, axis=1))
    res = {}
    for c, nm in NAMES.items():
        m = (y == 0) | (y == c)
        Xc, yc = Xs[m], (y[m] == c).astype(int)
        accs = []
        for off in range(5):
            folds = []
            for cl in (0, 1):
                idx = np.where(yc == cl)[0]
                idx = np.roll(idx, -off)
                folds.append(np.array_split(idx, 5))
            for f in range(5):
                te = np.concatenate([folds[0][f], folds[1][f]])
                tr = np.setdiff1d(np.arange(len(yc)), te)
                Y = np.eye(2)[yc[tr]]
                mdl = PLSRegression(n_components=10).fit(Xc[tr], Y)
                p = np.argmax(mdl.predict(Xc[te]), axis=1)
                accs.append(balanced_accuracy_score(yc[te], p))
        res[nm] = dict(bacc=float(np.mean(accs)), sd=float(np.std(accs)),
                       ci95=[float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                       n=int(m.sum()))
        print(f"{nm:12s} 纯 vs 单掺 bacc {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    with open(p5_paths.RESULTS + "/exp16_per_adulterant.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
