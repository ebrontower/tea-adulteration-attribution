"""exp13 —— 茶叶主表的诚实评价（去重 + 漂移阻断划分 + 预处理消融）。

背景（2026-10-30 审计）：
  * 「有硫磺/无硫磺」两个 xlsx **不是两批实验**：7/8 个配方逐行最大差 = 0.000，
    是同一组 50 条扫描的重复导出。有效独立谱 = 450（9 配方 × 50），不是 850。
  * 类内 PC1 解释 48–98%，且与扫描顺序相关（|r| 达 0.76）→ 扫描级随机 7:3 会把
    漂移方向插值到测试集，bacc 被高估。

因此本脚本给出三套协议下的对照：
  P1 scan   : 类内随机 7:3（沿用旧口径，仅用于量化虚高幅度）
  P2 block  : 类内连续 5 块，留一块（阻断漂移插值）
  P3 time   : 类内前 50% 训练 / 后 50% 测试（时间分离，最严）

预处理：raw / SNV / MSC / SG1 / SG2 / area（面积归一，去掉总吸光强度）
分类器：PLS-DA（主）、PCA-LDA（辅）、仅强度基线（单特征 = 平均吸光度）

输出 out/exp13_tea_clean.{json,md}
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS
FP = (900.0, 1800.0)
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 数据
def load_dedup():
    d = P.load_tea(verbose=False)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    X = np.asarray(d["X"], float)[:, sel]
    y, b = d["y"], d["y_batch"]
    # 保留「有硫磺」文件（含全部 9 配方），仅在滑石粉上它与另一份不同
    keep = b == 1
    return X[keep], y[keep], axis[sel], d


# ---------------------------------------------------------------- 预处理
def snv(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True) + 1e-12
    return (X - m) / s


def msc(X, ref=None):
    ref = X.mean(0) if ref is None else ref
    A = np.vstack([ref, np.ones_like(ref)]).T
    out = np.empty_like(X)
    for i, x in enumerate(X):
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        out[i] = (x - coef[1]) / (coef[0] + 1e-12)
    return out


def area(X):
    a = np.trapezoid(np.abs(X), axis=1)[:, None]
    return X / (a + 1e-12)


def sg(X, order):
    w = 11 if X.shape[1] > 11 else (X.shape[1] // 2) * 2 - 1
    return savgol_filter(X, window_length=w, polyorder=min(3, order + 1), deriv=order, axis=1)


PREP = {"raw": lambda X: X, "SNV": snv, "MSC": msc, "SG1": lambda X: sg(X, 1),
        "SG2": lambda X: sg(X, 2), "area": area}


# ---------------------------------------------------------------- 划分
def splits_for(y, protocol, seed=0):
    """返回 [(tr, te), ...]；全部为类内划分，保证训练/测试类集合一致。"""
    rng = np.random.default_rng(seed)
    cls = np.unique(y)
    if protocol == "scan":
        tr, te = [], []
        for c in cls:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            k = int(round(0.7 * len(idx)))
            tr.append(idx[:k]); te.append(idx[k:])
        return [(np.concatenate(tr), np.concatenate(te))]
    if protocol == "block":
        out = []
        for f in range(5):
            tr, te = [], []
            for c in cls:
                idx = np.where(y == c)[0]          # 保持原始行序 = 扫描顺序
                q = np.array_split(idx, 5)[f]
                te.append(q)
                tr.append(np.setdiff1d(idx, q))
            out.append((np.concatenate(tr), np.concatenate(te)))
        return out
    if protocol == "time":
        tr, te = [], []
        for c in cls:
            idx = np.where(y == c)[0]
            k = len(idx) // 2
            tr.append(idx[:k]); te.append(idx[k:])
        return [(np.concatenate(tr), np.concatenate(te))]
    raise ValueError(protocol)


def plsda(Xtr, ytr, Xte, ncomp=10):
    cls = np.unique(ytr)
    Y = np.zeros((len(ytr), len(cls)))
    for j, c in enumerate(cls):
        Y[ytr == c, j] = 1
    k = max(1, min(ncomp, len(ytr) - 1, Xtr.shape[1]))
    m = PLSRegression(n_components=k).fit(Xtr, Y)
    return cls[np.argmax(m.predict(Xte), axis=1)]


def pcalda(Xtr, ytr, Xte, ncomp=10):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    from sklearn.decomposition import PCA
    k = max(1, min(ncomp, len(Xtr) - 1, A.shape[1]))
    pc = PCA(n_components=k).fit(A)
    return LDA().fit(pc.transform(A), ytr).predict(pc.transform(B))


def eval_protocol(X, y, protocol, seed=0, ncomp=10):
    accs, f1s = [], []
    for tr, te in splits_for(y, protocol, seed):
        p = plsda(X[tr], y[tr], X[te], ncomp)
        accs.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def intensity_baseline(X, y, protocol, seed=0):
    """只用总强度（平均吸光度）一个标量做阈值判别 —— 物理效应的下限检验。"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    accs = []
    for tr, te in splits_for(y, protocol, seed):
        f_tr = X[tr].mean(1, keepdims=True)
        f_te = X[te].mean(1, keepdims=True)
        sc = StandardScaler().fit(f_tr)
        accs.append(balanced_accuracy_score(
            y[te], LDA().fit(sc.transform(f_tr), y[tr]).predict(sc.transform(f_te))))
    return float(np.mean(accs))


def main():
    X0, y, axis, d = load_dedup()
    res = {"n": int(len(y)), "classes": sorted(set(y.tolist())),
           "note": "dedup: 保留有硫磺文件（450 条），两文件 7/8 配方逐位相同",
           "grid": {}, "intensity": {}}

    tasks = {"9cls": (X0, y),
             "8cls_nosulfur": (X0[y != 3], y[y != 3]),
             "binary_pure_vs_adulter": (X0, (y > 0).astype(int)),
             "binary_pure_vs_adulter_nosulfur": (X0[y != 3], (y[y != 3] > 0).astype(int))}

    for tname, (X, yt) in tasks.items():
        res["grid"][tname] = {}
        for pname, pf in PREP.items():
            Xp = pf(X)
            Xp = StandardScaler().fit_transform(Xp)
            res["grid"][tname][pname] = {}
            for proto in ("scan", "block", "time"):
                m, s, f = eval_protocol(Xp, yt, proto, seed=0)
                res["grid"][tname][pname][proto] = dict(bacc=m, sd=s, macro_f1=f)
            print(f"[{tname}] {pname:5s} scan {res['grid'][tname][pname]['scan']['bacc']:.3f} "
                  f"block {res['grid'][tname][pname]['block']['bacc']:.3f} "
                  f"time {res['grid'][tname][pname]['time']['bacc']:.3f}", flush=True)
        res["intensity"][tname] = {proto: intensity_baseline(X, yt, proto)
                                   for proto in ("scan", "block", "time")}
        print(f"[{tname}] 仅强度基线: {res['intensity'][tname]}", flush=True)

    with open(os.path.join(OUT, "exp13_tea_clean.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("saved")


if __name__ == "__main__":
    main()
