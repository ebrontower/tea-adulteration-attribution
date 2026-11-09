"""exp14 —— 复合掺假的集合预测基准。

问题：单标签判别式（PLS-DA argmax 配方）在复合配方上**结构上**无法输出「{白糖,石蜡}」，
      旧结果因此给出严格集合正确率 0.000。这是模型缺陷还是任务本身不可学？

三臂：
  A 单标签 PLS-DA（旧口径）：argmax 配方 → 集合 = 该配方组成。给出严格/宽松指标。
  B 多标签 one-vs-rest PLS2（6 个组分，阈值在训练折内标定）：正常训练（含复合）
    → 复合谱上的集合预测性能。
  C 新组合外推：留一复合配方。训练 = 单掺 1–6 + 另一个复合，测试 = 留出复合。
    这是「从没见过的组合」能否给出正确集合。
  D 查表臂（参照）：9 类配方分类器 → 按配方名查组成表。仅当配方在训练集里才有意义。

特征：SG1 + StandardScaler（exp13 中 scan/block/time 最稳）。
划分：5 折连续块（阻断扫描顺序），重复 5 个随机块偏移。
指标：exact set match、micro-P / R / F1、逐组分 AUC。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS
FP = (900.0, 1800.0)
COMP = ["白糖", "滑石粉", "硫磺", "石蜡", "色素", "香精"]


def feats():
    d = P.load_tea(verbose=False)
    axis = np.asarray(d["axis"], float)
    sel = (axis >= FP[0]) & (axis <= FP[1])
    X = np.asarray(d["X"], float)[:, sel]
    keep = d["y_batch"] == 1
    X = X[keep]
    Xs = savgol_filter(X, 11, 4, deriv=1, axis=1)
    Xs = StandardScaler().fit_transform(Xs)
    return Xs, d["y"][keep], d["y_components"][keep], d["y_recipe_name"][keep]


def blocks(y, nfold=5, offset=0):
    """类内连续块划分（offset 让块边界滑动）。"""
    idx_all = np.arange(len(y))
    folds = [[] for _ in range(nfold)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = np.roll(idx, -offset)
        for f, q in enumerate(np.array_split(idx, nfold)):
            folds[f].append(q)
    folds = [np.concatenate(f) for f in folds]
    return folds


def pls_scores(Xtr, Ytr, Xte, k=10):
    k = max(1, min(k, len(Xtr) - 1, Xtr.shape[1]))
    m = PLSRegression(n_components=k).fit(Xtr, Ytr)
    return m.predict(Xte), m


def tune_threshold(scores, Y):
    """在训练折上用内部 4 折选每个组分的阈值（最大化 micro-F1）。"""
    ths = []
    for j in range(Y.shape[1]):
        s, t = scores[:, j], Y[:, j]
        if s.max() - s.min() < 1e-9:
            ths.append(0.5); continue
        cand = np.quantile(s, np.linspace(0.05, 0.95, 19))
        best, bth = -1, 0.5
        for th in cand:
            p = (s >= th).astype(int)
            tp = ((p == 1) & (t == 1)).sum(); fp = ((p == 1) & (t == 0)).sum()
            fn = ((p == 0) & (t == 1)).sum()
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            if f1 > best:
                best, bth = f1, float(th)
        ths.append(bth)
    return np.array(ths)


def set_metrics(Yt, P_):
    exact = float(np.mean([np.array_equal(a, b) for a, b in zip(Yt, P_)]))
    tp = int(((P_ == 1) & (Yt == 1)).sum()); fp = int(((P_ == 1) & (Yt == 0)).sum())
    fn = int(((P_ == 0) & (Yt == 1)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return dict(exact=exact, micro_p=float(p), micro_r=float(r), micro_f1=float(f1),
                tp=tp, fp=fp, fn=fn)


def main():
    X, y, Yc, rn = feats()
    n = len(y)
    res = {"n": int(n), "components": COMP, "arms": {}}

    # ---------- A 单标签 PLS-DA（旧口径） ----------
    exact_A, loose_A, acc_A = [], [], []
    for off in range(5):
        folds = blocks(y, 5, off)
        for f in range(5):
            te = folds[f]; tr = np.setdiff1d(np.arange(n), te)
            cls = np.unique(y[tr])
            Y = np.zeros((len(tr), len(cls)))
            for j, c in enumerate(cls):
                Y[y[tr] == c, j] = 1
            k = max(1, min(10, len(tr) - 1))
            sc, _ = pls_scores(X[tr], Y, X[te], k)
            pred = cls[np.argmax(sc, axis=1)]
            acc_A.append(balanced_accuracy_score(y[te], pred))
            for t, p_ in zip(te, pred):
                comps = set(P.TEA_RECIPE_NAMES[int(y[t])].split("+"))
                pc = set(P.TEA_RECIPE_NAMES[int(p_)].split("+"))
                loose_A.append(len(comps & pc) > 0)
                exact_A.append(comps == pc)
    res["arms"]["A_singlelabel"] = dict(bacc_recipe=float(np.mean(acc_A)),
                                        exact=float(np.mean(exact_A)),
                                        loose=float(np.mean(loose_A)))

    # ---------- B 多标签 one-vs-rest（正常训练，含复合） ----------
    for arm, keep_mask in (("B_multilabel_all", np.ones(n, bool)),):
        ex, metrics, aucs = [], [], {c: [] for c in COMP}
        for off in range(5):
            folds = blocks(y[keep_mask], 5, off)
            Xs, Yc_, ys = X[keep_mask], Yc[keep_mask], y[keep_mask]
            for f in range(5):
                te, tr = folds[f], np.setdiff1d(np.arange(len(ys)), folds[f])
                sc_tr, m = pls_scores(Xs[tr], Yc_[tr].astype(float), Xs[tr], 10)
                th = tune_threshold(sc_tr, Yc_[tr])
                sc_te, _ = pls_scores(Xs[tr], Yc_[tr].astype(float), Xs[te], 10)
                P_ = (sc_te >= th).astype(int)
                metrics.append(set_metrics(Yc_[te], P_))
                for j, c in enumerate(COMP):
                    if 0 < Yc_[te][:, j].sum() < len(te):
                        aucs[c].append(roc_auc_score(Yc_[te][:, j], sc_te[:, j]))
        res["arms"][arm] = dict(
            exact=float(np.mean([m["exact"] for m in metrics])),
            micro_p=float(np.mean([m["micro_p"] for m in metrics])),
            micro_r=float(np.mean([m["micro_r"] for m in metrics])),
            micro_f1=float(np.mean([m["micro_f1"] for m in metrics])),
            auc={c: (float(np.mean(v)) if v else None) for c, v in aucs.items()})

    # 只看复合配方（7/8）上的表现
    is_mix = np.isin(y, [7, 8])
    ex, metrics = [], []
    for off in range(5):
        folds = blocks(y, 5, off)
        for f in range(5):
            te, tr = folds[f], np.setdiff1d(np.arange(n), folds[f])
            sc_tr, _ = pls_scores(X[tr], Yc[tr].astype(float), X[tr], 10)
            th = tune_threshold(sc_tr, Yc[tr])
            sc_te, _ = pls_scores(X[tr], Yc[tr].astype(float), X[te], 10)
            P_ = (sc_te >= th).astype(int)
            m = is_mix[te]
            if m.sum():
                metrics.append(set_metrics(Yc[te][m], P_[m]))
    res["arms"]["B_multilabel_mixtures_only"] = dict(
        exact=float(np.mean([m["exact"] for m in metrics])),
        micro_f1=float(np.mean([m["micro_f1"] for m in metrics])),
        micro_p=float(np.mean([m["micro_p"] for m in metrics])),
        micro_r=float(np.mean([m["micro_r"] for m in metrics])))

    # ---------- C 新组合外推：留一复合配方 ----------
    for held in (7, 8):
        Xtr = X[(y != held) & (~np.isin(y, []) )]
        Ytr = Yc[(y != held)]
        ytr = y[y != held]
        mte = y == held
        sc_tr, _ = pls_scores(Xtr, Ytr.astype(float), Xtr, 10)
        th = tune_threshold(sc_tr, Ytr)
        sc_te, _ = pls_scores(Xtr, Ytr.astype(float), X[mte], 10)
        P_ = (sc_te >= th).astype(int)
        res["arms"][f"C_novelcombo_hold{held}"] = set_metrics(Yc[mte], P_)
        res["arms"][f"C_novelcombo_hold{held}"]["per_component_recall"] = {
            c: (float(P_[Yc[mte][:, j] == 1, j].mean())
                if (Yc[mte][:, j] == 1).any() else None)
            for j, c in enumerate(COMP)}
        res["arms"][f"C_novelcombo_hold{held}"]["per_component_fp"] = {
            c: int(((P_[:, j] == 1) & (Yc[mte][:, j] == 0)).sum()) for j, c in enumerate(COMP)}
        res["arms"][f"C_novelcombo_hold{held}"]["recipe"] = P.TEA_RECIPE_NAMES[held]
        # 同时给单标签臂在该留出下的严格率
        cls = np.unique(ytr)
        Yoh = np.zeros((len(ytr), len(cls)))
        for j, c in enumerate(cls):
            Yoh[ytr == c, j] = 1
        sc2, _ = pls_scores(Xtr, Yoh, X[mte], 10)
        pred = cls[np.argmax(sc2, axis=1)]
        tgt = set(P.TEA_RECIPE_NAMES[held].split("+"))
        res["arms"][f"C_novelcombo_hold{held}"]["singlelabel_exact"] = float(
            np.mean([set(P.TEA_RECIPE_NAMES[int(p_)].split("+")) == tgt for p_ in pred]))
        res["arms"][f"C_novelcombo_hold{held}"]["singlelabel_loose"] = float(
            np.mean([len(set(P.TEA_RECIPE_NAMES[int(p_)].split("+")) & tgt) > 0 for p_ in pred]))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp14_sets.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res["arms"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
