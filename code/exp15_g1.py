"""exp15 —— G1：数据集谱学性质能否预测 SSL 的增益（Δ）。

Δ 一律按**任务级**聚合（先在各标签预算内平均，再在任务内平均），不做逐点伪重复。
数据集级性质（无标签可算 / 需标签但估计量简单）：
  dim99      PCA 解释 99% 方差所需成分数（无标签）
  erank      协方差参与率有效秩（无标签）
  pc1        第一主成分方差占比（无标签）
  snr        类内/类间 Fisher 比（需标签）
  c_incr     净增量谱 d_i 两两 |相关| 均值（需标签）
  cond       端元矩阵条件数（需标签）
  mae_ratio  G0：MAE 重建 RMSE / 最优插值-低秩基线（来自 exp01 复核）

输出 out/exp15_g1.json / .md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

EARLIER = os.path.join(p5_paths.RESULTS, "earlier_round")
OUT = p5_paths.RESULTS
PUB = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]


def delta_task_level():
    """返回 {(dataset, task): Δ均值}，Δ = ssl − randinit。"""
    out = {}
    for nm in PUB:
        p = os.path.join(EARLIER, f"exp07_pub_eff_fix_{nm}.json")
        if not os.path.exists(p):
            continue
        for t, tv in json.load(open(p, encoding="utf-8"))["runs"][nm]["tasks"].items():
            v = [b["delta_ssl_minus_randinit"] for b in tv["budgets"].values()
                 if b.get("delta_ssl_minus_randinit") is not None]
            if v:
                out[(nm, t)] = float(np.mean(v))
    self_p = os.path.join(EARLIER, "exp07_self_label_eff.json")
    if os.path.exists(self_p):
        for nm, r in json.load(open(self_p, encoding="utf-8")).items():
            for t, tv in r["tasks"].items():
                v = [b["delta_ssl_minus_randinit"] for b in tv["budgets"].values()
                     if b.get("delta_ssl_minus_randinit") is not None]
                if v:
                    out[(nm, t)] = float(np.mean(v))
    return out


def props(name):
    """本地复刻 exp01 的载入/ATR 校正/取区，但**不 import torch**。"""
    d = P.load_self(name, verbose=False) if name in P.SELF_ALL else P.pub.load(name)
    axis = np.asarray(d.get("axis", d.get("wvn")), float)
    X = np.asarray(d["X"], float)
    if name != "S3_oil_uvvis":
        X = P.atr_mod.atr_correct(axis, X)
        lo, hi = 900.0, 1800.0
    else:
        lo, hi = 400.0, 1100.0
    sel = (axis >= lo) & (axis <= hi)
    X = X[:, sel]
    yl = d.get("y_recipe_name", d.get("y_adulterant", d.get("y")))
    n, p = X.shape
    Z = PCA(n_components=min(n, p, 50)).fit(X - X.mean(0))
    ev = Z.explained_variance_ratio_
    dim99 = int(np.searchsorted(np.cumsum(ev), 0.99) + 1)
    erank = float(np.exp(-np.sum(np.log(np.clip(ev, 1e-12, 1)) * ev)))
    out = dict(n=n, p=p, dim99=dim99, erank=erank, pc1=float(ev[0]))
    if yl is not None:
        yl = np.asarray(yl)
        classes = [c for c in np.unique(yl) if (yl == c).sum() >= 3]
        C = np.array([X[yl == c].mean(0) for c in classes])
        within = np.mean([X[yl == c].var(0).mean() for c in classes])
        between = float(C.var(0).mean())
        out["fisher"] = float(between / (within + 1e-15))
        D = C[1:] - C[0]                      # 净增量：相对第一类（纯品/基线）
        if len(D) >= 2:
            Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
            R = Dn @ Dn.T
            iu = np.triu_indices(len(D), 1)
            out["c_incr"] = float(np.mean(np.abs(R[iu])))
            out["cond"] = float(np.linalg.cond(D))
            out["n_endmembers"] = int(len(D))
    # G0 比值
    gp = os.path.join(EARLIER, "exp01_g0_fix.json")
    if os.path.exists(gp):
        g = json.load(open(gp, encoding="utf-8"))["runs"].get(name)
        if g:
            out["mae_ratio"] = float(1.0 - g["g0a"]["reduction_vs_best_baseline"])
    return out


def main():
    deltas = delta_task_level()
    ds = sorted({k[0] for k in deltas})
    rows = {}
    for nm in ds:
        try:
            rows[nm] = props(nm)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {nm}: {e}")
    # 数据集级 Δ
    for nm in rows:
        v = [d for (d_, t), d in deltas.items() if d_ == nm]
        rows[nm]["delta_task_mean"] = float(np.mean(v))
        rows[nm]["n_tasks"] = len(v)
    keys = [k for k in ("dim99", "erank", "pc1", "fisher", "c_incr", "cond", "mae_ratio")
            if all(k in rows[nm] and rows[nm][k] is not None for nm in rows)]
    res = {"datasets": rows, "n_datasets": len(rows), "spearman": {}}
    for k in keys:
        x = [rows[nm][k] for nm in rows]
        yv = [rows[nm]["delta_task_mean"] for nm in rows]
        r, p = spearmanr(x, yv)
        res["spearman"][k] = dict(rho=float(r), p=float(p))
        print(f"{k:10s} rho={r:+.3f} p={p:.3f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp15_g1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    L = ["# G1：谱学性质 vs SSL 增益（数据集级，n=%d）" % len(rows), "",
         "| 数据集 | dim99 | 有效秩 | PC1 | Fisher | 增量共线 | 条件数 | MAE/基线 | Δ任务级 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nm, v in rows.items():
        L.append(f"| {nm} | {v.get('dim99','-')} | {v.get('erank',0):.1f} | {v.get('pc1',0):.2f} "
                 f"| {v.get('fisher',float('nan')):.2f} | {v.get('c_incr',float('nan')):.3f} "
                 f"| {v.get('cond',float('nan')):.1f} | {v.get('mae_ratio',float('nan')):.1f} "
                 f"| {v['delta_task_mean']:+.4f} |")
    L += ["", "| 指标 | Spearman ρ | p |", "|---|---|---|"]
    for k, v in res["spearman"].items():
        L.append(f"| {k} | {v['rho']:+.3f} | {v['p']:.3f} |")
    with open(os.path.join(OUT, "exp15_g1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
