"""Cross-dataset increment geometry: is the tea result (increments dominated by the matrix
direction, mutually collinear) a tea peculiarity or the norm in published ATR-FTIR
adulteration sets?

For every public set (7) and for tea (6 single adulterants) compute, in a common fingerprint
window (900-1800 cm^-1):
  * r_pure(level)  = corr( mean(level) - mean(0%) , mean(0%) )   signed dilution index
  * snr(level)     = ||increment|| / (within-level noise sd / sqrt(n_replicates))
  * rotation       = corr( increment(level) , increment(max level) )
Cinnamon furnishes three adulterants on one matrix, so the same cross-adulterant increment
correlation / condition number reported for tea can be replicated there.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

REG = (900.0, 1800.0)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def level_geometry(X, y, axis, groups, label):
    m = (axis >= REG[0]) & (axis <= REG[1])
    X = X[:, m]
    levels = np.unique(np.round(y, 6))
    base = levels.min()
    X0 = X[np.isclose(y, base)]
    mu0 = X0.mean(0)
    # noise: pooled within-level residual rms (per channel), over all levels
    resid = []
    mus = {}
    for lv in levels:
        Xl = X[np.isclose(y, lv)]
        mul = Xl.mean(0)
        mus[float(lv)] = mul
        resid.append(Xl - mul)
    sigma = float(np.sqrt(np.mean(np.concatenate(resid, axis=0) ** 2)))
    out = []
    top = float(levels.max())
    inc_top = mus[top] - mu0
    for lv in levels:
        if lv <= base:
            continue
        n = int(np.isclose(y, lv).sum())
        inc = mus[float(lv)] - mu0
        out.append(dict(level=float(lv), n=n,
                        r_pure=round(corr(inc, mu0), 4),
                        snr=round(float(np.linalg.norm(inc) / (sigma / np.sqrt(n))), 1),
                        rot=round(corr(inc, inc_top), 4)))
    return dict(name=label, n=int(X.shape[0]), n_groups=int(len(np.unique(groups))),
                sigma=round(sigma, 5),
                n_channel_window=int(m.sum()),
                ax=(round(float(axis[m][0]), 1), round(float(axis[m][-1]), 1)),
                by_level=out,
                increments={float(lv): mus[float(lv)] - mu0 for lv in levels if lv > base})


def pair_geometry(incs, levels, label):
    """Cross-adulterant increment correlation matrix at a common level (nearest to median)."""
    keys = list(incs)
    common = sorted(set.intersection(*[set(incs[k]) for k in keys]))
    target = float(np.median(common))
    lv = min(common, key=lambda v: abs(v - target))
    R = np.stack([incs[k][lv] for k in keys], axis=1)     # channels x components
    C = np.corrcoef(R.T)
    iu = np.triu_indices(len(keys), 1)
    return dict(at_level=float(lv), labels=keys,
                corr=C.round(4).tolist(),
                range_corr=[float(C[iu].min()), float(C[iu].max())],
                cond=round(float(np.linalg.cond(R.T)), 2))


def main():
    res = {"region": REG, "datasets": {}}
    # ---- public sets -------------------------------------------------------
    cin_incs = {}
    for name in P.public_names():
        d = P.load_public(name)
        axis = np.asarray(d["wvn"], float)
        g = level_geometry(d["X"], np.asarray(d["y"], float), axis,
                           np.asarray(d["groups"], dtype=object), name)
        g["matrix"] = d["spec"]["matrix"]
        g["adulterant"] = d["spec"]["adulterant"]
        res["datasets"][name] = g
        if name.startswith("D7"):
            cin_incs[d["spec"]["adulterant"]] = g["increments"]
    res["cinnamon_increments"] = pair_geometry(cin_incs, None, "cinnamon")

    # ---- tea ---------------------------------------------------------------
    d = P.load_tea(verbose=False)
    X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
    X, yr = X[yb == 1], yr[yb == 1]
    codes = list(range(1, 7))
    names = P.TEA_RECIPE["sulfur"]
    m = (axis >= REG[0]) & (axis <= REG[1])
    Xw = X[:, m]
    mu = {int(c): Xw[yr == c].mean(0) for c in np.unique(yr)}
    tea = {"name": "S1_tea_ir", "n": int(Xw.shape[0]), "by_component": {},
           "n_channel_window": int(m.sum()),
           "ax": (round(float(axis[m][0]), 1), round(float(axis[m][-1]), 1))}
    sigma = float(np.sqrt(np.mean(np.concatenate(
        [Xw[yr == c] - mu[int(c)] for c in np.unique(yr)], axis=0) ** 2)))
    tea["sigma"] = round(sigma, 5)
    for c in codes:
        inc = mu[c] - mu[0]
        n = int((yr == c).sum())
        tea["by_component"][names[c]] = dict(
            r_pure=round(corr(inc, mu[0]), 4),
            snr=round(float(np.linalg.norm(inc) / (sigma / np.sqrt(n))), 1))
    R = np.stack([mu[c] - mu[0] for c in codes], axis=1)
    C = np.corrcoef(R.T)
    iu = np.triu_indices(6, 1)
    tea["increment_corr"] = dict(labels=[names[c] for c in codes],
                                 corr=C.round(4).tolist(),
                                 range_corr=[float(C[iu].min()), float(C[iu].max())],
                                 cond=round(float(np.linalg.cond(R.T)), 2))
    res["tea"] = tea
    res["tea_increments"] = {names[c]: (mu[c] - mu[0]).tolist() for c in codes}

    outp = os.path.join(p5_paths.RESULTS, "exp22_pubgeom.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    slim = {"region": res["region"], "datasets": {}, "tea": res["tea"],
            "cinnamon_increments": res["cinnamon_increments"]}
    for k, g in res["datasets"].items():
        slim["datasets"][k] = {kk: vv for kk, vv in g.items() if kk != "increments"}
    json.dump(slim, open(outp, "w"), indent=1, ensure_ascii=False)

    # ---- console summary ---------------------------------------------------
    print(f"region {REG}, tea window {tea['ax']}, channels {tea['n_channel_window']}")
    print(f"{'set':6s} {'matrix':14s} {'adulterant':12s} {'n':>4s} {'grp':>4s} "
          f"{'top_r':>7s} {'r@min>0':>8s} {'snr_top':>8s} {'rot':>6s}")
    for k, g in res["datasets"].items():
        bl = g["by_level"]
        print(f"{k:6s} {g['matrix'][:14]:14s} {g['adulterant'][:12]:12s} "
              f"{g['n']:4d} {g['n_groups']:4d} {bl[-1]['r_pure']:7.3f} "
              f"{bl[0]['r_pure']:8.3f} {bl[-1]['snr']:8.1f} {bl[-1]['rot']:6.3f}")
    print("cinnamon increments:", json.dumps(res["cinnamon_increments"], ensure_ascii=False))
    print("tea increments:", json.dumps({k: v for k, v in tea["by_component"].items()},
                                        ensure_ascii=False))
    print("tea corr:", json.dumps(tea["increment_corr"], ensure_ascii=False))


if __name__ == "__main__":
    main()
