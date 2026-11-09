"""exp42 - the second form of route (1): is there headroom where the signal is WEAK?

exp41 settled the paired-session version of "give the representation something to do"
(60 meat samples x RunA/RunB, 60 olive oils x two sessions): SNV + PLS-DA already transfers at
0.91-1.00 and target-supervised fitting is not better, so there the session shift is a
fixed-metric problem. The remaining place a learned representation could matter is low-dose
adulteration: the increment is small against sample-to-sample scatter and the calibration points
sit far above the level to be detected.

Seven deposited corpora record the adulterant fraction, so the transfer is over *magnitude*:
  D1 olive oil / sunflower oil   D2 roasted coffee / malt flour   D3 orange juice / apple juice
  D7w/D7p/D7d cinnamon / walnut, peanut, wood dust   D8 turmeric / wheat flour

Protocols; every split is disjoint at physical-sample level and the two classes are count-matched.
  det_HIGH_to_LOW   fit HIGH + authentic  -> detect LOW   (extrapolation down, the hard direction)
  det_LOW_to_HIGH   fit LOW  + authentic  -> detect HIGH  (the easy direction, for contrast)
  quant_down        PLS on HIGH, predict held-out LOW samples  (bias = under-estimation)
  quant_inter       PLS on all adulterated levels, same held-out LOW rows (interpolation control)

Geometry in the attribution line's own metric (whitened by the pooled within-level scatter of
authentic + HIGH):
  d_LOW_over_within / d_HIGH_over_within   band separation from authentic, whitened units
  ratio_LOW_over_HIGH                      observed distance ratio between the two bands
  dose_ratio_low_high                      what linearity of mixing would predict
  cos_LOW_HIGH                             cos(mu_LOW-mu_AUTH, mu_HIGH-mu_AUTH)
  subspace_kept_LOW                        share of the LOW-vs-authentic difference that lies in
                                           the top-20 PCs fitted on HIGH + authentic
  spearman_dose_vs_projection              monotonicity of level along the HIGH direction

A learned representation has a job only if the LOW direction departs from the HIGH direction and
the subspace test shows structure the calibration band does not span. If cos ~ 1 and the distance
ratio tracks the dose ratio, magnitude transfer is a scaling problem that linear models already
solve.
"""
import json
import os
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

import p5_public_data as PP                                   # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp42_magnitude_transfer.json"
SEED = 20261101
K_W = 60
PLS_COMP = 6


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def terciles(levels):
    lv = np.unique(levels[levels > 0])
    if len(lv) < 3:
        return np.array([]), np.array([]), np.array([])
    q = np.quantile(lv, [1 / 3, 2 / 3])
    return lv[lv <= q[0]], lv[(lv > q[0]) & (lv <= q[1])], lv[lv > q[1]]


def round6(v):
    return np.round(np.asarray(v, float), 6)


def whitener(X, y_level):
    """Pooled within-level covariance, top-K_W directions, whitened coordinates."""
    lev = np.unique(y_level)
    W = np.zeros((X.shape[1], X.shape[1]))
    dof = 0
    for c in lev:
        mm = X[y_level == c]
        if len(mm) > 1:
            d = mm - mm.mean(0)
            W += d.T @ d
            dof += len(mm) - 1
    W /= max(dof, 1)
    w, V = np.linalg.eigh(W + np.eye(W.shape[0]) * 1e-10)
    w = np.clip(w, 1e-8, None)
    order = np.argsort(w)[::-1][:int(min(K_W, len(w)))]
    return V[:, order] * w[order] ** -0.5          # (channels, k): project with X @ V


def auc(score, label):
    """Rank-based AUC of `score` separating label==1 from label==0."""
    s = np.asarray(score, float)
    l = np.asarray(label, bool)
    n1, n0 = int(l.sum()), int((~l).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return round(float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def half_groups(groups):
    """Physical-sample groups of one band split into two disjoint halves."""
    gs = np.unique(groups).copy()
    np.random.RandomState(SEED + 21).shuffle(gs)
    h = len(gs) // 2
    return set(gs[:h].tolist()), set(gs[h:].tolist())


def gate(X, g, y, idx):
    """One-class dose gate. Fitted on HALF of the authentic groups plus HALF of the HIGH groups;
    every band is scored on its other half, so the answer never depends on how many authentic
    samples the deposit happens to contain. Arms: absolute absorbance and SNV."""
    out = {}
    for pre in ("raw", "snv"):
        Z = X if pre == "raw" else snv(X)
        auth = idx["auth"]
        ga_auth, gb_auth = half_groups(g[auth])
        ga_high, gb_high = half_groups(g[idx["high"]])
        fitmask = (auth & np.isin(g, list(ga_auth))) | (idx["high"] & np.isin(g, list(ga_high)))
        Zf, yf = Z[fitmask], y[fitmask]
        if (yf == 0).sum() < 2 or (yf > 0).sum() < 2:
            out[pre] = dict(note="fit half too small", n_fit=int(fitmask.sum()))
            continue
        V = whitener(Zf, yf)
        P = lambda R: (Z[R] @ V)
        sfit = P(np.where(fitmask)[0])
        u = sfit[yf > 0].mean(0) - sfit[yf == 0].mean(0)
        u /= (np.linalg.norm(u) + 1e-12)
        tau = float(np.quantile(sfit[yf == 0] @ u, 0.95))
        test_auth = np.where(auth & np.isin(g, list(gb_auth)))[0]
        st = P(test_auth) @ u
        arm = dict(prep=pre, n_fit=int(fitmask.sum()), tau=round(tau, 3),
                   n_test_auth=int(len(test_auth)),
                   false_alarm_test_auth=round(float(np.mean(st > tau)), 4))
        for name, mask in [("LOW", idx["low"]), ("MID", idx["mid"]), ("HIGH", idx["high"])]:
            keep = gb_high if name == "HIGH" else half_groups(g[mask])[1]
            rows = np.where(mask & np.isin(g, list(keep)))[0]
            if len(rows) == 0:
                continue
            s = P(rows) @ u
            lab = np.concatenate([np.ones(len(rows)), np.zeros(len(test_auth))]).astype(bool)
            sc = np.concatenate([s, st])
            arm[name] = dict(n=int(len(rows)), levels=sorted(set(np.round(y[rows], 2))),
                             auc=auc(sc, lab), hit=round(float(np.mean(s > tau)), 4),
                             mean_score=round(float(s.mean()), 3))
        out[pre] = arm
    return out


def quantification(X, g, y, high_rows, low_rows):
    """PLS trained on HIGH, evaluated on group-disjoint LOW rows; interpolation control on the
    same test rows uses LOW+MID+HIGH training rows but the same held-out groups."""
    used = set(g[low_rows])
    high_fit = np.array([r for r in high_rows if g[r] not in used])
    low_fit = np.array([r for r in low_rows if g[r] in used])
    out = {}
    if len(high_fit) < 10 or len(low_fit) < 10:
        return dict(note="insufficient rows", n_high_fit=int(len(high_fit)), n_low_fit=int(len(low_fit)))
    for pre in ("raw", "snv"):
        tr_h = X[high_fit] if pre == "raw" else snv(X[high_fit])
        te = X[low_fit] if pre == "raw" else snv(X[low_fit])
        p = np.asarray(PLSRegression(n_components=PLS_COMP).fit(tr_h, y[high_fit]).predict(te)).ravel()
        err = p - y[low_fit]
        o = dict(rmse=round(float(np.sqrt(np.mean(err ** 2))), 3),
                 bias=round(float(np.mean(err)), 3),
                 rel_bias_pct=round(float(100 * np.mean(err) / np.mean(y[low_fit])), 2),
                 n_fit=int(len(high_fit)), n_test=int(len(low_fit)))
        tr_all = np.concatenate([high_fit, low_fit])
        off = len(high_fit)                        # low_fit starts at this offset in tr_all
        za = X[tr_all] if pre == "raw" else snv(X[tr_all])
        # leave-one-group-out over the same test groups
        pred = np.zeros(len(low_fit))
        for q in np.unique(g[low_fit]):
            mte = np.where(g[low_fit] == q)[0]
            mtr = tr_all[~np.isin(g[tr_all], [q])]
            if len(mtr) < 10:
                pred[mte] = np.nan
                continue
            ztr = X[mtr] if pre == "raw" else snv(X[mtr])
            pred[mte] = np.asarray(PLSRegression(n_components=PLS_COMP)
                                   .fit(ztr, y[mtr]).predict(za[off + mte])).ravel()
        e2 = pred - y[low_fit]
        o2 = dict(rmse=round(float(np.sqrt(np.nanmean(e2 ** 2))), 3),
                  bias=round(float(np.nanmean(e2)), 3),
                  rel_bias_pct=round(float(100 * np.nanmean(e2) / np.nanmean(y[low_fit])), 2))
        out[pre] = dict(extrapolate_down=o, interpolate_same_rows=o2)
    return out


def run(name):
    d = PP.load(name)
    X, y, g = d["X"], np.asarray(d["y"], float), np.asarray(d["groups"], object)
    low, mid, high = terciles(y)
    res = dict(name=name, matrix=d["meta"]["matrix"], adulterant=d["meta"]["adulterant"],
               n=int(len(y)), n_channel=int(X.shape[1]), n_groups=int(len(set(g))),
               axis=[float(d["wvn"][0]), float(d["wvn"][-1])],
               levels=[float(v) for v in np.unique(y)],
               bands=dict(low=[float(v) for v in low], mid=[float(v) for v in mid],
                          high=[float(v) for v in high]))
    if len(low) == 0 or len(high) == 0:
        res["skipped"] = "fewer than three distinct non-zero levels"
        return res
    m = lambda L: np.isin(round6(y), round6(L))
    idx = dict(auth=y == 0, low=m(low), mid=m(mid), high=m(high))

    # ---------------------------------------------------------------- geometry
    geo = {}
    for key, base_mask in [("from_HIGH", idx["auth"] | idx["high"]),
                           ("from_LOW", idx["auth"] | idx["low"])]:
        Xb, yb = X[base_mask], y[base_mask]
        V = whitener(Xb, yb)
        P = lambda Z: Z @ V
        within = float(np.mean([np.linalg.norm(P(Xb[yb == c]) - P(Xb[yb == c]).mean(0), axis=1).mean()
                                for c in np.unique(yb) if (yb == c).sum() > 1]))
        mus = {k: P(X[idx[k]]).mean(0) for k in ("auth", "low", "high")}
        dl = float(np.linalg.norm(mus["low"] - mus["auth"]))
        dh = float(np.linalg.norm(mus["high"] - mus["auth"]))
        dv = lambda a, b: mus[b] - mus[a]
        cos = float(np.dot(dv("auth", "low"), dv("auth", "high")) / (dl * dh + 1e-12))
        unit_h = dv("auth", "high") / (dh + 1e-12)
        rho = float(spearmanr(y, P(X) @ unit_h).statistic)
        pc = PCA(n_components=int(min(20, Xb.shape[0] - 1, Xb.shape[1])), random_state=0).fit(Xb)
        v = X[idx["low"]].mean(0) - X[idx["auth"]].mean(0)      # centring-free projection
        vp = pc.components_.T @ (pc.components_ @ v)
        kept = float(np.linalg.norm(vp) / (np.linalg.norm(v) + 1e-12))
        steps = {}
        for L in np.unique(y[y > 0]):
            mm = np.where(round6(y) == round(float(L), 6))[0]
            if len(mm) > 1:
                steps[str(float(L))] = round(float(np.linalg.norm(P(X[mm]).mean(0) - mus["auth"])), 3)
        geo[key] = dict(within_norm=round(within, 3),
                        d_LOW_over_within=round(dl / within, 3),
                        d_HIGH_over_within=round(dh / within, 3),
                        ratio_LOW_over_HIGH=round(dl / dh, 3),
                        dose_ratio_low_high=round(float(np.mean(low) / np.mean(high)), 3),
                        cos_LOW_HIGH=round(cos, 4),
                        subspace_kept_LOW=round(kept, 4),
                        spearman_dose_vs_projection=round(rho, 4),
                        level_distances=steps)

    # ---------------------------------------------------------------- transfer
    res["geometry"] = geo
    res["gate"] = gate(X, g, y, idx)
    res["quantification"] = quantification(X, g, y, np.where(idx["high"])[0], np.where(idx["low"])[0])
    return res


if __name__ == "__main__":
    recs = {}
    for nm in PP.ALL:
        try:
            r = run(nm)
        except Exception as e:
            r = dict(name=nm, error="%s: %s" % (type(e).__name__, e))
        recs[nm] = r
        g1 = (r.get("geometry") or {}).get("from_HIGH", {})
        keys = ["d_LOW_over_within", "d_HIGH_over_within", "ratio_LOW_over_HIGH",
                "dose_ratio_low_high", "cos_LOW_HIGH", "subspace_kept_LOW",
                "spearman_dose_vs_projection"]
        print("%-4s n=%4d bands=%s %s" % (nm, r.get("n", 0),
              [len(r.get("bands", {}).get(k, [])) for k in ("low", "mid", "high")],
              {k: g1.get(k) for k in keys} if g1 else (r.get("skipped") or r.get("error"))))
        gt = r.get("gate", {})
        for pre in ("raw", "snv"):
            a = gt.get(pre, {})
            if "LOW" in a:
                print("     gate %-4s fa=%s LOW auc=%s hit=%s | MID auc=%s | HIGH auc=%s hit=%s" %
                      (pre, a["false_alarm_test_auth"], a["LOW"]["auc"], a["LOW"]["hit"],
                       a.get("MID", {}).get("auc"), a["HIGH"]["auc"], a["HIGH"]["hit"]))
        q = r.get("quantification", {})
        for pre, o in q.items():
            print("     quant %-4s down rmse=%s rel_bias=%s%% | inter rmse=%s rel_bias=%s%%" %
                  (pre, o["extrapolate_down"]["rmse"], o["extrapolate_down"]["rel_bias_pct"],
                   o["interpolate_same_rows"]["rmse"], o["interpolate_same_rows"]["rel_bias_pct"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
