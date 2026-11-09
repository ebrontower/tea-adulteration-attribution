"""exp40 - the full attribution pipeline on a PUBLIC multi-adulterant ATR-FTIR corpus.

Why this script exists
----------------------
Every certified-set number in the manuscript came from the tea spectra acquired for this study,
and the seven public corpora were used only for the geometry diagnosis (Section 3.10). Six of those
seven are dilution series of a single adulterant, so the composition-attribution task is empty on
them: there is exactly one candidate component and the set can only be {none, that one}. The
cinnamon deposition is the exception - walnut shell, peanut and wood dust each adulterate the same
spice matrix, in their own 0-24 % series, on the same instrument. That is the only public setting
in which the question this paper asks ("which component, with what confidence") can be asked of
somebody else's data.

What is run here, unchanged from the tea pipeline: the increment dictionary, the truncated whitened
metric, split-conformal composition sets, the locally normalised abstention gate, the
leave-one-adulterant-out open-set test, block-cluster bootstrap intervals and the geometry numbers.
The sets are reported on both score arms: the tea protocol puts the guarantee on the standard
(un-normalised) whitened score and uses the locally normalised score for the gate, so a transfer
claim has to state which arm it is made on.

Protocol mirrors the tea one: 900-1800 cm^-1, contiguous five-scan blocks inside each
(series, level) group of 50 spectra, fold k tests blocks = k (mod 5), fit blocks < 5, calibration
blocks >= 5, alpha = 0.10, quantile at min{1, (1-alpha)(1+1/n)}, k_w = 60 with a mean-tail floor,
local scale from the 5 nearest fit spectra.

Two residual groupings for the metric are reported. `white_a` pools residuals by adulterant, which
is the exact analogue of the tea formulations, where the added amount inside a formulation was
never recorded and is therefore part of the residual. `white_b` pools by adulterant x level, so the
residual is measurement noise only and the concentration direction stays in the metric.

The public corpus contains no mixture spectra, so the composite and ternary analyses cannot be
repeated here; they stay specific to the tea set.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

WINDOW = (900.0, 1800.0)
K_WHITE = 60
K_LOCAL = 5
ALPHA = 0.10
NF = 5
SERIES = ((1, "D7w", "walnut shell"), (2, "D7p", "peanut"), (3, "D7d", "wood dust"))
NAME = [s[2] for s in SERIES]
OUT_DIR = p5_paths.RESULTS
os.makedirs(OUT_DIR, exist_ok=True)
OUT = OUT_DIR + "/exp40_public.json"


# --------------------------------------------------------------------------- data
def load_cinnamon():
    """X, adulterant code (0 pure / 1..3), level %, five-scan block id, series code."""
    Xs, comps, lvls, blks, srs = [], [], [], [], []
    n_dup = 0
    for code, key, _label in SERIES:
        d = P.load_public(key)
        ax = np.asarray(d["wvn"], float)
        m = (ax >= WINDOW[0]) & (ax <= WINDOW[1])
        X = np.asarray(d["X"], float)[:, m]
        lv = np.asarray(d["y"], float)
        nm = np.asarray(d["groups"], dtype=object).astype(str)
        o = np.argsort(nm, kind="stable")             # spectrum name carries the run order
        X, lv, nm = X[o], lv[o], nm[o]

        seen = set()
        for j in range(len(X)):
            k = tuple(np.round(X[j], 9).tolist())
            n_dup += int(k in seen)
            seen.add(k)

        counts = {}
        bl = np.zeros(len(X), int)
        for j in range(len(X)):
            g = float(lv[j])
            i = counts.get(g, 0)
            counts[g] = i + 1
            bl[j] = i // 5                            # 50 spectra per level -> blocks 0..9
        Xs.append(X)
        comps.append(np.where(lv > 0, code, 0))        # level 0 in each file is that series' pure control
        lvls.append(lv)
        blks.append(bl)
        srs.append(np.full(len(X), code))
    return (np.concatenate(Xs, 0), np.concatenate(comps, 0), np.concatenate(lvls, 0),
            np.concatenate(blks, 0), np.concatenate(srs, 0), n_dup, Xs[0].shape[1])


def true_set(c):
    return () if int(c) == 0 else (int(c) - 1,)


def subsets(n):
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def set_label(S, labels=None):
    labels = labels or NAME
    return "pure" if not S else "+".join(labels[i] for i in S)


# --------------------------------------------------------------------------- model
def fit_dict(X, comp, idx, n_atoms=3):
    """Pure mean plus one increment per adulterant, both estimated on idx only."""
    m0 = X[idx][comp[idx] == 0].mean(0)
    mus = {0: m0}
    for c in range(1, n_atoms + 1):
        mus[c] = X[idx][comp[idx] == c].mean(0)
    D = np.stack([mus[c] - m0 for c in range(1, n_atoms + 1)], 1)
    res = np.concatenate([X[idx][comp[idx] == c] - mus[c] for c in mus], 0)
    return m0, D, float(np.sqrt(np.mean(res ** 2)))


def pred(m0, D, S):
    return m0 + D[:, list(S)].sum(1) if S else m0.copy()


class Whitener:
    """Pooled within-group residual covariance, truncated at k components with a mean-tail floor."""

    def __init__(self, X, cls, idx, k=K_WHITE):
        mu = {int(c): X[idx][cls[idx] == c].mean(0) for c in np.unique(cls[idx])}
        R = np.concatenate([X[idx][cls[idx] == c] - mu[c] for c in mu], 0)
        R = R - R.mean(0)
        _, S, Vt = np.linalg.svd(R, full_matrices=False)
        lam = S ** 2 / max(1, len(R) - 1)
        k = int(min(k, len(lam)))
        self.V = Vt[:k].T
        self.inv = 1.0 / np.sqrt(np.maximum(lam[:k], 1e-10))
        self.delta = float(np.mean(lam[k:])) if len(lam) > k else float(lam[-1])
        self.dim = k
        self.evr = float(np.sum(lam[:k]) / np.sum(lam))

    def coords(self, V):
        proj = V @ self.V
        tail = V - proj @ self.V.T
        return np.hstack([proj * self.inv[None, :], tail / np.sqrt(self.delta)])

    def dist(self, V):
        return np.linalg.norm(self.coords(V), axis=1)


def make_metric(name, X, cls_a, cls_b, idx):
    if name == "euclid":
        return None
    return Whitener(X, cls_a if name == "white_a" else cls_b, idx)


# --------------------------------------------------------------------------- geometry
def geometry(X, comp, cls_a, cls_b, idx):
    m0, D, sigma = fit_dict(X, comp, idx)
    iu = np.triu_indices(D.shape[1], 1)
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    coh_raw = float((Dn.T @ Dn)[iu].max())
    gauge_raw = (D.T @ m0) / (np.linalg.norm(D, axis=0) * np.linalg.norm(m0))
    corr_raw = [[round(float(v), 3) for v in row] for row in (Dn.T @ Dn)]
    out = dict(n_atoms=int(D.shape[1]), coh_raw=round(coh_raw, 3),
               cond_raw=round(float(np.linalg.cond(Dn.T @ Dn)), 2),
               corr_raw=corr_raw,
               gauge_raw=[round(float(v), 3) for v in gauge_raw],
               sigma_rms=round(sigma, 6),
               incr_norm=[round(float(np.linalg.norm(D[:, j])), 3)
                          for j in range(D.shape[1])],
               incr_snr=[round(float(np.linalg.norm(D[:, j]) / sigma), 2)
                         for j in range(D.shape[1])])
    for nm, wh in (("white_a", make_metric("white_a", X, cls_a, cls_b, idx)),
                   ("white_b", make_metric("white_b", X, cls_a, cls_b, idx))):
        W = wh.coords(D.T)
        wn = np.linalg.norm(W, axis=1)
        Gn = (W @ W.T) / np.outer(wn, wn)
        Wm = wh.coords(m0[None, :])[0]
        gauge = (W @ Wm) / (wn * np.linalg.norm(Wm))
        out[nm] = dict(coh=round(float(Gn[iu].max()), 3),
                       cond=round(float(np.linalg.cond(Gn)), 2),
                       gauge=[round(float(v), 3) for v in gauge],
                       absmax_gauge=round(float(np.max(np.abs(gauge))), 3),
                       evr=round(wh.evr, 5), dim=wh.dim,
                       incr_norm=[round(float(w), 3) for w in wn],
                       incr_snr=[round(float(w / sigma), 2) for w in wn])
    return out


# --------------------------------------------------------------------------- one fold
def local_scale(wh, X, fit, i):
    V = X[i][None, :] - X[fit]
    d = wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)
    k = max(1, min(K_LOCAL, len(d)))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def _sq_dists(A, B):
    """(len(A), len(B)) matrix of squared Euclidean distances between rows of A and B."""
    return np.maximum((A * A).sum(1, keepdims=True) + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T), 0.0)


def fold_rows(X, comp, level, block, cls_a, cls_b, fit, cal, test, cand, atoms,
              alpha, metric, gate_cal, labels=None):
    """Per-test-spectrum records for one fold. `atoms` lists the material codes in the dictionary."""
    labels = labels or NAME
    m0 = X[fit][comp[fit] == 0].mean(0)
    mus = {0: m0}
    for c in atoms:
        mus[c] = X[fit][comp[fit] == c].mean(0)
    D = np.stack([mus[c] - m0 for c in atoms], 1)
    wh = make_metric(metric, X, cls_a, cls_b, fit)
    cand_m = np.stack([pred(m0, D, S) for S in cand])
    S_of = {S: j for j, S in enumerate(cand)}

    Z = X if wh is None else wh.coords(X)                 # metric coordinates, computed once
    Zc = cand_m if wh is None else wh.coords(cand_m)
    k = max(1, min(K_LOCAL, len(fit)))
    loc = np.sqrt(np.partition(_sq_dists(Z, Z[fit]), k - 1, axis=1)[:, :k]).mean(1)
    d2 = _sq_dists(Z, Zc)

    def dists(i):
        return np.sqrt(d2[i])

    def truth(i):
        c = int(comp[i])
        if c == 0:
            return ()
        return (atoms.index(c),) if c in atoms else None

    loc_cal = loc[cal]
    loc_test = loc[test]
    # two score arms: the standard score carries the guarantee in the tea protocol, the
    # locally normalised score drives the gate; both are reported so the arms are comparable.
    cal_raw = np.array([dists(i)[S_of[truth(i)]] for i in cal])
    q_std = float(np.quantile(cal_raw, min(1.0, (1 - alpha) * (1 + 1 / len(cal_raw)))))
    cal_sc = cal_raw / loc_cal
    q = float(np.quantile(cal_sc, min(1.0, (1 - alpha) * (1 + 1 / len(cal_sc)))))

    gc = np.asarray(gate_cal)
    pos = {int(i): j for j, i in enumerate(cal)}
    best_cal = np.array([float(np.min(dists(i))) / loc_cal[pos[int(i)]] for i in gc])
    tau = float(np.quantile(best_cal, min(1.0, (1 - alpha) * (1 + 1 / len(best_cal)))))
    abs_cal = np.array([float(np.min(dists(i))) for i in gc])
    tau_abs = float(np.quantile(abs_cal, min(1.0, (1 - alpha) * (1 + 1 / len(abs_cal)))))

    rows = []
    for j, i in enumerate(test):
        raw = dists(i)
        sc = raw / loc_test[j]
        C = np.where(sc <= q)[0]
        Cs = np.where(raw <= q_std)[0]
        tset = truth(i)
        kt = S_of[tset] if tset is not None else -1
        order = np.argsort(sc)
        order_s = np.argsort(raw)
        rank = int(np.where(order == kt)[0][0]) if kt >= 0 else len(cand)
        rank_std = int(np.where(order_s == kt)[0][0]) if kt >= 0 else len(cand)
        best = float(sc[order[0]])
        rows.append(dict(comp=int(comp[i]), level=float(level[i]),
                         cell=(int(comp[i]), float(level[i]), int(block[i])),
                         covered=bool(kt >= 0 and kt in C), size=int(len(C)),
                         empty=bool(len(C) == 0),
                         covered_std=bool(kt >= 0 and kt in Cs), size_std=int(len(Cs)),
                         empty_std=bool(len(Cs) == 0),
                         rank=rank, rank_std=rank_std, best=best, best_q=float(raw[order[0]]),
                         best_set=set_label(cand[order[0]], labels),
                         accepted=bool(best <= tau),
                         accepted_two=bool(best <= tau and float(np.min(raw)) <= tau_abs)))
    return rows, q, tau, tau_abs, q_std


# --------------------------------------------------------------------------- splits
def blocked_folds(block):
    triples = []
    for k in range(NF):
        test = np.sort(np.where(block % NF == k)[0])
        rest = np.where(block % NF != k)[0]
        triples.append((np.sort(rest[block[rest] < 5]), np.sort(rest[block[rest] >= 5]), test))
    return triples


def random_folds(comp, rng):
    folds = [[] for _ in range(NF)]
    for c in np.unique(comp):
        idx = np.where(comp == c)[0].copy()
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, NF)):
            folds[k] += list(chunk)
    triples = []
    for k in range(NF):
        test = np.array(sorted(folds[k]))
        rest = np.array(sorted(set(range(len(comp))) - set(test)))
        n_fit = int(round(0.7 * len(rest)))
        order = np.argsort(rng.random(len(rest)))
        triples.append((np.sort(rest[order[:n_fit]]), np.sort(rest[order[n_fit:]]), test))
    return triples


# --------------------------------------------------------------------------- summaries
def agg(rr, keys=("covered", "size", "empty", "rank")):
    kc, ks, ke, kr = keys
    if not rr:
        return None
    acc = [r for r in rr if r["accepted"]]
    return dict(n=len(rr),
                coverage=round(float(np.mean([r[kc] for r in rr])), 3),
                size=round(float(np.mean([r[ks] for r in rr])), 2),
                top1=round(float(np.mean([r[kr] == 0 for r in rr])), 3),
                abstain=round(float(np.mean([r[ke] for r in rr])), 3),
                accept=round(float(np.mean([r["accepted"] for r in rr])), 3),
                accept_two=round(float(np.mean([r["accepted_two"] for r in rr])), 3),
                n_accepted=len(acc),
                coverage_acc=(round(float(np.mean([r[kc] for r in acc])), 3) if acc else None),
                size_acc=(round(float(np.mean([r[ks] for r in acc])), 2) if acc else None),
                top1_acc=(round(float(np.mean([r[kr] == 0 for r in acc])), 3) if acc else None))


STD_KEYS = ("covered_std", "size_std", "empty_std", "rank_std")


def summarise(rows, keys=("covered", "size", "empty", "rank")):
    out = {"all": agg(rows, keys)}
    out["by_adulterant"] = {NAME[c - 1]: agg([r for r in rows if r["comp"] == c], keys)
                            for c in (1, 2, 3)}
    out["pure"] = agg([r for r in rows if r["comp"] == 0], keys)
    out["by_level"] = {lbl: agg([r for r in rows if lo <= r["level"] <= hi], keys)
                       for lbl, (lo, hi) in (("1-4%", (1.0, 4.0)), ("6-12%", (6.0, 12.0)),
                                             ("15-24%", (15.0, 24.0)))}
    return out


def variant(rows, arm):
    """Standard-arm records re-keyed onto the canonical names, for the bootstrap and the seeds."""
    if arm == "norm":
        return rows
    return [{**r, "covered": r["covered_std"], "size": r["size_std"],
             "empty": r["empty_std"], "rank": r["rank_std"]} for r in rows]


def run_arm(X, comp, level, block, cls_a, cls_b, cand, atoms, splits, metric):
    rows, qs, taus, tabs, qstd = [], [], [], [], []
    for fit, cal, test in splits:
        gate_cal = cal[comp[cal] > 0]
        rr, q, tau, tau_abs, q_s = fold_rows(X, comp, level, block, cls_a, cls_b, fit, cal, test,
                                             cand, atoms, ALPHA, metric, gate_cal)
        rows += rr
        qs.append(q)
        taus.append(tau)
        tabs.append(tau_abs)
        qstd.append(q_s)
    out = {"std": summarise(rows, STD_KEYS), "norm": summarise(rows)}
    for s in out.values():
        s["q_per_fold"] = [round(v, 3) for v in qs]
        s["q_std_per_fold"] = [round(v, 3) for v in qstd]
        s["tau_per_fold"] = [round(v, 3) for v in taus]
    return out, rows


def cluster_boot(rows, keys, n_boot=2000, seed=0):
    """Resample (adulterant, level, block) cells rather than individual spectra."""
    rng = np.random.default_rng(seed)
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault(r["cell"], []).append(i)
    groups = list(cells.values())
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = [i for g in pick for i in groups[g]]
        sub = [rows[i] for i in idx]
        for k in keys:
            if k == "size":
                out[k].append(float(np.mean([r["size"] for r in sub])))
            elif k == "coverage":
                out[k].append(float(np.mean([r["covered"] for r in sub])))
            elif k == "top1":
                out[k].append(float(np.mean([r["rank"] == 0 for r in sub])))
            elif k == "abstain":
                out[k].append(float(np.mean([r["empty"] for r in sub])))
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
            for k, v in out.items()}


def open_set_run(X, comp, level, block, cls_a, cls_b, held, alpha=ALPHA):
    """Dictionary rebuilt from pure cinnamon and the other two adulterants only."""
    atoms = [c for c in (1, 2, 3) if c != held]
    labels = [NAME[c - 1] for c in atoms]
    cand = subsets(len(atoms))
    held_idx = np.where(comp == held)[0]
    other_idx = np.where(comp != held)[0]
    rows = []
    for k in range(NF):
        test = np.sort(held_idx[block[held_idx] % NF == k])
        rest = other_idx[block[other_idx] % NF != k]
        fit = np.sort(rest[block[rest] < 5])
        cal = np.sort(rest[block[rest] >= 5])
        gate_cal = np.sort(cal[comp[cal] > 0])          # in-dictionary adulterants only
        rr, *_ = fold_rows(X, comp, level, block, cls_a, cls_b, fit, cal, test,
                           cand, atoms, alpha, "white_a", gate_cal, labels=labels)
        rows += rr
    s = summarise(rows)
    s["all_std"] = summarise(rows, STD_KEYS)["all"]
    s["n_test"] = len(rows)
    s["returned_on_accept"] = {}
    for lbl, cnt in _counts([r["best_set"] for r in rows if r["accepted"]]):
        s["returned_on_accept"][lbl] = cnt
    return s


def _counts(seq):
    c = {}
    for v in seq:
        c[v] = c.get(v, 0) + 1
    return sorted(c.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------- main
def main():
    X, comp, level, block, series, n_dup, nch = load_cinnamon()
    cls_a = comp.copy()
    cls_b = np.zeros(len(comp), int)
    for s in (1, 2, 3):
        for lv in np.unique(level[series == s]):
            cls_b[(series == s) & (level == lv)] = 100 * s + int(lv)

    cand = subsets(3)
    atoms = [1, 2, 3]
    out = dict(window=list(WINDOW), n_channels=int(nch), n_spectra=int(len(comp)),
               duplicate_rows=n_dup, levels=sorted(set(int(v) for v in level)),
               per_class={NAME[c - 1]: int((comp == c).sum()) for c in (1, 2, 3)},
               pure=int((comp == 0).sum()), n_candidates=len(cand), alpha=ALPHA,
               k_white=K_WHITE, k_local=K_LOCAL,
               source="D7w/D7p/D7d cinnamon (walnut shell, peanut, wood dust), 0-24% m/m, "
                      "three independent pure sets of 50")

    bf = blocked_folds(block)
    out["role_sizes_per_fold"] = dict(fit=[len(f) for f, c, t in bf],
                                      cal=[len(c) for f, c, t in bf],
                                      test=[len(t) for f, c, t in bf])
    out["geometry"] = geometry(X, comp, cls_a, cls_b, np.arange(len(comp)))

    arms, store = {}, {}
    for metric in ("euclid", "white_a", "white_b"):
        res, rows = run_arm(X, comp, level, block, cls_a, cls_b, cand, atoms, bf, metric)
        for score in ("std", "norm"):
            arms[f"blocked/{metric}/{score}"] = res[score]
        store[metric] = rows
    rf = random_folds(comp, np.random.default_rng(0))
    for metric in ("euclid", "white_a"):
        res, _ = run_arm(X, comp, level, block, cls_a, cls_b, cand, atoms, rf, metric)
        for score in ("std", "norm"):
            arms[f"random/{metric}/{score}"] = res[score]
    out["arms"] = arms
    out["intervals"] = {}
    for metric in ("euclid", "white_a", "white_b"):
        for score in ("std", "norm"):
            out["intervals"][f"blocked/{metric}/{score}"] = cluster_boot(
                variant(store[metric], score), ["coverage", "size", "top1", "abstain"])

    wa = store["white_a"]
    out["confusion_white_a"] = {NAME[c - 1]:
                                dict(_counts([r["best_set"] for r in wa if r["comp"] == c]))
                                for c in (1, 2, 3)}

    out["open_set"] = {NAME[h - 1]: open_set_run(X, comp, level, block, cls_a, cls_b, h)
                       for h in (1, 2, 3)}

    seeds = {}
    for sd in range(10):
        res, _ = run_arm(X, comp, level, block, cls_a, cls_b, cand, atoms,
                         random_folds(comp, np.random.default_rng(100 + sd)), "white_a")
        a = res["std"]["all"]
        seeds[sd] = dict(coverage=a["coverage"], top1=a["top1"], size=a["size"])
    cov = [v["coverage"] for v in seeds.values()]
    t1 = [v["top1"] for v in seeds.values()]
    out["random_split_seeds"] = dict(arm="white_a, standard score, random split",
                                     mean_cov=round(float(np.mean(cov)), 3),
                                     sd_cov=round(float(np.std(cov)), 3),
                                     range_cov=[round(min(cov), 3), round(max(cov), 3)],
                                     mean_top1=round(float(np.mean(t1)), 3),
                                     sd_top1=round(float(np.std(t1)), 3))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("cinnamon public corpus:", out["n_spectra"], "spectra,", nch, "channels,",
          "duplicate rows", n_dup, "| classes", out["per_class"], "pure", out["pure"])
    g = out["geometry"]
    print(f"geometry  raw coh {g['coh_raw']:.3f} cond {g['cond_raw']:.2f} "
          f"gauge {g['gauge_raw']} | white_a coh {g['white_a']['coh']:.3f} "
          f"cond {g['white_a']['cond']:.2f} gauge {g['white_a']['gauge']} "
          f"evr {g['white_a']['evr']}")
    print(f"{'arm':22s} {'cover':>6s} {'size':>6s} {'top1':>6s} {'abst':>6s} {'accept':>7s} | "
          f"{'w/nu/pu top1':>16s} | {'1-4%':>6s} {'6-12%':>6s} {'15-24%':>7s}")
    for k, v in arms.items():
        a = v["all"]
        bl = [v["by_level"][b]["top1"] if v["by_level"][b] else float("nan")
              for b in ("1-4%", "6-12%", "15-24%")]
        tops = " ".join(f"{v['by_adulterant'][nm]['top1']:.2f}" for nm in NAME)
        print(f"{k:22s} {a['coverage']:6.3f} {a['size']:6.2f} {a['top1']:6.3f} "
              f"{a['abstain']:6.3f} {a['accept']:7.3f} | {tops:>16s} | "
              + " ".join(f"{b:6.2f}" for b in bl))
    # accepted-only statistics are reported for the standard-score arms: for the normalised arm
    # the gate and the set threshold use the same score, so conditional coverage is trivially 1.
    print("gate (locally normalised best residual, calibrated on in-dictionary adulterated spectra):")
    for k in ("blocked/euclid/std", "blocked/white_a/std", "blocked/white_b/std"):
        a = arms[k]["all"]
        byadm = " ".join(f"{nm[:3]}={arms[k]['by_adulterant'][nm]['accept']:.2f}" for nm in NAME)
        print(f"  {k:22s} accept {a['accept']:6.3f} | accepted-only cov {a['coverage_acc']} "
              f"size {a['size_acc']} top1 {a['top1_acc']} (n {a['n_accepted']}) | {byadm}")
    print("intervals (block-cluster bootstrap, 2000 draws):")
    for m, v in out["intervals"].items():
        print(f"  {m:22s}", json.dumps(v))
    print("open set (dictionary without the material):")
    for nm, v in out["open_set"].items():
        print(f"  {nm:15s} n {v['n_test']:3d} abstain {1 - v['all']['accept']:.2f} "
              f"two-sided {1 - v['all']['accept_two']:.2f} "
              f"returned {v['returned_on_accept']}")
    print("random-split seeds:", json.dumps(out["random_split_seeds"]))
    print("confusion (white_a, predicted set | truth):",
          json.dumps(out["confusion_white_a"], ensure_ascii=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
