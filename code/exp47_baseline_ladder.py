"""exp47 - the information-budget ladder of Section 3.12.

One script, three parts. Each part asks what a laboratory would reach without one of the three
ingredients this pipeline adds - the dictionary, the whitened metric, the conformal layer - and gives
every arm the training rows and the selection rights its rung licenses, so that a difference in the
outcome is a difference in information and not a difference in effort.

PART A - refusal of the unseen, against dictionary-free detectors.
    "Is this outside what I have seen" is a novelty-detection question, and novelty detection needs no
    additive dictionary: Hotelling T2 / MSP, the SPE(Q) residual, one-class SVM, Mahalanobis, the
    distance to the authentic class, a whitened nearest-neighbour score, a PCA residual and an
    unlabelled autoencoder residual all answer it. Part A runs that ladder against the locally
    normalised gate on two orphan definitions (a held-out adulterant, a held-out composite) and at two
    reference pools (singles-only, and the whole published fit block), and pairs every detector with
    the SAME attribution engine so that only the refusal rule varies. Every detector is allowed to see
    the pure-tea mean and the pure-tea covariance, which is the most generous prior-free start.

PART B - the metric x model-class factorial.
    Representations: raw absorbance, SNV, the truncated whitened coordinates (Whitener.coords, the
    same feature map the paper's distance uses), and the 60 whitened directions alone.
    Model classes: unconstrained linear (LDA, multinomial logistic) - truncated linear (PLS-DA,
    PCA-LDA, one-vs-rest PLS2) - constrained linear (non-negative dictionary least squares, and
    subset-argmin dictionary fitting with and without whitening) - distance/kernel (kNN, SVM-RBF) -
    trees (random forest, gradient boosting) - deep (MLP, 1-D CNN).
    Tasks: T1 nine-class discrimination (pure + six singles + the two held-in composites, exactly
    the task whose balanced accuracy Section 3.2 reports);
    T2 held-out composite exact-set recovery (train on pure + six singles, test on the two
    composites - the composites are expressible in the label space but absent from training). Its
    dictionary rows come in two flavours: "nls" chooses its support cut as the best over a ladder
    computed on the composites themselves (an oracle, reported only as an upper bound), "nls_cal"
    chooses the cut on calibration singles under a rule fixed in advance (the quotable number).
    Same drift-blocked folds, same window, same alpha, same dictionary definition as Sections
    2.5-2.6; tea first, then the cinnamon deposition for the T1 half only (it has no mixtures).

PART C - the learned representation inside the same factorial.
    Section 3.10's own encoder (same architecture, same 80-epoch budget, same standardisation) drops
    onto the representation axis beside raw / SNV / whitened, retrained per fold on the unlabelled
    training spectra only - never on a held-out block, with a runtime guard on that - plus a
    random-initialisation control at identical capacity. The analytic metric and the learned
    representation are therefore compared in one factorial rather than in two separate sections.

PRE-REGISTERED DECISION RULES (written before looking at any number in this script)
  A1. If any prior-free detector reaches an orphan-vs-in-set AUROC within 0.02 of the gate, OR
      rejects a higher share of orphans at the same 10% false-abstention on singles, the abstention
      half of contribution 3 is downgraded to "a conformal threshold that also attributes", and the
      gate must be justified by its joint behaviour (refuse + cover), not by refusal alone.
  B1. If any model class on any representation exceeds 0.10 exact-set on held-out composites,
      Section 3.3 and the abstract sentence are rewritten from "not an artefact of one model
      family" to "not an artefact of one linear model family", and the failure becomes a claim about
      training-label coverage rather than about model capacity.
  B2. Whitening is a fixed invertible linear map, so an UNCONSTRAINED linear discriminant is provably
      unaffected. LDA(svd) balanced accuracy must therefore agree across raw and whitened coordinates
      up to numerical noise; a gap over 0.02 means the feature map is wrong (leakage or
      non-invertibility) and every Part B number is void until re-run. This is a bug check on
      ourselves, not a result. It covers LDA alone: an L2-penalised logistic, an MLP, an RBF kernel
      and a distance classifier are scale-SENSITIVE, so for them raw vs whitened is a genuine
      regularisation-by-metric interaction, not an invariance. To keep that interaction the only
      difference between arms, every scale-sensitive learner gets per-channel standardisation fitted
      on its own training rows in every representation (see SCALED) - otherwise "whitening helps"
      would partly be "the raw arm was denied a preprocessor".
  B3. "The prior helps across model classes" is earned only if the whitened representation improves
      the mean over the non-invariant classes (truncated, constrained, distance/kernel, trees, deep)
      on tea AND the direction of the effect replicates on the cinnamon corpus. If it helps only our
      own distance pipeline, the wording is scoped to distance-based attribution and nothing broader
      is claimed.
  C1. If a self-supervised representation reaches exact-set > 0.10 on held-out composites for any
      model class, Section 3.10's "the representation is not the bottleneck" is confined to
      discrimination and may not be quoted for attribution. If SSL matches the analytic
      representations on T1 but stays at or below them on T2, Sections 3.10 and 3.3 collapse into
      one statement: learned representations buy no attribution either.
Pure computation on already-acquired spectra; no new bench measurement, no new sample.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p5_data as P  # noqa: E402,F401  (loader cache for exp33)
import exp33_whitened_attr as E  # noqa: E402

OUT = os.environ.get("P47_OUT") or \
    str(Path(__file__).resolve().parent / "out" / "exp47_baseline_ladder.json")
ALPHA = 0.10
K_LOCAL = 5
NF = 5
SEED = 0
DO_TEA = os.environ.get("P47_TEA", "1") == "1"
DO_CIN = os.environ.get("P47_CIN", "1") == "1"
DO_CNN = os.environ.get("P47_CNN", "1") == "1"
DO_PART_A = os.environ.get("P47_A", "1") == "1"
DO_SSL = os.environ.get("P47_SSL", "0") == "1"   # slowest stage: one encoder per fold

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors  # noqa: E402
from sklearn.svm import SVC, OneClassSVM  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from scipy.optimize import nnls  # noqa: E402


# --------------------------------------------------------------------------- helpers
def auc(pos, neg):
    """P(score_pos > score_neg), rank-based, tie-safe (same estimator as exp36)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        k = i
        while k + 1 < len(sv) and sv[k + 1] == sv[i]:
            k += 1
        ranks[order[i:k + 1]] = 0.5 * (i + k) + 1
        i = k + 1
    rp = ranks[:len(pos)].sum()
    return round(float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def local_scale(wh, Xfit, x, k=K_LOCAL):
    V = x[None, :] - Xfit
    d = wh.dist(V) if wh is not None else np.linalg.norm(V, axis=1)
    k = max(1, min(k, len(d)))
    return float(np.mean(np.partition(d, k - 1)[:k]))


def q_level(vals, alpha=ALPHA):
    n = len(vals)
    return float(np.quantile(vals, min(1.0, (1 - alpha) * (1 + 1 / n)))) if n else None


def r3(v):
    if v is None or isinstance(v, (bool, np.bool_)):
        return v if v is None else bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return round(float(v), 3)


def snv(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True)
    return (X - m) / np.maximum(s, 1e-12)


# --------------------------------------------------------------------------- representations
def rep_matrix(name, Xraw, wh, Xfit_raw=None):
    """Feature matrix for a representation, computed only from information available at fit time."""
    if name == "raw":
        return Xraw
    if name == "snv":
        return snv(Xraw)
    if name == "white":
        return wh.coords(Xraw)
    if name == "whitek":
        return wh.coords(Xraw)[:, :wh.dim]
    raise KeyError(name)


def rep_dist(name, wh, V):
    """Distance of rows of V to a single reference row already in the same representation."""
    if name == "white" or name == "whitek":
        return wh.dist(V)
    return np.linalg.norm(V, axis=1)


# --------------------------------------------------------------------------- Part A
DET_NAMES = ["d2m0", "hotelling_t2", "spe_q", "maha_white", "ocsvm", "knn_all_raw",
             "knn_all_white", "knn_pure_raw", "ae_resid", "ours_gate", "ours_gate_abs",
             "ours_gate_pub", "knn_all_white_pub", "maha_white_pub", "hotelling_t2_pub",
             "spe_q_pub", "ocsvm_pub", "ae_resid_pub"]
# ours_gate          : whitener AND local-scale neighbourhood from the single-adulterant fit rows
#                      only (the honest setting when the composite is the thing being held out)
# ours_gate_pub      : both from the whole fit block, i.e. Section 3.7's published pool
# *_pub              : the same published pool handed to every prior-free detector, so the 100/100
#                      column is not our arm at one budget against baselines at another. d2m0 needs
#                      no twin: m0 is the pure-tea mean, the same row set in both pools.


def subsets_n(n):
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def dict_from(X, yr, idx, exclude=None):
    """Pure-tea mean plus one increment per single-adulterant formulation actually present.

    A leave-one-out dictionary really has five atoms, so the atom list travels with the dictionary
    and the candidate subsets are enumerated over the atoms that exist.
    """
    codes = [c for c in sorted(set(int(v) for v in yr[idx])) if 1 <= c <= 6 and c != exclude]
    mus = {0: X[idx][yr[idx] == 0].mean(0)}
    for c in codes:
        mus[c] = X[idx][yr[idx] == c].mean(0)
    m0 = mus[0]
    D = np.stack([mus[c] - m0 for c in codes], 1)
    res = np.concatenate([X[idx][yr[idx] == c] - mus[c] for c in mus], 0)
    return m0, D, float(np.sqrt(np.mean(res ** 2))), codes


def build_ctx(X, yr, fit, exclude=None, pub_fit=None):
    """Everything a detector may legitimately see, estimated on the fit indices only.

    pub_fit reproduces Section 3.7's published pool: the whole fit block, so composite spectra
    contribute to the residual covariance AND to the five nearest reference spectra used by the
    local scale. A 2x2 pool diagnostic showed that this neighbourhood pool - not the whitener - is
    what makes the published gate reject 100 of 100 composites, so the two pools are kept as
    separate arms rather than merged.
    """
    fit = fit[yr[fit] != exclude] if exclude is not None else fit
    wh = E.Whitener(X, yr, fit)
    m0 = X[fit][yr[fit] == 0].mean(0)
    Z = X[fit]
    mu_z = Z.mean(0)
    p = PCA(n_components=min(20, len(Z) - 1, Z.shape[1]), svd_solver="full").fit(Z - mu_z)
    ev = p.explained_variance_
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1)
    k = max(10, min(k, len(ev)))
    ctx = dict(X=X, yr=yr, fit=fit, wh=wh, m0=m0, Xfit=Z,
               pca=(mu_z, p.components_[:k], np.maximum(ev[:k], 1e-12)),
               ocsvm=OneClassSVM(nu=0.10, gamma="scale").fit(Z),
               ae=MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu",
                               max_iter=80, random_state=SEED, tol=1e-4).fit(Z, Z))
    if pub_fit is not None:
        pf = pub_fit[yr[pub_fit] != exclude] if exclude is not None else pub_fit
        pwh = E.Whitener(X, yr, pf)
        # the published pool must cost the prior-free competitor the same way it costs our gate,
        # otherwise the 100/100 column would compare our arm at one budget to baselines at another
        pref = pwh.coords(X[pf])
        pk = min(5, len(pref))
        P = X[pf]
        mu_p = P.mean(0)
        pp = PCA(n_components=min(20, len(P) - 1, P.shape[1]), svd_solver="full").fit(P - mu_p)
        ev_p = pp.explained_variance_
        kp = int(np.searchsorted(np.cumsum(ev_p) / ev_p.sum(), 0.90) + 1)
        kp = max(10, min(kp, len(ev_p)))
        ctx["pub"] = dict(wh=pwh, Xfit=X[pf],
                          knn=(NearestNeighbors(n_neighbors=pk).fit(pref), pk),
                          pca=(mu_p, pp.components_[:kp], np.maximum(ev_p[:kp], 1e-12)),
                          ocsvm=OneClassSVM(nu=0.10, gamma="scale").fit(P),
                          ae=MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu",
                                          max_iter=80, random_state=SEED, tol=1e-4).fit(P, P))
    for scope in ("all", "pure"):
        for metric in ("raw", "white"):
            if scope == "pure" and metric == "white":
                continue
            ref = Z if scope == "all" else Z[yr[fit] == 0]
            ref = wh.coords(ref) if metric == "white" else ref
            kk = min(5, len(ref))
            ctx[f"knn_{scope}_{metric}"] = (NearestNeighbors(n_neighbors=kk).fit(ref), kk)
    m0d, D, _, codes = dict_from(X, yr, fit, exclude)
    cand = subsets_n(len(codes))
    ctx["dict"] = dict(m0=m0d, D=D, codes=codes, loc={c: i for i, c in enumerate(codes)},
                       cand=cand,
                       cand_m=np.stack([E.pred(m0d, D, S) for S in cand]))
    return ctx


def det_scores(name, ctx, Xt):
    """Anomaly scores for the rows of Xt; every threshold is set later on calibration scores."""
    X, yr, fit, wh, Xfit = ctx["X"], ctx["yr"], ctx["fit"], ctx["wh"], ctx["Xfit"]
    m0 = ctx["m0"]
    if name == "d2m0":
        return np.linalg.norm(Xt - m0[None, :], axis=1)
    if name in ("hotelling_t2", "spe_q", "hotelling_t2_pub", "spe_q_pub"):
        mu, comps, lam = ctx["pub"]["pca"] if name.endswith("_pub") else ctx["pca"]
        V = Xt - mu[None, :]
        t = V @ comps.T
        if name.startswith("hotelling"):
            return np.sqrt((t * t / lam[None, :]).sum(1))
        return np.sqrt(np.maximum((V * V).sum(1) - (t * t).sum(1), 0.0))
    if name == "maha_white":
        return wh.dist(Xt - m0[None, :])
    if name == "ocsvm":
        return -ctx["ocsvm"].score_samples(Xt)
    if name == "ocsvm_pub":
        return -ctx["pub"]["ocsvm"].score_samples(Xt)
    if name == "knn_all_white_pub":
        pwh = ctx["pub"]["wh"]
        nn, kk = ctx["pub"]["knn"]
        d, _ = nn.kneighbors(pwh.coords(Xt), n_neighbors=kk)
        return d.mean(1)
    if name == "maha_white_pub":
        return ctx["pub"]["wh"].dist(Xt - m0[None, :])
    if name in ("knn_all_raw", "knn_all_white", "knn_pure_raw"):
        _, scope, metric = name.split("_")
        nn, kk = ctx[f"knn_{scope}_{metric}"]
        Q = wh.coords(Xt) if metric == "white" else Xt
        d, _ = nn.kneighbors(Q, n_neighbors=kk)
        return d.mean(1)
    if name in ("ae_resid", "ae_resid_pub"):
        model = ctx["pub"]["ae"] if name.endswith("_pub") else ctx["ae"]
        return np.linalg.norm(Xt - model.predict(Xt), axis=1)
    if name in ("ours_gate", "ours_gate_abs"):
        cm = ctx["dict"]["cand_m"]
        best = np.stack([wh.dist(x[None, :] - cm).min() for x in Xt])
        if name == "ours_gate_abs":
            return best
        ls = np.array([local_scale(wh, Xfit, x) for x in Xt])
        return best / ls
    if name == "ours_gate_pub":
        pwh, pfit = ctx["pub"]["wh"], ctx["pub"]["Xfit"]
        cm = ctx["dict"]["cand_m"]
        best = np.stack([pwh.dist(x[None, :] - cm).min() for x in Xt])
        ls = np.array([local_scale(pwh, pfit, x) for x in Xt])
        return best / ls
    raise KeyError(name)


def truth_index(code, d_):
    """Recipe code -> tuple of LOCAL dictionary columns; None when the fold cannot express it."""
    COMPOSITE = {7: (1, 4), 8: (2, 4, 6)}          # adulterant codes, from Sections 2.1 / 3.3
    if int(code) == 0:
        return ()
    if int(code) in COMPOSITE:
        cols = [d_["loc"].get(c) for c in COMPOSITE[int(code)]]
        return tuple(sorted(cols)) if all(c is not None for c in cols) else None
    c = d_["loc"].get(int(code))
    return (c,) if c is not None else None


def part_a_tea(X, yr, splits):
    NAMES = ["sugar", "talc", "sulfur", "paraffin", "dye", "flavouring"]
    res = {"gate_definition": "tau = (1-alpha)(1+1/n) quantile of the detector score on CALIBRATION "
                              "SINGLES; flag = score > tau; test scores are divided by their own "
                              "fold tau so the five folds are poolable and 1.0 is the operating point",
           "attribution_engine": "fixed for every arm: whitened dictionary conformal set at alpha="
                                 f"{ALPHA} plus argmin point attribution; only the refusal rule varies",
           "detectors": DET_NAMES, "orphans": {}}

    def one_orphan_setting(orphan_of, inset_of, exclude_of, singles_only_dict, names=None):
        # the *_pub twins are only informative where the two pools differ; in the leave-one-adulterant
        # setting the singles-only fit and the published fit are literally the same rows, so running
        # them would re-report one computation as two arms (verified: ours_gate = ours_gate_pub = 0.949)
        names = names if names is not None else DET_NAMES
        acc = {n: dict(orph=[], ins=[], a_orph=0, n_orph=0, a_ins=0, n_ins=0,
                       top1=[], cov=[], size=[]) for n in names}
        skips = {}
        for fi, (fit_idx, cal_idx, test_idx) in enumerate(splits):
            exclude = exclude_of(fi)
            used_fit = fit_idx[yr[fit_idx] <= 6] if singles_only_dict else fit_idx
            ctx = build_ctx(X, yr, used_fit, exclude, pub_fit=fit_idx)
            wh, d_ = ctx["wh"], ctx["dict"]
            S_of = {S: k for k, S in enumerate(d_["cand"])}
            cal_in = np.array([i for i in cal_idx if int(yr[i]) == 0 or
                               (1 <= int(yr[i]) <= 6 and int(yr[i]) != exclude)])
            # conformal threshold of the fixed attribution engine (identical for every arm)
            eng = []
            for i in cal_in:
                dv = wh.dist(X[i][None, :] - d_["cand_m"])
                truth = truth_index(int(yr[i]), d_)
                eng.append(float(dv[S_of[truth]]) / local_scale(wh, ctx["Xfit"], X[i]))
            q = q_level(eng)
            orph_idx = np.array([i for i in test_idx if orphan_of(int(yr[i]))])
            ins_idx = np.array([i for i in test_idx if inset_of(int(yr[i]))])
            for name in names:
                sc = det_scores(name, ctx, X[cal_in])
                t = q_level(sc)
                if t is None or t <= 0:
                    # an unusable operating point is a result, not a silent omission
                    skips.setdefault(name, []).append(fi)
                    continue
                so = det_scores(name, ctx, X[orph_idx]) / t
                si = det_scores(name, ctx, X[ins_idx]) / t
                a = acc[name]
                a["orph"] += list(so)
                a["ins"] += list(si)
                a["n_orph"] += len(orph_idx)
                a["n_ins"] += len(ins_idx)
                a["a_orph"] += int(np.sum(so <= 1.0))
                a["a_ins"] += int(np.sum(si <= 1.0))
                for i, s in zip(ins_idx, si):
                    if s > 1.0:
                        continue
                    dv = wh.dist(X[i][None, :] - d_["cand_m"])
                    dv = dv / local_scale(wh, ctx["Xfit"], X[i])
                    C = np.where(dv <= q)[0]
                    kt = S_of[truth_index(int(yr[i]), d_)]
                    a["top1"].append(float(int(np.argmin(dv)) == kt))
                    a["cov"].append(float(kt in C))
                    a["size"].append(float(len(C)))
        tab = {}
        for name in names:
            a = acc[name]
            sk = skips.get(name, [])
            if not a["orph"] or not a["ins"]:
                tab[name] = dict(arm_absent=f"no usable tau in {len(sk)}/{len(splits)} folds "
                                            f"(skipped folds {sk})")
                continue
            orph, ins = np.array(a["orph"]), np.array(a["ins"])
            thr90 = float(np.quantile(ins, 0.90))
            tab[name] = dict(n_orphan=int(len(orph)), n_inset=int(len(ins)),
                             folds_used=int(len(splits) - len(sk)), folds_skipped=sk,
                             auroc=auc(orph, ins),
                             orphan_flag_at_tau=r3(float(np.mean(orph > 1.0))),
                             inset_flag_at_tau=r3(float(np.mean(ins > 1.0))),
                             orphan_flag_at_10pct_inset=r3(float(np.mean(orph > thr90))),
                             accept_orphan=r3(a["a_orph"] / a["n_orph"]) if a["n_orph"] else None,
                             accept_inset=r3(a["a_ins"] / a["n_ins"]) if a["n_ins"] else None,
                             top1_accepted=r3(np.mean(a["top1"])) if a["top1"] else None,
                             coverage_accepted=r3(np.mean(a["cov"])) if a["cov"] else None,
                             mean_size_accepted=r3(np.mean(a["size"])) if a["size"] else None)
        return tab

    # ---- orphan definition 1: held-out adulterant (dictionary rebuilt without it) -------
    ADULT_NAMES = [n for n in DET_NAMES if not n.endswith("_pub")]
    res["pool_note"] = ("held_out_adulterant: used_fit == pub_fit == the whole fit block, so the "
                        "*_pub twins are the same computation as their non-pub originals and are "
                        "not listed twice (measured: ours_gate 0.949 == ours_gate_pub 0.949). "
                        "held_out_composite: the pools really differ, both families are reported.")
    per_arm = {}
    for j in range(1, 7):
        per_arm[NAMES[j - 1]] = one_orphan_setting(lambda c, j=j: c == j,
                                                  lambda c, j=j: 1 <= c <= 6 and c != j,
                                                  lambda fi, j=j: j, False, names=ADULT_NAMES)
    res["orphans"]["held_out_adulterant"] = per_arm
    res["orphans"]["held_out_adulterant_summary"] = {
        n: dict(mean_auroc=r3(np.mean([per_arm[a][n]["auroc"] for a in per_arm])),
                mean_orphan_flag=r3(np.mean([per_arm[a][n]["orphan_flag_at_tau"] for a in per_arm])),
                mean_inset_flag=r3(np.mean([per_arm[a][n]["inset_flag_at_tau"] for a in per_arm])),
                mean_orphan_flag_at_10pct=r3(np.mean([per_arm[a][n]["orphan_flag_at_10pct_inset"]
                                                      for a in per_arm])))
        for n in ADULT_NAMES
        if all(n in per_arm[a] and "auroc" in per_arm[a][n] for a in per_arm)}

    # ---- orphan definition 2: held-out composite (dictionary from singles only) ---------
    res["orphans"]["held_out_composite"] = one_orphan_setting(lambda c: c in (7, 8),
                                                             lambda c: 1 <= c <= 6,
                                                             lambda fi: None, True)
    return res


# --------------------------------------------------------------------------- Part B models
def multilabel(y_rows):
    """(n, 6) binary label matrix from recipe codes; composites are expressible, never trained."""
    Y = np.zeros((len(y_rows), 6))
    for i, c in enumerate(y_rows):
        for a in E.true_set(int(c)):
            Y[i, a] = 1.0
    return Y


def nls_support(Xtr, ytr, Xte, thr, relative=True):
    """Non-negative least squares on the dictionary; support = atoms kept by the threshold rule.

    relative=True  -> keep atoms with a_j >= thr * max(a)      (scale-free, transfers between reps)
    relative=False -> keep atoms with a_j >= thr               (the manuscript's absolute rule,
                    only meaningful in raw absorbance units)
    """
    m0, D, _ = E.fit_dict(Xtr, ytr, np.arange(len(Xtr)))
    out = []
    for x in Xte:
        a, _ = nnls(D, x - m0)
        cut = thr * a.max() if relative else thr
        out.append(tuple(sorted(int(j) for j in range(D.shape[1]) if a[j] >= cut))
                   if a.max() > 0 else ())
    return out


def _singleton(a, cut):
    """True when the relative cut keeps exactly one atom - what a single-adulterant spectra must do."""
    m = float(a.max())
    return bool(m > 0 and int(np.sum(a >= cut * m)) == 1)


def _support(A, cut):
    """Rows of NNLS coefficients -> predicted atom sets under the relative cut."""
    out = []
    for a in np.atleast_2d(A):
        m = float(a.max())
        out.append(tuple(sorted(int(j) for j in range(len(a)) if a[j] >= cut * m)) if m > 0 else ())
    return out


# FAIRNESS OF THE REPRESENTATION AXIS. Whitening is not only a feature map, it is also a
# preprocessor: it makes every retained direction homoscedastic. Handing raw absorbance to an
# L2-penalised logistic, an MLP, an RBF-SVM or a distance classifier with no normalisation at all
# would therefore compare "model + preprocessing" against "model", and any apparent whitening gain
# would partly be a gain the raw arm was never allowed to have. So every scale-sensitive learner is
# given a per-channel standardisation fitted on its own training rows, in EVERY representation arm:
# near a no-op on whitened coordinates, and the strongest cheap normalisation on raw ones.
SCALED = {"logit", "knn", "svmrbf", "mlp", "pcalda", "mlp6"}


class Scaled:
    """StandardScaler welded to the estimator so both arms of the representation axis get it."""

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.base.fit(self.sc.transform(X), y)
        return self

    def predict(self, X):
        return self.base.predict(self.sc.transform(X))

    def predict_proba(self, X):
        return self.base.predict_proba(self.sc.transform(X))

    def decision_function(self, X):
        return self.base.decision_function(self.sc.transform(X))


def _scale_if(name, est):
    return Scaled(est) if name in SCALED else est


def make_clf(name, n_cls=None):
    if name == "lda":                             # svd solver: provably invariant to B2's map
        return LinearDiscriminantAnalysis(solver="svd")
    if name == "logit":
        return _scale_if(name, LogisticRegression(max_iter=2000, C=1.0))
    if name == "plsda":
        return PLSWrap(10)
    if name == "pcalda":
        return _scale_if(name, PCALDA(10))
    if name == "knn":
        return _scale_if(name, KNeighborsClassifier(n_neighbors=5, weights="distance"))
    if name == "svmrbf":
        return _scale_if(name, SVC(kernel="rbf", C=10.0, gamma="scale"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=1, random_state=SEED,
                                      n_jobs=2)
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
    if name == "mlp":
        return _scale_if(name, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                             max_iter=2000, tol=1e-4, random_state=SEED))
    if name == "cnn":
        return TorchCNN(n_cls)                    # standardises per channel inside fit
    raise KeyError(name)


class PLSWrap:
    """PLS regression on one-hot labels, predicted class = argmax (PLS-DA)."""

    def __init__(self, ncomp=10):
        self.ncomp = ncomp

    def _enc(self, y):
        u = np.unique(y)
        Y = np.zeros((len(y), len(u)))
        for i, c in enumerate(y):
            Y[i, int(np.where(u == c)[0][0])] = 1.0
        self.u = u
        return Y

    def fit(self, X, y):
        self.m = PLSRegression(n_components=min(self.ncomp, X.shape[1], len(X) - 1))
        self.m.fit(X, self._enc(y))
        return self

    def predict(self, X):
        return self.u[np.argmax(self.m.predict(X), axis=1)]

    def decision_function(self, X):
        return self.m.predict(X)


class PCALDA:
    def __init__(self, ncomp=10):
        self.ncomp = ncomp

    def fit(self, X, y):
        self.mu = X.mean(0)
        self.p = PCA(n_components=min(self.ncomp, X.shape[1], len(X) - 1), svd_solver="full")
        Z = self.p.fit_transform(X - self.mu)
        self.m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, y)
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X - self.mu))


def make_multilabel_native(name, ncomp=10):
    if name == "pls2":                       # manuscript Arm B: one-vs-rest PLS2, 10 latent vars
        return PLSMulti(ncomp)
    if name == "mlp6":
        return _scale_if(name, MultiNeural(MLPClassifier, hidden_layer_sizes=(64, 32), alpha=1e-3,
                                           max_iter=2000, tol=1e-4, random_state=SEED))
    if name == "cnn6":
        return TorchCNNHead(out=6)            # one joint head, six sigmoids
    raise KeyError(name)


class PLSMulti:
    def __init__(self, ncomp=10):
        self.ncomp = ncomp

    def fit(self, X, Y):
        self.m = PLSRegression(n_components=min(self.ncomp, X.shape[1], len(X) - 1))
        self.m.fit(X, Y)
        return self

    def predict_proba(self, X):
        return np.clip(self.m.predict(X), 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(float)


class MultiNeural:
    """One binary MLP fitted per adulterant atom (a one-vs-rest MLP for the composite task)."""

    def __init__(self, cls, **kw):
        self.cls, self.kw = cls, kw

    def fit(self, X, Y):
        self.ms = []
        for j in range(Y.shape[1]):
            m = self.cls(**self.kw)
            m.fit(X, Y[:, j])
            self.ms.append(m)
        return self

    def predict_proba(self, X):
        return np.stack([m.predict_proba(X)[:, 1] for m in self.ms], 1)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(float)


# --------------------------------------------------------------------------- torch arms
TORCH = None
DEVICE = "cpu"
if DO_CNN:
    try:
        import torch
        from torch import nn
        TORCH = torch
        DEVICE = os.environ.get("P47_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
        _DEV = torch.device(DEVICE)
        if DEVICE == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        TORCH = None
        DEVICE = "cpu"


def _mk(X, mu, sd):
    """Standardised spectra as a (n, 1, p) tensor on the working device."""
    return torch.tensor((X - mu) / sd, dtype=torch.float32, device=_DEV)[:, None, :]


class TorchCNN:
    """Tiny 1-D CNN: conv(16)->conv(32)->global mean pool->linear.

    out=None sizes the head from the labels actually seen in fit(). T1 hands over every recipe code
    in the training block, composites included, so a hard-coded 7-way head is wrong there.
    Per-channel standardisation happens inside fit, in every representation arm.
    """

    def __init__(self, out=None, epochs=40, lr=1e-3, seed=SEED):
        self.out, self.epochs, self.lr, self.seed = out, epochs, lr, seed

    def _model(self, n_in):
        m = nn.Sequential(nn.Conv1d(1, 16, 7, stride=4), nn.ReLU(),
                          nn.Conv1d(16, 32, 5, stride=4), nn.ReLU(),
                          nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                          nn.Linear(32, self.out))
        g = torch.Generator().manual_seed(self.seed)      # same init on CPU and GPU
        for p in m.parameters():
            if p.dim() >= 2:
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g) * (1.0 / p.shape[1] ** 0.5))
            else:
                with torch.no_grad():
                    p.zero_()
        return m.to(_DEV)

    def _train(self, X, t, lossf):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = _mk(X, self.mu, self.sd)
        t = t.to(_DEV)
        self.m = self._model(X.shape[1])
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)
        n, bs = len(Z), min(32, len(Z))
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                lossf(self.m(Z[idx]), t[idx]).backward()
                opt.step()
        self.m.eval()
        return self

    def fit(self, X, y):
        t = torch.tensor(np.asarray(y), dtype=torch.long)
        if self.out is None:
            self.out = int(t.max().item()) + 1
        return self._train(X, t, nn.CrossEntropyLoss())

    def _logits(self, X):
        return self.m(_mk(X, self.mu, self.sd))

    def predict(self, X):
        with torch.no_grad():
            return self._logits(X).argmax(1).cpu().numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.softmax(self._logits(X), 1).cpu().numpy()


class TorchCNNHead(TorchCNN):
    """Multi-label variant: sigmoid outputs, one per adulterant."""

    def __init__(self, out=6, epochs=40, lr=1e-3, seed=SEED):
        super().__init__(out=out, epochs=epochs, lr=lr, seed=seed)

    def fit(self, X, y):
        return self._train(X, torch.tensor(np.asarray(y), dtype=torch.float32),
                           nn.BCEWithLogitsLoss())

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(float)

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.sigmoid(self._logits(X)).cpu().numpy()


# --------------------------------------------------------------------------- Part B runner
def task_T1(X, yr, splits, rep_name, rep_fn=None):
    """Single-adulterant discrimination run as the manuscript runs it: all nine recipe classes
    (pure, six singles, two held-in composites) over the same blocked folds.

    No-straw-man rule: every baseline here trains on fit + calibration indices, i.e. on everything
    the blocked protocol leaves outside the test blocks. That is strictly MORE labelled spectra than
    this paper's engine uses (dictionary from fit, thresholds from calibration), so a baseline that
    loses to ours cannot claim it was data-starved.

    rep_fn(tr_idx) -> matrix overrides the analytic representations, which is how a learned
    (self-supervised) representation gets the identical fold and the identical training set.
    """
    rows, errs = {}, []
    for fit, cal, test in splits:
        tr = np.concatenate([fit, cal])
        Xr = rep_fn(tr) if rep_fn is not None else \
            rep_matrix(rep_name, X, E.Whitener(X, yr, tr))
        for name in MODEL_NAMES_T1:
            if name == "cnn" and TORCH is None:
                continue
            try:
                est = make_clf(name)
                est.fit(Xr[tr], yr[tr])
                rows.setdefault(name, []).append(
                    balanced_accuracy_score(yr[test], est.predict(Xr[test])))
            except Exception as exc:
                errs.append(f"T1/{name}/{rep_name}: {type(exc).__name__}: {exc}")
    return {n: r3(np.mean(v)) if v else None for n, v in rows.items()}, errs


def task_T2(X, yr, splits, rep_name, out, rep_fn=None):
    """Held-out composite exact-set: train on pure + six singles, test on codes 7 and 8.

    The composite label vectors are expressible in every model's output space (six binary atoms),
    so this is a test of training-coverage failure, not of hypothesis-space failure. Baselines train
    on fit + calibration singles (see task_T1 for the no-straw-man rule).

    The dictionary-argmin arm is deliberately metric-free: it uses plain Euclidean distance inside
    whatever representation it is handed. Under rep=raw that is the ordinary dictionary baseline
    without whitening; under rep=white it IS this paper's point attribution, because the whitened
    coordinates are an isometry of the whitened metric.
    """
    rows = {m: [] for m in MODEL_NAMES_T2}
    f1 = {m: [] for m in MODEL_NAMES_T2}
    all_truth, all_grp = [], []
    for fit, cal, test in splits:
        tr = np.concatenate([fit, cal])
        tr = tr[yr[tr] <= 6]
        if rep_fn is not None:
            Xr = rep_fn(np.concatenate([fit, cal]))
        else:
            Xr = rep_matrix(rep_name, X, E.Whitener(X, yr, tr))
        t = test[yr[test] >= 7]
        if len(t) == 0:
            continue
        Yf = multilabel(yr[tr])
        truth = [tuple(sorted(E.true_set(int(c)))) for c in yr[t]]
        grp = np.where(yr[t] == 7, "binary", "ternary")
        all_truth += truth
        all_grp += list(grp)
        for name in MODEL_NAMES_T2:
            if name == "cnn6" and TORCH is None:
                continue
            try:
                if name == "dictargmin":
                    m0, D, _ = E.fit_dict(Xr[tr], yr[tr], np.arange(len(tr)))
                    cand = E.subsets()
                    cand_m = np.stack([E.pred(m0, D, S) for S in cand])
                    sc = np.stack([np.linalg.norm(x[None, :] - cand_m, axis=1) for x in Xr[t]])
                    pred_sets = [cand[int(np.argmin(sc[k]))] for k in range(len(t))]
                elif name == "nls":
                    best, detail, best_sets = -1.0, {}, None
                    for rule, thr in NLS_LADDER:
                        ps = nls_support(Xr[tr], yr[tr], Xr[t], thr, relative=(rule == "rel"))
                        e = float(np.mean([p == s for p, s in zip(ps, truth)]))
                        detail[f"{rule}{thr}"] = round(e, 3)
                        if e > best:
                            best, best_sets = e, ps
                    out.setdefault("nls_threshold_detail", {}).setdefault(rep_name, []).append(detail)
                    pred_sets = best_sets
                elif name == "nls_cal":
                    # the honest version of the row above: nothing about the composites is used to
                    # pick the cut. Dictionary from FIT singles, cut from CALIBRATION singles.
                    fs, ci = fit[yr[fit] <= 6], cal[yr[cal] <= 6]
                    m0, D, _ = E.fit_dict(Xr[fs], yr[fs], np.arange(len(fs)))
                    A = np.stack([nnls(D, x - m0)[0] for x in Xr[ci]]) if len(ci) else np.zeros((0, D.shape[1]))
                    cut = None
                    for c in NLS_CAL_LADDER:
                        acc = float(np.mean([_singleton(a, c) for a in A])) if len(A) else 0.0
                        if acc >= NLS_CAL_RULE:
                            cut = c
                            break
                    if cut is None:                       # rule unsatisfiable on this fold
                        cut = max(NLS_CAL_LADDER,
                                  key=lambda c: float(np.mean([_singleton(a, c) for a in A])))
                    out.setdefault("nls_cal_detail", {}).setdefault(rep_name, []).append(
                        dict(cut=cut, n_cal=int(len(ci)), n_fit=int(len(fs))))
                    At = np.stack([nnls(D, x - m0)[0] for x in Xr[t]]) if len(t) else np.zeros((0, D.shape[1]))
                    pred_sets = _support(At, cut)
                elif name in ("pls2", "mlp6", "cnn6"):
                    m = make_multilabel_native(name).fit(Xr[tr], Yf)
                    P = np.asarray(m.predict(Xr[t]), float)
                    pred_sets = [tuple(sorted(int(j) for j in range(P.shape[1]) if P[k, j] > 0.5))
                                 for k in range(len(t))]
                else:
                    # explicit one-vs-rest (one fresh binary classifier per atom) rather than
                    # sklearn's OneVsRestClassifier, which clones its base estimator and cannot
                    # rebuild the pre-processing wrapper that make_clf may return.
                    P = np.zeros((len(t), Yf.shape[1]))
                    for j in range(Yf.shape[1]):
                        m = make_clf(name, n_cls=2).fit(Xr[tr], Yf[:, j])
                        P[:, j] = np.asarray(m.predict(Xr[t]), float)
                    pred_sets = [tuple(sorted(int(j) for j in range(P.shape[1]) if P[k, j] > 0.5))
                                 for k in range(len(t))]
                rows[name].append(float(np.mean([ps == ts for ps, ts in zip(pred_sets, truth)])))
                Pf = np.zeros((len(t), 6))
                T = np.zeros((len(t), 6))
                for k, ps in enumerate(pred_sets):
                    for j in ps:
                        Pf[k, j] = 1.0
                for k, ts in enumerate(truth):
                    for j in ts:
                        T[k, j] = 1.0
                f1[name].append(f1_score(T, Pf, average="micro", zero_division=0))
                ex = np.array([float(ps == ts) for ps, ts in zip(pred_sets, truth)])
                d = out.setdefault("t2_diag", {}).setdefault(rep_name, {}).setdefault(name, [])
                d.append(dict(n=int(len(t)), mean_size=r3(np.mean([len(ps) for ps in pred_sets])),
                              empty=r3(float(np.mean([len(ps) == 0 for ps in pred_sets]))),
                              binary=r3(float(np.mean(ex[grp == "binary"]))) if (grp == "binary").any() else None,
                              ternary=r3(float(np.mean(ex[grp == "ternary"]))) if (grp == "ternary").any() else None,
                              exact=r3(float(ex.mean()))))
            except Exception as exc:
                out.setdefault("_errors", []).append(f"T2/{name}/{rep_name}: {type(exc).__name__}: {exc}")
    # Interpretation reference, not a method: the best CONSTANT atom set over the pooled held-out
    # composites, chosen with hindsight. Any exact-set row has to be read against this, because the
    # test pool contains only two distinct truths, so "always name the same pair" already scores.
    if all_truth:
        cands = subsets_n(6)
        sc = [(float(np.mean([c == s for c in all_truth])), k) for k, s in enumerate(cands)]
        best, k = max(sc)
        g = np.array(all_grp)
        out.setdefault("t2_oracle_constant", {})[rep_name] = dict(
            set=list(cands[k]), exact=r3(best),
            binary=r3(float(np.mean([cands[k] == s for s, gg in zip(all_truth, g) if gg == "binary"])))
            if (g == "binary").any() else None,
            ternary=r3(float(np.mean([cands[k] == s for s, gg in zip(all_truth, g) if gg == "ternary"])))
            if (g == "ternary").any() else None,
            n=len(all_truth), note="oracle: constant set selected on the test composites themselves")
    return ({n: r3(np.mean(v)) if v else None for n, v in rows.items()},
            {n: r3(np.mean(v)) if v else None for n, v in f1.items()})


MODEL_NAMES_T1 = ["lda", "logit", "plsda", "pcalda", "knn", "svmrbf", "rf", "gbdt", "mlp", "cnn"]
MODEL_NAMES_T2 = ["logit", "knn", "svmrbf", "rf", "gbdt", "pls2", "mlp6", "cnn6", "nls", "nls_cal",
                  "dictargmin"]
# 'abs' = the manuscript's absolute absorbance thresholds; 'rel' = scale-free, so it also means
# something in the whitened coordinates.
# "nls" is an ORACLE row: its threshold is chosen as the best exact-set over the ladder, which is
# computed on the held-out composites themselves. It can only be quoted as an upper bound.
# "nls_cal" is the quotable arm: dictionary from fit, cut from calibration singles (see below).
NLS_LADDER = [("abs", 0.02), ("abs", 0.05), ("abs", 0.10),
              ("rel", 0.10), ("rel", 0.20), ("rel", 0.35)]
NLS_CAL_LADDER = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
NLS_CAL_RULE = 0.90        # smallest relative cut whose singleton-support accuracy on cal singles
# REPS = raw absorbance, SNV, full whitened coordinates, whitened top-k subspace
REPS = ["raw", "snv", "white", "whitek"]


def part_b_tea(X, yr, splits, out):
    tab = {}
    for rep_name in REPS:
        t1, errs = task_T1(X, yr, splits, rep_name)
        t2, t2f1 = task_T2(X, yr, splits, rep_name, out)
        tab[rep_name] = dict(T1_balanced_acc=t1, T2_exact_set=t2, T2_micro_f1=t2f1)
        out["partB_tea"] = tab
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("T1", rep_name, json.dumps(t1), flush=True)
        print("T2", rep_name, json.dumps(t2), flush=True)
        if errs:
            out.setdefault("_errors", []).extend(errs)
    # B2: only the UNCONSTRAINED linear arm is a bug check. LDA(svd) is provably invariant to a
    # fixed invertible linear map, so a raw-vs-white gap here means the feature map itself is wrong.
    # Logistic is scale-sensitive (L2 penalty) and now standardised in both arms: any gap it shows is
    # a regularisation-by-metric interaction and is reported as a result, not used to void the run.
    inv = {}
    for m, hard in (("lda", True), ("logit", False)):
        vals = {rp: tab[rp]["T1_balanced_acc"].get(m) for rp in REPS if tab[rp]["T1_balanced_acc"]}
        if vals.get("raw") is not None and vals.get("white") is not None:
            d = abs(vals["raw"] - vals["white"])
            inv[m] = dict(raw=vals["raw"], white=vals["white"], delta=r3(d),
                          bug_check=hard, void_run=bool(hard and d > 0.02))
    tab["_invariance_check_B2"] = inv
    return tab


# --------------------------------------------------------------------- Part C: learned reps
# The self-supervised line belongs to THIS paper (Sections 3.10 / the FoodSpec-SSL lineage), so the
# analytic prior has to be faced against the learned representation inside the same factorial,
# not delegated to a companion study.
SSL_KINDS = ["vicreg", "barlow", "randinit"]
SSL_EPOCHS = 80            # identical to the §3.10 arms, so the representation is not undertrained
_ssl_cache = {}


def ssl_rep_fn(X0, kind):
    import exp26_nonrecon_ssl as S26

    def fn(tr):
        key = (kind, frozenset(int(i) for i in tr))
        if key in _ssl_cache:
            return _ssl_cache[key]
        import torch as _t
        mu, sd = X0[tr].mean(0), np.clip(X0[tr].std(0), 1e-8, None)
        Zs = (X0 - mu) / sd                       # standardised on training spectra only
        if kind == "randinit":
            _t.manual_seed(SEED)
            enc = S26.Enc().to(S26.DEV).eval()     # same capacity, zero optimisation steps
        else:
            enc = S26.train_ssl_encoder(X0[tr], kind, epochs=SSL_EPOCHS, seed=SEED)
        _ssl_cache[key] = S26.embed(enc, Zs)
        return _ssl_cache[key]
    return fn


def part_c_tea(X, yr, splits, out):
    tab = {}
    # per-fold guard: an encoder may not see the block its OWN fold tests. The union of the five test
    # blocks is the entire corpus (every spectrum is held out exactly once), so a guard written
    # against that union rejects every fold and proves nothing.
    fold_of = {frozenset(int(i) for i in np.concatenate([fit, cal])):
               set(int(i) for i in test) for fit, cal, test in splits}
    for kind in SSL_KINDS:
        fn = ssl_rep_fn(X, kind)
        t0 = time.time()

        def guarded(tr, fn=fn):
            key = frozenset(int(i) for i in tr)
            if key not in fold_of:
                raise RuntimeError("SSL arm asked for a training set that is no fold's fit+cal")
            bad = key & fold_of[key]
            if bad:
                raise RuntimeError(f"SSL encoder would see {len(bad)} spectra its fold tests")
            return fn(tr)

        t1, errs = task_T1(X, yr, splits, f"ssl_{kind}", rep_fn=guarded)
        t2, t2f1 = task_T2(X, yr, splits, f"ssl_{kind}", out, rep_fn=guarded)
        tab[f"ssl_{kind}"] = dict(T1_balanced_acc=t1, T2_exact_set=t2, T2_micro_f1=t2f1)
        out["partC_tea"] = tab
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("C T1", kind, json.dumps(t1), flush=True)
        print("C T2", kind, json.dumps(t2), "min", round((time.time() - t0) / 60, 1), flush=True)
        if errs:
            out.setdefault("_errors", []).extend(errs)
    return tab


def cinnamon_data():
    """exp40's loader reads the public deposits through the scratch p5_data, while exp33 has already
    pinned the manuscript's scripts/p5_data (which has no load_public). Bind the scratch copy for the
    duration of the call, then restore, so the tea loader cannot later pick up the wrong corpus."""
    import importlib.util
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location("p5_data", os.path.join(here, "p5_data.py"))
    alt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alt)
    prev = sys.modules.get("p5_data")
    sys.modules["p5_data"] = alt
    try:
        import exp40_public_pipeline as C40
        X, comp, lvl, blk, sr, n_dup, p = C40.load_cinnamon()
    finally:
        if prev is not None:
            sys.modules["p5_data"] = prev
    return X, comp, blk, n_dup, p


def blocked_folds_cinnamon(blk, nf=NF):
    triples = []
    for k in range(nf):
        test = np.where(blk % nf == k)[0]
        rest = np.where(blk % nf != k)[0]
        fit = rest[blk[rest] < 5]
        cal = rest[blk[rest] >= 5]
        triples.append((np.array(sorted(fit)), np.array(sorted(cal)), np.array(sorted(test))))
    return triples


def part_b_cinnamon(out):
    try:
        X, comp, blk, n_dup, p = cinnamon_data()
    except Exception as exc:
        out["cinnamon"] = f"unavailable: {type(exc).__name__}: {exc}"
        return
    splits = blocked_folds_cinnamon(blk)
    # exp33's Whitener needs dense class labels 0..k; comp is already 0..3
    tab = {}
    for rep_name in REPS:
        accs = {}
        for name in MODEL_NAMES_T1:
            if name == "cnn" and TORCH is None:
                continue
            vs = []
            for fit, cal, test in splits:
                tr = np.concatenate([fit, cal])
                wh = E.Whitener(X, comp, tr)
                Xr = rep_matrix(rep_name, X, wh)
                try:
                    est = make_clf(name, n_cls=4)
                    est.fit(Xr[tr], comp[tr])
                    vs.append(balanced_accuracy_score(comp[test], est.predict(Xr[test])))
                except Exception as exc:
                    out.setdefault("_errors", []).append(
                        f"CIN/{name}/{rep_name}: {type(exc).__name__}")
            accs[name] = r3(np.mean(vs)) if vs else None
        tab[rep_name] = accs
        print("CIN T1", rep_name, json.dumps(accs), flush=True)
    out["cinnamon"] = dict(n=int(len(X)), p=int(p), n_dup_skipped=int(n_dup),
                           window=E.WINDOW, T1_balanced_acc=tab)


def main():
    t0 = time.time()
    X, yr = E.load()
    rng = np.random.default_rng(SEED)
    splits = E.make_splits(yr, "blocked", rng)
    out = dict(alpha=ALPHA, k_local=K_LOCAL, window=E.WINDOW, seed=SEED, n_spectra=int(len(X)),
               n_channels=int(X.shape[1]),
               protocol="blocked 5-fold: fit blocks<5, cal blocks>=5, test blocks%5==k; "
                        "identical folds for every representation and every model",
               train_rule="Part A keeps the manuscript's three-way split (dictionary from fit, "
                          "thresholds from calibration). Part B baselines train on fit + "
                          "calibration indices, i.e. on more labelled spectra than the paper's own "
                          "engine consumes - the no-straw-man rule.",
               reps=REPS, models_T1=MODEL_NAMES_T1, models_T2=MODEL_NAMES_T2,
               scaled_models=sorted(SCALED),
               scaling_rule="per-channel StandardScaler fitted on each arm's own training rows for "
                            "the scale-sensitive learners above, in EVERY representation; LDA(svd), "
                            "the trees, the torch arms (which standardise internally) and the "
                            "dictionary arms are left unscaled because their invariance or their "
                            "metric semantics is exactly what is under test",
               nls_ladder=[f"{r}{t}" for r, t in NLS_LADDER],
               nls_row="ORACLE: best exact-set over the whole ladder, threshold read off the test "
                       "composites - an upper bound, not a claim",
               nls_cal_row=f"quotable NLS arm: dictionary from fit singles, relative cut from "
                           f"{NLS_CAL_LADDER} chosen as the smallest whose singleton-support "
                           f"accuracy on calibration singles is >= {NLS_CAL_RULE}",
               torch_device=(DEVICE if TORCH is not None else "torch unavailable - cnn arms skipped"),
               decision_rules=dict(
                   A1="prior-free detector within 0.02 AUROC of the gate => gate is not the "
                      "contribution; justify by joint refuse+cover behaviour",
                   B1="exact-set > 0.10 on held-out composites for any model/rep => rewrite "
                      "'not an artefact of one model family' as 'one linear model family'",
                   B2="LDA(svd) balanced accuracy must not move between raw and white (invertible "
                      "linear map); > 0.02 means the feature map is broken and Part B is void. "
                      "Penalised learners are scale-sensitive and standardised in both arms, so "
                      "their raw-vs-white gap is a result, not a bug signal",
                   B3="'prior helps across model classes' requires a gain averaged over the "
                      "non-invariant classes on tea AND the same direction on cinnamon",
                   C1="if a self-supervised representation reaches exact-set > 0.10 on held-out "
                      "composites for any model class, Section 3.10's 'the representation is not the "
                      "bottleneck' is confined to discrimination and may not be quoted for "
                      "attribution; if SSL matches the analytic representations on T1 but stays at "
                      "or below them on T2, Sections 3.10 and 3.3 collapse into one statement - "
                      "learned representations buy no attribution either"))

    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            prev.update(out)
            out = prev
        except Exception:
            pass

    # this file is merged across partial re-runs, sometimes on different hardware, so record what
    # this invocation actually did instead of leaving a single device string to imply all of it
    out.setdefault("run_log", []).append(dict(
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        stages=dict(partA=bool(DO_PART_A), tea=bool(DO_TEA), ssl=bool(DO_SSL),
                    cinnamon=bool(DO_CIN), cnn=bool(DO_CNN)),
        torch_device=(DEVICE if TORCH is not None else "torch skipped"),
        torch_version=(torch.__version__ if TORCH is not None else None)))

    if DO_PART_A:
        out["partA"] = part_a_tea(X, yr, splits)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("partA done", round(time.time() - t0, 1), flush=True)

    if DO_TEA:
        out["partB_tea"] = part_b_tea(X, yr, splits, out)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("partB tea done", round(time.time() - t0, 1), flush=True)

    if DO_SSL:
        out["partC_tea"] = part_c_tea(X, yr, splits, out)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("partC ssl done", round(time.time() - t0, 1), flush=True)

    if DO_CIN:
        part_b_cinnamon(out)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("partB cinnamon done", round(time.time() - t0, 1), flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT, "elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
