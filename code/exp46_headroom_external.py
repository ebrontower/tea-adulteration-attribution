"""exp46 - out-of-sample test of the "should you pretrain at all" criterion.

The criterion was formed on 8 corpora (exp28/E3, and the E1/E2 SSL arms): if the between-class
scatter of the labels sits inside the top-k principal subspace of the *unlabelled* spectra, then
no encoder whose job is to model that subspace can add information, and the ceiling is set by the
labels and the linear probe, not by the pretraining. Operationally the calibration corpora all had
coverage_pc20 >= 0.995 with residual-only probes at chance and headroom_full ~= 0.

exp46 runs the *identical five arms* on 7 corpora that were never part of that reasoning:
the honey/sucrose panel, both coconut-milk/water panels, and the four QIB MIR panels (fresh meats,
olive oils, instant coffee, fruit purees). Two hold-out kinds per corpus:
    random   - 80/20 over spectra, which is how exp28 was run (comparable to the calibration)
    phys     - 80/20 over *preparations*, so replicate spectra of one physical sample never appear
               on both sides (the honest version for paired panels)
Arms (all with a fixed labelled budget of 10 and 20 spectra per class, 3 seeds):
    pc20_lin   logistic probe on the top-20 PCs       (what a same-budget linear method gets)
    pc20_svm   RBF-SVM on the top-20 PCs              (nonlinear head, same subspace)
    full_svm   RBF-SVM on all channels                (flexible head, no dimension restriction)
    resid_svm  RBF-SVM on the PC-20 residual only     (what lives outside the subspace)
    coverage   share of between-class scatter inside the top-20 PC subspace
Rule under test, stated before running: coverage >= 0.99 and resid_lift <= 0.02 and
headroom_full <= 0.02  =>  a learned representation has nothing to add on that corpus.
"""
import json
import os
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = p5_paths.CODE
sys.path.insert(0, HERE)
import ext_data as ED                                          # noqa: E402
import qib_data as Q                                           # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.RESULTS + "/exp46_headroom_onsample.json"
CALIB = p5_paths.RESULTS + "/exp28_label_subspace.json"
BUDGETS = (10, 20)
SEEDS = (0, 1, 2)
K = 20


def probe(F, y, tr, te, n_per_class, seed, kind="linear"):
    rng = np.random.default_rng(seed)
    sub = []
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c].copy()
        rng.shuffle(idx)
        sub += list(idx[:min(n_per_class, len(idx))])
    sub = np.array(sub)
    sc = StandardScaler().fit(F[sub])
    clf = SVC(kernel="rbf", C=10.0, gamma="scale") if kind == "svm" else LogisticRegression(max_iter=3000)
    clf.fit(sc.transform(F[sub]), y[sub])
    return balanced_accuracy_score(y[te], clf.predict(sc.transform(F[te])))


def mean_probe(F, y, tr, te, kind="linear"):
    return {str(b): round(float(np.mean([probe(F, y, tr, te, b, s, kind) for s in SEEDS])), 3)
            for b in BUDGETS}


def coverage(X, y, tr, k):
    """share of between-class scatter inside the top-k PC subspace.

    trace(V'SbV) = sum_c w_c ||V'mu_c - V'mu||^2, so the p x p matrix is never formed
    (p = 3351 for the honey panel)."""
    pc = PCA(n_components=k, random_state=0).fit(X[tr])
    V = pc.components_.T                                   # (p, k)
    mu = X[tr].mean(0)
    tot = sub = 0.0
    for c in np.unique(y[tr]):
        m = X[tr][y[tr] == c].mean(0) - mu
        w = float((y[tr] == c).sum())
        tot += w * float(m @ m)
        v = V.T @ m
        sub += w * float(v @ v)
    return sub / tot if tot > 0 else float("nan")


def splits(y, groups, kind):
    n = len(y)
    idx = np.arange(n)
    if kind == "random":
        rng = np.random.default_rng(42)
        tr = []
        for c in np.unique(y):
            ii = idx[y == c].copy()
            rng.shuffle(ii)
            tr += list(ii[:int(0.8 * len(ii))])
        tr = np.array(sorted(tr))
        return tr, np.setdiff1d(idx, tr)
    gs = sorted(set(groups))
    rng = np.random.default_rng(7)
    rng.shuffle(gs)
    cut = int(round(0.8 * len(gs)))
    trg, teg = set(gs[:cut]), set(gs[cut:])
    tr = np.array([i for i in idx if groups[i] in trg])
    te = np.array([i for i in idx if groups[i] in teg])
    if len(te) < 4 or len(np.unique(y[te])) < 2:
        return None, None
    return tr, te


def arms(X, y, groups):
    out = {}
    for kind in ("random", "phys"):
        tr, te = splits(y, groups, kind)
        if tr is None:
            out[kind] = dict(note="not enough held-out preparations/classes")
            continue
        mu, sd = X[tr].mean(0), np.clip(X[tr].std(0), 1e-8, None)
        Z, yy = (X - mu) / sd, y
        k = int(min(K, len(tr) - len(np.unique(yy[tr])) - 1, Z.shape[1]))
        if k < 2:
            out[kind] = dict(note="too few labelled rows for a %d-dim subspace" % K)
            continue
        pc = PCA(n_components=k, random_state=0).fit(Z[tr])
        scores = pc.transform(Z)
        R = Z - pc.inverse_transform(scores)
        r = dict(n=int(len(y)), n_channel=int(X.shape[1]), n_class=int(len(np.unique(yy))),
                 n_train=int(len(tr)), n_test=int(len(te)), k=k, chance=round(1.0 / len(np.unique(yy)), 3),
                 coverage_pc20=round(float(coverage(Z, yy, tr, k)), 4))
        r["pc20_lin"] = mean_probe(scores, yy, tr, te, "linear")
        r["pc20_svm"] = mean_probe(scores, yy, tr, te, "svm")
        r["full_svm"] = mean_probe(Z, yy, tr, te, "svm")
        r["resid_svm"] = mean_probe(R, yy, tr, te, "svm")
        for tag, Ftr, Fte in (("pc20", scores[tr], scores[te]), ("resid", R[tr], R[te]),
                              ("full", Z[tr], Z[te])):
            sc = StandardScaler().fit(Ftr)
            clf = (LinearDiscriminantAnalysis() if tag == "pc20"
                   else SVC(kernel="rbf", C=10.0, gamma="scale"))
            clf.fit(sc.transform(Ftr), yy[tr])
            r[tag + "_full"] = round(float(balanced_accuracy_score(yy[te], clf.predict(sc.transform(Fte)))), 3)
        r["headroom_full"] = round(float(r["full_svm"]["20"] - max(r["pc20_lin"]["20"], r["pc20_svm"]["20"])), 3)
        r["resid_lift"] = round(float(r["resid_svm"]["20"] - r["chance"]), 3)
        r["verdict"] = ("no_headroom" if (r["coverage_pc20"] >= 0.99 and r["resid_lift"] <= 0.02
                                          and r["headroom_full"] <= 0.02)
                        else "headroom_present" if (r["resid_lift"] > 0.05 or r["headroom_full"] > 0.05)
                        else "borderline")
        out[kind] = r
    return out


def corpora():
    h = ED.load_honey()
    y = np.unique(np.round(h["y"], 2))
    lab = np.searchsorted(y, np.round(h["y"], 2))
    yield "HONEY_level", h["X"], lab, np.asarray(h["groups"], object), \
        dict(matrix="honey", adulterant="sucrose", task="declared level as class",
             instrument="FTIR transmission (%T -> A)", unit="absorbance")
    for tag, d in ED.load_coconut().items():
        lv = np.unique(np.round(d["y"], 2))
        yield "COCO_" + tag, d["X"], np.searchsorted(lv, np.round(d["y"], 2)), \
            np.asarray(d["groups"], object), \
            dict(matrix="coconut milk", adulterant="water", task="water level as class",
                 instrument=d["meta"]["instrument"], unit="as delivered")
    a, X, df = Q.meats()
    mp = {"Chicken": 0, "Pork": 1, "Turkey": 2}
    yield "QIB_meats", X, np.array([mp[v] for v in df["label"]]), \
        df["phys"].to_numpy(object), \
        dict(matrix="fresh meat", task="species", instrument="MIR ATR", unit="absorbance",
             note="60 preparations x 2 runs")
    a, X, df = Q.olive()
    yield "QIB_olive", X, np.array([sorted(df["label"].unique()).index(v) for v in df["label"]]), \
        df["phys"].to_numpy(object), \
        dict(matrix="olive oil", task="country of origin", instrument="MIR ATR",
             unit="absorbance", note="60 oils x 2 sessions")
    a, X, df = Q.coffee()
    yield "QIB_coffee", X, np.array([sorted(df["label"].unique()).index(v) for v in df["label"]]), \
        df["phys"].to_numpy(object), \
        dict(matrix="instant coffee", task="species", instrument="DRIFT (not ATR)",
             unit="suspected %T, un-converted - see memo", note="56 samples, one spectrum each")
    a, X, df = Q.purees()
    yield "QIB_purees", X, np.array([sorted(df["label"].unique()).index(v) for v in df["label"]]), \
        df["id"].to_numpy(object), \
        dict(matrix="fruit puree", task="strawberry / not", instrument="MIR",
             unit="values ~4e-3 with negatives: already preprocessed, not raw absorbance")


if __name__ == "__main__":
    res, meta = {}, {}
    for nm, X, y, g, m in corpora():
        t0 = time.time()
        try:
            res[nm] = arms(np.asarray(X, float), np.asarray(y, int), np.asarray(g, object))
            meta[nm] = m
            for k, v in res[nm].items():
                print("%-12s %-6s cov=%s head=%s lift=%s -> %s" %
                      (nm, k, v.get("coverage_pc20"), v.get("headroom_full"),
                       v.get("resid_lift"), v.get("verdict") or v.get("note")), flush=True)
        except Exception as e:                                  # noqa: BLE001
            res[nm] = dict(error="%s: %s" % (type(e).__name__, e))
            print(nm, "FAILED", repr(e), flush=True)
        print("   %.1fs" % (time.time() - t0), flush=True)
    calib = {}
    try:
        j = json.load(open(CALIB, encoding="utf-8"))
        for nm, v in j.items():
            if isinstance(v, dict) and "coverage_pc20" in v:
                calib[nm] = dict(coverage_pc20=v["coverage_pc20"], headroom_full=v.get("headroom_full"),
                                 resid_lift=v.get("resid_lift"), pc20_lin=v.get("pc20_lin"),
                                 pc20_svm=v.get("pc20_svm"), full_svm=v.get("full_svm"),
                                 resid_svm=v.get("resid_svm"), n=v.get("n"), n_class=v.get("n_class"))
    except Exception as e:                                      # noqa: BLE001
        calib = dict(error="%s: %s" % (type(e).__name__, e))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(rule=dict(coverage_min=0.99, resid_lift_max=0.02, headroom_full_max=0.02),
                       calibration_from_exp28=calib, new=res, meta=meta), f, indent=1, ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
