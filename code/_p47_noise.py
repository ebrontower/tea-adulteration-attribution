"""exp47-N - SNR / drift stress test on the information-budget ladder (2026-11-02).

Pre-registered design and decision rules (N1-N7):
    preregistrations/exp47-N_noise_stress_preregistration_v1.md

Every arm - baselines, gate, NLS ladder, prior-free detectors - is handed the SAME noisy spectrum.
Two injection regimes:
    matched : noise on every row (the bench is noisy and you build the library on that bench);
              exchangeability between calibration and test holds, so the conformal guarantee
              SHOULD survive.
    shift   : noise on the test rows only (library built clean, samples arrive dirty);
              exchangeability is broken on purpose - this column can only hurt us.
Three noise families, each calibrated to something actually measured in this corpus:
    iso    : iid Gaussian per channel, SD = k x mean within-class residual SD (detector noise);
    sub    : drawn from the within-class residual covariance itself (sample-prep repeatability) -
             this is the part the whitener absorbs by construction;
    drift  : per-spectrum offset + slope, per-channel amplitude = observed between-block drift SD.
k=1 replicates the corpus's own repeatability; k is a multiplier on it, never an arbitrary dB.
k=0 is re-run through this same per-fold code path so the reference numbers are poolable with the
noisy ones (the clean exp47 headline is cross-fold pooled, not fold-averaged - comparing the two
would mix the noise manipulation with an aggregation change).
"""
import os
import sys
import json
import time


import numpy as np  # noqa: E402

import exp33_whitened_attr as E  # noqa: E402
import exp47_baseline_ladder as L  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.environ.get("P47N_OUT") or p5_paths.RESULTS + "/exp47_noise.json"
FAMS = [f for f in (os.environ.get("P47N_FAM") or "iso,sub,drift").split(",") if f]
LEVELS = [float(v) for v in (os.environ.get("P47N_LV") or "2,10").split(",")]
REGIMES = [r for r in (os.environ.get("P47N_REG") or "matched,shift").split(",") if r]
DO_AB = os.environ.get("P47N_AB", "1") == "1"      # T1 discrimination + T2 held-out composites
DO_A = os.environ.get("P47N_A", "0") == "1"        # gate + prior-free detectors (slow)
SEED_BASE = int(os.environ.get("P47N_SEED") or 1000)
NOISE_DIM = 60        # residual directions used by the `sub` family

# torch arms are excluded from this sweep: the CNN row is the weakest arm in the clean table and
# its budget fairness was settled separately (exp47 result doc section 7.1). Dropping it here keeps
# every cell honest about what is being varied (the metric, not the model class).
L.MODEL_NAMES_T1 = [m for m in L.MODEL_NAMES_T1 if m != "cnn"]
L.MODEL_NAMES_T2 = [m for m in L.MODEL_NAMES_T2 if m != "cnn6"]

# Optional arm trimming. The full exp47 grid costs ~15 min per fold per representation; a noise
# sweep only needs the arms that carry N1-N7: an invariant linear reference (lda), a scale-sensitive
# penalised learner (logit), a distance learner (knn) and the best non-linear one-vs-rest family for
# attribution (gbdt), plus the quotable dictionary row (nls_cal) and the metric-free argmin (the
# paper's own engine). The ORACLE nls ladder is dropped - it is not a claim in any regime.
_keep1 = [m for m in (os.environ.get("P47N_T1") or "").split(",") if m]
_keep2 = [m for m in (os.environ.get("P47N_T2") or "").split(",") if m]
if _keep1:
    L.MODEL_NAMES_T1 = [m for m in L.MODEL_NAMES_T1 if m in _keep1]
if _keep2:
    L.MODEL_NAMES_T2 = [m for m in L.MODEL_NAMES_T2 if m in _keep2]


# --------------------------------------------------------------------------- noise model
def residual_model(X, yr, tr):
    """Per-channel statistics of what this instrument actually repeats, from TRAINING rows only."""
    sub, y = X[tr], yr[tr]
    classes = np.unique(y)
    mus = {int(c): sub[y == c].mean(0) for c in classes}
    R = np.concatenate([sub[y == c] - mus[c] for c in classes], 0)
    R = R - R.mean(0)
    sd_res = R.std(0, ddof=1)
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    lam = S ** 2 / max(1, len(R) - 1)
    b = E.block_ids(yr)[tr]
    bm = np.stack([sub[b == k].mean(0) for k in np.unique(b)], 0)
    sd_drift = bm.std(0, ddof=1) if len(bm) > 1 else sd_res.copy()
    return dict(sd_res=sd_res, lam=lam, Vt=Vt, sd_drift=sd_drift,
                n_blocks=int(len(bm)), n_res=int(len(R)))


def make_noise(fam, n, model, rng, k):
    p = len(model["sd_res"])
    if k == 0:
        return np.zeros((n, p))
    if fam == "iso":
        sd = float(np.mean(model["sd_res"])) * k
        return rng.normal(0.0, sd, size=(n, p))
    if fam == "sub":
        m = int(min(NOISE_DIM, p, len(model["lam"])))
        z = rng.normal(size=(n, m)) * np.sqrt(np.maximum(model["lam"][:m], 0.0)) * k
        return z @ model["Vt"][:m]
    if fam == "drift":
        g = (np.arange(p) - 0.5 * (p - 1)) / (0.5 * (p - 1))
        u = rng.normal(size=(n, 1))
        v = rng.normal(size=(n, 1))
        return k * (u * model["sd_drift"][None, :] + v * (g * model["sd_drift"])[None, :])
    raise KeyError(fam)


FAM_ORDER = {"clean": 0, "iso": 1, "sub": 2, "drift": 3}


def fold_seed(fi, fam, k):
    return SEED_BASE + 7919 * fi + int(100 * k) + 131 * FAM_ORDER.get(fam, 7)


def noisy_corpus(X, model, fam, k, regime, test_idx, rng):
    """Return a copy of X with noise on every row (matched) or on this fold's test rows (shift)."""
    Xn = X.copy()
    rows = np.arange(len(X)) if regime == "matched" else np.asarray(test_idx)
    N = make_noise(fam, len(rows), model, rng, k)
    Xn[rows] = Xn[rows] + N
    rms = float(np.mean(np.sqrt((N ** 2).mean(1)))) if k else 0.0
    return Xn, rms


# --------------------------------------------------------------------------- per-fold runner
def run_cell(X, yr, splits, fam, k, regime):
    t0 = time.time()
    folds = []
    for fi, (fit, cal, test) in enumerate(splits):
        tr = np.concatenate([fit, cal])
        model = residual_model(X, yr, tr)
        rng = np.random.default_rng(fold_seed(fi, fam, k))
        Xn, rms = noisy_corpus(X, model, fam, k, regime, test, rng)
        fold = dict(fold=fi, n_test=int(len(test)), n_test_comp=int(np.isin(yr[test], [7, 8]).sum()),
                    noise_rms_per_channel=r3(rms))
        splits1 = [(fit, cal, test)]
        if DO_AB:
            fold["T1"], fold["T2"] = {}, {}
            for rep in L.REPS:
                accs, errs = L.task_T1(Xn, yr, splits1, rep)
                fold["T1"][rep] = accs
                if errs:
                    fold.setdefault("errors", []).extend(errs)
                scratch = {}
                exact, micro = L.task_T2(Xn, yr, splits1, rep, scratch)
                fold["T2"][rep] = dict(exact_set=exact, micro_f1=micro)
        if DO_A:
            resA = L.part_a_tea(Xn, yr, splits1)
            fold["A_adult"] = {n: v.get("mean_auroc") for n, v in
                               resA["orphans"]["held_out_adulterant_summary"].items()}
            fold["A_adult_flag"] = {n: v.get("mean_orphan_flag_at_10pct") for n, v in
                                   resA["orphans"]["held_out_adulterant_summary"].items()}
            fold["A_comp"] = {n: {kk: v.get(kk) for kk in
                                  ("auroc", "orphan_flag_at_tau", "inset_flag_at_tau",
                                   "orphan_flag_at_10pct_inset", "coverage_accepted",
                                   "mean_size_accepted", "top1_accepted", "arm_absent")}
                              for n, v in resA["orphans"]["held_out_composite"].items()}
        folds.append(fold)
        print(f"  [{fam}/k={k}/{regime}] fold {fi} done "
              f"{round((time.time() - t0) / 60, 1)} min", flush=True)
    return aggregate(folds, fam, k, regime, t0)


def mean_over_folds(folds, path):
    vals = []
    for f in folds:
        cur = f
        for key in path:
            if cur is None or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            vals.append(float(cur))
    return r3(np.mean(vals)) if vals else None


def aggregate(folds, fam, k, regime, t0):
    out = dict(family=fam, k=k, regime=regime, n_folds=len(folds),
               minutes=r3((time.time() - t0) / 60),
               noise_rms_per_channel=[f["noise_rms_per_channel"] for f in folds],
               fold_errors=[e for f in folds for e in f.get("errors", [])])
    if DO_AB:
        out["T1"] = {rep: {m: mean_over_folds(folds, ["T1", rep, m]) for m in L.MODEL_NAMES_T1}
                     for rep in L.REPS}
        out["T1_spread"] = {rep: {m: r3(np.std([f["T1"][rep][m] for f in folds
                                                if f["T1"][rep].get(m) is not None], ddof=1))
                                  if len(folds) > 1 else None for m in L.MODEL_NAMES_T1}
                            for rep in L.REPS}
        out["T2"] = {rep: {m: mean_over_folds(folds, ["T2", rep, "exact_set", m]) for m in L.MODEL_NAMES_T2}
                     for rep in L.REPS}
        out["T2_micro"] = {rep: {m: mean_over_folds(folds, ["T2", rep, "micro_f1", m])
                                 for m in L.MODEL_NAMES_T2} for rep in L.REPS}
    if DO_A:
        names = sorted({n for f in folds for n in f.get("A_adult", {})})
        out["A_adult_auroc"] = {n: mean_over_folds(folds, ["A_adult", n]) for n in names}
        out["A_adult_flag10"] = {n: mean_over_folds(folds, ["A_adult_flag", n]) for n in names}
        cn = sorted({n for f in folds for n in f.get("A_comp", {})})
        out["A_composite"] = {n: {kk: mean_over_folds(folds, ["A_comp", n, kk])
                                  for kk in ("auroc", "orphan_flag_at_tau", "inset_flag_at_tau",
                                             "orphan_flag_at_10pct_inset", "coverage_accepted",
                                             "mean_size_accepted", "top1_accepted")}
                              for n in cn}
        out["A_composite_absent"] = {n: [f["A_comp"][n].get("arm_absent") for f in folds
                                         if n in f.get("A_comp", {})
                                         and f["A_comp"][n].get("arm_absent")] for n in cn}
    return out


def r3(v):
    if v is None or isinstance(v, (bool, np.bool_)):
        return v if v is None else bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return round(float(v), 3)


def main():
    X, yr = E.load()
    rng = np.random.default_rng(L.SEED)
    splits = E.make_splits(yr, "blocked", rng)
    out = dict(
        prereg=r"preregistrations/exp47-N_noise_stress_preregistration_v1.md",
        protocol="exp47 folds and arms unchanged (blocked 5-fold, fit blocks<5 / cal blocks>=5 / "
                 "test blocks%5==k); noise applied per fold with its own seed; every arm sees the "
                 "same X_n; k=0 reference cell runs through this same per-fold path",
        families=FAMS, levels=LEVELS, regimes=REGIMES,
        stages=dict(T1_T2=bool(DO_AB), partA=bool(DO_A)),
        torch_arms="cnn and cnn6 excluded from this sweep (see module docstring)",
        noise_definitions=dict(
            iso="iid Gaussian, SD = k x mean within-class residual SD over channels",
            sub=f"k x draw from the within-class residual covariance on its leading "
                f"{NOISE_DIM} directions - the part the whitener absorbs by construction",
            drift="k x (random offset + random slope), per-channel amplitude = observed "
                  "between-block drift SD fitted on training blocks only"),
        calibration="noise magnitude is a multiple of this corpus's own measured repeatability / "
                    "block drift, never an arbitrary SNR",
        window=E.WINDOW, seed_base=SEED_BASE, n_spectra=int(len(X)),
        n_channels=int(X.shape[1]))

    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            prev.update(out)
            out = prev
        except Exception:
            pass
    out.setdefault("cells", [])

    # k=0 makes the two regimes the same computation, so the reference cell is run once.
    cells = ([("clean", 0.0, REGIMES[0])] if REGIMES else [])
    cells += [(f, k, reg) for f in FAMS for k in LEVELS for reg in REGIMES]
    chunk = os.environ.get("P47N_CHUNK")
    if chunk:
        i, n = (int(v) for v in chunk.split("/"))
        cells = [c for j, c in enumerate(cells) if j % n == i]
    for fam, k, reg in cells:
        tag = f"{fam}/k={k}/{reg}"
        if any(c.get("family") == fam and c.get("k") == k and c.get("regime") == reg
               for c in out["cells"]):
            print("skip existing", tag, flush=True)
            continue
        print("cell", tag, flush=True)
        out["cells"] = [c for c in out["cells"]
                        if not (c.get("family") == fam and c.get("k") == k
                                and c.get("regime") == reg)]
        out["cells"].append(run_cell(X, yr, splits, fam, k, reg))
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
