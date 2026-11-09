"""exp47-N stage 3 - the SSL arms re-pretrained ON THE NOISY unlabelled spectra (2026-11-06).

Pre-registered design and decision rules (N1-N7, written before any of this ran):
    preregistrations/exp47-N_noise_stress_preregistration_v1.md

Why this stage exists. N7's precondition is met in stage 1: the instrument-noise family (`iso`) takes
T1 from 0.996 down to 0.609 (k=2, matched) / 0.831 (k=2, shift) / 0.262 (k=10), i.e. the discrimination
ceiling really is opened by noise. The pre-registration says the section-3.10 claim "the representation
is not the bottleneck" may not be written unconditionally until the learned arms are faced at that same
point - which is the only place a reader could argue we handicapped them: stages 1-2 evaluated encoders
that had never seen a noisy spectrum. Here they are pretrained on the noisy pool, which is what the
infrared self-supervision literature (A3M-CIR line) actually claims robustness about.

What changes relative to stage 1/2, and what does not:
  * changes: the representation is learned (vicreg / barlow / randinit) instead of analytic, and the
    encoder is trained on the noisy rows the fold is allowed to see (matched: noisy library; shift:
    the library is clean, so the encoder is trained clean and tested dirty - both regimes are run).
  * does NOT change: folds (blocked 5-fold, fit blocks<5 / cal blocks>=5 / test blocks%5==k), noise
    seeds and magnitudes, the arm lists (T1 lda/logit/knn/gbdt, T2 gbdt/nls_cal/dictargmin), the
    encoder architecture, SSL_EPOCHS=80 and SEED - identical to the clean Part C run.
    N1-N7 are not edited by this script or by its verdict file.

Grid = the pre-registered minimal probe: clean/k=0 reference + iso/k=2 matched + iso/k=2 shift.
The clean/k=0 cell goes through this same per-fold path so it is poolable with the noisy cells.

Per-fold leak guard (same rule as Part C, restated here because the fold list has length 1): an
encoder may not see any spectrum its own fold tests.

Safety: this writes to logs/exp47_noiseC*.json only. It never calls L.part_b_tea / L.part_c_tea /
L.main, because those dump to L.OUT and would overwrite the clean exp47 ladder JSON.
"""
import os
import sys
import json
import time

# arm lists must be set before _p47_noise is imported (it trims L.MODEL_NAMES_* at import time)
os.environ.setdefault("P47N_T1", "lda,logit,knn,gbdt")
os.environ.setdefault("P47N_T2", "gbdt,nls_cal,dictargmin")


import numpy as np  # noqa: E402

import exp33_whitened_attr as E  # noqa: E402
import exp47_baseline_ladder as L  # noqa: E402
import _p47_noise as N  # noqa: E402
import _p47_noise as N  # noqa: E402

import exp26_nonrecon_ssl as S26  # noqa: E402  (the module that owns the encoder device)
import p5_paths  # noqa: E402

OUT = os.environ.get("P47NC_OUT") or p5_paths.RESULTS + "/exp47_noiseC.json"
CELLS = [c for c in (os.environ.get("P47NC_CELLS") or
                     "clean/0.0/matched,iso/2.0/matched,iso/2.0/shift").split(",") if c]
KINDS = [k for k in (os.environ.get("P47NC_KINDS") or "").split(",") if k] or L.SSL_KINDS
NFOLD_LIM = int(os.environ.get("P47NC_FOLDS") or 0)   # smoke test only; 0 = the real 5 folds


def parse(tag):
    fam, k, reg = tag.split("/")
    return fam, float(k), reg


def run_cell(X, yr, splits, fam, k, regime):
    t0 = time.time()
    folds = []
    for fi, (fit, cal, test) in enumerate(splits):
        tr = np.concatenate([fit, cal])
        model = N.residual_model(X, yr, tr)
        rng = np.random.default_rng(N.fold_seed(fi, fam, k))
        Xn, rms = N.noisy_corpus(X, model, fam, k, regime, test, rng)
        splits1 = [(fit, cal, test)]
        # the encoder is fitted on Xn[tr]; guard it against this fold's test rows
        bad = set(int(i) for i in tr) & set(int(i) for i in test)
        if bad:
            raise RuntimeError(f"SSL encoder would see {len(bad)} spectra its fold tests")
        # ssl_rep_fn caches by (kind, training index set). Every cell of this sweep uses the SAME
        # fold indices with a DIFFERENT matrix, so a stale cache would silently hand the noisy cells
        # the clean embedding. Cleared per fold, before any encoder is asked for.
        L._ssl_cache.clear()
        fold = dict(fold=fi, n_test=int(len(test)),
                    n_test_comp=int(np.isin(yr[test], [7, 8]).sum()),
                    noise_rms_per_channel=N.r3(rms))
        for kind in KINDS:
            fn = L.ssl_rep_fn(Xn, kind)
            t1, errs = L.task_T1(Xn, yr, splits1, f"ssl_{kind}", rep_fn=fn)
            scratch = {}
            exact, micro = L.task_T2(Xn, yr, splits1, f"ssl_{kind}", scratch, rep_fn=fn)
            fold[f"ssl_{kind}"] = dict(T1=t1, T2=dict(exact_set=exact, micro_f1=micro))
            if errs:
                fold.setdefault("errors", []).extend(errs)
        folds.append(fold)
        print(f"  [{fam}/k={k}/{regime}] fold {fi} ssl done "
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
    return N.r3(np.mean(vals)) if vals else None


def aggregate(folds, fam, k, regime, t0):
    out = dict(family=fam, k=k, regime=regime, n_folds=len(folds),
               minutes=N.r3((time.time() - t0) / 60),
               noise_rms_per_channel=[f["noise_rms_per_channel"] for f in folds],
               fold_errors=[e for f in folds for e in f.get("errors", [])])
    out["T1"] = {f"ssl_{kind}": {m: mean_over_folds(folds, [f"ssl_{kind}", "T1", m])
                                 for m in L.MODEL_NAMES_T1} for kind in KINDS}
    out["T1_spread"] = {f"ssl_{kind}": {
        m: N.r3(np.std([f[f"ssl_{kind}"]["T1"][m] for f in folds
                        if f[f"ssl_{kind}"]["T1"].get(m) is not None], ddof=1))
        if len(folds) > 1 else None for m in L.MODEL_NAMES_T1} for kind in KINDS}
    out["T2"] = {f"ssl_{kind}": {m: mean_over_folds(folds, [f"ssl_{kind}", "T2", "exact_set", m])
                                 for m in L.MODEL_NAMES_T2} for kind in KINDS}
    out["T2_micro"] = {f"ssl_{kind}": {m: mean_over_folds(folds, [f"ssl_{kind}", "T2", "micro_f1", m])
                                       for m in L.MODEL_NAMES_T2} for kind in KINDS}
    return out


def main():
    import torch
    X, yr = E.load()
    rng = np.random.default_rng(L.SEED)
    splits = E.make_splits(yr, "blocked", rng)
    if NFOLD_LIM:
        splits = splits[:NFOLD_LIM]
    out = dict(
        prereg=r"preregistrations/exp47-N_noise_stress_preregistration_v1.md",
        stage="stage 3 minimal probe (pre-registration section 4): SSL encoders RE-pretrained on the "
              "noisy unlabelled pool each fold may see; N7's precondition was met in stage 1",
        protocol="folds, noise seeds/magnitudes, arm lists, encoder architecture, SSL_EPOCHS and SEED "
                 "are identical to stages 1-2 and to the clean Part C run - only the representation "
                 "source changes; ssl cache cleared per fold so no cell can inherit another cell's "
                 "embedding; leak guard raises if an encoder sees its fold's test rows",
        cells_requested=CELLS, kinds=KINDS,
        ssl_epochs=L.SSL_EPOCHS, ssl_seed=L.SEED,
        models_T1=list(L.MODEL_NAMES_T1), models_T2=list(L.MODEL_NAMES_T2),
        torch_version=torch.__version__, torch_device=S26.DEV,
        cuda_available=bool(torch.cuda.is_available()),
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        n_folds_limit=(NFOLD_LIM or None),
        window=E.WINDOW, seed_base=N.SEED_BASE, n_spectra=int(len(X)), n_channels=int(X.shape[1]))

    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            prev.update(out)
            out = prev
        except Exception:
            pass
    out.setdefault("cells", [])

    for tag in CELLS:
        fam, k, reg = parse(tag)
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
