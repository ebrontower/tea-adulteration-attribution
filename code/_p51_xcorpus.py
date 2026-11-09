"""exp51 - the attribution pipeline evaluated corpus by corpus, with no primary dataset.

Pre-registration (written before this run, rules fixed there):
    preregistrations/exp51_cross_corpus_evaluation_v1.md

Why. exp47's B3 showed the whitened metric's gain flips sign with the corpus (tea +0.116 over 9/9
model classes, cinnamon -0.089 over 4/9), so a single flagship matrix cannot carry the claim. The
author's decision on 2026-11-03 was to drop the primary/secondary split and report the method
corpus by corpus. This script runs the exp40 pipeline (increment dictionary, truncated whitened
metric, split-conformal composition sets, locally normalised abstention gate, block-cluster
bootstrap) on every public dose corpus on disk, one row per corpus, no averaging across rows, and
computes the pre-registered S1-S5 verdicts mechanically, so they cannot be re-read after the result.

Fixed design points, all taken from the pre-registration:
  * "which component" is only a question when a matrix carries >= 2 candidate adulterants, so naming
    is evaluated on tea (imported: 6 adulterants + 2 mixtures) and the pooled cinnamon corpus (3).
    On single-adulterant dilution corpora the composition set degenerates to {none, that one}; the
    same code path is still run there, because it measures the detection/set half of the guarantee.
  * graded level claims are admissible only where exp45's independent audit found a usable dose
    axis (D1 abs+snv, D2 snv, D7p, HONEY, COCO_instant). Non-monotone / aliased corpora (D3, D7w,
    D7d, D8, COCO_traditional) still report set behaviour; their MAF is computed and labelled
    inadmissible rather than dropped.
  * folds split preparations, never spectra of one preparation; dictionary and whitener are fit-pool
    only; alpha = 0.10; k_w = 60 with a mean-tail floor; local scale from 5 neighbours.
  * SNV is a contrast column, never merged with the absorbance column.
  * zero new measurement - everything is recomputation on deposited spectra.

Reproduction gate: the pooled cinnamon corpus is run through exp40's own loader and folds and the
standard-arm globals are asserted against exp40_public.json (Table 8) to 1e-9.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = p5_paths.CODE
sys.path.insert(0, HERE)

import exp40_public_pipeline as P                                   # noqa: E402
import ext_data as ED                                               # noqa: E402
import p5_data as PD                                                # noqa: E402
import p5_paths  # noqa: E402

OUT = os.environ.get("P51_OUT") or p5_paths.RESULTS + "/exp51_xcorpus.json"
R2 = p5_paths.RESULTS
EXP40_JSON = p5_paths.RESULTS + "/exp40_public.json"
EXP45_JSON = p5_paths.RESULTS + "/exp45_external_validation.json"
EXP33_JSON = p5_paths.RESULTS + "/exp33_whitened_attr.json"

WINDOW = (900.0, 1800.0)
MIN_WINDOW_CHANNELS = 200
NF = 5
ALPHA = 0.10
METRICS = ("euclid", "white_a")
PREPS = ("abs", "snv")
KEYS = ("covered", "size", "empty", "rank")
STD_KEYS = ("covered_std", "size_std", "empty_std", "rank_std")
TOP1_CRIT, COV_CRIT, ACCEPT_CRIT, ACC_TOP1_CRIT = 0.80, 0.90, 0.50, 0.80

# Fallback only; the real gate is read from exp45_external_validation.json at run time.
DOSE_ADMISSIBLE_FALLBACK = {"D1": ("abs", "snv"), "D2": ("snv",), "D7p": ("abs",),
                            "HONEY": ("abs",), "COCO_instant": ("abs",),
                            "D7_pooled": ("abs", "snv")}


# --------------------------------------------------------------------------- corpus loading
def band(X, wvn):
    """The deposited fingerprint window if the corpus covers it, else its own full axis."""
    ax = np.asarray(wvn, float)
    m = (ax >= WINDOW[0]) & (ax <= WINDOW[1])
    if int(m.sum()) >= MIN_WINDOW_CHANNELS:
        return X[:, m], ax[m], list(WINDOW)
    return X, ax, [float(ax[0]), float(ax[-1])]


def single_atom(X_full, y, groups, wvn, label, records_per_block=1):
    """comp 0 = authentic, 1 = the corpus's one adulterant; level = declared %; blocks = preparations
    (or name-adjacent runs of `records_per_block` spectra, used for the cinnamon series)."""
    Xb, ax, win = band(np.asarray(X_full, float), wvn)
    lv = np.round(np.asarray(y, float), 4)
    g = np.asarray(groups, object)
    comp = (lv > 0).astype(int)
    block = np.zeros(len(lv), int)
    stratum = np.zeros(len(lv), int)
    for i, l in enumerate(np.unique(lv)):
        m = np.where(lv == l)[0]
        if records_per_block == 1:
            gs = sorted(set(g[m]))
            pos = {q: j for j, q in enumerate(gs)}
            block[m] = [pos[v] for v in g[m]]
        else:
            order = np.argsort(np.asarray(g, object)[m], kind="stable")
            mm = m[order]
            block[mm] = np.arange(len(mm)) // records_per_block
            stratum[mm] = i
            continue
        stratum[m] = i
    return dict(X=Xb, wvn=ax, window=win, comp=comp, level=lv, block=block, stratum=stratum,
                groups=g, atoms=[1], labels=[label], n_candidates=2)


def kan(name, label):
    d = PD.load_public(name)
    return single_atom(d["X"], d["y"], d["groups"], d["wvn"], label)


def cinnamon_series(key, label):
    d = PD.load_public(key)
    ax = np.asarray(d["wvn"], float)
    nm = np.asarray(d["groups"], dtype=object).astype(str)
    o = np.argsort(nm, kind="stable")            # the sample name carries the run order
    return single_atom(np.asarray(d["X"], float)[o], np.asarray(d["y"], float)[o], nm[o], ax,
                       label, records_per_block=5)


def honey():
    d = ED.load_honey()
    keep = np.round(d["y"], 2) < 99.0            # the 1-row 100% sucrose endmember is excluded
    return single_atom(d["X"][keep], d["y"][keep], d["groups"][keep], d["wvn"], "sucrose")


def coconut(tag):
    d = ED.load_coconut()[tag]
    return single_atom(d["X"], d["y"], d["groups"], d["wvn"], "water")


def corpora():
    yield dict(name="D1", matrix="olive oil", adulterant="sunflower oil",
               source="KAN / Batayhi et al., Foods 2026 (D1)", naming="no",
               data=kan("D1", "sunflower oil"))
    yield dict(name="D2", matrix="roasted coffee", adulterant="malt flour",
               source="KAN / Batayhi et al., Foods 2026 (D2)", naming="no",
               data=kan("D2", "malt flour"))
    yield dict(name="D3", matrix="orange juice", adulterant="apple juice",
               source="KAN / Batayhi et al., Foods 2026 (D3)", naming="no",
               data=kan("D3", "apple juice"))
    yield dict(name="D8", matrix="turmeric", adulterant="wheat flour",
               source="Mendeley cb5kbt7bgv.2 (D8)", naming="no",
               data=kan("D8", "wheat flour"))
    for key, lab in (("D7w", "walnut shell"), ("D7p", "peanut"), ("D7d", "wood dust")):
        yield dict(name=key, matrix="cinnamon", adulterant=lab,
                   source="Mendeley rtbsf2r53x.1 (" + key + ")", naming="no",
                   data=cinnamon_series(key, lab))
    yield dict(name="HONEY", matrix="honey", adulterant="sucrose",
               source="honey_ftir_extract/ftir.csv, %T -> A", naming="no", data=honey())
    yield dict(name="COCO_traditional", matrix="coconut milk (traditional)", adulterant="water",
               source="coconut_milk_FTIR.xlsx, NIR transmission", naming="no",
               data=coconut("traditional"))
    yield dict(name="COCO_instant", matrix="coconut milk (instant)", adulterant="water",
               source="coconut_milk_FTIR.xlsx, NIR transmission", naming="no",
               data=coconut("instant"))


# --------------------------------------------------------------------------- folds
def blocked_folds(stratum, block, nf=NF):
    """Fold k tests the blocks whose order index is = k (mod nf) inside every level stratum. The
    non-test blocks of a stratum split by order: the lower half fits the dictionary, the upper half
    calibrates. For a stratum of ten blocks this is exactly exp40's rule (blocks<5 fit, >=5 cal)."""
    triples = []
    for k in range(nf):
        fit, cal, test = [], [], []
        for st in np.unique(stratum):
            m = stratum == st
            ids = sorted(set(block[m]))
            half = len(ids) // 2
            for j, b in enumerate(ids):
                rows = np.where(m & (block == b))[0]
                if j % nf == k:
                    test += list(rows)
                elif j < half:
                    fit += list(rows)
                else:
                    cal += list(rows)
        triples.append((np.sort(np.array(fit, int)), np.sort(np.array(cal, int)),
                        np.sort(np.array(test, int))))
    return triples


# --------------------------------------------------------------------------- statistics
def stat(rr, keys=KEYS):
    kc, ks, ke, kr = keys
    if not rr:
        return None
    acc = [r for r in rr if r["accepted"]]
    return dict(n=len(rr),
                top1=round(float(np.mean([r[kr] == 0 for r in rr])), 3),
                coverage=round(float(np.mean([r[kc] for r in rr])), 3),
                size=round(float(np.mean([r[ks] for r in rr])), 2),
                empty=round(float(np.mean([r[ke] for r in rr])), 3),
                accept=round(float(np.mean([r["accepted"] for r in rr])), 3),
                top1_acc=round(float(np.mean([r[kr] == 0 for r in acc])), 3) if acc else None,
                n_accepted=len(acc))


def rekey(rows, arm):
    if arm == "norm":
        return rows
    return [{**r, "covered": r["covered_std"], "size": r["size_std"],
             "empty": r["empty_std"], "rank": r["rank_std"]} for r in rows]


def maf(levels, per_level, crit):
    ls = sorted(levels)
    for i, l in enumerate(ls):
        if crit(per_level.get(l)) and all(crit(per_level.get(x)) for x in ls[i + 1:]):
            return l, "monotone tail holds"
    fails = [l for l in ls if not crit(per_level.get(l))]
    return None, f"no level satisfies the criterion together with all higher levels; failing: {fails}"


CRITS = dict(naming=lambda s: bool(s and s["top1"] >= TOP1_CRIT),
             certified_set=lambda s: bool(s and s["coverage"] >= COV_CRIT),
             deliverable=lambda s: bool(s and s["accept"] >= ACCEPT_CRIT
                                        and (s["top1_acc"] or 0.0) >= ACC_TOP1_CRIT))


def run_one(Z, comp, level, block, cls_a, cand, atoms, labels, splits, metric):
    """One arm on one corpus: per-spectrum records, per-fold thresholds, per-fold summaries."""
    rows, qs, fst = [], [], []
    for fit, cal, test in splits:
        gate_cal = cal[comp[cal] > 0]
        rr, q, tau, tau_abs, q_std = P.fold_rows(Z, comp, level, block, cls_a, cls_a,
                                                 fit, cal, test, cand, atoms, ALPHA, metric,
                                                 gate_cal, labels=labels)
        rows += rr
        qs.append(dict(q_norm=q, q_std=q_std, tau=tau, tau_abs=tau_abs,
                       n_fit=int(len(fit)), n_cal=int(len(cal)), n_test=int(len(test))))
        s = stat(rr, STD_KEYS) or {}
        fst.append(dict(test_n=int(len(test)), top1=s.get("top1"), coverage=s.get("coverage"),
                        size=s.get("size"), empty=s.get("empty"), accept=s.get("accept")))
    return rows, qs, fst


def pack_arm(rows, lvls, qs, fst, intervals):
    std = rekey(rows, "std")
    per = {l: stat([r for r in std if r["level"] == l and r["comp"] > 0]) for l in lvls}
    per_norm = {l: stat([r for r in rows if r["level"] == l and r["comp"] > 0]) for l in lvls}
    return dict(
        all=stat(std), all_norm=stat(rows),
        pure=stat([r for r in std if r["comp"] == 0]),
        pure_norm=stat([r for r in rows if r["comp"] == 0]),
        adulterated=stat([r for r in std if r["comp"] > 0]),
        by_level=per, by_level_norm=per_norm,
        intervals=intervals,
        q_std_per_fold=[round(v["q_std"], 4) for v in qs],
        q_norm_per_fold=[round(v["q_norm"], 4) for v in qs],
        tau_per_fold=[round(v["tau"], 4) for v in qs],
        n_fit_per_fold=[v["n_fit"] for v in qs],
        n_cal_per_fold=[v["n_cal"] for v in qs],
        n_test_per_fold=[v["n_test"] for v in qs],
        fold_summary_std=fst)


def geometry_one(Z, comp, cls_a, atoms, labels):
    """Dictionary geometry on the whole corpus (descriptive, same convention as exp40's geometry()).
    With a single atom the coherence / condition-number columns are vacuous, so the informative
    number is the gauge cosine: how much of the increment is a rescaling of the authentic spectrum
    in the raw space, and how much of an orthogonal direction survives whitening."""
    idx = np.arange(len(comp))
    m0, D, sigma = P.fit_dict(Z, comp, idx, n_atoms=len(atoms))
    W = P.Whitener(Z, cls_a, idx)
    Wc = W.coords(D.T)
    wn = np.linalg.norm(Wc, axis=1)
    Wm = W.coords(m0[None, :])[0]
    nrm = np.linalg.norm(D, axis=0)
    gauge_raw = (D.T @ m0) / (nrm * np.linalg.norm(m0))
    gauge_white = (Wc @ Wm) / (wn * np.linalg.norm(Wm))
    Dn = D / nrm[None, :]
    iu = np.triu_indices(len(atoms), 1)
    Gn = (Wc @ Wc.T) / np.outer(wn, wn)
    return dict(n_atoms=len(atoms), atoms=[labels[a - 1] for a in atoms],
                sigma_rms=round(float(sigma), 6),
                incr_norm_raw=[round(float(v), 4) for v in nrm],
                incr_norm_white=[round(float(v), 3) for v in wn],
                incr_snr_raw=[round(float(v / sigma), 2) for v in nrm],
                incr_snr_white=[round(float(v / sigma), 2) for v in wn],
                gauge_raw=[round(float(v), 3) for v in gauge_raw],
                gauge_white=[round(float(v), 3) for v in gauge_white],
                absmax_gauge_raw=round(float(np.max(np.abs(gauge_raw))), 3),
                absmax_gauge_white=round(float(np.max(np.abs(gauge_white))), 3),
                coherence_raw=(round(float((Dn.T @ Dn)[iu].max()), 3) if len(atoms) > 1 else None),
                coherence_white=(round(float(Gn[iu].max()), 3) if len(atoms) > 1 else None),
                cond_raw=(round(float(np.linalg.cond(Dn.T @ Dn)), 2) if len(atoms) > 1 else None),
                cond_white=(round(float(np.linalg.cond(Gn)), 2) if len(atoms) > 1 else None),
                evr=round(W.evr, 5), dim=int(W.dim), delta=round(W.delta, 8),
                rank_deficient=bool(W.dim < P.K_WHITE))


# --------------------------------------------------------------------------- main
def whitener_per_fold(Y, cls_a, splits):
    """The whitener each fold actually used: truncated dimension and tail floor on that fit pool.

    Reported per fold because the corpus-level fit is not what scores the test rows - on the small
    deposits a fold's fit pool carries fewer residual directions than k_w, which makes the tail
    floor (and with it every whitened distance) numerically degenerate.
    """
    dims, deltas, nres = [], [], []
    for fit, cal, test in splits:
        W = P.Whitener(Y, cls_a, fit)
        dims.append(int(W.dim))
        deltas.append(float(W.delta))
        nres.append(int(len(fit)))
    return dict(dim=dims, delta=[f"{v:.3e}" for v in deltas], fit_rows=nres,
                rank_deficient=bool(min(dims) < P.K_WHITE))


def arm_table(Z, comp, level, block, stratum, atoms, labels, splits, admissible, boot=True):
    cand = P.subsets(len(atoms))
    lvls = sorted(set(float(v) for v in np.unique(level) if v > 0))
    cls_a = comp.copy()
    arms, geos = {}, {}
    for prep in PREPS:
        Y = Z if prep == "abs" else ED.snv(Z)
        geos[prep] = geometry_one(Y, comp, cls_a, atoms, labels)
        geos[prep]["per_fold"] = whitener_per_fold(Y, cls_a, splits)
        geos[prep]["rank_deficient"] = geos[prep]["per_fold"]["rank_deficient"]
        for metric in METRICS:
            rows, qs, fst = run_one(Y, comp, level, block, cls_a, cand, atoms, labels,
                                    splits, metric)
            std = rekey(rows, "std")
            iv = P.cluster_boot(std, ["coverage", "size", "top1", "abstain"]) if boot else {}
            a = pack_arm(rows, lvls, qs, fst, iv)
            a["maf"] = {k: maf(lvls, a["by_level"], fn) for k, fn in CRITS.items()}
            a["maf_norm"] = {k: maf(lvls, a["by_level_norm"], fn) for k, fn in CRITS.items()}
            a["maf_admissible"] = bool(prep in admissible)
            a["levels"] = lvls
            a["n_candidates"] = len(cand)
            arms[f"{prep}/{metric}"] = a
    return arms, geos, lvls, len(cand)


def main():
    audit = json.load(open(EXP45_JSON, encoding="utf-8"))["corpora"]
    out = dict(prereg=r"preregistrations/exp51_cross_corpus_evaluation_v1.md",
               alpha=ALPHA, k_white=P.K_WHITE, k_local=P.K_LOCAL, nf=NF,
               window=list(WINDOW), min_window_channels=MIN_WINDOW_CHANNELS,
               metrics=list(METRICS), preps=list(PREPS),
               criteria=dict(top1=TOP1_CRIT, coverage=COV_CRIT, accept=ACCEPT_CRIT,
                             top1_on_accepted=ACC_TOP1_CRIT, monotone_tail=True),
               dose_admissibility_source="exp45_external_validation.json, audited 2026-11-01, "
                                         "independent of this run",
               corpora={}, verdicts={})

    for spec in corpora():
        nm, d = spec["name"], spec["data"]
        rec_a = audit.get(nm, {})
        adm = rec_a.get("usable_in_metrics") or list(DOSE_ADMISSIBLE_FALLBACK.get(nm, ()))
        splits = blocked_folds(d["stratum"], d["block"])
        t0 = time.time()
        arms, geos, lvls, ncand = arm_table(d["X"], d["comp"], d["level"], d["block"],
                                            d["stratum"], d["atoms"], d["labels"], splits, adm)
        out["corpora"][nm] = dict(
            matrix=spec["matrix"], adulterant=spec["adulterant"], source=spec["source"],
            naming_claim=spec["naming"], n_spectra=int(len(d["comp"])),
            n_channels=int(d["X"].shape[1]), window=[round(v, 1) for v in d["window"]],
            levels=lvls, n_candidates=ncand,
            blocks_per_stratum=[int(len(set(d["block"][d["stratum"] == s])))
                                for s in np.unique(d["stratum"])],
            fold_sizes=dict(fit=[len(f) for f, c, t in splits], cal=[len(c) for f, c, t in splits],
                            test=[len(t) for f, c, t in splits]),
            dose_axis=rec_a.get("dose_axis", "unknown"), dose_admissible_in=list(adm),
            aliasing_flag=rec_a.get("aliasing_flag"), geometry=geos, arms=arms,
            minutes=round((time.time() - t0) / 60, 2))
        r = out["corpora"][nm]
        print(f"[{nm}] {r['n_spectra']}x{r['n_channels']} win={r['window']} dose={r['dose_axis']} "
              f"top1 euclid={r['arms']['abs/euclid']['all']['top1']} "
              f"white={r['arms']['abs/white_a']['all']['top1']} "
              f"cov {r['arms']['abs/euclid']['all']['coverage']}/"
              f"{r['arms']['abs/white_a']['all']['coverage']} "
              f"MAF {r['arms']['abs/white_a']['maf']['naming'][0]}/"
              f"{r['arms']['abs/euclid']['maf']['naming'][0]} "
              f"gauge {r['geometry']['abs']['absmax_gauge_raw']}", flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)

    # ---------------- pooled cinnamon: the only public corpus with a naming question
    X, comp, level, blk, series, n_dup, nch = P.load_cinnamon()
    stratum7 = np.zeros(len(comp), int)
    for s in (1, 2, 3):
        for lv in np.unique(level[series == s]):
            stratum7[(series == s) & (level == lv)] = 100 * s + int(lv)
    t0 = time.time()
    arms, geos, lvls, ncand = arm_table(X, comp, level, blk, stratum7, [1, 2, 3], P.NAME,
                                       P.blocked_folds(blk),
                                       DOSE_ADMISSIBLE_FALLBACK["D7_pooled"])
    pooled = dict(matrix="cinnamon", adulterant=list(P.NAME),
                  source="Mendeley rtbsf2r53x.1, D7w+D7p+D7d pooled", naming_claim="yes",
                  n_spectra=int(len(comp)), n_channels=int(nch), window=list(WINDOW),
                  levels=lvls, n_candidates=ncand, duplicate_rows=n_dup,
                  folds="exp40's own blocked_folds (five name-adjacent records per block)",
                  geometry=geos, arms=arms, minutes=round((time.time() - t0) / 60, 2))
    ref = json.load(open(EXP40_JSON, encoding="utf-8"))
    checks = {}
    for metric in METRICS:
        want = ref["arms"][f"blocked/{metric}/std"]["all"]
        got = arms[f"abs/{metric}"]["all"]
        pairs = dict(coverage=got["coverage"], size=got["size"], top1=got["top1"],
                     abstain=got["empty"], accept=got["accept"])
        checks[f"blocked/{metric}/std"] = ({k: [v, want[k]] for k, v in pairs.items()
                                            if abs(v - want[k]) > 1e-9}
                                           or "identical to exp40_public.json")
    pooled["reproduce_exp40"] = checks
    pooled["open_set_reference"] = ref.get("open_set")
    pooled["confusion_reference"] = ref.get("confusion_white_a")
    out["corpora"]["D7_pooled"] = pooled
    print("[D7_pooled] reproduce exp40:", json.dumps(checks, ensure_ascii=False), flush=True)

    # ---------------- tea: imported, not re-run (multi-component sets need the tea pipeline)
    e33 = json.load(open(EXP33_JSON, encoding="utf-8"))
    tea = dict(matrix="tea", adulterant="sugar/talc/sulfur/paraffin/dye/flavouring (+2 mixtures)",
               source="self-acquired ATR-FTIR (this study)", naming_claim="yes",
               provenance="imported from exp33_whitened_attr.json (recorded blocked-fold aggregate); "
                          "NOT re-run through exp51's generic path, because the tea dictionary "
                          "carries multi-component sets and mixtures",
               geometry=e33["geometry"], candidates=e33["candidates"], window=e33["window"],
               arms={})
    for metric, tag in (("euclid", "euclid"), ("white_a", "white")):
        A, As = e33[tag]["blocked"]["A"], e33[tag]["blocked"]["A_single"]
        tea["arms"][f"abs/{metric}"] = dict(
            all=dict(coverage=A["coverage"], size=A["mean_size"], top1=A["point_exact"]),
            multi_component=dict(coverage=A["coverage"], size=A["mean_size"],
                                 exact_set=A["point_exact"], top3=A["top3"], top5=A["top5"],
                                 n_candidates=e33["candidates"]),
            single_component=dict(coverage=As["coverage"], size=As["mean_size"],
                                  exact_set=As["point_exact"]))
    out["corpora"]["TEA_imported"] = tea
    out["dose_axis_audit"] = {nm: dict(dose_axis=rec_a.get("dose_axis"),
                                       usable_in=rec_a.get("usable_in_metrics"),
                                       aliasing=bool(rec_a.get("aliasing_flag")))
                              for nm, rec_a in audit.items()}

    verdicts(out)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(out["verdicts"], indent=1, ensure_ascii=False)[:8000])
    print("saved", OUT)


# --------------------------------------------------------------------------- S1-S5
def verdicts(out):
    C = out["corpora"]
    singles = [nm for nm, r in C.items() if r.get("naming_claim") == "no"]

    s1 = {}
    for nm in singles:
        e, w = C[nm]["arms"]["abs/euclid"]["all"], C[nm]["arms"]["abs/white_a"]["all"]
        cols = dict(coverage=(e["coverage"], w["coverage"], w["coverage"] > e["coverage"]),
                    size=(e["size"], w["size"], w["size"] < e["size"]),
                    empty=(e["empty"], w["empty"], w["empty"] < e["empty"]))
        s1[nm] = dict(per_column={k: [v[0], v[1], bool(v[2])] for k, v in cols.items()},
                      n_better=int(sum(v[2] for v in cols.values())),
                      none_better=bool(not any(v[2] for v in cols.values())))
    n_none = sum(1 for v in s1.values() if v["none_better"])
    out["verdicts"]["S1"] = dict(
        fired=bool(2 * n_none >= len(s1)), corpora_with_no_benefit=n_none,
        n_single_atom_corpora=len(s1), per_corpus=s1,
        rule="if whitening beats euclid on none of coverage / set size / abstention for at least "
             "half of the single-adulterant corpora, the metric claim is restricted to multi-atom "
             "dictionaries and those corpora become a geometry display only")

    s2 = {}
    for nm, r in C.items():
        if nm == "TEA_imported":
            continue
        for key, a in r["arms"].items():
            if not key.endswith("/white_a"):
                continue
            cov, nmin = a["all"]["coverage"], min(a["n_cal_per_fold"])
            if cov < 0.85 or nmin < 20:
                s2[f"{nm}/{key}"] = dict(coverage=cov, min_n_cal=nmin,
                                         n_cal_per_fold=a["n_cal_per_fold"],
                                         whitener_dim_per_fold=r["geometry"]
                                         [key.split("/")[0]]["per_fold"]["dim"],
                                         rank_deficient=r["geometry"][key.split("/")[0]]
                                         ["rank_deficient"])
    out["verdicts"]["S2"] = dict(fired=bool(s2), offenders=s2,
                                 rule="whitened marginal coverage below 0.85, or a calibration pool "
                                      "below 20 spectra in any fold: report the interval and the "
                                      "pool size, do not quote the number as a guarantee")

    gauges, benefits = {}, {}
    for nm, r in C.items():
        if nm == "TEA_imported":
            continue
        gauges[nm] = r["geometry"]["abs"]["absmax_gauge_raw"]
        benefits[nm] = round(float(r["arms"]["abs/white_a"]["all"]["top1"]
                                   - r["arms"]["abs/euclid"]["all"]["top1"]), 3)
    ks = sorted(gauges)
    rho = spearmanr([gauges[k] for k in ks], [benefits[k] for k in ks])
    out["verdicts"]["S3"] = dict(
        fired=bool(abs(float(rho.correlation)) < 0.3),
        spearman_rho=round(float(rho.correlation), 3), p=round(float(rho.pvalue), 4),
        gauge_raw_abs=gauges, top1_benefit_white_minus_euclid=benefits, n_corpora=len(ks),
        rule="if the raw gauge cosine does not rank the whitening benefit across corpora "
             "(|rho| < 0.3), drop 'the dictionary condition number is a priori predictive' and "
             "report the benefit per corpus instead")

    s4 = {}
    for nm, r in C.items():
        if nm == "TEA_imported":
            continue
        for key, a in r["arms"].items():
            if not key.endswith("/white_a") or not a["maf_admissible"]:
                continue
            e = C[nm]["arms"][key.replace("white_a", "euclid")]
            wn, en = a["maf"]["naming"][0], e["maf"]["naming"][0]
            if wn is not None and (en is None or wn > en):
                s4[f"{nm}/{key}"] = dict(maf_white=wn, maf_euclid=en)
    out["verdicts"]["S4"] = dict(fired=bool(s4), offenders=s4,
                                 rule="any admissible corpus whose whitened naming MAF is worse "
                                      "(higher) than the euclidean one must be reported as a pair "
                                      "of curves, never as the favourable direction alone")

    s5 = {}
    for nm, r in C.items():
        if nm == "TEA_imported":
            continue
        chance = round(1.0 / r["arms"]["abs/euclid"]["n_candidates"], 3)
        for key, a in r["arms"].items():
            if key not in ("abs/euclid", "abs/white_a"):
                continue
            lo, hi = a["intervals"]["top1"]
            if lo <= chance <= hi:
                s5[f"{nm}/{key}"] = dict(top1=[a["all"]["top1"], [lo, hi]], chance_line=chance)
    out["verdicts"]["S5"] = dict(fired=bool(s5), offenders=s5,
                                 rule="a cluster-bootstrap interval on top-1 that covers the "
                                      "random-selection line must be written as 'not separable "
                                      "from chance', with no point estimate")
    out["verdicts"]["reproduce_exp40"] = C["D7_pooled"]["reproduce_exp40"]
    out["verdicts"]["rank_deficient_whiteners"] = {
        f"{nm}/{p}": dict(rank_deficient=r["geometry"][p]["rank_deficient"],
                          dim_per_fold=r["geometry"][p]["per_fold"]["dim"],
                          tail_floor_per_fold=r["geometry"][p]["per_fold"]["delta"])
        for nm, r in C.items() if nm != "TEA_imported" for p in PREPS}


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("total minutes", round((time.time() - t0) / 60, 2))
