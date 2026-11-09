"""Self-contained readers for the two non-KAN MIR deposits on disk that record an adulteration
level: the honey/sugar panel and the coconut-milk/water panel.

Both are read the same way as `p5_public_data`: spectra as rows, wavenumber axis ascending,
`y` = declared level in percent, `groups` = independent preparation. Nothing here reuses the
paper8 loaders, so this module can ship with the reproduction package.

Documented data facts (measured, not assumed):
  honey    65 rows, axis 4000 -> 650 cm-1 in 1.0 steps (3351 parseable channels + 90 stray
           'Unnamed' columns, dropped). Values 40.8-101.1 with median 95.9 => single-beam
           percent-transmittance, so A = -log10(T/100) is part of the protocol (the cinnamon
           lesson). Levels 0/10/20/40 with 16 rows each plus one 100% sugar row: names
           SampleN (level 0) and SampleN-<level>, so the 16 preparations are paired across
           levels -> group-disjoint band splits are available.
  coconut  42 rows, axis 2501.4 -> 4000.2 cm-1 (729 channels, step 1.29), values 0.002-0.341
           all positive => absorbance-like as delivered, no conversion applied. Two milk types
           x levels {0,10,20} x 7 rows; no sample identifiers, so each row is treated as one
           independent preparation and the panel is split by milk type (the level-0 control is
           type-specific). Region is NIR/transmission, not ATR fingerprint - kept as a
           cross-sampling-mode audit row, flagged in `meta`.
"""
import os
import re
import numpy as np
import pandas as pd
import p5_paths  # noqa: E402

ROOT = p5_paths.PUBLIC_DATA


def _numeric_axis(cols):
    out = []
    for c in cols:
        s = str(c).strip()
        if re.fullmatch(r"[\d.eE+-]+", s):
            out.append((c, float(s)))
    return out


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def to_absorbance(T):
    """percent transmittance -> absorbance; non-positive channels are dropped later."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.clip(np.asarray(T, float), 1e-6, None) / 100.0)


def orient(wvn, X):
    """return ascending-axis spectra (n, p)."""
    wvn = np.asarray(wvn, float)
    if wvn[0] > wvn[-1]:
        wvn, X = wvn[::-1], X[:, ::-1]
    return wvn, X


def _finish(X, y, groups, ids, wvn, meta):
    X = np.asarray(X, float)
    ok = np.isfinite(X).all(1)
    X, y, groups, ids = X[ok], np.asarray(y, float)[ok], np.asarray(groups, object)[ok], np.asarray(ids, object)[ok]
    wvn, X = orient(wvn, X)
    keep = np.isfinite(X).all(0)
    X, wvn = X[:, keep], wvn[keep]
    order = np.lexsort((y, np.asarray(groups, object)))       # deterministic row order
    return dict(X=X[order], y=y[order], groups=np.asarray(groups, object)[order],
                ids=np.asarray(ids, object)[order], wvn=wvn, meta=meta)


def load_honey():
    p = os.path.join(ROOT, "honey_ftir_extract", "ftir.csv")
    h = pd.read_csv(p)
    pairs = _numeric_axis(h.columns[3:])
    cols = [c for c, _ in pairs]
    wvn = [w for _, w in pairs]
    T = h[cols].to_numpy(float)
    X = to_absorbance(T)
    y = h["Sugar rate"].to_numpy(float)
    names = h["Sample"].astype(str).to_numpy()
    groups = np.array([re.sub(r"-\d+$", "", s) for s in names], object)
    meta = dict(name="HONEY", matrix="honey", adulterant="sucrose", instrument="FTIR (transmission)",
                unit="converted from %T", n_raw=int(len(h)), stray_columns=int(len(h.columns) - 3 - len(cols)))
    d = _finish(X, y, groups, names, wvn, meta)
    d["dnu"] = float(np.median(np.diff(d["wvn"])))
    return d


def load_coconut(split_types=True):
    p = os.path.join(ROOT, "coconut", "coconut_milk_FTIR.xlsx")
    c = pd.read_excel(p)
    pairs = _numeric_axis(c.columns[2:])
    cols = [x for x, _ in pairs]
    wvn = [w for _, w in pairs]
    X = c[cols].to_numpy(float)
    y = c["Add water (%)"].to_numpy(float)
    types = c["Coconut milk type"].astype(str).to_numpy()
    ids = np.array(["%s#%d" % (re.sub(r"\W+", "_", t)[:18], i) for i, t in enumerate(types)], object)
    out = {}
    for t in np.unique(types):
        m = types == t
        tag = "instant" if "instant" in t.lower() else "traditional"
        meta = dict(name="COCO_" + tag, matrix="coconut milk", adulterant="water",
                    instrument="FTIR, 2501-4000 cm-1 (NIR region, not ATR fingerprint)",
                    unit="as delivered", type=t)
        out[tag] = _finish(X[m], y[m], ids[m], ids[m], wvn, meta)
        out[tag]["dnu"] = float(np.median(np.diff(out[tag]["wvn"])))
    return out


if __name__ == "__main__":
    d = load_honey()
    lev, cnt = np.unique(np.round(d["y"], 4), return_counts=True)
    print("HONEY n=%d p=%d groups=%d levels=%s counts=%s axis %.0f-%.0f dnu=%.3f stray=%d" %
          (len(d["y"]), d["X"].shape[1], len(set(d["groups"])), list(lev), list(cnt),
           d["wvn"][0], d["wvn"][-1], d["dnu"], d["meta"]["stray_columns"]))
    tab = {}
    for g, y in zip(d["groups"], d["y"]):
        tab.setdefault(g, set()).add(round(float(y), 2))
    print("  preparations with all four of 0/10/20/40:",
          sum(1 for v in tab.values() if v == {0.0, 10.0, 20.0, 40.0}), "/", len(tab),
          "| group size distribution", sorted({len(v) for v in tab.values()}))
    print("  absorbance stats min %.4f median %.4f max %.4f" %
          (d["X"].min(), np.median(d["X"]), d["X"].max()))
    for tag, d in load_coconut().items():
        lev, cnt = np.unique(np.round(d["y"], 4), return_counts=True)
        print("COCO %-12s n=%d p=%d groups=%d levels=%s counts=%s axis %.0f-%.0f min %.5f med %.5f max %.5f" %
              (tag, len(d["y"]), d["X"].shape[1], len(set(d["groups"])), list(lev), list(cnt),
               d["wvn"][0], d["wvn"][-1], d["X"].min(), np.median(d["X"]), d["X"].max()))
