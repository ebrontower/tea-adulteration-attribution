"""p5_public_data - reader for the seven public ATR-FTIR adulteration corpora.

The data are not redistributed with this package. Download the depositions cited in the
manuscript and put them in one folder, then point the environment variable
P5_PUBLIC_DATA at it:

    <P5_PUBLIC_DATA>/
      kan/
        kan_oliveoil.xlsx     (or "zeytinyaghi tagdis.xlsx"), sheet Sayfa7
        kan_coffee.xlsx       (or "Kahve_Tagdis.xlsx"),       sheet Sayfa4
        kan_juice.xlsx        (or "meyve suyu tagdis.xlsx"),  sheet Sayfa2
      cinnamon/
        FTIR_data_Walnut.xlsx
        FTIR_data_peanut.xlsx
        FTIR_data_wood_dust.xlsx
      turmeric/
        FTIR_Data.csv

Sources (Section 2.7 and Data availability):
    kan/       https://github.com/abdulhamidbatayhi123/kan-ftir-food-adulteration
    cinnamon/  https://doi.org/10.17632/rtbsf2r53x.1
    turmeric/  https://doi.org/10.17632/cb5kbt7bgv.2

Every loader returns the same dictionary:

    X      (n, N) absorbance, wavenumber axis ascending, no interpolation
    y      (n,)   adulterant mass fraction in percent
    groups (n,)   physical-sample key (leakage-safe grouping)
    wvn    (N,)   wavenumbers, cm^-1, ascending
    dnu    float  median channel spacing
    ids    (n,)   sample identifier as recorded in the file
    spec   dict   matrix, adulterant, instrument, source, licence

Spectrum counts, level counts and channel counts are read from the files at run time and
never hard-coded. The three workbooks of the KAN trio carry the quirks described in the
comments below; each one has to be reproduced, because ignoring it either leaks replicate
scans of the same physical sample across a split or mislabels a concentration level.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import p5_paths  # noqa: E402

ROOT = p5_paths.PUBLIC_DATA

KAN_SHEETS = {
    "oil": dict(file="kan_oliveoil.xlsx", orig="zeytinyağı tağşiş.xlsx", sheet="Sayfa7", header=1),
    "coffee": dict(file="kan_coffee.xlsx", orig="Kahve_Tağşiş.xlsx", sheet="Sayfa4", header=0),
    "juice": dict(file="kan_juice.xlsx", orig="meyve suyu tağşiş.xlsx", sheet="Sayfa2", header=0),
}

CIN_FILES = {"walnut": "FTIR_data_Walnut.xlsx", "peanut": "FTIR_data_peanut.xlsx",
             "wood dust": "FTIR_data_wood_dust.xlsx"}

SPECS = {
    "D1": dict(family="kan", key="oil", matrix="olive oil", adulterant="sunflower oil",
               instrument="Bruker Tensor 27 + ATR",
               source="kan-ftir-food-adulteration / Batayhi et al., Foods 15 (2026) 2949"),
    "D2": dict(family="kan", key="coffee", matrix="roasted coffee", adulterant="malt flour",
               instrument="Bruker Tensor 27 + ATR",
               source="kan-ftir-food-adulteration / Batayhi et al., Foods 15 (2026) 2949"),
    "D3": dict(family="kan", key="juice", matrix="orange juice", adulterant="apple juice",
               instrument="Bruker Tensor 27 + ATR",
               source="kan-ftir-food-adulteration / Batayhi et al., Foods 15 (2026) 2949"),
    "D7w": dict(family="cinnamon", key="walnut", matrix="cinnamon", adulterant="walnut",
                instrument="ATR-FTIR, one instrument, three series",
                source="Mendeley rtbsf2r53x.1"),
    "D7p": dict(family="cinnamon", key="peanut", matrix="cinnamon", adulterant="peanut",
                instrument="ATR-FTIR, one instrument, three series",
                source="Mendeley rtbsf2r53x.1"),
    "D7d": dict(family="cinnamon", key="wood dust", matrix="cinnamon", adulterant="wood dust",
               instrument="ATR-FTIR, one instrument, three series",
               source="Mendeley rtbsf2r53x.1"),
    "D8": dict(family="turmeric", key="wheat flour", matrix="turmeric",
               adulterant="wheat flour", instrument="ATR-FTIR",
               source="Mendeley cb5kbt7bgv.2"),
}

ALL = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]


# --------------------------------------------------------------------- helpers
def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def _orient(wvn, X):
    """Return the axis in ascending order, with the columns of X permuted to match."""
    wvn = np.asarray(wvn, float)
    X = np.asarray(X, float)
    if wvn[0] > wvn[-1]:
        return wvn[::-1].copy(), X[:, ::-1].copy()
    return wvn, X


def _dnu(wvn):
    return float(np.median(np.diff(wvn)))


def _to_absorbance(X):
    """Absorbance if the stored panel already is one; A = -log10(T) if it is a ratio.

    The decision uses only the stored values: a single-beam ratio panel has a near-infrared
    baseline just below one and no large excursions (cinnamon: median 0.96-0.99, maximum
    1.02), whereas an absorbance panel has a median close to zero (this rule leaves the KAN
    trio and the turmeric target column unchanged in magnitude). Working in absorbance is
    what makes the increment arithmetic of Section 2.4 meaningful on these data.
    """
    X = np.asarray(X, float)
    trans = bool(np.nanmax(X) < 10 and np.nanmedian(X) > 0.2)
    return (-np.log10(np.clip(X, 1e-6, None)) if trans else X), trans


def _file(subdir, *names):
    if not ROOT:
        raise RuntimeError("set P5_PUBLIC_DATA to the folder holding kan/, cinnamon/, turmeric/")
    for n in names:
        p = os.path.join(ROOT, subdir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("%s: none of %s under %s" % (subdir, ", ".join(names), ROOT))


# --- physical-sample keys for the KAN trio ------------------------------------
# Identifiers encode brand and level, and a few of them repeat a scan of a sample
# that was already measured. A scan must never sit in two splits, so the group key
# has to collapse those repeats onto the same physical sample.
def _oil_group(sid):
    s = _norm(sid)
    m = re.fullmatch(r"z(\d+)", s)
    if m:                                     # pure-oil block Z<k> is the sample <k>1
        return m.group(1) + "1"
    return re.sub(r"([._]\d+)+$", "", s)      # 18_5_2 is a rescan of 18


def _coffee_group(sid):
    return re.sub(r"\.\d+$", "", _norm(sid))


def _juice_group(sid):
    s = _norm(sid)
    m = re.match(r"(p\d+)", s)
    return m.group(1) if m else s


GROUP_KEYS = {"oil": _oil_group, "coffee": _coffee_group, "juice": _juice_group}


def _kan_axis(df, n):
    """Parse the wavenumber axis out of the column headers."""
    vals = []
    for c in df.columns:
        try:
            vals.append(float(str(c).replace(",", ".").strip()))
        except Exception:
            pass
    if len(vals) >= n:
        arr = np.array(vals[-n:], float)
        if 400 < arr.min() and arr.max() < 5000:
            return arr
    return None


# ------------------------------------------------------------------- loaders
def load_kan(key):
    spec = KAN_SHEETS[key]
    path = _file("kan", spec["file"], spec["orig"])
    df = pd.read_excel(path, sheet_name=spec["sheet"], header=spec["header"])
    raw = df.to_numpy()
    ids = np.array([str(v) for v in raw[:, 0]], dtype=object)
    auth = pd.to_numeric(pd.Series(raw[:, 2]), errors="coerce").to_numpy(float)
    adul = pd.to_numeric(pd.Series(raw[:, 3]), errors="coerce").to_numpy(float)
    X = np.asarray(raw[:, 4:], dtype=float)
    keep = np.isfinite(auth) & np.isfinite(adul) & np.isfinite(X).all(axis=1)
    # rows without a recorded level are replicate scans of a sample already labelled
    # above; dropping them keeps the grouping honest.
    ids, adul, X = ids[keep], adul[keep], X[keep]
    if key == "juice":
        # one identifier carries a level outside the design grid in the source file
        hit = np.array([_norm(i) == "p59" for i in ids])
        if hit.any():
            adul[hit] = 45.0
    groups = np.array([GROUP_KEYS[key](i) for i in ids], dtype=object)
    wvn = _kan_axis(df, X.shape[1])
    if wvn is None:
        raise RuntimeError("%s: wavenumber axis not parsed from header" % key)
    raw_desc = bool(wvn[0] > wvn[-1])
    wvn, X = _orient(wvn, X)
    if wvn[0] < 100 or wvn[-1] > 5000:
        raise RuntimeError("%s: implausible wavenumber axis %s..%s" % (key, wvn[0], wvn[-1]))
    return dict(X=X, y=adul, groups=groups, wvn=wvn, dnu=_dnu(wvn), ids=ids,
                meta=dict(axis_source="header", raw_axis_descending=raw_desc))


def load_cinnamon(key):
    df = pd.read_excel(_file("cinnamon", CIN_FILES[key]), sheet_name=0, header=0,
                       engine="openpyxl")
    names = np.asarray(df.iloc[:, 0], dtype=object).astype(str)
    wvn = df.columns[1:-1].to_numpy(float)
    X = np.asarray(df.iloc[:, 1:-1], float)
    raw = df.iloc[:, -1]
    if raw.dtype == object or str(raw.dtype).startswith("str"):
        cat = np.array([float(str(v).strip().rstrip("%")) / 100.0 for v in raw])
    else:
        cat = raw.to_numpy(float)
    keep = np.isfinite(X).all(axis=1)
    X, cat, names = X[keep], np.asarray(cat, float)[keep], names[keep]
    X, trans = _to_absorbance(X)
    y = cat * 100.0
    # the recorded mass fraction must agree with the level written into the sample
    # name ("CW0S01" = 0 %, level then a letter S then the scan counter)
    lvl = np.array([int(n[2:n.index("S")]) for n in names])
    if not np.allclose(lvl, y):
        raise RuntimeError("cinnamon %s: 'category' column disagrees with the level "
                           "embedded in the sample name" % key)
    raw_desc = bool(wvn[0] > wvn[-1])
    wvn, X = _orient(wvn, X)
    return dict(X=X, y=y, groups=names, wvn=wvn, dnu=_dnu(wvn), ids=names,
                meta=dict(axis_source="header", raw_axis_descending=raw_desc,
                          from_transmittance=trans,
                          pct="category fraction x 100, checked against the sample name"))


def load_turmeric():
    df = pd.read_csv(_file("turmeric", "FTIR_Data.csv"))
    names = np.array([str(v).strip() for v in df.iloc[:, 0]], dtype=object)
    tcol = list(df.columns).index("Target")
    wvn = df.columns[1:tcol].to_numpy(float)
    X = np.asarray(df.iloc[:, 1:tcol], float)
    y = np.array([float(str(v).replace("%", "").strip()) for v in df.iloc[:, tcol]])
    keep = np.isfinite(X).all(axis=1)
    X, y, names = X[keep], y[keep], names[keep]
    # the deposition stores transmittance; the analysis works in absorbance
    trans = bool(np.nanmax(X) < 10 and np.nanmedian(X) > 0.2)
    if trans:
        X = -np.log10(np.clip(X, 1e-6, None))
    raw_desc = bool(wvn[0] > wvn[-1])
    wvn, X = _orient(wvn, X)
    return dict(X=X, y=y, groups=names, wvn=wvn, dnu=_dnu(wvn), ids=names,
                meta=dict(axis_source="header", raw_axis_descending=raw_desc,
                          from_transmittance=trans))


def load(name: str):
    """name is one of ALL."""
    spec = SPECS[name]
    if spec["family"] == "kan":
        d = load_kan(spec["key"])
    elif spec["family"] == "cinnamon":
        d = load_cinnamon(spec["key"])
    elif spec["family"] == "turmeric":
        d = load_turmeric()
    else:                                   # pragma: no cover
        raise KeyError(name)
    d["name"] = name
    d["spec"] = spec
    d["y_frac"] = d["y"] / 100.0
    d["meta"].update(dict(name=name, n_spectra=int(d["X"].shape[0]),
                         n_channel=int(d["X"].shape[1]),
                         n_groups=int(len(np.unique(d["groups"]))),
                         matrix=spec["matrix"], adulterant=spec["adulterant"],
                         instrument=spec["instrument"], source=spec["source"]))
    return d


if __name__ == "__main__":
    for nm in ALL:
        d = load(nm)
        print("%-4s n=%4d N=%4d groups=%3d levels=%2d dnu=%.3f range=%.0f-%.0f" % (
            nm, d["X"].shape[0], d["X"].shape[1],
            len(np.unique(d["groups"])), len(np.unique(np.round(d["y"], 6))),
            d["dnu"], d["wvn"][0], d["wvn"][-1]))
