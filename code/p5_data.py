"""p5_data - data access layer for the tea ATR-FTIR attribution study.

Expected input (bundled under the repository data/ folder, or pointed to by the environment variable
P5_TEA_DATA):

    data/tea_batch_sulfur/ALLfinish.xlsx     batch measured with the sulfur recipe
    data/tea_batch_clean/ALLfinish2.xlsx     batch measured without it

Both workbooks hold one spectrum per row: the first columns are absorbance at the
wavenumbers listed in the header (4501.042 -> 401.121 cm^-1, stored in descending
order, 4253 channels, step ~0.964 cm^-1) and the last column is the class code.
The two batches use *different* class-code conventions, so the folder is the batch
label and the code is translated to a canonical formulation index through
TEA_RECIPE below.

The seven public ATR-FTIR corpora used in Sections 3.10 and 3.11 are not redistributed
here. Their reader is bundled as p5_public_data.py; it expects the downloaded depositions
in the folder given by P5_PUBLIC_DATA (see that module for the layout), and only
exp22_pubgeom.py and exp40_public_pipeline.py need them.

Axis convention: every loader returns the wavenumber axis in ascending order and
never interpolates; cross-dataset comparisons use dimensionless geometry only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p5_public_data as ppub  # noqa: E402
import p5_paths  # noqa: E402

SHARED = os.environ.get("P5_TEA_DATA", p5_paths.DATA)

# ============================================================================
# Tea ATR-FTIR (self-acquired): batch -> class code -> formulation name
# ============================================================================
TEA_FILES = {
    "sulfur": ("tea_batch_sulfur/ALLfinish.xlsx", 1),
    "clean": ("tea_batch_clean/ALLfinish2.xlsx", 0),
}

# The folder name is the *batch*, not a label. The same class code denotes a
# different formulation in the two batches (e.g. sulfur-batch 3 is sulfur while
# clean-batch 3 is paraffin), so codes must be translated before pooling.
TEA_RECIPE = {
    "sulfur": {0: "pure", 1: "sugar", 2: "talc", 3: "sulfur", 4: "paraffin",
               5: "dye", 6: "flavouring", 7: "sugar+paraffin",
               8: "talc+paraffin+flavouring"},
    "clean": {0: "pure", 1: "sugar", 2: "talc", 3: "paraffin", 4: "dye",
              5: "flavouring", 6: "sugar+paraffin", 7: "talc+paraffin+flavouring"},
}
# Canonical formulation index shared by both batches: 0 pure / 1 sugar / 2 talc /
# 3 sulfur / 4 paraffin / 5 dye / 6 flavouring / 7 sugar+paraffin /
# 8 talc+paraffin+flavouring.  Sulfur (3) exists only in the sulfur batch.
TEA_RECIPE_CODE = {
    "pure": 0, "sugar": 1, "talc": 2, "sulfur": 3, "paraffin": 4, "dye": 5,
    "flavouring": 6, "sugar+paraffin": 7, "talc+paraffin+flavouring": 8,
}
TEA_RECIPE_NAMES = {v: k for k, v in TEA_RECIPE_CODE.items()}


def load_tea(verbose: bool = True):
    """Pool the two batches. Returns X (n, 4253) on an ascending axis plus labels."""
    Xs, y_s, y_c, y_r, ids = [], [], [], [], []
    for tag, (rel, ysv) in TEA_FILES.items():
        df = pd.read_excel(os.path.join(SHARED, rel), sheet_name="Sheet1", header=0)
        wvn = df.columns[:-1].to_numpy(float)          # last column is the class code
        X = df.iloc[:, :-1].to_numpy(float)
        cls = df.iloc[:, -1].to_numpy(int)
        Xs.append(X)
        y_s.append(np.full(len(X), ysv, int))
        y_c.append(cls)
        y_r.append(np.array([TEA_RECIPE_CODE[TEA_RECIPE[tag][int(c)]] for c in cls], int))
        ids += [f"{tag}#{i}" for i in range(len(X))]
        if verbose:
            print(f"[tea:{tag}] n={len(X)} N={X.shape[1]} classes={sorted(set(cls.tolist()))}")
    X = np.vstack(Xs)
    y_sulfur = np.concatenate(y_s)
    y_class = np.concatenate(y_c)
    y_recipe = np.concatenate(y_r)
    y_recipe_name = np.array([TEA_RECIPE_NAMES[k] for k in y_recipe], dtype=object)
    # Component membership (multi-label): sugar/talc/sulfur/paraffin/dye/flavouring.
    y_components = np.zeros((len(y_recipe), 6), dtype=int)
    _comp_idx = {"sugar": 0, "talc": 1, "sulfur": 2, "paraffin": 3, "dye": 4,
                 "flavouring": 5}
    for i, nmz in enumerate(y_recipe_name):
        for k, j in _comp_idx.items():
            if k in nmz:
                y_components[i, j] = 1
    wvn = np.asarray(wvn, float)
    if wvn[0] > wvn[-1]:                                # enforce ascending axis
        wvn, X = wvn[::-1].copy(), X[:, ::-1].copy()
    return dict(X=X, y=y_recipe, y_class=y_class, y_recipe=y_recipe,
                y_recipe_name=y_recipe_name, y_components=y_components,
                y_batch=y_sulfur, axis=wvn,
                dnu=float(np.median(np.diff(wvn))), groups=None,
                ids=np.asarray(ids, dtype=object),
                meta=dict(name="tea_ATR_FTIR", modality="ATR-FTIR", matrix="tea",
                          adulterant="sugar/talc/sulfur/paraffin/dye/flavouring (+mixtures)",
                          no_sample_ids=True,
                          class_semantics="formulation index 0-8 (batch-independent); "
                                          "folder = measurement batch, not a label"))


def load_public(name: str):
    """Load one of the seven public ATR-FTIR corpora (Sections 3.10 and 3.11).

    Reader and folder layout live in `p5_public_data`; the data themselves are public and
    are not redistributed with this package.
    """
    return ppub.load(name)


def public_names():
    return list(ppub.ALL)


if __name__ == "__main__":
    d = load_tea()
    print("X", d["X"].shape, "axis", d["axis"][0], "->", d["axis"][-1])
    print("formulations", sorted(set(d["y"].tolist())))
