"""exp35 - export the 6x6 geometry (raw and whitened Gram matrices, gauge cosines) for figures,
plus a window-robustness check of the same quantities.

The theory rests on three numbers that must not be artifacts of the 900-1800 cm^-1 window:
coherence, condition number, and cos(Delta_j, m0). Recompute them on three windows.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp33_whitened_attr as E  # noqa: E402
import p5_data as P  # noqa: E402
import p5_paths  # noqa: E402

OUT = os.path.join(p5_paths.RESULTS, "exp35_geometry.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
WINDOWS = [(900.0, 1800.0), (1000.0, 1800.0), (1200.0, 1750.0), (950.0, 1750.0)]
NAMES = ["sugar", "talc", "sulfur", "paraffin", "dye", "flavour"]


def full():
    d = P.load_tea(verbose=False)
    return d["X"][d["y_batch"] == 1], d["y_recipe"][d["y_batch"] == 1], d["axis"]


def one(Xw, yr):
    m0, D, sigma = E.fit_dict(Xw, yr, np.arange(len(Xw)))
    wh = E.Whitener(Xw, yr, np.arange(len(Xw)))
    iu = np.triu_indices(6, 1)
    nrm = np.linalg.norm(D, axis=0)
    C = (D / nrm).T @ (D / nrm)
    W = wh.coords(D.T)
    wn = np.linalg.norm(W, axis=1)
    Gn = (W @ W.T) / np.outer(wn, wn)
    wd = wh.coords(m0[None, :])
    cos_w = (W @ wd[0]) / (wn * np.linalg.norm(wd[0]))
    cos_r = (D.T @ m0) / (nrm * np.linalg.norm(m0))
    gam = (D.T @ m0) / float(m0 @ m0)
    perp = np.linalg.norm(D - m0[:, None] * gam[None, :], axis=0) / nrm
    # gauge-projected whitened norms (project out m0 in the Sigma metric)
    Dg = D - np.outer(m0, (W @ wd[0]) / float(np.linalg.norm(wd[0]) ** 2))
    return dict(
        p=int(Xw.shape[1]), sigma=round(sigma, 5),
        gram_raw=np.round(C, 4).tolist(), gram_white=np.round(Gn, 4).tolist(),
        coherence=round(float(C[iu].max()), 4), coherence_white=round(float(Gn[iu].max()), 4),
        cond=round(float(np.linalg.cond(D)), 2), cond_white=round(float(np.linalg.cond(Gn)), 2),
        cos_raw=np.round(cos_r, 4).tolist(), cos_white=np.round(cos_w, 4).tolist(),
        gamma_raw=np.round(gam, 4).tolist(), perp_frac=np.round(perp, 4).tolist(),
        norms_raw=np.round(nrm, 3).tolist(), norms_white=np.round(wn, 3).tolist(),
        norms_white_gauge=np.round(np.linalg.norm(wh.coords(Dg.T), axis=1), 3).tolist(),
        m0_norm=round(float(np.linalg.norm(m0)), 3),
    )


def main():
    X, yr, axis = full()
    out = dict(names=NAMES, windows={}, main=None)
    for lo, hi in WINDOWS:
        m = (axis >= lo) & (axis <= hi)
        r = one(X[:, m], yr)
        out["windows"][f"{int(lo)}-{int(hi)}"] = r
        print(f"{int(lo)}-{int(hi)} p={r['p']} coh={r['coherence']}->{r['coherence_white']} "
              f"cond={r['cond']}->{r['cond_white']} maxcos={max(abs(c) for c in r['cos_raw']):.3f}"
              f"->{max(abs(c) for c in r['cos_white']):.3f}", flush=True)
    out["main"] = out["windows"]["900-1800"]
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
