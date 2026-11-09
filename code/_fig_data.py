"""Real thumbnails for the workflow figure — no numbers are invented here, everything is read.

Three sources, each gated so the drawn curves are the ones the manuscript was computed from:

* one acquired tea spectrum and the six reference increments, from the project loader
  `code/p5_data.py` (the same loader `scripts/r2/exp20_increments.py` uses), checked
  against `out/r2/exp20_increments.json` (pairwise correlations of the increments and the condition
  number of the increment matrix);
* the 6x6 Gram of the dictionary before and after the noise whitening, read verbatim from
  `out/r2/exp35_geometry.json` — the same file the author's own geometry figure was drawn from;
* the window and channel count, read from the loader itself rather than assumed.

If a source file is missing or the gate fails, this module raises and the figure is not drawn.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import p5_paths  # noqa: E402

R2 = p5_paths.RESULTS

LO_WINDOW = (900.0, 1800.0)


def _need(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Fig.1 source missing: {path} — the thumbnail has no fallback")
    return json.load(open(path, encoding="utf8"))


def tea():
    """The measured spectra: axis, window, per-class means, the six increments, one real spectrum."""
    import p5_data as P

    d = P.load_tea(verbose=False)
    X, yr, yb, axis = d["X"], d["y"], d["y_batch"], d["axis"]
    lo, hi = LO_WINDOW
    m = (axis >= lo) & (axis <= hi)
    win = axis[m]
    W = X[:, m]

    # the same subset exp20 used: the 450 rows of the batch that carries all nine formulations
    W, yr = W[yb == 1], yr[yb == 1]
    mu = {int(c): W[yr == c].mean(0) for c in np.unique(yr)}
    codes = list(range(1, 7))
    deltas = np.array([mu[c] - mu[0] for c in codes])

    # gate: the increments drawn here must be the increments the manuscript was computed from
    g = _need(f"{R2}/exp20_increments.json")
    corr = np.corrcoef(deltas)
    if not np.allclose(corr, np.array(g["corr_matrix"]), atol=5e-5):
        raise SystemExit("Fig.1 gate failed: increment correlations differ from exp20_increments.json")
    if abs(np.linalg.cond(deltas.T) - g["cond_increment_matrix"]) > 0.05:
        raise SystemExit("Fig.1 gate failed: increment-matrix condition number differs from exp20")
    if W.shape[0] != g["n"]:
        raise SystemExit(f"Fig.1 gate failed: {W.shape[0]} rows against the {g['n']} recorded in exp20")

    # one acquired spectrum, first row of the authentic base, drawn exactly as measured
    row = int(np.flatnonzero((yr == 0))[0])
    return dict(axis=win, channels=int(win.size), spectrum=W[row], spectrum_id=str(d["ids"][yb == 1][row]),
                deltas=deltas, names=g["labels"], window=LO_WINDOW)


def gram():
    """The 6x6 dictionary Gram before and after the whitening, as recorded by exp35."""
    g = _need(f"{R2}/exp35_geometry.json")
    main, win = g["main"], g["windows"]["900-1800"]
    if main["p"] != win["p"]:
        raise SystemExit("Fig.1 gate failed: exp35 'main' is not the 900-1800 window")
    return dict(names=g["names"], raw=np.array(main["gram_raw"], float),
                white=np.array(main["gram_white"], float), p=int(main["p"]),
                coherence=float(main["coherence"]), coherence_white=float(main["coherence_white"]),
                window=LO_WINDOW)


if __name__ == "__main__":
    t = tea()
    g = gram()
    print("tea   ", t["channels"], "channels", t["window"], "id", t["spectrum_id"],
          "spectrum", round(float(t["spectrum"].min()), 3), "-", round(float(t["spectrum"].max()), 3),
          "absorbance")
    print("delta ", t["deltas"].shape, t["names"])
    print("gram  ", g["p"], "channels, raw", g["raw"].shape, "white", g["white"].shape,
          "| coherence", g["coherence"], "->", g["coherence_white"])
