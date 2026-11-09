"""Training-budget control for the convolutional arm of Part B on the public cinnamon corpus.

On tea the whitening prior improves all nine non-invariant model classes; on cinnamon the same
comparison averages -0.089, and the convolutional arm contributes -0.086 of that. A negative delta is
only informative if the arm was not limited by its training budget, so the budget is chosen by the
same pre-registered rule used on tea:

  1. candidate grid = epochs x lr (one rung smaller than tea, to keep the 1950-spectrum fits tractable);
  2. each config is fitted on the fold's FIT block only and scored on that fold's CALIBRATION block;
     the test blocks are never consulted;
  3. the pooled calibration balanced accuracy decides ONE config per representation;
  4. that config is refit on fit + calibration (what every other Part B baseline uses, whitener
     included) and reported on the test blocks.

Reps: raw and white only - those are the two columns the raw -> white delta is made of.
"""
import json
import sys
import time

import numpy as np

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp33_whitened_attr as E
import exp47_baseline_ladder as G

GRID = [(40, 1e-3), (200, 1e-3), (400, 3e-4)]
REPS_CIN = ["raw", "white"]

X, comp, blk, n_dup, p = G.cinnamon_data()
splits = G.blocked_folds_cinnamon(blk)
out = {"corpus": "cinnamon (D7w+D7p+D7d, absorbance)", "n": int(len(X)), "p": int(p),
       "grid": [f"e{e}_lr{lr}" for e, lr in GRID],
       "selection": "fit block only -> calibration block; test never consulted",
       "refit": "chosen config refit on fit + calibration, as every other Part B baseline",
       "ladder_reference": {"raw": 0.789, "white": 0.703}}

for rep in REPS_CIN:
    t_rep = time.time()
    per_cfg = {}
    for e, lr in GRID:
        vs = []
        for fit, cal, test in splits:
            tr = np.concatenate([fit, cal])
            wh = E.Whitener(X, comp, tr)
            Xr = G.rep_matrix(rep, X, wh)
            m = G.TorchCNN(out=4, epochs=e, lr=lr).fit(Xr[fit], comp[fit])
            vs.append(float(G.balanced_accuracy_score(comp[cal], m.predict(Xr[cal]))))
        per_cfg[f"e{e}_lr{lr}"] = round(float(np.mean(vs)), 3)
    best = max(per_cfg, key=lambda k: per_cfg[k])
    e_sel, lr_sel = int(best.split("_")[0][1:]), float(best.split("_lr")[1])
    te = []
    for fit, cal, test in splits:
        tr = np.concatenate([fit, cal])
        wh = E.Whitener(X, comp, tr)
        Xr = G.rep_matrix(rep, X, wh)
        m = G.TorchCNN(out=4, epochs=e_sel, lr=lr_sel).fit(Xr[tr], comp[tr])
        te.append(float(G.balanced_accuracy_score(comp[test], m.predict(Xr[test]))))
    out[rep] = dict(calibration_ba_by_config=per_cfg, chosen=best,
                    test_ba_tuned=round(float(np.mean(te)), 3),
                    n_train_refit=int(len(np.concatenate([splits[0][0], splits[0][1]]))))
    print(rep, json.dumps(out[rep], ensure_ascii=False),
          f"({round(time.time() - t_rep, 1)}s)", flush=True)

DEST = str(Path(__file__).resolve().parent / "out" / "exp47_cnn_check_cinnamon.json")
json.dump(out, open(DEST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", DEST, flush=True)
