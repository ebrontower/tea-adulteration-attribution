"""Training-budget control for the convolutional arm of Part B on the tea set.

Part B's T1 gave the 1-D CNN 0.369 (raw) / 0.50 (SNV) balanced accuracy while LDA reached 0.996, a
perfectly ordinary outcome for a small spectral corpus. The row is only meaningful if the budget was
not the limiting factor, so the budget is chosen by a rule fixed here, before looking at any test
number:

  1. candidate grid = epochs x lr (and the batch size the fit already uses);
  2. each config is fitted on the fold's FIT block only and scored on that fold's CALIBRATION block
     - the test blocks are never consulted;
  3. the pooled calibration balanced accuracy decides ONE config per representation;
  4. that config is then refit on fit + calibration (the same training set every other baseline in
     Part B uses) and reported on the test blocks.

Reference point: the current Part B budget is epochs=40, lr=1e-3. Whatever the search returns, the
convolutional row is reported at both budgets in Table 12, so the reading of the arm does not rest on
the fixed budget alone.
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
import p5_paths  # noqa: E402

GRID = [(40, 1e-3), (200, 1e-3), (200, 3e-4), (400, 3e-4)]
X, yr = E.load()
rng = np.random.default_rng(G.SEED)
splits = E.make_splits(yr, "blocked", rng)
out = {"grid": [f"e{e}_lr{lr}" for e, lr in GRID],
       "selection": "fit block only -> calibration block; test never consulted",
       "refit": "chosen config refit on fit + calibration, as every other Part B baseline"}

for rep in G.REPS:
    per_cfg = {}
    for e, lr in GRID:
        vs = []
        for fit, cal, test in splits:
            wh = E.Whitener(X, yr, fit)
            Xr = G.rep_matrix(rep, X, wh)
            m = G.TorchCNN(out=None, epochs=e, lr=lr).fit(Xr[fit], yr[fit])
            vs.append(float(G.balanced_accuracy_score(yr[cal], m.predict(Xr[cal]))))
        per_cfg[f"e{e}_lr{lr}"] = round(float(np.mean(vs)), 3)
    best = max(per_cfg, key=lambda k: per_cfg[k])
    e_sel, lr_sel = [int(best.split("_")[0][1:]), float(best.split("_lr")[1])]
    te = []
    for fit, cal, test in splits:
        tr = np.concatenate([fit, cal])
        wh = E.Whitener(X, yr, tr)
        Xr = G.rep_matrix(rep, X, wh)
        m = G.TorchCNN(out=None, epochs=e_sel, lr=lr_sel).fit(Xr[tr], yr[tr])
        te.append(float(G.balanced_accuracy_score(yr[test], m.predict(Xr[test]))))
    out[rep] = dict(calibration_ba_by_config=per_cfg, chosen=best,
                    test_ba_tuned=round(float(np.mean(te)), 3),
                    n_train_refit=int(len(np.concatenate([splits[0][0], splits[0][1]]))))
    print(rep, json.dumps(out[rep], ensure_ascii=False), flush=True)

print(json.dumps(out, indent=1, ensure_ascii=False))
OUT = Path(p5_paths.RESULTS) / "exp47_cnn_check_tea.json"
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("wrote", OUT)
