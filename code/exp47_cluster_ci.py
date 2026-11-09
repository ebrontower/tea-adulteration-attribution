"""Cluster confidence intervals for the Part B set-recovery rows.

The fold-heterogeneous numbers (whitened calibrated NLS: per-fold exact-set 0.25 -> 0.95) cannot be
quoted as a single pooled value without an interval, and the spectra inside one test block are not
independent (same scan session, same batch, same recipe). The unit that is independent enough to
resample is therefore the block, which here coincides with the fold: 5 clusters of 20 composites.

Pre-registered reporting rule, fixed before looking at the intervals:
  * percentile bootstrap over the 5 folds, 4000 draws, resample folds with replacement and pool by
    n (equal here), so the estimand is exactly the pooled mean the JSON already reports;
  * any row whose interval contains the hindsight constant-arm floor (t2_oracle_constant) may not be
    quoted as a gain over "always answer the same set";
  * binary and ternary are reported separately, because the pooled exact-set mixes a 0.50 floor with
    a 0.00 floor and hides which half of the claim survives.

Only arms that already carry per-fold diagnostics are touched; nothing is recomputed.
"""
import json
import numpy as np

BOOT = 4000
SEED = 0
from pathlib import Path
import p5_paths  # noqa: E402
HERE = Path(__file__).resolve().parent
SRC = str(Path(p5_paths.RESULTS) / "exp47_baseline_ladder.json")
OUT = str(Path(p5_paths.RESULTS) / "exp47_cluster_ci.json")

d = json.load(open(SRC, encoding="utf-8"))
rng = np.random.default_rng(SEED)
rows = {}
for rep, arms in d["t2_diag"].items():
    floor = d["t2_oracle_constant"][rep]["exact"]
    rows[rep] = {}
    for name, folds in arms.items():
        ex = [f["exact"] for f in folds]
        n = np.array([f["n"] for f in folds], float)
        # AUROC-style pooled mean over clusters == weighted mean of per-fold values
        draws = np.array([np.average(np.array(ex)[idx], weights=n[idx])
                          for idx in (rng.integers(0, len(ex), (BOOT, len(ex))))])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        sub = {}
        for g in ("binary", "ternary"):
            v = [f[g] for f in folds]
            dg = np.array([np.mean(np.array(v)[idx])
                           for idx in (rng.integers(0, len(v), (BOOT, len(v))))])
            sub[g] = [round(float(np.mean(v)), 3),
                      [round(float(np.percentile(dg, 2.5)), 3),
                       round(float(np.percentile(dg, 97.5)), 3)],
                      int(sum(1 for x in v if x > 0))]
        rows[rep][name] = dict(exact=[round(float(np.average(ex, weights=n)), 3),
                                      [round(float(lo), 3), round(float(hi), 3)]],
                               per_fold=[round(float(x), 3) for x in ex],
                               groups=sub,
                               above_constant_floor=bool(lo > floor),
                               constant_floor=floor)
json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

for rep in ("raw", "white"):
    print("==", rep, "constant floor", rows[rep][list(rows[rep])[0]]["constant_floor"])
    for name, v in sorted(rows[rep].items(), key=lambda kv: -kv[1]["exact"][0])[:6]:
        print(f"  {name:12s} exact {v['exact'][0]:.3f} [{v['exact'][1][0]:.3f},"
              f" {v['exact'][1][1]:.3f}]  binary {v['groups']['binary'][0]:.3f}"
              f" [{v['groups']['binary'][1][0]:.2f}, {v['groups']['binary'][1][1]:.2f}]"
              f"  ternary {v['groups']['ternary'][0]:.3f}"
              f" [{v['groups']['ternary'][1][0]:.2f}, {v['groups']['ternary'][1][1]:.2f}]"
              f"  >floor {v['above_constant_floor']}")
print("saved", OUT)
