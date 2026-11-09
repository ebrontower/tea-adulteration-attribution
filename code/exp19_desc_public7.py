"""Descriptor-vs-Delta associations restricted to the seven public ATR-FTIR
datasets used in the paper (the G1 table also carried three auxiliary sets,
S1-S3, which are outside the scope of this manuscript)."""
import json
import numpy as np
from scipy.stats import spearmanr
import p5_paths  # noqa: E402

SRC = p5_paths.RESULTS + "/exp15_g1.json"
OUT = p5_paths.RESULTS + "/exp19_desc_public7.json"
PUB = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
KEYS = ["dim99", "erank", "pc1", "fisher", "c_incr", "cond", "mae_ratio"]

ds = json.load(open(SRC, encoding="utf-8"))["datasets"]
res = {}
for k in KEYS:
    x = [ds[t][k] for t in PUB]
    y = [ds[t]["delta_task_mean"] for t in PUB]
    r, p = spearmanr(x, y)
    res[k] = {"rho": float(r), "p": float(p)}
res["_n"] = len(PUB)
res["_delta"] = {t: ds[t]["delta_task_mean"] for t in PUB}
json.dump(res, open(OUT, "w"), indent=1)
for k in KEYS:
    print(f"{k:10s} rho={res[k]['rho']:+.3f} p={res[k]['p']:.3f}")
