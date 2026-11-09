"""exp51 fold-level view: the whitener each fold could actually fit, and the per-fold coverage /
abstention that the pooled marginal hides."""
import json
import p5_paths  # noqa: E402

d = json.load(open(p5_paths.RESULTS + "/exp51_xcorpus.json", encoding="utf-8"))
C = d["corpora"]
ORDER = ["D1", "D2", "D3", "D8", "D7w", "D7p", "D7d", "HONEY", "COCO_traditional",
         "COCO_instant", "D7_pooled"]

print(f"{'corpus':16s} {'prep':4s} {'arm':8s} {'cov/fold':34s} {'empty/fold':34s} "
      f"{'top1/fold':30s} {'wh_dim/fold':16s} {'tail floor (min)':>16s}")
for nm in ORDER:
    r = C[nm]
    for prep in ("abs", "snv"):
        g = r["geometry"][prep]["per_fold"]
        for arm in ("euclid", "white_a"):
            a = r["arms"][f"{prep}/{arm}"]
            cov = [f["coverage"] for f in a["fold_summary_std"]]
            emp = [f["empty"] for f in a["fold_summary_std"]]
            t1 = [f["top1"] for f in a["fold_summary_std"]]
            print(f"{nm:16s} {prep:4s} {arm:8s} {str(cov):34s} {str(emp):34s} {str(t1):30s} "
                  f"{str(g['dim']):16s} {min(float(v) for v in g['delta']):.3e}")
    print("")

print("=== authentic-side (level 0) detail, abs arm")
print(f"{'corpus':16s} {'arm':8s} {'n_pure':>6s} {'top1':>5s} {'cov':>5s} {'size':>5s} "
      f"{'accept':>6s}   (top1 = argmin says authentic; cov = 'authentic' inside the set)")
for nm in ORDER:
    for arm in ("euclid", "white_a"):
        a = C[nm]["arms"][f"abs/{arm}"]
        p = a["pure"]
        if p is None:
            print(f"{nm:16s} {arm:8s}  no authentic rows survived the split")
            continue
        print(f"{nm:16s} {arm:8s} {p['n']:6d} {p['top1']:5.3f} {p['coverage']:5.3f} "
              f"{p['size']:5.2f} {p['accept']:6.3f}")

print("\n=== lowest-level per-adulterant / per-level whitened vs euclidean naming (abs)")
for nm in ORDER:
    r = C[nm]
    lv = r["levels"][:4]
    e, w = r["arms"]["abs/euclid"], r["arms"]["abs/white_a"]
    print(f"{nm:16s} " + "  ".join(
        f"{l:g}%: E{(e['by_level'].get(l) or {}).get('top1')}/W{(w['by_level'].get(l) or {}).get('top1')}"
        for l in lv))
