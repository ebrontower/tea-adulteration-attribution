"""Read exp51_xcorpus.json into the tables the results doc needs."""
import json
import p5_paths  # noqa: E402

d = json.load(open(p5_paths.RESULTS + "/exp51_xcorpus.json", encoding="utf-8"))

def fix(a):
    """by_level came back from JSON with string keys; restore floats."""
    for k in ("by_level", "by_level_norm"):
        if isinstance(a.get(k), dict):
            a[k] = {float(kk): v for kk, v in a[k].items()}
    return a

for _r in d["corpora"].values():
    if "arms" in _r:
        for _a in _r["arms"].values():
            if _a: fix(_a)
C = d["corpora"]
ORDER = ["D1", "D2", "D3", "D8", "D7w", "D7p", "D7d", "HONEY", "COCO_traditional",
         "COCO_instant", "D7_pooled"]

print(f"{'corpus':16s} {'n':>5s} {'p':>5s} {'gauge_raw':>9s} {'gauge_w':>8s} {'snr_r':>6s} "
      f"{'snr_w':>6s} {'prep':5s} {'arm':8s} {'cov':>5s} {'size':>5s} {'empt':>5s} {'top1':>5s} "
      f"{'acc':>5s} {'p_top1':>6s} {'p_cov':>5s} {'MAF_A':>5s} {'MAF_B':>5s} {'MAF_C':>5s} "
      f"{'CI_top1':>15s} {'nCal':>10s}")
for nm in ORDER:
    r = C[nm]
    for prep in ("abs", "snv"):
        for arm in ("euclid", "white_a"):
            a = r["arms"][f"{prep}/{arm}"]
            if a is None:
                continue
            g = r["geometry"][prep]
            iv = a["intervals"].get("top1", [None, None])
            print(f"{nm:16s} {a['all']['n']:5d} {r['n_channels']:5d} "
                  f"{str(g['absmax_gauge_raw']):>9s} {str(g['absmax_gauge_white']):>8s} "
                  f"{str(g['incr_snr_raw'][0] if len(g['incr_snr_raw'])==1 else round(sum(g['incr_snr_raw'])/len(g['incr_snr_raw']),2)):>6s} "
                  f"{str(g['incr_snr_white'][0] if len(g['incr_snr_white'])==1 else round(sum(g['incr_snr_white'])/len(g['incr_snr_white']),2)):>6s} "
                  f"{prep:5s} {arm:8s} {a['all']['coverage']:5.3f} {a['all']['size']:5.2f} "
                  f"{a['all']['empty']:5.3f} {a['all']['top1']:5.3f} {a['all']['accept']:5.3f} "
                  f"{str(a['pure']['top1'] if a['pure'] else None):>6s} "
                  f"{str(a['pure']['coverage'] if a['pure'] else None):>5s} "
                  f"{str(a['maf']['naming'][0]):>5s} {str(a['maf']['certified_set'][0]):>5s} "
                  f"{str(a['maf']['deliverable'][0]):>5s} {str(iv):>15s} "
                  f"{str(a['n_cal_per_fold']):>10s}")
    print("")

print("=== per-level top1 (abs), lowest four levels")
for nm in ORDER:
    r = C[nm]
    lv = [l for l in r["levels"]][:6]
    line = f"{nm:16s} "
    for arm in ("euclid", "white_a"):
        for prep in ("abs", "snv"):
            a = r["arms"][f"{prep}/{arm}"]
            line += f"| {prep}/{arm[0]} " + " ".join(
                f"{l:g}:{(a['by_level'].get(l) or {}).get('top1')}" for l in lv) + " "
    print(line)

print("\n=== per-level coverage/accept for white_a abs (lowest levels)")
for nm in ORDER:
    r = C[nm]
    lv = [l for l in r["levels"]][:4]
    a = r["arms"]["abs/white_a"]
    print(f"{nm:16s} " + " ".join(f"{l:g}: cov{(a['by_level'].get(l) or {}).get('coverage')} "
                                  f"sz{(a['by_level'].get(l) or {}).get('size')} "
                                  f"acc{(a['by_level'].get(l) or {}).get('accept')}" for l in lv))

print("\n=== verdicts S2..S5")
for k in ("S2", "S3", "S4", "S5", "reproduce_exp40", "rank_deficient_whiteners"):
    print(k, json.dumps(d["verdicts"][k], ensure_ascii=False)[:1600], "\n")

print("=== intervals (abs/white_a and abs/euclid) coverage/size/abstain")
for nm in ORDER:
    r = C[nm]
    for arm in ("euclid", "white_a"):
        a = r["arms"][f"abs/{arm}"]
        print(f"{nm:16s} {arm:8s}", json.dumps(a["intervals"]))

print("\n=== geometry extra")
for nm in ORDER:
    g = C[nm]["geometry"]["abs"]
    print(f"{nm:16s} atoms={g['n_atoms']} coh_raw={g['coherence_raw']} cond_raw={g['cond_raw']} "
          f"coh_w={g['coherence_white']} cond_w={g['cond_white']} evr={g['evr']} dim={g['dim']} "
          f"rankdef={g['rank_deficient']} sigma={g['sigma_rms']} gauge_raw={g['gauge_raw']} "
          f"gauge_white={g['gauge_white']}")
    gs = C[nm]["geometry"]["snv"]
    print(f"{'':16s} snv: gauge_raw={gs['absmax_gauge_raw']} gauge_white={gs['absmax_gauge_white']} "
          f"snr_raw={gs['incr_snr_raw']} snr_white={gs['incr_snr_white']} rankdef={gs['rank_deficient']}")

print("\n=== tea imported")
print(json.dumps(d["corpora"]["TEA_imported"]["arms"], ensure_ascii=False))
print("fold summary example (D1 abs white):",
      json.dumps(C["D1"]["arms"]["abs/white_a"]["fold_summary_std"], ensure_ascii=False))
