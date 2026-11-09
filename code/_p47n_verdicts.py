"""exp47-N: mechanical N1-N7 verdicts from the stage-1 (T1/T2) and stage-2 (Part A) chunk files.

Reads logs/exp47_noise_c*.json (T1 discrimination + T2 held-out-composite naming) and
logs/exp47_noiseA_c*.json (abstention gate + prior-free detectors), and evaluates the criteria
written in exp47-N_noise_stress_preregistration_v1.md exactly as they are phrased - including the readings
that hurt us. Nothing here re-tunes a threshold after seeing the numbers.
"""
import glob
import json
import os
import p5_paths  # noqa: E402

HERE = p5_paths.RESULTS
REPS = ["raw", "snv", "white", "whitek"]
FAMS = ["iso", "sub", "drift"]
KS = [2.0, 10.0]
COV_FLOOR = 0.85          # 1 - alpha - 0.05 as written in N4
N5_GAP = 0.02


def load(pat):
    cells = {}
    for path in sorted(glob.glob(pat)):
        for c in json.load(open(path, encoding="utf-8")).get("cells", []):
            cells[(c["family"], float(c["k"]), c["regime"])] = c
    return cells


def t1(c, rep, m="lda"):
    v = (c.get("T1", {}).get(rep, {}) or {}).get(m)
    return None if v is None else float(v)


def t2(c, rep, m="nls_cal", key="T2"):
    v = (c.get(key, {}).get(rep, {}) or {}).get(m)
    return None if v is None else float(v)


def main():
    S1 = load(os.path.join(HERE, "exp47_noise_c*.json"))
    S2 = load(os.path.join(HERE, "exp47_noiseA_c*.json"))
    ref1, ref2 = S1[("clean", 0.0, "matched")], S2[("clean", 0.0, "matched")]
    L = []
    w = L.append

    w("stage-1 cells: %d/13   stage-2 cells: %d/13" % (len(S1), len(S2)))

    # ---- N1: whitening's T1 gain over raw on the `sub` family, >= clean and not falling with k
    def t1max(c, rep):
        vals = [t1(c, rep, m) for m in ("lda", "logit", "knn", "gbdt")]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    w("\n## N1  sub family, whitened arm minus raw arm on T1, two readings")
    b_l = t1(ref1, "white", "lda") - t1(ref1, "raw", "lda")
    b_m = t1max(ref1, "white") - t1max(ref1, "raw")
    w(f"  clean reference: LDA gap {b_l:+.3f}   T1max gap {b_m:+.3f}")
    for reg in ("matched", "shift"):
        seq = []
        for k in KS:
            c = S1.get(("sub", k, reg))
            if not c:
                continue
            seq.append((k, t1(c, "white", "lda") - t1(c, "raw", "lda"),
                        t1max(c, "white") - t1max(c, "raw"),
                        (t1(c, "white", "lda"), t1(c, "raw", "lda"),
                         t1max(c, "white"), t1max(c, "raw"))))
        for j, (k, gl, gm, raw) in enumerate(seq):
            w(f"  [{reg}] k={int(k)}: LDA gap {gl:+.3f}  T1max gap {gm:+.3f}   "
              f"(white lda {raw[0]} raw lda {raw[1]} | white max {raw[2]} raw max {raw[3]})")
        for name, base, vals in (("LDA", b_l, [(k, g) for k, g, _, _ in seq]),
                                 ("T1max", b_m, [(k, g) for k, _, g, _ in seq])):
            ge = all(g >= base for _, g in vals)
            nfall = all(vals[i][1] >= vals[i - 1][1] for i in range(1, len(vals)))
            w(f"  [{reg}] N1 on {name}: gain>=clean {ge}, not falling with k {nfall} => "
              + ("TRIGGERED" if ge and nfall else "not triggered"))

    # ---- N2: iso or drift at k<=2 pushes the whitened NLS exact-set down to <= raw + 0.05
    w("\n## N2  whitened quotable NLS (exact set) vs raw, at k=2")
    cw = t2(ref1, "white")
    w(f"  clean white nls_cal = {cw} (pre-registration quoted the 0.88 binary-tier figure from the "
      f"cross-fold-pooled headline; the same-fold per-fold-path reference is {cw})")
    hit = []
    for fam in ("iso", "drift"):
        for reg in ("matched", "shift"):
            c = S1.get((fam, 2.0, reg))
            if not c:
                continue
            wh, rw = t2(c, "white"), t2(c, "raw")
            w(f"  {fam}/k=2/{reg}: white {wh} vs raw {rw} (white-raw {None if wh is None or rw is None else round(wh - rw, 3):+.3f})")
            if wh is not None and wh <= (rw or 0) + 0.05:
                hit.append(f"{fam}/{reg}")
    w("  N2 " + (f"TRIGGERED: {', '.join(hit)}" if hit else "not triggered"))

    # ---- N3: drift at k<=2 degrades the gate (coverage < 0.85 or AUROC -0.05 vs reference)
    w("\n## N3  drift family at k=2, abstention gate")
    for reg in ("matched", "shift"):
        c = S2.get(("drift", 2.0, reg))
        if not c:
            continue
        for g in ("ours_gate", "ours_gate_abs"):
            au, cov = c["A_adult_auroc"].get(g), c["A_composite"].get(g, {}).get("coverage_accepted")
            ref_au = ref2["A_adult_auroc"].get(g)
            w(f"  [{reg}] {g}: adult AUROC {au} vs {ref_au} ({None if au is None or ref_au is None else round(au - ref_au, 3):+.3f}), "
              f"composite coverage {cov}")
            if (cov is not None and cov < COV_FLOOR) or (au is not None and ref_au is not None and au <= ref_au - 0.05):
                w(f"  [{reg}] N3 TRIGGERED for {g}")

    # ---- N4: shift coverage below 0.85 while matched holds
    w("\n## N4  coverage floor 0.85 (ours_gate, composite column)")
    bad = {"matched": [], "shift": []}
    for reg in bad:
        for fam in FAMS:
            for k in KS:
                c = S2.get((fam, k, reg))
                if not c:
                    continue
                cov = c["A_composite"].get("ours_gate", {}).get("coverage_accepted")
                size = c["A_composite"].get("ours_gate", {}).get("mean_size_accepted")
                if cov is not None and cov < COV_FLOOR:
                    bad[reg].append(f"{fam}/k={int(k)} cov={cov} size={size}")
    w(f"  matched breaches: {bad['matched'] or 'none'}")
    w(f"  shift breaches:   {bad['shift'] or 'none'}")
    w("  N4 " + ("TRIGGERED" if bad["shift"] and not bad["matched"] else "not triggered as written"))
    # degenerate-set reading: coverage 1.0 with the set equal to the whole power set says nothing
    w("  degenerate-set check (coverage = 1.0 with mean_size = full dictionary):")
    for reg in ("matched", "shift"):
        for fam in FAMS:
            for k in KS:
                c = S2.get((fam, k, reg))
                if not c:
                    continue
                d = c["A_composite"].get("ours_gate", {})
                if d.get("coverage_accepted") == 1.0 and (d.get("mean_size_accepted") or 0) > 8:
                    w(f"    {fam}/k={int(k)}/{reg}: cov 1.000 but mean set size {d['mean_size_accepted']} "
                      f"(out of 64 candidates), top1_accepted {d.get('top1_accepted')}")

    # ---- N5: gate minus kNN(white) adult AUROC gap, >0.02, favourable, monotone in k
    w("\n## N5  ours_gate - knn_all_white adult AUROC (clean reference gap = "
      f"{round(ref2['A_adult_auroc']['ours_gate'] - ref2['A_adult_auroc']['knn_all_white'], 3):+.3f})")
    for reg in ("matched", "shift"):
        for fam in FAMS:
            seq = []
            for k in KS:
                c = S2.get((fam, k, reg))
                if not c:
                    continue
                seq.append((k, c["A_adult_auroc"]["ours_gate"] - c["A_adult_auroc"]["knn_all_white"]))
            w(f"  [{reg}/{fam}] " + "  ".join(f"k={int(k)}: {g:+.3f}" for k, g in seq)
              + ("   | monotone rising, endpoint >0.02"
                 if len(seq) == 2 and seq[1][1] > seq[0][1] and seq[1][1] > N5_GAP else ""))

    # ---- N6: does any prior-free detector rise with k AND beat the gate on the composite column
    w("\n## N6  prior-free detectors vs the gate on the composite column (iso / drift)")
    FREE = [a for a in ref2["A_composite"]
            if a not in ("ours_gate", "ours_gate_pub", "ours_gate_abs")
            and not a.endswith("_pub")]
    strict, exceed = [], []
    for reg in ("matched", "shift"):
        for fam in ("iso", "drift"):
            for a in FREE:
                ref_a = (ref2["A_composite"].get(a) or {}).get("auroc")
                if ref_a is None:
                    continue
                for lev in KS:
                    c = S2.get((fam, lev, reg))
                    if not c:
                        continue
                    d = c["A_composite"].get(a) or {}
                    g = c["A_composite"].get("ours_gate") or {}
                    au, gt = d.get("auroc"), g.get("auroc")
                    if au is None or gt is None:
                        continue
                    line = (f"  [{reg}/{fam}/k={int(lev)}] {a}: AUROC {au:.3f} "
                            f"(clean {ref_a:.3f}) vs gate {gt:.3f}; "
                            f"orphan flag {d.get('orphan_flag_at_tau')} vs gate "
                            f"{g.get('orphan_flag_at_tau')}, inset flag "
                            f"{d.get('inset_flag_at_tau')} vs gate {g.get('inset_flag_at_tau')}")
                    if au > ref_a + 1e-9 and au > gt:
                        strict.append(line + "   <- N6 STRICT (rises with k and exceeds the gate)")
                    elif au > gt:
                        exceed.append(line + "   <- exceeds the gate but does NOT rise with k")
    for line in strict:
        w(line)
    if not strict:
        w("  no detector both rises above its own clean AUROC and exceeds the gate "
          "=> N6's conjunction is NOT met on composite AUROC")
    w("  exceed-only rows (not N6 as written, but they weaken §3.7):")
    for line in exceed[:14]:
        w(line)
    if not exceed:
        w("    none")
    w("  N6 " + ("TRIGGERED" if strict else "not triggered on AUROC; see exceed-only rows"))

    # ---- N7 premise
    w("\n## N7  premise: does noise break the T1 ceiling (best balanced accuracy < 0.90)?")
    for reg in ("matched", "shift"):
        for fam in FAMS:
            for k in KS:
                c = S1.get((fam, k, reg))
                if not c:
                    continue
                mx = max(v for rep in REPS for v in [max(
                    x for x in (c["T1"][rep].get(m) for m in ("lda", "logit", "knn", "gbdt"))
                    if x is not None)])
                if mx < 0.90:
                    w(f"  premise met: {fam}/k={int(k)}/{reg} best T1 = {mx}")
    w("  N7 itself needs the SSL representation arm (stage 3, GPU, author's call) - NOT RUN.")

    print("\n".join(L))
    with open(os.path.join(HERE, "_p47n_verdicts.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
