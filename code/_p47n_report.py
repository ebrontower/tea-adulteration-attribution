"""Merge the exp47-N chunk files and print the N1-N7 readouts against the k=0 reference cell.

Usage: python _p47n_report.py [glob]
"""
import glob
import json
import os
import sys

import numpy as np
import p5_paths  # noqa: E402

HERE = p5_paths.RESULTS
REPS = ["raw", "snv", "white", "whitek"]
T1_KEY = ["lda", "logit", "knn", "gbdt"]
T2_KEY = ["gbdt", "nls_cal", "dictargmin"]


def load(pattern):
    cells = []
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path, encoding="utf-8"))
        cells += d.get("cells", [])
    return cells


def find(cells, fam, k, reg):
    for c in cells:
        if c.get("family") == fam and float(c.get("k")) == float(k) and c.get("regime") == reg:
            return c
    return None


def g(cell, section, rep, model):
    if not cell:
        return None
    v = cell.get(section, {}).get(rep, {}).get(model)
    return None if v is None else float(v)


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "exp47_noise_c*.json")
    cells = load(pat)
    if not cells:
        print("no cells yet:", pat)
        return
    done = {(c["family"], c["k"], c["regime"]) for c in cells}
    print(f"cells done: {len(done)} / expected 13")
    for reg in ("matched", "shift"):
        ref = find(cells, "clean", 0, reg) or find(cells, "clean", 0, "matched")
        print(f"\n===== regime = {reg} =====")
        hdr = (f"{'fam/k':<11}{'reg':<9}" + "".join(f"{r + ' T1max':>11}" for r in REPS)
               + f"{'white-raw':>11}" + "".join(f"{r + ' nlsC':>10}" for r in REPS)
               + f"{'white dictarg':>13}")
        print(hdr)
        rows = [("clean", 0)] + [(f, k) for f in ("iso", "sub", "drift") for k in (2, 10)]
        for fam, k in rows:
            c = find(cells, fam, k, reg)
            if c is None:
                continue
            t1 = {r: max([g(c, "T1", r, m) for m in T1_KEY if g(c, "T1", r, m) is not None],
                         default=None) for r in REPS}
            nls = {r: g(c, "T2", r, "nls_cal") for r in REPS}
            darg = g(c, "T2", "white", "dictargmin")
            # white-raw delta on the invariant linear reference (metric-only comparison: LDA(svd)
            # is scale-invariant, so its raw-vs-white gap isolates the metric, not the learner)
            wr = None
            if ref is not None:
                b0 = g(ref, "T1", "white", "lda")
                r0 = g(ref, "T1", "raw", "lda")
                base = None if (b0 is None or r0 is None) else b0 - r0
                b1, r1 = g(c, "T1", "white", "lda"), g(c, "T1", "raw", "lda")
                wr = None if (base is None or b1 is None or r1 is None) else (b1 - r1) - base
            cells_txt = "".join(f"{('-' if t1[r] is None else format(t1[r], '.3f')):>11}"
                               for r in REPS)
            cells_txt += f"{('-' if wr is None else format(wr, '+.3f')):>11}"
            cells_txt += "".join(f"{('-' if nls[r] is None else format(nls[r], '.3f')):>10}"
                                for r in REPS)
            cells_txt += f"{('-' if darg is None else format(darg, '.3f')):>13}"
            print(f"{fam + '/k=' + str(k):<11}{reg:<9}{cells_txt}")
        if ref is not None:
            print("  reference (k=0): " + " ".join(
                f"{r}/lda={g(ref, 'T1', r, 'lda')}" for r in REPS) +
                  "  nls_cal " + " ".join(f"{r}={g(ref, 'T2', r, 'nls_cal')}" for r in REPS))
    print("\nnotes:")
    print(" T1max = best balanced accuracy over lda/logit/knn/gbdt within that representation;")
    print(" white-raw = change, vs the k=0 cell, in the LDA(svd) white-minus-raw gap (LDA(svd) is")
    print("   scale-invariant, so that gap is the metric's contribution, not the learner's);")
    print(" nlsC = quotable NLS exact-set on held-out composites; the post-hoc best constant set")
    print("   scores 0.50 pooled, so pooled rows must be read against 0.50, not against 0;")
    print(" shift = noise on test rows only, which breaks split-conformal exchangeability by design.")


if __name__ == "__main__":
    main()
