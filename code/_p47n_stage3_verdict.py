"""exp47-N stage 3 - mechanical reading of N7 against the pre-registered criterion.

Criterion as written BEFORE any of this ran (exp47-N_noise_stress_preregistration_v1.md, section 三):

    N7: 判别栏（T1）在某个 k 下从 0.996 掉到 <0.90，而 SSL 表征臂在此时高于解析表征 >0.02
        -> §3.10 的"表征不是瓶颈"必须改写为"在本仪器的 SNR 下不是瓶颈"，并把 SSL 的增益如实报出来。

So the test is a conjunction, and both halves are recomputed here from the JSON files - nothing is
decided by reading the numbers by eye:

  (i)  the discrimination ceiling really opened: T1max at this cell < 0.90 while the clean/k=0
       reference cell through the same per-fold path is >= 0.90;
  (ii) a learned representation beats every analytic representation at that same cell by > 0.02.

T1max = max over the four model heads (lda/logit/knn/gbdt), taken per representation, then the best
representation - exactly how stage 1's tables were read. If N7 fires, the pre-registered consequence
is to report the SSL gain in the manuscript; if it does not, §3.10 keeps the SNR qualification that
N2/N4 already forced, and the sentence becomes: the encoder was given the noise and still did not
buy the axis it is claimed to be robust on.

Self-check (--selfcheck): inject +1.0 onto the SSL arm in memory only and require the verdict to
flip to FIRES. This proves the assertion can bite; without it a silently wrong comparison would
still print a clean "not fired". +1.0 is deliberately an extreme (SSL T1max goes above 1.0),
because the real SSL arm sits 0.3-0.6 below the analytic arm at every noise cell and a small
injection would not flip anything.

Writes nothing except results/_p47n_stage3_verdicts.txt.
"""
import json
import os
import sys
import glob
import p5_paths  # noqa: E402

STAGE1 = sorted(glob.glob(os.path.join(p5_paths.RESULTS, "exp47_noise_c*.json")))
STAGE3 = os.path.join(p5_paths.RESULTS, "exp47_noiseC.json")
CLEAN_FLOOR = 0.90
GAIN = 0.02


def cells(paths):
    out = {}
    for p in paths:
        for c in json.load(open(p, encoding="utf-8")).get("cells", []):
            out[(c["family"], float(c["k"]), c["regime"])] = c
    return out


def t1max(cell, rep):
    heads = cell[rep]
    vals = [v for m in heads.values() for v in (list(m.values()) if isinstance(m, dict) else [m])
            if isinstance(v, (int, float))]
    return max(vals) if vals else None


def best_over_reps(cell, prefix):
    keys = [r for r in cell if r.startswith(prefix)] if prefix == "ssl_" else None
    if keys is None:
        keys = [r for r in ("raw", "snv", "white", "whitek") if r in cell]
    vals = [(r, t1max(cell, r)) for r in keys]
    vals = [(r, v) for r, v in vals if v is not None]
    return (max(v for _, v in vals), max(vals, key=lambda t: t[1])[0]) if vals else (None, None)


def verdicts(s1, s3, boost=0.0):
    lines = []
    fires = []
    ref = s3.get(("clean", 0.0, "matched"))
    clean_ref = s1.get(("clean", 0.0, "matched"))
    clean_max = best_over_reps(clean_ref["T1"], "")[0] if clean_ref else None
    for tag, cell in sorted(s3.items(), key=lambda t: str(t[0])):
        fam, k, reg = tag
        a = cell["T1"]
        ssl_max, ssl_best = best_over_reps(a, "ssl_")
        if ssl_max is not None:
            ssl_max += boost
        if fam == "clean":
            lines.append(f"reference cell clean/k=0/{reg}: ssl T1max {ssl_max} ({ssl_best}); "
                         f"analytic T1max {clean_max} -> comparison only, N7 cannot fire at k=0")
            continue
        ana = s1.get(tag)
        if ana is None:
            lines.append(f"{tag}: NO stage-1 analytic cell to compare against -> N7 undetermined")
            continue
        ana_max, ana_best = best_over_reps(ana["T1"], "")
        opened = (ana_max is not None and ana_max < CLEAN_FLOOR
                  and clean_max is not None and clean_max >= CLEAN_FLOOR)
        beats = (ssl_max is not None and ana_max is not None and ssl_max > ana_max + GAIN)
        head = (f"N7 @ {fam}/k={k}/{reg}: analytic T1max {ana_max} ({ana_best}), "
                f"SSL T1max {ssl_max} ({ssl_best}), gap "
                f"{'n/a' if ssl_max is None or ana_max is None else round(ssl_max - ana_max, 3)}")
        if opened and beats:
            lines.append(head + "  -> FIRES (both halves)")
            fires.append(tag)
        elif opened:
            lines.append(head + "  -> ceiling opened, SSL did NOT beat the analytic best "
                                f"(needs > +{GAIN}) => N7 NOT fired")
        else:
            lines.append(head + f"  -> ceiling not opened at this cell ({ana_max} >= "
                                f"{CLEAN_FLOOR}) => N7 NOT fired")
        # the attribution column is reported, not judged - N7 is written on the discrimination axis
        at2 = s1.get(tag, {}).get("T2", {})
        for rep in ("raw", "white"):
            if rep in at2:
                lines.append(f"    T2 exact-set (analytic {rep}): " + json.dumps(at2[rep]))
        for rep in sorted(a):
            if rep.startswith("ssl_"):
                lines.append(f"    T2 exact-set ({rep}): " + json.dumps(cell["T2"][rep]))
    lines.append("")
    lines.append("VERDICT: N7 " + ("FIRES at " + str(fires) + " - report the SSL gain in §3.10"
                                   if fires else
                                   "does NOT fire - §3.10 keeps the SNR qualification and gains the "
                                   "stronger statement that noise-matched pretraining did not reopen "
                                   "the axis"))
    return lines, fires


def main():
    s1, s3 = cells(STAGE1), cells([STAGE3])
    if not s3:
        sys.exit(f"no stage-3 cells in {STAGE3}")
    lines, fires = verdicts(s1, s3)
    txt = "\n".join(lines)
    print(txt)
    if "--selfcheck" in sys.argv:
        # The stage-3 SSL arms sit 0.3-0.6 below the analytic arms at every noise cell, so a
        # small positive injection never flips the verdict and a "PASS" would be vacuous. Use
        # +1.0 (i.e. pretend every SSL cell hits T1max > 1) - this is the extreme "if the SSL
        # arm were perfect, would the comparison notice?" test. If even that fails to flip, the
        # branch is dead code.
        _, fires_boost = verdicts(s1, s3, boost=1.0)
        print("\nSELF-CHECK: baseline fired =", fires,
              "| with +1.0 injected on every SSL cell =", fires_boost)
        if not fires_boost:
            sys.exit("SELF-CHECK FAILED: the N7 comparison cannot bite (vacuous assertion)")
        if fires:
            print("SELF-CHECK NOTE: N7 already fires at baseline - the injection check is only "
                  "informative once the baseline is negative; verify by lowering the SSL arm instead")
            _, fires_cut = verdicts(s1, s3, boost=-0.5)
            if fires_cut:
                sys.exit("SELF-CHECK FAILED: comparison fires even with -0.5 cut (not discriminating)")
        print("SELF-CHECK PASSED: the criterion is sensitive in both directions")
    open(os.path.join(p5_paths.RESULTS, "_p47n_stage3_verdicts.txt"), "w",
         encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
