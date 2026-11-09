# Pre-registrations

Five criteria documents, each frozen **before** the run it governs. The rule every one of
them states in its opening lines is the same: *"跑完不许按结果加臂、不许改判据，只许按判据改稿件"*
(after the numbers come in, no arm, no criterion, and no threshold may be altered — only the
manuscript wording may change, and only in the direction the criterion says). Nothing in
this folder has been edited after its run.

Two integrity facts worth stating up front:

1. The originals were written in Chinese; only their **filenames** were anglicised for this
   repository. The bytes of every document are what the authors signed off before the run.
   The pointer inside the corresponding result JSON (`prereg:`) was rewritten to the new
   filename; that is the only line changed in the shipped JSONs — see
   `../results/README.md#prereg-pointer-relocations`.
2. `figures_criteria_CEA_v1.md` §7 records one mis-transcription inside §3's list of
   bootstrap intervals. The transcription error was found *after* drawing; the drawing
   script `_fig_tables.bootstrap_intervals()` reads the five intervals by regex from the
   manuscript table's own footnote (and raises if it cannot match them), so the error was
   caught at the tool level and the shipped figure uses the correct intervals. §3 was
   **not** edited to match the observed result — §7 says so in the original Chinese.

## Files

| File | Governs | Criterion set |
|---|---|---|
| `exp47-N_noise_stress_preregistration_v1.md` | `code/_p47_noise.py`, `_p47n_verdicts.py`, `_p47n_stage3.py`, `_p47n_stage3_verdict.py` | N1–N7, three noise families × two magnitudes × two regimes |
| `exp49_minimum_attributable_fraction_v1.md` | `code/_p49_maf.py`, `_p49b_confound.py`, `_p49c_structure.py` | A/B/C definitions of "minimum attributable fraction" plus L1–L4 |
| `exp50_conditional_coverage_and_baseline_correction_v1.md` | `code/_p50_bandcov.py` | C1–C7 on the {marg, band, band_oracle} × {abs, snv, msc} arm grid |
| `exp51_cross_corpus_evaluation_v1.md` | `code/_p51_xcorpus.py` (+ `_p51_folds.py`, `_p51_prep.py`) | S1–S5 across ten single-adulterant corpora + D7_pooled + tea |
| `figures_criteria_CEA_v1.md` | `code/_fig_tables.py`, `_mk_fig1.py`, `_mk_fig2.py`, `_mk_fig6.py`, `_mk_fig7.py`, `_fig_gates.py` | Global §0 layout rules plus per-figure criteria for Figs 1 / 2 / 6 / 7 |

## How each verdict came out

Numbers are read mechanically from `results/*.json` — every experiment's `verdicts` block
records both the fired / not-fired flag and the numeric reason. The consequence column is
the pre-registration's own "触发后必须做什么" clause; the manuscript follows it.

### exp47-N (noise stress test) — three stages, all run

| Criterion | Result | Pre-registered consequence, applied to the manuscript |
|---|---|---|
| N1 — sub family (residual-covariance noise) with whitened-minus-raw T1 gain ≥ clean and not falling with k | not fired under the strict reading; fires only if the comparison is read against T1max in the shift regime | §3.10 does not gain a robustness sentence from this — the family is by construction the one whitening absorbs, so the finding is treated as mechanism, not advantage |
| N2 — iso / drift at k ≤ 2 pushes whitened NLS binary set to ≤ raw + 0.05 | **fired** (iso/matched, iso/shift) | §3.10's positive claim is conditional: whitening buys attribution *within* the measured SNR range; failure point goes to §4 limitations |
| N3 — drift family at k ≤ 2 breaks the gate or coverage | not fired (drift barely moves any cell) | no protocol sentence added; the §3.8 finding "local normalisation absorbs within-block but not between-block" stands unchanged |
| N4 — shift regime coverage falls below 0.85 while matched holds | **fired** | Abstract / contribution's "finite-sample guarantee" gets an exchangeability qualifier; §4 records the failure point as an operating condition, not an unconditional property |
| N5 — gate-vs-kNN(white) AUROC gap widens > 0.02 monotonically with k | not fired (the widening came from the opponent's collapse, not our gain) | Contribution 3 stays written as joint behaviour; not restored from the noise cell |
| N6 — a prior-free detector rises above the gate on the composite column under iso / drift | not fired (no detector both rises above its own clean AUROC *and* exceeds the gate) | §3.7 does not degrade further; the "exceed-only" rows are logged as a limit, not as a defeat |
| N7 — noise opens the T1 ceiling (best bacc < 0.90 at some k) **and** an SSL representation arm is > analytic + 0.02 at that same k | not fired at stage 3 | §3.10's "representation is not the bottleneck" keeps its measured scope: the encoder was *given* the noise (re-pretrained on the noisy pool each fold may see, per-fold leak guard) and did not reopen the axis it is claimed to be robust on. Verdict file `results/_p47n_stage3_verdicts.txt`; `--selfcheck` injects +1.0 on the SSL arm and requires the criterion to flip, proving the assertion is live |

### exp49 (minimum attributable fraction on cinnamon)

| Criterion | Result | Consequence |
|---|---|---|
| L1 — whitened and Euclidean within 0.10 top-1 at 1% and 2% | not fired | The "extend the reporting limit" reading is allowed and has been taken |
| L2 — any dose level's whitened marginal coverage falls below 0.85 | **fired** | §3.11's "marginal coverage unchanged" sentence has to be paired with the per-band spread; MAF-certification is treated as failing at those levels |
| L3 — 0% pure spectra's acceptance rate outside 1 − α ± 0.10 (i.e. outside 0.80–1.00) | **fired** | The gate is renamed to "rejection on adulterated spectra"; MAF-deliverable must be reported next to it, not on the favourable side alone |
| L4 — Euclidean top-1 also rises monotonically with dose (Spearman ρ ≥ 0.7) | **fired** | MAF is reported as a pair of curves + Δ, never as "the method's floor"; the Euclidean floor cannot be claimed as the method's contribution |

Plus the mechanism column (SNR / raw vs whitened top-1(SNR)) recorded at every dose level;
the "concentration → accuracy" reading is not the paper's claim.

### exp50 (conditional coverage by unlabelled dose band × baseline correction)

| Criterion | Result | Consequence |
|---|---|---|
| C1 — reproduction gate: marg/abs/white_a matches `exp51_xcorpus.json` to 1e-9 | not fired (they agree exactly) | Scaffolding sound; the run counts |
| C2 — the `band` arm lifts the three lowest dose levels of every MAF-quotable corpus to coverage ≥ 0.85 while global stays in [0.88, 0.95] | **fired** | The word "conditional-coverage repair" is withdrawn from the manuscript; the surviving obligation is that "per-band coverage vector must be reported next to marginal coverage" |
| C3 — C2 holds but set size inflates > 1.5× or acceptance drops > 0.10 | not fired (the arm-invariant abstention gate means the drop branch cannot fire — see the JSON note; the set-size branch is what the band arm actually buys) | n/a |
| C4 — after banding, coverage spread across bands (width n ≥ 20) still > 0.10 | **fired** (D7d / abs, spread 0.104) | The claim downgrades to "conditioning aligns band structure, cannot guarantee per-band coverage"; the per-band vector is shipped in the same row |
| C5 — MSC lifts the four collapse folds to ≥ 0.80 | not fired — the second branch (still < 0.60) holds | §4 records "collapse is preparation-shift beyond any affine correction"; every "marginal 0.90" number in the manuscript must be reported with the per-fold vector |
| C6 — MSC changes the attribution *direction* (D7_pooled whitened top-1 falls > 0.05, or MAF moves upward on any MAF-quotable corpus) | not fired | MSC stays in the arm grid as a control; it is not promoted to the primary arm |
| C7 — cherry-picking rule (any "repair" sentence paired with cost, per-corpus per-arm; MAF reported with Euclidean and whitened together; `band_oracle` never in the prose claims) | followed throughout — see the shipped tables | any table row violating it was dropped before shipping |

### exp51 (cross-corpus evaluation, no primary dataset)

| Criterion | Result | Consequence |
|---|---|---|
| S1 — on single-adulterant corpora, whitened is worse than Euclidean on all three columns in ≥ half the corpora | not fired | The metric-utility claim survives; it stays scoped to the multi-atom dictionary setting where it is genuinely tested |
| S2 — any corpus's whitened marginal coverage falls below 0.85 (small n_cal particularly at risk) | **fired** | Reported with the calibration-pool size and interval width; the n_cal < 20 threshold means "quantile is not quotable" |
| S3 — raw gauge cosine does *not* correlate with whitening-benefit across corpora (\|Spearman ρ\| < 0.3) | not fired | The "condition number predicts benefit" sentence stays withdrawn (it was already down-graded after exp46's weak evidence) |
| S4 — any corpus marked MAF-quotable has whitened MAF *higher* than Euclidean MAF | **fired** | That corpus's curve is reported as a pair, never just the favourable direction; this is the criterion that stops "works across corpora" from being read as cherry-picked |
| S5 — any corpus's fold-to-fold spread makes the top-1 interval cover the 1/atoms random line | **fired** | That cell is written as "cannot be distinguished from random"; point estimate is not quoted alone |

### figures_criteria_CEA_v1 (drawing rules for the four new CEA figures)

Written before drawing. §0 fixes the layout (16 cm text block, 300 dpi PNG + vector PDF,
minimum ink size, palette and near-black text, no (a)(b)(c) on Fig. 1, no cross-references
or prose inside figures, no numeric results inside captions). §1–§4 fix what each of
Figs 1 / 2 / 6 / 7 may print and require the every-cell gate.

The gate `code/_fig_gates.py` reports 167 cells (72 / 50 / 42 / 3 across Figs 2 / 6 / 7 / 1)
equal to the manuscript tables at three decimals, the drawn-label set matches the parsed
table's set exactly (a missing row, an extra literal, a stale ledger all fail), and every
"—" or "absent" cell is recorded as *not drawn* (drawing it as 0 fails). `--corrupt`
deliberately rewrites one drawn value and asserts the gate catches it — see
`figures_criteria_CEA_v1.md` §7 for the exact failure sample run on the day.

## What we did **not** do

- No criterion, arm, band count, or threshold in any of the five documents was altered after
  the numbers came in. Where the criterion fired against us, the manuscript sentence was
  narrowed or withdrawn; see the consequence columns above.
- No new arm was added because a result looked promising. Where we found something worth
  exploring after the run (e.g. the set-size channel of the band arm in exp50), it went to
  the limitations section, not the method section.
- No pre-registration document was edited to remove an inconvenient criterion. The
  "implementation record" section (§7) of the figure criteria was appended in the same
  file after the run, not by rewriting §0–§6, and it explicitly reports the one
  mis-transcription we found in §3.
