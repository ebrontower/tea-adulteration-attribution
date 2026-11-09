# Results

Every file in this folder is an export produced by the scripts in `../code/`. Numbers in the
manuscript's tables were read from these JSONs (or from the two `.txt` verdict files); they
are shipped so a reviewer can trace each cell back to its producer without re-running the
pipeline.

## Layout

```
results/
  expNN_*.json              one file per experiment (exp13 - exp51)
  audit_tea.json            row-by-row de-duplication of the two tea workbooks
  dup_check.json            whole-repo duplicate-export audit
  dup_struct.json           structural duplicate audit
  _p47n_verdicts.txt        N1-N7 verdicts from noise stress stages 1 + 2
  _p47n_stage3_verdicts.txt N7 verdict from stage 3 (SSL re-pretrained on the noisy pool)
  _p47n_tables.txt          stage-1 / stage-2 tables in review-facing form
  figure_drawn_values.json  every numeric cell Fig. 1 / 2 / 6 / 7 actually drew,
                            produced by code/_fig_data.py and gated by code/_fig_gates.py
  earlier_round/            11 inputs consumed by code/exp15_g1.py (the SSL-era G1 probe);
                            kept for the historical record; the paper's current claims do
                            not rest on them
```

## Superseded files, deliberately excluded

Three JSONs from earlier rounds are marked `*_superseded.json` in the working tree and are
**not** shipped here:

- `exp40_diag_pure_ratio_space_superseded.json`
- `exp40_public_ratio_space_superseded.json`
- `exp47_baseline_ladder_prev0023_superseded.json`

They computed cells in a ratio-space / older-ladder convention that the manuscript does
not use. None of the shipped result files depends on them, and none of the manuscript's
tables reads from them.

## `prereg:` pointer relocations

Ten result JSONs carry a `prereg:` key naming the criteria document they were run against.
The pointer originally sat on the workspace-local folder that held these documents before
this repository existed; it has been rewritten to point inside the checkout at
`preregistrations/<english-name>.md`. Two things happened on those files and nothing else:

- the string on the `prereg` line changed value;
- line endings were normalised from CRLF (Windows-authored) to LF (cross-platform target).

**Every other byte in the parsed JSON is what the run produced** — cells, verdicts, CI
fields, arm labels, dataset identifiers. The gate `_gate_repo.py` and the reproduction
smoke test both treat this as an accepted relocation; no numeric content was rewritten.
The relocations are:

| File | New pointer |
|---|---|
| `exp47_noise_c0.json` | `preregistrations/exp47-N_noise_stress_preregistration_v1.md` |
| `exp47_noise_c1.json` | same relocation |
| `exp47_noise_c2.json` | same relocation |
| `exp47_noise_c3.json` | same relocation |
| `exp47_noiseA_c0.json` | same relocation |
| `exp47_noiseA_c1.json` | same relocation |
| `exp47_noiseA_c2.json` | same relocation |
| `exp47_noiseA_c3.json` | same relocation |
| `exp47_noiseC.json` | same relocation |
| `exp49_maf.json` | `preregistrations/exp49_minimum_attributable_fraction_v1.md` |
| `exp50_bandcov.json` | `preregistrations/exp50_conditional_coverage_and_baseline_correction_v1.md` |
| `exp51_xcorpus.json` | `preregistrations/exp51_cross_corpus_evaluation_v1.md` |

Everything else in those JSONs — `cells[]`, `verdicts`, CI fields, arm labels, dataset
identifiers — is equal under a parsed-JSON comparison with the pre-relocation file.
Line-endings differ; no numeric content does.

## File-by-file role

Read from top-level sections; "Section" refers to the manuscript's numbering (v3).

| File | Producer | Section / table | What it carries |
|---|---|---|---|
| `audit_tea.json` | `code/audit_tea.py` | §2.1 / §3.1 | Row-by-row comparison of the two tea workbooks; 7 of 8 shared formulations match to 0.000, talc ≤ 0.482 |
| `dup_check.json` | `code/dup_check.py` | §3.1 | Duplicate-export audit across the whole project |
| `dup_struct.json` | `code/dup_struct.py` | §3.1 | Structural duplicate audit (same shapes, different keys) |
| `exp13_tea_clean.json` | `code/exp13_tea_clean.py` | §2.1 / §3.1 / §4.5 | 850 → 450 unique spectra, autocorrelation time 1.7–19 scans, block-split rationale |
| `exp14_sets.json` | `code/exp14_sets.py` | §1 / §3.3 | Multi-label PLS-DA exact-set 0.864 in-distribution → 0.000 held-out |
| `exp15_g1.json` | `code/exp15_g1.py` | (historical) | G1 pre-check for SSL-gain predictability from dataset descriptors — negative |
| `exp16_per_adulterant.json` | `code/exp16_per_adulterant.py` | §3.2 | Seven of eight pure-vs-single-adulterant AUC = 1.00; dye 0.944 |
| `exp17_headline_ci.json` | `code/exp17_headline_ci.py` | §2.9 / every headline CI | 25-fold (5 offsets × 5 folds) mean + 2000× cluster-bootstrap 95% CI |
| `exp18_unmix.json` | `code/exp18_unmix.py` | §2.8 / §3.3 | NNLS with recipe means as endmembers; composite recovery still fails |
| `exp19_desc_public7.json` | `code/exp19_desc_public7.py` | (historical) | Descriptor–Δ association on the seven corpora actually used |
| `exp20_increments.json` | `code/exp20_increments.py` | §1 / §3.4 / Table 2 | Dictionary coherence 0.987, condition 44.7, gauge −0.993 (raw); 0.581 / 8.20 / −0.403 (white) |
| `exp21_attributability.json` | `code/exp21_attributability.py` | §3.4–3.5 discussion | Formalises "not attributable" as a dictionary-condition-number problem |
| `exp22_pubgeom.json` | `code/exp22_pubgeom.py` | §3.11 / §4.6 (cinnamon row) | Near-collinearity on cinnamon (coherence 0.994 / condition 550 / peanut gauge 0.988) |
| `exp23_bandsel.json` | `code/exp23_bandsel.py` | §3.4 end | Window robustness — raw coherence 0.976–0.987 across four fingerprint windows |
| `exp24_jointblocks.json` | `code/exp24_jointblocks.py` | (historical, negative) | Multi-band joint modelling cannot recover set recall |
| `exp25_interp_floor.json` | `code/exp25_interp_floor.py` | §3.10 | MAE embeddings 75–97% spanned by first 5 PCs |
| `exp26_nonrecon_ssl.json` | `code/exp26_nonrecon_ssl.py` | §1 / §3.10 | VICReg / Barlow beat random-init on 3 of 8 corpora only |
| `exp27_synthetic_phase.json` | `code/exp27_synthetic_phase.py` | (historical) | First-pass SSL-gain phase diagram — replaced by exp29 |
| `exp28_label_subspace.json` | `code/exp28_label_subspace.py` | §1 / §3.10 | 99.5–100% of between-class variance inside first 20 PCs on all eight corpora |
| `exp29_phase_band.json` | `code/exp29_phase_band.py` | §3.10 mechanism | Gain vs out-of-smooth-subspace label energy |
| `exp30_aug_control.json` | `code/exp30_aug_control.py` | §3.10 | VICReg / Barlow gains reproduce under augmentation alone |
| `exp31_synth_control.json` | `code/exp31_synth_control.py` | (historical, negative) | MAE loses to a 4-PC probe on the synthetic set for information reasons |
| `exp32_conformal_attr.json` | `code/exp32_conformal_attr.py` | §3.5–3.6 baseline | Euclidean conformal sets valid but ~30 of 64 candidates |
| `exp33_whitened_attr.json` | `code/exp33_whitened_attr.py` | §3.5 / Table 3 / Fig. 2 | Single-adulterant blocked top-1 0.503 → 0.974; set 23.2 → 16.0 of 64; coverage 0.946 |
| `exp34_selective_mondrian.json` | `code/exp34_selective_mondrian.py` | §3.6–3.8 / Table 5 | Per-group coverage, α-map (0.976 / 0.907 / 0.864 / 0.809) |
| `exp35_geometry.json` | `code/exp35_geometry_export.py` | Fig. 1 / Table 2 source | 6×6 Gram raw / whitened; three-window cross-check |
| `exp36_open_set.json` | `code/exp36_open_set.py` | §3.9 / §4.6 | Two-sided gate abstains on five of six out-of-dictionary adulterants |
| `exp37_samp_holdout.json` | `code/exp37_samp_holdout.py` | §4.6 | Integral autocorrelation time → 450 spectra ≈ 90 independent observations |
| `exp38_mondrian.json` | `code/exp38_mondrian.py` | §3.6 | Oracle conditional calibration lifts ternary coverage 0.280 → 0.940 at a set-size cost |
| `exp39_robust.json` | `code/exp39_robust.py` | §3.4 / §3.3 | Hyperparameter sweep: k_w 40 → 320 leaves top-1 in 0.960–0.966, coverage 0.902–0.916 |
| `exp40_diag_pure.json` | `code/exp40_diag.py` | §2.7 / §3.11 / Tables 8/9 | Cinnamon single-adulterant pipeline diagnostics |
| `exp40_public.json` | `code/exp40_public_pipeline.py` | §2.7 / §3.11 / Tables 8/9 | Cinnamon transfer: top-1 0.513 → 0.964, marginal coverage unchanged |
| `exp41_transfer.json` | `code/exp41_transfer.py` | (historical) | Cross-session transfer already at the linear ceiling; 6 nuisance arms flat |
| `exp42_magnitude_transfer.json` | `code/exp42_magnitude_transfer.py` | (historical) | Low-dose detection is not the bottleneck; low-dose extrapolation is (+103% ~ +541%) |
| `exp43_deposit_audit.json` | `code/exp43_deposit_audit.py` | §2.7 / §4.6 | Four of seven level-recorded corpora have a step-function "dose axis" |
| `exp44_level_block_alias.json` | `code/exp44_level_block_alias.py` | §2.7 / §4.6 | Level-by-row segmentation in D7w / D7d / D7p / D8 |
| `exp45_external_validation.json` | `code/exp45_external_validation.py` | §4.6 | Leakage-free external validation — Spearman 0.81 [0.21, 1.00], R² −0.24 |
| `exp46_headroom_onsample.json` | `code/exp46_headroom_external.py` | §3.10 | Out-of-sample headroom: 17 of 22 arms ≤ 0.02, min coverage 0.9953 |
| `exp47_baseline_ladder.json` | `code/exp47_baseline_ladder.py` | §2.8 / §3.12 / Tables 11/12 / Fig. 7 | Nine-rung ladder, gate 0.949 vs whitened kNN 0.947 (tie); whitening +0.116 tea 9/9, −0.089 cinnamon 4/9 |
| `exp47_cluster_ci.json` | `code/exp47_cluster_ci.py` | §2.9 | Cluster-bootstrap CI on ladder rungs (4000 draws over 5 test blocks) |
| `exp47_cnn_check_tea.json` | `code/exp47_cnn_check_tea.py` | §4 audit | CNN budget-parity check on tea |
| `exp47_cnn_check_cinnamon.json` | `code/exp47_cnn_check_cinnamon.py` | §4 audit | Same on cinnamon |
| `exp47_partA_pub.json` | `code/exp47_baseline_ladder.py` (Part A path) | §3.12 / Table 12 | Prior-free detectors and the gate re-run on the published pool (no held-out adulterants in training) |
| `exp47_partC_gpu.json` | `code/exp47_baseline_ladder.py` (Part C path, GPU) | §3.10 / Table S4 | SSL encoder arms (VICReg / Barlow / randinit) on the same ladder — the "representation not the bottleneck" reading |
| `exp47_noise_c{0..3}.json` | `code/_p47_noise.py` | §3.10 / §3.12 / Tables S8a–S8h | Noise stress stages 1 / 2: iso / sub / drift × {2, 10} × {matched, shift}, blocked 5-fold |
| `exp47_noiseA_c{0..3}.json` | `code/_p47_noise.py` (stage 2, Part A path) | Tables S8a–S8h | Same cells as above, this time including the abstention gate and prior-free detectors |
| `exp47_noiseC.json` | `code/_p47n_stage3.py` | Table S8h (stage 3) | SSL encoders re-pretrained on the noisy pool, per-fold leak guard; three cells |
| `_p47n_verdicts.txt` | `code/_p47n_verdicts.py` | (mechanical N1–N6 readings for stages 1+2) | One line per criterion with the numeric reason it fired or did not |
| `_p47n_stage3_verdicts.txt` | `code/_p47n_stage3_verdict.py` | (mechanical N7 reading) | Same, for N7 and the self-check injection |
| `_p47n_tables.txt` | `code/_p47n_report.py` | (review-facing tables) | Every stage-1 / stage-2 cell rendered as a text table |
| `exp49_maf.json` | `code/_p49_maf.py` | §3.11 / Table S6 series | Monotone-tail MAF on cinnamon 0–24% w/w |
| `exp49b_confound.json` | `code/_p49b_confound.py` | §3.11 controls | Level × block confounding controls |
| `exp49c_structure.json` | `code/_p49c_structure.py` | §3.11 controls | Level structure controls |
| `exp50_bandcov.json` | `code/_p50_bandcov.py` | §3.6 / Tables S7a–S7e | Conditional coverage by unlabelled dose-band × {abs, snv, msc} |
| `exp51_xcorpus.json` | `code/_p51_xcorpus.py` | (cross-corpus, ready for revision) | Ten single-adulterant corpora + D7_pooled + tea (loaded from `exp47` JSON, not re-run) |
| `figure_drawn_values.json` | `code/_fig_data.py` (via `_fig_style.py`) | Figures 1 / 2 / 6 / 7 | Every value each figure actually drew — the gate re-checks them cell-by-cell against the manuscript tables |
| `earlier_round/…` | (external inputs to `exp15_g1.py`) | (historical) | exp07 / exp01 fixes consumed by the G1 pre-check |

## Reproduction status

See `../README.md#reproduction-status` — every ported script was re-run from a disposable
copy of this checkout with only `P5_TEA_DATA` and `P5_PUBLIC_DATA` set, and its JSON output
was diffed leaf-by-leaf against the file shipped here (numerical tolerance 5 × 10⁻⁶ + 1 × 10⁻⁴ · |value|;
keys and label strings must match exactly). The per-script verdict is recorded at the
bottom of the top-level README after the smoke run completes.

## Reading the Chinese label values

`exp13_tea_clean.json`, `exp14_sets.json`, `exp16_per_adulterant.json`,
`exp21_attributability.json`, `exp23_bandsel.json`, `exp24_jointblocks.json`,
`exp33_whitened_attr.json` and the audits carry Chinese adulterant names as **label
values** — 白糖 / 滑石粉 / 硫磺 / 石蜡 / 色素 / 香精 / 糖 + 石蜡 / 滑石 + 石蜡 + 香精 /
纯茶. Their English equivalents are in `../data/README.md`; they were recorded in the
original scripts before the loaders standardised the vocabulary, and the JSONs ship
untouched. Do not silently translate them inside the JSONs — the manuscript's tables were
read from these exact files.
