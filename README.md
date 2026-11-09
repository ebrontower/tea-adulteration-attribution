# Tea adulterant attribution from FTIR spectra

Reproduction package for the accompanying manuscript (under review).
Every number in the paper's tables traces to a JSON file in [`results/`](results/), and every
JSON file traces to a script in [`code/`](code/). Five criteria documents frozen before the
runs are in [`preregistrations/`](preregistrations/); nothing there was edited after the
numbers came in.

The paper reports that a whitened residual metric — with the reference dictionary, candidate
set and split-conformal protocol all fixed — turns single-adulterant naming on tea ATR-FTIR
from 0.503 to 0.974 and reproduces on a public cinnamon corpus (0.513 → 0.964) at the same
nominal coverage, and that this gain follows from a geometry (dictionary coherence, condition
number, gauge cosine) computable before any model is fitted. It also reports the negative
side: the same metric does not buy discrimination headroom, whitening is not uniformly
helpful across corpora, and the assurance is only valid on spectra exchangeable with the
calibration pool.

## Layout

```
code/                 68 scripts: analysis, figure drawing, gates, data loaders
data/                 data documentation only — the tea spectra are not public (see data/README.md)
results/              60 exported JSONs + earlier-round inputs + 47-N text reports
figures/              the seven delivered figure files (vector PDF primary, PNG/TIFF backup)
preregistrations/     five frozen criteria documents (see preregistrations/README.md)
requirements.txt      pinned lower bounds; Python 3.12 tested
```

Every path resolves from [`code/p5_paths.py`](code/p5_paths.py) — the only module that
knows the repository layout. Scripts do not hard-code a machine location.

## Setup

```bash
# Python 3.12, tested with uv on Windows and Linux
pip install -r requirements.txt

# The self-acquired tea spectra are NOT included in this repository. If you have been given
# a copy, put it under data/ or point P5_TEA_DATA at it.
# Public depositions are NOT redistributed here (see data/README.md for the six sources
# and the expected subfolder layout).
export P5_PUBLIC_DATA=/path/to/depositions           # required by exp22 / exp40 / exp45 /
                                                     # exp46 / exp49 / exp50 / exp51
```

Only the figure gate (`code/_fig_gates.py` and `code/_fig_tables.py`) needs a local copy
of the manuscript text, because it re-parses the tables the figures were drawn from so
a figure cannot silently drift away from its table:

```bash
export P5_MANUSCRIPT_MD=/path/to/manuscript.md       # only for the figure gate
```

Every accessor raises with the sentence saying what to set. A missing input fails loudly
instead of quietly reading somebody else's copy.

## Running

Scripts write into `results/` and are safe to run individually. Rough cost on a laptop
CPU; two GPU sections are marked.

| Command | What it produces | Typical wall time |
|---|---|---|
| `python code/exp13_tea_clean.py` | the tea dedup / autocorrelation audit (§2.1, §3.1) | ~1 min |
| `python code/exp20_increments.py` | dictionary coherence / condition number / gauge cosine (§3.4, Table 2) | ~1 min |
| `python code/exp33_whitened_attr.py` | the whitening gain and conformal sets (§3.5, Table 3) | ~2 min |
| `python code/exp34_selective_mondrian.py` | per-group coverage, α-map (§3.6–3.8, Table 5) | ~3 min |
| `python code/exp36_open_set.py` | dictionary-out adulterants, two-sided gate (§3.9) | ~3 min |
| `python code/exp40_public_pipeline.py` | cinnamon transfer reproduction (§3.11, Tables 8/9) — needs `P5_PUBLIC_DATA/cinnamon/` | ~8 min |
| `python code/exp47_baseline_ladder.py` | the nine-rung information-budget ladder (§3.12, Tables 11/12) | ~20 min |
| `python code/_p47_noise.py` + `_p47n_*.py` | noise stress test stages 1 and 2 (Table S8a–S8h) | ~30 min |
| `python code/_p47n_stage3.py` | noise stress stage 3: SSL encoders re-pretrained on the noisy pool (GPU; `--cells clean/iso2-matched/iso2-shift`) | 45–60 min on an RTX 3060 laptop |
| `python code/_p49_maf.py` | dose–response MAF on cinnamon (Table S6 series) | ~10 min |
| `python code/_p50_bandcov.py` | conditional coverage by unlabelled dose-band × baseline correction (Table S7a–S7e) | ~15 min |
| `python code/_p51_xcorpus.py` | cross-corpus evaluation, ten corpora in one table | ~25 min |
| `python code/_fig_tables.py && python code/_mk_fig{1,2,6,7}.py && python code/_fig_gates.py` | regenerate Figs 1, 2, 6, 7 from parsed manuscript tables, then run the drawn-vs-table gate | ~2 min |

Scripts for the older SSL-era probes (`exp15`, `exp19`, `exp24`–`exp31`) are shipped because
they produce the negative controls cited in §3.10, but they are marked as historical in
[`results/README.md`](results/README.md); the current claims do not depend on them.

## Script → result → manuscript section

This is the full map; each row names the JSON file(s) a table's cells are drawn from.

| Script | Result(s) | Manuscript role |
|---|---|---|
| `exp13_tea_clean.py` | `exp13_tea_clean.json` | §2.1 / §3.1 / §4.5: 850 rows → 450 unique spectra, autocorrelation time, block-split rationale |
| `exp14_sets.py` | `exp14_sets.json` | §1 / §3.3: multi-label PLS-DA exact-set 0.864 → 0.000 on held-out blends |
| `exp15_g1.py` | `exp15_g1.json` | Historical: L/C → Δ(SSL) pre-check, negative |
| `exp16_per_adulterant.py` | `exp16_per_adulterant.json` | §3.2: seven of eight pairwise AUC = 1.00, dye 0.944 |
| `exp17_headline_ci.py` | `exp17_headline_ci.json` | §2.9: 25-fold mean + cluster-bootstrap CI for every headline cell |
| `exp18_unmix.py` | `exp18_unmix.json` | §2.8 / §3.3: NNLS with recipe means as endmembers, still 0 composite recovery |
| `exp19_desc_public7.py` | `exp19_desc_public7.json` | Historical: descriptor–Δ association restricted to the seven corpora actually used |
| `exp20_increments.py` | `exp20_increments.json` | §1 / §3.4 / Table 2: coherence 0.987 / condition 44.7 / gauge −0.993 (raw); 0.581 / 8.20 / −0.403 (whitened) |
| `exp21_attributability.py` | `exp21_attributability.json` | §3.4–3.5 discussion base — formalises "not attributable" as a dictionary-condition-number issue |
| `exp22_pubgeom.py` | `exp22_pubgeom.json` | §3.11 / §4.6: same near-collinearity on cinnamon (coherence 0.994 / condition 550 / peanut gauge 0.988) |
| `exp23_bandsel.py` | `exp23_bandsel.json` | §3.4 end: window robustness, 0.976–0.987 across four fingerprint windows |
| `exp24_jointblocks.py` | `exp24_jointblocks.json` | Historical (negative): multi-band joint modelling cannot recover set recall |
| `exp25_interp_floor.py` | `exp25_interp_floor.json` | §3.10: MAE embeddings are 75–97% spanned by the first five PCs |
| `exp26_nonrecon_ssl.py` | `exp26_nonrecon_ssl.json` | §1 / §3.10: VICReg / Barlow beat random-init on only 3 of 8 corpora |
| `exp27_synthetic_phase.py` | `exp27_synthetic_phase.json` | Historical: replaced by exp29 |
| `exp28_label_subspace.py` | `exp28_label_subspace.json` | §1 / §3.10: 99.5–100% of between-class variance is inside the first 20 PCs on all eight corpora |
| `exp29_phase_band.py` | `exp29_phase_band.json` | §3.10 mechanism: gain vs out-of-smooth-subspace label energy |
| `exp30_aug_control.py` | `exp30_aug_control.json` | §3.10: VICReg/Barlow gains reproduce under augmentation alone |
| `exp31_synth_control.py` | `exp31_synth_control.json` | Historical (negative): MAE loses to a 4-PC probe on the synthetic set for information reasons |
| `exp32_conformal_attr.py` | `exp32_conformal_attr.json` | §3.5–3.6 baseline: Euclidean conformal sets valid but ~30 of 64 candidates |
| `exp33_whitened_attr.py` | `exp33_whitened_attr.json` | §3.5 / Table 3 / Fig. 2: single-adulterant blocked top-1 0.503 → 0.974, set 23.2 → 16.0 of 64, coverage 0.946 |
| `exp34_selective_mondrian.py` | `exp34_selective_mondrian.json` | §3.6–3.8 / Table 5: per-group coverage, α-map (0.976 / 0.907 / 0.864 / 0.809) |
| `exp35_geometry_export.py` | `exp35_geometry.json` | Fig. 1 (author's original) and Table 2 source — 6×6 Gram raw / whitened |
| `exp36_open_set.py` | `exp36_open_set.json` | §3.9 / §4.6: two-sided gate abstains on five of six out-of-dictionary adulterants |
| `exp37_samp_holdout.py` | `exp37_samp_holdout.json` | §4.6: integral autocorrelation time 1.7–19 scans → 450 spectra ≈ 90 independent observations |
| `exp38_mondrian.py` | `exp38_mondrian.json` | §3.6: oracle conditional calibration lifts ternary coverage 0.280 → 0.940 at the cost of set size |
| `exp39_robust.py` | `exp39_robust.json` | §3.4 / §3.3: hyperparameter sweep k_w 40 → 320, top-1 stays 0.960–0.966, coverage 0.902–0.916 |
| `exp40_diag.py` | `exp40_diag_pure.json` | §2.7 / §3.11 / Tables 8/9: cinnamon single-adulterant pipeline diagnostics |
| `exp40_public_pipeline.py` | `exp40_public.json` | §2.7 / §3.11 / Tables 8/9: top-1 0.513 → 0.964 on cinnamon, marginal coverage unchanged |
| `exp41_transfer.py` | `exp41_transfer.json` | Historical: cross-session transfer already at the linear ceiling; 6 nuisance arms flat |
| `exp42_magnitude_transfer.py` | `exp42_magnitude_transfer.json` | Historical: low-dose detection is not the bottleneck, low-dose extrapolation is (+103% ~ +541% bias) |
| `exp43_deposit_audit.py` | `exp43_deposit_audit.json` | §2.7 / §4.6: four of seven level-recorded corpora have a step-function "dose axis" (r²_dose 0.12–0.76) |
| `exp44_level_block_alias.py` | `exp44_level_block_alias.json` | §2.7 / §4.6: level-by-row segmentation in D7w / D7d / D7p / D8 vs interleaving in KAN |
| `exp45_external_validation.py` | `exp45_external_validation.json` | §4.6: leakage-free external validation — ranking holds, numbers do not (Spearman 0.81 [0.21, 1.00], R² −0.24) |
| `exp46_headroom_external.py` | `exp46_headroom_onsample.json` | §3.10: out-of-sample headroom check, 17 of 22 arms ≤ 0.02 |
| `exp47_baseline_ladder.py` | `exp47_baseline_ladder.json` | §2.8 / §3.12 / Tables 11/12 / Fig. 7: nine-rung ladder, gate 0.949 vs whitened kNN 0.947 (tie); whitening +0.116 tea 9/9, −0.089 cinnamon 4/9 |
| `exp47_cluster_ci.py` | `exp47_cluster_ci.json` | §2.9: cluster-bootstrap CI for the ladder rungs |
| `exp47_cnn_check_tea.py` | `exp47_cnn_check_tea.json` | §4 audit: CNN budget-parity check on tea |
| `exp47_cnn_check_cinnamon.py` | `exp47_cnn_check_cinnamon.json` | §4 audit: same check on cinnamon |
| `_p47_noise.py` | `exp47_noise_c{0..3}.json`, `exp47_noiseA_c{0..3}.json` | §3.10 / §3.12 / Tables S8a–S8h: iso / sub / drift × {2, 10} × {matched, shift}, blocked 5-fold |
| `_p47n_verdicts.py` | `_p47n_verdicts.txt`, `_p47n_tables.txt` | N1–N7 mechanical readings (see preregistrations/README.md) |
| `_p47n_stage3.py` | `exp47_noiseC.json` | Stage 3 (GPU): SSL encoders re-pretrained on the noisy pool, per-fold leak guard |
| `_p49_maf.py` | `exp49_maf.json` | §3.11 / Table S6 series: monotone-tail MAF on cinnamon 0–24% w/w |
| `_p49b_confound.py`, `_p49c_structure.py` | `exp49b_confound.json`, `exp49c_structure.json` | Level×block aliasing controls for the MAF curve |
| `_p50_bandcov.py` | `exp50_bandcov.json` | §3.6 / Tables S7a–S7e: conditional coverage by unlabelled dose-band × {abs, snv, msc} |
| `_p51_folds.py`, `_p51_prep.py`, `_p51_xcorpus.py`, `_p51_report.py` | `exp51_xcorpus.json` | Cross-corpus evaluation over ten single-adulterant corpora + D7_pooled |
| `make_figs.py` | the seven files under `figures/` | Regenerates Figs 3/4/5 (author's originals — see below) |
| `_fig_data.py`, `_fig_style.py`, `_fig_tables.py` | `figure_drawn_values.json` | Parses Figs 1/2/6/7 back into the manuscript tables they were drawn from |
| `_mk_fig1.py`, `_mk_fig2.py`, `_mk_fig6.py`, `_mk_fig7.py` | PDF + PNG under `figures/` | Draws the four CEA-only new figures |
| `_fig_gates.py` | stdout | Cell-by-cell gate: drawn value == parsed table value (three decimals); also runs the §0 style checks (font size, near-black text, no cross-reference, no spill) |

Auditing side-programs `audit_tea.py`, `dup_check.py`, `dup_struct.py`, `ext_data.py`,
`qib_data.py`, `p5_data.py`, `p5_public_data.py`, `p5_paths.py` are the loaders /
de-duplication scripts the pipeline imports.

## Figures

Three of the seven figures are the authors' own originals (`Fig. 3 geometry`,
`Fig. 4 gate coverage`, `Fig. 5 α-map`) and are shipped unchanged (bytes and pixels
identical; only the filename prefix was renumbered from the older `Fig1/2/3_…`).
Four figures (workflow, discrimination, cinnamon transfer, information-budget ladder)
were drawn for this submission from the manuscript tables by the four
`_mk_fig*.py` scripts. Their numeric content is gated by `_fig_gates.py` — every cell
must equal the corresponding table cell to three decimals, and the failure-sample mode
(`python code/_fig_gates.py --corrupt`) proves the gate catches a 0.01 drift.

## What is not redistributed here

- **The self-acquired tea FTIR spectra.** They are not open-sourced. The scripts and
  result JSONs are provided so the analysis can be inspected; running the tea experiments
  requires the data, available from the authors on reasonable request.
- **The public ATR-FTIR / MIR depositions** (kan, cinnamon, turmeric, qib, honey,
  coconut, oil-blend). Download them from the DOIs listed in
  [`data/README.md`](data/README.md) and set `P5_PUBLIC_DATA`. Their licences are theirs
  to serve.
- **The manuscript markdown** while the article is under review. Only the figure gate
  needs it (`P5_MANUSCRIPT_MD`). Post-acceptance this note will change.
- **The docx / PDF compilation pipeline**. That is a separate deliverable — the
  `results/*.json` and `figures/*.pdf` are the paper's numeric and visual substrate.

## Pre-registration discipline

Five criteria documents frozen before the runs live in [`preregistrations/`](preregistrations/).
Their original text is Chinese; the shipped filenames are anglicised and one `prereg:`
pointer line inside ten result JSONs was rewritten to the new path. **No criterion, arm,
band count or threshold in any of the five documents was altered after the numbers came
in** — see `preregistrations/README.md` for the N1–N7 / L1–L4 / C1–C7 / S1–S5 statements
and how each verdict came out.

## Reproduction status

Every ported analysis script was re-run from a disposable copy of this checkout with only
`P5_TEA_DATA` and `P5_PUBLIC_DATA` set, and its JSON output was diffed leaf-by-leaf
against the shipped file (numerical tolerance 5 × 10⁻⁶ + 1 × 10⁻⁴ · |value|, key sets
and label strings must match exactly).

| Verdict | Scripts | Notes |
|---|---|---|
| **Reproduces** (identical JSON) | exp13, exp16, exp17, exp18, exp22, exp32, exp35, exp36, exp37, exp38 | 10 scripts; zero leaf diffs |
| **Numerically identical; label encoding differs** | exp14, exp20, exp23, exp24, exp33, exp34 | 6 scripts; the shipped JSON carries Chinese adulterant names from the original workspace; the repo's `p5_data.py` returns English translations. All float leaves match. |
| **Requires companion modules or GPU** | exp15, exp19, exp21, exp25, exp27, exp28, exp29, exp39, exp40_diag | 9 scripts; depend on `P.pub.load()`, `P.atr_mod` (论文7's ATR correction), `P.SELF_ALL` (auxiliary self-acquired corpora), or GPU SSL pretraining. These are analysis scripts whose outputs are shipped as results but whose full execution requires the original multi-paper workspace. |

Scripts in the third category are included for transparency (the code shows exactly what was
computed) but a standalone reproduction requires the companion modules described in
[`data/README.md`](data/README.md).

## License

Code in this repository is released under the MIT License ([`LICENSE`](LICENSE)). The
self-acquired tea spectra are not distributed. Public
depositions retain their own licences — see their respective DOIs in
[`data/README.md`](data/README.md).

## Citation

If you use the code, please cite the manuscript (BibTeX will be added
once the venue DOI is issued; until then cite by title).
