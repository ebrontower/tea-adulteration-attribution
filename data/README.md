# Data

Two kinds of data feed the pipeline:

1. **Self-acquired tea FTIR workbooks** — **not open-sourced**; not included in this repository.
2. **Public ATR-FTIR / MIR depositions** — *not* redistributed; you download them from the
   DOIs below and point `P5_PUBLIC_DATA` at the folder that holds them.

Everything path-related resolves through `code/p5_paths.py`. Missing inputs raise with a
sentence saying what to set, so a broken environment never silently reads someone else's
copy.

## 1. Self-acquired tea workbooks (not distributed)

The tea FTIR spectra are not open-sourced and are not part of this repository. They are
available from the authors on reasonable request. The scripts expect them at:

```
data/
  tea_batch_sulfur/ALLfinish.xlsx
  tea_batch_clean/ALLfinish2.xlsx
```

or under the folder named by `P5_TEA_DATA`.

Batch → class code → formulation translation is in `code/p5_data.py::TEA_RECIPE`. The two
batches use *different* code conventions (e.g. sulfur-batch code 3 is sulfur while
clean-batch code 3 is paraffin), so the folder name is the batch and the code is translated
before pooling. The nine canonical formulations are:

| Index | Formulation (English) | 中文 |
|---|---|---|
| 0 | pure | 纯茶 |
| 1 | sugar | 白糖 |
| 2 | talc | 滑石粉 |
| 3 | sulfur | 硫磺 |
| 4 | paraffin | 石蜡 |
| 5 | dye | 色素 |
| 6 | flavouring | 香精 |
| 7 | sugar + paraffin (blend) | 糖 + 石蜡 |
| 8 | talc + paraffin + flavouring (blend) | 滑石 + 石蜡 + 香精 |

Older scripts in `code/` (`exp13_tea_clean.py`, `exp14_sets.py`, `exp16_per_adulterant.py`,
`exp23_bandsel.py`) and their result JSONs use the Chinese strings as *label values*, not
as source-code identifiers. Do not silently translate them inside the JSONs — that would
change the values the tables were drawn from. Read this table when you encounter one.

Nominal adulterant loadings 0.5–5% (m/m), 50 spectra per formulation, five-scan blocks.
The two workbooks are the *same* measurements; 7 of the 8 shared formulations are
row-by-row byte-identical, and talc peaks at 0.482 absorbance units of difference. The
pooled set has 450 unique spectra (see §2.1 / `exp13_tea_clean.json`).

## 2. Public depositions (not redistributed)

Point `P5_PUBLIC_DATA` at a folder with this layout. Only the loaders in `code/` care about
the exact filename; you can keep the originals with their native names — the loaders try
both.

```
$P5_PUBLIC_DATA/
  kan/                          # three workbooks, one sheet each; see KAN_SHEETS in
                                # code/p5_public_data.py
    kan_oliveoil.xlsx           # or "zeytinyağı tağşiş.xlsx", sheet Sayfa7
    kan_coffee.xlsx             # or "Kahve_Tağşiş.xlsx",       sheet Sayfa4
    kan_juice.xlsx              # or "meyve suyu tağşiş.xlsx",  sheet Sayfa2
  cinnamon/                     # three series, one adulterant each, same instrument
    FTIR_data_Walnut.xlsx
    FTIR_data_peanut.xlsx
    FTIR_data_wood_dust.xlsx
  turmeric/                     # one csv + its readme
    FTIR_Data.csv
  qib/                          # four Quadram / QIB open corpora
    MIRFreshMeats.csv
    MIR_Fruit_purees.csv
    FTIR_Spectra_olive_oils.csv
    FTIR_Spectra_instant_coffee.csv
  honey_ftir_extract/
    ftir.csv                    # 65 rows, single-beam percent-T; A = -log10(T/100) is
                                # applied inside code/ext_data.py
  coconut/
    coconut_milk_FTIR.xlsx      # 42 rows, NIR transmission, absorbance-like as delivered
```

The `oilblend/` corpus referenced from the pre-flight audit was inspected but is not read
by any script that ships (its `Data_set_FTIR_Raman/` companion is a Raman panel and out of
scope). Retain it locally if you want to re-audit it; nothing in the paper depends on it.

**Sources (cite them alongside the manuscript):**

| Folder | Deposition | DOI / URL |
|---|---|---|
| `kan/` | Batayhi et al., *Foods* 15 (2026) 2949 — olive / coffee / juice ATR-FTIR | https://github.com/abdulhamidbatayhi123/kan-ftir-food-adulteration |
| `cinnamon/` | Mendeley Data, walnut / peanut / wood-dust in cinnamon, three series | https://doi.org/10.17632/rtbsf2r53x.1 |
| `turmeric/` | Mendeley Data, wheat-flour adulterated turmeric | https://doi.org/10.17632/cb5kbt7bgv.2 |
| `qib/` | Quadram Institute / QIB open FTIR + MIR corpora (fresh meats, fruit purées, olive oils, instant coffee) | see the README in the deposition |
| `honey_ftir_extract/` | honey / sugar ATR-FTIR panel (paired preparations, 16 × {0,10,20,40} % w/w) | see the deposition landing page |
| `coconut/` | coconut milk / water NIR transmission (traditional + instant, 7 × {0,10,20} %) | see the deposition landing page |

Which experiments read which folder is in `results/README.md`. Only `exp22_pubgeom.py`,
`exp40_public_pipeline.py`, `exp43_deposit_audit.py`, `exp44_level_block_alias.py`,
`exp45_external_validation.py`, `exp46_headroom_external.py`, `_p49_maf.py`, `_p50_bandcov.py`
and `_p51_xcorpus.py` require `P5_PUBLIC_DATA`; every other script needs only the (non-public) tea data.

## 3. Licence

- Code: MIT — see `../LICENSE`.
- Tea workbooks: not distributed.
- Public depositions: their own licences. This repository redistributes none of them.

## 4. What we did not do

- No interpolation, no resampling to a shared axis. Cross-corpus geometry uses dimensionless
  quantities (correlations, condition numbers, gauge cosines) computed inside each corpus's
  own band.
- No unit rewriting at the loader level: A-axes stay as delivered, except where
  `code/ext_data.py` explicitly converts percent-T to absorbance for the honey panel
  (documented inside that module, and consistent with the cinnamon lesson A = −log₁₀ T).
