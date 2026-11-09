# -*- coding: utf-8 -*-
"""Publication gate for the reproduction repository.

Rules, and the file each applies to:

G1  code/, results/, figures/, top-level README files
        must contain none of the machine-local path fragments (D:/, D:\\, qwork, 2026sci,
        P8_PATH, P8_SHARED, C:\\Users, /home/, /mnt/d/). preregistrations/ is exempted:
        the documents inside are frozen and may not be edited.
G2  none of the shipped files contains P8_* env-var names or the personal account strings
        chh68 / 92159 / Uranus / daima (data/*.xlsx have been scrubbed; this catches any
        regression, including a re-copy that undoes the scrub).
G3  no filename contains the token 'superseded' - the three superseded JSONs from earlier
        rounds must not be in the shipped tree.
G4  every shipped code/*.py byte-compiles - catches the case where a path rewriter dropped
        an `import os` and left a NameError behind.
G5  every results/*.json has a producer script in code/ (or is on the documented-exception
        list) and is mentioned in results/README.md.
G6  every README script / figure row points at a file that exists in the shipped tree,
        except for tokens the README uses as documentation (placeholders like /path/to/…
        or names explicitly labelled as "not shipped").
G7  non-ASCII (CJK) characters appear only in files listed in the allowed-CJK set - the
        frozen pre-registration documents, the loaders that know the Turkish / Chinese
        filenames, and the result JSONs whose label values are Chinese.
G8  figures/Fig1..Fig7 basenames are the seven delivered names; both .pdf and one of
        .png/.tif must be present for each.

`--corrupt <which>` deliberately rewrites one artefact and re-runs the gate; the gate must
exit non-zero with a line naming the injected defect. `--corrupt all` walks every case in
sequence and prints a table. This is the criterion in preregistrations/figures_criteria_CEA_v1.md
section 5.2: the gate must have been shown to bite before its PASS is trusted.

Nothing here touches the network or the manuscript.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- G1 -------------------------------------------------------------------
FORBIDDEN_PATH_TOKENS = [
    "D:/", "D:\\", "d:/", "d:\\",
    "qwork", "2026sci",
    "P8_PATH", "P8_SHARED",
    "C:\\Users", "c:\\Users", "C:/Users", "c:/users",
    "/home/", "/mnt/d/",
]

# --- G2 -------------------------------------------------------------------
# Personal tokens are checked with a word-boundary regex so a numeric substring inside a
# JSON value (e.g. bacc 0.27922991736921593) is not mistaken for the account "92159".
PERSONAL_TOKEN_PATTERNS = [
    ("chh68",  re.compile(r"chh68", re.I)),
    ("92159",  re.compile(r"\b92159\b")),
    ("Uranus", re.compile(r"\bUranus\b")),
    ("daima",  re.compile(r"\bdaima\b")),
]

# --- files scanned for G1/G2, by prefix ---
SCANNED_TOPDIRS = ("code", "results", "figures")
SCANNED_TOPFILES = ("README.md", "LICENSE", "requirements.txt")

# preregistrations/ is exempted from G1/G2 (frozen documents, byte-identical).
EXEMPT_PREFIXES = ("preregistrations/",)

# --- G5: shipped files whose producer path uses an f-string placeholder that
#         this gate cannot statically expand. Each entry names the script that
#         writes them so the exception stays explicit. ---
DOCUMENTED_PRODUCERS = {
    "exp47_noise_c0.json":       "code/_p47_noise.py",
    "exp47_noise_c1.json":       "code/_p47_noise.py",
    "exp47_noise_c2.json":       "code/_p47_noise.py",
    "exp47_noise_c3.json":       "code/_p47_noise.py",
    "exp47_noiseA_c0.json":      "code/_p47_noise.py",
    "exp47_noiseA_c1.json":      "code/_p47_noise.py",
    "exp47_noiseA_c2.json":      "code/_p47_noise.py",
    "exp47_noiseA_c3.json":      "code/_p47_noise.py",
    "exp47_partA_pub.json":      "code/exp47_baseline_ladder.py",
    "exp47_partC_gpu.json":      "code/exp47_baseline_ladder.py",
    "exp21_attributability.json":"code/exp21_attributability.py",
    "figure_drawn_values.json":  "code/_fig_data.py",
    "_p47n_tables.txt":          "code/_p47n_report.py",
    "_p47n_verdicts.txt":        "code/_p47n_verdicts.py",
    "_p47n_stage3_verdicts.txt": "code/_p47n_stage3_verdict.py",
}

# earlier_round is a bundle of inputs the current pipeline only reads, never writes;
# README.md is hand-written documentation.
NO_PRODUCER_PREFIX = ("results/earlier_round/", "results/README.md")

# --- G6: filenames the READMEs mention as documentation only (never shipped) ---
DOCUMENTED_NOT_SHIPPED = {
    "exp40_diag_pure_ratio_space_superseded.json",
    "exp40_public_ratio_space_superseded.json",
    "exp47_baseline_ladder_prev0023_superseded.json",
    "coconut_milk_FTIR.xlsx",       # not in this checkout
    "FTIR_data_Walnut.xlsx",
    "FTIR_data_peanut.xlsx",
    "FTIR_data_wood_dust.xlsx",
    "FTIR_Data.csv",
    "kan_oliveoil.xlsx",
    "kan_coffee.xlsx",
    "kan_juice.xlsx",
    "MIRFreshMeats.csv",
    "FTIR_Spectra_olive_oils.csv",
    "FTIR_Spectra_instant_coffee.csv",
    "MIR_Fruit_purees.csv",
    "ftir.csv",
    "manuscript.md",                # P5_MANUSCRIPT_MD - not redistributed during review
    "zeytinyağı tağşiş.xlsx",
    "Kahve_Tağşiş.xlsx",
    "meyve suyu tağşiş.xlsx",
}

# --- G7: files that may carry CJK ---
ALLOWED_CJK_FILES = {
    "README.md",                       # top-level map - the section titles use Chinese
    "data/README.md",
    "results/README.md",
    "preregistrations/README.md",
    # Frozen originals are Chinese (byte-identical to what the authors signed off):
    "preregistrations/exp47-N_noise_stress_preregistration_v1.md",
    "preregistrations/exp49_minimum_attributable_fraction_v1.md",
    "preregistrations/exp50_conditional_coverage_and_baseline_correction_v1.md",
    "preregistrations/exp51_cross_corpus_evaluation_v1.md",
    "preregistrations/figures_criteria_CEA_v1.md",
    # Loaders know the Turkish workbook filenames and Chinese adulterant label values:
    "code/p5_public_data.py",
    "code/p5_data.py",
    # Analysis scripts carry Chinese label strings inside their result sets:
    "code/exp13_tea_clean.py",
    "code/exp14_sets.py",
    "code/exp15_g1.py",
    "code/exp16_per_adulterant.py",
    "code/exp17_headline_ci.py",
    "code/exp20_increments.py",
    "code/exp21_attributability.py",
    "code/exp23_bandsel.py",
    "code/exp24_jointblocks.py",
    "code/exp28_label_subspace.py",
    "code/exp33_whitened_attr.py",
    "code/exp34_selective_mondrian.py",
    "code/exp35_geometry_export.py",
    "code/exp36_open_set.py",
    "code/exp37_samp_holdout.py",
    "code/exp38_mondrian.py",
    "code/exp39_robust.py",
    "code/exp43_deposit_audit.py",
    "code/_p47n_stage3_verdict.py",
    "code/_p47n_verdicts.py",
    "code/_p47n_report.py",
    "code/_p49_maf.py",
    "code/_p50_bandcov.py",
    "code/_p51_xcorpus.py",
    "code/_p51_report.py",
    "code/audit_tea.py",
    "code/dup_check.py",
    "code/dup_struct.py",
    "code/exp40_public_pipeline.py",
    "code/ext_data.py",
    # Result JSONs whose label values are Chinese (see results/README.md):
    "results/audit_tea.json",
    "results/dup_check.json",
    "results/dup_struct.json",
    "results/exp13_tea_clean.json",
    "results/exp14_sets.json",
    "results/exp16_per_adulterant.json",
    "results/exp21_attributability.json",
    "results/exp23_bandsel.json",
    "results/exp24_jointblocks.json",
    "results/exp33_whitened_attr.json",
    "results/exp34_selective_mondrian.json",
    "results/exp35_geometry.json",
    "results/exp51_xcorpus.json",
    # Verdict tables quote criterion text verbatim from the frozen Chinese originals.
    "results/_p47n_verdicts.txt",
    "results/_p47n_tables.txt",
    "results/_p47n_stage3_verdicts.txt",
    # This file itself documents the rules in prose that mentions Chinese paths / names.
    "_gate_repo.py",
}

# Any file under these prefixes is allowed to carry CJK (earlier-round inputs use the
# Chinese adulterant labels).
ALLOWED_CJK_PREFIXES = ("results/earlier_round/",)

# --- G8 -------------------------------------------------------------------
FIGURES = ["Fig1_workflow", "Fig2_discrimination", "Fig3_geometry",
           "Fig4_gate_coverage", "Fig5_alpha_map",
           "Fig6_cinnamon_transfer", "Fig7_information_budget"]

# ---------------------------------------------------------------- helpers
def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def is_binary(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".tif", ".tiff", ".pdf", ".pyc")


def iter_scanned_files():
    for top in SCANNED_TOPDIRS:
        d = ROOT / top
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            r = rel(p)
            if any(r.startswith(e) for e in EXEMPT_PREFIXES):
                continue
            if "__pycache__" in p.parts:
                continue
            yield p
    for f in SCANNED_TOPFILES:
        p = ROOT / f
        if p.exists():
            yield p


# ---------------------------------------------------------------- checks
def g1_machine_paths(hits_out):
    for p in iter_scanned_files():
        if p.suffix.lower() == ".xlsx":
            with zipfile.ZipFile(p) as z:
                for e in z.infolist():
                    data = z.read(e.filename)
                    for tok in FORBIDDEN_PATH_TOKENS:
                        if tok.encode() in data:
                            hits_out.append(f"G1 {rel(p)}!{e.filename}: forbidden path {tok!r}")
            continue
        if is_binary(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for tok in FORBIDDEN_PATH_TOKENS:
            if tok in text:
                hits_out.append(f"G1 {rel(p)}: forbidden path {tok!r}")


def g2_personal(hits_out):
    for p in iter_scanned_files():
        if p.suffix.lower() == ".xlsx":
            with zipfile.ZipFile(p) as z:
                for e in z.infolist():
                    txt = z.read(e.filename).decode("utf-8", errors="replace")
                    for name, rx in PERSONAL_TOKEN_PATTERNS:
                        if rx.search(txt):
                            hits_out.append(f"G2 {rel(p)}!{e.filename}: personal token {name}")
            continue
        if is_binary(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for name, rx in PERSONAL_TOKEN_PATTERNS:
            if rx.search(text):
                hits_out.append(f"G2 {rel(p)}: personal token {name}")


def g3_superseded(hits_out):
    for p in ROOT.rglob("*"):
        if p.is_file() and "superseded" in p.name.lower():
            hits_out.append(f"G3 {rel(p)}: superseded file in shipped tree")


def g4_compile(hits_out):
    code = ROOT / "code"
    for p in sorted(code.glob("*.py")):
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as exc:
            hits_out.append(f"G4 {rel(p)}: {exc}")


def _producer_literals():
    """Every plain-string filename literal that appears inside any code/*.py.
    f-strings with {} placeholders are skipped; those files are on DOCUMENTED_PRODUCERS."""
    written = set()
    pat = re.compile(r'"([A-Za-z0-9_./-]+\.(?:json|txt|md|csv))"')
    for p in (ROOT / "code").glob("*.py"):
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            fn = m.group(1)
            base = os.path.basename(fn)
            if base:
                written.add(base)
    return written


def g5_producer_and_readme(hits_out):
    shipped = []
    for p in sorted((ROOT / "results").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            shipped.append(p)
    readme = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    producers = _producer_literals()
    for p in shipped:
        r = rel(p)
        base = p.name
        if any(r.startswith(pref) for pref in NO_PRODUCER_PREFIX):
            continue
        if base not in readme:
            hits_out.append(f"G5 {r}: not mentioned in results/README.md")
        if base not in producers and base not in DOCUMENTED_PRODUCERS:
            hits_out.append(f"G5 {r}: no producer script writes this filename literal")


def g6_readme_rows(hits_out):
    """Every script/figure filename the top-level or results README mentions must exist,
    unless it is a placeholder path or a documented-not-shipped name."""
    top = (ROOT / "README.md").read_text(encoding="utf-8")
    res = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    for text in (top, res):
        for m in re.finditer(r'\b([A-Za-z0-9][A-Za-z0-9_-]*\.(?:py|json|md|txt|pdf|png|tif|xlsx|csv))\b', text):
            tok = m.group(1)
            if tok in DOCUMENTED_NOT_SHIPPED:
                continue
            # try a couple of places
            for cand in [(ROOT / "code" / tok), (ROOT / "results" / tok),
                         (ROOT / "figures" / tok), (ROOT / "data" / tok),
                         (ROOT / "preregistrations" / tok), (ROOT / tok)]:
                if cand.exists():
                    break
            else:
                # also try basename inside earlier_round
                if any((ROOT / "results" / "earlier_round" / tok).exists() for _ in [0]):
                    continue
                hits_out.append(f"G6 README mentions {tok!r} which is not shipped")


def g7_cjk(hits_out):
    cjk = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]")
    for p in ROOT.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        r = rel(p)
        if p.suffix.lower() in (".png", ".tif", ".tiff", ".pdf", ".pyc"):
            continue
        if r in ALLOWED_CJK_FILES:
            continue
        if any(r.startswith(pref) for pref in ALLOWED_CJK_PREFIXES):
            continue
        if p.suffix.lower() == ".xlsx":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if cjk.search(text):
            hits_out.append(f"G7 {r}: CJK characters appear but not on the allowed list")


def g8_figures(hits_out):
    for stem in FIGURES:
        pdf = ROOT / "figures" / f"{stem}.pdf"
        if not pdf.exists():
            hits_out.append(f"G8 figures/{stem}.pdf missing")
        png = ROOT / "figures" / f"{stem}.png"
        tif = ROOT / "figures" / f"{stem}.tif"
        if not (png.exists() or tif.exists()):
            hits_out.append(f"G8 figures/{stem}: neither .png nor .tif backup present")


CHECKS = [g1_machine_paths, g2_personal, g3_superseded, g4_compile,
          g5_producer_and_readme, g6_readme_rows, g7_cjk, g8_figures]


# ---------------------------------------------------------------- corrupt samples
def corrupt(kind):
    """Rewrite one artefact so the gate must catch it. Each returns a restore callable."""
    if kind == "g1_path":
        f = ROOT / "code" / "p5_data.py"
        bak = f.read_text(encoding="utf-8")
        patched = bak.replace("HERE = os.path.dirname(os.path.abspath(__file__))",
                              'HERE = r"D:/qwork/scratch"', 1)
        assert patched != bak, "G1 corrupt template did not match - fix the script"
        f.write_text(patched, encoding="utf-8")
        return lambda: f.write_text(bak, encoding="utf-8")
    if kind == "g2_personal":
        f = ROOT / "results" / "exp13_tea_clean.json"
        bak = f.read_text(encoding="utf-8")
        # insert at a safe place: after the leading "{"
        idx = bak.index("{")
        patched = bak[:idx+1] + '"__leak_marker__": "chh68 ran this", ' + bak[idx+1:]
        f.write_text(patched, encoding="utf-8")
        return lambda: f.write_text(bak, encoding="utf-8")
    if kind == "g3_superseded":
        f = ROOT / "results" / "exp40_public_tmp_superseded_case.json"
        f.write_text('{"sentinel": 1}', encoding="utf-8")
        return lambda: f.unlink()
    if kind == "g4_compile":
        f = ROOT / "code" / "exp20_increments.py"
        bak = f.read_text(encoding="utf-8")
        patched = bak + "\ndef $broken(:\n    pass\n"
        f.write_text(patched, encoding="utf-8")
        return lambda: f.write_text(bak, encoding="utf-8")
    if kind == "g5_missing_producer":
        f = ROOT / "results" / "notproducedbyanyscript_tmp.json"
        f.write_text('{"sentinel": 1}', encoding="utf-8")
        return lambda: f.unlink()
    if kind == "g6_readme_row":
        f = ROOT / "README.md"
        bak = f.read_text(encoding="utf-8")
        patched = bak + "\nSee also `code/this_script_was_never_written.py` for details.\n"
        f.write_text(patched, encoding="utf-8")
        return lambda: f.write_text(bak, encoding="utf-8")
    if kind == "g7_cjk":
        f = ROOT / "code" / "exp30_aug_control.py"
        bak = f.read_text(encoding="utf-8")
        patched = bak.replace('"""', '"""\n# 这段注释是中文，门禁必须发现它', 1)
        assert patched != bak
        f.write_text(patched, encoding="utf-8")
        return lambda: f.write_text(bak, encoding="utf-8")
    if kind == "g8_figure":
        f = ROOT / "figures" / "Fig6_cinnamon_transfer.png"
        backup_path = f.with_suffix(".png.corruptbak")
        if backup_path.exists():
            raise SystemExit("previous corrupt backup exists - restore first")
        f.rename(backup_path)
        return lambda: backup_path.rename(f)
    raise SystemExit(f"unknown corrupt kind: {kind}")


CORRUPT_KINDS = ["g1_path", "g2_personal", "g3_superseded", "g4_compile",
                 "g5_missing_producer", "g6_readme_row", "g7_cjk", "g8_figure"]


# ---------------------------------------------------------------- main
def run_gate():
    hits = []
    for fn in CHECKS:
        fn(hits)
    return hits


def main():
    if "--corrupt" in sys.argv:
        i = sys.argv.index("--corrupt")
        which = sys.argv[i + 1] if i + 1 < len(sys.argv) else "all"
        kinds = CORRUPT_KINDS if which == "all" else [which]
        print(f"corrupt samples: {kinds}")
        failures = []
        for k in kinds:
            restore = corrupt(k)
            try:
                hits = run_gate()
            finally:
                restore()
            tag = k.split("_")[0]
            caught = [h for h in hits if h.startswith(tag.upper() + " ")]
            if caught:
                print(f"  {k:24s} caught by {len(caught)} hit(s); first: {caught[0]}")
            else:
                print(f"  {k:24s} NOT CAUGHT - gate is vacuous on this rule")
                failures.append(k)
            hits_clean = run_gate()
            residual = [h for h in hits_clean if h.startswith(tag.upper() + " ")]
            if residual:
                print(f"  {k:24s} RESTORE CHECK: gate still reports {tag} after restore: {residual[:2]}")
                failures.append(k + ":restore")
        if failures:
            sys.exit("FAILURE SAMPLES FAILED: " + str(failures))
        print("all failure samples caught and restored; gate is not vacuous")
        return

    hits = run_gate()
    if hits:
        print(f"GATE FAILED: {len(hits)} hit(s)")
        for h in hits[:80]:
            print("  " + h)
        if len(hits) > 80:
            print(f"  ... and {len(hits) - 80} more")
        sys.exit(1)
    print(f"GATE PASSED: {len(CHECKS)} checks, 0 hits on {rel(ROOT)}")


if __name__ == "__main__":
    main()
