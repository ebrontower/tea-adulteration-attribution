"""Gate for the four new CEA figures — run this before anything is copied into the package.

Checks, in the order the criteria file fixes them (preregistrations/figures_criteria_CEA_v1.md §5):

1. every value a figure recorded as drawn is compared, cell by cell, against the table re-parsed
   from manuscript.md at three decimals, and the label sets must match exactly, so a missing row,
   a retyped literal or a stale ledger all fail;
2. a dash or an 'absent' cell must be recorded as nothing drawn, never as a zero;
3. the house style of §0 is re-checked on the live figures (minimum type, near-black text, no
   overlapping or clipped text, no in-art cross-references, no first-person prose);
4. Fig. 1 may only print numbers the manuscript already prints, so each one is grepped out of v3;
5. the delivered files exist as PNG + vector PDF pairs at 16.0 cm and at least 300 dpi.

`--corrupt` deliberately breaks one drawn value and asserts that check 1 catches it; the criteria
require that failure sample to be run before the gate is trusted on the real data.
"""
from __future__ import annotations

import importlib
import re
import struct
import sys

import _fig_style as S
from _fig_tables import (S4, S5, T5a, T5b, V3, bootstrap_intervals, prose_numbers)
import p5_paths  # noqa: E402

FIGS = ["Fig1_workflow", "Fig2_discrimination", "Fig6_cinnamon_transfer", "Fig7_information_budget"]
TOL = 5e-4
WIDTH_CM, MIN_DPI = 16.0, 300.0


def expected():
    """The label -> value map the four figures must have drawn, straight from the tables."""
    e = {}

    s4 = S4()
    for k, v in s4.items():
        for i, col in enumerate(("tea raw", "tea whitened", "tea Δ",
                                 "cinnamon raw", "cinnamon whitened", "cinnamon Δ")):
            e["Fig2_discrimination", f"{col}/{k}"] = v[i]

    t5a, bs = T5a(), bootstrap_intervals()
    EU, WH = "Euclidean | standard", "Whitened (by adulterant) | standard"
    a_rows = {"Coverage": ("coverage", None, "coverage"),
              "Top-1": ("top1", "top1_euclidean", "top1"),
              "Mean set\n(of 8)": ("size", None, "size"),
              "Empty sets": ("empty", None, "empty")}
    for name, (cell, ei, ew) in a_rows.items():
        e["Fig6_cinnamon_transfer", f"a raw/{name}"] = t5a[EU][cell]
        e["Fig6_cinnamon_transfer", f"a whitened/{name}"] = t5a[WH][cell]
        e["Fig6_cinnamon_transfer", f"a interval raw/{name}"] = list(bs[ei]) if ei else None
        e["Fig6_cinnamon_transfer", f"a interval whitened/{name}"] = list(bs[ew])
    t5b, prov = T5b(), prose_numbers()
    for g in ("Walnut shell", "Peanut", "Wood dust", "Authentic cinnamon",
              "Loading 1–4%", "Loading 6–12%", "Loading 15–24%", "All spectra"):
        for what, key in (("b coverage", "coverage"), ("b top-1", "top1"),
                          ("b set", "size"), ("b acceptance", "accept")):
            e["Fig6_cinnamon_transfer", f"{what}/{g}"] = list(t5b[g][key])
    e["Fig6_cinnamon_transfer", "b caveat/Wood dust"] = prov["wood dust empty sets"]
    e["Fig6_cinnamon_transfer", "b caveat/Authentic cinnamon"] = prov["authentic only among accepted"]

    for k, d in S5().items():
        # the 'absent' row has no AUROC in any column, so only its orphan cell is recorded as undrawn
        whats = ("orphan",) if d["absent"] else ("orphan", "comp_singles", "comp_pub")
        for what in whats:
            e["Fig7_information_budget", f"{what}/{k}"] = d[what]
        e["Fig7_information_budget", f"rung/{k}"] = d["rung"]

    import _fig_data as D
    t, g = D.tea(), D.gram()
    e["Fig1_workflow", "window channels"] = t["channels"]
    e["Fig1_workflow", "gram coherence raw"] = g["coherence"]
    e["Fig1_workflow", "gram coherence whitened"] = g["coherence_white"]
    return e


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        a, b = list(a), list(b)
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= TOL


def check_values(drawn, exp):
    bad = []
    got = {(f, d["label"]): d["value"] for f, v in drawn.items() for d in v}
    for key, val in exp.items():
        if key not in got:
            bad.append(f"not drawn at all: {key[0]} {key[1]}")
        elif not same(got[key], val):
            bad.append(f"differs from the table: {key[0]} {key[1]} drew {got[key]} against {val}")
    for key in got:
        if key not in exp:
            bad.append(f"drawn but not in any table: {key[0]} {key[1]} = {got[key]}")
    # a dash must never become a zero
    for key, val in exp.items():
        if val is None and isinstance(got.get(key), (int, float)) and got[key] == 0:
            bad.append(f"an empty cell was drawn as zero: {key[0]} {key[1]}")
    return bad


def check_files():
    import os
    bad = []
    for fid in FIGS:
        for ext in (".png", ".pdf"):
            if not os.path.exists(f"{S.OUT}/{fid}{ext}"):
                bad.append(f"missing {fid}{ext}")
        p = f"{S.OUT}/{fid}.png"
        if os.path.exists(p):
            with open(p, "rb") as fh:
                data = fh.read(4096)
            w, h = struct.unpack(">II", data[16:24])
            i = data.find(b"pHYs")
            ppu_x = struct.unpack(">I", data[i + 4:i + 8])[0] if i > 0 else 0
            dpi = ppu_x * 0.0254
            w_cm, h_cm = w / dpi * 2.54, h / dpi * 2.54
            if abs(w_cm - WIDTH_CM) > 0.06:
                bad.append(f"{fid}: {w_cm:.2f} cm wide, the text block is {WIDTH_CM} cm")
            if dpi < MIN_DPI - 0.5:
                bad.append(f"{fid}: {dpi:.0f} dpi, minimum is {MIN_DPI:.0f}")
            if h_cm > 25.0:
                bad.append(f"{fid}: {h_cm:.1f} cm tall, taller than one page body")
    return bad


def check_fig1_strings(fig_texts):
    """Every number printed inside the workflow figure must be a number the manuscript prints."""
    md = open(V3, encoding="utf8").read()
    bad = []
    for s in fig_texts["Fig1_workflow"]:
        for tok in re.findall(r"\d+(?:\.\d+)?", s):
            if tok not in md:
                bad.append(f"{s!r} prints {tok}, which is not in the manuscript text")
    return bad


def main(corrupt: bool):
    saved, figs, texts = {}, {}, {}
    orig_save = S.save

    def save(fig, fid):
        problems = S.check_type(fig) + S.scan_strings(fig)
        boxes = [t.get_text() for t in S._fig_texts(fig) if t.get_text().strip()]
        # axis tick labels are a scale, not a claim, so they are kept out of the manuscript grep
        ticks = {t.get_text() for ax in fig.axes if ax.get_visible()
                 for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels())}
        figs[fid] = problems
        texts[fid] = [b for b in boxes if b not in ticks]
        return orig_save(fig, fid)

    S.save = save
    # the ledger on disk is a product, not an input: it is rebuilt from this run, so a value a
    # figure stopped drawing cannot survive in it and be compared as if it were current
    import os
    if os.path.exists(f"{p5_paths.RESULTS}/figure_drawn_values.json"):
        os.remove(f"{p5_paths.RESULTS}/figure_drawn_values.json")
    for mod in ("_mk_fig1", "_mk_fig2", "_mk_fig6", "_mk_fig7"):
        importlib.import_module(mod)
    S.save = orig_save

    exp = expected()
    bad = check_values(S.DRAWN, exp)
    for fid in FIGS:
        bad += [f"{fid}: {p}" for p in figs.get(fid, [])]
    bad += check_fig1_strings(texts)
    bad += check_files()

    if corrupt:
        key = ("Fig6_cinnamon_transfer", "a whitened/Coverage")
        for d in S.DRAWN["Fig6_cinnamon_transfer"]:
            if d["label"] == "a whitened/Coverage":
                d["value"] = round(d["value"] + 0.01, 3)
        bad = check_values(S.DRAWN, exp)
        caught = [b for b in bad if "a whitened/Coverage" in b]
        print(f"failure sample: 0.899 -> {0.899 + 0.01:.3f} on {key[1]} of {key[0]}")
        print(f"caught by the gate: {caught}")
        sys.exit(0 if caught else 1)

    print(f"cells compared: {len(exp)} across {len(FIGS)} figures")
    if bad:
        print(f"GATE FAILED ({len(bad)})")
        for b in bad:
            print("   -", b)
        sys.exit(1)
    print("gate passed: drawn values equal the tables, house style clean, files sized and resolved")


if __name__ == "__main__":
    main("--corrupt" in sys.argv)
