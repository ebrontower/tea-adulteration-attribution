"""Parse the four v3 tables that the new CEA figures are drawn from.

Single source of truth = manuscript.md itself, so a figure cannot disagree with its table.
Returns plain dicts of floats (and None for the em-dash / 'absent' cells, which must be left
out of the drawing rather than plotted as zero).
"""
from __future__ import annotations

import re

import numpy as np  # noqa: F401  (used by the figure scripts through these dicts)
import p5_paths  # noqa: E402

V3 = p5_paths.manuscript_md()
DASH = ("—", "–", "-", "absent")


def _lines(md: str, key: str):
    """Return the block-quoted markdown table that follows the '**Table key.**' caption.
    Captions wrap over several lines, so the scan skips forward to the first pipe row."""
    m = re.search(r"^\*\*Table " + re.escape(key) + r"\.\*\*", md, flags=re.M)
    if not m:
        raise SystemExit(f"Table {key} not found in v3")
    start = md.index("\n|", m.end()) + 1            # first line of the grid
    rows, i = [], start - 1
    while True:
        j = md.find("\n", i + 1)
        ln = md[i + 1:j] if j > i else md[i + 1:]
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if not set(cells[0]) <= set("-— "):          # skip the |---|---| rule row
            rows.append(cells)
        if j < 0:
            break
        i = j
    if not rows:
        raise SystemExit(f"Table {key}: no grid rows found")
    return rows[1:], rows[0]                        # data rows, header


def num(cell: str):
    """Float of the first number in a cell; None when the cell is a dash or 'absent'."""
    c = cell.strip()
    if c in DASH:
        return None
    m = re.search(r"[+−-]?\d+(?:\.\d+)?", c.replace(",", ""))
    if not m:
        return None
    return float(m.group(0).replace("−", "-"))


def table(key: str) -> dict:
    rows, head = _lines(open(V3, encoding="utf8").read(), key)
    return {"head": head, "rows": rows}


def S4() -> dict:
    """Discrimination: model class -> (tea raw, tea white, tea d, cin raw, cin white, cin d)."""
    t = table("S4")
    out = {}
    for r in t["rows"]:
        out[r[0]] = [num(c) for c in r[1:7]]
    return out


def S5() -> dict:
    """Refusal ladder: rule -> (rung, AUROC orphan, AUROC composite/singles, AUROC composite/published,
    refusal at tau, false abstention, refusal at 10% false abstention)."""
    t = table("S5")
    out = {}
    for r in t["rows"]:
        rung = re.search(r"\((\d+|oracle)\)", r[0])
        out[r[0]] = {"rung": rung.group(1) if rung else None,
                     "orphan": num(r[1]), "comp_singles": num(r[2]), "comp_pub": num(r[3]),
                     "refusal_tau": num(r[4]), "false_abst": num(r[5]), "refusal_10": num(r[6]),
                     "absent": r[1].strip() == "absent"}
    return out


def T5a() -> dict:
    """Cinnamon transfer, standard/normalised score rows."""
    t = table("5a")
    out = {}
    for r in t["rows"]:
        out[f"{r[0]} | {r[1]}"] = {"coverage": num(r[2]), "size": num(r[3]), "top1": num(r[4]),
                                   "empty": num(r[5]), "accept": num(r[6]), "accepted": r[7]}
    return out


def T5b() -> dict:
    """Group conditional: group -> raw/white pairs for coverage, size, top-1, acceptance."""
    t = table("5b")
    out = {}
    for r in t["rows"]:
        def pair(c):
            parts = re.split(r"\s*/\s*", c)
            return [num(p) for p in parts]
        out[r[0]] = {"n": num(r[1]), "coverage": pair(r[2]), "size": pair(r[3]),
                     "top1": pair(r[4]), "accept": pair(r[5])}
    return out


def bootstrap_intervals() -> dict:
    """The five block-cluster bootstrap intervals quoted in Table 5a's note — taken verbatim,
    never recomputed."""
    md = open(V3, encoding="utf8").read()
    m = re.search(r"places 95% intervals of ([\d.–-]+) on\s*\n?coverage, ([\d.–-]+) on set size, "
                  r"([\d.–-]+) on top-1 and ([\d.–-]+) on the empty-set rate for the\s*\n?"
                  r"whitened standard-score row, and ([\d.–-]+) on top-1 for the two euclidean rows", md)
    if not m:
        raise SystemExit("the bootstrap intervals are not in Table 5a's note as expected")
    def two(s):
        a, b = re.findall(r"\d*\.?\d+", s)
        return float(a), float(b)
    return {"coverage": two(m.group(1)), "size": two(m.group(2)),
            "top1": two(m.group(3)), "empty": two(m.group(4)),
            "top1_euclidean": two(m.group(5))}


def prose_numbers() -> dict:
    """The two Fig. 6 caveats that are stated in Section 3.10 prose rather than in a table cell.

    Read from the manuscript sentence itself, with the sentence pinned, so the figure cannot
    introduce a number the text does not print.
    """
    md = open(V3, encoding="utf8").read()
    pats = {
        "wood dust empty sets": r"and (\d+(?:\.\d+)?)% of its spectra are returned as empty sets",
        "authentic only among accepted": r"the first candidate is authentic only (\d+(?:\.\d+)?)% of the time",
    }
    out = {}
    for k, p in pats.items():
        m = re.search(p, md)
        if not m:
            raise SystemExit(f"prose number moved or vanished: {k!r} ({p})")
        out[k] = float(m.group(1)) / 100.0
    return out


if __name__ == "__main__":
    s4, s5 = S4(), S5()
    print(f"S4 rows {len(s4)}: {list(s4)[:3]} …")
    for k, v in list(s4.items()):
        print(f"  {k[:46]:<48} {v}")
    print(f"\nS5 rows {len(s5)}")
    for k, v in s5.items():
        print(f"  {k[:44]:<46} {v}")
    print(f"\n5a rows {len(T5a())}")
    for k, v in T5a().items():
        print(f"  {k:<44} {v}")
    print(f"\n5b rows {len(T5b())}")
    for k, v in T5b().items():
        print(f"  {k:<24} {v}")
    print("\nbootstrap:", bootstrap_intervals())
