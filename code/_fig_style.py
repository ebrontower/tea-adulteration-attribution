"""Shared house style for the four new CEA figures (paper 5).

Everything here is fixed by preregistrations/figures_criteria_CEA_v1.md §0: 16.0 cm text width, near-black text only,
the palette of the author's own three originals, 300 dpi PNG plus vector PDF, and a recorded list
of the exact values drawn so the gate can compare them against the manuscript tables.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import p5_paths  # noqa: E402

OUT = p5_paths.FIGURES
os.makedirs(OUT, exist_ok=True)

CM = 1 / 2.54
TEXT_W = 16.0                      # cm, A4 single column with 2.5 cm margins (Manuscript_CEA_base)
INK = "#1A1A1A"                    # every piece of text, no exceptions
RAW = "#B2182B"                    # the author's Fig.1 palette: raw absorbance
WHITE = "#2166AC"                  # … and the whitened metric
ACCENT = "#0B5563"                 # third series only (teal, same family)
RULE = "#9A9A9A"                   # reference lines and rules may be grey; text may not

MIN_PT = 7.0                       # smallest type, in points, at 16.0 cm print width

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,             # editable text in the vector PDF, fonts embedded
    "svg.fonttype": "none",
})

DRAWN = {}                          # figure id -> list of {label, value} actually plotted


def note(fid: str, label: str, value):
    """Record a value that the figure plots, so the gate can check it against the table."""
    DRAWN.setdefault(fid, []).append({"label": label, "value": value})
    return value


def newfig(width_cm: float, height_cm: float):
    return plt.subplots(figsize=(width_cm * CM, height_cm * CM))


def barefig(width_cm: float, height_cm: float):
    """Figure without axes, for scripts that lay their own GridSpec columns."""
    return plt.figure(figsize=(width_cm * CM, height_cm * CM))


def save(fig, fid: str):
    fig.savefig(f"{OUT}/{fid}.png", dpi=300)
    fig.savefig(f"{OUT}/{fid}.pdf")
    plt.close(fig)
    return f"{OUT}/{fid}"


def dump_drawn(path: str | None = None):
    p = path or f"{p5_paths.RESULTS}/figure_drawn_values.json"
    if os.path.exists(p):                                   # figures are built one script at a time
        prev = json.load(open(p, encoding="utf8"))
        key = lambda d: (d["label"], json.dumps(d["value"], sort_keys=True))  # noqa: E731
        for k, v in prev.items():
            DRAWN.setdefault(k, [])
            have = {key(d) for d in DRAWN[k]}
            DRAWN[k] += [d for d in v if key(d) not in have]
    json.dump(DRAWN, open(p, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    return p, {k: len(v) for k, v in DRAWN.items()}


def _fig_texts(fig):
    """Every text artist the figure actually draws, axes labels and legend labels included."""
    artists = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        artists += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.xaxis.get_visible():
            artists += ax.get_xticklabels()
        if ax.yaxis.get_visible():
            artists += ax.get_yticklabels()
    for leg in fig.legends:
        artists += list(leg.get_texts())
    return [t for t in artists if t.get_visible()]


def check_type(fig):
    """Assert no text below MIN_PT and that no two text boxes overlap. Returns the messages."""
    problems = []
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for t in _fig_texts(fig):
        s = t.get_text()
        if not s.strip():
            continue
        if t.get_fontsize() < MIN_PT - 1e-9:
            problems.append(f"under {MIN_PT} pt: {t.get_fontsize()} pt on {s[:32]!r}")
        if not (t.get_color() == INK or t.get_color() == "black"):
            problems.append(f"non-black text: {s[:24]!r} in {t.get_color()}")
        boxes.append((s, t.get_window_extent(renderer=rend)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ov = a.intersection(a, b)
            if ov is not None and ov.width > 1 and ov.height > 1:
                problems.append(f"overlap: {boxes[i][0][:24]!r} / {boxes[j][0][:24]!r}")
    fb = fig.bbox
    for s, bx in boxes:
        if bx.x0 < fb.x0 - 0.5 or bx.x1 > fb.x1 + 0.5:
            problems.append(f"runs off the canvas: {s[:24]!r} x=({bx.x0:.1f},{bx.x1:.1f}) vs ({fb.x0:.1f},{fb.x1:.1f})")
    return problems


def scan_strings(fig):
    """Cross-reference / prose guards required by the criteria file."""
    import re
    bad = []
    for t in _fig_texts(fig):
        s = t.get_text()
        if re.search(r"(Fig|Table)\s*\.?\s*\d", s):
            bad.append(s)
        if re.search(r"\b(we|our)\b", s, flags=re.I):
            bad.append(s)
    return bad
