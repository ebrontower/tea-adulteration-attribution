"""Where this package finds its files - the only module that knows the repository layout.

No script in this package hard-codes a machine location. Everything is resolved from the
folder this file sits in, so the package runs from any checkout:

    <root>/code/             the scripts (this module among them)
    <root>/data/             expected location of the tea workbooks (not distributed)
    <root>/results/          every exported result JSON, table text and figure value record
    <root>/figures/          the seven delivered figure files (vector primary, raster backup)
    <root>/preregistrations/ the frozen criteria documents, shipped byte-identical

Two inputs are deliberately NOT redistributed, because their depositions are someone else's
to serve (see data/README.md for the layout and the DOIs):

    P5_PUBLIC_DATA    folder holding the public ATR-FTIR / MIR corpora used by Sections
                      3.9-3.11; defaults to <root>/data/public if you put them there
    P5_MANUSCRIPT_MD  a local copy of the manuscript text, needed only by the figure gate
                      (_fig_tables.py, _fig_gates.py), which re-reads the table cells a
                      figure was drawn from so a figure cannot drift away from its table

The tea workbooks can be pointed elsewhere with P5_TEA_DATA. Every accessor raises with the
sentence saying what to set, so a missing input fails loudly instead of quietly reading
somebody else's copy.
"""
from __future__ import annotations

import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
PREREG = os.path.join(ROOT, "preregistrations")

PUBLIC_DATA = os.environ.get("P5_PUBLIC_DATA") or os.path.join(DATA, "public")

_MISSING_PUBLIC = (
    "P5_PUBLIC_DATA is not set and %s does not exist. The public corpora are not "
    "redistributed with this package: download the depositions listed in data/README.md "
    "and either put them in <root>/data/public or export P5_PUBLIC_DATA." % PUBLIC_DATA
)


def public_data(*needed: str) -> str:
    """Directory of the downloaded public depositions (raises with instructions if absent)."""
    if not os.path.isdir(PUBLIC_DATA):
        raise SystemExit(_MISSING_PUBLIC)
    for name in needed:
        if not os.path.exists(os.path.join(PUBLIC_DATA, name)):
            raise SystemExit("%s is missing from %s - see data/README.md for the expected "
                             "layout of each deposition" % (name, PUBLIC_DATA))
    return PUBLIC_DATA


def manuscript_md() -> str:
    """Path of a local copy of the manuscript text (figure gate only)."""
    p = os.environ.get("P5_MANUSCRIPT_MD")
    if not p:
        raise SystemExit("P5_MANUSCRIPT_MD is not set. Only _fig_tables.py / _fig_gates.py "
                         "need it: they re-parse the manuscript tables that the figures were "
                         "drawn from. Point it at your own copy of the manuscript markdown; "
                         "the text itself is not redistributed in this repository while the "
                         "article is under review.")
    if not os.path.exists(p):
        raise SystemExit("P5_MANUSCRIPT_MD points at %s, which does not exist" % p)
    return p
