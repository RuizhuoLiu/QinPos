import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Design history: all three Viterbi cost functions, evaluated side by side

    Runs v1 (cramped-only), v2 (comfort band) and v3 (center pull,
    the live `qinpos.viterbi`) against the CURRENT
    `data/gq39_clean.csv` and reports string accuracy for each.

    Note: the numbers annotated inside the v1/v2 files (17.4% /
    23.2%) were measured on an EARLIER version of the cleaned CSV
    (before the period-split loader fix and the altered-tuning
    exclusions), so the numbers printed here will differ slightly —
    what matters and stays stable is the ORDERING and the size of
    the jumps between versions.
    """)
    return


@app.cell
def _():
    import csv
    import importlib.util
    import sys
    from collections import defaultdict
    from pathlib import Path

    from qinpos.theory import Note
    import qinpos.viterbi as v3

    ROOT = Path(__file__).resolve().parent.parent
    CLEAN_CSV = ROOT / "data/gq39_clean.csv"
    HISTORY = ROOT / "src/qinpos/OldVersions_scr" 

    def load_version(path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        # dataclasses resolve `from __future__ import annotations` string
        # annotations via sys.modules[module_name]; a manually-loaded
        # module must be registered there BEFORE exec, or @dataclass
        # raises AttributeError on NoneType.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    v1 = load_version(HISTORY / "viterbi_v1_cramped_only.py")
    v2 = load_version(HISTORY / "viterbi_v2_comfort_band.py")
    return CLEAN_CSV, Note, csv, defaultdict, v1, v2, v3


@app.cell
def _(CLEAN_CSV, csv, defaultdict):
    rows = [r for r in csv.DictReader(CLEAN_CSV.open())
            if r["status"] != "needs_review"]

    def notated_pitch(r) -> float:
        phys = float(r["physical_semitones"])
        resid = float(r["residual"])
        if r["status"] == "repaired":
            resid -= 12 if resid > 0 else -12
        return round(phys - resid, 3)

    by_piece = defaultdict(list)
    for _r in rows:
        by_piece[_r["piece"]].append(_r)
    for _rs in by_piece.values():
        _rs.sort(key=lambda r: int(r["idx"]))
    print(f"{len(rows)} usable stopped notes across {len(by_piece)} pieces")
    return by_piece, notated_pitch


@app.cell
def _(Note, by_piece, notated_pitch):
    def evaluate(mod):
        match = total = 0
        for rs in by_piece.values():
            notes = [Note(semitones=notated_pitch(r)) for r in rs]
            path = mod.decode(notes, mod.Weights(), kinds=["stopped"] * len(notes))
            match += sum(c.string == int(r["string"]) for c, r in zip(path, rs))
            total += len(rs)
        return match, total

    return (evaluate,)


@app.cell
def _(evaluate, v1, v2, v3):
    print(f"{'version':40s} {'accuracy':>12s}")
    print("-" * 54)
    for _name, _mod in [
        ("v1 cramped-only (one-sided wall)", v1),
        ("v2 comfort band [5,10] (flat interior)", v2),
        ("v3 center pull toward hui 8.5 (LIVE)", v3),
    ]:
        _m, _t = evaluate(_mod)
        print(f"{_name:40s} {_m:4d}/{_t} = {_m/_t:5.1%}")
    return


if __name__ == "__main__":
    app.run()
