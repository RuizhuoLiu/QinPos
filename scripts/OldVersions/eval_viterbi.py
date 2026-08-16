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
    # Viterbi string choice vs GQ39 experts
    """)
    return


@app.cell
def _():
    import csv
    import itertools
    from collections import Counter, defaultdict
    from pathlib import Path

    from qinpos.theory import Note, stopped_position
    from qinpos.viterbi import Weights, decode

    ROOT = Path(__file__).resolve().parent.parent
    CLEAN_CSV = ROOT / "data/gq39_clean.csv"
    return (
        CLEAN_CSV,
        Counter,
        Note,
        Weights,
        csv,
        decode,
        defaultdict,
        itertools,
        stopped_position,
    )


@app.cell
def _(Note):
    from qinpos.candidates import candidates_for
    _c = [c for c in candidates_for(Note(semitones=12)) if c.kind == "stopped"]
    print(sorted(round(c.position, 2) for c in _c))
    return


@app.cell
def _(CLEAN_CSV, by_piece, rows):
    import hashlib
    print(len(rows), sum(len(v) for v in by_piece.values()))
    print(hashlib.sha256(CLEAN_CSV.read_bytes()).hexdigest()[:12])
    return


@app.cell
def _(CLEAN_CSV, csv, defaultdict):
    rows = [r for r in csv.DictReader(CLEAN_CSV.open())
            if r["status"] != "needs_review"]

    def notated_pitch(r) -> float:
        """Notation-derived pitch: physical minus the (repair-corrected)
        residual. Integer-exact; does NOT leak the expert's position."""
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
    return by_piece, notated_pitch, rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baselines
    """)
    return


@app.cell
def _(Counter, by_piece, notated_pitch, rows, stopped_position):
    # Baseline 1: majority string
    _counts = Counter(int(r["string"]) for r in rows)
    _maj_string, _maj_n = _counts.most_common(1)[0]
    print(f"majority-string baseline (always string {_maj_string}): "
          f"{_maj_n}/{len(rows)} = {_maj_n/len(rows):.1%}")

    # Baseline 2: context-free nearest-to-hui-8.5
    _m = 0
    for _rs in by_piece.values():
        for _r in _rs:
            _p = notated_pitch(_r)
            _best_s, _best_d = None, 99.0
            for _s in range(1, 8):
                _pos = stopped_position(_s, _p)
                if _pos and 2.0 <= _pos <= 13.9:
                    _d = abs(_pos - 8.5)
                    if _d < _best_d:
                        _best_d, _best_s = _d, _s
            _m += (_best_s == int(_r["string"]))
    print(f"nearest-to-hui-8.5 baseline: {_m}/{len(rows)} = {_m/len(rows):.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Viterbi (Path A hand-crafted weights)
    """)
    return


@app.cell
def _(Note, by_piece, decode, notated_pitch):
    def evaluate(w):
        match = total = 0
        per_piece = {}
        for piece, rs in sorted(by_piece.items()):
            notes = [Note(semitones=notated_pitch(r)) for r in rs]
            path = decode(notes, w, kinds=["stopped"] * len(notes))
            m = sum(c.string == int(r["string"]) for c, r in zip(path, rs))
            per_piece[piece] = (m, len(rs))
            match += m
            total += len(rs)
        return match, total, per_piece

    return (evaluate,)


@app.cell
def _(Weights, evaluate):
    match, total, per_piece = evaluate(Weights())
    for _piece, (_m, _n) in sorted(per_piece.items()):
        print(f"{_piece:20s} {_m:3d}/{_n:3d} = {_m/_n:5.1%}")
    print(f"\nVITERBI (default weights): {match}/{total} = {match/total:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coarse weight grid

    A small grid over the three most influential weights. NOTE: this
    tunes on the full dataset (no train/test split), so treat the
    best number as an optimistic Path-A ceiling, not a final result —
    proper fitting with held-out pieces is exactly Phase 3's job.
    """)
    return


@app.cell
def _(Weights, evaluate, itertools):
    _best = (0, None)
    for _side, _tr, _cx in itertools.product(
            [0.4, 0.6, 1.0], [0.1, 0.3, 0.6], [0.05, 0.15, 0.3]):
        _w = Weights(below_center=_side, above_center=_side,
                     hand_travel=_tr, string_cross=_cx)
        _m, _t, _ = evaluate(_w)
        if _m > _best[0]:
            _best = (_m, (_side, _tr, _cx), _t)
    print(f"best on grid: {_best[0]}/{_best[2]} = {_best[0]/_best[2]:.1%}")
    print(f"  weights: side={_best[1][0]}, travel={_best[1][1]}, cross={_best[1][2]}")
    return


if __name__ == "__main__":
    app.run()
