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
    # Validate the physics layer against GQ39 expert annotations

    Checks whether the string-length-ratio hui/position model (qinpos.theory) agrees with GQ39's note-by-note (degree, string, position) annotations under the
    plain "at1" range convention, with no per-piece gong offset (K)
    or convention correction applied yet?

    This deliberately does NOT do per-piece K/convention fitting or
    outlier repair — that logic (and the corresponding manual-review
    tables) lives in `clean_gq39.py`, so it's computed once, not
    duplicated here. This notebook only answers "is the physics layer
    itself sound", independent of the cleaning pipeline.
    """)
    return


@app.cell
def _():
    import math
    from collections import defaultdict
    from pathlib import Path

    from qinpos.dataset_gq39 import load_all
    from qinpos.theory import (DEGREE_SEMITONES, OPEN_STRING_SEMITONES,
                               huifen_to_lambda)

    return (
        DEGREE_SEMITONES,
        OPEN_STRING_SEMITONES,
        Path,
        defaultdict,
        huifen_to_lambda,
        load_all,
        math,
    )


@app.cell
def _():
    RESIDUAL_TOL = 0.6      # semitones: max deviation still counted as agree
    OFFSET_FRAC_TOL = 0.25  # semitones: how far a piece's raw offset may sit from an integer
    AGREE_RATE_TOL = 0.80   # min per-piece agreement rate before flagging
    SEMITONES_PER_OCTAVE = 12  # 12-TET
    return AGREE_RATE_TOL, OFFSET_FRAC_TOL, RESIDUAL_TOL, SEMITONES_PER_OCTAVE


@app.cell
def _(Path, load_all):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data/GQ39/score_annotation"
    print(data_dir, data_dir.exists())
    events = load_all(data_dir)
    print(f"loaded {len(events)} events")
    stopped = [e for e in events if e.kind == "stopped" and 1.0 <= e.position <= 13.9]
    print(f"{len(stopped)} stopped notes")
    return (stopped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Physics/notation helpers (convention fixed at "at1")
    """)
    return


@app.cell
def _(
    DEGREE_SEMITONES,
    OPEN_STRING_SEMITONES,
    SEMITONES_PER_OCTAVE,
    huifen_to_lambda,
    math,
):
    def physical_semitones(note) -> float:
        lam = huifen_to_lambda(note.position)
        return OPEN_STRING_SEMITONES[note.string] + SEMITONES_PER_OCTAVE * math.log2(1 / lam)

    def notated_semitones(note) -> float:
        return DEGREE_SEMITONES[note.degree] + SEMITONES_PER_OCTAVE * note.range_

    def raw_offset(note) -> float:
        return physical_semitones(note) - notated_semitones(note)

    return (raw_offset,)


@app.cell
def _(defaultdict, raw_offset, stopped):
    def group_offsets_by_piece(notes):
        by_piece = defaultdict(list)
        for note in notes:
            by_piece[note.piece].append(raw_offset(note))
        return by_piece

    piece_offsets = group_offsets_by_piece(stopped)
    return (piece_offsets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overall agreement (convention = "at1", no per-piece K yet)

    `frac`: how far the piece's raw median offset sits from the
    nearest integer.

    A large `frac`: this piece likely needs the "at5" convention or has a real modulation, since plain "at1" is failing to land on a clean gong offset.
    """)
    return


@app.cell
def _(AGREE_RATE_TOL, OFFSET_FRAC_TOL, RESIDUAL_TOL, piece_offsets):
    def report_agreement(by_piece, tol, offset_frac_tol, agree_rate_tol):
        agree = tot = 0
        suspects = []
        print(f"{'piece':20s} {'K':>4s} {'frac':>6s} {'agree':>6s}  flag")
        print("-" * 46)
        for piece, offs in sorted(by_piece.items()):
            offs = sorted(offs)
            med = offs[len(offs) // 2]
            K = round(med)
            frac = abs(med - K)
            ok = sum(abs(o - med) <= tol for o in offs)
            rate = ok / len(offs)
            agree += ok
            tot += len(offs)

            flag = ""
            if frac > offset_frac_tol:
                flag += "OFFSET_FAR "
            if rate < agree_rate_tol:
                flag += "LOW_AGREE "
            if flag:
                suspects.append((frac, rate, piece, K))

            print(f"{piece:20s} {K:4d} {frac:6.2f} {rate:6.1%}  {flag}")

        print(f"\nOVERALL: {agree}/{tot} = {agree/tot:.1%}")

        print("\n=== needs manual review (sorted by suspicion) ===")
        for frac, rate, piece, K in sorted(suspects, key=lambda s: (-s[0], s[1])):
            print(f"  {piece:20s} frac={frac:.2f} agree={rate:.1%} K={K}")

        return suspects

    suspects = report_agreement(piece_offsets, RESIDUAL_TOL, OFFSET_FRAC_TOL, AGREE_RATE_TOL)
    return


if __name__ == "__main__":
    app.run()
