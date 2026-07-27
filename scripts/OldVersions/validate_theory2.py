import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Validate the physics layer against GQ39 expert annotations

    Checks whether the string-length-ratio hui/position model (`qinpos.theory`) agrees with GQ39's note-by-note (degree, string,
    position) annotations, after accounting for each piece's gong offset K and range-boundary convention.
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
    OFFSET_FRAC_TOL = 0.25  # semitones: how far a piece's K may sit from an integer
    AGREE_RATE_TOL = 0.80   # min per-piece agreement rate before flagging
    SEMITONES_PER_OCTAVE = 12 # 12-TET
    CONVENTIONS = ("at1", "at5")
    return (
        AGREE_RATE_TOL,
        CONVENTIONS,
        OFFSET_FRAC_TOL,
        RESIDUAL_TOL,
        SEMITONES_PER_OCTAVE,
    )


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
    ## Overall agreement (convention = "at1")
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

    def notated_semitones(note, convention: str = "at1") -> float:
        r = note.range_ # note 1-7
        if convention == "at5" and note.degree in (5, 6, 7):
            r -= 1
        return DEGREE_SEMITONES[note.degree] + SEMITONES_PER_OCTAVE * r

    def raw_offset(note, convention: str = "at1") -> float:
        return physical_semitones(note) - notated_semitones(note, convention)

    return notated_semitones, physical_semitones, raw_offset


@app.cell
def _(defaultdict, raw_offset, stopped):
    # organize by piece
    def group_offsets_by_piece(notes):
        by_piece = defaultdict(list)
        for note in notes:
            by_piece[note.piece].append(raw_offset(note))
        return by_piece

    piece_offsets = group_offsets_by_piece(stopped)
    return (piece_offsets,)


@app.cell
def _(RESIDUAL_TOL, piece_offsets):
    def report_agreement(by_piece, tol):
        agree = tot = 0
        for piece, offs in sorted(by_piece.items()):
            offs = sorted(offs)
            med = offs[len(offs) // 2]
            ok = sum(abs(o - med) <= tol for o in offs)
            agree += ok
            tot += len(offs)
            print(f"{piece:20s} n={len(offs):3d}  offset={med:7.2f}  agree={ok/len(offs):5.1%}")
        print(f"\nOVERALL: {agree}/{tot} = {agree/tot:.1%}")

    report_agreement(piece_offsets, RESIDUAL_TOL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-degree residual breakdown for one piece
    """)
    return


@app.cell
def _():
    TARGET_PIECE = "jiou01"  # edit to inspect a different piece
    return (TARGET_PIECE,)


@app.cell
def _(RESIDUAL_TOL, TARGET_PIECE, defaultdict, raw_offset, stopped):
    def diagnose_piece(notes, target, tol):
        pairs = sorted(
            ((note, raw_offset(note)) for note in notes if note.piece == target),
            key=lambda p: p[1],
        )
        med = pairs[len(pairs) // 2][1]

        by_degree = defaultdict(list)
        for note, off in pairs:
            resid = off - med
            by_degree[note.degree].append(resid)
            if abs(resid) > tol:
                print(f"  string={note.string} position={note.position:5.2f} "
                      f"degree={note.degree} resid={resid:+.2f}")

        print("--- summary by degree ---")
        for deg, resids in sorted(by_degree.items()):
            print(f"  degree={deg}: mean={sum(resids)/len(resids):+.2f}  n={len(resids)}")

    diagnose_piece(stopped, TARGET_PIECE, RESIDUAL_TOL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convention + offset diagnostic across all pieces
    """)
    return


@app.cell
def _(defaultdict, physical_semitones, stopped):
    def group_pairs_by_piece(notes):
        by_piece = defaultdict(list)
        for note in notes:
            by_piece[note.piece].append((note, physical_semitones(note)))
        return by_piece

    piece_pairs = group_pairs_by_piece(stopped)
    return (piece_pairs,)


@app.cell
def _(RESIDUAL_TOL, notated_semitones):
    def fit_piece(pairs, convention, tol=RESIDUAL_TOL):
        offs = sorted(phys - notated_semitones(note, convention) for note, phys in pairs)
        K = round(offs[len(offs) // 2])
        agree = sum(
            abs(phys - notated_semitones(note, convention) - K) <= tol
            for note, phys in pairs
        )
        return K, agree

    return (fit_piece,)


@app.cell
def _(
    AGREE_RATE_TOL,
    CONVENTIONS,
    OFFSET_FRAC_TOL,
    fit_piece,
    notated_semitones,
    piece_pairs,
):
    def convention_diagnostic(by_piece):
        print(f"{'piece':20s} {'conv':4s} {'K':>4s} {'frac':>6s} {'agree':>6s}  flag")
        print("-" * 52)
        suspects = []
        for piece, pairs in sorted(by_piece.items()):
            candidates = {c: fit_piece(pairs, c) for c in CONVENTIONS}
            best = max(candidates, key=lambda c: candidates[c][1])
            K, agree = candidates[best]

            offs = sorted(phys - notated_semitones(note, best) for note, phys in pairs)
            med = offs[len(offs) // 2]
            frac = abs(med - round(med))
            rate = agree / len(pairs)

            flag = ""
            if frac > OFFSET_FRAC_TOL:
                flag += "OFFSET_FAR "
            if rate < AGREE_RATE_TOL:
                flag += "LOW_AGREE "
            if flag:
                suspects.append((frac, rate, piece, best, K))
            print(f"{piece:20s} {best:4s} {K:4d} {frac:6.2f} {rate:6.1%}  {flag}")

        print("\n=== needs manual review (sorted by suspicion) ===")
        for frac, rate, piece, best, K in sorted(suspects, key=lambda s: (-s[0], s[1])):
            print(f"  {piece:20s} frac={frac:.2f} agree={rate:.1%} conv={best} K={K}")

    convention_diagnostic(piece_pairs)
    return


if __name__ == "__main__":
    app.run()
