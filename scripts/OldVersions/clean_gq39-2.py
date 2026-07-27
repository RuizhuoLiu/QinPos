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
    # Clean the GQ39 stopped-note annotations into a reliable training set

    1. **Per-piece gong offset K**: each piece's jianpu "1" sits at a fixed integer semitone offset from open string 1.

       e.g. -7 = gong at open string 4's pitch class, -12 = gong at string 1's.

       K is estimated as the rounded median of (physical - notated) over the piece.

    2. **Range-boundary convention differs per piece**:
        - convention "at1" (original default): the range digit increments at degree 1 (strict octave arithmetic). (1-7)
        - convention "at5": degrees 5, 6, 7 are labelled with the range of the octave they LEAD INTO, i.e. their physical octave is (annotated range - 1). This matches how qin players group the scale as 5,6,1,2,3 (the tuning sequence) rather than 1-7.

       Empirically, all K=-7 pieces (guan/jiou/liang, and yi07's modulated section) use "at5"; the rest use "at1". We select per piece by best fit rather than hard-coding the correlation.

    (jiou04 agreement 56/86 under at1 -> 86/86 under at5; yi07_modulation prefers at5 while yi07_originalmode prefers at1 — same piece, same annotator, convention follows the gong reading.)

    3. After convention selection, remaining exact ±1-octave residuals are isolated range typos: repaired by adjusting range, and marked `status="repaired"` so downstream users can exclude them if wanted.

    4. Anything still off by >0.6 semitones is flagged `status="outlier"` and excluded from the clean set (candidates: chuo/zhu ornament pitches annotated at slide targets, or genuine typos — future work).

    Output: `data/gq39_clean.csv` with one row per stopped note.
    """)
    return


@app.cell
def _():
    import csv
    import math
    from collections import defaultdict
    from pathlib import Path

    from qinpos.dataset_gq39 import load_all
    from qinpos.theory import (DEGREE_SEMITONES, OPEN_STRING_SEMITONES,
                               huifen_to_lambda)

    RESIDUAL_TOL = 0.6         # semitones
    SEMITONES_PER_OCTAVE = 12
    CONVENTIONS = ("at1", "at5")

    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA = ROOT/"data/GQ39/score_annotation"
    OUT = ROOT/"data/gq39_clean.csv"
    return (
        CONVENTIONS,
        DATA,
        DEGREE_SEMITONES,
        OPEN_STRING_SEMITONES,
        OUT,
        RESIDUAL_TOL,
        SEMITONES_PER_OCTAVE,
        csv,
        defaultdict,
        huifen_to_lambda,
        load_all,
        math,
    )


@app.cell
def _(OPEN_STRING_SEMITONES, SEMITONES_PER_OCTAVE, huifen_to_lambda, math):
    def physical_semitones(e) -> float:
        lam = huifen_to_lambda(e.position)
        return OPEN_STRING_SEMITONES[e.string] + SEMITONES_PER_OCTAVE * math.log2(1/lam)

    return (physical_semitones,)


@app.cell
def _(DEGREE_SEMITONES, SEMITONES_PER_OCTAVE):
    def notated_semitones(degree: int, range_: int, convention: str) -> float:
        r = range_
        if convention == "at5" and degree in (5, 6, 7):
            r -= 1
        return DEGREE_SEMITONES[degree] + SEMITONES_PER_OCTAVE * r

    return (notated_semitones,)


@app.cell
def _(RESIDUAL_TOL, notated_semitones):
    def fit_piece(pairs, convention, tol=RESIDUAL_TOL):
        #Return (K, n_agree) for a piece under a convention.
        offs = sorted(p - notated_semitones(e.degree, e.range_, convention)
                      for e, p in pairs)
        K = round(offs[len(offs) // 2])
        agree = sum(
            abs(p - notated_semitones(e.degree, e.range_, convention) - K) <= tol
            for e, p in pairs
        )
        return K, agree

    return (fit_piece,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and classify every stopped note
    """)
    return


@app.cell
def _(DATA, defaultdict, load_all, physical_semitones):
    events = load_all(DATA)
    stopped = [_e for _e in events
               if _e.kind == "stopped" and 1.0 <= _e.position <= 13.9]

    by_piece = defaultdict(list)
    for _e in stopped:
        by_piece[_e.piece].append((_e, physical_semitones(_e)))
    return (by_piece,)


@app.cell
def _(
    CONVENTIONS,
    RESIDUAL_TOL,
    SEMITONES_PER_OCTAVE,
    by_piece,
    fit_piece,
    notated_semitones,
):
    rows = []
    stats = {"clean": 0, "repaired": 0, "outlier": 0}
    for piece, pairs in sorted(by_piece.items()):
        # pick the range convention that explains this piece best
        best = max(CONVENTIONS, key=lambda c: fit_piece(pairs, c)[1])
        K, _ = fit_piece(pairs, best)

        for _e, phys in pairs:
            notated = notated_semitones(_e.degree, _e.range_, best) + K
            resid = phys - notated
            corrected_range = _e.range_

            if abs(resid) <= RESIDUAL_TOL:
                status = "clean"
            elif abs(abs(resid) - SEMITONES_PER_OCTAVE) <= RESIDUAL_TOL:
                # isolated octave typo: repair the range digit
                corrected_range = _e.range_ + (1 if resid > 0 else -1)
                status = "repaired"
            else:
                status = "outlier"

            stats[status] += 1
            rows.append({
                "piece": piece, "section": _e.section, "idx": _e.idx,
                "degree": _e.degree, "range_annotated": _e.range_,
                "range_corrected": corrected_range,
                "string": _e.string, "position": _e.position,
                "onset": _e.onset, "multi": _e.multi,
                "convention": best, "K": K,
                "physical_semitones": round(phys, 3),
                "residual": round(resid, 3),
                "status": status,
            })
    return rows, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write the cleaned CSV
    """)
    return


@app.cell
def _(OUT, csv, rows, stats):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"wrote {OUT} ({n} stopped notes)")
    for k in ("clean", "repaired", "outlier"):
        print(f"  {k:9s}: {stats[k]:4d}  ({stats[k]/n:.1%})")
    usable = stats["clean"] + stats["repaired"]
    print(f"  usable for training: {usable}/{n} = {usable/n:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: outlier notes

    These are excluded from the clean set. Worth checking against the
    original score for chuo/zhu slide ornaments (annotated at the
    slide target rather than the starting pitch) versus genuine typos.
    """)
    return


@app.cell
def _(mo, outlier_rows, r):
    _df = mo.sql(
        f"""
        outlier_rows = [r for r in rows if r["status"] == "outlier"]
        print(f"{len(outlier_rows)} outlier notes to review\n")
        for r in sorted(outlier_rows, key=lambda r: (r["piece"], r["idx"])):
            print(f"  {r['piece']:20s} idx={r['idx']:4d} string={r['string']} "
                  f"position={r['position']:5.2f} degree={r['degree']} "
                  f"range={r['range_annotated']} resid={r['residual']:+.2f}")
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: residual by degree, across the cleaned set

    Uses the already-corrected `residual` column (correct K and
    best-fit convention baked in per piece), so a nonzero mean here
    for one degree across many pieces is a signal of a systematic
    physics-layer issue (e.g. `DEGREE_SEMITONES` or a just-intonation
    ratio in `theory.py`) — not per-piece K/convention noise.
    """)
    return


@app.cell
def _(defaultdict, rows):
    def summarize_residual_by_degree(rows):
        by_degree = defaultdict(list)
        for r in rows:
            by_degree[r["degree"]].append(r["residual"])

        for deg, resids in sorted(by_degree.items()):
            mean = sum(resids)/len(resids)
            print(f"  degree={deg}: mean={mean:+.3f}  n={len(resids)}")

    summarize_residual_by_degree(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: per-piece residual by degree
    """)
    return


@app.cell
def _():
    TARGET_PIECE = "yi01"  # edit to inspect the piece
    return (TARGET_PIECE,)


@app.cell
def _(RESIDUAL_TOL, TARGET_PIECE, defaultdict, rows):
    def diagnose_piece(all_rows, target, tol):
        piece_rows = [r for r in all_rows if r["piece"] == target]
        if not piece_rows:
            print(f"no rows found for piece={target!r}")
            return

        convention = piece_rows[0]["convention"]
        K = piece_rows[0]["K"]
        print(f"piece={target}  convention={convention}  K={K}")

        by_degree = defaultdict(list)
        for r in piece_rows:
            resid = r["residual"]
            by_degree[r["degree"]].append(resid)
            if abs(resid) > tol:
                print(f"  string={r['string']} position={r['position']:5.2f} "
                      f"degree={r['degree']} range={r['range_annotated']} "
                      f"resid={resid:+.2f} status={r['status']}")

        print("--- summary by degree ---")
        for deg, resids in sorted(by_degree.items()):
            print(f"  degree={deg}: mean={sum(resids)/len(resids):+.2f}  n={len(resids)}")

    diagnose_piece(rows, TARGET_PIECE, RESIDUAL_TOL)
    return


if __name__ == "__main__":
    app.run()
