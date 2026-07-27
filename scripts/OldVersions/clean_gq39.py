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

    3. After convention selection, remaining exact ±1-octave residuals are isolated range typos: repaired by adjusting range, and marked `repaired=True` so downstream users can exclude them if wanted.

    4. Anything still off by >0.6 semitones is flagged `outlier=True` and excluded from the clean set (candidates: chuo/zhu ornament pitches annotated at slide targets, or genuine typos — future work).

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

    TOL = 0.6  # semitones

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT/"data/GQ39/score_annotation"
    OUT = ROOT/"data/gq39_clean.csv"
    return (
        DATA,
        DEGREE_SEMITONES,
        OPEN_STRING_SEMITONES,
        OUT,
        TOL,
        csv,
        defaultdict,
        huifen_to_lambda,
        load_all,
        math,
    )


@app.cell
def _(OPEN_STRING_SEMITONES, huifen_to_lambda, math):
    def physical_semitones(e) -> float:
        lam = huifen_to_lambda(e.position)
        return OPEN_STRING_SEMITONES[e.string] + 12 * math.log2(1/lam)

    return (physical_semitones,)


@app.cell
def _(DEGREE_SEMITONES):
    def notated_semitones(degree: int, range_: int, convention: str) -> float:
        r = range_
        if convention == "at5" and degree in (5, 6, 7):
            r -= 1
        return DEGREE_SEMITONES[degree] + 12 * r

    return (notated_semitones,)


@app.cell
def _(notated_semitones):
    def fit_piece(pairs, convention, TOL=0.6):
        """Return (K, n_agree) for a piece under a convention."""
        offs = sorted(p - notated_semitones(e.degree, e.range_, convention)
                      for e, p in pairs)
        K = round(offs[len(offs) // 2])
        agree = sum(
            abs(p - notated_semitones(e.degree, e.range_, convention) - K) <= TOL
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
def _(TOL, by_piece, fit_piece, notated_semitones):
    rows = []
    stats = {"clean": 0, "repaired": 0, "outlier": 0}
    for piece, pairs in sorted(by_piece.items()):
        # pick the range convention that explains this piece best
        best = max(("at1", "at5"), key=lambda c: fit_piece(pairs, c)[1])
        K, _ = fit_piece(pairs, best)

        for _e, phys in pairs:
            notated = notated_semitones(_e.degree, _e.range_, best) + K
            resid = phys - notated
            repaired = False
            outlier = False
            corrected_range = _e.range_
            if abs(resid) <= TOL:
                pass  # clean
            elif abs(abs(resid) - 12) <= TOL:
                # isolated octave typo: repair the range digit
                corrected_range = _e.range_ + (1 if resid > 0 else -1)
                repaired = True
            else:
                outlier = True

            stats["clean" if not (repaired or outlier)
                  else ("repaired" if repaired else "outlier")] += 1
            rows.append({
                "piece": piece, "section": _e.section, "idx": _e.idx,
                "degree": _e.degree, "range_annotated": _e.range_,
                "range_corrected": corrected_range,
                "string": _e.string, "position": _e.position,
                "onset": _e.onset, "multi": _e.multi,
                "convention": best, "K": K,
                "physical_semitones": round(phys, 3),
                "residual": round(resid, 3),
                "repaired": repaired, "outlier": outlier,
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


if __name__ == "__main__":
    app.run()
