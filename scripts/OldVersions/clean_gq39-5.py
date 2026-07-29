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

    3. **Altered physical tuning (紧五弦) is out of scope. But a handful of pieces retune a string's actual physical pitch (e.g. 紧五弦), which K/convention cannot correct, since K is a single constant per piece and can't shift one string differently from the rest. These pieces are excluded up front, before fitting, rather than left to fall out as per-note outliers — confirmed pieces: `EXCLUDED_TUNING_PIECES` below.

    4. **Loader-level ambiguity (`suspect`)**: some cells pack multiple notes as a comma-separated list, but Excel silently turned that comma into a decimal point on integer columns (`degree`, `range`, `string`), and/or the `position` cell arrived as a single fractional value even though the row holds >1 note. `dataset_gq39.py` reconstructs these, and when it has to *infer* a split for `position` from a physical necessity argument (two different degrees on the same string can't share a position) rather than reading an explicit comma, it marks that note `suspect=True`.

    5. **Alternate gong reading within one piece (`status="alt_gong"`)**: some pieces' score explicitly states a tuning-to-degree mapping that differs from the piece's dominant K for a handful of notes (e.g. 憶故人's header states "正調定弦 1 2 4 5 6 1 2" — gong on string 1 — while most of the piece's notes fit better with gong on string 3). For any note that fails the piece's primary K, we test the integer K implied by that note's own residual; if that alternate K resolves it within tolerance, we keep the note (it's real signal, not noise) but tag it `status="alt_gong"` with `K_used` recorded, distinct from the piece's primary `K`, so it's usable for training while remaining visible for a sanity check against the score.

    6. After convention/K resolution, remaining exact ±1-octave residuals are isolated range typos: repaired by adjusting range, and marked `status="repaired"` so downstream users can exclude them if wanted.

    7. Anything still off by >0.6 semitones (under both the primary and any alternate K) is flagged `status="outlier"` and excluded from the clean set (candidates: chuo/zhu ornament pitches annotated at slide targets, genuine typos, or an unmarked mid-piece 借调 not reflected in `section` — future work).

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

    # Confirmed altered-tuning (紧五弦) pieces: 借调 elsewhere in the dataset
    # is a relabeling only and is handled by K, so it stays in scope.
    EXCLUDED_TUNING_PIECES = {"yang01", "yang02", "yang03", "yu02", "yu03"}

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    OUT = ROOT / "data/gq39_clean.csv"
    return (
        CONVENTIONS,
        DATA,
        DEGREE_SEMITONES,
        EXCLUDED_TUNING_PIECES,
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
        return OPEN_STRING_SEMITONES[e.string] + SEMITONES_PER_OCTAVE * math.log2(1 / lam)

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
        """Return (K, n_agree) for a piece under a convention."""
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

    Altered-tuning pieces are dropped here, before any K/convention
    fitting happens, so they never influence a fit and never show up
    as false outliers.
    """)
    return


@app.cell
def _(DATA, EXCLUDED_TUNING_PIECES, defaultdict, load_all, physical_semitones):
    events = load_all(DATA)
    stopped = [_e for _e in events
               if _e.kind == "stopped" and 1.0 <= _e.position <= 13.9
               and _e.piece not in EXCLUDED_TUNING_PIECES]

    n_excluded = sum(
        1 for _e in events
        if _e.kind == "stopped" and _e.piece in EXCLUDED_TUNING_PIECES
    )
    print(f"excluded {n_excluded} stopped notes from altered-tuning pieces: "
          f"{sorted(EXCLUDED_TUNING_PIECES)}")
    print(f"{len(stopped)} stopped notes remain in scope")

    by_piece = defaultdict(list)
    for _e in stopped:
        by_piece[_e.piece].append((_e, physical_semitones(_e)))
    return by_piece, stopped


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
    stats = {"clean": 0, "repaired": 0, "alt_gong": 0, "outlier": 0}
    for piece, pairs in sorted(by_piece.items()):
        # pick the range convention and gong offset that explain this
        # piece's majority of notes best
        best = max(CONVENTIONS, key=lambda c: fit_piece(pairs, c)[1])
        K, _ = fit_piece(pairs, best)

        for _e, phys in pairs:
            notated = notated_semitones(_e.degree, _e.range_, best) + K
            resid = phys - notated
            corrected_range = _e.range_
            K_used = K
            alt_gong = False

            if abs(resid) <= RESIDUAL_TOL:
                status = "clean"
            elif abs(abs(resid) - SEMITONES_PER_OCTAVE) <= RESIDUAL_TOL:
                # isolated octave typo: repair the range digit
                corrected_range = _e.range_ + (1 if resid > 0 else -1)
                status = "repaired"
            else:
                # not explained by the piece's primary K/convention —
                # test whether a single alternate integer K (a
                # different gong-string reading used for just this
                # note) resolves it, before giving up as an outlier.
                alt_K = round(resid + K)
                alt_resid = phys - notated_semitones(_e.degree, _e.range_, best) - alt_K
                if alt_K != K and abs(alt_resid) <= RESIDUAL_TOL:
                    K_used = alt_K
                    resid = alt_resid
                    status = "alt_gong"
                    alt_gong = True
                else:
                    status = "outlier"

            stats[status] += 1
            rows.append({
                "piece": piece, "section": _e.section, "idx": _e.idx,
                "degree": _e.degree, "range_annotated": _e.range_,
                "range_corrected": corrected_range,
                "string": _e.string, "position": _e.position,
                "onset": _e.onset, "multi": _e.multi, "suspect": _e.suspect,
                "convention": best, "K": K, "K_used": K_used, "alt_gong": alt_gong,
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
    for k in ("clean", "repaired", "alt_gong", "outlier"):
        print(f"  {k:9s}: {stats[k]:4d}  ({stats[k]/n:.1%})")
    usable = stats["clean"] + stats["repaired"] + stats["alt_gong"]
    print(f"  usable for training: {usable}/{n} = {usable/n:.1%}")

    n_suspect = sum(1 for r in rows if r["suspect"])
    print(f"  suspect (position inferred from packed cell, worth spot-checking): "
          f"{n_suspect} ({n_suspect/n:.1%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: alt_gong notes (alternate gong reading within a piece)

    Kept and usable for training, but worth a quick sanity check: does
    `K_used` correspond to a plausible gong-string reading (compare
    against `-OPEN_STRING_SEMITONES[s]` for some string s), and does
    the score itself (header tuning notation, or a mode change) support
    a second reading at this point in the piece?
    """)
    return


@app.cell
def _(rows):
    def print_alt_gong_rows(all_rows):
        alt = [r for r in all_rows if r["status"] == "alt_gong"]
        print(f"{len(alt)} alt_gong notes\n")
        for r in sorted(alt, key=lambda r: (r["piece"], r["idx"])):
            print(f"  {r['piece']:20s} idx={r['idx']:4d} string={r['string']} "
                  f"position={r['position']:5.2f} degree={r['degree']} "
                  f"range={r['range_annotated']} K={r['K']:+d} -> "
                  f"K_used={r['K_used']:+d}  resid={r['residual']:+.2f}")
        return alt

    alt_gong_rows = print_alt_gong_rows(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: suspect notes (loader-inferred position split)

    These passed the residual check (mostly `status="clean"`), but their
    `position` was reconstructed by the loader from a packed cell rather
    than read directly — see point 4 above. Worth a first-pass spot
    check against the score, since the split relies on a physical
    argument (same string, different degree implies different position)
    rather than an explicit comma in the source file.
    """)
    return


@app.cell
def _(rows):
    def print_suspect_rows(all_rows):
        suspect = [r for r in all_rows if r["suspect"]]
        print(f"{len(suspect)} suspect notes to spot-check\n")
        for r in sorted(suspect, key=lambda r: (r["piece"], r["idx"])):
            print(f"  {r['piece']:20s} idx={r['idx']:4d} string={r['string']} "
                  f"position={r['position']:5.2f} degree={r['degree']} "
                  f"range={r['range_annotated']} status={r['status']:9s} "
                  f"resid={r['residual']:+.2f}")
        return suspect

    suspect_rows = print_suspect_rows(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: outlier notes

    These are excluded from the clean set. Worth checking against the
    original score for chuo/zhu slide ornaments (annotated at the
    slide target rather than the starting pitch), genuine typos, or an
    unmarked mid-piece 借调 that a single per-piece K can't absorb
    (compare `section` values across the flagged rows for a piece).
    """)
    return


@app.cell
def _(rows):
    def print_outlier_rows(all_rows):
        outliers = [r for r in all_rows if r["status"] == "outlier"]
        print(f"{len(outliers)} outlier notes to review\n")
        for r in sorted(outliers, key=lambda r: (r["piece"], r["idx"])):
            print(f"  {r['piece']:20s} idx={r['idx']:4d} "
                  f"string={r['string']} position={r['position']:5.2f} "
                  f"degree={r['degree']} range={r['range_annotated']} "
                  f"suspect={r['suspect']!s:5s} resid={r['residual']:+.2f}")
        return outliers

    outlier_rows = print_outlier_rows(rows)
    return (outlier_rows,)


@app.cell
def _(defaultdict, outlier_rows):
    def group_outliers_by_residual(rows, target_piece):
        groups = defaultdict(list)
        for r in rows:
            if r["piece"] == target_piece:
                key = round(r["residual"], 1)
                groups[key].append(r)

        for resid, group in sorted(groups.items()):
            example = group[0]
            print(f"resid={resid:+.2f}  n={len(group)}  "
                  f"e.g. string={example['string']} position={example['position']:5.2f} "
                  f"degree={example['degree']} range={example['range_annotated']}")
            idxs = [r["idx"] for r in group]
            print(f"    idx: {idxs}")

    group_outliers_by_residual(outlier_rows, "yi01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: same (piece, string, position) mapped to multiple degrees

    Re-run against the corrected loader's `stopped` list. Cases that
    were purely a loader parsing artifact (same string, degrees that
    are one scale-step apart, position now correctly split) should
    have dropped out already. What remains here is the set worth
    checking with a performer for genuine chuo/zhu/yin/nao ornaments.
    """)
    return


@app.cell
def _(defaultdict, stopped):
    def find_ornament_candidates(notes):
        by_position = defaultdict(set)
        for note in notes:
            key = (note.piece, note.string, round(note.position, 1))
            by_position[key].add(note.degree)

        print(f"{'piece':20s} {'string':6s} {'position':8s}  degrees")
        print("-" * 50)
        for (piece, string, pos), degrees in sorted(by_position.items()):
            if len(degrees) > 1:
                print(f"{piece:20s} {string:6d} {pos:8.2f}  {sorted(degrees)}")

    find_ornament_candidates(stopped)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: residual by degree, across the cleaned set

    Uses the already-corrected `residual` column (correct K/K_used and
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
            if r["status"] in ("clean", "repaired", "alt_gong"):
                by_degree[r["degree"]].append(r["residual"])

        for deg, resids in sorted(by_degree.items()):
            mean = sum(resids) / len(resids)
            print(f"  degree={deg}: mean={mean:+.3f}  n={len(resids)}")

    summarize_residual_by_degree(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: per-piece residual by degree, and by section

    Change `TARGET_PIECE` to inspect a different piece. The by-section
    breakdown matters for pieces like yi01/03/04/05/06 (借调, not
    altered tuning, so still in scope) — if the outlier rows above
    cluster in one `section`, that section likely needs its own K
    rather than sharing the whole piece's single fitted K.
    """)
    return


@app.cell
def _():
    TARGET_PIECE = "jiou01"  # edit to inspect a different piece
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
        by_section = defaultdict(list)
        for r in piece_rows:
            resid = r["residual"]
            by_degree[r["degree"]].append(resid)
            by_section[r["section"]].append(resid)
            if abs(resid) > tol:
                print(f"  section={r['section']} string={r['string']} "
                      f"position={r['position']:5.2f} degree={r['degree']} "
                      f"range={r['range_annotated']} resid={resid:+.2f} "
                      f"status={r['status']} suspect={r['suspect']}")

        print("--- summary by degree ---")
        for deg, resids in sorted(by_degree.items()):
            print(f"  degree={deg}: mean={sum(resids)/len(resids):+.2f}  n={len(resids)}")

        print("--- summary by section ---")
        for sec, resids in sorted(by_section.items(), key=lambda kv: str(kv[0])):
            print(f"  section={sec}: mean={sum(resids)/len(resids):+.2f}  n={len(resids)}")

    diagnose_piece(rows, TARGET_PIECE, RESIDUAL_TOL)
    return


if __name__ == "__main__":
    app.run()
