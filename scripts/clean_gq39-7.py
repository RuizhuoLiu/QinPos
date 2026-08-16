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
        - convention "at1" (original default): the range digit increments at degree 1 (octave: 1-7)
        - convention "at5": degrees 5, 6, 7 are labelled with the range of the octave they LEAD INTO, i.e. their physical octave is annotated range - 1. This matches how qin players group the scale as 5,6,1,2,3 rather than 1-7.

       Empirically, all K=-7 pieces (guan/jiou/liang, and yi07's modulated section) use "at5"; the rest use "at1". Each piece is selected by best fit rather than hard-coding the correlation.

    (jiou04 agreement 56/86 under at1 -> 86/86 under at5; yi07_modulation prefers at5 while yi07_originalmode prefers at1.)

    3. **Degree 4 has two theory-recognized readings (`degree4_reading`)**: 清角 (qingjiao, 5 semitones above gong, the modern/清乐 reading, default) vs 変徵 (bianzhi, 6 semitones, the older 雅乐 reading). Both are legitimate, well-documented readings of the same scale step in different qin traditions, unlike an arbitrary semitone offset on any other degree. I fit this per piece exactly like `convention`, picking whichever reading the piece's degree-4 notes agree with best, rather than searching for an arbitrary correcting K per note.

    4. **Altered physical tuning (紧五弦) is out of scope, not a K/convention case**: K can't work when a piece retunes a single string's actual physical pitch (e.g. 紧五弦), since K is a single constant per piece and can't shift one string differently from the rest. These pieces are excluded up front, before fitting, rather than left to fall out as per-note outliers: `EXCLUDED_TUNING_PIECES`.

    5. **Loader-level ambiguity (`suspect`)**: some cells pack multiple notes as a comma-separated list, but Excel silently turned that comma into a decimal point on integer columns (`degree`, `range`, `string`), and/or the `position` cell arrived as a single fractional value even though the row holds >1 note. `dataset_gq39.py` reconstructs these, and when it has to *infer* a split for `position` from the rule that:two different degrees on the same string can't share a position, rather than reading an explicit comma, it marks that note `suspect=True`.

    6. **Confirmed alternate gong-string reading (`status="alt_gong"`)**: for a note that fails the piece's primary K (even after trying both convention and degree4_reading candidates), we test the integer K implied by that note's own residual. We only accept this if that K actually corresponds to some string's open pitch class (`gong_string_match` non-empty).

    7. **Anything still unexplained (`status="needs_review"`)**: notes that don't fit the piece's primary reading, aren't a degree-4 清角/変徵 case, and whose implied K doesn't match any real string are NOT auto-accepted into the clean set. There is no music-theory justification for treating an arbitrary semitone offset on degrees 1/2/3/5/6/7 as legitimate the way there is for degree 4 or a real gong-string shift — so these are surfaced for direct inspection against the score, and excluded from "usable for training" until confirmed.

    8. Remaining exact ±1-octave residuals are isolated range typos: repaired by adjusting range, and marked `status = "repaired"`.

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

    # 清角 (qingjiao, modern/清乐 reading, the default already in DEGREE_SEMITONES) vs 変徵 (bianzhi, older 雅乐 reading), the one scale step with a theory-recognized alternate semitone value.
    DEGREE4_READINGS = (5, 6)

    # Confirmed altered-tuning (紧五弦) pieces: 借调 elsewhere in the dataset is a relabeling only and is handled by K, so it stays in scope.
    EXCLUDED_TUNING_PIECES = {"yang01", "yang02", "yang03", "yu02", "yu03"}

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    OUT = ROOT / "data/gq39_clean.csv"
    return (
        CONVENTIONS,
        DATA,
        DEGREE4_READINGS,
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
    def notated_semitones(degree: int, range_: int, convention: str,
                           degree4_semitones: int) -> float:
        r = range_
        if convention == "at5" and degree in (5, 6, 7):
            r -= 1
        base = degree4_semitones if degree == 4 else DEGREE_SEMITONES[degree]
        return base + SEMITONES_PER_OCTAVE * r

    return (notated_semitones,)


@app.cell
def _(OPEN_STRING_SEMITONES):
    def gong_strings_for_K(K: int) -> list[int]:
        #Which open strings (if any) this K would place the gong on.
        return [s for s, v in OPEN_STRING_SEMITONES.items() if (v - K) % 12 == 0]

    return (gong_strings_for_K,)


@app.cell
def _(RESIDUAL_TOL, notated_semitones):
    def fit_piece(pairs, convention, degree4_semitones, tol=RESIDUAL_TOL):
        # Return (K, n_agree) for a piece under a (convention, degree4_semitones) pair.
        offs = sorted(p - notated_semitones(e.degree, e.range_, convention, degree4_semitones)
                      for e, p in pairs)
        K = round(offs[len(offs) // 2])
        agree = sum(
            abs(p - notated_semitones(e.degree, e.range_, convention, degree4_semitones) - K) <= tol
            for e, p in pairs
        )
        return K, agree

    return (fit_piece,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and classify every stopped note

    Altered-tuning pieces are dropped here, before any fitting happens,
    so they never influence a fit and never show up as false outliers.
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
    return (by_piece,)


@app.cell
def _(
    CONVENTIONS,
    DEGREE4_READINGS,
    RESIDUAL_TOL,
    SEMITONES_PER_OCTAVE,
    by_piece,
    fit_piece,
    gong_strings_for_K,
    notated_semitones,
):
    rows = []
    stats = {"clean": 0, "repaired": 0, "alt_gong": 0, "needs_review": 0}
    for piece, pairs in sorted(by_piece.items()):
        # pick the (convention, degree4 reading) combo that explains this piece's majority of notes best
        candidates = {
            (c, d4): fit_piece(pairs, c, d4)
            for c in CONVENTIONS for d4 in DEGREE4_READINGS
        }
        best_combo = max(candidates, key=lambda combo: candidates[combo][1])
        best, degree4_semitones = best_combo
        K, _ = candidates[best_combo]

        for _e, phys in pairs:
            notated = notated_semitones(_e.degree, _e.range_, best, degree4_semitones) + K
            resid = phys - notated
            corrected_range = _e.range_
            K_used = K
            alt_gong = False
            gong_match: list[int] = []

            if abs(resid) <= RESIDUAL_TOL:
                status = "clean"
            elif abs(abs(resid) - SEMITONES_PER_OCTAVE) <= RESIDUAL_TOL:
                # isolated octave typo: repair the range digit
                corrected_range = _e.range_ + (1 if resid > 0 else -1)
                status = "repaired"
            else:
                # Only accept an alternate K if it's a REAL open-string pitch class, no blind "search any integer" fallback.
                alt_K = round(resid + K)
                alt_resid = phys - notated + K - alt_K  # recompute with alt_K instead of K
                gm = gong_strings_for_K(alt_K)
                if alt_K != K and gm and abs(alt_resid) <= RESIDUAL_TOL:
                    K_used = alt_K
                    resid = alt_resid
                    status = "alt_gong"
                    alt_gong = True
                    gong_match = gm
                else:
                    status = "needs_review"

            stats[status] += 1
            rows.append({
                "piece": piece, "section": _e.section, "idx": _e.idx,
                "degree": _e.degree, "range_annotated": _e.range_,
                "range_corrected": corrected_range,
                "string": _e.string, "position": _e.position,
                "onset": _e.onset, "multi": _e.multi, "suspect": _e.suspect,
                "convention": best, "degree4_semitones": degree4_semitones,
                "K": K, "K_used": K_used, "alt_gong": alt_gong,
                "gong_string_match": ",".join(map(str, gong_match)),
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
    for k in ("clean", "repaired", "alt_gong", "needs_review"):
        print(f"  {k:12s}: {stats[k]:4d}  ({stats[k]/n:.1%})")
    usable = stats["clean"] + stats["repaired"] + stats["alt_gong"]
    print(f"  usable for training: {usable}/{n} = {usable/n:.1%}")

    n_suspect = sum(1 for r in rows if r["suspect"])
    print(f"  suspect (position inferred from packed cell, worth spot-checking): "
          f"{n_suspect} ({n_suspect/n:.1%})")

    n_bianzhi = sum(1 for r in rows if r["degree4_semitones"] == 6)
    if n_bianzhi:
        pieces_bianzhi = sorted({r["piece"] for r in rows if r["degree4_semitones"] == 6})
        print(f"  pieces using 変徵 (6 semitones) for degree 4: {pieces_bianzhi}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: needs_review notes (NOT counted as usable for training)

    These don't fit the piece's primary (convention, degree4_reading, K),
    aren't a confirmed real gong-string shift. need to be checked
    """)
    return


@app.cell
def _(rows):
    def print_needs_review_rows(all_rows):
        review = [r for r in all_rows if r["status"] == "needs_review"]
        print(f"{len(review)} needs_review notes\n")
        for r in sorted(review, key=lambda r: (r["piece"], r["idx"])):
            print(f"  {r['piece']:20s} idx={r['idx']:4d} string={r['string']} "
                  f"position={r['position']:5.2f} degree={r['degree']} "
                  f"range={r['range_annotated']} K={r['K']:+d}  "
                  f"resid={r['residual']:+.2f}")
        return review

    needs_review_rows = print_needs_review_rows(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: alt_gong notes (confirmed alternate gong-string reading)

    Every note here has `K_used` matching some string's real open pitch
    class — a genuine retuning/借调 reading, not a numerical coincidence.
    """)
    return


@app.cell
def _(rows):
    def print_alt_gong_rows(all_rows):
        alt = [r for r in all_rows if r["status"] == "alt_gong"]
        print(f"{len(alt)} alt_gong notes (all confirmed against a real open string)\n")
        for r in sorted(alt, key=lambda r: (r["piece"], r["idx"])):
            print(f"  {r['piece']:20s} idx={r['idx']:4d} string={r['string']} "
                  f"position={r['position']:5.2f} degree={r['degree']} "
                  f"range={r['range_annotated']} K={r['K']:+d} -> "
                  f"K_used={r['K_used']:+d} (gong on string {r['gong_string_match']})  "
                  f"resid={r['residual']:+.2f}")
        return alt

    alt_gong_rows = print_alt_gong_rows(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: suspect notes (loader-inferred position split)

    These passed the residual check (mostly `status="clean"`), but their
    `position` was reconstructed by the loader from a packed cell rather
    than read directly. Worth a first-pass spot check against the score.
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
                  f"range={r['range_annotated']} status={r['status']:12s} "
                  f"resid={r['residual']:+.2f}")
        return suspect

    suspect_rows = print_suspect_rows(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: same (piece, string, position) mapped to multiple degrees

    Cross-checked against the classification above. An entry fully
    explained by (clean + alt_gong) or (clean + degree4 reading) isn't
    a real ambiguity — it's the same fingering reused under two
    legitimate readings elsewhere in the piece. What's left after that
    is worth checking for genuine chuo/zhu/yin/nao ornaments.
    """)
    return


@app.cell
def _(defaultdict, rows):
    def find_ornament_candidates(all_rows):
        by_position = defaultdict(list)
        for r in all_rows:
            key = (r["piece"], r["string"], round(r["position"], 1))
            by_position[key].append(r)

        print(f"{'piece':20s} {'string':6s} {'position':8s}  degree:status pairs")
        print("-" * 60)
        for (piece, string, pos), group in sorted(by_position.items()):
            degrees = {r["degree"] for r in group}
            if len(degrees) <= 1:
                continue
            unresolved = [r for r in group if r["status"] == "needs_review"]
            if not unresolved:
                continue  # fully explained by clean/alt_gong/degree4 reading
            tags = ", ".join(f"{r['degree']}:{r['status']}" for r in group)
            print(f"{piece:20s} {string:6d} {pos:8.2f}  {tags}")

    find_ornament_candidates(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual review: residual by degree, across the cleaned set

    Only `clean`/`repaired`/`alt_gong` rows (already correctly
    explained) are included — a nonzero mean here for one degree
    across many pieces would be a signal of a systematic
    physics-layer issue, not per-piece K/convention noise.
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

    Change `TARGET_PIECE` to inspect a different piece.
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
        degree4_semitones = piece_rows[0]["degree4_semitones"]
        K = piece_rows[0]["K"]
        print(f"piece={target}  convention={convention}  "
              f"degree4_semitones={degree4_semitones}  K={K}")

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
