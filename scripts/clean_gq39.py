"""Clean the GQ39 stopped-note annotations into a reliable training set.

Findings this script encodes (validated empirically, July 2026):

1. Per-piece gong offset K: each piece's jianpu "1" sits at a fixed
   integer semitone offset from open string 1 (e.g. -7 = gong at open
   string 4's pitch class, -12 = gong at string 1's). K is estimated
   as the rounded median of (physical - notated) over the piece.

2. Range-boundary convention differs per piece:
   * convention "at1" (our default): the range digit increments at
     degree 1 (strict octave arithmetic).
   * convention "at5": degrees 5, 6, 7 are labelled with the range of
     the octave they LEAD INTO, i.e. their physical octave is
     (annotated range - 1). This matches how qin players group the
     scale as 5,6,1,2,3 (the tuning sequence) rather than 1..7.
   Empirically, all K=-7 pieces (guan/jiou/liang, and yi07's modulated
   section) use "at5"; the rest use "at1". We select per piece by best
   fit rather than hard-coding the correlation.
   Evidence: jiou04 agreement 56/86 under at1 -> 86/86 under at5;
   yi07_modulation prefers at5 while yi07_originalmode prefers at1 —
   same piece, same annotator, convention follows the gong reading.

3. After convention selection, remaining exact +-1-octave residuals are
   isolated range typos: repaired by adjusting range, and marked
   `repaired=True` so downstream users can exclude them if wanted.

4. Anything still off by >0.6 semitones is flagged `outlier=True` and
   excluded from the clean set (candidates: chuo/zhu ornament pitches
   annotated at slide targets, or genuine typos — future work).

Output: data/gq39_clean.csv with one row per stopped note.

Usage:  uv run python scripts/clean_gq39.py
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from qinpos.dataset_gq39 import load_all
from qinpos.theory import (DEGREE_SEMITONES, OPEN_STRING_SEMITONES,
                           huifen_to_lambda)

TOL = 0.6  # semitones

DATA = Path("data/GQ39/score_annotation")
OUT = Path("data/gq39_clean.csv")


def physical_semitones(e) -> float:
    lam = huifen_to_lambda(e.position)
    return OPEN_STRING_SEMITONES[e.string] + 12 * math.log2(1 / lam)


def notated_semitones(degree: int, range_: int, convention: str) -> float:
    r = range_
    if convention == "at5" and degree in (5, 6, 7):
        r -= 1
    return DEGREE_SEMITONES[degree] + 12 * r


def fit_piece(pairs, convention):
    """Return (K, n_agree) for a piece under a convention."""
    offs = sorted(p - notated_semitones(e.degree, e.range_, convention)
                  for e, p in pairs)
    K = round(offs[len(offs) // 2])
    agree = sum(
        abs(p - notated_semitones(e.degree, e.range_, convention) - K) <= TOL
        for e, p in pairs
    )
    return K, agree


def main() -> None:
    events = load_all(DATA)
    stopped = [e for e in events
               if e.kind == "stopped" and 1.0 <= e.position <= 13.9]

    by_piece = defaultdict(list)
    for e in stopped:
        by_piece[e.piece].append((e, physical_semitones(e)))

    rows = []
    stats = defaultdict(int)
    for piece, pairs in sorted(by_piece.items()):
        # 1) pick the range convention that explains this piece best
        best = max(("at1", "at5"), key=lambda c: fit_piece(pairs, c)[1])
        K, _ = fit_piece(pairs, best)

        for e, phys in pairs:
            notated = notated_semitones(e.degree, e.range_, best) + K
            resid = phys - notated
            repaired = False
            outlier = False
            corrected_range = e.range_
            if abs(resid) <= TOL:
                pass  # clean
            elif abs(abs(resid) - 12) <= TOL:
                # 2) isolated octave typo: repair the range digit
                corrected_range = e.range_ + (1 if resid > 0 else -1)
                repaired = True
            else:
                outlier = True

            stats["clean" if not (repaired or outlier)
                  else ("repaired" if repaired else "outlier")] += 1
            rows.append({
                "piece": piece, "section": e.section, "idx": e.idx,
                "degree": e.degree, "range_annotated": e.range_,
                "range_corrected": corrected_range,
                "string": e.string, "position": e.position,
                "onset": e.onset, "multi": e.multi,
                "convention": best, "K": K,
                "physical_semitones": round(phys, 3),
                "residual": round(resid, 3),
                "repaired": repaired, "outlier": outlier,
            })

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


if __name__ == "__main__":
    main()
