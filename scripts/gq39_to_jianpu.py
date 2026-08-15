#!/usr/bin/env python3
"""Export a GQ39 piece to the .jianpu text format.

Why this exists: the app's demo scores were transcribed from scans, so their
notes are only as good as my reading of a dot. A GQ39 piece has no such
problem -- the pitches come from the same loader the model trains on -- and it
arrives with an expert annotation, so the model's output can be compared
against a real player's choices instead of just looking plausible.

    python scripts/gq39_to_jianpu.py --list
    python scripts/gq39_to_jianpu.py ciou01
    python scripts/gq39_to_jianpu.py ciou01 --expert --out data/ciou01.jianpu

The pitches written are exactly `PieceSequence.notes`, i.e. the decoder's own
input, and the file is verified to parse back to those same pitches before it
is written. `--expert` appends each note's annotated realisation as a trailing
comment, which is only useful for reading, never for input: the parser ignores
comments, so the model still decides for itself.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qinpos.jianpu import format_pitch, parse_jianpu  # noqa: E402
from qinpos.learn import build_sequences  # noqa: E402
from qinpos.theory import OPEN_STRING_SEMITONES  # noqa: E402

DATA = ROOT / "data/GQ39/score_annotation"
CLEAN = ROOT / "data/gq39_clean.csv"


def best_gong_string(semitones: list[float]) -> tuple[int, int]:
    """Which open string to call 宫 so the piece needs fewest accidentals.

    GQ39 pitches are relative to the open 1st string; where the piece's own 宫
    sits is a property of the piece, not of the tuning. Picking the string that
    minimises sharps just makes the exported text readable -- it does not move
    a single note, because the model works in semitones either way.
    """
    # Ties are common (a pentatonic piece needs no accidentals wherever 宫
    # goes), so try string 3 first: that is 正调 convention and keeps exported
    # files consistent with the hand-written ones.
    best, best_bad = 3, None
    for s in [3] + [k for k in sorted(OPEN_STRING_SEMITONES) if k != 3]:
        gong = OPEN_STRING_SEMITONES[s]
        bad = sum("#" in format_pitch(p, gong) for p in semitones)
        if best_bad is None or bad < best_bad:
            best, best_bad = s, bad
    return best, best_bad or 0


def render(seq, gong_string: int, per_line: int, with_expert: bool) -> str:
    gong = OPEN_STRING_SEMITONES[gong_string]
    kinds = Counter(c.kind for c in seq.expert)
    n_scored = sum(seq.scored)

    lines = [
        f"// {seq.piece} — exported from GQ39 by scripts/gq39_to_jianpu.py",
        "// Pitches are PieceSequence.notes, the decoder's own input: no",
        "// transcription step, so no transcription errors.",
        f"// {len(seq.notes)} notes, {n_scored} with a trustworthy annotation.",
        f"// Expert timbre mix: 按 {kinds['stopped']} · 散 {kinds['open']} · 泛 {kinds['harmonic']}",
        "//",
        "// 宫 is placed on string {} because that needs the fewest accidentals;"
        .format(gong_string),
        "// it does not change any pitch. No timbre is forced — the whole point",
        "// is to let the model choose and then compare with the annotation.",
        f"title: {seq.piece} (GQ39)",
        f"gong_string: {gong_string}",
        "",
    ]

    for start in range(0, len(seq.notes), per_line):
        chunk = seq.notes[start:start + per_line]
        text = " ".join(format_pitch(n.semitones, gong) for n in chunk)
        if with_expert:
            marks = " ".join(
                ("?" if not ok else {"open": "散", "stopped": "按", "harmonic": "泛"}
                 .get(c.kind, "?")) + str(c.string)
                for c, ok in zip(seq.expert[start:start + per_line],
                                 seq.scored[start:start + per_line])
            )
            lines.append(f"{text}   // {start + 1}: {marks}")
        else:
            lines.append(text)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("piece", nargs="?", help="piece id, e.g. ciou01")
    ap.add_argument("--list", action="store_true", help="list available pieces")
    ap.add_argument("--out", type=Path, help="default: data/<piece>.jianpu")
    ap.add_argument("--gong-string", type=int, help="override the chosen 宫 string")
    ap.add_argument("--per-line", type=int, default=8)
    ap.add_argument("--expert", action="store_true",
                    help="append the annotated realisation as trailing comments")
    args = ap.parse_args()

    if not CLEAN.exists():
        print(f"{CLEAN} not found — run the GQ39 cleaning step first.")
        return 1
    seqs = {s.piece: s for s in build_sequences(DATA, CLEAN)}

    if args.list or not args.piece:
        print(f"{len(seqs)} pieces:")
        for name in sorted(seqs):
            s = seqs[name]
            print(f"  {name:12s} {len(s.notes):5d} notes  "
                  f"{sum(s.scored):5d} scored")
        return 0

    if args.piece not in seqs:
        close = [n for n in sorted(seqs) if n.startswith(args.piece[:3])]
        print(f"no piece {args.piece!r}. Did you mean: {', '.join(close) or '—'}")
        print("Run with --list to see them all.")
        return 1

    seq = seqs[args.piece]
    pitches = [n.semitones for n in seq.notes]
    gong_string = args.gong_string or best_gong_string(pitches)[0]
    text = render(seq, gong_string, args.per_line, args.expert)

    # Verify the text parses back to the pitches it came from. A silent
    # rounding error here would be indistinguishable from a model mistake
    # later, which is the worst kind of bug to ship into an evaluation.
    back = parse_jianpu(text)
    if back.errors:
        print("REFUSING TO WRITE — the exported text does not parse:")
        for e in back.errors[:5]:
            print("   ", e)
        return 1
    got = [n.semitones for n in back.notes]
    if len(got) != len(pitches) or any(abs(a - b) > 1e-6 for a, b in zip(got, pitches)):
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(pitches, got)) if abs(a - b) > 1e-6]
        print(f"REFUSING TO WRITE — round-trip mismatch on {len(bad)} notes, "
              f"first: index {bad[0][0]}, {bad[0][1]} -> {bad[0][2]}"
              if bad else
              f"REFUSING TO WRITE — note count {len(pitches)} -> {len(got)}")
        return 1

    unplayable = back.unplayable()
    out = args.out or (ROOT / "data" / f"{seq.piece}.jianpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(f"  {len(pitches)} notes, round-trip exact, 宫 on string {gong_string}")
    print(f"  range {format_pitch(min(pitches), OPEN_STRING_SEMITONES[gong_string])}"
          f" … {format_pitch(max(pitches), OPEN_STRING_SEMITONES[gong_string])}"
          f"  ({min(pitches):.0f}–{max(pitches):.0f} semitones)")
    if unplayable:
        print(f"  ⚠ {len(unplayable)} notes have no candidate — this should be "
              f"impossible for GQ39 data; report it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
