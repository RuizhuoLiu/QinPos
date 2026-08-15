#!/usr/bin/env python3
"""Compare what each weight file does to the same piece.

Written to answer one specific question: the app's timbre mix looks nothing
like `train_weights.py`'s. That comparison is between two DIFFERENT MODELS —
the notebook prints the perceptron, the app loads the CRF — so a difference is
expected; the question is whether the size of it is explained by the weights or
by a bug in the app's decoding path.

This prints, for one piece:
  * the timbre mix and exact agreement under each weight file
  * the same, decoded exactly (beam 0) and with the default beam, so beam
    width is ruled in or out
  * the weights that actually decide timbre, side by side
  * which feature families are switched off in each file

    python scripts/compare_weights.py ciou01
    python scripts/compare_weights.py --jianpu data/shenglvqimeng.jianpu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qinpos.jianpu import parse_jianpu  # noqa: E402
from qinpos.learn import WeightVector, build_sequences  # noqa: E402
from qinpos.viterbi import (ARC_FEATURES, BAND_FEATURES, CONTEXT_FEATURES,  # noqa: E402
                            FEATURES, Weights, decode, melody_context,
                            node_features)

DATA = ROOT / "data/GQ39/score_annotation"
CLEAN = ROOT / "data/gq39_clean.csv"
TIMBRE_KEYS = ["is_open", "is_harmonic", "below_center", "above_center"]


def load_all() -> dict[str, WeightVector]:
    hand = Weights()
    out = {"hand-crafted": WeightVector({k: getattr(hand, k, 0.0) for k in FEATURES})}
    for label, name in (("perceptron", "learned_weights.json"),
                        ("CRF (deployed)", "crf_weights.json")):
        path = ROOT / "data" / name
        if path.exists():
            out[label] = WeightVector(json.load(path.open()))
        else:
            print(f"  (missing {path.name})")
    return out


def mix(path) -> str:
    c = Counter(x.kind for x in path)
    return f"按 {c['stopped']:4d} · 散 {c['open']:4d} · 泛 {c['harmonic']:4d}"


def agreement(pred, expert, scored) -> str:
    if expert is None:
        return "     —"
    hit = n = 0
    for p, g, ok in zip(pred, expert, scored):
        if not ok:
            continue
        n += 1
        hit += (p.kind == g.kind and p.string == g.string)
    return f"{hit / max(n, 1):6.1%}"


def kind_breakdown(pred, expert, scored, notes) -> list[str]:
    """Where the open/stopped decision goes wrong, specifically.

    A timbre count alone cannot say whether the model is picking open strings
    it should not, or picking them in the right places and getting the string
    wrong. The row that matters is the last one: notes where an open string
    was AVAILABLE and the expert declined it anyway. That choice is not
    forced by pitch — it is the musical judgement the model has to learn, and
    it is the one that produces a wall of 散音 when it is not learned.
    """
    from qinpos.candidates import candidates_for

    kinds = ("stopped", "open", "harmonic")
    grid = {(g, p): 0 for g in kinds for p in kinds}
    declined = took_it = 0
    for note, pc, gc, ok in zip(notes, pred, expert, scored):
        if not ok:
            continue
        grid[(gc.kind, pc.kind)] += 1
        if gc.kind == "stopped" and any(c.kind == "open" for c in candidates_for(note)):
            declined += 1
            took_it += pc.kind == "open"

    out = ["    expert \\ predicted   " + " ".join(f"{k:>9s}" for k in kinds)]
    for g in kinds:
        row = " ".join(f"{grid[(g, p)]:9d}" for p in kinds)
        out.append(f"    {g:<20s} {row}")
    if declined:
        out.append(f"    open string available but the expert played 按音: {declined} notes; "
                   f"the model took the open string on {took_it} of them "
                   f"({took_it / declined:.0%})")
    return out


def explain(notes, index: int, w, label: str, expert=None) -> None:
    """Decompose the node cost of every candidate for one note.

    A timbre count says the model prefers open strings; this says WHY, in
    units of cost. It matters because the obvious suspect (`is_open`) can be
    positive — discouraging open strings — while open strings still win,
    which happens when the alternatives carry costs that open candidates are
    structurally exempt from.
    """
    from qinpos.candidates import candidates_for

    ctx = melody_context(notes)
    note = notes[index]
    cands = candidates_for(note)
    gold = expert[index] if expert is not None and index < len(expert) else None

    print(f"\n  {label} — note {index} ({note.semitones:+.0f} semitones)")
    rows = []
    for c in cands:
        feats = node_features(c, ctx[index])
        parts = {k: v * w[k] for k, v in feats.items() if v and abs(w[k]) > 1e-9}
        rows.append((sum(parts.values()), c, parts))
    rows.sort()

    for total, c, parts in rows:
        marks = ("←model" if total == rows[0][0] else "      ")
        marks += " ←EXPERT" if gold is not None and c == gold else ""
        detail = "  ".join(f"{k}{v:+.2f}" for k, v in
                           sorted(parts.items(), key=lambda kv: -abs(kv[1]))[:4])
        print(f"    {str(c):<28s} cost {total:+7.3f}  {marks:<14s} {detail}")
    print("    (lowest cost wins; this is the NODE score only, arcs excluded)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", nargs="?", help="a GQ39 piece id, e.g. ciou01")
    ap.add_argument("--jianpu", type=Path, help="a .jianpu file instead")
    ap.add_argument("--explain", type=int, nargs="*", metavar="N",
                    help="decompose the cost of every candidate for these notes")
    args = ap.parse_args()

    expert = scored = None
    if args.jianpu:
        score = parse_jianpu(args.jianpu.read_text(encoding="utf-8"))
        notes, kinds, name = score.notes, score.kinds, args.jianpu.stem
        if score.unplayable():
            print(f"{len(score.unplayable())} unplayable notes; fix the range first.")
            return 1
    elif args.piece:
        seqs = {s.piece: s for s in build_sequences(DATA, CLEAN)}
        if args.piece not in seqs:
            print(f"no piece {args.piece!r}; try --help or gq39_to_jianpu.py --list")
            return 1
        seq = seqs[args.piece]
        notes, kinds, name = seq.notes, None, seq.piece
        expert, scored = seq.expert, seq.scored
    else:
        ap.print_help()
        return 1

    weights = load_all()
    print(f"\n{name}: {len(notes)} notes")
    if expert is not None:
        print(f"  {'expert':<16s} {mix(expert)}")
    print()
    print(f"  {'weights':<16s} {'timbre mix (exact decode)':<34s} {'exact':>7s}   "
          f"{'beam 64':>8s}")
    print("  " + "-" * 72)
    for label, w in weights.items():
        p0 = decode(notes, w, kinds=kinds, beam_width=0)
        p64 = decode(notes, w, kinds=kinds, beam_width=64)
        same = "same" if p0 == p64 else "DIFFERS"
        print(f"  {label:<16s} {mix(p0):<34s} {agreement(p0, expert, scored)}   {same:>8s}")

    if expert is not None:
        for label, w in weights.items():
            print(f"\n  {label}:")
            for line in kind_breakdown(decode(notes, w, kinds=kinds, beam_width=0),
                                       expert, scored, notes):
                print(line)

    print(f"\n  {'weight':<16s} " + " ".join(f"{k:>14s}" for k in weights))
    print("  " + "-" * (17 + 15 * len(weights)))
    for key in TIMBRE_KEYS:
        row = " ".join(f"{w[key]:+14.3f}" for w in weights.values())
        print(f"  {key:<16s} {row}")
    print("\n  (cost, so NEGATIVE = cheaper = the decoder picks it more often)")

    print(f"\n  {'family':<16s} " + " ".join(f"{k:>14s}" for k in weights))
    print("  " + "-" * (17 + 15 * len(weights)))
    families = {"arc": ARC_FEATURES, "bands": BAND_FEATURES,
                "context": CONTEXT_FEATURES,
                "string bias": tuple(f"string_{i}" for i in range(1, 8))}
    for fam, keys in families.items():
        cells = []
        for w in weights.values():
            live = sum(abs(w[k]) > 1e-9 for k in keys)
            cells.append(f"{live}/{len(keys)} live".rjust(14))
        print(f"  {fam:<16s} " + " ".join(cells))
    if args.explain:
        for idx in args.explain:
            if 0 <= idx < len(notes):
                for label, w in weights.items():
                    explain(notes, idx, w, label, expert)

    print("\n  A family showing 0/N was frozen during training. crf_weights.json is\n"
          "  saved from the 'context, no bands' ablation, so arc and bands are off\n"
          "  there by design — and with no arc features there is no travel cost at\n"
          "  all, which is the first thing to suspect if open strings look cheap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
