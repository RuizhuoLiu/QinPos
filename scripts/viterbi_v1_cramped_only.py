"""[DESIGN HISTORY — VERSION 1 of 3] Viterbi decoder, first cost function.

STATUS: superseded. Kept for the dissertation's design-evolution record.
This file is runnable (absolute imports) but should NOT be imported by
the pipeline — the live decoder is src/qinpos/viterbi.py (v3).

-------------------------------------------------------------------------
WHAT THIS VERSION TRIED
    Node cost for stopped notes: a single ONE-SIDED penalty,
        cramped_high = max(0, 5 - position)
    i.e. only positions above hui 5 (toward the yueshan, where pressing
    is physically cramped) are penalised. Everything from hui 5 to the
    nut was treated as equally fine. Arc costs: string_cross (w=0.7),
    hand_travel (w=1.0), reposition (w=0.4).

MEASURED RESULT (GQ39 cleaned stopped notes, string accuracy)
    17.4%  (546 would be 49.7% — see v3; majority-class baseline 25.4%)
    -> WORSE than always guessing string 7.

WHY IT FAILED (diagnosed by comparing predicted-vs-expert string
histograms, not by guessing):
    Experts put 49% of stopped notes on strings 6-7 and 78% of positions
    inside hui 7-10. This cost function contains NO force pulling the
    decoder toward that region: everything >= hui 5 costs zero, so the
    DP minimises the remaining terms (hand_travel, string_cross) and
    settles wherever those are cheapest — which turned out to be the
    LOW strings (1-3), the exact opposite of expert practice.

LESSON ENCODED IN LATER VERSIONS
    A cost function needs a gradient everywhere the decoder must choose,
    not only a wall at the region you want to forbid. See v2 (comfort
    band — still flat inside, still failed) and v3 (two-sided pull
    toward hui 8.5 — matches the expert distribution).
-------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from qinpos.theory import Candidate, Note
from qinpos.candidates import candidates_for

# [v1] One-sided node feature: only "too close to the yueshan" is bad.
NODE_FEATURES = ("is_open", "is_harmonic", "cramped_high")
ARC_FEATURES = ("string_cross", "hand_travel", "reposition")
FEATURES = NODE_FEATURES + ARC_FEATURES


def node_features(c: Candidate) -> dict[str, float]:
    return {
        "is_open": 1.0 if c.kind == "open" else 0.0,
        "is_harmonic": 1.0 if c.kind == "harmonic" else 0.0,
        # [v1] the fatal choice: zero cost for ALL positions >= hui 5,
        # leaving the decoder directionless across most of the fingerboard
        "cramped_high": max(0.0, 5.0 - c.position) if c.kind == "stopped" else 0.0,
    }


def arc_features(a: Candidate, b: Candidate) -> dict[str, float]:
    both_stopped = a.kind == "stopped" and b.kind == "stopped"
    return {
        "string_cross": float(abs(a.string - b.string)),
        "hand_travel": abs(a.position - b.position) if both_stopped else 0.0,
        "reposition": 1.0 if (b.kind == "stopped" and a.kind != "stopped") else 0.0,
    }


@dataclass(frozen=True)
class Weights:
    is_open: float = -0.3
    is_harmonic: float = 0.0
    cramped_high: float = 0.8   # [v1] the only positional force, one-sided
    string_cross: float = 0.7   # [v1] relatively heavy arc weights —
    hand_travel: float = 1.0    #      with no positional anchor these
    reposition: float = 0.4     #      dominated and dragged paths astray

    def dot(self, feats: dict[str, float]) -> float:
        return sum(getattr(self, k) * v for k, v in feats.items())


def node_cost(c: Candidate, w: Weights) -> float:
    return w.dot(node_features(c))


def arc_cost(a: Candidate, b: Candidate, w: Weights) -> float:
    return w.dot(arc_features(a, b))


def decode_lattice(lattice: list[list[Candidate]],
                   w: Weights = Weights()) -> list[Candidate]:
    if not lattice:
        return []
    for i, col in enumerate(lattice):
        if not col:
            raise ValueError(f"note {i} has no candidates")
    INF = float("inf")
    best = [[node_cost(c, w) for c in lattice[0]]]
    back: list[list[int]] = [[-1] * len(lattice[0])]
    for i in range(1, len(lattice)):
        col_best: list[float] = []
        col_back: list[int] = []
        for j, cur in enumerate(lattice[i]):
            nc = node_cost(cur, w)
            b, arg = INF, -1
            for k, prev in enumerate(lattice[i - 1]):
                cost = best[i - 1][k] + arc_cost(prev, cur, w)
                if cost < b:
                    b, arg = cost, k
            col_best.append(b + nc)
            col_back.append(arg)
        best.append(col_best)
        back.append(col_back)
    j = min(range(len(lattice[-1])), key=lambda x: best[-1][x])
    path = [lattice[-1][j]]
    for i in range(len(lattice) - 1, 0, -1):
        j = back[i][j]
        path.append(lattice[i - 1][j])
    path.reverse()
    return path


def decode(notes: list[Note], w: Weights = Weights(),
           kinds: list[str] | None = None) -> list[Candidate]:
    lattice = []
    for i, n in enumerate(notes):
        cands = candidates_for(n)
        if kinds is not None and kinds[i] is not None:
            restricted = [c for c in cands if c.kind == kinds[i]]
            cands = restricted or cands
        lattice.append(cands)
    return decode_lattice(lattice, w)
