"""[DESIGN HISTORY — VERSION 2 of 3] Viterbi decoder, comfort-band cost.

STATUS: superseded. Kept for the dissertation's design-evolution record.
This file is runnable (absolute imports) but should NOT be imported by
the pipeline — the live decoder is src/qinpos/viterbi.py (v3).

-------------------------------------------------------------------------
WHAT THIS VERSION TRIED (reacting to v1's failure)
    v1's diagnosis showed experts concentrate in hui 7-10 while v1's
    one-sided penalty left everything >= hui 5 free. v2's fix attempt:
    a two-sided COMFORT BAND [hui 5, hui 10] —
        below_hui5  = max(0, 5 - position)    (toward yueshan: cramped)
        above_hui10 = max(0, position - 10)   (toward nut: weak tone)
    zero cost inside the band, linear penalty outside. Arc weights were
    also softened (cross 0.7->0.3, travel 1.0->0.5) so the positional
    terms could compete.

MEASURED RESULT (GQ39 cleaned stopped notes, string accuracy)
    23.2% with default weights; 28.2% best on a coarse weight grid.
    Better than v1's 17.4%, still worse than the 25.4% majority baseline
    at default, and far below the 51.1% of a trivial context-free
    "nearest string to hui 8.5" heuristic measured afterwards.

WHY IT STILL FAILED
    The band is FLAT inside [5, 10]: for a typical pitch, several
    strings all land inside the band and cost exactly the same, so the
    choice among them is again surrendered to hand_travel/string_cross
    — which, with hand-crafted weights, do not encode expert
    preference. The failure mode of v1 was reduced, not removed: the
    decoder still had no gradient precisely where most decisions happen.

    The decisive evidence came from measuring context-free heuristics:
    "pick the string whose position is nearest to hui H" scored
        H=7.0: 30.9%   H=8.0: 37.6%   H=8.5: 51.1%   H=9.0: 51.1%
    i.e. expert choice behaves like a continuous PULL toward ~hui 8.5-9,
    not like a flat acceptable band. v3 encodes exactly that.

LESSON
    Match the SHAPE of the cost to the shape of the empirical
    distribution, not just its support. A plateau where the data shows
    a peak throws away the strongest available signal.
-------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from qinpos.theory import Candidate, Note
from qinpos.candidates import candidates_for

# [v2] Two-sided band replaces v1's one-sided wall.
NODE_FEATURES = ("is_open", "is_harmonic", "below_hui5", "above_hui10")
ARC_FEATURES = ("string_cross", "hand_travel", "reposition")
FEATURES = NODE_FEATURES + ARC_FEATURES


def node_features(c: Candidate) -> dict[str, float]:
    stopped = c.kind == "stopped"
    return {
        "is_open": 1.0 if c.kind == "open" else 0.0,
        "is_harmonic": 1.0 if c.kind == "harmonic" else 0.0,
        # [v2] penalties only OUTSIDE [5, 10]; the flat interior is this
        # version's remaining flaw — no gradient where it matters most
        "below_hui5": max(0.0, 5.0 - c.position) if stopped else 0.0,
        "above_hui10": max(0.0, c.position - 10.0) if stopped else 0.0,
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
    below_hui5: float = 1.0     # [v2] yueshan-side wall
    above_hui10: float = 0.6    # [v2] nut-side wall (new vs v1)
    string_cross: float = 0.3   # [v2] softened from v1's 0.7
    hand_travel: float = 0.5    # [v2] softened from v1's 1.0
    reposition: float = 0.4

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
