"""Comfort-band cost. SUPERSEDED.

Two-sided comfort BAND [hui 5, hui 10]: zero cost inside, linear penalty outside. 
Fixed v1's missing nut-side wall but kept the flat interior, no gradient where most string choices actually happen,
so the decoder still surrendered those choices to hand-crafted arc weights. 
Measured 23.2% (old CSV) vs v1 17.4%; superseded by v3's two-sided pull toward hui 8.5 after context-free heuristic probes
showed expert choice behaves like a continuous pull (~51% alone).
LESSON: match the SHAPE of the cost to the shape of the empirical distribution, not just its support.
Runnable standalone (absolute imports); do NOT import from the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from qinpos.theory import Candidate, Note
from qinpos.candidates import candidates_for

# Feature extraction
# Node features (about one candidate in isolation):
#   is_open       : 1 if open (散音) (free left hand, resonant) else 0
#   is_harmonic   : 1 if harmonic (泛音) else 0
#   below_center  : max(0, 8.5 - position) for stopped notes, pull toward the hui-8.5 home region from the yueshan side
#   above_center  : max(0, position - 8.5), pull from the nut side (Empirical basis, July 2026: GQ39 expert stopped positions cluster tightly around hui 7-10;
#   a context-free "nearest string to hui 8.5" heuristic alone reproduces the expert string 51% of the time.
#   A flat comfort BAND gives the decoder no gradient inside the band and it drifts;
#   a two-sided pull toward the center fixes that while leaving the asymmetry learnable in Path B.)
# Arc features (about a transition):
#   string_cross  : |Δstring|
#   hand_travel   : |Δposition| when both notes are stopped (left-hand slide)
#   reposition    : 1 when entering a stopped note from open/harmonic
#                   (the hand must land from the air)

NODE_FEATURES = ("is_open", "is_harmonic", "below_hui5", "above_hui10")
ARC_FEATURES = ("string_cross", "hand_travel", "reposition")
FEATURES = NODE_FEATURES + ARC_FEATURES


def node_features(c: Candidate) -> dict[str, float]:
    stopped = c.kind == "stopped"
    return {
        "is_open": 1.0 if c.kind == "open" else 0.0,
        "is_harmonic": 1.0 if c.kind == "harmonic" else 0.0,
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


# Weights (Path A: hand-crafted starting point; Path B will learn these)
@dataclass(frozen=True)
class Weights:
    is_open: float = -0.3       # mild preference for resonant open strings
    is_harmonic: float = 0.0    # neutral unless the score demands 泛音
    below_hui5: float = 1.0     # [v2] yueshan-side wall
    above_hui10: float = 0.6    # [v2] nut-side wall (new vs v1)
    string_cross: float = 0.3   # [v2] softened from v1's 0.7
    hand_travel: float = 0.5    # [v2] softened from v1's 1.0
    reposition: float = 0.4     # landing the hand from open/harmonic

    def dot(self, feats: dict[str, float]) -> float:
        return sum(getattr(self, k) * v for k, v in feats.items())


def node_cost(c: Candidate, w: Weights) -> float:
    return w.dot(node_features(c))


def arc_cost(a: Candidate, b: Candidate, w: Weights) -> float:
    return w.dot(arc_features(a, b))


# Decoder
def decode_lattice(lattice: list[list[Candidate]],
                   w: Weights = Weights()) -> list[Candidate]:
    """Minimum-cost path through an explicit candidate lattice.
    lattice[i] is the (non-empty) candidate list for note i.
    Returns one Candidate per note.
    """
    if not lattice:
        return []
    for i, col in enumerate(lattice):
        if not col:
            raise ValueError(f"note {i} has no candidates")

    INF = float("inf")
    # best[i][j] = min cost of any path ending at lattice[i][j]
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

    # backtrack from the cheapest final state
    j = min(range(len(lattice[-1])), key=lambda x: best[-1][x])
    path = [lattice[-1][j]]
    for i in range(len(lattice) - 1, 0, -1):
        j = back[i][j]
        path.append(lattice[i - 1][j])
    path.reverse()
    return path


def decode(notes: list[Note], w: Weights = Weights(),
           kinds: list[str] | None = None) -> list[Candidate]:
    """Decode a melody. If `kinds` is given (one of 'open'/'stopped'/ 'harmonic'/None per note), 
    the lattice is restricted to candidates of that kind for that note."""
    lattice = []
    for i, n in enumerate(notes):
        cands = candidates_for(n)
        if kinds is not None and kinds[i] is not None:
            restricted = [c for c in cands if c.kind == kinds[i]]
            cands = restricted or cands  # fall back rather than dead-end
        lattice.append(cands)
    return decode_lattice(lattice, w)


def path_cost(path: list[Candidate], w: Weights = Weights()) -> float:
    """Total cost of a concrete path (needed by Path B learning)."""
    total = sum(node_cost(c, w) for c in path)
    total += sum(arc_cost(a, b, w) for a, b in zip(path, path[1:]))
    return total


def path_features(path: list[Candidate]) -> dict[str, float]:
    """Summed feature vector of a path. Path-difference learning updates
    weights by (features(expert_path) - features(best_path))."""
    totals = {k: 0.0 for k in FEATURES}
    for c in path:
        for k, v in node_features(c).items():
            totals[k] += v
    for a, b in zip(path, path[1:]):
        for k, v in arc_features(a, b).items():
            totals[k] += v
    return totals
