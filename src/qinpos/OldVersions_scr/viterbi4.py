"""Viterbi decoding over the candidate lattice.
(a) Hand position now persists across open strings. v3 only charged movement between two stopped notes,
so large shifts via open strings cost nothing—unreasonable since open notes are 29% of GQ39.
DP state is now (candidate, hand).

(b) Added string × hui‑band features. v3's per‑string biases couldn't encode register‑dependent preferences;
new cross‑features give that capability.

Framework: Sayegh (1989) DP over candidate lattices. Cost = linear weights × features.
Phase 3 learns weights via path‑difference gradients.

Consistency: Both decoder and feature extractor must derive hand through the same _next_hand() function.

Complexity: O(n · S² · H), with H capped by beam width → linear in n.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import candidates_for
from .theory import Candidate, Note

# Feature extraction
# Node features (about one candidate in isolation):
#   is_open       : 1 if open 散音 (free left hand, resonant) else 0
#   is_harmonic   : 1 if harmonic 泛音 else 0
#   below_center  : max(0, 8.5 - position) for stopped notes - pull toward the hui-8.5 home region from the yueshan side
#   above_center  : max(0, position - 8.5) - pull from the nut side
#   string_1..7   : one-hot marker of which string a candidate uses
#   sb_{s}_{b}    : string s crossed with hui band b, stopped notes only
# Arc features (about a transition):
#   string_cross  : |Δstring| -- right-hand plucking cost, always the two ADJACENT notes regardless of timbre
#   hand_travel   : |position - hand| when entering a stopped note,
#                   where hand is the last stopped position at or before the previous note
#   reposition    : 1 when entering a stopped note from open/harmonic

# Coarse hui bands. Boundaries chosen from the GQ39 expert position
# histogram: the mass sits in 7-10, so the two central bands are narrow and the outer two absorb the tails.
# Bands (not raw position) keep the feature count at 7 x 4 = 28 rather than one per distinct hui.fen.
HUI_BANDS = ((0.0, 6.5), (6.5, 9.0), (9.0, 11.0), (11.0, 99.0))

NODE_FEATURES = (
    "is_open",
    "is_harmonic",
    "below_center",
    "above_center",
    *[f"string_{s}" for s in range(1, 8)],
    *[f"sb_{s}_{b}" for s in range(1, 8) for b in range(len(HUI_BANDS))],
)
ARC_FEATURES = ("string_cross", "hand_travel", "reposition")
FEATURES = NODE_FEATURES + ARC_FEATURES

# Convenient groups for ablation experiments.
STRING_BIAS_FEATURES = tuple(f"string_{s}" for s in range(1, 8))
BAND_FEATURES = tuple(f"sb_{s}_{b}" for s in range(1, 8) for b in range(len(HUI_BANDS)))

CENTER_HUI = 8.5  # empirical home position of the left hand (see above)

_UNSET = object()


def hui_band(position: float) -> int:
    """Index of the coarse hui band containing `position`."""
    for i, (lo, hi) in enumerate(HUI_BANDS):
        if lo <= position < hi:
            return i
    return len(HUI_BANDS) - 1


def _next_hand(hand: float | None, c: Candidate) -> float | None:
    """Left-hand position after playing `c`.

    Stopped and harmonic notes both place the finger at a hui position;
    only open strings leave the left hand uninvolved. This matters for harmonic passages (泛音段),
    where consecutive harmonics require real left-hand travel between hui points.
    """
    return hand if c.kind == "open" else c.position


def node_features(c: Candidate) -> dict[str, float]:
    stopped = c.kind == "stopped"
    feats = {
        "is_open": 1.0 if c.kind == "open" else 0.0,
        "is_harmonic": 1.0 if c.kind == "harmonic" else 0.0,
        "below_center": max(0.0, CENTER_HUI - c.position) if stopped else 0.0,
        "above_center": max(0.0, c.position - CENTER_HUI) if stopped else 0.0,
        **{f"string_{s}": 1.0 if c.string == s else 0.0 for s in range(1, 8)},
        **{k: 0.0 for k in BAND_FEATURES},
    }
    # Band features fire for stopped notes only: an open string has no hui position,
    # and a harmonic is a light touch with different mechanics from a pressed note.
    if stopped:
        feats[f"sb_{c.string}_{hui_band(c.position)}"] = 1.0
    return feats


def arc_features(a: Candidate, b: Candidate, hand=_UNSET) -> dict[str, float]:
    """Transition features for a -> b.

    `hand` is the position of the most recent stopped note at or before `a`, or None if there has not been one.
    """
    if hand is _UNSET:
        hand = a.position if a.kind == "stopped" else None

    travel = 0.0
    if b.kind == "stopped" and hand is not None:
        travel = abs(b.position - hand)

    return {
        "string_cross": float(abs(a.string - b.string)),
        "hand_travel": travel,
        "reposition": 1.0 if (b.kind == "stopped" and a.kind != "stopped") else 0.0,
    }


# Weights (Path A: hand-crafted starting point; Path B learns these)
@dataclass(frozen=True)
class Weights:
    is_open: float = -0.3  # mild preference for resonant open strings
    is_harmonic: float = 0.0  # neutral unless the score demands 泛音
    below_center: float = 0.6  # per-hui pull toward hui 8.5 (yueshan side)
    above_center: float = 0.6  # per-hui pull toward hui 8.5 (nut side)
    string_cross: float = 0.3  # per-string crossing cost (right hand)
    hand_travel: float = 0.5  # per-hui left-hand slide cost
    reposition: float = 0.4  # landing the hand from open/harmonic
    # per-string biases and string x band crossings: neutral by hand,
    # learnable by Path B. Accessed via getattr with a 0.0 default in
    # dot(), so they need no explicit fields.

    def dot(self, feats: dict[str, float]) -> float:
        return sum(getattr(self, k, 0.0) * v for k, v in feats.items())


def node_cost(c: Candidate, w: Weights) -> float:
    return w.dot(node_features(c))


def arc_cost(a: Candidate, b: Candidate, w: Weights, hand=_UNSET) -> float:
    return w.dot(arc_features(a, b, hand))


# Decoder
DEFAULT_BEAM = 64


def decode_lattice(
    lattice: list[list[Candidate]], w: Weights = Weights(), beam_width: int = DEFAULT_BEAM
) -> list[Candidate]:
    """Minimum-cost path through an explicit candidate lattice."""
    if not lattice:
        return []
    for i, col in enumerate(lattice):
        if not col:
            raise ValueError(f"note {i} has no candidates")

    # trail[i]: dict[(cand_index, hand)] -> (cost, previous_state_key)
    first: dict[tuple[int, float | None], tuple[float, object]] = {}
    for j, c in enumerate(lattice[0]):
        key = (j, _next_hand(None, c))
        cost = node_cost(c, w)
        if key not in first or cost < first[key][0]:
            first[key] = (cost, None)
    trail = [first]

    for i in range(1, len(lattice)):
        prev_states = trail[-1]
        nxt: dict[tuple[int, float | None], tuple[float, object]] = {}
        for j, cur in enumerate(lattice[i]):
            nc = node_cost(cur, w)
            for state, (pc, _) in prev_states.items():
                k, hand = state
                prev = lattice[i - 1][k]
                cost = pc + arc_cost(prev, cur, w, hand) + nc
                key = (j, _next_hand(hand, cur))
                if key not in nxt or cost < nxt[key][0]:
                    nxt[key] = (cost, state)
        if beam_width and len(nxt) > beam_width:
            nxt = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:beam_width])
        trail.append(nxt)

    # backtrack from the cheapest final state
    key = min(trail[-1], key=lambda k: trail[-1][k][0])
    path: list[Candidate] = []
    for i in range(len(lattice) - 1, -1, -1):
        path.append(lattice[i][key[0]])
        key = trail[i][key][1]
    path.reverse()
    return path


def decode(
    notes: list[Note],
    w: Weights = Weights(),
    kinds: list[str] | None = None,
    beam_width: int = DEFAULT_BEAM,
    on_fallback=None,
) -> list[Candidate]:
    """Decode a melody. If `kinds` is given (one of 'open'/'stopped'/
    'harmonic'/None per note), the lattice is restricted to candidates of that kind for that note.

    When a note has no candidate of the requested kind the restriction is dropped for that note rather than dead-ending.
    That silently weakens the "timbre given" protocol,
    so pass on_fallback to count how often it happens before reporting numbers under that protocol.
    """
    lattice = []
    for i, n in enumerate(notes):
        cands = candidates_for(n)
        if kinds is not None and kinds[i] is not None:
            restricted = [c for c in cands if c.kind == kinds[i]]
            if not restricted and on_fallback is not None:
                on_fallback(i, kinds[i])
            cands = restricted or cands
        lattice.append(cands)
    return decode_lattice(lattice, w, beam_width=beam_width)


def path_cost(path: list[Candidate], w: Weights = Weights()) -> float:
    """Total cost of a concrete path (needed by Path B learning)."""
    return w.dot(path_features(path))


def path_features(path: list[Candidate]) -> dict[str, float]:
    """Summed feature vector of a path. Path-difference learning updates weights by (features(best_path) - features(expert_path)).

    Hand tracking here MUST match decode_lattice(); both call _next_hand().
    If they diverge, the learner optimises toward a path the decoder scores differently and the weights drift.
    """
    totals = {k: 0.0 for k in FEATURES}
    for c in path:
        for k, v in node_features(c).items():
            totals[k] += v

    hand: float | None = None
    prev: Candidate | None = None
    for c in path:
        if prev is not None:
            # `hand` is the last stopped position at or before `prev`
            for k, v in arc_features(prev, c, hand).items():
                totals[k] += v
        hand = _next_hand(hand, c)
        prev = c
    return totals
