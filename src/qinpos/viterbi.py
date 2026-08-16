"""Viterbi decoding over the candidate lattice.
moved from arcs to nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import candidates_for
from .theory import Candidate, Note

# Feature extraction
# Node features (about one candidate in isolation):
#   is_open       : 1 if open散音 (free left hand, resonant) else 0
#   is_harmonic   : 1 if harmonic泛音 else 0
#   below_center  : max(0, 8.5 - position) for stopped notes -- pull toward the hui-8.5 home region from the yueshan side
#   above_center  : max(0, position - 8.5) -- pull from the nut side
#   string_1..7   : one-hot marker of which string a candidate uses
#   sb_{s}_{b}    : string s crossed with hui band b, stopped notes only
#
# Arc features (about a transition a -> b, given `hand`):
#   string_cross    : |Δstring| -- right-hand plucking cost, always the two ADJACENT notes regardless of timbre
#   reposition      : 1 when entering a stopped note from open/harmonic
#   travel_s_*      : one-hot bucket of |position - hand| when entering
#   travel_h_*        a stopped (_s_) or harmonic (_h_) note, where hand is the last fingered position at or before the previous note.
#                     Buckets, not a slope, so the cost of distance can be non-monotonic;
#                     split by target timbre because the two geometries differ.
#   string_cross_harm: |Δstring| again, but only inside a 泛音段
#   harm_run        : 1 when both notes are harmonic 泛音 (inside a set of harmonic泛音段)
#   harm_enter      : 1 on entering harmonic泛音 from anything else (泛起)
#   harm_exit       : 1 on leaving harmonic泛音 for anything else (泛止)
#   open_run        : 1 when both notes are 散音
#   repeat_identical: 1 when both notes use the same (string, position, kind) -- i.e. a repeated pitch played identically
#   same_hui_cross  : 1 when two notes sit at the same hui on different
#   same_hui_cross_harm  strings (finger barely relocates); _harm is the same event inside a set of harmonic泛音段,
#                     where it is the single most common transition in the corpus

# Coarse hui bands. Boundaries chosen from the GQ39 expert position
# histogram: the mass sits in 7-10, so the two central bands are narrow
# and the outer two absorb the tails. Bands (not raw position) keep the
# feature count at 7 x 4 = 28 rather than one per distinct hui.fen.
HUI_BANDS = ((0.0, 6.5), (6.5, 9.0), (9.0, 11.0), (11.0, 99.0))

TRAVEL_BUCKETS = ((0.0, 0.3), (0.3, 3.5), (3.5, 7.5), (7.5, 99.0))
TRAVEL_FEATURES_STOPPED = ("travel_s_0", "travel_s_1_3", "travel_s_4_7", "travel_s_8p")
TRAVEL_FEATURES_HARM = ("travel_h_0", "travel_h_1_3", "travel_h_4_7", "travel_h_8p")
TRAVEL_FEATURES = TRAVEL_FEATURES_STOPPED + TRAVEL_FEATURES_HARM

SAME_HUI_TOL = 0.3

# Distance from a candidate to the nearest position the NEXT note can be fingered at.
# Same boundaries as TRAVEL_BUCKETS so the two are directly comparable in the weight table:
# travel_s_* is what the hand did, and next_reach_* is what it will have to do.
NEXT_REACH_FEATURES = ("next_reach_0", "next_reach_1_3", "next_reach_4_7", "next_reach_8p")
CONTEXT_FEATURES = (*NEXT_REACH_FEATURES, "next_same_string")

NODE_FEATURES = (
    "is_open",
    "is_harmonic",
    "below_center",
    "above_center",
    *[f"string_{s}" for s in range(1, 8)],
    *[f"sb_{s}_{b}" for s in range(1, 8) for b in range(len(HUI_BANDS))],
    *CONTEXT_FEATURES,
)
ARC_FEATURES = (
    "string_cross",
    "string_cross_harm",
    "reposition",
    *TRAVEL_FEATURES,
    "harm_run",
    "harm_enter",
    "harm_exit",
    "open_run",
    "repeat_identical",
    "same_hui_cross",
    "same_hui_cross_harm",
)
FEATURES = NODE_FEATURES + ARC_FEATURES

# Convenient groups for ablation experiments.
STRING_BIAS_FEATURES = tuple(f"string_{s}" for s in range(1, 8))
BAND_FEATURES = tuple(f"sb_{s}_{b}" for s in range(1, 8) for b in range(len(HUI_BANDS)))
EFFORT_FEATURES = (
    "string_cross",
    "string_cross_harm",
    "reposition",
    *TRAVEL_FEATURES,
    "same_hui_cross",
    "same_hui_cross_harm",
)
TIMBRE_RUN_FEATURES = ("harm_run", "harm_enter", "harm_exit", "open_run")
REPETITION_FEATURES = ("repeat_identical",)

CENTER_HUI = 8.5  # empirical home position of the left hand (see above)

_UNSET = object()


@dataclass(frozen=True)
class NoteContext:
    """What is observable about the NEXT note, from the melody alone."""

    positions: tuple[float, ...]  # positions of the next note's fingered candidates
    strings: frozenset[int]  # every string the next note can be played on


def melody_context(notes: list[Note], cols=None) -> list[NoteContext | None]:
    """Per-note view of the following note. Entry i describes note i+1;
    the last entry is None because there is nothing after it.

    `cols` may be supplied when the caller has already enumerated the
    candidate lattice, to avoid a second candidates_for() pass."""
    if cols is None:
        cols = [candidates_for(n) for n in notes]
    out: list[NoteContext | None] = []
    for i in range(len(cols)):
        if i + 1 >= len(cols):
            out.append(None)
            continue
        nxt = cols[i + 1]
        out.append(
            NoteContext(
                positions=tuple(sorted(c.position for c in nxt if c.kind != "open")),
                strings=frozenset(c.string for c in nxt),
            )
        )
    return out


def hui_band(position: float) -> int:
    """Index of the coarse hui band containing `position`."""
    for i, (lo, hi) in enumerate(HUI_BANDS):
        if lo <= position < hi:
            return i
    return len(HUI_BANDS) - 1


def travel_bucket(distance: float) -> int:
    """Index of the travel bucket containing `distance` (hui units)."""
    for i, (lo, hi) in enumerate(TRAVEL_BUCKETS):
        if lo <= distance < hi:
            return i
    return len(TRAVEL_BUCKETS) - 1


def _next_hand(hand: float | None, c: Candidate) -> float | None:
    """Left-hand position after playing `c`.

    Stopped and harmonic notes both place the finger at a hui position;
    only open strings leave the left hand uninvolved. This matters for harmonic passages (泛音段),
    where consecutive harmonics require real left-hand travel between hui points.
    """
    return hand if c.kind == "open" else c.position


def node_features(c: Candidate, ctx: NoteContext | None = None) -> dict[str, float]:
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

    # anticipatory
    for k in CONTEXT_FEATURES:
        feats[k] = 0.0
    if ctx is not None:
        if c.kind != "open" and ctx.positions:
            reach = min(abs(p - c.position) for p in ctx.positions)
            feats[NEXT_REACH_FEATURES[travel_bucket(reach)]] = 1.0
        if c.string in ctx.strings:
            feats["next_same_string"] = 1.0
    return feats


def arc_features(a: Candidate, b: Candidate, hand=_UNSET) -> dict[str, float]:
    """Transition features for a -> b.

    `hand` is the position of the most recent stopped note at or before `a`, or None if there has not been one.
    """
    if hand is _UNSET:
        hand = a.position if a.kind != "open" else None

    f = {k: 0.0 for k in ARC_FEATURES}

    a_harm, b_harm = a.kind == "harmonic", b.kind == "harmonic"
    both_harm = a_harm and b_harm

    # right hand: plucking distance is always between the two adjacent notes, whatever their timbre
    f["string_cross"] = float(abs(a.string - b.string))
    if both_harm:
        f["string_cross_harm"] = float(abs(a.string - b.string))
    f["reposition"] = 1.0 if (b.kind == "stopped" and a.kind != "stopped") else 0.0

    # left hand: travel into any fingered note, bucketed, and kept separate per target timbre (see TRAVEL_BUCKETS)
    if b.kind != "open" and hand is not None:
        bucket = travel_bucket(abs(b.position - hand))
        names = TRAVEL_FEATURES_HARM if b_harm else TRAVEL_FEATURES_STOPPED
        f[names[bucket]] = 1.0

    if both_harm:
        f["harm_run"] = 1.0
    elif b_harm:
        f["harm_enter"] = 1.0  # 泛起
    elif a_harm:
        f["harm_exit"] = 1.0  # 泛止
    if a.kind == "open" and b.kind == "open":
        f["open_run"] = 1.0

    if a.kind == b.kind and a.string == b.string and abs(a.position - b.position) < 1e-9:
        f["repeat_identical"] = 1.0

    # same hui, different string: cheap for both timbres, and the DOMINANT harmonic move
    if a.string != b.string and abs(a.position - b.position) < SAME_HUI_TOL:
        if both_harm:
            f["same_hui_cross_harm"] = 1.0
        elif a.kind == "stopped" and b.kind == "stopped":
            f["same_hui_cross"] = 1.0

    return f


# Weights (Path A: hand-crafted starting point; Path B learns these)
@dataclass(frozen=True)
class Weights:
    """Hand-crafted Path A weights."""

    is_open: float = -0.3  # mild preference for resonant open strings
    is_harmonic: float = 0.0  # neutral unless the score demands 泛音
    below_center: float = 0.6  # per-hui pull toward hui 8.5 (yueshan side)
    above_center: float = 0.6  # per-hui pull toward hui 8.5 (nut side)
    string_cross: float = 0.3  # per-string crossing cost (right hand)
    string_cross_harm: float = -0.3  # crossing is the norm inside 泛音段
    reposition: float = 0.4  # landing the hand from open/harmonic
    travel_s_0: float = 0.2
    travel_s_1_3: float = 0.0
    travel_s_4_7: float = 3.0
    travel_s_8p: float = 5.0
    travel_h_0: float = 0.0
    travel_h_1_3: float = 1.5
    travel_h_4_7: float = 3.0
    travel_h_8p: float = 5.0
    # timbre runs: staying inside a set of harmonic泛音段 is cheap, crossing its edge is a marked event
    harm_run: float = -0.5
    harm_enter: float = 1.0
    harm_exit: float = 1.0
    open_run: float = 0.0
    repeat_identical: float = 0.5
    same_hui_cross: float = -0.3
    same_hui_cross_harm: float = -0.8
    # by hand, learnable by Path B
    next_reach_0: float = 0.0
    next_reach_1_3: float = 0.0
    next_reach_4_7: float = 0.0
    next_reach_8p: float = 0.0
    next_same_string: float = 0.0

    def dot(self, feats: dict[str, float]) -> float:
        return sum(getattr(self, k, 0.0) * v for k, v in feats.items())


def node_cost(c: Candidate, w: Weights, ctx: NoteContext | None = None) -> float:
    return w.dot(node_features(c, ctx))


def arc_cost(a: Candidate, b: Candidate, w: Weights, hand=_UNSET) -> float:
    return w.dot(arc_features(a, b, hand))


# Decoder
DEFAULT_BEAM = 64


def decode_lattice(
    lattice: list[list[Candidate]],
    w: Weights = Weights(),
    beam_width: int = DEFAULT_BEAM,
    ctx: list | None = None,
) -> list[Candidate]:
    """Minimum-cost path through an explicit candidate lattice"""
    if not lattice:
        return []
    for i, col in enumerate(lattice):
        if not col:
            raise ValueError(f"note {i} has no candidates")
    if ctx is None:
        ctx = [None] * len(lattice)
    elif len(ctx) != len(lattice):
        raise ValueError(f"ctx has {len(ctx)} entries for {len(lattice)} notes")

    # trail[i]: dict[(cand_index, hand)] -> (cost, previous_state_key)
    first: dict[tuple[int, float | None], tuple[float, object]] = {}
    for j, c in enumerate(lattice[0]):
        key = (j, _next_hand(None, c))
        cost = node_cost(c, w, ctx[0])
        if key not in first or cost < first[key][0]:
            first[key] = (cost, None)
    trail = [first]

    for i in range(1, len(lattice)):
        prev_states = trail[-1]
        nxt: dict[tuple[int, float | None], tuple[float, object]] = {}
        for j, cur in enumerate(lattice[i]):
            nc = node_cost(cur, w, ctx[i])
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
    'harmonic'/None )"""
    full = [candidates_for(n) for n in notes]
    lattice = []
    for i, cands in enumerate(full):
        if kinds is not None and kinds[i] is not None:
            restricted = [c for c in cands if c.kind == kinds[i]]
            if not restricted and on_fallback is not None:
                on_fallback(i, kinds[i])
            cands = restricted or cands
        lattice.append(cands)
    return decode_lattice(lattice, w, beam_width=beam_width, ctx=melody_context(notes, full))


def path_cost(path: list[Candidate], w: Weights = Weights(), ctx: list | None = None) -> float:
    """Total cost of a concrete path (needed by Path B learning)."""
    return w.dot(path_features(path, ctx))


def path_features(path: list[Candidate], ctx: list | None = None) -> dict[str, float]:
    """Summed feature vector of a path. Path-difference learning updates weights by
    (features(best_path) - features(expert_path))."""
    totals = {k: 0.0 for k in FEATURES}
    if ctx is None:
        ctx = [None] * len(path)
    elif len(ctx) != len(path):
        raise ValueError(f"ctx has {len(ctx)} entries for {len(path)} notes")
    for c, cx in zip(path, ctx):
        for k, v in node_features(c, cx).items():
            totals[k] += v

    hand: float | None = None
    prev: Candidate | None = None
    for c in path:
        if prev is not None:
            # `hand` is the last fingered position at or before `prev`
            for k, v in arc_features(prev, c, hand).items():
                totals[k] += v
        hand = _next_hand(hand, c)
        prev = c
    return totals
