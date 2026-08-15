"""Viterbi decoding over the candidate lattice.

[VERSION 5]
    v1 (design_history/viterbi_v1_cramped_only.py, 17.4%): one-sided
       "cramped above hui 5" penalty; no force toward the expert's
       hui 7-10 region, decoder drifted to low strings.
    v2 (design_history/viterbi_v2_comfort_band.py, 23.2%): flat comfort
       band [5, 10]; no gradient inside the band where most decisions
       happen, so arc terms decided everything and it drifted.
    v3 (design_history/viterbi_v3_center_pull.py, 49.7% hand-crafted):
       two-sided linear pull toward CENTER_HUI = 8.5.
    v4 (design_history/viterbi_v4_hand_state.py): (a) hand position
       carried through non-stopped notes via a (candidate, hand) DP
       state; (b) string x hui-band crossed node features. Result:
       node-only 48.5% exact, full model WITH arc features 40.6% --
       the arc features were worth -7.9pp and an arc-scale sweep
       improved monotonically toward zero weight.
    v5 (design_history/viterbi_v5_arc_rebuild.py): arc features rebuilt
       into three learnable families. The learned weights recovered the
       measured regularities exactly -- harm_enter/harm_exit +9.2 against
       harm_run -5.33 (break-even at ~4.5 notes, matching the measured
       run-length distribution), travel_s_4_7/8p +5.4 against
       travel_s_0/1_3 negative -- and STILL LOST 10pp of held-out exact
       match (node-only 48.5% -> full 38.8%, CRF train-test gap +4.4%
       -> +14.9%). See the note below: correct transition rules make
       error propagation worse, not better.
    v6 (this file): the same context, moved from arcs to nodes.

WHY CORRECT TRANSITION RULES STILL HURT
---------------------------------------
Every v5 arc feature encodes a regularity measured on GQ39:
P(harmonic | prev harmonic) = 0.943 at a x5.1 lift; 604 of 605
consecutive stopped pairs move 3.5 hui or less; 58.6% of consecutive
harmonic pairs cross strings at a fixed hui. The learner recovered all
of them. Held-out accuracy fell anyway, and the measured error-run
length rose from 2.10 to 3.08 notes.

The reason is that an arc feature conditions on the model's OWN
previous decision, which is right about half the time at this accuracy.
Its expected value is

    P(prev correct) * gain  -  P(prev wrong) * damage

and `damage` scales with how confidently the constraint rules
alternatives out. A near-deterministic rule (travel_s_8p at +5.4 is
effectively a prohibition) is maximally destructive when applied from a
wrong anchor: it does not merely fail to help, it actively excludes the
correct candidate. The STRENGTH of the regularity is what makes it
harmful. This is the same reason the honest backoff table gains +7.8pp
from next-pitch context but +0.2pp from previous-pitch context: what
generalises is context the model does not have to predict first.

WHAT v6 DOES ABOUT IT
---------------------
The next note's CANDIDATE SET is observable -- it follows from the
input melody alone, with no reference to any decision the model makes.
Conditioning on it is therefore free of error propagation, while still
expressing the anticipatory planning the arc features were reaching
for: a player chooses where to be now partly by where they must be
next.

`next_reach_*` buckets the distance from a candidate to the nearest
position the next note could be fingered at, and `next_same_string`
marks whether the next note is reachable on the string currently in
use. Both are node features, so the DP state and the error-propagation
behaviour are exactly those of the node-only model (train-test gap
+4.4%), not those of the arc model (+14.9%).

Set CONTEXT_FEATURES aside in an ablation to recover v5 behaviour;
pass ctx=None anywhere in the API to recover pre-v6 behaviour exactly.

WHY v4's ARC FEATURES FAILED
----------------------------
v4 charged three things: |Delta string|, linear |Delta position|, and a
flat "landing" cost. All three model PHYSICAL EFFORT, inherited from
the plucked/bowed-string fingering literature (Sayegh 1989;
Radisavljevic and Driessen 2004), where "do not move the hand" is the
dominant constraint.

On the guqin it is not. Error analysis on ciou01 (v4 weights, test
split) found the expert repeatedly alternating timbre on REPEATED
PITCHES while the model held one fingering:

    notes 13/14/15   expert  散7弦  -> 按5弦10徽 -> 散7弦
    notes 35/36/37   expert  散6弦  -> 按4弦10徽 -> 散6弦
    notes 48/49/50   expert  散7弦  -> 按5弦10徽 -> 散7弦
    notes 67/68/69   expert  散6弦  -> 按4弦10徽 -> 散6弦
    notes 74/75/76   expert  散7弦  -> 按5弦10徽 -> 散7弦

The model predicted the SAME realisation for all three notes of each
group. Alternating costs MORE physical effort, and the expert does it
anyway: the driver is timbre contrast, not economy. A cost function
that only knows "moving is expensive" cannot represent this, and the
best it can do is switch itself off -- which is exactly what the arc
weight sweep found.

WHAT v5 ADDS
------------
Three families, all learnable, none hand-tuned into the result:

  (a) REPETITION. `repeat_identical` fires when consecutive notes use
      the identical (string, position, kind). Two adjacent notes can
      only share a realisation if they share a pitch, so this is a
      pitch-free encoding of "the same note played the same way twice"
      and needs no change to any caller's signature. A positive learned
      weight reproduces the alternation above; a negative one would say
      guqin players prefer to repeat. The data decides.

  (b) TIMBRE-RUN STRUCTURE. 泛音 passages are sectional in guqin
      practice -- harmonics arrive as a marked passage (泛起 ... 泛止),
      not as isolated events. `harm_run` / `harm_enter` / `harm_exit`
      let the model learn a low within-passage cost and a high entry
      and exit cost, which is what "sectional" means in a chain model.
      `open_run` is the same construction for 散音.

  (c) NON-MONOTONIC TRAVEL. v4's linear `hand_travel` can only say
      "further is worse", one slope for everything. The real profile is
      not monotonic: a 1-3 hui move on one string is the home ground of
      the 走手音 techniques (上, 下, 绰, 注, 吟, 猱) and is musically
      preferred over staying put, while an 8-hui jump is genuinely
      expensive. Four indicator buckets give the model a free-form
      shape over distance. `same_hui_cross` covers the other cheap
      move a linear term cannot see: crossing to an adjacent string at
      the same hui, where the finger barely relocates at all.

v5 also completes the v4-era harmonic fix. `_next_hand` was updated to
let harmonics set the hand position, but `arc_features` still gated
travel on `b.kind == "stopped"`, so harmonic-to-harmonic travel was
still charged zero and consecutive harmonics could teleport. Travel now
fires for any fingered (non-open) target.

Framework follows Sayegh (1989): fingering assignment as a minimum-cost
path through a per-note candidate lattice, solved by dynamic programming.

Costs are structured as  weight . feature  (a linear model):
    total_cost(path) = sum_i  w . f_node(c_i)  +  sum_i  w . f_arc(c_{i-1}, c_i, hand_i)
so Phase 3 (path-difference learning, Radisavljevic and Driessen, 2004)
can replace hand-crafted weights with learned ones without touching the
decoder: the gradient of total cost w.r.t. w is the feature-count
difference between the expert path and the current best path.

CONSISTENCY REQUIREMENT: decode_lattice() and path_features() must
derive `hand` identically, or the learner's target features describe a
path the decoder scores differently and the perceptron drifts. Both go
through _next_hand(); do not inline that logic anywhere else. crf.py
imports _next_hand/node_cost/arc_cost from here for the same reason.

Complexity: O(n * S^2 * H) for n notes, S candidates per note (S <= ~12)
and H distinct live hand positions. H is capped by `beam_width`, so
decoding stays linear in n.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import candidates_for
from .theory import Candidate, Note

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# Node features (about one candidate in isolation):
#   is_open       : 1 if 散音 (free left hand, resonant) else 0
#   is_harmonic   : 1 if 泛音 else 0
#   below_center  : max(0, 8.5 - position) for stopped notes -- pull
#                   toward the hui-8.5 home region from the yueshan side
#   above_center  : max(0, position - 8.5) -- pull from the nut side
#   string_1..7   : one-hot marker of which string a candidate uses
#   sb_{s}_{b}    : string s crossed with hui band b, stopped notes only
#
# Arc features (about a transition a -> b, given `hand`):
#   string_cross    : |Δstring| -- right-hand plucking cost, always the
#                     two ADJACENT notes regardless of timbre
#   reposition      : 1 when entering a stopped note from open/harmonic
#   travel_s_*      : one-hot bucket of |position - hand| when entering
#   travel_h_*        a stopped (_s_) or harmonic (_h_) note, where hand
#                     is the last fingered position at or before the
#                     previous note. Buckets, not a slope, so the cost
#                     of distance can be non-monotonic; split by target
#                     timbre because the two geometries differ.
#   string_cross_harm: |Δstring| again, but only inside a 泛音段
#   harm_run        : 1 when both notes are 泛音 (inside a 泛音段)
#   harm_enter      : 1 on entering 泛音 from anything else (泛起)
#   harm_exit       : 1 on leaving 泛音 for anything else (泛止)
#   open_run        : 1 when both notes are 散音
#   repeat_identical: 1 when both notes use the same (string, position,
#                     kind) -- i.e. a repeated pitch played identically
#   same_hui_cross  : 1 when two notes sit at the same hui on different
#   same_hui_cross_harm  strings (finger barely relocates); _harm is the
#                     same event inside a 泛音段, where it is the single
#                     most common transition in the corpus

# Coarse hui bands. Boundaries chosen from the GQ39 expert position
# histogram: the mass sits in 7-10, so the two central bands are narrow
# and the outer two absorb the tails. Bands (not raw position) keep the
# feature count at 7 x 4 = 28 rather than one per distinct hui.fen.
HUI_BANDS = ((0.0, 6.5), (6.5, 9.0), (9.0, 11.0), (11.0, 99.0))

# Left-hand travel buckets, in hui units. The first is "did not really
# move"; the second is the 走手音 range reachable without releasing the
# string; the third is a deliberate shift; the fourth is a jump across
# most of the playing length. Deliberately coarse -- four indicators on
# ~1300 training notes, against v4's single slope.
TRAVEL_BUCKETS = ((0.0, 0.3), (0.3, 3.5), (3.5, 7.5), (7.5, 99.0))
# Split by the TIMBRE OF THE TARGET note, because the two geometries
# measured on GQ39 barely overlap (consecutive-pair counts, all pieces):
#
#                     same string   same hui,     stayed    short slide
#                                   diff string   put       (1-3 hui)
#   stopped (n=605)      66.8%         18.0%       19.7%       47.1%
#   harmonic (n=244)      5.3%         58.6%        4.1%        1.2%
#
# Stopped notes walk ALONG a string; harmonics walk ACROSS strings at a
# fixed hui line. One shared set of travel weights would average two
# opposite profiles into neither.
TRAVEL_FEATURES_STOPPED = ("travel_s_0", "travel_s_1_3", "travel_s_4_7", "travel_s_8p")
TRAVEL_FEATURES_HARM = ("travel_h_0", "travel_h_1_3", "travel_h_4_7", "travel_h_8p")
TRAVEL_FEATURES = TRAVEL_FEATURES_STOPPED + TRAVEL_FEATURES_HARM

# Same-hui tolerance for the adjacent-string move, in hui units. Sits
# well above the measured GQ39 snap noise (stopped p99 = 0.10 hui).
SAME_HUI_TOL = 0.3

# Distance from a candidate to the nearest position the NEXT note can be
# fingered at. Same boundaries as TRAVEL_BUCKETS so the two are directly
# comparable in the weight table: travel_s_* is what the hand did, and
# next_reach_* is what it will have to do.
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
# v5 families, so each new idea can be ablated on its own rather than
# only as "all arc features vs none" (the v4 ablation, which could not
# tell a bad feature from a bad feature FAMILY).
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

# Distance from a candidate to the nearest position the NEXT note can be
# fingered at. Same boundaries as TRAVEL_BUCKETS so the two are directly
# comparable in the weight table: travel_s_* is what the hand did, and
# next_reach_* is what it will have to do.
CENTER_HUI = 8.5  # empirical home position of the left hand (see above)

_UNSET = object()


@dataclass(frozen=True)
class NoteContext:
    """What is observable about the NEXT note, from the melody alone.

    Deliberately holds the next note's whole candidate SET rather than
    any single realisation of it: the set is a function of the input
    pitch, so conditioning on it cannot propagate a decoding error the
    way an arc feature does.
    """

    positions: tuple[float, ...]  # positions of the next note's fingered candidates
    strings: frozenset[int]  # every string the next note can be played on


def melody_context(notes: list[Note], cols=None) -> list[NoteContext | None]:
    """Per-note view of the FOLLOWING note. Entry i describes note i+1;
    the last entry is None because there is nothing after it.

    `cols` may be supplied when the caller has already enumerated the
    candidate lattice, to avoid a second candidates_for() pass.

    IMPORTANT: pass the UNRESTRICTED candidate columns. Building this
    from a lattice with the expert's choice pinned (as learn.gold_path
    does) would leak the answer for note i+1 into the features of note
    i, and the resulting accuracy would be meaningless.
    """
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

    Stopped and harmonic notes both place a finger at a hui position;
    only open strings leave the left hand uninvolved. This matters for
    harmonic passages (泛音段), where consecutive harmonics require
    real left-hand travel between hui points.

    None means no fingered note has occurred yet in this piece, so
    there is no position to travel from.

    Single source of truth for hand tracking: decode_lattice(),
    path_features() and crf.forward_backward() all call this.
    """
    return hand if c.kind == "open" else c.position


def node_features(c: Candidate, ctx: NoteContext | None = None) -> dict[str, float]:
    """Features of one candidate. `ctx` describes the NEXT note (see
    melody_context); ctx=None disables the v6 context family and
    reproduces v5 exactly.

    Every context feature must VARY ACROSS the candidates of a column.
    A feature constant within a column shifts every path score by the
    same amount and cancels in both the argmax and the softmax, so it
    can never affect a decision -- which is why "does the next note
    have an open candidate" is absent here: it is a property of the
    next note alone, identical for every candidate being compared.
    """
    stopped = c.kind == "stopped"
    feats = {
        "is_open": 1.0 if c.kind == "open" else 0.0,
        "is_harmonic": 1.0 if c.kind == "harmonic" else 0.0,
        "below_center": max(0.0, CENTER_HUI - c.position) if stopped else 0.0,
        "above_center": max(0.0, c.position - CENTER_HUI) if stopped else 0.0,
        **{f"string_{s}": 1.0 if c.string == s else 0.0 for s in range(1, 8)},
        **{k: 0.0 for k in BAND_FEATURES},
    }
    # Band features fire for stopped notes only: an open string has no
    # hui position, and a harmonic is a light touch with different
    # mechanics from a pressed note.
    if stopped:
        feats[f"sb_{c.string}_{hui_band(c.position)}"] = 1.0

    # --- v6 anticipatory context ---
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

    `hand` is the position of the most recent FINGERED note (stopped or
    harmonic) at or before `a`, or None if there has not been one.
    Omitting it falls back to reading the hand off `a` alone, which is
    only correct when `a` is itself fingered; live code always passes
    it explicitly from the DP state.
    """
    if hand is _UNSET:
        hand = a.position if a.kind != "open" else None

    f = {k: 0.0 for k in ARC_FEATURES}

    a_harm, b_harm = a.kind == "harmonic", b.kind == "harmonic"
    both_harm = a_harm and b_harm

    # --- right hand: plucking distance is always between the two
    # adjacent notes, whatever their timbre ---
    f["string_cross"] = float(abs(a.string - b.string))
    # Correction term inside a 泛音段, where crossing strings is the
    # NORM rather than a cost: 94.7% of consecutive harmonic pairs in
    # GQ39 change string, against 33.2% of stopped pairs. Learned on
    # top of string_cross, so the harmonic rate is string_cross +
    # string_cross_harm and the two need not agree in sign.
    if both_harm:
        f["string_cross_harm"] = float(abs(a.string - b.string))
    f["reposition"] = 1.0 if (b.kind == "stopped" and a.kind != "stopped") else 0.0

    # --- left hand: travel into any fingered note, bucketed, and kept
    # separate per target timbre (see TRAVEL_BUCKETS) ---
    # NOTE (v5): gated on "not open", not on "== stopped". The v4 gate
    # silently exempted harmonic-to-harmonic moves from every travel
    # cost, so 泛音段 could teleport between hui points.
    if b.kind != "open" and hand is not None:
        bucket = travel_bucket(abs(b.position - hand))
        names = TRAVEL_FEATURES_HARM if b_harm else TRAVEL_FEATURES_STOPPED
        f[names[bucket]] = 1.0

    # --- timbre-run structure (泛音段 / 散音 runs) ---
    # Measured on GQ39 train pieces: P(harmonic) = 0.185 but
    # P(harmonic | prev harmonic) = 0.943, a x5.1 lift, over 17 runs of
    # mean length 14. Harmonics are sectional, and a flat per-note
    # is_harmonic cost cannot make the second harmonic of a passage
    # cheaper than the first. open_run is the matching construction for
    # 散音, where the measured lift is only x1.17 -- it is kept as a
    # NEGATIVE CONTROL and is expected to learn a weight near zero.
    if both_harm:
        f["harm_run"] = 1.0
    elif b_harm:
        f["harm_enter"] = 1.0  # 泛起
    elif a_harm:
        f["harm_exit"] = 1.0  # 泛止
    if a.kind == "open" and b.kind == "open":
        f["open_run"] = 1.0

    # --- repetition ---
    # Adjacent notes can only share a realisation if they share a pitch,
    # so this identifies "repeated pitch, played the same way" without
    # the arc scorer ever needing to see the pitch itself. Measured on
    # the 321 adjacent same-pitch pairs in GQ39: only 43.0% repeat the
    # realisation, 35.2% change timbre and 21.8% change string. A
    # context-free model repeats by construction, so 43.0% is its hard
    # ceiling on the second note of every such pair.
    if a.kind == b.kind and a.string == b.string and abs(a.position - b.position) < 1e-9:
        f["repeat_identical"] = 1.0

    # --- same hui, different string: cheap for both timbres, and the
    # DOMINANT harmonic move (58.6% of consecutive harmonic pairs) ---
    if a.string != b.string and abs(a.position - b.position) < SAME_HUI_TOL:
        if both_harm:
            f["same_hui_cross_harm"] = 1.0
        elif a.kind == "stopped" and b.kind == "stopped":
            f["same_hui_cross"] = 1.0

    return f


# ---------------------------------------------------------------------------
# Weights (Path A: hand-crafted starting point; Path B learns these)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Weights:
    """Hand-crafted Path A weights.

    The v5 arc defaults are deliberately mild and are NOT the result
    reported anywhere: they exist so decode() is usable before training
    and so `train(init=Weights())` has a sane start. Every claim in the
    write-up should come from learned weights.
    """

    is_open: float = -0.3  # mild preference for resonant open strings
    is_harmonic: float = 0.0  # neutral unless the score demands 泛音
    below_center: float = 0.6  # per-hui pull toward hui 8.5 (yueshan side)
    above_center: float = 0.6  # per-hui pull toward hui 8.5 (nut side)
    string_cross: float = 0.3  # per-string crossing cost (right hand)
    string_cross_harm: float = -0.3  # crossing is the norm inside 泛音段
    reposition: float = 0.4  # landing the hand from open/harmonic
    # stopped travel: 604 of 605 consecutive stopped pairs in GQ39 move
    # 3.5 hui or less, so the far buckets are near-prohibitions rather
    # than costs; and the 1-3 hui slide (47.1%) is MORE common than
    # staying put (19.7%), which no monotonic slope can express.
    travel_s_0: float = 0.2
    travel_s_1_3: float = 0.0
    travel_s_4_7: float = 3.0
    travel_s_8p: float = 5.0
    # harmonic travel: harmonics move across strings at a fixed hui, so
    # any sliding at all is unusual (1.2% of pairs)
    travel_h_0: float = 0.0
    travel_h_1_3: float = 1.5
    travel_h_4_7: float = 3.0
    travel_h_8p: float = 5.0
    # timbre runs: staying inside a 泛音段 is cheap, crossing its edge
    # is a marked event
    harm_run: float = -0.5
    harm_enter: float = 1.0
    harm_exit: float = 1.0
    open_run: float = 0.0
    # repeating the identical fingering: mildly discouraged, because
    # repeated pitches in GQ39 tend to alternate timbre
    repeat_identical: float = 0.5
    same_hui_cross: float = -0.3
    same_hui_cross_harm: float = -0.8
    # v6 context: neutral by hand, learnable by Path B
    next_reach_0: float = 0.0
    next_reach_1_3: float = 0.0
    next_reach_4_7: float = 0.0
    next_reach_8p: float = 0.0
    next_same_string: float = 0.0
    # per-string biases and string x band crossings: neutral by hand,
    # learnable by Path B. Accessed via getattr with a 0.0 default in
    # dot(), so they need no explicit fields.
    #
    # REMOVED in v5: `hand_travel` (linear slope) is superseded by the
    # travel_* buckets. Old learned_weights.json files still load --
    # WeightVector only reads keys present in FEATURES and ignores the
    # rest -- but their hand_travel value is silently dropped, so
    # RETRAIN rather than reloading a v4 weight file.

    def dot(self, feats: dict[str, float]) -> float:
        return sum(getattr(self, k, 0.0) * v for k, v in feats.items())


def node_cost(c: Candidate, w: Weights, ctx: NoteContext | None = None) -> float:
    return w.dot(node_features(c, ctx))


def arc_cost(a: Candidate, b: Candidate, w: Weights, hand=_UNSET) -> float:
    return w.dot(arc_features(a, b, hand))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
DEFAULT_BEAM = 64


def decode_lattice(
    lattice: list[list[Candidate]],
    w: Weights = Weights(),
    beam_width: int = DEFAULT_BEAM,
    ctx: list | None = None,
) -> list[Candidate]:
    """Minimum-cost path through an explicit candidate lattice.

    lattice[i] is the (non-empty) candidate list for note i.
    Returns one Candidate per note.

    The DP state is (candidate index, hand) rather than just the
    candidate, because travel depends on where the left hand last
    landed, which can be several notes back when open notes intervene.
    Fingered candidates (stopped and harmonic) collapse every incoming
    hand to their own position, so the state count only grows across
    runs of 散音.

    beam_width caps live states per note, keeping the search linear in
    piece length. This makes decoding approximate rather than exact, so
    the structured perceptron's convergence guarantee no longer strictly
    holds; in practice pruned states are far from optimal. Pass
    beam_width=0 to disable pruning and decode exactly.

    `ctx` is melody_context(notes), one entry per note, or None to
    disable the v6 context features. Callers holding a lattice with the
    expert pinned must build ctx from the UNRESTRICTED candidates --
    see melody_context.
    """
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
    'harmonic'/None per note, e.g. from a score's timbre markings), the
    lattice is restricted to candidates of that kind for that note.

    When a note has no candidate of the requested kind the restriction
    is dropped for that note rather than dead-ending. That silently
    weakens the "timbre given" protocol, so pass on_fallback to count
    how often it happens before reporting numbers under that protocol.
    """
    full = [candidates_for(n) for n in notes]
    lattice = []
    for i, cands in enumerate(full):
        if kinds is not None and kinds[i] is not None:
            restricted = [c for c in cands if c.kind == kinds[i]]
            if not restricted and on_fallback is not None:
                on_fallback(i, kinds[i])
            cands = restricted or cands
        lattice.append(cands)
    # Context comes from `full`, not `lattice`: a timbre constraint says
    # how a note is played, not what the melody ahead looks like, so the
    # free-timbre and timbre-given protocols see identical context.
    return decode_lattice(lattice, w, beam_width=beam_width, ctx=melody_context(notes, full))


def path_cost(path: list[Candidate], w: Weights = Weights(), ctx: list | None = None) -> float:
    """Total cost of a concrete path (needed by Path B learning)."""
    return w.dot(path_features(path, ctx))


def path_features(path: list[Candidate], ctx: list | None = None) -> dict[str, float]:
    """Summed feature vector of a path. Path-difference learning updates
    weights by (features(best_path) - features(expert_path)).

    Hand tracking here MUST match decode_lattice(); both call
    _next_hand(). If they diverge, the learner optimises toward a path
    the decoder scores differently and the weights drift.
    """
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
