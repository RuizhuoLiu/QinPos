"""Phase 3: path-difference weight learning (Radisavljevic and Driessen,
2004 style, implemented as an averaged structured perceptron).

Idea: the decoder's cost is linear, cost(path) = w . features(path).
If the decoder's best path differs from the expert's, move w so the
expert path gets relatively cheaper:

    w  <-  w + lr * (features(pred) - features(gold))

Training sequences are FULL melodies rebuilt from the loader. Open 散音
and harmonic 泛音 events included, with NO timbre constraint given to
the decoder. So the learned is_open / is_harmonic weights encode when
experts actually choose those timbres. This fixes the Phase 2 caveat
(stopped-only sequences with gaps) and is also what makes the
user-facing "difficulty" bias possible: an offset to is_open / is_harmonic
shifts how often the decoder chooses open/harmonic 散/泛.

Reachability
------------
PD learning assumes the expert path exists inside the DP graph. Here it
may not: candidates_for() computes positions from just intonation, while
the expert's annotated position carries transcription noise, so the two
never match exactly. Feeding the raw annotation as the learning target
would push below_center / above_center toward a path the decoder can
never produce.

Two mechanisms address this:

  * _snap() locates the lattice candidate that corresponds to the
    expert's annotation, and the SNAPPED candidate becomes the target.
    Its existence is recorded per note in PieceSequence.reachable.
  * gold_path() pins the expert only where reachable and leaves the rest
    free, then re-decodes. The learning target is therefore always a
    real path through the lattice (latent structured perceptron).

reachable and scored are DIFFERENT axes and are kept separate:
  scored    = is the annotation trustworthy?      (data quality)
  reachable = can the system express it at all?   (candidate generation)
A note with scored=True, reachable=False is a candidate-generation gap
and is an upper bound on achievable accuracy, not a decoder error.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .candidates import candidates_for
from .dataset_gq39 import load_all
from .theory import DEGREE_SEMITONES, HARMONIC_SEMITONES_AT_HUI, OPEN_STRING_SEMITONES, Candidate, Note
from .viterbi import FEATURES, decode, decode_lattice, path_features

# Pieces with physically altered tuning (紧五弦). Out of scope for the
# standard-tuning (正调) system. Same as clean_gq39.
EXCLUDED_TUNING_PIECES = {"yang01", "yang02", "yang03", "yu02", "yu03"}

# Maximum |annotated position - computed position| still treated as the
# same fingering, in hui units. PROVISIONAL: set these from the measured
# distribution printed by reach_report(), not by guesswork. Hui spacing
# is not linear, so a single tolerance is a simplification -- if the
# distribution is strongly kind-dependent or position-dependent, revisit.
SNAP_TOL = {"open": float("inf"), "harmonic": 0.3, "stopped": 0.4}


def _snap(note: Note, gold: Candidate) -> tuple[Candidate | None, float]:
    """Find the lattice candidate matching the expert's annotation.

    Returns (candidate, distance). candidate is None when no candidate
    shares the expert's (kind, string), or when the closest one is
    further than SNAP_TOL. The distance is returned either way so the
    near-misses can be inspected.
    """
    best, best_d = None, float("inf")
    for c in candidates_for(note):
        if c.kind != gold.kind or c.string != gold.string:
            continue
        d = 0.0 if gold.kind == "open" else abs(c.position - gold.position)
        if d < best_d:
            best, best_d = c, d
    if best is not None and best_d <= SNAP_TOL[gold.kind]:
        return best, best_d
    return None, best_d


# Weight vector: dict-backed, duck-types viterbi.Weights via .dot()
class WeightVector(dict):
    """Mutable weight vector over viterbi.FEATURES. Any object with a
    .dot(features) method works as `w` for the decoder, so learned
    vectors plug into decode() unchanged."""

    def __init__(self, init=None):
        super().__init__({k: 0.0 for k in FEATURES})
        if init is not None:
            for k in FEATURES:
                if isinstance(init, dict):
                    self[k] = float(init.get(k, 0.0))
                else:
                    self[k] = float(getattr(init, k, 0.0))

    def dot(self, feats: dict[str, float]) -> float:
        return sum(self[k] * v for k, v in feats.items())

    def copy(self) -> "WeightVector":
        out = WeightVector()
        out.update(self)
        return out

    def biased(self, open_bias: float = 0.0, harmonic_bias: float = 0.0) -> "WeightVector":
        """User-facing style control: negative bias makes 散音/泛音
        cheaper (decoder chooses them more), positive makes them rarer.
        Backend of the difficulty slider."""
        out = self.copy()
        out["is_open"] += open_bias
        out["is_harmonic"] += harmonic_bias
        return out


# Full-sequence training data
@dataclass
class PieceSequence:
    piece: str
    notes: list[Note]  # decoder input (pitch only, no timbre)
    expert: list[Candidate]  # snapped to the lattice where possible
    scored: list[bool]  # is the annotation trustworthy?
    reachable: list[bool]  # is it expressible in the lattice?
    snap_dist: list[float]  # |annotated - computed| position, for diagnosis


def _notated(degree: int, range_: int, convention: str, d4: int) -> float:
    r = range_ - (1 if convention == "at5" and degree in (5, 6, 7) else 0)
    base = d4 if degree == 4 else DEGREE_SEMITONES[degree]
    return base + 12 * r


def build_sequences(data_dir: Path, clean_csv: Path) -> list[PieceSequence]:
    """Rebuild full melodies (open + stopped + harmonic, in idx order).

    Pitch source per event:
      * stopped notes present in the cleaned CSV: physical minus the
        corrected residual (handles repaired ranges and alt_gong K
        automatically); scored unless status == needs_review.
      * open / harmonic: notation formula with the piece's fitted
        (convention, degree4, K); scored only if the expert's annotated
        choice is physically consistent with that pitch.
    Events with unknown timbre (kind '?') are dropped.
    """
    reader = csv.DictReader(clean_csv.open())
    required = {
        "piece", "section", "idx", "convention", "degree4_semitones",
        "K", "physical_semitones", "residual", "status",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"{clean_csv} is missing {sorted(missing)} - stale artifact? "
            f"re-run clean_gq39. present: {sorted(reader.fieldnames or [])}"
        )
    rows = list(reader)

    valid_status = {"clean", "repaired", "alt_gong", "needs_review"}
    unknown = {r["status"] for r in rows} - valid_status
    if unknown:
        raise ValueError(f"unexpected status values: {sorted(unknown)}")

    piece_params: dict[str, tuple[str, int, int]] = {}
    stopped_info: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        piece_params.setdefault(r["piece"], (r["convention"], int(r["degree4_semitones"]), int(r["K"])))
        stopped_info[(r["piece"], r["section"], int(r["idx"]))] = r

    def csv_pitch(r) -> float:
        phys, resid = float(r["physical_semitones"]), float(r["residual"])
        if r["status"] == "repaired":
            resid -= 12 if resid > 0 else -12
        return round(phys - resid, 3)

    events = [e for e in load_all(data_dir) if e.piece not in EXCLUDED_TUNING_PIECES and e.kind != "?"]
    by_piece = defaultdict(list)
    for e in events:
        by_piece[e.piece].append(e)

    seqs: list[PieceSequence] = []
    for piece, evs in sorted(by_piece.items()):
        if piece not in piece_params:
            continue
        conv, d4, K = piece_params[piece]
        evs.sort(key=lambda e: e.idx)
        notes: list[Note] = []
        expert: list[Candidate] = []
        scored: list[bool] = []
        reachable: list[bool] = []
        snap_dist: list[float] = []
        for e in evs:
            key = (e.piece, e.section, e.idx)
            if e.kind == "stopped" and key in stopped_info:
                r = stopped_info[key]
                pitch = csv_pitch(r)
                ok = r["status"] != "needs_review"
            else:
                pitch = _notated(e.degree, e.range_, conv, d4) + K
                if e.kind == "open":
                    ok = abs(pitch - OPEN_STRING_SEMITONES[e.string]) <= 0.6
                elif e.kind == "harmonic":
                    h = int(round(e.position))
                    ok = (
                        h in HARMONIC_SEMITONES_AT_HUI
                        and abs(OPEN_STRING_SEMITONES[e.string] + HARMONIC_SEMITONES_AT_HUI[h] - pitch) <= 0.9
                    )
                else:
                    ok = False  # stopped but not in CSV (out-of-range pos)

            note = Note(semitones=float(pitch))
            raw = Candidate(e.string, float(e.position), e.kind)
            snapped, dist = _snap(note, raw)

            notes.append(note)
            expert.append(snapped or raw)  # keep raw when unreachable, for reporting only
            scored.append(ok)
            reachable.append(snapped is not None)
            snap_dist.append(dist)
        seqs.append(PieceSequence(piece, notes, expert, scored, reachable, snap_dist))
    return seqs


# Reachability diagnostics
def reach_report(seqs: list[PieceSequence], scored_only: bool = True) -> None:
    """Print per-kind reachability and the snap-distance distribution.

    Run this BEFORE trusting any accuracy number: reach rate is the
    ceiling on exact accuracy, and the distance distribution is what
    SNAP_TOL should be set from.
    """
    tally: Counter = Counter()
    dists: dict[str, list[float]] = defaultdict(list)
    misses: dict[str, list[float]] = defaultdict(list)
    for seq in seqs:
        for g, ok, r, d in zip(seq.expert, seq.scored, seq.reachable, seq.snap_dist):
            if scored_only and not ok:
                continue
            tally[(g.kind, r)] += 1
            if r:
                dists[g.kind].append(d)
            elif d < float("inf"):
                misses[g.kind].append(d)  # right string, position too far
            else:
                misses[g.kind].append(float("inf"))  # no candidate on that string at all

    print("reachability" + (" (scored notes only)" if scored_only else " (all notes)"))
    for kind in ("stopped", "open", "harmonic"):
        y, n = tally[(kind, True)], tally[(kind, False)]
        if y + n == 0:
            continue
        print(f"  {kind:9s} {y:5d} / {y + n:5d} = {y / (y + n):6.1%} reachable")
    ty = sum(v for (_, r), v in tally.items() if r)
    tn = sum(v for (_, r), v in tally.items() if not r)
    if ty + tn:
        print(f"  {'TOTAL':9s} {ty:5d} / {ty + tn:5d} = {ty / (ty + tn):6.1%}  <- ceiling on exact_acc")

    print("\nsnap distance among reachable notes (hui units)")
    for kind in ("stopped", "harmonic"):
        ds = sorted(dists[kind])
        if not ds:
            continue
        p = lambda q: ds[min(len(ds) - 1, int(len(ds) * q))]  # noqa: E731
        print(f"  {kind:9s} n={len(ds):5d}  median {statistics.median(ds):.3f}  "
              f"p90 {p(0.90):.3f}  p99 {p(0.99):.3f}  max {ds[-1]:.3f}")

    print("\nunreachable breakdown")
    for kind in ("stopped", "open", "harmonic"):
        ms = misses[kind]
        if not ms:
            continue
        no_cand = sum(1 for d in ms if d == float("inf"))
        too_far = [d for d in ms if d < float("inf")]
        print(f"  {kind:9s} {no_cand:4d} no candidate on that string, "
              f"{len(too_far):4d} position too far", end="")
        if too_far:
            print(f" (closest {min(too_far):.3f}, median {statistics.median(too_far):.3f})")
        else:
            print()


# Learning target
def gold_path(seq: PieceSequence, w) -> list[Candidate]:
    """Best path that agrees with the expert wherever the expert is
    reachable, and is free elsewhere.

    This is the learning target. Because every column is drawn from the
    real lattice, the target is always a path the decoder could produce,
    so the perceptron update direction is attainable. When every note is
    reachable this reduces to the expert path itself.
    """
    lattice = [
        [gold] if ok else candidates_for(note)
        for note, gold, ok in zip(seq.notes, seq.expert, seq.reachable)
    ]
    return decode_lattice(lattice, w)


# Averaged structured perceptron
def train(
    train_seqs: list[PieceSequence],
    epochs: int = 15,
    lr: float = 0.05,
    init=None,
    use_gold_path: bool = True,
) -> WeightVector:
    """Averaged structured perceptron over whole-piece paths.

    use_gold_path=False reproduces the pre-reachability behaviour (raw
    expert annotation as the target). Keep it for the ablation table:
    the difference between the two is the measured cost of targeting an
    unattainable path.
    """
    w = WeightVector(init)
    total = WeightVector()
    n_accum = 0
    for _ in range(epochs):
        for seq in train_seqs:
            pred = decode(seq.notes, w)  # NO kinds: free timbre choice
            f_pred = path_features(pred)
            f_gold = path_features(gold_path(seq, w) if use_gold_path else seq.expert)
            if f_pred != f_gold:
                for k in FEATURES:
                    w[k] += lr * (f_pred[k] - f_gold[k])
            for k in FEATURES:
                total[k] += w[k]
            n_accum += 1
    avg = WeightVector()
    for k in FEATURES:
        avg[k] = total[k] / max(1, n_accum)
    return avg


def evaluate(seqs: list[PieceSequence], w, kinds_known: bool = False) -> dict[str, float]:
    """Accuracy over scored notes.

    kinds_known=True feeds the expert's timbre as a lattice constraint
    (string-choice-only protocol, comparable to the Phase 2 eval);
    False is full free choice.

    Reported separately:
      exact_acc         over all scored notes (what the user would see)
      reach_rate        fraction expressible at all (ceiling on exact_acc)
      exact_given_reach accuracy where the answer was attainable
    Unreachable notes are NOT dropped from exact_acc -- a candidate that
    cannot be generated is a real system failure, not a free pass.
    """
    n = kind_ok = string_ok = both_ok = n_reach = both_ok_reach = 0
    for seq in seqs:
        kinds = [c.kind for c in seq.expert] if kinds_known else None
        pred = decode(seq.notes, w, kinds=kinds)
        for p, g, s, r in zip(pred, seq.expert, seq.scored, seq.reachable):
            if not s:
                continue
            n += 1
            exact = p.kind == g.kind and p.string == g.string
            kind_ok += p.kind == g.kind
            string_ok += p.string == g.string
            both_ok += exact
            if r:
                n_reach += 1
                both_ok_reach += exact
    d = max(1, n)
    return {
        "n": n,
        "kind_acc": kind_ok / d,
        "string_acc": string_ok / d,
        "exact_acc": both_ok / d,
        "reach_rate": n_reach / d,
        "exact_given_reach": both_ok_reach / max(1, n_reach),
    }
