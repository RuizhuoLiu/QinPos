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

  * _snap() locates the lattice candidate that corresponds to the
    expert's annotation, and the SNAPPED candidate becomes the target.
    Its existence is recorded per note in PieceSequence.reachable.
  * gold_path() pins the expert only where reachable and leaves the rest
    free, then re-decodes, so the target is always a real lattice path.

reachable and scored are DIFFERENT axes and are kept separate:
  scored    = is the annotation trustworthy?      (data quality)
  reachable = can the system express it at all?   (candidate generation)

Context-free baseline
---------------------
The lower bound that matters is not "always string N" but "look the
pitch up in a table". context_free_table() fits that table on the
training pieces and context_free_eval() applies it to held-out pieces.
A sequence model that does not beat that pair is not using context.

context_free_ceiling() is the ORACLE version (fit and scored on the same
notes). It is leaky by construction and is not a baseline -- it answers
"how far could any pitch-only mapping possibly go on this data", and the
gap from it to 100% is the share of the task that provably needs context.
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
# same fingering, in hui units. Measured distribution (Aug 2026): stopped
# median 0.010 / p99 0.090 / max 0.180, harmonic exactly 0.000. These
# tolerances sit far above the observed noise, so they are not binding.
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

    def without_string_bias(self) -> "WeightVector":
        """Copy with the seven per-string biases zeroed. Post-hoc
        ablation only -- the other weights were learned WITH these
        present, so the observed drop is an upper bound on their real
        contribution. A true ablation removes them from FEATURES and
        retrains from scratch."""
        out = self.copy()
        for s in range(1, 8):
            out[f"string_{s}"] = 0.0
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

    Ordering: verified Aug 2026 that each GQ39 piece holds a single
    section with idx running 0..n-1, so sorting on idx alone preserves
    melodic order. Re-run check_idx() if the loader changes.
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
        params = (r["convention"], int(r["degree4_semitones"]), int(r["K"]))
        if r["piece"] in piece_params and piece_params[r["piece"]] != params:
            raise ValueError(
                f"{r['piece']}: rows disagree on (convention, degree4, K): "
                f"{piece_params[r['piece']]} vs {params}. clean_gq39 fits "
                f"these per piece -- per-section variation is unsupported."
            )
        piece_params.setdefault(r["piece"], params)
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


# Loader sanity check
def check_idx(data_dir: Path) -> None:
    """Verify that sorting events on idx alone preserves melodic order.

    Prints only the pieces carrying more than one section -- those are
    the ones where a flat idx sort could interleave sections. A clean
    run means build_sequences' ordering assumption holds.
    """
    spans = defaultdict(list)
    for e in load_all(data_dir):
        spans[(e.piece, e.section)].append(e.idx)

    by_piece = defaultdict(list)
    for (p, sec), idxs in spans.items():
        by_piece[p].append((sec, min(idxs), max(idxs), len(idxs)))

    gaps = [
        (p, sec) for (p, sec), idxs in sorted(spans.items())
        if sorted(idxs) != list(range(min(idxs), min(idxs) + len(idxs)))
    ]

    multi = {p: v for p, v in by_piece.items() if len(v) > 1}
    print(f"{len(by_piece)} pieces, {len(spans)} (piece, section) groups")
    print(f"{len(multi)} piece(s) with more than one section")
    for p, v in sorted(multi.items()):
        print(f"  {p}:")
        for sec, lo, hi, n in sorted(v):
            print(f"    {str(sec):24s} n={n:4d}  idx {lo}..{hi}")
    if not multi:
        print("  -> one section per piece; sorting by idx alone is safe")
    if gaps:
        print(f"non-contiguous idx in {len(gaps)} group(s): {gaps[:10]}")
    else:
        print("idx contiguous within every group")


# Reachability diagnostics
def reach_report(seqs: list[PieceSequence], scored_only: bool = True) -> None:
    """Print per-kind reachability and the snap-distance distribution.

    On scored notes this is near-tautological: clean_gq39's residual test
    and candidates_for() share the same tuning theory, so a note that
    passes one passes the other. The informative run is
    scored_only=False, where unreachable notes act as an independent
    check on the needs_review verdict.
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
                misses[g.kind].append(float("inf"))  # no candidate on that string

    print("reachability" + (" (scored notes only)" if scored_only else " (all notes)"))
    for kind in ("stopped", "open", "harmonic"):
        y, n = tally[(kind, True)], tally[(kind, False)]
        if y + n == 0:
            continue
        print(f"  {kind:9s} {y:5d} / {y + n:5d} = {y / (y + n):6.1%} reachable")
    ty = sum(v for (_, r), v in tally.items() if r)
    tn = sum(v for (_, r), v in tally.items() if not r)
    if ty + tn:
        print(f"  {'TOTAL':9s} {ty:5d} / {ty + tn:5d} = {ty / (ty + tn):6.1%}")

    print("\nsnap distance among reachable notes (hui units)")
    for kind in ("stopped", "harmonic"):
        ds = sorted(dists[kind])
        if not ds:
            continue
        pick = lambda q: ds[min(len(ds) - 1, int(len(ds) * q))]  # noqa: E731
        print(f"  {kind:9s} n={len(ds):5d}  median {statistics.median(ds):.3f}  "
              f"p90 {pick(0.90):.3f}  p99 {pick(0.99):.3f}  max {ds[-1]:.3f}")

    if any(misses.values()):
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


# Context-free baseline: the number the sequence model must beat
def context_free_table(seqs: list[PieceSequence]) -> dict[float, tuple[int, str]]:
    """Fit pitch -> most common (string, kind) on the given pieces.

    Fit on TRAINING pieces only, then score held-out pieces with
    context_free_eval(). That pair is the honest baseline.
    """
    by_pitch: dict[float, Counter] = defaultdict(Counter)
    for seq in seqs:
        for note, g, ok in zip(seq.notes, seq.expert, seq.scored):
            if ok:
                by_pitch[round(note.semitones, 1)][(g.string, g.kind)] += 1
    return {p: c.most_common(1)[0][0] for p, c in by_pitch.items()}


def context_free_eval(table: dict[float, tuple[int, str]],
                      seqs: list[PieceSequence]) -> dict[str, float]:
    """Score a fitted pitch table on (ideally held-out) pieces.

    A pitch absent from the table counts as a miss, not a skip: the
    lookup model genuinely has no answer there.
    """
    n = hit = unseen = 0
    for seq in seqs:
        for note, g, ok in zip(seq.notes, seq.expert, seq.scored):
            if not ok:
                continue
            n += 1
            entry = table.get(round(note.semitones, 1))
            if entry is None:
                unseen += 1
                continue
            if entry == (g.string, g.kind):
                hit += 1
    return {"n": n, "exact_acc": hit / max(1, n), "unseen_pitch": unseen}


def context_free_ceiling(seqs: list[PieceSequence], per_piece: bool = False,
                         detail: bool = True) -> float:
    """ORACLE upper bound: fit and score the pitch table on the SAME notes.

    Leaky by construction -- not a baseline. It answers "how far could
    any pitch-only mapping possibly go on this data". The gap from it to
    100% is the share of the task that provably requires context.
    """
    def tally(group):
        by_pitch: dict[float, Counter] = defaultdict(Counter)
        for seq in group:
            for note, g, ok in zip(seq.notes, seq.expert, seq.scored):
                if ok:
                    by_pitch[round(note.semitones, 1)][(g.string, g.kind)] += 1
        n = sum(sum(c.values()) for c in by_pitch.values())
        m = sum(c.most_common(1)[0][1] for c in by_pitch.values())
        return m, n, by_pitch

    if per_piece:
        for seq in seqs:
            m, n, _ = tally([seq])
            if n:
                print(f"  {seq.piece:14s} ceiling {m:5d}/{n:5d} = {m / n:6.1%}")

    m, n, by_pitch = tally(seqs)
    print(f"\nCONTEXT-FREE CEILING (oracle): {m}/{n} = {m / max(1, n):.1%}")
    if detail:
        for pitch, c in sorted(by_pitch.items()):
            tot = sum(c.values())
            print(f"  pitch {pitch:5.1f}  n={tot:5d}  {len(c)} realisations  "
                  f"majority {c.most_common(1)[0][1] / tot:6.1%}")
    return m / max(1, n)


# Learning target
def gold_path(seq: PieceSequence, w) -> list[Candidate]:
    """Best path that agrees with the expert wherever the expert is
    reachable, and is free elsewhere.

    Because every column is drawn from the real lattice, the target is
    always a path the decoder could produce. When every note is
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
    frozen: tuple[str, ...] = (),
) -> WeightVector:
    """frozen: feature names held at 0.0 throughout. Freezing a weight
    at zero is equivalent to removing the feature from the model, so
    this gives a TRUE ablation rather than a post-hoc knockout."""
    w = WeightVector(init)
    for k in frozen:
        w[k] = 0.0
    total = WeightVector()
    n_accum = 0
    for _ in range(epochs):
        for seq in train_seqs:
            pred = decode(seq.notes, w)
            f_pred = path_features(pred)
            f_gold = path_features(gold_path(seq, w) if use_gold_path else seq.expert)
            if f_pred != f_gold:
                for k in FEATURES:
                    if k in frozen:
                        continue
                    w[k] += lr * (f_pred[k] - f_gold[k])
            for k in FEATURES:
                total[k] += w[k]
            n_accum += 1
    avg = WeightVector()
    for k in FEATURES:
        avg[k] = 0.0 if k in frozen else total[k] / max(1, n_accum)
    return avg


def evaluate(seqs: list[PieceSequence], w, kinds_known: bool = False) -> dict[str, float]:
    """Accuracy over scored notes.

    kinds_known=True feeds the expert's timbre as a lattice constraint
    (string-choice-only protocol, comparable to the Phase 2 eval);
    False is full free choice. The two protocols are NOT comparable to
    each other and must be reported on separate lines.
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
