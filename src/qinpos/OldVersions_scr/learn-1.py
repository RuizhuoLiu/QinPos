"""Phase 3: path-difference weight learning (Radisavljevic and Driessen,
2004 style, implemented as an averaged structured perceptron).

Idea: the decoder's cost is linear, cost(path) = w · features(path).
If the decoder's best path differs from the expert's, move w so the
expert path gets relatively cheaper:

    w  <-  w + lr * (features(pred) - features(expert))

Training sequences are FULL melodies rebuilt from the loader. Open 散音
and harmonic 泛音 events included, with NO timbre constraint given to
the decoder. So the learned is_open / is_harmonic weights encode when
experts actually choose those timbres. This fixes the Phase 2 caveat
(stopped-only sequences with gaps) and is also what makes the
user-facing "difficulty" bias possible: an offset to is_open / is_harmonic
shifts how often the decoder chooses open/harmonic 散/泛.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .candidates import candidates_for
from .dataset_gq39 import load_all
from .theory import DEGREE_SEMITONES, HARMONIC_SEMITONES_AT_HUI, OPEN_STRING_SEMITONES, Candidate, Note
from .viterbi import FEATURES, decode, decode_lattice, path_features

# Pieces with physically altered tuning (紧五弦).Out of scope for the standard-tuning (正调) system. same as clean_gq39.
EXCLUDED_TUNING_PIECES = {"yang01", "yang02", "yang03", "yu02", "yu03"}

SNAP_TOL = {"open": float("inf"), "harmonic": 0.3, "stopped": 0.4}


# Deviation
def _snap(note: Note, gold: Candidate) -> tuple[Candidate | None, float]:
    """Find the lattice candidate matching the expert's annotation.
    Returns (candidate, distance) or (None, inf) if unreachable."""
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


# Weight vector: dict-backed, duck-types viterbi. Weights via .dot()
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
        cheaper (decoder chooses them more), positive makes them rarer. difficulty slider."""
        out = self.copy()
        out["is_open"] += open_bias
        out["is_harmonic"] += harmonic_bias
        return out


# Full-sequence training data
@dataclass
class PieceSequence:
    piece: str
    notes: list[Note]  # decoder input (pitch only, no timbre)
    expert: list[Candidate]  # snapped to lattice where possible
    scored: list[bool]  # accuracy
    reachable: list[bool]  # can be expressed?


def _notated(degree: int, range_: int, convention: str, d4: int) -> float:
    r = range_ - (1 if convention == "at5" and degree in (5, 6, 7) else 0)
    base = d4 if degree == 4 else DEGREE_SEMITONES[degree]
    return base + 12 * r


def build_sequences(data_dir: Path, clean_csv: Path) -> list[PieceSequence]:
    """Rebuild full melodies (open + stopped + harmonic, in idx order).

    Pitch source per event:
      * stopped notes present in the cleaned CSV: physical, corrected
        residual (handles repaired ranges and alt_gong K automatically);
        scored unless status == needs_review.
      * open / harmonic: scored only if the expert's annotated
        choice is physically consistent with that pitch.
    Events with unknown timbre (kind '?') are dropped.
    """
    rows = list(csv.DictReader(clean_csv.open()))
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
        notes, expert, scored = [], [], []
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
            expert.append(snapped or raw)  # 不可达时保留原标注,仅用于报告
            scored.append(ok)
            reachable.append(snapped is not None)
        seqs.append(PieceSequence(piece, notes, expert, scored))
    return seqs


# Averaged structured perceptron
def train(train_seqs: list[PieceSequence], epochs: int = 15, lr: float = 0.05, init=None) -> WeightVector:
    """Averaged structured perceptron over whole-piece paths."""
    w = WeightVector(init)
    total = WeightVector()
    n_accum = 0
    for _ in range(epochs):
        for seq in train_seqs:
            pred = decode(seq.notes, w)  # NO kinds: free timbre choice
            f_pred = path_features(pred)
            f_gold = path_features(seq.expert)
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
    """Accuracy over scored notes. kinds_known=True feeds the expert's
    timbre as a lattice constraint (string-choice-only protocol,
    comparable to the Phase 2 eval); False is full free choice."""
    n = kind_ok = string_ok = both_ok = 0
    for seq in seqs:
        kinds = [c.kind for c in seq.expert] if kinds_known else None
        pred = decode(seq.notes, w, kinds=kinds)
        for p, g, s in zip(pred, seq.expert, seq.scored):
            if not s:
                continue
            n += 1
            kind_ok += p.kind == g.kind
            string_ok += p.string == g.string
            both_ok += p.kind == g.kind and p.string == g.string
    return {"n": n, "kind_acc": kind_ok / n, "string_acc": string_ok / n, "exact_acc": both_ok / n}
