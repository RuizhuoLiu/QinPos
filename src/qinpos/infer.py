"""Inference on melodies that are not in GQ39.

`learn.build_sequences` couples a melody to its expert annotation, which is
right for training and useless for a user who just typed a tune in. Everything
here takes a bare `list[Note]` and gives back a fingering plus the marginals
that drive the fingerboard view.

Nothing in `viterbi.node_features` / `arc_features` reads pitch or duration
directly -- the features are all about string, hui band, travel and timbre --
so the trained weights apply to any melody in 正调 relative pitch space
without retraining. What does *not* transfer is coverage: the weights were
fitted on 34 pieces of largely pentatonic material, so a melody full of 偏音
is extrapolation, not interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import candidates_for
from .crf import note_marginals
from .theory import Candidate, Note
from .viterbi import decode

__all__ = [
    "Prediction",
    "predict",
    "describe",
    "hui_fen_text",
    "left_hand_finger",
    "fingering_rows",
    "timbre_mix",
]

CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
          8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二", 13: "十三"}
KIND_CN = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}
KIND_SHORT = {"open": "散", "stopped": "按", "harmonic": "泛"}


@dataclass
class Prediction:
    notes: list[Note]
    path: list[Candidate]                      # Viterbi best fingering
    marginals: list[dict[Candidate, float]]    # per-note posterior
    baseline_path: list[Candidate] | None = None       # same model, bias = 0
    baseline_marginals: list[dict[Candidate, float]] | None = None

    @property
    def confidence(self) -> list[float]:
        return [m[c] if c in m else 0.0 for c, m in zip(self.path, self.marginals)]

    @property
    def changed_vs_baseline(self) -> list[bool]:
        if self.baseline_path is None:
            return [False] * len(self.path)
        return [a != b for a, b in zip(self.path, self.baseline_path)]

    def top_k(self, i: int, k: int = 3) -> list[tuple[Candidate, float]]:
        return sorted(self.marginals[i].items(), key=lambda kv: -kv[1])[:k]


def predict(
    notes: list[Note],
    w,
    kinds: list[str | None] | None = None,
    baseline_w=None,
) -> Prediction:
    """Decode a melody and compute its marginals under the same weights.

    `kinds` forces a timbre per note (from 泛/散 markings in the score);
    None entries leave the choice to the model.

    `baseline_w` is the reference the bias sliders are compared against --
    pass the unbiased weight vector to get `changed_vs_baseline` populated,
    which is what makes the difficulty control legible: the interesting fact
    is not where the biased model puts each note, it is which notes moved.

    Decoding uses beam_width=0 (exact). The default beam of 64 exists for
    long GQ39 pieces; a typed-in melody is short enough that there is no
    reason to let the decoder and the marginals disagree.
    """
    missing = [i for i, n in enumerate(notes) if not candidates_for(n)]
    if missing:
        raise ValueError(
            f"notes {missing} have no playable candidate -- check the range "
            f"before decoding (see jianpu.ParsedScore.unplayable)"
        )

    path = decode(notes, w, kinds=kinds, beam_width=0)
    margs = note_marginals(notes, w, kinds=kinds)
    base_path = base_margs = None
    if baseline_w is not None:
        base_path = decode(notes, baseline_w, kinds=kinds, beam_width=0)
        base_margs = note_marginals(notes, baseline_w, kinds=kinds)
    return Prediction(list(notes), path, margs, base_path, base_margs)


def hui_fen_text(position: float, cjk: bool = True) -> str:
    """0.0 -> '' ; 7.6 -> 七徽六分 ; 10.0 -> 十徽."""
    hui = int(position)
    fen = int(round((position - hui) * 10))
    if fen >= 10:
        hui, fen = hui + 1, 0
    if hui < 1:
        return ""
    if not cjk:
        return f"hui {hui}" if fen == 0 else f"hui {hui}.{fen}"
    body = f"{CN_NUM.get(hui, hui)}徽"
    return body if fen == 0 else f"{body}{CN_NUM.get(fen, fen)}分"


def describe(c: Candidate, cjk: bool = True) -> str:
    """Compact human label: '按 三弦 七徽六分'."""
    if not cjk:
        return (f"{c.kind} str{c.string}"
                + ("" if c.kind == "open" else f" {hui_fen_text(c.position, False)}"))
    string = f"{CN_NUM[c.string]}弦"
    if c.kind == "open":
        return f"散 {string}"
    return f"{KIND_SHORT[c.kind]} {string} {hui_fen_text(c.position)}"


def left_hand_finger(c: Candidate, previous: Candidate | None = None) -> str:
    """HEURISTIC left-hand finger, not a learned prediction.

    Common teaching practice, not a rule of physics: the thumb (大指) takes
    the region from the yueshan to about hui 7, the ring finger (名指) takes
    hui 7 to the nut, and 泛音 are usually touched with 名指 low on the board
    and 中指/大指 higher up. Real fingering also depends on 绰注, 上下,
    slides and the following note, none of which this looks at.

    It is here so the tablature layer has something to print; it needs a
    performer's review before it goes in front of anyone, and it should
    eventually be a learned second output rather than an if-statement.
    """
    if c.kind == "open":
        return ""
    if c.kind == "harmonic":
        return "名指" if c.position >= 7.0 else "中指"
    return "名指" if c.position >= 7.3 else "大指"


def fingering_rows(
    pred: Prediction,
    tokens=None,
    top_k: int = 3,
    with_finger: bool = True,
) -> list[dict]:
    """One dict per note, ready for a table widget or a CSV dump."""
    rows = []
    prev = None
    for i, (note, c) in enumerate(zip(pred.notes, pred.path)):
        m = pred.marginals[i]
        row = {
            "#": i + 1,
            "jianpu": tokens[i].raw if tokens else "",
            "semitones": round(note.semitones, 2),
            "timbre": KIND_CN[c.kind],
            "string": f"{CN_NUM[c.string]}弦",
            "position": hui_fen_text(c.position) or "—",
            "P": round(m.get(c, 0.0), 3),
            "alternatives": " / ".join(
                f"{describe(k)} {v:.2f}"
                for k, v in sorted(m.items(), key=lambda kv: -kv[1])[1:top_k]
            ),
        }
        if with_finger:
            row["左手 (heuristic)"] = left_hand_finger(c, prev)
        if pred.baseline_path is not None:
            base = pred.baseline_path[i]
            row["vs baseline"] = "" if base == c else f"was {describe(base)}"
        rows.append(row)
        prev = c
    return rows


def timbre_mix(path: list[Candidate]) -> dict[str, int]:
    out = {"stopped": 0, "open": 0, "harmonic": 0}
    for c in path:
        out[c.kind] += 1
    return out
