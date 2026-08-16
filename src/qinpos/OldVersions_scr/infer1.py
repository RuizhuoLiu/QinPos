"""Inference on melodies that are not in GQ39.

learn.build_sequences ties a melody to its expert fingering: useful for training, but useless for a user. 
So everything here works from a bare list[Note] and returns a fingering plus the marginals for the fingerboard view.

viterbi.node_features and arc_features never read pitch or duration directly—only string, hui band, travel, and timbre. 
So the trained weights work on any melody in trained tune relative pitch space without retraining.

What doesn't transfer: the weights were fitted on 34 mostly pentatonic pieces. A melody full of turned tune 偏音 is extrapolation, 
not interpolation.
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
    """Decode a melody and compute its marginals under the same weights."""
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
    hui 7 to the nut, and harmonic 泛音 are usually touched with 名指 low on the board
    and middel finger 中指/大指 higher up. Real fingering also depends on 绰注, 上下,
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
            "string": f"{CN_NUM[c.string]}string弦",
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
