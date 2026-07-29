"""Guqin position physics for standard tuning (正调).

Core idea of the whole project:
    Given a target pitch and a chosen string, the hui position (徽分) is deterministic via string-length ratios. The only real decision left is which string (and stopped vs open vs harmonic), that is the combinatorial optimisation problem solved later by Viterbi.

Conventions used throughout:
    * Pitch is measured in semitones relative to the OPEN 1st string.
      (Absolute tuning, e.g. string 1 = C2, is irrelevant to the string/position decision, so we stay in relative space.)
    * A position is expressed as hui.fen, e.g. 7.6 == 七徽六分, following the GQ39 annotation convention. 0.0 means open (散音).
    * lam (lambda) is the vibrating-length fraction: stopping a string at length fraction x from the yueshan (岳山) leaves a vibrating length of x, producing a frequency ratio of 1/x over the open string. Hui 7 sits at x = 1/2, hence the octave. Frequency ratio 2ⁿ:1 = n octaves
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

# 1. The thirteen hui: length fraction from the yueshan (岳山)
# Hui are placed at the simple-integer nodes of the string. Fraction of the FULL string length, measured from the yueshan (the bridge on the player's right, where the vibrating segment ends).
HUI_FRACTIONS: dict[int, Fraction] = {
    1: Fraction(1, 8),
    2: Fraction(1, 6),
    3: Fraction(1, 5),
    4: Fraction(1, 4),
    5: Fraction(1, 3),
    6: Fraction(2, 5),
    7: Fraction(1, 2),
    8: Fraction(3, 5),
    9: Fraction(2, 3),
    10: Fraction(3, 4),
    11: Fraction(4, 5),
    12: Fraction(5, 6),
    13: Fraction(7, 8),
}

# Convenience: floats, plus the nut (龙龈) at 1.0 as a virtual "hui 14", so that positions beyond 13 hui (十三徽外) can still be interpolated.
_HUI_X = {h: float(f) for h, f in HUI_FRACTIONS.items()}
_HUI_X[14] = 1.0


# 2. Standard tuning (正调): open strings as jianpu degrees 5,6,1,2,3,5,6
# In semitones relative to open string 1 (C D F G A c d):
OPEN_STRING_SEMITONES: dict[int, int] = {
    1: 0,  # 5  (low)
    2: 2,  # 6  (low)
    3: 5,  # 1
    4: 7,  # 2
    5: 9,  # 3
    6: 12,  # 5
    7: 14,  # 6
}

# numbered notation degree (1-7) -> semitones above gong (宫), diatonic major mapping
DEGREE_SEMITONES: dict[int, int] = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

# Harmonic (泛音) pitch produced at each hui: touching at length fraction
# p/q excites the q-th partial => pitch = q-th harmonic of the open string.
HARMONIC_SEMITONES_AT_HUI: dict[int, float] = {}
for _h, _f in HUI_FRACTIONS.items():
    _q = _f.denominator  # touching at p/q gives partial q
    import math as _m

    HARMONIC_SEMITONES_AT_HUI[_h] = 12.0 * _m.log2(_q)
# e.g. hui 7 (1/2) -> 12.0 (octave), hui 5 & 9 (1/3, 2/3) -> 19.02 (12th)


# 3. Position <-> pitch conversion (the deterministic core)
def stopped_lambda(semitones_above_open: float) -> float:
    """Vibrating-length fraction needed to raise the open string by the given number of semitones (12-TET reference - 12-tone equal temperament 十二平均律)."""
    return 2.0 ** (-semitones_above_open / 12.0)


def lambda_to_huifen(lam: float) -> float | None:
    """Convert a length fraction to hui.fen notation.
    Fen convention: the gap between hui h and hui h+1 is divided into ten linear parts, so 7.6 lies 6/10 of the way from hui 7 to hui 8.
    Returns None if lam falls right of the nut or unplayably close to the yueshan (above hui 1).
    """
    if lam >= 1.0 or lam < _HUI_X[1] - 1e-9:
        return None
    for h in range(1, 14):
        x0, x1 = _HUI_X[h], _HUI_X[h + 1]
        if x0 - 1e-9 <= lam <= x1 + 1e-9:
            fen = (lam - x0) / (x1 - x0) * 10.0
            return h + fen / 10.0
    return None


def huifen_to_lambda(pos: float) -> float:
    """Inverse of lambda_to_huifen. pos = h + fen/10, 1.0 <= pos <= 14.0."""
    h = int(pos)
    fen = (pos - h) * 10.0
    x0, x1 = _HUI_X[h], _HUI_X[min(h + 1, 14)]
    return x0 + (x1 - x0) * fen / 10.0


def stopped_position(string: int, target_semitones: float) -> float | None:
    """Deterministic 按音 position: which hui.fen on `string` produces `target_semitones` (relative to open string 1).
    None if impossible (below the open pitch, or off the playable range)."""
    interval = target_semitones - OPEN_STRING_SEMITONES[string]
    if interval < -1e-6:
        return None  # cannot stop below the open-string pitch
    if interval < 1e-6:
        return 0.0  # open string: represented as position 0 by GQ39
    return lambda_to_huifen(stopped_lambda(interval))


def harmonic_positions(string: int, target_semitones: float, tol: float = 0.9) -> list[float]:
    """All hui where a harmonic (泛音) on `string` matches the target pitch within `tol` semitones."""
    interval = target_semitones - OPEN_STRING_SEMITONES[string]
    return [float(h) for h, s in HARMONIC_SEMITONES_AT_HUI.items() if abs(s - interval) <= tol]


# 4. Note / candidate datatypes shared by the rest of the pipeline
@dataclass(frozen=True)
class Note:
    """One melody event from jianpu: pitch in semitones relative to open string 1."""

    semitones: float
    duration: float = 1.0
    is_harmonic: bool | None = None  # None = unspecified, solver may choose


@dataclass(frozen=True)
class Candidate:
    """One playable realisation of a Note."""

    string: int
    position: float  # hui.fen, 0.0 = open
    kind: str  # 'open' | 'stopped' | 'harmonic'

    def __str__(self) -> str:
        if self.kind == "open":
            return f"散(open){self.string}弦(string)"
        h = int(self.position)
        f = round((self.position - h) * 10)
        if f == 10:  # snap 9徽10分 - 10徽
            h, f = h + 1, 0
        fen = f"{f}分(fen)" if f else ""
        prefix = "泛(harmonic)" if self.kind == "harmonic" else ""
        return f"{prefix}{h}徽(hui){fen}{self.string}弦(string)"
