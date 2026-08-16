"""Loader for the GQ39 score annotations (Huang, Liang, Wei and Su, 2020).

Columns consumed per event row:
    degree, range        -> melody pitch (supervision INPUT)
    string, position     -> expert choice (supervision TARGET)
    'L tech (timbre)'    -> 1 = harmonic, 2 = open, 3 = stopped

Messy realities handled:
    * Chords 撮 and multi-string sweeps 历/滚拂 pack several values in one cell, either as strings '2,3,5' or floats 3.6
    Exploded here into simultaneous events sharing an onset.
    * `position` mixes float / str / 0 (open).
    * `position` when hui values run 1-13 (two digits for 10-13), so a packed pair like "10,12" collapses to the float 10.12 then became (10, 1).
      try both a one-digit and a two-digit reading, keep whichever falls inside the physically playable hui range.
    * A row's packed fields (degree/range/string/position) can disagree in how many values they hold 
     (e.g. 3 degrees packed but only 2 positions given). - skipped with warning
    * Legend text in trailing columns is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TIMBRE = {1: "harmonic", 2: "open", 3: "stopped"}
KEEP = ["degree", "range", "onset", "string", "position", "L tech (timbre)"]

MIN_HUI = 2.0 # STOPPED_MIN_HUI/STOPPED_MAX_HUI): below hui ~2 the string is too tense to stop cleanly
MAX_HUI = 13.9


class PackedLengthMismatch(ValueError):
    """Raised when a row's packed fields imply different note counts
    and there's no safe way to reconcile them (see module docstring)."""


@dataclass
class Event:
    piece: str
    section: str
    idx: int
    degree: int
    range_: int
    string: int
    position: float
    kind: str            # open | stopped | harmonic 
    onset: float
    multi: bool = False    # part of a chord / sweep
    suspect: bool = False  # position was repaired from a period-packed cell


def _values(cell, integer_column: bool) -> list[float]:
    """2 strings: 3.6 -> (3, 6)
    1 string: 7.9 is a real
    """
    if isinstance(cell, str):
        text = cell.replace("\uff0c", ",")
        if "," in text:
            return [float(p) for p in text.split(",") if p.strip() != ""]
        cell = float(text)  # no comma: fall through to numeric handling
    v = float(cell)
    if integer_column and v != int(v):
        a = int(v)
        rest = round((v - a) * 10)
        return [float(a), float(rest)]
    return [v]


def _split_packed_position(v: float) -> tuple[float, float]:
    a = int(v)
    frac = v - a
    one_digit = round(frac * 10)
    two_digit = round(frac * 100)
    two_digit_valid = MIN_HUI <= two_digit <= MAX_HUI
    one_digit_valid = MIN_HUI <= one_digit <= MAX_HUI
    if two_digit_valid and not one_digit_valid:
        return float(a), float(two_digit)
    return float(a), float(one_digit)


def _explode(row) -> list[tuple]:
    degrees = _values(row["degree"], True)
    ranges = _values(row["range"], True)
    strings = _values(row["string"], True)
    positions = _values(row["position"], False)
    timbre_cell = row["L tech (timbre)"]
    timbres = [None] if pd.isna(timbre_cell) else _values(timbre_cell, True)

    n = max(map(len, (degrees, ranges, strings, timbres)))

    """Count-consistency rule for the ambiguous position column:
    if the row holds n>1 notes, but position parsed to a single fractional (e.g. '8.7'),
    it should be comma, '8.7' means positions (8, 7), one per note. 
    
    Internal evidence: both comma and period appear (e.g. '7,6.5') - both notes share a string 
    (two different degrees cannot sound at the same position on the same string). 
    will be flagged suspect=True"""
    suspect = False
    if n > 1 and len(positions) == 1 and positions[0] != int(positions[0]):
        positions = list(_split_packed_position(positions[0]))
        suspect = True

    n = max(n, len(positions))

    # Length-consistency check: number of values need to be consistent.
    lengths = {"degree": len(degrees), "range": len(ranges),
               "string": len(strings), "position": len(positions)}
    packed_lengths = {name: l for name, l in lengths.items() if l > 1}
    if packed_lengths and len(set(packed_lengths.values())) > 1:
        raise PackedLengthMismatch(
            f"inconsistent packed field lengths {lengths} "
            f"(degree={row['degree']!r} range={row['range']!r} "
            f"string={row['string']!r} position={row['position']!r})"
        )

    def pad(lst):
        return lst + [lst[-1]] * (n - len(lst))

    multi = n > 1
    return [
        (int(d), int(r), int(s), p, None if t is None else int(t), multi, suspect)
        for d, r, s, p, t in zip(pad(degrees), pad(ranges), pad(strings),
                                 pad(positions), pad(timbres))
    ]


def load_piece(path: Path) -> list[Event]:
    out: list[Event] = []
    for sheet_name, df in pd.read_excel(path, sheet_name=None).items():
        if not set(KEEP) <= set(df.columns):
            continue
        rows = df[KEEP].dropna(subset=["degree", "string", "position"])
        events: list[Event] = []
        for _, row in rows.iterrows():
            try:
                parts = _explode(row)
            except PackedLengthMismatch as exc:
                print(f"[{path.stem}/{sheet_name}] skipping row: {exc}")
                continue
            except (ValueError, TypeError):
                continue  # legend junk that slipped through
            onset = float(row["onset"]) if not pd.isna(row["onset"]) else -1.0
            for deg, rng, st, pos, tim, multi, suspect in parts:
                kind = TIMBRE.get(tim, "?")
                if pos == 0.0 and kind == "?":
                    kind = "open"
                events.append(Event(path.stem, sheet_name, len(events),
                                    deg, rng, st, pos, kind, onset, multi,
                                    suspect))
        out.extend(events)
    return out


def load_all(root: Path) -> list[Event]:
    events: list[Event] = []
    for f in sorted(Path(root).glob("*.xlsx")):
        if f.name.startswith(("~", "0_")):
            continue
        events.extend(load_piece(f))
    return events
