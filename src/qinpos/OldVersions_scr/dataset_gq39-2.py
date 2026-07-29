"""Loader for the GQ39 score annotations (Huang, Liang, Wei and Su, 2020).

Columns consumed per event row:
    degree, range        -> melody pitch (supervision INPUT)
    string, position     -> expert choice (supervision TARGET)
    'L tech (timbre)'    -> 1 = harmonic, 2 = open, 3 = stopped

Messy realities handled:
    * Chords 撮 and multi-string sweeps 历/滚拂 pack several values in
      one cell, either as strings '2,3,5' or floats 3.6 (Excel turned a
      comma into a decimal point on integer columns). Exploded here
      into simultaneous events sharing an onset.
    * `position` mixes float / str / 0 (open).
    * Legend text in trailing columns is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TIMBRE = {1: "harmonic", 2: "open", 3: "stopped"}
KEEP = ["degree", "range", "onset", "string", "position", "L tech (timbre)"]


@dataclass
class Event:
    piece: str
    section: str
    idx: int
    degree: int
    range_: int
    string: int
    position: float
    kind: str            # open | stopped | harmonic | ?
    onset: float
    multi: bool = False    # part of a chord / sweep
    suspect: bool = False  # position was repaired from a period-packed cell


def _values(cell, integer_column: bool) -> list[float]:
    """Parse a cell into a list of numeric values.

    integer_column=True means the column semantically holds integers
    (degree, range, string, timbre), so a fractional value like 3.6 must
    be the packed pair (3, 6) — the annotator's comma became a period.
    This applies whether the cell arrived as a FLOAT (Excel numeric) or
    as TEXT ('3.6'): before this fix, text cells were only split on
    commas, so '6.1' fell through to float('6.1') and int() truncation
    silently DROPPED the second chord note.

    position is NOT such a column: 7.9 is a real hui.fen value there.
    Period-vs-comma disambiguation for position needs the multi-note
    context and happens in _explode, not here.
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


def _explode(row) -> list[tuple]:
    degrees = _values(row["degree"], True)
    ranges = _values(row["range"], True)
    strings = _values(row["string"], True)
    positions = _values(row["position"], False)
    timbre_cell = row["L tech (timbre)"]
    timbres = [None] if pd.isna(timbre_cell) else _values(timbre_cell, True)

    n = max(map(len, (degrees, ranges, strings, timbres)))

    # Count-consistency rule for the ambiguous position column:
    # if the OTHER columns say this row holds n>1 notes but position
    # parsed to a single value with a fractional part (e.g. '8.7'),
    # the period is almost certainly a mistyped comma — '8.7' means
    # positions (8, 7), one per note. Internal evidence: rows where the
    # annotator DID type the comma (e.g. '7,6.5') sit right next to
    # period rows in the same sheet, and the split reading is the only
    # physically possible one when both notes share a string (two
    # different degrees cannot sound at the same position on the same
    # string). Rows repaired this way are flagged suspect=True so the
    # cleaning stage can double-check them against the physics.
    suspect = False
    if n > 1 and len(positions) == 1 and positions[0] != int(positions[0]):
        v = positions[0]
        a = int(v)
        rest = round((v - a) * 10)
        positions = [float(a), float(rest)]
        suspect = True

    n = max(n, len(positions))

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
