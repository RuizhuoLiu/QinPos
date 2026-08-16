"""Parse numbered notation (简谱) into the decoder's Note sequence.

The model works in *relative* pitch space (semitones above the open 1st
string) so the absolute key of the score is irrelevant to the fingering
decision. What matters is where gong宫 sits, which in at5正调 is the open 3rd string.
A `1=F` header is therefore accepted and remembered for display, but it does
not change a single decision.

Text format
-----------
Header lines (optional, `key: value`, anywhere before the music)::

    key: 1=F                # decorative only, recorded for the caption
    gong_string: 3          # which open string is 宫; 3 = 正调 default
    transpose: 0            # extra semitone shift applied to every note
    title: 秋风词

``//`` starts a comment, at the beginning of a line or after the music on it.
Everything else is music, whitespace separated::

    5, 6, | 1 2 3 - | ^5 ^6 | o1 2 #4 |

Token grammar (in this order)::

    [timbre] [accidental] digit [octave marks] [duration marks]

    timbre      ^ or 泛  -> force 泛音      (harmonic)
                o or 散  -> force 散音      (open string)
                p or 按  -> force 按音      (stopped)
                (absent) -> let the model choose, which is the interesting case
    accidental  # sharp, b flat (repeatable: ## = whole tone up)
    digit       0-7. 0 is a rest.
    octave      ' up an octave, , down an octave (repeatable: 1'' , 5,,)
    duration    _ halves the value (repeatable), . dots it
    standalone  -  extends the previous note by one beat
                |  bar line, ignored
                xN at the END of a line repeats that whole line N times,
                   e.g. ``6, 5, 6, 1 | 7, 1 7, 2 x12`` for an ostinato
                :| |: repeat marks, ignored (the melody is NOT expanded)

Unmarked octave holds 三弦散音 (1) up to 七弦散音 (6), i.e. the seven open
strings of 正调 are exactly ``5, 6, 1 2 3 5 6`` -- the same convention
`learn._notated`: "at5".

Duration is parsed and carried on the Note, but no feature in `viterbi.py`
reads it: it exists for the fingering table and for future works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .candidates import candidates_for
from .theory import DEGREE_SEMITONES, OPEN_STRING_SEMITONES, Note

__all__ = ["Token", "ParsedScore", "parse_jianpu", "EXAMPLE", "format_pitch",
           "range_report", "fit_options", "suggest_header"]

TIMBRE_PREFIX = {
    "^": "harmonic", "泛": "harmonic",
    "o": "open", "O": "open", "散": "open",
    "p": "stopped", "P": "stopped", "按": "stopped",
}

_TOKEN_RE = re.compile(
    r"^(?P<timbre>[\^泛oO散pP按])?"
    r"(?P<accidental>[#b＃♭]*)"
    r"(?P<degree>[0-7])"
    r"(?P<octave>['’,，]*)"
    r"(?P<duration>[_.]*)$"
)


@dataclass
class Token:
    """One parsed unit of the source text, music or not."""

    raw: str
    kind: str                      # 'note' | 'rest' | 'hold' | 'bar' | 'unknown'
    line: int
    degree: int | None = None
    octave: int = 0
    accidental: int = 0
    duration: float = 1.0
    timbre: str | None = None      # 'open' | 'stopped' | 'harmonic' | None
    semitones: float | None = None
    note_index: int | None = None  # index into ParsedScore.notes, if it is one


@dataclass
class ParsedScore:
    notes: list[Note] = field(default_factory=list)
    kinds: list[str | None] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def gong_string(self) -> int:
        return int(self.meta.get("gong_string", 3))

    @property
    def gong_semitones(self) -> int:
        return OPEN_STRING_SEMITONES[self.gong_string]

    def note_tokens(self) -> list[Token]:
        return [t for t in self.tokens if t.kind == "note"]

    def unplayable(self) -> list[tuple[int, Note, str]]:
        """Notes the lattice cannot express, with a diagnosis each."""
        out = []
        for i, n in enumerate(self.notes):
            if candidates_for(n):
                continue
            if n.semitones < 0:
                why = ("below the open 1st string (the instrument's floor); "
                       "raise this note an octave or transpose the piece up")
            elif n.is_harmonic and not candidates_for(Note(n.semitones)):
                why = "no harmonic泛音 on any string matches this pitch"
            elif n.is_harmonic:
                why = "no harmonic泛音 node produces this pitch; let the model choose the timbre"
            else:
                why = ("above the highest practical stopped position "
                       "(hui 2 on the 7th string); lower it an octave")
            out.append((i, n, why))
        return out


def format_pitch(semitones: float, gong: int = 5) -> str:
    """Relative semitones back to a jianpu-ish label, for tables and captions."""
    rel = semitones - gong
    octave = int(rel // 12)
    within = rel - 12 * octave
    names = {v: k for k, v in DEGREE_SEMITONES.items()}
    base = names.get(round(within))
    if base is None:  # semitone 偏音 that is not a diatonic degree: describe it as sharp
        lower = max((s for s in names if s <= within), default=0)
        base = f"#{names[lower]}"
    marks = "'" * octave if octave > 0 else "," * (-octave)
    return f"{base}{marks}"


def parse_jianpu(text: str) -> ParsedScore:
    """Parse a jianpu string. Never raises: problems land in .errors/.warnings."""
    score = ParsedScore()
    body_lines: list[tuple[int, str]] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # `//` starts a comment anywhere on the line, not only at the start:
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        head = re.match(r"^([A-Za-z_]+)\s*[:=]\s*(.+)$", line)
        if head:
            score.meta[head.group(1).strip().lower()] = head.group(2).strip()
            continue
        body_lines.append((lineno, line))

    gong_string = 3
    if "gong_string" in score.meta:
        try:
            gong_string = int(score.meta["gong_string"])
        except ValueError:
            score.errors.append(f"gong_string must be an integer 1-7, got {score.meta['gong_string']!r}")
        if gong_string not in OPEN_STRING_SEMITONES:
            score.errors.append(f"gong_string {gong_string} is not a string; use 1-7")
            gong_string = 3
    score.meta["gong_string"] = str(gong_string)
    gong = OPEN_STRING_SEMITONES[gong_string]

    transpose = 0.0
    if "transpose" in score.meta:
        try:
            transpose = float(score.meta["transpose"])
        except ValueError:
            score.errors.append(f"transpose must be a number, got {score.meta['transpose']!r}")

    last_note_token: Token | None = None

    for lineno, line in body_lines:
        tokens = line.split()
        # A trailing xN repeats the line, for ostinati written once.
        if tokens:
            rep = re.match(r"^[x\u00d7](\d+)$", tokens[-1], re.I)
            if rep:
                count = int(rep.group(1))
                if count < 1 or count > 200:
                    score.errors.append(
                        f"line {lineno}: repeat count {count} out of range (1-200)")
                    count = 1
                tokens = tokens[:-1] * count

        for raw in tokens:
            if raw in {"|", "||", "|:", ":|", ":||", "‖"}:
                score.tokens.append(Token(raw, "bar", lineno))
                continue
            if raw == "-":
                token = Token(raw, "hold", lineno)
                score.tokens.append(token)
                if last_note_token is None:
                    score.warnings.append(f"line {lineno}: '-' before any note, ignored")
                else:
                    last_note_token.duration += 1.0
                    idx = last_note_token.note_index
                    if idx is not None:
                        old = score.notes[idx]
                        score.notes[idx] = Note(old.semitones, old.duration + 1.0, old.is_harmonic)
                continue

            m = _TOKEN_RE.match(raw)
            if not m:
                score.tokens.append(Token(raw, "unknown", lineno))
                score.errors.append(
                    f"line {lineno}: cannot parse {raw!r} in {line!r} "
                    f"(use // for comments, one note per token)")
                continue

            degree = int(m.group("degree"))
            octave = (m.group("octave").count("'") + m.group("octave").count("’")
                      - m.group("octave").count(",") - m.group("octave").count("，"))
            accidental = (m.group("accidental").count("#") + m.group("accidental").count("＃")
                          - m.group("accidental").count("b") - m.group("accidental").count("♭"))
            duration = 1.0
            for ch in m.group("duration"):
                if ch == "_":
                    duration /= 2.0
                elif ch == ".":
                    duration *= 1.5
            timbre = TIMBRE_PREFIX.get(m.group("timbre") or "")

            if degree == 0:
                score.tokens.append(Token(raw, "rest", lineno, duration=duration))
                continue

            semitones = (gong + DEGREE_SEMITONES[degree] + 12 * octave
                         + accidental + transpose)
            is_harmonic = {"harmonic": True, "open": False, "stopped": False}.get(timbre)
            note = Note(semitones=float(semitones), duration=duration, is_harmonic=is_harmonic)

            token = Token(raw, "note", lineno, degree, octave, accidental, duration,
                          timbre, float(semitones), len(score.notes))
            score.tokens.append(token)
            score.notes.append(note)
            score.kinds.append(timbre)
            last_note_token = token

    if any(t.kind == "rest" for t in score.tokens):
        score.warnings.append(
            "rests are dropped from the sequence: the chain model has no rest state, "
            "so the notes either side of a rest are scored as adjacent"
        )
    if not score.notes and not score.errors:
        score.errors.append("no notes found")
    return score


EXAMPLE = """\
// A random demo phrase, not a transcription of any published score.
// (scripts/gq39_to_jianpu.py) when the notes themselves need to be right.
title: demo phrase (random)
key: 1=F
gong_string: 3

6, 1 2 | 3 - 2 1 | 6, 1 2 3 | 5 - - - |
^5 ^6 ^1' | ^6 ^5 - - |
2 3 5 6 | 1' 6 5 3 | 2 - 1 - |
"""


# fitting an arbitrary tune onto the instrument


def _relative_degrees(score: ParsedScore) -> list[tuple[float, bool | None]]:
    """Each note as (semitones above 宫, is_harmonic), independent of placement.

    This is what stays fixed when you move 宫 to another string or shift the
    tune by octaves, so it is the right basis for asking "where does this
    melody fit".
    """
    gong = score.gong_semitones
    try:
        transpose = float(score.meta.get("transpose", 0))
    except ValueError:
        transpose = 0.0
    return [(n.semitones - gong - transpose, n.is_harmonic) for n in score.notes]


def range_report(score: ParsedScore) -> dict:
    """Span of the melody against what the instrument can actually reach."""
    if not score.notes:
        return {"n": 0}
    pitches = [n.semitones for n in score.notes]
    return {
        "n": len(pitches),
        "low": min(pitches),
        "high": max(pitches),
        "span_semitones": max(pitches) - min(pitches),
        "low_label": format_pitch(min(pitches), score.gong_semitones),
        "high_label": format_pitch(max(pitches), score.gong_semitones),
        "n_unplayable": len(score.unplayable()),
    }


def fit_options(
    score: ParsedScore,
    transposes: Iterable[int] = (-24, -12, 0, 12, 24),
    gong_strings: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
) -> list[dict]:
    """Which (gong_string, transpose) settings make the whole tune playable."""
    rel = _relative_degrees(score)
    current_gong = score.gong_string
    try:
        current_t = int(float(score.meta.get("transpose", 0)))
    except ValueError:
        current_t = 0

    rows = []
    for g in gong_strings:
        base = OPEN_STRING_SEMITONES[g]
        for t in transposes:
            bad = 0
            pitches = []
            for offset, harm in rel:
                pitch = base + offset + t
                pitches.append(pitch)
                if not candidates_for(Note(pitch, is_harmonic=harm)):
                    bad += 1
            rows.append({
                "gong_string": g,
                "transpose": t,
                "n_unplayable": bad,
                "low": min(pitches),
                "high": max(pitches),
                "current": g == current_gong and t == current_t,
            })
    rows.sort(key=lambda r: (r["n_unplayable"], abs(r["transpose"]),
                             abs(r["gong_string"] - current_gong)))
    return rows


def suggest_header(score: ParsedScore) -> str | None:
    """Header lines that would make the tune playable, or None if it already is."""
    if not score.notes or not score.unplayable():
        return None
    best = fit_options(score)[0]
    if best["n_unplayable"]:
        return None
    lines = [f"gong_string: {best['gong_string']}"]
    if best["transpose"]:
        lines.append(f"transpose: {best['transpose']}")
    return "\n".join(lines)
