"""Phase 1: enumerate all playable (string, position, kind) realisations
of a pitch. Viterbi state space.

For any pitch there are at most:
    7 stopped candidates (one deterministic hui.fen per string, if reachable)
  + up to 7 open candidates (exact unison only)
  + harmonic candidates (only at the 13 hui, only if pitch matches a partial)
"""

from __future__ import annotations

from .theory import OPEN_STRING_SEMITONES, Candidate, Note, harmonic_positions, stopped_position

# Practical playability limits for stopped notes: below hui ~2 the
# vibrating string is too short/tense to press cleanly; beyond hui 13
# (十三徽外) is used but we cap interpolation at the nut.
STOPPED_MIN_HUI = 2.0
STOPPED_MAX_HUI = 13.9

PITCH_TOL = 0.6  # semitone tolerance when matching open/harmonic pitches


def candidates_for(note: Note) -> list[Candidate]:
    out: list[Candidate] = []
    for s in range(1, 8):
        interval = note.semitones - OPEN_STRING_SEMITONES[s]

        # open string (散音)
        if abs(interval) <= PITCH_TOL and note.is_harmonic is not True:
            out.append(Candidate(s, 0.0, "open"))

        # stopped (按音)
        if note.is_harmonic is not True:
            pos = stopped_position(s, note.semitones)
            if pos is not None and STOPPED_MIN_HUI <= pos <= STOPPED_MAX_HUI:
                out.append(Candidate(s, round(pos, 2), "stopped"))

        # harmonic (泛音)
        if note.is_harmonic is not False:
            for h in harmonic_positions(s, note.semitones, PITCH_TOL):
                out.append(Candidate(s, h, "harmonic"))
    return out
