import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from qinpos.dataset_gq39 import load_all
from qinpos.theory import (DEGREE_SEMITONES, OPEN_STRING_SEMITONES,
                           huifen_to_lambda)

events = load_all(Path("data/GQ39/score_annotation"))
stopped = [e for e in events if e.kind == "stopped" and 1.0 <= e.position <= 13.9]

offsets = defaultdict(list)
for e in stopped:
    lam = huifen_to_lambda(e.position)
    physical = OPEN_STRING_SEMITONES[e.string] + 12 * math.log2(1 / lam)
    notated = DEGREE_SEMITONES[e.degree] + 12 * e.range_
    offsets[e.piece].append(physical - notated)

agree = tot = 0
for piece, offs in sorted(offsets.items()):
    offs.sort()
    med = offs[len(offs) // 2]
    ok = sum(abs(o - med) <= 0.6 for o in offs)
    agree += ok
    tot += len(offs)
    print(f"{piece:20s} n={len(offs):3d}  offset={med:7.2f}  agree={ok/len(offs):5.1%}")
print(f"\nOVERALL: {agree}/{tot} = {agree/tot:.1%}")
