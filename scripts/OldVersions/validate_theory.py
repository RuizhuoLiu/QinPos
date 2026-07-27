import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    from collections import defaultdict
    from pathlib import Path

    from qinpos.dataset_gq39 import load_all
    from qinpos.theory import (DEGREE_SEMITONES, OPEN_STRING_SEMITONES,
                               huifen_to_lambda)

    return (
        DEGREE_SEMITONES,
        OPEN_STRING_SEMITONES,
        Path,
        defaultdict,
        huifen_to_lambda,
        load_all,
        math,
    )


@app.cell
def _(Path, load_all):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root/"data/GQ39/score_annotation"
    print(data_dir, data_dir.exists())
    events = load_all(data_dir)
    print(f"loaded {len(events)} events")
    stopped = [e for e in events if e.kind == "stopped" and 1.0 <= e.position <= 13.9]
    print(f"{len(stopped)} stopped notes")
    return (stopped,)


@app.cell
def _(
    DEGREE_SEMITONES,
    OPEN_STRING_SEMITONES,
    defaultdict,
    huifen_to_lambda,
    math,
    stopped,
):
    offsets = defaultdict(list)
    for e in stopped:
        lam = huifen_to_lambda(e.position)
        physical = OPEN_STRING_SEMITONES[e.string] + 12 * math.log2(1/lam)
        notated = DEGREE_SEMITONES[e.degree] + 12 * e.range_
        offsets[e.piece].append(physical - notated)
    return (offsets,)


@app.cell
def _(offsets):
    agree = tot = 0
    for piece, offs in sorted(offsets.items()):
        offs.sort()
        med = offs[len(offs) // 2]
        ok = sum(abs(o - med) <= 0.6 for o in offs)
        agree += ok
        tot += len(offs)
        print(f"{piece:20s} n={len(offs):3d}  offset={med:7.2f}  agree={ok/len(offs):5.1%}")
    print(f"\nOVERALL: {agree}/{tot} = {agree/tot:.1%}")
    return


if __name__ == "__main__":
    app.run()
