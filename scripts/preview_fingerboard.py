#!/usr/bin/env python3
"""Render a fingerboard preview to a file, with no Streamlit involved.

Point of this script: when nothing shows up in the app, it isolates whether
the problem is the pipeline or the UI. If this writes a file you can open and
see, the model and renderer are fine and the fault is in Streamlit; if it
fails here, the traceback names the real cause.

    python scripts/preview_fingerboard.py
    python scripts/preview_fingerboard.py --jianpu data/shenglvqimeng.jianpu
    python scripts/preview_fingerboard.py --note 12 --open-bias 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qinpos import jianzipu as jz  # noqa: E402
from qinpos import viz  # noqa: E402
from qinpos.infer import describe, predict  # noqa: E402
from qinpos.jianpu import EXAMPLE, parse_jianpu, range_report  # noqa: E402
from qinpos.theory import HUI_FRACTIONS  # noqa: E402


def load_weights():
    from qinpos.learn import WeightVector
    from qinpos.viterbi import FEATURES, Weights

    path = ROOT / "data/crf_weights.json"
    if path.exists():
        return WeightVector(json.load(path.open())), f"trained weights ({path.name})"
    hand = Weights()
    return (WeightVector({k: getattr(hand, k, 0.0) for k in FEATURES}),
            "HAND-CRAFTED defaults — run scripts/train_crf.py for real results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jianpu", type=Path, help="a .jianpu file (default: built-in example)")
    ap.add_argument("--note", type=int, default=1, help="which note to draw (1-based)")
    ap.add_argument("--open-bias", type=float, default=0.0)
    ap.add_argument("--harm-bias", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=ROOT / "exports/preview.html")
    args = ap.parse_args()

    text = args.jianpu.read_text(encoding="utf-8") if args.jianpu else EXAMPLE
    score = parse_jianpu(text)
    print(f"parsed {len(score.notes)} notes  ·  {range_report(score)}")
    for e in score.errors:
        print(f"  ERROR {e}")
    for i, n, why in score.unplayable():
        print(f"  UNPLAYABLE note {i + 1} ({n.semitones:+.0f}): {why}")
    if not score.notes or score.unplayable():
        print("\nnothing to draw until the score parses and fits.")
        return 1

    w_base, note = load_weights()
    print(f"weights: {note}")
    w_view = w_base.biased(open_bias=-args.open_bias, harmonic_bias=-args.harm_bias)
    pred = predict(score.notes, w_view, kinds=score.kinds, baseline_w=w_base)

    i = max(0, min(args.note - 1, len(pred.path) - 1))
    print(f"note {i + 1}: {describe(pred.path[i])}  P={pred.confidence[i]:.2f}")

    panel, panel_w = None, 0.0
    try:
        assets = jz.load_assets()
        panel_w = 280.0
        panel = ("<rect width='280' height='454' fill='#faf6ec' rx='10'/>"
                 + jz.glyph_group(pred.path[i], assets=assets, x=52, y=34,
                                  scale=176 / jz.CELL[2]))
        tablature = jz.render_score(
            pred.path, assets=assets, glyph_width=95, per_column=10,
            orientation="horizontal",
            labels=[t.raw for t in score.note_tokens()],
        )
    except jz.MissingAssets as exc:
        print(f"  (no glyphs: {exc})")
        tablature = "<p>Run scripts/fetch_jianzipu_assets.py for tablature.</p>"

    frame = viz.render_frame(
        pred.marginals, i, hui_fractions=HUI_FRACTIONS,
        baselines=pred.baseline_path, changed=pred.changed_vs_baseline,
        side_panel=panel, side_panel_width=panel_w,
        title=f"note {i + 1} / {len(pred.path)}",
        subtitle=f"jianpu {score.note_tokens()[i].raw}",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<body style='background:#15100a;color:#f4ecd8;font-family:sans-serif;padding:20px'>"
        f"<h2>Fingerboard</h2>{frame}<h2>Tablature</h2>{tablature}</body>",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}  — open it in a browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
