import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Jianpu → fingering

    Paste any melody in 正调 and the trained CRF assigns 弦 / 徽位 / 音色 to
    every note. Unlike `fingerboard_en.py` this needs no annotation, so it is
    the path towards the Streamlit app — and the honest test of whether the
    weights generalise past GQ39.

    **Format** — whitespace separated, `//` starts a comment line:

    | token | meaning |
    | --- | --- |
    | `1` … `7` | jianpu degree; unmarked octave holds 三弦散音 (1) to 七弦散音 (6) |
    | `0` | rest (dropped: the chain model has no rest state) |
    | `-` | extends the previous note |
    | `'` `,` | octave up / down, repeatable (`1''`, `5,`) |
    | `#` `b` | sharp / flat, e.g. `#4` |
    | `^` `o` `p` | force 泛音 / 散音 / 按音 for that note; omit to let the model decide |
    | `\|` | bar line, ignored |
    | `key:` `gong_string:` `transpose:` `title:` | header lines |

    `key: 1=F` is decorative. The model works in semitones above the open 1st
    string, so absolute pitch never enters a decision — only where 宫 sits,
    which `gong_string` sets (3 = 正调).
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos import viz
    from qinpos.infer import (Prediction, describe, fingering_rows, predict,
                              timbre_mix)
    from qinpos.jianpu import EXAMPLE, parse_jianpu
    from qinpos.learn import WeightVector
    from qinpos.theory import HUI_FRACTIONS

    ROOT = Path(__file__).resolve().parent.parent
    WFILE = ROOT / "data/crf_weights.json"
    EXPORTS = ROOT / "exports"

    if not WFILE.exists():
        raise FileNotFoundError(
            f"{WFILE} not found -- run scripts/train_crf.py first; this "
            f"notebook does no training of its own."
        )
    w_base = WeightVector(json.load(WFILE.open()))
    return (
        EXAMPLE,
        EXPORTS,
        HUI_FRACTIONS,
        describe,
        fingering_rows,
        parse_jianpu,
        predict,
        timbre_mix,
        viz,
        w_base,
    )


@app.cell
def _(EXAMPLE, mo):
    source = mo.ui.text_area(
        value=EXAMPLE, label="Jianpu", rows=10, full_width=True
    )
    source
    return (source,)


@app.cell
def _(mo, parse_jianpu, source):
    score = parse_jianpu(source.value)
    unplayable = score.unplayable()

    _lines = [f"Parsed **{len(score.notes)}** notes."]
    if score.meta.get("title"):
        _lines.append(f"Title: {score.meta['title']}")
    _lines += [f"- !! {e}" for e in score.errors]
    _lines += [f"- {w}" for w in score.warnings]
    for _i, _n, _why in unplayable:
        _lines.append(f"- !! note {_i + 1} ({_n.semitones:+.0f} semitones): {_why}")
    mo.md("\n".join(_lines))
    return score, unplayable


@app.cell
def _(mo):
    open_bias = mo.ui.slider(
        -3.0, 3.0, step=0.25, value=0.0,
        label="散音 open preference (right = more open strings = easier)",
    )
    harm_bias = mo.ui.slider(
        -3.0, 3.0, step=0.25, value=0.0,
        label="泛音 harmonics preference (right = more harmonics)",
    )
    cjk_labels = mo.ui.checkbox(value=True, label="Chinese labels")
    mo.vstack([
        mo.hstack([open_bias, harm_bias], justify="start", gap=2),
        cjk_labels,
    ])
    return cjk_labels, harm_bias, open_bias


@app.cell
def _(harm_bias, mo, open_bias, predict, score, unplayable, w_base):
    mo.stop(
        bool(unplayable) or not score.notes,
        mo.md("**Fix the ranges above before decoding.**"),
    )
    # Negative cost = cheaper = chosen more often, so the slider is negated.
    w_view = w_base.biased(
        open_bias=-open_bias.value, harmonic_bias=-harm_bias.value
    )
    pred = predict(score.notes, w_view, kinds=score.kinds, baseline_w=w_base)
    return (pred,)


@app.cell
def _(mo, pred):
    note_slider = mo.ui.slider(
        0, len(pred.path) - 1, value=0, label=f"Note (of {len(pred.path)})"
    )
    note_slider
    return (note_slider,)


@app.cell(hide_code=True)
def _(HUI_FRACTIONS, cjk_labels, describe, mo, note_slider, pred, score, viz):
    _i = note_slider.value
    _tok = score.note_tokens()[_i]
    _base = pred.baseline_path[_i] if pred.baseline_path else None

    _svg = viz.render_frame(
        pred.marginals, _i,
        hui_fractions=HUI_FRACTIONS,
        baselines=pred.baseline_path,
        changed=pred.changed_vs_baseline,
        cjk=cjk_labels.value,
        title=f"{score.meta.get('title', 'untitled')} — note {_i + 1} / {len(pred.path)}",
        subtitle=f"jianpu {_tok.raw}  ·  {_tok.semitones:+.0f} semitones above 一弦散",
    )
    _caption = (
        f"**{describe(pred.path[_i])}** (P={pred.confidence[_i]:.2f})"
        + ("" if _base is None or _base == pred.path[_i]
           else f" · moved from **{describe(_base)}** at bias 0")
    )
    mo.vstack([mo.Html(_svg), mo.md(_caption)])
    return


@app.cell(hide_code=True)
def _(fingering_rows, mo, pred, score, timbre_mix):
    _rows = fingering_rows(pred, score.note_tokens())
    _mix = timbre_mix(pred.path)
    _changed = sum(pred.changed_vs_baseline)
    _low = sum(p < 0.5 for p in pred.confidence)
    mo.vstack([
        mo.md(
            f"按 {_mix['stopped']} · 散 {_mix['open']} · 泛 {_mix['harmonic']}"
            f"  ·  {_low}/{len(pred.path)} notes below P=0.5"
            f"  ·  {_changed} moved since bias 0"
        ),
        mo.ui.table(_rows, selection=None, page_size=25),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Export

    Same renderer as `fingerboard_en.py`, so a clip made here matches the one
    made from a GQ39 piece. The CSV is the input to the tablature layer:
    弦 / 徽位 / 音色 is everything the model predicts, and a 减字谱 glyph also
    needs a right-hand finger and a left-hand finger, which it does not.
    """)
    return


@app.cell
def _(mo):
    export_fmt = mo.ui.multiselect(
        options=["mp4", "gif", "html", "csv", "svg (current note)"],
        value=["html", "csv"],
        label="Formats",
    )
    export_fps = mo.ui.slider(0.5, 8.0, step=0.5, value=2.0, label="fps")
    export_cjk = mo.ui.checkbox(value=False, label="Chinese labels in the export")
    export_go = mo.ui.run_button(label="Export")
    mo.hstack([export_fmt, export_fps, export_cjk, export_go], justify="start", gap=2)
    return export_cjk, export_fmt, export_fps, export_go


@app.cell
def _(
    EXPORTS,
    HUI_FRACTIONS,
    export_cjk,
    export_fmt,
    export_fps,
    export_go,
    fingering_rows,
    mo,
    note_slider,
    pred,
    score,
    viz,
):
    mo.stop(not export_go.value, mo.md("*Press **Export** to render.*"))

    _stem = (score.meta.get("title") or "melody").replace(" ", "_")[:40]
    _toks = score.note_tokens()
    _frames = [
        viz.render_frame(
            pred.marginals, _k,
            hui_fractions=HUI_FRACTIONS,
            baselines=pred.baseline_path,
            changed=pred.changed_vs_baseline,
            cjk=export_cjk.value,
            title=f"{_stem} — note {_k + 1} / {len(pred.path)}",
            subtitle=f"jianpu {_toks[_k].raw}",
        )
        for _k in range(len(pred.path))
    ]

    EXPORTS.mkdir(parents=True, exist_ok=True)
    _written, _errors = [], []
    for _fmt in export_fmt.value:
        try:
            if _fmt == "mp4":
                _written.append(viz.export_video(
                    _frames, EXPORTS / f"{_stem}.mp4", fps=export_fps.value))
            elif _fmt == "gif":
                _written.append(viz.export_gif(
                    _frames, EXPORTS / f"{_stem}.gif", fps=export_fps.value))
            elif _fmt == "html":
                _written.append(viz.export_html_player(
                    _frames, EXPORTS / f"{_stem}.html", fps=export_fps.value,
                    title=f"QinPos — {_stem}"))
            elif _fmt == "csv":
                import csv as _csv
                _rows = fingering_rows(pred, _toks)
                _path = EXPORTS / f"{_stem}_fingering.csv"
                with _path.open("w", newline="", encoding="utf-8-sig") as _fh:
                    _wtr = _csv.DictWriter(_fh, fieldnames=list(_rows[0]))
                    _wtr.writeheader()
                    _wtr.writerows(_rows)
                _written.append(_path)
            elif _fmt == "svg (current note)":
                _written.append(viz.export_svg(
                    _frames[note_slider.value],
                    EXPORTS / f"{_stem}_note{note_slider.value + 1}.svg"))
        except Exception as _exc:
            _errors.append(f"- `{_fmt}`: {_exc}")

    _msg = [f"Rendered **{len(_frames)}** frames."]
    _msg += [f"- wrote `{p}`" for p in _written]
    if _errors:
        _msg += ["", "**Failed:**"] + _errors
    mo.md("\n".join(_msg))
    return


if __name__ == "__main__":
    app.run()
