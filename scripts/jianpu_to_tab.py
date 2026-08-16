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

    Paste any melody in default tune 正调 and the trained CRF assigns string弦 / hui徽位 / timbre音色 to
    every note.

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
    | `//` | comment — at the start of a line, or after the music on it |
    | `xN` | at the END of a line, repeats that whole line N times |
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

    from qinpos import jianzipu as jz
    from qinpos import viz
    from qinpos.infer import (Prediction, describe, fingering_rows, predict,
                              timbre_mix)
    from qinpos.jianpu import (EXAMPLE, parse_jianpu, range_report,
                               suggest_header)
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

    try:
        jz_assets = jz.load_assets()
        _missing = jz_assets.check()
        if _missing:
            print(f"{len(_missing)} glyph components missing -- re-run "
                  f"scripts/fetch_jianzipu_assets.py")
    except jz.MissingAssets as _exc:
        jz_assets = None
        print(f"tablature disabled: {_exc}")
    return (
        EXAMPLE,
        EXPORTS,
        HUI_FRACTIONS,
        describe,
        fingering_rows,
        jz,
        jz_assets,
        parse_jianpu,
        predict,
        range_report,
        suggest_header,
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
def _(mo, parse_jianpu, range_report, source, suggest_header):
    score = parse_jianpu(source.value)
    unplayable = score.unplayable()
    _rng = range_report(score)

    _lines = [f"Parsed **{len(score.notes)}** notes."]
    if score.meta.get("title"):
        _lines.append(f"Title: {score.meta['title']}")
    if _rng.get("n"):
        _lines.append(
            f"Range **{_rng['low_label']} … {_rng['high_label']}** "
            f"({_rng['span_semitones']:.0f} semitones), "
            f"gong宫 on string {score.gong_string}."
        )
    _lines += [f"- !! {e}" for e in score.errors]
    _lines += [f"- {w}" for w in score.warnings]
    for _i, _n, _why in unplayable[:8]:
        _lines.append(f"- !! note {_i + 1} ({_n.semitones:+.0f} semitones): {_why}")
    if len(unplayable) > 8:
        _lines.append(f"- … and {len(unplayable) - 8} more")

    # Only two moves keep a tune on a default tune正调 instrument: whole octaves, or gong宫 on a different open string (借调). If one of them rescues the piece, show the header lines to paste rather than silently fixing it behind the scenes.
    _fix = suggest_header(score)
    if _fix:
        _lines.append("")
        _lines.append("**This fits if you paste these lines at the top:**")
        _lines.append(f"```\n{_fix}\n```")
    mo.md("\n".join(_lines))
    return score, unplayable


@app.cell
def _(mo):
    open_bias = mo.ui.slider(
        -3.0, 3.0, step=0.25, value=0.0,
        label="open散音 preference (right = more open strings = easier)",
    )
    harm_bias = mo.ui.slider(
        -3.0, 3.0, step=0.25, value=0.0,
        label="harmonic泛音 preference (right = more harmonics)",
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
def _(
    HUI_FRACTIONS,
    cjk_labels,
    describe,
    jz,
    jz_assets,
    mo,
    note_slider,
    pred,
    score,
    viz,
):
    _i = note_slider.value
    _tok = score.note_tokens()[_i]
    _base = pred.baseline_path[_i] if pred.baseline_path else None

    # The glyph for this note, rendered as a panel beside the board. The empty
    # dashed boxes are the two fingering slots the system does not predict.
    _panel, _panel_w = None, 0.0
    if jz_assets is not None:
        _panel_w = 280.0
        _panel = (f"<rect width='{_panel_w:.0f}' height='454' fill='#faf6ec' rx='10'/>"
                  + jz.glyph_group(pred.path[_i], assets=jz_assets,
                                   x=52, y=34, scale=176 / jz.CELL[2]))

    _svg = viz.render_frame(
        pred.marginals, _i,
        hui_fractions=HUI_FRACTIONS,
        baselines=pred.baseline_path,
        changed=pred.changed_vs_baseline,
        side_panel=_panel,
        side_panel_width=_panel_w,
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
            f"按 {_mix['stopped']} · open散 {_mix['open']} · harmonic泛 {_mix['harmonic']}"
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
    ### Skeleton jianzipu减字谱

    Columns read top to bottom, right to left. Each glyph carries the three
    things the model predicts — string弦序 below, hui徽位 above, open/harmonic 散/泛 where the timbre
    calls for it. The two dashed boxes are left hand左手指法 and right hand 右手指法: outside the
    system's scope, drawn empty rather than guessed. open散音 has no left hand左手 box
    because an open string genuinely has no left hand.
    """)
    return


@app.cell
def _(mo):
    score_width = mo.ui.slider(70, 160, step=5, value=105, label="glyph size")
    score_percol = mo.ui.slider(4, 16, value=9, label="glyphs per line")
    score_labels = mo.ui.checkbox(value=True, label="print the jianpu above each glyph")
    score_colour = mo.ui.checkbox(value=True, label="colour glyphs by timbre")
    score_flow = mo.ui.dropdown(
        options={
            "horizontal, left to right": ("horizontal", True),
            "vertical, right to left (traditional)": ("vertical", True),
            "vertical, left to right": ("vertical", False),
        },
        value="horizontal, left to right",
        label="Layout",
    )
    mo.hstack([score_width, score_percol, score_flow, score_labels, score_colour],
              justify="start", gap=2)
    return score_colour, score_flow, score_labels, score_percol, score_width


@app.cell(hide_code=True)
def _(
    jz,
    jz_assets,
    mo,
    note_slider,
    pred,
    score,
    score_colour,
    score_flow,
    score_labels,
    score_percol,
    score_width,
):
    mo.stop(jz_assets is None,
            mo.md("*Run `scripts/fetch_jianzipu_assets.py` to enable tablature.*"))
    _orientation, _rtl = score_flow.value
    tablature_svg = jz.render_score(
        pred.path,
        assets=jz_assets,
        glyph_width=score_width.value,
        per_column=score_percol.value,
        orientation=_orientation,
        right_to_left=_rtl,
        labels=([t.raw for t in score.note_tokens()] if score_labels.value else None),
        colour_timbre=score_colour.value,
        highlight=note_slider.value,
    )
    mo.Html(tablature_svg)
    return (tablature_svg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Export

    Same renderer as `fingerboard_en.py`, so a clip made here matches the one
    made from a GQ39 piece. The CSV is the input to the tablature layer:
    string弦 / hui徽位 / timbre音色 is the complete output of this system. Right-hand and
    left-hand technique are outside its scope and are not inferred anywhere
    in the package, so the tablature layer renders those slots empty.
    """)
    return


@app.cell(hide_code=True)
def _(mo, viz):
    # mp4 needs a rasteriser AND an encoder; say so before the button is pressed rather than after a minute of rendering.
    import shutil

    _raster = "cairosvg"
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        _raster = next((e for e in ("resvg", "rsvg-convert", "inkscape")
                        if shutil.which(e)), None)
    _ffmpeg = viz._ffmpeg() is not None
    _notes = []
    if _raster is None:
        _notes.append("no SVG rasteriser — `uv add cairosvg` (mp4/gif/png unavailable)")
    if not _ffmpeg:
        _notes.append("no ffmpeg — `uv add imageio-ffmpeg`, or mp4 falls back to gif")
    mo.md("!! " + " · ".join(_notes) if _notes
          else f"Encoders ready: {_raster} + ffmpeg.")
    return


@app.cell
def _(mo):
    export_fmt = mo.ui.multiselect(
        options=["mp4", "gif", "html", "csv", "svg (current note)",
                 "tablature svg", "tablature png"],
        value=["mp4", "html", "tablature svg"],
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
    jz,
    jz_assets,
    mo,
    note_slider,
    pred,
    score,
    tablature_svg,
    viz,
):
    mo.stop(not export_go.value, mo.md("*Press **Export** to render.*"))

    _stem = (score.meta.get("title") or "melody").replace(" ", "_")[:40]
    _toks = score.note_tokens()

    def _panel(k):
        if jz_assets is None:
            return None
        return (f"<rect width='280' height='454' fill='#faf6ec' rx='10'/>"
                + jz.glyph_group(pred.path[k], assets=jz_assets,
                                 x=52, y=34, scale=176 / jz.CELL[2]))

    _frames = [
        viz.render_frame(
            pred.marginals, _k,
            hui_fractions=HUI_FRACTIONS,
            baselines=pred.baseline_path,
            changed=pred.changed_vs_baseline,
            side_panel=_panel(_k),
            side_panel_width=0.0 if jz_assets is None else 280.0,
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
            elif _fmt == "tablature svg":
                _written.append(viz.export_svg(
                    tablature_svg, EXPORTS / f"{_stem}_jianzipu.svg"))
            elif _fmt == "tablature png":
                _p = EXPORTS / f"{_stem}_jianzipu.png"
                _p.write_bytes(viz.svg_to_png_bytes(tablature_svg, scale=2.0))
                _written.append(_p)
        except Exception as _exc:
            _errors.append(f"- `{_fmt}`: {_exc}")

    _mimes = {".mp4": "video/mp4", ".gif": "image/gif", ".html": "text/html",
              ".svg": "image/svg+xml", ".png": "image/png", ".csv": "text/csv"}
    _buttons = [
        mo.download(data=p.read_bytes(), filename=p.name,
                    mimetype=_mimes.get(p.suffix, "application/octet-stream"),
                    label=f"⬇ {p.name}")
        for p in _written if p.is_file()
    ]

    _msg = [f"Rendered **{len(_frames)}** frames into `{EXPORTS}`."]
    if _errors:
        _msg += ["", "**Failed:**"] + _errors
    mo.vstack([mo.md("\n".join(_msg)),
               mo.hstack(_buttons, justify="start", gap=1, wrap=True)]
              if _buttons else mo.md("\n".join(_msg)))
    return


if __name__ == "__main__":
    app.run()
