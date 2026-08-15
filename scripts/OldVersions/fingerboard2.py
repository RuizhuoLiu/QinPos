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
    # Fingerboard viewer — CRF marginal heatmap on a physically true 琴面

    Every 徽 below sits at its real string-length fraction
    (`theory.HUI_FRACTIONS`): 徽七 at 1/2, 徽九 at 2/3, and so on, so a qin
    player recognises the geometry immediately.

    For the selected note every playable candidate lights up with opacity
    proportional to its CRF marginal probability (`crf.note_marginals`): one
    bright dot means the model is confident, several half-lit dots mean the
    note is genuinely ambiguous — which is exactly where a human expert would
    also have a choice. The annotated realisation is drawn as a dashed ring.

    The two bias sliders reweight 散音 / 泛音 at **inference** time
    (`WeightVector.biased`), i.e. a difficulty control: fewer open strings and
    harmonics generally means a harder, more left-hand-heavy fingering.
    Marginals recompute live.

    All drawing lives in `qinpos.viz`, so the export section at the bottom
    renders exactly what you see here — no second implementation to drift.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos import viz
    from qinpos.crf import note_marginals, train_crf
    from qinpos.learn import WeightVector, build_sequences
    from qinpos.theory import HUI_FRACTIONS

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    CLEAN = ROOT / "data/gq39_clean.csv"
    WFILE = ROOT / "data/crf_weights.json"
    EXPORTS = ROOT / "exports"

    seqs = build_sequences(DATA, CLEAN)

    if WFILE.exists():
        w_base = WeightVector(json.load(WFILE.open()))
        print(f"loaded trained weights from {WFILE}")
    else:
        print(f"{WFILE} missing -> quick-training a fallback (about a minute)...")
        _tr = [s for i, s in enumerate(seqs) if i % 4]
        w_base = train_crf(_tr, epochs=5, lr=0.5, l2=1e-4, verbose=False)
        json.dump(dict(w_base), WFILE.open("w"))
    return EXPORTS, HUI_FRACTIONS, note_marginals, seqs, viz, w_base


@app.function(hide_code=True)
def describe_note(note) -> str:
    """Best-effort one-line label for a note, whatever fields it carries."""
    for attr in ("pitch", "midi", "midi_pitch", "note", "degree"):
        value = getattr(note, attr, None)
        if value is not None:
            return f"{attr}={value}"
    return str(note)


@app.cell
def _(mo, seqs):
    piece_pick = mo.ui.dropdown(
        options={s.piece: i for i, s in enumerate(seqs)},
        value=seqs[0].piece,
        label="Piece 曲目",
    )
    open_bias = mo.ui.slider(
        -2.0, 2.0, step=0.25, value=0.0,
        label="散音 open bias (right = more open strings = easier)",
    )
    harm_bias = mo.ui.slider(
        -2.0, 2.0, step=0.25, value=0.0,
        label="泛音 harmonics bias (right = more harmonics)",
    )
    cjk_labels = mo.ui.checkbox(
        value=True, label="Chinese labels (uncheck for export without CJK fonts)"
    )
    mo.hstack([piece_pick, open_bias, harm_bias, cjk_labels], justify="start", gap=2)
    return cjk_labels, harm_bias, open_bias, piece_pick


@app.cell
def _(harm_bias, note_marginals, open_bias, piece_pick, seqs, w_base):
    seq = seqs[piece_pick.value]
    # positive slider = prefer that timbre = NEGATIVE added cost
    w_view = w_base.biased(
        open_bias=-open_bias.value, harmonic_bias=-harm_bias.value
    )
    margs = note_marginals(seq.notes, w_view)
    experts = list(seq.expert)
    return experts, margs, seq


@app.cell
def _(mo, seq):
    note_slider = mo.ui.slider(
        0, len(seq.notes) - 1, value=0, label=f"Note (of {len(seq.notes)})"
    )
    note_slider
    return (note_slider,)


@app.cell(hide_code=True)
def _(HUI_FRACTIONS, cjk_labels, experts, margs, mo, note_slider, seq, viz):
    i = note_slider.value
    m = margs[i]
    top = max(m.values())
    argmax = max(m, key=lambda c: m[c])
    expert = experts[i] if i < len(experts) else None

    svg = viz.render_fingerboard(
        m,
        hui_fractions=HUI_FRACTIONS,
        expert=expert,
        cjk=cjk_labels.value,
        title=f"{seq.piece} — note {i + 1} / {len(margs)}",
        subtitle=f"{describe_note(seq.notes[i])} · {len(m)} playable candidates",
    )

    _state = "confident" if top > 0.8 else ("ambiguous" if top < 0.5 else "leaning")
    caption = (
        f"**Note {i}** · model's choice **{argmax}** (P={top:.2f}, {_state})"
        + (f" · expert: **{expert}** (dashed red ring)" if expert is not None else "")
        + ("" if expert is None or argmax == expert else " — **mismatch**")
    )
    mo.vstack([mo.Html(svg), mo.md(caption)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Whole-piece confidence profile

    Bright bars = the model is sure. Dark bars = genuinely multi-solution
    notes, which are precisely the ones worth putting in front of a 琴人.
    Red bars = the model's top choice differs from the annotation.
    """)
    return


@app.cell(hide_code=True)
def _(cjk_labels, experts, margs, mo, note_slider, seq, viz):
    profile_svg = viz.render_confidence_profile(
        margs,
        experts=experts,
        cursor=note_slider.value,
        cjk=cjk_labels.value,
    )
    stats = viz.summarise(margs, experts)

    _rows = [
        ("notes", f"{stats['n_notes']}"),
        ("top-1 (exact match)", f"{stats.get('top1', float('nan')):.1%}"),
        ("top-3", f"{stats.get('top3', float('nan')):.1%}"),
        ("mean confidence", f"{stats['mean_confidence']:.2f}"),
        ("confident (P>0.8)", f"{stats['n_confident']} / {stats['n_notes']}"),
        ("ambiguous (P<0.5)", f"{stats['n_ambiguous']} / {stats['n_notes']}"),
        ("mean candidates / note", f"{stats['mean_candidates']:.1f}"),
    ]
    _table = "\n".join(f"| {k} | {v} |" for k, v in _rows)
    mo.vstack([
        mo.Html(profile_svg),
        mo.md(f"**{seq.piece}**\n\n| metric | value |\n| --- | --- |\n{_table}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ambiguity gallery

    The least confident notes in the piece, side by side. This is the shortlist
    to hand to the performer: "the model sees several defensible fingerings
    here — which would you actually play?"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    gallery_n = mo.ui.slider(2, 8, value=4, label="how many")
    return (gallery_n,)


@app.cell(hide_code=True)
def _(HUI_FRACTIONS, cjk_labels, experts, gallery_n, margs, mo, viz):
    _idx = viz.most_ambiguous(margs, gallery_n.value)
    _cards = []
    for _j in _idx:
        _e = experts[_j] if _j < len(experts) else None
        _cards.append(mo.Html(viz.render_fingerboard(
            margs[_j],
            hui_fractions=HUI_FRACTIONS,
            expert=_e,
            geom=viz.COMPACT,
            cjk=cjk_labels.value,
            title=f"note {_j + 1}  ·  P(top)={max(margs[_j].values()):.2f}",
            show_legend=False,
        )))
    mo.vstack([gallery_n] + _cards)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Export for presentations

    * **piece walkthrough** — one frame per note, board plus a playhead moving
      through the confidence profile. This is the "watch the model read the
      score" clip.
    * **bias sweep** — one note held fixed while 散音/泛音 bias runs from −2 to
      +2. Shows the fingering migrating between timbres, i.e. the difficulty
      knob, in about eight seconds.
    * **still figure** — the current frame as vector SVG for the dissertation.

    `mp4` needs ffmpeg (`brew install ffmpeg`) and an SVG rasteriser
    (`uv add cairosvg`, or `resvg` / `rsvg-convert` on PATH). `html` needs
    nothing at all and is the safe fallback if the venue laptop is not yours.
    """)
    return


@app.cell
def _(mo):
    export_mode = mo.ui.dropdown(
        options={
            "piece walkthrough (one frame per note)": "walk",
            "bias sweep (fixed note, 散音/泛音 −2 → +2)": "sweep",
        },
        value="piece walkthrough (one frame per note)",
        label="Mode",
    )
    export_fmt = mo.ui.multiselect(
        options=["mp4", "gif", "html", "png frames", "svg (current frame)"],
        value=["mp4", "html"],
        label="Formats",
    )
    export_fps = mo.ui.slider(0.5, 8.0, step=0.5, value=2.0, label="fps")
    export_scale = mo.ui.slider(1.0, 4.0, step=0.5, value=2.0, label="raster scale")
    export_cjk = mo.ui.checkbox(
        value=False, label="Chinese labels in the export (needs a CJK font installed)"
    )
    export_limit = mo.ui.slider(
        20, 600, step=20, value=200, label="max frames (walkthrough)"
    )
    export_go = mo.ui.run_button(label="Export")
    mo.vstack([
        mo.hstack([export_mode, export_fmt], justify="start", gap=2),
        mo.hstack([export_fps, export_scale, export_limit], justify="start", gap=2),
        mo.hstack([export_cjk, export_go], justify="start", gap=2),
    ])
    return (
        export_cjk,
        export_fmt,
        export_fps,
        export_go,
        export_limit,
        export_mode,
        export_scale,
    )


@app.cell
def _(
    EXPORTS,
    HUI_FRACTIONS,
    experts,
    export_cjk,
    export_fmt,
    export_fps,
    export_go,
    export_limit,
    export_mode,
    export_scale,
    harm_bias,
    margs,
    mo,
    note_marginals,
    note_slider,
    seq,
    viz,
    w_base,
):
    mo.stop(not export_go.value, mo.md("*Press **Export** to render.*"))

    _cjk = export_cjk.value
    _stem = f"{seq.piece}_{export_mode.value}"
    _frames: list[str] = []

    if export_mode.value == "walk":
        _n = min(len(margs), export_limit.value)
        for _k in range(_n):
            _frames.append(viz.render_frame(
                margs, _k,
                hui_fractions=HUI_FRACTIONS,
                experts=experts,
                cjk=_cjk,
                title=f"{seq.piece} — note {_k + 1} / {len(margs)}",
                subtitle=describe_note(seq.notes[_k]),
            ))
    else:
        _i = note_slider.value
        _steps = [round(-2.0 + 0.25 * _t, 2) for _t in range(17)]
        for _b in _steps:
            _w = w_base.biased(open_bias=-_b, harmonic_bias=-harm_bias.value)
            _mm = note_marginals(seq.notes, _w)
            _frames.append(viz.render_frame(
                _mm, _i,
                hui_fractions=HUI_FRACTIONS,
                experts=experts,
                cjk=_cjk,
                with_profile=False,
                title=f"{seq.piece} — note {_i + 1}: 散音 bias {_b:+.2f}",
                subtitle="inference-time reweighting, same trained model",
            ))

    EXPORTS.mkdir(parents=True, exist_ok=True)
    _written, _errors = [], []
    for _fmt in export_fmt.value:
        try:
            if _fmt == "mp4":
                _written.append(viz.export_video(
                    _frames, EXPORTS / f"{_stem}.mp4",
                    fps=export_fps.value, scale=export_scale.value))
            elif _fmt == "gif":
                _written.append(viz.export_gif(
                    _frames, EXPORTS / f"{_stem}.gif",
                    fps=export_fps.value, scale=min(export_scale.value, 2.0)))
            elif _fmt == "html":
                _written.append(viz.export_html_player(
                    _frames, EXPORTS / f"{_stem}.html",
                    fps=export_fps.value, title=f"QinPos — {seq.piece}"))
            elif _fmt == "png frames":
                _paths = viz.export_png_frames(
                    _frames, EXPORTS / _stem, scale=export_scale.value)
                _written.append(_paths[0].parent)
            elif _fmt == "svg (current frame)":
                _written.append(viz.export_svg(
                    _frames[min(note_slider.value, len(_frames) - 1)],
                    EXPORTS / f"{_stem}_note{note_slider.value}.svg"))
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
