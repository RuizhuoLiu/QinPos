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
    # 琴面可视化 — CRF marginal heatmap on a physically-true fingerboard

    Every hui mark below sits at its REAL string-length fraction
    (`theory.HUI_FRACTIONS`): hui 7 at 1/2, hui 9 at 2/3, and so on —
    a qin player will recognise the geometry immediately.

    For the selected note, every playable candidate lights up with
    opacity proportional to its CRF marginal probability
    (`crf.note_marginals`): one bright dot = the model is confident;
    several half-lit dots = genuinely ambiguous, which is exactly where
    a human expert would also have choices. The expert's annotated
    realisation is drawn as a dashed ring for comparison.

    The two bias sliders reweight 散音/泛音 at INFERENCE time
    (`WeightVector.biased`), i.e. the "difficulty" control: fewer open
    strings and harmonics generally means a harder, more left-hand
    fingering. Marginals recompute live.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos.crf import note_marginals, train_crf
    from qinpos.learn import WeightVector, build_sequences
    from qinpos.theory import HUI_FRACTIONS

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    CLEAN = ROOT / "data/gq39_clean.csv"
    WFILE = ROOT / "data/crf_weights.json"

    seqs = build_sequences(DATA, CLEAN)

    if WFILE.exists():
        w_base = WeightVector(json.load(WFILE.open()))
        print(f"loaded trained weights from {WFILE}")
    else:
        print(f"{WFILE} missing -> quick-training a fallback (about a minute)...")
        _tr = [s for i, s in enumerate(seqs) if i % 4]
        w_base = train_crf(_tr, epochs=5, lr=0.5, l2=1e-4, verbose=False)
        json.dump(dict(w_base), WFILE.open("w"))
    return HUI_FRACTIONS, note_marginals, seqs, w_base


@app.cell
def _(mo, seqs):
    piece_pick = mo.ui.dropdown(
        options={s.piece: i for i, s in enumerate(seqs)},
        value=seqs[0].piece,
        label="曲目",
    )
    open_bias = mo.ui.slider(-2.0, 2.0, step=0.25, value=0.0,
                             label="散音偏好 (右= 更多散音, 更易)")
    harm_bias = mo.ui.slider(-2.0, 2.0, step=0.25, value=0.0,
                             label="泛音偏好 (右= 更多泛音)")
    mo.hstack([piece_pick, open_bias, harm_bias], justify="start", gap=2)
    return harm_bias, open_bias, piece_pick


@app.cell
def _(harm_bias, note_marginals, open_bias, piece_pick, seqs, w_base):
    seq = seqs[piece_pick.value]
    # positive slider = prefer that timbre = NEGATIVE added cost
    w_view = w_base.biased(open_bias=-open_bias.value,
                           harmonic_bias=-harm_bias.value)
    margs = note_marginals(seq.notes, w_view)
    return margs, seq


@app.cell
def _(margs, mo, seq):
    note_slider = mo.ui.slider(0, len(seq.notes) - 1, value=0,
                               label=f"音符 (共 {len(seq.notes)})")
    note_slider
    return (note_slider,)


@app.cell(hide_code=True)
def _(HUI_FRACTIONS, margs, mo, note_slider, seq):
    # ---------------- geometry ----------------
    W, H = 980, 330
    X0, X1 = 70, 930           # 龙龈 (nut) .. 岳山 (yueshan)
    YS = [58 + 34 * s for s in range(7)]  # y of strings 1..7 (top=1)

    def hui_x(frac: float) -> float:
        # frac = length fraction measured FROM the yueshan
        return X1 - (X1 - X0) * frac

    def pos_x(position: float) -> float:
        h = int(position)
        fen = position - h
        f0 = float(HUI_FRACTIONS.get(h, 1.0)) if h >= 1 else 1.0
        f1 = float(HUI_FRACTIONS.get(h + 1, 1.0)) if h + 1 <= 13 else 1.0
        return hui_x(f0 + (f1 - f0) * fen)

    i = note_slider.value
    m = margs[i]
    top = max(m.values())
    argmax = max(m, key=m.get)
    expert = seq.expert[i] if i < len(seq.expert) else None

    ZH = "一二三四五六七八九十".ljust(0)
    HUI_LABEL = {1:"一",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",
                 8:"八",9:"九",10:"十",11:"十一",12:"十二",13:"十三"}

    parts = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:#2b1d0e;border-radius:12px;font-family:serif">']
    # board
    parts.append(f'<rect x="{X0-18}" y="34" width="{X1-X0+36}" height="{H-70}" '
                 f'rx="10" fill="#5a3d1e"/>')
    # yueshan and nut
    parts.append(f'<rect x="{X1+6}" y="34" width="8" height="{H-70}" fill="#d9c9a3"/>')
    parts.append(f'<rect x="{X0-14}" y="34" width="5" height="{H-70}" fill="#caa96b"/>')
    # hui inlays with real fractions
    for h, fr in HUI_FRACTIONS.items():
        x = hui_x(float(fr))
        parts.append(f'<circle cx="{x:.1f}" cy="44" r="5" fill="#efe6cf"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-18}" fill="#cbb686" font-size="12" '
                     f'text-anchor="middle">{HUI_LABEL[h]}</text>')
    # strings (1 thickest)
    for s in range(1, 8):
        yy = YS[s - 1]
        parts.append(f'<line x1="{X0-14}" y1="{yy}" x2="{X1+6}" y2="{yy}" '
                     f'stroke="#e8ddc2" stroke-width="{3.4 - 0.35*(s-1):.2f}"/>')
        parts.append(f'<text x="{X0-40}" y="{yy+4}" fill="#cbb686" font-size="13">'
                     f'{s}弦</text>')

    # candidates, dimmest first so bright ones draw on top
    for c, p in sorted(m.items(), key=lambda kv: kv[1]):
        yy = YS[c.string - 1]
        op = 0.15 + 0.85 * p
        hot = (c is argmax)
        if c.kind == "open":
            x = X1 - 6
            parts.append(f'<path d="M {x-9} {yy} L {x} {yy-9} L {x+9} {yy} L {x} {yy+9} Z" '
                         f'fill="#7fd0ff" opacity="{op:.2f}"'
                         f'{" stroke=\'#ffffff\' stroke-width=\'2\'" if hot else ""}/>')
        elif c.kind == "harmonic":
            x = pos_x(c.position)
            parts.append(f'<circle cx="{x:.1f}" cy="{yy}" r="10" fill="none" '
                         f'stroke="#b7ff9e" stroke-width="3" opacity="{op:.2f}"/>')
            if hot:
                parts.append(f'<circle cx="{x:.1f}" cy="{yy}" r="14" fill="none" '
                             f'stroke="#ffffff" stroke-width="1.5"/>')
        else:
            x = pos_x(c.position)
            parts.append(f'<circle cx="{x:.1f}" cy="{yy}" r="9" fill="#ffb347" '
                         f'opacity="{op:.2f}"'
                         f'{" stroke=\'#ffffff\' stroke-width=\'2\'" if hot else ""}/>')
        if p >= 0.10:
            lx = (X1 - 6) if c.kind == "open" else pos_x(c.position)
            parts.append(f'<text x="{lx:.1f}" y="{yy-14}" fill="#f4ecd8" '
                         f'font-size="11" text-anchor="middle">{p:.2f}</text>')

    # expert reference: dashed ring
    if expert is not None:
        yy = YS[expert.string - 1]
        ex = (X1 - 6) if expert.kind == "open" else pos_x(expert.position)
        parts.append(f'<circle cx="{ex:.1f}" cy="{yy}" r="17" fill="none" '
                     f'stroke="#ff6b6b" stroke-width="2" stroke-dasharray="5,4"/>')

    parts.append("</svg>")
    svg = "".join(parts)

    _conf = ("确定" if top > 0.8 else ("多解" if top < 0.5 else "倾向"))
    _kindzh = {"open": "散", "stopped": "按", "harmonic": "泛"}
    caption = (f"**音符 {i}** · 模型首选 **{argmax}** (P={top:.2f}, {_conf})"
               + (f" · 专家: **{expert}** (红色虚线圈)" if expert is not None else "")
               + f" · 图例: 橙实心=按音 / 绿空心=泛音 / 蓝菱形=散音, 亮度=边缘概率")
    mo.vstack([mo.Html(svg), mo.md(caption)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 整曲置信剖面 — 亮条 = 模型确定, 暗条 = 真实多解处 (也正是最值得请琴人评议的音)""")
    return


@app.cell(hide_code=True)
def _(margs, mo, note_slider, seq):
    _W = 980
    _n = len(margs)
    _bw = max(3.0, min(12.0, (_W - 20) / max(_n, 1)))
    _parts = [f'<svg width="{_W}" height="70" xmlns="http://www.w3.org/2000/svg" '
              f'style="background:#20160b;border-radius:8px">']
    for _j, _m in enumerate(margs):
        _t = max(_m.values())
        _x = 10 + _j * _bw
        _hh = 8 + 44 * _t
        _fill = "#ffd27f" if _j != note_slider.value else "#ff6b6b"
        _parts.append(f'<rect x="{_x:.1f}" y="{62-_hh:.1f}" width="{_bw*0.8:.1f}" '
                      f'height="{_hh:.1f}" fill="{_fill}" opacity="{0.35+0.65*_t:.2f}"/>')
    _parts.append("</svg>")
    mo.vstack([mo.Html("".join(_parts)),
               mo.md(f"曲目 **{seq.piece}**: "
                     f"{sum(max(m.values())>0.8 for m in margs)}/{_n} 音确定 (P>0.8), "
                     f"{sum(max(m.values())<0.5 for m in margs)}/{_n} 音多解 (P<0.5)")])
    return


if __name__ == "__main__":
    app.run()
