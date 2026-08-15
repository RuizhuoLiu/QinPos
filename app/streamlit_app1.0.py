"""QinPos — jianpu in, skeleton 减字谱 out.

    streamlit run app/streamlit_app.py

Everything here is UI. The pipeline is `qinpos.jianpu` -> `qinpos.infer` ->
`qinpos.viz` / `qinpos.jianzipu`, all pure functions, so this file adds no
modelling logic of its own and nothing in it needs to be tested twice.
"""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qinpos import jianzipu as jz  # noqa: E402
from qinpos import viz  # noqa: E402
from qinpos.infer import fingering_rows, predict, timbre_mix  # noqa: E402
from qinpos.jianpu import (parse_jianpu, range_report,  # noqa: E402
                           suggest_header)
from qinpos.theory import HUI_FRACTIONS  # noqa: E402

WEIGHTS = ROOT / "data/crf_weights.json"
PRESET_DIR = ROOT / "data"
MAX_VIDEO_FRAMES = 400

st.set_page_config(page_title="QinPos", page_icon="🎼", layout="wide")


# --------------------------------------------------------------------------
# cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_weights():
    """Trained CRF weights, or the hand-crafted defaults with a warning."""
    import json

    from qinpos.learn import WeightVector
    from qinpos.viterbi import FEATURES, Weights

    if WEIGHTS.exists():
        return WeightVector(json.load(WEIGHTS.open())), True
    hand = Weights()
    return WeightVector({k: getattr(hand, k, 0.0) for k in FEATURES}), False


@st.cache_resource(show_spinner=False)
def load_glyphs():
    try:
        assets = jz.load_assets()
        return assets, assets.check()
    except jz.MissingAssets as exc:
        return None, [str(exc)]


@st.cache_data(show_spinner=False)
def run_model(text: str, open_bias: float, harm_bias: float):
    """Parse and decode. Cached on the inputs so sliders stay responsive."""
    score = parse_jianpu(text)
    if score.unplayable() or not score.notes:
        return score, None
    w_base, _ = load_weights()
    w_view = w_base.biased(open_bias=-open_bias, harmonic_bias=-harm_bias)
    baseline = w_base if (open_bias or harm_bias) else None
    return score, predict(score.notes, w_view, kinds=score.kinds, baseline_w=baseline)


@st.cache_data(show_spinner=False, max_entries=8)
def rasterise(svg: str, fmt: str, scale: float) -> bytes:
    """SVG -> PNG/JPEG bytes. Cached, because rasterising a 200-glyph score
    takes seconds and a Streamlit rerun happens on every widget touch."""
    if fmt == "JPG":
        return viz.svg_to_jpg_bytes(svg, scale)
    return viz.svg_to_png_bytes(svg, scale)


PRESET_ORDER = ["ciou01", "shenglvqimeng", "qingpingyue", "buran"]


def presets() -> dict[str, str]:
    """Scores available in the picker, labelled by their own `title:` header.

    Only real scores from `data/*.jianpu`. The synthetic demo phrase that used
    to sit at the top is deliberately gone: it was invented to exercise the
    parser, and a made-up melody presented next to real repertoire is a claim
    the project cannot back.

    Ordering is deliberate: a GQ39 export first when one exists, because its
    pitches come from the same loader the model trains on and it carries an
    expert annotation to compare against; then the transcribed pieces in
    increasing order of how far they push the model — pentatonic and sitting
    on the open strings, then 偏音 at 5%, then 偏音 at 8% over 339 notes.
    """
    out: dict[str, str] = {}
    found = {q.stem: q for q in PRESET_DIR.glob("*.jianpu")}
    for stem in PRESET_ORDER + sorted(set(found) - set(PRESET_ORDER)):
        path = found.get(stem)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^\s*title\s*[:=]\s*(.+)$", text, re.M | re.I)
        n = len(parse_jianpu(text).notes)
        label = (title.group(1).strip() if title else stem) + f" — {n} notes"
        out[label] = text
    out["(blank — type your own)"] = "gong_string: 3\n\n"
    return out


def preset_diagnostics() -> str | None:
    """Explain an empty picker instead of silently showing two entries.

    Deliberately NOT cached: the whole point is to reflect what is on disk
    right now, and a cached directory listing is how a file you just added
    stays invisible until you restart the server.
    """
    if any(PRESET_DIR.glob("*.jianpu")):
        return None
    # With no scores on disk the picker holds only the blank entry, so this
    # message is the only thing telling the user where scores come from.
    if not PRESET_DIR.is_dir():
        return f"No `{PRESET_DIR}` directory — scores are loaded from there."
    others = sorted(q.name for q in PRESET_DIR.iterdir() if q.is_file())[:8]
    listing = ", ".join(f"`{n}`" for n in others) or "(empty)"
    return (f"No `*.jianpu` files in `{PRESET_DIR}`. It currently holds: "
            f"{listing}. Note that browsers often append `.txt` on download.")


def show_svg(svg: str, height: int | None = None) -> None:
    """Display an SVG, trying the methods most likely to survive in order.

    `st.image` takes an SVG string directly and hands it to the frontend as an
    image, which is the only path here that does not go through the HTML
    sanitiser or share a DOM with the app's own styles. The fallbacks exist
    because this has to work on whatever Streamlit version is installed, and a
    blank panel is the worst possible failure: it looks like the model produced
    nothing.
    """
    try:
        st.image(svg, width="stretch")
        return
    except Exception:
        pass
    try:
        st.components.v1.html(
            f"<div style='overflow:auto;width:100%'>{svg}</div>",
            height=height or 520, scrolling=True,
        )
        return
    except Exception:
        pass
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    st.html(
        f"<div style='overflow:auto;width:100%'>"
        f"<img src='data:image/svg+xml;base64,{b64}' "
        f"style='display:block;max-width:100%;height:auto'/></div>"
    )


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

st.sidebar.title("QinPos")
st.sidebar.caption(
    "Numbered notation in, fingering out: 弦位 · 徽位 · 音色. "
    "Fingering for either hand is outside the model's scope and is drawn as "
    "empty slots rather than guessed."
)

_w, _trained = load_weights()
if not _trained:
    st.sidebar.warning(
        f"No trained weights at `{WEIGHTS.name}` — using the hand-crafted "
        f"defaults. Run `scripts/train_crf.py` for real results."
    )

_assets, _missing = load_glyphs()
if _assets is None:
    st.sidebar.error(
        "Glyph components missing. Run `python scripts/fetch_jianzipu_assets.py`."
    )
elif _missing:
    st.sidebar.warning(f"{len(_missing)} glyph components missing; re-run the fetcher.")

st.sidebar.subheader("Difficulty")
open_bias = st.sidebar.slider(
    "散音 preference", -3.0, 3.0, 0.0, 0.25,
    help="Right = more open strings = easier. Left = more stopped notes.",
)
harm_bias = st.sidebar.slider(
    "泛音 preference", -3.0, 3.0, 0.0, 0.25,
    help="Right = more harmonics.",
)

st.sidebar.subheader("Tablature layout")
orientation_label = st.sidebar.selectbox(
    "Reading order",
    ["horizontal, left to right",
     "vertical, right to left (traditional)",
     "vertical, left to right"],
)
orientation = "horizontal" if orientation_label.startswith("horizontal") else "vertical"
right_to_left = "right to left" in orientation_label
show_labels = st.sidebar.checkbox(
    "Print the jianpu above each glyph", value=True,
    help="Numbered notation and tablature on one page, the way modern 琴谱 are set.",
)
glyph_width = st.sidebar.slider("Glyph size", 60, 160, 95, 5)
per_line = st.sidebar.slider("Glyphs per line", 4, 20, 10)


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

st.title("Jianpu → skeleton 减字谱")

_presets = presets()
col_pick, col_help = st.columns([1, 2])
with col_pick:
    _diag = preset_diagnostics()
    if _diag:
        st.warning(_diag)
    choice = st.selectbox(
        "Load a score", list(_presets),
        help="Pick one to see the model work, or choose blank and type your own.",
    )
with col_help:
    with st.expander("Notation format"):
        st.markdown(
            "- `1`–`7` degrees; unmarked octave holds 三弦散音 (1) to 七弦散音 (6)\n"
            "- `'` up an octave, `,` down (repeatable: `1''`, `5,`)\n"
            "- `#` `b` accidentals · `0` rest · `-` extends the previous note\n"
            "- `^` `o` `p` force 泛音 / 散音 / 按音 for one note\n"
            "- `|` bar line · `//` comment, at the start of a line or after the music\n"
            "- `xN` at the END of a line repeats that line N times (ostinati)\n"
            "- headers: `title:` `key:` `gong_string:` `transpose:`\n\n"
            "`key:` is decorative — the model works in semitones above the "
            "open 1st string, so absolute pitch never enters a decision. "
            "`gong_string` (3 = 正调) is what places the tune on the instrument."
        )

text = st.text_area("Jianpu", value=_presets[choice], height=220, key=f"src_{choice}")

score, pred = run_model(text, open_bias, harm_bias)
rng = range_report(score)

# ---- diagnostics ----
if score.errors:
    for e in score.errors:
        st.error(e)
unplayable = score.unplayable()
if unplayable:
    st.error(f"{len(unplayable)} notes cannot be played in 正调:")
    for i, n, why in unplayable[:8]:
        st.write(f"- note {i + 1} ({n.semitones:+.0f} semitones): {why}")
    fix = suggest_header(score)
    if fix:
        st.info("Paste these lines at the top of the score and it fits:")
        st.code(fix)
    else:
        st.info(
            "No choice of 宫 string or octave shift fits this melody: its span "
            "is wider than the instrument. Split it or move a phrase by an octave."
        )
    st.stop()
if not score.notes:
    st.info("Type or load a melody to begin.")
    st.stop()

mix = timbre_mix(pred.path)
low_conf = sum(p < 0.5 for p in pred.confidence)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Notes", rng["n"])
m2.metric("Range", f"{rng['low_label']} … {rng['high_label']}")
m3.metric("按 / 散 / 泛", f"{mix['stopped']} / {mix['open']} / {mix['harmonic']}")
m4.metric("Ambiguous (P<0.5)", f"{low_conf}", help="Notes with no clearly best fingering")

if score.warnings:
    st.caption(" · ".join(score.warnings))
if pred.baseline_path is not None:
    st.caption(
        f"{sum(pred.changed_vs_baseline)} of {len(pred.path)} notes moved "
        f"since the sliders were at zero."
    )


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

tab_score, tab_note, tab_table, tab_export = st.tabs(
    ["Tablature", "Fingerboard", "Table", "Export"]
)

tablature_svg = None
with tab_score:
    if _assets is None:
        st.warning("Glyph components not installed.")
    else:
        st.caption(
            "Each glyph carries 弦序 below and 徽位 above, with 散 or the 泛音 "
            "marker where the timbre calls for it. The dashed boxes are 左手指法 "
            "and 右手指法 — not predicted, drawn empty. 散音 has no 左手 box "
            "because an open string genuinely has no left hand."
        )
        tablature_svg = jz.render_score(
            pred.path, assets=_assets, glyph_width=glyph_width,
            per_column=per_line, orientation=orientation,
            right_to_left=right_to_left,
            labels=([t.raw for t in score.note_tokens()] if show_labels else None),
        )
        _rows = (len(pred.path) + per_line - 1) // per_line
        _h = int(min(2400, 160 + _rows * glyph_width * (2.6 if show_labels else 2.2)))
        show_svg(tablature_svg, height=_h)

with tab_note:
    i = st.slider("Note", 1, len(pred.path), 1) - 1
    tok = score.note_tokens()[i]
    panel, panel_w = None, 0.0
    if _assets is not None:
        panel_w = 280.0
        panel = ("<rect width='280' height='454' fill='#faf6ec' rx='10'/>"
                 + jz.glyph_group(pred.path[i], assets=_assets, x=52, y=34,
                                  scale=176 / jz.CELL[2]))
    if not pred.marginals[i]:
        st.error(f"Note {i + 1} has no candidates — this should be impossible; "
                 f"please report the melody that caused it.")
        st.stop()
    frame = viz.render_frame(
        pred.marginals, i, hui_fractions=HUI_FRACTIONS,
        baselines=pred.baseline_path, changed=pred.changed_vs_baseline,
        side_panel=panel, side_panel_width=panel_w,
        title=f"note {i + 1} / {len(pred.path)}",
        subtitle=f"jianpu {tok.raw} · {tok.semitones:+.0f} semitones above 一弦散",
    )
    show_svg(frame)
    st.caption(
        "Brightness is the marginal probability of each candidate: one bright "
        "dot means the model is confident, several half-lit dots mean the note "
        "is genuinely ambiguous — which is where a human player also has a choice."
    )
    st.write("**Top candidates**")
    for cand, prob in pred.top_k(i, 5):
        st.write(f"- {jz.spec_for(cand).reading} — {prob:.0%}")

with tab_table:
    st.dataframe(fingering_rows(pred, score.note_tokens()),
                 width="stretch", hide_index=True)

with tab_export:
    stem = (score.meta.get("title") or "melody").replace(" ", "_")[:40]

    st.subheader("Score")
    if tablature_svg:
        st.caption(
            "SVG stays sharp at any size and is the one to put in a thesis. "
            "PNG and JPG are the ones to paste into slides, Word or a chat — "
            "they need `cairosvg` installed."
        )
        i1, i2, i3 = st.columns([1, 1, 2])
        img_fmt = i1.radio("Format", ["PNG", "JPG"], horizontal=True,
                           label_visibility="collapsed")
        img_scale = i2.select_slider("Resolution", [1.0, 1.5, 2.0, 3.0, 4.0], 2.0,
                                     format_func=lambda v: f"{v:g}×")
        with i3:
            st.write("")
            prepare = st.button(f"Prepare {img_fmt}", width="stretch")

        b1, b2 = st.columns(2)
        b1.download_button("⬇ Tablature (SVG)", tablature_svg.encode("utf-8"),
                           f"{stem}_jianzipu.svg", "image/svg+xml", width="stretch")
        if prepare:
            try:
                with st.spinner(f"rendering {img_fmt} at {img_scale:g}×"):
                    data = rasterise(tablature_svg, img_fmt, img_scale)
                ext = img_fmt.lower()
                b2.download_button(
                    f"⬇ Tablature ({img_fmt}, {len(data) // 1024} KB)", data,
                    f"{stem}_jianzipu.{ext}", f"image/{'jpeg' if ext == 'jpg' else 'png'}",
                    width="stretch", type="primary")
            except Exception as exc:
                b2.error(f"{exc}")

    st.subheader("Data")
    c2, c3 = st.columns(2)
    import csv as _csv

    _rows = fingering_rows(pred, score.note_tokens())
    _buf = io.StringIO()
    _wtr = _csv.DictWriter(_buf, fieldnames=list(_rows[0]))
    _wtr.writeheader()
    _wtr.writerows(_rows)
    c2.download_button("⬇ Fingering (CSV)", _buf.getvalue().encode("utf-8-sig"),
                       f"{stem}_fingering.csv", "text/csv", width="stretch")
    c3.download_button("⬇ Jianpu source", text.encode("utf-8"),
                       f"{stem}.jianpu", "text/plain", width="stretch")

    st.subheader("Video")
    st.caption(
        "One frame per note: fingerboard on the left, the glyph for that note "
        "on the right, playhead moving through the confidence profile. "
        "The HTML player needs nothing installed; mp4 and gif need "
        "`cairosvg` and `imageio-ffmpeg`."
    )
    vc1, vc2, vc3 = st.columns(3)
    fps = vc1.slider("fps", 0.5, 8.0, 2.0, 0.5)
    fmt = vc2.selectbox("Format", ["html (no dependencies)", "mp4", "gif"])
    scale = vc3.slider("Raster scale", 1.0, 3.0, 2.0, 0.5,
                       disabled=fmt.startswith("html"))

    n_frames = min(len(pred.path), MAX_VIDEO_FRAMES)
    if n_frames < len(pred.path):
        st.caption(f"Capped at the first {MAX_VIDEO_FRAMES} notes.")

    if st.button("Render video", type="primary"):
        progress = st.progress(0.0, "rendering frames")
        frames = []
        for k in range(n_frames):
            side = None
            if _assets is not None:
                side = ("<rect width='280' height='454' fill='#faf6ec' rx='10'/>"
                        + jz.glyph_group(pred.path[k], assets=_assets, x=52, y=34,
                                         scale=176 / jz.CELL[2]))
            frames.append(viz.render_frame(
                pred.marginals, k, hui_fractions=HUI_FRACTIONS,
                baselines=pred.baseline_path, changed=pred.changed_vs_baseline,
                side_panel=side, side_panel_width=0.0 if side is None else 280.0,
                cjk=False,
                title=f"{stem} — note {k + 1} / {len(pred.path)}",
                subtitle=f"jianpu {score.note_tokens()[k].raw}",
            ))
            if k % 10 == 0:
                progress.progress(k / max(n_frames, 1), "rendering frames")

        progress.progress(1.0, "encoding")
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as d:
                if fmt.startswith("html"):
                    path = viz.export_html_player(
                        frames, Path(d) / f"{stem}.html", fps=fps, title=stem)
                    mime = "text/html"
                elif fmt == "mp4":
                    path = viz.export_video(
                        frames, Path(d) / f"{stem}.mp4", fps=fps, scale=scale)
                    mime = "video/mp4" if path.suffix == ".mp4" else "image/gif"
                else:
                    path = viz.export_gif(
                        frames, Path(d) / f"{stem}.gif", fps=fps, scale=scale)
                    mime = "image/gif"
                data = path.read_bytes()
            progress.empty()
            if mime == "video/mp4":
                st.video(data)
            st.download_button(f"⬇ {path.name}", data, path.name, mime,
                               type="primary")
        except Exception as exc:
            progress.empty()
            st.error(f"Encoding failed: {exc}")
