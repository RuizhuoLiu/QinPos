"""Render QinPos predictions as skeleton 减字谱.

SCOPE. This system predicts 弦位, 徽位 and 音色. It does not predict either
hand's fingering, and nothing here fills those in by rule. A complete 减字
glyph carries four things -- 左手指法 + 徽位 on top, 右手指法 + 弦序 below --
and this renderer supplies exactly the two the model is responsible for,
drawing the other two slots as visible empty boxes. The gap in the output is
the scope boundary of the system, rendered honestly rather than papered over.

What survives the reduction is still unambiguous: a 徽位 in the upper slot
means 按音, 散 in that slot means 散音, and the 泛音 marker above the glyph
means 泛音. The model's three-way timbre decision maps onto components that
already exist in the notation; nothing is invented.

Glyph assets
------------
Components come from the JianZiPu font project by Nancy Yi Liang
(https://github.com/neuralfirings/JianZiPu), SIL Open Font License 1.1 with
Reserved Font Name "JianZiPu"; the surrounding build code there is MIT.

We do not build or load the font. Each component ships as a plain SVG path in
a 350x350 box, and `builder/inputs/layouts.json` gives explicit x/y/w/h for
every slot on a 1000-unit canvas, so composition is a matter of placing paths.
That means no FontForge, no ImageMagick, no OpenType ligature engine, and no
system font install -- the output is pure SVG that renders anywhere.

Run `scripts/fetch_jianzipu_assets.py` once to populate `assets/jianzipu/`.

Slot conventions established by reading the upstream data:
    area_string  <- md_{s}          full-size numeral, 弦序
    area_hui     <- md_{h}.blank    hui with no fen (upper half of the box)
                    md_{h}.{f}      hui.fen stacked, e.g. md_7.6 = 七徽六分
                    md_san          散, which occupies the hui slot because an
                                    open string has no hui position
    area_fy      <- mod_fanyin      泛音 marker above the glyph
    area_left    left-hand fingering   -- NOT PREDICTED, drawn empty
    area_right   right-hand fingering  -- NOT PREDICTED, drawn empty
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "Assets",
    "GlyphSpec",
    "MissingAssets",
    "spec_for",
    "glyph_group",
    "render_glyph",
    "render_score",
    "jianpu_label",
    "CELL",
    "REQUIRED_COMPONENTS",
    "ASSET_SOURCE",
]

ASSET_SOURCE = "https://github.com/neuralfirings/JianZiPu"

# Every glyph is composed inside this window of the upstream 1000-unit canvas.
# Bounds are the union of the slots used by layout_gou (area_fy reaches up to
# y=-230, area_right down to y=1000), plus margin, so a column of glyphs can
# use one fixed cell without per-glyph reflow.
CELL = (100.0, -280.0, 800.0, 1320.0)  # x, y, w, h

LAYOUT = "layout_gou"  # the only layout that reserves BOTH fingering slots

SLOT_LABEL = {"area_left": "左", "area_right": "右"}

# Darkened counterparts of the fingerboard colours in viz.PALETTE, so a
# timbre means the same colour in both views: amber = 按音, blue = 散音,
# green = 泛音. The fingerboard sits on near-black and these sit on cream,
# hence the different lightness for the same hue.
TIMBRE_INK = {
    "stopped": "#8a4b00",
    "open": "#12608f",
    "harmonic": "#2f6b2a",
}

PALETTE = {
    "ink": "#141414",
    "empty": "#c0392b",
    "cell": "#00000000",
    "caption": "#666666",
}


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _hui_fen(position: float) -> tuple[int, int]:
    """Split a hui.fen position, snapping x.10 up to (x+1).0."""
    hui = int(position)
    fen = int(round((position - hui) * 10))
    if fen >= 10:
        hui, fen = hui + 1, 0
    return hui, fen


# --------------------------------------------------------------------------
# jianpu annotation
# --------------------------------------------------------------------------

_JIANPU_RE = re.compile(r"^(?P<acc>[#b\u266f\u266d]*)(?P<deg>[0-7])(?P<oct>['\u2019,\uff0c]*)$")


def jianpu_label(text: str, x: float, y: float, size: float, fill: str) -> str:
    """Render a jianpu token the way a score prints it.

    Octave marks become dots above or below the digit rather than the ASCII
    apostrophes and commas the input format uses, because this ends up next to
    real tablature glyphs and a player reads the dots, not the punctuation.
    Anything that is not a bare degree (a lyric, a bar number) is drawn as
    plain centred text.
    """
    m = _JIANPU_RE.match(text.strip())
    if not m:
        return (f"<text x='{x:.1f}' y='{y:.1f}' fill='{fill}' font-size='{size:.1f}' "
                f"text-anchor='middle'>{_esc(text)}</text>")

    acc = m.group("acc").replace("#", "\u266f").replace("b", "\u266d")
    octave = (m.group("oct").count("'") + m.group("oct").count("\u2019")
              - m.group("oct").count(",") - m.group("oct").count("\uff0c"))
    parts = [f"<text x='{x:.1f}' y='{y:.1f}' fill='{fill}' font-size='{size:.1f}' "
             f"text-anchor='middle'>{_esc(acc)}{m.group('deg')}</text>"]

    r = size * 0.09
    for k in range(abs(octave)):
        dy = -(size * 0.72 + k * r * 3.2) if octave > 0 else (size * 0.30 + k * r * 3.2)
        parts.append(f"<circle cx='{x:.1f}' cy='{y + dy:.1f}' r='{r:.2f}' fill='{fill}'/>")
    return "".join(parts)


def component_names() -> list[str]:
    names = [f"md_{n}.blank" for n in range(1, 14)]
    names += [f"md_{n}.{d}" for n in range(1, 14) for d in range(1, 10)]
    names += [f"md_{s}" for s in range(1, 8)]
    names += ["md_san", "mod_fanyin", "md_wai", "md_placeholder"]
    return sorted(set(names))


REQUIRED_COMPONENTS = component_names()


class MissingAssets(FileNotFoundError):
    pass


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------


def default_asset_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "assets" / "jianzipu"


@dataclass
class Assets:
    """Component paths and slot geometry, loaded once and cached."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        layouts = self.root / "layouts.json"
        if not layouts.exists():
            raise MissingAssets(
                f"{layouts} not found. Run scripts/fetch_jianzipu_assets.py once "
                f"to download the glyph components from {ASSET_SOURCE}."
            )
        self._layouts = json.loads(layouts.read_text(encoding="utf-8"))
        self._cache: dict[str, tuple[str, tuple[float, float, float, float]]] = {}

    def layout(self, name: str = LAYOUT) -> dict:
        try:
            return self._layouts[name]
        except KeyError as exc:
            raise KeyError(f"unknown layout {name!r}") from exc

    def component(self, name: str) -> tuple[str, tuple[float, float, float, float]]:
        """Return (inner SVG markup, viewBox) for a component."""
        if name in self._cache:
            return self._cache[name]
        path = self.root / "components" / f"{name}.svg"
        if not path.exists():
            raise MissingAssets(f"component {name!r} not found at {path}")
        text = path.read_text(encoding="utf-8")
        body = text[text.index(">", text.index("<svg")) + 1 : text.rindex("</svg>")]
        # Every stroke path carries fill="black", which wins over a fill
        # inherited from an enclosing <g>. Swapping it for currentColor lets a
        # caller ink the glyph without touching the outline, which the OFL
        # cares about and timbre colouring needs.
        body = body.replace('fill="black"', 'fill="currentColor"')
        m = re.search(r'viewBox="([-\d.\s]+)"', text)
        box = tuple(float(v) for v in m.group(1).split()) if m else (0.0, 0.0, 350.0, 350.0)
        self._cache[name] = (body, box)  # type: ignore[assignment]
        return self._cache[name]

    def check(self) -> list[str]:
        """Names of required components that are missing."""
        return [n for n in REQUIRED_COMPONENTS
                if not (self.root / "components" / f"{n}.svg").exists()]


@lru_cache(maxsize=4)
def _assets(root: str) -> Assets:
    return Assets(Path(root))


def load_assets(root: str | Path | None = None) -> Assets:
    return _assets(str(Path(root) if root is not None else default_asset_dir()))


# --------------------------------------------------------------------------
# candidate -> component assignment
# --------------------------------------------------------------------------


@dataclass
class GlyphSpec:
    """Which component goes in which slot, and which slots stay empty."""

    filled: list[tuple[str, str]]      # (area, component name)
    empty: list[str]                   # areas drawn as empty boxes
    reading: str                       # human-readable gloss
    warning: str | None = None


CN = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
      8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二", 13: "十三"}


def spec_for(candidate: Any) -> GlyphSpec:
    """Map one Candidate onto slots.

    The right-hand slot is always empty: every note is plucked, and the system
    never says with which finger. The left-hand slot is empty for 按音 and 泛音,
    but ABSENT for 散音 -- an open string genuinely has no left hand, so drawing
    a gap there would claim a missing prediction that was never owed.
    """
    string = int(getattr(candidate, "string"))
    kind = str(getattr(candidate, "kind"))
    position = float(getattr(candidate, "position", 0.0) or 0.0)

    if not 1 <= string <= 7:
        return GlyphSpec([], ["area_left", "area_right"], f"string {string}?",
                         f"string {string} is outside 1-7")

    filled: list[tuple[str, str]] = [("area_string", f"md_{string}")]
    empty = ["area_right"]
    warning = None

    if kind == "open":
        filled.append(("area_hui", "md_san"))
        return GlyphSpec(filled, empty, f"散 {CN[string]}弦")

    hui, fen = _hui_fen(round(position, 1))
    empty.insert(0, "area_left")

    if hui < 1 or hui > 13:
        filled.append(("area_hui", "md_placeholder"))
        warning = f"hui {position:.1f} is outside the 13 hui; no glyph exists"
        reading = f"{position:.1f} {CN[string]}弦"
    else:
        filled.append(("area_hui", f"md_{hui}.blank" if fen == 0 else f"md_{hui}.{fen}"))
        pos_text = f"{CN[hui]}徽" + (f"{CN[fen]}分" if fen else "")
        reading = f"{'泛' if kind == 'harmonic' else '按'} {CN[string]}弦 {pos_text}"

    if kind == "harmonic":
        filled.append(("area_fy", "mod_fanyin"))

    return GlyphSpec(filled, empty, reading, warning)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _place(area: dict, body: str, box: Sequence[float]) -> str:
    sx = area["w"] / (box[2] or 350.0)
    sy = area["h"] / (box[3] or 350.0)
    return (f"<g transform='translate({area['x']},{area['y']}) "
            f"scale({sx:.5f},{sy:.5f})'>{body}</g>")


def _empty_box(area: dict, label: str, ink: str) -> str:
    return (
        f"<rect x='{area['x']}' y='{area['y']}' width='{area['w']}' height='{area['h']}' "
        f"fill='none' stroke='{ink}' stroke-width='7' stroke-dasharray='24,18' rx='16' "
        f"opacity='0.75'/>"
        f"<text x='{area['x'] + area['w'] / 2:.0f}' y='{area['y'] + area['h'] / 2 + 22:.0f}' "
        f"fill='{ink}' font-size='62' text-anchor='middle' opacity='0.75'>{label}</text>"
    )


def glyph_group(
    candidate: Any,
    *,
    assets: Assets | None = None,
    x: float = 0.0,
    y: float = 0.0,
    scale: float = 1.0,
    ink: str = PALETTE["ink"],
    empty_ink: str = PALETTE["empty"],
    show_empty_slots: bool = True,
    layout: str = LAYOUT,
) -> str:
    """One glyph as an SVG <g>, positioned so the CELL's top-left lands at x,y.

    Returned markup carries no <svg> wrapper, so it can be dropped into a
    fingerboard frame, a score column, or anything else.
    """
    a = assets or load_assets()
    lay = a.layout(layout)
    spec = spec_for(candidate)

    parts: list[str] = []
    if show_empty_slots:
        for area_name in spec.empty:
            if area_name in lay:
                parts.append(_empty_box(lay[area_name], SLOT_LABEL.get(area_name, "?"), empty_ink))
    for area_name, comp in spec.filled:
        if area_name not in lay:
            continue
        body, box = a.component(comp)
        parts.append(_place(lay[area_name], body, box))

    cx, cy = CELL[0], CELL[1]
    return (f"<g transform='translate({x},{y}) scale({scale:.5f}) "
            f"translate({-cx},{-cy})' color='{ink}' fill='{ink}'>"
            + "".join(parts) + "</g>")


def render_glyph(
    candidate: Any,
    *,
    assets: Assets | None = None,
    width: float = 180.0,
    caption: str | None = None,
    background: str | None = None,
    **kwargs: Any,
) -> str:
    """A single glyph as a standalone SVG, sized to `width` pixels."""
    scale = width / CELL[2]
    height = CELL[3] * scale
    cap_h = 34.0 if caption is not None else 0.0
    bg = (f"<rect width='{width:.0f}' height='{height + cap_h:.0f}' fill='{background}'/>"
          if background else "")
    body = glyph_group(candidate, assets=assets, scale=scale, **kwargs)
    cap = ""
    if caption is not None:
        cap = (f"<text x='{width / 2:.0f}' y='{height + 24:.0f}' fill='{PALETTE['caption']}' "
               f"font-size='20' text-anchor='middle'>{caption}</text>")
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' "
        f"height='{height + cap_h:.0f}' viewBox='0 0 {width:.0f} {height + cap_h:.0f}' "
        f"font-family=\"'Noto Sans CJK SC','Source Han Sans SC',sans-serif\">"
        f"{bg}{body}{cap}</svg>"
    )


def render_score(
    path: Iterable[Any],
    *,
    assets: Assets | None = None,
    glyph_width: float = 110.0,
    per_column: int = 10,
    orientation: str = "vertical",
    right_to_left: bool = True,
    gap: float = 0.18,
    number_notes: bool = True,
    labels: Sequence[str] | None = None,
    sublabels: Sequence[str] | None = None,
    colour_timbre: bool = False,
    highlight: int | None = None,
    background: str = "#faf6ec",
    **kwargs: Any,
) -> str:
    """A whole fingering laid out as a score.

    orientation="vertical" (default) is the traditional reading order:
    glyphs run top to bottom within a column, columns run right to left
    (set right_to_left=False to run them left to right instead).

    orientation="horizontal" runs glyphs left to right and wraps onto a new
    line below, like staff notation. Nothing about the glyphs changes -- only
    the order they are placed in -- so this is purely a presentation choice for
    an audience that reads left to right.

    `per_column` is the wrap length either way: glyphs per column when
    vertical, glyphs per line when horizontal.

    `colour_timbre` inks each glyph by 按/散/泛 in the same hues the
    fingerboard uses. Real tablature is monochrome and the default keeps it
    that way, but the three timbres are hard to tell apart at a glance —
    散 sits in the same slot as the 徽位 and is a similar size — so when the
    question is "what is the model doing with timbre", colour answers it in
    one look where reading every glyph does not.

    `labels` prints a jianpu token beside each glyph and `sublabels` a second
    line (a lyric syllable, say), which is how modern 琴谱 are actually set:
    the numbered notation says what to sound and the 减字 says how. Bare
    degrees are typeset with proper octave dots; anything else is drawn as
    plain text. Note that CJK sublabels need a CJK font at RASTER time -- fine
    in a browser, tofu boxes in a PNG export on a machine without one.
    """
    cands = list(path)
    a = assets or load_assets()
    scale = glyph_width / CELL[2]
    cell_h = CELL[3] * scale
    pad = glyph_width * 0.35
    horizontal = orientation == "horizontal"

    if orientation not in {"vertical", "horizontal"}:
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {orientation!r}")

    col_w = glyph_width * (1.0 + gap if horizontal else 1.45)
    row_h = cell_h * (1.0 + (0.12 if horizontal else gap))
    n_full = max(1, (len(cands) + per_column - 1) // per_column)
    across = min(per_column, max(len(cands), 1))

    # Size to the last glyph's own box, not to a full extra pitch, or the
    # figure ends up with a band of dead space along two edges.
    label_size = glyph_width * 0.30
    num_size = max(11.0, glyph_width * 0.14)
    lab_pad = (label_size * 1.85 if labels else 0.0) + (num_size + 6.0 if number_notes else 0.0)
    sub_pad = label_size * 1.7 if sublabels else 0.0
    num_pad = lab_pad if (labels or number_notes) else 0.0
    legend_h = label_size * 2.0 if colour_timbre else 0.0
    if horizontal:
        # A row must clear its own glyphs PLUS the annotations of the row
        # below it, or the next line's numbers land on this line's lyrics.
        top = pad + num_pad + legend_h
        row_h += sub_pad + lab_pad
        width = pad * 2 + col_w * (across - 1) + glyph_width
        height = top + pad + row_h * (n_full - 1) + cell_h + sub_pad
    else:
        top = pad + legend_h
        side = max(num_pad, label_size * 2.0 if labels else 0.0)
        col_w += side
        width = pad * 2 + col_w * (n_full - 1) + glyph_width + side
        height = top + pad + row_h * (across - 1) + cell_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' "
        f"viewBox='0 0 {width:.0f} {height:.0f}' "
        f"font-family=\"'Noto Sans CJK SC','Source Han Sans SC',sans-serif\">",
        f"<rect width='{width:.0f}' height='{height:.0f}' fill='{background}'/>",
    ]

    if colour_timbre:
        lx = pad
        for kind, ink in TIMBRE_INK.items():
            parts.append(
                f"<circle cx='{lx + label_size * 0.4:.1f}' cy='{pad + label_size * 0.5:.1f}' "
                f"r='{label_size * 0.34:.1f}' fill='{ink}'/>"
            )
            text = {"stopped": "按音", "open": "散音", "harmonic": "泛音"}[kind]
            parts.append(
                f"<text x='{lx + label_size:.1f}' y='{pad + label_size * 0.82:.1f}' "
                f"fill='{ink}' font-size='{label_size * 0.85:.1f}'>{text}</text>"
            )
            lx += label_size * 4.2

    for i, c in enumerate(cands):
        if horizontal:
            line, slot_i = divmod(i, per_column)
            cx = pad + slot_i * col_w
            cy = top + line * row_h
        else:
            col, slot_i = divmod(i, per_column)
            cx = pad + (n_full - 1 - col if right_to_left else col) * col_w
            cy = top + slot_i * row_h

        if highlight is not None and i == highlight:
            parts.append(
                f"<rect x='{cx - 6:.1f}' y='{cy - 6:.1f}' width='{glyph_width + 12:.1f}' "
                f"height='{cell_h + 12:.1f}' fill='#ffe9a8' rx='10'/>"
            )
        ink = TIMBRE_INK.get(str(getattr(c, "kind", "")), PALETTE["ink"]) \
            if colour_timbre else kwargs.pop("ink", PALETTE["ink"])
        parts.append(glyph_group(c, assets=a, x=cx, y=cy, scale=scale,
                                 ink=ink, **kwargs))
        if labels is not None and i < len(labels):
            if horizontal:
                lx, ly = cx + glyph_width / 2, cy - label_size * 0.75
            else:
                lx, ly = cx + glyph_width + label_size * 1.1, cy + label_size * 1.2
            parts.append(jianpu_label(labels[i], lx, ly, label_size, PALETTE["ink"]))

        if sublabels is not None and i < len(sublabels) and sublabels[i]:
            sx = cx + glyph_width / 2
            sy = cy + cell_h + label_size * 1.15
            parts.append(
                f"<text x='{sx:.1f}' y='{sy:.1f}' fill='{PALETTE['ink']}' "
                f"font-size='{label_size * 0.92:.1f}' text-anchor='middle'>"
                f"{_esc(sublabels[i])}</text>"
            )

        if number_notes:
            if horizontal:
                nx, ny, anchor = (cx + glyph_width / 2,
                                  cy - label_size * 0.75 - (label_size * 1.1 if labels else 0.0),
                                  "middle")
            else:
                nx, ny, anchor = (cx + glyph_width + 14,
                                  cy + (label_size * 2.6 if labels else 26.0), "start")
            parts.append(
                f"<text x='{nx:.0f}' y='{ny:.0f}' fill='{PALETTE['caption']}' "
                f"font-size='{num_size:.0f}' text-anchor='{anchor}'>{i + 1}</text>"
            )

    parts.append("</svg>")
    return "".join(parts)
