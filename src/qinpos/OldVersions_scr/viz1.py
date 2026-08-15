"""Fingerboard visualisation and presentation exports for QinPos.

This module is deliberately model-free: it only knows how to turn
`{candidate: probability}` mappings into SVG, and how to turn a list of SVG
frames into a video / GIF / self-contained HTML player.  Nothing here imports
marimo, `crf` or `learn`, so the same code can be driven from a notebook, a
unit test, or a batch script that renders figures for the dissertation.

A "candidate" is any object exposing ``.string`` (1-7), ``.kind``
("open" | "stopped" | "harmonic") and ``.position`` (hui position as a float,
e.g. 7.6 for 七徽六分).

Coordinate convention
---------------------
Hui fractions in `theory.HUI_FRACTIONS` are string-length fractions measured
from the 岳山 (bridge / speaking end).  hui 7 = 1/2, hui 9 = 2/3, hui 13 = 7/8.
Fraction 0.0 therefore sits at the 岳山 and 1.0 at the 龙龈, which is how the
board is laid out below: x_bridge on the right, x_nut on the left.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Geometry",
    "PALETTE",
    "render_fingerboard",
    "render_confidence_profile",
    "render_frame",
    "summarise",
    "most_ambiguous",
    "svg_to_png_bytes",
    "export_svg",
    "export_png_frames",
    "export_gif",
    "export_video",
    "export_html_player",
]


# --------------------------------------------------------------------------
# geometry and palette
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """All layout constants in one place so figures can be rescaled."""

    width: float = 980.0
    x_nut: float = 70.0          # 龙龈 end, string-length fraction 1.0
    x_bridge: float = 930.0      # 岳山 end, string-length fraction 0.0
    y_top: float = 48.0          # top edge of the board
    y_first: float = 94.0        # centre line of string 1
    y_step: float = 32.0         # vertical spacing between strings
    font: float = 13.0
    dot: float = 9.0             # radius of a stopped-note marker

    # -- derived ----------------------------------------------------------
    @property
    def y_last(self) -> float:
        return self.y_first + 6 * self.y_step

    @property
    def board_bottom(self) -> float:
        return self.y_last + 24.0

    @property
    def y_hui_inlay(self) -> float:
        return self.y_top + 20.0

    @property
    def y_hui_label(self) -> float:
        return self.board_bottom + 18.0

    @property
    def y_legend(self) -> float:
        return self.board_bottom + 42.0

    @property
    def height(self) -> float:
        return self.board_bottom + 56.0

    def y(self, string: int) -> float:
        """Centre line of `string` (1 = thickest, drawn at the top)."""
        return self.y_first + self.y_step * (string - 1)

    def scaled(self, k: float) -> "Geometry":
        return replace(self, **{f.name: getattr(self, f.name) * k for f in fields(self)})


COMPACT = Geometry().scaled(0.62)

PALETTE = {
    "page": "#20160b",
    "board": "#5a3d1e",
    "yueshan": "#d9c9a3",
    "nut": "#caa96b",
    "hui": "#efe6cf",
    "string": "#e8ddc2",
    "label": "#cbb686",
    "text": "#f4ecd8",
    "stopped": "#ffb347",
    "harmonic": "#b7ff9e",
    "open": "#7fd0ff",
    "expert": "#ff6b6b",
    "highlight": "#ffffff",
    "bar": "#ffd27f",
    "bar_error": "#ff6b6b",
    "baseline": "#c9bfae",
    "changed": "#7fd0ff",
}

HUI_CN = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
          8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二", 13: "十三"}

KIND_EN = {"open": "open", "stopped": "stopped", "harmonic": "harmonic"}
KIND_CN = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}

# Latin-only stack is used when cjk=False so that rasterisers without a CJK
# font installed do not render tofu boxes into the exported video.
FONT_LATIN = "Georgia, 'Times New Roman', serif"
FONT_CJK = "'Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', serif"


def _font(cjk: bool) -> str:
    return FONT_CJK if cjk else FONT_LATIN


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------
# candidate accessors (defensive: works with dataclasses or namedtuples)
# --------------------------------------------------------------------------


def _string_of(c: Any) -> int:
    return int(getattr(c, "string"))


def _kind_of(c: Any) -> str:
    return str(getattr(c, "kind"))


def _position_of(c: Any) -> float:
    return float(getattr(c, "position", 0.0) or 0.0)


# --------------------------------------------------------------------------
# horizontal placement
# --------------------------------------------------------------------------


def hui_x(fraction: float, geom: Geometry) -> float:
    """x pixel for a string-length fraction measured from the 岳山."""
    return geom.x_bridge - (geom.x_bridge - geom.x_nut) * float(fraction)


def position_x(position: float, hui_fractions: Mapping[int, Any], geom: Geometry) -> float:
    """x pixel for a hui position such as 7.6 (七徽六分).

    Interpolates linearly in *string length* between the two bracketing hui,
    which is what 徽分 means in practice.  Positions above hui 13 extrapolate
    towards the 龙龈 (fraction 1.0).
    """
    h = int(position)
    fen = float(position) - h
    f0 = float(hui_fractions[h]) if h in hui_fractions else 1.0
    f1 = float(hui_fractions[h + 1]) if (h + 1) in hui_fractions else 1.0
    return hui_x(f0 + (f1 - f0) * fen, geom)


def candidate_x(c: Any, hui_fractions: Mapping[int, Any], geom: Geometry) -> float:
    if _kind_of(c) == "open":
        return geom.x_bridge - 6.0
    return position_x(_position_of(c), hui_fractions, geom)


# --------------------------------------------------------------------------
# board body
# --------------------------------------------------------------------------


def _board_body(
    marginals: Mapping[Any, float],
    hui_fractions: Mapping[int, Any],
    geom: Geometry,
    expert: Any | None = None,
    baseline: Any | None = None,
    cjk: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    prob_threshold: float = 0.10,
    show_legend: bool = True,
) -> list[str]:
    p: list[str] = []
    k = geom.width / 980.0  # stroke/radius scale factor

    if title:
        p.append(
            f"<text x='{geom.x_nut - 14:.1f}' y='{20 * k + 4:.1f}' fill='{PALETTE['text']}' "
            f"font-size='{17 * k:.1f}' font-weight='bold'>{_esc(title)}</text>"
        )
    if subtitle:
        p.append(
            f"<text x='{geom.x_nut - 14:.1f}' y='{38 * k + 4:.1f}' fill='{PALETTE['label']}' "
            f"font-size='{12 * k:.1f}'>{_esc(subtitle)}</text>"
        )

    board_h = geom.board_bottom - geom.y_top
    p.append(
        f"<rect x='{geom.x_nut - 18 * k:.1f}' y='{geom.y_top:.1f}' "
        f"width='{geom.x_bridge - geom.x_nut + 36 * k:.1f}' height='{board_h:.1f}' "
        f"rx='{10 * k:.1f}' fill='{PALETTE['board']}'/>"
    )
    # 岳山 (right) and 龙龈 (left)
    p.append(
        f"<rect x='{geom.x_bridge + 6 * k:.1f}' y='{geom.y_top:.1f}' width='{8 * k:.1f}' "
        f"height='{board_h:.1f}' fill='{PALETTE['yueshan']}'/>"
    )
    p.append(
        f"<rect x='{geom.x_nut - 14 * k:.1f}' y='{geom.y_top:.1f}' width='{5 * k:.1f}' "
        f"height='{board_h:.1f}' fill='{PALETTE['nut']}'/>"
    )

    # hui inlays at their true string-length fractions
    for h, fr in sorted(hui_fractions.items()):
        x = hui_x(float(fr), geom)
        p.append(
            f"<circle cx='{x:.1f}' cy='{geom.y_hui_inlay:.1f}' r='{5 * k:.1f}' fill='{PALETTE['hui']}'/>"
        )
        lab = HUI_CN.get(int(h), str(h)) if cjk else str(h)
        p.append(
            f"<text x='{x:.1f}' y='{geom.y_hui_label:.1f}' fill='{PALETTE['label']}' "
            f"font-size='{geom.font * 0.92:.1f}' text-anchor='middle'>{_esc(lab)}</text>"
        )

    # strings, 1 thickest
    for s in range(1, 8):
        yy = geom.y(s)
        p.append(
            f"<line x1='{geom.x_nut - 14 * k:.1f}' y1='{yy:.1f}' x2='{geom.x_bridge + 6 * k:.1f}' "
            f"y2='{yy:.1f}' stroke='{PALETTE['string']}' stroke-width='{(3.4 - 0.35 * (s - 1)) * k:.2f}'/>"
        )
        lab = f"{s}弦" if cjk else f"str {s}"
        p.append(
            f"<text x='{geom.x_nut - 30 * k:.1f}' y='{yy + 4 * k:.1f}' fill='{PALETTE['label']}' "
            f"font-size='{geom.font:.1f}' text-anchor='end'>{_esc(lab)}</text>"
        )

    if not marginals:
        return p

    argmax = max(marginals, key=lambda c: marginals[c])
    used_labels: dict[tuple[int, int], int] = {}

    # dimmest first so the confident candidates draw on top
    for c, prob in sorted(marginals.items(), key=lambda kv: kv[1]):
        yy = geom.y(_string_of(c))
        x = candidate_x(c, hui_fractions, geom)
        opacity = 0.15 + 0.85 * float(prob)
        hot = c == argmax
        kind = _kind_of(c)

        if kind == "open":
            d = 9 * k
            p.append(
                f"<path d='M {x - d:.1f} {yy:.1f} L {x:.1f} {yy - d:.1f} L {x + d:.1f} {yy:.1f} "
                f"L {x:.1f} {yy + d:.1f} Z' fill='{PALETTE['open']}' opacity='{opacity:.2f}'/>"
            )
            if hot:
                p.append(
                    f"<path d='M {x - d - 4 * k:.1f} {yy:.1f} L {x:.1f} {yy - d - 4 * k:.1f} "
                    f"L {x + d + 4 * k:.1f} {yy:.1f} L {x:.1f} {yy + d + 4 * k:.1f} Z' "
                    f"fill='none' stroke='{PALETTE['highlight']}' stroke-width='{1.5 * k:.1f}'/>"
                )
        elif kind == "harmonic":
            p.append(
                f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='{geom.dot + 1:.1f}' fill='none' "
                f"stroke='{PALETTE['harmonic']}' stroke-width='{3 * k:.1f}' opacity='{opacity:.2f}'/>"
            )
            if hot:
                p.append(
                    f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='{geom.dot + 5:.1f}' fill='none' "
                    f"stroke='{PALETTE['highlight']}' stroke-width='{1.5 * k:.1f}'/>"
                )
        else:  # stopped
            p.append(
                f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='{geom.dot:.1f}' fill='{PALETTE['stopped']}' "
                f"opacity='{opacity:.2f}'/>"
            )
            if hot:
                p.append(
                    f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='{geom.dot + 4:.1f}' fill='none' "
                    f"stroke='{PALETTE['highlight']}' stroke-width='{1.5 * k:.1f}'/>"
                )

        if prob >= prob_threshold:
            # nudge colliding labels upward instead of overprinting them
            bucket = (_string_of(c), int(x // (26 * k)))
            n = used_labels.get(bucket, 0)
            used_labels[bucket] = n + 1
            ly = yy - (14 + 12 * n) * k
            p.append(
                f"<text x='{x:.1f}' y='{ly:.1f}' fill='{PALETTE['text']}' "
                f"font-size='{geom.font * 0.85:.1f}' text-anchor='middle'>{prob:.2f}</text>"
            )

    # Reference ring FIRST so the expert ring stays on top where they coincide.
    if baseline is not None:
        yy = geom.y(_string_of(baseline))
        bx = candidate_x(baseline, hui_fractions, geom)
        p.append(
            f"<circle cx='{bx:.1f}' cy='{yy:.1f}' r='{geom.dot + 12:.1f}' fill='none' "
            f"stroke='{PALETTE['baseline']}' stroke-width='{1.6 * k:.1f}' "
            f"stroke-dasharray='{2 * k:.1f},{3 * k:.1f}'/>"
        )

    if expert is not None:
        yy = geom.y(_string_of(expert))
        ex = candidate_x(expert, hui_fractions, geom)
        p.append(
            f"<circle cx='{ex:.1f}' cy='{yy:.1f}' r='{geom.dot + 8:.1f}' fill='none' "
            f"stroke='{PALETTE['expert']}' stroke-width='{2 * k:.1f}' stroke-dasharray='{5 * k:.1f},{4 * k:.1f}'/>"
        )

    if show_legend:
        p.extend(_legend(geom, cjk, with_expert=expert is not None,
                         with_baseline=baseline is not None))
    return p


def _legend(geom: Geometry, cjk: bool, with_expert: bool, with_baseline: bool = False) -> list[str]:
    k = geom.width / 980.0
    y = geom.y_legend
    x = geom.x_nut - 14 * k
    p: list[str] = []
    items = [("stopped", PALETTE["stopped"]), ("harmonic", PALETTE["harmonic"]), ("open", PALETTE["open"])]
    for kind, colour in items:
        if kind == "open":
            d = 6 * k
            p.append(
                f"<path d='M {x - d:.1f} {y:.1f} L {x:.1f} {y - d:.1f} L {x + d:.1f} {y:.1f} "
                f"L {x:.1f} {y + d:.1f} Z' fill='{colour}'/>"
            )
        elif kind == "harmonic":
            p.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{6 * k:.1f}' fill='none' stroke='{colour}' stroke-width='{2.5 * k:.1f}'/>")
        else:
            p.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{6 * k:.1f}' fill='{colour}'/>")
        text = f"{KIND_EN[kind]} {KIND_CN[kind]}" if cjk else KIND_EN[kind]
        p.append(
            f"<text x='{x + 12 * k:.1f}' y='{y + 4 * k:.1f}' fill='{PALETTE['label']}' "
            f"font-size='{geom.font * 0.9:.1f}'>{_esc(text)}</text>"
        )
        x += (len(text) * 7.4 + 30) * k

    if with_expert:
        p.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{6 * k:.1f}' fill='none' stroke='{PALETTE['expert']}' "
            f"stroke-width='{2 * k:.1f}' stroke-dasharray='{3 * k:.1f},{3 * k:.1f}'/>"
        )
        text = "expert annotation" if not cjk else "expert 专家标注"
        p.append(
            f"<text x='{x + 12 * k:.1f}' y='{y + 4 * k:.1f}' fill='{PALETTE['label']}' "
            f"font-size='{geom.font * 0.9:.1f}'>{_esc(text)}</text>"
        )
        x += (len(text) * 7.4 + 30) * k

    if with_baseline:
        p.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{6 * k:.1f}' fill='none' "
            f"stroke='{PALETTE['baseline']}' stroke-width='{1.6 * k:.1f}' "
            f"stroke-dasharray='{2 * k:.1f},{3 * k:.1f}'/>"
        )
        text = "baseline (bias 0)"
        p.append(
            f"<text x='{x + 12 * k:.1f}' y='{y + 4 * k:.1f}' fill='{PALETTE['label']}' "
            f"font-size='{geom.font * 0.9:.1f}'>{_esc(text)}</text>"
        )
        x += (len(text) * 7.4 + 30) * k

    p.append(
        f"<text x='{geom.x_bridge + 14 * k:.1f}' y='{y + 4 * k:.1f}' fill='{PALETTE['label']}' "
        f"font-size='{geom.font * 0.9:.1f}' text-anchor='end'>opacity = marginal probability</text>"
    )
    return p


def render_fingerboard(
    marginals: Mapping[Any, float],
    *,
    hui_fractions: Mapping[int, Any],
    expert: Any | None = None,
    baseline: Any | None = None,
    geom: Geometry = Geometry(),
    cjk: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    prob_threshold: float = 0.10,
    show_legend: bool = True,
) -> str:
    """One fingerboard with every candidate lit by its marginal probability.

    `baseline` is a single Candidate drawn as a faint dashed ring: the choice
    the SAME model makes with the bias sliders at zero. Without it a slider
    move is unreadable, because the board shows where the model ended up but
    not what it moved away from.
    """
    body = _board_body(
        marginals, hui_fractions, geom, expert=expert, baseline=baseline,
        cjk=cjk, title=title, subtitle=subtitle,
        prob_threshold=prob_threshold, show_legend=show_legend,
    )
    return _wrap(body, geom.width, geom.height, cjk, rounded=12)


# --------------------------------------------------------------------------
# confidence profile
# --------------------------------------------------------------------------


def _profile_body(
    marginals_all: Sequence[Mapping[Any, float]],
    geom: Geometry,
    experts: Sequence[Any] | None = None,
    cursor: int | None = None,
    height: float = 96.0,
    cjk: bool = True,
    mark_errors: bool = True,
    changed: Sequence[bool] | None = None,
) -> list[str]:
    k = geom.width / 980.0
    n = max(len(marginals_all), 1)
    left, right = 12 * k, geom.width - 12 * k
    bw = max(1.5, (right - left) / n)
    base = height - 24 * k
    p: list[str] = []

    for j, m in enumerate(marginals_all):
        if not m:
            continue
        top = max(m.values())
        x = left + j * bw
        h = 6 * k + (base - 12 * k) * top
        wrong = False
        if mark_errors and experts is not None and j < len(experts) and experts[j] is not None:
            wrong = max(m, key=lambda c: m[c]) != experts[j]
        fill = PALETTE["bar_error"] if wrong else PALETTE["bar"]
        p.append(
            f"<rect x='{x:.2f}' y='{base - h:.2f}' width='{max(bw * 0.8, 1.0):.2f}' height='{h:.2f}' "
            f"fill='{fill}' opacity='{0.35 + 0.65 * top:.2f}'/>"
        )
        if changed is not None and j < len(changed) and changed[j]:
            p.append(
                f"<rect x='{x:.2f}' y='{base - h - 6 * k:.2f}' "
                f"width='{max(bw * 0.8, 1.0):.2f}' height='{3 * k:.2f}' "
                f"fill='{PALETTE['changed']}'/>"
            )

    if cursor is not None and 0 <= cursor < n:
        x = left + cursor * bw + bw * 0.4
        p.append(
            f"<line x1='{x:.2f}' y1='{4 * k:.1f}' x2='{x:.2f}' y2='{base:.1f}' "
            f"stroke='{PALETTE['highlight']}' stroke-width='{1.6 * k:.1f}' opacity='0.9'/>"
        )

    caption = "bar height = confidence of the top candidate"
    if mark_errors and experts is not None:
        caption += "   ·   red = model's top choice differs from the expert"
    if changed is not None:
        caption += "   ·   blue tick = moved since bias 0"
    p.append(
        f"<text x='{left:.1f}' y='{height - 6 * k:.1f}' fill='{PALETTE['label']}' "
        f"font-size='{geom.font * 0.88:.1f}'>{_esc(caption)}</text>"
    )
    return p


def render_confidence_profile(
    marginals_all: Sequence[Mapping[Any, float]],
    *,
    geom: Geometry = Geometry(),
    experts: Sequence[Any] | None = None,
    cursor: int | None = None,
    height: float = 96.0,
    cjk: bool = True,
    mark_errors: bool = True,
    changed: Sequence[bool] | None = None,
) -> str:
    body = _profile_body(marginals_all, geom, experts, cursor, height, cjk,
                         mark_errors, changed)
    return _wrap(body, geom.width, height, cjk, rounded=8)


def _wrap(body: list[str], width: float, height: float, cjk: bool, rounded: int = 10) -> str:
    head = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' "
        f"viewBox='0 0 {width:.0f} {height:.0f}' font-family=\"{_font(cjk)}\">"
        f"<rect width='{width:.0f}' height='{height:.0f}' rx='{rounded}' fill='{PALETTE['page']}'/>"
    )
    return head + "".join(body) + "</svg>"


# --------------------------------------------------------------------------
# composite frame (board + profile) -- this is what the video uses
# --------------------------------------------------------------------------


def render_frame(
    marginals_all: Sequence[Mapping[Any, float]],
    index: int,
    *,
    hui_fractions: Mapping[int, Any],
    experts: Sequence[Any] | None = None,
    baselines: Sequence[Any] | None = None,
    geom: Geometry = Geometry(),
    cjk: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    with_profile: bool = True,
    profile_height: float = 96.0,
    changed: Sequence[bool] | None = None,
) -> str:
    """Single self-contained frame: board on top, whole-piece profile below."""
    expert = None
    if experts is not None and index < len(experts):
        expert = experts[index]
    baseline = None
    if baselines is not None and index < len(baselines):
        baseline = baselines[index]
    body = _board_body(
        marginals_all[index], hui_fractions, geom, expert=expert,
        baseline=baseline, cjk=cjk, title=title, subtitle=subtitle,
    )
    total_h = geom.height
    if with_profile:
        prof = _profile_body(marginals_all, geom, experts, index, profile_height,
                             cjk, True, changed)
        body.append(f"<g transform='translate(0,{geom.height:.1f})'>" + "".join(prof) + "</g>")
        total_h += profile_height
    return _wrap(body, geom.width, total_h, cjk, rounded=12)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------


def summarise(
    marginals_all: Sequence[Mapping[Any, float]],
    experts: Sequence[Any] | None = None,
    ks: Iterable[int] = (1, 3),
) -> dict[str, Any]:
    """Metrics for one piece: top-k accuracy plus a confidence breakdown."""
    n = len(marginals_all)
    out: dict[str, Any] = {"n_notes": n}
    if n == 0:
        return out

    tops = [max(m.values()) if m else 0.0 for m in marginals_all]
    out["mean_confidence"] = sum(tops) / n
    out["n_confident"] = sum(t > 0.8 for t in tops)
    out["n_ambiguous"] = sum(t < 0.5 for t in tops)
    out["mean_candidates"] = sum(len(m) for m in marginals_all) / n

    if experts is not None:
        for k in ks:
            hit = 0
            scored = 0
            for m, e in zip(marginals_all, experts):
                if e is None or not m:
                    continue
                scored += 1
                ranked = sorted(m, key=lambda c: m[c], reverse=True)[:k]
                hit += any(c == e for c in ranked)
            out[f"top{k}"] = hit / scored if scored else float("nan")
        out["n_scored"] = scored
    return out


def most_ambiguous(marginals_all: Sequence[Mapping[Any, float]], n: int = 6) -> list[int]:
    """Indices of the least confident notes -- the ones worth asking a 琴人 about."""
    scored = [(max(m.values()) if m else 0.0, i) for i, m in enumerate(marginals_all)]
    scored.sort()
    return [i for _, i in scored[:n]]


# --------------------------------------------------------------------------
# rasterisation and export
# --------------------------------------------------------------------------


class ExportError(RuntimeError):
    pass


def svg_to_png_bytes(svg: str, scale: float = 2.0) -> bytes:
    """Rasterise SVG.  Tries cairosvg, then any CLI converter on PATH."""
    try:
        import cairosvg  # type: ignore

        return cairosvg.svg2png(bytestring=svg.encode("utf-8"), scale=scale)
    except ImportError:
        pass

    converters = [
        ("resvg", lambda i, o: ["resvg", "--zoom", str(scale), i, o]),
        ("rsvg-convert", lambda i, o: ["rsvg-convert", "-z", str(scale), "-o", o, i]),
        ("inkscape", lambda i, o: ["inkscape", "--export-type=png",
                                   f"--export-dpi={96 * scale:.0f}", "-o", o, i]),
    ]
    for exe, argv in converters:
        if shutil.which(exe):
            with tempfile.TemporaryDirectory() as d:
                ip, op = Path(d) / "f.svg", Path(d) / "f.png"
                ip.write_text(svg, encoding="utf-8")
                subprocess.run(argv(str(ip), str(op)), check=True, capture_output=True)
                return op.read_bytes()

    raise ExportError(
        "No SVG rasteriser found. Install one of:\n"
        "  uv add cairosvg          (pure-python-ish, needs libcairo)\n"
        "  brew install resvg / apt install librsvg2-bin\n"
        "You can still use export_html_player(), which needs no extra tools."
    )


def export_svg(svg: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def export_png_frames(frames: Sequence[str], out_dir: str | Path, scale: float = 2.0) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, svg in enumerate(frames):
        p = out_dir / f"frame_{i:05d}.png"
        p.write_bytes(svg_to_png_bytes(svg, scale))
        paths.append(p)
    return paths


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def export_video(
    frames: Sequence[str],
    path: str | Path,
    *,
    fps: float = 2.0,
    scale: float = 2.0,
    hold_last: int = 4,
) -> Path:
    """SVG frames -> H.264 mp4.  Falls back to GIF if ffmpeg is unavailable."""
    if not frames:
        raise ExportError("no frames to export")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        gif = path.with_suffix(".gif")
        export_gif(frames, gif, fps=fps, scale=scale, hold_last=hold_last)
        return gif

    seq = list(frames) + [frames[-1]] * max(hold_last, 0)
    with tempfile.TemporaryDirectory() as d:
        export_png_frames(seq, d, scale=scale)
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(Path(d) / "frame_%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return path


def export_gif(
    frames: Sequence[str],
    path: str | Path,
    *,
    fps: float = 2.0,
    scale: float = 1.5,
    hold_last: int = 4,
) -> Path:
    from io import BytesIO

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ExportError("GIF export needs Pillow: uv add pillow") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = list(frames) + [frames[-1]] * max(hold_last, 0)
    imgs = [Image.open(BytesIO(svg_to_png_bytes(s, scale))).convert("P", palette=Image.ADAPTIVE)
            for s in seq]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return path


_PLAYER_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 body{margin:0;background:#15100a;color:#f4ecd8;
      font-family:'Noto Sans CJK SC','Helvetica Neue',Arial,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:14px;padding:24px}
 #stage{width:min(1200px,96vw)}
 #stage svg{width:100%;height:auto;display:block}
 .bar{display:flex;align-items:center;gap:14px;width:min(1200px,96vw);flex-wrap:wrap}
 button{background:#5a3d1e;color:#f4ecd8;border:1px solid #8a6636;border-radius:8px;
        padding:8px 16px;font-size:15px;cursor:pointer}
 button:hover{background:#6d4a25}
 input[type=range]{flex:1;min-width:220px;accent-color:#ffb347}
 code{color:#ffd27f}
 .hint{opacity:.6;font-size:13px}
</style>
<div id="stage"></div>
<div class="bar">
  <button id="play">▶ play</button>
  <input type="range" id="scrub" min="0" value="0">
  <span id="counter"><code>1</code></span>
  <label>fps <input type="number" id="fps" value="__FPS__" min="0.25" max="30" step="0.25" style="width:64px"></label>
</div>
<div class="hint">space = play/pause · ← → = step one note · full-screen the browser and screen-record if you need a video</div>
<script>
const FRAMES = __FRAMES__;
const stage = document.getElementById('stage');
const scrub = document.getElementById('scrub');
const counter = document.getElementById('counter');
const playBtn = document.getElementById('play');
const fpsBox = document.getElementById('fps');
let i = 0, timer = null;
scrub.max = FRAMES.length - 1;
function draw(){ stage.innerHTML = FRAMES[i]; scrub.value = i;
  counter.innerHTML = '<code>' + (i+1) + '</code> / ' + FRAMES.length; }
function step(){ i = (i + 1) % FRAMES.length; draw(); }
function stop(){ clearInterval(timer); timer = null; playBtn.textContent = '▶ play'; }
function play(){ stop(); timer = setInterval(step, 1000 / Number(fpsBox.value || 2));
  playBtn.textContent = '❚❚ pause'; }
playBtn.onclick = () => timer ? stop() : play();
scrub.oninput = e => { stop(); i = Number(e.target.value); draw(); };
fpsBox.onchange = () => { if (timer) play(); };
document.onkeydown = e => {
  if (e.code === 'Space'){ e.preventDefault(); timer ? stop() : play(); }
  if (e.code === 'ArrowRight'){ stop(); i = Math.min(i+1, FRAMES.length-1); draw(); }
  if (e.code === 'ArrowLeft'){ stop(); i = Math.max(i-1, 0); draw(); }
};
draw();
</script>
"""


def export_html_player(
    frames: Sequence[str],
    path: str | Path,
    *,
    fps: float = 2.0,
    title: str = "QinPos fingerboard",
) -> Path:
    """Self-contained HTML player.  No external tools, works in any browser.

    Handy as a presentation fallback: open it full-screen and drive it with the
    keyboard, or screen-record it if the venue wants a plain video file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(list(frames)).replace("</", "<\\/")
    html = (_PLAYER_TEMPLATE
            .replace("__FRAMES__", payload)
            .replace("__FPS__", str(fps))
            .replace("__TITLE__", _esc(title)))
    path.write_text(html, encoding="utf-8")
    return path
