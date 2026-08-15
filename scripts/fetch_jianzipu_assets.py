#!/usr/bin/env python3
"""Download the 减字谱 glyph components QinPos needs, once.

Assets come from the JianZiPu font project by Nancy Yi Liang
(https://github.com/neuralfirings/JianZiPu). The font is licensed under the
SIL Open Font License 1.1 with Reserved Font Name "JianZiPu"; the build code
there is MIT. Both licences permit redistribution; the OFL requires that the
licence travel with the files and that a modified version not reuse the
reserved name, so this script writes LICENSE-JianZiPu.txt and ATTRIBUTION.md
next to the components and does not alter any glyph.

Only the ~140 components QinPos can actually emit are fetched -- the hui.fen
numerals, the string numerals, 散, and the 泛音 marker. Right-hand and
left-hand fingering components are deliberately NOT downloaded: the system
does not predict fingering, so having those glyphs available would only invite
someone to fill the empty slots in later.

Usage:
    python scripts/fetch_jianzipu_assets.py            # -> assets/jianzipu/
    python scripts/fetch_jianzipu_assets.py --force    # re-download
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "neuralfirings/JianZiPu"
REF = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REF}"

LAYOUTS_SRC = "builder/inputs/layouts.json"
COMPONENT_DIR = "src/components"
LICENCE_SRC = "LICENSE.font"

ATTRIBUTION = f"""\
# Jianzipu glyph components

Source: {REPO} ({RAW})
Font licence: SIL Open Font License 1.1, Reserved Font Name "JianZiPu"
Copyright: Nancy Yi Liang / Nellodee LLC
Build code in the upstream repository: MIT

These files are redistributed unmodified. QinPos composes them into skeleton
tablature by placing the SVG paths at the slot coordinates given in
layouts.json; no glyph outline is edited and the reserved font name is not
reused for anything QinPos produces.

QinPos uses only the position components (hui.fen numerals, string numerals,
散, 泛音). Fingering components are not vendored, because QinPos does not
predict fingering.
"""


def component_names() -> list[str]:
    names = [f"md_{n}.blank" for n in range(1, 14)]
    names += [f"md_{n}.{d}" for n in range(1, 14) for d in range(1, 10)]
    names += [f"md_{s}" for s in range(1, 8)]
    names += ["md_san", "mod_fanyin", "md_wai", "md_placeholder"]
    return sorted(set(names))


def fetch(url: str, dest: Path, force: bool) -> tuple[str, str]:
    if dest.exists() and not force:
        return ("skip", dest.name)
    try:
        with urllib.request.urlopen(url, timeout=30) as fh:
            data = fh.read()
    except urllib.error.HTTPError as exc:
        return ("fail", f"{dest.name}: HTTP {exc.code}")
    except Exception as exc:  # network, DNS, TLS
        return ("fail", f"{dest.name}: {exc}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return ("ok", dest.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets" / "jianzipu")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    out: Path = args.out
    comp_dir = out / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(f"{RAW}/{LAYOUTS_SRC}", out / "layouts.json"),
            (f"{RAW}/{LICENCE_SRC}", out / "LICENSE-JianZiPu.txt")]
    jobs += [(f"{RAW}/{COMPONENT_DIR}/{n}.svg", comp_dir / f"{n}.svg")
             for n in component_names()]

    print(f"fetching {len(jobs)} files from {REPO}@{REF} into {out}")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for status, msg in pool.map(lambda j: fetch(*j, args.force), jobs):
            counts[status] += 1
            if status == "fail":
                failures.append(msg)

    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")

    print(f"  downloaded {counts['ok']}, already present {counts['skip']}, failed {counts['fail']}")
    for f in failures[:10]:
        print(f"    {f}")
    if failures:
        print("\nSome files are missing. Re-run, or check whether the upstream "
              "repository has reorganised its paths.")
        return 1

    print(f"\nwrote {out / 'ATTRIBUTION.md'} and {out / 'LICENSE-JianZiPu.txt'}")
    print("qinpos.jianzipu will pick these up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
