#!/usr/bin/env python3
"""Regenerate chapter notebooks with HITA setup + export cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = ROOT / "notebooks"

_TITLES = {
    1: "The Sigmoid",
    2: "Parameters & the σ plane",
    3: "Loss landscapes",
    4: "Tutorial rails & likelihood",
    5: "Can we trust the model?",
    6: "Frequentist sampling — how much does the line wiggle?",
    7: "Collinearity & regularization (native hita)",
}

_STORY_IMPORT = {
    1: "from hita.stories.ch1 import CH1_EXPORT_SPECS as EXPORT_SPECS, export_clip",
    2: "from hita.stories.ch2 import CH2_EXPORT_SPECS as EXPORT_SPECS, export_clip",
    3: "from hita.stories.ch3 import CH3_EXPORT_SPECS as EXPORT_SPECS, export_clip",
    4: "from hita.stories.ch4 import CH4_EXPORT_SPECS as EXPORT_SPECS, export_clip",
    5: "from hita.stories.ch5 import CH5_EXPORT_SPECS as EXPORT_SPECS, export_clip",
    6: "from hita.stories.ch6 import CH6_EXPORT_SPECS as EXPORT_SPECS, export_clip",
    7: "from hita.stories.ch7 import CH7_EXPORT_SPECS as EXPORT_SPECS, export_clip",
}

# Modules to drop from sys.modules before chapter 5 setup so notebook kernels
# pick up synced legacy story files (e.g. new clips 54–56).
_CH5_RELOAD_MODULES = (
    "ch5_story",
    "ch5_prior_landscape",
    "ch5_core",
    "ch5_layout",
    "ch5_datasets",
    "hita.stories.ch5",
    "hita.stories.builders.ch5_47",
    "hita.stories.builders.ch5_landscape",
)

_INHERITS = {
    1: [],
    2: [1],
    3: [],
    4: [3],
    5: [3, 4],
    6: [3, 4, 5],
    7: [],
}


def _setup_source(chapter: int) -> str:
    reload_block = ""
    if chapter == 5:
        mods = ",\n    ".join(repr(m) for m in _CH5_RELOAD_MODULES)
        reload_block = f'''
import sys
for _mod in (
    {mods},
):
    sys.modules.pop(_mod, None)
from hita.export.context import clear_context_cache
clear_context_cache()
'''

    return f'''# --- Chapter {chapter} setup (HITA) ---

import os
from hita import load_chapter
{reload_block}
ctx = load_chapter(
    {chapter},
    inherits={_INHERITS[chapter]!r},
    profile=os.environ.get("HITA_RENDER_PROFILE", "hq"),
)
globals().update(ctx.globals_dict)
# Bind after update so inherited chapter shims cannot overwrite this exporter.
{_STORY_IMPORT[chapter]}

n = len(EXPORT_SPECS)
native = getattr(ctx, "native", False)
print(
    f"Chapter {{ctx.chapter}} OK — profile={{ctx.profile.name}}, "
    f"exports={{n}}, native={{native}}, series={{ctx.series.version}}"
)
'''


def _intro(chapter: int, n_exports: int) -> str:
    native_note = (
        "\nThis chapter is **native hita** (no `exec(notebook)`).\n"
        if chapter >= 7
        else ""
    )
    return f"""# Logistic Regression — Chapter {chapter} ({_TITLES[chapter]})

Migrated to the **HITA** library. Run the setup cell once, then each export cell.
{native_note}
Use `HITA_RENDER_PROFILE=draft` for fast previews, `hq` for final.

**{n_exports}** registered exports → `renders/`

```bash
HITA_RENDER_PROFILE=draft python -m hita.export --chapter {chapter} --list
```
"""


def _exports_for(chapter: int) -> list[tuple[str, str]]:
    if chapter == 1:
        from hita.stories.ch1 import CH1_EXPORT_SPECS

        return [(a, b) for a, b, *_ in CH1_EXPORT_SPECS]
    if chapter == 2:
        from hita.stories.ch2 import CH2_EXPORT_SPECS

        return list(CH2_EXPORT_SPECS)
    if chapter == 5:
        from ch5_story import CH5_EXPORT_SPECS

        return [(a, b) for a, b, _ in CH5_EXPORT_SPECS]
    if chapter == 6:
        from ch6_story import CH6_EXPORT_SPECS

        return [(a, b) for a, b, _ in CH6_EXPORT_SPECS]
    if chapter == 7:
        from hita.stories.ch7 import CH7_EXPORT_SPECS

        return list(CH7_EXPORT_SPECS)
    return [(f"ch{chapter}_00", f"ch{chapter}_00_placeholder.mp4")]


def gen_notebook(chapter: int) -> Path:
    import hita  # noqa: F401

    exports = _exports_for(chapter)
    cells = [
        {
            "cell_type": "markdown",
            "id": f"ch{chapter}_intro",
            "metadata": {},
            "source": [_intro(chapter, len(exports))],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": f"ch{chapter}_setup",
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in _setup_source(chapter).splitlines()],
        },
    ]
    for clip_id, filename in exports:
        if "placeholder" in filename:
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"md_{clip_id}",
                    "metadata": {},
                    "source": [
                        f"## {clip_id}\n\n"
                        "Clip registry for this chapter is still being migrated. "
                        f"`load_chapter({chapter})` already loads builders.\n"
                    ],
                }
            )
            continue
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"export_{clip_id}",
                "metadata": {},
                "outputs": [],
                "source": [f"# {filename}\n", f"export_clip({filename!r})\n"],
            }
        )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "HITA (3.12)",
                "language": "python",
                "name": "hita",
            },
            "language_info": {"name": "python", "version": "3.12.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out = NOTEBOOKS_DIR / f"logistic-regression-chap{chapter}.ipynb"
    out.write_text(json.dumps(nb, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, default=0, help="1–7, or 0 for all")
    args = parser.parse_args()
    chapters = [args.chapter] if args.chapter else [1, 2, 3, 4, 5, 6, 7]
    for ch in chapters:
        print("wrote", gen_notebook(ch))


if __name__ == "__main__":
    main()
