"""Outcome icons (check/cross) used across chapters."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hita.assets import icons_dir
from hita.config.export import output_dir
from hita.export.context import project_root


def ensure_outcome_icons(output_dir_path: Path | None = None, size: int = 120, line_width: int = 15) -> tuple[Path, Path]:
    out = Path(output_dir_path or output_dir(project_root()))
    out.mkdir(parents=True, exist_ok=True)
    check = out / "check.png"
    cross = out / "cross.png"

    bundled = icons_dir()
    for name, dest in (("check.png", check), ("cross.png", cross)):
        src = bundled / name
        if src.is_file() and (not dest.is_file() or dest.stat().st_size < 16):
            shutil.copy2(src, dest)

    if check.is_file() and cross.is_file() and check.stat().st_size > 16 and cross.stat().st_size > 16:
        return check, cross

    sc = size / 96.0
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.line(
        [
            (int(round(20 * sc)), int(round(55 * sc))),
            (int(round(42 * sc)), int(round(76 * sc))),
            (int(round(78 * sc)), int(round(24 * sc))),
        ],
        fill=(44, 160, 44, 255),
        width=line_width,
    )
    img.save(check)

    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.line(
        [(int(round(24 * sc)), int(round(24 * sc))), (int(round(72 * sc)), int(round(72 * sc)))],
        fill=(214, 39, 40, 255),
        width=line_width,
    )
    draw.line(
        [(int(round(72 * sc)), int(round(24 * sc))), (int(round(24 * sc)), int(round(72 * sc)))],
        fill=(214, 39, 40, 255),
        width=line_width,
    )
    img.save(cross)
    return check, cross


def load_icon_arrays(output_dir_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    check, cross = ensure_outcome_icons(output_dir_path)
    return (
        np.asarray(Image.open(check).convert("RGBA")),
        np.asarray(Image.open(cross).convert("RGBA")),
    )
