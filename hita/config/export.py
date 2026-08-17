"""Export paths, output dir, and MP4 sizing helpers."""
from __future__ import annotations

import os
from pathlib import Path

SAVE_PAD_INCHES = 0.02
_OUTPUT_ENV = "HITA_OUTPUT_DIR"


def output_dir(root: Path | None = None) -> Path:
    """Resolve render output directory (overridable via ``HITA_OUTPUT_DIR``)."""
    raw = os.environ.get(_OUTPUT_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    base = Path(root) if root is not None else Path.cwd()
    path = (base / "renders").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def pad_to_macro_block(size: tuple[int, int], block: int = 16) -> tuple[int, int]:
    """Round width/height up to multiples of ``block`` (ffmpeg-friendly)."""
    w, h = int(size[0]), int(size[1])
    return (
        ((w + block - 1) // block) * block,
        ((h + block - 1) // block) * block,
    )


def fit_image_to_macro_block(img, block: int = 16):
    """Pad image with edge color so dims are divisible by ``block``."""
    from PIL import Image

    w, h = img.size
    tw, th = pad_to_macro_block((w, h), block=block)
    if (tw, th) == (w, h):
        return img
    canvas = Image.new(img.mode, (tw, th), img.getpixel((0, 0)))
    canvas.paste(img, (0, 0))
    return canvas
