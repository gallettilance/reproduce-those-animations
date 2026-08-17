"""MP4 encoding wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hita.config.export import OUTPUT_DIR


def save_mp4(frames: Iterable, filename: str, *, duration: int = 95, output_dir: Path | None = None) -> Path:
    """Encode frames to MP4 using imageio (delegates to chapter save_mp4 when available)."""
    try:
        from hita.export.context import get_context

        ctx = get_context()
        ctx["save_mp4"](list(frames) if not isinstance(frames, list) else frames, filename, duration=duration)
        return Path(ctx.get("OUTPUT_DIR", output_dir or OUTPUT_DIR)) / filename
    except RuntimeError:
        import imageio.v2 as imageio

        out = Path(output_dir or OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        frame_list = list(frames)
        imageio.mimsave(path, frame_list, fps=1000.0 / float(duration))
        return path
