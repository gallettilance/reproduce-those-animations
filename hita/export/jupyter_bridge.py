"""Subprocess bridge when Jupyter spawn is unsafe (macOS default)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hita.config.profile import _spawn_unsafe


def export_via_subprocess(
    clip_id: str,
    *,
    workers: int | None = None,
    root: Path | str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``python -m hita.export`` in a child process."""
    root = Path(root or Path.cwd()).resolve()
    cmd = [sys.executable, "-m", "hita.export", clip_id]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    env = os.environ.copy()
    return subprocess.run(cmd, cwd=str(root), env=env, check=False)


def export_clip_safe(clip_id: str, *, workers: int | None = None, root: Path | str | None = None):
    """Use subprocess from notebooks; in-process from scripts."""
    if _spawn_unsafe():
        result = export_via_subprocess(clip_id, workers=workers, root=root)
        if result.returncode != 0:
            raise RuntimeError(f"hita export subprocess failed (exit {result.returncode})")
        return None
    from hita.export.pipeline import export_clip

    return export_clip(clip_id, workers=workers, root=root)
