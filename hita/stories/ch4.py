"""Chapter 4 story shim."""
from __future__ import annotations

from pathlib import Path
from typing import Any

CH4_EXPORT_SPECS: list[tuple[str, str]] = []


def export_clip(filename: str, **kwargs) -> Path:
    raise NotImplementedError(
        "Chapter 4 clip registry migration in progress. "
        "load_chapter(4) already provides Ch3+Ch4 builders."
    )


def install(globals_dict: dict[str, Any]) -> None:
    globals_dict["CH4_EXPORT_SPECS"] = CH4_EXPORT_SPECS
    globals_dict["export_clip"] = export_clip
