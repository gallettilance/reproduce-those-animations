"""Chapter 3–5 story shims (builders still in legacy notebooks / modules)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def _make_shim(chapter: int, spec_name: str):
    EXPORT_SPECS: list[tuple[str, str]] = []

    def export_clip(filename: str, **kwargs) -> Path:
        raise NotImplementedError(
            f"Chapter {chapter} clip export via hita registry is in progress. "
            f"Setup via load_chapter({chapter}) already works — use legacy builders "
            f"from ctx.globals_dict, or migrate clips next."
        )

    def install(globals_dict: dict[str, Any]) -> None:
        globals_dict[spec_name] = EXPORT_SPECS
        globals_dict["export_clip"] = export_clip

    return EXPORT_SPECS, export_clip, install


CH3_EXPORT_SPECS, export_clip, install = _make_shim(3, "CH3_EXPORT_SPECS")
