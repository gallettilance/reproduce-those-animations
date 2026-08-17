"""Chapter context loader — replaces exec(notebook) inheritance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hita.config.profile import RenderProfile, apply_profile_env
from hita.config.series import SeriesConfig
from hita.export.context import load_export_context, project_root

_DEFAULT_INHERITS = {
    1: [],
    2: [1],
    3: [],
    4: [3],
    5: [3, 4],
    6: [3, 4, 5],
    7: [],  # native — no notebook inheritance
}


@dataclass
class ChapterContext:
    chapter: int
    profile: RenderProfile
    series: SeriesConfig
    globals_dict: dict[str, Any]
    root: Path

    @property
    def native(self) -> bool:
        return bool(self.globals_dict.get("_HITA_NATIVE"))

    def get(self, name: str, default: Any = None) -> Any:
        return self.globals_dict.get(name, default)

    def __getitem__(self, name: str) -> Any:
        return self.globals_dict[name]


def load_chapter(
    chapter: int,
    *,
    inherits: list[int] | None = None,
    profile: str | None = None,
    root: Path | str | None = None,
) -> ChapterContext:
    """Load chapter builders.

    Chapters ``>= 7`` are **native**: pure Python imports, no ``exec(notebook)``.
    """
    import os

    if profile:
        os.environ["HITA_RENDER_PROFILE"] = profile
    prof = apply_profile_env()
    root_path = Path(root or project_root()).resolve()
    g = load_export_context(root_path, chapter=int(chapter))
    g["_HITA_PROFILE"] = prof.name
    g["_HITA_INHERITS"] = (
        list(inherits) if inherits is not None else list(_DEFAULT_INHERITS.get(int(chapter), []))
    )

    return ChapterContext(
        chapter=int(chapter),
        profile=prof,
        series=SeriesConfig.load(),
        globals_dict=g,
        root=root_path,
    )
