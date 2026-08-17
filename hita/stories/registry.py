"""Export registry and ExportSpec entries."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hita.config.series import SeriesConfig
from hita.export.context import project_root
from hita.export.spec import FrameSpec

SERIES_VERSION = SeriesConfig.load().version


@dataclass(frozen=True)
class ExportSpec:
    clip_id: str
    filename: str
    builder_plan: Callable[[str, dict[str, Any]], list[FrameSpec]]
    build_pack: Callable[[str, dict[str, Any]], dict[str, Any]]
    render_fn: str = "hita.export.renderers.render_sigmoid_frame"
    ms_per_frame: int | None = None
    chapter: int = 5
    root: Path = field(default_factory=project_root)
    tags: tuple[str, ...] = ()
    series_version: str = SERIES_VERSION
    dedupe_consecutive: bool = True
    requires_frame_spec: bool = False  # enforced True for chapter >= 7


def register(spec: ExportSpec) -> None:
    if spec.chapter >= 7 or spec.requires_frame_spec:
        # New chapters must use the two-phase contract.
        if not callable(spec.builder_plan) or not callable(spec.build_pack):
            raise ValueError(f"{spec.clip_id}: chapter>={spec.chapter} requires plan+pack callables")
    _REGISTRY[spec.clip_id] = spec
    _REGISTRY[spec.filename] = spec
    stem = spec.filename.replace(".mp4", "")
    _REGISTRY[stem] = spec


_REGISTRY: dict[str, ExportSpec] = {}


def get_export_spec(clip_id: str, *, chapter: int = 0) -> ExportSpec:
    key = clip_id
    if key not in _REGISTRY and not key.endswith(".mp4"):
        # try bare and with .mp4
        for candidate in (key, f"{key}.mp4"):
            if candidate in _REGISTRY:
                return _REGISTRY[candidate]
    if key not in _REGISTRY:
        for reg_key, spec in _REGISTRY.items():
            if clip_id in (reg_key, spec.clip_id, spec.filename):
                if chapter and spec.chapter != chapter:
                    continue
                return spec
        raise KeyError(f"unknown export: {clip_id!r}")
    return _REGISTRY[key]


def list_exports(chapter: int | None = None) -> list[ExportSpec]:
    seen: set[str] = set()
    out: list[ExportSpec] = []
    for spec in _REGISTRY.values():
        if spec.clip_id in seen:
            continue
        if chapter is not None and spec.chapter != chapter:
            continue
        seen.add(spec.clip_id)
        out.append(spec)
    return sorted(out, key=lambda s: (s.chapter, s.clip_id))


def _register_all() -> None:
    from hita.stories.builders.ch5_47 import build_pack_ch5_47, plan_ch5_47
    from hita.stories.builders.ch5_54 import (
        build_pack_ch5_54,
        build_pack_ch5_56,
        plan_ch5_54,
        plan_ch5_56,
    )
    from hita.stories.builders.ch5_57 import build_pack_ch5_57, plan_ch5_57

    register(
        ExportSpec(
            clip_id="ch5_47",
            filename="ch5_47_uniform_landscape_grid_2x2.mp4",
            builder_plan=plan_ch5_47,
            build_pack=build_pack_ch5_47,
            render_fn="hita.export.renderers.render_ch5_frame",
            ms_per_frame=90,
            chapter=5,
            tags=("landscape", "grid_2x2", "uniform"),
        )
    )
    register(
        ExportSpec(
            clip_id="ch5_54",
            filename="ch5_54_grid_2d_zoom_shadow_orbit.mp4",
            builder_plan=plan_ch5_54,
            build_pack=build_pack_ch5_54,
            render_fn="hita.export.renderers.render_ch5_frame",
            ms_per_frame=90,
            chapter=5,
            dedupe_consecutive=False,
            tags=("landscape", "grid_2x2", "zoom", "shadow_orbit"),
        )
    )
    register(
        ExportSpec(
            clip_id="ch5_56",
            filename="ch5_56_grid_2d_zoom_shadow_orbit_topview.mp4",
            builder_plan=plan_ch5_56,
            build_pack=build_pack_ch5_56,
            render_fn="hita.export.renderers.render_ch5_frame",
            ms_per_frame=90,
            chapter=5,
            dedupe_consecutive=False,
            tags=("landscape", "grid_2x2", "zoom", "shadow_orbit", "topview"),
        )
    )
    register(
        ExportSpec(
            clip_id="ch5_57",
            filename="ch5_57_grid_map_labeled_rotate90.mp4",
            builder_plan=plan_ch5_57,
            build_pack=build_pack_ch5_57,
            render_fn="hita.export.renderers.render_ch5_frame",
            ms_per_frame=90,
            chapter=5,
            dedupe_consecutive=False,
            tags=("landscape", "grid_2x2", "map_label", "rotate90"),
        )
    )
    # Wave 2a: sibling landscape clip reuses shared pack + same plan DNA
    from hita.stories.builders.ch5_landscape import (
        build_pack_ch5_landscape_shared,
        plan_ch5_47_from_pack,
    )

    def _plan_alias(clip_id: str, ctx: dict):
        return plan_ch5_47_from_pack(clip_id, ctx)

    # Note: ch5_48–51 / 53 / 55 still use legacy+MP4 cache until dedicated
    # FrameSpec planners land. Parallel: ch5_47, ch5_54, ch5_56, ch5_57.
    _ = (build_pack_ch5_landscape_shared, _plan_alias)

    import hita.stories.ch1  # noqa: F401
    import hita.stories.ch2  # noqa: F401
    import hita.stories.ch7  # noqa: F401


_register_all()
