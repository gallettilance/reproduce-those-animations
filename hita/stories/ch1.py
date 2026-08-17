"""Chapter 1 story registry — sigmoid 3D / colormap / rotation (Scene 8)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hita.export.pipeline import export_clip as _pipeline_export
from hita.stories.registry import ExportSpec, get_export_spec, list_exports, register

CH1_EXPORT_SPECS: list[tuple[str, str, str]] = [
    ("ch1_74", "ch1_74_sigmoid_pass_colormap_orbit.mp4", "orbit"),
    ("ch1_75", "ch1_75_sigmoid_colormap_to_topdown.mp4", "topdown"),
]


def _register() -> None:
    from hita.stories.builders.ch1_sigmoid import (
        build_pack_ch1_sigmoid_reveal,
        plan_ch1_sigmoid_colormap_orbit,
        plan_ch1_sigmoid_colormap_topdown,
    )

    register(
        ExportSpec(
            clip_id="ch1_74",
            filename="ch1_74_sigmoid_pass_colormap_orbit.mp4",
            builder_plan=plan_ch1_sigmoid_colormap_orbit,
            build_pack=build_pack_ch1_sigmoid_reveal,
            render_fn="hita.export.renderers.render_sigmoid_frame",
            ms_per_frame=32,
            chapter=1,
            tags=("sigmoid_3d", "colormap", "orbit"),
        )
    )
    register(
        ExportSpec(
            clip_id="ch1_75",
            filename="ch1_75_sigmoid_colormap_to_topdown.mp4",
            builder_plan=plan_ch1_sigmoid_colormap_topdown,
            build_pack=build_pack_ch1_sigmoid_reveal,
            render_fn="hita.export.renderers.render_sigmoid_frame",
            ms_per_frame=32,
            chapter=1,
            tags=("sigmoid_3d", "colormap", "topdown"),
        )
    )


_register()


def export_clip(filename: str, *, workers: int | None = None, parallel: bool = True) -> Path:
    clip_key = filename.replace(".mp4", "")
    spec = get_export_spec(clip_key, chapter=1)
    return _pipeline_export(spec.clip_id, workers=workers, chapter=1)


def install(globals_dict: dict[str, Any]) -> None:
    """Inject Ch1 helpers into a chapter context."""
    from hita.primitives import (
        CMAP,
        Dataset2D,
        SigmoidMesh,
        ch1_survey_dataset,
        sigmoid,
        style_sigmoid_axes,
    )
    from hita.choreography import SigmoidRevealSequence

    globals_dict.update(
        {
            "sigmoid": sigmoid,
            "CMAP": CMAP,
            "ch1_survey_dataset": ch1_survey_dataset,
            "SigmoidMesh": SigmoidMesh,
            "Dataset2D": Dataset2D,
            "style_sigmoid_axes": style_sigmoid_axes,
            "SigmoidRevealSequence": SigmoidRevealSequence,
            "CH1_EXPORT_SPECS": CH1_EXPORT_SPECS,
            "export_clip": export_clip,
        }
    )


__all__ = ["CH1_EXPORT_SPECS", "export_clip", "install", "list_exports"]
