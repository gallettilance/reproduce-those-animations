"""Chapter 9 — support vector machines (native hita)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hita.export.pipeline import export_clip as _pipeline_export
from hita.stories.builders.ch9_svm import (
    build_pack_ch9,
    plan_ch9_01,
    plan_ch9_02,
    plan_ch9_03,
    plan_ch9_04,
    plan_ch9_05,
    plan_ch9_06,
    plan_ch9_07,
    plan_ch9_08,
    plan_ch9_09,
    plan_ch9_10,
    plan_ch9_11,
    plan_ch9_12,
    plan_ch9_13,
    plan_ch9_14,
)
from hita.stories.registry import ExportSpec, get_export_spec, register

CH9_EXPORT_SPECS: list[tuple[str, str]] = [
    ("ch9_01", "ch9_01_scatter_fade_probability.mp4"),
    ("ch9_02", "ch9_02_thin_to_two_points.mp4"),
    ("ch9_03", "ch9_03_best_line_two_points.mp4"),
    ("ch9_04", "ch9_04_outside_points_hold.mp4"),
    ("ch9_05", "ch9_05_point_between.mp4"),
    ("ch9_06", "ch9_06_score_versus_distance.mp4"),
    ("ch9_07", "ch9_07_objective_slice.mp4"),
    ("ch9_08", "ch9_08_voxel_constraints.mp4"),
    ("ch9_09", "ch9_09_lagrangian_weights.mp4"),
    ("ch9_10", "ch9_10_dual_meets_primal.mp4"),
    ("ch9_11", "ch9_11_reevaluate_program.mp4"),
    ("ch9_12", "ch9_12_soft_margin_hinge.mp4"),
    ("ch9_13", "ch9_13_hinge_versus_logistic.mp4"),
    ("ch9_14", "ch9_14_kernel_question.mp4"),
]

_PLANS = {
    "ch9_01": plan_ch9_01,
    "ch9_02": plan_ch9_02,
    "ch9_03": plan_ch9_03,
    "ch9_04": plan_ch9_04,
    "ch9_05": plan_ch9_05,
    "ch9_06": plan_ch9_06,
    "ch9_07": plan_ch9_07,
    "ch9_08": plan_ch9_08,
    "ch9_09": plan_ch9_09,
    "ch9_10": plan_ch9_10,
    "ch9_11": plan_ch9_11,
    "ch9_12": plan_ch9_12,
    "ch9_13": plan_ch9_13,
    "ch9_14": plan_ch9_14,
}


def _register() -> None:
    for clip_id, filename in CH9_EXPORT_SPECS:
        register(
            ExportSpec(
                clip_id=clip_id,
                filename=filename,
                builder_plan=_PLANS[clip_id],
                build_pack=build_pack_ch9,
                render_fn="hita.export.renderers.render_ch9_frame",
                ms_per_frame=50,
                chapter=9,
                tags=("svm", clip_id),
                requires_frame_spec=True,
                # Holds are repeated identical specs; keep them so a pose can be read.
                dedupe_consecutive=False,
            )
        )


_register()


def export_clip(filename: str, *, workers: int | None = None, parallel: bool = True) -> Path:
    del parallel
    clip_key = filename.replace(".mp4", "")
    spec = get_export_spec(clip_key, chapter=9)
    return _pipeline_export(spec.clip_id, workers=workers, chapter=9)


def install(globals_dict: dict[str, Any]) -> None:
    globals_dict.update(
        {
            "CH9_EXPORT_SPECS": CH9_EXPORT_SPECS,
            "export_clip": export_clip,
        }
    )


__all__ = ["CH9_EXPORT_SPECS", "export_clip", "install", "build_pack_ch9"]
