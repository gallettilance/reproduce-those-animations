"""Chapter 7 — first chapter born on hita (no notebook exec)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hita.export.pipeline import export_clip as _pipeline_export
from hita.export.spec import FrameSpec
from hita.stories.registry import ExportSpec, get_export_spec, register

CH7_EXPORT_SPECS: list[tuple[str, str]] = [
    ("ch7_01", "ch7_01_sigmoid_reveal_demo.mp4"),
]


def build_pack_ch7_01(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """Pure-library pack — no legacy notebook state required."""
    from hita.config.profile import active_profile
    from hita.primitives.data_2d import ch1_survey_dataset
    from hita.primitives.icons import load_icon_arrays
    from hita.primitives.sigmoid_3d import SigmoidMesh

    ds = ch1_survey_dataset()
    profile = active_profile()
    mesh = SigmoidMesh.build(ds.xlim, ds.ylim, n=120 if profile.is_draft() else 180)
    check, cross = load_icon_arrays()
    return {
        "clip_id": clip_id,
        "dataset": ds,
        "mesh": mesh,
        "check_icon": check,
        "cross_icon": cross,
        "figsize": (15.0, 9.5),
        "dpi": profile.export_dpi,
        "font_size": 13.75,
        "axis_label_size": 15.0,
    }


def plan_ch7_01(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    from hita.choreography.sequences import SigmoidRevealSequence
    from hita.config.profile import active_profile

    profile = active_profile()
    seq = SigmoidRevealSequence(
        clip_id=clip_id,
        morph_n=profile.scale_motion(48, 8),
        flat_hold=profile.scale_motion(8, 2),
        top_hold=profile.scale_motion(6, 2),
        orbit_n=0,
    )
    return seq.plan()


def _register() -> None:
    register(
        ExportSpec(
            clip_id="ch7_01",
            filename="ch7_01_sigmoid_reveal_demo.mp4",
            builder_plan=plan_ch7_01,
            build_pack=build_pack_ch7_01,
            render_fn="hita.export.renderers.render_sigmoid_frame",
            ms_per_frame=40,
            chapter=7,
            tags=("sigmoid_3d", "demo", "native"),
            requires_frame_spec=True,
            dedupe_consecutive=True,
        )
    )


_register()


def export_clip(filename: str, *, workers: int | None = None, parallel: bool = True) -> Path:
    clip_key = filename.replace(".mp4", "")
    spec = get_export_spec(clip_key, chapter=7)
    return _pipeline_export(spec.clip_id, workers=workers, chapter=7)


def install(globals_dict: dict[str, Any]) -> None:
    from hita.choreography import SigmoidRevealSequence
    from hita.primitives import CMAP, SigmoidMesh, ch1_survey_dataset, sigmoid

    globals_dict.update(
        {
            "sigmoid": sigmoid,
            "CMAP": CMAP,
            "SigmoidMesh": SigmoidMesh,
            "SigmoidRevealSequence": SigmoidRevealSequence,
            "ch1_survey_dataset": ch1_survey_dataset,
            "CH7_EXPORT_SPECS": CH7_EXPORT_SPECS,
            "export_clip": export_clip,
        }
    )


__all__ = ["CH7_EXPORT_SPECS", "export_clip", "install", "build_pack_ch7_01", "plan_ch7_01"]
