"""Chapter 2 story registry — param morph / reverse-77 / knob motions (scaffolded)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hita.export.pipeline import export_clip as _pipeline_export
from hita.stories.builders.ch1_sigmoid import (
    build_pack_ch1_sigmoid_reveal,
    plan_ch1_sigmoid_colormap_orbit,
)
from hita.stories.registry import ExportSpec, get_export_spec, register

# Ch2 reuses the shared sigmoid morph DNA with Ch2 camera (-152° elev 32°).
CH2_EXPORT_SPECS: list[tuple[str, str]] = [
    ("ch2_03", "ch2_03_dataset_st_el_3d_sigmoid_morph.mp4"),
]


def plan_ch2_03(clip_id: str, ctx: dict[str, Any]):
    """Ch2 morph-style orbit at Ch2 camera (elev 32, azim −152)."""
    from hita.choreography.camera import orbit_azims
    from hita.config.profile import active_profile
    from hita.export.spec import FrameSpec
    from hita.primitives.math import smoothstep

    profile = active_profile()
    morph_n = profile.scale_motion(52, 8)
    hold = profile.scale_motion(14, 3)
    orbit_n = profile.frame_count("orbit_360")
    elev, az0 = 32.0, -152.0
    specs: list[FrameSpec] = []
    idx = 0

    def add(**params):
        nonlocal idx
        specs.append(
            FrameSpec(
                index=idx,
                kind="sigmoid_surface",
                params={"clip_id": clip_id, "kind": "sigmoid_surface", "pass_surface": True, **params},
            )
        )
        idx += 1

    for _ in range(hold):
        add(morph_u=0.0, elev=elev, azim=az0)
    for i in range(max(2, morph_n)):
        u = i / (morph_n - 1) if morph_n > 1 else 1.0
        add(morph_u=smoothstep(u), elev=elev, azim=az0)
    for az in orbit_azims(az0, 62.0, max(8, orbit_n // 4), endpoint=True):
        add(morph_u=1.0, elev=elev, azim=float(az))
    return specs


def _register() -> None:
    register(
        ExportSpec(
            clip_id="ch2_03",
            filename="ch2_03_dataset_st_el_3d_sigmoid_morph.mp4",
            builder_plan=plan_ch2_03,
            build_pack=build_pack_ch1_sigmoid_reveal,
            render_fn="hita.export.renderers.render_sigmoid_frame",
            ms_per_frame=40,
            chapter=2,
            tags=("sigmoid_3d", "morph", "ch2"),
        )
    )


_register()


def export_clip(filename: str, *, workers: int | None = None, parallel: bool = True) -> Path:
    clip_key = filename.replace(".mp4", "")
    spec = get_export_spec(clip_key, chapter=2)
    return _pipeline_export(spec.clip_id, workers=workers, chapter=2)


def install(globals_dict: dict[str, Any]) -> None:
    from hita.stories import ch1 as ch1_story

    ch1_story.install(globals_dict)
    globals_dict["CH2_EXPORT_SPECS"] = CH2_EXPORT_SPECS
    globals_dict["export_clip"] = export_clip


__all__ = ["CH2_EXPORT_SPECS", "export_clip", "install"]
