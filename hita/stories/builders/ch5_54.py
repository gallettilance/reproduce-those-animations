"""Plan + pack for ch5_54 / ch5_56 grid 2D zoom + shadow orbit (parallel)."""
from __future__ import annotations

from typing import Any

from hita.config.profile import active_profile
from hita.export.spec import FrameSpec


def build_pack_ch5_54(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    import ch5_prior_landscape as cpl
    from ch5_story import _ch5_landscape_grid_datasets

    datasets = _ch5_landscape_grid_datasets()
    cfg = cpl.ch5_grid_landscape_config()
    fk = dict(ctx.get("CH5_HQ_LAND_FRAME_KW", {}))
    pack = cpl.ch5_build_grid_2d_zoom_shadow_orbit_pack(
        datasets, config=cfg, frame_kwargs=fk,
    )
    pack["clip_id"] = clip_id
    return pack


build_pack_ch5_56 = build_pack_ch5_54


def _plan_orbit(
    clip_id: str,
    ctx: dict[str, Any],
    *,
    camera_pan: bool,
) -> list[FrameSpec]:
    import ch5_prior_landscape as cpl

    profile = active_profile()
    pack = ctx.get("_hita_pack")
    if pack is None:
        pack = build_pack_ch5_54(clip_id, ctx)

    hold_tail = profile.scale_motion(4, 2)
    specs: list[FrameSpec] = []
    for i, sd in enumerate(
        cpl.ch5_iter_grid_2d_zoom_shadow_orbit_specs(
            pack,
            camera_pan=bool(camera_pan),
            hold_tail=hold_tail,
            clip_id=clip_id,
        )
    ):
        specs.append(FrameSpec(index=i, kind=sd.get("kind", "landscape_grid"), params=sd))
    return specs


def plan_ch5_54(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_orbit(clip_id, ctx, camera_pan=False)


def plan_ch5_56(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    return _plan_orbit(clip_id, ctx, camera_pan=True)
