"""Plan + pack for ch5_47 uniform landscape 2×2 grid (parallel target)."""
from __future__ import annotations

from typing import Any

from hita.config.profile import active_profile
from hita.export.spec import FrameSpec


def build_pack_ch5_47(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    import ch5_prior_landscape as cpl
    from ch5_story import _ch5_landscape_grid_datasets

    datasets = _ch5_landscape_grid_datasets()
    cfg = cpl.ch5_grid_landscape_config()
    fk = dict(ctx.get("CH5_HQ_LAND_FRAME_KW", {}))
    pack = cpl.ch5_build_uniform_landscape_grid_pack(datasets, config=cfg, frame_kwargs=fk)
    pack["clip_id"] = clip_id
    return pack


def plan_ch5_47(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Phase A: emit picklable frame specs (no rendering).

    Uses ``ctx["_hita_pack"]`` when provided by the export pipeline so meshes
    are not rebuilt after ``build_pack``.
    """
    import ch5_prior_landscape as cpl

    profile = active_profile()
    pack = ctx.get("_hita_pack")
    if pack is None:
        pack = build_pack_ch5_47(clip_id, ctx)
    hold = profile.scale_motion(4, 2)

    specs: list[FrameSpec] = []
    for i, sd in enumerate(
        cpl.ch5_iter_uniform_landscape_grid_specs(
            pack,
            n_cell_reveal_hold=profile.scale_motion(8, 3),
            n_seq_hold=int(ctx.get("CH5_HQ_N_SEQ_HOLD", 3)),
            n_annot_hold=profile.frame_count("annot_hold"),
            n_orbit=profile.frame_count("orbit_360"),
            opening_d1_zoom=True,
            n_open_zoom=profile.frame_count("grid_zoom"),
            n_open_zoom_hold=profile.frame_count("grid_zoom_hold"),
            hold_tail=hold,
            clip_id=clip_id,
        )
    ):
        specs.append(FrameSpec(index=i, kind=sd.get("kind", "landscape_grid"), params=sd))
    return specs
