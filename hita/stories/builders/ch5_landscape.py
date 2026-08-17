"""Additional Ch5 landscape FrameSpec builders (wave 2a).

``ch5_47`` remains the flagship parallel grid. Sibling clips share the same pack
builder when possible; others fall back to legacy+MP4 cache until fully migrated.
"""
from __future__ import annotations

from typing import Any

from hita.config.profile import active_profile
from hita.export.spec import FrameSpec
from hita.stories.builders.ch5_47 import build_pack_ch5_47


def build_pack_ch5_landscape_shared(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """Reuse the uniform grid pack (same meshes as ch5_47)."""
    pack = build_pack_ch5_47("ch5_47", ctx)
    pack["clip_id"] = clip_id
    return pack


def plan_ch5_47_from_pack(clip_id: str, ctx: dict[str, Any], **spec_kw: Any) -> list[FrameSpec]:
    import ch5_prior_landscape as cpl

    profile = active_profile()
    pack = ctx.get("_hita_pack")
    if pack is None:
        pack = build_pack_ch5_landscape_shared(clip_id, ctx)

    defaults = dict(
        n_cell_reveal_hold=profile.scale_motion(8, 3),
        n_seq_hold=int(ctx.get("CH5_HQ_N_SEQ_HOLD", 3)),
        n_annot_hold=profile.frame_count("annot_hold"),
        n_orbit=profile.frame_count("orbit_360"),
        opening_d1_zoom=True,
        n_open_zoom=profile.frame_count("grid_zoom"),
        n_open_zoom_hold=profile.frame_count("grid_zoom_hold"),
        hold_tail=profile.scale_motion(4, 2),
        clip_id=clip_id,
    )
    defaults.update(spec_kw)

    specs: list[FrameSpec] = []
    for i, sd in enumerate(cpl.ch5_iter_uniform_landscape_grid_specs(pack, **defaults)):
        specs.append(FrameSpec(index=i, kind=sd.get("kind", "landscape_grid"), params=sd))
    return specs
