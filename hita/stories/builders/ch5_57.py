"""Plan + pack for ch5_57 labeled 2×2 belief grid + 90° CCW rotate."""
from __future__ import annotations

from typing import Any

from hita.config.profile import active_profile
from hita.export.spec import FrameSpec
from hita.stories.builders.ch5_54 import build_pack_ch5_54


build_pack_ch5_57 = build_pack_ch5_54


def plan_ch5_57(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    import ch5_prior_landscape as cpl

    profile = active_profile()
    pack = ctx.get("_hita_pack")
    if pack is None:
        pack = build_pack_ch5_54(clip_id, ctx)

    hold_tail = profile.scale_motion(4, 2)
    specs: list[FrameSpec] = []
    for i, sd in enumerate(
        cpl.ch5_iter_grid_map_labeled_rotate90_specs(
            pack,
            hold_tail=hold_tail,
            clip_id=clip_id,
        )
    ):
        specs.append(FrameSpec(index=i, kind=sd.get("kind", "landscape_grid"), params=sd))
    return specs
