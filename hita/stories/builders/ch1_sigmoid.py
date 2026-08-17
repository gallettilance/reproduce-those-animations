"""Ch1 sigmoid reveal pack + plan (parallel-ready)."""
from __future__ import annotations

from typing import Any

from hita.choreography.sequences import SigmoidRevealSequence
from hita.config.profile import active_profile
from hita.export.spec import FrameSpec
from hita.primitives.data_2d import ch1_survey_dataset
from hita.primitives.icons import load_icon_arrays
from hita.primitives.sigmoid_3d import SigmoidMesh


def build_pack_ch1_sigmoid_reveal(clip_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
    ds = ch1_survey_dataset()
    mesh = SigmoidMesh.build(ds.xlim, ds.ylim, n=160 if active_profile().is_draft() else 220)
    check, cross = load_icon_arrays()
    return {
        "clip_id": clip_id,
        "dataset": ds,
        "mesh": mesh,
        "check_icon": check,
        "cross_icon": cross,
        "figsize": tuple(ctx.get("EXPORT_FIGSIZE", (15.0, 9.5))),
        "dpi": int(ctx.get("EXPORT_DPI", active_profile().export_dpi)),
        "font_size": float(ctx.get("FONT_SIZE", 13.75)),
        "axis_label_size": float(ctx.get("AXIS_LABEL_SIZE", 15.0)),
    }


def plan_ch1_sigmoid_colormap_topdown(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Ch1 clip 75 DNA: flat→colormap morph → tilt to top-down → 32° turn."""
    profile = active_profile()
    seq = SigmoidRevealSequence(
        clip_id=clip_id,
        morph_n=profile.scale_motion(100, 12),
        flat_hold=profile.scale_motion(16, 3),
        top_hold=profile.scale_motion(12, 2),
        orbit_n=0,
    )
    return seq.plan()


def plan_ch1_sigmoid_colormap_orbit(clip_id: str, ctx: dict[str, Any]) -> list[FrameSpec]:
    """Ch1 clip 74 DNA: morph then full orbit on curved colormap surface."""
    profile = active_profile()
    seq = SigmoidRevealSequence(
        clip_id=clip_id,
        morph_n=profile.scale_motion(100, 12),
        flat_hold=profile.scale_motion(16, 3),
        top_hold=0,
        topdown_turn_deg=0.0,
        elev_top=26.0,  # skip tilt — orbit at scene elev
        orbit_n=profile.frame_count("orbit_360"),
        orbit_deg=360.0,
    )
    # Custom plan: morph at fixed elev, then orbit (override default tilt-heavy plan)
    specs: list[FrameSpec] = []
    idx = 0
    az0, elev0 = 25.0, 26.0

    def add(kind: str, **params: Any) -> None:
        nonlocal idx
        specs.append(FrameSpec(index=idx, kind=kind, params={"clip_id": clip_id, "kind": kind, "pass_surface": True, **params}))
        idx += 1

    from hita.primitives.math import smoothstep
    from hita.choreography.camera import orbit_azims

    for _ in range(seq.flat_hold):
        add("sigmoid_surface", morph_u=0.0, elev=elev0, azim=az0)
    for i in range(max(2, seq.morph_n)):
        u = i / (seq.morph_n - 1) if seq.morph_n > 1 else 1.0
        add("sigmoid_surface", morph_u=smoothstep(u), elev=elev0, azim=az0)
    for az in orbit_azims(az0, 360.0, seq.orbit_n, endpoint=False):
        add("sigmoid_surface", morph_u=1.0, elev=elev0, azim=float(az))
    return specs
