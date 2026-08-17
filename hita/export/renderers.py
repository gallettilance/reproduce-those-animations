"""Top-level picklable render functions for worker processes."""
from __future__ import annotations

from typing import Any


def render_ch5_frame(pack: dict[str, Any], spec: dict[str, Any]):
    """Render one Ch6 frame from a spec dict."""
    from hita.export.context import load_export_context

    ctx = load_export_context(chapter=5)
    kind = spec.get("kind", "landscape_grid")
    clip_id = spec.get("clip_id", "ch5_47")

    import ch5_prior_landscape as cpl

    if kind == "landscape_grid":
        img = cpl.ch5_render_landscape_grid_from_spec(pack, spec)
    elif kind == "quadrant_zoom":
        img = cpl.ch5_quadrant_zoom_frame(
            spec["grid_image"],
            int(spec["row"]),
            int(spec["col"]),
            float(spec["zoom_u"]),
        )
    elif kind == "raw_image":
        img = spec["image"]
    else:
        raise ValueError(f"unknown frame kind: {kind!r}")

    finish = ctx.get("_ch5_finish_duo_export")
    if finish is not None:
        return finish(img, clip_id)
    return img


def render_sigmoid_frame(pack: dict[str, Any], spec: dict[str, Any]):
    """Render one Ch1/Ch2 sigmoid-surface frame (flat→curve / orbit / top-down)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from hita.export.raster import fig_to_image
    from hita.primitives.colormap import CMAP
    from hita.primitives.sigmoid_3d import (
        draw_sigmoid_surface,
        scatter_outcome_icons_3d,
        style_sigmoid_axes,
    )

    mesh = pack["mesh"]
    ds = pack["dataset"]
    morph_u = float(spec.get("morph_u", 1.0))
    elev = float(spec.get("elev", 26.0))
    azim = float(spec.get("azim", 25.0))
    pass_surface = bool(spec.get("pass_surface", True))
    Z = mesh.morph_z(morph_u, pass_surface=pass_surface)
    z_pts = morph_u * (np.asarray(
        __import__("hita.primitives.math", fromlist=["sigmoid"]).sigmoid(
            ds.diff if pass_surface else -ds.diff
        ),
        dtype=float,
    ))

    fig = plt.figure(figsize=pack.get("figsize", (15.0, 9.5)))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.96)
    draw_sigmoid_surface(ax, mesh, Z, cmap=CMAP, alpha=0.32)
    scatter_outcome_icons_3d(
        ax,
        ds.study,
        ds.exam,
        ds.y,
        z_pts,
        check_icon=pack["check_icon"],
        cross_icon=pack["cross_icon"],
        xlim=ds.xlim,
        ylim=ds.ylim,
        rotate_icons_180=True,
    )
    style_sigmoid_axes(
        ax,
        azim,
        elev=elev,
        xlim=ds.xlim,
        ylim=ds.ylim,
        hide_z=bool(spec.get("hide_z", False)),
        exam_label_2d=bool(spec.get("exam_label_2d", False)),
        font_size=float(pack.get("font_size", 13.75)),
        axis_label_size=float(pack.get("axis_label_size", 15.0)),
    )
    return fig_to_image(fig, dpi=int(pack.get("dpi", 200)))


def render_ch7_demo_frame(pack: dict[str, Any], spec: dict[str, Any]):
    """Alias — Ch7 demo uses the shared sigmoid renderer."""
    return render_sigmoid_frame(pack, spec)
