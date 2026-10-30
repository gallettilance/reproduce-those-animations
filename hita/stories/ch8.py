"""Chapter 8 — linear regression with plants (native hita)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hita.export.pipeline import export_clip as _pipeline_export
from hita.stories.builders.ch8_linreg import (
    build_pack_ch8,
    plan_ch8_00,
    plan_ch8_01,
    plan_ch8_01b,
    plan_ch8_01c,
    plan_ch8_01d,
    plan_ch8_01e,
    plan_ch8_01f,
    plan_ch8_01g,
    plan_ch8_02,
    plan_ch8_02b,
    plan_ch8_02c,
    plan_ch8_03,
    plan_ch8_03b,
    plan_ch8_04a,
    plan_ch8_04b,
    plan_ch8_04,
    plan_ch8_05,
    plan_ch8_05b,
    plan_ch8_05c,
    plan_ch8_06,
    plan_ch8_07,
    plan_ch8_08,
    plan_ch8_09,
    plan_ch8_10,
    plan_ch8_11,
)
from hita.stories.registry import ExportSpec, get_export_spec, register

CH8_EXPORT_SPECS: list[tuple[str, str]] = [
    ("ch8_00", "ch8_00_measure_plant_heights.mp4"),
    ("ch8_01", "ch8_01_plants_linear_points_line.mp4"),
    ("ch8_01b", "ch8_01b_slope_rise_run.mp4"),
    ("ch8_01c", "ch8_01c_line_through_mean.mp4"),
    ("ch8_01d", "ch8_01d_average_vs_line.mp4"),
    ("ch8_01e", "ch8_01e_predict_and_extrapolate.mp4"),
    ("ch8_01f", "ch8_01f_fitted_and_residual.mp4"),
    ("ch8_01g", "ch8_01g_r_squared.mp4"),
    ("ch8_02", "ch8_02_height18_histogram_bell.mp4"),
    ("ch8_02b", "ch8_02b_height18_samples_then_hist.mp4"),
    ("ch8_02c", "ch8_02c_height18_spread_variance.mp4"),
    ("ch8_03", "ch8_03_all_heights_conditional_means.mp4"),
    ("ch8_03b", "ch8_03b_conditional_distribution_shapes.mp4"),
    ("ch8_04a", "ch8_04a_residual_squares.mp4"),
    ("ch8_04b", "ch8_04b_squares_vs_absolute.mp4"),
    ("ch8_04", "ch8_04_sse_curve_two_points.mp4"),
    ("ch8_05", "ch8_05_sse_surface_orbit_minimum.mp4"),
    ("ch8_05b", "ch8_05b_shift_and_tilt.mp4"),
    ("ch8_05c", "ch8_05c_leverage.mp4"),
    ("ch8_06", "ch8_06_quadratic_points_in.mp4"),
    ("ch8_07", "ch8_07_quadratic_residual_plot.mp4"),
    ("ch8_08", "ch8_08_linear_fill_points_in.mp4"),
    ("ch8_09", "ch8_09_linear_fill_residual_plot.mp4"),
    ("ch8_10", "ch8_10_heteroscedastic_points_in.mp4"),
    ("ch8_11", "ch8_11_heteroscedastic_residual_plot.mp4"),
]

_PLANS = {
    "ch8_00": plan_ch8_00,
    "ch8_01": plan_ch8_01,
    "ch8_01b": plan_ch8_01b,
    "ch8_01c": plan_ch8_01c,
    "ch8_01d": plan_ch8_01d,
    "ch8_01e": plan_ch8_01e,
    "ch8_01f": plan_ch8_01f,
    "ch8_01g": plan_ch8_01g,
    "ch8_02": plan_ch8_02,
    "ch8_02b": plan_ch8_02b,
    "ch8_02c": plan_ch8_02c,
    "ch8_03": plan_ch8_03,
    "ch8_03b": plan_ch8_03b,
    "ch8_04a": plan_ch8_04a,
    "ch8_04b": plan_ch8_04b,
    "ch8_04": plan_ch8_04,
    "ch8_05": plan_ch8_05,
    "ch8_05b": plan_ch8_05b,
    "ch8_05c": plan_ch8_05c,
    "ch8_06": plan_ch8_06,
    "ch8_07": plan_ch8_07,
    "ch8_08": plan_ch8_08,
    "ch8_09": plan_ch8_09,
    "ch8_10": plan_ch8_10,
    "ch8_11": plan_ch8_11,
}

_TAGS = {
    "ch8_00": ("linreg", "plants", "measure"),
    "ch8_01": ("linreg", "plants", "scatter"),
    "ch8_01b": ("linreg", "slope", "intercept"),
    "ch8_01c": ("linreg", "centroid"),
    "ch8_01d": ("linreg", "residuals", "mean"),
    "ch8_01e": ("linreg", "prediction", "extrapolation"),
    "ch8_01f": ("linreg", "fitted", "residual"),
    "ch8_01g": ("linreg", "r_squared"),
    "ch8_02": ("linreg", "histogram", "conditional"),
    "ch8_02b": ("linreg", "histogram", "samples", "conditional"),
    "ch8_02c": ("linreg", "histogram", "variance"),
    "ch8_03": ("linreg", "conditional_means"),
    "ch8_03b": ("linreg", "histogram", "conditional", "shapes"),
    "ch8_04a": ("linreg", "sse", "squares"),
    "ch8_04b": ("linreg", "sse", "l1", "l2"),
    "ch8_04": ("linreg", "sse", "curve"),
    "ch8_05": ("linreg", "sse", "surface", "orbit"),
    "ch8_05b": ("linreg", "intercept", "slope"),
    "ch8_05c": ("linreg", "leverage"),
    "ch8_06": ("linreg", "quadratic"),
    "ch8_07": ("linreg", "residuals", "quadratic"),
    "ch8_08": ("linreg", "residuals", "linear"),
    "ch8_09": ("linreg", "residuals", "linear"),
    "ch8_10": ("linreg", "heteroscedastic"),
    "ch8_11": ("linreg", "residuals", "heteroscedastic"),
}


def _register() -> None:
    for clip_id, filename in CH8_EXPORT_SPECS:
        register(
            ExportSpec(
                clip_id=clip_id,
                filename=filename,
                builder_plan=_PLANS[clip_id],
                build_pack=build_pack_ch8,
                render_fn="hita.export.renderers.render_ch8_frame",
                ms_per_frame=50,
                chapter=8,
                tags=_TAGS[clip_id],
                requires_frame_spec=True,
                dedupe_consecutive=True,
            )
        )


_register()


def export_clip(filename: str, *, workers: int | None = None, parallel: bool = True) -> Path:
    del parallel
    clip_key = filename.replace(".mp4", "")
    spec = get_export_spec(clip_key, chapter=8)
    return _pipeline_export(spec.clip_id, workers=workers, chapter=8)


def install(globals_dict: dict[str, Any]) -> None:
    from hita.primitives.linreg import plant_datasets
    from hita.primitives.plants import make_plant_icon

    globals_dict.update(
        {
            "CH8_EXPORT_SPECS": CH8_EXPORT_SPECS,
            "export_clip": export_clip,
            "plant_datasets": plant_datasets,
            "make_plant_icon": make_plant_icon,
        }
    )


__all__ = ["CH8_EXPORT_SPECS", "export_clip", "install", "build_pack_ch8"]
