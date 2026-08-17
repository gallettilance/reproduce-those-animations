"""Versioned series DNA (layout, typography, theme)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "series_snapshot_v0.json"


@dataclass(frozen=True)
class CanvasConfig:
    canvas_h_in: float
    figsize: tuple[float, float]
    plot_w_frac: float
    plot_h_frac: float


@dataclass(frozen=True)
class TypographyConfig:
    font_size: int
    title_size: int


@dataclass(frozen=True)
class SeriesConfig:
    name: str
    version: str
    canvas: CanvasConfig
    typography: TypographyConfig
    theme: str

    @classmethod
    def load(cls, path: Path | None = None) -> SeriesConfig:
        snap: dict[str, Any] = json.loads((path or _SNAPSHOT_PATH).read_text())
        layout = snap["layout"]
        typo = snap["typography"]
        return cls(
            name=snap["name"],
            version=snap["version"],
            canvas=CanvasConfig(
                canvas_h_in=float(layout["canvas_h_in"]),
                figsize=(float(layout["figsize"][0]), float(layout["figsize"][1])),
                plot_w_frac=float(layout["plot_w_frac"]),
                plot_h_frac=float(layout["plot_h_frac"]),
            ),
            typography=TypographyConfig(
                font_size=int(typo["font_size"]),
                title_size=int(typo["title_size"]),
            ),
            theme=str(layout["theme"]),
        )
