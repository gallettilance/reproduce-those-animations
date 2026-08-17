"""Packaged assets (fonts, icons) — prefer package paths over cwd."""
from __future__ import annotations

from pathlib import Path


def package_assets_dir() -> Path:
    return Path(__file__).resolve().parent


def fonts_dir() -> Path:
    return package_assets_dir() / "fonts"


def icons_dir() -> Path:
    return package_assets_dir() / "icons"


def knobs_dir() -> Path:
    """Bundled dial art: ``numbered/`` (1/2/3) and ``labeled/`` (w_ST / w_EL / b)."""
    return package_assets_dir() / "knobs"


def font_path(name: str) -> Path:
    """Resolve a bundled font by filename (e.g. ``PatrickHand-Regular.ttf``)."""
    path = fonts_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"bundled font missing: {path}")
    return path
