"""Thin Ch5 story registry — parallel FrameSpec path + cached legacy fallback."""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from typing import Any

from hita.config.profile import active_profile
from hita.config.series import SeriesConfig
from hita.export.cache import (
    clip_result_key,
    code_hash,
    get_clip_mp4,
    put_clip_mp4,
)
from hita.export.jupyter_bridge import export_clip_safe
from hita.stories.registry import get_export_spec

_LEGACY_DEPS = (
    "ch5_datasets",
    "ch5_core",
    "ch5_layout",
    "ch5_prior_landscape",
    "ch5_story",
)


def _reinstall_into_context(mod) -> None:
    """After reload, restore ``_G`` / ``_PACKS`` from the live chapter context."""
    try:
        from hita.export.context import get_context

        g = get_context()
    except RuntimeError:
        return
    mod.install(g)


def _ch5_story(*, ensure_installed: bool = True):
    """Live ``ch5_story`` module; reinstall packs if a reload wiped them."""
    import ch5_story as mod

    if ensure_installed:
        packs = getattr(mod, "_PACKS", None)
        keys = getattr(mod, "CH5_DATASET_KEYS", ("D1", "D2", "D3", "D4"))
        if not (isinstance(packs, dict) and all(k in packs for k in keys)):
            _reinstall_into_context(mod)
            packs = getattr(mod, "_PACKS", None)
            if not (isinstance(packs, dict) and all(k in packs for k in keys)):
                # Hard reload + reinstall (notebook stale module graph)
                for name in _LEGACY_DEPS:
                    sys.modules.pop(name, None)
                mod = importlib.import_module("ch5_story")
                _reinstall_into_context(mod)
    return mod


def _export_specs():
    return list(_ch5_story(ensure_installed=False).CH5_EXPORT_SPECS)


def _resolve_legacy_filename(filename: str) -> tuple[str, str]:
    """Return ``(clip_id, filename)`` from the live export registry."""
    clip_key = filename.replace(".mp4", "")
    mod = _ch5_story()
    specs = list(mod.CH5_EXPORT_SPECS)
    for cid, fn, _ in specs:
        if fn == filename or cid == clip_key or fn.replace(".mp4", "") == clip_key:
            return cid, fn
    # Stale module: reload deps and reinstall into current chapter globals.
    for name in _LEGACY_DEPS:
        sys.modules.pop(name, None)
    mod = importlib.import_module("ch5_story")
    _reinstall_into_context(mod)
    specs = list(mod.CH5_EXPORT_SPECS)
    for cid, fn, _ in specs:
        if fn == filename or cid == clip_key or fn.replace(".mp4", "") == clip_key:
            return cid, fn
    known = ", ".join(s[0] for s in specs[-5:])
    raise KeyError(
        f"unknown export: {filename!r} "
        f"(registry has {len(specs)} clips; last are {known}. "
        f"Re-run the chapter setup cell after pulling library updates.)"
    )


def _legacy_export_cached(filename: str) -> Path:
    """Run legacy serial builder with durable MP4 cache."""
    clip_id, filename = _resolve_legacy_filename(filename)

    series = SeriesConfig.load()
    profile = active_profile()
    key = clip_result_key(
        series_version=series.version,
        profile=profile.name,
        clip_id=clip_id,
        code=code_hash(("ch5_story", "ch5_prior_landscape", "ch5_core")),
    )
    from hita.config.export import output_dir
    from hita.export.context import project_root

    root = project_root()
    cached = get_clip_mp4(5, key, root)
    out = output_dir(root) / filename
    if cached is not None:
        print(f"  legacy clip cache: hit {clip_id}", flush=True)
        if cached.resolve() != out.resolve():
            shutil.copy2(cached, out)
        return out

    path = _ch5_story().ch5_export_clip(filename)
    put_clip_mp4(5, key, path, root)
    return path


def export_clip(filename: str, *, workers: int | None = None, parallel: bool = True) -> Path:
    """Export one clip: parallel spec path when registered, else cached legacy serial."""
    clip_key = filename.replace(".mp4", "")
    try:
        spec = get_export_spec(clip_key, chapter=5)
    except KeyError:
        spec = None

    if spec is not None and parallel and spec.chapter == 5:
        from hita.export.pipeline import export_clip as _hita_export

        return _hita_export(spec.clip_id, workers=workers, chapter=5)

    return _legacy_export_cached(filename)


def install(globals_dict: dict[str, Any]) -> None:
    mod = _ch5_story(ensure_installed=False)
    mod.install(globals_dict)
    globals_dict["CH5_EXPORT_SPECS"] = list(mod.CH5_EXPORT_SPECS)
    globals_dict["export_clip"] = export_clip


def __getattr__(name: str):
    if name == "CH5_EXPORT_SPECS":
        return _export_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CH5_EXPORT_SPECS", "export_clip", "export_clip_safe", "install"]
