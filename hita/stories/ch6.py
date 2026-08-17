"""Chapter 6 story shim — frequentist sampling variability."""
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

_LEGACY_DEPS = (
    "ch6_frequentist",
    "ch6_story",
    "ch5_datasets",
)


def _reinstall_into_context(mod) -> None:
    try:
        from hita.export.context import get_context

        g = get_context()
    except RuntimeError:
        return
    mod.install(g)


def _ch6_story(*, ensure_installed: bool = True):
    import ch6_story as mod

    if ensure_installed:
        specs = getattr(mod, "CH6_EXPORT_SPECS", None)
        if not specs:
            _reinstall_into_context(mod)
            for name in _LEGACY_DEPS:
                sys.modules.pop(name, None)
            mod = importlib.import_module("ch6_story")
            _reinstall_into_context(mod)
    return mod


def _export_specs():
    return list(_ch6_story(ensure_installed=False).CH6_EXPORT_SPECS)


def _resolve_legacy_filename(filename: str) -> tuple[str, str]:
    clip_key = filename.replace(".mp4", "")
    mod = _ch6_story()
    specs = list(mod.CH6_EXPORT_SPECS)
    for cid, fn, *_ in specs:
        if fn == filename or cid == clip_key or fn.replace(".mp4", "") == clip_key:
            return cid, fn
    for name in _LEGACY_DEPS:
        sys.modules.pop(name, None)
    mod = importlib.import_module("ch6_story")
    _reinstall_into_context(mod)
    specs = list(mod.CH6_EXPORT_SPECS)
    for cid, fn, *_ in specs:
        if fn == filename or cid == clip_key or fn.replace(".mp4", "") == clip_key:
            return cid, fn
    known = ", ".join(s[0] for s in specs[-5:])
    raise KeyError(
        f"unknown export: {filename!r} "
        f"(registry has {len(specs)} clips; last are {known})"
    )


def _legacy_export_cached(filename: str) -> Path:
    clip_id, filename = _resolve_legacy_filename(filename)
    series = SeriesConfig.load()
    profile = active_profile()
    key = clip_result_key(
        series_version=series.version,
        profile=profile.name,
        clip_id=clip_id,
        code=code_hash(("ch6_story", "ch6_frequentist", "ch5_datasets")),
    )
    from hita.config.export import output_dir
    from hita.export.context import project_root

    root = project_root()
    cached = get_clip_mp4(6, key, root)
    out = output_dir(root) / filename
    if cached is not None:
        print(f"  legacy clip cache: hit {clip_id}", flush=True)
        if cached.resolve() != out.resolve():
            shutil.copy2(cached, out)
        return out

    path = _ch6_story().ch6_export_clip(filename)
    put_clip_mp4(6, key, path, root)
    return path


def export_clip(filename: str, **kwargs) -> Path:
    del kwargs  # workers/parallel reserved for future FrameSpec path
    return _legacy_export_cached(filename)


def install(globals_dict: dict[str, Any]) -> None:
    mod = _ch6_story(ensure_installed=False)
    mod.install(globals_dict)
    globals_dict["CH6_EXPORT_SPECS"] = list(mod.CH6_EXPORT_SPECS)
    globals_dict["export_clip"] = export_clip


def __getattr__(name: str):
    if name == "CH6_EXPORT_SPECS":
        return _export_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CH6_EXPORT_SPECS", "export_clip", "export_clip_safe", "install"]
