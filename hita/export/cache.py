"""Content-addressed disk cache for packs and frames.

Layout::

    renders/cache/ch{N}/
      packs/{pack_key}.pkl
      frames/{frame_key}.png
      clips/{clip_key}.mp4
      context/{artifact_key}.pkl
      meta.json

Env:
  HITA_CACHE=0          disable durable cache
  HITA_CACHE_REBUILD=1  ignore hits, rewrite entries
  HITA_CACHE_DIR        override cache root (default: <root>/renders/cache)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_CACHE_ENV = "HITA_CACHE"
_REBUILD_ENV = "HITA_CACHE_REBUILD"
_CACHE_DIR_ENV = "HITA_CACHE_DIR"

# Modules whose source contributes to code_hash invalidation
_CODE_HASH_MODULES = (
    "hita.export.renderers",
    "hita.export.pipeline",
    "hita.export.spec",
    "hita.stories.builders.ch5_47",
    "hita.stories.builders.ch5_54",
    "ch5_prior_landscape",
)


def cache_enabled() -> bool:
    raw = os.environ.get(_CACHE_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def cache_rebuild() -> bool:
    raw = os.environ.get(_REBUILD_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "y"}


def cache_root(root: Path | str | None = None) -> Path:
    raw = os.environ.get(_CACHE_DIR_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        from hita.export.context import project_root

        base = Path(root) if root is not None else project_root()
        path = (Path(base) / "renders" / "cache").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def chapter_cache_dir(chapter: int, root: Path | str | None = None) -> Path:
    d = cache_root(root) / f"ch{int(chapter)}"
    for sub in ("packs", "frames", "clips", "context"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _blake(*parts: Any, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


def code_hash(extra_modules: tuple[str, ...] = ()) -> str:
    """Fingerprint renderer/pack source for cache invalidation."""
    chunks: list[str] = []
    for mod_name in (*_CODE_HASH_MODULES, *extra_modules):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            src = getattr(mod, "__file__", None)
            if src and Path(src).is_file():
                chunks.append(f"{mod_name}:{Path(src).stat().st_mtime_ns}")
                # Include a short content digest for safety
                data = Path(src).read_bytes()
                chunks.append(hashlib.blake2b(data, digest_size=8).hexdigest())
        except Exception:
            chunks.append(f"{mod_name}:missing")
    return _blake(*chunks)


def pack_cache_key(
    *,
    series_version: str,
    profile: str,
    clip_id: str,
    pack_fingerprint_payload: Any,
    code: str | None = None,
) -> str:
    return _blake(
        "pack",
        series_version,
        profile,
        clip_id,
        pack_fingerprint_payload,
        code or code_hash(),
    )


def frame_cache_key(
    *,
    series_version: str,
    profile: str,
    export_dpi: int | float,
    render_fn_name: str,
    pack_fingerprint: str,
    content_key: str,
    code: str | None = None,
) -> str:
    return _blake(
        "frame",
        series_version,
        profile,
        float(export_dpi),
        render_fn_name,
        pack_fingerprint,
        content_key,
        code or code_hash(),
    )


def pack_fingerprint(pack: dict[str, Any]) -> str:
    """Stable hash of array shapes/dtypes + light metadata (not full pack bytes)."""
    parts: list[Any] = []
    for key in sorted(pack.keys()):
        if key.startswith("_hita_"):
            continue
        val = pack[key]
        if isinstance(val, np.ndarray):
            parts.append((key, "ndarray", val.dtype.str, val.shape, hashlib.blake2b(val.tobytes(), digest_size=8).hexdigest()))
        elif isinstance(val, dict):
            # nested mesh packs
            sub = []
            for sk in sorted(val.keys()):
                sv = val[sk]
                if isinstance(sv, np.ndarray):
                    sub.append((sk, sv.dtype.str, sv.shape, hashlib.blake2b(sv.tobytes(), digest_size=8).hexdigest()))
                elif isinstance(sv, dict):
                    # one more level for meshes[name][W1m] etc.
                    inner = []
                    for ik, iv in sorted(sv.items()):
                        if isinstance(iv, np.ndarray):
                            inner.append((ik, iv.dtype.str, iv.shape, hashlib.blake2b(iv.tobytes(), digest_size=8).hexdigest()))
                        else:
                            inner.append((ik, type(iv).__name__, repr(iv)[:120]))
                    sub.append((sk, "dict", tuple(inner)))
                else:
                    sub.append((sk, type(sv).__name__, repr(sv)[:120]))
            parts.append((key, "dict", tuple(sub)))
        else:
            parts.append((key, type(val).__name__, repr(val)[:200]))
    return _blake(*parts)


def get_frame(chapter: int, key: str, root: Path | str | None = None) -> Image.Image | None:
    if not cache_enabled() or cache_rebuild():
        return None
    path = chapter_cache_dir(chapter, root) / "frames" / f"{key}.png"
    if not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except OSError:
        return None


def put_frame(chapter: int, key: str, image: Image.Image, root: Path | str | None = None) -> Path | None:
    if not cache_enabled():
        return None
    path = chapter_cache_dir(chapter, root) / "frames" / f"{key}.png"
    tmp = path.with_suffix(".tmp.png")
    try:
        image.save(tmp, format="PNG")
        tmp.replace(path)
        _touch_meta(chapter, root, frames=1)
        return path
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


def get_pack(chapter: int, key: str, root: Path | str | None = None) -> dict[str, Any] | None:
    if not cache_enabled() or cache_rebuild():
        return None
    import pickle

    path = chapter_cache_dir(chapter, root) / "packs" / f"{key}.pkl"
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            pack = pickle.load(f)
        if not isinstance(pack, dict):
            return None
        return pack
    except Exception:
        return None


def put_pack(chapter: int, key: str, pack: dict[str, Any], root: Path | str | None = None) -> Path | None:
    if not cache_enabled():
        return None
    import pickle

    path = chapter_cache_dir(chapter, root) / "packs" / f"{key}.pkl"
    tmp = path.with_suffix(".tmp.pkl")
    try:
        # Drop ephemeral hita keys from durable store
        store = {k: v for k, v in pack.items() if not str(k).startswith("_hita_")}
        with open(tmp, "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        _touch_meta(chapter, root, packs=1)
        return path
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


def _touch_meta(chapter: int, root: Path | str | None, *, frames: int = 0, packs: int = 0) -> None:
    path = chapter_cache_dir(chapter, root) / "meta.json"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    from hita.config.series import SeriesConfig

    data["series_version"] = SeriesConfig.load().version
    data["updated_at"] = time.time()
    data["frame_writes"] = int(data.get("frame_writes", 0)) + frames
    data["pack_writes"] = int(data.get("pack_writes", 0)) + packs
    path.write_text(json.dumps(data, indent=2))


def clip_result_key(
    *,
    series_version: str,
    profile: str,
    clip_id: str,
    code: str | None = None,
) -> str:
    """Key for legacy serial clip MP4 / frame-list cache."""
    return _blake("clip", series_version, profile, clip_id, code or code_hash())


def get_clip_mp4(chapter: int, key: str, root: Path | str | None = None) -> Path | None:
    if not cache_enabled() or cache_rebuild():
        return None
    path = chapter_cache_dir(chapter, root) / "clips" / f"{key}.mp4"
    return path if path.is_file() and path.stat().st_size > 64 else None


def put_clip_mp4(chapter: int, key: str, src: Path, root: Path | str | None = None) -> Path | None:
    if not cache_enabled():
        return None
    import shutil

    d = chapter_cache_dir(chapter, root) / "clips"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.mp4"
    tmp = path.with_suffix(".tmp.mp4")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(path)
        _touch_meta(chapter, root)
        return path
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


def get_chapter_artifacts(chapter: int, key: str, root: Path | str | None = None) -> dict[str, Any] | None:
    if not cache_enabled() or cache_rebuild():
        return None
    import pickle

    path = chapter_cache_dir(chapter, root) / "context" / f"{key}.pkl"
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def put_chapter_artifacts(
    chapter: int, key: str, data: dict[str, Any], root: Path | str | None = None
) -> Path | None:
    if not cache_enabled():
        return None
    import pickle

    d = chapter_cache_dir(chapter, root) / "context"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.pkl"
    tmp = path.with_suffix(".tmp.pkl")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        _touch_meta(chapter, root)
        return path
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


def clear_chapter_cache(chapter: int, root: Path | str | None = None) -> int:
    """Delete all pack/frame/clip/context files for a chapter. Returns file count removed."""
    d = chapter_cache_dir(chapter, root)
    n = 0
    for sub in ("packs", "frames", "clips", "context"):
        p = d / sub
        if p.is_dir():
            for f in p.iterdir():
                f.unlink(missing_ok=True)
                n += 1
    meta = d / "meta.json"
    if meta.is_file():
        meta.unlink()
        n += 1
    return n


def cache_size_bytes(chapter: int | None = None, root: Path | str | None = None) -> int:
    base = cache_root(root)
    total = 0
    paths = [base / f"ch{chapter}"] if chapter is not None else list(base.glob("ch*"))
    for p in paths:
        if not p.is_dir():
            continue
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def gc_cache(
    *,
    chapter: int | None = None,
    keep_series: str | None = None,
    root: Path | str | None = None,
) -> dict[str, int]:
    """Remove cache files. If keep_series is set, only wipe chapters whose meta mismatches."""
    removed = 0
    bytes_freed = 0
    base = cache_root(root)
    chapters = [chapter] if chapter is not None else [
        int(p.name[2:]) for p in base.glob("ch*") if p.name[2:].isdigit()
    ]
    for ch in chapters:
        d = chapter_cache_dir(ch, root)
        meta_path = d / "meta.json"
        if keep_series and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("series_version") == keep_series:
                    continue
            except json.JSONDecodeError:
                pass
        for sub in ("packs", "frames", "clips", "context"):
            p = d / sub
            if not p.is_dir():
                continue
            for f in list(p.iterdir()):
                if f.is_file():
                    bytes_freed += f.stat().st_size
                    f.unlink(missing_ok=True)
                    removed += 1
        if meta_path.is_file():
            meta_path.unlink(missing_ok=True)
            removed += 1
    return {"removed": removed, "bytes_freed": bytes_freed}


__all__ = [
    "cache_enabled",
    "cache_rebuild",
    "cache_root",
    "chapter_cache_dir",
    "code_hash",
    "pack_cache_key",
    "frame_cache_key",
    "clip_result_key",
    "pack_fingerprint",
    "get_frame",
    "put_frame",
    "get_pack",
    "put_pack",
    "get_clip_mp4",
    "put_clip_mp4",
    "get_chapter_artifacts",
    "put_chapter_artifacts",
    "clear_chapter_cache",
    "cache_size_bytes",
    "gc_cache",
]
