"""Two-phase export: plan specs → parallel render → encode."""
from __future__ import annotations

import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from hita.config.export import output_dir
from hita.config.profile import active_profile, default_workers, parallel_enabled
from hita.config.series import SeriesConfig
from hita.export.cache import (
    cache_enabled,
    code_hash,
    frame_cache_key,
    get_frame,
    get_pack,
    pack_cache_key,
    pack_fingerprint,
    put_frame,
    put_pack,
)
from hita.export.context import load_export_context, worker_init
from hita.export.spec import FrameSpec

# Shared pack for worker processes (set by initializer; avoids per-task pickle).
_WORKER_PACK: dict[str, Any] | None = None
_WORKER_PACK_ID: str | None = None
_WORKER_ROOT: str | None = None
_WORKER_CHAPTER: int | None = None
_WORKER_RENDER_FN: str | None = None
_WORKER_SERIES: str | None = None
_WORKER_PROFILE: str | None = None
_WORKER_DPI: float | None = None
_WORKER_CODE: str | None = None
_WORKER_PACK_FP: str | None = None


def dedupe_consecutive_by_key(keys: list[str], frames: list[Image.Image]) -> list[Image.Image]:
    """Drop consecutive frames that share the same content key (deterministic)."""
    if len(frames) < 2:
        return frames
    out = [frames[0]]
    last_key = keys[0]
    for key, fr in zip(keys[1:], frames[1:]):
        if key != last_key:
            out.append(fr)
            last_key = key
    if len(out) != len(frames):
        print(f"  deduped consecutive specs: {len(frames)} → {len(out)}", flush=True)
    return out


def _worker_init_with_pack(
    root_str: str,
    chapter: int,
    pack: dict[str, Any],
    pack_id: str,
    render_fn_name: str,
    series_version: str,
    profile_name: str,
    export_dpi: float,
    code: str,
    pack_fp: str,
) -> None:
    global _WORKER_PACK, _WORKER_PACK_ID, _WORKER_ROOT, _WORKER_CHAPTER
    global _WORKER_RENDER_FN, _WORKER_SERIES, _WORKER_PROFILE, _WORKER_DPI
    global _WORKER_CODE, _WORKER_PACK_FP
    worker_init(root_str, chapter)
    _WORKER_PACK = pack
    _WORKER_PACK_ID = pack_id
    _WORKER_ROOT = root_str
    _WORKER_CHAPTER = chapter
    _WORKER_RENDER_FN = render_fn_name
    _WORKER_SERIES = series_version
    _WORKER_PROFILE = profile_name
    _WORKER_DPI = export_dpi
    _WORKER_CODE = code
    _WORKER_PACK_FP = pack_fp


def _render_one_task(args: tuple) -> tuple[int, Image.Image, bool]:
    """Return (idx, frame, cache_hit)."""
    idx, spec_dict, content_key = args
    assert _WORKER_PACK is not None
    assert _WORKER_CHAPTER is not None
    assert _WORKER_ROOT is not None

    fkey = frame_cache_key(
        series_version=_WORKER_SERIES or "",
        profile=_WORKER_PROFILE or "",
        export_dpi=_WORKER_DPI or 200,
        render_fn_name=_WORKER_RENDER_FN or "",
        pack_fingerprint=_WORKER_PACK_FP or "",
        content_key=content_key,
        code=_WORKER_CODE,
    )
    cached = get_frame(_WORKER_CHAPTER, fkey, _WORKER_ROOT)
    if cached is not None:
        return idx, cached, True

    render_fn = _resolve_render_fn(_WORKER_RENDER_FN or "")
    frame = render_fn(_WORKER_PACK, spec_dict)
    put_frame(_WORKER_CHAPTER, fkey, frame, _WORKER_ROOT)
    return idx, frame, False


def _resolve_render_fn(name: str) -> Callable:
    if name == "hita.export.renderers.render_ch5_frame":
        from hita.export.renderers import render_ch5_frame

        return render_ch5_frame
    if name == "hita.export.renderers.render_sigmoid_frame":
        from hita.export.renderers import render_sigmoid_frame

        return render_sigmoid_frame
    if name == "hita.export.renderers.render_ch7_demo_frame":
        from hita.export.renderers import render_ch7_demo_frame

        return render_ch7_demo_frame
    raise KeyError(f"unknown render function: {name!r}")


def render_unique_specs(
    unique_specs: list[FrameSpec],
    pack: dict[str, Any],
    *,
    render_fn_name: str,
    workers: int,
    root_path: str,
    chapter: int,
    progress_label: str,
) -> dict[str, Image.Image]:
    """Render each unique content key once; return ``{content_key: Image}``."""
    n = len(unique_specs)
    if n == 0:
        return {}
    t0 = time.perf_counter()
    render_fn = _resolve_render_fn(render_fn_name)
    cache: dict[str, Image.Image] = {}
    profile = active_profile()
    series = SeriesConfig.load()
    code = code_hash()
    pack_fp = pack_fingerprint(pack)
    hits = 0

    def _frame_key(spec: FrameSpec) -> str:
        return frame_cache_key(
            series_version=series.version,
            profile=profile.name,
            export_dpi=profile.export_dpi,
            render_fn_name=render_fn_name,
            pack_fingerprint=pack_fp,
            content_key=spec.content_key(),
            code=code,
        )

    if workers <= 1:
        load_export_context(root_path, chapter=chapter)
        step = max(n // 10, 1)
        for i, spec in enumerate(unique_specs):
            ck = spec.content_key()
            fkey = _frame_key(spec)
            frame = get_frame(chapter, fkey, root_path)
            if frame is not None:
                hits += 1
            else:
                frame = render_fn(pack, spec.to_dict())
                put_frame(chapter, fkey, frame, root_path)
            cache[ck] = frame
            if i == 0 or (i + 1) % step == 0 or i + 1 == n:
                print(
                    f"  {progress_label}: {i + 1}/{n} unique "
                    f"({time.perf_counter() - t0:.0f}s, disk_hits={hits})",
                    flush=True,
                )
        if hits:
            print(f"  disk frame cache: {hits}/{n} hits", flush=True)
        return cache

    pack_id = pack_fp[:16]
    results: list[tuple[int, Image.Image, bool]] = []
    tasks = [
        (i, unique_specs[i].to_dict(), unique_specs[i].content_key())
        for i in range(n)
    ]
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init_with_pack,
            initargs=(
                root_path,
                chapter,
                pack,
                pack_id,
                render_fn_name,
                series.version,
                profile.name,
                float(profile.export_dpi),
                code,
                pack_fp,
            ),
        ) as pool:
            futures = [pool.submit(_render_one_task, t) for t in tasks]
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % max(n // 10, 1) == 0 or done == n:
                    print(
                        f"  {progress_label}: {done}/{n} unique "
                        f"({time.perf_counter() - t0:.0f}s, {workers} workers)",
                        flush=True,
                    )
    except (BrokenExecutor, PermissionError, OSError, TypeError) as exc:
        print(f"  parallel render failed ({exc}); retrying serially", flush=True)
        return render_unique_specs(
            unique_specs, pack,
            render_fn_name=render_fn_name,
            workers=1,
            root_path=root_path,
            chapter=chapter,
            progress_label=progress_label,
        )

    for i, frame, hit in sorted(results, key=lambda x: x[0]):
        cache[unique_specs[i].content_key()] = frame
        if hit:
            hits += 1
    if hits:
        print(f"  disk frame cache: {hits}/{n} hits", flush=True)
    return cache


def render_specs(
    specs: list[FrameSpec],
    pack: dict[str, Any],
    *,
    render_fn_name: str = "hita.export.renderers.render_ch5_frame",
    workers: int | None = None,
    root: Path | str | None = None,
    chapter: int = 5,
    progress_label: str = "frames",
) -> list[Image.Image]:
    """Phase B: render frame specs (unique-by-content, then assemble in order)."""
    from hita.export.context import project_root

    n = len(specs)
    if n == 0:
        return []

    root_path = str(Path(root or project_root()).resolve())
    use_workers = workers if workers is not None else default_workers()
    if not parallel_enabled() or use_workers <= 1:
        use_workers = 1

    key_to_spec: dict[str, FrameSpec] = {}
    for spec in specs:
        key_to_spec.setdefault(spec.content_key(), spec)
    unique_specs = list(key_to_spec.values())
    if len(unique_specs) < n:
        print(
            f"  unique specs: {len(unique_specs)}/{n} "
            f"({100.0 * (1.0 - len(unique_specs) / n):.0f}% cache)",
            flush=True,
        )

    cache = render_unique_specs(
        unique_specs,
        pack,
        render_fn_name=render_fn_name,
        workers=use_workers,
        root_path=root_path,
        chapter=chapter,
        progress_label=progress_label,
    )
    return [cache[spec.content_key()] for spec in specs]


def assemble_frames(
    specs: list[FrameSpec],
    rendered: list[Image.Image],
    *,
    dedupe: bool = True,
) -> list[Image.Image]:
    if not dedupe:
        return rendered
    keys = [s.content_key() for s in specs]
    if len(rendered) == len(specs):
        return dedupe_consecutive_by_key(keys, rendered)
    return rendered


def _build_pack_cached(
    spec_entry: Any,
    ctx: dict[str, Any],
    *,
    root_path: Path,
    series_version: str,
) -> dict[str, Any]:
    profile = active_profile()
    payload = (spec_entry.clip_id, profile.name, series_version, code_hash())
    pkey = pack_cache_key(
        series_version=series_version,
        profile=profile.name,
        clip_id=spec_entry.clip_id,
        pack_fingerprint_payload=payload,
    )
    cached = get_pack(int(spec_entry.chapter), pkey, root_path)
    if cached is not None:
        print("  pack disk cache: hit", flush=True)
        return cached
    pack = dict(spec_entry.build_pack(spec_entry.clip_id, ctx))
    fp = pack_fingerprint(pack)
    pkey2 = pack_cache_key(
        series_version=series_version,
        profile=profile.name,
        clip_id=spec_entry.clip_id,
        pack_fingerprint_payload=(spec_entry.clip_id, profile.name, fp),
    )
    put_pack(int(spec_entry.chapter), pkey2, pack, root_path)
    if pkey2 != pkey:
        put_pack(int(spec_entry.chapter), pkey, pack, root_path)
    return pack


def export_clip(
    clip_id: str,
    *,
    workers: int | None = None,
    root: Path | str | None = None,
    chapter: int | None = None,
    dedupe: bool | None = None,
) -> Path:
    """Full pipeline: plan → render unique specs → assemble → encode."""
    from hita.stories.registry import get_export_spec

    spec_entry = get_export_spec(clip_id, chapter=chapter or 0)
    chapter = int(spec_entry.chapter)
    root_path = Path(root or spec_entry.root).resolve()
    series = SeriesConfig.load()
    ctx = load_export_context(root_path, chapter=chapter)

    use_dedupe = spec_entry.dedupe_consecutive if dedupe is None else dedupe

    print(
        f"hita export {spec_entry.clip_id} profile={active_profile().name} "
        f"series={series.version} cache={'on' if cache_enabled() else 'off'}",
        flush=True,
    )
    t0 = time.perf_counter()

    pack = _build_pack_cached(spec_entry, ctx, root_path=root_path, series_version=series.version)
    pack = dict(pack)
    pack["_hita_series_version"] = series.version
    pack["_hita_clip_id"] = spec_entry.clip_id

    plan_ctx = dict(ctx)
    plan_ctx["_hita_pack"] = pack
    frame_specs = list(spec_entry.builder_plan(spec_entry.clip_id, plan_ctx))
    if spec_entry.chapter >= 7 and not frame_specs:
        raise ValueError(f"{spec_entry.clip_id}: builder_plan returned no FrameSpecs")
    for i, fs in enumerate(frame_specs):
        if not isinstance(fs, FrameSpec):
            raise TypeError(
                f"{spec_entry.clip_id}: builder_plan[{i}] must be FrameSpec, got {type(fs)!r}"
            )

    n_specs = len(frame_specs)
    print(f"  planned {n_specs} frame specs ({time.perf_counter() - t0:.0f}s)", flush=True)

    rendered = render_specs(
        frame_specs,
        pack,
        render_fn_name=spec_entry.render_fn,
        workers=workers,
        root=root_path,
        chapter=chapter,
        progress_label=spec_entry.clip_id,
    )
    frames = assemble_frames(frame_specs, rendered, dedupe=use_dedupe)
    # Macro-block padding happens once inside ctx["save_mp4"].

    ms = spec_entry.ms_per_frame or int(ctx.get("CH5_HQ_LAND_MS", 90))
    out_dir = output_dir(root_path)
    ctx["OUTPUT_DIR"] = out_dir
    ctx["save_mp4"](frames, spec_entry.filename, duration=ms)
    out = out_dir / spec_entry.filename
    print(f"wrote {out} ({len(frames)} frames, {time.perf_counter() - t0:.0f}s total)", flush=True)
    return out
