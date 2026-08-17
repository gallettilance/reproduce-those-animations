"""Modular fast export for ch4 tutorial animations (3D + template compose).

Speedups:
- Pre-render static rail layers once per formula variant (corner + bottom TeX).
- Use cached-rails compose path for every frame (never full re-draw of static rails).
- Optional parallel frame rendering across worker processes.
- Skip duplicate consecutive frames when specs produce identical pixels.
- Batch export: render each unique spec once across multiple MP4 jobs (shared frame cache).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from pathlib import Path
from typing import Any, Callable

from PIL import Image

import numpy as np

_ROOT: Path | None = None
_CTX: dict | None = None

CH4_EXPORT_WORKERS_ENV = "CH4_EXPORT_WORKERS"
_CH3_SETUP_LOCK_NAME = ".ch3_setup.lock"


def _outcome_icons_valid(root: Path) -> bool:
    """True when check/cross PNGs exist and PIL can read them."""
    for name in ("check.png", "cross.png"):
        path = root / "renders" / name
        if not path.is_file() or path.stat().st_size < 16:
            return False
        try:
            with Image.open(path) as im:
                im.load()
        except OSError:
            return False
    return True


_FIG_TO_IMAGE_OLD = (
    'def fig_to_image(fig, dpi=None, tight_layout=False, transparent=False):\n'
    '    if dpi is None:\n'
    '        dpi = EXPORT_DPI\n'
    '    buf = io.BytesIO()\n'
    '    kw = {"format": "png", "dpi": dpi, "pad_inches": SAVE_PAD_INCHES}\n'
    '    kw["bbox_inches"] = "tight" if tight_layout else None\n'
    '    kw["transparent"] = bool(transparent)\n'
    '    fig.savefig(buf, **kw)\n'
    '    plt.close(fig)\n'
    '    buf.seek(0)\n'
    '    return Image.open(buf).convert("RGBA" if transparent else "RGB")'
)
_FIG_TO_IMAGE_NEW = (
    'def fig_to_image(fig, dpi=None, tight_layout=False, transparent=False):\n'
    '    """Rasterize via Agg canvas (faster than PNG encode/decode through BytesIO)."""\n'
    '    if dpi is None:\n'
    '        dpi = EXPORT_DPI\n'
    '    if tight_layout:\n'
    '        fig.tight_layout()\n'
    '    fig.set_dpi(float(dpi))\n'
    '    fig.canvas.draw()\n'
    '    w, h = fig.canvas.get_width_height()\n'
    '    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))\n'
    '    plt.close(fig)\n'
    '    if transparent:\n'
    '        return Image.fromarray(rgba, mode="RGBA")\n'
    '    return Image.fromarray(rgba[..., :3], mode="RGB")'
)


def _patch_ch3_setup_src(ch3_src: str, root: Path) -> str:
    """Avoid parallel workers racing on ``renders/check.png`` and ``cross.png``."""
    if _FIG_TO_IMAGE_OLD in ch3_src:
        ch3_src = ch3_src.replace(_FIG_TO_IMAGE_OLD, _FIG_TO_IMAGE_NEW, 1)
    if _outcome_icons_valid(root):
        ch3_src = ch3_src.replace(
            "_ensure_outcome_icons()\n",
            "pass  # outcome icons already on disk\n",
            1,
        )
    return ch3_src


def _spawn_unsafe() -> bool:
    """``ProcessPoolExecutor`` spawn re-imports ``__main__`` — fails from Jupyter/stdin."""
    import __main__

    mf = getattr(__main__, "__file__", None)
    if not mf:
        return True
    p = Path(str(mf))
    if not p.is_file():
        return True
    if p.suffix in {".ipy", ".ipynb", ""} or "ipykernel" in str(p):
        return True
    return False


def load_export_context(root: Path | str | None = None) -> dict:
    """Load ch3 + ch4 builder globals (once per process)."""
    global _ROOT, _CTX
    if _CTX is not None:
        return _CTX

    import fcntl
    import json

    import matplotlib

    matplotlib.use("Agg")

    root = Path(root or Path.cwd()).resolve()
    _ROOT = root
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    lock_path = root / "renders" / _CH3_SETUP_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            if _CTX is not None:
                return _CTX

            g: dict[str, Any] = {"__name__": "__export_worker__"}
            exec("from ch4_layout import *", g)
            ch3_src = "".join(
                json.loads((root / "logistic-regression-chap3.ipynb").read_text())["cells"][1]["source"]
            )
            ch3_src = _patch_ch3_setup_src(ch3_src, root)
            exec(compile(ch3_src, "ch3", "exec"), g)
            ch4_src = (root / "blender/ch4_02_likelihood_w12_landscape.inc").read_text()
            ch4_src += "\n\n" + (root / "blender/ch4_03_04_likelihood_story.inc").read_text()
            ch4_src += "\n\n" + (root / "blender/ch4_05b_partial_derivative.inc").read_text()
            exec(compile(ch4_src, "ch4", "exec"), g)
            _CTX = g
            return g
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _worker_init(root_str: str) -> None:
    load_export_context(root_str)


def default_workers() -> int:
    raw = os.environ.get(CH4_EXPORT_WORKERS_ENV, "").strip()
    if raw.isdigit():
        return max(int(raw), 1)
    if _spawn_unsafe():
        return 1
    n = os.cpu_count() or 4
    return max(min(n - 1, 8), 1)


def collect_gd_rails_variants(specs: list[dict]) -> list[tuple]:
    seen = set()
    out = []
    for spec in specs:
        key = (
            spec.get("bold"),
            bool(spec.get("bold_all", False)),
            bool(spec.get("grad_red", False)),
        )
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def prewarm_gd_rails(composer, variants: list[tuple], *, g: dict | None = None) -> None:
    """Render each static bottom-rail variant once (corner + formulas, no plot)."""
    g = g or load_export_context()
    from ch4_layout import (
        CH4_FORMULAS_SECTION_TITLE,
        CH4_HERE_SECTION_TITLE,
        CH4_NOTATION_SECTION_TITLE,
        CH4_COMPOSER,
        ch4_cached_notation_corner_blocks,
        ch4_formula_blocks_gd_story,
        ch4_rails_cache_key_gd,
    )
    from tutorial_template import TutorialScene

    comp = composer or CH4_COMPOSER
    corner = ch4_cached_notation_corner_blocks()
    for bold, bold_all, grad_red in variants:
        bottom = ch4_formula_blocks_gd_story(
            highlight_update_idx=bold,
            highlight_all_updates=bold_all,
            grad_red=grad_red,
        )
        cache_key = ch4_rails_cache_key_gd(
            highlight_update_idx=bold,
            highlight_all=bold_all,
            grad_red=grad_red,
        )
        scene = TutorialScene(
            plot=None,
            math_bottom_blocks=bottom,
            math_corner_blocks=corner,
            right_section_title=CH4_HERE_SECTION_TITLE,
            bottom_section_title=CH4_FORMULAS_SECTION_TITLE,
            corner_section_title=CH4_NOTATION_SECTION_TITLE,
        )
        comp._render_static_rails(scene, panel_u=1.0, title_u=1.0, cache_key=cache_key)


def prewarm_shell_rails(composer, *, g: dict | None = None) -> None:
    """Pre-render shell rails (backgrounds + titles + corner) for partial-motivation export."""
    g = g or load_export_context()
    from ch4_layout import (
        CH4_FORMULAS_SECTION_TITLE,
        CH4_HERE_SECTION_TITLE,
        CH4_NOTATION_SECTION_TITLE,
        CH4_COMPOSER,
        CH4_RAILS_CACHE_SHELL,
        ch4_cached_notation_corner_blocks,
        ch4_formula_blocks_3d_story,
    )
    from tutorial_template import TutorialScene

    comp = composer or CH4_COMPOSER
    scene = TutorialScene(
        plot=None,
        math_bottom_blocks=ch4_formula_blocks_3d_story(),
        math_corner_blocks=ch4_cached_notation_corner_blocks(),
        right_section_title=CH4_HERE_SECTION_TITLE,
        bottom_section_title=CH4_FORMULAS_SECTION_TITLE,
        corner_section_title=CH4_NOTATION_SECTION_TITLE,
    )
    comp._render_static_shell_rails(scene, panel_u=1.0, title_u=1.0, cache_key=CH4_RAILS_CACHE_SHELL)


def prewarm_partial_motivation_rails(composer, *, g: dict | None = None) -> None:
    """Prewarm shell + static 3D + default GD bottom rails for ch4_05b."""
    g = g or load_export_context()
    prewarm_shell_rails(composer, g=g)
    prewarm_3d_rails(composer, g=g)
    prewarm_gd_rails(composer, [(None, False, False)], g=g)


def _render_frames_serial(
    specs: list[dict],
    render_frame: Callable,
    pack: dict,
    *,
    progress_label: str,
    t0: float,
) -> list[Image.Image]:
    n = len(specs)
    frames: list[Image.Image] = []
    step = max(n // 10, 1)
    for i, spec in enumerate(specs):
        frames.append(render_frame(pack, spec, cam_azim_u=float(i) / float(max(n - 1, 1))))
        if i == 0 or (i + 1) % step == 0 or i + 1 == n:
            dt = time.perf_counter() - t0
            print(f"  {progress_label}: {i + 1}/{n}  ({dt:.0f}s)", flush=True)
    return frames


def _render_frames_parallel(
    specs: list[dict],
    pack: dict,
    *,
    workers: int,
    root_path: str,
    render_fn_name: str,
    progress_label: str,
    t0: float,
) -> list[Image.Image]:
    n = len(specs)
    frames: list[Image.Image | None] = [None] * n
    tasks = [
        (i, spec, pack, float(i) / float(max(n - 1, 1)), root_path, render_fn_name)
        for i, spec in enumerate(specs)
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(root_path,)) as pool:
        futures = [pool.submit(_render_one_task, t) for t in tasks]
        for fut in as_completed(futures):
            idx, frame = fut.result()
            frames[idx] = frame
            done += 1
            if done % max(n // 10, 1) == 0 or done == n:
                dt = time.perf_counter() - t0
                print(f"  {progress_label}: {done}/{n}  ({dt:.0f}s, {workers} workers)", flush=True)
    return [f for f in frames if f is not None]


def _render_one_task(args: tuple) -> tuple[int, Any]:
    idx, spec, pack, cam_u, root_str, render_fn_name = args
    g = load_export_context(root_str)
    render_fn = g[render_fn_name]
    frame = render_fn(pack, spec, cam_azim_u=float(cam_u))
    return idx, frame


def _norm_spec_for_cache(value: Any) -> Any:
    """Convert a spec value to a hashable, stable representation."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return ("ndarray", arr.dtype.str, tuple(int(x) for x in arr.shape), arr.tobytes())
    if isinstance(value, dict):
        if "xs" in value and "colors" in value:
            vd = value
            return (
                "voxel_draw",
                _norm_spec_for_cache(vd.get("xs")),
                _norm_spec_for_cache(vd.get("ys")),
                _norm_spec_for_cache(vd.get("zs")),
                float(vd.get("dx", 0.0)),
                float(vd.get("dy", 0.0)),
                float(vd.get("dz", 0.0)),
                _norm_spec_for_cache(vd.get("colors")),
            )
        return tuple(sorted((str(k), _norm_spec_for_cache(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_norm_spec_for_cache(v) for v in value)
    return repr(value)


def spec_render_key(spec: dict) -> str:
    """Stable hash key for a frame spec (identical specs → identical pixels)."""
    payload = tuple(sorted((str(k), _norm_spec_for_cache(v)) for k, v in spec.items()))
    return hashlib.blake2b(repr(payload).encode(), digest_size=16).hexdigest()


def build_frame_cache(
    pack: dict,
    unique_specs: list[dict],
    render_frame: Callable,
    *,
    prewarm: str | None = "gd",
    parallel: bool | None = None,
    progress_label: str = "unique frames",
    root: Path | str | None = None,
    render_fn_name: str = "_ch3_lik_gd_render_frame",
) -> dict[str, Image.Image]:
    """Render each unique spec once; return ``{spec_render_key: Image}``."""
    g = load_export_context(root)
    from ch4_layout import CH4_COMPOSER

    comp = CH4_COMPOSER
    comp.clear_rails_cache()
    prewarm_rails(comp, prewarm, unique_specs, g=g)

    n = len(unique_specs)
    if n == 0:
        return {}

    use_parallel = parallel if parallel is not None else default_workers() > 1
    workers = 1 if not use_parallel else default_workers()
    root_path = str(Path(root or _ROOT or Path.cwd()).resolve())
    t0 = time.perf_counter()
    cache: dict[str, Image.Image] = {}

    if workers <= 1:
        step = max(n // 10, 1)
        for i, spec in enumerate(unique_specs):
            key = spec_render_key(spec)
            cache[key] = render_frame(pack, spec, cam_azim_u=0.0)
            if i == 0 or (i + 1) % step == 0 or i + 1 == n:
                dt = time.perf_counter() - t0
                print(f"  {progress_label}: {i + 1}/{n}  ({dt:.0f}s)", flush=True)
    else:
        try:
            tasks = [
                (i, spec, pack, 0.0, root_path, render_fn_name)
                for i, spec in enumerate(unique_specs)
            ]
            results: list[tuple[int, Image.Image]] = []
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(root_path,)) as pool:
                futures = [pool.submit(_render_one_task, t) for t in tasks]
                done = 0
                for fut in as_completed(futures):
                    results.append(fut.result())
                    done += 1
                    if done % max(n // 10, 1) == 0 or done == n:
                        dt = time.perf_counter() - t0
                        print(
                            f"  {progress_label}: {done}/{n}  ({dt:.0f}s, {workers} workers)",
                            flush=True,
                        )
            for i, frame in sorted(results, key=lambda x: x[0]):
                cache[spec_render_key(unique_specs[i])] = frame
        except (BrokenExecutor, PermissionError, OSError) as exc:
            print(
                f"  parallel render failed ({exc}); retrying {n} unique frames serially",
                flush=True,
            )
            step = max(n // 10, 1)
            for i, spec in enumerate(unique_specs):
                key = spec_render_key(spec)
                cache[key] = render_frame(pack, spec, cam_azim_u=0.0)
                if i == 0 or (i + 1) % step == 0 or i + 1 == n:
                    dt = time.perf_counter() - t0
                    print(f"  {progress_label}: {i + 1}/{n}  ({dt:.0f}s)", flush=True)

    return cache


def assemble_frames_from_specs(
    specs: list[dict],
    frame_cache: dict[str, Image.Image],
    *,
    dedupe_consecutive: bool = True,
) -> list[Image.Image]:
    """Map a spec timeline onto pre-rendered cache entries."""
    out = [frame_cache[spec_render_key(s)] for s in specs]
    if dedupe_consecutive and len(out) >= 2:
        deduped = [out[0]]
        for fr in out[1:]:
            if fr.tobytes() != deduped[-1].tobytes():
                deduped.append(fr)
        if len(deduped) != len(out):
            print(f"  deduped consecutive frames: {len(out)} → {len(deduped)}", flush=True)
        out = deduped
    return out


def export_mp4_batch_from_jobs(
    pack: dict,
    jobs: list[dict],
    render_frame: Callable,
    *,
    save_mp4,
    prewarm: str | None = "gd",
    parallel: bool | None = None,
    dedupe_consecutive: bool = True,
    progress_label: str = "ch4_08 batch",
    root: Path | str | None = None,
    render_fn_name: str = "_ch3_lik_gd_render_frame",
) -> list[Path]:
    """Export multiple MP4s sharing one deduplicated frame render pass.

    Each job dict: ``filename``, ``specs``, optional ``duration_ms``, ``story_hold_fn``,
    ``progress_label``.
    """
    g = load_export_context(root)
    output_dir = Path(g.get("OUTPUT_DIR", Path("renders")))

    all_specs: list[dict] = []
    total_slots = 0
    for job in jobs:
        specs = list(job["specs"])
        total_slots += len(specs)
        all_specs.extend(specs)

    key_to_spec: dict[str, dict] = {}
    for spec in all_specs:
        key_to_spec.setdefault(spec_render_key(spec), spec)
    unique_specs = list(key_to_spec.values())
    n_unique = len(unique_specs)
    print(
        f"{progress_label}: {len(jobs)} clips, {total_slots} frame slots, "
        f"{n_unique} unique specs ({100.0 * (1.0 - n_unique / max(total_slots, 1)):.1f}% cache savings)",
        flush=True,
    )
    if n_unique > 2000 and (parallel is None or parallel is not False):
        print(
            "  tip: run from a script with CH4_EXPORT_WORKERS=8 for parallel unique-frame renders "
            "(Jupyter disables workers by default)",
            flush=True,
        )

    t_spec = time.perf_counter()
    frame_cache = build_frame_cache(
        pack,
        unique_specs,
        render_frame,
        prewarm=prewarm,
        parallel=parallel,
        progress_label=f"{progress_label} render",
        root=root,
        render_fn_name=render_fn_name,
    )
    print(f"  {progress_label} render cache: {time.perf_counter() - t_spec:.0f}s", flush=True)

    paths: list[Path] = []
    for job in jobs:
        label = str(job.get("progress_label", job["filename"]))
        frames = assemble_frames_from_specs(
            list(job["specs"]),
            frame_cache,
            dedupe_consecutive=dedupe_consecutive,
        )
        story_hold_fn = job.get("story_hold_fn")
        if story_hold_fn is not None:
            frames = story_hold_fn(frames)
        filename = str(job["filename"])
        duration_ms = int(job.get("duration_ms", 130))
        save_mp4(frames, filename, duration=int(duration_ms))
        path = output_dir / filename
        print(f"wrote {path}  ({len(frames)} frames) [{label}]", flush=True)
        paths.append(path)
    return paths


def build_tutorial_frames(
    pack: dict,
    specs: list[dict],
    render_frame: Callable,
    *,
    prewarm: str | None = "gd",
    parallel: bool | None = None,
    dedupe_consecutive: bool = True,
    progress_label: str = "frames",
    root: Path | str | None = None,
    render_fn_name: str = "_ch3_lik_gd_render_frame",
) -> list[Image.Image]:
    """Build animation frames from specs with optional prewarm + parallelism."""
    g = load_export_context(root)
    from ch4_layout import CH4_COMPOSER

    comp = CH4_COMPOSER
    comp.clear_rails_cache()

    n = len(specs)
    if n == 0:
        return []

    if prewarm == "gd":
        prewarm_gd_rails(comp, collect_gd_rails_variants(specs), g=g)
    elif prewarm == "3d":
        prewarm_3d_rails(comp, g=g)
    elif prewarm == "partial":
        prewarm_partial_motivation_rails(comp, g=g)

    use_parallel = parallel if parallel is not None else default_workers() > 1
    workers = 1 if not use_parallel else default_workers()
    root_path = str(Path(root or _ROOT or Path.cwd()).resolve())

    t0 = time.perf_counter()
    if workers <= 1 and _spawn_unsafe() and parallel is not False:
        print(
            "  note: parallel export disabled in Jupyter (set CH4_EXPORT_WORKERS>1 from a .py script to override)",
            flush=True,
        )

    if workers <= 1:
        out = _render_frames_serial(
            specs, render_frame, pack, progress_label=progress_label, t0=t0,
        )
    else:
        try:
            out = _render_frames_parallel(
                specs, pack,
                workers=workers,
                root_path=root_path,
                render_fn_name=render_fn_name,
                progress_label=progress_label,
                t0=t0,
            )
        except BrokenExecutor as exc:
            print(
                f"  parallel export failed ({exc}); retrying serially "
                "(common when running from a notebook on macOS)",
                flush=True,
            )
            out = _render_frames_serial(
                specs, render_frame, pack, progress_label=progress_label, t0=t0,
            )

    if dedupe_consecutive and len(out) >= 2:
        deduped = [out[0]]
        for fr in out[1:]:
            if fr.tobytes() != deduped[-1].tobytes():
                deduped.append(fr)
        if len(deduped) != len(out):
            print(f"  deduped consecutive frames: {len(out)} → {len(deduped)}", flush=True)
        out = deduped

    return out


def prewarm_3d_rails(composer, *, g: dict | None = None) -> None:
    """Pre-render static 3D-story bottom + corner rails (ch4_04-style)."""
    g = g or load_export_context()
    from ch4_layout import (
        CH4_FORMULAS_SECTION_TITLE,
        CH4_HERE_SECTION_TITLE,
        CH4_NOTATION_SECTION_TITLE,
        CH4_COMPOSER,
        ch4_cached_formula_blocks_3d_story,
        ch4_cached_notation_corner_blocks,
        ch4_rails_cache_key,
    )
    from tutorial_template import TutorialScene

    comp = composer or CH4_COMPOSER
    cache_key = ch4_rails_cache_key(gd_formulas=False)
    scene = TutorialScene(
        plot=None,
        math_bottom_blocks=ch4_cached_formula_blocks_3d_story(),
        math_corner_blocks=ch4_cached_notation_corner_blocks(),
        right_section_title=CH4_HERE_SECTION_TITLE,
        bottom_section_title=CH4_FORMULAS_SECTION_TITLE,
        corner_section_title=CH4_NOTATION_SECTION_TITLE,
    )
    comp._render_static_rails(scene, panel_u=1.0, title_u=1.0, cache_key=cache_key)


def prewarm_rails(composer, prewarm: str | None, specs: list[dict], *, g: dict | None = None) -> None:
    if prewarm == "gd":
        prewarm_gd_rails(composer, collect_gd_rails_variants(specs), g=g)
    elif prewarm == "3d":
        prewarm_3d_rails(composer, g=g)
    elif prewarm == "partial":
        prewarm_partial_motivation_rails(composer, g=g)


def export_mp4_from_specs(
    pack: dict,
    specs: list[dict],
    render_frame: Callable,
    *,
    save_mp4,
    filename: str,
    duration_ms: int,
    story_hold_fn: Callable | None = None,
    prewarm: str | None = "gd",
    parallel: bool | None = None,
    progress_label: str = "frames",
    root: Path | str | None = None,
    render_fn_name: str = "_ch3_lik_gd_render_frame",
) -> Path:
    """Build frames from specs, optional hold extension, write MP4."""
    g = load_export_context(root)
    output_dir = g.get("OUTPUT_DIR", Path("renders"))

    frames = build_tutorial_frames(
        pack,
        specs,
        render_frame,
        prewarm=prewarm,
        parallel=parallel,
        progress_label=progress_label,
        root=root,
        render_fn_name=render_fn_name,
    )
    if story_hold_fn is not None:
        frames = story_hold_fn(frames)

    save_mp4(frames, filename, duration=int(duration_ms))
    path = Path(output_dir) / filename
    print("wrote", path, f"({len(frames)} frames)", flush=True)
    return path
