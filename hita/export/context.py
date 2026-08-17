"""Load chapter builder globals once per process (macOS-safe spawn)."""
from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from hita.config.profile import apply_profile_env

_ROOT: Path | None = None
_CTX: dict[str, Any] | None = None
_CTX_CHAPTER: int | None = None
_CTX_BY_CHAPTER: dict[int, dict[str, Any]] = {}
_CH3_SETUP_LOCK = ".ch3_setup.lock"
_SLIM_WORKER_ENV = "HITA_SLIM_WORKER"


def project_root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(__file__).resolve().parents[2]
    return _ROOT


def legacy_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "legacy"


def legacy_notebooks_dir() -> Path:
    return legacy_dir() / "notebooks"


def _ensure_legacy_on_path() -> Path:
    leg = legacy_dir()
    leg_s = str(leg)
    if leg_s not in sys.path:
        sys.path.insert(0, leg_s)
    return leg


def _nb_cell_source(nb_path: Path, cell_index: int) -> str:
    return "".join(json.loads(nb_path.read_text())["cells"][cell_index]["source"])


def _patch_ch3_setup_src(ch3_src: str, root: Path) -> str:
    from PIL import Image

    def icons_valid() -> bool:
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

    old = (
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
    new = (
        'def fig_to_image(fig, dpi=None, tight_layout=False, transparent=False):\n'
        '    """Rasterize via Agg canvas (faster than PNG round-trip)."""\n'
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
    if old in ch3_src:
        ch3_src = ch3_src.replace(old, new, 1)
    if icons_valid():
        ch3_src = ch3_src.replace(
            "_ensure_outcome_icons()\n",
            "pass  # outcome icons already on disk\n",
            1,
        )
    return ch3_src


def _patch_ch4_setup_src(ch4_src: str, ch3_nb: Path) -> str:
    old = '_CH3_NB = Path("logistic-regression-chap3.ipynb")'
    new = f'_CH3_NB = Path({str(ch3_nb)!r})'
    if old in ch4_src:
        return ch4_src.replace(old, new, 1)
    return ch4_src


def _base_globals(root: Path) -> dict[str, Any]:
    from hita.config.export import output_dir
    from hita.config.profile import active_profile
    from hita.primitives.icons import ensure_outcome_icons
    from hita.primitives.knobs import ensure_knob_assets
    from hita.primitives.math import sigmoid

    out = output_dir(root)
    ensure_outcome_icons(out)
    ensure_knob_assets(dest=out)
    profile = active_profile()

    def save_mp4(frames, filename, duration=95):
        import imageio.v2 as imageio

        from hita.config.export import fit_image_to_macro_block

        path = out / filename
        frame_list = [fit_image_to_macro_block(im) for im in frames]
        imageio.mimsave(
            path,
            frame_list,
            fps=max(1.0, 1000.0 / float(duration)),
            macro_block_size=1,
        )
        return path

    def save_gif(images, filename, duration=40):
        from PIL import Image

        rgb = [im.convert("RGB") for im in images]
        rgb[0].save(
            out / filename,
            save_all=True,
            append_images=rgb[1:],
            duration=duration,
            loop=0,
        )

    return {
        "__name__": "__hita_export_worker__",
        "OUTPUT_DIR": out,
        "EXPORT_FIGSIZE": (15.0, 9.5),
        "EXPORT_DPI": profile.export_dpi,
        "FONT_SIZE": 11 * 1.25,
        "AXIS_LABEL_SIZE": 12 * 1.25,
        "LEGEND_SIZE": 20,
        "TITLE_SIZE": 12 * 1.25,
        "SAVE_PAD_INCHES": 0.12,
        "sigmoid": sigmoid,
        "save_mp4": save_mp4,
        "save_gif": save_gif,
    }


def _load_ch1(root: Path, g: dict[str, Any]) -> dict[str, Any]:
    """Ch1: library primitives + key notebook setup cells for full Scene access."""
    from hita.stories import ch1 as ch1_story

    nb = legacy_notebooks_dir() / "logistic-regression-chap1.ipynb"
    for idx in (1, 2, 22):
        src = _nb_cell_source(nb, idx)
        exec(compile(src, f"ch1_cell{idx}", "exec"), g)
    ch1_story.install(g)
    return g


def _load_ch2(root: Path, g: dict[str, Any]) -> dict[str, Any]:
    from hita.stories import ch2 as ch2_story

    _load_ch1(root, g)
    nb = legacy_notebooks_dir() / "logistic-regression-chap2.ipynb"
    src = _nb_cell_source(nb, 1)
    exec(compile(src, "ch2_cell1", "exec"), g)
    ch2_story.install(g)
    return g


def _load_ch3_plus(root: Path, g: dict[str, Any], chapter: int) -> dict[str, Any]:
    _ensure_legacy_on_path()
    exec("from ch4_layout import *", g)
    exec("from ch4_layout import _ch4_formula_hand_block", g)

    ch3_nb = legacy_notebooks_dir() / "logistic-regression-chap3.ipynb"
    ch4_nb = legacy_notebooks_dir() / "logistic-regression-chap4.ipynb"
    ch3_src = _patch_ch3_setup_src(_nb_cell_source(ch3_nb, 1), root)
    exec(compile(ch3_src, "ch3", "exec"), g)

    if chapter >= 4 and ch4_nb.is_file():
        ch4_cell = _patch_ch4_setup_src(_nb_cell_source(ch4_nb, 1), ch3_nb)
        exec(compile(ch4_cell, str(ch4_nb), "exec"), g)

    import ch4_layout as c4

    if hasattr(c4, "_ch4_formula_hand_block"):
        g["_ch4_formula_hand_block"] = c4._ch4_formula_hand_block

    if chapter >= 5:
        # Duo/diagnostics layout helpers live in ch6_layout; trust (ch5) depends on them.
        exec("from ch6_layout import *", g)
    # Story shims install export_clip only for the *target* chapter so
    # inherited stubs do not overwrite a later chapter's exporter.
    if chapter == 5:
        exec("from ch5_layout import *", g)
        from ch5_story import _ch5_finish_duo_export
        from hita.stories import ch5 as ch5_story

        ch5_story.install(g)
        g["_ch5_finish_duo_export"] = _ch5_finish_duo_export
    if chapter == 6:
        from hita.stories import ch6 as ch6_story

        ch6_story.install(g)
    if chapter == 4:
        from hita.stories import ch4 as ch4_story

        ch4_story.install(g)
    if chapter == 3:
        from hita.stories import ch3 as ch3_story

        ch3_story.install(g)

    from hita.primitives.knobs import KnobStyle, load_knob_pack, probe_canvas_side

    # Prefer library probe + packs so numbered/labeled assets resolve without notebook cwd.
    c4._CH4_KNOB_PROBE_FN = probe_canvas_side
    g["_ch3_probe_knob_canvas_side"] = probe_canvas_side
    g["ch3_knob_asset_pack"] = lambda: load_knob_pack(KnobStyle.NUMBERED).as_legacy()
    g["ch4_knob_asset_pack"] = lambda: load_knob_pack(KnobStyle.LABELED).as_legacy()
    return g


def _load_native(chapter: int, g: dict[str, Any]) -> dict[str, Any]:
    """Chapters born on hita — import stories only, never exec notebooks."""
    if chapter == 7:
        from hita.stories import ch7 as ch7_story

        ch7_story.install(g)
        return g
    raise ValueError(f"no native hita loader for chapter {chapter}")


def load_export_context(root: Path | str | None = None, *, chapter: int = 5) -> dict[str, Any]:
    """Load chapter builder globals (once per worker process per chapter).

    Multiple chapters are retained in ``_CTX_BY_CHAPTER`` so switching chapters
    in one process does not discard prior cold starts.
    """
    global _ROOT, _CTX, _CTX_CHAPTER
    chapter = int(chapter)
    if chapter in _CTX_BY_CHAPTER:
        _CTX = _CTX_BY_CHAPTER[chapter]
        _CTX_CHAPTER = chapter
        return _CTX

    apply_profile_env()
    root = Path(root or project_root()).resolve()
    _ROOT = root
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Native chapters never need legacy on path / notebook exec
    if chapter >= 7:
        g = _base_globals(root)
        _load_native(chapter, g)
        g["_HITA_CHAPTER"] = chapter
        g["_HITA_NATIVE"] = True
        _CTX_BY_CHAPTER[chapter] = g
        _CTX = g
        _CTX_CHAPTER = chapter
        return g

    _ensure_legacy_on_path()

    lock_path = root / "renders" / _CH3_SETUP_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            if chapter in _CTX_BY_CHAPTER:
                _CTX = _CTX_BY_CHAPTER[chapter]
                _CTX_CHAPTER = chapter
                return _CTX

            g = _base_globals(root)
            slim = os.environ.get(_SLIM_WORKER_ENV, "").strip().lower() in {"1", "true", "yes", "y"}
            if slim and chapter == 5:
                _load_ch5_slim(root, g)
            elif chapter <= 1:
                _load_ch1(root, g)
            elif chapter == 2:
                _load_ch2(root, g)
            elif 3 <= chapter <= 6:
                _load_ch3_plus(root, g, chapter)
            else:
                raise ValueError(f"unsupported chapter: {chapter}")

            g["_HITA_CHAPTER"] = chapter
            g["_HITA_NATIVE"] = False
            g["_HITA_SLIM"] = bool(slim and chapter == 5)
            _CTX_BY_CHAPTER[chapter] = g
            _CTX = g
            _CTX_CHAPTER = chapter
            return g
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _load_ch5_slim(root: Path, g: dict[str, Any]) -> dict[str, Any]:
    """Render-oriented Ch5 load: layouts + story install without re-execing notebooks when possible.

    Falls back to full ``_load_ch3_plus`` if required Ch3 symbols are missing after a light import.
    """
    try:
        exec("from ch4_layout import *", g)
        exec("from ch6_layout import *", g)
        # Minimal stubs some finish helpers expect
        if "EXPORT_DPI" not in g:
            from hita.config.profile import active_profile

            g["EXPORT_DPI"] = active_profile().export_dpi
        from hita.stories import ch5 as ch5_story

        # Still need Ch3 NLL for pack calibration — use full path if missing
        if "_ch3_nll_sum_on_flat_grid" not in g:
            return _load_ch3_plus(root, g, 5)
        ch5_story.install(g)
        from ch5_story import _ch5_finish_duo_export

        g["_ch5_finish_duo_export"] = _ch5_finish_duo_export
        return g
    except Exception:
        g.clear()
        g.update(_base_globals(root))
        return _load_ch3_plus(root, g, 5)


def get_context() -> dict[str, Any]:
    if _CTX is None:
        raise RuntimeError("export context not loaded — call load_export_context() first")
    return _CTX


def worker_init(root_str: str, chapter: int = 5) -> None:
    load_export_context(root_str, chapter=chapter)


def clear_context_cache() -> None:
    """Drop in-process chapter globals (tests / chapter switch)."""
    global _CTX, _CTX_CHAPTER
    _CTX_BY_CHAPTER.clear()
    _CTX = None
    _CTX_CHAPTER = None
