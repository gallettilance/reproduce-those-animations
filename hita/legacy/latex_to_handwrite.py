"""LaTeX → Patrick Hand helpers, including handwritten matrix layout."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

_MATRIX_ENVS = ("pmatrix", "bmatrix", "matrix", "Bmatrix", "vmatrix", "Vmatrix")


def parse_tex_matrix(tex: str) -> tuple[str, list[list[str]]]:
    """Parse ``label\\begin{pmatrix}...\\end{pmatrix}`` into label TeX + cell grid."""
    s = str(tex).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    env_re = r"\\begin\{(" + "|".join(_MATRIX_ENVS) + r")\*?\}"
    m = re.search(env_re, s)
    if not m:
        raise ValueError(f"No matrix environment found in: {tex!r}")
    label = s[: m.start()].strip()
    env_name = m.group(1)
    end_re = rf"\\end\{{{re.escape(env_name)}\*?\}}"
    end_m = re.search(end_re, s[m.end() :])
    if not end_m:
        raise ValueError(f"Unclosed matrix environment in: {tex!r}")
    inner = s[m.end() : m.end() + end_m.start()]
    rows: list[list[str]] = []
    for row in inner.split("\\\\"):
        row = row.strip()
        if not row:
            continue
        cells = [_clean_matrix_cell(c.strip()) for c in row.split("&")]
        rows.append(cells)
    if not rows:
        raise ValueError(f"Empty matrix in: {tex!r}")
    return label, rows


def _clean_matrix_cell(cell: str) -> str:
    s = str(cell).strip()
    s = re.sub(r"\\scriptstyle\s*", "", s)
    s = re.sub(r"\\displaystyle\s*", "", s)
    return s.strip()


def matrix_cell_style(base_style, spec: dict):
    """Per-matrix typography overrides (∂ size, weight subscripts, …)."""
    from dataclasses import replace

    kw: dict = {}
    if "cell_subscript_scale" in spec:
        kw["subscript_scale"] = float(spec["cell_subscript_scale"])
    if "cell_superscript_scale" in spec:
        kw["superscript_scale"] = float(spec["cell_superscript_scale"])
    if "cell_superscript_raise_frac" in spec:
        kw["superscript_raise_frac"] = float(spec["cell_superscript_raise_frac"])
    if "cell_symbol_scale" in spec:
        kw["symbol_scale"] = float(spec["cell_symbol_scale"])
    if "cell_frac_gap_frac" in spec:
        kw["frac_gap_frac"] = float(spec["cell_frac_gap_frac"])
    if "cell_frac_pad_frac" in spec:
        kw["frac_pad_frac"] = float(spec["cell_frac_pad_frac"])
    if "cell_frac_scale" in spec:
        kw["frac_scale"] = float(spec["cell_frac_scale"])
    return replace(base_style, **kw) if kw else base_style


def expand_matrix_row_gaps(
    row_heights: list[float],
    row_gap_px: float,
    *,
    max_h_px: float,
    fill_height_frac: float,
) -> float:
    """Stretch row spacing so the grid uses ``fill_height_frac`` of cell height."""
    n_rows = len(row_heights)
    if n_rows <= 1 or fill_height_frac <= 0.0:
        return row_gap_px
    grid_h = sum(row_heights) + row_gap_px * max(n_rows - 1, 0)
    target = float(max_h_px) * float(fill_height_frac)
    if grid_h >= target:
        return row_gap_px
    return row_gap_px + (target - grid_h) / float(n_rows - 1)


def matrix_spec(block: dict) -> dict[str, Any]:
    raw = block.get("handwrite_matrix") or {}
    if not isinstance(raw, dict):
        raw = {}
    return raw


def matrix_cells_handwrite(
    cells_tex: list[list[str]],
    *,
    converter=None,
) -> list[list[str]]:
    if converter is None:
        from handwrite_tutorial import latex_line_to_handwrite

        converter = latex_line_to_handwrite
    out: list[list[str]] = []
    for row in cells_tex:
        out.append([])
        for cell in row:
            tex = str(cell).strip()
            if not tex:
                out[-1].append("")
                continue
            if not tex.startswith("$"):
                tex = f"${tex}$"
            out[-1].append(converter(tex))
    return out


def _matrix_cell_metrics(
    renderer,
    cells_hw: list[list[str]],
    cell_fs: float,
    *,
    style,
    bold: bool,
    fp_hand,
    col_gap_px: float,
    row_gap_px: float,
):
    from handwrite_tutorial import (
        mixed_line_height_px,
        mixed_line_width_px,
        parse_handwrite_runs,
    )

    n_rows = len(cells_hw)
    n_cols = max((len(r) for r in cells_hw), default=0)
    col_widths = [0.0] * n_cols
    row_heights = [0.0] * n_rows
    for ri, row in enumerate(cells_hw):
        for ci, cell in enumerate(row):
            runs = parse_handwrite_runs(cell)
            col_widths[ci] = max(
                col_widths[ci],
                mixed_line_width_px(renderer, runs, cell_fs, style=style, bold=bold, fp_hand=fp_hand),
            )
            row_heights[ri] = max(
                row_heights[ri],
                mixed_line_height_px(renderer, runs, cell_fs, style=style, bold=bold, fp_hand=fp_hand),
            )
    grid_w = sum(col_widths) + col_gap_px * max(n_cols - 1, 0)
    grid_h = sum(row_heights) + row_gap_px * max(n_rows - 1, 0)
    return col_widths, row_heights, grid_w, grid_h


def matrix_bracket_pair(spec: dict | None) -> tuple[str, str]:
    """Return left/right delimiter chars for a matrix (round ``()`` or square ``[]``)."""
    if str((spec or {}).get("bracket_style", "round")).lower() == "square":
        return "[", "]"
    return "(", ")"


def matrix_bracket_is_lines(spec: dict | None) -> bool:
    return str((spec or {}).get("bracket_draw", "glyph")).lower() == "lines"


def matrix_bracket_tick_width_px(fig, block: dict, spec: dict | None) -> float:
    from handwrite_tutorial import block_pt_to_px

    tick_pt = float((spec or {}).get("bracket_tick_width_pt", 2.2))
    return block_pt_to_px(fig, block, tick_pt)


def _draw_matrix_line_bracket(
    target,
    x_vert: float,
    y_top: float,
    y_bot: float,
    *,
    side: str,
    transform,
    color: str,
    fig,
    block: dict,
    spec: dict,
) -> None:
    renderer = fig.canvas.get_renderer()
    if hasattr(target, "transAxes"):
        win = target.get_window_extent(renderer)
    else:
        win = fig.bbox
    ax_w_px = max(float(win.width), 1.0)
    tick_w = matrix_bracket_tick_width_px(fig, block, spec) / ax_w_px
    lw = float(spec.get("bracket_line_width_pt", 0.75))
    solid_capstyle = "projecting"
    target.plot(
        [x_vert, x_vert], [y_bot, y_top],
        transform=transform, color=color, linewidth=lw,
        solid_capstyle=solid_capstyle, clip_on=False, zorder=5,
    )
    if str(side) == "left":
        target.plot(
            [x_vert, x_vert + tick_w], [y_top, y_top],
            transform=transform, color=color, linewidth=lw,
            solid_capstyle=solid_capstyle, clip_on=False, zorder=5,
        )
        target.plot(
            [x_vert, x_vert + tick_w], [y_bot, y_bot],
            transform=transform, color=color, linewidth=lw,
            solid_capstyle=solid_capstyle, clip_on=False, zorder=5,
        )
    else:
        target.plot(
            [x_vert - tick_w, x_vert], [y_top, y_top],
            transform=transform, color=color, linewidth=lw,
            solid_capstyle=solid_capstyle, clip_on=False, zorder=5,
        )
        target.plot(
            [x_vert - tick_w, x_vert], [y_bot, y_bot],
            transform=transform, color=color, linewidth=lw,
            solid_capstyle=solid_capstyle, clip_on=False, zorder=5,
        )


def matrix_bracket_metrics(
    spec: dict | None,
    cell_fs: float,
    grid_h: float,
    fig,
) -> tuple[float, str]:
    """Return ``(bracket_fs, bracket_weight)`` for matrix delimiters."""
    spec = spec or {}
    height_frac = float(spec.get("bracket_height_frac", 1.28))
    min_cell = float(spec.get("bracket_min_cell_scale", 1.05))
    fs_scale = float(spec.get("bracket_fs_scale", 0.92))
    paren_fs = max(
        float(cell_fs) * min_cell,
        float(grid_h) * height_frac * (72.0 / fig.dpi),
    ) * fs_scale
    weight = str(spec.get("bracket_weight", "ultralight"))
    return float(paren_fs), weight


def _matrix_total_width_px(
    renderer,
    cells_hw: list[list[str]],
    cell_fs: float,
    *,
    style,
    bold: bool,
    fp_hand,
    col_gap_px: float,
    row_gap_px: float,
    label_hw: str,
    label_fs: float,
    fig,
    block: dict,
    paren_gap_pt: float,
    label_gap_pt: float,
    bracket_left: str = "(",
    bracket_right: str = ")",
    bracket_spec: dict | None = None,
) -> tuple[float, float, list[float]]:
    """Return ``(total_w_px, grid_h_px, row_heights)`` including label and brackets."""
    from handwrite_tutorial import (
        block_pt_to_px,
        mixed_line_height_px,
        mixed_line_width_px,
        parse_handwrite_runs,
        run_size_px,
    )

    col_widths, row_heights, grid_w, grid_h = _matrix_cell_metrics(
        renderer, cells_hw, cell_fs, style=style, bold=bold, fp_hand=fp_hand,
        col_gap_px=col_gap_px, row_gap_px=row_gap_px,
    )
    paren_gap_px = block_pt_to_px(fig, block, paren_gap_pt)
    label_gap_px = block_pt_to_px(fig, block, label_gap_pt)
    label_w = 0.0
    if str(label_hw or "").strip():
        label_w = mixed_line_width_px(
            renderer, parse_handwrite_runs(str(label_hw)), label_fs,
            style=style, bold=bold, fp_hand=fp_hand,
        )
    paren_fs, _ = matrix_bracket_metrics(bracket_spec, cell_fs, grid_h, fig)
    if matrix_bracket_is_lines(bracket_spec):
        lp_w = rp_w = matrix_bracket_tick_width_px(fig, block, bracket_spec)
    else:
        lp_w, _ = run_size_px(renderer, "sym", bracket_left, paren_fs, style=style, bold=bold, fp_hand=fp_hand)
        rp_w, _ = run_size_px(renderer, "sym", bracket_right, paren_fs, style=style, bold=bold, fp_hand=fp_hand)
    inner_w = lp_w + paren_gap_px + grid_w + paren_gap_px + rp_w
    total_w = inner_w + (label_w + label_gap_px if label_hw else 0.0)
    return float(total_w), float(grid_h), row_heights


def fit_handwrite_matrix_fontsize(
    renderer,
    cells_hw: list[list[str]],
    max_w_px: float,
    max_h_px: float,
    start_fs: float,
    *,
    style,
    bold: bool,
    fp_hand,
    col_gap_pt: float,
    row_gap_pt: float,
    fig,
    block: dict,
    spec: dict,
    label_hw: str = "",
    label_fs: float | None = None,
) -> tuple[float, float, float]:
    """Return ``(cell_fs, col_gap_px, row_gap_px)`` — width-limited, height filled."""
    from handwrite_tutorial import block_pt_to_px

    min_fs = float(spec.get("min_fs", block.get("matrix_min_fs", 7.0)))
    fit_mode = str(spec.get("fit_mode", "width")).lower()
    fill_frac = float(spec.get("fill_height_frac", 0.0))
    max_up = float(spec.get("cell_fs_max_scale", 1.08))
    paren_gap_pt = float(spec.get("paren_gap_pt", 2.0))
    label_gap_pt = float(spec.get("label_gap_pt", 4.0))
    l_fs = float(label_fs if label_fs is not None else start_fs)

    fs = float(start_fs)

    def _gaps(fsz: float) -> tuple[float, float]:
        scale = fsz / start_fs if start_fs > 1e-6 else 1.0
        col_fixed = bool(spec.get("col_gap_fixed", False))
        row_fixed = bool(spec.get("row_gap_fixed", False))
        cg_pt = float(col_gap_pt) if col_fixed else float(col_gap_pt) * scale
        rg_pt = float(row_gap_pt) if row_fixed else float(row_gap_pt) * scale
        return (
            block_pt_to_px(fig, block, cg_pt),
            block_pt_to_px(fig, block, rg_pt),
        )

    bl, br = matrix_bracket_pair(spec)

    def _fits(fsz: float) -> tuple[bool, list[float], float, float, float, float]:
        cg, rg = _gaps(fsz)
        total_w, gh, row_h = _matrix_total_width_px(
            renderer, cells_hw, fsz, style=style, bold=bold, fp_hand=fp_hand,
            col_gap_px=cg, row_gap_px=rg, label_hw=label_hw, label_fs=l_fs,
            fig=fig, block=block, paren_gap_pt=paren_gap_pt, label_gap_pt=label_gap_pt,
            bracket_left=bl, bracket_right=br, bracket_spec=spec,
        )
        ok_w = total_w <= float(max_w_px)
        ok_h = gh <= float(max_h_px)
        if fit_mode == "width":
            ok = ok_w
        elif fit_mode == "height":
            ok = ok_h
        else:
            ok = ok_w and ok_h
        return ok, row_h, cg, rg, total_w, gh

    while fs > min_fs:
        ok, _, _, _, _, _ = _fits(fs)
        if ok:
            break
        fs -= 0.5

    fs_up = fs
    cap = start_fs * max_up
    while fs_up + 0.25 <= cap:
        ok, _, _, _, total_w, _ = _fits(fs_up + 0.25)
        if ok and total_w <= float(max_w_px):
            fs_up += 0.25
        else:
            break
    fs = fs_up

    ok, row_heights, col_gap_px, row_gap_px, _, _ = _fits(fs)
    if fill_frac > 0.0:
        row_gap_px = expand_matrix_row_gaps(
            row_heights, row_gap_px, max_h_px=max_h_px, fill_height_frac=fill_frac,
        )
    return fs, col_gap_px, row_gap_px


def draw_handwrite_matrix(
    target,
    x: float,
    y_top: float,
    cells_hw: list[list[str]],
    *,
    style,
    label: str = "",
    label_fs: float | None = None,
    cell_fs: float,
    bold: bool = False,
    color: str,
    ha: str = "center",
    transform=None,
    write_progress: float = 1.0,
    reveal_mode: str = "char",
    fp_hand=None,
    col_gap_pt: float = 5.0,
    row_gap_pt: float = 2.5,
    col_gap_px: float | None = None,
    row_gap_px: float | None = None,
    paren_gap_pt: float = 2.0,
    label_gap_pt: float = 4.0,
    block: dict | None = None,
) -> float:
    """Draw ``label [ grid ]`` (or round brackets) in Patrick Hand with aligned columns."""
    from handwrite_tutorial import (
        block_pt_to_px,
        draw_mixed_line,
        hand_font,
        line_write_progresses,
        mixed_line_height_px,
        mixed_line_width_px,
        parse_handwrite_runs,
        run_size_px,
        symbol_font,
    )

    if not cells_hw:
        return 0.0
    block = block or {}
    mspec = matrix_spec(block)
    bl, br = matrix_bracket_pair(mspec)
    fig = target.figure
    if transform is None:
        transform = getattr(target, "transAxes", fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if hasattr(target, "transAxes"):
        win = target.get_window_extent(renderer)
    else:
        win = fig.bbox
    ax_w_px = max(float(win.width), 1.0)
    ax_h_px = max(float(win.height), 1.0)

    col_gap_px = float(col_gap_px) if col_gap_px is not None else block_pt_to_px(fig, block, col_gap_pt)
    row_gap_px = float(row_gap_px) if row_gap_px is not None else block_pt_to_px(fig, block, row_gap_pt)
    paren_gap_px = block_pt_to_px(fig, block, paren_gap_pt)
    label_gap_px = block_pt_to_px(fig, block, label_gap_pt)

    col_widths, row_heights, grid_w, grid_h = _matrix_cell_metrics(
        renderer, cells_hw, cell_fs, style=style, bold=bold, fp_hand=fp_hand,
        col_gap_px=col_gap_px, row_gap_px=row_gap_px,
    )
    n_rows = len(cells_hw)
    n_cols = max((len(r) for r in cells_hw), default=0)

    label_hw = str(label or "").strip()
    label_w = 0.0
    label_h = 0.0
    l_fs = float(label_fs if label_fs is not None else cell_fs)
    if label_hw:
        label_runs = parse_handwrite_runs(label_hw)
        label_w = mixed_line_width_px(
            renderer, label_runs, l_fs, style=style, bold=bold, fp_hand=fp_hand,
        )
        label_h = mixed_line_height_px(
            renderer, label_runs, l_fs, style=style, bold=bold, fp_hand=fp_hand,
        )

    paren_fs, bracket_weight = matrix_bracket_metrics(mspec, cell_fs, grid_h, fig)
    use_line_brackets = matrix_bracket_is_lines(mspec)
    use_hand_brackets = bool(mspec.get("bracket_use_handwrite", False)) and not use_line_brackets
    if use_line_brackets:
        lp_w = rp_w = matrix_bracket_tick_width_px(fig, block, mspec)
        lp_h = grid_h
        bracket_fp = None
    elif use_hand_brackets:
        from handwrite_tutorial import hand_font
        bracket_fp = hand_font(paren_fs, bold=False)
        lp_w, lp_h = run_size_px(renderer, "hand", bl, paren_fs, style=style, bold=False, fp_hand=bracket_fp)
        rp_w, _ = run_size_px(renderer, "hand", br, paren_fs, style=style, bold=False, fp_hand=bracket_fp)
    else:
        lp_w, lp_h = run_size_px(renderer, "sym", bl, paren_fs, style=style, bold=bold, fp_hand=fp_hand)
        rp_w, _ = run_size_px(renderer, "sym", br, paren_fs, style=style, bold=bold, fp_hand=fp_hand)
        bracket_fp = symbol_font(paren_fs, style=style, weight=bracket_weight)

    inner_w = lp_w + paren_gap_px + grid_w + paren_gap_px + rp_w
    total_w = (label_w + label_gap_px + inner_w) if label_hw else inner_w
    total_h = max(grid_h, label_h, lp_h)

    x_cursor = x
    if ha == "center":
        x_cursor = x - (total_w / ax_w_px) / 2.0
    elif ha == "right":
        x_cursor = x - (total_w / ax_w_px)

    row_tops: list[float] = []
    y_row = y_top - max(0.0, (total_h - grid_h) / (2.0 * ax_h_px))
    for rh in row_heights:
        row_tops.append(y_row)
        y_row -= (rh + row_gap_px) / ax_h_px
    grid_bottom = row_tops[-1] - row_heights[-1] / ax_h_px if row_tops else y_top
    y_shift = 0.0
    if mspec.get("bracket_y_shift_pt") is not None:
        y_shift = -block_pt_to_px(fig, block, float(mspec["bracket_y_shift_pt"])) / ax_h_px
    y_br_top = row_tops[0] + y_shift if row_tops else y_top
    y_br_bot = grid_bottom + y_shift
    grid_center_y = (row_tops[0] + grid_bottom) / 2.0 if row_tops else y_top
    bracket_align = str(mspec.get("bracket_align", "grid")).lower()
    if bracket_align == "center_row" and n_rows > 0 and not use_line_brackets:
        mid = n_rows // 2
        bracket_y = row_tops[mid] - (row_heights[mid] / ax_h_px) / 2.0 + y_shift
    else:
        bracket_y = grid_center_y + y_shift

    bracket_color = color
    if mspec.get("bracket_color") is not None:
        bracket_color = str(mspec["bracket_color"])

    cell_y_down = 0.0
    if mspec.get("matrix_cell_y_shift_pt") is not None:
        cell_y_down = block_pt_to_px(fig, block, float(mspec["matrix_cell_y_shift_pt"])) / ax_h_px

    prog = float(np.clip(float(write_progress), 0.0, 1.0))
    if label_hw and prog > 0.0:
        y_label = y_top - max(0.0, (total_h - label_h) / (2.0 * ax_h_px))
        draw_mixed_line(
            target, x_cursor, y_label, label_hw, l_fs, style=style, bold=bold, color=color,
            ha="left", va="top", transform=transform, write_progress=prog,
            reveal_mode=reveal_mode, fp_hand=fp_hand,
        )
        x_cursor += (label_w + label_gap_px) / ax_w_px

    if prog > 0.0 and use_line_brackets:
        _draw_matrix_line_bracket(
            target, x_cursor, y_br_top, y_br_bot,
            side="left", transform=transform, color=bracket_color,
            fig=fig, block=block, spec=mspec,
        )
    elif prog > 0.0:
        target.text(
            x_cursor, bracket_y, bl, transform=transform, va="center", ha="left",
            color=bracket_color, fontproperties=bracket_fp, clip_on=False,
        )
    x_grid = x_cursor + (lp_w + paren_gap_px) / ax_w_px

    raw_cell_colors = mspec.get("cell_colors")
    cell_progs = line_write_progresses(n_rows * n_cols, prog, overlap=style.reveal_overlap)
    pi = 0
    for ri, row in enumerate(cells_hw):
        x_cell = x_grid
        y_cell = row_tops[ri] - cell_y_down
        for ci in range(n_cols):
            cell = row[ci] if ci < len(row) else ""
            cell_prog = cell_progs[pi] if pi < len(cell_progs) else prog
            pi += 1
            cell_color = color
            if isinstance(raw_cell_colors, (list, tuple)) and ri < len(raw_cell_colors):
                row_cols = raw_cell_colors[ri]
                if isinstance(row_cols, (list, tuple)) and ci < len(row_cols) and row_cols[ci]:
                    cell_color = str(row_cols[ci])
            if cell:
                draw_mixed_line(
                    target, x_cell, y_cell, cell, cell_fs, style=style, bold=bold, color=cell_color,
                    ha="left", va="top", transform=transform, write_progress=cell_prog,
                    reveal_mode=reveal_mode, fp_hand=fp_hand,
                )
            if ci < n_cols - 1:
                x_cell += (col_widths[ci] + col_gap_px) / ax_w_px

    x_paren_r = x_grid + (grid_w + paren_gap_px) / ax_w_px
    if prog > 0.0 and use_line_brackets:
        _draw_matrix_line_bracket(
            target, x_paren_r, y_br_top, y_br_bot,
            side="right", transform=transform, color=bracket_color,
            fig=fig, block=block, spec=mspec,
        )
    elif prog > 0.0:
        target.text(
            x_paren_r, bracket_y, br, transform=transform, va="center", ha="left",
            color=bracket_color, fontproperties=bracket_fp, clip_on=False,
        )
    return float(total_h)


def draw_handwrite_matrix_in_cell(
    ax,
    block: dict,
    *,
    style,
    block_fs: float,
    align: str = "center",
    line_progress: dict | None = None,
    text_color: str | None = None,
    text_x_frac: float | None = None,
    text_y_inset_pt: float = 0.0,
):
    """Render a ``handwrite_matrix`` block inside a tutorial formula cell."""
    from handwrite_tutorial import block_pt_to_px, hand_font

    spec = matrix_spec(block)
    cells_tex = spec.get("cells_tex")
    if cells_tex is None and spec.get("matrix_tex"):
        _, cells_tex = parse_tex_matrix(str(spec["matrix_tex"]))
    if not cells_tex:
        raise ValueError("handwrite_matrix block requires cells_tex or matrix_tex")

    label_tex = spec.get("label_tex", "")
    if not label_tex and spec.get("matrix_tex"):
        label_tex, _ = parse_tex_matrix(str(spec["matrix_tex"]))

    cells_hw = matrix_cells_handwrite(cells_tex)
    if label_tex:
        from handwrite_tutorial import latex_line_to_handwrite

        lt = str(label_tex).strip()
        if not lt.startswith("$"):
            lt = f"${lt}$"
        label_hw = latex_line_to_handwrite(lt)
    else:
        label_hw = ""

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_h_px = max(float(ax.get_window_extent(renderer).height), 1.0)
    ax_w_px = max(float(ax.get_window_extent(renderer).width), 1.0)

    body_fs = float(block.get("block_fs", block_fs))
    cell_scale = float(spec.get("cell_fs_scale", block.get("matrix_cell_fs_scale", 0.78)))
    label_scale = float(spec.get("label_fs_scale", 1.0))
    start_cell_fs = body_fs * cell_scale
    label_fs = body_fs * label_scale
    cell_style = matrix_cell_style(style, spec)

    ha = str(spec.get("align", align or "center")).lower()
    if ha not in ("left", "center", "right"):
        ha = "center"
    x = float(text_x_frac if text_x_frac is not None else (0.5 if ha == "center" else 0.04))
    max_frac = float(block.get("matrix_max_frac", spec.get("max_frac", 0.96)))
    max_w_px = ax_w_px * max_frac
    max_h_px = max(
        ax_h_px
        - block_pt_to_px(fig, block, float(block.get("top_pad_pt", style.top_pad_pt)))
        - block_pt_to_px(fig, block, float(style.bottom_pad_pt))
        - block_pt_to_px(fig, block, float(text_y_inset_pt)),
        1.0,
    )

    fp_hand = hand_font(body_fs, bold=bool(block.get("bold_lhs", False))) if style.enabled else None
    cell_fs, col_gap, row_gap = fit_handwrite_matrix_fontsize(
        renderer, cells_hw, max_w_px, max_h_px, start_cell_fs,
        style=cell_style, bold=bool(block.get("bold_lhs", False)), fp_hand=fp_hand,
        col_gap_pt=float(spec.get("col_gap_pt", 5.0)),
        row_gap_pt=float(spec.get("row_gap_pt", 6.0)),
        fig=fig, block=block, spec=spec,
        label_hw=label_hw, label_fs=label_fs,
    )

    y_px = block_pt_to_px(fig, block, float(block.get("top_pad_pt", style.top_pad_pt)))
    y_px += block_pt_to_px(fig, block, float(text_y_inset_pt))
    if block.get("matrix_y_shift_pt") is not None:
        y_px += block_pt_to_px(fig, block, float(block["matrix_y_shift_pt"]))
    if spec.get("y_shift_pt") is not None:
        y_px += block_pt_to_px(fig, block, float(spec["y_shift_pt"]))
    y_top = 1.0 - (y_px / ax_h_px)

    if block.get("matrix_x_shift_pt") is not None:
        x = (x * ax_w_px + block_pt_to_px(fig, block, float(block["matrix_x_shift_pt"]))) / ax_w_px
    if spec.get("x_shift_pt") is not None:
        x = (x * ax_w_px + block_pt_to_px(fig, block, float(spec["x_shift_pt"]))) / ax_w_px

    prog = float((line_progress or {}).get(0, (line_progress or {}).get("0", 1.0)))
    color = str(text_color or block.get("text_color") or style.text_color)

    draw_handwrite_matrix(
        ax, x, y_top, cells_hw,
        style=cell_style,
        label=label_hw,
        label_fs=label_fs,
        cell_fs=cell_fs,
        bold=bool(block.get("bold_lhs", False)),
        color=color,
        ha=ha,
        transform=ax.transAxes,
        write_progress=prog,
        reveal_mode=style.line_mode,
        fp_hand=fp_hand,
        col_gap_px=col_gap,
        row_gap_px=row_gap,
        paren_gap_pt=float(spec.get("paren_gap_pt", 2.0)),
        label_gap_pt=float(spec.get("label_gap_pt", 4.0)),
        block=block,
    )
