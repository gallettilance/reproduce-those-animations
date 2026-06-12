"""Reusable handwriting font + typewriter reveal for plot + math tutorial frames."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib import font_manager as fm
import numpy as np

HANDWRITING_FAMILIES = (
    "Patrick Hand",
    "Caveat",
    "Bradley Hand",
    "Marker Felt",
    "Snell Roundhand",
    "Comic Sans MS",
)
HANDWRITING_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
BUNDLED_FONTS = {
    "Caveat": HANDWRITING_FONT_DIR / "Caveat-Variable.ttf",
    "Patrick Hand": HANDWRITING_FONT_DIR / "PatrickHand-Regular.ttf",
}
_FONTS_REGISTERED = False
_ACTIVE_FAMILY: str | None = None
_AMSMATH_PREPARED = False


def ensure_amsmath_preamble() -> None:
    """Ensure LaTeX ``cases`` / ``align`` environments work with ``usetex=True``."""
    global _AMSMATH_PREPARED
    if _AMSMATH_PREPARED:
        return
    pre = str(mpl.rcParams.get("text.latex.preamble", "") or "")
    if "amsmath" not in pre:
        mpl.rcParams["text.latex.preamble"] = pre + r"\usepackage{amsmath}"
    _AMSMATH_PREPARED = True


@dataclass
class HandwriteStyle:
    """Typography, colors, and reveal settings — layout-agnostic."""

    enabled: bool = True
    section_title_fs: float = 24.2
    line_mode: str = "char"
    title_mode: str = "word"
    ease: str = "out_cubic"
    reveal_overlap: float = 0.28
    label_color: str = "#55556a"
    text_color: str = "#000000"
    title_color: str = "#1a1a28"
    accent_color: str = "#000000"
    section_title_bold: bool = True
    label_gap_pt: float = 10.0
    line_dy_pt: float = 15.0
    top_pad_pt: float = 3.0
    bottom_pad_pt: float = 5.0
    title_line_dy_pt: float = 3.0
    frame_edge: str = "#4a4a5a"
    frame_lw: float = 0.7
    title_reveal_boost: float = 1.4
    symbol_font_family: str = "STIX Two Math"
    symbol_scale: float = 1.0          # ∂ ∇ ∈ … (not Σ/Π limops)
    subscript_scale: float = 0.62
    subscript_drop_frac: float = 0.12
    superscript_scale: float = 0.58
    superscript_raise_frac: float = 0.38
    limop_symbol_scale: float = 1.82
    limop_below_scale: float = 0.36
    limop_above_scale: float = 0.36
    limop_below_drop_frac: float = 0.34
    limop_above_raise_frac: float = 0.62
    limop_limit_gap_frac: float = 0.10
    limop_above_gap_frac: float = 0.18
    limop_below_gap_frac: float = 0.14
    cases_brace_height_ratio: float = 2.15
    cases_row_gap_frac: float = 0.68
    cases_row_gap_pt: float | None = 8.0 * 72.0 / 25.4   # default 8 mm between case rows
    cases_pad_frac: float = 0.028


SYMBOL_CHARS = frozenset("∇∂Σ∑ℒΠ∈σα←→Δδ")
CASES_OPEN = "<<CASES>>"
CASES_CLOSE = "<</CASES>>"
BOLD_OPEN = "<<B>>"
BOLD_CLOSE = "<</B>>"
Run = tuple[str, ...]  # hand|hand_b|sub|sup|sym|(limop, op, below, above)


def register_handwriting_fonts() -> None:
    global _FONTS_REGISTERED, _ACTIVE_FAMILY
    if _FONTS_REGISTERED:
        return
    if HANDWRITING_FONT_DIR.is_dir():
        for ttf in sorted(HANDWRITING_FONT_DIR.glob("*.ttf")):
            fm.fontManager.addfont(str(ttf))
    names = {f.name for f in fm.fontManager.ttflist}
    for fam in HANDWRITING_FAMILIES:
        bundled = BUNDLED_FONTS.get(fam)
        if bundled is not None and bundled.is_file():
            _ACTIVE_FAMILY = fam
            break
        if fam in names:
            _ACTIVE_FAMILY = fam
            break
    _FONTS_REGISTERED = True


def active_handwriting_family() -> str:
    register_handwriting_fonts()
    return _ACTIVE_FAMILY or "sans-serif"


def hand_font(size: float, *, bold: bool = False) -> fm.FontProperties:
    register_handwriting_fonts()
    fam = active_handwriting_family()
    bundled = BUNDLED_FONTS.get(fam)
    if bundled is not None and bundled.is_file():
        weight = 700 if (bold and fam == "Caveat") else ("bold" if bold else "normal")
        return fm.FontProperties(fname=str(bundled), size=float(size), weight=weight)
    weight = "bold" if bold else "normal"
    return fm.FontProperties(family=fam, size=float(size), weight=weight)


def symbol_font(
    size: float,
    *,
    style: HandwriteStyle | None = None,
    weight: str | None = None,
) -> fm.FontProperties:
    style = style or HandwriteStyle()
    fp = fm.FontProperties(family=style.symbol_font_family, size=float(size))
    if weight is not None:
        fp.set_weight(str(weight))
    return fp


def _merge_adjacent_runs(runs: list[Run]) -> list[Run]:
    if not runs:
        return runs
    out: list[Run] = [runs[0]]
    for run in runs[1:]:
        if len(run) == 2 and len(out[-1]) == 2 and run[0] == out[-1][0]:
            out[-1] = (run[0], str(out[-1][1]) + str(run[1]))
        else:
            out.append(run)
    return out


def _read_braced_raw(s: str, i: int) -> tuple[str, int]:
    if i >= len(s) or s[i] != "{":
        return "", i
    depth, start = 0, i + 1
    i += 1
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            if depth == 0:
                return s[start:i], i + 1
            depth -= 1
        i += 1
    return s[start:], len(s)


def _read_braced(s: str, i: int) -> tuple[str, int]:
    inner, j = _read_braced_raw(s, i)
    return _tex_convert(inner), j


def _read_script(s: str, i: int) -> tuple[str, int]:
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _read_braced(s, i)
    if s[i] in "_^":
        i += 1
    start = i
    if i < len(s) and s[i].isalpha():
        while i < len(s) and s[i].isalpha():
            i += 1
        return s[start:i], i
    while i < len(s) and (s[i].isalnum() or s[i] in ",=+-."):
        i += 1
    return s[start:i], i


def _parse_handwrite_runs_plain(text: str) -> list[Run]:
    """Split plain display text into handwriting / scripts / limits / symbols."""
    runs: list[Run] = []
    i, n = 0, len(str(text))
    s = str(text)
    while i < n:
        if s[i] in "ΣΠ" and i + 1 < n and s[i + 1] == "{":
            op = s[i]
            i += 1
            below, i = _read_braced(s, i)
            if i < n and s[i] == "{":
                above, i = _read_braced(s, i)
            else:
                above = ""
            runs.append(("limop", op, below, above))
            continue
        ch = s[i]
        if ch in SYMBOL_CHARS:
            runs.append(("sym", ch))
            i += 1
        elif ch == "_":
            i += 1
            if i < n and s[i] == "{":
                inner, i = _read_braced_raw(s, i)
                runs.append(("sub", inner))
            else:
                start = i
                while i < n and s[i].isalnum():
                    i += 1
                sub = s[start:i]
                runs.append(("sub", sub) if sub else ("hand", "_"))
        elif ch == "^":
            i += 1
            if i < n and s[i] == "{":
                inner, i = _read_braced_raw(s, i)
                runs.append(("sup", inner))
            else:
                start = i
                while i < n and s[i].isalnum():
                    i += 1
                sup = s[start:i]
                runs.append(("sup", sup) if sup else ("hand", "^"))
        else:
            start = i
            while i < n and s[i] not in SYMBOL_CHARS and s[i] not in "_^ΣΠ":
                i += 1
            if start < i:
                runs.append(("hand", s[start:i]))
    return runs


def _bold_wrap_runs(runs: list[Run]) -> list[Run]:
    out: list[Run] = []
    for run in runs:
        if run[0] in ("hand", "sym"):
            out.append(("hand_b", str(run[1])))
        else:
            out.append(run)
    return out


def parse_handwrite_runs(text: str) -> list[Run]:
    """Split display text into handwriting / scripts / limits / symbols."""
    s = str(text)
    if BOLD_OPEN not in s:
        return _merge_adjacent_runs(_parse_handwrite_runs_plain(s))
    runs: list[Run] = []
    pos = 0
    while pos < len(s):
        idx = s.find(BOLD_OPEN, pos)
        if idx < 0:
            runs.extend(_parse_handwrite_runs_plain(s[pos:]))
            break
        if idx > pos:
            runs.extend(_parse_handwrite_runs_plain(s[pos:idx]))
        end = s.find(BOLD_CLOSE, idx + len(BOLD_OPEN))
        if end < 0:
            runs.extend(_parse_handwrite_runs_plain(s[idx:]))
            break
        inner = s[idx + len(BOLD_OPEN):end]
        runs.extend(_bold_wrap_runs(_parse_handwrite_runs_plain(inner)))
        pos = end + len(BOLD_CLOSE)
    return _merge_adjacent_runs(runs)


def flatten_runs(runs: list[Run]) -> list[Run]:
    atoms: list[Run] = []
    for run in runs:
        if run[0] == "limop":
            atoms.append(run)
            continue
        kind, text = run[0], str(run[1])
        if kind in ("sub", "sup") and len(text) > 1:
            atoms.append(run)
        else:
            for ch in text:
                atoms.append((kind, ch))
    return atoms


def runs_from_atoms(atoms: list[Run]) -> list[Run]:
    return _merge_adjacent_runs(atoms)


def reveal_handwrite_runs(
    runs: list[Run], progress: float, *, mode: str = "char", ease: str = "out_cubic"
) -> list[Run]:
    if progress >= 1.0:
        return runs
    atoms = flatten_runs(runs)
    if not atoms:
        return runs
    if mode == "word":
        # Reveal by hand-run words; symbols/subscripts stay attached to prior atom.
        plain = runs_plain(runs)
        shown = typewriter_reveal(plain, progress, mode="word", ease=ease)
        return parse_handwrite_runs(shown)
    n = int(round(ease_write_progress(progress, kind=ease) * len(atoms)))
    return runs_from_atoms(atoms[:n])


def runs_plain(runs: list[Run]) -> str:
    parts: list[str] = []
    for run in runs:
        if run[0] == "limop":
            parts.append(f"{run[1]}{{{run[2]}}}{{{run[3]}}}")
        else:
            parts.append(str(run[1]))
    return "".join(parts)


def _sym_scale(main_fs: float, text: str, *, style: HandwriteStyle) -> float:
    if str(text) in "ΣΠ":
        return main_fs * style.limop_symbol_scale
    return main_fs * float(getattr(style, "symbol_scale", 1.0))


def _run_font(
    kind: str,
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    fp_hand: fm.FontProperties | None,
    sym_text: str = "",
):
    if kind == "sym":
        return symbol_font(_sym_scale(main_fs, sym_text, style=style), style=style)
    if kind == "sub":
        sub_fs = main_fs * style.subscript_scale
        return hand_font(sub_fs, bold=False) if style.enabled else fm.FontProperties(size=sub_fs)
    if kind == "sup":
        sup_fs = main_fs * style.superscript_scale
        return hand_font(sup_fs, bold=False) if style.enabled else fm.FontProperties(size=sup_fs)
    if kind == "hand_b":
        return hand_font(main_fs, bold=True) if style.enabled else fm.FontProperties(size=main_fs, weight="bold")
    if fp_hand is not None:
        return fp_hand
    return hand_font(main_fs, bold=bold) if style.enabled else fm.FontProperties(size=main_fs, weight="bold" if bold else "normal")


def _run_fontsize(kind: str, main_fs: float, *, style: HandwriteStyle, sym_text: str = "") -> float:
    if kind == "sym":
        return _sym_scale(main_fs, sym_text, style=style)
    if kind == "sub":
        return main_fs * style.subscript_scale
    if kind == "sup":
        return main_fs * style.superscript_scale
    return main_fs


def run_size_px(
    renderer,
    kind: str,
    text: str,
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    fp_hand,
):
    sym_text = str(text) if kind == "sym" else ""
    fs = _run_fontsize(kind, main_fs, style=style, sym_text=sym_text)
    run_bold = bold or kind == "hand_b"
    fp = _run_font(kind, main_fs, style=style, bold=run_bold, fp_hand=fp_hand, sym_text=sym_text)
    w_px, h_px, d_px = renderer.get_text_width_height_descent(str(text), fp, ismath=False)
    return float(w_px), float(h_px + d_px)


def limop_size_px(renderer, op: str, below: str, above: str, main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    op_fs = _sym_scale(main_fs, op, style=style)
    op_w, op_h, op_d = _limop_sym_metrics(renderer, op, main_fs, style=style, bold=bold, fp_hand=fp_hand)
    below_fs = main_fs * style.limop_below_scale
    above_fs = main_fs * style.limop_above_scale
    below_w, below_h = (
        mixed_line_width_px(renderer, parse_handwrite_runs(below), below_fs, style=style, bold=False, fp_hand=fp_hand),
        mixed_line_height_px(renderer, parse_handwrite_runs(below), below_fs, style=style, bold=False, fp_hand=fp_hand),
    ) if below else (0.0, 0.0)
    above_w, above_h = (
        mixed_line_width_px(renderer, parse_handwrite_runs(above), above_fs, style=style, bold=False, fp_hand=fp_hand),
        mixed_line_height_px(renderer, parse_handwrite_runs(above), above_fs, style=style, bold=False, fp_hand=fp_hand),
    ) if above else (0.0, 0.0)
    w_px = max(op_w, below_w, above_w)
    gap = op_fs * style.limop_limit_gap_frac
    above_raise = (gap + op_fs * style.limop_above_gap_frac + above_h) if above else 0.0
    below_drop = (gap + op_fs * style.limop_below_gap_frac + below_h) if below else 0.0
    h_px = op_h + above_raise + below_drop
    return float(w_px), float(h_px)


def mixed_line_width_px(renderer, runs: list[Run], main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    total = 0.0
    for run in runs:
        if run[0] == "limop":
            w_px, _ = limop_size_px(renderer, run[1], run[2], run[3], main_fs, style=style, bold=bold, fp_hand=fp_hand)
            total += w_px
        else:
            w_px, _ = run_size_px(renderer, run[0], str(run[1]), main_fs, style=style, bold=bold, fp_hand=fp_hand)
            total += w_px
    return total


def mixed_line_height_px(renderer, runs: list[Run], main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    fp_main = _run_font("hand", main_fs, style=style, bold=bold, fp_hand=fp_hand)
    _, main_h_px, main_d_px = renderer.get_text_width_height_descent("x", fp_main, ismath=False)
    sub_ext_px = 0.0
    sup_ext_px = 0.0
    limop_ext_px = 0.0
    for run in runs:
        if run[0] == "limop":
            _, h_px = limop_size_px(renderer, run[1], run[2], run[3], main_fs, style=style, bold=bold, fp_hand=fp_hand)
            limop_ext_px = max(limop_ext_px, h_px - (main_h_px + main_d_px))
            continue
        kind, text = run[0], str(run[1])
        if kind == "sub":
            fp = _run_font(kind, main_fs, style=style, bold=bold, fp_hand=fp_hand)
            _, sub_h_px, sub_d_px = renderer.get_text_width_height_descent(str(text), fp, ismath=False)
            drop = main_d_px + main_fs * style.subscript_drop_frac * 0.85
            sub_ext_px = max(sub_ext_px, drop + sub_d_px)
        elif kind == "sup":
            fp = _run_font(kind, main_fs, style=style, bold=bold, fp_hand=fp_hand)
            _, sup_h_px, _ = renderer.get_text_width_height_descent(str(text), fp, ismath=False)
            raise_px = main_fs * style.superscript_raise_frac
            sup_ext_px = max(sup_ext_px, raise_px + sup_h_px)
    return float(main_h_px + main_d_px + sub_ext_px + sup_ext_px + limop_ext_px)


def _line_ascent_above_baseline_px(
    renderer,
    runs: list[Run],
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    fp_hand,
    fig,
) -> float:
    fp_main = _run_font("hand", main_fs, style=style, bold=bold, fp_hand=fp_hand)
    _, main_h_px, main_d_px = renderer.get_text_width_height_descent("x", fp_main, ismath=False)
    ascent_px = float(main_h_px - main_d_px)
    sup_raise_px = main_fs * style.superscript_raise_frac * (float(fig.dpi) / 72.0)
    for run in runs:
        if run[0] == "limop":
            op, _, above = run[1], run[2], run[3]
            op_w, op_h, op_d = _limop_sym_metrics(renderer, op, main_fs, style=style, bold=bold, fp_hand=fp_hand)
            op_top = float(op_h - op_d)
            if above:
                above_fs = main_fs * style.limop_above_scale
                above_h = mixed_line_height_px(
                    renderer, parse_handwrite_runs(above), above_fs, style=style, bold=False, fp_hand=fp_hand,
                )
                gap_above = _sym_scale(main_fs, op, style=style) * style.limop_above_gap_frac * (float(fig.dpi) / 72.0)
                op_top = op_top + gap_above + above_h
            ascent_px = max(ascent_px, op_top)
        elif run[0] == "sup":
            fp = _run_font("sup", main_fs, style=style, bold=bold, fp_hand=fp_hand)
            _, sup_h_px, _ = renderer.get_text_width_height_descent(str(run[1]), fp, ismath=False)
            ascent_px = max(ascent_px, sup_raise_px + sup_h_px)
    return ascent_px


def _equals_prefix_width_px(
    renderer,
    line: str,
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    fp_hand,
) -> float:
    """Pixel width from line start to the first ``=`` (for alignat-style rows)."""
    runs = parse_handwrite_runs(line)
    w = 0.0
    for run in runs:
        if run[0] == "limop":
            w_px, _ = limop_size_px(renderer, run[1], run[2], run[3], main_fs, style=style, bold=bold, fp_hand=fp_hand)
            w += w_px
            continue
        kind, text = run[0], str(run[1])
        if kind == "hand" and "=" in text:
            before, _, _ = text.partition("=")
            if before:
                w += mixed_line_width_px(
                    renderer, parse_handwrite_runs(before), main_fs, style=style, bold=bold, fp_hand=fp_hand,
                )
            return w
        w_px, _ = run_size_px(renderer, kind, text, main_fs, style=style, bold=bold, fp_hand=fp_hand)
        w += w_px
    return w


def _equals_anchor_x(
    ha: str,
    x: float,
    prev_eq_prefix_px: float,
    prev_line_width_px: float,
    cur_line_width_px: float,
    ax_w_px: float,
) -> float:
    """Return ``x`` anchor so a line starting with ``=`` aligns with a prior line's ``=``."""
    eq_axes = prev_eq_prefix_px / ax_w_px
    prev_span = prev_line_width_px / ax_w_px
    cur_half = (cur_line_width_px / ax_w_px) / 2.0
    if ha == "center":
        return x - prev_span / 2.0 + eq_axes + cur_half
    if ha == "right":
        return x - prev_span + eq_axes + cur_line_width_px / ax_w_px
    return x + eq_axes


def _limop_sym_metrics(renderer, op: str, main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    fp = _run_font("sym", main_fs, style=style, bold=bold, fp_hand=fp_hand, sym_text=op)
    w_px, h_px, d_px = renderer.get_text_width_height_descent(str(op), fp, ismath=False)
    return float(w_px), float(h_px), float(d_px)


def _draw_runs_at(
    target,
    x_cursor: float,
    baseline_y: float,
    runs: list[Run],
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    color: str,
    transform,
    ax_w_px: float,
    ax_h_px: float,
    fp_hand: fm.FontProperties | None,
    anchor_va: str = "baseline",
) -> float:
    fig = target.figure
    renderer = fig.canvas.get_renderer()
    fp_main = _run_font("hand", main_fs, style=style, bold=bold, fp_hand=fp_hand)
    _, main_h_px, main_d_px = renderer.get_text_width_height_descent("x", fp_main, ismath=False)
    if anchor_va == "bottom":
        baseline_y = baseline_y + main_d_px / ax_h_px
    elif anchor_va == "top":
        baseline_y = baseline_y - (main_h_px - main_d_px) / ax_h_px
    sub_drop_px = main_d_px + main_fs * style.subscript_drop_frac * (fig.dpi / 72.0)
    sup_raise_px = main_fs * style.superscript_raise_frac * (fig.dpi / 72.0)

    for run in runs:
        if not run:
            continue
        if run[0] == "limop":
            op, below, above = run[1], run[2], run[3]
            op_fs = _sym_scale(main_fs, op, style=style)
            op_w, op_h, op_d = _limop_sym_metrics(renderer, op, main_fs, style=style, bold=bold, fp_hand=fp_hand)
            below_fs = main_fs * style.limop_below_scale
            above_fs = main_fs * style.limop_above_scale
            below_w = mixed_line_width_px(renderer, parse_handwrite_runs(below), below_fs, style=style, bold=False, fp_hand=fp_hand) if below else 0.0
            above_w = mixed_line_width_px(renderer, parse_handwrite_runs(above), above_fs, style=style, bold=False, fp_hand=fp_hand) if above else 0.0
            below_h = mixed_line_height_px(renderer, parse_handwrite_runs(below), below_fs, style=style, bold=False, fp_hand=fp_hand) if below else 0.0
            block_w = max(op_w, below_w, above_w)
            op_x = x_cursor + (block_w - op_w) / (2.0 * ax_w_px)
            op_artist = target.text(
                op_x, baseline_y, op, transform=transform, va="baseline", ha="left",
                color=color,
                fontproperties=_run_font("sym", main_fs, style=style, bold=bold, fp_hand=fp_hand, sym_text=op),
                clip_on=False,
            )
            gap_above_px = op_fs * style.limop_above_gap_frac * (fig.dpi / 72.0)
            gap_below_px = op_fs * style.limop_below_gap_frac * (fig.dpi / 72.0)
            fig.canvas.draw()
            bb_op = op_artist.get_window_extent(renderer)
            inv = target.transAxes.inverted()
            sigma_top_y = float(inv.transform((bb_op.x0, bb_op.y1))[1])
            sigma_bot_y = float(inv.transform((bb_op.x0, bb_op.y0))[1])
            gap_above_axes = gap_above_px / ax_h_px
            gap_below_axes = gap_below_px / ax_h_px
            if above:
                above_runs = parse_handwrite_runs(above)
                above_y = sigma_top_y + gap_above_axes
                _draw_runs_at(
                    target, x_cursor + (block_w - above_w) / (2.0 * ax_w_px), above_y, above_runs, above_fs,
                    style=style, bold=False, color=color, transform=transform,
                    ax_w_px=ax_w_px, ax_h_px=ax_h_px, fp_hand=fp_hand, anchor_va="bottom",
                )
            if below:
                below_runs = parse_handwrite_runs(below)
                below_y = sigma_bot_y - gap_below_axes
                _draw_runs_at(
                    target, x_cursor + (block_w - below_w) / (2.0 * ax_w_px), below_y, below_runs, below_fs,
                    style=style, bold=False, color=color, transform=transform,
                    ax_w_px=ax_w_px, ax_h_px=ax_h_px, fp_hand=fp_hand, anchor_va="top",
                )
            x_cursor += block_w / ax_w_px
            continue
        kind, text = run[0], str(run[1])
        if not text:
            continue
        run_bold = bold or kind == "hand_b"
        fp = _run_font(kind, main_fs, style=style, bold=run_bold, fp_hand=fp_hand, sym_text=text if kind == "sym" else "")
        y_draw = baseline_y
        if kind == "sub":
            y_draw = baseline_y - sub_drop_px / ax_h_px
        elif kind == "sup":
            y_draw = baseline_y + sup_raise_px / ax_h_px
        target.text(
            x_cursor, y_draw, text, transform=transform, va="baseline", ha="left",
            color=color, fontproperties=fp, clip_on=False,
        )
        w_px, _ = run_size_px(renderer, kind, text, main_fs, style=style, bold=run_bold, fp_hand=fp_hand)
        x_cursor += w_px / ax_w_px
    return x_cursor


def draw_mixed_line(
    target,
    x: float,
    y: float,
    line: str,
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool = False,
    color: str,
    ha: str = "left",
    va: str = "top",
    transform=None,
    write_progress: float = 1.0,
    reveal_mode: str = "char",
    fp_hand: fm.FontProperties | None = None,
):
    """Draw one line with handwriting + lowered subscripts + math-symbol font."""
    fig = getattr(target, "figure", target)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    runs = parse_handwrite_runs(line)
    if write_progress < 1.0:
        runs = reveal_handwrite_runs(runs, write_progress, mode=reveal_mode, ease=style.ease)
    if not runs:
        return 0.0

    if transform is None:
        transform = getattr(target, "transAxes", fig.transFigure)

    if hasattr(target, "transAxes"):
        win = target.get_window_extent(renderer)
    else:
        win = fig.bbox
    ax_w_px = max(float(win.width), 1.0)
    ax_h_px = max(float(win.height), 1.0)

    fp_main = _run_font("hand", main_fs, style=style, bold=bold, fp_hand=fp_hand)
    _, main_h_px, main_d_px = renderer.get_text_width_height_descent("x", fp_main, ismath=False)
    ascent_px = _line_ascent_above_baseline_px(
        renderer, runs, main_fs, style=style, bold=bold, fp_hand=fp_hand, fig=fig,
    )
    # Caller passes a top anchor (va="top"); convert once to a shared baseline.
    baseline_y = y - ascent_px / ax_h_px
    dpi = float(fig.get_dpi())
    sub_drop_px = main_d_px + main_fs * style.subscript_drop_frac * (dpi / 72.0)

    total_w_px = mixed_line_width_px(renderer, runs, main_fs, style=style, bold=bold, fp_hand=fp_hand)
    x_cursor = x
    if ha == "center":
        x_cursor = x - (total_w_px / ax_w_px) / 2.0
    elif ha == "right":
        x_cursor = x - (total_w_px / ax_w_px)

    _draw_runs_at(
        target, x_cursor, baseline_y, runs, main_fs, style=style, bold=bold, color=color,
        transform=transform, ax_w_px=ax_w_px, ax_h_px=ax_h_px, fp_hand=fp_hand,
    )

    default_ascent = float(main_h_px - main_d_px)
    extra_top = max(0.0, ascent_px - default_ascent)
    return mixed_line_height_px(renderer, runs, main_fs, style=style, bold=bold, fp_hand=fp_hand) + extra_top


def cases_row_gap_px(
    fig,
    main_fs: float,
    *,
    style: HandwriteStyle,
    block: dict | None = None,
) -> float:
    """Vertical gap between rows inside a ``cases`` block (pt-based when set)."""
    gap_pt = None
    if block is not None:
        gap_pt = block.get("cases_row_gap_pt")
    if gap_pt is None:
        gap_pt = style.cases_row_gap_pt
    if gap_pt is not None:
        faux = {"pt_units": True}
        return block_pt_to_px(fig, faux, float(gap_pt))
    return main_fs * style.cases_row_gap_frac * (float(fig.dpi) / 72.0)


def _draw_cases_brace(
    target,
    x: float,
    y: float,
    brace_fs: float,
    *,
    style: HandwriteStyle,
    color: str,
    transform,
    write_progress: float = 1.0,
) -> None:
    """Draw a ``cases`` opening brace, honoring write/erase progress."""
    if write_progress <= 0.0:
        return
    txt = target.text(
        x, y, "{", transform=transform, va="center", ha="left",
        color=color, fontproperties=symbol_font(brace_fs, style=style), clip_on=False,
    )
    if write_progress >= 1.0 - 1e-9:
        return
    fig = target.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    prog = ease_write_progress(float(write_progress), kind=style.ease)
    bb = txt.get_window_extent(renderer)
    clip_w = max(float(bb.width) * float(prog), 0.0)
    if clip_w > 0.0:
        txt.set_clip_path(mpl.patches.Rectangle((bb.x0, bb.y0), clip_w, bb.height, transform=None))
    else:
        txt.set_visible(False)


def draw_cases_block(
    target,
    x: float,
    y_top: float,
    rows: list[str],
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    color: str,
    ha: str,
    transform,
    write_progress: float = 1.0,
    reveal_mode: str = "char",
    fp_hand: fm.FontProperties | None = None,
    row_gap_px: float | None = None,
) -> float:
    if not rows:
        return 0.0
    fig = target.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if hasattr(target, "transAxes"):
        win = target.get_window_extent(renderer)
    else:
        win = fig.bbox
    ax_w_px = max(float(win.width), 1.0)
    ax_h_px = max(float(win.height), 1.0)
    if row_gap_px is None:
        row_gap_px = main_fs * style.cases_row_gap_frac * (fig.dpi / 72.0)
    pad_px = main_fs * style.cases_pad_frac * (fig.dpi / 72.0)
    row_heights = [
        mixed_line_height_px(renderer, parse_handwrite_runs(r), main_fs, style=style, bold=bold, fp_hand=fp_hand)
        for r in rows
    ]
    row_widths = [
        mixed_line_width_px(renderer, parse_handwrite_runs(r), main_fs, style=style, bold=bold, fp_hand=fp_hand)
        for r in rows
    ]
    content_h = sum(row_heights) + row_gap_px * max(len(rows) - 1, 0)
    rows_only_h = sum(row_heights)
    x_cursor = x
    if ha == "center":
        est_brace = content_h * style.cases_brace_height_ratio / max(len(rows), 1)
        total_w = est_brace + pad_px + max(row_widths, default=0.0)
        x_cursor = x - (total_w / ax_w_px) / 2.0
    elif ha == "right":
        est_brace = content_h * style.cases_brace_height_ratio / max(len(rows), 1)
        total_w = est_brace + pad_px + max(row_widths, default=0.0)
        x_cursor = x - (total_w / ax_w_px)

    row_tops: list[float] = []
    y_row = y_top
    for rh in row_heights:
        row_tops.append(y_row)
        y_row -= (rh + row_gap_px) / ax_h_px
    block_bottom = row_tops[-1] - row_heights[-1] / ax_h_px
    block_span_px = rows_only_h
    brace_fs = max(main_fs * 2.25, block_span_px * style.cases_brace_height_ratio * (72.0 / fig.dpi))
    brace_w, brace_h = run_size_px(renderer, "sym", "{", brace_fs, style=style, bold=bold, fp_hand=fp_hand)
    brace_center_y = (y_top + block_bottom) / 2.0
    _draw_cases_brace(
        target, x_cursor, brace_center_y, brace_fs,
        style=style, color=color, transform=transform, write_progress=write_progress,
    )
    x_rows = x_cursor + (brace_w + pad_px) / ax_w_px
    prog_rows = line_write_progresses(len(rows), write_progress, overlap=style.reveal_overlap)
    for row, row_prog, y_rt in zip(rows, prog_rows, row_tops):
        draw_mixed_line(
            target, x_rows, y_rt, row, main_fs, style=style, bold=bold, color=color,
            ha="left", va="top", transform=transform, write_progress=row_prog,
            reveal_mode=reveal_mode, fp_hand=fp_hand,
        )
    return float(content_h)


def draw_lhs_cases_inline(
    target,
    x: float,
    y_top: float,
    lhs: str,
    rows: list[str],
    main_fs: float,
    *,
    style: HandwriteStyle,
    bold: bool,
    color: str,
    ha: str,
    transform,
    write_progress: float = 1.0,
    reveal_mode: str = "char",
    fp_hand: fm.FontProperties | None = None,
    row_gap_px: float | None = None,
) -> float:
    """``lhs = { … }`` on one row — brace starts inline after the equals sign."""
    if not lhs and not rows:
        return 0.0
    fig = target.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if hasattr(target, "transAxes"):
        win = target.get_window_extent(renderer)
    else:
        win = fig.bbox
    ax_w_px = max(float(win.width), 1.0)
    ax_h_px = max(float(win.height), 1.0)
    if row_gap_px is None:
        row_gap_px = main_fs * style.cases_row_gap_frac * (fig.dpi / 72.0)
    pad_px = main_fs * style.cases_pad_frac * (fig.dpi / 72.0)
    row_heights = [
        mixed_line_height_px(renderer, parse_handwrite_runs(r), main_fs, style=style, bold=bold, fp_hand=fp_hand)
        for r in rows
    ]
    row_widths = [
        mixed_line_width_px(renderer, parse_handwrite_runs(r), main_fs, style=style, bold=bold, fp_hand=fp_hand)
        for r in rows
    ]
    content_h = sum(row_heights) + row_gap_px * max(len(rows) - 1, 0)
    rows_only_h = sum(row_heights)
    lhs_h = (
        mixed_line_height_px(renderer, parse_handwrite_runs(lhs), main_fs, style=style, bold=bold, fp_hand=fp_hand)
        if lhs else 0.0
    )
    lhs_w = (
        mixed_line_width_px(renderer, parse_handwrite_runs(lhs), main_fs, style=style, bold=bold, fp_hand=fp_hand)
        if lhs else 0.0
    )
    block_h_px = max(content_h, lhs_h)
    x_cursor = x
    if ha == "center":
        est_brace = block_h_px * style.cases_brace_height_ratio / max(len(rows), 1)
        total_w = lhs_w + est_brace + pad_px + max(row_widths, default=0.0)
        x_cursor = x - (total_w / ax_w_px) / 2.0
    elif ha == "right":
        est_brace = block_h_px * style.cases_brace_height_ratio / max(len(rows), 1)
        total_w = lhs_w + est_brace + pad_px + max(row_widths, default=0.0)
        x_cursor = x - (total_w / ax_w_px)

    row_tops: list[float] = []
    y_cases_top = y_top - max(0.0, (block_h_px - content_h) / (2.0 * ax_h_px))
    y_row = y_cases_top
    for rh in row_heights:
        row_tops.append(y_row)
        y_row -= (rh + row_gap_px) / ax_h_px
    block_bottom = row_tops[-1] - row_heights[-1] / ax_h_px if row_tops else y_cases_top
    block_span_px = rows_only_h
    brace_fs = max(main_fs * 2.25, block_span_px * style.cases_brace_height_ratio * (72.0 / fig.dpi))
    brace_w, _ = run_size_px(renderer, "sym", "{", brace_fs, style=style, bold=bold, fp_hand=fp_hand)
    brace_center_y = (y_cases_top + block_bottom) / 2.0
    x_brace = x_cursor + lhs_w / ax_w_px
    if lhs:
        y_lhs = y_top - max(0.0, (block_h_px - lhs_h) / (2.0 * ax_h_px))
        draw_mixed_line(
            target, x_cursor, y_lhs, lhs, main_fs, style=style, bold=bold, color=color,
            ha="left", va="top", transform=transform, write_progress=write_progress,
            reveal_mode=reveal_mode, fp_hand=fp_hand,
        )
    _draw_cases_brace(
        target, x_brace, brace_center_y, brace_fs,
        style=style, color=color, transform=transform, write_progress=write_progress,
    )
    x_rows = x_brace + (brace_w + pad_px) / ax_w_px
    prog_rows = line_write_progresses(len(rows), write_progress, overlap=style.reveal_overlap)
    for row, row_prog, y_rt in zip(rows, prog_rows, row_tops):
        draw_mixed_line(
            target, x_rows, y_rt, row, main_fs, style=style, bold=bold, color=color,
            ha="left", va="top", transform=transform, write_progress=row_prog,
            reveal_mode=reveal_mode, fp_hand=fp_hand,
        )
    return float(block_h_px)


def bold_lhs_mathtext(lhs: str) -> str:
    lhs = str(lhs).strip()
    for cmd in ("mathrm", "text"):
        m = re.fullmatch(rf"\\{cmd}\{{(.+)\}}", lhs)
        if m:
            lhs = m.group(1).strip()
            break
    return rf"\mathbf{{{lhs}}}"


def bold_equation_lhs(text: str) -> str:
    s = str(text).strip()
    if not s or "=" not in s:
        return s
    inner = s[1:-1] if s.startswith("$") and s.endswith("$") else s
    lhs, rhs = inner.split("=", 1)
    return rf"${bold_lhs_mathtext(lhs)}={rhs.strip()}$"


def _tex_skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _tex_emit_hand_space(out: list[str], s: str, i: int) -> int:
    """Skip whitespace but keep a single hand space between prose tokens."""
    start = i
    while i < len(s) and s[i].isspace():
        i += 1
    if i > start and out:
        nxt = s[i] if i < len(s) else ""
        prev = str(out[-1])
        if nxt and nxt not in "_^" and not prev.endswith((" ", "\n", "_", "^", "(", "[", "{")):
            out.append(" ")
    return i


_TEX_CMDS = (
    "begin", "end", "boldsymbol", "mathrm", "mathbf", "mathcal", "displaystyle",
    "limits", "mathop", "frac", "prod", "sum", "sigma", "alpha", "partial",
    "nabla", "left", "right", "bigl", "bigr", "cdot", "quad", "qquad",
    "text", "log", "hat", "mid", "leftarrow", "in",
)


def _tex_read_cmd(s: str, i: int) -> tuple[str, int]:
    if i >= len(s) or s[i] != "\\":
        return "", i
    rest = s[i + 1 :]
    for name in _TEX_CMDS:
        if rest.startswith(name) and (len(rest) == len(name) or not rest[len(name)].isalpha()):
            return name, i + 1 + len(name)
    i += 1
    start = i
    while i < len(s) and s[i].isalpha():
        i += 1
    return s[start:i], i


def _tex_emit_limop(op: str, below: str, above: str) -> str:
    return f"{op}{{{below}}}{{{above}}}"


def _tex_emit_sub(content: str) -> str:
    if not content:
        return "_"
    return f"_{{{content}}}"


def _tex_emit_sup(content: str) -> str:
    if not content:
        return "^"
    return f"^{{{content}}}"


def _tex_read_script(s: str, i: int) -> tuple[str, int]:
    """Read a TeX sub/superscript value (raw, then converted)."""
    if i >= len(s):
        return "", i
    if s[i] == "{":
        raw, i = _read_braced_raw(s, i)
        return _tex_convert(raw), i
    start = i
    if i < len(s) and s[i].isalpha():
        while i < len(s) and s[i].isalpha():
            i += 1
        return s[start:i], i
    while i < len(s) and (s[i].isalnum() or s[i] in ",=+-."):
        i += 1
    return s[start:i], i


def _tex_normalize(s: str) -> str:
    s = re.sub(r"\\mathop\s*\{\\sum\}", r"\\sum", s)
    s = re.sub(r"\\mathop\s*\{\\prod\}", r"\\prod", s)
    s = re.sub(r"\\limits\s*", "", s)
    return s


def _tex_convert_cases(body: str) -> str:
    body = re.sub(r"\\\[[^\]]*\]", r"\\\\", body)
    rows: list[str] = []
    for chunk in re.split(r"(?<!\\)\\\\", body):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "&" in chunk:
            lhs, rhs = chunk.split("&", 1)
            rows.append(f"{_tex_convert(lhs.strip())}  {_tex_convert(rhs.strip())}")
        else:
            rows.append(_tex_convert(chunk))
    if not rows:
        return ""
    return CASES_OPEN + "\n" + "\n".join(rows) + "\n" + CASES_CLOSE


def _tex_convert_align(body: str) -> str:
    lines: list[str] = []
    align_col = 0
    for chunk in re.split(r"(?<!\\)\\\\", body):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "&" in chunk:
            parts = [p.strip() for p in chunk.split("&")]
            if align_col == 0 and parts:
                align_col = len(_tex_convert(parts[0]))
            pad = max(align_col - len(_tex_convert(parts[0])), 0) if parts else 0
            conv = [_tex_convert(p) for p in parts]
            if len(conv) > 1:
                conv[0] = " " * pad + conv[0]
            lines.append(" ".join(conv))
        else:
            lines.append(_tex_convert(chunk))
    return "\n".join(lines)


def _tex_convert(s: str) -> str:
    s = _tex_normalize(s)
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i].isspace():
            i = _tex_emit_hand_space(out, s, i)
            continue
        if i >= n:
            break
        ch = s[i]
        if ch == "\\":
            if i + 1 < n and s[i + 1] in ",;":
                if s[i + 1] == ",":
                    out.append(" " if out and str(out[-1]).endswith(",") else ", ")
                else:
                    out.append(" ")
                i += 2
                continue
            cmd, i = _tex_read_cmd(s, i)
            if cmd in ("sum", "prod"):
                op = "Σ" if cmd == "sum" else "Π"
                i = _tex_skip_ws(s, i)
                if s.startswith("\\limits", i):
                    i += len("\\limits")
                    i = _tex_skip_ws(s, i)
                below, above = "", ""
                if i < n and s[i] == "_":
                    i += 1
                    below, i = _tex_read_script(s, i)
                if i < n and s[i] == "^":
                    i += 1
                    above, i = _tex_read_script(s, i)
                if above or (below and "=" in below):
                    out.append(_tex_emit_limop(op, below, above))
                elif below:
                    # index-only: Σ_i (subscript), not a limit under the operator
                    out.append(op + _tex_emit_sub(below))
                else:
                    out.append(op)
            elif cmd == "frac":
                i = _tex_skip_ws(s, i)
                num, i = _read_braced(s, i)
                i = _tex_skip_ws(s, i)
                den, i = _read_braced(s, i)
                out.append(f"{_tex_convert(num)}/{_tex_convert(den)}")
            elif cmd in ("mathcal",):
                i = _tex_skip_ws(s, i)
                inner, i = _read_braced(s, i)
                if inner.strip() == "L":
                    out.append("ℒ")
                else:
                    out.append(_tex_convert(inner))
            elif cmd in ("text", "mathrm"):
                i = _tex_skip_ws(s, i)
                inner, i = _read_braced_raw(s, i)
                out.append(inner)
            elif cmd in ("mathbf", "boldsymbol"):
                i = _tex_skip_ws(s, i)
                inner, i = _read_braced(s, i)
                out.append(BOLD_OPEN + _tex_convert(inner) + BOLD_CLOSE)
            elif cmd == "sigma":
                out.append("σ")
            elif cmd == "alpha":
                out.append("α")
            elif cmd == "Delta":
                out.append("Δ")
            elif cmd == "delta":
                out.append("δ")
            elif cmd == "log":
                if out and not str(out[-1]).endswith((" ", "\n")):
                    out.append(" ")
                out.append("log ")
            elif cmd in ("partial", "nabla", "mid", "cdot", "in", "leftarrow"):
                repl = {
                    "partial": "∂", "nabla": "∇", "mid": " | ",
                    "cdot": "·", "in": "∈", "leftarrow": "←",
                }
                out.append(repl.get(cmd, ""))
            elif cmd in ("left", "right", "bigl", "bigr", "limits", "mathop", "!", "displaystyle", "quad", "qquad"):
                if cmd in ("quad", "qquad"):
                    out.append("  " if cmd == "quad" else "    ")
            elif cmd == "begin":
                i = _tex_skip_ws(s, i)
                env, i = _read_braced(s, i)
                env_name = env.replace("*", "").strip()
                end_m = re.search(rf"\\end\{{{re.escape(env_name)}\*?\}}", s[i:])
                if end_m:
                    inner = s[i:i + end_m.start()]
                    i = i + end_m.end()
                else:
                    inner = ""
                if env_name == "cases":
                    out.append(_tex_convert_cases(inner))
                elif env_name in ("align", "alignat", "aligned", "eqnarray"):
                    out.append(_tex_convert_align(inner))
                else:
                    out.append(_tex_convert(inner))
            elif cmd == "end":
                i = _tex_skip_ws(s, i)
                _, i = _read_braced(s, i)
            elif cmd == "hat":
                i = _tex_skip_ws(s, i)
                if i < n and s[i] == "p":
                    out.append("p\u0302")
                    i += 1
                    if i < n and s[i] == "_":
                        i += 1
                        sub, i = _tex_read_script(s, i)
                        out.append(_tex_emit_sub(sub))
                else:
                    inner, i = _read_braced(s, i)
                    out.append(_tex_convert(inner))
            else:
                out.append(cmd)
        elif ch == "{":
            inner, i = _read_braced(s, i)
            out.append(_tex_convert(inner))
        elif ch == "^":
            i += 1
            sup, i = _tex_read_script(s, i)
            out.append(_tex_emit_sup(sup))
        elif ch == "_":
            i += 1
            sub, i = _tex_read_script(s, i)
            out.append(_tex_emit_sub(sub))
        elif ch in "[]()":
            out.append(ch)
            i += 1
        elif ch == "+":
            out.append(" + ")
            i += 1
        else:
            out.append(ch)
            i += 1
    text = "".join(out)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = text.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    return text.strip()


def latex_line_to_handwrite(line: str) -> str:
    s = str(line).strip()
    if not s:
        return s
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return _tex_convert(s)


def ease_write_progress(progress: float, *, kind: str = "out_cubic") -> float:
    t = float(np.clip(float(progress), 0.0, 1.0))
    if kind == "linear":
        return t
    if kind == "out_quad":
        return 1.0 - (1.0 - t) ** 2
    return 1.0 - (1.0 - t) ** 3


def typewriter_reveal(text: str, progress: float, *, mode: str = "char", ease: str = "out_cubic") -> str:
    progress = ease_write_progress(progress, kind=ease)
    s = str(text)
    if not s:
        return s
    if mode == "word":
        parts = re.split(r"(\s+)", s)
        words = [p for p in parts if p and not p.isspace()]
        if not words:
            return s if progress >= 1.0 else ""
        n_words = int(round(progress * len(words)))
        out, seen = [], 0
        for part in parts:
            if part.isspace():
                if seen > 0 and seen <= n_words:
                    out.append(part)
            elif seen < n_words:
                out.append(part)
                seen += 1
        return "".join(out)
    return s[: int(round(progress * len(s)))]


def line_write_progresses(n_lines: int, progress: float, *, overlap: float = 0.35) -> list[float]:
    n_lines = int(n_lines)
    if n_lines <= 0:
        return []
    progress = float(np.clip(float(progress), 0.0, 1.0))
    if n_lines == 1:
        return [progress]
    step = 1.0 / (n_lines - overlap * (n_lines - 1))
    out = []
    for i in range(n_lines):
        local = (progress - i * step * (1.0 - overlap)) / step
        out.append(float(np.clip(local, 0.0, 1.0)))
    return out


def text_uses_mathtext(line: str, *, style: HandwriteStyle) -> bool:
    return (not style.enabled) and ("$" in str(line))


def display_line(line: str, *, style: HandwriteStyle, bold_lhs: bool = False) -> str:
    if style.enabled:
        return latex_line_to_handwrite(line)
    if bold_lhs:
        return bold_equation_lhs(line)
    return str(line).strip()


def block_mathtext_lines(block: dict) -> list[str]:
    raw = block.get("mathtext_lines") or []
    return [str(ln).strip() for ln in raw if str(ln).strip()]


def block_mathtext_usetex(block: dict) -> bool:
    return bool(block.get("mathtext_usetex", False))


def block_pt_to_px(fig, block: dict, pt: float) -> float:
    """Typographic pt → axis pixels when ``block['pt_units']`` is set."""
    if block.get("pt_units"):
        return float(pt) * float(fig.dpi) / 72.0
    return float(pt)


def block_n_lines(block: dict, *, style: HandwriteStyle) -> int:
    return len(block_display_lines(block, style=style)) + len(block_mathtext_lines(block))


def mathtext_line_width_px(renderer, line: str, fontsize: float, *, usetex: bool = False) -> float:
    if usetex:
        ensure_amsmath_preamble()
    fp = fm.FontProperties(size=float(fontsize))
    ismath = "TeX" if usetex else True
    w_px, _, _ = renderer.get_text_width_height_descent(str(line), fp, ismath=ismath)
    return float(w_px)


def fit_mathtext_fontsize(
    renderer,
    line: str,
    max_w_px: float,
    start_fs: float,
    *,
    usetex: bool = False,
    min_fs: float = 9.0,
) -> float:
    fs = float(start_fs)
    while fs > float(min_fs) and mathtext_line_width_px(renderer, line, fs, usetex=usetex) > float(max_w_px):
        fs -= 0.5
    return fs


def fit_mathtext_block_fontsize(
    renderer,
    lines: list[str],
    max_w_px: float,
    max_h_px: float,
    start_fs: float,
    *,
    usetex: bool = False,
    line_dy_pt: float = 0.0,
    min_fs: float = 9.0,
) -> float:
    """Shrink one shared fontsize until all lines fit width and total height."""
    fs = float(start_fs)
    dy = float(line_dy_pt)
    while fs > float(min_fs):
        for line in lines:
            if mathtext_line_width_px(renderer, line, fs, usetex=usetex) > float(max_w_px):
                fs -= 0.5
                break
        else:
            total_h = 0.0
            for j, line in enumerate(lines):
                total_h += mathtext_line_height_px(renderer, line, fs, usetex=usetex)
                if j < len(lines) - 1:
                    total_h += dy
            if total_h <= float(max_h_px):
                return fs
            fs -= 0.5
    return max(float(min_fs), fs)


def mathtext_line_height_px(renderer, line: str, fontsize: float, *, usetex: bool = False) -> float:
    if usetex:
        ensure_amsmath_preamble()
    fp = fm.FontProperties(size=float(fontsize))
    ismath = "TeX" if usetex else True
    _, h_px, d_px = renderer.get_text_width_height_descent(str(line), fp, ismath=ismath)
    return float(h_px + d_px)


def draw_mathtext_line(
    target,
    x: float,
    y: float,
    line: str,
    fontsize: float,
    *,
    color: str,
    ha: str = "left",
    write_progress: float = 1.0,
    transform=None,
    usetex: bool = False,
):
    if write_progress <= 0.0 or not str(line).strip():
        return 0.0
    if usetex:
        ensure_amsmath_preamble()
    fig = getattr(target, "figure", target)
    if transform is None:
        transform = getattr(target, "transAxes", fig.transFigure)
    prog = 1.0 if float(write_progress) >= 1.0 else ease_write_progress(float(write_progress))
    txt = target.text(
        x, y, str(line), transform=transform, fontsize=float(fontsize), color=color,
        ha=ha, va="top", alpha=1.0, clip_on=False, usetex=usetex, zorder=100,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if float(write_progress) < 1.0 - 1e-9:
        bb = txt.get_window_extent(renderer)
        clip_w = max(float(bb.width) * float(prog), 0.0)
        if clip_w > 0.0:
            txt.set_clip_path(mpl.patches.Rectangle((bb.x0, bb.y0), clip_w, bb.height, transform=None))
        else:
            txt.set_visible(False)
    return mathtext_line_height_px(renderer, line, fontsize, usetex=usetex)


def block_display_lines(block: dict, *, style: HandwriteStyle) -> list[str]:
    bold = bool(block.get("bold_lhs", False))
    raw = [ln for ln in str(block.get("text", "")).split("\n") if ln.strip()]
    out: list[str] = []
    for ln in raw:
        if style.enabled:
            converted = display_line(ln, style=style, bold_lhs=bold)
            for sub in converted.split("\n"):
                sub = sub.strip()
                if sub:
                    out.append(sub)
        elif bold:
            out.append(bold_equation_lhs(ln))
        else:
            out.append(str(ln).strip())
    return out


def label_display(label, *, style: HandwriteStyle) -> str:
    if label is None:
        return ""
    s = str(label)
    return latex_line_to_handwrite(s) if style.enabled else s


def block_display_text(block: dict, *, style: HandwriteStyle) -> str:
    return "\n".join(block_display_lines(block, style=style))


def text_kw(style: HandwriteStyle, size: float, *, bold: bool = False) -> dict[str, Any]:
    if style.enabled:
        return {"fontproperties": hand_font(size, bold=bold)}
    kw: dict[str, Any] = {"fontsize": size}
    if bold:
        kw["fontweight"] = "bold"
    return kw


def block_write_progress(blocks: list[dict], progress: float, *, style: HandwriteStyle) -> dict[int, dict]:
    """Map global write progress -> per-block label/line reveal progress."""
    slots = []
    for bi, block in enumerate(blocks):
        if block.get("label"):
            slots.append(("label", bi))
        for li in range(block_n_lines(block, style=style)):
            slots.append(("line", bi, li))
    if not slots:
        return {}
    progresses = line_write_progresses(len(slots), progress, overlap=style.reveal_overlap)
    out: dict[int, dict] = {}
    for (kind, bi, *rest), prog in zip(slots, progresses):
        out.setdefault(bi, {})
        if kind == "label":
            out[bi]["__label__"] = prog
        else:
            out[bi][int(rest[0])] = prog
    return out


def stagger_groups_progress(
    block_groups: list[list[dict]], progress: float, *, style: HandwriteStyle
) -> list[dict[int, dict]]:
    """Stagger reveal across multiple block groups (e.g. right rail + bottom columns)."""
    slots = []
    for gi, blocks in enumerate(block_groups):
        for bi, block in enumerate(blocks):
            if block.get("label"):
                slots.append((gi, bi, "label", None))
            for li in range(block_n_lines(block, style=style)):
                slots.append((gi, bi, "line", li))
    if not slots:
        return [{} for _ in block_groups]
    progresses = line_write_progresses(len(slots), progress, overlap=style.reveal_overlap)
    out: list[dict[int, dict]] = [{} for _ in block_groups]
    for (gi, bi, kind, li), prog in zip(slots, progresses):
        out[gi].setdefault(bi, {})
        if kind == "label":
            out[gi][bi]["__label__"] = prog
        else:
            out[gi][bi][int(li)] = prog
    return out


def text_line_height_px(renderer, text, fontsize, *, fontprop=None, style: HandwriteStyle | None = None):
    style = style or HandwriteStyle()
    prop = fontprop if fontprop is not None else fm.FontProperties(size=float(fontsize))
    ismath = text_uses_mathtext(text, style=style)
    _, h_px, d_px = renderer.get_text_width_height_descent(str(text), prop, ismath=ismath)
    return float(h_px + d_px)


def text_width_px(renderer, text, fontsize, *, bold=False, is_math=None, fontprop=None, style=None):
    style = style or HandwriteStyle()
    if is_math is None:
        is_math = "$" in str(text)
    prop = fontprop if fontprop is not None else fm.FontProperties(
        size=float(fontsize), weight="bold" if bold else "normal"
    )
    w_px, _, _ = renderer.get_text_width_height_descent(str(text), prop, ismath=bool(is_math))
    return float(w_px)


def wrap_plain_text_px(renderer, text, fontsize, max_w_px, *, bold=False, style=None):
    style = style or HandwriteStyle()
    s = str(text)
    if text_width_px(renderer, s, fontsize, bold=bold, is_math=False, style=style) <= max_w_px:
        return [s]
    words = s.split()
    if not words:
        return []
    lines, cur = [], []
    for word in words:
        trial = " ".join(cur + [word])
        if text_width_px(renderer, trial, fontsize, bold=bold, is_math=False, style=style) <= max_w_px or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def wrap_text_for_width(fig, text, fontsize, max_width_frac, *, bold=False, style=None):
    style = style or HandwriteStyle()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_w_px = float(max_width_frac) * float(fig.bbox.width)
    s = str(text)
    if "$" not in s:
        return wrap_plain_text_px(renderer, s, fontsize, max_w_px, bold=bold, style=style)
    if text_width_px(renderer, s, fontsize, bold=bold, is_math=True, style=style) <= max_w_px:
        return [s]
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    return parts if parts else [s]


def section_title_lines(fig, title, max_width_frac, *, style: HandwriteStyle, title_fs: float | None = None):
    fs = float(style.section_title_fs if title_fs is None else title_fs)
    return wrap_text_for_width(fig, title, fs, max_width_frac, bold=style.section_title_bold, style=style)


def section_title_band_frac(fig, title, max_width_frac, *, style: HandwriteStyle, gap_frac: float, title_fs: float | None = None):
    fs = float(style.section_title_fs if title_fs is None else title_fs)
    lines = section_title_lines(fig, title, max_width_frac, style=style, title_fs=fs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    total_px = 0.0
    for i, line in enumerate(lines):
        total_px += text_line_height_px(renderer, line, fs, style=style)
        if i < len(lines) - 1:
            total_px += float(style.title_line_dy_pt)
    total_px += gap_frac * float(fig.bbox.height)
    return total_px / float(fig.bbox.height)


def fit_plain_fontsize(
    renderer,
    text,
    max_w_px: float,
    start_fs: float,
    *,
    min_fs: float = 18.0,
    bold: bool = False,
    style: HandwriteStyle | None = None,
) -> float:
    style = style or HandwriteStyle()
    fs = float(start_fs)
    while fs > float(min_fs) and text_width_px(
        renderer, str(text), fs, bold=bold, is_math=False, style=style,
    ) > float(max_w_px):
        fs -= 0.5
    return fs


def draw_section_title(
    fig,
    xy,
    title,
    *,
    style: HandwriteStyle,
    ha="left",
    va="top",
    max_width_frac=None,
    write_progress=1.0,
    pad_frac: float = 0.010,
    title_fs: float | None = None,
    title_color: str | None = None,
    single_line: bool = False,
):
    fs = float(style.section_title_fs if title_fs is None else title_fs)
    color = title_color if title_color is not None else style.title_color
    if max_width_frac is None:
        max_width_frac = max(1.0 - float(xy[0]) - pad_frac, 0.05)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_w_px = float(max_width_frac) * float(fig.bbox.width)
    if single_line:
        fs = fit_plain_fontsize(
            renderer, title, max_w_px, fs, bold=style.section_title_bold, style=style,
        )
        lines = [str(title)]
    else:
        lines = section_title_lines(fig, title, max_width_frac, style=style, title_fs=fs)
    x, y = xy
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y_cursor = float(y)
    line_step = (
        text_line_height_px(renderer, "Ay", fs, style=style)
        + float(style.title_line_dy_pt)
    ) / float(fig.bbox.height)
    for line in lines:
        draw_mixed_line(
            fig,
            x,
            y_cursor,
            line,
            fs,
            style=style,
            bold=style.section_title_bold,
            color=color,
            ha=ha,
            va=va,
            transform=fig.transFigure,
            write_progress=write_progress,
            reveal_mode=style.title_mode,
        )
        y_cursor -= line_step if va == "top" else -line_step


def draw_block_cell(
    ax,
    block: dict,
    *,
    style: HandwriteStyle,
    block_fs: float,
    label_fs: float,
    align: str = "center",
    line_progress: dict | None = None,
    label_color: str | None = None,
    text_color: str | None = None,
    accent_color: str | None = None,
    show_frame: bool = True,
    text_x_frac: float | None = None,
    text_y_inset_pt: float = 0.0,
):
    if block.get("nll_colorbar"):
        from ch4_layout import ch4_draw_nll_colorbar_cell

        ch4_draw_nll_colorbar_cell(
            ax,
            block,
            style=style,
            block_fs=block_fs,
            text_color=text_color,
        )
        return

    if block.get("handwrite_matrix"):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)
        from latex_to_handwrite import draw_handwrite_matrix_in_cell

        draw_handwrite_matrix_in_cell(
            ax,
            block,
            style=style,
            block_fs=block_fs,
            align=align,
            line_progress=line_progress,
            text_color=text_color,
            text_x_frac=text_x_frac,
            text_y_inset_pt=text_y_inset_pt,
        )
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("none")
    if block_mathtext_usetex(block):
        ax.set_clip_on(False)
    if show_frame and style.frame_edge and style.frame_edge.lower() not in ("none", "transparent"):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(style.frame_edge)
            spine.set_linewidth(style.frame_lw)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_h_px = max(float(ax.get_window_extent(renderer).height), 1.0)
    ax_w_px = max(float(ax.get_window_extent(renderer).width), 1.0)
    body_fs = float(block.get("block_fs", block_fs))
    label_fs_eff = float(block.get("label_fs", label_fs))
    ha = "left" if align == "left" else "center"
    x = float(text_x_frac if text_x_frac is not None else (0.07 if align == "left" else 0.5))
    max_frac = float(block.get(
        "mathtext_max_line_frac",
        0.94 if ha == "left" else 0.88,
    ))
    max_line_px = ax_w_px * max_frac
    max_line_frac = max_line_px / float(fig.bbox.width)
    fp_block = hand_font(body_fs, bold=bool(block.get("bold_lhs", False))) if style.enabled else None
    fp_label = hand_font(label_fs_eff) if style.enabled else None

    if block.get("fit_to_column_width"):
        min_fs = float(block.get("block_fs_min", 12.0))
        preview_lines = block_display_lines(block, style=style)
        bold_lhs = bool(block.get("bold_lhs", False))
        while body_fs > min_fs and preview_lines:
            fp_test = hand_font(body_fs, bold=bold_lhs) if style.enabled else None
            widest = 0.0
            for raw in preview_lines:
                runs = parse_handwrite_runs(raw)
                widest = max(
                    widest,
                    mixed_line_width_px(
                        renderer, runs, body_fs, style=style, bold=bold_lhs, fp_hand=fp_test,
                    ),
                )
            if widest <= max_line_px:
                break
            body_fs -= 0.5
        if style.enabled:
            fp_block = hand_font(body_fs, bold=bold_lhs)

    line_progress = line_progress or {}
    lab_color = label_color if label_color is not None else style.label_color
    body_color = text_color if text_color is not None else style.text_color
    if block.get("text_color"):
        body_color = str(block["text_color"])

    def _pt(val):
        return block_pt_to_px(fig, block, val)

    if block.get("text_x_shift_pt") is not None:
        x = (x * ax_w_px + _pt(float(block["text_x_shift_pt"]))) / ax_w_px

    line_dy = _pt(block.get("line_dy_pt", style.line_dy_pt))
    label_gap = _pt(block.get("label_gap_pt", style.label_gap_pt))
    y_px = _pt(block.get("top_pad_pt", style.top_pad_pt)) + _pt(text_y_inset_pt)
    if block.get("text_y_shift_pt") is not None:
        y_px += _pt(float(block["text_y_shift_pt"]))

    def _y_axes():
        return 1.0 - (y_px / ax_h_px)

    def _h(text, fs, fp=None):
        return text_line_height_px(renderer, text or "A", fs, fontprop=fp, style=style)

    label = block.get("label")
    if label:
        lab = label_display(label, style=style)
        lp = float(line_progress.get("__label__", 1.0))
        if lab:
            h_px = draw_mixed_line(
                ax, x, _y_axes(), lab, label_fs_eff, style=style, bold=False,
                color=lab_color, ha=ha, write_progress=lp,
                reveal_mode=style.title_mode, fp_hand=fp_label,
            )
            y_px += h_px if h_px else _h(lab, label_fs_eff, fp_label)
        y_px += label_gap

    def _block_line_val(key: str, line_idx: int, default):
        raw = block.get(key) or {}
        if line_idx in raw:
            return raw[line_idx]
        if str(line_idx) in raw:
            return raw[str(line_idx)]
        return default

    def _line_fs(line_idx: int) -> float:
        return float(_block_line_val("line_body_fs", line_idx, body_fs))

    def _line_style(line_idx: int) -> HandwriteStyle:
        ss = _block_line_val("line_subscript_scale", line_idx, None)
        gap_pt = block.get("cases_row_gap_pt")
        if ss is not None or gap_pt is not None:
            kw = {}
            if ss is not None:
                kw["subscript_scale"] = float(ss)
            if gap_pt is not None:
                kw["cases_row_gap_pt"] = float(gap_pt)
            return replace(style, **kw)
        return style

    def _line_extra_dy(line_idx: int) -> float:
        return _pt(float(_block_line_val("line_extra_dy_pt", line_idx, 0)))

    def _line_row_y_px(line_idx: int) -> float | None:
        raw = block.get("line_row_y_pt") or {}
        if line_idx in raw:
            return _pt(float(raw[line_idx]))
        if str(line_idx) in raw:
            return _pt(float(raw[str(line_idx)]))
        return None

    def _block_content_top_px() -> float:
        base = _pt(block.get("top_pad_pt", style.top_pad_pt)) + _pt(text_y_inset_pt)
        if block.get("text_y_shift_pt") is not None:
            base += _pt(float(block["text_y_shift_pt"]))
        return base

    use_abs_rows = bool(block.get("line_row_y_pt"))

    def _y_axes_line(line_idx: int = 0):
        abs_y = _line_row_y_px(line_idx)
        if abs_y is not None:
            inset = float(_block_line_val("line_y_inset_pt", line_idx, 0))
            return 1.0 - ((_block_content_top_px() + abs_y + _pt(inset)) / ax_h_px)
        inset = float(_block_line_val("line_y_inset_pt", line_idx, 0))
        return 1.0 - ((y_px + _pt(inset)) / ax_h_px)

    def _advance_line_y(line_idx: int, h_px: float) -> None:
        nonlocal y_px
        if use_abs_rows and _line_row_y_px(line_idx) is not None:
            return
        y_px += h_px + line_dy + _line_extra_dy(line_idx)

    def _fp_line(line_idx: int, *, bold: bool) -> fm.FontProperties | None:
        if style.enabled:
            return hand_font(_line_fs(line_idx), bold=bold)
        return None

    raw_lines = block_display_lines(block, style=style)
    lines = []
    for raw in raw_lines:
        runs = parse_handwrite_runs(raw)
        w_px = mixed_line_width_px(renderer, runs, body_fs, style=style, bold=bool(block.get("bold_lhs", False)), fp_hand=fp_block)
        if w_px <= max_line_px or (style.enabled and not bool(block.get("force_wrap", False))):
            lines.append(raw)
        else:
            lines.extend(wrap_text_for_width(fig, raw, body_fs, max_line_frac, bold=False, style=style))

    line_idx = 0
    i = 0
    prev_align_line: str | None = None
    while i < len(lines):
        stripped = lines[i].strip()
        lhs_inline: str | None = None
        if CASES_OPEN in stripped and stripped != CASES_OPEN:
            lhs_inline, _ = stripped.split(CASES_OPEN, 1)
            lhs_inline = lhs_inline.strip() or None
            i += 1
        elif (
            stripped
            and stripped != CASES_OPEN
            and i + 1 < len(lines)
            and lines[i + 1].strip() == CASES_OPEN
        ):
            lhs_inline = stripped
            i += 2
        elif stripped == CASES_OPEN:
            i += 1
        else:
            lhs_inline = None

        if lhs_inline is not None or (stripped == CASES_OPEN and i > 0):
            case_rows: list[str] = []
            while i < len(lines) and lines[i].strip() != CASES_CLOSE:
                case_rows.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            prog = float(line_progress.get(line_idx, line_progress.get(str(line_idx), 1.0)))
            li_fs = _line_fs(line_idx)
            li_style = _line_style(line_idx)
            cases_gap = block.get("cases_row_gap_frac")
            if cases_gap is not None:
                li_style = replace(li_style, cases_row_gap_frac=float(cases_gap))
            gap_px = cases_row_gap_px(fig, li_fs, style=li_style, block=block)
            li_fp = _fp_line(line_idx, bold=bool(block.get("bold_lhs", False)))
            if lhs_inline is not None:
                h_px = draw_lhs_cases_inline(
                    ax, x, _y_axes_line(line_idx), lhs_inline, case_rows, li_fs, style=li_style,
                    bold=bool(block.get("bold_lhs", False)),
                    color=body_color,
                    ha=ha, transform=ax.transAxes, write_progress=prog,
                    reveal_mode=style.line_mode, fp_hand=li_fp, row_gap_px=gap_px,
                )
            else:
                h_px = draw_cases_block(
                    ax, x, _y_axes_line(line_idx), case_rows, li_fs, style=li_style,
                    bold=bool(block.get("bold_lhs", False)),
                    color=body_color,
                    ha=ha, transform=ax.transAxes, write_progress=prog,
                    reveal_mode=style.line_mode, fp_hand=li_fp, row_gap_px=gap_px,
                )
            if h_px > 0:
                _advance_line_y(line_idx, h_px)
            line_idx += 1
            prev_align_line = None
            continue

        if stripped == CASES_OPEN:
            continue

        prog = float(line_progress.get(line_idx, line_progress.get(str(line_idx), 1.0)))
        line_text = lines[i].strip()
        li_fs = _line_fs(line_idx)
        li_style = _line_style(line_idx)
        li_fp = _fp_line(line_idx, bold=bool(block.get("bold_lhs", False)))
        line_color = body_color
        raw_ltc = block.get("line_text_colors")
        if isinstance(raw_ltc, (list, tuple)) and line_idx < len(raw_ltc) and raw_ltc[line_idx]:
            line_color = str(raw_ltc[line_idx])
        else:
            ltc = _block_line_val("line_text_colors", line_idx, None)
            if ltc is not None:
                line_color = str(ltc)
        line_x = x
        if (
            prev_align_line is not None
            and "=" in prev_align_line
            and line_text.startswith("=")
        ):
            eq_x_px = _equals_prefix_width_px(
                renderer, prev_align_line, li_fs, style=li_style,
                bold=bool(block.get("bold_lhs", False)), fp_hand=li_fp,
            )
            prev_w_px = mixed_line_width_px(
                renderer, parse_handwrite_runs(prev_align_line), li_fs, style=li_style,
                bold=bool(block.get("bold_lhs", False)), fp_hand=li_fp,
            )
            cur_w_px = mixed_line_width_px(
                renderer, parse_handwrite_runs(line_text), li_fs, style=li_style,
                bold=bool(block.get("bold_lhs", False)), fp_hand=li_fp,
            )
            line_x = _equals_anchor_x(ha, x, eq_x_px, prev_w_px, cur_w_px, ax_w_px)
        h_px = draw_mixed_line(
            ax, line_x, _y_axes_line(line_idx), line_text, li_fs, style=li_style,
            bold=bool(block.get("bold_lhs", False)), color=line_color,
            ha=ha, write_progress=prog, reveal_mode=style.line_mode, fp_hand=li_fp,
        )
        if not line_text.startswith("="):
            prev_align_line = line_text
        i += 1
        line_idx += 1
        if h_px <= 0:
            continue
        _advance_line_y(line_idx - 1, h_px)

    n_hand = len(lines)
    m_usetex = block_mathtext_usetex(block)
    m_fs_raw = block.get("mathtext_fs", body_fs)
    if isinstance(m_fs_raw, (list, tuple)):
        m_fs_list = list(m_fs_raw)
    else:
        m_fs_list = None
    mlines = block_mathtext_lines(block)
    m_line_y_inset = block.get("mathtext_line_y_inset_pt")
    if isinstance(m_line_y_inset, (list, tuple)):
        m_line_y_inset = list(m_line_y_inset)
    else:
        m_line_y_inset = None
    if mlines and m_fs_list is None:
        start_fs = float(m_fs_raw)
        max_h_px = max(
            ax_h_px
            - _pt(block.get("top_pad_pt", style.top_pad_pt))
            - _pt(style.bottom_pad_pt)
            - _pt(text_y_inset_pt),
            1.0,
        )
        start_fs = fit_mathtext_block_fontsize(
            renderer, mlines, max_line_px, max_h_px, start_fs,
            usetex=m_usetex, line_dy_pt=line_dy,
            min_fs=float(block.get("mathtext_min_fs", 9.0)),
        )
        m_fs_raw = start_fs
    for j, mline in enumerate(mlines):
        li = n_hand + j
        prog = float(line_progress.get(li, line_progress.get(str(li), 1.0)))
        m_fs = float(m_fs_list[j] if m_fs_list and j < len(m_fs_list) else m_fs_raw)
        min_fs = float(block.get("mathtext_min_fs", 9.0))
        m_fs = fit_mathtext_fontsize(renderer, mline, max_line_px, m_fs, usetex=m_usetex, min_fs=min_fs)
        line_y = y_px
        if m_line_y_inset is not None and j < len(m_line_y_inset):
            line_y += _pt(m_line_y_inset[j])
        mcolors = block.get("mathtext_line_colors")
        line_color = body_color
        if isinstance(mcolors, (list, tuple)) and j < len(mcolors) and mcolors[j]:
            line_color = str(mcolors[j])
        h_px = draw_mathtext_line(
            ax, x, 1.0 - (line_y / ax_h_px), mline, m_fs, color=line_color, ha=ha, write_progress=prog,
            usetex=m_usetex,
        )
        if h_px <= 0:
            continue
        y_px = line_y + h_px
        if j < len(mlines) - 1:
            y_px += line_dy


def export_handwrite_mp4(
    frame_at_progress,
    filename: str,
    *,
    save_mp4,
    output_dir: Path | None = None,
    n_frames: int = 32,
    ms_per_frame: int = 100,
):
    """Build an MP4 from ``frame_at_progress(t)`` for t in [0, 1]."""
    frames = []
    for i in range(n_frames):
        t = 0.0 if n_frames <= 1 else float(i) / float(n_frames - 1)
        frames.append(frame_at_progress(t))
    save_mp4(frames, filename, duration=ms_per_frame)
    out = (output_dir or Path("renders")) / filename
    print(f"wrote {out}  ({len(frames)} frames)")
    return out
