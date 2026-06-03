"""Reusable handwriting font + typewriter reveal for plot + math tutorial frames."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    text_color: str = "#1a1a28"
    title_color: str = "#1a1a28"
    accent_color: str = "#1a1a28"
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
    subscript_scale: float = 0.62
    subscript_drop_frac: float = 0.12


SYMBOL_CHARS = frozenset("∇∂Σ∑ℒΠ∈")
Run = tuple[str, str]  # (kind, text) — kind: hand | sub | sym


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


def symbol_font(size: float, *, style: HandwriteStyle | None = None) -> fm.FontProperties:
    style = style or HandwriteStyle()
    return fm.FontProperties(family=style.symbol_font_family, size=float(size))


def _merge_adjacent_runs(runs: list[Run]) -> list[Run]:
    if not runs:
        return runs
    out: list[list[str]] = [list(runs[0])]
    for kind, text in runs[1:]:
        if kind == out[-1][0]:
            out[-1][1] += text
        else:
            out.append([kind, text])
    return [(k, t) for k, t in out]


def parse_handwrite_runs(text: str) -> list[Run]:
    """Split plain display text into handwriting / subscript / symbol runs."""
    runs: list[Run] = []
    i, n = 0, len(str(text))
    s = str(text)
    while i < n:
        ch = s[i]
        if ch in SYMBOL_CHARS:
            runs.append(("sym", ch))
            i += 1
        elif ch == "_":
            i += 1
            if i < n and s[i] == "{":
                j = s.find("}", i + 1)
                if j > i:
                    sub = s[i + 1:j]
                    i = j + 1
                    runs.append(("sub", sub))
                else:
                    runs.append(("hand", "_"))
            else:
                start = i
                while i < n and s[i].isalnum():
                    i += 1
                sub = s[start:i]
                if sub:
                    runs.append(("sub", sub))
                else:
                    runs.append(("hand", "_"))
        else:
            start = i
            while i < n and s[i] not in SYMBOL_CHARS and s[i] != "_":
                i += 1
            if start < i:
                runs.append(("hand", s[start:i]))
    return _merge_adjacent_runs(runs)


def flatten_runs(runs: list[Run]) -> list[Run]:
    atoms: list[Run] = []
    for kind, text in runs:
        if kind == "sub" and len(text) > 1:
            atoms.append((kind, text))
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
    return "".join(text for _, text in runs)


def _run_font(kind: str, main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand: fm.FontProperties | None):
    if kind == "sym":
        return symbol_font(main_fs, style=style)
    if kind == "sub":
        sub_fs = main_fs * style.subscript_scale
        return hand_font(sub_fs, bold=False) if style.enabled else fm.FontProperties(size=sub_fs)
    if fp_hand is not None:
        return fp_hand
    return hand_font(main_fs, bold=bold) if style.enabled else fm.FontProperties(size=main_fs, weight="bold" if bold else "normal")


def _run_fontsize(kind: str, main_fs: float, *, style: HandwriteStyle) -> float:
    if kind == "sub":
        return main_fs * style.subscript_scale
    return main_fs


def run_size_px(renderer, kind: str, text: str, main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    fs = _run_fontsize(kind, main_fs, style=style)
    fp = _run_font(kind, main_fs, style=style, bold=bold, fp_hand=fp_hand)
    w_px, h_px, d_px = renderer.get_text_width_height_descent(str(text), fp, ismath=False)
    return float(w_px), float(h_px + d_px)


def mixed_line_width_px(renderer, runs: list[Run], main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    total = 0.0
    for kind, text in runs:
        w_px, _ = run_size_px(renderer, kind, text, main_fs, style=style, bold=bold, fp_hand=fp_hand)
        total += w_px
    return total


def mixed_line_height_px(renderer, runs: list[Run], main_fs: float, *, style: HandwriteStyle, bold: bool, fp_hand):
    fp_main = _run_font("hand", main_fs, style=style, bold=bold, fp_hand=fp_hand)
    _, main_h_px, main_d_px = renderer.get_text_width_height_descent("x", fp_main, ismath=False)
    sub_ext_px = 0.0
    for kind, text in runs:
        if kind == "sub":
            fp = _run_font(kind, main_fs, style=style, bold=bold, fp_hand=fp_hand)
            _, sub_h_px, sub_d_px = renderer.get_text_width_height_descent(str(text), fp, ismath=False)
            drop = main_d_px + main_fs * style.subscript_drop_frac * 0.85
            sub_ext_px = max(sub_ext_px, drop + sub_d_px)
    return float(main_h_px + main_d_px + sub_ext_px)


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
    # Caller passes a top anchor (va="top"); convert once to a shared baseline.
    baseline_y = y - main_h_px / ax_h_px
    dpi = float(fig.get_dpi())
    sub_drop_px = main_d_px + main_fs * style.subscript_drop_frac * (dpi / 72.0)

    total_w_px = mixed_line_width_px(renderer, runs, main_fs, style=style, bold=bold, fp_hand=fp_hand)
    x_cursor = x
    if ha == "center":
        x_cursor = x - (total_w_px / ax_w_px) / 2.0
    elif ha == "right":
        x_cursor = x - (total_w_px / ax_w_px)

    for kind, text in runs:
        if not text:
            continue
        fp = _run_font(kind, main_fs, style=style, bold=bold, fp_hand=fp_hand)
        if kind == "sub":
            y_draw = baseline_y - sub_drop_px / ax_h_px
        else:
            y_draw = baseline_y
        target.text(
            x_cursor,
            y_draw,
            text,
            transform=transform,
            va="baseline",
            ha="left",
            color=color,
            fontproperties=fp,
            clip_on=False,
        )
        w_px, _ = run_size_px(renderer, kind, text, main_fs, style=style, bold=bold, fp_hand=fp_hand)
        x_cursor += w_px / ax_w_px

    return mixed_line_height_px(renderer, runs, main_fs, style=style, bold=bold, fp_hand=fp_hand)


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


def latex_line_to_handwrite(line: str) -> str:
    s = str(line).strip()
    if not s:
        return s
    s = s.replace("$", "")
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{L\}", "ℒ", s)
    s = re.sub(r"\\prod_i", "Π_i", s)
    s = re.sub(r"\\prod", "Π", s)
    s = re.sub(r"\\sum_i", "Σ_i", s)
    s = re.sub(r"\\sum", "Σ", s)
    s = s.replace(r"\partial", "∂")
    s = s.replace(r"\nabla_w", "∇_w").replace(r"\nabla", "∇")
    s = s.replace(r"\mid", " | ").replace(r"\,", " ").replace(r"\;", " ")
    s = s.replace(r"\log", "log ")
    s = s.replace(r"\in", "∈")
    s = s.replace(r"\{", "").replace(r"\}", "")
    s = re.sub(r"\\hat\s*p_i", "p\u0302_i", s)
    s = re.sub(r"\\hat\s*p", "p\u0302", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"_\{([^}]+)\}", r"_\1", s)
    s = re.sub(r"\s*=\s*", " = ", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("log ℒ", "log ℒ").replace("log  ", "log ")
    return s


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
    alpha = 1.0 if float(write_progress) >= 1.0 else ease_write_progress(float(write_progress))
    target.text(
        x, y, str(line), transform=transform, fontsize=float(fontsize), color=color,
        ha=ha, va="top", alpha=alpha, clip_on=False, usetex=usetex,
    )
    fig.canvas.draw()
    return mathtext_line_height_px(fig.canvas.get_renderer(), line, fontsize, usetex=usetex)


def block_display_lines(block: dict, *, style: HandwriteStyle) -> list[str]:
    bold = bool(block.get("bold_lhs", False))
    raw = [ln for ln in str(block.get("text", "")).split("\n") if ln.strip()]
    return [display_line(ln, style=style, bold_lhs=bold) for ln in raw]


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
    words = str(text).split()
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
):
    fs = float(style.section_title_fs if title_fs is None else title_fs)
    color = title_color if title_color is not None else style.title_color
    if max_width_frac is None:
        max_width_frac = max(1.0 - float(xy[0]) - pad_frac, 0.05)
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
    max_line_px = ax_w_px * (0.94 if ha == "left" else 0.88)
    max_line_frac = max_line_px / float(fig.bbox.width)
    fp_block = hand_font(body_fs, bold=bool(block.get("bold_lhs", False))) if style.enabled else None
    fp_label = hand_font(label_fs_eff) if style.enabled else None
    line_progress = line_progress or {}
    lab_color = label_color if label_color is not None else style.label_color
    body_color = text_color if text_color is not None else style.text_color
    if block.get("text_color"):
        body_color = str(block["text_color"])
    lhs_color = accent_color if accent_color is not None else style.accent_color

    def _pt(val):
        return block_pt_to_px(fig, block, val)

    line_dy = _pt(block.get("line_dy_pt", style.line_dy_pt))
    label_gap = _pt(block.get("label_gap_pt", style.label_gap_pt))
    y_px = _pt(block.get("top_pad_pt", style.top_pad_pt)) + _pt(text_y_inset_pt)

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

    raw_lines = block_display_lines(block, style=style)
    lines = []
    for raw in raw_lines:
        runs = parse_handwrite_runs(raw)
        w_px = mixed_line_width_px(renderer, runs, body_fs, style=style, bold=bool(block.get("bold_lhs", False)), fp_hand=fp_block)
        if w_px <= max_line_px or style.enabled:
            lines.append(raw)
        else:
            lines.extend(wrap_text_for_width(fig, raw, body_fs, max_line_frac, bold=False, style=style))

    for i, line in enumerate(lines):
        prog = float(line_progress.get(i, line_progress.get(str(i), 1.0)))
        h_px = draw_mixed_line(
            ax, x, _y_axes(), line, body_fs, style=style,
            bold=bool(block.get("bold_lhs", False)), color=lhs_color if block.get("bold_lhs") else body_color,
            ha=ha, write_progress=prog, reveal_mode=style.line_mode, fp_hand=fp_block,
        )
        if h_px <= 0:
            continue
        y_px += h_px
        if i < len(lines) - 1 or block_mathtext_lines(block):
            y_px += line_dy

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
