"""Parameter dials: numbered (1/2/3) vs labeled (w_ST / w_EL / b).

Switch style with ``KnobStyle`` or a per-slot ``blend`` (0=numbered → 1=labeled).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image, ImageDraw

from hita.assets import knobs_dir
from hita.config.export import output_dir
from hita.export.context import project_root

# --- constants (match Ch2/Ch3 series) ---
KNOB_NORM_R = 448
KNOB_DEG_PER_UNIT = 360.0
KNOB_STRIP_MAX_DEG = 52.0
KNOB_PIL_MAX_DEG = 20.0 * 360.0
KNOB_ACTIVE_SCALE = 1.25
KNOB_CROP_PAD = (11, 11, 2, 1)  # L, R, B, T
KNOB_CROP_ALPHA = 128
KNOB_CROP_WHITE = 250

NUMBERED_KEYS = ("1", "2", "3")
LABELED_KEYS = ("wst", "wel", "b")

Param = Literal["st", "el", "b"]
Emphasize = Param | Literal["all", "both"] | Sequence[Param] | None

_NORM_CACHE: dict[int, Image.Image] = {}
_PACK_CACHE: dict[tuple, "KnobPack"] = {}


class KnobStyle(str, Enum):
    NUMBERED = "numbered"  # knob_1 / knob_2 / knob_3
    LABELED = "labeled"  # knob_wst / knob_wel / knob_b (param faces)


@dataclass(frozen=True)
class KnobPack:
    """Three dial images + canvas sides for rotation without frame-to-frame zoom."""

    images: tuple[Image.Image, Image.Image, Image.Image]
    canvas_sides: tuple[int, int, int]
    style: KnobStyle

    def as_legacy(self) -> tuple[tuple[Image.Image, ...], tuple[int, int, int]]:
        """``(rgbs, canvas_sides)`` shape expected by Ch3/Ch4/Ch6 draw helpers."""
        return self.images, self.canvas_sides


def knobs_assets_dir(style: KnobStyle | None = None) -> Path:
    base = knobs_dir()
    if style is None:
        return base
    return base / style.value


def crop_knob_image(
    src: Path | Image.Image,
    *,
    pad: tuple[int, int, int, int] = KNOB_CROP_PAD,
    alpha_thr: int = KNOB_CROP_ALPHA,
    white_thr: int = KNOB_CROP_WHITE,
) -> Image.Image:
    """Tight crop around the dial (same rules as ``knob_1_cropped``)."""
    pl, pr, pb, pt = pad
    im = Image.open(src).convert("RGBA") if not isinstance(src, Image.Image) else src.convert("RGBA")
    arr = np.asarray(im)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = (alpha > alpha_thr) & (rgb.max(axis=2) < white_thr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - int(pl))
    x1 = min(im.width - 1, x1 + int(pr))
    y0 = max(0, y0 - int(pt))
    y1 = min(im.height - 1, y1 + int(pb))
    return im.crop((x0, y0, x1 + 1, y1 + 1))


def blend_knob_images(old_im: Image.Image, new_im: Image.Image, u: float) -> Image.Image:
    u = float(np.clip(float(u), 0.0, 1.0))
    if u <= 1e-9:
        return old_im.convert("RGBA")
    if u >= 1.0 - 1e-9:
        return new_im.convert("RGBA")
    o = np.asarray(old_im.convert("RGBA"), dtype=np.float32)
    n = np.asarray(new_im.convert("RGBA"), dtype=np.float32)
    if o.shape != n.shape:
        n_im = new_im.convert("RGBA").resize(old_im.size, Image.Resampling.LANCZOS)
        n = np.asarray(n_im, dtype=np.float32)
    out = (1.0 - u) * o + u * n
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")


def _copy_if_needed(src: Path, dest: Path) -> None:
    if not src.is_file():
        return
    if dest.is_file() and dest.stat().st_size > 16 and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _placeholder_numbered(path: Path, label: str) -> None:
    im = Image.new("RGBA", (180, 180), (248, 248, 248, 0))
    dr = ImageDraw.Draw(im)
    dr.ellipse((10, 10, 170, 170), outline=(100, 100, 100, 255), width=5)
    dr.text((74, 68), label, fill=(50, 50, 50, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def ensure_knob_assets(
    style: KnobStyle | None = None,
    *,
    dest: Path | None = None,
    force_crop: bool = False,
) -> Path:
    """Copy bundled knobs into the output dir; crop labeled faces if needed."""
    out = Path(dest or output_dir(project_root()))
    out.mkdir(parents=True, exist_ok=True)
    styles = (KnobStyle.NUMBERED, KnobStyle.LABELED) if style is None else (style,)

    if KnobStyle.NUMBERED in styles:
        bundled = knobs_assets_dir(KnobStyle.NUMBERED)
        for key in NUMBERED_KEYS:
            name = f"knob_{key}_cropped.png"
            dest_p = out / name
            _copy_if_needed(bundled / name, dest_p)
            if not dest_p.is_file() or dest_p.stat().st_size < 16:
                _placeholder_numbered(dest_p, key)

    if KnobStyle.LABELED in styles:
        bundled = knobs_assets_dir(KnobStyle.LABELED)
        for key in LABELED_KEYS:
            full_name = f"knob_{key}.png"
            crop_name = f"knob_{key}_cropped.png"
            full_dest = out / full_name
            crop_dest = out / crop_name
            _copy_if_needed(bundled / full_name, full_dest)
            _copy_if_needed(bundled / crop_name, crop_dest)
            if not full_dest.is_file():
                raise FileNotFoundError(
                    f"missing labeled knob source: {full_dest} "
                    f"(expected bundled at {bundled / full_name})"
                )
            if force_crop or not crop_dest.is_file() or crop_dest.stat().st_mtime < full_dest.stat().st_mtime:
                crop_knob_image(full_dest).save(crop_dest)

    return out


def normalize_square(knob_rgba: Image.Image, *, norm_r: int = KNOB_NORM_R) -> Image.Image:
    kid = id(knob_rgba)
    hit = _NORM_CACHE.get(kid)
    if hit is not None:
        return hit
    R = int(norm_r)
    w0, h0 = knob_rgba.size
    scale = float(R) / float(max(int(w0), int(h0), 1))
    nw = max(1, int(round(float(w0) * scale)))
    nh = max(1, int(round(float(h0) * scale)))
    im1 = knob_rgba.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    out.paste(im1, ((R - nw) // 2, (R - nh) // 2), im1)
    _NORM_CACHE[kid] = out
    return out


def probe_canvas_side(
    knob_rgba: Image.Image,
    *,
    deg_lo: float = -180.0,
    deg_hi: float = 180.0,
    n: int = 37,
    pad: int = 8,
) -> int:
    """Max rotated bbox on the normalized square (cover ±180° for half-integer coeffs)."""
    base = normalize_square(knob_rgba)
    s = 0
    for deg in np.linspace(float(deg_lo), float(deg_hi), int(n)):
        im = base.rotate(float(deg), resample=Image.Resampling.BICUBIC, expand=True)
        s = max(s, im.width, im.height)
    return int(s + pad)


def rotated_square(knob_rgba: Image.Image, deg: float, canvas_side: int) -> Image.Image:
    """Rotate on a fixed square raster, then resize — no per-angle fit zoom."""
    side = int(canvas_side)
    base = normalize_square(knob_rgba)
    if abs(float(deg)) <= 1e-9:
        return base.resize((side, side), resample=Image.Resampling.LANCZOS)
    im = base.rotate(float(deg), resample=Image.Resampling.BICUBIC, expand=False)
    return im.resize((side, side), resample=Image.Resampling.LANCZOS)


def rotation_absolute(ws: float, we: float, bb: float) -> tuple[float, float, float]:
    """Integer → 0°; half-integer → 180°; else −frac·360° (PIL CW when value increases)."""

    def one(param: float) -> float:
        v = float(param)
        if abs(v - round(v)) < 1e-5:
            deg = 0.0
        elif abs(2.0 * v - round(2.0 * v)) < 1e-5:
            deg = 180.0
        else:
            frac = v - round(v)
            deg = float(-frac * KNOB_DEG_PER_UNIT)
        return float(np.clip(deg, -KNOB_PIL_MAX_DEG, KNOB_PIL_MAX_DEG))

    return one(ws), one(we), one(bb)


def rotation_relative(
    ws: float,
    we: float,
    bb: float,
    ref: tuple[float, float, float],
) -> tuple[float, float, float]:
    """360°/unit vs ``ref`` (Ch2 training/gain clips)."""
    rw, re, rb = ref
    md = float(KNOB_PIL_MAX_DEG)
    return (
        float(np.clip(-(float(ws) - float(rw)) * KNOB_DEG_PER_UNIT, -md, md)),
        float(np.clip(-(float(we) - float(re)) * KNOB_DEG_PER_UNIT, -md, md)),
        float(np.clip(-(float(bb) - float(rb)) * KNOB_DEG_PER_UNIT, -md, md)),
    )


def rotation_strip(
    val: float,
    lo: float,
    hi: float,
    *,
    ref: float | None = None,
    max_deg: float = KNOB_STRIP_MAX_DEG,
) -> float:
    """Limited strip sweep: 0° at ``ref`` (default ``lo``)."""
    lo_f, hi_f = float(lo), float(hi)
    if ref is None:
        ref = lo_f
    span = hi_f - lo_f
    amp = max(abs(span), 1e-6)
    deg = -float(max_deg) * ((float(val) - float(ref)) / amp)
    lim = float(max_deg) * 1.05
    return float(np.clip(deg, -lim, lim))


def active_mask(emphasize: Emphasize) -> tuple[bool, bool, bool]:
    if emphasize is None:
        return (False, False, False)
    if isinstance(emphasize, (tuple, list, set, frozenset)) and not isinstance(emphasize, str):
        s = {str(x).lower() for x in emphasize}
        return ("st" in s, "el" in s, "b" in s)
    key = str(emphasize).lower()
    if key == "st":
        return (True, False, False)
    if key == "el":
        return (False, True, False)
    if key == "b":
        return (False, False, True)
    if key == "both":
        return (True, True, False)
    if key == "all":
        return (True, True, True)
    return (False, False, False)


def scales_emphasize(
    emphasize: Emphasize,
    active_scale: float = KNOB_ACTIVE_SCALE,
) -> tuple[float, float, float]:
    sc = float(np.clip(float(active_scale), 1.0, float(KNOB_ACTIVE_SCALE)))
    return tuple(sc if a else 1.0 for a in active_mask(emphasize))  # type: ignore[return-value]


def _load_cropped_triple(out: Path, keys: tuple[str, ...]) -> tuple[Image.Image, Image.Image, Image.Image]:
    return tuple(  # type: ignore[return-value]
        Image.open(out / f"knob_{k}_cropped.png").convert("RGBA") for k in keys
    )


def load_knob_pack(
    style: KnobStyle = KnobStyle.NUMBERED,
    *,
    blend: tuple[float, float, float] | None = None,
    dest: Path | None = None,
    unify_sides: bool = True,
    probe_deg: tuple[float, float] = (-180.0, 180.0),
) -> KnobPack:
    """Load a dial pack. ``blend=(u0,u1,u2)`` crossfades numbered → labeled per slot."""
    out = ensure_knob_assets(None if blend is not None else style, dest=dest)
    cache_key = (style.value, blend, str(out.resolve()), unify_sides, probe_deg)
    hit = _PACK_CACHE.get(cache_key)
    if hit is not None:
        return hit

    deg_lo, deg_hi = probe_deg

    if blend is not None:
        ensure_knob_assets(None, dest=out)
        num = _load_cropped_triple(out, NUMBERED_KEYS)
        lab = _load_cropped_triple(out, LABELED_KEYS)
        blends = tuple(float(np.clip(float(u), 0.0, 1.0)) for u in blend)
        if all(u <= 1e-9 for u in blends):
            images = num
            resolved = KnobStyle.NUMBERED
        elif all(u >= 1.0 - 1e-9 for u in blends):
            images = lab
            resolved = KnobStyle.LABELED
        else:
            images = tuple(blend_knob_images(num[i], lab[i], blends[i]) for i in range(3))  # type: ignore[assignment]
            resolved = KnobStyle.LABELED if sum(blends) >= 1.5 else KnobStyle.NUMBERED
        sides_src = num  # keep numbered canvas sides during morph (no flicker)
    else:
        keys = NUMBERED_KEYS if style is KnobStyle.NUMBERED else LABELED_KEYS
        images = _load_cropped_triple(out, keys)
        sides_src = images
        resolved = style

    sides = tuple(probe_canvas_side(im, deg_lo=deg_lo, deg_hi=deg_hi) for im in sides_src)
    if unify_sides:
        side_uni = int(max(sides))
        sides = (side_uni, side_uni, side_uni)

    pack = KnobPack(images=images, canvas_sides=sides, style=resolved)  # type: ignore[arg-type]
    _PACK_CACHE[cache_key] = pack
    return pack


def resolve_knob_pack(
    *,
    style: KnobStyle | None = None,
    blend: tuple[float, float, float] | None = None,
    dest: Path | None = None,
) -> KnobPack:
    """Pick numbered / labeled / blended (Ch4 morph helper)."""
    if blend is not None:
        return load_knob_pack(KnobStyle.NUMBERED, blend=blend, dest=dest)
    return load_knob_pack(style or KnobStyle.NUMBERED, dest=dest)


def place_knob_row(
    fig,
    axes_k,
    pack: KnobPack,
    rot_degs: Sequence[float],
    slot_scales: Sequence[float],
) -> None:
    """Place three dials into ``axes_k`` using current axis positions as slots."""
    axes_k = tuple(axes_k)
    scales = [
        float(np.clip(float(slot_scales[i]), 1.0, float(KNOB_ACTIVE_SCALE))) for i in range(3)
    ]
    slot_rects = [
        (float(axk.get_position().x0), float(axk.get_position().y0),
         float(axk.get_position().width), float(axk.get_position().height))
        for axk in axes_k
    ]
    active_i = int(np.argmax(scales))
    draw_order = [i for i in range(3) if i != active_i] + [active_i]
    for i in draw_order:
        axk = axes_k[i]
        x0, y0, w, h = slot_rects[i]
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        sc = scales[i]
        axk.set_position((cx - 0.5 * sc * w, cy - 0.5 * sc * h, sc * w, sc * h))
        arr = np.asarray(rotated_square(pack.images[i], float(rot_degs[i]), pack.canvas_sides[i]))
        axk.clear()
        axk.imshow(arr, interpolation="nearest")
        axk.axis("off")
    fig.canvas.draw()


def draw_knob_row(
    fig,
    axes_k,
    ws: float,
    we: float,
    bb: float,
    *,
    style: KnobStyle = KnobStyle.NUMBERED,
    blend: tuple[float, float, float] | None = None,
    emphasize: Emphasize = None,
    rotations: Sequence[float] | None = None,
    scales: Sequence[float] | None = None,
    pack: KnobPack | None = None,
    angle_refs: tuple[float, float, float] | None = None,
) -> KnobPack:
    """Draw dials. Switch faces with ``style`` or morph with ``blend``.

    Examples::

        draw_knob_row(fig, axes_k, ws, we, bb, style=KnobStyle.NUMBERED)
        draw_knob_row(fig, axes_k, ws, we, bb, style=KnobStyle.LABELED, emphasize="all")
        draw_knob_row(fig, axes_k, ws, we, bb, blend=(0.0, 0.5, 1.0))
    """
    resolved = pack or resolve_knob_pack(style=style, blend=blend)
    if rotations is not None:
        rots = tuple(float(r) for r in rotations)
    elif angle_refs is not None:
        rots = rotation_relative(ws, we, bb, angle_refs)
    else:
        rots = rotation_absolute(ws, we, bb)
    if scales is not None:
        sc = tuple(float(s) for s in scales)
    else:
        sc = scales_emphasize(emphasize)
    place_knob_row(fig, axes_k, resolved, rots, sc)
    return resolved


def clear_knob_caches() -> None:
    _NORM_CACHE.clear()
    _PACK_CACHE.clear()
