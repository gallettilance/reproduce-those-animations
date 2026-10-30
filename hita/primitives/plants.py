"""Simple green plant icons for the linear-regression plant dataset."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


STEM = (46, 125, 50, 255)
LEAF = (76, 175, 80, 255)
LEAF_DARK = (46, 140, 58, 255)
BUD = (129, 199, 132, 255)


def _leaf_poly(
    cx: float,
    cy: float,
    length: float,
    width: float,
    angle_deg: float,
    n: int = 22,
) -> list[tuple[float, float]]:
    """Teardrop leaf. PIL y grows downward; angle 0 points right, −90 is up."""
    th = np.radians(angle_deg)
    c, s = float(np.cos(th)), float(np.sin(th))
    pts: list[tuple[float, float]] = []
    for t in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        lx = (np.cos(t) + 1.0) * 0.5 * length
        ly = np.sin(t) * width
        pts.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return pts


def make_plant_icon(size: int = 128) -> np.ndarray:
    """Return an RGBA plant (stem + leaves + bud) on a transparent background."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = size * 0.50
    base = size * 0.88
    top = size * 0.18
    stem_w = max(3, size // 26)

    d.line([(cx, base), (cx, top)], fill=STEM, width=stem_w)

    leaves = (
        (cx, size * 0.62, size * 0.34, size * 0.12, -155, LEAF_DARK),
        (cx, size * 0.58, size * 0.36, size * 0.13, -28, LEAF),
        (cx, size * 0.40, size * 0.30, size * 0.11, -148, LEAF),
        (cx, size * 0.36, size * 0.28, size * 0.10, -22, LEAF_DARK),
        (cx, size * 0.26, size * 0.22, size * 0.08, -110, LEAF),
    )
    for x, y, length, width, ang, color in leaves:
        d.polygon(_leaf_poly(x, y, length, width, ang), fill=color)

    bud_r = size * 0.055
    d.ellipse(
        (cx - bud_r, top - bud_r * 0.2, cx + bud_r, top + bud_r * 1.8),
        fill=BUD,
    )
    return np.asarray(img, dtype=np.uint8)


def icon_with_alpha(icon: np.ndarray, alpha: float) -> np.ndarray:
    out = np.array(icon, copy=True)
    a = float(np.clip(alpha, 0.0, 1.0))
    out[..., 3] = np.clip(np.round(out[..., 3].astype(np.float64) * a), 0, 255).astype(np.uint8)
    return out


def crop_plant_icon(icon: np.ndarray, pad: int = 6) -> np.ndarray:
    """Tight RGBA crop around opaque pixels, with a small pad."""
    alpha = icon[..., 3]
    rows = np.any(alpha > 8, axis=1)
    cols = np.any(alpha > 8, axis=0)
    if not np.any(rows) or not np.any(cols):
        return icon
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - np.argmax(cols[::-1]))
    r0 = max(0, r0 - pad)
    r1 = min(icon.shape[0], r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(icon.shape[1], c1 + pad)
    return icon[r0:r1, c0:c1]


def standing_plant_icon(size: int = 256, pad: int = 6) -> tuple[np.ndarray, float, float]:
    """Cropped plant, stem-base box_alignment y, and stem-to-bud span in pixels."""
    raw = make_plant_icon(size)
    alpha = raw[..., 3]
    rows = np.any(alpha > 8, axis=1)
    r0_raw = int(np.argmax(rows)) if np.any(rows) else 0
    r0 = max(0, r0_raw - pad)
    cropped = crop_plant_icon(raw, pad=pad)
    crop_h = float(cropped.shape[0])
    stem_from_top = size * 0.88 - r0
    stem_align = float(np.clip(1.0 - stem_from_top / max(crop_h, 1.0), 0.0, 1.0))
    # Stem base to the top opaque pixel (bud tip), matching the drawn plant.
    plant_span = max(1.0, size * 0.88 - float(r0_raw))
    return cropped, stem_align, plant_span
