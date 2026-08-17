import gc
import io
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mlines
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from PIL import Image, ImageDraw

np.random.seed(7)

OUTPUT_DIR = Path("renders")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Match `logistic-regression-chap1.ipynb` (figure size, DPI, margins, typography) ---
EXPORT_FIGSIZE = (15.0, 9.5)
EXPORT_DPI = 200
EXPORT_ADJ = dict(left=0.04, right=0.96, bottom=0.06, top=0.96)
# Chapter 1 dataset / `fig_to_image` GIFs use **default** `plt.subplots` margins (they do **not** call
# `subplots_adjust(**EXPORT_ADJ)`); that avoids clipping axis labels. Use EXPORT_ADJ only where chap1 does.
SIG3D_ADJ = EXPORT_ADJ  # same name as `logistic-regression-chap1.ipynb` Scene 8 (`SIG3D_ADJ = EXPORT_ADJ`).

FONT_SIZE = 11 * 1.25
AXIS_LABEL_SIZE = 12 * 1.25
LEGEND_SIZE = 20
TITLE_SIZE = 12 * 1.25
NOTE_SIZE = 10 * 1.25
ANNOT_SIZE = 9 * 1.25
SAVE_PAD_INCHES = 0.12

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelpad": 8.0 * 1.25,
        "axes.titlepad": 10.0 * 1.25,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.borderaxespad": 0.55,
        "legend.labelspacing": 0.35,
        "legend.handlelength": 1.35,
        "legend.handletextpad": 0.65,
        "savefig.pad_inches": SAVE_PAD_INCHES,
    }
)

PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"
NEUTRAL_COLOR = "#4f4f4f"

CHECK_ICON_PATH = OUTPUT_DIR / "check.png"
CROSS_ICON_PATH = OUTPUT_DIR / "cross.png"


def _ensure_outcome_icons(size=120, line_width=15):
    _sc = size / 96.0
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.line(
        [(int(round(20 * _sc)), int(round(55 * _sc))), (int(round(42 * _sc)), int(round(76 * _sc))), (int(round(78 * _sc)), int(round(24 * _sc)))],
        fill=(44, 160, 44, 255),
        width=line_width,
    )
    img.save(CHECK_ICON_PATH)

    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.line([(int(round(24 * _sc)), int(round(24 * _sc))), (int(round(72 * _sc)), int(round(72 * _sc)))], fill=(214, 39, 40, 255), width=line_width)
    draw.line([(int(round(72 * _sc)), int(round(24 * _sc))), (int(round(24 * _sc)), int(round(72 * _sc)))], fill=(214, 39, 40, 255), width=line_width)
    img.save(CROSS_ICON_PATH)


_ensure_outcome_icons()
CHECK_ICON = np.asarray(Image.open(CHECK_ICON_PATH).convert("RGBA"))
CROSS_ICON = np.asarray(Image.open(CROSS_ICON_PATH).convert("RGBA"))


def add_outcome_icon(ax, x, y_value, passed, zoom=0.2, alpha=1.0, rotation_deg=0):
    image_array = CHECK_ICON if passed else CROSS_ICON
    if rotation_deg:
        arr = np.asarray(image_array, dtype=np.uint8)
        if int(round(float(rotation_deg) / 180.0)) % 2:
            arr = np.flip(arr, axis=1)
            arr = np.rot90(arr, k=2, axes=(0, 1))
        image_array = arr
    icon = OffsetImage(image_array, zoom=zoom)
    icon.set_alpha(alpha)
    ab = AnnotationBbox(icon, (x, y_value), frameon=False)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def fig_to_image(fig, dpi=None, tight_layout=False, transparent=False):
    """Same PNG raster path as `logistic-regression-chap1.ipynb`."""
    if dpi is None:
        dpi = EXPORT_DPI
    buf = io.BytesIO()
    kw = {"format": "png", "dpi": dpi, "pad_inches": SAVE_PAD_INCHES}
    if tight_layout:
        kw["bbox_inches"] = "tight"
    else:
        kw["bbox_inches"] = None
    if transparent:
        kw["transparent"] = True
    fig.savefig(buf, **kw)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA" if transparent else "RGB")


def save_gif(images, filename, duration=40):
    if not images:
        raise ValueError("images list is empty")
    rgb = [im.convert("RGB") for im in images]
    w = max(im.width for im in rgb)
    h = max(im.height for im in rgb)
    if any(im.size != (w, h) for im in rgb):
        bg = (255, 255, 255)
        normed = []
        for im in rgb:
            if im.size == (w, h):
                normed.append(im.copy())
            else:
                canvas = Image.new("RGB", (w, h), bg)
                canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
                normed.append(canvas)
        rgb = normed
    rgb[0].save(
        OUTPUT_DIR / filename,
        save_all=True,
        append_images=rgb[1:],
        duration=duration,
        loop=0,
    )



def save_mp4(images, filename, duration=40):
    if not images:
        raise ValueError("images list is empty")
    rgb = [im.convert("RGB") for im in images]
    w = max(im.width for im in rgb)
    h = max(im.height for im in rgb)
    if any(im.size != (w, h) for im in rgb):
        bg = (255, 255, 255)
        normed = []
        for im in rgb:
            if im.size == (w, h):
                normed.append(im.copy())
            else:
                canvas = Image.new("RGB", (w, h), bg)
                canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
                normed.append(canvas)
        rgb = normed
    rgb[0].save(
        OUTPUT_DIR / filename,
        save_all=True,
        append_images=rgb[1:],
        duration=duration,
        loop=0,
    )


def save_mp4(images, filename, duration=40):
    """H.264 **MP4** from PIL frames; **fps** = ``1000 / duration_ms`` (same convention as ``save_gif``). Requires **imageio** + **imageio-ffmpeg**."""
    import imageio.v2 as imageio

    if not images:
        raise ValueError("images list is empty")
    rgb = [im.convert("RGB") for im in images]
    w = max(im.width for im in rgb)
    h = max(im.height for im in rgb)
    if any(im.size != (w, h) for im in rgb):
        bg = (255, 255, 255)
        normed = []
        for im in rgb:
            if im.size == (w, h):
                normed.append(im.copy())
            else:
                canvas = Image.new("RGB", (w, h), bg)
                canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
                normed.append(canvas)
        rgb = normed
    frames = [np.asarray(im) for im in rgb]
    fps = 1000.0 / max(float(duration), 1e-3)
    path = OUTPUT_DIR / filename
    imageio.mimsave(
        str(path),
        frames,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", "20"],
    )


def save_fig(fig, filename, dpi=None, tight_layout=False, transparent=False):
    """Chapter 1 style: raster via `fig_to_image`, then write PNG."""
    im = fig_to_image(fig, dpi=dpi, tight_layout=tight_layout, transparent=transparent)
    im.save(OUTPUT_DIR / filename)


# --- GIF densification: same parameter ranges, finer steps, shorter ms/frame (~higher FPS) ---
GIF_SMOOTH_FACTOR = 4  # ↑ denser samples along the same ranges; paired with ``_gif_dur`` (~↑FPS)


def _gif_dur(base_ms):
    """Per-frame duration when frame count is scaled by ``GIF_SMOOTH_FACTOR``."""
    return max(8, int(round(float(base_ms) / GIF_SMOOTH_FACTOR)))


def _smooth_n(n):
    """Multiply discrete frame/sample counts for smooth parameter sweeps."""
    return max(2, int(round(float(n) * GIF_SMOOTH_FACTOR)))


# --- Separable roster only (no `noisy_symmetric_points` from chap1) ---
separable_points = [
    (2, 3, 0),
    (4, 5, 0), (5, 6, 0),
    (1, 3, 0), (2, 4, 0), (4, 6, 0),
    (1, 4, 0), (3, 6, 0), (1, 6, 0),
    (3, 2, 1),
    (5, 4, 1), (6, 5, 1),
    (4, 2, 1), (6, 4, 1), (3, 1, 1),
    (4, 1, 1), (5, 2, 1), (6, 3, 1),
    (6, 2, 1),
    (6, 1, 1),
]


def unpack_points(point_list):
    arr = np.array(point_list, dtype=floa