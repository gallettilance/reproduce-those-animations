import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw

np.random.seed(7)

OUTPUT_DIR = Path("renders")
OUTPUT_DIR.mkdir(exist_ok=True)

# Canonical figure size and DPI for every render (matches sigmoid 3D 61–64).
EXPORT_FIGSIZE = (15.0, 9.5)
EXPORT_DPI = 200
EXPORT_ADJ = dict(left=0.04, right=0.96, bottom=0.06, top=0.96)
EXPORT_DUO_ADJ = dict(left=0.07, right=0.98, top=0.96, bottom=0.12)
# Same width/height in inches as EXPORT_FIGSIZE; new tuples so names never share one mutable/list identity.
NORM_DUO_FIGSIZE = (float(EXPORT_FIGSIZE[0]), float(EXPORT_FIGSIZE[1]))

# Typography / padding (single source of truth for exports); all scaled +25%.
FONT_SIZE = 11 * 1.25
AXIS_LABEL_SIZE = 12 * 1.25
LEGEND_SIZE = 20
TITLE_SIZE = 12 * 1.25
NOTE_SIZE = 10 * 1.25
ANNOT_SIZE = 9 * 1.25
SAVE_PAD_INCHES = 0.12

# Exponential scenes: same inch size as exports; EXP_SUBPLOT_ADJ is the margin preset (not the duo layout).
EXP_FIGSIZE = (float(EXPORT_FIGSIZE[0]), float(EXPORT_FIGSIZE[1]))
EXP_SUBPLOT_ADJ = dict(left=0.09, right=0.91, bottom=0.09, top=0.91)


def finalize_exponential_figure(fig):
    fig.subplots_adjust(**EXP_SUBPLOT_ADJ)


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
        # Match 3D icons: left–right flip then 180° rotation (odd half-turns of 180°)
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
    return 1.0 / (1.0 + np.exp(-z))


def fig_to_image(fig, dpi=None, tight_layout=False):
    if dpi is None:
        dpi = EXPORT_DPI
    buf = io.BytesIO()
    kw = {"format": "png", "dpi": dpi, "pad_inches": SAVE_PAD_INCHES}
    if tight_layout:
        kw["bbox_inches"] = "tight"
    else:
        kw["bbox_inches"] = None
    fig.savefig(buf, **kw)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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
