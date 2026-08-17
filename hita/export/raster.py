"""Fast Agg rasterization (from ch4_export_pipeline)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def fig_to_image(fig, dpi: int | None = None, *, tight_layout: bool = False, transparent: bool = False):
    import matplotlib.pyplot as plt

    if dpi is not None:
        fig.set_dpi(float(dpi))
    if tight_layout:
        fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    plt.close(fig)
    if transparent:
        return Image.fromarray(rgba, mode="RGBA")
    return Image.fromarray(rgba[..., :3], mode="RGB")
