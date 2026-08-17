"""Series colormaps (red→white→green σ field from Ch1/Ch2)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def sigma_rwg_cmap(name: str = "hita_sigma_rwg", n: int = 100):
    """Canonical pass/fail probability colormap used across Ch1–Ch2 Scene 8 / GD."""
    cvals = [0.0, 0.5, 1.0]
    colors = ["red", "white", "green"]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    return mpl.colors.LinearSegmentedColormap.from_list(name, tuples, n)


# Module-level defaults (lazy-safe)
CMAP = sigma_rwg_cmap()
CMAP_GD = sigma_rwg_cmap("hita_sigma_gd")
