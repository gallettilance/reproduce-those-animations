"""3D sigmoid surface helpers (Ch1 Scene 8 + Ch2 morph DNA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from hita.primitives.colormap import CMAP
from hita.primitives.math import sigmoid


@dataclass(frozen=True)
class SigmoidMesh:
    """ST/EL grid + σ(ST−EL) / σ(−(ST−EL)) surfaces."""

    ST: np.ndarray
    EL: np.ndarray
    DIFF: np.ndarray
    P_PASS: np.ndarray
    P_FAIL: np.ndarray
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    @classmethod
    def build(
        cls,
        xlim: tuple[float, float] = (0.5, 6.5),
        ylim: tuple[float, float] = (0.5, 6.5),
        n: int = 220,
        *,
        w_st: float = 1.0,
        w_el: float = -1.0,
        b: float = 0.0,
    ) -> SigmoidMesh:
        st_axis = np.linspace(xlim[0], xlim[1], int(n))
        el_axis = np.linspace(ylim[0], ylim[1], int(n))
        ST, EL = np.meshgrid(st_axis, el_axis)
        DIFF = float(w_st) * ST + float(w_el) * EL + float(b)
        return cls(
            ST=ST,
            EL=EL,
            DIFF=DIFF,
            P_PASS=sigmoid(DIFF),
            P_FAIL=sigmoid(-DIFF),
            xlim=(float(xlim[0]), float(xlim[1])),
            ylim=(float(ylim[0]), float(ylim[1])),
        )

    def morph_z(self, u: float, *, pass_surface: bool = True) -> np.ndarray:
        """Flat z=0 → curved σ surface."""
        target = self.P_PASS if pass_surface else self.P_FAIL
        return float(u) * target


def style_sigmoid_axes(
    ax,
    az: float,
    *,
    elev: float = 26.0,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    hide_z: bool = False,
    exam_label_2d: bool = False,
    diag_prob_scale: float = 1.0,
    show_threshold: bool = True,
    font_size: float = 13.75,
    axis_label_size: float = 15.0,
) -> None:
    """Ch1 Scene 8 camera + threshold diagonal on the σ surface."""
    if show_threshold:
        diag = np.linspace(max(xlim[0], ylim[0]), min(xlim[1], ylim[1]), 220)
        ax.plot(diag, diag, float(diag_prob_scale) * 0.5, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    if hide_z:
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.tick_params(axis="z", labelsize=0, colors="none")
    else:
        ax.tick_params(axis="z", labelsize=font_size)
    ax.set_xlabel("Study time (hours)", fontsize=axis_label_size, labelpad=12.5)
    if exam_label_2d:
        ax.set_ylabel("")
        ax.text2D(
            -0.2,
            0.5,
            "Exam length (hours)",
            transform=ax.transAxes,
            fontsize=axis_label_size,
            rotation=90,
            va="center",
            ha="center",
        )
    else:
        ax.set_ylabel("Exam length (hours)", fontsize=axis_label_size, labelpad=12.5)
    ax.set_zlabel("" if hide_z else "Probability", fontsize=axis_label_size, labelpad=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, 1)
    ax.view_init(elev=float(elev), azim=float(az))


def scatter_outcome_icons_3d(
    ax,
    study,
    exam,
    y,
    z_values,
    *,
    check_icon: np.ndarray,
    cross_icon: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rotate_icons_180: bool = False,
    span_frac: float | None = None,
) -> None:
    """Draw pass/fail icons as flat RGBA quads on the surface (Ch1 Scene 8)."""
    xr = float(xlim[1] - xlim[0])
    yr = float(ylim[1] - ylim[0])
    frac = 0.045 if span_frac is None else float(span_frac)
    span = min(xr, yr) * frac
    nx, ny = 18, 18
    study = np.asarray(study, dtype=float)
    exam = np.asarray(exam, dtype=float)
    y = np.asarray(y, dtype=int)
    z_values = np.asarray(z_values, dtype=float)
    polys: list = []
    fcols: list = []
    for k in range(len(study)):
        s, e, z = float(study[k]), float(exam[k]), float(z_values[k])
        img_arr = check_icon if int(y[k]) == 1 else cross_icon
        im = Image.fromarray(np.asarray(img_arr, dtype=np.uint8), mode="RGBA")
        im = im.resize((nx, ny), Image.LANCZOS)
        if rotate_icons_180:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(180, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255, 0))
            im = im.resize((nx, ny), Image.LANCZOS)
        pix = np.asarray(im).astype(float) / 255.0
        for jj in range(ny):
            for ii in range(nx):
                rgba = pix[jj, ii]
                if rgba[3] < 0.06:
                    continue
                x0 = s - 0.5 * span + (ii / nx) * span
                x1 = s - 0.5 * span + ((ii + 1) / nx) * span
                y0 = e - 0.5 * span + (jj / ny) * span
                y1 = e - 0.5 * span + ((jj + 1) / ny) * span
                polys.append([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
                fcols.append((rgba[0], rgba[1], rgba[2], float(rgba[3]) * 0.97))
    if polys:
        coll = Poly3DCollection(polys, facecolors=fcols, edgecolors="none", linewidths=0.0, shade=False)
        ax.add_collection3d(coll)


def draw_sigmoid_surface(
    ax,
    mesh: SigmoidMesh,
    Z: np.ndarray,
    *,
    cmap=None,
    alpha: float = 0.32,
) -> Any:
    return ax.plot_surface(
        mesh.ST,
        mesh.EL,
        Z,
        alpha=alpha,
        cmap=cmap if cmap is not None else CMAP,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
