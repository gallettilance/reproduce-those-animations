import io
import matplotlib as mpl
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
st_axis = np.linspace(xlim[0], xlim[1], 220)
el_axis = np.linspace(ylim[0], ylim[1], 220)
ST, EL = np.meshgrid(st_axis, el_axis)
DIFF = ST - EL
P_PASS = sigmoid(DIFF)
P_FAIL = sigmoid(-DIFF)
cvals  = [0, .5, 1]
colors = ['red', 'white', 'green']
norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
CMAP = mpl.colors.LinearSegmentedColormap.from_list("", tuples, 100)

# Threshold curve where ST=EL and sigmoid(0)=1/2
diag_line = np.linspace(max(xlim[0], ylim[0]), min(xlim[1], ylim[1]), 220)
threshold_half = np.full_like(diag_line, 0.5)

# Reference cross-section for readability (fixed exam length)
el_ref = np.full_like(st_axis, np.mean(ylim))

all_study = study_real
all_exam = exam_real
all_diff = all_study - all_exam

# Match the red/green palette used in neural-networks.ipynb (21.gif section)
SIG_PASS_COLOR = "green"
SIG_FAIL_COLOR = "red"


def style_sigmoid_axes(
    ax,
    az,
    elev=26,
    hide_z=False,
    exam_label_2d=False,
    diag_prob_scale=1.0,
):
    _dps = float(diag_prob_scale)
    ax.plot(
        diag_line,
        diag_line,
        _dps * threshold_half,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    if hide_z:
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.tick_params(axis="z", labelsize=0, colors="none")
    else:
        ax.tick_params(axis="z", labelsize=FONT_SIZE)
    ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=12.5)
    if exam_label_2d:
        ax.set_ylabel("")
        ax.text2D(
            -0.2,
            0.5,
            "Exam length (hours)",
            transform=ax.transAxes,
            fontsize=AXIS_LABEL_SIZE,
            rotation=90,
            va="center",
            ha="center",
        )
    else:
        ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=12.5)
    ax.set_zlabel("" if hide_z else "Probability", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, 1)
    ax.view_init(elev=elev, azim=az)


def scatter_whole_data_by_class(ax, prob_values, rotate_icons_180=False):
    """Draw pass/fail outcome icons as flat RGBA quads parallel to the XY plane at each z."""
    xr = float(xlim[1] - xlim[0])
    yr = float(ylim[1] - ylim[0])
    span = min(xr, yr) * 0.045
    nx, ny = 18, 18
    prob_values = np.asarray(prob_values, dtype=float)
    polys = []
    fcols = []
    for k in range(len(all_study)):
        s = float(all_study[k])
        e = float(all_exam[k])
        z = float(prob_values[k])
        img_arr = CHECK_ICON if int(y_real[k]) == 1 else CROSS_ICON
        im = Image.fromarray(np.asarray(img_arr, dtype=np.uint8), mode="RGBA")
        im = im.resize((nx, ny), Image.LANCZOS)
        if rotate_icons_180:
            # Horizontal flip then 180° image rotation (re-sized back to nx×ny for quad grid)
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(
                180,
                expand=True,
                resample=Image.BICUBIC,
                fillcolor=(255, 255, 255, 0),
            )
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
        coll = Poly3DCollection(
            polys,
            facecolors=fcols,
            edgecolors="none",
            linewidths=0.0,
            shade=False,
        )
        ax.add_collection3d(coll)


ROTATION_FRAMES = 120
ROTATION_MS = 32
SIG3D_FIGSIZE = EXPORT_FIGSIZE
SIG3D_DPI = EXPORT_DPI
SIG3D_ADJ = EXPORT_ADJ
SIG3D_ROT_STEP_DEG = 0.5
SIG3D_AZ0 = 25.0
SIG3D_ORBIT_AZ = SIG3D_AZ0 + np.arange(0.0, 360.0, SIG3D_ROT_STEP_DEG, dtype=float)
Z_flat = np.zeros_like(ST, dtype=float)
zeros_pts = np.zeros_like(all_diff, dtype=float)

from matplotlib.colors import to_rgb

extra_angles = SIG3D_ORBIT_AZ
_z_pass_pts = sigmoid(all_diff)
_z_fail_pts = sigmoid(-all_diff)
_last_az_pass = float(extra_angles[-1])

SIG3D_TOP_ELEV0 = 26.0
SIG3D_TOP_ELEV1 = 89.0
TOPDOWN_FRAMES = max(2, int(np.round((SIG3D_TOP_ELEV1 - SIG3D_TOP_ELEV0) / SIG3D_ROT_STEP_DEG)) + 1)
TOPDOWN_TURN_FRAMES = max(2, int(np.round(32.0 / SIG3D_ROT_STEP_DEG)) + 1)
TOPDOWN_END_HOLD_FRAMES = 12
topdown_elevations = np.linspace(SIG3D_TOP_ELEV0, SIG3D_TOP_ELEV1, TOPDOWN_FRAMES)
topdown_azimuth = 180 + 58
counterclockwise_turn = np.linspace(topdown_azimuth, topdown_azimuth + 32.0, TOPDOWN_TURN_FRAMES)
SIG3D_MORPH_FRAMES = 100
SIG3D_FLAT_HOLD = 16


def sig3d_frame_png(fig):
    fig.subplots_adjust(**SIG3D_ADJ)
    return fig_to_image(fig, tight_layout=False)


SIG2D_FIGSIZE = EXPORT_FIGSIZE
SIG2D_DPI = EXPORT_DPI
SIG2D_X = np.linspace(-7.0, 7.0, 900)
SIG2D_Y = sigmoid(SIG2D_X)
SIG2D_Y_NEG = sigmoid(-SIG2D_X)
SIG2D_COLOR = "#2ca02c"  # green (same family as pass / 3D hints)
SIG2D_MIRROR_PASS = "#2ca02c"
SIG2D_MIRROR_FAIL = "#d62728"
SIG2D_ADJ = dict(left=0.11, right=0.97, top=0.94, bottom=0.12)

from matplotlib import lines as mlines
from matplotlib.ticker import FixedFormatter, FixedLocator


def _finalize_sig2d(fig):
    fig.subplots_adjust(**SIG2D_ADJ)


def _save_sig2d(fig, filename):
    _finalize_sig2d(fig)
    fig.savefig(
        OUTPUT_DIR / filename,
        format="png",
        dpi=SIG2D_DPI,
        bbox_inches=None,
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(fig)


def _sig2d_axes_and_crosshairs(ax):
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$z = \mathrm{ST}-\mathrm{EL}$", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel(r"$\sigma(z)$", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    ax.axhline(0.5, color="black", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.set_autoscale_on(False)
    _sig2d_apply_fixed_ticks(ax)


def _plot_sigmoid_2d_base(ax):
    _sig2d_axes_and_crosshairs(ax)
    ax.plot(SIG2D_X, SIG2D_Y, color=SIG2D_COLOR, linewidth=2.2)


SIG2D_LEGEND_FS = LEGEND_SIZE + 6
SIG2D_LEGEND_48 = r"$\sigma(z)=\frac{e^{z}}{e^{z}+e^{0}}$"


def _legend_sig2d(ax, label_math):
    h = [mlines.Line2D([], [], linestyle="none", linewidth=0, color="none", label=label_math)]
    ax.legend(
        handles=h,
        loc="upper left",
        prop={"size": SIG2D_LEGEND_FS},
        frameon=True,
        borderaxespad=0.9,
        labelspacing=0.9,
    )


def _sig2d_frame_png(fig):
    _finalize_sig2d(fig)
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=SIG2D_DPI,
        bbox_inches=None,
        pad_inches=SAVE_PAD_INCHES,
    )
    buf.seek(0)
    im = Image.open(buf).convert("RGB").copy()
    buf.close()
    return im


def _sig2d_smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _sig2d_draw_reveal(ax, x_right):
    """Axes, crosshairs, legend (same as 48), and sigmoid curve for z <= x_right."""
    ax.clear()
    _sig2d_axes_and_crosshairs(ax)
    _legend_sig2d(ax, SIG2D_LEGEND_48)
    m = SIG2D_X <= x_right
    if np.count_nonzero(m) >= 2:
        ax.plot(
            SIG2D_X[m],
            SIG2D_Y[m],
            color=SIG2D_COLOR,
            linewidth=2.2,
            solid_capstyle="round",
        )


SIG2D_REVEAL_MS = 40
SIG2D_REVEAL_HOLD = 16
SIG2D_REVEAL_ANIM = 88
SIG2D_REVEAL_TAIL = 18
SIG2D_HIGHLIGHT_MS = 42
SIG2D_HIGHLIGHT_HOLD = 14
SIG2D_HIGHLIGHT_STAGE = 10

# Fixed layout for 62–66: integer ticks on z; σ(z) axis uses 0 and 1 at the ends (not decimal σ(±6)).
SIG2D_HIGHLIGHT_XTICKS = np.arange(-7, 8, dtype=float)
SIG2D_HIGHLIGHT_YTICKS = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], dtype=float)


def _sig2d_frame_full_green(ax):
    ax.clear()
    _sig2d_axes_and_crosshairs(ax)
    _legend_sig2d(ax, SIG2D_LEGEND_48)
    ax.plot(
        SIG2D_X,
        SIG2D_Y,
        color=SIG2D_COLOR,
        linewidth=2.2,
        solid_capstyle="round",
        zorder=3,
    )


def _sig2d_highlight_ytick_label(v):
    v = float(v)
    if abs(v) < 1e-10:
        return "0"
    if abs(v - 1.0) < 1e-10:
        return "1"
    return f"{v:.1f}"


def _sig2d_apply_fixed_ticks(ax):
    """Match 62: fixed integer z ticks; σ(z) ticks 0,0.2,…,1 with ends labeled 0 and 1 only."""
    ax.minorticks_off()
    xlabs = [str(int(t)) for t in SIG2D_HIGHLIGHT_XTICKS]
    ylabs = [_sig2d_highlight_ytick_label(float(t)) for t in SIG2D_HIGHLIGHT_YTICKS]
    ax.xaxis.set_major_locator(FixedLocator(SIG2D_HIGHLIGHT_XTICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(xlabs))
    ax.yaxis.set_major_locator(FixedLocator(SIG2D_HIGHLIGHT_YTICKS))
    ax.yaxis.set_major_formatter(FixedFormatter(ylabs))
    ax.tick_params(axis="both", which="minor", left=False, bottom=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)


def _sig2d_highlight_freeze_ticks(ax):
    ax.set_autoscale_on(False)
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(0.0, 1.0)
    _sig2d_apply_fixed_ticks(ax)


def _bold_nearest_ticklabel(ax, axis, value, tol):
    labs = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    for lab, tk in zip(labs, ticks):
        w = "bold" if abs(float(tk) - float(value)) <= tol else "normal"
        lab.set_fontweight(w)
        lab.set_fontsize(FONT_SIZE)


def _sig2d_gif_three_highlights():
    x_left = float(SIG2D_X[0])
    specs = [
        (6.0, "63_sigmoid_2d_hint_plus6h_prob_near1.gif"),
        (-6.0, "64_sigmoid_2d_hint_deficit6h_prob_near0.gif"),
        (0.0, "65_sigmoid_2d_threshold_st_eq_el_prob_half.gif"),
    ]
    for z0, fn in specs:
        y0 = float(sigmoid(z0))
        if abs(float(z0) - 6.0) < 1e-9:
            y_bold_ref = 1.0
        elif abs(float(z0) + 6.0) < 1e-9:
            y_bold_ref = 0.0
        else:
            y_bold_ref = y0
        frames = []
        fig, ax = plt.subplots(figsize=SIG2D_FIGSIZE)
        try:
            for _ in range(SIG2D_HIGHLIGHT_HOLD):
                _sig2d_frame_full_green(ax)
                _sig2d_highlight_freeze_ticks(ax)
                frames.append(_sig2d_frame_png(fig))
            for st in range(4):
                for _ in range(SIG2D_HIGHLIGHT_STAGE):
                    _sig2d_frame_full_green(ax)
                    _sig2d_highlight_freeze_ticks(ax)
                    if st >= 1:
                        ax.plot([z0, z0], [0.0, y0], color="black", linewidth=1.9, zorder=5)
                    if st >= 2:
                        ax.plot([x_left, z0], [y0, y0], color="black", linewidth=1.9, zorder=5)
                    _bold_nearest_ticklabel(ax, "x", z0, tol=1e-6)
                    if st >= 3:
                        _bold_nearest_ticklabel(ax, "y", y_bold_ref, tol=0.05)
                    frames.append(_sig2d_frame_png(fig))
            for _ in range(SIG2D_HIGHLIGHT_HOLD):
                _sig2d_frame_full_green(ax)
                _sig2d_highlight_freeze_ticks(ax)
                ax.plot([z0, z0], [0.0, y0], color="black", linewidth=1.9, zorder=5)
                ax.plot([x_left, z0], [y0, y0], color="black", linewidth=1.9, zorder=5)
                _bold_nearest_ticklabel(ax, "x", z0, tol=1e-6)
                _bold_nearest_ticklabel(ax, "y", y_bold_ref, tol=0.05)
                frames.append(_sig2d_frame_png(fig))
        finally:
            plt.close(fig)
        save_gif(frames, fn, duration=SIG2D_HIGHLIGHT_MS)

def _sig2d_draw_mirror_reveal_frame(ax, x_fail_right):
    """Green sigma(z) always; red sigma(-z) revealed for z <= x_fail_right."""
    ax.clear()
    _sig2d_axes_and_crosshairs(ax)
    ax.plot(
        SIG2D_X,
        SIG2D_Y,
        color=SIG2D_MIRROR_PASS,
        linewidth=2.35,
        solid_capstyle="round",
        zorder=3,
    )
    m = SIG2D_X <= x_fail_right
    if np.count_nonzero(m) >= 2:
        ax.plot(
            SIG2D_X[m],
            SIG2D_Y_NEG[m],
            color=SIG2D_MIRROR_FAIL,
            linewidth=2.35,
            solid_capstyle="round",
            zorder=4,
        )
    h_p = mlines.Line2D([], [], color=SIG2D_MIRROR_PASS, linewidth=2.6, label=r"$\sigma(z)$")
    h_f = mlines.Line2D([], [], color=SIG2D_MIRROR_FAIL, linewidth=2.6, label=r"$\sigma(-z)$")
    ax.legend(handles=[h_p, h_f], loc="upper right", fontsize=LEGEND_SIZE, frameon=True, borderaxespad=0.55)
    _sig2d_highlight_freeze_ticks(ax)


SIG2D_MIRROR_REVEAL_MS = SIG2D_REVEAL_MS
