# --- 74 — half-plane → shrink → 3D → −45° az (top) → tilt → spin (360°) → de-amplify → ~½
Z_27 = sigmoid(0.5 * ST - 0.5 * EL)
z_pts_27 = sigmoid(0.5 * all_diff)

def _Z27_gain_surface(g):
    """σ(0.5·g·(ST−EL)); g=1 matches Z_27; g>1 sharpens; g→0 flattens to ~½."""
    return sigmoid(0.5 * float(g) * DIFF)


def _Z27_gain_scatter(g):
    return sigmoid(0.5 * float(g) * all_diff)


# Sharpened field for all of `74` until the post-spin de-amplify (matches former peak gain).
_74_GAIN_HI = 5.0
_74_GAIN_PLAY = float(_74_GAIN_HI)
Z_74 = _Z27_gain_surface(_74_GAIN_PLAY)
z_pts_74 = _Z27_gain_scatter(_74_GAIN_PLAY)
_74_GAIN_LO = 0.06  # de-amplify target: σ(0.5·g·Δ) → ~½ everywhere


def _draw_74_topdown_on_ax(ax, zm):
    """2D σ field + threshold + icons (same elements as static ``76_sigmoid_colormap_topdown_2d.png``)."""
    ax.contourf(
        ST,
        EL,
        zm,
        levels=np.linspace(0.0, 1.0, 45),
        cmap=CMAP,
        vmin=0,
        vmax=1,
        antialiased=True,
        alpha=0.32,
    )
    add_threshold_line(ax, shift=midpoint_shift, label="ST - EL = 0", color="black", linewidth=1)
    for s, e, lbl in zip(all_study, all_exam, y_real):
        add_outcome_icon(ax, float(s), float(e), passed=bool(lbl), zoom=0.2, alpha=0.95)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.grid(alpha=0.12)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    add_combined_legend(ax, loc="upper left")


# Default ``plt.subplots`` margins only (matches ``76`` / other static 2D dataset exports; no EXPORT_ADJ).
_fig_probe2d, _ax_probe2d = plt.subplots(figsize=EXPORT_FIGSIZE)
_draw_74_topdown_on_ax(_ax_probe2d, Z_74)
_ax_probe2d.set_aspect("auto")
_fig_probe2d.canvas.draw()
_pos_full_2d = _ax_probe2d.get_position()
_74_BB0_FULL = (
    float(_pos_full_2d.x0),
    float(_pos_full_2d.y0),
    float(_pos_full_2d.width),
    float(_pos_full_2d.height),
)
plt.close(_fig_probe2d)

_W0_2d = float(_pos_full_2d.width)
_H0_2d = float(_pos_full_2d.height)
_cx_2d = float(_pos_full_2d.x0 + 0.5 * _W0_2d)
_cy_2d = float(_pos_full_2d.y0 + 0.5 * _H0_2d)


# Shrink with x slightly tighter than y (independent end scales), then cross-fade to 3D.
_74_SHRINK_END_SCALE_W = 0.45  # width scale at end (center fixed); lower = narrower
_74_SHRINK_END_SCALE_H = 0.72  # height scale at end


def _shrink_bbox_uniform(u_lin, W0, H0, cx, cy, end_scale_w, end_scale_h):
    """(W0,H0) → (sw·W0, sh·H0) with sw, sh easing from 1 to end_scale_* on one timeline."""
    u_lin = float(np.clip(u_lin, 0.0, 1.0))
    p = _sig2d_smoothstep(_ease_top27(u_lin))
    sw = 1.0 + (float(end_scale_w) - 1.0) * p
    sh = 1.0 + (float(end_scale_h) - 1.0) * p
    w = W0 * sw
    h = H0 * sh
    return mpl.transforms.Bbox.from_bounds(cx - 0.5 * w, cy - 0.5 * h, w, h)


_LIFT2D_SHRINK_FRAMES = 120


def _ease_top27(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _pil_topdown_27style_masked(zm, ax_rect=None):
    """Always ``EXPORT_FIGSIZE`` + ``add_axes`` so reveal and shrink share one smooth layout path (like ``38``)."""
    fig = plt.figure(figsize=EXPORT_FIGSIZE)
    if ax_rect is None:
        bb = _74_BB0_FULL
    else:
        bb = tuple(float(x) for x in ax_rect.bounds)
    ax = fig.add_axes(bb)
    _draw_74_topdown_on_ax(ax, zm)
    ax.set_aspect("auto")
    return fig_to_image(fig)


def _strip_74_3d_axis_labels(ax):
    """Remove axis titles (and exam text2D) after `style_sigmoid_axes` — keeps `74` self-contained."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for t in list(ax.texts):
        try:
            if "Exam length" in t.get_text():
                t.remove()
        except Exception:
            pass


def _frame_3d_z27_surface(
    Z_surf, z_scatter, az, elev, hide_z, exam_label_2d, diag_scale=1.0, hide_axis_labels=True
):
    fig = plt.figure(figsize=SIG3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        ST,
        EL,
        Z_surf,
        alpha=0.32,
        cmap=CMAP,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    scatter_whole_data_by_class(ax, z_scatter, rotate_icons_180=True)
    style_sigmoid_axes(
        ax,
        float(az),
        elev=float(elev),
        hide_z=hide_z,
        exam_label_2d=exam_label_2d,
        diag_prob_scale=diag_scale,
    )
    if hide_axis_labels:
        _strip_74_3d_axis_labels(ax)
    return sig3d_frame_png(fig)


_74_BLEND_FRAMES = 28
_74_DIP_FRAMES = 72
_74_FULL_SPIN_FRAMES = 120
_74_AMP_DOWN_FRAMES = 88
_74_TAIL_HOLD = 10
_74_PRE_TILT_AZ_DEG = -45.0  # azimuth at top elevation before tilt (then orbit makes up +45°)
_74_PRE_TILT_FRAMES = 32
_74_ORBIT_DEG = 360.0  # −45° pre-tilt + 360° spin → same final az as baseline +315° from pure top

_az_top_end = float(counterclockwise_turn[-1])

frames_74 = []

# (1) Half-plane reveal (same cadence as `34_sigmoid_colormap_topdown_reveal.gif`)
_dmax27 = float(np.max(DIFF))
_dmin27 = float(np.min(DIFF))
_eps27 = 0.05 * max(_dmax27, abs(_dmin27))
_NP27, _NF27 = 52, 52
_HOLDPF27, _HOLDA27 = 10, 14

for j in range(_NP27):
    u = _eps27 + (_dmax27 - _eps27) * _ease_top27(j / max(_NP27 - 1, 1))
    zm = np.where((DIFF > 0) & (DIFF <= u), Z_74, np.nan)
    frames_74.append(_pil_topdown_27style_masked(zm))
zm_pass_done = np.where(DIFF > 0, Z_74, np.nan)
for _ in range(_HOLDPF27):
    frames_74.append(_pil_topdown_27style_masked(zm_pass_done))
for j in range(_NF27):
    lo = -_eps27 + (_dmin27 + _eps27) * _ease_top27(j / max(_NF27 - 1, 1))
    zm = np.where(
        DIFF > 0,
        Z_74,
        np.where((DIFF >= lo) & (DIFF < 0), Z_74, np.nan),
    )
    frames_74.append(_pil_topdown_27style_masked(zm))
for _ in range(_HOLDA27):
    frames_74.append(_pil_topdown_27style_masked(Z_74))

# (2) Shrink (x slightly more than y), then 3D top
for j in range(_LIFT2D_SHRINK_FRAMES):
    u_lin = j / max(_LIFT2D_SHRINK_FRAMES - 1, 1)
    rect = _shrink_bbox_uniform(
        u_lin, _W0_2d, _H0_2d, _cx_2d, _cy_2d, _74_SHRINK_END_SCALE_W, _74_SHRINK_END_SCALE_H
    )
    frames_74.append(_pil_topdown_27style_masked(Z_74, ax_rect=rect))

im2d_last = frames_74[-1].copy()
im3d_top = _frame_3d_z27_surface(
    Z_74,
    z_pts_74,
    _az_top_end,
    SIG3D_TOP_ELEV1,
    hide_z=True,
    exam_label_2d=True,
    diag_scale=1.0,
)

# (3) Cross-fade 2D (shrunk) → 3D top-down
_nb = max(_74_BLEND_FRAMES, 2)
for k in range(_nb):
    u = _sig2d_smoothstep(k / float(_nb - 1))
    frames_74.append(Image.blend(im2d_last, im3d_top, u))

# (4) Brief hold on top-down 3D (already at im3d_top; duplicate a few frames for readability)
for _ in range(8):
    frames_74.append(im3d_top.copy())

# (4b) −45° azimuth at top elevation before tilt; orbit adds +45° vs a plain 315° pass (360° total) to match final bearing
_npre = max(_74_PRE_TILT_FRAMES, 2)
for j in range(_npre):
    u = _sig2d_smoothstep(j / float(_npre - 1))
    az = _az_top_end + _74_PRE_TILT_AZ_DEG * u
    frames_74.append(
        _frame_3d_z27_surface(
            Z_74,
            z_pts_74,
            float(az),
            SIG3D_TOP_ELEV1,
            hide_z=True,
            exam_label_2d=True,
            diag_scale=1.0,
        )
    )
_az74_post_top = _az_top_end + _74_PRE_TILT_AZ_DEG

# (5) Smooth tilt: top-down → oblique / side (same thresholds as `72` elevation pass)
_nd = max(_74_DIP_FRAMES, 2)
for j in range(_nd):
    u = _sig2d_smoothstep(j / float(_nd - 1))
    elev = SIG3D_TOP_ELEV1 + (SIG3D_TOP_ELEV0 - SIG3D_TOP_ELEV1) * u
    near_top = elev >= 87.5
    exam2d = elev >= 78.0
    frames_74.append(
        _frame_3d_z27_surface(
            Z_74,
            z_pts_74,
            _az74_post_top,
            float(elev),
            hide_z=near_top,
            exam_label_2d=exam2d,
            diag_scale=1.0,
        )
    )

# (6) One full rotation at side view (elevation fixed at opening angle)
_nspin = max(_74_FULL_SPIN_FRAMES, 1)
for j in range(1, _nspin + 1):
    az = _az74_post_top + _74_ORBIT_DEG * float(j) / float(_nspin)
    frames_74.append(
        _frame_3d_z27_surface(
            Z_74,
            z_pts_74,
            az,
            SIG3D_TOP_ELEV0,
            hide_z=False,
            exam_label_2d=False,
            diag_scale=1.0,
        )
    )

_az_spin_end = _az74_post_top + _74_ORBIT_DEG

# (7) De-amplify after spin: g from G_play down to G_LO so σ → ~½ everywhere

_du_de = max(_74_AMP_DOWN_FRAMES - 1, 1)
for j in range(_74_AMP_DOWN_FRAMES):
    u = _sig2d_smoothstep(j / float(_du_de))
    g = _74_GAIN_PLAY + (_74_GAIN_LO - _74_GAIN_PLAY) * u
    Zg = _Z27_gain_surface(g)
    zg = _Z27_gain_scatter(g)
    frames_74.append(
        _frame_3d_z27_surface(
            Zg,
            zg,
            _az_spin_end,
            SIG3D_TOP_ELEV0,
            hide_z=False,
            exam_label_2d=False,
            diag_scale=1.0,
        )
    )

Z_flat = _Z27_gain_surface(_74_GAIN_LO)
z_flat = _Z27_gain_scatter(_74_GAIN_LO)
for _ in range(_74_TAIL_HOLD):
    frames_74.append(
        _frame_3d_z27_surface(
            Z_flat,
            z_flat,
            _az_spin_end,
            SIG3D_TOP_ELEV0,
            hide_z=False,
            exam_label_2d=False,
            diag_scale=1.0,
        )
    )

save_gif(
    frames_74,
    "77_from27_topdown_reveal_lift_3d_reverse64.gif",
    duration=ROTATION_MS,
)
