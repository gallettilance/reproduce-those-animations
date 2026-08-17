SIG3D_MS = _gif_dur(34)
SIG3D_MS_3D = max(4, SIG3D_MS // 2)  # 2× playback vs ``SIG3D_MS`` — only ``ch2_04`` σ surface 3D (``ch2_04b`` 2D GIFs still use ``SIG3D_MS``).


def _style_3d(ax, *, elev, azim, zlabel=r"$\sigma(\mathrm{logit})$"):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=11)
    ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=11)
    ax.set_zlabel(zlabel, fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 1)


def frame_3d_scaled(k_scale, *, elev=32.0, azim=-152.0):
    w_st, w_el, b = k_scale * W_ST0, k_scale * W_EL0, k_scale * B0
    Zs = sigmoid(logits_plane(w_st, w_el, b, ST_3D, EL_3D))
    fig = plt.figure(figsize=EXPORT_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(**SIG3D_ADJ)
    ax.plot_surface(ST_3D, EL_3D, Zs, cmap=CMAP_GD, vmin=0, vmax=1, alpha=0.42, linewidth=0, antialiased=True, shade=False)
    zs = sigmoid(logits_plane(w_st, w_el, b, study_sep, exam_sep))
    for i in range(n_sep):
        c = PASS_COLOR if y_sep[i] else FAIL_COLOR
        ax.scatter([study_sep[i]], [exam_sep[i]], [zs[i]], color=c, s=55, depthshade=True, zorder=6)
    bxy = boundary_line_xy(w_st, w_el, b, float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
    if bxy is not None:
        bx, by = bxy
        z0 = np.full_like(bx, 0.5)
        ax.plot(bx, by, z0, "k--", linewidth=1.4, zorder=4)
    _style_3d(ax, elev=elev, azim=azim)
    return fig_to_image(fig)


k_up = np.linspace(0.35, 2.4, _smooth_n(28))
k_down = np.linspace(2.4, 0.35, _smooth_n(28))
k_hold = np.full(_smooth_n(10), 0.35)
k_seq = np.concatenate([k_up, k_down, k_hold])


def _build_ch2_04b_frames(*, legend_mode):
    frames = []
    for k in k_seq:
        kk = float(k)
        w_st, w_el, b = kk * W_ST0, kk * W_EL0, kk * B0
        if legend_mode == "values":
            leg = legend_linear_equation_values(w_st, w_el, b, difference_form=True)
            lg_tex = False
        else:
            leg = legend_linear_equation_style(w_st, w_el, b, "both", mag_ref=kk, difference_form=True)
            lg_tex = True
        fig, ax = plt.subplots(figsize=EXPORT_FIGSIZE)
        draw_probability_panel(ax, w_st, w_el, b, leg, show_contour=True, st_grid=ST_A, el_grid=EL_A, legend_tex=lg_tex)
        frames.append(fig_to_image(fig))
    return frames


nph = _smooth_n(22)
seg_st_1 = np.linspace(1.0, 0.01, nph)
seg_st_2 = np.linspace(0.01, 2.0, nph)
seg_st_3 = np.linspace(2.0, 1.0, max(nph // 2, _smooth_n(10) // GIF_SMOOTH_FACTOR))
seg_el_1 = np.linspace(-1.0, 1.0, nph)
seg_el_2 = np.linspace(1.0, -1.0, nph)


def _frame_ws_we_bb(ws, we, bb, legend_label, *, legend_tex=False):
    wrong_now = mistake_mask(study_sep, exam_sep, y_sep, float(ws), float(we), float(bb))
    fig, ax = plt.subplots(figsize=EXPORT_FIGSIZE)
    draw_probability_panel(
        ax,
        float(ws),
        float(we),
        float(bb),
        legend_label,
        show_contour=True,
        st_grid=ST_A,
        el_grid=EL_A,
        highlight_mask=wrong_now,
        legend_tex=legend_tex,
    )
    return fig_to_image(fig)


def _build_ch2_05_frames(*, legend_mode):
    frames = []
    for ws in np.concatenate([seg_st_1, seg_st_2, seg_st_3]):
        we, bb = W_EL0, B0
        if legend_mode == "values":
            leg = legend_linear_equation_values(ws, we, bb, difference_form=True)
            frames.append(_frame_ws_we_bb(ws, we, bb, leg))
        else:
            leg = legend_linear_equation_style(ws, we, bb, "st", mag_ref=abs(float(ws)), difference_form=True)
            frames.append(_frame_ws_we_bb(ws, we, bb, leg, legend_tex=True))
    for we in np.concatenate([seg_el_1, seg_el_2]):
        ws, bb = W_ST0, B0
        if legend_mode == "values":
            leg = legend_linear_equation_values(ws, we, bb, difference_form=True)
            frames.append(_frame_ws_we_bb(ws, we, bb, leg))
        else:
            leg = legend_linear_equation_style(ws, we, bb, "el", mag_ref=abs(float(we)), difference_form=True)
            frames.append(_frame_ws_we_bb(ws, we, bb, leg, legend_tex=True))
    return frames


n_seg = _smooth_n(32)
b_path = np.concatenate([np.linspace(0.0, -2.0, n_seg), np.linspace(-2.0, 2.0, n_seg), np.linspace(2.0, 0.0, n_seg)])


def _build_ch2_06_frames(*, legend_mode):
    frames = []
    for bb in b_path:
        bb = float(bb)
        wrong_now = mistake_mask(study_sep, exam_sep, y_sep, W_ST0, W_EL0, bb)
        if legend_mode == "values":
            leg = legend_linear_equation_values(W_ST0, W_EL0, bb)
            lg_tex = False
        else:
            leg = legend_linear_equation_style(W_ST0, W_EL0, bb, "b", mag_ref=abs(bb))
            lg_tex = True
        fig, ax = plt.subplots(figsize=EXPORT_FIGSIZE)
        draw_probability_panel(ax, W_ST0, W_EL0, bb, leg, show_contour=True, st_grid=ST_A, el_grid=EL_A, highlight_mask=wrong_now, legend_tex=lg_tex)
        frames.append(fig_to_image(fig))
    return frames


alpha0 = np.pi / 4.0
d_alpha = 0.65
alphas = np.concatenate([np.linspace(alpha0 - d_alpha, alpha0 + d_alpha, _smooth_n(36)), np.linspace(alpha0 + d_alpha, alpha0 - d_alpha, _smooth_n(36))])


def w_from_alpha(a):
    return float(np.cos(a)), float(-np.sin(a))


def frame_rotate_3d(a, *, elev=36.0, azim=-152.0):
    ws, we = w_from_alpha(a)
    Zs = sigmoid(logits_plane(ws, we, 0.0, ST_3D, EL_3D))
    fig = plt.figure(figsize=EXPORT_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(**SIG3D_ADJ)
    ax.plot_surface(ST_3D, EL_3D, Zs, cmap=CMAP_GD, vmin=0, vmax=1, alpha=0.4, linewidth=0, antialiased=True, shade=False)
    zs = sigmoid(logits_plane(ws, we, 0.0, study_sep, exam_sep))
    for i in range(n_sep):
        c = PASS_COLOR if y_sep[i] else FAIL_COLOR
        ax.scatter([study_sep[i]], [exam_sep[i]], [zs[i]], color=c, s=52, zorder=6)
    bxy = boundary_line_xy(ws, we, 0.0, float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
    if bxy is not None:
        bx, by = bxy
        ax.plot(bx, by, np.full_like(bx, 0.5), "k--", linewidth=1.2, zorder=4)
    _style_3d(ax, elev=elev, azim=azim)
    return fig_to_image(fig)


def frame_shift_3d(bb, *, elev=36.0, azim=-152.0):
    Zs = sigmoid(logits_plane(W_ST0, W_EL0, float(bb), ST_3D, EL_3D))
    fig = plt.figure(figsize=EXPORT_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(**SIG3D_ADJ)
    ax.plot_surface(ST_3D, EL_3D, Zs, cmap=CMAP_GD, vmin=0, vmax=1, alpha=0.4, linewidth=0, antialiased=True, shade=False)
    zs = sigmoid(logits_plane(W_ST0, W_EL0, float(bb), study_sep, exam_sep))
    for i in range(n_sep):
        c = PASS_COLOR if y_sep[i] else FAIL_COLOR
        ax.scatter([study_sep[i]], [exam_sep[i]], [zs[i]], color=c, s=52, zorder=6)
    bxy = boundary_line_xy(W_ST0, W_EL0, float(bb), float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
    if bxy is not None:
        bx, by = bxy
        ax.plot(bx, by, np.full_like(bx, 0.5), "k--", linewidth=1.2, zorder=4)
    _style_3d(ax, elev=elev, azim=azim)
    return fig_to_image(fig)


def _parallel_lines_frame(g, *, with_contour, note=None, legend_mode="values", smooth=False, with_knobs=False, prev=None, coeff_scale=1.0):
    _ = (legend_mode, smooth)  # retained for **ch2_09**–**ch2_16** call sites (values vs style pairs).
    if with_knobs:
        fig, ax, axes_k = _ch2_figure_main_and_knob_axes()
    else:
        fig, ax = plt.subplots(figsize=EXPORT_FIGSIZE)
    g = float(g)
    c = float(coeff_scale)
    ws, we, bb = c * g * W_ST0, c * g * W_EL0, 0.0
    if with_contour:
        Z = sigmoid(logits_plane(ws, we, bb, ST_A, EL_A))
        _ch2_sigma_contourf(ax, ST_A, EL_A, Z, zorder=1)
    xs = np.linspace(xlim[0], xlim[1], 400)
    L = 1
    shift = (float(L) / g) if abs(g) > 1e-9 else float(L)
    ys = xs - shift
    m = (ys >= ylim[0]) & (ys <= ylim[1])
    ax.plot(xs[m], ys[m], "--", color=PARALLEL_COLORS[1], linewidth=1.75, label="1 unit of work", zorder=3)
    draw_dataset(ax, study_sep, exam_sep, y_sep)
    if note:
        ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=NOTE_SIZE)
    add_combined_legend(ax, loc="upper left")
    if with_knobs:
        knob_rgbs, canvas_sides = _ch2_knob_strip_asset_pack()
        # **Gain ×2**: ``wᵢ = c·g·W_*`` (`c` = ``coeff_scale``). Dials still use **360°/unit** vs **chapter unit** ``(W_ST0,W_EL0,B0)`` so e.g. ``c=½``, ``g=1`` ⇒ ``w₁`` is **½** unit below the dial's **1** reference (not neutral).
        _ch2_draw_pil_knob_row(
            ws,
            we,
            bb,
            prev,
            "both",
            fig,
            axes_k,
            knob_rgbs,
            canvas_sides,
            knob_angle_refs=(float(W_ST0), float(W_EL0), float(B0)),
        )
    dpi_k = min(int(EXPORT_DPI), 110) if with_knobs else None
    return fig_to_image(fig, dpi=dpi_k)


PARALLEL_COLORS = ["black", "#1f77b4", "#2ca02c", "#bcbd22"]
_ph = _smooth_n(6)
_g_up = np.linspace(1.0, 2.0, _smooth_n(40))
_g_dn = np.linspace(2.0, 1.0, _smooth_n(40))


def _parallel_intro_frames(*, with_contour, legend_mode, with_knobs=False):
    prev = None
    out = []
    for _ in range(_ph):
        out.append(_parallel_lines_frame(1.0, with_contour=with_contour, legend_mode=legend_mode, with_knobs=with_knobs, prev=prev))
        prev = (float(W_ST0), float(W_EL0), 0.0)
    return out


def _parallel_step_frames(*, with_contour, legend_mode, with_knobs=False):
    prev = None
    a = []
    for _ in range(_smooth_n(6)):
        a.append(_parallel_lines_frame(1.0, with_contour=with_contour, legend_mode=legend_mode, with_knobs=with_knobs, prev=prev))
        prev = (float(W_ST0), float(W_EL0), 0.0)
    for _ in range(_smooth_n(8)):
        a.append(_parallel_lines_frame(2.0, with_contour=with_contour, legend_mode=legend_mode, with_knobs=with_knobs, prev=prev))
        prev = (2.0 * float(W_ST0), 2.0 * float(W_EL0), 0.0)
    return a


def _parallel_smooth_frames(gs, *, with_contour, legend_mode, with_knobs=False, coeff_scale=1.0):
    prev = None
    out = []
    c = float(coeff_scale)
    for g in np.asarray(gs, dtype=float):
        g = float(g)
        ws, we, bb = c * g * W_ST0, c * g * W_EL0, 0.0
        out.append(
            _parallel_lines_frame(
                g,
                with_contour=with_contour,
                legend_mode=legend_mode,
                smooth=True,
                with_knobs=with_knobs,
                prev=prev,
                coeff_scale=coeff_scale,
            )
        )
        prev = (ws, we, bb)
    return out



def _parallel_threshold_sigma_frame(idx, *, n, legend_mode="values", draw_unit_work_line=True, with_knobs=False, prev=None):
    """Sigma contour gain k_contour on (ST-EL), k_contour: 0.5 -> 4; threshold ST-EL=0 at midpoint_shift; legend k ST - k EL = 0 with k: 1 -> 2."""
    _ = legend_mode
    idx = int(idx)
    n = int(n)
    u = float(idx) / float(max(n - 1, 1))
    k_contour = 0.5 + u * (4.0 - 0.5)
    k_leg = 1.0 + u * (2.0 - 1.0)
    ws_c = k_contour * float(W_ST0)
    we_c = k_contour * float(W_EL0)
    bb_c = 0.0
    if with_knobs:
        fig, ax, axes_k = _ch2_figure_main_and_knob_axes()
    else:
        fig, ax = plt.subplots(figsize=EXPORT_FIGSIZE)
    Z = sigmoid(logits_plane(ws_c, we_c, bb_c, ST_A, EL_A))
    _ch2_sigma_contourf(ax, ST_A, EL_A, Z, zorder=1)
    if draw_unit_work_line:
        g = float(np.asarray(_g_up, dtype=float)[idx])
        xs = np.linspace(xlim[0], xlim[1], 400)
        L = 1
        shift = (float(L) / g) if abs(g) > 1e-9 else float(L)
        ys = xs - shift
        m = (ys >= ylim[0]) & (ys <= ylim[1])
        ax.plot(xs[m], ys[m], "--", color=PARALLEL_COLORS[1], linewidth=1.75, label="1 unit of work", zorder=3)
    draw_dataset(ax, study_sep, exam_sep, y_sep)
    ks = _trim_num_coeff(k_leg)
    leg_thr = rf"${ks}\,\mathrm{{ST}} - {ks}\,\mathrm{{EL}} = 0$"
    x0, x1 = float(xlim[0]), float(xlim[1])
    xt = np.linspace(x0, x1, 200)
    ax.plot(xt, xt - float(midpoint_shift), "--", color="black", linewidth=1.8, label=leg_thr, zorder=5)
    if with_knobs:
        knob_rgbs, canvas_sides = _ch2_knob_strip_asset_pack()
        _ch2_draw_pil_knob_row(
            ws_c,
            we_c,
            bb_c,
            prev,
            "both",
            fig,
            axes_k,
            knob_rgbs,
            canvas_sides,
            knob_angle_refs=(float(W_ST0), float(W_EL0), float(B0)),
        )
    add_combined_legend(ax, loc="upper left")
    dpi_k = min(int(EXPORT_DPI), 110) if with_knobs else None
    return fig_to_image(fig, dpi=dpi_k)


def _parallel_threshold_sigma_smooth_frames(*, legend_mode="values", draw_unit_work_line=True, with_knobs=False):
    n = len(_g_up)
    prev = None
    out = []
    for idx in range(n):
        out.append(
            _parallel_threshold_sigma_frame(
                idx,
                n=n,
                legend_mode=legend_mode,
                draw_unit_work_line=draw_unit_work_line,
                with_knobs=with_knobs,
                prev=prev,
            )
        )
        u = float(idx) / float(max(n - 1, 1))
        k_contour = 0.5 + u * (4.0 - 0.5)
        prev = (k_contour * float(W_ST0), k_contour * float(W_EL0), 0.0)
    return out

# --- Parallel “1 unit of work” + gain **×2** — knob variants ---
