
# --- ch4_02: likelihood(w1,w2) landscape — duo left, tall 3D right ---
CH3_LIK_W12_W_ST0 = float(CH3_SCRIPT_K1_W_ST)
CH3_LIK_W12_W_EL0 = float(CH3_SCRIPT_K1_W_EL)
CH3_LIK_W12_B0 = float(CH3_SCRIPT_K1_B)
CH3_LIK_W12_W1_LO = float(CH3_SCRIPT_W1_WIDE[0])
CH3_LIK_W12_W1_HI = float(CH3_SCRIPT_W1_WIDE[1])
CH3_LIK_W12_W2_LO = float(CH3_LIK_W12_W_EL0 - CH3_SCRIPT_K1_STOP_SWEEP_DELTA)
CH3_LIK_W12_W2_HI = float(CH3_LIK_W12_W_EL0 + CH3_SCRIPT_K1_STOP_SWEEP_DELTA)
CH3_LIK_W12_B_HALF = float(CH3_SCRIPT_K1_TIGHT_SWEEP_DELTA * 20.0)
CH3_LIK_W12_FIGSIZE = CH4_DUO_FIGSIZE
CH3_LIK_W12_WIDTH_RATIOS = (1.22, 1.42)
CH3_LIK_W12_GRID_N = 52 if not _CH3_DRAFT else 28
CH3_LIK_W12_GRID_N_COARSE = 22 if not _CH3_DRAFT else 14
CH3_LIK_W12_GRID_N_FINE = 72 if not _CH3_DRAFT else 36
CH3_LIK_W12_QUAD_MID = 0.0
CH3_LIK_W12_CURVE_LW = 2.4
# Split-complementary likelihood mesh (ch4_02/03) — deeper red vs 2D FAIL_COLOR.
CH4_LIK_SURFACE_COLOR = "#b71c1c"
CH4_LIK_SURFACE_ALPHA = 0.80
CH3_LIK_W12_SURFACE_ALPHA = CH4_LIK_SURFACE_ALPHA
CH3_LIK_W12_SLICE_ALPHA = 0.42 * 0.9
CH3_LIK_W12_MS = 90 if not _CH3_DRAFT else 110
CH3_LIK_W12_N_HOLD = max(10, CH3_SCRIPT_N_HOLD // 4)
CH3_LIK_W12_N_KNOB = max(28, _smooth_n(22))
CH3_LIK_W12_N_ROT = max(64, _smooth_n(48))
CH3_LIK_W12_N_REVEAL = max(40, _smooth_n(30))
CH3_LIK_W12_N_FILL_LINES = CH3_LIK_W12_GRID_N
CH3_LIK_W12_N_FILL_TRACE = max(48, _smooth_n(36)) if not _CH3_DRAFT else max(24, _smooth_n(18))
CH3_LIK_W12_N_B_SLICES = 7 if not _CH3_DRAFT else 4
CH3_LIK_W12_N_ORBIT = max(96, _smooth_n(72))
CH3_LIK_W12_AZIM_W1 = 90.0
CH3_LIK_W12_AZIM_W2 = 180.0
CH3_LIK_W12_AZIM_BOTH = -54.0
CH3_LIK_W12_AZIM_STACK = 90.0
CH3_LIK_W12_ELEV_W1 = 0.0
CH3_LIK_W12_ELEV_W2 = 0.0
CH3_LIK_W12_ELEV_BOTH = 26.0
CH3_LIK_W12_ELEV_STACK = 0.0
CH3_LIK_W12_N_SQUISH = max(28, _smooth_n(22))
CH3_LIK_W12_CORNER_W1 = CH3_LIK_W12_W1_LO
CH3_LIK_W12_CORNER_W2 = CH3_LIK_W12_W2_LO
# 3-D knob 1 / knob 2 axis display (matches ch4_03 ±4 cube)
CH3_LIK_W12_3D_LO = -4.0
CH3_LIK_W12_3D_HI = 4.0
CH3_LIK_W12_CT_ELEV = 24.0
CH3_LIK_W12_CT_AZIM = -128.0
CH3_LIK_W12_ZOOM_GRID = 36 if not _CH3_DRAFT else 20
# Lift scatter slightly above the mesh so it draws in front (mpl 3D depth sort).
CH3_LIK_MARKER_Z_BUMP_FRAC = 0.018


def _ch3_lik_w12_axis_refined(lo, hi, mid, n_coarse, n_fine, *, fine_lo_half):
    """Denser samples on one half-axis; ``mid`` is the split (typically 0)."""
    lo, hi, mid = float(lo), float(hi), float(mid)
    nc, nf = max(2, int(n_coarse)), max(2, int(n_fine))
    if fine_lo_half:
        g_a = np.linspace(lo, mid, nf, dtype=np.float64)
        g_b = np.linspace(mid, hi, nc, dtype=np.float64)
    else:
        g_a = np.linspace(lo, mid, nc, dtype=np.float64)
        g_b = np.linspace(mid, hi, nf, dtype=np.float64)
    return np.unique(np.concatenate([g_a, g_b[1:]]))


def ch3_lik_w12_mesh_pack(
    study, exam, y, b, *, w1_lo, w1_hi, w2_lo, w2_hi, grid_n=None, quadrant_fine=False,
):
    if quadrant_fine:
        mid = float(CH3_LIK_W12_QUAD_MID)
        nc = int(CH3_LIK_W12_GRID_N_COARSE)
        nf = int(CH3_LIK_W12_GRID_N_FINE)
        g1 = _ch3_lik_w12_axis_refined(w1_lo, w1_hi, mid, nc, nf, fine_lo_half=False)
        g2 = _ch3_lik_w12_axis_refined(w2_lo, w2_hi, mid, nc, nf, fine_lo_half=True)
    else:
        gn = int(CH3_LIK_W12_GRID_N if grid_n is None else grid_n)
        g1 = np.linspace(float(w1_lo), float(w1_hi), gn, dtype=np.float64)
        g2 = np.linspace(float(w2_lo), float(w2_hi), gn, dtype=np.float64)
    W1m, W2m = np.meshgrid(g1, g2, indexing="ij")
    bf = np.full(W1m.size, float(b), dtype=np.float64)
    Zf = _ch3_likelihood_on_flat_w12_grid(study, exam, y, W1m.ravel(), W2m.ravel(), bf)
    Z = Zf.reshape(W1m.shape)
    return {
        "W1m": W1m, "W2m": W2m, "Z": Z,
        "w1_lo": float(w1_lo), "w1_hi": float(w1_hi),
        "w2_lo": float(w2_lo), "w2_hi": float(w2_hi),
    }


def ch3_figure_lik_w12_3d(*, duo_width_ratios=None):
    """Split screen: dataset+knobs (left), tall 3D likelihood ridge (right)."""
    wr = CH3_LIK_W12_WIDTH_RATIOS if duo_width_ratios is None else duo_width_ratios
    fig = plt.figure(figsize=CH3_LIK_W12_FIGSIZE)
    gs = fig.add_gridspec(
        1, 2, width_ratios=wr, wspace=CH3_DUO_WSPACE,
    )
    g_left = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=CH3_LEFT_HEIGHT_RATIOS, hspace=CH3_LEFT_HSPACE,
    )
    ax_data = fig.add_subplot(g_left[0,  0])
    g_k = GridSpecFromSubplotSpec(1, 3, subplot_spec=g_left[1, 0], wspace=CH3_KNOB_WSPACE)
    axes_k = tuple(fig.add_subplot(g_k[0, j]) for j in range(3))
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.06)
    from ch4_layout import ch4_duo_plot_layout_tune, ch4_duo_knob_layout_tune

    ch4_duo_plot_layout_tune(fig, ax_data, ax3d)
    _ch3_align_knob_axes_under_data(fig, ax_data, axes_k)
    ch3_layout_knob_axes_like_bridge_end(fig, ax_data, axes_k)
    ch4_duo_knob_layout_tune(fig, ax_data, axes_k)
    return fig, ax_data, ax3d, axes_k


def ch4_figure_duo_weight3d():
    """Ch4 duo layout — legacy 2D/knob column, wider square 3D."""
    from ch4_layout import CH4_DUO_WIDTH_RATIOS

    return ch3_figure_lik_w12_3d(duo_width_ratios=CH4_DUO_WIDTH_RATIOS)


def ch3_lik_w12_z_limits(Z, *, scale=1.0):
    z_hi = float(np.nanmax(Z)) * float(scale)
    pad = 0.10 * max(z_hi, 1e-15)
    return 0.0, z_hi + pad


def ch3_lik_w12_facecolors_full(W1m, W2m, reveal_u, *, rgba):
    t = float(np.clip(float(reveal_u), 0.0, 1.0))
    fc = np.empty(W1m.shape + (4,), dtype=float)
    rgba = mpl.colors.to_rgba(rgba)
    fc[..., :] = rgba
    fc[..., 3] = rgba[3] * t
    return fc


def ch3_lik_w12_stack_z_lim(z_lik_hi, *, pad_frac=0.06):
    """Fixed display box: all stacked surfaces compress into [0, z_hi]."""
    z_hi = float(z_lik_hi)
    pad = float(pad_frac) * max(z_hi, 1.0)
    return 0.0, z_hi + pad


def ch3_lik_w12_squish_z(Z, layer_i, n_layers, z_lo, z_hi, z_ref):
    """Map mistake height into layer_i of n_layers equal slots in [z_lo, z_hi]."""
    n = max(float(n_layers), 1.0)
    span = float(z_hi) - float(z_lo)
    slot = span / n
    z_base = float(z_lo) + float(layer_i) * slot
    scale = slot / max(float(z_ref), 1e-9)
    return z_base + np.asarray(Z, dtype=float) * scale


def ch3_lik_w12_squish_scalar(z_val, layer_i, n_layers, z_lo, z_hi, z_ref):
    n = max(float(n_layers), 1.0)
    span = float(z_hi) - float(z_lo)
    slot = span / n
    z_base = float(z_lo) + float(layer_i) * slot
    scale = slot / max(float(z_ref), 1e-9)
    return z_base + float(z_val) * scale


def _ch3_lik_w12_z_at(W1m, W2m, Z, w1, w2):
    d = (np.asarray(W1m, dtype=float) - float(w1)) ** 2 + (np.asarray(W2m, dtype=float) - float(w2)) ** 2
    return float(np.ravel(np.asarray(Z, dtype=float))[int(np.nanargmin(d))])


def _ch3_lik_w12_crop_mesh_xy(W1m, W2m, Z, *, xlo, xhi, ylo, yhi, fc=None):
    """Subset a meshgrid surface to an (x, y) window so zoomed axes don't show out-of-box geometry."""
    W1m = np.asarray(W1m, dtype=float)
    W2m = np.asarray(W2m, dtype=float)
    Z = np.asarray(Z, dtype=float)
    g1 = W1m[:, 0]
    g2 = W2m[0, :]
    i_idx = np.flatnonzero((g1 >= float(xlo) - 1e-12) & (g1 <= float(xhi) + 1e-12))
    j_idx = np.flatnonzero((g2 >= float(ylo) - 1e-12) & (g2 <= float(yhi) + 1e-12))
    if i_idx.size < 2 or j_idx.size < 2:
        return W1m, W2m, Z, fc
    i0, i1 = int(i_idx[0]), int(i_idx[-1]) + 1
    j0, j1 = int(j_idx[0]), int(j_idx[-1]) + 1
    W1c = W1m[i0:i1, j0:j1]
    W2c = W2m[i0:i1, j0:j1]
    Zc = Z[i0:i1, j0:j1]
    fcc = fc[i0:i1, j0:j1] if fc is not None else None
    return W1c, W2c, Zc, fcc


def ch3_lik_w12_knob3_z_ticks(stack_layers, n_layers, z_lo, z_hi):
    """Tick positions at stacked-layer centers; labels are Knob 3 (b) values."""
    if not stack_layers:
        return None, None
    n = max(float(n_layers), 1.0)
    span = float(z_hi) - float(z_lo)
    slot = span / n
    layers = sorted(
        (
            sl for sl in stack_layers
            if float(sl.get("reveal", 1.0)) > 1e-4 and sl.get("b") is not None
        ),
        key=lambda sl: int(sl.get("layer_i", 0)),
    )
    if not layers:
        return None, None
    tick_z, tick_lbl = [], []
    for sl in layers:
        li = int(sl.get("layer_i", 0))
        tick_z.append(float(z_lo) + (float(li) + 0.5) * slot)
        tick_lbl.append(f"{float(sl['b']):.2g}")
    return tick_z, tick_lbl


def ch3_lik_w12_facecolors_diag(W1m, W2m, reveal_u, *, w1_lo, w1_hi, w2_lo, w2_hi, rgba, origin="lo_lo"):
    """Diagonal surface reveal; ``origin='lo_hi'`` starts at (w1_lo, w2_hi)."""
    u1 = (W1m - float(w1_lo)) / max(float(w1_hi) - float(w1_lo), 1e-9)
    if str(origin) == "lo_hi":
        u2 = (float(w2_hi) - W2m) / max(float(w2_hi) - float(w2_lo), 1e-9)
    else:
        u2 = (W2m - float(w2_lo)) / max(float(w2_hi) - float(w2_lo), 1e-9)
    t = float(np.clip(float(reveal_u), 0.0, 1.0))
    mask = (u1 + u2) <= 2.0 * t + 1e-9
    fc = np.empty(W1m.shape + (4,), dtype=float)
    rgba = mpl.colors.to_rgba(rgba)
    fc[..., :] = rgba
    fc[..., 3] = rgba[3] * mask.astype(float)
    return fc


def _ch3_lik_w12_here_annotation(
    fig, ax3d, mx, my, z_mark, *,
    label="WE ARE HERE",
    label_fig=None,
    color=None,
    text_color=None,
    edgecolor=None,
):
    """Arrow + label in figure space, upper-right of the 3D panel."""
    from matplotlib.patches import FancyArrowPatch

    fig.canvas.draw()
    px, py = ch3_ax3d_to_fig_xy(fig, ax3d, float(mx), float(my), float(z_mark))
    if label_fig is None:
        label_fig = CH3_LIK_RIDGE_LABEL_FIG
    lx, ly = float(label_fig[0]), float(label_fig[1])
    arrow_col = FAIL_COLOR if color is None else color
    # Label text stays white on the dark callout; arrow/marker keep accent color.
    text_col = "white" if text_color is None else text_color
    medge = "white" if edgecolor is None else edgecolor
    fig.text(
        lx, ly, str(label),
        transform=fig.transFigure,
        fontsize=CH3_LIK_RIDGE_HERE_FS,
        color=text_col,
        fontfamily="DejaVu Sans",
        fontweight="bold",
        ha="right",
        va="top",
        zorder=60,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", edgecolor=medge, alpha=0.95),
    )
    arrow = FancyArrowPatch(
        (lx, ly - 0.05), (px, py),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.4,
        color=arrow_col,
        shrinkA=6,
        shrinkB=6,
        zorder=59,
    )
    fig.patches.append(arrow)


def ch3_frame_lik_w12_3d(
    study,
    exam,
    y,
    w_st,
    w_el,
    b,
    *,
    mesh_pack,
    z_lim,
    curves,
    elev,
    azim,
    emphasize_knob="st",
    landscape_reveal=0.0,
    landscape_reveal_origin="lo_lo",
    landscape_rgba=None,
    b_slices=None,
    slice_reveal=0.0,
    show_curves=True,
    marker=True,
    marker_z_offset=0.0,
    stack_layers=None,
    stack_n_layers=1.0,
    z_lik_ref=None,
    show_axis_labels=True,
    flat_surface=None,
    overlay_surface=None,
    z_label=None,
    knob_pack=None,
    knob_scales=None,
    weight_axis_labels=False,
    marker_ws=None,
    marker_we=None,
    marker_z=None,
    marker_color=None,
    marker_edgecolors="white",
    here_annotation=False,
    here_label="WE ARE HERE",
    here_label_fig=None,
    here_text_color="white",
    ax3d_xlim=None,
    ax3d_ylim=None,
    show_colormap=True,
    show_legend=True,
    show_threshold=False,
    threshold_ws=None,
    threshold_we=None,
    threshold_bb=None,
    threshold_label="most plausible line",
    threshold_legend_dark=False,
    show_shadow_marker=False,
    shadow_marker_ws=None,
    shadow_marker_we=None,
    shadow_marker_z=None,
    show_shadow_threshold=False,
    shadow_threshold_ws=None,
    shadow_threshold_we=None,
    shadow_threshold_bb=None,
    sigma_Z=None,
    origin_guides=None,
    data_xlim=None,
    data_ylim=None,
    zero_axis_guides_u=0.0,
    vertical_sprout=None,
    belief_stems=None,
):
    spec = CH3_LOSS_SPECS["likelihood"]
    loss_fn = spec["fn"]
    w_st, w_el, b = float(w_st), float(w_el), float(b)
    W1m, W2m, Z = mesh_pack["W1m"], mesh_pack["W2m"], mesh_pack["Z"]
    w1_lo, w1_hi = mesh_pack["w1_lo"], mesh_pack["w1_hi"]
    w2_lo, w2_hi = mesh_pack["w2_lo"], mesh_pack["w2_hi"]
    z_lo_ax, z_hi_ax = float(z_lim[0]), float(z_lim[1])
    z_ref = float(np.nanmax(Z) if z_lik_ref is None else z_lik_ref)
    n_stack = max(float(stack_n_layers), 1.0)
    panel_xlim = xlim if data_xlim is None else (float(data_xlim[0]), float(data_xlim[1]))
    panel_ylim = ylim if data_ylim is None else (float(data_ylim[0]), float(data_ylim[1]))

    fig, ax_data, ax3d, axes_k = ch4_figure_duo_weight3d()
    leg = legend_linear_equation_values_bold_param(w_st, w_el, b, emphasize_knob)
    _panel_kw = dict(
        show_colormap=bool(show_colormap), highlight_mistakes_flag=False,
        boundary_xlim=panel_xlim, boundary_ylim=panel_ylim,
    )
    if sigma_Z is not None:
        _panel_kw["sigma_Z"] = sigma_Z
    ch3_draw_left_panel(
        ax_data, w_st, w_el, b, study, exam, y, leg, **_panel_kw,
    )
    ax_data.set_xlim(*panel_xlim)
    ax_data.set_ylim(*panel_ylim)
    _zac_u = float(zero_axis_guides_u)
    if _zac_u > 1e-4:
        _zac2 = globals().get("ch5_draw_zero_axis_cross_2d")
        if _zac2 is not None:
            _zac2(ax_data, u=_zac_u, xlim=panel_xlim, ylim=panel_ylim)
    if bool(show_threshold):
        tws = float(threshold_ws if threshold_ws is not None else w_st)
        twe = float(threshold_we if threshold_we is not None else w_el)
        tbb = float(threshold_bb if threshold_bb is not None else b)
        lbl = str(threshold_label if threshold_label is not None else "most plausible line")
        bxy = boundary_line_xy(
            tws, twe, tbb,
            float(panel_xlim[0]), float(panel_xlim[1]),
            float(panel_ylim[0]), float(panel_ylim[1]),
        )
        if bxy is not None:
            bx, by = bxy
            ax_data.plot(
                bx, by, c="#666666", linestyle="--", linewidth=2.2, alpha=0.95,
                label=lbl, zorder=4,
            )
        else:
            ax_data.plot(
                [], [], c="#666666", linestyle="--", linewidth=2.2, label=lbl,
            )
        handles, _labels = ax_data.get_legend_handles_labels()
        if handles:
            ax_data.legend(loc="upper left", prop={"size": LEGEND_SIZE})
        finalize_style_legend_tex(ax_data)
        if bool(threshold_legend_dark):
            leg = ax_data.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor("black")
                leg.get_frame().set_edgecolor("white")
                leg.get_frame().set_alpha(0.95)
                for txt in leg.get_texts():
                    txt.set_color("white")
    elif bool(show_legend):
        finalize_style_legend_tex(ax_data)
    if bool(show_shadow_threshold):
        sws = float(shadow_threshold_ws if shadow_threshold_ws is not None else w_st)
        swe = float(shadow_threshold_we if shadow_threshold_we is not None else w_el)
        sbb = float(shadow_threshold_bb if shadow_threshold_bb is not None else b)
        tws_map = float(threshold_ws if threshold_ws is not None else w_st)
        twe_map = float(threshold_we if threshold_we is not None else w_el)
        if abs(sws - tws_map) > 1e-5 or abs(swe - twe_map) > 1e-5:
            sbxy = boundary_line_xy(
                sws, swe, sbb,
                float(panel_xlim[0]), float(panel_xlim[1]),
                float(panel_ylim[0]), float(panel_ylim[1]),
            )
            if sbxy is not None:
                bx, by = sbxy
                # Shadow line: thicker + darker than the MAP "most plausible" threshold.
                ax_data.plot(
                    bx, by, c="#222222", linestyle="--", linewidth=3.6, alpha=1.0,
                    zorder=5,
                )
    if knob_pack is None:
        knob_rgbs, canvas_sides = ch3_knob_asset_pack()
    else:
        knob_rgbs, canvas_sides = knob_pack
    if knob_scales is None:
        scales = ch3_knob_scales_emphasize(emphasize_knob, CH3_KNOB_ACTIVE_SCALE)
    else:
        scales = list(knob_scales)
    ch3_draw_knob_row(
        fig, axes_k, w_st, w_el, b, emphasize_knob,
        knob_rgbs, canvas_sides,
        rot_strip_deg=0.0, strip_scale=1.0,
        knob_rots=ch3_k1_knob_rots_at(w_st, w_el, b),
        knob_scales=scales, ax_data=ax_data,
    )

    ax3d.cla()
    ax3d.computed_zorder = False
    lr = landscape_rgba if landscape_rgba is not None else (CH4_LIK_SURFACE_COLOR, CH3_LIK_W12_SURFACE_ALPHA)
    z_pt = None
    mark_x = mark_y = mark_z = None

    def _maybe_crop_xy(W1, W2, Z, fc=None):
        if ax3d_xlim is None and ax3d_ylim is None:
            return W1, W2, Z, fc
        xlo = float(w1_lo if ax3d_xlim is None else ax3d_xlim[0])
        xhi = float(w1_hi if ax3d_xlim is None else ax3d_xlim[1])
        ylo = float(w2_lo if ax3d_ylim is None else ax3d_ylim[0])
        yhi = float(w2_hi if ax3d_ylim is None else ax3d_ylim[1])
        W1, W2, Z, fc = _ch3_lik_w12_crop_mesh_xy(
            W1, W2, Z, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, fc=fc,
        )
        z_lo_c, z_hi_c = float(z_lim[0]), float(z_lim[1])
        Z = np.clip(np.asarray(Z, dtype=float), z_lo_c, z_hi_c)
        return W1, W2, Z, fc

    def _draw_overlay_surface():
        if overlay_surface is None:
            return
        orev = float(overlay_surface.get("reveal", 1.0))
        if orev <= 1e-4:
            return
        Wo1 = overlay_surface["W1m"]
        Wo2 = overlay_surface["W2m"]
        Zo = np.asarray(overlay_surface["Z"], dtype=float)
        o_rgba = overlay_surface.get("rgba", ("#8a8a8a", 0.45))
        o_origin = str(overlay_surface.get("reveal_origin", "lo_hi"))
        ow1_lo = float(overlay_surface.get("w1_lo", w1_lo))
        ow1_hi = float(overlay_surface.get("w1_hi", w1_hi))
        ow2_lo = float(overlay_surface.get("w2_lo", w2_lo))
        ow2_hi = float(overlay_surface.get("w2_hi", w2_hi))
        fc_o = ch3_lik_w12_facecolors_diag(
            Wo1, Wo2, orev,
            w1_lo=ow1_lo, w1_hi=ow1_hi, w2_lo=ow2_lo, w2_hi=ow2_hi,
            rgba=o_rgba, origin=o_origin,
        )
        # Hide faces outside the overlay support (e.g. likelihood ±2.5 box).
        invalid = ~np.isfinite(Zo)
        if np.any(invalid):
            fc_o = np.asarray(fc_o, dtype=float).copy()
            fc_o[invalid, 3] = 0.0
        Wo1, Wo2, Zo, fc_o = _maybe_crop_xy(Wo1, Wo2, Zo, fc_o)
        edge_kw = {}
        if overlay_surface.get("surface_grid"):
            edge_kw = dict(
                edgecolor=overlay_surface.get("surface_grid_edgecolor", "#3a3a3a"),
                linewidth=float(overlay_surface.get("surface_grid_linewidth", 0.35)),
                rstride=int(overlay_surface.get("surface_grid_rstride", 1)),
                cstride=int(overlay_surface.get("surface_grid_cstride", 1)),
            )
        else:
            edge_kw = dict(linewidth=0, rstride=1, cstride=1)
        ax3d.plot_surface(
            Wo1, Wo2, Zo, facecolors=fc_o, shade=False,
            antialiased=False,
            zorder=int(overlay_surface.get("zorder", 0)),
            **edge_kw,
        )
        if overlay_surface.get("marker") and orev > 0.55:
            omx = float(overlay_surface["marker_ws"])
            omy = float(overlay_surface["marker_we"])
            omz = float(overlay_surface["marker_z"])
            z_span = max(z_hi_ax - z_lo_ax, 1e-12)
            omz_mark = omz + float(CH3_LIK_MARKER_Z_BUMP_FRAC) * z_span
            ocol = overlay_surface.get("marker_color", CH4_LIK_SURFACE_COLOR)
            ax3d.scatter(
                [omx], [omy], [omz_mark],
                color=ocol, edgecolors=overlay_surface.get("marker_edgecolors", "white"),
                linewidths=2.0, s=260.0, depthshade=False, zorder=31,
            )
            overlay_surface["_mark_xyz"] = (omx, omy, omz_mark)


    # Likelihood (etc.) overlays draw under the belief surface by default.
    if overlay_surface is not None and bool(overlay_surface.get("under", True)):
        _draw_overlay_surface()

    if flat_surface is not None:
        Wb = flat_surface["W1m"]
        W2b = flat_surface["W2m"]
        Zb = np.asarray(flat_surface["Z"], dtype=float)
        if flat_surface.get("nll_heatmap") is not None:
            Wb, W2b, Zb, _ = _maybe_crop_xy(Wb, W2b, Zb)
            _ch4_nll_heatmap_plot_surface(
                ax3d, Wb, W2b, Zb, flat_surface["nll_heatmap"],
            )
        elif flat_surface.get("facecolors") is not None:
            fc_b = flat_surface["facecolors"]
            Wb, W2b, Zb, fc_b = _maybe_crop_xy(Wb, W2b, Zb, fc_b)
            if flat_surface.get("surface_grid"):
                ch5_plot_belief_surface_with_grid(
                    ax3d, Wb, W2b, Zb, facecolors=fc_b, zorder=1,
                    rstride=int(flat_surface.get("surface_grid_rstride", 1)),
                    cstride=int(flat_surface.get("surface_grid_cstride", 1)),
                    edgecolor=flat_surface.get("surface_grid_edgecolor", "#4d4d4d"),
                    linewidth=float(flat_surface.get("surface_grid_linewidth", 0.55)),
                )
            else:
                ax3d.plot_surface(
                    Wb, W2b, Zb, facecolors=fc_b, shade=False,
                    linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=1,
                )
        else:
            fc_b = ch3_lik_w12_facecolors_full(
                Wb, W2b, 1.0,
                rgba=(CH4_LIK_SURFACE_COLOR, CH3_LIK_W12_SURFACE_ALPHA * float(flat_surface.get("alpha_scale", 1.0))),
            )
            Wb, W2b, Zb, fc_b = _maybe_crop_xy(Wb, W2b, Zb, fc_b)
            ax3d.plot_surface(
                Wb, W2b, Zb, facecolors=fc_b, shade=False,
                linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=1,
            )
        z_pt = float(
            flat_surface["marker_z"]
            if flat_surface.get("marker_z") is not None
            else _ch3_lik_w12_z_at(Wb, W2b, Zb, w_st, w_el)
        )
    elif float(landscape_reveal) > 1e-4:
        fc = ch3_lik_w12_facecolors_diag(
            W1m, W2m, landscape_reveal,
            w1_lo=w1_lo, w1_hi=w1_hi, w2_lo=w2_lo, w2_hi=w2_hi, rgba=lr,
            origin=landscape_reveal_origin,
        )
        W1p, W2p, Zp, fc = _maybe_crop_xy(W1m, W2m, Z, fc)
        ax3d.plot_surface(
            W1p, W2p, Zp, facecolors=fc, shade=False,
            linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=1,
        )
    if overlay_surface is not None and not bool(overlay_surface.get("under", True)):
        _draw_overlay_surface()
    if stack_layers:
        for sl in stack_layers:
            rev = float(sl.get("reveal", 1.0)) * float(slice_reveal)
            if rev < 1e-4:
                continue
            pack_b = sl["pack"]
            Wb, Zb = pack_b["W1m"], pack_b["Z"]
            li = int(sl.get("layer_i", 0))
            Zplot = ch3_lik_w12_squish_z(Zb, li, n_stack, z_lo_ax, z_hi_ax, z_ref)
            fc_b = ch3_lik_w12_facecolors_full(
                Wb, pack_b["W2m"], rev,
                rgba=(CH4_LIK_SURFACE_COLOR, CH3_LIK_W12_SURFACE_ALPHA * float(sl.get("alpha_scale", 1.0))),
            )
            ax3d.plot_surface(
                Wb, pack_b["W2m"], Zplot, facecolors=fc_b, shade=False,
                linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=int(sl.get("zorder", 3)),
            )
    elif b_slices:
        for sl in b_slices:
            bb = float(sl["b"])
            rev = float(sl.get("reveal", 1.0)) * float(slice_reveal)
            if rev < 1e-4:
                continue
            pack_b = sl.get("pack")
            if pack_b is None:
                pack_b = ch3_lik_w12_mesh_pack(
                    study, exam, y, bb, w1_lo=w1_lo, w1_hi=w1_hi, w2_lo=w2_lo, w2_hi=w2_hi,
                )
            Wb, Zb = pack_b["W1m"], pack_b["Z"]
            z0 = float(sl.get("z_base", 0.0))
            fc_b = ch3_lik_w12_facecolors_full(
                Wb, pack_b["W2m"], rev,
                rgba=(CH4_LIK_SURFACE_COLOR, CH3_LIK_W12_SLICE_ALPHA * float(sl.get("alpha_scale", 1.0))),
            )
            ax3d.plot_surface(
                Wb, pack_b["W2m"], Zb + z0, facecolors=fc_b, shade=False,
                linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=3,
            )
    if show_curves and curves:
        for item in curves:
            if len(item) == 4:
                cw1, cw2, cz, ccol = item
            else:
                cw1, cw2, cz = item
                ccol = CH4_LIK_SURFACE_COLOR
            cw1 = np.asarray(cw1, dtype=float)
            cw2 = np.asarray(cw2, dtype=float)
            cz = np.asarray(cz, dtype=float) + float(marker_z_offset)
            if cw1.size >= 2:
                ax3d.plot(
                    cw1, cw2, cz, color=ccol, linewidth=CH3_LIK_W12_CURVE_LW,
                    alpha=0.95, zorder=8,
                )
    if z_pt is None:
        z_raw = float(loss_fn(w_st, w_el, b, study, exam, y))
        if stack_layers:
            top_i = max(int(sl.get("layer_i", 0)) for sl in stack_layers if float(sl.get("reveal", 0)) > 1e-4)
            z_pt = ch3_lik_w12_squish_scalar(z_raw, top_i, n_stack, z_lo_ax, z_hi_ax, z_ref)
        else:
            z_pt = z_raw + float(marker_z_offset)
    if marker:
        mx = float(w_st if marker_ws is None else marker_ws)
        my = float(w_el if marker_we is None else marker_we)
        if marker_z is not None:
            z_pt = float(marker_z)
        elif marker_ws is not None or marker_we is not None:
            if flat_surface is not None:
                Wb = flat_surface["W1m"]
                W2b = flat_surface["W2m"]
                Zb = np.asarray(flat_surface["Z"], dtype=float)
                z_pt = _ch3_lik_w12_z_at(Wb, W2b, Zb, mx, my)
            else:
                z_pt = float(loss_fn(mx, my, b, study, exam, y)) + float(marker_z_offset)
        mcol = FAIL_COLOR if marker_color is None else marker_color
        z_span = max(z_hi_ax - z_lo_ax, 1e-12)
        z_mark = float(z_pt) + float(CH3_LIK_MARKER_Z_BUMP_FRAC) * z_span
        mark_x, mark_y, mark_z = mx, my, z_mark
        ax3d.scatter(
            [mx], [my], [z_mark],
            color=mcol, edgecolors=marker_edgecolors, linewidths=2.0,
            s=260.0, depthshade=False, zorder=30,
        )
    if bool(show_shadow_marker):
        sx = float(shadow_marker_ws if shadow_marker_ws is not None else w_st)
        sy = float(shadow_marker_we if shadow_marker_we is not None else w_el)
        mx_ref = float(w_st if marker_ws is None else marker_ws)
        my_ref = float(w_el if marker_we is None else marker_we)
        if abs(sx - mx_ref) > 1e-5 or abs(sy - my_ref) > 1e-5:
            if shadow_marker_z is not None:
                sz_sh = float(shadow_marker_z)
            elif flat_surface is not None:
                Wb = flat_surface["W1m"]
                W2b = flat_surface["W2m"]
                Zb = np.asarray(flat_surface["Z"], dtype=float)
                sz_sh = _ch3_lik_w12_z_at(Wb, W2b, Zb, sx, sy)
            else:
                sz_sh = float(loss_fn(sx, sy, b, study, exam, y)) + float(marker_z_offset)
            z_span = max(z_hi_ax - z_lo_ax, 1e-12)
            sz_mark = float(sz_sh) + float(CH3_LIK_MARKER_Z_BUMP_FRAC) * z_span
            ax3d.scatter(
                [sx], [sy], [sz_mark],
                color="#111111", edgecolors="white", linewidths=1.4,
                s=220.0, depthshade=False, zorder=29,
            )
    if vertical_sprout is not None:
        sws = float(vertical_sprout["ws"])
        swe = float(vertical_sprout["we"])
        z0 = float(vertical_sprout.get("z0", z_lo_ax))
        z1 = float(vertical_sprout.get("z1", z0))
        if bool(vertical_sprout.get("show_floor_dot", True)):
            ax3d.scatter(
                [sws], [swe], [z0],
                color=str(vertical_sprout.get("floor_color", "#111111")),
                edgecolors=str(vertical_sprout.get("floor_edgecolors", "white")),
                linewidths=1.6, s=200.0, depthshade=False, zorder=28,
            )
        if z1 > z0 + 1e-6:
            n_seg = int(vertical_sprout.get("n_seg", 64))
            zs = np.linspace(z0, z1, max(8, n_seg), dtype=float)
            ax3d.plot(
                np.full_like(zs, sws), np.full_like(zs, swe), zs,
                color=str(vertical_sprout.get("line_color", "#111111")),
                linestyle=str(vertical_sprout.get("linestyle", ":")),
                linewidth=float(vertical_sprout.get("linewidth", 2.2)),
                alpha=float(vertical_sprout.get("alpha", 0.95)),
                zorder=27,
            )
    if belief_stems is not None:
        # Parallelepiped pillars: surface-colored tops + white walls with dark edges.
        from matplotlib.colors import to_rgba
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

        _ws = np.asarray(belief_stems["ws"], dtype=float).ravel()
        _we = np.asarray(belief_stems["we"], dtype=float).ravel()
        _zz = np.asarray(belief_stems["z"], dtype=float).ravel()
        _n = int(_ws.size)
        if _n > 0:
            _rgba = belief_stems.get("tip_rgba")
            _fc_all = None if _rgba is None else np.asarray(_rgba, dtype=float)
            _wf = float(np.clip(float(belief_stems.get("width_frac", 0.0)), 0.0, 1.0))
            _dx0 = belief_stems.get("dx", 0.0)
            _dy0 = belief_stems.get("dy", 0.0)
            _dx_src = np.asarray(_dx0, dtype=float).ravel()
            _dy_src = np.asarray(_dy0, dtype=float).ravel()
            _dx_base = np.full(_n, float(_dx_src[0]), dtype=float) if _dx_src.size <= 1 else _dx_src[:_n]
            _dy_base = np.full(_n, float(_dy_src[0]), dtype=float) if _dy_src.size <= 1 else _dy_src[:_n]
            _dx = np.maximum(_dx_base * _wf, 0.0)
            _dy = np.maximum(_dy_base * _wf, 0.0)
            _ba = float(belief_stems.get("bar_alpha", 0.92))
            _side_a = float(belief_stems.get("side_alpha", 0.55))
            _side_rgba = to_rgba(str(belief_stems.get("side_color", "#ffffff")), alpha=_side_a)
            _side_edge = str(belief_stems.get("side_edge_color", "#2a2a2a"))
            _side_ew = float(belief_stems.get("side_edge_width", 0.55))
            _la = float(belief_stems.get("line_alpha", 0.18))
            _lw = float(belief_stems.get("linewidth", 0.7))
            _lc = str(belief_stems.get("line_color", "#888888"))
            _line_cut = float(belief_stems.get("line_width_frac", 0.04))
            _draw_bars = _wf > _line_cut and float(np.nanmax(_dx)) > 1e-9 and float(np.nanmax(_dy)) > 1e-9
            if _fc_all is not None and _fc_all.ndim == 2 and _fc_all.shape[0] >= _n:
                _fc = np.array(_fc_all[:_n], copy=True)
            else:
                _fc = np.tile(np.array([0.85, 0.35, 0.35, 1.0], dtype=float), (_n, 1))
            # tip_rgba is already final surface RGBA; only scale fallback colors.
            if not bool(belief_stems.get("preserve_tip_alpha", False)) and _fc.shape[1] >= 4:
                _fc[:, 3] = np.clip(_fc[:, 3] * _ba, 0.0, 1.0)
            if _draw_bars:
                _z0 = float(z_lo_ax)
                _hx = 0.5 * _dx
                _hy = 0.5 * _dy
                _x0 = _ws - _hx
                _x1 = _ws + _hx
                _y0 = _we - _hy
                _y1 = _we + _hy
                _zt = np.maximum(_zz, _z0)
                _tall = _zt > (_z0 + 1e-6)
                # White walls + dark edges for risen pillars (under tops).
                if np.any(_tall):
                    _ti = np.flatnonzero(_tall)
                    _x0t, _x1t = _x0[_ti], _x1[_ti]
                    _y0t, _y1t = _y0[_ti], _y1[_ti]
                    _ztt = _zt[_ti]
                    _nt = int(_ti.size)
                    _z0a = np.full(_nt, _z0)
                    _f_y0 = np.stack([
                        np.column_stack([_x0t, _y0t, _z0a]),
                        np.column_stack([_x1t, _y0t, _z0a]),
                        np.column_stack([_x1t, _y0t, _ztt]),
                        np.column_stack([_x0t, _y0t, _ztt]),
                    ], axis=1)
                    _f_y1 = np.stack([
                        np.column_stack([_x0t, _y1t, _z0a]),
                        np.column_stack([_x1t, _y1t, _z0a]),
                        np.column_stack([_x1t, _y1t, _ztt]),
                        np.column_stack([_x0t, _y1t, _ztt]),
                    ], axis=1)
                    _f_x0 = np.stack([
                        np.column_stack([_x0t, _y0t, _z0a]),
                        np.column_stack([_x0t, _y1t, _z0a]),
                        np.column_stack([_x0t, _y1t, _ztt]),
                        np.column_stack([_x0t, _y0t, _ztt]),
                    ], axis=1)
                    _f_x1 = np.stack([
                        np.column_stack([_x1t, _y0t, _z0a]),
                        np.column_stack([_x1t, _y1t, _z0a]),
                        np.column_stack([_x1t, _y1t, _ztt]),
                        np.column_stack([_x1t, _y0t, _ztt]),
                    ], axis=1)
                    _walls = np.concatenate([_f_y0, _f_y1, _f_x0, _f_x1], axis=0)
                    _wcols = np.tile(np.asarray(_side_rgba, dtype=float), (_walls.shape[0], 1))
                    ax3d.add_collection3d(Poly3DCollection(
                        _walls, facecolors=_wcols, edgecolors=_side_edge,
                        linewidths=_side_ew, shade=False, zorder=26,
                    ))
                # Tops: exact surface RGBA (batched Poly3DCollection).
                _top = np.stack([
                    np.column_stack([_x0, _y0, _zt]),
                    np.column_stack([_x1, _y0, _zt]),
                    np.column_stack([_x1, _y1, _zt]),
                    np.column_stack([_x0, _y1, _zt]),
                ], axis=1)
                ax3d.add_collection3d(Poly3DCollection(
                    _top, facecolors=_fc, edgecolors="none", linewidths=0.0,
                    shade=False, zorder=27,
                ))
            else:
                _tall = _zz > (float(z_lo_ax) + 1e-6)
                if np.any(_tall):
                    _ws_t = _ws[_tall]
                    _we_t = _we[_tall]
                    _zz_t = _zz[_tall]
                    try:
                        _segs = np.stack([
                            np.column_stack([_ws_t, _we_t, np.full_like(_zz_t, float(z_lo_ax))]),
                            np.column_stack([_ws_t, _we_t, _zz_t]),
                        ], axis=1)
                        ax3d.add_collection3d(Line3DCollection(
                            _segs, colors=_lc, linewidths=_lw, alpha=_la, zorder=26,
                        ))
                    except Exception:
                        for _i in range(int(_ws_t.size)):
                            ax3d.plot(
                                [_ws_t[_i], _ws_t[_i]], [_we_t[_i], _we_t[_i]],
                                [float(z_lo_ax), float(_zz_t[_i])],
                                color=_lc, linewidth=_lw, alpha=_la, zorder=26,
                            )
                if bool(belief_stems.get("draw_tips", False)):
                    _ps = float(belief_stems.get("point_size", 14.0))
                    _pa = float(belief_stems.get("point_alpha_scale", 1.0))
                    _fc_tip = np.array(_fc, copy=True)
                    if _fc_tip.shape[1] >= 4 and abs(_pa - 1.0) > 1e-6:
                        _fc_tip[:, 3] = np.clip(_fc_tip[:, 3] * _pa, 0.0, 1.0)
                    _show = _tall if np.any(_tall) else np.ones(_n, dtype=bool)
                    ax3d.scatter(
                        _ws[_show], _we[_show], _zz[_show],
                        c=_fc_tip[_show], s=_ps, depthshade=False, linewidths=0.0, zorder=27,
                    )
    if ax3d_xlim is not None:
        ax3d.set_xlim(float(ax3d_xlim[0]), float(ax3d_xlim[1]))
    else:
        ax3d.set_xlim(w1_lo, w1_hi)
    if ax3d_ylim is not None:
        ax3d.set_ylim(float(ax3d_ylim[0]), float(ax3d_ylim[1]))
    else:
        ax3d.set_ylim(w2_lo, w2_hi)
    ax3d.set_zlim(float(z_lim[0]), float(z_lim[1]))
    knob3_zticks, knob3_zlabels = ch3_lik_w12_knob3_z_ticks(stack_layers, n_stack, z_lo_ax, z_hi_ax)
    if show_axis_labels:
        fs_ax = float(AXIS_LABEL_SIZE) * float(CH3_LIK_3D_AXIS_LABEL_SCALE)
        if weight_axis_labels:
            ax3d.set_xlabel(r"$w_{\mathrm{ST}}$", fontsize=fs_ax, labelpad=10)
            ax3d.set_ylabel(r"$w_{\mathrm{EL}}$", fontsize=fs_ax, labelpad=10)
        else:
            ax3d.set_xlabel("Knob 1", fontsize=AXIS_LABEL_SIZE, labelpad=10)
            ax3d.set_ylabel("Knob 2", fontsize=AXIS_LABEL_SIZE, labelpad=10)
        if knob3_zticks:
            ax3d.set_zlabel("Knob 3", fontsize=AXIS_LABEL_SIZE, labelpad=10)
            ax3d.set_zticks(knob3_zticks)
            ax3d.set_zticklabels(knob3_zlabels)
        elif weight_axis_labels:
            from matplotlib.ticker import MaxNLocator

            ax3d.set_zlabel(str(z_label or r"$b$"), fontsize=fs_ax, labelpad=10)
            ax3d.zaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            ax3d.set_zlabel(str(z_label or "Likelihood"), fontsize=AXIS_LABEL_SIZE, labelpad=10)
        ax3d.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
        ax3d.grid(True)
    else:
        ax3d.set_xlabel("")
        ax3d.set_ylabel("")
        ax3d.set_zlabel("")
        ax3d.set_xticklabels([])
        ax3d.set_yticklabels([])
        ax3d.set_zticklabels([])
    ax3d.view_init(elev=float(elev), azim=float(azim))
    if origin_guides is not None:
        _og_fn = globals().get("ch5_draw_landscape_origin_guides")
        if _og_fn is not None:
            _og_fn(ax3d, guides=origin_guides)
    if float(zero_axis_guides_u) > 1e-4:
        _zac3 = globals().get("ch5_draw_zero_axis_cross_3d")
        if _zac3 is not None:
            _zac3(
                ax3d, u=float(zero_axis_guides_u),
                w1_lo=w1_lo, w1_hi=w1_hi, w2_lo=w2_lo, w2_hi=w2_hi,
                z_floor=float(z_lim[0]),
            )
    if bool(here_annotation) and mark_x is not None:
        _ch3_lik_w12_here_annotation(
            fig, ax3d, mark_x, mark_y, mark_z,
            label=here_label,
            label_fig=here_label_fig,
            color=marker_color,
            text_color=here_text_color,
            edgecolor=marker_edgecolors,
        )
    if (
        overlay_surface is not None
        and overlay_surface.get("here_annotation")
        and overlay_surface.get("_mark_xyz") is not None
    ):
        omx, omy, omz = overlay_surface["_mark_xyz"]
        _ch3_lik_w12_here_annotation(
            fig, ax3d, omx, omy, omz,
            label=str(overlay_surface.get("here_label", "best line")),
            label_fig=overlay_surface.get("here_label_fig"),
            color=overlay_surface.get("here_color", overlay_surface.get("marker_color")),
            text_color=overlay_surface.get("here_text_color", "white"),
            edgecolor=overlay_surface.get("marker_edgecolors", "white"),
        )
    return fig_to_image(fig, dpi=CH3_ANIM_DPI)


def _ch3_lik_w12_azim_shortest_delta(az0, az1):
    """Signed azimuth delta in (-180, 180] — shortest rotation."""
    return (float(az1) - float(az0) + 180.0) % 360.0 - 180.0


def _ch3_lik_w12_lerp_azim_shortest(az0, az1, u):
    return float(az0) + _ch3_lik_w12_azim_shortest_delta(az0, az1) * float(u)


def ch3_lik_w12_frame_opening(mesh_pack, z_lim, *, w_st=None, w_el=None):
    """Opening frame: empty 3-D at ``(w_st, w_el)`` (defaults: script corner)."""
    w1_c = float(CH3_LIK_W12_CORNER_W1 if w_st is None else w_st)
    w2_c = float(CH3_LIK_W12_CORNER_W2 if w_el is None else w_el)
    b0 = float(CH3_LIK_W12_B0)
    return ch3_frame_lik_w12_3d(
        study_sep, exam_sep, y_sep, w1_c, w2_c, b0,
        mesh_pack=mesh_pack, z_lim=z_lim, curves=[],
        elev=CH3_LIK_W12_ELEV_W1, azim=CH3_LIK_W12_AZIM_W1,
        emphasize_knob="st", show_curves=False, marker=False,
    )


def _ch3_lik_w12_trace_knob1(study, exam, y, w2_fix, b, w1_from, w1_to, n):
    w1s = np.linspace(float(w1_from), float(w1_to), int(n), dtype=float)
    z = [_ch3_likelihood_on_flat_w12_grid(study, exam, y, [w], [w2_fix], [b])[0] for w in w1s]
    return w1s, np.full_like(w1s, float(w2_fix)), np.asarray(z, dtype=float)


def _ch3_lik_w12_trace_knob2(study, exam, y, w1_fix, b, w2_from, w2_to, n):
    w2s = np.linspace(float(w2_from), float(w2_to), int(n), dtype=float)
    z = [_ch3_likelihood_on_flat_w12_grid(study, exam, y, [w1_fix], [w], [b])[0] for w in w2s]
    return np.full_like(w2s, float(w1_fix)), w2s, np.asarray(z, dtype=float)


def _ch4_lik_02_03_handoff_state():
    """Shared end-of-02 / start-of-03 pose: weights (3, -3, 0), ±3 mesh, no marker."""
    study, exam, y = study_sep, exam_sep, y_sep
    lo = float(CH3_LIK_W12_3D_LO)
    hi = float(CH3_LIK_W12_3D_HI)
    b0 = float(CH3_LIK_W12_B0)
    mesh = ch3_lik_w12_mesh_pack(
        study, exam, y, b0,
        w1_lo=lo, w1_hi=hi, w2_lo=lo, w2_hi=hi,
        quadrant_fine=True,
    )
    return {
        "study": study, "exam": exam, "y": y,
        "w_st": hi, "w_el": lo, "b": b0,
        "mesh": mesh, "z_lim": ch3_lik_w12_z_limits(mesh["Z"], scale=1.0),
    }


def _ch4_lik_resolve_knob_pack(knob_labeled_blend):
    """Stable numbered / labeled / blended knob row — avoids blend flicker at u≈1."""
    if knob_labeled_blend is None:
        return None
    blends = tuple(float(x) for x in knob_labeled_blend)
    if all(b >= 1.0 - 1e-6 for b in blends):
        from ch4_layout import ch4_knob_asset_pack

        return ch4_knob_asset_pack()
    if all(b <= 1e-6 for b in blends):
        return ch3_knob_asset_pack()
    from ch4_layout import ch4_knob_asset_pack, ch4_knob_asset_pack_blended

    return ch4_knob_asset_pack_blended(
        ch3_knob_asset_pack(),
        ch4_knob_asset_pack(),
        blends,
    )


def _ch4_lik_02_03_handoff_plot(
    *,
    knob_labeled_blend=None,
    emphasize_knob=None,
    display_ws=None,
    display_we=None,
    display_b=None,
    marker=False,
    marker_ws=None,
    marker_we=None,
    marker_color=None,
    marker_edgecolors="white",
    ax3d_xlim=None,
    ax3d_ylim=None,
    elev=None,
    azim=None,
):
    """Full likelihood landscape at CT view — matches 02 end and 03 opening."""
    st = _ch4_lik_02_03_handoff_state()
    ws = float(display_ws if display_ws is not None else st["w_st"])
    we = float(display_we if display_we is not None else st["w_el"])
    bb = float(display_b if display_b is not None else st["b"])
    labeled = (
        knob_labeled_blend is not None
        and all(float(x) >= 1.0 - 1e-6 for x in knob_labeled_blend)
    )
    emp = "all" if labeled else ("st" if emphasize_knob is None else emphasize_knob)
    knob_pack = _ch4_lik_resolve_knob_pack(knob_labeled_blend)
    el = float(CH3_LIK_W12_CT_ELEV if elev is None else elev)
    az = float(CH3_LIK_W12_CT_AZIM if azim is None else azim)
    mesh = st["mesh"]
    z_lim = st["z_lim"]
    if ax3d_xlim is not None and ax3d_ylim is not None:
        xlo, xhi = float(ax3d_xlim[0]), float(ax3d_xlim[1])
        ylo, yhi = float(ax3d_ylim[0]), float(ax3d_ylim[1])
        mesh = ch3_lik_w12_mesh_pack(
            st["study"], st["exam"], st["y"], bb,
            w1_lo=xlo, w1_hi=xhi, w2_lo=ylo, w2_hi=yhi,
            grid_n=int(CH3_LIK_W12_ZOOM_GRID),
            quadrant_fine=False,
        )
        z_lim = ch3_lik_w12_z_limits(mesh["Z"], scale=1.0)
    return ch3_frame_lik_w12_3d(
        st["study"], st["exam"], st["y"], ws, we, bb,
        mesh_pack=mesh, z_lim=z_lim, curves=[],
        elev=el, azim=az, emphasize_knob=emp,
        landscape_reveal=1.0, landscape_reveal_origin="lo_hi",
        show_curves=False, marker=marker,
        knob_pack=knob_pack,
        knob_scales=[1.0, 1.0, 1.0],
        weight_axis_labels=labeled,
        z_label="Likelihood",
        marker_ws=marker_ws,
        marker_we=marker_we,
        marker_color=marker_color,
        marker_edgecolors=marker_edgecolors,
        ax3d_xlim=ax3d_xlim,
        ax3d_ylim=ax3d_ylim,
    )


def _ch4_02_compose_plot(plot_img):
    """Same shell as ch4_03 opening (layout_u=0, full-width plot, no rails)."""
    return _ch4_lik_03_opening_compose(plot_img, layout_u=0.0)


def ch3_build_frames_likelihood_w12_landscape_story():
    study, exam, y = study_sep, exam_sep, y_sep
    b0 = float(CH3_LIK_W12_B0)
    lo = float(CH3_LIK_W12_3D_LO)
    hi = float(CH3_LIK_W12_3D_HI)
    w1s, w2s = hi, lo  # start (knob1, knob2) = (3, -3)
    el_ct = float(CH3_LIK_W12_CT_ELEV)
    az_ct = float(CH3_LIK_W12_CT_AZIM)
    el_w1 = float(CH3_LIK_W12_ELEV_W1)
    az_w1 = float(CH3_LIK_W12_AZIM_W1)
    el_w2 = float(CH3_LIK_W12_ELEV_W2)
    az_w2 = float(CH3_LIK_W12_AZIM_W2)

    mesh0 = ch3_lik_w12_mesh_pack(
        study, exam, y, b0,
        w1_lo=lo, w1_hi=hi,
        w2_lo=lo, w2_hi=hi,
        quadrant_fine=True,
    )
    z_lik_hi = float(np.nanmax(mesh0["Z"]))
    z_lim_full = ch3_lik_w12_z_limits(mesh0["Z"], scale=1.0)
    n_trace = max(24, CH3_LIK_W12_N_KNOB)

    frames = []
    curves = []

    def emit(
        ws, we, bb, *, elev, azim, emp="st", lrev=0.0, show_curves=True,
        z_lim=None, marker=True, srev=0.0, marker_z_offset=0.0, stack_layers=None,
        stack_n_layers=1.0, z_lik_ref=None, landscape_reveal_origin="lo_lo",
    ):
        fr = ch3_frame_lik_w12_3d(
            study, exam, y, ws, we, bb,
            mesh_pack=mesh0, z_lim=z_lim or z_lim_full,
            curves=list(curves) if show_curves else [],
            elev=elev, azim=azim, emphasize_knob=emp,
            landscape_reveal=lrev, landscape_reveal_origin=landscape_reveal_origin,
            slice_reveal=srev,
            show_curves=show_curves, marker=marker,
            marker_z_offset=marker_z_offset, stack_layers=stack_layers,
            stack_n_layers=stack_n_layers,
            z_lik_ref=z_lik_hi if z_lik_ref is None else z_lik_ref,
        )
        frames.append(_ch4_02_compose_plot(fr))

    def hold(ws, we, bb, n, **kw):
        for _ in range(int(n)):
            emit(ws, we, bb, **kw)

    # 1–2: empty 3D at (3, -3)
    open_fr = ch3_lik_w12_frame_opening(mesh0, z_lim_full, w_st=w1s, w_el=w2s)
    open_comp = _ch4_02_compose_plot(open_fr)
    for _ in range(CH3_LIK_W12_N_HOLD):
        frames.append(open_comp.copy())

    # 3: knob 1 — w1: 3 → -3 (w2 = -3)
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_KNOB, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        w1c = ch3_lerp(hi, lo, u)
        cw1, cw2, cz = _ch3_lik_w12_trace_knob1(study, exam, y, w2s, b0, hi, w1c, n_trace)
        if curves:
            curves[-1] = (cw1, cw2, cz)
        else:
            curves.append((cw1, cw2, cz))
        emit(w1c, w2s, b0, elev=el_w1, azim=az_w1, emp="st")
    hold(lo, w2s, b0, CH3_LIK_W12_N_HOLD // 2, elev=el_w1, azim=az_w1)
    curves[-1] = _ch3_lik_w12_trace_knob1(study, exam, y, w2s, b0, hi, lo, n_trace)

    # 3b: knob 1 — w1: -3 → 3
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_KNOB, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        w1c = ch3_lerp(lo, hi, u)
        emit(w1c, w2s, b0, elev=el_w1, azim=az_w1, emp="st")
    hold(w1s, w2s, b0, CH3_LIK_W12_N_HOLD // 2, elev=el_w1, azim=az_w1)

    # 4: shortest rotation to knob-2 view
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_ROT, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        emit(
            w1s, w2s, b0,
            elev=ch3_lerp(el_w1, el_w2, u),
            azim=_ch3_lik_w12_lerp_azim_shortest(az_w1, az_w2, u),
        )

    # 5: knob 2 — w2: -3 → 3 (w1 = 3)
    curves.append(_ch3_lik_w12_trace_knob2(study, exam, y, w1s, b0, w2s, w2s, 2))
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_KNOB, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        w2c = ch3_lerp(w2s, hi, u)
        cw1, cw2, cz = _ch3_lik_w12_trace_knob2(study, exam, y, w1s, b0, w2s, w2c, n_trace)
        curves[-1] = (cw1, cw2, cz)
        emit(w1s, w2c, b0, elev=el_w2, azim=az_w2, emp="el")
    hold(w1s, hi, b0, CH3_LIK_W12_N_HOLD // 2, elev=el_w2, azim=az_w2)

    # 5b: knob 2 — w2: 3 → -3
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_KNOB, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        w2c = ch3_lerp(hi, w2s, u)
        emit(w1s, w2c, b0, elev=el_w2, azim=az_w2, emp="el")
    hold(w1s, w2s, b0, CH3_LIK_W12_N_HOLD // 2, elev=el_w2, azim=az_w2)

    # 6: shortest rotation to ch4_03 CT view — keep for fill + landscape
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_ROT, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        emit(
            w1s, w2s, b0,
            elev=ch3_lerp(el_w2, el_ct, u),
            azim=_ch3_lik_w12_lerp_azim_shortest(az_w2, az_ct, u),
        )

    # 7: raster fill — evenly spaced curves (surface mesh stays finer in one quadrant)
    w1_vals = np.linspace(lo, hi, int(CH3_LIK_W12_N_FILL_LINES), dtype=np.float64)
    w2_vals = np.linspace(lo, hi, int(CH3_LIK_W12_N_FILL_LINES), dtype=np.float64)
    n_trace_fill = int(CH3_LIK_W12_N_FILL_TRACE)
    for w1v in w1_vals:
        cw1, cw2, cz = _ch3_lik_w12_trace_knob2(
            study, exam, y, float(w1v), b0, lo, hi, n_trace_fill,
        )
        curves.append((cw1, cw2, cz))
        emit(w1s, w2s, b0, elev=el_ct, azim=az_ct, emp="st", marker=True)
    for w2v in w2_vals:
        cw1, cw2, cz = _ch3_lik_w12_trace_knob1(
            study, exam, y, float(w2v), b0, lo, hi, n_trace_fill,
        )
        curves.append((cw1, cw2, cz))
        emit(w1s, w2s, b0, elev=el_ct, azim=az_ct, emp="el", marker=True)

    hold(w1s, w2s, b0, CH3_LIK_W12_N_HOLD // 2, elev=el_ct, azim=az_ct, marker=True)

    # 8: diagonal landscape reveal from (-3, 3); then drop marker
    for tv in np.linspace(0.0, 1.0, CH3_LIK_W12_N_REVEAL, endpoint=True):
        u = ch3_knob_smoothstep(float(tv))
        emit(
            w1s, w2s, b0, elev=el_ct, azim=az_ct,
            lrev=u, show_curves=True, marker=False,
            landscape_reveal_origin="lo_hi",
        )
    emit(
        w1s, w2s, b0, elev=el_ct, azim=az_ct, lrev=1.0,
        show_curves=False, marker=False, landscape_reveal_origin="lo_hi",
    )
    handoff_comp = _ch4_02_compose_plot(_ch4_lik_02_03_handoff_plot())
    for _ in range(CH3_LIK_W12_N_HOLD):
        frames.append(handoff_comp.copy())
    return frames


def ch4_preview_likelihood_w12_landscape_last_frame():
    return _ch4_02_compose_plot(_ch4_lik_02_03_handoff_plot())


def ch4_export_likelihood_w12_landscape():
    frames = ch3_build_frames_likelihood_w12_landscape_story()
    fn = "ch4_02_likelihood_w12_landscape.mp4"
    save_mp4(frames, fn, duration=int(CH3_LIK_W12_MS))
    print("wrote", OUTPUT_DIR / fn)
    return OUTPUT_DIR / fn
