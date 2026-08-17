# --- ch2_00c — 3D: points rise on z = ST−EL, σ-colored logit plane, morph to σ surface ---
# Prerequisites: Cell 1 + dataset config cell (**`reveal_order`**, **`ST_A`** / **`EL_A`**, **`SIG3D_ADJ`**, …).


def _frames_ch2_00c_dataset_st_el_3d_sigmoid_morph():
    """Scatter on z = ST−EL, then a tilted **logit** sheet (**Z = ST − EL**) painted by **σ(ST−EL)**; morph **Z** and **z-limits** to the curved **σ** surface."""
    Z_log = ST_A - EL_A
    S_m = sigmoid(Z_log)
    z_lo_m = float(Z_log.min())
    z_hi_m = float(Z_log.max())
    z_lo_raw = min(0.0, z_lo_m, float(np.min(z_sep)))
    z_hi_raw = max(0.0, z_hi_m, float(np.max(z_sep)))
    pad = 0.45
    zlim0 = (z_lo_raw - pad, z_hi_raw + pad)

    _norm01 = plt.Normalize(0.0, 1.0)

    elev, azim = 32.0, -152.0
    zs_sigma = sigmoid(z_sep.astype(float))

    _z0_plane_n = 28

    def draw_axes(ax, z_lo, z_hi, zlbl):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(z_lo, z_hi)
        ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=11)
        ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=11)
        ax.set_zlabel(zlbl, fontsize=AXIS_LABEL_SIZE, labelpad=8)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 1)

    def draw_z0_plane_and_height_lines(ax, zs_arr):
        """Grey **z = 0** reference sheet + vertical segments from the floor to each marker (orthogonal to the floor)."""
        gx = np.linspace(float(xlim[0]), float(xlim[1]), _z0_plane_n)
        gy = np.linspace(float(ylim[0]), float(ylim[1]), _z0_plane_n)
        GX, GY = np.meshgrid(gx, gy)
        GZ = np.zeros_like(GX, dtype=float)
        ax.plot_surface(
            GX,
            GY,
            GZ,
            color=(0.72, 0.72, 0.72),
            alpha=0.14,
            linewidth=0,
            antialiased=True,
            shade=False,
            zorder=1,
        )
        for i in range(n_sep):
            zi = float(zs_arr[i])
            if zi <= 1e-9:
                continue
            st, el = float(study_sep[i]), float(exam_sep[i])
            ax.plot(
                [st, st],
                [el, el],
                [0.0, zi],
                color="#5c5c5c",
                linewidth=1.35,
                alpha=0.42,
                zorder=2,
            )

    def _fc_sigma_for_Z(Z_grid):
        Zq = 0.25 * (
            Z_grid[:-1, :-1]
            + Z_grid[1:, :-1]
            + Z_grid[:-1, 1:]
            + Z_grid[1:, 1:]
        )
        return CMAP_GD(_norm01(sigmoid(Zq)))

    def snap(zs_arr, *, Z_surf, show_surface, surf_alpha, z_lo, z_hi, zlbl, cmap_mix=1.0):
        fig = plt.figure(figsize=EXPORT_FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(**SIG3D_ADJ)
        if not show_surface:
            draw_z0_plane_and_height_lines(ax, zs_arr)
        if show_surface and Z_surf is not None:
            fc_s = _fc_sigma_for_Z(Z_surf)
            cm = float(np.clip(cmap_mix, 0.0, 1.0))
            grey = 0.78
            blend = fc_s.copy()
            blend[..., :3] = (1.0 - cm) * grey + cm * fc_s[..., :3]
            ax.plot_surface(
                ST_A,
                EL_A,
                Z_surf,
                facecolors=blend,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=float(surf_alpha),
            )
        for i in range(n_sep):
            c = PASS_COLOR if y_sep[i] else FAIL_COLOR
            ax.scatter(
                [float(study_sep[i])],
                [float(exam_sep[i])],
                [float(zs_arr[i])],
                color=c,
                s=54,
                depthshade=True,
                zorder=6,
            )
        draw_axes(ax, z_lo, z_hi, zlbl)
        return fig_to_image(fig)

    out = []
    zs = np.zeros(n_sep, dtype=float)
    z_pre = r"$\mathrm{ST}-\mathrm{EL}$"
    z_post = r"$\sigma(\mathrm{ST}-\mathrm{EL})$"

    HOLD_ZERO = 20
    RISE_N = 9
    HOLD_RISEN = 14
    HOLD_PLANE = 16
    MORPH_N = 52
    HOLD_END = 20

    for _ in range(HOLD_ZERO):
        out.append(
            snap(
                zs,
                Z_surf=None,
                show_surface=False,
                surf_alpha=0.0,
                z_lo=zlim0[0],
                z_hi=zlim0[1],
                zlbl=z_pre,
            )
        )

    for j in reveal_order:
        tgt = float(z_sep[j])
        for k in range(RISE_N):
            t = float(k + 1) / RISE_N
            zc = zs.copy()
            zc[j] = tgt * t
            out.append(
                snap(
                    zc,
                    Z_surf=None,
                    show_surface=False,
                    surf_alpha=0.0,
                    z_lo=zlim0[0],
                    z_hi=zlim0[1],
                    zlbl=z_pre,
                )
            )
        zs[j] = tgt

    for _ in range(HOLD_RISEN):
        out.append(
            snap(
                zs,
                Z_surf=None,
                show_surface=False,
                surf_alpha=0.0,
                z_lo=zlim0[0],
                z_hi=zlim0[1],
                zlbl=z_pre,
            )
        )

    HOLD_PLANE_GREY = max(6, HOLD_PLANE // 3)
    HOLD_PLANE_CMAP = max(HOLD_PLANE - HOLD_PLANE_GREY, 1)
    for _ in range(HOLD_PLANE_GREY):
        out.append(
            snap(
                zs,
                Z_surf=Z_log,
                show_surface=True,
                surf_alpha=0.48,
                z_lo=zlim0[0],
                z_hi=zlim0[1],
                zlbl=z_pre,
                cmap_mix=0.0,
            )
        )
    for ic in range(HOLD_PLANE_CMAP):
        u = float(ic) / max(HOLD_PLANE_CMAP - 1, 1)
        out.append(
            snap(
                zs,
                Z_surf=Z_log,
                show_surface=True,
                surf_alpha=0.48,
                z_lo=zlim0[0],
                z_hi=zlim0[1],
                zlbl=z_pre,
                cmap_mix=u,
            )
        )

    for m in range(MORPH_N):
        t = float(m) / max(MORPH_N - 1, 1)
        Zs = (1.0 - t) * Z_log + t * S_m
        z_lo_t = (1.0 - t) * zlim0[0] + t * (-0.02)
        z_hi_t = (1.0 - t) * zlim0[1] + t * 1.02
        zsc = (1.0 - t) * zs + t * zs_sigma
        lbl = z_post if t >= 0.5 else z_pre
        out.append(
            snap(
                zsc,
                Z_surf=Zs,
                show_surface=True,
                surf_alpha=0.48,
                z_lo=z_lo_t,
                z_hi=z_hi_t,
                zlbl=lbl,
                cmap_mix=1.0,
            )
        )

    last = out[-1]
    for _ in range(HOLD_END):
        out.append(last.copy())
    CMAP_HOLD_FINAL = 10
    for _ in range(CMAP_HOLD_FINAL):
        out.append(last.copy())

    return out

