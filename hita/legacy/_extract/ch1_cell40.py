# 3) 63 — flat z=0 colormap surface, bend to σ(ST−EL), then one full 360° orbit
frames = []
_az63 = float(extra_angles[0])
for _ in range(SIG3D_FLAT_HOLD):
    fig = plt.figure(figsize=SIG3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(ST, EL, Z_flat, alpha=0.32, cmap=CMAP, vmin=0, vmax=1, linewidth=0, antialiased=True, shade=False)
    scatter_whole_data_by_class(ax, zeros_pts, rotate_icons_180=True)
    style_sigmoid_axes(ax, _az63, diag_prob_scale=0.0)
    frames.append(sig3d_frame_png(fig))
for i in range(SIG3D_MORPH_FRAMES):
    u = i / (SIG3D_MORPH_FRAMES - 1) if SIG3D_MORPH_FRAMES > 1 else 1.0
    su = _sig2d_smoothstep(u)
    Zc = su * P_PASS
    zpt = su * _z_pass_pts
    fig = plt.figure(figsize=SIG3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(ST, EL, Zc, alpha=0.32, cmap=CMAP, vmin=0, vmax=1, linewidth=0, antialiased=True, shade=False)
    scatter_whole_data_by_class(ax, zpt, rotate_icons_180=True)
    style_sigmoid_axes(ax, _az63, diag_prob_scale=su)
    frames.append(sig3d_frame_png(fig))
for az in extra_angles:
    fig = plt.figure(figsize=SIG3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        ST,
        EL,
        P_PASS,
        alpha=0.32,
        cmap=CMAP,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    scatter_whole_data_by_class(ax, _z_pass_pts, rotate_icons_180=True)
    style_sigmoid_axes(ax, az)
    frames.append(sig3d_frame_png(fig))
save_gif(frames, "74_sigmoid_pass_colormap.gif", duration=ROTATION_MS)
