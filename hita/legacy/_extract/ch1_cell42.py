# 4) 64 — flat z=0, bend to colormap σ surface; tilt to top-down (half-degree steps);
# fixed left exam-length label near top; strip z ticks/labels once top; then 32° azimuth sweep.
frames = []
_az64 = float(extra_angles[0])
for _ in range(SIG3D_FLAT_HOLD):
    fig = plt.figure(figsize=SIG3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(ST, EL, Z_flat, alpha=0.32, cmap=CMAP, vmin=0, vmax=1, linewidth=0, antialiased=True, shade=False)
    scatter_whole_data_by_class(ax, zeros_pts, rotate_icons_180=True)
    style_sigmoid_axes(ax, _az64)
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
    _az_morph = float(_az64 + (topdown_azimuth - _az64) * su)
    style_sigmoid_axes(ax, _az_morph)
    frames.append(sig3d_frame_png(fig))

# Phase A: tilt upward (half-degree elevation steps)
for elev in topdown_elevations:
    _near_top = elev >= 87.5
    _exam2d = elev >= 78.0
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
    style_sigmoid_axes(
        ax,
        topdown_azimuth,
        elev=float(elev),
        hide_z=_near_top,
        exam_label_2d=_exam2d,
    )
    frames.append(sig3d_frame_png(fig))

# Phase B: at top, rotate counterclockwise 32 degrees (half-degree azimuth steps)
for az in counterclockwise_turn:
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
    style_sigmoid_axes(ax, float(az), elev=SIG3D_TOP_ELEV1, hide_z=True, exam_label_2d=True)
    frames.append(sig3d_frame_png(fig))

hold = frames[-1]
for _ in range(TOPDOWN_END_HOLD_FRAMES):
    frames.append(hold.copy())

save_gif(frames, "75_sigmoid_colormap_to_topdown.gif", duration=ROTATION_MS)
