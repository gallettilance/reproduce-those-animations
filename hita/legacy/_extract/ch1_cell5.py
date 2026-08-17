def draw_dataset(ax, study_vals, exam_vals, labels, mask=None, alpha=0.95, title=None):
    if mask is None:
        mask = np.ones(len(study_vals), dtype=bool)

    for i in np.where(mask)[0]:
        add_outcome_icon(ax, study_vals[i], exam_vals[i], passed=bool(labels[i]), alpha=alpha)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)


def add_combined_legend(ax, loc="upper left"):
    line_handles, line_labels = ax.get_legend_handles_labels()
    merged_handles = []
    merged_labels = []
    seen = set()

    for h, lbl in zip(line_handles, line_labels):
        if lbl and lbl not in seen:
            merged_handles.append(h)
            merged_labels.append(lbl)
            seen.add(lbl)

    if merged_handles:
        ax.legend(
            handles=merged_handles,
            labels=merged_labels,
            loc=loc,
            prop={"size": LEGEND_SIZE},
        )


def add_threshold_line(ax, shift=0.0, label=None, style="--", color=NEUTRAL_COLOR, linewidth=1.0, x_range=None):
    x0, x1 = x_range if x_range is not None else xlim
    x = np.linspace(x0, x1, 200)
    y_line = x - shift
    ax.plot(x, y_line, style, color=color, linewidth=linewidth, label=label)



def draw_axes_only(ax):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)


def draw_axes_only_study_bold(ax):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(
        "Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10, fontweight="bold"
    )
    ax.set_ylabel("Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)


def draw_axes_only_exam_bold(ax):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Study time (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel(
        "Exam length (hours)", fontsize=AXIS_LABEL_SIZE, labelpad=10, fontweight="bold"
    )
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)


def shade_pass_half(ax, shift=0.0, alpha=0.22):
    xa, xb = xlim
    ya, yb = ylim
    xs = np.linspace(xa, xb, 400)
    y_line = xs - shift
    top = np.minimum(y_line, yb)
    bot = np.full_like(xs, ya)
    ax.fill_between(xs, bot, top, where=(top > bot), alpha=alpha, color=PASS_COLOR, linewidth=0, zorder=0)


def shade_fail_half(ax, shift=0.0, alpha=0.22):
    xa, xb = xlim
    ya, yb = ylim
    xs = np.linspace(xa, xb, 400)
    y_line = xs - shift
    bot2 = np.maximum(y_line, ya)
    ax.fill_between(xs, bot2, yb, where=(yb > bot2), alpha=alpha, color=FAIL_COLOR, linewidth=0, zorder=0)


def draw_z_axis_panel(ax, z_value, label):
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter([z_value], [0], color="#ff7f0e", s=150, zorder=5)
    ax.text(z_value + 0.05, 0.08, label, fontsize=NOTE_SIZE)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("ST - EL", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.tick_params(axis="x", which="major", labelsize=FONT_SIZE)
    ax.grid(alpha=0.15)


def save_fig(fig, filename, dpi=None, tight_layout=False):
    """Same PNG raster path as ``fig_to_image`` (and thus GIF frames); writes one file under ``OUTPUT_DIR``."""
    im = fig_to_image(fig, dpi=dpi, tight_layout=tight_layout)
    im.save(OUTPUT_DIR / filename)
