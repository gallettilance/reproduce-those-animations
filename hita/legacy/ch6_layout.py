"""Chapter 6 — dual-panel 16:9 layout (2D + knobs | topic panel), no formula rails."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec

from ch4_layout import (
    CH4_DUO_PLOTS_X_SHIFT_PT,
    CH4_DUO_WIDTH_RATIOS,
    CH4_FIGSIZE,
    OUTPUT_DIR,
    _ch4_fig_x_shift_frac,
)

# Full 16:9 canvas — plot only, no tutorial compose step.
CH6_FIGSIZE = CH4_FIGSIZE
CH6_EXPORT_DPI = 200
CH6_WIDTH_RATIOS = CH4_DUO_WIDTH_RATIOS
CH6_DATA_SCALE = 0.90
CH6_RIGHT_WIDTH_SCALE = 1.05  # width only; height set after knobs via ch6_duo_right_panel_extent

# --- Datasets (study hrs, exam hrs, label) ---

# Noisy roster: separable core + boundary flips → miscalibrated probabilities.
CH6_CALIBRATION_POINTS = [
    (2, 3, 0), (4, 5, 0), (5, 6, 0), (1, 3, 0), (2, 4, 0), (4, 6, 0),
    (1, 4, 0), (3, 6, 0), (1, 6, 0),
    (3, 2, 1), (5, 4, 1), (6, 5, 1), (4, 2, 1), (6, 4, 1), (3, 1, 1),
    (4, 1, 1), (5, 2, 1), (6, 3, 1), (6, 2, 1), (6, 1, 1),
    (2, 1, 0), (1, 2, 1), (3, 4, 1), (4, 3, 0), (3, 5, 1), (5, 3, 0),
]

# Ambiguous zone near the boundary → interesting ROC trade-offs.
CH6_ROC_POINTS = [
    (1, 5, 0), (2, 5, 0), (2, 4, 0), (1, 4, 0),
    (3, 4, 0), (4, 4, 0), (3, 3, 1), (4, 3, 1),
    (2, 3, 1), (3, 2, 1), (4, 2, 1), (5, 2, 1),
    (5, 3, 1), (6, 3, 1), (5, 4, 1), (6, 4, 1),
    (6, 5, 1), (5, 5, 1), (4, 5, 1), (6, 6, 1),
    (6, 1, 1), (5, 1, 1), (6, 2, 1),
]

# Heavy pass skew (20 pass / 5 fail) — minority fails upper-left.
CH6_IMBALANCED_POINTS = [
    (3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1), (3, 2, 1), (4, 2, 1),
    (5, 2, 1), (6, 2, 1), (3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 1),
    (4, 4, 1), (5, 4, 1), (6, 4, 1), (5, 5, 1), (6, 5, 1), (6, 6, 1),
    (4, 5, 1), (5, 6, 1),
    (1, 4, 0), (1, 5, 0), (2, 5, 0), (2, 6, 0), (3, 6, 0),
]

CH6_WEIGHTS_CALIBRATION = (0.50, -0.50, 0.0)
CH6_WEIGHTS_ROC = (0.55, -0.55, 0.25)
CH6_WEIGHTS_IMBALANCED = (0.50, -0.50, -1.0)
CH6_WEIGHTS_ROC_ALT = (0.35, -0.35, 1.5)


def ch6_unpack_points(point_list):
    arr = np.array(point_list, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(int)


def ch6_duo_layout_tune(
    fig,
    ax_data,
    ax_right,
    *,
    data_scale: float | None = None,
    right_width_scale: float | None = None,
    x_shift_pt: float | None = None,
) -> None:
    """Shrink left 2D; set provisional right-panel width (height finalized after knobs)."""
    data_scale = CH6_DATA_SCALE if data_scale is None else float(data_scale)
    right_width_scale = CH6_RIGHT_WIDTH_SCALE if right_width_scale is None else float(right_width_scale)
    x_shift_pt = CH4_DUO_PLOTS_X_SHIFT_PT if x_shift_pt is None else float(x_shift_pt)
    x_shift = _ch4_fig_x_shift_frac(fig, x_shift_pt)
    fig.canvas.draw()
    d = ax_data.get_position()
    r = ax_right.get_position()
    dw = d.width * data_scale
    dh = d.height * data_scale
    dx = d.x0 + 0.5 * (d.width - dw) + x_shift
    dy = d.y0 + 0.5 * (d.height - dh)
    ax_data.set_position([dx, dy, dw, dh])
    rw = r.width * right_width_scale
    rx = r.x0 + 0.5 * (r.width - rw) + x_shift
    ax_right.set_position([rx, dy, rw, dh])


def ch6_duo_right_panel_extent(fig, ax_data, ax_right, axes_k) -> None:
    """Right panel: top aligned with 2D plot, bottom aligned with knob-row bottom."""
    fig.canvas.draw()
    d = ax_data.get_position()
    r = ax_right.get_position()
    knob_bottom = min(float(ax.get_position().y0) for ax in axes_k)
    y_top = float(d.y1)
    y_bottom = knob_bottom
    rh = max(y_top - y_bottom, 1e-3)
    ax_right.set_position([float(r.x0), y_bottom, float(r.width), rh])


def ch6_figure_duo():
    """16:9 figure: left = 2D + knob row, right = open axes for topic panel."""
    fig = plt.figure(figsize=CH6_FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=CH6_WIDTH_RATIOS, wspace=0.16)
    g_left = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=(1.0, 0.34), hspace=0.12,
    )
    ax_data = fig.add_subplot(g_left[0, 0])
    g_k = GridSpecFromSubplotSpec(1, 3, subplot_spec=g_left[1, 0], wspace=0.08)
    axes_k = tuple(fig.add_subplot(g_k[0, j]) for j in range(3))
    ax_right = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.06)
    ch6_duo_layout_tune(fig, ax_data, ax_right)
    return fig, ax_data, ax_right, axes_k


def ch6_roc_curve(y, scores):
    """Manual ROC: return fpr, tpr arrays (includes (0,0) and (1,1))."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    y_sorted = y[order]
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - y.sum()), 1)
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    for lab in y_sorted:
        if lab:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    return np.asarray(fpr), np.asarray(tpr)


def ch6_auc(fpr, tpr):
    return float(np.trapezoid(tpr, fpr))


def ch6_style_right_axis(ax, *, title=None):
    ax.tick_params(axis="both", which="major", labelsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.8)


# --- Right-panel drawers (take predicted probabilities ``p``) ---

def ch6_draw_calibration_reliability(ax, y, p, *, n_bins=8):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    centers, obs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(m):
            continue
        centers.append(0.5 * (lo + hi))
        obs.append(float(y[m].mean()))
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.6, label="Perfect")
    ax.scatter(centers, obs, s=70, c="#2a7", edgecolors="0.15", linewidths=0.8, zorder=3)
    ax.plot(centers, obs, color="#2a7", linewidth=1.4, alpha=0.85, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(pass)")
    ax.set_ylabel("Observed pass rate")
    ax.legend(loc="upper left", fontsize=10)
    ch6_style_right_axis(ax, title="Reliability diagram")


def ch6_draw_calibration_histogram(ax, y, p, *, n_bins=12):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    fail_hist, _ = np.histogram(p[y == 0], bins=bins)
    pass_hist, _ = np.histogram(p[y == 1], bins=bins)
    xc = 0.5 * (bins[:-1] + bins[1:])
    w = 0.85 * (bins[1] - bins[0])
    ax.bar(xc, pass_hist, width=w, color="#2a7", alpha=0.85, label="Pass")
    ax.bar(xc, fail_hist, width=w, bottom=pass_hist, color="#c44", alpha=0.85, label="Fail")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted P(pass)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=10)
    ch6_style_right_axis(ax, title="Predicted probability by outcome")


def ch6_draw_calibration_gap(ax, y, p, *, n_bins=8):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    centers, gaps = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(m):
            continue
        pred = float(p[m].mean())
        obs = float(y[m].mean())
        centers.append(0.5 * (lo + hi))
        gaps.append(obs - pred)
    colors = ["#c44" if g < 0 else "#2a7" for g in gaps]
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.4)
    ax.bar(centers, gaps, width=0.85 / n_bins, color=colors, alpha=0.88, edgecolor="0.15")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted P(pass) bin")
    ax.set_ylabel("Observed − predicted")
    ch6_style_right_axis(ax, title="Calibration gap")


def ch6_draw_roc_standard(ax, y, p):
    p = np.asarray(p, dtype=float)
    fpr, tpr = ch6_roc_curve(y, p)
    auc = ch6_auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#06c", linewidth=2.2, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=10)
    ch6_style_right_axis(ax, title="ROC curve")


def ch6_draw_roc_thresholds(ax, y, p, *, thresholds=(0.3, 0.5, 0.7)):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    fpr, tpr = ch6_roc_curve(y, p)
    ax.plot(fpr, tpr, color="#06c", linewidth=2.0)
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.2)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - y.sum()), 1)
    for thr in thresholds:
        pred = p >= float(thr)
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tpr_pt = tp / n_pos
        fpr_pt = fp / n_neg
        ax.scatter([fpr_pt], [tpr_pt], s=55, zorder=4)
        ax.annotate(
            f"{thr:.1f}", (fpr_pt, tpr_pt), textcoords="offset points",
            xytext=(6, 4), fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ch6_style_right_axis(ax, title="ROC with thresholds")


def ch6_draw_roc_compare(ax, y, p_main, p_alt):
    p_main = np.asarray(p_main, dtype=float)
    p_alt = np.asarray(p_alt, dtype=float)
    fpr1, tpr1 = ch6_roc_curve(y, p_main)
    fpr2, tpr2 = ch6_roc_curve(y, p_alt)
    ax.plot(fpr1, tpr1, color="#06c", linewidth=2.0, label=f"Model A (AUC {ch6_auc(fpr1, tpr1):.2f})")
    ax.plot(fpr2, tpr2, color="#c60", linewidth=2.0, label=f"Model B (AUC {ch6_auc(fpr2, tpr2):.2f})")
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=9)
    ch6_style_right_axis(ax, title="ROC comparison")


def ch6_draw_imbalance_bars(ax, y, p):
    y = np.asarray(y, dtype=int)
    n_pass = int(np.sum(y == 1))
    n_fail = int(len(y) - n_pass)
    ax.bar(["Pass", "Fail"], [n_pass, n_fail], color=["#2a7", "#c44"], alpha=0.88, edgecolor="0.15")
    ax.set_ylabel("Students")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ratio = n_pass / max(n_fail, 1)
    ax.text(
        0.5, 0.92, f"ratio {ratio:.1f}:1", transform=ax.transAxes,
        ha="center", fontsize=11, color="0.25",
    )
    ch6_style_right_axis(ax, title="Class counts")


def ch6_draw_imbalance_confusion(ax, y, p, *, threshold=0.5):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = p >= float(threshold)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=16, color="0.1")
    ax.set_xticks([0, 1], labels=["Pred fail", "Pred pass"])
    ax.set_yticks([0, 1], labels=["Actual fail", "Actual pass"])
    ax.set_xlabel(f"Threshold = {threshold:.1f}")
    ch6_style_right_axis(ax, title="Confusion matrix")


def ch6_draw_imbalance_pr(ax, y, p):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-p)
    y_sorted = y[order]
    n_pos = max(int(y.sum()), 1)
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / n_pos
    ax.plot(rec, prec, color="#906", linewidth=2.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    baseline = n_pos / max(len(y), 1)
    ax.axhline(baseline, color="0.35", linestyle="--", linewidth=1.2, label=f"Baseline {baseline:.2f}")
    ax.legend(loc="upper right", fontsize=10)
    ch6_style_right_axis(ax, title="Precision–recall")


def ch6_save_preview(img, filename: str) -> Path:
    out = OUTPUT_DIR / filename
    out.parent.mkdir(exist_ok=True)
    img.save(out)
    return out
