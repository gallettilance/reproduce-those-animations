"""Chapter 6 — Bayesian posterior grids and density helpers."""
from __future__ import annotations

import os

import numpy as np

from ch5_datasets import CH5_DATASET_KEYS, CH5_DATASET_META, ch5_unpack_dataset

_CH3_DRAFT = os.environ.get("CH3_DRAFT_EXPORT", "").strip().lower() in {"1", "true", "yes", "y"}
_CH5_PRIOR_LAND_FULL = os.environ.get("CH5_PRIOR_LANDSCAPE_FULL", "").strip().lower() in {"1", "true", "yes", "y"}

CH5_PRIOR_SIGMA = 2.0
CH5_PRIOR_LANDSCAPE_SIGMA = 1.0  # N(0, I) on w_ST, w_EL — contained in ±3 view
CH5_VIEW_BOUNDS = (-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)
CH5_UNIFORM_AXIS_SHRINK = 1.0  # uniform support span = axis span − this (±0.5 per side on [-3,3])
CH5_W12_B_FIXED = 0.0
CH5_DENSITY_Z_HI = 1.0
# Global 3D color scale: log-posterior ref (peak across D1–D4 on voxel + w_ST–w_EL grids).
# Overwritten by ``ch5_calibrate_global_density_scales`` at story install.
CH5_GLOBAL_LOG_POST_REF: dict[str, float] = {
    "gaussian": -1.8857953690116305,
    "uniform": -0.6193882846488873,
}
# Max 2D marginal belief-pdf peak on the canonical w_ST–w_EL grid (D1–D4).
# Overwritten by ``ch5_calibrate_global_density_scales`` at story install.
CH5_BELIEF_W12_Z_HI: dict[str, float] = {
    "gaussian": 10.941,
    "uniform": 10.536,
}
# Max belief-pdf peak on D1 (HQ sequential / ch5_44–45 landscape clips).
CH5_BELIEF_W12_Z_HI_D1: dict[str, float] = {
    "gaussian": 1.214,
    "uniform": 1.382,
}
# Tempering for belief-surface display: low when data are uninformative (D4 stays wide).
CH5_INFO_SCALE = 4.0
CH5_GRID = 14 if _CH3_DRAFT else 26
CH5_CT_GRID = 10 if _CH3_DRAFT else 18
CH5_VOXEL_GRID = 14 if _CH3_DRAFT else 28
CH5_CREDIBLE_MASS = 0.95
CH5_VOXEL_N_FILL = 8 if _CH3_DRAFT else 48
CH5_VOXEL_N_ROT = 6 if _CH3_DRAFT else 48
CH5_VOXEL_N_PROJ_SHADOW = 6 if _CH3_DRAFT else 20
CH5_VOXEL_N_PROJ_COLLAPSE = 6 if _CH3_DRAFT else 24
CH5_VOXEL_ORBIT_DEG = 360.0
CH5_VOXEL_N_ORBIT = 8 if _CH3_DRAFT else 96
CH5_MS = 120 if _CH3_DRAFT else 95
CH5_N_HOLD = 3 if _CH3_DRAFT else 10
CH5_CT_N_SWEEP = 6 if _CH3_DRAFT else 40
CH5_CT_N_PIVOT = 4 if _CH3_DRAFT else 56
CH5_N_DECOMP = 6 if _CH3_DRAFT else 32
CH5_N_SEQ_HOLD = 2 if _CH3_DRAFT else 4
CH5_BELIEF_SURFACE_ALPHA = 0.72
CH5_PLANE_ALPHA = 0.50
# Uniform belief colormap: t = (pdf/z_hi)^gamma — gamma < 1 boosts visibility of low pdf (prior shelf).
CH5_UNIFORM_BELIEF_COLOR_GAMMA = 0.32

# Prior landscape clips (ch5_04b / ch5_04c) — preview-grade by default; set CH5_PRIOR_LANDSCAPE_FULL=1 for final.
CH5_PRIOR_LAND_GRID = 52 if _CH5_PRIOR_LAND_FULL else 12
CH5_PRIOR_LAND_GRID_COARSE = 22 if _CH5_PRIOR_LAND_FULL else 8
CH5_PRIOR_LAND_GRID_FINE = 36 if _CH5_PRIOR_LAND_FULL else 14
CH5_PRIOR_LAND_QUADRANT_FINE = bool(_CH5_PRIOR_LAND_FULL)
CH5_PRIOR_LAND_N_KNOB = 28 if _CH5_PRIOR_LAND_FULL else 6
CH5_PRIOR_LAND_N_ROT = 64 if _CH5_PRIOR_LAND_FULL else 6
CH5_PRIOR_LAND_N_REVEAL = 40 if _CH5_PRIOR_LAND_FULL else 8
CH5_PRIOR_LAND_N_FILL_LINES = CH5_PRIOR_LAND_GRID if _CH5_PRIOR_LAND_FULL else 10
CH5_PRIOR_LAND_N_FILL_TRACE = 48 if _CH5_PRIOR_LAND_FULL else 12
CH5_PRIOR_LAND_N_HOLD = 10 if _CH5_PRIOR_LAND_FULL else 2
CH5_PRIOR_LAND_MS = 90 if _CH5_PRIOR_LAND_FULL else 80
CH5_PRIOR_LAND_DPI = 110 if _CH5_PRIOR_LAND_FULL else 72

# HQ duo landscape clips (ch5_44–47) — higher grid than preview prior_bowl.
CH5_HQ_LAND_GRID = 132 if not _CH3_DRAFT else 52
CH5_HQ_LAND_GRID_COARSE = 56 if not _CH3_DRAFT else 24
CH5_HQ_LAND_GRID_FINE = 96 if not _CH3_DRAFT else 36
CH5_HQ_LAND_QUADRANT_FINE = not _CH3_DRAFT
CH5_HQ_LAND_N_KNOB = 28 if not _CH3_DRAFT else 6
CH5_HQ_LAND_N_ROT = 64 if not _CH3_DRAFT else 6
CH5_HQ_LAND_N_REVEAL = 40 if not _CH3_DRAFT else 8
CH5_HQ_LAND_N_FILL_LINES = CH5_HQ_LAND_GRID if not _CH3_DRAFT else 10
CH5_HQ_LAND_N_FILL_TRACE = 96 if not _CH3_DRAFT else 24
CH5_HQ_LAND_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_HQ_LAND_MS = 90 if not _CH3_DRAFT else 80
CH5_HQ_LAND_DPI = 110 if not _CH3_DRAFT else 72
# 2×2 grid clips (ch5_47–49): denser mesh per cell — same figsize, quarter-frame on screen.
CH5_HQ_GRID_LAND_GRID = 196 if not _CH3_DRAFT else 72
CH5_HQ_GRID_LAND_DPI = 120 if not _CH3_DRAFT else 80
CH5_HQ_N_FADE = 16 if not _CH3_DRAFT else 6
CH5_HQ_N_TEXT_FADE = 14 if not _CH3_DRAFT else 5
CH5_HQ_N_HIST3D = 18 if not _CH3_DRAFT else 6
CH5_HQ_N_LAYOUT = 22 if not _CH3_DRAFT else 6
CH5_HQ_N_KNOB_ZERO = 20 if not _CH3_DRAFT else 6
CH5_HQ_N_SEQ_HOLD = 3 if not _CH3_DRAFT else 2
CH5_HQ_N_ANNOT = 14 if not _CH3_DRAFT else 4
CH5_HQ_GRID_ORBIT_DEG = 360.0
CH5_HQ_GRID_N_ORBIT = 96 if not _CH3_DRAFT else 8
CH5_HQ_LAND_ELEV_OFFSET = -10.0
CH5_HQ_GRID_N_ZOOM = 96 if not _CH3_DRAFT else 12
CH5_HQ_GRID_N_ZOOM_HOLD = 8 if not _CH3_DRAFT else 2
CH5_HQ_GRID_N_ZOOM_ORBIT = 96 if not _CH3_DRAFT else 8
CH5_HQ_GRID_N_SQUISH = 36 if not _CH3_DRAFT else 8
CH5_HQ_GRID_N_CAM_MORPH = 24 if not _CH3_DRAFT else 6
CH5_HQ_GRID_N_FOCUS_OPEN = 12 if not _CH3_DRAFT else 3
CH5_HQ_GRID_N_FOCUS_HOLD = 28 if not _CH3_DRAFT else 6
CH5_HQ_GRID_N_FOCUS_FADE = 20 if not _CH3_DRAFT else 5
CH5_HQ_GRID_FOCUS_DIM_GREY = 0.78
CH5_HQ_GRID_FOCUS_DIM_ALPHA = 0.26
CH5_HQ_CT_GRID = 96 if not _CH3_DRAFT else 36
CH5_HQ_CT_N_SWEEP = 48 if not _CH3_DRAFT else 8
CH5_HQ_CT_N_PIVOT = 72 if not _CH3_DRAFT else 6
CH5_HQ_CT_N_HOLD = 10 if not _CH3_DRAFT else 3
# ch5_59: CT end → empty → MAP → voxels → 48-style zoom/orbit tour.
CH5_CT_VOXEL_TOUR_N_HOLD = 8 if not _CH3_DRAFT else 2
CH5_CT_VOXEL_TOUR_N_ERASE = 28 if not _CH3_DRAFT else 6
CH5_CT_VOXEL_TOUR_N_MAP = 20 if not _CH3_DRAFT else 5
# After first orbit: fade voxels ↔ interval box (shadow/collapse use CH5_VOXEL_N_PROJ_*).
CH5_CT_VOXEL_TOUR_N_BOX = 28 if not _CH3_DRAFT else 6
CH5_CT_VOXEL_TOUR_N_VOXEL_SWAP = 24 if not _CH3_DRAFT else 6
# ch5_61: zoomed credible region — black probe wanders HPD; dark 2D threshold + σ colormap.
CH5_CRED_WANDER_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_CRED_WANDER_N_MORPH = 18 if not _CH3_DRAFT else 4
CH5_CRED_WANDER_N_SEG = 40 if not _CH3_DRAFT else 6
CH5_CRED_WANDER_N_RETURN = 28 if not _CH3_DRAFT else 4
CH5_CRED_WANDER_PROBE_COLOR = "#111111"
CH5_CRED_WANDER_THRESHOLD_COLOR = "#222222"
CH5_CRED_WANDER_THRESHOLD_LW = 3.6
# On-interval probe markers (61–63): position on fixed marginal lines.
CH5_CRED_WANDER_INTERVAL_MARK_SIZE = 130.0
CH5_CRED_WANDER_INTERVAL_MARK_ALPHA = 0.92
# ch5_62: 90° CCW then scale all params by a shared factor (…, 0.999, 1, 1.001, …)
# until the probe exits the HPD; reverse each excursion back through MAP.
# ch5_64: same wander at 0° then 45° CCW (62 remains the 90° cut).
CH5_CRED_THICK_N_ROT = 32 if not _CH3_DRAFT else 6
CH5_CRED_THICK_ROT_DEG = 90.0  # counter-clockwise azim (ch5_62)
CH5_CRED_THICK_N_SEG = 48 if not _CH3_DRAFT else 8
CH5_CRED_THICK_SCALE_STEP = 0.001  # factor increments: 1.001, 1.002, … / 0.999, 0.998, …
# ch5_63: like 62, but w_ST / w_EL / b each ± one-at-a-time inside the HPD.
# ch5_65: D1 only — ST +45° CCW, EL −45° CW, b at og 0°.
CH5_CRED_AXIS_N_SEG = 36 if not _CH3_DRAFT else 6
CH5_CRED_AXIS_N_HOLD = 6 if not _CH3_DRAFT else 2
# ch5_66: D1 MAP Hessian eigen-quivers (soft→stiff) + side Hessian + 360° orbit.
CH5_MAP_QUIVER_LEN = 1.48  # fallback length if eigen frame unavailable
CH5_MAP_QUIVER_LENGTHS = (1.48, 0.724608, 1.48)  # legacy geometric triad lengths
CH5_MAP_QUIVER_LW = 5.2
CH5_MAP_QUIVER_N_ORBIT = 96 if not _CH3_DRAFT else 24
CH5_MAP_QUIVER_N_HOLD = 10 if not _CH3_DRAFT else 3
CH5_MAP_QUIVER_N_SLIDE = 28 if not _CH3_DRAFT else 6
CH5_MAP_QUIVER_COLORS = ("#111111", "#111111", "#111111")  # overridden by eigen plan
CH5_MAP_QUIVER_DOMINANCE = 0.72  # |v_i| share required to color a quiver / H diagonal
CH5_MAP_QUIVER_BLACK = "#111111"
# ch5_66 ellipsoid preview: nested Laplace shells (no voxels / intervals / H text).
CH5_ELLIPSOID_LAYER_SCALES = (0.40, 0.70, 1.00)
CH5_ELLIPSOID_FACE_ALPHAS = (0.28, 0.18, 0.10)
CH5_ELLIPSOID_EDGE_ALPHAS = (0.55, 0.40, 0.28)
CH5_ELLIPSOID_N_LAYER = 4 if not _CH3_DRAFT else 2
CH5_ELLIPSOID_N_ORBIT = 192 if not _CH3_DRAFT else 16  # dense 360° for high-FPS slowdown
CH5_ELLIPSOID_N_HOLD = 6 if not _CH3_DRAFT else 2
CH5_ELLIPSOID_MESH_U = 20 if not _CH3_DRAFT else 12
CH5_ELLIPSOID_MESH_V = 14 if not _CH3_DRAFT else 8
# ch5_64 D1 pre-rotation end → fade HPD+intervals → Laplace shells → Newton quivers.
CH5_ELLIP_FROM64_N_FADE = 28 if not _CH3_DRAFT else 6
CH5_ELLIP_FROM64_N_QUIVER = 14 if not _CH3_DRAFT else 4
# ch5_67: D3 belief landscape → flat data floor → raise bands → colormap → 360° orbit.
CH5_D3_SIG_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_D3_SIG_N_CROSS = 18 if not _CH3_DRAFT else 5  # belief → top-down data floor (same duo)
CH5_D3_SIG_N_TILT = 36 if not _CH3_DRAFT else 8
CH5_D3_SIG_N_RAISE = 72 if not _CH3_DRAFT else 14  # row-wise point/threshold lift
CH5_D3_SIG_N_SURF = 36 if not _CH3_DRAFT else 8  # colormap after all points raised
CH5_D3_SIG_N_ORBIT = 96 if not _CH3_DRAFT else 8  # match landscape / HQ grid 360° resolution
CH5_D3_SIG_ELEV_FLAT = 89.0  # true top-down (= 2D view of the data plane)
CH5_D3_SIG_ELEV_3D = 26.0
CH5_D3_SIG_AZ_TOP = -90.0  # ST right, EL up — matches standard 2D axes
CH5_D3_SIG_AZ0 = -135.0  # post-tilt / sigmoid init (−110 − 25° CW)
CH5_D3_SIG_MESH_N = 160 if not _CH3_DRAFT else 72
# Left 3D top-view: size relative to the 2D data panel (2D itself never rescales).
CH5_D3_SIG_TOP_SCALE = 1.005  # locked 2D↔top-down footprint scale
CH5_D3_SIG_TOP_DY = -0.010  # was +0.008; −¼ EL unit (≈ panel_h/28) downward
CH5_D3_SIG_LEFT_ZOOM = 1.35  # top-view overlay match with 2D
CH5_D3_SIG_LEFT_ZOOM_POST = 1.35  # match pre-tilt size (same as LEFT_ZOOM)
CH5_D3_SIG_MESH_PAD = 0.35  # mesh overhang past axis lims → flush walls, no mid-face snips
# Slight Z headroom so σ≈1 faces aren't clipped by zlim during orbit.
CH5_D3_SIG_ZLIM = (-0.02, 1.04)
# Pull camera back a hair on tilted/orbit views so cube corners stay inside the axes rect.
CH5_D3_SIG_LEFT_ZOOM_ORBIT = 1.18
CH5_D3_SIG_ORBIT_ELEV_DELTA = 5.0  # fixed elev change before orbit (not lerped during spin)
CH5_D3_SIG_OVERLAP_WIDTH = 0.55  # unused while overlap_u stays 0
CH5_D3_SIG_OVERLAP_HEIGHT = 0.40
# During tilt (late ease): left panel grows + shifts right; belief shrinks with it.
CH5_D3_SIG_LEFT_GROW = 1.26  # post-tilt left footprint (+5% vs prior 1.20; belief unchanged)
CH5_D3_SIG_RIGHT_SHRINK = 0.78  # post-tilt belief scale (unchanged — left grows into gutter)
CH5_D3_SIG_TILT_DX = 0.042  # rightward shift of left panel (with grow)
CH5_D3_SIG_GROW_START = 0.30  # tilt progress [0,1] when size morph begins
CH5_D3_SIG_BAND_STEP = 0.5  # parallel-to-threshold band spacing (d = st − el)
CH5_D3_SIG_BAND_OVERLAP = 0.55  # cascade soft edge between successive rows
CH5_D3_SIG_ICON_SPAN_FRAC = 0.060  # 3D check/cross at top-view
CH5_D3_SIG_ICON_SPAN_FRAC_POST = 0.078  # post-tilt check/cross (a bit larger)
# ch5_69: zoomed D2 duo — pulse check/cross icons on the four mislabeled students.
CH5_D2_NOISE_EMPH_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_D2_NOISE_EMPH_N_GROW = 28 if not _CH3_DRAFT else 6
CH5_D2_NOISE_EMPH_SCALE = 2.5  # peak icon size vs baseline
# Pass-on-fail-side (left of MAP threshold), then fail-on-pass-side (right).
CH5_D2_NOISE_PASS = ((1.0, 2.0, 1), (3.0, 4.0, 1))
CH5_D2_NOISE_FAIL = ((2.0, 1.0, 0), (4.0, 3.0, 0))
# ch5_70: D1 best-line celebrate → howithinkabout → D4 rotating cuts → D1 parallel one-at-a-time.
CH5_BEST_LINE_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_BEST_LINE_N_THRESH = 18 if not _CH3_DRAFT else 5
CH5_BEST_LINE_N_CELEBRATE = 36 if not _CH3_DRAFT else 8
CH5_BEST_LINE_N_GREY_LOGO = 24 if not _CH3_DRAFT else 6
CH5_BEST_LINE_N_FADE = 16 if not _CH3_DRAFT else 4
CH5_BEST_LINE_N_D4_CUT = 10 if not _CH3_DRAFT else 3  # hold per rotating D4 cut
CH5_BEST_LINE_N_D4_CUTS = 10 if not _CH3_DRAFT else 4  # number of rotating orientations
CH5_BEST_LINE_N_PARALLEL = 20 if not _CH3_DRAFT else 5  # hold after parallels appear
CH5_BEST_LINE_PARALLEL_BIAS = 2.0  # ±bb span for D1 parallel family
CH5_BEST_LINE_LOGO_FRAC = 0.48  # logo height vs frame height
CH5_BEST_LINE_LOGO_CX = 0.72  # figure x of logo center (center-right)
CH5_BEST_LINE_LW_THICK = 3.4  # emphasized MAP among dashed family
CH5_BEST_LINE_LW_NORMAL = 1.8  # default single threshold
CH5_BEST_LINE_LW_THIN = 1.35  # secondary dashed thresholds
CH5_BEST_LINE_N_LAYOUT = 28 if not _CH3_DRAFT else 6  # wide 2D → duo 3D morph
CH5_BEST_LINE_N_LAND_REVEAL = 72 if not _CH3_DRAFT else 10  # D1 landscape wipe (like 55)
# ch5_68: D1 + credible → morph into E2 triangle fills (pass/fail at ST∈[0,1]).
CH5_D1_FAN_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_D1_FAN_N_MOVE = 28 if not _CH3_DRAFT else 6  # parallel D1→E2 point glide (credible each frame)
CH5_D1_FAN_N_ADD_MORPH = 6 if not _CH3_DRAFT else 2  # quick crossfade per add batch
CH5_D1_FAN_N_ADD_HOLD = 2 if not _CH3_DRAFT else 1
CH5_D1_FAN_ADD_BATCH = 5  # points appended per batch (20 adds → 4 batches)
CH5_D1_FAN_N_SETUP_ROT = 24 if not _CH3_DRAFT else 6  # open: turn to SETUP_SPIN CCW
CH5_D1_FAN_SETUP_SPIN = 60.0  # CCW from og CT view (45° + 15°); held until finale orbit
CH5_D1_FAN_N_ORBIT = 96 if not _CH3_DRAFT else 8  # finale 360° on the 3D HPD panel
CH5_D1_FAN_ORBIT_DEG = 360.0
# E2 end state (seed 43): pass △ ST∈[0,1] EL∈[0,3]; fail △ ST∈[0,1] EL∈[3,7].
CH5_D1_FAN_END_POINTS = (
    (0.652299, 5.670080, 0),
    (0.020030, 6.369732, 0),
    (0.587143, 5.719657, 0),
    (0.751792, 6.268971, 0),
    (0.419978, 5.726344, 0),
    (0.955315, 6.980678, 0),
    (0.278633, 4.918234, 0),
    (0.422000, 4.697426, 0),
    (0.308924, 6.866633, 0),
    (0.838205, 6.699144, 0),
    (0.485489, 6.887068, 0),
    (0.593174, 6.907029, 0),
    (0.033765, 3.593288, 0),
    (0.761199, 6.165282, 0),
    (0.341843, 5.157211, 0),
    (0.551045, 5.295084, 0),
    (0.450970, 6.023642, 0),
    (0.119171, 4.679631, 0),
    (0.504552, 5.620603, 0),
    (0.417836, 5.636715, 0),
    (0.093314, 1.599385, 1),
    (0.074056, 1.612586, 1),
    (0.082872, 1.483337, 1),
    (0.850769, 0.129292, 1),
    (0.082745, 0.881886, 1),
    (0.424499, 1.056905, 1),
    (0.151532, 0.010083, 1),
    (0.843082, 0.214235, 1),
    (0.573631, 1.024985, 1),
    (0.763247, 0.612196, 1),
    (0.321951, 0.394207, 1),
    (0.012704, 1.173077, 1),
    (0.606505, 0.668765, 1),
    (0.878816, 0.300940, 1),
    (0.496535, 0.185312, 1),
    (0.500577, 1.377639, 1),
    (0.880856, 0.240937, 1),
    (0.658906, 0.848334, 1),
    (0.687038, 0.586045, 1),
    (0.401762, 0.717001, 1),
)
# Destinations for each D1 point (same order as CH5_D1_POINTS / pack["D1"]).
CH5_D1_FAN_MOVE_TO = (
    (0.033765, 3.593288),  # (2,3,0)
    (0.652299, 5.670080),  # (4,5,0)
    (0.955315, 6.980678),  # (5,6,0)
    (0.422000, 4.697426),  # (1,3,0)
    (0.551045, 5.295084),  # (2,4,0)
    (0.838205, 6.699144),  # (4,6,0)
    (0.119171, 4.679631),  # (1,4,0)
    (0.761199, 6.165282),  # (3,6,0)
    (0.751792, 6.268971),  # (1,6,0)
    (0.500577, 1.377639),  # (3,2,1)
    (0.573631, 1.024985),  # (5,4,1)
    (0.658906, 0.848334),  # (6,5,1)
    (0.763247, 0.612196),  # (4,2,1)
    (0.424499, 1.056905),  # (6,4,1)
    (0.878816, 0.300940),  # (3,1,1)
    (0.880856, 0.240937),  # (4,1,1)
    (0.843082, 0.214235),  # (5,2,1)
    (0.687038, 0.586045),  # (6,3,1)
    (0.850769, 0.129292),  # (6,2,1)
    (0.606505, 0.668765),  # (6,1,1)
)
# Remaining E2 points (not used as move targets) — batched adds after the glide.
CH5_D1_FAN_ADDS = (
    (0.020030, 6.369732, 0),
    (0.587143, 5.719657, 0),
    (0.419978, 5.726344, 0),
    (0.278633, 4.918234, 0),
    (0.308924, 6.866633, 0),
    (0.485489, 6.887068, 0),
    (0.593174, 6.907029, 0),
    (0.341843, 5.157211, 0),
    (0.450970, 6.023642, 0),
    (0.504552, 5.620603, 0),
    (0.417836, 5.636715, 0),
    (0.093314, 1.599385, 1),
    (0.074056, 1.612586, 1),
    (0.082872, 1.483337, 1),
    (0.082745, 0.881886, 1),
    (0.151532, 0.010083, 1),
    (0.321951, 0.394207, 1),
    (0.012704, 1.173077, 1),
    (0.496535, 0.185312, 1),
    (0.401762, 0.717001, 1),
)
# Legacy name kept for imports; unused by the E2 morph story.
CH5_D1_FAN_OPS = ()
CH5_D1_FAN_N_MORPH = CH5_D1_FAN_N_MOVE  # alias
CH5_D1_FAN_N_EDIT_HOLD = CH5_D1_FAN_N_ADD_HOLD  # alias
# MAP shadow perturbation clip (ch5_52): camera ±90°, knob shadow on 2D/3D.
CH5_MAP_PERTURB_ROT_DEG = 90.0
CH5_MAP_PERTURB_DW_EL = 0.85
CH5_MAP_PERTURB_DW_ST = 0.85
CH5_MAP_PERTURB_N_ROT = 32 if not _CH3_DRAFT else 6
CH5_MAP_PERTURB_N_KNOB = 24 if not _CH3_DRAFT else 6
CH5_MAP_PERTURB_N_HOLD = 8 if not _CH3_DRAFT else 2
# D4 origin-axis tutorial (ch5_53): w_EL/w_ST floor guides → colormap → coupled perturb.
CH5_D4_ORIGIN_N_GUIDE = 28 if not _CH3_DRAFT else 6
CH5_D4_ORIGIN_N_FADE = 18 if not _CH3_DRAFT else 4
CH5_D4_ORIGIN_N_CMAP = 36 if not _CH3_DRAFT else 6
CH5_D4_ORIGIN_N_KNOB = 24 if not _CH3_DRAFT else 6
CH5_D4_ORIGIN_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_D4_ORIGIN_DW_EL = 0.75
CH5_D4_ORIGIN_DW_ST = 0.75
# 2×2 data-panel zoom-out to ±7, then shadow-threshold orbit (ch5_54).
CH5_GRID_2D_ZOOM_TARGET = (-7.0, 7.0)
CH5_GRID_2D_ZOOM_N = 48 if not _CH3_DRAFT else 8
CH5_GRID_2D_ZOOM_N_HOLD = 8 if not _CH3_DRAFT else 2
CH5_GRID_2D_ORBIT_N = 72 if not _CH3_DRAFT else 12
CH5_GRID_2D_ORBIT_R_MAX = 2.85  # keep shadow inside CH5_VIEW_BOUNDS ±3
CH5_GRID_2D_ORBIT_R_MIN = 2.0   # visible orbit when MAP ≈ 0
CH5_GRID_2D_GUIDE_N = 24 if not _CH3_DRAFT else 6
CH5_GRID_2D_CAM_TOP_ELEV = 70.0  # 20° from full top-down (90°)
CH5_GRID_2D_CAM_N = 36 if not _CH3_DRAFT else 8
CH5_GRID_2D_D4_DW_ST = 1.0   # D4: +Δw_ST before / −Δw_ST after the circle
CH5_GRID_2D_D4_DW_EL = -1.0  # D4: −Δw_EL before / +Δw_EL after the circle
CH5_GRID_2D_D4_N_SWING = 28 if not _CH3_DRAFT else 6
# 2×2 labeled belief grid + 90° CCW camera spin (ch5_57).
CH5_GRID_MAP_ROT_DEG = 90.0  # counter-clockwise azim
CH5_GRID_MAP_ROT_N = 48 if not _CH3_DRAFT else 8
CH5_GRID_MAP_ROT_N_HOLD = 10 if not _CH3_DRAFT else 2
# Belief hist → surface build (ch5_58): pillars tile the full (w_ST, w_EL) plane.
# Span 6 / spacing 0.25 → 24×24 = 576 pillars; no tighten-to-stems phase.
CH5_STEM_SURF_N_PILLARS = 576 if not _CH3_DRAFT else 64
CH5_STEM_SURF_N_GROW_PER = 3 if not _CH3_DRAFT else 2  # faster with full-plane count; fps unchanged
CH5_STEM_SURF_N_TIGHTEN = 0  # unused — pillars stay full width until surface wipe
CH5_STEM_SURF_N_HOLD = 10 if not _CH3_DRAFT else 2
CH5_STEM_SURF_N_MORPH = 40 if not _CH3_DRAFT else 6
CH5_STEM_SURF_REVEAL_ORIGIN = "lo_hi"
CH5_STEM_SURF_HIST_WIDTH = 1.0  # full CH5_SURFACE_GRID_SPACING (0.25) cell
CH5_STEM_SURF_SIDE_COLOR = "#ffffff"
CH5_STEM_SURF_SIDE_ALPHA = 1.0
CH5_STEM_SURF_SIDE_EDGE_COLOR = "#2a2a2a"
CH5_STEM_SURF_SIDE_EDGE_WIDTH = 0.55
CH5_STEM_SURF_LINE_WIDTH_FRAC = 0.035
CH5_STEM_SURF_LINE_COLOR = "#888888"
CH5_STEM_SURF_LINE_ALPHA = 0.22
CH5_STEM_SURF_LINEWIDTH = 0.45
CH5_STEM_SURF_POINT_SIZE = 3.5
CH5_STEM_SURF_POINT_ALPHA = 0.90
CH5_STEM_SURF_PILLAR_SEED = 11
CH5_STEM_SURF_MS = 55 if not _CH3_DRAFT else 70
# Legacy aliases used by older call sites / draft toggles.
CH5_STEM_SURF_STRIDE = 1 if not _CH3_DRAFT else 2
CH5_STEM_SURF_N_GROW = CH5_STEM_SURF_N_GROW_PER
CH5_STEM_SURF_GROW_WINDOW = 1.0
CH5_STEM_SURF_HIST_ALPHA = 1.0  # tip RGBA already carries CH5_BELIEF_SURFACE_ALPHA
# D1 likelihood under translucent belief (ch5_55): clear → lik → soft belief.
CH5_LL_OVERLAY_COLOR = "#b71c1c"  # same red as CH4_LIK_SURFACE_COLOR
CH5_LL_OVERLAY_ALPHA = 0.50  # below CH4_LIK_SURFACE_ALPHA (0.80)
CH5_LL_OVERLAY_BELIEF_ALPHA = 0.18  # very translucent belief on top of likelihood
CH5_LL_OVERLAY_HEIGHT_FRAC = 0.55
CH5_LL_OVERLAY_W_LIM = (-2.5, 2.5)  # likelihood support in (w_ST, w_EL)
CH5_LL_OVERLAY_N_REVEAL = 72 if not _CH3_DRAFT else 10  # slow corner-to-corner wipes
CH5_LL_OVERLAY_N_HOLD = 12 if not _CH3_DRAFT else 2
CH5_LL_OVERLAY_REVEAL_ORIGIN = "lo_hi"
CH5_LL_OVERLAY_BEST_LABEL = "best line"
CH5_LL_OVERLAY_BEST_LABEL_FIG = (0.93, 0.72)  # below "most plausible line"
CH5_LL_OVERLAY_N_ORBIT = 96 if not _CH3_DRAFT else 8  # final 360° spin
CH5_LL_OVERLAY_ORBIT_DEG = 360.0
CH5_KNOB_ZERO = (0.0, 0.0, 0.0)
# Equal-size dials, no legend emphasis — knobs visually "unset".
CH5_KNOBS_UNSET_FRAME_KW = {"emphasize_knob": None, "knob_scales": [1.0, 1.0, 1.0]}
CH5_SURFACE_GRID_SPACING = 0.25
CH5_SURFACE_GRID_COLOR = "#4d4d4d"
CH5_SURFACE_GRID_LINEWIDTH = 0.55

CH5_PRIOR_TEX = r"$p(w)=\mathcal{N}(0,\sigma^2 I)$"
CH5_LIK_TEX = r"$p(D \mid w)=\prod_i \hat p_i^{\,y_i}(1-\hat p_i)^{1-y_i}$"
CH5_BAYES_TEX = r"$p(w \mid D) \propto p(D \mid w)\, p(w)$"

# Clips through per-dataset packs use text; intro + 4×4 montage onward do not.
CH5_TEXT_LAST_ID = "ch5_25"


def ch5_hq_land_elev(base_elev: float | None = None) -> float:
    """HQ landscape / CT elevation: canonical CT elev minus 10°."""
    be = 24.0 if base_elev is None else float(base_elev)
    return be + float(CH5_HQ_LAND_ELEV_OFFSET)


def ch5_surface_grid_stride(w1, w2, *, spacing: float | None = None) -> tuple[int, int]:
    """``plot_surface`` rstride/cstride for ~fixed parameter-space grid spacing."""
    sp = float(CH5_SURFACE_GRID_SPACING if spacing is None else spacing)
    W1 = np.asarray(w1, dtype=np.float64)
    W2 = np.asarray(w2, dtype=np.float64)
    if W1.size < 2 or W2.size < 2:
        return 1, 1
    ni, nj = int(W1.shape[0]), int(W1.shape[1])
    w1_lo, w1_hi = float(np.nanmin(W1)), float(np.nanmax(W1))
    w2_lo, w2_hi = float(np.nanmin(W2)), float(np.nanmax(W2))
    dw1 = max((w1_hi - w1_lo) / max(ni - 1, 1), 1e-12)
    dw2 = max((w2_hi - w2_lo) / max(nj - 1, 1), 1e-12)
    return max(1, int(round(sp / dw1))), max(1, int(round(sp / dw2)))


def ch5_surface_grid_plot_kw(w1, w2, *, spacing: float | None = None) -> dict:
    """Dark-grey belief-surface grid kwargs for matplotlib ``plot_surface``."""
    rs, cs = ch5_surface_grid_stride(w1, w2, spacing=spacing)
    return dict(
        rstride=rs,
        cstride=cs,
        linewidth=float(CH5_SURFACE_GRID_LINEWIDTH),
        edgecolor=str(CH5_SURFACE_GRID_COLOR),
    )


def ch5_plot_belief_surface_with_grid(
    ax3d,
    w1,
    w2,
    z,
    *,
    facecolors=None,
    color=None,
    alpha=None,
    spacing: float | None = None,
    rstride: int | None = None,
    cstride: int | None = None,
    edgecolor: str | None = None,
    linewidth: float | None = None,
    zorder: float = 1.0,
    antialiased: bool = False,
) -> None:
    """
    Belief surface + dark-grey parameter grid.

    Matplotlib sets edgecolors=facecolors when facecolors is an array, so grid
    lines are drawn via an overlaid ``plot_wireframe``.
    """
    W1 = np.asarray(w1, dtype=np.float64)
    W2 = np.asarray(w2, dtype=np.float64)
    Z = np.asarray(z, dtype=np.float64)
    if rstride is None or cstride is None:
        grid = ch5_surface_grid_plot_kw(W1, W2, spacing=spacing)
        rs = int(grid["rstride"])
        cs = int(grid["cstride"])
        ec = str(CH5_SURFACE_GRID_COLOR if edgecolor is None else edgecolor)
        lw = float(CH5_SURFACE_GRID_LINEWIDTH if linewidth is None else linewidth)
    else:
        rs, cs = int(rstride), int(cstride)
        ec = str(CH5_SURFACE_GRID_COLOR if edgecolor is None else edgecolor)
        lw = float(CH5_SURFACE_GRID_LINEWIDTH if linewidth is None else linewidth)
    surf_kw: dict = dict(
        shade=False,
        linewidth=0,
        antialiased=bool(antialiased),
        rstride=1,
        cstride=1,
        zorder=float(zorder),
    )
    if facecolors is not None:
        surf_kw["facecolors"] = np.asarray(facecolors, dtype=float)
    if color is not None:
        surf_kw["color"] = color
    if alpha is not None:
        surf_kw["alpha"] = float(alpha)
    ax3d.plot_surface(W1, W2, Z, **surf_kw)
    Z_wf = Z
    if facecolors is not None:
        fc = np.asarray(facecolors, dtype=float)
        if fc.ndim == 3 and fc.shape[:2] == Z.shape and fc.shape[-1] >= 4:
            # Hide wireframe with the same diagonal / alpha wipe as the surface.
            Z_wf = np.where(fc[..., 3] > 1e-4, Z, np.nan)
    ax3d.plot_wireframe(
        W1, W2, Z_wf,
        rstride=rs,
        cstride=cs,
        color=ec,
        linewidth=lw,
        alpha=1.0,
        zorder=float(zorder) + 0.5,
    )


def ch5_axis_inner_limits(lo, hi, *, shrink=None):
    """Axis limits inset so span shrinks by ``shrink`` (default 1): [-3,3] → [-2.5, 2.5]."""
    lo, hi = float(lo), float(hi)
    margin = 0.5 * float(CH5_UNIFORM_AXIS_SHRINK if shrink is None else shrink)
    return lo + margin, hi - margin


def ch5_uniform_support_bounds(bounds=None):
    """Inner uniform-support box: axis range minus 1 on each axis."""
    if bounds is None:
        bounds = CH5_VIEW_BOUNDS
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    w1lo, w1hi = ch5_axis_inner_limits(dlo1, dhi1)
    w2lo, w2hi = ch5_axis_inner_limits(dlo2, dhi2)
    blo, bhi = ch5_axis_inner_limits(dlob, dhib)
    return w1lo, w1hi, w2lo, w2hi, blo, bhi


def ch5_log_prior(w1, w2, b, *, sigma=None):
    sig = float(CH5_PRIOR_SIGMA if sigma is None else sigma)
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return -0.5 * (w1 * w1 + w2 * w2 + b * b) / (sig * sig)


def ch5_log_prior_uniform(w1, w2, b, *, bounds=None):
    """Flat log-prior on inner box (axis span − 1); −∞ outside support."""
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if bounds is None:
        w1lo, w1hi, w2lo, w2hi, blo, bhi = ch5_uniform_support_bounds()
    else:
        w1lo, w1hi, w2lo, w2hi, blo, bhi = bounds
    inside = (
        (w1 >= float(w1lo)) & (w1 <= float(w1hi))
        & (w2 >= float(w2lo)) & (w2 <= float(w2hi))
        & (b >= float(blo)) & (b <= float(bhi))
    )
    lp = np.full(w1.shape, -np.inf, dtype=np.float64)
    lp[inside] = 0.0
    return lp


def ch5_prior_w12_log_flat(w1, w2, b, *, kind="gaussian", sigma=None, bounds=None):
    """Log-prior on flat (w1, w2) grids with scalar ``b`` (fixed intercept)."""
    kind = str(kind).lower()
    if kind == "uniform":
        return ch5_log_prior_uniform(w1, w2, b)
    sig = float(CH5_PRIOR_LANDSCAPE_SIGMA if sigma is None else sigma)
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    return -0.5 * (w1 * w1 + w2 * w2) / (sig * sig)


def ch5_prior_w12_z_display(log_vals, *, log_z_max=None):
    """Map log-prior to display height using one global ``log_z_max`` (not per-point)."""
    lp = np.asarray(log_vals, dtype=np.float64)
    lz = float(np.nanmax(lp) if log_z_max is None else log_z_max)
    return np.exp(lp - lz) * float(CH5_DENSITY_Z_HI)


def ch5_prior_w12_density_flat(w1, w2, b, *, kind="gaussian", sigma=None, bounds=None, log_z_max=None):
    lp = ch5_prior_w12_log_flat(w1, w2, b, kind=kind, sigma=sigma, bounds=bounds)
    return ch5_prior_w12_z_display(lp, log_z_max=log_z_max)


def ch5_uniform_support_volume(*, bounds=None) -> float:
    """Volume of the inner uniform-prior box in (w_ST, w_EL, b)."""
    w1lo, w1hi, w2lo, w2hi, blo, bhi = ch5_uniform_support_bounds(bounds)
    return float((w1hi - w1lo) * (w2hi - w2lo) * (bhi - blo))


def ch5_w12_cell_area(w1_lo, w1_hi, w2_lo, w2_hi, shape) -> float:
    """Grid cell area for a regular (w_ST, w_EL) mesh."""
    n1, n2 = int(shape[0]), int(shape[1])
    span1 = max(float(w1_hi) - float(w1_lo), 0.0)
    span2 = max(float(w2_hi) - float(w2_lo), 0.0)
    if n1 <= 1 and n2 <= 1:
        return span1 * span2
    dw1 = span1 / max(n1 - 1, 1)
    dw2 = span2 / max(n2 - 1, 1)
    return dw1 * dw2


def ch5_grid_cell_volume(w1_edges, w2_edges, b_edges) -> float:
    """Cell volume for a regular 3-D parameter grid."""
    return float(np.diff(w1_edges)[0] * np.diff(w2_edges)[0] * np.diff(b_edges)[0])


def ch5_prior_w12_pdf(w1, w2, b, *, kind="gaussian", sigma=None, bounds=None):
    """Normalized prior pdf on the (w_ST, w_EL) slice with fixed ``b``."""
    kind = str(kind).lower()
    lp = ch5_prior_w12_log_flat(w1, w2, b, kind=kind, sigma=sigma, bounds=bounds)
    lp = np.asarray(lp, dtype=np.float64)
    pdf = np.zeros_like(lp, dtype=np.float64)
    inside = np.isfinite(lp)
    if not np.any(inside):
        return pdf
    if kind == "uniform":
        pdf[inside] = 1.0 / ch5_uniform_support_volume(bounds=bounds)
        return pdf
    sig = float(CH5_PRIOR_LANDSCAPE_SIGMA if sigma is None else sigma)
    norm = 1.0 / (2.0 * np.pi * sig * sig)
    pdf[inside] = np.exp(lp[inside]) * norm
    return pdf


def ch5_prior_3d_pdf(w1, w2, b, *, kind="gaussian", sigma=None, bounds=None):
    """Normalized prior pdf on a 3-D (w_ST, w_EL, b) grid."""
    kind = str(kind).lower()
    lp = ch5_log_prior_grid(w1, w2, b, kind=kind, sigma=sigma)
    lp = np.asarray(lp, dtype=np.float64)
    pdf = np.zeros_like(lp, dtype=np.float64)
    inside = np.isfinite(lp)
    if not np.any(inside):
        return pdf
    if kind == "uniform":
        pdf[inside] = 1.0 / ch5_uniform_support_volume(bounds=bounds)
        return pdf
    sig = float(CH5_PRIOR_SIGMA if sigma is None else sigma)
    norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sig ** 3)
    pdf[inside] = np.exp(lp[inside]) * norm
    return pdf


def ch5_posterior_pdf_from_log(log_post, cell_volume) -> np.ndarray:
    """Normalize ``exp(log_post)`` on a grid so the Riemann sum integrates to 1."""
    lp = np.asarray(log_post, dtype=np.float64)
    finite = np.isfinite(lp)
    pdf = np.zeros_like(lp, dtype=np.float64)
    if not np.any(finite):
        return pdf
    lp_max = float(np.max(lp[finite]))
    unnorm = np.zeros_like(lp, dtype=np.float64)
    unnorm[finite] = np.exp(lp[finite] - lp_max)
    Z = float(np.sum(unnorm) * float(cell_volume))
    if Z > 0.0:
        pdf[finite] = unnorm[finite] / Z
    return pdf


def ch5_posterior_w12_pdf(log_post, *, w1_lo, w1_hi, w2_lo, w2_hi) -> np.ndarray:
    """Posterior pdf on a (w_ST, w_EL) mesh with ``b`` fixed."""
    return ch5_posterior_pdf_from_log(
        log_post,
        ch5_w12_cell_area(w1_lo, w1_hi, w2_lo, w2_hi, np.asarray(log_post).shape),
    )


def ch5_belief_w12_pdf(log_post, *, w1_lo, w1_hi, w2_lo, w2_hi) -> np.ndarray:
    """2D marginal belief pdf on (w_ST, w_EL) — shared by prior and posterior surfaces."""
    return ch5_posterior_w12_pdf(
        log_post, w1_lo=float(w1_lo), w1_hi=float(w1_hi), w2_lo=float(w2_lo), w2_hi=float(w2_hi),
    )


def ch5_belief_w12_pdf_trace(log_post_1d, *, w1_lo, w1_hi, w2_lo, w2_hi, grid_n) -> np.ndarray:
    """Belief pdf along a 1D trace using the same cell area as a square w_ST–w_EL grid."""
    gn = int(grid_n)
    cell = ch5_w12_cell_area(w1_lo, w1_hi, w2_lo, w2_hi, (gn, gn))
    return ch5_posterior_pdf_from_log(np.asarray(log_post_1d, dtype=float), cell)


def ch5_prior_w12_belief_peak(*, kind="gaussian", bounds=None, gn=None) -> float:
    """Peak of the 2D marginal prior pdf on the canonical (w_ST, w_EL) grid."""
    W1, W2 = ch5_w12_mesh(bounds=bounds, gn=gn)
    B = np.full_like(W1, CH5_W12_B_FIXED)
    if bounds is None:
        dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    else:
        dlo1, dhi1, dlo2, dhi2, _, _ = bounds
    lp = ch5_prior_w12_log_flat(W1, W2, B, kind=kind)
    pdf = ch5_belief_w12_pdf(lp, w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2)
    return float(np.nanmax(pdf))


def ch5_prior_w12_pdf_peak(*, kind="gaussian", sigma=None, bounds=None) -> float:
    """Maximum prior pdf height on the (w_ST, w_EL) slice (joint 3D value at fixed ``b``)."""
    kind = str(kind).lower()
    if kind == "uniform":
        return 1.0 / ch5_uniform_support_volume(bounds=bounds)
    sig = float(CH5_PRIOR_LANDSCAPE_SIGMA if sigma is None else sigma)
    return 1.0 / (2.0 * np.pi * sig * sig)


def ch5_prior_w12_z_lim(prior_kind, *, pad_frac=0.06, scope="global") -> tuple[float, float]:
    """Fixed z-axis limits for belief landscapes.

    ``scope``:
      - ``global``: max peak across D1–D4 (duo grids, montage)
      - ``d1``: D1 posterior peak (ch5_45 sequential landscape)
      - ``prior``: 2D marginal prior peak only (ch5_44 / prior-bowl construction)
    """
    pk = str(prior_kind).lower()
    sc = str(scope).lower()
    if sc == "prior":
        z_hi = float(ch5_prior_w12_belief_peak(kind=pk))
    elif sc == "d1":
        if pk not in CH5_BELIEF_W12_Z_HI_D1:
            raise ValueError(f"unknown prior_kind {prior_kind!r}")
        z_hi = float(CH5_BELIEF_W12_Z_HI_D1[pk])
    else:
        if pk not in CH5_BELIEF_W12_Z_HI:
            raise ValueError(f"unknown prior_kind {prior_kind!r}")
        z_hi = float(CH5_BELIEF_W12_Z_HI[pk])
    pad = float(pad_frac) * max(z_hi, 1e-15)
    return 0.0, z_hi + pad


def ch5_belief_landscape_z_lim(
    prior_kind,
    *,
    phase_u: float = 1.0,
    pad_frac=0.06,
) -> tuple[float, float]:
    """Growing belief landscape z-axis: ``phase_u=0`` prior shelf → ``1`` global peak."""
    pk = str(prior_kind).lower()
    u = float(np.clip(float(phase_u), 0.0, 1.0))
    _, z_prior = ch5_prior_w12_z_lim(pk, scope="prior", pad_frac=pad_frac)
    _, z_global = ch5_prior_w12_z_lim(pk, scope="global", pad_frac=pad_frac)
    z_hi = float(z_prior) + u * (float(z_global) - float(z_prior))
    return 0.0, z_hi


def ch5_clip_belief_height(z, *, prior_kind="gaussian", z_lim=None) -> np.ndarray:
    """Clip belief/pdf heights to the calibrated z range (safety guard; avoids flat plateaus)."""
    if z_lim is None:
        z_lim = ch5_prior_w12_z_lim(prior_kind)
    z = np.asarray(z, dtype=float)
    return np.clip(z, float(z_lim[0]), float(z_lim[1]))


def ch5_log_prior_grid(w1, w2, b, *, kind="gaussian", sigma=None):
    kind = str(kind).lower()
    if kind == "uniform":
        return ch5_log_prior_uniform(w1, w2, b)
    sig = float(CH5_PRIOR_SIGMA if sigma is None else sigma)
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return -0.5 * (w1 * w1 + w2 * w2 + b * b) / (sig * sig)


def ch5_log_likelihood_grid(study, exam, y, w1, w2, b, *, nll_fn):
    shape = np.asarray(w1, dtype=np.float64).shape
    nll = nll_fn(
        study, exam, y,
        np.asarray(w1, dtype=np.float64).ravel(),
        np.asarray(w2, dtype=np.float64).ravel(),
        np.asarray(b, dtype=np.float64).ravel(),
    )
    return (-np.asarray(nll, dtype=np.float64)).reshape(shape)


def ch5_log_posterior_grid(study, exam, y, w1, w2, b, *, sigma=None, nll_fn=None, prior_kind="gaussian"):
    lp = ch5_log_prior_grid(w1, w2, b, kind=prior_kind, sigma=sigma)
    if study is None or len(study) == 0:
        return np.asarray(lp, dtype=np.float64)
    ll = ch5_log_likelihood_grid(study, exam, y, w1, w2, b, nll_fn=nll_fn)
    return lp + ll


def ch5_posterior_w12_surface(study, exam, y, *, prior_kind="gaussian", nll_fn=None, b_fixed=None):
    """Belief surface over (w_ST, w_EL) with b fixed — for sequential / prior frames."""
    b_fixed = float(CH5_W12_B_FIXED if b_fixed is None else b_fixed)
    W1, W2 = ch5_w12_mesh()
    B = np.full_like(W1, b_fixed)
    study = np.asarray(study, dtype=float)
    exam = np.asarray(exam, dtype=float)
    y = np.asarray(y, dtype=int)
    pk = str(prior_kind).lower()
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    lp = ch5_prior_w12_log_flat(W1, W2, B, kind=pk)
    if study.size == 0:
        log_post = lp
        density = ch5_belief_w12_pdf(
            log_post.reshape(W1.shape), w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
        )
    else:
        ll = ch5_log_likelihood_grid(study, exam, y, W1, W2, B, nll_fn=nll_fn)
        log_post = lp + ll
        density = ch5_belief_w12_pdf(
            log_post.reshape(W1.shape), w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
        )
    density = ch5_clip_belief_height(density, prior_kind=pk)
    k = int(np.nanargmax(log_post))
    z_lim = ch5_prior_w12_z_lim(pk)
    return {
        "W1": W1,
        "W2": W2,
        "density": density,
        "log_post": log_post,
        "log_prior": lp,
        "ws": float(W1.ravel()[k]),
        "we": float(W2.ravel()[k]),
        "bb": b_fixed,
        "z_lim": z_lim,
        "prior_kind": pk,
    }


def ch5_density_from_log(log_vals):
    """Per-grid posterior shape normalized to local peak (legacy / diagnostics)."""
    lp = np.asarray(log_vals, dtype=float)
    d = np.exp(lp - np.nanmax(lp))
    return d / max(float(np.nanmax(d)), 1e-12) * float(CH5_DENSITY_Z_HI)


def ch5_global_log_post_ref(prior_kind: str = "gaussian") -> float:
    """Shared log-posterior reference for cross-clip 3D color scales."""
    pk = str(prior_kind).lower()
    if pk not in CH5_GLOBAL_LOG_POST_REF:
        raise ValueError(f"unknown prior_kind {prior_kind!r}")
    return float(CH5_GLOBAL_LOG_POST_REF[pk])


def ch5_global_density_limits(prior_kind: str = "gaussian") -> tuple[float, float]:
    """Colormap domain: true prior pdf range (fixed z-axis for landscapes)."""
    return ch5_prior_w12_z_lim(prior_kind)


def ch5_posterior_color_density(log_post, *, prior_kind: str = "gaussian"):
    """Posterior color height on a shared scale (D1 peak → ``CH5_DENSITY_Z_HI``)."""
    ref = ch5_global_log_post_ref(prior_kind)
    lp = np.asarray(log_post, dtype=float)
    d = np.zeros_like(lp, dtype=float)
    finite = np.isfinite(lp)
    if np.any(finite):
        d[finite] = np.exp(lp[finite] - ref) * float(CH5_DENSITY_Z_HI)
    return np.clip(d, 0.0, float(CH5_DENSITY_Z_HI))


def ch5_calibrate_global_density_scales(
    *,
    nll_fn,
    keys=None,
    gn_voxel=None,
    gn_w12=None,
    bounds=None,
):
    """Set global log-posterior refs and belief z-axis peaks from canonical datasets."""
    global CH5_GLOBAL_LOG_POST_REF, CH5_BELIEF_W12_Z_HI, CH5_BELIEF_W12_Z_HI_D1
    keys = tuple(CH5_DATASET_KEYS if keys is None else keys)
    gn_voxel = int(CH5_VOXEL_GRID if gn_voxel is None else gn_voxel)
    gn_w12 = int(CH5_GRID if gn_w12 is None else gn_w12)
    if bounds is None:
        bounds = CH5_VIEW_BOUNDS
    dlo1, dhi1, dlo2, dhi2, _, _ = bounds
    w1v, w2v, bv, _, _, _ = ch5_posterior_3d_axes(gn=gn_voxel, bounds=bounds)
    W1v, W2v, Bv = np.meshgrid(w1v, w2v, bv, indexing="ij")
    W1w, W2w = ch5_w12_mesh(gn=gn_w12, bounds=bounds)
    Bw = np.full_like(W1w, CH5_W12_B_FIXED)
    gn_hq = int(CH5_HQ_LAND_GRID)
    W1h, W2h = ch5_w12_mesh(gn=gn_hq, bounds=bounds)
    Bh = np.full_like(W1h, CH5_W12_B_FIXED)
    log_refs: dict[str, float] = {}
    belief_peaks: dict[str, float] = {}
    d1_peaks: dict[str, float] = {}
    study_d1, exam_d1, y_d1 = ch5_unpack_dataset("D1")
    for pk in ("gaussian", "uniform"):
        log_peak = -np.inf
        pdf_peak = 0.0
        d1_peak = 0.0
        lp0 = ch5_prior_w12_log_flat(W1w, W2w, Bw, kind=pk)
        pdf0 = ch5_belief_w12_pdf(lp0, w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2)
        pdf_peak = max(pdf_peak, float(np.nanmax(pdf0)))
        lp_d1 = ch5_log_posterior_grid(
            study_d1, exam_d1, y_d1, W1h, W2h, Bh,
            nll_fn=nll_fn, prior_kind=pk,
        )
        pdf_d1 = ch5_belief_w12_pdf(
            lp_d1, w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
        )
        d1_peak = max(d1_peak, float(np.nanmax(pdf_d1)))
        for key in keys:
            study, exam, y = ch5_unpack_dataset(str(key))
            for W1, W2, B in ((W1v, W2v, Bv), (W1w, W2w, Bw)):
                lp = ch5_log_posterior_grid(
                    study, exam, y, W1, W2, B,
                    nll_fn=nll_fn, prior_kind=pk,
                )
                finite = np.isfinite(lp)
                if np.any(finite):
                    log_peak = max(log_peak, float(np.max(lp[finite])))
            lp_w = ch5_log_posterior_grid(
                study, exam, y, W1w, W2w, Bw,
                nll_fn=nll_fn, prior_kind=pk,
            )
            pdf_w = ch5_belief_w12_pdf(
                lp_w, w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
            )
            pdf_peak = max(pdf_peak, float(np.nanmax(pdf_w)))
        if not np.isfinite(log_peak):
            log_peak = float(CH5_GLOBAL_LOG_POST_REF[pk])
        log_refs[pk] = log_peak
        belief_peaks[pk] = max(pdf_peak, float(np.nanmax(pdf0)))
        d1_peaks[pk] = max(d1_peak, float(np.nanmax(pdf0)))
    CH5_GLOBAL_LOG_POST_REF = log_refs
    CH5_BELIEF_W12_Z_HI = belief_peaks
    CH5_BELIEF_W12_Z_HI_D1 = d1_peaks
    return dict(CH5_GLOBAL_LOG_POST_REF)


def ch5_posterior_grid_weights(log_post):
    """Discrete posterior probability weights on a grid (sum to 1 over finite cells)."""
    lp = np.asarray(log_post, dtype=float)
    finite = np.isfinite(lp)
    w = np.zeros_like(lp, dtype=float)
    if not np.any(finite):
        return w
    lp_max = float(np.nanmax(lp[finite]))
    w[finite] = np.exp(lp[finite] - lp_max)
    s = float(np.sum(w))
    if s > 0.0:
        w /= s
    return w


def ch5_posterior_display_density(log_post, log_prior, w1, w2, b=None, *, info_scale=None):
    """Belief-surface height: prior bowl × softened likelihood evidence.

    Max-normalizing ``exp(log_post)`` alone makes every dataset look needle-sharp
    because log-likelihood spans hundreds of nats off the prior.  For uninformative
    data (D4) the evidence factor is ~flat so the landscape stays wide like the prior.
    """
    log_post = np.asarray(log_post, dtype=float)
    log_prior = np.asarray(log_prior, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    inside = np.isfinite(log_prior)
    ll = np.full_like(log_post, -np.inf, dtype=float)
    ll[inside] = log_post[inside] - log_prior[inside]
    if b is None:
        dist = w1 * w1 + w2 * w2
    else:
        b = np.asarray(b, dtype=float)
        dist = w1 * w1 + w2 * w2 + b * b
    k0 = int(np.nanargmin(np.where(inside, dist, np.inf)))
    ll_ref = float(ll.ravel()[k0])
    ll_finite = ll[inside]
    info = max(0.0, float(np.nanmax(ll_finite)) - ll_ref) if ll_finite.size else 0.0
    scale = float(CH5_INFO_SCALE if info_scale is None else info_scale)
    tau = info / (info + scale)
    lp_max = float(np.nanmax(log_prior[inside])) if np.any(inside) else 0.0
    prior_z = np.zeros_like(log_post, dtype=float)
    prior_z[inside] = np.exp(log_prior[inside] - lp_max)
    z = prior_z * np.exp((ll - ll_ref) * tau)
    zmax = max(float(np.nanmax(z)), 1e-300)
    return z / zmax * float(CH5_DENSITY_Z_HI)


def ch5_w12_mesh(bounds=None, *, gn=None):
    gn = int(CH5_GRID if gn is None else gn)
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS if bounds is None else bounds
    w1 = np.linspace(dlo1, dhi1, gn, dtype=np.float64)
    w2 = np.linspace(dlo2, dhi2, gn, dtype=np.float64)
    return np.meshgrid(w1, w2, indexing="ij")


def ch5_posterior_pack(study, exam, y, *, b_fixed=None, nll_fn=None, prior_kind="gaussian"):
    b_fixed = float(CH5_W12_B_FIXED if b_fixed is None else b_fixed)
    W1, W2 = ch5_w12_mesh()
    B = np.full_like(W1, b_fixed)
    pk = str(prior_kind).lower()
    lp = ch5_prior_w12_log_flat(W1, W2, B, kind=pk)
    dlo1, dhi1, dlo2, dhi2, _, _ = CH5_VIEW_BOUNDS
    if study.size == 0:
        log_post = lp
        density = ch5_belief_w12_pdf(
            log_post.reshape(W1.shape), w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
        )
    else:
        ll = ch5_log_likelihood_grid(study, exam, y, W1, W2, B, nll_fn=nll_fn)
        log_post = lp + ll
        density = ch5_belief_w12_pdf(
            log_post.reshape(W1.shape), w1_lo=dlo1, w1_hi=dhi1, w2_lo=dlo2, w2_hi=dhi2,
        )
    density = ch5_clip_belief_height(density, prior_kind=pk)
    k = int(np.nanargmax(log_post))
    ws, we = float(W1.ravel()[k]), float(W2.ravel()[k])
    return {
        "W1": W1, "W2": W2, "density": density,
        "log_post": log_post, "log_prior": lp,
        "ws": ws, "we": we, "bb": b_fixed,
        "z_lim": ch5_prior_w12_z_lim(pk),
        "prior_kind": pk,
    }


def ch5_reveal_order(y):
    yy = np.asarray(y, dtype=int)
    fail = np.flatnonzero(yy == 0)
    pass_ = np.flatnonzero(yy == 1)
    order = []
    for i in range(max(len(fail), len(pass_))):
        if i < len(fail):
            order.append(int(fail[i]))
        if i < len(pass_):
            order.append(int(pass_[i]))
    return order


def ch5_all_dataset_packs(*, nll_fn=None, nll_grad_fn=None):
    return {k: ch5_dataset_pack(k, nll_fn=nll_fn, nll_grad_fn=nll_grad_fn) for k in CH5_DATASET_KEYS}


def ch5_dataset_pack(key: str, *, nll_fn=None, nll_grad_fn=None, prior_kind="gaussian"):
    study, exam, y = ch5_unpack_dataset(key)
    meta = CH5_DATASET_META[key]
    post = ch5_posterior_pack(study, exam, y, nll_fn=nll_fn, prior_kind=prior_kind)
    map3 = ch5_posterior_3d_pack(
        study, exam, y, nll_fn=nll_fn, prior_kind=prior_kind, gn=CH5_GRID,
    )
    cont = ch5_posterior_map_continuous(
        study, exam, y, nll_fn=nll_fn, nll_grad_fn=nll_grad_fn, prior_kind=prior_kind,
        x0=(map3["ws"], map3["we"], map3["bb"]),
    )
    post["ws"] = cont["ws"]
    post["we"] = cont["we"]
    post["bb"] = cont["bb"]
    ws, we, bb = post["ws"], post["we"], post["bb"]
    return {
        "key": key,
        "study": study,
        "exam": exam,
        "y": y,
        "meta": meta,
        "display_w": (float(ws), float(we), float(bb)),
        "order": ch5_reveal_order(y),
        **post,
    }


def ch5_posterior_3d_axes(*, gn=None, bounds=None):
    """Cell centers and bin edges on (w_ST, w_EL, b)."""
    gn = int(CH5_VOXEL_GRID if gn is None else gn)
    if bounds is None:
        bounds = CH5_VIEW_BOUNDS
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    w1 = np.linspace(dlo1, dhi1, gn, dtype=np.float64)
    w2 = np.linspace(dlo2, dhi2, gn, dtype=np.float64)
    b = np.linspace(dlob, dhib, gn, dtype=np.float64)
    w1_edges = np.linspace(dlo1, dhi1, gn + 1, dtype=np.float64)
    w2_edges = np.linspace(dlo2, dhi2, gn + 1, dtype=np.float64)
    b_edges = np.linspace(dlob, dhib, gn + 1, dtype=np.float64)
    return w1, w2, b, w1_edges, w2_edges, b_edges


def ch5_posterior_3d_pack(
    study,
    exam,
    y,
    *,
    prior_kind="gaussian",
    gn=None,
    bounds=None,
    mass=None,
    nll_fn=None,
):
    """Full 3D posterior on a grid: MAP, density colors, HPD mask, marginal intervals."""
    from ch5_layout import ch5_hpd_mask_3d, ch5_hpd_voxel_diag_projection, ch5_marginal_credible_intervals

    cm = float(CH5_CREDIBLE_MASS if mass is None else mass)
    if bounds is None:
        bounds = CH5_VIEW_BOUNDS
    w1, w2, b, w1_edges, w2_edges, b_edges = ch5_posterior_3d_axes(gn=gn, bounds=bounds)
    W1, W2, B = np.meshgrid(w1, w2, b, indexing="ij")
    log_post = ch5_log_posterior_grid(
        study, exam, y, W1, W2, B,
        nll_fn=nll_fn, prior_kind=str(prior_kind),
    )
    weights = ch5_posterior_grid_weights(log_post)
    cell_vol = ch5_grid_cell_volume(w1_edges, w2_edges, b_edges)
    density = ch5_posterior_pdf_from_log(log_post, cell_vol)
    k = int(np.nanargmax(log_post))
    mask = ch5_hpd_mask_3d(weights, mass=cm)
    proj, max_proj = ch5_hpd_voxel_diag_projection(w1, w2, b, bounds=bounds)
    intervals = ch5_marginal_credible_intervals(w1, w2, b, mask)
    return {
        "w1": w1,
        "w2": w2,
        "b": b,
        "w1_edges": w1_edges,
        "w2_edges": w2_edges,
        "b_edges": b_edges,
        "W1": W1,
        "W2": W2,
        "B": B,
        "log_post": log_post,
        "weights": weights,
        "density": density,
        "mask": mask,
        "proj": proj,
        "max_proj": max_proj,
        "intervals": intervals,
        "bounds": bounds,
        "mass": cm,
        "covered": float(np.sum(weights[mask])),
        "ws": float(W1.ravel()[k]),
        "we": float(W2.ravel()[k]),
        "bb": float(B.ravel()[k]),
    }


def ch5_log_posterior_at(study, exam, y, w_st, w_el, b, *, prior_kind="gaussian", nll_fn=None):
    """Scalar log-posterior at one parameter triple."""
    w1 = np.asarray([float(w_st)], dtype=np.float64)
    w2 = np.asarray([float(w_el)], dtype=np.float64)
    bb = np.asarray([float(b)], dtype=np.float64)
    lp = ch5_log_posterior_grid(
        study, exam, y, w1, w2, bb,
        nll_fn=nll_fn, prior_kind=str(prior_kind),
    )
    return float(lp[0])


def ch5_posterior_map_continuous(
    study,
    exam,
    y,
    *,
    prior_kind="gaussian",
    bounds=None,
    nll_fn=None,
    nll_grad_fn=None,
    x0=None,
):
    """Continuous MAP via L-BFGS-B on log-posterior (same model as the 3D grid)."""
    from scipy.optimize import minimize

    if bounds is None:
        bounds = CH5_VIEW_BOUNDS
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = bounds
    bnds = [(dlo1, dhi1), (dlo2, dhi2), (dlob, dhib)]
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pk = str(prior_kind).lower()
    sig2 = float(CH5_PRIOR_SIGMA) ** 2

    def neg_log_post(x):
        val = ch5_log_posterior_at(
            study, exam, y, x[0], x[1], x[2],
            prior_kind=pk, nll_fn=nll_fn,
        )
        if not np.isfinite(val):
            return 1e12
        return -val

    def jac(x):
        if nll_grad_fn is None:
            return None
        g1, g2, gb = nll_grad_fn(study, exam, y, float(x[0]), float(x[1]), float(x[2]))
        if pk == "gaussian":
            g1 += float(x[0]) / sig2
            g2 += float(x[1]) / sig2
            gb += float(x[2]) / sig2
        return np.array([g1, g2, gb], dtype=np.float64)

    kw = {"method": "L-BFGS-B", "bounds": bnds}
    if nll_grad_fn is not None:
        kw["jac"] = jac
    res = minimize(neg_log_post, np.asarray(x0, dtype=np.float64), **kw)
    x = res.x
    return {"ws": float(x[0]), "we": float(x[1]), "bb": float(x[2]), "success": bool(res.success)}


def ch5_laplace_marginal_intervals(
    study,
    exam,
    y,
    map_w,
    *,
    prior_kind="gaussian",
    mass=0.95,
    nll_fn=None,
    eps=1e-3,
):
    """Normal approximation credible intervals at MAP (validation reference)."""
    from scipy import stats

    ws, we, bb = (float(map_w[0]), float(map_w[1]), float(map_w[2]))
    hess = ch5_posterior_hessian_at(
        study, exam, y, ws, we, bb, prior_kind=prior_kind, nll_fn=nll_fn, eps=eps,
    )
    cov = np.linalg.pinv(-hess)
    z = float(stats.norm.ppf(0.5 + float(mass) / 2.0))
    out = {}
    keys = ("st", "el", "b")
    mus = (ws, we, bb)
    for i, name in enumerate(keys):
        sd = float(np.sqrt(max(cov[i, i], 0.0)))
        mu = mus[i]
        out[name] = (mu - z * sd, mu + z * sd)
    return out, cov


def ch5_posterior_hessian_at(
    study,
    exam,
    y,
    ws,
    we,
    bb,
    *,
    prior_kind="gaussian",
    nll_fn=None,
    eps=1e-4,
):
    """Finite-difference Hessian of log-posterior at ``(ws, we, bb)``.

    Default ``eps`` is small enough that uniform-prior MAP points near the
    support wall stay finite under central differences.
    """
    ws, we, bb = float(ws), float(we), float(bb)
    pk = str(prior_kind).lower()
    # Keep the stencil inside uniform support when the MAP sits on the wall.
    if pk == "uniform":
        w1lo, w1hi, w2lo, w2hi, blo, bhi = ch5_uniform_support_bounds()
        pad = 3.0 * float(eps)
        ws = float(np.clip(ws, w1lo + pad, w1hi - pad))
        we = float(np.clip(we, w2lo + pad, w2hi - pad))
        bb = float(np.clip(bb, blo + pad, bhi - pad))

    hess = np.zeros((3, 3), dtype=np.float64)
    f0 = ch5_log_posterior_at(
        study, exam, y, ws, we, bb, prior_kind=prior_kind, nll_fn=nll_fn,
    )
    if not np.isfinite(f0):
        return hess

    for i in range(3):
        e = np.zeros(3, dtype=np.float64)
        e[i] = float(eps)
        fp = ch5_log_posterior_at(
            study, exam, y, ws + e[0], we + e[1], bb + e[2],
            prior_kind=prior_kind, nll_fn=nll_fn,
        )
        fm = ch5_log_posterior_at(
            study, exam, y, ws - e[0], we - e[1], bb - e[2],
            prior_kind=prior_kind, nll_fn=nll_fn,
        )
        if np.isfinite(fp) and np.isfinite(fm):
            hess[i, i] = (fp - 2.0 * f0 + fm) / (eps * eps)
        elif np.isfinite(fp):
            hess[i, i] = 2.0 * (fp - f0) / (eps * eps)  # one-sided
        elif np.isfinite(fm):
            hess[i, i] = 2.0 * (f0 - fm) / (eps * eps)

    for i in range(3):
        for j in range(i + 1, 3):
            e1 = np.zeros(3, dtype=np.float64)
            e2 = np.zeros(3, dtype=np.float64)
            e1[i] = float(eps)
            e2[j] = float(eps)
            fpp = ch5_log_posterior_at(
                study, exam, y,
                ws + e1[0] + e2[0], we + e1[1] + e2[1], bb + e1[2] + e2[2],
                prior_kind=prior_kind, nll_fn=nll_fn,
            )
            fpm = ch5_log_posterior_at(
                study, exam, y,
                ws + e1[0] - e2[0], we + e1[1] - e2[1], bb + e1[2] - e2[2],
                prior_kind=prior_kind, nll_fn=nll_fn,
            )
            fmp = ch5_log_posterior_at(
                study, exam, y,
                ws - e1[0] + e2[0], we - e1[1] + e2[1], bb - e1[2] + e2[2],
                prior_kind=prior_kind, nll_fn=nll_fn,
            )
            fmm = ch5_log_posterior_at(
                study, exam, y,
                ws - e1[0] - e2[0], we - e1[1] - e2[1], bb - e1[2] - e2[2],
                prior_kind=prior_kind, nll_fn=nll_fn,
            )
            if all(np.isfinite(v) for v in (fpp, fpm, fmp, fmm)):
                hij = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
                hess[i, j] = hij
                hess[j, i] = hij
    return hess


def ch5_hessian_eigen_frame(
    study,
    exam,
    y,
    map_w,
    *,
    prior_kind="uniform",
    nll_fn=None,
    bounds=None,
    margin=0.28,
    length_scale=1.15,
    min_len=0.35,
    eps=1e-4,
):
    """Principal axes of local posterior curvature at the MAP.

    Returns unit eigenvectors of the observed information ``−∇∇ log p``, ordered
    softest→stiffest (longest→shortest arrows), plus on-screen lengths
    ``∝ 1/√λ`` clipped so tips stay inside ``bounds``.
    """
    if bounds is None:
        bounds = CH5_VIEW_BOUNDS
    ws, we, bb = float(map_w[0]), float(map_w[1]), float(map_w[2])
    hess = None
    for trial_eps in (float(eps), 5e-4, 1e-3, 2e-4):
        H = ch5_posterior_hessian_at(
            study, exam, y, ws, we, bb,
            prior_kind=prior_kind, nll_fn=nll_fn, eps=trial_eps,
        )
        if np.isfinite(H).all() and float(np.max(np.abs(H))) > 1e-8:
            hess = H
            break
    if hess is None:
        hess = np.eye(3, dtype=np.float64) * -1.0

    info = -0.5 * (hess + hess.T)
    info = np.nan_to_num(info, nan=0.0, posinf=0.0, neginf=0.0)
    # Force PSD for visualization (tiny ridge if needed).
    try:
        evals, evecs = np.linalg.eigh(info)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(info + 1e-6 * np.eye(3))
    evals = np.clip(np.asarray(evals, dtype=np.float64), 1e-12, None)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    origin = np.array([ws, we, bb], dtype=np.float64)
    dlo1, dhi1, dlo2, dhi2, dlob, dhib = [float(x) for x in bounds]
    lo = np.array([dlo1, dlo2, dlob], dtype=np.float64) + float(margin)
    hi = np.array([dhi1, dhi2, dhib], dtype=np.float64) - float(margin)

    def _max_fit(u):
        u = np.asarray(u, dtype=np.float64)
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            return float(min_len)
        u = u / n
        lim = np.inf
        for k in range(3):
            if abs(u[k]) < 1e-12:
                continue
            if u[k] > 0.0:
                lim = min(lim, (hi[k] - origin[k]) / u[k])
            else:
                lim = min(lim, (lo[k] - origin[k]) / u[k])
        if not np.isfinite(lim) or lim <= 0.0:
            # Point toward interior if MAP is on a wall.
            return float(min_len)
        return float(max(lim, float(min_len)))

    dirs = []
    for j in range(3):
        u = np.asarray(evecs[:, j], dtype=np.float64)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        # Prefer the half-space that keeps more of the arrow on-screen.
        if _max_fit(u) < _max_fit(-u):
            u = -u
        dirs.append(u)

    inv_sqrt = 1.0 / np.sqrt(np.clip(evals, 1e-12, None))
    raw = float(length_scale) * inv_sqrt / float(np.max(inv_sqrt))
    fit_caps = np.array([_max_fit(u) for u in dirs], dtype=np.float64)
    # Scale all arrows together so the longest still fits.
    safe_raw = np.maximum(raw, 1e-9)
    global_fit = float(np.min(fit_caps / safe_raw))
    global_fit = float(np.clip(global_fit, 0.0, 1.0))
    lengths = []
    for j, u in enumerate(dirs):
        L = float(raw[j] * global_fit)
        L = min(L, float(fit_caps[j]) * 0.92)
        lengths.append(max(L, float(min_len) * 0.85))

    return {
        "origin": origin,
        "evals": np.asarray(evals, dtype=np.float64),
        "dirs": dirs,
        "lengths": lengths,
        "hess": hess,
        "info": info,
    }