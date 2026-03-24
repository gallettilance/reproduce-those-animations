import bpy
import math
import mathutils
import random
from mathutils import Vector
# ============================================================
# Hierarchical clustering / dendrogram animation in Blender
# ------------------------------------------------------------
# What this script builds:
# 1. A 2D-looking dendrogram above the scene (merge ovals start as thin chords, then grow to member bounds)
# 2. A point cloud dataset below it
# 3. Merge rings: thin LR chord (scaled circle) → expanded circle around members
# 4. Top view (~1s hold + tilt): threshold slice; sweep in Z; ovals thicken; points purple vs white by cluster cut
# 5. Camera: perspective throughout — front → elevate ~60° → full CCW yaw → expand ovals → glide to top → +X orbit to diagonal (sweep) → return
#
# Other ways to read “depth” while staying graphically flat (2D):
# - Edge-fade shader: map plane UV to mix(transparent, emission) so only a soft band is visible.
# - Replace the fill with a dashed line + tick marks on a slim vertical “ruler” mesh.
# - Small square cut marker on a fixed-size threshold plane or a bright curve at the cut height.
# - Picture-in-picture: tiny ortho side camera showing only the Z stack next to the top view.
# - Animate only color/opacity on the dendrogram stems at height h, with no plane at all.
#
# How to use:
# - Open a new Blender file.
# - Switch to Scripting. 
# - Paste this script and Run.
# - Scrub the timeline.
#
# Notes:
# - This script implements simple agglomerative clustering directly,
#   so you do not need scipy inside Blender.
# - It is designed for teaching/visualization, not huge datasets.
# ============================================================

# -----------------------------
# User controls
# -----------------------------
N_POINTS = 15
# After generation, drop point(s) by index (same indices as Blender objects P0, P1, …). Re-run the script
# after changing this; deleting meshes by hand leaves a broken dendrogram until you run again.
POINT_EXCLUDE_INDICES = {1}
# For n >= 9: minimum horizontal separation so front view does not stack points.
MIN_X_PAIR_GAP = 0.34

LINKAGE = 'single'   # 'single', 'complete', 'average', 'centroid'
POINT_RADIUS = 0.1
CURVE_BEVEL = 0.03
# Vertical dendrogram pipes (scaled from CURVE_BEVEL when building connector curves).
DENDRO_CURVE_BEVEL_SCALE = 1.35
CIRCLE_BEVEL = 0.01
# Collapsed merge ovals: major axis = half the L–R chord; minor axis ~ line thickness (reads as a diagonal segment).
OVAL_COLLAPSE_MINOR = 0.018
THRESHOLD_HEIGHT = 2.6
HEIGHT_SCALE = 1.05  # slightly tighter Z stack for small sets (clearer side-view separation)
DATA_Y_OFFSET = 0.0
DENDRO_Y_OFFSET = 0.0
DENDRO_X_SCALE = 0.7
DATA_Z = 0.0
# Stems start at the top of the point spheres; add a small lift if you want a visible gap.
DENDRO_Z_LIFT = 0.0
DENDRO_BASE_Z = DATA_Z + POINT_RADIUS + DENDRO_Z_LIFT
PLANE_SIZE = 18.0
# Dark floor under the dataset; large so wide shots / top views don’t show a bright rim past the mesh.
GROUND_PLANE_SIZE = 140.0
# Inverted emission sphere around the scene pivot (see add_dark_sky_dome); catches rays when world BG stays bright.
SKY_DOME_RADIUS = 480.0

# Threshold slice: simple translucent emission card (no grid shader).
# Emission brightness for the slice (only the emission side of the mix; see THRESHOLD_PLANE_MIX).
THRESHOLD_PLANE_EMISSION_STRENGTH = 0.32
THRESHOLD_LABEL_SCALE = 0.32
# Mix Shader Fac while the slice is visible: 0 = all Transparent BSDF, 1 = full Emission (lower = more see-through).
THRESHOLD_PLANE_MIX = 0.2
# Plane starts no lower than (highest merge z) + this (above top merge “circle” level).
THRESHOLD_ABOVE_TOP_MERGE_Z = 0.22
# Sweep low Z: dip slightly past the leaf / zero-merge level (DENDRO_BASE_Z) so the slice clears the finest clusters.
THRESHOLD_SWEEP_Z_BELOW_LEAF = 0.065
# Down: u = 1-(1-t)^a → larger a = faster motion at sweep start (still eases near bottom).
SWEEP_DOWN_FAST_EXP = 3.65
# Lower = more even speed up; higher = slower start then rush (paired with shorter F_SWEEP_UP_FRAMES).
SWEEP_UP_ACCEL_EXP = 2.35

# Timeline: (0) points slide in Y from 0 → layout (dendrogram/circles hidden); (1) elevate camera ~60° from front,
# (2) full CCW yaw orbit, (3) expand merge circles, (4) persp glide to top, (5) pause, (6) threshold orbit + sweep, (7) return.
F_SCENE_START = 1
# Data-space Y starts at 0 (world Y = DATA_Y_OFFSET); eases to each point’s layout Y over this span.
POINT_Y_SETTLE_FRAMES = 24
F_MAIN_START = F_SCENE_START + POINT_Y_SETTLE_FRAMES
F_ELEV_END = F_MAIN_START + 24
F_YAW_ORBIT_START = F_ELEV_END
F_YAW_ORBIT_END = F_ELEV_END + 96
F_EXTEND_START = F_YAW_ORBIT_END + 1
F_EXTEND_END = F_EXTEND_START + 24
F_EXTEND2_START = F_EXTEND_END + 1
F_EXTEND2_END = F_EXTEND2_START + 24

F_POST_EXPAND_HOLD_END = F_EXTEND2_END + 24
F_TO_TOP_START = F_POST_EXPAND_HOLD_END + 1
F_TO_TOP_FRAMES = 96
F_TOP = F_TO_TOP_START + F_TO_TOP_FRAMES - 1
# After top: ~1s hold before tilting toward threshold diagonal.
TOP_VIEW_PAUSE_FRAMES = 24
TILT_TO_DIAGONAL_FRAMES = 24
F_TOP_HOLD_END = F_TOP + TOP_VIEW_PAUSE_FRAMES
F_THRESHOLD_DIAG_VIEW = F_TOP_HOLD_END + TILT_TO_DIAGONAL_FRAMES
# ConnL/ConnR/ConnUp stems: hide only during the top-view hold (F_TOP … F_TOP_HOLD_END). Set False to always show.
DENDRO_HIDE_IN_TOP_VIEW_PHASES = True
# Wire slice appears only after the top→diagonal orbit finishes (avoids a duplicate “top” read of the grid).
F_THRESHOLD_SWEEP_START = F_THRESHOLD_DIAG_VIEW + 1
# Slice stays at z_top (visible) before moving down, and again at z_top after rising before hide / camera orbit.
SWEEP_HOLD_TOP_BEFORE_DESCENT = 24
SWEEP_HOLD_TOP_AFTER_ASCENT = 24
F_SWEEP_DESCENT_START = F_THRESHOLD_SWEEP_START + SWEEP_HOLD_TOP_BEFORE_DESCENT
F_SWEEP_DOWN_FRAMES = 48  # quick descent
F_THRESHOLD_SWEEP_END = F_SWEEP_DESCENT_START + F_SWEEP_DOWN_FRAMES
F_SWEEP_UP_FRAMES = 48  # faster climb than legacy 300; SWEEP_UP_ACCEL_EXP still shapes ease
F_THRESHOLD_RETURN_END = F_THRESHOLD_SWEEP_END + F_SWEEP_UP_FRAMES
F_THRESHOLD_TOP_HOLD_END = F_THRESHOLD_RETURN_END + SWEEP_HOLD_TOP_AFTER_ASCENT
# First frame after top hold: hide threshold slice / reset point colors; camera return orbit starts here too.
F_THRESHOLD_VFX_END = F_THRESHOLD_TOP_HOLD_END + 1
# Reverse +X orbit (same duration as top→diagonal): back to true top, then ortho glide to side matching intro length.
F_BACK_AT_TOP = F_THRESHOLD_VFX_END + (F_THRESHOLD_DIAG_VIEW - F_TOP)
F_FINAL_GLIDE_END = F_BACK_AT_TOP + (F_TOP - F_TO_TOP_START)
F_RETURN_FRONT = F_FINAL_GLIDE_END + 1

# -----------------------------
# Cleanup
# -----------------------------
# Re-running from the Text Editor often leaves objects behind: bpy.ops.object.delete depends on a 3D
# viewport context and can no-op, so a second run stacks Oval_*, Conn_*, etc. (different growth = two actions).
_SCRIPT_COLLECTION_BASES = frozenset({'Points', 'Dendrogram', 'ClusterCircles', 'Guides'})


def _is_script_collection(col):
    if col is None:
        return False
    base = col.name.split('.')[0]
    return col.name in _SCRIPT_COLLECTION_BASES or base in _SCRIPT_COLLECTION_BASES


def _purge_script_collections():
    for parent in list(bpy.data.collections):
        for child in list(parent.children):
            if _is_script_collection(child):
                parent.children.unlink(child)
    for col in list(bpy.data.collections):
        if _is_script_collection(col):
            bpy.data.collections.remove(col)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    _purge_script_collections()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.curves):
        if block.users == 0:
            bpy.data.curves.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)

clear_scene()

scene = bpy.context.scene
scene.frame_start = F_SCENE_START
scene.frame_end = F_RETURN_FRONT + 48
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.eevee.use_bloom = True
scene.eevee.taa_render_samples = 32
scene.eevee.taa_samples = 32
scene.render.film_transparent = False
# Compositor trees (e.g. “white backdrop” viewer setups) override the render; this script expects no compositing.
scene.use_nodes = False
scene.view_settings.exposure = 0.0

# World: new datablock + minimal tree. EEVEE Next can still show a bright default in some files; see SkyDome mesh below.
RENDER_WORLD_BG_RGB = (0.0, 0.0, 0.0)


def configure_dark_render_world(target_scene, rgb=RENDER_WORLD_BG_RGB):
    w = bpy.data.worlds.new('DendrogramWorld')
    w.use_nodes = True
    nt = w.node_tree
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new(type='ShaderNodeOutputWorld')
    bg = nodes.new(type='ShaderNodeBackground')
    bg.location = (-280, 0)
    bg.inputs['Color'].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    bg.inputs['Strength'].default_value = 1.0
    try:
        links.new(bg.outputs['Background'], out.inputs['Surface'])
    except (KeyError, TypeError, RuntimeError):
        links.new(bg.outputs[0], out.inputs[0])
    target_scene.world = w


configure_dark_render_world(scene)

# -----------------------------
# Materials
# -----------------------------
# Palette: purple clusters / threshold, white singletons, cool stems on black.
CLUSTER_PURPLE = (0.58, 0.32, 0.88, 1.0)
POINT_WHITE = (1.0, 1.0, 1.0, 1.0)
DENDRO_VERTICAL_COLOR = (0.62, 0.78, 0.98, 1.0)
THRESHOLD_SLICE_COLOR = (0.75, 0.55, 0.98, 1.0)


def make_emission_material(name, color=(1,1,1,1), strength=2.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)
    out = nodes.new('ShaderNodeOutputMaterial')
    mix = nodes.new('ShaderNodeMixShader')
    transp = nodes.new('ShaderNodeBsdfTransparent')
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = color
    emit.inputs['Strength'].default_value = strength
    mix.inputs['Fac'].default_value = 1.0 - alpha
    links.new(transp.outputs[0], mix.inputs[1])
    links.new(emit.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs[0])
    return mat

def make_threshold_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in list(nodes):
        nodes.remove(n)

    output = nodes.new('ShaderNodeOutputMaterial')
    # Outer mix: keyed by animate_threshold_sweep (must stay named 'Mix Shader').
    mix_main = nodes.new('ShaderNodeMixShader')
    mix_main.name = 'Mix Shader'
    transp = nodes.new('ShaderNodeBsdfTransparent')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = THRESHOLD_SLICE_COLOR
    emission.inputs['Strength'].default_value = THRESHOLD_PLANE_EMISSION_STRENGTH
    links.new(transp.outputs[0], mix_main.inputs[1])
    links.new(emission.outputs[0], mix_main.inputs[2])
    links.new(mix_main.outputs[0], output.inputs[0])
    mix_main.inputs['Fac'].default_value = 0

    return mat

def make_principled_material(name, color=(1,1,1,1), roughness=0.35, alpha=1.0, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Alpha'].default_value = alpha
    if 'Emission Color' in bsdf.inputs:
        bsdf.inputs['Emission Color'].default_value = color
        bsdf.inputs['Emission Strength'].default_value = emission_strength
    return mat

mat_point = make_principled_material(
    'PointMat', POINT_WHITE, roughness=0.25, emission_strength=0.28
)
mat_dendro = make_principled_material(
    'DendroMat',
    DENDRO_VERTICAL_COLOR,
    roughness=0.12,
    emission_strength=0.55,
)
mat_threshold = make_threshold_material('ThresholdWireMat')
mat_ground = make_principled_material('GroundMat', (0.03, 0.035, 0.045, 1.0), roughness=0.75)
# Inverted giant sphere: opaque black (do not use make_emission_material with alpha=1 — that mixes to transparent).
mat_sky_dome = make_principled_material(
    'SkyDomeMat', (0.0, 0.0, 0.0, 1.0), roughness=1.0, alpha=1.0, emission_strength=0.0
)
mat_sky_dome.blend_method = 'OPAQUE'

# -----------------------------
# Helpers
# -----------------------------
def add_uv_sphere(name, location, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    return obj

def add_plane(name, location, size, material):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_dark_sky_dome(name, center, radius, material):
    """UV sphere with normals flipped so the interior is shaded; place cameras inside for a black backdrop."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(float(center.x), float(center.y), float(center.z)))
    dome = bpy.context.active_object
    dome.name = name
    r = max(float(radius), 1.0)
    dome.scale = (r, r, r)
    dome.data.materials.append(material)
    bpy.context.view_layer.objects.active = dome
    dome.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    dome.select_set(False)
    if hasattr(dome, 'visible_shadow'):
        try:
            dome.visible_shadow = False
        except (TypeError, AttributeError):
            pass
    return dome


def new_curve_object(name, points, bevel=0.02, material=None, cyclic=False):
    clean_points = []
    for i, p in enumerate(points):
        if p is None:
            raise ValueError(f"{name}: point {i} is None")
        if len(p) != 3:
            raise ValueError(f"{name}: point {i} does not have 3 coordinates: {p}")
        if any(v is None for v in p):
            raise ValueError(f"{name}: point {i} has None coordinate: {p}")
        clean_points.append((float(p[0]), float(p[1]), float(p[2])))

    curve_data = bpy.data.curves.new(name=name + '_Curve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = bevel
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(clean_points) - 1)
    for i, p in enumerate(clean_points):
        spline.points[i].co = (p[0], p[1], p[2], 1.0)
    spline.use_cyclic_u = cyclic

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    if material:
        obj.data.materials.append(material)
    return obj


def ellipse_points_local(a, b, z=0.0, n=100):
    """Closed ellipse in local XY: semi-axis a along +X, b along +Y (object Z rotation aligns +X with merge chord)."""
    pts = []
    for i in range(n):
        phi = 2 * math.pi * i / n
        x = a * math.cos(phi)
        y = b * math.sin(phi)
        pts.append((x, y, z))
    return pts

# -----------------------------
# Agglomerative clustering
# -----------------------------
class ClusterNode:
    def __init__(self, node_id, members, left=None, right=None, height=0.0):
        self.id = node_id
        self.members = members
        self.left = left
        self.right = right
        self.height = height
        self.x = None  # dendrogram horizontal position


def euclid(a, b):
    return math.dist(a, b)


def centroid(indices, pts):
    x = sum(pts[i][0] for i in indices) / len(indices)
    y = sum(pts[i][1] for i in indices) / len(indices)
    return (x, y)


def cluster_distance(a_inds, b_inds, pts, linkage='single'):
    pairs = [euclid(pts[i], pts[j]) for i in a_inds for j in b_inds]
    if linkage == 'single':
        return min(pairs)
    if linkage == 'complete':
        return max(pairs)
    if linkage == 'average':
        return sum(pairs) / len(pairs)
    if linkage == 'centroid':
        ca = centroid(a_inds, pts)
        cb = centroid(b_inds, pts)
        return euclid(ca, cb)
    raise ValueError(f'Unknown linkage: {linkage}')


def agglomerative(points, linkage='single'):
    nodes = {}
    active = []
    for i, _ in enumerate(points):
        n = ClusterNode(i, [i], height=0.0)
        nodes[i] = n
        active.append(i)

    next_id = len(points)
    merges = []

    while len(active) > 1:
        best = None
        best_d = 1e18
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a_id, b_id = active[i], active[j]
                d = cluster_distance(nodes[a_id].members, nodes[b_id].members, points, linkage)
                if d < best_d:
                    best_d = d
                    best = (a_id, b_id)

        a_id, b_id = best
        new_members = nodes[a_id].members + nodes[b_id].members
        new_node = ClusterNode(next_id, new_members, left=a_id, right=b_id, height=best_d)
        nodes[next_id] = new_node
        merges.append(next_id)

        active = [x for x in active if x not in (a_id, b_id)]
        active.append(next_id)
        next_id += 1

    root_id = active[0]
    return nodes, root_id, merges


def merge_height_min_gap(pts, linkage):
    """Larger min gap between consecutive merge heights => merges less piled at one threshold."""
    nodes, _, merge_ids = agglomerative(pts, linkage=linkage)
    if not merge_ids:
        return -1.0
    hs = sorted(nodes[mid].height for mid in merge_ids)
    gaps = [hs[i + 1] - hs[i] for i in range(len(hs) - 1)]
    return min(gaps) if gaps else 0.0


def min_pairwise_x_gap(pts):
    """Smallest |x_i - x_j| over distinct points (sorted adjacent gap)."""
    xs = sorted(p[0] for p in pts)
    if len(xs) < 2:
        return 1e18
    return min(xs[i + 1] - xs[i] for i in range(len(xs) - 1))


def build_even_merge_points(n, linkage, steps=12000, seed=42):
    """Optimize merge-height spacing; n=8 uses a compact two-cluster seed for a readable dendrogram.

    For n >= 9, x coordinates are seeded evenly on [-span, span] (shuffled among points) so the
    minimum x gap is ~2*span/(n-1); optimization only accepts moves that keep min pairwise x gap
    >= MIN_X_PAIR_GAP.
    """
    random.seed(seed)
    enforce_x_gap = n >= 9
    span = 2.35 if n <= 8 else max(3.85, MIN_X_PAIR_GAP * (n - 1) * 0.52)
    sigma = 0.11 if n <= 8 else 0.17
    y_lim = span * 0.58 + 0.35
    x_lim = span + 0.5

    if n == 8:
        # Two groups in X (clean root split in side view); jitter within groups for inner merges.
        best = [
            (-2.05, 0.05),
            (-1.72, -0.42),
            (-1.92, 0.52),
            (-1.48, -0.18),
            (1.48, 0.08),
            (1.92, -0.38),
            (1.72, 0.55),
            (2.02, -0.12),
        ]
    elif enforce_x_gap:
        denom = max(1, n - 1)
        xs_seed = [-span + (2 * span * i / denom) for i in range(n)]
        random.shuffle(xs_seed)
        best = [
            (xs_seed[i], random.uniform(-span * 0.55, span * 0.55))
            for i in range(n)
        ]
    else:
        best = [
            (random.uniform(-span, span), random.uniform(-span * 0.55, span * 0.55))
            for _ in range(n)
        ]

    best_score = merge_height_min_gap(best, linkage)
    for _ in range(steps):
        cand = [list(p) for p in best]
        i = random.randrange(n)
        cand[i][0] += random.gauss(0, sigma)
        cand[i][1] += random.gauss(0, sigma)
        cand[i][0] = min(max(cand[i][0], -x_lim), x_lim)
        cand[i][1] = min(max(cand[i][1], -y_lim), y_lim)
        tup = [tuple(p) for p in cand]
        if enforce_x_gap and min_pairwise_x_gap(tup) < MIN_X_PAIR_GAP:
            continue
        sc = merge_height_min_gap(tup, linkage)
        if sc > best_score:
            best_score = sc
            best = tup
    return best


_raw_points = build_even_merge_points(N_POINTS, LINKAGE)
POINTS = [
    p for i, p in enumerate(_raw_points) if i not in POINT_EXCLUDE_INDICES
]


def keyframe_circle_state(obj, frame, x, y, z, sx=1.0, sy=1.0, sz=1.0):
    obj.location = (x, y, z)
    obj.scale = (sx, sy, sz)
    obj.keyframe_insert(data_path='location', frame=frame)
    obj.keyframe_insert(data_path='scale', frame=frame)


def leaf_cluster_labels(node_id, nodes, thr, counter):
    """Cut the tree at merge height thr: leaves in the same returned group share one id."""
    node = nodes[node_id]
    if len(node.members) == 1:
        i = node.members[0]
        c = counter[0]
        counter[0] += 1
        return {i: c}
    if node.height <= thr:
        c = counter[0]
        counter[0] += 1
        return {leaf: c for leaf in node.members}
    d = {}
    d.update(leaf_cluster_labels(node.left, nodes, thr, counter))
    d.update(leaf_cluster_labels(node.right, nodes, thr, counter))
    return d


def leaf_cluster_member_counts(root_id, nodes, thr, n_pts):
    """How many leaves share each cluster id at threshold thr (singleton ids → count 1)."""
    labels = leaf_cluster_labels(root_id, nodes, thr, [0])
    counts = {}
    for li in range(n_pts):
        cid = labels.get(li, -1)
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def height_from_plane_z(z, max_h):
    return (z - DENDRO_BASE_Z) / (HEIGHT_SCALE * 4.0) * max_h


def plane_z_from_height(h, max_h):
    return DENDRO_BASE_Z + (h / max_h) * HEIGHT_SCALE * 4.0 if max_h > 0 else DENDRO_BASE_Z


def merge_qualifies_for_circle(node, nodes):
    """
    All internal merges get a ring except degenerate zero-distance merges
    (duplicate/near-duplicate points), which caused a bogus lone-point ring at h≈0.
    """
    if node.left is None or node.right is None:
        return False
    if len(node.members) < 2:
        return False
    return node.height > 1e-12


def threshold_sweep_ends(max_h, z_top_floor=None):
    """Z range for the threshold sweep (horizontal band; XY size is fixed separately).

    z_top_floor: if set, z_top is at least this (e.g. above top merge geometry).
    """
    z_hi0 = plane_z_from_height(max_h * 1.08, max_h)
    z_lo0 = DATA_Z - 0.04
    R = max(z_hi0 - z_lo0, 1e-6)
    z_top = z_hi0 - 0.25 * R
    if z_top_floor is not None:
        z_top = max(z_top, float(z_top_floor))
    z_bot = z_lo0 + 0.25 * R
    z_bot = min(z_bot, DENDRO_BASE_Z - THRESHOLD_SWEEP_Z_BELOW_LEAF)
    return z_top, z_bot


def merge_levels_z_max(nodes, cluster_geom):
    zs = [
        float(cluster_geom[nid]["z"])
        for nid, node in nodes.items()
        if len(node.members) >= 2
    ]
    return max(zs) if zs else float(DENDRO_BASE_Z)


def threshold_plane_z_for_sweep_frame(f, z_top, z_bot):
    """Z: hold at top → ease down → faster ease up → hold at top; then flat z_top after F_THRESHOLD_TOP_HOLD_END."""
    if f < F_THRESHOLD_SWEEP_START:
        return z_top
    if f < F_SWEEP_DESCENT_START:
        return z_top
    if f > F_THRESHOLD_TOP_HOLD_END:
        return z_top
    if f <= F_THRESHOLD_SWEEP_END:
        d_down = max(1, F_THRESHOLD_SWEEP_END - F_SWEEP_DESCENT_START)
        t = (f - F_SWEEP_DESCENT_START) / float(d_down)
        t = max(0.0, min(1.0, t))
        u = 1.0 - (1.0 - t) ** SWEEP_DOWN_FAST_EXP
        return z_top + (z_bot - z_top) * u
    if f <= F_THRESHOLD_RETURN_END:
        d_up = max(1, F_THRESHOLD_RETURN_END - F_THRESHOLD_SWEEP_END)
        t = (f - F_THRESHOLD_SWEEP_END) / float(d_up)
        t = max(0.0, min(1.0, t))
        u = t ** SWEEP_UP_ACCEL_EXP
        return z_bot + (z_top - z_bot) * u
    return z_top


def keyframe_points_y_settle(point_objs, points_xy, f_start, f_end):
    """World Y eases from ``DATA_Y_OFFSET`` (data y = 0) to ``DATA_Y_OFFSET + y`` for each point."""
    f0, f1 = int(f_start), int(f_end)
    for obj, (x, y) in zip(point_objs, points_xy):
        if obj.animation_data is None:
            obj.animation_data_create()
        if f1 <= f0:
            yy = float(DATA_Y_OFFSET) + float(y)
            obj.location = (float(x), yy, float(DATA_Z))
            obj.keyframe_insert(data_path='location', frame=f0)
            if obj.animation_data.action:
                _linearize_action_range(obj.animation_data.action, f0, f0)
            continue
        span = float(max(1, f1 - f0))
        for f in range(f0, f1 + 1):
            u = _smoothstep01((f - f0) / span)
            yy = float(DATA_Y_OFFSET) + float(y) * u
            obj.location = (float(x), yy, float(DATA_Z))
            obj.keyframe_insert(data_path='location', frame=f)
        if obj.animation_data.action:
            _linearize_action_range(obj.animation_data.action, f0, f1)


def keyframe_dendro_connectors_y_settle(specs, f_start, f_end):
    """POLY curves with two points sharing (x, z); world Y = DATA_Y_OFFSET + u * y_data (matches point settle)."""
    f0, f1 = int(f_start), int(f_end)
    for spec in specs:
        obj = spec['obj']
        cd = obj.data
        if cd.animation_data is None:
            cd.animation_data_create()
        x = float(spec['x'])
        yd = float(spec['y_data'])
        z0 = float(spec['z0'])
        z1 = float(spec['z1'])
        sp = cd.splines[0]

        def _apply_frame(f, u):
            yw = float(DATA_Y_OFFSET) + yd * u
            sp.points[0].co = (x, yw, z0, 1.0)
            sp.points[1].co = (x, yw, z1, 1.0)
            sp.points[0].keyframe_insert(data_path='co', frame=f)
            sp.points[1].keyframe_insert(data_path='co', frame=f)

        if f1 <= f0:
            _apply_frame(f0, 1.0)
            if cd.animation_data.action:
                _linearize_action_range(cd.animation_data.action, f0, f0)
            continue
        span = float(max(1, f1 - f0))
        for f in range(f0, f1 + 1):
            u = _smoothstep01((f - f0) / span)
            _apply_frame(f, u)
        if cd.animation_data.action:
            _linearize_action_range(cd.animation_data.action, f0, f1)


def keyframe_merge_ovals_follow_vertical_tips(meta_items, f_start, f_end):
    """Oval midpoint, Z rotation, and major-axis scale track the chord between child tips (same u as verticals)."""
    f0, f1 = int(f_start), int(f_end)
    for item in meta_items:
        obj = item['obj']
        lx, ly = float(item['lx']), float(item['ly'])
        rx, ry = float(item['rx']), float(item['ry'])
        zf = float(item['z'])
        th_final = float(item['theta'])
        r_circ = max(float(item['radius']), 1e-9)
        s0y = float(item['s0y'])
        if obj.animation_data is None:
            obj.animation_data_create()
        dx_full = rx - lx

        def _pose(u):
            yl = float(DATA_Y_OFFSET) + ly * u
            yr = float(DATA_Y_OFFSET) + ry * u
            mx = 0.5 * (lx + rx)
            my = 0.5 * (yl + yr)
            dy = (ry - ly) * u
            if abs(dx_full) < 1e-12 and abs(dy) < 1e-12:
                th = th_final
            else:
                th = math.atan2(dy, dx_full)
            d_chord = math.hypot(dx_full, (ry - ly) * u)
            a_u = max(0.5 * d_chord, 1e-4)
            sx_u = min(1.0, a_u / r_circ)
            obj.location = (mx, my, zf)
            obj.rotation_euler = (0.0, 0.0, th)
            obj.scale = (sx_u, s0y, 1.0)

        if f1 <= f0:
            _pose(1.0)
            obj.keyframe_insert(data_path='location', frame=f0)
            obj.keyframe_insert(data_path='rotation_euler', frame=f0)
            obj.keyframe_insert(data_path='scale', frame=f0)
            if obj.animation_data.action:
                _linearize_action_range(obj.animation_data.action, f0, f0)
            continue
        span = float(max(1, f1 - f0))
        for f in range(f0, f1 + 1):
            u = _smoothstep01((f - f0) / span)
            _pose(u)
            obj.keyframe_insert(data_path='location', frame=f)
            obj.keyframe_insert(data_path='rotation_euler', frame=f)
            obj.keyframe_insert(data_path='scale', frame=f)
        if obj.animation_data.action:
            _linearize_action_range(obj.animation_data.action, f0, f1)


def _force_constant_interpolation_on_hide_fcurves(action):
    if action is None:
        return
    for fc in action.fcurves:
        if fc.data_path in ('hide_viewport', 'hide_render'):
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'


def keyframe_dendro_hide_for_top_view_phases(dendro_objects):
    """Hide connector curves (vertical stems) only during the top-view hold (inclusive).

    Run after ``animate_threshold_sweep`` so sweep keys stay authoritative elsewhere.
    """
    if not DENDRO_HIDE_IN_TOP_VIEW_PHASES or not dendro_objects:
        return
    h0, h1 = int(F_TOP)- 20, int(F_TOP) + 20
    if h1 < h0:
        return
    fa = max(h0 - 1, int(F_SCENE_START))
    fafter = h1 + 1
    for obj in dendro_objects:
        if obj.animation_data is None:
            obj.animation_data_create()
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_viewport', frame=fa)
        obj.keyframe_insert(data_path='hide_render', frame=fa)
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path='hide_viewport', frame=h0)
        obj.keyframe_insert(data_path='hide_render', frame=h0)
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path='hide_viewport', frame=h1)
        obj.keyframe_insert(data_path='hide_render', frame=h1)
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_viewport', frame=fafter)
        obj.keyframe_insert(data_path='hide_render', frame=fafter)
        _force_constant_interpolation_on_hide_fcurves(obj.animation_data.action)


def threshold_slice_xy_scale_and_center(points, mesh_plane_size):
    """Uniform XY scale and world (x, y) for the threshold slice plane (matches point cloud extent)."""
    if not points:
        return 1.0, 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    rx = max(xs) - min(xs)
    ry = max(ys) - min(ys)
    span = max(rx, ry, 0.15)
    margin = 1.15
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys)) + DATA_Y_OFFSET
    ps = max(margin * span / max(float(mesh_plane_size), 1e-6), 0.25)
    return ps, cx, cy


def _smoothstep01(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _smootherstep01(t):
    """Perlin-style smoother step; gentler in/out than _smoothstep01 (better for long camera glides)."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def camera_euler_look_at(cam_pos, target, up_axis='Y'):
    """World-space euler so camera -Z aims at target (Blender camera convention)."""
    d = target - cam_pos
    if d.length < 1e-8:
        return mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
    return d.to_track_quat('-Z', up_axis).to_euler('XYZ')


def perspective_lens_mm_match_span(cam_data, target_span_world, subject_distance_world):
    """Focal length (mm) so perspective width at ``subject_distance_world`` matches ``target_span_world``.

    Same sensor convention as Blender ortho_scale (AUTO → max of width/height).
    """
    sw = float(getattr(cam_data, 'sensor_width', 36.0) or 36.0)
    sh = float(getattr(cam_data, 'sensor_height', 24.0) or 24.0)
    fit = getattr(cam_data, 'sensor_fit', 'AUTO')
    if fit == 'VERTICAL':
        sensor_ref = sh
    elif fit == 'HORIZONTAL':
        sensor_ref = sw
    else:
        sensor_ref = max(sw, sh)
    d = max(float(subject_distance_world), 1e-6)
    span = max(float(target_span_world), 1e-6)
    return min(250.0, max(1.0, sensor_ref * d / span))


def _linearize_action_range(action, f_start, f_end):
    if action is None:
        return
    for fc in action.fcurves:
        # Enum properties (e.g. camera type): stepped keys; LINEAR can produce invalid in-betweens.
        if fc.data_path.endswith('type'):
            for kp in fc.keyframe_points:
                xf = kp.co[0]
                if f_start <= xf <= f_end:
                    kp.interpolation = 'CONSTANT'
            continue
        for kp in fc.keyframe_points:
            xf = kp.co[0]
            if f_start <= xf <= f_end:
                kp.interpolation = 'LINEAR'


def _ensure_camera_actions(camera):
    if camera.animation_data is None:
        camera.animation_data_create()
    if camera.data.animation_data is None:
        camera.data.animation_data_create()


def _commit_persp_camera_keyframe(camera, frame, cam_pivot, target_span_world):
    """Insert keys after ``camera.location`` / ``rotation_euler`` are set; lens matches span at pivot distance."""
    camera.data.type = 'PERSP'
    d = (camera.location - cam_pivot).length
    camera.data.lens = perspective_lens_mm_match_span(
        camera.data, target_span_world, d
    )
    camera.keyframe_insert(data_path='location', frame=frame)
    camera.keyframe_insert(data_path='rotation_euler', frame=frame)
    camera.data.keyframe_insert(data_path='type', frame=frame)
    camera.data.keyframe_insert(data_path='lens', frame=frame)


def _linearize_camera_segment(camera, f_start, f_end):
    for ad in (camera.animation_data, camera.data.animation_data):
        if ad is not None:
            _linearize_action_range(ad.action, f_start, f_end)


def keyframe_camera_persp_elevate(
    camera,
    cam_pivot,
    loc_start,
    loc_end,
    target_span_world,
    f_start,
    f_end,
):
    """Smooth perspective move between two positions, always looking at pivot; lens tracks distance."""
    _ensure_camera_actions(camera)
    span = max(1, int(f_end) - int(f_start))
    q_s = camera_euler_look_at(loc_start, cam_pivot).to_quaternion()
    q_e = camera_euler_look_at(loc_end, cam_pivot).to_quaternion()
    for f in range(int(f_start), int(f_end) + 1):
        u = _smoothstep01((f - f_start) / float(span))
        camera.location = loc_start.lerp(loc_end, u)
        camera.rotation_euler = q_s.slerp(q_e, u).to_euler('XYZ')
        _commit_persp_camera_keyframe(camera, f, cam_pivot, target_span_world)
    _linearize_camera_segment(camera, f_start, f_end)


def keyframe_camera_persp_yaw_orbit(
    camera,
    cam_pivot,
    horizontal_distance,
    elev_deg,
    target_span_world,
    f_start,
    f_end,
):
    """Full CCW orbit (viewed from +Z) at fixed elevation; phi=0 is toward -Y (front)."""
    _ensure_camera_actions(camera)
    elev = math.radians(float(elev_deg))
    hd = max(0.01, float(horizontal_distance))
    ch = hd * math.cos(elev)
    sz = hd * math.sin(elev)
    arm0 = Vector((0.0, -ch, sz))
    span = max(1, int(f_end) - int(f_start))
    for f in range(int(f_start), int(f_end) + 1):
        u = _smoothstep01((f - f_start) / float(span))
        ang = u * 2.0 * math.pi
        rz = mathutils.Matrix.Rotation(ang, 3, 'Z')
        off = rz @ arm0
        camera.location = cam_pivot + off
        camera.rotation_euler = camera_euler_look_at(camera.location, cam_pivot)
        _commit_persp_camera_keyframe(camera, f, cam_pivot, target_span_world)
    _linearize_camera_segment(camera, f_start, f_end)


def keyframe_camera_orbit_rx_yz_arc(
    camera,
    cam_pivot,
    top_loc,
    rx_end_rad,
    distance_scale_end,
    target_span_world,
    f_start,
    f_end,
    *,
    reverse=False,
):
    """Orbit camera around world +X through pivot (arm from pivot along +Z toward ``top_loc``).

    ``reverse=False``: top (u=0) → tilted end (u=1). ``reverse=True``: the same path backward.
    """
    _ensure_camera_actions(camera)
    span = max(1, int(f_end) - int(f_start))
    off0 = top_loc - cam_pivot
    H = off0.z if abs(off0.z) > 1e-6 else 20.0
    base_arm = Vector((0.0, 0.0, H))

    for f in range(int(f_start), int(f_end) + 1):
        u_raw = (f - f_start) / float(span)
        u = 1.0 - _smoothstep01(u_raw) if reverse else _smoothstep01(u_raw)
        ang = u * rx_end_rad
        rx_m = mathutils.Matrix.Rotation(ang, 3, 'X')
        off = rx_m @ base_arm
        rmul = 1.0 + u * (float(distance_scale_end) - 1.0)
        camera.location = cam_pivot + off * rmul
        camera.rotation_euler = camera_euler_look_at(camera.location, cam_pivot)
        _commit_persp_camera_keyframe(camera, f, cam_pivot, target_span_world)

    _linearize_camera_segment(camera, f_start, f_end)


def keyframe_camera_persp_glide(
    camera,
    loc_a,
    rot_a,
    loc_b,
    rot_b,
    cam_pivot,
    target_span_world,
    f_start,
    f_end,
):
    """Smooth perspective move: lerp location, slerp rotation; lens tracks span at pivot distance."""
    _ensure_camera_actions(camera)
    span = max(1, int(f_end) - int(f_start))
    q_a = mathutils.Euler(rot_a, 'XYZ').to_quaternion()
    q_b = mathutils.Euler(rot_b, 'XYZ').to_quaternion()

    for f in range(int(f_start), int(f_end) + 1):
        t_lin = (f - f_start) / float(span)
        u = _smootherstep01(t_lin)
        camera.location = loc_a.lerp(loc_b, u)
        camera.rotation_euler = q_a.slerp(q_b, u).to_euler('XYZ')
        _commit_persp_camera_keyframe(camera, f, cam_pivot, target_span_world)

    _linearize_camera_segment(camera, f_start, f_end)


def dendrogram_scene_focus(nodes, cluster_geom):
    """BBox center of point spheres + internal merge levels (world coords) for camera aim."""
    xs, ys, zs = [], [], []
    for x, y in POINTS:
        xs.append(float(x))
        ys.append(float(y + DATA_Y_OFFSET))
        zs.append(float(DATA_Z))
    for nid, node in nodes.items():
        if len(node.members) < 2:
            continue
        g = cluster_geom[nid]
        cx, cy = g["center"]
        xs.append(float(cx))
        ys.append(float(cy + DATA_Y_OFFSET))
        zs.append(float(g["z"]))
    if not xs:
        return Vector((0.0, 0.0, 0.0))
    return Vector(
        (
            0.5 * (min(xs) + max(xs)),
            0.5 * (min(ys) + max(ys)),
            0.5 * (min(zs) + max(zs)),
        )
    )


def dendrogram_orbit_pivot(nodes, cluster_geom):
    """Vertical axis for camera yaw / top-down orbit: x,y = mean of dataset (world Y includes offset); z = vertical mid of data+dendrogram.

    Horizontal camera motion (constant z offset style paths) then circles this line; motion stays in planes
    parallel to the dataset (XY) when varying azimuth."""
    fc = dendrogram_scene_focus(nodes, cluster_geom)
    npt = len(POINTS)
    if npt == 0:
        return fc
    mx = sum(p[0] for p in POINTS) / npt
    my = sum(p[1] + DATA_Y_OFFSET for p in POINTS) / npt
    return Vector((mx, my, fc.z))


def create_threshold_label(col_guides):
    mat = make_emission_material(
        'ThresholdLabelMat', THRESHOLD_SLICE_COLOR, strength=2.0, alpha=0.0
    )
    bpy.ops.object.text_add(location=(0.0, 0.0, 0.0))
    txt = bpy.context.active_object
    txt.name = 'ThresholdLabel'
    cd = txt.data
    cd.body = ''
    cd.size = THRESHOLD_LABEL_SCALE
    cd.extrude = 0.012
    cd.align_x = 'LEFT'
    txt.hide_render = True
    txt.hide_viewport = True
    txt.data.materials.append(mat)
    for c in list(txt.users_collection):
        c.objects.unlink(txt)
    col_guides.objects.link(txt)
    txt.rotation_mode = 'QUATERNION'
    return txt


def _threshold_label_face_camera(obj, scene):
    """Billboard: curve text lies in local XY with +Z out of the plane; align +Z toward the active camera."""
    cam = scene.camera if scene else None
    if cam is None:
        return
    to_cam = cam.matrix_world.translation - obj.location
    if to_cam.length_squared < 1e-14:
        return
    to_cam.normalize()
    if obj.rotation_mode != 'QUATERNION':
        obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = to_cam.to_track_quat('Z', 'Y')


def install_threshold_label_handler(max_h, nodes, cluster_geom):
    """Live-updating label + Z position; survives scrub/render via frame_change_pre."""
    z_mmax = merge_levels_z_max(nodes, cluster_geom)
    z_top, z_bot = threshold_sweep_ends(
        max_h, z_top_floor=z_mmax + THRESHOLD_ABOVE_TOP_MERGE_Z
    )
    xs = [p[0] for p in POINTS]
    ys = [p[1] for p in POINTS]
    lx = min(xs) - 1.15
    ly = max(ys) + 0.95 + DATA_Y_OFFSET

    def dendrogram_threshold_label_handler(scene):
        obj = bpy.data.objects.get('ThresholdLabel')
        if obj is None:
            return
        f = scene.frame_current
        if F_THRESHOLD_SWEEP_START <= f <= F_THRESHOLD_TOP_HOLD_END:
            z = threshold_plane_z_for_sweep_frame(f, z_top, z_bot)
            th = height_from_plane_z(z, max_h)
            th = max(0.0, min(max_h * 1.001, th))
            obj.data.body = f"threshold = {th:.3f}"
            obj.location = (lx, ly, z + 0.08)
            _threshold_label_face_camera(obj, scene)
            obj.hide_viewport = False
            obj.hide_render = False
        else:
            obj.hide_viewport = True
            obj.hide_render = True

    dendrogram_threshold_label_handler._dendro_thr_label = True

    bpy.app.handlers.frame_change_pre[:] = [
        h
        for h in bpy.app.handlers.frame_change_pre
        if not getattr(h, '_dendro_thr_label', False)
    ]
    bpy.app.handlers.frame_change_pre.append(dendrogram_threshold_label_handler)


def _oval_scale_start(g):
    """Collapsed: thin chord via (a_col, b_col) scale on a circular base curve of radius g['radius']."""
    r_exp = max(float(g.get("radius", 1e-6)), 1e-9)
    a_col = float(g.get("a_col", r_exp * 0.25))
    b_col = float(g.get("b_col", r_exp * 0.25))
    s0x = min(1.0, a_col / r_exp)
    s0y = min(1.0, b_col / r_exp)
    return s0x, s0y


def animate_root_only_extension(circle_by_nid, root_id, cluster_geom, nodes):
    """Root merge: thin chord → full circle. Other merges stay collapsed until phase 2."""
    for nid, node in nodes.items():
        if len(node.members) == 1:
            continue
        obj = circle_by_nid.get(nid)
        if obj is None:
            continue
        g = cluster_geom[nid]
        s0x, s0y = _oval_scale_start(g)
        if nid == root_id:
            obj.scale = (s0x, s0y, 1.0)
            obj.keyframe_insert(data_path='scale', frame=F_EXTEND_START)
            obj.scale = (1.0, 1.0, 1.0)
            obj.keyframe_insert(data_path='scale', frame=F_EXTEND_END)
        else:
            obj.scale = (s0x, s0y, 1.0)
            obj.keyframe_insert(data_path='scale', frame=F_EXTEND_START)
            obj.keyframe_insert(data_path='scale', frame=F_EXTEND_END)


def animate_non_root_merges_extension(circle_by_nid, merge_ids, root_id, cluster_geom, nodes):
    """Stagger remaining merges (merge order) before the camera leaves the front view."""
    root_obj = circle_by_nid.get(root_id)
    if root_obj:
        root_obj.scale = (1.0, 1.0, 1.0)
        root_obj.keyframe_insert(data_path='scale', frame=F_EXTEND2_START)
        root_obj.keyframe_insert(data_path='scale', frame=F_EXTEND2_END)

    rest = [nid for nid in merge_ids if nid != root_id]
    n_rest = len(rest)
    span = max(1, F_EXTEND2_END - F_EXTEND2_START + 1)
    for k, nid in enumerate(rest):
        node = nodes[nid]
        if node.left is None:
            continue
        obj = circle_by_nid.get(nid)
        if obj is None:
            continue
        g = cluster_geom[nid]
        s0x, s0y = _oval_scale_start(g)
        t0 = F_EXTEND2_START + int((k / max(1, n_rest)) * (span * 0.82))
        t1 = min(F_EXTEND2_END, t0 + max(4, span // max(2, 2 * max(n_rest, 1))))
        obj.scale = (s0x, s0y, 1.0)
        obj.keyframe_insert(data_path='scale', frame=F_EXTEND2_START)
        obj.scale = (s0x, s0y, 1.0)
        obj.keyframe_insert(data_path='scale', frame=t0)
        obj.scale = (1.0, 1.0, 1.0)
        obj.keyframe_insert(data_path='scale', frame=t1)
        obj.scale = (1.0, 1.0, 1.0)
        obj.keyframe_insert(data_path='scale', frame=F_EXTEND2_END)


def animate_threshold_sweep(
    circle_meta,
    threshold_slice,
    threshold_mat,
    max_h,
    point_materials,
    root_id,
    nodes,
    cluster_geom,
    plane_xy_scale,
    threshold_wipe_z_pairs,
):
    """Threshold slice sweeps in Z at fixed XY scale; oval bevel pulses at cut; points recolor.
    Optional meshes named ``ThresholdPlane*`` (except ``threshold_slice``) get the same Z/scale keys but
    stay hidden in viewport/render so they never occlude the wire.
    Merge ovals and dendrogram verticals hide while their reference Z is above the slice; they
    return as the slice moves back up."""
    for mat in point_materials:
        if mat.node_tree.animation_data is None:
            mat.node_tree.animation_data_create()

    if threshold_slice.animation_data is None:
        threshold_slice.animation_data_create()

    for mat in point_materials:
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = POINT_WHITE
            bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=F_TOP - 1)
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = POINT_WHITE
                bsdf.inputs['Emission Color'].keyframe_insert(
                    data_path='default_value', frame=F_TOP - 1
                )

    z_mmax = merge_levels_z_max(nodes, cluster_geom)
    z_top, z_bot = threshold_sweep_ends(
        max_h, z_top_floor=z_mmax + THRESHOLD_ABOVE_TOP_MERGE_Z
    )
    ps = plane_xy_scale

    slice_xy = (float(threshold_slice.location.x), float(threshold_slice.location.y))
    threshold_plane_followers = [
        o
        for o in bpy.data.objects
        if o.type == 'MESH'
        and o.name.startswith('ThresholdPlane')
        and o is not threshold_slice
    ]
    for _fol in threshold_plane_followers:
        if _fol.animation_data is None:
            _fol.animation_data_create()

    def _sync_threshold_plane_followers(z_world, frame_insert):
        for fol in threshold_plane_followers:
            fol.location = (slice_xy[0], slice_xy[1], z_world)
            fol.scale = (ps, ps, 1.0)
            fol.keyframe_insert(data_path='location', frame=frame_insert)
            fol.keyframe_insert(data_path='scale', frame=frame_insert)

    if threshold_mat.node_tree.animation_data is None:
        threshold_mat.node_tree.animation_data_create()
    mix_node = threshold_mat.node_tree.nodes.get('Mix Shader')
    if mix_node:
        mix_node.inputs['Fac'].default_value = 0.0
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_TOP - 1)
        mix_node.inputs['Fac'].default_value = 0.0
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_TOP)
        mix_node.inputs['Fac'].default_value = 0.0
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_TOP_HOLD_END)
        mix_node.inputs['Fac'].default_value = 0.0
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_THRESHOLD_DIAG_VIEW)
        mix_node.inputs['Fac'].default_value = THRESHOLD_PLANE_MIX
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_THRESHOLD_SWEEP_START)
        mix_node.inputs['Fac'].default_value = THRESHOLD_PLANE_MIX
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_THRESHOLD_RETURN_END)
        mix_node.inputs['Fac'].default_value = THRESHOLD_PLANE_MIX
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_THRESHOLD_TOP_HOLD_END)
        mix_node.inputs['Fac'].default_value = 0.0
        mix_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=F_THRESHOLD_VFX_END)

    # Slice object + shader stay fully off through top-down, pause, and tilt; on only after diagonal orbit ends.
    threshold_slice.hide_viewport = True
    threshold_slice.hide_render = True
    threshold_slice.keyframe_insert(data_path='hide_viewport', frame=F_SCENE_START)
    threshold_slice.keyframe_insert(data_path='hide_render', frame=F_SCENE_START)
    for _hf in (F_TOP, F_TOP_HOLD_END, F_THRESHOLD_DIAG_VIEW):
        threshold_slice.hide_viewport = True
        threshold_slice.hide_render = True
        threshold_slice.keyframe_insert(data_path='hide_viewport', frame=_hf)
        threshold_slice.keyframe_insert(data_path='hide_render', frame=_hf)
    threshold_slice.hide_viewport = False
    threshold_slice.hide_render = False
    threshold_slice.keyframe_insert(data_path='hide_viewport', frame=F_THRESHOLD_SWEEP_START)
    threshold_slice.keyframe_insert(data_path='hide_render', frame=F_THRESHOLD_SWEEP_START)

    for fol in threshold_plane_followers:
        fol.hide_viewport = True
        fol.hide_render = True
        fol.keyframe_insert(data_path='hide_viewport', frame=F_SCENE_START)
        fol.keyframe_insert(data_path='hide_render', frame=F_SCENE_START)
        for _hf in (F_TOP, F_TOP_HOLD_END, F_THRESHOLD_DIAG_VIEW):
            fol.hide_viewport = True
            fol.hide_render = True
            fol.keyframe_insert(data_path='hide_viewport', frame=_hf)
            fol.keyframe_insert(data_path='hide_render', frame=_hf)
        fol.hide_viewport = True
        fol.hide_render = True
        fol.keyframe_insert(data_path='hide_viewport', frame=F_THRESHOLD_SWEEP_START)
        fol.keyframe_insert(data_path='hide_render', frame=F_THRESHOLD_SWEEP_START)

    # Slice: fixed XY for full sweep (scale from data bbox at build time)
    threshold_slice.location.z = z_top
    threshold_slice.scale = (ps, ps, 1.0)
    threshold_slice.keyframe_insert(data_path='location', frame=F_THRESHOLD_SWEEP_START - 1)
    threshold_slice.keyframe_insert(data_path='scale', frame=F_THRESHOLD_SWEEP_START - 1)
    _sync_threshold_plane_followers(z_top, F_THRESHOLD_SWEEP_START - 1)

    threshold_slice.location.z = z_top
    threshold_slice.scale = (ps, ps, 1.0)
    threshold_slice.keyframe_insert(data_path='location', frame=F_THRESHOLD_SWEEP_START)
    threshold_slice.keyframe_insert(data_path='scale', frame=F_THRESHOLD_SWEEP_START)
    _sync_threshold_plane_followers(z_top, F_THRESHOLD_SWEEP_START)

    threshold_slice.location.z = z_bot
    threshold_slice.scale = (ps, ps, 1.0)
    threshold_slice.keyframe_insert(data_path='location', frame=F_THRESHOLD_SWEEP_END)
    threshold_slice.keyframe_insert(data_path='scale', frame=F_THRESHOLD_SWEEP_END)
    _sync_threshold_plane_followers(z_bot, F_THRESHOLD_SWEEP_END)

    threshold_slice.location.z = z_top
    threshold_slice.scale = (ps, ps, 1.0)
    threshold_slice.keyframe_insert(data_path='location', frame=F_THRESHOLD_RETURN_END)
    threshold_slice.keyframe_insert(data_path='scale', frame=F_THRESHOLD_RETURN_END)
    _sync_threshold_plane_followers(z_top, F_THRESHOLD_RETURN_END)

    sigma = max_h * 0.12
    base_b = CIRCLE_BEVEL
    peak_b = CIRCLE_BEVEL * 6.0

    n_pts = len(point_materials)

    def set_point_colors_for_threshold(th):
        counts = leaf_cluster_member_counts(root_id, nodes, th, n_pts)
        labels = leaf_cluster_labels(root_id, nodes, th, [0])
        for li in range(n_pts):
            cid = labels.get(li, -1)
            merged = counts.get(cid, 0) >= 2
            col = CLUSTER_PURPLE if merged else POINT_WHITE
            mat = point_materials[li]
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].default_value = col
                if 'Emission Color' in bsdf.inputs:
                    bsdf.inputs['Emission Color'].default_value = col

    for item in circle_meta:
        curve = item["obj"].data
        if curve.animation_data is None:
            curve.animation_data_create()
        curve.bevel_depth = base_b
        curve.keyframe_insert(data_path='bevel_depth', frame=F_THRESHOLD_SWEEP_START - 1)

    vis_eps = 1e-4

    for obj, _ in threshold_wipe_z_pairs:
        if obj.animation_data is None:
            obj.animation_data_create()

    for obj, _ in threshold_wipe_z_pairs:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_viewport', frame=F_THRESHOLD_SWEEP_START - 1)
        obj.keyframe_insert(data_path='hide_render', frame=F_THRESHOLD_SWEEP_START - 1)

    sweep_frames = list(range(F_THRESHOLD_SWEEP_START, F_THRESHOLD_TOP_HOLD_END + 1))
    for f in sweep_frames:
        z = threshold_plane_z_for_sweep_frame(f, z_top, z_bot)

        threshold_slice.location.z = z
        threshold_slice.keyframe_insert(data_path='location', frame=f)
        _sync_threshold_plane_followers(z, f)

        for obj, z_ref in threshold_wipe_z_pairs:
            show = z_ref <= z + vis_eps
            obj.hide_viewport = not show
            obj.hide_render = not show
            obj.keyframe_insert(data_path='hide_viewport', frame=f)
            obj.keyframe_insert(data_path='hide_render', frame=f)

        th = height_from_plane_z(z, max_h)
        th = max(0.0, min(max_h * 1.001, th))

        for item in circle_meta:
            curve = item["obj"].data
            h_m = item["merge_h"]
            w = math.exp(-((h_m - th) / max(sigma, 1e-9)) ** 2)
            curve.bevel_depth = base_b + (peak_b - base_b) * w
            curve.keyframe_insert(data_path='bevel_depth', frame=f)

        set_point_colors_for_threshold(th)
        for mat in point_materials:
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=f)
                if 'Emission Color' in bsdf.inputs:
                    bsdf.inputs['Emission Color'].keyframe_insert(
                        data_path='default_value', frame=f
                    )

    for item in circle_meta:
        curve = item["obj"].data
        curve.bevel_depth = base_b
        curve.keyframe_insert(data_path='bevel_depth', frame=F_THRESHOLD_VFX_END)

    threshold_slice.hide_viewport = True
    threshold_slice.hide_render = True
    threshold_slice.keyframe_insert(data_path='hide_viewport', frame=F_THRESHOLD_VFX_END)
    threshold_slice.keyframe_insert(data_path='hide_render', frame=F_THRESHOLD_VFX_END)

    for fol in threshold_plane_followers:
        fol.hide_viewport = True
        fol.hide_render = True
        fol.keyframe_insert(data_path='hide_viewport', frame=F_THRESHOLD_VFX_END)
        fol.keyframe_insert(data_path='hide_render', frame=F_THRESHOLD_VFX_END)

    for obj, _ in threshold_wipe_z_pairs:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_viewport', frame=F_THRESHOLD_VFX_END)
        obj.keyframe_insert(data_path='hide_render', frame=F_THRESHOLD_VFX_END)

    for mat in point_materials:
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = POINT_WHITE
            bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=F_THRESHOLD_VFX_END)
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = POINT_WHITE
                bsdf.inputs['Emission Color'].keyframe_insert(
                    data_path='default_value', frame=F_THRESHOLD_VFX_END
                )

nodes, root_id, merge_ids = agglomerative(POINTS, linkage=LINKAGE)
max_h = max(nodes[mid].height for mid in merge_ids) if merge_ids else 1.0

cluster_geom = {}
for nid, node in nodes.items():
    pts = [POINTS[i] for i in node.members]
    if len(node.members) == 1:
        cx, cy = pts[0][0], pts[0][1]
        radius = 0.18
        z = DENDRO_BASE_Z
        cluster_geom[nid] = {
            "center": (float(cx), float(cy)),
            "radius": float(radius),
            "z": float(z),
        }
    else:
        lg = cluster_geom[node.left]
        rg = cluster_geom[node.right]
        lx, ly = lg["center"]
        rx, ry = rg["center"]
        mx = (lx + rx) * 0.5
        my = (ly + ry) * 0.5
        cx, cy = mx, my
        z = DENDRO_BASE_Z + (node.height / max_h) * HEIGHT_SCALE * 4.0

        L = Vector((lx, ly))
        R = Vector((rx, ry))
        w = R - L
        d_lr = w.length
        if d_lr < 1e-9:
            u = Vector((1.0, 0.0))
        else:
            u = w.normalized()
        pad = 0.25
        r_chord_half = 0.5 * d_lr
        r_hull = (
            max(math.hypot(POINTS[i][0] - cx, POINTS[i][1] - cy) for i in node.members)
            + pad
        )
        r_circ = max(r_chord_half, r_hull, 1e-4)
        theta = math.atan2(u.y, u.x)
        a_col = max(r_chord_half, 1e-4)
        b_col = OVAL_COLLAPSE_MINOR

        cluster_geom[nid] = {
            "center": (float(cx), float(cy)),
            "radius": float(r_circ),
            "z": float(z),
            "theta": float(theta),
            "a_col": float(a_col),
            "b_col": float(b_col),
        }

parent_of = {}
for nid, node in nodes.items():
    if node.left is not None:
        parent_of[node.left] = nid
        parent_of[node.right] = nid

# -----------------------------
# Scene collections
# -----------------------------
def ensure_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    master = bpy.context.scene.collection
    try:
        master.children.link(col)
    except RuntimeError:
        pass  # already under scene collection (e.g. partial re-run)
    return col

col_points = ensure_collection('Points')
col_dendro = ensure_collection('Dendrogram')
col_circles = ensure_collection('ClusterCircles')
col_guides = ensure_collection('Guides')

cam_pivot = dendrogram_orbit_pivot(nodes, cluster_geom)

# -----------------------------
# Ground plane
# -----------------------------
ground = add_plane('Ground', (0, 0, -0.02), GROUND_PLANE_SIZE, mat_ground)
for c in list(ground.users_collection):
    c.objects.unlink(ground)
col_guides.objects.link(ground)

sky_dome = add_dark_sky_dome('SkyDome', cam_pivot, SKY_DOME_RADIUS, mat_sky_dome)
for c in list(sky_dome.users_collection):
    c.objects.unlink(sky_dome)
col_guides.objects.link(sky_dome)

# -----------------------------
# Add point cloud
# -----------------------------
point_objects = []
point_materials = []
for i, (x, y) in enumerate(POINTS):
    pm = mat_point.copy()
    pm.name = f'PointMat_{i}'
    point_materials.append(pm)
    obj = add_uv_sphere(f'P{i}', (x, y + DATA_Y_OFFSET, DATA_Z), POINT_RADIUS, pm)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col_points.objects.link(obj)
    point_objects.append(obj)

keyframe_points_y_settle(point_objects, POINTS, F_SCENE_START, F_MAIN_START)

# -----------------------------
# Geometry-driven merge curves (thin chord → circle) + connector lines
# -----------------------------
dendro_objects = []
# (object, z_world): hide during threshold sweep while z_world > plane Z (re-shown as plane rises).
threshold_wipe_z_pairs = []
cluster_circle_objects = []
circle_meta = []
circle_by_nid = {}
dendro_curve_y_specs = []

for nid, node in nodes.items():
    if len(node.members) == 1:
        continue

    geom = cluster_geom[nid]
    cx, cy = geom["center"]
    z = geom["z"]

    if node.left is not None and node.right is not None:
        left_geom = cluster_geom[node.left]
        right_geom = cluster_geom[node.right]

        lx, ly = left_geom["center"]
        rx, ry = right_geom["center"]
        lz = left_geom["z"]
        rz = right_geom["z"]

        wy_l = ly + DATA_Y_OFFSET
        wy_r = ry + DATA_Y_OFFSET
        wy_p = cy + DATA_Y_OFFSET

        # vertical connector from left child up to parent height
        conn1 = new_curve_object(
            f'ConnL_{nid}',
            [(lx, wy_l, lz), (lx, wy_l, z)],
            bevel=CURVE_BEVEL * DENDRO_CURVE_BEVEL_SCALE,
            material=mat_dendro
        )

        # vertical connector from right child up to parent height
        conn2 = new_curve_object(
            f'ConnR_{nid}',
            [(rx, wy_r, rz), (rx, wy_r, z)],
            bevel=CURVE_BEVEL * DENDRO_CURVE_BEVEL_SCALE,
            material=mat_dendro
        )
        
        for obj in (conn1, conn2):
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            col_dendro.objects.link(obj)
            dendro_objects.append(obj)
            threshold_wipe_z_pairs.append((obj, float(z)))

        dendro_curve_y_specs.append(
            {'obj': conn1, 'x': lx, 'y_data': ly, 'z0': lz, 'z1': z}
        )
        dendro_curve_y_specs.append(
            {'obj': conn2, 'x': rx, 'y_data': ry, 'z0': rz, 'z1': z}
        )

        pid = parent_of.get(nid)
        if pid is not None:
            z_parent = cluster_geom[pid]["z"]
            conn_up = new_curve_object(
                f'ConnUp_{nid}',
                [(cx, wy_p, z), (cx, wy_p, z_parent)],
                bevel=CURVE_BEVEL * DENDRO_CURVE_BEVEL_SCALE,
                material=mat_dendro,
            )
            for c in list(conn_up.users_collection):
                c.objects.unlink(conn_up)
            col_dendro.objects.link(conn_up)
            dendro_objects.append(conn_up)
            threshold_wipe_z_pairs.append((conn_up, float(z_parent)))
            dendro_curve_y_specs.append(
                {'obj': conn_up, 'x': cx, 'y_data': cy, 'z0': z, 'z1': z_parent}
            )

    if not merge_qualifies_for_circle(node, nodes):
        continue

    color = CLUSTER_PURPLE

    r_circ = float(geom["radius"])
    theta = float(geom["theta"])
    a_col = float(geom["a_col"])
    b_col = float(geom["b_col"])

    circle_mat = make_emission_material(
        f'OvalMat_{nid}',
        color,
        strength=2.0,
        alpha=0.42
    )

    circle_obj = new_curve_object(
        f'Oval_{nid}',
        ellipse_points_local(r_circ, r_circ, 0.0, n=100),
        bevel=CIRCLE_BEVEL,
        material=circle_mat,
        cyclic=True
    )

    circle_obj.location = (cx, cy + DATA_Y_OFFSET, z)
    circle_obj.rotation_euler = (0.0, 0.0, theta)
    s0x = min(1.0, a_col / max(r_circ, 1e-9))
    s0y = min(1.0, b_col / max(r_circ, 1e-9))
    circle_obj.scale = (s0x, s0y, 1.0)

    for c in list(circle_obj.users_collection):
        c.objects.unlink(circle_obj)
    col_circles.objects.link(circle_obj)
    cluster_circle_objects.append((circle_obj, node.height, nid))
    circle_by_nid[nid] = circle_obj

    circle_meta.append({
        "nid": nid,
        "obj": circle_obj,
        "center": (cx, cy),
        "lx": lx,
        "ly": ly,
        "rx": rx,
        "ry": ry,
        "theta": float(geom["theta"]),
        "z": z,
        "radius": r_circ,
        "merge_h": node.height,
        "s_collapsed": min(s0x, s0y),
        "s0x": s0x,
        "s0y": s0y,
    })

for _cm in circle_meta:
    threshold_wipe_z_pairs.append((_cm["obj"], float(_cm["z"])))

keyframe_dendro_connectors_y_settle(dendro_curve_y_specs, F_SCENE_START, F_MAIN_START)
keyframe_merge_ovals_follow_vertical_tips(circle_meta, F_SCENE_START, F_MAIN_START)

# Threshold slice: translucent plane; Z/hide keyed in animate_threshold_sweep.
for _tw in list(bpy.data.objects):
    if _tw.type == 'MESH' and _tw.name.startswith('ThresholdWire'):
        bpy.data.objects.remove(_tw, do_unlink=True)
for _tp in list(bpy.data.objects):
    if _tp.type == 'MESH' and _tp.name.startswith('ThresholdPlane'):
        bpy.data.objects.remove(_tp, do_unlink=True)

_z_mmax_slice = merge_levels_z_max(nodes, cluster_geom)
_threshold_z_top_build, _ = threshold_sweep_ends(
    max_h, z_top_floor=_z_mmax_slice + THRESHOLD_ABOVE_TOP_MERGE_Z
)
_th_plane_xy, _th_cx, _th_cy = threshold_slice_xy_scale_and_center(POINTS, PLANE_SIZE)
threshold_wire = add_plane(
    'ThresholdWire',
    (_th_cx, _th_cy, _threshold_z_top_build),
    PLANE_SIZE,
    mat_threshold,
)
threshold_wire.scale = (_th_plane_xy, _th_plane_xy, 1.0)
for c in list(threshold_wire.users_collection):
    c.objects.unlink(threshold_wire)
col_guides.objects.link(threshold_wire)

create_threshold_label(col_guides)

# -----------------------------
# Camera
# -----------------------------
bpy.ops.object.camera_add(location=(0, -18, 0), rotation=(math.radians(90), 0, 0))
camera = bpy.context.active_object
camera.name = 'MainCamera'
scene.camera = camera
camera.data.type = 'PERSP'
# Target world span at the pivot for lens math (same numbers you used for ortho_scale before).
FRONT_ORTHO = 17.0
ORBIT_ORTHO_SCALE = 17.0
ORBIT_CAM_DIST = 17.0
ORBIT_ELEV_DEG = 35.0

# All positions are offsets from the orbit pivot: mean (x,y) of points, z = vertical mid of scene.
front_loc = cam_pivot + Vector((0.0, -ORBIT_CAM_DIST, 0.0))
_front_e = camera_euler_look_at(front_loc, cam_pivot)
front_rot = (float(_front_e.x), float(_front_e.y), float(_front_e.z))

_el_rad = math.radians(ORBIT_ELEV_DEG)
elev_loc = cam_pivot + Vector(
    (0.0, -ORBIT_CAM_DIST * math.cos(_el_rad), ORBIT_CAM_DIST * math.sin(_el_rad))
)
_elev_e = camera_euler_look_at(elev_loc, cam_pivot)
elev_rot = (float(_elev_e.x), float(_elev_e.y), float(_elev_e.z))

# Top-down: camera above pivot; rotation aims at pivot (not identity euler).
TOP_CAMERA_Z_OFFSET = 20.0
top_loc = cam_pivot + Vector((0.0, 0.0, TOP_CAMERA_Z_OFFSET))
_top_e = camera_euler_look_at(top_loc, cam_pivot)
top_rot = (float(_top_e.x), float(_top_e.y), float(_top_e.z))

# Threshold approach: orbit around world +X through pivot (arm along +Z, tilt in YZ plane).
THRESHOLD_ORBIT_RX_END_DEG = 75.0
THRESHOLD_CAM_DISTANCE_SCALE = 0.8  # 1 = no radial change; >1 pulls back slightly at end of tilt
_rx_end = math.radians(THRESHOLD_ORBIT_RX_END_DEG)
_H = float((top_loc - cam_pivot).z)
_arm_end = mathutils.Matrix.Rotation(_rx_end, 3, 'X') @ Vector((0.0, 0.0, _H))
diag_threshold_loc = cam_pivot + _arm_end * THRESHOLD_CAM_DISTANCE_SCALE
_diag_e = camera_euler_look_at(diag_threshold_loc, cam_pivot)
diag_threshold_rot = (float(_diag_e.x), float(_diag_e.y), float(_diag_e.z))

# Hold front view while points settle in Y; then elevate ~60°, yaw, expand, glide, etc.
camera.location = front_loc
camera.rotation_euler = front_rot
_commit_persp_camera_keyframe(camera, F_SCENE_START, cam_pivot, FRONT_ORTHO)
if F_MAIN_START > F_SCENE_START:
    _commit_persp_camera_keyframe(camera, F_MAIN_START - 1, cam_pivot, FRONT_ORTHO)

keyframe_camera_persp_elevate(
    camera,
    cam_pivot,
    front_loc,
    elev_loc,
    ORBIT_ORTHO_SCALE,
    F_MAIN_START,
    F_ELEV_END,
)
keyframe_camera_persp_yaw_orbit(
    camera,
    cam_pivot,
    ORBIT_CAM_DIST,
    ORBIT_ELEV_DEG,
    ORBIT_ORTHO_SCALE,
    F_YAW_ORBIT_START,
    F_YAW_ORBIT_END,
)

camera.location = elev_loc
camera.rotation_euler = elev_rot
for _hold_f in (F_EXTEND_START, F_EXTEND_END, F_EXTEND2_END):
    _commit_persp_camera_keyframe(camera, _hold_f, cam_pivot, ORBIT_ORTHO_SCALE)

keyframe_camera_persp_glide(
    camera,
    elev_loc,
    elev_rot,
    top_loc,
    top_rot,
    cam_pivot,
    ORBIT_ORTHO_SCALE,
    F_TO_TOP_START,
    F_TOP,
)

# Top → threshold: +X orbit through pivot, look-at pivot.
keyframe_camera_orbit_rx_yz_arc(
    camera,
    cam_pivot,
    top_loc,
    _rx_end,
    THRESHOLD_CAM_DISTANCE_SCALE,
    ORBIT_ORTHO_SCALE,
    F_TOP,
    F_THRESHOLD_DIAG_VIEW,
)

for _diag_hold_f in (F_THRESHOLD_SWEEP_END, F_THRESHOLD_RETURN_END):
    camera.location = diag_threshold_loc
    camera.rotation_euler = diag_threshold_rot
    _commit_persp_camera_keyframe(camera, _diag_hold_f, cam_pivot, ORBIT_ORTHO_SCALE)

# Threshold diagonal → top: reverse the same +X orbit (full return to top view).
keyframe_camera_orbit_rx_yz_arc(
    camera,
    cam_pivot,
    top_loc,
    _rx_end,
    THRESHOLD_CAM_DISTANCE_SCALE,
    ORBIT_ORTHO_SCALE,
    F_THRESHOLD_VFX_END,
    F_BACK_AT_TOP,
    reverse=True,
)

# Top → starting side: same duration as intro glide.
keyframe_camera_persp_glide(
    camera,
    top_loc,
    top_rot,
    front_loc,
    front_rot,
    cam_pivot,
    FRONT_ORTHO,
    F_BACK_AT_TOP,
    F_FINAL_GLIDE_END,
)

# Return front (match opening side view).
camera.location = front_loc
camera.rotation_euler = front_rot
_commit_persp_camera_keyframe(camera, F_RETURN_FRONT, cam_pivot, FRONT_ORTHO)

animate_root_only_extension(circle_by_nid, root_id, cluster_geom, nodes)
animate_non_root_merges_extension(
    circle_by_nid, merge_ids, root_id, cluster_geom, nodes
)
animate_threshold_sweep(
    circle_meta,
    threshold_wire,
    mat_threshold,
    max_h,
    point_materials,
    root_id,
    nodes,
    cluster_geom,
    _th_plane_xy,
    threshold_wipe_z_pairs,
)
keyframe_dendro_hide_for_top_view_phases(dendro_objects)
install_threshold_label_handler(max_h, nodes, cluster_geom)

# -----------------------------
# Force viewport to active camera view
# -----------------------------
def switch_viewport_to_camera():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    space.lock_camera = False
                    break

# Full timeline from thin merge chords → circles → camera moves; viewport shows active camera (perspective).
scene.frame_start = F_SCENE_START
scene.frame_set(F_SCENE_START)
switch_viewport_to_camera()

for obj in bpy.data.objects:
    if obj.type in {'EMPTY', 'CAMERA'}:
        obj.hide_viewport = True
        
ground.hide_render = True

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.overlay.show_relationship_lines = False
