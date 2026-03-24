"""
Multiscale dendrogram: K**4 points (K=5 → 625) in XY; top-down zoom-out through three scales, then a
tilt down one YZ meridian (fixed world X) to 45° for the layered dendrogram reveal, pull-out on the
same meridian (still 45°), one full CCW dome orbit at that latitude, meridian tilt to a 90° side view,
then a second full CCW orbit — end.

Geometry matches ``dendrogram.py``: vertical stems and straight merge chords (hidden until the reveal phase).

Single linkage: sorted-edge builder (O(n² log n)).

Run from a fresh Blender file (Scripting → Run).
"""
import bpy
import colorsys
import math
import mathutils
import random
from mathutils import Vector

# -----------------------------------------------------------------------------
# Hierarchy: K**4 leaves — mega ⊃ super ⊃ micro ⊃ point (each level has K children).
# -----------------------------------------------------------------------------
K = 5
# Nested rings in XY: R_LEAF ≪ R_MICRO ≪ R_SUPER ≪ R_MEGA (four single-linkage gap scales).
# Smaller radii = tighter packing; keep the same ordering so linkage stays clean.
R_LEAF = 0.05
R_MICRO = 0.16
R_SUPER = .5
R_MEGA = 1.5

LINKAGE = 'single'
POINT_RADIUS = 0.048
CURVE_BEVEL = 0.02
DENDRO_CURVE_BEVEL_SCALE = 1.35
# Vertical stems: uniform thin bevel for the bottom 90% of merge height (low Z); linear ramp to
# full thickness in the top 10% (near root). BOTTOM_FRAC is stem radius as a fraction of full bevel.
DENDRO_STEM_BEVEL_UNIFORM_HEIGHT_FRAC = 0.9
DENDRO_STEM_BEVEL_BOTTOM_FRAC = 0.035
# Horizontal merge segments (straight chords between vertical stems at each merge height).
MERGE_CHORD_BEVEL = 0.0048
MERGE_CHORD_EMIT_STRENGTH = 1.55
MERGE_CHORD_ALPHA = 0.52

HEIGHT_SCALE = 1.05
DATA_Y_OFFSET = 0.0
DATA_Z = 0.0
DENDRO_Z_LIFT = 0.0
DENDRO_BASE_Z = DATA_Z + POINT_RADIUS + DENDRO_Z_LIFT
GROUND_PLANE_SIZE = 360.0
SKY_DOME_RADIUS = 620.0

# Side view: camera offset from target (mostly −Y, slight +Z) — classic dendrogram read.
CAM_VIEW_DIR = Vector((0.1, -1.0, 0.42)).normalized()

# Top view framing: higher dist_mul / span_mul = camera higher + wider lens (more zoomed out).
TOP_ZOOM_DIST_MUL_1 = 2.08
TOP_ZOOM_SPAN_MUL_1 = 1.22
TOP_ZOOM_DIST_MUL_2 = 1.8
TOP_ZOOM_SPAN_MUL_2 = 1.1
TOP_ZOOM_DIST_MUL_3 = 2.14
TOP_ZOOM_SPAN_MUL_3 = 1.22
# Reference side pose (used to build 45° tilt direction from level-3 top).
SIDE_ZOOM_DIST_MUL = 2.8
SIDE_ZOOM_SPAN_MUL = 1.1

# Post level-3: YZ meridian tilt → dendrogram → meridian pull-out at 45° → dome orbit → tilt to side → second orbit.
# World +Y is scene “north”; side camera uses mostly −Y (``CAM_VIEW_DIR``). More negative = farther back on −Y.
# Distance + lens zoom-out after level 3: use ``DENDRO_TOP_ZOOM_*`` only (tilt starts at baseline; ramp during reveal).

# Point scale multipliers (mesh radius = POINT_RADIUS × scale): ramp up on each top-view zoom-out.
# L2/L3 are larger so 125 / 625 points stay visible at wide framing.
POINT_SCALE_L1 = 0.2
POINT_SCALE_L2 = 0.3
POINT_SCALE_L3 = 0.4

# Timeline: … → L3 top → meridian tilt to 45° → dendrogram → pull back on meridian (still 45°) → dome orbit → side tilt → dome orbit.
F_START = 1
SETTLE_FRAMES = 0
HOLD_L1_TOP = 10
ZOOM_1_2 = 20
HOLD_L2_TOP = 10
ZOOM_2_3 = 10
HOLD_L3_TOP = 20
# During L2→L3 zoom, blend view direction from straight-down toward the post-L3 tilt ray (0 = pure top).
ZOOM_2_3_TILT_MIX = 0.16
TILT_FROM_TOP_DEG = 60.0
TILT_FROM_TOP_FRAMES = 20
DENDRO_REVEAL_LAYERS = 31
DENDRO_REVEAL_FRAMES_PER_LAYER = 6
# Last N reveal layers: meridian pull-out at fixed 45° + wider FOV; same multipliers set dome orbit radius + lens span.
DENDRO_TOP_ZOOM_LAYERS = 3
DENDRO_TOP_ZOOM_DISTANCE_MUL = 1.3
DENDRO_TOP_ZOOM_LENS_SPAN_MUL = 1.3
_DENDRO_TOP_ZOOM_LAYERS_USE = max(0, min(DENDRO_TOP_ZOOM_LAYERS, DENDRO_REVEAL_LAYERS))
# Parallel shift along camera local +Y: final pull-out zoom + first dome orbit; ramps out before side tilt.
DENDRO_VIEW_PARALLEL_SHIFT = 1.15
DENDRO_TOP_ZOOM_SHIFT_RAMP_IN_FRAMES = 10
DENDRO_DOME_SHIFT_RAMP_OUT_FRAMES = 22
# One full CCW orbit (seen from +Z) on the dome at the tilt angle, after pull-out.
DOME_YAW_ORBIT_FRAMES = 120
# After the first orbit: meridian arc to horizontal side view (90° from +Z), then second full CCW orbit.
TILT_45_TO_SIDE_FRAMES = 26
DOME_SIDE_YAW_ORBIT_FRAMES = 120
HOLD_FINAL_SIDE = 8
# Shift-lens along camera +Y during side view so the look target sits mid–lower frame.
SIDE_VIEW_PARALLEL_SHIFT = 1.12

F_SETTLE_LAST = F_START + SETTLE_FRAMES
F_L1_START = F_SETTLE_LAST
F_L1_END = F_L1_START + HOLD_L1_TOP - 1
F_Z12_START = F_L1_END + 1
F_Z12_END = F_Z12_START + ZOOM_1_2 - 1
F_L2_END = F_Z12_END + HOLD_L2_TOP
F_Z23_START = F_L2_END + 1
F_Z23_END = F_Z23_START + ZOOM_2_3 - 1
F_L3_TOP_END = F_Z23_END + HOLD_L3_TOP
F_TILT45_START = F_L3_TOP_END + 1
F_TILT45_END = F_TILT45_START + TILT_FROM_TOP_FRAMES - 1
F_DENDRO_REVEAL_START = F_TILT45_END + 1
F_DENDRO_REVEAL_END = (
    F_DENDRO_REVEAL_START + DENDRO_REVEAL_LAYERS * DENDRO_REVEAL_FRAMES_PER_LAYER - 1
)
F_DENDRO_TOP_ZOOM_START = F_DENDRO_REVEAL_START + (
    DENDRO_REVEAL_LAYERS - _DENDRO_TOP_ZOOM_LAYERS_USE
) * DENDRO_REVEAL_FRAMES_PER_LAYER
F_DOME_ORBIT_START = F_DENDRO_REVEAL_END + 1
F_DOME_ORBIT_END = F_DOME_ORBIT_START + DOME_YAW_ORBIT_FRAMES - 1
F_DENDRO_SHIFT_RAMP_OUT_START = max(
    F_DOME_ORBIT_START,
    F_DOME_ORBIT_END - DENDRO_DOME_SHIFT_RAMP_OUT_FRAMES + 1,
)
F_SIDE_TILT_START = F_DOME_ORBIT_END + 1
F_SIDE_TILT_END = F_SIDE_TILT_START + TILT_45_TO_SIDE_FRAMES - 1
F_SIDE_ORBIT_START = F_SIDE_TILT_END + 1
F_SIDE_ORBIT_END = F_SIDE_ORBIT_START + DOME_SIDE_YAW_ORBIT_FRAMES - 1
F_END = F_SIDE_ORBIT_END + HOLD_FINAL_SIDE
# First dendrogram pieces appear with layered reveal (after 45° tilt).
F_DENDRO_SHOW = F_DENDRO_REVEAL_START

F_CAP_L2 = F_Z12_START
F_CAP_L3 = F_Z23_START
F_CAP_SIDE = F_DENDRO_SHOW

# -----------------------------------------------------------------------------
# Cleanup (match dendrogram.py)
# -----------------------------------------------------------------------------
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
scene.frame_start = F_START
scene.frame_end = F_END
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.eevee.use_bloom = True
scene.eevee.taa_render_samples = 24
scene.eevee.taa_samples = 24
scene.render.film_transparent = False
if hasattr(scene.render, 'use_persistent_data'):
    scene.render.use_persistent_data = True
scene.use_nodes = False
scene.view_settings.exposure = 0.0

RENDER_WORLD_BG_RGB = (0.0, 0.0, 0.0)


def configure_dark_render_world(target_scene, rgb=RENDER_WORLD_BG_RGB):
    w = bpy.data.worlds.new('MultiScaleWorld')
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

POINT_WHITE = (1.0, 1.0, 1.0, 1.0)
DENDRO_VERTICAL_COLOR = (0.62, 0.78, 0.98, 1.0)
CAPTION_COLOR = (0.85, 0.9, 0.98, 1.0)
CLUSTER_PURPLE = (0.58, 0.32, 0.88, 1.0)


def make_emission_material(name, color=(1, 1, 1, 1), strength=2.0, alpha=1.0):
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


def make_principled_material(name, color=(1, 1, 1, 1), roughness=0.35, alpha=1.0, emission_strength=0.0):
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
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0, location=(float(center.x), float(center.y), float(center.z))
    )
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


# -----------------------------------------------------------------------------
# Agglomerative clustering (same as dendrogram.py)
# -----------------------------------------------------------------------------
class ClusterNode:
    def __init__(self, node_id, members, left=None, right=None, height=0.0):
        self.id = node_id
        self.members = members
        self.left = left
        self.right = right
        self.height = height


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
    """Naive O(n³) agglomerative path (fine for small n)."""
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


def agglomerative_single_linkage_sorted_edges(points):
    """Same tree as standard single linkage; Kruskal on complete graph (ties: lower leaf indices)."""
    n = len(points)
    edges = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            edges.append((math.dist(pi, points[j]), i, j))
    edges.sort(key=lambda t: (t[0], t[1], t[2]))

    nodes = {i: ClusterNode(i, [i], height=0.0) for i in range(n)}
    merges = []
    root_of_leaf = list(range(n))
    next_id = n

    for d, i, j in edges:
        ri = root_of_leaf[i]
        rj = root_of_leaf[j]
        if ri == rj:
            continue
        if ri > rj:
            ri, rj = rj, ri
        left_id, right_id = ri, rj
        mem = nodes[left_id].members + nodes[right_id].members
        new_node = ClusterNode(next_id, mem, left=left_id, right=right_id, height=d)
        nodes[next_id] = new_node
        merges.append(next_id)
        for leaf in mem:
            root_of_leaf[leaf] = next_id
        next_id += 1

    root_id = merges[-1] if merges else 0
    return nodes, root_id, merges


def merge_qualifies_for_chord(node, nodes):
    if node.left is None or node.right is None:
        return False
    if len(node.members) < 2:
        return False
    return node.height > 1e-12


def camera_euler_look_at(cam_pos, target, up_axis='Y'):
    d = target - cam_pos
    if d.length < 1e-8:
        return mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
    return d.to_track_quat('-Z', up_axis).to_euler('XYZ')


def camera_quaternion_look_at(cam_pos, target, up_axis='Y'):
    """Blender camera −Z views toward ``target``; Gram–Schmidt from a world up vector fixes roll.

    Use ``up_axis='Z'`` when the scene vertical is world +Z (XY horizontal): on a dome orbit at fixed
    polar angle, the view stays on the target and the tilt vs the XY plane stays constant without banking.
    Use ``'Y'`` when you want image up tied to world +Y (e.g. some top-down framings).
    """
    z_axis = (Vector(cam_pos) - Vector(target)).normalized()
    if z_axis.length < 1e-8:
        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    if up_axis == 'Y':
        world_up = Vector((0.0, 1.0, 0.0))
    elif up_axis == 'Z':
        world_up = Vector((0.0, 0.0, 1.0))
    else:
        world_up = Vector((1.0, 0.0, 0.0))
    x_axis = world_up.cross(z_axis)
    if x_axis.length < 1e-10:
        alt = Vector((1.0, 0.0, 0.0))
        if abs(z_axis.dot(alt)) > 0.95:
            alt = Vector((0.0, 0.0, 1.0))
        x_axis = alt.cross(z_axis)
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis)
    y_axis.normalize()
    x_axis = y_axis.cross(z_axis)
    x_axis.normalize()
    m3 = mathutils.Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x),
            (x_axis.y, y_axis.y, z_axis.y),
            (x_axis.z, y_axis.z, z_axis.z),
        )
    )
    q = m3.to_quaternion()
    q.normalize()
    return q


def _slerp_unit_vec(v0, v1, t):
    """Spherical lerp between two directions; ``t`` in [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.0:
        return Vector(v0).normalized()
    if t >= 1.0:
        return Vector(v1).normalized()
    v0n = Vector(v0).normalized()
    v1n = Vector(v1).normalized()
    dp = max(-1.0, min(1.0, v0n.dot(v1n)))
    omega = math.acos(dp)
    if omega < 1e-6:
        return v0n.lerp(v1n, t).normalized()
    so = math.sin(omega)
    w0 = math.sin((1.0 - t) * omega) / so
    w1 = math.sin(t * omega) / so
    out = v0n * w0 + v1n * w1
    out.normalize()
    return out


def meridian_cam_offset_yz(theta_from_plus_z_rad, radius):
    """Camera offset from look-at in world YZ only (ΔX = 0): θ=0 is +Z above; θ=π/2 is horizontal −Y."""
    s = math.sin(float(theta_from_plus_z_rad))
    c = math.cos(float(theta_from_plus_z_rad))
    r = float(radius)
    return Vector((0.0, -s * r, c * r))


def meridian_cam_dir_yz(theta_from_plus_z_rad):
    """Unit direction (look-at → camera) on the same YZ meridian as ``meridian_cam_offset_yz``."""
    return meridian_cam_offset_yz(theta_from_plus_z_rad, 1.0).normalized()


# Azimuth (radians) of the YZ meridian toward −Y: matches ``meridian_cam_offset_yz`` at this φ.
_DOME_PHI_MERIDIAN_MINUS_Y = -0.5 * math.pi


def dome_cam_offset_from_look(theta_from_plus_z_rad, azimuth_rad, radius):
    """Offset from look-at on a sphere: θ = polar angle from +Z; φ = azimuth in XY from +X (CCW from +Z)."""
    th = float(theta_from_plus_z_rad)
    ph = float(azimuth_rad)
    r = float(radius)
    st = math.sin(th)
    return Vector((st * math.cos(ph) * r, st * math.sin(ph) * r, math.cos(th) * r))


def _quaternion_align_previous(q_new, q_prev):
    """Keep quaternion in same hemisphere as previous key (±q are same rotation; wrong sign = visible snap)."""
    if q_prev is None:
        return mathutils.Quaternion(q_new)
    qn = mathutils.Quaternion(q_new)
    if qn.dot(q_prev) < 0.0:
        qn = -qn
    return qn


def _camera_parallel_shift_offset(quat, strength):
    """World offset along camera local +Y; same rotation = optical axis unchanged (shift lens)."""
    if abs(float(strength)) < 1e-12:
        return Vector((0.0, 0.0, 0.0))
    up = quat.to_matrix() @ Vector((0.0, 1.0, 0.0))
    if up.length < 1e-10:
        return Vector((0.0, 0.0, 0.0))
    up.normalize()
    return float(strength) * up


def _dendro_shift_scale_for_frame(f):
    """Pull-out zoom: ease in to 1. First dome orbit: hold 1, ease out at end before side tilt."""
    f = int(f)
    z0, z1 = int(F_DENDRO_TOP_ZOOM_START), int(F_DENDRO_REVEAL_END)
    if z0 <= f <= z1:
        n = z1 - z0 + 1
        rin = max(1, int(DENDRO_TOP_ZOOM_SHIFT_RAMP_IN_FRAMES))
        eff_rin = min(rin, max(1, n))
        if eff_rin <= 1:
            return 1.0
        if f < z0 + eff_rin:
            u = (f - z0) / float(eff_rin - 1)
            return _smootherstep01(u)
        return 1.0
    d0, d1 = int(F_DOME_ORBIT_START), int(F_DOME_ORBIT_END)
    if f < d0 or f > d1:
        return 0.0
    rs = int(F_DENDRO_SHIFT_RAMP_OUT_START)
    if f < rs:
        return 1.0
    span = max(1, d1 - rs)
    u = (f - rs) / float(span)
    return 1.0 - _smootherstep01(u)


def _camera_use_quaternion_rotation(cam):
    if cam.rotation_mode != 'QUATERNION':
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.rotation_euler.to_quaternion()


def perspective_lens_mm_match_span(cam_data, target_span_world, subject_distance_world):
    """Focal length (mm) so perspective span at subject distance matches target (AUTO sensor)."""
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


def _smoothstep01(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _smootherstep01(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _linearize_action_range(action, f_start, f_end):
    if action is None:
        return
    for fc in action.fcurves:
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


def _ensure_camera_actions(camera_obj):
    """Same as ``dendrogram.py``: lens/type keys live on ``camera_obj.data`` actions too."""
    if camera_obj.animation_data is None:
        camera_obj.animation_data_create()
    if camera_obj.data.animation_data is None:
        camera_obj.data.animation_data_create()


def _linearize_camera_object_and_data(camera_obj, f_start, f_end):
    for ad in (camera_obj.animation_data, camera_obj.data.animation_data):
        if ad is not None and ad.action is not None:
            _linearize_action_range(ad.action, f_start, f_end)


def _view3d_temp_override_context(target_scene):
    """VIEW_3D window/area/region so ``bpy.ops.object.camera_add`` works from Text Editor."""
    wm = bpy.context.window_manager
    if wm is None:
        return None
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = None
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    region = reg
                    break
            if region is None:
                continue
            vl = target_scene.view_layers[0] if target_scene.view_layers else None
            ctx = {
                'window': window,
                'screen': window.screen,
                'area': area,
                'region': region,
                'scene': target_scene,
            }
            if vl is not None:
                ctx['view_layer'] = vl
            return ctx
    return None


def create_main_camera_like_dendrogram(target_scene, location, rotation_euler):
    """Prefer ``bpy.ops.object.camera_add`` (``dendrogram.py``); fallback to data API if no 3D view."""
    loc = (float(location[0]), float(location[1]), float(location[2]))
    rot = (
        float(rotation_euler[0]),
        float(rotation_euler[1]),
        float(rotation_euler[2]),
    )
    override = _view3d_temp_override_context(target_scene)
    cam = None
    if override is not None:
        try:
            with bpy.context.temp_override(**override):
                bpy.ops.object.camera_add(location=loc, rotation=rot)
            cam = bpy.context.active_object
        except (RuntimeError, AttributeError):
            cam = None
    if cam is None or cam.type != 'CAMERA':
        cam_data = bpy.data.cameras.new('MainCameraData')
        cam = bpy.data.objects.new('MainCamera', cam_data)
        cam.location = loc
        cam.rotation_euler = rot
        target_scene.collection.objects.link(cam)
    cam.name = 'MainCamera'
    target_scene.camera = cam
    cam.data.type = 'PERSP'
    return cam


# -----------------------------------------------------------------------------
# Build hierarchical point cloud in XY: mega → super → micro → leaf (nested rings + jitter)
# Indices: LEAF_MEGA (a), LEAF_SUPER (b), LEAF_MICRO (c); inner leaf index d ∈ 0..K-1 implicit in order.
# -----------------------------------------------------------------------------
def build_hierarchical_points_xy_4(k, r_leaf, r_micro, r_super, r_mega, seed=31415):
    random.seed(seed)
    pts = []
    leaf_mega = []
    leaf_super = []
    leaf_micro = []

    def ang(t):
        return 2.0 * math.pi * float(t) / float(k)

    mega_centers = []
    for a in range(k):
        ta = ang(a) + random.uniform(-0.025, 0.025)
        mega_centers.append(
            (r_mega * 1.1 * math.cos(ta), r_mega * 0.9 * math.sin(ta))
        )

    for a in range(k):
        mx, my = mega_centers[a]
        for b in range(k):
            tb = ang(b) + random.uniform(-0.04, 0.04)
            sx = mx + r_super * math.cos(tb)
            sy = my + r_super * math.sin(tb)
            for c in range(k):
                tc = ang(c) + random.uniform(-0.05, 0.05)
                ux = sx + r_micro * math.cos(tc)
                uy = sy + r_micro * math.sin(tc)
                for d in range(k):
                    td = ang(d) + random.uniform(-0.055, 0.055)
                    jx = random.uniform(-0.005, 0.005)
                    jy = random.uniform(-0.005, 0.005)
                    x = ux + r_leaf * math.cos(td) + jx
                    y = uy + r_leaf * math.sin(td) + jy
                    pts.append((x, y))
                    leaf_mega.append(a)
                    leaf_super.append(b)
                    leaf_micro.append(c)
    return pts, leaf_mega, leaf_super, leaf_micro


POINTS, LEAF_MEGA, LEAF_SUPER, LEAF_MICRO = build_hierarchical_points_xy_4(
    K, R_LEAF, R_MICRO, R_SUPER, R_MEGA
)

nodes, root_id, merge_ids = agglomerative_single_linkage_sorted_edges(POINTS)
max_h = max(nodes[mid].height for mid in merge_ids) if merge_ids else 1.0

cluster_geom = {}
for nid, node in nodes.items():
    pts = [POINTS[i] for i in node.members]
    if len(node.members) == 1:
        cx, cy = pts[0][0], pts[0][1]
        radius = 0.11
        z = DENDRO_BASE_Z
        cluster_geom[nid] = {
            'center': (float(cx), float(cy)),
            'radius': float(radius),
            'z': float(z),
        }
    else:
        lg = cluster_geom[node.left]
        rg = cluster_geom[node.right]
        lx, ly = lg['center']
        rx, ry = rg['center']
        mx = (lx + rx) * 0.5
        my = (ly + ry) * 0.5
        cx, cy = mx, my
        z = DENDRO_BASE_Z + (node.height / max_h) * HEIGHT_SCALE * 4.0
        cluster_geom[nid] = {
            'center': (float(cx), float(cy)),
            'z': float(z),
        }

parent_of = {}
for nid, node in nodes.items():
    if node.left is not None:
        parent_of[node.left] = nid
        parent_of[node.right] = nid


def dendrogram_vertical_span_z():
    z_lo = float(DENDRO_BASE_Z)
    z_hi = z_lo
    for g in cluster_geom.values():
        z_hi = max(z_hi, float(g['z']))
    return z_lo, z_hi


def dendro_stem_bevel_for_height(z_merge, z_base, z_top, bevel_max):
    """Bevel radius: flat minimum for the lower 90% of Z-span; ramp to ``bevel_max`` in the top 10%."""
    span = max(float(z_top) - float(z_base), 1e-9)
    t = (float(z_merge) - float(z_base)) / span
    t = max(0.0, min(1.0, t))
    b_thin = float(bevel_max) * float(DENDRO_STEM_BEVEL_BOTTOM_FRAC)
    u0 = float(DENDRO_STEM_BEVEL_UNIFORM_HEIGHT_FRAC)
    if t <= u0:
        return b_thin
    u = (t - u0) / max(1e-9, 1.0 - u0)
    return b_thin + (float(bevel_max) - b_thin) * u


def ensure_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    master = bpy.context.scene.collection
    try:
        master.children.link(col)
    except RuntimeError:
        pass
    return col


def leaf_indices_level1_block():
    """25 points: mega 0, super 0 — five micro-clusters of five."""
    return [
        i
        for i, (a, b) in enumerate(zip(LEAF_MEGA, LEAF_SUPER))
        if a == 0 and b == 0
    ]


def leaf_indices_level2_block():
    """125 points: mega 0 — five super-clusters of twenty-five."""
    return [i for i, a in enumerate(LEAF_MEGA) if a == 0]


def bbox_3d_for_leaves(leaf_ids, pad_xy=0.42, pad_z=(0.14, 0.58)):
    """World-space bounds: XY from leaves; Z includes dendrogram merges inside the leaf set."""
    if not leaf_ids:
        return -1.0, 1.0, -1.0, 1.0, float(DENDRO_BASE_Z), float(DENDRO_BASE_Z) + 0.5
    s = set(leaf_ids)
    xs = [POINTS[i][0] for i in leaf_ids]
    ys = [POINTS[i][1] for i in leaf_ids]
    xmin, xmax = min(xs) - pad_xy, max(xs) + pad_xy
    ymin, ymax = min(ys) - pad_xy, max(ys) + pad_xy
    z_lo = float(DENDRO_BASE_Z)
    z_hi = z_lo
    for nid, node in nodes.items():
        if len(node.members) < 2:
            continue
        if set(node.members).issubset(s):
            z_hi = max(z_hi, float(cluster_geom[nid]['z']))
    z_hi = max(z_hi, z_lo + 0.08)
    z_lo = z_lo - pad_z[0]
    z_hi = z_hi + pad_z[1]
    return xmin, xmax, ymin, ymax, z_lo, z_hi


def camera_pose_top(cam_data, bbox_3d, dist_mul=1.62, span_mul=1.06):
    """Straight-down view (+Z): see K clusters in the XY plane."""
    xmin, xmax, ymin, ymax, z_lo, z_hi = bbox_3d
    cx = 0.5 * (xmin + xmax)
    cy_world = 0.5 * (ymin + ymax) + DATA_Y_OFFSET
    z_look = float(DATA_Z + POINT_RADIUS * 0.35)
    target = Vector((cx, cy_world, z_look))
    span_xy = max(xmax - xmin, ymax - ymin) * span_mul
    span_xy = max(span_xy, 0.22)
    dist = max(span_xy * dist_mul, 7.5)
    cam_pos = target + Vector((0.0, 0.0, dist))
    euler = camera_euler_look_at(cam_pos, target)
    lens = perspective_lens_mm_match_span(cam_data, span_xy, dist)
    return cam_pos, euler, float(lens), target


def camera_pose_side(cam_data, bbox_3d, dist_mul=2.32, span_mul=1.05):
    """Side view: X vs merge height Z (dendrogram.py style)."""
    xmin, xmax, ymin, ymax, z_lo, z_hi = bbox_3d
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax) + DATA_Y_OFFSET
    cz = 0.5 * (z_lo + z_hi)
    target = Vector((cx, cy, cz))
    span = max(xmax - xmin, ymax - ymin, z_hi - z_lo) * span_mul
    span = max(span, 0.35)
    dist = max(span * dist_mul, 11.0)
    cam_pos = target + CAM_VIEW_DIR * dist
    euler = camera_euler_look_at(cam_pos, target)
    lens = perspective_lens_mm_match_span(cam_data, span, dist)
    return cam_pos, euler, float(lens), target


def persp_pose_shift_cam_world_z(pose, dz):
    """Same target and lens; move camera in world Z and re-aim at target."""
    cam_pos, _euler, lens, target = pose
    p = Vector(cam_pos) + Vector((0.0, 0.0, float(dz)))
    eul = camera_euler_look_at(p, target)
    return (p, eul, float(lens), target)


def persp_pose_tilt_from_top_deg(look_at_pt, top_pose, side_pose, tilt_deg_from_top, cam_data):
    """Swing camera from top toward side by ``tilt_deg_from_top``; aim stays at ``look_at_pt`` (level-3 frame center)."""
    ct, _, lt, _tt = top_pose
    cs, _, ls, _ts = side_pose
    p0 = Vector(look_at_pt)
    v0 = Vector(ct) - p0
    v1 = Vector(cs) - p0
    u0 = v0.normalized()
    u1 = v1.normalized()
    ang = u0.angle(u1)
    want = math.radians(float(tilt_deg_from_top))
    t = (want / ang) if ang > 1e-6 else 0.0
    t = min(1.0, max(0.0, t))
    u = u0.lerp(u1, t).normalized()
    dist = v0.length * (1.0 - t) + v1.length * t
    loc = p0 + u * dist
    lens = float(lt) + (float(ls) - float(lt)) * t
    return (loc, None, float(lens), p0)


def split_indices_evenly(n_items, n_groups):
    if n_items <= 0 or n_groups <= 0:
        return []
    q, r = divmod(n_items, n_groups)
    chunks = []
    i0 = 0
    for g in range(n_groups):
        take = q + (1 if g < r else 0)
        chunks.append((i0, i0 + take))
        i0 += take
    return chunks


# Reference poses (K=5 nested rings): three top-down framings, then one full side framing.
_lvl1_25 = leaf_indices_level1_block()
_lvl2_125 = leaf_indices_level2_block()
_all = list(range(len(POINTS)))
Z_LO_FULL, Z_HI_FULL = dendrogram_vertical_span_z()
Z_PAD_BOTTOM = 0.2
Z_PAD_TOP = 0.95
X_ALL = [POINTS[i][0] for i in _all]
Y_ALL = [POINTS[i][1] for i in _all]
GLOBAL_XMIN = min(X_ALL) - 1.15
GLOBAL_XMAX = max(X_ALL) + 1.15
GLOBAL_YMIN = min(Y_ALL) - 1.15
GLOBAL_YMAX = max(Y_ALL) + 1.15
GLOBAL_ZLO = float(DENDRO_BASE_Z) - Z_PAD_BOTTOM
GLOBAL_ZHI = float(Z_HI_FULL) + Z_PAD_TOP

bx1 = bbox_3d_for_leaves(_lvl1_25, pad_xy=0.28, pad_z=(0.1, 0.42))
bx2 = bbox_3d_for_leaves(_lvl2_125, pad_xy=0.52, pad_z=(0.12, 0.65))
bx3 = (GLOBAL_XMIN, GLOBAL_XMAX, GLOBAL_YMIN, GLOBAL_YMAX, GLOBAL_ZLO, GLOBAL_ZHI)

pivot = Vector(
    (
        sum(p[0] for p in POINTS) / len(POINTS),
        sum(p[1] for p in POINTS) / len(POINTS) + DATA_Y_OFFSET,
        0.5 * (GLOBAL_ZLO + GLOBAL_ZHI),
    )
)

_cam_ref = bpy.data.cameras.new('_MultiscaleLensRef')
TOP_POSE_1 = camera_pose_top(
    _cam_ref, bx1, dist_mul=TOP_ZOOM_DIST_MUL_1, span_mul=TOP_ZOOM_SPAN_MUL_1
)
TOP_POSE_2 = camera_pose_top(
    _cam_ref, bx2, dist_mul=TOP_ZOOM_DIST_MUL_2, span_mul=TOP_ZOOM_SPAN_MUL_2
)
TOP_POSE_3 = camera_pose_top(
    _cam_ref, bx3, dist_mul=TOP_ZOOM_DIST_MUL_3, span_mul=TOP_ZOOM_SPAN_MUL_3
)
# Level-3 onward: keep this point as the camera look-at so the frame center does not drift (avoids pivot offset).
L3_CAM_TARGET = Vector(TOP_POSE_3[3])
_bx3_x = bx3[1] - bx3[0]
_bx3_y = bx3[3] - bx3[2]
_bx3_z = bx3[5] - bx3[4]
YAW_ORBIT_TARGET_SPAN = max(_bx3_x, _bx3_y, _bx3_z) * 1.06

_look = Vector(L3_CAM_TARGET)
_TH45 = math.radians(float(TILT_FROM_TOP_DEG))
# Horizontal side view on the same YZ meridian (θ = π/2: camera in −Y from look-at).
_TH90 = 0.5 * math.pi
_R_TOP = max((Vector(TOP_POSE_3[0]) - _look).length, 7.5)
# Dome radius at 45° (meridian in YZ through ``_look``; camera ΔX = 0).
CONE_ORBIT_RADIUS_BASE = max(_R_TOP * 1.08, 10.0)
_R_45 = float(CONE_ORBIT_RADIUS_BASE)
loc45 = _look + meridian_cam_offset_yz(_TH45, _R_45)
_d45 = _R_45
_lens45 = perspective_lens_mm_match_span(_cam_ref, YAW_ORBIT_TARGET_SPAN, _d45)
POSE_TILT_45 = (loc45, None, float(_lens45), _look)
_u45_hint = meridian_cam_dir_yz(_TH45)

CONE_ORBIT_RADIUS = float(CONE_ORBIT_RADIUS_BASE) * float(DENDRO_TOP_ZOOM_DISTANCE_MUL)
_orbit_lens_span = YAW_ORBIT_TARGET_SPAN * float(DENDRO_TOP_ZOOM_LENS_SPAN_MUL)

_lens_orbit = perspective_lens_mm_match_span(
    _cam_ref, _orbit_lens_span, max(float(CONE_ORBIT_RADIUS), 1e-6)
)

bpy.data.cameras.remove(_cam_ref)

mat_point_tpl = make_principled_material(
    'PointMatTpl', POINT_WHITE, roughness=0.25, emission_strength=0.26
)
mat_dendro = make_principled_material(
    'DendroMat', DENDRO_VERTICAL_COLOR, roughness=0.12, emission_strength=0.52
)
mat_ground = make_principled_material(
    'GroundMat', (0.03, 0.035, 0.045, 1.0), roughness=0.75
)
mat_sky_dome = make_principled_material(
    'SkyDomeMat', (0.0, 0.0, 0.0, 1.0), roughness=1.0, alpha=1.0, emission_strength=0.0
)
mat_sky_dome.blend_method = 'OPAQUE'
mat_caption = make_emission_material('CaptionMat', CAPTION_COLOR, strength=1.8, alpha=1.0)
mat_merge_chord = make_emission_material(
    'MergeChordMat',
    CLUSTER_PURPLE,
    strength=float(MERGE_CHORD_EMIT_STRENGTH),
    alpha=float(MERGE_CHORD_ALPHA),
)

col_points = ensure_collection('Points')
col_dendro = ensure_collection('Dendrogram')
col_circles = ensure_collection('ClusterCircles')
col_guides = ensure_collection('Guides')

ground = add_plane('Ground', (0, 0, -0.02), GROUND_PLANE_SIZE, mat_ground)
for c in list(ground.users_collection):
    c.objects.unlink(ground)
col_guides.objects.link(ground)
sky_dome = add_dark_sky_dome('SkyDome', pivot, SKY_DOME_RADIUS, mat_sky_dome)
for c in list(sky_dome.users_collection):
    c.objects.unlink(sky_dome)
col_guides.objects.link(sky_dome)

# -----------------------------------------------------------------------------
# Points: each top-view scale shows five natural clusters; hue = that partition,
# with small sat/value variation from finer labels (story: “five groups” → zoom → bigger five).
#   L1 (25):  micro-cluster c ∈ 0..K-1  → K distinct hues
#   L2 (125): super-cluster b
#   L3/side:  mega-cluster a
# -----------------------------------------------------------------------------
point_objects = []


def leaf_color_for_level(i, zoom_level):
    """zoom_level 0 = L1 (color by micro), 1 = L2 (by super), 2 = L3 / side (by mega)."""
    a = LEAF_MEGA[i]
    b = LEAF_SUPER[i]
    c = LEAF_MICRO[i]
    if zoom_level <= 0:
        g = c
    elif zoom_level == 1:
        g = b
    else:
        g = a
    kk = max(K, 1)
    # Evenly space K hues so the five visible clusters read as clearly different.
    h = (float(g) / float(kk)) % 1.0
    # Finer hierarchy still nudges sat/value so points differ slightly within a group.
    mix = (a * (kk * kk) + b * kk + c) / float(max(kk * kk * kk, 1))
    s = min(1.0, 0.5 + 0.28 * mix)
    v = 0.86 + 0.1 * mix
    r, g_, b_ = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (r, g_, b_, 1.0)


for i, (x, y) in enumerate(POINTS):
    pm = mat_point_tpl.copy()
    pm.name = f'PointMat_{i}'
    bsdf = pm.node_tree.nodes['Principled BSDF']
    c = leaf_color_for_level(i, 0)
    bsdf.inputs['Base Color'].default_value = c
    if 'Emission Color' in bsdf.inputs:
        bsdf.inputs['Emission Color'].default_value = c
    obj = add_uv_sphere(f'P{i}', (x, y + DATA_Y_OFFSET, DATA_Z), POINT_RADIUS, pm)
    for c_ in list(obj.users_collection):
        c_.objects.unlink(obj)
    col_points.objects.link(obj)
    point_objects.append(obj)

# -----------------------------------------------------------------------------
# Dendrogram curves + merge chords (horizontal connectors at each merge)
# -----------------------------------------------------------------------------
dendro_objects = []
dendro_curve_y_specs = []
merge_chord_meta = []
dendro_reveal_specs = []
_BEVEL_MAX_STEM = CURVE_BEVEL * DENDRO_CURVE_BEVEL_SCALE

for nid, node in nodes.items():
    if len(node.members) == 1:
        continue
    geom = cluster_geom[nid]
    cx, cy = geom['center']
    z = geom['z']
    left_geom = cluster_geom[node.left]
    right_geom = cluster_geom[node.right]
    lx, ly = left_geom['center']
    rx, ry = right_geom['center']
    lz = left_geom['z']
    rz = right_geom['z']
    wy_l = ly + DATA_Y_OFFSET
    wy_r = ry + DATA_Y_OFFSET
    wy_p = cy + DATA_Y_OFFSET

    b_stem = dendro_stem_bevel_for_height(z, DENDRO_BASE_Z, Z_HI_FULL, _BEVEL_MAX_STEM)
    conn1 = new_curve_object(
        f'ConnL_{nid}',
        [(lx, wy_l, lz), (lx, wy_l, z)],
        bevel=b_stem,
        material=mat_dendro,
    )
    conn2 = new_curve_object(
        f'ConnR_{nid}',
        [(rx, wy_r, rz), (rx, wy_r, z)],
        bevel=b_stem,
        material=mat_dendro,
    )
    for obj in (conn1, conn2):
        for c_ in list(obj.users_collection):
            c_.objects.unlink(obj)
        col_dendro.objects.link(obj)
        dendro_objects.append(obj)
    dendro_reveal_specs.append({'obj': conn1, 'reveal_z': float(z)})
    dendro_reveal_specs.append({'obj': conn2, 'reveal_z': float(z)})
    dendro_curve_y_specs.append({'obj': conn1, 'x': lx, 'y_data': ly, 'z0': lz, 'z1': z})
    dendro_curve_y_specs.append({'obj': conn2, 'x': rx, 'y_data': ry, 'z0': rz, 'z1': z})

    pid = parent_of.get(nid)
    if pid is not None:
        z_parent = cluster_geom[pid]['z']
        b_up = dendro_stem_bevel_for_height(z, DENDRO_BASE_Z, Z_HI_FULL, _BEVEL_MAX_STEM)
        conn_up = new_curve_object(
            f'ConnUp_{nid}',
            [(cx, wy_p, z), (cx, wy_p, z_parent)],
            bevel=b_up,
            material=mat_dendro,
        )
        for c_ in list(conn_up.users_collection):
            c_.objects.unlink(conn_up)
        col_dendro.objects.link(conn_up)
        dendro_objects.append(conn_up)
        dendro_reveal_specs.append({'obj': conn_up, 'reveal_z': float(z_parent)})
        dendro_curve_y_specs.append({'obj': conn_up, 'x': cx, 'y_data': cy, 'z0': z, 'z1': z_parent})

    if not merge_qualifies_for_chord(node, nodes):
        continue

    chord_obj = new_curve_object(
        f'MergeChord_{nid}',
        [(lx, wy_l, z), (rx, wy_r, z)],
        bevel=float(MERGE_CHORD_BEVEL),
        material=mat_merge_chord,
    )
    for c_ in list(chord_obj.users_collection):
        c_.objects.unlink(chord_obj)
    col_dendro.objects.link(chord_obj)
    dendro_objects.append(chord_obj)
    dendro_reveal_specs.append({'obj': chord_obj, 'reveal_z': float(z)})
    merge_chord_meta.append(
        {
            'nid': nid,
            'obj': chord_obj,
            'lx': lx,
            'ly': ly,
            'rx': rx,
            'ry': ry,
            'z': z,
        }
    )


def keyframe_dendro_connectors_y_settle(specs, f_start, f_end):
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

        span = float(max(1, f1 - f0))
        for f in range(f0, f1 + 1):
            u = (f - f0) / span
            u = u * u * (3.0 - 2.0 * u)
            _apply_frame(f, u)
        if cd.animation_data.action:
            _linearize_action_range(cd.animation_data.action, f0, f1)


def keyframe_merge_chords_follow_vertical_tips(meta_items, f_start, f_end):
    """Animate merge chord endpoints while leaf Y spreads (same timing as vertical connectors)."""
    f0, f1 = int(f_start), int(f_end)
    for item in meta_items:
        obj = item['obj']
        cd = obj.data
        if cd.animation_data is None:
            cd.animation_data_create()
        lx, ly = float(item['lx']), float(item['ly'])
        rx, ry = float(item['rx']), float(item['ry'])
        zf = float(item['z'])
        sp = cd.splines[0]

        def _apply_frame(f, u):
            yl = float(DATA_Y_OFFSET) + ly * u
            yr = float(DATA_Y_OFFSET) + ry * u
            sp.points[0].co = (lx, yl, zf, 1.0)
            sp.points[1].co = (rx, yr, zf, 1.0)
            sp.points[0].keyframe_insert(data_path='co', frame=f)
            sp.points[1].keyframe_insert(data_path='co', frame=f)

        span = float(max(1, f1 - f0))
        for f in range(f0, f1 + 1):
            u = (f - f0) / span
            u = u * u * (3.0 - 2.0 * u)
            _apply_frame(f, u)
        if cd.animation_data.action:
            _linearize_action_range(cd.animation_data.action, f0, f1)


keyframe_dendro_connectors_y_settle(dendro_curve_y_specs, F_START, F_SETTLE_LAST)
keyframe_merge_chords_follow_vertical_tips(merge_chord_meta, F_START, F_SETTLE_LAST)


def keyframe_dendro_connectors_y_final_hold(specs, frame):
    """Lock spline Y to full spread (u=1) — avoids extrapolation flattening to XZ before reveal."""
    f = int(frame)
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
        yw = float(DATA_Y_OFFSET) + yd
        sp.points[0].co = (x, yw, z0, 1.0)
        sp.points[1].co = (x, yw, z1, 1.0)
        sp.points[0].keyframe_insert(data_path='co', frame=f)
        sp.points[1].keyframe_insert(data_path='co', frame=f)
    for spec in specs:
        cd = spec['obj'].data
        if cd.animation_data and cd.animation_data.action and f <= F_END:
            _linearize_action_range(cd.animation_data.action, f, F_END)


def keyframe_merge_chords_final_hold(meta_items, frame):
    f = int(frame)
    for item in meta_items:
        obj = item['obj']
        cd = obj.data
        if cd.animation_data is None:
            cd.animation_data_create()
        lx, ly = float(item['lx']), float(item['ly'])
        rx, ry = float(item['rx']), float(item['ry'])
        zf = float(item['z'])
        sp = cd.splines[0]
        yl = float(DATA_Y_OFFSET) + ly
        yr = float(DATA_Y_OFFSET) + ry
        sp.points[0].co = (lx, yl, zf, 1.0)
        sp.points[1].co = (rx, yr, zf, 1.0)
        sp.points[0].keyframe_insert(data_path='co', frame=f)
        sp.points[1].keyframe_insert(data_path='co', frame=f)
    for item in meta_items:
        cd = item['obj'].data
        act = cd.animation_data.action if cd.animation_data else None
        if act is not None and f <= F_END:
            _linearize_action_range(act, f, F_END)


keyframe_dendro_connectors_y_final_hold(dendro_curve_y_specs, F_DENDRO_REVEAL_START)
keyframe_dendro_connectors_y_final_hold(dendro_curve_y_specs, F_END)
keyframe_merge_chords_final_hold(merge_chord_meta, F_DENDRO_REVEAL_START)
keyframe_merge_chords_final_hold(merge_chord_meta, F_END)

# -----------------------------------------------------------------------------
# Caption text (one active phase at a time)
# -----------------------------------------------------------------------------
CAPTION_Y = GLOBAL_ZHI + 1.15
CAPTION_X = GLOBAL_XMIN + 2.4


def add_caption(name, body, frame_on, frame_off=None):
    bpy.ops.object.text_add(location=(CAPTION_X, DATA_Y_OFFSET, CAPTION_Y))
    txt = bpy.context.active_object
    txt.name = name
    cd = txt.data
    cd.body = body
    cd.size = 0.42
    cd.extrude = 0.01
    cd.align_x = 'LEFT'
    txt.data.materials.append(mat_caption)
    for c_ in list(txt.users_collection):
        c_.objects.unlink(txt)
    col_guides.objects.link(txt)
    txt.rotation_euler = (math.radians(90), 0.0, 0.0)
    if txt.animation_data is None:
        txt.animation_data_create()
    txt.hide_viewport = frame_on > F_START
    txt.hide_render = frame_on > F_START
    txt.keyframe_insert(data_path='hide_viewport', frame=F_START)
    txt.keyframe_insert(data_path='hide_render', frame=F_START)
    txt.hide_viewport = False
    txt.hide_render = False
    txt.keyframe_insert(data_path='hide_viewport', frame=frame_on)
    txt.keyframe_insert(data_path='hide_render', frame=frame_on)
    if frame_off is not None:
        txt.hide_viewport = True
        txt.hide_render = True
        txt.keyframe_insert(data_path='hide_viewport', frame=frame_off)
        txt.keyframe_insert(data_path='hide_render', frame=frame_off)
    act = txt.animation_data.action if txt.animation_data else None
    if act is not None:
        _linearize_action_range(act, F_START, F_END)
        for fc in act.fcurves:
            if fc.data_path in ('hide_viewport', 'hide_render'):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'
    return txt


add_caption(
    'Cap_L1',
    'Level 1 — 25 points, five clusters of five (fine-scale merges)',
    F_START,
    F_CAP_L2,
)
add_caption(
    'Cap_L2',
    'Level 2 — 125 points, five clusters of twenty-five',
    F_CAP_L2,
    F_CAP_L3,
)
add_caption(
    'Cap_L3',
    'Level 3 — 625 points, five clusters of 125 (top view)',
    F_CAP_L3,
    F_CAP_SIDE,
)
add_caption(
    'Cap_SIDE',
    '45° tilt — dendrogram builds, pull-out, full CCW orbit, side view, second orbit',
    F_CAP_SIDE,
    None,
)

# -----------------------------------------------------------------------------
# Point scale: small when tight top view, larger after each zoom-out
# -----------------------------------------------------------------------------
def keyframe_point_scale_segment(objs, f_a, f_b, scale_a, scale_b):
    f0, f1 = int(f_a), int(f_b)
    if f1 < f0:
        return
    span = max(1, f1 - f0)
    for obj in objs:
        if obj.animation_data is None:
            obj.animation_data_create()
    for f in range(f0, f1 + 1):
        u = _smootherstep01((f - f0) / float(span))
        s = scale_a + (scale_b - scale_a) * u
        for obj in objs:
            obj.scale = (s, s, s)
            obj.keyframe_insert(data_path='scale', frame=f)
    for obj in objs:
        if obj.animation_data.action:
            _linearize_action_range(obj.animation_data.action, f0, f1)


def _keyframes_bsdf_point_colors(mat, rgba, frame):
    nt = mat.node_tree
    if nt is None:
        return
    if nt.animation_data is None:
        nt.animation_data_create()
    bsdf = nt.nodes['Principled BSDF']
    bc = bsdf.inputs['Base Color']
    bc.default_value = rgba
    bc.keyframe_insert(data_path='default_value', frame=frame)
    if 'Emission Color' in bsdf.inputs:
        em = bsdf.inputs['Emission Color']
        em.default_value = rgba
        em.keyframe_insert(data_path='default_value', frame=frame)


def keyframe_point_palette_ramp(point_objs, f_a, f_b, level_a, level_b):
    f0, f1 = int(f_a), int(f_b)
    if f1 < f0:
        return
    span = max(1, f1 - f0)
    for f in range(f0, f1 + 1):
        u = _smootherstep01((f - f0) / float(span))
        for i, obj in enumerate(point_objs):
            mat = obj.data.materials[0]
            c0 = leaf_color_for_level(i, level_a)
            c1 = leaf_color_for_level(i, level_b)
            rgba = tuple(c0[k] + (c1[k] - c0[k]) * u for k in range(4))
            _keyframes_bsdf_point_colors(mat, rgba, f)
    for obj in point_objs:
        nt = obj.data.materials[0].node_tree
        if nt and nt.animation_data and nt.animation_data.action:
            _linearize_action_range(nt.animation_data.action, f0, f1)


def keyframe_point_palette_twoframe_hold(point_objs, f_a, f_b, level):
    f0, f1 = int(f_a), int(f_b)
    if f1 < f0:
        return
    for i, obj in enumerate(point_objs):
        mat = obj.data.materials[0]
        rgba = leaf_color_for_level(i, level)
        _keyframes_bsdf_point_colors(mat, rgba, f0)
        if f1 != f0:
            _keyframes_bsdf_point_colors(mat, rgba, f1)
    for obj in point_objs:
        nt = obj.data.materials[0].node_tree
        if nt and nt.animation_data and nt.animation_data.action:
            _linearize_action_range(nt.animation_data.action, f0, f1)


for _po in point_objects:
    if _po.animation_data is None:
        _po.animation_data_create()
    _po.scale = (POINT_SCALE_L1, POINT_SCALE_L1, POINT_SCALE_L1)
    _po.keyframe_insert(data_path='scale', frame=F_START)
    _po.keyframe_insert(data_path='scale', frame=F_SETTLE_LAST)

keyframe_point_scale_segment(point_objects, F_L1_START, F_L1_END, POINT_SCALE_L1, POINT_SCALE_L1)
keyframe_point_scale_segment(point_objects, F_Z12_START, F_Z12_END, POINT_SCALE_L1, POINT_SCALE_L2)
keyframe_point_scale_segment(point_objects, F_Z12_END + 1, F_L2_END, POINT_SCALE_L2, POINT_SCALE_L2)
keyframe_point_scale_segment(point_objects, F_Z23_START, F_Z23_END, POINT_SCALE_L2, POINT_SCALE_L3)
keyframe_point_scale_segment(
    point_objects, F_Z23_END + 1, F_L3_TOP_END, POINT_SCALE_L3, POINT_SCALE_L3
)
keyframe_point_scale_segment(
    point_objects, F_TILT45_START, F_END, POINT_SCALE_L3, POINT_SCALE_L3
)

keyframe_point_palette_twoframe_hold(point_objects, F_START, F_SETTLE_LAST, 0)
keyframe_point_palette_twoframe_hold(point_objects, F_L1_START, F_L1_END, 0)
keyframe_point_palette_ramp(point_objects, F_Z12_START, F_Z12_END, 0, 1)
keyframe_point_palette_twoframe_hold(point_objects, F_Z12_END + 1, F_L2_END, 1)
keyframe_point_palette_ramp(point_objects, F_Z23_START, F_Z23_END, 1, 2)
# Key from last Z23 frame through end so ramp end and L3/post-L3 share identical level-2 keys (no RGB seam).
keyframe_point_palette_twoframe_hold(point_objects, F_Z23_END, F_END, 2)

# -----------------------------------------------------------------------------
# Dendrogram + ovals: hidden until layered reveal (bottom merge height → top).
# -----------------------------------------------------------------------------
def _force_constant_hide_fcurves(action):
    if action is None:
        return
    for fc in action.fcurves:
        if fc.data_path in ('hide_viewport', 'hide_render'):
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'


dendro_reveal_specs.sort(key=lambda s: (s['reveal_z'], s['obj'].name))
_n_rev = len(dendro_reveal_specs)
_layer_spans = split_indices_evenly(_n_rev, DENDRO_REVEAL_LAYERS)
for _spec in dendro_reveal_specs:
    _vo = _spec['obj']
    if _vo.animation_data is None:
        _vo.animation_data_create()
    _vo.hide_viewport = True
    _vo.hide_render = True
    _vo.keyframe_insert(data_path='hide_viewport', frame=F_START)
    _vo.keyframe_insert(data_path='hide_render', frame=F_START)

for _layer_i, (_i0, _i1) in enumerate(_layer_spans):
    if _i0 >= _i1:
        continue
    _f_show = F_DENDRO_REVEAL_START + _layer_i * DENDRO_REVEAL_FRAMES_PER_LAYER
    for _j in range(_i0, _i1):
        _vo = dendro_reveal_specs[_j]['obj']
        _vo.hide_viewport = False
        _vo.hide_render = False
        _vo.keyframe_insert(data_path='hide_viewport', frame=_f_show)
        _vo.keyframe_insert(data_path='hide_render', frame=_f_show)

for _spec in dendro_reveal_specs:
    _force_constant_hide_fcurves(_spec['obj'].animation_data.action)

# -----------------------------------------------------------------------------
# Camera: same creation pattern as ``dendrogram.py`` (ops + VIEW_3D override), plus data fallback
# -----------------------------------------------------------------------------
def keyframe_persp_segment(cam, f_a, f_b, pose_a, pose_b, up_axis='Y', quaternion_prev=None):
    """Interpolate camera; key ``camera.data`` lens/type like ``_commit_persp_camera_keyframe``."""
    pos_a, _, lens_a, tgt_a = pose_a
    pos_b, _, lens_b, tgt_b = pose_b
    _ensure_camera_actions(cam)
    _camera_use_quaternion_rotation(cam)
    span = max(1, int(f_b) - int(f_a))
    prev_q = quaternion_prev
    for f in range(int(f_a), int(f_b) + 1):
        t = (f - f_a) / float(span)
        u = _smootherstep01(t)
        loc = pos_a.lerp(pos_b, u)
        tgt = tgt_a.lerp(tgt_b, u)
        lens = lens_a + (lens_b - lens_a) * u
        cam.location = loc
        q = camera_quaternion_look_at(loc, tgt, up_axis=up_axis)
        q = _quaternion_align_previous(q, prev_q)
        prev_q = q
        cam.rotation_quaternion = q
        cam.data.type = 'PERSP'
        cam.data.lens = lens
        cam.keyframe_insert(data_path='location', frame=f)
        cam.keyframe_insert(data_path='rotation_quaternion', frame=f)
        cam.data.keyframe_insert(data_path='type', frame=f)
        cam.data.keyframe_insert(data_path='lens', frame=f)
    _linearize_camera_object_and_data(cam, f_a, f_b)


def keyframe_persp_zoom_with_dir_tilt(
    cam,
    f_a,
    f_b,
    pose_a,
    pose_b,
    tilt_dir_unit,
    tilt_mix_end,
    up_axis='Y',
    quaternion_prev=None,
):
    """Zoom between two top poses; tilt toward ``tilt_dir_unit`` peaks mid-zoom (sin envelope) so u=1 matches pose_b."""
    pos_a, _, lens_a, tgt_a = pose_a
    pos_b, _, lens_b, tgt_b = pose_b
    Ta, Tb = Vector(tgt_a), Vector(tgt_b)
    Pa, Pb = Vector(pos_a), Vector(pos_b)
    va = Pa - Ta
    vb = Pb - Tb
    r_a = max(va.length, 1e-8)
    r_b = max(vb.length, 1e-8)
    d0 = va / r_a
    d_end = vb / r_b
    vt = Vector(tilt_dir_unit).normalized()
    _ensure_camera_actions(cam)
    _camera_use_quaternion_rotation(cam)
    span = max(1, int(f_b) - int(f_a))
    prev_q = quaternion_prev
    for f in range(int(f_a), int(f_b) + 1):
        t = (f - f_a) / float(span)
        u = _smootherstep01(t)
        T = Ta.lerp(Tb, u)
        mid_mix = math.sin(u * math.pi) * float(tilt_mix_end)
        dir_in = _slerp_unit_vec(d0, vt, mid_mix)
        dir_u = _slerp_unit_vec(dir_in, d_end, u)
        r_u = r_a + (r_b - r_a) * u
        loc = T + dir_u * r_u
        lens = lens_a + (lens_b - lens_a) * u
        cam.location = loc
        q = camera_quaternion_look_at(loc, T, up_axis=up_axis)
        q = _quaternion_align_previous(q, prev_q)
        prev_q = q
        cam.rotation_quaternion = q
        cam.data.type = 'PERSP'
        cam.data.lens = lens
        cam.keyframe_insert(data_path='location', frame=f)
        cam.keyframe_insert(data_path='rotation_quaternion', frame=f)
        cam.data.keyframe_insert(data_path='type', frame=f)
        cam.data.keyframe_insert(data_path='lens', frame=f)
    _linearize_camera_object_and_data(cam, f_a, f_b)


def keyframe_persp_meridian_arc(
    cam,
    f_a,
    f_b,
    look,
    theta_a,
    theta_b,
    r_a,
    r_b,
    lens_a,
    lens_b,
    up_axis='Z',
    quaternion_prev=None,
    quaternion_slerp_end=None,
    parallel_shift_per_frame=None,
):
    """Interpolate camera on the YZ meridian through ``look`` (ΔX = 0): polar angle θ and radius r.

    If ``quaternion_slerp_end`` is set (with ``quaternion_prev``), rotation is a slerp from the start
    quaternion to that end quaternion so tilt-from-top matches the prior POV with no per-frame
    look-at roll flip.

    ``parallel_shift_per_frame(f)`` adds a shift-lens offset along camera +Y after rotation is fixed.
    """
    look_v = Vector(look)
    _ensure_camera_actions(cam)
    _camera_use_quaternion_rotation(cam)
    span = max(1, int(f_b) - int(f_a))
    if quaternion_slerp_end is not None:
        if quaternion_prev is None:
            raise ValueError('quaternion_slerp_end requires quaternion_prev')
        q_start = mathutils.Quaternion(quaternion_prev)
        q_end = mathutils.Quaternion(quaternion_slerp_end)
        for f in range(int(f_a), int(f_b) + 1):
            t = (f - f_a) / float(span)
            u = _smootherstep01(t)
            th = float(theta_a) + (float(theta_b) - float(theta_a)) * u
            r = float(r_a) + (float(r_b) - float(r_a)) * u
            base_loc = look_v + meridian_cam_offset_yz(th, r)
            lens = float(lens_a) + (float(lens_b) - float(lens_a)) * u
            q = q_start.slerp(q_end, u)
            sh = (
                0.0
                if parallel_shift_per_frame is None
                else float(parallel_shift_per_frame(f))
            )
            cam.location = base_loc + _camera_parallel_shift_offset(q, sh)
            cam.rotation_quaternion = q
            cam.data.type = 'PERSP'
            cam.data.lens = lens
            cam.keyframe_insert(data_path='location', frame=f)
            cam.keyframe_insert(data_path='rotation_quaternion', frame=f)
            cam.data.keyframe_insert(data_path='type', frame=f)
            cam.data.keyframe_insert(data_path='lens', frame=f)
    else:
        prev_q = quaternion_prev
        for f in range(int(f_a), int(f_b) + 1):
            t = (f - f_a) / float(span)
            u = _smootherstep01(t)
            th = float(theta_a) + (float(theta_b) - float(theta_a)) * u
            r = float(r_a) + (float(r_b) - float(r_a)) * u
            base_loc = look_v + meridian_cam_offset_yz(th, r)
            lens = float(lens_a) + (float(lens_b) - float(lens_a)) * u
            q = camera_quaternion_look_at(base_loc, look_v, up_axis=up_axis)
            q = _quaternion_align_previous(q, prev_q)
            prev_q = q
            sh = (
                0.0
                if parallel_shift_per_frame is None
                else float(parallel_shift_per_frame(f))
            )
            cam.location = base_loc + _camera_parallel_shift_offset(q, sh)
            cam.rotation_quaternion = q
            cam.data.type = 'PERSP'
            cam.data.lens = lens
            cam.keyframe_insert(data_path='location', frame=f)
            cam.keyframe_insert(data_path='rotation_quaternion', frame=f)
            cam.data.keyframe_insert(data_path='type', frame=f)
            cam.data.keyframe_insert(data_path='lens', frame=f)
    _linearize_camera_object_and_data(cam, f_a, f_b)


def keyframe_persp_meridian_hold(
    cam,
    f_a,
    f_b,
    look,
    theta,
    radius,
    lens,
    up_axis='Z',
    quaternion_prev=None,
    parallel_shift_per_frame=None,
):
    """Hold pose on the YZ meridian."""
    look_v = Vector(look)
    base_loc = look_v + meridian_cam_offset_yz(float(theta), float(radius))
    _ensure_camera_actions(cam)
    _camera_use_quaternion_rotation(cam)
    prev_q = quaternion_prev
    for f in range(int(f_a), int(f_b) + 1):
        q = camera_quaternion_look_at(base_loc, look_v, up_axis=up_axis)
        q = _quaternion_align_previous(q, prev_q)
        prev_q = q
        sh = (
            0.0
            if parallel_shift_per_frame is None
            else float(parallel_shift_per_frame(f))
        )
        cam.location = base_loc + _camera_parallel_shift_offset(q, sh)
        cam.rotation_quaternion = q
        cam.data.type = 'PERSP'
        cam.data.lens = float(lens)
        cam.keyframe_insert(data_path='location', frame=f)
        cam.keyframe_insert(data_path='rotation_quaternion', frame=f)
        cam.data.keyframe_insert(data_path='type', frame=f)
        cam.data.keyframe_insert(data_path='lens', frame=f)
    _linearize_camera_object_and_data(cam, f_a, f_b)


def keyframe_persp_dome_orbit_ccw(
    cam,
    f_a,
    f_b,
    look,
    theta_from_plus_z,
    radius,
    azimuth_start,
    lens,
    up_axis='Z',
    quaternion_prev=None,
    parallel_shift_per_frame=None,
):
    """Fixed θ and R on the dome; φ goes 2π CCW (from +Z). ``up_axis='Z'`` keeps horizon in XY (no banking)."""
    look_v = Vector(look)
    th = float(theta_from_plus_z)
    r = float(radius)
    phi0 = float(azimuth_start)
    _ensure_camera_actions(cam)
    _camera_use_quaternion_rotation(cam)
    f0, f1 = int(f_a), int(f_b)
    span = max(1, f1 - f0)
    prev_q = quaternion_prev
    for f in range(f0, f1 + 1):
        u = _smoothstep01((f - f0) / float(span))
        phi = phi0 + u * 2.0 * math.pi
        base_loc = look_v + dome_cam_offset_from_look(th, phi, r)
        q = camera_quaternion_look_at(base_loc, look_v, up_axis=up_axis)
        q = _quaternion_align_previous(q, prev_q)
        prev_q = q
        sh = (
            0.0
            if parallel_shift_per_frame is None
            else float(parallel_shift_per_frame(f))
        )
        cam.location = base_loc + _camera_parallel_shift_offset(q, sh)
        cam.rotation_quaternion = q
        cam.data.type = 'PERSP'
        cam.data.lens = float(lens)
        cam.keyframe_insert(data_path='location', frame=f)
        cam.keyframe_insert(data_path='rotation_quaternion', frame=f)
        cam.data.keyframe_insert(data_path='type', frame=f)
        cam.data.keyframe_insert(data_path='lens', frame=f)
    _linearize_camera_object_and_data(cam, f_a, f_b)


def _dendro_par_shift_amt(f):
    return float(DENDRO_VIEW_PARALLEL_SHIFT) * _dendro_shift_scale_for_frame(f)


def _side_view_par_shift_amt(f):
    f = int(f)
    t0, t1 = int(F_SIDE_TILT_START), int(F_SIDE_TILT_END)
    s = float(SIDE_VIEW_PARALLEL_SHIFT)
    if f < t0:
        return 0.0
    if t1 >= t0 and f <= t1:
        span = max(1, t1 - t0)
        u = _smootherstep01((f - t0) / float(span))
        return s * u
    return s


_euler_top1 = camera_euler_look_at(TOP_POSE_1[0], TOP_POSE_1[3])
camera = create_main_camera_like_dendrogram(
    scene,
    TOP_POSE_1[0],
    (_euler_top1.x, _euler_top1.y, _euler_top1.z),
)
_ensure_camera_actions(camera)
_camera_use_quaternion_rotation(camera)
camera.data.lens = TOP_POSE_1[2]
camera.location = TOP_POSE_1[0]
camera.rotation_quaternion = camera_quaternion_look_at(TOP_POSE_1[0], TOP_POSE_1[3])
for _cf in (F_START, F_SETTLE_LAST):
    camera.data.type = 'PERSP'
    camera.keyframe_insert(data_path='location', frame=_cf)
    camera.keyframe_insert(data_path='rotation_quaternion', frame=_cf)
    camera.data.keyframe_insert(data_path='type', frame=_cf)
    camera.data.keyframe_insert(data_path='lens', frame=_cf)
_linearize_camera_object_and_data(camera, F_START, F_SETTLE_LAST)

_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_segment(
    camera,
    F_L1_START,
    F_L1_END,
    TOP_POSE_1,
    TOP_POSE_1,
    quaternion_prev=_cam_q_carry,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_segment(
    camera,
    F_Z12_START,
    F_Z12_END,
    TOP_POSE_1,
    TOP_POSE_2,
    quaternion_prev=_cam_q_carry,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_segment(
    camera,
    F_Z12_END + 1,
    F_L2_END,
    TOP_POSE_2,
    TOP_POSE_2,
    quaternion_prev=_cam_q_carry,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_zoom_with_dir_tilt(
    camera,
    F_Z23_START,
    F_Z23_END,
    TOP_POSE_2,
    TOP_POSE_3,
    _u45_hint,
    ZOOM_2_3_TILT_MIX,
    up_axis='Y',
    quaternion_prev=_cam_q_carry,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
# Same Y-up as L1/L2/Z23 so level-3 top matches earlier tops (Z-up would re-pick roll and look rotated).
# Tilt uses quaternion slerp from this carry into meridian, so no Y→Z snap at tilt start.
keyframe_persp_segment(
    camera,
    F_Z23_END + 1,
    F_L3_TOP_END,
    TOP_POSE_3,
    TOP_POSE_3,
    quaternion_prev=_cam_q_carry,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
_q_tilt45 = camera_quaternion_look_at(loc45, _look, up_axis='Z')
_q_tilt45 = _quaternion_align_previous(_q_tilt45, _cam_q_carry)
keyframe_persp_meridian_arc(
    camera,
    F_TILT45_START,
    F_TILT45_END,
    _look,
    0.0,
    _TH45,
    _R_TOP,
    _R_45,
    TOP_POSE_3[2],
    POSE_TILT_45[2],
    quaternion_prev=_cam_q_carry,
    quaternion_slerp_end=_q_tilt45,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
_pre_zoom_end = min(F_DENDRO_REVEAL_END, F_DENDRO_TOP_ZOOM_START - 1)
if F_DENDRO_REVEAL_START <= _pre_zoom_end:
    keyframe_persp_meridian_hold(
        camera,
        F_DENDRO_REVEAL_START,
        _pre_zoom_end,
        _look,
        _TH45,
        _R_45,
        POSE_TILT_45[2],
        quaternion_prev=_cam_q_carry,
    )
    _cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
if F_DENDRO_TOP_ZOOM_START <= F_DENDRO_REVEAL_END:
    keyframe_persp_meridian_arc(
        camera,
        F_DENDRO_TOP_ZOOM_START,
        F_DENDRO_REVEAL_END,
        _look,
        _TH45,
        _TH45,
        _R_45,
        float(CONE_ORBIT_RADIUS),
        POSE_TILT_45[2],
        float(_lens_orbit),
        quaternion_prev=_cam_q_carry,
        parallel_shift_per_frame=_dendro_par_shift_amt,
    )
    _cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_dome_orbit_ccw(
    camera,
    F_DOME_ORBIT_START,
    F_DOME_ORBIT_END,
    _look,
    _TH45,
    float(CONE_ORBIT_RADIUS),
    _DOME_PHI_MERIDIAN_MINUS_Y,
    float(_lens_orbit),
    quaternion_prev=_cam_q_carry,
    parallel_shift_per_frame=_dendro_par_shift_amt,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_meridian_arc(
    camera,
    F_SIDE_TILT_START,
    F_SIDE_TILT_END,
    _look,
    _TH45,
    _TH90,
    float(CONE_ORBIT_RADIUS),
    float(CONE_ORBIT_RADIUS),
    float(_lens_orbit),
    float(_lens_orbit),
    quaternion_prev=_cam_q_carry,
    parallel_shift_per_frame=_side_view_par_shift_amt,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
keyframe_persp_dome_orbit_ccw(
    camera,
    F_SIDE_ORBIT_START,
    F_SIDE_ORBIT_END,
    _look,
    _TH90,
    float(CONE_ORBIT_RADIUS),
    _DOME_PHI_MERIDIAN_MINUS_Y,
    float(_lens_orbit),
    quaternion_prev=_cam_q_carry,
    parallel_shift_per_frame=_side_view_par_shift_amt,
)
_cam_q_carry = mathutils.Quaternion(camera.rotation_quaternion)
if F_SIDE_ORBIT_END + 1 <= F_END:
    keyframe_persp_meridian_hold(
        camera,
        F_SIDE_ORBIT_END + 1,
        F_END,
        _look,
        _TH90,
        float(CONE_ORBIT_RADIUS),
        float(_lens_orbit),
        quaternion_prev=_cam_q_carry,
        parallel_shift_per_frame=_side_view_par_shift_amt,
    )

scene.camera = camera

# -----------------------------------------------------------------------------
# Viewport (match ``dendrogram.py``: camera view + hide empties/cameras in outliner clutter)
# -----------------------------------------------------------------------------
def switch_viewport_to_camera():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    space.lock_camera = False
                    break


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

scene.frame_set(F_START)
