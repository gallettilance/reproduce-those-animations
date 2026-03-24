"""
Threshold sweep only: same diagonal camera and geometry style as ``dendrogram.py`` at sweep start
(orbit pivot, +X tilt from top, lens matched to ORBIT_ORTHO_SCALE). Differences from the main script:
lighter threshold plane as a grid with transparent elliptical cutouts where the slice intersects merge
ovals, and per-cluster point colors: leaves 7 and 13 use CLUSTER_PURPLE;
other leaves get deterministic random hues; when clusters merge, the larger child sets the color; if
sizes tie, the subtree that contains the purple-designated leaves wins; otherwise left wins.

Run from a fresh Blender file (Scripting → Run).
"""
import bpy
import colorsys
import math
import mathutils
import random
from mathutils import Vector

# -----------------------------------------------------------------------------
# User controls
# -----------------------------------------------------------------------------
N_POINTS = 15
POINT_EXCLUDE_INDICES = {1}
MIN_X_PAIR_GAP = 0.34
LINKAGE = 'single'

POINT_RADIUS = 0.1
CURVE_BEVEL = 0.03
DENDRO_CURVE_BEVEL_SCALE = 1.35
# Same merge rings as dendrogram.py
CIRCLE_BEVEL = 0.01
# Threshold sweep: merge-ring bevel peak = CIRCLE_BEVEL * this (main script uses 6.0; top view is softer).
OVAL_BEVEL_PEAK_SCALE = 3.25
# Fraction of max merge height used to ease oval bold on/off at cluster boundaries (avoids hard pops).
OVAL_BOLD_EDGE_FRAC = 0.014
OVAL_COLLAPSE_MINOR = 0.018

HEIGHT_SCALE = 1.05
DATA_Y_OFFSET = 0.0
DATA_Z = 0.0
DENDRO_Z_LIFT = 0.0
DENDRO_BASE_Z = DATA_Z + POINT_RADIUS + DENDRO_Z_LIFT
PLANE_SIZE = 18.0
GROUND_PLANE_SIZE = 140.0
SKY_DOME_RADIUS = 480.0

# Threshold slice: match ``dendrogram.py`` (readable card + keyed mix); grid/cutouts sit on this base.
THRESHOLD_SLICE_COLOR = (0.75, 0.55, 0.98, 1.0)
THRESHOLD_PLANE_EMISSION_STRENGTH = 0.32
THRESHOLD_PLANE_MIX = 0.2
THRESHOLD_ABOVE_TOP_MERGE_Z = 0.22
# Sweep low Z: dip slightly past the leaf / zero-merge level (DENDRO_BASE_Z) so the slice clears the finest clusters.
THRESHOLD_SWEEP_Z_BELOW_LEAF = 0.065
# Threshold plane: Brick grid + shader cutouts at merge ovals when |plane Z − merge Z| is small.
THRESHOLD_GRID_BRICK_SCALE = 22.0
THRESHOLD_GRID_MORTAR_SIZE = 0.035
THRESHOLD_GRID_LINE_BRIGHT = 1.25
# Cell fill (brick Color2): keep well above black so the plane stays readable between cutouts.
THRESHOLD_GRID_CELL_DIM = 0.52
# Cutouts never go fully transparent (avoids “plane vanishes” into dark world / stacked holes).
THRESHOLD_CUTOUT_MAX_DEPTH = 0.78
# Z: Lorentzian gate 1/(1+(|dz|/sigma)^2); needs true world Z (see TexCoord → Vector Transform in material).
THRESHOLD_CUTOUT_Z_SIGMA = 0.055
# Max semi-axis (world) for shader holes; real merge hulls can span the whole plane (root) otherwise.
THRESHOLD_CUTOUT_MAX_SEMI_FRAC = 0.4
THRESHOLD_CUTOUT_MAX_SEMI_MIN = 0.26
THRESHOLD_CUTOUT_MAX_SEMI_CAP = 1.05
THRESHOLD_CUTOUT_ELLIPSE_SOFT = 0.045
THRESHOLD_CUTOUT_RADIUS_PAD = 1.04
SWEEP_DOWN_FAST_EXP = 2
SWEEP_UP_ACCEL_EXP = 2

# Same camera construction as dendrogram.py (diagonal threshold view)
ORBIT_ORTHO_SCALE = 17.0
TOP_CAMERA_Z_OFFSET = 20.0
THRESHOLD_ORBIT_RX_END_DEG = 75.0
THRESHOLD_CAM_DISTANCE_SCALE = 0.8

# Timeline: short pre-roll → same sweep timing as dendrogram.py (hold / down / up / hold)
F_START = 1
PRE_SWEEP_HOLD_FRAMES = 24
F_SWEEP_SLICE_ON = F_START + PRE_SWEEP_HOLD_FRAMES
SWEEP_HOLD_TOP_BEFORE_DESCENT = 60
F_SWEEP_DESCENT_START = F_SWEEP_SLICE_ON + SWEEP_HOLD_TOP_BEFORE_DESCENT
SWEEP_DOWN_FRAMES = 180
F_SWEEP_BOTTOM = F_SWEEP_DESCENT_START + SWEEP_DOWN_FRAMES
SWEEP_UP_FRAMES = 180
F_SWEEP_TOP_AGAIN = F_SWEEP_BOTTOM + SWEEP_UP_FRAMES
SWEEP_HOLD_TOP_AFTER_ASCENT = 60
F_END = F_SWEEP_TOP_AGAIN + SWEEP_HOLD_TOP_AFTER_ASCENT

# Aliases so ``threshold_plane_z_for_sweep_frame`` matches dendrogram.py math
F_THRESHOLD_SWEEP_START = F_SWEEP_SLICE_ON
F_THRESHOLD_SWEEP_END = F_SWEEP_BOTTOM
F_THRESHOLD_RETURN_END = F_SWEEP_TOP_AGAIN
F_THRESHOLD_TOP_HOLD_END = F_END

# Per-cluster point colors (dendrogram uses purple / white only)
CLUSTER_COLOR_S = 0.62
CLUSTER_COLOR_V = 0.92
POINT_WHITE = (1.0, 1.0, 1.0, 1.0)
CLUSTER_PURPLE = (0.58, 0.32, 0.88, 1.0)
# 0-based indices into POINTS (after POINT_EXCLUDE_INDICES) that share purple before any merge.
PURPLE_LEAF_INDICES = frozenset({7, 13})
LEAF_COLOR_RANDOM_SEED = 317
POINT_EMISSION = 0.28

# -----------------------------------------------------------------------------
# Cleanup (same strategy as dendrogram.py)
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
scene.eevee.use_bloom = False
scene.eevee.taa_render_samples = 24
scene.eevee.taa_samples = 24
scene.render.film_transparent = False
if hasattr(scene.render, 'use_persistent_data'):
    scene.render.use_persistent_data = True
scene.use_nodes = False
scene.view_settings.exposure = 0.0

RENDER_WORLD_BG_RGB = (0.0, 0.0, 0.0)


def configure_dark_render_world(target_scene, rgb=RENDER_WORLD_BG_RGB):
    w = bpy.data.worlds.new('ThrTopWorld')
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

# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------
DENDRO_VERTICAL_COLOR = (0.62, 0.78, 0.98, 1.0)


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


def threshold_cutout_max_semi_major():
    """Cap cutout ellipse size so the root / large hulls do not erase the whole grid."""
    if not POINTS:
        return THRESHOLD_CUTOUT_MAX_SEMI_MIN
    xs = [p[0] for p in POINTS]
    ys = [p[1] for p in POINTS]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.12)
    lim = THRESHOLD_CUTOUT_MAX_SEMI_FRAC * span + 0.1
    return max(THRESHOLD_CUTOUT_MAX_SEMI_MIN, min(THRESHOLD_CUTOUT_MAX_SEMI_CAP, lim))


def make_threshold_grid_cutout_material(name, circle_meta, cutout_max_semi_major, n_leaves):
    """Procedural brick grid on the slice; transparent holes only near merge Z and capped in XY."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    def place(node, x, y):
        node.location = (x, y)
        return node

    out = place(nodes.new('ShaderNodeOutputMaterial'), 1400, 0)
    mix_slice = place(nodes.new('ShaderNodeMixShader'), 1180, 0)
    mix_slice.name = 'Mix Shader'
    _fslice = mix_slice.inputs.get('Fac') or mix_slice.inputs[0]
    _fslice.default_value = 0.0
    transp_out = place(nodes.new('ShaderNodeBsdfTransparent'), 1000, -140)
    _sout = transp_out.outputs.get('BSDF') or transp_out.outputs[0]
    links.new(_sout, mix_slice.inputs[1])

    mix_cut = place(nodes.new('ShaderNodeMixShader'), 1000, 80)
    transp_hole = place(nodes.new('ShaderNodeBsdfTransparent'), 820, 20)
    emit_grid = place(nodes.new('ShaderNodeEmission'), 820, 160)
    emit_grid.inputs['Strength'].default_value = THRESHOLD_PLANE_EMISSION_STRENGTH * 1.12
    _eout = emit_grid.outputs.get('Emission') or emit_grid.outputs[0]
    links.new(_eout, mix_cut.inputs[1])
    _tout = transp_hole.outputs.get('BSDF') or transp_hole.outputs[0]
    links.new(_tout, mix_cut.inputs[2])
    _mcout = mix_cut.outputs.get('Shader') or mix_cut.outputs[0]
    links.new(_mcout, mix_slice.inputs[2])

    # Cutouts compare world Z to merge Z. Plain Position/Geometry often stays in object space (Z≈0 on the
    # plane), which kills the Z gate — use Object coords → Vector Transform → World when available.
    texcoord = place(nodes.new('ShaderNodeTexCoord'), -520, 220)
    try:
        _obj_sock = texcoord.outputs['Object']
    except (KeyError, IndexError, TypeError, AttributeError):
        _obj_sock = None
    wpos_out = None
    if hasattr(bpy.types, 'ShaderNodeVectorTransform') and _obj_sock is not None:
        vt = place(nodes.new('ShaderNodeVectorTransform'), -320, 220)
        if hasattr(vt, 'vector_type'):
            try:
                vt.vector_type = 'POINT'
            except TypeError:
                pass
        for _attr, _val in (('convert_from', 'OBJECT'), ('convert_to', 'WORLD')):
            if hasattr(vt, _attr):
                try:
                    setattr(vt, _attr, _val)
                except TypeError:
                    pass
        _vtin = vt.inputs.get('Vector') or vt.inputs[0]
        links.new(_obj_sock, _vtin)
        wpos_out = vt.outputs.get('Vector') or vt.outputs[0]
    if wpos_out is None and hasattr(bpy.types, 'ShaderNodePosition'):
        posn = place(nodes.new('ShaderNodePosition'), -200, 220)
        wpos_out = posn.outputs.get('Vector') or posn.outputs[0]
    if wpos_out is None:
        geo = place(nodes.new('ShaderNodeNewGeometry'), -200, 220)
        wpos_out = geo.outputs.get('Position') or geo.outputs[0]

    sep = place(nodes.new('ShaderNodeSeparateXYZ'), 0, 220)
    _vin = sep.inputs.get('Vector') or sep.inputs[0]
    links.new(wpos_out, _vin)

    comb_xy = place(nodes.new('ShaderNodeCombineXYZ'), 200, 220)
    links.new(sep.outputs['X'], comb_xy.inputs['X'])
    links.new(sep.outputs['Y'], comb_xy.inputs['Y'])
    comb_xy.inputs['Z'].default_value = 0.0

    # Blender builds differ: legacy ``ShaderNodeTexBrick``, 3.x ``ShaderNodeBrickTexture``, or Checker.
    grid_tex = None
    for bt in ('ShaderNodeTexBrick', 'ShaderNodeBrickTexture', 'ShaderNodeTexChecker'):
        try:
            grid_tex = place(nodes.new(bt), 400, 220)
            break
        except RuntimeError:
            continue
    if grid_tex is None:
        raise RuntimeError(
            'No grid texture node (ShaderNodeTexBrick / ShaderNodeBrickTexture / ShaderNodeTexChecker)'
        )
    _bin = grid_tex.inputs.get('Vector') or grid_tex.inputs[0]
    _cout = comb_xy.outputs.get('Vector') or comb_xy.outputs[0]
    links.new(_cout, _bin)
    if 'Scale' in grid_tex.inputs:
        grid_tex.inputs['Scale'].default_value = THRESHOLD_GRID_BRICK_SCALE
    if 'Mortar Size' in grid_tex.inputs:
        grid_tex.inputs['Mortar Size'].default_value = THRESHOLD_GRID_MORTAR_SIZE
    elif 'Mortar' in grid_tex.inputs:
        grid_tex.inputs['Mortar'].default_value = THRESHOLD_GRID_MORTAR_SIZE
    sc = THRESHOLD_SLICE_COLOR
    c1 = (
        min(1.0, sc[0] * THRESHOLD_GRID_LINE_BRIGHT),
        min(1.0, sc[1] * THRESHOLD_GRID_LINE_BRIGHT),
        min(1.0, sc[2] * THRESHOLD_GRID_LINE_BRIGHT),
        1.0,
    )
    c2 = (
        sc[0] * THRESHOLD_GRID_CELL_DIM,
        sc[1] * THRESHOLD_GRID_CELL_DIM,
        sc[2] * THRESHOLD_GRID_CELL_DIM,
        1.0,
    )
    if 'Color1' in grid_tex.inputs:
        grid_tex.inputs['Color1'].default_value = c1
        grid_tex.inputs['Color2'].default_value = c2
    _gcol = grid_tex.outputs.get('Color') or grid_tex.outputs[0]
    _ecol = emit_grid.inputs.get('Color') or emit_grid.inputs[0]
    links.new(_gcol, _ecol)

    sigma = THRESHOLD_CUTOUT_Z_SIGMA
    esoft = max(THRESHOLD_CUTOUT_ELLIPSE_SOFT, 1e-4)

    hole_acc_out = None
    y0 = -200
    ci = 0
    for item in circle_meta:
        if int(item.get('n_members', 0)) >= int(n_leaves) and n_leaves > 1:
            continue
        cx = float(item['center'][0])
        cy = float(item['center'][1]) + DATA_Y_OFFSET
        zm = float(item['z'])
        theta = float(item['theta'])
        r = float(item['radius'])
        s0x = float(item.get('s0x', 1.0))
        s0y = float(item.get('s0y', 1.0))
        ax = max(r * s0x * THRESHOLD_CUTOUT_RADIUS_PAD, 1e-4)
        ay = max(r * s0y * THRESHOLD_CUTOUT_RADIUS_PAD, 1e-4)
        m_ax = max(ax, ay)
        if m_ax > cutout_max_semi_major:
            s = cutout_max_semi_major / m_ax
            ax *= s
            ay *= s
        ct = math.cos(theta)
        st = math.sin(theta)
        ox = -400
        oy = y0 - ci * 200
        ci += 1

        def V(val, dx=0, dy=0):
            v = nodes.new('ShaderNodeValue')
            place(v, ox + dx, oy + dy)
            v.outputs[0].default_value = float(val)
            return v.outputs[0]

        def M(op, x, y):
            m = nodes.new('ShaderNodeMath')
            m.operation = op
            place(m, x, y)
            return m

        v_cx = V(cx, 0, 0)
        v_cy = V(cy, 0, -40)
        v_zm = V(zm, 0, -80)
        v_ax = V(ax, 0, -120)
        v_ay = V(ay, 0, -160)
        v_ct = V(ct, 0, -200)
        v_st = V(st, 0, -240)
        v_sigma = V(sigma, 0, -280)
        v_one = V(1.0, 0, -320)
        v_soft = V(esoft, 0, -360)
        v_zero = V(0.0, 0, -400)

        m_dpx = M('SUBTRACT', ox + 220, oy)
        links.new(sep.outputs['X'], m_dpx.inputs[0])
        links.new(v_cx, m_dpx.inputs[1])
        m_dpy = M('SUBTRACT', ox + 220, oy - 50)
        links.new(sep.outputs['Y'], m_dpy.inputs[0])
        links.new(v_cy, m_dpy.inputs[1])

        t1 = M('MULTIPLY', ox + 400, oy)
        links.new(m_dpx.outputs[0], t1.inputs[0])
        links.new(v_ct, t1.inputs[1])
        t2 = M('MULTIPLY', ox + 400, oy - 40)
        links.new(m_dpy.outputs[0], t2.inputs[0])
        links.new(v_st, t2.inputs[1])
        dx = M('ADD', ox + 580, oy)
        links.new(t1.outputs[0], dx.inputs[0])
        links.new(t2.outputs[0], dx.inputs[1])

        t3 = M('MULTIPLY', ox + 400, oy - 100)
        links.new(m_dpx.outputs[0], t3.inputs[0])
        links.new(v_st, t3.inputs[1])
        neg_st = M('MULTIPLY', ox + 400, oy - 140)
        links.new(t3.outputs[0], neg_st.inputs[0])
        neg_st.inputs[1].default_value = -1.0
        t4 = M('MULTIPLY', ox + 400, oy - 180)
        links.new(m_dpy.outputs[0], t4.inputs[0])
        links.new(v_ct, t4.inputs[1])
        dy = M('ADD', ox + 580, oy - 120)
        links.new(neg_st.outputs[0], dy.inputs[0])
        links.new(t4.outputs[0], dy.inputs[1])

        ndx = M('DIVIDE', ox + 760, oy)
        links.new(dx.outputs[0], ndx.inputs[0])
        links.new(v_ax, ndx.inputs[1])
        ndy = M('DIVIDE', ox + 760, oy - 50)
        links.new(dy.outputs[0], ndy.inputs[0])
        links.new(v_ay, ndy.inputs[1])
        sx2 = M('MULTIPLY', ox + 940, oy)
        links.new(ndx.outputs[0], sx2.inputs[0])
        links.new(ndx.outputs[0], sx2.inputs[1])
        sy2 = M('MULTIPLY', ox + 940, oy - 50)
        links.new(ndy.outputs[0], sy2.inputs[0])
        links.new(ndy.outputs[0], sy2.inputs[1])
        ell = M('ADD', ox + 1120, oy - 25)
        links.new(sx2.outputs[0], ell.inputs[0])
        links.new(sy2.outputs[0], ell.inputs[1])

        m_dz = M('SUBTRACT', ox + 220, oy - 260)
        links.new(sep.outputs['Z'], m_dz.inputs[0])
        links.new(v_zm, m_dz.inputs[1])
        adz = M('ABSOLUTE', ox + 400, oy - 260)
        links.new(m_dz.outputs[0], adz.inputs[0])
        rat = M('DIVIDE', ox + 580, oy - 260)
        links.new(adz.outputs[0], rat.inputs[0])
        links.new(v_sigma, rat.inputs[1])
        rsq = M('MULTIPLY', ox + 760, oy - 260)
        links.new(rat.outputs[0], rsq.inputs[0])
        links.new(rat.outputs[0], rsq.inputs[1])
        den = M('ADD', ox + 940, oy - 260)
        links.new(v_one, den.inputs[0])
        links.new(rsq.outputs[0], den.inputs[1])
        gate = M('DIVIDE', ox + 1120, oy - 260)
        links.new(v_one, gate.inputs[0])
        links.new(den.outputs[0], gate.inputs[1])

        one_p_s = M('ADD', ox + 760, oy - 340)
        links.new(v_one, one_p_s.inputs[0])
        links.new(v_soft, one_p_s.inputs[1])
        num_e = M('SUBTRACT', ox + 940, oy - 340)
        links.new(one_p_s.outputs[0], num_e.inputs[0])
        links.new(ell.outputs[0], num_e.inputs[1])
        in_raw = M('DIVIDE', ox + 1120, oy - 340)
        links.new(num_e.outputs[0], in_raw.inputs[0])
        links.new(v_soft, in_raw.inputs[1])
        in_hi = M('MINIMUM', ox + 1300, oy - 340)
        links.new(in_raw.outputs[0], in_hi.inputs[0])
        links.new(v_one, in_hi.inputs[1])
        in_mask = M('MAXIMUM', ox + 1480, oy - 340)
        links.new(in_hi.outputs[0], in_mask.inputs[0])
        links.new(v_zero, in_mask.inputs[1])

        hole_i = M('MULTIPLY', ox + 1660, oy - 300)
        links.new(gate.outputs[0], hole_i.inputs[0])
        links.new(in_mask.outputs[0], hole_i.inputs[1])

        if hole_acc_out is None:
            hole_acc_out = hole_i.outputs[0]
        else:
            mx = M('MAXIMUM', ox + 1840, oy - 300)
            links.new(hole_acc_out, mx.inputs[0])
            links.new(hole_i.outputs[0], mx.inputs[1])
            hole_acc_out = mx.outputs[0]

    if hole_acc_out is None:
        vh = nodes.new('ShaderNodeValue')
        place(vh, 820, -80)
        vh.outputs[0].default_value = 0.0
        hole_acc_out = vh.outputs[0]
    v_hole_cap = nodes.new('ShaderNodeValue')
    place(v_hole_cap, 900, -60)
    v_hole_cap.outputs[0].default_value = float(THRESHOLD_CUTOUT_MAX_DEPTH)
    m_hole_cap = nodes.new('ShaderNodeMath')
    m_hole_cap.operation = 'MINIMUM'
    place(m_hole_cap, 1020, -40)
    links.new(hole_acc_out, m_hole_cap.inputs[0])
    links.new(v_hole_cap.outputs[0], m_hole_cap.inputs[1])
    _fcut = mix_cut.inputs.get('Fac') or mix_cut.inputs[0]
    links.new(m_hole_cap.outputs[0], _fcut)

    _msout = mix_slice.outputs.get('Shader') or mix_slice.outputs[0]
    links.new(_msout, out.inputs[0])
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


mat_dendro = make_principled_material(
    'DendroMatThrTop', DENDRO_VERTICAL_COLOR, roughness=0.12, emission_strength=0.55
)
mat_ground = make_principled_material('GroundMatThrTop', (0.03, 0.035, 0.045, 1.0), roughness=0.75)
mat_sky_dome = make_principled_material(
    'SkyDomeMatThrTop', (0.0, 0.0, 0.0, 1.0), roughness=1.0, alpha=1.0, emission_strength=0.0
)
mat_sky_dome.blend_method = 'OPAQUE'

# -----------------------------------------------------------------------------
# Mesh / curve helpers
# -----------------------------------------------------------------------------
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
    pts = []
    for i in range(n):
        phi = 2 * math.pi * i / n
        pts.append((a * math.cos(phi), b * math.sin(phi), z))
    return pts


# -----------------------------------------------------------------------------
# Clustering (same as dendrogram.py)
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
        ca, cb = centroid(a_inds, pts), centroid(b_inds, pts)
        return euclid(ca, cb)
    raise ValueError(f'Unknown linkage: {linkage}')


def agglomerative(points, linkage='single'):
    nodes = {}
    active = []
    for i, _ in enumerate(points):
        nodes[i] = ClusterNode(i, [i], height=0.0)
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
    return nodes, active[0], merges


def merge_height_min_gap(pts, linkage):
    nodes, _, merge_ids = agglomerative(pts, linkage=linkage)
    if not merge_ids:
        return -1.0
    hs = sorted(nodes[mid].height for mid in merge_ids)
    gaps = [hs[i + 1] - hs[i] for i in range(len(hs) - 1)]
    return min(gaps) if gaps else 0.0


def min_pairwise_x_gap(pts):
    xs = sorted(p[0] for p in pts)
    if len(xs) < 2:
        return 1e18
    return min(xs[i + 1] - xs[i] for i in range(len(xs) - 1))


def build_even_merge_points(n, linkage, steps=12000, seed=42):
    random.seed(seed)
    enforce_x_gap = n >= 9
    span = 2.35 if n <= 8 else max(3.85, MIN_X_PAIR_GAP * (n - 1) * 0.52)
    sigma = 0.11 if n <= 8 else 0.17
    y_lim = span * 0.58 + 0.35
    x_lim = span + 0.5
    if n == 8:
        best = [
            (-2.05, 0.05), (-1.72, -0.42), (-1.92, 0.52), (-1.48, -0.18),
            (1.48, 0.08), (1.92, -0.38), (1.72, 0.55), (2.02, -0.12),
        ]
    elif enforce_x_gap:
        denom = max(1, n - 1)
        xs_seed = [-span + (2 * span * i / denom) for i in range(n)]
        random.shuffle(xs_seed)
        best = [(xs_seed[i], random.uniform(-span * 0.55, span * 0.55)) for i in range(n)]
    else:
        best = [(random.uniform(-span, span), random.uniform(-span * 0.55, span * 0.55)) for _ in range(n)]
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
POINTS = [p for i, p in enumerate(_raw_points) if i not in POINT_EXCLUDE_INDICES]
for _pi in PURPLE_LEAF_INDICES:
    if _pi < 0 or _pi >= len(POINTS):
        raise ValueError(
            f'PURPLE_LEAF_INDICES contains {_pi} but len(POINTS)={len(POINTS)}'
        )


def leaf_cluster_labels(node_id, nodes, thr, counter):
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
    if node.left is None or node.right is None:
        return False
    if len(node.members) < 2:
        return False
    return node.height > 1e-12


def threshold_sweep_ends(max_h, z_top_floor=None):
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
    zs = [float(cluster_geom[nid]["z"]) for nid, node in nodes.items() if len(node.members) >= 2]
    return max(zs) if zs else float(DENDRO_BASE_Z)


def threshold_plane_z_for_sweep_frame(f, z_top, z_bot):
    """Same piecewise Z as ``dendrogram.py`` (uses F_THRESHOLD_* aliases)."""
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


def _smoothstep01(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def merge_oval_bold_weight(merge_nid, th, nodes, parent_of, max_h):
    """How much to emphasize a merge oval at dendrogram cut height ``th``.

    A ring is bold when its merge has occurred (``th`` at/above that height) but the cluster is
    not yet absorbed into the parent merge — so every *current* cluster (not only the root) can
    read as bold at once. Root uses no upper cap.
    """
    h_m = float(nodes[merge_nid].height)
    band = max(float(max_h) * OVAL_BOLD_EDGE_FRAC, 1e-9)
    p = parent_of.get(merge_nid)
    if p is None:
        if th < h_m - band:
            return 0.0
        if th < h_m + band:
            return _smoothstep01((th - (h_m - band)) / (2.0 * band))
        return 1.0
    h_p = float(nodes[p].height)
    if th < h_m - band:
        return 0.0
    if th < h_m + band:
        return _smoothstep01((th - (h_m - band)) / (2.0 * band))
    if th < h_p - band:
        return 1.0
    if th < h_p + band:
        return _smoothstep01(((h_p + band) - th) / (2.0 * band))
    return 0.0


def keyframe_points_y_settle(point_objs, points_xy, f_start, f_end):
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


def _force_constant_hide_fcurves(action):
    if action is None:
        return
    for fc in action.fcurves:
        if fc.data_path in ('hide_viewport', 'hide_render'):
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'


def _random_leaf_color(leaf_index):
    rng = random.Random(LEAF_COLOR_RANDOM_SEED + leaf_index * 10007)
    h = rng.random()
    r, g, b = colorsys.hsv_to_rgb(h, CLUSTER_COLOR_S, CLUSTER_COLOR_V)
    return (float(r), float(g), float(b), 1.0)


def _subtree_contains_any_purple_leaf(node_id, nodes, purple_leaf_indices):
    return any(m in purple_leaf_indices for m in nodes[node_id].members)


def compute_node_colors_majority(nodes, purple_leaf_indices):
    """Leaf colors: purple set or deterministic random; internal = larger child; ties favor purple if present."""
    n_leaves = len(POINTS)
    colors = {}
    for i in range(n_leaves):
        if i in purple_leaf_indices:
            colors[i] = tuple(CLUSTER_PURPLE)
        else:
            colors[i] = _random_leaf_color(i)
    internal_ids = [nid for nid, n in nodes.items() if len(n.members) >= 2]
    internal_ids.sort(key=lambda nid: nodes[nid].height)
    for nid in internal_ids:
        node = nodes[nid]
        L, R = node.left, node.right
        nl = len(nodes[L].members)
        nr = len(nodes[R].members)
        if nl > nr:
            colors[nid] = colors[L]
        elif nr > nl:
            colors[nid] = colors[R]
        else:
            lp = _subtree_contains_any_purple_leaf(L, nodes, purple_leaf_indices)
            rp = _subtree_contains_any_purple_leaf(R, nodes, purple_leaf_indices)
            if lp and not rp:
                colors[nid] = colors[L]
            elif rp and not lp:
                colors[nid] = colors[R]
            else:
                colors[nid] = colors[L]
    return colors


def cluster_rep_node_at_th(leaf_idx, th, nodes, parent_of):
    """Deepest ancestor of leaf whose merge height is <= th (same cut as leaf_cluster_labels at thr)."""
    nid = leaf_idx
    while True:
        p = parent_of.get(nid)
        if p is None:
            return nid
        if nodes[p].height <= th:
            nid = p
        else:
            return nid


def camera_euler_look_at(cam_pos, target, up_axis='Y'):
    d = target - cam_pos
    if d.length < 1e-8:
        return mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')
    return d.to_track_quat('-Z', up_axis).to_euler('XYZ')


def dendrogram_scene_focus(nodes, cluster_geom):
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
        (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.5 * (min(zs) + max(zs)))
    )


def dendrogram_orbit_pivot(nodes, cluster_geom):
    fc = dendrogram_scene_focus(nodes, cluster_geom)
    npt = len(POINTS)
    if npt == 0:
        return fc
    mx = sum(p[0] for p in POINTS) / npt
    my = sum(p[1] + DATA_Y_OFFSET for p in POINTS) / npt
    return Vector((mx, my, fc.z))


def perspective_lens_mm_match_span(cam_data, target_span_world, subject_distance_world):
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


def _ensure_camera_actions(camera):
    if camera.animation_data is None:
        camera.animation_data_create()
    if camera.data.animation_data is None:
        camera.data.animation_data_create()


def _commit_persp_camera_keyframe(camera, frame, cam_pivot, target_span_world):
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


def threshold_slice_xy_scale_and_center(points, mesh_plane_size):
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


# -----------------------------------------------------------------------------
# Build scene graph
# -----------------------------------------------------------------------------
nodes, root_id, merge_ids = agglomerative(POINTS, linkage=LINKAGE)
max_h = max(nodes[mid].height for mid in merge_ids) if merge_ids else 1.0

cluster_geom = {}
for nid, node in nodes.items():
    pts = [POINTS[i] for i in node.members]
    if len(node.members) == 1:
        cx, cy = pts[0][0], pts[0][1]
        cluster_geom[nid] = {
            "center": (float(cx), float(cy)),
            "radius": 0.18,
            "z": float(DENDRO_BASE_Z),
        }
    else:
        lg = cluster_geom[node.left]
        rg = cluster_geom[node.right]
        lx, ly = lg["center"]
        rx, ry = rg["center"]
        cx, cy = (lx + rx) * 0.5, (ly + ry) * 0.5
        z = DENDRO_BASE_Z + (node.height / max_h) * HEIGHT_SCALE * 4.0
        w = Vector((rx - lx, ry - ly))
        d_lr = w.length
        u = w.normalized() if d_lr > 1e-9 else Vector((1.0, 0.0))
        pad = 0.25
        r_chord_half = 0.5 * d_lr
        r_hull = max(math.hypot(POINTS[i][0] - cx, POINTS[i][1] - cy) for i in node.members) + pad
        r_circ = max(r_chord_half, r_hull, 1e-4)
        theta = math.atan2(u.y, u.x)
        cluster_geom[nid] = {
            "center": (float(cx), float(cy)),
            "radius": float(r_circ),
            "z": float(z),
            "theta": float(theta),
            "a_col": float(max(r_chord_half, 1e-4)),
            "b_col": float(OVAL_COLLAPSE_MINOR),
        }

parent_of = {}
for nid, node in nodes.items():
    if node.left is not None:
        parent_of[node.left] = nid
        parent_of[node.right] = nid

NODE_COLORS = compute_node_colors_majority(nodes, PURPLE_LEAF_INDICES)

col_points = ensure_collection('Points')
col_dendro = ensure_collection('Dendrogram')
col_circles = ensure_collection('ClusterCircles')
col_guides = ensure_collection('Guides')

cam_pivot = dendrogram_orbit_pivot(nodes, cluster_geom)
top_loc = cam_pivot + Vector((0.0, 0.0, TOP_CAMERA_Z_OFFSET))
_rx_end = math.radians(THRESHOLD_ORBIT_RX_END_DEG)
_H = float((top_loc - cam_pivot).z)
_arm_end = mathutils.Matrix.Rotation(_rx_end, 3, 'X') @ Vector((0.0, 0.0, _H))
diag_threshold_loc = cam_pivot + _arm_end * THRESHOLD_CAM_DISTANCE_SCALE
_diag_e = camera_euler_look_at(diag_threshold_loc, cam_pivot)
diag_threshold_rot = (float(_diag_e.x), float(_diag_e.y), float(_diag_e.z))

bpy.ops.object.camera_add(location=diag_threshold_loc.to_tuple())
camera = bpy.context.active_object
camera.name = 'MainCamera'
scene.camera = camera
camera.rotation_euler = diag_threshold_rot
_ensure_camera_actions(camera)
_commit_persp_camera_keyframe(camera, F_START, cam_pivot, ORBIT_ORTHO_SCALE)
_commit_persp_camera_keyframe(camera, F_END, cam_pivot, ORBIT_ORTHO_SCALE)
_linearize_camera_segment(camera, F_START, F_END)

ground = add_plane('Ground', (0, 0, -0.02), GROUND_PLANE_SIZE, mat_ground)
for c in list(ground.users_collection):
    c.objects.unlink(ground)
col_guides.objects.link(ground)

sky_dome = add_dark_sky_dome('SkyDome', cam_pivot, SKY_DOME_RADIUS, mat_sky_dome)
for c in list(sky_dome.users_collection):
    c.objects.unlink(sky_dome)
col_guides.objects.link(sky_dome)

point_objects = []
point_materials = []
for i in range(len(POINTS)):
    x, y = POINTS[i]
    pm = make_principled_material(
        f'PointMatThrTop_{i}', POINT_WHITE, roughness=0.25, emission_strength=POINT_EMISSION
    )
    point_materials.append(pm)
    obj = add_uv_sphere(f'P{i}', (x, DATA_Y_OFFSET, DATA_Z), POINT_RADIUS, pm)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col_points.objects.link(obj)
    point_objects.append(obj)

dendro_objects = []
threshold_wipe_z_pairs = []
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
        conn1 = new_curve_object(
            f'ConnL_{nid}',
            [(lx, wy_l, lz), (lx, wy_l, z)],
            bevel=CURVE_BEVEL * DENDRO_CURVE_BEVEL_SCALE,
            material=mat_dendro,
        )
        conn2 = new_curve_object(
            f'ConnR_{nid}',
            [(rx, wy_r, rz), (rx, wy_r, z)],
            bevel=CURVE_BEVEL * DENDRO_CURVE_BEVEL_SCALE,
            material=mat_dendro,
        )
        for obj in (conn1, conn2):
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            col_dendro.objects.link(obj)
            dendro_objects.append(obj)
            threshold_wipe_z_pairs.append((obj, float(z)))
        dendro_curve_y_specs.append({'obj': conn1, 'x': lx, 'y_data': ly, 'z0': lz, 'z1': z})
        dendro_curve_y_specs.append({'obj': conn2, 'x': rx, 'y_data': ry, 'z0': rz, 'z1': z})
        pid = parent_of.get(nid)
        if pid is not None:
            z_parent = cluster_geom[pid]["z"]
            wy_p = cy + DATA_Y_OFFSET
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
    r_circ = float(geom["radius"])
    theta = float(geom["theta"])
    a_col = float(geom["a_col"])
    b_col = float(geom["b_col"])
    circle_mat = make_emission_material(
        f'OvalMatThrTop_{nid}',
        CLUSTER_PURPLE,
        strength=2.0,
        alpha=0.42,
    )
    circle_obj = new_curve_object(
        f'Oval_{nid}',
        ellipse_points_local(r_circ, r_circ, 0.0, n=100),
        bevel=CIRCLE_BEVEL,
        material=circle_mat,
        cyclic=True,
    )
    circle_obj.location = (cx, cy + DATA_Y_OFFSET, z)
    circle_obj.rotation_euler = (0.0, 0.0, theta)
    s0x = min(1.0, a_col / max(r_circ, 1e-9))
    s0y = min(1.0, b_col / max(r_circ, 1e-9))
    circle_obj.scale = (s0x, s0y, 1.0)
    for c in list(circle_obj.users_collection):
        c.objects.unlink(circle_obj)
    col_circles.objects.link(circle_obj)
    circle_by_nid[nid] = circle_obj
    lx, ly = cluster_geom[node.left]["center"]
    rx, ry = cluster_geom[node.right]["center"]
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
        "s0x": s0x,
        "s0y": s0y,
        "n_members": len(node.members),
    })
    threshold_wipe_z_pairs.append((circle_obj, float(z)))

keyframe_points_y_settle(point_objects, POINTS, F_START, F_START)
keyframe_dendro_connectors_y_settle(dendro_curve_y_specs, F_START, F_START)
keyframe_merge_ovals_follow_vertical_tips(circle_meta, F_START, F_START)
for _item in circle_meta:
    _o = _item['obj']
    if _o.animation_data is None:
        _o.animation_data_create()
    _o.scale = (1.0, 1.0, 1.0)
    _o.keyframe_insert(data_path='scale', frame=F_START)
    if _o.animation_data.action:
        _linearize_action_range(_o.animation_data.action, F_START, F_START)

_z_mmax = merge_levels_z_max(nodes, cluster_geom)
_threshold_z_top_build, _threshold_z_bot_build = threshold_sweep_ends(
    max_h, z_top_floor=_z_mmax + THRESHOLD_ABOVE_TOP_MERGE_Z
)
_th_plane_xy, _th_cx, _th_cy = threshold_slice_xy_scale_and_center(POINTS, PLANE_SIZE)
mat_threshold = make_threshold_grid_cutout_material(
    'ThresholdWireMatThrTop',
    circle_meta,
    threshold_cutout_max_semi_major(),
    len(POINTS),
)
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

# -----------------------------------------------------------------------------
# Threshold sweep + majority-merge point colors
# -----------------------------------------------------------------------------
n_pts = len(point_materials)


def set_point_colors_majority(th):
    for li in range(n_pts):
        rep = cluster_rep_node_at_th(li, th, nodes, parent_of)
        col = NODE_COLORS[rep]
        mat = point_materials[li]
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = col
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = col


def animate_threshold_sweep_top():
    z_mmax = merge_levels_z_max(nodes, cluster_geom)
    z_top, z_bot = threshold_sweep_ends(max_h, z_top_floor=z_mmax + THRESHOLD_ABOVE_TOP_MERGE_Z)
    ps = _th_plane_xy
    tw = threshold_wire

    for mat in point_materials:
        if mat.node_tree.animation_data is None:
            mat.node_tree.animation_data_create()

    if tw.animation_data is None:
        tw.animation_data_create()

    # Initial colors: cut at h≈0 → per-leaf hues; purple leaves share CLUSTER_PURPLE
    set_point_colors_majority(0.0)
    for mat in point_materials:
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=F_START)
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].keyframe_insert(data_path='default_value', frame=F_START)

    if mat_threshold.node_tree.animation_data is None:
        mat_threshold.node_tree.animation_data_create()
    mix_node = mat_threshold.node_tree.nodes.get('Mix Shader')
    if mix_node:
        fac_m = mix_node.inputs.get('Fac') or mix_node.inputs[0]
        fac_m.default_value = 0.0
        fac_m.keyframe_insert(data_path='default_value', frame=F_START)
        fac_m.default_value = THRESHOLD_PLANE_MIX
        fac_m.keyframe_insert(data_path='default_value', frame=F_SWEEP_SLICE_ON)
        fac_m.default_value = THRESHOLD_PLANE_MIX
        fac_m.keyframe_insert(data_path='default_value', frame=F_SWEEP_TOP_AGAIN)
        fac_m.default_value = 0.0
        fac_m.keyframe_insert(data_path='default_value', frame=F_END)

    tw.hide_viewport = True
    tw.hide_render = True
    tw.keyframe_insert(data_path='hide_viewport', frame=F_START)
    tw.keyframe_insert(data_path='hide_render', frame=F_START)
    tw.hide_viewport = False
    tw.hide_render = False
    tw.keyframe_insert(data_path='hide_viewport', frame=F_SWEEP_SLICE_ON)
    tw.keyframe_insert(data_path='hide_render', frame=F_SWEEP_SLICE_ON)

    tw.location.z = z_top
    tw.scale = (ps, ps, 1.0)
    tw.keyframe_insert(data_path='location', frame=F_SWEEP_SLICE_ON - 1)
    tw.keyframe_insert(data_path='scale', frame=F_SWEEP_SLICE_ON - 1)
    tw.location.z = z_top
    tw.keyframe_insert(data_path='location', frame=F_SWEEP_SLICE_ON)
    tw.keyframe_insert(data_path='scale', frame=F_SWEEP_SLICE_ON)

    tw.location.z = z_bot
    tw.keyframe_insert(data_path='location', frame=F_SWEEP_BOTTOM)
    tw.keyframe_insert(data_path='scale', frame=F_SWEEP_BOTTOM)

    tw.location.z = z_top
    tw.keyframe_insert(data_path='location', frame=F_SWEEP_TOP_AGAIN)
    tw.keyframe_insert(data_path='scale', frame=F_SWEEP_TOP_AGAIN)

    base_b = CIRCLE_BEVEL
    peak_b = CIRCLE_BEVEL * OVAL_BEVEL_PEAK_SCALE

    for item in circle_meta:
        curve = item["obj"].data
        if curve.animation_data is None:
            curve.animation_data_create()
        curve.bevel_depth = base_b
        curve.keyframe_insert(data_path='bevel_depth', frame=F_SWEEP_SLICE_ON - 1)

    for obj, _ in threshold_wipe_z_pairs:
        if obj.animation_data is None:
            obj.animation_data_create()

    for obj, _ in threshold_wipe_z_pairs:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_viewport', frame=F_SWEEP_SLICE_ON - 1)
        obj.keyframe_insert(data_path='hide_render', frame=F_SWEEP_SLICE_ON - 1)

    vis_eps = 1e-4
    sweep_frames = list(range(F_SWEEP_SLICE_ON, F_END + 1))
    for f in sweep_frames:
        z = threshold_plane_z_for_sweep_frame(f, z_top, z_bot)
        tw.location.z = z
        tw.keyframe_insert(data_path='location', frame=f)
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
            w = merge_oval_bold_weight(item["nid"], th, nodes, parent_of, max_h)
            curve.bevel_depth = base_b + (peak_b - base_b) * w
            curve.keyframe_insert(data_path='bevel_depth', frame=f)
        set_point_colors_majority(th)
        for mat in point_materials:
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=f)
                if 'Emission Color' in bsdf.inputs:
                    bsdf.inputs['Emission Color'].keyframe_insert(data_path='default_value', frame=f)

    for item in circle_meta:
        curve = item["obj"].data
        curve.bevel_depth = base_b
        curve.keyframe_insert(data_path='bevel_depth', frame=F_END)

    tw.hide_viewport = True
    tw.hide_render = True
    tw.keyframe_insert(data_path='hide_viewport', frame=F_END)
    tw.keyframe_insert(data_path='hide_render', frame=F_END)

    for obj, _ in threshold_wipe_z_pairs:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path='hide_viewport', frame=F_END)
        obj.keyframe_insert(data_path='hide_render', frame=F_END)

    neutral = (0.52, 0.52, 0.55, 1.0)
    for mat in point_materials:
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = neutral
            bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=F_END)
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = neutral
                bsdf.inputs['Emission Color'].keyframe_insert(data_path='default_value', frame=F_END)

    for obj, _z in threshold_wipe_z_pairs:
        if obj.animation_data and obj.animation_data.action:
            _force_constant_hide_fcurves(obj.animation_data.action)
    if tw.animation_data and tw.animation_data.action:
        _force_constant_hide_fcurves(tw.animation_data.action)


animate_threshold_sweep_top()

scene.frame_set(F_START)
ground.hide_render = True

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.region_3d.view_perspective = 'CAMERA'
                space.lock_camera = False
                break
