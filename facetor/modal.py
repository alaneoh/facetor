# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Alanéo Houssart
#
# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ***** END GPL LICENSE BLOCK *****

"""The interactive Facetor operator (Alt+B), its status bar and menu entry."""

import bpy
import bmesh
import mathutils.kdtree
from mathutils import Vector

from .attributes import Keeper, triangulate_ngons
from .corners import (bevel_network, has_bevel_poles, measure_shape, network_collapse, network_rebuild,
                      network_scale, square_in_corners)
from .core import (
    AXES, _face_components, apply_profile, direction_stats, edge_between, grid_profiles, profile_reduce,
    ordered,
)
from .dome import dome_cut_counts, dome_cut_rebuild, dome_cut_setup
from .grid import (
    _subgrid, collapse_tiny_edges, collapse_to_corners, execute_plan, extend_t_columns, merge_caps,
    plan_grid, resample_bands, tiny_edges,
)
from .holes import hole_rebuild, selected_loops, tube_loop
from .outline import outline_arc_rebuild, outline_arcs
from .recess import (
    RECESS_KEY,
    _clip_cache,
    _with_clip_cache,
    recess_counts,
    revolution_clip_count,
    revolution_clip_profile_count,
    revolution_clip_rebuild,
)
from .revolution import (
    revolution_count,
    revolution_cut_count,
    revolution_cut_profile_count,
    revolution_cut_rebuild,
    revolution_cut_rings,
    revolution_profile_count,
    revolution_rebuild,
    revolution_rings,
)
from .safety import _restore, guarded
from .selection import analyse, band_core, find_round_part, gather_faces, grid_faces_of, tri_count
from .strips import (
    STRIPS_TOUCH,
    chain_only_check,
    chain_only_columns,
    chain_only_rebuild,
    edge_chain_region,
    stacked_rebuild,
    stacked_strips,
    strip_check,
    strip_rebuild,
)


# ---------------------------------------------------------------------------
# Operator: selection
# ---------------------------------------------------------------------------

def _no_width(op, what):
    """Header error when a width is asked for in a mode that can't change it, else None."""
    if abs(getattr(op, "width", 1.0) - 1.0) < 1e-6:
        return None
    return "The width can't change on %s: set it back to 1." % what


def _recess_change(op, bm, me, sel, infos):
    """Part of revolution with a recess: (err, did, infos), or None if it isn't one."""
    rc = recess_counts(me, bm, sel)
    if rc:
        werr = _no_width(op, "a part of revolution with a recess")
        if werr:
            return werr, False, infos
        sides_now, rings_now = rc
        if op.swap:
            if rings_now and op.segments == rings_now:
                infos.append("rounded profile %d -> %d" % (rings_now, op.segments))
                return None, True, infos
            err = revolution_clip_rebuild(bm, sel, sides_now, profile_segments=op.segments)
            if err:
                return err, False, infos
            me[RECESS_KEY] = [sides_now, op.segments]
            infos.append("rounded profile %s -> %d" % (rings_now, op.segments))
        else:
            err = revolution_clip_rebuild(bm, sel, op.segments)
            if err:
                return err, False, infos
            me[RECESS_KEY] = [op.segments, rings_now or 0]
            infos.append("round part with recess -> %d" % op.segments)
        bm.normal_update()
        bmesh.update_edit_mesh(me)
        return None, True, infos
    return None


def _dome_change(op, bm, me, sel, infos):
    """Dome cut by unselected geometry (a bar across a dish): (err, did, infos), or None if it isn't one."""
    data, err = dome_cut_setup(bm, sel)
    if err:
        return None
    werr = _no_width(op, "a dome cut by other geometry")
    if werr:
        return werr, False, infos
    if op.swap:
        old = data["profile_count"]
        if old is None:
            return "This dome has no rounded profile to change.", False, infos
        infos.append("rounded profile %d -> %d" % (old, op.segments))
        if op.segments == old:
            return None, True, infos
        err = dome_cut_rebuild(bm, sel, data["sides"], profile_segments=op.segments, data=data)
    else:
        infos.append("around %d -> %d" % (data["sides"], op.segments))
        err = dome_cut_rebuild(bm, sel, op.segments, data=data)
    if err:
        return err, False, infos
    bm.normal_update()
    bmesh.update_edit_mesh(me)
    return None, True, infos


def _tube_ring_region(bm):
    """Faces along the selected closed edge loop(s) when they go around a tube (not a hole's
    rim): processed like an edge chain, so the tube's sides change. None otherwise."""
    if any(f.select for f in bm.faces):
        return None
    loops = selected_loops(bm)
    if not loops or not all(tube_loop(bm, loop) for loop in loops):
        return None
    # the whole tube: quads joined without a crease (a bent tube, a torus, a fillet at its end),
    # if that is one clean grid; else the faces along the loop
    start = {f for e in bm.edges if e.select for f in e.link_faces}
    region, stack = set(start), list(start)
    while stack and len(region) < 20000:
        f = stack.pop()
        for e in f.edges:
            if len(e.link_faces) != 2:
                continue
            g = e.link_faces[0] if e.link_faces[1] is f else e.link_faces[1]
            if g in region or len(g.verts) != 4 or f.normal.angle(g.normal, 0.0) > 0.87:
                continue
            region.add(g)
            stack.append(g)
    grids, errs = analyse(bm, False, ordered(region))
    if len(grids) == 1 and not errs:
        return ordered(region)
    return edge_chain_region(bm, closed_ok=True)


def _reselect_hole_loops(bm, centres):
    """After a hole change whose wall was rebuilt (its faces are selected): select the new rim
    loop nearest each old loop's centre instead, so Alt+B can go on with the loop."""
    sel = {f for f in bm.faces if f.select}
    if not sel:
        return
    border = [e for f in sel for e in f.edges if sum(1 for g in e.link_faces if g in sel) == 1]
    adj = {}
    for e in set(border):
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    loops, seen = [], set()
    for e0 in ordered(set(border)):
        if e0 in seen:
            continue
        comp, stack = [], [e0]
        seen.add(e0)
        while stack:
            e = stack.pop()
            comp.append(e)
            for v in e.verts:
                for e2 in adj[v]:
                    if e2 not in seen:
                        seen.add(e2)
                        stack.append(e2)
        loops.append(comp)
    if not loops:
        return
    keep = set()
    for c in centres:
        def dist(comp):
            vs = {v for e in comp for v in e.verts}
            return (sum((v.co for v in vs), Vector()) / len(vs) - c).length
        keep.update(min(loops, key=dist))
    for f in bm.faces:
        f.select = False
    for e in bm.edges:
        e.select = False
    for v in bm.verts:
        v.select = False
    for e in keep:
        e.select = True
        for v in e.verts:
            v.select = True
    bm.select_history.clear()


def _chain_dirs(grids, chain_verts, swap):
    """Started from an edge chain: the main direction is the one the chain runs along (a ring
    around a filleted tube gives its sides, not the fillet's segments)."""
    if not chain_verts:
        return grids
    out = []
    for g, d in grids:
        hits = [max((len(chain_verts & set(p)) for p in grid_profiles(g, dd)), default=0) for dd in (0, 1)]
        if hits[0] != hits[1] and max(hits) >= 2:
            main = 0 if hits[0] > hits[1] else 1
            d = 1 - main if swap else main
        out.append((g, d))
    return out


def _all_grid(bm, faces, swap):
    """The faces are quad grids, apart from caps of closed ones (the n-gon closing a hole wall,
    selected with it)."""
    rest = set(faces) - set(grid_faces_of(faces))
    if not rest:
        return True
    grids, errs = analyse(bm, swap)
    closed_verts = {v for g, _ in grids if any(g.cyclic) for v in g.V.values()}
    return bool(grids) and not errs and all(all(v in closed_verts for v in f.verts) for f in rest)


def _border_ngons(sel, grids, errs):
    """Selected faces besides one open quad grid that only join it along one side of its border
    (the last segment of a fillet notched by cut-outs: an n-gon). They are kept as they are and
    the grid changes alone. None if the selection isn't that."""
    if not sel or errs or len(grids) != 1 or any(grids[0][0].cyclic):
        return None
    g = grids[0][0]
    rest = [f for f in sel if f not in g.fset]
    if not rest:
        return None
    pos = {v: ij for ij, v in g.V.items()}
    edges = {e for f in g.fset for e in f.edges}
    for f in rest:
        shared = [pos.get(v) for e in f.edges if e in edges for v in e.verts]
        if not shared or None in shared:
            return None
        if not any(len({ij[dim] for ij in shared}) == 1 and shared[0][dim] in (0, g.n[dim] - 1)
                   for dim in (0, 1)):
            return None
    return rest


def _grid_ok(bm, grids, errors, segments, op):
    """True if the grid mode can change these grids to `segments` (None: their own count).
    Used to choose between the grid mode and the strip mode when n-gons are selected."""
    if errors or not grids:
        return False
    try:
        for g, dirn in grids:
            for band in resample_bands(g, dirn):
                gg = g if band is None else _subgrid(g, dirn, band[0], band[1])
                n = segments or (band[1] - band[0] if band else direction_stats(gg, dirn)[0])
                err, _ = plan_grid(bm, gg, dirn, n, op, None)
                if err:
                    return False
    except Exception:
        return False
    return True


class MESH_OT_facetor(bpy.types.Operator):
    """Change the number of segments of a cylinder, pipe, bevel, fillet or hole while keeping its silhouette.
Select its faces, the edge loop of a hole, or the edges of a rounded corner on a flat face. Use the mouse wheel to adjust"""
    bl_idname = "mesh.facetor"
    bl_label = "Facetor"
    bl_options = {"REGISTER", "UNDO"}

    segments: bpy.props.IntProperty(name="Segments", default=8, min=1, max=1024,
                                    description="New number of segments (sides of a cylinder, segments of a bevel)")
    swap: bpy.props.BoolProperty(name="Swap Direction", default=False, options={"SKIP_SAVE"},
                                 description="Resample the other direction of the face grid")
    shape: bpy.props.EnumProperty(name="Profile Type", default="KEEP", items=[
        ("KEEP", "Keep Silhouette", "Follow the existing shape (arc, circle or smooth curve)"),
        ("ROUND", "Round", "Rebuild a round bevel from its corner (also turns chamfers into round bevels)"),
        ("CUSTOM", "Custom Profile", "Use the bevel tool's custom profile"),
    ], options={"SKIP_SAVE"})
    profile_shape: bpy.props.FloatProperty(name="Shape", default=0.5, min=0.0, max=1.0, options={"SKIP_SAVE"},
                                           description="Round shape: 0.25 = straight, 0.5 = circle, 1 = square corner, below 0.25 = concave, 0 = square inward")
    width: bpy.props.FloatProperty(name="Width", default=1.0, min=0.0, soft_max=3.0, options={"SKIP_SAVE"},
                                   description="Bevel width, relative to the current one (1 = unchanged, "
                                               "0 = merged back into a sharp edge)")
    triangulate: bpy.props.BoolProperty(name="Triangulate N-gons", default=False,
                                        description="Triangulate the faces of more than 4 sides that the change "
                                                    "leaves (around a hole, on a cap, on the walls of a recess)")
    segments_other: bpy.props.IntProperty(name="Other Direction", default=0, min=0, max=1024, options={"SKIP_SAVE"},
                                          description="Number of segments in the other direction of the part "
                                                      "(the one S switches to), changed after the first one")
    info: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    # counts the part had before the change (-1: unknown): a direction whose value didn't change
    # is left alone when the panel runs the operator again
    main_was: bpy.props.IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})
    other_was: bpy.props.IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})
    # replayed like the modal: which direction came first, and the profile it was changed with
    # (first_shape "" = the same as the panel's)
    other_first: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})
    other_visited: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})
    first_shape: bpy.props.StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    first_width: bpy.props.FloatProperty(default=1.0, options={"HIDDEN", "SKIP_SAVE"})
    first_profile: bpy.props.FloatProperty(default=0.5, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def draw(self, context):
        lay = self.layout
        if self.info:
            for line in self.info.split("\n"):
                lay.label(text=line, icon="INFO")
        lay.prop(self, "segments")
        if self.swap:
            lay.prop(self, "swap")
        elif self.other_was > 0 or self.segments_other > 0:
            lay.prop(self, "segments_other")
        lay.prop(self, "shape")
        lay.prop(self, "width")
        if self.shape == "ROUND":
            lay.prop(self, "profile_shape")
        elif self.shape == "CUSTOM":
            lay.template_curveprofile(context.tool_settings, "custom_bevel_profile_preset")
        lay.prop(self, "triangulate")

    # ----- what is selected, and how many segments it has now
    @_with_clip_cache
    def _current_count(self, context):
        for obj in context.objects_in_mode_unique_data:
            if obj.type != "MESH":
                continue
            bm = bmesh.from_edit_mesh(obj.data)
            arcs = outline_arcs(bm)
            if arcs:
                act = bm.select_history.active
                pick = next((a for a in arcs if act in a["chain"] or
                             (isinstance(act, bmesh.types.BMEdge) and act.verts[0] in a["chain"]
                              and act.verts[1] in a["chain"])), arcs[0])
                # one direction only (S has nothing to switch to)
                return (None, None) if self.swap else (len(pick["chain"]) - 1, False)
            chain_verts = None
            from_chain = False
            if not any(f.select for f in bm.faces):
                region = edge_chain_region(bm)
                ring = False
                if not region:
                    region = _tube_ring_region(bm)
                    ring = bool(region)
                if region and any(len(f.verts) != 4 for f in region):
                    cinfo = chain_only_check(bm)
                    if cinfo:
                        # S: the columns between the chain's two neighbour lines (two now)
                        return (2, False) if self.swap else (len(cinfo["chain"]) - 1, False)
                from_chain = bool(region)
                if region:
                    # counted on a copy with the faces along the chain selected: the edit mesh
                    # keeps its edge selection, so process() still sees a chain (and selects the
                    # new one afterwards)
                    bm.faces.index_update()
                    bm.verts.index_update()
                    idx = [f.index for f in region]
                    vidx = {v.index for e in bm.edges if e.select for v in e.verts}
                    bm = bm.copy()
                    bm.faces.ensure_lookup_table()
                    bm.verts.ensure_lookup_table()
                    for i in idx:
                        bm.faces[i].select = True
                    if ring:
                        chain_verts = {bm.verts[i] for i in vidx}
            if not any(f.select for f in bm.faces) and not gather_faces(bm):
                loops = selected_loops(bm)
                if loops:
                    return (None, None) if self.swap else (len(loops[0]), True)
            old_verts = None
            sel0 = [f for f in bm.faces if f.select]
            if sel0 and (len(grid_faces_of(sel0)) != len(sel0) or tiny_edges(sel0)):
                # CAD tiny edges / T-junctions inside the selection: counted as process() will
                # fix them (on a copy)
                bmc = bm.copy()
                collapse_tiny_edges(bmc, [f for f in bmc.faces if f.select])
                selc = [f for f in bmc.faces if f.select]
                before = set(bmc.verts)
                newc = selc if _all_grid(bmc, selc, self.swap) else extend_t_columns(bmc, selc)
                if newc and _all_grid(bmc, newc, self.swap) and not analyse(bmc, self.swap)[1]:
                    bm = bmc
                    if newc is not selc:
                        old_verts = before
            grids, errs = analyse(bm, self.swap)
            sel = [f for f in bm.faces if f.select]
            closed_verts = {v for g, _ in grids if any(g.cyclic) for v in g.V.values()}
            clean = bool(grids) and not errs and all(
                all(v in closed_verts for v in f.verts) for f in set(sel) - set(grid_faces_of(sel)))
            if sel and not clean:
                dc = dome_cut_counts(bm, sel)
                if dc and dc[1 if self.swap else 0]:
                    return dc[1 if self.swap else 0], not self.swap
                if revolution_rings(bm, sel)[1] is not None:
                    rc = recess_counts(obj.data, bm, sel)
                    if rc and rc[1 if self.swap else 0]:
                        return rc[1 if self.swap else 0], not self.swap
                if self.swap:
                    n = revolution_profile_count(bm, sel) if revolution_rings(bm, sel)[1] is None \
                        else (revolution_clip_profile_count(bm, sel) or revolution_cut_profile_count(bm, sel))
                    if n:
                        return n, False
                n = revolution_count(bm, sel) or revolution_clip_count(bm, sel) or revolution_cut_count(bm, sel)
                if n:
                    return n, True
            if _border_ngons(sel, grids, errs):
                g, dirn = grids[0]
                bands = resample_bands(g, dirn)
                if bands[0] is not None:
                    return max(hi - lo for lo, hi in bands), False
                return direction_stats(g, dirn)[0], g.cyclic[dirn]
            if sel and (errs or not grids):
                net, nerr = bevel_network(sel)
                if net:
                    # fillets joined by corners: one direction only (the fillets' segments)
                    return (None, None) if self.swap else (net["count"], False)
            ref = None if from_chain else find_round_part(bm, grids)
            if ref is not None:
                return (None, None) if self.swap else (len(ref.verts), True)
            if sel and (len(grid_faces_of(sel)) != len(sel) or
                        (any(len(f.verts) != 4 for f in sel) and
                         not _grid_ok(bm, grids, errs, None, self))):
                info, err = strip_check(sel)
                if self.swap and (not err or stacked_strips(sel)):
                    return None, None          # strips: one direction only
                if not err:
                    if "count" in info:
                        return info["count"], False
                    return (len(info["A"]) if info["closed"] else len(info["A"]) - 1), info["closed"]
                parts = stacked_strips(sel)
                if parts:
                    return min(info.get("count", len(info["A"]) - 1) for _, info in parts), False
            if sel and not clean:
                core = band_core(sel)
                if core is not None:
                    g, dirn = analyse(bm, self.swap, core)[0][0]
                    bands = resample_bands(g, dirn)
                    if bands[0] is not None:
                        return max(hi - lo for lo, hi in bands), False
                    return direction_stats(g, dirn)[0], g.cyclic[dirn]
            if grids:
                grids = _chain_dirs(grids, chain_verts, self.swap)
                act = bm.faces.active
                pick = next(((g, d) for g, d in grids if act in g.fset), grids[0])
                g, dirn = pick
                bands = resample_bands(g, dirn)
                if old_verts is not None and not g.cyclic[dirn] and bands[0] is None:
                    # lines were carried across (e.g. a box corner whose fillet and wall have
                    # different column counts): the count the part had, along its finest border
                    profs = grid_profiles(g, dirn)
                    return max(sum(1 for v in p if v in old_verts) for p in (profs[0], profs[-1])) - 1, False
                if bands[0] is not None:
                    return max(hi - lo for lo, hi in bands), False
                return direction_stats(g, dirn)[0], g.cyclic[dirn]
            if sel and not self.swap:
                return len(max(sel, key=lambda f: len(f.verts)).verts), True
        return None, None

    # ----- one run on every object; returns (error, did_anything)
    def _run(self, context):
        did_any = False
        infos = []
        tris = [0, 0]
        for obj in context.objects_in_mode_unique_data:
            if obj.type != "MESH":
                continue
            bm = bmesh.from_edit_mesh(obj.data)
            tris[0] += tri_count(bm)
            err, did, info = self.process(context, obj)
            if err:
                return err, False
            did_any |= did
            infos += info
            tris[1] += tri_count(bmesh.from_edit_mesh(obj.data))
        if not did_any:
            return "Nothing to do: select the faces of a cylinder, pipe, bevel or fillet, or the edge loop of a hole.", False
        self.info = ", ".join(infos)[:110] + "\nTris: %d -> %d" % (tris[0], tris[1])
        return None, True

    def execute(self, context):
        if self.swap:
            # only the other direction (scripts)
            err, ok = self._run(context)
            if err:
                self.report({"ERROR"}, err)
                return {"CANCELLED"}
            return {"FINISHED"}
        # both directions, as set in the panel, in the order they were changed in the modal (the
        # second on top of the first). A direction whose count didn't change and that has no
        # width / profile to apply is left alone (S without change keeps nothing)
        main_was = getattr(self, "main_was", -1)
        other, other_was = getattr(self, "segments_other", 0), getattr(self, "other_was", -1)
        cur = (self.shape, self.width, self.profile_shape)
        first_shape = getattr(self, "first_shape", "")
        first = (first_shape, self.first_width, self.first_profile) if first_shape else cur
        order = [True, False] if getattr(self, "other_first", False) else [False, True]
        seg = self.segments
        infos = []
        ran = False
        for i, swap in enumerate(order):
            shape = first if i == 0 else cur
            # a width / profile only redoes a direction the modal went to
            shaped = (shape[0] != "KEEP" or abs(shape[1] - 1.0) > 1e-6) and                 (not swap or getattr(self, "other_visited", False))
            if swap:
                count, was = other, other_was
                if count <= 0 or (count == was and not shaped):
                    continue
            else:
                count, was = seg, main_was
                if count == was and not shaped:
                    continue
            self.swap, self.segments = swap, count
            self.shape, self.width, self.profile_shape = shape
            try:
                err, ok = self._run(context)
            finally:
                self.swap, self.segments = False, seg
                self.shape, self.width, self.profile_shape = cur
            if err:
                self.report({"ERROR"}, err)
                return {"CANCELLED"}
            infos.append(self.info)
            ran = True
        if not ran:
            err, ok = self._run(context)
            if err:
                self.report({"ERROR"}, err)
                return {"CANCELLED"}
            infos.append(self.info)
        if len(infos) > 1:
            heads = [i.split("\n")[0] for i in infos]
            self.info = ", ".join(heads)[:110] + "\n" + infos[-1].split("\n")[-1]
        return {"FINISHED"}

    def _store_counts(self):
        """Values for the redo panel: each direction's count (what was set, else what it had)."""
        vals, was = getattr(self, "_vals", {}), getattr(self, "_was", {})
        self.main_was = was.get(False) or -1
        self.other_was = was.get(True) or -1
        self.segments = vals.get(False, was.get(False) or self.segments)
        self.segments_other = vals.get(True, was.get(True) or 0)
        self.other_visited = getattr(self, "_other_visited", False)
        first = getattr(self, "_first", None) if getattr(self, "_kept", False) else None
        if first:
            self.other_first = first[0]
            self.first_shape, self.first_width, self.first_profile = first[1:]
        else:
            self.other_first, self.first_shape = False, ""
        self.swap = False

    # ----- interactive: mouse wheel
    def invoke(self, context, event):
        # every Alt+B starts fresh: main direction, Keep Silhouette, default width; the shape is
        # the part's own (a fillet already changed to 0.85 starts P / Round at 0.85, not 0.5)
        self.swap = False
        self.shape = "KEEP"
        self.width = 1.0
        self._shape_found = 0.5
        try:
            obj = context.edit_object
            if obj is not None and obj.type == "MESH":
                sel = [f for f in bmesh.from_edit_mesh(obj.data).faces if f.select]
                found = measure_shape(sel) if sel else None
                if found is not None:
                    self._shape_found = found
        except Exception:
            pass
        self.profile_shape = self._shape_found
        self._touched = False
        # modes already found not to fit this selection: skipped on the next mouse moves
        self._not_modes = {}
        n, _ = self._current_count(context)
        if n:
            self.segments = n
        self.segments_other, self.main_was, self.other_was = 0, -1, -1
        if context.area is None or context.area.type != "VIEW_3D":
            return self.execute(context)
        # counts of both directions before the change, and the values set in each (redo panel)
        self.swap = True
        try:
            n_other, _ = self._current_count(context)
        except Exception:
            n_other = None
        self.swap = False
        self._was = {False: n, True: n_other}
        self._vals, self._vals_visit = {}, {}
        self._snaps, self._orig, self._active = {}, {}, {}
        self._props, self._props_orig = {}, {}
        for obj in context.objects_in_mode_unique_data:
            if obj.type == "MESH":
                bm = bmesh.from_edit_mesh(obj.data)
                self._snaps[obj.name] = bm.copy()
                self._orig[obj.name] = bm.copy()
                self._props[obj.name] = self._props_orig[obj.name] = list(obj.data.get(RECESS_KEY, []))
                self._active[obj.name] = bm.faces.active.index if bm.faces.active else None
        self._start_x = event.mouse_x
        self._start_width = self.width
        self._start_profile = self.profile_shape
        self._mouse_profile = False
        self._mouse_width = False
        self._fine = False
        self._typed = ""
        self._err = ""
        self._kept = False
        self._last_ok = {}
        self._other_visited = False
        self._first = None
        self._apply(context)
        if self._err and not self._applied_once:
            self._cleanup(context)
            self.report({"ERROR"}, self._err)
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _restore_all(self, context, original=False):
        for obj in context.objects_in_mode_unique_data:
            snap = (self._orig if original else self._snaps).get(obj.name)
            if snap is None:
                continue
            prop = (self._props_orig if original else self._props).get(obj.name)
            if prop:
                obj.data[RECESS_KEY] = prop
            elif RECESS_KEY in obj.data:
                del obj.data[RECESS_KEY]
            bm = bmesh.from_edit_mesh(obj.data)
            _restore(bm, snap)
            idx = self._active.get(obj.name)
            if idx is not None:
                bm.faces.ensure_lookup_table()
                if idx < len(bm.faces):
                    bm.faces.active = bm.faces[idx]
            bmesh.update_edit_mesh(obj.data)

    def _apply(self, context):
        self._restore_all(context)
        err, ok = self._run(context)
        if err:
            self._restore_all(context)
        self._err = err or ""
        self._applied_once = getattr(self, "_applied_once", False) or bool(ok)
        if ok and self._touched and hasattr(self, "_vals"):
            self._vals[self.swap] = self.segments
        if ok and not err:
            # last values that worked in this direction (confirming on an error goes back to them)
            if not hasattr(self, "_last_ok"):
                self._last_ok = {}
            self._last_ok[self.swap] = (self.segments, self.shape, self.width, self.profile_shape,
                                        self._touched)
        # header: only the values (like Blender's Bevel)
        shape_txt = {"KEEP": "Keep Silhouette", "ROUND": "Round", "CUSTOM": "Custom Profile (Bevel tool's profile, straight by default)"}[self.shape]
        what = self.info.split("\n")[0] if (self.info and not self._err) else ""
        typed = getattr(self, "_typed", "")
        target = _typing_target(self)
        parts = ["Segments: %s|" % typed if typed and target == "segments" else "Segments: %d" % self.segments]
        if what:
            head = next((p for p in what.split(", ") if " -> " in p), what.split(",")[0])
            first = head.split(" -> ")[0]          # e.g. "around 48", "fillet 12", "round part"
            kind, _, old = first.rpartition(" ")
            if old.isdigit() and kind:
                parts.append("%s (was %s)" % (kind.capitalize(), old))
            else:
                parts.append(first.capitalize())
        parts.append("Profile Type: %s" % shape_txt)
        if self.shape != "KEEP" or getattr(self, "_mouse_width", False) or abs(self.width - 1.0) > 1e-6:
            parts.append("Width: %s|" % typed if typed and target == "width" else "Width: %.2f" % self.width)
        if self.shape == "ROUND":
            parts.append("Shape: %s|" % typed if typed and target == "shape" else "Shape: %.2f" % self.profile_shape)
        parts.append("Triangulate: %s" % ("On" if self.triangulate else "Off"))
        txt = ", ".join(parts)
        if self._err:
            txt += "   —   " + self._err
        self._header = txt
        context.area.header_text_set(txt)
        # status bar: the shortcuts (like Blender's Bevel)
        ws = getattr(context, "workspace", None)
        if ws is not None and hasattr(ws, "status_text_set"):
            ws.status_text_set(_status_draw(self.shape, getattr(self, "_mouse_width", False)))
    def _cleanup(self, context):
        for snap in list(self._snaps.values()) + list(getattr(self, "_orig", {}).values()):
            snap.free()
        self._snaps, self._orig = {}, {}
        if context.area:
            context.area.header_text_set(None)
        ws = getattr(context, "workspace", None)
        if ws is not None and hasattr(ws, "status_text_set"):
            ws.status_text_set(None)

    def modal(self, context, event):
        t, v = event.type, event.value
        if t in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "NDOF_MOTION"} or t.startswith("NUMPAD_") and t not in {
                "NUMPAD_PLUS", "NUMPAD_MINUS", "NUMPAD_ENTER", "NUMPAD_PERIOD"} and t not in _DIGITS:
            return {"PASS_THROUGH"}
        up = t in {"WHEELUPMOUSE", "NUMPAD_PLUS", "EQUAL", "UP_ARROW"} and v in {"PRESS", "NOTHING"}
        down = t in {"WHEELDOWNMOUSE", "NUMPAD_MINUS", "MINUS", "DOWN_ARROW"} and v in {"PRESS", "NOTHING"}
        digit = _DIGITS.get(t)
        target = _typing_target(self)
        if t in {"PERIOD", "NUMPAD_PERIOD"} and target != "segments":
            digit = "."
        typed = getattr(self, "_typed", "")
        if v == "PRESS" and (digit is not None or (t == "BACK_SPACE" and typed)):
            # a value typed on the keyboard (top row or numpad, like Blender's tools): the
            # number of segments, or the width / shape when W / P makes the mouse set them.
            # Applied as it is typed, Enter / click confirms
            if t == "BACK_SPACE":
                typed = typed[:-1]
            elif not (digit == "." and "." in typed):
                typed = (typed + digit)[:6]
            self._typed = typed
            try:
                val = float(typed) if typed and typed != "." else None
            except ValueError:
                val = None
            if val is not None:
                if target == "segments" and val >= 1:
                    self.segments = min(1024, int(val))
                elif target == "width":
                    self.width = val
                elif target == "shape":
                    self.profile_shape = min(1.0, val)
                self._touched = True
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t not in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE", "TIMER", "LEFT_SHIFT", "RIGHT_SHIFT"} and v == "PRESS":
            self._typed = ""
        if up or down:
            n = self.segments
            if event.ctrl:
                n = (n // 4 + 1) * 4 if up else max(4, ((n - 1) // 4) * 4)
            else:
                n = n + 1 if up else n - 1
            self.segments = max(1, min(1024, n))
            self._touched = True
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t == "S" and v == "PRESS":
            if not self.swap:
                # a part with one direction only (hole, flat rounded corner, strip, round part
                # with details): nothing to switch to
                self.swap = True
                try:
                    n_other, _ = self._current_count(context)
                except Exception:
                    n_other = None
                self.swap = False
                if not n_other:
                    context.area.header_text_set(getattr(self, "_header", "") +
                                                 "   —   This part has only one direction")
                    return {"RUNNING_MODAL"}
            if self._err or not self._touched:
                # nothing was set in this direction (or it failed): don't keep anything from it
                self._restore_all(context)
                self._vals = dict(self._vals_visit)
            else:
                # keep what was just set (e.g. the number of sides) and change the other direction on top
                self._vals_visit = dict(self._vals)
                if not getattr(self, "_kept", False):
                    # the first direction kept, and the profile it was changed with
                    self._first = (self.swap, self.shape, self.width, self.profile_shape)
                self._kept = True
                for obj in context.objects_in_mode_unique_data:
                    if obj.name in self._snaps:
                        self._snaps[obj.name].free()
                        bm = bmesh.from_edit_mesh(obj.data)
                        self._snaps[obj.name] = bm.copy()
                        self._props[obj.name] = list(obj.data.get(RECESS_KEY, []))
                        self._active[obj.name] = bm.faces.active.index if bm.faces.active else None
            self.swap = not self.swap
            # profile type, width and shape belong to the direction they were set in (the first
            # one is kept in the snapshot): the other direction starts in Keep
            self.shape, self.width = "KEEP", 1.0
            self.profile_shape = 0.5 if self.swap else getattr(self, "_shape_found", 0.5)
            self._mouse_width = self._mouse_profile = False
            if self.swap:
                self._other_visited = True
            self._touched = False
            n, _ = self._current_count(context)
            if n:
                self.segments = n
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t == "P" and v == "PRESS":
            # the mouse sets the profile instead of the width (like Blender's Bevel)
            if self.shape == "KEEP":
                self.shape = "ROUND"
            self._mouse_profile = not getattr(self, "_mouse_profile", False)
            self._mouse_width = False
            self._start_x = event.mouse_x
            self._start_width = self.width
            self._start_profile = self.profile_shape
            self._touched = True
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t == "W" and v == "PRESS":
            # the mouse sets the width (in Keep too: the shape is scaled around its corner)
            self._mouse_width = not getattr(self, "_mouse_width", False)
            self._mouse_profile = False
            self._start_x = event.mouse_x
            self._start_width = self.width
            self._start_profile = self.profile_shape
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t == "T" and v == "PRESS":
            self.triangulate = not self.triangulate
            self._touched = True
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t == "R" and v == "PRESS":
            self._touched = True
            order = ["KEEP", "ROUND", "CUSTOM"]
            self.shape = order[(order.index(self.shape) + 1) % 3]
            if self.shape == "KEEP" and not getattr(self, "_mouse_width", False):
                self.width = 1.0          # back to Keep: the shape as it was (W sets a width in Keep)
            self._start_x = event.mouse_x
            self._start_width = self.width
            self._apply(context)
            return {"RUNNING_MODAL"}
        if t == "MOUSEMOVE":
            if getattr(self, "_typed", ""):
                return {"RUNNING_MODAL"}           # a typed value wins over the mouse
            if self.shape != "KEEP" or getattr(self, "_mouse_width", False):
                # Shift held: ten times finer (like Blender's tools), from where the value is
                # when it's pressed or released (no jump)
                fine = bool(getattr(event, "shift", False))
                if fine != getattr(self, "_fine", False):
                    self._fine = fine
                    self._start_x = event.mouse_x
                    self._start_width = self.width
                    self._start_profile = self.profile_shape
                dx = (event.mouse_x - self._start_x) * (0.1 if fine else 1.0)
                if getattr(self, "_mouse_profile", False) and self.shape == "ROUND":
                    self.profile_shape = max(0.0, min(1.0, self._start_profile + dx / 400.0))
                else:
                    self.width = max(0.0, self._start_width + dx / 200.0)
                self._touched = True
                self._apply(context)
            return {"RUNNING_MODAL"}
        if t in {"LEFTMOUSE", "RET", "NUMPAD_ENTER", "SPACE"} and v == "PRESS":
            last = getattr(self, "_last_ok", {}).get(self.swap)
            if self._err and last:
                # the current value can't be done: confirm the last one that worked instead
                bad = self.segments
                self.segments, self.shape, self.width, self.profile_shape, self._touched = last
                self._apply(context)
                if not self._err:
                    self.report({"WARNING"}, "%d segments isn't possible here: %d kept" % (bad, self.segments)
                                if bad != self.segments else "The last value that worked was kept")
                    self._store_counts()
                    self._cleanup(context)
                    return {"FINISHED"}
            if self._err:
                if getattr(self, "_kept", False):
                    # this direction failed, but what was set before S is kept
                    self._restore_all(context)
                    self._vals = dict(self._vals_visit)
                    self.report({"WARNING"}, self._err + " (only the first direction was changed)")
                    self._store_counts()
                    self._cleanup(context)
                    return {"FINISHED"}
                self._restore_all(context, original=True)
                self._cleanup(context)
                self.report({"ERROR"}, self._err)
                return {"CANCELLED"}
            self._store_counts()
            self._cleanup(context)
            return {"FINISHED"}
        if t in {"RIGHTMOUSE", "ESC"} and v == "PRESS":
            self._restore_all(context, original=True)
            self._cleanup(context)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def process(self, context, obj):
        """Run the change, then carry UVs / seams / sharp edges over and triangulate if asked."""
        bm = bmesh.from_edit_mesh(obj.data)
        snap = bm.copy()            # a failed change leaves nothing behind (caps joined first...)
        keeper = Keeper(bm)
        self._own_uvs = False
        try:
            err, did, infos = self._process(context, obj)
            bm = bmesh.from_edit_mesh(obj.data)
            if err:
                _restore(bm, snap)
            if did and not err:
                keeper.bm = bm
                keeper.apply(new_faces=not self._own_uvs)
                if self.triangulate:
                    n = triangulate_ngons(bm, keeper)
                    if n:
                        infos.append("%d face%s triangulated" % (n, "s" if n > 1 else ""))
                bm.normal_update()
            return err, did, infos
        finally:
            snap.free()
            keeper.bm = bmesh.from_edit_mesh(obj.data)
            keeper.free()
            bmesh.update_edit_mesh(obj.data)

    def _process(self, context, obj):
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        infos = []
        sel = [f for f in bm.faces if f.select]
        chain_verts = None
        chain_closed = False
        chain_pairs = None

        # 0a) rounded corner of a flat outline (edges selected on an n-gon's border)
        if not sel:
            arcs = outline_arcs(bm)
            if arcs:
                self._own_uvs = True       # outline_arc_rebuild fits the UVs itself
                profile = None
                if self.shape == "CUSTOM":
                    profile = context.tool_settings.custom_bevel_profile_preset
                    try:
                        profile.update()
                    except Exception:
                        pass
                for arc in arcs:
                    m = len(arc["chain"]) - 1
                    err = outline_arc_rebuild(bm, arc, self.segments, self, profile)
                    if err:
                        return err, False, infos
                    infos.append("arc merged (width 0)" if getattr(self, "width", 1.0) < 1e-4 else "arc %d -> %d" % (m, self.segments))
                bm.normal_update()
                bmesh.update_edit_mesh(me)
                return None, True, infos

        # 0) an open chain of edges along a curve: work on the strips that run along it
        if not sel:
            region = edge_chain_region(bm) or _tube_ring_region(bm)
            cinfo = chain_only_check(bm) if region and any(len(f.verts) != 4 for f in region) else None
            if cinfo:
                # the faces along the chain are not a grid (n-gons from an earlier change): only
                # the chain changes, the faces on each side become n-gons up to the next line
                werr = _no_width(self, "a single line of edges between n-gons")
                if werr:
                    return werr, False, infos
                m = len(cinfo["chain"]) - 1
                if self.swap:
                    err, N = chain_only_columns(bm, cinfo, self.segments)
                else:
                    err, N = chain_only_rebuild(bm, cinfo, self.segments)
                if err:
                    return err, False, infos
                for f in bm.faces:
                    f.select = False
                for e in bm.edges:
                    e.select = False
                for v in bm.verts:
                    v.select = False
                for a, b in zip(N, N[1:]):
                    e = edge_between(a, b)
                    if e is not None:
                        e.select = True
                        a.select = b.select = True
                bm.select_history.clear()
                bmesh.update_edit_mesh(me)
                infos.append("columns 2 -> %d" % self.segments if self.swap else "line %d -> %d" % (m, self.segments))
                return None, True, infos
            if region:
                chain_verts = {v for e in bm.edges if e.select for v in e.verts}
                chain_closed = all(sum(1 for e in v.link_edges if e.select) == 2 for v in chain_verts)
                chain_pairs = [tuple(e.verts) for e in bm.edges if e.select]
                for f in region:
                    f.select = True
                sel = region

        # open edge chains whose vertices close whole faces (the two end edges of a chamfer):
        # processed as those faces, the new chains are selected afterwards
        if not sel and chain_verts is None:
            sel_e = [e for e in bm.edges if e.select]
            deg = {}
            for e in sel_e:
                for v in e.verts:
                    deg[v] = deg.get(v, 0) + 1
            if sel_e and max(deg.values()) <= 2 and 1 in deg.values() and gather_faces(bm):
                chain_verts = set(deg)
                chain_pairs = [tuple(e.verts) for e in sel_e]

        # 1) holes: only an edge loop is selected
        if not sel and not gather_faces(bm):
            loops = selected_loops(bm)
            if loops:
                centres = [sum((v.co for v in loop), Vector()) / len(loop) for loop in loops]
                for loop in loops:
                    m = len(loop)
                    err = hole_rebuild(bm, loop, self.segments, getattr(self, "width", 1.0))
                    if err:
                        return err, False, infos
                    infos.append("hole %d -> %d" % (m, self.segments))
                _reselect_hole_loops(bm, centres)
                bm.normal_update()
                bmesh.update_edit_mesh(me)
                return None, True, infos

        # CAD vertices a hair's breadth apart (tiny edges) merged, then T-junctions inside the
        # selection (lines of faces ending in the middle of a face) carried on to the border or
        # extra rows taken out. Kept only if it gives a quad grid (not on recesses, revolutions...)
        if sel and (len(grid_faces_of(sel)) != len(sel) or tiny_edges(sel)):
            snap = bm.copy()
            if chain_verts:
                bm.verts.index_update()
                chain_idx = ({v.index for v in chain_verts},
                             [(a.index, b.index) for a, b in chain_pairs])
            n_tiny = collapse_tiny_edges(bm, sel)
            sel2 = [f for f in bm.faces if f.select]
            new = sel2 if _all_grid(bm, sel2, self.swap) else extend_t_columns(bm, sel2)
            if new and _all_grid(bm, new, self.swap) and not analyse(bm, self.swap)[1]:
                if n_tiny:
                    infos.append("%d tiny edge%s merged" % (n_tiny, "s" if n_tiny > 1 else ""))
                if new is not sel2:
                    infos.append("T-junctions joined")
                sel = new
            else:
                _restore(bm, snap)
                sel = [f for f in bm.faces if f.select]
                if chain_verts:
                    # the mesh was rebuilt from the copy: the chain's vertices are new objects
                    bm.verts.ensure_lookup_table()
                    chain_verts = {bm.verts[i] for i in chain_idx[0]}
                    chain_pairs = [(bm.verts[a], bm.verts[b]) for a, b in chain_idx[1]]
            snap.free()

        grids, errors = analyse(bm, self.swap)
        closed_verts = {v for g, _ in grids if any(g.cyclic) for v in g.V.values()}
        clean = bool(grids) and not errors and all(
            all(v in closed_verts for v in f.verts) for f in set(sel) - set(grid_faces_of(sel)))

        # 2) part of revolution made of pieces with different side counts (MoI): one count for all
        # (in the modal, a detection that failed once for this selection isn't run again: the
        # mesh is back to its original state before every change)
        not_modes = getattr(self, "_not_modes", None)
        if sel and not clean:
            bm.faces.index_update()
            skip = set()
            if not_modes is not None:
                key = (self.swap, len(bm.verts), len(bm.faces), tuple(sorted(f.index for f in sel)))
                skip = not_modes.setdefault(key, set())
            rerr = True
            if "revolution" not in skip:
                res, rerr = revolution_rings(bm, sel)
                if rerr:
                    skip.add("revolution")
            if not rerr:
                if self.swap:
                    sides = revolution_count(bm, sel)
                    old_p = revolution_profile_count(bm, sel)
                    err = revolution_rebuild(bm, sel, sides, profile_segments=self.segments,
                                             width=getattr(self, "width", 1.0))
                    if err:
                        return err, False, infos
                    infos.append("rounded profile %s -> %d" % (old_p, self.segments))
                    bm.normal_update()
                    bmesh.update_edit_mesh(me)
                    return None, True, infos
                err = revolution_rebuild(bm, sel, self.segments, width=getattr(self, "width", 1.0))
                if err:
                    return err, False, infos
                infos.append("part of revolution -> %d" % self.segments)
                bm.normal_update()
                bmesh.update_edit_mesh(me)
                return None, True, infos
            # same, but cut by a detail (slot, recess): the detail is kept as it is
            if "recess" not in skip:
                with _clip_cache():
                    res = _recess_change(self, bm, me, sel, infos)
                if res is not None:
                    return res
                skip.add("recess")
            # a dome cut by unselected geometry (bar across a dish): the cut edge is kept
            if "dome" not in skip:
                res = _dome_change(self, bm, me, sel, infos)
                if res is not None:
                    return res
                skip.add("dome")
            cdata, cerr = (None, True) if "cut" in skip else revolution_cut_rings(bm, sel)
            if cerr or len(cdata["order"]) < 2:
                skip.add("cut")
            if not cerr and len(cdata["order"]) >= 2 and _no_width(self, "a part of revolution cut by a detail"):
                return _no_width(self, "a part of revolution cut by a detail"), False, infos
            if not cerr and len(cdata["order"]) >= 2 and self.swap:
                sides = revolution_clip_count(bm, sel) or revolution_cut_count(bm, sel)
                old_p = revolution_cut_profile_count(bm, sel)
                err = revolution_cut_rebuild(bm, sel, sides, profile_segments=self.segments)
                if err:
                    return err, False, infos
                infos.append("rounded profile %s -> %d" % (old_p, self.segments))
                bm.normal_update()
                bmesh.update_edit_mesh(me)
                return None, True, infos
            if not cerr and len(cdata["order"]) >= 2:
                err = revolution_cut_rebuild(bm, sel, self.segments)
                if not err:
                    infos.append("round part with recess -> %d" % self.segments)
                    bm.normal_update()
                    bmesh.update_edit_mesh(me)
                    return None, True, infos

        if sel:
            sel = [f for f in bm.faces if f.select]
            grids, errs = analyse(bm, self.swap)
        # last segment of a fillet notched by cut-outs (an n-gon) selected with it: kept as it is
        kept = _border_ngons(sel, grids, errs) if sel else None
        if kept:
            for f in kept:
                f.select = False
            sel = [f for f in bm.faces if f.select]
            grids, errors = analyse(bm, self.swap)
            infos.append("%d n-gon%s kept" % (len(kept), "s" if len(kept) > 1 else ""))
        # started from an edge chain: the faces along it are the part (a cap they touch isn't a
        # "round part" reference)
        ref = None if (kept or chain_verts) else find_round_part(bm, grids)

        # 2b) round part with details (slotted screw head...)
        if ref is not None:
            werr = _no_width(self, "a round part with details")
            if werr:
                return werr, False, infos
            faces = (sel or gather_faces(bm)) + ([ref] if ref not in sel else [])
            perr, data = profile_reduce(bm, faces, self.segments, AXES)
            if perr:
                return perr, False, infos
            apply_profile(bm, data)
            bmesh.update_edit_mesh(me)
            infos.append("round part -> %d" % self.segments)
            return None, True, infos

        # 2c) fillets joined by corners where three of them meet (a box bevelled on all its edges)
        if sel and (errors or not grids):
            net, nerr = bevel_network(sel)
            if nerr:
                return nerr, False, infos
            if net:
                if self.swap:
                    return "This part has only one direction.", False, infos
                profile = None
                if self.shape == "CUSTOM":
                    profile = context.tool_settings.custom_bevel_profile_preset
                res = {}
                merge = getattr(self, "width", 1.0) < 1e-4

                def rebuild():
                    if merge:
                        res["err"], res["points"] = network_collapse(bm, net)
                    else:
                        # the width first, the same for the whole network (the corners join
                        # several fillets), then the segments
                        w = getattr(self, "width", 1.0)
                        res["err"] = network_scale(net, w) if abs(w - 1.0) > 1e-6 else None
                        if res["err"] is None:
                            self.width = 1.0
                            try:
                                res["err"] = network_rebuild(bm, net, self.segments, self, profile)
                            finally:
                                self.width = w
                    return res["err"] is None
                ok, _ = guarded(bm, rebuild)
                ps = self.profile_shape
                if not ok and not merge and self.shape == "ROUND" and 0.0 < ps < 0.01:
                    # a hair above 0 the corners are too small to be built (their faces have no
                    # area left): the square profile, which looks the same
                    sel = [f for f in bm.faces if f.select]
                    net, nerr = bevel_network(sel)
                    if net and not nerr:
                        self.profile_shape = 0.0
                        try:
                            ok, _ = guarded(bm, rebuild)
                        finally:
                            self.profile_shape = ps
                if not ok:
                    return (res.get("err") or "These fillets are joined to the rest of the mesh in a way "
                            "that can't be changed safely."), False, infos
                if merge:
                    # the sharp edges left, selected
                    pts = res["points"] or []
                    for f in bm.faces:
                        f.select = False
                    for e in bm.edges:
                        e.select = False
                    kd = mathutils.kdtree.KDTree(len(pts))
                    for i, p in enumerate(pts):
                        kd.insert(p, i)
                    kd.balance()
                    span = max((p - pts[0]).length for p in pts) if pts else 1.0
                    for v in bm.verts:
                        v.select = bool(pts) and kd.find(v.co)[2] < 1e-6 * (span or 1.0)
                    for e in bm.edges:
                        e.select = e.verts[0].select and e.verts[1].select
                    bm.select_history.clear()
                    infos.append("bevel merged (width 0)")
                    bm.normal_update()
                    bmesh.update_edit_mesh(me)
                    return None, True, infos
                infos.append("fillets %s -> %d" % ("/".join(map(str, net.get("counts", [net["count"]]))),
                                                   self.segments))
                infos.append("%d corner%s" % (len(net["corners"]), "s" if len(net["corners"]) > 1 else ""))
                bm.normal_update()
                bmesh.update_edit_mesh(me)
                return None, True, infos

        # fillets already squared inward (shape 0) on a network: steps welded at the corners, no
        # corner faces; the strip modes would fold them
        if sel and square_in_corners(sel):
            return ("These fillets have a square inward profile (Shape 0) and no corners left: "
                    "they can't be changed again (undo to go back)."), False, infos

        # 3) messy strip (n-gons / T-junctions from CAD): rebuilt from its two ends (n-gons that
        # are quads with T-junctions on the border go to the grid mode when it can change them)
        strip_err = None
        if sel and (errors or len(grid_faces_of(sel)) != len(sel) or
                    (any(len(f.verts) != 4 for f in sel) and
                     not _grid_ok(bm, grids, errors, self.segments, self))):
            stacked = None
            for comp in _face_components(sel):
                info, err = strip_check(comp)
                if err:
                    stacked = stacked_strips(comp) if len(_face_components(sel)) == 1 else None
                    strip_err = err
                    break
            else:
                for comp in _face_components(sel):
                    if not all(f.is_valid for f in comp):
                        # rebuilding a previous piece changed this one (pieces touching at a corner)
                        return STRIPS_TOUCH, False, infos
                    err = strip_rebuild(bm, comp, self.segments, width=getattr(self, "width", 1.0))
                    if err:
                        return err, False, infos
                    infos.append("strip -> %d" % self.segments)
                bmesh.update_edit_mesh(me)
                return None, True, infos

            if stacked:
                werr = _no_width(self, "strips stacked along one curve")
                if werr:
                    return werr, False, infos
                old = sorted({info.get("count", len(info["A"]) - 1) for _, info in stacked})
                err = stacked_rebuild(bm, stacked, self.segments)
                if err:
                    return err, False, infos
                infos.append("strips %s -> %d" % ("/".join(map(str, old)), self.segments))
                bmesh.update_edit_mesh(me)
                return None, True, infos

        # 3b) band whose ends are messy CAD corners: rebuild the clean part, keep the corners
        core = band_core(sel) if sel and (errors or not grids) else None
        if core is not None:
            grids, errors = analyse(bm, self.swap, core)
            core_set = set(core)
            for f in sel:
                if f not in core_set:
                    f.select = False
        if errors or not grids:
            if sel:
                # last resort: round part whose largest face is the reference
                perr, data = profile_reduce(bm, sel, self.segments, AXES)
                if not perr and _no_width(self, "a round part with details"):
                    return _no_width(self, "a round part with details"), False, infos
                if not perr:
                    apply_profile(bm, data)
                    bmesh.update_edit_mesh(me)
                    infos.append("round part -> %d" % self.segments)
                    return None, True, infos
                if has_bevel_poles(sel):
                    return ("Corners where three fillets meet: select every fillet around them (the "
                            "whole bevel), they change together."), False, infos
                return (errors[0] if errors else (strip_err or perr)), False, infos
            return None, False, infos

        # 4) clean quad grids (cylinders, tubes, bevels, fillets)
        self._own_uvs = True               # execute_plan transfers the UVs itself
        if getattr(self, "width", 1.0) < 1e-4:
            # width 0: the bevel is merged back into a sharp edge
            grids = _chain_dirs(grids, chain_verts if chain_closed else None, self.swap)
            for f in bm.faces:
                f.select = False
            corner_verts = []

            def collapse():
                for g, dirn in grids:
                    for band in resample_bands(g, dirn):
                        gg = g if band is None else _subgrid(g, dirn, band[0], band[1])
                        err, pts = collapse_to_corners(bm, gg, dirn)
                        if err:
                            infos.append(err)
                            return False
                        corner_verts.extend(pts)
                return True
            ok, _ = guarded(bm, collapse)
            if not ok:
                return (infos[-1] if infos and infos[-1].endswith(".") else
                        "This bevel is joined to the rest of the mesh in a way that can't be merged safely."), False, infos
            for e in bm.edges:
                e.select = False
            for v in bm.verts:
                v.select = False
            # the merged vertices (one per corner point) and the sharp edge between them
            near = {v for v in bm.verts for c in corner_verts if (v.co - c).length < 1e-6 * (1 + c.length)}
            for v in near:
                v.select = True
            for e in bm.edges:
                if e.verts[0] in near and e.verts[1] in near:
                    e.select = True
            bm.select_history.clear()
            bm.normal_update()
            bmesh.update_edit_mesh(me)
            infos.insert(0, "bevel merged (width 0)")
            return None, True, infos
        # a flat cap cut into strips (bottom of a blind hole) would hold every vertex of the
        # outline: joined into one n-gon first. Only here: in the modes above, that n-gon would
        # become the "round part" reference face.
        grid_faces = [f for g, _ in grids for f in g.fset]
        if grid_faces and merge_caps(bm, grid_faces):
            infos.append("cap joined")
        profile = None
        if self.shape == "CUSTOM":
            profile = context.tool_settings.custom_bevel_profile_preset
            try:
                profile.update()
            except Exception:
                pass
        # a ring around a tube: its sides change (an open chain keeps the automatic direction)
        grids = _chain_dirs(grids, chain_verts if chain_closed else None, self.swap)
        plans = []
        for g, dirn in grids:
            for band in resample_bands(g, dirn):
                if band is None:
                    gg, n_old = g, direction_stats(g, dirn)[0]
                    kind = "around" if g.cyclic[dirn] else "across"
                else:
                    gg, n_old = _subgrid(g, dirn, band[0], band[1]), band[1] - band[0]
                    kind = "fillet"
                err, plan = plan_grid(bm, gg, dirn, self.segments, self, profile)
                if err:
                    return err, False, infos
                plans.append(plan)
                infos.append("%s %d -> %d" % (kind, n_old, self.segments))
        for f in bm.faces:
            f.select = False

        def do():
            for plan in plans:
                execute_plan(bm, plan)
            return True
        # started from an edge chain: the new chain is selected afterwards, not the faces
        chain_pick = None
        chain_pos = [v.co.copy() for v in chain_verts if v.is_valid] if chain_verts else []
        if chain_verts and not self.swap:
            # every profile that is one of the selected chains (both ends of a chamfer...)
            hits = [(len(chain_verts & set(prof)), p_i, k) for p_i, plan in enumerate(plans)
                    for k, prof in enumerate(plan["profs"])]
            best = max((h[0] for h in hits), default=0)
            if best:
                chain_pick = [(p_i, k) for n, p_i, k in hits if n == best or n >= 2]
        ok, _ = guarded(bm, do)
        if not ok:
            return "This part is joined to the rest of the mesh in a way that can't be changed safely.", False, infos
        if core is not None:
            infos.append("ends kept")
        prof = None
        if chain_pick:
            prof = [plans[p_i]["new_profs"][k] for p_i, k in chain_pick]
        elif chain_pairs and all(a.is_valid and b.is_valid and edge_between(a, b) for a, b in chain_pairs):
            # the chain itself was left as it is (e.g. a fillet band changed beside it): kept
            for e in bm.edges:
                e.select = False
            for v in bm.verts:
                v.select = False
            for f in bm.faces:
                f.select = False
            for a, b in chain_pairs:
                edge_between(a, b).select_set(True)
            bm.select_flush_mode()
            bm.select_history.clear()
            bm.normal_update()
            bmesh.update_edit_mesh(me)
            return None, True, infos
        elif chain_pos:
            # other direction changed: the line of the chain's direction nearest the old chain
            # (so S back still works on the chain's direction)
            best = None
            for plan in plans:
                nps = plan["new_profs"]
                if not nps or len({len(p) for p in nps}) != 1:
                    continue
                for i in range(len(nps[0])):
                    line = [p[i] for p in nps]
                    d = sum(min((v.co - c).length for c in chain_pos) for v in line) / len(line)
                    if best is None or d < best[0]:
                        best = (d, line)
            prof = [best[1]] if best else None
        if prof is not None:
            for f in bm.faces:
                f.select = False
            for e in bm.edges:
                e.select = False
            for v in bm.verts:
                v.select = False
            for line in prof:
                line = list(line) + (list(line[:1]) if chain_closed else [])
                for a, b in zip(line, line[1:]):
                    e = edge_between(a, b)
                    if e is not None:
                        e.select = True
                        a.select = b.select = True
            bm.select_history.clear()
            bm.normal_update()
            bmesh.update_edit_mesh(me)
            return None, True, infos
        bm.select_flush_mode()
        bm.normal_update()
        bmesh.update_edit_mesh(me)
        return None, True, infos


_DIGITS = {name: str(i) for i, name in enumerate(
    ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"))}
_DIGITS.update({"NUMPAD_%d" % i: str(i) for i in range(10)})

def _typing_target(op):
    """What a typed value sets: the shape after P, the width after W, else the segments."""
    if getattr(op, "_mouse_profile", False) and op.shape == "ROUND":
        return "shape"
    if getattr(op, "_mouse_width", False):
        return "width"
    return "segments"


_ICONS = None


def _icon(name):
    global _ICONS
    if _ICONS is None:
        try:
            _ICONS = {i.identifier for i in
                      bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items}
        except Exception:
            _ICONS = set()
    return name if name in _ICONS else "NONE"


def _status_draw(shape, mouse_width=False):
    """Status bar shortcuts, drawn like the ones of Blender's own modal tools."""
    def draw(header, context):
        lay = header.layout
        row = lay.row(align=True)
        row.label(text="Confirm", icon=_icon("MOUSE_LMB"))
        row.separator(factor=2)
        row.label(text="Cancel", icon=_icon("EVENT_ESC"))
        row.separator(factor=2)
        row.label(text="Segments", icon=_icon("MOUSE_MMB_SCROLL"))
        row.separator(factor=2)
        row.label(icon=_icon("EVENT_CTRL"))
        row.label(text="Multiple of 4", icon=_icon("MOUSE_MMB_SCROLL"))
        row.separator(factor=2)
        row.label(text="0-9  Type a Value")
        row.separator(factor=2)
        row.label(text="Other Direction", icon=_icon("EVENT_S"))
        row.separator(factor=2)
        row.label(text="Profile Type", icon=_icon("EVENT_R"))
        row.separator(factor=2)
        row.label(text="Width (Mouse)", icon=_icon("EVENT_W"))
        if shape != "KEEP" or mouse_width:
            row.separator(factor=2)
            row.label(text="Width", icon=_icon("MOUSE_MOVE"))
            row.separator(factor=2)
            row.label(text="Precision", icon=_icon("EVENT_SHIFT"))
        row.separator(factor=2)
        row.label(text="Shape", icon=_icon("EVENT_P"))
        row.separator(factor=2)
        row.label(text="Triangulate", icon=_icon("EVENT_T"))
        row.separator(factor=2)
        row.label(text="Navigate", icon=_icon("MOUSE_MMB"))
    return draw


def menu_func(self, context):
    self.layout.operator(MESH_OT_facetor.bl_idname)
