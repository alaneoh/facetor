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

"""Holes: an edge loop in a flat or curved area, with or without a wall."""



from mathutils.bvhtree import BVHTree

from .core import edge_between, hole_frame, hole_points, ordered, radial_scaler
from .grid import _free_flat_side, _resize_chain
from .safety import guarded
from .strips import strip_rebuild


# ---------------------------------------------------------------------------
# Holes: a selected closed edge loop (in a flat face, or crossing a curved area)
# ---------------------------------------------------------------------------

def selected_loops(bm):
    edges = [e for e in bm.edges if e.select]
    adj = {}
    for e in edges:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    if not adj or any(len(l) != 2 for l in adj.values()):
        return None
    loops, seen = [], set()
    for v0 in adj:
        if v0 in seen:
            continue
        loop, prev, cur = [v0], None, v0
        seen.add(v0)
        while True:
            e = adj[cur][0] if adj[cur][0] is not prev else adj[cur][1]
            nxt = e.other_vert(cur)
            if nxt is v0:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = e, nxt
        loops.append(loop)
    return loops


def hole_wall(bm, loop, frame):
    """Faces of the hole's wall: connected faces next to the outline lying on the same cylinder."""
    axis, c, r, u, w = frame
    tol = r * 0.01

    def on_cyl(v):
        d = v.co - c
        return abs((d - axis * d.dot(axis)).length - r) < tol
    lset = set(loop)
    start = set()
    for i, v in enumerate(loop):
        e = edge_between(v, loop[(i + 1) % len(loop)])
        if e:
            for f in e.link_faces:
                if all(on_cyl(x) for x in f.verts) and any(x not in lset for x in f.verts):
                    start.add(f)
    wall, stack = set(start), list(start)
    while stack:
        f = stack.pop()
        for e in f.edges:
            for g in e.link_faces:
                if g not in wall and all(on_cyl(x) for x in g.verts):
                    wall.add(g)
                    stack.append(g)
    return ordered(wall)


def tube_loop(bm, loop, frame=None):
    """True if the closed loop runs around a tube rather than a hole's rim: the faces on both
    sides of each of its edges lie on the loop's cylinder."""
    edges = [edge_between(v, loop[(i + 1) % len(loop)]) for i, v in enumerate(loop)]
    if all(e is not None and len(e.link_faces) == 2 and all(len(f.verts) == 4 for f in e.link_faces)
           and e.link_faces[0].normal.angle(e.link_faces[1].normal, 0.0) < 0.87 for e in edges):
        # quads on both sides and no crease along the loop (a hole's rim is a sharp edge): a ring
        # around a smooth tube, bent or not (a torus)
        return True
    if frame is None:
        frame = hole_frame([v.co.copy() for v in loop])
        if frame is None:
            return False
    axis, c, r, _, _ = frame

    def on_cyl(v):
        d = v.co - c
        return abs((d - axis * d.dot(axis)).length - r) < r * 0.01
    for i, v in enumerate(loop):
        e = edge_between(v, loop[(i + 1) % len(loop)])
        if e is None or len(e.link_faces) != 2 or \
                not all(all(on_cyl(x) for x in f.verts) for f in e.link_faces):
            return False
    return True


def _hole_width(bm, loops, wall, frame, width):
    """(point -> point scaling the hole's radius, None) or (None, error). The new outline slides
    onto the faces around it; it must stay inside the first ring of vertices around the hole."""
    if width < 1e-4:
        return None, "A hole can't be merged (width 0): use a width above 0."
    axis, c, r, _, _ = frame
    inside = {v for loop in loops for v in loop} | {v for f in wall for v in f.verts}
    wset = set(wall)
    around = ordered({f for loop in loops for v in loop for f in v.link_faces if f not in wset})

    def flat(p):
        d = p - c
        return d - axis * d.dot(axis)

    def seg_dist(a, b):
        # distance from the axis to the edge a-b, seen along the axis
        a, b = flat(a), flat(b)
        ab = b - a
        t = 0.0 if ab.length < 1e-12 else max(0.0, min(1.0, -a.dot(ab) / ab.dot(ab)))
        return (a + ab * t).length
    out = [seg_dist(l.vert.co, l.link_loop_next.vert.co) for f in around for l in f.loops
           if l.vert not in inside and l.link_loop_next.vert not in inside]
    out += [flat(v.co).length for f in around for v in f.verts if v not in inside]
    if width > 1.0 and out and r * width > 0.98 * min(out):
        return None, ("This width makes the hole reach the edges around it (at most %.2f here)."
                      % max(1.0, 0.98 * min(out) / r))
    vmap, coords, polys = {}, [], []
    for f in around:
        poly = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = len(coords)
                coords.append(v.co.copy())
            poly.append(vmap[v])
        polys.append(poly)
    bvh = BVHTree.FromPolygons(coords, polys) if polys else None
    return radial_scaler(frame, width, bvh), None


def hole_rebuild(bm, loop, count, width=1.0):
    if count < 3:
        return "A hole needs at least 3 sides."
    pts = [v.co.copy() for v in loop]
    frame = hole_frame(pts)
    if frame is None:
        return "This loop isn't round (seen from any direction)."
    if tube_loop(bm, loop, frame):
        return ("This edge loop goes around a tube, not a hole: select the tube's faces, or one "
                "edge chain along it.")
    wall = hole_wall(bm, loop, frame)
    scale = None
    if abs(width - 1.0) > 1e-6:
        loops = [loop]
        if wall:
            # the wall's other outline (the far side of a plate, the floor of a blind hole)
            wv = {v for f in wall for v in f.verts}
            loops.append([v for v in wv if v not in set(loop) and
                          any(g not in set(wall) for g in v.link_faces)])
        scale, err = _hole_width(bm, loops, wall, frame, width)
        if err:
            return err
    if wall:
        return strip_rebuild(bm, wall, count, transform=scale, first=loop)
    targets = hole_points(frame, pts, count)[0]
    if scale is not None:
        targets = [scale(p) for p in targets]

    def do():
        if count < len(loop):
            _free_flat_side(bm, loop, True, fans_only=True)     # flat area cut into triangles around the hole
        chain = _resize_chain(bm, loop, True, count, targets)
        if len(chain) != count:
            return False
        for v, p in zip(chain, targets):
            v.co = p
        for v in chain:
            v.select = True
        return True
    ok, _ = guarded(bm, do)
    if not ok:
        return "This hole's outline is joined to the mesh in a way that can't be changed safely."
    return None
