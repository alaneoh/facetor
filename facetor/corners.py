# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Alaneo Houssart
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

"""Fillet networks: fillets (quad strips) joined by corners where three of them meet (a box
bevelled on all its edges). The fillets change like grids, each corner is rebuilt with the same
pattern as Blender's Bevel (three quad sectors around a pole, plus a centre triangle for an odd
count), laid on the old corner's surface."""

import collections

import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .core import _face_components, edge_between, extract_grid, grid_profiles, ordered
from .grid import execute_plan, plan_grid


def has_bevel_poles(faces):
    """Inner vertices where three quads meet (the middle of a corner of Blender's Bevel)."""
    S = set(faces)
    return any(len(v.link_edges) == 3 and len(v.link_faces) == 3 and
               all(g in S and len(g.verts) == 4 for g in v.link_faces) for f in S for v in f.verts)


def _looks_like_network(S):
    """A pole (inner vertex of 3 quads, or of 5 or more: where that many fillets meet), or a
    triangle or a polygon closed in by selected faces."""
    for f in S:
        if len(f.verts) == 3 and all(g in S for e in f.edges for g in e.link_faces):
            return 1
        for v in f.verts:
            if len(v.link_edges) == 3 and len(v.link_faces) == 3 and \
                    all(g in S and len(g.verts) == 4 for g in v.link_faces):
                return 1
    for f in S:
        if len(f.verts) >= 5 and all(g in S and len(g.verts) == 4 for e in f.edges for g in e.link_faces
                                     if g is not f):
            return 2
        for v in f.verts:
            k = len(v.link_edges)
            if k >= 5 and len(v.link_faces) == k and all(g in S and len(g.verts) == 4 for g in v.link_faces):
                return 2
    return 0


def bevel_network(faces):
    """(network, None), (None, error message) when it looks like one that can't be done, or
    (None, None) when the faces aren't a fillet network."""
    S = set(faces)
    kind = _looks_like_network(S) if S else 0
    if not kind:
        return None, None
    if any(len(f.verts) > 4 for f in S):
        kind = 2       # n-gons (corners of fillets with different counts...): only if it all fits
    net, err = _bevel_network(S, faces)
    if err and kind == 2:
        return None, None      # only poles of 5+ quads / closed polygons: maybe something else
    return net, err


def _bevel_network(S, faces):
    """bevel_network() once the faces look like a network."""
    if any(len(f.verts) > 4 for f in S) and not any(len(f.verts) == 4 for f in S):
        return None, None
    border = [e for f in S for e in f.edges if sum(1 for g in e.link_faces if g in S) == 1]
    if not border:
        return None, None
    border_verts = {v for f in S for v in f.verts if any(g not in S for g in v.link_faces)}

    # rungs: face columns that cross the selection from one border edge to another (the
    # fillets' profiles); the faces no rung goes through are the corners
    rung_faces, lengths = set(), set()
    for e in ordered(set(border)):
        f = next(g for g in e.link_faces if g in S)
        cur, col = e, []
        while f is not None and len(f.verts) == 4 and f not in col:
            col.append(f)
            lp = next((l for l in f.loops if l.edge is cur), None)
            if lp is None:
                break
            cur = lp.link_loop_next.link_loop_next.edge
            nxt = [g for g in cur.link_faces if g is not f and g in S]
            if not nxt:
                rung_faces.update(col)
                lengths.add(len(col))
                break
            f = nxt[0] if len(nxt) == 1 else None
    corners = S - rung_faces
    if not corners or not rung_faces:
        return None, None
    if any(len(f.verts) > 4 for f in rung_faces):
        return None, None

    # fillets: their profiles go from one border edge to another (each may have its own count)
    strips, counts = [], []
    for comp in _face_components(ordered(rung_faces)):
        g, err = extract_grid(comp)
        if err:
            return None, err
        d = None
        for dd in (0, 1):
            profs = grid_profiles(g, dd)
            if g.cyclic[dd]:
                continue
            if len({len(p) for p in profs}) == 1 and len(profs[0]) - 1 in lengths and                     all(p[0] in border_verts and p[-1] in border_verts for p in profs):
                d = dd
                break
        if d is None:
            return None, "Couldn't find the profile of one of the fillets."
        strips.append((g, d))
        counts.append(len(grid_profiles(g, d)[0]) - 1)
    tally = collections.Counter(counts)
    n = max(tally, key=lambda c: (tally[c], c))
    equal = len(tally) == 1

    # corners: two or more corner points on the border, no border edge (every fillet around it is
    # selected); when the fillets have one count, Blender's pattern for that count
    m = n // 2
    corner_data = []
    border_set = set(border)
    for comp in _face_components(ordered(corners)):
        if any(e in border_set for f in comp for e in f.edges):
            return None, ("A corner joins a fillet that isn't selected: select every fillet around "
                          "the corners too.")
        pts = {v for f in comp for v in f.verts if v in border_verts}
        k = len(pts)
        if k < 2:
            return None, "This corner isn't made like Blender's Bevel: it can't be rebuilt."
        if equal and k >= 3 and len(comp) != k * m * m + (k * m + 1 if n % 2 else 0):
            return None, "This corner isn't made like Blender's Bevel: it can't be rebuilt."
        pts = ordered(pts)
        corner_data.append({"faces": comp, "points": [v.co.copy() for v in pts], "pverts": pts,
                            "material": comp[0].material_index, "smooth": comp[0].smooth})
    return {"strips": strips, "corners": corner_data, "count": n, "counts": sorted(tally)}, None


def _coons(bottom, top, left, right):
    """Grid of points (i along bottom/top, j along left/right) from its four sides:
    bottom[i] = P(i, 0), top[i] = P(i, m), left[j] = P(0, j), right[j] = P(m, j)."""
    mi, mj = len(bottom) - 1, len(left) - 1
    P = [[None] * (mj + 1) for _ in range(mi + 1)]
    for i in range(mi + 1):
        u = i / mi if mi else 0.0
        for j in range(mj + 1):
            v = j / mj if mj else 0.0
            P[i][j] = ((1 - v) * bottom[i] + v * top[i] + (1 - u) * left[j] + u * right[j]
                       - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[mi]
                          + (1 - u) * v * top[0] + u * v * top[mi]))
    return P


def _fit_sphere(pts):
    """(centre, radius, relative error) of the sphere through pts (least squares), or None."""
    import numpy as np
    if len(pts) < 4:
        return None
    A = np.array([[2 * p.x, 2 * p.y, 2 * p.z, 1.0] for p in pts])
    b = np.array([p.dot(p) for p in pts])
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    c = Vector(sol[:3])
    r2 = sol[3] + c.dot(c)
    if r2 <= 0:
        return None
    r = r2 ** 0.5
    err = max(abs((p - c).length - r) for p in pts) / r
    return c, r, err


def _fill_corner(bm, arcs, n, project, material, smooth):
    """Fill the corner closed by k arcs of n segments each (A: P0->P1, B: P1->P2, ...,
    the last back to P0, lists of BMVerts) with Blender's Bevel pattern (k sectors of quads
    around a centre vertex, plus a centre k-gon for an odd count). Two arcs (a corner between
    two points): strips between them, triangles at both ends. Returns the new faces."""
    new_faces = []
    k = len(arcs)

    def vert(co):
        return bm.verts.new(project(co))

    def face(vs):
        if len(set(vs)) < 3:
            return
        f = bm.faces.new(vs)
        f.material_index = material
        f.smooth = smooth
        new_faces.append(f)

    if k == 2:
        A, B = arcs[0], arcs[1][::-1]
        mids = [project(a.co.lerp(b.co, 0.5)) for a, b in zip(A, B)]
        M = [A[0]] + [bm.verts.new(p) for p in mids[1:-1]] + [A[-1]]
        for i in range(n):
            face([A[i], A[i + 1], M[i + 1], M[i]])
            face([M[i], M[i + 1], B[i + 1], B[i]])
        return new_faces
    if n == 1:
        face([a[0] for a in arcs])
        return new_faces
    m = n // 2
    bpts = [v.co for a in arcs for v in a]
    centre = project(sum(bpts, Vector()) / len(bpts))
    cv = None

    def spoke(a_vert, b_co, b_vert=None):
        """a_vert -> point b in m segments (new vertices in between, b_vert at the end)."""
        out = [a_vert]
        for j in range(1, m):
            out.append(vert(a_vert.co.lerp(b_co, j / m)))
        out.append(b_vert)
        return out

    if n % 2 == 0:
        cv = bm.verts.new(centre)
        spokes = [spoke(a[m], centre, cv) for a in arcs]
        for s in range(k):
            X, Y = arcs[s], arcs[(s + 1) % k]
            sx, sy = spokes[s], spokes[(s + 1) % k]
            bottom = [v.co for v in X[m:]]               # X[m] .. corner
            right = [v.co for v in Y[:m + 1]]             # corner .. Y[m]
            left = [v.co for v in sx]                     # X[m] .. centre
            top = [v.co for v in reversed(sy)]            # centre .. Y[m]
            P = _coons(bottom, top, left, right)
            G = [[None] * (m + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                G[i][0] = X[m + i]
                G[i][m] = sy[m - i]
            for j in range(m + 1):
                G[0][j] = sx[j]
                G[m][j] = Y[j]
            for i in range(1, m):
                for j in range(1, m):
                    G[i][j] = vert(P[i][j])
            for i in range(m):
                for j in range(m):
                    face([G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1]])
        return new_faces

    # odd: a centre triangle, one vertex towards each corner point
    tri = []
    for s in range(k):
        q = arcs[s][-1]                                   # corner point between arc s and s+1
        tri.append(bm.verts.new(project(centre.lerp(q.co, 1.0 / n))))
    # spokes from the ends of each arc's middle segment to the triangle
    s_start, s_end = [], []
    for s in range(k):
        X = arcs[s]
        s_start.append(spoke(X[m], tri[s - 1].co, tri[s - 1]))     # towards the previous corner's tri vertex
        s_end.append(spoke(X[m + 1], tri[s].co, tri[s]))
    for s in range(k):
        X, Y = arcs[s], arcs[(s + 1) % k]
        sx, sy = s_end[s], s_start[(s + 1) % k]
        bottom = [v.co for v in X[m + 1:]]
        right = [v.co for v in Y[:m + 1]]
        left = [v.co for v in sx]
        top = [v.co for v in reversed(sy)]
        P = _coons(bottom, top, left, right)
        G = [[None] * (m + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            G[i][0] = X[m + 1 + i]
            G[i][m] = sy[m - i]
        for j in range(m + 1):
            G[0][j] = sx[j]
            G[m][j] = Y[j]
        for i in range(1, m):
            for j in range(1, m):
                G[i][j] = vert(P[i][j])
        for i in range(m):
            for j in range(m):
                face([G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1]])
        # the strip along the middle segment of arc s
        a, b = s_start[s], s_end[s]
        for j in range(m):
            face([a[j], b[j], b[j + 1], a[j + 1]])
    face(tri)
    return new_faces


def network_rebuild(bm, net, count, op, ctx_profile):
    """Change every fillet of the network to `count` segments and rebuild its corners.
    Returns an error message or None. Run it inside guarded()."""
    keep = op.shape == "KEEP" and abs(getattr(op, "width", 1.0) - 1.0) < 1e-6
    # old corner surfaces (Keep: the new corners are laid on them)
    for c in net["corners"]:
        verts = ordered({v for f in c["faces"] for v in f.verts})
        idx = {v: i for i, v in enumerate(verts)}
        c["bvh"] = BVHTree.FromPolygons([v.co.copy() for v in verts],
                                        [[idx[v] for v in f.verts] for f in c["faces"]])
        # a round corner (Blender's Bevel, equal widths): its sphere, smoother than its old facets
        c["sphere"] = _fit_sphere([v.co.copy() for v in verts])
    size = max((p - q).length for c in net["corners"] for p in c["points"] for q in c["points"]) or 1.0
    S = {f for g, _ in net["strips"] for f in g.fset} | {f for c in net["corners"] for f in c["faces"]}
    outside = {g for f in S for e in f.edges for g in e.link_faces if g not in S}
    bmesh.ops.delete(bm, geom=list({f for c in net["corners"] for f in c["faces"]}), context="FACES")

    plans = []
    for g, d in net["strips"]:
        err, plan = plan_grid(bm, g, d, count, op, ctx_profile)
        if err:
            return err
        plans.append(plan)
    for plan in plans:
        execute_plan(bm, plan)

    ends = []
    for plan in plans:
        profs = plan["new_profs"]
        for p in (profs[0], profs[-1]):
            ends.append(list(p))

    tol = 1e-4 * size
    out_faces = []
    for c in net["corners"]:
        pts = c["points"]

        def which(v, pv=c["pverts"]):
            # the corner points stay the same vertices (they may move with a width change)
            for i, q in enumerate(pv):
                if q.is_valid and q is v:
                    return i
            return next((i for i, p in enumerate(pts) if (v.co - p).length < tol), None)
        arcs = {}
        for p in ends:
            a, b = which(p[0]), which(p[-1])
            if a is None or b is None or a == b:
                continue
            arcs[(a, b)] = p
            arcs[(b, a)] = list(reversed(p))
        k = len(pts)
        if k == 3:
            try:
                ring = [arcs[(0, 1)], arcs[(1, 2)], arcs[(2, 0)]]
            except KeyError:
                return "Couldn't find the three fillets around a corner."
        elif k == 2:
            # two fillets end on the same two points: both arcs, from point 0 and back
            two = [p for p in ends if {which(p[0]), which(p[-1])} == {0, 1}]
            if len(two) != 2:
                return "Couldn't find the two fillets around a corner."
            ring = [two[0] if which(two[0][0]) == 0 else list(reversed(two[0])),
                    two[1] if which(two[1][0]) == 1 else list(reversed(two[1]))]
        else:
            # the arcs around the corner in order, from point 0 back to it
            ring, cur, prev = [], 0, None
            for _ in range(k):
                nxt = sorted(b for (a, b) in arcs if a == cur and b != prev)
                if not nxt or (prev is not None and len(nxt) != 1):
                    return "Couldn't find the fillets around a corner."
                ring.append(arcs[(cur, nxt[0])])
                prev, cur = cur, nxt[0]
            if cur != 0:
                return "Couldn't find the fillets around a corner."
        if any(len(x) != count + 1 for x in ring):
            return "The fillets around a corner didn't get the same number of segments."
        A = ring[0]
        if keep and not (c["sphere"] and c["sphere"][2] < 0.01):
            bvh = c["bvh"]

            def project(co, bvh=bvh):
                hit = bvh.find_nearest(co)
                return hit[0] if hit and hit[0] is not None else co
        else:
            sph = c["sphere"] if keep else _fit_sphere([v.co for a in ring for v in a])

            def project(co, sph=sph):
                if sph and sph[2] < 0.02:
                    cen, r, _ = sph
                    d = co - cen
                    return cen + d.normalized() * r if d.length > 1e-12 else co
                return co
        new = _fill_corner(bm, ring, count, project, c["material"], c["smooth"])
        # same winding as the fillet next to it
        e = edge_between(A[0], A[1])
        nb = next((f for f in e.link_faces if f not in new), None) if e else None
        mine = next((f for f in e.link_faces if f in new), None) if e else None
        if nb and mine:
            def runs(f):
                for l in f.loops:
                    if l.edge is e:
                        return l.vert
            if runs(nb) is runs(mine):
                for f in new:
                    f.normal_flip()
        out_faces += new
    # the whole new network selected (the faces joined to the new corners, up to the faces
    # around it), so Alt+B can go on with it
    for f in bm.faces:
        f.select = False
    region, stack = set(out_faces), list(out_faces)
    while stack:
        f = stack.pop()
        for e in f.edges:
            for g in e.link_faces:
                if g not in region and g not in outside:
                    region.add(g)
                    stack.append(g)
    for f in region:
        f.select_set(True)
    return None


def _network_targets(net):
    """Where each vertex of the network goes at width 0: a fillet's profile to its sharp corner,
    a corner (with the ends of the fillets around it) to its point. (targets, error)."""
    from .core import bevel_corner
    from .grid import _corner_ok
    target = {}
    for g, d in net["strips"]:
        profs = grid_profiles(g, d)
        rails_cyclic = g.cyclic[1 - d]
        for k, p in enumerate(profs):
            c = bevel_corner(profs, k, g.fset, rails_cyclic)
            if not _corner_ok(c, p):
                return None, "Couldn't find the corner of a fillet from the faces next to it."
            for v in p:
                target[v] = c
    points = []
    for c in net["corners"]:
        verts = ordered({v for f in c["faces"] for v in f.verts})
        known = [target[v] for v in verts if v in target]
        if not known:
            return None, "Couldn't find where a corner merges."
        p = sum(known, Vector()) / len(known)
        for v in verts:
            target[v] = p
        points.append(p)
    return target, None


def network_scale(net, width):
    """Width of the whole network scaled: each vertex moves towards (or away from) where it
    would merge, so the fillets and the corners stay round and the flat faces between them
    grow or shrink at their corners."""
    target, err = _network_targets(net)
    if err:
        return err
    for v, p in target.items():
        v.co = p + (v.co - p) * width
    return None


def network_collapse(bm, net):
    """Width 0: every fillet merged into its sharp edge and every corner into its point.
    Returns (error message or None, merged points)."""
    target, err = _network_targets(net)
    if err:
        return err + " It can't be merged.", None
    points = list(target.values())
    for v, p in target.items():
        v.co = p
    size = max(((p - points[0]).length for p in points), default=1.0) or 1.0
    bmesh.ops.remove_doubles(bm, verts=list(target), dist=1e-6 * size)
    return None, sorted({p.freeze() for p in target.values()}, key=tuple)
