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

"""Shared helpers: quad grids, curve fitting and resampling, profiles."""

import math
import bmesh
import mathutils
import numpy as np
from mathutils import Vector


AXES = (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
TAU = 2 * math.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quad_components(quads):
    qset = set(quads)
    seen, comps = set(), []
    for f in quads:
        if f in seen:
            continue
        comp, stack = [], [f]
        seen.add(f)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for e in cur.edges:
                for o in e.link_faces:
                    if o in qset and o not in seen:
                        seen.add(o)
                        stack.append(o)
        comps.append(comp)
    return comps


def elem_key(x):
    """Sort key of a BMesh element by position. Sets of BMesh elements iterate in an order that
    changes from one run to the next (their hash is a memory address): sorted with this key
    wherever the order matters, results are the same every time."""
    if isinstance(x, bmesh.types.BMVert):
        return tuple(x.co)
    if isinstance(x, bmesh.types.BMEdge):
        a, b = sorted((tuple(x.verts[0].co), tuple(x.verts[1].co)))
        return a + b
    if isinstance(x, bmesh.types.BMFace):
        c = x.calc_center_median()
        return (c.x, c.y, c.z, len(x.verts))
    if isinstance(x, bmesh.types.BMLoop):
        return tuple(x.vert.co) + elem_key(x.face)
    return x


def ordered(items):
    """BMesh elements (any iterable, often a set) in a fixed order: see elem_key."""
    return sorted(items, key=elem_key)


def edge_between(a, b):
    for e in a.link_edges:
        if e.other_vert(a) is b:
            return e
    return None


def newell(pts, center):
    n = Vector()
    for i, p in enumerate(pts):
        n += (p - center).cross(pts[(i + 1) % len(pts)] - center)
    return n.normalized() if n.length > 1e-12 else None


# ---------------------------------------------------------------------------
# Grid extraction: give every vertex of a quad patch (i, j) coordinates.
# Handles strips, tubes (one closed direction) and tori (two closed directions).
# ---------------------------------------------------------------------------

class Grid:
    pass


STRAIGHT = math.radians(160)


def quad_corners(f, fset):
    """The 4 corners of face f in loop order: its vertices for a quad; for an n-gon whose other
    vertices are straight and lie on the border of fset (CAD: a line of the next face ends on
    its edge, a T-junction), its 4 real corners. None otherwise."""
    if len(f.verts) == 4:
        return [l.vert for l in f.loops]
    if len(f.verts) < 4:
        return None
    corners = []
    for l in f.loops:
        if l.calc_angle() > STRAIGHT:
            for e in (l.edge, l.link_loop_prev.edge):
                if any(g is not f and g in fset for g in e.link_faces):
                    return None
        else:
            corners.append(l.vert)
    return corners if len(corners) == 4 else None


def border_extras(faces, fset):
    """{frozenset((a, b)): [a, t1, ..., b]} for the T-junction vertices t on the border side a-b
    of the n-gons that quad_corners accepts: new faces built on a-b must keep them."""
    out = {}
    for f in faces:
        if len(f.verts) <= 4:
            continue
        cs = quad_corners(f, fset)
        if cs is None:
            continue
        loop = [l.vert for l in f.loops]
        n = len(loop)
        for i, a in enumerate(cs):
            b = cs[(i + 1) % 4]
            ia = loop.index(a)
            path = [a]
            j = (ia + 1) % n
            while loop[j] is not b:
                path.append(loop[j])
                j = (j + 1) % n
            path.append(b)
            if len(path) > 2:
                out[frozenset((a, b))] = path
    return out


def with_extras(vs, extras):
    """Face vertex list vs with the T-junction vertices of `extras` put back on its sides."""
    if not extras:
        return vs
    out = []
    n = len(vs)
    for i, a in enumerate(vs):
        b = vs[(i + 1) % n]
        out.append(a)
        path = extras.get(frozenset((a, b)))
        if path:
            mid = path[1:-1] if path[0] is a else path[-2:0:-1]
            out += mid
    return out


def _other_neighbour(cyc, v, not_v):
    """Neighbour of corner v in the corner cycle cyc of a face, other than not_v."""
    if v not in cyc:
        return None
    i = cyc.index(v)
    a, b = cyc[(i + 1) % len(cyc)], cyc[i - 1]
    return b if a is not_v else a


def extract_grid(comp, active=None):
    fset = set(comp)
    cyc = {}
    for f in comp:
        cs = quad_corners(f, fset)
        if cs is None:
            return None, "Selection is not a connected quad patch."
        cyc[f] = cs
    f0 = active if active in fset else comp[0]
    c0 = cyc[f0]
    corners0 = {c0[0]: (0, 0), c0[1]: (1, 0), c0[2]: (1, 1), c0[3]: (0, 1)}
    face_corners = {f0: corners0}
    stack = [f0]
    while stack:
        f = stack.pop()
        cf = face_corners[f]
        for e in f.edges:
            a, b = e.verts
            for g in e.link_faces:
                if g is f or g not in fset or g in face_corners:
                    continue
                d = _other_neighbour(cyc[f], a, b)
                c = _other_neighbour(cyc[f], b, a)
                a2 = _other_neighbour(cyc[g], a, b)
                b2 = _other_neighbour(cyc[g], b, a)
                if None in (d, c, a2, b2):
                    return None, "Unexpected topology in the selection."
                ca, cb, cd, cc = cf[a], cf[b], cf[d], cf[c]
                face_corners[g] = {
                    a: ca, b: cb,
                    a2: (2 * ca[0] - cd[0], 2 * ca[1] - cd[1]),
                    b2: (2 * cb[0] - cc[0], 2 * cb[1] - cc[1]),
                }
                stack.append(g)
    if len(face_corners) != len(fset):
        return None, "Selection is not a connected quad patch."

    vcoords = {}
    for f, cf in face_corners.items():
        for v, c in cf.items():
            vcoords.setdefault(v, set()).add(c)

    # periods of the grid: a vertex reached with two coordinates was reached around a closed
    # direction. A torus is closed both ways: a vertex can be reached around both at once
    # (a diagonal difference), which is fine as long as it is made of the two periods.
    diffs = set()
    for v, cs in vcoords.items():
        cs = list(cs)
        for c in cs[1:]:
            di, dj = c[0] - cs[0][0], c[1] - cs[0][1]
            if di or dj:
                diffs.add((abs(di), abs(dj)) if di >= 0 else (abs(di), -dj if dj else 0))
    pi = pj = 0
    for di, dj in diffs:
        if di and not dj:
            pi = math.gcd(pi, di)
        elif dj and not di:
            pj = math.gcd(pj, abs(dj))
    changed = True
    while changed:
        changed = False
        for di, dj in diffs:
            if di and dj:
                if pi and di % pi == 0 and (not pj or dj % pj):
                    npj = math.gcd(pj, abs(dj))
                    changed |= npj != pj
                    pj = npj
                elif pj and dj % pj == 0 and (not pi or di % pi):
                    npi = math.gcd(pi, di)
                    changed |= npi != pi
                    pi = npi
    for di, dj in diffs:
        if (di and not (pi and di % pi == 0)) or (dj and not (pj and dj % pj == 0)):
            return None, "The quad patch is twisted (not a clean grid)."

    all_c = [c for cs in vcoords.values() for c in cs]
    imin = min(c[0] for c in all_c)
    jmin = min(c[1] for c in all_c)

    def canon(c):
        i = c[0] % pi if pi else c[0] - imin
        j = c[1] % pj if pj else c[1] - jmin
        return (i, j)

    V = {}
    for v, cs in vcoords.items():
        cc = {canon(c) for c in cs}
        if len(cc) != 1:
            return None, "The quad patch overlaps itself."
        key = cc.pop()
        if key in V and V[key] is not v:
            return None, "The quad patch is not a clean grid (poles or merged vertices)."
        V[key] = v
    ni = pi if pi else max(c[0] for c in all_c) - imin + 1
    nj = pj if pj else max(c[1] for c in all_c) - jmin + 1
    if len(V) != ni * nj:
        return None, "The quad patch is not rectangular (holes, poles or uneven rows)."

    faces = {}
    for f, cf in face_corners.items():
        mi = min(c[0] for c in cf.values())
        mj = min(c[1] for c in cf.values())
        key = canon((mi, mj))
        faces[key] = (f, {v: (c[0] - mi, c[1] - mj) for v, c in cf.items()})

    g = Grid()
    g.V, g.faces, g.fset = V, faces, fset
    g.extras = border_extras(comp, fset)
    g.n = (ni, nj)
    g.cyclic = (bool(pi), bool(pj))
    return g, None


def grid_profiles(g, d):
    """Profiles run along dimension d; there is one per index of the other dimension."""
    n_along, n_other = g.n[d], g.n[1 - d]
    profs = []
    for k in range(n_other):
        if d == 0:
            profs.append([g.V[(a, k)] for a in range(n_along)])
        else:
            profs.append([g.V[(k, a)] for a in range(n_along)])
    return profs


def direction_stats(g, d):
    """(segments, mean turning per segment in radians, mean profile length)"""
    closed = g.cyclic[d]
    profs = grid_profiles(g, d)
    segs = len(profs[0]) if closed else len(profs[0]) - 1
    turns, lengths = [], []
    for p in profs:
        pts = [v.co for v in p]
        if closed:
            pts = pts + pts[:2]
        vecs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
        vecs = [x for x in vecs if x.length > 1e-12]
        for i in range(len(vecs) - 1):
            turns.append(vecs[i].angle(vecs[i + 1]))
        lengths.append(sum(x.length for x in vecs[:segs]))
    t = sum(turns) / len(turns) if turns else 0.0
    return segs, t, sum(lengths) / len(lengths)


def side_tangency(g, d):
    """Mean angle between the grid and the outside faces along the side rails of direction d."""
    profs = grid_profiles(g, d)
    angs = []
    for k in range(len(profs) - 1):
        for idx in (0, -1):
            e = edge_between(profs[k][idx], profs[k + 1][idx])
            if not e:
                continue
            inside = [f for f in e.link_faces if f in g.fset]
            outside = [f for f in e.link_faces if f not in g.fset]
            for fi in inside:
                for fo in outside:
                    angs.append(fi.normal.angle(fo.normal, math.pi / 2))
    return sum(angs) / len(angs) if angs else math.pi / 2


def auto_direction(g):
    s0, t0, l0 = direction_stats(g, 0)
    s1, t1, l1 = direction_stats(g, 1)
    S, T, Lg = (s0, s1), (t0, t1), (l0, l1)
    c0, c1 = g.cyclic
    small = math.radians(3)
    if c0 != c1:
        cyc = 0 if c0 else 1
        opn = 1 - cyc
        if S[opn] == 1 or T[opn] < small:
            return cyc
        if Lg[opn] < 0.5 * Lg[cyc] and T[opn] > T[cyc]:
            return opn
        return cyc
    if not c0 and not c1:
        for d in (0, 1):
            if S[d] == 1 and S[1 - d] > 1 and T[1 - d] < small:
                return d  # chamfer strip: go across
        if S[0] == 1 and S[1] == 1:
            # single quad: go across the direction where the neighbours are most tangent
            return 0 if side_tangency(g, 0) <= side_tangency(g, 1) else 1
    if abs(T[0] - T[1]) < 1e-6:
        return 0 if S[0] >= S[1] else 1
    return 0 if T[0] > T[1] else 1


# ---------------------------------------------------------------------------
# Curve fitting / resampling ("Keep silhouette")
# ---------------------------------------------------------------------------

def fit_circle(pts):
    P = np.array([tuple(p) for p in pts], dtype=float)
    c = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - c)
    if S[0] < 1e-12 or S[1] < 1e-6 * S[0]:
        return None  # collinear
    ex, ey, ez = Vt[0], Vt[1], Vt[2]
    xy = (P - c) @ np.c_[ex, ey]
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = sol
    r2 = k + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    resid = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r).max()
    flat = np.abs((P - c) @ ez).max()
    center = Vector(tuple(c + cx * ex + cy * ey))
    return center, Vector(tuple(ez)), r, max(resid, flat)


def is_collinear(pts):
    P = np.array([tuple(p) for p in pts], dtype=float)
    c = P.mean(axis=0)
    S = np.linalg.svd(P - c, compute_uv=False)
    return S[0] < 1e-12 or S[1] < 1e-6 * S[0]


def _cr_point(p0, p1, p2, p3, t, alpha=0.5):
    def tj(ti, a, b):
        return ti + max((b - a).length, 1e-9) ** alpha
    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    tt = t1 + (t2 - t1) * t
    a1 = p0 * ((t1 - tt) / (t1 - t0)) + p1 * ((tt - t0) / (t1 - t0))
    a2 = p1 * ((t2 - tt) / (t2 - t1)) + p2 * ((tt - t1) / (t2 - t1))
    a3 = p2 * ((t3 - tt) / (t3 - t2)) + p3 * ((tt - t2) / (t3 - t2))
    b1 = a1 * ((t2 - tt) / (t2 - t0)) + a2 * ((tt - t0) / (t2 - t0))
    b2 = a2 * ((t3 - tt) / (t3 - t1)) + a3 * ((tt - t1) / (t3 - t1))
    return b1 * ((t2 - tt) / (t2 - t1)) + b2 * ((tt - t1) / (t2 - t1))


def resample_curve(pts, count, closed, sub=24):
    """Smooth (centripetal Catmull-Rom) curve through pts, sampled at equal arc length.
    Open: count+1 points incl. exact ends. Closed: count points starting at pts[0]."""
    n = len(pts)
    dense = []
    segs = n if closed else n - 1
    for i in range(segs):
        p1, p2 = pts[i], pts[(i + 1) % n]
        if closed:
            p0, p3 = pts[(i - 1) % n], pts[(i + 2) % n]
        else:
            p0 = pts[i - 1] if i > 0 else p1 * 2 - p2
            p3 = pts[i + 2] if i + 2 < n else p2 * 2 - p1
        for s in range(sub):
            dense.append(_cr_point(p0, p1, p2, p3, s / sub))
    dense.append(pts[0].copy() if closed else pts[-1].copy())
    cum = [0.0]
    for i in range(1, len(dense)):
        cum.append(cum[-1] + (dense[i] - dense[i - 1]).length)
    total = cum[-1]
    out = []
    num = count if closed else count + 1
    j = 0
    for k in range(num):
        target = total * k / count
        while j < len(cum) - 2 and cum[j + 1] < target:
            j += 1
        seg = cum[j + 1] - cum[j]
        t = 0 if seg < 1e-12 else (target - cum[j]) / seg
        out.append(dense[j].lerp(dense[j + 1], t))
    if not closed:
        out[0], out[-1] = pts[0].copy(), pts[-1].copy()
    return out


def keep_open(pts, count):
    if count < 1:
        count = 1
    if len(pts) == 2 or is_collinear(pts):
        return [pts[0].lerp(pts[-1], i / count) for i in range(count + 1)]
    fit = fit_circle(pts)
    if fit:
        c, n, r, err = fit
        if err < 0.01 * r:
            u = (pts[0] - c)
            u = (u - n * u.dot(n)).normalized()
            w = n.cross(u)

            def ang(p):
                d = p - c
                return math.atan2(d.dot(w), d.dot(u))
            if ang(pts[1]) < 0:
                w = -w
            angs = [ang(p) % TAU for p in pts]
            angs[0] = 0.0
            # unwrap so the sequence increases
            for i in range(1, len(angs)):
                while angs[i] < angs[i - 1] - 1e-9:
                    angs[i] += TAU
            end = angs[-1]
            out = [c + r * (math.cos(end * i / count) * u + math.sin(end * i / count) * w) for i in range(count + 1)]
            out[0], out[-1] = pts[0].copy(), pts[-1].copy()
            return out
    return resample_curve(pts, count, closed=False)


def keep_closed(pts, count, ref_axis, anchor=None):
    """Regular polygon if the loop is round, else smooth resample. First point near pts[0].
    anchor: index of an old point a new point must sit on (UV seam of a cylinder)."""
    fit = fit_circle(pts)
    c0 = sum(pts, Vector()) / len(pts)
    normal = newell(pts, c0)
    if fit and normal is not None and fit[3] < 0.01 * fit[2]:
        c, _, r, _ = fit
        m = len(pts)
        u = ref_axis - normal * ref_axis.dot(normal) if ref_axis is not None else Vector()
        if u.length < 1e-6:
            u = pts[0] - c
            u = u - normal * u.dot(normal)
        u.normalize()
        w = normal.cross(u)
        angs = [math.atan2((p - c).dot(w), (p - c).dot(u)) for p in pts]
        step_old, step_new = TAU / m, TAU / count
        delta = min(angs, key=abs)
        if anchor is not None:
            base = angs[anchor]
        elif abs(delta) <= step_old / 4:
            base = delta
        else:
            base = delta - math.copysign(step_old / 2, delta) + step_new / 2
        a0 = angs[0]
        shift = int(round((a0 - base) / step_new)) % count
        return [c + r * (math.cos(base + step_new * ((i + shift) % count)) * u +
                         math.sin(base + step_new * ((i + shift) % count)) * w) for i in range(count)]
    if anchor:
        # start the resampling on the anchor, then put the point nearest pts[0] first again
        out = resample_curve(pts[anchor:] + pts[:anchor], count, closed=True)
        s = min(range(count), key=lambda i: (out[i] - pts[0]).length)
        return out[s:] + out[:s]
    return resample_curve(pts, count, closed=True)


def choose_ref_axis(closed_profiles_pts):
    normals = []
    for pts in closed_profiles_pts:
        c = sum(pts, Vector()) / len(pts)
        n = newell(pts, c)
        if n is not None:
            normals.append(n)
    if not normals:
        return None
    score = lambda a: max(abs(a.dot(n)) for n in normals)
    best = min(score(a) for a in AXES)
    cands = [a for a in AXES if score(a) < best + 1e-3]
    pts0, n0 = closed_profiles_pts[0], normals[0]
    c0 = sum(pts0, Vector()) / len(pts0)

    def vert_offset(a):
        u = a - n0 * a.dot(n0)
        if u.length < 1e-6:
            return 9
        u.normalize()
        w = n0.cross(u)
        return min(abs(math.atan2((p - c0).dot(w), (p - c0).dot(u))) for p in pts0)
    return min(cands, key=vert_offset)


# ---------------------------------------------------------------------------
# "Round" shape: corner rebuilt from the tangents of the surrounding faces
# ---------------------------------------------------------------------------

def _rail_neighbour(profs, k, idx, rails_cyclic):
    K = len(profs)
    for kk in (k + 1, k - 1):
        if rails_cyclic:
            kk %= K
        if 0 <= kk < K and kk != k:
            return profs[kk][idx]
    return None


def _surface_tangent(v, other_end, neighbour, grid_faces, plane_n=None):
    """Direction leaving v along the face next to the bevel. It stays in the plane of the profile
    (plane_n) when there is one, so that the corner is in that plane even when the bevel's sides
    are slanted; otherwise it is perpendicular to the bevel's side."""
    if neighbour is None:
        return None
    rail = neighbour.co - v.co
    if rail.length < 1e-12:
        return None
    rail.normalize()
    e = edge_between(v, neighbour)
    faces = [f for f in (e.link_faces if e else v.link_faces) if f not in grid_faces]
    if not faces:
        faces = [f for f in v.link_faces if f not in grid_faces]
    if not faces:
        return None
    n = faces[0].normal
    t = n.cross(plane_n if plane_n is not None else rail)
    if t.length < 1e-6 and plane_n is not None:
        t = n.cross(rail)
    if t.length < 1e-9:
        return None
    t.normalize()
    if t.dot(other_end.co - v.co) < 0:
        t = -t
    return t


def _profile_plane(p, rail):
    """Normal of the plane the profile lies in. A straight profile (chamfer) has no plane of its
    own: take the one through its ends that is closest to perpendicular to the rails."""
    d = p[-1].co - p[0].co
    if d.length < 1e-12:
        return None
    d = d.normalized()
    if len(p) >= 3:
        c = sum((v.co for v in p), mathutils.Vector()) / len(p)
        n = mathutils.Vector()
        for i in range(len(p)):
            # Newell-like sum over the profile closed by its chord
            a, b = p[i].co - c, p[(i + 1) % len(p)].co - c
            n += a.cross(b)
        span = max((v.co - c).length for v in p)
        if n.length > 1e-6 * span * span:
            n.normalize()
            if all(abs((v.co - c).dot(n)) <= 1e-3 * span for v in p):
                return n
    if rail is None or rail.length < 1e-12:
        return None
    n = rail - d * rail.dot(d)
    return n.normalized() if n.length > 1e-9 else None


def bevel_corner(profs, k, grid_faces, rails_cyclic):
    p = profs[k]
    v1, v2 = p[0], p[-1]
    n1 = _rail_neighbour(profs, k, 0, rails_cyclic)
    n2 = _rail_neighbour(profs, k, -1, rails_cyclic)
    rail = None
    if n1 is not None and n2 is not None:
        rail = (n1.co - v1.co).normalized() + (n2.co - v2.co).normalized()
    plane_n = _profile_plane(p, rail)
    t1 = _surface_tangent(v1, v2, n1, grid_faces, plane_n)
    t2 = _surface_tangent(v2, v1, n2, grid_faces, plane_n)
    if t1 is None or t2 is None:
        return None
    r = mathutils.geometry.intersect_line_line(v1.co, v1.co + t1, v2.co, v2.co + t2)
    if not r:
        return None
    c = (r[0] + r[1]) / 2
    # the corner has to be in front of both ends
    if (c - v1.co).dot(t1) <= 0 or (c - v2.co).dot(t2) <= 0:
        return None
    return c


def _equal_arc(dense, count):
    cum = [0.0]
    for i in range(1, len(dense)):
        cum.append(cum[-1] + (dense[i] - dense[i - 1]).length)
    total, out, j = cum[-1], [], 0
    for k in range(count + 1):
        target = total * k / count
        while j < len(cum) - 2 and cum[j + 1] < target:
            j += 1
        seg = cum[j + 1] - cum[j]
        t = 0 if seg < 1e-12 else (target - cum[j]) / seg
        out.append(dense[j].lerp(dense[j + 1], t))
    out[0], out[-1] = dense[0].copy(), dense[-1].copy()
    return out


def profile_exponent(shape):
    """Super-ellipse exponent of a Round profile, as Blender's Bevel: 0.25 = straight chamfer,
    0.5 = circle, 1 = square (the sharp corner), 0 = square the other way (concave)."""
    shape = min(max(shape, 0.0), 1.0)
    if shape > 0.999:
        return 400.0
    if shape < 0.001:
        return 1.0 / 400.0
    return -math.log(2.0) / math.log(math.sqrt(shape))


def round_profile(a, b, c, shape, count, steps=256):
    """Quarter super-ellipse from a to b bulging towards corner c (profile_exponent)."""
    o = a + b - c
    ex = profile_exponent(shape)
    dense = []
    for i in range(1, steps):
        th = (math.pi / 2) * i / steps
        x = max(math.cos(th), 0.0) ** (2.0 / ex)
        y = max(math.sin(th), 0.0) ** (2.0 / ex)
        dense.append(o + (a - o) * x + (b - o) * y)
    # exact ends (cos 90° isn't 0 in floats, and a square shape's exponent lifts it to ~0.8)
    return _equal_arc([a.copy()] + dense + [b.copy()], count)


def custom_profile(a, b, c, profile, count, steps=256):
    o = a + b - c
    dense = []
    for i in range(steps + 1):
        x, y = profile.evaluate(i / steps)
        dense.append(o + (a - o) * x + (b - o) * y)
    dense[0], dense[-1] = a.copy(), b.copy()
    return _equal_arc(dense, count)


# ---------------------------------------------------------------------------
# Parametrisation (for UVs and attribute transfer)
# ---------------------------------------------------------------------------

def polyline_param(pts, closed, p):
    """Parameter (in original segment units) of the point of the polyline closest to p."""
    n = len(pts)
    segs = n if closed else n - 1
    best, best_d = 0.0, float("inf")
    for i in range(segs):
        a, b = pts[i], pts[(i + 1) % n]
        ab = b - a
        L2 = ab.length_squared
        t = 0.0 if L2 < 1e-18 else max(0.0, min(1.0, (p - a).dot(ab) / L2))
        d = (a + ab * t - p).length_squared
        if d < best_d:
            best_d, best = d, i + t
    return best


# ---------------------------------------------------------------------------
# Fallback: round part with details that must stay in place (slotted screw head...)
# ---------------------------------------------------------------------------

def order_preserving_match(cols, targets):
    """Match sorted column angles to sorted target angles (keeping order), min total movement.
    Returns list of (col_index, target_index)."""
    a, b = len(cols), len(targets)
    INF = float("inf")
    # dp[i][j] = best cost using first i cols and first j targets, with min(a,b) matches overall
    need = min(a, b)
    dp = [[INF] * (b + 1) for _ in range(a + 1)]
    cnt = [[0] * (b + 1) for _ in range(a + 1)]
    for i in range(a + 1):
        for j in range(b + 1):
            if i == 0 or j == 0:
                dp[i][j], cnt[i][j] = 0.0, 0
                continue
            best = (dp[i - 1][j - 1] + abs(cols[i - 1] - targets[j - 1]), cnt[i - 1][j - 1] + 1, "m")
            for cand in ((dp[i - 1][j], cnt[i - 1][j], "c"), (dp[i][j - 1], cnt[i][j - 1], "t")):
                if (-cand[1], cand[0]) < (-best[1], best[0]):
                    best = cand
            dp[i][j], cnt[i][j] = best[0], best[1]
            cnt[i][j] = best[1]
            dp[i][j] = best[0]
    # backtrack
    pairs, i, j = [], a, b
    while i > 0 and j > 0:
        m = dp[i - 1][j - 1] + abs(cols[i - 1] - targets[j - 1])
        if cnt[i][j] == cnt[i - 1][j - 1] + 1 and abs(dp[i][j] - m) < 1e-12:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif cnt[i][j] == cnt[i - 1][j] and abs(dp[i][j] - dp[i - 1][j]) < 1e-12:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def profile_reduce(bm, faces, n, prefer_axes):
    ref_face = max(faces, key=lambda f: len(f.verts))
    m = len(ref_face.verts)
    if m < 5:
        return "No round face found in the selection.", None
    if n < 3:
        return "Need at least 3 sides.", None
    axis = ref_face.normal.copy()
    c = sum((v.co for v in ref_face.verts), Vector()) / m
    c_ref = c.copy()

    def radial(co):
        d = co - c
        return d - axis * d.dot(axis)

    R = sum(radial(v.co).length for v in ref_face.verts) / m
    tol = R * 0.01
    verts = {v for f in faces for v in f.verts}
    circle = {v for v in verts if abs(radial(v.co).length - R) < tol}
    if len([v for v in ref_face.verts if v in circle]) < m * 0.6:
        return "The largest face isn't round enough to be treated as a cylinder.", None

    # Frame: pick the local axis perpendicular to the cylinder where the original is vertex-aligned
    cands = sorted(prefer_axes, key=lambda a: abs(a.dot(axis)))
    cands = [a for a in cands if abs(a.dot(axis)) < abs(cands[0].dot(axis)) + 1e-3]
    step_old = 2 * math.pi / m

    def frame(a):
        u = a - axis * a.dot(axis)
        u.normalize()
        return u, axis.cross(u)

    def ang_in(u, w, co):
        d = radial(co)
        return math.atan2(d.dot(w), d.dot(u))

    best = None
    for a in cands:
        u, w = frame(a)
        delta = min((ang_in(u, w, v.co) for v in ref_face.verts), key=abs)
        if best is None or abs(delta) < abs(best[2]) - 1e-6:
            best = (u, w, delta)
    u, w, delta = best

    def is_axial(e):
        d = (e.verts[1].co - e.verts[0].co)
        return d.length > 0 and abs(d.normalized().dot(axis)) > 0.999

    # Columns: circle verts linked by edges parallel to the axis
    col_of, columns = {}, []
    for v in ordered(circle):
        if v in col_of:
            continue
        comp, stack = [], [v]
        col_of[v] = len(columns)
        while stack:
            x = stack.pop()
            comp.append(x)
            for e in x.link_edges:
                o = e.other_vert(x)
                if o in circle and o not in col_of and is_axial(e):
                    col_of[o] = len(columns)
                    stack.append(o)
        columns.append(comp)

    def angdiff(a, b):
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

    col_ang = []
    for comp in columns:
        sx = sum(math.cos(ang_in(u, w, v.co)) for v in comp)
        sy = sum(math.sin(ang_in(u, w, v.co)) for v in comp)
        col_ang.append(math.atan2(sy, sx))

    rim_cols = [ci for ci, comp in enumerate(columns) if any(v in ref_face.verts for v in comp)]
    if rim_cols and sum(1 for ci in rim_cols if len(columns[ci]) >= 2) < 0.5 * len(rim_cols):
        return "This round part isn't a straight extrusion (domed or chamfered rim): it's left as it is.", None

    pinned, spokes = set(), set()
    for ci, comp in enumerate(columns):
        for v in comp:
            for e in v.link_edges:
                o = e.other_vert(v)
                if is_axial(e) and o in circle:
                    continue
                if o in circle and angdiff(ang_in(u, w, o.co), ang_in(u, w, v.co)) < 3 * step_old:
                    continue  # neighbour on the arc
                if radial(o.co).length < tol and abs(((o.co - v.co).normalized()).dot(axis)) < 1e-3:
                    spokes.add(e)  # cap fan spoke towards the center
                    continue
                pinned.add(ci)
                break

    if len(pinned) > max(8, 0.25 * len(columns)):
        return "Too many details around this round part: it's left as it is.", None

    # Target angles of the new regular polygon (same symmetry as the original)
    step_new = 2 * math.pi / n
    if abs(delta) <= step_old / 4:
        base = delta
    else:
        base = delta - math.copysign(step_old / 2, delta) + step_new / 2
    targets = [((base + k * step_new + math.pi) % (2 * math.pi)) - math.pi for k in range(n)]

    free = [ci for ci in range(len(columns)) if ci not in pinned]
    keep_moves = {}
    new_angles = []  # targets with no existing column: new columns get inserted there
    if pinned:
        pins = sorted(col_ang[ci] for ci in pinned)
        # unwrap everything relative to the first pin, cut the circle at the pins
        p0 = pins[0]
        unwrap = lambda a: (a - p0) % (2 * math.pi)
        bounds = sorted(unwrap(p) for p in pins) + [2 * math.pi]
        for k in range(len(bounds) - 1):
            lo, hi = bounds[k], bounds[k + 1]
            cs = sorted((unwrap(col_ang[ci]), ci) for ci in free if lo < unwrap(col_ang[ci]) < hi)
            ts = sorted(unwrap(t) for t in targets if lo < unwrap(t) < hi)
            for i, j in order_preserving_match([a for a, _ in cs], ts):
                keep_moves[cs[i][1]] = ts[j] + p0
            matched = {j for _, j in order_preserving_match([a for a, _ in cs], ts)}
            new_angles.extend(ts[j] + p0 for j in range(len(ts)) if j not in matched)
    else:
        cs = sorted((col_ang[ci], ci) for ci in free)
        ts = sorted(targets)
        # rotate target list to best match (circular)
        best_shift, best_cost = 0, None
        for sft in range(n):
            cost = sum(min(angdiff(col_ang[ci], ts[(i * n // len(cs) + sft) % n]), 9) for i, (_, ci) in enumerate(cs[:1]))
            if best_cost is None or cost < best_cost:
                best_shift, best_cost = sft, cost
        ts = ts[best_shift:] + [t + 2 * math.pi for t in ts[:best_shift]]
        a0 = cs[0][0]
        cs_u = [((a - a0) % (2 * math.pi), ci) for a, ci in cs]
        ts_u = sorted((t - a0) % (2 * math.pi) for t in ts)
        pairs = order_preserving_match([a for a, _ in cs_u], ts_u)
        for i, j in pairs:
            keep_moves[cs_u[i][1]] = ts_u[j] + a0
        matched = {j for _, j in pairs}
        new_angles.extend(ts_u[j] + a0 for j in range(len(ts_u)) if j not in matched)

    removed_cols = [ci for ci in free if ci not in keep_moves]
    rem_verts = [v for ci in removed_cols for v in columns[ci]]
    rem_set = set(rem_verts)
    cross = set()
    for v in rem_verts:
        for e in v.link_edges:
            if (is_axial(e) and e.other_vert(v) in rem_set) or e in spokes:
                cross.add(e)
    region = set(verts)
    return None, (cross, rem_verts, keep_moves, columns, (c_ref, axis, u, w, radial),
                  new_angles, region, R)


def apply_profile(bm, data):
    cross, rem_verts, keep_moves, columns, (c, axis, u, w, radial), new_angles, region, R = data
    # move first (positions are independent of the dissolve)
    for ci, t in keep_moves.items():
        for v in columns[ci]:
            d = v.co - c
            h = d.dot(axis)
            r = radial(v.co).length
            v.co = c + axis * h + r * (math.cos(t) * u + math.sin(t) * w)
    if cross:
        bmesh.ops.dissolve_edges(bm, edges=ordered(cross), use_verts=False, use_face_split=False)
    rem = [v for v in rem_verts if v.is_valid]
    if rem:
        bmesh.ops.dissolve_verts(bm, verts=rem, use_face_split=False, use_boundary_tear=False)

    # Insert new columns (more sides than before): split every arc edge crossing the
    # target angle, then join the new vertices that share a face.
    if not new_angles:
        return
    region = {v for v in region if v.is_valid}
    tol = R * 0.01

    def ang(co):
        d = radial(co)
        return math.atan2(d.dot(w), d.dot(u))

    def adiff(a, b):
        return abs((a - b + math.pi) % TAU - math.pi)

    for t in new_angles:
        t = ((t + math.pi) % TAU) - math.pi
        new_vs = []
        edges = {e for v in region for e in v.link_edges
                 if e.verts[0] in region and e.verts[1] in region}
        for e in ordered(edges):
            a, b = e.verts
            if abs(radial(a.co).length - R) > tol or abs(radial(b.co).length - R) > tol:
                continue
            aa, ab = ang(a.co), ang(b.co)
            span = adiff(aa, ab)
            if span < 1e-6 or span > math.radians(60):
                continue  # axial edge, or a chord across the part (slot...)
            da = adiff(aa, t)
            if da < 1e-6 or adiff(t, ab) < 1e-6 or abs(da + adiff(t, ab) - span) > 1e-6:
                continue
            fac = da / span
            h = (a.co - c).dot(axis) * (1 - fac) + (b.co - c).dot(axis) * fac
            r = radial(a.co).length * (1 - fac) + radial(b.co).length * fac
            ne, nv = bmesh.utils.edge_split(e, a, fac)
            nv.co = c + axis * h + r * (math.cos(t) * u + math.sin(t) * w)
            new_vs.append(nv)
            region.add(nv)
        if len(new_vs) > 1:
            bmesh.ops.connect_verts(bm, verts=new_vs)


def _projected_circle(pts, axis):
    """Circle fit of the points seen along `axis`: (center, radius, rms residual, u, w)."""
    u = axis.orthogonal().normalized()
    w = axis.cross(u)
    P = np.array([(p.dot(u), p.dot(w)) for p in pts])
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(axis=1)
    (cx, cy, k), *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = k + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = float(np.sqrt(((np.linalg.norm(P - [cx, cy], axis=1) - r) ** 2).mean()))
    mean_h = sum(p.dot(axis) for p in pts) / len(pts)
    return u * cx + w * cy + axis * mean_h, r, res, u, w


def hole_axis(pts):
    """Direction along which the loop looks like a perfect circle (the drilling axis)."""
    c = sum(pts, Vector()) / len(pts)
    axis = newell(pts, c) or Vector((0, 0, 1))

    def cost(a):
        fit = _projected_circle(pts, a.normalized())
        return fit[2] / fit[1] if fit else 1e9
    best, best_c = axis.copy(), cost(axis)
    step = math.radians(10)
    while step > math.radians(0.02):
        improved = False
        u = best.orthogonal().normalized()
        w = best.cross(u)
        for d in (u, -u, w, -w):
            cand = (best + d * math.tan(step)).normalized()
            cc = cost(cand)
            if cc < best_c - 1e-12:
                best, best_c, improved = cand, cc, True
        if not improved:
            step /= 2
    return best, best_c


def hole_frame(pts):
    """Cylinder the outline lies on: (axis, centre, radius, u, w) or None."""
    axis, rel_err = hole_axis(pts)
    if rel_err > 0.01:
        return None
    c, r, _, _, _ = _projected_circle(pts, axis)
    ref = choose_ref_axis([[p - axis * p.dot(axis) for p in pts]])
    u = (ref - axis * ref.dot(axis)) if ref is not None else axis.orthogonal()
    if u.length < 1e-6:
        u = axis.orthogonal()
    u.normalize()
    w = axis.cross(u)
    return axis, c, r, u, w


def hole_points(frame, pts, count, base=None):
    """Regular points around the frame's cylinder, at the depth of the outline `pts`.
    The list starts near pts[0] and turns the same way as pts. Returns (points, base)."""
    axis, c, r, u, w = frame
    cen = sum(pts, Vector()) / len(pts)
    ang = [math.atan2((p - c).dot(w), (p - c).dot(u)) for p in pts]
    hgt = [(p - c).dot(axis) for p in pts]
    m = len(pts)
    # turning direction of the outline around the axis
    turn = sum(((ang[(i + 1) % m] - ang[i] + math.pi) % TAU) - math.pi for i in range(m))
    sgn = 1 if turn >= 0 else -1
    step_old, step_new = TAU / m, TAU / count
    if base is None:
        delta = min(ang, key=abs)
        base = delta if abs(delta) <= step_old / 4 else delta - math.copysign(step_old / 2, delta) + step_new / 2
    k0 = int(round((ang[0] - base) / step_new))
    targets = [base + step_new * (k0 + sgn * i) for i in range(count)]
    order = sorted(range(m), key=lambda i: ang[i] % TAU)
    A = [ang[i] % TAU for i in order]
    Hh = [hgt[i] for i in order]
    out = []
    for t in targets:
        tt = t % TAU
        j = 0
        while j < m and A[j] < tt:
            j += 1
        a0, h0 = (A[j - 1], Hh[j - 1]) if j > 0 else (A[-1] - TAU, Hh[-1])
        a1, h1 = (A[j], Hh[j]) if j < m else (A[0] + TAU, Hh[0])
        f = 0.0 if a1 - a0 < 1e-12 else (tt - a0) / (a1 - a0)
        h = h0 + (h1 - h0) * f
        out.append(c + axis * h + r * (math.cos(t) * u + math.sin(t) * w))
    return out, base


# ---------------------------------------------------------------------------
# Width in the other modes: what a shape is scaled around
# ---------------------------------------------------------------------------

def _onward(v, nxt, inside):
    """Vertex joined to v by an edge (outside `inside`) that carries the curve on past v, in the
    direction it leaves v (v - nxt): the start of the straight part the curve is tangent to."""
    d0 = v.co - nxt.co
    if d0.length < 1e-12:
        return None
    d0.normalize()
    best = None
    for e in v.link_edges:
        o = e.other_vert(v)
        if o in inside:
            continue
        d = o.co - v.co
        if d.length < 1e-12:
            continue
        k = d.normalized().dot(d0)
        if k > 0.5 and (best is None or k > best[0]):
            best = (k, o)
    return best[1] if best else None


def chain_end_corner(chain, inside):
    """Corner of an open curve (vertices `chain`) between two straight parts: where the edges
    going on past its two ends (not into `inside`) meet. None if there is no such corner."""
    if len(chain) < 2:
        return None
    pa = _onward(chain[0], chain[1], inside)
    pb = _onward(chain[-1], chain[-2], inside)
    if pa is None or pb is None:
        return None
    a, b = chain[0].co, chain[-1].co
    ta, tb = (a - pa.co).normalized(), (b - pb.co).normalized()
    if abs(ta.dot(tb)) > 0.9999:
        return None
    r = mathutils.geometry.intersect_line_line(a, a + ta, b, b + tb)
    if not r or (r[0] - r[1]).length > 1e-3 * max((a - b).length, 1e-12):
        return None
    c = (r[0] + r[1]) / 2
    if (c - a).dot(ta) <= 0 or (c - b).dot(tb) <= 0:
        return None
    return c


def radial_scaler(frame, width, surface=None):
    """Point -> point scaled `width` times away from the frame's axis (a hole's radius). With a
    BVH `surface`, the point then slides along the axis onto it (the faces around the hole), if
    it meets them near its height."""
    axis, c, r, _, _ = frame
    reach = r * max(width, 1.0)

    def f(p):
        d = p - c
        h = d.dot(axis)
        q = c + axis * h + (d - axis * h) * width
        if surface is not None:
            best = None
            eps = 1e-4 * r
            for s in (axis, -axis):
                hit = surface.ray_cast(q - s * eps, s, reach)
                if hit[0] is not None and (best is None or (hit[0] - q).length < (best - q).length):
                    best = hit[0]
            if best is not None:
                q = best
        return q
    return f


def _face_components(faces):
    fset = set(faces)
    seen, comps = set(), []
    for f in faces:
        if f in seen:
            continue
        comp, stack = [], [f]
        seen.add(f)
        while stack:
            x = stack.pop()
            comp.append(x)
            for e in x.edges:
                for g in e.link_faces:
                    if g in fset and g not in seen:
                        seen.add(g)
                        stack.append(g)
        comps.append(comp)
    return comps
