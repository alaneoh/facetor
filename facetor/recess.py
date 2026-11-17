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

"""Parts of revolution with a convex recess (slot, hexagon): rebuilt around it."""

import math
import bmesh
import numpy as np
from mathutils import Vector

from .core import TAU, edge_between, keep_open, ordered
from .revolution import profile_runs, revolution_cut_count, revolution_cut_rings
from .safety import _restore, guarded


# ---------------------------------------------------------------------------
# Part of revolution cut by a convex recess (slot...): rebuilt as a regular grid of rings and
# radial columns, clipped by the recess walls. The edge flow of a turned part is kept right up
# to the recess, the walls are re-cut where the new edges meet them.
# ---------------------------------------------------------------------------

def _circle2d(pts):
    P = np.array([(p.x, p.y) for p in pts], dtype=float)
    if len(P) < 3:
        return None
    A = np.c_[P, np.ones(len(P))]
    b = -(P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    cx, cy = -D / 2, -E / 2
    r2 = cx * cx + cy * cy - F
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    if np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r).max() > 0.02 * r:
        return None
    return cx, cy, r


class _Curve:
    """Open polyline in the (r, h) plane, evaluated by arc length."""

    def __init__(self, pts):
        self.p = [Vector((q.x, q.y)) for q in pts]
        self.s = [0.0]
        for q0, q1 in zip(self.p, self.p[1:]):
            self.s.append(self.s[-1] + (q1 - q0).length)

    def at(self, s):
        p, S = self.p, self.s
        if s <= 0:
            return p[0].copy()
        if s >= S[-1]:
            return p[-1].copy()
        lo, hi = 0, len(S) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if S[mid] <= s:
                lo = mid
            else:
                hi = mid
        t = (s - S[lo]) / max(S[hi] - S[lo], 1e-30)
        return p[lo].lerp(p[hi], t)

    def project(self, q):
        """(distance, s, segment index) of the nearest point."""
        key = (q[0], q[1])
        memo = self.__dict__.setdefault("_memo", {})
        if key in memo:
            return memo[key]
        memo[key] = res = self._project(q)
        return res

    def _project(self, q):
        best = None
        for i, (p0, p1) in enumerate(zip(self.p, self.p[1:])):
            d = p1 - p0
            L2 = d.length_squared
            t = 0.0 if L2 < 1e-30 else max(0.0, min(1.0, (q - p0).dot(d) / L2))
            dist = (p0 + d * t - q).length
            if best is None or dist < best[0]:
                best = (dist, self.s[i] + t * math.sqrt(L2), i)
        return best

    def normal(self, i):
        d = (self.p[i + 1] - self.p[i]).normalized()
        return Vector((d.y, -d.x))


_CLIP_CACHE = None   # dict while one change is computed (same mesh, same selection), else None


class _clip_cache:
    """The analysis of a part with a recess is the same for the count and the rebuild:
    do it once per change instead of four times."""
    def __enter__(self):
        global _CLIP_CACHE
        self.prev = _CLIP_CACHE
        if _CLIP_CACHE is None:
            _CLIP_CACHE = {}

    def __exit__(self, *a):
        global _CLIP_CACHE
        _CLIP_CACHE = self.prev


def _with_clip_cache(fn):
    def wrapped(*a, **k):
        with _clip_cache():
            return fn(*a, **k)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


def _profile_order(rings, row_ids, adj):
    """Rings ordered along the profile by position (shortest nearest-neighbour path in the
    radius/height plane), when faces join non-consecutive rings. None unless every face joins
    rings at most two places apart in that order."""
    if any(not adj[i] for i in row_ids):
        return None
    pt = {i: Vector((rings[i]["r"], rings[i]["h"])) for i in row_ids}
    best = None
    for start in row_ids:
        order, left, length = [start], set(row_ids) - {start}, 0.0
        while left:
            j = min(left, key=lambda k: (pt[k] - pt[order[-1]]).length)
            length += (pt[j] - pt[order[-1]]).length
            order.append(j)
            left.discard(j)
        if best is None or length < best[0] - 1e-12:
            best = (length, order)
    order = best[1]
    pos = {i: k for k, i in enumerate(order)}
    if any(abs(pos[i] - pos[j]) > 2 for i in row_ids for j in adj[i]):
        return None
    if any(not (adj[order[k]] & set(order[k + 1:k + 3])) for k in range(len(order) - 1)):
        return None
    return order


def _pocket_outline(comp, surface, axis):
    """Inside test of a pocket that isn't convex (a cross recess): its opening (the loop of wall
    edges along the turned surface) seen along the direction the walls run in (their common
    direction, else the axis). Returns (point -> bool, None) or (None, error)."""
    sset = set(surface)
    bedges = [e for f in comp for e in f.edges if any(g in sset for g in e.link_faces)]
    bedges = list(dict.fromkeys(bedges))
    adj = {}
    for e in bedges:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    if not adj or any(len(l) != 2 for l in adj.values()):
        return None, "This recess isn't a simple pocket."
    v0 = min(adj, key=lambda v: tuple(v.co))
    loop, prev, cur = [v0], None, v0
    while True:
        e = adj[cur][0] if adj[cur][0] is not prev else adj[cur][1]
        cur = e.other_vert(cur)
        prev = e
        if cur is v0:
            break
        loop.append(cur)
        if len(loop) > len(adj):
            return None, "This recess isn't a simple pocket."
    if len(loop) != len(adj):
        return None, "This recess isn't a simple pocket."
    N = np.array([tuple(f.normal) for f in comp])
    _, sv, vt = np.linalg.svd(N, full_matrices=True)
    d = Vector(tuple(vt[2])).normalized()
    if len(sv) < 3 or sv[2] > 0.1 * sv[0]:
        d = axis.normalized()
    u2 = d.orthogonal().normalized()
    w2 = d.cross(u2)
    poly = [(v.co.dot(u2), v.co.dot(w2)) for v in loop]
    n = len(poly)

    def inside(p):
        x, y = p.dot(u2), p.dot(w2)
        hit = False
        for i in range(n):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                hit = not hit
        return hit
    return inside, None


def revolution_clip_setup(bm, faces, profile_segments=None):
    if _CLIP_CACHE is None:
        return _revolution_clip_setup(bm, faces, profile_segments)
    key = (id(bm), frozenset(faces), profile_segments)
    if key not in _CLIP_CACHE:
        _CLIP_CACHE[key] = _revolution_clip_setup(bm, faces, profile_segments)
    return _CLIP_CACHE[key]


def _revolution_clip_setup(bm, faces, profile_segments=None):
    res = _clip_setup(bm, faces, profile_segments, False)
    if res[1] is not None:
        # end walls of a cross recess look like turned faces: tried again without the faces
        # that touch the recess floor
        res2 = _clip_setup(bm, faces, profile_segments, True)
        if res2[1] is None:
            return res2
    return res


def _clip_setup(bm, faces, profile_segments, floor):
    data, err = revolution_cut_rings(bm, faces, loose=True)
    if err:
        return None, err
    c, a, u, w = data["c"], data["a"], data["u"], data["w"]
    rings, ring_of, ang = data["rings"], data["ring_of"], data["ang"]
    fset = set(faces)

    def rh(p):
        d = p - c
        h = d.dot(a)
        return Vector(((d - a * h).length, h))
    # faces that may belong to the turned surface: normal in the plane of the axis, narrow in angle
    cand = set()
    for f in faces:
        if len({ring_of[v] for v in f.verts}) == 1:
            continue
        d = f.calc_center_median() - c
        rad = d - a * d.dot(a)
        if rad.length < 1e-12:
            continue
        tang = a.cross(rad).normalized()
        if abs(f.normal.dot(tang)) > 0.2:
            continue
        ts = sorted(ang(v) % TAU for v in f.verts)
        gaps = [(ts[(i + 1) % len(ts)] - ts[i]) % TAU for i in range(len(ts))]
        if TAU - max(gaps) > math.radians(100):
            continue
        cand.add(f)
    # walls facing away from the axis (the end of an arm of a cross recess) look like turned
    # faces: they touch the floor of the recess (a face across the axis, facing along it)
    # (on no ring that goes all the way round: a flat top around a slot has its rim on one)
    big_ring = max(len(r["verts"]) for r in rings)
    floor_verts = set()
    for f in faces:
        if abs(f.normal.dot(a)) < 0.9 or len({ring_of[v] for v in f.verts}) < 2:
            continue
        if all(len(rings[ring_of[v]]["verts"]) < 0.5 * big_ring for v in f.verts):
            floor_verts.update(f.verts)
    if floor and floor_verts:
        cand = {f for f in cand if not any(v in floor_verts for v in f.verts)}

    def full_cap(f):
        if len({ring_of[x] for x in f.verts}) != 1:
            return False
        ts = sorted(ang(v) % TAU for v in f.verts)
        return max((ts[(i + 1) % len(ts)] - ts[i]) % TAU for i in range(len(ts))) < math.radians(100)
    wall_verts = {v for f in faces if f not in cand and not full_cap(f) for v in f.verts}
    # rows: rings that still have vertices away from the recess
    row_ids = [i for i, r in enumerate(rings) if len(r["verts"]) >= 3 and
               any(v not in wall_verts for v in r["verts"])]
    if len(row_ids) < 2:
        return None, "No faces turn around an axis here."
    row_set = set(row_ids)
    adj = {i: set() for i in row_ids}
    for f in cand:
        ids = {ring_of[v] for v in f.verts} & row_set
        if len(ids) != 2:
            continue
        for i in ids:
            for j in ids:
                if i != j:
                    adj[i].add(j)
    if any(len(n) > 2 for n in adj.values()):
        # coarse CAD triangulation (dome made of long triangles): some faces skip a ring.
        # Order the rings along the profile instead, if faces skip at most one ring.
        order = _profile_order(rings, row_ids, adj)
        if order is None:
            return None, "The rings don't follow each other like a profile."
    else:
        ends = [i for i, n in adj.items() if len(n) <= 1]
        if not ends:
            return None, "Closed profile (torus): not supported."
        order, prev = [ends[0]], None
        while True:
            nxt = [j for j in adj[order[-1]] if j != prev and j not in order]
            if not nxt:
                break
            prev = order[-1]
            order.append(nxt[0])
        if len(order) != len(row_ids):
            return None, "The selection holds several separate parts."
    rmax = max(rings[i]["r"] for i in order)
    prof = [Vector((rings[i]["r"], rings[i]["h"])) for i in order]
    # points of the surface past an end row (top of a dome under a slot): the profile goes on
    ext = {0: [], 1: []}
    if len(prof) >= 2:
        for v in {v for f in cand for v in f.verts if v in wall_verts}:
            q = rh(v.co)
            for side, (pe, pp) in enumerate(((prof[0], prof[1]), (prof[-1], prof[-2]))):
                d = pe - pp
                if d.length > 1e-12 and (q - pe).dot(d) > 1e-6 * rmax:
                    ext[side].append(q)
    if ext[0] and ext[1]:
        return None, "The recess sits in the middle of the profile."
    if ext[0]:
        order.reverse()
        prof.reverse()
        ext[1] = ext[0]
    leaf_end = bool(ext[1])
    if leaf_end and not any(q.x < prof[-1].x - 1e-6 * rmax for q in ext[1]):
        return None, "The recess sits in the middle of the profile."
    if any(rings[i]["r"] < 1e-9 for i in order):
        return None, "Pole on the axis."
    dense = list(prof)
    pole = None
    pole_run = None  # (index of the first row of the rounded part that ends on the axis, circle)
    if leaf_end:
        pts_ = [q for q in sorted(ext[1], key=lambda q: -q.x) if q.x < prof[-1].x]
        runs_real = profile_runs([Vector((q.x, q.y, 0.0)) for q in prof])
        start = next((lo for lo, hi in runs_real if hi == len(prof) - 1), len(prof) - 1)
        fit_pts = prof[start:] if len(prof) - start >= 3 else prof[start:] + pts_
        circ = _circle2d(fit_pts) if len(fit_pts) >= 3 else None
        # the rounded part goes back as far as the rows stay on the same arc
        while circ and start > 0:
            cx, cy, r = circ
            q = prof[start - 1]
            if abs(math.hypot(q.x - cx, q.y - cy) - r) > 0.004 * r:
                break
            start -= 1
            fit_pts = prof[start:] + pts_
            circ = _circle2d(fit_pts) or circ
        hp = None
        if circ:
            cx, cy, r = circ
            if r * r >= cx * cx:
                k = math.sqrt(r * r - cx * cx)
                hp = min((cy + k, cy - k), key=lambda y: abs(y - prof[-1].y))
        if hp is None:
            circ = None
            q0, q1 = (prof[-1], pts_[-1]) if pts_ else (prof[-2], prof[-1])
            if abs(q1.x - q0.x) < 1e-12:
                return None, "Can't find the top of the rounded part."
            hp = q1.y + (q1.y - q0.y) * (0 - q1.x) / (q1.x - q0.x)
        pole = Vector((0.0, hp))
        if circ:
            cx, cy, r = circ
            ctr = Vector((cx, cy))
            a0 = math.atan2(prof[start].y - cy, prof[start].x - cx)
            a1 = math.atan2(hp - cy, 0.0 - cx)
            da = (a1 - a0 + math.pi) % TAU - math.pi
            arc = [ctr + r * Vector((math.cos(a0 + da * i / 96), math.sin(a0 + da * i / 96))) for i in range(97)]
            arc[0], arc[-1] = prof[start].copy(), pole.copy()
            dense = list(prof[:start]) + arc
            pole_run = (start, (ctr, r, a0, da))
        else:
            dense += [q for q in pts_ if (q - dense[-1]).length > 1e-4 * rmax]
            dense.append(pole)
            pole_run = (start, None)
    curve = _Curve(dense)
    tol = 0.03 * rmax

    def surf_normal(p):
        q = rh(p)
        dist, s, i = curve.project(q)
        n2 = curve.normal(i)
        d = p - c
        rad = d - a * d.dot(a)
        rad = rad.normalized() if rad.length > 1e-12 else u
        return dist, n2.x * rad + n2.y * a

    order_set = set(order)
    order_caps = [f for f in data["caps"] if ring_of[f.verts[0]] in order_set and full_cap(f)]
    surface = set()
    for f in faces:
        if f in order_caps:
            continue
        cen = f.calc_center_median()
        dist, n3 = surf_normal(cen)
        ts = sorted(ang(v) % TAU for v in f.verts)
        span = TAU - max((ts[(i + 1) % len(ts)] - ts[i]) % TAU for i in range(len(ts)))
        if span > math.radians(100):
            continue
        sag = rh(cen).x * (1 - math.cos(min(span, math.pi) / 2))
        ss = [curve.project(rh(v.co))[1] for v in f.verts]
        q0, q1, qm = curve.at(min(ss)), curve.at(max(ss)), curve.at((min(ss) + max(ss)) / 2)
        dq = q1 - q0
        tq = 0.0 if dq.length < 1e-12 else max(0.0, min(1.0, (qm - q0).dot(dq) / dq.length_squared))
        sag += (q0 + dq * tq - qm).length
        if dist > tol + 1.5 * sag or abs(f.normal.dot(n3)) < 0.85:
            continue
        if all(curve.project(rh(v.co))[0] < tol for v in f.verts):
            surface.add(f)
    if not surface:
        return None, "No faces turn around an axis here."
    walls = [f for f in faces if f not in surface and f not in order_caps and
             any(g in surface for e in f.edges for g in e.link_faces)]
    if not walls:
        return None, "Nothing cuts this part."
    wall_set = set(walls)
    cap_set = set(order_caps)
    for f in surface:
        for e in f.edges:
            lf = e.link_faces
            # an open end (free border) is fine, unselected faces are not: they'd be cut off
            if len(lf) > 2 or any(g not in fset for g in lf):
                return None, "The part is joined to unselected geometry."
        for v in f.verts:
            if any(g not in surface and g not in wall_set and g not in cap_set for g in v.link_faces):
                return None, "A recess face touches the turned surface at a single point."
    for f in order_caps:
        if any(g not in surface and g not in cap_set for e in f.edges for g in e.link_faces):
            return None, "An end cap is cut by the recess."
    # walls must be flat; recess = convex: on the void side of every one of its walls
    size = max(rmax, max(abs(rings[i]["h"]) for i in order), 1e-9)
    planes = {}
    for f in walls:
        p0, n = f.calc_center_median(), f.normal.copy()
        if n.length < 0.5 or max(abs((v.co - p0).dot(n)) for v in f.verts) > 1e-3 * size:
            return None, "A wall of the recess isn't flat."
        planes[f] = [p0, n]
    comp_of, comps = {}, []
    for f in walls:
        if f in comp_of:
            continue
        stack, comp = [f], []
        comp_of[f] = len(comps)
        while stack:
            g = stack.pop()
            comp.append(g)
            for v in g.verts:
                for h2 in v.link_faces:
                    if h2 in wall_set and h2 not in comp_of:
                        comp_of[h2] = len(comps)
                        stack.append(h2)
        comps.append(comp)
    eps = 1e-6 * size
    outlines = {}       # non-convex pockets (cross recess): comp index -> 2D test
    for ci, comp in enumerate(comps):
        if len(comp) < 2:
            return None, "This recess isn't a simple pocket."
        convex = True
        for f in comp:
            p0, n = planes[f]
            others = [v.co for g in comp if g is not f for v in g.verts]
            sgn = sum((q - p0).dot(n) for q in others)
            if sgn < 0:
                n = -n
                planes[f][1] = n
            if min((q - p0).dot(n) for q in others) < -1e-3 * size:
                convex = False
        if not convex:
            test, oerr = _pocket_outline(comp, surface, a)
            if oerr:
                return None, oerr
            outlines[ci] = test

    def void(p):
        for ci, comp in enumerate(comps):
            if ci in outlines:
                if outlines[ci](p):
                    return True
            elif all((p - planes[f][0]).dot(planes[f][1]) > eps for f in comp):
                return True
        return False
    # rows of the new grid
    rows = list(prof) + ([pole] if pole is not None else [])
    old_p = None

    def pole_count():
        start, circ = pole_run
        segs = len(prof) - 1 - start
        last = (pole - prof[-1]).length
        if segs <= 0:
            return 1
        mean = sum((prof[i + 1] - prof[i]).length for i in range(start, len(prof) - 1)) / segs
        return segs + max(1, int(round(last / max(mean, 1e-12))))
    if profile_segments is not None:
        if profile_segments < 1:
            return None, "Needs at least 1 segment."
        if pole_run is not None:
            start, circ = pole_run
            old_p = pole_count()
            if circ:
                ctr, r, a0, da = circ
                mid = [ctr + r * Vector((math.cos(a0 + da * i / profile_segments),
                                         math.sin(a0 + da * i / profile_segments)))
                       for i in range(1, profile_segments)]
            else:
                mid = [curve.at(curve.project(prof[start])[1] + (curve.s[-1] - curve.project(prof[start])[1])
                                * i / profile_segments) for i in range(1, profile_segments)]
            rows = list(prof[:start + 1]) + mid + [pole]
        else:
            runs = profile_runs([Vector((q.x, q.y, 0.0)) for q in rows])
            if not runs:
                return None, "This part has no rounded profile to change."
            old_p = max(hi - lo for lo, hi in runs)
            new_rows, last = [], 0
            for lo, hi in runs:
                new_rows += rows[last:lo]
                sub = keep_open([Vector((q.x, q.y, 0.0)) for q in rows[lo:hi + 1]], profile_segments)
                new_rows += [Vector((q.x, q.y)) for q in sub[:-1]]
                last = hi
            new_rows += rows[last:]
            rows = new_rows
    row_s = [curve.project(q)[1] for q in rows]
    if any(s1 <= s0 + 1e-9 * size for s0, s1 in zip(row_s, row_s[1:])):
        return None, "The rings of the new profile overlap."
    return dict(data=data, c=c, a=a, u=u, w=w, order=order, rows=rows, row_s=row_s, curve=curve,
                pole=pole is not None, surface=surface, walls=walls, planes=planes, comps=comps,
                void=void, order_caps=order_caps, size=size, rh=rh, old_p=old_p, nonconvex=bool(outlines),
                pole_count=pole_count() if pole_run is not None else None), None


def revolution_clip_profile_count(bm, faces):
    res, err = revolution_clip_setup(bm, faces)
    if err:
        return None
    if res["pole_count"]:
        return res["pole_count"]
    rows = res["rows"]
    runs = profile_runs([Vector((q.x, q.y, 0.0)) for q in rows])
    return max((hi - lo for lo, hi in runs), default=None)


def revolution_clip_count(bm, faces):
    res, err = revolution_clip_setup(bm, faces)
    if err:
        return None
    rings = res["data"]["rings"]
    wv = {v for f in res["walls"] for v in f.verts}
    sizes = [len(rings[i]["verts"]) for i in res["order"] if not any(v in wv for v in rings[i]["verts"])]
    return max(sizes) if sizes else None


RECESS_KEY = "facetor_recess"


def recess_counts(me, bm, faces):
    """(sides, rings of the rounded part) of a part with a recess, or None. The rings are the
    ones set last time when the sides haven't changed since (rings hidden in the recess can't be
    counted from the mesh)."""
    if revolution_clip_setup(bm, faces)[1] is not None:
        return None
    sides = revolution_clip_count(bm, faces) or revolution_cut_count(bm, faces)
    stored = list(me.get(RECESS_KEY, []))
    if len(stored) == 2 and stored[0] == sides and stored[1] > 0:
        return sides, int(stored[1])
    return sides, revolution_clip_profile_count(bm, faces)


def _wall_groups(S):
    """Walls split in several faces on one plane (CAD fans), joined by edges: lists of 2+ faces."""
    planes, walls = S["planes"], S["walls"]
    tol = 1e-4 * S["size"]
    wall_set, seen, groups = set(walls), set(), []
    for f in walls:
        if f in seen:
            continue
        seen.add(f)
        grp, stack = [f], [f]
        p0, n0 = planes[f]
        while stack:
            g = stack.pop()
            for e in g.edges:
                for h in e.link_faces:
                    if h in wall_set and h not in seen and planes[h][1].dot(n0) > 0.9999                             and abs((planes[h][0] - p0).dot(n0)) < tol:
                        seen.add(h)
                        grp.append(h)
                        stack.append(h)
        if len(grp) > 1:
            groups.append(grp)
    return groups


def revolution_clip_rebuild(bm, faces, count, profile_segments=None):
    """Rebuild a part of revolution with a recess. Flat walls made of several faces are joined
    into one face first (undone if the rebuild fails)."""
    S, err = revolution_clip_setup(bm, faces, profile_segments)
    groups = _wall_groups(S) if not err else []
    if not groups:
        return _revolution_clip_rebuild(bm, faces, count, profile_segments)
    if _CLIP_CACHE is not None:
        _CLIP_CACHE.clear()
    snap = bm.copy()
    keep = [f for f in faces if not any(f in g for g in groups)]
    for grp in groups:
        res = bmesh.ops.dissolve_faces(bm, faces=grp)
        keep += res["region"]
    err = _revolution_clip_rebuild(bm, [f for f in keep if f.is_valid], count, profile_segments)
    if err:
        _restore(bm, snap)
    snap.free()
    return err


def _revolution_clip_rebuild(bm, faces, count, profile_segments=None):
    S, err = revolution_clip_setup(bm, faces, profile_segments)
    if _CLIP_CACHE is not None:
        _CLIP_CACHE.clear()     # the mesh changes from here on
    if err:
        return err
    if count < 8:
        return "Too few sides around this recess (8 at least)."
    if count < 3:
        return "Needs at least 3 sides."
    data = S["data"]
    c, a, u, w = S["c"], S["a"], S["u"], S["w"]
    rows, row_s, curve, void = S["rows"], S["row_s"], S["curve"], S["void"]
    planes, walls, surface = S["planes"], S["walls"], S["surface"]
    size, rh = S["size"], S["rh"]
    plane_tol = 1e-4 * size

    def same_plane(f, g):
        """Two wall faces on the same plane (a CAD wall is often split in several faces)."""
        if f is g:
            return True
        (p0, n0), (p1, n1) = planes[f], planes[g]
        return n0.dot(n1) > 0.9999 and abs((p1 - p0).dot(n0)) < plane_tol

    has_pole = S["pole"]
    ang = data["ang"]
    rings, pins = data["rings"], data["pins"]
    step = TAU / count
    # phase: same symmetry as the ring with the most free vertices
    big = max((rings[i] for i in S["order"]),
              key=lambda r: (sum(1 for v in r["verts"] if v not in pins), r["r"]))
    pts = [v for v in big["verts"] if v not in pins] or big["verts"]
    delta = min((ang(v) for v in pts), key=abs)
    m0 = max(len(big["verts"]), 3)
    base = delta if abs(delta) <= (TAU / m0) / 4 else delta - math.copysign(TAU / m0 / 2, delta) + step / 2
    # a recess with a main direction (slot): columns symmetric to it, so both sides are cut alike
    rv = ordered({v for f in walls for v in f.verts})
    Q = np.array([((v.co - c).dot(u), (v.co - c).dot(w)) for v in rv])
    if len(Q) >= 3 and count % 2 == 0:
        Q = Q - Q.mean(axis=0)
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        if sv[0] > 1.2 * max(sv[1], 1e-12):
            ts = math.atan2(vt[0][1], vt[0][0])
            opts = [ts, ts + step / 2]
            base = min(opts, key=lambda t: abs(((t - base + step / 2) % step) - step / 2))

    def surf(theta, s):
        q = curve.at(s)
        return c + a * q.y + q.x * (math.cos(theta) * u + math.sin(theta) * w)
    nrow = len(rows)
    last_pole = has_pole

    def is_pole(k):
        return last_pole and k == nrow - 1

    def theta(j):
        return base + j * step

    def G(k, j):
        return ("P",) if is_pole(k) else ("G", k, j % count)
    pos = {}
    for k in range(nrow):
        for j in range(count):
            key = G(k, j)
            if key not in pos:
                pos[key] = surf(theta(j), row_s[k])
    state = {key: void(p) for key, p in pos.items()}

    # corners of the recess: boundary vertices shared by two walls
    wall_set = set(walls)
    bedges = [e for e in {e for f in walls for e in f.edges}
              if any(g in surface for g in e.link_faces)]
    bverts = {v for e in bedges for v in e.verts}
    # (faces of one flat wall don't make a corner between them)
    corners = [v for v in bverts if len({min((h for h in v.link_faces if h in wall_set and same_plane(h, g)),
                                             key=lambda h: h.index) for g in v.link_faces if g in wall_set}) >= 2]
    wall_list = list(walls)
    close = 1e-5 * size

    edge_cache = {}

    def edge(kind, k, j):
        """Crossings along a grid edge, in increasing t: list of (t, ref, wall, enter)."""
        key = (kind, k, j % count)
        if key in edge_cache:
            return edge_cache[key]
        if kind == "R":
            def P(t):
                return surf(theta(j) + t * step, row_s[k])
            r0, r1 = G(k, j), G(k, j + 1)
        else:
            def P(t):
                return surf(theta(j), row_s[k] + t * (row_s[k + 1] - row_s[k]))
            r0, r1 = G(k, j), G(k + 1, j)
        n = 24
        ts = [i / n for i in range(n + 1)]
        P_s = [P(t) for t in ts]
        brk = []
        for f in wall_list:
            p0, nn = planes[f]
            vals = [(q - p0).dot(nn) for q in P_s]
            for i in range(n):
                if (vals[i] > 0) != (vals[i + 1] > 0):
                    lo, hi = ts[i], ts[i + 1]
                    vlo = vals[i]
                    for _ in range(50):
                        mid = (lo + hi) / 2
                        vm = (P(mid) - p0).dot(nn)
                        if (vm > 0) == (vlo > 0):
                            lo, vlo = mid, vm
                        else:
                            hi = mid
                    brk.append(((lo + hi) / 2, f))
        brk.sort(key=lambda x: x[0])
        out = []
        prev_state = state[r0]
        cand = brk + [(1.0, None)]
        for idx, (t, f) in enumerate(cand):
            if f is None:
                break
            nxt = cand[idx + 1][0]
            st = void(P((t + nxt) / 2))
            if st != prev_state:
                p = P(t)
                ref = ("X",) + key + (len(out),)
                if t < 1e-4:
                    ref = r0
                elif t > 1 - 1e-4:
                    ref = r1
                else:
                    for v in corners:
                        # (on the line where its two walls meet: the old corner sits on the old
                        # facets, the new point on the smooth surface)
                        dv = v.co - p
                        if dv.length < close or (S["nonconvex"] and (dv - a * dv.dot(a)).length < 20 * close and
                                                 dv.length < 0.02 * size):
                            ref = ("C", v)
                            break
                if ref[0] == "X":
                    pos[ref] = p
                out.append((t, ref, f, st))
                prev_state = st
        edge_cache[key] = out
        return out

    corner_cell = {}
    for v in corners:
        q = rh(v.co)
        s = curve.project(q)[1]
        t = ang(v)
        jj = int(math.floor((t - base) / step + 1e-9))
        kk = max((k for k in range(nrow - 1) if row_s[k] <= s + 1e-9), default=None)
        if kk is None or s > row_s[-1] + 1e-9:
            return "A corner of the recess lies outside the turned surface."
        corner_cell.setdefault((kk, jj % count), []).append(v)

    polys = []
    boundary_refs = {}
    for k in range(nrow - 1):
        for j in range(count):
            # walk the cell: (corner ref, list of crossings along the edge to the next corner)
            segs = [(G(k, j), [(t, r, f, st) for t, r, f, st in edge("R", k, j)])]
            segs.append((G(k, j + 1), edge("C", k, j + 1)))
            if is_pole(k + 1):
                segs.append((G(k + 1, j), [(1 - t, r, f, not st) for t, r, f, st in reversed(edge("C", k, j))]))
            else:
                segs.append((G(k + 1, j + 1), [(1 - t, r, f, not st) for t, r, f, st in reversed(edge("R", k + 1, j))]))
                segs.append((G(k + 1, j), [(1 - t, r, f, not st) for t, r, f, st in reversed(edge("C", k, j))]))
            # fix the 'state after crossing' of reversed edges: recompute from the corner states
            items = []
            for cref, xs in segs:
                items.append(("c", cref, state[cref]))
                cur = state[cref]
                for t, r, f, st in xs:
                    cur = not cur
                    items.append(("x", r, f, cur))
            cpins = corner_cell.get((k, j), [])
            if not any(it[0] == "x" for it in items):
                if all(not it[2] for it in items):
                    if cpins:
                        return "The recess sits inside one face: use more sides."
                    poly = list(dict.fromkeys(it[1] for it in items))
                    if len(poly) >= 3:
                        polys.append(poly)
                elif any(not it[2] for it in items):
                    return "Can't follow the recess here: try another number of sides."
                continue
            # pieces of kept surface: from an exit (void -> kept) to the next enter
            start = next((i for i, it in enumerate(items) if it[0] == "x" and not it[3]), None)
            if start is None:
                return "Can't follow the recess here: try another number of sides."
            items = items[start:] + items[:start]
            pieces, cur = [], None
            for it in items:
                if it[0] == "x" and not it[3]:
                    cur = [it]
                elif it[0] == "x" and it[3]:
                    if cur is None:
                        return "Can't follow the recess here: try another number of sides."
                    cur.append(it)
                    pieces.append(cur)
                    cur = None
                elif cur is not None and not it[2]:
                    cur.append(it)
            used = set()
            for pc in pieces:
                ex, en = pc[0], pc[-1]
                poly = [it[1] for it in pc]
                # corners of the recess between the enter and the exit
                chain, wall_now = [], en[2]
                avail = [v for v in cpins if v not in used]
                guard = 0
                while not same_plane(wall_now, ex[2]) and guard < 8:
                    guard += 1
                    nxt = next((v for v in avail if v not in chain and any(
                        g in wall_set and same_plane(g, wall_now) for g in v.link_faces)), None)
                    if nxt is None:
                        break
                    chain.append(nxt)
                    others = [g for g in nxt.link_faces if g in wall_set and not same_plane(g, wall_now)]
                    if ex[2] in others:
                        wall_now = ex[2]
                    elif others:
                        wall_now = others[0]
                    else:
                        break
                if not same_plane(ex[2], wall_now):
                    return "Can't follow the recess corner here: try another number of sides."
                used |= set(chain)
                poly += [("C", v) for v in chain]
                for it in (ex, en):
                    boundary_refs.setdefault(it[1], set()).add(it[2])
                for v in chain:
                    boundary_refs.setdefault(("C", v), set())
                poly = list(dict.fromkeys(poly))
                if len(poly) >= 3:
                    polys.append(poly)
            if len(used) != len(cpins):
                return "Can't follow the recess corner here: try another number of sides."

    # new outline of each wall: its stretches along the turned surface get the new points
    def chains_of(f):
        loop = list(f.verts)
        n = len(loop)
        isb = [edge_between(loop[i], loop[(i + 1) % n]) in bedges_set for i in range(n)]
        if all(isb):
            return None
        out = []
        i0 = next(i for i in range(n) if not isb[i - 1] and isb[i]) if any(isb) else None
        if i0 is None:
            return []
        i = i0
        seen = 0
        while seen < n:
            if isb[i % n] and not isb[(i - 1) % n]:
                ch = [loop[i % n]]
                k = i
                while isb[k % n]:
                    k += 1
                    ch.append(loop[k % n])
                out.append((i % n, k % n, ch))
                seen += k - i
                i = k
            else:
                i += 1
                seen += 1
        return out
    bedges_set = set(bedges)
    wall_chains = {}
    for f in walls:
        ch = chains_of(f)
        if ch is None:
            return "A wall of the recess is surrounded by the turned surface."
        wall_chains[f] = ch
    for f, chs in wall_chains.items():
        for _, _, ch in chs:
            if ch[0] not in corners or ch[-1] not in corners:
                return "A wall of the recess ends in an unexpected way."

    def refpos(r):
        return r[1].co if r[0] == "C" else pos[r]
    assign = {}
    for r, fs in boundary_refs.items():
        if r[0] == "C":
            continue
        p = refpos(r)
        best = None
        for f in walls:
            p0, nn = planes[f]
            if abs((p - p0).dot(nn)) > 1e-4 * size:
                continue
            for ci, (_, _, ch) in enumerate(wall_chains[f]):
                for q0, q1 in zip(ch, ch[1:]):
                    d = q1.co - q0.co
                    L2 = max(d.length_squared, 1e-30)
                    t = max(0.0, min(1.0, (p - q0.co).dot(d) / L2))
                    dist = (q0.co + d * t - p).length
                    if best is None or dist < best[0]:
                        best = (dist, f, ci)
        if best is None:
            return "A new edge misses the recess walls."
        assign.setdefault((best[1], best[2]), []).append(r)

    def chain_param(ch, p):
        best, acc = None, 0.0
        for q0, q1 in zip(ch, ch[1:]):
            d = q1.co - q0.co
            L = d.length
            t = 0.0 if L < 1e-30 else max(0.0, min(1.0, (p - q0.co).dot(d) / (L * L)))
            dist = (q0.co + d * t - p).length
            if best is None or dist < best[0]:
                best = (dist, acc + t * L)
            acc += L
        return best[1]
    new_walls = {}
    for f in walls:
        loop = list(f.verts)
        n = len(loop)
        chs = wall_chains[f]
        skip = {}
        for ci, (i0, i1, ch) in enumerate(chs):
            refs = sorted(assign.get((f, ci), []), key=lambda r: chain_param(ch, refpos(r)))
            skip[i0] = (i1, refs)
        out, i, cnt = [], 0, 0
        # start on a vertex that isn't inside a chain
        inside = set()
        for i0, i1, ch in chs:
            k = i0 + 1
            while k % n != i1:
                inside.add(k % n)
                k += 1
        start = next((k for k in range(n) if k not in inside), 0)
        k = start
        while cnt < n:
            out.append(("C", loop[k]) if loop[k] in corners else ("V", loop[k]))
            if k in skip:
                i1, refs = skip[k]
                out += refs
                adv = (i1 - k) % n
                k, cnt = i1, cnt + adv
            else:
                k, cnt = (k + 1) % n, cnt + 1
        new_walls[f] = list(dict.fromkeys(out))

    caps_new = []
    for f in S["order_caps"]:
        ri = data["ring_of"][f.verts[0]]
        k = S["order"].index(ri) if ri in S["order"] else None
        if k is None or k not in (0, nrow - 1):
            return "An end cap is in the middle of the profile."
        if any(state[G(k, j)] for j in range(count)):
            return "An end cap is cut by the recess."
        caps_new.append((f, [G(k, j) for j in range(count)], f.normal.copy()))

    from mathutils.bvhtree import BVHTree
    surf_list = ordered(surface)
    vmap, coords, ps = {}, [], []
    for f in surf_list:
        poly = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = len(coords)
                coords.append(v.co.copy())
            poly.append(vmap[v])
        ps.append(poly)
    ref_bvh = BVHTree.FromPolygons(coords, ps)
    ref_normals = [f.normal.copy() for f in surf_list]
    mats = [f.material_index for f in surf_list]
    mat = max(set(mats), key=mats.count)
    smooth = sum(f.smooth for f in surf_list) * 2 >= len(surf_list)
    wall_attr = {f: (f.material_index, f.smooth, f.normal.copy()) for f in walls}
    cap_attr = {f: (f.material_index, f.smooth) for f, _, _ in caps_new}

    def do():
        old_verts = {v for f in list(surface) + list(walls) + [f for f, _, _ in caps_new] for v in f.verts}
        bmesh.ops.delete(bm, geom=list(surface) + list(walls) + [f for f, _, _ in caps_new],
                         context="FACES_ONLY")
        made = {}

        def V(r):
            if r[0] in ("C", "V"):
                return r[1]
            if r not in made:
                made[r] = bm.verts.new(pos[r])
            return made[r]
        new_faces = []
        for poly in polys:
            vs = list(dict.fromkeys(V(r) for r in poly))
            if len(vs) >= 3:
                new_faces.append(bm.faces.new(vs))
        bm.normal_update()
        flip = []
        for f in new_faces:
            hit = ref_bvh.find_nearest(f.calc_center_median())
            if hit[0] is not None and f.normal.dot(ref_normals[hit[2]]) < 0:
                flip.append(f)
        for f in new_faces:
            f.material_index, f.smooth, f.select = mat, smooth, True
        others = []
        for f, refs in new_walls.items():
            vs = list(dict.fromkeys(V(r) for r in refs))
            nf = bm.faces.new(vs)
            mi, sm, nrm = wall_attr[f]
            nf.material_index, nf.smooth, nf.select = mi, sm, True
            nf.normal_update()
            if nf.normal.dot(nrm) < 0:
                flip.append(nf)
            others.append(nf)
        for f, refs, nrm in caps_new:
            vs = list(dict.fromkeys(V(r) for r in refs))
            nf = bm.faces.new(vs)
            nf.material_index, nf.smooth = cap_attr[f]
            nf.select = True
            nf.normal_update()
            if nf.normal.dot(nrm) < 0:
                flip.append(nf)
        if flip:
            bmesh.ops.reverse_faces(bm, faces=flip)
        wire = list({e for v in old_verts if v.is_valid for e in v.link_edges if not e.link_faces})
        if wire:
            bmesh.ops.delete(bm, geom=wire, context="EDGES")
        loose = [v for v in old_verts if v.is_valid and not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        return True

    ok, _ = guarded(bm, do)
    if not ok:
        return "This part can't be rebuilt safely with this number of sides (try another one)."
    return None
