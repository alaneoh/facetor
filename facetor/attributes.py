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

"""UVs, seams, sharp edges and edge weights carried over to the rebuilt geometry, whatever the
mode, plus the optional triangulation of the n-gons left by a change.

Before a change every face, edge and vertex gets an id (temporary integer layers). After it:
- loops of new faces, and loops of old faces whose vertex is new or moved, get their UV from the
  old faces: the old face nearest to the vertex if its UV agrees with the old face nearest to the
  new face (same UV island), otherwise extrapolated from the latter (a face never straddles an
  island border);
- new or changed edges lying along an old seam / sharp / weighted chain get its marks.
"""

import math
import bmesh
import mathutils.kdtree
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform, intersect_point_line

F_ID, E_ID, V_ID = "_facetor_f", "_facetor_e", "_facetor_v"
EDGE_FLOAT_LAYERS = ("bevel_weight_edge", "crease_edge")


def _layer(seq, name):
    lay = seq.layers.int.get(name)
    return lay if lay is not None else seq.layers.int.new(name)


def _remove(seq, name):
    lay = seq.layers.int.get(name)
    if lay is not None:
        seq.layers.int.remove(lay)


def _region(bm):
    """Faces around the selection (2 rings), where the change will happen."""
    seeds = {f for v in bm.verts if v.select for f in v.link_faces}
    if not seeds:
        return list(bm.faces)
    reg = set(seeds)
    for _ in range(2):
        reg |= {g for f in list(reg) for v in f.verts for g in v.link_faces}
    return [f for f in bm.faces if f in reg]     # fixed order (a set's changes between runs)


class Keeper:
    """Snapshot of UVs and edge marks around the selection; apply() carries them over."""

    def __init__(self, bm):
        self.bm = bm
        fl, el, vl = _layer(bm.faces, F_ID), _layer(bm.edges, E_ID), _layer(bm.verts, V_ID)
        for i, f in enumerate(bm.faces):
            f[fl] = i + 1
        for i, e in enumerate(bm.edges):
            e[el] = i + 1
        self.vpos = {}
        self.fcount = {i + 1: len(f.verts) for i, f in enumerate(bm.faces)}
        for i, v in enumerate(bm.verts):
            v[vl] = i + 1
            self.vpos[i + 1] = v.co.copy()
        self.size = max((max(v.co[k] for v in bm.verts) - min(v.co[k] for v in bm.verts)
                         for k in range(3)), default=1.0) if bm.verts else 1.0
        # UVs: old faces as triangles (positions + UV per layer)
        self.uv_layers = list(bm.loops.layers.uv.values())
        self.tris, self.face_tris = [], {}
        if self.uv_layers:
            for f in _region(bm):
                loops = list(f.loops)
                ids = []
                for i in range(1, len(loops) - 1):
                    tri = (loops[0], loops[i], loops[i + 1])
                    ids.append(len(self.tris))
                    self.tris.append(([l.vert.co.copy() for l in tri],
                                      [[Vector((l[lay].uv.x, l[lay].uv.y, 0.0)) for l in tri]
                                       for lay in self.uv_layers], f[fl], f.normal.copy()))
                self.face_tris[f[fl]] = ids
            self.bvh = BVHTree.FromPolygons([p for t in self.tris for p in t[0]],
                                            [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(self.tris))],
                                            all_triangles=True) if self.tris else None
        # edge marks: chains of marked edges (seam, sharp, weights), as segments
        self.fl_layers = {n: bm.edges.layers.float.get(n) for n in EDGE_FLOAT_LAYERS}
        self.marks = []     # (p0, p1, attrs, chain id)
        marked = [e for e in bm.edges if self._attrs(e)]
        chain_of = {}
        for e in marked:
            if e in chain_of:
                continue
            cid = len(chain_of)
            stack = [e]
            chain_of[e] = cid
            while stack:
                x = stack.pop()
                for v in x.verts:
                    for o in v.link_edges:
                        if o not in chain_of and self._attrs(o) == self._attrs(x):
                            chain_of[o] = cid
                            stack.append(o)
        for e in marked:
            self.marks.append((e.verts[0].co.copy(), e.verts[1].co.copy(), self._attrs(e), chain_of[e]))

    def _attrs(self, e):
        a = {}
        if e.seam:
            a["seam"] = True
        if not e.smooth:
            a["sharp"] = True
        for n, lay in self.fl_layers.items():
            if lay is not None and e[lay] > 0.0:
                a[n] = e[lay]
        return tuple(sorted(a.items()))

    # ----- after the change
    def _moved(self, v, vl):
        i = v[vl]
        return i == 0 or (v.co - self.vpos[i]).length > 1e-7 * self.size

    def _uv_from(self, ti, p):
        pts, uvs = self.tris[ti][0], self.tris[ti][1]
        return [barycentric_transform(p, pts[0], pts[1], pts[2], uv[0], uv[1], uv[2]) for uv in uvs]

    def _nearest_tri(self, ids, p):
        best = None
        for ti in ids:
            pts = self.tris[ti][0]
            c = (pts[0] + pts[1] + pts[2]) / 3
            d = (c - p).length
            if best is None or d < best[0]:
                best = (d, ti)
        return best[1] if best else None

    def _ref_face(self, f):
        """Old face a new face takes its UVs from: the one found most often under the face
        (centre and points near its corners) among those facing the same way."""
        c = f.calc_center_median()
        n = f.normal
        votes = {}
        for p in [c] + [v.co.lerp(c, 0.3) for v in f.verts]:
            hit = self._nearest_same_side(p, n)
            if hit is not None:
                fid = self.tris[hit[2]][2]
                votes[fid] = votes.get(fid, 0) + 1
        if votes:
            return max(votes, key=votes.get)
        hit = self.bvh.find_nearest(c)
        return self.tris[hit[2]][2] if hit[2] is not None else None

    def _nearest_same_side(self, p, n):
        """Nearest old triangle whose face looks the same way as n (within 60°), else None."""
        hit = self.bvh.find_nearest(p)
        if hit[2] is None:
            return None
        if n.length < 0.5 or self.tris[hit[2]][3].dot(n) > 0.5:
            return hit
        hits = self.bvh.find_nearest_range(p, max(hit[3] * 4, 1e-4 * self.size))
        hits = [h for h in hits if self.tris[h[2]][3].dot(n) > 0.5]
        return min(hits, key=lambda h: h[3]) if hits else None

    def apply(self, new_faces=True):
        """new_faces=False: the mode set the UVs of the faces it made, only fix the old faces
        that lost / gained / moved a vertex."""
        bm = self.bm
        fl, el, vl = bm.faces.layers.int.get(F_ID), bm.edges.layers.int.get(E_ID), bm.verts.layers.int.get(V_ID)
        if fl is None or el is None or vl is None:
            return
        moved = {v for v in bm.verts if self._moved(v, vl)}
        if self.uv_layers and self.tris:
            for f in bm.faces:
                if f[fl] != 0 and f[fl] not in self.face_tris:
                    continue        # an old face away from the change (another piece): as it was
                new_face = f[fl] == 0
                if new_face and not new_faces:
                    continue
                todo = [l for l in f.loops if new_face or l.vert in moved]
                if not todo:
                    continue
                if new_face:
                    fid = self._ref_face(f)
                    if fid is None:
                        continue
                    ref = self.face_tris[fid]
                else:
                    ref = self.face_tris[f[fl]]
                diag = 0.0
                for ti in ref:
                    u = self.tris[ti][1][0]
                    diag = max(diag, (u[0] - u[1]).length, (u[1] - u[2]).length, (u[2] - u[0]).length)
                tol = max(1e-4, 0.25 * diag)
                for l in todo:
                    p = l.vert.co
                    a = self._uv_from(self._nearest_tri(ref, p), p)
                    hit = self._nearest_same_side(p, f.normal)
                    b = self._uv_from(hit[2], hit[0]) if hit is not None else None
                    uv = b if b is not None and (a[0] - b[0]).length < tol else a
                    for lay, q in zip(self.uv_layers, uv):
                        l[lay].uv = (q.x, q.y)
        if self.marks:
            self._apply_marks(el, moved)

    def _apply_marks(self, el, moved):
        bm = self.bm
        # old marked segments by their middles: a segment's points are within half its length of
        # its middle, so the ones that can be closer than r are the middles within r + max_half
        # (long segments are cut into pieces of about the median length, so the search stays local)
        lens = sorted((m[1] - m[0]).length for m in self.marks)
        piece = max(lens[len(lens) // 2] if lens else 0.0, 1e-9)
        pts = []
        for i, m in enumerate(self.marks):
            k = max(1, int(math.ceil((m[1] - m[0]).length / piece)))
            for j in range(k):
                pts.append((m[0].lerp(m[1], (j + 0.5) / k), i))
        kd = mathutils.kdtree.KDTree(max(1, len(pts)))
        for n, (p, i) in enumerate(pts):
            kd.insert(p, n)
        kd.balance()
        max_half = piece / 2

        def candidates(p, r):
            seen = {pts[n][1] for _, n, _ in kd.find_range(p, r + max_half)}
            return [self.marks[i] for i in seen]

        def near(p):
            """(distance, old segment length, mark) of the closest old marked segment."""
            co, i, d0 = kd.find(p)
            if i is None:
                return None
            best = None
            for m in candidates(p, d0 + max_half):
                p0, p1 = m[0], m[1]
                q, t = intersect_point_line(p, p0, p1)
                q = p0 if t < 0 else (p1 if t > 1 else q)
                d = (q - p).length
                if best is None or d < best[0]:
                    best = (d, (p1 - p0).length, m)
            return best
        eps = 1e-6 * self.size
        for e in bm.edges:
            if e[el] != 0 and not any(v in moved for v in e.verts):
                continue
            a, b = e.verts[0].co, e.verts[1].co
            mid = near((a + b) / 2)
            length = (b - a).length
            if mid is None or length < eps:
                continue
            dm, lm, m = mid
            dirn = (b - a).normalized()
            seg = m[1] - m[0]
            # the middle of the edge lies on an old marked edge, in the same direction
            if dm > 0.45 * max(lm, 0.5 * length) + eps:
                continue
            if seg.length < eps or abs(seg.normalized().dot(dirn)) < 0.8:
                continue
            # and each end lies on the same marked line, next to a marked edge going roughly the
            # same way (an edge between two different marked lines, e.g. along a bore from one
            # marked circle to the other, or a fan edge from an outline to a circle, fails). At a
            # corner several marked edges meet: any of them may be the one.
            spans = [min(x.calc_length() for x in v.link_edges) for v in e.verts]

            def end_ok(p, span):
                for x in candidates(p, 0.45 * span + eps):
                    if x[3] != m[3]:
                        continue
                    sx = x[1] - x[0]
                    lx = sx.length
                    if lx < eps or abs(sx.dot(dirn)) < 0.6 * lx:
                        continue
                    q, t = intersect_point_line(p, x[0], x[1])
                    q = x[0] if t < 0 else (x[1] if t > 1 else q)
                    if (q - p).length <= 0.45 * min(lx, span) + eps:
                        return True
                return False
            if not (end_ok(a, spans[0]) and end_ok(b, spans[1])):
                continue
            attrs = dict(m[2])
            if attrs.get("seam"):
                e.seam = True
            if attrs.get("sharp"):
                e.smooth = False
            for n, lay in self.fl_layers.items():
                if lay is not None and n in attrs:
                    e[lay] = max(e[lay], attrs[n])

    def changed_faces(self):
        """Faces that are new or lost / gained / moved a vertex in the change."""
        bm = self.bm
        fl, vl = bm.faces.layers.int.get(F_ID), bm.verts.layers.int.get(V_ID)
        if fl is None:
            return []
        return [f for f in bm.faces if f[fl] == 0 or len(f.verts) != self.fcount.get(f[fl])
                or any(self._moved(v, vl) for v in f.verts)]

    def free(self):
        for seq, name in ((self.bm.faces, F_ID), (self.bm.edges, E_ID), (self.bm.verts, V_ID)):
            try:
                _remove(seq, name)
            except Exception:
                pass


def triangulate_ngons(bm, keeper):
    """Triangulate what a change left: faces of more than 4 sides, and old triangles that got
    a vertex more (beauty method). New quads (the rebuilt part) are kept."""
    fl = bm.faces.layers.int.get(F_ID)
    ngons = [f for f in keeper.changed_faces() if f.is_valid and (
        len(f.verts) > 4 or len(f.verts) == 4 and f[fl] != 0 and keeper.fcount.get(f[fl]) == 3)]
    if ngons:
        bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")
    return len(ngons)
