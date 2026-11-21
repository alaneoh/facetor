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

"""Messy CAD strips, one-face-wide ribbons, stacked strips, edge-chain selections."""

import math
import bmesh
from mathutils import Vector

from .core import (
    _face_components,
    chain_end_corner,
    choose_ref_axis,
    edge_between,
    hole_frame,
    hole_points,
    keep_closed,
    keep_open,
    resample_curve,
)
from .grid import _free_flat_side, _resize_chain
from .safety import guarded


# ---------------------------------------------------------------------------
# Messy extruded strips: rebuilt from their two end curves
# (MoI-style strips with extra vertices and T-junctions, not a clean quad grid)
# ---------------------------------------------------------------------------

def selection_boundary(faces):
    fset = set(faces)
    bedges = [e for f in faces for e in f.edges if sum(1 for g in e.link_faces if g in fset) == 1]
    bedges = list(dict.fromkeys(bedges))
    adj = {}
    for e in bedges:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    if any(len(l) != 2 for l in adj.values()):
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


def _straight_runs(loop, tol=math.radians(2)):
    """Maximal runs of collinear boundary edges: list of (start, end, length) vertex indices."""
    n = len(loop)
    dirs = [(loop[(i + 1) % n].co - loop[i].co) for i in range(n)]
    straight_join = [False] * n   # at vertex i: edge i-1 and edge i are collinear
    for i in range(n):
        a, b = dirs[i - 1], dirs[i]
        straight_join[i] = a.length > 1e-12 and b.length > 1e-12 and a.angle(b) < tol
    # rotate so we start at a corner
    starts = [i for i in range(n) if not straight_join[i]]
    if not starts:
        return []
    runs = []
    for si, s0 in enumerate(starts):
        s1 = starts[(si + 1) % len(starts)]
        length = 0.0
        i = s0
        while True:
            length += dirs[i].length
            i = (i + 1) % n
            if i == s1:
                break
        runs.append((s0, s1, length))
    return runs


def _chain(loop, a, b):
    """Vertices of the loop from index a to index b inclusive (forward, cyclic)."""
    n = len(loop)
    out, i = [loop[a]], a
    while i != b:
        i = (i + 1) % n
        out.append(loop[i])
    return out


def _ribbon_order(faces):
    """Faces of a one-face-wide band in order (each joined to the next by one edge), and the
    edges between them. None if the selection isn't such a band."""
    fset = set(faces)
    if len(faces) < 2:
        return None
    nbrs = {}
    for f in faces:
        lst = []
        for e in f.edges:
            for g in e.link_faces:
                if g is not f and g in fset:
                    lst.append((g, e))
        if len({g for g, _ in lst}) != len(lst) or len(lst) > 2:
            return None
        nbrs[f] = lst
    ends = [f for f in faces if len(nbrs[f]) == 1]
    if len(ends) != 2:
        return None
    order, rungs, prev, cur = [ends[0]], [], None, ends[0]
    while len(order) < len(faces):
        nxt = [(g, e) for g, e in nbrs[cur] if g is not prev]
        if not nxt:
            return None
        g, e = nxt[0]
        rungs.append(e)
        order.append(g)
        prev, cur = cur, g
    return order, rungs


def _end_run(loop, p, q, rung):
    """In the loop part from index p to q (forward), the run of straight edges that goes across
    the band (most parallel to the edge `rung`): (start, end) loop indices, or None."""
    n = len(loop)
    idx = [p]
    while idx[-1] != q:
        idx.append((idx[-1] + 1) % n)
        if len(idx) > n:
            return None
    if len(idx) < 2:
        return None
    rd = (rung.verts[1].co - rung.verts[0].co)
    if rd.length < 1e-12:
        return None
    rd = rd.normalized()
    runs, start = [], 0
    for t in range(1, len(idx) - 1):
        a = loop[idx[t]].co - loop[idx[t - 1]].co
        b = loop[idx[t + 1]].co - loop[idx[t]].co
        if a.length < 1e-12 or b.length < 1e-12 or a.angle(b) > math.radians(2):
            runs.append((start, t))
            start = t
    runs.append((start, len(idx) - 1))
    best, best_key = None, None
    for a, b in runs:
        d = loop[idx[b]].co - loop[idx[a]].co
        if d.length < 1e-12:
            continue
        key = (round(abs(d.normalized().dot(rd)), 2), d.length)
        if best_key is None or key > best_key:
            best, best_key = (idx[a], idx[b]), key
    if best is None or best_key[0] < 0.5:
        return None
    return best


def ribbon_ends(faces):
    """Ends and sides of a band one face wide (CAD wall with n-gons and T-junctions):
    dict like strip_ends (+ count = number of faces), or None if it isn't one."""
    ro = _ribbon_order(faces)
    if ro is None:
        return None
    order, rungs = ro
    loops = selection_boundary(faces)
    if not loops or len(loops) != 1:
        return None
    loop = loops[0]
    n = len(loop)
    pos = {v: i for i, v in enumerate(loop)}
    if any(v not in pos for e in rungs for v in e.verts):
        return None
    m = len(rungs)
    rung_of = {}
    for k, e in enumerate(rungs):
        for v in e.verts:
            rung_of.setdefault(v, []).append(k)
    good = False
    for start in rungs[0].verts:
        seq, i = [], pos[start]
        for _ in range(n):
            if loop[i] in rung_of:
                seq.append(loop[i])
            i = (i + 1) % n
        if len(seq) != 2 * m:
            break
        good = all(k in rung_of[seq[k]] for k in range(m)) and \
            all((m - 1 - k) in rung_of[seq[m + k]] for k in range(m))
        if good:
            a_side, b_side = seq[:m], seq[m:][::-1]
            break
    else:
        return None
    if not good:
        return None
    s1 = _end_run(loop, pos[b_side[0]], pos[a_side[0]], rungs[0])
    s2 = _end_run(loop, pos[a_side[-1]], pos[b_side[-1]], rungs[-1])
    if s1 is None or s2 is None:
        return None
    i, j = s1
    k, l = s2
    A = _chain(loop, j, k)
    B = _chain(loop, l, i)[::-1]
    S1 = _chain(loop, i, j)
    S2 = _chain(loop, k, l)
    if len(A) < 2 or len(B) < 2 or set(A) & set(B):
        return None
    return dict(closed=False, A=A, B=B, S1=S1, S2=S2, count=len(faces))


def strip_ends(faces, first=None):
    """Find the two end curves (and the two straight sides) of an extruded strip selection.
    Returns dict(closed, A, B, S1, S2) or (None, error)."""
    rib = ribbon_ends(faces)
    if rib is not None:
        return rib, None
    loops = selection_boundary(faces)
    if not loops:
        return None, "The selection's border isn't a simple outline."
    if len(loops) == 2:
        A, B = loops
        if first is not None and set(B) & set(first) and not set(A) & set(first):
            A, B = B, A          # the outline the user selected (a hole's rim) leads
        if len(A) < 3 or len(B) < 3:
            return None, "Selection too small."
        cA = sum((v.co for v in A), Vector()) / len(A)
        cB = sum((v.co for v in B), Vector()) / len(B)
        t = cB - cA
        start = min(range(len(B)), key=lambda i: (B[i].co - (A[0].co + t)).length)
        B = B[start:] + B[:start]
        # same direction as A?
        if len(B) > 2 and (B[1].co - (A[1].co + t)).length > (B[-1].co - (A[1].co + t)).length:
            B = [B[0]] + B[1:][::-1]
        return dict(closed=True, A=A, B=B, S1=None, S2=None), None
    if len(loops) != 1:
        return None, "The selection must be one strip (one outline) or one ring (two outlines)."
    loop = loops[0]
    runs = sorted(_straight_runs(loop), key=lambda r: -r[2])
    if len(runs) < 4:
        return None, "Couldn't find the two straight sides of the strip."
    r1, r2 = runs[0], runs[1]
    d1 = (loop[r1[1]].co - loop[r1[0]].co)
    d2 = (loop[r2[1]].co - loop[r2[0]].co)
    if d1.length < 1e-12 or d2.length < 1e-12 or abs(d1.normalized().dot(d2.normalized())) < 0.98:
        return None, "The strip's two long sides aren't parallel (not an extrusion)."
    n = len(loop)
    # sides S1 = r1 (i..j), S2 = r2 (k..l); A = j..k, B = l..i
    i, j = r1[0], r1[1]
    k, l = r2[0], r2[1]
    A = _chain(loop, j, k)
    B = _chain(loop, l, i)[::-1]       # from S1 side (i) to S2 side (l)
    S1 = _chain(loop, i, j)            # from B[0] to A[0]
    S2 = _chain(loop, k, l)            # from A[-1] to B[-1]
    if len(A) < 2 or len(B) < 2:
        return None, "Couldn't find the ends of the strip."
    return dict(closed=False, A=A, B=B, S1=S1, S2=S2), None


def _ruled_ok(Apts, Bpts, closed, test_pts, tol):
    """Are all test points on the surface made of straight lines between the two end curves?"""
    from mathutils.bvhtree import BVHTree
    n = max(len(Apts), len(Bpts), 8) * 2
    if closed:
        a = resample_curve(Apts, n, True)
        b = resample_curve(Bpts, n, True)
        polys = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    else:
        a = resample_curve(Apts, n, False)
        b = resample_curve(Bpts, n, False)
        polys = [(i, i + 1, n + 1 + i + 1, n + 1 + i) for i in range(n)]
    bvh = BVHTree.FromPolygons(a + b, polys)
    for p in test_pts:
        hit = bvh.find_nearest(p)
        if hit[0] is None or hit[3] > tol:
            return False
    return True


STRIP_NOT_RULED = "This strip isn't straight between its two ends (not an extrusion): it can't be rebuilt this way."
STRIPS_TOUCH = "These strips touch each other at a corner: change them one at a time."


def strip_check(faces, first=None):
    """strip_ends() plus the check that the strip is an extrusion between its two ends."""
    info, err = strip_ends(faces, first)
    if err:
        return info, err
    region_verts = {v for f in faces for v in f.verts}
    v0 = min(region_verts, key=lambda v: tuple(v.co))
    diag = max((v.co - v0.co).length for v in region_verts) or 1.0
    if not _ruled_ok([v.co.copy() for v in info["A"]], [v.co.copy() for v in info["B"]], info["closed"],
                     [v.co for v in region_verts], diag * 0.005):
        return None, STRIP_NOT_RULED
    return info, None


def strip_rebuild(bm, faces, count, closed_ref_axis=True, width=1.0, transform=None, first=None):
    """Rebuild a messy extruded strip with `count` segments across. Returns error or None.
    width: an open strip's end curves are scaled around their corners (a fillet's width);
    transform: point -> point applied to the new end curves instead (a hole's radius)."""
    info, err = strip_check(faces, first)
    if err:
        return err
    closed, A, B = info["closed"], info["A"], info["B"]
    if closed and count < 3:
        return "A closed ring needs at least 3 segments."
    Apts, Bpts = [v.co.copy() for v in A], [v.co.copy() for v in B]
    region_verts = {v for f in faces for v in f.verts}
    if closed:
        frame = hole_frame(Apts + Bpts)
        if frame is not None:
            # a hole / round wall: same angles on both outlines, so the columns follow the axis
            newA, base = hole_points(frame, Apts, count)
            newB, _ = hole_points(frame, Bpts, count, base)
            axis = frame[0]
            s = min(range(count), key=lambda i: ((newB[i] - newA[0]) - axis * (newB[i] - newA[0]).dot(axis)).length)
            newB = newB[s:] + newB[:s]
        else:
            ref = choose_ref_axis([Apts, Bpts])
            newA = keep_closed(Apts, count, ref)
            newB = keep_closed(Bpts, count, ref)
            t = sum(Bpts, Vector()) / len(Bpts) - sum(Apts, Vector()) / len(Apts)
            s = min(range(count), key=lambda i: (newB[i] - (newA[0] + t)).length)
            newB = newB[s:] + newB[:s]
    else:
        newA = keep_open(Apts, count)
        newB = keep_open(Bpts, count)
    pin = None
    if not closed:
        pin = pinned_plan([A, B], {e for f in faces for e in f.edges}, count)
        if pin:
            newA = pinned_points(A, *pin[0])
            newB = pinned_points(B, *pin[1])
    rails = []
    if transform is not None:
        newA = [transform(p) for p in newA]
        newB = [transform(p) for p in newB]
    elif abs(width - 1.0) > 1e-6:
        if closed:
            return "The width only changes open strips (a fillet between two faces), not a closed ring."
        if width < 1e-4:
            return "A width of 0 isn't available on strips: use a width above 0."
        cA = chain_end_corner(A, region_verts)
        cB = chain_end_corner(B, region_verts)
        if cA is None or cB is None:
            return "Couldn't find the corner of this strip's ends from the faces next to them: the width can't change."
        newA = [cA + (p - cA) * width for p in newA]
        newB = [cB + (p - cB) * width for p in newB]
    if not closed and (transform is not None or abs(width - 1.0) > 1e-6):
        # the ends moved: the straight sides between them follow (same fractions along them)
        for side, b0, a0 in ((info["S1"], newB[0], newA[0]), (info["S2"], newA[-1], newB[-1])):
            if side and len(side) > 2:
                tot = sum((y.co - x.co).length for x, y in zip(side, side[1:])) or 1.0
                acc = 0.0
                for x, y in zip(side, side[1:-1]):
                    acc += (y.co - x.co).length
                    rails.append((y, b0.lerp(a0, acc / tot)))
    boundary = set(A) | set(B) | set(info["S1"] or []) | set(info["S2"] or [])
    interior = [v for v in region_verts if v not in boundary]
    fset = set(faces)
    for v in interior:
        if any(f not in fset for f in v.link_faces):
            return "Some geometry is attached inside the selection."
    ref_normal = sum((f.normal * f.calc_area() for f in faces), Vector())
    mats = [f.material_index for f in faces]
    mat = max(set(mats), key=mats.count)
    smooth = sum(f.smooth for f in faces) * 2 >= len(faces)

    keep_edges = set()
    for chain in (A, B):
        seq = chain + ([chain[0]] if closed else [])
        for a, b in zip(seq, seq[1:]):
            e = bm.edges.get((a, b))
            if e:
                keep_edges.add(e)
    region_edges = {e for f in faces for e in f.edges}

    def do():
        bmesh.ops.delete(bm, geom=list(faces), context="FACES_ONLY")
        # edges that went across the strip (now loose) would hold the ends' vertices
        loose = [e for e in region_edges if e.is_valid and not e.link_faces and e not in keep_edges]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="EDGES")
        if interior:
            bmesh.ops.delete(bm, geom=[v for v in interior if v.is_valid], context="VERTS")
        ends = []
        for ci, (chain, pts) in enumerate(((A, newA), (B, newB))):
            has_out = any(f for v in chain for f in v.link_faces)
            if pin and pin[ci][1]:
                c = resize_pinned(bm, chain, pin[ci][1], pts)
                if c is None:
                    return False
            elif has_out:
                if count < len(chain) - (0 if closed else 1):
                    # held by a flat area cut into triangles (CAD fans around a hole): join
                    # them into one n-gon so the outline's vertices can go
                    _free_flat_side(bm, chain, closed, fans_only=True)
                c = _resize_chain(bm, chain, closed, count, pts)
            else:
                if closed:
                    c = [bm.verts.new(p) for p in pts]
                else:
                    c = [chain[0]] + [bm.verts.new(p) for p in pts[1:-1]] + [chain[-1]]
            if len(c) != len(pts):
                return False
            for v, p in zip(c, pts):
                v.co = p
            ends.append(c)
        for v, p in rails:
            if v.is_valid:
                v.co = p
        a, b = ends
        S1 = info["S1"][1:-1] if info["S1"] else []
        S2 = info["S2"][1:-1] if info["S2"] else []
        new_faces = []
        for c in range(count):
            c1 = (c + 1) % len(a)
            vs = [a[c], a[c1]]
            if not closed and c == count - 1:
                vs += S2
            vs += [b[c1], b[c]]
            if not closed and c == 0:
                vs += list(S1)
            f = bm.faces.new(vs)
            f.material_index = mat
            f.smooth = smooth
            new_faces.append(f)
        # a free end (open border, e.g. a hole in a thin shell) got new vertices: drop the old ones
        old_free = [v for v in list(A) + list(B) if v.is_valid and not v.link_faces]
        if old_free:
            bmesh.ops.delete(bm, geom=old_free, context="VERTS")
        bm.normal_update()
        n_new = sum((f.normal * f.calc_area() for f in new_faces), Vector())
        if n_new.dot(ref_normal) < 0:
            bmesh.ops.reverse_faces(bm, faces=new_faces)
        for f in new_faces:
            f.select = True
        return True

    ok, _ = guarded(bm, do)
    if not ok:
        return "This strip is joined to the rest of the mesh in a way that can't be changed safely."
    return None


# ---------------------------------------------------------------------------
# Stacked strips: several bands side by side along the same curve (a rounded corner with a
# wall and a chamfer, each cut differently by the CAD export). All get the same count.
# ---------------------------------------------------------------------------

SHARP_SPLIT = math.radians(30)


def _arc_fracs(pts):
    d = [0.0]
    for a, b in zip(pts, pts[1:]):
        d.append(d[-1] + (b - a).length)
    L = d[-1] or 1.0
    return [x / L for x in d]


def _at_frac(pts, fr, t):
    for i in range(1, len(pts)):
        if t <= fr[i] or i == len(pts) - 1:
            span = fr[i] - fr[i - 1]
            u = 0.0 if span <= 0 else min(1.0, max(0.0, (t - fr[i - 1]) / span))
            return pts[i - 1].lerp(pts[i], u)
    return pts[-1].copy()


def pinned_plan(chains, region_edges, count):
    """Vertices of the curves that are joined to other geometry (T-junctions) must not move.
    Returns {curve index: (slots t, {chain index: slot})} with `count` slots shared by all
    curves, or None when there is no such vertex or too many of them."""
    fr_all, pins = [], {}
    for ci, chain in enumerate(chains):
        fr = _arc_fracs([v.co for v in chain])
        for k in range(1, len(chain) - 1):
            if any(e not in region_edges for e in chain[k].link_edges):
                pins.setdefault(ci, []).append(k)
                fr_all.append(fr[k])
    if not pins:
        return None
    tol = 0.5 / count
    brk = [0.0]
    for x in sorted(fr_all):
        if x - brk[-1] > tol and 1.0 - x > tol:
            brk.append(x)
    brk.append(1.0)
    spans = [b - a for a, b in zip(brk, brk[1:])]
    if len(spans) > count:
        return None
    n = [max(1, int(round(count * sp))) for sp in spans]
    while sum(n) > count:
        i = max((i for i in range(len(n)) if n[i] > 1), key=lambda i: n[i] / spans[i])
        n[i] -= 1
    while sum(n) < count:
        i = max(range(len(n)), key=lambda i: spans[i] / n[i])
        n[i] += 1
    t = [0.0]
    for (a, b), m in zip(zip(brk, brk[1:]), n):
        t += [a + (b - a) * j / m for j in range(1, m + 1)]
    out = {}
    for ci, chain in enumerate(chains):
        fr = _arc_fracs([v.co for v in chain])
        slot = {}
        for k in pins.get(ci, []):
            j = min(range(1, count), key=lambda j: abs(t[j] - fr[k])) if count > 1 else None
            if j is None or j in slot.values():
                return None
            slot[k] = j
        out[ci] = (t, slot)
    return out


def pinned_points(chain, t, slot):
    """Points at the slots t along the chain, the pinned vertices exactly where they are."""
    pts = [v.co.copy() for v in chain]
    fr = _arc_fracs(pts)
    new = [_at_frac(pts, fr, x) for x in t]
    for k, j in slot.items():
        new[j] = chain[k].co.copy()
    return new


def resize_pinned(bm, chain, slot, new_pts):
    """Like _resize_chain, but the pinned vertices end up at their own slot: each piece of the
    chain between two of them is resized on its own."""
    marks = [(0, 0)] + sorted(slot.items(), key=lambda kv: kv[0]) + [(len(chain) - 1, len(new_pts) - 1)]
    out = [chain[0]]
    for (k0, j0), (k1, j1) in zip(marks, marks[1:]):
        if j1 <= j0 or k1 <= k0:
            return None
        piece = chain[k0:k1 + 1]
        pts = new_pts[j0:j1 + 1]
        if len(piece) == len(pts):
            c = piece
        else:
            c = _resize_chain(bm, piece, False, j1 - j0, pts)
        if c is None or len(c) != len(pts):
            return None
        out += c[1:]
    return out


def _smooth_parts(faces):
    """Split the faces where two neighbours make a sharp angle."""
    fset, seen, parts = set(faces), set(), []
    for f in faces:
        if f in seen:
            continue
        stack, part = [f], []
        seen.add(f)
        while stack:
            g = stack.pop()
            part.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h in fset and h not in seen and g.normal.angle(h.normal, 0.0) < SHARP_SPLIT:
                        seen.add(h)
                        stack.append(h)
        parts.append(part)
    return parts


def stacked_strips(faces):
    """[(faces, strip info)] if the selection is several strips joined along their long sides,
    else None."""
    parts = _smooth_parts(faces)
    if len(parts) < 2 or len(_face_components(faces)) != 1:
        return None
    out = []
    for part in parts:
        info, err = strip_check(part)
        if err or info["closed"]:
            return None
        out.append((part, info))
    keys = [(frozenset(info["A"]), frozenset(info["B"])) for _, info in out]
    for i, (ka, kb) in enumerate(keys):
        others = {k for j, kk in enumerate(keys) if j != i for k in kk}
        if ka not in others and kb not in others:
            return None
    return out


def stacked_rebuild(bm, parts, count):
    """Rebuild stacked strips with `count` segments each, sharing the curves between them."""
    curves = {}            # key -> [chain, new points, pinned slots]
    for _, info in parts:
        for c in (info["A"], info["B"]):
            k = frozenset(c)
            if k not in curves:
                curves[k] = [list(c), keep_open([v.co.copy() for v in c], count), None]
    all_faces = [f for part, _ in parts for f in part]
    keys = list(curves)
    # all curves must go the same way for the shared slots
    ref = curves[keys[0]][0]
    chains = []
    for k in keys:
        c = curves[k][0]
        if (c[0].co - ref[0].co).length > (c[-1].co - ref[0].co).length:
            c = c[::-1]
            curves[k][0] = c
        chains.append(c)
    pin = pinned_plan(chains, {e for f in all_faces for e in f.edges}, count)
    if pin:
        for i, k in enumerate(keys):
            curves[k][1] = pinned_points(chains[i], *pin[i])
            curves[k][2] = pin[i][1]
    fset = set(all_faces)
    region_verts = {v for f in all_faces for v in f.verts}
    edge_verts = set()
    for _, info in parts:
        edge_verts |= set(info["A"]) | set(info["B"]) | set(info["S1"] or []) | set(info["S2"] or [])
    interior = [v for v in region_verts if v not in edge_verts]
    for v in interior:
        if any(f not in fset for f in v.link_faces):
            return "Some geometry is attached inside the selection."
    curve_edges = set()
    for chain, _, _ in curves.values():
        for a, b in zip(chain, chain[1:]):
            e = bm.edges.get((a, b))
            if e:
                curve_edges.add(e)
    region_edges = {e for f in all_faces for e in f.edges}
    looks = [(part, sum((f.normal * f.calc_area() for f in part), Vector()),
              max(set(f.material_index for f in part), key=[f.material_index for f in part].count),
              sum(f.smooth for f in part) * 2 >= len(part)) for part, _ in parts]

    def do():
        bmesh.ops.delete(bm, geom=all_faces, context="FACES_ONLY")
        # edges that went across the strips (now loose) would hold the curves' vertices
        loose = [e for e in region_edges if e.is_valid and not e.link_faces and e not in curve_edges]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="EDGES")
        if interior:
            bmesh.ops.delete(bm, geom=[v for v in interior if v.is_valid], context="VERTS")
        new_chain = {}
        for k, (chain, pts, slots) in curves.items():
            if slots:
                c = resize_pinned(bm, chain, slots, pts)
                if c is None:
                    return False
            elif any(f for v in chain for f in v.link_faces):
                c = _resize_chain(bm, chain, False, count, pts)
            else:
                c = [chain[0]] + [bm.verts.new(p) for p in pts[1:-1]] + [chain[-1]]
            if len(c) != len(pts):
                return False
            for v, p in zip(c, pts):
                v.co = p
            new_chain[k] = (chain, c)

        def oriented(c):
            old, new = new_chain[frozenset(c)]
            return new if old[0] is c[0] else new[::-1]
        for (part, info), (_, ref_normal, mat, smooth) in zip(parts, looks):
            a, b = oriented(info["A"]), oriented(info["B"])
            S1 = info["S1"][1:-1] if info["S1"] else []
            S2 = info["S2"][1:-1] if info["S2"] else []
            new_faces = []
            for c in range(count):
                vs = [a[c], a[c + 1]]
                if c == count - 1:
                    vs += S2
                vs += [b[c + 1], b[c]]
                if c == 0:
                    vs += list(S1)
                f = bm.faces.new(vs)
                f.material_index = mat
                f.smooth = smooth
                new_faces.append(f)
            bm.normal_update()
            if sum((f.normal * f.calc_area() for f in new_faces), Vector()).dot(ref_normal) < 0:
                bmesh.ops.reverse_faces(bm, faces=new_faces)
            for f in new_faces:
                f.select = True
        return True

    ok, _ = guarded(bm, do)
    if not ok:
        return "These strips are joined to the rest of the mesh in a way that can't be changed safely."
    return None


# ---------------------------------------------------------------------------
# Edge selection: an open chain of edges along a curve -> the strips that run along it
# ---------------------------------------------------------------------------

def edge_chain_region(bm, closed_ok=False):
    """Faces of the strips that run along the selected open edge chain(s), or None. With
    closed_ok, closed loops too (a ring around a tube)."""
    if any(f.select for f in bm.faces):
        return None
    sel = [e for e in bm.edges if e.select]
    if not sel:
        return None
    deg = {}
    for e in sel:
        for v in e.verts:
            deg[v] = deg.get(v, 0) + 1
    if any(d > 2 for d in deg.values()) or not (closed_ok or any(d == 1 for d in deg.values())):
        return None                    # not an open chain (closed loops are holes)
    chain = set(sel)
    mids = [((e.verts[0].co + e.verts[1].co) / 2, (e.verts[1].co - e.verts[0].co).normalized()) for e in sel]
    length = sum(e.calc_length() for e in sel)

    def along(e):
        m = (e.verts[0].co + e.verts[1].co) / 2
        d = e.verts[1].co - e.verts[0].co
        if d.length < 1e-12:
            return False
        _, cd = min(mids, key=lambda md: (md[0] - m).length)
        return abs(d.normalized().dot(cd)) > 0.8

    # A planar chain across a straight extrusion (profile of the straight side of a fillet): the
    # region may go on along the extrusion, as long as the crossed edges are copies of the chain's
    # own edges moved along the plane's normal, i.e. up to where the side starts to turn.
    pts = [v.co for v in deg]
    c0 = sum(pts, Vector()) / len(pts)
    nrm = Vector()
    for (ma, da), (mb, db) in zip(mids, mids[1:]):
        n = da.cross(db)
        if n.length > 1e-6:
            n = n.normalized()
            nrm += n if n.dot(nrm) >= 0 else -n
    planar = nrm.length > 1e-6 and all(abs((p - c0).dot(nrm.normalized())) < 0.02 * length for p in pts)
    nrm = nrm.normalized() if planar else None

    def extruded(e):
        m = (e.verts[0].co + e.verts[1].co) / 2
        d = e.verts[1].co - e.verts[0].co
        pm = m - nrm * (m - c0).dot(nrm)
        cm, cd = min(mids, key=lambda md: (md[0] - pm).length)
        return abs(d.normalized().dot(cd)) > 0.995 and (cm - pm).length < 0.3 * d.length

    first = {f for e in sel for f in e.link_faces}
    small = [f for f in first if sum(1 for e in f.edges if e in chain) <= max(2, len(sel) // 3)
             and len(f.verts) <= 12]
    if not small:
        return None
    big = 4 * max(f.calc_area() for f in small)
    region = set(small)
    stack = list(small)
    while stack and len(region) < 2000:
        f = stack.pop()
        for e in f.edges:
            if e in chain or not along(e):
                continue
            for g in e.link_faces:
                if g in region or len(g.verts) > 12 or g.calc_area() > big:
                    continue
                # stay next to the chain: no farther than the chain is long, unless the faces are
                # the same profile extruded straight on
                if min((g.calc_center_median() - m).length for m, _ in mids) > length and \
                        not (nrm is not None and extruded(e)):
                    continue
                region.add(g)
                stack.append(g)
    return list(region)


# ---------------------------------------------------------------------------
# One edge chain across a straight extrusion (profile of the side of a fillet), when the faces
# along it are not a clean grid (e.g. n-gons left by a previous change): only the chain changes,
# the faces on each side are rebuilt as n-gons up to the next line, which stays as it is
# ---------------------------------------------------------------------------

def _poly_param(pts, p):
    """Arc length along polyline pts of the point nearest to p, and the distance to it."""
    best, acc = None, 0.0
    for a, b in zip(pts, pts[1:]):
        ab = b - a
        L = ab.length
        t = 0.0 if L < 1e-12 else max(0.0, min(1.0, (p - a).dot(ab) / (L * L)))
        d = (a + ab * t - p).length
        if best is None or d < best[1]:
            best = (acc + t * L, d)
        acc += L
    return best


def chain_only_check(bm):
    """The selected open edge chain (no face selected) if it lies in a plane and the faces on each
    side of it are one column of a straight extrusion of it (their other vertices project onto the
    chain along the plane's normal). Returns dict(chain, sides=[(faces, path, forward)...]) or None."""
    if any(f.select for f in bm.faces):
        return None
    sel = [e for e in bm.edges if e.select]
    if len(sel) < 1:
        return None
    adj = {}
    for e in sel:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    ends = [v for v, es in adj.items() if len(es) == 1]
    if len(ends) != 2 or any(len(es) > 2 for es in adj.values()):
        return None
    chain, v, prev = [ends[0]], ends[0], None
    while True:
        nxt = [e for e in adj[v] if e is not prev]
        if not nxt:
            break
        prev = nxt[0]
        v = prev.other_vert(v)
        chain.append(v)
    if len(chain) != len(sel) + 1:
        return None
    pts = [v.co.copy() for v in chain]
    L = sum((b - a).length for a, b in zip(pts, pts[1:]))
    c0 = sum(pts, Vector()) / len(pts)
    nrm = Vector()
    for i in range(len(pts) - 2):
        n = (pts[i + 1] - pts[i]).cross(pts[i + 2] - pts[i + 1])
        if n.length > 1e-9:
            n = n.normalized()
            nrm += n if n.dot(nrm) >= 0 else -n
    if nrm.length < 1e-6:
        # straight chain (one edge...): the extrusion goes towards the faces on its sides
        t = (pts[-1] - pts[0]).normalized()
        cset0 = set(chain)
        others = [v.co for e in sel for f in e.link_faces for v in f.verts if v not in cset0]
        if not others:
            return None
        nrm = Vector()
        for p in others:
            dv = p - c0
            dv -= t * dv.dot(t)
            if dv.length > 1e-9:
                dv = dv.normalized()
                nrm += dv if dv.dot(nrm) >= 0 else -dv
        if nrm.length < 1e-6:
            return None
    nrm = nrm.normalized()
    if any(abs((p - c0).dot(nrm)) > 0.02 * L for p in pts):
        return None
    cset = set(chain)
    sides = {1: set(), -1: set()}
    for i in range(len(chain) - 1):
        e = next(x for x in adj[chain[i]] if x.other_vert(chain[i]) is chain[i + 1])
        if len(e.link_faces) != 2:
            return None
        for f in e.link_faces:
            lp = next(l for l in f.loops if l.edge is e)
            sides[1 if lp.vert is chain[i] else -1].add(f)
    if not sides[1] or not sides[-1] or sides[1] & sides[-1]:
        return None
    # faces that touch the chain at a vertex only (fans left by an earlier change): on the side
    # where their other vertices are
    ref = {}
    for sgn, faces in sides.items():
        f = next(iter(faces))
        v = next(v for v in f.verts if v not in cset)
        ref[sgn] = 1 if (v.co - c0).dot(nrm) > 0 else -1
    for v in chain[1:-1]:
        for f in v.link_faces:
            if f in sides[1] or f in sides[-1]:
                continue
            others = [x for x in f.verts if x not in cset]
            if not others:
                return None
            o_s = 1 if (others[0].co - c0).dot(nrm) > 0 else -1
            sides[1 if ref[1] == o_s else -1].add(f)
    out = []
    for sgn, faces in sides.items():
        sgn_off = None
        for f in faces:
            for v in f.verts:
                if v in cset:
                    continue
                off = (v.co - c0).dot(nrm)
                pp = v.co - nrm * off
                _, d = _poly_param(pts, pp)
                if d > 0.5 * L or abs(off) < 1e-9:
                    return None
                s = 1 if off > 0 else -1
                if sgn_off is None:
                    sgn_off = s
                elif s != sgn_off:
                    return None
        # outer path of the column, from chain[0] to chain[-1]
        bedges = {e for f in faces for e in f.edges
                  if sum(1 for g in e.link_faces if g in faces) == 1 and not
                  (e.verts[0] in cset and e.verts[1] in cset)}
        path, v, used = [chain[0]], chain[0], set()
        for _ in range(len(bedges) + 1):
            nxt = [e for e in v.link_edges if e in bedges and e not in used]
            if len(nxt) != 1:
                break
            used.add(nxt[0])
            v = nxt[0].other_vert(v)
            path.append(v)
            if v is chain[-1]:
                break
        if path[-1] is not chain[-1] or len(used) != len(bedges):
            return None
        out.append((faces, path, sgn == 1))
    if out[0][0] & out[1][0]:
        return None
    return dict(chain=chain, nrm=nrm, c0=c0, sides=out)


def chain_only_rebuild(bm, info, count):
    """Give the chain of chain_only_check `count` segments along its curve; the faces on each side
    are rebuilt between the new chain and their outer path (n-gons). Returns (error, new chain)."""
    chain, nrm, c0 = info["chain"], info["nrm"], info["c0"]
    pts = [v.co.copy() for v in chain]
    new_pts = keep_open(pts, count)
    old_faces = [f for faces, _, _ in info["sides"] for f in faces]
    mat, smooth = old_faces[0].material_index, old_faces[0].smooth
    plans = []
    for faces, path, forward in info["sides"]:
        s = []
        for v in path:
            off = (v.co - c0).dot(nrm)
            s.append(_poly_param(pts, v.co - nrm * off)[0])
        for i in range(1, len(s)):
            s[i] = max(s[i], s[i - 1])          # monotone along the path
        plans.append((path, s, forward))
    u = [_poly_param(pts, p)[0] for p in new_pts]

    def do():
        bmesh.ops.delete(bm, geom=old_faces, context="FACES_ONLY")
        inner = [v for v in chain[1:-1] if v.is_valid]
        if inner:
            bmesh.ops.delete(bm, geom=inner, context="VERTS")
        for a, b in zip(chain, chain[1:]):
            e = edge_between(a, b) if a.is_valid and b.is_valid else None
            if e is not None and not e.link_faces:
                bm.edges.remove(e)
        N = [chain[0]] + [bm.verts.new(p) for p in new_pts[1:-1]] + [chain[-1]]
        new_faces = []
        for path, s, forward in plans:
            L_c = u[-1] if u[-1] > 1e-12 else 1.0
            # path vertices across from the chain (not the rails at its two ends)
            cand = [k for k in range(1, len(path) - 1) if 1e-3 * L_c < s[k] < L_c * (1 - 1e-3)]
            if count - 1 <= len(cand):
                # fewer new vertices: each one is linked to the nearest path vertex
                links = [(0, 0)]
                for i in range(1, count):
                    rng = [k for k in range(max(links[-1][1], 1), len(path) - 1)]
                    j = min(rng, key=lambda q: abs(s[q] - u[i])) if rng else links[-1][1]
                    links.append((i, max(j, links[-1][1])))
                links.append((count, len(path) - 1))
            else:
                # more new vertices: each path vertex across is linked to the nearest new one, so
                # each old segment takes the new vertices in it (n-gons, not triangles)
                links = [(0, 0)]
                for k in cand:
                    i = min(range(max(links[-1][0], 1), count), key=lambda q: abs(u[q] - s[k]))
                    links.append((max(i, links[-1][0]), k))
                links.append((count, len(path) - 1))
            for (i0, k0), (i1, k1) in zip(links, links[1:]):
                vs = N[i0:i1 + 1] + path[k0:k1 + 1][::-1]
                # drop repeated vertices (the chain's ends are on the path too)
                clean = []
                for v in vs:
                    if not clean or clean[-1] is not v:
                        clean.append(v)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()
                if len(set(clean)) < 3:
                    continue
                if not forward:
                    clean = clean[::-1]
                try:
                    f = bm.faces.new(clean)
                except ValueError:
                    return False
                f.material_index, f.smooth = mat, smooth
                new_faces.append(f)
        bm.normal_update()
        return N

    ok, N = guarded(bm, do)
    if not ok:
        return "This line is joined to the rest of the mesh in a way that can't be changed safely.", None
    return None, N


def chain_only_columns(bm, info, count):
    """S on the chain of chain_only_check: the straight extrusion between the two lines next to
    the chain (they never change) is cut into `count` columns by copies of the chain, evenly
    spaced along the extrusion (the chain itself goes). The rails gain or lose the copies' end
    vertices; the faces next to the two fixed lines are n-gons where their vertices differ.
    Returns (error, the new line nearest the old chain)."""
    if count < 1:
        return "Use 1 column or more.", None
    chain, nrm, c0 = info["chain"], info["nrm"], info["c0"]
    pts = [v.co.copy() for v in chain]
    L = sum((b - a).length for a, b in zip(pts, pts[1:])) or 1.0
    M = len(chain) - 1
    cset = set(chain)

    def off(v):
        return (v.co - c0).dot(nrm)

    def par(v):
        return _poly_param(pts, v.co - nrm * off(v))[0]
    rows, rails = [], ([], [])
    for faces, path, _ in info["sides"]:
        # path: chain[0] -> along the first rail -> across (the next line) -> back along the other
        # rail -> chain[-1]
        def along(a, b):
            d = b.co - a.co
            return d.length > 1e-12 and abs(d.normalized().dot(nrm)) > 0.7
        i0 = 0
        while i0 + 1 < len(path) and along(path[i0], path[i0 + 1]):
            i0 += 1
        i1 = len(path) - 1
        while i1 - 1 > i0 and along(path[i1 - 1], path[i1]):
            i1 -= 1
        if i0 == 0 or i1 == len(path) - 1:
            return "The line next to this chain doesn't go from one side to the other: the columns can't change.", None
        rails[0].extend(path[1:i0 + 1])
        rails[1].extend(path[i1:-1])
        rows.append(path[i0:i1 + 1])
    T = [sum(off(v) for v in r) / len(r) for r in rows]
    if T[0] * T[1] >= 0:
        return "The chain isn't between its two neighbour lines: the columns can't change.", None
    if T[0] > T[1]:
        rows, T = rows[::-1], T[::-1]
    t_new = [T[0] + (T[1] - T[0]) * k / count for k in range(1, count)]
    old_faces = [f for faces, _, _ in info["sides"] for f in faces]
    ref = [(f.calc_center_median(), f.normal.copy()) for f in old_faces]
    mat, smooth = old_faces[0].material_index, old_faces[0].smooth
    spacing = abs(T[1] - T[0]) / count

    def do():
        bmesh.ops.delete(bm, geom=old_faces, context="FACES_ONLY")
        inner = [v for v in chain[1:-1] if v.is_valid]
        if inner:
            bmesh.ops.delete(bm, geom=inner, context="VERTS")
        for a, b in zip(chain, chain[1:]):
            e = edge_between(a, b) if a.is_valid and b.is_valid else None
            if e is not None and not e.link_faces:
                bm.edges.remove(e)
        # each rail, sorted along the extrusion, from the first row's end to the second's
        R = []
        for side in (0, 1):
            ends = (rows[0][0], rows[1][0]) if side == 0 else (rows[0][-1], rows[1][-1])
            rv = [ends[0]] + sorted(set(rails[side]) | {chain[0] if side == 0 else chain[-1]},
                                    key=off) + [ends[1]]
            rv = list(dict.fromkeys(rv))
            R.append(rv)
        used_ends = set()
        line_ends = []
        for t in t_new:
            pair = []
            for side in (0, 1):
                rv = R[side]
                near = min(rv[1:-1], key=lambda v: abs(off(v) - t), default=None)
                if near is not None and abs(off(near) - t) < 0.25 * spacing and near not in used_ends:
                    v = near
                else:
                    k = next((k for k in range(len(rv) - 1) if off(rv[k]) < t < off(rv[k + 1])), None)
                    if k is None:
                        return False
                    a, b = rv[k], rv[k + 1]
                    e = edge_between(a, b)
                    if e is None:
                        return False
                    fr = (t - off(a)) / (off(b) - off(a))
                    p = a.co.lerp(b.co, fr)
                    _, v = bmesh.utils.edge_split(e, a, fr)
                    v.co = p
                    rv.insert(k + 1, v)
                used_ends.add(v)
                pair.append(v)
            line_ends.append(pair)
        lines = []
        for t, (a, b) in zip(t_new, line_ends):
            lines.append([a] + [bm.verts.new(p + nrm * t) for p in pts[1:-1]] + [b])
        all_rows = [rows[0]] + lines + [rows[1]]
        new_faces = []
        for A, B in zip(all_rows, all_rows[1:]):
            sa, sb = [par(v) for v in A], [par(v) for v in B]
            for q in (sa, sb):
                for i in range(1, len(q)):
                    q[i] = max(q[i], q[i - 1])
            ta, tb = off(A[0]), off(B[0])
            left = [v for v in R[0] if ta + 1e-9 < off(v) < tb - 1e-9 and v not in A and v not in B]
            ta, tb = off(A[-1]), off(B[-1])
            right = [v for v in R[1] if ta + 1e-9 < off(v) < tb - 1e-9 and v not in A and v not in B]
            # links between the two rows (monotone, nearest along the chain)
            if len(A) <= len(B):
                links = [(0, 0)]
                for i in range(1, len(A) - 1):
                    rng = range(max(links[-1][1], 1), len(B) - 1)
                    k = min(rng, key=lambda q: abs(sb[q] - sa[i]), default=links[-1][1])
                    links.append((i, max(k, links[-1][1])))
            else:
                links = [(0, 0)]
                for k in range(1, len(B) - 1):
                    rng = range(max(links[-1][0], 1), len(A) - 1)
                    i = min(rng, key=lambda q: abs(sa[q] - sb[k]), default=links[-1][0])
                    links.append((max(i, links[-1][0]), k))
            links.append((len(A) - 1, len(B) - 1))
            for n_l, ((i0, k0), (i1, k1)) in enumerate(zip(links, links[1:])):
                vs = A[i0:i1 + 1]
                if n_l == len(links) - 2:
                    vs = vs + right
                vs = vs + B[k0:k1 + 1][::-1]
                if n_l == 0:
                    vs = vs + left[::-1]
                vs = list(dict.fromkeys(vs))
                if len(vs) < 3:
                    continue
                try:
                    f = bm.faces.new(vs)
                except ValueError:
                    return False
                f.material_index, f.smooth = mat, smooth
                new_faces.append(f)
        bm.normal_update()
        flip = []
        for f in new_faces:
            c = f.calc_center_median()
            _, n = min(ref, key=lambda x: (x[0] - c).length)
            if f.normal.dot(n) < 0:
                flip.append(f)
        if flip:
            bmesh.ops.reverse_faces(bm, faces=flip)
        # the chain's ends, when no copy uses them and nothing else holds them, go
        for v in (chain[0], chain[-1]):
            if v.is_valid and v not in used_ends and len(v.link_edges) == 2:
                bmesh.ops.dissolve_verts(bm, verts=[v])
        if not lines:
            return []
        return min(lines, key=lambda ln: abs(off(ln[1 if len(ln) > 2 else 0])))

    ok, N = guarded(bm, do)
    if not ok:
        return "These columns are joined to the rest of the mesh in a way that can't be changed safely.", None
    return None, N
