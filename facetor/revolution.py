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

"""Parts of revolution tessellated in several pieces, or cut by a detail."""

import math
import bmesh
import mathutils.geometry
import numpy as np
from mathutils import Vector

from .core import (
    TAU,
    _projected_circle,
    choose_ref_axis,
    edge_between,
    extract_grid,
    fit_circle,
    grid_profiles,
    keep_open,
    quad_components,
    ordered,
)
from .grid import _resize_chain
from .safety import guarded
from .strips import selection_boundary


# ---------------------------------------------------------------------------
# Parts of revolution tessellated in several pieces (e.g. a rod whose tip has 51
# sides and its shaft 52, joined by a ring of n-gons): rebuilt with one side count
# ---------------------------------------------------------------------------

def _revolution_axis(bm, faces):
    """Axis of a part of revolution: from a closed round grid, a round n-gon or a round border."""
    cands = []
    quads = [f for f in faces if len(f.verts) == 4]
    for comp in quad_components(quads):
        g, err = extract_grid(comp)
        if err or not any(g.cyclic) or all(g.cyclic):
            continue
        d = 0 if g.cyclic[0] else 1
        for prof in grid_profiles(g, d):
            cands.append([v.co.copy() for v in prof])
    for f in faces:
        if len(f.verts) >= 8:
            cands.append([v.co.copy() for v in f.verts])
    for loop in selection_boundary(faces) or []:
        if len(loop) >= 8:
            cands.append([v.co.copy() for v in loop])
    best = None
    for pts in sorted(cands, key=lambda p: -len(p)):
        fit = fit_circle(pts)
        if fit and fit[3] < 0.01 * fit[2]:
            c, n, r, _ = fit
            if best is None or r > best[2]:
                best = (c, n.normalized(), r)
    return best


def revolution_rings(bm, faces):
    """Group the vertices into rings around the axis. Returns (axis, centre, rings) or (None, error).
    rings: list ordered along the profile, each dict(h, r, verts) (a pole has r = 0 and one vertex)."""
    ax = _revolution_axis(bm, faces)
    if ax is None:
        return None, "No round section found to get the axis from."
    c, a, rmax = ax
    verts = ordered({v for f in faces for v in f.verts})
    size = max(rmax, max(abs((v.co - c).dot(a)) for v in verts))
    htol, rtol = size * 2e-4, rmax * 0.01

    def hr(v):
        d = v.co - c
        h = d.dot(a)
        return h, (d - a * h).length
    data = sorted(((hr(v), v) for v in verts), key=lambda x: (x[0][0], x[0][1]))
    rings = []
    for (h, r), v in data:
        for ring in rings:
            if abs(ring["h"] - h) < htol and abs(ring["r"] - r) < max(rtol, 1e-9):
                ring["verts"].append(v)
                break
        else:
            rings.append({"h": h, "r": r, "verts": [v]})
    # every ring must go all the way round (no slot), poles are single vertices on the axis
    u = a.orthogonal().normalized()
    w = a.cross(u)
    for ring in rings:
        if ring["r"] < rtol:
            if len(ring["verts"]) != 1:
                return None, "Several vertices on the axis."
            ring["r"] = 0.0
            continue
        if len(ring["verts"]) < 3:
            return None, "This part isn't a clean part of revolution (a ring is incomplete)."
        angs = sorted(math.atan2((v.co - c).dot(w), (v.co - c).dot(u)) for v in ring["verts"])
        gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)] + [angs[0] + TAU - angs[-1]]
        if max(gaps) > min(math.pi / 2, 3 * TAU / len(angs)):
            return None, "This part isn't a clean part of revolution (something cuts its rings)."
        ring["verts"].sort(key=lambda v: math.atan2((v.co - c).dot(w), (v.co - c).dot(u)))
    # rings must follow each other like the points of a profile (a path)
    index = {v: i for i, ring in enumerate(rings) for v in ring["verts"]}
    adj = {i: set() for i in range(len(rings))}
    fset = set(faces)
    for f in faces:
        ids = {index[v] for v in f.verts}
        if len(ids) > 2:
            return None, "A face spans more than two rings."
        if len(ids) == 2:
            i, j = ids
            adj[i].add(j)
            adj[j].add(i)
    if any(len(n) > 2 for n in adj.values()):
        return None, "The rings don't follow each other like a profile."
    ends = [i for i, n in adj.items() if len(n) <= 1]
    if len(rings) > 1 and not ends:
        return None, "Closed profile (torus): not supported."
    order, prev = [ends[0] if ends else 0], None
    while True:
        nxt = [j for j in adj[order[-1]] if j != prev]
        if not nxt:
            break
        prev = order[-1]
        order.append(nxt[0])
    if len(order) != len(rings):
        return None, "The selection holds several separate parts."
    return (a, c, [rings[i] for i in order]), None


def profile_runs(pts, min_turn=math.radians(0.5), max_turn=math.radians(75)):
    """Curved parts (index ranges lo..hi) of an open polyline: the rounded parts of a profile."""
    n = len(pts)
    turn = [0.0] * n
    ratio = [99.0] * n
    for k in range(1, n - 1):
        v1, v2 = pts[k] - pts[k - 1], pts[k + 1] - pts[k]
        if v1.length > 1e-12 and v2.length > 1e-12:
            turn[k] = v1.angle(v2)
            ratio[k] = max(v1.length, v2.length) / min(v1.length, v2.length)

    def inside(k):
        return min_turn < turn[k] < max_turn and ratio[k] < 2.5

    def similar(k1, k2):
        t1, t2 = turn[k1], turn[k2]
        return max(t1, t2) <= 3 * min(t1, t2)
    runs, k = [], 1
    while k < n - 1:
        if inside(k):
            b, end = k, k
            while b + 1 < n - 1 and inside(b + 1) and similar(b, b + 1):
                b += 1
            if b == k and turn[k] > math.radians(50):
                k += 1  # a lone sharp bend is a corner, not a rounded part
                continue
            end = b
            inner = sorted(turn[k:b + 1])
            med = inner[len(inner) // 2]
            while k < b and turn[k] < 0.7 * med:
                k += 1
            while b > k and turn[b] < 0.7 * med:
                b -= 1
            runs.append((k - 1, b + 1))
            k = end + 1
        else:
            k += 1
    # a rounded tip ends on the axis: the pole is part of the run
    if runs and runs[-1][1] == n - 2 and n >= 2 and pts[-1].x < 1e-9:
        runs[-1] = (runs[-1][0], n - 1)
    if runs and runs[0][0] == 1 and pts[0].x < 1e-9:
        runs[0] = (0, runs[0][1])
    return runs


def revolution_profile_count(bm, faces):
    res, err = revolution_rings(bm, faces)
    if err:
        return None
    a, c, rings = res
    pts = [Vector((r["r"], r["h"], 0.0)) for r in rings]
    runs = profile_runs(pts)
    return max((hi - lo for lo, hi in runs), default=None)


def scale_profile_runs(rings, width):
    """Width on a part of revolution: each rounded part of the profile scaled around the corner
    of the straight parts on each side of it. Returns (new rings, None) or (None, error)."""
    if width < 1e-4:
        return None, "A width of 0 isn't available on a part of revolution: use a width above 0."
    pts = [Vector((r["r"], r["h"], 0.0)) for r in rings]
    runs = profile_runs(pts)
    if not runs:
        return None, "This part has no rounded profile: the width can't change."
    out = [dict(r) for r in rings]
    for lo, hi in runs:
        if lo < 1 or hi > len(pts) - 2:
            return None, "A rounded part of this profile ends at the edge of the selection: the width needs a straight part on each side."
        a, b = pts[lo], pts[hi]
        ta, tb = a - pts[lo - 1], b - pts[hi + 1]
        if ta.length < 1e-12 or tb.length < 1e-12 or abs(ta.normalized().dot(tb.normalized())) > 0.9999:
            return None, "Couldn't find the corner of a rounded part of this profile: the width can't change."
        r_ = mathutils.geometry.intersect_line_line(a, a + ta, b, b + tb)
        if not r_:
            return None, "Couldn't find the corner of a rounded part of this profile: the width can't change."
        cn = (r_[0] + r_[1]) / 2
        if (cn - a).dot(ta) <= 0 or (cn - b).dot(tb) <= 0:
            return None, "Couldn't find the corner of a rounded part of this profile: the width can't change."
        lim = min((cn - pts[lo - 1]).length / max((cn - a).length, 1e-12),
                  (cn - pts[hi + 1]).length / max((cn - b).length, 1e-12))
        if width > 0.98 * lim:
            return None, "This width goes past the straight parts next to the rounded profile (at most %.2f here)." % (0.98 * lim)
        for k in range(lo, hi + 1):
            p = cn + (pts[k] - cn) * width
            out[k]["r"], out[k]["h"] = max(p.x, 0.0), p.y
    return out, None


def revolution_rebuild(bm, faces, count, profile_segments=None, width=1.0):
    if count < 3:
        return "Needs at least 3 sides."
    res, err = revolution_rings(bm, faces)
    if err:
        return err
    a, c, rings = res
    if abs(width - 1.0) > 1e-6:
        rings, err = scale_profile_runs(rings, width)
        if err:
            return err
    fset = set(faces)
    removed_verts = []
    if profile_segments is not None:
        # resample the rounded parts of the profile (number of rings), keeping their shape
        if profile_segments < 1:
            return "Needs at least 1 segment."
        pts = [Vector((r["r"], r["h"], 0.0)) for r in rings]
        runs = profile_runs(pts)
        if not runs:
            return "This part has no rounded profile to change."
        new_rings, last = [], 0
        for lo, hi in runs:
            new_rings += rings[last:lo + 1]
            for r in rings[lo + 1:hi]:
                removed_verts += r["verts"]
            newp = keep_open(pts[lo:hi + 1], profile_segments)
            for p in newp[1:-1]:
                new_rings.append({"h": p.y, "r": max(p.x, 0.0), "verts": []})
            last = hi
        new_rings += rings[last:]
        orig_rings = rings
        rings = new_rings
    else:
        orig_rings = rings
    # orientation reference: the original faces
    from mathutils.bvhtree import BVHTree
    vmap, coords, polys = {}, [], []
    for f in faces:
        poly = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = len(coords)
                coords.append(v.co.copy())
            poly.append(vmap[v])
        polys.append(poly)
    ref_bvh = BVHTree.FromPolygons(coords, polys)
    ref_normals = [f.normal.copy() for f in faces]
    mats = [f.material_index for f in faces]
    mat = max(set(mats), key=mats.count)
    smooth = sum(f.smooth for f in faces) * 2 >= len(faces)
    # caps: selected n-gons whose vertices are all on one ring
    ring_of0 = {v: id(ring) for ring in orig_rings for v in ring["verts"]}
    cap_ids = set()
    for f in faces:
        if len(f.verts) >= 3 and len({ring_of0[v] for v in f.verts}) == 1:
            cap_ids.add(ring_of0[f.verts[0]])
    caps = {i for i, ring in enumerate(rings) if id(ring) in cap_ids}
    if any(id(r) in cap_ids for r in orig_rings if r not in rings):
        return "A cap sits on a ring that would be removed."
    # angles of the new sides: same symmetry as the largest ring
    big = max((r for r in orig_rings if r["r"] > 0 and r["verts"]), key=lambda r: (len(r["verts"]), r["r"]))
    pts = [v.co.copy() for v in big["verts"]]
    ref = choose_ref_axis([[p - a * (p - c).dot(a) for p in pts]])
    u = (ref - a * ref.dot(a)) if ref is not None else a.orthogonal()
    if u.length < 1e-6:
        u = a.orthogonal()
    u.normalize()
    w = a.cross(u)
    angs = [math.atan2((p - c).dot(w), (p - c).dot(u)) for p in pts]
    step_old, step_new = TAU / len(pts), TAU / count
    delta = min(angs, key=abs)
    base = delta if abs(delta) <= step_old / 4 else delta - math.copysign(step_old / 2, delta) + step_new / 2
    targets = []
    for ring in rings:
        if ring["r"] == 0:
            targets.append([c + a * ring["h"]])
        else:
            targets.append([c + a * ring["h"] + ring["r"] * (math.cos(base + step_new * i) * u +
                                                              math.sin(base + step_new * i) * w)
                            for i in range(count)])
    chain_rings = {i for i, ring in enumerate(rings)
                   if any(f not in fset for v in ring["verts"] for f in v.link_faces)}
    held = {v for i in chain_rings for v in rings[i]["verts"]}
    inner_edges = {e for f in faces for e in f.edges}

    def do():
        bmesh.ops.delete(bm, geom=list(faces), context="FACES_ONLY")
        if removed_verts:
            bmesh.ops.delete(bm, geom=[v for v in removed_verts if v.is_valid], context="VERTS")
        # edges left without faces towards rings that are rebuilt: removed first, or every vertex
        # of a held ring would look joined to outside geometry in _resize_chain (its T vertices
        # where outside lines end could then be dissolved, tearing the outside faces)
        for e in [e for e in inner_edges if e.is_valid and not e.link_faces
                  and not all(v in held for v in e.verts)]:
            bm.edges.remove(e)
        new_rings = []
        for i, ring in enumerate(rings):
            if i in chain_rings:
                vs = ring["verts"]
                if ring["r"] == 0:
                    new_rings.append(vs)
                    continue
                anchor = min(range(len(vs)), key=lambda k: (vs[k].co - targets[i][0]).length)
                vs = vs[anchor:] + vs[:anchor]
                chain = _resize_chain(bm, vs, True, count, targets[i])
                if len(chain) != count:
                    return False
                # keep the new list in the same order as the targets
                for v, p in zip(chain, targets[i]):
                    v.co = p
                new_rings.append(chain)
            else:
                old = [v for v in ring["verts"] if v.is_valid]
                if old:
                    bmesh.ops.delete(bm, geom=old, context="VERTS")
                new_rings.append([bm.verts.new(p) for p in targets[i]])
        new_faces = []
        for i in range(len(rings) - 1):
            A, B = new_rings[i], new_rings[i + 1]
            for k in range(count):
                k1 = (k + 1) % count
                if len(A) == 1:
                    vs = [A[0], B[k], B[k1]]
                elif len(B) == 1:
                    vs = [A[k], A[k1], B[0]]
                else:
                    vs = [A[k], A[k1], B[k1], B[k]]
                new_faces.append(bm.faces.new(vs))
        for i in caps:
            if len(new_rings[i]) >= 3:
                new_faces.append(bm.faces.new(new_rings[i]))
        bm.normal_update()
        flip = []
        for f in new_faces:
            hit = ref_bvh.find_nearest(f.calc_center_median())
            if hit[0] is not None and f.normal.dot(ref_normals[hit[2]]) < 0:
                flip.append(f)
        if flip:
            bmesh.ops.reverse_faces(bm, faces=flip)
        for f in new_faces:
            f.material_index = mat
            f.smooth = smooth
            f.select = True
        return True

    ok, _ = guarded(bm, do)
    if not ok:
        return "This part is joined to the rest of the mesh in a way that can't be changed safely."
    return None


def revolution_count(bm, faces):
    res, err = revolution_rings(bm, faces)
    if err:
        return None
    sizes = [len(r["verts"]) for r in res[2] if r["r"] > 0]
    return max(set(sizes), key=sizes.count) if sizes else None


# ---------------------------------------------------------------------------
# Parts of revolution cut by a detail (slot, recess...), possibly domed and made of
# pieces with different side counts: every ring gets the same columns, the edges of
# the detail stay exactly where they are
# ---------------------------------------------------------------------------

def _ring_clusters(verts, c, a, size, rmax):
    htol, rtol = size * 2e-4, rmax * 0.005
    data = sorted(((((v.co - c).dot(a), ((v.co - c) - a * (v.co - c).dot(a)).length), v) for v in verts),
                  key=lambda x: x[0])
    rings = []
    for (h, r), v in data:
        for ring in rings:
            if abs(ring["h"] - h) < htol and abs(ring["r"] - r) < max(rtol, 1e-9):
                ring["verts"].append(v)
                break
        else:
            rings.append({"h": h, "r": r, "verts": [v]})
    return rings


def _refine_axis(verts, c, a):
    """Tilt a nearly right axis onto the plane of its biggest ring of vertices."""
    rr = [((v.co - c) - a * (v.co - c).dot(a)).length for v in verts]
    rmax = max(rr) if rr else 0.0
    if rmax < 1e-12:
        return None
    size = max(rmax, max(abs((v.co - c).dot(a)) for v in verts))
    best = None
    for ring in _ring_clusters(verts, c, a, size * 15, rmax * 2):
        if len(ring["verts"]) >= 6 and (best is None or len(ring["verts"]) > len(best["verts"])):
            best = ring
    if best is None:
        return None
    P = np.array([tuple(v.co) for v in best["verts"]])
    idx = np.arange(len(P))
    for _ in range(3):
        # plane of the ring, without the stray points (a point of the recess near the ring)
        m = P.mean(axis=0)
        _, sv, vt = np.linalg.svd(P - m)
        if sv[1] < 1e-9:
            return None
        d = np.abs((P - m) @ vt[2])
        R = np.linalg.norm((P - m) - np.outer((P - m) @ vt[2], vt[2]), axis=1)
        dr = np.abs(R - np.median(R))
        keep = (d <= max(3 * np.median(d), 1e-9 * size)) & (dr <= max(3 * np.median(dr), 1e-7 * size))
        if keep.all() or keep.sum() < 6:
            break
        P = P[keep]
        idx = idx[keep]
    if sv[2] > 0.05 * sv[1]:
        return None
    best = {"verts": [best["verts"][i] for i in idx]}
    n = Vector(tuple(vt[2])).normalized()
    if n.dot(a) < 0:
        n = -n
    if n.dot(a) < math.cos(math.radians(5)):
        return None
    fit = _projected_circle([v.co for v in best["verts"]], n)
    if not fit or fit[2] > 0.01 * fit[1]:
        return None
    return fit[0], n


def _closed_edge_loops(faces, min_len=6):
    """Closed edge loops running straight across 4-edge vertices of the selection (the whole
    rings of a dish whose other rings are cut): lists of vertices."""
    fset = set(faces)
    edges = {e for f in faces for e in f.edges}
    seen, loops = set(), []
    for e0 in ordered(edges):
        if e0 in seen:
            continue
        loop, e, v = [], e0, e0.verts[0]
        while True:
            seen.add(e)
            loop.append(v)
            v = e.other_vert(v)
            if v is e0.verts[0]:
                break
            lf = [f for f in e.link_faces if f in fset]
            if len(v.link_edges) != 4 or len(lf) != 2 or any(f not in fset for f in v.link_faces):
                loop = None
                break
            # the edge across: shares no face with e
            nxt = [x for x in v.link_edges if x is not e and not (set(x.link_faces) & set(e.link_faces))]
            if len(nxt) != 1 or nxt[0] in seen:
                loop = None
                break
            e = nxt[0]
        if loop and len(loop) >= min_len:
            loops.append(loop)
    return loops


def _axis_candidates(bm, faces):
    out = []
    ax = _revolution_axis(bm, faces)
    if ax:
        out.append((ax[0], ax[1]))
    best = None
    for loop in _closed_edge_loops(faces):
        fit = fit_circle([v.co.copy() for v in loop])
        if fit and fit[3] < 0.01 * fit[2] and (best is None or fit[2] > best[2]):
            best = fit
    if best:
        out.append((best[0], best[1].normalized()))
    verts = ordered({v for f in faces for v in f.verts})
    P = np.array([tuple(v.co) for v in verts])
    c0 = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c0)
    for row in (Vt[2], Vt[0], Vt[1]):
        a = Vector(tuple(row)).normalized()
        d = [(v.co - Vector(tuple(c0))) for v in verts]
        rr = [(x - a * x.dot(a)).length for x in d]
        top = max(rr)
        rim = [v.co for v, r in zip(verts, rr) if r > top * 0.95]
        if len(rim) >= 5:
            fit = _projected_circle(rim, a)
            for _ in range(2):
                if not fit or fit[2] < 0.01 * fit[1]:
                    break
                # the mean of the vertices may be off the axis: take the rim again around the fit
                cc = fit[0]
                rr2 = [((v.co - cc) - a * (v.co - cc).dot(a)).length for v in verts]
                rim = [v.co for v, r in zip(verts, rr2) if r > max(rr2) * 0.98]
                fit = _projected_circle(rim, a) if len(rim) >= 5 else None
            if fit and fit[2] < 0.01 * fit[1]:
                out.append((fit[0], a))
    for c, a in list(out):
        for _ in range(3):
            ref = _refine_axis(verts, c, a.normalized())
            if not ref:
                break
            if ref[1].dot(a.normalized()) > 1 - 1e-12 and (ref[0] - c).length < 1e-9:
                break
            c, a = ref
            out.append(ref)
    return out


def _border_circles(faces):
    """Axis candidates from the selection's border: the vertices it shares with each outside face,
    when they lie on a circle (the rim of a dish cut by a bar, once it has only a few columns)."""
    fset = set(faces)
    groups = {}
    for f in faces:
        for e in f.edges:
            outs = [g for g in e.link_faces if g not in fset]
            if len(outs) == 1:
                groups.setdefault(outs[0], set()).update(e.verts)
    out = []
    for g in ordered(groups):
        vs = ordered(groups[g])
        if len(vs) < 5:
            continue
        fit = fit_circle([v.co.copy() for v in vs])
        if fit and fit[3] < 0.002 * fit[2]:
            out.append((fit[0], fit[1].normalized()))
    return out


def revolution_cut_rings(bm, faces, loose=False):
    """Rings of a part of revolution that a detail cuts. Returns (data, None) or (None, error).
    loose: stop once the axis and the rings are known (no check of how the rings follow)."""
    verts = ordered({v for f in faces for v in f.verts})
    lo = Vector((min(v.co.x for v in verts), min(v.co.y for v in verts), min(v.co.z for v in verts)))
    hi = Vector((max(v.co.x for v in verts), max(v.co.y for v in verts), max(v.co.z for v in verts)))
    extent = (hi - lo).length
    best = None
    cands = _axis_candidates(bm, faces)
    extra = _border_circles(faces) if loose else []
    best_extra = None
    for n_c, (c, a) in enumerate(cands + extra):
        if n_c == len(cands) and best is not None and best[0] >= 0.6 * len(verts):
            break           # the border circles are only a fallback
        a = a.normalized()
        rr = [((v.co - c) - a * (v.co - c).dot(a)).length for v in verts]
        rmax = max(rr)
        if rmax > 10 * extent:
            continue  # circle fitted on a nearly straight row: no real axis

        size = max(rmax, max(abs((v.co - c).dot(a)) for v in verts))
        rings = _ring_clusters(verts, c, a, size, rmax)
        score = sum(len(r["verts"]) for r in rings if len(r["verts"]) >= 4)
        # big rings rather than many small ones: a slightly tilted axis splits each ring
        tight = sum(len(r["verts"]) ** 2 for r in rings if len(r["verts"]) >= 4)
        if n_c >= len(cands):
            # a border circle: the one whose rings make the longest profile (the rim of a dish,
            # not the circle where a bar cuts it)
            rows = sum(1 for r in rings if len(r["verts"]) >= 3)
            if score >= 0.6 * len(verts) and (best_extra is None or (rows, score) > best_extra[0]):
                best_extra = ((rows, score), (score, c, a, rings, rmax, tight))
            continue
        if best is None or (score >= 0.6 * len(verts) and (best[0] < 0.6 * len(verts) or tight > best[5])) \
                or (best[0] < 0.6 * len(verts) and score > best[0]):
            best = (score, c, a, rings, rmax, tight)
    if (best is None or best[0] < 0.6 * len(verts)) and best_extra is not None:
        best = best_extra[1]
    if best is None or best[0] < 0.6 * len(verts):
        return None, "No axis found that this part turns around."
    _, c, a, rings, rmax, _ = best
    ring_of = {v: i for i, r in enumerate(rings) for v in r["verts"]}
    u = a.orthogonal().normalized()
    w = a.cross(u)

    def ang(v):
        d = v.co - c
        return math.atan2(d.dot(w), d.dot(u))
    fset = set(faces)
    rev, caps, detail = [], [], []
    for f in faces:
        ids = {ring_of[v] for v in f.verts}
        if len(ids) == 2:
            rev.append(f)
        elif len(ids) == 1:
            caps.append(f)
        else:
            detail.append(f)
    pins = {v for f in detail for v in f.verts}
    pins |= {v for f in faces for v in f.verts if any(g not in fset for g in v.link_faces)}
    if loose:
        return dict(c=c, a=a, u=u, w=w, rings=rings, ring_of=ring_of, rev=rev, caps=caps,
                    detail=detail, ang=ang, pins=pins), None
    if not rev:
        return None, "No faces turn around an axis here."
    adj = {}
    for f in rev:
        i, j = {ring_of[v] for v in f.verts}
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    # small rings made only of detail vertices (top of a dome cut by a slot): leaves of the profile
    leaves = {i for i, n in adj.items() if len(n) == 1 and all(v in pins for v in rings[i]["verts"])}
    core = {i: {j for j in n if j not in leaves} for i, n in adj.items() if i not in leaves}
    extra = [(next(iter(adj[i])), i) for i in leaves]
    if not core:
        return None, "No faces turn around an axis here."
    if any(len(n) > 2 for n in core.values()):
        return None, "The rings don't follow each other like a profile."
    ends = [i for i, n in core.items() if len(n) <= 1]
    start = ends[0] if ends else next(iter(core))
    order, prev = [start], None
    while True:
        nxt = [j for j in core[order[-1]] if j != prev and j not in order]
        if not nxt:
            break
        prev = order[-1]
        order.append(nxt[0])
    if len(order) != len(core):
        return None, "The selection holds several separate parts."
    return dict(c=c, a=a, u=u, w=w, rings=rings, order=order, ring_of=ring_of, rev=rev, caps=caps,
                detail=detail, ang=ang, closed_profile=not ends and len(order) > 2, pins=pins,
                extra=extra, leaves=leaves), None


def revolution_cut_rebuild(bm, faces, count, profile_segments=None):
    data, err = revolution_cut_rings(bm, faces)
    if err:
        return err
    if count < 3:
        return "Needs at least 3 sides."
    if data["closed_profile"]:
        return "Closed profile (torus): not supported."
    c, a, u, w = data["c"], data["a"], data["u"], data["w"]
    rings, order, ring_of = data["rings"], data["order"], data["ring_of"]
    rev, caps, ang, pins = set(data["rev"]), data["caps"], data["ang"], data["pins"]
    leaves, extra = data["leaves"], data["extra"]
    if any(rings[ri]["r"] < 1e-9 for ri in order):
        return "Pole on the axis: use the part-of-revolution mode instead."
    step = TAU / count
    fset = set(faces)
    rebuilt = set(rev) | {f for f in caps if ring_of[f.verts[0]] in order}

    # phase: same symmetry as the ring with the most free vertices
    big = max((rings[i] for i in order), key=lambda r: (sum(1 for v in r["verts"] if v not in pins), r["r"]))
    pts = [v for v in big["verts"] if v not in pins] or big["verts"]
    delta = min((ang(v) for v in pts), key=abs)
    m0 = max(len(big["verts"]), 3)
    base = delta if abs(delta) <= (TAU / m0) / 4 else delta - math.copysign(TAU / m0 / 2, delta) + step / 2

    def unwrap(t, ref):
        while t < ref - 1e-9:
            t += TAU
        while t >= ref + TAU - 1e-9:
            t -= TAU
        return t

    from mathutils.bvhtree import BVHTree
    vmap, coords, polys = {}, [], []
    rev_list = ordered(rev)
    for f in rev_list:
        poly = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = len(coords)
                coords.append(v.co.copy())
            poly.append(vmap[v])
        polys.append(poly)
    ref_bvh = BVHTree.FromPolygons(coords, polys)
    ref_normals = [f.normal.copy() for f in rev_list]
    mats = [f.material_index for f in faces]
    mat = max(set(mats), key=mats.count)
    smooth = sum(f.smooth for f in faces) * 2 >= len(faces)

    # where the surface between two rings exists: angular spans of the old faces between them
    def spans(ri, rj):
        if (ri, rj) in run_pairs or (rj, ri) in run_pairs:
            key = (ri, rj) if (ri, rj) in run_pairs else (rj, ri)
            return run_spans(run_pairs[key][2])
        return spans_old(ri, rj)

    def spans_old(ri, rj):
        out = []
        for f in rev_list:
            ids = {ring_of[v] for v in f.verts}
            if ids != {ri, rj}:
                continue
            angs_ = sorted(ang(v) % TAU for v in f.verts)
            gaps = [(angs_[(i + 1) % len(angs_)] - angs_[i]) % TAU for i in range(len(angs_))]
            g = max(range(len(gaps)), key=lambda i: gaps[i])
            lo = angs_[(g + 1) % len(angs_)]
            hi = lo + (TAU - gaps[g])
            out.append((lo, hi))
        # merge (cyclic)
        out.sort()
        merged = []
        for lo, hi in out:
            if merged and lo <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        if len(merged) > 1 and merged[-1][1] - TAU >= merged[0][0] - 1e-6:
            merged[0] = [merged[-1][0], max(merged[-1][1], merged[0][1] + TAU)]
            merged.pop()
        return merged

    # ----- optional: change the number of rings of the rounded part of the profile. The walls of
    # the recess are re-cut where each new ring crosses them.
    removed_verts, rewrite, run_pairs, new_ids, newpin_key = [], {}, {}, set(), {}
    detail = data["detail"]
    if profile_segments is not None:
        if profile_segments < 1:
            return "Needs at least 1 segment."
        prof = [Vector((rings[i]["r"], rings[i]["h"], 0.0)) for i in order]
        runs = profile_runs(prof)
        if not runs:
            return "This part has no rounded profile to change."
        new_order, last = [], 0
        removed_ids = set()
        for lo, hi in runs:
            new_order += order[last:lo + 1]
            gone = order[lo + 1:hi]
            removed_ids |= set(gone)
            newp = keep_open(prof[lo:hi + 1], profile_segments)
            ids = []
            for p2 in newp[1:-1]:
                rings.append({"h": p2.y, "r": max(p2.x, 1e-9), "verts": [], "new": True, "newpins": []})
                ids.append(len(rings) - 1)
            new_ids |= set(ids)
            chain = [order[lo]] + ids + [order[hi]]
            for x, y in zip(chain, chain[1:]):
                run_pairs[(x, y)] = (order[lo], order[hi], order[lo:hi + 1])
            new_order += ids
            last = hi
        new_order += order[last:]
        for ri in removed_ids:
            for v in rings[ri]["verts"]:
                if any(f not in fset for f in v.link_faces):
                    return "The rounded part is joined to unselected geometry: select the whole part."
                removed_verts.append(v)
        if any(ri in removed_ids for ri in {ring_of[f.verts[0]] for f in caps}):
            return "A cap sits on a ring that would be removed."
        lo_hi = {}
        for (x, y), (rlo, rhi, seq) in run_pairs.items():
            lo_hi[rlo] = lo_hi.get(rlo, set()) | {rhi}
        # walls of the recess: replace the removed rings' vertices by the crossings with the new rings
        for f in detail:
            loop = [v for v in f.verts]
            if not any(ring_of.get(v) in removed_ids for v in loop):
                continue
            n = len(loop)
            fn = f.normal.copy()
            fp = f.calc_center_median()
            out, i = [], 0
            # start on a vertex that isn't removed
            start = next((k for k in range(n) if ring_of.get(loop[k]) not in removed_ids), None)
            if start is None:
                return "A face of the recess lies only on rings that would be removed."
            loop = loop[start:] + loop[:start]
            k = 0
            while k < n:
                v = loop[k]
                if ring_of.get(v) not in removed_ids:
                    out.append(v)
                    k += 1
                    continue
                j = k
                while j < n and ring_of.get(loop[j]) in removed_ids:
                    j += 1
                before, after = loop[k - 1], loop[j % n]
                rb, ra = ring_of.get(before), ring_of.get(after)
                seq = None
                for (rlo, rhi, sq) in run_pairs.values():
                    if {rb, ra} == {rlo, rhi}:
                        seq = (rlo, rhi)
                        break
                if seq is None:
                    return "The recess meets the rounded part in a way that can't be rebuilt yet."
                chain_ids = [rid for (x, y), (rlo, rhi, sq) in run_pairs.items() if (rlo, rhi) == seq for rid in (x, y)]
                chain_ids = list(dict.fromkeys(chain_ids))
                inner = [rid for rid in chain_ids if rid in new_ids]
                inner.sort(key=lambda rid: rings[rid]["h"] if True else 0)
                # order of the chain along the profile: from rlo to rhi
                prof_order = [rid for rid in new_order if rid in inner]
                if rb == seq[1]:
                    prof_order = prof_order[::-1]
                t_ref = [ang(x) for x in loop[k:j]]
                tr = math.atan2(sum(math.sin(t) for t in t_ref), sum(math.cos(t) for t in t_ref))
                for rid in prof_order:
                    ring = rings[rid]
                    A = ring["r"] * u.dot(fn)
                    B = ring["r"] * w.dot(fn)
                    C = (c + a * ring["h"] - fp).dot(fn)
                    R = math.hypot(A, B)
                    if R < 1e-12 or abs(C) > R:
                        return "A new ring would miss the recess wall: try fewer segments."
                    phi = math.atan2(B, A)
                    dlt = math.acos(max(-1.0, min(1.0, -C / R)))
                    cands = [phi + dlt, phi - dlt]
                    t = min(cands, key=lambda x: abs(((x - tr + math.pi) % TAU) - math.pi))
                    key = (rid, round(math.cos(t), 5), round(math.sin(t), 5))
                    if key not in newpin_key:
                        pos = c + a * ring["h"] + ring["r"] * (math.cos(t) * u + math.sin(t) * w)
                        newpin_key[key] = (t, pos)
                        ring["newpins"].append((t, key, pos))
                    out.append(("NEW", key))
                k = j
            rewrite[f] = out
        pins = {v for v in pins if ring_of.get(v) not in removed_ids}
        order = new_order

        # where the surface is, for the new rings: angular spans of the old faces of the run
        def run_spans(seq):
            out = []
            for x, y in zip(seq, seq[1:]):
                out += spans_old(x, y)
            return _merge_spans(out)
    else:
        def run_spans(seq):
            return []

    # final vertex list of each ring: kept vertices (pins) + new regular columns in the free runs
    final = {}
    for ri in order:
        ring = rings[ri]
        if ring.get("new"):
            def pos(t, ring=ring):
                return c + a * ring["h"] + ring["r"] * (math.cos(t) * u + math.sin(t) * w)
            seq = next(sq for (x, y), (rlo, rhi, sq) in run_pairs.items() if ri in (x, y))
            sp = run_spans(seq)
            np_ = sorted(ring["newpins"], key=lambda x: x[0] % TAU)
            items = [(t, "pin", ("NEW", key)) for t, key, p in np_]
            if not np_:
                items = [(base + k * step, k, pos(base + k * step)) for k in range(count)]
            else:
                for idx in range(len(np_)):
                    t0 = np_[idx][0]
                    t1 = unwrap(np_[(idx + 1) % len(np_)][0], t0 + 1e-9) if len(np_) > 1 else t0 + TAU
                    mid = (t0 + t1) / 2
                    if not any(unwrap(mid, lo2) <= hi2 for lo2, hi2 in sp):
                        continue  # the recess is there
                    k_lo = int(math.ceil((t0 + 0.05 * step - base) / step))
                    k_hi = int(math.floor((t1 - 0.05 * step - base) / step))
                    for kk in range(k_lo, k_hi + 1):
                        tt = base + kk * step
                        items.append((tt, kk % count, pos(tt)))
                taken, fixed = {}, []
                for t, kind, v in items:
                    if kind != "pin":
                        continue
                    kk = int(round((t - base) / step)) % count
                    if kk in taken:
                        fixed.append((t, None, v))
                    else:
                        taken[kk] = True
                        fixed.append((t, kk, v))
                items = fixed + [it for it in items if it[1] != "pin" and it[1] not in taken]
                pin_t = [it[0] for it in fixed]
                items = [it for it in items if it in fixed or
                         all(abs(((it[0] - pt + math.pi) % TAU) - math.pi) > 0.3 * step for pt in pin_t)]
            items.sort(key=lambda x: x[0] % TAU)
            final[ri] = items
            continue
        vs = sorted(ring["verts"], key=ang)
        n = len(vs)
        free = []
        for k in range(n):
            e = edge_between(vs[k], vs[(k + 1) % n])
            free.append(bool(e) and all(f in rebuilt for f in e.link_faces))
        keep = [v for v in vs if v in pins]
        items = [(ang(v), "pin", v) for v in keep]

        def pos(t, ring=ring):
            return c + a * ring["h"] + ring["r"] * (math.cos(t) * u + math.sin(t) * w)
        if not keep:
            if not all(free):
                return "A ring of this part is cut without any detail: it can't be rebuilt safely."
            items = [(base + k * step, k, pos(base + k * step)) for k in range(count)]
        else:
            # free runs go from one kept vertex to the next one through free edges only
            for i, v in enumerate(vs):
                if v not in pins:
                    continue
                j, ok = i, True
                while True:
                    if not free[j % n]:
                        ok = False
                        break
                    j += 1
                    if vs[j % n] in pins:
                        break
                if not ok:
                    continue
                t0 = ang(v)
                t1 = unwrap(ang(vs[j % n]), t0 + 1e-9) if vs[j % n] is not v else t0 + TAU
                k_lo = int(math.ceil((t0 + 0.05 * step - base) / step))
                k_hi = int(math.floor((t1 - 0.05 * step - base) / step))
                for k in range(k_lo, k_hi + 1):
                    t = base + k * step
                    items.append((t, k % count, pos(t)))
            # each kept vertex takes the place of the nearest column, so that neighbouring
            # rings stay aligned (quads) right up to the detail
            taken = {}
            fixed = []
            for t, kind, v in items:
                if kind != "pin":
                    continue
                k = int(round((t - base) / step)) % count
                if k in taken:
                    fixed.append((t, None, v))
                else:
                    taken[k] = True
                    fixed.append((t, k, v))
            items = fixed + [it for it in items if it[1] != "pin" and it[1] not in taken]
            # a column squeezed right next to a kept vertex makes a sliver: drop it
            pin_t = [it[0] for it in fixed]
            items = [it for it in items if it in fixed or
                     all(abs(((it[0] - pt + math.pi) % TAU) - math.pi) > 0.3 * step for pt in pin_t)]
        items = [(t, ("pin" if kind is None else kind), x) if False else (t, kind, x) for t, kind, x in items]
        items.sort(key=lambda x: x[0] % TAU)
        final[ri] = items

    pairs_rings = [(order[i], order[i + 1]) for i in range(len(order) - 1)] + extra
    span_list = [spans(ri, rj) for ri, rj in pairs_rings]
    for li in leaves:
        final[li] = sorted(((ang(v), None, v) for v in rings[li]["verts"]), key=lambda x: x[0] % TAU)
    cap_rings = {ring_of[f.verts[0]] for f in caps if ring_of[f.verts[0]] in order}
    cap_normal = {}
    for f in caps:
        ri = ring_of[f.verts[0]]
        if ri in order:
            cap_normal[ri] = cap_normal.get(ri, Vector()) + f.normal * f.calc_area()

    def do():
        old = {v for ri in order for v in rings[ri]["verts"]}
        saved = {f: (f.material_index, f.smooth) for f in rewrite}
        bmesh.ops.delete(bm, geom=list(rebuilt) + list(rewrite.keys()), context="FACES_ONLY")
        gone = [v for v in old if v.is_valid and v not in pins] + [v for v in removed_verts if v.is_valid]
        if gone:
            bmesh.ops.delete(bm, geom=ordered(set(gone)), context="VERTS")
        newv = {}

        def resolve(x):
            if isinstance(x, bmesh.types.BMVert):
                return x
            if isinstance(x, tuple) and len(x) == 2 and x[0] == "NEW":
                if x[1] not in newv:
                    newv[x[1]] = bm.verts.new(newpin_key[x[1]][1])
                return newv[x[1]]
            return bm.verts.new(x)
        made = {}
        for ri in list(order) + list(leaves):
            lst = []
            for t, kind, x in final[ri]:
                lst.append((t % TAU, kind, resolve(x)))
            made[ri] = lst
        rewritten = []
        for f, vl in rewrite.items():
            vs = list(dict.fromkeys(resolve(x) for x in vl))
            if len(vs) >= 3:
                nf = bm.faces.new(vs)
                nf.material_index, nf.smooth = saved[f]
                rewritten.append(nf)
        new_faces = []
        for (ri, rj), sp in zip(pairs_rings, span_list):
            A, B = made[ri], made[rj]
            for lo, hi in sp:
                full = hi - lo >= TAU - 1e-6

                def inside(lst, lo=lo, hi=hi, full=full):
                    if full:
                        return sorted(lst, key=lambda x: x[0])
                    out = []
                    for t, kind, v in lst:
                        tt = unwrap(t, lo - 1e-6)
                        if tt <= hi + 1e-6:
                            out.append((tt, kind, v))
                    return sorted(out, key=lambda x: x[0])
                a_l, b_l = inside(A), inside(B)
                if not a_l or not b_l:
                    return False
                for vs in _stitch(a_l, b_l, full, step):
                    vs = list(dict.fromkeys(vs))
                    if len(vs) >= 3:
                        new_faces.append(bm.faces.new(vs))
        cap_faces = {}
        for ri in cap_rings:
            vs = [v for t, kind, v in sorted(made[ri], key=lambda x: x[0])]
            if len(vs) >= 3:
                cf = bm.faces.new(vs)
                cap_faces[cf] = cap_normal[ri]
                new_faces.append(cf)
        bm.normal_update()
        flip = []
        for f in new_faces:
            if f in cap_faces:
                if f.normal.dot(cap_faces[f]) < 0:
                    flip.append(f)
                continue
            hit = ref_bvh.find_nearest(f.calc_center_median())
            if hit[0] is not None and f.normal.dot(ref_normals[hit[2]]) < 0:
                flip.append(f)
        if flip:
            bmesh.ops.reverse_faces(bm, faces=flip)
        for f in new_faces:
            f.material_index = mat
            f.smooth = smooth
            f.select = True
        return True

    ok, _ = guarded(bm, do)
    if not ok:
        return "This part can't be rebuilt safely with this number of sides (try another one)."
    return None


def _merge_spans(out):
    out = sorted(out)
    merged = []
    for lo, hi in out:
        if merged and lo <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if len(merged) > 1 and merged[-1][1] - TAU >= merged[0][0] - 1e-6:
        merged[0] = [merged[-1][0], max(merged[-1][1], merged[0][1] + TAU)]
        merged.pop()
    return merged


def _stitch(a_l, b_l, cyclic, step):
    """Faces between two rows of vertices sorted by angle: quads where both rows have a vertex
    at about the same angle, triangles elsewhere."""
    na, nb = len(a_l), len(b_l)
    if cyclic:
        # start at the closest pair
        j0 = min(range(nb), key=lambda j: abs(((b_l[j][0] - a_l[0][0] + math.pi) % TAU) - math.pi))
        b_l = b_l[j0:] + [(t + TAU, k, v) for t, k, v in b_l[:j0]]
        a_l = a_l + [(a_l[0][0] + TAU, a_l[0][1], a_l[0][2])]
        b_l = b_l + [(b_l[0][0] + TAU, b_l[0][1], b_l[0][2])]
        # keep angles increasing from the start
        base = a_l[0][0]
        b_l = [((t - base + math.pi) % TAU + base - math.pi if i == 0 else t, k, v) for i, (t, k, v) in enumerate(b_l)]
    faces = []
    i = j = 0
    na, nb = len(a_l) - 1, len(b_l) - 1
    while i < na or j < nb:
        ta = a_l[i + 1][0] if i < na else float("inf")
        tb = b_l[j + 1][0] if j < nb else float("inf")
        same_col = a_l[i + 1][1] is not None and a_l[i + 1][1] == b_l[j + 1][1] if (i < na and j < nb) else False
        if i < na and j < nb and (same_col or abs(ta - tb) < 0.35 * step):
            faces.append([a_l[i][2], a_l[i + 1][2], b_l[j + 1][2], b_l[j][2]])
            i, j = i + 1, j + 1
        elif ta <= tb:
            faces.append([a_l[i][2], a_l[i + 1][2], b_l[j][2]])
            i += 1
        else:
            faces.append([a_l[i][2], b_l[j + 1][2], b_l[j][2]])
            j += 1
    return faces


def revolution_cut_profile_count(bm, faces):
    data, err = revolution_cut_rings(bm, faces)
    if err:
        return None
    prof = [Vector((data["rings"][i]["r"], data["rings"][i]["h"], 0.0)) for i in data["order"]]
    return max((hi - lo for lo, hi in profile_runs(prof)), default=None)


def revolution_cut_count(bm, faces):
    data, err = revolution_cut_rings(bm, faces)
    if err:
        return None
    sizes = [len(data["rings"][ri]["verts"]) for ri in data["order"]]
    if not sizes:
        return None
    return max(set(sizes), key=lambda n: (sizes.count(n), n))
