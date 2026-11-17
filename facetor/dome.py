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

"""Domes cut by unselected geometry (a bar crossing a dish, its walls not selected): the
pieces on each side are rebuilt as one turned grid, the cut edge stays exactly as it is."""

import math
import bmesh
from mathutils import Vector

from .core import TAU, _face_components
from .grid import _resize_chain
from .recess import _circle2d, _Curve
from .revolution import profile_runs, revolution_cut_rings
from .safety import guarded
from .strips import selection_boundary


# ---------------------------------------------------------------------------
# Each selected piece is bounded by an arc of one ring of the dome (its rim, joined to the
# rest of the mesh) and by a cut chain (the edge along the unselected walls of a bar). The new
# grid has regular columns around the axis and rings along the profile; every ring ends on a
# vertex of the cut chain, and the cut chain vertices are joined to the nearest ones.
# ---------------------------------------------------------------------------

def _unwrap(ts):
    out = [ts[0]]
    for t in ts[1:]:
        out.append(out[-1] + (t - out[-1] + math.pi) % TAU - math.pi)
    return out


def _profile(prof, beyond, has_pole=False):
    """Profile curve of the dome in the (r, h) plane: the rows from the border row inwards,
    then on to the axis (circle through the rounded part) when it can be found.
    has_pole: the last row is the pole itself (the dome's top is selected).
    Returns (curve, start, arc, pole): arc = (centre, radius, a0, da) of the rounded part that
    goes from prof[start] to the pole, or None."""
    pts_ = [] if has_pole else sorted(beyond, key=lambda q: -q.x)
    runs = profile_runs([Vector((q.x, q.y, 0.0)) for q in prof]) if len(prof) >= 3 else []
    start = next((lo for lo, hi in runs if hi == len(prof) - 1), max(len(prof) - 2, 0))
    if has_pole:
        start = min(start, max(len(prof) - 3, 0))   # a circle needs three rows
    circ = None
    for fit_pts in (prof[start:] + pts_, prof[start:]):
        if len(fit_pts) >= 3:
            circ = _circle2d(fit_pts)
            if circ:
                break
    while circ and start > 0:
        cx, cy, r = circ
        q = prof[start - 1]
        if abs(math.hypot(q.x - cx, q.y - cy) - r) > 0.004 * r:
            break
        start -= 1
        circ = _circle2d(prof[start:] + pts_) or circ
    if circ:
        cx, cy, r = circ
        if r * r < cx * cx:
            circ = None
    if not circ:
        dense = list(prof) + [q for q in pts_ if (q - prof[-1]).length > 1e-9]
        return _Curve(dense), start, None, None
    cx, cy, r = circ
    if has_pole:
        pole = Vector((0.0, prof[-1].y))
    else:
        k = math.sqrt(r * r - cx * cx)
        pole = Vector((0.0, min((cy + k, cy - k), key=lambda y: abs(y - prof[-1].y))))
    ctr = Vector((cx, cy))
    a0 = math.atan2(prof[start].y - cy, prof[start].x - cx)
    da = (math.atan2(pole.y - cy, -cx) - a0 + math.pi) % TAU - math.pi
    arc = [ctr + r * Vector((math.cos(a0 + da * i / 96), math.sin(a0 + da * i / 96))) for i in range(97)]
    arc[0], arc[-1] = prof[start].copy(), pole.copy()
    return _Curve(list(prof[:start]) + arc), start, (ctr, r, a0, da), pole


def dome_cut_setup(bm, faces):
    """Analysis of a dome cut by unselected geometry. Returns (data, None) or (None, error)."""
    if len(faces) < 4:
        return None, "Not a dome."
    fset = set(faces)
    comps = _face_components(faces)
    data, err = revolution_cut_rings(bm, faces, loose=True)
    if err:
        return None, err
    c, a, u, w = data["c"], data["a"], data["u"], data["w"]
    rings, ring_of, ang = data["rings"], data["ring_of"], data["ang"]

    def rh(v):
        d = v.co - c
        h = d.dot(a)
        return Vector(((d - a * h).length, h))

    # ----- pieces: one border arc on a ring + one cut chain each
    pieces = []
    for comp in comps:
        loops = selection_boundary(comp)
        if not loops or len(loops) != 1:
            return None, "Each piece of the dome must have one border."
        loop = loops[0]
        for _ in range(2):
            n = len(loop)
            ring_edge = [ring_of[loop[i]] == ring_of[loop[(i + 1) % n]] for i in range(n)]
            changes = [i for i in range(n) if ring_edge[i] and not ring_edge[i - 1]]
            if len(changes) != 1 or sum(ring_edge) == n:
                return None, "This piece isn't bounded by a ring and one cut edge."
            loop = loop[changes[0]:] + loop[:changes[0]]
            k = sum(ring_edge)
            th = _unwrap([ang(v) for v in loop])
            wind = th[-1] + (ang(loop[0]) - th[-1] + math.pi) % TAU - math.pi - th[0]
            # around: the piece holds the axis (a dish with its bottom, its rim cut on one side)
            around = abs(abs(wind) - TAU) < 1e-3
            if not around and abs(wind) > 1e-3:
                return None, "This piece isn't bounded by a ring and one cut edge."
            if th[k] > th[0]:
                break
            loop.reverse()
        else:
            return None, "This piece isn't bounded by a ring and one cut edge."
        B = loop[:k + 1]
        thB = th[:k + 1]
        if around:
            # the cut edge goes on around the axis from the end of the border to its start
            if wind < 0:
                return None, "The cut edge turns back around the axis."
            C = loop[k:] + [loop[0]]
            thC = th[k:] + [th[0] + wind]
        else:
            C = [loop[0]] + loop[k:][::-1]
            thC = [th[0]] + th[k:][::-1]
        if any(t1 < t0 - 1e-4 for t0, t1 in zip(thB, thB[1:])) or \
                any(t1 < t0 - 1e-4 for t0, t1 in zip(thC, thC[1:])):
            return None, "The cut edge turns back around the axis."
        pieces.append(dict(faces=comp, B=B, thB=thB, C=C, thC=thC, ring=ring_of[B[1]], around=around))
    if any(p["around"] for p in pieces) and len(pieces) > 1:
        return None, "Only one piece can go around the axis."
    if len({p["ring"] for p in pieces}) != 1:
        return None, "The pieces don't share one border ring."
    border = pieces[0]["ring"]
    cut = {v for p in pieces for v in p["C"]}
    if any(len(e.link_faces) != 2 or not any(g not in fset for g in e.link_faces)
           for p in pieces for e in _chain_edges(p["C"])):
        return None, "The cut edge isn't joined to unselected faces."

    # ----- rows: rings holding vertices away from the cut, in order along the profile
    row_ids = {ring_of[v] for f in faces for v in f.verts if v not in cut} | {border}
    adj = {i: set() for i in row_ids}
    for f in faces:
        ids = {ring_of[v] for v in f.verts} & row_ids
        if len(ids) > 2:
            return None, "A face spans more than two rings."
        if len(ids) == 2:
            i, j = ids
            adj[i].add(j)
            adj[j].add(i)
    if any(len(n) > 2 for n in adj.values()) or len(adj[border]) > 1:
        return None, "The rings don't follow each other like a profile."
    order, prev = [border], None
    while True:
        nxt = [j for j in adj[order[-1]] if j != prev and j not in order]
        if not nxt:
            break
        prev = order[-1]
        order.append(nxt[0])
    if len(order) != len(row_ids):
        return None, "The rings don't follow each other like a profile."

    def ring_rh(i):
        vs = [v for v in rings[i]["verts"] if v not in cut] or rings[i]["verts"]
        q = sum((rh(v) for v in vs), Vector((0.0, 0.0))) / len(vs)
        return q
    prof = [ring_rh(i) for i in order]
    on_axis = [q.x < 1e-6 * max(p.x for p in prof) for q in prof]
    has_pole = on_axis[-1] and len(rings[order[-1]]["verts"]) == 1
    if any(on_axis[:-1]) or (on_axis[-1] and not has_pole):
        return None, "Pole on the axis."
    if has_pole != pieces[0]["around"]:
        return None, "The dome's top must be selected with the part going around it."
    if has_pole:
        prof[-1] = Vector((0.0, prof[-1].y))
    beyond = []
    if has_pole:
        beyond = None
    elif len(prof) >= 2:
        d = prof[-1] - prof[-2]
        for v in cut:
            q = rh(v)
            if (q - prof[-1]).dot(d) > 1e-6 * prof[0].x:
                beyond.append(q)
    else:
        beyond = [rh(v) for v in cut if rh(v).x < prof[0].x * (1 - 1e-6)]
    if beyond is not None and not beyond:
        return None, "Nothing cuts the dome past its last ring."
    curve, start, arc, pole = _profile(prof, beyond or [], has_pole)
    row_s = [curve.project(q)[1] for q in prof]
    if any(s1 <= s0 for s0, s1 in zip(row_s, row_s[1:])):
        return None, "The rings of the profile overlap."
    total = curve.s[-1]
    for p in pieces:
        p["sC"] = sC = [curve.project(rh(v))[1] for v in p["C"]]
        im = max(range(len(sC)), key=lambda i: sC[i])
        tol = 1e-4 * total
        if any(s1 < s0 - tol for s0, s1 in zip(sC[:im], sC[1:im + 1])) or \
                any(s1 > s0 + tol for s0, s1 in zip(sC[im:], sC[im + 1:])):
            return None, "The cut edge goes in and out of the dome more than once."
        p["imax"] = im

    # ----- number of sides now: the spacing of the vertices along the rows
    # (inside each piece: the gap across the bar isn't a column)
    gaps = []
    row_set = set(order)
    for p in pieces:
        t0 = p["thB"][0]
        pverts = {v for f in p["faces"] for v in f.verts if v not in cut}
        for i in order:
            ts = sorted((ang(v) - t0) % TAU for v in pverts if ring_of[v] == i)
            gaps += [t2 - t1 for t1, t2 in zip(ts, ts[1:]) if t2 - t1 > 1e-6]
    if not gaps:
        # one column per row at most: from the border row and its ends
        gaps = [(p["thB"][-1] - p["thB"][0]) / (len(p["B"]) - 1) for p in pieces]
    gaps.sort()
    sides = max(3, int(round(TAU / gaps[len(gaps) // 2])))
    if not any(p["around"] for p in pieces):
        # few sides (see LOW_SIDES): each piece holds its share of them, evenly spread
        rim = sum(len(p["B"]) - 1 for p in pieces)
        if rim < LOW_SIDES:
            sides = rim
    # phase of each piece: its middle (its axis of symmetry), and the column now nearest to it
    # (each piece may have its own: two halves of an odd number of sides)
    for p in pieces:
        p["mid"] = mid = (p["thB"][0] + p["thB"][-1]) / 2
        free = [ang(v) for v in {v for f in p["faces"] for v in f.verts}
                if v not in cut and ring_of[v] in row_set]
        p["phase_now"] = mid + min(((t - mid + math.pi) % TAU - math.pi for t in free), key=abs, default=0.0)
    profile_count = None
    if arc is not None:
        segs = len(prof) - 1 - start
        if has_pole:
            profile_count = max(segs, 1)
        elif segs <= 0:
            profile_count = 1
        else:
            mean = (row_s[-1] - row_s[start]) / segs
            profile_count = segs + max(1, int(round((total - row_s[-1]) / max(mean, 1e-12))))
    return dict(c=c, a=a, u=u, w=w, ang=ang, rh=rh, pieces=pieces, cut=cut, order=order,
                rings=rings, prof=prof, row_s=row_s, curve=curve, start=start, arc=arc, pole=pole,
                sides=sides, profile_count=profile_count,
                border=border, fset=fset, has_pole=has_pole), None


LOW_SIDES = 8


def _share(total, weights, least):
    """`total` split in proportion to the weights (largest remainders), each part >= its least."""
    s = sum(weights) or 1.0
    want = [total * w / s for w in weights]
    out = [max(m, int(math.floor(x))) for x, m in zip(want, least)]
    for i in sorted(range(len(out)), key=lambda i: (out[i] - want[i], i)):
        if sum(out) >= total:
            break
        out[i] += 1
    while sum(out) > total:
        i = max((i for i in range(len(out)) if out[i] > least[i]), key=lambda i: (out[i] - want[i], -i))
        out[i] -= 1
    return out


def _chain_edges(chain):
    out = []
    for v0, v1 in zip(chain, chain[1:]):
        e = next((e for e in v0.link_edges if e.other_vert(v0) is v1), None)
        if e is not None:
            out.append(e)
    return out


def dome_cut_counts(bm, faces):
    """(sides, rings of the profile) of a dome cut by unselected geometry, or None."""
    data, err = dome_cut_setup(bm, faces)
    if err:
        return None
    return data["sides"], data["profile_count"]


def dome_cut_rebuild(bm, faces, count, profile_segments=None, data=None):
    if data is None:
        data, err = dome_cut_setup(bm, faces)
        if err:
            return err
    if count < 3:
        return "Needs at least 3 sides."
    c, a, u, w, ang = data["c"], data["a"], data["u"], data["w"], data["ang"]
    curve, prof, row_s = data["curve"], data["prof"], data["row_s"]
    step = TAU / count

    # ----- rows of the new grid: (r, h, s)
    rows = [(q.x, q.y, s) for q, s in zip(prof, row_s)]
    if profile_segments is not None:
        if profile_segments < 1:
            return "Needs at least 1 segment."
        if data["arc"] is None:
            return "This dome has no rounded profile to change."
        start = data["start"]
        ctr, rad, a0, da = data["arc"]
        rows = rows[:start + 1]
        for i in range(1, profile_segments):
            q = ctr + rad * Vector((math.cos(a0 + da * i / profile_segments),
                                    math.sin(a0 + da * i / profile_segments)))
            rows.append((q.x, q.y, curve.project(q)[1]))
        if data["has_pole"]:
            rows.append((0.0, prof[-1].y, row_s[-1]))

    def pos(t, r, h):
        return c + a * h + r * (math.cos(t) * u + math.sin(t) * w)

    def columns(t0, t1, base):
        if base is None:
            return []
        if isinstance(base, tuple):
            # few sides: the piece's own columns, evenly spread between its two ends
            angs, st = base
            return [(t, k) for k, t in enumerate(angs) if t0 + 0.3 * st <= t <= t1 - 0.3 * st]
        k_lo = int(math.ceil((t0 + 0.3 * step - base) / step))
        k_hi = int(math.floor((t1 - 0.3 * step - base) / step))
        return [(base + k * step, k % count) for k in range(k_lo, k_hi + 1)]

    # few sides: under LOW_SIDES the columns of one grid around the axis would leave each half
    # of a dish crossed by a bar with one column whatever the count (4..7 looked the same). The
    # pieces share the count instead, in proportion to their spans; the rim's vertices held by
    # outside geometry (pins) always stay and count as columns
    low = None
    if count < LOW_SIDES and not any(p["around"] for p in data["pieces"]):
        pcs = data["pieces"]
        pin_th = []
        for p in pcs:
            bset = set(p["B"])
            pins = [0] + [i for i in range(1, len(p["B"]) - 1) if _anchor(p["B"][i], bset, data["fset"])]                 + [len(p["B"]) - 1]
            pin_th.append([p["thB"][i] for i in pins])
        least = sum(len(t) - 1 for t in pin_th)
        if count < least:
            return "Too few sides for this dome (%d at least)." % least
        n_p = _share(count, [t[-1] - t[0] for t in pin_th], [len(t) - 1 for t in pin_th])
        low = {}
        for p, n, th in zip(pcs, n_p, pin_th):
            arcs = [t1 - t0 for t0, t1 in zip(th, th[1:])]
            n_a = _share(n, arcs, [1] * len(arcs))
            angs = []
            for (t0, t1), k in zip(zip(th, th[1:]), n_a):
                angs += [t0 + (t1 - t0) * j / k for j in range(1 if t0 == th[0] else 0, k)]
            low[id(p)] = (angs, min(a / k for a, k in zip(arcs, n_a)))

    # ----- per piece: where each row meets the cut edge, and the row's vertices
    plans = []
    for p in data["pieces"]:
        C, thC, sC, im = p["C"], p["thC"], p["sC"], p["imax"]
        B, thB = p["B"], p["thB"]
        base = p["phase_now"] if count == data["sides"] else p["mid"]
        if low is not None:
            base = low[id(p)]
        # border row: arcs between kept vertices (ends, and vertices joined to more outside geometry)
        bset = set(B)
        pins = [0] + [i for i in range(1, len(B) - 1) if _anchor(B[i], bset, data["fset"])] + [len(B) - 1]
        arcs = []
        for i0, i1 in zip(pins, pins[1:]):
            arcs.append((B[i0:i1 + 1], columns(thB[i0], thB[i1], base)))
        prow = [dict(border=True, arcs=arcs, pins=[(thB[i], B[i]) for i in pins], i_in=0, i_out=len(C) - 1)]
        prev_in, prev_out = 0, len(C) - 1
        around = p["around"]
        for r, h, s in rows[1:]:
            if s >= sC[im]:
                if not around:
                    break
                # past the cut: whole rings, then the pole
                if r < 1e-9:
                    prow.append(dict(kind="pole", items=[(0.0, None, pos(0.0, 0.0, h))]))
                    break
                items = [(base + k * step, k, pos(base + k * step, r, h)) for k in range(count)]
                prow.append(dict(kind="full", items=items))
                continue
            if around and prow[-1].get("kind") in ("full", "pole"):
                return "The cut edge goes in and out of the dome more than once."
            i_in = min(range(prev_in, im + 1), key=lambda i: abs(sC[i] - s))
            i_out = min(range(im, prev_out + 1), key=lambda i: abs(sC[i] - s))
            if i_in >= i_out:
                if around:
                    continue
                break
            if around:
                # the row goes the long way round, from the far side of the cut to its near side
                t0, t1 = thC[i_out] - TAU, thC[i_in]
                cols = columns(t0, t1, base)
                items = [(t0, None, C[i_out])] + [(t, k, pos(t, r, h)) for t, k in cols] + \
                        [(t1, None, C[i_in])]
            else:
                cols = columns(thC[i_in], thC[i_out], base)
                if not cols:
                    break
                items = [(thC[i_in], None, C[i_in])] + [(t, k, pos(t, r, h)) for t, k in cols] + \
                        [(thC[i_out], None, C[i_out])]
            prow.append(dict(border=False, kind="part", items=items, i_in=i_in, i_out=i_out))
            prev_in, prev_out = i_in, i_out
        if around and prow[-1].get("kind") != "pole":
            return "The dome's top can't be rebuilt with this number of rings."
        plans.append((p, prow))

    # ----- orientation and look of the old faces
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
    keep = set(data["cut"]) | {v for p in data["pieces"] for v in p["B"]}
    old_verts = {v for f in faces for v in f.verts}
    old_edges = {e for f in faces for e in f.edges}

    def do():
        bmesh.ops.delete(bm, geom=list(faces), context="FACES_ONLY")
        gone = [v for v in old_verts if v not in keep and v.is_valid]
        if gone:
            bmesh.ops.delete(bm, geom=gone, context="VERTS")
        wire = [e for e in old_edges if e.is_valid and not e.link_faces]
        if wire:
            bmesh.ops.delete(bm, geom=wire, context="EDGES_ONLY")
        new_faces = []
        for p, prow in plans:
            C, thC = p["C"], p["thC"]
            # border row: each arc resized to its new number of columns, the vertices put on them
            items0 = []
            for chain, cols in prow[0]["arcs"]:
                if not all(v.is_valid for v in chain):
                    return False
                new = _resize_chain(bm, chain, False, len(cols) + 1, None)
                if len(new) != len(cols) + 2:
                    return False
                for v, (t, k) in zip(new[1:-1], cols):
                    v.co = pos(t, data["prof"][0].x, data["prof"][0].y)
                if not items0:
                    items0.append((ang_un(new[0], p["thB"][0], ang), None, new[0]))
                for v, (t, k) in zip(new[1:-1], cols):
                    items0.append((t, k, v))
                items0.append((ang_un(new[-1], items0[-1][0], ang), None, new[-1]))
            prow[0]["items"] = items0
            made = []
            for row in prow:
                made.append([(t, k, x if isinstance(x, bmesh.types.BMVert) else bm.verts.new(x))
                             for t, k, x in row["items"]])
            if p["around"]:
                for vs in _around_faces(prow, made, C, thC):
                    vs = list(dict.fromkeys(vs))
                    if len(vs) >= 3:
                        new_faces.append(bm.faces.new(vs))
                continue
            for j in range(len(prow)):
                lower = made[j]
                if j + 1 < len(prow):
                    up_row = made[j + 1]
                    i0, i1 = prow[j]["i_in"], prow[j + 1]["i_in"]
                    o0, o1 = prow[j + 1]["i_out"], prow[j]["i_out"]
                    upper = [(thC[x], None, C[x]) for x in range(i0, i1)] + up_row + \
                            [(thC[x], None, C[x]) for x in range(o0 + 1, o1 + 1)]
                else:
                    i0, o1 = prow[j]["i_in"], prow[j]["i_out"]
                    upper = [(thC[x], None, C[x]) for x in range(i0, o1 + 1)]
                if lower[0][2] is not upper[0][2] or lower[-1][2] is not upper[-1][2]:
                    return False
                for vs in _stitch_ngons(lower, upper):
                    vs = list(dict.fromkeys(vs))
                    if len(vs) >= 3:
                        new_faces.append(bm.faces.new(vs))
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
        return "This dome can't be rebuilt safely with this number of segments (try another one)."
    return None


def _around_faces(prow, made, C, thC):
    """Faces of a piece that goes around the axis: border row, rows cut on one side (each one
    longer than the one before, the cut edge receding), whole rings past the cut, the pole."""
    TAUv = TAU
    out = []
    for j in range(len(prow) - 1):
        cur, nxt = prow[j], prow[j + 1]
        kind = nxt.get("kind")
        if kind == "part":
            # the inner row is the longer one: the outer row plus the cut edge at both ends
            o1, o0 = nxt["i_out"], cur["i_out"]
            i0, i1 = cur["i_in"], nxt["i_in"]
            path = [(thC[x] - TAUv, None, C[x]) for x in range(o1, o0)] + made[j] + \
                   [(thC[x], None, C[x]) for x in range(i0 + 1, i1 + 1)]
            if made[j + 1][0][2] is not path[0][2] or made[j + 1][-1][2] is not path[-1][2]:
                raise ValueError("rows don't meet the cut edge")
            out += _stitch_ngons(made[j + 1], path)
            continue
        # closed loop of the outer row: a ring, or a cut row closed by the tip of the cut edge
        if cur.get("kind") in ("full",):
            loop = made[j]
        else:
            loop = made[j] + [(thC[x], None, C[x]) for x in range(cur["i_in"] + 1, cur["i_out"])]
        if kind == "pole":
            pole = made[j + 1][0][2]
            vs = [x[2] for x in loop]
            out += [[vs[i], vs[(i + 1) % len(vs)], pole] for i in range(len(vs))]
        else:
            out += _stitch_cyclic(made[j + 1], loop)
    return out


def _stitch_cyclic(lower, upper):
    """Faces between two closed rows of (angle, column, vertex), each going once around."""
    t0 = lower[0][0]
    j0 = min(range(len(upper)), key=lambda j: abs((upper[j][0] - t0 + math.pi) % TAU - math.pi))
    up = upper[j0:] + upper[:j0]
    lo = [(t0 + (t - t0) % TAU, k, v) for t, k, v in lower]
    u0 = t0 + (up[0][0] - t0 + math.pi) % TAU - math.pi
    up = [(u0 + (t - up[0][0]) % TAU, k, v) for t, k, v in up]
    lo.append((lo[0][0] + TAU, lo[0][1], lo[0][2]))
    up.append((up[0][0] + TAU, up[0][1], up[0][2]))
    return _stitch_ngons(lo, up, shared=False)


def _stitch_ngons(lower, upper, shared=True):
    """Faces between two rows of (angle, column, vertex) that start and end on the same
    vertices (shared) or on facing ones: one face per edge of the lower row; each inner lower
    vertex is joined to the upper vertex nearest in angle. Aligned columns give quads; the
    vertices of a finer upper row (the cut edge) go into n-gons instead of fans of triangles."""
    na, nb = len(lower) - 1, len(upper) - 1
    if shared and nb < 2:
        return [[x[2] for x in lower] + [x[2] for x in upper[1:-1][::-1]]]
    m = [0]
    for i in range(1, na):
        t = lower[i][0]
        cand = range(max(m[-1], 1), nb) if shared else range(m[-1], nb + 1)
        m.append(min(cand, key=lambda j: (abs(upper[j][0] - t), j)))
    m.append(nb)
    return [[lower[i][2], lower[i + 1][2]] + [upper[j][2] for j in range(m[i + 1], m[i] - 1, -1)]
            for i in range(na)]


def ang_un(v, ref, ang):
    """Angle of v, unwrapped next to ref."""
    t = ang(v)
    return ref + (t - ref + math.pi) % TAU - math.pi


def _anchor(v, chain_set, fset):
    """A vertex of the border row that must stay: joined to more than one outside face, or to
    an outside edge that doesn't run along the row."""
    outside = {f for f in v.link_faces if f not in fset}
    if len(outside) > 1:
        return True
    return any(e.other_vert(v) not in chain_set and all(f not in fset for f in e.link_faces)
               for e in v.link_edges)
