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

"""Rebuild of clean quad grids, fillets inside longer profiles, T-junction anchors."""

import math
import bmesh
import mathutils.kdtree

from .core import (
    Grid,
    bevel_corner,
    choose_ref_axis,
    custom_profile,
    edge_between,
    grid_profiles,
    keep_closed,
    keep_open,
    polyline_param,
    round_profile,
    with_extras,
    STRAIGHT,
    ordered,
)


# ---------------------------------------------------------------------------
# Rebuild one grid along direction d with `count` segments
# ---------------------------------------------------------------------------

EDGE_FLOAT_LAYERS = ("bevel_weight_edge", "crease_edge")


def _edge_attrs(bm, e):
    a = {"seam": e.seam, "smooth": e.smooth}
    for name in EDGE_FLOAT_LAYERS:
        lay = bm.edges.layers.float.get(name)
        if lay is not None:
            a[name] = e[lay]
    return a


def _merge_attrs(lst):
    if not lst:
        return None
    out = {"seam": any(a["seam"] for a in lst), "smooth": all(a["smooth"] for a in lst)}
    for name in EDGE_FLOAT_LAYERS:
        vals = [a[name] for a in lst if name in a]
        if vals:
            out[name] = max(vals)
    return out


def _set_edge_attrs(bm, e, a):
    if not a or e is None:
        return
    e.seam = a["seam"]
    e.smooth = a["smooth"]
    for name in EDGE_FLOAT_LAYERS:
        lay = bm.edges.layers.float.get(name)
        if lay is not None and name in a:
            e[lay] = a[name]


MAX_JOIN_SPAN = 5   # most old segments one joining n-gon may take (n-gon of up to 7 sides)


def _corner_ok(c, prof):
    """c is a real corner for the profile prof (a bevel between two faces at an angle), not a
    far-off intersection of nearly parallel faces."""
    if c is None:
        return False
    a, b = prof[0].co, prof[-1].co
    chord = (b - a).length
    da, db = a - c, b - c
    if chord < 1e-12 or da.length < 1e-12 or db.length < 1e-12:
        return False
    if da.length > 5 * chord or db.length > 5 * chord:
        return False
    return math.radians(20) < da.angle(db) < math.radians(160)


def collapse_to_corners(bm, g, d):
    """Width 0: every profile of the open grid g (direction d) is merged into its corner point,
    so the bevel becomes a sharp edge. Returns (error message or None, corner points)."""
    if g.cyclic[d]:
        return "A width of 0 needs a bevel between two faces (not a closed loop).", None
    profs = grid_profiles(g, d)
    rails_cyclic = g.cyclic[1 - d]
    corners = [bevel_corner(profs, k, g.fset, rails_cyclic) for k in range(len(profs))]
    if any(not _corner_ok(c, p) for c, p in zip(corners, profs)):
        return "Couldn't find the corner of this bevel from the faces next to it: it can't be merged.", None
    size = max(max((p[0].co - p[-1].co).length for p in profs),
               max((c - corners[0]).length for c in corners)) or 1.0
    verts = set()
    for p, c in zip(profs, corners):
        for v in p:
            v.co = c
            verts.add(v)
    bmesh.ops.remove_doubles(bm, verts=ordered(verts), dist=1e-6 * size)
    return None, corners


def plan_grid(bm, g, d, count, op, ctx_profile):
    """Compute everything (positions, attributes, UVs) before touching topology."""
    closed = g.cyclic[d]
    rails_cyclic = g.cyclic[1 - d]
    if closed and count < 3:
        return "A closed loop needs at least 3 segments.", None
    if count < 1:
        return "Need at least 1 segment.", None
    profs = grid_profiles(g, d)
    K = len(profs)
    M = len(profs[0]) if closed else len(profs[0]) - 1
    Q = [[v.co.copy() for v in p] for p in profs]
    for pts in Q:
        span = max((pts[i] - pts[0]).length for i in range(len(pts)))
        if span < 1e-9 or any((pts[(i + 1) % len(pts)] - pts[i]).length < span * 1e-5
                              for i in range(M)):
            return "This part has overlapping vertices (zero-length edges): clean it first (Merge by Distance).", None
    for k in range(K - 1 if not rails_cyclic else K):
        A, B = Q[k], Q[(k + 1) % K]
        span = max((A[i] - A[0]).length for i in range(len(A))) or 1.0
        if any((A[i] - B[i]).length < span * 1e-5 for i in range(len(A))):
            return "This part has overlapping vertices (zero-length edges): clean it first (Merge by Distance).", None

    # ----- UV seams across the profiles: rails where the faces on each side don't share UVs
    uv_layers = list(bm.loops.layers.uv.values())
    seam_rails = []
    if closed:
        for a in range(M):
            for k in range(K if rails_cyclic else K - 1):
                e = edge_between(profs[k][a], profs[(k + 1) % K][a])
                if e is None:
                    continue
                if e.seam or _uv_split(e, uv_layers):
                    seam_rails.append(a)
                    break

    # ----- new positions
    new_pts = []
    shape = op.shape if not closed else "KEEP"
    width = getattr(op, "width", 1.0)
    corners = None
    if not closed and (shape != "KEEP" or abs(width - 1.0) > 1e-6 or (M == 1 and count > 1)):
        corners = [bevel_corner(profs, k, g.fset, rails_cyclic) for k in range(K)]
    if shape == "KEEP" and M == 1 and count > 1 and corners and \
            all(_corner_ok(c, p) for c, p in zip(corners, profs)):
        # a chamfer (one straight segment between two faces) has no curve to keep: more
        # segments round it, like Blender's Bevel
        shape = "ROUND"
    if shape == "KEEP":
        ref = choose_ref_axis(Q) if closed else None
        # a closed profile cut by a UV seam keeps a new rail exactly on it: the UV island stays
        # cut in the same place whatever the count
        anchor = seam_rails[0] if seam_rails else None
        for pts in Q:
            new_pts.append(keep_closed(pts, count, ref, anchor) if closed else keep_open(pts, count))
        if closed and K > 1:
            # same starting column on every profile (a tie on the first point could shift one
            # profile by a column and twist the faces): match the first profile's positions
            p0 = [polyline_param(Q[0], True, p) for p in new_pts[0]]

            def cyc(x):
                x %= M
                return min(x, M - x)
            for k in range(1, K):
                pk = [polyline_param(Q[k], True, p) for p in new_pts[k]]
                r = min(range(count), key=lambda r: sum(cyc(pk[(i + r) % count] - p0[i]) for i in range(count)))
                new_pts[k] = new_pts[k][r:] + new_pts[k][:r]
        if not closed and abs(width - 1.0) > 1e-6:
            # width in Keep: the kept shape is scaled around the corner (its ends slide along
            # the faces next to it)
            if not corners or any(c is None for c in corners):
                return "Couldn't find the corner of this bevel from the faces next to it: the width can't change.", None
            new_pts = [[c + (p - c) * width for p in pts] for pts, c in zip(new_pts, corners)]
    else:
        if not corners or any(c is None for c in corners):
            return "Couldn't rebuild the corner from the surrounding faces (try Keep Silhouette).", None
        if shape == "CUSTOM" and ctx_profile is None:
            return "No custom profile available.", None
        for p, c in zip(profs, corners):
            a, b = p[0].co.copy(), p[-1].co.copy()
            if abs(width - 1.0) > 1e-6:
                a, b = c + (a - c) * width, c + (b - c) * width
            if shape == "CUSTOM":
                pts = custom_profile(a, b, c, ctx_profile, count)
            else:
                pts = round_profile(a, b, c, op.profile_shape, count)
            new_pts.append(pts)

    # ----- parameters of new points on the old profiles (UV / attribute transfer)
    params = [[polyline_param(Q[k], closed, p) for p in new_pts[k]] for k in range(K)]

    # ----- attributes
    rail_attr = {}   # old profile index a -> merged attrs of rail edges at a
    prof_attr = {}   # (k, a) -> attrs of profile edge a of profile k
    for k in range(K):
        p = profs[k]
        for a in range(M):
            e = edge_between(p[a], p[(a + 1) % len(p)])
            if e:
                prof_attr[(k, a)] = _edge_attrs(bm, e)
    for a in range(len(profs[0])):
        lst = []
        for k in range(K if rails_cyclic else K - 1):
            e = edge_between(profs[k][a], profs[(k + 1) % K][a])
            if e:
                lst.append(_edge_attrs(bm, e))
        rail_attr[a] = _merge_attrs(lst)

    old_faces = {}  # (a, k) -> (material, smooth, {(da, dk): [uv per layer]})
    for (fi, fj), (f, local) in g.faces.items():
        a, k = (fi, fj) if d == 0 else (fj, fi)
        corners = {}
        for l in f.loops:
            if l.vert not in local:
                continue        # T-junction vertex on the border side of an n-gon
            di, dj = local[l.vert]
            key = (di, dj) if d == 0 else (dj, di)
            corners[key] = [l[lay].uv.copy() for lay in uv_layers]
        old_faces[(a, k)] = (f.material_index, f.smooth, corners)

    # ----- which vertices survive
    grid_edges = {e for f in g.fset for e in f.edges}
    end_ks = [] if rails_cyclic else [0, K - 1]
    chain_ks = set()
    for k in end_ks:
        inner = profs[k] if closed else profs[k][1:-1]
        p = profs[k]
        p_edges = [edge_between(p[a], p[(a + 1) % len(p)]) for a in range(M)]
        if (any(f not in g.fset for v in inner for f in v.link_faces) or
                any(f not in g.fset for e in p_edges if e for f in e.link_faces)):
            chain_ks.add(k)
    # The edge of the selection must not be joined to other geometry in its middle (T-junctions
    # from CAD exports): dissolving such a vertex would tear the neighbouring faces.
    frozen_ks = set()
    flat_ks = set()
    for k in chain_ks:
        inner = profs[k] if closed else profs[k][1:-1]
        outs_all = [f for v in inner for f in v.link_faces if f not in g.fset]
        fan_centres = None
        out_set = set(outs_all)
        n_tris = sum(1 for f in out_set if len(f.verts) == 3)
        if outs_all and n_tris == len(out_set):
            common = set(outs_all[0].verts)
            for f in outs_all[1:]:
                common &= set(f.verts)
            fan_centres = common - set(inner)
        # mostly triangles (a few quads where the fans meet, e.g. a bore coming out through a
        # slanted face and a fillet): still fans
        if outs_all and not fan_centres and n_tris >= 0.85 * len(out_set):
            # several fans side by side (e.g. an arch on a flat face, triangulated from the
            # face's corners): a few apexes, each shared by several triangles
            pset = set(profs[k])
            apex = {}
            for f in ordered(set(outs_all)):
                for v in f.verts:
                    if v not in pset:
                        apex[v] = apex.get(v, 0) + 1
            if apex and len(apex) <= max(2, len(inner) // 3) and min(apex.values()) >= 2:
                fan_centres = set(apex)
        if (not fan_centres and outs_all and _coplanar_faces(set(outs_all))
                and _one_border_loop(set(outs_all))):
            # the side is held by a flat area cut into strips or triangles (CAD floor under an
            # arch): its inner edges are dissolved first, so the chain is free (see execute_plan)
            flat_ks.add(k)
            continue
        anchors = sum(1 for v in inner if any(e not in grid_edges for e in v.link_edges))
        extras = getattr(g, "extras", {})
        p = profs[k]
        t_verts = [v for a in range(len(p) - 1) for v in extras.get(frozenset((p[a], p[a + 1])), [])[1:-1]]
        t_anchors = sum(1 for v in t_verts if any(e not in grid_edges and e.other_vert(v) not in t_verts
                                                  and e.other_vert(v) not in p for e in v.link_edges))
        if (not closed and not fan_centres and t_verts and anchors + t_anchors > count - 1
                and math.ceil(M / count) <= MAX_JOIN_SPAN):
            # more T-junction vertices on this end profile than the new count can keep: it's kept
            # as it is (with them) and joined to the new segments with n-gons. Otherwise they go
            # into the chain as vertices held by outside geometry (see _chain_with_extras)
            frozen_ks.add(k)
            continue
        if (not fan_centres and not closed and count < M and anchors > count - 1 and
                math.ceil(M / count) <= MAX_JOIN_SPAN and
                all(len([f for f in v.link_faces if f not in g.fset]) >= 2 for v in inner)):
            # every vertex of this end profile carries an outside edge (the rows go on past the
            # selection, e.g. the straight side of a fillet that goes on round a corner): it can't
            # lose any, so it's kept as it is and joined to the new segments with n-gons
            frozen_ks.add(k)
            continue
        if fan_centres or not any(len([f for f in v.link_faces if f not in g.fset]) >= 2 and
                                  any(len(f.verts) == 3 for f in v.link_faces if f not in g.fset)
                                  for v in inner):
            continue
        # vertices joined to outside geometry are always kept (see _resize_chain). When there are
        # more of them than the new count allows, this end profile is kept as it is and a row of
        # small n-gons joins it to the new segments (messy CAD corners at the ends of a band)
        anchors = sum(1 for v in inner if any(e not in grid_edges for e in v.link_edges))
        if count < M and anchors > (count if closed else count - 1):
            if not closed and math.ceil(M / count) <= MAX_JOIN_SPAN:
                frozen_ks.add(k)
                continue
            least = anchors if closed else min(anchors + 1, math.ceil(M / MAX_JOIN_SPAN))
            return ("The edge of this part is joined to other geometry in %d places (T-junctions): "
                    "use %d segments or more." % (anchors, max(least, 1))), None
    delete_verts = []
    for k in range(K):
        if k in chain_ks:
            continue
        inner = profs[k] if closed else profs[k][1:-1]
        for v in inner:
            for e in v.link_edges:
                if e not in grid_edges:
                    return ("Some geometry is attached to the inside of the selection. "
                            "Select the whole bevel/cylinder band (or only the faces you want to rebuild)."), None
            delete_verts.append(v)

    plan = dict(seam_rails=seam_rails, Q=Q, d=d, closed=closed, rails_cyclic=rails_cyclic, K=K, M=M, profs=profs, new_pts=new_pts,
                params=params, rail_attr=rail_attr, prof_attr=prof_attr, old_faces=old_faces,
                uv_layers=uv_layers, chain_ks=chain_ks, delete_verts=delete_verts, count=count, fset=g.fset,
                frozen_ks=frozen_ks, old_edges=ordered(grid_edges), flat_ks=flat_ks,
                extras=getattr(g, "extras", {}))
    if len(frozen_ks) >= K:
        return "Both sides of this part are held by other geometry: it can't be changed.", None
    return None, plan


def _resize_chain(bm, chain, closed, count, new_pts):
    """Change the vertex count of a boundary chain still attached to outside faces."""
    M = len(chain) if closed else len(chain) - 1
    if closed:
        anchor = min(range(len(chain)), key=lambda i: (chain[i].co - new_pts[0]).length)
        chain = chain[anchor:] + chain[:anchor]
    if count < M:
        if closed:
            keep = sorted({int(math.floor(i * M / count + 0.5)) % M for i in range(count)})
        else:
            keep = sorted({int(math.floor(i * M / count + 0.5)) for i in range(count + 1)})
        # Vertices also joined to other geometry (T-junctions) are kept first: removing them
        # would tear the neighbouring faces. The rest are spread as evenly as possible.
        cset = set(chain)
        inner = range(len(chain)) if closed else range(1, M)
        anchors = [i for i in inner
                   if any(e.other_vert(chain[i]) not in cset for e in chain[i].link_edges)]
        need = count if closed else count + 1
        fixed = set(anchors) if closed else set(anchors) | {0, M}
        if anchors and len(fixed) <= need:
            chosen = set(fixed)
            free = [i for i in range(len(chain)) if i not in chosen]
            while len(chosen) < need and free:
                sel_sorted = sorted(chosen)
                if closed:
                    gaps = [(sel_sorted[(k + 1) % len(sel_sorted)] - sel_sorted[k]) % len(chain) or len(chain)
                            for k in range(len(sel_sorted))]
                else:
                    gaps = [sel_sorted[k + 1] - sel_sorted[k] for k in range(len(sel_sorted) - 1)]
                k = max(range(len(gaps)), key=lambda q: gaps[q])
                mid = (sel_sorted[k] + gaps[k] / 2.0) % len(chain) if closed else sel_sorted[k] + gaps[k] / 2.0
                best = min(free, key=lambda i: min(abs(i - mid), len(chain) - abs(i - mid)) if closed else abs(i - mid))
                chosen.add(best)
                free.remove(best)
            keep = sorted(chosen)
        kset = set(keep)
        rem = [v for i, v in enumerate(chain) if i not in kset]
        if rem:
            bmesh.ops.dissolve_verts(bm, verts=rem, use_face_split=False, use_boundary_tear=False)
        chain = [chain[i] for i in keep]
    elif count > M:
        extra = count - M
        segs = M
        out = []
        for e_i in range(segs):
            a = chain[e_i]
            b = chain[(e_i + 1) % len(chain)]
            out.append(a)
            c = (e_i + 1) * extra // segs - e_i * extra // segs
            edge = edge_between(a, b)
            cur = a
            for s in range(c):
                ne, nv = bmesh.utils.edge_split(edge, cur, 1.0 / (c + 1 - s))
                # edge_split returns the new edge attached to cur and keeps `edge` on the far side
                if b in edge.verts:
                    pass
                else:
                    edge = ne
                out.append(nv)
                cur = nv
        if not closed:
            out.append(chain[-1])
        chain = out
    return chain


def _chain_with_extras(p, closed, extras):
    """End profile p with the T-junction vertices on its sides put in (in order)."""
    if not extras:
        return list(p)
    out = []
    n = len(p)
    for a in range(n if closed else n - 1):
        out.append(p[a])
        path = extras.get(frozenset((p[a], p[(a + 1) % n])))
        if path:
            out += path[1:-1] if path[0] is p[a] else path[-2:0:-1]
    if not closed:
        out.append(p[-1])
    return out


def execute_plan(bm, plan):
    d, closed, K, count = plan["d"], plan["closed"], plan["K"], plan["count"]
    profs, new_pts = plan["profs"], plan["new_pts"]
    old_endpoints = [(p[0], p[-1]) for p in profs] if not closed else None

    bmesh.ops.delete(bm, geom=list(plan["fset"]), context="FACES_ONLY")
    dv = [v for v in plan["delete_verts"] if v.is_valid]
    if dv:
        bmesh.ops.delete(bm, geom=dv, context="VERTS")
    # rungs between two kept profiles (one row of faces held on both sides) left without faces:
    # removed now, or every vertex of the profiles would look joined to outside geometry
    keep_e = set()
    for k in set(plan["chain_ks"]) | set(plan.get("frozen_ks", ())):
        p = profs[k]
        for a in range(len(p) if closed else len(p) - 1):
            e = edge_between(p[a], p[(a + 1) % len(p)])
            if e is not None:
                keep_e.add(e)
    for e in [e for e in plan.get("old_edges", ()) if e.is_valid and not e.link_faces and e not in keep_e]:
        bm.edges.remove(e)

    frozen = plan.get("frozen_ks", set())
    # T-junction vertices on a resized end profile become vertices of its chain: no longer put
    # back in the new faces (the new chain doesn't have the old pairs around them)
    extras = dict(plan.get("extras") or {})
    kept = dict(extras)
    for k in plan["chain_ks"]:
        if k not in frozen:
            p = profs[k]
            for a in range(len(p) if closed else len(p) - 1):
                kept.pop(frozenset((p[a], p[(a + 1) % len(p)])), None)
    plan["extras"] = kept
    new_profs = []
    for k in range(K):
        if k in frozen:
            new_profs.append(list(profs[k]))      # kept as it is
            continue
        if k in plan["chain_ks"]:
            if k in plan.get("flat_ks", ()) and count < plan["M"]:
                _free_flat_side(bm, profs[k], closed)
            chain = _resize_chain(bm, _chain_with_extras(profs[k], closed, extras), closed, count, new_pts[k])
        else:
            if closed:
                chain = [bm.verts.new(p) for p in new_pts[k]]
            else:
                a, b = old_endpoints[k]
                chain = [a] + [bm.verts.new(p) for p in new_pts[k][1:-1]] + [b]
        for v, p in zip(chain, new_pts[k]):
            v.co = p
        new_profs.append(chain)
    plan["new_profs"] = new_profs

    # faces
    n_faces_k = K if plan["rails_cyclic"] else K - 1
    seg = count
    new_faces = []
    uv_layers = plan["uv_layers"]
    M = plan["M"]
    for k in range(n_faces_k):
        A, B = new_profs[k], new_profs[(k + 1) % K]
        pa, pb = plan["params"][k], plan["params"][(k + 1) % K]
        if k in frozen or (k + 1) % K in frozen:
            new_faces += _join_frozen(bm, plan, k, A, B, k in frozen)
            continue
        for i in range(seg):
            i1 = (i + 1) % len(A)
            if d == 0:
                vs = [A[i], A[i1], B[i1], B[i]]
            else:
                vs = [A[i], B[i], B[i1], A[i1]]
            vs = with_extras(vs, plan.get("extras"))
            try:
                f = bm.faces.new(vs)
            except ValueError:
                f = bm.faces.get(vs)
                if f is None:
                    continue
            new_faces.append(f)
            # parameters along the old profile
            s0, s1, t0, t1 = pa[i], pa[i1], pb[i], pb[i1]
            if closed:
                def unwrap(x, ref):
                    while x < ref - M / 2:
                        x += M
                    while x > ref + M / 2:
                        x -= M
                    return x
                s1, t0, t1 = unwrap(s1, s0), unwrap(t0, s0), unwrap(t1, s0)
                if s1 <= s0:
                    s1 += M
                if t1 <= t0:
                    t1 += M
            mid = (s0 + s1 + t0 + t1) / 4
            a_old = int(math.floor(mid))
            a_key = a_old % M if closed else max(0, min(M - 1, a_old))
            old = plan["old_faces"].get((a_key, k))
            if old:
                mat, smooth, corners = old
                f.material_index = mat
                f.smooth = smooth
                if uv_layers and len(corners) == 4:
                    # each corner takes its UV inside the old face it lies in (so neighbouring new
                    # faces agree on their shared vertices); a corner on an old rail takes the
                    # old face on this face's side (the two sides of a UV seam differ)
                    cuv = {}
                    for v, s, kk in ((A[i], s0, 0), (A[i1], s1, 0), (B[i], t0, 1), (B[i1], t1, 1)):
                        if abs(s - round(s)) < 0.01:
                            s = float(round(s))      # on an old rail (31.999 is rail 32)
                        a = int(math.floor(s))
                        if s == a and mid < s:
                            a -= 1
                        # never take UVs across a UV seam: the old face next to the seam on this
                        # face's side, extended if needed
                        for r in plan.get("seam_rails", ()) if closed else ():
                            r = r + M * round((mid - r) / M)
                            if mid < r <= s or (mid < r and a >= r):
                                a = r - 1
                            elif s <= r < mid or (mid > r and a < r):
                                a = r
                        if not closed:
                            a = max(0, min(M - 1, a))
                        src = plan["old_faces"].get((a % M if closed else a, k))
                        src = src[2] if src and len(src[2]) == 4 else corners
                        fr = s - a if src is not corners else s - (a_key if not closed else a_old)
                        cuv[v] = [c0.lerp(c1, fr) for c0, c1 in zip(src[(0, kk)], src[(1, kk)])]
                    loops = list(f.loops)
                    for n_l, l in enumerate(loops):
                        uvs = cuv.get(l.vert)
                        if uvs is None:
                            # T-junction vertex on a side: between the corners around it
                            p = next(loops[n_l - q] for q in range(1, len(loops)) if loops[n_l - q].vert in cuv)
                            nx = next(loops[(n_l + q) % len(loops)] for q in range(1, len(loops))
                                      if loops[(n_l + q) % len(loops)].vert in cuv)
                            da = (l.vert.co - p.vert.co).length
                            db = (nx.vert.co - l.vert.co).length
                            t = da / (da + db) if da + db > 1e-12 else 0.5
                            uvs = [u0.lerp(u1, t) for u0, u1 in zip(cuv[p.vert], cuv[nx.vert])]
                        for lay, uv in zip(uv_layers, uvs):
                            l[lay].uv = uv

    # edge attributes
    for k in range(K):
        if k in frozen:
            continue
        P = new_profs[k]
        for i in range(seg):
            e = edge_between(P[i], P[(i + 1) % len(P)])
            mid = (plan["params"][k][i] + plan["params"][k][(i + 1) % len(P)]) / 2
            if closed and abs(plan["params"][k][(i + 1) % len(P)] - plan["params"][k][i]) > M / 2:
                mid = (mid + M / 2) % M
            a = int(math.floor(mid))
            a = a % M if closed else max(0, min(M - 1, a))
            _set_edge_attrs(bm, e, plan["prof_attr"].get((k, a)))
    n_rails = seg + (0 if closed else 1)
    for i in range(n_rails):
        if not closed and i in (0, n_rails - 1):
            continue  # original side rails kept as they were
        a = int(round(plan["params"][0][i]))
        a = a % len(profs[0]) if closed else max(0, min(len(profs[0]) - 1, a))
        attrs = plan["rail_attr"].get(a)
        if not attrs:
            continue
        # only transfer "strong" features (seams / sharp / weights) to the nearest new rail
        for k in range(n_faces_k):
            if k in frozen or (k + 1) % K in frozen:
                continue
            e = edge_between(new_profs[k][i], new_profs[(k + 1) % K][i])
            if e is None:
                continue
            e.seam = attrs["seam"] and _nearest_rail(plan, i, a)
            e.smooth = attrs["smooth"] or not _nearest_rail(plan, i, a)
            for name in EDGE_FLOAT_LAYERS:
                lay = bm.edges.layers.float.get(name)
                if lay is not None and name in attrs:
                    e[lay] = attrs[name] if _nearest_rail(plan, i, a) else 0.0

    # old edges of the deleted faces that no new face uses any more (rungs between vertices that
    # were both kept but are no longer facing each other) would be left loose
    for e in plan.get("old_edges", ()):
        if e.is_valid and not e.link_faces:
            bm.edges.remove(e)

    for f in new_faces:
        f.select = True
    return new_faces


def _one_border_loop(faces):
    """The faces make one piece without a hole (their border is a single loop): joining them
    gives one n-gon. A flat ring cut in two (the side of a knob) would give a face with a hole."""
    bedges = {e for f in faces for e in f.edges if sum(1 for g in e.link_faces if g in faces) == 1}
    if not bedges:
        return False
    seen, loops = set(), 0
    for e0 in bedges:
        if e0 in seen:
            continue
        loops += 1
        st = [e0]
        seen.add(e0)
        while st:
            x = st.pop()
            for v in x.verts:
                for y in v.link_edges:
                    if y in bedges and y not in seen:
                        seen.add(y)
                        st.append(y)
    return loops == 1


def _coplanar_faces(faces, tol=math.radians(1.0)):
    """All faces in one plane (and at least two of them sharing an edge with each other)."""
    faces = list(faces)
    if len(faces) < 2:
        return False
    n0 = faces[0].normal
    if n0.length < 0.5:
        return False
    p0 = faces[0].verts[0].co
    size = max((v.co - p0).length for f in faces for v in f.verts) or 1.0
    for f in faces:
        if f.normal.length < 0.5 or f.normal.angle(n0, 0.0) > tol:
            return False
        if any(abs((v.co - p0).dot(n0)) > 1e-3 * size for v in f.verts):
            return False
    fs = set(faces)
    return any(sum(1 for g in e.link_faces if g in fs) == 2 for f in faces for e in f.edges)


def merge_caps(bm, faces):
    """Flat caps cut into strips or triangles (CAD: bottom of a blind hole, end of a rod) that
    only use vertices of the selection's outline: joined into one n-gon, so the outline's
    vertices are free. Returns the number of caps joined."""
    fset = set(faces)
    outline = {v for f in faces for e in f.edges
               if any(g not in fset for g in e.link_faces) for v in e.verts}
    cand = {g for v in outline for g in v.link_faces if g not in fset and all(x in outline for x in g.verts)}
    done, n = set(), 0
    for f in ordered(cand):
        if f in done:
            continue
        grp, stack = [f], [f]
        done.add(f)
        while stack:
            g = stack.pop()
            for e in g.edges:
                for h in e.link_faces:
                    if h in cand and h not in done:
                        done.add(h)
                        grp.append(h)
                        stack.append(h)
        if len(grp) >= 2 and _coplanar_faces(grp):
            res = bmesh.ops.dissolve_faces(bm, faces=grp)
            for g in res["region"]:
                g.select = False
            n += 1
    return n


def _free_flat_side(bm, chain, closed, fans_only=False):
    """Dissolve the edges between the flat faces that hold the inner vertices of a chain, so the
    chain vertices can be removed (the flat area becomes one n-gon).
    fans_only: only if the flat area is cut into triangles (CAD fans), not a clean quad ring."""
    inner = chain if closed else chain[1:-1]
    cset = set(chain)
    faces = {f for v in inner for f in v.link_faces}
    if not faces or not _coplanar_faces(faces):
        return
    if fans_only and not any(len(f.verts) == 3 for f in faces):
        return
    edges = set()
    for v in inner:
        for e in v.link_edges:
            if e.other_vert(v) in cset and edge_between(v, e.other_vert(v)) is e and \
                    abs(chain.index(e.other_vert(v)) - chain.index(v)) in (1, len(chain) - 1):
                continue                    # an edge of the chain itself
            if len(e.link_faces) == 2 and all(f in faces for f in e.link_faces):
                edges.add(e)
    if edges:
        bmesh.ops.dissolve_edges(bm, edges=ordered(edges), use_verts=False, use_face_split=False)


def _join_frozen(bm, plan, k, A, B, a_frozen):
    """Faces between a profile kept as it is (old vertices) and a new one: each new segment
    takes the old vertices that fall in it, so the faces are small n-gons (or triangles)."""
    d, M, seg = plan["d"], plan["M"], plan["count"]
    F, N = (A, B) if a_frozen else (B, A)
    pn = plan["params"][(k + 1) if a_frozen else k]
    groups = []      # (old j0..j1, new i0..i1) facing each other
    if seg <= M:
        for i in range(seg):
            j0 = max(0, min(M, int(round(pn[i]))))
            j1 = max(j0, min(M, int(round(pn[i + 1]))))
            if i == seg - 1:
                j1 = M
            if i == 0:
                j0 = 0
            groups.append((j0, j1, i, i + 1))
    else:
        # more new segments than old ones: each old segment takes the new vertices in it
        def nearest(j):
            return min(range(seg + 1), key=lambda i: abs(pn[i] - j))
        i0 = 0
        for j in range(M):
            i1 = seg if j == M - 1 else max(i0 + 1, nearest(j + 1))
            groups.append((j, j + 1, i0, i1))
            i0 = i1
    out = []
    for j0, j1, i0, i1 in groups:
        run = F[j0:j1 + 1]
        new = N[i0:i1 + 1]
        if a_frozen:
            vs = run + new[::-1] if d == 0 else [run[0]] + new + run[:0:-1]
        else:
            vs = new + run[::-1] if d == 0 else [new[0]] + run + new[:0:-1]
        if len(set(vs)) < 3:
            continue
        vs = with_extras(vs, plan.get("extras"))
        try:
            f = bm.faces.new(vs)
        except ValueError:
            f = bm.faces.get(vs)
            if f is None:
                continue
        old = plan["old_faces"].get((max(0, min(M - 1, j0)), k))
        if old:
            f.material_index, f.smooth = old[0], old[1]
        out.append(f)
    return out


def tiny_edges(faces, rel=1e-4):
    """Edges of the faces far shorter than the selection (CAD: two vertices a few
    ten-thousandths apart where a line ends)."""
    vs = {v for f in faces for v in f.verts}
    if not vs:
        return []
    lo = [min(v.co[i] for v in vs) for i in range(3)]
    hi = [max(v.co[i] for v in vs) for i in range(3)]
    diag = sum((b - a) ** 2 for a, b in zip(lo, hi)) ** 0.5
    return ordered({e for f in faces for e in f.edges if e.calc_length() < rel * diag})


def collapse_tiny_edges(bm, faces):
    """Merge the vertices of tiny_edges(faces) (a local Merge by Distance). Returns how many."""
    tiny = tiny_edges(faces)
    if tiny:
        # collapse rebuilds the faces around a merged vertex without their selection (a hole
        # wall whose faces each hold one tiny edge lost them all): marked first, reselected after
        # (a new layer invalidates the Python references: indices kept across it)
        bm.faces.index_update()
        bm.edges.index_update()
        fi = [f.index for f in faces if f.is_valid]
        ei = [e.index for e in tiny]
        lay = bm.faces.layers.int.new("_facetor_tiny")
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        for i in fi:
            bm.faces[i][lay] = 1
        bmesh.ops.collapse(bm, edges=[bm.edges[i] for i in ei], uvs=True)
        for f in bm.faces:
            if f[lay]:
                f.select = True
        bm.faces.layers.int.remove(lay)
    return len(tiny)


def extend_t_columns(bm, faces, max_iter=None):
    """Interior T-junctions of a selection (CAD: a line of faces ends in the middle of the side of
    a face, e.g. a fillet with more rows on its corners than on its straight sides): the line is
    carried on across that face, and the next ones, up to the border of the selection or to a
    vertex already there, so the selection becomes a quad grid again. The new vertex on the
    opposite side is put at the same fraction of its length, or on a vertex already close to it.
    Returns the faces of the selection afterwards (selected), or None if it can't be done."""
    fset = set(faces)
    if max_iter is None:
        max_iter = 20 * len(fset) + 100

    def side_path(loops, i, j):
        out = [loops[i].vert]
        k = i
        while k != j:
            k = (k + 1) % len(loops)
            out.append(loops[k].vert)
        return out

    def arc(path):
        return sum((b.co - a.co).length for a, b in zip(path, path[1:]))

    def find_target(f):
        if not f.is_valid or len(f.verts) <= 4:
            return None
        loops = list(f.loops)
        corner_i = [n for n, l in enumerate(loops) if l.calc_angle() <= STRAIGHT]
        if len(corner_i) != 4:
            return None
        for n in range(len(loops)):
            if n in corner_i:
                continue
            l = loops[n]
            if any(g is not f and g in fset for e in (l.edge, loops[n - 1].edge) for g in e.link_faces):
                return loops, corner_i, n
        return None

    # the selection's border, once (splits and joins inside don't change its directions)
    border = [(e.verts[0].co + e.verts[1].co) / 2 for f in fset for e in f.edges
              if sum(1 for g in e.link_faces if g in fset) == 1]
    border_dirs = [(e.verts[1].co - e.verts[0].co) for f in fset for e in f.edges
                   if sum(1 for g in e.link_faces if g in fset) == 1]
    bkd = mathutils.kdtree.KDTree(max(1, len(border)))
    for i, m in enumerate(border):
        bkd.insert(m, i)
    bkd.balance()

    def border_dir(p):
        """Direction of the selection's border edge nearest to p."""
        if not border:
            return None
        _, i, _ = bkd.find(p)
        bd = border_dirs[i]
        return bd.normalized() if bd.length > 1e-12 else None

    def line_from(t, stem):
        """Edges of the line of faces that ends at T-junction t (stem: its edge at t), up to
        its other end (another T-junction or the border)."""
        edges, v, e = [stem], stem.other_vert(t), stem
        for _ in range(len(fset) + 2):
            inside = [x for x in v.link_edges if x is not e and
                      all(g in fset for g in x.link_faces) and len(x.link_faces) == 2]
            if len(v.link_edges) != 4 or len(inside) != 3:
                break                    # end of the line: T-junction or border
            # the edge across from e: not sharing a face with it
            ef = set(e.link_faces)
            nxt = [x for x in inside if not (set(x.link_faces) & ef)]
            if len(nxt) != 1:
                break
            e = nxt[0]
            edges.append(e)
            v = e.other_vert(v)
        return edges

    # rows are taken out only on a ring (a band closed on itself, two border loops): there a line
    # carried on along the ring would go round and round. On an open patch a line carried on
    # always reaches the border.
    bedges = {e for f in fset for e in f.edges if sum(1 for g in e.link_faces if g in fset) == 1}
    loops, seen = 0, set()
    for e0 in bedges:
        if e0 in seen:
            continue
        loops += 1
        st = [e0]
        seen.add(e0)
        while st:
            x = st.pop()
            for v in x.verts:
                for y in v.link_edges:
                    if y in bedges and y not in seen:
                        seen.add(y)
                        st.append(y)
    ring = loops >= 2

    queue = ordered(f for f in fset if len(f.verts) > 4)
    it = 0
    while queue:
        f = queue.pop()
        tg = find_target(f)
        if tg is None:
            continue
        it += 1
        if it > max_iter:
            return None
        loops, corner_i, n = tg
        t = loops[n].vert
        stems = [e for e in t.link_edges if e not in set(f.edges)]
        if len(stems) != 1:
            return None
        sd = stems[0].verts[1].co - stems[0].verts[0].co
        bdir = border_dir(t.co)
        if ring and bdir is not None and sd.length > 1e-12 and abs(sd.normalized().dot(bdir)) > 0.7:
            # a line running along the border (a row that only some parts have, e.g. the corners
            # of a fillet): taken out, its faces are joined two by two
            line = line_from(t, stems[0])
            res = bmesh.ops.dissolve_edges(bm, edges=line, use_verts=True, use_face_split=False)
            for g in res.get("region", []):
                g.select = True
            fset = {g for g in bm.faces if g.select}
            queue = ordered(g for g in fset if len(g.verts) > 4)
            continue
        # side A..B holding t, and the opposite side C..D (A faces D, B faces C)
        before = [c for c in corner_i if c < n]
        ia = before[-1] if before else corner_i[-1]
        q = corner_i.index(ia)
        ib, ic, idd = corner_i[(q + 1) % 4], corner_i[(q + 2) % 4], corner_i[(q + 3) % 4]
        s = arc(side_path(loops, ia, n)) / max(arc(side_path(loops, ia, ib)), 1e-12)
        cd = side_path(loops, ic, idd)
        want = (1.0 - s) * arc(cd)
        # a vertex already on the opposite side, close to the spot: use it
        acc, pos = 0.0, [0.0]
        for a, b in zip(cd, cd[1:]):
            acc += (b.co - a.co).length
            pos.append(acc)
        t2 = None
        for m in range(1, len(cd) - 1):
            gap = min(pos[m] - pos[m - 1], pos[m + 1] - pos[m])
            if abs(pos[m] - want) < 0.5 * gap:
                t2 = cd[m]
        if t2 is None:
            for m in range(len(cd) - 1):
                if pos[m] <= want <= pos[m + 1]:
                    a, b = cd[m], cd[m + 1]
                    seg = pos[m + 1] - pos[m]
                    r = (want - pos[m]) / seg if seg > 1e-12 else 0.5
                    r = min(max(r, 0.05), 0.95)
                    e = edge_between(a, b)
                    if e is None:
                        return None
                    _, t2 = bmesh.utils.edge_split(e, a, r)
                    break
        if t2 is None or t2 is t or edge_between(t, t2) is not None:
            return None
        try:
            nf, _ = bmesh.utils.face_split(f, t, t2)
        except Exception:
            return None
        if nf is None:
            return None
        nf.select = True
        fset.add(nf)
        queue += [f, nf] + [g for g in t2.link_faces if g in fset]
    return [f for f in bm.faces if f in fset]


def _uv_split(e, uv_layers):
    """True if the two faces of edge e have different UVs on it (border of a UV island)."""
    if len(e.link_faces) != 2 or not uv_layers:
        return False
    f, g = e.link_faces
    for v in e.verts:
        lf = next((l for l in f.loops if l.vert is v), None)
        lg = next((l for l in g.loops if l.vert is v), None)
        if lf is None or lg is None:
            return False
        if any((lf[lay].uv - lg[lay].uv).length > 1e-5 for lay in uv_layers):
            return True
    return False


def _nearest_rail(plan, i, a):
    """True if new rail i is the closest new rail to old rail a (so a seam isn't duplicated)."""
    ps = plan["params"][0]
    M = plan["M"]
    closed = plan["closed"]

    def dist(x):
        dd = abs(x - a)
        return min(dd, M - dd) if closed else dd
    return min(range(len(ps)), key=lambda j: dist(ps[j])) == i


# ---------------------------------------------------------------------------
# Fillets inside a longer profile (straight neck + fillet + flat ring...): only the
# curved part is resampled across, the straight parts keep their loops
# ---------------------------------------------------------------------------

def _subgrid(g, d, lo, hi):
    """Band of the grid between vertex indices lo..hi along dimension d (d must be open)."""
    sg = Grid()
    sg.V, sg.faces = {}, {}
    for (i, j), v in g.V.items():
        a = i if d == 0 else j
        if lo <= a <= hi:
            sg.V[(i - lo, j) if d == 0 else (i, j - lo)] = v
    for (fi, fj), val in g.faces.items():
        a = fi if d == 0 else fj
        if lo <= a < hi:
            sg.faces[(fi - lo, fj) if d == 0 else (fi, fj - lo)] = val
    sg.fset = {val[0] for val in sg.faces.values()}
    sg.extras = getattr(g, "extras", {})
    n = list(g.n)
    n[d] = hi - lo + 1
    sg.n = tuple(n)
    cyc = list(g.cyclic)
    cyc[d] = False
    sg.cyclic = tuple(cyc)
    return sg


def curved_runs(g, d, min_turn=math.radians(0.5), max_turn=math.radians(60)):
    """Vertex index ranges (lo, hi) along open dimension d that are smooth curves (fillets)."""
    profs = grid_profiles(g, d)
    n = len(profs[0])
    turn = [0.0] * n
    ratio = [99.0] * n
    for a in range(1, n - 1):
        vals, rats = [], []
        for p in profs:
            v1, v2 = p[a].co - p[a - 1].co, p[a + 1].co - p[a].co
            if v1.length > 1e-12 and v2.length > 1e-12:
                vals.append(v1.angle(v2))
                rats.append(max(v1.length, v2.length) / min(v1.length, v2.length))
        turn[a] = sum(vals) / len(vals) if vals else 0.0
        ratio[a] = sum(rats) / len(rats) if rats else 99.0

    def inside(a):
        return min_turn < turn[a] < max_turn and ratio[a] < 2.5

    runs, a = [], 1
    while a < n - 1:
        if inside(a):
            b = a
            while b + 1 < n - 1 and inside(b + 1):
                b += 1
            # a tangent junction bends about half as much as the arc itself: leave it out
            end = b
            inner = sorted(turn[a:b + 1])
            med = inner[len(inner) // 2]
            # (not at the ends of the selection: a last face as long as the others belongs to
            # the arc, there is no straight part after it to keep)
            while a < b and a > 1 and turn[a] < 0.7 * med:
                a += 1
            while b > a and b < n - 2 and turn[b] < 0.7 * med:
                b -= 1
            runs.append((a - 1, b + 1))
            a = end + 1
        else:
            a += 1
    # one arc cut into pieces only by uneven spacing (CAD lines of two strips carried across
    # each other): the vertices between the pieces still turn like the arc -> one run
    if len(runs) > 1:
        inner = sorted(t for lo, hi in runs for t in turn[lo + 1:hi])
        med = inner[len(inner) // 2] if inner else 0.0
        def run_med(lo, hi):
            ts = sorted(turn[lo + 1:hi]) or [0.0]
            return ts[len(ts) // 2]
        merged = [runs[0]]
        last = runs[0]
        for lo, hi in runs[1:]:
            plo, phi = merged[-1]
            gap = turn[phi:lo + 1] if lo >= phi else []
            m1, m2 = run_med(*last), run_med(lo, hi)
            last = (lo, hi)
            # cut only by the spacing (a segment 2.5x longer than its neighbour), still turning
            # like the arc, same radius on both sides (two different fillets stay apart)
            if gap and lo - phi <= 2 and any(ratio[x] >= 2.5 for x in range(phi, lo + 1)) \
                    and all(min_turn < t < max_turn and 0.5 * med < t < 2 * med for t in gap) \
                    and min(m1, m2) > 0 and max(m1, m2) / min(m1, m2) < 1.6:
                merged[-1] = (plo, hi)
            else:
                merged.append((lo, hi))
        runs = merged
    return runs


def resample_bands(g, d):
    """The bands to resample along open dimension d: only the fillets when the profile also has
    straight parts, else the whole profile."""
    if g.cyclic[d]:
        return [None]
    n = g.n[d]
    runs = curved_runs(g, d)
    if runs and runs != [(0, n - 1)]:
        return runs
    return [None]
