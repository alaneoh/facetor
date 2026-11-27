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

"""Rounded corners on the outline of a flat n-gon."""

import bmesh
import mathutils
import numpy as np

from .core import custom_profile, edge_between, keep_open, ordered, round_profile
from .grid import _resize_chain
from .safety import guarded


# ---------------------------------------------------------------------------
# Rounded corner of an outline: selected open edge chain lying on a flat face
# (e.g. a plane bevelled with Blender's vertex bevel)
# ---------------------------------------------------------------------------

def _open_chains(edges):
    adj = {}
    for e in edges:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    if any(len(l) > 2 for l in adj.values()):
        return None
    chains, seen = [], set()
    for v0, l in adj.items():
        if len(l) != 1 or v0 in seen:
            continue
        chain, prev, cur = [v0], None, v0
        seen.add(v0)
        while True:
            nxt_e = [e for e in adj[cur] if e is not prev]
            if not nxt_e:
                break
            prev = nxt_e[0]
            cur = prev.other_vert(cur)
            chain.append(cur)
            seen.add(cur)
        chains.append(chain)
    if len(seen) != len(adj):
        return None                      # a closed loop is in the selection (hole)
    return chains


def _outside_neighbour(v, inner, faces):
    """Vertex next to chain end v along the outline, away from the chain."""
    for f in faces:
        loops = list(f.loops)
        for i, l in enumerate(loops):
            if l.vert is v:
                for o in (loops[i - 1].vert, loops[(i + 1) % len(loops)].vert):
                    if o is not inner:
                        return o
    cands = [e.other_vert(v) for e in v.link_edges if e.other_vert(v) is not inner]
    return cands[0] if len(cands) == 1 else None


def outline_arcs(bm):
    """Selected open edge chains that lie on the outline of flat faces (every face along the
    chain holds the whole chain). Returns a list of arcs, or None."""
    if any(f.select for f in bm.faces):
        return None
    sel = [e for e in bm.edges if e.select]
    if not sel:
        return None
    chains = _open_chains(sel)
    if not chains:
        return None
    arcs = []
    for chain in chains:
        cset = set(chain)
        faces = set()
        for a, b in zip(chain, chain[1:]):
            e = edge_between(a, b)
            if e is None:
                return None
            faces |= set(e.link_faces)
        # a wire line (a profile drawn with edges only, no face anywhere on it): any chain,
        # even one edge (a chamfer, rounded like Bevel)
        wire = not faces and not any(v.link_faces for v in chain)
        if len(chain) < 3 and not wire:
            return None
        if not wire and (not faces or any(not cset <= set(f.verts) for f in faces)):
            return None
        pa = _outside_neighbour(chain[0], chain[1], faces)
        pb = _outside_neighbour(chain[-1], chain[-2], faces)
        if pa is None or pb is None or pa in cset or pb in cset:
            return None
        arcs.append({"chain": chain, "pa": pa, "pb": pb, "faces": ordered(faces), "wire": wire})
    return arcs


def outline_corner(arc):
    """Corner where the two straight edges next to the arc meet, or None."""
    a, b = arc["chain"][0].co, arc["chain"][-1].co
    ta = a - arc["pa"].co
    tb = b - arc["pb"].co
    if ta.length < 1e-12 or tb.length < 1e-12:
        return None
    ta.normalize()
    tb.normalize()
    if abs(ta.dot(tb)) > 0.9999:
        return None
    r = mathutils.geometry.intersect_line_line(a, a + ta, b, b + tb)
    if not r:
        return None
    c = (r[0] + r[1]) / 2
    if (c - a).dot(ta) <= 0 or (c - b).dot(tb) <= 0:
        return None
    return c


def outline_arc_points(arc, count, shape, profile_shape, width, profile=None):
    pts = [v.co.copy() for v in arc["chain"]]
    if shape == "KEEP" and len(pts) == 2 and count > 1 and outline_corner(arc) is not None:
        # one straight segment (a chamfer) has no curve to keep: more segments round it, like
        # Blender's Bevel
        shape = "ROUND"
    if shape == "KEEP" and abs(width - 1.0) < 1e-6:
        return keep_open(pts, count), None
    c = outline_corner(arc)
    if c is None:
        if shape == "KEEP":
            return None, "The width needs two straight edges that meet at a corner on each side of the arc."
        return None, "Round needs two straight edges that meet at a corner on each side of the arc."
    a, b = pts[0], pts[-1]
    # the new ends slide along the straight edges, never past their other end
    lim = min((c - arc["pa"].co).length / max((c - a).length, 1e-12),
              (c - arc["pb"].co).length / max((c - b).length, 1e-12))
    w = max(0.01, min(width, 0.98 * lim))
    if shape == "KEEP":
        # the kept curve scaled around the corner (its ends slide along the straight edges)
        return [c + (p - c) * w for p in keep_open(pts, count)], None
    a2, b2 = c + (a - c) * w, c + (b - c) * w
    if shape == "CUSTOM" and profile is not None:
        return custom_profile(a2, b2, c, profile, count), None
    return round_profile(a2, b2, c, profile_shape, count), None


def _uv_fit(bm, faces):
    """Affine map position -> UV for each UV layer, fitted on the given (flat) faces."""
    fits = []
    loops = [l for f in faces for l in f.loops]
    if len(loops) < 3:
        return fits
    P = np.array([tuple(l.vert.co) + (1.0,) for l in loops])
    for layer in bm.loops.layers.uv:
        UV = np.array([tuple(l[layer].uv) for l in loops])
        sol, *_ = np.linalg.lstsq(P, UV, rcond=None)
        if np.abs(P @ sol - UV).max() < 1e-4:
            fits.append((layer, sol))
    return fits


def outline_arc_merge(bm, arc):
    """Width 0: the arc goes back to the sharp corner (one vertex). Returns error or None."""
    c = outline_corner(arc)
    if c is None:
        return "A width of 0 needs two straight edges that meet at a corner on each side of the arc."
    faces = {f for a, b in zip(arc["chain"], arc["chain"][1:]) for f in edge_between(a, b).link_faces}
    fits = _uv_fit(bm, ordered(faces))
    chain = list(arc["chain"])

    def do():
        for v in chain:
            v.co = c
        size = max((arc["pa"].co - c).length, (arc["pb"].co - c).length, 1e-9)
        bmesh.ops.remove_doubles(bm, verts=chain, dist=1e-6 * size)
        left = [v for v in chain if v.is_valid]
        if len(left) != 1:
            return False
        v = left[0]
        for f in v.link_faces:
            for l in f.loops:
                if l.vert is v:
                    for layer, sol in fits:
                        l[layer].uv = tuple(np.array(tuple(v.co) + (1.0,)) @ sol)
        v.select = True
        return True
    ok, _ = guarded(bm, do)
    if not ok:
        return "This arc can't be merged back into its corner safely."
    return None


def outline_arc_rebuild(bm, arc, count, op, profile=None):
    if getattr(op, "width", 1.0) < 1e-4:
        return outline_arc_merge(bm, arc)
    if count < 1:
        return "Use 1 segment or more."
    pts, err = outline_arc_points(arc, count, op.shape, op.profile_shape, op.width, profile)
    if err:
        return err
    faces = {f for a, b in zip(arc["chain"], arc["chain"][1:]) for f in edge_between(a, b).link_faces}
    fits = _uv_fit(bm, ordered(faces))

    def do():
        chain = _resize_chain(bm, arc["chain"], False, count, pts)
        if len(chain) != count + 1:
            return False
        for v, p in zip(chain, pts):
            v.co = p
        cset = set(chain)
        for f in {f for v in chain for f in v.link_faces}:
            for l in f.loops:
                if l.vert in cset:
                    for layer, sol in fits:
                        l[layer].uv = tuple(np.array(tuple(l.vert.co) + (1.0,)) @ sol)
        for v in chain:
            v.select = True
        for a, b in zip(chain, chain[1:]):
            e = edge_between(a, b)
            if e:
                e.select = True
        return True
    ok, _ = guarded(bm, do, allow_wire=arc.get("wire", False))
    if not ok:
        return "This arc can't be changed safely (try more segments)."
    return None
