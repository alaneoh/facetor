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

"""Safety net: every change is checked and undone if it tears the mesh."""

import bpy


# ---------------------------------------------------------------------------
# Safety net: apply a change, check it didn't tear the mesh, roll back if it did
# ---------------------------------------------------------------------------

def _torn_edges(bm):
    """Edges used by more than two faces, plus wire edges (no face)."""
    return sum(1 for e in bm.edges if len(e.link_faces) > 2 or not e.link_faces)


def _boundary_edges(bm):
    """Number of separate holes/open borders in the mesh."""
    bedges = {e for e in bm.edges if len(e.link_faces) == 1}
    seen, n = set(), 0
    for e in bedges:
        if e in seen:
            continue
        n += 1
        stack = [e]
        seen.add(e)
        while stack:
            x = stack.pop()
            for v in x.verts:
                for o in v.link_edges:
                    if o in bedges and o not in seen:
                        seen.add(o)
                        stack.append(o)
    return n


def _restore(bm, snap):
    me = bpy.data.meshes.new("_facetor_tmp")
    snap.to_mesh(me)
    bm.clear()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)


def guarded(bm, fn, allow_boundary_change=False):
    """Run fn(); if the mesh got torn (new holes, wire edges, >2 faces per edge), undo it.
    Returns (ok, result)."""
    snap = bm.copy()
    t0, b0 = _torn_edges(bm), _boundary_edges(bm)
    try:
        res = fn()
        ok = res is not False
        if ok:
            bm.normal_update()
            if _torn_edges(bm) > t0:
                ok = False
            elif not allow_boundary_change and _boundary_edges(bm) > b0:
                ok = False
            elif any(len(f.verts) < 3 or f.calc_area() < 1e-14 for f in bm.faces):
                ok = False
    except Exception:
        ok, res = False, None
    if not ok:
        _restore(bm, snap)
    snap.free()
    return ok, res
