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

"""Selection analysis: which faces to process and how."""

import collections
import math

from .core import (
    _face_components,
    auto_direction,
    direction_stats,
    extract_grid,
    fit_circle,
    quad_components,
    quad_corners,
    ordered,
)
from .grid import resample_bands


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def grid_faces_of(faces):
    """The quads of `faces`, and the n-gons that are quads with T-junction vertices on the
    border of `faces` (see quad_corners)."""
    fset = set(faces)
    return [f for f in faces if len(f.verts) == 4 or (len(f.verts) > 4 and quad_corners(f, fset))]


def gather_faces(bm):
    quads = grid_faces_of([f for f in bm.faces if f.select])
    if not quads:
        quads = [f for f in bm.faces if len(f.verts) == 4 and all(v.select for v in f.verts)]
    return quads


def analyse(bm, swap, faces=None):
    """Return (grids [(grid, direction)], errors) for the current selection (or the given faces)."""
    quads = gather_faces(bm) if faces is None else grid_faces_of(faces)
    active = bm.faces.active if bm is not None else None
    grids, errors = [], []
    for comp in quad_components(quads):
        g, err = extract_grid(comp, active)
        if err:
            errors.append(err)
            continue
        dirn = auto_direction(g)
        if not any(g.cyclic) and min(g.n) > 2:
            # rounded corner of a box (a fillet patch and the round wall under it): the profile
            # holds a fillet plus a straight part, the other direction is all one arc -> by
            # default the corner's sides change, S gives the fillet
            other = 1 - dirn
            bands = resample_bands(g, dirn)
            if (bands and bands[0] is not None and resample_bands(g, other) == [None]
                    and direction_stats(g, other)[0] * direction_stats(g, other)[1] > math.radians(30)):
                dirn = other
            else:
                # rounded corner of a frame made of a wall and a chamfer (CAD): the profile is
                # two flat faces at an angle (a crease, nothing round), the other direction is
                # the round corner -> the corner's segments change
                s_d, t_d, _ = direction_stats(g, dirn)
                s_o, t_o, _ = direction_stats(g, other)
                if s_d <= 2 and s_o >= 4 and s_o * t_o > math.radians(30) and t_d > 2 * t_o \
                        and (bands == [None] or not bands):
                    dirn = other
        if swap:
            dirn = 1 - dirn
        grids.append((g, dirn))
    return grids, errors


def band_core(faces):
    """Clean part of a band whose ends are messy (CAD corners with triangles and poles): the face
    columns that cross the whole band, from one long side to the other. None if there is none."""
    S = {f for f in faces if len(f.verts) == 4}
    if len(S) < 4:
        return None
    border = [e for f in ordered(S) for e in f.edges if sum(1 for g in e.link_faces if g in S) == 1]
    strips = []
    for e in border:
        f = next(g for g in e.link_faces if g in S)
        strip, cur, ok = [], e, False
        while f is not None and f in S and f not in strip:
            strip.append(f)
            lp = next((l for l in f.loops if l.edge is cur), None)
            if lp is None:
                break
            cur = lp.link_loop_next.link_loop_next.edge
            others = [g for g in cur.link_faces if g is not f]
            if not any(g in S for g in others):
                ok = True
                break
            f = others[0] if len(others) == 1 else None
        if ok:
            strips.append(strip)
    if not strips:
        return None
    lengths = collections.Counter(len(st) for st in strips)
    L = max(lengths, key=lambda n: (lengths[n], -n))
    if lengths[L] < 4:
        return None
    keep = {f for st in strips if len(st) == L for f in st}
    comps = _face_components(ordered(keep))
    core = max(comps, key=len) if comps else []
    if len(core) < 4 or len(core) == len(S):
        return None
    grids, errs = analyse(None, False, core)
    if errs or len(grids) != 1:
        return None
    return core


def find_round_part(bm, grids):
    """Largest round n-gon touching the selection whose rim isn't simply the end of a closed tube."""
    closed_verts = {v for g, _ in grids if any(g.cyclic) for v in g.V.values()}
    sel = [f for f in bm.faces if f.select] or gather_faces(bm)
    sel_verts = {v for f in sel for v in f.verts}
    cands = {f for v in sel_verts for f in v.link_faces if len(f.verts) >= 8}
    best = None
    for f in sorted(ordered(cands), key=lambda f: -len(f.verts)):
        if all(v in closed_verts for v in f.verts):
            continue
        if not f.select and sum(1 for v in f.verts if v in sel_verts) < 0.5 * len(f.verts):
            continue  # only touches the selection: it isn't the part being edited
        fit = fit_circle([v.co for v in f.verts])
        if fit and fit[3] < 0.01 * fit[2]:
            best = f
            break
    return best


def tri_count(bm):
    return sum(len(f.verts) - 2 for f in bm.faces)
