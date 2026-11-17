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

bl_info = {
    "name": "Facetor",
    "author": "Alaneo Houssart",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "Edit Mode > Face menu (Ctrl+F) / Edge menu (Ctrl+E) > Facetor, or Alt+B",
    "description": "Change the number of segments of cylinders, holes, bevels, fillets and other round shapes while keeping their silhouette",
    "category": "Mesh",
}

"""Facetor: change the number of segments of any round shape, keep its silhouette.

Modules:
    core        shared helpers: quad grids, curve fitting, profiles, circles
    grid        clean quad grids, fillets inside longer profiles
    safety      every change is checked and undone if it tears the mesh
    strips      messy CAD strips, ribbons, stacked strips, edge chains
    outline     rounded corners on the outline of a flat n-gon
    holes       holes in flat or curved areas
    revolution  parts of revolution in several pieces or cut by a detail
    recess      parts of revolution with a convex recess
    attributes  UVs, seams, sharp edges carried over; optional triangulation
    selection   which faces to process and how
    modal       the interactive operator (Alt+B), status bar, menu entry
"""

import bpy

from .modal import MESH_OT_facetor, menu_func


classes = (MESH_OT_facetor,)
addon_keymaps = []


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_edit_mesh_faces.append(menu_func)
    bpy.types.VIEW3D_MT_edit_mesh_edges.append(menu_func)
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon if wm else None
    if kc:
        km = kc.keymaps.new(name="Mesh", space_type="EMPTY")
        kmi = km.keymap_items.new(MESH_OT_facetor.bl_idname, "B", "PRESS", alt=True)
        addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()
    bpy.types.VIEW3D_MT_edit_mesh_edges.remove(menu_func)
    bpy.types.VIEW3D_MT_edit_mesh_faces.remove(menu_func)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
