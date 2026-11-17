# Changelog

## 1.0.0 – 2026-11-17

First public release.

- Interactive segment change (Alt+B) for cylinders, tubes, tori, holes, bevels,
  fillets, parts of revolution and domes, from faces or edge loops.
- Fillet networks: boxes bevelled on all their edges and corners where several
  fillets meet are rebuilt like Blender's Bevel.
- Width (W), profile type (R: Keep / Round / Custom) and shape (P) like Blender's
  Bevel tool; width 0 gives the sharp edge back, chamfers are rounded when their
  segments go up.
- Values can be typed on the keyboard; both directions (S) are adjustable in the
  redo panel.
- Handles CAD topology: n-gon strips, ribbons, fans, T-junctions, slots and cross
  recesses, domes cut by other geometry.
- UVs and edge marks (seams, sharp, crease, bevel weight) carried over.
- Changes that would tear the mesh are refused with a message.
