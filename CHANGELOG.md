# Changelog

## 1.0.1 – 2026-11-21

Fixes for boxes bevelled on all their edges (fillets meeting at corners), and
finer control.

- Width: the corners now scale around the box's real corner. Before, a wider
  bevel pushed the flat faces outwards (the box grew) and width 0 didn't give
  the exact sharp box back.
- Shape now follows Blender's Bevel scale: 1 is a square corner, 0 a square
  step inwards (it stopped short of both before). The corners follow the
  chosen shape: they were pinched as soon as it wasn't round, and very concave
  shapes failed or jumped near 0. Shape 0 makes Bevel's square steps; such a
  part can't be changed again afterwards (undo to go back).
- A second Alt+B on a box whose shape was changed rebuilt its corners with
  folds: fixed.
- Alt+B starts the shape from the part's own, so P and R don't jump back to a
  round profile.
- Shift while moving the mouse: width and shape change ten times slower.
- Faster: changing the width or the shape with the mouse no longer looks
  again for what kind of part is selected at each move (about five times
  faster on a bevelled box).

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
