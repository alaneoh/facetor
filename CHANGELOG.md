# Changelog

## 1.0.2 – 2026-11-27

Fixes: more kinds of parts are handled, and cleaner results around them.

- A ring held by a band of triangles towards a wider loop (the foot of a flared
  boss on a cylinder top): the triangles are now cut again evenly around the
  new ring, whatever the count. Fewer sides welded vertices together and gave
  uneven, thin triangles; more sides mixed quads into the band.
- A ring held by fans from a few far-off vertices can now go below its number
  of held vertices (they are welded into the kept ones); a count so low that
  the faces around would fold over is refused with a message.
- Lines of edges without faces (a profile drawn for a Spin) work like rounded
  corners on an outline: a single straight edge between two others becomes a
  Bevel-like round with more segments, and Keep / Round / Shape / width apply.
- Slotted screw heads whose slot walls are triangulated (walls touching the
  dome at a single vertex) were refused: they are now rebuilt as parts of
  revolution with a recess (sides and profile rings).
- Domes cut by a bar: a coarse dish (rings of only a few vertices, long CAD
  triangles across them, a bar resting on it along a ring, an open rim) is now
  recognised, with every side count (odd ones too) and profile rings. The dish
  can be selected together with the bar: the bar is kept as it is.
- Fixed: UVs of unselected parts of the same object far from the change (another
  piece) were recomputed in some modes, which broke them.
- After changing the fillet of a part, the rest of the part stays selected, so
  S works on the whole part straight away.

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
