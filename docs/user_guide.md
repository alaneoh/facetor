# Facetor – User guide (version 1.0.2)

Blender add-on that changes the number of segments of a round shape while
keeping its silhouette: sides of a cylinder or a hole, segments of a bevel
or a fillet. Works on any mesh, and is especially handy on CAD exports.
It can reduce or increase the number of segments.

## Installation (Blender 4.2 or later)

**From Blender:**
Edit > Preferences > Get Extensions, search for **Facetor**, click Install.

**From a file:**
1. Edit > Preferences > Get Extensions > arrow at the top right > Install from Disk.
2. Pick `facetor-1.0.2.zip`. You can also drag the .zip into Blender.

If an older version is installed, remove it and restart Blender first,
otherwise both fight over Alt+B.

## Usage

1. In Edit Mode, select the faces of the shape.
   - **Hole:** select only the hole's edge loop (edge mode + Alt+click).
   - **Tube:** you can also Alt+click one ring of it (edge mode): its sides
     change, S gives its rings.
   - **Fillet:** you can also select only the edges of one curve (an open
     chain). Facetor takes the strips of faces running along it, up to the
     large flat faces.
   - **Rounded corner on a flat face** (e.g. a plane bevelled with Blender's
     Bevel in vertex mode): select the vertices or edges of the rounded
     corner. Several corners at once is fine.
2. Press **Alt+B** (or Ctrl+F > Facetor, Ctrl+E > Facetor, or F3 > Facetor).
3. Adjust:

| Input | Action |
|---|---|
| Mouse wheel | +1 / -1 segment |
| Ctrl + wheel | next / previous multiple of 4 |
| 0-9, . | Type a value (top row or numpad, Backspace to correct), applied as you type; Enter or click confirms. It sets the segments, or the width after W, or the shape after P |
| S | Switch direction (see below) |
| R | Profile Type: Keep / Round / Custom |
| W | The mouse sets the bevel width, in Keep too (the shape is scaled around its corner). Width 0 merges the bevel back into a sharp edge. |
| Mouse move | Bevel width (in Round / Custom, or after W). Hold **Shift** for ten times finer steps (width and shape). |
| T | Triangulate: the n-gons left by the change (around a hole, caps, recess walls) and the old triangles that got a vertex more become triangles. Remembered for next time. |
| P | Shape: the mouse sets the shape of the round profile instead of the width (switches to Round), on the same scale as Blender's Bevel: 0.25 = flat (chamfer), 0.5 = round, 1 = square corner, below 0.25 concave, 0 = square step inwards. It starts from the shape the part already has. Press P again to go back to the width. |
| Left click / Enter | Confirm |
| Right click / Esc | Cancel |

**S – switch direction.** On a cylinder with a fillet: number of sides ↔
segments of the fillet. Only the fillet changes, the straight parts keep
their loops. On a part of revolution (a rod with a rounded tip, a domed
screw with a recess): number of rings of the rounded part; the recess is
kept.
What you set before pressing S is kept (e.g. 32 sides, then S for the
fillet). If nothing was set before S, nothing is kept.
Alt+B always starts in the main direction, in Keep and width 1 (nothing is
carried over from the last time). The shape starts from the part's own: fillets
already set to 0.85 start at 0.85 when you press P or R.

**Chamfers.** A chamfer (one flat segment between two faces) has no curve
to keep: going up from 1 segment rounds it, like Blender's Bevel. Select the
chamfer's face, or one or both of its end edges.

**R – profile type.** Custom uses the custom profile of Blender's Bevel tool. By
default it is a straight line, so the rounded part becomes flat (this is
expected). To change it: Bevel tool > Profile Type Custom.

4. After confirming, the Adjust Last Operation panel (bottom left) still lets
   you tweak the values.

## What it handles

- Cylinders, tubes, bent pipes (select the ring or the whole object).
- Bevels and fillets (select the bevel faces).
- Round parts with a slot or a recess, flat or domed, like a screw head
  (select the whole part): every ring gets the same number of sides and the
  recess stays in place. Cross recesses work too.
- CAD strips with n-gons (if it is a straight extrusion). Width scales each
  end around its corner when the strip is a fillet between two faces.
- Parts of revolution: width scales each rounded part of the profile around
  its corner (it needs a straight part on each side).
- One-face-wide ribbons following a curve, even open ones (e.g. a curved
  wall on a floor): the last face counts as part of the curve.
- One-face-wide CAD walls, even slightly twisted (e.g. the back of an
  S-shaped grip).
- Several strips side by side along the same curve, cut differently (e.g.
  a 12-face wall and a 20-face chamfer): they all get the same count and
  the columns line up.
- Curves sitting on a flat face cut up by CAD (triangle fans from the
  corners, or strips): fan triangles are reduced along with the curve; a
  flat area cut in strips becomes a single flat n-gon.
- Vertices where other faces attach (T-junctions) stay in place: the
  neighbouring faces don't move.
- Fillets whose two ends are messy CAD corners (triangles, poles), like
  the band around the end of a grip: select the whole band. The clean part
  is rebuilt, the corners are kept as they are and joined with small
  n-gons. The corners still need manual cleanup.
- Rounded corners on a flat face (n-gon): number of segments, width (in Keep
  the curve is scaled around its corner; width 0 gives the sharp corner back)
  and, in Round, the profile, like Blender's Bevel. A line of edges without
  faces (a profile drawn for a Spin) works the same way: select the edges of
  the corner, even a single straight edge, which rounds like a chamfer.
- Holes in a flat face or a curved area, with or without a wall. If the flat
  area around the hole is cut into triangle fans (CAD), the fans are joined
  into n-gons so the hole can lose sides; press T to get triangles back.
  Width changes the hole's radius (the wall follows); it can't reach the
  edges around the hole.
- Domes cut by geometry you don't select (a dish crossed by a raised bar,
  a dish whose rim is cut by a step): select the dome's faces (a bar
  standing on the dish may be selected with it: it is kept as it is). They
  change together as one dome (S: rings of the profile); the cut edge stays
  exactly where it is and the faces next to it become n-gons where needed.
  Under 8 sides the two halves share the count (an odd count gives one more
  segment to one half).
- A box bevelled on all its edges (corners where three fillets meet, made by
  Blender's Bevel): select every fillet. They all change, and each corner is
  rebuilt the same way as Bevel does it, on the corner's sphere (or on the
  surface of the chosen shape). Width works too (0 gives the sharp box back).
  Shape 0 makes square steps, like Bevel: the corners disappear and the part
  can't be changed again afterwards (undo to go back). Corners where 4, 5 or more fillets meet
  work the same way, and so do fillets with different segment counts: they
  all get the new count.
- A whole torus, both directions (also from an edge loop around it).
- Fillets around a rounded box (a button): one edge chain across a straight
  side, the whole fillet, or one corner (S: the fillet segments). A last
  fillet segment notched by cut-outs, selected with it, is kept as it is.
  On an edge chain between n-gons, S sets the number of columns of the
  straight side between the two lines next to the chain.
- Parts of revolution split into pieces with different side counts (e.g. a
  51-sided tip and a 52-sided rod): select everything to get the same count
  everywhere, or select only the piece to fix and give it the other piece's
  count.

## Tips

- Same number of segments on several parts: select them all, click the
  reference part last, then Alt+B.
- UVs, seams, sharp edges, bevel weights and creases are carried over to the
  new geometry. The UVs of new faces are interpolated from the old ones: a
  quick check (and a re-unwrap if needed) is still a good idea on parts that
  change a lot.
- Select the whole strip, otherwise the neighbouring faces become n-gons.
- If a message mentions duplicate vertices: M > By Distance, then try again.

## Troubleshooting

- **Alt+B does nothing and no message shows:** another add-on probably uses
  the same shortcut, or an older version of Facetor is still installed. Keep
  a single version, disable the other add-on (or change the shortcut in
  Preferences > Keymap), then restart Blender. Ctrl+F > Facetor always works.
- **An error message shows:** nothing was changed. Try another number of
  segments, or do that area by hand.
- **"use N segments or more":** the edge of the selection is attached to
  other faces (triangle fans) in too many places to go lower.
- **Not handled yet:** threads; corners from CAD exports (made of triangles)
  are kept as they are; a network whose corners all join exactly four
  fillets (no three-way or five-way corner) isn't recognised as one.
- **"The width can't change on ...":** that mode keeps its shape (recesses,
  domes cut by a bar, round parts with details...): set Width back to 1.

## Reporting a bug

Send the error message, a screenshot of the selection, the add-on version
and, if possible, the .blend file in the state just before the problem.
