# Where FG² and PanoRoMa fail (full Chicago same-area test split, 12,739 panoramas)

`make failure-viz` (scripts/viz_failure_cases.py), frames from `fail_cases.json` plus the 2 "both fail" frames with
the largest common error min(PanoRoMa, FG²). PanoRoMa D (two-pass, se2, bf16 decoder, 6 m gate) and the released
FG² same-area checkpoint were re-run on these frames (Eagle job 8890308, `compute_8890308.log`). Every recomputed
error equals the full-split file: PanoRoMa |d| = 0 against `eval_vigor_full_chicago_pano_D_samearea.json`; FG² within
4e-6 m against `ransac_m` of `eval_fg2_full_chicago_fg2_samearea.csv`. For FG², the frame's whole batch of 24 from the
full run was re-run with the CUDA RNG offset of that batch, and all 24 frames of each such batch matched. FG²'s
ground truth, converted to our tile frame, equals ours to the centimetre.

Figures: left, the panorama (brightened); right, the VIGOR tile (north up, metres from the tile centre). Green + =
ground truth, red x = PanoRoMa final, blue triangle = FG², with lines from GT to each prediction. Dots = 160 of
PanoRoMa's depth-placed tokens (filled = RANSAC inlier), drawn on the panorama and on the tile where PanoRoMa's
final pose puts them, coloured by azimuth.
"Δpred" is the distance between the two methods' predictions.

Overall rates (from the two files): both < 5 m on 90.3 %, FG² > 5 m on 5.3 %, PanoRoMa > 5 m on 8.1 %, both > 5 m on
3.6 %.

## FG² fails, PanoRoMa works (`fg2_fails/`)

| # | panorama | PanoRoMa | FG² | Δpred | what probably went wrong |
|---|---|---|---|---|---|
| 1 | E5-fxqKdtqohJjgBSDBP2A | 1.15 m | 34.50 m | 33.4 m | Under a highway overpass next to construction barriers. The tile shows the overpass deck, not the ground below it. FG² jumps to the far side of the expressway, almost point-symmetric to the GT through the tile centre. PanoRoMa holds on to the barrier and road edge near the camera. |
| 2 | jGzQImBrHXF8sUD3WwA0Hw | 0.68 m | 16.61 m | 16.4 m | Multi-lane expressway with a uniform lane pattern. FG² slides 16 m east along the road; PanoRoMa's near-ground tokens (lane marks, shoulder) pin it. |
| 3 | 1BXlokrs7aE2xfI1l8CIwQ | 1.51 m | 9.61 m | 8.6 m | Wacker Drive by the river, among tall glass towers. The tile is strongly off-nadir (buildings lean across the road). FG² slides about 9 m SW along the street. |
| 4 | N5na-NWJTmViOVspOpiDwg | 0.98 m | 5.04 m | 4.3 m | Parking lot. FG² is only just over the 5 m line, about 5 m south; the parked cars differ from the tile. Borderline, not a real failure. |

## PanoRoMa fails, FG² works (`pano_fails/`)

| # | panorama | PanoRoMa | FG² | Δpred | what probably went wrong |
|---|---|---|---|---|---|
| 1 | Y9eWNNRHVVleDul0suJTUw | 35.50 m | 1.30 m | 36.8 m | Park (curving paths and a splash pad) under tree canopy. The canopy hides most paths on the tile, and the ground tokens see only paved path and lawn, which repeat. PanoRoMa locks onto another path junction 35 m NE, already in the coarse pass (34 m). |
| 2 | Gnh-0xvD_DX227ndxKsN4g | 14.76 m | 1.62 m | 13.4 m | On a steel truss bridge deck. Overhead trusses cover the upper half of the view and are placed by depth as if they were on the ground. The bridge looks the same along its axis, and PanoRoMa slides 15 m north along it. |
| 3 | KUz__VKFOcVBDm9aY3amKQ | 7.83 m | 1.06 m | 7.1 m | Loop street canyon. On the tile the street is hidden under a leaning roof (off-nadir view, so the GT appears on a rooftop). PanoRoMa slides 8 m south along the street. |
| 4 | -oxRcnPAwIk7xdTy74oASg | 5.01 m | 1.11 m | 4.0 m | Plain street next to a parking lot. PanoRoMa is 5 m north along the street, on the threshold; nothing fixes the position along the street. Borderline. |

## Both fail (`both_fail/`; 5 and 6 are the extra frames with the largest common error)

| # | panorama | PanoRoMa | FG² | Δpred | what probably went wrong |
|---|---|---|---|---|---|
| 1 | wRT0regD2jAHAUDUUs0EgA | 36.87 m | 38.89 m | 2.1 m | The panorama is at a street intersection, but the GT lies on a building roof 35 m south of the only street in the tile. Both methods agree within 2 m on that street, so the label/GPS position is most likely wrong. |
| 2 | -xqRG7MYYMOGiITk7fGnxQ | 20.78 m | 19.39 m | 3.2 m | Loop intersection between tall buildings. Both predict the same point 20 m west, near the neighbouring street/crosswalk. Street canyons repeat, and building lean hides the street surface. Could also be a label offset; unclear. |
| 3 | uk0RPFvOq6Os72VsDw_3yQ | 14.29 m | 13.47 m | 3.5 m | Corner by a glass tower. The GT is in deep building shadow next to tree canopy; both methods agree on the open road 13 m east. Unclear: either a label offset or the shadowed GT area has nothing to match. |
| 4 | S5JUAgiweJNt_27KOrKApQ | 12.39 m | 10.00 m | 8.1 m | The panorama is taken on the water (boat on the Chicago River), with bridges to the S and E, but the GT is on the bridge deck. Both predict open water north of the bridge, which fits the panorama better than the GT does. Label error, most likely. |
| 5 | ZJZp4VCr5oxlMT2WzfRqkw | 36.83 m | 36.58 m | 1.2 m | Expressway with an overpass ahead. Both predict the same spot 37 m SE along the highway (PanoRoMa's second pass was gated off, so this is its coarse pose). The highway looks the same along its length, but agreement within 1.2 m suggests a real cue there (the sign gantry or overpass) or a label offset. Unclear. |
| 6 | NEERhpUONAYHhrPYNHUDMw | 36.33 m | 40.46 m | 9.4 m | Chicago Riverwalk (lower level, garden path, river to the N). The GT is in the middle of a raised circular plaza at street level. Both put the camera on the riverwalk path 35 m south, which is what the panorama shows. A multi-level scene: most likely a label error. |

## Failure types per category

- **FG² fails, PanoRoMa works:** in all four frames FG² slides along a repetitive linear structure (expressway lanes, a
  street under leaning towers) or flips to the far side of a roughly symmetric scene (frame 1 is almost point-mirrored
  through the tile centre). PanoRoMa keeps the position because its depth-placed tokens next to the camera (barriers,
  lane marks, kerbs) carry the fine detail. One of the four (5.04 m) is only just over the threshold.
- **PanoRoMa fails, FG² works:** in three of four frames the ground near the camera is not visible on the tile or
  looks the same elsewhere: tree canopy over a park, trusses over a bridge deck, a street canyon covered by leaning
  roofs. PanoRoMa then locks onto a similar-looking spot, already in the coarse pass (coarse errors 34 / 12 / 7.5 m). The
  bridge frame also shows a PanoRoMa-specific weakness: overhead structure is placed by depth as if it were on the
  ground. The fourth frame (5.01 m) is borderline.
- **Both fail:** in 5 of 6 frames the two methods, which work very differently, put the camera within 1–4 m of each other
  (8–9 m for the river and Riverwalk frames) and 10–40 m from the label. In three of them (rooftop GT, GT on a bridge
  for a boat panorama, GT on an upper-level plaza for a Riverwalk panorama) the panorama plainly contradicts the label.
  A good share of the "both fail" tail is therefore probably VIGOR label/GPS error, not method error. The rest are
  street-canyon or highway scenes that repeat along the road, where building lean hides the street. This suggests the
  3.6 % "both > 5 m" rate is partly a floor set by the labels.
