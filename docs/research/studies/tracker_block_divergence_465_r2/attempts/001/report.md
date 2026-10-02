# tracker_block_divergence_465_r2 attempt report (generated; exploratory, not citable)

terminal: **FIRST_DIVERGENCE_ASSOCIATION** (rule over R_T first-divergence blocks, declaration §4)

## First structural divergence against R_C

`first` = earliest (frame, block); per-block first frame and divergent-frame counts follow. `ti@first` = tracker_input structural / bit equality at that frame.

| run | seq | first | A first (n) | P first (n) | E first = f* (n) | ti@first struct/bit | ti struct frames before | GMC row mismatch |
|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| R_T#1 | 02 | 222/A | 222 (6) | 283 (310) | 283 (307) | diff/diff | 10 | 0 |
| R_T#1 | 04 | 2/A | 2 (14) | 15 (1036) | 3 (1048) | same/diff | 0 | 0 |
| R_T#1 | 05 | 15/A | 15 (7) | 15 (38) | 15 (12) | same/diff | 0 | 0 |
| R_T#1 | 09 | 189/A | 189 (3) | 189 (209) | 319 (207) | same/diff | 2 | 0 |
| R_T#1 | 10 | 3/A | 3 (1) | 3 (652) | 6 (648) | same/diff | 0 | 0 |
| R_T#1 | 11 | 3/A | 3 (9) | 64 (66) | 49 (35) | diff/diff | 0 | 0 |
| R_T#1 | 13 | 7/A | 7 (1) | 7 (744) | 14 (736) | same/diff | 1 | 0 |
| R_E | 02 | 469/A | 469 (1) | — (0) | — (0) | diff/diff | 0 | 0 |
| R_E | 04 | — | — (0) | — (0) | — (0) | — | — | 0 |
| R_E | 05 | — | — (0) | — (0) | — (0) | — | — | 0 |
| R_E | 09 | 189/A | 189 (1) | 189 (1) | — (0) | same/diff | 1 | 0 |
| R_E | 10 | — | — (0) | — (0) | — (0) | — | — | 0 |
| R_E | 11 | — | — (0) | — (0) | — (0) | — | — | 0 |
| R_E | 13 | 646/A | 646 (1) | — (0) | — (0) | diff/diff | 0 | 0 |

Per-divergence details (association records, state-transition keys, emission ids) are in result.json.
