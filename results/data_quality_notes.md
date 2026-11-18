# Data-quality notes and assumptions (Person 1)

Partitions analysed: `train` (80 trips) and `student_test` (10 trips). The evaluator partition is never read.
Reproduce every number below with `scripts/explore_dataset.py` and `scripts/build_labels.py`
(tables in `results/`, figures in `results/figures/`).

## How the data is organised
- One folder per trip with `.track`, `.nodes`, `.arcs`, `.route`. Node and arc IDs are row indices
  **within that trip's own road network**, so arc 57 in two trips is two different roads.
- The `.route` file lists arcs in travel order and is much shorter than the GPS track
  (for example trip 00000001: 1,315 GPS rows, 525 route arcs). It is not one label per GPS row.
- Train: 208,417 GPS observations. Student test: 19,160.

## GPS quality
- No missing or non-finite values and no duplicate rows in any trip. Timestamps strictly increase.
- Sampling is nominally 1 Hz: the median share of exactly-1 s steps is 0.991 (train, minimum 0.531)
  and 0.999 (student test, minimum 0.861).
- Gaps over 5 s: 45 of 80 train trips and 2 of 10 student-test trips. The largest are train trip
  00000090 (153,064 s) and 00000078 (52,810 s); the next is 2,658 s. These look like trip breaks and
  should be handled as resets by the feature code and the decoder.
- Steps implying more than 60 m/s (an arbitrary plausibility threshold): 546 steps in 18 train trips and
  2 steps in 2 student-test trips. Heaviest: 00000032 (358 steps), 00000013 (103), 00000001 (52).

## Road networks
- No isolated nodes in any trip.
- Networks are fragmented: 79 of 80 train and 10 of 10 student-test networks have more than one weakly
  connected component (median 20 train, 11 student test; maximum 421 train). The largest component holds
  at least 69% of nodes in train and 87% in student test.
- In all 90 trips the reference route lies inside a single weakly connected component.
- Route chaining: in 11 trips (10 train, 1 student test) 14 consecutive route-arc pairs do not connect
  (head of one arc is not the tail of the next). 24 trips contain immediate reverse-direction pairs (54 in total).
- Duplicate (from, to) arcs: 1,775 in train networks, 128 in student test. Self-loop arcs: 22 and 2.

## Reference-route alignment (labels)
`pathsnap.labels.align_trip_to_route` assigns each GPS row to a route arc by monotone dynamic programming:
cost is the distance (m) to the arc, capped at 100 m; the route index may stay or move forward but never
backwards; skipping an arc costs 1 m. Coordinates are projected to local metres per trip.
A row is `label_valid` when its aligned arc is within 50 m. Both numbers are my own choices, not course limits.

- Student test: 99.86% of observations valid. Train: 88.9%, or 98.6% excluding trips 00000033 and 00000054.
  66 of 90 trips are 100% valid.
- Invalid rows come in contiguous runs with small spatial spread, not as a constant offset. Trip 00000054:
  its longest run is 15,584 consecutive points with an RMS spread of about 66 m and a median distance of
  140 m from the route (a plot of a mid-trip slice shows a tight cluster away from the route). Trip 00000033:
  a 3,042-point run with an RMS spread of about 15 m. This is consistent with long stationary periods away from the mapped route, not a shifted
  reference route. These two trips hold 25,157 of the 208,417 train observations (12%).
- A median of 95.8% of route arcs receive at least one valid observation (minimum 71%); very short arcs can fall
  between 1 Hz samples.

## Decisions and caveats
- Invalid rows are dropped from labelled data, never labelled 0, so training does not learn from unreliable labels.
- Consider capping or down-weighting very long dwell trips so they do not dominate cross-validation.
- The reference route is reference information, not independently verified truth. The alignment inherits any
  route error, and the 50 m / 100 m / 1 m parameters were not tuned against anything.
