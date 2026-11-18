from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.viterbi import Candidate, DirectedRoadGraph, Observation, decode


def obs(i, *states, ts=None, lon=0.0):
    return Observation(i, lon, 0., i if ts is None else ts, [Candidate(*s) for s in states])


def test_backtracking_corrects_local_winner():
    g = DirectedRoadGraph([(0, 0), (.0001, 0), (1, 0), (1.0001, 0)], [(0, 1), (2, 3)])
    # Greedy picks arc 1 then arc 0. The complete path favours arc 0 at both fixes.
    path = decode(g, [obs(0, (1, .2, .6), (0, .2, .4)), obs(1, (0, .3, .99), (1, .3, .01))])
    assert [(m.arc_id, m.status) for m in path] == [(0, "start"), (0, "matched")]


def test_backward_position_requires_directed_cycle():
    nodes = [(0, 0), (.0001, 0)]
    fixes = [obs(0, (0, .8, 1)), obs(1, (0, .2, 1))]
    assert decode(DirectedRoadGraph(nodes, [(0, 1)]), fixes)[1].status == "transition_reset"
    assert decode(DirectedRoadGraph(nodes, [(0, 1), (1, 0)]), fixes)[1].status == "matched"


def test_directed_connector_offset_and_elapsed_budget():
    nodes = [(i * .0001, 0) for i in range(4)]
    def last_status(edges, dt):
        fixes = [obs(0, (0, .9, 1)), obs(1, (2, .1, 1), ts=dt)]
        return decode(DirectedRoadGraph(nodes, edges), fixes, max_speed_mps=10, gps_slack_m=0)[1].status
    # About 1.1 + 11.1 + 1.1 metres, including the unobserved middle arc.
    assert last_status([(0, 1), (1, 2), (2, 3)], 2) == "matched"
    assert last_status([(0, 1), (1, 2), (2, 3)], 1) == "transition_reset"
    assert last_status([(0, 1), (2, 1), (2, 3)], 2) == "transition_reset"


def test_gps_displacement_changes_selected_position():
    g = DirectedRoadGraph([(0, 0), (.0002, 0), (.0004, 0)], [(0, 1), (1, 2)])
    first = obs(0, (0, 0, 1))
    states = [(0, 0, .55), (1, 0, .45)]
    assert decode(g, [first, obs(1, *states)])[1].arc_id == 0
    assert decode(g, [first, obs(1, *states, lon=.0002)])[1].arc_id == 1


def test_zero_lengths_missing_gap_and_ties():
    g = DirectedRoadGraph([(0, 0), (0, 0), (0, 0)], [(0, 1), (1, 2), (1, 2)])
    rows = [obs(0, (0, .9, 1)), obs(1, (2, .5, 0), (1, .5, 0)), obs(2),
            obs(3, (2, 0, .5)), obs(4, (0, 1, 1), ts=10)]
    matches = decode(g, rows)
    assert [m.observation_id for m in matches] == list(range(5))
    assert [m.arc_id for m in matches] == [0, 1, None, 2, 0]
    assert [m.status for m in matches] == [
        "start", "matched", "no_candidates", "restart_after_missing", "gap_reset"]
    assert decode(g, []) == []
    with pytest.raises(ValueError, match="Duplicate candidate"):
        decode(g, [obs(0, (0, 0, .5), (0, 0, .5))])
    with pytest.raises(ValueError, match="strictly increase"):
        decode(g, [obs(0), obs(1, ts=0)])
