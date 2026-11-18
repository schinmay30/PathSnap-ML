import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_decoder_ablation import partition_ids
from pathsnap.viterbi import Candidate, DirectedRoadGraph, Observation, decode


def test_soft_jump_competes_with_feasible_path_and_flags_only_chosen_link():
    graph = DirectedRoadGraph([(0, 0), (.0001, 0), (1, 0), (1.0001, 0)], [(0, 1), (2, 3)])
    obs = [Observation(i, 0, 0, i, [Candidate(j, .5, p) for j, p in enumerate(ps)])
           for i, ps in enumerate([[.999, .001], [.001, .999], [.001, .999]])]
    assert len({m.arc_id for m in decode(graph, obs)}) == 1
    matches = decode(graph, obs, unreachable_penalty=2.)
    assert [m.arc_id for m in matches] == [0, 1, 1]
    assert [m.status for m in matches] == ["start", "soft_jump", "matched"]
    assert len({m.arc_id for m in decode(graph, obs, unreachable_penalty=12.)}) == 1
    assert decode(graph, obs, unreachable_penalty=12., emission_weight=3.) == matches


def test_displacement_reset_and_negative_penalty():
    graph = DirectedRoadGraph([(0, 0), (.001, 0)], [(0, 1)])
    obs = [Observation(i, i*.001, 0, i, [Candidate(0, .5, 1)]) for i in range(2)]
    assert decode(graph, obs, displacement_reset_m=60)[1].status == "displacement_reset"
    with pytest.raises(ValueError):
        decode(graph, obs, distance_deadband_m=-1)


def test_selection_trip_isolation():
    fold = dict(fit_trips=list(range(40)), held_out_trips=list(range(40, 80)))
    fit, inner, held = partition_ids(fold, 43)
    assert len(inner) == 8 and len(fit) == 32
    assert set(fit).isdisjoint(inner) and set(held).isdisjoint(fit + inner)
    assert (fit, inner, held) == partition_ids(fold, 43)
