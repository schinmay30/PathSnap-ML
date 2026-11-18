import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.candidate_features import additional_features


def test_geometry_motion_topology_and_rivals():
    nodes = [(0, 0), (0.001, 0), (0.002, 0)]
    arcs = [(0, 1), (0, 1), (1, 2), (1, 1), (1, 0)]
    track = [(0, 0, 0), (0.0005, 0, 1), (0.0005, 0, 2), (0.001, 0, 20)]
    candidates = pd.DataFrame({"observation_id": [1, 0, 1, 2, 3, 1, 1],
                               "candidate_arc_id": [0, 0, 1, 0, 0, 3, 4],
                               "dist_m": [0, 0, 0, 0, 0, 55.6, 0]}, index=range(9, 16))
    x = additional_features(candidates, track, nodes, arcs)
    assert x.index.equals(candidates.index)
    assert x.loc[9, "nearest_endpoint_m"] == pytest.approx(55.6, abs=0.1)
    assert x.loc[9, "previous_step_m"] == pytest.approx(55.6, abs=0.1)
    assert x.previous_step_available.tolist() == [1, 0, 1, 1, 0, 1, 1]
    assert x.previous_bearing_available.tolist() == [1, 0, 1, 0, 0, 0, 1]
    np.testing.assert_allclose(x.previous_bearing_cos, [1, 0, 1, 0, 0, 0, -1])
    assert x.loc[14, "zero_length_arc"] == 1
    assert x.loc[9, "exit_out_degree"] == 2
    assert x.loc[15, "entry_in_degree"] == 1
    assert x.nearest_runner_up_gap_m.eq(0).all()
    assert x.has_runner_up.tolist() == [1, 0, 1, 0, 0, 1, 1]
    candidates["label"] = np.arange(len(candidates))
    pd.testing.assert_frame_equal(x, additional_features(candidates, track, nodes, arcs))


def test_separation_and_invalid_inputs():
    rows = pd.DataFrame({"observation_id": [0, 0], "candidate_arc_id": [1, 0], "dist_m": [4., 1.]})
    args = ([(0, 0, 0)], [(0, 0), (1, 1)], [(0, 1), (1, 0)])
    assert additional_features(rows, *args).nearest_runner_up_gap_m.tolist() == [3., 3.]
    with pytest.raises(ValueError, match="Duplicate candidate"):
        additional_features(pd.concat([rows, rows]), *args)
    with pytest.raises(ValueError, match="indices"):
        additional_features(rows.assign(candidate_arc_id=-1), *args)
    with pytest.raises(ValueError, match="finite"):
        additional_features(rows, [(np.nan, 0, 0)], *args[1:])
