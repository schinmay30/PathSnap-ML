import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_route_evaluation import (FEATURES, DirectedRoadGraph, RoadTrip, attach_reference,
                                  build_models, feature_table, predict_trip, summarize)


def test_unlabelled_track_to_paired_scores(tmp_path):
    folder = tmp_path / "01"
    folder.mkdir()
    (folder / "01.track").write_text("0 0 0\n0 0 9\n1 1 10\n0 0 11\n")
    (folder / "01.nodes").write_text("0 0\n0.001 0\n")
    (folder / "01.arcs").write_text("0 1\n1 0\n")
    trip = RoadTrip.read(folder)
    graph = DirectedRoadGraph(trip.nodes, trip.arcs)
    table = feature_table(trip)
    model = build_models()["extra_trees"].fit(table[FEATURES], table.candidate_arc_id)
    pred, _ = predict_trip(trip, graph, table, model)
    assert pred.status.tolist() == ["start", "gap_reset", "no_candidates", "restart_after_missing"]
    ref = pd.DataFrame(dict(trip_id="01", observation_id=range(4), arc_id=[0, 1, 1, 3],
                            label_valid=[True, False, True, True]))
    scores = summarize(attach_reference(pred, ref))
    for row in scores:
        assert row["matched_observations"] == 1 and row["valid_labels"] == 3
        assert row["candidate_recall"] == pytest.approx(1/3)
        assert row["prediction_coverage"] == .75
    with pytest.raises(ValueError, match="observation IDs"):
        attach_reference(pred, ref.iloc[:-1])
    trip.track = [(1, 1, 0)]
    empty = feature_table(trip)
    assert empty.empty
    assert predict_trip(trip, graph, empty, model)[0].status.tolist() == ["no_candidates"]
    incompatible = build_models()["extra_trees"].fit(table[FEATURES[:9]], table.candidate_arc_id)
    with pytest.raises(ValueError, match="19-feature"):
        predict_trip(trip, graph, table, incompatible)
