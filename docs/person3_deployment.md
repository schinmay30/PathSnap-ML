# Person 3 deployment package v1

Place the versioned classifier and matching Viterbi JSON together in `artifacts/final/`. Use Python and numerical library versions from the manifest, with the repository's `src` on the backend process's Python path. The JSON doubles as the release manifest and the decoder configuration. Only load trusted server-local artifacts.

Muruli's existing registry discovers the pair using the following backend imports. Cache the result once at application startup. The loader verifies the classifier hash, scikit-learn version, ordered features and class order.

```python
from models.loader import find_model_artifact, find_hmm_artifact
from pathsnap.deployment import load_package
from pathsnap.route_inference import RoadTrip

dirs = [settings.artifacts_final_dir, settings.artifacts_models_dir, settings.backend_models_dir]
binary = find_model_artifact("final", dirs)
manifest = find_hmm_artifact(dirs)
if binary is None or manifest is None:
    raise RuntimeError("Final release artifact missing")
package = load_package(binary, manifest)
# trip_dir is a validated directory resolved by the service under the active partition.
matches = package.predict(RoadTrip.read(trip_dir))
```

Each returned dictionary contains `observation_id`, nullable `arc_id`, and `status`. Keep original observation order. Split displayed paths at empty states, resets and `soft_jump` discontinuities. Arc IDs identify rows in this trip's directed arc file. Convert segment endpoints from longitude/latitude to the response schema's latitude/longitude order. Each match describes the arc at a GPS fix rather than all intermediate connectors. No accuracy is available from an unlabeled request.

For candidate-only inference, `package.predict_proba(frame)` accepts exactly the 19 manifest feature names in order and returns the two columns for classes 0 and 1. The track-level entry point rebuilds those features and applies the calibrated decoder.

## FastAPI prediction

Set `PATHSNAP_TRIP_DATA_DIR` to the partition directory containing the trip folders. The application lifespan loads the verified pair once per worker. Restart the worker after replacing a release. Discovery uses the first sorted glob match, so pin the pair explicitly when hosting multiple releases.

POST `{"trip_id":"00000001","model":"final"}` to `/api/predict` or `/predict`. The response includes one ordered entry per fix in `matches`; missing candidates have a null arc ID. The original `predicted_arc_ids` retains every non-null assignment. Each entry in `predicted_route` is a separate latitude/longitude polyline. Render each independently. The service starts a new component at discontinuities and between arcs without a shared directed endpoint. No confidence is supplied by the track-level package. Accuracy fields remain null. Inference time covers trip reading, scoring, decoding and geometry assembly.

Unknown model keys return 400. Unknown or incomplete trips return 404. Invalid requests and malformed trip data return 422. Unavailable datasets, unreadable files, incompatible releases, and unimplemented KNN/MLP packages return 503. Health and model readiness reflect successful package loading. Failed startup verification is logged and leaves discovery endpoints available.

For the real-data backend integration test set `PATHSNAP_SMOKE_TRIP_DIR` to a development trip folder. The test copies only its three inference files to a temporary partition.

The export command is `python scripts/export_person3_release.py`. It checks the candidate labels against the valid training alignment, fits the full training roster, verifies a reload and refuses occupied v1 filenames. The selection records and processed data listed in the manifest must exist to rebuild the artifact. Decoder calibration used the fixed HGB emitter; its scores are not a held-out evaluation of this final tuned combination.
