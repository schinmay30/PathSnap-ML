import logging
from time import perf_counter

from fastapi import HTTPException, status

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.dataset_service import DatasetService
from app.services.model_registry_service import ModelRegistryService
from pathsnap.route_inference import RoadTrip

logger = logging.getLogger("pathsnap.api")


def route_geometry(matches, trip):
    """Join only observed arcs with a shared directed endpoint in the same segment."""
    components = []
    previous = None
    for match in matches:
        arc = match["arc_id"]
        if arc is None:
            previous = None
            continue
        src, dst = trip.arcs[arc]
        if match["status"] == "matched" and previous == arc:
            continue
        endpoint = list(reversed(trip.nodes[dst]))
        if match["status"] == "matched" and previous is not None and trip.arcs[previous][1] == src:
            components[-1].append(endpoint)
        else:
            components.append([list(reversed(trip.nodes[src])), endpoint])
        previous = arc
    return components


class PredictionService:
    """Coordinates live route prediction requests with strict readiness checks.

    Never fabricates predictions, routes, or evaluation accuracy when trained
    models or the HMM/Viterbi decoder are unavailable.
    """

    def __init__(
        self,
        dataset_service: DatasetService,
        model_registry_service: ModelRegistryService,
    ) -> None:
        self._dataset_service = dataset_service
        self._model_registry = model_registry_service

    def predict(self, request: PredictRequest) -> PredictResponse:
        logger.info(
            "Prediction requested | trip_id=%s | model=%s",
            request.trip_id,
            request.model,
        )

        # 1. Verify requested model key is recognized
        model_info = self._model_registry.get_model_info(request.model)
        if model_info is None:
            valid_keys = ", ".join(sorted(self._model_registry.known_model_keys()))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported model '{request.model}'. "
                    f"Supported model keys: {valid_keys}."
                ),
            )

        # 2. Verify dataset availability and trip existence (never hardcode trip IDs)
        if not self._dataset_service.is_dataset_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configured trip dataset directory is not available on the server.",
            )

        folder = self._dataset_service.resolve_trip_dir(request.trip_id)
        if folder is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Trip '{request.trip_id}' was not found with all required inference files in the active "
                    f"'{self._dataset_service.partition_name}' partition."
                ),
            )

        # 3. Guard against unready models — never fabricate routes or accuracy
        if not model_info.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model '{request.model}' is not ready for inference. "
                    f"{model_info.message}"
                ),
            )

        start = perf_counter()
        try:
            trip = RoadTrip.read(folder)
            matches = self._model_registry.final_package.predict(trip)
        except OSError as exc:
            raise HTTPException(503, "Trip inference files could not be read.") from exc
        except (ValueError, IndexError) as exc:
            raise HTTPException(422, "Invalid trip coordinates, timestamps or road graph.") from exc
        arc_ids = [m["arc_id"] for m in matches if m["arc_id"] is not None]
        geometry = route_geometry(matches, trip)
        corridor = [[list(reversed(trip.nodes[n])) for n in trip.arcs[a]] for a in dict.fromkeys(arc_ids)]
        elapsed = (perf_counter() - start) * 1000
        return PredictResponse(
            trip_id=request.trip_id, model=request.model, status="success",
            num_gps_points=len(trip.track), num_predicted_arcs=len(arc_ids),
            inference_time_ms=elapsed, hmm_applied=True, matches=matches,
            gps_points=[[lat, lon] for lon, lat, _ in trip.track], road_network=corridor,
            predicted_route=geometry, predicted_arc_ids=arc_ids, metrics={"inference_time_ms": elapsed},
        )
