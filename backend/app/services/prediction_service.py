import logging

from fastapi import HTTPException, status

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.dataset_service import DatasetService
from app.services.model_registry_service import ModelRegistryService

logger = logging.getLogger("pathsnap.api")


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

        if not self._dataset_service.trip_exists(request.trip_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Trip '{request.trip_id}' was not found in the active "
                    f"'{self._dataset_service.partition_name}' partition."
                ),
            )

        # 3. Guard against unready models — never fabricate routes or accuracy
        if not model_info.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model '{request.model}' is not ready for inference. "
                    "Trained ML model artifacts and the HMM/Viterbi decoder "
                    "have not been integrated yet."
                ),
            )

        # 4. Placeholder guard until real feature extraction + HMM pipeline is wired
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Live inference pipeline (feature engineering + model scoring + "
                "HMM/Viterbi decoding) is not yet integrated."
            ),
        )
