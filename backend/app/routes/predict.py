from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_prediction_service
from app.schemas.common import ErrorResponse
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="Run live route prediction for a selected trip",
    description=(
        "Validates the requested trip_id and model key, executes live map-matching "
        "inference when trained artifacts are loaded, and computes evaluation "
        "metrics strictly server-side. Returns HTTP 503 when models are not yet ready."
    ),
)
def post_predict(
    payload: PredictRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    return prediction_service.predict(payload)
