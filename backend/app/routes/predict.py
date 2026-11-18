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
        "Runs the verified final package on track, nodes and arcs. Returns 400 for unknown models, "
        "404 for absent or incomplete trips, 422 for invalid inputs, and 503 for unavailable "
        "datasets, unreadable files or unready packages. predicted_route contains separate polylines."
    ),
)
def post_predict(
    payload: PredictRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    return prediction_service.predict(payload)
