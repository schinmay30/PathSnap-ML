from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_dataset_service
from app.schemas.common import ErrorResponse
from app.schemas.trips import TripsListResponse
from app.services.dataset_service import DatasetService

router = APIRouter(tags=["Trips"])


@router.get(
    "/trips",
    response_model=TripsListResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="List available trips in active partition",
    description=(
        "Dynamically discovers trips inside the active trip partition directory "
        "(student_test during development) without hardcoding trip IDs. "
        "Ground-truth .route files are never read or exposed."
    ),
)
def get_trips(
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> TripsListResponse:
    if not dataset_service.is_dataset_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configured trip dataset directory is unavailable on the server.",
        )
    return dataset_service.list_trips()
