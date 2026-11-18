from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.dependencies import (
    get_dataset_service,
    get_model_registry_service,
)
from app.schemas.common import HealthResponse
from app.services.dataset_service import DatasetService
from app.services.model_registry_service import ModelRegistryService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend health & readiness check",
    description=(
        "Returns FastAPI server availability, active trip partition readiness "
        "(student_test during development), discovered trip count, and ML/HMM "
        "model load status."
    ),
)
def get_health(
    settings: Settings = Depends(get_settings),
    dataset_service: DatasetService = Depends(get_dataset_service),
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
) -> HealthResponse:
    models_summary = model_registry.list_models()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        dataset_dir_available=dataset_service.is_dataset_available(),
        active_partition=dataset_service.partition_name,
        partition_role=dataset_service.partition_role,
        available_trip_count=dataset_service.count_available_trips(),
        models_ready=models_summary.models_ready,
        hmm_decoder_loaded=models_summary.hmm_decoder_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
