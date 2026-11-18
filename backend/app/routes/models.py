from fastapi import APIRouter, Depends

from app.core.dependencies import get_model_registry_service
from app.schemas.models import ModelComparisonResponse, ModelsListResponse
from app.services.model_registry_service import ModelRegistryService

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List registered ML models & readiness status",
    description=(
        "Inspects artifacts/models/ and backend/models/ for real exported model "
        "weights and HMM/Viterbi configuration. Clearly reports 'not_ready' "
        "when models have not yet been trained or integrated."
    ),
)
def get_models(
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelsListResponse:
    return model_registry.list_models()


@router.get(
    "/models/comparison",
    response_model=ModelComparisonResponse,
    summary="Retrieve recorded model comparison metrics",
    description=(
        "Returns genuine experimental comparison metrics from "
        "artifacts/metrics/model_comparison.json when available. Returns an "
        "empty list until real experiments are recorded."
    ),
)
def get_model_comparison(
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
) -> ModelComparisonResponse:
    return model_registry.get_model_comparison()
