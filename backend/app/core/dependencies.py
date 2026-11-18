from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.dataset_service import DatasetService
from app.services.model_registry_service import ModelRegistryService
from app.services.prediction_service import PredictionService


def get_dataset_service(
    settings: Settings = Depends(get_settings),
) -> DatasetService:
    return DatasetService(settings)


def get_model_registry_service(
    settings: Settings = Depends(get_settings),
) -> ModelRegistryService:
    return ModelRegistryService(settings)


def get_prediction_service(
    dataset_service: DatasetService = Depends(get_dataset_service),
    model_registry_service: ModelRegistryService = Depends(
        get_model_registry_service
    ),
) -> PredictionService:
    return PredictionService(dataset_service, model_registry_service)
