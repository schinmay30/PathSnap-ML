import json
from pathlib import Path

from app.core.config import Settings
from app.schemas.models import (
    ModelComparisonResponse,
    ModelComparisonRow,
    ModelInfo,
    ModelsListResponse,
)
from models.loader import find_hmm_artifact, find_model_artifact

MODEL_CATALOG: tuple[dict[str, str], ...] = (
    {
        "key": "knn",
        "name": "K-Nearest Neighbors (KNN)",
        "family": "Instance-based",
    },
    {
        "key": "mlp",
        "name": "Multi-Layer Perceptron (MLP)",
        "family": "Neural Networks",
    },
    {
        "key": "final",
        "name": "Final Tuned Model (Integrated)",
        "family": "Selected Best Model",
    },
)


class ModelRegistryService:
    """Reports real model & metric artifact availability without fabrication."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def _model_search_dirs(self) -> list[Path]:
        return [
            self._settings.artifacts_final_dir,
            self._settings.artifacts_models_dir,
            self._settings.backend_models_dir,
        ]

    def _relative_to_workspace(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(
                self._settings.workspace_root.resolve()
            ).as_posix()
        except ValueError:
            return path.name

    def is_hmm_loaded(self) -> bool:
        return find_hmm_artifact(self._model_search_dirs) is not None

    def known_model_keys(self) -> set[str]:
        return {entry["key"] for entry in MODEL_CATALOG}

    def get_model_info(self, model_key: str) -> ModelInfo | None:
        hmm_loaded = self.is_hmm_loaded()
        for entry in MODEL_CATALOG:
            if entry["key"] != model_key:
                continue
            artifact = find_model_artifact(model_key, self._model_search_dirs)
            loaded = artifact is not None
            return ModelInfo(
                key=entry["key"],
                name=entry["name"],
                family=entry["family"],
                loaded=loaded,
                hmm_enabled=loaded and hmm_loaded,
                status="ready" if loaded else "not_ready",
                artifact_path=self._relative_to_workspace(artifact),
                message=(
                    "Trained model artifact detected."
                    if loaded
                    else f"{entry['name']} artifact is not trained or exported yet."
                ),
            )
        return None

    def list_models(self) -> ModelsListResponse:
        hmm_loaded = self.is_hmm_loaded()
        models: list[ModelInfo] = []
        for entry in MODEL_CATALOG:
            info = self.get_model_info(entry["key"])
            if info is not None:
                models.append(info)

        models_ready = any(m.loaded for m in models)
        message = (
            "One or more trained model artifacts are available."
            if models_ready
            else (
                "No trained ML model artifacts are currently available. "
                "Export trained KNN, MLP, or final model artifacts to "
                "artifacts/models/ to enable live inference."
            )
        )
        return ModelsListResponse(
            models_ready=models_ready,
            hmm_decoder_loaded=hmm_loaded,
            models=models,
            message=message,
        )

    def get_model_comparison(self) -> ModelComparisonResponse:
        metrics_file = self._settings.artifacts_metrics_dir / "model_comparison.json"
        if not metrics_file.is_file():
            return ModelComparisonResponse(
                available=False,
                models=[],
                message=(
                    "No recorded model comparison metrics found in "
                    "artifacts/metrics/model_comparison.json."
                ),
            )

        try:
            raw = json.loads(metrics_file.read_text(encoding="utf-8"))
            raw_rows = raw if isinstance(raw, list) else raw.get("models", [])
            rows = [ModelComparisonRow.model_validate(item) for item in raw_rows]
        except Exception as exc:
            return ModelComparisonResponse(
                available=False,
                models=[],
                message=f"Failed to parse model_comparison.json: {exc}",
            )

        return ModelComparisonResponse(
            available=len(rows) > 0,
            models=rows,
            message=(
                f"Loaded {len(rows)} recorded model evaluation entries."
                if rows
                else "model_comparison.json contains no model entries."
            ),
        )
