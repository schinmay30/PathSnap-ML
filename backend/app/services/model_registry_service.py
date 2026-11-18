import json
import logging
from inspect import signature
from pathlib import Path

from app.core.config import Settings
from app.schemas.models import (
    ModelComparisonResponse,
    ModelComparisonRow,
    ModelInfo,
    ModelsListResponse,
)
from models.loader import find_hmm_artifact, find_model_artifact
from pathsnap.deployment import load_package
from pathsnap.viterbi import DirectedRoadGraph, decode

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
        self.final_package = None
        self.final_artifact = find_model_artifact("final", self._model_search_dirs)
        config = find_hmm_artifact(self._model_search_dirs)
        self.final_message = "Final classifier or Viterbi manifest is missing."
        if self.final_artifact and config:
            try:
                package = load_package(self.final_artifact, config)
                policy = package.manifest["decoder"]
                required = {p.name for p in signature(decode).parameters.values()
                            if p.kind == p.KEYWORD_ONLY}
                if set(policy) != required or package.manifest["classes"] != [0, 1]:
                    raise ValueError("Invalid v1 decoder manifest")
                decode(DirectedRoadGraph([], []), [], **policy)
                self.final_package = package
                self.final_message = "Verified final classifier and decoder loaded."
            except Exception:
                logging.getLogger("pathsnap.api").exception("Failed to load final release")
                self.final_message = "Final package failed startup verification; check the server log."

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
        return self.final_package is not None

    def known_model_keys(self) -> set[str]:
        return {entry["key"] for entry in MODEL_CATALOG}

    def get_model_info(self, model_key: str) -> ModelInfo | None:
        hmm_loaded = self.is_hmm_loaded()
        for entry in MODEL_CATALOG:
            if entry["key"] != model_key:
                continue
            artifact = (self.final_artifact if model_key == "final" else
                        find_model_artifact(model_key, self._model_search_dirs))
            loaded = model_key == "final" and self.final_package is not None
            return ModelInfo(
                key=entry["key"],
                name=entry["name"],
                family=entry["family"],
                loaded=loaded,
                hmm_enabled=loaded and hmm_loaded,
                status="ready" if loaded else "not_ready",
                artifact_path=self._relative_to_workspace(artifact),
                message=(
                    self.final_message if model_key == "final"
                    else f"No compatible inference package is configured for {entry['name']}."
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
                "No verified inference package is ready. Check the final release in artifacts/final/."
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
