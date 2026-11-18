from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    key: str = Field(..., description="Short model identifier used by POST /predict.")
    name: str = Field(..., description="Display name of the model.")
    family: str = Field(..., description="Algorithm family.")
    loaded: bool = Field(..., description="Whether trained model weights are loaded.")
    hmm_enabled: bool = Field(
        ..., description="Whether HMM/Viterbi decoding is attached to this model."
    )
    status: str = Field(
        ..., description="Current readiness status ('ready' or 'not_ready')."
    )
    artifact_path: str | None = Field(
        default=None,
        description="Relative artifact path when available, otherwise null.",
    )
    message: str = Field(
        ..., description="Human-readable readiness note for this model."
    )


class ModelsListResponse(BaseModel):
    models_ready: bool = Field(
        ..., description="True when at least one trained model is ready for inference."
    )
    hmm_decoder_loaded: bool = Field(
        ..., description="True when the HMM/Viterbi decoder is integrated and loaded."
    )
    models: list[ModelInfo] = Field(
        default_factory=list, description="Registered ML model status entries."
    )
    message: str = Field(..., description="Summary status message for the model registry.")


class ModelComparisonRow(BaseModel):
    model: str = Field(..., description="Model name.")
    family: str = Field(..., description="Algorithm family.")
    cv_macro_f1: float | None = Field(default=None)
    test_top1_accuracy: float | None = Field(default=None)
    test_macro_f1: float | None = Field(default=None)
    log_loss: float | None = Field(default=None)
    route_continuity: float | None = Field(default=None)
    training_time_s: float | None = Field(default=None)
    inference_time_s: float | None = Field(default=None)


class ModelComparisonResponse(BaseModel):
    available: bool = Field(
        ..., description="Whether recorded experimental comparison metrics exist."
    )
    models: list[ModelComparisonRow] = Field(
        default_factory=list,
        description="Recorded experimental comparison metrics (empty until experiments run).",
    )
    message: str = Field(..., description="Status message regarding comparison metrics.")
