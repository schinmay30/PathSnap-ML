from pydantic import BaseModel, Field, field_validator
import re

TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MODEL_KEY_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")


class PredictRequest(BaseModel):
    trip_id: str = Field(
        ...,
        description="Selected evaluation trip ID (e.g., '00000000').",
        examples=["00000000"],
    )
    model: str = Field(
        default="knn",
        description="Classifier identifier to run ('knn', 'mlp', or registered model key).",
        examples=["knn", "mlp"],
    )

    @field_validator("trip_id")
    @classmethod
    def validate_trip_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not TRIP_ID_PATTERN.match(cleaned):
            raise ValueError(
                "trip_id must contain only alphanumeric characters, underscores, or hyphens."
            )
        return cleaned

    @field_validator("model")
    @classmethod
    def validate_model_key(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned or not MODEL_KEY_PATTERN.match(cleaned):
            raise ValueError(
                "model must be a lowercase identifier such as 'knn' or 'mlp'."
            )
        return cleaned


class PredictionMetrics(BaseModel):
    accuracy: float | None = Field(
        default=None, description="Server-computed route prediction accuracy."
    )
    top1_accuracy: float | None = Field(
        default=None, description="Observation-level Top-1 candidate accuracy."
    )
    route_continuity: float | None = Field(
        default=None, description="Fraction of consecutive predicted arcs connected in graph."
    )
    inference_time_ms: float | None = Field(
        default=None, description="End-to-end inference latency in milliseconds."
    )


class PredictResponse(BaseModel):
    trip_id: str = Field(..., description="Evaluated trip identifier.")
    model: str = Field(..., description="Model key used for inference.")
    status: str = Field(..., description="Inference execution status.")
    num_gps_points: int = Field(..., ge=0)
    num_predicted_arcs: int = Field(..., ge=0)
    inference_time_ms: float = Field(..., ge=0.0)
    hmm_applied: bool = Field(
        ..., description="Whether HMM/Viterbi decoding was applied."
    )
    gps_points: list[list[float]] = Field(
        default_factory=list,
        description="Observed GPS points as [latitude, longitude] pairs.",
    )
    road_network: list[list[list[float]]] = Field(
        default_factory=list,
        description="Relevant road-network corridor polylines for visualization.",
    )
    predicted_route: list[list[list[float]]] = Field(
        default_factory=list,
        description="Predicted road-segment polylines (never ground truth).",
    )
    predicted_arc_ids: list[int] = Field(
        default_factory=list,
        description="Predicted chronological arc_id sequence.",
    )
    metrics: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="Server-side evaluation metrics computed without exposing .route labels.",
    )
