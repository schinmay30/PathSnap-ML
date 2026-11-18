from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error explanation.")
    code: str | None = Field(
        default=None, description="Machine-readable error identifier."
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="API service status ('ok').")
    service: str = Field(..., description="Backend service name.")
    version: str = Field(..., description="Backend API version.")
    dataset_dir_available: bool = Field(
        ..., description="Whether the configured trip dataset directory exists."
    )
    active_partition: str = Field(
        ...,
        description="Name of the active trip partition (e.g., 'student_test' during development).",
    )
    partition_role: str = Field(
        ...,
        description="Explicit role of the active partition ('student_development_test', not evaluator data by default).",
    )
    available_trip_count: int = Field(
        ..., ge=0, description="Number of valid trips currently discovered in the active partition."
    )
    models_ready: bool = Field(
        ..., description="Whether at least one trained ML model artifact is loaded."
    )
    hmm_decoder_loaded: bool = Field(
        ..., description="Whether the HMM/Viterbi sequential decoder is loaded."
    )
    timestamp: str = Field(..., description="UTC ISO-8601 server timestamp.")
