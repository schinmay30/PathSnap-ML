from pydantic import BaseModel, Field


class TripSummary(BaseModel):
    trip_id: str = Field(
        ..., description="Zero-padded trip folder identifier discovered on disk."
    )
    track_points: int = Field(
        ..., ge=0, description="Number of GPS observations in <trip_id>.track."
    )
    num_nodes: int = Field(
        ..., ge=0, description="Number of road-network nodes in <trip_id>.nodes."
    )
    num_arcs: int = Field(
        ..., ge=0, description="Number of directed road segments in <trip_id>.arcs."
    )


class TripsListResponse(BaseModel):
    count: int = Field(..., ge=0, description="Number of discovered trips in the active partition.")
    partition: str = Field(
        ...,
        description="Active dataset partition directory name (e.g., 'student_test' during development).",
    )
    partition_role: str = Field(
        ...,
        description="Explicit role of the partition ('student_development_test' for student_test).",
    )
    trips: list[TripSummary] = Field(
        default_factory=list,
        description="Dynamically discovered trip summaries (ground-truth routes excluded).",
    )
