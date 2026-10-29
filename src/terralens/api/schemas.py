"""Response models for the TerraLens API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionOut(BaseModel):
    class_name: str = Field(examples=["Forest"])
    label: str = Field(examples=["Forest"])
    probability: float = Field(ge=0, le=1, examples=[0.987])
    color: str = Field(description="Map colour (hex)", examples=["#1b5e20"])


class HealthOut(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    version: str
    device: str | None = None


class ClassInfo(BaseModel):
    name: str
    label: str
    color: str


class ModelInfoOut(BaseModel):
    architecture: str
    img_size: int
    parameters: int
    classes: list[ClassInfo]
    metrics: dict
    trained_with: str | None = None
    created_at: str | None = None
    device: str
    terralens_version: str


class PredictOut(BaseModel):
    top: PredictionOut
    predictions: list[PredictionOut]
    image_size: tuple[int, int]
    elapsed_ms: float


class ExplainOut(BaseModel):
    explained: PredictionOut
    predictions: list[PredictionOut]
    overlay: str = Field(description="Grad-CAM overlay as a PNG data URL")
    elapsed_ms: float


class ClassAreaOut(BaseModel):
    class_name: str
    label: str
    color: str
    fraction: float = Field(ge=0, le=1)
    area_km2: float
    mean_confidence: float


class MapSummaryOut(BaseModel):
    image_size: tuple[int, int]
    area_km2: float
    tiles: int
    grid: tuple[int, int]
    mean_confidence: float
    uncertain_fraction: float = Field(description="Share of the area with confidence < 0.5")
    seconds: float


class MapOut(BaseModel):
    summary: MapSummaryOut
    classes: list[ClassAreaOut]
    map: str = Field(description="Land-cover map as a PNG data URL")
    overlay: str = Field(description="Map blended over the scene, JPEG data URL")
    confidence: str = Field(description="Confidence map as a PNG data URL")
    elapsed_ms: float


class SampleOut(BaseModel):
    name: str
    kind: str = Field(description="'tile' or 'scene'")
    url: str
