from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check and model readiness response."""

    status: str = Field(
        ...,
        description="API health status ('ok' when model is ready, 'degraded' otherwise)",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="True only after the YOLO model has been loaded successfully",
    )
    model_ready: bool = Field(
        ...,
        description="True only when the model is loaded and ready for inference",
    )
    model_name: str = Field(
        ...,
        description="Configured YOLO model identifier",
        examples=["yolo11n.pt"],
    )
    device: str = Field(
        ...,
        description="Compute device used for inference (cpu, mps, or cuda)",
        examples=["cpu"],
    )
    classes_count: int = Field(
        ...,
        description="Number of object classes supported by the loaded model",
        examples=[80],
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if model initialization failed",
    )


class ClassesResponse(BaseModel):
    """Supported COCO classes response."""

    count: int = Field(..., description="Total number of supported classes", examples=[80])
    classes: Dict[int, str] = Field(
        ...,
        description="Mapping of COCO class ID to class label",
        examples=[{0: "person", 1: "bicycle", 2: "car"}],
    )
    class_names: List[str] = Field(
        ...,
        description="Ordered list of COCO class names",
        examples=[["person", "bicycle", "car"]],
    )


class ImageDimensions(BaseModel):
    """Pixel dimensions of the uploaded image."""

    width: int = Field(..., gt=0, description="Image width in pixels", examples=[640])
    height: int = Field(..., gt=0, description="Image height in pixels", examples=[480])


class DetectionItem(BaseModel):
    """A single detected object in an image frame."""

    label: str = Field(..., description="Detected COCO class name", examples=["person"])
    class_id: int = Field(..., ge=0, description="Detected COCO class ID", examples=[0])
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score between 0.0 and 1.0",
        examples=[0.892],
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x1, y1, x2, y2] in pixels relative to the uploaded image",
        examples=[[120.5, 45.0, 340.2, 410.8]],
    )


class DetectionResponse(BaseModel):
    """Structured response returned by POST /detect."""

    image_width: int = Field(..., gt=0, description="Uploaded image width in pixels")
    image_height: int = Field(..., gt=0, description="Uploaded image height in pixels")
    image_dimensions: ImageDimensions = Field(
        ..., description="Structured width and height of the uploaded image"
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Model inference execution latency in milliseconds",
        examples=[42.6],
    )
    confidence_threshold: float = Field(
        ...,
        ge=0.01,
        le=1.0,
        description="Confidence threshold applied during inference",
        examples=[0.35],
    )
    detection_count: int = Field(
        ...,
        ge=0.0,
        description="Total number of objects detected in the frame",
        examples=[2],
    )
    detections: List[DetectionItem] = Field(
        default_factory=list,
        description="List of detected objects sorted by descending confidence",
    )


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str = Field(..., description="Human-readable error description")
