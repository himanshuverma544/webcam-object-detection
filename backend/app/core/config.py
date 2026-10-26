from __future__ import annotations

from functools import lru_cache
from typing import List, Set

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(
        default="YOLO11 Real-Time Object Detection API",
        alias="APP_NAME",
    )
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Comma-separated list of allowed frontend origins
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        alias="CORS_ORIGINS",
    )

    # AI / Computer Vision model settings
    yolo_model_path: str = Field(default="yolo11n.pt", alias="YOLO_MODEL_PATH")
    default_confidence: float = Field(
        default=0.35, ge=0.01, le=1.0, alias="DEFAULT_CONFIDENCE"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.05, le=0.95, alias="IOU_THRESHOLD"
    )
    max_detections: int = Field(
        default=100, ge=1, le=300, alias="MAX_DETECTIONS"
    )
    device: str = Field(default="auto", alias="DEVICE")

    # Upload validation limits
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        ge=1024,
        alias="MAX_UPLOAD_SIZE_BYTES",
    )

    @property
    def cors_origin_list(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def allowed_content_types(self) -> Set[str]:
        return {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
