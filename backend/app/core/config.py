from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Set

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

    # Comma-separated list of allowed frontend origins (supports '*' or specific domains)
    cors_origins: str = Field(
        default="*,https://webcam-object-detection.vercel.app,http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )

    # AI / Computer Vision model settings
    yolo_model_path: str = Field(default="yolo11n.pt", alias="YOLO_MODEL_PATH")
    yoloe_model_path: str = Field(
        default="yoloe-11s-seg.pt", alias="YOLOE_MODEL_PATH"
    )
    custom_items_dir: str = Field(
        default="data/custom_items", alias="CUSTOM_ITEMS_DIR"
    )
    default_similarity_threshold: float = Field(
        default=0.65, ge=0.1, le=1.0, alias="DEFAULT_SIMILARITY_THRESHOLD"
    )
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

    # Resend Email API Settings
    resend_api_key: Optional[str] = Field(default=None, alias="RESEND_API_KEY")
    resend_from_email: str = Field(
        default="Webcam Detection <onboarding@resend.dev>",
        alias="RESEND_FROM_EMAIL",
    )
    default_recipient_email: Optional[str] = Field(
        default=None,
        alias="DEFAULT_RECIPIENT_EMAIL",
    )

    # SMTP / Fallback Email Settings
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, alias="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")
    smtp_from_email: Optional[str] = Field(default=None, alias="SMTP_FROM_EMAIL")
    smtp_tls: bool = Field(default=True, alias="SMTP_TLS")

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
