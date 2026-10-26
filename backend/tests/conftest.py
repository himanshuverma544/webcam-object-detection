from __future__ import annotations

from typing import Any, List
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.detector import COCO_CLASSES, YOLOObjectDetector


class FakeBoxes:
    """Mock Ultralytics Boxes container with xyxy, conf, and cls arrays."""

    def __init__(
        self,
        xyxy: List[List[float]],
        conf: List[float],
        cls: List[int],
    ) -> None:
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)


class FakeResult:
    """Mock Ultralytics Result object."""

    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes


class FakeYOLOModel:
    """Mock Ultralytics YOLO model that returns deterministic detections without network downloads."""

    def __init__(self) -> None:
        self.names = dict(COCO_CLASSES)
        self.should_raise: bool = False

    def predict(self, source: Any, conf: float = 0.35, **kwargs: Any) -> List[FakeResult]:
        if self.should_raise:
            raise RuntimeError("Simulated GPU/CPU tensor execution failure")

        # Return two deterministic detections: person (0.91) and laptop (0.48)
        raw_boxes = [
            ([50.0, 40.0, 280.0, 420.0], 0.91, 0),   # person
            ([310.0, 200.0, 520.0, 390.0], 0.48, 63),  # laptop
        ]
        filtered = [b for b in raw_boxes if b[1] >= conf]
        if not filtered:
            return [FakeResult(FakeBoxes([], [], []))]

        xyxy = [b[0] for b in filtered]
        confs = [b[1] for b in filtered]
        classes = [b[2] for b in filtered]
        return [FakeResult(FakeBoxes(xyxy, confs, classes))]


@pytest.fixture()
def sample_jpeg_bytes() -> bytes:
    """Generate a valid 640x480 synthetic JPEG image using OpenCV."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (280, 420), (0, 255, 0), -1)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


@pytest.fixture()
def sample_png_bytes() -> bytes:
    """Generate a valid 320x240 synthetic PNG image using OpenCV."""
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


@pytest.fixture()
def sample_webp_bytes() -> bytes:
    """Generate a valid 320x240 synthetic WebP image using OpenCV."""
    image = np.full((240, 320, 3), 200, dtype=np.uint8)
    success, encoded = cv2.imencode(".webp", image)
    assert success
    return encoded.tobytes()


@pytest.fixture()
def mock_detector() -> YOLOObjectDetector:
    """Provide a YOLOObjectDetector backed by FakeYOLOModel."""
    detector = YOLOObjectDetector(
        model_path="yolo11n.pt",
        default_confidence=0.35,
        preferred_device="cpu",
    )
    detector.model = FakeYOLOModel()
    detector.model_loaded = True
    detector.device = "cpu"
    return detector


@pytest.fixture()
def client(mock_detector: YOLOObjectDetector) -> TestClient:
    """Provide a FastAPI TestClient with a pre-loaded mocked detector."""
    app = create_app()
    app.state.detector = mock_detector
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
