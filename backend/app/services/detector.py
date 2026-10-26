from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from app.schemas.detection import (
    DetectionItem,
    DetectionResponse,
    ImageDimensions,
)

logger = logging.getLogger(__name__)

# Standard 80 COCO classes pretrained into YOLO11n
COCO_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class YOLOObjectDetector:
    """Manages Ultralytics YOLO11 model lifecycle, image decoding, and object detection."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        default_confidence: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        preferred_device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.default_confidence = default_confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.preferred_device = preferred_device

        self.model: Optional[Any] = None
        self.model_loaded: bool = False
        self.load_error: Optional[str] = None
        self.device: str = "cpu"
        self.classes: Dict[int, str] = dict(COCO_CLASSES)

    @property
    def is_ready(self) -> bool:
        """Return True only when the YOLO model is loaded and ready for inference."""
        return bool(self.model_loaded and self.model is not None)

    def _resolve_device(self) -> str:
        """Select optimal hardware acceleration device (CUDA -> Apple Silicon MPS -> CPU)."""
        import platform

        pref = (self.preferred_device or "auto").strip().lower()
        if pref in {"cuda", "mps", "cpu"}:
            if pref == "cuda" and torch.cuda.is_available():
                return "cuda"
            if (
                pref == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
            if pref != "cpu":
                logger.warning(
                    "Preferred device '%s' is not available; falling back to CPU.",
                    pref,
                )
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if (
            platform.machine() == "arm64"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load the Ultralytics YOLO11 model once during startup and run warmup."""
        self.model_loaded = False
        self.load_error = None
        self.device = self._resolve_device()

        try:
            from ultralytics import YOLO

            logger.info(
                "Loading YOLO model '%s' on device '%s'...",
                self.model_path,
                self.device,
            )
            loaded_model = YOLO(self.model_path)

            # Extract class names dictionary from loaded model if available
            raw_names = getattr(loaded_model, "names", None)
            if isinstance(raw_names, dict) and raw_names:
                self.classes = {int(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, list) and raw_names:
                self.classes = {idx: str(name) for idx, name in enumerate(raw_names)}

            # Perform a lightweight warmup pass to verify device execution
            warmup_frame = np.zeros((64, 64, 3), dtype=np.uint8)
            try:
                loaded_model.predict(
                    source=warmup_frame,
                    conf=self.default_confidence,
                    device=self.device,
                    verbose=False,
                )
            except Exception as warmup_exc:
                if self.device != "cpu":
                    logger.warning(
                        "Warmup on device '%s' failed (%s). Falling back to CPU.",
                        self.device,
                        warmup_exc,
                    )
                    self.device = "cpu"
                    loaded_model.predict(
                        source=warmup_frame,
                        conf=self.default_confidence,
                        device="cpu",
                        verbose=False,
                    )
                else:
                    raise

            self.model = loaded_model
            self.model_loaded = True
            logger.info(
                "YOLO model '%s' loaded and ready on '%s' (%d classes).",
                self.model_path,
                self.device,
                len(self.classes),
            )
        except Exception as exc:
            self.model = None
            self.model_loaded = False
            self.load_error = f"Failed to load YOLO model '{self.model_path}': {exc}"
            logger.exception("Model initialization failed.")
            raise RuntimeError(self.load_error) from exc

    def unload_model(self) -> None:
        """Release model reference on shutdown."""
        self.model = None
        self.model_loaded = False

    def get_classes(self) -> Dict[int, str]:
        """Return mapping of supported class IDs to class labels."""
        return dict(self.classes)

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Safely decode raw image bytes into a BGR OpenCV numpy array."""
        if not image_bytes:
            raise ValueError("Uploaded image file is empty.")

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        if buffer.size == 0:
            raise ValueError("Uploaded image buffer is empty.")

        try:
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise ValueError("Failed to decode image bytes.") from exc

        if frame is None or frame.ndim != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("Invalid or corrupted image file.")

        return frame

    def detect(
        self,
        image_bytes: bytes,
        confidence: Optional[float] = None,
    ) -> DetectionResponse:
        """Decode image frame and run YOLO object detection."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Object detection model is not loaded yet.")

        conf_threshold = (
            float(confidence) if confidence is not None else float(self.default_confidence)
        )

        # Decode image safely with OpenCV
        frame = self.decode_image(image_bytes)
        height, width = int(frame.shape[0]), int(frame.shape[1])

        # Run YOLO inference and time execution
        start_time = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference error: %s", exc)
            raise RuntimeError("Model inference failed on the provided image.") from exc

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        detections: List[DetectionItem] = []
        if results and len(results) > 0:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_tensor = boxes.xyxy
                conf_tensor = boxes.conf
                cls_tensor = boxes.cls

                xyxy_list = (
                    xyxy_tensor.cpu().numpy().tolist()
                    if hasattr(xyxy_tensor, "cpu")
                    else np.asarray(xyxy_tensor).tolist()
                )
                conf_list = (
                    conf_tensor.cpu().numpy().tolist()
                    if hasattr(conf_tensor, "cpu")
                    else np.asarray(conf_tensor).tolist()
                )
                cls_list = (
                    cls_tensor.cpu().numpy().tolist()
                    if hasattr(cls_tensor, "cpu")
                    else np.asarray(cls_tensor).tolist()
                )

                for coords, score, cls_idx in zip(xyxy_list, conf_list, cls_list):
                    score_val = float(score)
                    if score_val < conf_threshold:
                        continue

                    class_id = int(cls_idx)
                    label = self.classes.get(class_id, f"class_{class_id}")

                    x1 = round(max(0.0, min(float(width), float(coords[0]))), 2)
                    y1 = round(max(0.0, min(float(height), float(coords[1]))), 2)
                    x2 = round(max(0.0, min(float(width), float(coords[2]))), 2)
                    y2 = round(max(0.0, min(float(height), float(coords[3]))), 2)

                    detections.append(
                        DetectionItem(
                            label=label,
                            class_id=class_id,
                            confidence=round(score_val, 4),
                            bbox=[x1, y1, x2, y2],
                        )
                    )

        # Sort detections by confidence descending
        detections.sort(key=lambda item: item.confidence, reverse=True)

        return DetectionResponse(
            image_width=width,
            image_height=height,
            image_dimensions=ImageDimensions(width=width, height=height),
            inference_time_ms=inference_time_ms,
            confidence_threshold=round(conf_threshold, 4),
            detection_count=len(detections),
            detections=detections,
        )
