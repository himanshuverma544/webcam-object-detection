from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from app.schemas.detection import (
    DetectionItem,
    DetectionResponse,
    ImageDimensions,
    TamperStatus,
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
        self.prev_gray_frame: Optional[np.ndarray] = None
        self.tamper_blinded_streak: int = 0

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

    def reset_tracker(self) -> None:
        """Reset internal ByteTrack tracking state, track IDs, and tamper history."""
        self.prev_gray_frame = None
        self.tamper_blinded_streak = 0
        if (
            self.model is not None
            and hasattr(self.model, "predictor")
            and self.model.predictor is not None
        ):
            trackers = getattr(self.model.predictor, "trackers", [])
            for tracker in trackers:
                if hasattr(tracker, "reset"):
                    try:
                        tracker.reset()
                    except Exception as exc:
                        logger.warning("Error calling tracker.reset(): %s", exc)
                if hasattr(tracker, "reset_id"):
                    try:
                        tracker.reset_id()
                    except Exception as exc:
                        logger.warning("Error calling tracker.reset_id(): %s", exc)
            logger.info("Tracker states have been reset.")

    def check_tamper(self, frame: np.ndarray) -> TamperStatus:
        """Analyze frame for camera occlusion (blinded) and sudden global shift (moved/bumped)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_val, std_dev = cv2.meanStdDev(gray)
            brightness = float(mean_val[0][0])
            std_val = float(std_dev[0][0])
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Blinded condition:
            # 1. Pitch dark (covered with hand/tape) -> brightness < 20 and low variance
            # 2. Or covered with flat object (cloth/paper) -> variance < 6 and sharpness < 12
            is_blinded_frame = (brightness < 20.0 and std_val < 12.0) or (std_val < 6.0 and sharpness < 12.0)
            if is_blinded_frame:
                self.tamper_blinded_streak += 1
            else:
                self.tamper_blinded_streak = 0

            # Persistent blinded: 2 or more consecutive frames (prevents single-frame glitches)
            is_blinded = self.tamper_blinded_streak >= 2

            # Moved / Bumped condition:
            # Downscale for fast phase correlation (global 2D translation)
            small = cv2.resize(gray, (160, 120)).astype(np.float32)
            shift_magnitude = 0.0
            is_moved = False

            if self.prev_gray_frame is not None and not is_blinded:
                # Phase correlation between previous and current frame
                (shift_x, shift_y), response = cv2.phaseCorrelate(self.prev_gray_frame, small)
                shift_magnitude = float(np.hypot(shift_x, shift_y))
                # Sudden shift > 12 pixels on 160x120 frame (~10% global shift) with strong correlation
                if shift_magnitude > 12.0 and response > 0.35:
                    is_moved = True

            self.prev_gray_frame = small

            return TamperStatus(
                blinded=is_blinded,
                moved=is_moved,
                brightness=round(brightness, 1),
                sharpness=round(sharpness, 1),
                shift_magnitude=round(shift_magnitude, 1),
            )
        except Exception as exc:
            logger.warning("Error evaluating tamper status: %s", exc)
            return TamperStatus()

    def get_classes(self) -> Dict[int, str]:
        """Return mapping of supported class IDs to class labels."""
        return dict(self.classes)

    def resolve_class_ids(
        self, class_inputs: Optional[List[Union[str, int]]]
    ) -> Optional[List[int]]:
        """Resolve a list of class names or IDs into valid integer class IDs."""
        if not class_inputs:
            return None
        resolved: List[int] = []
        name_to_id = {name.lower(): cid for cid, name in self.classes.items()}
        for item in class_inputs:
            if isinstance(item, int):
                if item in self.classes:
                    resolved.append(item)
            elif isinstance(item, str):
                cleaned = item.strip().lower()
                if not cleaned:
                    continue
                if cleaned.isdigit():
                    cid = int(cleaned)
                    if cid in self.classes:
                        resolved.append(cid)
                elif cleaned in name_to_id:
                    resolved.append(name_to_id[cleaned])
        return sorted(list(set(resolved))) if resolved else None

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
        classes: Optional[List[int]] = None,
    ) -> DetectionResponse:
        """Decode image frame and run YOLO object detection, optionally filtered by class IDs."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Object detection model is not loaded yet.")

        conf_threshold = (
            float(confidence) if confidence is not None else float(self.default_confidence)
        )

        # Decode image safely with OpenCV
        frame = self.decode_image(image_bytes)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        tamper_status = self.check_tamper(frame)

        # If empty classes list was explicitly provided (e.g. no valid matched classes), return zero detections immediately
        if classes is not None and len(classes) == 0:
            return DetectionResponse(
                image_width=width,
                image_height=height,
                image_dimensions=ImageDimensions(width=width, height=height),
                inference_time_ms=0.0,
                confidence_threshold=round(conf_threshold, 4),
                detection_count=0,
                detections=[],
                tamper=tamper_status,
            )

        # Run YOLO inference and time execution
        start_time = time.perf_counter()
        results = None

        track_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": conf_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "device": self.device,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "verbose": False,
        }
        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": conf_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "device": self.device,
            "verbose": False,
        }
        if classes is not None:
            track_kwargs["classes"] = classes
            predict_kwargs["classes"] = classes

        try:
            if hasattr(self.model, "track"):
                try:
                    results = self.model.track(**track_kwargs)
                except Exception as track_exc:
                    logger.warning("YOLO track call failed (%s). Falling back to predict.", track_exc)
                    results = None

            if results is None:
                results = self.model.predict(**predict_kwargs)
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
                id_tensor = getattr(boxes, "id", None)

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
                id_list = None
                if id_tensor is not None:
                    id_list = (
                        id_tensor.cpu().numpy().tolist()
                        if hasattr(id_tensor, "cpu")
                        else np.asarray(id_tensor).tolist()
                    )

                for idx, (coords, score, cls_idx) in enumerate(zip(xyxy_list, conf_list, cls_list)):
                    score_val = float(score)
                    if score_val < conf_threshold:
                        continue

                    class_id = int(cls_idx)
                    if classes is not None and class_id not in classes:
                        continue

                    label = self.classes.get(class_id, f"class_{class_id}")

                    x1 = round(max(0.0, min(float(width), float(coords[0]))), 2)
                    y1 = round(max(0.0, min(float(height), float(coords[1]))), 2)
                    x2 = round(max(0.0, min(float(width), float(coords[2]))), 2)
                    y2 = round(max(0.0, min(float(height), float(coords[3]))), 2)

                    track_id: Optional[int] = None
                    if id_list is not None and idx < len(id_list) and id_list[idx] is not None:
                        try:
                            track_id = int(id_list[idx])
                        except (ValueError, TypeError):
                            track_id = None

                    detections.append(
                        DetectionItem(
                            label=label,
                            class_id=class_id,
                            confidence=round(score_val, 4),
                            bbox=[x1, y1, x2, y2],
                            track_id=track_id,
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
            tamper=tamper_status,
        )
