from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

# Allow MPS fallback for torchvision::nms on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.schemas.detection import AutoBoxResponse, CustomItemOut, DetectionItem

logger = logging.getLogger(__name__)

# ImageNet normalization constants for MobileNetV3-Large
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

CUSTOM_CLASS_ID_BASE = 1000
CUSTOM_TRACK_ID_OFFSET = 10000
MAX_REF_SIDE_PX = 1024
DEFAULT_BBOX_NORM = [0.03, 0.03, 0.97, 0.97]
MAX_LIVE_BOX_AREA_RATIO = 0.82
BG_CONTRAST_MARGIN = 0.06


def _sanitize_bbox_norm(bbox: Optional[Sequence[float]]) -> List[float]:
    """Validate and clamp a normalized [x1, y1, x2, y2] box into [0.0, 1.0]."""
    if not bbox or len(bbox) < 4:
        return list(DEFAULT_BBOX_NORM)
    try:
        x1 = max(0.0, min(1.0, float(bbox[0])))
        y1 = max(0.0, min(1.0, float(bbox[1])))
        x2 = max(0.0, min(1.0, float(bbox[2])))
        y2 = max(0.0, min(1.0, float(bbox[3])))
    except (TypeError, ValueError):
        return list(DEFAULT_BBOX_NORM)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Ensure box has non-trivial area
    if (x2 - x1) < 0.03 or (y2 - y1) < 0.03:
        return list(DEFAULT_BBOX_NORM)

    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _norm_bbox_to_px(
    bbox_norm: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Convert normalized [x1, y1, x2, y2] into clamped integer pixel bounds."""
    x1, y1, x2, y2 = _sanitize_bbox_norm(bbox_norm)
    ix1 = max(0, min(width - 2, int(round(x1 * width))))
    iy1 = max(0, min(height - 2, int(round(y1 * height))))
    ix2 = max(ix1 + 2, min(width, int(round(x2 * width))))
    iy2 = max(iy1 + 2, min(height, int(round(y2 * height))))
    return ix1, iy1, ix2, iy2


def _crop_by_norm_bbox(img_bgr: np.ndarray, bbox_norm: Sequence[float]) -> np.ndarray:
    """Return the tightly cropped foreground region from img_bgr."""
    h, w = img_bgr.shape[:2]
    ix1, iy1, ix2, iy2 = _norm_bbox_to_px(bbox_norm, w, h)
    crop = img_bgr[iy1:iy2, ix1:ix2]
    return crop if crop.size > 0 else img_bgr


def _extract_background_strips(
    img_bgr: np.ndarray, bbox_norm: Sequence[float]
) -> List[np.ndarray]:
    """
    Extract clean background patches strictly OUTSIDE the user's selected foreground box.
    Used by the Background-Rejection Guard to distinguish the object from its desk/room background.
    """
    x1, y1, x2, y2 = _sanitize_bbox_norm(bbox_norm)
    fg_area_ratio = (x2 - x1) * (y2 - y1)
    # If the user selected nearly the whole photo, there isn't meaningful background to model
    if fg_area_ratio > 0.78:
        return []

    h, w = img_bgr.shape[:2]
    ix1, iy1, ix2, iy2 = _norm_bbox_to_px(bbox_norm, w, h)

    # Leave a tiny 2% buffer away from the object edges so object pixels don't leak into background
    pad_x = max(2, int(w * 0.02))
    pad_y = max(2, int(h * 0.02))
    min_strip_w = max(24, int(w * 0.14))
    min_strip_h = max(24, int(h * 0.14))

    candidates: List[Tuple[int, np.ndarray]] = []
    # Top strip (e.g. keyboard above wallet)
    if iy1 - pad_y >= min_strip_h:
        patch = img_bgr[0 : iy1 - pad_y, :]
        candidates.append((patch.shape[0] * patch.shape[1], patch))
    # Bottom strip (e.g. table edge / floor below wallet)
    if h - (iy2 + pad_y) >= min_strip_h:
        patch = img_bgr[iy2 + pad_y : h, :]
        candidates.append((patch.shape[0] * patch.shape[1], patch))
    # Left strip
    if ix1 - pad_x >= min_strip_w:
        patch = img_bgr[iy1:iy2, 0 : ix1 - pad_x]
        candidates.append((patch.shape[0] * patch.shape[1], patch))
    # Right strip
    if w - (ix2 + pad_x) >= min_strip_w:
        patch = img_bgr[iy1:iy2, ix2 + pad_x : w]
        candidates.append((patch.shape[0] * patch.shape[1], patch))

    # Keep the 3 largest clean background strips
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [patch for _, patch in candidates[:3] if patch.size > 0]


def auto_detect_foreground_bbox(image_bytes: bytes) -> AutoBoxResponse:
    """
    Estimate a tight foreground bounding box around the main subject of an uploaded photo
    using OpenCV GrabCut (zero extra model downloads, ~150-250ms).
    """
    img = _decode_bgr_image(image_bytes)
    h, w = img.shape[:2]

    fallback_norm = [0.15, 0.15, 0.85, 0.85]
    try:
        scale = min(1.0, 320.0 / float(max(h, w)))
        sw = max(32, int(round(w * scale)))
        sh = max(32, int(round(h * scale)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        mask = np.zeros((sh, sw), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Center-biased initial rect excluding outer border noise
        rx = max(1, int(sw * 0.10))
        ry = max(1, int(sh * 0.14))
        rw = max(16, int(sw * 0.80))
        rh = max(16, int(sh * 0.72))
        rect = (rx, ry, min(rw, sw - rx - 1), min(rh, sh - ry - 1))

        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

        # Clean small noise specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(fg_mask)
        best_stat = None
        best_score = -1.0
        min_area = (sw * sh) * 0.02

        for i in range(1, num_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = centroids[i]
            dist_from_center = float(np.hypot((cx - sw / 2.0) / sw, (cy - sh / 2.0) / sh))
            score = area * max(0.1, 1.0 - dist_from_center)
            if score > best_score:
                best_score = score
                best_stat = stats[i]

        if best_stat is not None:
            bx = float(best_stat[cv2.CC_STAT_LEFT]) / scale
            by = float(best_stat[cv2.CC_STAT_TOP]) / scale
            bw = float(best_stat[cv2.CC_STAT_WIDTH]) / scale
            bh = float(best_stat[cv2.CC_STAT_HEIGHT]) / scale

            # Add subtle 2% breathing pad so object boundaries aren't clipped
            pad_x = w * 0.02
            pad_y = h * 0.02
            x1 = max(0.0, bx - pad_x)
            y1 = max(0.0, by - pad_y)
            x2 = min(float(w), bx + bw + pad_x)
            y2 = min(float(h), by + bh + pad_y)

            area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
            if 0.02 <= area_ratio <= 0.90:
                bbox_norm = _sanitize_bbox_norm([x1 / w, y1 / h, x2 / w, y2 / h])
                return AutoBoxResponse(
                    width=w,
                    height=h,
                    bbox_norm=bbox_norm,
                    bbox_px=[
                        round(bbox_norm[0] * w, 1),
                        round(bbox_norm[1] * h, 1),
                        round(bbox_norm[2] * w, 1),
                        round(bbox_norm[3] * h, 1),
                    ],
                    method="grabcut",
                )
    except Exception as exc:
        logger.warning("GrabCut auto-box failed, returning center fallback: %s", exc)

    return AutoBoxResponse(
        width=w,
        height=h,
        bbox_norm=fallback_norm,
        bbox_px=[
            round(fallback_norm[0] * w, 1),
            round(fallback_norm[1] * h, 1),
            round(fallback_norm[2] * w, 1),
            round(fallback_norm[3] * h, 1),
        ],
        method="fallback_center",
    )


@dataclass
class CustomItemRecord:
    """In-memory and on-disk representation of a user-uploaded custom item."""

    id: str
    name: str
    class_id: int
    image_files: List[str]
    image_bboxes: List[List[float]]
    thumbnail_base64: str
    created_at: str

    # Runtime-computed tensors/features (computed lazily when models load)
    vpe: Optional[torch.Tensor] = field(default=None, repr=False)
    ref_embeddings: List[np.ndarray] = field(default_factory=list, repr=False)
    ref_hists: List[np.ndarray] = field(default_factory=list, repr=False)
    ref_bg_embeddings: List[np.ndarray] = field(default_factory=list, repr=False)
    ref_bg_hists: List[np.ndarray] = field(default_factory=list, repr=False)
    features_ready: bool = False

    def to_meta_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "class_id": self.class_id,
            "image_files": list(self.image_files),
            "image_bboxes": [list(b) for b in self.image_bboxes],
            "thumbnail_base64": self.thumbnail_base64,
            "created_at": self.created_at,
        }

    def to_schema(self) -> CustomItemOut:
        return CustomItemOut(
            id=self.id,
            name=self.name,
            class_id=self.class_id,
            image_count=len(self.image_files),
            thumbnail_base64=self.thumbnail_base64,
            created_at=self.created_at,
        )


def _decode_bgr_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to OpenCV BGR numpy array, raising ValueError if invalid."""
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0 or img.ndim != 3:
        raise ValueError("Unable to decode image. Please upload a valid JPEG, PNG, or WebP file.")
    return img


def _resize_max_side(img_bgr: np.ndarray, max_side: int = MAX_REF_SIDE_PX) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return img_bgr
    scale = max_side / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _make_thumbnail_data_url(img_bgr: np.ndarray, size: int = 160) -> str:
    """Create a square center-cropped JPEG base64 data URL for UI previews."""
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    square = img_bgr[y0 : y0 + side, x0 : x0 + side]
    thumb = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _compute_central_hsv_hist(img_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV color histogram from the inner 70% of the object crop."""
    h, w = img_bgr.shape[:2]
    if h >= 10 and w >= 10:
        margin_y = int(h * 0.15)
        margin_x = int(w * 0.15)
        crop = img_bgr[margin_y : h - margin_y, margin_x : w - margin_x]
    else:
        crop = img_bgr

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 12], [0, 180, 0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    hs_norm = np.linalg.norm(hist_hs)
    if hs_norm > 1e-6:
        hist_hs = hist_hs / hs_norm
    v_norm = np.linalg.norm(hist_v)
    if v_norm > 1e-6:
        hist_v = hist_v / v_norm

    combined = np.concatenate([hist_hs * 0.85, hist_v * 0.52]).astype(np.float32)
    total_norm = np.linalg.norm(combined)
    if total_norm > 1e-6:
        combined /= total_norm
    return combined


def _prep_crop_tensor(img_bgr: np.ndarray) -> np.ndarray:
    """Resize BGR crop to 224x224 RGB and normalize for ImageNet MobileNetV3."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))  # CHW


class CustomObjectDetector:
    """
    Visual-Prompted Custom Object Detector combining:
      1. Ultralytics YOLOE visual prompt embeddings with exact user-selected foreground boxes
      2. MobileNetV3-Large + HSV color instance verification on tight object crops
      3. Background-Rejection Guard (foreground vs excluded background contrast + full-frame area cap)
    """

    def __init__(
        self,
        yoloe_model_path: str = "yoloe-11s-seg.pt",
        storage_dir: str = "data/custom_items",
        preferred_device: str = "auto",
        default_similarity_threshold: float = 0.65,
    ) -> None:
        self._backend_root = Path(__file__).resolve().parents[2]
        self.yoloe_model_path = self._resolve_model_path(yoloe_model_path)
        self.storage_dir = self._resolve_storage_dir(storage_dir)
        self.preferred_device = preferred_device
        self.default_similarity_threshold = default_similarity_threshold

        self._lock = threading.RLock()
        self._items: Dict[str, CustomItemRecord] = {}
        self._next_class_id: int = CUSTOM_CLASS_ID_BASE

        # Lazy-loaded ML models
        self._yoloe_model: Optional[Any] = None
        self._embedder_model: Optional[Any] = None
        self._models_loaded: bool = False
        self._load_error: Optional[str] = None
        self.device: str = "cpu"

        # Cache currently active classes on YOLOE to avoid redundant set_classes() calls
        self._active_prompt_signature: Optional[Tuple[str, ...]] = None
        self._prompt_version: int = 0

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata_from_disk()

    @property
    def is_ready(self) -> bool:
        return bool(self._models_loaded and self._yoloe_model is not None)

    def _resolve_model_path(self, raw_path: str) -> str:
        p = Path(raw_path)
        if p.is_absolute() or p.exists():
            return str(p)
        candidate = self._backend_root / raw_path
        if candidate.exists():
            return str(candidate)
        return raw_path

    def _resolve_storage_dir(self, raw_dir: str) -> Path:
        p = Path(raw_dir)
        if p.is_absolute():
            return p
        return self._backend_root / raw_dir

    def _resolve_device(self) -> str:
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

    def _meta_file(self) -> Path:
        return self.storage_dir / "items.json"

    def _load_metadata_from_disk(self) -> None:
        meta_path = self._meta_file()
        if not meta_path.exists():
            return
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            items = data.get("items", [])
            max_cid = CUSTOM_CLASS_ID_BASE - 1
            for raw in items:
                raw_files = raw.get("image_files", [])
                raw_bboxes = raw.get("image_bboxes", [])
                img_files: List[str] = []
                img_bboxes: List[List[float]] = []
                for idx, fname in enumerate(raw_files):
                    if (self.storage_dir / fname).exists():
                        img_files.append(fname)
                        box = raw_bboxes[idx] if idx < len(raw_bboxes) else None
                        img_bboxes.append(_sanitize_bbox_norm(box))
                if not img_files:
                    continue
                cid = int(raw.get("class_id", max_cid + 1))
                max_cid = max(max_cid, cid)
                rec = CustomItemRecord(
                    id=str(raw["id"]),
                    name=str(raw["name"]),
                    class_id=cid,
                    image_files=img_files,
                    image_bboxes=img_bboxes,
                    thumbnail_base64=str(raw.get("thumbnail_base64", "")),
                    created_at=str(
                        raw.get("created_at")
                        or datetime.now(timezone.utc).isoformat()
                    ),
                )
                self._items[rec.id] = rec
            self._next_class_id = max(CUSTOM_CLASS_ID_BASE, max_cid + 1)
            logger.info("Loaded %d custom reference item(s) from disk.", len(self._items))
        except Exception as exc:
            logger.warning("Failed to load custom items metadata: %s", exc)

    def _save_metadata_to_disk(self) -> None:
        payload = {
            "items": [rec.to_meta_dict() for rec in self._items.values()],
        }
        tmp_path = self.storage_dir / "items.json.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self._meta_file())

    def ensure_models_loaded(self) -> None:
        """Lazily initialize YOLOE visual-prompt model and MobileNetV3-Large embedder."""
        with self._lock:
            if self._models_loaded and self._yoloe_model is not None:
                return

            self.device = self._resolve_device()
            self._load_error = None
            try:
                from ultralytics import YOLOE
                import torchvision.models as tv_models

                logger.info(
                    "Loading YOLOE visual-prompt model '%s' and MobileNetV3-Large on '%s'...",
                    self.yoloe_model_path,
                    self.device,
                )
                yoloe = YOLOE(self.yoloe_model_path)

                embedder = tv_models.mobilenet_v3_large(
                    weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT
                )
                embedder.classifier = torch.nn.Identity()
                embedder.eval()
                try:
                    embedder = embedder.to(self.device)
                except Exception as emb_exc:
                    logger.warning(
                        "Failed to move embedder to '%s' (%s); using CPU.",
                        self.device,
                        emb_exc,
                    )
                    self.device = "cpu"
                    embedder = embedder.to("cpu")

                self._yoloe_model = yoloe
                self._embedder_model = embedder
                self._models_loaded = True
                logger.info("Custom visual detection models ready on '%s'.", self.device)
            except Exception as exc:
                self._load_error = f"Failed to initialize custom visual detector: {exc}"
                logger.exception("Custom model initialization failed.")
                raise RuntimeError(self._load_error) from exc

    def _extract_single_vpe(
        self, img_bgr: np.ndarray, bbox_norm: Optional[Sequence[float]] = None
    ) -> torch.Tensor:
        """Extract a (1, 1, 512) YOLOE visual prompt embedding for the exact selected box."""
        from ultralytics.models.yolo.yoloe.predict import (
            YOLOEVPDetectPredictor,
            YOLOEVPSegPredictor,
        )

        h, w = img_bgr.shape[:2]
        ix1, iy1, ix2, iy2 = _norm_bbox_to_px(bbox_norm or DEFAULT_BBOX_NORM, w, h)

        task = getattr(self._yoloe_model.model, "task", "segment")
        predictor_cls = YOLOEVPSegPredictor if task == "segment" else YOLOEVPDetectPredictor
        overrides = {
            "verbose": False,
            "task": task,
            "mode": "predict",
            "save": False,
            "batch": 1,
            "device": "cpu",
        }
        pred = predictor_cls(overrides=overrides)
        pred.set_prompts(
            {
                "bboxes": np.array([[float(ix1), float(iy1), float(ix2), float(iy2)]], dtype=np.float32),
                "cls": np.array([0]),
            }
        )
        pred.setup_model(model=self._yoloe_model.model, verbose=False)
        with torch.no_grad():
            vpe = pred.get_vpe(img_bgr)
        return vpe.detach().float().cpu()

    def _extract_batch_embeddings(self, crops_bgr: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Compute L2-normalized 960-dim MobileNetV3-Large embeddings for a batch of BGR crops."""
        if not crops_bgr or self._embedder_model is None:
            return []

        batch_np = np.stack([_prep_crop_tensor(crop) for crop in crops_bgr], axis=0)
        batch_tensor = torch.from_numpy(batch_np).to(self.device)
        with torch.no_grad():
            feats = self._embedder_model(batch_tensor)
            feats = F.normalize(feats.float(), p=2, dim=1)
        arr = feats.detach().cpu().numpy()
        return [arr[i].astype(np.float32) for i in range(arr.shape[0])]

    def _compute_item_features(self, record: CustomItemRecord) -> None:
        """
        Compute:
          1. YOLOE VPE on full reference image(s) with exact user-selected foreground box(es)
          2. MobileNetV3 + HSV fingerprints strictly on the tight foreground crop(s)
          3. Background signatures on excluded strips outside the box(es) for Background Guard
        """
        self.ensure_models_loaded()

        full_imgs_with_boxes: List[Tuple[np.ndarray, List[float]]] = []
        for idx, fname in enumerate(record.image_files):
            fpath = self.storage_dir / fname
            if not fpath.exists():
                continue
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                box = (
                    record.image_bboxes[idx]
                    if idx < len(record.image_bboxes)
                    else list(DEFAULT_BBOX_NORM)
                )
                full_imgs_with_boxes.append((img, _sanitize_bbox_norm(box)))

        if not full_imgs_with_boxes:
            raise ValueError(f"No readable reference images found for custom item '{record.name}'.")

        vpes = [self._extract_single_vpe(img, box) for img, box in full_imgs_with_boxes]
        avg_vpe = torch.mean(torch.stack(vpes, dim=0), dim=0)
        avg_vpe = F.normalize(avg_vpe, p=2, dim=-1)

        # Tight foreground crops for pure object appearance & color matching
        fg_crops = [_crop_by_norm_bbox(img, box) for img, box in full_imgs_with_boxes]
        ref_embeddings = self._extract_batch_embeddings(fg_crops)
        ref_hists = [_compute_central_hsv_hist(crop) for crop in fg_crops]

        # Background strips strictly outside user-selected box for contrast guard
        bg_strips: List[np.ndarray] = []
        for img, box in full_imgs_with_boxes:
            bg_strips.extend(_extract_background_strips(img, box))

        # Cap total background strips stored per item to keep per-frame math instant
        bg_strips = bg_strips[:6]
        ref_bg_embeddings = self._extract_batch_embeddings(bg_strips) if bg_strips else []
        ref_bg_hists = [_compute_central_hsv_hist(strip) for strip in bg_strips]

        record.vpe = avg_vpe
        record.ref_embeddings = ref_embeddings
        record.ref_hists = ref_hists
        record.ref_bg_embeddings = ref_bg_embeddings
        record.ref_bg_hists = ref_bg_hists
        record.features_ready = True

    def list_items(self) -> List[CustomItemOut]:
        with self._lock:
            return [rec.to_schema() for rec in self._items.values()]

    def get_item(self, item_id: str) -> Optional[CustomItemRecord]:
        with self._lock:
            return self._items.get(item_id)

    def add_item(
        self,
        name: str,
        image_bytes_list: Sequence[bytes],
        bboxes_norm: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> CustomItemOut:
        """Register a new custom item from uploaded reference images and optional tight foreground boxes."""
        cleaned_name = " ".join((name or "").strip().split())
        if not cleaned_name:
            raise ValueError("Item name cannot be empty.")
        if len(cleaned_name) > 60:
            raise ValueError("Item name must be 60 characters or fewer.")
        if not image_bytes_list:
            raise ValueError("At least one reference image is required.")

        decoded_imgs = [_resize_max_side(_decode_bgr_image(b)) for b in image_bytes_list]
        resolved_bboxes = [
            _sanitize_bbox_norm(
                bboxes_norm[idx] if bboxes_norm and idx < len(bboxes_norm) else None
            )
            for idx in range(len(decoded_imgs))
        ]

        with self._lock:
            self.ensure_models_loaded()

            # Enforce unique custom item names (case-insensitive and trimmed)
            for existing in self._items.values():
                if existing.name.strip().lower() == cleaned_name.lower():
                    raise ValueError(
                        f"A custom item named '{existing.name}' already exists. Please choose a unique name."
                    )

            item_id = f"item_{uuid.uuid4().hex[:8]}"
            class_id = self._next_class_id
            self._next_class_id += 1

            saved_files: List[str] = []
            for idx, img in enumerate(decoded_imgs):
                fname = f"{item_id}_{idx}_{int(time.time() * 1000)}.jpg"
                out_path = self.storage_dir / fname
                ok = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    raise RuntimeError("Failed to save reference image to disk.")
                saved_files.append(fname)

            # Generate clean thumbnail strictly from the tight foreground crop
            primary_fg_crop = _crop_by_norm_bbox(decoded_imgs[0], resolved_bboxes[0])
            thumbnail = _make_thumbnail_data_url(primary_fg_crop)
            record = CustomItemRecord(
                id=item_id,
                name=cleaned_name,
                class_id=class_id,
                image_files=saved_files,
                image_bboxes=resolved_bboxes,
                thumbnail_base64=thumbnail,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            try:
                self._compute_item_features(record)
            except Exception:
                for fname in saved_files:
                    (self.storage_dir / fname).unlink(missing_ok=True)
                raise

            self._items[item_id] = record
            self._prompt_version += 1
            self._active_prompt_signature = None
            self._save_metadata_to_disk()

            logger.info(
                "Registered custom item '%s' (id=%s, class_id=%d, photos=%d, bg_strips=%d).",
                record.name,
                record.id,
                record.class_id,
                len(saved_files),
                len(record.ref_bg_embeddings),
            )
            return record.to_schema()

    def add_images_to_item(
        self,
        item_id: str,
        image_bytes_list: Sequence[bytes],
        bboxes_norm: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> CustomItemOut:
        """Add additional angle photos (with optional tight boxes) to an existing custom item."""
        if not image_bytes_list:
            raise ValueError("At least one reference image is required.")

        decoded_imgs = [_resize_max_side(_decode_bgr_image(b)) for b in image_bytes_list]

        with self._lock:
            record = self._items.get(item_id)
            if record is None:
                raise KeyError(f"Custom item '{item_id}' not found.")

            to_add = decoded_imgs
            new_bboxes = [
                _sanitize_bbox_norm(
                    bboxes_norm[idx] if bboxes_norm and idx < len(bboxes_norm) else None
                )
                for idx in range(len(to_add))
            ]
            self.ensure_models_loaded()

            new_files: List[str] = []
            for idx, img in enumerate(to_add):
                fname = f"{item_id}_add_{idx}_{int(time.time() * 1000)}.jpg"
                out_path = self.storage_dir / fname
                ok = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    raise RuntimeError("Failed to save reference image to disk.")
                new_files.append(fname)

            record.image_files.extend(new_files)
            record.image_bboxes.extend(new_bboxes)
            self._compute_item_features(record)
            self._prompt_version += 1
            self._active_prompt_signature = None
            self._save_metadata_to_disk()

            logger.info(
                "Added %d photo(s) to custom item '%s' (total=%d).",
                len(new_files),
                record.name,
                len(record.image_files),
            )
            return record.to_schema()

    def delete_item(self, item_id: str) -> bool:
        """Delete a custom item and its stored reference photos."""
        with self._lock:
            record = self._items.pop(item_id, None)
            if record is None:
                return False

            for fname in record.image_files:
                (self.storage_dir / fname).unlink(missing_ok=True)

            self._prompt_version += 1
            self._active_prompt_signature = None
            self._save_metadata_to_disk()
            logger.info("Deleted custom item '%s' (id=%s).", record.name, item_id)
            return True

    def reset_tracker(self) -> None:
        """Reset internal ByteTrack tracker on the YOLOE model if loaded."""
        with self._lock:
            if (
                self._yoloe_model is not None
                and hasattr(self._yoloe_model, "predictor")
                and self._yoloe_model.predictor is not None
            ):
                trackers = getattr(self._yoloe_model.predictor, "trackers", [])
                for tracker in trackers:
                    if hasattr(tracker, "reset"):
                        try:
                            tracker.reset()
                        except Exception:
                            pass
                    if hasattr(tracker, "reset_id"):
                        try:
                            tracker.reset_id()
                        except Exception:
                            pass

    def _sync_active_prompts(self, active_items: List[CustomItemRecord]) -> None:
        """Configure YOLOE model classes & combined VPE when the active custom item set changes."""
        sig = (str(self._prompt_version),) + tuple(item.id for item in active_items)
        if self._active_prompt_signature == sig:
            return

        for item in active_items:
            if not item.features_ready or item.vpe is None:
                self._compute_item_features(item)

        names = [item.name for item in active_items]
        combined_vpe = torch.cat([item.vpe for item in active_items], dim=1)
        self._yoloe_model.set_classes(names, combined_vpe)
        self._active_prompt_signature = sig

    def detect_bgr(
        self,
        image_bgr: np.ndarray,
        item_ids: Sequence[str],
        confidence: float = 0.35,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[List[DetectionItem], float]:
        """
        Run YOLOE visual-prompted detection on a BGR frame for the requested custom item IDs,
        then verify each candidate crop with MobileNetV3-Large + HSV + Background-Rejection Guard.
        """
        if not item_ids:
            return [], 0.0

        sim_thresh = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        with self._lock:
            active_items = [self._items[iid] for iid in item_ids if iid in self._items]
            if not active_items:
                return [], 0.0

            self.ensure_models_loaded()
            self._sync_active_prompts(active_items)

            img_h, img_w = image_bgr.shape[:2]
            frame_area = float(max(1, img_w * img_h))
            proposal_conf = max(0.15, min(float(confidence) * 0.75, 0.50))

            start_t = time.perf_counter()
            try:
                results = self._run_yoloe_inference(image_bgr, proposal_conf)
            except Exception as exc:
                if self.device != "cpu":
                    logger.warning(
                        "YOLOE inference failed on '%s' (%s). Falling back to CPU.",
                        self.device,
                        exc,
                    )
                    self.device = "cpu"
                    if self._embedder_model is not None:
                        self._embedder_model = self._embedder_model.to("cpu")
                    results = self._run_yoloe_inference(image_bgr, proposal_conf)
                else:
                    raise

            # Extract candidate boxes + crops for batch verification
            candidates: List[Tuple[CustomItemRecord, float, List[float], Optional[int], np.ndarray]] = []
            if results:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy_arr = boxes.xyxy.detach().cpu().numpy()
                    conf_arr = boxes.conf.detach().cpu().numpy()
                    cls_arr = boxes.cls.detach().cpu().numpy()
                    raw_ids = getattr(boxes, "id", None)
                    id_arr = raw_ids.detach().cpu().numpy() if raw_ids is not None else None

                    for i in range(len(xyxy_arr)):
                        local_cls_idx = int(cls_arr[i])
                        if local_cls_idx < 0 or local_cls_idx >= len(active_items):
                            continue
                        item = active_items[local_cls_idx]
                        yoloe_conf = float(conf_arr[i])

                        x1 = max(0.0, min(float(xyxy_arr[i][0]), float(img_w)))
                        y1 = max(0.0, min(float(xyxy_arr[i][1]), float(img_h)))
                        x2 = max(0.0, min(float(xyxy_arr[i][2]), float(img_w)))
                        y2 = max(0.0, min(float(xyxy_arr[i][3]), float(img_h)))

                        # Background Guard Layer 1: Reject oversized full-frame background boxes
                        box_area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                        if box_area_ratio > MAX_LIVE_BOX_AREA_RATIO:
                            continue

                        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                        if ix2 - ix1 < 12 or iy2 - iy1 < 12:
                            continue

                        crop = image_bgr[iy1:iy2, ix1:ix2]
                        if crop.size == 0:
                            continue

                        tid: Optional[int] = None
                        if id_arr is not None and i < len(id_arr):
                            tid = CUSTOM_TRACK_ID_OFFSET + int(id_arr[i])

                        candidates.append(
                            (
                                item,
                                yoloe_conf,
                                [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                                tid,
                                crop,
                            )
                        )

            detections: List[DetectionItem] = []
            if candidates:
                crop_embeddings = self._extract_batch_embeddings([c[4] for c in candidates])
                for idx, (item, yoloe_conf, bbox, tid, crop) in enumerate(candidates):
                    crop_emb = crop_embeddings[idx]
                    crop_hist = _compute_central_hsv_hist(crop)

                    # Background Guard Layer 2: Tight foreground appearance + color match
                    deep_fg_sim = (
                        max(float(np.dot(crop_emb, ref_emb)) for ref_emb in item.ref_embeddings)
                        if item.ref_embeddings
                        else 0.0
                    )
                    deep_fg_sim = max(0.0, min(1.0, deep_fg_sim))

                    color_fg_sim = (
                        max(float(np.dot(crop_hist, ref_h)) for ref_h in item.ref_hists)
                        if item.ref_hists
                        else 0.0
                    )
                    color_fg_sim = max(0.0, min(1.0, color_fg_sim))

                    fg_sim = 0.72 * deep_fg_sim + 0.28 * color_fg_sim
                    if fg_sim < sim_thresh:
                        continue

                    # Background Guard Layer 3: Foreground vs excluded background contrast check
                    if item.ref_bg_embeddings and item.ref_bg_hists:
                        deep_bg_sim = max(
                            float(np.dot(crop_emb, bg_emb))
                            for bg_emb in item.ref_bg_embeddings
                        )
                        deep_bg_sim = max(0.0, min(1.0, deep_bg_sim))
                        color_bg_sim = max(
                            float(np.dot(crop_hist, bg_h))
                            for bg_h in item.ref_bg_hists
                        )
                        color_bg_sim = max(0.0, min(1.0, color_bg_sim))
                        bg_sim = 0.72 * deep_bg_sim + 0.28 * color_bg_sim

                        if fg_sim < bg_sim + BG_CONTRAST_MARGIN:
                            continue

                    blended_conf = max(0.0, min(1.0, 0.45 * yoloe_conf + 0.55 * fg_sim))
                    detections.append(
                        DetectionItem(
                            label=item.name,
                            class_id=item.class_id,
                            confidence=round(blended_conf, 4),
                            bbox=bbox,
                            track_id=tid,
                            is_custom=True,
                            custom_item_id=item.id,
                            similarity=round(fg_sim, 4),
                        )
                    )

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections, round(elapsed_ms, 2)

    def _run_yoloe_inference(self, image_bgr: np.ndarray, proposal_conf: float) -> Any:
        try:
            return self._yoloe_model.track(
                source=image_bgr,
                conf=proposal_conf,
                iou=0.45,
                max_det=30,
                imgsz=480,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                agnostic_nms=True,
                verbose=False,
            )
        except Exception as track_exc:
            logger.debug("YOLOE track fallback to predict: %s", track_exc)
            return self._yoloe_model.predict(
                source=image_bgr,
                conf=proposal_conf,
                iou=0.45,
                max_det=30,
                imgsz=480,
                device=self.device,
                agnostic_nms=True,
                verbose=False,
            )

    def unload(self) -> None:
        with self._lock:
            self._yoloe_model = None
            self._embedder_model = None
            self._models_loaded = False
            self._active_prompt_signature = None
