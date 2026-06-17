"""
YOLO Detector Module — Ultralytics YOLO wrapper for object detection.

Provides a clean interface around ``ultralytics.YOLO`` with automatic
device selection, FP16 (half-precision) support for the NVIDIA GTX 1650,
and a structured :class:`Detection` dataclass for results.

Typical usage:
    detector = YOLODetector(model_path="models/best.pt")
    detections = detector.detect(frame)
    for det in detections:
        print(det.class_name, det.confidence, det.bbox)
"""

import sys
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Detection data class
# ============================================================================
@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection result.

    Attributes:
        class_id: Integer class label.
        class_name: Human-readable class name.
        confidence: Detection confidence in ``[0, 1]``.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel
            coordinates (top-left, bottom-right).
        center: Centre of the bounding box ``(cx, cy)``.
        width: Bounding-box width in pixels.
        height: Bounding-box height in pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        # Bypass frozen dataclass restrictions
        object.__setattr__(self, "center", ((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        object.__setattr__(self, "width", x2 - x1)
        object.__setattr__(self, "height", y2 - y1)


# ============================================================================
# YOLODetector
# ============================================================================
class YOLODetector:
    """Wrapper around Ultralytics YOLO for FPS-game object detection.

    Handles model loading, device selection, and inference.  Returns
    structured :class:`Detection` objects instead of raw tensors.

    Attributes:
        model_path: Path to the YOLO weights file.
        confidence: Confidence threshold for detections.
        device: Inference device (e.g. ``"0"``, ``"cpu"``).
        half: Whether FP16 half-precision inference is enabled.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = settings.YOLO_CONFIDENCE,
        device: str = "auto",
        half: bool = settings.YOLO_HALF_PRECISION,
    ) -> None:
        """Load the YOLO model.

        Args:
            model_path: Path to a ``.pt`` weights file.  If ``None``,
                the path from ``settings.YOLO_MODEL_PATH`` is used.
                If the file does not exist, the pretrained
                ``yolov8n.pt`` fallback is loaded with a warning.
            confidence: Minimum confidence threshold.
            device: ``"auto"`` for automatic GPU/CPU selection,
                ``"0"`` for the first CUDA GPU, or ``"cpu"``.
            half: Enable FP16 half-precision (recommended for
                NVIDIA GTX 1650).
        """
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self.confidence: float = confidence
        self.half: bool = half

        # --- Resolve model path ------------------------------------------
        if model_path is None:
            model_path = settings.YOLO_MODEL_PATH

        resolved = Path(model_path)
        if resolved.exists():
            self.model_path: str = str(resolved)
        else:
            warnings.warn(
                f"Model not found at '{model_path}'. "
                f"Falling back to pretrained '{settings.YOLO_FALLBACK_MODEL}'.",
                UserWarning,
                stacklevel=2,
            )
            self.model_path = settings.YOLO_FALLBACK_MODEL

        # --- Select device -----------------------------------------------
        self.device: str = self._select_device(device)

        # --- Load model --------------------------------------------------
        logger.info(
            "Loading YOLO model: %s  (device=%s, half=%s, conf=%.2f)",
            self.model_path,
            self.device,
            self.half,
            self.confidence,
        )
        self._model = YOLO(self.model_path)

        # FP16 is only supported on CUDA
        if self.half and self.device == "cpu":
            logger.warning(
                "FP16 half-precision is not supported on CPU — disabling."
            )
            self.half = False

        # Warm-up: run a dummy inference to initialise CUDA kernels
        self._warmup()

        logger.info("YOLODetector ready.")

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_device(device: str) -> str:
        """Return a valid device string.

        ``"auto"`` resolves to ``"0"`` when a CUDA GPU is present, and
        ``"cpu"`` otherwise.
        """
        if device == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("CUDA GPU detected: %s", gpu_name)
                return "0"
            logger.info("No CUDA GPU detected — using CPU.")
            return "cpu"
        return device

    def _warmup(self) -> None:
        """Run a single dummy inference to warm up CUDA kernels."""
        try:
            dummy = np.zeros(
                (settings.YOLO_IMAGE_SIZE, settings.YOLO_IMAGE_SIZE, 3),
                dtype=np.uint8,
            )
            self._model.predict(
                dummy,
                conf=self.confidence,
                device=self.device,
                half=self.half,
                verbose=False,
                imgsz=settings.YOLO_IMAGE_SIZE,
            )
            logger.debug("Warm-up inference completed.")
        except Exception:
            logger.warning("Warm-up inference failed (non-critical).", exc_info=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO inference on a single frame.

        Args:
            frame: BGR image as an ``np.ndarray`` (H × W × 3).

        Returns:
            A list of :class:`Detection` objects sorted by confidence
            (highest first).
        """
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=settings.YOLO_IOU_THRESHOLD,
            device=self.device,
            half=self.half,
            verbose=False,
            imgsz=settings.YOLO_IMAGE_SIZE,
        )

        return self._parse_results(results)

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Detection]]:
        """Run YOLO inference on a batch of frames.

        Args:
            frames: List of BGR images.

        Returns:
            A list (one per frame) of detection lists.
        """
        if not frames:
            return []

        results = self._model.predict(
            frames,
            conf=self.confidence,
            iou=settings.YOLO_IOU_THRESHOLD,
            device=self.device,
            half=self.half,
            verbose=False,
            imgsz=settings.YOLO_IMAGE_SIZE,
        )

        batch_detections: List[List[Detection]] = []
        for result in results:
            batch_detections.append(self._parse_results([result]))

        return batch_detections

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------
    def _parse_results(self, results) -> List[Detection]:  # noqa: ANN001
        """Convert Ultralytics result objects into :class:`Detection` list.

        Args:
            results: Output from ``model.predict()``.

        Returns:
            Sorted list of :class:`Detection` (highest confidence first).
        """
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Move tensors to CPU / numpy
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            # Build the class-name mapping from the model if available,
            # otherwise fall back to the project-level settings map.
            model_names = getattr(result, "names", None) or {}

            for i in range(len(xyxy)):
                class_id = int(cls_ids[i])
                class_name = (
                    model_names.get(class_id)
                    or settings.CLASS_NAMES.get(class_id, f"class_{class_id}")
                )
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confs[i]),
                        bbox=(
                            float(xyxy[i][0]),
                            float(xyxy[i][1]),
                            float(xyxy[i][2]),
                            float(xyxy[i][3]),
                        ),
                    )
                )

        # Sort highest confidence first
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def class_names(self) -> dict:
        """Return the class-name map used by the loaded model."""
        return getattr(self._model, "names", settings.CLASS_NAMES)

    def __repr__(self) -> str:
        return (
            f"YOLODetector(model={self.model_path!r}, "
            f"device={self.device!r}, "
            f"half={self.half}, "
            f"conf={self.confidence})"
        )
