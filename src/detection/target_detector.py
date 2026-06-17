"""
Target Detector Module — Enemy and target detection with tracking.

Wraps :class:`YOLODetector` to filter only game-relevant detections
(e.g. ``enemy_head``, ``enemy_body``, ``target``), compute each
target's distance to the crosshair, classify bounding-box regions,
and track target visibility across frames.

Typical usage:
    target_det = TargetDetector(yolo_detector)
    targets = target_det.detect(frame, crosshair=(960, 540))
    for t in targets:
        print(t.class_name, t.distance_to_crosshair, t.is_visible)
"""

import sys
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

from src.detection.yolo_detector import Detection, YOLODetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default target classes
# ---------------------------------------------------------------------------
_DEFAULT_TARGET_CLASSES: Set[str] = {"enemy_head", "enemy_body", "target", "head", "body"}


# ============================================================================
# TargetDetection data class
# ============================================================================
@dataclass(frozen=True, slots=True)
class TargetDetection:
    """An enemy/target detection enriched with game-context metadata.

    Extends the base :class:`Detection` fields with spatial and
    temporal information.

    Attributes:
        class_id: Integer class label.
        class_name: Human-readable class name.
        confidence: Detection confidence in ``[0, 1]``.
        bbox: Bounding box ``(x1, y1, x2, y2)``.
        center: Centre of the bounding box ``(cx, cy)``.
        width: Bounding-box width in pixels.
        height: Bounding-box height in pixels.
        is_visible: Whether the target is currently visible.
        region: Body region — ``"head"``, ``"body"``, or ``"unknown"``.
        distance_to_crosshair: Euclidean distance (px) from the
            target centre to the crosshair position.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    is_visible: bool
    region: str
    distance_to_crosshair: float

    # Computed fields
    center: Tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        object.__setattr__(self, "center", ((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        object.__setattr__(self, "width", x2 - x1)
        object.__setattr__(self, "height", y2 - y1)


# ============================================================================
# TargetDetector
# ============================================================================
class TargetDetector:
    """Detect and track game targets (enemies / aim-training targets).

    Wraps a :class:`YOLODetector`, filters detections to target-related
    classes, annotates each with region and distance information, and
    tracks visibility across consecutive frames.

    Attributes:
        yolo_detector: The underlying YOLO inference engine.
        target_classes: Set of class names considered as targets.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        yolo_detector: YOLODetector,
        target_classes: Optional[List[str]] = None,
    ) -> None:
        """Initialise the target detector.

        Args:
            yolo_detector: A :class:`YOLODetector` instance for
                running inference.
            target_classes: Class names to keep.  If ``None``, the
                default set ``{"enemy_head", "enemy_body", "target"}``
                is used.
        """
        self.yolo_detector: YOLODetector = yolo_detector
        self.target_classes: Set[str] = (
            set(target_classes) if target_classes else _DEFAULT_TARGET_CLASSES
        )

        # Visibility tracking — keys are class_name strings
        self._prev_visible: Set[str] = set()
        self._visibility_history: Dict[str, List[bool]] = {}

        logger.info(
            "TargetDetector initialised — target_classes=%s",
            sorted(self.target_classes),
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        crosshair: Optional[Tuple[float, float]] = None,
    ) -> List[TargetDetection]:
        """Detect targets in *frame* and compute per-target metadata.

        Args:
            frame: BGR image as ``np.ndarray`` (H × W × 3).
            crosshair: ``(x, y)`` crosshair position.  If ``None``,
                the screen centre from settings is used.

        Returns:
            A list of :class:`TargetDetection` objects sorted by
            distance to the crosshair (nearest first).
        """
        if crosshair is None:
            crosshair = (
                float(settings.CROSSHAIR_DEFAULT_X),
                float(settings.CROSSHAIR_DEFAULT_Y),
            )

        # Run YOLO inference
        raw_detections: List[Detection] = self.yolo_detector.detect(frame)

        # Filter to target classes
        filtered: List[Detection] = [
            d for d in raw_detections if d.class_name in self.target_classes
        ]

        # Build enriched detections
        current_visible: Set[str] = set()
        target_detections: List[TargetDetection] = []

        for det in filtered:
            region = self._classify_region(det.class_name)
            distance = self._compute_distance(det.center, crosshair)

            td = TargetDetection(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                is_visible=True,
                region=region,
                distance_to_crosshair=distance,
            )
            target_detections.append(td)
            current_visible.add(det.class_name)

        # Update visibility tracking
        self._update_visibility(current_visible)

        # Sort by distance (nearest first)
        target_detections.sort(key=lambda t: t.distance_to_crosshair)

        return target_detections

    # ------------------------------------------------------------------
    # Region classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_region(class_name: str) -> str:
        """Map a class name to a body region.

        Args:
            class_name: Detected class name.

        Returns:
            ``"head"``, ``"body"``, or ``"unknown"``.
        """
        lower = class_name.lower()
        if "head" in lower:
            return "head"
        if "body" in lower:
            return "body"
        return "unknown"

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_distance(
        target_center: Tuple[float, float],
        crosshair: Tuple[float, float],
    ) -> float:
        """Compute Euclidean distance between two 2-D points.

        Args:
            target_center: ``(x, y)`` of the target centre.
            crosshair: ``(x, y)`` of the crosshair.

        Returns:
            Distance in pixels.
        """
        dx = target_center[0] - crosshair[0]
        dy = target_center[1] - crosshair[1]
        return math.sqrt(dx * dx + dy * dy)

    # ------------------------------------------------------------------
    # Visibility tracking
    # ------------------------------------------------------------------
    def _update_visibility(self, current_visible: Set[str]) -> None:
        """Track which target classes appeared or disappeared.

        Logs transitions at DEBUG level and updates the internal
        visibility history.

        Args:
            current_visible: Set of class names visible in the
                current frame.
        """
        appeared = current_visible - self._prev_visible
        disappeared = self._prev_visible - current_visible

        if appeared:
            logger.debug("Targets appeared: %s", appeared)
        if disappeared:
            logger.debug("Targets disappeared: %s", disappeared)

        # Update history
        all_classes = current_visible | self._prev_visible
        for cls in all_classes:
            self._visibility_history.setdefault(cls, []).append(
                cls in current_visible
            )

        self._prev_visible = current_visible

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_visibility_history(self) -> Dict[str, List[bool]]:
        """Return the full visibility history for each tracked class.

        Returns:
            A dict mapping class names to lists of boolean values
            (``True`` = visible in that frame).
        """
        return dict(self._visibility_history)

    def reset_tracking(self) -> None:
        """Clear all accumulated visibility-tracking state."""
        self._prev_visible.clear()
        self._visibility_history.clear()
        logger.debug("Visibility tracking state reset.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TargetDetector("
            f"target_classes={sorted(self.target_classes)}, "
            f"tracked_classes={len(self._visibility_history)})"
        )
