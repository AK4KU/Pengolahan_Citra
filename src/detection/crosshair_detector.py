"""
Crosshair Detector Module — Locate the player's crosshair on screen.

Three detection strategies are available (tried in order of priority):

1. **Screen-center assumption** (default for Valorant) — the crosshair
   is always at the exact screen centre ``(960, 540)`` for 1920 × 1080.
2. **YOLO detection** — use a :class:`YOLODetector` to find a
   ``crosshair`` class in the frame.
3. **Template matching** — OpenCV ``matchTemplate`` with a user-
   supplied crosshair template image.

Typical usage:
    detector = CrosshairDetector(use_screen_center=True)
    x, y, conf = detector.detect(frame)
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

if TYPE_CHECKING:
    from src.detection.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class CrosshairDetector:
    """Detect the crosshair position within a game frame.

    Attributes:
        use_screen_center: If ``True``, always return the screen-centre
            coordinates (fastest, ideal for Valorant).
        yolo_detector: Optional :class:`YOLODetector` used for
            crosshair-class detection.
        template: Optional template image for ``matchTemplate``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        yolo_detector: Optional["YOLODetector"] = None,
        use_screen_center: bool = settings.CROSSHAIR_USE_SCREEN_CENTER,
        template_path: Optional[str] = None,
    ) -> None:
        """Initialise the crosshair detector.

        Args:
            yolo_detector: An existing :class:`YOLODetector` instance.
                If provided and *use_screen_center* is ``False``, the
                detector will attempt YOLO-based crosshair detection.
            use_screen_center: Use the fixed screen-centre position.
                Default is ``True`` for Valorant.
            template_path: Path to a crosshair template image (PNG or
                JPG).  Used as a secondary fallback via OpenCV
                ``matchTemplate``.
        """
        self.use_screen_center: bool = use_screen_center
        self.yolo_detector: Optional["YOLODetector"] = yolo_detector
        self.template: Optional[np.ndarray] = None

        # Pre-compute screen-centre coordinates
        self._center_x: float = float(settings.CROSSHAIR_DEFAULT_X)
        self._center_y: float = float(settings.CROSSHAIR_DEFAULT_Y)

        # Load template if provided
        if template_path is not None:
            self._load_template(template_path)

        strategy = "screen_center" if self.use_screen_center else "yolo+template"
        logger.info(
            "CrosshairDetector initialised — strategy=%s, "
            "template=%s, yolo=%s",
            strategy,
            template_path is not None,
            yolo_detector is not None,
        )

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------
    def _load_template(self, template_path: str) -> None:
        """Load a crosshair template image from disk.

        Args:
            template_path: Path to the template image.
        """
        path = Path(template_path)
        if not path.exists():
            logger.warning("Template image not found: %s", template_path)
            return

        template = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if template is None:
            logger.warning("Failed to read template image: %s", template_path)
            return

        self.template = template
        logger.info(
            "Crosshair template loaded (%dx%d) from %s.",
            template.shape[1],
            template.shape[0],
            path.name,
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """Detect the crosshair position within *frame*.

        The method tries the following strategies in order:

        1. Screen-centre (if ``use_screen_center`` is ``True``).
        2. YOLO detection of the ``crosshair`` class.
        3. Template matching (if a template is loaded).
        4. Fallback to screen centre with confidence ``0.5``.

        Args:
            frame: BGR image as ``np.ndarray`` (H × W × 3).

        Returns:
            ``(x, y, confidence)`` where *(x, y)* is the estimated
            crosshair position in pixel coordinates and *confidence*
            is in ``[0, 1]``.
        """
        # Strategy 1: screen centre
        if self.use_screen_center:
            return self._center_x, self._center_y, 1.0

        # Strategy 2: YOLO
        if self.yolo_detector is not None:
            result = self._detect_with_yolo(frame)
            if result is not None:
                return result

        # Strategy 3: template matching
        if self.template is not None:
            result = self._detect_with_template(frame)
            if result is not None:
                return result

        # Strategy 4: fallback
        logger.debug("All crosshair strategies failed — using screen centre.")
        return self._center_x, self._center_y, 0.5

    # ------------------------------------------------------------------
    # YOLO strategy
    # ------------------------------------------------------------------
    def _detect_with_yolo(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Attempt to locate the crosshair via YOLO inference.

        Args:
            frame: BGR image.

        Returns:
            ``(x, y, confidence)`` if a crosshair-class detection is
            found, otherwise ``None``.
        """
        try:
            detections = self.yolo_detector.detect(frame)  # type: ignore[union-attr]

            # Look for the "crosshair" class
            crosshair_class_name = settings.CLASS_NAMES.get(0, "crosshair")
            for det in detections:
                if det.class_name == crosshair_class_name:
                    cx, cy = det.center
                    logger.debug(
                        "YOLO crosshair found at (%.1f, %.1f) conf=%.2f",
                        cx, cy, det.confidence,
                    )
                    return cx, cy, det.confidence

        except Exception:
            logger.warning(
                "YOLO crosshair detection failed.", exc_info=True
            )

        return None

    # ------------------------------------------------------------------
    # Template-matching strategy
    # ------------------------------------------------------------------
    def _detect_with_template(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Locate the crosshair via OpenCV ``matchTemplate``.

        Searches within a region around the screen centre (bounded by
        ``settings.CROSSHAIR_SEARCH_RADIUS``) for efficiency.

        Args:
            frame: BGR image.

        Returns:
            ``(x, y, confidence)`` if the match confidence is
            above ``0.5``, otherwise ``None``.
        """
        try:
            h, w = frame.shape[:2]
            th, tw = self.template.shape[:2]  # type: ignore[union-attr]

            # Define search region around screen centre
            radius = settings.CROSSHAIR_SEARCH_RADIUS
            cx_int = int(self._center_x)
            cy_int = int(self._center_y)
            x1 = max(0, cx_int - radius - tw // 2)
            y1 = max(0, cy_int - radius - th // 2)
            x2 = min(w, cx_int + radius + tw // 2)
            y2 = min(h, cy_int + radius + th // 2)

            roi = frame[y1:y2, x1:x2]

            # Ensure ROI is large enough for the template
            if roi.shape[0] < th or roi.shape[1] < tw:
                return None

            result = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < 0.5:
                return None

            # Convert local coordinates back to full-frame coordinates
            match_x = float(x1 + max_loc[0] + tw / 2.0)
            match_y = float(y1 + max_loc[1] + th / 2.0)

            logger.debug(
                "Template match at (%.1f, %.1f) conf=%.2f",
                match_x, match_y, max_val,
            )
            return match_x, match_y, float(max_val)

        except Exception:
            logger.warning(
                "Template matching failed.", exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"CrosshairDetector("
            f"use_screen_center={self.use_screen_center}, "
            f"yolo={'yes' if self.yolo_detector else 'no'}, "
            f"template={'yes' if self.template is not None else 'no'})"
        )
