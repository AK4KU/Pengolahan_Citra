"""
Screen Capture Module — High-performance screen capture using DXcam.

Provides real-time screen capture optimized for FPS game analysis.
Uses DXcam (DirectX-based) for minimal-latency capture on Windows,
with an automatic fallback to mss if DXcam is unavailable.

Typical usage:
    with ScreenCapture(target_fps=60) as cap:
        cap.start()
        frame, timestamp = cap.get_latest_frame()
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from threading import Lock

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------
_DXCAM_AVAILABLE = False
_MSS_AVAILABLE = False

try:
    import dxcam  # type: ignore[import-untyped]
    _DXCAM_AVAILABLE = True
except ImportError:
    logger.info("dxcam not available — will attempt mss fallback.")

try:
    import mss  # type: ignore[import-untyped]
    import mss.tools  # type: ignore[import-untyped]
    _MSS_AVAILABLE = True
except ImportError:
    pass

if not _DXCAM_AVAILABLE and not _MSS_AVAILABLE:
    raise ImportError(
        "Neither dxcam nor mss is installed.  "
        "Install at least one: pip install dxcam   OR   pip install mss"
    )


class ScreenCapture:
    """High-performance screen capture for FPS game analysis.

    Attributes:
        target_fps: Desired capture framerate.
        region: Capture region as ``(left, top, right, bottom)`` or
            ``None`` for the full screen.
        backend: The active capture backend (``"dxcam"`` or ``"mss"``).
    """

    # ------------------------------------------------------------------
    # Construction / Teardown
    # ------------------------------------------------------------------
    def __init__(
        self,
        target_fps: int = settings.CAPTURE_FPS,
        region: Optional[Tuple[int, int, int, int]] = settings.CAPTURE_REGION,
    ) -> None:
        """Initialise the screen-capture pipeline.

        Args:
            target_fps: Target capture framerate (default from settings).
            region: ``(left, top, right, bottom)`` pixel region to
                capture, or ``None`` for the full screen.
        """
        self.target_fps: int = target_fps
        self.region: Optional[Tuple[int, int, int, int]] = region
        self._is_capturing: bool = False
        self._lock: Lock = Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0

        # Select and initialise the backend
        if _DXCAM_AVAILABLE:
            self.backend: str = "dxcam"
            self._init_dxcam()
        else:
            self.backend = "mss"
            self._init_mss()

        logger.info(
            "ScreenCapture initialised — backend=%s, fps=%d, region=%s",
            self.backend,
            self.target_fps,
            self.region,
        )

    # --- DXcam -----------------------------------------------------------
    def _init_dxcam(self) -> None:
        """Create the DXcam camera object."""
        self._camera = dxcam.create(output_color="BGR")  # type: ignore[attr-defined]

    # --- mss -------------------------------------------------------------
    def _init_mss(self) -> None:
        """Create the mss screen-capture context."""
        self._sct = mss.mss()
        if self.region is not None:
            left, top, right, bottom = self.region
            self._mss_monitor = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }
        else:
            self._mss_monitor = self._sct.monitors[1]  # primary monitor

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "ScreenCapture":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit the context manager — ensures capture is stopped."""
        self.stop()

    # ------------------------------------------------------------------
    # Start / Stop continuous capture
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start continuous screen capture.

        For the DXcam backend this launches its internal capture thread.
        For mss, this simply marks the capture as active (frames are
        grabbed on demand in :meth:`get_latest_frame`).
        """
        if self._is_capturing:
            logger.warning("Capture is already running.")
            return

        if self.backend == "dxcam":
            self._camera.start(
                target_fps=self.target_fps,
                region=self.region,
            )

        self._is_capturing = True
        logger.info("Screen capture started (backend=%s).", self.backend)

    def stop(self) -> None:
        """Stop continuous screen capture and release resources."""
        if not self._is_capturing:
            return

        if self.backend == "dxcam":
            try:
                self._camera.stop()
            except Exception:  # noqa: BLE001
                logger.debug("DXcam camera stop raised an exception (ignored).")

        self._is_capturing = False
        logger.info("Screen capture stopped.")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    def grab_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Grab a single frame (one-shot, does **not** require :meth:`start`).

        Returns:
            A tuple ``(frame, timestamp)`` where *frame* is an
            ``np.ndarray`` in BGR format (H × W × 3) and *timestamp* is
            the ``time.perf_counter()`` value at capture time.  If the
            capture fails, ``(None, timestamp)`` is returned.
        """
        timestamp = time.perf_counter()

        try:
            if self.backend == "dxcam":
                frame = self._camera.grab(region=self.region)
                if frame is None:
                    logger.debug("DXcam grab returned None.")
                    return None, timestamp
                # DXcam already returns BGR when output_color="BGR"
                return np.asarray(frame), timestamp

            # mss fallback
            screenshot = self._sct.grab(self._mss_monitor)
            # mss returns BGRA — drop alpha channel
            frame = np.asarray(screenshot)[:, :, :3].copy()
            return frame, timestamp

        except Exception:
            logger.exception("Frame grab failed.")
            return None, timestamp

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame from continuous capture.

        If the DXcam backend is active, this returns the most recent
        frame buffered by its capture thread.  For the mss backend,
        this is equivalent to :meth:`grab_frame`.

        Returns:
            ``(frame, timestamp)`` — see :meth:`grab_frame`.
        """
        if not self._is_capturing:
            logger.warning(
                "get_latest_frame called but capture is not running. "
                "Falling back to grab_frame."
            )
            return self.grab_frame()

        timestamp = time.perf_counter()

        try:
            if self.backend == "dxcam":
                frame = self._camera.get_latest_frame()
                if frame is None:
                    return None, timestamp
                frame = np.asarray(frame)
            else:
                screenshot = self._sct.grab(self._mss_monitor)
                frame = np.asarray(screenshot)[:, :, :3].copy()

            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = timestamp

            return frame, timestamp

        except Exception:
            logger.exception("get_latest_frame failed.")
            return None, timestamp

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @property
    def is_capturing(self) -> bool:
        """Whether continuous capture is currently active."""
        return self._is_capturing

    def __repr__(self) -> str:
        return (
            f"ScreenCapture(backend={self.backend!r}, "
            f"target_fps={self.target_fps}, "
            f"region={self.region}, "
            f"capturing={self._is_capturing})"
        )
