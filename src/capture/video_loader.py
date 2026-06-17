"""
Video Loader Module — Frame-by-frame video file processing with OpenCV.

Supports MP4, AVI, and MKV formats.  Provides iteration with optional
frame-skipping (``sample_rate``), random access via :meth:`get_frame`,
and rich metadata queries.

Typical usage:
    loader = VideoLoader("gameplay.mp4", sample_rate=2)
    for frame, timestamp, idx in loader:
        process(frame)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: F401 — imported for potential future use

logger = logging.getLogger(__name__)

# Supported container formats
_SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv"}


class VideoLoader:
    """Load and iterate through video files for offline analysis.

    Attributes:
        video_path: Resolved path to the video file.
        sample_rate: Take every *n*-th frame (1 = every frame).
    """

    # ------------------------------------------------------------------
    # Construction / Teardown
    # ------------------------------------------------------------------
    def __init__(
        self,
        video_path: str | Path,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Open a video file for processing.

        Args:
            video_path: Path to the video file (MP4, AVI, or MKV).
            sample_rate: If given, yield every *n*-th frame during
                iteration (e.g. ``2`` yields every second frame).
                ``None`` or ``1`` yields every frame.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            ValueError: If the extension is unsupported or OpenCV
                cannot open the file.
        """
        self.video_path: Path = Path(video_path).resolve()

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        suffix = self.video_path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{suffix}'.  "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        self.sample_rate: int = max(1, sample_rate or 1)

        # Open the video
        self._cap: cv2.VideoCapture = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(
                f"OpenCV failed to open video: {self.video_path}"
            )

        # Cache metadata
        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "VideoLoader opened %s — %dx%d @ %.1f FPS, %d frames (%.1fs)",
            self.video_path.name,
            self._width,
            self._height,
            self._fps,
            self._total_frames,
            self.duration,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return self._total_frames

    @property
    def duration(self) -> float:
        """Duration of the video in seconds."""
        if self._fps <= 0:
            return 0.0
        return self._total_frames / self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_metadata(self) -> Dict[str, object]:
        """Return a dictionary of video metadata.

        Returns:
            A dict with keys ``fps``, ``total_frames``, ``duration``,
            and ``resolution`` (as ``(width, height)``).
        """
        return {
            "fps": self._fps,
            "total_frames": self._total_frames,
            "duration": self.duration,
            "resolution": (self._width, self._height),
        }

    # ------------------------------------------------------------------
    # Random access
    # ------------------------------------------------------------------
    def get_frame(self, index: int) -> Tuple[np.ndarray, float]:
        """Retrieve a specific frame by its zero-based index.

        Args:
            index: Frame index (0-based).

        Returns:
            ``(frame, timestamp)`` where *frame* is a BGR ``np.ndarray``
            and *timestamp* is the frame's position in seconds.

        Raises:
            IndexError: If *index* is out of range.
            RuntimeError: If the frame could not be read.
        """
        if index < 0 or index >= self._total_frames:
            raise IndexError(
                f"Frame index {index} out of range "
                f"[0, {self._total_frames - 1}]."
            )

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame at index {index}.")

        timestamp = index / self._fps if self._fps > 0 else 0.0
        return frame, timestamp

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int]]:
        """Iterate over frames, respecting :attr:`sample_rate`.

        Yields:
            ``(frame, timestamp, frame_index)`` for each sampled frame.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0

        while frame_index < self._total_frames:
            # Seek if we need to skip frames
            if self.sample_rate > 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.debug("End of video reached at frame %d.", frame_index)
                break

            timestamp = frame_index / self._fps if self._fps > 0 else 0.0
            yield frame, timestamp, frame_index

            frame_index += self.sample_rate

    def __len__(self) -> int:
        """Number of frames that will be yielded during iteration.

        Takes :attr:`sample_rate` into account.
        """
        if self._total_frames <= 0:
            return 0
        return (self._total_frames + self.sample_rate - 1) // self.sample_rate

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.release()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def release(self) -> None:
        """Release the underlying OpenCV ``VideoCapture``."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug("VideoCapture released for %s.", self.video_path.name)

    def __del__(self) -> None:
        """Ensure resources are freed on garbage collection."""
        try:
            self.release()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"VideoLoader(path={self.video_path.name!r}, "
            f"fps={self._fps:.1f}, "
            f"frames={self._total_frames}, "
            f"sample_rate={self.sample_rate})"
        )
