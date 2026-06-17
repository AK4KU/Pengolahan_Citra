"""
Position Tracker Module - Tracks crosshair and target positions frame-by-frame.

This module provides the PositionTracker class which records per-frame positions
of the crosshair and detected targets, computes distances between them, and
identifies engagement windows (continuous periods where a target is visible).

Engagement windows are segmented using a configurable gap threshold: if no target
is detected for ENGAGEMENT_GAP_FRAMES consecutive frames, the current engagement
is considered ended.

Author: FPS Aim Performance Analyzer
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class TargetDetection:
    """Represents a single detected target in a frame.

    Attributes:
        target_id: Unique identifier for this target across frames.
        x: Horizontal centre coordinate of the target bounding box (pixels).
        y: Vertical centre coordinate of the target bounding box (pixels).
        width: Width of the target bounding box (pixels).
        height: Height of the target bounding box (pixels).
        confidence: Detection confidence score in [0, 1].
        class_id: YOLO class index (see settings.CLASS_NAMES).
        class_name: Human-readable class name, e.g. ``"enemy_head"``.
    """

    target_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int = 3
    class_name: str = "target"


@dataclass
class EngagementWindow:
    """A contiguous period during which at least one target is visible.

    The window stores the crosshair and target positions observed during the
    engagement so that downstream modules can analyse aim behaviour within
    each engagement independently.

    Attributes:
        start_frame: Frame index where the engagement begins.
        end_frame: Frame index where the engagement ends (inclusive).
        start_time: Timestamp (seconds) of the first frame in the window.
        end_time: Timestamp (seconds) of the last frame in the window.
        target_id: Primary target identifier tracked during this engagement.
        crosshair_positions: Array of shape (N, 2) with crosshair (x, y) per frame.
        target_positions: Array of shape (N, 2) with target (x, y) per frame.
        timestamps: 1-D array of length N with timestamps for each frame.
    """

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    target_id: int
    crosshair_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    target_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))


# ===========================================================================
# Frame Record (internal)
# ===========================================================================

@dataclass
class _FrameRecord:
    """Internal per-frame record stored by PositionTracker."""

    frame_index: int
    timestamp: float
    crosshair_x: float
    crosshair_y: float
    targets: List[TargetDetection] = field(default_factory=list)


# ===========================================================================
# PositionTracker
# ===========================================================================

class PositionTracker:
    """Tracks crosshair and target positions across video frames.

    The tracker accumulates per-frame records and provides efficient
    bulk accessors that return NumPy arrays suitable for downstream
    analysis (trajectory building, metrics computation, visualisation).

    Example usage::

        tracker = PositionTracker()
        for idx, frame in enumerate(frames):
            ts = idx / 60.0
            crosshair = (960.0, 540.0)
            targets = [TargetDetection(target_id=1, x=800, y=400,
                                       width=40, height=60, confidence=0.9)]
            tracker.update(idx, ts, crosshair, targets)

        distances = tracker.get_distances()
        engagements = tracker.get_engagement_windows()
    """

    def __init__(self) -> None:
        """Initialise an empty tracking state."""

        self._frames: List[_FrameRecord] = []
        self._frame_index_set: set = set()
        logger.debug("PositionTracker initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_index: int,
        timestamp: float,
        crosshair_pos: Tuple[float, float],
        targets: List[TargetDetection],
    ) -> None:
        """Record crosshair and target positions for a single frame.

        Args:
            frame_index: Zero-based index of the current frame.
            timestamp: Timestamp of the frame in seconds from session start.
            crosshair_pos: ``(x, y)`` pixel coordinates of the crosshair.
            targets: List of :class:`TargetDetection` instances visible in
                this frame. May be empty.

        Raises:
            ValueError: If *frame_index* has already been recorded.
        """

        if frame_index in self._frame_index_set:
            raise ValueError(
                f"Frame {frame_index} has already been recorded. "
                "Call reset() before re-processing."
            )

        record = _FrameRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            crosshair_x=crosshair_pos[0],
            crosshair_y=crosshair_pos[1],
            targets=list(targets),
        )
        self._frames.append(record)
        self._frame_index_set.add(frame_index)

    def get_crosshair_positions(self) -> np.ndarray:
        """Return crosshair positions as an Nx3 array.

        Returns:
            ``np.ndarray`` of shape ``(N, 3)`` where each row is
            ``[timestamp, x, y]``. Returns an empty ``(0, 3)`` array
            when no frames have been recorded.
        """

        if not self._frames:
            return np.empty((0, 3), dtype=np.float64)

        data = np.array(
            [[f.timestamp, f.crosshair_x, f.crosshair_y] for f in self._frames],
            dtype=np.float64,
        )
        return data

    def get_target_positions(
        self, target_id: Optional[int] = None
    ) -> np.ndarray:
        """Return target positions as an Nx4 array.

        Args:
            target_id: If specified, only return rows for this target.
                If ``None``, return all target observations.

        Returns:
            ``np.ndarray`` of shape ``(N, 4)`` where each row is
            ``[timestamp, x, y, target_id]``. Returns an empty ``(0, 4)``
            array when no matching targets exist.
        """

        rows: List[List[float]] = []
        for frame in self._frames:
            for t in frame.targets:
                if target_id is not None and t.target_id != target_id:
                    continue
                rows.append([frame.timestamp, t.x, t.y, float(t.target_id)])

        if not rows:
            return np.empty((0, 4), dtype=np.float64)

        return np.array(rows, dtype=np.float64)

    def get_distances(self) -> np.ndarray:
        """Compute crosshair-to-nearest-target distance per frame.

        For frames where no target is visible the distance is recorded as
        ``np.nan``.

        Returns:
            ``np.ndarray`` of shape ``(N, 2)`` where each row is
            ``[timestamp, distance_to_nearest_target]``.
        """

        if not self._frames:
            return np.empty((0, 2), dtype=np.float64)

        data = np.empty((len(self._frames), 2), dtype=np.float64)

        for i, frame in enumerate(self._frames):
            data[i, 0] = frame.timestamp

            if not frame.targets:
                data[i, 1] = np.nan
                continue

            # Compute Euclidean distances to every target in this frame
            cx, cy = frame.crosshair_x, frame.crosshair_y
            min_dist = float("inf")
            for t in frame.targets:
                dist = np.sqrt((cx - t.x) ** 2 + (cy - t.y) ** 2)
                if dist < min_dist:
                    min_dist = dist

            data[i, 1] = min_dist

        return data

    def get_engagement_windows(self) -> List[EngagementWindow]:
        """Detect engagement windows from the recorded frame history.

        An engagement window is a contiguous sequence of frames during which
        at least one target is visible.  A gap of
        :pydata:`settings.ENGAGEMENT_GAP_FRAMES` or more consecutive frames
        without any target detection signals the end of an engagement.

        Within a single engagement the *primary* target is the target id that
        appears most frequently.

        Returns:
            List of :class:`EngagementWindow` instances sorted by start time.
        """

        gap_threshold: int = getattr(settings, "ENGAGEMENT_GAP_FRAMES", 15)

        if not self._frames:
            return []

        # Sort frames by frame_index to ensure chronological order
        sorted_frames = sorted(self._frames, key=lambda f: f.frame_index)

        engagements: List[EngagementWindow] = []
        current_segment: List[_FrameRecord] = []
        frames_since_last_target: int = 0

        for frame in sorted_frames:
            has_target = len(frame.targets) > 0

            if has_target:
                # If we were in a gap that exceeded threshold, flush segment
                if (
                    frames_since_last_target >= gap_threshold
                    and current_segment
                ):
                    engagements.append(
                        self._build_engagement(current_segment)
                    )
                    current_segment = []

                current_segment.append(frame)
                frames_since_last_target = 0
            else:
                if current_segment:
                    frames_since_last_target += 1
                    # Still within tolerance — include the frame
                    if frames_since_last_target < gap_threshold:
                        current_segment.append(frame)

        # Flush remaining segment
        if current_segment:
            engagements.append(self._build_engagement(current_segment))

        logger.info(
            "Detected %d engagement window(s) from %d frames.",
            len(engagements),
            len(self._frames),
        )
        return engagements

    def reset(self) -> None:
        """Clear all stored tracking data and reset the tracker state."""

        self._frames.clear()
        self._frame_index_set.clear()
        logger.debug("PositionTracker reset.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)

    @property
    def has_data(self) -> bool:
        """Whether at least one frame has been recorded."""
        return len(self._frames) > 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_engagement(segment: List[_FrameRecord]) -> EngagementWindow:
        """Construct an :class:`EngagementWindow` from a list of frame records.

        The primary target is determined by majority vote across frames.
        For frames without a target detection the most recent target position
        is forward-filled.

        Args:
            segment: Chronologically ordered list of ``_FrameRecord`` objects
                forming one engagement.

        Returns:
            A populated :class:`EngagementWindow`.
        """

        # Determine primary target (most frequent target_id)
        target_counts: Dict[int, int] = {}
        for frame in segment:
            for t in frame.targets:
                target_counts[t.target_id] = target_counts.get(t.target_id, 0) + 1

        if target_counts:
            primary_target_id = max(target_counts, key=target_counts.get)  # type: ignore[arg-type]
        else:
            primary_target_id = -1

        timestamps = np.array(
            [f.timestamp for f in segment], dtype=np.float64
        )
        crosshair_positions = np.array(
            [[f.crosshair_x, f.crosshair_y] for f in segment],
            dtype=np.float64,
        )

        # Build target positions — prefer the primary target, forward-fill gaps
        target_positions = np.empty((len(segment), 2), dtype=np.float64)
        last_known_x: float = np.nan
        last_known_y: float = np.nan

        for i, frame in enumerate(segment):
            found = False
            for t in frame.targets:
                if t.target_id == primary_target_id:
                    target_positions[i] = [t.x, t.y]
                    last_known_x, last_known_y = t.x, t.y
                    found = True
                    break

            if not found:
                # Fall back to any visible target, or forward-fill
                if frame.targets:
                    best = frame.targets[0]
                    target_positions[i] = [best.x, best.y]
                    last_known_x, last_known_y = best.x, best.y
                else:
                    target_positions[i] = [last_known_x, last_known_y]

        return EngagementWindow(
            start_frame=segment[0].frame_index,
            end_frame=segment[-1].frame_index,
            start_time=segment[0].timestamp,
            end_time=segment[-1].timestamp,
            target_id=primary_target_id,
            crosshair_positions=crosshair_positions,
            target_positions=target_positions,
            timestamps=timestamps,
        )

    def __repr__(self) -> str:
        return (
            f"PositionTracker(frames={self.frame_count})"
        )
