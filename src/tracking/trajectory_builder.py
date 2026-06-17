"""
Trajectory Builder Module - Builds smooth trajectories from raw position data.

This module provides the TrajectoryBuilder class which takes raw per-frame
position data (timestamp, x, y), applies smoothing via Savitzky-Golay
filtering, computes full kinematic profiles (velocity, acceleration, jerk),
and segments the trajectory into discrete aim movements (idle, moving,
flick candidate).

The smoothing and differentiation pipeline is:
    1. Interpolate any temporal gaps (missing frames) via linear interpolation.
    2. Smooth positions using ``scipy.signal.savgol_filter``.
    3. Compute velocity, acceleration, and jerk via ``np.gradient``.

Movement segmentation classifies each frame as *idle* (speed below threshold)
or *moving*, then merges adjacent frames of the same type into contiguous
segments.  Very short segments are absorbed into their neighbours.

Author: FPS Aim Performance Analyzer
"""

import sys
import logging
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import savgol_filter

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
class Trajectory:
    """Full kinematic trajectory computed from position data.

    All arrays share the same first-axis length *N* (number of time samples).

    Attributes:
        timestamps: 1-D array of shape ``(N,)`` — time in seconds.
        positions: Smoothed positions, shape ``(N, 2)`` — ``[x, y]``.
        raw_positions: Original (un-smoothed) positions, shape ``(N, 2)``.
        velocities: Velocity vectors, shape ``(N, 2)`` — ``[vx, vy]``
            in pixels/second.
        speeds: Scalar speed (magnitude of velocity), shape ``(N,)``.
        accelerations: Acceleration vectors, shape ``(N, 2)`` — ``[ax, ay]``
            in pixels/second².
        acceleration_magnitudes: Scalar acceleration magnitude, shape ``(N,)``.
        jerks: Jerk vectors, shape ``(N, 2)`` in pixels/second³.
        jerk_magnitudes: Scalar jerk magnitude, shape ``(N,)``.
    """

    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    raw_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    speeds: np.ndarray = field(default_factory=lambda: np.empty(0))
    accelerations: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    acceleration_magnitudes: np.ndarray = field(default_factory=lambda: np.empty(0))
    jerks: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    jerk_magnitudes: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def n_samples(self) -> int:
        """Number of time samples in the trajectory."""
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self.n_samples < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def peak_speed(self) -> float:
        """Maximum scalar speed observed."""
        if self.n_samples == 0:
            return 0.0
        return float(np.nanmax(self.speeds))

    @property
    def mean_speed(self) -> float:
        """Average scalar speed."""
        if self.n_samples == 0:
            return 0.0
        return float(np.nanmean(self.speeds))


@dataclass
class MovementSegment:
    """A contiguous segment of the trajectory classified by movement type.

    Attributes:
        start_idx: Start index into the parent trajectory arrays.
        end_idx: End index (inclusive) into the parent trajectory arrays.
        start_time: Timestamp (seconds) at the segment start.
        end_time: Timestamp (seconds) at the segment end.
        duration: Duration of the segment in seconds.
        trajectory_slice: Sub-trajectory covering only this segment.
        movement_type: Classification — one of ``'idle'``, ``'moving'``,
            or ``'flick_candidate'``.
        peak_speed: Maximum speed within the segment (pixels/second).
        mean_speed: Mean speed within the segment (pixels/second).
        distance: Euclidean displacement from start to end position (pixels).
        path_length: Total path length (sum of inter-frame displacements).
    """

    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration: float
    trajectory_slice: Trajectory = field(default_factory=Trajectory)
    movement_type: str = "idle"
    peak_speed: float = 0.0
    mean_speed: float = 0.0
    distance: float = 0.0
    path_length: float = 0.0


# ===========================================================================
# TrajectoryBuilder
# ===========================================================================

class TrajectoryBuilder:
    """Builds smooth kinematic trajectories and segments aim movements.

    The builder is configured once and can process multiple position arrays.
    All heavy computation is vectorised with NumPy; only the Savitzky-Golay
    filter introduces a SciPy dependency.

    Args:
        fps: Capture framerate, used to convert frame-based differentiation
            into real-time units (pixels/second etc.).
        smoothing_window: Window length for the Savitzky-Golay filter.
            Must be a positive odd integer.
        poly_order: Polynomial order for the Savitzky-Golay filter.
            Must satisfy ``poly_order < smoothing_window``.

    Example::

        builder = TrajectoryBuilder(fps=60.0)
        traj = builder.build_trajectory(positions_nx3)
        segments = builder.segment_movements(traj)
    """

    def __init__(
        self,
        fps: float = 60.0,
        smoothing_window: int = 11,
        poly_order: int = 3,
    ) -> None:
        if smoothing_window % 2 == 0:
            smoothing_window += 1
            logger.warning(
                "smoothing_window must be odd; adjusted to %d.",
                smoothing_window,
            )
        if poly_order >= smoothing_window:
            poly_order = smoothing_window - 1
            logger.warning(
                "poly_order must be < smoothing_window; adjusted to %d.",
                poly_order,
            )

        self.fps: float = fps
        self.smoothing_window: int = smoothing_window
        self.poly_order: int = poly_order

        # Settings-driven defaults (allow runtime override via settings)
        self._idle_threshold: float = float(
            getattr(settings, "MOVEMENT_IDLE_THRESHOLD", 3)
        )
        self._min_duration_frames: int = int(
            getattr(settings, "MOVEMENT_MIN_DURATION_FRAMES", 3)
        )

        logger.debug(
            "TrajectoryBuilder initialised: fps=%.1f, window=%d, order=%d, "
            "idle_threshold=%.1f, min_duration=%d frames.",
            self.fps,
            self.smoothing_window,
            self.poly_order,
            self._idle_threshold,
            self._min_duration_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_trajectory(self, positions: np.ndarray) -> Trajectory:
        """Build a full kinematic trajectory from raw position data.

        Args:
            positions: Array of shape ``(N, 3)`` where each row is
                ``[timestamp, x, y]``.

        Returns:
            A :class:`Trajectory` with smoothed positions and all kinematic
            derivatives computed.

        Raises:
            ValueError: If *positions* has fewer than 2 rows or an
                unexpected number of columns.
        """

        positions = np.asarray(positions, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"Expected positions of shape (N, 3), got {positions.shape}."
            )

        if positions.shape[0] < 2:
            raise ValueError(
                "Need at least 2 position samples to build a trajectory; "
                f"got {positions.shape[0]}."
            )

        timestamps = positions[:, 0].copy()
        raw_xy = positions[:, 1:3].copy()

        # 1. Fill temporal gaps
        timestamps_filled, xy_filled = self._interpolate_gaps(
            raw_xy, timestamps
        )

        # 2. Smooth positions
        smoothed_xy = self._smooth_positions(xy_filled)

        # 3. Compute kinematics via np.gradient
        dt = np.gradient(timestamps_filled)
        # Guard against zero dt (duplicate timestamps)
        dt = np.where(dt == 0, 1.0 / self.fps, dt)

        vx = np.gradient(smoothed_xy[:, 0], timestamps_filled)
        vy = np.gradient(smoothed_xy[:, 1], timestamps_filled)
        velocities = np.column_stack([vx, vy])
        speeds = np.sqrt(vx ** 2 + vy ** 2)

        ax = np.gradient(vx, timestamps_filled)
        ay = np.gradient(vy, timestamps_filled)
        accelerations = np.column_stack([ax, ay])
        acceleration_magnitudes = np.sqrt(ax ** 2 + ay ** 2)

        jx = np.gradient(ax, timestamps_filled)
        jy = np.gradient(ay, timestamps_filled)
        jerks = np.column_stack([jx, jy])
        jerk_magnitudes = np.sqrt(jx ** 2 + jy ** 2)

        # Build raw_positions to match interpolated length (pad original)
        raw_positions = xy_filled.copy()

        trajectory = Trajectory(
            timestamps=timestamps_filled,
            positions=smoothed_xy,
            raw_positions=raw_positions,
            velocities=velocities,
            speeds=speeds,
            accelerations=accelerations,
            acceleration_magnitudes=acceleration_magnitudes,
            jerks=jerks,
            jerk_magnitudes=jerk_magnitudes,
        )

        logger.info(
            "Built trajectory: %d samples, duration=%.3fs, peak_speed=%.1f px/s.",
            trajectory.n_samples,
            trajectory.duration,
            trajectory.peak_speed,
        )
        return trajectory

    def segment_movements(
        self, trajectory: Trajectory
    ) -> List[MovementSegment]:
        """Split a trajectory into discrete aim-movement segments.

        Each frame is first classified as *idle* (speed below threshold,
        expressed in pixels/second) or *moving*.  Adjacent frames of the
        same type are merged.  Segments shorter than
        :pydata:`settings.MOVEMENT_MIN_DURATION_FRAMES` are absorbed into
        the preceding segment.

        Moving segments whose peak-to-mean speed ratio exceeds
        :pydata:`settings.FLICK_VELOCITY_RATIO_THRESHOLD` are promoted to
        ``'flick_candidate'``.

        Args:
            trajectory: A :class:`Trajectory` produced by
                :meth:`build_trajectory`.

        Returns:
            List of :class:`MovementSegment` instances in chronological order.
        """

        if trajectory.n_samples < 2:
            logger.warning(
                "Trajectory has fewer than 2 samples; cannot segment."
            )
            return []

        # Convert idle threshold from pixels/frame to pixels/second
        idle_threshold_ps = self._idle_threshold * self.fps

        # Classify each frame
        is_moving = trajectory.speeds > idle_threshold_ps

        # Find contiguous runs
        raw_segments = self._runs_of(is_moving)

        # Merge short segments into neighbours
        merged = self._merge_short_segments(
            raw_segments, min_length=self._min_duration_frames
        )

        # Build MovementSegment objects
        flick_ratio_thresh = float(
            getattr(settings, "FLICK_VELOCITY_RATIO_THRESHOLD", 4.0)
        )

        segments: List[MovementSegment] = []
        for start, end, moving_flag in merged:
            sub_ts = trajectory.timestamps[start : end + 1]
            sub_pos = trajectory.positions[start : end + 1]
            sub_raw = trajectory.raw_positions[start : end + 1]
            sub_vel = trajectory.velocities[start : end + 1]
            sub_spd = trajectory.speeds[start : end + 1]
            sub_acc = trajectory.accelerations[start : end + 1]
            sub_amg = trajectory.acceleration_magnitudes[start : end + 1]
            sub_jrk = trajectory.jerks[start : end + 1]
            sub_jmg = trajectory.jerk_magnitudes[start : end + 1]

            traj_slice = Trajectory(
                timestamps=sub_ts,
                positions=sub_pos,
                raw_positions=sub_raw,
                velocities=sub_vel,
                speeds=sub_spd,
                accelerations=sub_acc,
                acceleration_magnitudes=sub_amg,
                jerks=sub_jrk,
                jerk_magnitudes=sub_jmg,
            )

            pk_speed = float(np.max(sub_spd)) if len(sub_spd) > 0 else 0.0
            mn_speed = float(np.mean(sub_spd)) if len(sub_spd) > 0 else 0.0

            # Determine movement type
            if not moving_flag:
                movement_type = "idle"
            else:
                # Check for flick candidate
                if mn_speed > 0 and (pk_speed / mn_speed) >= flick_ratio_thresh:
                    movement_type = "flick_candidate"
                else:
                    movement_type = "moving"

            # Displacement (start→end straight-line)
            displacement = float(
                np.sqrt(
                    (sub_pos[-1, 0] - sub_pos[0, 0]) ** 2
                    + (sub_pos[-1, 1] - sub_pos[0, 1]) ** 2
                )
            )

            # Path length (sum of frame-to-frame distances)
            if len(sub_pos) > 1:
                diffs = np.diff(sub_pos, axis=0)
                path_len = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
            else:
                path_len = 0.0

            seg = MovementSegment(
                start_idx=start,
                end_idx=end,
                start_time=float(sub_ts[0]),
                end_time=float(sub_ts[-1]),
                duration=float(sub_ts[-1] - sub_ts[0]),
                trajectory_slice=traj_slice,
                movement_type=movement_type,
                peak_speed=pk_speed,
                mean_speed=mn_speed,
                distance=displacement,
                path_length=path_len,
            )
            segments.append(seg)

        logger.info(
            "Segmented trajectory into %d segments: %d idle, %d moving, "
            "%d flick candidates.",
            len(segments),
            sum(1 for s in segments if s.movement_type == "idle"),
            sum(1 for s in segments if s.movement_type == "moving"),
            sum(1 for s in segments if s.movement_type == "flick_candidate"),
        )
        return segments

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _smooth_positions(self, positions: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to position data.

        If the position array is shorter than ``smoothing_window``, the
        window is automatically reduced to the largest valid odd value.

        Args:
            positions: Array of shape ``(N, 2)`` with ``[x, y]`` per sample.

        Returns:
            Smoothed array of the same shape.
        """

        n = positions.shape[0]
        window = self.smoothing_window
        order = self.poly_order

        # Adapt window if data is too short
        if n < window:
            window = n if n % 2 == 1 else max(n - 1, 1)
            order = min(order, window - 1)

        if window < 3 or order < 1:
            # Not enough data to smooth; return copy
            logger.debug(
                "Skipping Savitzky-Golay: only %d samples available.", n
            )
            return positions.copy()

        try:
            smoothed = np.empty_like(positions)
            smoothed[:, 0] = savgol_filter(
                positions[:, 0], window_length=window, polyorder=order
            )
            smoothed[:, 1] = savgol_filter(
                positions[:, 1], window_length=window, polyorder=order
            )
            return smoothed
        except Exception as exc:
            logger.warning(
                "Savitzky-Golay smoothing failed (%s); returning raw data.", exc
            )
            return positions.copy()

    def _interpolate_gaps(
        self, positions: np.ndarray, timestamps: np.ndarray
    ) -> tuple:
        """Fill temporal gaps in the position data via linear interpolation.

        If the time difference between consecutive samples exceeds
        ``1.5 / fps`` (i.e. more than 1.5 frame periods), intermediate
        samples are inserted at uniform intervals of ``1 / fps``.

        Args:
            positions: Array of shape ``(N, 2)`` — ``[x, y]``.
            timestamps: 1-D array of length N — seconds.

        Returns:
            Tuple of ``(new_timestamps, new_positions)`` where gaps have
            been linearly interpolated.
        """

        if len(timestamps) < 2:
            return timestamps.copy(), positions.copy()

        dt_expected = 1.0 / self.fps
        gap_threshold = 1.5 * dt_expected

        # Check whether any gaps exist to avoid unnecessary work
        dts = np.diff(timestamps)
        if np.all(dts <= gap_threshold):
            return timestamps.copy(), positions.copy()

        new_ts: List[float] = []
        new_x: List[float] = []
        new_y: List[float] = []

        for i in range(len(timestamps) - 1):
            new_ts.append(timestamps[i])
            new_x.append(positions[i, 0])
            new_y.append(positions[i, 1])

            gap = timestamps[i + 1] - timestamps[i]
            if gap > gap_threshold:
                n_fill = int(round(gap / dt_expected)) - 1
                if n_fill > 0:
                    fill_ts = np.linspace(
                        timestamps[i], timestamps[i + 1], n_fill + 2
                    )[1:-1]
                    fill_x = np.linspace(
                        positions[i, 0], positions[i + 1, 0], n_fill + 2
                    )[1:-1]
                    fill_y = np.linspace(
                        positions[i, 1], positions[i + 1, 1], n_fill + 2
                    )[1:-1]
                    new_ts.extend(fill_ts.tolist())
                    new_x.extend(fill_x.tolist())
                    new_y.extend(fill_y.tolist())

        # Append last sample
        new_ts.append(timestamps[-1])
        new_x.append(positions[-1, 0])
        new_y.append(positions[-1, 1])

        out_ts = np.array(new_ts, dtype=np.float64)
        out_pos = np.column_stack(
            [np.array(new_x, dtype=np.float64), np.array(new_y, dtype=np.float64)]
        )

        if len(out_ts) > len(timestamps):
            logger.debug(
                "Interpolated %d missing samples across gaps.",
                len(out_ts) - len(timestamps),
            )

        return out_ts, out_pos

    def _detect_movement_onset(
        self, velocity: np.ndarray
    ) -> List[tuple]:
        """Find movement start and end indices from a velocity profile.

        A movement onset is detected when the scalar speed exceeds the idle
        threshold, and a movement offset when it drops below.

        Args:
            velocity: Array of shape ``(N, 2)`` — ``[vx, vy]``.

        Returns:
            List of ``(start_index, end_index)`` tuples for each detected
            movement bout.
        """

        speeds = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
        idle_threshold_ps = self._idle_threshold * self.fps
        is_moving = speeds > idle_threshold_ps

        movements: List[tuple] = []
        in_movement = False
        start = 0

        for i, moving in enumerate(is_moving):
            if moving and not in_movement:
                start = i
                in_movement = True
            elif not moving and in_movement:
                if (i - start) >= self._min_duration_frames:
                    movements.append((start, i - 1))
                in_movement = False

        # Handle movement that extends to the end
        if in_movement:
            end = len(is_moving) - 1
            if (end - start + 1) >= self._min_duration_frames:
                movements.append((start, end))

        return movements

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _runs_of(mask: np.ndarray) -> List[tuple]:
        """Find contiguous runs of True/False in a boolean array.

        Args:
            mask: 1-D boolean array.

        Returns:
            List of ``(start_index, end_index_inclusive, is_true)`` tuples.
        """

        if len(mask) == 0:
            return []

        runs: List[tuple] = []
        current_val = bool(mask[0])
        start = 0

        for i in range(1, len(mask)):
            if bool(mask[i]) != current_val:
                runs.append((start, i - 1, current_val))
                current_val = bool(mask[i])
                start = i

        runs.append((start, len(mask) - 1, current_val))
        return runs

    def _merge_short_segments(
        self,
        runs: List[tuple],
        min_length: int,
    ) -> List[tuple]:
        """Merge segments shorter than *min_length* into their neighbours.

        Short segments are absorbed into the preceding segment. If the
        short segment is the very first one, it is absorbed into the
        following segment.

        Args:
            runs: Output of :meth:`_runs_of`.
            min_length: Minimum number of frames for a segment to survive.

        Returns:
            Merged list in the same ``(start, end, flag)`` format.
        """

        if not runs:
            return []

        merged: List[list] = [[s, e, f] for s, e, f in runs]

        changed = True
        while changed:
            changed = False
            new_merged: List[list] = []
            i = 0
            while i < len(merged):
                start, end, flag = merged[i]
                length = end - start + 1
                if length < min_length and len(merged) > 1:
                    # Absorb into previous or next segment
                    if new_merged:
                        new_merged[-1][1] = end
                        changed = True
                    elif i + 1 < len(merged):
                        merged[i + 1][0] = start
                        changed = True
                    else:
                        new_merged.append([start, end, flag])
                else:
                    new_merged.append([start, end, flag])
                i += 1
            merged = new_merged

        return [(s, e, f) for s, e, f in merged]

    def __repr__(self) -> str:
        return (
            f"TrajectoryBuilder(fps={self.fps}, "
            f"window={self.smoothing_window}, "
            f"order={self.poly_order})"
        )
