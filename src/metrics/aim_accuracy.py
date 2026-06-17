"""
FPS Aim Performance Analyzer - Aim Accuracy Metric

Computes aim accuracy by measuring how close the crosshair is to the target
using Euclidean distance. Provides per-frame and per-engagement analysis,
including path efficiency (direct vs actual path distance).

Author: FPS Aim Performance Analyzer
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class AimAccuracy:
    """Analyzes aim accuracy by comparing crosshair and target positions.

    Measures how often and how closely the player's crosshair aligns with
    detected targets. Uses Euclidean distance with configurable thresholds
    to determine on-target frames.

    Attributes:
        default_threshold: Default pixel distance threshold for on-target
            classification.
    """

    def __init__(self, default_threshold: Optional[float] = None) -> None:
        """Initialize AimAccuracy analyzer.

        Args:
            default_threshold: Default distance threshold in pixels.
                Falls back to settings.ON_TARGET_THRESHOLD if not specified.
        """
        self.default_threshold: float = (
            default_threshold if default_threshold is not None
            else settings.ON_TARGET_THRESHOLD
        )

    @staticmethod
    def _euclidean_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """Compute Euclidean distance between two 2D points.

        Args:
            p1: First point as (x, y).
            p2: Second point as (x, y).

        Returns:
            Euclidean distance as a float.
        """
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    @staticmethod
    def _path_length(positions: np.ndarray) -> float:
        """Compute total path length from a sequence of positions.

        Args:
            positions: Array of shape (N, 2) representing sequential positions.

        Returns:
            Total Euclidean path length.
        """
        if len(positions) < 2:
            return 0.0
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.sum(segment_lengths))

    def compute(
        self,
        crosshair_positions: Union[List[Tuple[float, float]], np.ndarray],
        target_positions: Union[List[Tuple[float, float]], np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[str, Union[float, int]]:
        """Compute overall aim accuracy metrics.

        Analyzes how accurately the crosshair tracks the target across all
        provided frames. A frame is considered "on target" if the Euclidean
        distance between crosshair and target is within the threshold.

        Args:
            crosshair_positions: List/array of (x, y) crosshair positions,
                one per frame.
            target_positions: List/array of (x, y) target positions, one
                per frame. Must have same length as crosshair_positions.
            threshold: Distance threshold in pixels to count as on-target.
                Uses default_threshold if not specified.

        Returns:
            Dictionary with keys:
                - accuracy_rate (float): Fraction of frames on target [0, 1].
                - frames_on_target (int): Count of on-target frames.
                - total_frames (int): Total number of analyzed frames.
                - mean_distance (float): Mean Euclidean distance across frames.
                - path_efficiency (float): Ratio of direct distance to actual
                    path distance [0, 1]. Higher = more efficient movement.

        Raises:
            ValueError: If input arrays have different lengths or are empty.
        """
        threshold = threshold if threshold is not None else self.default_threshold

        crosshair_arr = np.asarray(crosshair_positions, dtype=np.float64)
        target_arr = np.asarray(target_positions, dtype=np.float64)

        if crosshair_arr.shape[0] == 0 or target_arr.shape[0] == 0:
            logger.warning("Empty position arrays provided to aim accuracy compute.")
            return {
                "accuracy_rate": 0.0,
                "frames_on_target": 0,
                "total_frames": 0,
                "mean_distance": 0.0,
                "path_efficiency": 0.0,
            }

        if crosshair_arr.shape[0] != target_arr.shape[0]:
            raise ValueError(
                f"Crosshair positions ({crosshair_arr.shape[0]}) and target "
                f"positions ({target_arr.shape[0]}) must have the same length."
            )

        # Compute per-frame distances
        distances = np.sqrt(
            np.sum((crosshair_arr - target_arr) ** 2, axis=1)
        )

        total_frames = len(distances)
        frames_on_target = int(np.sum(distances <= threshold))
        accuracy_rate = frames_on_target / total_frames if total_frames > 0 else 0.0
        mean_distance = float(np.mean(distances))

        # Path efficiency: direct_distance / actual_path_distance
        path_efficiency = self._compute_path_efficiency(crosshair_arr, target_arr)

        result = {
            "accuracy_rate": float(accuracy_rate),
            "frames_on_target": frames_on_target,
            "total_frames": total_frames,
            "mean_distance": mean_distance,
            "path_efficiency": path_efficiency,
        }

        logger.info(
            "Aim accuracy computed: %.1f%% on target (%d/%d frames), "
            "mean distance: %.1f px, path efficiency: %.3f",
            accuracy_rate * 100, frames_on_target, total_frames,
            mean_distance, path_efficiency
        )

        return result

    def _compute_path_efficiency(
        self,
        crosshair_arr: np.ndarray,
        target_arr: np.ndarray
    ) -> float:
        """Compute path efficiency as direct_distance / actual_path_distance.

        Path efficiency measures how directly the crosshair moved toward
        the target. A value of 1.0 means perfectly direct movement.

        Args:
            crosshair_arr: Array of crosshair positions, shape (N, 2).
            target_arr: Array of target positions, shape (N, 2).

        Returns:
            Path efficiency ratio in [0, 1]. Returns 1.0 if no movement.
        """
        if len(crosshair_arr) < 2:
            return 1.0

        # Direct distance from first crosshair position to last target position
        direct_distance = self._euclidean_distance(
            tuple(crosshair_arr[0]),
            tuple(target_arr[-1])
        )

        # Actual path distance traversed by the crosshair
        actual_path_distance = self._path_length(crosshair_arr)

        if actual_path_distance < 1e-6:
            return 1.0  # No movement means perfectly efficient (no need to move)

        if direct_distance < 1e-6:
            # Crosshair started at target — efficiency depends on staying still
            return 1.0 if actual_path_distance < 1e-6 else 0.0

        efficiency = direct_distance / actual_path_distance
        return float(min(efficiency, 1.0))

    def compute_per_engagement(
        self,
        engagement_windows: List[Dict],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Union[float, int]]]:
        """Compute accuracy metrics for each engagement window separately.

        An engagement window represents a period where a target was visible
        and the player was actively aiming at it.

        Args:
            engagement_windows: List of engagement dictionaries, each
                containing:
                - crosshair_positions: List of (x, y) crosshair positions.
                - target_positions: List of (x, y) target positions.
                - start_frame (optional): Frame index when engagement started.
                - end_frame (optional): Frame index when engagement ended.
            threshold: Distance threshold in pixels. Uses default if not set.

        Returns:
            List of accuracy metric dictionaries, one per engagement.
            Each dictionary contains the same keys as compute() plus:
                - engagement_index (int): Index of the engagement.
                - start_frame (int): Start frame if available.
                - end_frame (int): End frame if available.
        """
        threshold = threshold if threshold is not None else self.default_threshold
        results: List[Dict[str, Union[float, int]]] = []

        for idx, engagement in enumerate(engagement_windows):
            try:
                crosshair_pos = engagement.get("crosshair_positions", [])
                target_pos = engagement.get("target_positions", [])

                if not crosshair_pos or not target_pos:
                    logger.warning(
                        "Engagement %d has empty positions, skipping.", idx
                    )
                    continue

                metrics = self.compute(crosshair_pos, target_pos, threshold)
                metrics["engagement_index"] = idx
                metrics["start_frame"] = engagement.get("start_frame", -1)
                metrics["end_frame"] = engagement.get("end_frame", -1)

                results.append(metrics)

            except Exception as e:
                logger.error(
                    "Error computing accuracy for engagement %d: %s",
                    idx, str(e)
                )

        logger.info(
            "Per-engagement accuracy computed for %d/%d engagements.",
            len(results), len(engagement_windows)
        )

        return results
