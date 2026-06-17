"""
FPS Aim Performance Analyzer - Overshoot Analysis Metric

Detects overshoot events where the crosshair passes beyond the target
and reverses direction. Computes overshoot ratio, magnitude, and
directional bias to characterize aim correction behavior.

Overshoot indicates the player moved the crosshair past the target
and had to correct back. Frequent overshooting suggests issues with
mouse sensitivity or motor control precision.

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


class OvershootAnalyzer:
    """Analyzes overshoot behavior in aim movements.

    Detects when the crosshair passes beyond the target position and
    reverses direction, indicating an overshoot correction. Computes
    statistics including ratio, magnitude, and directional bias.

    Attributes:
        min_movement: Minimum movement in pixels to qualify as an aim event.
        reversal_threshold: Minimum reversal magnitude to count as overshoot.
    """

    def __init__(
        self,
        min_movement: Optional[float] = None,
        reversal_threshold: Optional[float] = None
    ) -> None:
        """Initialize OvershootAnalyzer.

        Args:
            min_movement: Minimum pixels of movement to qualify as aim event.
                Defaults to settings.OVERSHOOT_MIN_MOVEMENT.
            reversal_threshold: Minimum reversal in pixels to count as
                overshoot. Defaults to settings.OVERSHOOT_REVERSAL_THRESHOLD.
        """
        self.min_movement: float = (
            min_movement if min_movement is not None
            else settings.OVERSHOOT_MIN_MOVEMENT
        )
        self.reversal_threshold: float = (
            reversal_threshold if reversal_threshold is not None
            else settings.OVERSHOOT_REVERSAL_THRESHOLD
        )

    def compute(
        self,
        movement_segments: List[Dict],
        target_positions: Union[List[Tuple[float, float]], np.ndarray]
    ) -> Dict[str, Union[float, int, List[Dict]]]:
        """Compute overshoot metrics from movement segments and targets.

        Analyzes each movement segment to detect if the crosshair overshot
        the target (passed beyond it and reversed). Computes aggregate
        statistics and per-event details.

        Args:
            movement_segments: List of movement segment dictionaries, each
                containing:
                - positions: List of (x, y) crosshair positions during the
                    movement.
                - start_frame (optional): Starting frame index.
                - end_frame (optional): Ending frame index.
            target_positions: Corresponding target positions. Can be:
                - A single (x, y) tuple applied to all segments.
                - A list of (x, y) tuples, one per segment.

        Returns:
            Dictionary with keys:
                - overshoot_ratio (float): Fraction of movements with
                    overshoot [0, 1].
                - overshoot_count (int): Number of overshoot events.
                - total_movements (int): Total movement segments analyzed.
                - mean_overshoot_magnitude (float): Mean overshoot distance
                    in pixels.
                - directional_bias (float): Bias toward overshoot vs
                    undershoot [-1, 1]. Positive = more overshoots.
                - per_event_details (List[Dict]): Per-event analysis with
                    is_overshoot, magnitude, direction, etc.
        """
        target_arr = np.asarray(target_positions, dtype=np.float64)
        is_single_target = target_arr.ndim == 1

        overshoot_count = 0
        undershoot_count = 0
        magnitudes: List[float] = []
        per_event_details: List[Dict] = []
        total_movements = 0

        for idx, segment in enumerate(movement_segments):
            try:
                positions = np.asarray(
                    segment.get("positions", []), dtype=np.float64
                )

                if positions.shape[0] < 3:
                    continue  # Need at least 3 points to detect reversal

                # Determine target for this segment
                if is_single_target:
                    target = target_arr
                else:
                    if idx < len(target_arr):
                        target = target_arr[idx]
                    else:
                        continue

                # Check minimum movement threshold
                total_displacement = np.sqrt(
                    np.sum((positions[-1] - positions[0]) ** 2)
                )
                if total_displacement < self.min_movement:
                    continue

                total_movements += 1

                # Analyze overshoot for this segment
                event_detail = self._analyze_segment(
                    positions, target, idx
                )

                per_event_details.append(event_detail)

                if event_detail["is_overshoot"]:
                    overshoot_count += 1
                    magnitudes.append(event_detail["magnitude"])
                elif event_detail["is_undershoot"]:
                    undershoot_count += 1

            except Exception as e:
                logger.error(
                    "Error analyzing overshoot for segment %d: %s",
                    idx, str(e)
                )

        # Compute aggregate metrics
        overshoot_ratio = (
            overshoot_count / total_movements
            if total_movements > 0 else 0.0
        )

        mean_overshoot_magnitude = (
            float(np.mean(magnitudes)) if magnitudes else 0.0
        )

        # Directional bias: positive = more overshoots, negative = more
        # undershoots
        total_classified = overshoot_count + undershoot_count
        if total_classified > 0:
            directional_bias = float(
                (overshoot_count - undershoot_count) / total_classified
            )
        else:
            directional_bias = 0.0

        result = {
            "overshoot_ratio": float(overshoot_ratio),
            "overshoot_count": overshoot_count,
            "total_movements": total_movements,
            "mean_overshoot_magnitude": mean_overshoot_magnitude,
            "directional_bias": directional_bias,
            "per_event_details": per_event_details,
        }

        logger.info(
            "Overshoot analysis: ratio=%.2f (%d/%d), "
            "mean magnitude=%.1f px, directional bias=%.2f",
            overshoot_ratio, overshoot_count, total_movements,
            mean_overshoot_magnitude, directional_bias
        )

        return result

    def _analyze_segment(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        segment_index: int
    ) -> Dict[str, Union[bool, float, int]]:
        """Analyze a single movement segment for overshoot.

        Detects reversal points where the crosshair passes through the
        target and reverses direction. An overshoot occurs when the
        crosshair's signed distance from the target changes sign,
        meaning it crossed past the target.

        Args:
            positions: Array of crosshair positions, shape (N, 2).
            target: Target position as (x, y).
            segment_index: Index of this segment for identification.

        Returns:
            Dictionary with per-event detail:
                - segment_index (int): Index of the segment.
                - is_overshoot (bool): True if overshoot detected.
                - is_undershoot (bool): True if undershoot detected (stopped
                    short).
                - magnitude (float): Peak overshoot distance in pixels.
                - reversal_frame (int): Frame index within segment where
                    reversal was detected.
                - direction (str): 'overshoot', 'undershoot', or 'direct'.
        """
        # Compute vector from each position to the target
        deltas = positions - target  # shape (N, 2)

        # Distances from each position to target
        distances = np.sqrt(np.sum(deltas ** 2, axis=1))

        # Find the index of closest approach
        min_dist_idx = int(np.argmin(distances))

        # Project movement onto the direction from start to target
        # to determine overshoot vs undershoot
        start_to_target = target - positions[0]
        movement_direction_magnitude = np.sqrt(
            np.sum(start_to_target ** 2)
        )

        if movement_direction_magnitude < 1e-6:
            return {
                "segment_index": segment_index,
                "is_overshoot": False,
                "is_undershoot": False,
                "magnitude": 0.0,
                "reversal_frame": -1,
                "direction": "direct",
            }

        # Unit vector from start to target
        direction_unit = start_to_target / movement_direction_magnitude

        # Project each position onto the movement axis (signed distance)
        positions_relative = positions - positions[0]
        projections = np.dot(positions_relative, direction_unit)

        # Target projection (should be ~movement_direction_magnitude)
        target_projection = movement_direction_magnitude

        # Signed overshoot: projection - target_projection
        signed_overshoot = projections - target_projection

        # Find peak signed overshoot (max projection beyond target)
        peak_overshoot_idx = int(np.argmax(projections))
        peak_projection = float(projections[peak_overshoot_idx])

        # Check for reversal: does the crosshair pass the target?
        is_overshoot = False
        is_undershoot = False
        magnitude = 0.0
        reversal_frame = -1

        # Detect reversal by checking sign changes in the velocity
        # along the movement axis
        if peak_projection > target_projection + self.reversal_threshold:
            # Crosshair went past the target
            # Check if it then reversed (subsequent projections decrease)
            post_peak = projections[peak_overshoot_idx:]
            if len(post_peak) > 1 and post_peak[-1] < peak_projection:
                is_overshoot = True
                magnitude = float(peak_projection - target_projection)
                reversal_frame = peak_overshoot_idx
        else:
            # Check if crosshair stopped short (undershoot)
            final_projection = float(projections[-1])
            if final_projection < target_projection - self.reversal_threshold:
                is_undershoot = True
                magnitude = float(target_projection - final_projection)

        if is_overshoot:
            direction_label = "overshoot"
        elif is_undershoot:
            direction_label = "undershoot"
        else:
            direction_label = "direct"

        return {
            "segment_index": segment_index,
            "is_overshoot": is_overshoot,
            "is_undershoot": is_undershoot,
            "magnitude": magnitude,
            "reversal_frame": reversal_frame,
            "direction": direction_label,
        }
