"""
FPS Aim Performance Analyzer - Aim Movement Classifier

Classifies aim movements into categories based on kinematic features:
- FLICK: Fast, ballistic movements with high velocity ratio and short
    duration. Typically used for sudden target acquisition.
- TRACKING: Sustained, smooth movements that follow a moving target
    with high crosshair-target velocity correlation.
- HYBRID: Movements that don't clearly fit either category, often
    combining elements of both flick and tracking.

Classification Rules:
    FLICK if velocity_ratio > 4.0 AND duration < 300ms
    TRACKING if correlation > 0.6 AND duration > 500ms
    else HYBRID

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


class AimClassifier:
    """Classifies aim movement segments into FLICK, TRACKING, or HYBRID.

    Uses kinematic features (velocity ratio, acceleration spike factor,
    duration, correlation with target) to categorize each aim movement.

    Attributes:
        fps: Frames per second for time calculations.
        flick_max_duration_ms: Maximum duration for a flick classification.
        tracking_min_duration_ms: Minimum duration for tracking classification.
        flick_velocity_ratio: Peak/mean velocity ratio threshold for flick.
        tracking_correlation: Min crosshair-target correlation for tracking.
    """

    # Movement type constants
    FLICK = "FLICK"
    TRACKING = "TRACKING"
    HYBRID = "HYBRID"

    def __init__(self, fps: Optional[float] = None) -> None:
        """Initialize AimClassifier.

        Args:
            fps: Frames per second. Defaults to settings.CAPTURE_FPS.
        """
        self.fps: float = fps if fps is not None else settings.CAPTURE_FPS
        self.flick_max_duration_ms: float = settings.FLICK_MAX_DURATION_MS
        self.tracking_min_duration_ms: float = settings.TRACKING_MIN_DURATION_MS
        self.flick_velocity_ratio: float = settings.FLICK_VELOCITY_RATIO_THRESHOLD
        self.tracking_correlation: float = settings.TRACKING_CORRELATION_THRESHOLD

    def classify(
        self,
        movement_segment: Dict,
        target_velocity: Optional[
            Union[List[Tuple[float, float]], np.ndarray]
        ] = None
    ) -> str:
        """Classify a single movement segment.

        Extracts kinematic features from the movement and applies
        rule-based classification to determine movement type.

        Args:
            movement_segment: Dictionary containing:
                - positions: List of (x, y) crosshair positions.
                - target_positions (optional): List of (x, y) target
                    positions for correlation analysis.
            target_velocity: Optional array of target velocity vectors
                for crosshair-target correlation computation.

        Returns:
            Classification string: 'FLICK', 'TRACKING', or 'HYBRID'.
        """
        features = self.extract_features(movement_segment, target_velocity)
        return self._apply_rules(features)

    def classify_batch(
        self,
        segments: List[Dict],
        target_velocities: Optional[List[
            Union[List[Tuple[float, float]], np.ndarray, None]
        ]] = None
    ) -> List[Dict[str, Union[str, Dict]]]:
        """Classify all movement segments in a batch.

        Args:
            segments: List of movement segment dictionaries.
            target_velocities: Optional list of target velocity arrays,
                one per segment. Use None for segments without target data.

        Returns:
            List of dictionaries, each containing:
                - classification (str): 'FLICK', 'TRACKING', or 'HYBRID'.
                - features (Dict): Extracted feature values.
                - segment_index (int): Index of the segment.
        """
        results: List[Dict[str, Union[str, Dict]]] = []

        for idx, segment in enumerate(segments):
            try:
                target_vel = None
                if target_velocities is not None and idx < len(target_velocities):
                    target_vel = target_velocities[idx]

                features = self.extract_features(segment, target_vel)
                classification = self._apply_rules(features)

                results.append({
                    "classification": classification,
                    "features": features,
                    "segment_index": idx,
                })

            except Exception as e:
                logger.error(
                    "Error classifying segment %d: %s", idx, str(e)
                )
                results.append({
                    "classification": self.HYBRID,
                    "features": {},
                    "segment_index": idx,
                })

        # Log classification distribution
        counts = {
            self.FLICK: sum(1 for r in results if r["classification"] == self.FLICK),
            self.TRACKING: sum(1 for r in results if r["classification"] == self.TRACKING),
            self.HYBRID: sum(1 for r in results if r["classification"] == self.HYBRID),
        }
        logger.info(
            "Batch classification: %d FLICK, %d TRACKING, %d HYBRID "
            "(%d total)",
            counts[self.FLICK], counts[self.TRACKING],
            counts[self.HYBRID], len(results)
        )

        return results

    def extract_features(
        self,
        segment: Dict,
        target_velocity: Optional[
            Union[List[Tuple[float, float]], np.ndarray]
        ] = None
    ) -> Dict[str, float]:
        """Extract classification features from a movement segment.

        Computes kinematic features used for movement classification:
        - velocity_ratio: peak_speed / mean_speed, indicates burstiness
        - accel_spike_factor: peak_accel / mean_accel
        - duration_ms: total movement duration
        - peak_speed: maximum speed reached
        - correlation_with_target: crosshair-target velocity correlation

        Args:
            segment: Movement segment dictionary with 'positions' key.
            target_velocity: Optional target velocity data for correlation.

        Returns:
            Dictionary of extracted features.
        """
        positions = np.asarray(
            segment.get("positions", []), dtype=np.float64
        )

        if positions.shape[0] < 2:
            return {
                "velocity_ratio": 0.0,
                "accel_spike_factor": 0.0,
                "duration_ms": 0.0,
                "peak_speed": 0.0,
                "correlation_with_target": 0.0,
            }

        # Compute velocity (per-frame displacements)
        velocity_xy = np.diff(positions, axis=0)  # (N-1, 2)
        speed = np.sqrt(np.sum(velocity_xy ** 2, axis=1))  # px/frame

        # Convert to px/s
        speed_per_s = speed * self.fps

        # Velocity ratio
        mean_speed = float(np.mean(speed_per_s)) if len(speed_per_s) > 0 else 1e-9
        peak_speed = float(np.max(speed_per_s)) if len(speed_per_s) > 0 else 0.0
        velocity_ratio = peak_speed / max(mean_speed, 1e-9)

        # Acceleration spike factor
        if len(velocity_xy) >= 2:
            accel_xy = np.diff(velocity_xy, axis=0) * self.fps  # px/s per frame -> px/s²
            accel_magnitude = np.sqrt(np.sum(accel_xy ** 2, axis=1))
            mean_accel = float(np.mean(accel_magnitude)) if len(accel_magnitude) > 0 else 1e-9
            peak_accel = float(np.max(accel_magnitude)) if len(accel_magnitude) > 0 else 0.0
            accel_spike_factor = peak_accel / max(mean_accel, 1e-9)
        else:
            accel_spike_factor = 0.0

        # Duration in milliseconds
        n_frames = len(positions)
        duration_ms = (n_frames / self.fps) * 1000.0 if self.fps > 0 else 0.0

        # Correlation with target velocity
        correlation_with_target = self._compute_target_correlation(
            velocity_xy, segment, target_velocity
        )

        features = {
            "velocity_ratio": float(velocity_ratio),
            "accel_spike_factor": float(accel_spike_factor),
            "duration_ms": float(duration_ms),
            "peak_speed": float(peak_speed),
            "correlation_with_target": float(correlation_with_target),
        }

        return features

    def _compute_target_correlation(
        self,
        crosshair_velocity: np.ndarray,
        segment: Dict,
        target_velocity: Optional[
            Union[List[Tuple[float, float]], np.ndarray]
        ] = None
    ) -> float:
        """Compute correlation between crosshair and target velocities.

        High correlation indicates the crosshair is following the target
        (tracking behavior). Low correlation suggests independent movement
        (flick behavior).

        Args:
            crosshair_velocity: Crosshair velocity vectors, shape (N, 2).
            segment: Movement segment dictionary (may contain target_positions).
            target_velocity: Explicit target velocity data.

        Returns:
            Pearson correlation coefficient [0, 1]. Returns 0.0 if target
            data is insufficient.
        """
        # Try to get target velocity data
        if target_velocity is not None:
            target_vel = np.asarray(target_velocity, dtype=np.float64)
        elif "target_positions" in segment:
            target_pos = np.asarray(
                segment["target_positions"], dtype=np.float64
            )
            if target_pos.shape[0] >= 2:
                target_vel = np.diff(target_pos, axis=0)
            else:
                return 0.0
        else:
            return 0.0

        # Ensure matching lengths
        min_len = min(len(crosshair_velocity), len(target_vel))
        if min_len < 3:
            return 0.0

        ch_vel = crosshair_velocity[:min_len]
        tg_vel = target_vel[:min_len]

        # Compute speed profiles
        ch_speed = np.sqrt(np.sum(ch_vel ** 2, axis=1))
        tg_speed = np.sqrt(np.sum(tg_vel ** 2, axis=1))

        # Check for zero variance
        if np.std(ch_speed) < 1e-9 or np.std(tg_speed) < 1e-9:
            return 0.0

        # Pearson correlation of speed profiles
        try:
            correlation = float(np.corrcoef(ch_speed, tg_speed)[0, 1])
            if np.isnan(correlation):
                return 0.0
            return max(0.0, correlation)  # Clamp to non-negative
        except Exception:
            return 0.0

    def _apply_rules(self, features: Dict[str, float]) -> str:
        """Apply classification rules to extracted features.

        Decision rules:
        1. FLICK: velocity_ratio > threshold AND duration < max_flick_ms
        2. TRACKING: correlation > threshold AND duration > min_tracking_ms
        3. HYBRID: Everything else

        Args:
            features: Dictionary of extracted kinematic features.

        Returns:
            Classification string: 'FLICK', 'TRACKING', or 'HYBRID'.
        """
        velocity_ratio = features.get("velocity_ratio", 0.0)
        duration_ms = features.get("duration_ms", 0.0)
        correlation = features.get("correlation_with_target", 0.0)

        # Rule 1: FLICK
        if (velocity_ratio > self.flick_velocity_ratio
                and duration_ms < self.flick_max_duration_ms):
            return self.FLICK

        # Rule 2: TRACKING
        if (correlation > self.tracking_correlation
                and duration_ms > self.tracking_min_duration_ms):
            return self.TRACKING

        # Default: HYBRID
        return self.HYBRID
