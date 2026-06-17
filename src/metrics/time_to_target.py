"""
FPS Aim Performance Analyzer - Time-to-Target Metric

Measures the time it takes for the crosshair to first reach the target
after the target appears. Includes Fitts' Law analysis to model the
speed-accuracy tradeoff in aiming movements.

Fitts' Law: MT = a + b * log2(D/W + 1)
    MT = Movement Time
    D  = Distance to target
    W  = Target width
    a, b = empirically fitted coefficients

Author: FPS Aim Performance Analyzer
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class TimeToTarget:
    """Analyzes time-to-target (TTT) for aim performance evaluation.

    Computes how quickly the player's crosshair reaches each target after
    it appears. Provides descriptive statistics and Fitts' Law modeling
    for speed-accuracy analysis.

    Attributes:
        fps: Frame rate of the capture, used for time conversion.
        default_threshold: Default pixel distance threshold for alignment.
    """

    def __init__(
        self,
        fps: float = None,
        default_threshold: Optional[float] = None
    ) -> None:
        """Initialize TimeToTarget analyzer.

        Args:
            fps: Frames per second of the capture. Defaults to
                settings.CAPTURE_FPS.
            default_threshold: Default alignment threshold in pixels.
                Defaults to settings.TTT_ALIGNMENT_THRESHOLD.
        """
        self.fps: float = fps if fps is not None else settings.CAPTURE_FPS
        self.default_threshold: float = (
            default_threshold if default_threshold is not None
            else settings.TTT_ALIGNMENT_THRESHOLD
        )

    def _frames_to_ms(self, frame_count: int) -> float:
        """Convert a frame count to milliseconds.

        Args:
            frame_count: Number of frames.

        Returns:
            Duration in milliseconds.
        """
        if self.fps <= 0:
            return 0.0
        return (frame_count / self.fps) * 1000.0

    def compute(
        self,
        engagement_windows: List[Dict],
        threshold: Optional[float] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """Compute time-to-target statistics across engagement windows.

        For each engagement, TTT is the duration from target appearance
        to the first frame where the crosshair is within the threshold
        distance of the target.

        Args:
            engagement_windows: List of engagement dictionaries, each with:
                - crosshair_positions: List of (x, y) crosshair positions.
                - target_positions: List of (x, y) target positions.
                - start_frame (optional): Frame index of target appearance.
                - fps (optional): Override FPS for this engagement.
            threshold: Pixel distance threshold for alignment. Uses default
                if not specified.

        Returns:
            Dictionary with keys:
                - mean_ttt_ms (float): Mean time-to-target in milliseconds.
                - median_ttt_ms (float): Median time-to-target in ms.
                - std_ttt_ms (float): Standard deviation of TTT in ms.
                - min_ttt_ms (float): Minimum TTT in ms.
                - max_ttt_ms (float): Maximum TTT in ms.
                - p25 (float): 25th percentile TTT in ms.
                - p75 (float): 75th percentile TTT in ms.
                - individual_ttts (List[float]): Per-engagement TTTs in ms.

            Returns zero-filled dict if no valid TTTs computed.
        """
        threshold = threshold if threshold is not None else self.default_threshold
        individual_ttts: List[float] = []

        for idx, engagement in enumerate(engagement_windows):
            try:
                ttt_ms = self._compute_single_ttt(engagement, threshold)
                if ttt_ms is not None:
                    individual_ttts.append(ttt_ms)
            except Exception as e:
                logger.error(
                    "Error computing TTT for engagement %d: %s", idx, str(e)
                )

        if not individual_ttts:
            logger.warning("No valid TTTs computed from %d engagements.",
                           len(engagement_windows))
            return {
                "mean_ttt_ms": 0.0,
                "median_ttt_ms": 0.0,
                "std_ttt_ms": 0.0,
                "min_ttt_ms": 0.0,
                "max_ttt_ms": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "individual_ttts": [],
            }

        ttt_array = np.array(individual_ttts)
        result = {
            "mean_ttt_ms": float(np.mean(ttt_array)),
            "median_ttt_ms": float(np.median(ttt_array)),
            "std_ttt_ms": float(np.std(ttt_array)),
            "min_ttt_ms": float(np.min(ttt_array)),
            "max_ttt_ms": float(np.max(ttt_array)),
            "p25": float(np.percentile(ttt_array, 25)),
            "p75": float(np.percentile(ttt_array, 75)),
            "individual_ttts": individual_ttts,
        }

        logger.info(
            "TTT computed: mean=%.1f ms, median=%.1f ms, std=%.1f ms "
            "(%d engagements)",
            result["mean_ttt_ms"], result["median_ttt_ms"],
            result["std_ttt_ms"], len(individual_ttts)
        )

        return result

    def _compute_single_ttt(
        self,
        engagement: Dict,
        threshold: float
    ) -> Optional[float]:
        """Compute TTT for a single engagement window.

        TTT = time when crosshair first enters target radius - time when
        target appeared. In frame terms, this is the index of the first
        frame where distance <= threshold.

        Args:
            engagement: Engagement dictionary with crosshair_positions
                and target_positions.
            threshold: Distance threshold in pixels.

        Returns:
            TTT in milliseconds, or None if crosshair never reached target.
        """
        crosshair_pos = np.asarray(
            engagement.get("crosshair_positions", []), dtype=np.float64
        )
        target_pos = np.asarray(
            engagement.get("target_positions", []), dtype=np.float64
        )

        if crosshair_pos.shape[0] == 0 or target_pos.shape[0] == 0:
            return None

        min_len = min(len(crosshair_pos), len(target_pos))
        crosshair_pos = crosshair_pos[:min_len]
        target_pos = target_pos[:min_len]

        # Compute per-frame distances
        distances = np.sqrt(
            np.sum((crosshair_pos - target_pos) ** 2, axis=1)
        )

        # Find first frame where crosshair enters target radius
        aligned_frames = np.where(distances <= threshold)[0]

        if len(aligned_frames) == 0:
            return None  # Crosshair never reached target

        first_aligned_frame = int(aligned_frames[0])
        fps = engagement.get("fps", self.fps)

        ttt_ms = (first_aligned_frame / fps) * 1000.0 if fps > 0 else 0.0
        return ttt_ms

    def fitts_law_analysis(
        self,
        ttts: List[float],
        distances: List[float],
        target_widths: List[float]
    ) -> Dict[str, Union[float, List[float]]]:
        """Perform Fitts' Law analysis on aim movement data.

        Fits the model MT = a + b * log2(D/W + 1) using linear regression,
        where MT is movement time, D is distance to target, and W is target
        width. This models the speed-accuracy tradeoff in pointing tasks.

        Args:
            ttts: List of time-to-target values in milliseconds.
            distances: List of distances to target in pixels (one per TTT).
            target_widths: List of target widths in pixels (one per TTT).

        Returns:
            Dictionary with keys:
                - a_coefficient (float): Intercept of the linear fit (ms).
                - b_coefficient (float): Slope of the linear fit (ms/bit).
                - r_squared (float): Coefficient of determination [0, 1].
                - throughput_bits_per_sec (float): Information throughput
                    (bits/second), computed as 1/b * 1000.
                - index_of_difficulty (List[float]): ID values (bits) for
                    each movement, computed as log2(D/W + 1).

        Raises:
            ValueError: If input lists have different lengths or fewer
                than 2 data points.
        """
        if len(ttts) != len(distances) or len(ttts) != len(target_widths):
            raise ValueError(
                f"All input lists must have the same length. Got "
                f"ttts={len(ttts)}, distances={len(distances)}, "
                f"target_widths={len(target_widths)}."
            )

        if len(ttts) < 2:
            raise ValueError(
                "At least 2 data points are required for Fitts' Law analysis."
            )

        ttts_arr = np.array(ttts, dtype=np.float64)
        distances_arr = np.array(distances, dtype=np.float64)
        widths_arr = np.array(target_widths, dtype=np.float64)

        # Avoid division by zero for very small target widths
        widths_arr = np.maximum(widths_arr, 1e-6)

        # Index of Difficulty: ID = log2(D/W + 1) in bits
        index_of_difficulty = np.log2(distances_arr / widths_arr + 1.0)

        # Linear regression: MT = a + b * ID
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            index_of_difficulty, ttts_arr
        )

        r_squared = float(r_value ** 2)

        # Throughput: TP = 1/b (bits/ms) -> convert to bits/sec
        if abs(slope) > 1e-9:
            throughput_bits_per_sec = (1.0 / slope) * 1000.0
        else:
            throughput_bits_per_sec = float("inf")

        result = {
            "a_coefficient": float(intercept),
            "b_coefficient": float(slope),
            "r_squared": r_squared,
            "throughput_bits_per_sec": float(throughput_bits_per_sec),
            "index_of_difficulty": index_of_difficulty.tolist(),
        }

        logger.info(
            "Fitts' Law analysis: a=%.2f, b=%.2f, R²=%.4f, "
            "throughput=%.2f bits/sec",
            intercept, slope, r_squared, throughput_bits_per_sec
        )

        return result
