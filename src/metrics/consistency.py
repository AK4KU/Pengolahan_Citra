"""
FPS Aim Performance Analyzer - Consistency Analysis Metric

Measures how consistently the player maintains crosshair position near
the target. Uses Bivariate Contour Ellipse Area (BCEA) to quantify the
spatial spread of crosshair positions, commonly used in eye-tracking
and motor control research.

BCEA = 2 * pi * k * sigma_x * sigma_y * sqrt(1 - rho^2)
where:
    k = chi2.ppf(confidence, df=2) / 2
    sigma_x, sigma_y = standard deviations of x and y positions
    rho = Pearson correlation coefficient between x and y

Author: FPS Aim Performance Analyzer
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from scipy.stats import chi2

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class ConsistencyAnalyzer:
    """Analyzes the spatial consistency of crosshair positioning.

    Computes how tightly clustered the crosshair positions are around
    the mean or ideal position. Uses standard deviation, BCEA, and a
    normalized consistency score.

    Attributes:
        screen_diagonal: Diagonal distance of the screen in pixels, used
            for normalization.
        window_size: Number of frames for windowed consistency analysis.
    """

    def __init__(
        self,
        screen_width: int = None,
        screen_height: int = None,
        window_size: Optional[int] = None
    ) -> None:
        """Initialize ConsistencyAnalyzer.

        Args:
            screen_width: Screen width in pixels. Defaults to
                settings.SCREEN_WIDTH.
            screen_height: Screen height in pixels. Defaults to
                settings.SCREEN_HEIGHT.
            window_size: Frame window for windowed analysis. Defaults to
                settings.CONSISTENCY_WINDOW_SIZE.
        """
        width = screen_width if screen_width is not None else settings.SCREEN_WIDTH
        height = screen_height if screen_height is not None else settings.SCREEN_HEIGHT
        self.screen_diagonal: float = float(np.sqrt(width ** 2 + height ** 2))
        self.window_size: int = (
            window_size if window_size is not None
            else settings.CONSISTENCY_WINDOW_SIZE
        )

    def compute(
        self,
        crosshair_positions: Union[List[Tuple[float, float]], np.ndarray],
        target_positions: Optional[
            Union[List[Tuple[float, float]], np.ndarray]
        ] = None
    ) -> Dict[str, float]:
        """Compute consistency metrics from crosshair positions.

        Measures the spatial spread of crosshair positions using standard
        deviations, BCEA, and a normalized consistency score. If target
        positions are provided, deviations are computed relative to targets;
        otherwise, deviations are from the mean crosshair position.

        Args:
            crosshair_positions: Array of (x, y) crosshair positions.
            target_positions: Optional array of (x, y) target positions.
                If provided, consistency is measured as deviation from
                the target. If None, deviation is from the mean position.

        Returns:
            Dictionary with keys:
                - sigma_x (float): Standard deviation of x-positions (pixels).
                - sigma_y (float): Standard deviation of y-positions (pixels).
                - sigma_2d (float): Combined 2D standard deviation =
                    sqrt(sigma_x^2 + sigma_y^2).
                - consistency_score (float): Normalized score [0, 1] where
                    1.0 = perfectly consistent, computed as
                    1 - (sigma_2d / max_distance).
                - bcea (float): Bivariate Contour Ellipse Area in pixels^2.
                - mean_deviation_from_ideal (float): Mean distance from
                    ideal position (target or mean) in pixels.
        """
        positions = np.asarray(crosshair_positions, dtype=np.float64)

        if positions.shape[0] < 2:
            logger.warning(
                "Insufficient positions (%d) for consistency analysis.",
                positions.shape[0]
            )
            return {
                "sigma_x": 0.0,
                "sigma_y": 0.0,
                "sigma_2d": 0.0,
                "consistency_score": 1.0,
                "bcea": 0.0,
                "mean_deviation_from_ideal": 0.0,
            }

        # Determine ideal positions (target or mean)
        if target_positions is not None:
            ideal = np.asarray(target_positions, dtype=np.float64)
            min_len = min(len(positions), len(ideal))
            positions = positions[:min_len]
            ideal = ideal[:min_len]
            deviations = positions - ideal
        else:
            mean_pos = np.mean(positions, axis=0)
            deviations = positions - mean_pos

        # Standard deviations
        sigma_x = float(np.std(deviations[:, 0]))
        sigma_y = float(np.std(deviations[:, 1]))
        sigma_2d = float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))

        # Consistency score: 1 - (sigma_2d / screen_diagonal)
        consistency_score = max(0.0, 1.0 - (sigma_2d / self.screen_diagonal))

        # BCEA at default 68% confidence
        bcea = self.compute_bcea(positions, confidence=0.68)

        # Mean deviation from ideal position
        deviation_distances = np.sqrt(np.sum(deviations ** 2, axis=1))
        mean_deviation_from_ideal = float(np.mean(deviation_distances))

        result = {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_2d": sigma_2d,
            "consistency_score": float(consistency_score),
            "bcea": bcea,
            "mean_deviation_from_ideal": mean_deviation_from_ideal,
        }

        logger.info(
            "Consistency analysis: sigma_x=%.1f, sigma_y=%.1f, "
            "sigma_2d=%.1f, score=%.3f, BCEA=%.1f px²",
            sigma_x, sigma_y, sigma_2d, consistency_score, bcea
        )

        return result

    def compute_bcea(
        self,
        positions: Union[List[Tuple[float, float]], np.ndarray],
        confidence: float = 0.68
    ) -> float:
        """Compute Bivariate Contour Ellipse Area (BCEA).

        BCEA quantifies the area of the ellipse that contains the specified
        proportion of position samples. This metric is commonly used in
        eye-tracking and fixation stability research.

        Formula: BCEA = 2 * pi * k * sigma_x * sigma_y * sqrt(1 - rho^2)
        where k = chi2.ppf(confidence, df=2) / 2

        Args:
            positions: Array of (x, y) positions, shape (N, 2).
            confidence: Confidence level for the ellipse [0, 1].
                0.68 ≈ 1-sigma ellipse, 0.95 ≈ 2-sigma ellipse.

        Returns:
            BCEA value in pixels squared. Returns 0.0 if insufficient data.
        """
        pos_arr = np.asarray(positions, dtype=np.float64)

        if pos_arr.shape[0] < 3:
            return 0.0

        x = pos_arr[:, 0]
        y = pos_arr[:, 1]

        sigma_x = np.std(x)
        sigma_y = np.std(y)

        if sigma_x < 1e-9 or sigma_y < 1e-9:
            return 0.0

        # Pearson correlation coefficient
        if len(x) > 1:
            correlation_matrix = np.corrcoef(x, y)
            rho = float(correlation_matrix[0, 1])
            # Handle numerical issues
            rho = np.clip(rho, -0.9999, 0.9999)
        else:
            rho = 0.0

        # k from chi-squared distribution for the given confidence level
        # For 2 degrees of freedom (bivariate)
        k = float(chi2.ppf(confidence, df=2)) / 2.0

        # BCEA formula
        bcea = 2.0 * np.pi * k * sigma_x * sigma_y * np.sqrt(1.0 - rho ** 2)

        return float(bcea)
