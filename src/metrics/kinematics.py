"""
FPS Aim Performance Analyzer - Kinematics Analysis Metric

Computes kinematic properties of crosshair trajectory including velocity,
acceleration, and jerk profiles. Includes smoothness metrics:

- LDLJ (Log Dimensionless Jerk): Measures movement smoothness based on
  jerk (rate of change of acceleration). Lower (more negative) values
  indicate smoother movement.
  LDLJ = -ln(T^5 / L^2 * integral(|j|^2 dt))

- SPARC (Spectral Arc Length): Measures smoothness from the frequency
  spectrum of the velocity profile. Values closer to 0 indicate smoother
  movement (typically negative, range roughly [-7, 0]).

- Velocity peak counting: Number of local maxima in the speed profile,
  indicating submovements or corrections.

Author: FPS Aim Performance Analyzer
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter, find_peaks

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class KinematicsAnalyzer:
    """Analyzes kinematic properties of aim trajectories.

    Computes velocity, acceleration, and jerk profiles from crosshair
    position data. Provides movement smoothness metrics (LDLJ, SPARC)
    and velocity peak analysis.

    Attributes:
        fps: Frames per second of the capture.
        smoothing_window: Savitzky-Golay filter window size.
        poly_order: Savitzky-Golay polynomial order.
    """

    def __init__(
        self,
        fps: Optional[float] = None,
        smoothing_window: Optional[int] = None,
        poly_order: Optional[int] = None
    ) -> None:
        """Initialize KinematicsAnalyzer.

        Args:
            fps: Frames per second. Defaults to settings.CAPTURE_FPS.
            smoothing_window: Savitzky-Golay filter window (must be odd).
                Defaults to settings.KINEMATICS_SMOOTHING_WINDOW.
            poly_order: Savitzky-Golay polynomial order.
                Defaults to settings.KINEMATICS_POLY_ORDER.
        """
        self.fps: float = fps if fps is not None else settings.CAPTURE_FPS
        self.smoothing_window: int = (
            smoothing_window if smoothing_window is not None
            else settings.KINEMATICS_SMOOTHING_WINDOW
        )
        self.poly_order: int = (
            poly_order if poly_order is not None
            else settings.KINEMATICS_POLY_ORDER
        )
        self._dt: float = 1.0 / self.fps if self.fps > 0 else 1.0 / 60.0

    def compute(
        self,
        trajectory: Union[List[Tuple[float, float]], np.ndarray]
    ) -> Dict[str, Union[float, int, List[float]]]:
        """Compute full kinematic analysis of a trajectory.

        Derives velocity, acceleration, and jerk profiles from position
        data using numerical differentiation with optional Savitzky-Golay
        smoothing. Computes smoothness metrics (LDLJ, SPARC) and counts
        velocity peaks.

        Args:
            trajectory: Array of (x, y) positions, shape (N, 2).
                Must have at least 5 data points for meaningful analysis.

        Returns:
            Dictionary with keys:
                - peak_velocity (float): Maximum speed (px/s).
                - mean_velocity (float): Mean speed (px/s).
                - peak_acceleration (float): Maximum acceleration (px/s²).
                - mean_acceleration (float): Mean acceleration (px/s²).
                - peak_jerk (float): Maximum jerk (px/s³).
                - mean_jerk (float): Mean jerk (px/s³).
                - time_to_peak_velocity (float): Time to reach peak speed (s).
                - ldlj (float): Log Dimensionless Jerk smoothness metric.
                - sparc (float): Spectral Arc Length smoothness metric.
                - num_velocity_peaks (int): Number of speed local maxima.
                - velocity_profile (List[float]): Speed at each frame (px/s).
                - acceleration_profile (List[float]): Accel at each frame.
                - jerk_profile (List[float]): Jerk at each frame.
        """
        positions = np.asarray(trajectory, dtype=np.float64)

        if positions.shape[0] < 3:
            logger.warning(
                "Trajectory too short (%d points) for kinematics analysis.",
                positions.shape[0]
            )
            return self._empty_result()

        # Smooth positions if enough data points
        smoothed = self._smooth_trajectory(positions)

        # Compute derivatives
        velocity_xy = self._compute_derivative(smoothed)  # (N-1, 2) in px/frame
        speed_profile = np.sqrt(np.sum(velocity_xy ** 2, axis=1))  # px/frame

        # Convert to px/s
        speed_profile_per_s = speed_profile * self.fps
        velocity_xy_per_s = velocity_xy * self.fps

        # Acceleration
        if len(velocity_xy_per_s) >= 2:
            accel_xy = self._compute_derivative(velocity_xy_per_s)  # px/s per frame
            accel_xy_per_s = accel_xy * self.fps  # px/s²
            accel_magnitude = np.sqrt(np.sum(accel_xy_per_s ** 2, axis=1))
        else:
            accel_magnitude = np.array([0.0])
            accel_xy_per_s = np.array([[0.0, 0.0]])

        # Jerk
        if len(accel_xy_per_s) >= 2:
            jerk_xy = self._compute_derivative(accel_xy_per_s)  # px/s² per frame
            jerk_xy_per_s = jerk_xy * self.fps  # px/s³
            jerk_magnitude = np.sqrt(np.sum(jerk_xy_per_s ** 2, axis=1))
        else:
            jerk_magnitude = np.array([0.0])

        # Kinematics stats
        peak_velocity = float(np.max(speed_profile_per_s)) if len(speed_profile_per_s) > 0 else 0.0
        mean_velocity = float(np.mean(speed_profile_per_s)) if len(speed_profile_per_s) > 0 else 0.0
        peak_acceleration = float(np.max(accel_magnitude)) if len(accel_magnitude) > 0 else 0.0
        mean_acceleration = float(np.mean(accel_magnitude)) if len(accel_magnitude) > 0 else 0.0
        peak_jerk = float(np.max(jerk_magnitude)) if len(jerk_magnitude) > 0 else 0.0
        mean_jerk = float(np.mean(jerk_magnitude)) if len(jerk_magnitude) > 0 else 0.0

        # Time to peak velocity
        if len(speed_profile_per_s) > 0:
            peak_vel_idx = int(np.argmax(speed_profile_per_s))
            time_to_peak_velocity = float(peak_vel_idx * self._dt)
        else:
            time_to_peak_velocity = 0.0

        # Smoothness metrics
        duration = len(positions) * self._dt
        path_length = float(np.sum(speed_profile)) * self._dt * self.fps  # total px

        ldlj = self.compute_ldlj(jerk_magnitude, duration, path_length)
        sparc = self.compute_sparc(speed_profile_per_s, self.fps)
        num_peaks = self.compute_num_velocity_peaks(speed_profile_per_s)

        result = {
            "peak_velocity": peak_velocity,
            "mean_velocity": mean_velocity,
            "peak_acceleration": peak_acceleration,
            "mean_acceleration": mean_acceleration,
            "peak_jerk": peak_jerk,
            "mean_jerk": mean_jerk,
            "time_to_peak_velocity": time_to_peak_velocity,
            "ldlj": ldlj,
            "sparc": sparc,
            "num_velocity_peaks": num_peaks,
            "velocity_profile": speed_profile_per_s.tolist(),
            "acceleration_profile": accel_magnitude.tolist(),
            "jerk_profile": jerk_magnitude.tolist(),
        }

        logger.info(
            "Kinematics: peak_vel=%.1f px/s, mean_vel=%.1f px/s, "
            "LDLJ=%.3f, SPARC=%.3f, velocity_peaks=%d",
            peak_velocity, mean_velocity, ldlj, sparc, num_peaks
        )

        return result

    def _smooth_trajectory(self, positions: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to trajectory positions.

        Smooths both x and y coordinates independently to reduce noise
        while preserving movement features. Falls back to raw data if
        the trajectory is too short for the filter window.

        Args:
            positions: Array of shape (N, 2).

        Returns:
            Smoothed positions array of same shape.
        """
        n_points = len(positions)
        window = self.smoothing_window

        # Ensure window is valid (odd, >= poly_order + 1, <= data length)
        if window > n_points:
            window = n_points if n_points % 2 == 1 else n_points - 1
        if window < self.poly_order + 2:
            return positions.copy()
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return positions.copy()

        try:
            smoothed_x = savgol_filter(
                positions[:, 0], window, self.poly_order
            )
            smoothed_y = savgol_filter(
                positions[:, 1], window, self.poly_order
            )
            return np.column_stack([smoothed_x, smoothed_y])
        except Exception as e:
            logger.warning("Savitzky-Golay smoothing failed: %s", str(e))
            return positions.copy()

    def _compute_derivative(self, data: np.ndarray) -> np.ndarray:
        """Compute numerical derivative using np.gradient.

        Uses central differences for interior points and forward/backward
        differences at the boundaries.

        Args:
            data: Array of shape (N,) or (N, 2).

        Returns:
            Derivative array of same shape as input.
        """
        if data.ndim == 1:
            return np.gradient(data)
        else:
            dx = np.gradient(data[:, 0])
            dy = np.gradient(data[:, 1])
            return np.column_stack([dx, dy])

    def compute_ldlj(
        self,
        jerk_magnitudes: Union[List[float], np.ndarray],
        duration: float,
        path_length: float
    ) -> float:
        """Compute Log Dimensionless Jerk (LDLJ) smoothness metric.

        LDLJ normalizes the jerk integral by movement duration and path
        length, producing a dimensionless measure of smoothness. More
        negative values indicate smoother movements.

        Formula: LDLJ = -ln(T^5 / L^2 * integral(|j|^2 dt))

        Args:
            jerk_magnitudes: Array of jerk magnitudes at each time step.
            duration: Total movement duration in seconds.
            path_length: Total path length in pixels.

        Returns:
            LDLJ value (typically negative, more negative = smoother).
            Returns 0.0 if computation is invalid.
        """
        jerk_arr = np.asarray(jerk_magnitudes, dtype=np.float64)

        if duration <= 0 or path_length <= 0 or len(jerk_arr) == 0:
            return 0.0

        # Integrate jerk squared using trapezoidal rule
        dt = duration / len(jerk_arr) if len(jerk_arr) > 1 else duration
        jerk_squared_integral = float(np.trapz(jerk_arr ** 2, dx=dt))

        if jerk_squared_integral <= 0:
            return 0.0

        # Dimensionless jerk
        dimensionless_jerk = (
            (duration ** 5) / (path_length ** 2) * jerk_squared_integral
        )

        if dimensionless_jerk <= 0:
            return 0.0

        ldlj = -float(np.log(dimensionless_jerk))
        return ldlj

    def compute_sparc(
        self,
        speed_profile: Union[List[float], np.ndarray],
        fps: Optional[float] = None,
        fc: float = 10.0
    ) -> float:
        """Compute Spectral Arc Length (SPARC) smoothness metric.

        SPARC measures smoothness from the frequency-domain representation
        of the speed profile. It computes the arc length of the normalized
        magnitude spectrum up to a cutoff frequency. Values closer to 0
        (less negative) indicate smoother movements.

        Args:
            speed_profile: Speed values at each time step (px/s).
            fps: Sampling rate in Hz. Uses self.fps if not specified.
            fc: Cutoff frequency in Hz for spectral analysis.

        Returns:
            SPARC value (negative, closer to 0 = smoother).
            Returns 0.0 if computation is invalid.
        """
        speed_arr = np.asarray(speed_profile, dtype=np.float64)
        sample_fps = fps if fps is not None else self.fps

        if len(speed_arr) < 4 or sample_fps <= 0:
            return 0.0

        # Normalize speed profile
        max_speed = np.max(speed_arr)
        if max_speed < 1e-9:
            return 0.0

        speed_normalized = speed_arr / max_speed

        # Compute FFT
        n_fft = len(speed_normalized)
        freq_spectrum = np.fft.rfft(speed_normalized)
        magnitude_spectrum = np.abs(freq_spectrum) / n_fft
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_fps)

        # Normalize magnitude spectrum
        max_magnitude = np.max(magnitude_spectrum)
        if max_magnitude < 1e-9:
            return 0.0
        magnitude_normalized = magnitude_spectrum / max_magnitude

        # Apply cutoff frequency
        cutoff_mask = frequencies <= fc
        freq_cutoff = frequencies[cutoff_mask]
        mag_cutoff = magnitude_normalized[cutoff_mask]

        if len(freq_cutoff) < 2:
            return 0.0

        # Compute arc length of the normalized magnitude spectrum
        d_freq = np.diff(freq_cutoff)
        d_mag = np.diff(mag_cutoff)

        # Normalize frequency axis to [0, 1] for dimensionless computation
        freq_range = freq_cutoff[-1] - freq_cutoff[0]
        if freq_range < 1e-9:
            return 0.0

        d_freq_norm = d_freq / freq_range
        arc_lengths = np.sqrt(d_freq_norm ** 2 + d_mag ** 2)
        sparc = -float(np.sum(arc_lengths))

        return sparc

    def compute_num_velocity_peaks(
        self,
        speed_profile: Union[List[float], np.ndarray],
        prominence: Optional[float] = None
    ) -> int:
        """Count the number of local maxima in the speed profile.

        Each velocity peak typically corresponds to a submovement or
        correction. Smooth movements have few peaks (ideally 1 for a
        ballistic movement), while corrective movements have many.

        Args:
            speed_profile: Speed values at each time step.
            prominence: Minimum prominence for a peak to be counted.
                If None, uses 10% of the maximum speed.

        Returns:
            Number of detected velocity peaks.
        """
        speed_arr = np.asarray(speed_profile, dtype=np.float64)

        if len(speed_arr) < 3:
            return 0

        max_speed = np.max(speed_arr)
        if max_speed < 1e-6:
            return 0

        if prominence is None:
            prominence = max_speed * 0.1

        peaks, properties = find_peaks(speed_arr, prominence=prominence)

        return len(peaks)

    def _empty_result(self) -> Dict[str, Union[float, int, List[float]]]:
        """Return an empty result dictionary with zero values.

        Returns:
            Dictionary with all kinematic keys set to zero/empty.
        """
        return {
            "peak_velocity": 0.0,
            "mean_velocity": 0.0,
            "peak_acceleration": 0.0,
            "mean_acceleration": 0.0,
            "peak_jerk": 0.0,
            "mean_jerk": 0.0,
            "time_to_peak_velocity": 0.0,
            "ldlj": 0.0,
            "sparc": 0.0,
            "num_velocity_peaks": 0,
            "velocity_profile": [],
            "acceleration_profile": [],
            "jerk_profile": [],
        }
