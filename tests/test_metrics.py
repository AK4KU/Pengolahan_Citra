"""
Unit Tests for Metrics Module
===============================
Tests all aim performance metrics with synthetic data.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAimAccuracy:
    """Tests for the AimAccuracy metric."""

    def setup_method(self):
        from src.metrics.aim_accuracy import AimAccuracy
        self.metric = AimAccuracy()

    def test_perfect_accuracy(self):
        """Crosshair exactly on target should give 100% accuracy."""
        crosshair = np.array([[100, 200], [100, 200], [100, 200]])
        target = np.array([[100, 200], [100, 200], [100, 200]])
        result = self.metric.compute(crosshair, target, threshold=30)
        assert result["accuracy_rate"] == 1.0
        assert result["frames_on_target"] == 3

    def test_zero_accuracy(self):
        """Crosshair far from target should give 0% accuracy."""
        crosshair = np.array([[0, 0], [0, 0], [0, 0]])
        target = np.array([[1000, 1000], [1000, 1000], [1000, 1000]])
        result = self.metric.compute(crosshair, target, threshold=30)
        assert result["accuracy_rate"] == 0.0

    def test_partial_accuracy(self):
        """Mixed on/off target should give partial accuracy."""
        crosshair = np.array([[100, 200], [500, 500], [100, 200]])
        target = np.array([[100, 200], [100, 200], [100, 200]])
        result = self.metric.compute(crosshair, target, threshold=30)
        assert 0 < result["accuracy_rate"] < 1.0

    def test_threshold_sensitivity(self):
        """Larger threshold should capture more frames as on-target."""
        crosshair = np.array([[100, 200], [120, 210]])
        target = np.array([[100, 200], [100, 200]])

        result_tight = self.metric.compute(crosshair, target, threshold=10)
        result_loose = self.metric.compute(crosshair, target, threshold=50)

        assert result_loose["accuracy_rate"] >= result_tight["accuracy_rate"]

    def test_mean_distance(self):
        """Mean distance should be computed correctly."""
        crosshair = np.array([[0, 0], [10, 0]])
        target = np.array([[5, 0], [5, 0]])
        result = self.metric.compute(crosshair, target, threshold=100)
        assert abs(result["mean_distance"] - 5.0) < 0.01


class TestTimeToTarget:
    """Tests for the TimeToTarget metric."""

    def setup_method(self):
        from src.metrics.time_to_target import TimeToTarget
        self.metric = TimeToTarget()

    def test_fitts_law_basic(self):
        """Fitts' Law analysis should produce valid coefficients."""
        # Synthetic data following Fitts' Law pattern
        distances = [100, 200, 300, 400, 500]
        target_widths = [50, 50, 50, 50, 50]
        ttts = [200, 300, 380, 440, 500]  # Increasing with distance

        result = self.metric.fitts_law_analysis(ttts, distances, target_widths)

        assert "a_coefficient" in result
        assert "b_coefficient" in result
        assert "r_squared" in result
        assert result["b_coefficient"] > 0  # Should be positive (harder = slower)


class TestOvershoot:
    """Tests for the OvershootAnalyzer metric."""

    def setup_method(self):
        from src.metrics.overshoot import OvershootAnalyzer
        self.metric = OvershootAnalyzer()


class TestConsistency:
    """Tests for the ConsistencyAnalyzer metric."""

    def setup_method(self):
        from src.metrics.consistency import ConsistencyAnalyzer
        self.metric = ConsistencyAnalyzer()

    def test_perfect_consistency(self):
        """Same position every frame = perfect consistency."""
        positions = np.array([[960, 540]] * 100)
        result = self.metric.compute(positions)
        assert result["sigma_2d"] < 0.01
        assert result["consistency_score"] > 0.99

    def test_bcea_calculation(self):
        """BCEA should be computable from position data."""
        np.random.seed(42)
        positions = np.random.normal(loc=[960, 540], scale=[10, 8], size=(100, 2))
        bcea = self.metric.compute_bcea(positions)
        assert bcea > 0  # Should be positive area


class TestKinematics:
    """Tests for the KinematicsAnalyzer metric."""

    def setup_method(self):
        from src.metrics.kinematics import KinematicsAnalyzer
        self.metric = KinematicsAnalyzer()

    def test_stationary_kinematics(self):
        """Stationary crosshair should have near-zero velocity."""
        from src.tracking.trajectory_builder import Trajectory
        n = 60
        positions = np.column_stack([np.full(n, 960.0), np.full(n, 540.0)])
        timestamps = np.arange(n) / 60.0

        trajectory = Trajectory(
            timestamps=timestamps,
            positions=positions,
            raw_positions=positions.copy(),
            velocities=np.zeros_like(positions),
            speeds=np.zeros(n),
            accelerations=np.zeros_like(positions),
            acceleration_magnitudes=np.zeros(n),
            jerks=np.zeros_like(positions),
            jerk_magnitudes=np.zeros(n),
        )

        result = self.metric.compute(trajectory.positions)
        assert result["peak_velocity"] < 1.0
        assert result["mean_velocity"] < 1.0


class TestAimClassifier:
    """Tests for the AimClassifier."""

    def setup_method(self):
        from src.metrics.aim_classifier import AimClassifier
        self.classifier = AimClassifier()


class TestSessionAnalyzer:
    """Tests for the SessionAnalyzer."""

    def setup_method(self):
        from src.metrics.session_analyzer import SessionAnalyzer
        self.analyzer = SessionAnalyzer(fps=60.0)

    def test_skill_classification(self):
        """Skill classification should return valid levels."""
        from src.metrics.session_analyzer import SessionReport

        # Create a beginner-level report
        report = SessionReport(
            session_id="test",
            timestamp="2024-01-01T00:00:00",
            duration_seconds=60.0,
            total_frames=3600,
            aim_accuracy={"accuracy_rate": 0.2},
            time_to_target={"mean_ttt_ms": 900},
            overshoot={"overshoot_ratio": 0.6},
            consistency={"consistency_score": 0.3},
            kinematics={"ldlj": -5.0, "sparc": -3.0},
            aim_classification={"flick_count": 5, "tracking_count": 3},
            overall_score=0.0,
            skill_level="",
            engagement_count=8,
            per_engagement_details=[],
        )

        level = self.analyzer.classify_skill_level(report)
        assert level in ["beginner", "intermediate", "advanced"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
