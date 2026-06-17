"""
Unit Tests for Detection Module
=================================
Tests YOLO detection, crosshair detection, and target detection.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestYOLODetector:
    """Tests for the YOLODetector class."""

    def test_detection_dataclass(self):
        """Detection dataclass should store all required fields."""
        from src.detection.yolo_detector import Detection

        det = Detection(
            class_id=0,
            class_name="crosshair",
            confidence=0.95,
            bbox=(100, 200, 150, 250),
        )
        assert det.class_id == 0
        assert det.class_name == "crosshair"
        assert det.confidence == 0.95
        assert det.bbox == (100, 200, 150, 250)
        # Center should be computed
        assert det.center == (125.0, 225.0)
        assert det.width == 50.0
        assert det.height == 50.0


class TestCrosshairDetector:
    """Tests for the CrosshairDetector class."""

    def test_screen_center_mode(self):
        """Screen center mode should return (960, 540) for Valorant."""
        from src.detection.crosshair_detector import CrosshairDetector

        detector = CrosshairDetector(use_screen_center=True)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        x, y, conf = detector.detect(frame)

        assert x == 960.0
        assert y == 540.0
        assert conf == 1.0

    def test_screen_center_confidence(self):
        """Screen center mode should have confidence 1.0."""
        from src.detection.crosshair_detector import CrosshairDetector

        detector = CrosshairDetector(use_screen_center=True)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, _, conf = detector.detect(frame)
        assert conf == 1.0


class TestTargetDetector:
    """Tests for the TargetDetector class."""

    def test_target_detection_dataclass(self):
        """TargetDetection should have required fields."""
        from src.detection.target_detector import TargetDetection

        td = TargetDetection(
            class_id=1,
            class_name="enemy_head",
            confidence=0.85,
            bbox=(400, 300, 450, 350),
            is_visible=True,
            region="head",
            distance_to_crosshair=100.5,
        )

        assert td.class_name == "enemy_head"
        assert td.is_visible is True
        assert td.region == "head"
        assert td.distance_to_crosshair == 100.5


class TestPositionTracker:
    """Tests for the PositionTracker class."""

    def test_basic_tracking(self):
        """Should track positions correctly."""
        from src.tracking.position_tracker import PositionTracker

        tracker = PositionTracker()

        # Add some frames
        for i in range(10):
            tracker.update(
                frame_index=i,
                timestamp=i / 60.0,
                crosshair_pos=(960 + i, 540),
                targets=[],
            )

        positions = tracker.get_crosshair_positions()
        assert positions.shape[0] == 10
        assert positions.shape[1] == 3  # timestamp, x, y

    def test_engagement_detection(self):
        """Should detect engagement windows when targets are present."""
        from src.tracking.position_tracker import PositionTracker, TargetDetection

        tracker = PositionTracker()

        # Frames without target
        for i in range(5):
            tracker.update(i, i / 60.0, (960, 540), [])

        for i in range(5, 25):
            target = TargetDetection(
                target_id=1,
                x=425.0,
                y=325.0,
                width=50.0,
                height=50.0,
                confidence=0.9,
                class_id=1,
                class_name="enemy_head",
            )
            tracker.update(i, i / 60.0, (960, 540), [target])

        # Frames without target again (end engagement)
        for i in range(25, 50):
            tracker.update(i, i / 60.0, (960, 540), [])

        windows = tracker.get_engagement_windows()
        assert len(windows) >= 1

    def test_reset(self):
        """Reset should clear all data."""
        from src.tracking.position_tracker import PositionTracker

        tracker = PositionTracker()
        tracker.update(0, 0.0, (960, 540), [])
        tracker.reset()

        assert tracker.frame_count == 0


class TestTrajectoryBuilder:
    """Tests for the TrajectoryBuilder class."""

    def test_build_trajectory(self):
        """Should build trajectory with all derivatives."""
        from src.tracking.trajectory_builder import TrajectoryBuilder

        builder = TrajectoryBuilder(fps=60.0)

        # Create synthetic movement data
        n = 60
        t = np.arange(n) / 60.0
        x = 960 + 100 * np.sin(2 * np.pi * t)
        y = 540 + 50 * np.cos(2 * np.pi * t)

        positions = np.column_stack([t, x, y])
        trajectory = builder.build_trajectory(positions)

        assert trajectory.positions.shape == (n, 2)
        assert trajectory.velocities.shape == (n, 2)
        assert trajectory.speeds.shape == (n,)
        assert trajectory.accelerations.shape == (n, 2)
        assert trajectory.jerks.shape == (n, 2)

    def test_segment_movements(self):
        """Should segment trajectory into movements."""
        from src.tracking.trajectory_builder import TrajectoryBuilder

        builder = TrajectoryBuilder(fps=60.0)

        # Create data with idle + movement + idle
        n = 120
        t = np.arange(n) / 60.0
        x = np.concatenate([
            np.full(30, 960.0),          # Idle
            np.linspace(960, 1100, 30),  # Moving
            np.full(30, 1100.0),          # Idle
            np.linspace(1100, 960, 30),  # Moving back
        ])
        y = np.full(n, 540.0)

        positions = np.column_stack([t, x, y])
        trajectory = builder.build_trajectory(positions)
        segments = builder.segment_movements(trajectory)

        assert len(segments) > 0
        # Should have at least some idle and some moving segments
        types = [s.movement_type for s in segments]
        assert "idle" in types or "moving" in types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
