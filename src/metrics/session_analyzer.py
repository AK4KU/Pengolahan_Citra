"""
FPS Aim Performance Analyzer - Session Analyzer

Orchestrates all individual metrics (aim accuracy, time-to-target, overshoot,
consistency, kinematics, aim classification) into a comprehensive session
analysis. Produces a SessionReport dataclass with aggregate scores, skill
level classification, and export capabilities.

Overall Score (weighted composite):
    accuracy * 0.30
  + ttt_score * 0.20
  + overshoot_score * 0.15
  + consistency * 0.20
  + smoothness * 0.15

Author: FPS Aim Performance Analyzer
"""

import sys
import json
import csv
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

from .aim_accuracy import AimAccuracy
from .time_to_target import TimeToTarget
from .overshoot import OvershootAnalyzer
from .consistency import ConsistencyAnalyzer
from .kinematics import KinematicsAnalyzer
from .aim_classifier import AimClassifier

logger = logging.getLogger(__name__)


@dataclass
class SessionReport:
    """Comprehensive session analysis report.

    Contains all computed metrics, overall composite score, skill level
    classification, and per-engagement details for a single analysis session.

    Attributes:
        session_id: Unique identifier for this session.
        timestamp: ISO-format timestamp when the session was analyzed.
        duration_seconds: Total session duration in seconds.
        total_frames: Total number of frames analyzed.
        aim_accuracy: Aim accuracy metrics dictionary.
        time_to_target: Time-to-target metrics dictionary.
        overshoot: Overshoot analysis metrics dictionary.
        consistency: Consistency metrics dictionary.
        kinematics: Kinematics analysis metrics dictionary.
        aim_classification: Aim classification summary dictionary.
        overall_score: Weighted composite score [0, 1].
        skill_level: Classified skill level string.
        engagement_count: Number of engagement windows analyzed.
        per_engagement_details: List of per-engagement metric dictionaries.
    """

    session_id: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    total_frames: int = 0
    aim_accuracy: Dict[str, Any] = field(default_factory=dict)
    time_to_target: Dict[str, Any] = field(default_factory=dict)
    overshoot: Dict[str, Any] = field(default_factory=dict)
    consistency: Dict[str, Any] = field(default_factory=dict)
    kinematics: Dict[str, Any] = field(default_factory=dict)
    aim_classification: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    skill_level: str = "beginner"
    engagement_count: int = 0
    per_engagement_details: List[Dict[str, Any]] = field(default_factory=list)


class SessionAnalyzer:
    """Orchestrates all metrics into a comprehensive session analysis.

    Runs each metric analyzer on the provided data and aggregates
    results into a SessionReport. Also classifies skill level and
    provides JSON/CSV export.

    Attributes:
        fps: Frames per second of the capture.
        aim_accuracy: AimAccuracy analyzer instance.
        time_to_target: TimeToTarget analyzer instance.
        overshoot: OvershootAnalyzer instance.
        consistency: ConsistencyAnalyzer instance.
        kinematics: KinematicsAnalyzer instance.
        classifier: AimClassifier instance.
    """

    # Weight constants for overall score
    WEIGHT_ACCURACY = 0.30
    WEIGHT_TTT = 0.20
    WEIGHT_OVERSHOOT = 0.15
    WEIGHT_CONSISTENCY = 0.20
    WEIGHT_SMOOTHNESS = 0.15

    def __init__(self, fps: float = 60.0) -> None:
        """Initialize SessionAnalyzer with all sub-analyzers.

        Args:
            fps: Frames per second of the capture for time calculations.
        """
        self.fps: float = fps
        self.aim_accuracy = AimAccuracy()
        self.time_to_target = TimeToTarget(fps=fps)
        self.overshoot = OvershootAnalyzer()
        self.consistency = ConsistencyAnalyzer()
        self.kinematics = KinematicsAnalyzer(fps=fps)
        self.classifier = AimClassifier(fps=fps)

    def analyze(
        self,
        tracker_data: Any,
        trajectory: Any,
        engagement_windows: List[Dict]
    ) -> SessionReport:
        """Run all metrics and aggregate into a SessionReport.

        Extracts position data from tracker_data and trajectory objects,
        runs each metric analyzer, computes the weighted overall score,
        classifies skill level, and returns a comprehensive report.

        Args:
            tracker_data: PositionTracker instance or dict-like object
                providing crosshair and target position histories.
                Expected attributes/keys:
                - crosshair_positions: List of (x, y) positions.
                - target_positions: List of (x, y) positions.
                - movement_segments (optional): List of segmented movements.
            trajectory: Trajectory instance or dict-like object providing
                the smoothed crosshair trajectory.
                Expected attributes/keys:
                - positions: List of (x, y) positions.
            engagement_windows: List of engagement window dictionaries,
                each containing crosshair_positions, target_positions,
                start_frame, end_frame.

        Returns:
            SessionReport dataclass with all computed metrics.
        """
        report = SessionReport()
        report.session_id = str(uuid.uuid4())[:8]
        report.timestamp = datetime.now().isoformat()
        report.engagement_count = len(engagement_windows)

        # Extract position data from tracker_data
        crosshair_positions = self._extract_attr(
            tracker_data, "crosshair_positions", []
        )
        target_positions = self._extract_attr(
            tracker_data, "target_positions", []
        )
        movement_segments = self._extract_attr(
            tracker_data, "movement_segments", []
        )

        # Extract trajectory positions
        trajectory_positions = self._extract_attr(
            trajectory, "positions", []
        )

        # Compute total frames and duration
        report.total_frames = len(crosshair_positions)
        report.duration_seconds = (
            report.total_frames / self.fps if self.fps > 0 else 0.0
        )

        # --- Run individual metrics ---

        # 1. Aim Accuracy
        try:
            if crosshair_positions and target_positions:
                report.aim_accuracy = self.aim_accuracy.compute(
                    crosshair_positions, target_positions
                )
            else:
                report.aim_accuracy = {"accuracy_rate": 0.0}
                logger.warning("No position data for aim accuracy.")
        except Exception as e:
            logger.error("Aim accuracy computation failed: %s", str(e))
            report.aim_accuracy = {"accuracy_rate": 0.0, "error": str(e)}

        # 2. Time-to-Target
        try:
            if engagement_windows:
                report.time_to_target = self.time_to_target.compute(
                    engagement_windows
                )
            else:
                report.time_to_target = {"mean_ttt_ms": 0.0}
                logger.warning("No engagement windows for TTT.")
        except Exception as e:
            logger.error("Time-to-target computation failed: %s", str(e))
            report.time_to_target = {"mean_ttt_ms": 0.0, "error": str(e)}

        # 3. Overshoot
        try:
            if movement_segments and target_positions:
                report.overshoot = self.overshoot.compute(
                    movement_segments, target_positions
                )
            elif engagement_windows:
                # Build movement segments from engagement windows
                segments = [
                    {"positions": ew.get("crosshair_positions", [])}
                    for ew in engagement_windows
                    if ew.get("crosshair_positions")
                ]
                targets = [
                    ew.get("target_positions", [[0, 0]])[-1]
                    for ew in engagement_windows
                    if ew.get("target_positions")
                ]
                if segments and targets:
                    report.overshoot = self.overshoot.compute(
                        segments, targets
                    )
                else:
                    report.overshoot = {"overshoot_ratio": 0.0}
            else:
                report.overshoot = {"overshoot_ratio": 0.0}
        except Exception as e:
            logger.error("Overshoot computation failed: %s", str(e))
            report.overshoot = {"overshoot_ratio": 0.0, "error": str(e)}

        # 4. Consistency
        try:
            if crosshair_positions:
                report.consistency = self.consistency.compute(
                    crosshair_positions,
                    target_positions if target_positions else None
                )
            else:
                report.consistency = {"consistency_score": 0.0}
        except Exception as e:
            logger.error("Consistency computation failed: %s", str(e))
            report.consistency = {"consistency_score": 0.0, "error": str(e)}

        # 5. Kinematics
        try:
            traj_data = (
                trajectory_positions if trajectory_positions
                else crosshair_positions
            )
            if traj_data and len(traj_data) >= 3:
                report.kinematics = self.kinematics.compute(traj_data)
            else:
                report.kinematics = self.kinematics._empty_result()
        except Exception as e:
            logger.error("Kinematics computation failed: %s", str(e))
            report.kinematics = {"ldlj": 0.0, "sparc": 0.0, "error": str(e)}

        # 6. Aim Classification
        try:
            if movement_segments:
                classification_results = self.classifier.classify_batch(
                    movement_segments
                )
            elif engagement_windows:
                segments_for_class = [
                    {"positions": ew.get("crosshair_positions", []),
                     "target_positions": ew.get("target_positions", [])}
                    for ew in engagement_windows
                ]
                classification_results = self.classifier.classify_batch(
                    segments_for_class
                )
            else:
                classification_results = []

            report.aim_classification = self._summarize_classifications(
                classification_results
            )
        except Exception as e:
            logger.error("Aim classification failed: %s", str(e))
            report.aim_classification = {"error": str(e)}

        # 7. Per-engagement details
        try:
            if engagement_windows:
                report.per_engagement_details = (
                    self.aim_accuracy.compute_per_engagement(
                        engagement_windows
                    )
                )
        except Exception as e:
            logger.error(
                "Per-engagement details computation failed: %s", str(e)
            )

        # --- Compute overall score ---
        report.overall_score = self._compute_overall_score(report)

        # --- Classify skill level ---
        report.skill_level = self.classify_skill_level(report)

        logger.info(
            "Session analysis complete: score=%.3f, skill=%s, "
            "%d engagements, %d frames",
            report.overall_score, report.skill_level,
            report.engagement_count, report.total_frames
        )

        return report

    def _extract_attr(self, obj: Any, attr: str, default: Any) -> Any:
        """Extract attribute from an object or dictionary.

        Supports both attribute access (for dataclass/object instances)
        and dictionary key access.

        Args:
            obj: Source object or dictionary.
            attr: Attribute name or dictionary key.
            default: Default value if attribute is not found.

        Returns:
            Extracted value or default.
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def _summarize_classifications(
        self,
        classification_results: List[Dict]
    ) -> Dict[str, Any]:
        """Summarize batch classification results.

        Args:
            classification_results: List of classification result dicts.

        Returns:
            Summary dictionary with counts and ratios per type.
        """
        total = len(classification_results)
        if total == 0:
            return {
                "total_segments": 0,
                "flick_count": 0,
                "tracking_count": 0,
                "hybrid_count": 0,
                "flick_ratio": 0.0,
                "tracking_ratio": 0.0,
                "hybrid_ratio": 0.0,
            }

        flick_count = sum(
            1 for r in classification_results
            if r.get("classification") == AimClassifier.FLICK
        )
        tracking_count = sum(
            1 for r in classification_results
            if r.get("classification") == AimClassifier.TRACKING
        )
        hybrid_count = total - flick_count - tracking_count

        return {
            "total_segments": total,
            "flick_count": flick_count,
            "tracking_count": tracking_count,
            "hybrid_count": hybrid_count,
            "flick_ratio": flick_count / total,
            "tracking_ratio": tracking_count / total,
            "hybrid_ratio": hybrid_count / total,
        }

    def _compute_overall_score(self, report: SessionReport) -> float:
        """Compute weighted composite overall score.

        Score components (all normalized to [0, 1]):
          - accuracy:  aim_accuracy.accuracy_rate
          - ttt_score: normalized from TTT (lower TTT = higher score)
          - overshoot_score: 1 - overshoot_ratio (fewer overshoots = better)
          - consistency: consistency_score
          - smoothness: normalized from SPARC (closer to 0 = smoother)

        Weights:
          accuracy * 0.30 + ttt_score * 0.20 + overshoot_score * 0.15
          + consistency * 0.20 + smoothness * 0.15

        Args:
            report: SessionReport with individual metric results.

        Returns:
            Overall score in [0, 1].
        """
        # Accuracy component (already [0, 1])
        accuracy = report.aim_accuracy.get("accuracy_rate", 0.0)

        # TTT component: normalize using skill level thresholds
        # Map TTT from [0, 1000ms] to [1, 0] (lower TTT = better)
        mean_ttt = report.time_to_target.get("mean_ttt_ms", 1000.0)
        ttt_score = max(0.0, 1.0 - (mean_ttt / 1000.0))

        # Overshoot component: 1 - overshoot_ratio (fewer = better)
        overshoot_ratio = report.overshoot.get("overshoot_ratio", 0.0)
        overshoot_score = 1.0 - overshoot_ratio

        # Consistency component (already [0, 1])
        consistency = report.consistency.get("consistency_score", 0.0)

        # Smoothness component from SPARC
        # SPARC is typically in range [-7, 0], closer to 0 = smoother
        sparc = report.kinematics.get("sparc", -7.0)
        # Normalize: map [-7, 0] to [0, 1]
        smoothness = max(0.0, min(1.0, (sparc + 7.0) / 7.0))

        # Weighted composite
        overall = (
            self.WEIGHT_ACCURACY * accuracy
            + self.WEIGHT_TTT * ttt_score
            + self.WEIGHT_OVERSHOOT * overshoot_score
            + self.WEIGHT_CONSISTENCY * consistency
            + self.WEIGHT_SMOOTHNESS * smoothness
        )

        return float(np.clip(overall, 0.0, 1.0))

    def classify_skill_level(self, report: SessionReport) -> str:
        """Classify the player's skill level based on metric thresholds.

        Compares session metrics against thresholds defined in
        settings.SKILL_LEVELS to determine if the player is 'beginner',
        'intermediate', or 'advanced'.

        Uses a point-based system: each metric within a skill level's
        range adds a point for that level. The level with the most
        points wins.

        Args:
            report: SessionReport with computed metrics.

        Returns:
            Skill level string: 'beginner', 'intermediate', or 'advanced'.
        """
        accuracy_pct = report.aim_accuracy.get("accuracy_rate", 0.0) * 100
        mean_ttt = report.time_to_target.get("mean_ttt_ms", 1000.0)
        overshoot = report.overshoot.get("overshoot_ratio", 0.5)
        consistency = report.consistency.get("consistency_score", 0.0)

        scores: Dict[str, int] = {"beginner": 0, "intermediate": 0, "advanced": 0}

        for level, thresholds in settings.SKILL_LEVELS.items():
            # Aim accuracy check
            acc_low, acc_high = thresholds["aim_accuracy"]
            if acc_low <= accuracy_pct <= acc_high:
                scores[level] += 1

            # TTT check
            ttt_low, ttt_high = thresholds["ttt_ms"]
            if ttt_low <= mean_ttt <= ttt_high:
                scores[level] += 1

            # Overshoot check
            os_low, os_high = thresholds["overshoot_ratio"]
            if os_low <= overshoot <= os_high:
                scores[level] += 1

            # Consistency check
            con_low, con_high = thresholds["consistency"]
            if con_low <= consistency <= con_high:
                scores[level] += 1

        # Return level with highest score (prefer higher skill on ties)
        best_level = max(
            scores,
            key=lambda level: (
                scores[level],
                {"advanced": 2, "intermediate": 1, "beginner": 0}[level]
            )
        )

        logger.info(
            "Skill classification: %s (scores: %s)", best_level, scores
        )

        return best_level

    def export_json(
        self,
        report: SessionReport,
        filepath: Union[str, Path]
    ) -> None:
        """Export SessionReport to a JSON file.

        Converts the dataclass to a dictionary and writes it as formatted
        JSON. Creates parent directories if they don't exist.

        Args:
            report: SessionReport to export.
            filepath: Output file path for the JSON file.

        Raises:
            IOError: If the file cannot be written.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            report_dict = asdict(report)
            # Convert any numpy types to native Python types
            report_dict = self._convert_numpy_types(report_dict)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)

            logger.info("Session report exported to JSON: %s", filepath)

        except Exception as e:
            logger.error("Failed to export JSON report: %s", str(e))
            raise

    def export_csv(
        self,
        report: SessionReport,
        filepath: Union[str, Path]
    ) -> None:
        """Export SessionReport to a CSV file.

        Flattens the report dictionary into key-value pairs and writes
        them as rows in a CSV file. Nested dictionaries are flattened
        with dot-separated keys (e.g., 'aim_accuracy.accuracy_rate').

        Args:
            report: SessionReport to export.
            filepath: Output file path for the CSV file.

        Raises:
            IOError: If the file cannot be written.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            report_dict = asdict(report)
            report_dict = self._convert_numpy_types(report_dict)
            flat = self._flatten_dict(report_dict)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for key, value in sorted(flat.items()):
                    writer.writerow([key, value])

            logger.info("Session report exported to CSV: %s", filepath)

        except Exception as e:
            logger.error("Failed to export CSV report: %s", str(e))
            raise

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary into dot-separated keys.

        Args:
            d: Dictionary to flatten.
            parent_key: Prefix for nested keys.
            sep: Separator between parent and child keys.

        Returns:
            Flattened dictionary with dot-separated keys.
        """
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    SessionAnalyzer._flatten_dict(v, new_key, sep).items()
                )
            elif isinstance(v, list):
                # Store lists as JSON strings for CSV compatibility
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types.

        Ensures JSON serialization compatibility by converting numpy
        integers, floats, and arrays to their Python equivalents.

        Args:
            obj: Object to convert (can be dict, list, or scalar).

        Returns:
            Converted object with native Python types.
        """
        if isinstance(obj, dict):
            return {k: SessionAnalyzer._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SessionAnalyzer._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
