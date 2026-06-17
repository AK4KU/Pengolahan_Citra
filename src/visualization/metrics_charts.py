"""
Metrics Chart Visualization Module for FPS Aim Performance Analyzer.

Provides a comprehensive set of interactive Plotly charts for displaying
aim performance metrics including gauges, velocity profiles, distributions,
radar charts, timelines, kinematics panels, Fitts' Law plots, and
overshoot visualizations.

All charts use a dark theme (#0F172A background) with the color palette
from settings.DASHBOARD_THEME.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class MetricsChartBuilder:
    """Builds interactive metric visualization charts using Plotly.

    Creates a variety of chart types for displaying aim performance
    metrics, all styled with a consistent dark theme.

    Attributes:
        theme: Dashboard theme dictionary from settings.
    """

    def __init__(self) -> None:
        """Initialize MetricsChartBuilder with theme settings."""
        self.theme: Dict[str, str] = settings.DASHBOARD_THEME

    def _get_dark_layout(self, title: str = "", width: int = 700,
                         height: int = 450) -> Dict[str, Any]:
        """Create a standard dark-themed Plotly layout.

        Args:
            title: Chart title text.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            Dictionary of Plotly layout properties.
        """
        return dict(
            title=dict(
                text=title,
                font=dict(color=self.theme["text_primary"], size=18),
                x=0.5,
            ),
            paper_bgcolor=self.theme["background_dark"],
            plot_bgcolor=self.theme["background_dark"],
            font=dict(color=self.theme["text_primary"]),
            width=width,
            height=height,
            margin=dict(l=60, r=30, t=60, b=60),
            xaxis=dict(
                gridcolor=self.theme["background_card"],
                zerolinecolor=self.theme["background_card"],
                showgrid=True,
                gridwidth=1,
            ),
            yaxis=dict(
                gridcolor=self.theme["background_card"],
                zerolinecolor=self.theme["background_card"],
                showgrid=True,
                gridwidth=1,
            ),
        )

    # ------------------------------------------------------------------
    # 1. Accuracy Gauge
    # ------------------------------------------------------------------
    def create_accuracy_gauge(
        self,
        accuracy_rate: float,
    ) -> go.Figure:
        """Create a gauge/indicator chart for aim accuracy.

        Displays accuracy as a gauge with color-coded regions:
        green (>60%), yellow (30-60%), red (<30%).

        Args:
            accuracy_rate: Accuracy percentage value (0-100).

        Returns:
            Plotly Figure with gauge indicator.
        """
        try:
            accuracy_rate = max(0.0, min(100.0, accuracy_rate))

            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=accuracy_rate,
                number=dict(
                    suffix="%",
                    font=dict(color=self.theme["text_primary"], size=40),
                ),
                title=dict(
                    text="Aim Accuracy",
                    font=dict(color=self.theme["text_primary"], size=18),
                ),
                delta=dict(
                    reference=50,
                    increasing=dict(color=self.theme["success_color"]),
                    decreasing=dict(color=self.theme["danger_color"]),
                    font=dict(size=14),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickwidth=1,
                        tickcolor=self.theme["text_secondary"],
                        tickfont=dict(
                            color=self.theme["text_secondary"]),
                        dtick=10,
                    ),
                    bar=dict(
                        color=self.theme["primary_color"],
                        thickness=0.7,
                    ),
                    bgcolor=self.theme["background_card"],
                    borderwidth=2,
                    bordercolor=self.theme["background_card"],
                    steps=[
                        dict(range=[0, 30],
                             color=self.theme["danger_color"] + "44"),
                        dict(range=[30, 60],
                             color=self.theme["accent_color"] + "44"),
                        dict(range=[60, 100],
                             color=self.theme["success_color"] + "44"),
                    ],
                    threshold=dict(
                        line=dict(
                            color="white", width=3),
                        thickness=0.8,
                        value=accuracy_rate,
                    ),
                ),
            ))

            layout = self._get_dark_layout(
                title="", width=450, height=350)
            # Remove axes for gauge
            layout.pop("xaxis", None)
            layout.pop("yaxis", None)
            fig.update_layout(**layout)

            logger.info(
                "Created accuracy gauge: %.1f%%", accuracy_rate)
            return fig

        except Exception as e:
            logger.error("Failed to create accuracy gauge: %s", e)
            raise

    # ------------------------------------------------------------------
    # 2. Velocity Profile
    # ------------------------------------------------------------------
    def create_velocity_profile(
        self,
        trajectory: np.ndarray,
    ) -> go.Figure:
        """Create line chart of crosshair speed over time.

        Plots velocity magnitude with peak velocities highlighted
        as markers and a mean velocity reference line.

        Args:
            trajectory: Array of shape (N, 2+) with columns [x, y, ...].
                Optional 3rd column for timestamps (seconds).

        Returns:
            Plotly Figure with velocity profile line chart.
        """
        try:
            n_points = len(trajectory)
            if n_points < 2:
                logger.warning("Trajectory too short for velocity profile")
                fig = go.Figure()
                fig.update_layout(**self._get_dark_layout(
                    title="Velocity Profile (Insufficient Data)"))
                return fig

            positions = trajectory[:, :2].astype(float)
            displacements = np.diff(positions, axis=0)
            distances = np.linalg.norm(displacements, axis=1)

            # Determine time axis
            if trajectory.shape[1] >= 3:
                timestamps = trajectory[:, 2].astype(float)
                dt = np.diff(timestamps)
                dt = np.where(dt > 0, dt, 1e-6)
                velocities = distances / dt
                time_axis = timestamps[1:]
                x_label = "Time (s)"
            else:
                velocities = distances
                time_axis = np.arange(1, n_points)
                x_label = "Frame"

            mean_vel = float(np.mean(velocities))
            std_vel = float(np.std(velocities))

            # Detect peaks (velocities > mean + 1.5 * std)
            peak_threshold = mean_vel + 1.5 * std_vel
            peak_mask = velocities > peak_threshold
            peak_indices = np.where(peak_mask)[0]

            fig = go.Figure()

            # Main velocity line
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=velocities,
                mode="lines",
                line=dict(
                    color=self.theme["secondary_color"],
                    width=1.5,
                ),
                name="Velocity",
                fill="tozeroy",
                fillcolor=self.theme["secondary_color"] + "22",
                hovertemplate=(
                    f"{x_label}: %{{x:.2f}}<br>"
                    "Velocity: %{y:.1f} px/s<extra></extra>"
                ),
            ))

            # Peak markers
            if len(peak_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=time_axis[peak_indices],
                    y=velocities[peak_indices],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=self.theme["danger_color"],
                        symbol="triangle-up",
                        line=dict(color="white", width=1),
                    ),
                    name="Peaks",
                    hovertemplate=(
                        "Peak Velocity: %{y:.1f} px/s<extra></extra>"
                    ),
                ))

            # Mean velocity reference line
            fig.add_hline(
                y=mean_vel,
                line=dict(
                    color=self.theme["accent_color"],
                    width=1.5,
                    dash="dash",
                ),
                annotation=dict(
                    text=f"Mean: {mean_vel:.1f}",
                    font=dict(color=self.theme["accent_color"], size=11),
                    bgcolor=self.theme["background_card"],
                ),
            )

            layout = self._get_dark_layout(
                title="Crosshair Velocity Profile",
                width=800,
                height=400,
            )
            layout.update(
                xaxis=dict(**layout["xaxis"], title=x_label),
                yaxis=dict(**layout["yaxis"], title="Velocity (px/s)"),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info("Created velocity profile with %d points", n_points)
            return fig

        except Exception as e:
            logger.error("Failed to create velocity profile: %s", e)
            raise

    # ------------------------------------------------------------------
    # 3. TTT Distribution
    # ------------------------------------------------------------------
    def create_ttt_distribution(
        self,
        ttt_values: List[float],
    ) -> go.Figure:
        """Create histogram of time-to-target values.

        Shows the distribution of TTT with mean and median vertical
        lines for reference.

        Args:
            ttt_values: List of time-to-target values in milliseconds.

        Returns:
            Plotly Figure with TTT histogram and statistical markers.
        """
        try:
            if not ttt_values:
                logger.warning("No TTT values provided")
                fig = go.Figure()
                fig.update_layout(**self._get_dark_layout(
                    title="TTT Distribution (No Data)"))
                fig.add_annotation(
                    text="No TTT data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    font=dict(
                        color=self.theme["text_secondary"], size=16),
                    showarrow=False,
                )
                return fig

            ttt_arr = np.array(ttt_values, dtype=float)
            mean_ttt = float(np.mean(ttt_arr))
            median_ttt = float(np.median(ttt_arr))

            fig = go.Figure()

            # Histogram
            fig.add_trace(go.Histogram(
                x=ttt_arr,
                nbinsx=min(30, max(10, len(ttt_arr) // 3)),
                marker=dict(
                    color=self.theme["primary_color"],
                    line=dict(
                        color=self.theme["background_dark"], width=1),
                ),
                opacity=0.85,
                name="TTT Distribution",
                hovertemplate="TTT: %{x:.0f} ms<br>Count: %{y}"
                              "<extra></extra>",
            ))

            # Mean line
            fig.add_vline(
                x=mean_ttt,
                line=dict(
                    color=self.theme["accent_color"],
                    width=2,
                    dash="dash",
                ),
                annotation=dict(
                    text=f"Mean: {mean_ttt:.0f} ms",
                    font=dict(color=self.theme["accent_color"], size=11),
                    bgcolor=self.theme["background_card"],
                    yshift=10,
                ),
            )

            # Median line
            fig.add_vline(
                x=median_ttt,
                line=dict(
                    color=self.theme["secondary_color"],
                    width=2,
                    dash="dot",
                ),
                annotation=dict(
                    text=f"Median: {median_ttt:.0f} ms",
                    font=dict(
                        color=self.theme["secondary_color"], size=11),
                    bgcolor=self.theme["background_card"],
                    yshift=-10,
                ),
            )

            layout = self._get_dark_layout(
                title="Time-to-Target Distribution",
                width=700,
                height=400,
            )
            layout.update(
                xaxis=dict(**layout["xaxis"], title="Time-to-Target (ms)"),
                yaxis=dict(**layout["yaxis"], title="Frequency"),
                bargap=0.05,
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info(
                "Created TTT distribution with %d values (mean=%.0f ms)",
                len(ttt_values), mean_ttt,
            )
            return fig

        except Exception as e:
            logger.error("Failed to create TTT distribution: %s", e)
            raise

    # ------------------------------------------------------------------
    # 4. Radar Chart
    # ------------------------------------------------------------------
    def create_radar_chart(
        self,
        metrics_dict: Dict[str, float],
    ) -> go.Figure:
        """Create radar/spider chart with 5 performance axes.

        Displays Accuracy, Speed, Consistency, Smoothness, and Precision
        on a polar chart for a single player/session.

        Args:
            metrics_dict: Dictionary with keys:
                - 'Accuracy' (float): 0-100 scale.
                - 'Speed' (float): 0-100 scale.
                - 'Consistency' (float): 0-100 scale.
                - 'Smoothness' (float): 0-100 scale.
                - 'Precision' (float): 0-100 scale.

        Returns:
            Plotly Figure with radar/spider chart.
        """
        try:
            categories = [
                "Accuracy", "Speed", "Consistency",
                "Smoothness", "Precision",
            ]
            values = [
                min(100, max(0, metrics_dict.get(cat, 0)))
                for cat in categories
            ]
            # Close the polygon
            values.append(values[0])
            categories_closed = categories + [categories[0]]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill="toself",
                fillcolor=self.theme["primary_color"] + "33",
                line=dict(
                    color=self.theme["primary_color"], width=2),
                marker=dict(
                    size=8,
                    color=self.theme["primary_color"],
                ),
                name="Performance",
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            ))

            layout = self._get_dark_layout(
                title="Performance Radar",
                width=600,
                height=500,
            )
            layout.update(
                polar=dict(
                    bgcolor=self.theme["background_dark"],
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        gridcolor=self.theme["background_card"],
                        linecolor=self.theme["background_card"],
                        tickfont=dict(
                            color=self.theme["text_secondary"]),
                        ticksuffix="",
                    ),
                    angularaxis=dict(
                        gridcolor=self.theme["background_card"],
                        linecolor=self.theme["background_card"],
                        tickfont=dict(
                            color=self.theme["text_primary"], size=13),
                    ),
                ),
                showlegend=False,
            )
            # Remove standard axes (not used for polar)
            layout.pop("xaxis", None)
            layout.pop("yaxis", None)
            fig.update_layout(**layout)

            logger.info("Created radar chart with metrics: %s", metrics_dict)
            return fig

        except Exception as e:
            logger.error("Failed to create radar chart: %s", e)
            raise

    # ------------------------------------------------------------------
    # 5. Engagement Timeline
    # ------------------------------------------------------------------
    def create_engagement_timeline(
        self,
        engagement_windows: List[Dict[str, Any]],
        metrics_per_engagement: List[Dict[str, Any]],
    ) -> go.Figure:
        """Create timeline showing each engagement with color-coded performance.

        Displays engagements as horizontal bars on a timeline, colored
        by their performance score (accuracy/TTT).

        Args:
            engagement_windows: List of engagement dictionaries with:
                - 'start_time' (float): Start time in seconds.
                - 'end_time' (float): End time in seconds.
                - 'hit' (bool, optional): Whether it was a hit.
                - 'label' (str, optional): Engagement label.
            metrics_per_engagement: List of metric dictionaries with:
                - 'ttt_ms' (float, optional): Time-to-target in ms.
                - 'accuracy' (float, optional): Engagement accuracy (0-100).
                - 'overshoot' (float, optional): Overshoot magnitude.

        Returns:
            Plotly Figure with engagement timeline.
        """
        try:
            if not engagement_windows:
                logger.warning("No engagement windows for timeline")
                fig = go.Figure()
                fig.update_layout(**self._get_dark_layout(
                    title="Engagement Timeline (No Data)"))
                return fig

            fig = go.Figure()

            for i, (ew, metrics) in enumerate(
                zip(engagement_windows, metrics_per_engagement)
            ):
                start = ew.get("start_time", i)
                end = ew.get("end_time", start + 0.5)
                hit = ew.get("hit", None)
                label = ew.get("label", f"E{i + 1}")
                ttt = metrics.get("ttt_ms", None)

                # Color by hit/miss
                if hit is True:
                    color = self.theme["success_color"]
                elif hit is False:
                    color = self.theme["danger_color"]
                else:
                    color = self.theme["text_secondary"]

                # Hover text
                hover_parts = [f"<b>{label}</b>"]
                hover_parts.append(
                    f"Duration: {(end - start) * 1000:.0f} ms")
                if hit is not None:
                    hover_parts.append(
                        f"Result: {'Hit' if hit else 'Miss'}")
                if ttt is not None:
                    hover_parts.append(f"TTT: {ttt:.0f} ms")

                fig.add_trace(go.Bar(
                    x=[end - start],
                    y=[0.5],
                    base=[start],
                    orientation="h",
                    marker=dict(
                        color=color,
                        line=dict(
                            color=self.theme["background_dark"],
                            width=1,
                        ),
                    ),
                    name=label,
                    showlegend=False,
                    hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
                    text=label,
                    textposition="inside",
                    textfont=dict(color="white", size=9),
                ))

            # Summary statistics bar at top
            n_total = len(engagement_windows)
            n_hits = sum(1 for ew in engagement_windows
                         if ew.get("hit") is True)
            n_misses = sum(1 for ew in engagement_windows
                          if ew.get("hit") is False)

            layout = self._get_dark_layout(
                title=(f"Engagement Timeline — {n_total} engagements "
                       f"({n_hits} hits, {n_misses} misses)"),
                width=1000,
                height=250,
            )
            layout.update(
                xaxis=dict(
                    **layout["xaxis"],
                    title="Time (s)",
                ),
                yaxis=dict(
                    visible=False,
                    range=[0, 1],
                ),
                barmode="stack",
                showlegend=False,
            )
            fig.update_layout(**layout)

            logger.info(
                "Created engagement timeline with %d engagements",
                n_total,
            )
            return fig

        except Exception as e:
            logger.error("Failed to create engagement timeline: %s", e)
            raise

    # ------------------------------------------------------------------
    # 6. Kinematics Panel
    # ------------------------------------------------------------------
    def create_kinematics_panel(
        self,
        trajectory: np.ndarray,
    ) -> go.Figure:
        """Create subplots for position, velocity, acceleration, and jerk.

        Displays four vertically stacked time-series plots showing
        the kinematic profile of crosshair movement.

        Args:
            trajectory: Array of shape (N, 2+) with columns [x, y, ...].
                Optional 3rd column for timestamps (seconds).

        Returns:
            Plotly Figure with 4-row subplot panel.
        """
        try:
            n_points = len(trajectory)
            positions = trajectory[:, :2].astype(float)

            # Time axis
            if trajectory.shape[1] >= 3:
                timestamps = trajectory[:, 2].astype(float)
                time_axis = timestamps
                x_label = "Time (s)"
            else:
                time_axis = np.arange(n_points, dtype=float)
                x_label = "Frame"

            # Position magnitude (distance from screen center)
            center = np.array([
                settings.CROSSHAIR_DEFAULT_X,
                settings.CROSSHAIR_DEFAULT_Y,
            ], dtype=float)
            pos_magnitude = np.linalg.norm(positions - center, axis=1)

            # Velocity
            if n_points >= 2:
                dp = np.diff(positions, axis=0)
                dt_arr = np.diff(time_axis)
                dt_arr = np.where(dt_arr > 0, dt_arr, 1e-6)
                velocity_vec = dp / dt_arr[:, np.newaxis]
                velocity_mag = np.linalg.norm(velocity_vec, axis=1)
                vel_time = time_axis[1:]
            else:
                velocity_mag = np.array([0.0])
                vel_time = time_axis[:1]

            # Acceleration
            if len(velocity_mag) >= 2:
                dv = np.diff(velocity_mag)
                dt_vel = np.diff(vel_time)
                dt_vel = np.where(dt_vel > 0, dt_vel, 1e-6)
                acceleration = dv / dt_vel
                acc_time = vel_time[1:]
            else:
                acceleration = np.array([0.0])
                acc_time = vel_time[:1]

            # Jerk
            if len(acceleration) >= 2:
                da = np.diff(acceleration)
                dt_acc = np.diff(acc_time)
                dt_acc = np.where(dt_acc > 0, dt_acc, 1e-6)
                jerk = da / dt_acc
                jerk_time = acc_time[1:]
            else:
                jerk = np.array([0.0])
                jerk_time = acc_time[:1]

            # Create 4-row subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    "Position (Distance from Center)",
                    "Velocity",
                    "Acceleration",
                    "Jerk",
                ),
            )

            # Row 1: Position
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=pos_magnitude,
                mode="lines",
                line=dict(
                    color=self.theme["primary_color"], width=1.5),
                name="Position",
                hovertemplate="%{y:.1f} px<extra>Position</extra>",
            ), row=1, col=1)

            # Row 2: Velocity
            fig.add_trace(go.Scatter(
                x=vel_time,
                y=velocity_mag,
                mode="lines",
                line=dict(
                    color=self.theme["secondary_color"], width=1.5),
                name="Velocity",
                fill="tozeroy",
                fillcolor=self.theme["secondary_color"] + "22",
                hovertemplate="%{y:.1f} px/s<extra>Velocity</extra>",
            ), row=2, col=1)

            # Row 3: Acceleration
            fig.add_trace(go.Scatter(
                x=acc_time,
                y=acceleration,
                mode="lines",
                line=dict(
                    color=self.theme["accent_color"], width=1.5),
                name="Acceleration",
                hovertemplate="%{y:.1f} px/s²<extra>Acceleration</extra>",
            ), row=3, col=1)

            # Row 4: Jerk
            fig.add_trace(go.Scatter(
                x=jerk_time,
                y=jerk,
                mode="lines",
                line=dict(
                    color=self.theme["danger_color"], width=1.5),
                name="Jerk",
                hovertemplate="%{y:.1f} px/s³<extra>Jerk</extra>",
            ), row=4, col=1)

            # Apply dark theme to all subplots
            fig.update_layout(
                title=dict(
                    text="Kinematics Panel",
                    font=dict(
                        color=self.theme["text_primary"], size=18),
                    x=0.5,
                ),
                paper_bgcolor=self.theme["background_dark"],
                plot_bgcolor=self.theme["background_dark"],
                font=dict(color=self.theme["text_primary"]),
                width=900,
                height=800,
                margin=dict(l=70, r=30, t=80, b=60),
                showlegend=False,
            )

            # Style all axes
            for i in range(1, 5):
                fig.update_xaxes(
                    gridcolor=self.theme["background_card"],
                    zerolinecolor=self.theme["background_card"],
                    row=i, col=1,
                )
                fig.update_yaxes(
                    gridcolor=self.theme["background_card"],
                    zerolinecolor=self.theme["background_card"],
                    row=i, col=1,
                )

            # Y-axis labels
            fig.update_yaxes(title_text="Distance (px)", row=1, col=1)
            fig.update_yaxes(title_text="px/s", row=2, col=1)
            fig.update_yaxes(title_text="px/s²", row=3, col=1)
            fig.update_yaxes(title_text="px/s³", row=4, col=1)
            fig.update_xaxes(title_text=x_label, row=4, col=1)

            # Style subplot titles
            for annotation in fig.layout.annotations:
                annotation.font = dict(
                    color=self.theme["text_secondary"], size=12)

            logger.info(
                "Created kinematics panel with %d data points", n_points)
            return fig

        except Exception as e:
            logger.error("Failed to create kinematics panel: %s", e)
            raise

    # ------------------------------------------------------------------
    # 7. Fitts' Law Plot
    # ------------------------------------------------------------------
    def create_fitts_law_plot(
        self,
        distances: List[float],
        times: List[float],
        target_widths: List[float],
        a: float,
        b: float,
    ) -> go.Figure:
        """Create Fitts' Law scatter plot with regression line.

        Plots movement time vs. index of difficulty (ID = log2(2D/W))
        with the linear regression line MT = a + b * ID.

        Args:
            distances: List of movement distances in pixels.
            times: List of movement times in milliseconds.
            target_widths: List of target widths in pixels.
            a: Fitts' Law intercept parameter (ms).
            b: Fitts' Law slope parameter (ms/bit).

        Returns:
            Plotly Figure with scatter plot and regression line.
        """
        try:
            distances_arr = np.array(distances, dtype=float)
            times_arr = np.array(times, dtype=float)
            widths_arr = np.array(target_widths, dtype=float)

            # Avoid division by zero and log of non-positive
            widths_arr = np.maximum(widths_arr, 1.0)
            ratio = 2.0 * distances_arr / widths_arr
            ratio = np.maximum(ratio, 1.0)

            # Index of Difficulty (Shannon formulation)
            id_values = np.log2(ratio + 1)

            fig = go.Figure()

            # Scatter points
            fig.add_trace(go.Scatter(
                x=id_values,
                y=times_arr,
                mode="markers",
                marker=dict(
                    size=8,
                    color=self.theme["primary_color"],
                    opacity=0.7,
                    line=dict(color="white", width=0.5),
                ),
                name="Observations",
                hovertemplate=(
                    "ID: %{x:.2f} bits<br>"
                    "MT: %{y:.0f} ms<extra></extra>"
                ),
            ))

            # Regression line: MT = a + b * ID
            id_range = np.linspace(
                float(np.min(id_values)) - 0.2,
                float(np.max(id_values)) + 0.2,
                100,
            )
            mt_predicted = a + b * id_range

            fig.add_trace(go.Scatter(
                x=id_range,
                y=mt_predicted,
                mode="lines",
                line=dict(
                    color=self.theme["accent_color"],
                    width=2.5,
                    dash="solid",
                ),
                name=f"Fitts' Law: MT = {a:.0f} + {b:.0f} × ID",
                hovertemplate=(
                    "ID: %{x:.2f} bits<br>"
                    "Predicted MT: %{y:.0f} ms<extra></extra>"
                ),
            ))

            # R² annotation
            if len(id_values) > 2:
                ss_res = np.sum((times_arr - (a + b * id_values)) ** 2)
                ss_tot = np.sum((times_arr - np.mean(times_arr)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            else:
                r_squared = 0.0

            fig.add_annotation(
                text=(f"MT = {a:.0f} + {b:.0f} × ID<br>"
                      f"R² = {r_squared:.3f}"),
                xref="paper", yref="paper",
                x=0.05, y=0.95,
                font=dict(color=self.theme["text_primary"], size=13),
                bgcolor=self.theme["background_card"],
                bordercolor=self.theme["primary_color"],
                borderwidth=1,
                borderpad=6,
                showarrow=False,
                align="left",
            )

            layout = self._get_dark_layout(
                title="Fitts' Law Analysis",
                width=700,
                height=500,
            )
            layout.update(
                xaxis=dict(
                    **layout["xaxis"],
                    title="Index of Difficulty (bits)",
                ),
                yaxis=dict(
                    **layout["yaxis"],
                    title="Movement Time (ms)",
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info(
                "Created Fitts' Law plot with %d observations (R²=%.3f)",
                len(distances), r_squared,
            )
            return fig

        except Exception as e:
            logger.error("Failed to create Fitts' Law plot: %s", e)
            raise

    # ------------------------------------------------------------------
    # 8. Overshoot Chart
    # ------------------------------------------------------------------
    def create_overshoot_chart(
        self,
        overshoot_data: Dict[str, Any],
    ) -> go.Figure:
        """Create bar chart of overshoot magnitudes per engagement.

        Displays the overshoot distance for each engagement as a
        vertical bar, with color intensity indicating severity.

        Args:
            overshoot_data: Dictionary containing:
                - 'labels' (List[str]): Engagement labels.
                - 'magnitudes' (List[float]): Overshoot magnitude in px.
                - 'directions' (List[str], optional): Overshoot direction
                    ('left', 'right', 'up', 'down').

        Returns:
            Plotly Figure with overshoot bar chart.
        """
        try:
            labels = overshoot_data.get("labels", [])
            magnitudes = overshoot_data.get("magnitudes", [])
            directions = overshoot_data.get("directions", [])

            if not labels or not magnitudes:
                logger.warning("No overshoot data provided")
                fig = go.Figure()
                fig.update_layout(**self._get_dark_layout(
                    title="Overshoot Magnitudes (No Data)"))
                fig.add_annotation(
                    text="No overshoot data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    font=dict(
                        color=self.theme["text_secondary"], size=16),
                    showarrow=False,
                )
                return fig

            mag_arr = np.array(magnitudes, dtype=float)
            max_mag = float(np.max(mag_arr)) if len(mag_arr) > 0 else 1.0
            mean_mag = float(np.mean(mag_arr))

            # Color by magnitude (gradient from green to red)
            normalized = mag_arr / max_mag if max_mag > 0 else mag_arr
            colors = []
            for n in normalized:
                if n < 0.33:
                    colors.append(self.theme["success_color"])
                elif n < 0.66:
                    colors.append(self.theme["accent_color"])
                else:
                    colors.append(self.theme["danger_color"])

            # Hover text with direction if available
            hover_texts = []
            for i, (lab, mag) in enumerate(zip(labels, magnitudes)):
                text = f"<b>{lab}</b><br>Overshoot: {mag:.1f} px"
                if i < len(directions) and directions[i]:
                    text += f"<br>Direction: {directions[i]}"
                hover_texts.append(text)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=labels,
                y=mag_arr,
                marker=dict(
                    color=colors,
                    line=dict(
                        color=self.theme["background_dark"], width=1),
                ),
                text=[f"{m:.1f}" for m in mag_arr],
                textposition="outside",
                textfont=dict(
                    color=self.theme["text_primary"], size=10),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
                name="Overshoot",
            ))

            # Mean overshoot reference line
            fig.add_hline(
                y=mean_mag,
                line=dict(
                    color=self.theme["secondary_color"],
                    width=1.5,
                    dash="dash",
                ),
                annotation=dict(
                    text=f"Mean: {mean_mag:.1f} px",
                    font=dict(
                        color=self.theme["secondary_color"], size=11),
                    bgcolor=self.theme["background_card"],
                ),
            )

            # Overshoot threshold reference
            threshold = settings.OVERSHOOT_REVERSAL_THRESHOLD
            fig.add_hline(
                y=threshold,
                line=dict(
                    color=self.theme["text_secondary"],
                    width=1,
                    dash="dot",
                ),
                annotation=dict(
                    text=f"Threshold: {threshold} px",
                    font=dict(
                        color=self.theme["text_secondary"], size=10),
                    bgcolor=self.theme["background_card"],
                    xanchor="right",
                ),
            )

            layout = self._get_dark_layout(
                title="Overshoot Magnitudes per Engagement",
                width=max(600, len(labels) * 50),
                height=450,
            )
            layout.update(
                xaxis=dict(
                    **layout["xaxis"],
                    title="Engagement",
                    tickangle=-45 if len(labels) > 10 else 0,
                ),
                yaxis=dict(
                    **layout["yaxis"],
                    title="Overshoot Magnitude (px)",
                ),
                showlegend=False,
            )
            fig.update_layout(**layout)

            logger.info(
                "Created overshoot chart with %d engagements "
                "(mean=%.1f px)", len(labels), mean_mag,
            )
            return fig

        except Exception as e:
            logger.error("Failed to create overshoot chart: %s", e)
            raise
