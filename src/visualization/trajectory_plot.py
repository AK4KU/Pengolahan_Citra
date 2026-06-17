"""
Trajectory Visualization Module for FPS Aim Performance Analyzer.

Provides interactive Plotly visualizations of crosshair trajectories
with velocity-based color coding, target overlays, and animated playback.

Uses the 'RdYlBu_r' colorscale where blue indicates slow movement
and red indicates fast movement.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class TrajectoryPlotter:
    """Generates interactive trajectory visualizations using Plotly.

    Visualizes crosshair movement paths with velocity-based color coding,
    target position overlays, and animated playback capabilities.

    Attributes:
        theme: Dashboard theme dictionary from settings.
        colorscale: Plotly colorscale for velocity mapping.
        line_width: Trajectory line width from settings.
    """

    def __init__(self) -> None:
        """Initialize TrajectoryPlotter with theme settings."""
        self.theme: Dict[str, str] = settings.DASHBOARD_THEME
        self.colorscale: str = settings.TRAJECTORY_COLORMAP  # 'RdYlBu_r'
        self.line_width: int = settings.TRAJECTORY_LINE_WIDTH

    def _get_dark_layout(self, title: str = "", width: int = 900,
                         height: int = 600) -> Dict[str, Any]:
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

    def _compute_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute per-frame velocity magnitude from trajectory positions.

        Args:
            trajectory: Array of shape (N, 2+) where columns 0,1 are x,y.
                If a third column exists, it is treated as timestamp.

        Returns:
            Array of shape (N,) with velocity magnitudes (pixels/frame or
            pixels/second if timestamps are provided).
        """
        if len(trajectory) < 2:
            return np.zeros(len(trajectory))

        positions = trajectory[:, :2].astype(float)
        displacements = np.diff(positions, axis=0)
        distances = np.linalg.norm(displacements, axis=1)

        # If timestamps are available, compute pixels/second
        if trajectory.shape[1] >= 3:
            timestamps = trajectory[:, 2].astype(float)
            dt = np.diff(timestamps)
            dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero
            velocities = distances / dt
        else:
            velocities = distances

        # Prepend first velocity to match array length
        velocities = np.concatenate([[velocities[0]], velocities])
        return velocities

    def plot_trajectory_2d(
        self,
        trajectory: np.ndarray,
        target_positions: Optional[np.ndarray] = None,
        background_image: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """Plot 2D crosshair path with velocity-based color coding.

        Creates an interactive scatter plot of the crosshair trajectory
        where each point is colored by its velocity (blue=slow, red=fast).
        Optionally overlays target positions and a background screenshot.

        Args:
            trajectory: Array of shape (N, 2+) with columns [x, y, ...].
                Optional 3rd column for timestamps.
            target_positions: Optional array of shape (M, 2+) with columns
                [x, y, ...] for target locations. Additional columns are
                ignored.
            background_image: Optional numpy array (H, W, 3) of a screenshot
                to display as background.

        Returns:
            Plotly Figure with the 2D trajectory visualization.

        Raises:
            ValueError: If trajectory has fewer than 2 columns.
        """
        try:
            if trajectory.shape[1] < 2:
                raise ValueError(
                    f"Trajectory must have at least 2 columns (x, y), "
                    f"got {trajectory.shape[1]}"
                )

            fig = go.Figure()

            # Add background image if provided
            if background_image is not None:
                self._add_background_image(fig, background_image)

            # Compute velocities for color coding
            velocities = self._compute_velocity(trajectory)

            x_coords = trajectory[:, 0]
            y_coords = trajectory[:, 1]

            # Main trajectory line with velocity color coding
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+lines",
                marker=dict(
                    size=4,
                    color=velocities,
                    colorscale=self.colorscale,
                    colorbar=dict(
                        title=dict(text="Velocity", font=dict(
                            color=self.theme["text_primary"])),
                        tickfont=dict(color=self.theme["text_secondary"]),
                        bgcolor=self.theme["background_card"],
                    ),
                    showscale=True,
                ),
                line=dict(
                    color=self.theme["text_secondary"],
                    width=self.line_width,
                ),
                name="Crosshair Path",
                hovertemplate=(
                    "X: %{x:.0f}<br>Y: %{y:.0f}<br>"
                    "Velocity: %{marker.color:.1f}<extra></extra>"
                ),
            ))

            # Start point (green circle)
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=self.theme["success_color"],
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                text=["Start"],
                textposition="top center",
                textfont=dict(color=self.theme["success_color"], size=11),
                name="Start",
                showlegend=True,
            ))

            # End point (red circle)
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=self.theme["danger_color"],
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                text=["End"],
                textposition="top center",
                textfont=dict(color=self.theme["danger_color"], size=11),
                name="End",
                showlegend=True,
            ))

            # Target positions
            if target_positions is not None and len(target_positions) > 0:
                fig.add_trace(go.Scatter(
                    x=target_positions[:, 0],
                    y=target_positions[:, 1],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color=self.theme["accent_color"],
                        symbol="crosshair",
                        line=dict(color="white", width=1),
                    ),
                    name="Targets",
                    hovertemplate=(
                        "Target X: %{x:.0f}<br>"
                        "Target Y: %{y:.0f}<extra></extra>"
                    ),
                ))

            # Layout
            layout = self._get_dark_layout(
                title="Crosshair Trajectory (2D)",
                width=960,
                height=700,
            )
            layout.update(
                xaxis=dict(
                    **layout["xaxis"],
                    title="X Position (px)",
                    range=[0, settings.SCREEN_WIDTH],
                ),
                yaxis=dict(
                    **layout["yaxis"],
                    title="Y Position (px)",
                    range=[settings.SCREEN_HEIGHT, 0],  # Inverted Y
                    scaleanchor="x",
                    scaleratio=1,
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info(
                "Created 2D trajectory plot with %d points", len(trajectory)
            )
            return fig

        except Exception as e:
            logger.error("Failed to create 2D trajectory plot: %s", e)
            raise

    def plot_trajectory_animation(
        self,
        trajectory: np.ndarray,
        target_positions: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """Create animated playback of crosshair movement.

        Generates a Plotly figure with animation frames showing the
        progressive drawing of the crosshair trajectory path.

        Args:
            trajectory: Array of shape (N, 2+) with columns [x, y, ...].
            target_positions: Optional array of shape (M, 2+) for target
                locations.

        Returns:
            Plotly Figure with animation controls (play/pause/slider).
        """
        try:
            x_coords = trajectory[:, 0]
            y_coords = trajectory[:, 1]
            velocities = self._compute_velocity(trajectory)
            n_points = len(trajectory)

            # Determine frame step to keep animation manageable
            max_frames = 200
            step = max(1, n_points // max_frames)
            frame_indices = list(range(0, n_points, step))
            if frame_indices[-1] != n_points - 1:
                frame_indices.append(n_points - 1)

            # Create initial frame (first point only)
            fig = go.Figure(
                data=[
                    # Trajectory trail
                    go.Scatter(
                        x=[x_coords[0]],
                        y=[y_coords[0]],
                        mode="markers+lines",
                        marker=dict(
                            size=4,
                            color=[velocities[0]],
                            colorscale=self.colorscale,
                            cmin=float(np.min(velocities)),
                            cmax=float(np.max(velocities)),
                            showscale=True,
                            colorbar=dict(
                                title="Velocity",
                                tickfont=dict(
                                    color=self.theme["text_secondary"]),
                                bgcolor=self.theme["background_card"],
                            ),
                        ),
                        line=dict(
                            color=self.theme["text_secondary"],
                            width=self.line_width,
                        ),
                        name="Path",
                    ),
                    # Current position marker
                    go.Scatter(
                        x=[x_coords[0]],
                        y=[y_coords[0]],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=self.theme["primary_color"],
                            symbol="circle",
                            line=dict(color="white", width=2),
                        ),
                        name="Crosshair",
                    ),
                ],
            )

            # Add target positions as static trace
            if target_positions is not None and len(target_positions) > 0:
                fig.add_trace(go.Scatter(
                    x=target_positions[:, 0],
                    y=target_positions[:, 1],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color=self.theme["accent_color"],
                        symbol="crosshair",
                        line=dict(color="white", width=1),
                    ),
                    name="Targets",
                ))

            # Build animation frames
            frames = []
            for idx in frame_indices:
                end = idx + 1
                frame_data = [
                    go.Scatter(
                        x=x_coords[:end],
                        y=y_coords[:end],
                        mode="markers+lines",
                        marker=dict(
                            size=4,
                            color=velocities[:end],
                            colorscale=self.colorscale,
                            cmin=float(np.min(velocities)),
                            cmax=float(np.max(velocities)),
                            showscale=True,
                        ),
                        line=dict(
                            color=self.theme["text_secondary"],
                            width=self.line_width,
                        ),
                    ),
                    go.Scatter(
                        x=[x_coords[idx]],
                        y=[y_coords[idx]],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=self.theme["primary_color"],
                            symbol="circle",
                            line=dict(color="white", width=2),
                        ),
                    ),
                ]

                frames.append(go.Frame(
                    data=frame_data,
                    name=str(idx),
                    traces=[0, 1],  # Update first two traces
                ))

            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )],
                sliders=[dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        prefix="Frame: ",
                        visible=True,
                        font=dict(color=self.theme["text_secondary"]),
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    bgcolor=self.theme["background_card"],
                    activebgcolor=self.theme["primary_color"],
                    font=dict(color=self.theme["text_secondary"]),
                    steps=[
                        dict(
                            args=[
                                [str(idx)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(idx),
                            method="animate",
                        )
                        for idx in frame_indices
                    ],
                )],
            )

            # Apply dark theme layout
            layout = self._get_dark_layout(
                title="Crosshair Trajectory Animation",
                width=960,
                height=750,
            )
            layout.update(
                xaxis=dict(
                    **layout["xaxis"],
                    title="X Position (px)",
                    range=[0, settings.SCREEN_WIDTH],
                ),
                yaxis=dict(
                    **layout["yaxis"],
                    title="Y Position (px)",
                    range=[settings.SCREEN_HEIGHT, 0],
                    scaleanchor="x",
                    scaleratio=1,
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info(
                "Created trajectory animation with %d frames",
                len(frame_indices),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create trajectory animation: %s", e)
            raise

    def plot_engagement_trajectory(
        self,
        engagement_window: Dict[str, Any],
        trajectory_slice: np.ndarray,
    ) -> go.Figure:
        """Plot a single engagement with start/end markers and target.

        Visualizes the crosshair path during a specific engagement window,
        highlighting the start position, end position, target location,
        and velocity profile.

        Args:
            engagement_window: Dictionary with engagement metadata:
                - 'start_frame' (int): Frame index where engagement begins.
                - 'end_frame' (int): Frame index where engagement ends.
                - 'target_x' (float): Target X position.
                - 'target_y' (float): Target Y position.
                - 'hit' (bool, optional): Whether the engagement was a hit.
                - 'label' (str, optional): Engagement label/ID.
            trajectory_slice: Array of shape (N, 2+) with trajectory points
                for this engagement only.

        Returns:
            Plotly Figure showing the engagement trajectory with annotations.
        """
        try:
            fig = go.Figure()

            x_coords = trajectory_slice[:, 0]
            y_coords = trajectory_slice[:, 1]
            velocities = self._compute_velocity(trajectory_slice)

            # Trajectory with velocity coloring
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+lines",
                marker=dict(
                    size=5,
                    color=velocities,
                    colorscale=self.colorscale,
                    colorbar=dict(
                        title=dict(text="Velocity", font=dict(
                            color=self.theme["text_primary"])),
                        tickfont=dict(color=self.theme["text_secondary"]),
                        bgcolor=self.theme["background_card"],
                    ),
                    showscale=True,
                ),
                line=dict(
                    color=self.theme["text_secondary"],
                    width=self.line_width + 1,
                ),
                name="Aim Path",
                hovertemplate=(
                    "X: %{x:.0f}<br>Y: %{y:.0f}<br>"
                    "Velocity: %{marker.color:.1f}<extra></extra>"
                ),
            ))

            # Start point (green circle)
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=self.theme["success_color"],
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                text=["Start"],
                textposition="bottom center",
                textfont=dict(color=self.theme["success_color"], size=12),
                name="Start",
            ))

            # End point (red circle)
            hit = engagement_window.get("hit", None)
            end_color = (self.theme["success_color"] if hit
                         else self.theme["danger_color"])
            end_label = "Hit" if hit else ("Miss" if hit is not None else "End")
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=end_color,
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
                text=[end_label],
                textposition="top center",
                textfont=dict(color=end_color, size=12),
                name=end_label,
            ))

            # Target position (crosshair marker)
            target_x = engagement_window.get("target_x")
            target_y = engagement_window.get("target_y")
            if target_x is not None and target_y is not None:
                fig.add_trace(go.Scatter(
                    x=[target_x],
                    y=[target_y],
                    mode="markers+text",
                    marker=dict(
                        size=20,
                        color=self.theme["accent_color"],
                        symbol="crosshair",
                        line=dict(color="white", width=2),
                    ),
                    text=["Target"],
                    textposition="bottom right",
                    textfont=dict(
                        color=self.theme["accent_color"], size=12),
                    name="Target",
                ))

                # On-target radius circle (annotation)
                theta = np.linspace(0, 2 * np.pi, 64)
                radius = settings.ON_TARGET_THRESHOLD
                circle_x = target_x + radius * np.cos(theta)
                circle_y = target_y + radius * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(
                        color=self.theme["accent_color"],
                        width=1,
                        dash="dot",
                    ),
                    name=f"On-Target Radius ({radius}px)",
                    showlegend=True,
                    hoverinfo="skip",
                ))

            # Title with engagement label
            label = engagement_window.get("label", "")
            title_text = f"Engagement Trajectory"
            if label:
                title_text = f"Engagement: {label}"

            # Compute display range with padding
            all_x = list(x_coords)
            all_y = list(y_coords)
            if target_x is not None:
                all_x.append(target_x)
                all_y.append(target_y)
            padding = 60
            x_min = max(0, min(all_x) - padding)
            x_max = min(settings.SCREEN_WIDTH, max(all_x) + padding)
            y_min = max(0, min(all_y) - padding)
            y_max = min(settings.SCREEN_HEIGHT, max(all_y) + padding)

            layout = self._get_dark_layout(
                title=title_text,
                width=800,
                height=700,
            )
            layout.update(
                xaxis=dict(
                    **layout["xaxis"],
                    title="X Position (px)",
                    range=[x_min, x_max],
                ),
                yaxis=dict(
                    **layout["yaxis"],
                    title="Y Position (px)",
                    range=[y_max, y_min],  # Inverted Y
                    scaleanchor="x",
                    scaleratio=1,
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info(
                "Created engagement trajectory plot with %d points",
                len(trajectory_slice),
            )
            return fig

        except Exception as e:
            logger.error(
                "Failed to create engagement trajectory plot: %s", e
            )
            raise

    def _add_background_image(
        self, fig: go.Figure, image: np.ndarray
    ) -> None:
        """Add a background screenshot image to the figure.

        Args:
            fig: Plotly Figure to add the image to.
            image: Numpy array (H, W, 3) BGR or RGB image.
        """
        try:
            import base64
            from io import BytesIO
            from PIL import Image

            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # Encode as base64
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()

            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{encoded}",
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=settings.SCREEN_WIDTH,
                    sizey=settings.SCREEN_HEIGHT,
                    sizing="stretch",
                    opacity=0.3,
                    layer="below",
                )
            )
            logger.debug("Added background image to trajectory plot")

        except ImportError:
            logger.warning(
                "PIL not available, skipping background image overlay"
            )
        except Exception as e:
            logger.warning("Failed to add background image: %s", e)
