"""
Heatmap Visualization Module for FPS Aim Performance Analyzer.

Generates interactive Plotly heatmaps showing:
- Screen-space crosshair position density
- Relative aim spread patterns around targets
- Engagement location distributions

Uses 'Hot' and 'Inferno' colorscales with dark theme styling.
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


class HeatmapGenerator:
    """Generates interactive heatmap visualizations using Plotly.

    Creates screen-space density maps, relative aim spread heatmaps,
    and engagement location visualizations with dark theme styling.

    Attributes:
        theme: Dashboard theme dictionary from settings.
        screen_width: Game screen width in pixels.
        screen_height: Game screen height in pixels.
    """

    def __init__(self) -> None:
        """Initialize HeatmapGenerator with theme and screen settings."""
        self.theme: Dict[str, str] = settings.DASHBOARD_THEME
        self.screen_width: int = settings.SCREEN_WIDTH
        self.screen_height: int = settings.SCREEN_HEIGHT
        self.heatmap_resolution: Tuple[int, int] = settings.HEATMAP_RESOLUTION

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
        )

    def generate_screen_heatmap(
        self,
        crosshair_positions: np.ndarray,
        resolution: Tuple[int, int] = (1920, 1080),
    ) -> go.Figure:
        """Generate heatmap of where crosshair spends most time on screen.

        Creates a 2D histogram heatmap over the full screen resolution,
        showing crosshair position density with hot spots indicating
        frequently visited screen regions.

        Args:
            crosshair_positions: Array of shape (N, 2+) with columns
                [x, y, ...]. Only the first two columns are used.
            resolution: Screen resolution as (width, height) tuple.
                Defaults to (1920, 1080).

        Returns:
            Plotly Figure with the screen-space density heatmap.
        """
        try:
            x_coords = crosshair_positions[:, 0].astype(float)
            y_coords = crosshair_positions[:, 1].astype(float)

            fig = go.Figure()

            # 2D histogram heatmap
            fig.add_trace(go.Histogram2d(
                x=x_coords,
                y=y_coords,
                colorscale="Hot",
                nbinsx=self.heatmap_resolution[0],
                nbinsy=self.heatmap_resolution[1],
                colorbar=dict(
                    title=dict(text="Density", font=dict(
                        color=self.theme["text_primary"])),
                    tickfont=dict(color=self.theme["text_secondary"]),
                    bgcolor=self.theme["background_card"],
                ),
                hovertemplate=(
                    "X: %{x:.0f}<br>Y: %{y:.0f}<br>"
                    "Count: %{z}<extra></extra>"
                ),
                reversescale=False,
            ))

            # Screen center crosshair indicator
            fig.add_trace(go.Scatter(
                x=[resolution[0] / 2],
                y=[resolution[1] / 2],
                mode="markers",
                marker=dict(
                    size=12,
                    color="white",
                    symbol="cross-thin",
                    line=dict(width=2, color="white"),
                ),
                name="Screen Center",
                hovertemplate="Screen Center<extra></extra>",
            ))

            # Screen boundary rectangle
            fig.add_shape(
                type="rect",
                x0=0, y0=0,
                x1=resolution[0], y1=resolution[1],
                line=dict(color=self.theme["text_secondary"], width=1),
            )

            layout = self._get_dark_layout(
                title="Crosshair Screen Position Heatmap",
                width=960,
                height=int(960 * resolution[1] / resolution[0]),
            )
            layout.update(
                xaxis=dict(
                    title="X Position (px)",
                    range=[0, resolution[0]],
                    gridcolor=self.theme["background_card"],
                    showgrid=False,
                    constrain="domain",
                ),
                yaxis=dict(
                    title="Y Position (px)",
                    range=[resolution[1], 0],  # Inverted Y
                    gridcolor=self.theme["background_card"],
                    showgrid=False,
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
                "Created screen heatmap with %d positions",
                len(crosshair_positions),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create screen heatmap: %s", e)
            raise

    def generate_relative_heatmap(
        self,
        crosshair_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> go.Figure:
        """Generate heatmap of crosshair positions relative to target center.

        Computes the offset of each crosshair position from its
        corresponding target center, creating a spread pattern that
        shows aim precision and bias.

        Args:
            crosshair_positions: Array of shape (N, 2+) with columns
                [x, y, ...] for crosshair locations.
            target_positions: Array of shape (N, 2+) with columns
                [x, y, ...] for target locations. Must have the same
                length as crosshair_positions.

        Returns:
            Plotly Figure with relative aim spread heatmap centered
            at (0, 0) representing the target center.

        Raises:
            ValueError: If arrays have different lengths.
        """
        try:
            if len(crosshair_positions) != len(target_positions):
                raise ValueError(
                    f"Crosshair positions ({len(crosshair_positions)}) and "
                    f"target positions ({len(target_positions)}) must have "
                    f"the same length."
                )

            # Compute relative offsets
            dx = (crosshair_positions[:, 0].astype(float) -
                  target_positions[:, 0].astype(float))
            dy = (crosshair_positions[:, 1].astype(float) -
                  target_positions[:, 1].astype(float))

            fig = go.Figure()

            # Determine symmetric range
            max_offset = max(
                np.percentile(np.abs(dx), 98),
                np.percentile(np.abs(dy), 98),
                50,  # Minimum range
            )
            range_val = max_offset * 1.1

            # 2D histogram heatmap of relative offsets
            fig.add_trace(go.Histogram2d(
                x=dx,
                y=dy,
                colorscale="Inferno",
                nbinsx=80,
                nbinsy=80,
                colorbar=dict(
                    title=dict(text="Density", font=dict(
                        color=self.theme["text_primary"])),
                    tickfont=dict(color=self.theme["text_secondary"]),
                    bgcolor=self.theme["background_card"],
                ),
                hovertemplate=(
                    "ΔX: %{x:.1f} px<br>ΔY: %{y:.1f} px<br>"
                    "Count: %{z}<extra></extra>"
                ),
            ))

            # Target center marker (origin)
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(
                    size=16,
                    color=self.theme["accent_color"],
                    symbol="crosshair",
                    line=dict(color="white", width=2),
                ),
                name="Target Center",
            ))

            # On-target threshold circle
            theta = np.linspace(0, 2 * np.pi, 64)
            radius = settings.ON_TARGET_THRESHOLD
            fig.add_trace(go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode="lines",
                line=dict(
                    color=self.theme["success_color"],
                    width=2,
                    dash="dash",
                ),
                name=f"On-Target ({radius}px)",
                hoverinfo="skip",
            ))

            # Headshot threshold circle
            head_radius = settings.ON_TARGET_HEAD_THRESHOLD
            fig.add_trace(go.Scatter(
                x=head_radius * np.cos(theta),
                y=head_radius * np.sin(theta),
                mode="lines",
                line=dict(
                    color=self.theme["danger_color"],
                    width=2,
                    dash="dot",
                ),
                name=f"Headshot ({head_radius}px)",
                hoverinfo="skip",
            ))

            # Mean offset marker
            mean_dx = float(np.mean(dx))
            mean_dy = float(np.mean(dy))
            fig.add_trace(go.Scatter(
                x=[mean_dx],
                y=[mean_dy],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=self.theme["secondary_color"],
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                text=[f"Mean ({mean_dx:.1f}, {mean_dy:.1f})"],
                textposition="top right",
                textfont=dict(
                    color=self.theme["secondary_color"], size=10),
                name="Mean Offset",
            ))

            layout = self._get_dark_layout(
                title="Aim Spread Pattern (Relative to Target)",
                width=700,
                height=700,
            )
            layout.update(
                xaxis=dict(
                    title="ΔX from Target (px)",
                    range=[-range_val, range_val],
                    gridcolor=self.theme["background_card"],
                    zerolinecolor=self.theme["text_secondary"],
                    zerolinewidth=1,
                    showgrid=True,
                ),
                yaxis=dict(
                    title="ΔY from Target (px)",
                    range=[range_val, -range_val],  # Inverted Y
                    gridcolor=self.theme["background_card"],
                    zerolinecolor=self.theme["text_secondary"],
                    zerolinewidth=1,
                    showgrid=True,
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
                "Created relative heatmap with %d offset pairs",
                len(crosshair_positions),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create relative heatmap: %s", e)
            raise

    def generate_engagement_heatmap(
        self,
        engagement_windows: List[Dict[str, Any]],
    ) -> go.Figure:
        """Generate heatmap of engagement locations on screen.

        Shows where on the screen engagements (aim duels) occur,
        with density indicating frequently contested screen regions.
        Colors distinguish hits from misses.

        Args:
            engagement_windows: List of engagement dictionaries, each
                containing at least:
                - 'target_x' (float): Target X position on screen.
                - 'target_y' (float): Target Y position on screen.
                - 'hit' (bool, optional): Whether the engagement was a hit.

        Returns:
            Plotly Figure with engagement location heatmap.
        """
        try:
            if not engagement_windows:
                logger.warning("No engagement windows provided for heatmap")
                fig = go.Figure()
                layout = self._get_dark_layout(
                    title="Engagement Locations (No Data)")
                fig.update_layout(**layout)
                fig.add_annotation(
                    text="No engagement data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    font=dict(
                        color=self.theme["text_secondary"], size=16),
                    showarrow=False,
                )
                return fig

            # Extract positions and hit status
            target_xs = []
            target_ys = []
            hits = []
            for ew in engagement_windows:
                tx = ew.get("target_x")
                ty = ew.get("target_y")
                if tx is not None and ty is not None:
                    target_xs.append(float(tx))
                    target_ys.append(float(ty))
                    hits.append(ew.get("hit", None))

            if not target_xs:
                logger.warning("No valid target positions in engagements")
                fig = go.Figure()
                layout = self._get_dark_layout(
                    title="Engagement Locations (No Valid Positions)")
                fig.update_layout(**layout)
                return fig

            target_xs_arr = np.array(target_xs)
            target_ys_arr = np.array(target_ys)

            fig = go.Figure()

            # Background density heatmap
            fig.add_trace(go.Histogram2d(
                x=target_xs_arr,
                y=target_ys_arr,
                colorscale="Inferno",
                nbinsx=self.heatmap_resolution[0] // 2,
                nbinsy=self.heatmap_resolution[1] // 2,
                opacity=0.7,
                colorbar=dict(
                    title=dict(text="Engagements", font=dict(
                        color=self.theme["text_primary"])),
                    tickfont=dict(color=self.theme["text_secondary"]),
                    bgcolor=self.theme["background_card"],
                ),
                hovertemplate=(
                    "X: %{x:.0f}<br>Y: %{y:.0f}<br>"
                    "Count: %{z}<extra></extra>"
                ),
                showscale=True,
            ))

            # Overlay individual engagement markers, separated by hit/miss
            hit_x = [x for x, h in zip(target_xs, hits) if h is True]
            hit_y = [y for y, h in zip(target_ys, hits) if h is True]
            miss_x = [x for x, h in zip(target_xs, hits) if h is False]
            miss_y = [y for y, h in zip(target_ys, hits) if h is False]
            unknown_x = [x for x, h in zip(target_xs, hits) if h is None]
            unknown_y = [y for y, h in zip(target_ys, hits) if h is None]

            if hit_x:
                fig.add_trace(go.Scatter(
                    x=hit_x,
                    y=hit_y,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=self.theme["success_color"],
                        symbol="circle",
                        line=dict(color="white", width=1),
                        opacity=0.8,
                    ),
                    name=f"Hits ({len(hit_x)})",
                    hovertemplate="Hit<br>X: %{x:.0f}<br>Y: %{y:.0f}"
                                  "<extra></extra>",
                ))

            if miss_x:
                fig.add_trace(go.Scatter(
                    x=miss_x,
                    y=miss_y,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=self.theme["danger_color"],
                        symbol="x",
                        line=dict(color="white", width=1),
                        opacity=0.8,
                    ),
                    name=f"Misses ({len(miss_x)})",
                    hovertemplate="Miss<br>X: %{x:.0f}<br>Y: %{y:.0f}"
                                  "<extra></extra>",
                ))

            if unknown_x:
                fig.add_trace(go.Scatter(
                    x=unknown_x,
                    y=unknown_y,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=self.theme["text_secondary"],
                        symbol="circle-open",
                        line=dict(width=1),
                        opacity=0.6,
                    ),
                    name=f"Unknown ({len(unknown_x)})",
                    hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}"
                                  "<extra></extra>",
                ))

            # Screen boundary
            fig.add_shape(
                type="rect",
                x0=0, y0=0,
                x1=self.screen_width, y1=self.screen_height,
                line=dict(color=self.theme["text_secondary"], width=1),
            )

            layout = self._get_dark_layout(
                title=f"Engagement Locations ({len(target_xs)} total)",
                width=960,
                height=int(960 * self.screen_height / self.screen_width),
            )
            layout.update(
                xaxis=dict(
                    title="X Position (px)",
                    range=[0, self.screen_width],
                    gridcolor=self.theme["background_card"],
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Y Position (px)",
                    range=[self.screen_height, 0],  # Inverted Y
                    gridcolor=self.theme["background_card"],
                    showgrid=False,
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
                "Created engagement heatmap with %d engagements",
                len(target_xs),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create engagement heatmap: %s", e)
            raise
