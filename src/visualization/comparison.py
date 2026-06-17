"""
Comparison Visualization Module for FPS Aim Performance Analyzer.

Provides side-by-side and overlaid visualizations for comparing
performance metrics across multiple players or sessions.

Includes radar charts, metric tables, box plots, bar charts,
and skill level distribution views.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)

# Consistent color palette for multi-player comparisons
PLAYER_COLORS = [
    "#7C3AED",  # Purple (primary)
    "#06B6D4",  # Cyan (secondary)
    "#F59E0B",  # Amber (accent)
    "#10B981",  # Green (success)
    "#EF4444",  # Red (danger)
    "#EC4899",  # Pink
    "#8B5CF6",  # Violet
    "#14B8A6",  # Teal
    "#F97316",  # Orange
    "#6366F1",  # Indigo
]


class ComparisonView:
    """Generates multi-player/session comparison visualizations.

    Creates overlaid and side-by-side charts for comparing aim
    performance metrics across different players or sessions.

    Attributes:
        theme: Dashboard theme dictionary from settings.
        player_colors: Color palette for distinguishing players.
    """

    def __init__(self) -> None:
        """Initialize ComparisonView with theme settings."""
        self.theme: Dict[str, str] = settings.DASHBOARD_THEME
        self.player_colors: List[str] = PLAYER_COLORS

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

    def _get_player_color(self, index: int) -> str:
        """Get a consistent color for a player by index.

        Args:
            index: Player index (0-based).

        Returns:
            Hex color string for the player.
        """
        return self.player_colors[index % len(self.player_colors)]

    def _extract_radar_metrics(
        self, report: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract normalized radar chart metrics from a player report.

        Extracts and normalizes the five radar axes (Accuracy, Speed,
        Consistency, Smoothness, Precision) from a performance report
        to 0-100 scale.

        Args:
            report: Player performance report dictionary. Expected keys:
                - 'aim_accuracy' (float): Accuracy percentage (0-100).
                - 'avg_ttt_ms' (float): Average time-to-target in ms.
                - 'consistency' (float): Consistency score (0-1).
                - 'smoothness' (float): Smoothness score (0-1).
                - 'precision' (float): Precision score (0-1).

        Returns:
            Dictionary with normalized metric values (0-100).
        """
        return {
            "Accuracy": min(100, max(0, report.get("aim_accuracy", 0))),
            "Speed": min(100, max(0, 100 - report.get(
                "avg_ttt_ms", 1000) / 10)),
            "Consistency": min(100, max(0, report.get(
                "consistency", 0) * 100)),
            "Smoothness": min(100, max(0, report.get(
                "smoothness", 0) * 100)),
            "Precision": min(100, max(0, report.get(
                "precision", 0) * 100)),
        }

    def compare_radar(
        self,
        player_reports: List[Dict[str, Any]],
        player_names: List[str],
    ) -> go.Figure:
        """Create overlaid radar charts comparing multiple players.

        Each player's performance is shown as a polygon on the same
        radar chart, with five axes representing key performance
        dimensions.

        Args:
            player_reports: List of performance report dictionaries,
                one per player.
            player_names: List of player/session names corresponding
                to the reports.

        Returns:
            Plotly Figure with overlaid radar/spider charts.

        Raises:
            ValueError: If reports and names lists have different lengths.
        """
        try:
            if len(player_reports) != len(player_names):
                raise ValueError(
                    f"Number of reports ({len(player_reports)}) must match "
                    f"number of names ({len(player_names)})"
                )

            fig = go.Figure()

            categories = [
                "Accuracy", "Speed", "Consistency",
                "Smoothness", "Precision",
            ]

            for i, (report, name) in enumerate(
                zip(player_reports, player_names)
            ):
                metrics = self._extract_radar_metrics(report)
                values = [metrics[cat] for cat in categories]
                # Close the radar polygon
                values.append(values[0])
                cats_closed = categories + [categories[0]]

                color = self._get_player_color(i)

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=cats_closed,
                    fill="toself",
                    fillcolor=color.replace(")", ", 0.15)").replace(
                        "#", "rgba(") if color.startswith("rgba") else None,
                    opacity=0.8,
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "%{theta}: %{r:.1f}<extra></extra>"
                    ),
                ))

            layout = self._get_dark_layout(
                title="Performance Comparison (Radar)",
                width=700,
                height=600,
            )
            layout.update(
                polar=dict(
                    bgcolor=self.theme["background_dark"],
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        gridcolor=self.theme["background_card"],
                        linecolor=self.theme["background_card"],
                        tickfont=dict(color=self.theme["text_secondary"]),
                    ),
                    angularaxis=dict(
                        gridcolor=self.theme["background_card"],
                        linecolor=self.theme["background_card"],
                        tickfont=dict(
                            color=self.theme["text_primary"], size=13),
                    ),
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
                showlegend=True,
            )
            fig.update_layout(**layout)

            logger.info(
                "Created comparison radar chart for %d players",
                len(player_names),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create comparison radar chart: %s", e)
            raise

    def compare_metrics_table(
        self,
        player_reports: List[Dict[str, Any]],
        player_names: List[str],
    ) -> pd.DataFrame:
        """Create a comparison table with all metrics across players.

        Extracts key performance metrics from each player report and
        assembles them into a DataFrame for tabular display.

        Args:
            player_reports: List of performance report dictionaries.
            player_names: List of player/session names.

        Returns:
            pandas DataFrame with players as rows and metrics as columns.
            Columns include: Accuracy (%), Avg TTT (ms), Overshoot Ratio,
            Consistency, Smoothness, Precision, Skill Level.

        Raises:
            ValueError: If reports and names lists have different lengths.
        """
        try:
            if len(player_reports) != len(player_names):
                raise ValueError(
                    f"Number of reports ({len(player_reports)}) must match "
                    f"number of names ({len(player_names)})"
                )

            rows = []
            for report, name in zip(player_reports, player_names):
                row = {
                    "Player": name,
                    "Accuracy (%)": round(
                        report.get("aim_accuracy", 0), 1),
                    "Avg TTT (ms)": round(
                        report.get("avg_ttt_ms", 0), 1),
                    "Median TTT (ms)": round(
                        report.get("median_ttt_ms", 0), 1),
                    "Overshoot Ratio": round(
                        report.get("overshoot_ratio", 0), 3),
                    "Consistency": round(
                        report.get("consistency", 0), 3),
                    "Smoothness": round(
                        report.get("smoothness", 0), 3),
                    "Precision": round(
                        report.get("precision", 0), 3),
                    "Engagements": report.get("total_engagements", 0),
                    "Hits": report.get("total_hits", 0),
                    "Skill Level": report.get("skill_level", "Unknown"),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df = df.set_index("Player")

            logger.info(
                "Created comparison table for %d players with %d metrics",
                len(player_names),
                len(df.columns),
            )
            return df

        except Exception as e:
            logger.error("Failed to create comparison table: %s", e)
            raise

    def compare_ttt_boxplot(
        self,
        player_ttts: Dict[str, List[float]],
    ) -> go.Figure:
        """Create box plot comparing time-to-target distributions.

        Shows the distribution of TTT values for each player/session
        as side-by-side box plots, revealing median, quartiles, and
        outliers.

        Args:
            player_ttts: Dictionary mapping player names to lists of
                TTT values in milliseconds.

        Returns:
            Plotly Figure with side-by-side box plots.
        """
        try:
            fig = go.Figure()

            for i, (name, ttts) in enumerate(player_ttts.items()):
                color = self._get_player_color(i)
                fig.add_trace(go.Box(
                    y=ttts,
                    name=name,
                    marker=dict(
                        color=color,
                        outliercolor=color,
                        line=dict(outliercolor=color),
                    ),
                    line=dict(color=color),
                    fillcolor=color + "33",  # 20% opacity hex
                    boxmean="sd",
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "TTT: %{y:.0f} ms<extra></extra>"
                    ),
                ))

            layout = self._get_dark_layout(
                title="Time-to-Target Distribution Comparison",
                width=800,
                height=500,
            )
            layout.update(
                yaxis=dict(
                    title="Time-to-Target (ms)",
                    gridcolor=self.theme["background_card"],
                    zerolinecolor=self.theme["background_card"],
                ),
                xaxis=dict(
                    gridcolor=self.theme["background_card"],
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
                boxmode="group",
            )
            fig.update_layout(**layout)

            logger.info(
                "Created TTT boxplot comparison for %d players",
                len(player_ttts),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create TTT boxplot: %s", e)
            raise

    def compare_accuracy_bars(
        self,
        player_reports: List[Dict[str, Any]],
        player_names: List[str],
    ) -> go.Figure:
        """Create grouped bar chart comparing accuracy metrics.

        Shows multiple accuracy-related metrics as grouped bars for
        each player, enabling side-by-side comparison.

        Args:
            player_reports: List of performance report dictionaries.
                Expected keys: 'aim_accuracy', 'headshot_accuracy',
                'consistency'.
            player_names: List of player/session names.

        Returns:
            Plotly Figure with grouped bar chart.

        Raises:
            ValueError: If reports and names lists have different lengths.
        """
        try:
            if len(player_reports) != len(player_names):
                raise ValueError(
                    f"Number of reports ({len(player_reports)}) must match "
                    f"number of names ({len(player_names)})"
                )

            # Define metric categories for bars
            metric_keys = [
                ("aim_accuracy", "Aim Accuracy (%)", 1.0),
                ("headshot_accuracy", "Headshot Accuracy (%)", 1.0),
                ("consistency", "Consistency (%)", 100.0),
                ("precision", "Precision (%)", 100.0),
            ]

            fig = go.Figure()

            for i, (name, report) in enumerate(
                zip(player_names, player_reports)
            ):
                color = self._get_player_color(i)
                values = []
                labels = []
                for key, label, scale in metric_keys:
                    val = report.get(key, 0) * scale
                    values.append(round(val, 1))
                    labels.append(label)

                fig.add_trace(go.Bar(
                    x=labels,
                    y=values,
                    name=name,
                    marker=dict(
                        color=color,
                        line=dict(color="white", width=0.5),
                    ),
                    text=[f"{v:.1f}%" for v in values],
                    textposition="outside",
                    textfont=dict(
                        color=self.theme["text_primary"], size=10),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "%{x}: %{y:.1f}%<extra></extra>"
                    ),
                ))

            layout = self._get_dark_layout(
                title="Accuracy Metrics Comparison",
                width=900,
                height=500,
            )
            layout.update(
                barmode="group",
                yaxis=dict(
                    title="Value (%)",
                    range=[0, 110],
                    gridcolor=self.theme["background_card"],
                    zerolinecolor=self.theme["background_card"],
                ),
                xaxis=dict(
                    gridcolor=self.theme["background_card"],
                    tickfont=dict(size=11),
                ),
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                ),
            )
            fig.update_layout(**layout)

            logger.info(
                "Created accuracy bar comparison for %d players",
                len(player_names),
            )
            return fig

        except Exception as e:
            logger.error("Failed to create accuracy bar chart: %s", e)
            raise

    def skill_level_distribution(
        self,
        player_reports: List[Dict[str, Any]],
    ) -> go.Figure:
        """Create pie chart showing distribution of skill levels.

        Aggregates skill level classifications across all player
        reports and displays the distribution as a pie chart.

        Args:
            player_reports: List of performance report dictionaries.
                Each should contain a 'skill_level' key with values
                like 'beginner', 'intermediate', or 'advanced'.

        Returns:
            Plotly Figure with skill level distribution pie chart.
        """
        try:
            # Count skill levels
            level_counts: Dict[str, int] = {}
            for report in player_reports:
                level = report.get("skill_level", "Unknown").capitalize()
                level_counts[level] = level_counts.get(level, 0) + 1

            # Color mapping for skill levels
            level_colors = {
                "Beginner": self.theme["danger_color"],
                "Intermediate": self.theme["accent_color"],
                "Advanced": self.theme["success_color"],
                "Unknown": self.theme["text_secondary"],
            }

            labels = list(level_counts.keys())
            values = list(level_counts.values())
            colors = [level_colors.get(l, self.theme["text_secondary"])
                      for l in labels]

            fig = go.Figure()

            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                marker=dict(
                    colors=colors,
                    line=dict(color=self.theme["background_dark"], width=2),
                ),
                textinfo="label+percent",
                textposition="inside",
                textfont=dict(color="white", size=14),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Percentage: %{percent}<extra></extra>"
                ),
                hole=0.4,  # Donut chart
            ))

            # Center annotation
            total = sum(values)
            fig.add_annotation(
                text=f"<b>{total}</b><br>Players",
                x=0.5, y=0.5,
                font=dict(color=self.theme["text_primary"], size=18),
                showarrow=False,
                xref="paper",
                yref="paper",
            )

            layout = self._get_dark_layout(
                title="Skill Level Distribution",
                width=600,
                height=500,
            )
            layout.update(
                legend=dict(
                    bgcolor=self.theme["background_card"],
                    font=dict(color=self.theme["text_primary"]),
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                ),
                showlegend=True,
            )
            fig.update_layout(**layout)

            logger.info(
                "Created skill level distribution pie chart: %s",
                level_counts,
            )
            return fig

        except Exception as e:
            logger.error(
                "Failed to create skill level distribution: %s", e
            )
            raise
