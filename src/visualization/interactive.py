"""
Interactive visualization module for the Diversity Decoding Arena.

Generates Plotly-compatible plot specifications as plain dicts (no plotly
dependency required).  Also provides matplotlib-style static plot configs,
dashboard layout generation, frame-based animation specs, style management,
and HTML/JSON export utilities.
"""

from __future__ import annotations

import colorsys
import copy
import json
import math
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PlotSpec = Dict[str, Any]
Number = Union[int, float]

# ========================================================================= #
#  1.  PlotConfig                                                           #
# ========================================================================= #


@dataclass
class PlotConfig:
    """Centralised configuration for every plot produced by this module."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    width: int = 900
    height: int = 600
    color_scheme: str = "default"
    font_size: int = 14
    legend_position: str = "top-right"
    export_format: str = "html"  # html | png | svg | json
    dpi: int = 150

    # Optional overrides
    margin: Optional[Dict[str, int]] = None
    template: Optional[str] = None
    show_grid: bool = True
    title_font_size: int = 18
    axis_font_size: int = 12
    background_color: str = "#ffffff"
    plot_background_color: str = "#ffffff"
    font_family: str = "Arial, sans-serif"

    def to_layout_dict(self) -> Dict[str, Any]:
        """Convert config into a Plotly-compatible *layout* dict."""
        legend_map = {
            "top-right": {"x": 0.98, "y": 0.98, "xanchor": "right", "yanchor": "top"},
            "top-left": {"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
            "bottom-right": {"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
            "bottom-left": {"x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"},
            "outside-right": {"x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"},
        }
        legend = legend_map.get(self.legend_position, legend_map["top-right"])

        layout: Dict[str, Any] = {
            "title": {
                "text": self.title,
                "font": {"size": self.title_font_size, "family": self.font_family},
            },
            "xaxis": {
                "title": {"text": self.xlabel, "font": {"size": self.axis_font_size}},
                "showgrid": self.show_grid,
                "gridcolor": "#e0e0e0",
                "zeroline": False,
            },
            "yaxis": {
                "title": {"text": self.ylabel, "font": {"size": self.axis_font_size}},
                "showgrid": self.show_grid,
                "gridcolor": "#e0e0e0",
                "zeroline": False,
            },
            "width": self.width,
            "height": self.height,
            "font": {"size": self.font_size, "family": self.font_family},
            "legend": legend,
            "paper_bgcolor": self.background_color,
            "plot_bgcolor": self.plot_background_color,
            "margin": self.margin or {"l": 80, "r": 40, "t": 60, "b": 60},
        }
        if self.template:
            layout["template"] = self.template
        return layout

    def copy(self, **overrides: Any) -> "PlotConfig":
        d = asdict(self)
        d.update(overrides)
        return PlotConfig(**d)


# ========================================================================= #
#  6.  PlotStyleManager                                                     #
# ========================================================================= #


class PlotStyleManager:
    """Manages colour palettes and style presets."""

    # --- built-in palettes ---------------------------------------------------
    _DEFAULT_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    _COLORBLIND_COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
        "#D55E00", "#F0E442", "#000000",
    ]
    _DARK_COLORS = [
        "#bb86fc", "#03dac6", "#cf6679", "#ffb74d", "#81c784",
        "#64b5f6", "#e57373", "#fff176", "#4dd0e1", "#ba68c8",
    ]
    _PASTEL_COLORS = [
        "#a8d8ea", "#aa96da", "#fcbad3", "#ffffd2", "#b5ead7",
        "#c7ceea", "#f7dc6f", "#f0b27a", "#d7bde2", "#a9dfbf",
    ]
    _ACADEMIC_COLORS = [
        "#2c3e50", "#e74c3c", "#3498db", "#27ae60", "#f39c12",
        "#8e44ad", "#1abc9c", "#d35400", "#7f8c8d", "#c0392b",
    ]

    _PALETTE_MAP: Dict[str, List[str]] = {
        "default": _DEFAULT_COLORS,
        "colorblind": _COLORBLIND_COLORS,
        "dark": _DARK_COLORS,
        "pastel": _PASTEL_COLORS,
        "academic": _ACADEMIC_COLORS,
    }

    # --- style presets -------------------------------------------------------
    @classmethod
    def academic_style(cls) -> PlotConfig:
        """Style suitable for academic papers."""
        return PlotConfig(
            font_size=12,
            title_font_size=14,
            axis_font_size=11,
            font_family="Times New Roman, serif",
            width=700,
            height=500,
            color_scheme="academic",
            background_color="#ffffff",
            plot_background_color="#ffffff",
            show_grid=True,
            legend_position="top-right",
            dpi=300,
            export_format="svg",
        )

    @classmethod
    def presentation_style(cls) -> PlotConfig:
        """Style for slide decks – large fonts, bold colours."""
        return PlotConfig(
            font_size=20,
            title_font_size=28,
            axis_font_size=18,
            font_family="Helvetica, Arial, sans-serif",
            width=1200,
            height=700,
            color_scheme="default",
            background_color="#ffffff",
            plot_background_color="#ffffff",
            show_grid=False,
            legend_position="outside-right",
            dpi=150,
            export_format="png",
        )

    @classmethod
    def dark_mode(cls) -> PlotConfig:
        """Dark background style."""
        return PlotConfig(
            font_size=14,
            title_font_size=18,
            axis_font_size=12,
            font_family="Arial, sans-serif",
            width=900,
            height=600,
            color_scheme="dark",
            background_color="#1e1e1e",
            plot_background_color="#2d2d2d",
            show_grid=True,
            legend_position="top-right",
            dpi=150,
            export_format="html",
        )

    @classmethod
    def light_mode(cls) -> PlotConfig:
        """Default light style."""
        return PlotConfig()

    @classmethod
    def colorblind_friendly(cls) -> PlotConfig:
        """Palette safe for colour-vision deficiency."""
        return PlotConfig(color_scheme="colorblind")

    # --- colour helpers ------------------------------------------------------
    @classmethod
    def get_colors(cls, n: int, scheme: str = "default") -> List[str]:
        """Return *n* hex colour strings from the requested palette.

        If *n* exceeds the palette length the colours are cycled or
        interpolated via HSL.
        """
        base = cls._PALETTE_MAP.get(scheme, cls._DEFAULT_COLORS)
        if n <= len(base):
            return base[:n]
        # Interpolate extra colours in HSL space
        colors = list(base)
        while len(colors) < n:
            idx = len(colors) % len(base)
            r, g, b = cls._hex_to_rgb(base[idx])
            h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            h = (h + 0.618033988749895) % 1.0  # golden-ratio offset
            nr, ng, nb = colorsys.hls_to_rgb(h, l, s)
            colors.append(cls._rgb_to_hex(int(nr * 255), int(ng * 255), int(nb * 255)))
        return colors[:n]

    @staticmethod
    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    @staticmethod
    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"


# ========================================================================= #
#  8.  Helper functions                                                     #
# ========================================================================= #


def create_colorscale(n_colors: int, scheme: str = "default") -> List[List[Any]]:
    """Return a Plotly-compatible colorscale list ``[[0, c0], …, [1, cn]]``."""
    colors = PlotStyleManager.get_colors(max(n_colors, 2), scheme)
    scale: List[List[Any]] = []
    for i, c in enumerate(colors[:n_colors]):
        scale.append([i / max(n_colors - 1, 1), c])
    return scale


def create_annotation(
    text: str,
    x: Number,
    y: Number,
    *,
    font_size: int = 12,
    font_color: str = "#000000",
    showarrow: bool = True,
    arrowhead: int = 2,
    ax: int = 0,
    ay: int = -30,
    xref: str = "x",
    yref: str = "y",
    bgcolor: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a single Plotly annotation dict."""
    ann: Dict[str, Any] = {
        "text": text,
        "x": x,
        "y": y,
        "xref": xref,
        "yref": yref,
        "showarrow": showarrow,
        "arrowhead": arrowhead,
        "ax": ax,
        "ay": ay,
        "font": {"size": font_size, "color": font_color},
    }
    if bgcolor:
        ann["bgcolor"] = bgcolor
    return ann


def create_shape(
    shape_type: str,
    coordinates: Dict[str, Number],
    *,
    line_color: str = "#000000",
    line_width: int = 1,
    line_dash: str = "solid",
    fill_color: Optional[str] = None,
    opacity: float = 1.0,
    xref: str = "x",
    yref: str = "y",
) -> Dict[str, Any]:
    """Create a Plotly shape dict.

    *shape_type* is one of ``line``, ``rect``, ``circle``.
    *coordinates* should contain ``x0, y0, x1, y1``.
    """
    shape: Dict[str, Any] = {
        "type": shape_type,
        "xref": xref,
        "yref": yref,
        "x0": coordinates.get("x0", 0),
        "y0": coordinates.get("y0", 0),
        "x1": coordinates.get("x1", 1),
        "y1": coordinates.get("y1", 1),
        "line": {"color": line_color, "width": line_width, "dash": line_dash},
        "opacity": opacity,
    }
    if fill_color:
        shape["fillcolor"] = fill_color
    return shape


def format_axis_labels(
    values: Sequence[Number], format_str: str = ".2f"
) -> List[str]:
    """Format numeric tick values using *format_str*."""
    return [f"{v:{format_str}}" for v in values]


# internal helpers

def _ensure_list(x: Any) -> list:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _minmax(vals: Sequence[Number]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _close_polygon(theta: List[float], r: List[float]) -> Tuple[List[float], List[float]]:
    """Close a radar polygon by repeating the first point."""
    return theta + [theta[0]], r + [r[0]]


# ========================================================================= #
#  2.  InteractivePlotter                                                   #
# ========================================================================= #


class InteractivePlotter:
    """Generates Plotly-compatible figure specs as plain Python dicts."""

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()
        self._style = PlotStyleManager()

    # ----- internal helpers -------------------------------------------------
    def _base_layout(self, **overrides: Any) -> Dict[str, Any]:
        layout = self.config.to_layout_dict()
        layout.update(overrides)
        return layout

    def _colors(self, n: int) -> List[str]:
        return self._style.get_colors(n, self.config.color_scheme)

    # ----- 2a. Pareto frontier -----------------------------------------------
    def pareto_frontier_plot(
        self,
        points: np.ndarray,
        frontier_indices: Sequence[int],
        labels: Optional[Sequence[str]] = None,
    ) -> PlotSpec:
        """Scatter plot with Pareto frontier highlighted and connected.

        Parameters
        ----------
        points : array-like, shape (N, 2)
        frontier_indices : indices of Pareto-optimal points
        labels : optional per-point labels
        """
        pts = np.asarray(points, dtype=float)
        xs = pts[:, 0].tolist()
        ys = pts[:, 1].tolist()
        fi = list(frontier_indices)

        hover = labels if labels else [f"Point {i}" for i in range(len(xs))]

        dominated_idx = [i for i in range(len(xs)) if i not in fi]
        colors = self._colors(3)

        # dominated scatter
        traces: List[Dict[str, Any]] = []
        if dominated_idx:
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": [xs[i] for i in dominated_idx],
                "y": [ys[i] for i in dominated_idx],
                "text": [hover[i] for i in dominated_idx],
                "hoverinfo": "text+x+y",
                "marker": {"color": colors[0], "size": 8, "opacity": 0.5},
                "name": "Dominated",
            })

        # frontier scatter
        fx = [xs[i] for i in fi]
        fy = [ys[i] for i in fi]
        fh = [hover[i] for i in fi]
        sorted_pairs = sorted(zip(fx, fy, fh), key=lambda t: t[0])
        fx_s = [p[0] for p in sorted_pairs]
        fy_s = [p[1] for p in sorted_pairs]
        fh_s = [p[2] for p in sorted_pairs]

        traces.append({
            "type": "scatter",
            "mode": "markers+lines",
            "x": fx_s,
            "y": fy_s,
            "text": fh_s,
            "hoverinfo": "text+x+y",
            "marker": {"color": colors[1], "size": 12, "symbol": "star"},
            "line": {"color": colors[1], "width": 2, "dash": "dash"},
            "name": "Pareto Frontier",
        })

        # ideal / nadir annotations
        annotations: List[Dict[str, Any]] = []
        if fx_s:
            annotations.append(
                create_annotation("Best X", fx_s[0], fy_s[0], ay=-40)
            )
            annotations.append(
                create_annotation("Best Y", fx_s[-1], fy_s[-1], ay=-40)
            )

        layout = self._base_layout(
            title={"text": self.config.title or "Pareto Frontier",
                   "font": {"size": self.config.title_font_size}},
            xaxis={
                "title": {"text": self.config.xlabel or "Objective 1"},
                "showgrid": self.config.show_grid,
                "gridcolor": "#e0e0e0",
            },
            yaxis={
                "title": {"text": self.config.ylabel or "Objective 2"},
                "showgrid": self.config.show_grid,
                "gridcolor": "#e0e0e0",
            },
            annotations=annotations,
        )

        return {"data": traces, "layout": layout}

    # ----- 2b. Metric heatmap ------------------------------------------------
    def metric_heatmap(
        self,
        correlation_matrix: np.ndarray,
        metric_names: Sequence[str],
    ) -> PlotSpec:
        """Correlation heatmap with annotated cells."""
        mat = np.asarray(correlation_matrix, dtype=float)
        n = mat.shape[0]
        names = list(metric_names)

        # cell text
        text = [[f"{mat[i, j]:.2f}" for j in range(n)] for i in range(n)]

        trace: Dict[str, Any] = {
            "type": "heatmap",
            "z": mat.tolist(),
            "x": names,
            "y": names,
            "text": text,
            "texttemplate": "%{text}",
            "colorscale": [
                [0.0, "#3b4cc0"],
                [0.25, "#7b9ff9"],
                [0.5, "#f7f7f7"],
                [0.75, "#f4987a"],
                [1.0, "#b40426"],
            ],
            "zmin": -1,
            "zmax": 1,
            "colorbar": {"title": "Correlation", "titleside": "right"},
            "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>",
        }

        layout = self._base_layout(
            title={"text": self.config.title or "Metric Correlation Heatmap",
                   "font": {"size": self.config.title_font_size}},
            xaxis={"tickangle": -45, "side": "bottom"},
            yaxis={"autorange": "reversed"},
        )
        layout["height"] = max(self.config.height, 100 + 40 * n)
        layout["width"] = max(self.config.width, 100 + 40 * n)

        return {"data": [trace], "layout": layout}

    # ----- 2c. Radar chart ---------------------------------------------------
    def algorithm_radar_chart(
        self,
        algorithm_scores: Dict[str, Sequence[Number]],
        metric_names: Sequence[str],
    ) -> PlotSpec:
        """Radar / spider chart comparing algorithms across metrics."""
        names = list(metric_names)
        colors = self._colors(len(algorithm_scores))
        traces: List[Dict[str, Any]] = []

        for idx, (algo, scores) in enumerate(algorithm_scores.items()):
            r = _ensure_list(scores)
            theta = list(names)
            r_c, theta_c = _close_polygon(theta, r)
            traces.append({
                "type": "scatterpolar",
                "r": r_c,
                "theta": theta_c,
                "fill": "toself",
                "fillcolor": colors[idx] + "33",  # 20% opacity via hex alpha
                "line": {"color": colors[idx], "width": 2},
                "name": algo,
                "hovertemplate": "%{theta}: %{r:.3f}<extra>" + algo + "</extra>",
            })

        layout = self._base_layout(
            title={"text": self.config.title or "Algorithm Comparison (Radar)",
                   "font": {"size": self.config.title_font_size}},
            polar={
                "radialaxis": {"visible": True, "range": [0, 1], "showticklabels": True},
                "angularaxis": {"direction": "clockwise"},
            },
        )
        # remove cartesian axes
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)

        return {"data": traces, "layout": layout}

    # ----- 2d. Diversity–quality scatter ------------------------------------
    def diversity_quality_scatter(
        self,
        diversity_scores: Sequence[Number],
        quality_scores: Sequence[Number],
        labels: Sequence[str],
    ) -> PlotSpec:
        """Scatter of diversity vs quality with labelled quadrants."""
        xs = _ensure_list(diversity_scores)
        ys = _ensure_list(quality_scores)
        labs = list(labels)
        n = len(xs)
        colors = self._colors(n)

        traces: List[Dict[str, Any]] = []
        for i in range(n):
            traces.append({
                "type": "scatter",
                "mode": "markers+text",
                "x": [xs[i]],
                "y": [ys[i]],
                "text": [labs[i]],
                "textposition": "top center",
                "marker": {"color": colors[i], "size": 14},
                "name": labs[i],
                "hovertemplate": f"{labs[i]}<br>Diversity: %{{x:.3f}}<br>Quality: %{{y:.3f}}<extra></extra>",
            })

        xmid = float(np.median(xs))
        ymid = float(np.median(ys))
        xmin, xmax = _minmax(xs)
        ymin, ymax = _minmax(ys)

        shapes = [
            create_shape("line", {"x0": xmid, "y0": ymin - 0.05, "x1": xmid, "y1": ymax + 0.05},
                         line_dash="dot", line_color="#888888"),
            create_shape("line", {"x0": xmin - 0.05, "y0": ymid, "x1": xmax + 0.05, "y1": ymid},
                         line_dash="dot", line_color="#888888"),
        ]
        annotations = [
            create_annotation("High Q + High D", xmax, ymax, showarrow=False,
                              font_size=10, font_color="#27ae60", xref="x", yref="y"),
            create_annotation("High Q + Low D", xmin, ymax, showarrow=False,
                              font_size=10, font_color="#e67e22", xref="x", yref="y"),
            create_annotation("Low Q + High D", xmax, ymin, showarrow=False,
                              font_size=10, font_color="#e67e22", xref="x", yref="y"),
            create_annotation("Low Q + Low D", xmin, ymin, showarrow=False,
                              font_size=10, font_color="#e74c3c", xref="x", yref="y"),
        ]

        layout = self._base_layout(
            title={"text": self.config.title or "Diversity vs Quality",
                   "font": {"size": self.config.title_font_size}},
            xaxis={"title": {"text": self.config.xlabel or "Diversity Score"}, "showgrid": True, "gridcolor": "#eee"},
            yaxis={"title": {"text": self.config.ylabel or "Quality Score"}, "showgrid": True, "gridcolor": "#eee"},
            shapes=shapes,
            annotations=annotations,
        )
        return {"data": traces, "layout": layout}

    # ----- 2e. Hyperparameter surface ----------------------------------------
    def hyperparameter_surface(
        self,
        param_values: Dict[str, Sequence[Number]],
        metric_values: np.ndarray,
    ) -> PlotSpec:
        """3-D surface or contour plot over a 2-D parameter grid.

        Parameters
        ----------
        param_values : dict with exactly two keys mapping to 1-D arrays
        metric_values : 2-D array shaped (len(p1), len(p2))
        """
        keys = list(param_values.keys())
        if len(keys) < 2:
            raise ValueError("param_values must have at least two parameter names")

        p1_name, p2_name = keys[0], keys[1]
        p1 = _ensure_list(param_values[p1_name])
        p2 = _ensure_list(param_values[p2_name])
        z = np.asarray(metric_values, dtype=float).tolist()

        surface_trace: Dict[str, Any] = {
            "type": "surface",
            "x": p1,
            "y": p2,
            "z": z,
            "colorscale": "Viridis",
            "colorbar": {"title": "Metric", "titleside": "right"},
            "hovertemplate": (
                f"{p1_name}: %{{x}}<br>{p2_name}: %{{y}}<br>Metric: %{{z:.4f}}<extra></extra>"
            ),
        }

        contour_trace: Dict[str, Any] = {
            "type": "contour",
            "x": p1,
            "y": p2,
            "z": z,
            "colorscale": "Viridis",
            "contours": {"showlabels": True, "labelfont": {"size": 10, "color": "white"}},
            "colorbar": {"title": "Metric", "titleside": "right"},
            "visible": False,
            "hovertemplate": (
                f"{p1_name}: %{{x}}<br>{p2_name}: %{{y}}<br>Metric: %{{z:.4f}}<extra></extra>"
            ),
        }

        layout = self._base_layout(
            title={"text": self.config.title or "Hyperparameter Surface",
                   "font": {"size": self.config.title_font_size}},
            scene={
                "xaxis": {"title": p1_name},
                "yaxis": {"title": p2_name},
                "zaxis": {"title": "Metric Value"},
            },
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": 1.12,
                "buttons": [
                    {"label": "3D Surface", "method": "update",
                     "args": [{"visible": [True, False]}]},
                    {"label": "Contour", "method": "update",
                     "args": [{"visible": [False, True]}]},
                ],
            }],
        )
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)

        return {"data": [surface_trace, contour_trace], "layout": layout}

    # ----- 2f. Convergence plot ----------------------------------------------
    def convergence_plot(
        self,
        iterations: Sequence[int],
        values: Dict[str, Sequence[Number]],
        labels: Optional[Dict[str, str]] = None,
    ) -> PlotSpec:
        """Line plot showing metric convergence over iterations.

        Parameters
        ----------
        iterations : x-axis values
        values : {series_name: y_values}
        labels : optional {series_name: display_name}
        """
        iters = _ensure_list(iterations)
        colors = self._colors(len(values))
        traces: List[Dict[str, Any]] = []

        for idx, (name, vals) in enumerate(values.items()):
            yv = _ensure_list(vals)
            display = (labels or {}).get(name, name)
            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "x": iters[:len(yv)],
                "y": yv,
                "name": display,
                "line": {"color": colors[idx], "width": 2},
                "marker": {"color": colors[idx], "size": 4},
                "hovertemplate": f"{display}<br>Iter: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            })

        layout = self._base_layout(
            title={"text": self.config.title or "Convergence Plot",
                   "font": {"size": self.config.title_font_size}},
            xaxis={"title": {"text": self.config.xlabel or "Iteration"}, "showgrid": True, "gridcolor": "#eee"},
            yaxis={"title": {"text": self.config.ylabel or "Metric Value"}, "showgrid": True, "gridcolor": "#eee"},
        )
        return {"data": traces, "layout": layout}

    # ----- 2g. Box-plot comparison -------------------------------------------
    def box_plot_comparison(
        self,
        algorithm_results: Dict[str, Sequence[Number]],
    ) -> PlotSpec:
        """Side-by-side box plots for algorithm comparison."""
        colors = self._colors(len(algorithm_results))
        traces: List[Dict[str, Any]] = []

        for idx, (algo, results) in enumerate(algorithm_results.items()):
            vals = _ensure_list(results)
            traces.append({
                "type": "box",
                "y": vals,
                "name": algo,
                "boxpoints": "outliers",
                "marker": {"color": colors[idx], "outliercolor": colors[idx]},
                "line": {"color": colors[idx]},
                "fillcolor": colors[idx] + "55",
                "hoverinfo": "y+name",
            })

        layout = self._base_layout(
            title={"text": self.config.title or "Algorithm Comparison (Box Plot)",
                   "font": {"size": self.config.title_font_size}},
            xaxis={"title": {"text": self.config.xlabel or "Algorithm"}},
            yaxis={"title": {"text": self.config.ylabel or "Score"}},
            boxmode="group",
        )
        return {"data": traces, "layout": layout}

    # ----- 2h. Violin-plot comparison ----------------------------------------
    def violin_plot_comparison(
        self,
        algorithm_results: Dict[str, Sequence[Number]],
    ) -> PlotSpec:
        """Violin plots for distributional comparison."""
        colors = self._colors(len(algorithm_results))
        traces: List[Dict[str, Any]] = []

        for idx, (algo, results) in enumerate(algorithm_results.items()):
            vals = _ensure_list(results)
            traces.append({
                "type": "violin",
                "y": vals,
                "name": algo,
                "box": {"visible": True},
                "meanline": {"visible": True},
                "line": {"color": colors[idx]},
                "fillcolor": colors[idx] + "55",
                "points": "all",
                "jitter": 0.3,
                "pointpos": -1.5,
                "hoverinfo": "y+name",
            })

        layout = self._base_layout(
            title={"text": self.config.title or "Algorithm Comparison (Violin)",
                   "font": {"size": self.config.title_font_size}},
            xaxis={"title": {"text": self.config.xlabel or "Algorithm"}},
            yaxis={"title": {"text": self.config.ylabel or "Score"}},
            violinmode="group",
        )
        return {"data": traces, "layout": layout}

    # ----- 2i. Parallel coordinates ------------------------------------------
    def parallel_coordinates(
        self,
        data: Dict[str, Sequence[Number]],
        dimensions: Sequence[str],
        labels: Optional[Sequence[str]] = None,
    ) -> PlotSpec:
        """Parallel-coordinates plot.

        Parameters
        ----------
        data : {dimension_name: values}  – each value list has length N
        dimensions : ordered dimension names to display
        labels : optional colour-group labels of length N
        """
        dims_spec: List[Dict[str, Any]] = []
        for dim in dimensions:
            vals = _ensure_list(data[dim])
            lo, hi = _minmax(vals)
            pad = (hi - lo) * 0.05 if hi > lo else 0.5
            dims_spec.append({
                "range": [lo - pad, hi + pad],
                "label": dim,
                "values": vals,
            })

        n = len(_ensure_list(data[dimensions[0]]))
        if labels is not None:
            unique_labels = list(OrderedDict.fromkeys(labels))
            label_to_int = {l: i for i, l in enumerate(unique_labels)}
            color_vals = [label_to_int[l] for l in labels]
            n_groups = len(unique_labels)
            colorscale = create_colorscale(n_groups, self.config.color_scheme)
            line_spec: Dict[str, Any] = {
                "color": color_vals,
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {"title": "Group", "tickvals": list(range(n_groups)),
                             "ticktext": unique_labels},
            }
        else:
            line_spec = {"color": "#1f77b4"}

        trace: Dict[str, Any] = {
            "type": "parcoords",
            "line": line_spec,
            "dimensions": dims_spec,
        }

        layout = self._base_layout(
            title={"text": self.config.title or "Parallel Coordinates",
                   "font": {"size": self.config.title_font_size}},
        )
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)

        return {"data": [trace], "layout": layout}

    # ----- 2j. Sunburst chart ------------------------------------------------
    def sunburst_chart(
        self,
        hierarchy_data: Dict[str, Any],
    ) -> PlotSpec:
        """Sunburst chart for hierarchical data.

        Parameters
        ----------
        hierarchy_data : dict with keys ``ids``, ``labels``, ``parents``,
            and optionally ``values``.
        """
        ids = hierarchy_data.get("ids", [])
        lab = hierarchy_data.get("labels", ids)
        parents = hierarchy_data.get("parents", [""] * len(ids))
        values = hierarchy_data.get("values", [1] * len(ids))

        trace: Dict[str, Any] = {
            "type": "sunburst",
            "ids": list(ids),
            "labels": list(lab),
            "parents": list(parents),
            "values": list(values),
            "branchvalues": "total",
            "hovertemplate": "<b>%{label}</b><br>Value: %{value}<extra></extra>",
            "textinfo": "label+percent parent",
            "insidetextorientation": "radial",
        }

        layout = self._base_layout(
            title={"text": self.config.title or "Sunburst Chart",
                   "font": {"size": self.config.title_font_size}},
        )
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)

        return {"data": [trace], "layout": layout}


# ========================================================================= #
#  3.  DashboardGenerator                                                   #
# ========================================================================= #


class DashboardGenerator:
    """Generates multi-plot dashboard layouts as plain dicts.

    Each dashboard is a dict with ``plots`` (list of PlotSpecs) and
    ``layout`` metadata describing grid positions, sizing, etc.
    """

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()
        self._plotter = InteractivePlotter(self.config)
        self._style = PlotStyleManager()

    # ---- layout helpers -----------------------------------------------------
    @staticmethod
    def _grid_position(
        index: int, cols: int = 2
    ) -> Dict[str, Any]:
        row = index // cols
        col = index % cols
        return {"row": row, "col": col, "index": index}

    @staticmethod
    def _dashboard_shell(
        title: str,
        n_plots: int,
        cols: int = 2,
        plot_width: int = 550,
        plot_height: int = 400,
    ) -> Dict[str, Any]:
        rows = math.ceil(n_plots / cols)
        return {
            "dashboard_title": title,
            "grid": {"rows": rows, "cols": cols},
            "total_width": cols * plot_width,
            "total_height": rows * plot_height,
            "plot_width": plot_width,
            "plot_height": plot_height,
            "plots": [],
        }

    # ---- 3a. Overview dashboard ---------------------------------------------
    def generate_overview_dashboard(
        self,
        experiment_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Top-level overview with radar, scatter, heatmap, box, convergence, sunburst.

        Parameters
        ----------
        experiment_results : dict with keys
            ``algorithms`` : {algo: {metric: [values]}}
            ``metric_names`` : list of metric names
            ``iterations``  : optional iteration list
            ``convergence`` : optional {algo: [values]}
            ``hierarchy``   : optional sunburst hierarchy dict
            ``correlation_matrix`` : optional np.ndarray
        """
        algos: Dict[str, Dict[str, list]] = experiment_results.get("algorithms", {})
        metric_names: List[str] = experiment_results.get("metric_names", [])
        dashboard = self._dashboard_shell("Experiment Overview", 6, cols=3)

        # 1) radar chart: mean score per algo per metric
        algo_means: Dict[str, List[float]] = {}
        for algo, metrics in algos.items():
            algo_means[algo] = [float(np.mean(metrics.get(m, [0]))) for m in metric_names]
        # normalise to [0,1]
        if algo_means:
            all_vals = np.array(list(algo_means.values()))
            mins = all_vals.min(axis=0)
            maxs = all_vals.max(axis=0)
            ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
            algo_norm = {a: ((np.array(v) - mins) / ranges).tolist() for a, v in algo_means.items()}
        else:
            algo_norm = {}

        cfg_radar = self.config.copy(title="Algorithm Radar", width=550, height=400)
        plotter_r = InteractivePlotter(cfg_radar)
        radar = plotter_r.algorithm_radar_chart(algo_norm, metric_names)
        radar["_grid"] = self._grid_position(0, 3)
        dashboard["plots"].append(radar)

        # 2) box plot comparison for first metric
        if metric_names and algos:
            first_metric = metric_names[0]
            box_data = {a: m.get(first_metric, []) for a, m in algos.items()}
            cfg_box = self.config.copy(title=f"Box Plot – {first_metric}", width=550, height=400)
            plotter_b = InteractivePlotter(cfg_box)
            box = plotter_b.box_plot_comparison(box_data)
            box["_grid"] = self._grid_position(1, 3)
            dashboard["plots"].append(box)

        # 3) heatmap
        corr = experiment_results.get("correlation_matrix")
        if corr is not None:
            cfg_hm = self.config.copy(title="Metric Correlations", width=550, height=400)
            plotter_h = InteractivePlotter(cfg_hm)
            hm = plotter_h.metric_heatmap(np.asarray(corr), metric_names)
            hm["_grid"] = self._grid_position(2, 3)
            dashboard["plots"].append(hm)

        # 4) diversity-quality scatter (use first two metrics as proxies)
        if len(metric_names) >= 2 and algos:
            div_scores: List[float] = []
            qual_scores: List[float] = []
            scatter_labels: List[str] = []
            for a, m in algos.items():
                div_scores.append(float(np.mean(m.get(metric_names[0], [0]))))
                qual_scores.append(float(np.mean(m.get(metric_names[1], [0]))))
                scatter_labels.append(a)
            cfg_s = self.config.copy(title="Diversity vs Quality", width=550, height=400,
                                     xlabel=metric_names[0], ylabel=metric_names[1])
            plotter_s = InteractivePlotter(cfg_s)
            sc = plotter_s.diversity_quality_scatter(div_scores, qual_scores, scatter_labels)
            sc["_grid"] = self._grid_position(3, 3)
            dashboard["plots"].append(sc)

        # 5) convergence
        conv = experiment_results.get("convergence")
        iters = experiment_results.get("iterations")
        if conv and iters:
            cfg_c = self.config.copy(title="Convergence", width=550, height=400)
            plotter_c = InteractivePlotter(cfg_c)
            cp = plotter_c.convergence_plot(iters, conv)
            cp["_grid"] = self._grid_position(4, 3)
            dashboard["plots"].append(cp)

        # 6) sunburst
        hier = experiment_results.get("hierarchy")
        if hier:
            cfg_sb = self.config.copy(title="Experiment Hierarchy", width=550, height=400)
            plotter_sb = InteractivePlotter(cfg_sb)
            sb = plotter_sb.sunburst_chart(hier)
            sb["_grid"] = self._grid_position(5, 3)
            dashboard["plots"].append(sb)

        dashboard["grid"]["rows"] = math.ceil(len(dashboard["plots"]) / 3)
        dashboard["total_height"] = dashboard["grid"]["rows"] * 400
        return dashboard

    # ---- 3b. Algorithm dashboard -------------------------------------------
    def generate_algorithm_dashboard(
        self,
        algorithm_name: str,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Per-algorithm detail dashboard.

        Parameters
        ----------
        results : dict with keys
            ``metrics`` : {metric: [values]}
            ``convergence`` : {metric: [values per iter]}
            ``iterations`` : list[int]
            ``hyperparameters`` : optional {(p1, p2): metric_grid}
        """
        metrics: Dict[str, list] = results.get("metrics", {})
        metric_names = list(metrics.keys())
        dashboard = self._dashboard_shell(f"Dashboard – {algorithm_name}", 4, cols=2)

        # 1) violin per metric
        if metrics:
            cfg_v = self.config.copy(title=f"{algorithm_name} – Score Distribution", width=550, height=400)
            plotter_v = InteractivePlotter(cfg_v)
            violin = plotter_v.violin_plot_comparison(metrics)
            violin["_grid"] = self._grid_position(0, 2)
            dashboard["plots"].append(violin)

        # 2) box per metric
        if metrics:
            cfg_b = self.config.copy(title=f"{algorithm_name} – Box Plot", width=550, height=400)
            plotter_b = InteractivePlotter(cfg_b)
            box = plotter_b.box_plot_comparison(metrics)
            box["_grid"] = self._grid_position(1, 2)
            dashboard["plots"].append(box)

        # 3) convergence
        conv = results.get("convergence")
        iters = results.get("iterations")
        if conv and iters:
            cfg_c = self.config.copy(title=f"{algorithm_name} – Convergence", width=550, height=400)
            plotter_c = InteractivePlotter(cfg_c)
            cp = plotter_c.convergence_plot(iters, conv)
            cp["_grid"] = self._grid_position(2, 2)
            dashboard["plots"].append(cp)

        # 4) hyperparameter surface
        hp = results.get("hyperparameters")
        if hp:
            param_names = list(hp.keys())
            if len(param_names) >= 2:
                p1, p2 = param_names[0], param_names[1]
                p1v = _ensure_list(hp[p1])
                p2v = _ensure_list(hp[p2])
                grid = results.get("hyperparameter_grid")
                if grid is not None:
                    cfg_hp = self.config.copy(title=f"{algorithm_name} – Hyperparameters", width=550, height=400)
                    plotter_hp = InteractivePlotter(cfg_hp)
                    surf = plotter_hp.hyperparameter_surface({p1: p1v, p2: p2v}, grid)
                    surf["_grid"] = self._grid_position(3, 2)
                    dashboard["plots"].append(surf)

        dashboard["grid"]["rows"] = math.ceil(len(dashboard["plots"]) / 2)
        dashboard["total_height"] = dashboard["grid"]["rows"] * 400
        return dashboard

    # ---- 3c. Metric dashboard ----------------------------------------------
    def generate_metric_dashboard(
        self,
        metric_name: str,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Per-metric detail dashboard.

        Parameters
        ----------
        results : dict with
            ``algorithms`` : {algo: [metric_values]}
            ``correlation_matrix`` : ndarray  (all metrics)
            ``metric_names`` : list
        """
        algos: Dict[str, list] = results.get("algorithms", {})
        dashboard = self._dashboard_shell(f"Metric Dashboard – {metric_name}", 3, cols=2)

        # 1) violin
        if algos:
            cfg_v = self.config.copy(title=f"{metric_name} – Distribution per Algorithm", width=550, height=400)
            plotter_v = InteractivePlotter(cfg_v)
            violin = plotter_v.violin_plot_comparison(algos)
            violin["_grid"] = self._grid_position(0, 2)
            dashboard["plots"].append(violin)

        # 2) box
        if algos:
            cfg_b = self.config.copy(title=f"{metric_name} – Box Plot", width=550, height=400)
            plotter_b = InteractivePlotter(cfg_b)
            box = plotter_b.box_plot_comparison(algos)
            box["_grid"] = self._grid_position(1, 2)
            dashboard["plots"].append(box)

        # 3) heatmap
        corr = results.get("correlation_matrix")
        mnames = results.get("metric_names")
        if corr is not None and mnames:
            cfg_h = self.config.copy(title="Metric Correlation", width=550, height=400)
            plotter_h = InteractivePlotter(cfg_h)
            hm = plotter_h.metric_heatmap(np.asarray(corr), mnames)
            hm["_grid"] = self._grid_position(2, 2)
            dashboard["plots"].append(hm)

        dashboard["grid"]["rows"] = math.ceil(max(len(dashboard["plots"]), 1) / 2)
        dashboard["total_height"] = dashboard["grid"]["rows"] * 400
        return dashboard


# ========================================================================= #
#  4.  AnimationGenerator                                                   #
# ========================================================================= #


class AnimationGenerator:
    """Generates Plotly-compatible animation specs (frames + sliders)."""

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()
        self._style = PlotStyleManager()

    def _colors(self, n: int) -> List[str]:
        return self._style.get_colors(n, self.config.color_scheme)

    @staticmethod
    def _slider_step(label: str, frame_name: str) -> Dict[str, Any]:
        return {
            "args": [[frame_name], {"frame": {"duration": 500, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 300}}],
            "label": label,
            "method": "animate",
        }

    def _animation_buttons(self) -> List[Dict[str, Any]]:
        return [{
            "type": "buttons",
            "showactive": False,
            "x": 0.05,
            "y": 1.12,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 300}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]},
            ],
        }]

    # ---- 4a. Beam-search animation ------------------------------------------
    def beam_search_animation(
        self,
        beam_states: List[Dict[str, Any]],
    ) -> PlotSpec:
        """Animate beam search as a tree expanding step by step.

        Parameters
        ----------
        beam_states : list of dicts, one per step, each containing:
            ``nodes`` : list of {id, x, y, label, score}
            ``edges`` : list of {source, target}
        """
        if not beam_states:
            return {"data": [], "layout": {}, "frames": []}

        all_xs: List[float] = []
        all_ys: List[float] = []
        for state in beam_states:
            for n in state.get("nodes", []):
                all_xs.append(float(n["x"]))
                all_ys.append(float(n["y"]))

        x_pad = 0.5
        y_pad = 0.5
        xr = [min(all_xs, default=0) - x_pad, max(all_xs, default=1) + x_pad]
        yr = [min(all_ys, default=0) - y_pad, max(all_ys, default=1) + y_pad]

        frames: List[Dict[str, Any]] = []
        slider_steps: List[Dict[str, Any]] = []

        for step_idx, state in enumerate(beam_states):
            nodes = state.get("nodes", [])
            edges = state.get("edges", [])
            node_map = {n["id"]: n for n in nodes}

            # edges as line segments
            edge_x: List[Optional[float]] = []
            edge_y: List[Optional[float]] = []
            for e in edges:
                src = node_map.get(e["source"])
                tgt = node_map.get(e["target"])
                if src and tgt:
                    edge_x += [float(src["x"]), float(tgt["x"]), None]
                    edge_y += [float(src["y"]), float(tgt["y"]), None]

            edge_trace: Dict[str, Any] = {
                "type": "scatter",
                "x": edge_x,
                "y": edge_y,
                "mode": "lines",
                "line": {"color": "#888888", "width": 1.5},
                "hoverinfo": "none",
                "name": "Edges",
            }

            scores = [float(n.get("score", 0)) for n in nodes]
            lo = min(scores) if scores else 0
            hi = max(scores) if scores else 1
            rng = hi - lo if hi > lo else 1.0
            norm_scores = [(s - lo) / rng for s in scores]

            colors_list = []
            for ns in norm_scores:
                r = int(255 * (1 - ns))
                g = int(255 * ns)
                colors_list.append(f"rgb({r},{g},80)")

            node_trace: Dict[str, Any] = {
                "type": "scatter",
                "x": [float(n["x"]) for n in nodes],
                "y": [float(n["y"]) for n in nodes],
                "mode": "markers+text",
                "text": [n.get("label", str(n["id"])) for n in nodes],
                "textposition": "top center",
                "marker": {"size": 14, "color": colors_list,
                           "line": {"width": 1, "color": "#333"}},
                "hovertemplate": "Token: %{text}<br>Score: %{customdata:.3f}<extra></extra>",
                "customdata": scores,
                "name": "Nodes",
            }

            frame_name = f"step_{step_idx}"
            frames.append({
                "name": frame_name,
                "data": [edge_trace, node_trace],
            })
            slider_steps.append(self._slider_step(f"Step {step_idx}", frame_name))

        # initial data from first frame
        initial_data = frames[0]["data"] if frames else []

        layout: Dict[str, Any] = {
            "title": {"text": self.config.title or "Beam Search Animation",
                      "font": {"size": self.config.title_font_size}},
            "xaxis": {"range": xr, "showgrid": False, "zeroline": False,
                      "showticklabels": False},
            "yaxis": {"range": yr, "showgrid": False, "zeroline": False,
                      "showticklabels": False},
            "width": self.config.width,
            "height": self.config.height,
            "updatemenus": self._animation_buttons(),
            "sliders": [{
                "active": 0,
                "steps": slider_steps,
                "x": 0.1, "len": 0.8,
                "y": -0.05,
                "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
                "transition": {"duration": 300},
            }],
            "showlegend": False,
            "paper_bgcolor": self.config.background_color,
            "plot_bgcolor": self.config.plot_background_color,
        }

        return {"data": initial_data, "layout": layout, "frames": frames}

    # ---- 4b. Particle evolution animation -----------------------------------
    def particle_evolution_animation(
        self,
        particle_states: List[Dict[str, Any]],
    ) -> PlotSpec:
        """Animate SVD-guided particle swarm in 2-D projected space.

        Parameters
        ----------
        particle_states : list of dicts per generation, each with:
            ``positions`` : array-like (N, 2) – projected positions
            ``fitness``   : array-like (N,) – per-particle fitness
            ``best``      : optional (x, y) global best position
        """
        if not particle_states:
            return {"data": [], "layout": {}, "frames": []}

        all_pos = np.concatenate(
            [np.asarray(s["positions"]) for s in particle_states if len(s.get("positions", [])) > 0],
            axis=0,
        )
        xr = [float(all_pos[:, 0].min()) - 1, float(all_pos[:, 0].max()) + 1]
        yr = [float(all_pos[:, 1].min()) - 1, float(all_pos[:, 1].max()) + 1]

        frames: List[Dict[str, Any]] = []
        slider_steps: List[Dict[str, Any]] = []

        for gen, state in enumerate(particle_states):
            pos = np.asarray(state["positions"], dtype=float)
            fit = np.asarray(state.get("fitness", np.zeros(len(pos))), dtype=float)

            f_min, f_max = float(fit.min()), float(fit.max())
            f_range = f_max - f_min if f_max > f_min else 1.0
            norm_fit = ((fit - f_min) / f_range).tolist()

            particle_trace: Dict[str, Any] = {
                "type": "scatter",
                "x": pos[:, 0].tolist(),
                "y": pos[:, 1].tolist(),
                "mode": "markers",
                "marker": {
                    "size": 10,
                    "color": norm_fit,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Fitness"},
                    "opacity": 0.8,
                    "line": {"width": 0.5, "color": "#333"},
                },
                "hovertemplate": "x: %{x:.2f}<br>y: %{y:.2f}<br>Fitness: %{marker.color:.3f}<extra></extra>",
                "name": "Particles",
            }

            trace_list = [particle_trace]

            best = state.get("best")
            if best is not None:
                best_trace: Dict[str, Any] = {
                    "type": "scatter",
                    "x": [float(best[0])],
                    "y": [float(best[1])],
                    "mode": "markers",
                    "marker": {"size": 18, "color": "#e74c3c", "symbol": "star",
                               "line": {"width": 2, "color": "#c0392b"}},
                    "name": "Global Best",
                    "hovertemplate": "Best: (%{x:.2f}, %{y:.2f})<extra></extra>",
                }
                trace_list.append(best_trace)

            frame_name = f"gen_{gen}"
            frames.append({"name": frame_name, "data": trace_list})
            slider_steps.append(self._slider_step(f"Gen {gen}", frame_name))

        initial_data = frames[0]["data"] if frames else []

        layout: Dict[str, Any] = {
            "title": {"text": self.config.title or "Particle Evolution",
                      "font": {"size": self.config.title_font_size}},
            "xaxis": {"range": xr, "title": "PC-1", "showgrid": True, "gridcolor": "#eee"},
            "yaxis": {"range": yr, "title": "PC-2", "showgrid": True, "gridcolor": "#eee"},
            "width": self.config.width,
            "height": self.config.height,
            "updatemenus": self._animation_buttons(),
            "sliders": [{
                "active": 0,
                "steps": slider_steps,
                "x": 0.1, "len": 0.8,
                "y": -0.05,
                "currentvalue": {"prefix": "Generation: ", "visible": True, "xanchor": "center"},
                "transition": {"duration": 300},
            }],
            "paper_bgcolor": self.config.background_color,
            "plot_bgcolor": self.config.plot_background_color,
        }

        return {"data": initial_data, "layout": layout, "frames": frames}

    # ---- 4c. Archive filling animation --------------------------------------
    def archive_filling_animation(
        self,
        archive_states: List[Dict[str, Any]],
    ) -> PlotSpec:
        """Animate a QD-BS MAP-Elites style archive filling over time.

        Parameters
        ----------
        archive_states : list of dicts per snapshot:
            ``grid`` : 2-D array (rows x cols) of fitness (NaN = empty)
            ``x_bins`` : bin labels for x axis
            ``y_bins`` : bin labels for y axis
        """
        if not archive_states:
            return {"data": [], "layout": {}, "frames": []}

        frames: List[Dict[str, Any]] = []
        slider_steps: List[Dict[str, Any]] = []

        vmin = float("inf")
        vmax = float("-inf")
        for s in archive_states:
            g = np.asarray(s["grid"], dtype=float)
            valid = g[~np.isnan(g)]
            if len(valid) > 0:
                vmin = min(vmin, float(valid.min()))
                vmax = max(vmax, float(valid.max()))
        if vmin == float("inf"):
            vmin, vmax = 0.0, 1.0

        for snap_idx, state in enumerate(archive_states):
            grid = np.asarray(state["grid"], dtype=float)
            x_bins = state.get("x_bins", list(range(grid.shape[1])))
            y_bins = state.get("y_bins", list(range(grid.shape[0])))

            display_grid = np.where(np.isnan(grid), None, grid).tolist()
            # replace Python None with None (JSON null)
            z_clean = []
            for row in display_grid:
                z_clean.append([v if v is not None else None for v in row])

            n_filled = int(np.count_nonzero(~np.isnan(grid)))
            total_cells = grid.size

            heatmap_trace: Dict[str, Any] = {
                "type": "heatmap",
                "z": z_clean,
                "x": _ensure_list(x_bins),
                "y": _ensure_list(y_bins),
                "colorscale": "Viridis",
                "zmin": vmin,
                "zmax": vmax,
                "colorbar": {"title": "Fitness"},
                "hovertemplate": "x: %{x}<br>y: %{y}<br>Fitness: %{z}<extra></extra>",
            }

            annotation: Dict[str, Any] = {
                "text": f"Filled: {n_filled}/{total_cells} ({100 * n_filled / max(total_cells, 1):.1f}%)",
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 13},
            }

            frame_name = f"snap_{snap_idx}"
            frames.append({
                "name": frame_name,
                "data": [heatmap_trace],
                "layout": {"annotations": [annotation]},
            })
            slider_steps.append(self._slider_step(f"Snap {snap_idx}", frame_name))

        initial_data = frames[0]["data"] if frames else []

        layout: Dict[str, Any] = {
            "title": {"text": self.config.title or "Archive Filling",
                      "font": {"size": self.config.title_font_size}},
            "xaxis": {"title": "Behaviour Dim 1", "showgrid": False},
            "yaxis": {"title": "Behaviour Dim 2", "showgrid": False, "autorange": "reversed"},
            "width": self.config.width,
            "height": self.config.height,
            "updatemenus": self._animation_buttons(),
            "sliders": [{
                "active": 0,
                "steps": slider_steps,
                "x": 0.1, "len": 0.8,
                "y": -0.05,
                "currentvalue": {"prefix": "Snapshot: ", "visible": True, "xanchor": "center"},
                "transition": {"duration": 300},
            }],
            "paper_bgcolor": self.config.background_color,
            "plot_bgcolor": self.config.plot_background_color,
            "annotations": frames[0].get("layout", {}).get("annotations", []) if frames else [],
        }

        return {"data": initial_data, "layout": layout, "frames": frames}

    # ---- 4d. Metric evolution animation -------------------------------------
    def metric_evolution_animation(
        self,
        metric_history: Dict[str, List[List[Number]]],
    ) -> PlotSpec:
        """Animate metric curves growing over time.

        Parameters
        ----------
        metric_history : {metric_name: [[step0_vals], [step1_vals], …]}
            Each inner list is the cumulative series up to that animation frame.
        """
        metric_names = list(metric_history.keys())
        if not metric_names:
            return {"data": [], "layout": {}, "frames": []}

        n_frames = max(len(v) for v in metric_history.values())
        colors = self._colors(len(metric_names))

        frames: List[Dict[str, Any]] = []
        slider_steps: List[Dict[str, Any]] = []

        # compute y-range across all data
        all_vals: List[float] = []
        for series_list in metric_history.values():
            for series in series_list:
                all_vals.extend([float(v) for v in series])
        if all_vals:
            y_lo, y_hi = min(all_vals), max(all_vals)
            pad = (y_hi - y_lo) * 0.1 if y_hi > y_lo else 0.5
            y_range = [y_lo - pad, y_hi + pad]
        else:
            y_range = [0, 1]

        for fi in range(n_frames):
            trace_list: List[Dict[str, Any]] = []
            for mi, mname in enumerate(metric_names):
                series_list = metric_history[mname]
                if fi < len(series_list):
                    vals = _ensure_list(series_list[fi])
                else:
                    vals = _ensure_list(series_list[-1]) if series_list else []
                trace_list.append({
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": list(range(len(vals))),
                    "y": vals,
                    "name": mname,
                    "line": {"color": colors[mi], "width": 2},
                    "marker": {"color": colors[mi], "size": 4},
                })

            frame_name = f"frame_{fi}"
            frames.append({"name": frame_name, "data": trace_list})
            slider_steps.append(self._slider_step(str(fi), frame_name))

        initial_data = frames[0]["data"] if frames else []

        layout: Dict[str, Any] = {
            "title": {"text": self.config.title or "Metric Evolution",
                      "font": {"size": self.config.title_font_size}},
            "xaxis": {"title": "Step", "showgrid": True, "gridcolor": "#eee",
                      "range": [0, max(len(s) for sl in metric_history.values() for s in sl) if metric_history else 1]},
            "yaxis": {"title": "Value", "showgrid": True, "gridcolor": "#eee",
                      "range": y_range},
            "width": self.config.width,
            "height": self.config.height,
            "updatemenus": self._animation_buttons(),
            "sliders": [{
                "active": 0,
                "steps": slider_steps,
                "x": 0.1, "len": 0.8,
                "y": -0.05,
                "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
                "transition": {"duration": 300},
            }],
            "legend": {"x": 0.98, "y": 0.98, "xanchor": "right", "yanchor": "top"},
            "paper_bgcolor": self.config.background_color,
            "plot_bgcolor": self.config.plot_background_color,
        }

        return {"data": initial_data, "layout": layout, "frames": frames}


# ========================================================================= #
#  5.  StaticPlotter                                                        #
# ========================================================================= #


class StaticPlotter:
    """Generates matplotlib-style plot configuration dicts.

    Each method returns a dict describing the plot in terms that can be
    directly consumed by ``matplotlib`` (axes, series, annotations, etc.)
    or serialised for deferred rendering.
    """

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()
        self._style = PlotStyleManager()

    def _colors(self, n: int) -> List[str]:
        return self._style.get_colors(n, self.config.color_scheme)

    # ---- 5a. Critical-difference diagram -----------------------------------
    def cd_diagram(
        self,
        ranks: Dict[str, float],
        names: Sequence[str],
        cd_value: float,
    ) -> PlotSpec:
        """Critical-difference diagram (Demšar, 2006).

        Parameters
        ----------
        ranks : {algorithm_name: mean_rank}
        names : display order
        cd_value : critical difference threshold
        """
        ordered = sorted(names, key=lambda n: ranks.get(n, 0))
        n = len(ordered)

        # figure dimensions
        fig_w = max(8, n * 0.9)
        fig_h = max(3, n * 0.35)
        rank_min = min(ranks.values()) if ranks else 1
        rank_max = max(ranks.values()) if ranks else n

        # clique detection: groups not significantly different
        cliques: List[List[str]] = []
        for i in range(n):
            group = [ordered[i]]
            for j in range(i + 1, n):
                if abs(ranks[ordered[j]] - ranks[ordered[i]]) < cd_value:
                    group.append(ordered[j])
            if len(group) > 1:
                # only keep if not a subset of existing clique
                is_subset = any(set(group).issubset(set(c)) for c in cliques)
                if not is_subset:
                    cliques.append(group)

        # build lines connecting algorithms in same clique
        clique_lines: List[Dict[str, Any]] = []
        for ci, clique in enumerate(cliques):
            r_vals = [ranks[a] for a in clique]
            clique_lines.append({
                "x_start": min(r_vals),
                "x_end": max(r_vals),
                "y": -0.15 * (ci + 1),
                "linewidth": 3,
                "color": "#333333",
            })

        # tick marks for each algorithm
        ticks: List[Dict[str, Any]] = []
        half = n // 2
        for idx, alg in enumerate(ordered):
            r = ranks[alg]
            side = "left" if idx < half else "right"
            ticks.append({
                "name": alg,
                "rank": r,
                "side": side,
                "y_offset": (idx % half) * 0.12 + 0.1,
            })

        return {
            "type": "cd_diagram",
            "renderer": "matplotlib",
            "figsize": [fig_w, fig_h],
            "dpi": self.config.dpi,
            "title": self.config.title or "Critical Difference Diagram",
            "rank_range": [rank_min - 0.5, rank_max + 0.5],
            "cd_value": cd_value,
            "cd_bar": {
                "x_center": (rank_min + rank_max) / 2,
                "y": 0.25,
                "half_width": cd_value / 2,
                "label": f"CD = {cd_value:.2f}",
            },
            "ticks": ticks,
            "clique_lines": clique_lines,
            "axis_label": "Average Rank",
            "font": {"family": self.config.font_family, "size": self.config.font_size},
        }

    # ---- 5b. Pareto frontier 2-D -------------------------------------------
    def pareto_frontier_2d(
        self,
        points: np.ndarray,
    ) -> PlotSpec:
        """Static 2-D Pareto frontier plot config for matplotlib.

        Parameters
        ----------
        points : (N, 2) array
        """
        pts = np.asarray(points, dtype=float)
        xs = pts[:, 0].tolist()
        ys = pts[:, 1].tolist()

        # compute frontier (maximisation on both objectives)
        sorted_idx = np.argsort(-pts[:, 0])
        frontier_idx: List[int] = []
        best_y = -float("inf")
        for i in sorted_idx:
            if pts[i, 1] > best_y:
                frontier_idx.append(int(i))
                best_y = pts[i, 1]
        frontier_idx.sort(key=lambda i: pts[i, 0])

        colors = self._colors(2)

        dominated_x = [xs[i] for i in range(len(xs)) if i not in frontier_idx]
        dominated_y = [ys[i] for i in range(len(ys)) if i not in frontier_idx]
        frontier_x = [xs[i] for i in frontier_idx]
        frontier_y = [ys[i] for i in frontier_idx]

        return {
            "type": "pareto_2d",
            "renderer": "matplotlib",
            "figsize": [self.config.width / 100, self.config.height / 100],
            "dpi": self.config.dpi,
            "title": self.config.title or "Pareto Frontier",
            "xlabel": self.config.xlabel or "Objective 1",
            "ylabel": self.config.ylabel or "Objective 2",
            "series": [
                {
                    "x": dominated_x,
                    "y": dominated_y,
                    "marker": "o",
                    "color": colors[0],
                    "alpha": 0.5,
                    "label": "Dominated",
                    "markersize": 6,
                },
                {
                    "x": frontier_x,
                    "y": frontier_y,
                    "marker": "*",
                    "color": colors[1],
                    "alpha": 1.0,
                    "label": "Pareto Frontier",
                    "markersize": 12,
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "connect": True,
                },
            ],
            "grid": self.config.show_grid,
            "font": {"family": self.config.font_family, "size": self.config.font_size},
        }

    # ---- 5c. Correlation heatmap -------------------------------------------
    def metric_correlation_heatmap(
        self,
        matrix: np.ndarray,
        metric_names: Optional[Sequence[str]] = None,
    ) -> PlotSpec:
        """Heatmap config for matplotlib ``imshow``."""
        mat = np.asarray(matrix, dtype=float)
        n = mat.shape[0]
        names = list(metric_names) if metric_names else [f"M{i}" for i in range(n)]
        annotations: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(n):
                annotations.append({
                    "x": j,
                    "y": i,
                    "text": f"{mat[i, j]:.2f}",
                    "color": "white" if abs(mat[i, j]) > 0.5 else "black",
                    "fontsize": max(8, self.config.font_size - 4),
                })

        return {
            "type": "heatmap",
            "renderer": "matplotlib",
            "figsize": [max(6, n * 0.8), max(5, n * 0.7)],
            "dpi": self.config.dpi,
            "title": self.config.title or "Metric Correlation",
            "data": mat.tolist(),
            "cmap": "RdBu_r",
            "vmin": -1,
            "vmax": 1,
            "xticklabels": names,
            "yticklabels": names,
            "annotations": annotations,
            "colorbar": {"label": "Correlation"},
            "font": {"family": self.config.font_family, "size": self.config.font_size},
        }

    # ---- 5d. Distribution comparison ----------------------------------------
    def distribution_comparison(
        self,
        distributions: Dict[str, Sequence[Number]],
        labels: Optional[Dict[str, str]] = None,
    ) -> PlotSpec:
        """Overlapping histograms + KDE curves for distributions."""
        colors = self._colors(len(distributions))
        series: List[Dict[str, Any]] = []
        for idx, (name, vals) in enumerate(distributions.items()):
            arr = np.asarray(vals, dtype=float)
            display_name = (labels or {}).get(name, name)

            # histogram bins
            n_bins = max(10, int(np.sqrt(len(arr))))
            counts, bin_edges = np.histogram(arr, bins=n_bins, density=True)

            # simple KDE via Gaussian convolution
            x_kde = np.linspace(float(arr.min()) - 1, float(arr.max()) + 1, 200)
            bw = 1.06 * float(np.std(arr)) * len(arr) ** (-0.2) if len(arr) > 1 else 1.0
            kde_vals = np.zeros_like(x_kde)
            for v in arr:
                kde_vals += np.exp(-0.5 * ((x_kde - v) / bw) ** 2) / (bw * math.sqrt(2 * math.pi))
            kde_vals /= max(len(arr), 1)

            series.append({
                "name": display_name,
                "color": colors[idx],
                "histogram": {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                    "alpha": 0.35,
                },
                "kde": {
                    "x": x_kde.tolist(),
                    "y": kde_vals.tolist(),
                    "linewidth": 2,
                },
                "stats": {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "n": len(arr),
                },
            })

        return {
            "type": "distribution_comparison",
            "renderer": "matplotlib",
            "figsize": [self.config.width / 100, self.config.height / 100],
            "dpi": self.config.dpi,
            "title": self.config.title or "Distribution Comparison",
            "xlabel": self.config.xlabel or "Value",
            "ylabel": self.config.ylabel or "Density",
            "series": series,
            "grid": self.config.show_grid,
            "font": {"family": self.config.font_family, "size": self.config.font_size},
        }

    # ---- 5e. Error-bar plot -------------------------------------------------
    def error_bar_plot(
        self,
        means: Sequence[Number],
        stds: Sequence[Number],
        labels: Sequence[str],
    ) -> PlotSpec:
        """Horizontal bar chart with error bars."""
        n = len(labels)
        colors = self._colors(n)
        mean_list = _ensure_list(means)
        std_list = _ensure_list(stds)
        label_list = list(labels)

        bars: List[Dict[str, Any]] = []
        for i in range(n):
            bars.append({
                "label": label_list[i],
                "mean": float(mean_list[i]),
                "std": float(std_list[i]),
                "color": colors[i],
                "ci_lower": float(mean_list[i] - 1.96 * std_list[i]),
                "ci_upper": float(mean_list[i] + 1.96 * std_list[i]),
            })

        return {
            "type": "error_bar",
            "renderer": "matplotlib",
            "figsize": [self.config.width / 100, self.config.height / 100],
            "dpi": self.config.dpi,
            "title": self.config.title or "Error Bar Plot",
            "xlabel": self.config.xlabel or "Score",
            "ylabel": self.config.ylabel or "",
            "orientation": "horizontal",
            "bars": bars,
            "capsize": 5,
            "grid": self.config.show_grid,
            "font": {"family": self.config.font_family, "size": self.config.font_size},
        }


# ========================================================================= #
#  7.  PlotExporter                                                         #
# ========================================================================= #


class PlotExporter:
    """Export plot specification dicts to HTML, JSON or plain dict."""

    _PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

    # ---- 7a. to_html -------------------------------------------------------
    @classmethod
    def to_html(
        cls,
        plot_dict: PlotSpec,
        path: Union[str, Path],
        *,
        title: str = "Plot",
        include_plotlyjs: bool = True,
        auto_open: bool = False,
    ) -> str:
        """Write a standalone HTML file that renders the plot via Plotly.js.

        Returns the absolute path written.
        """
        data_json = json.dumps(plot_dict.get("data", []), default=_json_default)
        layout_json = json.dumps(plot_dict.get("layout", {}), default=_json_default)
        frames_json = json.dumps(plot_dict.get("frames", []), default=_json_default)

        plotly_src = (
            f'<script src="{cls._PLOTLY_CDN}"></script>'
            if include_plotlyjs
            else ""
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape_html(title)}</title>
  {plotly_src}
  <style>
    body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
    #plot {{ width: 100%; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    var data = {data_json};
    var layout = {layout_json};
    var frames = {frames_json};
    var config = {{responsive: true, displayModeBar: true}};
    if (frames && frames.length > 0) {{
      Plotly.newPlot('plot', data, layout, config).then(function() {{
        Plotly.addFrames('plot', frames);
      }});
    }} else {{
      Plotly.newPlot('plot', data, layout, config);
    }}
  </script>
</body>
</html>"""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out.resolve())

    # ---- 7b. to_json -------------------------------------------------------
    @classmethod
    def to_json(
        cls,
        plot_dict: PlotSpec,
        path: Union[str, Path],
        *,
        indent: int = 2,
    ) -> str:
        """Save plot spec as a JSON file.  Returns path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(plot_dict, indent=indent, default=_json_default),
            encoding="utf-8",
        )
        return str(out.resolve())

    # ---- 7c. to_dict -------------------------------------------------------
    @staticmethod
    def to_dict(plot_dict: PlotSpec) -> Dict[str, Any]:
        """Return a JSON-serialisable copy of the plot spec."""
        return json.loads(json.dumps(plot_dict, default=_json_default))

    # ---- dashboard export --------------------------------------------------
    @classmethod
    def dashboard_to_html(
        cls,
        dashboard: Dict[str, Any],
        path: Union[str, Path],
        *,
        title: Optional[str] = None,
    ) -> str:
        """Export a multi-plot dashboard to a single HTML page.

        Each sub-plot gets its own ``<div>`` laid out in a CSS grid.
        """
        dash_title = title or dashboard.get("dashboard_title", "Dashboard")
        grid = dashboard.get("grid", {"rows": 1, "cols": 1})
        cols = grid.get("cols", 2)
        plots = dashboard.get("plots", [])

        plot_divs: List[str] = []
        plot_scripts: List[str] = []

        for idx, p in enumerate(plots):
            div_id = f"plot_{idx}"
            data_j = json.dumps(p.get("data", []), default=_json_default)
            layout_j = json.dumps(p.get("layout", {}), default=_json_default)
            frames_j = json.dumps(p.get("frames", []), default=_json_default)

            plot_divs.append(f'<div id="{div_id}" class="plot-cell"></div>')
            plot_scripts.append(f"""
(function() {{
  var d = {data_j};
  var l = {layout_j};
  var f = {frames_j};
  l.autosize = true;
  var cfg = {{responsive: true}};
  if (f && f.length > 0) {{
    Plotly.newPlot('{div_id}', d, l, cfg).then(function() {{
      Plotly.addFrames('{div_id}', f);
    }});
  }} else {{
    Plotly.newPlot('{div_id}', d, l, cfg);
  }}
}})();""")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape_html(dash_title)}</title>
  <script src="{cls._PLOTLY_CDN}"></script>
  <style>
    body {{ margin: 0; padding: 16px; font-family: Arial, sans-serif;
           background: #fafafa; }}
    h1 {{ text-align: center; color: #333; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat({cols}, 1fr);
      gap: 12px;
    }}
    .plot-cell {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1);
      min-height: 380px;
    }}
  </style>
</head>
<body>
  <h1>{_escape_html(dash_title)}</h1>
  <div class="grid">
    {"".join(plot_divs)}
  </div>
  <script>
    {"".join(plot_scripts)}
  </script>
</body>
</html>"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out.resolve())


# ========================================================================= #
#  Private utilities                                                        #
# ========================================================================= #


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "__float__"):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ========================================================================= #
#  Convenience factory                                                      #
# ========================================================================= #


def create_plotter(
    style: str = "default",
    **config_kwargs: Any,
) -> InteractivePlotter:
    """Create an ``InteractivePlotter`` using a named style preset.

    Parameters
    ----------
    style : one of ``default``, ``academic``, ``presentation``, ``dark``,
        ``light``, ``colorblind``
    **config_kwargs : additional overrides passed to ``PlotConfig``.
    """
    presets = {
        "default": PlotConfig,
        "academic": PlotStyleManager.academic_style,
        "presentation": PlotStyleManager.presentation_style,
        "dark": PlotStyleManager.dark_mode,
        "light": PlotStyleManager.light_mode,
        "colorblind": PlotStyleManager.colorblind_friendly,
    }
    factory = presets.get(style, PlotConfig)
    if callable(factory) and factory is PlotConfig:
        cfg = PlotConfig(**config_kwargs)
    else:
        cfg = factory()
        for k, v in config_kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return InteractivePlotter(cfg)


def create_static_plotter(
    style: str = "default",
    **config_kwargs: Any,
) -> StaticPlotter:
    """Create a ``StaticPlotter`` with a named style preset."""
    presets = {
        "default": PlotConfig,
        "academic": PlotStyleManager.academic_style,
        "presentation": PlotStyleManager.presentation_style,
        "dark": PlotStyleManager.dark_mode,
        "light": PlotStyleManager.light_mode,
        "colorblind": PlotStyleManager.colorblind_friendly,
    }
    factory = presets.get(style, PlotConfig)
    if callable(factory) and factory is PlotConfig:
        cfg = PlotConfig(**config_kwargs)
    else:
        cfg = factory()
        for k, v in config_kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return StaticPlotter(cfg)


def create_dashboard_generator(
    style: str = "default",
    **config_kwargs: Any,
) -> DashboardGenerator:
    """Create a ``DashboardGenerator`` with a named style preset."""
    presets = {
        "default": PlotConfig,
        "academic": PlotStyleManager.academic_style,
        "presentation": PlotStyleManager.presentation_style,
        "dark": PlotStyleManager.dark_mode,
        "light": PlotStyleManager.light_mode,
        "colorblind": PlotStyleManager.colorblind_friendly,
    }
    factory = presets.get(style, PlotConfig)
    if callable(factory) and factory is PlotConfig:
        cfg = PlotConfig(**config_kwargs)
    else:
        cfg = factory()
        for k, v in config_kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return DashboardGenerator(cfg)


def create_animation_generator(
    style: str = "default",
    **config_kwargs: Any,
) -> AnimationGenerator:
    """Create an ``AnimationGenerator`` with a named style preset."""
    presets = {
        "default": PlotConfig,
        "academic": PlotStyleManager.academic_style,
        "presentation": PlotStyleManager.presentation_style,
        "dark": PlotStyleManager.dark_mode,
        "light": PlotStyleManager.light_mode,
        "colorblind": PlotStyleManager.colorblind_friendly,
    }
    factory = presets.get(style, PlotConfig)
    if callable(factory) and factory is PlotConfig:
        cfg = PlotConfig(**config_kwargs)
    else:
        cfg = factory()
        for k, v in config_kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return AnimationGenerator(cfg)


# ========================================================================= #
#  Self-test / quick smoke test                                             #
# ========================================================================= #


# ========================================================================= #
#  Extended interactive visualizations                                      #
# ========================================================================= #


class FullDashboardGenerator:
    """Generate a complete HTML dashboard with multiple tabs and views."""

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()

    def generate_tabbed_dashboard(
        self,
        tabs: Dict[str, str],
        title: str = "Diversity Decoding Arena Dashboard",
    ) -> str:
        """Generate an HTML page with tab navigation.

        Parameters
        ----------
        tabs : dict mapping tab name → HTML content for that tab
        title : page title
        """
        tab_names = list(tabs.keys())
        css = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 0; background: #f5f5f5; }
.header { background: #1a1a2e; color: white; padding: 16px 24px;
           font-size: 20px; font-weight: 600; }
.tab-bar { background: #16213e; display: flex; gap: 0; }
.tab-btn { padding: 12px 24px; color: #aaa; cursor: pointer; border: none;
           background: none; font-size: 14px; font-weight: 500;
           border-bottom: 3px solid transparent; transition: all 0.2s; }
.tab-btn:hover { color: #ddd; background: rgba(255,255,255,0.05); }
.tab-btn.active { color: #fff; border-bottom-color: #4fc3f7; }
.tab-panel { display: none; padding: 24px; }
.tab-panel.active { display: block; }
.card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 20px; margin-bottom: 20px; }
.card h3 { margin-top: 0; color: #333; }
"""
        html = [
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
            f'<title>{_escape_html(title)}</title>',
            f'<style>{css}</style></head><body>',
            f'<div class="header">{_escape_html(title)}</div>',
            '<div class="tab-bar">',
        ]
        for i, name in enumerate(tab_names):
            cls = "tab-btn active" if i == 0 else "tab-btn"
            html.append(
                f'<button class="{cls}" onclick="switchTab({i})">'
                f'{_escape_html(name)}</button>'
            )
        html.append('</div>')

        for i, name in enumerate(tab_names):
            cls = "tab-panel active" if i == 0 else "tab-panel"
            html.append(f'<div class="{cls}" id="panel-{i}">')
            html.append(f'<div class="card">{tabs[name]}</div>')
            html.append('</div>')

        html.append(f'''
<script>
function switchTab(idx) {{
  var btns = document.querySelectorAll('.tab-btn');
  var panels = document.querySelectorAll('.tab-panel');
  for (var i = 0; i < btns.length; i++) {{
    btns[i].className = 'tab-btn' + (i === idx ? ' active' : '');
    panels[i].className = 'tab-panel' + (i === idx ? ' active' : '');
  }}
}}
</script></body></html>''')
        return "\n".join(html)


class InteractiveMetricExplorer:
    """Interactive metric explorer with dropdown selectors."""

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()

    def generate_explorer(
        self,
        metrics_data: Dict[str, Dict[str, List[float]]],
        title: str = "Interactive Metric Explorer",
    ) -> str:
        """Generate HTML metric explorer with dropdown-driven scatter plot.

        Parameters
        ----------
        metrics_data : dict mapping algorithm_name → {metric_name: [values]}
        """
        algo_names = sorted(metrics_data.keys())
        if not algo_names:
            return "<html><body><p>No data</p></body></html>"

        first_algo = algo_names[0]
        metric_names = sorted(metrics_data[first_algo].keys())

        data_json = json.dumps(
            {a: {m: list(map(float, v)) for m, v in metrics_data[a].items()}
             for a in algo_names},
            default=_json_default,
        )
        metrics_json = json.dumps(metric_names)
        algos_json = json.dumps(algo_names)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors_json = json.dumps(colors)

        html = f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{_escape_html(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
.controls {{ margin-bottom: 15px; display: flex; gap: 15px; align-items: center; }}
select {{ padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }}
label {{ font-weight: 600; color: #333; }}
canvas {{ background: white; border: 1px solid #ddd; border-radius: 4px; }}
.legend {{ margin-top: 10px; display: flex; gap: 15px; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 13px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
h2 {{ color: #1a1a2e; }}
</style></head><body>
<h2>{_escape_html(title)}</h2>
<div class="controls">
  <label>X Axis:</label>
  <select id="xMetric" onchange="draw()"></select>
  <label>Y Axis:</label>
  <select id="yMetric" onchange="draw()"></select>
</div>
<canvas id="plot" width="{self.config.width}" height="{self.config.height}"></canvas>
<div class="legend" id="legend"></div>
<script>
var data = {data_json};
var metrics = {metrics_json};
var algos = {algos_json};
var colors = {colors_json};

var xSel = document.getElementById("xMetric");
var ySel = document.getElementById("yMetric");
metrics.forEach(function(m, i) {{
  xSel.add(new Option(m, m, i===0, i===0));
  ySel.add(new Option(m, m, i===Math.min(1,metrics.length-1), i===Math.min(1,metrics.length-1)));
}});

var legendDiv = document.getElementById("legend");
algos.forEach(function(a, i) {{
  var d = document.createElement("div"); d.className = "legend-item";
  var dot = document.createElement("div"); dot.className = "legend-dot";
  dot.style.background = colors[i % colors.length];
  d.appendChild(dot); d.appendChild(document.createTextNode(a));
  legendDiv.appendChild(d);
}});

function draw() {{
  var xm = xSel.value, ym = ySel.value;
  var canvas = document.getElementById("plot");
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;
  var ml=70, mr=30, mt=30, mb=50;
  var pw=W-ml-mr, ph=H-mt-mb;
  ctx.clearRect(0, 0, W, H);

  // find range
  var xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  algos.forEach(function(a){{
    var xv=data[a][xm]||[], yv=data[a][ym]||[];
    for(var i=0;i<Math.min(xv.length,yv.length);i++){{
      if(xv[i]<xmin)xmin=xv[i]; if(xv[i]>xmax)xmax=xv[i];
      if(yv[i]<ymin)ymin=yv[i]; if(yv[i]>ymax)ymax=yv[i];
    }}
  }});
  if(xmin===xmax){{ xmin-=0.5; xmax+=0.5; }}
  if(ymin===ymax){{ ymin-=0.5; ymax+=0.5; }}

  // grid
  ctx.strokeStyle="#eee"; ctx.lineWidth=0.5;
  for(var i=0;i<=4;i++){{
    var gy=mt+ph*i/4; ctx.beginPath(); ctx.moveTo(ml,gy); ctx.lineTo(ml+pw,gy); ctx.stroke();
    var gx=ml+pw*i/4; ctx.beginPath(); ctx.moveTo(gx,mt); ctx.lineTo(gx,mt+ph); ctx.stroke();
  }}
  ctx.strokeStyle="#ccc"; ctx.lineWidth=1;
  ctx.strokeRect(ml,mt,pw,ph);

  // points
  algos.forEach(function(a,ai){{
    var xv=data[a][xm]||[], yv=data[a][ym]||[];
    ctx.fillStyle=colors[ai%colors.length];
    for(var i=0;i<Math.min(xv.length,yv.length);i++){{
      var sx=ml+(xv[i]-xmin)/(xmax-xmin)*pw;
      var sy=mt+ph-(yv[i]-ymin)/(ymax-ymin)*ph;
      ctx.beginPath(); ctx.arc(sx,sy,4,0,Math.PI*2); ctx.fill();
    }}
  }});

  // labels
  ctx.fillStyle="#333"; ctx.font="13px sans-serif"; ctx.textAlign="center";
  ctx.fillText(xm, ml+pw/2, H-10);
  ctx.save(); ctx.translate(15, mt+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(ym, 0, 0); ctx.restore();

  // ticks
  ctx.font="11px sans-serif"; ctx.fillStyle="#666"; ctx.textAlign="center";
  for(var i=0;i<=4;i++){{
    var v=xmin+(xmax-xmin)*i/4;
    ctx.fillText(v.toFixed(2), ml+pw*i/4, mt+ph+18);
  }}
  ctx.textAlign="right";
  for(var i=0;i<=4;i++){{
    var v=ymin+(ymax-ymin)*(4-i)/4;
    ctx.fillText(v.toFixed(2), ml-8, mt+ph*i/4+4);
  }}
}}
draw();
</script></body></html>'''
        return html


class AlgorithmComparisonView:
    """Side-by-side algorithm comparison view."""

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()

    def generate_comparison(
        self,
        algorithm_results: Dict[str, Dict[str, float]],
        title: str = "Algorithm Comparison",
    ) -> str:
        """Generate side-by-side comparison HTML table with bar indicators.

        Parameters
        ----------
        algorithm_results : dict mapping algorithm_name → {metric: value}
        """
        algo_names = sorted(algorithm_results.keys())
        if not algo_names:
            return "<html><body>No algorithms</body></html>"

        all_metrics = set()
        for v in algorithm_results.values():
            all_metrics.update(v.keys())
        metric_names = sorted(all_metrics)

        # compute min/max for normalization
        metric_range: Dict[str, Tuple[float, float]] = {}
        for m in metric_names:
            vals = [algorithm_results[a].get(m, 0.0) for a in algo_names]
            metric_range[m] = (min(vals), max(vals))

        css = """
body { font-family: sans-serif; margin: 20px; background: #fafafa; }
table { border-collapse: collapse; width: 100%; background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }
th { background: #1a1a2e; color: white; padding: 12px 16px; text-align: left;
     font-size: 13px; font-weight: 600; }
td { padding: 10px 16px; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover { background: #f8f9ff; }
.bar-bg { background: #e9ecef; border-radius: 3px; height: 8px; width: 100px;
           display: inline-block; vertical-align: middle; margin-left: 8px; }
.bar-fg { background: #4fc3f7; border-radius: 3px; height: 8px; display: block; }
.val { font-weight: 600; }
.best { color: #2e7d32; font-weight: 700; }
h2 { color: #1a1a2e; }
"""
        html = [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            f'<title>{_escape_html(title)}</title>',
            f'<style>{css}</style></head><body>',
            f'<h2>{_escape_html(title)}</h2>',
            '<table><thead><tr><th>Metric</th>',
        ]
        for a in algo_names:
            html.append(f'<th>{_escape_html(a)}</th>')
        html.append('</tr></thead><tbody>')

        for m in metric_names:
            html.append(f'<tr><td><strong>{_escape_html(m)}</strong></td>')
            vals = {a: algorithm_results[a].get(m, 0.0) for a in algo_names}
            best_val = max(vals.values())
            lo, hi = metric_range[m]
            rng = hi - lo if hi != lo else 1.0
            for a in algo_names:
                v = vals[a]
                pct = (v - lo) / rng * 100
                cls = "val best" if v == best_val and len(algo_names) > 1 else "val"
                html.append(
                    f'<td><span class="{cls}">{v:.4f}</span>'
                    f'<span class="bar-bg"><span class="bar-fg" '
                    f'style="width:{pct:.0f}%"></span></span></td>'
                )
            html.append('</tr>')

        html.append('</tbody></table></body></html>')
        return "\n".join(html)


class StandaloneHTMLExporter:
    """Export any visualization content to a standalone HTML file."""

    def __init__(self) -> None:
        pass

    def export_to_html(
        self,
        content: str,
        title: str = "Visualization Export",
        output_path: Optional[str] = None,
        include_print_styles: bool = True,
    ) -> str:
        """Wrap content in a complete standalone HTML document.

        Parameters
        ----------
        content : HTML/SVG content to wrap
        title : page title
        output_path : if provided, write to this file path
        include_print_styles : add print-friendly CSS

        Returns
        -------
        Complete HTML string.
        """
        print_css = """
@media print {
  body { background: white !important; }
  .no-print { display: none !important; }
  .card { box-shadow: none !important; border: 1px solid #ddd; }
  svg { max-width: 100%; height: auto; }
}
""" if include_print_styles else ""

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape_html(title)}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0; padding: 20px; background: #f5f5f5; color: #333;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
.export-header {{
  background: #1a1a2e; color: white; padding: 16px 24px;
  border-radius: 8px 8px 0 0; margin-bottom: 0;
}}
.export-body {{
  background: white; padding: 24px; border-radius: 0 0 8px 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
.toolbar {{ padding: 10px 0; text-align: right; }}
.toolbar button {{
  padding: 8px 16px; border: 1px solid #ccc; border-radius: 4px;
  background: white; cursor: pointer; font-size: 13px; margin-left: 8px;
}}
.toolbar button:hover {{ background: #f0f0f0; }}
svg {{ max-width: 100%; height: auto; }}
{print_css}
</style>
</head>
<body>
<div class="container">
  <div class="export-header">
    <h2 style="margin:0;">{_escape_html(title)}</h2>
    <p style="margin:4px 0 0; opacity:0.7; font-size:13px;">
      Generated by Diversity Decoding Arena
    </p>
  </div>
  <div class="toolbar no-print">
    <button onclick="window.print()">🖨️ Print</button>
  </div>
  <div class="export-body">
    {content}
  </div>
</div>
</body>
</html>'''
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
        return html


def _smoke_test() -> None:  # pragma: no cover
    """Run a quick self-test to verify all plot generators produce valid dicts."""
    rng = np.random.RandomState(42)

    # InteractivePlotter
    ip = create_plotter("academic", title="Test")
    pts = rng.rand(30, 2)
    frontier = [0, 5, 12, 20]
    p1 = ip.pareto_frontier_plot(pts, frontier, [f"p{i}" for i in range(30)])
    assert "data" in p1 and "layout" in p1

    corr = np.corrcoef(rng.rand(5, 50))
    p2 = ip.metric_heatmap(corr, [f"m{i}" for i in range(5)])
    assert "data" in p2

    scores = {"Algo1": rng.rand(5).tolist(), "Algo2": rng.rand(5).tolist()}
    p3 = ip.algorithm_radar_chart(scores, [f"m{i}" for i in range(5)])
    assert "data" in p3

    p4 = ip.diversity_quality_scatter([0.1, 0.5, 0.9], [0.9, 0.4, 0.7], ["A", "B", "C"])
    assert "data" in p4

    p5 = ip.hyperparameter_surface(
        {"lr": [0.001, 0.01, 0.1], "wd": [0.0, 0.01, 0.1]},
        rng.rand(3, 3),
    )
    assert "data" in p5

    p6 = ip.convergence_plot(list(range(10)), {"loss": rng.rand(10).tolist()})
    assert "data" in p6

    algo_res = {"A": rng.rand(50).tolist(), "B": rng.rand(50).tolist()}
    p7 = ip.box_plot_comparison(algo_res)
    p8 = ip.violin_plot_comparison(algo_res)
    assert "data" in p7 and "data" in p8

    data = {"d1": rng.rand(20).tolist(), "d2": rng.rand(20).tolist(), "d3": rng.rand(20).tolist()}
    p9 = ip.parallel_coordinates(data, ["d1", "d2", "d3"], ["A"] * 10 + ["B"] * 10)
    assert "data" in p9

    hier = {"ids": ["root", "a", "b"], "labels": ["Root", "A", "B"],
            "parents": ["", "root", "root"], "values": [10, 6, 4]}
    p10 = ip.sunburst_chart(hier)
    assert "data" in p10

    # StaticPlotter
    sp = create_static_plotter("academic")
    cd = sp.cd_diagram({"A": 1.5, "B": 2.3, "C": 3.1}, ["A", "B", "C"], 1.0)
    assert cd["type"] == "cd_diagram"

    pf = sp.pareto_frontier_2d(pts)
    assert pf["type"] == "pareto_2d"

    hm = sp.metric_correlation_heatmap(corr, [f"m{i}" for i in range(5)])
    assert hm["type"] == "heatmap"

    dc = sp.distribution_comparison(algo_res)
    assert dc["type"] == "distribution_comparison"

    eb = sp.error_bar_plot([0.5, 0.7], [0.1, 0.05], ["X", "Y"])
    assert eb["type"] == "error_bar"

    # DashboardGenerator
    dg = create_dashboard_generator()
    exp = {
        "algorithms": {"A": {"m0": rng.rand(20).tolist(), "m1": rng.rand(20).tolist()},
                       "B": {"m0": rng.rand(20).tolist(), "m1": rng.rand(20).tolist()}},
        "metric_names": ["m0", "m1"],
        "correlation_matrix": np.corrcoef(rng.rand(2, 20)).tolist(),
    }
    d1 = dg.generate_overview_dashboard(exp)
    assert "plots" in d1

    d2 = dg.generate_algorithm_dashboard("A", {
        "metrics": {"m0": rng.rand(20).tolist()},
        "convergence": {"m0": rng.rand(10).tolist()},
        "iterations": list(range(10)),
    })
    assert "plots" in d2

    d3 = dg.generate_metric_dashboard("m0", {
        "algorithms": {"A": rng.rand(20).tolist(), "B": rng.rand(20).tolist()},
    })
    assert "plots" in d3

    # AnimationGenerator
    ag = create_animation_generator()
    beam_states = []
    for s in range(4):
        nodes = [{"id": f"n{s}_{j}", "x": j, "y": s, "label": f"t{j}", "score": rng.rand()} for j in range(s + 1)]
        edges = [{"source": f"n{max(s-1,0)}_{0}", "target": f"n{s}_{j}"} for j in range(s + 1)] if s > 0 else []
        beam_states.append({"nodes": nodes, "edges": edges})
    a1 = ag.beam_search_animation(beam_states)
    assert "frames" in a1

    particle_states = []
    for g in range(5):
        pos = rng.rand(15, 2) * 10
        fit = rng.rand(15)
        particle_states.append({"positions": pos, "fitness": fit, "best": pos[fit.argmax()].tolist()})
    a2 = ag.particle_evolution_animation(particle_states)
    assert "frames" in a2

    archive_states = []
    for snap in range(6):
        grid = np.full((5, 5), np.nan)
        n_fill = min(snap * 4 + 1, 25)
        idxs = rng.choice(25, size=n_fill, replace=False)
        for idx in idxs:
            grid[idx // 5, idx % 5] = rng.rand()
        archive_states.append({"grid": grid, "x_bins": list(range(5)), "y_bins": list(range(5))})
    a3 = ag.archive_filling_animation(archive_states)
    assert "frames" in a3

    mh = {"loss": [rng.rand(i + 1).tolist() for i in range(8)],
           "acc": [rng.rand(i + 1).tolist() for i in range(8)]}
    a4 = ag.metric_evolution_animation(mh)
    assert "frames" in a4

    # PlotExporter (dict round-trip)
    exported = PlotExporter.to_dict(p1)
    assert isinstance(exported, dict)

    # PlotStyleManager
    c8 = PlotStyleManager.get_colors(8, "colorblind")
    assert len(c8) == 8
    c20 = PlotStyleManager.get_colors(20)
    assert len(c20) == 20

    # helpers
    cs = create_colorscale(5)
    assert len(cs) == 5

    ann = create_annotation("hello", 1, 2)
    assert ann["text"] == "hello"

    sh = create_shape("rect", {"x0": 0, "y0": 0, "x1": 1, "y1": 1})
    assert sh["type"] == "rect"

    fl = format_axis_labels([0.1234, 5.678], ".3f")
    assert fl == ["0.123", "5.678"]

    print("All smoke tests passed ✓")


if __name__ == "__main__":
    _smoke_test()
