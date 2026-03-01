"""
Pareto frontier visualization for the Diversity Decoding Arena.

Generates SVG/HTML visualizations of Pareto frontiers, hypervolume convergence,
radar charts, parallel coordinates, and multi-panel comparison dashboards
without requiring matplotlib. Uses numpy for numerical operations and produces
standalone SVG strings wrapped in HTML.
"""

from __future__ import annotations

import json
import math
import html as html_mod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Default colour palette – 20 visually distinct colours
# ---------------------------------------------------------------------------

DEFAULT_COLOR_PALETTE: Dict[str, str] = {
    "algorithm_0": "#1f77b4",
    "algorithm_1": "#ff7f0e",
    "algorithm_2": "#2ca02c",
    "algorithm_3": "#d62728",
    "algorithm_4": "#9467bd",
    "algorithm_5": "#8c564b",
    "algorithm_6": "#e377c2",
    "algorithm_7": "#7f7f7f",
    "algorithm_8": "#bcbd22",
    "algorithm_9": "#17becf",
    "algorithm_10": "#aec7e8",
    "algorithm_11": "#ffbb78",
    "algorithm_12": "#98df8a",
    "algorithm_13": "#ff9896",
    "algorithm_14": "#c5b0d5",
    "algorithm_15": "#c49c94",
    "algorithm_16": "#f7b6d2",
    "algorithm_17": "#c7c7c7",
    "algorithm_18": "#dbdb8d",
    "algorithm_19": "#9edae5",
}

# Ordered list for deterministic assignment
_PALETTE_ORDER: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

DEFAULT_MARKER_STYLES: Dict[str, str] = {
    "circle": "circle",
    "square": "square",
    "diamond": "diamond",
    "triangle_up": "triangle_up",
    "triangle_down": "triangle_down",
    "star": "star",
    "cross": "cross",
    "plus": "plus",
    "hexagon": "hexagon",
    "pentagon": "pentagon",
}

_MARKER_ORDER: List[str] = [
    "circle", "square", "diamond", "triangle_up", "triangle_down",
    "star", "cross", "plus", "hexagon", "pentagon",
]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParetoPlotConfig:
    """Full configuration for Pareto frontier plots."""

    figsize: Tuple[int, int] = (800, 600)
    dpi: int = 96
    color_palette: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLOR_PALETTE))
    marker_styles: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MARKER_STYLES))
    font_size: int = 14
    title: str = "Pareto Frontier"
    xlabel: str = "Objective 1"
    ylabel: str = "Objective 2"
    show_dominated: bool = True
    show_frontier_line: bool = True
    annotate_points: bool = False
    alpha: float = 0.85
    grid: bool = True
    legend_loc: str = "upper right"
    save_path: Optional[str] = None
    format: str = "svg"  # png | svg | pdf
    show_confidence_band: bool = False
    frontier_line_style: str = "solid"  # solid | dashed | dotted
    dominated_alpha: float = 0.30
    highlight_algorithms: List[str] = field(default_factory=list)

    # Margins inside the SVG (pixels)
    margin_left: int = 80
    margin_right: int = 40
    margin_top: int = 50
    margin_bottom: int = 60

    @property
    def plot_width(self) -> int:
        return self.figsize[0] - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> int:
        return self.figsize[1] - self.margin_top - self.margin_bottom


@dataclass
class FrontierAnimation:
    """Container for an animated Pareto frontier sequence."""

    frames: List[Dict[str, Any]] = field(default_factory=list)
    fps: int = 2
    output_path: Optional[str] = None
    title: str = "Pareto Frontier Evolution"


# ---------------------------------------------------------------------------
# Helper: lightweight SVG builder
# ---------------------------------------------------------------------------


class _SVGBuilder:
    """Accumulates SVG elements and renders a complete document."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._defs: List[str] = []
        self._elements: List[str] = []
        self._styles: List[str] = []

    # -- primitives ---------------------------------------------------------

    def add_def(self, d: str) -> None:
        self._defs.append(d)

    def add_style(self, css: str) -> None:
        self._styles.append(css)

    def add(self, element: str) -> None:
        self._elements.append(element)

    # -- render -------------------------------------------------------------

    def render(self) -> str:
        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'style="background:#ffffff;">',
        ]
        if self._styles:
            parts.append("<style>")
            parts.extend(self._styles)
            parts.append("</style>")
        if self._defs:
            parts.append("<defs>")
            parts.extend(self._defs)
            parts.append("</defs>")
        parts.extend(self._elements)
        parts.append("</svg>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# ParetoPlotter
# ---------------------------------------------------------------------------


class ParetoPlotter:
    """Generates Pareto frontier visualizations as SVG / HTML strings.

    All *plot_** methods return a ``dict`` with at least:
      - ``"svg"``: SVG string
      - ``"html"``: standalone HTML wrapping the SVG
      - ``"data"``: raw numerical data used in the plot
    """

    def __init__(self, config: Optional[ParetoPlotConfig] = None) -> None:
        self.config = config or ParetoPlotConfig()
        self._algo_color_cache: Dict[str, str] = {}
        self._algo_marker_cache: Dict[str, str] = {}
        self._algo_counter: int = 0

    # ======================================================================
    # Public plotting API
    # ======================================================================

    def plot_2d_frontier(
        self,
        points: np.ndarray,
        frontier_indices: Sequence[int],
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot a 2-D Pareto frontier.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            All evaluated points (objective1, objective2).
        frontier_indices : sequence of int
            Indices into *points* that lie on the Pareto frontier.
        title : str, optional
            Override the config title.

        Returns
        -------
        dict with keys ``svg``, ``html``, ``data``.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("points must have shape (N, >=2)")
        frontier_indices = list(frontier_indices)
        title = title or self.config.title

        cfg = self.config
        w, h = cfg.figsize
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
            f'.annotation {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 3}px; }}'
        )

        x_vals = points[:, 0]
        y_vals = points[:, 1]
        x_norm, x_min, x_max = self._normalize_axis(x_vals)
        y_norm, y_min, y_max = self._normalize_axis(y_vals)

        plot_bounds = (cfg.margin_left, cfg.margin_top, cfg.plot_width, cfg.plot_height)

        if cfg.grid:
            self._add_grid_lines(builder, plot_bounds, n_lines=6)

        self._add_axes(builder, plot_bounds)
        self._add_axis_labels(builder, cfg.xlabel, cfg.ylabel, plot_bounds)
        builder.add(self._svg_text(w / 2, cfg.margin_top - 18, title, cfg.font_size + 2, "middle", cls="title"))

        # Tick labels
        self._add_tick_labels(builder, plot_bounds, x_min, x_max, y_min, y_max, n_ticks=6)

        frontier_set = set(frontier_indices)
        dominated_elements: List[str] = []
        frontier_elements: List[str] = []

        for i in range(len(points)):
            cx = cfg.margin_left + x_norm[i] * cfg.plot_width
            cy = cfg.margin_top + (1.0 - y_norm[i]) * cfg.plot_height
            if i in frontier_set:
                color = _PALETTE_ORDER[0]
                opacity = cfg.alpha
                r = 6
                frontier_elements.append(self._svg_circle(cx, cy, r, color, opacity))
                if cfg.annotate_points:
                    label = self._format_label(points[i, 0], 2) + ", " + self._format_label(points[i, 1], 2)
                    frontier_elements.append(
                        self._svg_text(cx + 8, cy - 6, label, cfg.font_size - 3, "start", cls="annotation")
                    )
            else:
                if cfg.show_dominated:
                    color = "#999999"
                    opacity = cfg.dominated_alpha
                    r = 4
                    dominated_elements.append(self._svg_circle(cx, cy, r, color, opacity))

        for el in dominated_elements:
            builder.add(el)
        for el in frontier_elements:
            builder.add(el)

        # Frontier line
        if cfg.show_frontier_line and len(frontier_indices) >= 2:
            f_pts = points[frontier_indices]
            order = np.argsort(f_pts[:, 0])
            f_sorted = f_pts[order]
            line_points: List[Tuple[float, float]] = []
            for pt in f_sorted:
                nx = (pt[0] - x_min) / max(x_max - x_min, 1e-12)
                ny = (pt[1] - y_min) / max(y_max - y_min, 1e-12)
                cx = cfg.margin_left + nx * cfg.plot_width
                cy = cfg.margin_top + (1.0 - ny) * cfg.plot_height
                line_points.append((cx, cy))
            dash = self._dash_for_style(cfg.frontier_line_style)
            builder.add(self._svg_polyline(line_points, _PALETTE_ORDER[0], 2.0, dash=dash))

        # Legend
        legend_items = [{"label": "Frontier", "color": _PALETTE_ORDER[0]}]
        if cfg.show_dominated:
            legend_items.append({"label": "Dominated", "color": "#999999"})
        self._add_legend(builder, legend_items, cfg.legend_loc)

        svg = builder.render()
        html_str = self._generate_html_plot({"svg": svg, "title": title})

        data = {
            "points": points.tolist(),
            "frontier_indices": frontier_indices,
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
        }

        result = {"svg": svg, "html": html_str, "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_2d_with_confidence(
        self,
        points: np.ndarray,
        frontier: np.ndarray,
        bootstrap_bands: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Plot 2-D frontier with bootstrap confidence bands.

        Parameters
        ----------
        points : ndarray (N, 2)
        frontier : ndarray (F, 2) – frontier points sorted by x.
        bootstrap_bands : dict with ``"lower"`` and ``"upper"`` arrays (F, 2).
        """
        points = np.asarray(points, dtype=float)
        frontier = np.asarray(frontier, dtype=float)
        lower = np.asarray(bootstrap_bands.get("lower", frontier), dtype=float)
        upper = np.asarray(bootstrap_bands.get("upper", frontier), dtype=float)

        cfg = self.config
        w, h = cfg.figsize
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
        )

        all_x = np.concatenate([points[:, 0], frontier[:, 0], lower[:, 0], upper[:, 0]])
        all_y = np.concatenate([points[:, 1], frontier[:, 1], lower[:, 1], upper[:, 1]])
        x_norm_all, x_min, x_max = self._normalize_axis(all_x)
        y_norm_all, y_min, y_max = self._normalize_axis(all_y)

        plot_bounds = (cfg.margin_left, cfg.margin_top, cfg.plot_width, cfg.plot_height)
        title = cfg.title + " (with confidence band)"

        if cfg.grid:
            self._add_grid_lines(builder, plot_bounds, n_lines=6)
        self._add_axes(builder, plot_bounds)
        self._add_axis_labels(builder, cfg.xlabel, cfg.ylabel, plot_bounds)
        self._add_tick_labels(builder, plot_bounds, x_min, x_max, y_min, y_max, n_ticks=6)
        builder.add(self._svg_text(w / 2, cfg.margin_top - 18, title, cfg.font_size + 2, "middle", cls="title"))

        def to_px(pt: np.ndarray) -> Tuple[float, float]:
            nx = (pt[0] - x_min) / max(x_max - x_min, 1e-12)
            ny = (pt[1] - y_min) / max(y_max - y_min, 1e-12)
            return (
                cfg.margin_left + nx * cfg.plot_width,
                cfg.margin_top + (1.0 - ny) * cfg.plot_height,
            )

        # confidence band polygon
        band_pts: List[Tuple[float, float]] = []
        for i in range(len(upper)):
            band_pts.append(to_px(upper[i]))
        for i in range(len(lower) - 1, -1, -1):
            band_pts.append(to_px(lower[i]))
        builder.add(self._svg_polygon(band_pts, _PALETTE_ORDER[0], "none", 0.18))

        # frontier line
        f_px = [to_px(frontier[i]) for i in range(len(frontier))]
        builder.add(self._svg_polyline(f_px, _PALETTE_ORDER[0], 2.5))

        # dominated points
        if cfg.show_dominated:
            for pt in points:
                cx, cy = to_px(pt)
                builder.add(self._svg_circle(cx, cy, 3.5, "#999999", cfg.dominated_alpha))

        # frontier points
        for pt in frontier:
            cx, cy = to_px(pt)
            builder.add(self._svg_circle(cx, cy, 5.5, _PALETTE_ORDER[0], cfg.alpha))

        legend_items = [
            {"label": "Frontier", "color": _PALETTE_ORDER[0]},
            {"label": "95% band", "color": _PALETTE_ORDER[0]},
        ]
        if cfg.show_dominated:
            legend_items.append({"label": "Dominated", "color": "#999999"})
        self._add_legend(builder, legend_items, cfg.legend_loc)

        svg = builder.render()
        data = {
            "points": points.tolist(),
            "frontier": frontier.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
        }
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": title}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_3d_frontier(
        self,
        points: np.ndarray,
        frontier_indices: Sequence[int],
    ) -> Dict[str, Any]:
        """Project a 3-D Pareto frontier into three 2-D scatter panels.

        Because we generate SVG (inherently 2-D) we produce a 1×3 panel
        showing pairwise projections: (obj0 vs obj1), (obj0 vs obj2),
        (obj1 vs obj2).

        Parameters
        ----------
        points : ndarray (N, 3)
        frontier_indices : sequence of int
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape (N, >=3)")
        frontier_indices = list(frontier_indices)
        frontier_set = set(frontier_indices)

        cfg = self.config
        panel_w = cfg.figsize[0]
        panel_h = cfg.figsize[1]
        total_w = panel_w * 3 + 20
        total_h = panel_h + 30
        builder = _SVGBuilder(total_w, total_h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 1}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 1}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 3}px; }}'
        )

        pairs = [(0, 1), (0, 2), (1, 2)]
        labels_map = {0: "Obj 1", 1: "Obj 2", 2: "Obj 3"}

        for panel_idx, (xi, yi) in enumerate(pairs):
            ox = panel_idx * (panel_w + 10)
            oy = 0
            x_vals = points[:, xi]
            y_vals = points[:, yi]
            x_norm, x_min, x_max = self._normalize_axis(x_vals)
            y_norm, y_min, y_max = self._normalize_axis(y_vals)

            pb = (ox + cfg.margin_left, oy + cfg.margin_top, cfg.plot_width, cfg.plot_height)

            if cfg.grid:
                self._add_grid_lines(builder, pb, n_lines=5)
            self._add_axes(builder, pb)
            self._add_axis_labels(builder, labels_map[xi], labels_map[yi], pb)
            self._add_tick_labels(builder, pb, x_min, x_max, y_min, y_max, n_ticks=5)
            panel_title = f"{labels_map[xi]} vs {labels_map[yi]}"
            builder.add(
                self._svg_text(
                    ox + panel_w / 2, oy + cfg.margin_top - 18, panel_title, cfg.font_size, "middle", cls="title"
                )
            )

            for i in range(len(points)):
                cx = pb[0] + x_norm[i] * pb[2]
                cy = pb[1] + (1.0 - y_norm[i]) * pb[3]
                if i in frontier_set:
                    builder.add(self._svg_circle(cx, cy, 5, _PALETTE_ORDER[0], cfg.alpha))
                elif cfg.show_dominated:
                    builder.add(self._svg_circle(cx, cy, 3, "#999999", cfg.dominated_alpha))

        svg = builder.render()
        data = {
            "points": points.tolist(),
            "frontier_indices": frontier_indices,
            "projections": [list(p) for p in pairs],
        }
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": "3-D Frontier Projections"}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_frontier_evolution(
        self,
        frontiers_over_time: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Create an animated HTML page showing the Pareto frontier evolving.

        Each entry in *frontiers_over_time* is an (F_t, 2) array of frontier
        points at time-step t.  We create an HTML page with inline JavaScript
        that steps through SVG frames.

        Returns dict with ``svg`` (first frame), ``html``, ``data``, ``animation``.
        """
        cfg = self.config
        w, h = cfg.figsize

        all_pts = np.concatenate(frontiers_over_time, axis=0)
        _, x_min, x_max = self._normalize_axis(all_pts[:, 0])
        _, y_min, y_max = self._normalize_axis(all_pts[:, 1])

        frames_svg: List[str] = []
        for t, frontier in enumerate(frontiers_over_time):
            frontier = np.asarray(frontier, dtype=float)
            builder = _SVGBuilder(w, h)
            builder.add_style(
                f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
                f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
                f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
            )
            plot_bounds = (cfg.margin_left, cfg.margin_top, cfg.plot_width, cfg.plot_height)
            if cfg.grid:
                self._add_grid_lines(builder, plot_bounds, n_lines=6)
            self._add_axes(builder, plot_bounds)
            self._add_axis_labels(builder, cfg.xlabel, cfg.ylabel, plot_bounds)
            self._add_tick_labels(builder, plot_bounds, x_min, x_max, y_min, y_max, n_ticks=6)
            frame_title = f"{cfg.title} — step {t + 1}/{len(frontiers_over_time)}"
            builder.add(self._svg_text(w / 2, cfg.margin_top - 18, frame_title, cfg.font_size + 2, "middle", cls="title"))

            order = np.argsort(frontier[:, 0])
            f_sorted = frontier[order]
            px_pts: List[Tuple[float, float]] = []
            for pt in f_sorted:
                nx = (pt[0] - x_min) / max(x_max - x_min, 1e-12)
                ny = (pt[1] - y_min) / max(y_max - y_min, 1e-12)
                cx = cfg.margin_left + nx * cfg.plot_width
                cy = cfg.margin_top + (1.0 - ny) * cfg.plot_height
                px_pts.append((cx, cy))
                builder.add(self._svg_circle(cx, cy, 5, _PALETTE_ORDER[t % len(_PALETTE_ORDER)], cfg.alpha))

            if len(px_pts) >= 2:
                builder.add(self._svg_polyline(px_pts, _PALETTE_ORDER[t % len(_PALETTE_ORDER)], 2.0))

            frames_svg.append(builder.render())

        anim_html = self._build_animation_html(frames_svg, fps=2, title=cfg.title)
        anim = FrontierAnimation(
            frames=[{"step": t, "n_points": len(f)} for t, f in enumerate(frontiers_over_time)],
            fps=2,
            title=cfg.title,
        )
        data = {
            "n_steps": len(frontiers_over_time),
            "frontiers": [f.tolist() for f in frontiers_over_time],
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
        }
        result = {
            "svg": frames_svg[0] if frames_svg else "",
            "html": anim_html,
            "data": data,
            "animation": anim,
        }
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_multiple_frontiers(
        self,
        frontiers_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Overlay multiple algorithm frontiers on a single plot.

        Parameters
        ----------
        frontiers_dict : {algo_name: ndarray (F_i, 2)}
        """
        cfg = self.config
        w, h = cfg.figsize
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
        )

        # global range
        all_pts = np.concatenate(list(frontiers_dict.values()), axis=0)
        _, x_min, x_max = self._normalize_axis(all_pts[:, 0])
        _, y_min, y_max = self._normalize_axis(all_pts[:, 1])

        plot_bounds = (cfg.margin_left, cfg.margin_top, cfg.plot_width, cfg.plot_height)
        title = cfg.title

        if cfg.grid:
            self._add_grid_lines(builder, plot_bounds, n_lines=6)
        self._add_axes(builder, plot_bounds)
        self._add_axis_labels(builder, cfg.xlabel, cfg.ylabel, plot_bounds)
        self._add_tick_labels(builder, plot_bounds, x_min, x_max, y_min, y_max, n_ticks=6)
        builder.add(self._svg_text(w / 2, cfg.margin_top - 18, title, cfg.font_size + 2, "middle", cls="title"))

        legend_items: List[Dict[str, str]] = []
        for algo_idx, (algo_name, frontier) in enumerate(frontiers_dict.items()):
            frontier = np.asarray(frontier, dtype=float)
            color = self._color_for_algorithm(algo_name)
            marker = self._marker_for_algorithm(algo_name)

            order = np.argsort(frontier[:, 0])
            f_sorted = frontier[order]

            px_pts: List[Tuple[float, float]] = []
            for pt in f_sorted:
                nx = (pt[0] - x_min) / max(x_max - x_min, 1e-12)
                ny = (pt[1] - y_min) / max(y_max - y_min, 1e-12)
                cx = cfg.margin_left + nx * cfg.plot_width
                cy = cfg.margin_top + (1.0 - ny) * cfg.plot_height
                px_pts.append((cx, cy))
                r = 6 if algo_name in cfg.highlight_algorithms else 5
                builder.add(self._render_marker(cx, cy, r, marker, color, cfg.alpha))

            if len(px_pts) >= 2 and cfg.show_frontier_line:
                dash = self._dash_for_style(cfg.frontier_line_style)
                builder.add(self._svg_polyline(px_pts, color, 2.0, dash=dash))

            legend_items.append({"label": algo_name, "color": color})

        self._add_legend(builder, legend_items, cfg.legend_loc)

        svg = builder.render()
        data = {
            "algorithms": {k: v.tolist() for k, v in frontiers_dict.items()},
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
        }
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": title}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_hypervolume_convergence(
        self,
        hv_values_over_time: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Line chart of hypervolume indicator over iterations.

        Parameters
        ----------
        hv_values_over_time : {algo_name: [hv_0, hv_1, …]}
        """
        cfg = self.config
        w, h = cfg.figsize
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
        )

        all_vals: List[float] = []
        max_len = 0
        for vals in hv_values_over_time.values():
            all_vals.extend(vals)
            max_len = max(max_len, len(vals))

        if not all_vals:
            all_vals = [0.0, 1.0]
        y_arr = np.array(all_vals, dtype=float)
        _, y_min, y_max = self._normalize_axis(y_arr)
        x_min_v, x_max_v = 0.0, float(max(max_len - 1, 1))

        plot_bounds = (cfg.margin_left, cfg.margin_top, cfg.plot_width, cfg.plot_height)
        title = "Hypervolume Convergence"

        if cfg.grid:
            self._add_grid_lines(builder, plot_bounds, n_lines=6)
        self._add_axes(builder, plot_bounds)
        self._add_axis_labels(builder, "Iteration", "Hypervolume", plot_bounds)
        self._add_tick_labels(builder, plot_bounds, x_min_v, x_max_v, y_min, y_max, n_ticks=6)
        builder.add(self._svg_text(w / 2, cfg.margin_top - 18, title, cfg.font_size + 2, "middle", cls="title"))

        legend_items: List[Dict[str, str]] = []
        for algo_name, vals in hv_values_over_time.items():
            color = self._color_for_algorithm(algo_name)
            px_pts: List[Tuple[float, float]] = []
            for i, v in enumerate(vals):
                nx = i / max(x_max_v, 1e-12)
                ny = (v - y_min) / max(y_max - y_min, 1e-12)
                cx = cfg.margin_left + nx * cfg.plot_width
                cy = cfg.margin_top + (1.0 - ny) * cfg.plot_height
                px_pts.append((cx, cy))
            if px_pts:
                builder.add(self._svg_polyline(px_pts, color, 2.0))
                # markers at every 10th point or all if short
                step = max(1, len(px_pts) // 10)
                for j in range(0, len(px_pts), step):
                    builder.add(self._svg_circle(px_pts[j][0], px_pts[j][1], 3, color, cfg.alpha))
                # last point
                builder.add(self._svg_circle(px_pts[-1][0], px_pts[-1][1], 4, color, 1.0))
            legend_items.append({"label": algo_name, "color": color})

        self._add_legend(builder, legend_items, cfg.legend_loc)

        svg = builder.render()
        data = {"algorithms": {k: list(v) for k, v in hv_values_over_time.items()}}
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": title}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_contribution_bar(
        self,
        contributions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Horizontal bar chart of per-algorithm hypervolume contribution.

        Parameters
        ----------
        contributions : {algo_name: contribution_value}
        """
        cfg = self.config
        w, h = cfg.figsize
        n_bars = len(contributions)
        if n_bars == 0:
            n_bars = 1

        # Adjust height dynamically
        bar_h = 28
        spacing = 10
        needed_h = cfg.margin_top + cfg.margin_bottom + n_bars * (bar_h + spacing)
        h = max(h, needed_h)
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
            f'.bar-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 1}px; }}'
        )

        title = "Hypervolume Contribution by Algorithm"
        builder.add(self._svg_text(w / 2, cfg.margin_top - 18, title, cfg.font_size + 2, "middle", cls="title"))

        max_val = max(contributions.values()) if contributions else 1.0
        if max_val <= 0:
            max_val = 1.0

        sorted_items = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

        for idx, (algo_name, val) in enumerate(sorted_items):
            color = self._color_for_algorithm(algo_name)
            bar_top = cfg.margin_top + idx * (bar_h + spacing)
            bar_width = (val / max_val) * cfg.plot_width
            bar_width = max(bar_width, 2)

            # algo label
            builder.add(
                self._svg_text(
                    cfg.margin_left - 5,
                    bar_top + bar_h / 2 + 4,
                    algo_name,
                    cfg.font_size - 1,
                    "end",
                    cls="bar-label",
                )
            )
            # bar
            builder.add(self._svg_rect(cfg.margin_left, bar_top, bar_width, bar_h, color, cfg.alpha))
            # value label
            builder.add(
                self._svg_text(
                    cfg.margin_left + bar_width + 6,
                    bar_top + bar_h / 2 + 4,
                    self._format_label(val, 4),
                    cfg.font_size - 2,
                    "start",
                    cls="tick-label",
                )
            )

        svg = builder.render()
        data = {"contributions": dict(sorted_items)}
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": title}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_parallel_coordinates(
        self,
        points: np.ndarray,
        objective_names: List[str],
    ) -> Dict[str, Any]:
        """Parallel coordinates plot.

        Parameters
        ----------
        points : ndarray (N, D)
        objective_names : list of D strings
        """
        points = np.asarray(points, dtype=float)
        n_points, n_dims = points.shape
        if len(objective_names) != n_dims:
            raise ValueError("objective_names length must equal points.shape[1]")

        cfg = self.config
        w, h = cfg.figsize
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
        )

        title = "Parallel Coordinates"
        builder.add(self._svg_text(w / 2, cfg.margin_top - 18, title, cfg.font_size + 2, "middle", cls="title"))

        plot_left = cfg.margin_left
        plot_right = w - cfg.margin_right
        plot_top = cfg.margin_top
        plot_bottom = h - cfg.margin_bottom

        axis_xs = np.linspace(plot_left, plot_right, n_dims).tolist()

        # normalise each dimension independently
        norm_cols: List[np.ndarray] = []
        col_ranges: List[Tuple[float, float]] = []
        for d in range(n_dims):
            col = points[:, d]
            n, cmin, cmax = self._normalize_axis(col)
            norm_cols.append(n)
            col_ranges.append((cmin, cmax))

        # draw axes
        for d in range(n_dims):
            ax = axis_xs[d]
            builder.add(self._svg_line(ax, plot_top, ax, plot_bottom, "#444444", 1.5))
            builder.add(
                self._svg_text(ax, plot_bottom + 18, objective_names[d], cfg.font_size - 1, "middle", cls="axis-label")
            )
            cmin, cmax = col_ranges[d]
            builder.add(self._svg_text(ax + 4, plot_top - 4, self._format_label(cmax, 2), cfg.font_size - 3, "start", cls="tick-label"))
            builder.add(self._svg_text(ax + 4, plot_bottom + 4, self._format_label(cmin, 2), cfg.font_size - 3, "start", cls="tick-label"))

        # draw polylines
        for i in range(n_points):
            color = _PALETTE_ORDER[i % len(_PALETTE_ORDER)]
            pts: List[Tuple[float, float]] = []
            for d in range(n_dims):
                ax = axis_xs[d]
                y = plot_bottom - norm_cols[d][i] * (plot_bottom - plot_top)
                pts.append((ax, y))
            builder.add(self._svg_polyline(pts, color, 1.5, opacity=0.55))

        svg = builder.render()
        data = {
            "points": points.tolist(),
            "objective_names": objective_names,
            "ranges": col_ranges,
        }
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": title}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_radar_chart(
        self,
        algorithm_scores: Dict[str, List[float]],
        metric_names: List[str],
    ) -> Dict[str, Any]:
        """Radar / spider chart comparing algorithms across metrics.

        Parameters
        ----------
        algorithm_scores : {algo_name: [score_per_metric]}
        metric_names : list of metric names (same length as score lists)
        """
        cfg = self.config
        w, h = cfg.figsize
        builder = _SVGBuilder(w, h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 1}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size + 2}px; font-weight: bold; }}'
        )

        title = "Algorithm Radar Chart"
        builder.add(self._svg_text(w / 2, 25, title, cfg.font_size + 2, "middle", cls="title"))

        n_metrics = len(metric_names)
        if n_metrics < 3:
            raise ValueError("Radar chart needs ≥3 metrics")

        cx_center = w / 2
        cy_center = h / 2 + 10
        radius = min(cfg.plot_width, cfg.plot_height) * 0.40

        angles = [2 * math.pi * i / n_metrics - math.pi / 2 for i in range(n_metrics)]

        # Normalise across all algorithms per metric
        all_scores = np.array(list(algorithm_scores.values()), dtype=float)  # (A, M)
        metric_max = np.max(all_scores, axis=0)
        metric_max = np.where(metric_max == 0, 1.0, metric_max)

        # concentric grid rings
        for ring in [0.25, 0.50, 0.75, 1.0]:
            ring_pts: List[Tuple[float, float]] = []
            for a in angles:
                rx = cx_center + radius * ring * math.cos(a)
                ry = cy_center + radius * ring * math.sin(a)
                ring_pts.append((rx, ry))
            ring_pts.append(ring_pts[0])
            builder.add(self._svg_polyline(ring_pts, "#cccccc", 0.8))

        # axis spokes and labels
        for i, a in enumerate(angles):
            ex = cx_center + radius * math.cos(a)
            ey = cy_center + radius * math.sin(a)
            builder.add(self._svg_line(cx_center, cy_center, ex, ey, "#cccccc", 0.8))
            lx = cx_center + (radius + 18) * math.cos(a)
            ly = cy_center + (radius + 18) * math.sin(a)
            anchor = "middle"
            if math.cos(a) > 0.3:
                anchor = "start"
            elif math.cos(a) < -0.3:
                anchor = "end"
            builder.add(self._svg_text(lx, ly + 4, metric_names[i], cfg.font_size - 1, anchor, cls="axis-label"))

        legend_items: List[Dict[str, str]] = []
        for algo_name, scores in algorithm_scores.items():
            color = self._color_for_algorithm(algo_name)
            norm_scores = [s / m for s, m in zip(scores, metric_max)]
            poly_pts: List[Tuple[float, float]] = []
            for i, a in enumerate(angles):
                r = radius * norm_scores[i]
                px = cx_center + r * math.cos(a)
                py = cy_center + r * math.sin(a)
                poly_pts.append((px, py))
            builder.add(self._svg_polygon(poly_pts, color, color, 0.20))
            poly_pts_closed = poly_pts + [poly_pts[0]]
            builder.add(self._svg_polyline(poly_pts_closed, color, 2.0))
            for px, py in poly_pts:
                builder.add(self._svg_circle(px, py, 3, color, 0.9))
            legend_items.append({"label": algo_name, "color": color})

        self._add_legend(builder, legend_items, "upper right")

        svg = builder.render()
        data = {
            "algorithms": {k: list(v) for k, v in algorithm_scores.items()},
            "metric_names": metric_names,
        }
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": title}), "data": data}
        self._maybe_save(result)
        return result

    # ------------------------------------------------------------------

    def plot_tradeoff_curves(
        self,
        results: Dict[str, np.ndarray],
        diversity_metrics: List[str],
        quality_metrics: List[str],
    ) -> Dict[str, Any]:
        """Grid of tradeoff scatter panels: each diversity metric vs each quality metric.

        Parameters
        ----------
        results : {algo_name: ndarray (N, D+Q)} columns are diversity_metrics then quality_metrics
        diversity_metrics, quality_metrics : names
        """
        cfg = self.config
        n_div = len(diversity_metrics)
        n_qual = len(quality_metrics)

        panel_w = 300
        panel_h = 250
        total_w = cfg.margin_left + n_qual * panel_w + cfg.margin_right
        total_h = cfg.margin_top + n_div * panel_h + cfg.margin_bottom + 30

        builder = _SVGBuilder(total_w, total_h)
        builder.add_style(
            f'.axis-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 2}px; }}'
            f'.title {{ font-family: Arial, sans-serif; font-size: {cfg.font_size}px; font-weight: bold; }}'
            f'.tick-label {{ font-family: Arial, sans-serif; font-size: {cfg.font_size - 4}px; }}'
        )

        builder.add(
            self._svg_text(
                total_w / 2, 25, "Diversity–Quality Tradeoff Grid", cfg.font_size + 2, "middle", cls="title"
            )
        )

        legend_items: List[Dict[str, str]] = []

        for row, div_name in enumerate(diversity_metrics):
            for col, qual_name in enumerate(quality_metrics):
                ox = cfg.margin_left + col * panel_w
                oy = cfg.margin_top + row * panel_h
                inner_margin = 40
                inner_w = panel_w - 2 * inner_margin
                inner_h = panel_h - 2 * inner_margin
                pb = (ox + inner_margin, oy + inner_margin, inner_w, inner_h)

                # Collect all values for this panel
                all_x_vals: List[float] = []
                all_y_vals: List[float] = []
                for algo_name, data_arr in results.items():
                    data_arr = np.asarray(data_arr, dtype=float)
                    all_x_vals.extend(data_arr[:, n_div + col].tolist())
                    all_y_vals.extend(data_arr[:, row].tolist())

                if not all_x_vals:
                    continue
                _, x_min, x_max = self._normalize_axis(np.array(all_x_vals))
                _, y_min, y_max = self._normalize_axis(np.array(all_y_vals))

                self._add_grid_lines(builder, pb, n_lines=4)
                self._add_axes(builder, pb)

                # Labels
                builder.add(
                    self._svg_text(
                        ox + panel_w / 2, oy + panel_h - 5, qual_name, cfg.font_size - 2, "middle", cls="axis-label"
                    )
                )
                if col == 0:
                    builder.add(
                        self._svg_text(
                            ox + 10, oy + panel_h / 2, div_name, cfg.font_size - 2, "middle", cls="axis-label",
                            transform=f"rotate(-90, {ox + 10}, {oy + panel_h / 2})"
                        )
                    )

                for algo_name, data_arr in results.items():
                    data_arr = np.asarray(data_arr, dtype=float)
                    color = self._color_for_algorithm(algo_name)
                    x_col = data_arr[:, n_div + col]
                    y_col = data_arr[:, row]
                    for k in range(len(data_arr)):
                        nx = (x_col[k] - x_min) / max(x_max - x_min, 1e-12)
                        ny = (y_col[k] - y_min) / max(y_max - y_min, 1e-12)
                        cx = pb[0] + nx * pb[2]
                        cy = pb[1] + (1.0 - ny) * pb[3]
                        builder.add(self._svg_circle(cx, cy, 3, color, 0.6))

        for algo_name in results:
            legend_items.append({"label": algo_name, "color": self._color_for_algorithm(algo_name)})
        self._add_legend(builder, legend_items, "upper right")

        svg = builder.render()
        data_out = {
            "diversity_metrics": diversity_metrics,
            "quality_metrics": quality_metrics,
            "algorithms": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()},
        }
        result = {"svg": svg, "html": self._generate_html_plot({"svg": svg, "title": "Tradeoff Curves"}), "data": data_out}
        self._maybe_save(result)
        return result

    # ======================================================================
    # SVG primitive helpers
    # ======================================================================

    @staticmethod
    def _svg_circle(cx: float, cy: float, r: float, fill: str, opacity: float) -> str:
        """Return an SVG ``<circle>`` element string."""
        return (
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.1f}" '
            f'fill="{fill}" opacity="{opacity:.3f}"/>'
        )

    @staticmethod
    def _svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float) -> str:
        """Return an SVG ``<line>`` element string."""
        return (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
        )

    @staticmethod
    def _svg_text(
        x: float,
        y: float,
        text: str,
        font_size: int,
        anchor: str = "middle",
        cls: str = "",
        transform: str = "",
    ) -> str:
        """Return an SVG ``<text>`` element string."""
        escaped = html_mod.escape(str(text))
        cls_attr = f' class="{cls}"' if cls else ""
        transform_attr = f' transform="{transform}"' if transform else ""
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-size="{font_size}"{cls_attr}{transform_attr}>{escaped}</text>'
        )

    @staticmethod
    def _svg_polygon(
        points: Sequence[Tuple[float, float]],
        fill: str,
        stroke: str,
        opacity: float,
    ) -> str:
        """Return an SVG ``<polygon>`` element string."""
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke and stroke != "none" else ""
        return (
            f'<polygon points="{pts_str}" fill="{fill}" '
            f'fill-opacity="{opacity:.3f}"{stroke_attr}/>'
        )

    @staticmethod
    def _svg_rect(
        x: float, y: float, w: float, h: float, fill: str, opacity: float
    ) -> str:
        """Return an SVG ``<rect>`` element string."""
        return (
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" opacity="{opacity:.3f}"/>'
        )

    @staticmethod
    def _svg_polyline(
        points: Sequence[Tuple[float, float]],
        stroke: str,
        width: float,
        dash: str = "",
        opacity: float = 1.0,
    ) -> str:
        """Return an SVG ``<polyline>`` element string."""
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts_str}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.1f}" opacity="{opacity:.3f}"{dash_attr}/>'
        )

    # ======================================================================
    # SVG document builder
    # ======================================================================

    @staticmethod
    def _build_svg_document(width: int, height: int, elements: List[str]) -> str:
        """Wrap a list of SVG element strings into a full SVG document."""
        header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            f'style="background:#ffffff;">'
        )
        return "\n".join([header] + elements + ["</svg>"])

    # ======================================================================
    # HTML / SVG generation
    # ======================================================================

    def _generate_svg(self, plot_data: Dict[str, Any]) -> str:
        """Extract or generate SVG from *plot_data*.

        If *plot_data* already has an ``"svg"`` key, return it.
        Otherwise build a minimal placeholder SVG.
        """
        if "svg" in plot_data:
            return plot_data["svg"]
        w, h = self.config.figsize
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<text x="{w // 2}" y="{h // 2}" text-anchor="middle" font-size="18">'
            f'No SVG data</text></svg>'
        )

    def _generate_html_plot(self, plot_data: Dict[str, Any]) -> str:
        """Return a self-contained HTML page embedding the SVG plot."""
        svg = self._generate_svg(plot_data)
        title = html_mod.escape(plot_data.get("title", "Pareto Plot"))
        return (
            "<!DOCTYPE html>\n"
            "<html lang='en'>\n<head>\n"
            f"<meta charset='utf-8'/>\n<title>{title}</title>\n"
            "<style>\n"
            "body { margin: 0; display: flex; justify-content: center; "
            "align-items: center; min-height: 100vh; background: #f5f5f5; }\n"
            ".plot-container { background: #fff; padding: 16px; "
            "border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }\n"
            "</style>\n</head>\n<body>\n"
            f"<div class='plot-container'>\n{svg}\n</div>\n"
            "</body>\n</html>"
        )

    # ======================================================================
    # Colour / marker assignment
    # ======================================================================

    def _color_for_algorithm(self, algo_name: str) -> str:
        """Return a deterministic hex colour for *algo_name*."""
        if algo_name in self._algo_color_cache:
            return self._algo_color_cache[algo_name]
        # Check if explicitly set in the palette
        if algo_name in self.config.color_palette:
            c = self.config.color_palette[algo_name]
            self._algo_color_cache[algo_name] = c
            return c
        idx = self._algo_counter % len(_PALETTE_ORDER)
        c = _PALETTE_ORDER[idx]
        self._algo_color_cache[algo_name] = c
        self._algo_counter += 1
        return c

    def _marker_for_algorithm(self, algo_name: str) -> str:
        """Return a marker shape name for *algo_name*."""
        if algo_name in self._algo_marker_cache:
            return self._algo_marker_cache[algo_name]
        idx = len(self._algo_marker_cache) % len(_MARKER_ORDER)
        m = _MARKER_ORDER[idx]
        self._algo_marker_cache[algo_name] = m
        return m

    # ======================================================================
    # Axis utilities
    # ======================================================================

    @staticmethod
    def _normalize_axis(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Min-max normalise *values* to [0, 1].

        Returns ``(normalised, min_val, max_val)``.  If all values are equal
        a small epsilon is added to *max_val* to avoid division by zero.
        """
        values = np.asarray(values, dtype=float)
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if max_val - min_val < 1e-12:
            max_val = min_val + 1e-6
        norm = (values - min_val) / (max_val - min_val)
        return norm, min_val, max_val

    @staticmethod
    def _format_label(value: float, precision: int = 2) -> str:
        """Format a numeric label with *precision* significant digits."""
        if abs(value) < 1e-12:
            return "0"
        if abs(value) >= 1e4 or abs(value) < 1e-2:
            return f"{value:.{precision}e}"
        return f"{value:.{precision}f}"

    # ======================================================================
    # Grid, axes, ticks, legend helpers (mutate builder)
    # ======================================================================

    def _add_grid_lines(
        self,
        builder: _SVGBuilder,
        bounds: Tuple[float, float, float, float],
        n_lines: int = 6,
    ) -> None:
        """Add grid lines to *builder* inside *bounds* = (x0, y0, w, h)."""
        x0, y0, pw, ph = bounds
        for i in range(n_lines + 1):
            frac = i / max(n_lines, 1)
            # horizontal
            y = y0 + frac * ph
            builder.add(self._svg_line(x0, y, x0 + pw, y, "#e8e8e8", 0.7))
            # vertical
            x = x0 + frac * pw
            builder.add(self._svg_line(x, y0, x, y0 + ph, "#e8e8e8", 0.7))

    @staticmethod
    def _add_axes(
        builder: _SVGBuilder,
        bounds: Tuple[float, float, float, float],
    ) -> None:
        """Draw the x and y axis lines."""
        x0, y0, pw, ph = bounds
        # x-axis (bottom)
        builder.add(
            f'<line x1="{x0:.1f}" y1="{y0 + ph:.1f}" '
            f'x2="{x0 + pw:.1f}" y2="{y0 + ph:.1f}" stroke="#333" stroke-width="1.2"/>'
        )
        # y-axis (left)
        builder.add(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" '
            f'x2="{x0:.1f}" y2="{y0 + ph:.1f}" stroke="#333" stroke-width="1.2"/>'
        )

    def _add_axis_labels(
        self,
        builder: _SVGBuilder,
        xlabel: str,
        ylabel: str,
        bounds: Tuple[float, float, float, float],
    ) -> None:
        """Add axis title labels."""
        x0, y0, pw, ph = bounds
        cfg = self.config
        # x-axis label
        builder.add(
            self._svg_text(x0 + pw / 2, y0 + ph + 45, xlabel, cfg.font_size, "middle", cls="axis-label")
        )
        # y-axis label (rotated)
        lx = x0 - 55
        ly = y0 + ph / 2
        builder.add(
            self._svg_text(
                lx, ly, ylabel, cfg.font_size, "middle", cls="axis-label",
                transform=f"rotate(-90, {lx}, {ly})"
            )
        )

    def _add_tick_labels(
        self,
        builder: _SVGBuilder,
        bounds: Tuple[float, float, float, float],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n_ticks: int = 6,
    ) -> None:
        """Add numeric tick labels along both axes."""
        x0, y0, pw, ph = bounds
        cfg = self.config
        for i in range(n_ticks + 1):
            frac = i / max(n_ticks, 1)
            # x-axis
            xv = x_min + frac * (x_max - x_min)
            xp = x0 + frac * pw
            builder.add(self._svg_text(xp, y0 + ph + 18, self._format_label(xv, 2), cfg.font_size - 3, "middle", cls="tick-label"))
            builder.add(self._svg_line(xp, y0 + ph, xp, y0 + ph + 4, "#333", 1.0))
            # y-axis
            yv = y_min + frac * (y_max - y_min)
            yp = y0 + ph - frac * ph
            builder.add(self._svg_text(x0 - 8, yp + 4, self._format_label(yv, 2), cfg.font_size - 3, "end", cls="tick-label"))
            builder.add(self._svg_line(x0 - 4, yp, x0, yp, "#333", 1.0))

    def _add_legend(
        self,
        builder: _SVGBuilder,
        items: List[Dict[str, str]],
        position: str = "upper right",
    ) -> None:
        """Add a legend box to the SVG."""
        if not items:
            return
        cfg = self.config
        entry_h = 20
        box_w = 150
        box_h = len(items) * entry_h + 10

        # position mapping
        if "right" in position:
            lx = cfg.figsize[0] - cfg.margin_right - box_w - 5
        else:
            lx = cfg.margin_left + 10
        if "lower" in position:
            ly = cfg.figsize[1] - cfg.margin_bottom - box_h - 5
        else:
            ly = cfg.margin_top + 10

        builder.add(self._svg_rect(lx, ly, box_w, box_h, "#ffffff", 0.88))
        builder.add(
            f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{box_w}" height="{box_h}" '
            f'fill="none" stroke="#cccccc" stroke-width="1"/>'
        )

        for idx, item in enumerate(items):
            ey = ly + 8 + idx * entry_h
            builder.add(self._svg_rect(lx + 8, ey, 12, 12, item["color"], 0.9))
            builder.add(
                self._svg_text(
                    lx + 26, ey + 11, item["label"], cfg.font_size - 3, "start", cls="tick-label"
                )
            )

    # ======================================================================
    # Geometry helpers
    # ======================================================================

    @staticmethod
    def _polygon_area(vertices: Sequence[Tuple[float, float]]) -> float:
        """Compute the area of a simple polygon using the shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> List[int]:
        """Compute the convex hull of a set of 2-D points (Andrew's monotone chain).

        Returns indices into *points* that form the hull in counter-clockwise order.
        """
        points = np.asarray(points, dtype=float)
        n = len(points)
        if n <= 2:
            return list(range(n))

        idx = np.lexsort((points[:, 1], points[:, 0]))

        def _cross(o: int, a: int, b: int) -> float:
            return float(
                (points[a, 0] - points[o, 0]) * (points[b, 1] - points[o, 1])
                - (points[a, 1] - points[o, 1]) * (points[b, 0] - points[o, 0])
            )

        lower: List[int] = []
        for i in idx:
            while len(lower) >= 2 and _cross(lower[-2], lower[-1], i) <= 0:
                lower.pop()
            lower.append(int(i))

        upper: List[int] = []
        for i in reversed(idx):
            while len(upper) >= 2 and _cross(upper[-2], upper[-1], i) <= 0:
                upper.pop()
            upper.append(int(i))

        hull = lower[:-1] + upper[:-1]
        return hull

    @staticmethod
    def _interpolate_frontier(
        frontier_points: np.ndarray,
        n_interp: int = 100,
    ) -> np.ndarray:
        """Linearly interpolate frontier points to produce a smooth curve.

        Parameters
        ----------
        frontier_points : ndarray (F, 2) sorted by x.
        n_interp : number of output points.

        Returns
        -------
        ndarray (n_interp, 2)
        """
        frontier_points = np.asarray(frontier_points, dtype=float)
        if len(frontier_points) < 2:
            return frontier_points

        # cumulative arc-length parameterisation
        diffs = np.diff(frontier_points, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum[-1]
        if total < 1e-12:
            return frontier_points

        t_new = np.linspace(0.0, total, n_interp)
        x_interp = np.interp(t_new, cum, frontier_points[:, 0])
        y_interp = np.interp(t_new, cum, frontier_points[:, 1])
        return np.column_stack([x_interp, y_interp])

    # ======================================================================
    # Marker rendering
    # ======================================================================

    def _render_marker(
        self,
        cx: float,
        cy: float,
        r: float,
        shape: str,
        color: str,
        opacity: float,
    ) -> str:
        """Render a marker of given *shape* centred at (cx, cy)."""
        if shape == "square":
            half = r * 0.85
            return self._svg_rect(cx - half, cy - half, 2 * half, 2 * half, color, opacity)
        if shape == "diamond":
            pts = [
                (cx, cy - r),
                (cx + r, cy),
                (cx, cy + r),
                (cx - r, cy),
            ]
            return self._svg_polygon(pts, color, "none", opacity)
        if shape == "triangle_up":
            h = r * 1.1
            pts = [
                (cx, cy - h),
                (cx - h * 0.866, cy + h * 0.5),
                (cx + h * 0.866, cy + h * 0.5),
            ]
            return self._svg_polygon(pts, color, "none", opacity)
        if shape == "triangle_down":
            h = r * 1.1
            pts = [
                (cx, cy + h),
                (cx - h * 0.866, cy - h * 0.5),
                (cx + h * 0.866, cy - h * 0.5),
            ]
            return self._svg_polygon(pts, color, "none", opacity)
        if shape == "star":
            return self._star_path(cx, cy, r, color, opacity)
        if shape == "cross":
            arm = r * 0.35
            lines = [
                self._svg_line(cx - r, cy, cx + r, cy, color, arm * 3),
                self._svg_line(cx, cy - r, cx, cy + r, color, arm * 3),
            ]
            return "\n".join(lines)
        if shape == "plus":
            arm = r * 0.3
            return "\n".join([
                self._svg_line(cx - r, cy, cx + r, cy, color, arm * 2.5),
                self._svg_line(cx, cy - r, cx, cy + r, color, arm * 2.5),
            ])
        if shape == "hexagon":
            pts = [
                (cx + r * math.cos(math.pi / 3 * i), cy + r * math.sin(math.pi / 3 * i))
                for i in range(6)
            ]
            return self._svg_polygon(pts, color, "none", opacity)
        if shape == "pentagon":
            pts = [
                (cx + r * math.cos(2 * math.pi * i / 5 - math.pi / 2),
                 cy + r * math.sin(2 * math.pi * i / 5 - math.pi / 2))
                for i in range(5)
            ]
            return self._svg_polygon(pts, color, "none", opacity)
        # default: circle
        return self._svg_circle(cx, cy, r, color, opacity)

    @staticmethod
    def _star_path(cx: float, cy: float, r: float, color: str, opacity: float) -> str:
        """Five-pointed star as SVG polygon."""
        outer = r
        inner = r * 0.45
        pts: List[Tuple[float, float]] = []
        for i in range(5):
            angle_outer = -math.pi / 2 + 2 * math.pi * i / 5
            angle_inner = angle_outer + math.pi / 5
            pts.append((cx + outer * math.cos(angle_outer), cy + outer * math.sin(angle_outer)))
            pts.append((cx + inner * math.cos(angle_inner), cy + inner * math.sin(angle_inner)))
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return (
            f'<polygon points="{pts_str}" fill="{color}" '
            f'fill-opacity="{opacity:.3f}"/>'
        )

    # ======================================================================
    # Line-style helpers
    # ======================================================================

    @staticmethod
    def _dash_for_style(style: str) -> str:
        """Return SVG stroke-dasharray for a line style name."""
        if style == "dashed":
            return "8,4"
        if style == "dotted":
            return "2,4"
        return ""

    # ======================================================================
    # Export helpers
    # ======================================================================

    def export_plot_data(self, plot_data: Dict[str, Any], fmt: str = "json") -> str:
        """Serialise the ``data`` portion of a plot result to JSON or CSV.

        Parameters
        ----------
        plot_data : dict returned by any ``plot_*`` method.
        fmt : ``"json"`` or ``"csv"``.
        """
        data = plot_data.get("data", {})
        if fmt == "json":
            return json.dumps(data, indent=2, default=_json_default)
        if fmt == "csv":
            return self._data_to_csv(data)
        raise ValueError(f"Unsupported format: {fmt!r}")

    @staticmethod
    def _data_to_csv(data: Dict[str, Any]) -> str:
        """Best-effort conversion of a nested data dict to CSV."""
        lines: List[str] = []
        # Attempt to find a "points" key which is a list-of-lists
        if "points" in data and isinstance(data["points"], list):
            pts = data["points"]
            if pts and isinstance(pts[0], list):
                header = ",".join(f"obj_{i}" for i in range(len(pts[0])))
                lines.append(header)
                for row in pts:
                    lines.append(",".join(str(v) for v in row))
                return "\n".join(lines)
        # Fallback: per-algorithm frontiers
        if "algorithms" in data and isinstance(data["algorithms"], dict):
            first = True
            for algo, vals in data["algorithms"].items():
                if isinstance(vals, list) and vals and isinstance(vals[0], list):
                    if first:
                        n_cols = len(vals[0])
                        header = "algorithm," + ",".join(f"obj_{i}" for i in range(n_cols))
                        lines.append(header)
                        first = False
                    for row in vals:
                        lines.append(algo + "," + ",".join(str(v) for v in row))
                elif isinstance(vals, list):
                    if first:
                        lines.append("algorithm,value")
                        first = False
                    for v in vals:
                        lines.append(f"{algo},{v}")
            if lines:
                return "\n".join(lines)
        # Contributions dict
        if "contributions" in data:
            lines.append("algorithm,contribution")
            for k, v in data["contributions"].items():
                lines.append(f"{k},{v}")
            return "\n".join(lines)
        # generic: dump as key,value
        lines.append("key,value")
        for k, v in data.items():
            lines.append(f"{k},{v}")
        return "\n".join(lines)

    # ======================================================================
    # Dashboard
    # ======================================================================

    def generate_comparison_dashboard(
        self,
        all_results: Dict[str, Any],
    ) -> str:
        """Build a multi-panel HTML dashboard from a collection of plot results.

        Parameters
        ----------
        all_results : dict mapping panel titles to plot result dicts (each
            containing at least an ``"svg"`` key).

        Returns
        -------
        str – self-contained HTML page.
        """
        n_panels = len(all_results)
        if n_panels == 0:
            return "<html><body><p>No panels</p></body></html>"

        # Grid layout: up to 2 columns
        n_cols = min(n_panels, 2)
        col_pct = 100 // n_cols

        panels_html: List[str] = []
        for panel_title, result in all_results.items():
            svg = result.get("svg", "")
            escaped_title = html_mod.escape(panel_title)
            panels_html.append(
                f'<div class="panel" style="width:{col_pct - 2}%;">'
                f'<h3>{escaped_title}</h3>'
                f'<div class="svg-wrap">{svg}</div>'
                f'</div>'
            )

        dashboard_html = (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='utf-8'/>\n<title>Comparison Dashboard</title>\n"
            "<style>\n"
            "* { box-sizing: border-box; }\n"
            "body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #f0f2f5; }\n"
            "h1 { text-align: center; color: #333; }\n"
            ".dashboard { display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; }\n"
            ".panel { background: #fff; border-radius: 8px; padding: 16px; "
            "box-shadow: 0 2px 6px rgba(0,0,0,0.10); }\n"
            ".panel h3 { margin: 0 0 12px 0; font-size: 15px; color: #444; }\n"
            ".svg-wrap svg { width: 100%; height: auto; }\n"
            "</style>\n</head>\n<body>\n"
            "<h1>Diversity Decoding Arena &mdash; Comparison Dashboard</h1>\n"
            '<div class="dashboard">\n'
            + "\n".join(panels_html) +
            "\n</div>\n</body>\n</html>"
        )
        return dashboard_html

    # ======================================================================
    # Animation HTML builder
    # ======================================================================

    def _build_animation_html(
        self,
        frames_svg: List[str],
        fps: int = 2,
        title: str = "Animation",
    ) -> str:
        """Build an HTML page that cycles through SVG frames using JavaScript."""
        escaped_title = html_mod.escape(title)
        interval_ms = int(1000 / max(fps, 1))

        # JSON-encode frames (each is an SVG string)
        frames_json = json.dumps(frames_svg)

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            f"<meta charset='utf-8'/>\n<title>{escaped_title}</title>\n"
            "<style>\n"
            "body { margin: 0; display: flex; flex-direction: column; "
            "align-items: center; padding: 20px; background: #f5f5f5; "
            "font-family: Arial, sans-serif; }\n"
            "#frame-container { background: #fff; padding: 16px; "
            "border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }\n"
            ".controls { margin-top: 12px; }\n"
            "button { padding: 6px 16px; margin: 0 4px; cursor: pointer; "
            "border: 1px solid #ccc; border-radius: 4px; background: #fff; }\n"
            "button:hover { background: #eee; }\n"
            "#info { margin-top: 8px; font-size: 14px; color: #666; }\n"
            "</style>\n</head>\n<body>\n"
            f"<h2>{escaped_title}</h2>\n"
            '<div id="frame-container"></div>\n'
            '<div class="controls">\n'
            '  <button id="btn-prev">&#9664; Prev</button>\n'
            '  <button id="btn-play">&#9654; Play</button>\n'
            '  <button id="btn-next">Next &#9654;</button>\n'
            '</div>\n'
            '<div id="info">Frame 1 / 1</div>\n'
            "<script>\n"
            f"const frames = {frames_json};\n"
            "let idx = 0, timer = null;\n"
            "const container = document.getElementById('frame-container');\n"
            "const info = document.getElementById('info');\n"
            "function show(i) {\n"
            "  idx = ((i % frames.length) + frames.length) % frames.length;\n"
            "  container.innerHTML = frames[idx];\n"
            "  info.textContent = 'Frame ' + (idx+1) + ' / ' + frames.length;\n"
            "}\n"
            "document.getElementById('btn-prev').onclick = () => { stop(); show(idx-1); };\n"
            "document.getElementById('btn-next').onclick = () => { stop(); show(idx+1); };\n"
            "function stop() { if(timer) { clearInterval(timer); timer=null; } }\n"
            "document.getElementById('btn-play').onclick = () => {\n"
            f"  if(timer) stop(); else timer = setInterval(() => show(idx+1), {interval_ms});\n"
            "};\n"
            "show(0);\n"
            "</script>\n</body>\n</html>"
        )

    # ======================================================================
    # Save helper
    # ======================================================================

    def _maybe_save(self, result: Dict[str, Any]) -> None:
        """Write SVG/HTML to disk if ``config.save_path`` is set."""
        if not self.config.save_path:
            return
        content = result.get("html", result.get("svg", ""))
        try:
            with open(self.config.save_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass  # silently skip if path is invalid


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for ``json.dumps``."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Extended Pareto visualizations (3D surface, animation, family comparison)
# ---------------------------------------------------------------------------


class ParetoSurface3D:
    """3D Pareto surface visualization rendered as an SVG projection."""

    def __init__(self, config: Optional[ParetoPlotConfig] = None) -> None:
        self.config = config or ParetoPlotConfig()

    @staticmethod
    def _project_3d(
        x: float, y: float, z: float,
        angle_x: float = 0.6, angle_y: float = 0.8,
        scale: float = 1.0,
    ) -> Tuple[float, float]:
        """Simple isometric projection from 3D to 2D."""
        px = (x - z * math.cos(angle_x)) * scale
        py = (-y + z * math.sin(angle_y) * 0.5) * scale
        return px, py

    @staticmethod
    def _is_pareto_3d(points: np.ndarray, maximize: bool = True) -> np.ndarray:
        """Return boolean mask of Pareto-optimal points in 3D."""
        n = points.shape[0]
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maximize:
                    if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                        is_pareto[i] = False
                        break
                else:
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        is_pareto[i] = False
                        break
        return is_pareto

    def plot_3d_surface(
        self,
        points: np.ndarray,
        labels: Optional[List[str]] = None,
        objective_names: Optional[List[str]] = None,
        title: str = "3D Pareto Surface",
    ) -> str:
        """Render 3D Pareto surface as projected SVG."""
        cfg = self.config
        w, h = cfg.figsize
        if points.shape[1] < 3:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80">' \
                   '<text x="150" y="40" text-anchor="middle">Need 3 objectives</text></svg>'

        obj_names = objective_names or ["Obj 1", "Obj 2", "Obj 3"]
        pareto_mask = self._is_pareto_3d(points)

        # normalize to [0,1]
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        rng = maxs - mins
        rng[rng == 0] = 1.0
        normed = (points - mins) / rng

        scale = min(w, h) * 0.35
        cx_off, cy_off = w * 0.5, h * 0.55

        svg = _SVGBuilder(w, h)
        svg.add(f'<text x="{w/2}" y="28" text-anchor="middle" '
                f'font-size="{cfg.font_size + 2}" font-weight="bold">{title}</text>')

        # draw axes
        origin = self._project_3d(0, 0, 0, scale=scale)
        for dim, label in enumerate(obj_names[:3]):
            end_pt = [0.0, 0.0, 0.0]
            end_pt[dim] = 1.1
            ex, ey = self._project_3d(*end_pt, scale=scale)
            svg.add(
                f'<line x1="{origin[0]+cx_off}" y1="{origin[1]+cy_off}" '
                f'x2="{ex+cx_off}" y2="{ey+cy_off}" '
                f'stroke="#666" stroke-width="1.5" '
                f'marker-end="url(#arrowhead)"/>'
            )
            lx, ly = self._project_3d(*(v * 1.2 for v in end_pt), scale=scale)
            svg.add(
                f'<text x="{lx+cx_off}" y="{ly+cy_off}" text-anchor="middle" '
                f'font-size="{cfg.font_size - 2}">{label}</text>'
            )

        svg.add_def(
            '<marker id="arrowhead" markerWidth="6" markerHeight="4" '
            'refX="6" refY="2" orient="auto">'
            '<polygon points="0 0, 6 2, 0 4" fill="#666"/></marker>'
        )

        # draw dominated points first
        for i in range(len(points)):
            if pareto_mask[i]:
                continue
            px, py = self._project_3d(normed[i, 0], normed[i, 1], normed[i, 2],
                                       scale=scale)
            svg.add(
                f'<circle cx="{px+cx_off}" cy="{py+cy_off}" r="3" '
                f'fill="#aaa" opacity="0.4"/>'
            )

        # draw Pareto points
        for i in range(len(points)):
            if not pareto_mask[i]:
                continue
            px, py = self._project_3d(normed[i, 0], normed[i, 1], normed[i, 2],
                                       scale=scale)
            color = _PALETTE_ORDER[i % len(_PALETTE_ORDER)]
            svg.add(
                f'<circle cx="{px+cx_off}" cy="{py+cy_off}" r="5" '
                f'fill="{color}" stroke="#333" stroke-width="1"/>'
            )
            if labels and i < len(labels):
                svg.add(
                    f'<text x="{px+cx_off+7}" y="{py+cy_off+3}" '
                    f'font-size="{cfg.font_size - 3}">{labels[i]}</text>'
                )

        # connect Pareto surface (convex hull on projected points)
        pareto_pts = normed[pareto_mask]
        if len(pareto_pts) >= 3:
            projected = []
            for pt in pareto_pts:
                px, py = self._project_3d(pt[0], pt[1], pt[2], scale=scale)
                projected.append((px + cx_off, py + cy_off))
            # sort by angle from centroid
            cx_c = sum(p[0] for p in projected) / len(projected)
            cy_c = sum(p[1] for p in projected) / len(projected)
            projected.sort(key=lambda p: math.atan2(p[1] - cy_c, p[0] - cx_c))
            poly_str = " ".join(f"{p[0]},{p[1]}" for p in projected)
            svg.add(
                f'<polygon points="{poly_str}" fill="#1f77b4" '
                f'fill-opacity="0.15" stroke="#1f77b4" stroke-width="1.5" '
                f'stroke-dasharray="4,3"/>'
            )

        return svg.render()


class ParetoFrontierEvolution:
    """Animated Pareto frontier evolution over iterations."""

    def __init__(self, config: Optional[ParetoPlotConfig] = None) -> None:
        self.config = config or ParetoPlotConfig()

    def plot_frontier_evolution(
        self,
        snapshots: List[np.ndarray],
        snapshot_labels: Optional[List[str]] = None,
        title: str = "Pareto Frontier Evolution",
    ) -> str:
        """Generate HTML with animated SVG frames showing frontier evolution."""
        cfg = self.config
        w, h = cfg.figsize
        n_frames = len(snapshots)
        if n_frames == 0:
            return "<html><body><p>No snapshots</p></body></html>"

        s_labels = snapshot_labels or [f"Step {i}" for i in range(n_frames)]

        # find global bounds
        all_pts = np.vstack(snapshots)
        x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
        x_rng = x_max - x_min if x_max != x_min else 1.0
        y_rng = y_max - y_min if y_max != y_min else 1.0

        ml, mr, mt, mb = cfg.margin_left, cfg.margin_right, cfg.margin_top, cfg.margin_bottom
        pw = w - ml - mr
        ph = h - mt - mb

        frames_html: List[str] = []
        for fi, pts in enumerate(snapshots):
            svg = _SVGBuilder(w, h)
            svg.add(f'<text x="{w/2}" y="28" text-anchor="middle" '
                    f'font-size="{cfg.font_size+2}" font-weight="bold">'
                    f'{title} — {s_labels[fi]}</text>')

            # grid
            svg.add(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
                    f'fill="#fafafa" stroke="#ccc"/>')
            for gi in range(5):
                gy = mt + ph * gi / 4
                svg.add(f'<line x1="{ml}" y1="{gy}" x2="{ml+pw}" y2="{gy}" '
                        f'stroke="#eee" stroke-width="0.5"/>')

            # compute Pareto front for this snapshot
            is_pareto = np.ones(len(pts), dtype=bool)
            for i in range(len(pts)):
                for j in range(len(pts)):
                    if i != j and np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                        is_pareto[i] = False
                        break

            # plot points
            for i in range(len(pts)):
                sx = ml + (pts[i, 0] - x_min) / x_rng * pw
                sy = mt + ph - (pts[i, 1] - y_min) / y_rng * ph
                color = "#d62728" if is_pareto[i] else "#aaa"
                r = 5 if is_pareto[i] else 3
                opacity = 0.9 if is_pareto[i] else 0.4
                svg.add(
                    f'<circle cx="{sx}" cy="{sy}" r="{r}" '
                    f'fill="{color}" opacity="{opacity}"/>'
                )

            # frontier line
            frontier_pts = pts[is_pareto]
            if len(frontier_pts) > 1:
                order = np.argsort(frontier_pts[:, 0])
                frontier_sorted = frontier_pts[order]
                path_d = []
                for k, fp in enumerate(frontier_sorted):
                    sx = ml + (fp[0] - x_min) / x_rng * pw
                    sy = mt + ph - (fp[1] - y_min) / y_rng * ph
                    cmd = "M" if k == 0 else "L"
                    path_d.append(f"{cmd}{sx},{sy}")
                svg.add(
                    f'<path d="{" ".join(path_d)}" fill="none" '
                    f'stroke="#d62728" stroke-width="2"/>'
                )

            frames_html.append(svg.render())

        # wrap in HTML with JavaScript animation
        html_parts = [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<style>body{font-family:sans-serif;text-align:center;}'
            '.frame{display:none;}.frame.active{display:block;}'
            '#controls{margin:10px;}</style></head><body>',
            '<div id="controls">',
            '<button onclick="prev()">◀ Prev</button>',
            '<span id="label" style="margin:0 15px;"></span>',
            '<button onclick="next()">Next ▶</button>',
            '<button onclick="togglePlay()" id="playBtn">▶ Play</button>',
            '</div>',
        ]
        for fi, svg_str in enumerate(frames_html):
            cls = "frame active" if fi == 0 else "frame"
            html_parts.append(f'<div class="{cls}" id="f{fi}">{svg_str}</div>')

        html_parts.append(f'''
<script>
var cur=0, n={n_frames}, playing=false, timer=null;
var labels={json.dumps(s_labels)};
function show(i){{
  document.getElementById("f"+cur).className="frame";
  cur=i; document.getElementById("f"+cur).className="frame active";
  document.getElementById("label").textContent=labels[cur];
}}
function next(){{ show((cur+1)%n); }}
function prev(){{ show((cur-1+n)%n); }}
function togglePlay(){{
  playing=!playing;
  document.getElementById("playBtn").textContent=playing?"⏸ Pause":"▶ Play";
  if(playing){{ timer=setInterval(next, 800); }}
  else {{ clearInterval(timer); }}
}}
show(0);
</script></body></html>''')
        return "".join(html_parts)


class ParetoFamilyComparison:
    """Compare Pareto frontiers between algorithm families."""

    def __init__(self, config: Optional[ParetoPlotConfig] = None) -> None:
        self.config = config or ParetoPlotConfig()

    def plot_family_comparison(
        self,
        families: Dict[str, np.ndarray],
        title: str = "Pareto Frontier Comparison by Algorithm Family",
    ) -> str:
        """Overlay Pareto frontiers from different algorithm families."""
        cfg = self.config
        w, h = cfg.figsize
        if not families:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80">' \
                   '<text x="150" y="40" text-anchor="middle">No data</text></svg>'

        family_names = sorted(families.keys())
        all_pts = np.vstack(list(families.values()))
        x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
        x_rng = x_max - x_min if x_max != x_min else 1.0
        y_rng = y_max - y_min if y_max != y_min else 1.0

        ml, mt = cfg.margin_left, cfg.margin_top
        pw, ph = cfg.plot_width, cfg.plot_height

        svg = _SVGBuilder(w, h)
        svg.add(f'<text x="{w/2}" y="28" text-anchor="middle" '
                f'font-size="{cfg.font_size+2}" font-weight="bold">{title}</text>')
        svg.add(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
                f'fill="#fafafa" stroke="#ccc"/>')

        for fi, fname in enumerate(family_names):
            pts = families[fname]
            color = _PALETTE_ORDER[fi % len(_PALETTE_ORDER)]

            # compute Pareto front
            is_pareto = np.ones(len(pts), dtype=bool)
            for i in range(len(pts)):
                for j in range(len(pts)):
                    if i != j and np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                        is_pareto[i] = False
                        break

            # scatter all points
            for i in range(len(pts)):
                sx = ml + (pts[i, 0] - x_min) / x_rng * pw
                sy = mt + ph - (pts[i, 1] - y_min) / y_rng * ph
                r = 4 if is_pareto[i] else 2.5
                opacity = 0.8 if is_pareto[i] else 0.3
                svg.add(
                    f'<circle cx="{sx}" cy="{sy}" r="{r}" '
                    f'fill="{color}" opacity="{opacity}"/>'
                )

            # frontier line
            frontier_pts = pts[is_pareto]
            if len(frontier_pts) > 1:
                order = np.argsort(frontier_pts[:, 0])
                frontier_sorted = frontier_pts[order]
                path_d = []
                for k, fp in enumerate(frontier_sorted):
                    sx = ml + (fp[0] - x_min) / x_rng * pw
                    sy = mt + ph - (fp[1] - y_min) / y_rng * ph
                    cmd = "M" if k == 0 else "L"
                    path_d.append(f"{cmd}{sx},{sy}")
                svg.add(
                    f'<path d="{" ".join(path_d)}" fill="none" '
                    f'stroke="{color}" stroke-width="2.5"/>'
                )

                # dominated region shading
                shade_d = list(path_d)
                last_fp = frontier_sorted[-1]
                lx = ml + (last_fp[0] - x_min) / x_rng * pw
                shade_d.append(f"L{lx},{mt+ph}")
                first_fp = frontier_sorted[0]
                fx = ml + (first_fp[0] - x_min) / x_rng * pw
                shade_d.append(f"L{fx},{mt+ph}")
                shade_d.append("Z")
                svg.add(
                    f'<path d="{" ".join(shade_d)}" '
                    f'fill="{color}" fill-opacity="0.08" stroke="none"/>'
                )

        # legend
        leg_x = ml + pw - 140
        leg_y = mt + 15
        svg.add(f'<rect x="{leg_x-5}" y="{leg_y-12}" width="145" '
                f'height="{len(family_names)*20+10}" fill="white" '
                f'stroke="#ccc" rx="3"/>')
        for fi, fname in enumerate(family_names):
            color = _PALETTE_ORDER[fi % len(_PALETTE_ORDER)]
            cy = leg_y + fi * 20
            svg.add(f'<rect x="{leg_x}" y="{cy-4}" width="12" height="12" '
                    f'fill="{color}" rx="2"/>')
            svg.add(f'<text x="{leg_x+18}" y="{cy+6}" '
                    f'font-size="{cfg.font_size-2}">{fname}</text>')

        # axis labels
        svg.add(f'<text x="{ml+pw/2}" y="{mt+ph+40}" text-anchor="middle" '
                f'font-size="{cfg.font_size}">{cfg.xlabel}</text>')
        svg.add(f'<text x="18" y="{mt+ph/2}" text-anchor="middle" '
                f'font-size="{cfg.font_size}" '
                f'transform="rotate(-90,18,{mt+ph/2})">{cfg.ylabel}</text>')

        return svg.render()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_default_plotter(**kwargs: Any) -> ParetoPlotter:
    """Create a :class:`ParetoPlotter` with sensible defaults.

    Any keyword argument is forwarded to :class:`ParetoPlotConfig`.
    """
    return ParetoPlotter(ParetoPlotConfig(**kwargs))


# ---------------------------------------------------------------------------
# CLI-style quick demo (not executed on import)
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Quick smoke-test: generate sample plots and print SVG lengths."""
    rng = np.random.RandomState(42)

    # 2-D frontier
    n = 80
    pts = rng.rand(n, 2)
    # compute actual Pareto frontier (maximising both objectives)
    frontier_idx: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and (
                pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]
            ):
                dominated = True
                break
        if not dominated:
            frontier_idx.append(i)

    plotter = ParetoPlotter(
        ParetoPlotConfig(
            title="Demo Pareto Frontier",
            xlabel="Diversity",
            ylabel="Quality",
            annotate_points=True,
            show_frontier_line=True,
        )
    )

    r1 = plotter.plot_2d_frontier(pts, frontier_idx)
    print(f"2D frontier SVG length: {len(r1['svg'])} chars")

    # Confidence band
    frontier_pts = pts[frontier_idx]
    order = np.argsort(frontier_pts[:, 0])
    frontier_sorted = frontier_pts[order]
    offset = rng.rand(len(frontier_sorted), 2) * 0.05
    lower = frontier_sorted - offset
    upper = frontier_sorted + offset
    r2 = plotter.plot_2d_with_confidence(
        pts, frontier_sorted, {"lower": lower, "upper": upper}
    )
    print(f"2D confidence SVG length: {len(r2['svg'])} chars")

    # 3-D
    pts3 = rng.rand(60, 3)
    r3 = plotter.plot_3d_frontier(pts3, list(range(10)))
    print(f"3D frontier SVG length: {len(r3['svg'])} chars")

    # Multiple frontiers
    frontiers = {
        "Beam Search": rng.rand(15, 2),
        "Nucleus Sampling": rng.rand(12, 2),
        "DPP Decoding": rng.rand(18, 2),
    }
    r4 = plotter.plot_multiple_frontiers(frontiers)
    print(f"Multiple frontiers SVG length: {len(r4['svg'])} chars")

    # Hypervolume convergence
    hv = {
        "Beam": np.cumsum(rng.rand(30) * 0.05).tolist(),
        "Nucleus": np.cumsum(rng.rand(30) * 0.04).tolist(),
        "DPP": np.cumsum(rng.rand(30) * 0.06).tolist(),
    }
    r5 = plotter.plot_hypervolume_convergence(hv)
    print(f"HV convergence SVG length: {len(r5['svg'])} chars")

    # Contribution bar
    contribs = {"Beam": 0.23, "Nucleus": 0.35, "DPP": 0.42, "Top-k": 0.18, "MBR": 0.29}
    r6 = plotter.plot_contribution_bar(contribs)
    print(f"Contribution bar SVG length: {len(r6['svg'])} chars")

    # Parallel coordinates
    pc_pts = rng.rand(20, 5)
    r7 = plotter.plot_parallel_coordinates(pc_pts, ["Div", "Qual", "Fluency", "Coh", "Nov"])
    print(f"Parallel coords SVG length: {len(r7['svg'])} chars")

    # Radar chart
    scores = {
        "Beam": [0.7, 0.9, 0.6, 0.5, 0.8],
        "Nucleus": [0.8, 0.7, 0.8, 0.6, 0.7],
        "DPP": [0.9, 0.6, 0.7, 0.9, 0.6],
    }
    r8 = plotter.plot_radar_chart(scores, ["Diversity", "Quality", "Fluency", "Coherence", "Novelty"])
    print(f"Radar chart SVG length: {len(r8['svg'])} chars")

    # Tradeoff curves
    results_td = {
        "Beam": rng.rand(25, 4),
        "Nucleus": rng.rand(25, 4),
    }
    r9 = plotter.plot_tradeoff_curves(results_td, ["Div1", "Div2"], ["Qual1", "Qual2"])
    print(f"Tradeoff curves SVG length: {len(r9['svg'])} chars")

    # Frontier evolution
    evol = [rng.rand(8 + t * 2, 2) for t in range(6)]
    r10 = plotter.plot_frontier_evolution(evol)
    print(f"Frontier evolution HTML length: {len(r10['html'])} chars")

    # Dashboard
    dashboard = plotter.generate_comparison_dashboard({
        "2D Frontier": r1,
        "HV Convergence": r5,
        "Radar": r8,
        "Contributions": r6,
    })
    print(f"Dashboard HTML length: {len(dashboard)} chars")

    # Export
    csv_export = plotter.export_plot_data(r1, "csv")
    json_export = plotter.export_plot_data(r1, "json")
    print(f"CSV export length: {len(csv_export)}, JSON export length: {len(json_export)}")

    # Convex hull test
    hull = ParetoPlotter._convex_hull_2d(pts[:, :2])
    print(f"Convex hull: {len(hull)} vertices")

    # Interpolation test
    interp = ParetoPlotter._interpolate_frontier(frontier_sorted, 50)
    print(f"Interpolated frontier: {interp.shape}")

    # Polygon area test
    area = ParetoPlotter._polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)])
    print(f"Unit square area: {area}")


if __name__ == "__main__":
    _demo()
