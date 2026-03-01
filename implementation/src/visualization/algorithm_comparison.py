"""
Algorithm comparison chart visualization for the Diversity Decoding Arena.

Generates SVG strings directly for all chart types — no matplotlib dependency.
Uses numpy for numerical computation (statistics, KDE, coordinate math).
"""

from __future__ import annotations

import math
import html as html_mod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration & data containers
# ---------------------------------------------------------------------------

@dataclass
class ComparisonConfig:
    """Master configuration for every comparison chart."""

    figsize: Tuple[float, float] = (800, 500)
    color_palette: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#4C72B0",
        "secondary": "#DD8452",
        "tertiary": "#55A868",
        "quaternary": "#C44E52",
        "quinary": "#8172B3",
        "senary": "#937860",
        "septenary": "#DA8BC3",
        "octonary": "#8C8C8C",
        "nonary": "#CCB974",
        "denary": "#64B5CD",
    })
    sort_by: str = "score"
    ascending: bool = False
    show_error_bars: bool = True
    show_significance: bool = True
    reference_algorithm: Optional[str] = None
    title: str = ""
    annotation_format: str = ".3f"
    bar_width: float = 0.6
    group_spacing: float = 0.3
    font_family: str = "Arial, Helvetica, sans-serif"
    font_size: int = 12
    title_size: int = 16
    axis_color: str = "#333333"
    grid_color: str = "#E0E0E0"
    background_color: str = "#FFFFFF"
    margin: Dict[str, float] = field(default_factory=lambda: {
        "top": 60, "right": 40, "bottom": 80, "left": 100,
    })

    # ---- helpers ----
    def palette_list(self) -> List[str]:
        return list(self.color_palette.values())

    def plot_width(self) -> float:
        return self.figsize[0] - self.margin["left"] - self.margin["right"]

    def plot_height(self) -> float:
        return self.figsize[1] - self.margin["top"] - self.margin["bottom"]


@dataclass
class RankingChart:
    """Container returned by ranking-table helpers."""

    algorithm_names: List[str]
    ranks: List[int]
    scores: List[float]
    svg_content: str


# ---------------------------------------------------------------------------
# SVG boilerplate helpers (module-level for reuse)
# ---------------------------------------------------------------------------

def _svg_open(width: float, height: float, bg: str = "#FFFFFF") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="{bg}"/>\n'
    )


def _svg_close() -> str:
    return "</svg>\n"


def _escape(text: str) -> str:
    return html_mod.escape(str(text))


# ---------------------------------------------------------------------------
# AlgorithmComparisonPlotter
# ---------------------------------------------------------------------------

class AlgorithmComparisonPlotter:
    """Generate SVG comparison charts for algorithm benchmarking results."""

    def __init__(self, config: Optional[ComparisonConfig] = None) -> None:
        self.config = config or ComparisonConfig()

    # ------------------------------------------------------------------
    # colour helpers
    # ------------------------------------------------------------------

    def _color(self, index: int) -> str:
        pal = self.config.palette_list()
        return pal[index % len(pal)]

    # ------------------------------------------------------------------
    # low-level SVG primitives
    # ------------------------------------------------------------------

    def _svg_bar(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        label: Optional[str] = None,
        opacity: float = 1.0,
        rx: float = 2,
    ) -> str:
        parts = [
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
            f'height="{max(height, 0):.2f}" fill="{fill}" '
            f'opacity="{opacity}" rx="{rx}"/>'
        ]
        if label is not None:
            lx = x + width / 2
            ly = y - 4
            parts.append(
                f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" '
                f'font-size="{self.config.font_size - 1}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(label)}</text>'
            )
        return "\n".join(parts)

    def _svg_error_bar(
        self,
        x: float,
        y_center: float,
        lower: float,
        upper: float,
        stroke: str = "#333",
        cap_width: float = 6,
    ) -> str:
        half = cap_width / 2
        return (
            f'<line x1="{x:.2f}" y1="{lower:.2f}" x2="{x:.2f}" '
            f'y2="{upper:.2f}" stroke="{stroke}" stroke-width="1.5"/>\n'
            f'<line x1="{x - half:.2f}" y1="{lower:.2f}" '
            f'x2="{x + half:.2f}" y2="{lower:.2f}" stroke="{stroke}" '
            f'stroke-width="1.5"/>\n'
            f'<line x1="{x - half:.2f}" y1="{upper:.2f}" '
            f'x2="{x + half:.2f}" y2="{upper:.2f}" stroke="{stroke}" '
            f'stroke-width="1.5"/>\n'
        )

    def _svg_box_plot(
        self,
        x: float,
        center: float,
        q1: float,
        q3: float,
        whisker_low: float,
        whisker_high: float,
        width: float,
        fill: str = "#4C72B0",
        outliers: Optional[List[float]] = None,
    ) -> str:
        half = width / 2
        left = x - half
        parts: List[str] = []
        # whisker lines
        parts.append(
            f'<line x1="{x:.2f}" y1="{whisker_low:.2f}" '
            f'x2="{x:.2f}" y2="{q1:.2f}" '
            f'stroke="{self.config.axis_color}" stroke-width="1.2" '
            f'stroke-dasharray="4,2"/>'
        )
        parts.append(
            f'<line x1="{x:.2f}" y1="{q3:.2f}" '
            f'x2="{x:.2f}" y2="{whisker_high:.2f}" '
            f'stroke="{self.config.axis_color}" stroke-width="1.2" '
            f'stroke-dasharray="4,2"/>'
        )
        # whisker caps
        cap = width * 0.4
        for wy in (whisker_low, whisker_high):
            parts.append(
                f'<line x1="{x - cap:.2f}" y1="{wy:.2f}" '
                f'x2="{x + cap:.2f}" y2="{wy:.2f}" '
                f'stroke="{self.config.axis_color}" stroke-width="1.5"/>'
            )
        # box
        box_h = abs(q3 - q1)
        box_top = min(q1, q3)
        parts.append(
            f'<rect x="{left:.2f}" y="{box_top:.2f}" '
            f'width="{width:.2f}" height="{box_h:.2f}" '
            f'fill="{fill}" fill-opacity="0.6" '
            f'stroke="{self.config.axis_color}" stroke-width="1.2" rx="2"/>'
        )
        # median line
        parts.append(
            f'<line x1="{left:.2f}" y1="{center:.2f}" '
            f'x2="{left + width:.2f}" y2="{center:.2f}" '
            f'stroke="{self.config.axis_color}" stroke-width="2"/>'
        )
        # outliers
        if outliers:
            for oy in outliers:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{oy:.2f}" r="3" '
                    f'fill="none" stroke="{self.config.axis_color}" '
                    f'stroke-width="1"/>'
                )
        return "\n".join(parts)

    def _compute_box_stats(self, values: List[float]) -> Dict[str, Any]:
        arr = np.asarray(values, dtype=float)
        q1 = float(np.percentile(arr, 25))
        median = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        whisker_low = float(max(np.min(arr), q1 - 1.5 * iqr))
        whisker_high = float(min(np.max(arr), q3 + 1.5 * iqr))
        outliers = arr[(arr < whisker_low) | (arr > whisker_high)].tolist()
        return {
            "median": median,
            "q1": q1,
            "q3": q3,
            "whisker_low": whisker_low,
            "whisker_high": whisker_high,
            "outliers": outliers,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    def _svg_violin(
        self,
        x: float,
        center_y: float,
        density_curve: Tuple[np.ndarray, np.ndarray],
        width: float,
        fill: str,
    ) -> str:
        y_grid, density = density_curve
        if len(y_grid) == 0:
            return ""
        max_d = float(np.max(density)) if np.max(density) > 0 else 1.0
        half = width / 2
        right_points: List[str] = []
        left_points: List[str] = []
        for yv, dv in zip(y_grid, density):
            dx = (dv / max_d) * half
            right_points.append(f"{x + dx:.2f},{yv:.2f}")
            left_points.append(f"{x - dx:.2f},{yv:.2f}")
        left_points.reverse()
        all_pts = " ".join(right_points + left_points)
        return (
            f'<polygon points="{all_pts}" fill="{fill}" '
            f'fill-opacity="0.5" stroke="{fill}" stroke-width="1"/>'
        )

    def _kernel_density_estimate(
        self,
        values: List[float],
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values, dtype=float)
        if len(arr) < 2:
            return np.array([float(np.mean(arr))]), np.array([1.0])
        std = float(np.std(arr, ddof=1))
        if std < 1e-12:
            std = 1.0
        bandwidth = 1.06 * std * len(arr) ** (-0.2)
        lo = float(np.min(arr)) - 3 * bandwidth
        hi = float(np.max(arr)) + 3 * bandwidth
        grid = np.linspace(lo, hi, n_points)
        density = np.zeros(n_points)
        for v in arr:
            density += np.exp(-0.5 * ((grid - v) / bandwidth) ** 2)
        density /= len(arr) * bandwidth * math.sqrt(2 * math.pi)
        return grid, density

    def _svg_radar_polygon(
        self,
        center: Tuple[float, float],
        radii: List[float],
        angles: List[float],
        fill: str,
        stroke: str,
        fill_opacity: float = 0.25,
    ) -> str:
        pts: List[str] = []
        for r, a in zip(radii, angles):
            px = center[0] + r * math.cos(a)
            py = center[1] + r * math.sin(a)
            pts.append(f"{px:.2f},{py:.2f}")
        all_pts = " ".join(pts)
        return (
            f'<polygon points="{all_pts}" fill="{fill}" '
            f'fill-opacity="{fill_opacity}" stroke="{stroke}" '
            f'stroke-width="2"/>'
        )

    # ------------------------------------------------------------------
    # ranking / significance utilities
    # ------------------------------------------------------------------

    def _rank_algorithms(
        self,
        scores: Dict[str, float],
    ) -> List[Tuple[str, int]]:
        sorted_items = sorted(
            scores.items(),
            key=lambda kv: kv[1],
            reverse=not self.config.ascending,
        )
        ranked: List[Tuple[str, int]] = []
        for rank, (name, _) in enumerate(sorted_items, start=1):
            ranked.append((name, rank))
        return ranked

    def _significance_markers(
        self,
        p_values: List[float],
    ) -> List[str]:
        markers: List[str] = []
        for p in p_values:
            if p < 0.001:
                markers.append("***")
            elif p < 0.01:
                markers.append("**")
            elif p < 0.05:
                markers.append("*")
            else:
                markers.append("n.s.")
        return markers

    # ------------------------------------------------------------------
    # axis / grid helpers
    # ------------------------------------------------------------------

    def _nice_ticks(
        self, vmin: float, vmax: float, n_ticks: int = 5,
    ) -> List[float]:
        if abs(vmax - vmin) < 1e-12:
            return [vmin]
        raw_step = (vmax - vmin) / max(n_ticks - 1, 1)
        mag = 10 ** math.floor(math.log10(max(abs(raw_step), 1e-15)))
        residual = raw_step / mag
        if residual <= 1.5:
            nice = 1.0
        elif residual <= 3.0:
            nice = 2.0
        elif residual <= 7.0:
            nice = 5.0
        else:
            nice = 10.0
        step = nice * mag
        lo = math.floor(vmin / step) * step
        hi = math.ceil(vmax / step) * step
        ticks: List[float] = []
        t = lo
        while t <= hi + step * 0.001:
            ticks.append(round(t, 10))
            t += step
        return ticks

    def _draw_axes(
        self,
        width: float,
        height: float,
        x_labels: Optional[List[str]] = None,
        y_ticks: Optional[List[float]] = None,
        y_min: float = 0.0,
        y_max: float = 1.0,
        x_title: str = "",
        y_title: str = "",
        rotate_x: bool = False,
    ) -> str:
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()
        parts: List[str] = []
        # y-axis line
        parts.append(
            f'<line x1="{m["left"]:.1f}" y1="{m["top"]:.1f}" '
            f'x2="{m["left"]:.1f}" y2="{m["top"] + ph:.1f}" '
            f'stroke="{self.config.axis_color}" stroke-width="1.2"/>'
        )
        # x-axis line
        parts.append(
            f'<line x1="{m["left"]:.1f}" y1="{m["top"] + ph:.1f}" '
            f'x2="{m["left"] + pw:.1f}" y2="{m["top"] + ph:.1f}" '
            f'stroke="{self.config.axis_color}" stroke-width="1.2"/>'
        )
        # y ticks & gridlines
        if y_ticks:
            rng = y_max - y_min if abs(y_max - y_min) > 1e-12 else 1.0
            for tv in y_ticks:
                frac = (tv - y_min) / rng
                yy = m["top"] + ph * (1 - frac)
                parts.append(
                    f'<line x1="{m["left"] - 5:.1f}" y1="{yy:.1f}" '
                    f'x2="{m["left"]:.1f}" y2="{yy:.1f}" '
                    f'stroke="{self.config.axis_color}" stroke-width="1"/>'
                )
                parts.append(
                    f'<line x1="{m["left"]:.1f}" y1="{yy:.1f}" '
                    f'x2="{m["left"] + pw:.1f}" y2="{yy:.1f}" '
                    f'stroke="{self.config.grid_color}" stroke-width="0.8" '
                    f'stroke-dasharray="4,3"/>'
                )
                label = f"{tv:.4g}"
                parts.append(
                    f'<text x="{m["left"] - 8:.1f}" y="{yy + 4:.1f}" '
                    f'text-anchor="end" font-size="{self.config.font_size}" '
                    f'font-family="{self.config.font_family}" '
                    f'fill="{self.config.axis_color}">{label}</text>'
                )
        # x labels
        if x_labels:
            n = len(x_labels)
            slot = pw / n
            for i, lbl in enumerate(x_labels):
                cx = m["left"] + slot * (i + 0.5)
                cy = m["top"] + ph + 18
                if rotate_x:
                    parts.append(
                        f'<text x="{cx:.1f}" y="{cy:.1f}" '
                        f'text-anchor="end" '
                        f'transform="rotate(-35,{cx:.1f},{cy:.1f})" '
                        f'font-size="{self.config.font_size}" '
                        f'font-family="{self.config.font_family}" '
                        f'fill="{self.config.axis_color}">'
                        f'{_escape(lbl)}</text>'
                    )
                else:
                    parts.append(
                        f'<text x="{cx:.1f}" y="{cy:.1f}" '
                        f'text-anchor="middle" '
                        f'font-size="{self.config.font_size}" '
                        f'font-family="{self.config.font_family}" '
                        f'fill="{self.config.axis_color}">'
                        f'{_escape(lbl)}</text>'
                    )
        # axis titles
        if y_title:
            tx = 18
            ty = m["top"] + ph / 2
            parts.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                f'transform="rotate(-90,{tx:.1f},{ty:.1f})" '
                f'font-size="{self.config.font_size + 1}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(y_title)}</text>'
            )
        if x_title:
            tx = m["left"] + pw / 2
            ty = height - 10
            parts.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                f'font-size="{self.config.font_size + 1}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(x_title)}</text>'
            )
        return "\n".join(parts)

    def _draw_title(self, width: float, title: str) -> str:
        if not title:
            return ""
        tx = width / 2
        ty = self.config.margin["top"] / 2 + 4
        return (
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
            f'font-size="{self.config.title_size}" font-weight="bold" '
            f'font-family="{self.config.font_family}" '
            f'fill="{self.config.axis_color}">{_escape(title)}</text>'
        )

    def _draw_legend(
        self,
        items: List[Tuple[str, str]],
        x: float,
        y: float,
    ) -> str:
        parts: List[str] = []
        dy = 0.0
        for label, color in items:
            parts.append(
                f'<rect x="{x:.1f}" y="{y + dy:.1f}" width="14" '
                f'height="14" fill="{color}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + 20:.1f}" y="{y + dy + 11:.1f}" '
                f'font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(label)}</text>'
            )
            dy += 20
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # value → pixel mapping helpers
    # ------------------------------------------------------------------

    def _val_to_y(
        self, val: float, y_min: float, y_max: float,
    ) -> float:
        m = self.config.margin
        ph = self.config.plot_height()
        rng = y_max - y_min if abs(y_max - y_min) > 1e-12 else 1.0
        frac = (val - y_min) / rng
        return m["top"] + ph * (1.0 - frac)

    def _val_to_x(
        self, val: float, x_min: float, x_max: float,
    ) -> float:
        m = self.config.margin
        pw = self.config.plot_width()
        rng = x_max - x_min if abs(x_max - x_min) > 1e-12 else 1.0
        frac = (val - x_min) / rng
        return m["left"] + pw * frac

    # ==================================================================
    # PUBLIC CHART METHODS
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. Simple bar comparison
    # ------------------------------------------------------------------

    def plot_bar_comparison(
        self,
        algo_scores: Dict[str, float],
        metric_name: str = "Score",
    ) -> str:
        if not algo_scores:
            return "<svg></svg>"

        names, vals = zip(*sorted(
            algo_scores.items(),
            key=lambda kv: kv[1],
            reverse=not self.config.ascending,
        ))
        n = len(names)
        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        v_min = 0.0
        v_max = max(vals) * 1.15 if max(vals) > 0 else 1.0
        ticks = self._nice_ticks(v_min, v_max)
        v_max = max(ticks) if ticks else v_max

        slot = pw / n
        bw = slot * self.config.bar_width

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or f"{metric_name} Comparison"))
        parts.append(self._draw_axes(
            w, h, list(names), ticks, v_min, v_max,
            y_title=metric_name, rotate_x=n > 6,
        ))

        ref = self.config.reference_algorithm
        for i, (name, val) in enumerate(zip(names, vals)):
            cx = m["left"] + slot * (i + 0.5)
            bx = cx - bw / 2
            bar_h = (val - v_min) / (v_max - v_min) * ph if v_max != v_min else 0
            by = m["top"] + ph - bar_h
            color = self._color(i)
            if ref and name == ref:
                color = "#FFD700"
            fmt = f"{val:{self.config.annotation_format}}"
            parts.append(self._svg_bar(bx, by, bw, bar_h, color, fmt))

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 2. Grouped bar chart
    # ------------------------------------------------------------------

    def plot_grouped_bars(
        self,
        algo_metric_scores: Dict[str, Dict[str, float]],
    ) -> str:
        if not algo_metric_scores:
            return "<svg></svg>"

        algo_names = list(algo_metric_scores.keys())
        metric_names = list(
            dict.fromkeys(
                m for d in algo_metric_scores.values() for m in d
            )
        )
        n_algos = len(algo_names)
        n_metrics = len(metric_names)
        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        all_vals = [
            algo_metric_scores[a].get(met, 0)
            for a in algo_names for met in metric_names
        ]
        v_min = 0.0
        v_max = max(all_vals) * 1.15 if all_vals and max(all_vals) > 0 else 1.0
        ticks = self._nice_ticks(v_min, v_max)
        v_max = max(ticks) if ticks else v_max

        group_w = pw / n_metrics
        bar_w = (group_w * 0.8) / max(n_algos, 1)

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Grouped Metric Comparison"))
        parts.append(self._draw_axes(
            w, h, metric_names, ticks, v_min, v_max,
            rotate_x=n_metrics > 5,
        ))

        for mi, met in enumerate(metric_names):
            gx = m["left"] + group_w * mi
            offset = group_w * 0.1
            for ai, algo in enumerate(algo_names):
                val = algo_metric_scores[algo].get(met, 0)
                bx = gx + offset + bar_w * ai
                bar_h = (val - v_min) / (v_max - v_min) * ph
                by = m["top"] + ph - bar_h
                color = self._color(ai)
                fmt = f"{val:{self.config.annotation_format}}"
                parts.append(self._svg_bar(bx, by, bar_w * 0.9, bar_h, color, fmt))

        # legend
        legend_items = [(a, self._color(i)) for i, a in enumerate(algo_names)]
        parts.append(self._draw_legend(legend_items, w - m["right"] - 120, m["top"] + 10))

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 3. Box plot comparison
    # ------------------------------------------------------------------

    def plot_box_comparison(
        self,
        algo_score_lists: Dict[str, List[float]],
        metric_name: str = "Score",
    ) -> str:
        if not algo_score_lists:
            return "<svg></svg>"

        names = list(algo_score_lists.keys())
        n = len(names)
        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        all_vals = [v for vs in algo_score_lists.values() for v in vs]
        if not all_vals:
            return "<svg></svg>"
        raw_min = min(all_vals)
        raw_max = max(all_vals)
        pad = (raw_max - raw_min) * 0.1 if raw_max != raw_min else 1.0
        v_min = raw_min - pad
        v_max = raw_max + pad
        ticks = self._nice_ticks(v_min, v_max)
        v_min = min(ticks) if ticks else v_min
        v_max = max(ticks) if ticks else v_max

        slot = pw / n
        box_w = slot * 0.5

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or f"{metric_name} Distribution"))
        parts.append(self._draw_axes(
            w, h, names, ticks, v_min, v_max,
            y_title=metric_name, rotate_x=n > 6,
        ))

        for i, name in enumerate(names):
            stats = self._compute_box_stats(algo_score_lists[name])
            cx = m["left"] + slot * (i + 0.5)
            med_y = self._val_to_y(stats["median"], v_min, v_max)
            q1_y = self._val_to_y(stats["q1"], v_min, v_max)
            q3_y = self._val_to_y(stats["q3"], v_min, v_max)
            wl_y = self._val_to_y(stats["whisker_low"], v_min, v_max)
            wh_y = self._val_to_y(stats["whisker_high"], v_min, v_max)
            outlier_ys = [
                self._val_to_y(o, v_min, v_max) for o in stats["outliers"]
            ]
            parts.append(self._svg_box_plot(
                cx, med_y, q1_y, q3_y, wl_y, wh_y, box_w,
                fill=self._color(i), outliers=outlier_ys,
            ))

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 4. Violin comparison
    # ------------------------------------------------------------------

    def plot_violin_comparison(
        self,
        algo_score_lists: Dict[str, List[float]],
        metric_name: str = "Score",
    ) -> str:
        if not algo_score_lists:
            return "<svg></svg>"

        names = list(algo_score_lists.keys())
        n = len(names)
        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        all_vals = [v for vs in algo_score_lists.values() for v in vs]
        if not all_vals:
            return "<svg></svg>"
        raw_min = min(all_vals)
        raw_max = max(all_vals)
        pad = (raw_max - raw_min) * 0.1 if raw_max != raw_min else 1.0
        v_min = raw_min - pad
        v_max = raw_max + pad
        ticks = self._nice_ticks(v_min, v_max)
        v_min = min(ticks) if ticks else v_min
        v_max = max(ticks) if ticks else v_max

        slot = pw / n
        violin_w = slot * 0.7

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or f"{metric_name} Distribution (Violin)"))
        parts.append(self._draw_axes(
            w, h, names, ticks, v_min, v_max,
            y_title=metric_name, rotate_x=n > 6,
        ))

        rng = v_max - v_min if abs(v_max - v_min) > 1e-12 else 1.0

        for i, name in enumerate(names):
            values = algo_score_lists[name]
            if not values:
                continue
            cx = m["left"] + slot * (i + 0.5)
            grid_val, density = self._kernel_density_estimate(values)
            # map grid values to y-pixel coords
            grid_y = np.array([
                self._val_to_y(gv, v_min, v_max) for gv in grid_val
            ])
            parts.append(self._svg_violin(
                cx, 0, (grid_y, density), violin_w, self._color(i),
            ))
            # add median marker
            med = float(np.median(values))
            med_y = self._val_to_y(med, v_min, v_max)
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{med_y:.1f}" r="4" '
                f'fill="white" stroke="{self.config.axis_color}" '
                f'stroke-width="1.5"/>'
            )
            # inner quartile box
            stats = self._compute_box_stats(values)
            q1_y = self._val_to_y(stats["q1"], v_min, v_max)
            q3_y = self._val_to_y(stats["q3"], v_min, v_max)
            top_y = min(q1_y, q3_y)
            box_h = abs(q3_y - q1_y)
            tiny_w = violin_w * 0.15
            parts.append(
                f'<rect x="{cx - tiny_w / 2:.1f}" y="{top_y:.1f}" '
                f'width="{tiny_w:.1f}" height="{box_h:.1f}" '
                f'fill="{self.config.axis_color}" opacity="0.4" rx="1"/>'
            )

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 5. Radar (spider) chart
    # ------------------------------------------------------------------

    def plot_radar_chart(
        self,
        algo_scores: Dict[str, Dict[str, float]],
        metric_names: Optional[List[str]] = None,
    ) -> str:
        if not algo_scores:
            return "<svg></svg>"

        algo_names = list(algo_scores.keys())
        if metric_names is None:
            metric_names = list(
                dict.fromkeys(m for d in algo_scores.values() for m in d)
            )
        n_axes = len(metric_names)
        if n_axes < 3:
            return "<svg></svg>"

        w, h = self.config.figsize
        cx, cy = w / 2, h / 2 + 10
        radius = min(w, h) * 0.35
        angles = [
            -math.pi / 2 + 2 * math.pi * i / n_axes for i in range(n_axes)
        ]

        # normalise each metric to [0, 1]
        metric_max: Dict[str, float] = {}
        for met in metric_names:
            vals = [algo_scores[a].get(met, 0) for a in algo_names]
            metric_max[met] = max(vals) if max(vals) > 0 else 1.0

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Radar Comparison"))

        # grid rings
        for frac in (0.25, 0.5, 0.75, 1.0):
            ring_r = radius * frac
            pts = " ".join(
                f"{cx + ring_r * math.cos(a):.1f},"
                f"{cy + ring_r * math.sin(a):.1f}"
                for a in angles
            )
            parts.append(
                f'<polygon points="{pts}" fill="none" '
                f'stroke="{self.config.grid_color}" stroke-width="0.8"/>'
            )
            # tick label
            parts.append(
                f'<text x="{cx + 4:.1f}" y="{cy - ring_r + 4:.1f}" '
                f'font-size="{self.config.font_size - 2}" '
                f'fill="{self.config.axis_color}" opacity="0.6">'
                f'{frac:.0%}</text>'
            )

        # axis spokes & labels
        for i, (a, met) in enumerate(zip(angles, metric_names)):
            ex = cx + radius * math.cos(a) * 1.12
            ey = cy + radius * math.sin(a) * 1.12
            parts.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" '
                f'x2="{cx + radius * math.cos(a):.1f}" '
                f'y2="{cy + radius * math.sin(a):.1f}" '
                f'stroke="{self.config.grid_color}" stroke-width="1"/>'
            )
            anchor = "middle"
            if math.cos(a) > 0.3:
                anchor = "start"
            elif math.cos(a) < -0.3:
                anchor = "end"
            parts.append(
                f'<text x="{ex:.1f}" y="{ey:.1f}" text-anchor="{anchor}" '
                f'font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(met)}</text>'
            )

        # algorithm polygons
        for ai, algo in enumerate(algo_names):
            radii = []
            for met in metric_names:
                val = algo_scores[algo].get(met, 0)
                norm = val / metric_max[met] if metric_max[met] > 0 else 0
                radii.append(radius * min(norm, 1.0))
            color = self._color(ai)
            parts.append(self._svg_radar_polygon(
                (cx, cy), radii, angles, color, color, fill_opacity=0.2,
            ))
            # dots
            for r, a in zip(radii, angles):
                px = cx + r * math.cos(a)
                py = cy + r * math.sin(a)
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
                    f'fill="{color}" stroke="white" stroke-width="1.5"/>'
                )

        # legend
        legend_items = [(a, self._color(i)) for i, a in enumerate(algo_names)]
        parts.append(self._draw_legend(legend_items, 20, h - 30 * len(algo_names)))

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 6. Ranking table (HTML table with embedded styles)
    # ------------------------------------------------------------------

    def plot_ranking_table(
        self,
        rankings: Dict[str, Dict[str, int]],
    ) -> str:
        if not rankings:
            return "<table></table>"

        algo_names = list(rankings.keys())
        metrics = list(
            dict.fromkeys(m for d in rankings.values() for m in d)
        )

        avg_ranks: Dict[str, float] = {}
        for algo in algo_names:
            vals = [rankings[algo].get(m, len(algo_names)) for m in metrics]
            avg_ranks[algo] = float(np.mean(vals))

        sorted_algos = sorted(algo_names, key=lambda a: avg_ranks[a])

        headers = ["Rank", "Algorithm"] + metrics + ["Avg Rank"]
        rows: List[List[str]] = []
        for rank, algo in enumerate(sorted_algos, 1):
            row = [str(rank), algo]
            for met in metrics:
                row.append(str(rankings[algo].get(met, "-")))
            row.append(f"{avg_ranks[algo]:.2f}")
            rows.append(row)

        styles = {
            "table": (
                "border-collapse:collapse; font-family:{ff}; "
                "font-size:{fs}px; width:100%;"
            ).format(ff=self.config.font_family, fs=self.config.font_size),
            "th": (
                "background:#4C72B0; color:white; padding:8px 12px; "
                "text-align:center; border:1px solid #ddd;"
            ),
            "td": "padding:6px 12px; text-align:center; border:1px solid #ddd;",
            "tr_even": "background:#f9f9f9;",
            "tr_odd": "background:#ffffff;",
            "rank1": "background:#FFD700; font-weight:bold;",
            "rank2": "background:#C0C0C0;",
            "rank3": "background:#CD7F32; color:white;",
        }
        return self._format_table_html(headers, rows, styles)

    # ------------------------------------------------------------------
    # 7. Win/loss heatmap matrix
    # ------------------------------------------------------------------

    def plot_win_loss_matrix(
        self,
        pairwise_results: Dict[str, Dict[str, float]],
    ) -> str:
        if not pairwise_results:
            return "<svg></svg>"

        names = list(pairwise_results.keys())
        n = len(names)
        cell = 60
        label_w = 120
        w = label_w + n * cell + 40
        h = 60 + n * cell + label_w
        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Win/Loss Matrix"))

        ox = label_w
        oy = 50

        for i, row_name in enumerate(names):
            # row label
            parts.append(
                f'<text x="{ox - 6:.1f}" y="{oy + i * cell + cell / 2 + 4:.1f}" '
                f'text-anchor="end" font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(row_name)}</text>'
            )
            for j, col_name in enumerate(names):
                val = pairwise_results.get(row_name, {}).get(col_name, 0.0)
                # colour: green for wins (>0.5), red for losses (<0.5)
                if i == j:
                    fill = "#E8E8E8"
                elif val >= 0.5:
                    intensity = min((val - 0.5) * 2, 1.0)
                    g = int(120 + 135 * intensity)
                    fill = f"rgb(80,{g},80)"
                else:
                    intensity = min((0.5 - val) * 2, 1.0)
                    r = int(120 + 135 * intensity)
                    fill = f"rgb({r},80,80)"

                rx = ox + j * cell
                ry = oy + i * cell
                parts.append(
                    f'<rect x="{rx}" y="{ry}" width="{cell}" '
                    f'height="{cell}" fill="{fill}" '
                    f'stroke="white" stroke-width="1.5"/>'
                )
                txt_color = "white" if (val > 0.75 or val < 0.25) and i != j else self.config.axis_color
                parts.append(
                    f'<text x="{rx + cell / 2:.1f}" '
                    f'y="{ry + cell / 2 + 4:.1f}" text-anchor="middle" '
                    f'font-size="{self.config.font_size - 1}" '
                    f'font-family="{self.config.font_family}" '
                    f'fill="{txt_color}">{val:.2f}</text>'
                )

        # column headers (rotated)
        for j, col_name in enumerate(names):
            cx = ox + j * cell + cell / 2
            cy = oy - 6
            parts.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="end" '
                f'transform="rotate(-45,{cx:.1f},{cy:.1f})" '
                f'font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(col_name)}</text>'
            )

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 8. Critical difference diagram
    # ------------------------------------------------------------------

    def plot_critical_difference_diagram(
        self,
        avg_ranks: Dict[str, float],
        n_datasets: int,
        p_values: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> str:
        if not avg_ranks:
            return "<svg></svg>"

        names_sorted = sorted(avg_ranks.keys(), key=lambda k: avg_ranks[k])
        n = len(names_sorted)
        w = max(self.config.figsize[0], 700)
        h = max(200, 50 + n * 30)
        m_left, m_right = 160, 160
        line_w = w - m_left - m_right

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Critical Difference Diagram"))

        r_min = min(avg_ranks.values()) - 0.3
        r_max = max(avg_ranks.values()) + 0.3
        rng = r_max - r_min if abs(r_max - r_min) > 1e-9 else 1.0

        axis_y = 50
        parts.append(
            f'<line x1="{m_left}" y1="{axis_y}" '
            f'x2="{m_left + line_w}" y2="{axis_y}" '
            f'stroke="{self.config.axis_color}" stroke-width="1.5"/>'
        )

        # rank ticks on the axis
        for r in range(1, n + 1):
            rx = m_left + (r - r_min) / rng * line_w
            parts.append(
                f'<line x1="{rx:.1f}" y1="{axis_y - 5}" '
                f'x2="{rx:.1f}" y2="{axis_y + 5}" '
                f'stroke="{self.config.axis_color}" stroke-width="1.2"/>'
            )
            parts.append(
                f'<text x="{rx:.1f}" y="{axis_y - 10}" text-anchor="middle" '
                f'font-size="{self.config.font_size}" '
                f'fill="{self.config.axis_color}">{r}</text>'
            )

        # algorithm markers
        left_algos = names_sorted[: (n + 1) // 2]
        right_algos = names_sorted[(n + 1) // 2:]

        for i, name in enumerate(left_algos):
            rx = m_left + (avg_ranks[name] - r_min) / rng * line_w
            label_y = axis_y + 30 + i * 25
            parts.append(
                f'<line x1="{rx:.1f}" y1="{axis_y}" '
                f'x2="{rx:.1f}" y2="{label_y:.1f}" '
                f'stroke="{self.config.axis_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{rx:.1f}" y1="{label_y:.1f}" '
                f'x2="{m_left - 10:.1f}" y2="{label_y:.1f}" '
                f'stroke="{self.config.axis_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{m_left - 14:.1f}" y="{label_y + 4:.1f}" '
                f'text-anchor="end" font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">'
                f'{_escape(name)} ({avg_ranks[name]:.2f})</text>'
            )

        for i, name in enumerate(right_algos):
            rx = m_left + (avg_ranks[name] - r_min) / rng * line_w
            label_y = axis_y + 30 + i * 25
            parts.append(
                f'<line x1="{rx:.1f}" y1="{axis_y}" '
                f'x2="{rx:.1f}" y2="{label_y:.1f}" '
                f'stroke="{self.config.axis_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{rx:.1f}" y1="{label_y:.1f}" '
                f'x2="{m_left + line_w + 10:.1f}" y2="{label_y:.1f}" '
                f'stroke="{self.config.axis_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{m_left + line_w + 14:.1f}" y="{label_y + 4:.1f}" '
                f'text-anchor="start" font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">'
                f'{_escape(name)} ({avg_ranks[name]:.2f})</text>'
            )

        # draw cliques (groups of statistically equivalent algorithms)
        if p_values:
            clique_y = axis_y - 20
            for (a, b), pv in p_values.items():
                if pv > 0.05:  # not significantly different
                    x1 = m_left + (avg_ranks.get(a, 1) - r_min) / rng * line_w
                    x2 = m_left + (avg_ranks.get(b, 1) - r_min) / rng * line_w
                    parts.append(
                        f'<line x1="{min(x1, x2):.1f}" y1="{clique_y}" '
                        f'x2="{max(x1, x2):.1f}" y2="{clique_y}" '
                        f'stroke="{self.config.axis_color}" '
                        f'stroke-width="3"/>'
                    )
                    clique_y -= 8

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 9. Convergence curves
    # ------------------------------------------------------------------

    def plot_convergence_curves(
        self,
        algo_curves: Dict[str, List[float]],
    ) -> str:
        if not algo_curves:
            return "<svg></svg>"

        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        max_len = max(len(c) for c in algo_curves.values())
        all_vals = [v for c in algo_curves.values() for v in c]
        v_min = min(all_vals) if all_vals else 0
        v_max = max(all_vals) if all_vals else 1
        pad = (v_max - v_min) * 0.05 if v_max != v_min else 0.5
        v_min -= pad
        v_max += pad
        ticks = self._nice_ticks(v_min, v_max)
        v_min = min(ticks) if ticks else v_min
        v_max = max(ticks) if ticks else v_max

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Convergence Curves"))
        x_labels = [str(i) for i in range(0, max_len, max(1, max_len // 8))]
        parts.append(self._draw_axes(
            w, h, None, ticks, v_min, v_max,
            x_title="Iteration", y_title="Score",
        ))
        # x-axis ticks
        x_step = max(1, max_len // 8)
        for xi in range(0, max_len, x_step):
            px = self._val_to_x(xi, 0, max_len - 1)
            py = m["top"] + ph
            parts.append(
                f'<text x="{px:.1f}" y="{py + 16:.1f}" text-anchor="middle" '
                f'font-size="{self.config.font_size}" '
                f'fill="{self.config.axis_color}">{xi}</text>'
            )

        for ai, (algo, curve) in enumerate(algo_curves.items()):
            color = self._color(ai)
            pts: List[str] = []
            for j, val in enumerate(curve):
                px = self._val_to_x(j, 0, max(max_len - 1, 1))
                py = self._val_to_y(val, v_min, v_max)
                pts.append(f"{px:.2f},{py:.2f}")
            polyline = " ".join(pts)
            parts.append(
                f'<polyline points="{polyline}" fill="none" '
                f'stroke="{color}" stroke-width="2" '
                f'stroke-linejoin="round"/>'
            )
            # end-point label
            if curve:
                last_x = self._val_to_x(len(curve) - 1, 0, max(max_len - 1, 1))
                last_y = self._val_to_y(curve[-1], v_min, v_max)
                parts.append(
                    f'<text x="{last_x + 6:.1f}" y="{last_y + 4:.1f}" '
                    f'font-size="{self.config.font_size - 1}" '
                    f'font-family="{self.config.font_family}" '
                    f'fill="{color}">{_escape(algo)}</text>'
                )

        legend_items = [(a, self._color(i)) for i, a in enumerate(algo_curves)]
        parts.append(self._draw_legend(legend_items, m["left"] + 10, m["top"] + 10))
        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 10. Runtime comparison (horizontal bars)
    # ------------------------------------------------------------------

    def plot_runtime_comparison(
        self,
        algo_times: Dict[str, float],
    ) -> str:
        if not algo_times:
            return "<svg></svg>"

        sorted_items = sorted(
            algo_times.items(), key=lambda kv: kv[1], reverse=True,
        )
        names = [s[0] for s in sorted_items]
        vals = [s[1] for s in sorted_items]
        n = len(names)

        # horizontal bar chart: swap width/height roles
        w, h = self.config.figsize
        h = max(h, 60 + n * 35)
        m = self.config.margin
        m_left = max(m["left"], 140)
        pw = w - m_left - m["right"]
        ph = h - m["top"] - m["bottom"]

        v_max = max(vals) * 1.15 if vals else 1.0
        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Runtime Comparison"))

        slot = ph / n
        bar_h = slot * 0.6

        for i, (name, val) in enumerate(zip(names, vals)):
            cy = m["top"] + slot * (i + 0.5)
            by = cy - bar_h / 2
            bar_w = (val / v_max) * pw if v_max > 0 else 0
            color = self._color(i)
            parts.append(
                f'<rect x="{m_left:.1f}" y="{by:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{color}" rx="3"/>'
            )
            parts.append(
                f'<text x="{m_left - 6:.1f}" y="{cy + 4:.1f}" '
                f'text-anchor="end" font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(name)}</text>'
            )
            # value label
            label_x = m_left + bar_w + 6
            parts.append(
                f'<text x="{label_x:.1f}" y="{cy + 4:.1f}" '
                f'text-anchor="start" font-size="{self.config.font_size - 1}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{val:.2f}s</text>'
            )

        # x axis
        parts.append(
            f'<line x1="{m_left}" y1="{m["top"] + ph:.1f}" '
            f'x2="{m_left + pw:.1f}" y2="{m["top"] + ph:.1f}" '
            f'stroke="{self.config.axis_color}" stroke-width="1"/>'
        )
        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 11. Diversity–quality scatter
    # ------------------------------------------------------------------

    def plot_diversity_quality_scatter(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """results maps algo_name -> {'diversity': float, 'quality': float}."""
        if not results:
            return "<svg></svg>"

        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        divs = [r.get("diversity", 0) for r in results.values()]
        quals = [r.get("quality", 0) for r in results.values()]
        pad_x = (max(divs) - min(divs)) * 0.1 if max(divs) != min(divs) else 0.5
        pad_y = (max(quals) - min(quals)) * 0.1 if max(quals) != min(quals) else 0.5
        x_min = min(divs) - pad_x
        x_max = max(divs) + pad_x
        y_min = min(quals) - pad_y
        y_max = max(quals) + pad_y

        x_ticks = self._nice_ticks(x_min, x_max)
        y_ticks = self._nice_ticks(y_min, y_max)
        x_min = min(x_ticks)
        x_max = max(x_ticks)
        y_min = min(y_ticks)
        y_max = max(y_ticks)

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Diversity vs Quality"))
        parts.append(self._draw_axes(
            w, h, None, y_ticks, y_min, y_max,
            x_title="Diversity", y_title="Quality",
        ))

        # x-axis ticks
        for xt in x_ticks:
            px = self._val_to_x(xt, x_min, x_max)
            py = m["top"] + ph
            parts.append(
                f'<line x1="{px:.1f}" y1="{py}" x2="{px:.1f}" '
                f'y2="{py + 5}" stroke="{self.config.axis_color}" '
                f'stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{py + 18}" text-anchor="middle" '
                f'font-size="{self.config.font_size}" '
                f'fill="{self.config.axis_color}">{xt:.3g}</text>'
            )

        # Pareto front
        items = sorted(results.items(), key=lambda kv: kv[1].get("diversity", 0))
        pareto: List[Tuple[float, float]] = []
        best_q = -float("inf")
        for name, r in reversed(items):
            q = r.get("quality", 0)
            if q >= best_q:
                best_q = q
                pareto.append((r.get("diversity", 0), q))
        pareto.sort(key=lambda t: t[0])
        if len(pareto) > 1:
            ppts = " ".join(
                f"{self._val_to_x(d, x_min, x_max):.1f},"
                f"{self._val_to_y(q, y_min, y_max):.1f}"
                for d, q in pareto
            )
            parts.append(
                f'<polyline points="{ppts}" fill="none" '
                f'stroke="#999" stroke-width="1.5" '
                f'stroke-dasharray="6,3"/>'
            )

        # scatter points
        for i, (name, r) in enumerate(results.items()):
            px = self._val_to_x(r.get("diversity", 0), x_min, x_max)
            py = self._val_to_y(r.get("quality", 0), y_min, y_max)
            color = self._color(i)
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" '
                f'fill="{color}" stroke="white" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{px + 10:.1f}" y="{py + 4:.1f}" '
                f'font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{color}">{_escape(name)}</text>'
            )

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 12. Effect-size forest plot
    # ------------------------------------------------------------------

    def plot_effect_size_forest(
        self,
        effect_sizes: Dict[str, Dict[str, float]],
    ) -> str:
        """effect_sizes maps comparison_label -> {'d': float, 'ci_low': float, 'ci_high': float}."""
        if not effect_sizes:
            return "<svg></svg>"

        labels = list(effect_sizes.keys())
        n = len(labels)
        w = max(self.config.figsize[0], 700)
        h = max(200, 80 + n * 32)
        m_left = 220
        m_right = 60
        pw = w - m_left - m_right
        m_top = 60

        all_low = [e.get("ci_low", e["d"] - 0.5) for e in effect_sizes.values()]
        all_high = [e.get("ci_high", e["d"] + 0.5) for e in effect_sizes.values()]
        x_min = min(all_low) - 0.2
        x_max = max(all_high) + 0.2
        x_ticks = self._nice_ticks(x_min, x_max)
        x_min = min(x_ticks)
        x_max = max(x_ticks)
        x_rng = x_max - x_min if abs(x_max - x_min) > 1e-12 else 1.0

        def to_px(v: float) -> float:
            return m_left + (v - x_min) / x_rng * pw

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Effect Size Forest Plot"))

        # zero line
        zero_x = to_px(0)
        parts.append(
            f'<line x1="{zero_x:.1f}" y1="{m_top}" '
            f'x2="{zero_x:.1f}" y2="{m_top + n * 32:.1f}" '
            f'stroke="#CC0000" stroke-width="1.5" stroke-dasharray="6,3"/>'
        )

        # axis
        axis_y = m_top + n * 32 + 5
        parts.append(
            f'<line x1="{m_left}" y1="{axis_y}" '
            f'x2="{m_left + pw}" y2="{axis_y}" '
            f'stroke="{self.config.axis_color}" stroke-width="1"/>'
        )
        for xt in x_ticks:
            px = to_px(xt)
            parts.append(
                f'<line x1="{px:.1f}" y1="{axis_y}" x2="{px:.1f}" '
                f'y2="{axis_y + 5}" stroke="{self.config.axis_color}" '
                f'stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{axis_y + 18}" text-anchor="middle" '
                f'font-size="{self.config.font_size}" '
                f'fill="{self.config.axis_color}">{xt:.2g}</text>'
            )

        # effect size rows
        for i, (label, es) in enumerate(effect_sizes.items()):
            d = es["d"]
            ci_lo = es.get("ci_low", d - 0.3)
            ci_hi = es.get("ci_high", d + 0.3)
            cy = m_top + i * 32 + 16

            dx = to_px(d)
            lx = to_px(ci_lo)
            hx = to_px(ci_hi)

            # CI line
            parts.append(
                f'<line x1="{lx:.1f}" y1="{cy}" x2="{hx:.1f}" y2="{cy}" '
                f'stroke="{self._color(i)}" stroke-width="2"/>'
            )
            # point estimate diamond
            sz = 5
            diamond = (
                f"{dx},{cy - sz} {dx + sz},{cy} "
                f"{dx},{cy + sz} {dx - sz},{cy}"
            )
            parts.append(
                f'<polygon points="{diamond}" fill="{self._color(i)}" '
                f'stroke="white" stroke-width="1"/>'
            )
            # label
            parts.append(
                f'<text x="{m_left - 8:.1f}" y="{cy + 4:.1f}" '
                f'text-anchor="end" font-size="{self.config.font_size}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{self.config.axis_color}">{_escape(label)}</text>'
            )
            # numeric
            parts.append(
                f'<text x="{hx + 8:.1f}" y="{cy + 4:.1f}" '
                f'text-anchor="start" font-size="{self.config.font_size - 1}" '
                f'fill="{self.config.axis_color}">'
                f'{d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]</text>'
            )

        parts.append(
            f'<text x="{m_left + pw / 2:.1f}" y="{axis_y + 34}" '
            f'text-anchor="middle" font-size="{self.config.font_size + 1}" '
            f'fill="{self.config.axis_color}">Effect Size (Cohen\'s d)</text>'
        )

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 13. Bayesian comparison (posterior density)
    # ------------------------------------------------------------------

    def plot_bayesian_comparison(
        self,
        posterior_diffs: Dict[str, List[float]],
    ) -> str:
        """posterior_diffs maps comparison_label -> samples of posterior difference."""
        if not posterior_diffs:
            return "<svg></svg>"

        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        # global x range from all samples
        all_samples = [v for ss in posterior_diffs.values() for v in ss]
        x_min = float(np.min(all_samples))
        x_max = float(np.max(all_samples))
        pad = (x_max - x_min) * 0.1
        x_min -= pad
        x_max += pad
        x_ticks = self._nice_ticks(x_min, x_max)
        x_min = min(x_ticks)
        x_max = max(x_ticks)

        # compute all densities first to find global y_max
        densities: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        global_d_max = 0.0
        for label, samples in posterior_diffs.items():
            grid, dens = self._kernel_density_estimate(samples, n_points=200)
            # clip to x range
            mask = (grid >= x_min) & (grid <= x_max)
            grid = grid[mask]
            dens = dens[mask]
            densities[label] = (grid, dens)
            if len(dens) > 0:
                global_d_max = max(global_d_max, float(np.max(dens)))

        if global_d_max < 1e-15:
            global_d_max = 1.0

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Bayesian Posterior Differences"))

        y_ticks = self._nice_ticks(0, global_d_max)
        y_max_axis = max(y_ticks) if y_ticks else global_d_max
        parts.append(self._draw_axes(
            w, h, None, y_ticks, 0, y_max_axis,
            x_title="Difference", y_title="Density",
        ))
        # x-axis ticks
        for xt in x_ticks:
            px = self._val_to_x(xt, x_min, x_max)
            py = m["top"] + ph
            parts.append(
                f'<line x1="{px:.1f}" y1="{py}" x2="{px:.1f}" '
                f'y2="{py + 5}" stroke="{self.config.axis_color}" '
                f'stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{py + 18}" text-anchor="middle" '
                f'font-size="{self.config.font_size}" '
                f'fill="{self.config.axis_color}">{xt:.2g}</text>'
            )

        # zero reference line
        zero_px = self._val_to_x(0, x_min, x_max)
        parts.append(
            f'<line x1="{zero_px:.1f}" y1="{m["top"]}" '
            f'x2="{zero_px:.1f}" y2="{m["top"] + ph:.1f}" '
            f'stroke="#CC0000" stroke-width="1" stroke-dasharray="5,4"/>'
        )

        for ai, (label, (grid, dens)) in enumerate(densities.items()):
            if len(grid) == 0:
                continue
            color = self._color(ai)
            # build area polygon
            baseline_y = self._val_to_y(0, 0, y_max_axis)
            top_pts: List[str] = []
            for gv, dv in zip(grid, dens):
                px = self._val_to_x(gv, x_min, x_max)
                py = self._val_to_y(dv, 0, y_max_axis)
                top_pts.append(f"{px:.2f},{py:.2f}")
            # close at baseline
            first_x = self._val_to_x(float(grid[0]), x_min, x_max)
            last_x = self._val_to_x(float(grid[-1]), x_min, x_max)
            polygon_pts = (
                f"{first_x:.2f},{baseline_y:.2f} "
                + " ".join(top_pts)
                + f" {last_x:.2f},{baseline_y:.2f}"
            )
            parts.append(
                f'<polygon points="{polygon_pts}" fill="{color}" '
                f'fill-opacity="0.25" stroke="{color}" stroke-width="1.5"/>'
            )

            # ROPE / probability annotation
            samples_arr = np.asarray(posterior_diffs[label])
            prob_positive = float(np.mean(samples_arr > 0))
            # place label at density peak
            peak_idx = int(np.argmax(dens))
            peak_px = self._val_to_x(float(grid[peak_idx]), x_min, x_max)
            peak_py = self._val_to_y(float(dens[peak_idx]), 0, y_max_axis)
            parts.append(
                f'<text x="{peak_px:.1f}" y="{peak_py - 8:.1f}" '
                f'text-anchor="middle" font-size="{self.config.font_size - 1}" '
                f'font-family="{self.config.font_family}" '
                f'fill="{color}">{_escape(label)} '
                f'(P&gt;0={prob_positive:.1%})</text>'
            )

        legend_items = [(l, self._color(i)) for i, l in enumerate(posterior_diffs)]
        parts.append(self._draw_legend(legend_items, m["left"] + 10, m["top"] + 10))
        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 14. Performance profile
    # ------------------------------------------------------------------

    def plot_performance_profile(
        self,
        algo_ratios: Dict[str, List[float]],
    ) -> str:
        """algo_ratios maps algo_name -> sorted list of performance ratios (>=1)."""
        if not algo_ratios:
            return "<svg></svg>"

        w, h = self.config.figsize
        m = self.config.margin
        pw = self.config.plot_width()
        ph = self.config.plot_height()

        all_ratios = [r for rs in algo_ratios.values() for r in rs]
        x_min = 1.0
        x_max = max(all_ratios) * 1.05 if all_ratios else 2.0
        x_ticks = self._nice_ticks(x_min, x_max)
        x_max = max(x_ticks) if x_ticks else x_max

        y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Performance Profile"))
        parts.append(self._draw_axes(
            w, h, None, y_ticks, 0, 1.0,
            x_title="Performance Ratio (τ)", y_title="P(ratio ≤ τ)",
        ))
        # x-axis ticks
        for xt in x_ticks:
            px = self._val_to_x(xt, x_min, x_max)
            py = m["top"] + ph
            parts.append(
                f'<text x="{px:.1f}" y="{py + 18}" text-anchor="middle" '
                f'font-size="{self.config.font_size}" '
                f'fill="{self.config.axis_color}">{xt:.2g}</text>'
            )

        for ai, (algo, ratios) in enumerate(algo_ratios.items()):
            color = self._color(ai)
            sorted_r = sorted(ratios)
            n = len(sorted_r)
            if n == 0:
                continue
            pts: List[str] = []
            # step function
            prev_x = self._val_to_x(x_min, x_min, x_max)
            prev_y = self._val_to_y(0, 0, 1.0)
            pts.append(f"{prev_x:.2f},{prev_y:.2f}")
            for i, rv in enumerate(sorted_r):
                frac = (i + 1) / n
                rx = self._val_to_x(rv, x_min, x_max)
                # horizontal then vertical (step)
                pts.append(f"{rx:.2f},{prev_y:.2f}")
                ny = self._val_to_y(frac, 0, 1.0)
                pts.append(f"{rx:.2f},{ny:.2f}")
                prev_y = ny
            # extend to x_max
            rx_end = self._val_to_x(x_max, x_min, x_max)
            pts.append(f"{rx_end:.2f},{prev_y:.2f}")

            parts.append(
                f'<polyline points="{" ".join(pts)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )

        legend_items = [(a, self._color(i)) for i, a in enumerate(algo_ratios)]
        parts.append(self._draw_legend(legend_items, m["left"] + 10, m["top"] + 10))
        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 15. Ablation chart
    # ------------------------------------------------------------------

    def plot_ablation_chart(
        self,
        ablation_results: Dict[str, Dict[str, float]],
    ) -> str:
        """ablation_results maps variant_name -> {metric: score}."""
        if not ablation_results:
            return "<svg></svg>"

        variants = list(ablation_results.keys())
        metrics = list(
            dict.fromkeys(m for d in ablation_results.values() for m in d)
        )
        n_variants = len(variants)
        n_metrics = len(metrics)

        w = max(self.config.figsize[0], 600)
        h = max(self.config.figsize[1], 400)
        m = self.config.margin
        pw = w - m["left"] - m["right"]
        ph = h - m["top"] - m["bottom"]

        all_vals = [
            ablation_results[v].get(met, 0)
            for v in variants for met in metrics
        ]
        v_min = 0.0
        v_max = max(all_vals) * 1.15 if all_vals and max(all_vals) > 0 else 1.0
        ticks = self._nice_ticks(v_min, v_max)
        v_max = max(ticks) if ticks else v_max

        group_w = pw / n_variants
        bar_w = (group_w * 0.8) / max(n_metrics, 1)

        parts: List[str] = [_svg_open(w, h, self.config.background_color)]
        parts.append(self._draw_title(w, self.config.title or "Ablation Study"))
        parts.append(self._draw_axes(
            w, h, variants, ticks, v_min, v_max,
            y_title="Score", rotate_x=n_variants > 5,
        ))

        # reference variant (first or full model)
        ref_variant = variants[0]
        ref_scores = ablation_results[ref_variant]

        for vi, variant in enumerate(variants):
            gx = m["left"] + group_w * vi
            offset = group_w * 0.1
            for mi, met in enumerate(metrics):
                val = ablation_results[variant].get(met, 0)
                bx = gx + offset + bar_w * mi
                bar_h = (val / v_max) * ph if v_max > 0 else 0
                by = m["top"] + ph - bar_h
                color = self._color(mi)

                # highlight degradation from reference
                ref_val = ref_scores.get(met, 0)
                opacity = 1.0
                if variant != ref_variant and val < ref_val * 0.95:
                    opacity = 0.6  # dimmed for degradation

                parts.append(self._svg_bar(
                    bx, by, bar_w * 0.85, bar_h, color, opacity=opacity,
                ))

                # delta annotation
                if variant != ref_variant and ref_val > 0:
                    delta = ((val - ref_val) / ref_val) * 100
                    sign = "+" if delta >= 0 else ""
                    d_color = "#2ecc71" if delta >= 0 else "#e74c3c"
                    parts.append(
                        f'<text x="{bx + bar_w * 0.425:.1f}" '
                        f'y="{by - 4:.1f}" text-anchor="middle" '
                        f'font-size="{self.config.font_size - 2}" '
                        f'fill="{d_color}">{sign}{delta:.1f}%</text>'
                    )

        legend_items = [(met, self._color(i)) for i, met in enumerate(metrics)]
        parts.append(self._draw_legend(
            legend_items, w - m["right"] - 130, m["top"] + 10,
        ))

        parts.append(_svg_close())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # HTML table formatter
    # ------------------------------------------------------------------

    def _format_table_html(
        self,
        headers: List[str],
        rows: List[List[str]],
        styles: Optional[Dict[str, str]] = None,
    ) -> str:
        s = styles or {}
        tbl_style = s.get("table", "border-collapse:collapse; width:100%;")
        th_style = s.get(
            "th",
            "background:#4C72B0; color:white; padding:8px; "
            "text-align:center; border:1px solid #ddd;",
        )
        td_style = s.get("td", "padding:6px; text-align:center; border:1px solid #ddd;")
        tr_even = s.get("tr_even", "background:#f9f9f9;")
        tr_odd = s.get("tr_odd", "background:#ffffff;")
        rank_styles = {
            "1": s.get("rank1", "background:#FFD700; font-weight:bold;"),
            "2": s.get("rank2", "background:#C0C0C0;"),
            "3": s.get("rank3", "background:#CD7F32; color:white;"),
        }

        lines: List[str] = [f'<table style="{tbl_style}">']
        lines.append("<thead><tr>")
        for hdr in headers:
            lines.append(f'  <th style="{th_style}">{_escape(hdr)}</th>')
        lines.append("</tr></thead>")
        lines.append("<tbody>")

        for ri, row in enumerate(rows):
            row_style = tr_even if ri % 2 == 0 else tr_odd
            rank_val = row[0] if row else ""
            extra = rank_styles.get(rank_val, "")
            lines.append(f'<tr style="{row_style}{extra}">')
            for ci, cell in enumerate(row):
                cell_extra = ""
                if ci == 0 and cell in rank_styles:
                    cell_extra = rank_styles[cell]
                lines.append(f'  <td style="{td_style}{cell_extra}">{_escape(cell)}</td>')
            lines.append("</tr>")

        lines.append("</tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    # ==================================================================
    # Dashboard
    # ==================================================================

    def generate_comparison_dashboard(
        self,
        all_results: Dict[str, Any],
    ) -> str:
        """Generate a multi-panel HTML dashboard from heterogeneous results.

        Expected keys in *all_results* (all optional):
          - 'bar_scores': Dict[str, float]
          - 'grouped_scores': Dict[str, Dict[str, float]]
          - 'distributions': Dict[str, List[float]]
          - 'rankings': Dict[str, Dict[str, int]]
          - 'pairwise': Dict[str, Dict[str, float]]
          - 'convergence': Dict[str, List[float]]
          - 'runtimes': Dict[str, float]
          - 'diversity_quality': Dict[str, Dict[str, float]]
          - 'effect_sizes': Dict[str, Dict[str, float]]
          - 'posterior_diffs': Dict[str, List[float]]
          - 'performance_ratios': Dict[str, List[float]]
          - 'ablation': Dict[str, Dict[str, float]]
          - 'radar_scores': Dict[str, Dict[str, float]]
          - 'radar_metrics': List[str]
          - 'avg_ranks': Dict[str, float]
          - 'n_datasets': int
          - 'cd_p_values': Dict[Tuple[str, str], float]
        """

        panels: List[str] = []

        if "bar_scores" in all_results:
            svg = self.plot_bar_comparison(
                all_results["bar_scores"],
                all_results.get("bar_metric", "Score"),
            )
            panels.append(self._dashboard_panel("Bar Comparison", svg))

        if "grouped_scores" in all_results:
            svg = self.plot_grouped_bars(all_results["grouped_scores"])
            panels.append(self._dashboard_panel("Grouped Comparison", svg))

        if "distributions" in all_results:
            svg_box = self.plot_box_comparison(all_results["distributions"])
            panels.append(self._dashboard_panel("Box Plot", svg_box))
            svg_violin = self.plot_violin_comparison(all_results["distributions"])
            panels.append(self._dashboard_panel("Violin Plot", svg_violin))

        if "radar_scores" in all_results:
            svg = self.plot_radar_chart(
                all_results["radar_scores"],
                all_results.get("radar_metrics"),
            )
            panels.append(self._dashboard_panel("Radar Chart", svg))

        if "rankings" in all_results:
            tbl = self.plot_ranking_table(all_results["rankings"])
            panels.append(self._dashboard_panel("Rankings", tbl, is_html=True))

        if "pairwise" in all_results:
            svg = self.plot_win_loss_matrix(all_results["pairwise"])
            panels.append(self._dashboard_panel("Win/Loss Matrix", svg))

        if "avg_ranks" in all_results:
            svg = self.plot_critical_difference_diagram(
                all_results["avg_ranks"],
                all_results.get("n_datasets", 10),
                all_results.get("cd_p_values"),
            )
            panels.append(self._dashboard_panel("Critical Difference", svg))

        if "convergence" in all_results:
            svg = self.plot_convergence_curves(all_results["convergence"])
            panels.append(self._dashboard_panel("Convergence", svg))

        if "runtimes" in all_results:
            svg = self.plot_runtime_comparison(all_results["runtimes"])
            panels.append(self._dashboard_panel("Runtime", svg))

        if "diversity_quality" in all_results:
            svg = self.plot_diversity_quality_scatter(all_results["diversity_quality"])
            panels.append(self._dashboard_panel("Diversity vs Quality", svg))

        if "effect_sizes" in all_results:
            svg = self.plot_effect_size_forest(all_results["effect_sizes"])
            panels.append(self._dashboard_panel("Effect Sizes", svg))

        if "posterior_diffs" in all_results:
            svg = self.plot_bayesian_comparison(all_results["posterior_diffs"])
            panels.append(self._dashboard_panel("Bayesian Comparison", svg))

        if "performance_ratios" in all_results:
            svg = self.plot_performance_profile(all_results["performance_ratios"])
            panels.append(self._dashboard_panel("Performance Profile", svg))

        if "ablation" in all_results:
            svg = self.plot_ablation_chart(all_results["ablation"])
            panels.append(self._dashboard_panel("Ablation Study", svg))

        # Assemble full HTML page
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'/>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>",
            "<title>Algorithm Comparison Dashboard</title>",
            "<style>",
            "body { font-family: " + self.config.font_family + "; "
            "margin: 0; padding: 20px; background: #f5f5f5; }",
            "h1 { text-align: center; color: #333; margin-bottom: 30px; }",
            ".dashboard-grid { display: grid; "
            "grid-template-columns: repeat(auto-fit, minmax(700px, 1fr)); "
            "gap: 20px; max-width: 1600px; margin: 0 auto; }",
            ".panel { background: white; border-radius: 8px; "
            "box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 16px; "
            "overflow-x: auto; }",
            ".panel h2 { margin: 0 0 12px 0; font-size: 16px; "
            "color: #555; border-bottom: 2px solid #4C72B0; "
            "padding-bottom: 6px; }",
            ".panel svg { max-width: 100%; height: auto; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Algorithm Comparison Dashboard</h1>",
            '<div class="dashboard-grid">',
        ]
        html_parts.extend(panels)
        html_parts.extend([
            "</div>",
            "<footer style='text-align:center; margin-top:30px; "
            "color:#888; font-size:12px;'>",
            "Generated by Diversity Decoding Arena &mdash; "
            "AlgorithmComparisonPlotter",
            "</footer>",
            "</body>",
            "</html>",
        ])
        return "\n".join(html_parts)

    def _dashboard_panel(
        self,
        title: str,
        content: str,
        is_html: bool = False,
    ) -> str:
        inner = content if is_html else content
        return (
            f'<div class="panel">\n'
            f"  <h2>{_escape(title)}</h2>\n"
            f"  {inner}\n"
            f"</div>"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_plotter(**kwargs: Any) -> AlgorithmComparisonPlotter:
    """Create an AlgorithmComparisonPlotter with keyword config overrides."""
    config = ComparisonConfig(**kwargs)
    return AlgorithmComparisonPlotter(config)


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng_np = np.random.default_rng(42)
    plotter = create_plotter(title="Demo Dashboard")

    # Fabricate demo data
    algos = ["DPP", "MMR", "TopK", "Nucleus", "Beam"]
    metrics = ["BLEU", "Diversity", "Coherence"]

    bar_scores = {a: float(rng_np.uniform(0.3, 0.9)) for a in algos}
    grouped = {a: {m: float(rng_np.uniform(0.2, 1.0)) for m in metrics} for a in algos}
    distributions = {a: rng_np.normal(0.5 + i * 0.05, 0.1, 50).tolist() for i, a in enumerate(algos)}
    rankings = {a: {m: int(rng_np.integers(1, 6)) for m in metrics} for a in algos}
    pairwise = {a: {b: float(rng_np.uniform(0.2, 0.8)) for b in algos} for a in algos}
    for a in algos:
        pairwise[a][a] = 0.5
    convergence = {a: np.cumsum(rng_np.uniform(0, 0.05, 30)).tolist() for a in algos}
    runtimes = {a: float(rng_np.uniform(0.5, 5.0)) for a in algos}
    dq = {a: {"diversity": float(rng_np.uniform(0.3, 0.9)),
              "quality": float(rng_np.uniform(0.4, 0.95))} for a in algos}
    effect = {f"{algos[i]} vs {algos[i+1]}": {
        "d": float(rng_np.normal(0.3, 0.5)),
        "ci_low": float(rng_np.normal(-0.2, 0.3)),
        "ci_high": float(rng_np.normal(0.8, 0.3)),
    } for i in range(len(algos) - 1)}
    posterior = {f"{algos[i]} - {algos[i+1]}": rng_np.normal(0.1 * i, 0.2, 500).tolist()
                 for i in range(len(algos) - 1)}
    perf_ratios = {a: sorted(rng_np.uniform(1.0, 3.0, 20).tolist()) for a in algos}
    ablation = {
        "Full": {m: float(rng_np.uniform(0.7, 0.9)) for m in metrics},
        "No-Diversity": {m: float(rng_np.uniform(0.5, 0.8)) for m in metrics},
        "No-Rerank": {m: float(rng_np.uniform(0.4, 0.7)) for m in metrics},
    }

    dashboard = plotter.generate_comparison_dashboard({
        "bar_scores": bar_scores,
        "bar_metric": "Overall Score",
        "grouped_scores": grouped,
        "distributions": distributions,
        "radar_scores": grouped,
        "radar_metrics": metrics,
        "rankings": rankings,
        "pairwise": pairwise,
        "avg_ranks": {a: float(np.mean(list(rankings[a].values()))) for a in algos},
        "n_datasets": 10,
        "convergence": convergence,
        "runtimes": runtimes,
        "diversity_quality": dq,
        "effect_sizes": effect,
        "posterior_diffs": posterior,
        "performance_ratios": perf_ratios,
        "ablation": ablation,
    })
    print(f"Dashboard generated: {len(dashboard)} characters")
