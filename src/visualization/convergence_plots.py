"""
Convergence visualization module for the Diversity Decoding Arena.

Provides visualizations for metric convergence analysis including:
- Metric convergence as a function of sample size
- Bootstrap variance convergence
- EPD (Expected Pairwise Distance) convergence
- Convergence rate comparison across algorithms

All visualizations produce SVG strings using a lightweight SVG builder,
with numpy for numerical computation.
"""

from __future__ import annotations

import math
import html as html_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceConfig:
    """Configuration for convergence plots."""
    figsize: Tuple[int, int] = (800, 500)
    font_size: int = 13
    title: str = "Convergence Plot"
    xlabel: str = "Sample Size"
    ylabel: str = "Metric Value"
    grid: bool = True
    show_ci: bool = True
    ci_alpha: float = 0.2
    line_width: float = 2.0
    margin_left: int = 80
    margin_right: int = 40
    margin_top: int = 50
    margin_bottom: int = 55

    @property
    def plot_width(self) -> int:
        return self.figsize[0] - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> int:
        return self.figsize[1] - self.margin_top - self.margin_bottom


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="{w}" height="{h}" '
    'viewBox="0 0 {w} {h}" style="background:#ffffff;">'
)
_SVG_FOOTER = "</svg>"


def _svg_text(
    x: float, y: float, text: str, *,
    size: int = 12, anchor: str = "start",
    weight: str = "normal", rotate: Optional[float] = None,
    fill: str = "#333",
) -> str:
    escaped = html_module.escape(str(text))
    rot = f' transform="rotate({rotate},{x},{y})"' if rotate else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{rot}>'
        f'{escaped}</text>'
    )


def _svg_line(
    x1: float, y1: float, x2: float, y2: float,
    color: str = "#ccc", width: float = 1.0,
    dash: str = "",
) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{d}/>'
    )


def _empty_svg(message: str) -> str:
    escaped = html_module.escape(message)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
        f'<text x="200" y="45" text-anchor="middle" font-size="14" '
        f'fill="#888">{escaped}</text></svg>'
    )


# ---------------------------------------------------------------------------
# Core convergence computation helpers
# ---------------------------------------------------------------------------

def _compute_running_metric(
    values: np.ndarray,
    sample_sizes: Optional[List[int]] = None,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute running mean and CI of a metric at increasing sample sizes.

    Returns (sample_sizes, means, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if sample_sizes is None:
        sample_sizes_arr = np.unique(np.linspace(
            max(2, n // 20), n, min(30, n), dtype=int,
        ))
    else:
        sample_sizes_arr = np.array(sample_sizes)

    means = np.zeros(len(sample_sizes_arr))
    ci_lo = np.zeros(len(sample_sizes_arr))
    ci_hi = np.zeros(len(sample_sizes_arr))

    for si, ss in enumerate(sample_sizes_arr):
        ss = min(int(ss), n)
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.randint(0, n, size=ss)
            boot_means[b] = np.mean(values[idx])
        means[si] = np.mean(boot_means)
        ci_lo[si] = np.percentile(boot_means, 2.5)
        ci_hi[si] = np.percentile(boot_means, 97.5)

    return sample_sizes_arr, means, ci_lo, ci_hi


def _compute_bootstrap_variance_curve(
    values: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap variance of the mean at increasing sample sizes."""
    rng = np.random.RandomState(seed)
    n = len(values)
    sizes = np.unique(np.linspace(max(2, n // 20), n, min(30, n), dtype=int))
    variances = np.zeros(len(sizes))

    for si, ss in enumerate(sizes):
        ss = min(int(ss), n)
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.randint(0, n, size=ss)
            boot_means[b] = np.mean(values[idx])
        variances[si] = np.var(boot_means)

    return sizes, variances


def _compute_epd(
    values: np.ndarray, max_pairs: int = 5000, seed: int = 42,
) -> float:
    """Compute Expected Pairwise Distance (mean |x_i - x_j|)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < 2:
        return 0.0
    n_pairs = min(n * (n - 1) // 2, max_pairs)
    total = 0.0
    for _ in range(n_pairs):
        i, j = rng.randint(0, n, size=2)
        total += abs(values[i] - values[j])
    return total / n_pairs


# ---------------------------------------------------------------------------
# ConvergencePlotter
# ---------------------------------------------------------------------------


class ConvergencePlotter:
    """Generate convergence visualizations as SVG strings."""

    def __init__(self, config: Optional[ConvergenceConfig] = None) -> None:
        self.config = config or ConvergenceConfig()

    # ----- axis drawing helper -----

    def _draw_axes(
        self, parts: List[str],
        x_min: float, x_max: float, y_min: float, y_max: float,
        xlabel: str = "", ylabel: str = "",
    ) -> None:
        cfg = self.config
        ml, mt = cfg.margin_left, cfg.margin_top
        pw, ph = cfg.plot_width, cfg.plot_height

        # border
        parts.append(
            f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
            f'fill="#fafafa" stroke="#ccc" stroke-width="1"/>'
        )

        if cfg.grid:
            for i in range(1, 5):
                gy = mt + ph * i / 4
                parts.append(_svg_line(ml, gy, ml + pw, gy, "#eee", 0.5))
                gx = ml + pw * i / 4
                parts.append(_svg_line(gx, mt, gx, mt + ph, "#eee", 0.5))

        # x ticks
        x_rng = x_max - x_min if x_max != x_min else 1.0
        for i in range(5):
            v = x_min + x_rng * i / 4
            tx = ml + pw * i / 4
            parts.append(_svg_text(tx, mt + ph + 18, f"{v:.0f}",
                                   size=cfg.font_size - 3, anchor="middle"))

        # y ticks
        y_rng = y_max - y_min if y_max != y_min else 1.0
        for i in range(5):
            v = y_min + y_rng * (4 - i) / 4
            ty = mt + ph * i / 4
            parts.append(_svg_text(ml - 8, ty + 4, f"{v:.4f}",
                                   size=cfg.font_size - 3, anchor="end"))

        if xlabel:
            parts.append(_svg_text(ml + pw / 2, mt + ph + 42, xlabel,
                                   size=cfg.font_size, anchor="middle"))
        if ylabel:
            parts.append(_svg_text(16, mt + ph / 2, ylabel,
                                   size=cfg.font_size, anchor="middle", rotate=-90))

    # ----- coordinate transforms -----

    def _to_svg_x(self, val: float, vmin: float, vmax: float) -> float:
        cfg = self.config
        rng = vmax - vmin if vmax != vmin else 1.0
        return cfg.margin_left + (val - vmin) / rng * cfg.plot_width

    def _to_svg_y(self, val: float, vmin: float, vmax: float) -> float:
        cfg = self.config
        rng = vmax - vmin if vmax != vmin else 1.0
        return cfg.margin_top + cfg.plot_height - (val - vmin) / rng * cfg.plot_height

    # ----- public API -----

    def plot_metric_convergence(
        self,
        data_dict: Dict[str, np.ndarray],
        title: str = "Metric Convergence vs Sample Size",
        n_bootstrap: int = 50,
    ) -> str:
        """Plot convergence of multiple metrics as sample size grows.

        Parameters
        ----------
        data_dict : mapping metric_name → array of observed values
        """
        cfg = self.config
        if not data_dict:
            return _empty_svg("No data for convergence plot")

        w, h = cfg.figsize
        parts: List[str] = [_SVG_HEADER.format(w=w, h=h)]
        parts.append(_svg_text(w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, vals in data_dict.items():
            arr = np.asarray(vals, dtype=float)
            if len(arr) < 3:
                continue
            curves[name] = _compute_running_metric(arr, n_bootstrap=n_bootstrap)

        if not curves:
            parts.append('</svg>')
            return "".join(parts)

        all_sizes = np.concatenate([c[0] for c in curves.values()])
        all_vals = np.concatenate([
            np.concatenate([c[2], c[3]]) for c in curves.values()
        ])
        x_min, x_max = float(all_sizes.min()), float(all_sizes.max())
        y_min, y_max = float(all_vals.min()), float(all_vals.max())
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
        y_min -= pad
        y_max += pad

        self._draw_axes(parts, x_min, x_max, y_min, y_max, cfg.xlabel, cfg.ylabel)

        names = sorted(curves.keys())
        for ci, name in enumerate(names):
            sizes, means, lo, hi = curves[name]
            color = _COLORS[ci % len(_COLORS)]

            # CI band
            if cfg.show_ci:
                poly_points = []
                for k in range(len(sizes)):
                    sx = self._to_svg_x(sizes[k], x_min, x_max)
                    sy = self._to_svg_y(hi[k], y_min, y_max)
                    poly_points.append(f"{sx},{sy}")
                for k in range(len(sizes) - 1, -1, -1):
                    sx = self._to_svg_x(sizes[k], x_min, x_max)
                    sy = self._to_svg_y(lo[k], y_min, y_max)
                    poly_points.append(f"{sx},{sy}")
                parts.append(
                    f'<polygon points="{" ".join(poly_points)}" '
                    f'fill="{color}" fill-opacity="{cfg.ci_alpha}" stroke="none"/>'
                )

            # mean line
            path_d = []
            for k in range(len(sizes)):
                sx = self._to_svg_x(sizes[k], x_min, x_max)
                sy = self._to_svg_y(means[k], y_min, y_max)
                cmd = "M" if k == 0 else "L"
                path_d.append(f"{cmd}{sx},{sy}")
            parts.append(
                f'<path d="{" ".join(path_d)}" fill="none" '
                f'stroke="{color}" stroke-width="{cfg.line_width}"/>'
            )

        # legend
        leg_x = cfg.margin_left + cfg.plot_width - 130
        leg_y = cfg.margin_top + 15
        parts.append(
            f'<rect x="{leg_x-5}" y="{leg_y-12}" width="135" '
            f'height="{len(names)*20+8}" fill="white" stroke="#ccc" rx="3"/>'
        )
        for ci, name in enumerate(names):
            color = _COLORS[ci % len(_COLORS)]
            cy = leg_y + ci * 20
            parts.append(
                f'<line x1="{leg_x}" y1="{cy}" x2="{leg_x+20}" y2="{cy}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            parts.append(_svg_text(leg_x + 25, cy + 4, name,
                                   size=cfg.font_size - 2))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_bootstrap_variance_convergence(
        self,
        data_dict: Dict[str, np.ndarray],
        title: str = "Bootstrap Variance Convergence",
        n_bootstrap: int = 100,
    ) -> str:
        """Plot how bootstrap variance of each metric decreases with sample size."""
        cfg = self.config
        if not data_dict:
            return _empty_svg("No data")

        w, h = cfg.figsize
        parts: List[str] = [_SVG_HEADER.format(w=w, h=h)]
        parts.append(_svg_text(w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, vals in data_dict.items():
            arr = np.asarray(vals, dtype=float)
            if len(arr) < 3:
                continue
            curves[name] = _compute_bootstrap_variance_curve(arr, n_bootstrap)

        if not curves:
            parts.append(_SVG_FOOTER)
            return "".join(parts)

        all_sizes = np.concatenate([c[0] for c in curves.values()])
        all_vars = np.concatenate([c[1] for c in curves.values()])
        x_min, x_max = float(all_sizes.min()), float(all_sizes.max())
        y_min = 0.0
        y_max = float(all_vars.max()) * 1.1 if all_vars.max() > 0 else 1.0

        self._draw_axes(parts, x_min, x_max, y_min, y_max,
                        "Sample Size", "Bootstrap Variance")

        names = sorted(curves.keys())
        for ci, name in enumerate(names):
            sizes, variances = curves[name]
            color = _COLORS[ci % len(_COLORS)]
            path_d = []
            for k in range(len(sizes)):
                sx = self._to_svg_x(sizes[k], x_min, x_max)
                sy = self._to_svg_y(variances[k], y_min, y_max)
                cmd = "M" if k == 0 else "L"
                path_d.append(f"{cmd}{sx},{sy}")
            parts.append(
                f'<path d="{" ".join(path_d)}" fill="none" '
                f'stroke="{color}" stroke-width="{cfg.line_width}"/>'
            )
            # mark points
            for k in range(len(sizes)):
                sx = self._to_svg_x(sizes[k], x_min, x_max)
                sy = self._to_svg_y(variances[k], y_min, y_max)
                parts.append(
                    f'<circle cx="{sx}" cy="{sy}" r="2.5" fill="{color}"/>'
                )

        # legend
        leg_x = cfg.margin_left + cfg.plot_width - 130
        leg_y = cfg.margin_top + 15
        parts.append(
            f'<rect x="{leg_x-5}" y="{leg_y-12}" width="135" '
            f'height="{len(names)*20+8}" fill="white" stroke="#ccc" rx="3"/>'
        )
        for ci, name in enumerate(names):
            color = _COLORS[ci % len(_COLORS)]
            cy = leg_y + ci * 20
            parts.append(
                f'<line x1="{leg_x}" y1="{cy}" x2="{leg_x+20}" y2="{cy}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            parts.append(_svg_text(leg_x + 25, cy + 4, name,
                                   size=cfg.font_size - 2))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_epd_convergence(
        self,
        data_dict: Dict[str, np.ndarray],
        title: str = "EPD Convergence",
        n_steps: int = 25,
    ) -> str:
        """Plot EPD (Expected Pairwise Distance) convergence with sample size.

        The EPD is recomputed on increasing subsets to show convergence.
        """
        cfg = self.config
        if not data_dict:
            return _empty_svg("No data")

        w, h = cfg.figsize
        parts: List[str] = [_SVG_HEADER.format(w=w, h=h)]
        parts.append(_svg_text(w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, vals in data_dict.items():
            arr = np.asarray(vals, dtype=float)
            n = len(arr)
            if n < 4:
                continue
            sizes = np.unique(np.linspace(4, n, min(n_steps, n), dtype=int))
            epds = np.zeros(len(sizes))
            for si, ss in enumerate(sizes):
                epds[si] = _compute_epd(arr[:int(ss)])
            curves[name] = (sizes, epds)

        if not curves:
            parts.append(_SVG_FOOTER)
            return "".join(parts)

        all_sizes = np.concatenate([c[0] for c in curves.values()])
        all_epds = np.concatenate([c[1] for c in curves.values()])
        x_min, x_max = float(all_sizes.min()), float(all_sizes.max())
        y_min = 0.0
        y_max = float(all_epds.max()) * 1.1 if all_epds.max() > 0 else 1.0

        self._draw_axes(parts, x_min, x_max, y_min, y_max, "Sample Size", "EPD")

        names = sorted(curves.keys())
        for ci, name in enumerate(names):
            sizes, epds = curves[name]
            color = _COLORS[ci % len(_COLORS)]
            path_d = []
            for k in range(len(sizes)):
                sx = self._to_svg_x(sizes[k], x_min, x_max)
                sy = self._to_svg_y(epds[k], y_min, y_max)
                cmd = "M" if k == 0 else "L"
                path_d.append(f"{cmd}{sx},{sy}")
            parts.append(
                f'<path d="{" ".join(path_d)}" fill="none" '
                f'stroke="{color}" stroke-width="{cfg.line_width}"/>'
            )

        # legend
        leg_x = cfg.margin_left + cfg.plot_width - 130
        leg_y = cfg.margin_top + 15
        parts.append(
            f'<rect x="{leg_x-5}" y="{leg_y-12}" width="135" '
            f'height="{len(names)*20+8}" fill="white" stroke="#ccc" rx="3"/>'
        )
        for ci, name in enumerate(names):
            color = _COLORS[ci % len(_COLORS)]
            cy = leg_y + ci * 20
            parts.append(
                f'<line x1="{leg_x}" y1="{cy}" x2="{leg_x+20}" y2="{cy}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            parts.append(_svg_text(leg_x + 25, cy + 4, name,
                                   size=cfg.font_size - 2))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_convergence_rate_comparison(
        self,
        data_dict: Dict[str, np.ndarray],
        title: str = "Convergence Rate Comparison",
        n_bootstrap: int = 50,
    ) -> str:
        """Bar chart comparing convergence rates across algorithms/metrics.

        Convergence rate is estimated as the ratio of variance at 25% sample
        size to variance at full sample size (higher → slower convergence).
        """
        cfg = self.config
        if not data_dict:
            return _empty_svg("No data")

        rates: Dict[str, float] = {}
        for name, vals in data_dict.items():
            arr = np.asarray(vals, dtype=float)
            if len(arr) < 6:
                continue
            sizes, variances = _compute_bootstrap_variance_curve(arr, n_bootstrap)
            if len(variances) < 2 or variances[-1] == 0:
                rates[name] = 0.0
            else:
                quarter_idx = max(0, len(variances) // 4)
                rates[name] = float(variances[quarter_idx] / variances[-1])

        if not rates:
            return _empty_svg("Not enough data for convergence rates")

        names = sorted(rates.keys(), key=lambda k: rates[k], reverse=True)
        w, h = cfg.figsize
        parts: List[str] = [_SVG_HEADER.format(w=w, h=h)]
        parts.append(_svg_text(w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        max_rate = max(rates.values()) if rates else 1.0
        if max_rate == 0:
            max_rate = 1.0
        n = len(names)
        bar_area_h = cfg.plot_height
        bar_h = max(8, min(30, bar_area_h / max(n, 1) - 6))
        gap = max(4, (bar_area_h - bar_h * n) / max(n, 1))
        ml = cfg.margin_left + 60  # extra for labels
        bar_max_w = cfg.plot_width - 60

        for i, name in enumerate(names):
            val = rates[name]
            bw = val / max_rate * bar_max_w
            by = cfg.margin_top + i * (bar_h + gap)
            color = _COLORS[i % len(_COLORS)]
            parts.append(
                f'<rect x="{ml}" y="{by}" width="{bw}" height="{bar_h}" '
                f'fill="{color}" rx="3"/>'
            )
            parts.append(_svg_text(
                ml - 6, by + bar_h / 2 + 4, name,
                size=cfg.font_size - 2, anchor="end",
            ))
            parts.append(_svg_text(
                ml + bw + 6, by + bar_h / 2 + 4, f"{val:.2f}",
                size=cfg.font_size - 3, anchor="start",
            ))

        # x-axis label
        parts.append(_svg_text(
            ml + bar_max_w / 2, h - 12,
            "Convergence Rate (Var@25% / Var@100%)",
            size=cfg.font_size - 1, anchor="middle",
        ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_convergence_plotter(**kwargs: Any) -> ConvergencePlotter:
    """Create a ConvergencePlotter with optional config overrides."""
    return ConvergencePlotter(ConvergenceConfig(**kwargs))


def quick_convergence(data_dict: Dict[str, np.ndarray], **kwargs: Any) -> str:
    """One-call shortcut for a metric convergence plot."""
    plotter = create_convergence_plotter(**kwargs)
    return plotter.plot_metric_convergence(data_dict)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------


def _demo() -> None:  # pragma: no cover
    """Generate sample convergence visualizations."""
    rng = np.random.RandomState(42)

    data = {
        "diversity": rng.normal(0.5, 0.15, 200),
        "coherence": rng.normal(0.7, 0.10, 200),
        "novelty": rng.normal(0.4, 0.20, 200),
    }

    plotter = ConvergencePlotter()

    svg1 = plotter.plot_metric_convergence(data)
    print(f"Metric convergence SVG: {len(svg1)} chars")

    svg2 = plotter.plot_bootstrap_variance_convergence(data)
    print(f"Bootstrap variance SVG: {len(svg2)} chars")

    svg3 = plotter.plot_epd_convergence(data)
    print(f"EPD convergence SVG: {len(svg3)} chars")

    svg4 = plotter.plot_convergence_rate_comparison(data)
    print(f"Convergence rate SVG: {len(svg4)} chars")

    print("\nAll convergence visualizations generated successfully.")


if __name__ == "__main__":
    _demo()
