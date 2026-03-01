"""
Metric correlation heatmap visualization for the Diversity Decoding Arena.

Generates SVG strings directly without matplotlib. Uses numpy for numerical
computation and scipy.stats for correlation when available, with pure-numpy
fallbacks.
"""

from __future__ import annotations

import math
import html as html_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr, kendalltau
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeatmapConfig:
    """Configuration for heatmap rendering."""
    figsize: Tuple[int, int] = (800, 800)
    color_scheme: str = "diverging"
    annotate: bool = True
    font_size: int = 12
    title: str = "Metric Correlation Heatmap"
    vmin: float = -1.0
    vmax: float = 1.0
    cmap_name: str = "BlueWhiteRed"
    show_dendogram: bool = False
    cluster_method: str = "hierarchical"
    mask_upper_triangle: bool = False
    cell_format: str = ".2f"
    border_width: int = 1


# ---------------------------------------------------------------------------
# Helpers – pure-numpy fallbacks for scipy
# ---------------------------------------------------------------------------

def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Assign ranks to data (average method), pure numpy."""
    arr = np.asarray(arr, dtype=float)
    sorter = np.argsort(arr)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(arr))
    sorted_arr = arr[sorter]
    obs = np.concatenate(([True], sorted_arr[1:] != sorted_arr[:-1]))
    dense = np.cumsum(obs)[inv]
    count = np.concatenate(np.nonzero(np.concatenate((obs, [True]))))
    ranks = np.empty_like(arr)
    for i in range(len(count) - 1):
        lo = count[i]
        hi = count[i + 1]
        avg = (lo + hi - 1) / 2.0 + 1.0
        for j in range(lo, hi):
            ranks[sorter[j]] = avg
    return ranks


def _pearson_pair(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    dx, dy = x - mx, y - my
    denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(dx * dy) / denom)


def _spearman_pair(x: np.ndarray, y: np.ndarray) -> float:
    if HAS_SCIPY:
        rho, _ = spearmanr(x, y)
        return float(rho) if np.isfinite(rho) else 0.0
    return _pearson_pair(_rankdata(x), _rankdata(y))


def _kendall_pair(x: np.ndarray, y: np.ndarray) -> float:
    if HAS_SCIPY:
        tau, _ = kendalltau(x, y)
        return float(tau) if np.isfinite(tau) else 0.0
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _mutual_info_pair(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Estimate mutual information via histogram binning."""
    n = len(x)
    if n < 2:
        return 0.0
    eps = 1e-12
    x_edges = np.linspace(np.min(x) - eps, np.max(x) + eps, n_bins + 1)
    y_edges = np.linspace(np.min(y) - eps, np.max(y) + eps, n_bins + 1)
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1
    x_idx = np.clip(x_idx, 0, n_bins - 1)
    y_idx = np.clip(y_idx, 0, n_bins - 1)
    joint = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(n):
        joint[x_idx[i], y_idx[i]] += 1.0
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]))
    return mi


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


def _interpolate_color_impl(color_a: str, color_b: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(color_a)
    r2, g2, b2 = _hex_to_rgb(color_b)
    t = max(0.0, min(1.0, t))
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return _rgb_to_hex(r, g, b)


def _diverging_cmap(value: float, vmin: float = -1.0, vmax: float = 1.0) -> str:
    """Blue-white-red diverging colormap."""
    if vmax == vmin:
        return "#ffffff"
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    blue = "#2166ac"
    light_blue = "#92c5de"
    white = "#f7f7f7"
    light_red = "#f4a582"
    red = "#b2182b"
    if t < 0.25:
        return _interpolate_color_impl(blue, light_blue, t / 0.25)
    elif t < 0.5:
        return _interpolate_color_impl(light_blue, white, (t - 0.25) / 0.25)
    elif t < 0.75:
        return _interpolate_color_impl(white, light_red, (t - 0.5) / 0.25)
    else:
        return _interpolate_color_impl(light_red, red, (t - 0.75) / 0.25)


def _sequential_cmap(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """White-to-dark-blue sequential colormap."""
    if vmax == vmin:
        return "#ffffff"
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    white = "#f7fbff"
    mid = "#6baed6"
    dark = "#08306b"
    if t < 0.5:
        return _interpolate_color_impl(white, mid, t / 0.5)
    else:
        return _interpolate_color_impl(mid, dark, (t - 0.5) / 0.5)


def _viridis_cmap(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Viridis-like sequential colormap."""
    if vmax == vmin:
        return "#440154"
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    c0 = "#440154"
    c1 = "#31688e"
    c2 = "#35b779"
    c3 = "#fde725"
    if t < 0.333:
        return _interpolate_color_impl(c0, c1, t / 0.333)
    elif t < 0.666:
        return _interpolate_color_impl(c1, c2, (t - 0.333) / 0.333)
    else:
        return _interpolate_color_impl(c2, c3, (t - 0.666) / 0.334)


def _coolwarm_cmap(value: float, vmin: float = -1.0, vmax: float = 1.0) -> str:
    """Coolwarm-like diverging colormap."""
    if vmax == vmin:
        return "#f7f7f7"
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    cool = "#3b4cc0"
    mid = "#f7f7f7"
    warm = "#b40426"
    if t < 0.5:
        return _interpolate_color_impl(cool, mid, t / 0.5)
    else:
        return _interpolate_color_impl(mid, warm, (t - 0.5) / 0.5)


COLORMAPS = {
    "diverging": _diverging_cmap,
    "sequential": _sequential_cmap,
    "viridis": _viridis_cmap,
    "coolwarm": _coolwarm_cmap,
    "BlueWhiteRed": _diverging_cmap,
}


# ---------------------------------------------------------------------------
# Hierarchical clustering (single-linkage, pure numpy)
# ---------------------------------------------------------------------------

def _single_linkage(dist: np.ndarray) -> np.ndarray:
    """
    Single-linkage agglomerative clustering.

    Returns an (n-1, 4) linkage matrix: [idx_a, idx_b, distance, size].
    """
    n = dist.shape[0]
    active = list(range(n))
    sizes = [1] * n
    linkage_rows: List[List[float]] = []
    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    next_id = n

    for _ in range(n - 1):
        best_d = np.inf
        best_i, best_j = 0, 1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                d = work[active[ii], active[jj]]
                if d < best_d:
                    best_d = d
                    best_i, best_j = ii, jj

        a_id = active[best_i]
        b_id = active[best_j]
        new_size = sizes[a_id] + sizes[b_id]
        linkage_rows.append([float(a_id), float(b_id), best_d, float(new_size)])

        new_row = np.full(work.shape[0], np.inf)
        for kk in range(len(active)):
            k_id = active[kk]
            if k_id == a_id or k_id == b_id:
                continue
            new_row[k_id] = min(work[a_id, k_id], work[b_id, k_id])

        expanded = np.full((work.shape[0] + 1, work.shape[1] + 1), np.inf)
        expanded[:work.shape[0], :work.shape[1]] = work
        expanded[next_id, :work.shape[0]] = new_row
        expanded[:work.shape[0], next_id] = new_row
        expanded[next_id, next_id] = np.inf
        work = expanded

        sizes.append(new_size)
        active.append(next_id)
        active.pop(best_j)
        active.pop(best_i)
        next_id += 1

    return np.array(linkage_rows)


def _complete_linkage(dist: np.ndarray) -> np.ndarray:
    """Complete-linkage agglomerative clustering."""
    n = dist.shape[0]
    active = list(range(n))
    sizes = [1] * n
    linkage_rows: List[List[float]] = []
    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    next_id = n

    for _ in range(n - 1):
        best_d = np.inf
        best_i, best_j = 0, 1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                d = work[active[ii], active[jj]]
                if d < best_d:
                    best_d = d
                    best_i, best_j = ii, jj

        a_id = active[best_i]
        b_id = active[best_j]
        new_size = sizes[a_id] + sizes[b_id]
        linkage_rows.append([float(a_id), float(b_id), best_d, float(new_size)])

        new_row = np.full(work.shape[0], np.inf)
        for kk in range(len(active)):
            k_id = active[kk]
            if k_id == a_id or k_id == b_id:
                continue
            new_row[k_id] = max(work[a_id, k_id], work[b_id, k_id])

        expanded = np.full((work.shape[0] + 1, work.shape[1] + 1), np.inf)
        expanded[:work.shape[0], :work.shape[1]] = work
        expanded[next_id, :work.shape[0]] = new_row
        expanded[:work.shape[0], next_id] = new_row
        expanded[next_id, next_id] = np.inf
        work = expanded

        sizes.append(new_size)
        active.append(next_id)
        active.pop(best_j)
        active.pop(best_i)
        next_id += 1

    return np.array(linkage_rows)


def _average_linkage(dist: np.ndarray) -> np.ndarray:
    """Average-linkage (UPGMA) agglomerative clustering."""
    n = dist.shape[0]
    active = list(range(n))
    sizes = [1] * n
    linkage_rows: List[List[float]] = []
    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    next_id = n

    for _ in range(n - 1):
        best_d = np.inf
        best_i, best_j = 0, 1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                d = work[active[ii], active[jj]]
                if d < best_d:
                    best_d = d
                    best_i, best_j = ii, jj

        a_id = active[best_i]
        b_id = active[best_j]
        sa = sizes[a_id]
        sb = sizes[b_id]
        new_size = sa + sb
        linkage_rows.append([float(a_id), float(b_id), best_d, float(new_size)])

        new_row = np.full(work.shape[0], np.inf)
        for kk in range(len(active)):
            k_id = active[kk]
            if k_id == a_id or k_id == b_id:
                continue
            d_ak = work[a_id, k_id] if work[a_id, k_id] < np.inf else 0.0
            d_bk = work[b_id, k_id] if work[b_id, k_id] < np.inf else 0.0
            new_row[k_id] = (sa * d_ak + sb * d_bk) / new_size

        expanded = np.full((work.shape[0] + 1, work.shape[1] + 1), np.inf)
        expanded[:work.shape[0], :work.shape[1]] = work
        expanded[next_id, :work.shape[0]] = new_row
        expanded[:work.shape[0], next_id] = new_row
        expanded[next_id, next_id] = np.inf
        work = expanded

        sizes.append(new_size)
        active.append(next_id)
        active.pop(best_j)
        active.pop(best_i)
        next_id += 1

    return np.array(linkage_rows)


def _leaf_order(linkage: np.ndarray, n: int) -> List[int]:
    """Extract leaf ordering from linkage matrix via iterative DFS."""
    if linkage.shape[0] == 0:
        return list(range(n))
    root = n + linkage.shape[0] - 1
    order: List[int] = []
    stack = [int(root)]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node)
        else:
            row = int(node - n)
            stack.append(int(linkage[row, 1]))
            stack.append(int(linkage[row, 0]))
    return order


# ---------------------------------------------------------------------------
# CorrelationMatrix
# ---------------------------------------------------------------------------

class CorrelationMatrix:
    """Computes and stores various correlation matrices from metric data."""

    def __init__(self, data_dict: Dict[str, List[float]]) -> None:
        self._names = list(data_dict.keys())
        n_metrics = len(self._names)
        lengths = [len(data_dict[k]) for k in self._names]
        if not lengths:
            self._data = np.empty((0, 0))
            return
        n_samples = min(lengths)
        self._data = np.zeros((n_metrics, n_samples))
        for i, name in enumerate(self._names):
            self._data[i, :] = np.array(data_dict[name][:n_samples], dtype=float)

    def compute_pearson(self) -> np.ndarray:
        """Compute Pearson correlation matrix."""
        n = len(self._names)
        if n == 0:
            return np.empty((0, 0))
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                elif j > i:
                    corr[i, j] = _pearson_pair(self._data[i], self._data[j])
                    corr[j, i] = corr[i, j]
        return corr

    def compute_spearman(self) -> np.ndarray:
        """Compute Spearman rank correlation matrix."""
        n = len(self._names)
        if n == 0:
            return np.empty((0, 0))
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                elif j > i:
                    corr[i, j] = _spearman_pair(self._data[i], self._data[j])
                    corr[j, i] = corr[i, j]
        return corr

    def compute_kendall_tau(self) -> np.ndarray:
        """Compute Kendall tau correlation matrix."""
        n = len(self._names)
        if n == 0:
            return np.empty((0, 0))
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                elif j > i:
                    corr[i, j] = _kendall_pair(self._data[i], self._data[j])
                    corr[j, i] = corr[i, j]
        return corr

    def compute_mutual_information(self) -> np.ndarray:
        """Compute pairwise mutual information matrix."""
        n = len(self._names)
        if n == 0:
            return np.empty((0, 0))
        mi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mi[i, j] = _mutual_info_pair(self._data[i], self._data[i])
                elif j > i:
                    mi[i, j] = _mutual_info_pair(self._data[i], self._data[j])
                    mi[j, i] = mi[i, j]
        return mi

    def get_metric_names(self) -> List[str]:
        return list(self._names)

    def cluster_metrics(self, method: str = "hierarchical") -> List[int]:
        """Return reordered metric indices based on clustering."""
        corr = self.compute_pearson()
        n = corr.shape[0]
        if n <= 2:
            return list(range(n))
        dist = np.sqrt(np.clip(1.0 - corr, 0, 2))
        np.fill_diagonal(dist, 0.0)
        if method == "single":
            Z = _single_linkage(dist)
        elif method == "complete":
            Z = _complete_linkage(dist)
        elif method == "average":
            Z = _average_linkage(dist)
        else:
            Z = _average_linkage(dist)
        return _leaf_order(Z, n)

    def find_redundant_pairs(self, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """Find pairs of metrics with absolute correlation above threshold."""
        corr = self.compute_pearson()
        n = corr.shape[0]
        pairs: List[Tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    pairs.append((self._names[i], self._names[j], float(corr[i, j])))
        pairs.sort(key=lambda x: -abs(x[2]))
        return pairs

    def find_orthogonal_pairs(self, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
        """Find pairs of metrics with absolute correlation below threshold."""
        corr = self.compute_pearson()
        n = corr.shape[0]
        pairs: List[Tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) <= threshold:
                    pairs.append((self._names[i], self._names[j], float(corr[i, j])))
        pairs.sort(key=lambda x: abs(x[2]))
        return pairs

    def principal_components(self, n_components: int = 2) -> Dict[str, Any]:
        """Compute PCA on the metric data via SVD."""
        if self._data.shape[0] == 0 or self._data.shape[1] == 0:
            return {"loadings": np.empty((0, 0)), "explained_variance_ratio": np.array([]),
                    "scores": np.empty((0, 0)), "n_components": 0}
        centered = self._data - self._data.mean(axis=1, keepdims=True)
        n_samples = centered.shape[1]
        cov = centered @ centered.T / max(n_samples - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        n_components = min(n_components, len(eigenvalues))
        total_var = np.sum(np.maximum(eigenvalues, 0))
        if total_var < 1e-15:
            explained = np.zeros(n_components)
        else:
            explained = np.maximum(eigenvalues[:n_components], 0) / total_var
        loadings = eigenvectors[:, :n_components]
        scores = loadings.T @ centered
        return {
            "loadings": loadings,
            "explained_variance_ratio": explained,
            "scores": scores,
            "n_components": n_components,
        }

    def to_dataframe_dict(self) -> Dict[str, Any]:
        """Return data and correlations as plain dicts for serialization."""
        pearson = self.compute_pearson()
        spearman = self.compute_spearman()
        return {
            "metric_names": list(self._names),
            "n_metrics": len(self._names),
            "n_samples": self._data.shape[1] if self._data.ndim == 2 else 0,
            "pearson": pearson.tolist() if pearson.size else [],
            "spearman": spearman.tolist() if spearman.size else [],
            "redundant_pairs": [(a, b, float(c)) for a, b, c in self.find_redundant_pairs()],
            "orthogonal_pairs": [(a, b, float(c)) for a, b, c in self.find_orthogonal_pairs()],
            "raw_data": {name: self._data[i].tolist() for i, name in enumerate(self._names)},
        }


# ---------------------------------------------------------------------------
# SVG generation helpers
# ---------------------------------------------------------------------------

_SVG_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
    'font-family="monospace, sans-serif">\n'
    '<rect width="{w}" height="{h}" fill="white"/>\n'
)
_SVG_FOOTER = "</svg>\n"


def _svg_text(
    x: float, y: float, text: str, size: int = 12,
    anchor: str = "middle", color: str = "#333333",
    rotate: float = 0.0, weight: str = "normal",
) -> str:
    escaped = html_module.escape(str(text))
    transform = ""
    if rotate != 0.0:
        transform = f' transform="rotate({rotate},{x},{y})"'
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}"'
        f'{transform}>{escaped}</text>\n'
    )


def _svg_rect(
    x: float, y: float, w: float, h: float,
    fill: str = "#ffffff", stroke: str = "#cccccc",
    stroke_width: int = 1, rx: float = 0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}"/>\n'
    )


def _svg_line(
    x1: float, y1: float, x2: float, y2: float,
    color: str = "#333333", width: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"/>\n'
    )


def _svg_circle(
    cx: float, cy: float, r: float,
    fill: str = "#4a90d9", stroke: str = "none", stroke_width: float = 0,
) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>\n'
    )


def _svg_path(d: str, fill: str = "none", stroke: str = "#333", width: float = 1.0) -> str:
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}"/>\n'


def _svg_group(content: str, transform: str = "") -> str:
    if transform:
        return f'<g transform="{transform}">\n{content}</g>\n'
    return f"<g>\n{content}</g>\n"


# ---------------------------------------------------------------------------
# MetricHeatmapPlotter
# ---------------------------------------------------------------------------

class MetricHeatmapPlotter:
    """Generates SVG visualizations of metric correlations and distributions."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    # ----- public API -----

    def plot_correlation_heatmap(
        self, corr_matrix: np.ndarray, metric_names: List[str],
    ) -> str:
        """Render a correlation heatmap as an SVG string."""
        cfg = self.config
        n = corr_matrix.shape[0]
        if n == 0:
            return self._empty_svg("No data")

        label_margin = max(len(name) for name in metric_names) * cfg.font_size * 0.55 + 10
        label_margin = max(label_margin, 80)
        top_margin = 50 + label_margin
        left_margin = label_margin + 10
        colorbar_width = 60
        cell_size = min(
            (cfg.figsize[0] - left_margin - colorbar_width - 20) / max(n, 1),
            (cfg.figsize[1] - top_margin - 40) / max(n, 1),
            60,
        )
        cell_size = max(cell_size, 16)
        grid_w = cell_size * n
        grid_h = cell_size * n
        total_w = int(left_margin + grid_w + colorbar_width + 40)
        total_h = int(top_margin + grid_h + 40)

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 28, cfg.title, size=cfg.font_size + 4,
            anchor="middle", weight="bold",
        ))

        for i in range(n):
            parts.append(_svg_text(
                left_margin + i * cell_size + cell_size / 2,
                top_margin - 6, metric_names[i], size=max(cfg.font_size - 2, 8),
                anchor="end", rotate=-45,
            ))
            parts.append(_svg_text(
                left_margin - 6,
                top_margin + i * cell_size + cell_size / 2 + 4,
                metric_names[i], size=max(cfg.font_size - 2, 8),
                anchor="end",
            ))

        for i in range(n):
            for j in range(n):
                if cfg.mask_upper_triangle and j > i:
                    continue
                val = corr_matrix[i, j]
                color = self._color_for_value(val, cfg.vmin, cfg.vmax, cfg.color_scheme)
                cx = left_margin + j * cell_size
                cy = top_margin + i * cell_size
                parts.append(self._svg_heatmap_cell(
                    cx, cy, cell_size, cell_size, val, color,
                    f"{val:{cfg.cell_format}}" if cfg.annotate else "",
                ))

        parts.append(self._svg_colorbar(
            left_margin + grid_w + 20, top_margin, grid_h, cfg.vmin, cfg.vmax, cfg.color_scheme,
        ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_clustered_heatmap(
        self, corr_matrix: np.ndarray, metric_names: List[str],
    ) -> str:
        """Render a clustered heatmap with dendrogram as SVG."""
        cfg = self.config
        n = corr_matrix.shape[0]
        if n <= 2:
            return self.plot_correlation_heatmap(corr_matrix, metric_names)

        dist = self._compute_distance_matrix(corr_matrix)
        linkage, order = self._hierarchical_clustering(dist)
        reordered = self._reorder_by_clustering(corr_matrix, order)
        reordered_names = [metric_names[i] for i in order]

        dendro_height = 120
        label_margin = max(len(name) for name in metric_names) * cfg.font_size * 0.5 + 10
        label_margin = max(label_margin, 80)
        top_margin = 50 + dendro_height + label_margin
        left_margin = label_margin + 10 + dendro_height
        colorbar_width = 60
        cell_size = min(
            (cfg.figsize[0] - left_margin - colorbar_width - 20) / max(n, 1),
            (cfg.figsize[1] - top_margin - 40) / max(n, 1),
            55,
        )
        cell_size = max(cell_size, 16)
        grid_w = cell_size * n
        grid_h = cell_size * n
        total_w = int(left_margin + grid_w + colorbar_width + 40)
        total_h = int(top_margin + grid_h + 40)

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 28, cfg.title + " (clustered)", size=cfg.font_size + 4,
            anchor="middle", weight="bold",
        ))

        # top dendrogram
        parts.append(self._svg_dendrogram(
            left_margin, 50, linkage, dendro_height,
            n_leaves=n, cell_size=cell_size, orientation="top",
        ))

        # left dendrogram
        parts.append(self._svg_dendrogram(
            10, top_margin, linkage, dendro_height,
            n_leaves=n, cell_size=cell_size, orientation="left",
        ))

        # labels
        for i in range(n):
            parts.append(_svg_text(
                left_margin + i * cell_size + cell_size / 2,
                top_margin - 6, reordered_names[i],
                size=max(cfg.font_size - 2, 8),
                anchor="end", rotate=-45,
            ))
            parts.append(_svg_text(
                left_margin - 6,
                top_margin + i * cell_size + cell_size / 2 + 4,
                reordered_names[i], size=max(cfg.font_size - 2, 8),
                anchor="end",
            ))

        # cells
        for i in range(n):
            for j in range(n):
                val = reordered[i, j]
                color = self._color_for_value(val, cfg.vmin, cfg.vmax, cfg.color_scheme)
                cx = left_margin + j * cell_size
                cy = top_margin + i * cell_size
                parts.append(self._svg_heatmap_cell(
                    cx, cy, cell_size, cell_size, val, color,
                    f"{val:{cfg.cell_format}}" if cfg.annotate else "",
                ))

        parts.append(self._svg_colorbar(
            left_margin + grid_w + 20, top_margin, grid_h, cfg.vmin, cfg.vmax, cfg.color_scheme,
        ))
        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_metric_scatter_matrix(self, data_dict: Dict[str, List[float]]) -> str:
        """Render a pairwise scatter plot grid as SVG."""
        cfg = self.config
        names = list(data_dict.keys())
        n = len(names)
        if n == 0:
            return self._empty_svg("No metrics")

        arrays = {k: np.array(v, dtype=float) for k, v in data_dict.items()}
        min_len = min(len(a) for a in arrays.values())
        arrays = {k: v[:min_len] for k, v in arrays.items()}

        margin = 60
        panel_size = min(int((cfg.figsize[0] - margin * 2) / max(n, 1)), 140)
        panel_size = max(panel_size, 50)
        total_w = margin * 2 + panel_size * n
        total_h = margin * 2 + panel_size * n + 40

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 24, "Metric Scatter Matrix", size=cfg.font_size + 2,
            anchor="middle", weight="bold",
        ))

        for i in range(n):
            parts.append(_svg_text(
                margin + i * panel_size + panel_size / 2,
                margin - 8, names[i], size=max(cfg.font_size - 3, 7),
                anchor="middle",
            ))
            parts.append(_svg_text(
                margin - 8,
                margin + i * panel_size + panel_size / 2 + 4,
                names[i], size=max(cfg.font_size - 3, 7),
                anchor="end",
            ))

        for row in range(n):
            for col in range(n):
                ox = margin + col * panel_size
                oy = margin + row * panel_size
                parts.append(_svg_rect(ox, oy, panel_size, panel_size,
                                       fill="#fafafa", stroke="#cccccc"))

                y_arr = arrays[names[row]]
                x_arr = arrays[names[col]]

                if row == col:
                    # histogram on diagonal
                    parts.append(self._mini_histogram(
                        x_arr, ox + 4, oy + 4, panel_size - 8, panel_size - 8,
                    ))
                else:
                    # scatter
                    parts.append(self._mini_scatter(
                        x_arr, y_arr, ox + 4, oy + 4,
                        panel_size - 8, panel_size - 8,
                    ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_metric_distributions(self, data_dict: Dict[str, List[float]]) -> str:
        """Render histograms and violin-style outlines for each metric as SVG."""
        cfg = self.config
        names = list(data_dict.keys())
        n = len(names)
        if n == 0:
            return self._empty_svg("No metrics")

        panel_h = 140
        panel_w = max(cfg.figsize[0] - 120, 200)
        total_h = 60 + n * (panel_h + 30)
        total_w = panel_w + 140

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 28, "Metric Distributions",
            size=cfg.font_size + 2, anchor="middle", weight="bold",
        ))

        left = 100
        top = 55
        for idx, name in enumerate(names):
            arr = np.array(data_dict[name], dtype=float)
            arr = arr[np.isfinite(arr)]
            by = top + idx * (panel_h + 30)

            parts.append(_svg_text(
                left - 8, by + panel_h / 2 + 4, name,
                size=cfg.font_size - 1, anchor="end",
            ))
            parts.append(_svg_rect(
                left, by, panel_w, panel_h, fill="#fafafa", stroke="#cccccc",
            ))

            if len(arr) < 2:
                parts.append(_svg_text(
                    left + panel_w / 2, by + panel_h / 2,
                    "insufficient data", size=10, color="#999999",
                ))
                continue

            # histogram
            n_bins = min(30, max(5, int(math.sqrt(len(arr)))))
            hist_counts, bin_edges = np.histogram(arr, bins=n_bins)
            max_count = max(hist_counts.max(), 1)

            hist_h = panel_h * 0.55
            hist_y = by + 8
            bin_w = (panel_w - 16) / n_bins
            for bi in range(n_bins):
                bar_h = (hist_counts[bi] / max_count) * hist_h
                bx = left + 8 + bi * bin_w
                bar_y = hist_y + hist_h - bar_h
                color = _sequential_cmap(bi / max(n_bins - 1, 1), 0.0, 1.0)
                parts.append(_svg_rect(bx, bar_y, bin_w - 1, bar_h, fill=color, stroke="none"))

            # violin outline (KDE approximation)
            violin_y = by + panel_h * 0.65
            violin_h = panel_h * 0.3
            parts.append(self._mini_violin(
                arr, left + 8, violin_y, panel_w - 16, violin_h,
            ))

            # summary stats
            mu = np.mean(arr)
            sigma = np.std(arr)
            median = np.median(arr)
            parts.append(_svg_text(
                left + panel_w + 6, by + 18,
                f"μ={mu:.3f}", size=9, anchor="start", color="#555",
            ))
            parts.append(_svg_text(
                left + panel_w + 6, by + 32,
                f"σ={sigma:.3f}", size=9, anchor="start", color="#555",
            ))
            parts.append(_svg_text(
                left + panel_w + 6, by + 46,
                f"med={median:.3f}", size=9, anchor="start", color="#555",
            ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_metric_rankings(self, rankings_dict: Dict[str, List[float]]) -> str:
        """Render rank comparison (bump chart style) as SVG."""
        cfg = self.config
        names = list(rankings_dict.keys())
        n = len(names)
        if n == 0:
            return self._empty_svg("No rankings")

        arrays = {k: np.array(v, dtype=float) for k, v in rankings_dict.items()}
        min_len = min(len(a) for a in arrays.values())
        arrays = {k: v[:min_len] for k, v in arrays.items()}
        n_items = min_len

        margin_l = 120
        margin_r = 40
        margin_t = 60
        margin_b = 40
        plot_w = max(cfg.figsize[0] - margin_l - margin_r, 200)
        plot_h = max(cfg.figsize[1] - margin_t - margin_b, 200)
        total_w = margin_l + plot_w + margin_r
        total_h = margin_t + plot_h + margin_b

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 28, "Metric Rankings Comparison",
            size=cfg.font_size + 2, anchor="middle", weight="bold",
        ))

        parts.append(_svg_rect(margin_l, margin_t, plot_w, plot_h, fill="#fafafa", stroke="#cccccc"))

        colors = self._palette(n)
        all_vals = np.concatenate([a for a in arrays.values()])
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        if y_max == y_min:
            y_max = y_min + 1

        for idx, name in enumerate(names):
            arr = arrays[name]
            color = colors[idx]

            points: List[Tuple[float, float]] = []
            for t in range(n_items):
                px = margin_l + t / max(n_items - 1, 1) * plot_w
                py = margin_t + plot_h - (arr[t] - y_min) / (y_max - y_min) * plot_h
                points.append((px, py))
                parts.append(_svg_circle(px, py, 3, fill=color))

            if len(points) > 1:
                d_parts = [f"M{points[0][0]:.1f},{points[0][1]:.1f}"]
                for p in points[1:]:
                    d_parts.append(f"L{p[0]:.1f},{p[1]:.1f}")
                parts.append(_svg_path(" ".join(d_parts), stroke=color, width=1.5))

            parts.append(_svg_text(
                margin_l - 8, points[0][1] + 4 if points else margin_t + idx * 16,
                name, size=cfg.font_size - 2, anchor="end", color=color,
            ))

        # x-axis tick labels
        for t in range(min(n_items, 10)):
            step = max(1, n_items // 10)
            ti = t * step
            if ti >= n_items:
                break
            px = margin_l + ti / max(n_items - 1, 1) * plot_w
            parts.append(_svg_text(
                px, margin_t + plot_h + 18, str(ti + 1),
                size=cfg.font_size - 3, anchor="middle", color="#666",
            ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_redundancy_graph(
        self, corr_matrix: np.ndarray, threshold: float = 0.8,
    ) -> str:
        """Render a network graph of redundant metric pairs as SVG."""
        cfg = self.config
        n = corr_matrix.shape[0]
        if n == 0:
            return self._empty_svg("No data")

        total_w = cfg.figsize[0]
        total_h = cfg.figsize[1]
        cx_center = total_w / 2
        cy_center = total_h / 2
        radius = min(total_w, total_h) * 0.35

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 28, f"Redundancy Graph (|r| ≥ {threshold:.2f})",
            size=cfg.font_size + 2, anchor="middle", weight="bold",
        ))

        node_positions: List[Tuple[float, float]] = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            px = cx_center + radius * math.cos(angle)
            py = cy_center + radius * math.sin(angle)
            node_positions.append((px, py))

        # edges
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(corr_matrix[i, j])
                if val >= threshold:
                    opacity = 0.3 + 0.7 * (val - threshold) / max(1.0 - threshold, 0.01)
                    width = 1.0 + 4.0 * (val - threshold) / max(1.0 - threshold, 0.01)
                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]
                    color = "#e74c3c" if corr_matrix[i, j] > 0 else "#3498db"
                    parts.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{color}" stroke-width="{width:.1f}" '
                        f'stroke-opacity="{opacity:.2f}"/>\n'
                    )
                    mx = (x1 + x2) / 2
                    my = (y1 + y2) / 2
                    parts.append(_svg_text(
                        mx, my, f"{corr_matrix[i, j]:.2f}",
                        size=max(cfg.font_size - 4, 7), color="#555",
                    ))

        # nodes
        for i in range(n):
            px, py = node_positions[i]
            degree = sum(1 for j in range(n) if j != i and abs(corr_matrix[i, j]) >= threshold)
            node_r = 12 + degree * 3
            node_color = _sequential_cmap(degree / max(n - 1, 1), 0.0, 1.0)
            parts.append(_svg_circle(px, py, node_r, fill=node_color, stroke="#333", stroke_width=1.5))

        # labels (placed slightly outside the circle)
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            lx = cx_center + (radius + 30) * math.cos(angle)
            ly = cy_center + (radius + 30) * math.sin(angle)
            anchor = "start" if math.cos(angle) > 0.1 else ("end" if math.cos(angle) < -0.1 else "middle")
            parts.append(_svg_text(
                lx, ly + 4, f"M{i}", size=cfg.font_size - 1, anchor=anchor,
            ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_pca_loadings(
        self, loadings: np.ndarray, metric_names: List[str],
    ) -> str:
        """Render a PCA loading biplot as SVG."""
        cfg = self.config
        n = loadings.shape[0]
        if n == 0 or loadings.shape[1] < 2:
            return self._empty_svg("Need ≥2 components")

        total_w = cfg.figsize[0]
        total_h = cfg.figsize[1]
        margin = 80
        plot_w = total_w - margin * 2
        plot_h = total_h - margin * 2

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(
            total_w / 2, 28, "PCA Loadings Biplot",
            size=cfg.font_size + 2, anchor="middle", weight="bold",
        ))

        cx = margin + plot_w / 2
        cy = margin + plot_h / 2

        # axes
        parts.append(_svg_line(margin, cy, margin + plot_w, cy, color="#cccccc"))
        parts.append(_svg_line(cx, margin, cx, margin + plot_h, color="#cccccc"))
        parts.append(_svg_text(
            margin + plot_w + 4, cy + 4, "PC1",
            size=cfg.font_size - 1, anchor="start", color="#888",
        ))
        parts.append(_svg_text(
            cx + 4, margin - 6, "PC2",
            size=cfg.font_size - 1, anchor="start", color="#888",
        ))

        max_abs = max(float(np.max(np.abs(loadings[:, :2]))), 0.01)
        scale = min(plot_w, plot_h) / 2 / max_abs * 0.85

        colors = self._palette(n)
        for i in range(n):
            lx = loadings[i, 0] * scale
            ly = -loadings[i, 1] * scale
            ex = cx + lx
            ey = cy + ly
            parts.append(_svg_line(cx, cy, ex, ey, color=colors[i], width=2.0))

            # arrowhead
            arrow_len = 8
            angle = math.atan2(ly, lx)
            ax1 = ex - arrow_len * math.cos(angle - 0.3)
            ay1 = ey - arrow_len * math.sin(angle - 0.3)
            ax2 = ex - arrow_len * math.cos(angle + 0.3)
            ay2 = ey - arrow_len * math.sin(angle + 0.3)
            parts.append(_svg_path(
                f"M{ex:.1f},{ey:.1f} L{ax1:.1f},{ay1:.1f} L{ax2:.1f},{ay2:.1f} Z",
                fill=colors[i], stroke="none",
            ))

            label_x = ex + 8 * math.cos(angle)
            label_y = ey + 8 * math.sin(angle) + 4
            anchor = "start" if lx >= 0 else "end"
            parts.append(_svg_text(
                label_x, label_y, metric_names[i],
                size=cfg.font_size - 2, anchor=anchor, color=colors[i],
            ))

        # unit circle outline
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{scale:.1f}" '
            f'fill="none" stroke="#dddddd" stroke-width="1" stroke-dasharray="4,4"/>\n'
        )

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def generate_correlation_report(self, data_dict: Dict[str, List[float]]) -> str:
        """Generate a full HTML report with multiple embedded SVG visualizations."""
        cm = CorrelationMatrix(data_dict)
        names = cm.get_metric_names()
        pearson = cm.compute_pearson()
        spearman = cm.compute_spearman()
        redundant = cm.find_redundant_pairs()
        orthogonal = cm.find_orthogonal_pairs()
        pca = cm.principal_components(n_components=min(len(names), 3))

        heatmap_svg = self.plot_correlation_heatmap(pearson, names)
        clustered_svg = self.plot_clustered_heatmap(pearson, names)
        scatter_svg = self.plot_metric_scatter_matrix(data_dict)
        dist_svg = self.plot_metric_distributions(data_dict)
        redundancy_svg = self.plot_redundancy_graph(pearson, threshold=0.8)

        loadings = pca.get("loadings", np.empty((0, 0)))
        if loadings.size and loadings.shape[1] >= 2:
            pca_svg = self.plot_pca_loadings(loadings, names)
        else:
            pca_svg = ""

        explained = pca.get("explained_variance_ratio", np.array([]))

        redundant_rows = "".join(
            f"<tr><td>{a}</td><td>{b}</td><td>{c:.4f}</td></tr>\n"
            for a, b, c in redundant[:20]
        )
        orthogonal_rows = "".join(
            f"<tr><td>{a}</td><td>{b}</td><td>{c:.4f}</td></tr>\n"
            for a, b, c in orthogonal[:20]
        )

        spearman_table_rows = ""
        if spearman.size:
            for i in range(spearman.shape[0]):
                cells = "".join(f"<td>{spearman[i, j]:.3f}</td>" for j in range(spearman.shape[1]))
                spearman_table_rows += f"<tr><th>{names[i]}</th>{cells}</tr>\n"

        explained_text = ""
        if explained.size:
            for i, ev in enumerate(explained):
                explained_text += f"<li>PC{i + 1}: {ev * 100:.1f}%</li>\n"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Metric Correlation Report</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f9f9f9; color: #333; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
h2 {{ color: #2c3e50; margin-top: 32px; }}
h3 {{ color: #555; }}
.section {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
table {{ border-collapse: collapse; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f0f4f8; text-align: left; }}
.svg-container {{ overflow-x: auto; margin: 12px 0; }}
ul {{ line-height: 1.8; }}
.summary {{ background: #eef6fc; border-left: 4px solid #3498db; padding: 12px 16px; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Metric Correlation Analysis Report</h1>
<div class="summary">
<p><strong>{len(names)}</strong> metrics analysed &middot;
<strong>{len(redundant)}</strong> redundant pairs (|r| &ge; 0.9) &middot;
<strong>{len(orthogonal)}</strong> near-orthogonal pairs (|r| &le; 0.1)</p>
</div>

<div class="section">
<h2>1. Pearson Correlation Heatmap</h2>
<div class="svg-container">{heatmap_svg}</div>
</div>

<div class="section">
<h2>2. Clustered Heatmap</h2>
<div class="svg-container">{clustered_svg}</div>
</div>

<div class="section">
<h2>3. Scatter Matrix</h2>
<div class="svg-container">{scatter_svg}</div>
</div>

<div class="section">
<h2>4. Metric Distributions</h2>
<div class="svg-container">{dist_svg}</div>
</div>

<div class="section">
<h2>5. Redundancy Network</h2>
<div class="svg-container">{redundancy_svg}</div>
</div>

{"<div class='section'><h2>6. PCA Loadings</h2><div class='svg-container'>" + pca_svg + "</div>" + "<h3>Explained Variance</h3><ul>" + explained_text + "</ul></div>" if pca_svg else ""}

<div class="section">
<h2>Spearman Rank Correlation</h2>
<table>
<tr><th></th>{"".join(f"<th>{n}</th>" for n in names)}</tr>
{spearman_table_rows}
</table>
</div>

<div class="section">
<h2>Redundant Metric Pairs (|r| &ge; 0.9)</h2>
{"<table><tr><th>Metric A</th><th>Metric B</th><th>Correlation</th></tr>" + redundant_rows + "</table>" if redundant_rows else "<p>No redundant pairs found.</p>"}
</div>

<div class="section">
<h2>Near-Orthogonal Pairs (|r| &le; 0.1)</h2>
{"<table><tr><th>Metric A</th><th>Metric B</th><th>Correlation</th></tr>" + orthogonal_rows + "</table>" if orthogonal_rows else "<p>No near-orthogonal pairs found.</p>"}
</div>

</body>
</html>
"""
        return html

    # ----- colour helpers exposed as methods -----

    def _color_for_value(
        self, value: float, vmin: float, vmax: float, colorscheme: str,
    ) -> str:
        cmap_fn = COLORMAPS.get(colorscheme, _diverging_cmap)
        return cmap_fn(value, vmin, vmax)

    def _interpolate_color(self, color_a: str, color_b: str, t: float) -> str:
        return _interpolate_color_impl(color_a, color_b, t)

    def _diverging_colormap(self, value: float) -> str:
        return _diverging_cmap(value, self.config.vmin, self.config.vmax)

    def _sequential_colormap(self, value: float) -> str:
        return _sequential_cmap(value, 0.0, 1.0)

    # ----- clustering helpers -----

    def _hierarchical_clustering(
        self, distance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """Run hierarchical clustering; return (linkage, leaf_order)."""
        n = distance_matrix.shape[0]
        method = self.config.cluster_method
        if method == "single":
            Z = _single_linkage(distance_matrix)
        elif method == "complete":
            Z = _complete_linkage(distance_matrix)
        else:
            Z = _average_linkage(distance_matrix)
        order = _leaf_order(Z, n)
        return Z, order

    def _dendrogram_coordinates(
        self, linkage: np.ndarray, n_leaves: int, cell_size: float, height: float,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Compute dendrogram line segments as (start, end) pairs."""
        if linkage.shape[0] == 0:
            return []
        positions: Dict[int, float] = {}
        for i in range(n_leaves):
            positions[i] = (i + 0.5) * cell_size

        max_dist = float(linkage[:, 2].max()) if linkage[:, 2].max() > 0 else 1.0
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        for row_idx in range(linkage.shape[0]):
            a = int(linkage[row_idx, 0])
            b = int(linkage[row_idx, 1])
            dist = float(linkage[row_idx, 2])
            new_id = n_leaves + row_idx

            xa = positions.get(a, 0.0)
            xb = positions.get(b, 0.0)
            y_merge = height * (1.0 - dist / max_dist)

            y_a = height if a < n_leaves else height * (1.0 - float(linkage[a - n_leaves, 2]) / max_dist)
            y_b = height if b < n_leaves else height * (1.0 - float(linkage[b - n_leaves, 2]) / max_dist)

            segments.append(((xa, y_a), (xa, y_merge)))
            segments.append(((xb, y_b), (xb, y_merge)))
            segments.append(((xa, y_merge), (xb, y_merge)))

            positions[new_id] = (xa + xb) / 2

        return segments

    def _svg_heatmap_cell(
        self, x: float, y: float, w: float, h: float,
        value: float, color: str, label: str,
    ) -> str:
        out = _svg_rect(x, y, w, h, fill=color, stroke="#ffffff",
                        stroke_width=self.config.border_width)
        if label:
            brightness = sum(_hex_to_rgb(color)) / 3
            text_color = "#ffffff" if brightness < 140 else "#333333"
            font_size = min(self.config.font_size - 2, int(min(w, h) * 0.35))
            font_size = max(font_size, 7)
            out += _svg_text(
                x + w / 2, y + h / 2 + font_size * 0.35,
                label, size=font_size, color=text_color,
            )
        return out

    def _svg_colorbar(
        self, x: float, y: float, height: float,
        vmin: float, vmax: float, colorscheme: str,
    ) -> str:
        n_steps = 50
        bar_w = 18
        step_h = height / n_steps
        parts: List[str] = []
        for i in range(n_steps):
            val = vmax - (vmax - vmin) * i / (n_steps - 1)
            color = self._color_for_value(val, vmin, vmax, colorscheme)
            parts.append(_svg_rect(
                x, y + i * step_h, bar_w, step_h + 0.5,
                fill=color, stroke="none",
            ))
        parts.append(_svg_rect(x, y, bar_w, height, fill="none", stroke="#999999"))

        n_ticks = 5
        for t in range(n_ticks):
            val = vmax - (vmax - vmin) * t / (n_ticks - 1)
            ty = y + t / (n_ticks - 1) * height
            parts.append(_svg_text(
                x + bar_w + 5, ty + 4, f"{val:.2f}",
                size=self.config.font_size - 3, anchor="start", color="#555",
            ))
            parts.append(_svg_line(x + bar_w, ty, x + bar_w + 3, ty, color="#999"))

        return "".join(parts)

    def _svg_dendrogram(
        self, x: float, y: float, linkage: np.ndarray, height: float,
        n_leaves: int = 0, cell_size: float = 30, orientation: str = "top",
    ) -> str:
        segments = self._dendrogram_coordinates(linkage, n_leaves, cell_size, height)
        parts: List[str] = []
        for (x1, y1), (x2, y2) in segments:
            if orientation == "top":
                parts.append(_svg_line(
                    x + x1, y + y1, x + x2, y + y2, color="#555555", width=1.0,
                ))
            elif orientation == "left":
                parts.append(_svg_line(
                    x + y1, y + x1, x + y2, y + x2, color="#555555", width=1.0,
                ))
        return "".join(parts)

    def _compute_distance_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        dist = np.sqrt(np.clip(1.0 - corr_matrix, 0, 2.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    def _reorder_by_clustering(
        self, matrix: np.ndarray, order: List[int],
    ) -> np.ndarray:
        idx = np.array(order)
        return matrix[np.ix_(idx, idx)]

    # ----- internal drawing helpers -----

    def _empty_svg(self, message: str) -> str:
        return (
            _SVG_HEADER.format(w=300, h=80)
            + _svg_text(150, 45, message, size=14, color="#888888")
            + _SVG_FOOTER
        )

    def _palette(self, n: int) -> List[str]:
        base = [
            "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
            "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
            "#000075", "#a9a9a9",
        ]
        if n <= len(base):
            return base[:n]
        result = list(base)
        while len(result) < n:
            hue = len(result) * 137.508
            r = int(128 + 127 * math.cos(math.radians(hue)))
            g = int(128 + 127 * math.cos(math.radians(hue + 120)))
            b = int(128 + 127 * math.cos(math.radians(hue + 240)))
            result.append(_rgb_to_hex(r, g, b))
        return result[:n]

    def _mini_scatter(
        self, x: np.ndarray, y: np.ndarray,
        ox: float, oy: float, w: float, h: float,
    ) -> str:
        if len(x) < 2:
            return ""
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1
        parts: List[str] = []
        max_points = min(len(x), 200)
        step = max(1, len(x) // max_points)
        for i in range(0, len(x), step):
            px = ox + (x[i] - x_min) / (x_max - x_min) * w
            py = oy + h - (y[i] - y_min) / (y_max - y_min) * h
            parts.append(_svg_circle(px, py, 1.8, fill="#4a90d9"))
        return "".join(parts)

    def _mini_histogram(
        self, arr: np.ndarray, ox: float, oy: float, w: float, h: float,
        n_bins: int = 15,
    ) -> str:
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return ""
        counts, edges = np.histogram(arr, bins=n_bins)
        max_c = max(counts.max(), 1)
        bin_w = w / n_bins
        parts: List[str] = []
        for i in range(n_bins):
            bh = (counts[i] / max_c) * h
            bx = ox + i * bin_w
            by = oy + h - bh
            parts.append(_svg_rect(bx, by, bin_w - 0.5, bh, fill="#6baed6", stroke="none"))
        return "".join(parts)

    def _mini_violin(
        self, arr: np.ndarray, ox: float, oy: float, w: float, h: float,
    ) -> str:
        """Approximate violin plot via smoothed histogram mirrored around centre."""
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            return ""
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if a_max == a_min:
            return ""

        n_points = 40
        bandwidth = (a_max - a_min) / 15
        if bandwidth < 1e-10:
            bandwidth = 1.0
        xs = np.linspace(a_min, a_max, n_points)
        density = np.zeros(n_points)
        for val in arr:
            density += np.exp(-0.5 * ((xs - val) / bandwidth) ** 2)
        density /= len(arr) * bandwidth * math.sqrt(2 * math.pi)
        max_d = float(density.max())
        if max_d < 1e-12:
            max_d = 1.0

        mid_y = oy + h / 2
        parts: List[str] = []

        # upper half
        top_points = []
        for i in range(n_points):
            px = ox + (xs[i] - a_min) / (a_max - a_min) * w
            py = mid_y - (density[i] / max_d) * (h / 2) * 0.9
            top_points.append(f"{px:.1f},{py:.1f}")

        # lower half (mirrored)
        bottom_points = []
        for i in range(n_points - 1, -1, -1):
            px = ox + (xs[i] - a_min) / (a_max - a_min) * w
            py = mid_y + (density[i] / max_d) * (h / 2) * 0.9
            bottom_points.append(f"{px:.1f},{py:.1f}")

        all_pts = top_points + bottom_points
        d = "M" + " L".join(all_pts) + " Z"
        parts.append(_svg_path(d, fill="#b3cde3", stroke="#6baed6", width=1.0))

        # median line
        med = float(np.median(arr))
        mx = ox + (med - a_min) / (a_max - a_min) * w
        parts.append(_svg_line(mx, mid_y - h * 0.35, mx, mid_y + h * 0.35,
                               color="#d6604d", width=1.5))

        # quartile markers
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        q1x = ox + (q1 - a_min) / (a_max - a_min) * w
        q3x = ox + (q3 - a_min) / (a_max - a_min) * w
        parts.append(_svg_line(q1x, mid_y - h * 0.2, q1x, mid_y + h * 0.2,
                               color="#999", width=1.0))
        parts.append(_svg_line(q3x, mid_y - h * 0.2, q3x, mid_y + h * 0.2,
                               color="#999", width=1.0))

        return "".join(parts)


# ---------------------------------------------------------------------------
# Extended heatmap visualizations (hierarchical, stability, cross-domain)
# ---------------------------------------------------------------------------


class HierarchicalClusterHeatmap:
    """Hierarchical clustering heatmaps with full dendrogram rendering."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    @staticmethod
    def _linkage_single(dist: np.ndarray) -> List[Tuple[int, int, float, int]]:
        """Single-linkage agglomerative clustering on a distance matrix.

        Returns list of (cluster_i, cluster_j, distance, size) tuples.
        """
        n = dist.shape[0]
        active = list(range(n))
        sizes = [1] * n
        merges: List[Tuple[int, int, float, int]] = []
        next_id = n
        d = dist.copy()
        np.fill_diagonal(d, np.inf)

        for _ in range(n - 1):
            # find closest pair among active clusters
            min_val = np.inf
            mi, mj = 0, 1
            for ii, a in enumerate(active):
                for jj, b in enumerate(active):
                    if ii < jj and d[a, b] < min_val:
                        min_val = d[a, b]
                        mi, mj = a, b
            new_size = sizes[mi] + sizes[mj]
            merges.append((mi, mj, float(min_val), new_size))

            # create new row/col in distance matrix
            new_d = np.full((d.shape[0] + 1, d.shape[1] + 1), np.inf)
            new_d[:d.shape[0], :d.shape[1]] = d
            for a in active:
                if a != mi and a != mj:
                    val = min(d[a, mi], d[a, mj])
                    new_d[a, next_id] = val
                    new_d[next_id, a] = val
            d = new_d

            active.remove(mi)
            active.remove(mj)
            sizes.append(new_size)
            active.append(next_id)
            next_id += 1
        return merges

    @staticmethod
    def _leaf_order_from_linkage(
        merges: List[Tuple[int, int, float, int]], n: int,
    ) -> List[int]:
        """Extract leaf ordering from linkage results."""
        children: Dict[int, List[int]] = {}
        for i, (a, b, _, _) in enumerate(merges):
            cid = n + i
            children[cid] = [a, b]

        def _leaves(node: int) -> List[int]:
            if node < n:
                return [node]
            return _leaves(children[node][0]) + _leaves(children[node][1])

        root = n + len(merges) - 1
        return _leaves(root)

    def plot_hierarchical_heatmap(
        self,
        data_matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Hierarchical Clustering Heatmap",
    ) -> str:
        """Render a heatmap with hierarchical clustering applied to rows and columns."""
        cfg = self.config
        n_rows, n_cols = data_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return _empty_svg_global("No data for hierarchical heatmap")

        # cluster rows
        row_dist = np.zeros((n_rows, n_rows))
        for i in range(n_rows):
            for j in range(n_rows):
                row_dist[i, j] = np.sqrt(np.sum((data_matrix[i] - data_matrix[j]) ** 2))

        row_merges = self._linkage_single(row_dist)
        row_order = self._leaf_order_from_linkage(row_merges, n_rows)

        # cluster cols
        col_dist = np.zeros((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(n_cols):
                col_dist[i, j] = np.sqrt(
                    np.sum((data_matrix[:, i] - data_matrix[:, j]) ** 2)
                )
        col_merges = self._linkage_single(col_dist)
        col_order = self._leaf_order_from_linkage(col_merges, n_cols)

        reordered = data_matrix[np.ix_(row_order, col_order)]
        reordered_row_labels = [row_labels[i] for i in row_order]
        reordered_col_labels = [col_labels[i] for i in col_order]

        dendro_h = 80
        label_margin = 100
        cell_size = min(30, (cfg.figsize[0] - label_margin - 80) / max(n_cols, 1))
        cell_size = max(cell_size, 14)
        total_w = int(label_margin + cell_size * n_cols + 80)
        total_h = int(dendro_h + label_margin + cell_size * n_rows + 40)

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(total_w / 2, 22, title, size=cfg.font_size + 2,
                               anchor="middle", weight="bold"))

        top_offset = dendro_h + label_margin
        # draw cells
        vmin = float(np.nanmin(reordered))
        vmax = float(np.nanmax(reordered))
        rng = vmax - vmin if vmax != vmin else 1.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = reordered[i, j]
                t = (val - vmin) / rng
                r = int(255 * (1 - t))
                b = int(255 * t)
                color = f"#{r:02x}80{b:02x}"
                cx = label_margin + j * cell_size
                cy = top_offset + i * cell_size
                parts.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{color}" stroke="#ccc" '
                    f'stroke-width="0.5"/>'
                )
                if cell_size >= 20 and cfg.annotate:
                    parts.append(_svg_text(
                        cx + cell_size / 2, cy + cell_size / 2 + 4,
                        f"{val:.2f}", size=max(7, cfg.font_size - 4), anchor="middle",
                    ))

        # row labels
        for i, lbl in enumerate(reordered_row_labels):
            parts.append(_svg_text(
                label_margin - 4, top_offset + i * cell_size + cell_size / 2 + 4,
                lbl, size=max(8, cfg.font_size - 2), anchor="end",
            ))
        # col labels
        for j, lbl in enumerate(reordered_col_labels):
            parts.append(_svg_text(
                label_margin + j * cell_size + cell_size / 2,
                top_offset - 6, lbl, size=max(8, cfg.font_size - 2),
                anchor="end", rotate=-45,
            ))

        # dendrogram lines (simplified)
        max_dist = max((m[2] for m in col_merges), default=1.0)
        if max_dist == 0:
            max_dist = 1.0
        positions: Dict[int, float] = {}
        for j, idx in enumerate(col_order):
            positions[idx] = label_margin + j * cell_size + cell_size / 2
        for merge_idx, (a, b, dist, _) in enumerate(col_merges):
            cid = n_cols + merge_idx
            xa = positions.get(a, label_margin)
            xb = positions.get(b, label_margin)
            y_merge = dendro_h + label_margin - 10 - (dist / max_dist) * (dendro_h - 20)
            y_a = dendro_h + label_margin - 10
            y_b = dendro_h + label_margin - 10
            if a >= n_cols:
                row_m = col_merges[a - n_cols]
                y_a = dendro_h + label_margin - 10 - (row_m[2] / max_dist) * (dendro_h - 20)
            if b >= n_cols:
                row_m = col_merges[b - n_cols]
                y_b = dendro_h + label_margin - 10 - (row_m[2] / max_dist) * (dendro_h - 20)
            parts.append(
                f'<line x1="{xa}" y1="{y_a}" x2="{xa}" y2="{y_merge}" '
                f'stroke="#333" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{xb}" y1="{y_b}" x2="{xb}" y2="{y_merge}" '
                f'stroke="#333" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{xa}" y1="{y_merge}" x2="{xb}" y2="{y_merge}" '
                f'stroke="#333" stroke-width="1.5"/>'
            )
            positions[cid] = (xa + xb) / 2

        parts.append(_SVG_FOOTER)
        return "".join(parts)


class MetricStabilityHeatmap:
    """Heatmaps showing metric stability across bootstrap samples."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    @staticmethod
    def _bootstrap_correlations(
        data_dict: Dict[str, List[float]],
        n_bootstrap: int = 100,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std of pairwise correlations over bootstrap samples."""
        rng = np.random.RandomState(seed)
        names = sorted(data_dict.keys())
        n_metrics = len(names)
        n_obs = len(data_dict[names[0]])
        corr_samples = np.zeros((n_bootstrap, n_metrics, n_metrics))

        for b in range(n_bootstrap):
            idx = rng.randint(0, n_obs, size=n_obs)
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    xi = np.array(data_dict[ni])[idx]
                    xj = np.array(data_dict[nj])[idx]
                    if np.std(xi) == 0 or np.std(xj) == 0:
                        corr_samples[b, i, j] = 0.0
                    else:
                        corr_samples[b, i, j] = float(np.corrcoef(xi, xj)[0, 1])

        return np.mean(corr_samples, axis=0), np.std(corr_samples, axis=0)

    def plot_stability_heatmap(
        self,
        data_dict: Dict[str, List[float]],
        n_bootstrap: int = 100,
        title: str = "Metric Correlation Stability (Bootstrap Std)",
    ) -> str:
        """Generate a heatmap of bootstrap standard deviations of correlations."""
        cfg = self.config
        names = sorted(data_dict.keys())
        n = len(names)
        if n == 0:
            return _empty_svg_global("No metrics")

        _, std_matrix = self._bootstrap_correlations(data_dict, n_bootstrap)

        label_margin = 100
        cell_size = min(40, (cfg.figsize[0] - label_margin - 80) / max(n, 1))
        cell_size = max(cell_size, 18)
        total_w = int(label_margin + cell_size * n + 100)
        total_h = int(label_margin + cell_size * n + 60)

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(total_w / 2, 24, title, size=cfg.font_size + 2,
                               anchor="middle", weight="bold"))

        max_std = float(np.max(std_matrix)) if np.max(std_matrix) > 0 else 1.0
        top_off = label_margin
        for i in range(n):
            for j in range(n):
                val = std_matrix[i, j]
                t = val / max_std
                # Yellow-to-red scale: low std = green, high = red
                r = int(255)
                g = int(255 * (1 - t))
                color = f"#{r:02x}{g:02x}30"
                cx = label_margin + j * cell_size
                cy = top_off + i * cell_size
                parts.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{color}" stroke="#ddd" '
                    f'stroke-width="0.5"/>'
                )
                if cell_size >= 22:
                    parts.append(_svg_text(
                        cx + cell_size / 2, cy + cell_size / 2 + 4,
                        f"{val:.3f}", size=max(7, cfg.font_size - 4), anchor="middle",
                    ))

        for i, nm in enumerate(names):
            parts.append(_svg_text(
                label_margin - 4, top_off + i * cell_size + cell_size / 2 + 4,
                nm, size=max(8, cfg.font_size - 2), anchor="end",
            ))
            parts.append(_svg_text(
                label_margin + i * cell_size + cell_size / 2,
                top_off - 6, nm, size=max(8, cfg.font_size - 2),
                anchor="end", rotate=-45,
            ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)


class CrossDomainCorrelationHeatmap:
    """Cross-domain metric correlation heatmaps comparing metric families."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    def plot_cross_domain_heatmap(
        self,
        domain_a: Dict[str, List[float]],
        domain_b: Dict[str, List[float]],
        domain_a_name: str = "Domain A",
        domain_b_name: str = "Domain B",
        title: str = "Cross-Domain Metric Correlations",
    ) -> str:
        """Render a rectangular heatmap: rows = domain_a metrics, cols = domain_b."""
        cfg = self.config
        names_a = sorted(domain_a.keys())
        names_b = sorted(domain_b.keys())
        na, nb = len(names_a), len(names_b)
        if na == 0 or nb == 0:
            return _empty_svg_global("Empty domain")

        corr = np.zeros((na, nb))
        for i, ka in enumerate(names_a):
            for j, kb in enumerate(names_b):
                xa = np.array(domain_a[ka], dtype=float)
                xb = np.array(domain_b[kb], dtype=float)
                min_len = min(len(xa), len(xb))
                xa, xb = xa[:min_len], xb[:min_len]
                if np.std(xa) == 0 or np.std(xb) == 0:
                    corr[i, j] = 0.0
                else:
                    corr[i, j] = float(np.corrcoef(xa, xb)[0, 1])

        label_margin_left = 120
        label_margin_top = 100
        cell_size = min(40, (cfg.figsize[0] - label_margin_left - 80) / max(nb, 1))
        cell_size = max(cell_size, 18)
        total_w = int(label_margin_left + cell_size * nb + 100)
        total_h = int(label_margin_top + cell_size * na + 60)

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(total_w / 2, 22, title, size=cfg.font_size + 2,
                               anchor="middle", weight="bold"))
        # domain labels
        parts.append(_svg_text(
            total_w / 2, total_h - 10,
            domain_b_name, size=cfg.font_size, anchor="middle",
        ))
        parts.append(_svg_text(
            12, total_h / 2,
            domain_a_name, size=cfg.font_size, anchor="middle", rotate=-90,
        ))

        for i in range(na):
            for j in range(nb):
                val = corr[i, j]
                t = (val + 1) / 2  # map [-1,1] to [0,1]
                r = int(255 * (1 - t))
                b_c = int(255 * t)
                color = f"#{r:02x}80{b_c:02x}"
                cx = label_margin_left + j * cell_size
                cy = label_margin_top + i * cell_size
                parts.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{color}" stroke="#ccc" '
                    f'stroke-width="0.5"/>'
                )
                if cell_size >= 22:
                    parts.append(_svg_text(
                        cx + cell_size / 2, cy + cell_size / 2 + 4,
                        f"{val:.2f}", size=max(7, cfg.font_size - 4), anchor="middle",
                    ))

        for i, nm in enumerate(names_a):
            parts.append(_svg_text(
                label_margin_left - 4,
                label_margin_top + i * cell_size + cell_size / 2 + 4,
                nm, size=max(8, cfg.font_size - 2), anchor="end",
            ))
        for j, nm in enumerate(names_b):
            parts.append(_svg_text(
                label_margin_left + j * cell_size + cell_size / 2,
                label_margin_top - 6, nm, size=max(8, cfg.font_size - 2),
                anchor="end", rotate=-45,
            ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)


class AnnotatedSignificanceHeatmap:
    """Heatmaps with significance stars based on p-value thresholds."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    @staticmethod
    def _significance_stars(p_value: float) -> str:
        """Return significance stars for a given p-value."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        return ""

    @staticmethod
    def _compute_correlation_pvalues(
        data_dict: Dict[str, List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation matrix and corresponding p-values."""
        names = sorted(data_dict.keys())
        n = len(names)
        corr = np.zeros((n, n))
        pvals = np.ones((n, n))

        for i in range(n):
            for j in range(n):
                xi = np.array(data_dict[names[i]], dtype=float)
                xj = np.array(data_dict[names[j]], dtype=float)
                min_len = min(len(xi), len(xj))
                xi, xj = xi[:min_len], xj[:min_len]
                if np.std(xi) == 0 or np.std(xj) == 0:
                    corr[i, j] = 0.0
                    pvals[i, j] = 1.0
                    continue
                r = float(np.corrcoef(xi, xj)[0, 1])
                corr[i, j] = r
                # approximate p-value using t-distribution approximation
                nn = len(xi)
                if nn > 2 and abs(r) < 1.0:
                    t_stat = r * math.sqrt((nn - 2) / (1 - r * r))
                    # two-tailed p-value approximation via normal
                    p = 2 * math.exp(-0.5 * t_stat * t_stat) / math.sqrt(2 * math.pi)
                    p = min(p, 1.0)
                    pvals[i, j] = p
                elif abs(r) >= 1.0:
                    pvals[i, j] = 0.0
                else:
                    pvals[i, j] = 1.0
        return corr, pvals

    def plot_annotated_heatmap(
        self,
        data_dict: Dict[str, List[float]],
        title: str = "Correlation Heatmap with Significance",
    ) -> str:
        """Render correlation heatmap with significance star annotations."""
        cfg = self.config
        names = sorted(data_dict.keys())
        n = len(names)
        if n == 0:
            return _empty_svg_global("No data")

        corr, pvals = self._compute_correlation_pvalues(data_dict)

        label_margin = 110
        cell_size = min(50, (cfg.figsize[0] - label_margin - 80) / max(n, 1))
        cell_size = max(cell_size, 22)
        total_w = int(label_margin + cell_size * n + 100)
        total_h = int(label_margin + cell_size * n + 80)

        parts: List[str] = []
        parts.append(_SVG_HEADER.format(w=total_w, h=total_h))
        parts.append(_svg_text(total_w / 2, 24, title, size=cfg.font_size + 2,
                               anchor="middle", weight="bold"))

        top_off = label_margin
        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                t = (val + 1) / 2
                r_c = int(66 + 189 * (1 - t))
                b_c = int(66 + 189 * t)
                color = f"#{r_c:02x}88{b_c:02x}"
                cx = label_margin + j * cell_size
                cy = top_off + i * cell_size
                parts.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{color}" stroke="#bbb" '
                    f'stroke-width="0.5"/>'
                )
                # correlation value
                parts.append(_svg_text(
                    cx + cell_size / 2, cy + cell_size / 2 + 2,
                    f"{val:.2f}", size=max(8, cfg.font_size - 3), anchor="middle",
                ))
                # significance stars
                stars = self._significance_stars(pvals[i, j])
                if stars:
                    parts.append(_svg_text(
                        cx + cell_size / 2, cy + cell_size / 2 + 14,
                        stars, size=max(7, cfg.font_size - 4), anchor="middle",
                        weight="bold",
                    ))

        for i, nm in enumerate(names):
            parts.append(_svg_text(
                label_margin - 4, top_off + i * cell_size + cell_size / 2 + 4,
                nm, size=max(8, cfg.font_size - 2), anchor="end",
            ))
            parts.append(_svg_text(
                label_margin + i * cell_size + cell_size / 2,
                top_off - 6, nm, size=max(8, cfg.font_size - 2),
                anchor="end", rotate=-45,
            ))

        # legend
        legend_y = top_off + cell_size * n + 20
        parts.append(_svg_text(label_margin, legend_y,
                               "* p<0.05  ** p<0.01  *** p<0.001",
                               size=cfg.font_size - 2, anchor="start"))

        parts.append(_SVG_FOOTER)
        return "".join(parts)


def _empty_svg_global(message: str) -> str:
    """Return a minimal SVG with an error/info message."""
    escaped = html_module.escape(message)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80">'
        f'<text x="150" y="45" text-anchor="middle" font-size="14" '
        f'fill="#888">{escaped}</text></svg>'
    )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def create_default_plotter(**kwargs: Any) -> MetricHeatmapPlotter:
    """Create a MetricHeatmapPlotter with optional config overrides."""
    cfg = HeatmapConfig(**kwargs)
    return MetricHeatmapPlotter(cfg)


def quick_heatmap(data_dict: Dict[str, List[float]], **kwargs: Any) -> str:
    """One-call shortcut: data dict → SVG heatmap string."""
    plotter = create_default_plotter(**kwargs)
    cm = CorrelationMatrix(data_dict)
    corr = cm.compute_pearson()
    return plotter.plot_correlation_heatmap(corr, cm.get_metric_names())


def quick_report(data_dict: Dict[str, List[float]], **kwargs: Any) -> str:
    """One-call shortcut: data dict → full HTML report string."""
    plotter = create_default_plotter(**kwargs)
    return plotter.generate_correlation_report(data_dict)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Generate sample visualizations to verify the module works."""
    np.random.seed(42)
    n_samples = 100
    base = np.random.randn(n_samples)
    data: Dict[str, List[float]] = {
        "accuracy": (base * 0.5 + np.random.randn(n_samples) * 0.3).tolist(),
        "diversity": (base * -0.3 + np.random.randn(n_samples) * 0.5).tolist(),
        "novelty": (np.random.randn(n_samples) * 0.8).tolist(),
        "coherence": (base * 0.8 + np.random.randn(n_samples) * 0.1).tolist(),
        "fluency": (base * 0.7 + np.random.randn(n_samples) * 0.2).tolist(),
    }

    cm = CorrelationMatrix(data)
    print("Metric names:", cm.get_metric_names())
    print("Pearson:\n", cm.compute_pearson().round(3))
    print("Redundant pairs:", cm.find_redundant_pairs())
    print("Orthogonal pairs:", cm.find_orthogonal_pairs())
    print("Cluster order:", cm.cluster_metrics())

    pca = cm.principal_components(n_components=2)
    print("PCA explained variance:", pca["explained_variance_ratio"].round(3))

    plotter = MetricHeatmapPlotter(HeatmapConfig(
        figsize=(700, 700),
        title="Demo Correlation Heatmap",
        annotate=True,
    ))

    svg = plotter.plot_correlation_heatmap(cm.compute_pearson(), cm.get_metric_names())
    print(f"\nHeatmap SVG length: {len(svg)} chars")

    clustered = plotter.plot_clustered_heatmap(cm.compute_pearson(), cm.get_metric_names())
    print(f"Clustered SVG length: {len(clustered)} chars")

    scatter = plotter.plot_metric_scatter_matrix(data)
    print(f"Scatter SVG length: {len(scatter)} chars")

    dist_svg = plotter.plot_metric_distributions(data)
    print(f"Distribution SVG length: {len(dist_svg)} chars")

    rankings = {k: list(np.argsort(np.argsort(v)).astype(float)) for k, v in data.items()}
    rank_svg = plotter.plot_metric_rankings(rankings)
    print(f"Rankings SVG length: {len(rank_svg)} chars")

    redund = plotter.plot_redundancy_graph(cm.compute_pearson(), threshold=0.5)
    print(f"Redundancy SVG length: {len(redund)} chars")

    loadings = pca["loadings"]
    pca_svg = plotter.plot_pca_loadings(loadings, cm.get_metric_names())
    print(f"PCA SVG length: {len(pca_svg)} chars")

    report = plotter.generate_correlation_report(data)
    print(f"HTML report length: {len(report)} chars")

    print("\nAll visualizations generated successfully.")


if __name__ == "__main__":
    _demo()
