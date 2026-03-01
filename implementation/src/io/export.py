"""
Results export module for Diversity Decoding Arena.

Supports CSV, LaTeX, JSON, Markdown, and HTML export formats
with configurable formatting, statistical annotations, and
batch export pipelines.
"""

from __future__ import annotations

import csv
import gzip
import html as html_module
import io
import json
import math
import os
import re
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_path(path: Union[str, Path]) -> Path:
    """Ensure a path exists and return as Path object."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dictionary."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, (list, tuple)):
            for idx, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.extend(_flatten_dict(elem, f"{new_key}{sep}{idx}", sep).items())
                else:
                    items.append((f"{new_key}{sep}{idx}", elem))
        else:
            items.append((new_key, v))
    return dict(items)


def _try_numeric(value: Any) -> Any:
    """Attempt to coerce a value to float for formatting."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


def _fmt_number(value: Any, precision: int = 4) -> str:
    """Format a number to given decimal precision, pass-through strings."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _fmt_mean_std(mean: float, std: float, precision: int = 4) -> str:
    """Format mean ± std."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_safe_serialize(v) for v in obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # numpy compatibility without hard dependency
    type_name = type(obj).__name__
    if type_name == "ndarray":
        return obj.tolist()
    if type_name in ("int64", "int32", "float64", "float32"):
        return obj.item()
    return str(obj)


def _compute_column_widths(headers: List[str], rows: List[List[str]]) -> List[int]:
    """Compute minimum column widths for alignment."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
            else:
                widths.append(len(str(cell)))
    return widths


def _extract_headers_rows(
    data: List[Dict[str, Any]], precision: int = 4
) -> Tuple[List[str], List[List[str]]]:
    """Extract headers and formatted rows from list-of-dicts data."""
    if not data:
        return [], []
    all_keys: List[str] = []
    seen: Set[str] = set()
    for record in data:
        flat = _flatten_dict(record) if any(isinstance(v, dict) for v in record.values()) else record
        for k in flat:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    rows: List[List[str]] = []
    for record in data:
        flat = _flatten_dict(record) if any(isinstance(v, dict) for v in record.values()) else record
        row = [_fmt_number(flat.get(k, ""), precision) for k in all_keys]
        rows.append(row)
    return all_keys, rows


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _escape_html(text: str) -> str:
    """Escape HTML entities."""
    return html_module.escape(str(text))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for result exports."""

    format: str = "csv"
    output_path: str = ""
    include_metadata: bool = True
    precision: int = 4
    latex_options: Dict[str, Any] = field(default_factory=lambda: {
        "booktabs": True,
        "bold_best": True,
        "underline_second": True,
        "caption": "",
        "label": "",
        "footnotes": [],
        "font_size": "normalsize",
        "landscape": False,
        "multirow": True,
        "multicolumn": True,
        "siunitx": False,
        "highlight_pareto": False,
        "significance_markers": True,
    })
    csv_options: Dict[str, Any] = field(default_factory=lambda: {
        "delimiter": ",",
        "quoting": csv.QUOTE_MINIMAL,
        "header": True,
        "encoding": "utf-8",
        "line_terminator": "\n",
        "na_rep": "",
    })
    json_options: Dict[str, Any] = field(default_factory=lambda: {
        "indent": 2,
        "sort_keys": False,
        "ensure_ascii": False,
        "compress": False,
    })
    markdown_options: Dict[str, Any] = field(default_factory=lambda: {
        "alignment": "left",
        "include_toc": False,
    })
    html_options: Dict[str, Any] = field(default_factory=lambda: {
        "sortable": True,
        "alternating_rows": True,
        "css_theme": "default",
        "include_header": True,
    })

    def validate(self) -> None:
        valid_formats = {"csv", "latex", "json", "markdown", "html"}
        if self.format not in valid_formats:
            raise ValueError(f"Unknown format '{self.format}'. Must be one of {valid_formats}")
        if self.precision < 0:
            raise ValueError("precision must be non-negative")

    def copy(self, **overrides: Any) -> "ExportConfig":
        d = asdict(self)
        d.update(overrides)
        return ExportConfig(**d)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ResultsExporter(ABC):
    """Abstract base class for all exporters."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    @abstractmethod
    def export(self, data: Any, path: Union[str, Path], config: Optional[ExportConfig] = None) -> str:
        """Export *data* to *path*, return the path written."""
        ...

    @abstractmethod
    def export_comparison_table(
        self,
        algorithms: List[str],
        metrics: List[str],
        data: Dict[str, Dict[str, Any]],
    ) -> str:
        """Return a formatted comparison table as a string."""
        ...

    @abstractmethod
    def export_metric_summary(self, metric_values: Dict[str, List[float]]) -> str:
        """Return a summary string for the given metric values."""
        ...

    # Shared helpers --------------------------------------------------------

    def _cfg(self, config: Optional[ExportConfig] = None) -> ExportConfig:
        return config or self.config

    @staticmethod
    def _write_text(path: Union[str, Path], content: str, encoding: str = "utf-8") -> str:
        p = _ensure_path(path)
        p.write_text(content, encoding=encoding)
        return str(p)

    @staticmethod
    def _compute_summary_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                    "max": float("nan"), "median": float("nan"), "count": 0}
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "count": n,
        }

    @staticmethod
    def _rank_values(
        values: Dict[str, float], higher_is_better: bool = True
    ) -> Dict[str, int]:
        """Return rank mapping (1 = best)."""
        sorted_keys = sorted(values, key=lambda k: values[k], reverse=higher_is_better)
        return {k: rank + 1 for rank, k in enumerate(sorted_keys)}

    @staticmethod
    def _identify_pareto_optimal(
        points: Dict[str, Dict[str, float]], higher_is_better: Optional[Dict[str, bool]] = None
    ) -> Set[str]:
        """Return set of Pareto-optimal algorithm names."""
        if not points:
            return set()
        algos = list(points.keys())
        metrics = list(next(iter(points.values())).keys())
        if higher_is_better is None:
            higher_is_better = {m: True for m in metrics}
        pareto: Set[str] = set()
        for i, a in enumerate(algos):
            dominated = False
            for j, b in enumerate(algos):
                if i == j:
                    continue
                all_geq = True
                any_gt = False
                for m in metrics:
                    va = points[a][m]
                    vb = points[b][m]
                    if higher_is_better.get(m, True):
                        if vb < va:
                            all_geq = False
                        if vb > va:
                            any_gt = True
                    else:
                        if vb > va:
                            all_geq = False
                        if vb < va:
                            any_gt = True
                if all_geq and any_gt:
                    dominated = True
                    break
            if not dominated:
                pareto.add(a)
        return pareto


# ---------------------------------------------------------------------------
# CSV Exporter
# ---------------------------------------------------------------------------


class CSVExporter(ResultsExporter):
    """Export results to CSV files."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        super().__init__(config)
        if self.config.format == "csv":
            pass  # already correct
        else:
            self.config = self.config.copy(format="csv")

    # -- core interface -----------------------------------------------------

    def export(self, data: Any, path: Union[str, Path], config: Optional[ExportConfig] = None) -> str:
        cfg = self._cfg(config)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return self.export_flat(data, path, cfg)
        if isinstance(data, dict):
            return self.export_flat([data], path, cfg)
        raise TypeError(f"Unsupported data type {type(data)} for CSV export")

    def export_comparison_table(
        self,
        algorithms: List[str],
        metrics: List[str],
        data: Dict[str, Dict[str, Any]],
    ) -> str:
        buf = io.StringIO()
        opts = self.config.csv_options
        writer = csv.writer(
            buf,
            delimiter=opts.get("delimiter", ","),
            quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
            lineterminator=opts.get("line_terminator", "\n"),
        )
        writer.writerow(["Algorithm"] + metrics)
        for algo in algorithms:
            row: List[str] = [algo]
            for metric in metrics:
                val = data.get(algo, {}).get(metric, "")
                if isinstance(val, dict) and "mean" in val and "std" in val:
                    row.append(_fmt_mean_std(val["mean"], val["std"], self.config.precision))
                else:
                    row.append(_fmt_number(val, self.config.precision))
            writer.writerow(row)
        return buf.getvalue()

    def export_metric_summary(self, metric_values: Dict[str, List[float]]) -> str:
        buf = io.StringIO()
        opts = self.config.csv_options
        writer = csv.writer(
            buf,
            delimiter=opts.get("delimiter", ","),
            quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
            lineterminator=opts.get("line_terminator", "\n"),
        )
        writer.writerow(["Metric", "Mean", "Std", "Min", "Max", "Median", "Count"])
        for metric, values in sorted(metric_values.items()):
            stats = self._compute_summary_stats(values)
            prec = self.config.precision
            writer.writerow([
                metric,
                _fmt_number(stats["mean"], prec),
                _fmt_number(stats["std"], prec),
                _fmt_number(stats["min"], prec),
                _fmt_number(stats["max"], prec),
                _fmt_number(stats["median"], prec),
                int(stats["count"]),
            ])
        return buf.getvalue()

    # -- specialized exports ------------------------------------------------

    def export_flat(
        self,
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Flat CSV with one row per run, nested dicts flattened."""
        cfg = self._cfg(config)
        opts = cfg.csv_options
        flat_data = [self.handle_nested_data(record) for record in data]
        headers, rows = _extract_headers_rows(
            [{k: v for k, v in fd.items()} for fd in flat_data], cfg.precision
        )
        p = _ensure_path(path)
        encoding = opts.get("encoding", "utf-8")
        na_rep = opts.get("na_rep", "")
        with open(p, "w", newline="", encoding=encoding) as fh:
            writer = csv.writer(
                fh,
                delimiter=opts.get("delimiter", ","),
                quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
                lineterminator=opts.get("line_terminator", "\n"),
            )
            if opts.get("header", True):
                writer.writerow(headers)
            for row in rows:
                writer.writerow([na_rep if c == "" else c for c in row])
        return str(p)

    def export_pivot(
        self,
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        algorithm_key: str = "algorithm",
        metric_keys: Optional[List[str]] = None,
    ) -> str:
        """Pivoted CSV: rows = algorithms, columns = metrics."""
        grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        all_metrics: List[str] = []
        seen_metrics: Set[str] = set()
        for record in data:
            algo = record.get(algorithm_key, "unknown")
            for k, v in record.items():
                if k == algorithm_key:
                    continue
                numeric = _try_numeric(v)
                if isinstance(numeric, (int, float)):
                    grouped[algo][k].append(float(numeric))
                    if k not in seen_metrics:
                        all_metrics.append(k)
                        seen_metrics.add(k)
        if metric_keys:
            all_metrics = [m for m in metric_keys if m in seen_metrics]
        p = _ensure_path(path)
        opts = self.config.csv_options
        with open(p, "w", newline="", encoding=opts.get("encoding", "utf-8")) as fh:
            writer = csv.writer(
                fh,
                delimiter=opts.get("delimiter", ","),
                quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
                lineterminator=opts.get("line_terminator", "\n"),
            )
            header_row = ["Algorithm"]
            for m in all_metrics:
                header_row.extend([f"{m}_mean", f"{m}_std"])
            writer.writerow(header_row)
            for algo in sorted(grouped):
                row: List[str] = [algo]
                for m in all_metrics:
                    vals = grouped[algo].get(m, [])
                    stats = self._compute_summary_stats(vals)
                    row.append(_fmt_number(stats["mean"], self.config.precision))
                    row.append(_fmt_number(stats["std"], self.config.precision))
                writer.writerow(row)
        return str(p)

    def export_per_prompt(
        self,
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        prompt_key: str = "prompt_id",
        algorithm_key: str = "algorithm",
        metric_key: str = "score",
    ) -> str:
        """Per-prompt CSV: rows = prompts, columns = algorithms."""
        prompt_algo: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        algorithms: Set[str] = set()
        for record in data:
            prompt_id = str(record.get(prompt_key, "unknown"))
            algo = str(record.get(algorithm_key, "unknown"))
            val = _try_numeric(record.get(metric_key, 0))
            if isinstance(val, (int, float)):
                prompt_algo[prompt_id][algo].append(float(val))
                algorithms.add(algo)
        sorted_algos = sorted(algorithms)
        p = _ensure_path(path)
        opts = self.config.csv_options
        with open(p, "w", newline="", encoding=opts.get("encoding", "utf-8")) as fh:
            writer = csv.writer(
                fh,
                delimiter=opts.get("delimiter", ","),
                quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
                lineterminator=opts.get("line_terminator", "\n"),
            )
            writer.writerow(["Prompt"] + sorted_algos)
            for prompt_id in sorted(prompt_algo):
                row: List[str] = [prompt_id]
                for algo in sorted_algos:
                    vals = prompt_algo[prompt_id].get(algo, [])
                    if vals:
                        row.append(_fmt_number(statistics.mean(vals), self.config.precision))
                    else:
                        row.append(self.config.csv_options.get("na_rep", ""))
                writer.writerow(row)
        return str(p)

    def export_summary(
        self,
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        group_key: str = "algorithm",
    ) -> str:
        """Summary statistics CSV grouped by *group_key*."""
        grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        numeric_keys: List[str] = []
        seen: Set[str] = set()
        for record in data:
            group = str(record.get(group_key, "unknown"))
            for k, v in record.items():
                if k == group_key:
                    continue
                numeric = _try_numeric(v)
                if isinstance(numeric, (int, float)):
                    grouped[group][k].append(float(numeric))
                    if k not in seen:
                        numeric_keys.append(k)
                        seen.add(k)
        p = _ensure_path(path)
        opts = self.config.csv_options
        with open(p, "w", newline="", encoding=opts.get("encoding", "utf-8")) as fh:
            writer = csv.writer(
                fh,
                delimiter=opts.get("delimiter", ","),
                quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
                lineterminator=opts.get("line_terminator", "\n"),
            )
            header = [group_key]
            for k in numeric_keys:
                header.extend([f"{k}_mean", f"{k}_std", f"{k}_min", f"{k}_max", f"{k}_n"])
            writer.writerow(header)
            for group in sorted(grouped):
                row: List[str] = [group]
                for k in numeric_keys:
                    stats = self._compute_summary_stats(grouped[group].get(k, []))
                    prec = self.config.precision
                    row.extend([
                        _fmt_number(stats["mean"], prec),
                        _fmt_number(stats["std"], prec),
                        _fmt_number(stats["min"], prec),
                        _fmt_number(stats["max"], prec),
                        str(int(stats["count"])),
                    ])
                writer.writerow(row)
        return str(p)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def handle_nested_data(record: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested dicts/lists into dot-separated keys."""
        return _flatten_dict(record)


# ---------------------------------------------------------------------------
# LaTeX Exporter
# ---------------------------------------------------------------------------


class LaTeXExporter(ResultsExporter):
    """Export results to LaTeX tables."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        super().__init__(config)
        if self.config.format != "latex":
            self.config = self.config.copy(format="latex")

    # -- core interface -----------------------------------------------------

    def export(self, data: Any, path: Union[str, Path], config: Optional[ExportConfig] = None) -> str:
        cfg = self._cfg(config)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            content = self.export_table(data, path, cfg)
            return self._write_text(path, content)
        if isinstance(data, dict):
            content = self.export_table([data], path, cfg)
            return self._write_text(path, content)
        raise TypeError(f"Unsupported data type {type(data)} for LaTeX export")

    def export_comparison_table(
        self,
        algorithms: List[str],
        metrics: List[str],
        data: Dict[str, Dict[str, Any]],
        higher_is_better: Optional[Dict[str, bool]] = None,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Formatted comparison table with bold best and underlined second best."""
        cfg = self._cfg(config)
        opts = cfg.latex_options
        prec = cfg.precision

        # Collect raw values for ranking
        raw_values: Dict[str, Dict[str, float]] = {}
        for algo in algorithms:
            raw_values[algo] = {}
            for metric in metrics:
                val = data.get(algo, {}).get(metric, None)
                if isinstance(val, dict) and "mean" in val:
                    raw_values[algo][metric] = val["mean"]
                elif isinstance(val, (int, float)):
                    raw_values[algo][metric] = float(val)
                else:
                    raw_values[algo][metric] = float("nan")

        # Compute ranks per metric
        ranks: Dict[str, Dict[str, int]] = {}
        for metric in metrics:
            metric_vals = {a: raw_values[a][metric] for a in algorithms
                          if not math.isnan(raw_values[a][metric])}
            hib = (higher_is_better or {}).get(metric, True)
            ranks[metric] = self._rank_values(metric_vals, higher_is_better=hib)

        # Pareto
        pareto: Set[str] = set()
        if opts.get("highlight_pareto", False):
            pareto_points = {a: raw_values[a] for a in algorithms}
            pareto = self._identify_pareto_optimal(pareto_points, higher_is_better)

        # Build table
        ncols = 1 + len(metrics)
        col_spec = "l" + "c" * len(metrics)
        lines: List[str] = []
        if opts.get("landscape", False):
            lines.append("\\begin{landscape}")
        lines.append(f"\\begin{{table}}[htbp]")
        lines.append(f"\\centering")
        if opts.get("font_size", "normalsize") != "normalsize":
            lines.append(f"\\{opts['font_size']}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        if opts.get("booktabs", True):
            lines.append("\\toprule")
        else:
            lines.append("\\hline")

        # Header
        header = "Algorithm & " + " & ".join(_escape_latex(m) for m in metrics) + " \\\\"
        lines.append(header)
        lines.append("\\midrule" if opts.get("booktabs", True) else "\\hline")

        # Rows
        for algo in algorithms:
            cells: List[str] = []
            algo_display = _escape_latex(algo)
            if algo in pareto:
                algo_display = f"\\colorbox{{yellow!20}}{{{algo_display}}}"
            cells.append(algo_display)
            for metric in metrics:
                val = data.get(algo, {}).get(metric, None)
                if isinstance(val, dict) and "mean" in val and "std" in val:
                    cell_text = f"${val['mean']:.{prec}f} \\pm {val['std']:.{prec}f}$"
                elif isinstance(val, (int, float)):
                    cell_text = f"${val:.{prec}f}$"
                else:
                    cell_text = _escape_latex(str(val)) if val is not None else "--"
                # Formatting: bold best, underline second
                rank = ranks.get(metric, {}).get(algo, 999)
                if rank == 1 and opts.get("bold_best", True):
                    cell_text = f"\\textbf{{{cell_text}}}"
                elif rank == 2 and opts.get("underline_second", True):
                    cell_text = f"\\underline{{{cell_text}}}"
                cells.append(cell_text)
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule" if opts.get("booktabs", True) else "\\hline")
        lines.append("\\end{tabular}")

        # Caption and label
        caption = opts.get("caption", "")
        if caption:
            lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        label = opts.get("label", "")
        if label:
            lines.append(f"\\label{{{label}}}")

        # Footnotes
        footnotes = opts.get("footnotes", [])
        if footnotes:
            lines.append("\\vspace{2mm}")
            for fn in footnotes:
                lines.append(f"\\footnotesize{{{_escape_latex(fn)}}}")

        lines.append("\\end{table}")
        if opts.get("landscape", False):
            lines.append("\\end{landscape}")
        return "\n".join(lines)

    def export_metric_summary(self, metric_values: Dict[str, List[float]]) -> str:
        prec = self.config.precision
        lines: List[str] = []
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Metric & Mean & Std & Min & Max & Median \\\\")
        lines.append("\\midrule")
        for metric, values in sorted(metric_values.items()):
            stats = self._compute_summary_stats(values)
            cells = [
                _escape_latex(metric),
                f"${stats['mean']:.{prec}f}$",
                f"${stats['std']:.{prec}f}$",
                f"${stats['min']:.{prec}f}$",
                f"${stats['max']:.{prec}f}$",
                f"${stats['median']:.{prec}f}$",
            ]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    # -- specialized --------------------------------------------------------

    def export_table(
        self,
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Generic LaTeX table from list of dicts."""
        cfg = self._cfg(config)
        opts = cfg.latex_options
        headers, rows = _extract_headers_rows(data, cfg.precision)
        ncols = len(headers)
        col_spec = "l" + "r" * (ncols - 1)
        lines: List[str] = []
        lines.append(f"\\begin{{table}}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule" if opts.get("booktabs", True) else "\\hline")
        lines.append(" & ".join(_escape_latex(h) for h in headers) + " \\\\")
        lines.append("\\midrule" if opts.get("booktabs", True) else "\\hline")
        for row in rows:
            lines.append(" & ".join(_escape_latex(c) for c in row) + " \\\\")
        lines.append("\\bottomrule" if opts.get("booktabs", True) else "\\hline")
        lines.append("\\end{tabular}")
        caption = opts.get("caption", "")
        if caption:
            lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        label = opts.get("label", "")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def export_statistical_table(
        self,
        test_results: List[Dict[str, Any]],
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Table of statistical test results with p-values, effect sizes, and significance markers."""
        opts = self.config.latex_options
        prec = self.config.precision
        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{llrrrl}")
        lines.append("\\toprule" if opts.get("booktabs", True) else "\\hline")
        lines.append("Comparison & Metric & $p$-value & Effect Size & $\\Delta$ & Sig. \\\\")
        lines.append("\\midrule" if opts.get("booktabs", True) else "\\hline")

        for result in test_results:
            comparison = _escape_latex(result.get("comparison", ""))
            metric = _escape_latex(result.get("metric", ""))
            p_value = result.get("p_value", 1.0)
            effect_size = result.get("effect_size", 0.0)
            delta = result.get("delta", 0.0)
            sig = _significance_marker(p_value) if opts.get("significance_markers", True) else ""

            p_str = f"${p_value:.{prec}f}$"
            if p_value < 0.05:
                p_str = f"\\textbf{{{p_str}}}"

            cells = [
                comparison,
                metric,
                p_str,
                f"${effect_size:.{prec}f}$",
                f"${delta:+.{prec}f}$",
                sig,
            ]
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule" if opts.get("booktabs", True) else "\\hline")
        lines.append("\\end{tabular}")
        if opts.get("significance_markers", True):
            lines.append("\\vspace{2mm}")
            lines.append("\\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$}")
        caption = opts.get("caption", "")
        if caption:
            lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        label = opts.get("label", "")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        content = "\n".join(lines)
        if path is not None:
            self._write_text(path, content)
        return content

    def export_multirow_table(
        self,
        data: Dict[str, Dict[str, Dict[str, Any]]],
        row_groups: List[str],
        sub_rows: List[str],
        columns: List[str],
    ) -> str:
        """Table with multirow and multicolumn support.

        *data* is ``{row_group: {sub_row: {column: value}}}``.
        """
        opts = self.config.latex_options
        prec = self.config.precision
        ncols = 1 + 1 + len(columns)
        col_spec = "ll" + "r" * len(columns)
        lines: List[str] = []
        lines.append("\\begin{tabular}{" + col_spec + "}")
        lines.append("\\toprule" if opts.get("booktabs", True) else "\\hline")
        lines.append(
            "Group & Setting & " + " & ".join(_escape_latex(c) for c in columns) + " \\\\"
        )
        lines.append("\\midrule" if opts.get("booktabs", True) else "\\hline")

        for group in row_groups:
            group_data = data.get(group, {})
            first = True
            n_sub = len(sub_rows)
            for sub in sub_rows:
                cells: List[str] = []
                if first and opts.get("multirow", True):
                    cells.append(f"\\multirow{{{n_sub}}}{{*}}{{{_escape_latex(group)}}}")
                    first = False
                else:
                    cells.append("")
                cells.append(_escape_latex(sub))
                for col in columns:
                    val = group_data.get(sub, {}).get(col, "")
                    cells.append(_fmt_number(val, prec))
                lines.append(" & ".join(cells) + " \\\\")
            lines.append("\\midrule" if opts.get("booktabs", True) else "\\hline")

        # Replace last midrule with bottomrule
        if lines and lines[-1] in ("\\midrule", "\\hline"):
            lines[-1] = "\\bottomrule" if opts.get("booktabs", True) else "\\hline"

        lines.append("\\end{tabular}")
        return "\n".join(lines)

    def export_multicolumn_header(
        self,
        group_headers: List[Tuple[str, int]],
        sub_headers: List[str],
    ) -> str:
        """Generate a multicolumn header row.

        *group_headers* is ``[(group_name, span), ...]``.
        """
        parts: List[str] = []
        for name, span in group_headers:
            parts.append(f"\\multicolumn{{{span}}}{{c}}{{{_escape_latex(name)}}}")
        top_row = " & ".join(parts) + " \\\\"
        sub_row = " & ".join(_escape_latex(h) for h in sub_headers) + " \\\\"
        return f"{top_row}\n\\cmidrule(lr){{1-{len(sub_headers)}}}\n{sub_row}"

    @staticmethod
    def generate_preamble(options: Optional[Dict[str, Any]] = None) -> str:
        """Generate required LaTeX preamble for the tables produced."""
        opts = options or {}
        packages: List[str] = []
        packages.append("\\usepackage{booktabs}")
        packages.append("\\usepackage{multirow}")
        packages.append("\\usepackage{amsmath}")
        packages.append("\\usepackage{graphicx}")
        if opts.get("landscape", False):
            packages.append("\\usepackage{lscape}")
        if opts.get("highlight_pareto", False) or opts.get("bold_best", True):
            packages.append("\\usepackage[table]{xcolor}")
        if opts.get("siunitx", False):
            packages.append("\\usepackage{siunitx}")
            packages.append("\\sisetup{detect-weight=true, detect-inline-weight=math}")
        packages.append("\\usepackage{caption}")
        return "\n".join(packages)

    def format_uncertainty(self, mean: float, std: float, precision: Optional[int] = None) -> str:
        """Format value with uncertainty in LaTeX math mode."""
        prec = precision or self.config.precision
        if self.config.latex_options.get("siunitx", False):
            return f"\\num{{{mean:.{prec}f} +- {std:.{prec}f}}}"
        return f"${mean:.{prec}f} \\pm {std:.{prec}f}$"

    def format_with_significance(
        self, value: float, p_value: float, precision: Optional[int] = None
    ) -> str:
        """Format a value with significance marker."""
        prec = precision or self.config.precision
        marker = _significance_marker(p_value)
        return f"${value:.{prec}f}${marker}"


# ---------------------------------------------------------------------------
# JSON Exporter
# ---------------------------------------------------------------------------


class JSONExporter(ResultsExporter):
    """Export results to JSON files."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        super().__init__(config)
        if self.config.format != "json":
            self.config = self.config.copy(format="json")

    # -- core interface -----------------------------------------------------

    def export(self, data: Any, path: Union[str, Path], config: Optional[ExportConfig] = None) -> str:
        cfg = self._cfg(config)
        return self.export_full(data, path, cfg)

    def export_comparison_table(
        self,
        algorithms: List[str],
        metrics: List[str],
        data: Dict[str, Dict[str, Any]],
    ) -> str:
        output: Dict[str, Any] = {
            "type": "comparison_table",
            "algorithms": algorithms,
            "metrics": metrics,
            "results": {},
        }
        for algo in algorithms:
            output["results"][algo] = {}
            for metric in metrics:
                output["results"][algo][metric] = _safe_serialize(
                    data.get(algo, {}).get(metric, None)
                )
        opts = self.config.json_options
        return json.dumps(
            output,
            indent=opts.get("indent", 2),
            sort_keys=opts.get("sort_keys", False),
            ensure_ascii=opts.get("ensure_ascii", False),
        )

    def export_metric_summary(self, metric_values: Dict[str, List[float]]) -> str:
        output: Dict[str, Any] = {"type": "metric_summary", "metrics": {}}
        for metric, values in sorted(metric_values.items()):
            output["metrics"][metric] = self._compute_summary_stats(values)
        opts = self.config.json_options
        return json.dumps(
            output,
            indent=opts.get("indent", 2),
            sort_keys=opts.get("sort_keys", False),
            ensure_ascii=opts.get("ensure_ascii", False),
        )

    # -- specialized --------------------------------------------------------

    def export_full(
        self,
        data: Any,
        path: Union[str, Path],
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Full JSON export with metadata."""
        cfg = self._cfg(config)
        opts = cfg.json_options
        output: Dict[str, Any] = {}
        if cfg.include_metadata:
            output["metadata"] = {
                "exported_at": datetime.utcnow().isoformat(),
                "format": "json",
                "version": "1.0",
                "precision": cfg.precision,
            }
        output["data"] = _safe_serialize(data)
        content = json.dumps(
            output,
            indent=opts.get("indent", 2),
            sort_keys=opts.get("sort_keys", False),
            ensure_ascii=opts.get("ensure_ascii", False),
        )
        p = _ensure_path(path)
        if opts.get("compress", False):
            gz_path = p.with_suffix(p.suffix + ".gz")
            with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
                fh.write(content)
            return str(gz_path)
        p.write_text(content, encoding="utf-8")
        return str(p)

    def export_minimal(
        self, data: Any, path: Union[str, Path]
    ) -> str:
        """Compact JSON without indentation or metadata."""
        content = json.dumps(_safe_serialize(data), separators=(",", ":"), ensure_ascii=False)
        p = _ensure_path(path)
        p.write_text(content, encoding="utf-8")
        return str(p)

    def export_streaming(
        self, data: Iterable[Any], path: Union[str, Path]
    ) -> str:
        """Line-delimited JSON (NDJSON / JSON Lines)."""
        p = _ensure_path(path)
        compress = self.config.json_options.get("compress", False)
        if compress:
            gz_path = p.with_suffix(p.suffix + ".gz")
            with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
                for record in data:
                    fh.write(json.dumps(_safe_serialize(record), ensure_ascii=False) + "\n")
            return str(gz_path)
        with open(p, "w", encoding="utf-8") as fh:
            for record in data:
                fh.write(json.dumps(_safe_serialize(record), ensure_ascii=False) + "\n")
        return str(p)

    def export_compressed(
        self, data: Any, path: Union[str, Path]
    ) -> str:
        """Export JSON with gzip compression."""
        content = json.dumps(
            _safe_serialize(data),
            indent=self.config.json_options.get("indent", 2),
            ensure_ascii=False,
        )
        p = _ensure_path(path)
        gz_path = p.with_suffix(p.suffix + ".gz") if not str(p).endswith(".gz") else p
        with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
            fh.write(content)
        return str(gz_path)

    @staticmethod
    def generate_schema(data: Any) -> Dict[str, Any]:
        """Generate a JSON Schema from sample data."""

        def _infer_type(val: Any) -> Dict[str, Any]:
            if val is None:
                return {"type": "null"}
            if isinstance(val, bool):
                return {"type": "boolean"}
            if isinstance(val, int):
                return {"type": "integer"}
            if isinstance(val, float):
                return {"type": "number"}
            if isinstance(val, str):
                return {"type": "string"}
            if isinstance(val, list):
                if not val:
                    return {"type": "array", "items": {}}
                item_schemas = [_infer_type(v) for v in val[:5]]
                types = list({s.get("type") for s in item_schemas})
                if len(types) == 1:
                    return {"type": "array", "items": item_schemas[0]}
                return {"type": "array", "items": {"oneOf": item_schemas[:3]}}
            if isinstance(val, dict):
                properties: Dict[str, Any] = {}
                for k, v in val.items():
                    properties[k] = _infer_type(v)
                return {
                    "type": "object",
                    "properties": properties,
                    "required": list(val.keys()),
                }
            return {"type": "string"}

        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "DiversityDecodingResults",
        }
        if isinstance(data, list) and data:
            schema["type"] = "array"
            schema["items"] = _infer_type(data[0])
        elif isinstance(data, dict):
            obj_schema = _infer_type(data)
            schema.update(obj_schema)
        else:
            schema.update(_infer_type(data))
        return schema

    @staticmethod
    def serialize_numpy(obj: Any) -> Any:
        """Recursively convert numpy types for JSON serialization."""
        return _safe_serialize(obj)


# ---------------------------------------------------------------------------
# Markdown Exporter
# ---------------------------------------------------------------------------


class MarkdownExporter(ResultsExporter):
    """Export results to GitHub-flavored Markdown."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        super().__init__(config)
        if self.config.format != "markdown":
            self.config = self.config.copy(format="markdown")

    # -- core interface -----------------------------------------------------

    def export(self, data: Any, path: Union[str, Path], config: Optional[ExportConfig] = None) -> str:
        cfg = self._cfg(config)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            content = self.export_table(data, cfg)
        elif isinstance(data, dict):
            content = self.export_table([data], cfg)
        else:
            content = str(data)
        return self._write_text(path, content)

    def export_comparison_table(
        self,
        algorithms: List[str],
        metrics: List[str],
        data: Dict[str, Dict[str, Any]],
    ) -> str:
        prec = self.config.precision
        headers = ["Algorithm"] + metrics
        rows: List[List[str]] = []
        for algo in algorithms:
            row: List[str] = [algo]
            for metric in metrics:
                val = data.get(algo, {}).get(metric, "")
                if isinstance(val, dict) and "mean" in val and "std" in val:
                    row.append(_fmt_mean_std(val["mean"], val["std"], prec))
                else:
                    row.append(_fmt_number(val, prec))
            rows.append(row)
        return self._build_md_table(headers, rows)

    def export_metric_summary(self, metric_values: Dict[str, List[float]]) -> str:
        prec = self.config.precision
        headers = ["Metric", "Mean", "Std", "Min", "Max", "Median", "Count"]
        rows: List[List[str]] = []
        for metric, values in sorted(metric_values.items()):
            stats = self._compute_summary_stats(values)
            rows.append([
                metric,
                _fmt_number(stats["mean"], prec),
                _fmt_number(stats["std"], prec),
                _fmt_number(stats["min"], prec),
                _fmt_number(stats["max"], prec),
                _fmt_number(stats["median"], prec),
                str(int(stats["count"])),
            ])
        return self._build_md_table(headers, rows)

    # -- specialized --------------------------------------------------------

    def export_table(self, data: List[Dict[str, Any]], config: Optional[ExportConfig] = None) -> str:
        """GFM table from list of dicts."""
        cfg = self._cfg(config)
        headers, rows = _extract_headers_rows(data, cfg.precision)
        return self._build_md_table(headers, rows)

    def export_report(
        self,
        data: Dict[str, Any],
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Full markdown report with sections."""
        cfg = self._cfg(config)
        prec = cfg.precision
        sections: List[str] = []

        title = data.get("title", "Diversity Decoding Arena — Results Report")
        sections.append(f"# {title}\n")

        if cfg.include_metadata:
            sections.append("## Metadata\n")
            meta = data.get("metadata", {})
            if not meta:
                meta = {"generated": datetime.utcnow().isoformat()}
            for k, v in meta.items():
                sections.append(f"- **{k}**: {v}")
            sections.append("")

        if cfg.markdown_options.get("include_toc", False):
            sections.append("## Table of Contents\n")
            toc_items = [
                "1. [Summary](#summary)",
                "2. [Comparison Table](#comparison-table)",
                "3. [Per-Metric Analysis](#per-metric-analysis)",
                "4. [Statistical Tests](#statistical-tests)",
            ]
            sections.extend(toc_items)
            sections.append("")

        # Summary section
        summary = data.get("summary", "")
        if summary:
            sections.append("## Summary\n")
            sections.append(str(summary))
            sections.append("")

        # Comparison table
        comparison = data.get("comparison", None)
        if comparison and isinstance(comparison, dict):
            sections.append("## Comparison Table\n")
            algos = comparison.get("algorithms", [])
            metrics = comparison.get("metrics", [])
            results = comparison.get("results", {})
            if algos and metrics and results:
                sections.append(self.export_comparison_table(algos, metrics, results))
            sections.append("")

        # Per-metric analysis
        metrics_data = data.get("metrics", {})
        if metrics_data:
            sections.append("## Per-Metric Analysis\n")
            for metric_name, values in sorted(metrics_data.items()):
                sections.append(f"### {metric_name}\n")
                if isinstance(values, list) and values and isinstance(values[0], (int, float)):
                    stats = self._compute_summary_stats(values)
                    sections.append(f"- **Mean**: {stats['mean']:.{prec}f}")
                    sections.append(f"- **Std**: {stats['std']:.{prec}f}")
                    sections.append(f"- **Min**: {stats['min']:.{prec}f}")
                    sections.append(f"- **Max**: {stats['max']:.{prec}f}")
                    sections.append(f"- **Median**: {stats['median']:.{prec}f}")
                elif isinstance(values, dict):
                    for k, v in values.items():
                        sections.append(f"- **{k}**: {_fmt_number(v, prec)}")
                sections.append("")

        # Statistical tests
        stat_tests = data.get("statistical_tests", [])
        if stat_tests:
            sections.append("## Statistical Tests\n")
            headers = ["Comparison", "Metric", "p-value", "Effect Size", "Significant"]
            rows: List[List[str]] = []
            for t in stat_tests:
                p = t.get("p_value", 1.0)
                rows.append([
                    t.get("comparison", ""),
                    t.get("metric", ""),
                    _fmt_number(p, prec),
                    _fmt_number(t.get("effect_size", 0.0), prec),
                    "Yes" if p < 0.05 else "No",
                ])
            sections.append(self._build_md_table(headers, rows))
            sections.append("")

        # Raw data tables
        raw_tables = data.get("tables", {})
        if raw_tables:
            for table_name, table_data in raw_tables.items():
                sections.append(f"## {table_name}\n")
                if isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
                    sections.append(self.export_table(table_data, cfg))
                sections.append("")

        return "\n".join(sections)

    def export_summary(self, data: Dict[str, Any], config: Optional[ExportConfig] = None) -> str:
        """Brief summary of results."""
        cfg = self._cfg(config)
        prec = cfg.precision
        lines: List[str] = []
        lines.append("# Results Summary\n")
        if "best_algorithm" in data:
            lines.append(f"**Best Algorithm**: {data['best_algorithm']}")
        if "best_score" in data:
            lines.append(f"**Best Score**: {_fmt_number(data['best_score'], prec)}")
        if "num_algorithms" in data:
            lines.append(f"**Algorithms Evaluated**: {data['num_algorithms']}")
        if "num_prompts" in data:
            lines.append(f"**Prompts Used**: {data['num_prompts']}")
        if "metrics" in data:
            lines.append("\n## Metrics\n")
            for m, v in data["metrics"].items():
                if isinstance(v, dict):
                    lines.append(
                        f"- **{m}**: {_fmt_number(v.get('mean', 0), prec)} "
                        f"± {_fmt_number(v.get('std', 0), prec)}"
                    )
                else:
                    lines.append(f"- **{m}**: {_fmt_number(v, prec)}")
        return "\n".join(lines)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _build_md_table(headers: List[str], rows: List[List[str]]) -> str:
        """Build a GitHub-flavored markdown table string."""
        if not headers:
            return ""
        widths = _compute_column_widths(headers, rows)
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
        data_lines: List[str] = []
        for row in rows:
            padded = []
            for i, w in enumerate(widths):
                cell = row[i] if i < len(row) else ""
                padded.append(str(cell).ljust(w))
            data_lines.append("| " + " | ".join(padded) + " |")
        return "\n".join([header_line, sep_line] + data_lines)


# ---------------------------------------------------------------------------
# HTML Exporter
# ---------------------------------------------------------------------------


_DEFAULT_CSS = """
<style>
    .results-table {
        border-collapse: collapse;
        width: 100%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        margin: 20px 0;
    }
    .results-table th {
        background-color: #2c3e50;
        color: white;
        padding: 12px 15px;
        text-align: left;
        cursor: pointer;
        user-select: none;
        position: relative;
    }
    .results-table th:hover {
        background-color: #34495e;
    }
    .results-table th::after {
        content: ' \\2195';
        font-size: 12px;
        opacity: 0.5;
    }
    .results-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #ddd;
    }
    .results-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .results-table tr:hover {
        background-color: #e8f4fd;
    }
    .results-table .best {
        font-weight: bold;
        color: #27ae60;
    }
    .results-table .second-best {
        font-style: italic;
        color: #2980b9;
    }
    .report-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .report-container h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
    .report-container h2 { color: #34495e; margin-top: 30px; }
    .report-container h3 { color: #7f8c8d; }
    .metric-card {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .stat-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
</style>
"""

_SORT_JS = """
<script>
function sortTable(table, colIdx) {
    var rows = Array.from(table.tBodies[0].rows);
    var asc = table.dataset.sortCol == colIdx && table.dataset.sortDir == 'asc';
    table.dataset.sortCol = colIdx;
    table.dataset.sortDir = asc ? 'desc' : 'asc';
    rows.sort(function(a, b) {
        var va = a.cells[colIdx].textContent.trim();
        var vb = b.cells[colIdx].textContent.trim();
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) {
            return asc ? nb - na : na - nb;
        }
        return asc ? vb.localeCompare(va) : va.localeCompare(vb);
    });
    rows.forEach(function(row) { table.tBodies[0].appendChild(row); });
}
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.results-table th').forEach(function(th, idx) {
        th.addEventListener('click', function() {
            sortTable(th.closest('table'), idx);
        });
    });
});
</script>
"""


class HTMLExporter(ResultsExporter):
    """Export results to HTML."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        super().__init__(config)
        if self.config.format != "html":
            self.config = self.config.copy(format="html")

    # -- core interface -----------------------------------------------------

    def export(self, data: Any, path: Union[str, Path], config: Optional[ExportConfig] = None) -> str:
        cfg = self._cfg(config)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            content = self.export_table(data, cfg)
        elif isinstance(data, dict):
            content = self.export_table([data], cfg)
        else:
            content = f"<pre>{_escape_html(str(data))}</pre>"
        return self._write_text(path, self._wrap_html(content, "Results", cfg))

    def export_comparison_table(
        self,
        algorithms: List[str],
        metrics: List[str],
        data: Dict[str, Dict[str, Any]],
    ) -> str:
        prec = self.config.precision
        headers = ["Algorithm"] + metrics
        rows: List[List[str]] = []
        # Collect raw values for best/second detection
        raw: Dict[str, Dict[str, float]] = {}
        for algo in algorithms:
            raw[algo] = {}
            for metric in metrics:
                val = data.get(algo, {}).get(metric, None)
                if isinstance(val, dict) and "mean" in val:
                    raw[algo][metric] = val["mean"]
                elif isinstance(val, (int, float)):
                    raw[algo][metric] = float(val)
                else:
                    raw[algo][metric] = float("nan")

        # Rank per metric
        metric_ranks: Dict[str, Dict[str, int]] = {}
        for metric in metrics:
            vals = {a: raw[a][metric] for a in algorithms if not math.isnan(raw[a][metric])}
            metric_ranks[metric] = self._rank_values(vals, higher_is_better=True)

        lines: List[str] = []
        lines.append('<table class="results-table">')
        lines.append("  <thead><tr>")
        for h in headers:
            lines.append(f"    <th>{_escape_html(h)}</th>")
        lines.append("  </tr></thead>")
        lines.append("  <tbody>")

        for algo in algorithms:
            lines.append("  <tr>")
            lines.append(f"    <td>{_escape_html(algo)}</td>")
            for metric in metrics:
                val = data.get(algo, {}).get(metric, "")
                if isinstance(val, dict) and "mean" in val and "std" in val:
                    cell_text = _fmt_mean_std(val["mean"], val["std"], prec)
                else:
                    cell_text = _fmt_number(val, prec)
                rank = metric_ranks.get(metric, {}).get(algo, 999)
                css_class = ""
                if rank == 1:
                    css_class = ' class="best"'
                elif rank == 2:
                    css_class = ' class="second-best"'
                lines.append(f"    <td{css_class}>{_escape_html(cell_text)}</td>")
            lines.append("  </tr>")

        lines.append("  </tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    def export_metric_summary(self, metric_values: Dict[str, List[float]]) -> str:
        prec = self.config.precision
        headers = ["Metric", "Mean", "Std", "Min", "Max", "Median", "Count"]
        lines: List[str] = []
        lines.append('<table class="results-table">')
        lines.append("  <thead><tr>")
        for h in headers:
            lines.append(f"    <th>{_escape_html(h)}</th>")
        lines.append("  </tr></thead>")
        lines.append("  <tbody>")
        for metric, values in sorted(metric_values.items()):
            stats = self._compute_summary_stats(values)
            lines.append("  <tr>")
            lines.append(f"    <td>{_escape_html(metric)}</td>")
            for key in ["mean", "std", "min", "max", "median"]:
                lines.append(f"    <td>{_fmt_number(stats[key], prec)}</td>")
            lines.append(f"    <td>{int(stats['count'])}</td>")
            lines.append("  </tr>")
        lines.append("  </tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    # -- specialized --------------------------------------------------------

    def export_table(self, data: List[Dict[str, Any]], config: Optional[ExportConfig] = None) -> str:
        """HTML table with CSS styling from list of dicts."""
        cfg = self._cfg(config)
        opts = cfg.html_options
        headers, rows = _extract_headers_rows(data, cfg.precision)
        lines: List[str] = []
        lines.append('<table class="results-table">')
        lines.append("  <thead><tr>")
        for h in headers:
            lines.append(f"    <th>{_escape_html(h)}</th>")
        lines.append("  </tr></thead>")
        lines.append("  <tbody>")
        for i, row in enumerate(rows):
            row_class = ""
            if opts.get("alternating_rows", True) and i % 2 == 1:
                row_class = ' class="alt-row"'
            lines.append(f"  <tr{row_class}>")
            for cell in row:
                lines.append(f"    <td>{_escape_html(str(cell))}</td>")
            lines.append("  </tr>")
        lines.append("  </tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    def export_report(
        self, data: Dict[str, Any], config: Optional[ExportConfig] = None
    ) -> str:
        """Full HTML report with sections, cards, and tables."""
        cfg = self._cfg(config)
        prec = cfg.precision
        sections: List[str] = []

        title = data.get("title", "Diversity Decoding Arena — Results Report")
        sections.append(f"<h1>{_escape_html(title)}</h1>")

        # Metadata
        if cfg.include_metadata:
            meta = data.get("metadata", {"generated": datetime.utcnow().isoformat()})
            sections.append("<h2>Metadata</h2>")
            sections.append("<ul>")
            for k, v in meta.items():
                sections.append(f"  <li><strong>{_escape_html(str(k))}</strong>: {_escape_html(str(v))}</li>")
            sections.append("</ul>")

        # Summary cards
        summary = data.get("summary_stats", {})
        if summary:
            sections.append("<h2>Summary</h2>")
            sections.append('<div style="display: flex; flex-wrap: wrap; gap: 15px;">')
            for label, value in summary.items():
                sections.append('<div class="metric-card">')
                sections.append(f'  <div class="stat-value">{_escape_html(_fmt_number(value, prec))}</div>')
                sections.append(f'  <div class="stat-label">{_escape_html(str(label))}</div>')
                sections.append("</div>")
            sections.append("</div>")

        # Comparison table
        comparison = data.get("comparison", None)
        if comparison and isinstance(comparison, dict):
            sections.append("<h2>Algorithm Comparison</h2>")
            algos = comparison.get("algorithms", [])
            metrics = comparison.get("metrics", [])
            results = comparison.get("results", {})
            if algos and metrics and results:
                sections.append(self.export_comparison_table(algos, metrics, results))

        # Metric details
        metrics_data = data.get("metrics", {})
        if metrics_data:
            sections.append("<h2>Per-Metric Analysis</h2>")
            for metric_name, values in sorted(metrics_data.items()):
                sections.append(f"<h3>{_escape_html(metric_name)}</h3>")
                if isinstance(values, list) and values and isinstance(values[0], (int, float)):
                    stats = self._compute_summary_stats(values)
                    sections.append('<div class="metric-card">')
                    for k in ["mean", "std", "min", "max", "median"]:
                        sections.append(
                            f'<span class="stat-label">{k}: </span>'
                            f'<span class="stat-value">{_fmt_number(stats[k], prec)}</span>&nbsp;&nbsp;'
                        )
                    sections.append("</div>")

        # Raw tables
        raw_tables = data.get("tables", {})
        if raw_tables:
            for table_name, table_data in raw_tables.items():
                sections.append(f"<h2>{_escape_html(table_name)}</h2>")
                if isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
                    sections.append(self.export_table(table_data, cfg))

        body = "\n".join(sections)
        return self._wrap_html(body, title, cfg)

    # -- helpers ------------------------------------------------------------

    def _wrap_html(self, body: str, title: str, config: Optional[ExportConfig] = None) -> str:
        cfg = self._cfg(config)
        opts = cfg.html_options
        parts: List[str] = []
        if opts.get("include_header", True):
            parts.append("<!DOCTYPE html>")
            parts.append("<html lang=\"en\">")
            parts.append("<head>")
            parts.append(f"  <meta charset=\"utf-8\">")
            parts.append(f"  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
            parts.append(f"  <title>{_escape_html(title)}</title>")
            parts.append(_DEFAULT_CSS)
            parts.append("</head>")
            parts.append("<body>")
            parts.append('<div class="report-container">')
        parts.append(body)
        if opts.get("include_header", True):
            parts.append("</div>")
            if opts.get("sortable", True):
                parts.append(_SORT_JS)
            parts.append("</body>")
            parts.append("</html>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Export Pipeline
# ---------------------------------------------------------------------------


@dataclass
class ExportTask:
    """Single export task within a pipeline."""
    exporter: ResultsExporter
    config: ExportConfig
    output_path: str
    data_transform: Optional[Callable[[Any], Any]] = None
    post_process: Optional[Callable[[str], None]] = None
    name: str = ""


class ExportPipeline:
    """Configure and run multiple exports in a single call."""

    def __init__(self) -> None:
        self._tasks: List[ExportTask] = []
        self._results: Dict[str, str] = {}

    def add_task(self, task: ExportTask) -> "ExportPipeline":
        if not task.name:
            task.name = f"task_{len(self._tasks)}"
        self._tasks.append(task)
        return self

    def add_csv_export(
        self,
        output_path: str,
        config: Optional[ExportConfig] = None,
        name: str = "",
        data_transform: Optional[Callable[[Any], Any]] = None,
    ) -> "ExportPipeline":
        cfg = config or ExportConfig(format="csv")
        task = ExportTask(
            exporter=CSVExporter(cfg),
            config=cfg,
            output_path=output_path,
            data_transform=data_transform,
            name=name or f"csv_{len(self._tasks)}",
        )
        return self.add_task(task)

    def add_latex_export(
        self,
        output_path: str,
        config: Optional[ExportConfig] = None,
        name: str = "",
        data_transform: Optional[Callable[[Any], Any]] = None,
    ) -> "ExportPipeline":
        cfg = config or ExportConfig(format="latex")
        task = ExportTask(
            exporter=LaTeXExporter(cfg),
            config=cfg,
            output_path=output_path,
            data_transform=data_transform,
            name=name or f"latex_{len(self._tasks)}",
        )
        return self.add_task(task)

    def add_json_export(
        self,
        output_path: str,
        config: Optional[ExportConfig] = None,
        name: str = "",
        data_transform: Optional[Callable[[Any], Any]] = None,
    ) -> "ExportPipeline":
        cfg = config or ExportConfig(format="json")
        task = ExportTask(
            exporter=JSONExporter(cfg),
            config=cfg,
            output_path=output_path,
            data_transform=data_transform,
            name=name or f"json_{len(self._tasks)}",
        )
        return self.add_task(task)

    def add_markdown_export(
        self,
        output_path: str,
        config: Optional[ExportConfig] = None,
        name: str = "",
        data_transform: Optional[Callable[[Any], Any]] = None,
    ) -> "ExportPipeline":
        cfg = config or ExportConfig(format="markdown")
        task = ExportTask(
            exporter=MarkdownExporter(cfg),
            config=cfg,
            output_path=output_path,
            data_transform=data_transform,
            name=name or f"markdown_{len(self._tasks)}",
        )
        return self.add_task(task)

    def add_html_export(
        self,
        output_path: str,
        config: Optional[ExportConfig] = None,
        name: str = "",
        data_transform: Optional[Callable[[Any], Any]] = None,
    ) -> "ExportPipeline":
        cfg = config or ExportConfig(format="html")
        task = ExportTask(
            exporter=HTMLExporter(cfg),
            config=cfg,
            output_path=output_path,
            data_transform=data_transform,
            name=name or f"html_{len(self._tasks)}",
        )
        return self.add_task(task)

    def run(self, data: Any) -> Dict[str, str]:
        """Execute all configured export tasks and return {name: output_path}."""
        self._results.clear()
        errors: Dict[str, str] = {}
        for task in self._tasks:
            try:
                task_data = data
                if task.data_transform is not None:
                    task_data = task.data_transform(data)
                result_path = task.exporter.export(task_data, task.output_path, task.config)
                if task.post_process is not None:
                    task.post_process(result_path)
                self._results[task.name] = result_path
            except Exception as exc:
                errors[task.name] = str(exc)
        if errors:
            err_msg = "; ".join(f"{k}: {v}" for k, v in errors.items())
            raise RuntimeError(f"Export pipeline errors: {err_msg}")
        return dict(self._results)

    def run_safe(self, data: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Like *run* but collects errors instead of raising."""
        self._results.clear()
        errors: Dict[str, str] = {}
        for task in self._tasks:
            try:
                task_data = data
                if task.data_transform is not None:
                    task_data = task.data_transform(data)
                result_path = task.exporter.export(task_data, task.output_path, task.config)
                if task.post_process is not None:
                    task.post_process(result_path)
                self._results[task.name] = result_path
            except Exception as exc:
                errors[task.name] = str(exc)
        return dict(self._results), errors

    @property
    def results(self) -> Dict[str, str]:
        return dict(self._results)

    def clear(self) -> None:
        self._tasks.clear()
        self._results.clear()


# ---------------------------------------------------------------------------
# Format Converter
# ---------------------------------------------------------------------------


class FormatConverter:
    """Convert between export formats."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    def csv_to_latex(
        self,
        csv_path: Union[str, Path],
        delimiter: str = ",",
        booktabs: bool = True,
    ) -> str:
        """Read a CSV file and return a LaTeX table string."""
        p = Path(csv_path)
        with open(p, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        data_rows = rows[1:]
        ncols = len(headers)
        col_spec = "l" + "r" * (ncols - 1)
        lines: List[str] = []
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule" if booktabs else "\\hline")
        lines.append(" & ".join(_escape_latex(h) for h in headers) + " \\\\")
        lines.append("\\midrule" if booktabs else "\\hline")
        for row in data_rows:
            cells = [_escape_latex(c) for c in row]
            while len(cells) < ncols:
                cells.append("")
            lines.append(" & ".join(cells[:ncols]) + " \\\\")
        lines.append("\\bottomrule" if booktabs else "\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    def json_to_csv(
        self,
        json_path: Union[str, Path],
        csv_path: Union[str, Path],
        data_key: Optional[str] = None,
    ) -> str:
        """Convert a JSON file to CSV. Returns the output CSV path."""
        jp = Path(json_path)
        content = jp.read_text(encoding="utf-8")
        raw = json.loads(content)

        if data_key and isinstance(raw, dict):
            raw = raw.get(data_key, raw)

        if isinstance(raw, dict) and not isinstance(raw, list):
            # Single record or nested structure
            if "data" in raw and isinstance(raw["data"], list):
                records = raw["data"]
            else:
                records = [raw]
        elif isinstance(raw, list):
            records = raw
        else:
            records = [{"value": raw}]

        flat_records = [_flatten_dict(r) if isinstance(r, dict) else {"value": r} for r in records]

        all_keys: List[str] = []
        seen: Set[str] = set()
        for rec in flat_records:
            for k in rec:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        cp = _ensure_path(csv_path)
        with open(cp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for rec in flat_records:
                writer.writerow({k: _fmt_number(rec.get(k, ""), self.config.precision) for k in all_keys})
        return str(cp)

    @staticmethod
    def latex_to_markdown(latex_str: str) -> str:
        """Best-effort conversion of a LaTeX tabular to a Markdown table."""
        content = latex_str
        # Strip environment wrappers
        content = re.sub(r"\\begin\{table\}.*?\n?", "", content)
        content = re.sub(r"\\end\{table\}", "", content)
        content = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", content)
        content = re.sub(r"\\end\{tabular\}", "", content)
        content = re.sub(r"\\centering", "", content)
        content = re.sub(r"\\caption\{[^}]*\}", "", content)
        content = re.sub(r"\\label\{[^}]*\}", "", content)
        content = re.sub(r"\\(toprule|midrule|bottomrule|hline|cmidrule(\([^)]*\))?(\{[^}]*\})?)", "", content)
        content = re.sub(r"\\footnotesize\{[^}]*\}", "", content)
        content = re.sub(r"\\vspace\{[^}]*\}", "", content)

        # Strip formatting commands
        content = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", content)
        content = re.sub(r"\\underline\{([^}]*)\}", r"_\1_", content)
        content = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", content)
        content = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", content)
        content = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", content)
        content = re.sub(r"\\colorbox\{[^}]*\}\{([^}]*)\}", r"\1", content)
        content = re.sub(r"\\num\{([^}]*)\}", r"\1", content)

        # Strip remaining LaTeX commands
        content = re.sub(r"\\\w+\{([^}]*)\}", r"\1", content)
        content = re.sub(r"\$([^$]*)\$", r"\1", content)

        # Unescape LaTeX special characters
        content = content.replace("\\&", "&")
        content = content.replace("\\%", "%")
        content = content.replace("\\#", "#")
        content = content.replace("\\_", "_")
        content = content.replace("\\$", "$")
        content = content.replace("\\{", "{")
        content = content.replace("\\}", "}")
        content = content.replace("\\pm", "±")

        # Parse rows
        raw_rows = [r.strip() for r in content.split("\\\\") if r.strip()]
        if not raw_rows:
            return ""

        md_rows: List[List[str]] = []
        for raw_row in raw_rows:
            cells = [c.strip() for c in raw_row.split("&")]
            if any(c for c in cells):
                md_rows.append(cells)

        if not md_rows:
            return ""

        headers = md_rows[0]
        data_rows = md_rows[1:]
        widths = _compute_column_widths(headers, data_rows)
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
        lines: List[str] = [header_line, sep_line]
        for row in data_rows:
            padded = []
            for i, w in enumerate(widths):
                cell = row[i] if i < len(row) else ""
                padded.append(cell.ljust(w))
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)

    def csv_to_json(
        self,
        csv_path: Union[str, Path],
        json_path: Union[str, Path],
        delimiter: str = ",",
    ) -> str:
        """Convert a CSV file to JSON. Returns the output JSON path."""
        p = Path(csv_path)
        with open(p, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            records = list(reader)
        # Attempt numeric coercion
        coerced: List[Dict[str, Any]] = []
        for rec in records:
            row: Dict[str, Any] = {}
            for k, v in rec.items():
                row[k] = _try_numeric(v) if v else v
            coerced.append(row)
        jp = _ensure_path(json_path)
        jp.write_text(
            json.dumps(coerced, indent=self.config.json_options.get("indent", 2), ensure_ascii=False),
            encoding="utf-8",
        )
        return str(jp)

    def csv_to_markdown(
        self,
        csv_path: Union[str, Path],
        delimiter: str = ",",
    ) -> str:
        """Convert a CSV file to a Markdown table string."""
        p = Path(csv_path)
        with open(p, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        data_rows = [[str(c) for c in r] for r in rows[1:]]
        widths = _compute_column_widths(headers, data_rows)
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
        lines = [header_line, sep_line]
        for row in data_rows:
            padded = []
            for i, w in enumerate(widths):
                cell = row[i] if i < len(row) else ""
                padded.append(cell.ljust(w))
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)

    def csv_to_html(
        self,
        csv_path: Union[str, Path],
        delimiter: str = ",",
    ) -> str:
        """Convert a CSV file to an HTML table string."""
        p = Path(csv_path)
        with open(p, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        data_rows = rows[1:]
        lines: List[str] = ['<table class="results-table">', "  <thead><tr>"]
        for h in headers:
            lines.append(f"    <th>{_escape_html(h)}</th>")
        lines.append("  </tr></thead>")
        lines.append("  <tbody>")
        for row in data_rows:
            lines.append("  <tr>")
            for cell in row:
                lines.append(f"    <td>{_escape_html(cell)}</td>")
            lines.append("  </tr>")
        lines.append("  </tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    def json_to_latex(
        self,
        json_path: Union[str, Path],
        data_key: Optional[str] = None,
        booktabs: bool = True,
    ) -> str:
        """Convert a JSON file to a LaTeX table string."""
        jp = Path(json_path)
        raw = json.loads(jp.read_text(encoding="utf-8"))
        if data_key and isinstance(raw, dict):
            raw = raw.get(data_key, raw)
        if isinstance(raw, dict) and "data" in raw:
            records = raw["data"] if isinstance(raw["data"], list) else [raw["data"]]
        elif isinstance(raw, list):
            records = raw
        else:
            records = [raw]
        flat = [_flatten_dict(r) if isinstance(r, dict) else {"value": r} for r in records]
        exporter = LaTeXExporter(self.config.copy(format="latex"))
        return exporter.export_table(flat, "/dev/null")

    def json_to_markdown(
        self,
        json_path: Union[str, Path],
        data_key: Optional[str] = None,
    ) -> str:
        """Convert a JSON file to a Markdown table string."""
        jp = Path(json_path)
        raw = json.loads(jp.read_text(encoding="utf-8"))
        if data_key and isinstance(raw, dict):
            raw = raw.get(data_key, raw)
        if isinstance(raw, dict) and "data" in raw:
            records = raw["data"] if isinstance(raw["data"], list) else [raw["data"]]
        elif isinstance(raw, list):
            records = raw
        else:
            records = [raw]
        flat = [_flatten_dict(r) if isinstance(r, dict) else {"value": r} for r in records]
        exporter = MarkdownExporter(self.config.copy(format="markdown"))
        return exporter.export_table(flat)

    def markdown_to_html(self, md_str: str) -> str:
        """Basic Markdown-to-HTML for tables and headings."""
        lines = md_str.strip().split("\n")
        output: List[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            # Headings
            if stripped.startswith("###"):
                text = stripped.lstrip("#").strip()
                output.append(f"<h3>{_escape_html(text)}</h3>")
                continue
            if stripped.startswith("##"):
                text = stripped.lstrip("#").strip()
                output.append(f"<h2>{_escape_html(text)}</h2>")
                continue
            if stripped.startswith("#"):
                text = stripped.lstrip("#").strip()
                output.append(f"<h1>{_escape_html(text)}</h1>")
                continue

            # Table rows
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                # Check if separator row
                if all(set(c.strip()) <= {"-", ":"} for c in cells):
                    continue
                if not in_table:
                    in_table = True
                    output.append('<table class="results-table">')
                    output.append("  <thead><tr>")
                    for c in cells:
                        output.append(f"    <th>{_escape_html(c)}</th>")
                    output.append("  </tr></thead>")
                    output.append("  <tbody>")
                else:
                    output.append("  <tr>")
                    for c in cells:
                        output.append(f"    <td>{_escape_html(c)}</td>")
                    output.append("  </tr>")
                continue

            # End of table
            if in_table:
                output.append("  </tbody>")
                output.append("</table>")
                in_table = False

            # List items
            if stripped.startswith("- "):
                text = stripped[2:]
                text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
                output.append(f"<li>{text}</li>")
                continue

            # Bold
            processed = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", stripped)
            if processed:
                output.append(f"<p>{processed}</p>")

        if in_table:
            output.append("  </tbody>")
            output.append("</table>")

        return "\n".join(output)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_exporter(format_name: str, config: Optional[ExportConfig] = None) -> ResultsExporter:
    """Factory function returning the appropriate exporter."""
    exporters: Dict[str, type] = {
        "csv": CSVExporter,
        "latex": LaTeXExporter,
        "json": JSONExporter,
        "markdown": MarkdownExporter,
        "html": HTMLExporter,
    }
    cls = exporters.get(format_name)
    if cls is None:
        raise ValueError(f"Unknown format '{format_name}'. Available: {list(exporters.keys())}")
    return cls(config)


def export_all_formats(
    data: Any,
    output_dir: Union[str, Path],
    base_name: str = "results",
    config: Optional[ExportConfig] = None,
) -> Dict[str, str]:
    """Export data to all supported formats in *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config or ExportConfig()

    pipeline = ExportPipeline()
    pipeline.add_csv_export(str(out / f"{base_name}.csv"), cfg.copy(format="csv"), name="csv")
    pipeline.add_json_export(str(out / f"{base_name}.json"), cfg.copy(format="json"), name="json")
    pipeline.add_latex_export(str(out / f"{base_name}.tex"), cfg.copy(format="latex"), name="latex")
    pipeline.add_markdown_export(str(out / f"{base_name}.md"), cfg.copy(format="markdown"), name="markdown")
    pipeline.add_html_export(str(out / f"{base_name}.html"), cfg.copy(format="html"), name="html")
    return pipeline.run(data)


def quick_comparison_export(
    algorithms: List[str],
    metrics: List[str],
    data: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path],
    base_name: str = "comparison",
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Export a comparison table in multiple formats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_formats = formats or ["csv", "latex", "markdown"]
    results: Dict[str, str] = {}
    for fmt in target_formats:
        exporter = get_exporter(fmt)
        content = exporter.export_comparison_table(algorithms, metrics, data)
        ext = {"csv": "csv", "latex": "tex", "json": "json", "markdown": "md", "html": "html"}
        file_ext = ext.get(fmt, fmt)
        p = out / f"{base_name}.{file_ext}"
        p.write_text(content, encoding="utf-8")
        results[fmt] = str(p)
    return results


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------


class BatchExporter:
    """Export a collection of result sets, each to its own file."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    def export_batch(
        self,
        datasets: Dict[str, Any],
        output_dir: Union[str, Path],
        format_name: str = "csv",
    ) -> Dict[str, str]:
        """Export multiple named datasets.

        *datasets*: ``{name: data}``
        Returns ``{name: output_path}``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        exporter = get_exporter(format_name, self.config)
        ext_map = {"csv": "csv", "latex": "tex", "json": "json", "markdown": "md", "html": "html"}
        ext = ext_map.get(format_name, format_name)
        results: Dict[str, str] = {}
        for name, data in datasets.items():
            safe_name = re.sub(r"[^\w\-.]", "_", name)
            path = out / f"{safe_name}.{ext}"
            results[name] = exporter.export(data, path)
        return results

    def export_batch_all_formats(
        self,
        datasets: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> Dict[str, Dict[str, str]]:
        """Export every dataset in every format."""
        results: Dict[str, Dict[str, str]] = {}
        for name, data in datasets.items():
            results[name] = export_all_formats(data, output_dir, base_name=name, config=self.config)
        return results


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------


class TableFormatter:
    """Shared formatting utilities used across exporters."""

    def __init__(self, precision: int = 4) -> None:
        self.precision = precision

    def format_value(self, value: Any) -> str:
        return _fmt_number(value, self.precision)

    def format_mean_std(self, mean: float, std: float) -> str:
        return _fmt_mean_std(mean, std, self.precision)

    def format_percentage(self, value: float) -> str:
        return f"{value * 100:.{max(0, self.precision - 2)}f}%"

    def format_scientific(self, value: float) -> str:
        return f"{value:.{self.precision}e}"

    def format_with_rank(self, value: float, rank: int, total: int) -> str:
        return f"{self.format_value(value)} ({rank}/{total})"

    def highlight_best(
        self,
        values: Dict[str, float],
        higher_is_better: bool = True,
        format_type: str = "latex",
    ) -> Dict[str, str]:
        """Return formatted values with the best highlighted."""
        ranks = ResultsExporter._rank_values(values, higher_is_better)
        result: Dict[str, str] = {}
        for key, val in values.items():
            formatted = self.format_value(val)
            rank = ranks.get(key, 999)
            if format_type == "latex":
                if rank == 1:
                    formatted = f"\\textbf{{{formatted}}}"
                elif rank == 2:
                    formatted = f"\\underline{{{formatted}}}"
            elif format_type == "html":
                if rank == 1:
                    formatted = f'<strong class="best">{formatted}</strong>'
                elif rank == 2:
                    formatted = f'<em class="second-best">{formatted}</em>'
            elif format_type == "markdown":
                if rank == 1:
                    formatted = f"**{formatted}**"
                elif rank == 2:
                    formatted = f"*{formatted}*"
            result[key] = formatted
        return result

    def create_legend(self, format_type: str = "latex") -> str:
        """Return a legend string explaining formatting conventions."""
        if format_type == "latex":
            return (
                "\\footnotesize{"
                "\\textbf{Bold}: best result; "
                "\\underline{Underlined}: second best; "
                "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$"
                "}"
            )
        if format_type == "markdown":
            return (
                "*Legend*: **Bold** = best result, *Italic* = second best. "
                "\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001"
            )
        if format_type == "html":
            return (
                '<p class="legend">'
                '<strong>Bold</strong>: best result; '
                '<em>Italic</em>: second best; '
                '* p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001'
                "</p>"
            )
        return "Bold: best result; Underlined: second best"


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate exported JSON against a schema."""

    def __init__(self, schema: Dict[str, Any]) -> None:
        self.schema = schema

    def validate_type(self, value: Any, type_spec: Dict[str, Any]) -> List[str]:
        """Return list of validation error messages (empty = valid)."""
        errors: List[str] = []
        expected = type_spec.get("type")
        if expected is None:
            return errors

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None),
            "array": list,
            "object": dict,
        }

        if expected in type_map:
            ok_types = type_map[expected]
            if not isinstance(value, ok_types):
                errors.append(f"Expected {expected}, got {type(value).__name__}")
                return errors

        if expected == "object" and isinstance(value, dict):
            properties = type_spec.get("properties", {})
            required = type_spec.get("required", [])
            for req in required:
                if req not in value:
                    errors.append(f"Missing required field '{req}'")
            for key, prop_spec in properties.items():
                if key in value:
                    errors.extend(
                        f"{key}.{e}" for e in self.validate_type(value[key], prop_spec)
                    )

        if expected == "array" and isinstance(value, list):
            items_spec = type_spec.get("items", {})
            if items_spec:
                for i, item in enumerate(value[:10]):
                    errors.extend(
                        f"[{i}].{e}" for e in self.validate_type(item, items_spec)
                    )

        return errors

    def validate(self, data: Any) -> List[str]:
        """Validate *data* against the stored schema."""
        return self.validate_type(data, self.schema)


# ---------------------------------------------------------------------------
# Streaming / incremental exporter
# ---------------------------------------------------------------------------


class IncrementalExporter:
    """Append results incrementally to a file."""

    def __init__(
        self,
        path: Union[str, Path],
        format_name: str = "csv",
        config: Optional[ExportConfig] = None,
    ) -> None:
        self.path = _ensure_path(path)
        self.format = format_name
        self.config = config or ExportConfig(format=format_name)
        self._headers_written = False
        self._headers: Optional[List[str]] = None
        self._count = 0

    def append(self, record: Dict[str, Any]) -> None:
        """Append a single record."""
        flat = _flatten_dict(record)
        if self.format == "csv":
            self._append_csv(flat)
        elif self.format == "json":
            self._append_jsonl(flat)
        else:
            raise ValueError(f"Incremental export not supported for '{self.format}'")
        self._count += 1

    def append_batch(self, records: List[Dict[str, Any]]) -> None:
        for r in records:
            self.append(r)

    def _append_csv(self, flat: Dict[str, Any]) -> None:
        opts = self.config.csv_options
        mode = "a" if self._headers_written else "w"
        with open(self.path, mode, newline="", encoding=opts.get("encoding", "utf-8")) as fh:
            if not self._headers_written:
                self._headers = list(flat.keys())
                writer = csv.DictWriter(
                    fh,
                    fieldnames=self._headers,
                    delimiter=opts.get("delimiter", ","),
                    quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
                    lineterminator=opts.get("line_terminator", "\n"),
                    extrasaction="ignore",
                )
                if opts.get("header", True):
                    writer.writeheader()
                self._headers_written = True
            else:
                # Check for new columns
                new_keys = [k for k in flat if k not in (self._headers or [])]
                if new_keys and self._headers is not None:
                    self._headers.extend(new_keys)
            writer = csv.DictWriter(
                fh if mode == "a" else open(self.path, "a", newline="", encoding=opts.get("encoding", "utf-8")),
                fieldnames=self._headers or [],
                delimiter=opts.get("delimiter", ","),
                quoting=opts.get("quoting", csv.QUOTE_MINIMAL),
                lineterminator=opts.get("line_terminator", "\n"),
                extrasaction="ignore",
            )
            writer.writerow({k: _fmt_number(v, self.config.precision) for k, v in flat.items()})

    def _append_jsonl(self, flat: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_safe_serialize(flat), ensure_ascii=False) + "\n")

    @property
    def count(self) -> int:
        return self._count

    def finalize(self) -> str:
        """Return the output path after all records have been written."""
        return str(self.path)


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


class ComparisonReportGenerator:
    """Generate a multi-format comparison report from experimental data."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    def generate(
        self,
        algorithms: List[str],
        metrics: List[str],
        run_data: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        higher_is_better: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, str]:
        """Aggregate run data and export comparison in all formats."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Aggregate
        agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for record in run_data:
            algo = record.get("algorithm", "unknown")
            for m in metrics:
                val = _try_numeric(record.get(m))
                if isinstance(val, (int, float)):
                    agg[algo][m].append(float(val))

        # Build summary dict
        summary_data: Dict[str, Dict[str, Any]] = {}
        for algo in algorithms:
            summary_data[algo] = {}
            for m in metrics:
                vals = agg[algo].get(m, [])
                if vals:
                    stats = ResultsExporter._compute_summary_stats(vals)
                    summary_data[algo][m] = {"mean": stats["mean"], "std": stats["std"]}
                else:
                    summary_data[algo][m] = {"mean": float("nan"), "std": float("nan")}

        results: Dict[str, str] = {}
        for fmt, ext in [("csv", "csv"), ("latex", "tex"), ("json", "json"),
                         ("markdown", "md"), ("html", "html")]:
            exporter = get_exporter(fmt, self.config)
            if fmt == "latex":
                content = exporter.export_comparison_table(
                    algorithms, metrics, summary_data,
                    higher_is_better=higher_is_better,
                )
            else:
                content = exporter.export_comparison_table(algorithms, metrics, summary_data)
            p = out / f"comparison.{ext}"
            p.write_text(content, encoding="utf-8")
            results[fmt] = str(p)
        return results


# ---------------------------------------------------------------------------
# Aggregate statistics exporter
# ---------------------------------------------------------------------------


class AggregateExporter:
    """Compute and export aggregate statistics across multiple runs."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    def compute_aggregates(
        self,
        data: List[Dict[str, Any]],
        group_key: str = "algorithm",
        metric_keys: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return ``{group: {metric: {mean, std, min, max, ...}}}``."""
        grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        all_metrics: List[str] = []
        seen: Set[str] = set()
        for record in data:
            group = str(record.get(group_key, "unknown"))
            for k, v in record.items():
                if k == group_key:
                    continue
                if metric_keys and k not in metric_keys:
                    continue
                numeric = _try_numeric(v)
                if isinstance(numeric, (int, float)):
                    grouped[group][k].append(float(numeric))
                    if k not in seen:
                        all_metrics.append(k)
                        seen.add(k)
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for group, metrics_map in grouped.items():
            result[group] = {}
            for m in all_metrics:
                vals = metrics_map.get(m, [])
                result[group][m] = ResultsExporter._compute_summary_stats(vals)
        return result

    def export_aggregates(
        self,
        data: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        group_key: str = "algorithm",
        format_name: str = "csv",
    ) -> str:
        """Compute aggregates and export to a single file."""
        agg = self.compute_aggregates(data, group_key)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ext = {"csv": "csv", "latex": "tex", "json": "json", "markdown": "md", "html": "html"}.get(
            format_name, format_name
        )
        path = out / f"aggregates.{ext}"

        if format_name == "json":
            exporter = JSONExporter(self.config)
            return exporter.export_full(agg, path)

        # Flatten for tabular formats
        flat_rows: List[Dict[str, Any]] = []
        for group, metrics_map in sorted(agg.items()):
            row: Dict[str, Any] = {group_key: group}
            for m, stats in sorted(metrics_map.items()):
                for stat_name, stat_val in stats.items():
                    row[f"{m}_{stat_name}"] = stat_val
            flat_rows.append(row)

        exporter = get_exporter(format_name, self.config)
        return exporter.export(flat_rows, path)


# ---------------------------------------------------------------------------
# Public API summary
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "ExportConfig",
    # ABC
    "ResultsExporter",
    # Exporters
    "CSVExporter",
    "LaTeXExporter",
    "JSONExporter",
    "MarkdownExporter",
    "HTMLExporter",
    # Pipeline
    "ExportPipeline",
    "ExportTask",
    # Converter
    "FormatConverter",
    # Utilities
    "BatchExporter",
    "IncrementalExporter",
    "ComparisonReportGenerator",
    "AggregateExporter",
    "TableFormatter",
    "SchemaValidator",
    # Factory
    "get_exporter",
    "export_all_formats",
    "quick_comparison_export",
]
