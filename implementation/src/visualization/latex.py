"""
LaTeX table generation module for the Diversity Decoding Arena.

Generates publication-quality LaTeX tables, figures, and documents
for comparing algorithm performance, statistical significance, and metrics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# LaTeX escaping utilities
# ---------------------------------------------------------------------------

class LaTeXEscaper:
    """Escape and format text for safe LaTeX output."""

    _SPECIAL_CHARS: Dict[str, str] = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }

    @classmethod
    def escape(cls, text: str) -> str:
        """Escape special LaTeX characters in *text*."""
        if not isinstance(text, str):
            text = str(text)
        # Backslash must be replaced first so we don't double-escape.
        result = text.replace("\\", r"\textbackslash{}")
        for ch, replacement in cls._SPECIAL_CHARS.items():
            if ch == "\\":
                continue
            result = result.replace(ch, replacement)
        return result

    @staticmethod
    def math_mode(text: str) -> str:
        """Wrap *text* in inline math mode ``$...$``."""
        return f"${text}$"

    @staticmethod
    def bold(text: str) -> str:
        """Wrap *text* in ``\\textbf{...}``."""
        return rf"\textbf{{{text}}}"

    @staticmethod
    def italic(text: str) -> str:
        """Wrap *text* in ``\\textit{...}``."""
        return rf"\textit{{{text}}}"

    @staticmethod
    def texttt(text: str) -> str:
        """Wrap *text* in ``\\texttt{...}`` (monospace)."""
        return rf"\texttt{{{text}}}"

    @staticmethod
    def underline(text: str) -> str:
        """Wrap *text* in ``\\underline{...}``."""
        return rf"\underline{{{text}}}"

    @staticmethod
    def color(text: str, color_name: str) -> str:
        """Wrap *text* with ``\\textcolor{color}{...}``."""
        return rf"\textcolor{{{color_name}}}{{{text}}}"


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

class NumberFormatter:
    """Format numeric values for LaTeX tables."""

    @staticmethod
    def format_mean_std(
        mean: float,
        std: float,
        precision: int = 3,
    ) -> str:
        r"""Format as ``mean ± std``, e.g. ``0.734 $\\pm$ 0.021``."""
        mean_s = f"{mean:.{precision}f}"
        std_s = f"{std:.{precision}f}"
        return rf"{mean_s} $\pm$ {std_s}"

    @staticmethod
    def format_ci(
        mean: float,
        lo: float,
        hi: float,
        precision: int = 2,
    ) -> str:
        """Format as ``mean [lo, hi]``."""
        return (
            f"{mean:.{precision}f} "
            f"[{lo:.{precision}f}, {hi:.{precision}f}]"
        )

    @staticmethod
    def format_pvalue(p: float) -> str:
        """Format a p-value with conventional thresholds.

        Returns strings like ``< 0.001``, ``0.042``, etc.
        """
        if p < 0.001:
            return "< 0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        elif p < 0.1:
            return f"{p:.3f}"
        else:
            return f"{p:.2f}"

    @staticmethod
    def format_scientific(value: float, precision: int = 2) -> str:
        r"""Format in scientific notation, e.g. ``$1.23 \\times 10^{4}$``."""
        if value == 0:
            return LaTeXEscaper.math_mode("0")
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10 ** exponent)
        return LaTeXEscaper.math_mode(
            rf"{mantissa:.{precision}f} \times 10^{{{exponent}}}"
        )

    @staticmethod
    def format_percentage(value: float, precision: int = 1) -> str:
        r"""Format as a percentage, e.g. ``73.4\\%``."""
        return rf"{value * 100:.{precision}f}\%"

    @staticmethod
    def format_with_significance(
        value: float,
        p_value: float,
        precision: int = 3,
    ) -> str:
        """Append significance markers to a formatted value.

        Markers: ``*`` p < 0.05, ``**`` p < 0.01, ``***`` p < 0.001.
        """
        formatted = f"{value:.{precision}f}"
        if p_value < 0.001:
            marker = "***"
        elif p_value < 0.01:
            marker = "**"
        elif p_value < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            formatted = rf"{formatted}$^{{{marker}}}$"
        return formatted

    @staticmethod
    def format_integer(value: int) -> str:
        """Format with thousands separator."""
        return f"{value:,}".replace(",", r"{,}")

    @staticmethod
    def format_float(value: float, precision: int = 3) -> str:
        """Simple fixed-point formatting."""
        return f"{value:.{precision}f}"


# ---------------------------------------------------------------------------
# Color definitions
# ---------------------------------------------------------------------------

class ColorDefinitions:
    """Generate LaTeX color definitions and cell-coloring commands."""

    # Pre-defined color palette
    PALETTE: Dict[str, Tuple[int, int, int]] = {
        "bestgreen": (34, 139, 34),
        "secondblue": (70, 130, 180),
        "warnred": (220, 20, 60),
        "neutralgray": (200, 200, 200),
        "heatlow": (255, 255, 204),
        "heatmid": (253, 174, 97),
        "heathigh": (215, 48, 39),
        "rowalt": (245, 245, 255),
    }

    @classmethod
    def define_colors(cls) -> str:
        """Return ``\\definecolor`` commands for the palette."""
        lines: List[str] = []
        for name, (r, g, b) in cls.PALETTE.items():
            lines.append(
                rf"\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}"
            )
        return "\n".join(lines)

    @staticmethod
    def cell_color(
        value: float,
        min_val: float,
        max_val: float,
        colormap: str = "green",
    ) -> str:
        r"""Return a ``\\cellcolor`` command that interpolates a color.

        Parameters
        ----------
        value : float
            The cell value.
        min_val, max_val : float
            Range used for normalisation.
        colormap : str
            ``"green"`` (low=white, high=green),
            ``"red"`` (low=white, high=red),
            ``"blue"`` (low=white, high=blue), or
            ``"heat"`` (low=yellow, high=red).
        """
        if max_val == min_val:
            t = 0.5
        else:
            t = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

        cmap_endpoints: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
            "green": ((255, 255, 255), (34, 139, 34)),
            "red": ((255, 255, 255), (220, 20, 60)),
            "blue": ((255, 255, 255), (70, 130, 180)),
            "heat": ((255, 255, 204), (215, 48, 39)),
        }

        lo_c, hi_c = cmap_endpoints.get(colormap, cmap_endpoints["green"])
        r = int(lo_c[0] + t * (hi_c[0] - lo_c[0]))
        g = int(lo_c[1] + t * (hi_c[1] - lo_c[1]))
        b = int(lo_c[2] + t * (hi_c[2] - lo_c[2]))

        return rf"\cellcolor[RGB]{{{r},{g},{b}}}"

    @staticmethod
    def row_color(color_name: str) -> str:
        r"""Return ``\\rowcolor{...}``."""
        return rf"\rowcolor{{{color_name}}}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FontSize(str, Enum):
    """Allowed LaTeX font size commands."""

    NORMALSIZE = "normalsize"
    SMALL = "small"
    FOOTNOTESIZE = "footnotesize"
    SCRIPTSIZE = "scriptsize"
    TINY = "tiny"
    LARGE = "large"


@dataclass
class LaTeXTableConfig:
    """Configuration for ``LaTeXTable`` rendering."""

    caption: str = ""
    label: str = ""
    position: str = "htbp"
    booktabs: bool = True
    centering: bool = True
    font_size: str = "normalsize"
    column_format: str = ""
    highlight_best: str = "bold"
    highlight_second: str = "underline"
    precision: int = 3
    scientific_notation: bool = False
    escape_special_chars: bool = True
    resizebox: bool = False
    resizebox_width: str = r"\textwidth"
    landscape: bool = False
    alternating_row_colors: bool = False
    alternating_color: str = "rowalt"
    header_bold: bool = True
    vertical_lines: bool = False
    horizontal_lines: bool = False
    footnotes: List[str] = field(default_factory=list)
    extra_preamble: str = ""
    arraystretch: float = 1.0
    tabcolsep: float = 6.0


# ---------------------------------------------------------------------------
# Core LaTeX table
# ---------------------------------------------------------------------------

class _CellMeta:
    """Internal cell metadata."""

    __slots__ = ("content", "multicolumn", "multirow", "color", "alignment")

    def __init__(
        self,
        content: str = "",
        multicolumn: Optional[Tuple[int, str]] = None,
        multirow: Optional[int] = None,
        color: str = "",
        alignment: str = "",
    ) -> None:
        self.content = content
        self.multicolumn = multicolumn
        self.multirow = multirow
        self.color = color
        self.alignment = alignment


class LaTeXTable:
    """Build and render a LaTeX table from structured data.

    Parameters
    ----------
    data : list of list
        2-D data (rows × columns).  Elements may be ``str``, ``int``,
        ``float``, or ``None``.
    columns : list of str, optional
        Column headers.
    rows : list of str, optional
        Row headers (prepended to each data row).
    config : LaTeXTableConfig, optional
        Rendering configuration.
    """

    def __init__(
        self,
        data: Optional[List[List[Any]]] = None,
        columns: Optional[List[str]] = None,
        rows: Optional[List[str]] = None,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig()
        self.columns: List[str] = list(columns) if columns else []
        self.rows: List[str] = list(rows) if rows else []
        self._header_rows: List[List[str]] = []
        self._data_rows: List[Union[List[str], str]] = []
        self._footnotes: List[str] = list(self.config.footnotes)
        self._column_format: str = self.config.column_format

        if data is not None:
            self._ingest(data)

    # -- ingestion -----------------------------------------------------------

    def _ingest(self, data: List[List[Any]]) -> None:
        """Load 2-D data into internal row storage."""
        if self.columns:
            self._header_rows.append(list(self.columns))
        for i, row in enumerate(data):
            cells: List[str] = []
            if self.rows and i < len(self.rows):
                cells.append(self._format_cell(self.rows[i]))
            for val in row:
                cells.append(self._format_cell(val))
            self._data_rows.append(cells)

    def _format_cell(self, value: Any) -> str:
        """Format a single cell value to a string."""
        if value is None:
            return "---"
        if isinstance(value, float):
            if self.config.scientific_notation:
                return NumberFormatter.format_scientific(
                    value, self.config.precision
                )
            return f"{value:.{self.config.precision}f}"
        if isinstance(value, int):
            return str(value)
        text = str(value)
        if self.config.escape_special_chars:
            text = LaTeXEscaper.escape(text)
        return text

    # -- builder API ---------------------------------------------------------

    def add_header_row(self, cells: List[str]) -> "LaTeXTable":
        """Append an additional header row."""
        self._header_rows.append(list(cells))
        return self

    def add_data_row(self, cells: List[str]) -> "LaTeXTable":
        """Append a data row (list of already-formatted strings)."""
        self._data_rows.append(list(cells))
        return self

    def add_midrule(self) -> "LaTeXTable":
        r"""Insert a ``\\midrule`` (booktabs) or ``\\hline``."""
        if self.config.booktabs:
            self._data_rows.append(r"\midrule")
        else:
            self._data_rows.append(r"\hline")
        return self

    def add_cmidrule(self, start: int, end: int, trim: str = "") -> "LaTeXTable":
        r"""Insert ``\\cmidrule``."""
        trim_str = f"({trim})" if trim else ""
        self._data_rows.append(
            rf"\cmidrule{trim_str}{{{start}-{end}}}"
        )
        return self

    def add_multirow(self, n_rows: int, content: str) -> str:
        r"""Return a ``\\multirow`` cell string."""
        return rf"\multirow{{{n_rows}}}{{*}}{{{content}}}"

    def add_multicolumn(
        self,
        n_cols: int,
        alignment: str,
        content: str,
    ) -> str:
        r"""Return a ``\\multicolumn`` cell string."""
        return rf"\multicolumn{{{n_cols}}}{{{alignment}}}{{{content}}}"

    def set_column_format(self, format_str: str) -> "LaTeXTable":
        """Override the column format string (e.g. ``lcccc``)."""
        self._column_format = format_str
        return self

    def highlight_cells(
        self,
        condition_fn: Callable[[int, int, str], Optional[str]],
    ) -> "LaTeXTable":
        """Apply formatting to cells based on *condition_fn*.

        ``condition_fn(row_idx, col_idx, cell_text)`` should return a
        replacement string or ``None`` to leave the cell unchanged.
        """
        for r_idx, row in enumerate(self._data_rows):
            if isinstance(row, str):
                continue
            for c_idx, cell in enumerate(row):
                replacement = condition_fn(r_idx, c_idx, cell)
                if replacement is not None:
                    row[c_idx] = replacement
        return self

    def add_footnote(self, text: str) -> "LaTeXTable":
        """Add a table footnote."""
        self._footnotes.append(text)
        return self

    # -- auto column format --------------------------------------------------

    def _auto_column_format(self) -> str:
        """Detect column count and generate a default column format."""
        if self._column_format:
            return self._column_format
        n_cols = 0
        if self._header_rows:
            n_cols = max(len(r) for r in self._header_rows)
        for row in self._data_rows:
            if isinstance(row, list):
                n_cols = max(n_cols, len(row))
        if n_cols == 0:
            return "l"
        if self.rows:
            fmt = "l" + "c" * (n_cols - 1)
        else:
            fmt = "l" + "c" * (n_cols - 1) if n_cols > 1 else "c"
        if self.config.vertical_lines:
            fmt = "|" + "|".join(fmt) + "|"
        return fmt

    # -- rendering -----------------------------------------------------------

    def render(self) -> str:
        """Render the complete LaTeX table as a string."""
        parts: List[str] = []

        # table/table* environment
        wrap_in_table = bool(self.config.caption or self.config.label)
        if wrap_in_table:
            env = "table*" if self.config.landscape else "table"
            parts.append(rf"\begin{{{env}}}[{self.config.position}]")
            if self.config.centering:
                parts.append(r"\centering")

        # font size
        if self.config.font_size != "normalsize":
            parts.append(rf"\{self.config.font_size}")

        # array stretch
        if self.config.arraystretch != 1.0:
            parts.append(
                rf"\renewcommand{{\arraystretch}}{{{self.config.arraystretch:.2f}}}"
            )
        if self.config.tabcolsep != 6.0:
            parts.append(
                rf"\setlength{{\tabcolsep}}{{{self.config.tabcolsep:.1f}pt}}"
            )

        col_fmt = self._auto_column_format()

        # resizebox open
        if self.config.resizebox:
            parts.append(
                rf"\resizebox{{{self.config.resizebox_width}}}{{!}}{{"
            )

        # tabular
        parts.append(rf"\begin{{tabular}}{{{col_fmt}}}")

        # top rule
        if self.config.booktabs:
            parts.append(r"\toprule")
        elif self.config.horizontal_lines:
            parts.append(r"\hline")

        # header rows
        for h_row in self._header_rows:
            formatted = self._render_header_row(h_row)
            parts.append(formatted)
            if self.config.booktabs:
                parts.append(r"\midrule")
            elif self.config.horizontal_lines:
                parts.append(r"\hline")

        # data rows
        for d_idx, d_row in enumerate(self._data_rows):
            if isinstance(d_row, str):
                parts.append(d_row)
            else:
                line = " & ".join(d_row) + r" \\"
                if (
                    self.config.alternating_row_colors
                    and d_idx % 2 == 1
                ):
                    line = (
                        rf"\rowcolor{{{self.config.alternating_color}}} "
                        + line
                    )
                parts.append(line)
                if self.config.horizontal_lines:
                    parts.append(r"\hline")

        # bottom rule
        if self.config.booktabs:
            parts.append(r"\bottomrule")
        elif self.config.horizontal_lines:
            parts.append(r"\hline")

        parts.append(r"\end{tabular}")

        # resizebox close
        if self.config.resizebox:
            parts.append("}")

        # caption and label
        if self.config.caption:
            parts.append(rf"\caption{{{self.config.caption}}}")
        if self.config.label:
            parts.append(rf"\label{{{self.config.label}}}")

        # footnotes
        if self._footnotes:
            notes = "; ".join(self._footnotes)
            parts.append(rf"\vspace{{2pt}}")
            parts.append(rf"\parbox{{\textwidth}}{{\footnotesize {notes}}}")

        if wrap_in_table:
            env = "table*" if self.config.landscape else "table"
            parts.append(rf"\end{{{env}}}")

        return "\n".join(parts)

    def _render_header_row(self, cells: List[str]) -> str:
        """Render a header row with optional bolding."""
        if self.config.header_bold:
            cells = [LaTeXEscaper.bold(c) for c in cells]
        return " & ".join(cells) + r" \\"

    def __str__(self) -> str:  # pragma: no cover
        return self.render()


# ---------------------------------------------------------------------------
# Comparison table generator
# ---------------------------------------------------------------------------

@dataclass
class _AlgorithmResult:
    """Convenience struct for algorithm results."""

    name: str
    metrics: Dict[str, Union[float, Tuple[float, float]]]


class ComparisonTableGenerator:
    """Generate common comparison tables for algorithm benchmarks."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)
        self._formatter = NumberFormatter()
        self._escaper = LaTeXEscaper()

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _find_best_and_second(
        values: List[Optional[float]],
        higher_is_better: bool = True,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return indices of the best and second-best values."""
        indexed: List[Tuple[int, float]] = [
            (i, v) for i, v in enumerate(values) if v is not None
        ]
        if not indexed:
            return None, None
        indexed.sort(key=lambda x: x[1], reverse=higher_is_better)
        best_idx = indexed[0][0]
        second_idx = indexed[1][0] if len(indexed) > 1 else None
        return best_idx, second_idx

    def _apply_highlight(
        self,
        text: str,
        rank: str,
    ) -> str:
        """Wrap *text* with the configured highlight for *rank* (best/second)."""
        if rank == "best":
            mode = self.config.highlight_best
        else:
            mode = self.config.highlight_second

        if mode == "bold":
            return LaTeXEscaper.bold(text)
        elif mode == "underline":
            return LaTeXEscaper.underline(text)
        elif mode == "italic":
            return LaTeXEscaper.italic(text)
        elif mode.startswith("color:"):
            color = mode.split(":", 1)[1]
            return LaTeXEscaper.color(text, color)
        return text

    def _format_metric_value(
        self,
        value: Any,
        precision: Optional[int] = None,
    ) -> str:
        """Format a metric that might be ``float`` or ``(mean, std)``."""
        prec = precision if precision is not None else self.config.precision
        if isinstance(value, tuple) and len(value) == 2:
            return self._formatter.format_mean_std(value[0], value[1], prec)
        if isinstance(value, float):
            if self.config.scientific_notation:
                return self._formatter.format_scientific(value, prec)
            return f"{value:.{prec}f}"
        if isinstance(value, int):
            return str(value)
        if value is None:
            return "---"
        return str(value)

    @staticmethod
    def _mean_of(value: Any) -> Optional[float]:
        """Extract the mean from a value or tuple."""
        if isinstance(value, tuple):
            return value[0]
        if isinstance(value, (int, float)):
            return float(value)
        return None

    # ----- algorithm comparison ---------------------------------------------

    def algorithm_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str],
        higher_is_better: Optional[Dict[str, bool]] = None,
        caption: str = "Algorithm comparison",
        label: str = "tab:algorithm-comparison",
    ) -> str:
        """Generate an algorithm × metric comparison table.

        Parameters
        ----------
        results : dict
            ``{algorithm_name: {metric_name: value_or_(mean,std), ...}, ...}``
        metrics : list of str
            Metric names in desired column order.
        higher_is_better : dict, optional
            ``{metric_name: bool}``.  Defaults to ``True`` for all.
        """
        if higher_is_better is None:
            higher_is_better = {m: True for m in metrics}

        algo_names = list(results.keys())
        n_algos = len(algo_names)

        # Collect mean values per metric for ranking
        metric_vals: Dict[str, List[Optional[float]]] = {
            m: [] for m in metrics
        }
        for algo in algo_names:
            for m in metrics:
                raw = results[algo].get(m)
                metric_vals[m].append(self._mean_of(raw))

        # Find best / second-best per metric
        best_idx: Dict[str, Optional[int]] = {}
        second_idx: Dict[str, Optional[int]] = {}
        for m in metrics:
            hib = higher_is_better.get(m, True)
            b, s = self._find_best_and_second(metric_vals[m], hib)
            best_idx[m] = b
            second_idx[m] = s

        # Build table rows
        header = ["Algorithm"] + [self._escaper.escape(m) for m in metrics]
        col_fmt = "l" + "c" * len(metrics)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
            precision=self.config.precision,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        for a_idx, algo in enumerate(algo_names):
            row: List[str] = [self._escaper.escape(algo)]
            for m in metrics:
                raw = results[algo].get(m)
                cell = self._format_metric_value(raw)
                if a_idx == best_idx.get(m):
                    cell = self._apply_highlight(cell, "best")
                elif a_idx == second_idx.get(m):
                    cell = self._apply_highlight(cell, "second")
                row.append(cell)
            table.add_data_row(row)

        return table.render()

    # ----- task comparison --------------------------------------------------

    def task_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        tasks: List[str],
        metric_name: str = "score",
        higher_is_better: bool = True,
        caption: str = "Task comparison",
        label: str = "tab:task-comparison",
    ) -> str:
        """Generate an algorithm × task comparison table.

        Parameters
        ----------
        results : dict
            ``{algorithm: {task: value_or_(mean,std), ...}, ...}``
        tasks : list of str
            Task names in column order.
        """
        algo_names = list(results.keys())

        # per-task ranking
        task_vals: Dict[str, List[Optional[float]]] = {t: [] for t in tasks}
        for algo in algo_names:
            for t in tasks:
                task_vals[t].append(self._mean_of(results[algo].get(t)))

        best: Dict[str, Optional[int]] = {}
        second: Dict[str, Optional[int]] = {}
        for t in tasks:
            b, s = self._find_best_and_second(task_vals[t], higher_is_better)
            best[t] = b
            second[t] = s

        header = ["Algorithm"] + [self._escaper.escape(t) for t in tasks]
        col_fmt = "l" + "c" * len(tasks)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
            precision=self.config.precision,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        for a_idx, algo in enumerate(algo_names):
            row: List[str] = [self._escaper.escape(algo)]
            for t in tasks:
                raw = results[algo].get(t)
                cell = self._format_metric_value(raw)
                if a_idx == best.get(t):
                    cell = self._apply_highlight(cell, "best")
                elif a_idx == second.get(t):
                    cell = self._apply_highlight(cell, "second")
                row.append(cell)
            table.add_data_row(row)

        return table.render()

    # ----- hyperparameter table ---------------------------------------------

    def hyperparameter_table(
        self,
        sweep_results: List[Dict[str, Any]],
        param_names: Optional[List[str]] = None,
        metric_name: str = "score",
        caption: str = "Hyperparameter sweep results",
        label: str = "tab:hyperparameters",
    ) -> str:
        """Generate a table from a hyperparameter sweep.

        Parameters
        ----------
        sweep_results : list of dict
            Each dict has parameter values and a metric value, e.g.
            ``{"lr": 0.01, "batch_size": 32, "score": 0.85}``.
        param_names : list of str, optional
            Parameter names to display.  Auto-detected if omitted.
        metric_name : str
            Key used for the metric column.
        """
        if not sweep_results:
            return ""

        if param_names is None:
            param_names = [
                k for k in sweep_results[0].keys() if k != metric_name
            ]

        header = [self._escaper.escape(p) for p in param_names] + [
            self._escaper.escape(metric_name)
        ]
        col_fmt = "c" * len(param_names) + "c"

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
            precision=self.config.precision,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        # Sort by metric descending
        sorted_results = sorted(
            sweep_results,
            key=lambda r: (
                self._mean_of(r.get(metric_name, 0)) or 0.0
            ),
            reverse=True,
        )

        metric_values: List[Optional[float]] = [
            self._mean_of(r.get(metric_name)) for r in sorted_results
        ]
        best_i, second_i = self._find_best_and_second(metric_values)

        for idx, result in enumerate(sorted_results):
            row: List[str] = []
            for p in param_names:
                row.append(self._format_metric_value(result.get(p)))
            metric_cell = self._format_metric_value(
                result.get(metric_name)
            )
            if idx == best_i:
                metric_cell = self._apply_highlight(metric_cell, "best")
            elif idx == second_i:
                metric_cell = self._apply_highlight(metric_cell, "second")
            row.append(metric_cell)
            table.add_data_row(row)

        return table.render()

    # ----- statistical significance -----------------------------------------

    def statistical_significance_table(
        self,
        p_values: Dict[str, Dict[str, float]],
        effect_sizes: Optional[Dict[str, Dict[str, float]]] = None,
        caption: str = "Statistical significance",
        label: str = "tab:significance",
    ) -> str:
        r"""Generate a table of pairwise statistical significance.

        Parameters
        ----------
        p_values : dict
            ``{algo_a: {algo_b: p_value, ...}, ...}``
        effect_sizes : dict, optional
            ``{algo_a: {algo_b: cohens_d, ...}, ...}``

        Significance markers: ``*`` p < 0.05, ``**`` p < 0.01,
        ``***`` p < 0.001.
        """
        algorithms = list(p_values.keys())
        n = len(algorithms)

        header = [""] + [self._escaper.escape(a) for a in algorithms]
        col_fmt = "l" + "c" * n

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        footnote_parts: List[str] = [
            r"$^{*}$ $p < 0.05$",
            r"$^{**}$ $p < 0.01$",
            r"$^{***}$ $p < 0.001$",
        ]

        for a in algorithms:
            row: List[str] = [self._escaper.escape(a)]
            for b in algorithms:
                if a == b:
                    row.append("---")
                    continue
                p = p_values.get(a, {}).get(b)
                if p is None:
                    row.append("---")
                    continue

                # significance marker
                if p < 0.001:
                    marker = "***"
                elif p < 0.01:
                    marker = "**"
                elif p < 0.05:
                    marker = "*"
                else:
                    marker = ""

                if effect_sizes and a in effect_sizes and b in effect_sizes[a]:
                    d = effect_sizes[a][b]
                    cell = f"{d:.2f}"
                else:
                    cell = NumberFormatter.format_pvalue(p)

                if marker:
                    cell = rf"{cell}$^{{{marker}}}$"
                row.append(cell)
            table.add_data_row(row)

        table.add_footnote("; ".join(footnote_parts))
        return table.render()

    # ----- Pareto table -----------------------------------------------------

    def pareto_table(
        self,
        frontier_points: List[Dict[str, Any]],
        objective_names: Optional[List[str]] = None,
        caption: str = "Pareto frontier",
        label: str = "tab:pareto",
    ) -> str:
        """Generate a table of Pareto frontier points.

        Parameters
        ----------
        frontier_points : list of dict
            Each dict has ``"name"`` and objective values, e.g.
            ``{"name": "Config A", "quality": 0.9, "diversity": 0.7}``.
        objective_names : list of str, optional
            Objectives to show.  Auto-detected if omitted.
        """
        if not frontier_points:
            return ""

        if objective_names is None:
            objective_names = [
                k for k in frontier_points[0].keys() if k != "name"
            ]

        has_name = "name" in frontier_points[0]
        header_parts: List[str] = []
        if has_name:
            header_parts.append("Configuration")
        header_parts.append("Rank")
        header_parts += [self._escaper.escape(o) for o in objective_names]
        header_parts.append("Trade-off")

        n_cols = len(header_parts)
        col_fmt = "l" * (1 if has_name else 0) + "c" * (n_cols - (1 if has_name else 0))

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header_parts)

        for rank, point in enumerate(frontier_points, 1):
            row: List[str] = []
            if has_name:
                row.append(self._escaper.escape(str(point.get("name", ""))))
            row.append(str(rank))
            obj_vals: List[float] = []
            for o in objective_names:
                val = point.get(o, 0.0)
                obj_vals.append(float(val) if val is not None else 0.0)
                row.append(self._format_metric_value(val))
            # Simple trade-off indicator (geometric mean normalised to [0,1])
            if obj_vals:
                prod = 1.0
                for v in obj_vals:
                    prod *= max(v, 1e-12)
                geo_mean = prod ** (1.0 / len(obj_vals))
                row.append(f"{geo_mean:.{self.config.precision}f}")
            else:
                row.append("---")
            table.add_data_row(row)

        return table.render()

    # ----- metric correlation -----------------------------------------------

    def metric_correlation_table(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        caption: str = "Metric correlations",
        label: str = "tab:correlations",
    ) -> str:
        """Generate a correlation matrix table with colour coding.

        Parameters
        ----------
        correlation_matrix : dict
            ``{metric_a: {metric_b: r, ...}, ...}``
        """
        metrics = list(correlation_matrix.keys())
        n = len(metrics)

        header = [""] + [self._escaper.escape(m) for m in metrics]
        col_fmt = "l" + "c" * n

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        for m_a in metrics:
            row: List[str] = [self._escaper.escape(m_a)]
            for m_b in metrics:
                r = correlation_matrix.get(m_a, {}).get(m_b, 0.0)
                # Color intensity by |r|
                abs_r = abs(r)
                if abs_r > 0.7:
                    cell = LaTeXEscaper.bold(f"{r:.2f}")
                elif abs_r > 0.4:
                    cell = f"{r:.2f}"
                else:
                    cell = LaTeXEscaper.color(f"{r:.2f}", "gray")
                if m_a == m_b:
                    cell = "1.00"
                row.append(cell)
            table.add_data_row(row)

        return table.render()

    # ----- ablation study ---------------------------------------------------

    def ablation_table(
        self,
        base_name: str,
        base_results: Dict[str, Any],
        ablations: Dict[str, Dict[str, Any]],
        metrics: List[str],
        higher_is_better: Optional[Dict[str, bool]] = None,
        caption: str = "Ablation study",
        label: str = "tab:ablation",
    ) -> str:
        """Generate an ablation study table showing deltas from a baseline.

        Parameters
        ----------
        base_name : str
            Name of the full model / baseline.
        base_results : dict
            ``{metric: value, ...}`` for the baseline.
        ablations : dict
            ``{variant_name: {metric: value, ...}, ...}``
        metrics : list of str
            Which metrics to show.
        higher_is_better : dict, optional
            ``{metric: bool}``.
        """
        if higher_is_better is None:
            higher_is_better = {m: True for m in metrics}

        header = ["Variant"] + [self._escaper.escape(m) for m in metrics] + [
            r"$\Delta$ Avg"
        ]
        col_fmt = "l" + "c" * (len(metrics) + 1)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        # Baseline row
        base_row: List[str] = [
            LaTeXEscaper.bold(self._escaper.escape(base_name))
        ]
        for m in metrics:
            base_row.append(
                LaTeXEscaper.bold(self._format_metric_value(base_results.get(m)))
            )
        base_row.append("---")
        table.add_data_row(base_row)
        table.add_midrule()

        # Ablation rows
        for var_name, var_results in ablations.items():
            row: List[str] = [self._escaper.escape(var_name)]
            deltas: List[float] = []
            for m in metrics:
                base_val = self._mean_of(base_results.get(m))
                var_val = self._mean_of(var_results.get(m))
                cell = self._format_metric_value(var_results.get(m))
                if base_val is not None and var_val is not None:
                    delta = var_val - base_val
                    hib = higher_is_better.get(m, True)
                    if (hib and delta < 0) or (not hib and delta > 0):
                        # Performance degraded
                        sign = "-" if delta < 0 else "+"
                        cell = LaTeXEscaper.color(
                            f"{cell} ({sign}{abs(delta):.{self.config.precision}f})",
                            "red",
                        )
                    deltas.append(delta)
                row.append(cell)
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                row.append(f"{avg_delta:+.{self.config.precision}f}")
            else:
                row.append("---")
            table.add_data_row(row)

        return table.render()

    # ----- runtime comparison -----------------------------------------------

    def runtime_table(
        self,
        results: Dict[str, Dict[str, Any]],
        caption: str = "Runtime comparison",
        label: str = "tab:runtime",
    ) -> str:
        """Generate a runtime comparison table.

        Parameters
        ----------
        results : dict
            ``{algorithm: {"time_s": float, "memory_mb": float,
            "throughput": float}, ...}``
        """
        algo_names = list(results.keys())
        possible_cols = ["time_s", "memory_mb", "throughput", "flops"]
        present_cols: List[str] = []
        for c in possible_cols:
            if any(c in results[a] for a in algo_names):
                present_cols.append(c)

        col_labels = {
            "time_s": "Time (s)",
            "memory_mb": "Memory (MB)",
            "throughput": "Throughput",
            "flops": "FLOPs",
        }

        header = ["Algorithm"] + [col_labels.get(c, c) for c in present_cols]
        col_fmt = "l" + "c" * len(present_cols)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        # Determine best (lowest for time/memory, highest for throughput)
        lower_better = {"time_s", "memory_mb", "flops"}
        col_best: Dict[str, Optional[int]] = {}
        for c in present_cols:
            vals = [self._mean_of(results[a].get(c)) for a in algo_names]
            hib = c not in lower_better
            b, _ = self._find_best_and_second(vals, hib)
            col_best[c] = b

        for a_idx, algo in enumerate(algo_names):
            row: List[str] = [self._escaper.escape(algo)]
            for c in present_cols:
                val = results[algo].get(c)
                cell = self._format_metric_value(val)
                if a_idx == col_best.get(c):
                    cell = self._apply_highlight(cell, "best")
                row.append(cell)
            table.add_data_row(row)

        return table.render()


# ---------------------------------------------------------------------------
# LaTeX figure helpers
# ---------------------------------------------------------------------------

class LaTeXFigure:
    """Helpers for generating LaTeX figure environments."""

    @staticmethod
    def wrap_in_figure(
        content: str,
        caption: str = "",
        label: str = "",
        position: str = "htbp",
        centering: bool = True,
    ) -> str:
        r"""Wrap *content* in a ``figure`` environment."""
        lines: List[str] = [rf"\begin{{figure}}[{position}]"]
        if centering:
            lines.append(r"\centering")
        lines.append(content)
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{figure}")
        return "\n".join(lines)

    @staticmethod
    def subfigure(
        contents: List[str],
        captions: List[str],
        layout: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        overall_caption: str = "",
        overall_label: str = "",
        position: str = "htbp",
    ) -> str:
        r"""Generate a figure with ``subfigure`` environments.

        Parameters
        ----------
        contents : list of str
            LaTeX content for each subfigure (e.g. ``\\includegraphics``).
        captions : list of str
            Caption for each subfigure.
        layout : list of float, optional
            Width fractions for each subfigure (should sum to ≤ 1).
            Defaults to equal widths.
        labels : list of str, optional
            Labels for each subfigure.
        """
        n = len(contents)
        if layout is None:
            w = round(0.95 / n, 2)
            layout = [w] * n
        if labels is None:
            labels = [""] * n

        lines: List[str] = [rf"\begin{{figure}}[{position}]", r"\centering"]
        for i in range(n):
            width = layout[i]
            lines.append(
                rf"\begin{{subfigure}}[b]{{{width}\textwidth}}"
            )
            lines.append(r"\centering")
            lines.append(contents[i])
            if captions[i]:
                lines.append(rf"\caption{{{captions[i]}}}")
            if labels[i]:
                lines.append(rf"\label{{{labels[i]}}}")
            lines.append(r"\end{subfigure}")
            if i < n - 1:
                lines.append(r"\hfill")
        if overall_caption:
            lines.append(rf"\caption{{{overall_caption}}}")
        if overall_label:
            lines.append(rf"\label{{{overall_label}}}")
        lines.append(r"\end{figure}")
        return "\n".join(lines)

    @staticmethod
    def minipage(
        content: str,
        width: float = 0.48,
        position: str = "t",
    ) -> str:
        r"""Wrap *content* in a ``minipage`` environment."""
        lines: List[str] = [
            rf"\begin{{minipage}}[{position}]{{{width}\textwidth}}",
            content,
            r"\end{minipage}",
        ]
        return "\n".join(lines)

    @staticmethod
    def includegraphics(
        path: str,
        width: str = r"\textwidth",
        options: str = "",
    ) -> str:
        r"""Return an ``\\includegraphics`` command."""
        opts = f"width={width}"
        if options:
            opts = f"{opts},{options}"
        return rf"\includegraphics[{opts}]{{{path}}}"

    @staticmethod
    def side_by_side(
        left_content: str,
        right_content: str,
        left_width: float = 0.48,
        right_width: float = 0.48,
    ) -> str:
        """Place two blocks side by side using minipages."""
        left = LaTeXFigure.minipage(left_content, left_width)
        right = LaTeXFigure.minipage(right_content, right_width)
        return f"{left}\n\\hfill\n{right}"


# ---------------------------------------------------------------------------
# Full document assembly
# ---------------------------------------------------------------------------

class LaTeXDocument:
    """Assemble a complete LaTeX document from sections, tables, figures."""

    def __init__(
        self,
        title: str = "",
        author: str = "",
        document_class: str = "article",
        class_options: str = "11pt,a4paper",
        packages: Optional[List[str]] = None,
    ) -> None:
        self.title = title
        self.author = author
        self.document_class = document_class
        self.class_options = class_options
        self._packages: List[str] = list(packages) if packages else []
        self._body: List[str] = []
        self._extra_preamble: List[str] = []

    def add_section(self, title: str) -> "LaTeXDocument":
        """Add a section heading."""
        self._body.append(rf"\section{{{title}}}")
        return self

    def add_subsection(self, title: str) -> "LaTeXDocument":
        """Add a subsection heading."""
        self._body.append(rf"\subsection{{{title}}}")
        return self

    def add_table(self, table: Union[LaTeXTable, str]) -> "LaTeXDocument":
        """Add a table (rendered ``LaTeXTable`` or raw string)."""
        if isinstance(table, LaTeXTable):
            self._body.append(table.render())
        else:
            self._body.append(table)
        return self

    def add_figure(self, figure: str) -> "LaTeXDocument":
        """Add a figure (rendered LaTeX string)."""
        self._body.append(figure)
        return self

    def add_text(self, text: str) -> "LaTeXDocument":
        """Add free-form text."""
        self._body.append(text)
        return self

    def add_raw(self, latex: str) -> "LaTeXDocument":
        """Add raw LaTeX content."""
        self._body.append(latex)
        return self

    def add_preamble_line(self, line: str) -> "LaTeXDocument":
        """Add an extra preamble line."""
        self._extra_preamble.append(line)
        return self

    def add_clearpage(self) -> "LaTeXDocument":
        r"""Insert a ``\\clearpage``."""
        self._body.append(r"\clearpage")
        return self

    def add_newpage(self) -> "LaTeXDocument":
        r"""Insert a ``\\newpage``."""
        self._body.append(r"\newpage")
        return self

    # -- preamble ------------------------------------------------------------

    def get_preamble(self) -> str:
        """Return the required preamble (packages, colours, etc.)."""
        default_packages = [
            ("booktabs", ""),
            ("multirow", ""),
            ("multicol", ""),
            ("graphicx", ""),
            ("xcolor", "table"),
            ("amsmath", ""),
            ("amssymb", ""),
            ("subcaption", ""),
            ("geometry", "margin=1in"),
            ("hyperref", "colorlinks=true"),
            ("array", ""),
        ]
        lines: List[str] = []
        seen: set = set()
        for pkg, opts in default_packages:
            if pkg not in seen:
                if opts:
                    lines.append(rf"\usepackage[{opts}]{{{pkg}}}")
                else:
                    lines.append(rf"\usepackage{{{pkg}}}")
                seen.add(pkg)
        for pkg in self._packages:
            if pkg not in seen:
                lines.append(rf"\usepackage{{{pkg}}}")
                seen.add(pkg)
        lines.append("")
        lines.append(ColorDefinitions.define_colors())
        for extra in self._extra_preamble:
            lines.append(extra)
        return "\n".join(lines)

    # -- rendering -----------------------------------------------------------

    def render(self) -> str:
        """Render the full LaTeX document."""
        lines: List[str] = []
        lines.append(
            rf"\documentclass[{self.class_options}]{{{self.document_class}}}"
        )
        lines.append("")
        lines.append(self.get_preamble())
        lines.append("")

        if self.title:
            lines.append(rf"\title{{{self.title}}}")
        if self.author:
            lines.append(rf"\author{{{self.author}}}")
        lines.append("")
        lines.append(r"\begin{document}")
        if self.title:
            lines.append(r"\maketitle")
        lines.append("")

        for block in self._body:
            lines.append(block)
            lines.append("")

        lines.append(r"\end{document}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.render()


# ---------------------------------------------------------------------------
# Multi-table report builder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """High-level helper to assemble a full experimental report."""

    def __init__(
        self,
        title: str = "Experimental Results",
        author: str = "",
    ) -> None:
        self.doc = LaTeXDocument(title=title, author=author)
        self._comp_gen = ComparisonTableGenerator()

    def add_algorithm_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str],
        higher_is_better: Optional[Dict[str, bool]] = None,
        section_title: str = "Algorithm Comparison",
    ) -> "ReportBuilder":
        """Add an algorithm comparison section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.algorithm_comparison(
            results, metrics, higher_is_better,
        )
        self.doc.add_raw(table_str)
        return self

    def add_task_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        tasks: List[str],
        section_title: str = "Task Comparison",
    ) -> "ReportBuilder":
        """Add a task comparison section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.task_comparison(results, tasks)
        self.doc.add_raw(table_str)
        return self

    def add_significance_analysis(
        self,
        p_values: Dict[str, Dict[str, float]],
        effect_sizes: Optional[Dict[str, Dict[str, float]]] = None,
        section_title: str = "Statistical Significance",
    ) -> "ReportBuilder":
        """Add a significance analysis section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.statistical_significance_table(
            p_values, effect_sizes,
        )
        self.doc.add_raw(table_str)
        return self

    def add_pareto_analysis(
        self,
        frontier_points: List[Dict[str, Any]],
        section_title: str = "Pareto Analysis",
    ) -> "ReportBuilder":
        """Add a Pareto frontier section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.pareto_table(frontier_points)
        self.doc.add_raw(table_str)
        return self

    def add_correlation_analysis(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        section_title: str = "Metric Correlations",
    ) -> "ReportBuilder":
        """Add a correlation matrix section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.metric_correlation_table(
            correlation_matrix,
        )
        self.doc.add_raw(table_str)
        return self

    def add_ablation_study(
        self,
        base_name: str,
        base_results: Dict[str, Any],
        ablations: Dict[str, Dict[str, Any]],
        metrics: List[str],
        higher_is_better: Optional[Dict[str, bool]] = None,
        section_title: str = "Ablation Study",
    ) -> "ReportBuilder":
        """Add an ablation study section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.ablation_table(
            base_name, base_results, ablations, metrics, higher_is_better,
        )
        self.doc.add_raw(table_str)
        return self

    def add_hyperparameter_sweep(
        self,
        sweep_results: List[Dict[str, Any]],
        param_names: Optional[List[str]] = None,
        metric_name: str = "score",
        section_title: str = "Hyperparameter Sweep",
    ) -> "ReportBuilder":
        """Add a hyperparameter sweep section."""
        self.doc.add_section(section_title)
        table_str = self._comp_gen.hyperparameter_table(
            sweep_results, param_names, metric_name,
        )
        self.doc.add_raw(table_str)
        return self

    def add_custom_table(
        self,
        table: Union[LaTeXTable, str],
        section_title: str = "",
    ) -> "ReportBuilder":
        """Add a custom table, optionally under a new section."""
        if section_title:
            self.doc.add_section(section_title)
        self.doc.add_table(table)
        return self

    def add_figure(
        self,
        image_path: str,
        caption: str,
        label: str = "",
        width: str = r"0.8\textwidth",
        section_title: str = "",
    ) -> "ReportBuilder":
        """Add a figure section."""
        if section_title:
            self.doc.add_section(section_title)
        fig = LaTeXFigure.wrap_in_figure(
            LaTeXFigure.includegraphics(image_path, width=width),
            caption=caption,
            label=label,
        )
        self.doc.add_figure(fig)
        return self

    def render(self) -> str:
        """Render the complete document."""
        return self.doc.render()


# ---------------------------------------------------------------------------
# Specialised table builders
# ---------------------------------------------------------------------------

class DiversityMetricsTable:
    """Tables specific to diversity decoding evaluation."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)
        self._comp = ComparisonTableGenerator(config=self.config)
        self._fmt = NumberFormatter()

    def diversity_quality_tradeoff(
        self,
        results: Dict[str, Dict[str, Any]],
        diversity_metrics: List[str],
        quality_metrics: List[str],
        caption: str = "Diversity--quality trade-off",
        label: str = "tab:div-quality",
    ) -> str:
        """Multi-header table grouping diversity and quality metrics.

        Parameters
        ----------
        results : dict
            ``{algorithm: {metric: value_or_(mean,std), ...}, ...}``
        diversity_metrics : list of str
            Diversity metric names.
        quality_metrics : list of str
            Quality metric names.
        """
        all_metrics = diversity_metrics + quality_metrics
        n_div = len(diversity_metrics)
        n_qual = len(quality_metrics)
        algo_names = list(results.keys())

        col_fmt = "l" + "c" * len(all_metrics)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)

        # Group header
        group_row: List[str] = [
            "",
            table.add_multicolumn(n_div, "c", LaTeXEscaper.bold("Diversity")),
        ]
        # fill remaining diversity columns with empty
        for _ in range(n_div - 1):
            group_row.append("")
        group_row.append(
            table.add_multicolumn(n_qual, "c", LaTeXEscaper.bold("Quality"))
        )
        for _ in range(n_qual - 1):
            group_row.append("")
        # Manually build group header line
        group_line = (
            " & "
            + table.add_multicolumn(n_div, "c", LaTeXEscaper.bold("Diversity"))
            + " & "
            + table.add_multicolumn(n_qual, "c", LaTeXEscaper.bold("Quality"))
            + r" \\"
        )
        # We'll build custom rows
        lines: List[str] = []
        lines.append(rf"\begin{{table}}[{cfg.position}]")
        lines.append(r"\centering")
        lines.append(rf"\begin{{tabular}}{{{col_fmt}}}")
        lines.append(r"\toprule")
        lines.append(group_line)
        # cmidrules
        lines.append(
            rf"\cmidrule(lr){{2-{n_div + 1}}} "
            rf"\cmidrule(lr){{{n_div + 2}-{n_div + n_qual + 1}}}"
        )
        # Sub-header
        sub_header = ["Algorithm"]
        for m in all_metrics:
            sub_header.append(LaTeXEscaper.bold(LaTeXEscaper.escape(m)))
        lines.append(" & ".join(sub_header) + r" \\")
        lines.append(r"\midrule")

        # Ranking per metric
        metric_vals: Dict[str, List[Optional[float]]] = {}
        for m in all_metrics:
            metric_vals[m] = [
                self._comp._mean_of(results[a].get(m))
                for a in algo_names
            ]
        best_idx: Dict[str, Optional[int]] = {}
        second_idx: Dict[str, Optional[int]] = {}
        for m in all_metrics:
            b, s = self._comp._find_best_and_second(metric_vals[m], True)
            best_idx[m] = b
            second_idx[m] = s

        for a_idx, algo in enumerate(algo_names):
            row_parts: List[str] = [LaTeXEscaper.escape(algo)]
            for m in all_metrics:
                raw = results[algo].get(m)
                cell = self._comp._format_metric_value(raw)
                if a_idx == best_idx.get(m):
                    cell = self._comp._apply_highlight(cell, "best")
                elif a_idx == second_idx.get(m):
                    cell = self._comp._apply_highlight(cell, "second")
                row_parts.append(cell)
            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if cfg.caption:
            lines.append(rf"\caption{{{cfg.caption}}}")
        if cfg.label:
            lines.append(rf"\label{{{cfg.label}}}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def ngram_diversity_table(
        self,
        results: Dict[str, Dict[int, float]],
        n_values: Optional[List[int]] = None,
        caption: str = "N-gram diversity",
        label: str = "tab:ngram-diversity",
    ) -> str:
        """Table of n-gram diversity scores.

        Parameters
        ----------
        results : dict
            ``{algorithm: {n: diversity_score, ...}, ...}``
        n_values : list of int, optional
            Which n values to show.  Auto-detected if omitted.
        """
        algo_names = list(results.keys())
        if n_values is None:
            all_ns: set = set()
            for v in results.values():
                all_ns.update(v.keys())
            n_values = sorted(all_ns)

        header = ["Algorithm"] + [f"$n={n}$" for n in n_values]
        col_fmt = "l" + "c" * len(n_values)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        # Rank per n
        n_best: Dict[int, Optional[int]] = {}
        for n in n_values:
            vals = [results[a].get(n) for a in algo_names]
            b, _ = self._comp._find_best_and_second(
                [v if v is not None else None for v in vals], True
            )
            n_best[n] = b

        for a_idx, algo in enumerate(algo_names):
            row: List[str] = [LaTeXEscaper.escape(algo)]
            for n in n_values:
                val = results[algo].get(n)
                cell = f"{val:.{self.config.precision}f}" if val is not None else "---"
                if a_idx == n_best.get(n):
                    cell = LaTeXEscaper.bold(cell)
                row.append(cell)
            table.add_data_row(row)

        return table.render()

    def sample_examples_table(
        self,
        prompt: str,
        samples: Dict[str, List[str]],
        caption: str = "Generated samples",
        label: str = "tab:samples",
        max_samples: int = 3,
    ) -> str:
        """Table showing example generations from each algorithm.

        Parameters
        ----------
        prompt : str
            The input prompt.
        samples : dict
            ``{algorithm: [sample1, sample2, ...], ...}``
        """
        algo_names = list(samples.keys())

        lines: List[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\small")
        col_fmt = "l" + "p{10cm}"
        lines.append(rf"\begin{{tabular}}{{{col_fmt}}}")
        lines.append(r"\toprule")
        lines.append(
            rf"\multicolumn{{2}}{{l}}{{\textbf{{Prompt:}} "
            rf"\textit{{{LaTeXEscaper.escape(prompt)}}}}} \\"
        )
        lines.append(r"\midrule")
        lines.append(
            LaTeXEscaper.bold("Algorithm")
            + " & "
            + LaTeXEscaper.bold("Samples")
            + r" \\"
        )
        lines.append(r"\midrule")

        for algo in algo_names:
            algo_samples = samples[algo][:max_samples]
            n = len(algo_samples)
            if n == 0:
                continue
            first_sample = LaTeXEscaper.escape(algo_samples[0])
            row = (
                rf"\multirow{{{n}}}{{*}}{{{LaTeXEscaper.escape(algo)}}}"
                rf" & {first_sample} \\"
            )
            lines.append(row)
            for s in algo_samples[1:]:
                lines.append(rf" & {LaTeXEscaper.escape(s)} \\")
            lines.append(r"\midrule")

        # Remove last midrule, replace with bottomrule
        if lines and lines[-1] == r"\midrule":
            lines[-1] = r"\bottomrule"

        lines.append(r"\end{tabular}")
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{table}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table styling and post-processing
# ---------------------------------------------------------------------------

class TableStyler:
    """Post-process rendered LaTeX tables for styling adjustments."""

    @staticmethod
    def add_alternating_colors(
        latex: str,
        color_name: str = "rowalt",
    ) -> str:
        r"""Insert ``\\rowcolor`` on alternating data rows."""
        lines = latex.split("\n")
        result: List[str] = []
        in_data = False
        row_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped in (r"\midrule", r"\hline"):
                in_data = True
                row_count = 0
                result.append(line)
                continue
            if stripped in (r"\bottomrule", r"\end{tabular}"):
                in_data = False
                result.append(line)
                continue
            if in_data and "&" in line and row_count % 2 == 1:
                result.append(rf"\rowcolor{{{color_name}}} {line}")
            else:
                result.append(line)
            if in_data and "&" in line:
                row_count += 1
        return "\n".join(result)

    @staticmethod
    def rotate_headers(
        latex: str,
        angle: int = 45,
    ) -> str:
        r"""Rotate column headers using ``\\rotatebox``."""
        lines = latex.split("\n")
        result: List[str] = []
        header_processed = False
        for line in lines:
            stripped = line.strip()
            if (
                not header_processed
                and "&" in line
                and r"\\" in line
                and r"\textbf" in line
            ):
                # This looks like a header row
                cells = line.rstrip(r" \\").split(" & ")
                rotated = []
                for i, cell in enumerate(cells):
                    cell = cell.strip()
                    if i == 0:
                        rotated.append(cell)
                    else:
                        rotated.append(
                            rf"\rotatebox{{{angle}}}{{{cell}}}"
                        )
                result.append(" & ".join(rotated) + r" \\")
                header_processed = True
            else:
                result.append(line)
        return "\n".join(result)

    @staticmethod
    def add_column_spacing(
        latex: str,
        spacing_pt: float = 8.0,
    ) -> str:
        r"""Insert a ``\\setlength{\\tabcolsep}`` before the tabular."""
        insert = rf"\setlength{{\tabcolsep}}{{{spacing_pt:.1f}pt}}"
        return latex.replace(
            r"\begin{tabular}",
            insert + "\n" + r"\begin{tabular}",
            1,
        )

    @staticmethod
    def wrap_in_landscape(latex: str) -> str:
        r"""Wrap the table in a ``landscape`` environment."""
        return (
            r"\begin{landscape}" + "\n"
            + latex + "\n"
            + r"\end{landscape}"
        )

    @staticmethod
    def wrap_in_sidewaystable(latex: str) -> str:
        """Replace ``table`` with ``sidewaystable``."""
        result = latex.replace(r"\begin{table}", r"\begin{sidewaystable}")
        result = result.replace(r"\end{table}", r"\end{sidewaystable}")
        return result


# ---------------------------------------------------------------------------
# Standalone rendering helpers
# ---------------------------------------------------------------------------

def render_simple_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    booktabs: bool = True,
) -> str:
    """Quick helper to render a simple table from headers and rows."""
    cfg = LaTeXTableConfig(
        caption=caption,
        label=label,
        booktabs=booktabs,
    )
    table = LaTeXTable(config=cfg)
    table.add_header_row(headers)
    for row in rows:
        table.add_data_row(row)
    return table.render()


def render_key_value_table(
    data: Dict[str, Any],
    caption: str = "",
    label: str = "",
) -> str:
    """Render a two-column key–value table."""
    cfg = LaTeXTableConfig(
        caption=caption,
        label=label,
        booktabs=True,
        column_format="lc",
    )
    table = LaTeXTable(config=cfg)
    table.add_header_row(["Parameter", "Value"])
    for key, val in data.items():
        table.add_data_row([
            LaTeXEscaper.escape(str(key)),
            LaTeXEscaper.escape(str(val)),
        ])
    return table.render()


def render_matrix_table(
    matrix: List[List[float]],
    row_labels: List[str],
    col_labels: List[str],
    precision: int = 3,
    caption: str = "",
    label: str = "",
    colorize: bool = False,
) -> str:
    """Render a numeric matrix as a table, optionally with cell colours."""
    n_cols = len(col_labels)
    col_fmt = "l" + "c" * n_cols

    cfg = LaTeXTableConfig(
        caption=caption,
        label=label,
        booktabs=True,
        column_format=col_fmt,
        precision=precision,
    )
    table = LaTeXTable(config=cfg)
    table.add_header_row([""] + col_labels)

    # Compute global min/max for colouring
    flat = [v for row in matrix for v in row]
    min_val = min(flat) if flat else 0.0
    max_val = max(flat) if flat else 1.0

    for r_idx, row_label in enumerate(row_labels):
        row: List[str] = [LaTeXEscaper.escape(row_label)]
        for c_idx in range(n_cols):
            val = matrix[r_idx][c_idx]
            cell = f"{val:.{precision}f}"
            if colorize:
                color_cmd = ColorDefinitions.cell_color(
                    val, min_val, max_val, "heat"
                )
                cell = f"{color_cmd} {cell}"
            row.append(cell)
        table.add_data_row(row)

    return table.render()


# ---------------------------------------------------------------------------
# Batch export utilities
# ---------------------------------------------------------------------------

class BatchExporter:
    """Export multiple tables to individual .tex files."""

    def __init__(self, output_dir: str = ".") -> None:
        self.output_dir = output_dir
        self._tables: Dict[str, str] = {}

    def add(self, name: str, latex: str) -> "BatchExporter":
        """Register a table for export."""
        self._tables[name] = latex
        return self

    def get_input_commands(self) -> str:
        r"""Return ``\\input{...}`` commands for all registered tables."""
        lines: List[str] = []
        for name in self._tables:
            lines.append(rf"\input{{{self.output_dir}/{name}}}")
        return "\n".join(lines)

    def export(self) -> Dict[str, str]:
        """Return ``{filename: latex_content}`` for writing to disk."""
        import os

        result: Dict[str, str] = {}
        for name, latex in self._tables.items():
            fname = os.path.join(self.output_dir, f"{name}.tex")
            result[fname] = latex
        return result

    def write(self) -> List[str]:
        """Write all tables to disk and return the file paths."""
        import os

        paths: List[str] = []
        os.makedirs(self.output_dir, exist_ok=True)
        for name, latex in self._tables.items():
            fpath = os.path.join(self.output_dir, f"{name}.tex")
            with open(fpath, "w") as f:
                f.write(latex)
            paths.append(fpath)
        return paths


# ---------------------------------------------------------------------------
# Cross-referencing helpers
# ---------------------------------------------------------------------------

class CrossRef:
    """Helpers for LaTeX cross-referencing."""

    @staticmethod
    def ref(label: str) -> str:
        r"""Return ``\\ref{label}``."""
        return rf"\ref{{{label}}}"

    @staticmethod
    def eqref(label: str) -> str:
        r"""Return ``\\eqref{label}``."""
        return rf"\eqref{{{label}}}"

    @staticmethod
    def pageref(label: str) -> str:
        r"""Return ``\\pageref{label}``."""
        return rf"\pageref{{{label}}}"

    @staticmethod
    def autoref(label: str) -> str:
        r"""Return ``\\autoref{label}``."""
        return rf"\autoref{{{label}}}"

    @staticmethod
    def cref(label: str) -> str:
        r"""Return ``\\cref{label}``."""
        return rf"\cref{{{label}}}"

    @staticmethod
    def cite(key: str) -> str:
        r"""Return ``\\cite{key}``."""
        return rf"\cite{{{key}}}"

    @staticmethod
    def citep(key: str) -> str:
        r"""Return ``\\citep{key}``."""
        return rf"\citep{{{key}}}"


# ---------------------------------------------------------------------------
# Leaderboard table
# ---------------------------------------------------------------------------

class LeaderboardTable:
    """Generate a ranked leaderboard table."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)
        self._fmt = NumberFormatter()

    def render(
        self,
        entries: List[Dict[str, Any]],
        rank_by: str,
        columns: List[str],
        higher_is_better: bool = True,
        caption: str = "Leaderboard",
        label: str = "tab:leaderboard",
        top_n: Optional[int] = None,
    ) -> str:
        """Generate a leaderboard table.

        Parameters
        ----------
        entries : list of dict
            Each dict has ``"name"`` plus metric values.
        rank_by : str
            Metric name to rank by.
        columns : list of str
            Additional columns to display.
        higher_is_better : bool
            Sort direction for ranking.
        top_n : int, optional
            Limit to top N entries.
        """
        sorted_entries = sorted(
            entries,
            key=lambda e: e.get(rank_by, 0),
            reverse=higher_is_better,
        )
        if top_n is not None:
            sorted_entries = sorted_entries[:top_n]

        all_cols = ["Rank", "Name"] + [LaTeXEscaper.escape(c) for c in columns]
        col_fmt = "r" + "l" + "c" * len(columns)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(all_cols)

        prec = self.config.precision
        for rank, entry in enumerate(sorted_entries, 1):
            row: List[str] = [str(rank)]
            name = entry.get("name", f"Entry {rank}")
            if rank <= 3:
                name = LaTeXEscaper.bold(LaTeXEscaper.escape(str(name)))
            else:
                name = LaTeXEscaper.escape(str(name))
            row.append(name)
            for c in columns:
                val = entry.get(c)
                if isinstance(val, tuple) and len(val) == 2:
                    cell = self._fmt.format_mean_std(val[0], val[1], prec)
                elif isinstance(val, float):
                    cell = f"{val:.{prec}f}"
                elif val is None:
                    cell = "---"
                else:
                    cell = str(val)
                if rank == 1 and c == rank_by:
                    cell = LaTeXEscaper.bold(cell)
                row.append(cell)
            table.add_data_row(row)

        return table.render()


# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

class SummaryStatsTable:
    """Generate summary statistics tables."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)
        self._fmt = NumberFormatter()

    def render(
        self,
        stats: Dict[str, Dict[str, float]],
        stat_names: Optional[List[str]] = None,
        caption: str = "Summary statistics",
        label: str = "tab:summary-stats",
    ) -> str:
        """Generate a summary statistics table.

        Parameters
        ----------
        stats : dict
            ``{variable_name: {"mean": ..., "std": ..., "min": ...,
            "max": ..., "median": ...}, ...}``
        stat_names : list of str, optional
            Statistic names to include.  Defaults to common ones.
        """
        if stat_names is None:
            stat_names = ["mean", "std", "min", "max", "median"]

        variables = list(stats.keys())
        header = ["Variable"] + [LaTeXEscaper.escape(s) for s in stat_names]
        col_fmt = "l" + "c" * len(stat_names)

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        prec = self.config.precision
        for var in variables:
            row: List[str] = [LaTeXEscaper.escape(var)]
            for s in stat_names:
                val = stats[var].get(s)
                if val is not None:
                    row.append(f"{val:.{prec}f}")
                else:
                    row.append("---")
            table.add_data_row(row)

        return table.render()


# ---------------------------------------------------------------------------
# Confusion-matrix style table
# ---------------------------------------------------------------------------

class ConfusionMatrixTable:
    """Render a confusion matrix as a LaTeX table."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)

    def render(
        self,
        matrix: List[List[int]],
        class_names: List[str],
        caption: str = "Confusion matrix",
        label: str = "tab:confusion",
        colorize: bool = True,
        normalize: bool = False,
    ) -> str:
        """Render the confusion matrix.

        Parameters
        ----------
        matrix : list of list of int
            ``matrix[true][predicted]`` counts.
        class_names : list of str
            Class labels.
        colorize : bool
            Apply heat-map colouring.
        normalize : bool
            Show proportions instead of counts.
        """
        n = len(class_names)

        # Normalise if requested
        display: List[List[float]] = []
        if normalize:
            for row in matrix:
                total = sum(row)
                if total > 0:
                    display.append([v / total for v in row])
                else:
                    display.append([0.0] * n)
        else:
            display = [[float(v) for v in row] for row in matrix]

        flat = [v for row in display for v in row]
        min_val = min(flat) if flat else 0.0
        max_val = max(flat) if flat else 1.0

        col_fmt = "l" + "c" * n
        lines: List[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(rf"\begin{{tabular}}{{{col_fmt}}}")
        lines.append(r"\toprule")

        # Header: predicted labels
        pred_header = (
            r"\multicolumn{1}{c}{}"
            + " & "
            + rf"\multicolumn{{{n}}}{{c}}{{\textbf{{Predicted}}}}"
            + r" \\"
        )
        lines.append(pred_header)
        cmidrule = rf"\cmidrule(lr){{2-{n + 1}}}"
        lines.append(cmidrule)

        sub_header = [LaTeXEscaper.bold("True")]
        for name in class_names:
            sub_header.append(LaTeXEscaper.bold(LaTeXEscaper.escape(name)))
        lines.append(" & ".join(sub_header) + r" \\")
        lines.append(r"\midrule")

        # Data rows
        prec = 2 if normalize else 0
        for r_idx, name in enumerate(class_names):
            row_parts: List[str] = [LaTeXEscaper.escape(name)]
            for c_idx in range(n):
                val = display[r_idx][c_idx]
                if normalize:
                    cell_text = f"{val:.{prec}f}"
                else:
                    cell_text = str(int(val))
                if colorize:
                    color_cmd = ColorDefinitions.cell_color(
                        val, min_val, max_val, "heat"
                    )
                    cell_text = f"{color_cmd} {cell_text}"
                row_parts.append(cell_text)
            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{table}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-dataset results table
# ---------------------------------------------------------------------------

class MultiDatasetTable:
    """Tables comparing algorithms across multiple datasets."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)
        self._comp = ComparisonTableGenerator(config=self.config)
        self._fmt = NumberFormatter()

    def render(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        datasets: List[str],
        algorithms: List[str],
        metric_name: str = "score",
        higher_is_better: bool = True,
        show_average: bool = True,
        caption: str = "Multi-dataset comparison",
        label: str = "tab:multi-dataset",
    ) -> str:
        """Multi-dataset comparison table.

        Parameters
        ----------
        results : dict
            ``{dataset: {algorithm: {metric: value, ...}, ...}, ...}``
        datasets : list of str
            Dataset names in column order.
        algorithms : list of str
            Algorithm names in row order.
        show_average : bool
            Show an "Avg" column.
        """
        col_names = list(datasets)
        if show_average:
            col_names.append("Avg")
        n_cols = len(col_names)

        header = ["Algorithm"] + [LaTeXEscaper.escape(d) for d in col_names]
        col_fmt = "l" + "c" * n_cols

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        # Per-dataset ranking
        ds_best: Dict[str, Optional[int]] = {}
        ds_second: Dict[str, Optional[int]] = {}
        for ds in datasets:
            vals: List[Optional[float]] = []
            for algo in algorithms:
                raw = results.get(ds, {}).get(algo, {}).get(metric_name)
                vals.append(self._comp._mean_of(raw))
            b, s = self._comp._find_best_and_second(vals, higher_is_better)
            ds_best[ds] = b
            ds_second[ds] = s

        # Average ranking
        if show_average:
            avg_vals: List[Optional[float]] = []
            for algo in algorithms:
                algo_means: List[float] = []
                for ds in datasets:
                    raw = results.get(ds, {}).get(algo, {}).get(metric_name)
                    m = self._comp._mean_of(raw)
                    if m is not None:
                        algo_means.append(m)
                avg_vals.append(
                    sum(algo_means) / len(algo_means) if algo_means else None
                )
            avg_best, avg_second = self._comp._find_best_and_second(
                avg_vals, higher_is_better
            )

        prec = self.config.precision
        for a_idx, algo in enumerate(algorithms):
            row: List[str] = [LaTeXEscaper.escape(algo)]
            algo_means_list: List[float] = []
            for ds in datasets:
                raw = results.get(ds, {}).get(algo, {}).get(metric_name)
                cell = self._comp._format_metric_value(raw)
                if a_idx == ds_best.get(ds):
                    cell = self._comp._apply_highlight(cell, "best")
                elif a_idx == ds_second.get(ds):
                    cell = self._comp._apply_highlight(cell, "second")
                row.append(cell)
                m = self._comp._mean_of(raw)
                if m is not None:
                    algo_means_list.append(m)
            if show_average:
                if algo_means_list:
                    avg = sum(algo_means_list) / len(algo_means_list)
                    avg_cell = f"{avg:.{prec}f}"
                    if a_idx == avg_best:
                        avg_cell = self._comp._apply_highlight(avg_cell, "best")
                    elif a_idx == avg_second:
                        avg_cell = self._comp._apply_highlight(avg_cell, "second")
                    row.append(avg_cell)
                else:
                    row.append("---")
            table.add_data_row(row)

        return table.render()


# ---------------------------------------------------------------------------
# Win/Tie/Loss count table
# ---------------------------------------------------------------------------

class WinTieLossTable:
    """Count pairwise win/tie/loss across datasets or metrics."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)

    def render(
        self,
        counts: Dict[str, Dict[str, Tuple[int, int, int]]],
        caption: str = "Win / Tie / Loss",
        label: str = "tab:win-tie-loss",
    ) -> str:
        """Render a Win/Tie/Loss table.

        Parameters
        ----------
        counts : dict
            ``{algo_a: {algo_b: (wins, ties, losses), ...}, ...}``
        """
        algorithms = list(counts.keys())
        n = len(algorithms)
        header = [""] + [LaTeXEscaper.escape(a) for a in algorithms]
        col_fmt = "l" + "c" * n

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        for a in algorithms:
            row: List[str] = [LaTeXEscaper.escape(a)]
            for b in algorithms:
                if a == b:
                    row.append("---")
                else:
                    w, t, l = counts.get(a, {}).get(b, (0, 0, 0))
                    cell = f"{w}/{t}/{l}"
                    if w > l:
                        cell = LaTeXEscaper.bold(cell)
                    elif l > w:
                        cell = LaTeXEscaper.italic(cell)
                    row.append(cell)
            table.add_data_row(row)

        table.add_footnote("Format: W/T/L. Bold = net winner, italic = net loser.")
        return table.render()


# ---------------------------------------------------------------------------
# Critical difference diagram table companion
# ---------------------------------------------------------------------------

class RankTable:
    """Table of average ranks across datasets (companion to CD diagrams)."""

    def __init__(
        self,
        config: Optional[LaTeXTableConfig] = None,
    ) -> None:
        self.config = config or LaTeXTableConfig(booktabs=True)

    def render(
        self,
        ranks: Dict[str, float],
        caption: str = "Average ranks",
        label: str = "tab:ranks",
    ) -> str:
        """Render a rank table sorted by average rank.

        Parameters
        ----------
        ranks : dict
            ``{algorithm: average_rank}``
        """
        sorted_items = sorted(ranks.items(), key=lambda x: x[1])

        header = ["Rank", "Algorithm", "Avg Rank"]
        col_fmt = "rlc"

        cfg = LaTeXTableConfig(
            caption=caption,
            label=label,
            booktabs=True,
            column_format=col_fmt,
        )
        table = LaTeXTable(config=cfg)
        table.add_header_row(header)

        prec = self.config.precision
        for pos, (algo, avg_rank) in enumerate(sorted_items, 1):
            name = LaTeXEscaper.escape(algo)
            if pos == 1:
                name = LaTeXEscaper.bold(name)
            table.add_data_row([str(pos), name, f"{avg_rank:.{prec}f}"])

        return table.render()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LaTeXEscaper",
    "NumberFormatter",
    "ColorDefinitions",
    "FontSize",
    "LaTeXTableConfig",
    "LaTeXTable",
    "ComparisonTableGenerator",
    "LaTeXFigure",
    "LaTeXDocument",
    "ReportBuilder",
    "DiversityMetricsTable",
    "TableStyler",
    "render_simple_table",
    "render_key_value_table",
    "render_matrix_table",
    "BatchExporter",
    "CrossRef",
    "LeaderboardTable",
    "SummaryStatsTable",
    "ConfusionMatrixTable",
    "MultiDatasetTable",
    "WinTieLossTable",
    "RankTable",
]
