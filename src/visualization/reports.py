"""
HTML report generation for the Diversity Decoding Arena.

Produces self-contained HTML reports with inline CSS and JavaScript for
experiment results, algorithm comparisons, metric taxonomies, and more.
No external dependencies required.
"""

from __future__ import annotations

import html
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Top-level configuration for report generation."""

    title: str = "Diversity Decoding Arena Report"
    author: str = ""
    date: str = ""
    output_dir: str = "."
    format: str = "html"  # "html" or "markdown"
    include_plots: bool = True
    include_tables: bool = True
    include_raw_data: bool = False
    theme: str = "light"  # "light" or "dark"
    custom_css: str = ""
    logo_path: str = ""
    table_of_contents: bool = True

    def __post_init__(self) -> None:
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.format not in ("html", "markdown"):
            raise ValueError(f"Unsupported format: {self.format}")
        if self.theme not in ("light", "dark"):
            raise ValueError(f"Unsupported theme: {self.theme}")


@dataclass
class ReportSection:
    """A section (or subsection) inside a report."""

    title: str = ""
    content: str = ""
    level: int = 1  # heading level 1-4
    section_id: str = ""
    subsections: List["ReportSection"] = field(default_factory=list)
    plots: List[Dict[str, str]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.section_id:
            self.section_id = _slugify(self.title) or f"section-{uuid.uuid4().hex[:8]}"
        self.level = max(1, min(4, self.level))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Turn arbitrary text into a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _esc(text: str) -> str:
    """HTML-escape arbitrary text."""
    return html.escape(str(text))


def _format_number_util(value: Any, precision: int = 4) -> str:
    """Format a numeric value with the given decimal precision."""
    if value is None:
        return "N/A"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(fv) or math.isinf(fv):
        return str(fv)
    if fv == int(fv) and abs(fv) < 1e12:
        return str(int(fv))
    return f"{fv:.{precision}f}"


def _format_percentage_util(value: Any) -> str:
    """Format a value as a percentage string."""
    if value is None:
        return "N/A"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{fv * 100:.2f}%"


def _conditional_color_util(
    value: float,
    thresholds: Optional[List[Tuple[float, str]]] = None,
) -> str:
    """Return a hex colour string based on *value* and *thresholds*.

    *thresholds* is a sorted list of ``(cutoff, hex_colour)`` pairs.
    The first threshold whose cutoff is >= value wins.  Falls back to
    the last colour if the value exceeds all cutoffs.
    """
    if thresholds is None:
        thresholds = [
            (0.2, "#e74c3c"),
            (0.4, "#e67e22"),
            (0.6, "#f1c40f"),
            (0.8, "#2ecc71"),
            (1.0, "#27ae60"),
        ]
    for cutoff, colour in thresholds:
        if value <= cutoff:
            return colour
    return thresholds[-1][1] if thresholds else "#888888"


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """Incrementally build an HTML report from sections, plots, tables, etc."""

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        self._sections: List[ReportSection] = []
        self._section_map: Dict[str, ReportSection] = {}

    # -- public helpers to populate the report ---------------------------------

    def add_section(self, section: ReportSection) -> None:
        """Append a top-level section (or subsection if parent exists)."""
        self._sections.append(section)
        self._section_map[section.section_id] = section

    def add_plot(
        self,
        svg_content: str,
        caption: str = "",
        section_id: str = "",
    ) -> None:
        """Attach an SVG plot to a section (or create a new one)."""
        sec = self._resolve_section(section_id)
        sec.plots.append({"svg": svg_content, "caption": caption})

    def add_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        caption: str = "",
        section_id: str = "",
        sortable: bool = False,
    ) -> None:
        """Attach a data table to a section."""
        sec = self._resolve_section(section_id)
        sec.tables.append(
            {
                "headers": headers,
                "rows": rows,
                "caption": caption,
                "sortable": sortable,
            }
        )

    def add_text(self, text: str, section_id: str = "") -> None:
        """Append raw HTML/text content to a section."""
        sec = self._resolve_section(section_id)
        sec.content += text

    def add_metric_summary(
        self,
        metrics_dict: Dict[str, Any],
        section_id: str = "",
    ) -> None:
        """Render a dictionary of metric name -> value as summary cards."""
        cards_html = '<div class="metric-cards">\n'
        for name, value in metrics_dict.items():
            formatted = _format_number_util(value)
            cards_html += (
                f'<div class="metric-card">'
                f'<div class="metric-name">{_esc(str(name))}</div>'
                f'<div class="metric-value">{_esc(formatted)}</div>'
                f"</div>\n"
            )
        cards_html += "</div>\n"
        self.add_text(cards_html, section_id)

    def add_code_block(
        self,
        code: str,
        language: str = "python",
        section_id: str = "",
    ) -> None:
        """Append a syntax-highlighted code block."""
        block = (
            f'<pre class="code-block" data-lang="{_esc(language)}">'
            f"<code>{_esc(code)}</code></pre>\n"
        )
        self.add_text(block, section_id)

    def add_key_value_table(
        self,
        kv_dict: Dict[str, Any],
        title: str = "",
        section_id: str = "",
    ) -> None:
        """Render a simple two-column key/value table."""
        rows = [[str(k), str(v)] for k, v in kv_dict.items()]
        self.add_table(["Key", "Value"], rows, caption=title, section_id=section_id)

    # -- build -----------------------------------------------------------------

    def build(self) -> str:
        """Compile all sections into a complete HTML document."""
        gen = HTMLReportGenerator(self.config)
        toc_html = self._render_toc() if self.config.table_of_contents else ""
        body_parts: List[str] = []
        for sec in self._sections:
            body_parts.append(self._render_section(sec))
        body_html = "\n".join(body_parts)
        css = gen._css_theme(self.config.theme)
        if self.config.custom_css:
            css += f"\n/* --- custom css --- */\n{self.config.custom_css}\n"
        nav_html = gen._render_navigation(self._sections)
        header = gen._html_header(self.config.title, css)
        footer = gen._html_footer()
        js = gen._inline_js()
        content = (
            f"{header}\n"
            f'<div class="layout-wrapper">\n'
            f"{nav_html}\n"
            f'<main class="main-content">\n'
            f'<div class="report-meta">\n'
            f"<h1>{_esc(self.config.title)}</h1>\n"
        )
        if self.config.author:
            content += f'<p class="author">Author: {_esc(self.config.author)}</p>\n'
        content += f'<p class="date">Generated: {_esc(self.config.date)}</p>\n'
        content += "</div>\n"
        if toc_html:
            content += toc_html
        content += body_html
        content += f"\n{js}\n</main>\n</div>\n{footer}"
        return content

    # -- internal helpers ------------------------------------------------------

    def _render_toc(self) -> str:
        """Produce an HTML table-of-contents from current sections."""
        if not self._sections:
            return ""
        lines = ['<nav class="toc"><h2>Table of Contents</h2><ul>\n']
        for sec in self._sections:
            lines.append(self._toc_entry(sec))
        lines.append("</ul></nav>\n")
        return "".join(lines)

    def _toc_entry(self, sec: ReportSection) -> str:
        entry = (
            f'<li class="toc-l{sec.level}">'
            f'<a href="#{_esc(sec.section_id)}">{_esc(sec.title)}</a>'
        )
        if sec.subsections:
            entry += "<ul>\n"
            for sub in sec.subsections:
                entry += self._toc_entry(sub)
            entry += "</ul>\n"
        entry += "</li>\n"
        return entry

    def _render_section(self, section: ReportSection) -> str:
        """Recursively render a ReportSection into HTML."""
        tag = f"h{section.level}"
        parts: List[str] = [
            f'<section id="{_esc(section.section_id)}" class="report-section level-{section.level}">',
            f"<{tag}>{_esc(section.title)}</{tag}>",
        ]
        if section.content:
            parts.append(f'<div class="section-content">{section.content}</div>')
        if self.config.include_plots:
            for plot in section.plots:
                parts.append('<div class="plot-container">')
                parts.append(plot["svg"])
                if plot.get("caption"):
                    parts.append(
                        f'<p class="plot-caption">{_esc(plot["caption"])}</p>'
                    )
                parts.append("</div>")
        if self.config.include_tables:
            for tbl in section.tables:
                parts.append(self._render_table_html(tbl))
        for sub in section.subsections:
            parts.append(self._render_section(sub))
        parts.append("</section>\n")
        return "\n".join(parts)

    def _render_table_html(self, tbl: Dict[str, Any]) -> str:
        tid = f"tbl-{uuid.uuid4().hex[:8]}"
        sortable_cls = " sortable" if tbl.get("sortable") else ""
        parts = [f'<div class="table-wrapper">']
        if tbl.get("caption"):
            parts.append(f'<p class="table-caption">{_esc(tbl["caption"])}</p>')
        parts.append(f'<table id="{tid}" class="data-table{sortable_cls}">')
        parts.append("<thead><tr>")
        for h in tbl["headers"]:
            parts.append(f"<th>{_esc(str(h))}</th>")
        parts.append("</tr></thead>")
        parts.append("<tbody>")
        for ri, row in enumerate(tbl["rows"]):
            cls = "even" if ri % 2 == 0 else "odd"
            parts.append(f'<tr class="{cls}">')
            for cell in row:
                parts.append(f"<td>{_esc(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>\n")
        return "\n".join(parts)

    def get_sections(self) -> List[ReportSection]:
        """Return the list of top-level sections."""
        return list(self._sections)

    def _resolve_section(self, section_id: str) -> ReportSection:
        """Find or create the section with *section_id*."""
        if section_id and section_id in self._section_map:
            return self._section_map[section_id]
        sec = ReportSection(
            title=section_id or "Untitled",
            section_id=section_id or f"auto-{uuid.uuid4().hex[:8]}",
        )
        self.add_section(sec)
        return sec


# ---------------------------------------------------------------------------
# HTMLReportGenerator
# ---------------------------------------------------------------------------

class HTMLReportGenerator:
    """Full-featured generator for self-contained HTML reports."""

    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # High-level report generators
    # ------------------------------------------------------------------

    def generate_experiment_report(
        self,
        results: Dict[str, Any],
        plots: Optional[Dict[str, str]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a complete experiment-results HTML page.

        *results* maps algorithm names to dicts of metric values.
        *plots*   maps plot names to SVG strings.
        *stats*   contains optional statistical-test outputs.
        """
        plots = plots or {}
        stats = stats or {}

        builder = ReportBuilder(self.config)

        # -- overview section --
        overview = ReportSection(title="Experiment Overview", level=1,
                                 section_id="experiment-overview")
        overview.content = (
            '<p class="lead">This report summarises the results of a '
            "Diversity Decoding Arena experiment.  Each algorithm was "
            "evaluated across multiple diversity metrics and the outcomes "
            "are presented below.</p>\n"
        )
        algo_names = sorted(results.keys())
        overview.content += (
            f'<p>Algorithms evaluated: <strong>{len(algo_names)}</strong></p>\n'
        )
        builder.add_section(overview)

        # -- summary cards --
        if results:
            summary_sec = ReportSection(title="Summary Statistics", level=2,
                                        section_id="summary-stats")
            summary_data: Dict[str, Any] = {}
            all_metrics: set = set()
            for algo, mvals in results.items():
                if isinstance(mvals, dict):
                    all_metrics.update(mvals.keys())
            for m in sorted(all_metrics):
                vals = [
                    results[a].get(m) for a in algo_names
                    if isinstance(results[a], dict) and results[a].get(m) is not None
                ]
                numeric = []
                for v in vals:
                    try:
                        numeric.append(float(v))
                    except (TypeError, ValueError):
                        pass
                if numeric:
                    summary_data[f"{m} (mean)"] = sum(numeric) / len(numeric)
                    summary_data[f"{m} (max)"] = max(numeric)
                    summary_data[f"{m} (min)"] = min(numeric)
            builder.add_section(summary_sec)
            builder.add_metric_summary(summary_data, "summary-stats")

        # -- per-algorithm details --
        details_sec = ReportSection(title="Algorithm Results", level=2,
                                    section_id="algorithm-results")
        builder.add_section(details_sec)
        if results and all_metrics:
            headers = ["Algorithm"] + sorted(all_metrics)
            rows = []
            for algo in algo_names:
                row: List[Any] = [algo]
                mvals = results[algo] if isinstance(results[algo], dict) else {}
                for m in sorted(all_metrics):
                    row.append(_format_number_util(mvals.get(m)))
                rows.append(row)
            builder.add_table(headers, rows, caption="Results by Algorithm",
                              section_id="algorithm-results", sortable=True)

        # -- plots --
        if plots:
            plots_sec = ReportSection(title="Visualisations", level=2,
                                      section_id="visualisations")
            builder.add_section(plots_sec)
            for pname, svg in plots.items():
                builder.add_plot(svg, caption=pname, section_id="visualisations")

        # -- statistical tests --
        if stats:
            stat_sec = ReportSection(title="Statistical Analysis", level=2,
                                     section_id="statistical-analysis")
            builder.add_section(stat_sec)
            builder.add_text(
                self._render_statistical_section(stats),
                section_id="statistical-analysis",
            )

        # -- rankings --
        if results:
            rank_sec = ReportSection(title="Rankings", level=2,
                                     section_id="rankings")
            builder.add_section(rank_sec)
            rankings = self._compute_rankings(results)
            builder.add_text(
                self._render_ranking_section(rankings),
                section_id="rankings",
            )

        return builder.build()

    def generate_algorithm_profile(
        self,
        algorithm_name: str,
        results: Dict[str, Any],
    ) -> str:
        """Generate a single-page profile for one algorithm."""
        builder = ReportBuilder(
            ReportConfig(
                title=f"Algorithm Profile: {algorithm_name}",
                theme=self.config.theme,
                author=self.config.author,
                date=self.config.date,
                output_dir=self.config.output_dir,
                custom_css=self.config.custom_css,
            )
        )

        # overview
        sec = ReportSection(title="Overview", level=1, section_id="overview")
        sec.content = (
            f"<p>Detailed profile for algorithm "
            f"<strong>{_esc(algorithm_name)}</strong>.</p>\n"
        )
        builder.add_section(sec)

        # configuration
        config_vals = results.get("config", {})
        if config_vals and isinstance(config_vals, dict):
            cfg_sec = ReportSection(title="Configuration", level=2,
                                    section_id="configuration")
            builder.add_section(cfg_sec)
            builder.add_key_value_table(config_vals, title="Algorithm Parameters",
                                        section_id="configuration")

        # metrics
        metrics = results.get("metrics", results)
        if isinstance(metrics, dict):
            met_sec = ReportSection(title="Metrics", level=2, section_id="metrics")
            builder.add_section(met_sec)
            displayable = {
                k: v for k, v in metrics.items()
                if k != "config" and not isinstance(v, (dict, list))
            }
            builder.add_metric_summary(displayable, "metrics")
            builder.add_key_value_table(displayable, title="All Metrics",
                                        section_id="metrics")

        # strengths / weaknesses
        analysis_sec = ReportSection(title="Strengths & Weaknesses", level=2,
                                     section_id="strengths-weaknesses")
        analysis_sec.content = self._strengths_weaknesses_html(metrics)
        builder.add_section(analysis_sec)

        # per-run breakdown
        runs = results.get("runs", [])
        if runs and isinstance(runs, list):
            run_sec = ReportSection(title="Per-Run Breakdown", level=2,
                                    section_id="per-run")
            builder.add_section(run_sec)
            if runs and isinstance(runs[0], dict):
                run_headers = sorted(runs[0].keys())
                run_rows = [[r.get(h, "") for h in run_headers] for r in runs]
                builder.add_table(run_headers, run_rows, caption="Individual Runs",
                                  section_id="per-run", sortable=True)

        return builder.build()

    def generate_comparison_report(
        self,
        comparisons: Dict[str, Any],
    ) -> str:
        """Produce a pairwise comparison HTML report.

        *comparisons* is a dict mapping ``"algoA_vs_algoB"`` to a dict
        of comparison statistics.
        """
        builder = ReportBuilder(
            ReportConfig(
                title="Algorithm Comparison Report",
                theme=self.config.theme,
                author=self.config.author,
                date=self.config.date,
                output_dir=self.config.output_dir,
                custom_css=self.config.custom_css,
            )
        )

        intro = ReportSection(title="Pairwise Comparisons", level=1,
                              section_id="comparisons-intro")
        intro.content = (
            "<p>This report presents head-to-head comparisons between "
            "pairs of algorithms.</p>\n"
        )
        builder.add_section(intro)

        for pair_key, cdata in comparisons.items():
            pair_sec = ReportSection(
                title=pair_key.replace("_", " ").title(),
                level=2,
                section_id=_slugify(pair_key),
            )
            builder.add_section(pair_sec)

            if isinstance(cdata, dict):
                # summary badges
                winner = cdata.get("winner", "tie")
                badge = self._render_badge(
                    f"Winner: {winner}",
                    "#27ae60" if winner != "tie" else "#95a5a6",
                )
                builder.add_text(badge, _slugify(pair_key))

                flat = {
                    k: v for k, v in cdata.items()
                    if not isinstance(v, (dict, list))
                }
                if flat:
                    builder.add_key_value_table(
                        flat,
                        title="Comparison Statistics",
                        section_id=_slugify(pair_key),
                    )

                metric_diffs = cdata.get("metric_diffs", {})
                if metric_diffs and isinstance(metric_diffs, dict):
                    headers = ["Metric", "Difference", "Direction"]
                    rows = []
                    for mname, diff_val in metric_diffs.items():
                        try:
                            fv = float(diff_val)
                            direction = "↑ better" if fv > 0 else ("↓ worse" if fv < 0 else "=")
                        except (TypeError, ValueError):
                            fv = diff_val
                            direction = ""
                        rows.append([mname, _format_number_util(fv), direction])
                    builder.add_table(headers, rows,
                                      caption="Metric Differences",
                                      section_id=_slugify(pair_key),
                                      sortable=True)

                effect_sizes = cdata.get("effect_sizes", {})
                if effect_sizes and isinstance(effect_sizes, dict):
                    es_headers = ["Metric", "Effect Size", "Interpretation"]
                    es_rows = []
                    for mname, es in effect_sizes.items():
                        try:
                            fes = abs(float(es))
                        except (TypeError, ValueError):
                            fes = 0.0
                        interp = (
                            "large" if fes >= 0.8 else
                            "medium" if fes >= 0.5 else
                            "small" if fes >= 0.2 else "negligible"
                        )
                        es_rows.append([mname, _format_number_util(es), interp])
                    builder.add_table(es_headers, es_rows,
                                      caption="Effect Sizes",
                                      section_id=_slugify(pair_key),
                                      sortable=True)

        return builder.build()

    def generate_metric_taxonomy_report(
        self,
        correlations: Optional[Dict[str, Any]] = None,
        clusters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a report about metric inter-relationships."""
        correlations = correlations or {}
        clusters = clusters or {}

        builder = ReportBuilder(
            ReportConfig(
                title="Metric Taxonomy Report",
                theme=self.config.theme,
                author=self.config.author,
                date=self.config.date,
                output_dir=self.config.output_dir,
                custom_css=self.config.custom_css,
            )
        )

        intro = ReportSection(title="Metric Taxonomy", level=1,
                              section_id="metric-taxonomy")
        intro.content = (
            "<p>This report analyses the relationships between diversity "
            "metrics, grouping them into clusters and showing pairwise "
            "correlations.</p>\n"
        )
        builder.add_section(intro)

        # correlation matrix
        if correlations:
            corr_sec = ReportSection(title="Correlation Matrix", level=2,
                                     section_id="correlation-matrix")
            builder.add_section(corr_sec)

            if isinstance(correlations, dict):
                metric_names = sorted(correlations.keys())
                headers = [""] + metric_names
                rows = []
                for m1 in metric_names:
                    row: List[Any] = [m1]
                    m1_corrs = correlations[m1]
                    if isinstance(m1_corrs, dict):
                        for m2 in metric_names:
                            val = m1_corrs.get(m2, "")
                            row.append(_format_number_util(val, 3))
                    else:
                        row.extend([""] * len(metric_names))
                    rows.append(row)
                builder.add_table(headers, rows,
                                  caption="Pairwise Correlations",
                                  section_id="correlation-matrix",
                                  sortable=False)

            # highlight strong correlations
            strong: List[Tuple[str, str, float]] = []
            for m1 in sorted(correlations.keys()):
                m1_corrs = correlations[m1]
                if not isinstance(m1_corrs, dict):
                    continue
                for m2, val in m1_corrs.items():
                    if m1 >= m2:
                        continue
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        continue
                    if abs(fv) >= 0.7:
                        strong.append((m1, m2, fv))
            if strong:
                strong.sort(key=lambda t: -abs(t[2]))
                s_headers = ["Metric A", "Metric B", "Correlation"]
                s_rows = [[a, b, _format_number_util(c, 3)] for a, b, c in strong]
                builder.add_table(s_headers, s_rows,
                                  caption="Strong Correlations (|r| ≥ 0.7)",
                                  section_id="correlation-matrix",
                                  sortable=True)

        # clusters
        if clusters:
            cl_sec = ReportSection(title="Metric Clusters", level=2,
                                   section_id="metric-clusters")
            builder.add_section(cl_sec)

            if isinstance(clusters, dict):
                for cluster_name, members in clusters.items():
                    badge = self._render_badge(cluster_name, "#3498db")
                    member_list = ", ".join(str(m) for m in members) if isinstance(members, list) else str(members)
                    builder.add_text(
                        f'{badge}<p class="cluster-members">'
                        f"Members: {_esc(member_list)}</p>\n",
                        section_id="metric-clusters",
                    )

            # independence recommendations
            rec_sec = ReportSection(title="Independence Recommendations", level=3,
                                    section_id="independence-recs")
            rec_sec.content = (
                "<p>Based on the cluster analysis, the following metric "
                "subsets are recommended for independent evaluation:</p>\n"
            )
            if isinstance(clusters, dict):
                recs: List[str] = []
                for cname, members in clusters.items():
                    if isinstance(members, list) and members:
                        recs.append(str(members[0]))
                if recs:
                    rec_sec.content += "<ul>\n"
                    for r in recs:
                        rec_sec.content += f"<li>{_esc(r)}</li>\n"
                    rec_sec.content += "</ul>\n"
            builder.add_section(rec_sec)

        return builder.build()

    # ------------------------------------------------------------------
    # HTML structure helpers
    # ------------------------------------------------------------------

    def _html_header(self, title: str, css: str) -> str:
        """Return the ``<!DOCTYPE …>`` through ``<body>`` opening."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{_esc(title)}</title>\n"
            f"<style>\n{css}\n</style>\n"
            "</head>\n"
            f'<body class="theme-{_esc(self.config.theme)}">\n'
        )

    def _html_footer(self) -> str:
        """Closing ``</body></html>``."""
        return (
            '<footer class="report-footer">\n'
            "<p>Generated by Diversity Decoding Arena &mdash; "
            f"{_esc(self.config.date)}</p>\n"
            "</footer>\n"
            "</body>\n"
            "</html>\n"
        )

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    def _css_theme(self, theme_name: str) -> str:
        """Return a complete CSS stylesheet (~200+ lines)."""
        if theme_name == "dark":
            bg = "#1a1a2e"
            bg_card = "#16213e"
            bg_alt_row = "#1a1a3e"
            text = "#e0e0e0"
            text_muted = "#a0a0b0"
            heading = "#e8e8f0"
            border = "#2a2a4a"
            accent = "#4fc3f7"
            accent_dark = "#0288d1"
            link = "#4fc3f7"
            nav_bg = "#0f3460"
            card_shadow = "rgba(0,0,0,0.5)"
            code_bg = "#0d1117"
            badge_text = "#ffffff"
            alert_info_bg = "#1a3a5c"
            alert_warn_bg = "#5c4a1a"
            alert_err_bg = "#5c1a1a"
            alert_ok_bg = "#1a5c2e"
            table_header_bg = "#1e3a5f"
        else:
            bg = "#f5f7fa"
            bg_card = "#ffffff"
            bg_alt_row = "#f9fafb"
            text = "#2d3748"
            text_muted = "#718096"
            heading = "#1a202c"
            border = "#e2e8f0"
            accent = "#3182ce"
            accent_dark = "#2c5282"
            link = "#3182ce"
            nav_bg = "#2d3748"
            card_shadow = "rgba(0,0,0,0.08)"
            code_bg = "#f7fafc"
            badge_text = "#ffffff"
            alert_info_bg = "#ebf8ff"
            alert_warn_bg = "#fffaf0"
            alert_err_bg = "#fff5f5"
            alert_ok_bg = "#f0fff4"
            table_header_bg = "#edf2f7"

        return f"""
/* ===== Base Reset & Typography ===== */
*, *::before, *::after {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}
html {{
    font-size: 15px;
    scroll-behavior: smooth;
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Oxygen, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
    background: {bg};
    color: {text};
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}}

/* ===== Layout ===== */
.layout-wrapper {{
    display: flex;
    min-height: 100vh;
}}
.sidebar-nav {{
    width: 260px;
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    background: {nav_bg};
    color: #ecf0f1;
    padding: 1.5rem 1rem;
    overflow-y: auto;
    z-index: 100;
    transition: transform 0.3s ease;
}}
.sidebar-nav h3 {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {accent};
    margin-bottom: 0.75rem;
}}
.sidebar-nav ul {{
    list-style: none;
}}
.sidebar-nav li {{
    margin-bottom: 0.35rem;
}}
.sidebar-nav a {{
    color: #cbd5e0;
    text-decoration: none;
    font-size: 0.88rem;
    display: block;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    transition: background 0.15s, color 0.15s;
}}
.sidebar-nav a:hover {{
    background: rgba(255,255,255,0.08);
    color: #fff;
}}
.main-content {{
    margin-left: 260px;
    flex: 1;
    padding: 2rem 2.5rem 4rem;
    max-width: 1100px;
}}
@media (max-width: 900px) {{
    .sidebar-nav {{
        transform: translateX(-100%);
    }}
    .main-content {{
        margin-left: 0;
        padding: 1rem;
    }}
}}

/* ===== Report Meta ===== */
.report-meta {{
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid {border};
}}
.report-meta h1 {{
    font-size: 2rem;
    color: {heading};
    margin-bottom: 0.4rem;
}}
.report-meta .author,
.report-meta .date {{
    color: {text_muted};
    font-size: 0.9rem;
    margin: 0.15rem 0;
}}

/* ===== Headings ===== */
h1 {{ font-size: 1.8rem; color: {heading}; margin: 1.8rem 0 0.8rem; }}
h2 {{ font-size: 1.45rem; color: {heading}; margin: 1.5rem 0 0.65rem; border-bottom: 1px solid {border}; padding-bottom: 0.3rem; }}
h3 {{ font-size: 1.2rem; color: {heading}; margin: 1.2rem 0 0.5rem; }}
h4 {{ font-size: 1.05rem; color: {heading}; margin: 1rem 0 0.4rem; }}

/* ===== Links ===== */
a {{ color: {link}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* ===== TOC ===== */
.toc {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 4px {card_shadow};
}}
.toc h2 {{
    font-size: 1.1rem;
    border: none;
    margin-top: 0;
}}
.toc ul {{
    list-style: none;
    padding-left: 0;
}}
.toc li {{
    margin: 0.25rem 0;
}}
.toc .toc-l2 {{ padding-left: 1rem; }}
.toc .toc-l3 {{ padding-left: 2rem; }}
.toc .toc-l4 {{ padding-left: 3rem; }}

/* ===== Sections ===== */
.report-section {{
    margin-bottom: 1.5rem;
}}
.section-content p {{
    margin: 0.5rem 0;
}}
.lead {{
    font-size: 1.08rem;
    color: {text_muted};
}}

/* ===== Metric Cards ===== */
.metric-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}}
.metric-card {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px {card_shadow};
    text-align: center;
    transition: transform 0.15s, box-shadow 0.15s;
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px {card_shadow};
}}
.metric-name {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {text_muted};
    margin-bottom: 0.4rem;
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {accent_dark};
}}

/* ===== Summary Cards ===== */
.summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
}}
.summary-card {{
    background: {bg_card};
    border-left: 4px solid {accent};
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 4px {card_shadow};
}}
.summary-card .card-label {{
    font-size: 0.82rem;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.summary-card .card-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {heading};
    margin: 0.3rem 0;
}}
.summary-card .card-detail {{
    font-size: 0.85rem;
    color: {text_muted};
}}

/* ===== Algorithm Cards ===== */
.algorithm-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
}}
.algo-card {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 6px {card_shadow};
    transition: transform 0.15s;
}}
.algo-card:hover {{
    transform: translateY(-3px);
}}
.algo-card .algo-name {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {heading};
    margin-bottom: 0.4rem;
}}
.algo-card .algo-desc {{
    font-size: 0.9rem;
    color: {text_muted};
    margin-bottom: 0.8rem;
}}
.algo-card .algo-metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}}

/* ===== Tables ===== */
.table-wrapper {{
    overflow-x: auto;
    margin: 1rem 0;
}}
.table-caption {{
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    color: {heading};
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}
.data-table th {{
    background: {table_header_bg};
    color: {heading};
    font-weight: 600;
    text-align: left;
    padding: 0.6rem 0.8rem;
    border-bottom: 2px solid {border};
    white-space: nowrap;
}}
.data-table td {{
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid {border};
}}
.data-table tr.odd {{
    background: {bg_card};
}}
.data-table tr.even {{
    background: {bg_alt_row};
}}
.data-table tr:hover {{
    background: {accent}22;
}}
.data-table.sortable th {{
    cursor: pointer;
    user-select: none;
    position: relative;
    padding-right: 1.4rem;
}}
.data-table.sortable th::after {{
    content: '⇅';
    position: absolute;
    right: 0.4rem;
    opacity: 0.35;
    font-size: 0.75rem;
}}

/* ===== Code Blocks ===== */
.code-block {{
    background: {code_bg};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 1rem 1.2rem;
    overflow-x: auto;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.85rem;
    line-height: 1.55;
    margin: 0.8rem 0;
}}

/* ===== Badges ===== */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: {badge_text};
    margin: 0.15rem 0.2rem;
    white-space: nowrap;
}}

/* ===== Alerts ===== */
.alert {{
    padding: 0.9rem 1.2rem;
    border-radius: 6px;
    margin: 0.8rem 0;
    font-size: 0.92rem;
    border-left: 4px solid;
}}
.alert-info  {{ background: {alert_info_bg}; border-color: #3182ce; }}
.alert-warning {{ background: {alert_warn_bg}; border-color: #dd6b20; }}
.alert-error {{ background: {alert_err_bg}; border-color: #e53e3e; }}
.alert-success {{ background: {alert_ok_bg}; border-color: #38a169; }}

/* ===== Progress Bar ===== */
.progress-wrapper {{
    margin: 0.5rem 0;
}}
.progress-label {{
    font-size: 0.82rem;
    color: {text_muted};
    margin-bottom: 0.2rem;
}}
.progress-bar-bg {{
    height: 10px;
    background: {border};
    border-radius: 5px;
    overflow: hidden;
}}
.progress-bar-fill {{
    height: 100%;
    border-radius: 5px;
    transition: width 0.4s ease;
}}

/* ===== Tooltip ===== */
.tooltip-wrapper {{
    position: relative;
    display: inline-block;
    cursor: help;
    border-bottom: 1px dotted {text_muted};
}}
.tooltip-wrapper .tooltip-text {{
    visibility: hidden;
    background: {nav_bg};
    color: #fff;
    font-size: 0.8rem;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
    z-index: 10;
    opacity: 0;
    transition: opacity 0.2s;
}}
.tooltip-wrapper:hover .tooltip-text {{
    visibility: visible;
    opacity: 1;
}}

/* ===== Collapsible ===== */
.collapsible-toggle {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 0.7rem 1rem;
    width: 100%;
    text-align: left;
    font-size: 0.95rem;
    font-weight: 600;
    color: {heading};
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.5rem;
}}
.collapsible-toggle::after {{
    content: '▸';
    transition: transform 0.2s;
}}
.collapsible-toggle.active::after {{
    transform: rotate(90deg);
}}
.collapsible-body {{
    display: none;
    padding: 0.8rem 1rem;
    border: 1px solid {border};
    border-top: none;
    border-radius: 0 0 6px 6px;
    background: {bg_card};
}}

/* ===== Tabs ===== */
.tabs-wrapper {{
    margin: 1rem 0;
}}
.tab-buttons {{
    display: flex;
    gap: 0;
    border-bottom: 2px solid {border};
}}
.tab-btn {{
    padding: 0.55rem 1.2rem;
    background: transparent;
    border: none;
    font-size: 0.9rem;
    color: {text_muted};
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
}}
.tab-btn:hover {{
    color: {heading};
}}
.tab-btn.active {{
    color: {accent};
    border-bottom-color: {accent};
    font-weight: 600;
}}
.tab-panel {{
    display: none;
    padding: 1rem 0;
}}
.tab-panel.active {{
    display: block;
}}

/* ===== Plots ===== */
.plot-container {{
    margin: 1.2rem 0;
    text-align: center;
}}
.plot-container svg {{
    max-width: 100%;
    height: auto;
}}
.plot-caption {{
    font-size: 0.85rem;
    color: {text_muted};
    margin-top: 0.4rem;
    font-style: italic;
}}

/* ===== Rankings ===== */
.ranking-list {{
    list-style: none;
    counter-reset: ranking;
    margin: 0.5rem 0;
}}
.ranking-list li {{
    counter-increment: ranking;
    padding: 0.5rem 0.8rem 0.5rem 2.8rem;
    position: relative;
    border-bottom: 1px solid {border};
}}
.ranking-list li::before {{
    content: '#' counter(ranking);
    position: absolute;
    left: 0.5rem;
    font-weight: 700;
    color: {accent};
    font-size: 0.95rem;
}}

/* ===== Cluster Members ===== */
.cluster-members {{
    font-size: 0.9rem;
    color: {text_muted};
    margin: 0.3rem 0 1rem;
}}

/* ===== Footer ===== */
.report-footer {{
    text-align: center;
    padding: 1.5rem;
    font-size: 0.82rem;
    color: {text_muted};
    border-top: 1px solid {border};
    margin-top: 3rem;
}}

/* ===== Pareto ===== */
.pareto-highlight {{
    background: {accent}11;
    border: 1px dashed {accent};
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
}}
.pareto-highlight strong {{
    color: {accent_dark};
}}

/* ===== Utility ===== */
.text-muted {{ color: {text_muted}; }}
.text-center {{ text-align: center; }}
.mt-1 {{ margin-top: 0.5rem; }}
.mt-2 {{ margin-top: 1rem; }}
.mb-1 {{ margin-bottom: 0.5rem; }}
.mb-2 {{ margin-bottom: 1rem; }}
.sr-only {{
    position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0,0,0,0); border: 0;
}}
"""

    # ------------------------------------------------------------------
    # Component renderers
    # ------------------------------------------------------------------

    def _render_summary_cards(self, summary_stats: Dict[str, Any]) -> str:
        """Render a row of summary statistic cards."""
        parts = ['<div class="summary-cards">\n']
        for label, info in summary_stats.items():
            if isinstance(info, dict):
                value = info.get("value", "")
                detail = info.get("detail", "")
            else:
                value = info
                detail = ""
            parts.append(
                f'<div class="summary-card">\n'
                f'  <div class="card-label">{_esc(str(label))}</div>\n'
                f'  <div class="card-value">{_esc(str(value))}</div>\n'
            )
            if detail:
                parts.append(
                    f'  <div class="card-detail">{_esc(str(detail))}</div>\n'
                )
            parts.append("</div>\n")
        parts.append("</div>\n")
        return "".join(parts)

    def _render_metric_table(
        self,
        metrics: Dict[str, Any],
        highlights: Optional[Dict[str, str]] = None,
    ) -> str:
        """Render a metric table with optional conditional formatting.

        *highlights* maps metric names to CSS colour strings.
        """
        highlights = highlights or {}
        tid = f"mt-{uuid.uuid4().hex[:8]}"
        parts = [
            f'<table id="{tid}" class="data-table sortable">',
            "<thead><tr><th>Metric</th><th>Value</th></tr></thead>",
            "<tbody>",
        ]
        for i, (name, val) in enumerate(metrics.items()):
            cls = "even" if i % 2 == 0 else "odd"
            style = ""
            if name in highlights:
                style = f' style="border-left:4px solid {highlights[name]}"'
            formatted = _format_number_util(val)
            parts.append(
                f'<tr class="{cls}"{style}>'
                f"<td>{_esc(str(name))}</td>"
                f"<td>{_esc(formatted)}</td></tr>"
            )
        parts.append("</tbody></table>\n")
        return "\n".join(parts)

    def _render_ranking_section(self, rankings: Dict[str, List[str]]) -> str:
        """Render per-metric ranking lists.

        *rankings* maps metric names to ordered lists of algorithm names.
        """
        parts: List[str] = []
        for metric, ordered in rankings.items():
            parts.append(f"<h4>{_esc(metric)}</h4>")
            parts.append('<ol class="ranking-list">')
            for algo in ordered:
                parts.append(f"<li>{_esc(algo)}</li>")
            parts.append("</ol>\n")
        return "\n".join(parts)

    def _render_pareto_section(
        self,
        pareto_data: Dict[str, Any],
        svg_plot: str = "",
    ) -> str:
        """Render a Pareto-front summary section."""
        parts = ['<div class="pareto-highlight">']
        front = pareto_data.get("front", [])
        if front:
            parts.append(
                "<p><strong>Pareto-optimal algorithms:</strong> "
                + ", ".join(_esc(str(a)) for a in front)
                + "</p>\n"
            )
        dominated = pareto_data.get("dominated", [])
        if dominated:
            parts.append(
                "<p>Dominated: "
                + ", ".join(_esc(str(a)) for a in dominated)
                + "</p>\n"
            )
        parts.append("</div>\n")
        if svg_plot:
            parts.append(self._embed_svg(svg_plot))
        return "\n".join(parts)

    def _render_statistical_section(
        self,
        bayesian_results: Dict[str, Any],
    ) -> str:
        """Render statistical analysis results (Bayesian or frequentist)."""
        parts: List[str] = []

        p_values = bayesian_results.get("p_values", {})
        if p_values and isinstance(p_values, dict):
            parts.append("<h4>P-Values</h4>")
            headers = ["Test", "p-value", "Significant (α=0.05)"]
            rows = []
            for test_name, pval in p_values.items():
                try:
                    fp = float(pval)
                    sig = "Yes" if fp < 0.05 else "No"
                except (TypeError, ValueError):
                    fp = pval
                    sig = "?"
                rows.append([test_name, _format_number_util(fp, 6), sig])
            parts.append(
                self._render_interactive_table(
                    headers, rows, f"pval-{uuid.uuid4().hex[:6]}"
                )
            )

        bayes = bayesian_results.get("bayes_factors", {})
        if bayes and isinstance(bayes, dict):
            parts.append("<h4>Bayes Factors</h4>")
            headers = ["Comparison", "BF₁₀", "Evidence"]
            rows = []
            for comp, bf in bayes.items():
                try:
                    fbf = float(bf)
                except (TypeError, ValueError):
                    fbf = 0.0
                if fbf > 100:
                    evidence = "Extreme"
                elif fbf > 30:
                    evidence = "Very Strong"
                elif fbf > 10:
                    evidence = "Strong"
                elif fbf > 3:
                    evidence = "Moderate"
                elif fbf > 1:
                    evidence = "Anecdotal"
                else:
                    evidence = "Against"
                rows.append([comp, _format_number_util(bf, 2), evidence])
            parts.append(
                self._render_interactive_table(
                    headers, rows, f"bf-{uuid.uuid4().hex[:6]}"
                )
            )

        posteriors = bayesian_results.get("posteriors", {})
        if posteriors and isinstance(posteriors, dict):
            parts.append("<h4>Posterior Probabilities</h4>")
            for name, prob in posteriors.items():
                parts.append(
                    self._render_progress_bar(
                        float(prob) if prob is not None else 0,
                        1.0,
                        f"{name}: {_format_percentage_util(prob)}",
                    )
                )

        confidence = bayesian_results.get("confidence_intervals", {})
        if confidence and isinstance(confidence, dict):
            parts.append("<h4>Confidence Intervals</h4>")
            ci_headers = ["Parameter", "Lower", "Upper", "Width"]
            ci_rows = []
            for param, ci in confidence.items():
                if isinstance(ci, (list, tuple)) and len(ci) >= 2:
                    lo, hi = float(ci[0]), float(ci[1])
                    ci_rows.append([
                        param,
                        _format_number_util(lo),
                        _format_number_util(hi),
                        _format_number_util(hi - lo),
                    ])
            if ci_rows:
                parts.append(
                    self._render_interactive_table(
                        ci_headers, ci_rows, f"ci-{uuid.uuid4().hex[:6]}"
                    )
                )

        effect = bayesian_results.get("effect_sizes", {})
        if effect and isinstance(effect, dict):
            parts.append("<h4>Effect Sizes</h4>")
            parts.append(self._render_metric_table(effect))

        if not parts:
            parts.append(
                self._render_alert(
                    "No statistical analysis data available.", "info"
                )
            )

        return "\n".join(parts)

    def _render_algorithm_cards(
        self,
        algorithms_info: Dict[str, Dict[str, Any]],
    ) -> str:
        """Render a grid of algorithm information cards."""
        parts = ['<div class="algorithm-cards">\n']
        for algo_name, info in algorithms_info.items():
            desc = info.get("description", "")
            metrics = info.get("metrics", {})
            parts.append(f'<div class="algo-card">')
            parts.append(
                f'<div class="algo-name">{_esc(algo_name)}</div>'
            )
            if desc:
                parts.append(
                    f'<div class="algo-desc">{_esc(str(desc))}</div>'
                )
            if metrics and isinstance(metrics, dict):
                parts.append('<div class="algo-metrics">')
                for mname, mval in metrics.items():
                    colour = _conditional_color_util(
                        float(mval) if mval is not None else 0
                    )
                    parts.append(self._render_badge(
                        f"{mname}: {_format_number_util(mval)}", colour
                    ))
                parts.append("</div>")
            parts.append("</div>\n")
        parts.append("</div>\n")
        return "".join(parts)

    def _render_navigation(self, sections: List[ReportSection]) -> str:
        """Render a sidebar navigation panel."""
        parts = ['<nav class="sidebar-nav">\n<h3>Navigation</h3>\n<ul>\n']
        for sec in sections:
            parts.append(
                f'<li><a href="#{_esc(sec.section_id)}">'
                f"{_esc(sec.title)}</a></li>\n"
            )
            if sec.subsections:
                parts.append("<ul>")
                for sub in sec.subsections:
                    parts.append(
                        f'<li><a href="#{_esc(sub.section_id)}">'
                        f"{_esc(sub.title)}</a></li>\n"
                    )
                parts.append("</ul>")
        parts.append("</ul>\n</nav>\n")
        return "".join(parts)

    def _render_interactive_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        table_id: str,
    ) -> str:
        """Render a table with JavaScript sorting support."""
        parts = [
            f'<div class="table-wrapper">',
            f'<table id="{_esc(table_id)}" class="data-table sortable">',
            "<thead><tr>",
        ]
        for h in headers:
            parts.append(
                f'<th onclick="sortTable(\'{_esc(table_id)}\', '
                f'{headers.index(h)})">{_esc(str(h))}</th>'
            )
        parts.append("</tr></thead><tbody>")
        for ri, row in enumerate(rows):
            cls = "even" if ri % 2 == 0 else "odd"
            parts.append(f'<tr class="{cls}">')
            for cell in row:
                parts.append(f"<td>{_esc(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>\n")
        return "\n".join(parts)

    def _render_collapsible_section(self, title: str, content: str) -> str:
        """Render a collapsible (accordion-style) section."""
        cid = f"coll-{uuid.uuid4().hex[:8]}"
        return (
            f'<button class="collapsible-toggle" '
            f"onclick=\"toggleCollapsible('{cid}')\">"
            f"{_esc(title)}</button>\n"
            f'<div id="{cid}" class="collapsible-body">\n'
            f"{content}\n</div>\n"
        )

    def _render_tabs(self, tab_contents: Dict[str, str]) -> str:
        """Render a tabbed interface.

        *tab_contents* maps tab labels to HTML content strings.
        """
        group_id = f"tabs-{uuid.uuid4().hex[:8]}"
        btn_parts = [f'<div class="tabs-wrapper" id="{group_id}">\n<div class="tab-buttons">\n']
        panel_parts: List[str] = []
        for idx, (label, body) in enumerate(tab_contents.items()):
            tab_id = f"{group_id}-{idx}"
            active = " active" if idx == 0 else ""
            btn_parts.append(
                f'<button class="tab-btn{active}" '
                f"onclick=\"switchTab('{group_id}', {idx})\">"
                f"{_esc(label)}</button>\n"
            )
            panel_parts.append(
                f'<div class="tab-panel{active}" id="{tab_id}">\n'
                f"{body}\n</div>\n"
            )
        btn_parts.append("</div>\n")
        return "".join(btn_parts) + "".join(panel_parts) + "</div>\n"

    def _render_progress_bar(
        self,
        value: float,
        max_value: float,
        label: str = "",
    ) -> str:
        """Render a horizontal progress bar."""
        pct = min(100, max(0, (value / max_value) * 100)) if max_value else 0
        colour = _conditional_color_util(value / max_value if max_value else 0)
        parts = ['<div class="progress-wrapper">']
        if label:
            parts.append(f'<div class="progress-label">{_esc(label)}</div>')
        parts.append(
            f'<div class="progress-bar-bg">'
            f'<div class="progress-bar-fill" '
            f'style="width:{pct:.1f}%;background:{colour}"></div>'
            f"</div></div>\n"
        )
        return "".join(parts)

    def _render_badge(self, text: str, color: str = "#3498db") -> str:
        """Return an inline coloured badge."""
        return (
            f'<span class="badge" style="background:{_esc(color)}">'
            f"{_esc(text)}</span>"
        )

    def _render_tooltip(self, content: str, tooltip_text: str) -> str:
        """Wrap *content* in a tooltip that shows *tooltip_text* on hover."""
        return (
            f'<span class="tooltip-wrapper">{_esc(content)}'
            f'<span class="tooltip-text">{_esc(tooltip_text)}</span></span>'
        )

    def _render_alert(
        self,
        message: str,
        alert_type: str = "info",
    ) -> str:
        """Render a coloured alert box.

        *alert_type* must be one of ``info``, ``warning``, ``error``,
        ``success``.
        """
        if alert_type not in ("info", "warning", "error", "success"):
            alert_type = "info"
        icon_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
        }
        icon = icon_map.get(alert_type, "")
        return (
            f'<div class="alert alert-{alert_type}">'
            f"{icon} {_esc(message)}</div>\n"
        )

    def _embed_svg(self, svg_content: str) -> str:
        """Wrap raw SVG markup for inline embedding."""
        return f'<div class="plot-container">\n{svg_content}\n</div>\n'

    # ------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------

    def _inline_js(self) -> str:
        r"""Return inline ``<script>`` block for interactivity."""
        return """<script>
/* ===== Table sorting ===== */
function sortTable(tableId, colIdx) {
    var table = document.getElementById(tableId);
    if (!table) return;
    var tbody = table.querySelector('tbody');
    if (!tbody) return;
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var dir = table.getAttribute('data-sort-dir-' + colIdx);
    dir = (dir === 'asc') ? 'desc' : 'asc';
    table.setAttribute('data-sort-dir-' + colIdx, dir);
    rows.sort(function(a, b) {
        var cellA = a.children[colIdx] ? a.children[colIdx].textContent.trim() : '';
        var cellB = b.children[colIdx] ? b.children[colIdx].textContent.trim() : '';
        var numA = parseFloat(cellA.replace(/[^0-9.\\-]/g, ''));
        var numB = parseFloat(cellB.replace(/[^0-9.\\-]/g, ''));
        if (!isNaN(numA) && !isNaN(numB)) {
            return dir === 'asc' ? numA - numB : numB - numA;
        }
        if (dir === 'asc') return cellA.localeCompare(cellB);
        return cellB.localeCompare(cellA);
    });
    rows.forEach(function(row, i) {
        row.className = (i % 2 === 0) ? 'even' : 'odd';
        tbody.appendChild(row);
    });
}
/* Auto-bind sortable tables */
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('table.sortable').forEach(function(tbl) {
        var ths = tbl.querySelectorAll('thead th');
        ths.forEach(function(th, idx) {
            if (!th.getAttribute('onclick')) {
                th.addEventListener('click', function() {
                    sortTable(tbl.id, idx);
                });
            }
        });
    });
});

/* ===== Tabs ===== */
function switchTab(groupId, idx) {
    var wrapper = document.getElementById(groupId);
    if (!wrapper) return;
    var btns = wrapper.querySelectorAll('.tab-btn');
    var panels = wrapper.querySelectorAll('.tab-panel');
    btns.forEach(function(b, i) {
        b.classList.toggle('active', i === idx);
    });
    panels.forEach(function(p, i) {
        p.classList.toggle('active', i === idx);
    });
}

/* ===== Collapsible ===== */
function toggleCollapsible(id) {
    var el = document.getElementById(id);
    if (!el) return;
    var btn = el.previousElementSibling;
    if (el.style.display === 'block') {
        el.style.display = 'none';
        if (btn) btn.classList.remove('active');
    } else {
        el.style.display = 'block';
        if (btn) btn.classList.add('active');
    }
}

/* ===== Table search / filter ===== */
function filterTable(tableId, query) {
    var table = document.getElementById(tableId);
    if (!table) return;
    var tbody = table.querySelector('tbody');
    if (!tbody) return;
    var rows = tbody.querySelectorAll('tr');
    var q = query.toLowerCase();
    rows.forEach(function(row) {
        var text = row.textContent.toLowerCase();
        row.style.display = text.indexOf(q) >= 0 ? '' : 'none';
    });
}

/* ===== Smooth scroll for nav links ===== */
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.sidebar-nav a, .toc a').forEach(function(a) {
        a.addEventListener('click', function(e) {
            var href = a.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                var target = document.getElementById(href.substring(1));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    history.pushState(null, '', href);
                }
            }
        });
    });
});

/* ===== Highlight active nav link on scroll ===== */
(function() {
    var sections = [];
    var links = [];
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.sidebar-nav a').forEach(function(a) {
            var href = a.getAttribute('href');
            if (href && href.startsWith('#')) {
                var sec = document.getElementById(href.substring(1));
                if (sec) {
                    sections.push(sec);
                    links.push(a);
                }
            }
        });
    });
    window.addEventListener('scroll', function() {
        var scrollPos = window.scrollY + 120;
        var activeIdx = 0;
        for (var i = 0; i < sections.length; i++) {
            if (sections[i].offsetTop <= scrollPos) {
                activeIdx = i;
            }
        }
        links.forEach(function(l, i) {
            l.style.fontWeight = (i === activeIdx) ? '700' : '';
            l.style.color = (i === activeIdx) ? '#fff' : '';
        });
    });
})();

/* ===== Print helper ===== */
function printReport() {
    window.print();
}

/* ===== Copy code block ===== */
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.code-block').forEach(function(pre) {
        var btn = document.createElement('button');
        btn.textContent = 'Copy';
        btn.style.cssText = 'position:absolute;top:4px;right:4px;font-size:0.7rem;' +
            'padding:2px 8px;border:1px solid #ccc;border-radius:3px;cursor:pointer;' +
            'background:#fff;color:#333;opacity:0;transition:opacity 0.2s;';
        pre.style.position = 'relative';
        pre.appendChild(btn);
        pre.addEventListener('mouseenter', function() { btn.style.opacity = '1'; });
        pre.addEventListener('mouseleave', function() { btn.style.opacity = '0'; });
        btn.addEventListener('click', function() {
            var code = pre.querySelector('code');
            if (code) {
                navigator.clipboard.writeText(code.textContent).then(function() {
                    btn.textContent = 'Copied!';
                    setTimeout(function() { btn.textContent = 'Copy'; }, 1500);
                });
            }
        });
    });
});

/* ===== Export table to CSV ===== */
function exportTableCSV(tableId, filename) {
    var table = document.getElementById(tableId);
    if (!table) return;
    var csv = [];
    var rows = table.querySelectorAll('tr');
    rows.forEach(function(row) {
        var cols = row.querySelectorAll('th, td');
        var rowData = [];
        cols.forEach(function(col) {
            var text = col.textContent.replace(/"/g, '""');
            rowData.push('"' + text + '"');
        });
        csv.push(rowData.join(','));
    });
    var blob = new Blob([csv.join('\\n')], { type: 'text/csv' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename || (tableId + '.csv');
    a.click();
}
</script>
"""

    # ------------------------------------------------------------------
    # Formatting utilities
    # ------------------------------------------------------------------

    def _format_number(self, value: Any, precision: int = 4) -> str:
        return _format_number_util(value, precision)

    def _format_percentage(self, value: Any) -> str:
        return _format_percentage_util(value)

    def _conditional_color(
        self,
        value: float,
        thresholds: Optional[List[Tuple[float, str]]] = None,
    ) -> str:
        return _conditional_color_util(value, thresholds)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_report(self, html_content: str, path: str) -> None:
        """Write *html_content* to *path*, creating directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")

    def _minify_html(self, html_str: str) -> str:
        """Very lightweight HTML minification (no dependencies)."""
        text = re.sub(r"<!--.*?-->", "", html_str, flags=re.DOTALL)
        text = re.sub(r">\s+<", "><", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rankings(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Compute per-metric rankings from *results*.

        Returns a dict mapping metric names to ordered lists of algorithm
        names (best first).
        """
        algo_names = sorted(results.keys())
        all_metrics: set = set()
        for mvals in results.values():
            if isinstance(mvals, dict):
                all_metrics.update(mvals.keys())

        rankings: Dict[str, List[str]] = {}
        for metric in sorted(all_metrics):
            scored: List[Tuple[str, float]] = []
            for algo in algo_names:
                mvals = results[algo]
                if isinstance(mvals, dict):
                    val = mvals.get(metric)
                    if val is not None:
                        try:
                            scored.append((algo, float(val)))
                        except (TypeError, ValueError):
                            pass
            scored.sort(key=lambda t: t[1], reverse=True)
            rankings[metric] = [s[0] for s in scored]
        return rankings

    def _strengths_weaknesses_html(self, metrics: Any) -> str:
        """Produce a simple strengths / weaknesses summary."""
        if not isinstance(metrics, dict):
            return "<p>No metric data available.</p>"

        numeric: Dict[str, float] = {}
        for k, v in metrics.items():
            if k == "config" or isinstance(v, (dict, list)):
                continue
            try:
                numeric[k] = float(v)
            except (TypeError, ValueError):
                pass

        if not numeric:
            return "<p>No numeric metrics to analyse.</p>"

        sorted_metrics = sorted(numeric.items(), key=lambda t: t[1], reverse=True)
        n = max(1, len(sorted_metrics) // 3)
        strengths = sorted_metrics[:n]
        weaknesses = sorted_metrics[-n:]

        parts = ["<h4>Strengths</h4><ul>"]
        for name, val in strengths:
            parts.append(
                f"<li><strong>{_esc(name)}</strong>: "
                f"{_format_number_util(val)}</li>"
            )
        parts.append("</ul>")

        parts.append("<h4>Weaknesses</h4><ul>")
        for name, val in weaknesses:
            parts.append(
                f"<li><strong>{_esc(name)}</strong>: "
                f"{_format_number_util(val)}</li>"
            )
        parts.append("</ul>")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_experiment_report(
    results: Dict[str, Any],
    title: str = "Experiment Report",
    theme: str = "light",
    output_path: Optional[str] = None,
    plots: Optional[Dict[str, str]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    """One-call convenience wrapper to generate an experiment report."""
    config = ReportConfig(title=title, theme=theme)
    gen = HTMLReportGenerator(config)
    html = gen.generate_experiment_report(results, plots=plots, stats=stats)
    if output_path:
        gen.save_report(html, output_path)
    return html


def quick_comparison_report(
    comparisons: Dict[str, Any],
    title: str = "Algorithm Comparison",
    theme: str = "light",
    output_path: Optional[str] = None,
) -> str:
    """One-call convenience wrapper for comparison reports."""
    config = ReportConfig(title=title, theme=theme)
    gen = HTMLReportGenerator(config)
    html = gen.generate_comparison_report(comparisons)
    if output_path:
        gen.save_report(html, output_path)
    return html


def quick_taxonomy_report(
    correlations: Optional[Dict[str, Any]] = None,
    clusters: Optional[Dict[str, Any]] = None,
    title: str = "Metric Taxonomy",
    theme: str = "light",
    output_path: Optional[str] = None,
) -> str:
    """One-call convenience wrapper for metric taxonomy reports."""
    config = ReportConfig(title=title, theme=theme)
    gen = HTMLReportGenerator(config)
    html = gen.generate_metric_taxonomy_report(correlations, clusters)
    if output_path:
        gen.save_report(html, output_path)
    return html


# ---------------------------------------------------------------------------
# MarkdownReportGenerator  (lightweight alternative)
# ---------------------------------------------------------------------------

class MarkdownReportGenerator:
    """Generate Markdown reports instead of HTML.

    Shares the same ``ReportConfig`` as the HTML generator, but outputs
    GFM-compatible Markdown with tables and headings.
    """

    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    def generate_experiment_report(
        self,
        results: Dict[str, Any],
        plots: Optional[Dict[str, str]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        plots = plots or {}
        stats = stats or {}
        lines: List[str] = []
        lines.append(f"# {self.config.title}\n")
        if self.config.author:
            lines.append(f"**Author:** {self.config.author}  ")
        lines.append(f"**Date:** {self.config.date}\n")

        lines.append("## Experiment Overview\n")
        algo_names = sorted(results.keys())
        lines.append(f"Algorithms evaluated: **{len(algo_names)}**\n")

        all_metrics: set = set()
        for mvals in results.values():
            if isinstance(mvals, dict):
                all_metrics.update(mvals.keys())

        if all_metrics:
            lines.append("## Results\n")
            headers = ["Algorithm"] + sorted(all_metrics)
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for algo in algo_names:
                mvals = results[algo] if isinstance(results[algo], dict) else {}
                row = [algo] + [
                    _format_number_util(mvals.get(m)) for m in sorted(all_metrics)
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        if stats:
            lines.append("## Statistical Analysis\n")
            for key, val in stats.items():
                lines.append(f"- **{key}**: {val}")
            lines.append("")

        return "\n".join(lines)

    def generate_comparison_report(
        self,
        comparisons: Dict[str, Any],
    ) -> str:
        lines = [f"# {self.config.title}\n"]
        for pair, cdata in comparisons.items():
            lines.append(f"## {pair.replace('_', ' ').title()}\n")
            if isinstance(cdata, dict):
                for k, v in cdata.items():
                    if not isinstance(v, (dict, list)):
                        lines.append(f"- **{k}**: {v}")
                lines.append("")
        return "\n".join(lines)

    def save_report(self, content: str, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# ReportPipeline — orchestrate multi-page report suites
# ---------------------------------------------------------------------------

class ReportPipeline:
    """Generate a complete report suite with an index page linking to
    individual algorithm profiles, comparison pages, etc.
    """

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        self._gen = HTMLReportGenerator(config)
        self._pages: Dict[str, str] = {}

    def add_page(self, name: str, html: str) -> None:
        """Register a named HTML page."""
        self._pages[name] = html

    def build_suite(
        self,
        results: Dict[str, Any],
        comparisons: Optional[Dict[str, Any]] = None,
        correlations: Optional[Dict[str, Any]] = None,
        clusters: Optional[Dict[str, Any]] = None,
        plots: Optional[Dict[str, str]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Generate all pages and return ``{filename: html}``."""
        pages: Dict[str, str] = {}

        # main experiment report
        pages["index.html"] = self._gen.generate_experiment_report(
            results, plots=plots, stats=stats
        )

        # per-algorithm profiles
        for algo_name in sorted(results.keys()):
            algo_results = results[algo_name]
            if not isinstance(algo_results, dict):
                algo_results = {"value": algo_results}
            fname = f"algo_{_slugify(algo_name)}.html"
            pages[fname] = self._gen.generate_algorithm_profile(
                algo_name, algo_results
            )

        # comparison
        if comparisons:
            pages["comparison.html"] = self._gen.generate_comparison_report(
                comparisons
            )

        # taxonomy
        if correlations or clusters:
            pages["taxonomy.html"] = self._gen.generate_metric_taxonomy_report(
                correlations, clusters
            )

        # user-registered pages
        pages.update(self._pages)

        return pages

    def save_suite(
        self,
        pages: Dict[str, str],
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Write all pages to *output_dir* and return file paths."""
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for fname, html_content in pages.items():
            p = out / fname
            p.write_text(html_content, encoding="utf-8")
            paths.append(str(p))
        return paths


# ---------------------------------------------------------------------------
# TableFormatter — standalone table formatting utilities
# ---------------------------------------------------------------------------

class TableFormatter:
    """Utility class for formatting data into HTML tables with
    conditional formatting, heatmaps, and sparklines.
    """

    @staticmethod
    def heatmap_table(
        headers: List[str],
        rows: List[List[Any]],
        value_columns: Optional[List[int]] = None,
        caption: str = "",
        low_color: str = "#fee2e2",
        high_color: str = "#dcfce7",
    ) -> str:
        """Render a table where numeric cells are background-coloured
        on a gradient from *low_color* to *high_color*.
        """
        value_columns = value_columns or list(range(1, len(headers)))
        # collect min/max per column
        col_ranges: Dict[int, Tuple[float, float]] = {}
        for ci in value_columns:
            vals: List[float] = []
            for row in rows:
                if ci < len(row):
                    try:
                        vals.append(float(row[ci]))
                    except (TypeError, ValueError):
                        pass
            if vals:
                col_ranges[ci] = (min(vals), max(vals))

        tid = f"hm-{uuid.uuid4().hex[:8]}"
        parts = [f'<div class="table-wrapper">']
        if caption:
            parts.append(f'<p class="table-caption">{_esc(caption)}</p>')
        parts.append(f'<table id="{tid}" class="data-table sortable">')
        parts.append("<thead><tr>")
        for h in headers:
            parts.append(f"<th>{_esc(str(h))}</th>")
        parts.append("</tr></thead><tbody>")
        for ri, row in enumerate(rows):
            cls = "even" if ri % 2 == 0 else "odd"
            parts.append(f'<tr class="{cls}">')
            for ci, cell in enumerate(row):
                style = ""
                if ci in col_ranges:
                    lo, hi = col_ranges[ci]
                    try:
                        fv = float(cell)
                        ratio = (fv - lo) / (hi - lo) if hi != lo else 0.5
                        r_lo, g_lo, b_lo = TableFormatter._hex_to_rgb(low_color)
                        r_hi, g_hi, b_hi = TableFormatter._hex_to_rgb(high_color)
                        r = int(r_lo + (r_hi - r_lo) * ratio)
                        g = int(g_lo + (g_hi - g_lo) * ratio)
                        b = int(b_lo + (b_hi - b_lo) * ratio)
                        bg = f"#{r:02x}{g:02x}{b:02x}"
                        style = f' style="background:{bg}"'
                    except (TypeError, ValueError):
                        pass
                parts.append(f"<td{style}>{_esc(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>\n")
        return "\n".join(parts)

    @staticmethod
    def sparkline_svg(
        values: List[float],
        width: int = 120,
        height: int = 24,
        color: str = "#3182ce",
    ) -> str:
        """Generate a tiny inline SVG sparkline."""
        if not values:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1.0
        n = len(values)
        step = width / max(n - 1, 1)
        points: List[str] = []
        for i, v in enumerate(values):
            x = i * step
            y = height - ((v - mn) / rng) * (height - 2) - 1
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)
        return (
            f'<svg width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<polyline points="{polyline}" fill="none" '
            f'stroke="{_esc(color)}" stroke-width="1.5"/></svg>'
        )

    @staticmethod
    def rank_table(
        data: Dict[str, Dict[str, float]],
        higher_is_better: bool = True,
        caption: str = "",
    ) -> str:
        """Build a table showing rank of each algorithm per metric.

        *data* maps algorithm names to dicts of metric → value.
        """
        algos = sorted(data.keys())
        metrics: set = set()
        for v in data.values():
            if isinstance(v, dict):
                metrics.update(v.keys())
        metric_list = sorted(metrics)

        # compute ranks
        ranks: Dict[str, Dict[str, int]] = {a: {} for a in algos}
        for m in metric_list:
            scored = []
            for a in algos:
                val = data[a].get(m) if isinstance(data[a], dict) else None
                if val is not None:
                    try:
                        scored.append((a, float(val)))
                    except (TypeError, ValueError):
                        pass
            scored.sort(key=lambda t: t[1], reverse=higher_is_better)
            for rank, (a, _) in enumerate(scored, 1):
                ranks[a][m] = rank

        headers = ["Algorithm"] + metric_list + ["Avg Rank"]
        rows: List[List[Any]] = []
        for a in algos:
            row: List[Any] = [a]
            r_vals: List[int] = []
            for m in metric_list:
                r = ranks[a].get(m, len(algos))
                row.append(r)
                r_vals.append(r)
            avg_r = sum(r_vals) / len(r_vals) if r_vals else 0
            row.append(f"{avg_r:.2f}")
            rows.append(row)

        rows.sort(key=lambda r: float(r[-1]))
        tid = f"rank-{uuid.uuid4().hex[:8]}"
        parts = ['<div class="table-wrapper">']
        if caption:
            parts.append(f'<p class="table-caption">{_esc(caption)}</p>')
        parts.append(f'<table id="{tid}" class="data-table sortable">')
        parts.append("<thead><tr>")
        for h in headers:
            parts.append(f"<th>{_esc(str(h))}</th>")
        parts.append("</tr></thead><tbody>")
        for ri, row in enumerate(rows):
            cls = "even" if ri % 2 == 0 else "odd"
            parts.append(f'<tr class="{cls}">')
            for ci, cell in enumerate(row):
                medal = ""
                if ci == 0 and ri == 0:
                    medal = " 🥇"
                elif ci == 0 and ri == 1:
                    medal = " 🥈"
                elif ci == 0 and ri == 2:
                    medal = " 🥉"
                parts.append(f"<td>{_esc(str(cell))}{medal}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>\n")
        return "\n".join(parts)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )


# ---------------------------------------------------------------------------
# MetricDashboard — specialised dashboard for metric analysis
# ---------------------------------------------------------------------------

class MetricDashboard:
    """Build a dashboard-style HTML page focused on metric analysis."""

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        self._gen = HTMLReportGenerator(config)

    def build(
        self,
        metric_data: Dict[str, Dict[str, float]],
        correlations: Optional[Dict[str, Dict[str, float]]] = None,
        distributions: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        """Generate a full metric dashboard HTML page.

        *metric_data* maps algorithm names to metric dicts.
        *correlations* maps metric pairs to correlation coefficients.
        *distributions* maps metric names to lists of observed values.
        """
        builder = ReportBuilder(
            ReportConfig(
                title="Metric Dashboard",
                theme=self.config.theme,
                author=self.config.author,
                date=self.config.date,
                custom_css=self.config.custom_css,
            )
        )

        # overview
        overview = ReportSection(title="Dashboard Overview", level=1,
                                 section_id="dashboard-overview")
        algo_count = len(metric_data)
        metric_names: set = set()
        for v in metric_data.values():
            if isinstance(v, dict):
                metric_names.update(v.keys())
        overview.content = (
            f"<p>Algorithms: <strong>{algo_count}</strong> &nbsp;|&nbsp; "
            f"Metrics: <strong>{len(metric_names)}</strong></p>\n"
        )
        builder.add_section(overview)

        # summary cards
        summary_sec = ReportSection(title="Metric Averages", level=2,
                                    section_id="metric-averages")
        builder.add_section(summary_sec)
        averages: Dict[str, float] = {}
        for m in sorted(metric_names):
            vals: List[float] = []
            for algo_metrics in metric_data.values():
                if isinstance(algo_metrics, dict):
                    v = algo_metrics.get(m)
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            pass
            if vals:
                averages[m] = sum(vals) / len(vals)
        builder.add_metric_summary(averages, "metric-averages")

        # heatmap table
        heatmap_sec = ReportSection(title="Heatmap", level=2,
                                    section_id="heatmap")
        builder.add_section(heatmap_sec)
        headers = ["Algorithm"] + sorted(metric_names)
        rows = []
        for algo in sorted(metric_data.keys()):
            row: List[Any] = [algo]
            mvals = metric_data[algo] if isinstance(metric_data[algo], dict) else {}
            for m in sorted(metric_names):
                row.append(_format_number_util(mvals.get(m)))
            rows.append(row)
        heatmap_html = TableFormatter.heatmap_table(
            headers, rows,
            value_columns=list(range(1, len(headers))),
            caption="Algorithm × Metric Heatmap",
        )
        builder.add_text(heatmap_html, "heatmap")

        # rank table
        rank_sec = ReportSection(title="Rankings", level=2,
                                 section_id="rankings")
        builder.add_section(rank_sec)
        rank_html = TableFormatter.rank_table(
            metric_data, caption="Algorithm Rankings"
        )
        builder.add_text(rank_html, "rankings")

        # distributions (sparklines)
        if distributions:
            dist_sec = ReportSection(title="Distributions", level=2,
                                     section_id="distributions")
            builder.add_section(dist_sec)
            for mname, vals in sorted(distributions.items()):
                spark = TableFormatter.sparkline_svg(vals)
                builder.add_text(
                    f"<p><strong>{_esc(mname)}</strong>: {spark} "
                    f"(n={len(vals)}, μ={sum(vals)/len(vals):.4f})</p>\n",
                    "distributions",
                )

        # correlations
        if correlations:
            corr_sec = ReportSection(title="Correlations", level=2,
                                     section_id="correlations")
            builder.add_section(corr_sec)
            corr_metrics = sorted(correlations.keys())
            c_headers = [""] + corr_metrics
            c_rows = []
            for m1 in corr_metrics:
                row: List[Any] = [m1]
                m1c = correlations[m1]
                if isinstance(m1c, dict):
                    for m2 in corr_metrics:
                        row.append(_format_number_util(m1c.get(m2, ""), 3))
                else:
                    row.extend([""] * len(corr_metrics))
                c_rows.append(row)
            c_html = TableFormatter.heatmap_table(
                c_headers, c_rows,
                value_columns=list(range(1, len(c_headers))),
                caption="Metric Correlation Matrix",
                low_color="#fee2e2",
                high_color="#dbeafe",
            )
            builder.add_text(c_html, "correlations")

        return builder.build()


# ---------------------------------------------------------------------------
# ExperimentLogger — accumulate results during a run, then report
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Lightweight logger that collects experiment data and can emit a
    report at the end of a run.
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        self.config = config or ReportConfig()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._plots: Dict[str, str] = {}
        self._stats: Dict[str, Any] = {}
        self._comparisons: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}

    def log_result(self, algorithm: str, metrics: Dict[str, Any]) -> None:
        """Record metrics for an algorithm."""
        if algorithm not in self._results:
            self._results[algorithm] = {}
        self._results[algorithm].update(metrics)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record a timestamped event."""
        self._events.append({
            "type": event_type,
            "time": datetime.now().isoformat(),
            **data,
        })

    def log_plot(self, name: str, svg: str) -> None:
        self._plots[name] = svg

    def log_stat(self, name: str, value: Any) -> None:
        self._stats[name] = value

    def log_comparison(self, pair: str, data: Dict[str, Any]) -> None:
        self._comparisons[pair] = data

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Produce the final HTML report from accumulated data."""
        gen = HTMLReportGenerator(self.config)
        report_html = gen.generate_experiment_report(
            self._results, plots=self._plots, stats=self._stats
        )
        if output_path:
            gen.save_report(report_html, output_path)
        return report_html

    def generate_full_suite(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a full multi-page report suite."""
        pipeline = ReportPipeline(self.config)

        # event log page
        if self._events:
            builder = ReportBuilder(
                ReportConfig(
                    title="Event Log",
                    theme=self.config.theme,
                    date=self.config.date,
                )
            )
            ev_sec = ReportSection(title="Events", level=1,
                                   section_id="events")
            builder.add_section(ev_sec)
            ev_headers = ["Time", "Type", "Details"]
            ev_rows = []
            for ev in self._events:
                details = {k: v for k, v in ev.items()
                           if k not in ("type", "time")}
                ev_rows.append([
                    ev.get("time", ""),
                    ev.get("type", ""),
                    str(details) if details else "",
                ])
            builder.add_table(ev_headers, ev_rows,
                              caption="Experiment Event Log",
                              section_id="events", sortable=True)
            pipeline.add_page("events.html", builder.build())

        # metadata page
        if self._metadata:
            builder = ReportBuilder(
                ReportConfig(
                    title="Experiment Metadata",
                    theme=self.config.theme,
                    date=self.config.date,
                )
            )
            meta_sec = ReportSection(title="Metadata", level=1,
                                     section_id="metadata")
            builder.add_section(meta_sec)
            builder.add_key_value_table(self._metadata, title="Metadata",
                                        section_id="metadata")
            pipeline.add_page("metadata.html", builder.build())

        pages = pipeline.build_suite(
            self._results,
            comparisons=self._comparisons if self._comparisons else None,
            plots=self._plots if self._plots else None,
            stats=self._stats if self._stats else None,
        )

        if output_dir:
            pipeline.save_suite(pages, output_dir)

        return pages


# ---------------------------------------------------------------------------
# DiffReportGenerator — before/after comparison
# ---------------------------------------------------------------------------

class DiffReportGenerator:
    """Generate a report comparing two sets of experimental results,
    highlighting improvements and regressions.
    """

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        self._gen = HTMLReportGenerator(config)

    def generate(
        self,
        before: Dict[str, Dict[str, float]],
        after: Dict[str, Dict[str, float]],
        labels: Tuple[str, str] = ("Before", "After"),
    ) -> str:
        """Produce a diff report comparing *before* and *after* results."""
        builder = ReportBuilder(
            ReportConfig(
                title=f"Diff Report: {labels[0]} vs {labels[1]}",
                theme=self.config.theme,
                author=self.config.author,
                date=self.config.date,
                custom_css=self.config.custom_css,
            )
        )

        intro = ReportSection(title="Overview", level=1,
                              section_id="diff-overview")
        intro.content = (
            f"<p>Comparing results from <strong>{_esc(labels[0])}</strong> "
            f"to <strong>{_esc(labels[1])}</strong>.</p>\n"
        )
        builder.add_section(intro)

        all_algos = sorted(set(before.keys()) | set(after.keys()))
        all_metrics: set = set()
        for d in (before, after):
            for v in d.values():
                if isinstance(v, dict):
                    all_metrics.update(v.keys())
        metric_list = sorted(all_metrics)

        # summary of changes
        improvements = 0
        regressions = 0
        unchanged = 0
        total = 0

        detail_rows: List[List[str]] = []
        for algo in all_algos:
            b_vals = before.get(algo, {})
            a_vals = after.get(algo, {})
            if not isinstance(b_vals, dict):
                b_vals = {}
            if not isinstance(a_vals, dict):
                a_vals = {}
            for m in metric_list:
                bv = b_vals.get(m)
                av = a_vals.get(m)
                if bv is None and av is None:
                    continue
                total += 1
                try:
                    fb = float(bv) if bv is not None else 0.0
                    fa = float(av) if av is not None else 0.0
                    diff = fa - fb
                    pct = (diff / fb * 100) if fb != 0 else 0.0
                    if diff > 1e-9:
                        improvements += 1
                        direction = "↑"
                    elif diff < -1e-9:
                        regressions += 1
                        direction = "↓"
                    else:
                        unchanged += 1
                        direction = "="
                except (TypeError, ValueError):
                    diff = 0.0
                    pct = 0.0
                    direction = "?"
                    unchanged += 1
                detail_rows.append([
                    algo, m,
                    _format_number_util(bv),
                    _format_number_util(av),
                    f"{diff:+.4f}",
                    f"{pct:+.2f}%",
                    direction,
                ])

        summary_sec = ReportSection(title="Change Summary", level=2,
                                    section_id="change-summary")
        builder.add_section(summary_sec)
        builder.add_metric_summary(
            {
                "Total Comparisons": total,
                "Improvements ↑": improvements,
                "Regressions ↓": regressions,
                "Unchanged =": unchanged,
            },
            "change-summary",
        )

        if improvements > regressions:
            builder.add_text(
                self._gen._render_alert(
                    f"Net positive: {improvements - regressions} more improvements than regressions.",
                    "success",
                ),
                "change-summary",
            )
        elif regressions > improvements:
            builder.add_text(
                self._gen._render_alert(
                    f"Net negative: {regressions - improvements} more regressions than improvements.",
                    "warning",
                ),
                "change-summary",
            )

        # detail table
        detail_sec = ReportSection(title="Detailed Changes", level=2,
                                   section_id="detailed-changes")
        builder.add_section(detail_sec)
        builder.add_table(
            ["Algorithm", "Metric", labels[0], labels[1], "Diff", "% Change", ""],
            detail_rows,
            caption="All metric changes",
            section_id="detailed-changes",
            sortable=True,
        )

        # per-algorithm tabs
        if len(all_algos) > 1:
            tab_sec = ReportSection(title="Per-Algorithm View", level=2,
                                    section_id="per-algo-view")
            builder.add_section(tab_sec)
            tab_contents: Dict[str, str] = {}
            for algo in all_algos:
                algo_rows = [r for r in detail_rows if r[0] == algo]
                if algo_rows:
                    tbl = self._gen._render_interactive_table(
                        ["Metric", labels[0], labels[1], "Diff", "%", ""],
                        [[r[1]] + r[2:] for r in algo_rows],
                        f"algo-diff-{_slugify(algo)}",
                    )
                    tab_contents[algo] = tbl
            builder.add_text(
                self._gen._render_tabs(tab_contents),
                "per-algo-view",
            )

        return builder.build()


# ---------------------------------------------------------------------------
# ConvergenceReportGenerator — track metric convergence over iterations
# ---------------------------------------------------------------------------

class ConvergenceReportGenerator:
    """Report on how metrics evolve over successive iterations."""

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        self._gen = HTMLReportGenerator(config)

    def generate(
        self,
        iteration_data: List[Dict[str, Dict[str, float]]],
        labels: Optional[List[str]] = None,
    ) -> str:
        """Generate a convergence report.

        *iteration_data* is a list (one per iteration) of
        ``{algorithm: {metric: value}}`` dicts.
        """
        n_iters = len(iteration_data)
        labels = labels or [f"Iter {i}" for i in range(n_iters)]

        builder = ReportBuilder(
            ReportConfig(
                title="Convergence Report",
                theme=self.config.theme,
                author=self.config.author,
                date=self.config.date,
                custom_css=self.config.custom_css,
            )
        )

        overview = ReportSection(title="Convergence Overview", level=1,
                                 section_id="convergence-overview")
        overview.content = (
            f"<p>Tracking convergence across <strong>{n_iters}</strong> "
            f"iterations.</p>\n"
        )
        builder.add_section(overview)

        # collect all algos / metrics
        all_algos: set = set()
        all_metrics: set = set()
        for d in iteration_data:
            all_algos.update(d.keys())
            for v in d.values():
                if isinstance(v, dict):
                    all_metrics.update(v.keys())
        algo_list = sorted(all_algos)
        metric_list = sorted(all_metrics)

        # sparklines per (algo, metric)
        spark_sec = ReportSection(title="Metric Trajectories", level=2,
                                  section_id="trajectories")
        builder.add_section(spark_sec)
        for algo in algo_list:
            for metric in metric_list:
                vals: List[float] = []
                for d in iteration_data:
                    a_data = d.get(algo, {})
                    v = a_data.get(metric) if isinstance(a_data, dict) else None
                    try:
                        vals.append(float(v) if v is not None else 0.0)
                    except (TypeError, ValueError):
                        vals.append(0.0)
                spark = TableFormatter.sparkline_svg(vals)
                first = vals[0] if vals else 0.0
                last = vals[-1] if vals else 0.0
                diff = last - first
                direction = "↑" if diff > 1e-9 else ("↓" if diff < -1e-9 else "=")
                builder.add_text(
                    f"<p><strong>{_esc(algo)}</strong> / {_esc(metric)}: "
                    f"{spark} ({direction} {_format_number_util(abs(diff))})</p>\n",
                    "trajectories",
                )

        # final iteration table
        if iteration_data:
            final = iteration_data[-1]
            final_sec = ReportSection(title="Final Iteration Results", level=2,
                                      section_id="final-results")
            builder.add_section(final_sec)
            headers = ["Algorithm"] + metric_list
            rows = []
            for algo in algo_list:
                row: List[Any] = [algo]
                a_data = final.get(algo, {})
                if isinstance(a_data, dict):
                    for m in metric_list:
                        row.append(_format_number_util(a_data.get(m)))
                else:
                    row.extend([""] * len(metric_list))
                rows.append(row)
            builder.add_table(headers, rows, caption="Final Iteration",
                              section_id="final-results", sortable=True)

        # convergence check
        conv_sec = ReportSection(title="Convergence Check", level=2,
                                 section_id="convergence-check")
        builder.add_section(conv_sec)
        if n_iters >= 3:
            converged_count = 0
            total_count = 0
            for algo in algo_list:
                for metric in metric_list:
                    total_count += 1
                    recent: List[float] = []
                    for d in iteration_data[-3:]:
                        a_data = d.get(algo, {})
                        v = a_data.get(metric) if isinstance(a_data, dict) else None
                        try:
                            recent.append(float(v) if v is not None else 0.0)
                        except (TypeError, ValueError):
                            recent.append(0.0)
                    if recent:
                        rng = max(recent) - min(recent)
                        mean_abs = sum(abs(x) for x in recent) / len(recent) if recent else 1.0
                        rel_var = rng / mean_abs if mean_abs > 1e-12 else 0.0
                        if rel_var < 0.01:
                            converged_count += 1
            pct = converged_count / total_count if total_count else 0
            builder.add_text(
                self._gen._render_progress_bar(pct, 1.0,
                    f"Converged: {converged_count}/{total_count} "
                    f"({_format_percentage_util(pct)})"),
                "convergence-check",
            )
            if pct >= 0.9:
                builder.add_text(
                    self._gen._render_alert("Experiment has converged.", "success"),
                    "convergence-check",
                )
            elif pct >= 0.5:
                builder.add_text(
                    self._gen._render_alert("Partial convergence detected.", "info"),
                    "convergence-check",
                )
            else:
                builder.add_text(
                    self._gen._render_alert(
                        "Experiment has not converged — consider more iterations.",
                        "warning",
                    ),
                    "convergence-check",
                )
        else:
            builder.add_text(
                self._gen._render_alert(
                    "Fewer than 3 iterations — cannot assess convergence.",
                    "info",
                ),
                "convergence-check",
            )

        return builder.build()


# ---------------------------------------------------------------------------
# Standalone rendering helpers (can be used without a full generator)
# ---------------------------------------------------------------------------

def render_standalone_table(
    headers: List[str],
    rows: List[List[Any]],
    caption: str = "",
    sortable: bool = True,
    theme: str = "light",
) -> str:
    """Render a standalone HTML page containing just one table."""
    config = ReportConfig(title=caption or "Table", theme=theme)
    gen = HTMLReportGenerator(config)
    css = gen._css_theme(theme)
    header = gen._html_header(caption or "Table", css)
    footer = gen._html_footer()
    js = gen._inline_js()
    tid = f"standalone-{uuid.uuid4().hex[:8]}"
    body = gen._render_interactive_table(headers, rows, tid) if sortable else ""
    if not sortable:
        body = f'<table class="data-table"><thead><tr>'
        for h in headers:
            body += f"<th>{_esc(str(h))}</th>"
        body += "</tr></thead><tbody>"
        for ri, row in enumerate(rows):
            cls = "even" if ri % 2 == 0 else "odd"
            body += f'<tr class="{cls}">'
            for cell in row:
                body += f"<td>{_esc(str(cell))}</td>"
            body += "</tr>"
        body += "</tbody></table>"
    return f"{header}<main class='main-content' style='margin-left:0'>{body}{js}</main>{footer}"


def render_standalone_cards(
    metrics: Dict[str, Any],
    title: str = "Metrics",
    theme: str = "light",
) -> str:
    """Render a standalone HTML page with metric cards."""
    config = ReportConfig(title=title, theme=theme)
    gen = HTMLReportGenerator(config)
    css = gen._css_theme(theme)
    header = gen._html_header(title, css)
    footer = gen._html_footer()
    cards = '<div class="metric-cards">\n'
    for name, value in metrics.items():
        cards += (
            f'<div class="metric-card">'
            f'<div class="metric-name">{_esc(str(name))}</div>'
            f'<div class="metric-value">{_esc(_format_number_util(value))}</div>'
            f"</div>\n"
        )
    cards += "</div>\n"
    return f"{header}<main class='main-content' style='margin-left:0'><h1>{_esc(title)}</h1>{cards}</main>{footer}"


# ---------------------------------------------------------------------------
# Module-level __all__
# ---------------------------------------------------------------------------

__all__ = [
    "ReportConfig",
    "ReportSection",
    "ReportBuilder",
    "HTMLReportGenerator",
    "MarkdownReportGenerator",
    "ReportPipeline",
    "TableFormatter",
    "MetricDashboard",
    "ExperimentLogger",
    "DiffReportGenerator",
    "ConvergenceReportGenerator",
    "quick_experiment_report",
    "quick_comparison_report",
    "quick_taxonomy_report",
    "render_standalone_table",
    "render_standalone_cards",
]
