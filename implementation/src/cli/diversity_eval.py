#!/usr/bin/env python3
"""
Diversity Evaluation CLI — compute all diversity metrics and discover
which ones are redundant vs. complementary on your own generated texts.

Unique value: no other tool computes a full metric taxonomy showing
which diversity metrics capture the same signal and which are independent.

Usage
-----
    # Evaluate a single group of texts
    python -m src.cli.diversity_eval --input texts.json --format table

    # Pipe from stdin (one text per line)
    cat texts.txt | python -m src.cli.diversity_eval --format json

    # Multiple groups → correlation & redundancy analysis
    python -m src.cli.diversity_eval --input groups.json --format table

Input formats
-------------
Single group (list of strings):
    ["text1", "text2", "text3"]

Multiple groups (dict of name → list):
    {
      "temperature_0.7": ["text1", "text2"],
      "temperature_1.0": ["text3", "text4"]
    }
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- project imports (work with both `python -m src.cli.diversity_eval`
#     and direct invocation) ---
try:
    from src.metrics.diversity import (
        BehavioralDiversity,
        DistinctN,
        DiversityMetricSuite,
        EmbeddingPairwiseDistance,
        NGramEntropy,
        SelfBLEU,
        VendiScore,
    )
    from src.metrics.correlation import (
        MetricCorrelationAnalyzer,
        MetricRedundancyAnalyzer,
        SpectralClusterer,
        effective_dimensionality,
    )
except ImportError:
    # Fall back for running from the implementation/ directory
    from metrics.diversity import (  # type: ignore[no-redef]
        BehavioralDiversity,
        DistinctN,
        DiversityMetricSuite,
        EmbeddingPairwiseDistance,
        NGramEntropy,
        SelfBLEU,
        VendiScore,
    )
    from metrics.correlation import (  # type: ignore[no-redef]
        MetricCorrelationAnalyzer,
        MetricRedundancyAnalyzer,
        SpectralClusterer,
        effective_dimensionality,
    )

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Default metric suite (6 core metrics)
# -----------------------------------------------------------------------

CORE_METRIC_NAMES = [
    "self_bleu_4",
    "distinct_1",
    "distinct_2",
    "entropy_2",
    "embedding_pairwise_cosine",
    "vendi_cosine",
]


def build_core_suite() -> DiversityMetricSuite:
    """Return the 6-metric suite used by default."""
    return DiversityMetricSuite([
        SelfBLEU(max_order=4),
        DistinctN(n=1),
        DistinctN(n=2),
        NGramEntropy(n=2),
        EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method="tfidf"
        ),
        VendiScore(kernel_type="cosine"),
    ])


# -----------------------------------------------------------------------
# Input parsing
# -----------------------------------------------------------------------


def load_input(path: Optional[str]) -> Dict[str, List[str]]:
    """Load texts from *path* (or stdin) and return ``{group: [texts]}``."""
    if path is None or path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()

    raw = raw.strip()

    # Try JSON first
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Plain text: one text per line
        lines = [l for l in raw.splitlines() if l.strip()]
        if len(lines) < 2:
            print("Error: need at least 2 texts.", file=sys.stderr)
            sys.exit(1)
        return {"default": lines}

    if isinstance(data, list):
        return {"default": data}
    if isinstance(data, dict):
        groups: Dict[str, List[str]] = {}
        for k, v in data.items():
            if isinstance(v, list) and all(isinstance(s, str) for s in v):
                groups[k] = v
            else:
                print(
                    f"Warning: skipping group '{k}' (expected list of strings).",
                    file=sys.stderr,
                )
        if not groups:
            print("Error: no valid groups found.", file=sys.stderr)
            sys.exit(1)
        return groups
    print("Error: JSON must be a list of strings or a dict of groups.",
          file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------


def compute_group_metrics(
    suite: DiversityMetricSuite,
    groups: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """Return ``{group_name: {metric_name: value}}``."""
    results: Dict[str, Dict[str, float]] = {}
    for group_name, texts in groups.items():
        results[group_name] = suite.compute_all(texts)
    return results


# -----------------------------------------------------------------------
# Correlation / taxonomy analysis
# -----------------------------------------------------------------------


def run_taxonomy_analysis(
    results: Dict[str, Dict[str, float]],
    metric_names: List[str],
    redundancy_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Build correlation matrix across groups and identify redundancy.

    Requires ≥3 groups for meaningful correlation.
    """
    n_groups = len(results)
    if n_groups < 3:
        return {
            "status": "skipped",
            "reason": f"Need ≥3 groups for correlation analysis (got {n_groups}).",
        }

    # Build metric_values: {metric: [val_per_group]}
    group_names = list(results.keys())
    metric_values: Dict[str, List[float]] = {m: [] for m in metric_names}
    for g in group_names:
        for m in metric_names:
            metric_values[m].append(results[g].get(m, float("nan")))

    # Drop metrics that are constant (zero variance)
    active_metrics = [
        m for m in metric_names
        if np.nanstd(metric_values[m]) > 1e-12
    ]
    if len(active_metrics) < 2:
        return {
            "status": "skipped",
            "reason": "Fewer than 2 non-constant metrics — nothing to correlate.",
        }

    analyzer = MetricCorrelationAnalyzer(
        metrics=active_metrics, confidence_level=0.95
    )
    active_values = {m: metric_values[m] for m in active_metrics}
    corr_matrix = analyzer.compute_correlation_matrix(active_values)

    # Redundancy analysis
    redundancy = MetricRedundancyAnalyzer(threshold=redundancy_threshold)
    redundant_pairs = redundancy.find_redundant_pairs(corr_matrix, active_metrics)
    orthogonal_pairs = redundancy.find_orthogonal_pairs(corr_matrix, active_metrics)
    representatives = redundancy.select_representative_metrics(
        corr_matrix, active_metrics
    )
    independence = redundancy.metric_independence_score(corr_matrix)
    info_content = redundancy.information_content(corr_matrix)

    # Cluster taxonomy
    n_clusters = analyzer.optimal_num_clusters(corr_matrix)
    clusters_idx = analyzer.discover_metric_clusters(corr_matrix, n_clusters)
    clusters = [[active_metrics[i] for i in c] for c in clusters_idx]

    return {
        "status": "ok",
        "correlation_matrix": {
            "metrics": active_metrics,
            "values": corr_matrix.tolist(),
        },
        "redundant_pairs": [
            {"a": a, "b": b, "tau": round(tau, 4)}
            for a, b, tau in redundant_pairs
        ],
        "orthogonal_pairs": [
            {"a": a, "b": b, "tau": round(tau, 4)}
            for a, b, tau in orthogonal_pairs
        ],
        "metric_clusters": clusters,
        "recommended_subset": representatives,
        "independence_score": round(independence, 4),
        "effective_dimensionality": round(info_content, 2),
        "n_metrics": len(active_metrics),
    }


# -----------------------------------------------------------------------
# Formatters
# -----------------------------------------------------------------------


def _header_line(text: str, char: str = "=") -> str:
    return f"\n{text}\n{char * len(text)}"


def format_table(
    results: Dict[str, Dict[str, float]],
    taxonomy: Dict[str, Any],
    suite: DiversityMetricSuite,
) -> str:
    """Human-readable table output."""
    lines: List[str] = []

    # --- Per-group metric scores ---
    lines.append(_header_line("Diversity Metric Scores"))
    metric_names = suite.metric_names
    col_w = max(len(m) for m in metric_names) + 2
    group_w = max((len(g) for g in results), default=10) + 2

    # Header row
    header = f"{'Metric':<{col_w}}"
    for g in results:
        header += f"  {g:>{group_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for m in metric_names:
        row = f"{m:<{col_w}}"
        for g in results:
            v = results[g].get(m, float("nan"))
            row += f"  {v:>{group_w}.4f}"
        lines.append(row)

    # --- Taxonomy (if computed) ---
    if taxonomy.get("status") == "ok":
        lines.append(_header_line("Metric Taxonomy (Correlation Analysis)"))

        # Correlation matrix
        active = taxonomy["correlation_matrix"]["metrics"]
        matrix = taxonomy["correlation_matrix"]["values"]
        mc = max(len(n) for n in active) + 1
        hdr = " " * mc + "  ".join(f"{n[:8]:>8}" for n in active)
        lines.append(hdr)
        for i, name in enumerate(active):
            row_vals = "  ".join(f"{matrix[i][j]:>8.3f}" for j in range(len(active)))
            lines.append(f"{name:<{mc}}{row_vals}")

        # Redundant pairs
        lines.append(_header_line("Redundant Pairs (|τ| ≥ 0.8)", "-"))
        if taxonomy["redundant_pairs"]:
            for p in taxonomy["redundant_pairs"]:
                lines.append(f"  {p['a']}  ↔  {p['b']}  (τ = {p['tau']:+.3f})")
        else:
            lines.append("  None — all metrics capture different signals!")

        # Orthogonal pairs
        lines.append(_header_line("Complementary Pairs (|τ| ≤ 0.1)", "-"))
        if taxonomy["orthogonal_pairs"]:
            for p in taxonomy["orthogonal_pairs"]:
                lines.append(f"  {p['a']}  ↔  {p['b']}  (τ = {p['tau']:+.3f})")
        else:
            lines.append("  None found at threshold 0.1")

        # Clusters
        lines.append(_header_line("Metric Clusters", "-"))
        for i, cluster in enumerate(taxonomy["metric_clusters"], 1):
            lines.append(f"  Cluster {i}: {', '.join(cluster)}")

        # Recommendation
        lines.append(_header_line("Recommendation", "-"))
        lines.append(
            f"  Independence score: {taxonomy['independence_score']:.3f}  "
            f"(1.0 = all independent)"
        )
        lines.append(
            f"  Effective dimensionality: {taxonomy['effective_dimensionality']:.1f}"
            f" / {taxonomy['n_metrics']}"
        )
        lines.append(
            f"  Recommended subset: {', '.join(taxonomy['recommended_subset'])}"
        )
        lines.append(
            "  → Use these metrics for a non-redundant diversity evaluation."
        )
    elif taxonomy.get("status") == "skipped":
        lines.append(f"\n[Taxonomy skipped] {taxonomy['reason']}")

    return "\n".join(lines)


def format_json(
    results: Dict[str, Dict[str, float]],
    taxonomy: Dict[str, Any],
) -> str:
    """Machine-readable JSON output."""
    payload = {
        "metric_scores": {
            g: {m: round(v, 6) for m, v in vals.items()}
            for g, vals in results.items()
        },
        "taxonomy": taxonomy,
    }
    return json.dumps(payload, indent=2)


def format_csv(
    results: Dict[str, Dict[str, float]],
) -> str:
    """CSV with one row per group."""
    if not results:
        return ""
    groups = list(results.keys())
    metrics = list(results[groups[0]].keys())
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["group"] + metrics)
    for g in groups:
        row = [g] + [f"{results[g].get(m, float('nan')):.6f}" for m in metrics]
        writer.writerow(row)
    return buf.getvalue()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diversity_eval",
        description=(
            "Compute diversity metrics on LLM-generated texts and discover "
            "which metrics are redundant vs. complementary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Single file of texts
              python -m src.cli.diversity_eval --input texts.json

              # Pipe from stdin (one text per line)
              echo -e "hello world\\nhola mundo\\nbonjour le monde" | python -m src.cli.diversity_eval

              # Multiple groups with JSON output
              python -m src.cli.diversity_eval --input groups.json --format json

              # Custom redundancy threshold
              python -m src.cli.diversity_eval --input groups.json --redundancy-threshold 0.7
        """),
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to JSON file (list or dict of groups). Default: stdin.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.8,
        help="Kendall τ threshold for flagging redundant pairs (default: 0.8).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress warnings.",
    )
    return parser


import textwrap  # noqa: E402 (late import to keep top clean)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load input
    groups = load_input(args.input)

    # Build suite & compute
    suite = build_core_suite()
    results = compute_group_metrics(suite, groups)

    # Taxonomy analysis (multi-group only)
    taxonomy = run_taxonomy_analysis(
        results,
        suite.metric_names,
        redundancy_threshold=args.redundancy_threshold,
    )

    # Output
    if args.format == "json":
        print(format_json(results, taxonomy))
    elif args.format == "csv":
        print(format_csv(results))
    else:
        print(format_table(results, taxonomy, suite))


if __name__ == "__main__":
    main()
