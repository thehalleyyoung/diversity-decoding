#!/usr/bin/env python3
"""
DivFlow: Simple CLI for diversity metric computation and selection.

Usage:
    python -m src.cli.divflow metrics --texts "Hello world" "Goodbye world" "Hi there"
    python -m src.cli.divflow select --n-items 200 --k 10 --method farthest_point
    python -m src.cli.divflow benchmark --n 200 --k 10
    python -m src.cli.divflow cross-model --synthetic
    python -m src.cli.divflow sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def cmd_metrics(args):
    """Compute diversity metrics on input texts."""
    from src.cross_model_analysis import METRIC_FUNCTIONS

    if args.file:
        from src.io.jsonl_loader import load_texts_auto
        texts = load_texts_auto(args.file, text_field=args.text_field)
    elif args.texts:
        texts = args.texts
    else:
        print("Error: provide --texts or --file", file=sys.stderr)
        return 1

    print(f"Computing diversity metrics on {len(texts)} texts...\n")
    results = {}
    for name, fn in METRIC_FUNCTIONS.items():
        try:
            val = fn(texts)
            results[name] = round(val, 4)
            print(f"  {name:>12s}: {val:.4f}")
        except Exception as e:
            results[name] = f"error: {e}"
            print(f"  {name:>12s}: ERROR ({e})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return 0


def cmd_select(args):
    """Select diverse subset from embeddings."""
    from src.unified_selector import get_selector

    print(f"Generating {args.n} random items in {args.dim}D...")
    rng = np.random.RandomState(args.seed)
    embeddings = rng.randn(args.n, args.dim)

    selector = get_selector(args.method)
    t0 = time.time()
    indices, metadata = selector.select(embeddings, k=args.k)
    elapsed = (time.time() - t0) * 1000

    # Compute spread
    from scipy.spatial.distance import pdist
    selected = embeddings[indices]
    dists = pdist(selected)
    spread = float(np.min(dists)) if len(dists) > 0 else 0.0
    sum_dist = float(np.sum(dists))

    print(f"\nMethod: {args.method}")
    print(f"Selected {len(indices)} items from {args.n}")
    print(f"Spread (min dist): {spread:.4f}")
    print(f"Sum distance: {sum_dist:.2f}")
    print(f"Time: {elapsed:.1f}ms")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "method": args.method,
                "n": args.n, "k": args.k, "dim": args.dim,
                "indices": indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                "spread": round(spread, 4),
                "sum_dist": round(sum_dist, 2),
                "time_ms": round(elapsed, 1),
            }, f, indent=2)

    return 0


def cmd_benchmark(args):
    """Run selection algorithm benchmark."""
    from src.unified_selector import get_selector
    from scipy.spatial.distance import pdist

    methods = ['farthest_point', 'submodular', 'dpp', 'random']
    rng = np.random.RandomState(args.seed)
    embeddings = rng.randn(args.n, args.dim)

    print(f"Benchmark: n={args.n}, k={args.k}, dim={args.dim}, trials={args.trials}\n")
    print(f"{'Method':<20s} {'Spread':>10s} {'Sum Dist':>10s} {'Time(ms)':>10s}")
    print("-" * 55)

    results = {}
    for method in methods:
        spreads, sum_dists, times = [], [], []
        for trial in range(args.trials):
            sel = get_selector(method)
            t0 = time.time()
            idx, _ = sel.select(embeddings, k=args.k)
            elapsed = (time.time() - t0) * 1000

            selected = embeddings[list(idx)]
            d = pdist(selected)
            spreads.append(float(np.min(d)) if len(d) > 0 else 0.0)
            sum_dists.append(float(np.sum(d)))
            times.append(elapsed)

        results[method] = {
            "spread": round(np.mean(spreads), 3),
            "spread_std": round(np.std(spreads), 3),
            "sum_dist": round(np.mean(sum_dists), 1),
            "time_ms": round(np.mean(times), 1),
        }
        print(f"{method:<20s} {np.mean(spreads):>10.3f} {np.mean(sum_dists):>10.1f} {np.mean(times):>10.1f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    return 0


def cmd_cross_model(args):
    """Run cross-model analysis."""
    from src.cross_model_analysis import cross_model_analysis

    print("Running cross-model analysis...")
    results = cross_model_analysis(
        n_prompts=args.n_prompts,
        n_texts=args.n_texts,
        n_configs=args.n_configs,
        seed=args.seed,
    )

    print(f"\nMeta-τ: {results['meta_tau']}")
    print(f"CI: {results['meta_tau_ci']}")
    print(f"Models: {results['n_models']}")
    print(f"Pairs: {results['n_model_pairs']}")
    print(f"Power: {results['power']['power']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return 0


def cmd_sensitivity(args):
    """Run embedding kernel sensitivity analysis."""
    from src.embedding_sensitivity import run_sensitivity_analysis

    print("Running embedding sensitivity analysis...")
    result = run_sensitivity_analysis([], seed=args.seed)

    print(f"\nEPD ranking stability (cross-kernel τ): "
          f"{result.tau_stability['EPD_cross_kernel']}")
    print(f"Vendi ranking stability: "
          f"{result.tau_stability['Vendi_cross_kernel']}")
    print(f"\nRecommendation: {result.summary['recommendation']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.summary, f, indent=2, default=str)

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='divflow',
        description='DivFlow: Diversity Metrics & Selection for AI Text',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', '-o', type=str, default='')

    sub = parser.add_subparsers(dest='command')

    # metrics
    p_m = sub.add_parser('metrics', help='Compute diversity metrics on texts')
    p_m.add_argument('--texts', nargs='+', help='Input texts')
    p_m.add_argument('--file', type=str,
                     help='File with texts (.jsonl, .json, .csv, .parquet, or plain text)')
    p_m.add_argument('--text-field', type=str, default=None,
                     help='JSON field name containing text (for JSONL/JSON input)')

    # select
    p_s = sub.add_parser('select', help='Select diverse subset')
    p_s.add_argument('--n', type=int, default=200, help='Number of items')
    p_s.add_argument('--k', type=int, default=10, help='Selection budget')
    p_s.add_argument('--dim', type=int, default=50, help='Embedding dimension')
    p_s.add_argument('--method', type=str, default='farthest_point',
                     choices=['farthest_point', 'submodular', 'dpp', 'mmr',
                              'clustering', 'random'])

    # benchmark
    p_b = sub.add_parser('benchmark', help='Benchmark selection algorithms')
    p_b.add_argument('--n', type=int, default=200)
    p_b.add_argument('--k', type=int, default=10)
    p_b.add_argument('--dim', type=int, default=50)
    p_b.add_argument('--trials', type=int, default=10)

    # cross-model
    p_c = sub.add_parser('cross-model', help='Cross-model analysis')
    p_c.add_argument('--n-prompts', type=int, default=10)
    p_c.add_argument('--n-texts', type=int, default=5)
    p_c.add_argument('--n-configs', type=int, default=3)

    # sensitivity
    sub.add_parser('sensitivity', help='Embedding kernel sensitivity analysis')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    cmd_map = {
        'metrics': cmd_metrics,
        'select': cmd_select,
        'benchmark': cmd_benchmark,
        'cross-model': cmd_cross_model,
        'sensitivity': cmd_sensitivity,
    }

    return cmd_map[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
