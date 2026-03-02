#!/usr/bin/env python3
"""
Scalable certified diversity benchmarks.

Compares ScalableCertifiedOptimizer against DPP, MMR, FarthestPoint, and Random
at production scale (n=50 to 5000) — far beyond SMT's n≤12 limit.

Experiments:
  1. Quality comparison: certified value vs baselines at each (n, k)
  2. Certified gap analysis: how tight are LP bounds?
  3. Scaling: runtime and gap vs problem size
  4. Distribution robustness: performance across 4 distributions
  5. Small-n validation: verify agreement with exact SMT on n≤12
"""

import json
import os
import sys
import time

import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.scalable_certifier import (
    ScalableCertifiedOptimizer,
    dpp_select,
    mmr_select,
    farthest_point_select,
    random_select,
    generate_distance_matrix,
    _evaluate_objective,
)


def run_quality_comparison():
    """Exp 1: Quality comparison at scale.

    For each (n, k, objective, distribution), compare our certified selection
    against DPP, MMR, FarthestPoint, Random.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Quality Comparison at Scale")
    print("="*70)

    configs = [
        (50, 5),   (50, 10),
        (100, 5),  (100, 10), (100, 20),
        (200, 10), (200, 20),
        (500, 10), (500, 20),
        (1000, 10), (1000, 50),
        (2000, 20),
    ]
    objectives = ["sum_pairwise", "min_pairwise"]
    distributions = ["uniform", "clustered", "adversarial"]

    results = []
    opt = ScalableCertifiedOptimizer(timeout_seconds=60)

    for n, k in configs:
        for dist in distributions:
            D = generate_distance_matrix(n, dist, seed=42)
            for obj in objectives:
                print(f"\n  n={n}, k={k}, dist={dist}, obj={obj}")

                # Our method
                t0 = time.time()
                cert = opt.certified_select(D, k, objective=obj)
                our_time = time.time() - t0

                # Sanity check
                max_possible = float(k * (k-1) / 2 * np.max(D)) if obj == "sum_pairwise" else float(np.max(D))
                assert cert.objective_value <= max_possible * 1.01, \
                    f"Value {cert.objective_value} > max possible {max_possible}"

                print(f"    Ours: val={cert.objective_value:.4f}, "
                      f"gap={cert.certified_gap_pct:.2f}%, "
                      f"time={our_time:.3f}s, method={cert.method}")

                # Baselines (use copy to avoid LP solver side effects on D)
                D_b = D.copy()
                baselines = {}

                if n <= 300:
                    t0 = time.time()
                    dpp_idx, dpp_sum, dpp_min = dpp_select(D_b, k)
                    dpp_time = time.time() - t0
                    dpp_val = dpp_sum if obj == "sum_pairwise" else dpp_min
                    if dpp_val > max_possible * 1.01:
                        print(f"    DPP:  val={dpp_val:.4f} (WARNING: exceeds max)")
                        dpp_val = min(dpp_val, max_possible)
                    baselines["dpp"] = (dpp_val, dpp_time)
                    print(f"    DPP:  val={dpp_val:.4f}, time={dpp_time:.3f}s")

                t0 = time.time()
                mmr_idx, mmr_sum, mmr_min = mmr_select(D_b, k)
                mmr_time = time.time() - t0
                mmr_val = mmr_sum if obj == "sum_pairwise" else mmr_min
                baselines["mmr"] = (mmr_val, mmr_time)
                print(f"    MMR:  val={mmr_val:.4f}, time={mmr_time:.3f}s")

                t0 = time.time()
                fp_idx, fp_sum, fp_min = farthest_point_select(D_b, k)
                fp_time = time.time() - t0
                fp_val = fp_sum if obj == "sum_pairwise" else fp_min
                baselines["farthest_point"] = (fp_val, fp_time)
                print(f"    FP:   val={fp_val:.4f}, time={fp_time:.3f}s")

                t0 = time.time()
                rnd_idx, rnd_sum, rnd_min = random_select(D_b, k)
                rnd_time = time.time() - t0
                rnd_val = rnd_sum if obj == "sum_pairwise" else rnd_min
                baselines["random"] = (rnd_val, rnd_time)
                print(f"    Rand: val={rnd_val:.4f}, time={rnd_time:.3f}s")

                # Compute improvement over baselines
                improvements = {}
                for name, (bval, btime) in baselines.items():
                    if bval > 1e-12:
                        imp = (cert.objective_value - bval) / bval * 100
                    else:
                        imp = float('inf') if cert.objective_value > 0 else 0.0
                    improvements[name] = imp

                results.append({
                    "n": n, "k": k, "distribution": dist, "objective": obj,
                    "our_value": cert.objective_value,
                    "our_upper_bound": cert.upper_bound,
                    "our_gap_pct": cert.certified_gap_pct,
                    "our_method": cert.method,
                    "our_time": our_time,
                    "our_swaps": cert.local_search_swaps,
                    "baseline_values": {name: bv for name, (bv, _) in baselines.items()},
                    "baseline_times": {name: bt for name, (_, bt) in baselines.items()},
                    "improvements_pct": improvements,
                })

    return results


def run_gap_analysis():
    """Exp 2: How tight are the certified bounds?

    Measures LP relaxation gap across problem sizes.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Certified Gap Analysis")
    print("="*70)

    configs = [
        (20, 5), (50, 10), (100, 10), (100, 20),
        (200, 10), (200, 20), (200, 50),
        (500, 10), (500, 20),
    ]
    objectives = ["sum_pairwise", "min_pairwise"]
    distributions = ["uniform", "clustered", "adversarial"]

    results = []
    opt = ScalableCertifiedOptimizer(timeout_seconds=120)

    for n, k in configs:
        for dist in distributions:
            D = generate_distance_matrix(n, dist, seed=42)
            for obj in objectives:
                cert = opt.certified_select(D, k, objective=obj)
                print(f"  n={n:4d}, k={k:2d}, dist={dist:12s}, obj={obj:14s}: "
                      f"gap={cert.certified_gap_pct:6.2f}%, "
                      f"val={cert.objective_value:.3f}, UB={cert.upper_bound:.3f}")

                results.append({
                    "n": n, "k": k, "distribution": dist, "objective": obj,
                    "value": cert.objective_value,
                    "upper_bound": cert.upper_bound,
                    "gap_pct": cert.certified_gap_pct,
                    "method": cert.method,
                    "time": cert.solve_time_seconds,
                })

    return results


def run_scaling_analysis():
    """Exp 3: Runtime and gap scaling with n."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Scaling Analysis")
    print("="*70)

    sizes = [20, 50, 100, 200, 500, 1000, 2000]
    k_ratio = 0.1  # k = 10% of n

    results = []
    opt = ScalableCertifiedOptimizer(timeout_seconds=60)

    for n in sizes:
        k = max(3, int(n * k_ratio))
        D = generate_distance_matrix(n, "uniform", seed=42)

        for obj in ["sum_pairwise", "min_pairwise"]:
            t0 = time.time()
            cert = opt.certified_select(D, k, objective=obj)
            elapsed = time.time() - t0
            print(f"  n={n:5d}, k={k:4d}, obj={obj:14s}: "
                  f"gap={cert.certified_gap_pct:6.2f}%, "
                  f"time={elapsed:.3f}s, method={cert.method}")
            results.append({
                "n": n, "k": k, "objective": obj,
                "gap_pct": cert.certified_gap_pct,
                "time": elapsed,
                "method": cert.method,
                "value": cert.objective_value,
                "upper_bound": cert.upper_bound,
            })

    return results


def run_small_n_validation():
    """Exp 4: Validate against exact SMT on small instances (n ≤ 12)."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Small-n Validation vs Exact SMT")
    print("="*70)

    configs = [(8, 3), (10, 3), (10, 4), (12, 4)]
    objectives = ["sum_pairwise", "min_pairwise"]
    distributions = ["uniform", "clustered", "adversarial"]

    results = []
    opt = ScalableCertifiedOptimizer(timeout_seconds=30)

    try:
        from src.smt_diversity import SMTDiversityOptimizer
        smt_opt = SMTDiversityOptimizer(timeout_ms=30000)
        smt_available = True
    except ImportError:
        smt_available = False
        print("  Z3 not available, skipping SMT validation")

    for n, k in configs:
        for dist in distributions:
            D = generate_distance_matrix(n, dist, d=8, seed=42)
            for obj in objectives:
                # Our method
                cert = opt.certified_select(D, k, objective=obj)

                if smt_available:
                    # Exact SMT
                    smt_result = smt_opt.solve_exact(D, k, objective=obj)
                    if smt_result.status == "optimal":
                        exact_val = smt_result.objective_value
                        our_gap_to_exact = (
                            (exact_val - cert.objective_value) / exact_val * 100
                            if exact_val > 1e-12 else 0.0
                        )
                        print(f"  n={n}, k={k}, dist={dist:12s}, obj={obj:14s}: "
                              f"ours={cert.objective_value:.4f}, "
                              f"exact={exact_val:.4f}, "
                              f"gap_to_exact={our_gap_to_exact:.2f}%")
                        results.append({
                            "n": n, "k": k, "distribution": dist, "objective": obj,
                            "our_value": cert.objective_value,
                            "exact_value": exact_val,
                            "gap_to_exact_pct": our_gap_to_exact,
                            "certified_gap_pct": cert.certified_gap_pct,
                        })
                    else:
                        print(f"  n={n}, k={k}, dist={dist:12s}, obj={obj:14s}: "
                              f"SMT {smt_result.status}")
                else:
                    print(f"  n={n}, k={k}, dist={dist:12s}, obj={obj:14s}: "
                          f"ours={cert.objective_value:.4f}, "
                          f"gap={cert.certified_gap_pct:.2f}%")
                    results.append({
                        "n": n, "k": k, "distribution": dist, "objective": obj,
                        "our_value": cert.objective_value,
                        "certified_gap_pct": cert.certified_gap_pct,
                    })

    return results


def run_distribution_robustness():
    """Exp 5: Performance across all distributions."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Distribution Robustness")
    print("="*70)

    distributions = ["uniform", "clustered", "hierarchical", "adversarial"]
    configs = [(100, 10), (200, 20), (500, 20)]

    results = []
    opt = ScalableCertifiedOptimizer(timeout_seconds=120)

    for n, k in configs:
        for dist in distributions:
            D = generate_distance_matrix(n, dist, seed=42)
            for obj in ["sum_pairwise", "min_pairwise"]:
                cert = opt.certified_select(D, k, objective=obj)

                # Baselines
                fp_idx, fp_sum, fp_min = farthest_point_select(D, k)
                fp_val = fp_sum if obj == "sum_pairwise" else fp_min

                rnd_idx, rnd_sum, rnd_min = random_select(D, k)
                rnd_val = rnd_sum if obj == "sum_pairwise" else rnd_min

                imp_fp = ((cert.objective_value - fp_val) / fp_val * 100
                          if fp_val > 1e-12 else 0.0)
                imp_rnd = ((cert.objective_value - rnd_val) / rnd_val * 100
                           if rnd_val > 1e-12 else 0.0)

                print(f"  n={n}, k={k}, dist={dist:13s}, obj={obj:14s}: "
                      f"ours={cert.objective_value:.3f} "
                      f"(+{imp_fp:.1f}% vs FP, +{imp_rnd:.1f}% vs Rand), "
                      f"gap={cert.certified_gap_pct:.2f}%")

                results.append({
                    "n": n, "k": k, "distribution": dist, "objective": obj,
                    "our_value": cert.objective_value,
                    "gap_pct": cert.certified_gap_pct,
                    "farthest_point_value": fp_val,
                    "random_value": rnd_val,
                    "improvement_vs_fp_pct": imp_fp,
                    "improvement_vs_random_pct": imp_rnd,
                })

    return results


def compute_summary(all_results):
    """Compute aggregate statistics."""
    quality = all_results.get("quality_comparison", [])
    gaps = all_results.get("gap_analysis", [])

    summary = {}

    # Aggregate quality stats
    if quality:
        for baseline in ["dpp", "mmr", "farthest_point", "random"]:
            imps = [r["improvements_pct"].get(baseline, None)
                    for r in quality if baseline in r.get("improvements_pct", {})]
            imps = [x for x in imps if x is not None and abs(x) < 1e6]
            if imps:
                summary[f"vs_{baseline}"] = {
                    "mean_improvement_pct": float(np.mean(imps)),
                    "median_improvement_pct": float(np.median(imps)),
                    "max_improvement_pct": float(np.max(imps)),
                    "min_improvement_pct": float(np.min(imps)),
                    "win_rate_pct": float(np.mean([x > 0 for x in imps]) * 100),
                    "n_comparisons": len(imps),
                }

    # Aggregate gap stats
    if gaps:
        gap_vals = [r["gap_pct"] for r in gaps]
        summary["certified_gaps"] = {
            "mean_gap_pct": float(np.mean(gap_vals)),
            "median_gap_pct": float(np.median(gap_vals)),
            "max_gap_pct": float(np.max(gap_vals)),
            "pct_under_5": float(np.mean([g < 5 for g in gap_vals]) * 100),
            "pct_under_10": float(np.mean([g < 10 for g in gap_vals]) * 100),
        }

    return summary


def main():
    print("DivFlow Scalable Certified Diversity Benchmarks")
    print("=" * 70)

    all_results = {}

    # Run all experiments
    all_results["quality_comparison"] = run_quality_comparison()
    all_results["gap_analysis"] = run_gap_analysis()
    all_results["scaling"] = run_scaling_analysis()
    all_results["small_n_validation"] = run_small_n_validation()
    all_results["distribution_robustness"] = run_distribution_robustness()

    # Summary
    all_results["summary"] = compute_summary(all_results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for key, val in all_results["summary"].items():
        print(f"\n  {key}:")
        if isinstance(val, dict):
            for k2, v2 in val.items():
                print(f"    {k2}: {v2}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "scalable_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
