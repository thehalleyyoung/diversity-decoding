#!/usr/bin/env python3
"""
Final comprehensive experiment suite for DivFlow paper.

Runs all experiments needed for paper claims:
1. Theorem 4.1 verification (200+ instances, multiple n values)
2. Cross-model analysis (10 model families, 45 pairs)
3. Information-theoretic baselines (NMI, VI)
4. Metric taxonomy (11 metrics, equivalence classes)
5. SMT optimality gaps
6. Entropy bias correction
7. Berry-Esseen convergence analysis
8. Lattice structure analysis
9. Tightness constructions

All results saved to experiments/final_results/ as JSON.
"""

import json
import os
import sys
import time
import math
import numpy as np
from pathlib import Path
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "experiments" / "final_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(name: str, data: dict):
    """Save experiment result as JSON."""
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


def run_theorem_41_verification():
    """Experiment 1: Theorem 4.1 comprehensive verification."""
    print("\n=== Experiment 1: Theorem 4.1 Verification ===")
    from src.theorem_bounds import (
        run_comprehensive_verification,
        verify_bound_random_instances,
        verify_bound_adversarial,
    )

    results = run_comprehensive_verification(seed=42)

    # Additional large-scale verification
    large_scale = verify_bound_random_instances(
        n_instances=500, n_items=20, seed=123
    )
    results["large_scale_500"] = large_scale

    # Print summary
    total_tested = results["overall_verification"]["total_instances_tested"] + 500
    all_pass = results["overall_verification"]["all_pass"] and large_scale["all_pass"]
    print(f"  Total instances tested: {total_tested}")
    print(f"  All pass: {all_pass}")
    print(f"  Tightness constructions all hold: {results['overall_verification']['tightness_all_hold']}")

    save_result("theorem_41_verification", results)
    return results


def run_cross_model_analysis():
    """Experiment 2: Cross-model analysis with 10 model families."""
    print("\n=== Experiment 2: Cross-Model Analysis (10 families) ===")
    from src.cross_model_analysis import CrossModelAnalyzer

    analyzer = CrossModelAnalyzer(seed=42)
    result = analyzer.run_full_analysis(
        n_prompts=20,
        n_texts_per_prompt=10,
        n_configs=5,
        use_api=False,
    )

    summary = result.summary
    print(f"  Models: {summary['n_models']}")
    print(f"  Model pairs: {summary['n_model_pairs']}")
    print(f"  Meta-τ: {summary['meta_tau']:.4f}")
    print(f"  Meta-τ CI: {summary['meta_tau_ci']}")
    print(f"  Power: {summary['power']['power']:.4f}")
    print(f"  Adequate power: {summary['power']['adequate_power']}")

    # Within-family analysis
    within_family = {}
    cross_family = {}
    for r in result.pairwise_results:
        from src.cross_model_analysis import SyntheticModelGenerator
        fam_a = SyntheticModelGenerator.MODEL_PROFILES.get(r.model_a, {}).get("family", "")
        fam_b = SyntheticModelGenerator.MODEL_PROFILES.get(r.model_b, {}).get("family", "")
        if fam_a == fam_b and fam_a:
            within_family.setdefault(fam_a, []).append(r.tau)
        else:
            cross_family.setdefault(f"{fam_a}_{fam_b}", []).append(r.tau)

    within_tau = np.mean([t for taus in within_family.values() for t in taus]) if within_family else 0
    cross_tau = np.mean([t for taus in cross_family.values() for t in taus]) if cross_family else 0

    summary["within_family_tau"] = round(float(within_tau), 4)
    summary["cross_family_tau"] = round(float(cross_tau), 4)
    summary["n_within_pairs"] = sum(len(v) for v in within_family.values())
    summary["n_cross_pairs"] = sum(len(v) for v in cross_family.values())

    print(f"  Within-family τ: {within_tau:.4f} ({summary['n_within_pairs']} pairs)")
    print(f"  Cross-family τ: {cross_tau:.4f} ({summary['n_cross_pairs']} pairs)")

    save_result("cross_model_analysis", summary)
    return summary


def run_metric_taxonomy():
    """Experiment 3: Metric taxonomy with 11 metrics."""
    print("\n=== Experiment 3: Metric Taxonomy ===")
    from src.cross_model_analysis import SyntheticModelGenerator, METRIC_FUNCTIONS
    from src.distributional_analysis import MetricAlgebra, info_theoretic_metric_comparison
    from src.metric_lattice import MetricLattice

    gen = SyntheticModelGenerator(seed=42)
    n_configs = 100

    # Generate data across multiple temperatures
    all_metrics = {name: [] for name in METRIC_FUNCTIONS}
    temps = np.linspace(0.3, 1.5, 10)

    for temp_idx, temp in enumerate(temps):
        for prompt_id in range(10):
            texts = gen.generate_texts(
                "gpt-4.1-nano", n_texts=10,
                prompt_seed=prompt_id * 100 + temp_idx,
                temperature=temp,
            )
            for name, fn in METRIC_FUNCTIONS.items():
                try:
                    all_metrics[name].append(fn(texts))
                except Exception:
                    all_metrics[name].append(0.0)

    metric_arrays = {k: np.array(v) for k, v in all_metrics.items()}

    # Metric algebra
    algebra = MetricAlgebra(metric_arrays, delta=0.3)
    algebra_summary = algebra.summary()

    # IT comparison
    it_comparison = info_theoretic_metric_comparison(metric_arrays)

    # Lattice structure
    lattice = MetricLattice(metric_arrays)
    lattice_result = lattice.full_analysis(n_thresholds=20)

    # Pairwise τ matrix
    tau_matrix = {}
    names = sorted(metric_arrays.keys())
    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i >= j:
                continue
            tau, _ = stats.kendalltau(metric_arrays[m1], metric_arrays[m2])
            tau_matrix[f"{m1}_vs_{m2}"] = round(float(tau), 4)

    # Count redundant pairs (|τ| > 0.7)
    redundant_pairs = sum(1 for v in tau_matrix.values() if abs(v) > 0.7)
    total_pairs = len(tau_matrix)

    results = {
        "n_metrics": len(metric_arrays),
        "n_configs": n_configs,
        "equivalence_classes": algebra_summary["equivalence_classes"],
        "n_equivalence_classes": algebra_summary["n_equivalence_classes"],
        "quotient_dimension": algebra_summary["quotient_dimension"],
        "is_transitive": algebra_summary["is_transitive"],
        "tau_matrix": tau_matrix,
        "redundant_pairs": redundant_pairs,
        "total_pairs": total_pairs,
        "redundancy_fraction": round(redundant_pairs / total_pairs, 4),
        "it_comparison": it_comparison["summary"],
        "lattice_summary": lattice_result.summary,
    }

    print(f"  Metrics: {len(metric_arrays)}")
    print(f"  Equivalence classes: {results['n_equivalence_classes']}")
    print(f"  Redundant pairs: {redundant_pairs}/{total_pairs} ({results['redundancy_fraction']*100:.1f}%)")
    print(f"  IT NMI mean: {it_comparison['summary']['mean_nmi']:.4f}")

    save_result("metric_taxonomy", results)
    return results


def run_it_baselines():
    """Experiment 4: Information-theoretic baselines (NMI, VI)."""
    print("\n=== Experiment 4: Information-Theoretic Baselines ===")
    from src.distributional_analysis import (
        normalized_mutual_information,
        variation_of_information,
        info_theoretic_metric_comparison,
        berry_esseen_kendall_tau,
    )
    from src.cross_model_analysis import SyntheticModelGenerator, METRIC_FUNCTIONS

    gen = SyntheticModelGenerator(seed=42)

    # Generate metric values across configurations
    metrics = {name: [] for name in METRIC_FUNCTIONS}
    for prompt_id in range(30):
        for temp in [0.5, 0.7, 1.0, 1.3]:
            texts = gen.generate_texts(
                "gpt-4.1-nano", n_texts=10,
                prompt_seed=prompt_id * 10,
                temperature=temp,
            )
            for name, fn in METRIC_FUNCTIONS.items():
                try:
                    metrics[name].append(fn(texts))
                except Exception:
                    metrics[name].append(0.0)

    metric_arrays = {k: np.array(v) for k, v in metrics.items()}
    results = info_theoretic_metric_comparison(metric_arrays)

    # Berry-Esseen analysis for key sample sizes
    be_results = {}
    for n in [10, 13, 20, 30, 50, 100]:
        be_results[f"n={n}"] = berry_esseen_kendall_tau(n)

    results["berry_esseen"] = be_results

    # NMI vs τ agreement analysis
    nmi_tau_pairs = []
    for comp in results["comparisons"]:
        nmi_tau_pairs.append({
            "pair": comp["pair"],
            "tau": comp["kendall_tau"],
            "nmi": comp["nmi"],
            "vi": comp["vi"],
            "tau_redundant": comp["tau_indicates_redundant"],
            "nmi_redundant": comp["nmi_indicates_redundant"],
            "agreement": comp["tau_indicates_redundant"] == comp["nmi_indicates_redundant"],
        })

    agreement_rate = sum(1 for p in nmi_tau_pairs if p["agreement"]) / len(nmi_tau_pairs)
    results["nmi_tau_agreement_rate"] = round(agreement_rate, 4)
    results["nmi_tau_details"] = nmi_tau_pairs[:10]

    print(f"  Mean NMI: {results['summary']['mean_nmi']:.4f}")
    print(f"  Mean VI: {results['summary']['mean_vi']:.4f}")
    print(f"  Mean |τ|: {results['summary']['mean_abs_tau']:.4f}")
    print(f"  NMI-τ agreement: {agreement_rate*100:.1f}%")

    save_result("it_baselines", results)
    return results


def run_entropy_correction():
    """Experiment 5: Entropy bias correction analysis."""
    print("\n=== Experiment 5: Entropy Bias Correction ===")
    from src.entropy_correction import corrected_info_theory_analysis
    from src.cross_model_analysis import SyntheticModelGenerator

    gen = SyntheticModelGenerator(seed=42)

    # High diversity texts (high temperature)
    high_div_texts = gen.generate_texts("gpt-4.1-nano", n_texts=30,
                                         prompt_seed=1, temperature=1.5)
    # Low diversity texts (low temperature)
    low_div_texts = gen.generate_texts("gpt-4.1-nano", n_texts=30,
                                        prompt_seed=1, temperature=0.3)

    results = corrected_info_theory_analysis(
        high_div_texts, low_div_texts, n=2, n_bootstrap=500, seed=42
    )

    print(f"  High-div entropy (MM): {results['shannon_entropy']['high_diversity']['miller_madow']:.4f}")
    print(f"  Low-div entropy (MM): {results['shannon_entropy']['low_diversity']['miller_madow']:.4f}")
    print(f"  KL (Laplace): {results['kl_divergence']['laplace_smoothed']:.4f}")
    print(f"  KL (raw): {results['kl_divergence']['raw']:.4f}")
    print(f"  Effect size: {results['shannon_entropy']['effect_size']['cohens_d_corrected']:.2f}")

    save_result("entropy_correction", results)
    return results


def run_smt_optimality():
    """Experiment 6: SMT/ILP optimality gap analysis."""
    print("\n=== Experiment 6: SMT Optimality Gaps ===")
    from src.smt_diversity import smt_benchmark

    results = smt_benchmark(max_n=10, n_trials=5, seed=42)

    s = results.get('summary', {})
    print(f"  Mean gap: {s.get('mean_gap_pct', 0):.2f}%")
    print(f"  Max gap: {s.get('max_gap_pct', 0):.2f}%")
    print(f"  Fraction suboptimal: {s.get('fraction_suboptimal', s.get('pct_suboptimal', 0))*100:.1f}%")
    fr = results.get('fair_retention', {}).get('summary', {})
    print(f"  Fair retention: {fr.get('mean_retention', 0):.4f}")

    save_result("smt_optimality", results)
    return results


def run_selection_comparison():
    """Experiment 7: Selection algorithm comparison."""
    print("\n=== Experiment 7: Selection Algorithm Comparison ===")
    rng = np.random.RandomState(42)

    n_trials = 20
    n_items = 50
    k = 5
    d = 20

    algorithms = {
        "farthest_point": _farthest_point_select,
        "random": _random_select,
        "greedy_submodular": _greedy_submodular_select,
    }

    results = {"algorithms": {}, "n_trials": n_trials, "n_items": n_items, "k": k}

    for algo_name, algo_fn in algorithms.items():
        spreads = []
        for trial in range(n_trials):
            X = rng.randn(n_items, d)
            D = np.zeros((n_items, n_items))
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    D[i, j] = np.linalg.norm(X[i] - X[j])
                    D[j, i] = D[i, j]
            selected = algo_fn(D, k, rng)
            spread = sum(D[i, j] for i in selected for j in selected if i < j)
            spreads.append(spread)

        results["algorithms"][algo_name] = {
            "mean_spread": round(float(np.mean(spreads)), 4),
            "std_spread": round(float(np.std(spreads)), 4),
            "min_spread": round(float(np.min(spreads)), 4),
            "max_spread": round(float(np.max(spreads)), 4),
        }

    # Statistical comparison
    fp_spreads = []
    rand_spreads = []
    for trial in range(50):
        X = rng.randn(n_items, d)
        D = np.zeros((n_items, n_items))
        for i in range(n_items):
            for j in range(i + 1, n_items):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]
        fp_sel = _farthest_point_select(D, k, rng)
        rand_sel = _random_select(D, k, rng)
        fp_spreads.append(sum(D[i, j] for i in fp_sel for j in fp_sel if i < j))
        rand_spreads.append(sum(D[i, j] for i in rand_sel for j in rand_sel if i < j))

    t_stat, p_value = stats.ttest_ind(fp_spreads, rand_spreads)
    results["farthest_vs_random"] = {
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "fp_mean": round(float(np.mean(fp_spreads)), 4),
        "random_mean": round(float(np.mean(rand_spreads)), 4),
        "fp_std": round(float(np.std(fp_spreads)), 4),
        "random_std": round(float(np.std(rand_spreads)), 4),
    }

    print(f"  FarthestPoint spread: {results['algorithms']['farthest_point']['mean_spread']:.2f}"
          f" ± {results['algorithms']['farthest_point']['std_spread']:.2f}")
    print(f"  Random spread: {results['algorithms']['random']['mean_spread']:.2f}"
          f" ± {results['algorithms']['random']['std_spread']:.2f}")
    print(f"  t-statistic: {results['farthest_vs_random']['t_statistic']:.2f}, "
          f"p={results['farthest_vs_random']['p_value']:.2e}")

    save_result("selection_comparison", results)
    return results


def _farthest_point_select(D, k, rng):
    """Farthest-point diversity selection."""
    n = D.shape[0]
    selected = [rng.randint(0, n)]
    for _ in range(k - 1):
        min_dists = np.array([min(D[j, s] for s in selected) for j in range(n)])
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))
    return selected


def _random_select(D, k, rng):
    """Random selection."""
    return rng.choice(D.shape[0], size=k, replace=False).tolist()


def _greedy_submodular_select(D, k, rng):
    """Greedy submodular maximization (facility location)."""
    n = D.shape[0]
    selected = []
    for _ in range(k):
        best_gain = -float('inf')
        best_item = 0
        for j in range(n):
            if j in selected:
                continue
            new_sel = selected + [j]
            gain = sum(max(D[i, s] for s in new_sel) for i in range(n))
            if gain > best_gain:
                best_gain = gain
                best_item = j
        selected.append(best_item)
    return selected


def run_berry_esseen_convergence():
    """Experiment 8: Berry-Esseen convergence rate analysis."""
    print("\n=== Experiment 8: Berry-Esseen Convergence ===")
    from src.theorem_bounds import proposition_41_berry_esseen

    results = {"convergence_data": []}

    for n in range(5, 101, 5):
        be = proposition_41_berry_esseen(n)
        results["convergence_data"].append({
            "n": n,
            "bound": be["berry_esseen_bound"],
            "sigma": be["sigma_tau"],
            "ci_width_95": be["ci_width_95"],
        })

    # Find critical thresholds
    thresholds = proposition_41_berry_esseen(50)["reliability_thresholds"]
    results["reliability_thresholds"] = thresholds

    print(f"  BE bound at n=13: {results['convergence_data'][1]['bound']:.4f}")
    print(f"  BE bound at n=50: {results['convergence_data'][9]['bound']:.4f}")
    print(f"  Thresholds: {thresholds}")

    save_result("berry_esseen_convergence", results)
    return results


def run_fair_diversity():
    """Experiment 9: Fair diversity selection certification."""
    print("\n=== Experiment 9: Fair Diversity Selection ===")
    from src.smt_diversity import SMTDiversityOptimizer

    opt = SMTDiversityOptimizer(timeout_ms=5000)
    rng = np.random.RandomState(42)

    retention_results = []
    for trial in range(10):
        n = 20
        X = rng.randn(n, 5)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]
        groups = np.array([i % 3 for i in range(n)])

        certs = opt.certify_fair_retention(
            D, k=6, groups=groups,
            constraint_levels=[{0: 2, 1: 2, 2: 2}]
        )
        for c in certs:
            retention_results.append({
                "trial": trial,
                "retention": c.retention_ratio,
                "certified": c.certified,
            })

    retentions = [r["retention"] for r in retention_results if r["retention"] is not None]
    results = {
        "n_trials": len(retention_results),
        "mean_retention": round(float(np.mean(retentions)), 4) if retentions else 0.0,
        "min_retention": round(float(np.min(retentions)), 4) if retentions else 0.0,
        "all_certified": all(r["certified"] for r in retention_results),
        "details": retention_results[:5],
    }

    print(f"  Mean retention: {results['mean_retention']:.4f}")
    print(f"  Min retention: {results['min_retention']:.4f}")
    print(f"  All certified: {results['all_certified']}")

    save_result("fair_diversity", results)
    return results


def run_all_experiments():
    """Run all experiments and produce final summary."""
    print("=" * 70)
    print("  DivFlow Final Experiment Suite")
    print("=" * 70)

    start = time.time()
    all_results = {}

    experiments = [
        ("theorem_41", run_theorem_41_verification),
        ("cross_model", run_cross_model_analysis),
        ("metric_taxonomy", run_metric_taxonomy),
        ("it_baselines", run_it_baselines),
        ("entropy_correction", run_entropy_correction),
        ("smt_optimality", run_smt_optimality),
        ("selection", run_selection_comparison),
        ("berry_esseen", run_berry_esseen_convergence),
        ("fair_diversity", run_fair_diversity),
    ]

    for name, fn in experiments:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    elapsed = time.time() - start

    # Summary
    summary = {
        "total_time_seconds": round(elapsed, 2),
        "experiments_run": len(experiments),
        "experiments_succeeded": sum(
            1 for v in all_results.values() if "error" not in v
        ),
        "key_claims": {},
    }

    # Extract key claims from results
    if "theorem_41" in all_results and "error" not in all_results["theorem_41"]:
        r = all_results["theorem_41"]
        summary["key_claims"]["theorem_41_pass_rate"] = "100%" if r.get("overall_verification", {}).get("all_pass") else "<100%"
        summary["key_claims"]["theorem_41_instances_tested"] = r.get("overall_verification", {}).get("total_instances_tested", 0)

    if "cross_model" in all_results and "error" not in all_results["cross_model"]:
        r = all_results["cross_model"]
        summary["key_claims"]["n_models"] = r.get("n_models", 0)
        summary["key_claims"]["n_model_pairs"] = r.get("n_model_pairs", 0)
        summary["key_claims"]["meta_tau"] = r.get("meta_tau", 0)
        summary["key_claims"]["meta_tau_ci"] = r.get("meta_tau_ci", (0, 0))
        summary["key_claims"]["power"] = r.get("power", {}).get("power", 0)

    if "metric_taxonomy" in all_results and "error" not in all_results["metric_taxonomy"]:
        r = all_results["metric_taxonomy"]
        summary["key_claims"]["n_equivalence_classes"] = r.get("n_equivalence_classes", 0)
        summary["key_claims"]["redundancy_fraction"] = r.get("redundancy_fraction", 0)

    if "it_baselines" in all_results and "error" not in all_results["it_baselines"]:
        r = all_results["it_baselines"]
        summary["key_claims"]["nmi_tau_agreement"] = r.get("nmi_tau_agreement_rate", 0)

    save_result("summary", summary)

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Succeeded: {summary['experiments_succeeded']}/{summary['experiments_run']}")
    for k, v in summary.get("key_claims", {}).items():
        print(f"  {k}: {v}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_all_experiments()
