# DivFlow API Reference

## Unified Selector API

All selection algorithms share the same interface via `DiversitySelector`:

```python
from src.unified_selector import get_selector, DiversitySelector

selector = get_selector(name: str, **kwargs) -> DiversitySelector
```

### `DiversitySelector.select(candidates, k, **kwargs)`

Select k diverse items from candidates.

**Parameters:**
- `candidates` (`np.ndarray`): (n, d) feature matrix
- `k` (`int`): number of items to select

**Returns:**
- `indices` (`List[int]`): selected item indices
- `metadata` (`Dict[str, Any]`): algorithm-specific metadata

### Available Selectors

| Name | Class | Key Parameters |
|---|---|---|
| `'farthest_point'` | `FarthestPointSelector` | None |
| `'submodular'` | `SubmodularSelector` | `method='greedy'` or `'stochastic'` |
| `'dpp'` | `DPPSelector` | `kernel='rbf'`, `gamma=None` |
| `'mmr'` | `MMRSelector` | `lambda_param=0.5` |
| `'clustering'` | `ClusteringSelector` | `method='kmedoids'` |
| `'random'` | `RandomSelector` | None |

## SMT/ILP Exact Optimization (NEW)

```python
from src.smt_diversity import SMTDiversityOptimizer, ILPDiversityOptimizer

# Exact solution via Z3 SMT solver (n ≤ 50)
opt = SMTDiversityOptimizer(timeout_ms=10000)
result = opt.solve_exact(
    distance_matrix,          # (n, n) pairwise distances
    k=5,                      # items to select
    groups=group_labels,      # (n,) group labels (optional)
    min_per_group={0: 2, 1: 2},  # fairness constraints
    objective="sum_pairwise", # or "min_pairwise", "facility_location"
)
# result.selected_indices, result.objective_value, result.status

# Optimality gaps: compare greedy vs exact
gaps = opt.compute_optimality_gaps(n_values=[8,10,15], k_values=[3,5])

# Certified fair retention
certs = opt.certify_fair_retention(D, k, groups, constraint_levels)

# NP-hardness witnesses
witnesses = opt.np_hardness_reduction(n_instances=20)

# Comprehensive benchmark
benchmark = opt.run_benchmark(max_n=15, n_trials=5)

# ILP fallback for larger instances
selected, obj = ILPDiversityOptimizer.solve_ilp(D, k=10)
```

## Bias-Corrected Entropy (NEW)

```python
from src.entropy_correction import (
    entropy_miller_madow,      # Miller-Madow correction
    entropy_nsb,               # NSB Bayesian estimator
    entropy_jackknife,         # Jackknife correction
    bootstrap_entropy_bca,     # BCa bootstrap CI
    kl_laplace,                # Laplace-smoothed KL
    kl_jelinek_mercer,         # Jelinek-Mercer smoothed KL
    kl_dirichlet,              # Dirichlet prior smoothed KL
    smoothed_kl_analysis,      # Full KL diagnostic
    entropy_rate_corrected,    # Corrected entropy rate
    corrected_info_theory_analysis,  # Full analysis
)

# Miller-Madow bias correction
h_mle, h_mm, bias = entropy_miller_madow(texts, n=2)

# NSB estimator
h_nsb, h_std = entropy_nsb(texts, n=2)

# BCa bootstrap CI (fixes CI/point-estimate inconsistency)
result = bootstrap_entropy_bca(texts, n=2, n_bootstrap=2000)
# result.miller_madow, result.ci_lower, result.ci_upper, result.ci_method

# KL with proper smoothing (fixes zero-mass tail artifact)
kl = kl_laplace(texts_a, texts_b, n=2)
analysis = smoothed_kl_analysis(texts_a, texts_b, n=2)
# analysis.kl_raw, analysis.kl_laplace, analysis.smoothing_impact

# Full corrected analysis
results = corrected_info_theory_analysis(high_texts, low_texts)
```

## Cross-Model Analysis (NEW)

```python
from src.cross_model_analysis import CrossModelAnalyzer, cross_model_analysis

# Full analysis across 10 model families (45 pairs)
analyzer = CrossModelAnalyzer(seed=42)
result = analyzer.run_full_analysis(
    n_prompts=20, n_texts_per_prompt=10, n_configs=5
)
# result.meta_tau, result.meta_tau_ci, result.power_analysis
# result.bayesian_posterior, result.pairwise_results

# Quick JSON output
results = cross_model_analysis(n_prompts=20, n_texts=10)
```

## Metric Lattice Structure (NEW)

```python
from src.metric_lattice import MetricLattice, lattice_analysis

lattice = MetricLattice(metric_values_dict)
classes = lattice.equivalence_classes(delta=0.3)
hasse = lattice.hasse_diagram(delta=0.3)
filtration = lattice.delta_filtration(n_thresholds=50)
merges = lattice.merge_sequence()
is_lat = lattice.is_lattice(delta=0.3)
betti = lattice.betti_numbers(delta=0.3)
autos = lattice.compute_automorphisms(delta=0.3)
full = lattice.full_analysis()

# Quick JSON output
results = lattice_analysis(metric_values_dict)
```

## Failure Mode Taxonomy (NEW)

```python
from src.failure_taxonomy import (
    build_failure_taxonomy, FailureModeDetector,
    failure_taxonomy_analysis,
)

# Build taxonomy (13 modes, 6 categories)
taxonomy = build_failure_taxonomy()
for mode in taxonomy.modes:
    print(f"{mode.id}: {mode.name} [{mode.severity.value}]")
    print(f"  Cause: {mode.structural_cause}")
    print(f"  Affects: {mode.affected_metrics}")

# Detect failure modes on new texts
detector = FailureModeDetector(taxonomy)
detected = detector.detect(my_texts)
for mode, confidence in detected:
    print(f"  {mode.id}: {mode.name} (confidence {confidence:.2f})")
```

## Fair Diversity Selection

```python
from src.fair_diversity import FairDiverseSelector

selector = FairDiverseSelector()
indices = selector.select(
    items=embeddings,        # (n, d) features
    groups=group_labels,     # (n,) integer group labels
    k=20,
    min_per_group={0: 2, 1: 2, 2: 2},
    strategy='group_fair'    # or 'proportional', 'rooney'
)
```

## Information-Theoretic Baselines

```python
from src.metrics.information_theoretic import (
    shannon_entropy, kl_divergence, symmetric_kl,
    mutual_information, entropy_rate, bootstrap_entropy_ci,
)

h = shannon_entropy(texts, n=2)           # bits
kl = kl_divergence(texts_a, texts_b, n=2) # bits
mi = mutual_information(texts_a, texts_b)  # bits
rate, cond = entropy_rate(texts, max_order=5)
ci = bootstrap_entropy_ci(texts, n=2, n_bootstrap=2000)
```

## Distributional Analysis

```python
from src.distributional_analysis import (
    MetricAlgebra, SubmodularityVerifier, PermutationTest,
    tightness_construction,
    normalized_mutual_information,  # NMI for metric comparison
    variation_of_information,       # VI metric distance
    info_theoretic_metric_comparison,  # Full IT comparison
    berry_esseen_kendall_tau,       # Berry-Esseen convergence rate
)

algebra = MetricAlgebra(metric_values_dict, delta=0.3)
classes = algebra.equivalence_classes()
summary = algebra.summary()

# Information-theoretic metric comparison (NMI + VI)
it_results = info_theoretic_metric_comparison(metric_values_dict)
# it_results['summary']['mean_nmi'], ['redundant_by_nmi']

# Berry-Esseen convergence rate for Kendall τ CLT
be = berry_esseen_kendall_tau(n=13)
# be['berry_esseen_bound'], be['n_for_gaussian_reliable']

# Submodularity verification (exhaustive n≤8, random sampling n>8)
verifier = SubmodularityVerifier(f, ground_set_size=n)
result = verifier.verify_exact()
# result['is_submodular'], result['violation_rate_ci_95'] (for n>8)

pt = PermutationTest(n_permutations=10000)
result = pt.test(x, y)
```

## Adversarial Analysis

```python
from src.adversarial_analysis import (
    AdversarialDivergenceSearch, FairSelectionWorstCase,
    construct_np_hardness_witness,
)

searcher = AdversarialDivergenceSearch(seed=42)
results = searcher.search(n_trials=50)

analyzer = FairSelectionWorstCase(seed=42)
frontier = analyzer.pareto_frontier(n=200, d=50, k=20)
```
