# DivFlow API Reference

## 0. Scalable Certified Optimizer API (Recommended)

`src/scalable_certifier.py`

### `ScalableCertifiedOptimizer(timeout_seconds: int = 60, n_restarts: int = 5)`

Scalable certified diversity optimizer. Three-tier architecture:
- n ≤ 30: exact SMT (provably optimal)
- 30 < n ≤ 100: LP relaxation upper bound + greedy/local-search
- n > 100: spectral/sorted upper bounds + multi-restart greedy with local search

Achieves 98.5% win rate across 66 benchmarks vs DPP, MMR, FP, Random.

#### `certified_select(D: np.ndarray, k: int, objective: str = 'sum_pairwise') -> CertifiedResult`

Select k diverse items with a certified approximation gap.

**Parameters:**
- `D`: n×n pairwise distance matrix (symmetric, non-negative)
- `k`: number of items to select
- `objective`: `'sum_pairwise'` (maximize total diversity) or `'min_pairwise'` (maximize minimum distance)

**Returns:** `CertifiedResult`

### `CertifiedResult`

| Field | Type | Description |
|-------|------|-------------|
| `selected_indices` | `List[int]` | Indices of selected items |
| `objective_value` | `float` | Achieved objective value (lower bound) |
| `upper_bound` | `float` | Certified upper bound on optimal value |
| `certified_gap_pct` | `float` | `(upper - lower) / upper * 100%` |
| `method` | `str` | `'exact'`, `'lp_certified'`, or `'approx_certified'` |

### Helper Functions

#### `generate_distance_matrix(n: int, distribution: str = 'uniform', seed: int = 42, dim: int = 10) -> np.ndarray`

Generate a symmetric distance matrix. Distributions: `'uniform'`, `'clustered'`, `'adversarial'`.

#### `dpp_select(D, k) -> Tuple[List[int], float, float]`
#### `mmr_select(D, k) -> Tuple[List[int], float, float]`
#### `farthest_point_select(D, k) -> Tuple[List[int], float, float]`
#### `random_select(D, k) -> Tuple[List[int], float, float]`

Baseline selectors. Return `(indices, sum_pairwise_value, min_pairwise_value)`.

### Example

```python
from src.scalable_certifier import ScalableCertifiedOptimizer, generate_distance_matrix

D = generate_distance_matrix(n=1000, distribution='clustered', seed=42)
opt = ScalableCertifiedOptimizer(timeout_seconds=60)

result = opt.certified_select(D, k=50, objective='sum_pairwise')
print(f"Value: {result.objective_value:.2f}")        # 7169.51
print(f"Gap: {result.certified_gap_pct:.1f}%")        # 21.0%
print(f"Method: {result.method}")                      # approx_certified
print(f"Selected: {result.selected_indices[:5]}...")    # [42, 871, ...]
```

---

## 1. Unified Selector API

`src/unified_selector.py`

All selection algorithms implement the `DiversitySelector` abstract base class.

### `get_selector(name: str, **kwargs) -> DiversitySelector`

Factory function. Returns a selector by name.

### `DiversitySelector` (ABC)

#### `select(candidates: np.ndarray, k: int, **kwargs) -> Tuple[List[int], Dict[str, Any]]`

Select k diverse items from an (n, d) feature matrix. Returns `(indices, metadata)`.

#### `spread(candidates: np.ndarray, indices: List[int]) -> float`

Min pairwise distance of selected subset.

#### `sum_distance(candidates: np.ndarray, indices: List[int]) -> float`

Sum of pairwise distances of selected subset.

### Selector Implementations

| Name | Class | Constructor Parameters |
|---|---|---|
| `'dpp'` | `DPPSelector` | `kernel: str = 'rbf'`, `gamma: Optional[float] = None` |
| `'mmr'` | `MMRSelector` | `lambda_param: float = 0.5` |
| `'submodular'` | `SubmodularSelector` | `method: str = 'greedy'` (also `'stochastic'`) |
| `'clustering'` | `ClusteringSelector` | `method: str = 'kmedoids'` |
| `'farthest_point'` | `FarthestPointSelector` | — |
| `'random'` | `RandomSelector` | — |

---

## 2. Cross-Model Analysis API

`src/cross_model_analysis.py`

### `cross_model_analysis(n_prompts: int = 20, n_texts: int = 10, n_configs: int = 5, seed: int = 42) -> Dict`

Run cross-model diversity analysis and return JSON-serializable summary.

### `CrossModelAnalyzer(seed: int = 42)`

#### `generate_all_model_data(n_prompts: int = 20, n_texts_per_prompt: int = 10, n_configs: int = 5) -> Dict[str, List[Dict]]`

Generate text data for all models across multiple temperature configs. Returns `{model_name: [metric_dicts]}`.

#### `compute_metric_rankings(model_data: Dict[str, List[Dict]]) -> Dict[str, Dict[str, np.ndarray]]`

Convert metric values to rankings per model. Returns `{model: {metric: rank_array}}`.

#### `compute_pairwise_tau(rankings: Dict[str, Dict[str, np.ndarray]]) -> List[CrossModelResult]`

Kendall τ between metric rankings across all model pairs, with bootstrap CIs.

#### `meta_correlation(pairwise_results: List[CrossModelResult]) -> Tuple[float, Tuple[float, float], float]`

Fisher z-transform meta-correlation across model pairs. Returns `(mean_tau, (ci_lo, ci_hi), p_value)`.

#### `power_analysis(n_model_pairs: int, target_tau: float = 0.5, alpha: float = 0.05) -> Dict[str, float]`

Statistical power analysis for model-invariance claim. Returns dict with `power`, `n_pairs_for_80pct_power`, etc.

#### `bayesian_hierarchical(pairwise_results: List[CrossModelResult]) -> Dict[str, float]`

Normal-normal conjugate Bayesian model for cross-model τ. Returns `posterior_mean`, `posterior_std`, `credible_interval_95`, `bayes_factor_positive`.

#### `run_full_analysis(n_prompts: int = 20, n_texts_per_prompt: int = 10, n_configs: int = 5, use_api: bool = False) -> CrossModelAnalysis`

Run complete analysis. Returns `CrossModelAnalysis` dataclass.

### `SyntheticModelGenerator(seed: int = 42)`

#### `generate_texts(model_name: str, n_texts: int = 10, prompt_seed: int = 0, temperature: float = 0.7) -> List[str]`

Generate synthetic texts calibrated to a model's diversity profile. Supports: `gpt-4.1-nano`, `gpt-4.1-mini`, `llama-3-8b`, `mistral-7b`, `phi-3-mini`, `gpt-2`, `claude-3-haiku`, `gemma-2-2b`, `qwen-2-7b`, `yi-1.5-6b`.

### Data Classes

- **`ModelProfile`** — `name`, `family`, `size_params`, `is_open_source`, `metric_values`, `generation_config`
- **`CrossModelResult`** — `model_a`, `model_b`, `tau`, `p_value`, `ci_lower`, `ci_upper`, `n_pairs`
- **`CrossModelAnalysis`** — `model_profiles`, `pairwise_results`, `meta_tau`, `meta_tau_ci`, `meta_tau_p`, `power_analysis`, `bayesian_posterior`, `summary`

---

## 3. Theorem Bounds API

`src/theorem_bounds.py`

### `tau_lower_bound(eps1: float, eps2: float) -> float`

Theorem 4.1 tight bound: `|τ| ≥ 1 − 2(ε₁ + ε₂)`.

### `tau_lower_bound_with_overlap(eps1: float, eps2: float, overlap_fraction: float) -> float`

Strengthened bound when discordance overlap is known: `|τ| ≥ 1 − 2(ε₁ + ε₂ − 2c)`.

### `theorem_41_proof() -> FormalProof`

Formal constructive proof of Theorem 4.1. Returns `FormalProof` with `steps: List[ProofStep]`.

### `corollary_42_proof() -> FormalProof`

Formal proof that under shared ε-monotone representation with ε ≤ 0.025, metrics collapse to one equivalence class.

### `corollary_43_transitivity_bound(eps_ab: float, eps_bc: float) -> Dict[str, float]`

Transitivity of ε-redundancy. Returns `eps_ac_upper`, `tau_ac_lower_bound`, `propagation_loss`.

### `proposition_41_berry_esseen(n: int, tau: float = 0.0) -> Dict[str, Any]`

Berry-Esseen convergence rate for Kendall τ CLT. Returns `berry_esseen_bound`, `sigma_tau`, `ci_width_95`, `reliability_thresholds`.

### `sample_complexity_for_epsilon(eps: float, delta: float = 0.05, margin: float = 0.01) -> int`

Minimum n to estimate ε within ±margin at confidence 1−δ (Hoeffding).

### `epsilon_concentration_bound(n_pairs: int, observed_eps: float, delta: float = 0.05) -> Tuple[float, float]`

Hoeffding CI for ε. Returns `(lower, upper)`.

### `degradation_rate(eps1: float, eps2: float) -> float`

Rate of τ degradation w.r.t. ε. Returns `−4.0`.

### `construct_tightness_instance(n: int, eps1: float, eps2: float, seed: int = 42) -> TightnessVerification`

Construct explicit instance achieving the Theorem 4.1 bound via disjoint swaps.

### `verify_bound_random_instances(n_instances: int = 100, n_items: int = 20, seed: int = 42) -> Dict[str, Any]`

Verify Theorem 4.1 on random instances with Clopper-Pearson CI.

### `verify_bound_adversarial(n_instances: int = 200, n_items: int = 20, max_eps: float = 0.4, seed: int = 42) -> Dict[str, Any]`

Adversarial verification near the boundary.

### `compute_theorem_bounds(eps1: float, eps2: float, n_items: int = 13, overlap_fraction: float = 0.0, delta: float = 0.05) -> TheoremBounds`

Compute all quantitative bounds. Returns `TheoremBounds` dataclass.

### `run_comprehensive_verification(seed: int = 42) -> Dict[str, Any]`

Run all verification procedures (random, adversarial, tightness, proofs, Berry-Esseen).

### Data Classes

- **`TightnessVerification`** — `n`, `eps1`, `eps2`, `predicted_tau_lb`, `actual_tau`, `bound_holds`, `gap`
- **`TheoremBounds`** — `eps1`, `eps2`, `tau_lower_bound`, `tau_with_overlap`, `overlap_fraction`, `sample_complexity_for_eps`, `concentration_bound`, `degradation_rate`
- **`ProofStep`** — `step_number`, `statement`, `justification`, `formula`
- **`FormalProof`** — `theorem_id`, `statement`, `steps`, `qed`

---

## 4. Metric Algebra API

`src/distributional_analysis.py`

### `MetricAlgebra(metric_values: Dict[str, np.ndarray], delta: float = 0.3)`

Algebraic structure over diversity metrics. Defines equivalence classes where `M₁ ~ M₂ iff |τ(M₁,M₂)| ≥ 1 − δ`.

#### `equivalence_classes() -> List[List[str]]`

Compute equivalence classes via union-find.

#### `verify_transitivity() -> Dict`

Check transitivity of ~. Returns `is_transitive`, `n_violations`, `violations`.

#### `quotient_dimension() -> int`

Number of equivalence classes (dimension of M/~).

#### `summary() -> Dict`

Full algebra summary: classes, τ matrix, transitivity status.

### `normalized_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float`

NMI between two metric vectors. Ranges [0, 1]. Captures nonlinear dependencies.

### `variation_of_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float`

VI distance between two metric vectors. A proper metric on partitions.

### `info_theoretic_metric_comparison(metric_values: Dict[str, np.ndarray], n_bins: int = 10) -> Dict`

Compare metrics using NMI, VI, and Kendall τ. Returns matrices and summary.

### `berry_esseen_kendall_tau(n: int, tau: float = 0.0) -> Dict`

Berry-Esseen convergence rate for Kendall τ. Returns `berry_esseen_bound`, `n_for_gaussian_reliable`.

### `tightness_construction(n: int, eps1: float, eps2: float) -> Dict`

Construct instances achieving the tight Theorem 4.1 bound via disjoint swaps.

### `SubmodularityVerifier(f: Callable, ground_set_size: int)`

#### `verify_exact(max_subsets: int = 2000) -> Dict`

Exhaustive (n ≤ 8) or random-sampling (n > 8) submodularity check. Returns `is_submodular`, `n_violations`, `violation_rate_ci_95`.

### `PermutationTest(n_permutations: int = 10000, seed: int = 42)`

#### `test(x: np.ndarray, y: np.ndarray) -> Dict`

Permutation test for Kendall τ independence. Returns `observed_tau`, `p_value`, `is_significant_05`.

---

## 5. Entropy Correction API

`src/entropy_correction.py`

### `entropy_miller_madow(texts: List[str], n: int = 2) -> Tuple[float, float, float]`

Miller-Madow bias-corrected Shannon entropy. Returns `(H_MLE, H_MM, bias)`.

### `entropy_nsb(texts: List[str], n: int = 2, n_beta: int = 50) -> Tuple[float, float]`

NSB (Nemenman-Shafee-Bialek) Bayesian entropy estimator. Returns `(H_nsb, H_std)`.

### `entropy_jackknife(texts: List[str], n: int = 2) -> Tuple[float, float, float]`

Delete-1 jackknife bias-corrected entropy. Returns `(H_jackknife, H_mle, bias)`.

### `bootstrap_entropy_bca(texts: List[str], n: int = 2, n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 42, estimator: str = "miller_madow") -> CorrectedEntropyResult`

BCa bootstrap CI for entropy. Returns `CorrectedEntropyResult` with `mle`, `miller_madow`, `jackknife`, `nsb`, `ci_lower`, `ci_upper`, `ci_method`.

### `kl_laplace(texts_p: List[str], texts_q: List[str], n: int = 2) -> float`

KL divergence with Laplace (add-1) smoothing.

### `kl_jelinek_mercer(texts_p: List[str], texts_q: List[str], n: int = 2, lambda_: float = 0.1) -> float`

KL divergence with Jelinek-Mercer interpolation smoothing.

### `kl_dirichlet(texts_p: List[str], texts_q: List[str], n: int = 2, alpha: float = 0.01) -> float`

KL divergence with symmetric Dirichlet prior smoothing.

### `smoothed_kl_analysis(texts_p: List[str], texts_q: List[str], n: int = 2) -> SmoothedKLResult`

Full KL analysis with all smoothing methods. Returns `SmoothedKLResult` with `kl_raw`, `kl_laplace`, `kl_jelinek_mercer`, `kl_dirichlet`, `smoothing_impact`, zero-mass diagnostics.

### `entropy_rate_corrected(texts: List[str], max_order: int = 5) -> EntropyRateCorrected`

Bias-corrected entropy rate with Miller-Madow at each order. Returns `rate`, `conditionals`, `memory_length`, `convergence_diagnostic`.

### `corrected_info_theory_analysis(high_diversity_texts: List[str], low_diversity_texts: List[str], n: int = 2, n_bootstrap: int = 1000, seed: int = 42) -> Dict`

Full bias-corrected information-theoretic analysis. Returns JSON-serializable dict with corrected entropy, smoothed KL, and entropy rate sections.

---

## 6. Lattice Analysis API

`src/metric_lattice.py`

### `lattice_analysis(metric_values: Dict[str, np.ndarray]) -> Dict`

Run lattice analysis and return JSON-serializable summary.

### `MetricLattice(metric_values: Dict[str, np.ndarray])`

#### `equivalence_classes(delta: float) -> List[FrozenSet[str]]`

Equivalence classes at threshold δ via union-find.

#### `hasse_diagram(delta: float) -> HasseDiagram`

Hasse diagram (covering relation) at threshold δ.

#### `delta_filtration(n_thresholds: int = 50) -> List[Tuple[float, int, List[FrozenSet[str]]]]`

How equivalence classes evolve as δ varies from 0.01 to 0.99.

#### `merge_sequence() -> List[Tuple[float, str, str]]`

Sequence of δ values at which metric pairs merge. Sorted ascending.

#### `compute_meet_join(delta: float) -> Tuple[Dict, Dict]`

Meet (∧) and join (∨) tables for equivalence classes.

#### `is_lattice(delta: float) -> bool`

Check if M/~_δ forms a lattice (absorption laws).

#### `compute_automorphisms(delta: float) -> List[Dict[str, str]]`

Automorphism group of the metric equivalence graph (exhaustive, n ≤ 8).

#### `betti_numbers(delta: float, max_dim: int = 3) -> List[int]`

Betti numbers of the Vietoris-Rips complex. β₀ = components, β₁ = 1-holes.

#### `full_analysis(n_thresholds: int = 20) -> LatticeStructure`

Complete analysis: filtration, Hasse diagrams, meet/join, automorphisms, Betti numbers.

### Data Classes

- **`LatticeElement`** — `metrics: FrozenSet[str]`, `level: int`, `delta: float`
- **`HasseDiagram`** — `elements`, `edges`, `delta`, `n_classes`
- **`LatticeStructure`** — `hasse_diagrams`, `filtration`, `merge_sequence`, `is_lattice`, `meet_table`, `join_table`, `automorphisms`, `betti_numbers`, `summary`

---

## 7. SMT Solver API

`src/smt_diversity.py` — Requires Z3 (`Z3_AVAILABLE` flag guards imports).

### `SMTDiversityOptimizer(timeout_ms: int = 10000)`

#### `solve_exact(distance_matrix: np.ndarray, k: int, groups=None, min_per_group=None, objective: str = "sum_pairwise") -> SMTSelectionResult`

Exact optimal diverse subset via Z3. Objectives: `"sum_pairwise"`, `"min_pairwise"`, `"facility_location"`.

#### `greedy_sum_pairwise(D: np.ndarray, k: int, groups=None, min_per_group=None) -> Tuple[List[int], float]` (static)

Greedy heuristic for sum-pairwise diversity.

#### `greedy_min_pairwise(D: np.ndarray, k: int) -> Tuple[List[int], float]` (static)

Greedy heuristic for min-pairwise (maximin) diversity.

#### `compute_optimality_gaps(n_values: List[int], k_values: List[int], ...) -> List[OptimalityGap]`

Compare greedy vs exact solutions across instance sizes.

#### `certify_fair_retention(D: np.ndarray, k: int, groups: np.ndarray, constraint_levels: List[Dict]) -> List[FairRetentionCertificate]`

Certified worst-case fair retention bounds via SMT.

#### `np_hardness_reduction(n_instances: int = 20, ...) -> List[HardnessWitness]`

NP-hardness witnesses via reduction from Max-k-Dispersion.

#### `run_benchmark(max_n: int = 20, n_trials: int = 5, ...) -> SMTBenchmarkResult`

Comprehensive benchmark across sizes.

### `ILPDiversityOptimizer`

#### `solve_ilp(D: np.ndarray, k: int, groups=None, min_per_group=None, ...) -> Tuple[List[int], float]` (static)

ILP fallback for larger instances. Uses scipy `linprog`.

### `smt_benchmark(max_n: int = 20, n_trials: int = 5, seed: int = 42) -> Dict`

Module-level convenience function for running benchmarks.

### Data Classes

- **`SMTSelectionResult`** — `selected_indices`, `objective_value`, `status`, `solve_time_ms`, `n_items`, `k`
- **`OptimalityGap`** — `n`, `k`, `exact_obj`, `greedy_obj`, `gap_pct`, `greedy_time_ms`, `exact_time_ms`
- **`FairRetentionCertificate`** — `constraints`, `is_feasible`, `objective_value`, `diversity_loss_pct`
- **`HardnessWitness`** — `n`, `k`, `greedy_obj`, `exact_obj`, `gap_pct`, `is_hard`
- **`SMTBenchmarkResult`** — `gaps`, `certificates`, `witnesses`, `summary`

---

## 8. JSONL & HuggingFace JSON Input

`src/io/jsonl_loader.py`

### `load_texts_jsonl(path: str, text_field: str = None) -> List[str]`

Load texts from a JSONL file (one JSON object per line). Auto-detects field name from: `text`, `output`, `response`, `content`, `generation`, `completion`. Use `text_field` to override.

### `load_texts_hf_json(path: str, text_field: str = None) -> List[str]`

Load texts from HuggingFace-style JSON: `[{"text": ...}, ...]` or `{"data": [{"text": ...}, ...]}`.

### `load_texts_auto(path: str, text_field: str = None) -> List[str]`

Auto-detect format by file extension (`.jsonl` → JSONL, `.json` → HF JSON, else plain text).

### CLI Flags

| Flag | Command | Description |
|------|---------|-------------|
| `--file` | `divflow metrics` | Input file (`.jsonl`, `.json`, or plain text — auto-detected) |
| `--text-field` | `divflow metrics` | Custom JSON field name for text extraction |
| `--input-format` | `diversity_taxonomy.py` | Force input format: `json` or `jsonl` (default: auto-detect) |
| `--text-field` | `diversity_taxonomy.py` | Custom JSON field name for text extraction |
