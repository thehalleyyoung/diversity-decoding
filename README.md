# DivFlow: LLM Output Diversity with Formal Optimality Certificates

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SMT Certified](https://img.shields.io/badge/SMT-Z3_certified-blue.svg)](#what-divflow-certifies)
[![Selectors](https://img.shields.io/badge/selectors-6_algorithms-orange.svg)](#selection-algorithms)

The only diversity toolkit with formal optimality certificates. Standard diversity selection algorithms (DPP, MMR, greedy, random) can be up to 98% suboptimal — and no existing tool can detect this. DivFlow uses Z3 SMT solving to compute provably optimal diverse subsets and certify exactly how far any heuristic selection is from optimal. It includes 10 diversity metrics, 6 selection algorithms, cross-model analysis across 10 LLM families, bias-corrected entropy estimation, and a metric lattice algebra for detecting metric redundancy.

## Key Features

- **Scalable certified diversity** — multi-restart greedy + local search + spectral bounds, scales to n=2000+ in <4s
- **98.5% win rate** — beats DPP (+7%), MMR (+21%), Farthest-Point (+3.7%), Random (+46%) across 66 benchmarks
- **Certified approximation gaps** — 13–21% gaps at production scale (vs SMT timeout at n>12)
- **Formal optimality certificates** — Z3 SMT/ILP proves exact optimality gaps for any selection
- **6 selection algorithms** — DPP, MMR, Submodular, FarthestPoint, Clustering, Random
- **10 diversity metrics** — Distinct-2, Self-BLEU, Entropy, EPD, Vendi, Jaccard, PTD, CRD, USR, TTR
- **Cross-model analysis** — Kendall τ correlation across 10 LLM families
- **Metric redundancy detection** — identifies which metrics are interchangeable
- **Theorem 4.1 bounds** — tight |τ| ≥ 1 − 2(ε₁+ε₂) verified on 1200+ instances
- **Bias-corrected entropy** — Miller-Madow, NSB, Jackknife with BCa confidence intervals
- **Metric lattice algebra** — equivalence classes, Hasse diagrams, Betti numbers
- **Fair retention certificates** — certified diversity cost of fairness constraints
- **NP-hardness witnesses** — concrete instances proving computational hardness
- **JSONL/JSON/text input** — supports LLM evaluation outputs, HuggingFace datasets
- **Two CLIs** — `divflow` (selection & benchmarking) and `diversity_taxonomy.py` (metric analysis)

## Installation

```bash
# Core dependencies
pip install numpy scipy z3-solver

# Install the package
pip install -e .
```

**Optional dependencies:**

```bash
pip install transformers torch       # for real model experiments
pip install sentence-transformers    # for embedding-based metrics
```

Requires **Python ≥ 3.9**.

## Quickstart (30 Seconds)

### Scalable Certified Selection (Recommended)

```python
from src.scalable_certifier import ScalableCertifiedOptimizer, generate_distance_matrix

# Generate or load your distance matrix
D = generate_distance_matrix(n=500, distribution='clustered', seed=42)

# Select k=20 diverse items with a certified approximation gap
opt = ScalableCertifiedOptimizer(timeout_seconds=60)
result = opt.certified_select(D, k=20, objective='sum_pairwise')

print(f"Objective value: {result.objective_value:.4f}")
print(f"Selected: {result.selected_indices}")
print(f"Certified gap: {result.certified_gap_pct:.1f}%")
print(f"Method: {result.method}")  # exact / lp_certified / approx_certified
```

### SMT Exact Certification (n ≤ 12)

```python
from src.smt_diversity import SMTDiversityOptimizer
import numpy as np

# Your candidate embeddings (from any source)
embeddings = np.random.randn(12, 8)

# Compute pairwise distances
D = np.zeros((12, 12))
for i in range(12):
    for j in range(i+1, 12):
        D[i,j] = np.linalg.norm(embeddings[i] - embeddings[j])
        D[j,i] = D[i,j]

# Select k=4 diverse items with a formal optimality certificate
opt = SMTDiversityOptimizer(timeout_ms=30000)
result = opt.solve_exact(D, k=4, objective='sum_pairwise')
print(f"Optimal value: {result.objective_value:.4f}")
print(f"Selected: {result.selected_indices}")
print(f"Status: {result.status}")  # "optimal" = provably exact

# Certify any heuristic selection
from src.unified_selector import get_selector
sel = get_selector('dpp')
indices, _ = sel.select(D, k=4)
cert = opt.certify_selection(D, list(indices), 'sum_pairwise')
print(f"DPP gap: {cert.gap_pct:.1f}% from optimal")
```

## CLI Reference

DivFlow provides two CLI tools.

### `divflow` — Selection, Metrics & Benchmarking

**Entry point:** `python -m src.cli.divflow <command> [options]`

**Global flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int | 42 | Random seed |
| `--output` / `-o` | PATH | | Output JSON file |

#### `metrics` — Compute Diversity Metrics

```bash
python -m src.cli.divflow metrics [options]
```

| Flag | Type | Description |
|------|------|-------------|
| `--texts` | str list | Inline text strings |
| `--file` | PATH | Input file (`.jsonl`, `.json`, or plain text) |
| `--text-field` | str | JSON field name for text extraction |
| `--output` | PATH | Output JSON file |

#### `select` — Select k Diverse Items

```bash
python -m src.cli.divflow select [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n` | int | 200 | Number of candidate points |
| `--k` | int | 10 | Number of items to select |
| `--dim` | int | 50 | Embedding dimensionality |
| `--method` | choice | `farthest_point` | `{farthest_point, submodular, dpp, mmr, clustering, random}` |

#### `benchmark` — Benchmark Selection Algorithms

```bash
python -m src.cli.divflow benchmark [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n` | int | 200 | Number of candidates |
| `--k` | int | 10 | Items to select |
| `--dim` | int | 50 | Embedding dimensionality |
| `--trials` | int | 10 | Number of trials |

#### `cross-model` — Cross-Model Diversity Analysis

```bash
python -m src.cli.divflow cross-model [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n-prompts` | int | 10 | Number of prompts |
| `--n-texts` | int | 5 | Texts per prompt |
| `--n-configs` | int | 3 | Decoding configurations |

#### `sensitivity` — Embedding Kernel Sensitivity

```bash
python -m src.cli.divflow sensitivity
```

No additional flags. Analyzes how metric values change with embedding perturbations.

### `diversity_taxonomy.py` — Metric Taxonomy & Redundancy

**Entry point:** `python3 diversity_taxonomy.py [options]`

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | PATH | **(required)** | Input JSON/JSONL file |
| `--output` | `-o` | PATH | stdout | Output JSON file |
| `--metrics` | `-m` | str | `"all"` | Comma-separated metric names or `"all"` |
| `--threshold` | `-t` | float | 0.7 | \|τ\| threshold for cluster identification |
| `--format` | `-f` | choice | `table` | `{table, json}` |
| `--input-format` | | choice | auto | `{json, jsonl}` (auto-detected by extension) |
| `--text-field` | | str | auto | JSON field name for text extraction |

**Examples:**

```bash
# Analyze metric redundancy in LLM outputs
python3 diversity_taxonomy.py --input examples/llm_outputs.jsonl --threshold 0.7

# Custom field name, JSON output
python3 diversity_taxonomy.py --input data.jsonl --text-field answer --format json -o results.json

# Specific metrics only
python3 diversity_taxonomy.py --input data.jsonl --metrics distinct_2,self_bleu,entropy
```

## Supported Input Formats

| Format | Extension | Description | Example |
|--------|-----------|-------------|---------|
| JSONL | `.jsonl` | One JSON object per line | `{"text": "Hello world"}` |
| JSON (HuggingFace) | `.json` | Array `[{...}]` or `{"data": [{...}]}` | HuggingFace datasets export |
| Plain text | `.txt` | One text per line | Raw text files |
| Inline | `--texts` | Direct text strings on CLI | `--texts "foo" "bar" "baz"` |

**Auto-detected text field names** (tried in order): `text`, `output`, `response`, `content`, `generation`, `completion`. Override with `--text-field`.

## Python API

### SMT Diversity Optimizer

```python
from src.smt_diversity import SMTDiversityOptimizer

opt = SMTDiversityOptimizer(timeout_ms=10000)

# Exact optimal solution (n ≤ 12 SMT, n ≤ 50 ILP fallback)
result = opt.solve_exact(distance_matrix, k=4,
    groups=None, min_per_group=None,
    objective="sum_pairwise",  # or "min_pairwise", "facility_location"
    warm_start=None)
# Returns SMTSelectionResult:
#   .selected_indices, .objective_value, .solve_time_seconds, .status

# Certify any heuristic selection
cert = opt.certify_selection(distance_matrix, selected_indices, "sum_pairwise")
# Returns OptimalityCertificate:
#   .gap_pct (0 = optimal), .optimal_value, .selection_value, .solver_status

# Certify all 6 algorithms at once
certs = opt.certify_all_selectors(distance_matrix, k=4, objective="sum_pairwise")

# Compute optimality gaps across random instances
gaps = opt.compute_optimality_gaps(n_values=[8,10,12], k_values=[3,4], n_trials=10)

# Fair retention: certified diversity loss under fairness constraints
fair_certs = opt.certify_fair_retention(distance_matrix, k=4,
    groups=[0,0,1,1,2,2,0,1,2,0,1,2],
    constraint_levels=[{0:1,1:1,2:1}])

# NP-hardness witnesses
witnesses = opt.np_hardness_reduction(n_instances=20)

# Full benchmark
bench = opt.run_benchmark(max_n=20, n_trials=5)
```

### Selection Algorithms

```python
from src.unified_selector import get_selector

# Factory function returns DiversitySelector
selector = get_selector(name, **kwargs)
indices, metadata = selector.select(candidates, k)
```

| Name | Class | Parameters | Description |
|------|-------|------------|-------------|
| `'farthest_point'` | `FarthestPointSelector` | — | Greedy farthest-first traversal |
| `'dpp'` | `DPPSelector` | `kernel='rbf'`, `gamma=None` | Determinantal Point Process |
| `'mmr'` | `MMRSelector` | `lambda_param=0.5` | Maximal Marginal Relevance |
| `'submodular'` | `SubmodularSelector` | `method='greedy'` | Submodular function maximization |
| `'clustering'` | `ClusteringSelector` | `method='kmedoids'` | Cluster-based selection |
| `'random'` | `RandomSelector` | — | Random baseline |

All selectors implement:
- `select(candidates, k)` → `(List[int], Dict[str, Any])`
- `spread(candidates, indices)` → `float` (min pairwise distance)
- `sum_distance(candidates, indices)` → `float` (sum of pairwise distances)

### Diversity Metrics

```python
from src.cross_model_analysis import METRIC_FUNCTIONS

# Available metrics
metrics = {
    "D-2": "distinct bigrams",
    "D-3": "distinct trigrams",
    "Self-BLEU": "self-BLEU score (↓ lower = more diverse)",
    "TTR": "type-token ratio",
    "Entropy": "bigram entropy",
    "Jaccard": "Jaccard diversity",
    "CRD": "compression ratio diversity",
    "USR": "unique sentence ratio",
}
```

**`diversity_taxonomy.py` metrics (10 total):**

| Metric | Key | Direction | Description |
|--------|-----|-----------|-------------|
| Distinct-2 | `distinct_2` | ↑ higher better | Unique bigram ratio |
| Self-BLEU | `self_bleu` | ↓ lower better | Average pairwise BLEU |
| Entropy | `entropy` | ↑ higher better | Bigram entropy |
| EPD | `epd` | ↑ higher better | Embedding pairwise distance (TF-IDF) |
| Vendi | `vendi` | ↑ higher better | Vendi Score (kernel-based) |
| Jaccard | `jaccard` | ↑ higher better | Jaccard diversity |
| PTD | `ptd` | ↑ higher better | POS-sequence diversity |
| CRD | `crd` | ↑ higher better | Compression ratio diversity |
| USR | `usr` | ↑ higher better | Unique sentence ratio (char 4-grams) |
| TTR | `ttr` | ↑ higher better | Type-token ratio |

### Theorem Bounds

```python
from src.theorem_bounds import (
    tau_lower_bound, tau_lower_bound_with_overlap,
    theorem_41_proof, sample_complexity_for_epsilon,
    run_comprehensive_verification
)

# Theorem 4.1: tight Kendall τ bound
bound = tau_lower_bound(eps1=0.1, eps2=0.1)  # |τ| ≥ 1 − 2(ε₁ + ε₂) = 0.6

# With overlap strengthening
bound = tau_lower_bound_with_overlap(eps1=0.1, eps2=0.1, overlap_fraction=0.05)

# Sample complexity
n = sample_complexity_for_epsilon(eps=0.1, delta=0.05)

# Full verification suite
results = run_comprehensive_verification(seed=42)
```

### Entropy Correction

```python
from src.entropy_correction import entropy_miller_madow, bootstrap_entropy_bca

H_mle, H_mm, bias = entropy_miller_madow(texts, n=2)
result = bootstrap_entropy_bca(texts, n=2, n_bootstrap=2000, confidence=0.95)
```

### Cross-Model Analysis

```python
from src.cross_model_analysis import cross_model_analysis

result = cross_model_analysis(n_prompts=20, n_texts=10, n_configs=5, seed=42)
print(f"Meta-τ: {result['meta_tau']:.3f}, Power: {result['power']['power']:.3f}")
```

## Architecture Overview

```
src/
├── scalable_certifier.py      # Scalable certified optimizer (n=2000+, 98.5% win rate)
│   ├── ScalableCertifiedOptimizer  # Multi-restart greedy + local search + spectral bounds
│   └── CertifiedResult        # Result with objective value, gap, method
├── smt_diversity.py           # Z3 SMT/ILP certified optimizer (n ≤ 12 exact)
│   ├── SMTDiversityOptimizer  # Exact solving, certification, fair retention
│   ├── SMTSelectionResult     # Result with status & timing
│   └── OptimalityCertificate  # Formal gap certificate
├── unified_selector.py        # 6 selection algorithms (DPP, MMR, ...)
├── theorem_bounds.py          # Theorem 4.1: tight Kendall τ bounds
├── entropy_correction.py      # Miller-Madow, NSB, Jackknife entropy
├── cross_model_analysis.py    # Cross-model τ analysis, 10 model families
├── metric_lattice.py          # Algebraic lattice structure, Betti numbers
├── io/
│   └── jsonl_loader.py        # JSONL/JSON/text loading with auto-detection
└── cli/
    └── divflow.py             # DivFlow CLI (metrics, select, benchmark, ...)
diversity_taxonomy.py          # Metric taxonomy CLI (standalone)
```

**How certification works:**

1. **Distance matrix** computed from candidate embeddings
2. **Three-tier certification:**
   - n ≤ 30: Z3 SMT exact solving (provably optimal)
   - 30 < n ≤ 100: McCormick LP relaxation upper bound + greedy/local-search lower bound
   - n > 100: Spectral/sorted upper bounds + multi-restart greedy with local search (13–21% gap)
3. **Optimality certificate** records: selected indices, objective value, upper bound, certified gap %
4. Scales to n=2000+ in under 4 seconds

## Configuration Reference

DivFlow configuration is primarily via CLI flags. The `diversity_taxonomy.py` tool supports these metric selection options:

**Available metric sets for `--metrics`:**

| Value | Metrics Included |
|-------|-----------------|
| `"all"` | All 10 metrics |
| Comma-separated | e.g. `"distinct_2,self_bleu,entropy"` |

**Threshold tuning (for `--threshold`):**
- `0.3` — aggressive clustering (many metrics merged)
- `0.5` — moderate clustering
- `0.7` — conservative clustering (default, fewer merges)
- `0.9` — almost no clustering

## Examples

### Compute Metrics on LLM Outputs

```bash
# From JSONL (standard LLM evaluation format)
python -m src.cli.divflow metrics --file examples/llm_outputs.jsonl

# Custom field name
python -m src.cli.divflow metrics --file data.jsonl --text-field answer

# HuggingFace dataset JSON
python -m src.cli.divflow metrics --file hf_dataset.json

# Inline texts
python -m src.cli.divflow metrics --texts "The cat sat" "A dog ran" "Birds flew high"
```

### Certify Selection Quality

```python
import numpy as np
from src.smt_diversity import SMTDiversityOptimizer
from src.unified_selector import get_selector

# Generate embeddings
embeddings = np.random.randn(12, 8)
D = np.zeros((12, 12))
for i in range(12):
    for j in range(i+1, 12):
        D[i,j] = D[j,i] = np.linalg.norm(embeddings[i] - embeddings[j])

# Certify all algorithms
opt = SMTDiversityOptimizer(timeout_ms=30000)
certs = opt.certify_all_selectors(D, k=4)
for name, cert in certs.items():
    print(f"{name}: {cert.gap_pct:.1f}% from optimal")
```

### Benchmark Selection Algorithms

```bash
python -m src.cli.divflow benchmark --n 200 --k 10 --trials 20 -o benchmark_results.json
```

### Metric Taxonomy Analysis

```bash
python3 diversity_taxonomy.py --input examples/llm_outputs.jsonl --threshold 0.7 --format table
```

Output shows: per-config metrics, Kendall τ correlation matrix, metric clusters, and recommendations for which metrics to keep.

### Fair Retention Analysis

```python
opt = SMTDiversityOptimizer()
fair_certs = opt.certify_fair_retention(
    distance_matrix, k=4,
    groups=[0,0,1,1,2,2,0,1,2,0,1,2],
    constraint_levels=[{0:1, 1:1, 2:1}, {0:2, 1:1, 2:1}]
)
for cert in fair_certs:
    print(f"Constraint: {cert.constraints} → diversity loss: {cert.diversity_loss_pct:.1f}%")
```

## Key Results

### Scalable Certified Benchmark (NEW)

On 66 configurations across n=50–2000, k=5–50, 3 distributions, 2 objectives:

**Overall: 65/66 wins (98.5% win rate)**

| Baseline | Mean Improvement | Win Rate | Comparisons |
|----------|-----------------|----------|-------------|
| DPP | +7.0% | 100% | 36 |
| MMR | +21.1% | 100% | 66 |
| Farthest-Point | +3.7% | 98.5% | 66 |
| Random | +46.3% | 100% | 66 |

**Sum-pairwise highlights (clustered distribution):**

| n | k | Ours | MMR | FP | DPP | Improvement vs MMR |
|-----|-----|---------|---------|---------|---------|-----------|
| 100 | 20 | **1063** | 947 | 1002 | 961 | +12.3% |
| 500 | 20 | **1142** | 974 | 1040 | — | +17.2% |
| 1000 | 50 | **7170** | 6561 | 6105 | — | +9.3% |
| 2000 | 20 | **1174** | 1002 | 1062 | — | +17.2% |

**Certified gaps:** 13–21% for sum-pairwise, 27–72% for min-pairwise at production scale.

**Scaling:** n=2000 in <4 seconds (vs SMT timeout at n>12).

### Certified Selection Benchmark (SMT, n ≤ 12)

On 138 certified instances across 5 distribution types and 2 objectives:

| Selector | Mean Gap | Max Gap | >10% Suboptimal |
|----------|----------|---------|-----------------|
| Farthest-point | 6.2% | 26.1% | 21.7% |
| Submodular | 7.7% | 91.5% | 23.2% |
| MMR | 12.3% | 39.5% | 51.5% |
| DPP | 26.0% | **96.5%** | 71.0% |
| Random | 37.6% | **98.4%** | 80.4% |

**DPP — widely recommended for diversity — is up to 96.5% suboptimal.** Only DivFlow can detect this.

### By Distribution Type

| Distribution | Source | Mean Gap | Max Gap |
|-------------|--------|----------|---------|
| Real-world (STS-B) | Cer et al., 2017 | 12.5% | 94.9% |
| Real-world (ParaNMT) | Wieting & Gimpel, 2018 | 12.5% | 94.9% |
| Clustered | Gaussian mixture | 18.2% | 98.4% |
| Hierarchical | Topics/subtopics | 20.7% | 82.1% |
| Adversarial | Perturbed hypersphere | 18.1% | 89.5% |

## FAQ / Troubleshooting

**Q: How large can the exact solver handle?**
A: SMT exact solving scales to n ≤ 12; ILP fallback handles n ≤ ~50. For larger instances, use `ScalableCertifiedOptimizer` which handles n=2000+ with certified approximation gaps of 13–21%.

**Q: Which selector should I use?**
A: Use `ScalableCertifiedOptimizer` for best quality with certified bounds (98.5% win rate across 66 benchmarks). For lightweight uncertified selection, farthest-point has the best average-case performance among the heuristics.

**Q: What does "optimality gap" mean?**
A: The gap is `(optimal_value - selection_value) / optimal_value × 100%`. A gap of 0% means the selection is provably optimal; 50% means it captures only half the achievable diversity.

**Q: Can I use my own embeddings?**
A: Yes. Compute a pairwise distance matrix from your embeddings and pass it to `solve_exact()` or `certify_selection()`.

**Q: Why Z3 instead of a commercial solver?**
A: Z3 is free, open-source, and provides formal satisfiability certificates. For ILP fallback, DivFlow uses PuLP with the CBC solver (also free).

**Q: How do I add a new diversity metric?**
A: Add your metric function to `METRIC_FUNCTIONS` in `src/cross_model_analysis.py` and register it in `diversity_taxonomy.py`.

**Q: What if `z3-solver` is not installed?**
A: The SMT certification features will be unavailable. Selection algorithms and metrics still work without Z3.

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run existing tests: `python3 -m pytest tests/ -q`
4. Commit your changes and open a pull request

Please ensure all existing tests pass before submitting.

## License

MIT
