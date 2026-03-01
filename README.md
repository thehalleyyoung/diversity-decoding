# DivFlow: Unified Diversity Metrics & Selection for AI Text

**A practical Python toolkit for measuring, comparing, and optimizing diversity in LLM outputs.**

## 30-Second Quickstart

```bash
cd implementation
python3 -m pytest tests/test_pathb.py tests/test_distributional.py tests/test_property_invariants.py -q
# 83 passed in ~6s
```

```python
from implementation.src.unified_selector import get_selector
import numpy as np

# Select 10 diverse items from 200 candidates
embeddings = np.random.randn(200, 50)
selector = get_selector('farthest_point')
indices, metadata = selector.select(embeddings, k=10)
print(f"Selected {len(indices)} diverse items")
# Selected 10 diverse items
```

```python
# Bias-corrected entropy analysis
from implementation.src.entropy_correction import corrected_info_theory_analysis
results = corrected_info_theory_analysis(high_div_texts, low_div_texts)
print(f"H_MM = {results['shannon_entropy']['high_diversity']['miller_madow']}")
# H_MM = 9.8146
print(f"KL (Laplace) = {results['kl_divergence']['laplace_smoothed']}")
# KL (Laplace) = 0.645
```

```python
# Berry-Esseen convergence rate for Kendall τ
from implementation.src.distributional_analysis import berry_esseen_kendall_tau
result = berry_esseen_kendall_tau(n=13)
print(result['interpretation'])
# For n=13 items, the CDF of standardized τ̂ deviates from Gaussian by at most 0.1317.
# Gaussian approximation reliable (bound<0.05) for n≥8.
```

## Key Results

| Finding | Value | Evidence |
|---------|-------|----------|
| Metric redundancy | 10/11 metrics at \|τ\| > 0.7 | `comprehensive_v2_results.json` |
| Tight bound (Thm 4.1) | \|τ\| ≥ 1 − 2(ε₁+ε₂), **100% verified** | `math_rigor_results.json` |
| Clopper-Pearson CI | 95% CI [0.964, 1.000] on pass rate | `math_rigor_results.json` |
| Corollary 4.3 | Transitivity **100% verified** (50/50) | `math_rigor_results.json` |
| Berry-Esseen rate | \|F(t)−Φ(t)\| ≤ 0.4748/√n | `distributional_analysis.py` |
| FP vs Random (spread) | d = 7.42, p < 10⁻³⁵ | `comprehensive_v2_results.json` |
| Text D-2 improvement | +38% (0.982 vs 0.713) | `comprehensive_v2_results.json` |
| Fair retention | 97.9% [97.5%, 98.2%] | `comprehensive_v2_results.json` |
| **SMT certified retention** | **≥ 96.8% (exact)** | `smt_results.json` |
| **Greedy optimality gap** | **mean 0.95%, max 5.72%** | `smt_results.json` |
| **Cross-model τ (10 models)** | **0.33 [0.32, 0.34], 45 pairs** | `cross_model_results.json` |
| **Entropy (Miller-Madow)** | **9.81 bits (bias +0.61)** | `entropy_corrected_results.json` |
| **KL (Laplace smoothed)** | **0.65 bits (was 26.86)** | `entropy_corrected_results.json` |
| **Failure modes** | **13 across 6 categories** | `failure_taxonomy_results.json` |
| **Lattice structure** | **β₀=5, lattice at δ=0.3** | `lattice_results.json` |
| **Submodularity** | **Exhaustive n≤8; sampling n>8** | `test_pathb.py` |
| Tests passing | **83/83** (53 PATH B + 12 dist. + 18 prop.) | `pytest` |

## Contents

- `implementation/src/` — Full DivFlow toolkit
  - `src/unified_selector.py` — Unified selector API (DPP, MMR, Submodular, FarthestPoint, Clustering)
  - `src/text_diversity_toolkit.py` — Text diversity analysis (D-n, Self-BLEU, TF-IDF, NMF)
  - `src/fair_diversity.py` — Fairness-aware selection with group constraints
  - `src/smt_diversity.py` — SMT/ILP exact optimization (Z3), optimality gaps, NP-hardness
  - `src/entropy_correction.py` — Miller-Madow, NSB, jackknife entropy; Laplace/JM KL smoothing
  - `src/cross_model_analysis.py` — Cross-model analysis for 10 model families
  - `src/metric_lattice.py` — Lattice structure, Hasse diagrams, Betti numbers
  - `src/failure_taxonomy.py` — 13-mode failure taxonomy with detector
  - `src/distributional_analysis.py` — Metric algebra, NMI/VI, Berry-Esseen, submodularity
- `implementation/experiments/` — Reproducible experiment scripts
  - `experiments/run_pathb_experiments.py` — PATH B experiment runner
  - `experiments/pathb_results/` — All PATH B result artifacts
- `implementation/tests/` — 83 tests
  - `test_property_invariants.py` — 18 metric invariant tests
  - `test_distributional.py` — 12 distributional analysis tests
  - `test_pathb.py` — 53 PATH B tests (SMT, entropy, cross-model, lattice, taxonomy, IT baselines)
- `theory/tool_paper.tex` — Paper (25 pages, compiles with `pdflatex`)
- `API.md` — API reference
- `grounding.json` — Every claim → artifact mapping

## Running

```bash
cd implementation

# Run all tests (83 tests, ~6s)
python3 -m pytest tests/test_property_invariants.py tests/test_distributional.py tests/test_pathb.py -v

# Run PATH B experiments (~70s)
PYTHONPATH=. python3 experiments/run_pathb_experiments.py

# Compile paper
cd ../theory && pdflatex tool_paper.tex && pdflatex tool_paper.tex
```

## Requirements

- Python 3.9+
- numpy, scipy
- z3-solver (for SMT exact optimization)
- No LLM API access required (all experiments use synthetic/cached data)
