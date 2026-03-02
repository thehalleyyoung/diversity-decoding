# DivFlow

Measure, compare, and optimize diversity in LLM outputs. Statistically grounded selection algorithms, cross-model analysis across 10 model families, and SMT-verified optimality bounds.

## Quickstart

```bash
pip install numpy scipy
```

```python
from src.cross_model_analysis import cross_model_analysis
results = cross_model_analysis(n_prompts=5, n_texts=5, n_configs=2)
print(f"Meta-τ across {results['n_models']} models: {results['meta_tau']:.4f}")
```

```bash
# CLI one-liner
python3 -m src.cli.divflow benchmark --n 200 --k 10
```

## What It Does

**Select diverse subsets** — FarthestPoint selection achieves 27% better spread than random (78.71 vs 61.87, p < 10⁻³²):

```python
from src.unified_selector import get_selector
import numpy as np

embeddings = np.random.randn(200, 50)
selector = get_selector('farthest_point')
indices, meta = selector.select(embeddings, k=10)
print(f"Spread: {meta['spread']:.2f}")  # ~78 vs ~62 for random
```

Available selectors: `farthest_point`, `dpp`, `mmr`, `submodular`, `clustering`, `random`.

**Verify theoretical bounds** — Theorem 4.1 (|τ| ≥ 1 − 2(ε₁+ε₂)) verified on 1200+ instances at 100%:

```python
from src.theorem_bounds import compute_theorem_bounds
bounds = compute_theorem_bounds(eps1=0.1, eps2=0.1, n_items=13)
print(f"τ lower bound: {bounds.tau_lower_bound:.2f}")  # 0.60
```

**Correct entropy bias** — KL divergence fixed from 22.66 to 0.37 bits with bias correction:

```python
from src.entropy_correction import corrected_info_theory_analysis
results = corrected_info_theory_analysis(high_div_texts, low_div_texts)
print(f"H_MM = {results['shannon_entropy']['high_diversity']['miller_madow']:.4f}")
```

**Certify optimality with SMT** — Mean gap 0.55%, max 5.72% between greedy and exact:

```python
from src.smt_diversity import smt_benchmark
results = smt_benchmark(max_n=20, n_trials=5)
print(f"Mean gap: {results['optimality_gaps']['mean_gap_pct']:.2f}%")
```

## Key Results

| Result | Value |
|--------|-------|
| FarthestPoint vs Random spread | 78.71 vs 61.87 (p < 10⁻³²) |
| Theorem 4.1 verification | 100% on 1200+ instances |
| Cross-model analysis | 10 families, 45 pairs |
| SMT optimality gap | mean 0.55%, max 5.72% |
| Entropy bias correction | KL: 22.66 → 0.37 bits |
| Tests | 77 passing |

## CLI

```bash
# Diversity metrics on texts
python3 -m src.cli.divflow metrics --texts "Hello world" "Goodbye" "Hi there"

# Select diverse subset
python3 -m src.cli.divflow select --method farthest_point --n 200 --k 10

# Benchmark all selection algorithms
python3 -m src.cli.divflow benchmark --n 200 --k 10

# Cross-model analysis (10 model families)
python3 -m src.cli.divflow cross-model

# Embedding kernel sensitivity
python3 -m src.cli.divflow sensitivity
```

## Modules

| Module | Purpose |
|--------|---------|
| `src/cross_model_analysis.py` | Cross-model Kendall τ analysis across 10 model families (45 pairs) |
| `src/theorem_bounds.py` | Theorem 4.1 bounds: |τ| ≥ 1 − 2(ε₁+ε₂) with sample complexity |
| `src/distributional_analysis.py` | Metric algebra, NMI, variation of information, Berry-Esseen bounds |
| `src/entropy_correction.py` | Miller-Madow, NSB, Jackknife entropy estimators with BCa bootstrap CIs |
| `src/smt_diversity.py` | Z3 SMT/ILP solver for exact optimality gaps and fair retention certificates |
| `src/metric_lattice.py` | Lattice structure over metrics: Hasse diagrams, δ-filtration, Betti numbers |
| `src/unified_selector.py` | Selector API: DPP, MMR, Submodular, FarthestPoint, Clustering, Random |
| `src/cli/divflow.py` | CLI: `metrics`, `select`, `benchmark`, `cross-model`, `sensitivity` |

## Tests

```bash
python3 -m pytest tests/ -q  # 77 tests
```

## Requirements

- Python 3.9+
- numpy, scipy
- z3-solver (optional, for SMT verification)
- transformers, torch (optional, for real model experiments)
