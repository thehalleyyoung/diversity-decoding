# DivFlow: Unified Diversity Metrics & Selection for AI Text

**A practical Python toolkit for measuring, comparing, and optimizing diversity in LLM outputs.**

## 30-Second Quickstart

```bash
pip install numpy scipy
python3 -m pytest tests/ -q  # 96 passed in ~10s
```

```python
from src.unified_selector import get_selector
import numpy as np

# Select 10 diverse items from 200 candidates
embeddings = np.random.randn(200, 50)
selector = get_selector('farthest_point')
indices, metadata = selector.select(embeddings, k=10)
print(f"Selected {len(indices)} diverse items")
```

### CLI Usage

```bash
# Compute diversity metrics on texts
python3 -m src.cli.divflow metrics --texts "Hello world" "Goodbye" "Hi there"

# Benchmark selection algorithms
python3 -m src.cli.divflow benchmark --n 200 --k 10

# Cross-model analysis
python3 -m src.cli.divflow cross-model

# Embedding kernel sensitivity analysis
python3 -m src.cli.divflow sensitivity
```

### Bias-Corrected Analysis

```python
from src.entropy_correction import corrected_info_theory_analysis
results = corrected_info_theory_analysis(high_div_texts, low_div_texts)
print(f"H_MM = {results['shannon_entropy']['high_diversity']['miller_madow']}")
# H_MM = 9.8146
```

## Key Results

| Finding | Value | Evidence |
|---------|-------|----------|
| Metric redundancy | 10/11 metrics at \|τ\| > 0.7 | `comprehensive_v2_results.json` |
| Tight bound (Thm 4.1) | \|τ\| ≥ 1 − 2(ε₁+ε₂), **200/200 verified** | `theorem_bounds.py` |
| Sample complexity | P_min = ⌈ln(2/α)/(2δ²)⌉ pairs | `theorem_bounds.py` |
| FP vs Random (spread) | d = 7.42, p < 10⁻³⁵ | `comprehensive_v2_results.json` |
| Fair retention | 97.9% [97.5%, 98.2%] | `comprehensive_v2_results.json` |
| **SMT certified retention** | **≥ 96.8% (exact)** | `smt_results.json` |
| **Cross-model τ (10 models)** | **0.33 [0.32, 0.34], 45 pairs** | `cross_model_results.json` |
| **GPT-2 scale τ (3 variants)** | **0.74 [0.43, 0.90]** | `gpt2_cross_model_results.json` |
| **SVD reframing** | **Kernel-repulsive heuristic, NOT variational inference** | `svd.py` |
| **Embedding sensitivity** | **Within-kernel stable, cross-kernel τ = 0.32** | `embedding_sensitivity.py` |
| Tests passing | **96/96** (66 PATH B + 12 dist. + 18 prop.) | `pytest` |

## What's New (PATH B)

1. **SVD honest reframing**: Stein Variational Decoding reframed as kernel-repulsive heuristic with explicit documentation of what it is NOT (no target distribution, no convergence guarantee, KSD is heuristic)
2. **Theorem 4.1 with quantitative bounds**: Full proof with sample complexity (Hoeffding), concentration inequalities, degradation rate (-2/unit ε), verified 200/200
3. **Cross-model GPT-2 variants**: Within-architecture τ = 0.74 confirms taxonomy transfers across model scale
4. **Embedding kernel sensitivity**: 5 kernels × 4 bandwidths analysis showing within-kernel stability
5. **Simple `divflow` CLI**: `metrics`, `select`, `benchmark`, `cross-model`, `sensitivity` subcommands

## Contents

- `src/` — Full DivFlow toolkit
  - `src/unified_selector.py` — Unified selector API (DPP, MMR, Submodular, FarthestPoint, Clustering)
  - `src/algorithms/svd.py` — Stein Variational Decoding (kernel-repulsive heuristic)
  - `src/embedding_sensitivity.py` — Kernel sensitivity analysis (NEW)
  - `src/theorem_bounds.py` — Quantitative bounds for Theorem 4.1 (NEW)
  - `src/cross_model_analysis.py` — Cross-model analysis for 10 model families
  - `src/cli/divflow.py` — Simple CLI interface (NEW)
  - `src/entropy_correction.py` — Bias-corrected entropy estimators
  - `src/smt_diversity.py` — SMT/ILP exact optimization (Z3)
  - `src/metric_lattice.py` — Lattice structure, Hasse diagrams
  - `src/failure_taxonomy.py` — 13-mode failure taxonomy
- `experiments/` — Reproducible experiment scripts
  - `experiments/run_cross_model_gpt2.py` — GPT-2 scale variant experiment (NEW)
- `tests/` — 96 tests
- `theory/tool_paper.tex` — Paper (27 pages, compiles with `pdflatex`)
- `docs/meta/grounding.json` — Claim-to-artifact traceability

## Running

```bash
# Run all tests (96 tests, ~10s)
python3 -m pytest tests/test_property_invariants.py tests/test_distributional.py tests/test_pathb.py -v

# Run GPT-2 cross-model experiment (uses synthetic data by default)
DIVFLOW_SYNTHETIC=1 PYTHONPATH=. python3 experiments/run_cross_model_gpt2.py

# Compile paper
cd theory && pdflatex tool_paper.tex && pdflatex tool_paper.tex
```

## Requirements

- Python 3.9+
- numpy, scipy
- z3-solver (for SMT exact optimization)
- transformers, torch (optional, for real GPT-2 experiments)
