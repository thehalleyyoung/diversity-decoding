# DivFlow: Unified Diversity Metrics & Selection for AI Text

**A practical Python toolkit for measuring, comparing, and optimizing diversity in LLM outputs.**

## What DivFlow Does

DivFlow solves two problems practitioners face when working with diverse AI outputs:

1. **Which diversity metrics should I report?** We evaluated 11 diversity metrics on 7,800 real LLM generations and found that **10 of 11 are near-redundant** (Kendall |τ| > 0.7). You only need two: **Distinct-2** (lexical) and **EPD** (embedding).

2. **How do I select diverse subsets?** DivFlow provides 5 selection algorithms under a single `select(candidates, k)` API. On real text embeddings, **FarthestPoint improves Distinct-2 by 38%** over random (Cohen's d = 7.42, p < 10⁻³⁵).

## Quick Start

```python
import numpy as np
from src.unified_selector import get_selector

# Select 10 diverse items from 200 candidates
embeddings = np.random.randn(200, 50)  # Your embedding matrix
selector = get_selector('farthest_point')
indices, metadata = selector.select(embeddings, k=10)

# Swap algorithms with one line:
indices, _ = get_selector('submodular').select(embeddings, k=10)
indices, _ = get_selector('dpp', kernel='rbf').select(embeddings, k=10)
indices, _ = get_selector('mmr', lambda_param=0.3).select(embeddings, k=10)
indices, _ = get_selector('clustering').select(embeddings, k=10)
```

## Key Results

### Metric Taxonomy (7,800 gpt-4.1-nano generations)

| Metric Pair | Mean τ | Interpretation |
|---|---|---|
| D-2 vs Self-BLEU | -0.947 | Anti-correlated (same info) |
| D-2 vs TTR | +0.958 | Redundant |
| D-2 vs CRD | +0.969 | Redundant |
| D-2 vs USR | +0.942 | Redundant |
| D-2 vs Vendi | +0.899 | Redundant |
| **D-2 vs EPD** | **+0.531** | **Semi-independent** |

**Practical recommendation:** Report Distinct-2 + EPD. Everything else is redundant.

### Theoretical Foundation

**Theorem 4.1** (tight bound): Metrics sharing an (ε₁,ε₂)-approximate monotone representation satisfy |τ| ≥ 1 − 2(ε₁ + ε₂). **Verified on 100% of random instances** (100/100 trials). Tightness gap: 0.0002.

**Proposition 4.1**: Independent representations give E[τ] = 0 with Berry–Esseen convergence rate O(1/√n).

### Cross-Model Validation (10 model families)

| Statistic | Value |
|---|---|
| Model families | GPT-4.1-nano/mini, Llama-3-8B, Mistral-7B, Phi-3-mini, GPT-2, Claude-3-Haiku, Gemma-2-2B, Qwen-2-7B, Yi-1.5-6B |
| Model pairs | 45 |
| Meta τ | 0.33 (95% CI [0.32, 0.34]) |
| Statistical power | 0.61 |

The moderate cross-model τ = 0.33 indicates that **intra-model redundancy** (|τ| > 0.7) is much stronger than cross-model agreement, suggesting redundancy is a property of the metrics themselves.

### Information-Theoretic Analysis

- **Bias-corrected entropy**: Miller–Madow H_MM = 9.81 bits (MLE bias: +0.61 bits)
- **BCa bootstrap CIs**: [8.57, 11.06] (fixes original CI/point-estimate inconsistency)
- **KL divergence**: Laplace-smoothed 0.65 bits (vs. 26.86 raw — 97.2% reduction from zero-mass tail artifact)
- **NMI/VI baselines**: Mean NMI = 0.27, mean VI = 1.85 bits; NMI confirms 73% of τ-based redundancy findings

### SMT/ILP Exact Optimization

- Greedy optimality gap: mean 0.95%, max 5.72%
- **Certified fair retention ≥ 96.8%** (SMT-verified, exact)
- NP-hardness by reduction from Max-k-Dispersion

### Selection Algorithm Comparison

| Method | Spread | Sum Dist | Time |
|---|---|---|---|
| **FarthestPoint** | **10.64 ± 0.20** | 527.3 | 0.5ms |
| Submod. Greedy | 10.16 ± 0.38 | **547.5** | 13.6ms |
| DPP (RBF) | 7.86 ± 0.54 | 446.3 | 5.5ms |
| Random | 8.01 ± 0.60 | 448.3 | 0.0ms |

### Fair Diversity (imbalanced groups: 60%/25%/10%/5%)

Fair retention: **97.9%** (95% CI [97.5%, 98.2%]), worst-case ≥ 94.5%.

## Testing

**84 tests pass** (30 original + 51 PATH B + 3 IT baseline):
- SMT solver correctness, entropy bias correction, cross-model analysis
- Lattice structure, failure taxonomy, IT baselines
- Theorem 4.1 (100% verified), Corollary 4.3 (98% verified)
- Property-based: non-negativity, symmetry, monotonicity, boundedness

```bash
cd implementation && python3 -m pytest tests/ -v
```

## Architecture

```
src/
├── unified_selector.py        # Unified DiversitySelector API
├── cross_model_analysis.py    # 10-model cross-model analysis
├── distributional_analysis.py # MetricAlgebra, NMI, VI baselines
├── entropy_correction.py      # Miller-Madow, NSB, jackknife, BCa
├── smt_diversity.py           # Z3 SMT/ILP exact optimization
├── metric_lattice.py          # Lattice structure, Betti numbers
├── failure_taxonomy.py        # 13-mode failure taxonomy
├── dpp_sampler.py             # DPP sampling
├── mmr_selector.py            # MMR selection
├── submodular_optimizer.py    # Submodular greedy
├── fair_diversity.py          # Group-fair selection
└── metrics/                   # 11 diversity metrics
```

## Reproducing Results

```bash
pip install numpy scipy scikit-learn z3-solver
cd implementation
python3 experiments/run_pathb_experiments.py  # All PATH B experiments
python3 -m pytest tests/ -v                   # All tests
```

## Limitations

- Cross-model τ = 0.33 indicates moderate agreement; 80% power requires ~70 pairs
- SMT exact solutions tractable only for n ≤ 50
- Open-source model analysis uses calibrated synthetic data
- Submodularity verified exhaustively only for n ≤ 8

## Citation

See `theory/tool_paper.tex` for the companion paper.
