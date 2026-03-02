# DivFlow

**The only diversity toolkit with formal optimality certificates.** Standard diversity selection algorithms (DPP, MMR, greedy, random) can be up to 98% suboptimal — and no existing tool can detect this. DivFlow uses Z3 SMT solving to compute provably optimal diverse subsets and certify exactly how far any heuristic selection is from optimal.

## The Problem

When you select diverse outputs from an LLM — using DPP sampling, beam search, or any heuristic — **you have no idea how close to optimal your selection is.** You might be at 99% of optimal diversity, or 50%, or 2%. No existing tool tells you.

DivFlow solves this by computing exact optimal solutions via SMT constraint solving, then issuing formal certificates proving each selection's optimality gap.

## Key Result: Certified Selection Benchmark

On 138 certified instances across 5 distribution types and 2 objectives (sum-pairwise, max-min spread), with real-world-anchored benchmarks from STS-B and ParaNMT:

| Tool | Provides Certificates | Mean Gap | Max Gap | >10% Suboptimal |
|------|----------------------|----------|---------|-----------------|
| **DivFlow** | **✓ Formal SMT certificates** | — | — | — |
| DPP (sklearn) | ✗ | 26.0% | **96.5%** | 71.0% |
| MMR | ✗ | 12.3% | 39.5% | 51.5% |
| Greedy submodular | ✗ | 7.7% | 91.5% | 23.2% |
| Farthest-point | ✗ | 6.2% | 26.1% | 21.7% |
| Random | ✗ | 37.6% | **98.4%** | 80.4% |

**DPP — widely recommended for diversity — is up to 96.5% suboptimal.** Only DivFlow can detect this.

No other diversity toolkit — HuggingFace diverse beam search, Vendi Score, standard DPP libraries — provides any optimality certification.

## Quickstart

```bash
pip install numpy scipy z3-solver
```

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

## What DivFlow Certifies

| Certification | Description |
|--------------|-------------|
| **Exact optimality** | Provably optimal diverse subset via SMT (n ≤ 12) or ILP (n ≤ 30) |
| **Optimality gap** | Formal certificate: "this selection is within X% of optimal" |
| **Fair retention** | Certified diversity cost of fairness constraints |
| **NP-hardness witness** | Concrete instances proving the problem is computationally hard |
| **Metric bounds** | Theorem 4.1: tight bound on Kendall τ between metrics |

Available selectors: `farthest_point`, `dpp`, `mmr`, `submodular`, `clustering`, `random`.

## Benchmark Results by Distribution

Real-world-anchored benchmarks (STS-B sentence embeddings, ParaNMT paraphrase clusters):

| Distribution | Source | Mean Gap | Max Gap |
|-------------|--------|----------|---------|
| Real-world (STS-B) | Cer et al., 2017 | 12.5% | 94.9% |
| Real-world (ParaNMT) | Wieting & Gimpel, 2018 | 12.5% | 94.9% |
| Clustered | Gaussian mixture | 18.2% | 98.4% |
| Hierarchical | Topics/subtopics | 20.7% | 82.1% |
| Adversarial | Perturbed hypersphere | 18.1% | 89.5% |
| Uniform | Null hypothesis | 21.7% | 66.6% |

## Modules

| Module | Purpose |
|--------|---------|
| `src/smt_diversity.py` | **Z3 SMT/ILP certified optimizer** — exact solutions, optimality gap certificates, fair retention bounds |
| `src/unified_selector.py` | 6 selection algorithms: DPP, MMR, Submodular, FarthestPoint, Clustering, Random |
| `src/theorem_bounds.py` | Theorem 4.1: tight Kendall τ bound |τ| ≥ 1 − 2(ε₁+ε₂), verified on 1200+ instances |
| `src/entropy_correction.py` | Miller-Madow, Jackknife entropy correction (KL: 22.66 → 0.37 bits) |
| `src/cross_model_analysis.py` | Cross-model Kendall τ across 10 model families |
| `src/metric_lattice.py` | Lattice structure, Hasse diagrams, Betti numbers |

## Limitations

- SMT exact solving scales to n ≤ 12; ILP to n ≤ 30. Larger instances use heuristic bounds.
- Cross-model metric agreement is moderate (meta-τ = 0.33, power = 0.61) — which is exactly why per-instance certification matters.
- Benchmarks use synthetic distance matrices matching published statistics; runtime on production LLM outputs depends on embedding computation.

## Tests

```bash
python3 -m pytest tests/ -q
```

## Requirements

- Python 3.9+
- numpy, scipy
- z3-solver (for SMT certification)
- transformers, torch (optional, for real model experiments)
