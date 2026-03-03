# The Diversity Decoding Arena: A Metric Taxonomy and Systematic Comparison of Diversity-Promoting Text Generation

## Problem Statement

Diverse text generation—producing a set of outputs that are collectively varied while individually high-quality—underpins creative writing assistants, multi-candidate code generation, brainstorming agents, and any system where monoculture in outputs is a failure mode. Over a dozen decoding algorithms now claim to promote diversity: nucleus sampling, typical decoding, contrastive search, diverse beam search, DPP reranking, MBR-based diversity methods, and others. Yet each is evaluated in isolation, on idiosyncratic tasks, with hand-picked metrics, against incomplete baselines. While individual papers compare against several alternatives (e.g., Jinnai et al. 2024 compare 5–6 methods; Su et al. 2022 compare against multiple baselines), no published work provides a controlled comparison across the full space of algorithmically distinct decoding families under unified conditions. The result is a fragmented literature where practitioners cannot answer the most basic question: *which diversity-promoting decoding algorithm should I use for my task, and why?*

The problem runs deeper than missing baselines. The field lacks consensus on what "diversity" even means operationally. Self-BLEU, distinct-n, embedding pairwise distance, Vendi Score, parse-tree diversity, n-gram entropy, type-token ratio—these metrics are used interchangeably, yet no systematic study has established whether they measure the same construct, which are redundant, and which capture genuinely orthogonal diversity dimensions. Without this, every paper's headline result is contingent on its metric choice. A method that "wins" on Self-BLEU may lose on embedding coverage, and no one knows whether that disagreement reflects a real tradeoff or a measurement artifact.

We propose the **Diversity Decoding Arena**, with three contributions ordered by impact:

**Primary contribution: A diversity metric correlation taxonomy.** The first systematic empirical study of whether diversity metrics agree or fracture—Kendall τ across all metric pairs, conditioned on task type—answering "which diversity metrics measure the same thing?" This is a reference result that future papers will cite when justifying metric choices, analogous to how the MT community moved from BLEU to COMET based on correlation studies.

**Methodological contribution: A controlled evaluation framework.** Faithful, unified implementations of 10 algorithmically distinct decoding families, evaluated across 6 task domains on 7 diversity metrics and 3 quality metrics, analyzed through Pareto frontier analysis, Bayesian statistical comparison, and hypervolume indicators. The framework is designed for extensibility: new algorithms slot in with a single class implementation, immediately compared against the full baseline suite.

**Algorithmic contribution: Two novel decoding algorithms.** **Stein Variational Decoding (SVD)** adapts repulsive particle dynamics inspired by SVGD to autoregressive generation, maintaining "particle" sequences simultaneously attracted to high-probability regions and repelled from each other in embedding space—a kernel-repulsive heuristic that applies particle-method ideas to text generation for the first time. **Quality-Diversity Beam Search (QD-BS)** integrates MAP-Elites-style archive management into beam search using behavior descriptors over partial sequences, extending prior work on quality-diversity optimization in NLP (Steels 2025; Zammit et al. 2024) to the autoregressive beam search setting. These algorithms stress-test the arena's evaluation methodology and demonstrate that new algorithmic ideas can be rigorously validated by systematic comparison.

**Dataset contribution: A large-scale diversity generation corpus.** All ~1.4M generated sequences are released as a first-class artifact, enabling future researchers to compute new metrics on existing generations without re-running inference—an "ImageNet of diversity decoding" that provides lasting value independent of the framework code.

The entire system runs on laptop CPU (16GB+ RAM) using GPT-2 124M. This is a deliberate scientific choice: most decoding algorithms operate as transformations of logit distributions (though some, like contrastive search, additionally use hidden states), and studying them on small models enables exhaustive hyperparameter sweeps and full statistical replication that would be computationally infeasible on larger models. Every result is reproducible in under two weeks on a single consumer laptop, with no human annotation required at any stage.

## Value Proposition

**For researchers:** The metric correlation taxonomy gives principled guidance for metric selection: instead of reporting Self-BLEU because "everyone does," researchers can cite empirical evidence for which metrics carry independent information. The arena compresses weeks of reimplementation into a single `pip install`, providing validated baselines with documented hyperparameter ranges, shared inference caching, and a statistical framework that produces publication-ready comparisons. SVD and QD-BS provide new algorithmic primitives with clear methodological lineage, opening research directions in particle-based and archive-based decoding respectively.

**For practitioners:** The evaluation directly answers "which algorithm works for my task" with honest confidence intervals and failure-mode analysis. The Pareto frontier visualizations show exactly where each algorithm sits in the diversity-quality tradeoff space, broken down by task type. A practitioner's decision tree distills results into actionable guidance: "if your task is X and you care about Y, use algorithm Z." Engineering guidelines emerge naturally: if temperature scaling dominates in 70% of settings, practitioners should know that before investing in complex alternatives.

**For the field:** A standardized evaluation protocol that prevents cherry-picking of baselines, metrics, and tasks. The arena's API and evaluation harness are designed for extensibility: new algorithms slot in with a single class implementation, immediately compared against the full baseline suite. The released generation dataset enables reproducible re-analysis without re-running inference.

## Technical Difficulty

The arena's genuine complexity lives in three layers: novel algorithm research, evaluation methodology, and inference engineering. We distinguish the research core from the infrastructure that makes it a usable tool.

**Research core (~20K LoC):** This is where the intellectual difficulty concentrates.

| Subsystem | Core Code | Difficulty |
|---|---|---|
| **Stein Variational Decoding** | 2,000 | Research — no reference implementation exists for discrete-sequence SVGD. Kernel design over embeddings, score-function approximation without backprop, anisotropic bandwidth selection. |
| **Quality-Diversity Beam Search** | 1,500 | Research — partial-sequence behavior descriptors are an open problem. Archive update/pruning rules must interact correctly with beam management. |
| **Existing decoding algorithms (8 families)** | 4,000 | Engineering — individually simple (50–200 lines per algorithm), but faithful reimplementation from papers with ambiguous pseudocode requires careful validation. |
| **Diversity metrics (7)** | 2,500 | Moderate — Self-BLEU and TTR are trivial; behavioral diversity and parse-tree diversity require real implementation work. |
| **Quality metrics (3)** | 1,500 | Moderate — perplexity is trivial; NLI-based coherence on CPU requires batching and memory management. |
| **Pareto + statistical framework** | 2,500 | Moderate — non-dominated sorting, Monte Carlo hypervolume, Bayesian sign test with ROPE, bootstrap CIs, Bradley-Terry ranking. Mostly library-backed but the composition requires careful testing. |
| **Shared inference pipeline** | 3,000 | Hard engineering — LogitSource abstraction, content-addressed logit cache, quantized CPU inference at ~100 tok/s, prefix sharing. |
| **Task domains (6)** | 3,000 | Low-Moderate — dataset loading, prompt templates, answer extraction. |

**Infrastructure (~30K LoC of tests, docs, CLI, CI/CD, visualization):** This is standard software engineering that transforms a research prototype into a usable tool. It is important for a community artifact but does not represent research difficulty. We include it because for evaluation frameworks, the infrastructure determines whether the tool is actually used (as demonstrated by HELM, lm-eval-harness, and MTEB), but we do not claim it as intellectual novelty.

**Total deliverable: ~50K LoC** (20K research core + 30K infrastructure).

**Where the genuine difficulty lives:** ~3,500 LoC of research code (SVD + QD-BS) where correctness is not even well-defined because no reference implementation exists. ~6,000 LoC of methodology code (metrics, statistics, inference pipeline) where performance constraints and correctness requirements are non-trivial. The metric correlation taxonomy requires careful experimental design to ensure that observed correlations reflect genuine metric relationships rather than artifacts of task selection or generation quality.

## New Mathematics Required

Two items of genuinely novel algorithmic design directly enable the artifact. Everything else is standard machinery applied competently.

### 1. Stein Variational Decoding for Discrete Autoregressive Generation

SVGD updates continuous particles via $x_i \leftarrow x_i + \epsilon \hat{\phi}^*(x_i)$, where $\hat{\phi}^*(x) = \frac{1}{n}\sum_{j=1}^n [k(x_j, x) \nabla_{x_j} \log p(x_j) + \nabla_{x_j} k(x_j, x)]$. The first term drives particles toward high-density regions; the second repels them from each other. Adapting this to discrete token sequences requires three novel components:

**(a) Kernel over partial token sequences.** We define $k(y_{1:t}, y'_{1:t}) = \exp(-\|e(y_{1:t}) - e(y'_{1:t})\|^2 / 2h)$ where $e(\cdot)$ is a frozen sentence-transformer embedding. This lifts discrete sequences into a continuous space where kernel gradients are well-defined. The kernel operates in embedding space, not token space—a design choice that sacrifices SVGD's formal convergence guarantees but retains repulsive dynamics empirically.

**(b) Approximate score function without backpropagation.** The score $\nabla_x \log p(x)$ is undefined for discrete tokens. We approximate the "direction toward high probability" using the logit distribution at position $t+1$: the score-function proxy biases each particle's next-token selection toward tokens with high conditional probability $p(y_{t+1} | y_{1:t})$, while the repulsive term biases away from tokens that would make the sequence embedding closer to other particles. This avoids any backpropagation through the language model, which is critical for CPU feasibility.

**(c) Bandwidth selection for text embedding geometry.** The kernel bandwidth $h$ controls the repulsion radius. We adapt the median heuristic ($h = \text{median}(\|e_i - e_j\|^2) / \log n$) with an anisotropic correction that accounts for the non-uniform variance structure of sentence-transformer embedding spaces.

**Theoretical honesty:** SVD-text is a *kernel-repulsive heuristic inspired by SVGD*, not a provably convergent sampler. It differs from post-hoc DPP/MMR reranking in that repulsion is applied *during* generation, allowing particle trajectories to diverge early rather than selecting from an already-generated pool. Whether this online repulsion produces meaningfully different diversity-quality tradeoffs than post-hoc selection is an empirical question the arena is designed to answer.

### 2. Quality-Diversity Beam Search with Partial-Sequence Behavior Descriptors

Standard Quality-Diversity optimization (MAP-Elites) maintains an archive of high-performing solutions indexed by behavior descriptors, operating on *complete* solutions. Prior work has applied MAP-Elites to NLP text generation (Steels 2025; Zammit et al. 2024), but not within the autoregressive beam search loop. Integrating archive-based QD into beam search requires novel solutions for three subproblems:

**(a) Behavior descriptors on incomplete sequences.** We define $\beta(y_{1:t}) \in \mathbb{R}^d$ over partial sequences using lightweight features predictive of final-sequence behavior: POS-tag distribution (approximated via fast tagger), n-gram frequency vector, and lexical diversity trajectory. The key requirement is that $\beta(y_{1:t})$ must be cheap to compute at every beam step and approximately stable (a partial sequence's descriptor should not change radically as it is extended).

**(b) Tessellation granularity.** The archive partitions behavior space into cells via Voronoi tessellation (or grid discretization). Too fine a tessellation yields a sparse archive equivalent to standard beam search; too coarse yields one entry per cell equivalent to diverse beam search with grouping. The optimal granularity is task-dependent and must be set via the sweep.

**(c) Archive-beam interaction rules.** When a beam step produces candidates, the archive update rule must decide: keep the highest-scoring candidate per cell (MAP-Elites style), or maintain multiple candidates per cell with a diversity sub-objective? The pruning rule must be compatible with beam search's requirement for a fixed-size active set at each step.

## Evaluation Plan

**Model:** GPT-2 124M (primary, via ONNX Runtime INT8 quantization for ~100 tok/s on CPU).

**Tasks (6):** Story continuation, dialogue response, paraphrase generation, open-domain QA, instruction following, argument generation. These are tasks where GPT-2 124M produces minimally coherent output, ensuring diversity metrics measure meaningful variation rather than noise. We deliberately exclude code generation (HumanEval), math (GSM8K), and summarization (CNN/DM) because GPT-2 124M cannot perform these tasks competently—diversity of incoherent outputs is not scientifically meaningful.

**Algorithms (10 distinct families):**
- *Sampling-based (4):* Temperature scaling ($T \in \{0.7, 1.0, 1.3\}$), top-k, nucleus ($p \in \{0.9, 0.95\}$), typical ($\tau \in \{0.2, 0.95\}$).
- *Search-based (3):* Diverse beam search, stochastic beam search, contrastive search (note: uses hidden states, not just logits).
- *Reranking-based (2):* DPP reranking, MMR reranking (note: post-hoc methods operating on model embeddings).
- *Novel (2):* Stein Variational Decoding, Quality-Diversity Beam Search.
- *MBR-based (1):* Minimum Bayes Risk diversity decoding (Jinnai et al. 2024).

Each algorithm family receives 3–5 hyperparameter configurations varying the primary diversity-controlling parameter. We explicitly note which algorithms are pure logit transformations and which require additional model access (hidden states, embeddings, or multiple forward passes).

**Metrics (10):**
- *Diversity (7):* Self-BLEU (↓), distinct-n (1,2,3), embedding pairwise distance, Vendi Score, parse-tree edit distance, n-gram entropy, behavioral diversity.
- *Quality (3):* Perplexity, NLI-based coherence (via DistilBERT NLI, ~66M params, CPU-feasible), constraint satisfaction rate.

**Statistical framework:**
- Bayesian sign test with ROPE ($\epsilon = 0.01$) for pairwise algorithm comparisons.
- Bootstrap 95% confidence intervals on all metrics (10,000 resamples).
- Cliff's delta for standardized effect sizes.
- Pareto frontier computation via non-dominated sorting; Monte Carlo hypervolume indicator for aggregate frontier quality.
- Bradley-Terry ranking model for global algorithm ordering.
- Metric-metric Kendall $\tau$ correlation matrix, conditioned on task type, to produce the metric taxonomy.

**Experimental protocol:**
- Per task: 100 prompts, 20 generated sequences per algorithm-configuration-prompt triple, 3 random seeds.
- Hyperparameter sweep: 3–5 configurations per algorithm (varying the primary diversity-controlling parameter).
- Total generation budget: ~10 algorithms × 4 configs × 6 tasks × 100 prompts × 20 samples × 3 seeds ≈ 1.44M sequences (~16M tokens at ~11 tokens/sequence average).

**All evaluation is fully automated. Zero human annotation.** Quality metrics use model-based proxies (perplexity, NLI entailment scores, programmatic constraint checking). This is a deliberate design choice: human evaluation is expensive, non-reproducible, and unnecessary for the structural questions this arena addresses.

## Laptop CPU Feasibility

**Hardware requirement:** 16GB+ RAM laptop (M1/M2 MacBook Pro or equivalent). 8GB machines are not sufficient due to concurrent model loading and cache requirements.

| Component | Time Estimate | Notes |
|---|---|---|
| Generation (GPT-2 124M @ ~100 tok/s ONNX INT8) | 2 days | ~16M tokens. Content-addressed prefix caching shares computation across algorithms using the same prompt. |
| SVD generation overhead | +50–100% for SVD runs only (~4 hours) | Embedding computation + repulsive kernel updates per particle set. No backpropagation through the LM. SVD is 1 of 10 algorithm families. |
| Metric computation | 2–3 days | Embedding metrics (fast). Parse-tree diversity via dependency parsing (~1 day for 1.4M sequences). NLI coherence via DistilBERT NLI at ~200 seq/s batched (~2 hours). |
| Statistical analysis & visualization | < 1 hour | Pareto computation, bootstrap CIs, correlation matrices, plot generation. All library calls. |
| **Total** | **5–7 days** | Conservative estimate with buffer for re-runs. |

**Why this runs on CPU:** Most decoding algorithms are transformations of logit distributions, not training procedures. GPT-2 124M generates at ~100 tokens/second on modern laptop CPUs with ONNX Runtime INT8 quantization. SVD's embedding-space formulation uses a frozen sentence-transformer (~30M parameters) for kernel computation—fast on CPU, no gradient computation required. If SVD's online repulsive updates prove too slow, a pre-planned fallback generates candidates with standard sampling and applies repulsive selection post-hoc. Note: this fallback is conceptually similar to DPP/MMR reranking and would reduce SVD's novelty; the arena's evaluation will directly measure whether online repulsion provides benefits beyond post-hoc selection.

**Key engineering mitigations:**
- *Content-addressed logit cache:* Same prompt prefix → cached top-k logits, shared across all algorithms. Estimated cache size: ~6 GB on disk for the full experiment.
- *LogitSource abstraction:* Algorithms consume logit distributions, not model objects. Supports live inference, cached logits, or pre-computed logit files interchangeably.
- *Sequential processing with multiprocessing where safe:* Algorithms are independent given cached model outputs. Python multiprocessing across algorithm-task pairs where memory permits.
- *Checkpoint/resume:* State serialized every 100 prompts. Survives interruptions, allows incremental result analysis.

**Why no human evaluation:** The arena answers *structural* questions (metric correlations, Pareto dominance, algorithm-task interactions) that are inherently quantitative. Human evaluation adds noise, cost, and irreproducibility without improving the answers to these questions. This is not a limitation—it is a design principle aligned with the arena's purpose as a reproducible scientific instrument.

Slug: diversity-decoding
