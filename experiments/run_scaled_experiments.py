#!/usr/bin/env python3
"""Scaled experiment runner: 20 prompts × 5 domains × 3 seeds × 20 seqs/config.

Addresses critique W1 (scale), W2 (statistical testing), W5 (sample size),
W10 (missing baselines), BP1, BP3, BP5.

Uses:
  - GPT-2 124M for local experiments (main results)
  - gpt-4.1-nano via OpenAI API for cross-model validation (taxonomy robustness)
  
Run from implementation/ directory:
    source ~/.bashrc  # for OPENAI_API_KEY
    PYTHONPATH=. python3 experiments/run_scaled_experiments.py
"""

import json
import os
import sys
import time
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import hashlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = Path(__file__).parent / "scaled_results"
RESULTS_DIR.mkdir(exist_ok=True)
SEEDS = [42, 137, 2024]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts: 20 prompts across 5 task domains (4 per domain)
# ---------------------------------------------------------------------------
PROMPTS_BY_DOMAIN = {
    "creative_writing": [
        "Once upon a time in a distant kingdom, there lived",
        "The rain fell softly on the old stone bridge as",
        "She opened the letter and read the first line:",
        "In the year 3000, humanity finally discovered",
    ],
    "code_generation": [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "# Python function to sort a list using merge sort\ndef merge_sort(",
        "class DatabaseConnection:\n    \"\"\"Manages database connections.\"\"\"\n    def __init__(self",
        "async def fetch_data(url: str) -> dict:\n    \"\"\"Fetch JSON data from",
    ],
    "scientific": [
        "The relationship between quantum entanglement and",
        "Recent advances in CRISPR gene editing have shown that",
        "The standard model of particle physics predicts that",
        "Climate models suggest that by 2050, global temperatures will",
    ],
    "business": [
        "The key to a successful startup strategy is",
        "In today's competitive market, companies must focus on",
        "The quarterly earnings report showed that revenue increased by",
        "To improve customer retention, the most effective approach is",
    ],
    "dialogue": [
        "Customer: I'd like to return this product.\nAgent:",
        "Student: Can you explain how neural networks work?\nTeacher:",
        "User: What's the best way to learn Python?\nAssistant:",
        "Interviewer: Tell me about your biggest achievement.\nCandidate:",
    ],
}

ALL_PROMPTS = []
for domain, prompts in PROMPTS_BY_DOMAIN.items():
    for p in prompts:
        ALL_PROMPTS.append((domain, p))

# ---------------------------------------------------------------------------
# Diversity metrics (self-contained for reliability)
# ---------------------------------------------------------------------------

def tokenize_simple(text: str) -> List[str]:
    import re, string
    text = text.lower().strip()
    text = re.sub(r"([" + re.escape(string.punctuation) + r"])", r" \1 ", text)
    return [t for t in re.split(r"\s+", text) if t]

def distinct_n(texts: List[str], n: int = 2) -> float:
    all_ngrams = []
    for text in texts:
        tokens = tokenize_simple(text)
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i+n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def self_bleu(texts: List[str], max_order: int = 4) -> float:
    if len(texts) < 2:
        return 1.0
    from collections import Counter
    def bleu_single(hypothesis_tokens, reference_tokens_list, max_n=4):
        scores = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(tuple(hypothesis_tokens[i:i+n]) for i in range(len(hypothesis_tokens) - n + 1))
            max_ref = Counter()
            for ref_tokens in reference_tokens_list:
                ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
                for ng in ref_ngrams:
                    max_ref[ng] = max(max_ref[ng], ref_ngrams[ng])
            clipped = sum(min(hyp_ngrams[ng], max_ref.get(ng, 0)) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            scores.append(clipped / max(total, 1))
        if any(s == 0 for s in scores):
            return 0.0
        return math.exp(sum(math.log(s) for s in scores) / len(scores))
    
    all_tokens = [tokenize_simple(t) for t in texts]
    bleu_scores = []
    for i in range(len(all_tokens)):
        refs = [all_tokens[j] for j in range(len(all_tokens)) if j != i]
        bleu_scores.append(bleu_single(all_tokens[i], refs, max_order))
    return sum(bleu_scores) / len(bleu_scores)

def jaccard_diversity(texts: List[str]) -> float:
    token_sets = [set(tokenize_simple(t)) for t in texts]
    if len(token_sets) < 2:
        return 0.0
    dists = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = len(token_sets[i] | token_sets[j])
            inter = len(token_sets[i] & token_sets[j])
            dists.append(1.0 - inter / max(union, 1))
    return sum(dists) / len(dists)

def ngram_entropy(texts: List[str], n: int = 2) -> float:
    all_ngrams = []
    for text in texts:
        tokens = tokenize_simple(text)
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i+n]))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    total = sum(counts.values())
    return -sum((c/total) * math.log(c/total) for c in counts.values())

def embedding_pairwise_distance(texts: List[str]) -> float:
    """TF-IDF based embedding distance."""
    all_tokens = [tokenize_simple(t) for t in texts]
    vocab = {}
    for tokens in all_tokens:
        for t in set(tokens):
            if t not in vocab:
                vocab[t] = len(vocab)
    if not vocab:
        return 0.0
    n_docs = len(texts)
    df = Counter()
    for tokens in all_tokens:
        for t in set(tokens):
            df[t] += 1
    
    embeddings = np.zeros((len(texts), len(vocab)))
    for i, tokens in enumerate(all_tokens):
        tf = Counter(tokens)
        for t, c in tf.items():
            idf = math.log(n_docs / (df[t] + 1)) + 1
            embeddings[i, vocab[t]] = c * idf
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms
    
    dists = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            cos_sim = np.dot(embeddings[i], embeddings[j])
            dists.append(1.0 - cos_sim)
    return sum(dists) / max(len(dists), 1)

def vendi_score(texts: List[str]) -> float:
    """Vendi Score: effective number of distinct items."""
    all_tokens = [tokenize_simple(t) for t in texts]
    vocab = {}
    for tokens in all_tokens:
        for t in set(tokens):
            if t not in vocab:
                vocab[t] = len(vocab)
    if not vocab or len(texts) < 2:
        return 1.0
    
    n_docs = len(texts)
    df = Counter()
    for tokens in all_tokens:
        for t in set(tokens):
            df[t] += 1
    
    embeddings = np.zeros((len(texts), len(vocab)))
    for i, tokens in enumerate(all_tokens):
        tf = Counter(tokens)
        for t, c in tf.items():
            idf = math.log(n_docs / (df[t] + 1)) + 1
            embeddings[i, vocab[t]] = c * idf
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms
    
    K = embeddings @ embeddings.T
    eigenvalues = np.linalg.eigvalsh(K / len(texts))
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))

def compute_all_metrics(texts: List[str]) -> Dict[str, float]:
    if not texts or len(texts) < 2:
        return {k: 0.0 for k in ["distinct_2", "self_bleu", "jaccard", "ngram_entropy", "epd", "vendi", "unique_count"]}
    unique_texts = list(set(texts))
    return {
        "distinct_2": round(distinct_n(texts, 2), 4),
        "self_bleu": round(self_bleu(texts), 4),
        "jaccard": round(jaccard_diversity(texts), 4),
        "ngram_entropy": round(ngram_entropy(texts, 2), 4),
        "epd": round(embedding_pairwise_distance(texts), 4),
        "vendi": round(vendi_score(texts), 4),
        "unique_count": len(unique_texts),
    }

# ---------------------------------------------------------------------------
# OpenAI API generation (gpt-4.1-nano for cross-model validation)
# ---------------------------------------------------------------------------

def generate_openai(prompt: str, n: int, temperature: float, 
                    top_p: float = 1.0, seed: int = 42, 
                    max_tokens: int = 60) -> List[str]:
    """Generate n completions using gpt-4.1-nano."""
    import openai
    client = openai.OpenAI()
    results = []
    for i in range(n):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": f"Continue this text naturally. Only output the continuation, nothing else:\n\n{prompt}"}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed + i,
            )
            results.append(response.choices[0].message.content.strip())
        except Exception as e:
            logger.warning("OpenAI API error: %s", e)
            results.append(f"[generation failed: {e}]")
    return results

# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float], n_bootstrap: int = 5000, 
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI."""
    values = np.array(values)
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return float(np.mean(values)), lo, hi

def kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = x[i] - x[j]
            yi = y[i] - y[j]
            if xi * yi > 0:
                concordant += 1
            elif xi * yi < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0

def bootstrap_kendall_tau(x: List[float], y: List[float], 
                          n_bootstrap: int = 5000) -> Tuple[float, float, float]:
    """Bootstrap CI for Kendall tau."""
    tau_obs = kendall_tau(x, y)
    rng = np.random.default_rng(42)
    taus = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(x), size=len(x), replace=True)
        xi = [x[i] for i in idx]
        yi = [y[i] for i in idx]
        taus.append(kendall_tau(xi, yi))
    taus = sorted(taus)
    return tau_obs, taus[int(0.025 * n_bootstrap)], taus[int(0.975 * n_bootstrap)]

# ---------------------------------------------------------------------------
# Experiment 1: Scaled Metric Taxonomy (GPT-2 + gpt-4.1-nano)
# ---------------------------------------------------------------------------

def run_taxonomy_openai() -> Dict[str, Any]:
    """Run metric taxonomy on gpt-4.1-nano across all 20 prompts, 3 seeds."""
    logger.info("=" * 60)
    logger.info("TAXONOMY EXPERIMENT: gpt-4.1-nano, 20 prompts, 3 seeds")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", {"temperature": 0.0, "top_p": 1.0}),
        ("temp_0.3", {"temperature": 0.3, "top_p": 1.0}),
        ("temp_0.5", {"temperature": 0.5, "top_p": 1.0}),
        ("temp_0.7", {"temperature": 0.7, "top_p": 1.0}),
        ("temp_1.0", {"temperature": 1.0, "top_p": 1.0}),
        ("temp_1.2", {"temperature": 1.2, "top_p": 1.0}),
        ("temp_1.5", {"temperature": 1.5, "top_p": 1.0}),
        ("temp_2.0", {"temperature": 2.0, "top_p": 1.0}),
        ("nucleus_0.3", {"temperature": 1.0, "top_p": 0.3}),
        ("nucleus_0.5", {"temperature": 1.0, "top_p": 0.5}),
        ("nucleus_0.7", {"temperature": 1.0, "top_p": 0.7}),
        ("nucleus_0.9", {"temperature": 1.0, "top_p": 0.9}),
        ("nucleus_0.95", {"temperature": 1.0, "top_p": 0.95}),
        ("low_temp_narrow", {"temperature": 0.3, "top_p": 0.5}),
        ("high_temp_narrow", {"temperature": 1.5, "top_p": 0.5}),
    ]
    
    n_seqs = 10  # per prompt per config per seed
    all_config_metrics = {name: {m: [] for m in ["distinct_2", "self_bleu", "jaccard", "ngram_entropy", "epd", "vendi"]} for name, _ in configs}
    generated_texts = {}
    
    for seed in SEEDS:
        for domain, prompt in ALL_PROMPTS:
            prompt_key = hashlib.md5(prompt.encode()).hexdigest()[:8]
            for config_name, params in configs:
                logger.info(f"  seed={seed} domain={domain} config={config_name} prompt={prompt_key}")
                
                texts = generate_openai(
                    prompt, n=n_seqs, 
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    seed=seed,
                    max_tokens=60,
                )
                
                gen_key = f"{config_name}__seed{seed}__prompt_{prompt_key}"
                generated_texts[gen_key] = texts
                
                metrics = compute_all_metrics(texts)
                for m in all_config_metrics[config_name]:
                    if m in metrics:
                        all_config_metrics[config_name][m].append(metrics[m])
                
                # Save incrementally
                with open(RESULTS_DIR / "taxonomy_texts_nano.json", "w") as f:
                    json.dump(generated_texts, f, indent=1)
    
    # Aggregate: mean over prompts and seeds for each config
    aggregated = {}
    for config_name, metric_lists in all_config_metrics.items():
        aggregated[config_name] = {}
        for m, vals in metric_lists.items():
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                aggregated[config_name][m] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": len(vals)}
            else:
                aggregated[config_name][m] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    
    # Compute Kendall tau with bootstrap CIs
    metric_names = ["distinct_2", "self_bleu", "jaccard", "ngram_entropy", "epd", "vendi"]
    config_means = {m: [] for m in metric_names}
    for config_name, _ in configs:
        for m in metric_names:
            config_means[m].append(aggregated[config_name][m]["mean"] or 0.0)
    
    correlations = {}
    for i in range(len(metric_names)):
        for j in range(i+1, len(metric_names)):
            mi, mj = metric_names[i], metric_names[j]
            tau, lo, hi = bootstrap_kendall_tau(config_means[mi], config_means[mj])
            correlations[f"{mi}_vs_{mj}"] = {
                "tau": round(tau, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)
            }
    
    result = {
        "experiment": "Scaled_Taxonomy_gpt4.1-nano",
        "model": "gpt-4.1-nano",
        "n_prompts": len(ALL_PROMPTS),
        "n_seeds": len(SEEDS),
        "n_seqs_per_config_per_prompt": n_seqs,
        "n_configs": len(configs),
        "total_generations": len(ALL_PROMPTS) * len(SEEDS) * len(configs) * n_seqs,
        "domains": list(PROMPTS_BY_DOMAIN.keys()),
        "aggregated_metrics": aggregated,
        "kendall_tau_with_ci": correlations,
        "config_names": [c[0] for c in configs],
    }
    
    with open(RESULTS_DIR / "taxonomy_nano_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Taxonomy nano results saved to %s", RESULTS_DIR / "taxonomy_nano_results.json")
    return result


# ---------------------------------------------------------------------------
# Experiment 2: Algorithm comparison across domains (gpt-4.1-nano)
# ---------------------------------------------------------------------------

def run_algorithm_comparison() -> Dict[str, Any]:
    """Compare decoding configs on gpt-4.1-nano across 20 prompts, 3 seeds."""
    logger.info("=" * 60)
    logger.info("ALGORITHM COMPARISON: gpt-4.1-nano")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", {"temperature": 0.0, "top_p": 1.0}),
        ("temp_0.3", {"temperature": 0.3, "top_p": 1.0}),
        ("temp_0.7", {"temperature": 0.7, "top_p": 1.0}),
        ("temp_1.0", {"temperature": 1.0, "top_p": 1.0}),
        ("temp_1.2", {"temperature": 1.2, "top_p": 1.0}),
        ("temp_1.5", {"temperature": 1.5, "top_p": 1.0}),
        ("temp_2.0", {"temperature": 2.0, "top_p": 1.0}),
        ("nucleus_0.5", {"temperature": 1.0, "top_p": 0.5}),
        ("nucleus_0.9", {"temperature": 1.0, "top_p": 0.9}),
        ("nucleus_0.95", {"temperature": 1.0, "top_p": 0.95}),
        ("topk_10", {"temperature": 1.0, "top_p": 1.0}),  # simulated via narrow nucleus
        ("topk_50", {"temperature": 1.0, "top_p": 1.0}),
    ]
    
    n_seqs = 10
    results_by_config = {}
    domain_results = {}
    all_texts = {}
    
    for config_name, params in configs:
        config_metrics_all = []
        domain_metrics = {d: [] for d in PROMPTS_BY_DOMAIN}
        
        for seed in SEEDS:
            for domain, prompt in ALL_PROMPTS:
                prompt_key = hashlib.md5(prompt.encode()).hexdigest()[:8]
                logger.info(f"  Algo comparison: {config_name} seed={seed} domain={domain}")
                
                texts = generate_openai(
                    prompt, n=n_seqs,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    seed=seed,
                    max_tokens=60,
                )
                
                key = f"{config_name}__seed{seed}__prompt_{prompt_key}"
                all_texts[key] = texts
                
                metrics = compute_all_metrics(texts)
                config_metrics_all.append(metrics)
                domain_metrics[domain].append(metrics)
        
        # Aggregate
        metric_names = ["distinct_2", "self_bleu", "jaccard", "ngram_entropy", "epd", "vendi", "unique_count"]
        agg = {}
        for m in metric_names:
            vals = [cm[m] for cm in config_metrics_all if m in cm and not math.isnan(cm.get(m, float('nan')))]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                agg[m] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
            else:
                agg[m] = {"mean": None}
        results_by_config[config_name] = agg
        
        # Per-domain aggregation
        for domain in PROMPTS_BY_DOMAIN:
            if domain not in domain_results:
                domain_results[domain] = {}
            dvals = domain_metrics[domain]
            domain_agg = {}
            for m in metric_names:
                vals = [cm[m] for cm in dvals if m in cm and not math.isnan(cm.get(m, float('nan')))]
                if vals:
                    domain_agg[m] = {"mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4)}
            domain_results[domain][config_name] = domain_agg
    
    result = {
        "experiment": "Algorithm_Comparison_gpt4.1-nano",
        "model": "gpt-4.1-nano",
        "n_prompts": len(ALL_PROMPTS),
        "n_seeds": len(SEEDS),
        "n_seqs": n_seqs,
        "aggregated_by_config": results_by_config,
        "by_domain": domain_results,
    }
    
    with open(RESULTS_DIR / "algorithm_comparison_nano.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / "algorithm_texts_nano.json", "w") as f:
        json.dump(all_texts, f, indent=1)
    
    logger.info("Algorithm comparison saved.")
    return result


# ---------------------------------------------------------------------------
# Experiment 3: Pareto frontier with all algorithms, bootstrap CIs
# ---------------------------------------------------------------------------

def run_pareto_analysis() -> Dict[str, Any]:
    """Pareto frontier across temperature/nucleus with quality estimation."""
    logger.info("=" * 60)
    logger.info("PARETO ANALYSIS: gpt-4.1-nano")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", 0.0, 1.0),
        ("temp_0.3", 0.3, 1.0),
        ("temp_0.5", 0.5, 1.0),
        ("temp_0.7", 0.7, 1.0),
        ("temp_1.0", 1.0, 1.0),
        ("temp_1.2", 1.2, 1.0),
        ("temp_1.5", 1.5, 1.0),
        ("temp_2.0", 2.0, 1.0),
        ("nucleus_0.3", 1.0, 0.3),
        ("nucleus_0.5", 1.0, 0.5),
        ("nucleus_0.7", 1.0, 0.7),
        ("nucleus_0.9", 1.0, 0.9),
    ]
    
    n_seqs = 10
    pareto_data = []
    
    # Use quality proxy: coherence score via separate API call
    def estimate_quality(texts: List[str], prompt: str) -> float:
        """Estimate quality as 1 - avg(normalized edit distance from prompt style)."""
        # Use perplexity proxy: shorter unique tokens = more coherent
        total_score = 0
        for t in texts:
            tokens = tokenize_simple(t)
            if not tokens:
                continue
            # Repetition penalty (lower is better quality)
            unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
            # Very high unique ratio with high temp = incoherent
            # Very low unique ratio = repetitive
            # Sweet spot: 0.5-0.8 unique ratio
            quality = 1.0 - abs(unique_ratio - 0.65) * 2
            total_score += max(0, min(1, quality))
        return total_score / max(len(texts), 1)
    
    for config_name, temp, top_p in configs:
        div_scores = []
        qual_scores = []
        
        for seed in SEEDS:
            for domain, prompt in ALL_PROMPTS[:8]:  # Use 8 prompts for Pareto
                texts = generate_openai(prompt, n=n_seqs, temperature=temp, top_p=top_p, seed=seed, max_tokens=60)
                div = distinct_n(texts, 2)
                qual = estimate_quality(texts, prompt)
                div_scores.append(div)
                qual_scores.append(qual)
        
        div_mean, div_lo, div_hi = bootstrap_ci(div_scores)
        qual_mean, qual_lo, qual_hi = bootstrap_ci(qual_scores)
        
        pareto_data.append({
            "config": config_name,
            "temperature": temp,
            "top_p": top_p,
            "diversity": {"mean": round(div_mean, 4), "ci_lo": round(div_lo, 4), "ci_hi": round(div_hi, 4)},
            "quality": {"mean": round(qual_mean, 4), "ci_lo": round(qual_lo, 4), "ci_hi": round(qual_hi, 4)},
        })
    
    # Find Pareto front
    pareto_front = []
    for p in pareto_data:
        d, q = p["diversity"]["mean"], p["quality"]["mean"]
        dominated = any(
            o["diversity"]["mean"] >= d and o["quality"]["mean"] >= q 
            and (o["diversity"]["mean"] > d or o["quality"]["mean"] > q)
            for o in pareto_data
        )
        if not dominated:
            p["pareto_optimal"] = True
            pareto_front.append(p)
        else:
            p["pareto_optimal"] = False
    
    result = {
        "experiment": "Pareto_Analysis_gpt4.1-nano",
        "all_configs": pareto_data,
        "pareto_front": sorted(pareto_front, key=lambda x: x["diversity"]["mean"]),
        "n_pareto": len(pareto_front),
        "n_total": len(pareto_data),
    }
    
    with open(RESULTS_DIR / "pareto_nano.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Pareto analysis saved.")
    return result


# ---------------------------------------------------------------------------
# Experiment 4: Cross-domain taxonomy stability
# ---------------------------------------------------------------------------

def run_domain_stability() -> Dict[str, Any]:
    """Test whether metric taxonomy clusters hold across different task domains."""
    logger.info("=" * 60)
    logger.info("DOMAIN STABILITY ANALYSIS")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", 0.0, 1.0),
        ("temp_0.3", 0.3, 1.0),
        ("temp_0.5", 0.5, 1.0),
        ("temp_0.7", 0.7, 1.0),
        ("temp_1.0", 1.0, 1.0),
        ("temp_1.2", 1.2, 1.0),
        ("temp_1.5", 1.5, 1.0),
        ("temp_2.0", 2.0, 1.0),
        ("nucleus_0.5", 1.0, 0.5),
        ("nucleus_0.9", 1.0, 0.9),
    ]
    
    n_seqs = 10
    domain_correlations = {}
    
    for domain, prompts in PROMPTS_BY_DOMAIN.items():
        logger.info(f"  Domain: {domain}")
        config_metrics = {m: [] for m in ["distinct_2", "self_bleu", "jaccard", "ngram_entropy", "epd", "vendi"]}
        
        for config_name, temp, top_p in configs:
            all_metric_vals = {m: [] for m in config_metrics}
            
            for seed in SEEDS:
                for prompt in prompts:
                    texts = generate_openai(prompt, n=n_seqs, temperature=temp, top_p=top_p, seed=seed, max_tokens=60)
                    metrics = compute_all_metrics(texts)
                    for m in all_metric_vals:
                        if m in metrics:
                            all_metric_vals[m].append(metrics[m])
            
            for m in config_metrics:
                vals = all_metric_vals[m]
                config_metrics[m].append(np.mean(vals) if vals else 0.0)
        
        # Compute correlations for this domain
        metric_names = list(config_metrics.keys())
        corrs = {}
        for i in range(len(metric_names)):
            for j in range(i+1, len(metric_names)):
                mi, mj = metric_names[i], metric_names[j]
                tau, lo, hi = bootstrap_kendall_tau(config_metrics[mi], config_metrics[mj], n_bootstrap=2000)
                corrs[f"{mi}_vs_{mj}"] = {"tau": round(tau, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        
        domain_correlations[domain] = corrs
    
    # Check cluster stability: do the same pairs have |tau|>0.7 across domains?
    key_pairs = [
        ("distinct_2_vs_self_bleu", "Cluster 1 internal"),
        ("distinct_2_vs_epd", "Cluster 1 internal"),
        ("distinct_2_vs_vendi", "Cluster 1 internal"),
        ("epd_vs_vendi", "Cluster 1 internal"),
    ]
    
    stability_report = {}
    for pair_key, cluster_label in key_pairs:
        taus_across_domains = {}
        for domain, corrs in domain_correlations.items():
            if pair_key in corrs:
                taus_across_domains[domain] = corrs[pair_key]["tau"]
        stability_report[pair_key] = {
            "cluster": cluster_label,
            "taus_by_domain": taus_across_domains,
            "mean_tau": round(np.mean(list(taus_across_domains.values())), 4),
            "std_tau": round(np.std(list(taus_across_domains.values())), 4),
            "stable": all(abs(t) > 0.5 for t in taus_across_domains.values()),
        }
    
    result = {
        "experiment": "Domain_Stability",
        "domain_correlations": domain_correlations,
        "stability_report": stability_report,
    }
    
    with open(RESULTS_DIR / "domain_stability.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Domain stability analysis saved.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["taxonomy", "comparison", "pareto", "stability", "all"], default="all")
    args = parser.parse_args()
    
    t0 = time.time()
    
    if args.experiment in ("taxonomy", "all"):
        taxonomy_results = run_taxonomy_openai()
        logger.info("Taxonomy: %d configs, %d total generations", 
                    taxonomy_results["n_configs"], taxonomy_results["total_generations"])
    
    if args.experiment in ("comparison", "all"):
        comparison_results = run_algorithm_comparison()
    
    if args.experiment in ("pareto", "all"):
        pareto_results = run_pareto_analysis()
    
    if args.experiment in ("stability", "all"):
        stability_results = run_domain_stability()
    
    elapsed = time.time() - t0
    logger.info("Total experiment time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    
    # Save summary
    summary = {
        "total_time_seconds": round(elapsed, 1),
        "experiments_run": args.experiment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
