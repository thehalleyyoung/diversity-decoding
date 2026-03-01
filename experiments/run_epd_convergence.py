#!/usr/bin/env python3
"""
Extended EPD sample-size convergence ablation.

Addresses the critique: "EPD semi-independence may be artifact of sample size"

Tests whether EPD (Embedding Pairwise Distance) converges to the lexical
cluster at larger sample sizes (n=20, 50, 100), which is "the most
scientifically interesting question" per the critique.

Run from implementation/ directory:
    PYTHONPATH=. python3 experiments/run_epd_convergence.py

Can run with or without GPT-2 (falls back to synthetic data).
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.diversity import (
    SelfBLEU,
    DistinctN,
    NGramEntropy,
    EmbeddingPairwiseDistance,
    VendiScore,
    tokenize_simple,
)
from src.metrics.neural_diversity import (
    BERTScoreDiversity,
    CompressionRatioDiversity,
)

RESULTS_DIR = Path(__file__).parent / "epd_convergence_results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample sizes to test
# ---------------------------------------------------------------------------

SAMPLE_SIZES = [5, 8, 10, 15, 20, 30, 50, 75, 100]

# Prompts for generation
PROMPTS = [
    "The future of artificial intelligence depends on",
    "Once upon a time in a distant land,",
    "Recent advances in quantum computing have shown that",
    "The most important challenge facing education today is",
    "To make a perfect cup of coffee, you should",
    "The relationship between dark matter and galaxy formation",
    "Social media has fundamentally changed the way we",
    "As the spaceship entered orbit around the alien world,",
    "The detective examined the crime scene carefully and noticed",
    "Climate change is rapidly transforming ecosystems around",
]

# Decoding configurations to compare
CONFIGS = {
    "temp_0.3": {"temperature": 0.3},
    "temp_0.5": {"temperature": 0.5},
    "temp_0.7": {"temperature": 0.7},
    "temp_0.9": {"temperature": 0.9},
    "temp_1.0": {"temperature": 1.0},
    "temp_1.2": {"temperature": 1.2},
    "temp_1.5": {"temperature": 1.5},
    "nucleus_0.5": {"top_p": 0.5},
    "nucleus_0.7": {"top_p": 0.7},
    "nucleus_0.9": {"top_p": 0.9},
    "nucleus_0.95": {"top_p": 0.95},
}


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_synthetic_texts(
    prompt: str, n_texts: int, config: Dict, seed: int,
) -> List[str]:
    """Generate synthetic texts with controllable diversity (no GPU needed).

    Uses word-level perturbation with temperature-like control.
    """
    rng = np.random.default_rng(seed)
    temperature = config.get("temperature", config.get("top_p", 0.7))

    base_words = prompt.split()
    vocab = [
        "the", "a", "an", "is", "was", "are", "were", "has", "have", "had",
        "will", "would", "could", "should", "can", "may", "might",
        "in", "on", "at", "by", "for", "with", "from", "to", "of",
        "and", "but", "or", "not", "no", "yes", "so", "if", "then",
        "new", "old", "big", "small", "good", "bad", "great", "important",
        "world", "time", "people", "way", "day", "life", "work", "system",
        "first", "last", "long", "high", "even", "also", "just", "many",
        "research", "technology", "science", "data", "model", "method",
        "result", "analysis", "study", "approach", "problem", "solution",
        "development", "process", "change", "future", "power", "energy",
        "space", "light", "water", "earth", "nature", "human", "mind",
        "knowledge", "learning", "understanding", "thinking", "creating",
        "building", "exploring", "discovering", "developing", "improving",
    ]

    texts = []
    for i in range(n_texts):
        text_rng = np.random.default_rng(seed + i * 1000)
        words = list(base_words)
        n_extra = text_rng.integers(15, 40)
        for _ in range(n_extra):
            if text_rng.random() < temperature:
                # Higher temperature = more random word choice
                words.append(vocab[text_rng.integers(len(vocab))])
            else:
                # Lower temperature = repeat from prompt or recent words
                if words:
                    words.append(words[text_rng.integers(max(1, len(words) - 5), len(words))])
                else:
                    words.append(vocab[0])
        texts.append(" ".join(words) + ".")

    return texts


def try_gpt2_generation(
    prompt: str, n_texts: int, config: Dict, seed: int,
) -> List[str]:
    """Try to generate with GPT-2, fall back to synthetic."""
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompt_ids = tokenizer.encode(prompt)
        texts = []
        temperature = config.get("temperature", 1.0)
        top_p = config.get("top_p", None)

        for i in range(n_texts):
            rng = np.random.default_rng(seed + i)
            seq = list(prompt_ids)
            for _ in range(50):
                tensor = torch.tensor([seq], dtype=torch.long)
                with torch.no_grad():
                    out = model(input_ids=tensor)
                logits = out.logits[0, -1, :].numpy()
                shifted = logits / max(temperature, 1e-8) - np.max(logits / max(temperature, 1e-8))
                probs = np.exp(shifted) / np.exp(shifted).sum()

                if top_p is not None:
                    sorted_idx = np.argsort(-probs)
                    cumsum = np.cumsum(probs[sorted_idx])
                    cutoff = np.searchsorted(cumsum, top_p) + 1
                    top_idx = sorted_idx[:cutoff]
                    top_probs = probs[top_idx]
                    top_probs /= top_probs.sum()
                    tok = int(rng.choice(top_idx, p=top_probs))
                else:
                    tok = int(rng.choice(len(probs), p=probs))
                seq.append(tok)
                if tok == tokenizer.eos_token_id:
                    break
            texts.append(tokenizer.decode(seq, skip_special_tokens=True))
        return texts

    except Exception as e:
        logger.info("GPT-2 not available (%s), using synthetic generation", e)
        return generate_synthetic_texts(prompt, n_texts, config, seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute core metrics for EPD convergence analysis."""
    metrics = {}
    instances = [
        ("distinct_2", DistinctN(n=2)),
        ("self_bleu", SelfBLEU(max_order=4)),
        ("ngram_entropy", NGramEntropy(n=2)),
        ("epd", EmbeddingPairwiseDistance()),
        ("vendi_score", VendiScore()),
        ("bertscore_div", BERTScoreDiversity(backend="tfidf")),
        ("crd", CompressionRatioDiversity()),
    ]
    for name, metric in instances:
        try:
            metrics[name] = metric.compute(texts)
        except Exception:
            metrics[name] = float("nan")

    # TTR
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize_simple(t))
    metrics["ttr"] = len(set(all_tokens)) / max(len(all_tokens), 1)

    # Jaccard
    token_sets = [set(tokenize_simple(t)) for t in texts]
    dists = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = len(token_sets[i] | token_sets[j])
            inter = len(token_sets[i] & token_sets[j])
            dists.append(1.0 - inter / max(union, 1))
    metrics["jaccard"] = np.mean(dists) if dists else 0.0

    return metrics


def kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = x[i] - x[j], y[i] - y[j]
            if xi * yi > 0:
                c += 1
            elif xi * yi < 0:
                d += 1
    return (c - d) / (c + d) if (c + d) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_epd_convergence():
    """Run the EPD sample-size convergence experiment."""
    logger.info("=" * 60)
    logger.info("EPD SAMPLE-SIZE CONVERGENCE ABLATION")
    logger.info("=" * 60)
    logger.info("Sample sizes: %s", SAMPLE_SIZES)
    logger.info("Prompts: %d, Configs: %d", len(PROMPTS), len(CONFIGS))

    max_n = max(SAMPLE_SIZES)

    # Phase 1: Generate maximum-size sets
    logger.info("\nPhase 1: Generating texts (max n=%d)...", max_n)
    all_texts: Dict[str, Dict[str, List[str]]] = {}

    for config_name, config in CONFIGS.items():
        all_texts[config_name] = {}
        for pi, prompt in enumerate(PROMPTS):
            texts = try_gpt2_generation(
                prompt, max_n, config, seed=SEED + pi * 100
            )
            all_texts[config_name][prompt[:40]] = texts
            logger.info("  Generated %s / prompt %d: %d texts", config_name, pi, len(texts))

    # Phase 2: Compute metrics at each sample size
    logger.info("\nPhase 2: Computing metrics at each sample size...")
    results_by_n: Dict[int, Dict[str, Any]] = {}

    for n in SAMPLE_SIZES:
        logger.info("\n  --- n = %d ---", n)
        per_config_metrics: Dict[str, Dict[str, List[float]]] = {}

        for config_name, prompts_texts in all_texts.items():
            per_config_metrics[config_name] = {}
            for prompt_key, texts in prompts_texts.items():
                subset = texts[:n]
                if len(subset) < 2:
                    continue
                m = compute_metrics(subset)
                for mn, mv in m.items():
                    if mn not in per_config_metrics[config_name]:
                        per_config_metrics[config_name][mn] = []
                    per_config_metrics[config_name][mn].append(mv)

        # Aggregate: per config, average across prompts
        config_averages: Dict[str, Dict[str, float]] = {}
        for config_name in per_config_metrics:
            config_averages[config_name] = {}
            for mn in per_config_metrics[config_name]:
                vals = [
                    v for v in per_config_metrics[config_name][mn]
                    if not math.isnan(v)
                ]
                config_averages[config_name][mn] = (
                    float(np.mean(vals)) if vals else float("nan")
                )

        # Compute τ matrix at this sample size
        metric_names = sorted(
            set(mn for ca in config_averages.values() for mn in ca)
        )
        config_names_sorted = sorted(config_averages.keys())

        tau_matrix = {}
        for m1 in metric_names:
            tau_matrix[m1] = {}
            for m2 in metric_names:
                v1 = [
                    config_averages[c].get(m1, float("nan"))
                    for c in config_names_sorted
                ]
                v2 = [
                    config_averages[c].get(m2, float("nan"))
                    for c in config_names_sorted
                ]
                valid = [
                    (a, b) for a, b in zip(v1, v2)
                    if not math.isnan(a) and not math.isnan(b)
                ]
                if len(valid) >= 3:
                    vx, vy = zip(*valid)
                    tau_matrix[m1][m2] = kendall_tau(list(vx), list(vy))
                else:
                    tau_matrix[m1][m2] = 0.0

        # Key question: τ(EPD, distinct_2) at each n
        epd_d2_tau = tau_matrix.get("epd", {}).get("distinct_2", 0.0)
        epd_sb_tau = tau_matrix.get("epd", {}).get("self_bleu", 0.0)
        epd_vendi_tau = tau_matrix.get("epd", {}).get("vendi_score", 0.0)

        results_by_n[n] = {
            "tau_matrix": tau_matrix,
            "config_averages": config_averages,
            "key_correlations": {
                "epd_vs_distinct2": epd_d2_tau,
                "epd_vs_self_bleu": epd_sb_tau,
                "epd_vs_vendi": epd_vendi_tau,
            },
        }

        logger.info("  τ(EPD, D-2) = %.3f", epd_d2_tau)
        logger.info("  τ(EPD, SB)  = %.3f", epd_sb_tau)
        logger.info("  τ(EPD, VS)  = %.3f", epd_vendi_tau)

    # Phase 3: Convergence analysis
    logger.info("\n" + "=" * 60)
    logger.info("CONVERGENCE ANALYSIS")
    logger.info("=" * 60)

    convergence = {}
    for n in SAMPLE_SIZES:
        kc = results_by_n[n]["key_correlations"]
        convergence[n] = kc
        logger.info(
            "n=%3d: τ(EPD,D2)=%.3f  τ(EPD,SB)=%.3f  τ(EPD,VS)=%.3f",
            n, kc["epd_vs_distinct2"], kc["epd_vs_self_bleu"], kc["epd_vs_vendi"],
        )

    # Does EPD converge to lexical cluster?
    epd_d2_at_small = convergence[SAMPLE_SIZES[0]]["epd_vs_distinct2"]
    epd_d2_at_large = convergence[SAMPLE_SIZES[-1]]["epd_vs_distinct2"]
    logger.info(
        "\nτ(EPD, D-2) at n=%d: %.3f → n=%d: %.3f (Δ = %.3f)",
        SAMPLE_SIZES[0], epd_d2_at_small,
        SAMPLE_SIZES[-1], epd_d2_at_large,
        epd_d2_at_large - epd_d2_at_small,
    )

    if abs(epd_d2_at_large) > 0.8:
        logger.info("FINDING: EPD converges to lexical cluster at large n")
    elif abs(epd_d2_at_large) < 0.5:
        logger.info("FINDING: EPD remains semi-independent even at large n")
    else:
        logger.info("FINDING: EPD shows moderate convergence — inconclusive")

    # Save results
    output = {
        "sample_sizes": SAMPLE_SIZES,
        "n_prompts": len(PROMPTS),
        "n_configs": len(CONFIGS),
        "convergence_summary": convergence,
        "per_n_results": {str(k): v for k, v in results_by_n.items()},
    }
    out_path = RESULTS_DIR / "epd_convergence_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)

    return output


if __name__ == "__main__":
    run_epd_convergence()
