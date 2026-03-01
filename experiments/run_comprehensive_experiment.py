#!/usr/bin/env python3
"""
Comprehensive diversity metric experiment: scaled-up GPT-2 analysis.

Addresses critiques:
  - Underpowered experiment: 20+ prompts × 50+ sequences per config
  - Only 2 models: adds GPT-2 medium and GPT-2 large
  - Missing decoding strategies: contrastive, quality-diversity beam search
  - All metrics computed via src modules (no inline reimplementations)

Run from implementation/ directory:
    PYTHONPATH=. python3 experiments/run_comprehensive_experiment.py

Requires: transformers, torch
"""

import json
import logging
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.diversity import (
    SelfBLEU,
    DistinctN,
    NGramEntropy,
    EmbeddingPairwiseDistance,
    VendiScore,
    ParseTreeDiversity,
    BehavioralDiversity,
    tokenize_simple,
)
from src.metrics.neural_diversity import (
    MAUVE,
    BERTScoreDiversity,
    STSDiversity,
    CompressionRatioDiversity,
)
from src.metrics.correlation import MetricCorrelationAnalyzer

RESULTS_DIR = Path(__file__).parent / "comprehensive_results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts: 25 diverse prompts spanning 5 domains
# ---------------------------------------------------------------------------

PROMPTS = {
    "story": [
        "Once upon a time in a distant kingdom,",
        "The detective examined the crime scene carefully and noticed",
        "As the spaceship entered orbit around the alien world,",
        "The old woman sat by the fire and began to tell",
        "In the year 2150, humanity had finally discovered",
    ],
    "science": [
        "Recent advances in quantum computing have shown that",
        "The relationship between dark matter and galaxy formation suggests",
        "New research on CRISPR gene editing reveals",
        "Climate models predict that by 2100",
        "The discovery of gravitational waves confirmed",
    ],
    "opinion": [
        "The most important challenge facing education today is",
        "Social media has fundamentally changed the way we",
        "The future of artificial intelligence depends on",
        "Economic inequality can best be addressed by",
        "The role of government in regulating technology should",
    ],
    "instruction": [
        "To make a perfect cup of coffee, you should",
        "The steps to build a successful startup include",
        "Here is how to train a neural network from scratch:",
        "To improve your writing skills, consider the following",
        "The best way to learn a new programming language is",
    ],
    "dialogue": [
        '"I never thought I would see this day," said the captain. "',
        "The interviewer asked about future plans. The candidate replied,",
        '"Can you explain why this experiment failed?" The scientist answered,',
        '"What do you think about the new policy?" The senator responded,',
        '"Tell me about your childhood," the therapist said. The patient began,',
    ],
}

ALL_PROMPTS = [(domain, prompt) for domain in PROMPTS for prompt in PROMPTS[domain]]


# ---------------------------------------------------------------------------
# GPT-2 text generation with multiple models and strategies
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


class GPT2Generator:
    """GPT-2 text generator supporting multiple model sizes."""

    def __init__(self, model_name: str = "gpt2"):
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.eval()
            self.vocab_size = self.model.config.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self._torch = torch
            self._available = True
            logger.info("Loaded %s (vocab=%d)", model_name, self.vocab_size)
        except Exception as e:
            logger.warning("Could not load %s: %s", model_name, e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def get_logits(self, input_ids: List[int]) -> np.ndarray:
        """Get next-token logits for a single sequence."""
        tensor = self._torch.tensor([input_ids], dtype=self._torch.long)
        with self._torch.no_grad():
            out = self.model(input_ids=tensor)
        return out.logits[0, -1, :].numpy()

    def generate_temperature(
        self, prompt: str, n_seqs: int, max_tokens: int, temperature: float,
        seed: int = SEED,
    ) -> List[str]:
        prompt_ids = self.tokenizer.encode(prompt)
        results = []
        for i in range(n_seqs):
            rng = np.random.default_rng(seed + i)
            seq = list(prompt_ids)
            for _ in range(max_tokens):
                logits = self.get_logits(seq)
                probs = _softmax(logits / max(temperature, 1e-8))
                tok = int(rng.choice(len(probs), p=probs))
                seq.append(tok)
                if tok == self.tokenizer.eos_token_id:
                    break
            results.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return results

    def generate_nucleus(
        self, prompt: str, n_seqs: int, max_tokens: int, top_p: float,
        seed: int = SEED,
    ) -> List[str]:
        prompt_ids = self.tokenizer.encode(prompt)
        results = []
        for i in range(n_seqs):
            rng = np.random.default_rng(seed + i)
            seq = list(prompt_ids)
            for _ in range(max_tokens):
                logits = self.get_logits(seq)
                probs = _softmax(logits)
                sorted_idx = np.argsort(-probs)
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumsum, top_p) + 1
                top_idx = sorted_idx[:cutoff]
                top_probs = probs[top_idx]
                top_probs /= top_probs.sum()
                tok = int(rng.choice(top_idx, p=top_probs))
                seq.append(tok)
                if tok == self.tokenizer.eos_token_id:
                    break
            results.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return results

    def generate_typical(
        self, prompt: str, n_seqs: int, max_tokens: int, mass: float = 0.9,
        seed: int = SEED,
    ) -> List[str]:
        """Typical sampling (Meister et al. 2023)."""
        prompt_ids = self.tokenizer.encode(prompt)
        results = []
        for i in range(n_seqs):
            rng = np.random.default_rng(seed + i)
            seq = list(prompt_ids)
            for _ in range(max_tokens):
                logits = self.get_logits(seq)
                probs = _softmax(logits)
                log_probs = np.log(probs + 1e-30)
                entropy = -np.sum(probs * log_probs)
                deviations = np.abs(-log_probs - entropy)
                sorted_idx = np.argsort(deviations)
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumsum, mass) + 1
                top_idx = sorted_idx[:cutoff]
                top_probs = probs[top_idx]
                top_probs /= top_probs.sum()
                tok = int(rng.choice(top_idx, p=top_probs))
                seq.append(tok)
                if tok == self.tokenizer.eos_token_id:
                    break
            results.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return results

    def generate_contrastive(
        self, prompt: str, n_seqs: int, max_tokens: int,
        alpha: float = 0.6, k: int = 10, seed: int = SEED,
    ) -> List[str]:
        """Contrastive search decoding (Su et al. 2022)."""
        prompt_ids = self.tokenizer.encode(prompt)
        results = []
        for i in range(n_seqs):
            rng = np.random.default_rng(seed + i)
            seq = list(prompt_ids)
            for step in range(max_tokens):
                logits = self.get_logits(seq)
                probs = _softmax(logits)
                top_k_idx = np.argsort(-probs)[:k]
                if step > 0:
                    recent = seq[-min(5, len(seq)):]
                    penalties = np.array([
                        0.3 if idx in recent else 0.0 for idx in top_k_idx
                    ])
                    scores = (1 - alpha) * probs[top_k_idx] - alpha * penalties
                    scores = np.maximum(scores, 1e-10)
                else:
                    scores = probs[top_k_idx]
                scores /= scores.sum()
                tok = int(rng.choice(top_k_idx, p=scores))
                seq.append(tok)
                if tok == self.tokenizer.eos_token_id:
                    break
            results.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return results

    def generate_qdbs(
        self, prompt: str, n_groups: int = 4, beam_width: int = 3,
        max_tokens: int = 40, diversity_penalty: float = 0.5,
        quality_weight: float = 0.7,
    ) -> List[str]:
        """Quality-Diversity Beam Search: diverse beam search with quality weighting."""
        prompt_ids = self.tokenizer.encode(prompt)
        groups = [[(list(prompt_ids), 0.0)] for _ in range(n_groups)]
        for step in range(max_tokens):
            for g in range(n_groups):
                new_beams = []
                for seq, score in groups[g]:
                    logits = self.get_logits(seq)
                    log_probs = logits - np.log(np.exp(logits).sum() + 1e-30)
                    # Diversity penalty from other groups
                    for og in range(g):
                        for oseq, _ in groups[og]:
                            if len(oseq) > len(prompt_ids) + step:
                                tok = oseq[len(prompt_ids) + step]
                                log_probs[tok] -= diversity_penalty
                    # Quality-diversity trade-off
                    top_k = np.argsort(-log_probs)[: beam_width * 2]
                    for t in top_k[:beam_width]:
                        q_score = quality_weight * float(log_probs[t])
                        d_bonus = (1 - quality_weight) * diversity_penalty * g
                        new_beams.append((seq + [int(t)], score + q_score + d_bonus))
                new_beams.sort(key=lambda x: -x[1])
                groups[g] = new_beams[:beam_width]
        results = []
        for g in groups:
            for seq, _ in g:
                results.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return results


# ---------------------------------------------------------------------------
# Decoding configurations
# ---------------------------------------------------------------------------

DECODING_CONFIGS = {
    # Temperature sampling at multiple values
    "temp_0.3": {"method": "temperature", "temperature": 0.3},
    "temp_0.5": {"method": "temperature", "temperature": 0.5},
    "temp_0.7": {"method": "temperature", "temperature": 0.7},
    "temp_0.9": {"method": "temperature", "temperature": 0.9},
    "temp_1.0": {"method": "temperature", "temperature": 1.0},
    "temp_1.2": {"method": "temperature", "temperature": 1.2},
    "temp_1.5": {"method": "temperature", "temperature": 1.5},
    # Nucleus sampling at multiple p values
    "nucleus_0.5": {"method": "nucleus", "top_p": 0.5},
    "nucleus_0.7": {"method": "nucleus", "top_p": 0.7},
    "nucleus_0.8": {"method": "nucleus", "top_p": 0.8},
    "nucleus_0.9": {"method": "nucleus", "top_p": 0.9},
    "nucleus_0.95": {"method": "nucleus", "top_p": 0.95},
    # Typical sampling
    "typical_0.8": {"method": "typical", "mass": 0.8},
    "typical_0.9": {"method": "typical", "mass": 0.9},
    "typical_0.95": {"method": "typical", "mass": 0.95},
    # Contrastive search
    "contrastive_0.4": {"method": "contrastive", "alpha": 0.4, "k": 10},
    "contrastive_0.6": {"method": "contrastive", "alpha": 0.6, "k": 10},
    "contrastive_0.8": {"method": "contrastive", "alpha": 0.8, "k": 15},
    # Quality-Diversity Beam Search
    "qdbs_low": {"method": "qdbs", "diversity_penalty": 0.3, "quality_weight": 0.8},
    "qdbs_high": {"method": "qdbs", "diversity_penalty": 0.8, "quality_weight": 0.5},
}


# ---------------------------------------------------------------------------
# Metric computation using src modules
# ---------------------------------------------------------------------------

def compute_all_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute all 14 diversity metrics using the src implementations."""
    if len(texts) < 2:
        return {}

    metrics = {}
    metric_instances = [
        ("distinct_2", DistinctN(n=2)),
        ("self_bleu", SelfBLEU(max_order=4)),
        ("ngram_entropy", NGramEntropy(n=2)),
        ("embedding_distance", EmbeddingPairwiseDistance()),
        ("vendi_score", VendiScore()),
        ("parse_tree_diversity", ParseTreeDiversity()),
        ("behavioral_diversity", BehavioralDiversity()),
        ("mauve", MAUVE(n_clusters=min(50, len(texts) // 2), backend="tfidf")),
        ("bertscore_diversity", BERTScoreDiversity(backend="tfidf")),
        ("sts_diversity", STSDiversity(backend="tfidf")),
        ("crd", CompressionRatioDiversity()),
    ]

    for name, metric in metric_instances:
        try:
            metrics[name] = metric.compute(texts)
        except Exception as e:
            logger.warning("Failed to compute %s: %s", name, e)
            metrics[name] = float("nan")

    # Additional inline metrics that don't have class form
    try:
        all_tokens = []
        for t in texts:
            all_tokens.extend(tokenize_simple(t))
        if all_tokens:
            metrics["ttr"] = len(set(all_tokens)) / len(all_tokens)
        else:
            metrics["ttr"] = 0.0
    except Exception:
        metrics["ttr"] = float("nan")

    # Pairwise Jaccard
    try:
        token_sets = [set(tokenize_simple(t)) for t in texts]
        dists = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                union = len(token_sets[i] | token_sets[j])
                inter = len(token_sets[i] & token_sets[j])
                dists.append(1.0 - inter / max(union, 1))
        metrics["jaccard"] = sum(dists) / len(dists) if dists else 0.0
    except Exception:
        metrics["jaccard"] = float("nan")

    # Unique count
    metrics["n_unique"] = len(set(texts))

    return metrics


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Kendall τ rank correlation."""
    n = len(x)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = x[i] - x[j]
            yi = y[i] - y[j]
            if xi * yi > 0:
                concordant += 1
            elif xi * yi < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def bootstrap_kendall_ci(
    x: List[float], y: List[float], n_boot: int = 1000, seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (tau, ci_lo, ci_hi) via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    taus = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        t = kendall_tau([x[i] for i in idx], [y[i] for i in idx])
        taus.append(t)
    tau = kendall_tau(x, y)
    lo = float(np.percentile(taus, 2.5))
    hi = float(np.percentile(taus, 97.5))
    return tau, lo, hi


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_generation(
    generator: GPT2Generator,
    prompts: List[Tuple[str, str]],
    configs: Dict[str, Dict],
    n_seqs: int = 50,
    max_tokens: int = 50,
    seed: int = SEED,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate texts for all prompt × config combinations.

    Returns: {config_name: {prompt: [texts]}}
    """
    all_generations = {}
    total = len(configs) * len(prompts)
    done = 0

    for config_name, config in configs.items():
        all_generations[config_name] = {}
        method = config["method"]

        for domain, prompt in prompts:
            key = f"{domain}:{prompt[:40]}"
            try:
                if method == "temperature":
                    texts = generator.generate_temperature(
                        prompt, n_seqs, max_tokens,
                        temperature=config["temperature"], seed=seed,
                    )
                elif method == "nucleus":
                    texts = generator.generate_nucleus(
                        prompt, n_seqs, max_tokens,
                        top_p=config["top_p"], seed=seed,
                    )
                elif method == "typical":
                    texts = generator.generate_typical(
                        prompt, n_seqs, max_tokens,
                        mass=config["mass"], seed=seed,
                    )
                elif method == "contrastive":
                    texts = generator.generate_contrastive(
                        prompt, n_seqs, max_tokens,
                        alpha=config["alpha"], k=config["k"], seed=seed,
                    )
                elif method == "qdbs":
                    texts = generator.generate_qdbs(
                        prompt,
                        diversity_penalty=config["diversity_penalty"],
                        quality_weight=config["quality_weight"],
                        max_tokens=max_tokens,
                    )
                else:
                    texts = []

                all_generations[config_name][key] = texts
            except Exception as e:
                logger.error("Generation failed %s/%s: %s", config_name, key, e)
                all_generations[config_name][key] = []

            done += 1
            if done % 10 == 0:
                logger.info("Generation progress: %d/%d", done, total)

    return all_generations


def run_metric_analysis(
    generations: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Compute all metrics and correlation analysis."""
    results = {
        "per_config_per_prompt": {},
        "per_config_aggregated": {},
        "tau_matrix": {},
        "per_domain": {},
    }

    metric_names = None
    config_metric_vectors: Dict[str, Dict[str, List[float]]] = {}

    for config_name, prompt_texts in generations.items():
        config_results = {}
        for prompt_key, texts in prompt_texts.items():
            if len(texts) < 2:
                continue
            m = compute_all_metrics(texts)
            config_results[prompt_key] = m
            if metric_names is None:
                metric_names = sorted(m.keys())

        results["per_config_per_prompt"][config_name] = config_results

        # Aggregate per config
        if config_results:
            agg = {}
            for mn in metric_names:
                vals = [
                    cr[mn] for cr in config_results.values()
                    if mn in cr and not math.isnan(cr.get(mn, float("nan")))
                ]
                agg[mn] = float(np.mean(vals)) if vals else float("nan")
            results["per_config_aggregated"][config_name] = agg

            for mn in metric_names:
                if mn not in config_metric_vectors:
                    config_metric_vectors[mn] = {}
                config_metric_vectors[mn][config_name] = agg.get(mn, float("nan"))

    # Compute Kendall τ matrix
    if metric_names and config_metric_vectors:
        config_names_sorted = sorted(
            results["per_config_aggregated"].keys()
        )
        tau_matrix = {}
        for m1 in metric_names:
            tau_matrix[m1] = {}
            for m2 in metric_names:
                v1 = [
                    config_metric_vectors[m1].get(c, float("nan"))
                    for c in config_names_sorted
                ]
                v2 = [
                    config_metric_vectors[m2].get(c, float("nan"))
                    for c in config_names_sorted
                ]
                # Filter NaN pairs
                valid = [
                    (a, b) for a, b in zip(v1, v2)
                    if not math.isnan(a) and not math.isnan(b)
                ]
                if len(valid) >= 3:
                    vx, vy = zip(*valid)
                    tau, lo, hi = bootstrap_kendall_ci(list(vx), list(vy))
                    tau_matrix[m1][m2] = {
                        "tau": tau, "ci_lo": lo, "ci_hi": hi,
                    }
                else:
                    tau_matrix[m1][m2] = {"tau": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
        results["tau_matrix"] = tau_matrix

    # Per-domain analysis
    domains = set()
    for prompt_texts in generations.values():
        for key in prompt_texts:
            domains.add(key.split(":")[0])

    for domain in domains:
        domain_tau = {}
        domain_vectors: Dict[str, List[float]] = {mn: [] for mn in metric_names}
        for config_name in sorted(generations.keys()):
            for key, texts in generations[config_name].items():
                if key.split(":")[0] == domain and len(texts) >= 2:
                    m = compute_all_metrics(texts)
                    for mn in metric_names:
                        domain_vectors[mn].append(m.get(mn, float("nan")))
        for m1 in metric_names:
            domain_tau[m1] = {}
            for m2 in metric_names:
                v1 = domain_vectors[m1]
                v2 = domain_vectors[m2]
                valid = [
                    (a, b) for a, b in zip(v1, v2)
                    if not math.isnan(a) and not math.isnan(b)
                ]
                if len(valid) >= 3:
                    vx, vy = zip(*valid)
                    domain_tau[m1][m2] = kendall_tau(list(vx), list(vy))
                else:
                    domain_tau[m1][m2] = 0.0
        results["per_domain"][domain] = domain_tau

    return results


def main():
    """Run comprehensive experiment."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE DIVERSITY METRIC ANALYSIS")
    logger.info("=" * 60)

    models_to_try = ["gpt2", "gpt2-medium", "gpt2-large"]
    all_results = {}

    for model_name in models_to_try:
        logger.info("\n--- Model: %s ---", model_name)
        generator = GPT2Generator(model_name)

        if not generator.available:
            logger.warning("Skipping %s (not available)", model_name)
            continue

        logger.info(
            "Generating: %d prompts × %d configs × 50 sequences",
            len(ALL_PROMPTS), len(DECODING_CONFIGS),
        )

        generations = run_generation(
            generator, ALL_PROMPTS, DECODING_CONFIGS,
            n_seqs=50, max_tokens=50, seed=SEED,
        )

        logger.info("Computing metrics...")
        analysis = run_metric_analysis(generations)

        all_results[model_name] = {
            "analysis": analysis,
            "config": {
                "n_prompts": len(ALL_PROMPTS),
                "n_configs": len(DECODING_CONFIGS),
                "n_seqs_per_config": 50,
                "max_tokens": 50,
                "configs": DECODING_CONFIGS,
            },
        }

        # Save per-model results
        out_path = RESULTS_DIR / f"{model_name.replace('-', '_')}_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results[model_name], f, indent=2, default=str)
        logger.info("Saved %s", out_path)

    # Save combined results
    combined_path = RESULTS_DIR / "comprehensive_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved combined results to %s", combined_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for model_name, data in all_results.items():
        analysis = data["analysis"]
        logger.info("\nModel: %s", model_name)
        if "tau_matrix" in analysis and analysis["tau_matrix"]:
            metrics = sorted(analysis["tau_matrix"].keys())
            logger.info("Metric correlation matrix (Kendall τ):")
            header = f"{'':>20s}" + "".join(f"{m:>10s}" for m in metrics)
            logger.info(header)
            for m1 in metrics:
                row = f"{m1:>20s}"
                for m2 in metrics:
                    tau = analysis["tau_matrix"][m1][m2].get("tau", 0.0)
                    row += f"{tau:>10.3f}"
                logger.info(row)


if __name__ == "__main__":
    main()
