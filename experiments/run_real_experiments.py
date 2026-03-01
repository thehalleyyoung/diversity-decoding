#!/usr/bin/env python3
"""Real experiment runner using GPT-2 124M via HuggingFace Transformers.

Replaces the synthetic experiment runner with actual language model inference.
Run from implementation/ directory:
    PYTHONPATH=. python experiments/run_real_experiments.py
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

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = Path(__file__).parent / "real_results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPT-2 LogitSource (implements the Protocol from src.algorithms.base)
# ---------------------------------------------------------------------------

class GPT2LogitSource:
    """LogitSource backed by GPT-2 124M via HuggingFace Transformers.

    Implements the __call__ protocol: (List[List[int]]) -> np.ndarray
    returning logits of shape (batch, vocab_size).
    """

    def __init__(self, model_name: str = "gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        logger.info("Loading %s...", model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info("GPT-2 loaded. Vocab size: %d", self.vocab_size)

    def __call__(self, input_ids_batch: List[List[int]]) -> np.ndarray:
        """Return next-token logits for each sequence in the batch."""
        max_len = max(len(ids) for ids in input_ids_batch)
        padded = []
        attention_masks = []
        for ids in input_ids_batch:
            pad_len = max_len - len(ids)
            padded.append([self.tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_tensor = torch.tensor(padded, dtype=torch.long)
        attn_tensor = torch.tensor(attention_masks, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=attn_tensor)
        # Return logits at the last real token position for each sequence
        logits = outputs.logits  # (batch, seq_len, vocab)
        last_logits = []
        for i, ids in enumerate(input_ids_batch):
            pos = len(ids) - 1 + (max_len - len(ids))  # position of last real token
            last_logits.append(logits[i, pos, :].numpy())
        return np.array(last_logits)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Simple decoding strategies (operate on real GPT-2 logits)
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def _sample_from_probs(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def generate_greedy(source: GPT2LogitSource, prompt_ids: List[int],
                    n_seqs: int, max_tokens: int) -> List[List[int]]:
    """Greedy decoding - deterministic, all sequences identical."""
    seq = list(prompt_ids)
    for _ in range(max_tokens):
        logits = source([seq])[0]
        tok = int(np.argmax(logits))
        seq.append(tok)
        if tok == source.tokenizer.eos_token_id:
            break
    return [seq] * n_seqs  # greedy always returns same sequence


def generate_temperature(source: GPT2LogitSource, prompt_ids: List[int],
                         n_seqs: int, max_tokens: int, temperature: float,
                         seed: int = SEED) -> List[List[int]]:
    """Temperature sampling."""
    results = []
    for i in range(n_seqs):
        rng = np.random.default_rng(seed + i)
        seq = list(prompt_ids)
        for _ in range(max_tokens):
            logits = source([seq])[0]
            probs = _softmax(logits / temperature)
            tok = _sample_from_probs(probs, rng)
            seq.append(tok)
            if tok == source.tokenizer.eos_token_id:
                break
        results.append(seq)
    return results


def generate_nucleus(source: GPT2LogitSource, prompt_ids: List[int],
                     n_seqs: int, max_tokens: int, top_p: float,
                     seed: int = SEED) -> List[List[int]]:
    """Nucleus (top-p) sampling."""
    results = []
    for i in range(n_seqs):
        rng = np.random.default_rng(seed + i)
        seq = list(prompt_ids)
        for _ in range(max_tokens):
            logits = source([seq])[0]
            probs = _softmax(logits)
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, top_p) + 1
            top_idx = sorted_idx[:cutoff]
            top_probs = probs[top_idx]
            top_probs = top_probs / top_probs.sum()
            tok = int(rng.choice(top_idx, p=top_probs))
            seq.append(tok)
            if tok == source.tokenizer.eos_token_id:
                break
        results.append(seq)
    return results


def generate_typical(source: GPT2LogitSource, prompt_ids: List[int],
                     n_seqs: int, max_tokens: int, mass: float = 0.9,
                     seed: int = SEED) -> List[List[int]]:
    """Typical sampling (Meister et al. 2023)."""
    results = []
    for i in range(n_seqs):
        rng = np.random.default_rng(seed + i)
        seq = list(prompt_ids)
        for _ in range(max_tokens):
            logits = source([seq])[0]
            probs = _softmax(logits)
            log_probs = np.log(probs + 1e-30)
            entropy = -np.sum(probs * log_probs)
            surprisals = -log_probs
            deviations = np.abs(surprisals - entropy)
            sorted_idx = np.argsort(deviations)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, mass) + 1
            top_idx = sorted_idx[:cutoff]
            top_probs = probs[top_idx]
            top_probs = top_probs / top_probs.sum()
            tok = int(rng.choice(top_idx, p=top_probs))
            seq.append(tok)
            if tok == source.tokenizer.eos_token_id:
                break
        results.append(seq)
    return results


def generate_contrastive(source: GPT2LogitSource, prompt_ids: List[int],
                         n_seqs: int, max_tokens: int, alpha: float = 0.6,
                         k: int = 10, seed: int = SEED) -> List[List[int]]:
    """Simplified contrastive search."""
    results = []
    for i in range(n_seqs):
        rng = np.random.default_rng(seed + i)
        seq = list(prompt_ids)
        for step in range(max_tokens):
            logits = source([seq])[0]
            probs = _softmax(logits)
            top_k_idx = np.argsort(-probs)[:k]
            if step > 0 and len(seq) > 1:
                # Penalize tokens similar to recent tokens
                recent = seq[-min(5, len(seq)):]
                penalties = np.zeros(len(top_k_idx))
                for j, idx in enumerate(top_k_idx):
                    if idx in recent:
                        penalties[j] = 0.3
                scores = (1 - alpha) * probs[top_k_idx] - alpha * penalties
                scores = np.maximum(scores, 1e-10)
                scores = scores / scores.sum()
            else:
                scores = probs[top_k_idx]
                scores = scores / scores.sum()
            tok = int(rng.choice(top_k_idx, p=scores))
            seq.append(tok)
            if tok == source.tokenizer.eos_token_id:
                break
        results.append(seq)
    return results


def generate_diverse_beam(source: GPT2LogitSource, prompt_ids: List[int],
                          n_groups: int = 3, beam_width: int = 2,
                          max_tokens: int = 20, diversity_penalty: float = 0.5
                          ) -> List[List[int]]:
    """Diverse beam search with inter-group penalties."""
    groups = [[(list(prompt_ids), 0.0)] for _ in range(n_groups)]

    for step in range(max_tokens):
        for g in range(n_groups):
            new_beams = []
            for seq, score in groups[g]:
                logits = source([seq])[0]
                log_probs = logits - np.log(np.exp(logits).sum())
                # Penalize tokens chosen by earlier groups
                for og in range(g):
                    for oseq, _ in groups[og]:
                        if len(oseq) > len(prompt_ids) + step:
                            tok = oseq[len(prompt_ids) + step]
                            log_probs[tok] -= diversity_penalty
                top_k = np.argsort(-log_probs)[:beam_width * 2]
                for t in top_k[:beam_width]:
                    new_beams.append((seq + [int(t)], score + float(log_probs[t])))
            new_beams.sort(key=lambda x: -x[1])
            groups[g] = new_beams[:beam_width]

    results = []
    for g in groups:
        for seq, _ in g:
            results.append(seq)
    return results


# ---------------------------------------------------------------------------
# Diversity metrics (using the real metric implementations)
# ---------------------------------------------------------------------------

def compute_all_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute all 7 diversity metrics on a set of texts."""
    from src.metrics.diversity import (
        SelfBLEU, DistinctN, NGramEntropy,
        EmbeddingPairwiseDistance, VendiScore,
        ParseTreeDiversity, BehavioralDiversity,
    )

    metrics = {}

    try:
        sb = SelfBLEU(max_order=4)
        metrics["self_bleu"] = sb.compute(texts)
    except Exception as e:
        logger.warning("SelfBLEU failed: %s", e)
        metrics["self_bleu"] = float("nan")

    try:
        dn = DistinctN(n=2)
        metrics["distinct_2"] = dn.compute(texts)
    except Exception as e:
        logger.warning("DistinctN failed: %s", e)
        metrics["distinct_2"] = float("nan")

    try:
        ne = NGramEntropy(n=2)
        metrics["ngram_entropy"] = ne.compute(texts)
    except Exception as e:
        logger.warning("NGramEntropy failed: %s", e)
        metrics["ngram_entropy"] = float("nan")

    try:
        epd = EmbeddingPairwiseDistance()
        metrics["embedding_distance"] = epd.compute(texts)
    except Exception as e:
        logger.warning("EmbeddingPairwiseDistance failed: %s", e)
        metrics["embedding_distance"] = float("nan")

    try:
        vs = VendiScore()
        metrics["vendi_score"] = vs.compute(texts)
    except Exception as e:
        logger.warning("VendiScore failed: %s", e)
        metrics["vendi_score"] = float("nan")

    try:
        ptd = ParseTreeDiversity()
        metrics["parse_tree_diversity"] = ptd.compute(texts)
    except Exception as e:
        logger.warning("ParseTreeDiversity failed: %s", e)
        metrics["parse_tree_diversity"] = float("nan")

    try:
        bd = BehavioralDiversity()
        metrics["behavioral_diversity"] = bd.compute(texts)
    except Exception as e:
        logger.warning("BehavioralDiversity failed: %s", e)
        metrics["behavioral_diversity"] = float("nan")

    return metrics


def compute_pairwise_jaccard(texts: List[str]) -> float:
    """Average pairwise Jaccard distance between token sets."""
    from src.metrics.diversity import tokenize_simple
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


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Kendall tau rank correlation."""
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


# ---------------------------------------------------------------------------
# Correct hypervolume computation
# ---------------------------------------------------------------------------

def hypervolume_2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """Exact 2D hypervolume: Lebesgue measure of dominated region.

    For maximization: area where ref_i <= q_i <= s_i for some s in points.
    Uses sweep-line algorithm.
    """
    # Filter points that dominate the reference
    valid = [(x, y) for x, y in points if x > ref[0] and y > ref[1]]
    if not valid:
        return 0.0
    # Sort by x descending
    valid.sort(key=lambda p: -p[0])
    hv = 0.0
    prev_y = ref[1]
    for x, y in valid:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return hv


# ---------------------------------------------------------------------------
# Prompts for real experiments
# ---------------------------------------------------------------------------

PROMPTS = [
    "The future of artificial intelligence will",
    "Once upon a time in a distant kingdom,",
    "The key to a successful business strategy is",
    "In the depths of the ocean, scientists discovered",
    "The most important lesson I learned in life was",
]


# ---------------------------------------------------------------------------
# H1: Metric correlation taxonomy (all 7 metrics on real text)
# ---------------------------------------------------------------------------

def experiment_h1(source: GPT2LogitSource) -> Dict[str, Any]:
    logger.info("[H1] Metric correlation taxonomy on real GPT-2 outputs...")
    np.random.seed(SEED)

    all_metric_values: Dict[str, List[float]] = {
        "distinct_2": [], "self_bleu": [], "ngram_entropy": [],
        "embedding_distance": [], "vendi_score": [],
        "parse_tree_diversity": [], "behavioral_diversity": [],
        "jaccard_diversity": [],
    }

    configs = [
        ("temp_0.3", lambda p: generate_temperature(source, p, 8, 30, 0.3)),
        ("temp_0.5", lambda p: generate_temperature(source, p, 8, 30, 0.5)),
        ("temp_0.7", lambda p: generate_temperature(source, p, 8, 30, 0.7)),
        ("temp_1.0", lambda p: generate_temperature(source, p, 8, 30, 1.0)),
        ("temp_1.2", lambda p: generate_temperature(source, p, 8, 30, 1.2)),
        ("temp_1.5", lambda p: generate_temperature(source, p, 8, 30, 1.5)),
        ("nucleus_0.5", lambda p: generate_nucleus(source, p, 8, 30, 0.5)),
        ("nucleus_0.7", lambda p: generate_nucleus(source, p, 8, 30, 0.7)),
        ("nucleus_0.9", lambda p: generate_nucleus(source, p, 8, 30, 0.9)),
        ("nucleus_0.95", lambda p: generate_nucleus(source, p, 8, 30, 0.95)),
        ("typical_0.8", lambda p: generate_typical(source, p, 8, 30, 0.8)),
        ("typical_0.9", lambda p: generate_typical(source, p, 8, 30, 0.9)),
        ("typical_0.95", lambda p: generate_typical(source, p, 8, 30, 0.95)),
        ("greedy", lambda p: generate_greedy(source, p, 8, 30)),
        ("contrastive", lambda p: generate_contrastive(source, p, 8, 30)),
    ]

    generated_texts_all = {}  # for saving

    for config_name, gen_fn in configs:
        logger.info("  Config: %s", config_name)
        # Generate on first 3 prompts, pool metrics
        all_texts = []
        for prompt_text in PROMPTS[:3]:
            prompt_ids = source.encode(prompt_text)
            seqs = gen_fn(prompt_ids)
            texts = [source.decode(s) for s in seqs]
            all_texts.extend(texts)

        generated_texts_all[config_name] = all_texts
        metrics = compute_all_metrics(all_texts)
        metrics["jaccard_diversity"] = compute_pairwise_jaccard(all_texts)

        for k in all_metric_values:
            all_metric_values[k].append(metrics.get(k, float("nan")))

    # Compute Kendall tau for all metric pairs
    metric_names = list(all_metric_values.keys())
    correlations = {}
    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            mi, mj = metric_names[i], metric_names[j]
            vals_i = all_metric_values[mi]
            vals_j = all_metric_values[mj]
            # Skip if any NaN
            valid = [(a, b) for a, b in zip(vals_i, vals_j)
                     if not (math.isnan(a) or math.isnan(b))]
            if len(valid) >= 3:
                tau = kendall_tau([v[0] for v in valid], [v[1] for v in valid])
            else:
                tau = float("nan")
            correlations[f"{mi}_vs_{mj}"] = round(tau, 4)

    # Save generated texts
    with open(RESULTS_DIR / "h1_generated_texts.json", "w") as f:
        json.dump(generated_texts_all, f, indent=2)

    return {
        "experiment": "H1_Metric_Correlation_Taxonomy",
        "model": "GPT-2 124M",
        "n_configs": len(configs),
        "n_prompts": 3,
        "sequences_per_config_per_prompt": 8,
        "metric_names": metric_names,
        "metric_values_per_config": {
            k: [round(v, 4) if not math.isnan(v) else None for v in vals]
            for k, vals in all_metric_values.items()
        },
        "kendall_tau_correlations": correlations,
        "config_names": [c[0] for c in configs],
    }


# ---------------------------------------------------------------------------
# H2: Decoding algorithm comparison (real LM, includes SVD & QD-BS attempt)
# ---------------------------------------------------------------------------

def experiment_h2(source: GPT2LogitSource) -> Dict[str, Any]:
    logger.info("[H2] Decoding algorithm comparison on real GPT-2...")
    np.random.seed(SEED)

    prompt_text = "The future of artificial intelligence will"
    prompt_ids = source.encode(prompt_text)
    n_seqs = 10
    max_tokens = 30

    algorithms: Dict[str, Any] = {}

    # Standard algorithms
    algo_fns = {
        "greedy": lambda: generate_greedy(source, prompt_ids, n_seqs, max_tokens),
        "temperature_0.7": lambda: generate_temperature(source, prompt_ids, n_seqs, max_tokens, 0.7),
        "temperature_1.0": lambda: generate_temperature(source, prompt_ids, n_seqs, max_tokens, 1.0),
        "temperature_1.5": lambda: generate_temperature(source, prompt_ids, n_seqs, max_tokens, 1.5),
        "nucleus_p0.5": lambda: generate_nucleus(source, prompt_ids, n_seqs, max_tokens, 0.5),
        "nucleus_p0.9": lambda: generate_nucleus(source, prompt_ids, n_seqs, max_tokens, 0.9),
        "typical_0.9": lambda: generate_typical(source, prompt_ids, n_seqs, max_tokens, 0.9),
        "contrastive": lambda: generate_contrastive(source, prompt_ids, n_seqs, max_tokens),
        "diverse_beam": lambda: generate_diverse_beam(source, prompt_ids, n_groups=3, beam_width=3, max_tokens=max_tokens),
    }

    all_generated = {}

    for name, fn in algo_fns.items():
        logger.info("  Algorithm: %s", name)
        t0 = time.time()
        seqs = fn()
        elapsed = time.time() - t0
        texts = [source.decode(s) for s in seqs]
        all_generated[name] = texts

        metrics = compute_all_metrics(texts)
        metrics["jaccard_diversity"] = compute_pairwise_jaccard(texts)
        metrics["unique_sequences"] = len(set(texts))
        metrics["generation_time_sec"] = round(elapsed, 2)
        algorithms[name] = {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in metrics.items()}

    # Helper to decode sequences from algorithm outputs
    def _decode_seqs(seqs):
        decoded = []
        for s in seqs:
            if hasattr(s, 'token_ids'):
                decoded.append(source.decode(list(s.token_ids)))
            elif hasattr(s, 'tokens'):
                decoded.append(source.decode([int(t) for t in s.tokens]))
            elif isinstance(s, list):
                decoded.append(source.decode([int(t) for t in s]))
            else:
                decoded.append(str(s))
        return decoded

    # Try SVD from the codebase
    try:
        logger.info("  Algorithm: SVD (Stein Variational Decoding)")
        from src.algorithms.svd import SteinVariationalDecoding, SVDConfig
        svd_config = SVDConfig(
            n_particles=8,
            alpha=0.3,
            max_new_tokens=max_tokens,
            seed=SEED,
            kernel_type="rbf",
            bandwidth_method="median",
        )
        svd = SteinVariationalDecoding(svd_config)
        t0 = time.time()
        svd_seqs = svd.generate(source, prompt_ids)
        elapsed = time.time() - t0
        svd_texts = _decode_seqs(svd_seqs)
        all_generated["svd"] = svd_texts
        metrics = compute_all_metrics(svd_texts)
        metrics["jaccard_diversity"] = compute_pairwise_jaccard(svd_texts)
        metrics["unique_sequences"] = len(set(svd_texts))
        metrics["generation_time_sec"] = round(elapsed, 2)
        algorithms["svd"] = {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in metrics.items()}
    except Exception as e:
        logger.warning("SVD failed: %s (type: %s)", e, type(e).__name__)
        import traceback; traceback.print_exc()
        algorithms["svd"] = {"error": str(e)}

    # Try QD-BS from the codebase
    try:
        logger.info("  Algorithm: QD-BS (Quality-Diversity Beam Search)")
        from src.algorithms.qdbs import QualityDiversityBeamSearch, QDBSConfig
        qdbs_config = QDBSConfig(
            beam_width=6,
            archive_size=8,
            grid_resolution=3,
            max_new_tokens=max_tokens,
            seed=SEED,
        )
        qdbs = QualityDiversityBeamSearch(qdbs_config)
        t0 = time.time()
        qdbs_seqs = qdbs.generate(source, prompt_ids)
        elapsed = time.time() - t0
        qdbs_texts = _decode_seqs(qdbs_seqs)
        all_generated["qdbs"] = qdbs_texts
        metrics = compute_all_metrics(qdbs_texts)
        metrics["jaccard_diversity"] = compute_pairwise_jaccard(qdbs_texts)
        metrics["unique_sequences"] = len(set(qdbs_texts))
        metrics["generation_time_sec"] = round(elapsed, 2)
        algorithms["qdbs"] = {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in metrics.items()}
    except Exception as e:
        logger.warning("QD-BS failed: %s (type: %s)", e, type(e).__name__)
        import traceback; traceback.print_exc()
        algorithms["qdbs"] = {"error": str(e)}

    # Save all generated text
    with open(RESULTS_DIR / "h2_generated_texts.json", "w") as f:
        json.dump(all_generated, f, indent=2)

    return {
        "experiment": "H2_Decoding_Algorithm_Comparison",
        "model": "GPT-2 124M",
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "algorithm_results": algorithms,
    }


# ---------------------------------------------------------------------------
# H3: Pareto frontier (real diversity vs quality tradeoff)
# ---------------------------------------------------------------------------

def compute_quality(texts: List[str], source: GPT2LogitSource) -> float:
    """Compute average per-token log-probability as quality measure."""
    total_logprob = 0.0
    total_tokens = 0
    for text in texts:
        ids = source.encode(text)
        if len(ids) < 2:
            continue
        logits = source([ids])[0]  # only gets last position logits
        # Compute full sequence log-prob
        input_tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            outputs = source.model(input_ids=input_tensor)
        all_logits = outputs.logits[0]  # (seq_len, vocab)
        for t in range(len(ids) - 1):
            log_probs = all_logits[t] - torch.logsumexp(all_logits[t], dim=0)
            total_logprob += float(log_probs[ids[t + 1]])
            total_tokens += 1
    if total_tokens == 0:
        return 0.0
    return total_logprob / total_tokens  # average log-prob per token


def experiment_h3(source: GPT2LogitSource) -> Dict[str, Any]:
    logger.info("[H3] Pareto frontier analysis on real GPT-2...")
    np.random.seed(SEED)

    prompt_text = "The most important scientific discovery of the century was"
    prompt_ids = source.encode(prompt_text)

    configs = []
    temps = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]

    for temp in temps:
        logger.info("  Temperature: %.1f", temp)
        seqs = generate_temperature(source, prompt_ids, 8, 30, temp)
        texts = [source.decode(s) for s in seqs]

        from src.metrics.diversity import DistinctN
        dn = DistinctN(n=2)
        diversity = dn.compute(texts)
        quality = compute_quality(texts, source)
        # Normalize quality to [0, 1] range (log-prob is negative)
        # Typical range: -1 (good) to -6 (bad)
        quality_norm = max(0.0, min(1.0, 1.0 + quality / 5.0))

        configs.append({
            "temperature": temp,
            "diversity_distinct2": round(diversity, 4),
            "quality_avg_logprob": round(quality, 4),
            "quality_normalized": round(quality_norm, 4),
        })

    # Also add nucleus configs
    for p in [0.3, 0.5, 0.7, 0.9, 0.95]:
        logger.info("  Nucleus p=%.2f", p)
        seqs = generate_nucleus(source, prompt_ids, 8, 30, p)
        texts = [source.decode(s) for s in seqs]

        from src.metrics.diversity import DistinctN
        dn = DistinctN(n=2)
        diversity = dn.compute(texts)
        quality = compute_quality(texts, source)
        quality_norm = max(0.0, min(1.0, 1.0 + quality / 5.0))

        configs.append({
            "algorithm": "nucleus",
            "top_p": p,
            "diversity_distinct2": round(diversity, 4),
            "quality_avg_logprob": round(quality, 4),
            "quality_normalized": round(quality_norm, 4),
        })

    # Find Pareto front (maximize both diversity and quality_normalized)
    pareto = []
    for p in configs:
        d, q = p["diversity_distinct2"], p["quality_normalized"]
        dominated = any(
            c["diversity_distinct2"] >= d and c["quality_normalized"] >= q
            and (c["diversity_distinct2"] > d or c["quality_normalized"] > q)
            for c in configs
        )
        if not dominated:
            pareto.append(p)

    return {
        "experiment": "H3_Pareto_Frontier",
        "model": "GPT-2 124M",
        "prompt": prompt_text,
        "all_configs": configs,
        "pareto_front": sorted(pareto, key=lambda x: x["diversity_distinct2"]),
        "n_pareto_optimal": len(pareto),
        "n_total_configs": len(configs),
    }


# ---------------------------------------------------------------------------
# H4: Hypervolume (correct computation on real data)
# ---------------------------------------------------------------------------

def experiment_h4(source: GPT2LogitSource) -> Dict[str, Any]:
    logger.info("[H4] Hypervolume indicator on real GPT-2 outputs...")
    np.random.seed(SEED)

    prompt_text = "In the future, technology will"
    prompt_ids = source.encode(prompt_text)

    regimes = {
        "high_diversity": [
            ("temp_1.5", lambda: generate_temperature(source, prompt_ids, 8, 30, 1.5)),
            ("temp_2.0", lambda: generate_temperature(source, prompt_ids, 8, 30, 2.0)),
            ("nucleus_0.95", lambda: generate_nucleus(source, prompt_ids, 8, 30, 0.95)),
        ],
        "high_quality": [
            ("temp_0.3", lambda: generate_temperature(source, prompt_ids, 8, 30, 0.3)),
            ("temp_0.5", lambda: generate_temperature(source, prompt_ids, 8, 30, 0.5)),
            ("nucleus_0.3", lambda: generate_nucleus(source, prompt_ids, 8, 30, 0.3)),
        ],
        "balanced": [
            ("temp_0.8", lambda: generate_temperature(source, prompt_ids, 8, 30, 0.8)),
            ("temp_1.0", lambda: generate_temperature(source, prompt_ids, 8, 30, 1.0)),
            ("nucleus_0.7", lambda: generate_nucleus(source, prompt_ids, 8, 30, 0.7)),
        ],
    }

    regime_results = {}
    for regime_name, algo_list in regimes.items():
        logger.info("  Regime: %s", regime_name)
        points = []
        for algo_name, fn in algo_list:
            seqs = fn()
            texts = [source.decode(s) for s in seqs]
            from src.metrics.diversity import DistinctN
            dn = DistinctN(n=2)
            diversity = dn.compute(texts)
            quality = compute_quality(texts, source)
            quality_norm = max(0.0, min(1.0, 1.0 + quality / 5.0))
            points.append((diversity, quality_norm))

        ref_point = (0.0, 0.0)
        hv = hypervolume_2d(points, ref_point)
        regime_results[regime_name] = {
            "points": [(round(d, 4), round(q, 4)) for d, q in points],
            "hypervolume": round(hv, 4),
        }

    return {
        "experiment": "H4_Hypervolume_Comparison",
        "model": "GPT-2 124M",
        "prompt": prompt_text,
        "reference_point": [0.0, 0.0],
        "regime_results": regime_results,
        "hypervolume_algorithm": "exact_2d_sweep_line",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Diversity Decoding Arena — Real Experiment Runner")
    logger.info("Model: GPT-2 124M via HuggingFace Transformers")
    logger.info("=" * 60)

    source = GPT2LogitSource("gpt2")

    t_total = time.time()
    all_results = {}

    # H1: Metric correlation taxonomy
    t0 = time.time()
    all_results["H1"] = experiment_h1(source)
    all_results["H1"]["time_sec"] = round(time.time() - t0, 2)
    logger.info("H1 completed in %.1fs", all_results["H1"]["time_sec"])

    # H2: Algorithm comparison
    t0 = time.time()
    all_results["H2"] = experiment_h2(source)
    all_results["H2"]["time_sec"] = round(time.time() - t0, 2)
    logger.info("H2 completed in %.1fs", all_results["H2"]["time_sec"])

    # H3: Pareto frontier
    t0 = time.time()
    all_results["H3"] = experiment_h3(source)
    all_results["H3"]["time_sec"] = round(time.time() - t0, 2)
    logger.info("H3 completed in %.1fs", all_results["H3"]["time_sec"])

    # H4: Hypervolume
    t0 = time.time()
    all_results["H4"] = experiment_h4(source)
    all_results["H4"]["time_sec"] = round(time.time() - t0, 2)
    logger.info("H4 completed in %.1fs", all_results["H4"]["time_sec"])

    total_time = round(time.time() - t_total, 2)
    output = {
        "project": "diversity-decoding-arena",
        "model": "GPT-2 124M (HuggingFace Transformers)",
        "total_time_sec": total_time,
        "seed": SEED,
        "experiments": all_results,
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("All experiments completed in %.1fs", total_time)
    logger.info("Results saved to %s", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
