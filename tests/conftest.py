"""
Shared test fixtures for the Diversity Decoding Arena test suite.

Provides reusable fixtures for logit sources, algorithm configurations,
generation sets, metric computations, evaluation frameworks, task
definitions, and mock objects used across all test modules.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VOCAB_SIZE = 1000
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_SEQ_LEN = 20
DEFAULT_NUM_SEQUENCES = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_BEAMS = 8
DEFAULT_NUM_GROUPS = 4
DEFAULT_NUM_PARTICLES = 5
DEFAULT_MAX_NEW_TOKENS = 30

# Token IDs for special tokens
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
UNK_TOKEN_ID = 3


# =========================================================================
# Mock LogitSource
# =========================================================================


class MockLogitSource:
    """A deterministic logit source for testing.

    Generates logits based on a fixed distribution pattern that can be
    configured to produce predictable outputs for testing purposes.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        seed: int = DEFAULT_SEED,
        distribution: str = "uniform",
        concentration: float = 1.0,
        hot_tokens: Optional[List[int]] = None,
        eos_probability: float = 0.01,
        deterministic: bool = True,
    ):
        self.vocab_size = vocab_size
        self.seed = seed
        self.distribution = distribution
        self.concentration = concentration
        self.hot_tokens = hot_tokens or list(range(10, 60))
        self.eos_probability = eos_probability
        self.deterministic = deterministic
        self._rng = np.random.RandomState(seed)
        self._call_count = 0
        self._base_logits = self._create_base_logits()

    def _create_base_logits(self) -> np.ndarray:
        """Create base logit distribution."""
        if self.distribution == "uniform":
            logits = np.zeros(self.vocab_size, dtype=np.float32)
        elif self.distribution == "zipf":
            ranks = np.arange(1, self.vocab_size + 1, dtype=np.float32)
            logits = np.log(1.0 / (ranks ** self.concentration))
        elif self.distribution == "peaked":
            logits = self._rng.randn(self.vocab_size).astype(np.float32) * 0.5
            for tok in self.hot_tokens:
                if tok < self.vocab_size:
                    logits[tok] += self.concentration * 3.0
        elif self.distribution == "bimodal":
            logits = np.full(self.vocab_size, -5.0, dtype=np.float32)
            n_high = self.vocab_size // 10
            high_tokens = self._rng.choice(self.vocab_size, n_high, replace=False)
            logits[high_tokens] = 0.0
        elif self.distribution == "degenerate":
            logits = np.full(self.vocab_size, -100.0, dtype=np.float32)
            logits[self.hot_tokens[0] if self.hot_tokens else 10] = 0.0
        else:
            logits = self._rng.randn(self.vocab_size).astype(np.float32)

        if EOS_TOKEN_ID < self.vocab_size:
            logits[EOS_TOKEN_ID] = np.log(self.eos_probability + 1e-10)
        return logits

    def __call__(self, input_ids: List[List[int]]) -> np.ndarray:
        """Return logits of shape (batch, vocab_size)."""
        batch_size = len(input_ids)
        self._call_count += 1

        logits = np.tile(self._base_logits, (batch_size, 1)).copy()

        if not self.deterministic:
            noise = self._rng.randn(batch_size, self.vocab_size).astype(np.float32) * 0.1
            logits += noise

        for b in range(batch_size):
            seq = input_ids[b]
            if len(seq) > 0:
                last_token = seq[-1] % self.vocab_size
                logits[b, (last_token + 1) % self.vocab_size] += 0.5
                logits[b, (last_token + 2) % self.vocab_size] += 0.3

            seq_len = len(seq)
            if seq_len > 15:
                eos_boost = min((seq_len - 15) * 0.3, 5.0)
                if EOS_TOKEN_ID < self.vocab_size:
                    logits[b, EOS_TOKEN_ID] += eos_boost

        return logits

    def reset(self):
        """Reset internal state."""
        self._call_count = 0
        self._rng = np.random.RandomState(self.seed)

    @property
    def call_count(self) -> int:
        return self._call_count


class MockLogitSourceWithKVCache(MockLogitSource):
    """Mock logit source that simulates KV-cache behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, np.ndarray] = {}

    def __call__(self, input_ids: List[List[int]]) -> np.ndarray:
        batch_size = len(input_ids)
        logits = np.zeros((batch_size, self.vocab_size), dtype=np.float32)

        for b in range(batch_size):
            key = str(input_ids[b])
            if key in self._cache:
                logits[b] = self._cache[key]
            else:
                single_logit = super().__call__([input_ids[b]])[0]
                self._cache[key] = single_logit
                logits[b] = single_logit

        return logits

    def clear_cache(self):
        self._cache.clear()


class MockLogitSourceBatched(MockLogitSource):
    """Mock logit source that tracks batch sizes for testing batching logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sizes: List[int] = []

    def __call__(self, input_ids: List[List[int]]) -> np.ndarray:
        self.batch_sizes.append(len(input_ids))
        return super().__call__(input_ids)


class MockLogitSourceWithLatency(MockLogitSource):
    """Mock logit source that simulates inference latency."""

    def __init__(self, *args, latency_ms: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency_ms = latency_ms
        self.total_latency = 0.0

    def __call__(self, input_ids: List[List[int]]) -> np.ndarray:
        start = time.time()
        result = super().__call__(input_ids)
        elapsed = (time.time() - start) * 1000
        if elapsed < self.latency_ms:
            time.sleep((self.latency_ms - elapsed) / 1000.0)
        self.total_latency += self.latency_ms
        return result


class MockEmbedder:
    """A deterministic sentence embedder for testing."""

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        seed: int = DEFAULT_SEED,
        normalize: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.normalize = normalize
        self._rng = np.random.RandomState(seed)
        self._token_embeddings: Dict[int, np.ndarray] = {}
        self._cache: Dict[str, np.ndarray] = {}

    def _get_token_embedding(self, token_id: int) -> np.ndarray:
        if token_id not in self._token_embeddings:
            rng = np.random.RandomState(self.seed + token_id)
            emb = rng.randn(self.embedding_dim).astype(np.float32)
            if self.normalize:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
            self._token_embeddings[token_id] = emb
        return self._token_embeddings[token_id]

    def embed_sequence(self, token_ids: List[int]) -> np.ndarray:
        cache_key = str(token_ids)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not token_ids:
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            emb = np.mean(
                [self._get_token_embedding(t) for t in token_ids], axis=0
            )
            if self.normalize:
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm

        self._cache[cache_key] = emb
        return emb

    def embed_batch(self, sequences: List[List[int]]) -> np.ndarray:
        return np.array([self.embed_sequence(s) for s in sequences])

    def embed_text(self, text: str) -> np.ndarray:
        tokens = [ord(c) % DEFAULT_VOCAB_SIZE for c in text[:100]]
        return self.embed_sequence(tokens)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts])

    def token_embedding(self, token_id: int) -> np.ndarray:
        return self._get_token_embedding(token_id)

    def reset_cache(self):
        self._cache.clear()


class MockTokenizer:
    """A simple tokenizer mock for testing."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, seed: int = DEFAULT_SEED):
        self.vocab_size = vocab_size
        self.seed = seed
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.unk_token_id = UNK_TOKEN_ID
        self._rng = np.random.RandomState(seed)
        self._vocab = self._build_vocab()
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

    def _build_vocab(self) -> Dict[str, int]:
        vocab = {
            "<bos>": BOS_TOKEN_ID,
            "<eos>": EOS_TOKEN_ID,
            "<pad>": PAD_TOKEN_ID,
            "<unk>": UNK_TOKEN_ID,
        }
        words = [
            "the", "a", "an", "is", "was", "were", "are", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "need", "dare", "ought", "used", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how",
            "all", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "because", "but",
            "and", "or", "if", "while", "although", "though", "after",
            "that", "this", "these", "those", "what", "which", "who",
            "whom", "whose", "it", "he", "she", "they", "we", "you",
            "i", "me", "him", "her", "us", "them", "my", "your", "his",
            "its", "our", "their", "mine", "yours", "hers", "ours",
            "theirs", "myself", "yourself", "himself", "herself",
            "itself", "ourselves", "themselves",
            "hello", "world", "code", "python", "test", "function",
            "class", "method", "variable", "return", "import", "from",
            "def", "print", "input", "output", "data", "list", "dict",
            "set", "tuple", "string", "number", "float", "integer",
            "boolean", "true", "false", "none", "null", "error",
            "exception", "try", "except", "finally", "raise", "assert",
            "diversity", "quality", "metric", "score", "beam", "search",
            "sample", "token", "sequence", "generation", "model",
            "language", "neural", "network", "deep", "learning",
            "training", "inference", "embedding", "attention",
            "transformer", "encoder", "decoder", "layer", "weight",
            "bias", "gradient", "optimization", "loss", "accuracy",
            "precision", "recall", "f1", "bleu", "rouge", "perplexity",
            "entropy", "probability", "distribution", "categorical",
            "softmax", "temperature", "nucleus", "topk", "typical",
            "contrastive", "dpp", "kernel", "matrix", "eigenvalue",
            "determinant", "singular", "decomposition", "factorization",
            "algorithm", "implementation", "framework", "pipeline",
            "evaluation", "benchmark", "experiment", "result",
            "analysis", "comparison", "statistical", "significant",
            "correlation", "regression", "classification", "clustering",
            "dimensionality", "reduction", "visualization", "plot",
            "chart", "graph", "histogram", "scatter", "heatmap",
            "creative", "writing", "story", "poem", "dialogue",
            "character", "scene", "plot", "narrative", "fiction",
            "nonfiction", "essay", "article", "report", "summary",
            "abstract", "introduction", "conclusion", "discussion",
            "methodology", "results", "references", "appendix",
        ]
        for i, w in enumerate(words):
            token_id = i + 4
            if token_id < self.vocab_size:
                vocab[w] = token_id
        for i in range(len(words) + 4, self.vocab_size):
            vocab[f"<tok_{i}>"] = i
        return vocab

    def encode(self, text: str) -> List[int]:
        words = text.lower().split()
        return [self._vocab.get(w, UNK_TOKEN_ID) for w in words]

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            if tid in (BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID):
                continue
            words.append(self._reverse_vocab.get(tid, f"<tok_{tid}>"))
        return " ".join(words)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def batch_decode(self, token_id_lists: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in token_id_lists]

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab


# =========================================================================
# Text Generation Utilities
# =========================================================================


def generate_random_texts(
    n: int = DEFAULT_NUM_SEQUENCES,
    min_len: int = 10,
    max_len: int = 50,
    vocab: Optional[List[str]] = None,
    seed: int = DEFAULT_SEED,
) -> List[str]:
    """Generate a list of random text strings for testing."""
    rng = random.Random(seed)
    if vocab is None:
        vocab = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "a", "bright", "red", "sun", "rises", "in", "the", "east",
            "she", "wrote", "beautiful", "poems", "about", "nature", "and",
            "life", "they", "built", "amazing", "software", "with", "python",
            "code", "generates", "diverse", "outputs", "using", "algorithms",
            "machine", "learning", "models", "produce", "text", "from",
            "language", "patterns", "neural", "networks", "transform",
            "input", "to", "output", "sequences", "of", "tokens", "that",
            "form", "coherent", "sentences", "paragraphs", "stories",
            "every", "morning", "birds", "sing", "sweet", "melodies",
            "across", "valleys", "mountains", "rivers", "oceans",
            "data", "science", "explores", "hidden", "patterns",
            "algorithms", "optimize", "performance", "metrics",
            "quality", "diversity", "balance", "tradeoff", "between",
            "exploration", "exploitation", "search", "beam", "sample",
        ]
    texts = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        words = [rng.choice(vocab) for _ in range(length)]
        texts.append(" ".join(words))
    return texts


def generate_diverse_texts(
    n: int = DEFAULT_NUM_SEQUENCES,
    seed: int = DEFAULT_SEED,
) -> List[str]:
    """Generate texts with varying levels of diversity."""
    rng = random.Random(seed)
    topics = [
        ["the", "cat", "sat", "on", "mat", "purred", "softly", "warm", "cozy", "fur"],
        ["python", "code", "function", "class", "module", "import", "library", "debug"],
        ["stars", "galaxy", "universe", "cosmos", "space", "nebula", "light", "dark"],
        ["music", "melody", "harmony", "rhythm", "beat", "song", "note", "chord"],
        ["ocean", "wave", "beach", "sand", "shell", "tide", "reef", "coral", "fish"],
        ["mountain", "peak", "valley", "river", "trail", "forest", "tree", "rock"],
        ["recipe", "cook", "bake", "ingredient", "spice", "flour", "sugar", "salt"],
        ["paint", "canvas", "brush", "color", "art", "gallery", "frame", "studio"],
    ]
    texts = []
    for i in range(n):
        topic = topics[i % len(topics)]
        length = rng.randint(15, 40)
        words = [rng.choice(topic) for _ in range(length)]
        filler = ["the", "a", "is", "and", "with", "for", "of", "in"]
        for j in range(len(words)):
            if rng.random() < 0.3:
                words[j] = rng.choice(filler)
        texts.append(" ".join(words))
    return texts


def generate_identical_texts(
    n: int = DEFAULT_NUM_SEQUENCES,
    text: str = "the quick brown fox jumps over the lazy dog",
) -> List[str]:
    """Generate identical texts for testing minimum diversity."""
    return [text] * n


def generate_token_sequences(
    n: int = DEFAULT_NUM_SEQUENCES,
    min_len: int = 5,
    max_len: int = DEFAULT_SEQ_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = DEFAULT_SEED,
) -> List[List[int]]:
    """Generate random token ID sequences."""
    rng = np.random.RandomState(seed)
    sequences = []
    for _ in range(n):
        length = rng.randint(min_len, max_len + 1)
        seq = rng.randint(4, vocab_size, size=length).tolist()
        sequences.append(seq)
    return sequences


def generate_diverse_token_sequences(
    n: int = DEFAULT_NUM_SEQUENCES,
    seq_len: int = DEFAULT_SEQ_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = DEFAULT_SEED,
    n_clusters: int = 3,
) -> List[List[int]]:
    """Generate token sequences clustered around different regions of vocab."""
    rng = np.random.RandomState(seed)
    sequences = []
    cluster_centers = [
        int(vocab_size * (i + 0.5) / n_clusters) for i in range(n_clusters)
    ]
    cluster_width = vocab_size // (n_clusters * 3)

    for i in range(n):
        cluster = i % n_clusters
        center = cluster_centers[cluster]
        tokens = []
        for _ in range(seq_len):
            tok = int(rng.normal(center, cluster_width))
            tok = max(4, min(vocab_size - 1, tok))
            tokens.append(tok)
        sequences.append(tokens)
    return sequences


def generate_logit_matrix(
    batch_size: int = DEFAULT_BATCH_SIZE,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    distribution: str = "normal",
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Generate a logit matrix for testing."""
    rng = np.random.RandomState(seed)
    if distribution == "normal":
        return rng.randn(batch_size, vocab_size).astype(np.float32)
    elif distribution == "uniform":
        return rng.uniform(-3, 3, (batch_size, vocab_size)).astype(np.float32)
    elif distribution == "peaked":
        logits = rng.randn(batch_size, vocab_size).astype(np.float32) * 0.3
        for b in range(batch_size):
            hot = rng.choice(vocab_size, 5, replace=False)
            logits[b, hot] += 5.0
        return logits
    elif distribution == "flat":
        return np.zeros((batch_size, vocab_size), dtype=np.float32)
    else:
        return rng.randn(batch_size, vocab_size).astype(np.float32)


def generate_embedding_matrix(
    n: int = DEFAULT_NUM_SEQUENCES,
    dim: int = DEFAULT_EMBEDDING_DIM,
    seed: int = DEFAULT_SEED,
    normalize: bool = True,
) -> np.ndarray:
    """Generate an embedding matrix for testing."""
    rng = np.random.RandomState(seed)
    embeddings = rng.randn(n, dim).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
    return embeddings


def generate_kernel_matrix(
    n: int = DEFAULT_NUM_SEQUENCES,
    kernel_type: str = "rbf",
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Generate a valid kernel (positive semi-definite) matrix."""
    rng = np.random.RandomState(seed)
    if kernel_type == "rbf":
        embeddings = rng.randn(n, DEFAULT_EMBEDDING_DIM).astype(np.float32)
        dists = np.sum((embeddings[:, None] - embeddings[None, :]) ** 2, axis=-1)
        K = np.exp(-dists / (2.0 * np.median(dists + 1e-10)))
    elif kernel_type == "linear":
        X = rng.randn(n, DEFAULT_EMBEDDING_DIM).astype(np.float32)
        K = X @ X.T
    elif kernel_type == "identity":
        K = np.eye(n, dtype=np.float32)
    elif kernel_type == "uniform":
        K = np.ones((n, n), dtype=np.float32)
    else:
        A = rng.randn(n, n).astype(np.float32)
        K = A @ A.T + 0.1 * np.eye(n, dtype=np.float32)

    K = (K + K.T) / 2.0
    np.fill_diagonal(K, np.maximum(np.diag(K), 1e-6))
    return K


# =========================================================================
# Mock Generation Results
# =========================================================================


@dataclass
class MockGenerationResult:
    """Simplified generation result for testing."""
    texts: List[str]
    token_ids: List[List[int]]
    log_probs: List[float]
    algorithm: str
    config: Dict[str, Any]
    elapsed_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_mock_generation_result(
    algorithm: str = "temperature",
    n: int = DEFAULT_NUM_SEQUENCES,
    seed: int = DEFAULT_SEED,
    diverse: bool = True,
) -> MockGenerationResult:
    """Create a mock generation result for testing."""
    if diverse:
        texts = generate_diverse_texts(n, seed)
    else:
        texts = generate_random_texts(n, seed=seed)

    tokenizer = MockTokenizer()
    token_ids = [tokenizer.encode(t) for t in texts]
    rng = np.random.RandomState(seed)
    log_probs = (-rng.exponential(2.0, n)).tolist()

    return MockGenerationResult(
        texts=texts,
        token_ids=token_ids,
        log_probs=log_probs,
        algorithm=algorithm,
        config={"temperature": 1.0, "seed": seed},
        elapsed_time=rng.exponential(1.0),
    )


def create_multiple_generation_results(
    algorithms: Optional[List[str]] = None,
    n_per_algo: int = DEFAULT_NUM_SEQUENCES,
    seed: int = DEFAULT_SEED,
) -> Dict[str, MockGenerationResult]:
    """Create generation results for multiple algorithms."""
    if algorithms is None:
        algorithms = [
            "temperature", "top_k", "nucleus", "typical",
            "diverse_beam", "contrastive", "dpp", "mbr",
            "svd", "qdbs",
        ]
    results = {}
    for i, algo in enumerate(algorithms):
        results[algo] = create_mock_generation_result(
            algorithm=algo, n=n_per_algo, seed=seed + i
        )
    return results


# =========================================================================
# Metric Computation Helpers
# =========================================================================


def compute_ngram_frequencies(texts: List[str], n: int = 2) -> Counter:
    """Compute n-gram frequency distribution across texts."""
    counter: Counter = Counter()
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            counter[ngram] += 1
    return counter


def compute_pairwise_jaccard(texts: List[str], n: int = 2) -> np.ndarray:
    """Compute pairwise Jaccard similarity matrix."""
    ngram_sets = []
    for text in texts:
        words = text.lower().split()
        ngrams = set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        ngram_sets.append(ngrams)

    m = len(texts)
    similarity = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i == j:
                similarity[i, j] = 1.0
            else:
                inter = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                similarity[i, j] = inter / max(union, 1)
    return similarity


def compute_simple_self_bleu(texts: List[str], n: int = 4) -> float:
    """Compute a simplified Self-BLEU score."""
    if len(texts) < 2:
        return 0.0

    def _bleu_single(hypothesis_tokens, reference_tokens_list, max_n=4):
        brevity_penalty = 1.0
        hyp_len = len(hypothesis_tokens)
        ref_lens = [len(r) for r in reference_tokens_list]
        closest_len = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
        if hyp_len < closest_len:
            brevity_penalty = math.exp(1 - closest_len / max(hyp_len, 1))

        log_avg = 0.0
        for ng in range(1, min(max_n + 1, hyp_len + 1)):
            hyp_ngrams: Counter = Counter()
            for i in range(len(hypothesis_tokens) - ng + 1):
                hyp_ngrams[tuple(hypothesis_tokens[i:i + ng])] += 1

            max_ref: Counter = Counter()
            for ref_tokens in reference_tokens_list:
                ref_ngrams: Counter = Counter()
                for i in range(len(ref_tokens) - ng + 1):
                    ref_ngrams[tuple(ref_tokens[i:i + ng])] += 1
                for k, v in ref_ngrams.items():
                    max_ref[k] = max(max_ref[k], v)

            clipped = sum(min(c, max_ref.get(k, 0)) for k, c in hyp_ngrams.items())
            total = max(sum(hyp_ngrams.values()), 1)
            precision = clipped / total
            if precision == 0:
                return 0.0
            log_avg += math.log(precision + 1e-10) / min(max_n, hyp_len)

        return brevity_penalty * math.exp(log_avg)

    tokenized = [text.lower().split() for text in texts]
    scores = []
    for i in range(len(tokenized)):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        score = _bleu_single(tokenized[i], refs, n)
        scores.append(score)
    return sum(scores) / len(scores)


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """Compute Distinct-N metric."""
    all_ngrams: List[Tuple] = []
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            all_ngrams.append(tuple(words[i:i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_ngram_entropy(texts: List[str], n: int = 2) -> float:
    """Compute n-gram entropy."""
    counter = compute_ngram_frequencies(texts, n)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# =========================================================================
# Temporary Directory Management
# =========================================================================


class TempDirectoryManager:
    """Manage temporary directories for test artifacts."""

    def __init__(self, prefix: str = "arena_test_"):
        self.prefix = prefix
        self._dirs: List[str] = []

    def create(self) -> str:
        d = tempfile.mkdtemp(prefix=self.prefix)
        self._dirs.append(d)
        return d

    def cleanup(self):
        for d in self._dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        self._dirs.clear()


class TempDatabaseManager:
    """Manage temporary SQLite databases for test artifacts."""

    def __init__(self):
        self._dbs: List[Tuple[str, sqlite3.Connection]] = []

    def create(self) -> Tuple[str, sqlite3.Connection]:
        fd, path = tempfile.mkstemp(suffix=".db", prefix="arena_test_")
        os.close(fd)
        conn = sqlite3.connect(path)
        self._dbs.append((path, conn))
        return path, conn

    def cleanup(self):
        for path, conn in self._dbs:
            try:
                conn.close()
            except Exception:
                pass
            if os.path.exists(path):
                os.unlink(path)
        self._dbs.clear()


# =========================================================================
# Assertion Helpers
# =========================================================================


def assert_valid_probability_distribution(probs: np.ndarray, atol: float = 1e-5):
    """Assert that probs is a valid probability distribution."""
    assert np.all(probs >= -atol), f"Negative probabilities found: {probs.min()}"
    assert abs(np.sum(probs) - 1.0) < atol, f"Probabilities sum to {np.sum(probs)}"


def assert_valid_logits(logits: np.ndarray, vocab_size: int):
    """Assert logits have correct shape and finite values."""
    assert logits.shape[-1] == vocab_size, (
        f"Expected vocab_size={vocab_size}, got {logits.shape[-1]}"
    )
    assert np.all(np.isfinite(logits)), "Non-finite logits found"


def assert_diverse_texts(texts: List[str], min_unique_ratio: float = 0.5):
    """Assert that texts have sufficient diversity."""
    unique = len(set(texts))
    ratio = unique / max(len(texts), 1)
    assert ratio >= min_unique_ratio, (
        f"Insufficient diversity: {unique}/{len(texts)} unique "
        f"(ratio={ratio:.3f}, min={min_unique_ratio})"
    )


def assert_metric_in_range(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
    name: str = "metric",
):
    """Assert that a metric value is within expected range."""
    assert low <= value <= high, (
        f"{name} = {value} out of range [{low}, {high}]"
    )


def assert_monotonic(
    values: Sequence[float],
    increasing: bool = True,
    strict: bool = False,
    name: str = "sequence",
):
    """Assert that a sequence is monotonically increasing/decreasing."""
    for i in range(1, len(values)):
        if increasing:
            if strict:
                assert values[i] > values[i - 1], (
                    f"{name}[{i}]={values[i]} <= {name}[{i-1}]={values[i-1]}"
                )
            else:
                assert values[i] >= values[i - 1], (
                    f"{name}[{i}]={values[i]} < {name}[{i-1}]={values[i-1]}"
                )
        else:
            if strict:
                assert values[i] < values[i - 1], (
                    f"{name}[{i}]={values[i]} >= {name}[{i-1}]={values[i-1]}"
                )
            else:
                assert values[i] <= values[i - 1], (
                    f"{name}[{i}]={values[i]} > {name}[{i-1}]={values[i-1]}"
                )


def assert_positive_semidefinite(K: np.ndarray, atol: float = 1e-6):
    """Assert that K is a positive semi-definite matrix."""
    assert K.shape[0] == K.shape[1], "Matrix must be square"
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -atol), (
        f"Negative eigenvalue found: {eigenvalues.min()}"
    )


def assert_symmetric(M: np.ndarray, atol: float = 1e-6):
    """Assert that M is symmetric."""
    assert M.shape[0] == M.shape[1], "Matrix must be square"
    assert np.allclose(M, M.T, atol=atol), "Matrix is not symmetric"


def assert_stochastic_matrix(M: np.ndarray, atol: float = 1e-5):
    """Assert that M is a (row) stochastic matrix."""
    assert np.all(M >= -atol), "Negative entries"
    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=atol), (
        f"Row sums not 1: {row_sums}"
    )


# =========================================================================
# Statistical Test Helpers
# =========================================================================


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, n, replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return float(np.mean(arr)), lo, hi


def permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    n_permutations: int = 1000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sample permutation test, returns p-value."""
    rng = np.random.RandomState(seed)
    a = np.array(group_a, dtype=np.float64)
    b = np.array(group_b, dtype=np.float64)
    observed = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if perm_diff >= observed:
            count += 1
    return count / n_permutations


def effect_size_cohens_d(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """Compute Cohen's d effect size."""
    a = np.array(group_a, dtype=np.float64)
    b = np.array(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


# =========================================================================
# Pytest Fixtures
# =========================================================================


@pytest.fixture
def rng():
    """Seeded numpy random state."""
    return np.random.RandomState(DEFAULT_SEED)


@pytest.fixture
def mock_logit_source():
    """Basic mock logit source with uniform distribution."""
    return MockLogitSource(distribution="uniform")


@pytest.fixture
def mock_logit_source_zipf():
    """Mock logit source with Zipfian distribution."""
    return MockLogitSource(distribution="zipf")


@pytest.fixture
def mock_logit_source_peaked():
    """Mock logit source with peaked distribution."""
    return MockLogitSource(distribution="peaked", concentration=2.0)


@pytest.fixture
def mock_logit_source_bimodal():
    """Mock logit source with bimodal distribution."""
    return MockLogitSource(distribution="bimodal")


@pytest.fixture
def mock_logit_source_degenerate():
    """Mock logit source with degenerate (single-token) distribution."""
    return MockLogitSource(distribution="degenerate")


@pytest.fixture
def mock_logit_source_kv_cache():
    """Mock logit source with KV-cache simulation."""
    return MockLogitSourceWithKVCache(distribution="zipf")


@pytest.fixture
def mock_logit_source_batched():
    """Mock logit source that tracks batch sizes."""
    return MockLogitSourceBatched(distribution="zipf")


@pytest.fixture
def mock_embedder():
    """Mock sentence embedder."""
    return MockEmbedder()


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def random_texts():
    """List of random texts."""
    return generate_random_texts()


@pytest.fixture
def diverse_texts():
    """List of diverse texts from different topics."""
    return generate_diverse_texts()


@pytest.fixture
def identical_texts():
    """List of identical texts."""
    return generate_identical_texts()


@pytest.fixture
def random_token_sequences():
    """List of random token sequences."""
    return generate_token_sequences()


@pytest.fixture
def diverse_token_sequences():
    """List of diverse token sequences (clustered)."""
    return generate_diverse_token_sequences()


@pytest.fixture
def logit_matrix():
    """Random logit matrix (batch x vocab)."""
    return generate_logit_matrix()


@pytest.fixture
def peaked_logit_matrix():
    """Peaked logit matrix for testing top-k/top-p filtering."""
    return generate_logit_matrix(distribution="peaked")


@pytest.fixture
def embedding_matrix():
    """Random normalized embedding matrix."""
    return generate_embedding_matrix()


@pytest.fixture
def rbf_kernel_matrix():
    """RBF kernel matrix."""
    return generate_kernel_matrix(kernel_type="rbf")


@pytest.fixture
def linear_kernel_matrix():
    """Linear kernel matrix."""
    return generate_kernel_matrix(kernel_type="linear")


@pytest.fixture
def identity_kernel_matrix():
    """Identity kernel matrix."""
    return generate_kernel_matrix(kernel_type="identity")


@pytest.fixture
def mock_generation_result():
    """Single mock generation result."""
    return create_mock_generation_result()


@pytest.fixture
def multiple_generation_results():
    """Generation results for multiple algorithms."""
    return create_multiple_generation_results()


@pytest.fixture
def temp_dir():
    """Temporary directory cleaned up after test."""
    manager = TempDirectoryManager()
    d = manager.create()
    yield d
    manager.cleanup()


@pytest.fixture
def temp_db():
    """Temporary SQLite database cleaned up after test."""
    manager = TempDatabaseManager()
    path, conn = manager.create()
    yield path, conn
    manager.cleanup()


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing task domains."""
    return [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "Generate Python code to sort a list of integers.",
        "Summarize the key points of climate change research.",
        "Translate the following sentence to French: The weather is beautiful today.",
        "List five creative uses for a paperclip.",
        "Write a haiku about artificial intelligence.",
        "Describe the process of photosynthesis.",
        "Create a dialogue between two characters meeting for the first time.",
        "Write a product review for a fictional gadget.",
    ]


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "algorithm_name": "temperature",
        "num_sequences": 10,
        "max_new_tokens": 30,
        "min_new_tokens": 5,
        "seed": DEFAULT_SEED,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "params": {},
    }


@pytest.fixture
def small_vocab_logit_source():
    """Logit source with small vocabulary for exhaustive tests."""
    return MockLogitSource(vocab_size=50, distribution="zipf")


@pytest.fixture
def large_vocab_logit_source():
    """Logit source with large vocabulary."""
    return MockLogitSource(vocab_size=5000, distribution="zipf")


@pytest.fixture
def ngram_frequencies():
    """Pre-computed n-gram frequencies for metric tests."""
    texts = generate_random_texts(20, seed=DEFAULT_SEED)
    return {
        "unigram": compute_ngram_frequencies(texts, 1),
        "bigram": compute_ngram_frequencies(texts, 2),
        "trigram": compute_ngram_frequencies(texts, 3),
    }


@pytest.fixture
def pairwise_similarity_matrix():
    """Pre-computed pairwise similarity matrix."""
    texts = generate_diverse_texts(10, seed=DEFAULT_SEED)
    return compute_pairwise_jaccard(texts, n=2)


@pytest.fixture
def metric_values_for_correlation():
    """Pre-computed metric values for correlation tests."""
    rng = np.random.RandomState(DEFAULT_SEED)
    n_configs = 30
    return {
        "self_bleu": rng.uniform(0.1, 0.9, n_configs).tolist(),
        "distinct_2": rng.uniform(0.2, 0.95, n_configs).tolist(),
        "entropy_2": rng.uniform(2.0, 8.0, n_configs).tolist(),
        "embedding_distance": rng.uniform(0.1, 0.8, n_configs).tolist(),
        "vendi_score": rng.uniform(1.0, 10.0, n_configs).tolist(),
    }


@pytest.fixture
def pareto_points():
    """Points for Pareto analysis tests."""
    rng = np.random.RandomState(DEFAULT_SEED)
    n = 50
    diversity = rng.uniform(0, 1, n)
    quality = 1.0 - diversity + rng.normal(0, 0.1, n)
    quality = np.clip(quality, 0, 1)
    return np.column_stack([diversity, quality])


@pytest.fixture
def bayesian_comparison_data():
    """Paired metric data for Bayesian comparison tests."""
    rng = np.random.RandomState(DEFAULT_SEED)
    n = 100
    algo_a = rng.normal(0.5, 0.1, n)
    algo_b = rng.normal(0.55, 0.12, n)
    return algo_a, algo_b


@pytest.fixture
def sweep_configs():
    """Hyperparameter sweep configurations."""
    configs = []
    for temp in [0.5, 0.7, 1.0, 1.3, 1.5]:
        configs.append({
            "algorithm_name": "temperature",
            "temperature": temp,
            "num_sequences": 10,
            "max_new_tokens": 20,
        })
    for top_k in [10, 20, 50, 100]:
        configs.append({
            "algorithm_name": "top_k",
            "params": {"k": top_k},
            "num_sequences": 10,
            "max_new_tokens": 20,
        })
    for top_p in [0.7, 0.8, 0.9, 0.95]:
        configs.append({
            "algorithm_name": "nucleus",
            "params": {"p": top_p},
            "num_sequences": 10,
            "max_new_tokens": 20,
        })
    return configs


# =========================================================================
# Algorithm-specific Fixture Factories
# =========================================================================


class AlgorithmFixtureFactory:
    """Factory for creating algorithm-specific test fixtures."""

    @staticmethod
    def temperature_config(
        temperature: float = 1.0,
        schedule: str = "constant",
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "algorithm_name": "temperature",
            "temperature": temperature,
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {"schedule": schedule},
        }

    @staticmethod
    def top_k_config(k: int = DEFAULT_TOP_K, **kwargs) -> Dict[str, Any]:
        return {
            "algorithm_name": "top_k",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {"k": k},
        }

    @staticmethod
    def nucleus_config(p: float = DEFAULT_TOP_P, **kwargs) -> Dict[str, Any]:
        return {
            "algorithm_name": "nucleus",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {"p": p},
        }

    @staticmethod
    def typical_config(mass: float = 0.9, **kwargs) -> Dict[str, Any]:
        return {
            "algorithm_name": "typical",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {"mass": mass},
        }

    @staticmethod
    def diverse_beam_config(
        num_beams: int = DEFAULT_NUM_BEAMS,
        num_groups: int = DEFAULT_NUM_GROUPS,
        diversity_penalty: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "algorithm_name": "diverse_beam",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {
                "num_beams": num_beams,
                "num_groups": num_groups,
                "diversity_penalty": diversity_penalty,
            },
        }

    @staticmethod
    def contrastive_config(
        alpha: float = 0.6,
        k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "algorithm_name": "contrastive",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {"alpha": alpha, "k": k},
        }

    @staticmethod
    def dpp_config(
        pool_size: int = 50,
        select_size: int = 10,
        kernel_type: str = "rbf",
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "algorithm_name": "dpp",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {
                "pool_size": pool_size,
                "select_size": select_size,
                "kernel_type": kernel_type,
            },
        }

    @staticmethod
    def svd_config(
        n_particles: int = DEFAULT_NUM_PARTICLES,
        alpha: float = 0.1,
        bandwidth: str = "median",
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "algorithm_name": "svd",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {
                "n_particles": n_particles,
                "alpha": alpha,
                "bandwidth": bandwidth,
            },
        }

    @staticmethod
    def qdbs_config(
        beam_width: int = DEFAULT_NUM_BEAMS,
        n_cells: int = 20,
        behavior_dims: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "algorithm_name": "qdbs",
            "num_sequences": kwargs.get("num_sequences", DEFAULT_NUM_SEQUENCES),
            "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "seed": kwargs.get("seed", DEFAULT_SEED),
            "params": {
                "beam_width": beam_width,
                "n_cells": n_cells,
                "behavior_dims": behavior_dims,
            },
        }


@pytest.fixture
def algorithm_factory():
    """Algorithm fixture factory."""
    return AlgorithmFixtureFactory()


# =========================================================================
# Task Domain Fixtures
# =========================================================================


@pytest.fixture
def creative_writing_prompts():
    """Prompts for creative writing tasks."""
    return [
        "Write a short story about a time traveler who arrives in ancient Rome.",
        "Compose a poem about the changing seasons in a mountain village.",
        "Create a dialogue between the sun and the moon.",
        "Write a fairy tale about a magical library.",
        "Describe a futuristic city where plants and technology coexist.",
        "Write a letter from a sailor to their family back home.",
        "Create a monologue for a villain explaining their worldview.",
        "Write a scene where two old friends reunite after twenty years.",
    ]


@pytest.fixture
def code_generation_prompts():
    """Prompts for code generation tasks."""
    return [
        "Write a Python function to find the longest common subsequence.",
        "Implement a binary search tree with insert, delete, and search.",
        "Create a function to validate email addresses using regex.",
        "Write a class that implements a LRU cache.",
        "Implement the merge sort algorithm.",
        "Create a function to detect cycles in a linked list.",
        "Write a Python decorator for memoization.",
        "Implement a priority queue using a heap.",
    ]


@pytest.fixture
def summarization_prompts():
    """Prompts for summarization tasks."""
    return [
        "Summarize: Machine learning is a branch of AI that focuses on building systems that learn from data.",
        "Summarize: The Internet of Things connects billions of devices worldwide, enabling smart homes and cities.",
        "Summarize: Climate change is causing rising sea levels, more extreme weather, and biodiversity loss.",
        "Summarize: Quantum computing uses quantum bits that can exist in multiple states simultaneously.",
        "Summarize: Blockchain technology provides a decentralized and tamper-proof way to record transactions.",
    ]


@pytest.fixture
def qa_prompts():
    """Prompts for question answering tasks."""
    return [
        "Q: What is photosynthesis? A:",
        "Q: How does gravity work? A:",
        "Q: What causes rainbows? A:",
        "Q: Why is the sky blue? A:",
        "Q: How do vaccines work? A:",
    ]


@pytest.fixture
def translation_prompts():
    """Prompts for translation tasks."""
    return [
        "Translate to French: The weather is beautiful today.",
        "Translate to Spanish: I would like a cup of coffee.",
        "Translate to German: The book is on the table.",
        "Translate to Italian: Where is the nearest restaurant?",
        "Translate to Portuguese: Thank you for your help.",
    ]


@pytest.fixture
def brainstorming_prompts():
    """Prompts for brainstorming tasks."""
    return [
        "List creative uses for recycled materials.",
        "Generate ideas for a mobile app that helps people sleep better.",
        "Brainstorm ways to reduce food waste in restaurants.",
        "Come up with names for a new eco-friendly clothing brand.",
        "Suggest innovative features for a smart home system.",
    ]


# =========================================================================
# Evaluation Fixtures
# =========================================================================


@pytest.fixture
def arena_config():
    """Configuration for arena evaluation."""
    return {
        "algorithms": [
            "temperature", "top_k", "nucleus", "typical",
            "diverse_beam", "contrastive", "dpp", "svd", "qdbs",
        ],
        "metrics": {
            "diversity": ["self_bleu", "distinct_2", "entropy_2", "embedding_distance"],
            "quality": ["perplexity", "coherence"],
        },
        "tasks": ["creative_writing", "code_generation", "summarization"],
        "n_sequences": 10,
        "n_prompts": 5,
        "seed": DEFAULT_SEED,
        "statistical_test": "bayesian",
        "rope_epsilon": 0.01,
    }


@pytest.fixture
def pareto_config():
    """Configuration for Pareto analysis."""
    return {
        "objectives": [
            {"name": "diversity", "direction": "maximize"},
            {"name": "quality", "direction": "maximize"},
        ],
        "reference_point": [0.0, 0.0],
        "n_samples_hypervolume": 10000,
    }


@pytest.fixture
def statistical_test_config():
    """Configuration for statistical tests."""
    return {
        "alpha": 0.05,
        "n_bootstrap": 1000,
        "rope_epsilon": 0.01,
        "correction_method": "bonferroni",
        "seed": DEFAULT_SEED,
    }


# =========================================================================
# CLI Fixtures
# =========================================================================


@pytest.fixture
def cli_args_run():
    """CLI arguments for 'run' command."""
    return [
        "run",
        "--algorithms", "temperature", "nucleus",
        "--tasks", "creative_writing",
        "--n-sequences", "5",
        "--max-tokens", "20",
        "--seed", str(DEFAULT_SEED),
    ]


@pytest.fixture
def cli_args_evaluate():
    """CLI arguments for 'evaluate' command."""
    return [
        "evaluate",
        "--input-dir", "/tmp/arena_results",
        "--metrics", "self_bleu", "distinct_2",
        "--output-format", "json",
    ]


@pytest.fixture
def cli_args_compare():
    """CLI arguments for 'compare' command."""
    return [
        "compare",
        "--input-dirs", "/tmp/results_a", "/tmp/results_b",
        "--method", "bayesian",
        "--rope", "0.01",
    ]


@pytest.fixture
def cli_args_sweep():
    """CLI arguments for 'sweep' command."""
    return [
        "sweep",
        "--algorithm", "temperature",
        "--param", "temperature",
        "--values", "0.5", "0.7", "1.0", "1.3",
        "--tasks", "creative_writing",
    ]


# =========================================================================
# Integration Test Fixtures
# =========================================================================


@pytest.fixture
def full_pipeline_config():
    """Configuration for full pipeline integration tests."""
    return {
        "model": {
            "name": "gpt2",
            "device": "cpu",
            "quantization": "int8",
        },
        "algorithms": {
            "temperature": {"temperature": [0.7, 1.0]},
            "nucleus": {"p": [0.9, 0.95]},
        },
        "tasks": ["creative_writing"],
        "metrics": {
            "diversity": ["self_bleu", "distinct_2"],
            "quality": ["perplexity"],
        },
        "evaluation": {
            "n_sequences": 5,
            "max_tokens": 15,
            "seed": DEFAULT_SEED,
        },
    }


@pytest.fixture
def mock_experiment_results():
    """Complete mock experiment results for integration tests."""
    rng = np.random.RandomState(DEFAULT_SEED)
    algorithms = ["temperature", "nucleus", "diverse_beam", "svd"]
    tasks = ["creative_writing", "code_generation"]
    metrics = ["self_bleu", "distinct_2", "perplexity"]

    results = {}
    for algo in algorithms:
        results[algo] = {}
        for task in tasks:
            results[algo][task] = {}
            for metric in metrics:
                values = rng.uniform(0, 1, 10).tolist()
                results[algo][task][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
    return results


# =========================================================================
# Parametrize Helpers
# =========================================================================


ALL_ALGORITHM_NAMES = [
    "temperature", "top_k", "nucleus", "typical",
    "diverse_beam", "contrastive", "dpp", "mbr",
    "svd", "qdbs",
]

ALL_DIVERSITY_METRICS = [
    "self_bleu", "distinct_1", "distinct_2", "distinct_3",
    "entropy_1", "entropy_2", "entropy_3",
    "embedding_distance", "vendi_score",
    "parse_tree_diversity", "behavioral_diversity",
]

ALL_QUALITY_METRICS = [
    "perplexity", "coherence", "constraint_satisfaction",
]

ALL_TASK_DOMAINS = [
    "creative_writing", "code_generation", "summarization",
    "qa", "translation", "brainstorming",
]

KERNEL_TYPES = ["rbf", "linear", "cosine", "polynomial"]

DISTRIBUTIONS = ["uniform", "zipf", "peaked", "bimodal", "degenerate"]


def parametrize_algorithms(exclude: Optional[List[str]] = None):
    """Parametrize decorator for all algorithms."""
    algos = [a for a in ALL_ALGORITHM_NAMES if a not in (exclude or [])]
    return pytest.mark.parametrize("algorithm_name", algos)


def parametrize_diversity_metrics(exclude: Optional[List[str]] = None):
    """Parametrize decorator for all diversity metrics."""
    metrics = [m for m in ALL_DIVERSITY_METRICS if m not in (exclude or [])]
    return pytest.mark.parametrize("metric_name", metrics)


def parametrize_task_domains(exclude: Optional[List[str]] = None):
    """Parametrize decorator for all task domains."""
    tasks = [t for t in ALL_TASK_DOMAINS if t not in (exclude or [])]
    return pytest.mark.parametrize("task_domain", tasks)


def parametrize_distributions():
    """Parametrize decorator for logit distributions."""
    return pytest.mark.parametrize("distribution", DISTRIBUTIONS)


def parametrize_kernel_types():
    """Parametrize decorator for kernel types."""
    return pytest.mark.parametrize("kernel_type", KERNEL_TYPES)


# =========================================================================
# Numerical Stability Helpers
# =========================================================================


def check_numerical_stability(
    func: Callable,
    inputs: List[Any],
    n_trials: int = 10,
    atol: float = 1e-6,
) -> bool:
    """Check that a function produces stable results across multiple calls."""
    results = [func(*inputs) for _ in range(n_trials)]
    if isinstance(results[0], (int, float)):
        return all(abs(r - results[0]) < atol for r in results)
    elif isinstance(results[0], np.ndarray):
        return all(np.allclose(r, results[0], atol=atol) for r in results)
    return True


def check_gradient_finite(
    func: Callable,
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> bool:
    """Check that numerical gradient of func at x is finite."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
    return np.all(np.isfinite(grad))


# =========================================================================
# Performance Tracking
# =========================================================================


class PerformanceTracker:
    """Track execution time and memory for performance tests."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)

    def time_it(self, name: str):
        """Context manager for timing a block."""
        class _Timer:
            def __init__(self_inner, tracker, key):
                self_inner.tracker = tracker
                self_inner.key = key
                self_inner.start = None

            def __enter__(self_inner):
                self_inner.start = time.time()
                return self_inner

            def __exit__(self_inner, *args):
                elapsed = time.time() - self_inner.start
                self_inner.tracker.timings[self_inner.key].append(elapsed)

        return _Timer(self, name)

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for name, times in self.timings.items():
            result[name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "n": len(times),
            }
        return result


@pytest.fixture
def perf_tracker():
    """Performance tracker fixture."""
    return PerformanceTracker()


# =========================================================================
# Data Validation Helpers
# =========================================================================


def validate_generation_output(
    texts: List[str],
    min_texts: int = 1,
    max_texts: int = 1000,
    min_length: int = 1,
    max_length: int = 10000,
) -> List[str]:
    """Validate generation output and return list of issues."""
    issues = []
    if len(texts) < min_texts:
        issues.append(f"Too few texts: {len(texts)} < {min_texts}")
    if len(texts) > max_texts:
        issues.append(f"Too many texts: {len(texts)} > {max_texts}")
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            issues.append(f"Text {i} is not a string: {type(text)}")
        elif len(text) < min_length:
            issues.append(f"Text {i} too short: {len(text)} < {min_length}")
        elif len(text) > max_length:
            issues.append(f"Text {i} too long: {len(text)} > {max_length}")
    return issues


def validate_metric_output(
    value: Any,
    expected_type: type = float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> List[str]:
    """Validate metric output and return list of issues."""
    issues = []
    if not isinstance(value, (int, float, np.floating)):
        issues.append(f"Wrong type: {type(value)}, expected numeric")
        return issues
    if not np.isfinite(value):
        issues.append(f"Non-finite value: {value}")
    if min_val is not None and value < min_val:
        issues.append(f"Value {value} below minimum {min_val}")
    if max_val is not None and value > max_val:
        issues.append(f"Value {value} above maximum {max_val}")
    return issues


# =========================================================================
# Configuration Helpers
# =========================================================================


def create_minimal_config(**overrides) -> Dict[str, Any]:
    """Create a minimal valid configuration dictionary."""
    config = {
        "algorithm_name": "temperature",
        "num_sequences": 5,
        "max_new_tokens": 10,
        "min_new_tokens": 1,
        "seed": DEFAULT_SEED,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "params": {},
    }
    config.update(overrides)
    return config


def create_sweep_configs(
    algorithm: str,
    param_name: str,
    param_values: List[Any],
    base_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create a list of configs for a parameter sweep."""
    if base_config is None:
        base_config = create_minimal_config(algorithm_name=algorithm)
    configs = []
    for val in param_values:
        cfg = copy.deepcopy(base_config)
        if param_name in cfg:
            cfg[param_name] = val
        else:
            cfg["params"][param_name] = val
        configs.append(cfg)
    return configs
