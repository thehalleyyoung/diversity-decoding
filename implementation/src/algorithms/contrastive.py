"""
Contrastive Search for the Diversity Decoding Arena.
====================================================

Implements contrastive search (Su et al., 2022) — a deterministic decoding
strategy that balances model confidence (log-probability) with a
*degeneration penalty* derived from the similarity between the candidate
token's embedding and the embeddings of previously generated tokens.  At
each step the algorithm:

1. Selects the top-*k* candidate tokens by probability.
2. For every candidate, computes a degeneration penalty equal to the
   maximum (or mean / weighted) cosine similarity between its embedding
   and all context token embeddings.
3. Combines both signals via::

       score = (1 − α) · log_prob  +  α · (1 − penalty)

4. Greedily picks the candidate with the highest combined score.

By penalising tokens whose embeddings cluster with recent context, contrastive
search dramatically reduces the repetitive *degeneration* observed in greedy
and beam-search decoding while staying deterministic and avoiding the
incoherence of high-temperature sampling.

Key components
--------------
* **ContrastiveConfig** — dataclass with contrastive-specific hyper-parameters
  (``alpha``, ``k``, ``embedding_dim``, ``similarity_metric``, …).
* **ContrastiveSearch** — the main ``DecodingAlgorithm`` implementation.
* **ContrastiveScorer** — isolated scoring logic (degeneration penalties,
  combined scores).
* **EmbeddingManager** — embedding look-up / creation / similarity utilities.
* **ContrastiveAnalyzer** — sensitivity and diversity analysis helpers.
* **Helper functions** — ``contrastive_score``, ``batch_cosine_similarity``,
  ``build_context_embeddings``.

References
----------
- Su, Y., Lan, T., Wang, Y., Yogatama, D., Kong, L., & Collier, N. (2022).
  *A Contrastive Framework for Neural Text Generation*.  NeurIPS 2022.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence as SequenceT,
    Tuple,
    Type,
    Union,
)

import numpy as np
from scipy.spatial.distance import cdist, cosine as _scipy_cosine

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    TokenSequence,
    LogitSource,
    _log_softmax,
    _stable_softmax,
    _top_k_filter,
    AlgorithmRegistry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEG_INF: float = float("-inf")
_EPS: float = 1e-10
_DEFAULT_VOCAB_SIZE: int = 50_257  # GPT-2 default


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class ContrastiveConfig(DecodingConfig):
    """Configuration for contrastive search.

    Attributes
    ----------
    alpha : float
        Weight for the degeneration penalty.  ``alpha = 0`` recovers greedy
        decoding; ``alpha = 1`` ignores model confidence entirely and only
        penalises degeneration.
    k : int
        Number of top-*k* candidates to evaluate at each step.
    embedding_dim : int
        Dimensionality of token embeddings.  Used when generating fallback
        random embeddings.
    similarity_metric : str
        Which similarity metric to use for the degeneration penalty.
        One of ``"cosine"``, ``"euclidean"``, ``"dot"``.
    use_token_embeddings : bool
        If *True*, attempt to use the model's own token embedding matrix.
        If *False* (or if the model has no accessible embeddings), fall back
        to random embeddings.
    degeneration_penalty : str
        Aggregation strategy for the degeneration penalty across context
        tokens.  One of ``"max"`` (original paper), ``"mean"``,
        ``"weighted"`` (linearly increasing weights for more recent tokens).
    temperature : float
        Temperature applied to logits before computing log-probabilities.
    length_penalty : float
        Multiplicative penalty on score based on generated length.
        ``1.0`` means no penalty; values > 1 favour longer sequences.
    """

    algorithm_name: str = "ContrastiveSearch"
    alpha: float = 0.6
    k: int = 5
    embedding_dim: int = 768
    similarity_metric: str = "cosine"
    use_token_embeddings: bool = True
    degeneration_penalty: str = "max"
    temperature: float = 1.0
    length_penalty: float = 1.0

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Extend base validation with contrastive-specific checks."""
        errors = super().validate()
        if not 0.0 <= self.alpha <= 1.0:
            errors.append("alpha must be in [0, 1]")
        if self.k < 1:
            errors.append("k must be >= 1")
        if self.embedding_dim < 1:
            errors.append("embedding_dim must be >= 1")
        if self.similarity_metric not in ("cosine", "euclidean", "dot"):
            errors.append(
                f"similarity_metric must be one of cosine, euclidean, dot; "
                f"got {self.similarity_metric!r}"
            )
        if self.degeneration_penalty not in ("max", "mean", "weighted"):
            errors.append(
                f"degeneration_penalty must be one of max, mean, weighted; "
                f"got {self.degeneration_penalty!r}"
            )
        if self.length_penalty <= 0:
            errors.append("length_penalty must be > 0")
        return errors


# =========================================================================
# EmbeddingManager
# =========================================================================


class EmbeddingManager:
    """Manages token embeddings for contrastive scoring.

    The manager either loads pre-computed embeddings (keyed by model name)
    or generates random fallback embeddings.  It also provides utilities for
    similarity computation and nearest-neighbour look-up.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embedding_dim : int
        Dimensionality of each embedding vector.
    """

    def __init__(self, vocab_size: int = _DEFAULT_VOCAB_SIZE, embedding_dim: int = 768) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self._embedding_matrix: Optional[np.ndarray] = None
        self._model_name: Optional[str] = None
        self._norm_cache: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def load_embeddings(self, model_name: str = "random") -> np.ndarray:
        """Load (or generate) an embedding matrix.

        Parameters
        ----------
        model_name : str
            Model identifier.  If ``"random"`` or if no real embeddings can
            be located, falls back to random unit-norm embeddings.

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size, embedding_dim)``.
        """
        if self._embedding_matrix is not None and self._model_name == model_name:
            return self._embedding_matrix

        self._model_name = model_name

        # In a real integration you would load from e.g. HuggingFace here.
        # We always use the random fallback for the Arena's model-agnostic
        # environment.
        logger.info(
            "EmbeddingManager: generating random embeddings "
            "(vocab=%d, dim=%d) for model=%r",
            self.vocab_size,
            self.embedding_dim,
            model_name,
        )
        self._embedding_matrix = self._random_embeddings(
            self.vocab_size, self.embedding_dim
        )
        self._norm_cache = None
        return self._embedding_matrix

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Return the embedding vector for a single token.

        Parameters
        ----------
        token_id : int
            Token index in ``[0, vocab_size)``.

        Returns
        -------
        np.ndarray
            Shape ``(embedding_dim,)``.
        """
        mat = self._ensure_matrix()
        token_id = int(np.clip(token_id, 0, self.vocab_size - 1))
        return mat[token_id].copy()

    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Return embeddings for multiple tokens.

        Parameters
        ----------
        token_ids : list of int

        Returns
        -------
        np.ndarray
            Shape ``(len(token_ids), embedding_dim)``.
        """
        mat = self._ensure_matrix()
        ids = np.clip(token_ids, 0, self.vocab_size - 1).astype(int)
        return mat[ids].copy()

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity for a set of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, embedding_dim)``.

        Returns
        -------
        np.ndarray
            Shape ``(n, n)`` with similarities in ``[-1, 1]``.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, _EPS)
        normed = embeddings / norms
        return normed @ normed.T

    def nearest_neighbors(
        self, embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find the *k* nearest neighbours in the full embedding matrix.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding of shape ``(embedding_dim,)``.
        k : int
            Number of neighbours.

        Returns
        -------
        list of (token_id, similarity)
            Sorted by descending similarity.
        """
        mat = self._ensure_matrix()
        sims = self._batch_cosine(embedding.reshape(1, -1), mat).ravel()
        k = min(k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(int(i), float(sims[i])) for i in top_idx]

    # -- internals ----------------------------------------------------------

    def _ensure_matrix(self) -> np.ndarray:
        """Lazily initialise the embedding matrix if needed."""
        if self._embedding_matrix is None:
            self.load_embeddings("random")
        assert self._embedding_matrix is not None
        return self._embedding_matrix

    def _random_embeddings(self, vocab_size: int, dim: int) -> np.ndarray:
        """Generate random unit-norm embeddings.

        Uses a fixed seed so that the same ``(vocab_size, dim)`` always
        produces the same matrix, ensuring reproducibility.

        Parameters
        ----------
        vocab_size : int
        dim : int

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size, dim)`` with each row having unit L2 norm.
        """
        rng = np.random.default_rng(seed=42)
        raw = rng.standard_normal((vocab_size, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, _EPS)
        return raw / norms

    @staticmethod
    def _batch_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between rows of *a* and rows of *b*.

        Parameters
        ----------
        a : np.ndarray  — shape ``(m, d)``
        b : np.ndarray  — shape ``(n, d)``

        Returns
        -------
        np.ndarray  — shape ``(m, n)``
        """
        a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
        return a_norm @ b_norm.T


# =========================================================================
# ContrastiveScorer
# =========================================================================


class ContrastiveScorer:
    """Encapsulates degeneration-penalty and combined-score computations.

    This is deliberately stateless so that it can be used from both the main
    ``ContrastiveSearch`` algorithm and from the ``ContrastiveAnalyzer``.

    Parameters
    ----------
    similarity_metric : str
        ``"cosine"`` | ``"euclidean"`` | ``"dot"``
    penalty_mode : str
        ``"max"`` | ``"mean"`` | ``"weighted"``
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        penalty_mode: str = "max",
    ) -> None:
        self.similarity_metric = similarity_metric
        self.penalty_mode = penalty_mode

    # -- single-candidate scoring ------------------------------------------

    def score_candidate(
        self,
        candidate_id: int,
        context_embeddings: np.ndarray,
        embedding_matrix: np.ndarray,
        alpha: float,
        log_prob: float,
    ) -> float:
        """Compute the contrastive score for a single candidate token.

        Parameters
        ----------
        candidate_id : int
            Token index of the candidate.
        context_embeddings : np.ndarray
            Shape ``(context_len, dim)`` — embeddings of previously generated
            tokens.
        embedding_matrix : np.ndarray
            Full embedding matrix ``(vocab_size, dim)``.
        alpha : float
            Degeneration penalty weight.
        log_prob : float
            Log-probability of the candidate under the model.

        Returns
        -------
        float
            Combined contrastive score.
        """
        candidate_id = int(np.clip(candidate_id, 0, embedding_matrix.shape[0] - 1))
        candidate_emb = embedding_matrix[candidate_id]

        if context_embeddings.shape[0] == 0:
            # No context yet — penalty is 0
            penalty = 0.0
        else:
            penalty = self._compute_penalty(candidate_emb, context_embeddings)

        return self._combine_score(log_prob, penalty, alpha)

    def batch_score_candidates(
        self,
        candidate_ids: np.ndarray,
        context_embeddings: np.ndarray,
        embedding_matrix: np.ndarray,
        alpha: float,
        log_probs: np.ndarray,
    ) -> np.ndarray:
        """Score multiple candidate tokens at once.

        Parameters
        ----------
        candidate_ids : np.ndarray
            Shape ``(k,)`` — token indices.
        context_embeddings : np.ndarray
            Shape ``(context_len, dim)``.
        embedding_matrix : np.ndarray
            Shape ``(vocab_size, dim)``.
        alpha : float
        log_probs : np.ndarray
            Shape ``(k,)`` — log-probabilities for each candidate.

        Returns
        -------
        np.ndarray
            Shape ``(k,)`` — contrastive scores.
        """
        ids = np.clip(candidate_ids, 0, embedding_matrix.shape[0] - 1).astype(int)
        cand_embs = embedding_matrix[ids]  # (k, dim)

        if context_embeddings.shape[0] == 0:
            penalties = np.zeros(len(ids), dtype=np.float64)
        else:
            penalties = self._batch_penalty(cand_embs, context_embeddings)

        scores = (1.0 - alpha) * log_probs + alpha * (1.0 - penalties)
        return scores

    # -- degeneration penalties --------------------------------------------

    def degeneration_penalty_max(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """Max-similarity degeneration penalty (original paper).

        Parameters
        ----------
        candidate_emb : np.ndarray  — shape ``(dim,)``
        context_embs : np.ndarray   — shape ``(n, dim)``

        Returns
        -------
        float
            Maximum similarity between the candidate and any context token.
        """
        sims = self._similarities(candidate_emb, context_embs)
        return float(np.max(sims)) if sims.size > 0 else 0.0

    def degeneration_penalty_mean(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """Mean-similarity degeneration penalty.

        Parameters
        ----------
        candidate_emb : np.ndarray  — shape ``(dim,)``
        context_embs : np.ndarray   — shape ``(n, dim)``

        Returns
        -------
        float
            Mean similarity between the candidate and all context tokens.
        """
        sims = self._similarities(candidate_emb, context_embs)
        return float(np.mean(sims)) if sims.size > 0 else 0.0

    def degeneration_penalty_weighted(
        self,
        candidate_emb: np.ndarray,
        context_embs: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Weighted-similarity degeneration penalty.

        By default uses linearly increasing weights so that more recent
        context tokens are penalised more heavily.

        Parameters
        ----------
        candidate_emb : np.ndarray  — shape ``(dim,)``
        context_embs : np.ndarray   — shape ``(n, dim)``
        weights : np.ndarray or None
            Optional weight vector of shape ``(n,)``.  If *None*, uses
            linear ramp ``[1, 2, …, n] / sum``.

        Returns
        -------
        float
            Weighted similarity between the candidate and context tokens.
        """
        sims = self._similarities(candidate_emb, context_embs)
        if sims.size == 0:
            return 0.0
        if weights is None:
            n = len(sims)
            weights = np.arange(1, n + 1, dtype=np.float64)
            weights /= weights.sum()
        else:
            weights = np.asarray(weights, dtype=np.float64)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones_like(weights) / len(weights)
        return float(np.dot(sims, weights))

    # -- internals ----------------------------------------------------------

    def _compute_penalty(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """Dispatch to the configured penalty mode."""
        if self.penalty_mode == "max":
            return self.degeneration_penalty_max(candidate_emb, context_embs)
        elif self.penalty_mode == "mean":
            return self.degeneration_penalty_mean(candidate_emb, context_embs)
        elif self.penalty_mode == "weighted":
            return self.degeneration_penalty_weighted(candidate_emb, context_embs)
        else:
            return self.degeneration_penalty_max(candidate_emb, context_embs)

    def _batch_penalty(
        self, cand_embs: np.ndarray, context_embs: np.ndarray
    ) -> np.ndarray:
        """Compute degeneration penalties for a batch of candidates.

        Parameters
        ----------
        cand_embs : np.ndarray   — shape ``(k, dim)``
        context_embs : np.ndarray — shape ``(n, dim)``

        Returns
        -------
        np.ndarray — shape ``(k,)``
        """
        sim_matrix = self._similarity_matrix(cand_embs, context_embs)  # (k, n)

        if self.penalty_mode == "max":
            return np.max(sim_matrix, axis=1)
        elif self.penalty_mode == "mean":
            return np.mean(sim_matrix, axis=1)
        elif self.penalty_mode == "weighted":
            n = context_embs.shape[0]
            weights = np.arange(1, n + 1, dtype=np.float64)
            weights /= weights.sum()
            return sim_matrix @ weights
        else:
            return np.max(sim_matrix, axis=1)

    def _similarities(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> np.ndarray:
        """Compute similarities between one candidate and all context tokens.

        Parameters
        ----------
        candidate_emb : np.ndarray  — shape ``(dim,)``
        context_embs : np.ndarray   — shape ``(n, dim)``

        Returns
        -------
        np.ndarray — shape ``(n,)``
        """
        if self.similarity_metric == "cosine":
            return self._cosine_similarities(candidate_emb, context_embs)
        elif self.similarity_metric == "euclidean":
            dists = self._euclidean_distances(candidate_emb, context_embs)
            # Convert distance to similarity in [0, 1]
            return 1.0 / (1.0 + dists)
        elif self.similarity_metric == "dot":
            return self._dot_similarities(candidate_emb, context_embs)
        else:
            return self._cosine_similarities(candidate_emb, context_embs)

    def _similarity_matrix(
        self, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Compute similarity matrix between rows of *a* and rows of *b*.

        Parameters
        ----------
        a : np.ndarray — shape ``(m, dim)``
        b : np.ndarray — shape ``(n, dim)``

        Returns
        -------
        np.ndarray — shape ``(m, n)``
        """
        if self.similarity_metric == "cosine":
            a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
            b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
            return a_n @ b_n.T
        elif self.similarity_metric == "euclidean":
            dists = cdist(a, b, metric="euclidean")
            return 1.0 / (1.0 + dists)
        elif self.similarity_metric == "dot":
            return a @ b.T
        else:
            a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
            b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
            return a_n @ b_n.T

    @staticmethod
    def _cosine_similarities(
        vec: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Cosine similarities between a vector and each row of a matrix.

        Parameters
        ----------
        vec : np.ndarray    — shape ``(dim,)``
        matrix : np.ndarray — shape ``(n, dim)``

        Returns
        -------
        np.ndarray — shape ``(n,)``
        """
        vec_norm = np.linalg.norm(vec)
        if vec_norm < _EPS:
            return np.zeros(matrix.shape[0])
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms = np.maximum(row_norms, _EPS)
        return (matrix @ vec) / (row_norms * vec_norm)

    @staticmethod
    def _euclidean_distances(
        vec: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Euclidean distances between a vector and each row of a matrix.

        Parameters
        ----------
        vec : np.ndarray    — shape ``(dim,)``
        matrix : np.ndarray — shape ``(n, dim)``

        Returns
        -------
        np.ndarray — shape ``(n,)``
        """
        diff = matrix - vec[np.newaxis, :]
        return np.linalg.norm(diff, axis=1)

    @staticmethod
    def _dot_similarities(
        vec: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Dot-product similarities.

        Parameters
        ----------
        vec : np.ndarray    — shape ``(dim,)``
        matrix : np.ndarray — shape ``(n, dim)``

        Returns
        -------
        np.ndarray — shape ``(n,)``
        """
        return matrix @ vec

    @staticmethod
    def _combine_score(log_prob: float, penalty: float, alpha: float) -> float:
        """Combine model confidence and degeneration penalty.

        Parameters
        ----------
        log_prob : float
            Log-probability of the candidate.
        penalty : float
            Degeneration penalty (similarity value in [0, 1] typically).
        alpha : float
            Mixing weight.

        Returns
        -------
        float
            ``(1 - alpha) * log_prob + alpha * (1 - penalty)``
        """
        return (1.0 - alpha) * log_prob + alpha * (1.0 - penalty)


# =========================================================================
# ContrastiveSearch  (main DecodingAlgorithm)
# =========================================================================


class ContrastiveSearch(DecodingAlgorithm):
    """Contrastive search decoding algorithm (Su et al., 2022).

    At each decoding step the algorithm picks the top-*k* candidates by
    probability, scores them with a contrastive objective that balances
    log-probability against an embedding-space degeneration penalty, and
    deterministically selects the highest-scoring candidate.

    Parameters
    ----------
    config : ContrastiveConfig
        Algorithm configuration.
    """

    def __init__(self, config: Optional[ContrastiveConfig] = None) -> None:
        if config is None:
            config = ContrastiveConfig()
        self._cs_config: ContrastiveConfig = config
        super().__init__(config)

        # Scorer
        self._scorer = ContrastiveScorer(
            similarity_metric=config.similarity_metric,
            penalty_mode=config.degeneration_penalty,
        )

        # Embedding manager — lazily initialised
        self._emb_manager = EmbeddingManager(
            vocab_size=_DEFAULT_VOCAB_SIZE,
            embedding_dim=config.embedding_dim,
        )
        self._embedding_matrix: Optional[np.ndarray] = None

        # Per-sequence context embedding caches (populated during generation)
        self._context_embeddings: Dict[int, List[np.ndarray]] = {}

    # -- properties ---------------------------------------------------------

    @property
    def description(self) -> str:
        return (
            f"Contrastive search (alpha={self._cs_config.alpha}, "
            f"k={self._cs_config.k}, "
            f"metric={self._cs_config.similarity_metric})"
        )

    # -- public generate (overrides base for multi-sequence independence) ---

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate ``num_sequences`` continuations using contrastive search.

        Each sequence is generated independently with its own context
        embedding history.  Because contrastive search is deterministic for
        a given prompt, diversity across sequences is achieved by introducing
        a small per-sequence temperature jitter controlled by the seed.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        self._ensure_embeddings()

        sequences: List[TokenSequence] = []
        for seq_idx in range(self.config.num_sequences):
            seq = self._generate_single_sequence(logit_source, prompt_ids, seq_idx)
            sequences.append(seq)

        # Sort by length (longer = better for deterministic search)
        sequences.sort(key=len, reverse=True)
        return sequences

    # -- abstract _step implementation --------------------------------------

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute one contrastive-search step for all active sequences.

        For each active sequence:
        1.  Obtain logits from ``logit_source``.
        2.  Apply constraints (repetition penalty, n-gram blocking, …).
        3.  Compute log-probabilities and extract top-*k* candidates.
        4.  For each candidate compute the degeneration penalty.
        5.  Select the candidate with the best contrastive score.
        6.  Append the selected token and update context embeddings.

        Parameters
        ----------
        state : DecodingState
        logit_source : LogitSource

        Returns
        -------
        DecodingState
        """
        self._ensure_embeddings()
        assert self._embedding_matrix is not None

        alpha = self._cs_config.alpha
        k = self._cs_config.k
        temperature = self._cs_config.temperature

        active = state.active_indices()
        if not active:
            return state

        # Gather prefixes for active sequences
        prefixes = [state.sequences[i] for i in active]
        all_logits = logit_source(prefixes)  # (batch, vocab)

        for batch_idx, seq_idx in enumerate(active):
            logits = all_logits[batch_idx].copy()
            logits = self._apply_constraints(logits, state)

            # Temperature
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # Log-probabilities
            log_probs = _log_softmax(logits)

            # Top-k candidates
            actual_k = min(k, logits.shape[0])
            top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
            top_k_log_probs = log_probs[top_k_indices]

            # Context embeddings for this sequence
            ctx_embs = self._get_context_embeddings(seq_idx, state.sequences[seq_idx])

            # Score all candidates
            if ctx_embs.shape[0] == 0 or alpha == 0.0:
                # No context or alpha=0 ⇒ greedy
                best_local = int(np.argmax(top_k_log_probs))
                selected_token = int(top_k_indices[best_local])
                selected_score = float(top_k_log_probs[best_local])
            else:
                scores = self._scorer.batch_score_candidates(
                    candidate_ids=top_k_indices,
                    context_embeddings=ctx_embs,
                    embedding_matrix=self._embedding_matrix,
                    alpha=alpha,
                    log_probs=top_k_log_probs,
                )
                best_local = int(np.argmax(scores))
                selected_token = int(top_k_indices[best_local])
                selected_score = float(scores[best_local])

            # Update state
            state.update_sequence(seq_idx, selected_token)
            state.scores[seq_idx] += selected_score

            # Update context embedding cache
            self._append_context_embedding(seq_idx, selected_token)

            # Check EOS
            if (
                self.config.eos_token_id is not None
                and selected_token == self.config.eos_token_id
            ):
                state.mark_finished(seq_idx)

        # Apply length penalty to scores
        if self._cs_config.length_penalty != 1.0:
            prompt_len = state.metadata.get("prompt_length", 0)
            for i in active:
                gen_len = len(state.sequences[i]) - prompt_len
                if gen_len > 0:
                    lp = ((5.0 + gen_len) / 6.0) ** self._cs_config.length_penalty
                    state.scores[i] = state.scores[i] / lp

        return state

    # -- single-sequence generation -----------------------------------------

    def _generate_single_sequence(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        seq_idx: int = 0,
    ) -> TokenSequence:
        """Generate one complete sequence using contrastive search.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        seq_idx : int
            Sequence index — used for per-sequence temperature jitter.

        Returns
        -------
        TokenSequence
            Generated tokens (excluding the prompt).
        """
        assert self._embedding_matrix is not None

        alpha = self._cs_config.alpha
        k = self._cs_config.k
        temperature = self._cs_config.temperature

        # Small temperature jitter for diversity across sequences
        if seq_idx > 0 and self._rng is not None:
            jitter = self._rng.uniform(-0.05, 0.05)
            temperature = max(0.1, temperature + jitter * seq_idx)

        current_ids = list(prompt_ids)
        context_emb_list: List[np.ndarray] = []

        # Build initial context embeddings from prompt
        if len(prompt_ids) > 0:
            prompt_embs = self._get_token_embeddings(prompt_ids)
            context_emb_list = [prompt_embs[i] for i in range(prompt_embs.shape[0])]

        generated: List[int] = []

        for step in range(self.config.max_new_tokens):
            # Get logits
            logits = logit_source([current_ids])  # (1, vocab)
            logits = logits[0].copy()

            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            log_probs = _log_softmax(logits)

            # Top-k candidates
            actual_k = min(k, logits.shape[0])
            top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
            top_k_log_probs = log_probs[top_k_indices]

            # Context embeddings
            if len(context_emb_list) == 0 or alpha == 0.0:
                best_local = int(np.argmax(top_k_log_probs))
                selected_token = int(top_k_indices[best_local])
            else:
                ctx_embs = np.stack(context_emb_list, axis=0)
                scores = self._scorer.batch_score_candidates(
                    candidate_ids=top_k_indices,
                    context_embeddings=ctx_embs,
                    embedding_matrix=self._embedding_matrix,
                    alpha=alpha,
                    log_probs=top_k_log_probs,
                )
                best_local = int(np.argmax(scores))
                selected_token = int(top_k_indices[best_local])

            # Append
            current_ids.append(selected_token)
            generated.append(selected_token)
            context_emb_list.append(self._embedding_matrix[selected_token].copy())

            # Check stopping conditions
            if (
                self.config.eos_token_id is not None
                and selected_token == self.config.eos_token_id
            ):
                break
            if len(generated) >= self.config.max_new_tokens:
                break

        return generated

    # -- embedding helpers --------------------------------------------------

    def _ensure_embeddings(self) -> None:
        """Lazily load/create the embedding matrix."""
        if self._embedding_matrix is None:
            self._embedding_matrix = self._emb_manager.load_embeddings("random")

    def _get_token_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Look up embeddings for a list of token ids.

        Parameters
        ----------
        token_ids : list of int

        Returns
        -------
        np.ndarray
            Shape ``(len(token_ids), embedding_dim)``.
        """
        self._ensure_embeddings()
        assert self._embedding_matrix is not None
        ids = np.clip(token_ids, 0, self._embedding_matrix.shape[0] - 1).astype(int)
        return self._embedding_matrix[ids]

    def _build_embedding_matrix(self, token_ids: List[int]) -> np.ndarray:
        """Build a sub-matrix of embeddings for specific token ids.

        Useful for analysing a particular set of tokens without loading the
        full vocabulary embedding matrix.

        Parameters
        ----------
        token_ids : list of int

        Returns
        -------
        np.ndarray
            Shape ``(len(token_ids), embedding_dim)``.
        """
        return self._get_token_embeddings(token_ids)

    def _get_context_embeddings(
        self, seq_idx: int, sequence: List[int]
    ) -> np.ndarray:
        """Retrieve or build context embeddings for a sequence.

        Parameters
        ----------
        seq_idx : int
            Index of the sequence in the state.
        sequence : list of int
            Full token sequence (prompt + generated so far).

        Returns
        -------
        np.ndarray
            Shape ``(context_len, embedding_dim)``.
        """
        if seq_idx in self._context_embeddings and self._context_embeddings[seq_idx]:
            return np.stack(self._context_embeddings[seq_idx], axis=0)
        # Build from scratch
        if len(sequence) == 0:
            return np.empty((0, self._cs_config.embedding_dim), dtype=np.float32)
        embs = self._get_token_embeddings(sequence)
        self._context_embeddings[seq_idx] = [embs[i] for i in range(embs.shape[0])]
        return embs

    def _append_context_embedding(self, seq_idx: int, token_id: int) -> None:
        """Append a single token's embedding to the context cache.

        Parameters
        ----------
        seq_idx : int
        token_id : int
        """
        assert self._embedding_matrix is not None
        tid = int(np.clip(token_id, 0, self._embedding_matrix.shape[0] - 1))
        if seq_idx not in self._context_embeddings:
            self._context_embeddings[seq_idx] = []
        self._context_embeddings[seq_idx].append(self._embedding_matrix[tid].copy())

    # -- similarity helpers -------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors.

        Parameters
        ----------
        a : np.ndarray — shape ``(dim,)``
        b : np.ndarray — shape ``(dim,)``

        Returns
        -------
        float
            Similarity in ``[-1, 1]``.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < _EPS or norm_b < _EPS:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two vectors.

        Parameters
        ----------
        a : np.ndarray — shape ``(dim,)``
        b : np.ndarray — shape ``(dim,)``

        Returns
        -------
        float
            Non-negative distance.
        """
        return float(np.linalg.norm(a - b))

    def _compute_degeneration_penalty(
        self,
        candidate_embedding: np.ndarray,
        context_embeddings: np.ndarray,
    ) -> float:
        """Compute degeneration penalty for a candidate token.

        Delegates to the configured scorer.

        Parameters
        ----------
        candidate_embedding : np.ndarray — shape ``(dim,)``
        context_embeddings : np.ndarray  — shape ``(n, dim)``

        Returns
        -------
        float
            Penalty value (higher means more degenerate).
        """
        return self._scorer._compute_penalty(candidate_embedding, context_embeddings)

    def _compute_contrastive_score(
        self, log_prob: float, penalty: float, alpha: float
    ) -> float:
        """Compute the combined contrastive score.

        Parameters
        ----------
        log_prob : float
        penalty : float
        alpha : float

        Returns
        -------
        float
            ``(1 - alpha) * log_prob + alpha * (1 - penalty)``
        """
        return ContrastiveScorer._combine_score(log_prob, penalty, alpha)

    # -- introspection ------------------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        """Describe the hyper-parameter search space."""
        base = super().hyperparameter_space()
        base.update(
            {
                "alpha": {"type": "float", "low": 0.0, "high": 1.0},
                "k": {"type": "int", "low": 1, "high": 50},
                "similarity_metric": {
                    "type": "categorical",
                    "choices": ["cosine", "euclidean", "dot"],
                },
                "degeneration_penalty": {
                    "type": "categorical",
                    "choices": ["max", "mean", "weighted"],
                },
            }
        )
        return base

    def validate_config(self) -> List[str]:
        """Validate contrastive-specific configuration."""
        return self._cs_config.validate()

    # -- init state override ------------------------------------------------

    def _init_state(self, prompt_ids: List[int]) -> DecodingState:
        """Initialise state and context embedding caches."""
        self._context_embeddings.clear()
        state = self._prepare_generation(prompt_ids)
        # Pre-populate context embeddings from the prompt
        self._ensure_embeddings()
        for i in range(state.num_sequences):
            if len(prompt_ids) > 0:
                embs = self._get_token_embeddings(prompt_ids)
                self._context_embeddings[i] = [
                    embs[j] for j in range(embs.shape[0])
                ]
            else:
                self._context_embeddings[i] = []
        return state


# =========================================================================
# ContrastiveAnalyzer
# =========================================================================


class ContrastiveAnalyzer:
    """Analysis utilities for contrastive search outputs.

    Provides tools to study the effect of hyper-parameters (alpha, k) on
    generation quality and diversity, as well as metrics for repetition
    and degeneration.

    Parameters
    ----------
    config : ContrastiveConfig or None
        Base configuration.  If *None*, uses defaults.
    """

    def __init__(self, config: Optional[ContrastiveConfig] = None) -> None:
        self.config = config or ContrastiveConfig()
        self._scorer = ContrastiveScorer(
            similarity_metric=self.config.similarity_metric,
            penalty_mode=self.config.degeneration_penalty,
        )
        self._emb_manager = EmbeddingManager(
            vocab_size=_DEFAULT_VOCAB_SIZE,
            embedding_dim=self.config.embedding_dim,
        )

    # -- alpha sensitivity --------------------------------------------------

    def analyze_alpha_sensitivity(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        alphas: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate how different alpha values affect generation.

        For each alpha value, generates sequences and measures repetition
        rate, mean degeneration score, and average sequence length.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        alphas : list of float or None
            Alpha values to sweep.  Defaults to ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]``.

        Returns
        -------
        dict
            ``{"alphas": [...], "repetition_rates": [...],
              "degeneration_scores": [...], "mean_lengths": [...]}``
        """
        if alphas is None:
            alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        results: Dict[str, List[Any]] = {
            "alphas": list(alphas),
            "repetition_rates": [],
            "degeneration_scores": [],
            "mean_lengths": [],
        }

        for alpha in alphas:
            cfg = ContrastiveConfig(
                alpha=alpha,
                k=self.config.k,
                embedding_dim=self.config.embedding_dim,
                similarity_metric=self.config.similarity_metric,
                degeneration_penalty=self.config.degeneration_penalty,
                temperature=self.config.temperature,
                num_sequences=max(1, self.config.num_sequences // 2),
                max_new_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )
            algo = ContrastiveSearch(cfg)
            try:
                seqs = algo.generate(logit_source, prompt_ids)
            except Exception as e:
                logger.warning("Alpha sensitivity: failed for alpha=%.2f: %s", alpha, e)
                results["repetition_rates"].append(float("nan"))
                results["degeneration_scores"].append(float("nan"))
                results["mean_lengths"].append(0.0)
                continue

            results["repetition_rates"].append(self.repetition_rate(seqs))
            results["degeneration_scores"].append(self.degeneration_score(seqs))
            lengths = [len(s) for s in seqs]
            results["mean_lengths"].append(
                float(np.mean(lengths)) if lengths else 0.0
            )

        return results

    # -- k sensitivity ------------------------------------------------------

    def analyze_k_sensitivity(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate how different *k* values affect generation.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        k_values : list of int or None
            Values of *k* to sweep.  Defaults to ``[1, 3, 5, 10, 20, 50]``.

        Returns
        -------
        dict
            ``{"k_values": [...], "repetition_rates": [...],
              "degeneration_scores": [...], "mean_lengths": [...]}``
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 50]

        results: Dict[str, List[Any]] = {
            "k_values": list(k_values),
            "repetition_rates": [],
            "degeneration_scores": [],
            "mean_lengths": [],
        }

        for k_val in k_values:
            cfg = ContrastiveConfig(
                alpha=self.config.alpha,
                k=k_val,
                embedding_dim=self.config.embedding_dim,
                similarity_metric=self.config.similarity_metric,
                degeneration_penalty=self.config.degeneration_penalty,
                temperature=self.config.temperature,
                num_sequences=max(1, self.config.num_sequences // 2),
                max_new_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )
            algo = ContrastiveSearch(cfg)
            try:
                seqs = algo.generate(logit_source, prompt_ids)
            except Exception as e:
                logger.warning("K sensitivity: failed for k=%d: %s", k_val, e)
                results["repetition_rates"].append(float("nan"))
                results["degeneration_scores"].append(float("nan"))
                results["mean_lengths"].append(0.0)
                continue

            results["repetition_rates"].append(self.repetition_rate(seqs))
            results["degeneration_scores"].append(self.degeneration_score(seqs))
            lengths = [len(s) for s in seqs]
            results["mean_lengths"].append(
                float(np.mean(lengths)) if lengths else 0.0
            )

        return results

    # -- diversity vs alpha -------------------------------------------------

    def diversity_vs_alpha(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        alphas: Optional[List[float]] = None,
        n_samples: int = 5,
    ) -> Dict[str, Any]:
        """Measure token-level diversity as a function of alpha.

        For each alpha value, generates ``n_samples`` sequences and computes
        the ratio of unique tokens to total tokens (type-token ratio).

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        alphas : list of float or None
        n_samples : int

        Returns
        -------
        dict
            ``{"alphas": [...], "type_token_ratios": [...],
              "unique_token_counts": [...], "total_token_counts": [...]}``
        """
        if alphas is None:
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        results: Dict[str, List[Any]] = {
            "alphas": list(alphas),
            "type_token_ratios": [],
            "unique_token_counts": [],
            "total_token_counts": [],
        }

        for alpha in alphas:
            cfg = ContrastiveConfig(
                alpha=alpha,
                k=self.config.k,
                embedding_dim=self.config.embedding_dim,
                similarity_metric=self.config.similarity_metric,
                degeneration_penalty=self.config.degeneration_penalty,
                temperature=self.config.temperature,
                num_sequences=n_samples,
                max_new_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )
            algo = ContrastiveSearch(cfg)
            try:
                seqs = algo.generate(logit_source, prompt_ids)
            except Exception as e:
                logger.warning("Diversity vs alpha: failed for alpha=%.2f: %s", alpha, e)
                results["type_token_ratios"].append(float("nan"))
                results["unique_token_counts"].append(0)
                results["total_token_counts"].append(0)
                continue

            all_tokens: List[int] = []
            for s in seqs:
                all_tokens.extend(s)
            total = len(all_tokens)
            unique = len(set(all_tokens))
            ttr = unique / total if total > 0 else 0.0

            results["type_token_ratios"].append(ttr)
            results["unique_token_counts"].append(unique)
            results["total_token_counts"].append(total)

        return results

    # -- metrics ------------------------------------------------------------

    def repetition_rate(self, sequences: List[TokenSequence]) -> float:
        """Compute the fraction of tokens that are repetitions.

        A token at position *t* is a repetition if it appeared in the same
        sequence at any position before *t*.

        Parameters
        ----------
        sequences : list of TokenSequence

        Returns
        -------
        float
            Value in ``[0, 1]``.  Lower is better.
        """
        total = 0
        repeated = 0
        for seq in sequences:
            seen: set = set()
            for tok in seq:
                total += 1
                if tok in seen:
                    repeated += 1
                seen.add(tok)
        return repeated / total if total > 0 else 0.0

    def degeneration_score(self, sequences: List[TokenSequence]) -> float:
        """Compute a degeneration score based on consecutive repeated n-grams.

        Counts the fraction of bigrams that are immediately repeated
        (e.g. ``a b a b`` contains one repeated bigram).

        Parameters
        ----------
        sequences : list of TokenSequence

        Returns
        -------
        float
            Value in ``[0, 1]``.  Lower is better.
        """
        total_bigrams = 0
        repeated_bigrams = 0
        for seq in sequences:
            if len(seq) < 4:
                continue
            for i in range(len(seq) - 3):
                bg1 = (seq[i], seq[i + 1])
                bg2 = (seq[i + 2], seq[i + 3])
                total_bigrams += 1
                if bg1 == bg2:
                    repeated_bigrams += 1
        return repeated_bigrams / total_bigrams if total_bigrams > 0 else 0.0


# =========================================================================
# Helper functions (module-level)
# =========================================================================


def contrastive_score(
    logits: np.ndarray,
    embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    alpha: float = 0.6,
    k: int = 5,
) -> Tuple[int, float]:
    """Compute the contrastive-search selection for a single step.

    Stand-alone function that can be used outside the ``ContrastiveSearch``
    class.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(vocab_size,)`` — raw logits.
    embeddings : np.ndarray
        Shape ``(vocab_size, dim)`` — full embedding matrix.
    context_embeddings : np.ndarray
        Shape ``(context_len, dim)`` — embeddings of previous tokens.
    alpha : float
        Degeneration penalty weight.
    k : int
        Number of top-*k* candidates.

    Returns
    -------
    (token_id, score) : tuple
        The selected token id and its combined contrastive score.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    log_probs = _log_softmax(logits)

    actual_k = min(k, len(log_probs))
    top_k_idx = np.argpartition(log_probs, -actual_k)[-actual_k:]
    top_k_lp = log_probs[top_k_idx]

    if context_embeddings.shape[0] == 0 or alpha == 0.0:
        best = int(np.argmax(top_k_lp))
        return int(top_k_idx[best]), float(top_k_lp[best])

    scorer = ContrastiveScorer(similarity_metric="cosine", penalty_mode="max")
    scores = scorer.batch_score_candidates(
        candidate_ids=top_k_idx,
        context_embeddings=context_embeddings,
        embedding_matrix=embeddings,
        alpha=alpha,
        log_probs=top_k_lp,
    )
    best = int(np.argmax(scores))
    return int(top_k_idx[best]), float(scores[best])


def batch_cosine_similarity(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarities between a single vector and a matrix of vectors.

    Parameters
    ----------
    a : np.ndarray
        Query vector of shape ``(dim,)``.
    B : np.ndarray
        Matrix of shape ``(n, dim)``.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` — cosine similarities.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    norm_a = np.linalg.norm(a)
    if norm_a < _EPS:
        return np.zeros(B.shape[0])
    norms_B = np.linalg.norm(B, axis=1)
    norms_B = np.maximum(norms_B, _EPS)
    return (B @ a) / (norms_B * norm_a)


def build_context_embeddings(
    token_ids: List[int], embedding_matrix: np.ndarray
) -> np.ndarray:
    """Build a context embedding matrix from token ids.

    Parameters
    ----------
    token_ids : list of int
        Token ids of the context.
    embedding_matrix : np.ndarray
        Full embedding matrix of shape ``(vocab_size, dim)``.

    Returns
    -------
    np.ndarray
        Shape ``(len(token_ids), dim)``.  If ``token_ids`` is empty,
        returns an array of shape ``(0, dim)``.
    """
    if len(token_ids) == 0:
        dim = embedding_matrix.shape[1] if embedding_matrix.ndim >= 2 else 1
        return np.empty((0, dim), dtype=embedding_matrix.dtype)
    ids = np.clip(token_ids, 0, embedding_matrix.shape[0] - 1).astype(int)
    return embedding_matrix[ids].copy()


# =========================================================================
# Registry (optional — register with the global AlgorithmRegistry)
# =========================================================================

try:
    AlgorithmRegistry.register("ContrastiveSearch", ContrastiveSearch)
except Exception:
    # Registry may not be available or may reject duplicates — that's fine.
    pass


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("Contrastive Search — self-test")
    print("=" * 70)

    # --- Config ---
    cfg = ContrastiveConfig(
        alpha=0.6,
        k=5,
        embedding_dim=64,
        num_sequences=3,
        max_new_tokens=20,
        seed=42,
    )
    errors = cfg.validate()
    assert not errors, f"Config validation errors: {errors}"
    print("[PASS] ContrastiveConfig validates cleanly")

    # --- EmbeddingManager ---
    emb_mgr = EmbeddingManager(vocab_size=100, embedding_dim=64)
    mat = emb_mgr.load_embeddings("test")
    assert mat.shape == (100, 64), f"Expected (100, 64), got {mat.shape}"
    e = emb_mgr.get_embedding(5)
    assert e.shape == (64,)
    es = emb_mgr.get_embeddings([0, 1, 2])
    assert es.shape == (3, 64)
    sim_mat = emb_mgr.similarity_matrix(es)
    assert sim_mat.shape == (3, 3)
    nn = emb_mgr.nearest_neighbors(e, k=3)
    assert len(nn) == 3
    print("[PASS] EmbeddingManager works")

    # --- ContrastiveScorer ---
    scorer = ContrastiveScorer(similarity_metric="cosine", penalty_mode="max")
    ctx = emb_mgr.get_embeddings([10, 11, 12])
    s = scorer.score_candidate(5, ctx, mat, alpha=0.6, log_prob=-2.0)
    assert isinstance(s, float)
    bs = scorer.batch_score_candidates(
        np.array([5, 6, 7]), ctx, mat, alpha=0.6, log_probs=np.array([-2.0, -3.0, -1.5])
    )
    assert bs.shape == (3,)
    p_max = scorer.degeneration_penalty_max(e, ctx)
    p_mean = scorer.degeneration_penalty_mean(e, ctx)
    p_w = scorer.degeneration_penalty_weighted(e, ctx)
    assert all(isinstance(x, float) for x in [p_max, p_mean, p_w])
    print("[PASS] ContrastiveScorer works")

    # --- Dummy logit source ---
    VOCAB = 100
    rng = np.random.default_rng(123)

    def dummy_logit_source(input_ids: List[List[int]]) -> np.ndarray:
        batch = len(input_ids)
        return rng.standard_normal((batch, VOCAB)).astype(np.float64)

    # --- ContrastiveSearch.generate ---
    algo = ContrastiveSearch(cfg)
    algo._emb_manager = emb_mgr
    algo._embedding_matrix = mat
    seqs = algo.generate(dummy_logit_source, [1, 2, 3])
    assert len(seqs) == 3
    for s in seqs:
        assert isinstance(s, list)
        assert all(isinstance(t, int) for t in s)
    print(f"[PASS] ContrastiveSearch.generate produced {len(seqs)} sequences")
    print(f"       lengths: {[len(s) for s in seqs]}")

    # --- Helper functions ---
    logits = rng.standard_normal(VOCAB)
    ctx_emb = mat[:5]
    tok_id, score = contrastive_score(logits, mat, ctx_emb, alpha=0.6, k=5)
    assert 0 <= tok_id < VOCAB
    print(f"[PASS] contrastive_score -> token={tok_id}, score={score:.4f}")

    bcs = batch_cosine_similarity(e, ctx)
    assert bcs.shape == (3,)
    print(f"[PASS] batch_cosine_similarity -> {bcs}")

    ctx_built = build_context_embeddings([0, 1, 2], mat)
    assert ctx_built.shape == (3, 64)
    print("[PASS] build_context_embeddings works")

    # --- ContrastiveAnalyzer ---
    analyzer = ContrastiveAnalyzer(
        ContrastiveConfig(
            embedding_dim=64,
            num_sequences=2,
            max_new_tokens=10,
            seed=42,
        )
    )
    rr = analyzer.repetition_rate(seqs)
    ds = analyzer.degeneration_score(seqs)
    print(f"[PASS] repetition_rate={rr:.4f}, degeneration_score={ds:.4f}")

    print()
    print("All self-tests passed.")


# =========================================================================
# Multi-Reference Contrastive Search
# =========================================================================


class MultiReferenceContrastiveSearch(ContrastiveSearch):
    """Contrastive search that penalises similarity against multiple references.

    Standard contrastive search only penalises similarity to the *current*
    sequence's own context.  This variant additionally penalises candidates
    that are similar to any of the supplied *reference* sequences, encouraging
    outputs that diverge from all references.

    Parameters
    ----------
    config : ContrastiveConfig or None
        Algorithm configuration.
    reference_sequences : list of list of int
        Pre-existing sequences to repel from.
    aggregation_mode : str
        How to aggregate penalties across references: ``"max"``,
        ``"mean"``, or ``"weighted"``.
    reference_weight : float
        Relative weight of the reference penalty vs. the self-context
        penalty.  ``1.0`` means equal weight.
    """

    def __init__(
        self,
        config: Optional[ContrastiveConfig] = None,
        reference_sequences: Optional[List[List[int]]] = None,
        aggregation_mode: str = "max",
        reference_weight: float = 1.0,
    ) -> None:
        super().__init__(config)
        self._reference_sequences: List[List[int]] = reference_sequences or []
        self._aggregation_mode = aggregation_mode
        self._reference_weight = reference_weight
        self._reference_embeddings: Optional[List[np.ndarray]] = None

    # -- reference embedding cache ------------------------------------------

    def _build_reference_embeddings(self) -> List[np.ndarray]:
        """Build embedding matrices for all reference sequences.

        Returns
        -------
        list of np.ndarray
            Each element has shape ``(ref_len, embedding_dim)``.
        """
        self._ensure_embeddings()
        assert self._embedding_matrix is not None
        ref_embs: List[np.ndarray] = []
        for ref_seq in self._reference_sequences:
            if len(ref_seq) == 0:
                ref_embs.append(
                    np.empty((0, self._cs_config.embedding_dim), dtype=np.float32)
                )
            else:
                ids = np.clip(ref_seq, 0, self._embedding_matrix.shape[0] - 1).astype(int)
                ref_embs.append(self._embedding_matrix[ids])
        return ref_embs

    def _multi_reference_penalty(
        self, candidate_emb: np.ndarray, reference_embs_list: List[np.ndarray]
    ) -> List[float]:
        """Compute degeneration penalty against each reference sequence.

        Parameters
        ----------
        candidate_emb : np.ndarray
            Shape ``(dim,)``.
        reference_embs_list : list of np.ndarray
            Each element has shape ``(ref_len, dim)``.

        Returns
        -------
        list of float
            One penalty value per reference sequence.
        """
        penalties: List[float] = []
        for ref_embs in reference_embs_list:
            if ref_embs.shape[0] == 0:
                penalties.append(0.0)
            else:
                penalties.append(
                    self._scorer._compute_penalty(candidate_emb, ref_embs)
                )
        return penalties

    def _aggregate_penalties(
        self,
        penalties: List[float],
        weights: Optional[List[float]] = None,
    ) -> float:
        """Aggregate penalties from multiple references into a single value.

        Parameters
        ----------
        penalties : list of float
            Per-reference penalties.
        weights : list of float or None
            Optional per-reference weights (used only in ``"weighted"`` mode).

        Returns
        -------
        float
            Aggregated penalty.
        """
        if len(penalties) == 0:
            return 0.0

        arr = np.asarray(penalties, dtype=np.float64)

        if self._aggregation_mode == "max":
            return float(np.max(arr))
        elif self._aggregation_mode == "mean":
            return float(np.mean(arr))
        elif self._aggregation_mode == "weighted":
            if weights is None:
                # Default: linearly decreasing weights (first ref most important)
                n = len(arr)
                w = np.arange(n, 0, -1, dtype=np.float64)
                w /= w.sum()
            else:
                w = np.asarray(weights, dtype=np.float64)
                w_sum = w.sum()
                w = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
            return float(np.dot(arr, w))
        else:
            logger.warning(
                "Unknown aggregation mode %r, falling back to max",
                self._aggregation_mode,
            )
            return float(np.max(arr))

    def _score_with_references(
        self,
        candidate_ids: np.ndarray,
        log_probs: np.ndarray,
        context_embeddings: np.ndarray,
        reference_embs_list: List[np.ndarray],
    ) -> np.ndarray:
        """Score candidates using both self-context and reference penalties.

        Parameters
        ----------
        candidate_ids : np.ndarray
            Shape ``(k,)`` — token indices.
        log_probs : np.ndarray
            Shape ``(k,)`` — log-probabilities.
        context_embeddings : np.ndarray
            Shape ``(context_len, dim)`` — current sequence context.
        reference_embs_list : list of np.ndarray
            Embeddings for each reference sequence.

        Returns
        -------
        np.ndarray
            Shape ``(k,)`` — combined scores.
        """
        assert self._embedding_matrix is not None
        alpha = self._cs_config.alpha
        ids = np.clip(candidate_ids, 0, self._embedding_matrix.shape[0] - 1).astype(int)
        cand_embs = self._embedding_matrix[ids]

        scores = np.empty(len(ids), dtype=np.float64)
        for i in range(len(ids)):
            # Self-context penalty
            if context_embeddings.shape[0] == 0:
                self_penalty = 0.0
            else:
                self_penalty = self._scorer._compute_penalty(
                    cand_embs[i], context_embeddings
                )

            # Reference penalty
            ref_penalties = self._multi_reference_penalty(
                cand_embs[i], reference_embs_list
            )
            ref_penalty = self._aggregate_penalties(ref_penalties)

            # Combined penalty: blend self and reference
            combined_penalty = (
                self_penalty + self._reference_weight * ref_penalty
            ) / (1.0 + self._reference_weight)

            scores[i] = (1.0 - alpha) * log_probs[i] + alpha * (1.0 - combined_penalty)

        return scores

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate sequences with multi-reference contrastive penalties.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        self._ensure_embeddings()
        assert self._embedding_matrix is not None

        # Build reference embeddings once
        ref_embs_list = self._build_reference_embeddings()
        logger.info(
            "MultiReferenceContrastiveSearch: %d reference sequences, "
            "aggregation=%s, weight=%.2f",
            len(ref_embs_list),
            self._aggregation_mode,
            self._reference_weight,
        )

        alpha = self._cs_config.alpha
        k = self._cs_config.k
        temperature = self._cs_config.temperature

        sequences: List[TokenSequence] = []
        for seq_idx in range(self.config.num_sequences):
            temp = temperature
            if seq_idx > 0 and self._rng is not None:
                jitter = self._rng.uniform(-0.05, 0.05)
                temp = max(0.1, temperature + jitter * seq_idx)

            current_ids = list(prompt_ids)
            context_emb_list: List[np.ndarray] = []
            if len(prompt_ids) > 0:
                prompt_embs = self._get_token_embeddings(prompt_ids)
                context_emb_list = [prompt_embs[i] for i in range(prompt_embs.shape[0])]

            generated: List[int] = []

            for step in range(self.config.max_new_tokens):
                logits = logit_source([current_ids])[0].copy()
                if temp != 1.0 and temp > 0:
                    logits = logits / temp

                log_probs = _log_softmax(logits)
                actual_k = min(k, logits.shape[0])
                top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
                top_k_log_probs = log_probs[top_k_indices]

                ctx_embs = (
                    np.stack(context_emb_list, axis=0)
                    if context_emb_list
                    else np.empty((0, self._cs_config.embedding_dim), dtype=np.float32)
                )

                if alpha == 0.0:
                    best_local = int(np.argmax(top_k_log_probs))
                    selected_token = int(top_k_indices[best_local])
                else:
                    scores = self._score_with_references(
                        top_k_indices, top_k_log_probs, ctx_embs, ref_embs_list,
                    )
                    best_local = int(np.argmax(scores))
                    selected_token = int(top_k_indices[best_local])

                current_ids.append(selected_token)
                generated.append(selected_token)
                context_emb_list.append(self._embedding_matrix[selected_token].copy())

                if (
                    self.config.eos_token_id is not None
                    and selected_token == self.config.eos_token_id
                ):
                    break

            sequences.append(generated)

        sequences.sort(key=len, reverse=True)
        return sequences


# =========================================================================
# Learned Degeneration Penalty
# =========================================================================


class LearnedDegenerationPenalty(ContrastiveSearch):
    """Contrastive search with a learned / parameterised penalty function.

    Instead of a fixed cosine-similarity penalty, this variant maintains
    trainable weights for different penalty components:

    * **Recency-weighted similarity** — recent context tokens contribute more.
    * **Frequency-based penalty** — tokens that already appear frequently
      in the context receive a higher penalty.
    * **Position-based decay** — penalty contribution decays with distance
      from the current position.

    Parameters
    ----------
    config : ContrastiveConfig or None
        Algorithm configuration.
    recency_weight : float
        Initial weight for the recency component.
    frequency_weight : float
        Initial weight for the frequency component.
    position_weight : float
        Initial weight for the position decay component.
    decay_factor : float
        Exponential decay factor for recency weighting.
    learning_rate : float
        Step size for ``update_weights``.
    """

    def __init__(
        self,
        config: Optional[ContrastiveConfig] = None,
        recency_weight: float = 0.5,
        frequency_weight: float = 0.3,
        position_weight: float = 0.2,
        decay_factor: float = 0.95,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(config)
        self._weights = np.array(
            [recency_weight, frequency_weight, position_weight], dtype=np.float64,
        )
        self._decay_factor = decay_factor
        self._learning_rate = learning_rate
        # Track token frequencies per generation call
        self._token_frequencies: Dict[int, int] = defaultdict(int)

    # -- penalty components ------------------------------------------------

    def _recency_weight(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """Compute recency-weighted similarity penalty.

        More recent context tokens receive exponentially higher weight.

        Parameters
        ----------
        candidate_emb : np.ndarray — shape ``(dim,)``
        context_embs : np.ndarray  — shape ``(n, dim)``

        Returns
        -------
        float
            Recency-weighted similarity in ``[0, 1]``.
        """
        n = context_embs.shape[0]
        if n == 0:
            return 0.0

        sims = ContrastiveScorer._cosine_similarities(candidate_emb, context_embs)
        weights = compute_recency_weights(n, self._decay_factor)
        return float(np.dot(sims, weights))

    def _frequency_component(
        self, candidate_id: int, context_tokens: List[int]
    ) -> float:
        """Compute frequency-based penalty for a candidate token.

        Parameters
        ----------
        candidate_id : int
            Candidate token id.
        context_tokens : list of int
            All tokens generated so far.

        Returns
        -------
        float
            Frequency penalty in ``[0, 1]``.  Higher when the token has
            appeared more often.
        """
        if len(context_tokens) == 0:
            return 0.0

        count = sum(1 for t in context_tokens if t == candidate_id)
        # Normalise by context length, capped at 1.0
        return min(1.0, count / max(1, len(context_tokens)) * 5.0)

    def _position_decay(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """Compute position-decay similarity penalty.

        Similarity to tokens at earlier positions is decayed more heavily
        than similarity to recent tokens (simple linear decay).

        Parameters
        ----------
        candidate_emb : np.ndarray — shape ``(dim,)``
        context_embs : np.ndarray  — shape ``(n, dim)``

        Returns
        -------
        float
            Position-decayed similarity.
        """
        n = context_embs.shape[0]
        if n == 0:
            return 0.0

        sims = ContrastiveScorer._cosine_similarities(candidate_emb, context_embs)
        # Linear ramp: position 0 gets weight 1/n, position n-1 gets weight n/n
        positions = np.arange(1, n + 1, dtype=np.float64) / n
        weights = positions / positions.sum()
        return float(np.dot(sims, weights))

    def _compute_learned_penalty(
        self,
        candidate_id: int,
        candidate_emb: np.ndarray,
        context_embs: np.ndarray,
        context_tokens: List[int],
    ) -> float:
        """Compute the full learned degeneration penalty.

        Combines three penalty components using the current trainable weights.

        Parameters
        ----------
        candidate_id : int
        candidate_emb : np.ndarray — shape ``(dim,)``
        context_embs : np.ndarray  — shape ``(n, dim)``
        context_tokens : list of int

        Returns
        -------
        float
            Combined penalty value.
        """
        recency = self._recency_weight(candidate_emb, context_embs)
        frequency = self._frequency_component(candidate_id, context_tokens)
        position = self._position_decay(candidate_emb, context_embs)

        components = np.array([recency, frequency, position], dtype=np.float64)
        # Normalise weights to sum to 1
        w = np.abs(self._weights)
        w_sum = w.sum()
        if w_sum > _EPS:
            w = w / w_sum
        else:
            w = np.ones(3, dtype=np.float64) / 3.0

        penalty = float(np.dot(w, components))
        return np.clip(penalty, 0.0, 1.0)

    def update_weights(
        self, gradient: np.ndarray
    ) -> None:
        """Update the trainable penalty weights via gradient step.

        Parameters
        ----------
        gradient : np.ndarray
            Shape ``(3,)`` — gradient for ``[recency, frequency, position]``
            weights.  Positive gradients increase the weight.
        """
        gradient = np.asarray(gradient, dtype=np.float64).ravel()
        if gradient.shape[0] != 3:
            logger.warning(
                "LearnedDegenerationPenalty.update_weights: expected gradient "
                "of shape (3,), got %s — skipping",
                gradient.shape,
            )
            return
        self._weights += self._learning_rate * gradient
        # Keep weights non-negative
        self._weights = np.maximum(self._weights, 0.0)
        logger.debug("Updated penalty weights: %s", self._weights)

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate sequences using the learned degeneration penalty.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        self._ensure_embeddings()
        assert self._embedding_matrix is not None

        alpha = self._cs_config.alpha
        k = self._cs_config.k
        temperature = self._cs_config.temperature

        logger.info(
            "LearnedDegenerationPenalty: weights=[%.3f, %.3f, %.3f], "
            "decay=%.3f",
            *self._weights,
            self._decay_factor,
        )

        sequences: List[TokenSequence] = []
        for seq_idx in range(self.config.num_sequences):
            temp = temperature
            if seq_idx > 0 and self._rng is not None:
                jitter = self._rng.uniform(-0.05, 0.05)
                temp = max(0.1, temperature + jitter * seq_idx)

            current_ids = list(prompt_ids)
            context_emb_list: List[np.ndarray] = []
            context_tokens: List[int] = list(prompt_ids)
            if len(prompt_ids) > 0:
                prompt_embs = self._get_token_embeddings(prompt_ids)
                context_emb_list = [prompt_embs[i] for i in range(prompt_embs.shape[0])]

            generated: List[int] = []

            for step in range(self.config.max_new_tokens):
                logits = logit_source([current_ids])[0].copy()
                if temp != 1.0 and temp > 0:
                    logits = logits / temp

                log_probs = _log_softmax(logits)
                actual_k = min(k, logits.shape[0])
                top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
                top_k_log_probs = log_probs[top_k_indices]

                ctx_embs = (
                    np.stack(context_emb_list, axis=0)
                    if context_emb_list
                    else np.empty((0, self._cs_config.embedding_dim), dtype=np.float32)
                )

                if alpha == 0.0 or ctx_embs.shape[0] == 0:
                    best_local = int(np.argmax(top_k_log_probs))
                    selected_token = int(top_k_indices[best_local])
                else:
                    ids = np.clip(
                        top_k_indices, 0, self._embedding_matrix.shape[0] - 1
                    ).astype(int)
                    cand_embs = self._embedding_matrix[ids]

                    scores = np.empty(len(ids), dtype=np.float64)
                    for i in range(len(ids)):
                        penalty = self._compute_learned_penalty(
                            int(ids[i]), cand_embs[i], ctx_embs, context_tokens,
                        )
                        scores[i] = (
                            (1.0 - alpha) * top_k_log_probs[i]
                            + alpha * (1.0 - penalty)
                        )
                    best_local = int(np.argmax(scores))
                    selected_token = int(top_k_indices[best_local])

                current_ids.append(selected_token)
                generated.append(selected_token)
                context_tokens.append(selected_token)
                context_emb_list.append(self._embedding_matrix[selected_token].copy())

                if (
                    self.config.eos_token_id is not None
                    and selected_token == self.config.eos_token_id
                ):
                    break

            sequences.append(generated)

        sequences.sort(key=len, reverse=True)
        return sequences


# =========================================================================
# Batch Contrastive Search
# =========================================================================


class BatchContrastiveSearch(ContrastiveSearch):
    """Generates multiple sequences simultaneously with inter-sequence repulsion.

    Each sequence is penalised not only for internal repetition (standard
    contrastive penalty) but also for similarity to the *other* sequences
    being generated in the same batch.  This encourages diversity across
    the entire batch of outputs.

    Parameters
    ----------
    config : ContrastiveConfig or None
        Algorithm configuration.
    repulsion_strength : float
        Weight of the inter-sequence repulsion penalty.  ``0.0`` disables
        it; ``1.0`` gives it equal weight to the self-degeneration penalty.
    """

    def __init__(
        self,
        config: Optional[ContrastiveConfig] = None,
        repulsion_strength: float = 0.5,
    ) -> None:
        super().__init__(config)
        self._repulsion_strength = repulsion_strength

    def _inter_sequence_penalty(
        self,
        candidate_emb: np.ndarray,
        other_sequences_embs: List[np.ndarray],
    ) -> float:
        """Compute penalty for similarity to other sequences in the batch.

        Parameters
        ----------
        candidate_emb : np.ndarray
            Shape ``(dim,)`` — embedding of the candidate token.
        other_sequences_embs : list of np.ndarray
            Each element has shape ``(seq_len, dim)`` — context embeddings
            of each *other* sequence in the batch.

        Returns
        -------
        float
            Maximum similarity to any token in any other sequence.
        """
        if not other_sequences_embs:
            return 0.0

        max_sim = 0.0
        for other_embs in other_sequences_embs:
            if other_embs.shape[0] == 0:
                continue
            sims = ContrastiveScorer._cosine_similarities(candidate_emb, other_embs)
            local_max = float(np.max(sims))
            if local_max > max_sim:
                max_sim = local_max
        return max_sim

    def _batch_degeneration_score(
        self,
        candidate_id: int,
        log_prob: float,
        self_context_embs: np.ndarray,
        other_sequences_embs: List[np.ndarray],
    ) -> float:
        """Compute combined contrastive + inter-sequence repulsion score.

        Parameters
        ----------
        candidate_id : int
        log_prob : float
        self_context_embs : np.ndarray
            Shape ``(context_len, dim)``.
        other_sequences_embs : list of np.ndarray

        Returns
        -------
        float
            Combined score.
        """
        assert self._embedding_matrix is not None
        alpha = self._cs_config.alpha
        cid = int(np.clip(candidate_id, 0, self._embedding_matrix.shape[0] - 1))
        candidate_emb = self._embedding_matrix[cid]

        # Self-degeneration penalty
        if self_context_embs.shape[0] == 0:
            self_penalty = 0.0
        else:
            self_penalty = self._scorer._compute_penalty(
                candidate_emb, self_context_embs
            )

        # Inter-sequence penalty
        inter_penalty = self._inter_sequence_penalty(
            candidate_emb, other_sequences_embs
        )

        combined_penalty = (
            self_penalty + self._repulsion_strength * inter_penalty
        ) / (1.0 + self._repulsion_strength)

        return (1.0 - alpha) * log_prob + alpha * (1.0 - combined_penalty)

    def _select_batch_candidates(
        self,
        top_k_indices: np.ndarray,
        top_k_log_probs: np.ndarray,
        self_context_embs: np.ndarray,
        other_sequences_embs: List[np.ndarray],
    ) -> Tuple[int, float]:
        """Select the best candidate for one sequence within the batch.

        Parameters
        ----------
        top_k_indices : np.ndarray — shape ``(k,)``
        top_k_log_probs : np.ndarray — shape ``(k,)``
        self_context_embs : np.ndarray — shape ``(context_len, dim)``
        other_sequences_embs : list of np.ndarray

        Returns
        -------
        (token_id, score) : tuple
        """
        best_score = _NEG_INF
        best_token = int(top_k_indices[0])

        for i in range(len(top_k_indices)):
            score = self._batch_degeneration_score(
                int(top_k_indices[i]),
                float(top_k_log_probs[i]),
                self_context_embs,
                other_sequences_embs,
            )
            if score > best_score:
                best_score = score
                best_token = int(top_k_indices[i])

        return best_token, best_score

    def generate_batch(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate multiple sequences simultaneously with inter-sequence repulsion.

        Unlike the base ``generate`` which produces sequences independently,
        this method generates all sequences in lock-step so that each
        step can account for what the *other* sequences have generated.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        self._ensure_embeddings()
        assert self._embedding_matrix is not None

        k = self._cs_config.k
        temperature = self._cs_config.temperature
        num_seqs = self.config.num_sequences

        logger.info(
            "BatchContrastiveSearch: generating %d sequences with "
            "repulsion_strength=%.3f",
            num_seqs,
            self._repulsion_strength,
        )

        # Per-sequence state
        current_ids_list: List[List[int]] = [list(prompt_ids) for _ in range(num_seqs)]
        context_emb_lists: List[List[np.ndarray]] = []
        generated_lists: List[List[int]] = [[] for _ in range(num_seqs)]
        finished: List[bool] = [False] * num_seqs

        for s in range(num_seqs):
            emb_list: List[np.ndarray] = []
            if len(prompt_ids) > 0:
                prompt_embs = self._get_token_embeddings(prompt_ids)
                emb_list = [prompt_embs[i] for i in range(prompt_embs.shape[0])]
            context_emb_lists.append(emb_list)

        for step in range(self.config.max_new_tokens):
            if all(finished):
                break

            for s in range(num_seqs):
                if finished[s]:
                    continue

                # Per-sequence temperature jitter
                temp = temperature
                if s > 0 and self._rng is not None:
                    jitter = self._rng.uniform(-0.05, 0.05)
                    temp = max(0.1, temperature + jitter * s)

                logits = logit_source([current_ids_list[s]])[0].copy()
                if temp != 1.0 and temp > 0:
                    logits = logits / temp

                log_probs = _log_softmax(logits)
                actual_k = min(k, logits.shape[0])
                top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
                top_k_log_probs = log_probs[top_k_indices]

                self_ctx = (
                    np.stack(context_emb_lists[s], axis=0)
                    if context_emb_lists[s]
                    else np.empty(
                        (0, self._cs_config.embedding_dim), dtype=np.float32
                    )
                )

                # Gather other sequences' embeddings
                other_embs: List[np.ndarray] = []
                for o in range(num_seqs):
                    if o == s:
                        continue
                    if context_emb_lists[o]:
                        other_embs.append(np.stack(context_emb_lists[o], axis=0))

                selected_token, _ = self._select_batch_candidates(
                    top_k_indices, top_k_log_probs, self_ctx, other_embs,
                )

                current_ids_list[s].append(selected_token)
                generated_lists[s].append(selected_token)
                context_emb_lists[s].append(
                    self._embedding_matrix[selected_token].copy()
                )

                if (
                    self.config.eos_token_id is not None
                    and selected_token == self.config.eos_token_id
                ):
                    finished[s] = True

        result = [gen for gen in generated_lists]
        result.sort(key=len, reverse=True)
        return result


# =========================================================================
# Dynamic-Alpha Contrastive Search
# =========================================================================


class DynamicAlphaContrastive(ContrastiveSearch):
    """Contrastive search where alpha changes dynamically during generation.

    Supports three schedule types:

    * ``"linear"`` — alpha increases linearly from ``alpha_start`` to
      ``alpha_end`` over ``max_new_tokens`` steps.
    * ``"cosine"`` — alpha follows a cosine annealing schedule.
    * ``"adaptive"`` — alpha is adjusted based on detected repetition:
      increased when repetition is high, decreased when diversity is
      sufficient.

    Parameters
    ----------
    config : ContrastiveConfig or None
        Algorithm configuration.  The ``alpha`` field is used as the
        base / starting alpha.
    schedule : str
        One of ``"linear"``, ``"cosine"``, ``"adaptive"``.
    alpha_start : float or None
        Starting alpha.  Defaults to ``config.alpha``.
    alpha_end : float or None
        Ending alpha (used by linear and cosine schedules).
    repetition_threshold : float
        Fraction of recent tokens that are repeats before adaptive mode
        increases alpha.
    alpha_step : float
        Amount to increase / decrease alpha per step in adaptive mode.
    window_size : int
        Look-back window for repetition detection.
    """

    def __init__(
        self,
        config: Optional[ContrastiveConfig] = None,
        schedule: str = "linear",
        alpha_start: Optional[float] = None,
        alpha_end: Optional[float] = None,
        repetition_threshold: float = 0.3,
        alpha_step: float = 0.02,
        window_size: int = 10,
    ) -> None:
        super().__init__(config)
        self._schedule = schedule
        self._alpha_start = alpha_start if alpha_start is not None else self._cs_config.alpha
        self._alpha_end = alpha_end if alpha_end is not None else min(1.0, self._alpha_start + 0.3)
        self._repetition_threshold = repetition_threshold
        self._alpha_step = alpha_step
        self._window_size = window_size

    # -- schedule functions ------------------------------------------------

    def _linear_schedule(self, step: int, max_steps: int) -> float:
        """Compute alpha using a linear schedule.

        Parameters
        ----------
        step : int
            Current generation step (0-indexed).
        max_steps : int
            Total number of generation steps.

        Returns
        -------
        float
            Alpha value for this step.
        """
        if max_steps <= 1:
            return self._alpha_start
        t = min(step / (max_steps - 1), 1.0)
        return self._alpha_start + t * (self._alpha_end - self._alpha_start)

    def _cosine_schedule(self, step: int, max_steps: int) -> float:
        """Compute alpha using a cosine annealing schedule.

        Parameters
        ----------
        step : int
        max_steps : int

        Returns
        -------
        float
            Alpha value for this step.
        """
        if max_steps <= 1:
            return self._alpha_start
        t = min(step / (max_steps - 1), 1.0)
        cosine_value = 0.5 * (1.0 - math.cos(math.pi * t))
        return self._alpha_start + cosine_value * (self._alpha_end - self._alpha_start)

    def _adaptive_schedule(
        self, current_alpha: float, generated_tokens: List[int]
    ) -> float:
        """Adjust alpha based on detected repetition.

        Parameters
        ----------
        current_alpha : float
            Alpha value at the previous step.
        generated_tokens : list of int
            Tokens generated so far.

        Returns
        -------
        float
            Updated alpha value.
        """
        if len(generated_tokens) < 2:
            return current_alpha

        repetition = self._detect_repetition(generated_tokens)
        if repetition > self._repetition_threshold:
            # Too much repetition — increase penalty
            new_alpha = min(1.0, current_alpha + self._alpha_step)
            logger.debug(
                "Adaptive alpha: repetition=%.3f > threshold=%.3f, "
                "alpha %.3f -> %.3f",
                repetition,
                self._repetition_threshold,
                current_alpha,
                new_alpha,
            )
            return new_alpha
        elif repetition < self._repetition_threshold * 0.5:
            # Diversity is sufficient — decrease penalty to allow coherence
            new_alpha = max(0.0, current_alpha - self._alpha_step * 0.5)
            return new_alpha
        else:
            return current_alpha

    def _compute_current_alpha(
        self,
        step: int,
        max_steps: int,
        current_alpha: float,
        generated_tokens: List[int],
    ) -> float:
        """Dispatch to the configured schedule.

        Parameters
        ----------
        step : int
        max_steps : int
        current_alpha : float
            Current alpha (only used in adaptive mode).
        generated_tokens : list of int

        Returns
        -------
        float
            Alpha for this step.
        """
        if self._schedule == "linear":
            return self._linear_schedule(step, max_steps)
        elif self._schedule == "cosine":
            return self._cosine_schedule(step, max_steps)
        elif self._schedule == "adaptive":
            return self._adaptive_schedule(current_alpha, generated_tokens)
        else:
            logger.warning(
                "Unknown schedule %r, falling back to linear", self._schedule
            )
            return self._linear_schedule(step, max_steps)

    def _detect_repetition(self, tokens: List[int]) -> float:
        """Compute the repetition rate in a recent window of tokens.

        Parameters
        ----------
        tokens : list of int
            Generated tokens.

        Returns
        -------
        float
            Fraction of tokens in the window that are duplicates of an
            earlier token in the same window.
        """
        window = tokens[-self._window_size:] if len(tokens) > self._window_size else tokens
        if len(window) <= 1:
            return 0.0
        seen: set = set()
        repeated = 0
        for tok in window:
            if tok in seen:
                repeated += 1
            seen.add(tok)
        return repeated / len(window)

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate sequences with dynamically scheduled alpha.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        self._ensure_embeddings()
        assert self._embedding_matrix is not None

        k = self._cs_config.k
        temperature = self._cs_config.temperature
        max_steps = self.config.max_new_tokens

        logger.info(
            "DynamicAlphaContrastive: schedule=%s, alpha_start=%.3f, "
            "alpha_end=%.3f",
            self._schedule,
            self._alpha_start,
            self._alpha_end,
        )

        sequences: List[TokenSequence] = []
        for seq_idx in range(self.config.num_sequences):
            temp = temperature
            if seq_idx > 0 and self._rng is not None:
                jitter = self._rng.uniform(-0.05, 0.05)
                temp = max(0.1, temperature + jitter * seq_idx)

            current_ids = list(prompt_ids)
            context_emb_list: List[np.ndarray] = []
            if len(prompt_ids) > 0:
                prompt_embs = self._get_token_embeddings(prompt_ids)
                context_emb_list = [prompt_embs[i] for i in range(prompt_embs.shape[0])]

            generated: List[int] = []
            current_alpha = self._alpha_start

            for step in range(max_steps):
                current_alpha = self._compute_current_alpha(
                    step, max_steps, current_alpha, generated,
                )

                logits = logit_source([current_ids])[0].copy()
                if temp != 1.0 and temp > 0:
                    logits = logits / temp

                log_probs = _log_softmax(logits)
                actual_k = min(k, logits.shape[0])
                top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
                top_k_log_probs = log_probs[top_k_indices]

                ctx_embs = (
                    np.stack(context_emb_list, axis=0)
                    if context_emb_list
                    else np.empty((0, self._cs_config.embedding_dim), dtype=np.float32)
                )

                if current_alpha == 0.0 or ctx_embs.shape[0] == 0:
                    best_local = int(np.argmax(top_k_log_probs))
                    selected_token = int(top_k_indices[best_local])
                else:
                    scores = self._scorer.batch_score_candidates(
                        candidate_ids=top_k_indices,
                        context_embeddings=ctx_embs,
                        embedding_matrix=self._embedding_matrix,
                        alpha=current_alpha,
                        log_probs=top_k_log_probs,
                    )
                    best_local = int(np.argmax(scores))
                    selected_token = int(top_k_indices[best_local])

                current_ids.append(selected_token)
                generated.append(selected_token)
                context_emb_list.append(self._embedding_matrix[selected_token].copy())

                if (
                    self.config.eos_token_id is not None
                    and selected_token == self.config.eos_token_id
                ):
                    break

            sequences.append(generated)

        sequences.sort(key=len, reverse=True)
        return sequences


# =========================================================================
# Embedding-Space Contrastive Search
# =========================================================================


class EmbeddingSpaceContrastive(ContrastiveSearch):
    """Contrastive decoding directly in embedding space.

    Instead of evaluating candidates purely by token-level similarity,
    this variant projects candidate tokens into a (optionally reduced)
    embedding space and selects tokens that maximise distance from all
    previous token embeddings while maintaining probability.

    Supports PCA-style or random-projection dimensionality reduction.

    Parameters
    ----------
    config : ContrastiveConfig or None
        Algorithm configuration.
    projection_dim : int or None
        Target dimensionality for projection.  If *None*, no projection
        is applied.
    projection_method : str
        ``"pca"`` or ``"random"``.
    distance_metric : str
        Distance metric in embedding space: ``"cosine"`` or ``"euclidean"``.
    """

    def __init__(
        self,
        config: Optional[ContrastiveConfig] = None,
        projection_dim: Optional[int] = None,
        projection_method: str = "random",
        distance_metric: str = "cosine",
    ) -> None:
        super().__init__(config)
        self._projection_dim = projection_dim
        self._projection_method = projection_method
        self._distance_metric = distance_metric
        self._projection_matrix: Optional[np.ndarray] = None

    def _dimensionality_reduction(
        self, embeddings: np.ndarray, target_dim: int
    ) -> np.ndarray:
        """Reduce dimensionality of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, original_dim)``.
        target_dim : int
            Target dimensionality.

        Returns
        -------
        np.ndarray
            Shape ``(n, target_dim)``.
        """
        original_dim = embeddings.shape[1]
        if target_dim >= original_dim:
            return embeddings

        if self._projection_method == "pca":
            # Simple PCA via SVD on zero-centred data
            mean = embeddings.mean(axis=0)
            centred = embeddings - mean
            # Use min(n, d) components via economy SVD
            n_components = min(centred.shape[0], centred.shape[1], target_dim)
            if n_components == 0:
                return embeddings[:, :target_dim]
            try:
                _, _, Vt = np.linalg.svd(centred, full_matrices=False)
                projection = Vt[:target_dim]  # (target_dim, original_dim)
                return centred @ projection.T
            except np.linalg.LinAlgError:
                logger.warning(
                    "SVD failed for PCA; falling back to random projection"
                )
                return self._random_projection(embeddings, target_dim)
        elif self._projection_method == "random":
            return self._random_projection(embeddings, target_dim)
        else:
            logger.warning(
                "Unknown projection method %r, using random",
                self._projection_method,
            )
            return self._random_projection(embeddings, target_dim)

    def _random_projection(
        self, embeddings: np.ndarray, target_dim: int
    ) -> np.ndarray:
        """Apply random projection for dimensionality reduction.

        Parameters
        ----------
        embeddings : np.ndarray — shape ``(n, original_dim)``
        target_dim : int

        Returns
        -------
        np.ndarray — shape ``(n, target_dim)``
        """
        original_dim = embeddings.shape[1]
        if self._projection_matrix is None or self._projection_matrix.shape != (
            original_dim,
            target_dim,
        ):
            rng = np.random.default_rng(seed=123)
            self._projection_matrix = rng.standard_normal(
                (original_dim, target_dim)
            ).astype(np.float32)
            # Normalise columns
            col_norms = np.linalg.norm(self._projection_matrix, axis=0, keepdims=True)
            self._projection_matrix /= np.maximum(col_norms, _EPS)

        return embeddings @ self._projection_matrix

    def _project_to_embedding_space(
        self, token_ids: np.ndarray
    ) -> np.ndarray:
        """Project token embeddings, optionally reducing dimensionality.

        Parameters
        ----------
        token_ids : np.ndarray
            Shape ``(k,)`` — token indices.

        Returns
        -------
        np.ndarray
            Shape ``(k, dim)`` where ``dim`` is ``projection_dim`` if set,
            else ``embedding_dim``.
        """
        assert self._embedding_matrix is not None
        ids = np.clip(token_ids, 0, self._embedding_matrix.shape[0] - 1).astype(int)
        embs = self._embedding_matrix[ids]

        if self._projection_dim is not None:
            embs = self._dimensionality_reduction(embs, self._projection_dim)

        return embs

    def _embedding_distance(
        self, candidate_emb: np.ndarray, context_embs: np.ndarray
    ) -> float:
        """Compute minimum distance from candidate to all context embeddings.

        Parameters
        ----------
        candidate_emb : np.ndarray — shape ``(dim,)``
        context_embs : np.ndarray  — shape ``(n, dim)``

        Returns
        -------
        float
            Minimum distance (higher = more novel).  For cosine distance
            this is ``1 - max_similarity``; for euclidean it is the
            minimum L2 distance.
        """
        if context_embs.shape[0] == 0:
            return 1.0

        if self._distance_metric == "cosine":
            sims = ContrastiveScorer._cosine_similarities(candidate_emb, context_embs)
            return float(1.0 - np.max(sims))
        elif self._distance_metric == "euclidean":
            dists = np.linalg.norm(context_embs - candidate_emb[np.newaxis, :], axis=1)
            return float(np.min(dists))
        else:
            sims = ContrastiveScorer._cosine_similarities(candidate_emb, context_embs)
            return float(1.0 - np.max(sims))

    def _select_in_embedding_space(
        self,
        top_k_indices: np.ndarray,
        top_k_log_probs: np.ndarray,
        context_embs_projected: np.ndarray,
        alpha: float,
    ) -> Tuple[int, float]:
        """Select best candidate using embedding-space distance.

        Parameters
        ----------
        top_k_indices : np.ndarray — shape ``(k,)``
        top_k_log_probs : np.ndarray — shape ``(k,)``
        context_embs_projected : np.ndarray — shape ``(context_len, proj_dim)``
        alpha : float

        Returns
        -------
        (token_id, score) : tuple
        """
        cand_embs = self._project_to_embedding_space(top_k_indices)

        best_score = _NEG_INF
        best_token = int(top_k_indices[0])

        for i in range(len(top_k_indices)):
            distance = self._embedding_distance(cand_embs[i], context_embs_projected)
            # Distance is already a "diversity" measure — higher = better
            score = (1.0 - alpha) * top_k_log_probs[i] + alpha * distance
            if score > best_score:
                best_score = score
                best_token = int(top_k_indices[i])

        return best_token, best_score

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate sequences using embedding-space contrastive selection.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        self._ensure_embeddings()
        assert self._embedding_matrix is not None

        alpha = self._cs_config.alpha
        k = self._cs_config.k
        temperature = self._cs_config.temperature

        logger.info(
            "EmbeddingSpaceContrastive: projection_dim=%s, method=%s, "
            "distance=%s",
            self._projection_dim,
            self._projection_method,
            self._distance_metric,
        )

        sequences: List[TokenSequence] = []
        for seq_idx in range(self.config.num_sequences):
            temp = temperature
            if seq_idx > 0 and self._rng is not None:
                jitter = self._rng.uniform(-0.05, 0.05)
                temp = max(0.1, temperature + jitter * seq_idx)

            current_ids = list(prompt_ids)
            # Keep projected context embeddings
            context_projected_list: List[np.ndarray] = []
            if len(prompt_ids) > 0:
                prompt_projected = self._project_to_embedding_space(
                    np.array(prompt_ids)
                )
                context_projected_list = [
                    prompt_projected[i] for i in range(prompt_projected.shape[0])
                ]

            generated: List[int] = []

            for step in range(self.config.max_new_tokens):
                logits = logit_source([current_ids])[0].copy()
                if temp != 1.0 and temp > 0:
                    logits = logits / temp

                log_probs = _log_softmax(logits)
                actual_k = min(k, logits.shape[0])
                top_k_indices = np.argpartition(log_probs, -actual_k)[-actual_k:]
                top_k_log_probs = log_probs[top_k_indices]

                ctx_proj = (
                    np.stack(context_projected_list, axis=0)
                    if context_projected_list
                    else np.empty(
                        (0, self._projection_dim or self._cs_config.embedding_dim),
                        dtype=np.float32,
                    )
                )

                if alpha == 0.0 or ctx_proj.shape[0] == 0:
                    best_local = int(np.argmax(top_k_log_probs))
                    selected_token = int(top_k_indices[best_local])
                else:
                    selected_token, _ = self._select_in_embedding_space(
                        top_k_indices, top_k_log_probs, ctx_proj, alpha,
                    )

                current_ids.append(selected_token)
                generated.append(selected_token)

                # Project the selected token and cache
                sel_proj = self._project_to_embedding_space(
                    np.array([selected_token])
                )
                context_projected_list.append(sel_proj[0])

                if (
                    self.config.eos_token_id is not None
                    and selected_token == self.config.eos_token_id
                ):
                    break

            sequences.append(generated)

        sequences.sort(key=len, reverse=True)
        return sequences


# =========================================================================
# Additional helper functions
# =========================================================================


def multi_reference_degeneration_penalty(
    candidate_emb: np.ndarray,
    reference_embs_list: List[np.ndarray],
    mode: str = "max",
) -> float:
    """Compute degeneration penalty against multiple reference sequences.

    Parameters
    ----------
    candidate_emb : np.ndarray
        Shape ``(dim,)`` — candidate token embedding.
    reference_embs_list : list of np.ndarray
        Each element has shape ``(ref_len, dim)`` — embeddings of a
        reference sequence.
    mode : str
        Aggregation across references: ``"max"``, ``"mean"``, or
        ``"weighted"``.

    Returns
    -------
    float
        Aggregated penalty value.
    """
    if not reference_embs_list:
        return 0.0

    per_ref_penalties: List[float] = []
    for ref_embs in reference_embs_list:
        if ref_embs.shape[0] == 0:
            per_ref_penalties.append(0.0)
            continue
        sims = ContrastiveScorer._cosine_similarities(candidate_emb, ref_embs)
        per_ref_penalties.append(float(np.max(sims)))

    arr = np.asarray(per_ref_penalties, dtype=np.float64)
    if mode == "max":
        return float(np.max(arr))
    elif mode == "mean":
        return float(np.mean(arr))
    elif mode == "weighted":
        n = len(arr)
        w = np.arange(n, 0, -1, dtype=np.float64)
        w /= w.sum()
        return float(np.dot(arr, w))
    else:
        return float(np.max(arr))


def dynamic_alpha_schedule(
    step: int,
    max_steps: int,
    schedule_type: str = "linear",
    **kwargs: Any,
) -> float:
    """Compute alpha value according to a schedule.

    Stand-alone function usable outside ``DynamicAlphaContrastive``.

    Parameters
    ----------
    step : int
        Current step (0-indexed).
    max_steps : int
        Total steps.
    schedule_type : str
        ``"linear"`` or ``"cosine"``.
    **kwargs
        ``alpha_start`` (default ``0.3``), ``alpha_end`` (default ``0.9``).

    Returns
    -------
    float
        Alpha for this step.
    """
    alpha_start = float(kwargs.get("alpha_start", 0.3))
    alpha_end = float(kwargs.get("alpha_end", 0.9))

    if max_steps <= 1:
        return alpha_start

    t = min(step / (max_steps - 1), 1.0)

    if schedule_type == "cosine":
        cosine_value = 0.5 * (1.0 - math.cos(math.pi * t))
        return alpha_start + cosine_value * (alpha_end - alpha_start)
    else:
        # Linear (default)
        return alpha_start + t * (alpha_end - alpha_start)


def inter_sequence_similarity(
    sequences: List[List[int]],
    embedding_matrix: np.ndarray,
) -> np.ndarray:
    """Compute pairwise similarity between sequences.

    Each sequence is represented by the mean of its token embeddings.
    Returns a symmetric similarity matrix.

    Parameters
    ----------
    sequences : list of list of int
        Token id sequences.
    embedding_matrix : np.ndarray
        Shape ``(vocab_size, dim)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_sequences, n_sequences)`` — pairwise cosine similarities.
    """
    n = len(sequences)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    dim = embedding_matrix.shape[1]
    mean_embs = np.zeros((n, dim), dtype=np.float64)

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        ids = np.clip(seq, 0, embedding_matrix.shape[0] - 1).astype(int)
        embs = embedding_matrix[ids]
        mean_embs[i] = embs.mean(axis=0)

    # Compute pairwise cosine similarity
    norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    normed = mean_embs / norms
    sim_matrix = normed @ normed.T
    return sim_matrix


def embedding_space_diversity(
    sequences: List[List[int]],
    embedding_matrix: np.ndarray,
    projection_dim: Optional[int] = None,
) -> float:
    """Measure diversity of sequences in embedding space.

    Computes the mean pairwise distance between sequence representations
    (mean embedding of each sequence).  Higher values indicate more
    diverse outputs.

    Parameters
    ----------
    sequences : list of list of int
    embedding_matrix : np.ndarray
        Shape ``(vocab_size, dim)``.
    projection_dim : int or None
        If set, project embeddings to this dimensionality before computing
        distances.

    Returns
    -------
    float
        Mean pairwise cosine distance.  In ``[0, 2]``.
    """
    n = len(sequences)
    if n < 2:
        return 0.0

    dim = embedding_matrix.shape[1]
    mean_embs = np.zeros((n, dim), dtype=np.float64)

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        ids = np.clip(seq, 0, embedding_matrix.shape[0] - 1).astype(int)
        embs = embedding_matrix[ids]
        mean_embs[i] = embs.mean(axis=0)

    # Optional dimensionality reduction via random projection
    if projection_dim is not None and projection_dim < dim:
        rng = np.random.default_rng(seed=456)
        proj = rng.standard_normal((dim, projection_dim)).astype(np.float64)
        col_norms = np.linalg.norm(proj, axis=0, keepdims=True)
        proj /= np.maximum(col_norms, _EPS)
        mean_embs = mean_embs @ proj

    # Pairwise cosine distances
    norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    normed = mean_embs / norms
    sim_matrix = normed @ normed.T

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_distances = 1.0 - sim_matrix[triu_indices]
    return float(np.mean(pairwise_distances))


def compute_recency_weights(
    context_length: int,
    decay_factor: float = 0.95,
) -> np.ndarray:
    """Compute exponentially decaying recency weights.

    More recent positions receive higher weight.

    Parameters
    ----------
    context_length : int
        Number of context tokens.
    decay_factor : float
        Exponential decay factor.  ``1.0`` means uniform weights;
        values closer to ``0`` decay faster.

    Returns
    -------
    np.ndarray
        Shape ``(context_length,)`` — normalised weights that sum to 1.
    """
    if context_length <= 0:
        return np.empty(0, dtype=np.float64)

    # Position 0 is oldest, position n-1 is most recent
    positions = np.arange(context_length, dtype=np.float64)
    weights = decay_factor ** (context_length - 1 - positions)
    w_sum = weights.sum()
    if w_sum > _EPS:
        weights /= w_sum
    else:
        weights = np.ones(context_length, dtype=np.float64) / context_length
    return weights
