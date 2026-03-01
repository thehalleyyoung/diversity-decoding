"""
Determinantal Point Process (DPP) Reranking for the Diversity Decoding Arena.

DPP reranking is a two-phase diversity-aware decoding strategy:

1. **Candidate generation** — produce a large pool of candidate sequences
   (e.g., via temperature sampling from a language model).
2. **DPP selection** — construct an L-ensemble kernel matrix that balances
   sequence quality and pairwise diversity, then select a diverse subset of
   size *k* via DPP sampling (greedy MAP, exact k-DPP, or MCMC).

The L-ensemble kernel factorises as  L = diag(q) S diag(q),  where *q* is a
per-item quality vector and *S* is a similarity matrix (RBF, cosine, polynomial,
or n-gram string kernel).  This naturally trades off quality and diversity:
items that are both high-quality *and* dissimilar from already-selected items
receive higher marginal gain.

References
----------
* Kulesza & Taskar, "Determinantal Point Processes for Machine Learning",
  Foundations and Trends in Machine Learning, 2012.
* Chen et al., "Fast Greedy MAP Inference for Determinantal Point Process to
  Improve Recommendation Diversity", NeurIPS 2018.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg as sp_linalg

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

_EPS = 1e-10
_LOG_EPS = -23.0  # ≈ log(1e-10)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    c = logits.max(axis=-1, keepdims=True)
    shifted = logits - c
    lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - lse


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Element-wise log, clamping inputs to avoid -inf."""
    return np.log(np.clip(x, _EPS, None))


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalise each row; zero rows are left unchanged."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


# =========================================================================
# DPPConfig
# =========================================================================


@dataclass
class DPPConfig(DecodingConfig):
    """Configuration for DPP reranking decoding.

    Parameters
    ----------
    candidate_pool_size : int
        Number of candidate sequences to generate in phase 1.
    select_k : int
        Number of sequences to select from the candidate pool via DPP.
    kernel_type : str
        Kernel used for the similarity matrix.  One of ``"rbf"``,
        ``"cosine"``, ``"polynomial"``, ``"string"``.
    kernel_bandwidth : float
        Bandwidth (σ) for the RBF kernel.
    quality_weight : float
        Exponent applied to quality scores before constructing *L*.
        Higher values emphasise quality over diversity.
    embedding_model : str
        Name of the sentence-embedding model used to embed candidate
        sequences.  If unavailable, falls back to simple n-gram features.
    sampling_method : str
        DPP sampling algorithm: ``"greedy"`` (MAP), ``"exact"`` (k-DPP),
        or ``"mcmc"`` (Metropolis–Hastings).
    candidate_generation : str
        Strategy for generating the initial candidate pool.  Currently
        ``"temperature"`` sampling is supported.
    candidate_temperature : float
        Temperature used during candidate generation (phase 1).
    use_quality_model : bool
        Whether to incorporate quality scores in the L-ensemble kernel.
    max_mcmc_iterations : int
        Maximum iterations for MCMC-based sampling.
    polynomial_degree : int
        Degree for the polynomial kernel.
    polynomial_c : float
        Constant term for the polynomial kernel.
    string_kernel_n : int
        N-gram size for the string kernel.
    regularization_eps : float
        Ridge regularisation added to the kernel for numerical stability.
    """

    algorithm_name: str = "DPPReranking"
    candidate_pool_size: int = 100
    select_k: int = 20
    kernel_type: str = "rbf"
    kernel_bandwidth: float = 1.0
    quality_weight: float = 0.5
    embedding_model: str = "all-MiniLM-L6-v2"
    sampling_method: str = "greedy"
    candidate_generation: str = "temperature"
    candidate_temperature: float = 1.0
    use_quality_model: bool = True
    max_mcmc_iterations: int = 1000
    polynomial_degree: int = 3
    polynomial_c: float = 1.0
    string_kernel_n: int = 3
    regularization_eps: float = 1e-6

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate DPP-specific fields on top of the base checks."""
        errors = super().validate()

        if self.candidate_pool_size < 1:
            errors.append("candidate_pool_size must be >= 1")
        if self.select_k < 1:
            errors.append("select_k must be >= 1")
        if self.select_k > self.candidate_pool_size:
            errors.append(
                f"select_k ({self.select_k}) must be <= "
                f"candidate_pool_size ({self.candidate_pool_size})"
            )
        if self.kernel_type not in ("rbf", "cosine", "polynomial", "string"):
            errors.append(
                f"kernel_type must be one of rbf/cosine/polynomial/string, "
                f"got '{self.kernel_type}'"
            )
        if self.kernel_bandwidth <= 0:
            errors.append("kernel_bandwidth must be > 0")
        if self.quality_weight < 0:
            errors.append("quality_weight must be >= 0")
        if self.sampling_method not in ("greedy", "exact", "mcmc"):
            errors.append(
                f"sampling_method must be one of greedy/exact/mcmc, "
                f"got '{self.sampling_method}'"
            )
        if self.candidate_generation not in ("temperature",):
            errors.append(
                f"candidate_generation must be 'temperature', "
                f"got '{self.candidate_generation}'"
            )
        if self.candidate_temperature <= 0:
            errors.append("candidate_temperature must be > 0")
        if self.max_mcmc_iterations < 1:
            errors.append("max_mcmc_iterations must be >= 1")
        if self.polynomial_degree < 1:
            errors.append("polynomial_degree must be >= 1")
        if self.regularization_eps < 0:
            errors.append("regularization_eps must be >= 0")

        return errors


# =========================================================================
# DPPKernel — kernel construction utilities
# =========================================================================


class DPPKernel:
    """Build and manipulate DPP kernel matrices.

    The *L*-ensemble parameterisation is used throughout:

        P(S) ∝ det(L_S)

    where *L* is a positive semi-definite matrix of size ``n × n`` and
    ``L_S`` is the principal sub-matrix indexed by subset *S*.

    Parameters
    ----------
    kernel_type : str
        ``"rbf"``, ``"cosine"``, ``"polynomial"``, or ``"string"``.
    bandwidth : float
        Bandwidth for the RBF kernel.
    polynomial_degree : int
        Degree for the polynomial kernel.
    polynomial_c : float
        Constant for the polynomial kernel.
    string_kernel_n : int
        N-gram size for the string kernel.
    regularization_eps : float
        Small positive constant added to the diagonal for numerical stability.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        bandwidth: float = 1.0,
        polynomial_degree: int = 3,
        polynomial_c: float = 1.0,
        string_kernel_n: int = 3,
        regularization_eps: float = 1e-6,
    ) -> None:
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.polynomial_degree = polynomial_degree
        self.polynomial_c = polynomial_c
        self.string_kernel_n = string_kernel_n
        self.regularization_eps = regularization_eps

    # -- L-ensemble construction -------------------------------------------

    def build_L_ensemble(
        self,
        embeddings: np.ndarray,
        quality_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Construct the L-ensemble kernel.

        If *quality_scores* is provided the kernel is

            L = diag(q) S diag(q)

        where ``q = quality_scores`` and ``S`` is the similarity matrix
        determined by ``self.kernel_type``.  Otherwise ``L = S``.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, d)`` matrix of item embeddings.
        quality_scores : np.ndarray or None
            Shape ``(n,)`` non-negative quality scores.

        Returns
        -------
        np.ndarray
            Shape ``(n, n)`` positive semi-definite L matrix.
        """
        S = self._similarity_matrix(embeddings)
        S = self.regularize(S, self.regularization_eps)

        if quality_scores is not None:
            q = np.asarray(quality_scores, dtype=np.float64)
            q = np.clip(q, _EPS, None)
            L = np.outer(q, q) * S
        else:
            L = S.copy()

        # Ensure exact symmetry (floating-point rounding can break it).
        L = 0.5 * (L + L.T)
        return L

    def build_marginal_kernel(self, L: np.ndarray) -> np.ndarray:
        r"""Compute the marginal kernel  K = L (L + I)^{-1}.

        The diagonal ``K_{ii}`` gives the marginal inclusion probability
        of item *i* under the DPP defined by *L*.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.

        Returns
        -------
        np.ndarray
            ``(n, n)`` marginal kernel *K*.
        """
        n = L.shape[0]
        I = np.eye(n, dtype=L.dtype)
        try:
            K = L @ np.linalg.inv(L + I)
        except LinAlgError:
            logger.warning(
                "Direct inversion failed for marginal kernel; using pseudo-inverse."
            )
            K = L @ np.linalg.pinv(L + I)
        K = 0.5 * (K + K.T)
        return K

    # -- individual kernel functions ---------------------------------------

    @staticmethod
    def rbf_kernel(X: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
        """Radial basis function (Gaussian) kernel.

        .. math::
            S_{ij} = \\exp\\bigl(-\\|x_i - x_j\\|^2 / (2\\sigma^2)\\bigr)

        Parameters
        ----------
        X : np.ndarray
            ``(n, d)`` data matrix.
        bandwidth : float
            Bandwidth σ.

        Returns
        -------
        np.ndarray
            ``(n, n)`` kernel matrix.
        """
        sq_dists = np.sum(X ** 2, axis=1, keepdims=True) \
                   - 2.0 * X @ X.T \
                   + np.sum(X ** 2, axis=1, keepdims=False)
        sq_dists = np.clip(sq_dists, 0.0, None)
        return np.exp(-sq_dists / (2.0 * bandwidth ** 2))

    @staticmethod
    def cosine_kernel(X: np.ndarray) -> np.ndarray:
        """Cosine similarity kernel.

        Parameters
        ----------
        X : np.ndarray
            ``(n, d)`` data matrix.

        Returns
        -------
        np.ndarray
            ``(n, n)`` kernel matrix with values in ``[-1, 1]``.
        """
        X_norm = _normalize_rows(X)
        S = X_norm @ X_norm.T
        # Clamp to valid range to avoid numeric drift.
        return np.clip(S, -1.0, 1.0)

    @staticmethod
    def polynomial_kernel(
        X: np.ndarray,
        degree: int = 3,
        c: float = 1.0,
    ) -> np.ndarray:
        """Polynomial kernel  (X X^T + c)^d.

        Parameters
        ----------
        X : np.ndarray
            ``(n, d)`` data matrix.
        degree : int
            Polynomial degree.
        c : float
            Constant term.

        Returns
        -------
        np.ndarray
            ``(n, n)`` kernel matrix.
        """
        return (X @ X.T + c) ** degree

    @staticmethod
    def string_kernel(sequences: List[List[int]], n: int = 3) -> np.ndarray:
        """N-gram based string kernel.

        Each sequence is represented as a bag of n-grams; the kernel is the
        inner product of (normalised) n-gram count vectors.

        Parameters
        ----------
        sequences : list of list of int
            Token-id sequences.
        n : int
            N-gram size.

        Returns
        -------
        np.ndarray
            ``(m, m)`` kernel matrix.
        """
        m = len(sequences)

        # Build n-gram count vectors.
        ngram_counts: List[Counter] = []
        vocab: Dict[Tuple[int, ...], int] = {}
        for seq in sequences:
            counter: Counter = Counter()
            for i in range(len(seq) - n + 1):
                ng = tuple(seq[i: i + n])
                if ng not in vocab:
                    vocab[ng] = len(vocab)
                counter[ng] += 1
            ngram_counts.append(counter)

        # Assemble dense matrix.
        dim = len(vocab)
        if dim == 0:
            return np.ones((m, m), dtype=np.float64)

        X = np.zeros((m, dim), dtype=np.float64)
        for i, counter in enumerate(ngram_counts):
            for ng, cnt in counter.items():
                X[i, vocab[ng]] = cnt

        # Normalise each row.
        X_norm = _normalize_rows(X)
        S = X_norm @ X_norm.T
        return np.clip(S, -1.0, 1.0)

    # -- spectral helpers --------------------------------------------------

    @staticmethod
    def eigendecompose(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eigen-decomposition of a symmetric PSD matrix.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` symmetric matrix.

        Returns
        -------
        eigenvalues : np.ndarray
            ``(n,)`` eigenvalues in ascending order.
        eigenvectors : np.ndarray
            ``(n, n)`` matrix whose columns are eigenvectors.
        """
        L_sym = 0.5 * (L + L.T)
        eigenvalues, eigenvectors = sp_linalg.eigh(L_sym)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        return eigenvalues, eigenvectors

    @staticmethod
    def log_determinant(L: np.ndarray) -> float:
        """Log-determinant via Cholesky (with fallback to eigenvalues).

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` symmetric PSD matrix.

        Returns
        -------
        float
            ``log det(L)``.  Returns ``-inf`` for singular matrices.
        """
        try:
            chol = sp_linalg.cholesky(L, lower=True)
            return 2.0 * float(np.sum(np.log(np.diag(chol))))
        except LinAlgError:
            eigenvalues = sp_linalg.eigvalsh(L)
            pos = eigenvalues[eigenvalues > _EPS]
            if len(pos) == 0:
                return float("-inf")
            return float(np.sum(np.log(pos)))

    @staticmethod
    def condition_number(L: np.ndarray) -> float:
        """Condition number of the kernel matrix.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` symmetric PSD matrix.

        Returns
        -------
        float
            ``λ_max / λ_min`` among positive eigenvalues.
        """
        eigenvalues = sp_linalg.eigvalsh(L)
        pos = eigenvalues[eigenvalues > _EPS]
        if len(pos) == 0:
            return float("inf")
        return float(pos[-1] / pos[0])

    @staticmethod
    def regularize(L: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Add ridge regularisation  L ← L + εI.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` matrix.
        epsilon : float
            Regularisation constant.

        Returns
        -------
        np.ndarray
            Regularised matrix (copy).
        """
        n = L.shape[0]
        return L + epsilon * np.eye(n, dtype=L.dtype)

    # -- internal ----------------------------------------------------------

    def _similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Dispatch to the configured kernel function."""
        if self.kernel_type == "rbf":
            return self.rbf_kernel(embeddings, self.bandwidth)
        elif self.kernel_type == "cosine":
            return self.cosine_kernel(embeddings)
        elif self.kernel_type == "polynomial":
            return self.polynomial_kernel(
                embeddings, self.polynomial_degree, self.polynomial_c
            )
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")


# =========================================================================
# DPPSampler — subset selection algorithms
# =========================================================================


class DPPSampler:
    """Sampling / MAP inference algorithms for k-DPP selection.

    All methods take the L-ensemble kernel and return a list of selected
    indices.

    Parameters
    ----------
    rng : np.random.Generator or None
        Random number generator for stochastic methods.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    # -- greedy MAP --------------------------------------------------------

    def greedy_map(self, L: np.ndarray, k: int) -> List[int]:
        """Greedy MAP inference for k-DPP.

        Iteratively selects the item with the largest marginal gain in
        ``log det(L_S)``, using rank-one Cholesky updates for efficiency.

        Complexity: O(n k^2).

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.
        k : int
            Number of items to select.

        Returns
        -------
        list of int
            Selected indices.
        """
        n = L.shape[0]
        k = min(k, n)

        selected: List[int] = []
        # c_vecs[i] stores the Cholesky "column" components for item i.
        c_vecs: Dict[int, np.ndarray] = {}
        # d[i] = L[i,i] - ||c_i||^2  (residual diagonal for marginal gain).
        d = np.diag(L).copy().astype(np.float64)

        for _ in range(k):
            # Pick the item with the largest residual diagonal entry.
            best = int(np.argmax(d))
            if d[best] <= _EPS:
                logger.debug(
                    "Greedy MAP: marginal gain exhausted after %d selections.",
                    len(selected),
                )
                break

            selected.append(best)
            d[best] = -np.inf  # exclude from future selection

            # Cholesky update: compute e_best = (L[:, best] - C @ c_best) / sqrt(d_best)
            # where C is the matrix of previously computed Cholesky columns.
            sqrt_d = math.sqrt(max(d[best] if d[best] != -np.inf else L[best, best] - sum(
                c_vecs[s][best] ** 2 for s in selected[:-1]
            ), _EPS))

            # Compute the new Cholesky column for the selected item.
            e = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if i == best:
                    continue
                cross = L[i, best]
                for s in selected[:-1]:
                    cross -= c_vecs[s][i] * c_vecs[s][best]
                e[i] = cross / sqrt_d
            e[best] = sqrt_d

            c_vecs[best] = e

            # Update residual diagonals.
            for i in range(n):
                if d[i] == -np.inf:
                    continue
                d[i] -= e[i] ** 2

        return selected

    # -- exact k-DPP sampling ----------------------------------------------

    def exact_sample(self, L: np.ndarray, k: int) -> List[int]:
        """Exact k-DPP sampling via spectral decomposition.

        Algorithm (Kulesza & Taskar 2012, §5):

        1. Eigen-decompose L = V Λ V^T.
        2. Select eigenvectors: include eigenvector *i* with probability
           λ_i / (λ_i + 1), then condition on exactly *k* being selected
           using the elementary symmetric polynomials.
        3. Iteratively sample items from the span of selected eigenvectors.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.
        k : int
            Subset size.

        Returns
        -------
        list of int
            Selected indices.
        """
        n = L.shape[0]
        k = min(k, n)

        eigenvalues, eigenvectors = DPPKernel.eigendecompose(L)

        # --- Phase 1: select eigenvectors via elementary symmetric polys ---
        selected_ev_indices = self._select_eigenvectors(eigenvalues, k)

        if len(selected_ev_indices) == 0:
            # Degenerate case: fall back to greedy.
            logger.warning("Exact DPP sampling found 0 eigenvectors; falling back to greedy.")
            return self.greedy_map(L, k)

        V = eigenvectors[:, selected_ev_indices].copy()  # (n, k')

        # --- Phase 2: iteratively sample items ---
        selected: List[int] = []
        remaining = list(range(n))

        k_actual = V.shape[1]
        for _ in range(k_actual):
            # Compute marginal probabilities: P(i) ∝ ||V[i, :]||^2
            probs = np.sum(V[remaining] ** 2, axis=1)
            total = probs.sum()
            if total <= _EPS:
                break
            probs /= total

            # Sample one item.
            idx_in_remaining = int(self.rng.choice(len(remaining), p=probs))
            chosen = remaining[idx_in_remaining]
            selected.append(chosen)
            remaining.pop(idx_in_remaining)

            if V.shape[1] <= 1:
                break

            # Project V onto the orthogonal complement of V[chosen, :].
            v_chosen = V[chosen].copy()
            norm = np.linalg.norm(v_chosen)
            if norm < _EPS:
                break
            v_chosen /= norm

            V = V - np.outer(V @ v_chosen, v_chosen)

            # Remove one column dimension (rank decreased by 1).
            # QR re-orthogonalisation to maintain numerical stability.
            Q, R = np.linalg.qr(V)
            # Keep only columns with non-negligible norm.
            col_norms = np.abs(np.diag(R))
            keep = col_norms > _EPS
            if not np.any(keep):
                break
            V = Q[:, keep]

        return selected

    def _select_eigenvectors(
        self, eigenvalues: np.ndarray, k: int
    ) -> List[int]:
        """Select eigenvectors for exact k-DPP sampling.

        Uses the elementary symmetric polynomial recurrence to compute
        the conditional probabilities of including each eigenvector,
        conditioned on exactly *k* eigenvectors being selected.

        Parameters
        ----------
        eigenvalues : np.ndarray
            ``(n,)`` non-negative eigenvalues.
        k : int
            Target subset size.

        Returns
        -------
        list of int
            Indices of selected eigenvectors.
        """
        n = len(eigenvalues)
        k = min(k, n)

        # Compute elementary symmetric polynomials E[l, j] where
        # E[l, j] = e_l(λ_1, …, λ_j).
        E = np.zeros((k + 1, n + 1), dtype=np.float64)
        E[0, :] = 1.0  # e_0 = 1

        for j in range(1, n + 1):
            lam = eigenvalues[j - 1]
            for l in range(1, k + 1):
                E[l, j] = E[l, j - 1] + lam * E[l - 1, j - 1]

        # Sample eigenvectors in reverse order.
        selected: List[int] = []
        remaining_k = k

        for j in range(n, 0, -1):
            if remaining_k == 0:
                break
            lam = eigenvalues[j - 1]
            # Probability of including eigenvector j-1:
            # P = λ_j * E[remaining_k - 1, j - 1] / E[remaining_k, j]
            denom = E[remaining_k, j]
            if denom <= _EPS:
                continue
            prob = lam * E[remaining_k - 1, j - 1] / denom
            prob = np.clip(prob, 0.0, 1.0)

            if self.rng.random() < prob:
                selected.append(j - 1)
                remaining_k -= 1

        return selected

    # -- MCMC sampling -----------------------------------------------------

    def mcmc_sample(
        self,
        L: np.ndarray,
        k: int,
        n_iter: int = 1000,
    ) -> List[int]:
        """MCMC (Metropolis–Hastings) sampling for k-DPP.

        Initialises with a random subset and proposes single-item swaps.
        Accepts swaps according to the DPP probability ratio.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.
        k : int
            Subset size.
        n_iter : int
            Number of MCMC iterations.

        Returns
        -------
        list of int
            Selected indices after mixing.
        """
        n = L.shape[0]
        k = min(k, n)

        # Initialise with a random subset.
        current = list(self.rng.choice(n, size=k, replace=False))
        current_log_det = self.log_det_marginal(L, current)

        best = list(current)
        best_log_det = current_log_det

        n_accepted = 0

        for it in range(n_iter):
            # Propose: swap one element in the set with one outside.
            idx_in = int(self.rng.integers(k))
            remaining = [i for i in range(n) if i not in current]
            if len(remaining) == 0:
                break
            idx_out = int(self.rng.integers(len(remaining)))

            proposed = list(current)
            proposed[idx_in] = remaining[idx_out]

            proposed_log_det = self.log_det_marginal(L, proposed)

            # Metropolis acceptance ratio (log scale).
            log_alpha = proposed_log_det - current_log_det
            if log_alpha >= 0.0 or np.log(self.rng.random()) < log_alpha:
                current = proposed
                current_log_det = proposed_log_det
                n_accepted += 1

                if current_log_det > best_log_det:
                    best = list(current)
                    best_log_det = current_log_det

        acceptance_rate = n_accepted / max(n_iter, 1)
        logger.debug(
            "MCMC DPP sampling: %d iterations, acceptance rate %.3f",
            n_iter,
            acceptance_rate,
        )
        return best

    # -- utility -----------------------------------------------------------

    @staticmethod
    def log_det_marginal(L: np.ndarray, S: List[int]) -> float:
        """Log-determinant of the principal sub-matrix ``L[S, :][:, S]``.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.
        S : list of int
            Subset indices.

        Returns
        -------
        float
            ``log det(L_S)``.  Returns ``-inf`` when *S* is empty or
            the sub-matrix is singular.
        """
        if len(S) == 0:
            return float("-inf")
        L_S = L[np.ix_(S, S)]
        return DPPKernel.log_determinant(L_S)

    @staticmethod
    def _cholesky_update(
        L_chol: np.ndarray,
        new_col: np.ndarray,
    ) -> np.ndarray:
        """Rank-one Cholesky update.

        Given the lower Cholesky factor of a ``k × k`` matrix and a new
        column/row, returns the ``(k+1) × (k+1)`` lower Cholesky factor.

        Parameters
        ----------
        L_chol : np.ndarray
            ``(k, k)`` lower-triangular Cholesky factor.
        new_col : np.ndarray
            ``(k+1,)`` vector representing the new column of the expanded
            matrix (last element is the new diagonal entry).

        Returns
        -------
        np.ndarray
            ``(k+1, k+1)`` lower-triangular Cholesky factor.
        """
        k = L_chol.shape[0]
        # Solve L_chol @ z = new_col[:k]
        z = sp_linalg.solve_triangular(
            L_chol, new_col[:k], lower=True, check_finite=False
        )
        new_diag_sq = new_col[k] - np.dot(z, z)
        if new_diag_sq <= 0:
            new_diag_sq = _EPS
        new_diag = math.sqrt(new_diag_sq)

        # Build expanded factor.
        L_new = np.zeros((k + 1, k + 1), dtype=L_chol.dtype)
        L_new[:k, :k] = L_chol
        L_new[k, :k] = z
        L_new[k, k] = new_diag
        return L_new


# =========================================================================
# QualityModel — sequence scoring
# =========================================================================


class QualityModel:
    """Score candidate sequences along multiple quality axes.

    All scoring functions return a scalar *quality* value where higher is
    better.  When a ``logit_source`` is required, it should follow the
    :class:`LogitSource` protocol.

    Parameters
    ----------
    quality_weight : float
        Exponent applied to the raw score to modulate quality emphasis.
    """

    def __init__(self, quality_weight: float = 0.5) -> None:
        self.quality_weight = quality_weight

    # -- single-sequence ---------------------------------------------------

    def score_sequence(
        self,
        sequence: TokenSequence,
        logit_source: Optional[LogitSource] = None,
        prompt_length: int = 0,
    ) -> float:
        """Composite quality score for a single sequence.

        Combines log-probability (if *logit_source* available) and length
        normalisation.  The result is raised to ``self.quality_weight``.

        Parameters
        ----------
        sequence : list of int
            Full token sequence (prompt + generated).
        logit_source : LogitSource or None
            Model to evaluate log-probabilities.
        prompt_length : int
            Length of the prompt prefix to skip when scoring.

        Returns
        -------
        float
            Non-negative quality score.
        """
        if logit_source is not None:
            lp = self.log_probability_score(sequence, logit_source, prompt_length)
            ln = self.length_normalized_score(sequence, prompt_length)
            raw = math.exp(lp) * ln
        else:
            raw = self.length_normalized_score(sequence, prompt_length)

        return max(raw, _EPS) ** self.quality_weight

    # -- batch scoring -----------------------------------------------------

    def score_sequences(
        self,
        sequences: List[TokenSequence],
        logit_source: Optional[LogitSource] = None,
        prompt_length: int = 0,
    ) -> np.ndarray:
        """Score a batch of sequences, returning an ``(n,)`` array.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.
        logit_source : LogitSource or None
            Model for log-probability scoring.
        prompt_length : int
            Prompt prefix length.

        Returns
        -------
        np.ndarray
            ``(n,)`` quality scores.
        """
        scores = np.array(
            [
                self.score_sequence(seq, logit_source, prompt_length)
                for seq in sequences
            ],
            dtype=np.float64,
        )
        return scores

    # -- component scores --------------------------------------------------

    def log_probability_score(
        self,
        sequence: TokenSequence,
        logit_source: LogitSource,
        prompt_length: int = 0,
    ) -> float:
        """Sum of log-probabilities of generated tokens under the model.

        Parameters
        ----------
        sequence : list of int
            Full sequence (prompt + generated).
        logit_source : LogitSource
            Model producing logits.
        prompt_length : int
            Number of prompt tokens to skip.

        Returns
        -------
        float
            Total log-probability of the generated segment.
        """
        total_lp = 0.0
        start = max(prompt_length, 1)
        for t in range(start, len(sequence)):
            prefix = [sequence[:t]]
            logits = logit_source(prefix)  # (1, vocab)
            log_p = _log_softmax(logits[0])
            total_lp += float(log_p[sequence[t]])
        return total_lp

    def perplexity_score(
        self,
        sequence: TokenSequence,
        logit_source: LogitSource,
        prompt_length: int = 0,
    ) -> float:
        """Perplexity of the generated segment (lower is better).

        Returns the *negative* perplexity so that higher values are better
        (consistent with quality conventions).

        Parameters
        ----------
        sequence : list of int
            Full sequence.
        logit_source : LogitSource
            Model.
        prompt_length : int
            Prompt length.

        Returns
        -------
        float
            Negative perplexity.
        """
        gen_len = len(sequence) - prompt_length
        if gen_len <= 0:
            return 0.0
        lp = self.log_probability_score(sequence, logit_source, prompt_length)
        ppl = math.exp(-lp / gen_len)
        return -ppl

    def length_normalized_score(
        self,
        sequence: TokenSequence,
        prompt_length: int = 0,
    ) -> float:
        """Simple length-based quality heuristic.

        Rewards moderate-length generations and penalises very short or
        very long outputs.

        Parameters
        ----------
        sequence : list of int
            Full sequence.
        prompt_length : int
            Prompt prefix length.

        Returns
        -------
        float
            Non-negative score in ``(0, 1]``.
        """
        gen_len = len(sequence) - prompt_length
        if gen_len <= 0:
            return _EPS
        # Gaussian-shaped length preference centred at 50 tokens.
        target = 50.0
        sigma = 40.0
        return float(math.exp(-0.5 * ((gen_len - target) / sigma) ** 2))

    def coherence_score(self, sequence: TokenSequence) -> float:
        """Approximate coherence via n-gram self-overlap.

        Measures the ratio of repeated bigrams to total bigrams.  A lower
        repetition rate signals higher coherence.

        Parameters
        ----------
        sequence : list of int
            Token sequence.

        Returns
        -------
        float
            Coherence score in ``[0, 1]``.
        """
        if len(sequence) < 2:
            return 1.0
        bigrams = [tuple(sequence[i: i + 2]) for i in range(len(sequence) - 1)]
        total = len(bigrams)
        unique = len(set(bigrams))
        if total == 0:
            return 1.0
        repetition_rate = 1.0 - unique / total
        coherence = 1.0 - repetition_rate
        return float(np.clip(coherence, 0.0, 1.0))


# =========================================================================
# DPPAnalyzer — diagnostic tools
# =========================================================================


class DPPAnalyzer:
    """Diagnostic and visualisation utilities for DPP kernels and selections.

    Provides methods to inspect the spectral properties of kernel matrices,
    quantify diversity/quality trade-offs, and prepare data for
    visualisations.
    """

    # -- spectral analysis -------------------------------------------------

    @staticmethod
    def analyze_kernel_spectrum(L: np.ndarray) -> Dict[str, Any]:
        """Analyse the eigenvalue spectrum of a kernel matrix.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` symmetric PSD kernel.

        Returns
        -------
        dict
            * ``eigenvalues`` — sorted eigenvalue array.
            * ``effective_rank`` — number of eigenvalues above 1 %% of max.
            * ``trace`` — sum of eigenvalues.
            * ``log_det`` — log-determinant.
            * ``condition_number`` — ratio of largest to smallest positive
              eigenvalue.
            * ``spectral_gap`` — difference between two largest eigenvalues.
            * ``entropy`` — spectral entropy  -Σ p_i log(p_i)  where
              p_i = λ_i / Σ λ_j.
            * ``expected_size`` — expected cardinality  Σ λ_i / (λ_i + 1).
        """
        eigenvalues, _ = DPPKernel.eigendecompose(L)
        pos = eigenvalues[eigenvalues > _EPS]

        # Effective rank.
        if len(pos) == 0:
            effective_rank = 0
        else:
            threshold = 0.01 * pos[-1]
            effective_rank = int(np.sum(pos >= threshold))

        # Spectral entropy.
        if len(pos) > 0:
            p = pos / pos.sum()
            entropy = -float(np.sum(p * np.log(p + _EPS)))
        else:
            entropy = 0.0

        # Expected DPP size.
        expected_size = float(np.sum(eigenvalues / (eigenvalues + 1.0)))

        # Spectral gap.
        sorted_ev = np.sort(eigenvalues)[::-1]
        spectral_gap = float(sorted_ev[0] - sorted_ev[1]) if len(sorted_ev) > 1 else 0.0

        return {
            "eigenvalues": eigenvalues,
            "effective_rank": effective_rank,
            "trace": float(np.sum(eigenvalues)),
            "log_det": DPPKernel.log_determinant(L),
            "condition_number": DPPKernel.condition_number(L),
            "spectral_gap": spectral_gap,
            "entropy": entropy,
            "expected_size": expected_size,
            "num_eigenvalues": len(eigenvalues),
            "num_positive": len(pos),
        }

    # -- diversity / quality metrics ---------------------------------------

    @staticmethod
    def diversity_gain(
        selected: List[int],
        all_candidates: np.ndarray,
    ) -> float:
        """Measure diversity gain of the selected subset.

        Computes the mean pairwise distance among selected items relative
        to the mean pairwise distance among *all* candidates.  A value > 1
        indicates the selected subset is *more* diverse than a random
        same-sized sample would be on average.

        Parameters
        ----------
        selected : list of int
            Indices of selected items.
        all_candidates : np.ndarray
            ``(n, d)`` embedding matrix for all candidates.

        Returns
        -------
        float
            Diversity gain ratio.
        """
        if len(selected) < 2:
            return 0.0
        n = all_candidates.shape[0]

        def _mean_pairwise_dist(indices: List[int]) -> float:
            vecs = all_candidates[indices]
            sq = np.sum(vecs ** 2, axis=1, keepdims=True)
            D = sq - 2.0 * vecs @ vecs.T + sq.T
            D = np.clip(D, 0.0, None)
            dists = np.sqrt(D)
            m = len(indices)
            if m < 2:
                return 0.0
            return float(np.sum(dists) / (m * (m - 1)))

        selected_div = _mean_pairwise_dist(selected)
        all_div = _mean_pairwise_dist(list(range(min(n, 500))))  # cap for speed

        if all_div <= _EPS:
            return 0.0
        return selected_div / all_div

    @staticmethod
    def quality_loss(
        selected: List[int],
        all_quality_scores: np.ndarray,
    ) -> float:
        """Fraction of quality lost by selecting a subset.

        Compares the mean quality of selected items to the mean of the
        top-*k* items by quality (where *k* = ``len(selected)``).

        Parameters
        ----------
        selected : list of int
            Indices of selected items.
        all_quality_scores : np.ndarray
            ``(n,)`` quality scores for all candidates.

        Returns
        -------
        float
            Quality loss in ``[0, 1]``.  0 means no loss.
        """
        if len(selected) == 0:
            return 1.0
        k = len(selected)
        top_k_mean = float(np.mean(np.sort(all_quality_scores)[::-1][:k]))
        selected_mean = float(np.mean(all_quality_scores[selected]))

        if top_k_mean <= _EPS:
            return 0.0
        return max(0.0, 1.0 - selected_mean / top_k_mean)

    @staticmethod
    def pareto_analysis(
        L: np.ndarray,
        quality_scores: np.ndarray,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Analyse the quality–diversity Pareto frontier.

        For each *k* in *k_values*, performs greedy DPP selection and
        records the resulting quality and diversity metrics.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.
        quality_scores : np.ndarray
            ``(n,)`` quality scores.
        k_values : list of int or None
            Subset sizes to evaluate.  Defaults to
            ``[1, 2, 5, 10, 20, 50]`` capped by *n*.

        Returns
        -------
        dict
            * ``k_values`` — evaluated k values.
            * ``log_dets`` — log-determinant of selected sub-matrix.
            * ``mean_qualities`` — mean quality of selected items.
            * ``quality_losses`` — quality loss for each k.
        """
        n = L.shape[0]
        if k_values is None:
            k_values = [k for k in [1, 2, 5, 10, 20, 50] if k <= n]

        sampler = DPPSampler()
        log_dets: List[float] = []
        mean_qualities: List[float] = []
        q_losses: List[float] = []

        for k in k_values:
            selected = sampler.greedy_map(L, k)
            ld = DPPSampler.log_det_marginal(L, selected)
            mq = float(np.mean(quality_scores[selected])) if selected else 0.0
            ql = DPPAnalyzer.quality_loss(selected, quality_scores)

            log_dets.append(ld)
            mean_qualities.append(mq)
            q_losses.append(ql)

        return {
            "k_values": k_values,
            "log_dets": log_dets,
            "mean_qualities": mean_qualities,
            "quality_losses": q_losses,
        }

    @staticmethod
    def visualize_kernel(L: np.ndarray) -> Dict[str, Any]:
        """Prepare kernel matrix data for heatmap visualisation.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` kernel matrix.

        Returns
        -------
        dict
            * ``matrix`` — kernel values as a nested list.
            * ``shape`` — ``(rows, cols)``.
            * ``min_val``, ``max_val`` — value range.
            * ``diagonal`` — diagonal entries.
            * ``off_diagonal_mean`` — mean of off-diagonal entries.
        """
        n = L.shape[0]
        diag = np.diag(L)
        mask = ~np.eye(n, dtype=bool)
        off_diag = L[mask]

        return {
            "matrix": L.tolist(),
            "shape": (n, n),
            "min_val": float(np.min(L)),
            "max_val": float(np.max(L)),
            "diagonal": diag.tolist(),
            "off_diagonal_mean": float(np.mean(off_diag)) if off_diag.size > 0 else 0.0,
            "off_diagonal_std": float(np.std(off_diag)) if off_diag.size > 0 else 0.0,
        }


# =========================================================================
# CandidateEmbedder — embedding sequences for kernel construction
# =========================================================================


class CandidateEmbedder:
    """Compute dense embeddings for token sequences.

    Attempts to load a sentence-transformer model specified by *model_name*.
    Falls back to an n-gram bag-of-words representation when the model is
    unavailable.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model identifier.
    fallback_dim : int
        Dimensionality of the n-gram fallback representation.
    ngram_size : int
        N-gram size for the fallback.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        fallback_dim: int = 256,
        ngram_size: int = 3,
    ) -> None:
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.ngram_size = ngram_size
        self._model: Any = None
        self._model_loaded = False
        self._load_attempted = False

    def _try_load_model(self) -> None:
        """Attempt to load the sentence-transformer model (once)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            self._model_loaded = True
            logger.info("Loaded embedding model: %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Could not load embedding model '%s' (%s); using n-gram fallback.",
                self.model_name,
                exc,
            )

    def embed(self, sequences: List[TokenSequence]) -> np.ndarray:
        """Embed a list of token sequences.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.

        Returns
        -------
        np.ndarray
            ``(n, d)`` embedding matrix.
        """
        self._try_load_model()

        if self._model_loaded:
            return self._embed_with_model(sequences)
        return self._embed_ngram(sequences)

    def _embed_with_model(self, sequences: List[TokenSequence]) -> np.ndarray:
        """Embed using the loaded sentence-transformer."""
        texts = [" ".join(str(t) for t in seq) for seq in sequences]
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float64)

    def _embed_ngram(self, sequences: List[TokenSequence]) -> np.ndarray:
        """Hash-based n-gram bag-of-words embedding (fallback).

        Uses feature hashing (the hashing trick) to produce a fixed-dimensional
        representation without requiring a vocabulary.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.

        Returns
        -------
        np.ndarray
            ``(n, fallback_dim)`` embedding matrix.
        """
        n = len(sequences)
        X = np.zeros((n, self.fallback_dim), dtype=np.float64)

        for i, seq in enumerate(sequences):
            for j in range(len(seq) - self.ngram_size + 1):
                ng = tuple(seq[j: j + self.ngram_size])
                h = hash(ng) % self.fallback_dim
                sign = 1.0 if (hash(ng) // self.fallback_dim) % 2 == 0 else -1.0
                X[i, h] += sign

            # L2-normalise.
            norm = np.linalg.norm(X[i])
            if norm > _EPS:
                X[i] /= norm

        return X


# =========================================================================
# DPPReranking — main decoding algorithm
# =========================================================================


class DPPReranking(DecodingAlgorithm):
    """Two-phase DPP reranking decoding algorithm.

    **Phase 1 — Candidate generation:**
    Generate ``candidate_pool_size`` sequences via temperature sampling.

    **Phase 2 — DPP selection:**
    Embed candidates, build the L-ensemble kernel, and select ``select_k``
    diverse, high-quality sequences via DPP sampling.

    Parameters
    ----------
    config : DPPConfig
        Algorithm configuration.
    """

    def __init__(self, config: DPPConfig) -> None:
        super().__init__(config)
        self.dpp_config: DPPConfig = config
        self._kernel = DPPKernel(
            kernel_type=config.kernel_type,
            bandwidth=config.kernel_bandwidth,
            polynomial_degree=config.polynomial_degree,
            polynomial_c=config.polynomial_c,
            string_kernel_n=config.string_kernel_n,
            regularization_eps=config.regularization_eps,
        )
        self._sampler = DPPSampler(
            rng=self._rng or np.random.default_rng(),
        )
        self._quality_model = QualityModel(quality_weight=config.quality_weight)
        self._embedder = CandidateEmbedder(model_name=config.embedding_model)

    # -- properties --------------------------------------------------------

    @property
    def description(self) -> str:
        return (
            f"DPP reranking ({self.dpp_config.sampling_method}) "
            f"selecting {self.dpp_config.select_k} from "
            f"{self.dpp_config.candidate_pool_size} candidates"
        )

    # -- config validation -------------------------------------------------

    def validate_config(self) -> List[str]:
        """Validate the DPP configuration."""
        return self.dpp_config.validate()

    # -- main generate override --------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate diverse sequences via DPP reranking.

        Overrides the base class to implement the two-phase pipeline.

        Parameters
        ----------
        logit_source : LogitSource
            Model logit provider.
        prompt_ids : list of int
            Prompt token ids.

        Returns
        -------
        list of TokenSequence
            Selected diverse sequences, sorted by quality.
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)
            self._sampler.rng = self._rng

        t0 = time.monotonic()

        # Phase 1: generate candidate pool.
        logger.info(
            "DPP Phase 1: generating %d candidates via %s sampling.",
            self.dpp_config.candidate_pool_size,
            self.dpp_config.candidate_generation,
        )
        candidates = self._generate_candidates(
            logit_source, prompt_ids, self.dpp_config.candidate_pool_size
        )
        t_gen = time.monotonic() - t0
        logger.info(
            "Phase 1 complete: %d candidates generated in %.2fs.",
            len(candidates),
            t_gen,
        )

        if len(candidates) == 0:
            logger.warning("No candidates generated; returning empty list.")
            return []

        # Phase 2: DPP selection.
        logger.info(
            "DPP Phase 2: selecting %d sequences via %s.",
            self.dpp_config.select_k,
            self.dpp_config.sampling_method,
        )

        # Embed candidates.
        embeddings = self._compute_embeddings(candidates)

        # Compute quality scores.
        if self.dpp_config.use_quality_model:
            quality_scores = self._compute_quality_scores(
                candidates, logit_source, len(prompt_ids)
            )
        else:
            quality_scores = np.ones(len(candidates), dtype=np.float64)

        # Build kernel.
        L = self._build_quality_diversity_kernel(embeddings, quality_scores)

        # Sample.
        selected_indices = self._dpp_select(L, self.dpp_config.select_k)

        t_total = time.monotonic() - t0
        logger.info(
            "DPP selection complete: %d items selected in %.2fs total.",
            len(selected_indices),
            t_total,
        )

        # Return selected sequences sorted by quality.
        selected = [(quality_scores[i], candidates[i]) for i in selected_indices]
        selected.sort(key=lambda p: p[0], reverse=True)
        return [seq for _, seq in selected]

    # -- Phase 1: candidate generation -------------------------------------

    def _generate_candidates(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n: int,
    ) -> List[TokenSequence]:
        """Generate *n* candidate sequences via temperature sampling.

        Each candidate is independently sampled from the model using the
        configured ``candidate_temperature``.

        Parameters
        ----------
        logit_source : LogitSource
            Model logit provider.
        prompt_ids : list of int
            Prompt token ids.
        n : int
            Number of candidates to generate.

        Returns
        -------
        list of TokenSequence
            Generated sequences (excluding the prompt prefix).
        """
        rng = self._rng or np.random.default_rng()
        temperature = self.dpp_config.candidate_temperature
        max_tokens = self.config.max_new_tokens
        min_tokens = self.config.min_new_tokens
        eos_id = self.config.eos_token_id
        rep_penalty = self.config.repetition_penalty
        no_repeat_n = self.config.no_repeat_ngram_size

        candidates: List[TokenSequence] = []

        for idx in range(n):
            seq = list(prompt_ids)
            for step in range(max_tokens):
                logits = logit_source([seq])  # (1, vocab)
                logits = logits[0].astype(np.float64)

                # Apply constraints.
                if rep_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, seq, rep_penalty)
                if no_repeat_n > 0:
                    logits = self._apply_no_repeat_ngram(logits, seq, no_repeat_n)

                gen_len = len(seq) - len(prompt_ids)
                logits = self._enforce_min_length(logits, gen_len, min_tokens, eos_id)

                # Temperature sampling.
                logits = logits / temperature
                log_probs = _log_softmax(logits)
                probs = np.exp(log_probs)
                probs = probs / probs.sum()  # re-normalise for safety

                token = int(rng.choice(len(probs), p=probs))
                seq.append(token)

                if eos_id is not None and token == eos_id:
                    break

            # Strip prompt; store only generated tokens.
            generated = seq[len(prompt_ids):]
            candidates.append(generated)

        return candidates

    # -- embedding ---------------------------------------------------------

    def _compute_embeddings(self, sequences: List[TokenSequence]) -> np.ndarray:
        """Compute dense embeddings for a list of token sequences.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.

        Returns
        -------
        np.ndarray
            ``(n, d)`` embedding matrix.
        """
        return self._embedder.embed(sequences)

    # -- quality scoring ---------------------------------------------------

    def _compute_quality_scores(
        self,
        sequences: List[TokenSequence],
        logit_source: Optional[LogitSource] = None,
        prompt_length: int = 0,
    ) -> np.ndarray:
        """Compute quality scores for candidate sequences.

        Parameters
        ----------
        sequences : list of list of int
            Generated token sequences (*without* prompt prefix).
        logit_source : LogitSource or None
            Model for log-probability scoring.
        prompt_length : int
            Original prompt length (used for context but sequences here
            are already stripped of prompt).

        Returns
        -------
        np.ndarray
            ``(n,)`` non-negative quality scores.
        """
        scores = self._quality_model.score_sequences(
            sequences, logit_source=None, prompt_length=0
        )
        scores = np.clip(scores, _EPS, None)
        return scores

    # -- kernel construction -----------------------------------------------

    def _build_kernel_matrix(self, sequences: List[TokenSequence]) -> np.ndarray:
        """Build the L-ensemble kernel from raw sequences.

        Embeds sequences and constructs a similarity-only kernel (no quality
        weighting).

        Parameters
        ----------
        sequences : list of list of int
            Token sequences.

        Returns
        -------
        np.ndarray
            ``(n, n)`` L-ensemble kernel.
        """
        embeddings = self._compute_embeddings(sequences)
        return self._kernel.build_L_ensemble(embeddings)

    def _build_quality_diversity_kernel(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray,
    ) -> np.ndarray:
        """Build the quality–diversity L-ensemble kernel.

        Factorises as  L = diag(q) S diag(q)  where *S* is the similarity
        matrix and *q* are quality scores.

        Parameters
        ----------
        embeddings : np.ndarray
            ``(n, d)`` candidate embeddings.
        quality_scores : np.ndarray
            ``(n,)`` quality scores.

        Returns
        -------
        np.ndarray
            ``(n, n)`` L-ensemble kernel.
        """
        return self._kernel.build_L_ensemble(embeddings, quality_scores)

    # -- DPP selection dispatch --------------------------------------------

    def _dpp_select(self, L: np.ndarray, k: int) -> List[int]:
        """Select *k* items from the DPP defined by *L*.

        Dispatches to the configured sampling method.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` L-ensemble kernel.
        k : int
            Number of items to select.

        Returns
        -------
        list of int
            Selected indices.
        """
        method = self.dpp_config.sampling_method
        if method == "greedy":
            return self._dpp_sample_greedy(L, k)
        elif method == "exact":
            return self._dpp_sample_exact(L, k)
        elif method == "mcmc":
            return self._dpp_sample_mcmc(
                L, k, self.dpp_config.max_mcmc_iterations
            )
        else:
            raise ValueError(f"Unknown sampling_method: {method}")

    def _dpp_sample_greedy(self, L: np.ndarray, k: int) -> List[int]:
        """Greedy MAP inference for k-DPP.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` kernel.
        k : int
            Subset size.

        Returns
        -------
        list of int
            Selected indices.
        """
        return self._sampler.greedy_map(L, k)

    def _dpp_sample_exact(self, L: np.ndarray, k: int) -> List[int]:
        """Exact k-DPP sampling.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` kernel.
        k : int
            Subset size.

        Returns
        -------
        list of int
            Selected indices.
        """
        return self._sampler.exact_sample(L, k)

    def _dpp_sample_mcmc(
        self,
        L: np.ndarray,
        k: int,
        n_iterations: int = 1000,
    ) -> List[int]:
        """MCMC-based k-DPP sampling.

        Parameters
        ----------
        L : np.ndarray
            ``(n, n)`` kernel.
        k : int
            Subset size.
        n_iterations : int
            Number of MCMC iterations.

        Returns
        -------
        list of int
            Selected indices.
        """
        return self._sampler.mcmc_sample(L, k, n_iterations)

    # -- step (required by ABC but unused in two-phase pipeline) -----------

    def _step(
        self,
        state: DecodingState,
        logit_source: LogitSource,
    ) -> DecodingState:
        """Single-step generation (used internally during candidate generation).

        DPP reranking overrides :meth:`generate` entirely, so this method
        is only invoked if the base-class generation loop is called
        directly (e.g., for debugging).  It performs simple temperature
        sampling.

        Parameters
        ----------
        state : DecodingState
            Current generation state.
        logit_source : LogitSource
            Model logit provider.

        Returns
        -------
        DecodingState
            Updated state.
        """
        rng = self._rng or np.random.default_rng()
        temperature = self.dpp_config.candidate_temperature

        active = state.active_indices()
        if not active:
            return state

        # Batch query for active sequences.
        batch = [state.get_sequence(i) for i in active]
        all_logits = logit_source(batch)  # (batch, vocab)

        for batch_idx, seq_idx in enumerate(active):
            logits = all_logits[batch_idx].astype(np.float64)
            seq = state.get_sequence(seq_idx)

            # Constraints.
            if self.config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, seq, self.config.repetition_penalty
                )
            if self.config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram(
                    logits, seq, self.config.no_repeat_ngram_size
                )

            prompt_len = state.metadata.get("prompt_length", 0)
            gen_len = len(seq) - prompt_len
            logits = self._enforce_min_length(
                logits, gen_len, self.config.min_new_tokens, self.config.eos_token_id
            )

            # Temperature sampling.
            logits = logits / temperature
            log_probs = _log_softmax(logits)
            probs = np.exp(log_probs)
            probs = probs / probs.sum()

            token = int(rng.choice(len(probs), p=probs))
            state.update_sequence(seq_idx, token)

            # Update score.
            state.scores[seq_idx] += float(log_probs[token])

            # Check EOS.
            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                state.mark_finished(seq_idx)

        return state

    def _should_stop(self, state: DecodingState) -> bool:
        """Algorithm-level stopping check (always False for DPP)."""
        return False

    # -- analysis convenience ----------------------------------------------

    def analyze_selection(
        self,
        L: np.ndarray,
        selected: List[int],
        quality_scores: np.ndarray,
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """Run diagnostic analysis on a completed DPP selection.

        Parameters
        ----------
        L : np.ndarray
            Kernel matrix used for selection.
        selected : list of int
            Indices of selected items.
        quality_scores : np.ndarray
            Quality scores for all candidates.
        embeddings : np.ndarray
            Embeddings for all candidates.

        Returns
        -------
        dict
            Diagnostic metrics including spectrum analysis, diversity gain,
            and quality loss.
        """
        spectrum = DPPAnalyzer.analyze_kernel_spectrum(L)
        div_gain = DPPAnalyzer.diversity_gain(selected, embeddings)
        q_loss = DPPAnalyzer.quality_loss(selected, quality_scores)
        log_det = DPPSampler.log_det_marginal(L, selected)

        return {
            "spectrum": spectrum,
            "diversity_gain": div_gain,
            "quality_loss": q_loss,
            "selected_log_det": log_det,
            "num_selected": len(selected),
            "num_candidates": L.shape[0],
            "mean_selected_quality": float(np.mean(quality_scores[selected]))
            if selected
            else 0.0,
            "mean_all_quality": float(np.mean(quality_scores)),
        }

    def get_kernel(self) -> DPPKernel:
        """Return the internal :class:`DPPKernel` instance."""
        return self._kernel

    def get_sampler(self) -> DPPSampler:
        """Return the internal :class:`DPPSampler` instance."""
        return self._sampler

    def get_quality_model(self) -> QualityModel:
        """Return the internal :class:`QualityModel` instance."""
        return self._quality_model

    def get_analyzer(self) -> DPPAnalyzer:
        """Return a new :class:`DPPAnalyzer` instance."""
        return DPPAnalyzer()


# =========================================================================
# Convenience factory
# =========================================================================


def create_dpp_reranking(
    candidate_pool_size: int = 100,
    select_k: int = 20,
    kernel_type: str = "rbf",
    kernel_bandwidth: float = 1.0,
    quality_weight: float = 0.5,
    sampling_method: str = "greedy",
    candidate_temperature: float = 1.0,
    use_quality_model: bool = True,
    max_new_tokens: int = 100,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> DPPReranking:
    """Create a :class:`DPPReranking` instance with common defaults.

    Parameters
    ----------
    candidate_pool_size : int
        Number of candidates to generate.
    select_k : int
        Number of items to select via DPP.
    kernel_type : str
        Kernel for the similarity matrix.
    kernel_bandwidth : float
        RBF bandwidth.
    quality_weight : float
        Quality exponent.
    sampling_method : str
        ``"greedy"``, ``"exact"``, or ``"mcmc"``.
    candidate_temperature : float
        Temperature for candidate generation.
    use_quality_model : bool
        Whether to use quality scoring.
    max_new_tokens : int
        Maximum generated tokens per candidate.
    seed : int or None
        Random seed.
    **kwargs
        Extra keyword arguments forwarded to :class:`DPPConfig`.

    Returns
    -------
    DPPReranking
        Configured DPP reranking algorithm.
    """
    config = DPPConfig(
        candidate_pool_size=candidate_pool_size,
        select_k=select_k,
        kernel_type=kernel_type,
        kernel_bandwidth=kernel_bandwidth,
        quality_weight=quality_weight,
        sampling_method=sampling_method,
        candidate_temperature=candidate_temperature,
        use_quality_model=use_quality_model,
        max_new_tokens=max_new_tokens,
        seed=seed,
        **kwargs,
    )
    return DPPReranking(config)
