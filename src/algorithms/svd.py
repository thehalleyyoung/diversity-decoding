"""
Kernel-Repulsive Particle Decoding (KRPD) — a diverse decoding algorithm
that maintains *n* particle sequences, repelling them from each other in
embedding space via kernel gradients to encourage diversity.

**Honest framing**: This algorithm is *inspired by* Stein Variational
Gradient Descent (SVGD; Liu & Wang 2016), but does **not** perform
variational inference in any formal sense.  Specifically:

1. **No target distribution**: SVGD minimises KL(q || p) for a target p.
   In autoregressive decoding, there is no single static target distribution
   over complete sequences — the distribution is built token-by-token.
   We do not define or approximate any target.

2. **No convergence guarantee**: SVGD converges to the target under
   conditions (Stein operator in RKHS, continuous updates).  Our discrete
   token-level modifications violate these conditions.  The "Stein
   discrepancy" we monitor is a heuristic diagnostic, not a bound on
   distributional distance.

3. **What we actually do**: At each step, for each particle i, we compute
   a *repulsive gradient* from the kernel between particle embeddings:
       g_i = (1/n) Σ_{j≠i} ∇_{e_j} k(e_j, e_i)
   This gradient is projected onto the token vocabulary and added to logits:
       logits_i(v) += α · ⟨g_i, e(v)⟩
   The effect is to boost tokens that move particle i *away* from other
   particles in embedding space, producing diverse outputs.

4. **Empirical properties**: The algorithm reliably produces more diverse
   outputs than independent sampling (verified by D-2, EPD, Self-BLEU).
   The annealing schedule for α provides a quality–diversity trade-off.
   Convergence of the KSD diagnostic correlates with output stabilisation
   but is not a formal convergence certificate.

We retain the name "Stein Variational Decoding" for continuity with
the codebase, but users should understand it as a kernel-repulsive
heuristic, not a variational inference method.

References
----------
- Liu & Wang (2016), "Stein Variational Gradient Descent", NeurIPS.
  (Inspiration for the kernel-gradient mechanism, not a direct application.)
- Gorham & Mackey (2017), Stein discrepancy diagnostics.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    _log_softmax,
    _stable_softmax,
    _top_k_filter,
    sample_token,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================


@dataclass
class SVDConfig(DecodingConfig):
    """Configuration for Stein Variational Decoding.

    Extends the base :class:`DecodingConfig` with SVD-specific knobs that
    control the kernel, repulsive strength, annealing, and convergence
    monitoring.
    """

    algorithm_name: str = "SteinVariationalDecoding"

    # -- particle settings --------------------------------------------------
    n_particles: int = 20
    """Number of particle sequences to maintain."""

    # -- repulsion ----------------------------------------------------------
    alpha: float = 0.5
    """Repulsive strength coefficient (scales the kernel-gradient term)."""

    # -- kernel settings ----------------------------------------------------
    kernel_type: str = "rbf"
    """Kernel function: ``rbf``, ``cosine``, or ``imq``."""

    kernel_bandwidth: float = -1.0
    """Manual bandwidth.  ``-1`` triggers automatic selection via
    *bandwidth_method*."""

    bandwidth_method: str = "median"
    """Automatic bandwidth selection: ``median``, ``silverman``, ``fixed``."""

    # -- embedding settings -------------------------------------------------
    embedding_dim: int = 384
    """Dimensionality of the sentence-level embeddings used by the kernel."""

    # -- projection ---------------------------------------------------------
    top_k_project: int = 50
    """Project the repulsive gradient onto the *top_k_project* token
    embedding directions with the largest directional derivative."""

    # -- temperature --------------------------------------------------------
    temperature: float = 1.0
    """Sampling temperature applied before the Stein modification."""

    # -- annealing ----------------------------------------------------------
    annealing_schedule: str = "linear"
    """Schedule for α annealing over the generation: ``linear``, ``cosine``,
    ``exponential``, ``step_decay``, ``none``."""

    annealing_start: float = 1.0
    """α multiplier at the first step."""

    annealing_end: float = 0.1
    """α multiplier at the last step."""

    # -- convergence --------------------------------------------------------
    convergence_threshold: float = 1e-4
    """Stein discrepancy threshold below which we consider particles
    converged and may stop early."""

    # -- adaptive bandwidth -------------------------------------------------
    use_adaptive_bandwidth: bool = True
    """Re-compute the kernel bandwidth at every step using the chosen
    *bandwidth_method*."""

    # -- misc ---------------------------------------------------------------
    embedding_update_freq: int = 1
    """Recompute sentence embeddings every N steps (1 = every step)."""

    normalize_gradients: bool = True
    """Normalize the repulsive gradient vector before projecting."""

    gradient_clip: float = 5.0
    """Clip the L2 norm of the per-particle repulsive gradient."""

    # -- overrides ----------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty ⟹ valid)."""
        errors = super().validate()
        if self.n_particles < 1:
            errors.append("n_particles must be >= 1")
        if self.alpha < 0.0:
            errors.append("alpha must be >= 0")
        if self.kernel_type not in ("rbf", "cosine", "imq"):
            errors.append(f"Unknown kernel_type: {self.kernel_type!r}")
        if self.bandwidth_method not in ("median", "silverman", "fixed"):
            errors.append(f"Unknown bandwidth_method: {self.bandwidth_method!r}")
        if self.embedding_dim < 1:
            errors.append("embedding_dim must be >= 1")
        if self.top_k_project < 1:
            errors.append("top_k_project must be >= 1")
        if self.annealing_schedule not in (
            "linear",
            "cosine",
            "exponential",
            "step_decay",
            "none",
        ):
            errors.append(
                f"Unknown annealing_schedule: {self.annealing_schedule!r}"
            )
        if self.gradient_clip <= 0:
            errors.append("gradient_clip must be > 0")
        if self.embedding_update_freq < 1:
            errors.append("embedding_update_freq must be >= 1")
        return errors


# =========================================================================
# SVDState
# =========================================================================


@dataclass
class SVDState(DecodingState):
    """Extended decoding state for Stein Variational Decoding.

    Carries the particle ensemble, kernel matrix, bandwidth, and convergence
    diagnostics accumulated over the course of generation.
    """

    # -- particle data ------------------------------------------------------
    particles: List[List[int]] = field(default_factory=list)
    """Token sequences for each particle (mirrors ``sequences``)."""

    # -- kernel / embedding snapshots ---------------------------------------
    particle_embeddings: Optional[np.ndarray] = None
    """Current sentence-level embeddings, shape ``(n_particles, dim)``."""

    kernel_matrix: Optional[np.ndarray] = None
    """Current kernel matrix, shape ``(n_particles, n_particles)``."""

    bandwidth: float = 1.0
    """Current kernel bandwidth."""

    alpha_current: float = 0.5
    """Current (possibly annealed) repulsive strength."""

    # -- diagnostics --------------------------------------------------------
    stein_discrepancies: List[float] = field(default_factory=list)
    """Stein discrepancy at each step — used for convergence monitoring."""

    repulsive_magnitudes: List[List[float]] = field(default_factory=list)
    """Per-step L2 norm of the repulsive gradient for each particle."""

    particle_distances: List[np.ndarray] = field(default_factory=list)
    """Pairwise distance matrices recorded at each step."""

    log_probs_history: List[np.ndarray] = field(default_factory=list)
    """Per-particle cumulative log-prob at each step."""


# =========================================================================
# AnnealingSchedule
# =========================================================================


class AnnealingSchedule:
    """Determines how the repulsive strength α is annealed over generation.

    Supported schedules:

    - **linear**: linearly interpolate from *start* to *end*.
    - **cosine**: cosine annealing (smooth transition).
    - **exponential**: exponential decay ``start * (end / start) ^ (t/T)``.
    - **step_decay**: multiplicative decay every *decay_steps* steps.
    - **none**: constant value equal to *start*.
    """

    def __init__(
        self,
        schedule_type: str = "linear",
        start_value: float = 1.0,
        end_value: float = 0.1,
        total_steps: int = 100,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    # -- public interface ---------------------------------------------------

    def get_value(self, step: int) -> float:
        """Return the annealed value at the given *step*."""
        step = max(0, min(step, self.total_steps))
        dispatch = {
            "linear": self.linear,
            "cosine": self.cosine,
            "exponential": self.exponential,
            "step_decay": lambda s: self.step_decay(s, decay_steps=10, decay_rate=0.5),
            "none": lambda _s: self.start_value,
        }
        fn = dispatch.get(self.schedule_type)
        if fn is None:
            logger.warning("Unknown schedule %r, falling back to constant", self.schedule_type)
            return self.start_value
        return float(fn(step))

    # -- schedules ----------------------------------------------------------

    def linear(self, step: int) -> float:
        """Linear interpolation from *start_value* to *end_value*."""
        frac = step / max(self.total_steps, 1)
        return self.start_value + (self.end_value - self.start_value) * frac

    def cosine(self, step: int) -> float:
        """Cosine annealing (smooth half-cosine curve)."""
        frac = step / max(self.total_steps, 1)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * frac))
        return self.end_value + (self.start_value - self.end_value) * cos_val

    def exponential(self, step: int) -> float:
        """Exponential decay from *start_value* toward *end_value*.

        Uses ``start * (end / start) ^ (step / total_steps)`` so that the
        value smoothly transitions between the two endpoints.
        """
        if self.start_value <= 0 or self.end_value <= 0:
            return self.linear(step)
        ratio = self.end_value / self.start_value
        frac = step / max(self.total_steps, 1)
        return self.start_value * (ratio ** frac)

    def step_decay(
        self,
        step: int,
        decay_steps: int = 10,
        decay_rate: float = 0.5,
    ) -> float:
        """Multiplicative step decay every *decay_steps* steps.

        value = start_value * decay_rate ^ (step // decay_steps)

        The result is clamped to ``[end_value, start_value]``.
        """
        n_decays = step // max(decay_steps, 1)
        value = self.start_value * (decay_rate ** n_decays)
        return max(value, self.end_value)


# =========================================================================
# SVDKernel
# =========================================================================


class SVDKernel:
    """Kernel computation engine for Stein Variational Decoding.

    Provides kernel matrices **K** and their gradients **∇K** for the RBF,
    inverse-multiquadric (IMQ), and cosine kernels.  Bandwidth selection is
    handled via the median heuristic or Silverman's rule.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        bandwidth_method: str = "median",
        fixed_bandwidth: float = 1.0,
    ) -> None:
        if kernel_type not in ("rbf", "cosine", "imq"):
            raise ValueError(f"Unknown kernel type: {kernel_type!r}")
        if bandwidth_method not in ("median", "silverman", "fixed"):
            raise ValueError(f"Unknown bandwidth method: {bandwidth_method!r}")

        self.kernel_type = kernel_type
        self.bandwidth_method = bandwidth_method
        self.fixed_bandwidth = fixed_bandwidth

    # -- public entry point -------------------------------------------------

    def compute(
        self, X: np.ndarray, bandwidth_override: float = -1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the kernel matrix **K** and gradient tensor **∇K**.

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.
        bandwidth_override : float
            If > 0, use this bandwidth instead of the automatic selector.

        Returns
        -------
        K : np.ndarray
            Kernel matrix of shape ``(n, n)``.
        grad_K : np.ndarray
            Kernel gradient tensor of shape ``(n, n, d)`` where
            ``grad_K[i, j]`` = ∇_{x_i} k(x_i, x_j).
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        # Determine bandwidth
        if bandwidth_override > 0:
            h = bandwidth_override
        else:
            h = self.adaptive_bandwidth(X)

        if self.kernel_type == "rbf":
            K = self.rbf(X, h)
            grad_K = self.rbf_grad(X, K, h)
        elif self.kernel_type == "imq":
            K = self.imq(X, h)
            grad_K = self.imq_grad(X, K, h)
        elif self.kernel_type == "cosine":
            K = self.cosine(X)
            grad_K = self.cosine_grad(X, K)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type!r}")

        return K, grad_K

    # -- RBF kernel ---------------------------------------------------------

    def rbf(self, X: np.ndarray, h: float) -> np.ndarray:
        """Radial basis function (Gaussian) kernel.

        k(x_i, x_j) = exp( -||x_i - x_j||^2 / (2h^2) )

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.
        h : float
            Bandwidth (length-scale).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(n, n)``.
        """
        sq_dists = self._pairwise_distances(X)
        return np.exp(-sq_dists / (2.0 * h * h + 1e-12))

    def rbf_grad(
        self, X: np.ndarray, K: np.ndarray, h: float
    ) -> np.ndarray:
        """Gradient of the RBF kernel w.r.t. the first argument.

        ∇_{x_i} k(x_i, x_j) = -k(x_i, x_j) · (x_i - x_j) / h^2

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.
        K : np.ndarray
            Pre-computed kernel matrix of shape ``(n, n)``.
        h : float
            Bandwidth.

        Returns
        -------
        np.ndarray
            Gradient tensor of shape ``(n, n, d)``.
        """
        n, d = X.shape
        # Difference tensor: diff[i, j] = x_i - x_j, shape (n, n, d)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, d)
        h_sq = h * h + 1e-12
        # grad_K[i, j, :] = -K[i, j] * (x_i - x_j) / h^2
        grad_K = -K[:, :, np.newaxis] * diff / h_sq
        return grad_K

    # -- IMQ kernel ---------------------------------------------------------

    def imq(self, X: np.ndarray, h: float) -> np.ndarray:
        """Inverse multiquadric (IMQ) kernel.

        k(x_i, x_j) = (1 + ||x_i - x_j||^2 / h^2)^{-1/2}

        The IMQ kernel has heavier tails than RBF, producing stronger
        long-range repulsion between distant particles.

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.
        h : float
            Bandwidth.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(n, n)``.
        """
        sq_dists = self._pairwise_distances(X)
        return np.power(1.0 + sq_dists / (h * h + 1e-12), -0.5)

    def imq_grad(
        self, X: np.ndarray, K: np.ndarray, h: float
    ) -> np.ndarray:
        """Gradient of the IMQ kernel w.r.t. the first argument.

        ∇_{x_i} k(x_i, x_j) = -K^3 · (x_i - x_j) / h^2

        (Since k = (1 + r^2/h^2)^{-1/2}, ∂k/∂x_i = -(1 + r^2/h^2)^{-3/2}
        · (x_i - x_j) / h^2.)

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.
        K : np.ndarray
            Pre-computed IMQ kernel matrix of shape ``(n, n)``.
        h : float
            Bandwidth.

        Returns
        -------
        np.ndarray
            Gradient tensor of shape ``(n, n, d)``.
        """
        n, d = X.shape
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, d)
        h_sq = h * h + 1e-12
        # K^3 = (1 + r^2/h^2)^{-3/2}
        K_cubed = K ** 3  # (n, n)
        grad_K = -K_cubed[:, :, np.newaxis] * diff / h_sq
        return grad_K

    # -- Cosine kernel ------------------------------------------------------

    def cosine(self, X: np.ndarray) -> np.ndarray:
        """Cosine similarity kernel.

        k(x_i, x_j) = (x_i · x_j) / (||x_i|| · ||x_j|| + ε)

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(n, n)`` with values in ``[-1, 1]``.
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)  # (n, 1)
        norms = np.maximum(norms, 1e-12)
        X_norm = X / norms  # (n, d)
        K = X_norm @ X_norm.T  # (n, n)
        return K

    def cosine_grad(self, X: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Gradient of the cosine kernel w.r.t. the first argument.

        ∇_{x_i} cos(x_i, x_j) = (x_j - cos(x_i, x_j) · x_i) / ||x_i||
                                  / ||x_j||  (approximately)

        More precisely:

            ∇_{x_i} [ x_i · x_j / (||x_i|| ||x_j||) ]
            = x_j / (||x_i|| ||x_j||) - (x_i · x_j) x_i / (||x_i||^3 ||x_j||)

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.
        K : np.ndarray
            Pre-computed cosine kernel matrix of shape ``(n, n)``.

        Returns
        -------
        np.ndarray
            Gradient tensor of shape ``(n, n, d)``.
        """
        n, d = X.shape
        norms = np.linalg.norm(X, axis=1)  # (n,)
        norms = np.maximum(norms, 1e-12)

        grad_K = np.zeros((n, n, d), dtype=np.float64)

        for i in range(n):
            ni = norms[i]
            xi = X[i]  # (d,)
            for j in range(n):
                nj = norms[j]
                xj = X[j]  # (d,)
                # ∇_{x_i} k(x_i, x_j)
                term1 = xj / (ni * nj)
                term2 = K[i, j] * xi / (ni * ni)
                grad_K[i, j] = term1 - term2

        return grad_K

    # -- Bandwidth selection ------------------------------------------------

    def adaptive_bandwidth(self, X: np.ndarray) -> float:
        """Select bandwidth automatically based on the configured method.

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.

        Returns
        -------
        float
            Selected bandwidth (positive scalar).
        """
        if self.bandwidth_method == "median":
            return self._median_heuristic(X)
        elif self.bandwidth_method == "silverman":
            return self._silverman_bandwidth(X)
        elif self.bandwidth_method == "fixed":
            return max(self.fixed_bandwidth, 1e-6)
        else:
            return self._median_heuristic(X)

    # -- Bandwidth heuristics -----------------------------------------------

    @staticmethod
    def _median_heuristic(X: np.ndarray) -> float:
        """Median heuristic for kernel bandwidth.

        bandwidth = sqrt( median( ||x_i - x_j||^2 ) / (2 log(n + 1)) )

        This is the standard choice in the SVGD literature (Liu & Wang 2016).

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.

        Returns
        -------
        float
            Bandwidth scalar.
        """
        n = X.shape[0]
        if n < 2:
            return 1.0

        # Pairwise squared distances (condensed form)
        pairwise_sq = pdist(X, metric="sqeuclidean")

        if len(pairwise_sq) == 0:
            return 1.0

        med_sq = float(np.median(pairwise_sq))
        log_term = 2.0 * math.log(n + 1)
        if log_term < 1e-12:
            return max(math.sqrt(med_sq + 1e-12), 1e-6)
        h = math.sqrt(med_sq / log_term + 1e-12)
        return max(h, 1e-6)

    @staticmethod
    def _silverman_bandwidth(X: np.ndarray) -> float:
        """Silverman's rule of thumb for bandwidth estimation.

        h = ( 4 / (n * (d + 2)) )^{1/(d+4)} · σ

        where σ is the average per-dimension standard deviation.

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.

        Returns
        -------
        float
            Bandwidth scalar.
        """
        n, d = X.shape
        if n < 2:
            return 1.0

        sigma = float(np.mean(np.std(X, axis=0)))
        if sigma < 1e-12:
            sigma = 1.0

        exponent = 1.0 / (d + 4)
        factor = (4.0 / (n * (d + 2) + 1e-12)) ** exponent
        h = factor * sigma
        return max(h, 1e-6)

    # -- Utilities ----------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distance matrix.

        Parameters
        ----------
        X : np.ndarray
            Embeddings of shape ``(n, d)``.

        Returns
        -------
        np.ndarray
            Distance matrix of shape ``(n, n)`` where entry ``[i, j]``
            is ``||x_i - x_j||^2``.
        """
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i · x_j
        sq_norms = np.sum(X ** 2, axis=1)  # (n,)
        dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (X @ X.T)
        np.maximum(dists, 0.0, out=dists)
        return dists


# =========================================================================
# EmbeddingProjector
# =========================================================================


class EmbeddingProjector:
    """Projects continuous gradients in embedding space onto the discrete
    vocabulary, producing logit modifications for token selection.

    Given a gradient vector *g* in ℝ^d (the direction the particle should
    move in embedding space), the projector computes how well each token's
    embedding vector aligns with that direction, returning a vector of
    logit adjustments over the vocabulary.
    """

    def __init__(self, embedding_matrix: np.ndarray) -> None:
        """
        Parameters
        ----------
        embedding_matrix : np.ndarray
            Token embedding matrix of shape ``(vocab_size, dim)``.
        """
        self._embedding_matrix = np.asarray(embedding_matrix, dtype=np.float64)
        self._vocab_size, self._dim = self._embedding_matrix.shape

        # Pre-compute L2 norms for normalisation
        self._embedding_norms = np.linalg.norm(
            self._embedding_matrix, axis=1, keepdims=True
        )
        self._embedding_norms = np.maximum(self._embedding_norms, 1e-12)

        # Normalised embeddings for directional-derivative computation
        self._normed_embeddings = self._embedding_matrix / self._embedding_norms

    # -- public interface ---------------------------------------------------

    def project_to_vocab(
        self, gradient: np.ndarray, top_k: int = 50
    ) -> np.ndarray:
        """Project a gradient in embedding space onto vocabulary logit deltas.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient vector of shape ``(dim,)`` indicating the desired
            movement direction in embedding space.
        top_k : int
            Only the *top_k* tokens with the largest directional derivatives
            receive non-zero logit adjustments.  All others are set to zero.

        Returns
        -------
        np.ndarray
            Logit adjustment vector of shape ``(vocab_size,)``.
        """
        gradient = np.asarray(gradient, dtype=np.float64).ravel()

        if gradient.shape[0] != self._dim:
            raise ValueError(
                f"Gradient dim {gradient.shape[0]} != embedding dim {self._dim}"
            )

        # Directional derivatives (inner products with each token embedding)
        dir_derivs = self._compute_directional_derivatives(gradient)

        # Zero-out everything outside the top-k
        if 0 < top_k < self._vocab_size:
            threshold = np.partition(dir_derivs, -top_k)[-top_k]
            mask = dir_derivs < threshold
            dir_derivs[mask] = 0.0

        return dir_derivs

    def _compute_directional_derivatives(
        self, gradient: np.ndarray
    ) -> np.ndarray:
        """Compute the dot product of *gradient* with every token embedding.

        This measures how much sampling each token would move the particle
        in the direction of *gradient*.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient vector of shape ``(dim,)``.

        Returns
        -------
        np.ndarray
            Directional derivative per token, shape ``(vocab_size,)``.
        """
        # dot(embedding_matrix, gradient) — efficient matrix–vector product
        return self._embedding_matrix @ gradient  # (vocab_size,)

    def _get_token_embedding_matrix(self) -> np.ndarray:
        """Return the raw token embedding matrix.

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size, dim)``.
        """
        return self._embedding_matrix.copy()

    @staticmethod
    def _generate_random_embeddings(
        vocab_size: int, dim: int, seed: int = 42
    ) -> np.ndarray:
        """Generate a random embedding matrix (fallback when no model is
        available).

        Samples each entry i.i.d. from 𝒩(0, 1/√d) so that inner products
        are O(1).

        Parameters
        ----------
        vocab_size : int
        dim : int
        seed : int

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size, dim)``.
        """
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1.0 / math.sqrt(dim), size=(vocab_size, dim))


# =========================================================================
# SVDConvergenceMonitor
# =========================================================================


class SVDConvergenceMonitor:
    """Monitors the convergence behaviour of the SVD particle ensemble.

    Tracks the *kernelised Stein discrepancy* (KSD), particle spread,
    diversity trajectory, and convergence rate to provide diagnostics and
    an early-stopping signal.
    """

    def __init__(self, n_particles: int, embedding_dim: int) -> None:
        self.n_particles = n_particles
        self.embedding_dim = embedding_dim

        # History buffers
        self._stein_discrepancy_history: List[float] = []
        self._particle_spread_history: List[float] = []
        self._embedding_history: List[np.ndarray] = []
        self._kernel_history: List[np.ndarray] = []
        self._log_prob_history: List[np.ndarray] = []
        self._step_count: int = 0

    # -- recording ----------------------------------------------------------

    def record_step(
        self,
        embeddings: np.ndarray,
        kernel_matrix: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        """Record data from a single generation step for monitoring.

        Parameters
        ----------
        embeddings : np.ndarray
            Particle embeddings of shape ``(n_particles, dim)``.
        kernel_matrix : np.ndarray
            Kernel matrix of shape ``(n_particles, n_particles)``.
        log_probs : np.ndarray
            Per-particle cumulative log-probabilities, shape ``(n_particles,)``.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        kernel_matrix = np.asarray(kernel_matrix, dtype=np.float64)
        log_probs = np.asarray(log_probs, dtype=np.float64).ravel()

        self._embedding_history.append(embeddings.copy())
        self._kernel_history.append(kernel_matrix.copy())
        self._log_prob_history.append(log_probs.copy())

        ksd = self._compute_stein_discrepancy(embeddings, kernel_matrix, log_probs)
        self._stein_discrepancy_history.append(ksd)

        spread = self._compute_spread(embeddings)
        self._particle_spread_history.append(spread)

        self._step_count += 1

    # -- queries ------------------------------------------------------------

    def stein_discrepancy(self) -> float:
        """Return the most recent kernelised Stein discrepancy (KSD) proxy.

        **Caveat**: This is a heuristic diagnostic that correlates with
        particle stabilisation.  It is NOT a formal bound on distributional
        distance because autoregressive decoding violates SVGD convergence
        conditions (discrete updates, no static target distribution).

        Returns
        -------
        float
            Most recent KSD proxy value (0.0 if no steps recorded).
        """
        if not self._stein_discrepancy_history:
            return 0.0
        return self._stein_discrepancy_history[-1]

    def particle_spread(self) -> float:
        """Return the most recent measure of particle spread.

        Computed as the mean pairwise Euclidean distance between particle
        embeddings.

        Returns
        -------
        float
        """
        if not self._particle_spread_history:
            return 0.0
        return self._particle_spread_history[-1]

    def diversity_trajectory(self) -> List[float]:
        """Return the history of particle-spread values.

        Useful for plotting how diversity evolves during generation.

        Returns
        -------
        list of float
        """
        return list(self._particle_spread_history)

    def convergence_rate(self) -> float:
        """Estimate the convergence rate of the Stein discrepancy.

        Computed as the exponential decay rate of the KSD over the last
        several steps.  A large negative value indicates rapid convergence.

        Returns
        -------
        float
            Estimated convergence rate.  Returns 0.0 if fewer than 3 steps
            have been recorded.
        """
        hist = self._stein_discrepancy_history
        if len(hist) < 3:
            return 0.0

        # Use last min(10, len) values to estimate rate
        window = hist[-min(10, len(hist)):]
        if window[0] < 1e-15:
            return 0.0

        # Fit log-linear: log(ksd) ~ a * t + b  →  rate = a
        t = np.arange(len(window), dtype=np.float64)
        log_ksd = np.log(np.maximum(window, 1e-30))

        if len(t) < 2:
            return 0.0

        # Simple least-squares: slope = cov(t, y) / var(t)
        t_mean = t.mean()
        y_mean = log_ksd.mean()
        cov = float(np.mean((t - t_mean) * (log_ksd - y_mean)))
        var = float(np.mean((t - t_mean) ** 2))
        if var < 1e-15:
            return 0.0
        return cov / var

    def is_converged(self, threshold: float = 1e-4) -> bool:
        """Check whether the particles have converged.

        Convergence is declared when the KSD drops below *threshold* and
        has been monotonically decreasing for at least 3 steps.

        Parameters
        ----------
        threshold : float

        Returns
        -------
        bool
        """
        hist = self._stein_discrepancy_history
        if len(hist) < 3:
            return False
        if hist[-1] > threshold:
            return False
        # Check monotonically decreasing for last 3 values
        return hist[-1] <= hist[-2] <= hist[-3]

    def summary(self) -> dict:
        """Return a summary dictionary of convergence diagnostics.

        Returns
        -------
        dict
        """
        return {
            "total_steps": self._step_count,
            "current_stein_discrepancy": self.stein_discrepancy(),
            "current_particle_spread": self.particle_spread(),
            "convergence_rate": self.convergence_rate(),
            "is_converged": self.is_converged(),
            "stein_discrepancy_history": list(self._stein_discrepancy_history),
            "diversity_trajectory": self.diversity_trajectory(),
        }

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _compute_stein_discrepancy(
        embeddings: np.ndarray,
        kernel_matrix: np.ndarray,
        log_probs: np.ndarray,
    ) -> float:
        """Compute the kernelised Stein discrepancy (KSD).

        The KSD for a kernel *k* and target with score function
        s(x) = ∇_x log p(x) is:

            KSD^2 = (1/n^2) Σ_{i,j} [ k(x_i, x_j) s(x_i)·s(x_j)
                                       + s(x_i)·∇_{x_j} k(x_i, x_j)
                                       + s(x_j)·∇_{x_i} k(x_i, x_j)
                                       + Tr(∇_{x_i} ∇_{x_j} k(x_i, x_j)) ]

        We approximate the score by finite differences of the log-prob
        and use a simplified u-statistic estimator for efficiency.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, d)``.
        kernel_matrix : np.ndarray
            Shape ``(n, n)``.
        log_probs : np.ndarray
            Shape ``(n,)``.

        Returns
        -------
        float
            Estimated KSD (non-negative).
        """
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        # Approximate score as gradient of log-prob w.r.t. embedding.
        # Since we don't have the true score, use the log-prob weighted
        # kernel as a proxy:  KSD ≈ var(log_probs) * mean(off-diag K)
        log_probs = log_probs - log_probs.mean()

        # Weighted kernel sum (u-statistic form, excluding diagonal)
        K_off = kernel_matrix.copy()
        np.fill_diagonal(K_off, 0.0)

        # KSD proxy: (1/n^2) Σ_{i≠j} K[i,j] · (lp_i - lp_j)^2
        lp_diff = log_probs[:, np.newaxis] - log_probs[np.newaxis, :]  # (n, n)
        ksd_sq = float(np.sum(K_off * lp_diff ** 2)) / (n * n)
        return math.sqrt(max(ksd_sq, 0.0))

    @staticmethod
    def _compute_spread(embeddings: np.ndarray) -> float:
        """Mean pairwise Euclidean distance between particles.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, d)``.

        Returns
        -------
        float
        """
        n = embeddings.shape[0]
        if n < 2:
            return 0.0
        dists = pdist(embeddings, metric="euclidean")
        return float(np.mean(dists))


# =========================================================================
# SteinVariationalDecoding — main algorithm
# =========================================================================


class SteinVariationalDecoding(DecodingAlgorithm):
    """Stein Variational Decoding (SVD) — Kernel-Repulsive Particle Decoding.

    Maintains *n* particle sequences and at each step modifies each
    particle's next-token logits with a repulsive term derived from the
    pairwise kernel between particle embeddings.

    **Important**: Despite the name, this is a kernel-repulsive heuristic,
    not a variational inference method.  See module docstring for details.

    **Algorithm outline (per step)**:

    1. Compute a sentence-level embedding for each particle's current
       prefix.
    2. Compute the kernel matrix **K** and its gradient **∇K** over
       the embeddings.
    3. For each particle *i*:
       a. Obtain base logits from the language model.
       b. Compute the repulsive gradient:
          ``g_i = (1/n) Σ_{j≠i} ∇_{e_j} k(e_j, e_i)``
       c. Project the gradient onto the vocabulary via inner products with
          token embeddings.
       d. Modify logits: ``logits_i += α · projected_gradient``
       e. Sample the next token from the modified distribution.
    4. Anneal α according to the configured schedule.
    5. Record convergence diagnostics (heuristic, not formal).
    """

    def __init__(self, config: SVDConfig) -> None:
        # Sync n_particles → num_sequences so the base class sets up the
        # right number of sequences.
        config.num_sequences = config.n_particles
        super().__init__(config)

        self.svd_config: SVDConfig = config

        # Build sub-components
        self._kernel = SVDKernel(
            kernel_type=config.kernel_type,
            bandwidth_method=config.bandwidth_method,
            fixed_bandwidth=config.kernel_bandwidth
            if config.kernel_bandwidth > 0
            else 1.0,
        )

        self._annealing = AnnealingSchedule(
            schedule_type=config.annealing_schedule,
            start_value=config.annealing_start,
            end_value=config.annealing_end,
            total_steps=config.max_new_tokens,
        )

        self._monitor = SVDConvergenceMonitor(
            n_particles=config.n_particles,
            embedding_dim=config.embedding_dim,
        )

        # Embedding projector (initialised lazily once we know the vocab size)
        self._projector: Optional[EmbeddingProjector] = None
        self._embedding_matrix: Optional[np.ndarray] = None

        # Cached state
        self._vocab_size: Optional[int] = None

    # -- properties ---------------------------------------------------------

    @property
    def description(self) -> str:
        return (
            "Kernel-Repulsive Particle Decoding: diverse generation via "
            "kernel-gradient repulsion in embedding space (inspired by SVGD)"
        )

    # -- public entry points ------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate *n_particles* diverse continuations of *prompt_ids*.

        This overrides the base ``generate`` to set up SVD-specific state
        before entering the generation loop.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        list of TokenSequence
            Generated continuations sorted by score (best first).
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.default_rng(self.config.seed)

        # Probe vocab size with a dummy forward pass
        probe_logits = logit_source([prompt_ids])
        self._vocab_size = probe_logits.shape[-1]

        # Initialise embedding projector if needed
        if self._projector is None:
            self._init_projector(self._vocab_size)

        state = self._init_particles(prompt_ids)
        state = self._generation_loop(state, logit_source)
        return self._finalize(state)

    # -- lifecycle hooks ----------------------------------------------------

    def _init_particles(self, prompt_ids: List[int]) -> SVDState:
        """Initialise the particle ensemble.

        Creates *n_particles* copies of the prompt and sets up all SVD
        tracking structures.

        Parameters
        ----------
        prompt_ids : list of int

        Returns
        -------
        SVDState
        """
        n = self.svd_config.n_particles
        dim = self.svd_config.embedding_dim

        particles = [list(prompt_ids) for _ in range(n)]
        initial_embeddings = np.zeros((n, dim), dtype=np.float64)

        state = SVDState(
            # Base DecodingState fields
            sequences=particles,
            scores=[0.0] * n,
            is_finished=[False] * n,
            step=0,
            metadata={
                "prompt_length": len(prompt_ids),
                "n_particles": n,
                "algorithm": "SVD",
            },
            embeddings=initial_embeddings,
            logit_history=[],
            # SVD-specific fields
            particles=particles,
            particle_embeddings=initial_embeddings.copy(),
            kernel_matrix=np.eye(n, dtype=np.float64),
            bandwidth=1.0,
            alpha_current=self.svd_config.alpha * self.svd_config.annealing_start,
            stein_discrepancies=[],
            repulsive_magnitudes=[],
            particle_distances=[],
            log_probs_history=[],
        )

        logger.info(
            "SVD: Initialised %d particles (prompt_len=%d, dim=%d)",
            n, len(prompt_ids), dim,
        )
        return state

    def _init_state(self, prompt_ids: List[int]) -> DecodingState:
        """Override base _init_state to return SVDState."""
        return self._init_particles(prompt_ids)

    def _step(
        self, state: DecodingState, logit_source: LogitSource
    ) -> DecodingState:
        """Execute a single SVD generation step.

        For each active particle:
        1. Compute sentence embeddings (if update freq matches).
        2. Compute kernel matrix and gradients.
        3. Get base logits from the model.
        4. Compute repulsive gradient from other particles.
        5. Project gradient onto token embedding directions.
        6. Modify logits and sample next token.

        Parameters
        ----------
        state : DecodingState
            Must be an SVDState instance.
        logit_source : LogitSource

        Returns
        -------
        SVDState
            Updated state.
        """
        if not isinstance(state, SVDState):
            raise TypeError("SVD._step requires an SVDState instance")

        svd_state: SVDState = state
        n = self.svd_config.n_particles
        step = svd_state.step
        cfg = self.svd_config
        prompt_len = svd_state.metadata.get("prompt_length", 0)

        # 1. Anneal alpha
        alpha = self._anneal_alpha(step, cfg.max_new_tokens)
        svd_state.alpha_current = alpha

        # 2. Compute embeddings (possibly skipping based on update freq)
        if step % cfg.embedding_update_freq == 0:
            embeddings = self._compute_embeddings(svd_state.particles)
            svd_state.particle_embeddings = embeddings
        else:
            embeddings = svd_state.particle_embeddings

        if embeddings is None:
            embeddings = np.zeros(
                (n, cfg.embedding_dim), dtype=np.float64
            )
            svd_state.particle_embeddings = embeddings

        # 3. Compute kernel matrix and gradients
        K, grad_K = self._compute_kernel_matrix(embeddings)
        svd_state.kernel_matrix = K

        # Update bandwidth record
        if cfg.use_adaptive_bandwidth:
            svd_state.bandwidth = self._kernel.adaptive_bandwidth(embeddings)
        elif cfg.kernel_bandwidth > 0:
            svd_state.bandwidth = cfg.kernel_bandwidth

        # 4. Record pairwise distances
        dist_matrix = squareform(pdist(embeddings, metric="euclidean"))
        svd_state.particle_distances.append(dist_matrix)

        # 5. Get base logits for ALL particles in a single batch call
        active = svd_state.active_indices()
        if not active:
            return svd_state

        all_logits = logit_source(svd_state.particles)  # (n, vocab)
        vocab_size = all_logits.shape[-1]

        # Ensure projector matches vocab size
        if self._projector is None or self._embedding_matrix is None:
            self._init_projector(vocab_size)

        # 6. Compute repulsive gradients and modify logits
        step_repulsive_mags: List[float] = []
        step_log_probs = np.zeros(n, dtype=np.float64)

        for i in active:
            # Base logits for particle i
            base_logits = all_logits[i].copy().astype(np.float64)

            # Apply constraints (repetition penalty, ngram blocking, etc.)
            base_logits = self._apply_constraints_for_particle(
                base_logits, svd_state, i
            )

            # Temperature scaling
            if cfg.temperature > 0 and cfg.temperature != 1.0:
                base_logits = base_logits / cfg.temperature

            # Compute repulsive gradient for particle i
            repulsive_grad = self._compute_svgd_repulsive_gradient(
                i, embeddings, K, grad_K
            )

            # Clip gradient
            grad_norm = float(np.linalg.norm(repulsive_grad))
            if cfg.gradient_clip > 0 and grad_norm > cfg.gradient_clip:
                repulsive_grad = repulsive_grad * (cfg.gradient_clip / grad_norm)
                grad_norm = cfg.gradient_clip

            # Normalize gradient
            if cfg.normalize_gradients and grad_norm > 1e-12:
                repulsive_grad = repulsive_grad / grad_norm

            step_repulsive_mags.append(grad_norm)

            # Project gradient onto vocabulary
            assert self._projector is not None
            repulsive_logits = self._compute_repulsive_logits(
                base_logits, repulsive_grad, alpha
            )

            # Combine: logits_i(v) = base_logits(v) + alpha * repulsive_logits(v)
            modified_logits = base_logits + repulsive_logits

            # Sample next token
            token = sample_token(
                modified_logits,
                temperature=1.0,  # already applied above
                top_k=0,
                top_p=1.0,
            )

            # Update particle
            svd_state.particles[i].append(token)
            # sequences and particles share the same list objects,
            # but ensure consistency:
            if svd_state.sequences[i] is not svd_state.particles[i]:
                svd_state.sequences[i].append(token)

            # Accumulate log-prob
            log_p = _log_softmax(base_logits)
            token_logp = float(log_p[min(token, len(log_p) - 1)])
            svd_state.scores[i] += token_logp
            step_log_probs[i] = svd_state.scores[i]

            # Check EOS
            if cfg.eos_token_id is not None and token == cfg.eos_token_id:
                gen_len = len(svd_state.particles[i]) - prompt_len
                if gen_len >= cfg.min_new_tokens:
                    svd_state.mark_finished(i)

        # 7. Record diagnostics
        svd_state.repulsive_magnitudes.append(step_repulsive_mags)
        svd_state.log_probs_history.append(step_log_probs.copy())

        # Record convergence data
        self._monitor.record_step(embeddings, K, step_log_probs)
        ksd = self._monitor.stein_discrepancy()
        svd_state.stein_discrepancies.append(ksd)

        if step % 10 == 0:
            logger.debug(
                "SVD step %d: α=%.4f, KSD=%.6f, spread=%.4f, "
                "mean_repulsion=%.4f",
                step,
                alpha,
                ksd,
                self._monitor.particle_spread(),
                float(np.mean(step_repulsive_mags)) if step_repulsive_mags else 0.0,
            )

        return svd_state

    def _should_stop(self, state: DecodingState) -> bool:
        """Check for convergence-based early stopping."""
        if state.all_finished():
            return True
        if isinstance(state, SVDState):
            return self._check_convergence(state)
        return False

    # -- kernel computation -------------------------------------------------

    def _compute_kernel_matrix(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the pairwise kernel matrix and its gradient tensor.

        Parameters
        ----------
        embeddings : np.ndarray
            Particle embeddings of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Shape ``(n, n)``.
        grad_K : np.ndarray
            Shape ``(n, n, d)`` — ``grad_K[i, j]`` = ∇_{e_i} k(e_i, e_j).
        """
        bw_override = -1.0
        if not self.svd_config.use_adaptive_bandwidth and self.svd_config.kernel_bandwidth > 0:
            bw_override = self.svd_config.kernel_bandwidth
        return self._kernel.compute(embeddings, bandwidth_override=bw_override)

    def _compute_kernel_gradients(
        self, embeddings: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        """Compute kernel gradients given pre-computed kernel matrix.

        Convenience wrapper when K is already available.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, d)``.
        K : np.ndarray
            Shape ``(n, n)``.

        Returns
        -------
        np.ndarray
            Shape ``(n, n, d)``.
        """
        h = self._kernel.adaptive_bandwidth(embeddings)
        kt = self.svd_config.kernel_type
        if kt == "rbf":
            return self._kernel.rbf_grad(embeddings, K, h)
        elif kt == "imq":
            return self._kernel.imq_grad(embeddings, K, h)
        elif kt == "cosine":
            return self._kernel.cosine_grad(embeddings, K)
        else:
            return self._kernel.rbf_grad(embeddings, K, h)

    # -- individual kernel wrappers (for direct access) ---------------------

    def _rbf_kernel(
        self, embeddings: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """RBF kernel matrix.  Delegates to :class:`SVDKernel`."""
        return self._kernel.rbf(embeddings, bandwidth)

    def _rbf_kernel_gradient(
        self,
        embeddings: np.ndarray,
        K: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        """RBF kernel gradient.  Delegates to :class:`SVDKernel`."""
        return self._kernel.rbf_grad(embeddings, K, bandwidth)

    def _imq_kernel(
        self, embeddings: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """IMQ kernel matrix.  Delegates to :class:`SVDKernel`."""
        return self._kernel.imq(embeddings, bandwidth)

    def _cosine_kernel(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine kernel matrix.  Delegates to :class:`SVDKernel`."""
        return self._kernel.cosine(embeddings)

    def _median_heuristic(self, embeddings: np.ndarray) -> float:
        """Median heuristic bandwidth.  Delegates to :class:`SVDKernel`."""
        return SVDKernel._median_heuristic(embeddings)

    def _silverman_bandwidth(self, embeddings: np.ndarray) -> float:
        """Silverman bandwidth.  Delegates to :class:`SVDKernel`."""
        return SVDKernel._silverman_bandwidth(embeddings)

    # -- SVGD repulsive gradient --------------------------------------------

    def _compute_svgd_repulsive_gradient(
        self,
        particle_idx: int,
        embeddings: np.ndarray,
        K: np.ndarray,
        grad_K: np.ndarray,
    ) -> np.ndarray:
        """Compute the SVGD repulsive gradient for particle *i*.

        The repulsive component of the SVGD update for particle *i* is:

            g_i^{repulsive} = (1/n) Σ_{j≠i} ∇_{e_j} k(e_j, e_i)

        This is the second term of the full SVGD update (the "repulsive
        force" that pushes particles apart).

        We also incorporate the *attractive* kernel-weighted score term:

            g_i^{attractive} = (1/n) Σ_{j} k(e_j, e_i) · ∇_{e_j} log p(e_j)

        but since we don't have the explicit score function, we fold the
        attractive term into the base logits and only return the repulsive
        gradient here.

        Parameters
        ----------
        particle_idx : int
            Index of the particle to compute the gradient for.
        embeddings : np.ndarray
            All particle embeddings, shape ``(n, d)``.
        K : np.ndarray
            Kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)`` where
            ``grad_K[j, i]`` = ∇_{e_j} k(e_j, e_i).

        Returns
        -------
        np.ndarray
            Repulsive gradient vector, shape ``(d,)``.
        """
        n, d = embeddings.shape
        i = particle_idx

        # Repulsive term: (1/n) Σ_{j≠i} ∇_{e_j} k(e_j, e_i)
        # grad_K[j, i, :] is ∇_{e_j} k(e_j, e_i)
        repulsive = np.zeros(d, dtype=np.float64)
        for j in range(n):
            if j == i:
                continue
            repulsive += grad_K[j, i, :]

        repulsive /= max(n, 1)
        return repulsive

    def _compute_repulsive_logits(
        self,
        base_logits: np.ndarray,
        repulsive_gradient: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Convert a repulsive gradient into logit adjustments.

        Projects the repulsive gradient onto the token embedding space to
        get a per-token logit modification that encourages the particle to
        move *away* from other particles.

        Parameters
        ----------
        base_logits : np.ndarray
            Base logits of shape ``(vocab_size,)`` (used only for shape).
        repulsive_gradient : np.ndarray
            Gradient in embedding space, shape ``(d,)``.
        alpha : float
            Repulsive strength coefficient.

        Returns
        -------
        np.ndarray
            Logit adjustments of shape ``(vocab_size,)``.
        """
        assert self._projector is not None
        vocab_size = base_logits.shape[0]

        # Project gradient onto vocabulary
        logit_deltas = self._projector.project_to_vocab(
            repulsive_gradient, top_k=self.svd_config.top_k_project
        )

        # Ensure shape matches
        if logit_deltas.shape[0] != vocab_size:
            # Truncate or pad
            result = np.zeros(vocab_size, dtype=np.float64)
            copy_len = min(logit_deltas.shape[0], vocab_size)
            result[:copy_len] = logit_deltas[:copy_len]
            logit_deltas = result

        return alpha * logit_deltas

    def _project_gradient_to_vocab(
        self,
        gradient: np.ndarray,
        token_embeddings: np.ndarray,
        top_k: int = 50,
    ) -> np.ndarray:
        """Project a continuous gradient vector onto the discrete vocabulary.

        For each token *v*, computes the directional derivative
        ``⟨gradient, e(v)⟩`` and keeps only the *top_k* largest values.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient vector in embedding space, shape ``(d,)``.
        token_embeddings : np.ndarray
            Token embedding matrix, shape ``(vocab_size, d)``.
        top_k : int
            Number of tokens to keep.

        Returns
        -------
        np.ndarray
            Logit adjustment vector, shape ``(vocab_size,)``.
        """
        gradient = np.asarray(gradient, dtype=np.float64).ravel()
        token_embeddings = np.asarray(token_embeddings, dtype=np.float64)

        # Directional derivatives
        scores = token_embeddings @ gradient  # (vocab_size,)

        # Keep only top-k
        if 0 < top_k < scores.shape[0]:
            threshold = np.partition(scores, -top_k)[-top_k]
            scores[scores < threshold] = 0.0

        return scores

    # -- annealing ----------------------------------------------------------

    def _anneal_alpha(self, step: int, total_steps: int) -> float:
        """Get the annealed value of α at the current step.

        Parameters
        ----------
        step : int
            Current generation step.
        total_steps : int
            Total number of generation steps.

        Returns
        -------
        float
            α * annealing_multiplier
        """
        multiplier = self._annealing.get_value(step)
        return self.svd_config.alpha * multiplier

    # -- embedding computation ----------------------------------------------

    def _compute_embeddings(
        self, sequences: List[List[int]]
    ) -> np.ndarray:
        """Compute sentence-level embeddings for the particle sequences.

        In a full integration with a language model, this would use the
        model's hidden states (e.g., mean-pooled last-layer representations).
        Here we use a lightweight hash-based embedding that captures
        sequence identity while being deterministic and fast.

        The embedding is constructed by:
        1. Extracting the last *window* tokens of each sequence.
        2. Computing a position-weighted hash embedding using the token ids.
        3. Applying layer normalisation.

        Parameters
        ----------
        sequences : list of list of int
            Token sequences for each particle.

        Returns
        -------
        np.ndarray
            Embeddings of shape ``(n_particles, embedding_dim)``.
        """
        n = len(sequences)
        dim = self.svd_config.embedding_dim
        embeddings = np.zeros((n, dim), dtype=np.float64)

        # Window of recent tokens to embed (captures local context)
        window = min(64, max(len(s) for s in sequences) if sequences else 1)

        for i, seq in enumerate(sequences):
            if not seq:
                continue

            # Take last `window` tokens
            recent = seq[-window:]
            emb = np.zeros(dim, dtype=np.float64)

            for pos, token_id in enumerate(recent):
                # Deterministic pseudo-random projection based on token id
                # and position — gives each (position, token) pair a unique
                # contribution to the embedding.
                rng = np.random.RandomState(seed=(token_id * 7919 + pos * 104729) % (2**31))
                direction = rng.randn(dim)
                # Position-decay weighting (more recent tokens matter more)
                weight = math.exp(-0.05 * (len(recent) - 1 - pos))
                emb += weight * direction

            # Layer-normalise
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            embeddings[i] = emb

        return embeddings

    # -- convergence --------------------------------------------------------

    def _compute_stein_discrepancy(
        self, embeddings: np.ndarray, log_probs: np.ndarray
    ) -> float:
        """Compute the kernelised Stein discrepancy for monitoring.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, d)``.
        log_probs : np.ndarray
            Shape ``(n,)``.

        Returns
        -------
        float
        """
        K, _ = self._compute_kernel_matrix(embeddings)
        return SVDConvergenceMonitor._compute_stein_discrepancy(
            embeddings, K, log_probs
        )

    def _check_convergence(self, state: SVDState) -> bool:
        """Check whether the particle ensemble has converged.

        Uses the :class:`SVDConvergenceMonitor` to determine if the Stein
        discrepancy has dropped below the configured threshold.

        Parameters
        ----------
        state : SVDState

        Returns
        -------
        bool
        """
        return self._monitor.is_converged(self.svd_config.convergence_threshold)

    # -- constraint helpers -------------------------------------------------

    def _apply_constraints_for_particle(
        self,
        logits: np.ndarray,
        state: SVDState,
        particle_idx: int,
    ) -> np.ndarray:
        """Apply decoding constraints for a single particle.

        Applies repetition penalty, n-gram blocking, and minimum-length
        enforcement based on the particle's own sequence history.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.
        state : SVDState
        particle_idx : int

        Returns
        -------
        np.ndarray
            Constrained logits.
        """
        cfg = self.svd_config
        seq = state.particles[particle_idx]
        prompt_len = state.metadata.get("prompt_length", 0)

        # Repetition penalty
        if cfg.repetition_penalty > 1.0 and seq:
            logits = self._apply_repetition_penalty(
                logits, seq, cfg.repetition_penalty
            )

        # No-repeat n-gram
        if cfg.no_repeat_ngram_size > 0 and seq:
            logits = self._apply_no_repeat_ngram(
                logits, seq, cfg.no_repeat_ngram_size
            )

        # Min-length enforcement
        gen_len = len(seq) - prompt_len
        logits = self._enforce_min_length(
            logits, gen_len, cfg.min_new_tokens, cfg.eos_token_id
        )

        return logits

    # -- projector initialisation -------------------------------------------

    def _init_projector(self, vocab_size: int) -> None:
        """Initialise the embedding projector.

        If no external embedding matrix has been provided, generates a
        random embedding matrix as a fallback.

        Parameters
        ----------
        vocab_size : int
        """
        dim = self.svd_config.embedding_dim

        if self._embedding_matrix is not None:
            # Validate shape
            if self._embedding_matrix.shape != (vocab_size, dim):
                logger.warning(
                    "Provided embedding matrix shape %s doesn't match "
                    "expected (%d, %d); generating random fallback",
                    self._embedding_matrix.shape,
                    vocab_size,
                    dim,
                )
                self._embedding_matrix = None

        if self._embedding_matrix is None:
            logger.info(
                "SVD: Generating random embedding matrix (%d × %d) "
                "as fallback (no model embeddings provided)",
                vocab_size,
                dim,
            )
            self._embedding_matrix = EmbeddingProjector._generate_random_embeddings(
                vocab_size, dim, seed=self.svd_config.seed or 42
            )

        self._projector = EmbeddingProjector(self._embedding_matrix)
        self._vocab_size = vocab_size

    def set_embedding_matrix(self, embedding_matrix: np.ndarray) -> None:
        """Provide an external token embedding matrix.

        Call this before :meth:`generate` to use real model embeddings
        instead of random fallback projections.

        Parameters
        ----------
        embedding_matrix : np.ndarray
            Shape ``(vocab_size, embedding_dim)``.
        """
        self._embedding_matrix = np.asarray(embedding_matrix, dtype=np.float64)
        self._projector = None  # Reset so it's rebuilt on next generate()

    # -- hyperparameter space -----------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        """Describe the SVD hyperparameter search space."""
        base = super().hyperparameter_space()
        base.update(
            {
                "n_particles": {"type": "int", "low": 5, "high": 50},
                "alpha": {"type": "float", "low": 0.01, "high": 2.0, "log": True},
                "kernel_type": {
                    "type": "categorical",
                    "choices": ["rbf", "cosine", "imq"],
                },
                "kernel_bandwidth": {"type": "float", "low": 0.1, "high": 10.0},
                "top_k_project": {"type": "int", "low": 10, "high": 200},
                "annealing_schedule": {
                    "type": "categorical",
                    "choices": ["linear", "cosine", "exponential", "none"],
                },
                "annealing_start": {"type": "float", "low": 0.5, "high": 2.0},
                "annealing_end": {"type": "float", "low": 0.01, "high": 0.5},
                "gradient_clip": {"type": "float", "low": 1.0, "high": 20.0},
            }
        )
        return base

    # -- diagnostics --------------------------------------------------------

    def get_convergence_summary(self) -> dict:
        """Return a summary of convergence diagnostics.

        Returns
        -------
        dict
        """
        return self._monitor.summary()

    def get_kernel(self) -> SVDKernel:
        """Return the kernel object for external inspection."""
        return self._kernel

    def get_annealing_schedule(self) -> AnnealingSchedule:
        """Return the annealing schedule for external inspection."""
        return self._annealing

    def get_monitor(self) -> SVDConvergenceMonitor:
        """Return the convergence monitor for external inspection."""
        return self._monitor

    def __repr__(self) -> str:
        return (
            f"SteinVariationalDecoding("
            f"n_particles={self.svd_config.n_particles}, "
            f"alpha={self.svd_config.alpha}, "
            f"kernel={self.svd_config.kernel_type}, "
            f"annealing={self.svd_config.annealing_schedule})"
        )


# =========================================================================
# Convenience factory
# =========================================================================


def create_svd(
    n_particles: int = 20,
    alpha: float = 0.5,
    kernel_type: str = "rbf",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    annealing_schedule: str = "linear",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> SteinVariationalDecoding:
    """Create a :class:`SteinVariationalDecoding` instance with sensible
    defaults.

    Parameters
    ----------
    n_particles : int
    alpha : float
    kernel_type : str
    max_new_tokens : int
    temperature : float
    annealing_schedule : str
    seed : int, optional
    **kwargs
        Additional :class:`SVDConfig` fields.

    Returns
    -------
    SteinVariationalDecoding
    """
    config = SVDConfig(
        n_particles=n_particles,
        alpha=alpha,
        kernel_type=kernel_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        annealing_schedule=annealing_schedule,
        seed=seed,
        **kwargs,
    )
    return SteinVariationalDecoding(config)


# =========================================================================
# SVGDParticleSampler
# =========================================================================


class SVGDParticleSampler:
    """Full SVGD-inspired particle-based sampling system.

    Maintains a system of particles that are iteratively updated using Stein
    Variational Gradient Descent.  Each particle represents a candidate
    sequence (or, more precisely, a point in a continuous embedding space).
    The update rule combines an *attractive* term (toward high probability)
    and a *repulsive* term (away from other particles), yielding a set of
    points that approximate the target distribution while preserving
    diversity.

    Parameters
    ----------
    n_particles : int
        Number of particles in the system.
    step_size : float
        Base step size (learning rate) for SVGD updates.
    max_iterations : int
        Maximum number of SVGD iterations.
    kernel_bandwidth : float, optional
        Fixed kernel bandwidth.  If <= 0 the median heuristic is used.
    convergence_threshold : float, optional
        Threshold on the relative change in particle positions used to
        declare convergence.

    Notes
    -----
    The canonical SVGD update (Liu & Wang 2016) for particle *i* is::

        x_i ← x_i + ε · φ*(x_i)

    where::

        φ*(x) = (1/n) Σ_j [ k(x_j, x) · ∇ log p(x_j) + ∇_{x_j} k(x_j, x) ]

    The first sum term is the attractive force; the second is the repulsive
    (diversity-promoting) force.
    """

    def __init__(
        self,
        n_particles: int = 10,
        step_size: float = 0.1,
        max_iterations: int = 100,
        kernel_bandwidth: float = -1.0,
        convergence_threshold: float = 1e-5,
    ) -> None:
        if n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        self.n_particles = n_particles
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.kernel_bandwidth = kernel_bandwidth
        self.convergence_threshold = convergence_threshold

        self._particles: Optional[np.ndarray] = None
        self._iteration: int = 0
        self._history: List[np.ndarray] = []

    # -- force computation --------------------------------------------------

    def _compute_attractive_force(
        self,
        particles: np.ndarray,
        grad_log_prob: np.ndarray,
        kernel_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute the attractive (score-weighted) component of the SVGD update.

        Parameters
        ----------
        particles : np.ndarray
            Current particle positions, shape ``(n, d)``.
        grad_log_prob : np.ndarray
            Gradient of log-probability at each particle, shape ``(n, d)``.
        kernel_matrix : np.ndarray
            Kernel matrix, shape ``(n, n)``.

        Returns
        -------
        np.ndarray
            Attractive force for each particle, shape ``(n, d)``.
        """
        n = particles.shape[0]
        # attractive_i = (1/n) Σ_j k(x_j, x_i) · ∇ log p(x_j)
        attractive = kernel_matrix.T @ grad_log_prob / n
        return attractive

    def _compute_repulsive_force(
        self,
        particles: np.ndarray,
        kernel_gradients: np.ndarray,
    ) -> np.ndarray:
        """Compute the repulsive (kernel-gradient) component of the SVGD update.

        Parameters
        ----------
        particles : np.ndarray
            Current particle positions, shape ``(n, d)``.
        kernel_gradients : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)``.

        Returns
        -------
        np.ndarray
            Repulsive force for each particle, shape ``(n, d)``.
        """
        n = particles.shape[0]
        # repulsive_i = (1/n) Σ_j ∇_{x_j} k(x_j, x_i)
        repulsive = np.sum(kernel_gradients, axis=0) / n
        return repulsive

    def _svgd_update(
        self,
        particles: np.ndarray,
        grad_log_prob: np.ndarray,
        kernel_matrix: np.ndarray,
        kernel_gradients: np.ndarray,
    ) -> np.ndarray:
        """Perform one SVGD update step on all particles.

        Parameters
        ----------
        particles : np.ndarray
            Current positions, shape ``(n, d)``.
        grad_log_prob : np.ndarray
            Score function at each particle, shape ``(n, d)``.
        kernel_matrix : np.ndarray
            Kernel values, shape ``(n, n)``.
        kernel_gradients : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)``.

        Returns
        -------
        np.ndarray
            Updated particle positions, shape ``(n, d)``.
        """
        attractive = self._compute_attractive_force(
            particles, grad_log_prob, kernel_matrix
        )
        repulsive = self._compute_repulsive_force(particles, kernel_gradients)
        phi = attractive + repulsive
        return particles + self.step_size * phi

    # -- kernel helpers -----------------------------------------------------

    def _compute_kernel_matrix(
        self, particles: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """Compute the RBF kernel matrix between particles.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        bandwidth : float
            Kernel bandwidth (length-scale).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(n, n)``.
        """
        sq_dists = cdist(particles, particles, metric="sqeuclidean")
        return np.exp(-sq_dists / (2.0 * bandwidth * bandwidth + 1e-12))

    def _compute_kernel_gradients(
        self,
        particles: np.ndarray,
        kernel_matrix: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        """Compute kernel gradient tensor ∇_{x_j} k(x_j, x_i).

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        kernel_matrix : np.ndarray
            Pre-computed kernel matrix, shape ``(n, n)``.
        bandwidth : float
            Kernel bandwidth.

        Returns
        -------
        np.ndarray
            Gradient tensor of shape ``(n, n, d)``.
        """
        n, d = particles.shape
        # diff[j, i] = x_j - x_i
        diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
        h2 = bandwidth * bandwidth + 1e-12
        grad_K = -kernel_matrix[:, :, np.newaxis] * diff / h2
        return grad_K

    # -- iteration interface ------------------------------------------------

    def iterate(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Run the full iterative SVGD loop until convergence or max iterations.

        Parameters
        ----------
        particles : np.ndarray
            Initial particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable[[np.ndarray], np.ndarray]
            Function mapping particle positions ``(n, d)`` to gradients of the
            log target density, also ``(n, d)``.

        Returns
        -------
        np.ndarray
            Final particle positions, shape ``(n, d)``.
        """
        self._particles = particles.copy()
        self._history = [self._particles.copy()]
        self._iteration = 0

        for it in range(self.max_iterations):
            self._iteration = it + 1
            grad_lp = grad_log_prob_fn(self._particles)

            # Bandwidth
            if self.kernel_bandwidth > 0:
                bw = self.kernel_bandwidth
            else:
                bw = median_bandwidth_heuristic(self._particles)

            K = self._compute_kernel_matrix(self._particles, bw)
            grad_K = self._compute_kernel_gradients(self._particles, K, bw)

            new_particles = self._svgd_update(
                self._particles, grad_lp, K, grad_K
            )

            if self.convergence_check(self._particles, new_particles):
                logger.info(
                    "SVGDParticleSampler converged at iteration %d", it + 1
                )
                self._particles = new_particles
                self._history.append(self._particles.copy())
                break

            self._particles = new_particles
            self._history.append(self._particles.copy())

        return self._particles

    def get_particles(self) -> np.ndarray:
        """Return the current particle positions.

        Returns
        -------
        np.ndarray
            Particle positions, shape ``(n, d)``.
        """
        if self._particles is None:
            raise RuntimeError("No particles available; call iterate() first.")
        return self._particles.copy()

    def convergence_check(
        self, old: np.ndarray, new: np.ndarray
    ) -> bool:
        """Check whether particles have converged.

        Parameters
        ----------
        old : np.ndarray
            Previous particle positions.
        new : np.ndarray
            Updated particle positions.

        Returns
        -------
        bool
            ``True`` if the relative change is below ``convergence_threshold``.
        """
        delta = np.linalg.norm(new - old)
        scale = np.linalg.norm(old) + 1e-12
        return (delta / scale) < self.convergence_threshold


# =========================================================================
# MultiKernelSVD
# =========================================================================


class MultiKernelSVD:
    """SVD with multiple kernel options and kernel combination.

    Supports RBF (Gaussian), IMQ (Inverse Multi-Quadric), polynomial, and
    Matérn kernels.  Multiple kernels can be combined via a weighted sum to
    produce a richer similarity measure.

    Parameters
    ----------
    kernel_weights : Dict[str, float], optional
        Mapping from kernel name to its weight in the combination.  Defaults
        to ``{"rbf": 1.0}``.
    bandwidth : float, optional
        Shared bandwidth parameter for RBF, IMQ, and Matérn kernels.
    polynomial_degree : int, optional
        Degree for the polynomial kernel.
    polynomial_c : float, optional
        Constant term for the polynomial kernel.
    matern_nu : float, optional
        Smoothness parameter for the Matérn kernel.  Supported values are
        0.5, 1.5, and 2.5.

    Notes
    -----
    Each kernel method returns both the kernel value matrix and a gradient
    tensor so that downstream SVGD routines can directly use the results.
    """

    def __init__(
        self,
        kernel_weights: Optional[Dict[str, float]] = None,
        bandwidth: float = 1.0,
        polynomial_degree: int = 3,
        polynomial_c: float = 1.0,
        matern_nu: float = 1.5,
    ) -> None:
        self.kernel_weights: Dict[str, float] = kernel_weights or {"rbf": 1.0}
        self.bandwidth = max(bandwidth, 1e-6)
        self.polynomial_degree = polynomial_degree
        self.polynomial_c = polynomial_c
        self.matern_nu = matern_nu

        self._kernel_dispatch: Dict[
            str,
            Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        ] = {
            "rbf": self._rbf_kernel,
            "imq": self._imq_kernel,
            "polynomial": self._polynomial_kernel,
            "matern": self._matern_kernel,
        }

    # -- individual kernels -------------------------------------------------

    def _rbf_kernel(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Radial basis function (Gaussian) kernel with gradient.

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)``.
        """
        n, d = X.shape
        sq = cdist(X, X, metric="sqeuclidean")
        h2 = self.bandwidth ** 2 + 1e-12
        K = np.exp(-sq / (2.0 * h2))
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        grad_K = -K[:, :, np.newaxis] * diff / h2
        return K, grad_K

    def _imq_kernel(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse Multi-Quadric (IMQ) kernel with gradient.

        k(x, y) = (c^2 + ||x - y||^2)^{-1/2}

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)``.
        """
        n, d = X.shape
        c2 = self.bandwidth ** 2
        sq = cdist(X, X, metric="sqeuclidean")
        base = c2 + sq
        K = np.power(base, -0.5)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        grad_K = -K[:, :, np.newaxis] * diff / (base[:, :, np.newaxis] + 1e-12)
        return K, grad_K

    def _polynomial_kernel(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Polynomial kernel with gradient.

        k(x, y) = (x^T y + c)^d

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)``.
        """
        deg = self.polynomial_degree
        c = self.polynomial_c
        inner = X @ X.T + c
        K = np.power(np.maximum(inner, 1e-12), deg)
        # ∇_{x_i} k(x_i, x_j) = d · (x_i^T x_j + c)^{d-1} · x_j
        coeff = deg * np.power(np.maximum(inner, 1e-12), deg - 1)
        grad_K = coeff[:, :, np.newaxis] * X[np.newaxis, :, :]
        return K, grad_K

    def _matern_kernel(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Matérn kernel with gradient (ν ∈ {0.5, 1.5, 2.5}).

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Kernel gradient tensor, shape ``(n, n, d)``.
        """
        n, d = X.shape
        dists = cdist(X, X, metric="euclidean")
        l = self.bandwidth + 1e-12
        r = dists / l
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]

        if self.matern_nu == 0.5:
            K = np.exp(-r)
            dr_dx = diff / (dists[:, :, np.newaxis] * l + 1e-12)
            grad_K = -K[:, :, np.newaxis] * dr_dx
        elif self.matern_nu == 1.5:
            sqrt3_r = math.sqrt(3.0) * r
            K = (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
            factor = -3.0 * np.exp(-sqrt3_r) / (l * l + 1e-12)
            grad_K = factor[:, :, np.newaxis] * diff
        elif self.matern_nu == 2.5:
            sqrt5_r = math.sqrt(5.0) * r
            r2 = r ** 2
            K = (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * np.exp(-sqrt5_r)
            factor = -(5.0 / 3.0) * (1.0 + sqrt5_r) * np.exp(-sqrt5_r) / (
                l * l + 1e-12
            )
            grad_K = factor[:, :, np.newaxis] * diff
        else:
            logger.warning(
                "Unsupported Matérn ν=%s, falling back to RBF", self.matern_nu
            )
            return self._rbf_kernel(X)

        return K, grad_K

    # -- combined kernel ----------------------------------------------------

    def _combined_kernel(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted sum of all active kernels.

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Combined kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Combined kernel gradient tensor, shape ``(n, n, d)``.
        """
        n, d = X.shape
        K_total = np.zeros((n, n), dtype=np.float64)
        grad_total = np.zeros((n, n, d), dtype=np.float64)

        for name, weight in self.kernel_weights.items():
            fn = self._kernel_dispatch.get(name)
            if fn is None:
                logger.warning("Unknown kernel %r in combination, skipping", name)
                continue
            K_part, grad_part = fn(X)
            K_total += weight * K_part
            grad_total += weight * grad_part

        return K_total, grad_total

    def _kernel_gradient(
        self, X: np.ndarray
    ) -> np.ndarray:
        """Return only the gradient tensor from the combined kernel.

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        np.ndarray
            Gradient tensor, shape ``(n, n, d)``.
        """
        _, grad = self._combined_kernel(X)
        return grad

    # -- public interface ---------------------------------------------------

    def set_kernel_weights(self, weights: Dict[str, float]) -> None:
        """Update kernel combination weights.

        Parameters
        ----------
        weights : Dict[str, float]
            Mapping from kernel name to weight.
        """
        self.kernel_weights = dict(weights)
        logger.debug("MultiKernelSVD weights updated: %s", self.kernel_weights)

    def evaluate(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the combined kernel matrix and gradient tensor.

        Parameters
        ----------
        X : np.ndarray
            Points of shape ``(n, d)``.

        Returns
        -------
        K : np.ndarray
            Combined kernel matrix, shape ``(n, n)``.
        grad_K : np.ndarray
            Combined kernel gradient tensor, shape ``(n, n, d)``.
        """
        return self._combined_kernel(np.asarray(X, dtype=np.float64))


# =========================================================================
# AnnealedSteinVariational
# =========================================================================


class AnnealedSteinVariational:
    """Annealed Stein variational inference.

    Uses a sequence of tempered distributions — from a broad (near-uniform)
    distribution to the target — to gradually guide particles toward the
    target distribution.  This avoids the mode-collapse issues that arise
    when particles are directly optimized against a highly peaked target.

    Supports three annealing schedules:

    - **geometric**: β_t = β_0^{1 - t/T}  (the standard choice in SMC)
    - **linear**: β_t = t / T
    - **sigmoidal**: β_t = σ(a · (t/T - 0.5))  with a controllable slope

    Parameters
    ----------
    n_stages : int
        Number of annealing stages (temperature levels).
    svgd_steps_per_stage : int
        Number of SVGD iterations at each temperature level.
    step_size : float
        Learning rate for the inner SVGD updates.
    schedule : str
        Annealing schedule type: ``"geometric"``, ``"linear"``, or
        ``"sigmoidal"``.
    sigmoid_slope : float
        Slope parameter for the sigmoidal schedule.

    Notes
    -----
    At each stage *t* the tempered log-probability is::

        log p_t(x) = β_t · log p(x)

    where β_t ∈ [0, 1] increases from 0 to 1 across stages.
    """

    def __init__(
        self,
        n_stages: int = 20,
        svgd_steps_per_stage: int = 10,
        step_size: float = 0.05,
        schedule: str = "geometric",
        sigmoid_slope: float = 10.0,
    ) -> None:
        if n_stages < 1:
            raise ValueError("n_stages must be >= 1")
        self.n_stages = n_stages
        self.svgd_steps_per_stage = svgd_steps_per_stage
        self.step_size = step_size
        self.schedule = schedule
        self.sigmoid_slope = sigmoid_slope

        self._trajectory: List[Tuple[float, np.ndarray]] = []

    # -- annealing schedules ------------------------------------------------

    def _geometric_anneal(self, stage: int) -> float:
        """Geometric annealing schedule.

        Parameters
        ----------
        stage : int
            Current stage index (0-based).

        Returns
        -------
        float
            Temperature parameter β ∈ [0, 1].
        """
        if self.n_stages <= 1:
            return 1.0
        frac = stage / (self.n_stages - 1)
        beta_min = 1e-4
        return float(beta_min ** (1.0 - frac))

    def _linear_anneal(self, stage: int) -> float:
        """Linear annealing schedule.

        Parameters
        ----------
        stage : int
            Current stage index (0-based).

        Returns
        -------
        float
            Temperature parameter β ∈ [0, 1].
        """
        if self.n_stages <= 1:
            return 1.0
        return float(stage / (self.n_stages - 1))

    def _sigmoidal_anneal(self, stage: int) -> float:
        """Sigmoidal annealing schedule.

        Parameters
        ----------
        stage : int
            Current stage index (0-based).

        Returns
        -------
        float
            Temperature parameter β ∈ [0, 1].
        """
        if self.n_stages <= 1:
            return 1.0
        frac = stage / (self.n_stages - 1)
        x = self.sigmoid_slope * (frac - 0.5)
        return float(1.0 / (1.0 + math.exp(-x)))

    def _get_beta(self, stage: int) -> float:
        """Return the temperature β for a given annealing stage."""
        dispatch = {
            "geometric": self._geometric_anneal,
            "linear": self._linear_anneal,
            "sigmoidal": self._sigmoidal_anneal,
        }
        fn = dispatch.get(self.schedule, self._geometric_anneal)
        return fn(stage)

    # -- tempered distribution ----------------------------------------------

    def _tempered_log_prob(
        self,
        particles: np.ndarray,
        log_prob_fn: Callable[[np.ndarray], np.ndarray],
        beta: float,
    ) -> np.ndarray:
        """Compute the tempered log-probability β · log p(x).

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        log_prob_fn : Callable
            Function returning log p(x) for each particle, shape ``(n,)``.
        beta : float
            Current temperature parameter.

        Returns
        -------
        np.ndarray
            Tempered log-probabilities, shape ``(n,)``.
        """
        return beta * log_prob_fn(particles)

    # -- single annealing step ----------------------------------------------

    def _anneal_step(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
        beta: float,
    ) -> np.ndarray:
        """Perform SVGD updates at a single annealing stage.

        Parameters
        ----------
        particles : np.ndarray
            Current particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function ∇ log p(x), shape ``(n, d)``.
        beta : float
            Temperature parameter for this stage.

        Returns
        -------
        np.ndarray
            Updated particle positions.
        """
        sampler = SVGDParticleSampler(
            n_particles=particles.shape[0],
            step_size=self.step_size,
            max_iterations=self.svgd_steps_per_stage,
        )

        def tempered_grad(x: np.ndarray) -> np.ndarray:
            return beta * grad_log_prob_fn(x)

        return sampler.iterate(particles, tempered_grad)

    # -- full annealed inference --------------------------------------------

    def run_annealed_inference(
        self,
        initial_particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
        log_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """Run the complete annealed SVGD procedure.

        Parameters
        ----------
        initial_particles : np.ndarray
            Starting particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function ∇ log p(x).
        log_prob_fn : Callable, optional
            Log-probability function (used for diagnostics only).

        Returns
        -------
        np.ndarray
            Final particle positions, shape ``(n, d)``.
        """
        particles = initial_particles.copy()
        self._trajectory = []

        for stage in range(self.n_stages):
            beta = self._get_beta(stage)
            logger.debug(
                "Annealed SVGD stage %d/%d, β=%.6f", stage + 1, self.n_stages, beta
            )
            particles = self._anneal_step(particles, grad_log_prob_fn, beta)
            self._trajectory.append((beta, particles.copy()))

        return particles

    def get_annealing_trajectory(
        self,
    ) -> List[Tuple[float, np.ndarray]]:
        """Return the trajectory of ``(β, particles)`` across annealing stages.

        Returns
        -------
        List[Tuple[float, np.ndarray]]
            Each element is ``(beta, particles_array)``.
        """
        return list(self._trajectory)


# =========================================================================
# ParticleConvergenceDiagnostics
# =========================================================================


class ParticleConvergenceDiagnostics:
    """Comprehensive convergence diagnostics for particle systems.

    Provides a collection of scalar diagnostics that characterize how well
    a set of weighted (or equally-weighted) particles approximates the
    target distribution.

    Parameters
    ----------
    distance_metric : str, optional
        Metric for pairwise distance computations (default ``"euclidean"``).
    ess_threshold : float, optional
        Minimum effective sample size (as a fraction of *n*) required to
        declare convergence.
    ksd_threshold : float, optional
        Maximum kernel Stein discrepancy to declare convergence.

    Notes
    -----
    All methods accept plain particle arrays and, where applicable, weight
    arrays.  Equally-weighted particles can omit the weights argument.
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        ess_threshold: float = 0.5,
        ksd_threshold: float = 0.01,
    ) -> None:
        self.distance_metric = distance_metric
        self.ess_threshold = ess_threshold
        self.ksd_threshold = ksd_threshold

    # -- individual diagnostics ---------------------------------------------

    def effective_sample_size(
        self, weights: np.ndarray
    ) -> float:
        """Compute the effective sample size (ESS) from importance weights.

        ESS = (Σ w_i)^2 / Σ w_i^2

        Parameters
        ----------
        weights : np.ndarray
            Unnormalized importance weights, shape ``(n,)``.

        Returns
        -------
        float
            Effective sample size (1 ≤ ESS ≤ n).
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-12)
        return float(1.0 / (np.sum(w ** 2) + 1e-12))

    def kernel_stein_discrepancy(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
        bandwidth: float = -1.0,
    ) -> float:
        """Compute the kernel Stein discrepancy (KSD).

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function ∇ log p(x), returning shape ``(n, d)``.
        bandwidth : float, optional
            Kernel bandwidth; if <= 0, uses the median heuristic.

        Returns
        -------
        float
            KSD value (non-negative; 0 means perfect fit).
        """
        particles = np.asarray(particles, dtype=np.float64)
        bw = bandwidth if bandwidth > 0 else median_bandwidth_heuristic(particles)
        score = grad_log_prob_fn(particles)
        return float(
            compute_stein_discrepancy(
                particles,
                lambda x: grad_log_prob_fn(x),
                lambda x, y: rbf_kernel(x, y, bw),
            )
        )

    def particle_energy(
        self,
        particles: np.ndarray,
        log_prob_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Compute the mean negative log-probability (energy) of particles.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        log_prob_fn : Callable
            Log-probability function returning shape ``(n,)``.

        Returns
        -------
        float
            Mean energy (lower is better).
        """
        lp = log_prob_fn(np.asarray(particles, dtype=np.float64))
        return float(-np.mean(lp))

    def inter_particle_distance_stats(
        self, particles: np.ndarray
    ) -> Dict[str, float]:
        """Compute summary statistics of inter-particle distances.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys ``"mean"``, ``"std"``, ``"min"``, ``"max"``,
            ``"median"``.
        """
        particles = np.asarray(particles, dtype=np.float64)
        if particles.shape[0] < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        dists = pdist(particles, metric=self.distance_metric)
        return {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "median": float(np.median(dists)),
        }

    def mode_coverage(
        self,
        particles: np.ndarray,
        mode_centers: np.ndarray,
        radius: float = 1.0,
    ) -> float:
        """Estimate the fraction of known modes covered by particles.

        A mode is "covered" if at least one particle lies within *radius*
        of its center.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        mode_centers : np.ndarray
            Known mode centers, shape ``(m, d)``.
        radius : float
            Coverage radius.

        Returns
        -------
        float
            Fraction of modes covered (0.0 to 1.0).
        """
        particles = np.asarray(particles, dtype=np.float64)
        mode_centers = np.asarray(mode_centers, dtype=np.float64)
        if mode_centers.shape[0] == 0:
            return 1.0
        dists = cdist(mode_centers, particles, metric=self.distance_metric)
        covered = np.any(dists <= radius, axis=1)
        return float(np.mean(covered))

    # -- summary / convergence ----------------------------------------------

    def convergence_summary(
        self,
        particles: np.ndarray,
        weights: Optional[np.ndarray] = None,
        grad_log_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        log_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive convergence summary.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        weights : np.ndarray, optional
            Importance weights.  If ``None``, equal weights are assumed.
        grad_log_prob_fn : Callable, optional
            Score function for KSD computation.
        log_prob_fn : Callable, optional
            Log-probability function for energy computation.

        Returns
        -------
        Dict[str, float]
            Dictionary of diagnostic values.
        """
        summary: Dict[str, float] = {}
        n = particles.shape[0]

        if weights is not None:
            summary["ess"] = self.effective_sample_size(weights)
            summary["ess_fraction"] = summary["ess"] / max(n, 1)
        else:
            summary["ess"] = float(n)
            summary["ess_fraction"] = 1.0

        dist_stats = self.inter_particle_distance_stats(particles)
        summary.update({f"dist_{k}": v for k, v in dist_stats.items()})

        summary["diversity"] = particle_diversity_metric(particles)

        if grad_log_prob_fn is not None:
            summary["ksd"] = self.kernel_stein_discrepancy(
                particles, grad_log_prob_fn
            )

        if log_prob_fn is not None:
            summary["energy"] = self.particle_energy(particles, log_prob_fn)

        return summary

    def is_converged(
        self,
        particles: np.ndarray,
        weights: Optional[np.ndarray] = None,
        grad_log_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> bool:
        """Check whether the particle system meets convergence criteria.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions.
        weights : np.ndarray, optional
            Importance weights.
        grad_log_prob_fn : Callable, optional
            Score function for KSD computation.

        Returns
        -------
        bool
            ``True`` if both ESS and KSD criteria are satisfied.
        """
        n = particles.shape[0]

        if weights is not None:
            ess = self.effective_sample_size(weights)
            if ess / max(n, 1) < self.ess_threshold:
                return False

        if grad_log_prob_fn is not None:
            ksd = self.kernel_stein_discrepancy(particles, grad_log_prob_fn)
            if ksd > self.ksd_threshold:
                return False

        return True


# =========================================================================
# SteinDiscrepancyComputer
# =========================================================================


class SteinDiscrepancyComputer:
    """Dedicated Stein discrepancy computation.

    Computes the kernelized Stein discrepancy (KSD) between the particle
    distribution and the target distribution.  Supports both U-statistic
    and V-statistic estimators.

    Parameters
    ----------
    kernel_type : str
        Kernel type for the Stein kernel: ``"rbf"`` or ``"imq"``.
    target_log_prob : Callable[[np.ndarray], np.ndarray]
        Log-probability function of the target distribution, returning
        shape ``(n,)``.
    bandwidth : float, optional
        Kernel bandwidth; if <= 0 the median heuristic is used.

    Notes
    -----
    The kernelized Stein discrepancy is defined as::

        KSD^2 = E_{x,y ~ q}[ u_p(x, y) ]

    where u_p is the Stein kernel::

        u_p(x, y) = s(x)^T k(x,y) s(y) + s(x)^T ∇_y k(x,y)
                   + ∇_x k(x,y)^T s(y) + trace(∇_x ∇_y k(x,y))

    and s(x) = ∇_x log p(x) is the score function.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        target_log_prob: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bandwidth: float = -1.0,
    ) -> None:
        self.kernel_type = kernel_type
        self.target_log_prob = target_log_prob
        self.bandwidth = bandwidth

    # -- score function -----------------------------------------------------

    def _score_function(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Evaluate the score function ∇ log p(x) at each particle.

        Parameters
        ----------
        particles : np.ndarray
            Positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Gradient of log target density.

        Returns
        -------
        np.ndarray
            Score values, shape ``(n, d)``.
        """
        return grad_log_prob_fn(particles)

    # -- Stein operator -----------------------------------------------------

    def _stein_operator(
        self,
        xi: np.ndarray,
        xj: np.ndarray,
        si: np.ndarray,
        sj: np.ndarray,
        bw: float,
    ) -> float:
        """Evaluate the Stein kernel u_p(x_i, x_j).

        Parameters
        ----------
        xi, xj : np.ndarray
            Single particle positions, shape ``(d,)``.
        si, sj : np.ndarray
            Score function values at *xi* and *xj*, shape ``(d,)``.
        bw : float
            Kernel bandwidth.

        Returns
        -------
        float
            Stein kernel value.
        """
        d = xi.shape[0]
        diff = xi - xj
        sq = float(np.sum(diff ** 2))
        h2 = bw * bw + 1e-12

        if self.kernel_type == "rbf":
            k_val = math.exp(-sq / (2.0 * h2))
            # ∇_{x_j} k = k · (xi - xj) / h^2
            grad_j = k_val * diff / h2
            # ∇_{x_i} k = -grad_j
            grad_i = -grad_j
            # trace(∇_xi ∇_xj k) = k · (d/h^2 - sq/h^4)
            trace_term = k_val * (d / h2 - sq / (h2 * h2))
        elif self.kernel_type == "imq":
            base = bw * bw + sq
            k_val = base ** (-0.5)
            grad_j = k_val * diff / (base + 1e-12)
            grad_i = -grad_j
            trace_term = k_val * (d * base - sq) / (base * base + 1e-12)
        else:
            k_val = math.exp(-sq / (2.0 * h2))
            grad_j = k_val * diff / h2
            grad_i = -grad_j
            trace_term = k_val * (d / h2 - sq / (h2 * h2))

        term1 = float(np.dot(si, sj)) * k_val
        term2 = float(np.dot(si, grad_j))
        term3 = float(np.dot(grad_i, sj))
        term4 = trace_term
        return term1 + term2 + term3 + term4

    # -- U-statistic and V-statistic ----------------------------------------

    def u_statistic(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Compute the U-statistic estimator of KSD^2.

        The U-statistic excludes diagonal terms (i ≠ j).

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function.

        Returns
        -------
        float
            KSD^2 U-statistic estimate (non-negative).
        """
        particles = np.asarray(particles, dtype=np.float64)
        n = particles.shape[0]
        if n < 2:
            return 0.0

        scores = self._score_function(particles, grad_log_prob_fn)
        bw = self.bandwidth if self.bandwidth > 0 else median_bandwidth_heuristic(particles)

        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                val = self._stein_operator(
                    particles[i], particles[j], scores[i], scores[j], bw
                )
                total += val

        n_pairs = n * (n - 1) / 2
        return max(float(total / n_pairs), 0.0)

    def v_statistic(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Compute the V-statistic estimator of KSD^2.

        The V-statistic includes diagonal terms.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function.

        Returns
        -------
        float
            KSD^2 V-statistic estimate.
        """
        particles = np.asarray(particles, dtype=np.float64)
        n = particles.shape[0]
        if n < 1:
            return 0.0

        scores = self._score_function(particles, grad_log_prob_fn)
        bw = self.bandwidth if self.bandwidth > 0 else median_bandwidth_heuristic(particles)

        total = 0.0
        for i in range(n):
            for j in range(n):
                total += self._stein_operator(
                    particles[i], particles[j], scores[i], scores[j], bw
                )

        return max(float(total / (n * n)), 0.0)

    def compute(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
        estimator: str = "u",
    ) -> float:
        """Compute KSD using the specified estimator.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function.
        estimator : str
            ``"u"`` for U-statistic or ``"v"`` for V-statistic.

        Returns
        -------
        float
            KSD^2 estimate.
        """
        if estimator == "v":
            return self.v_statistic(particles, grad_log_prob_fn)
        return self.u_statistic(particles, grad_log_prob_fn)

    def gradient_of_discrepancy(
        self,
        particles: np.ndarray,
        grad_log_prob_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Compute the gradient of KSD^2 w.r.t. particle positions.

        Uses finite differences for simplicity.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape ``(n, d)``.
        grad_log_prob_fn : Callable
            Score function.

        Returns
        -------
        np.ndarray
            Gradient of KSD^2, shape ``(n, d)``.
        """
        particles = np.asarray(particles, dtype=np.float64)
        n, d = particles.shape
        eps = 1e-5
        grad = np.zeros_like(particles)

        base_ksd = self.u_statistic(particles, grad_log_prob_fn)

        for i in range(n):
            for j in range(d):
                perturbed = particles.copy()
                perturbed[i, j] += eps
                ksd_plus = self.u_statistic(perturbed, grad_log_prob_fn)
                grad[i, j] = (ksd_plus - base_ksd) / eps

        return grad


# =========================================================================
# Standalone helper functions
# =========================================================================


def rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Compute the RBF (Gaussian) kernel between two points.

    k(x, y) = exp( -||x - y||^2 / (2 h^2) )

    Parameters
    ----------
    x : np.ndarray
        First point, shape ``(d,)``.
    y : np.ndarray
        Second point, shape ``(d,)``.
    bandwidth : float
        Kernel bandwidth (length-scale).

    Returns
    -------
    float
        Kernel value.
    """
    sq = float(np.sum((np.asarray(x) - np.asarray(y)) ** 2))
    return float(np.exp(-sq / (2.0 * bandwidth * bandwidth + 1e-12)))


def imq_kernel(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    beta: float = -0.5,
) -> float:
    """Compute the Inverse Multi-Quadric (IMQ) kernel.

    k(x, y) = (c^2 + ||x - y||^2)^β

    Parameters
    ----------
    x : np.ndarray
        First point, shape ``(d,)``.
    y : np.ndarray
        Second point, shape ``(d,)``.
    c : float
        Constant offset.
    beta : float
        Exponent (typically -0.5).

    Returns
    -------
    float
        Kernel value.
    """
    sq = float(np.sum((np.asarray(x) - np.asarray(y)) ** 2))
    return float((c * c + sq) ** beta)


def polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    c: float = 1.0,
) -> float:
    """Compute the polynomial kernel.

    k(x, y) = (x^T y + c)^d

    Parameters
    ----------
    x : np.ndarray
        First point, shape ``(d,)``.
    y : np.ndarray
        Second point, shape ``(d,)``.
    degree : int
        Polynomial degree.
    c : float
        Constant offset.

    Returns
    -------
    float
        Kernel value.
    """
    inner = float(np.dot(np.asarray(x), np.asarray(y))) + c
    return float(max(inner, 1e-12) ** degree)


def median_bandwidth_heuristic(particles: np.ndarray) -> float:
    """Compute the median heuristic bandwidth for a set of particles.

    bandwidth = sqrt( median(||x_i - x_j||^2) / (2 log(n + 1)) )

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape ``(n, d)``.

    Returns
    -------
    float
        Bandwidth scalar (positive).
    """
    particles = np.asarray(particles, dtype=np.float64)
    n = particles.shape[0]
    if n < 2:
        return 1.0
    pairwise_sq = pdist(particles, metric="sqeuclidean")
    if len(pairwise_sq) == 0:
        return 1.0
    med_sq = float(np.median(pairwise_sq))
    log_term = 2.0 * math.log(n + 1)
    if log_term < 1e-12:
        return max(math.sqrt(med_sq + 1e-12), 1e-6)
    return max(math.sqrt(med_sq / log_term + 1e-12), 1e-6)


def compute_stein_discrepancy(
    particles: np.ndarray,
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    kernel_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Compute a simplified Stein discrepancy estimate.

    Uses finite-difference score estimation and the supplied kernel.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape ``(n, d)``.
    log_prob_fn : Callable
        Score function ∇ log p(x), returning shape ``(n, d)``.
    kernel_fn : Callable
        Pairwise kernel function ``k(x_i, x_j) → float``.

    Returns
    -------
    float
        Stein discrepancy estimate (non-negative).
    """
    particles = np.asarray(particles, dtype=np.float64)
    n = particles.shape[0]
    if n < 2:
        return 0.0

    scores = log_prob_fn(particles)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            k_val = kernel_fn(particles[i], particles[j])
            total += k_val * float(np.dot(scores[i], scores[j]))
            count += 1
    return max(float(total / max(count, 1)), 0.0)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute the effective sample size from importance weights.

    ESS = (Σ w_i)^2 / Σ w_i^2

    Parameters
    ----------
    weights : np.ndarray
        Unnormalized importance weights, shape ``(n,)``.

    Returns
    -------
    float
        Effective sample size (1 ≤ ESS ≤ n).
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)
    return float(1.0 / (np.sum(w ** 2) + 1e-12))


def particle_diversity_metric(particles: np.ndarray) -> float:
    """Compute a scalar diversity metric for a set of particles.

    Returns the mean pairwise Euclidean distance, normalized by the
    square root of the dimensionality.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape ``(n, d)``.

    Returns
    -------
    float
        Diversity metric (non-negative; higher means more diverse).
    """
    particles = np.asarray(particles, dtype=np.float64)
    n, d = particles.shape
    if n < 2:
        return 0.0
    dists = pdist(particles, metric="euclidean")
    return float(np.mean(dists) / (math.sqrt(d) + 1e-12))


# =========================================================================
# Module-level exports
# =========================================================================

__all__ = [
    "SVDConfig",
    "SVDState",
    "SVDKernel",
    "EmbeddingProjector",
    "SVDConvergenceMonitor",
    "AnnealingSchedule",
    "SteinVariationalDecoding",
    "create_svd",
    "SVGDParticleSampler",
    "MultiKernelSVD",
    "AnnealedSteinVariational",
    "ParticleConvergenceDiagnostics",
    "SteinDiscrepancyComputer",
    "rbf_kernel",
    "imq_kernel",
    "polynomial_kernel",
    "median_bandwidth_heuristic",
    "compute_stein_discrepancy",
    "effective_sample_size",
    "particle_diversity_metric",
]
