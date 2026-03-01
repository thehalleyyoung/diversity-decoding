"""
Vendi Score implementation for the Diversity Decoding Arena.

The Vendi Score measures the "effective number of distinct items" in a set,
defined as:

    VS(G) = exp( -sum_i lambda_i * log(lambda_i) )

where lambda_i are the eigenvalues of the normalized kernel matrix K.
Equivalently, VS = exp(H(K)) where H is the von Neumann (matrix) entropy
of the kernel matrix after trace-normalizing so eigenvalues sum to 1.

References:
    Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for
    Machine Learning", 2023.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist, squareform


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG2E = math.log2(math.e)
_EPSILON = 1e-12  # Guard against log(0)
_EIGENVALUE_FLOOR = -1e-10  # Clip tiny negatives from numerical noise


# ===================================================================
# Helper functions
# ===================================================================

def safe_log(x: np.ndarray, *, base: str = "natural") -> np.ndarray:
    """Element-wise log that maps 0 -> 0, avoiding -inf.

    Parameters
    ----------
    x : array-like
        Non-negative values.
    base : {"natural", "2", "10"}
        Logarithm base.

    Returns
    -------
    np.ndarray
        log(x) with the convention 0*log(0) = 0.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = x > 0
    if base == "natural":
        out[mask] = np.log(x[mask])
    elif base == "2":
        out[mask] = np.log2(x[mask])
    elif base == "10":
        out[mask] = np.log10(x[mask])
    else:
        raise ValueError(f"Unknown base: {base}")
    return out


def safe_entropy(probs: np.ndarray, *, base: str = "natural") -> float:
    """Shannon entropy H = -sum p_i log p_i with 0*log(0) = 0 convention.

    Parameters
    ----------
    probs : 1-D array
        Non-negative values that sum to 1 (or will be normalised).
    base : str
        Logarithm base.

    Returns
    -------
    float
        Entropy value >= 0.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    probs = probs / (probs.sum() + _EPSILON)
    log_p = safe_log(probs, base=base)
    return float(-np.sum(probs * log_p))


def eigenvalue_decomposition(
    K: np.ndarray,
    *,
    floor: float = _EIGENVALUE_FLOOR,
    normalise: bool = True,
) -> np.ndarray:
    """Numerically-stable eigenvalue decomposition of a symmetric PSD matrix.

    Parameters
    ----------
    K : (n, n) array
        Symmetric positive semi-definite kernel matrix.
    floor : float
        Eigenvalues below this threshold are clipped to zero.
    normalise : bool
        If True, divide eigenvalues by their sum so they form a probability
        distribution (needed for matrix entropy).

    Returns
    -------
    eigenvalues : (n,) array
        Sorted eigenvalues in descending order.
    """
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    # Symmetrise to avoid issues from floating-point asymmetry
    K_sym = 0.5 * (K + K.T)

    eigvals = eigh(K_sym, eigvals_only=True)

    # Clip tiny negatives introduced by numerical error
    eigvals = np.clip(eigvals, a_min=max(floor, 0.0), a_max=None)

    if normalise:
        total = eigvals.sum()
        if total > _EPSILON:
            eigvals = eigvals / total

    # Sort descending
    eigvals = np.sort(eigvals)[::-1]
    return eigvals


def kernel_normalize(K: np.ndarray) -> np.ndarray:
    """Normalise a kernel matrix so that K_ii = 1 for all i.

    K_ij <- K_ij / sqrt(K_ii * K_jj)

    Parameters
    ----------
    K : (n, n) array

    Returns
    -------
    K_norm : (n, n) array
    """
    K = np.asarray(K, dtype=np.float64)
    diag = np.diag(K).copy()
    diag = np.maximum(diag, _EPSILON)
    sqrt_diag = np.sqrt(diag)
    K_norm = K / np.outer(sqrt_diag, sqrt_diag)
    # Clamp to [-1, 1] for numerical safety
    np.clip(K_norm, -1.0, 1.0, out=K_norm)
    return K_norm


def median_bandwidth_heuristic(X: np.ndarray) -> float:
    """Estimate RBF bandwidth sigma via the median heuristic.

    sigma = median of pairwise Euclidean distances (excluding zero).

    Parameters
    ----------
    X : (n, d) array of embeddings.

    Returns
    -------
    sigma : float > 0
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    dists = pdist(X, metric="euclidean")
    if len(dists) == 0:
        return 1.0
    med = float(np.median(dists))
    return max(med, _EPSILON)


def silverman_bandwidth(X: np.ndarray) -> float:
    """Silverman's rule of thumb for RBF bandwidth.

    sigma = ( 4 / (n * (d + 2)) )^(1/(d+4)) * mean_std

    Parameters
    ----------
    X : (n, d) array

    Returns
    -------
    sigma : float > 0
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    std = np.std(X, axis=0)
    mean_std = float(np.mean(std)) if np.mean(std) > _EPSILON else 1.0
    factor = (4.0 / (n * (d + 2))) ** (1.0 / (d + 4))
    return max(factor * mean_std, _EPSILON)


def _validate_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Basic validation of a kernel matrix."""
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2:
        raise ValueError(f"Kernel matrix must be 2-D, got shape {K.shape}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(
            f"Kernel matrix must be square, got shape {K.shape}"
        )
    return K


def _pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix from row vectors."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPSILON)
    X_normed = X / norms
    return X_normed @ X_normed.T


# ===================================================================
# KernelBuilder
# ===================================================================

class KernelBuilder:
    """Factory for constructing kernel (Gram) matrices.

    Provides static methods for common kernels as well as a flexible
    ``build`` entry-point that dispatches by name.

    Examples
    --------
    >>> kb = KernelBuilder()
    >>> K = kb.rbf(X, sigma=1.0)
    >>> K = kb.build(X, kernel_type="cosine")
    """

    # ---- Individual kernel functions ---------------------------------

    @staticmethod
    def rbf(
        X: np.ndarray,
        *,
        sigma: Optional[float] = None,
        bandwidth_method: str = "median",
    ) -> np.ndarray:
        """Radial Basis Function (Gaussian) kernel.

        K_ij = exp( -||x_i - x_j||^2 / (2 * sigma^2) )

        Parameters
        ----------
        X : (n, d) array
        sigma : float or None
            Bandwidth.  If None, estimated automatically.
        bandwidth_method : {"median", "silverman"}
            Method used when *sigma* is None.

        Returns
        -------
        K : (n, n) array
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if sigma is None:
            if bandwidth_method == "median":
                sigma = median_bandwidth_heuristic(X)
            elif bandwidth_method == "silverman":
                sigma = silverman_bandwidth(X)
            else:
                raise ValueError(f"Unknown bandwidth method: {bandwidth_method}")

        sq_dists = cdist(X, X, metric="sqeuclidean")
        K = np.exp(-sq_dists / (2.0 * sigma ** 2))
        return K

    @staticmethod
    def linear(X: np.ndarray) -> np.ndarray:
        """Linear (dot-product) kernel.

        K_ij = x_i . x_j
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ X.T

    @staticmethod
    def cosine(X: np.ndarray) -> np.ndarray:
        """Cosine similarity kernel.

        K_ij = (x_i . x_j) / (||x_i|| ||x_j||)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return _pairwise_cosine_similarity(X)

    @staticmethod
    def polynomial(
        X: np.ndarray,
        *,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> np.ndarray:
        """Polynomial kernel.

        K_ij = (x_i . x_j + coef0) ^ degree
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X @ X.T + coef0) ** degree

    @staticmethod
    def laplacian(
        X: np.ndarray,
        *,
        sigma: Optional[float] = None,
        bandwidth_method: str = "median",
    ) -> np.ndarray:
        """Laplacian kernel.

        K_ij = exp( -||x_i - x_j||_1 / sigma )

        Parameters
        ----------
        X : (n, d) array
        sigma : float or None
            Bandwidth.
        bandwidth_method : str
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if sigma is None:
            if bandwidth_method == "median":
                sigma = median_bandwidth_heuristic(X)
            elif bandwidth_method == "silverman":
                sigma = silverman_bandwidth(X)
            else:
                raise ValueError(f"Unknown bandwidth method: {bandwidth_method}")

        dists = cdist(X, X, metric="cityblock")
        return np.exp(-dists / sigma)

    @staticmethod
    def string_kernel(
        texts: Sequence[str],
        *,
        n: int = 3,
        normalize: bool = True,
    ) -> np.ndarray:
        """String kernel based on shared character n-gram counts.

        For each pair of strings, the kernel value is the number of
        shared n-grams (with multiplicity).  Optionally normalised.

        Parameters
        ----------
        texts : sequence of str
        n : int
            N-gram size.
        normalize : bool
            If True, apply kernel normalisation K_ij / sqrt(K_ii K_jj).

        Returns
        -------
        K : (m, m) array
        """
        m = len(texts)

        def _ngram_counts(s: str) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for i in range(len(s) - n + 1):
                gram = s[i : i + n]
                counts[gram] = counts.get(gram, 0) + 1
            return counts

        profiles = [_ngram_counts(t) for t in texts]

        K = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i, m):
                shared = set(profiles[i].keys()) & set(profiles[j].keys())
                val = sum(
                    min(profiles[i][g], profiles[j][g]) for g in shared
                )
                K[i, j] = val
                K[j, i] = val

        if normalize and m > 0:
            K = kernel_normalize(K)
        return K

    @staticmethod
    def composite(
        kernels: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Weighted combination of multiple kernels.

        K = sum_t w_t K_t

        Parameters
        ----------
        kernels : list of (n, n) arrays
        weights : list of floats (summing to 1) or None for uniform.
        """
        if not kernels:
            raise ValueError("At least one kernel is required.")
        n = kernels[0].shape[0]
        if weights is None:
            weights = [1.0 / len(kernels)] * len(kernels)
        if len(weights) != len(kernels):
            raise ValueError("weights and kernels must have the same length.")

        K = np.zeros((n, n), dtype=np.float64)
        for w, Ki in zip(weights, kernels):
            Ki = np.asarray(Ki, dtype=np.float64)
            if Ki.shape != (n, n):
                raise ValueError(
                    f"All kernels must be ({n}, {n}); got {Ki.shape}"
                )
            K += w * Ki
        return K

    # ---- Dispatch by name --------------------------------------------

    def build(
        self,
        X: np.ndarray,
        kernel_type: str = "rbf",
        *,
        normalize: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Build a kernel matrix from embeddings.

        Parameters
        ----------
        X : (n, d) array
        kernel_type : str
            One of "rbf", "linear", "cosine", "polynomial", "laplacian".
        normalize : bool
            Apply kernel normalisation after construction.
        **kwargs
            Forwarded to the specific kernel function.

        Returns
        -------
        K : (n, n) array
        """
        builders = {
            "rbf": self.rbf,
            "gaussian": self.rbf,
            "linear": self.linear,
            "cosine": self.cosine,
            "polynomial": self.polynomial,
            "poly": self.polynomial,
            "laplacian": self.laplacian,
        }
        kernel_type_lower = kernel_type.lower()
        if kernel_type_lower not in builders:
            raise ValueError(
                f"Unknown kernel type '{kernel_type}'. "
                f"Choose from {list(builders.keys())}"
            )

        # Filter kwargs to those accepted by the kernel function
        func = builders[kernel_type_lower]
        K = func(X, **kwargs)

        if normalize:
            K = kernel_normalize(K)
        return K

    # ---- Bandwidth selection helpers ---------------------------------

    @staticmethod
    def select_bandwidth(
        X: np.ndarray,
        method: str = "median",
    ) -> float:
        """Select bandwidth for RBF / Laplacian kernels.

        Parameters
        ----------
        X : (n, d) array
        method : {"median", "silverman"}

        Returns
        -------
        sigma : float
        """
        if method == "median":
            return median_bandwidth_heuristic(X)
        elif method == "silverman":
            return silverman_bandwidth(X)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def bandwidth_grid(
        X: np.ndarray,
        *,
        num: int = 20,
        low_factor: float = 0.1,
        high_factor: float = 10.0,
    ) -> np.ndarray:
        """Generate a log-spaced grid of bandwidth values around the median.

        Parameters
        ----------
        X : (n, d) array
        num : int
            Number of grid points.
        low_factor, high_factor : float
            Multiply the median bandwidth by these factors for the range.

        Returns
        -------
        sigmas : (num,) array
        """
        med = median_bandwidth_heuristic(X)
        return np.geomspace(med * low_factor, med * high_factor, num=num)


# ===================================================================
# VendiScoreComputer
# ===================================================================

class VendiScoreComputer:
    """Core Vendi Score computation.

    The Vendi Score is defined as:

        VS = exp( H(λ) )

    where λ are the eigenvalues of the trace-normalised kernel matrix
    (so sum(λ) = 1), and H is Shannon entropy.

    Parameters
    ----------
    eigenvalue_floor : float
        Eigenvalues below this value are clipped to zero.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self.eigenvalue_floor = eigenvalue_floor
        self._kernel_builder = KernelBuilder()

    # ---- Matrix entropy ----------------------------------------------

    def matrix_entropy(
        self,
        K: np.ndarray,
        *,
        base: str = "natural",
    ) -> float:
        """Von Neumann entropy of a kernel matrix.

        H(K) = - sum_i lambda_i log(lambda_i)

        where lambda are the eigenvalues of K / tr(K).

        Parameters
        ----------
        K : (n, n) symmetric PSD matrix.
        base : {"natural", "2", "10"}

        Returns
        -------
        entropy : float >= 0
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        if n == 0:
            return 0.0
        if n == 1:
            return 0.0

        eigvals = eigenvalue_decomposition(
            K,
            floor=self.eigenvalue_floor,
            normalise=True,
        )
        return safe_entropy(eigvals, base=base)

    # ---- Main computation --------------------------------------------

    def compute(self, kernel_matrix: np.ndarray) -> float:
        """Compute the Vendi Score from a kernel matrix.

        VS = exp( H(K) )

        Parameters
        ----------
        kernel_matrix : (n, n) symmetric PSD array.

        Returns
        -------
        vendi_score : float >= 1
        """
        K = _validate_kernel_matrix(kernel_matrix)
        n = K.shape[0]
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        entropy = self.matrix_entropy(K)
        return math.exp(entropy)

    def compute_from_embeddings(
        self,
        embeddings: np.ndarray,
        kernel_type: str = "rbf",
        *,
        normalize_kernel: bool = True,
        **kernel_kwargs: Any,
    ) -> float:
        """Compute Vendi Score directly from embedding vectors.

        Parameters
        ----------
        embeddings : (n, d) array
        kernel_type : str
            Kernel to use (see ``KernelBuilder.build``).
        normalize_kernel : bool
            Normalise the kernel matrix before scoring.
        **kernel_kwargs
            Extra arguments forwarded to the kernel builder.

        Returns
        -------
        vendi_score : float
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        K = self._kernel_builder.build(
            embeddings,
            kernel_type=kernel_type,
            **kernel_kwargs,
        )
        if normalize_kernel:
            K = kernel_normalize(K)
        return self.compute(K)

    def compute_from_texts(
        self,
        texts: Sequence[str],
        similarity_func: Optional[Callable[[str, str], float]] = None,
        *,
        kernel_type: str = "ngram",
        ngram_size: int = 3,
    ) -> float:
        """Compute Vendi Score from a collection of text strings.

        Parameters
        ----------
        texts : sequence of str
        similarity_func : callable(str, str) -> float, optional
            Pairwise similarity function in [0, 1].  If None, use character
            n-gram string kernel.
        kernel_type : str
            Only used if similarity_func is None.
        ngram_size : int
            N-gram size for the string kernel.

        Returns
        -------
        vendi_score : float
        """
        m = len(texts)
        if m == 0:
            return 0.0
        if m == 1:
            return 1.0

        if similarity_func is not None:
            K = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                for j in range(i, m):
                    s = similarity_func(texts[i], texts[j])
                    K[i, j] = s
                    K[j, i] = s
            K = kernel_normalize(K)
        else:
            K = KernelBuilder.string_kernel(
                texts, n=ngram_size, normalize=True
            )

        return self.compute(K)

    def eigenvalues(
        self,
        kernel_matrix: np.ndarray,
        *,
        normalise: bool = True,
    ) -> np.ndarray:
        """Return the eigenvalues used in the Vendi computation.

        Parameters
        ----------
        kernel_matrix : (n, n) array
        normalise : bool
            If True, eigenvalues sum to 1.

        Returns
        -------
        eigvals : (n,) array, descending order.
        """
        K = _validate_kernel_matrix(kernel_matrix)
        return eigenvalue_decomposition(
            K, floor=self.eigenvalue_floor, normalise=normalise
        )


# ===================================================================
# VendiScoreVariants
# ===================================================================

class VendiScoreVariants:
    """Alternative formulations of the Vendi Score.

    All variants are based on the eigenspectrum of the kernel matrix.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self.eigenvalue_floor = eigenvalue_floor

    def _get_eigvals(self, K: np.ndarray) -> np.ndarray:
        """Trace-normalised eigenvalues."""
        return eigenvalue_decomposition(
            K, floor=self.eigenvalue_floor, normalise=True
        )

    # ---- Standard ----------------------------------------------------

    def standard(self, K: np.ndarray) -> float:
        """Standard Vendi Score = exp(H_Shannon(lambda))."""
        eigvals = self._get_eigvals(K)
        H = safe_entropy(eigvals)
        return math.exp(H)

    # ---- Alpha-order (Rényi) -----------------------------------------

    def alpha_order(self, K: np.ndarray, *, alpha: float = 2.0) -> float:
        """Alpha-order Vendi Score based on Rényi entropy.

        For alpha != 1:
            H_alpha = (1 / (1 - alpha)) * log( sum_i lambda_i^alpha )
            VS_alpha = exp( H_alpha )

        For alpha == 1, falls back to standard Shannon-based score.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        alpha : float > 0, != 1
            Rényi order.

        Returns
        -------
        score : float
        """
        if abs(alpha - 1.0) < 1e-9:
            return self.standard(K)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        eigvals = self._get_eigvals(K)
        # Remove zeros
        eigvals = eigvals[eigvals > _EPSILON]
        if len(eigvals) == 0:
            return 0.0

        log_sum = np.log(np.sum(eigvals ** alpha) + _EPSILON)
        H_alpha = log_sum / (1.0 - alpha)
        return math.exp(H_alpha)

    # ---- Truncated ---------------------------------------------------

    def truncated(self, K: np.ndarray, *, top_k: int = 10) -> float:
        """Truncated Vendi Score using only the top-k eigenvalues.

        After selecting the top-k eigenvalues, they are re-normalised
        to sum to 1 and the standard entropy is computed.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        top_k : int
            Number of eigenvalues to keep.

        Returns
        -------
        score : float
        """
        eigvals = self._get_eigvals(K)
        top_k = min(top_k, len(eigvals))
        eigvals_trunc = eigvals[:top_k]
        total = eigvals_trunc.sum()
        if total < _EPSILON:
            return 0.0
        eigvals_trunc = eigvals_trunc / total
        H = safe_entropy(eigvals_trunc)
        return math.exp(H)

    # ---- Regularized -------------------------------------------------

    def regularized(
        self,
        K: np.ndarray,
        *,
        ridge: float = 1e-4,
    ) -> float:
        """Regularized Vendi Score with ridge added for numerical stability.

        K_reg = K + ridge * I

        Parameters
        ----------
        K : (n, n) kernel matrix.
        ridge : float >= 0
            Regularization strength.

        Returns
        -------
        score : float
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        K_reg = K + ridge * np.eye(n, dtype=np.float64)
        eigvals = eigenvalue_decomposition(
            K_reg, floor=self.eigenvalue_floor, normalise=True
        )
        H = safe_entropy(eigvals)
        return math.exp(H)

    # ---- Weighted (quality-diversity) --------------------------------

    def weighted(
        self,
        K: np.ndarray,
        quality_scores: np.ndarray,
    ) -> float:
        """Weighted Vendi Score incorporating per-item quality.

        The kernel is reweighted as:
            K_w[i,j] = sqrt(q_i) * K[i,j] * sqrt(q_j)

        so that higher-quality items contribute more to the score.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        quality_scores : (n,) array of non-negative quality weights.

        Returns
        -------
        score : float
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        q = np.asarray(quality_scores, dtype=np.float64).ravel()
        if len(q) != n:
            raise ValueError(
                f"quality_scores length {len(q)} != kernel size {n}"
            )
        q = np.maximum(q, 0.0)
        total_q = q.sum()
        if total_q < _EPSILON:
            return 0.0
        q = q / total_q  # normalise so weights sum to 1

        sqrt_q = np.sqrt(q)
        K_w = K * np.outer(sqrt_q, sqrt_q)

        eigvals = eigenvalue_decomposition(
            K_w, floor=self.eigenvalue_floor, normalise=True
        )
        H = safe_entropy(eigvals)
        return math.exp(H)

    # ---- Conditional -------------------------------------------------

    def conditional(
        self,
        K: np.ndarray,
        group_labels: Sequence[int],
    ) -> Dict[str, Any]:
        """Conditional Vendi Score: diversity within and across groups.

        Computes:
        - Within-group VS for each group
        - Between-group VS on cluster centroids (approximated via
          average kernel values)
        - Overall VS

        Parameters
        ----------
        K : (n, n) kernel matrix.
        group_labels : sequence of int, length n.

        Returns
        -------
        result : dict with keys "overall", "within", "between",
                 "group_sizes", "group_labels_unique".
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        labels = np.asarray(group_labels).ravel()
        if len(labels) != n:
            raise ValueError(
                f"group_labels length {len(labels)} != kernel size {n}"
            )

        unique_labels = np.unique(labels)
        num_groups = len(unique_labels)

        # Overall score
        overall = self.standard(K)

        # Within-group scores
        within: Dict[int, float] = {}
        group_sizes: Dict[int, int] = {}
        for g in unique_labels:
            idx = np.where(labels == g)[0]
            group_sizes[int(g)] = len(idx)
            if len(idx) <= 1:
                within[int(g)] = 1.0
                continue
            K_g = K[np.ix_(idx, idx)]
            within[int(g)] = self.standard(K_g)

        # Between-group kernel: mean kernel value between groups
        K_between = np.zeros((num_groups, num_groups), dtype=np.float64)
        label_to_idx = {int(g): i for i, g in enumerate(unique_labels)}
        for g1 in unique_labels:
            for g2 in unique_labels:
                idx1 = np.where(labels == g1)[0]
                idx2 = np.where(labels == g2)[0]
                K_between[label_to_idx[int(g1)], label_to_idx[int(g2)]] = (
                    K[np.ix_(idx1, idx2)].mean()
                )

        between = self.standard(K_between)

        return {
            "overall": overall,
            "within": within,
            "between": between,
            "group_sizes": group_sizes,
            "group_labels_unique": [int(g) for g in unique_labels],
        }


# ===================================================================
# VendiScoreAnalyzer
# ===================================================================

@dataclass
class EigenspectrumAnalysis:
    """Container for eigenspectrum analysis results."""

    eigenvalues: np.ndarray
    cumulative_energy: np.ndarray
    effective_rank: float
    condition_number: float
    spectrum_decay_rate: float
    top_k_energy: Dict[int, float] = field(default_factory=dict)
    vendi_score: float = 0.0
    matrix_entropy: float = 0.0


@dataclass
class KernelDiagnostics:
    """Container for kernel quality diagnostics."""

    condition_number: float
    trace: float
    frobenius_norm: float
    spectral_norm: float
    rank: int
    numerical_rank: int
    min_eigenvalue: float
    max_eigenvalue: float
    is_positive_definite: bool
    diagonal_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class BandwidthSensitivity:
    """Results of bandwidth sensitivity analysis."""

    bandwidths: np.ndarray
    vendi_scores: np.ndarray
    entropies: np.ndarray
    best_bandwidth: float = 0.0
    score_range: Tuple[float, float] = (0.0, 0.0)
    score_std: float = 0.0


@dataclass
class ItemContribution:
    """Per-item contribution to the Vendi Score."""

    item_index: int
    leave_one_out_score: float
    marginal_contribution: float
    leverage_score: float = 0.0


class VendiScoreAnalyzer:
    """Analysis and diagnostic tools for the Vendi Score.

    Provides eigenspectrum analysis, kernel diagnostics, bandwidth
    sensitivity studies, and per-item contribution decomposition.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self.eigenvalue_floor = eigenvalue_floor
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)

    # ---- Eigenspectrum -----------------------------------------------

    def eigenspectrum_analysis(
        self,
        K: np.ndarray,
        *,
        top_k_values: Optional[List[int]] = None,
    ) -> EigenspectrumAnalysis:
        """Full eigenspectrum analysis of a kernel matrix.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        top_k_values : list of int, optional
            Compute cumulative energy for these k values.
            Defaults to [1, 5, 10, 20, 50].

        Returns
        -------
        EigenspectrumAnalysis
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        if top_k_values is None:
            top_k_values = [k for k in [1, 5, 10, 20, 50] if k <= n]

        # Raw eigenvalues (not normalised)
        eigvals_raw = eigenvalue_decomposition(
            K, floor=self.eigenvalue_floor, normalise=False
        )
        # Normalised eigenvalues
        eigvals_norm = eigenvalue_decomposition(
            K, floor=self.eigenvalue_floor, normalise=True
        )

        cumulative = np.cumsum(eigvals_norm)

        # Effective rank: exp(entropy)
        ent = safe_entropy(eigvals_norm)
        effective_rank = math.exp(ent)

        # Condition number
        nonzero = eigvals_raw[eigvals_raw > _EPSILON]
        if len(nonzero) >= 2:
            cond = float(nonzero[0] / nonzero[-1])
        else:
            cond = 1.0

        # Spectrum decay rate: fit log(lambda_k) ~ -rate * k
        positive = eigvals_norm[eigvals_norm > _EPSILON]
        if len(positive) >= 2:
            log_eig = np.log(positive)
            ks = np.arange(1, len(positive) + 1, dtype=np.float64)
            # Simple linear regression
            slope, _ = np.polyfit(ks, log_eig, 1)
            decay_rate = float(-slope)
        else:
            decay_rate = 0.0

        # Top-k energy
        top_k_energy: Dict[int, float] = {}
        for k in top_k_values:
            if k <= len(cumulative):
                top_k_energy[k] = float(cumulative[k - 1])

        vs = self._computer.compute(K)

        return EigenspectrumAnalysis(
            eigenvalues=eigvals_norm,
            cumulative_energy=cumulative,
            effective_rank=effective_rank,
            condition_number=cond,
            spectrum_decay_rate=decay_rate,
            top_k_energy=top_k_energy,
            vendi_score=vs,
            matrix_entropy=ent,
        )

    # ---- Effective rank ----------------------------------------------

    def effective_rank(self, K: np.ndarray) -> float:
        """Compute the effective rank of a kernel matrix.

        Defined as exp(H(lambda)) where lambda are the normalised
        eigenvalues.  This is exactly the Vendi Score.

        Parameters
        ----------
        K : (n, n) kernel matrix.

        Returns
        -------
        rank : float
        """
        return self._computer.compute(K)

    # ---- Kernel diagnostics ------------------------------------------

    def kernel_diagnostics(self, K: np.ndarray) -> KernelDiagnostics:
        """Compute diagnostic statistics for a kernel matrix.

        Parameters
        ----------
        K : (n, n) kernel matrix.

        Returns
        -------
        KernelDiagnostics
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        if n == 0:
            return KernelDiagnostics(
                condition_number=0.0,
                trace=0.0,
                frobenius_norm=0.0,
                spectral_norm=0.0,
                rank=0,
                numerical_rank=0,
                min_eigenvalue=0.0,
                max_eigenvalue=0.0,
                is_positive_definite=True,
                diagonal_stats={},
            )

        eigvals = eigenvalue_decomposition(
            K, floor=self.eigenvalue_floor, normalise=False
        )

        nonzero = eigvals[eigvals > _EPSILON]
        cond = float(nonzero[0] / nonzero[-1]) if len(nonzero) >= 2 else 1.0

        diag = np.diag(K)

        # Numerical rank: number of eigenvalues above tolerance
        tol = max(n, 1) * float(eigvals[0]) * np.finfo(np.float64).eps
        numerical_rank = int(np.sum(eigvals > tol))

        is_pd = bool(np.all(eigvals > -_EPSILON) and eigvals[-1] > _EPSILON)

        return KernelDiagnostics(
            condition_number=cond,
            trace=float(np.trace(K)),
            frobenius_norm=float(np.linalg.norm(K, "fro")),
            spectral_norm=float(eigvals[0]) if len(eigvals) > 0 else 0.0,
            rank=int(np.sum(eigvals > _EPSILON)),
            numerical_rank=numerical_rank,
            min_eigenvalue=float(eigvals[-1]) if len(eigvals) > 0 else 0.0,
            max_eigenvalue=float(eigvals[0]) if len(eigvals) > 0 else 0.0,
            is_positive_definite=is_pd,
            diagonal_stats={
                "mean": float(np.mean(diag)),
                "std": float(np.std(diag)),
                "min": float(np.min(diag)),
                "max": float(np.max(diag)),
            },
        )

    # ---- Kernel comparison -------------------------------------------

    def compare_kernels(
        self,
        X: np.ndarray,
        kernel_types: Optional[List[str]] = None,
        *,
        normalize: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare Vendi Scores under different kernel choices.

        Parameters
        ----------
        X : (n, d) embeddings.
        kernel_types : list of str, optional
            Defaults to ["rbf", "linear", "cosine", "polynomial", "laplacian"].
        normalize : bool

        Returns
        -------
        results : dict mapping kernel_type -> {score, entropy, eff_rank, ...}
        """
        if kernel_types is None:
            kernel_types = ["rbf", "linear", "cosine", "polynomial", "laplacian"]

        builder = KernelBuilder()
        results: Dict[str, Dict[str, Any]] = {}

        for kt in kernel_types:
            try:
                K = builder.build(X, kernel_type=kt)
                if normalize:
                    K = kernel_normalize(K)
                vs = self._computer.compute(K)
                ent = self._computer.matrix_entropy(K)
                diag = self.kernel_diagnostics(K)
                results[kt] = {
                    "vendi_score": vs,
                    "entropy": ent,
                    "condition_number": diag.condition_number,
                    "rank": diag.rank,
                    "is_positive_definite": diag.is_positive_definite,
                }
            except Exception as exc:
                results[kt] = {"error": str(exc)}

        return results

    # ---- Bandwidth sensitivity ---------------------------------------

    def bandwidth_sensitivity(
        self,
        X: np.ndarray,
        *,
        num_bandwidths: int = 20,
        low_factor: float = 0.1,
        high_factor: float = 10.0,
    ) -> BandwidthSensitivity:
        """Study how the Vendi Score varies with the RBF bandwidth.

        Parameters
        ----------
        X : (n, d) embeddings.
        num_bandwidths : int
        low_factor, high_factor : float

        Returns
        -------
        BandwidthSensitivity
        """
        X = np.asarray(X, dtype=np.float64)
        sigmas = KernelBuilder.bandwidth_grid(
            X,
            num=num_bandwidths,
            low_factor=low_factor,
            high_factor=high_factor,
        )

        scores = np.zeros(num_bandwidths, dtype=np.float64)
        entropies = np.zeros(num_bandwidths, dtype=np.float64)

        for i, sigma in enumerate(sigmas):
            K = KernelBuilder.rbf(X, sigma=sigma)
            K = kernel_normalize(K)
            scores[i] = self._computer.compute(K)
            entropies[i] = self._computer.matrix_entropy(K)

        best_idx = int(np.argmax(scores))
        return BandwidthSensitivity(
            bandwidths=sigmas,
            vendi_scores=scores,
            entropies=entropies,
            best_bandwidth=float(sigmas[best_idx]),
            score_range=(float(scores.min()), float(scores.max())),
            score_std=float(scores.std()),
        )

    # ---- Per-item contribution (leave-one-out) -----------------------

    def item_contributions(
        self,
        K: np.ndarray,
    ) -> List[ItemContribution]:
        """Decompose the Vendi Score by contribution of each item.

        Uses leave-one-out: for each item i, compute VS on K without
        row/column i.  Marginal contribution = VS_full - VS_{-i}.

        Also computes leverage scores (diagonal of the hat matrix
        K (K)^{-1}).

        Parameters
        ----------
        K : (n, n) kernel matrix.

        Returns
        -------
        contributions : list of ItemContribution, sorted by marginal
                        contribution descending.
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        full_score = self._computer.compute(K)

        # Leverage scores via eigendecomposition
        K_sym = 0.5 * (K + K.T)
        eigvals_raw, eigvecs = eigh(K_sym)
        eigvals_raw = np.maximum(eigvals_raw, 0.0)
        total_eig = eigvals_raw.sum()
        if total_eig > _EPSILON:
            normed = eigvals_raw / total_eig
        else:
            normed = np.ones(n) / n
        # Leverage = diagonal of normalised projection
        leverage = np.sum(eigvecs ** 2 * normed[np.newaxis, :], axis=1)

        contributions: List[ItemContribution] = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            K_sub = K[np.ix_(mask, mask)]
            loo_score = self._computer.compute(K_sub) if n > 2 else 1.0
            contributions.append(
                ItemContribution(
                    item_index=i,
                    leave_one_out_score=loo_score,
                    marginal_contribution=full_score - loo_score,
                    leverage_score=float(leverage[i]),
                )
            )

        contributions.sort(key=lambda c: c.marginal_contribution, reverse=True)
        return contributions

    # ---- Eigenspectrum visualisation data -----------------------------

    def eigenspectrum_plot_data(
        self,
        K: np.ndarray,
    ) -> Dict[str, Any]:
        """Return data suitable for plotting the eigenspectrum.

        Parameters
        ----------
        K : (n, n) kernel matrix.

        Returns
        -------
        data : dict with "indices", "eigenvalues", "cumulative_energy",
               "log_eigenvalues".
        """
        K = _validate_kernel_matrix(K)
        eigvals = eigenvalue_decomposition(
            K, floor=self.eigenvalue_floor, normalise=True
        )
        positive = eigvals[eigvals > _EPSILON]
        return {
            "indices": list(range(1, len(positive) + 1)),
            "eigenvalues": positive.tolist(),
            "cumulative_energy": np.cumsum(positive).tolist(),
            "log_eigenvalues": np.log(positive).tolist(),
            "vendi_score": math.exp(safe_entropy(eigvals)),
        }


# ===================================================================
# OnlineVendiScore
# ===================================================================

class OnlineVendiScore:
    """Incremental Vendi Score computation.

    Maintains a growing kernel matrix and recomputes the score
    efficiently as new items are added.  For moderate sizes (n < 1000),
    full eigen-decomposition is used.  For larger sets, a Nyström
    approximation is available.

    Parameters
    ----------
    kernel_func : callable(x_i, x_j) -> float
        Pairwise kernel function.
    eigenvalue_floor : float
        Floor for eigenvalue clipping.
    """

    def __init__(
        self,
        kernel_func: Callable[[Any, Any], float],
        *,
        eigenvalue_floor: float = _EIGENVALUE_FLOOR,
    ) -> None:
        self.kernel_func = kernel_func
        self.eigenvalue_floor = eigenvalue_floor
        self._items: List[Any] = []
        self._K: Optional[np.ndarray] = None
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)
        self._scores_history: List[float] = []

    @property
    def n(self) -> int:
        """Current number of items."""
        return len(self._items)

    @property
    def current_score(self) -> float:
        """Most recently computed Vendi Score."""
        if not self._scores_history:
            return 0.0
        return self._scores_history[-1]

    @property
    def score_history(self) -> List[float]:
        """History of Vendi Scores after each addition."""
        return list(self._scores_history)

    def add(self, item: Any) -> float:
        """Add a single item and return the updated Vendi Score.

        Parameters
        ----------
        item : Any
            Item compatible with the kernel function.

        Returns
        -------
        score : float
            Updated Vendi Score.
        """
        self._items.append(item)
        n = len(self._items)

        if n == 1:
            self._K = np.array([[self.kernel_func(item, item)]], dtype=np.float64)
            self._scores_history.append(1.0)
            return 1.0

        # Rank-one update: expand kernel matrix
        new_col = np.array(
            [self.kernel_func(self._items[j], item) for j in range(n - 1)],
            dtype=np.float64,
        )
        k_nn = self.kernel_func(item, item)

        K_new = np.zeros((n, n), dtype=np.float64)
        K_new[: n - 1, : n - 1] = self._K
        K_new[: n - 1, n - 1] = new_col
        K_new[n - 1, : n - 1] = new_col
        K_new[n - 1, n - 1] = k_nn
        self._K = K_new

        score = self._computer.compute(K_new)
        self._scores_history.append(score)
        return score

    def add_batch(self, items: Sequence[Any]) -> List[float]:
        """Add multiple items, returning the score after each addition.

        Parameters
        ----------
        items : sequence
            Items to add.

        Returns
        -------
        scores : list of float
        """
        return [self.add(item) for item in items]

    def remove_last(self) -> float:
        """Remove the most recently added item and return updated score.

        Returns
        -------
        score : float
        """
        if not self._items:
            return 0.0

        self._items.pop()
        self._scores_history.pop()
        n = len(self._items)

        if n == 0:
            self._K = None
            return 0.0

        self._K = self._K[:n, :n].copy()
        if not self._scores_history:
            score = self._computer.compute(self._K)
            self._scores_history.append(score)
        return self._scores_history[-1]

    def get_kernel_matrix(self) -> Optional[np.ndarray]:
        """Return a copy of the current kernel matrix."""
        if self._K is None:
            return None
        return self._K.copy()

    def reset(self) -> None:
        """Clear all items and reset state."""
        self._items.clear()
        self._K = None
        self._scores_history.clear()


# ===================================================================
# ApproximateVendiScore  (for large n)
# ===================================================================

class ApproximateVendiScore:
    """Approximate Vendi Score computation for large datasets.

    Uses either random Nyström approximation or random Fourier features
    to avoid O(n^3) eigendecomposition.

    Parameters
    ----------
    eigenvalue_floor : float
        Floor for eigenvalue clipping.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self.eigenvalue_floor = eigenvalue_floor

    def nystrom(
        self,
        K: np.ndarray,
        *,
        num_landmarks: int = 100,
        seed: int = 42,
    ) -> float:
        """Nyström approximation of the Vendi Score.

        Approximate the full kernel matrix using a subset of landmark
        points, then compute the score from the approximate spectrum.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        num_landmarks : int
            Number of landmark (inducing) points.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        score : float
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        m = min(num_landmarks, n)

        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=m, replace=False)
        idx.sort()

        K_mm = K[np.ix_(idx, idx)]  # (m, m)
        K_nm = K[:, idx]            # (n, m)

        # Eigendecomposition of K_mm
        eigvals_mm, eigvecs_mm = eigh(K_mm)
        eigvals_mm = np.maximum(eigvals_mm, _EPSILON)

        # Nyström approximation eigenvalues: (n/m) * eigvals of K_nm^T K_nm / K_mm
        # Using the standard Nyström formula:
        # K ≈ K_nm K_mm^{-1} K_nm^T
        # The eigenvalues of this approximation can be found via:
        #   Lambda_mm^{-1/2} U_mm^T K_nm^T K_nm U_mm Lambda_mm^{-1/2} / m
        inv_sqrt = 1.0 / np.sqrt(eigvals_mm)
        # Transform: (m, n) @ (n, m) -> (m, m)
        B = eigvecs_mm.T @ K_nm.T  # (m, n)
        B = np.diag(inv_sqrt) @ B   # scale rows
        # Gram matrix of transformed features
        G = (B @ B.T) / m  # (m, m)

        eigvals_approx = np.linalg.eigvalsh(G)
        eigvals_approx = np.maximum(eigvals_approx, 0.0)
        # Scale to approximate the full spectrum
        eigvals_approx = eigvals_approx * (float(n) / m)
        eigvals_approx = np.sort(eigvals_approx)[::-1]

        total = eigvals_approx.sum()
        if total < _EPSILON:
            return 0.0
        eigvals_approx = eigvals_approx / total

        H = safe_entropy(eigvals_approx)
        return math.exp(H)

    def random_features(
        self,
        X: np.ndarray,
        *,
        sigma: Optional[float] = None,
        num_features: int = 500,
        seed: int = 42,
    ) -> float:
        """Vendi Score via Random Fourier Features (RFF) for RBF kernel.

        Approximates the RBF kernel K using D random Fourier features,
        then computes the score from the resulting (n x D) feature matrix
        using the SVD instead of eigendecomposition of (n x n) matrix.

        Parameters
        ----------
        X : (n, d) embeddings.
        sigma : float or None
            RBF bandwidth (median heuristic if None).
        num_features : int
            Number of random features D.
        seed : int

        Returns
        -------
        score : float
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape

        if sigma is None:
            sigma = median_bandwidth_heuristic(X)

        rng = np.random.RandomState(seed)
        # Sample random frequencies
        W = rng.randn(d, num_features) / sigma  # (d, D)
        b = rng.uniform(0, 2 * np.pi, size=num_features)  # (D,)

        # Random features: z(x) = sqrt(2/D) cos(Wx + b)
        Z = np.sqrt(2.0 / num_features) * np.cos(X @ W + b[np.newaxis, :])

        # The approximate kernel is K ≈ Z Z^T
        # Eigenvalues of Z Z^T are the same as singular values^2 of Z
        # divided by n for normalisation.
        # Use SVD of Z which is O(n D^2) instead of O(n^3).
        _, s, _ = np.linalg.svd(Z, full_matrices=False)
        eigvals = s ** 2  # eigenvalues of Z Z^T

        total = eigvals.sum()
        if total < _EPSILON:
            return 0.0
        eigvals = eigvals / total

        H = safe_entropy(eigvals)
        return math.exp(H)

    def block_diagonal(
        self,
        K: np.ndarray,
        *,
        block_size: int = 200,
    ) -> float:
        """Block-diagonal approximation for very large kernel matrices.

        Splits the kernel matrix into blocks along the diagonal and
        averages the Vendi Scores, weighted by block size.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        block_size : int

        Returns
        -------
        score : float
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        if n == 0:
            return 0.0

        computer = VendiScoreComputer(eigenvalue_floor=self.eigenvalue_floor)
        num_blocks = max(1, (n + block_size - 1) // block_size)
        block_scores: List[float] = []
        block_sizes: List[int] = []

        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            K_block = K[start:end, start:end]
            bsize = end - start
            block_scores.append(computer.compute(K_block))
            block_sizes.append(bsize)

        # Weighted average of log-scores (geometric mean)
        total = sum(block_sizes)
        log_score = sum(
            (bs / total) * math.log(max(s, _EPSILON))
            for bs, s in zip(block_sizes, block_scores)
        )
        return math.exp(log_score)


# ===================================================================
# Batch & Comparison Utilities
# ===================================================================

@dataclass
class VendiComparison:
    """Comparison of Vendi Scores across multiple sets."""

    set_names: List[str]
    scores: Dict[str, float]
    entropies: Dict[str, float]
    effective_ranks: Dict[str, float]
    set_sizes: Dict[str, int]
    relative_diversities: Dict[str, float] = field(default_factory=dict)


class VendiBatchComputer:
    """Batch computation and comparison of Vendi Scores.

    Useful for comparing diversity across multiple output sets (e.g.,
    different decoding strategies).

    Parameters
    ----------
    kernel_type : str
        Default kernel type for all computations.
    normalize_kernel : bool
        Whether to normalise kernels by default.
    eigenvalue_floor : float
    """

    def __init__(
        self,
        kernel_type: str = "cosine",
        *,
        normalize_kernel: bool = True,
        eigenvalue_floor: float = _EIGENVALUE_FLOOR,
    ) -> None:
        self.kernel_type = kernel_type
        self.normalize_kernel = normalize_kernel
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)
        self._builder = KernelBuilder()

    def compute_batch(
        self,
        named_embeddings: Dict[str, np.ndarray],
        **kernel_kwargs: Any,
    ) -> VendiComparison:
        """Compute and compare Vendi Scores for multiple embedding sets.

        Parameters
        ----------
        named_embeddings : dict mapping name -> (n_i, d) embeddings.
        **kernel_kwargs : forwarded to the kernel builder.

        Returns
        -------
        VendiComparison
        """
        scores: Dict[str, float] = {}
        entropies: Dict[str, float] = {}
        effective_ranks: Dict[str, float] = {}
        set_sizes: Dict[str, int] = {}

        for name, emb in named_embeddings.items():
            emb = np.asarray(emb, dtype=np.float64)
            if emb.ndim == 1:
                emb = emb.reshape(-1, 1)
            K = self._builder.build(
                emb, kernel_type=self.kernel_type, **kernel_kwargs
            )
            if self.normalize_kernel:
                K = kernel_normalize(K)
            scores[name] = self._computer.compute(K)
            entropies[name] = self._computer.matrix_entropy(K)
            effective_ranks[name] = scores[name]  # VS == effective rank
            set_sizes[name] = emb.shape[0]

        # Relative diversities (normalised to max)
        max_score = max(scores.values()) if scores else 1.0
        relative = {
            name: s / max(max_score, _EPSILON) for name, s in scores.items()
        }

        return VendiComparison(
            set_names=list(named_embeddings.keys()),
            scores=scores,
            entropies=entropies,
            effective_ranks=effective_ranks,
            set_sizes=set_sizes,
            relative_diversities=relative,
        )

    def compute_batch_from_texts(
        self,
        named_texts: Dict[str, List[str]],
        *,
        similarity_func: Optional[Callable[[str, str], float]] = None,
        ngram_size: int = 3,
    ) -> VendiComparison:
        """Compute and compare Vendi Scores for multiple text sets.

        Parameters
        ----------
        named_texts : dict mapping name -> list of strings.
        similarity_func : optional pairwise similarity function.
        ngram_size : int

        Returns
        -------
        VendiComparison
        """
        scores: Dict[str, float] = {}
        entropies: Dict[str, float] = {}
        effective_ranks: Dict[str, float] = {}
        set_sizes: Dict[str, int] = {}

        for name, texts in named_texts.items():
            vs = self._computer.compute_from_texts(
                texts,
                similarity_func=similarity_func,
                ngram_size=ngram_size,
            )
            scores[name] = vs
            entropies[name] = math.log(max(vs, _EPSILON))
            effective_ranks[name] = vs
            set_sizes[name] = len(texts)

        max_score = max(scores.values()) if scores else 1.0
        relative = {
            name: s / max(max_score, _EPSILON) for name, s in scores.items()
        }

        return VendiComparison(
            set_names=list(named_texts.keys()),
            scores=scores,
            entropies=entropies,
            effective_ranks=effective_ranks,
            set_sizes=set_sizes,
            relative_diversities=relative,
        )


# ===================================================================
# Cross-set Vendi Score
# ===================================================================

class CrossSetVendiScore:
    """Vendi Score comparisons across and between sets.

    Enables measuring how much diversity one set adds to another,
    or how diverse the union of two sets is compared to each alone.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)
        self._builder = KernelBuilder()

    def union_score(
        self,
        K_a: np.ndarray,
        K_b: np.ndarray,
        K_ab: np.ndarray,
    ) -> float:
        """Vendi Score of the union of two sets.

        Parameters
        ----------
        K_a : (n_a, n_a) kernel for set A.
        K_b : (n_b, n_b) kernel for set B.
        K_ab : (n_a, n_b) cross-kernel between A and B.

        Returns
        -------
        score : float
        """
        n_a = K_a.shape[0]
        n_b = K_b.shape[0]
        n = n_a + n_b

        K = np.zeros((n, n), dtype=np.float64)
        K[:n_a, :n_a] = K_a
        K[n_a:, n_a:] = K_b
        K[:n_a, n_a:] = K_ab
        K[n_a:, :n_a] = K_ab.T

        return self._computer.compute(K)

    def marginal_diversity(
        self,
        K_a: np.ndarray,
        K_b: np.ndarray,
        K_ab: np.ndarray,
    ) -> Dict[str, float]:
        """Compute marginal diversity: how much B adds to A and vice versa.

        Returns
        -------
        dict with keys:
            "vs_a": VS of A alone
            "vs_b": VS of B alone
            "vs_union": VS of A ∪ B
            "marginal_b_given_a": VS_union - VS_a (diversity B adds to A)
            "marginal_a_given_b": VS_union - VS_b (diversity A adds to B)
            "synergy": VS_union - VS_a - VS_b + min(VS_a, VS_b)
        """
        vs_a = self._computer.compute(K_a)
        vs_b = self._computer.compute(K_b)
        vs_union = self.union_score(K_a, K_b, K_ab)

        return {
            "vs_a": vs_a,
            "vs_b": vs_b,
            "vs_union": vs_union,
            "marginal_b_given_a": vs_union - vs_a,
            "marginal_a_given_b": vs_union - vs_b,
            "synergy": vs_union - vs_a - vs_b + min(vs_a, vs_b),
        }

    def redundancy_ratio(
        self,
        K_a: np.ndarray,
        K_b: np.ndarray,
        K_ab: np.ndarray,
    ) -> float:
        """Redundancy ratio: 1 - (VS_union / (VS_a + VS_b)).

        A value near 0 means the sets are complementary (low redundancy).
        A value near 1 means the sets are highly redundant.

        Returns
        -------
        ratio : float in [0, 1]
        """
        vs_a = self._computer.compute(K_a)
        vs_b = self._computer.compute(K_b)
        vs_union = self.union_score(K_a, K_b, K_ab)

        denom = vs_a + vs_b
        if denom < _EPSILON:
            return 0.0
        ratio = 1.0 - vs_union / denom
        return max(0.0, min(1.0, ratio))


# ===================================================================
# Statistical Testing
# ===================================================================

class VendiScoreStatistics:
    """Bootstrap and permutation tests for Vendi Score significance.

    Useful for assessing whether observed diversity differences are
    statistically meaningful.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)

    def bootstrap_confidence_interval(
        self,
        K: np.ndarray,
        *,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Bootstrap confidence interval for the Vendi Score.

        Resamples rows/columns of the kernel matrix with replacement.

        Parameters
        ----------
        K : (n, n) kernel matrix.
        n_bootstrap : int
        confidence : float in (0, 1)
        seed : int

        Returns
        -------
        dict with "mean", "std", "lower", "upper", "observed".
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        rng = np.random.RandomState(seed)

        observed = self._computer.compute(K)
        bootstrap_scores = np.zeros(n_bootstrap, dtype=np.float64)

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            K_boot = K[np.ix_(idx, idx)]
            bootstrap_scores[b] = self._computer.compute(K_boot)

        alpha = 1.0 - confidence
        lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

        return {
            "observed": observed,
            "mean": float(np.mean(bootstrap_scores)),
            "std": float(np.std(bootstrap_scores)),
            "lower": lower,
            "upper": upper,
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
        }

    def permutation_test(
        self,
        K_a: np.ndarray,
        K_b: np.ndarray,
        *,
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Two-sample permutation test for Vendi Score difference.

        Tests H0: VS(A) = VS(B) by permuting the combined kernel.

        Parameters
        ----------
        K_a : (n_a, n_a) kernel matrix for set A.
        K_b : (n_b, n_b) kernel matrix for set B.
        n_permutations : int
        seed : int

        Returns
        -------
        dict with "observed_diff", "p_value", "permuted_diffs", etc.
        """
        K_a = _validate_kernel_matrix(K_a)
        K_b = _validate_kernel_matrix(K_b)
        n_a = K_a.shape[0]
        n_b = K_b.shape[0]
        n = n_a + n_b

        vs_a = self._computer.compute(K_a)
        vs_b = self._computer.compute(K_b)
        observed_diff = vs_a - vs_b

        # Build combined block-diagonal kernel (no cross terms)
        K_combined = np.zeros((n, n), dtype=np.float64)
        K_combined[:n_a, :n_a] = K_a
        K_combined[n_a:, n_a:] = K_b

        rng = np.random.RandomState(seed)
        perm_diffs = np.zeros(n_permutations, dtype=np.float64)

        for p in range(n_permutations):
            perm = rng.permutation(n)
            K_perm = K_combined[np.ix_(perm, perm)]
            vs_perm_a = self._computer.compute(K_perm[:n_a, :n_a])
            vs_perm_b = self._computer.compute(K_perm[n_a:, n_a:])
            perm_diffs[p] = vs_perm_a - vs_perm_b

        p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))

        return {
            "observed_diff": observed_diff,
            "vs_a": vs_a,
            "vs_b": vs_b,
            "p_value": p_value,
            "n_permutations": n_permutations,
            "permuted_diffs_mean": float(np.mean(perm_diffs)),
            "permuted_diffs_std": float(np.std(perm_diffs)),
            "significant_at_005": p_value < 0.05,
        }


# ===================================================================
# Normalised / Relative Vendi Score
# ===================================================================

class NormalisedVendiScore:
    """Normalised variants of the Vendi Score.

    Provides scores normalised by the set size n so that the result
    lies in [1/n, 1] (or equivalently [0, 1] for the ratio version).
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)

    def ratio(self, K: np.ndarray) -> float:
        """VS / n.  1.0 means every item is unique; 1/n means all identical.

        Parameters
        ----------
        K : (n, n) kernel matrix.

        Returns
        -------
        ratio : float in [1/n, 1].
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        if n <= 1:
            return 1.0
        vs = self._computer.compute(K)
        return vs / n

    def log_ratio(self, K: np.ndarray) -> float:
        """log(VS) / log(n).  Normalised entropy in [0, 1].

        Parameters
        ----------
        K : (n, n) kernel matrix.

        Returns
        -------
        ratio : float in [0, 1].
        """
        K = _validate_kernel_matrix(K)
        n = K.shape[0]
        if n <= 1:
            return 1.0
        ent = self._computer.matrix_entropy(K)
        max_ent = math.log(n)
        if max_ent < _EPSILON:
            return 1.0
        return ent / max_ent

    def relative_to_uniform(self, K: np.ndarray) -> float:
        """VS / VS_uniform where VS_uniform = n (identity kernel).

        This is equivalent to ``ratio``, provided for semantic clarity.
        """
        return self.ratio(K)

    def deficit(self, K: np.ndarray) -> float:
        """Diversity deficit: 1 - VS/n.

        0.0 means maximum diversity, 1.0 means all items identical.
        """
        return 1.0 - self.ratio(K)


# ===================================================================
# Feature-space Vendi Score
# ===================================================================

class FeatureSpaceVendiScore:
    """Vendi Score computed from feature representations.

    When the number of features d is much smaller than n, it is more
    efficient to work with the d×d covariance matrix than the n×n kernel.
    This class exploits that dual relationship.
    """

    def __init__(self, eigenvalue_floor: float = _EIGENVALUE_FLOOR) -> None:
        self.eigenvalue_floor = eigenvalue_floor
        self._computer = VendiScoreComputer(eigenvalue_floor=eigenvalue_floor)

    def compute_dual(self, X: np.ndarray) -> float:
        """Vendi Score via the dual (feature-space) formulation.

        When d << n, compute eigenvalues of (X^T X / n) instead of
        (X X^T / n), since they share the same non-zero eigenvalues.

        Parameters
        ----------
        X : (n, d) feature matrix.

        Returns
        -------
        score : float
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape

        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        # Choose the smaller formulation
        if d < n:
            # Dual: d x d matrix
            C = (X.T @ X) / n  # (d, d)
            eigvals = np.linalg.eigvalsh(C)
        else:
            # Primal: n x n Gram matrix
            G = (X @ X.T) / n  # (n, n)
            eigvals = np.linalg.eigvalsh(G)

        eigvals = np.maximum(eigvals, 0.0)
        total = eigvals.sum()
        if total < _EPSILON:
            return 0.0
        eigvals = eigvals / total
        eigvals = np.sort(eigvals)[::-1]

        H = safe_entropy(eigvals)
        return math.exp(H)

    def compute_from_normalised_features(self, X: np.ndarray) -> float:
        """Vendi Score from L2-normalised features (cosine kernel).

        Parameters
        ----------
        X : (n, d) feature matrix (rows need not be normalised).

        Returns
        -------
        score : float
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, _EPSILON)
        X_normed = X / norms
        K = X_normed @ X_normed.T
        return self._computer.compute(K)


# ===================================================================
# Convenience API
# ===================================================================

def vendi_score(K: np.ndarray) -> float:
    """Compute the Vendi Score from a kernel matrix.

    Convenience function wrapping ``VendiScoreComputer.compute``.

    Parameters
    ----------
    K : (n, n) symmetric positive semi-definite kernel matrix.

    Returns
    -------
    score : float
        The effective number of distinct items.
    """
    return VendiScoreComputer().compute(K)


def vendi_score_from_embeddings(
    embeddings: np.ndarray,
    kernel_type: str = "cosine",
    **kwargs: Any,
) -> float:
    """Compute Vendi Score directly from embedding vectors.

    Parameters
    ----------
    embeddings : (n, d) array
    kernel_type : str
    **kwargs : forwarded to kernel builder.

    Returns
    -------
    score : float
    """
    return VendiScoreComputer().compute_from_embeddings(
        embeddings, kernel_type=kernel_type, **kwargs
    )


def vendi_score_from_texts(
    texts: Sequence[str],
    similarity_func: Optional[Callable[[str, str], float]] = None,
    **kwargs: Any,
) -> float:
    """Compute Vendi Score from text strings.

    Parameters
    ----------
    texts : sequence of str
    similarity_func : optional pairwise similarity.
    **kwargs : forwarded to compute_from_texts.

    Returns
    -------
    score : float
    """
    return VendiScoreComputer().compute_from_texts(
        texts, similarity_func=similarity_func, **kwargs
    )


def matrix_entropy(K: np.ndarray) -> float:
    """Von Neumann entropy of a kernel matrix.

    Parameters
    ----------
    K : (n, n) kernel matrix.

    Returns
    -------
    entropy : float
    """
    return VendiScoreComputer().matrix_entropy(K)


def compare_diversity(
    named_embeddings: Dict[str, np.ndarray],
    kernel_type: str = "cosine",
) -> VendiComparison:
    """Compare Vendi Scores across multiple named embedding sets.

    Parameters
    ----------
    named_embeddings : dict mapping name -> (n, d) embeddings.
    kernel_type : str

    Returns
    -------
    VendiComparison
    """
    return VendiBatchComputer(kernel_type=kernel_type).compute_batch(
        named_embeddings
    )


# ===================================================================
# Module-level exports
# ===================================================================

__all__ = [
    # Helpers
    "safe_log",
    "safe_entropy",
    "eigenvalue_decomposition",
    "kernel_normalize",
    "median_bandwidth_heuristic",
    "silverman_bandwidth",
    # Core classes
    "KernelBuilder",
    "VendiScoreComputer",
    "VendiScoreVariants",
    "VendiScoreAnalyzer",
    "OnlineVendiScore",
    "ApproximateVendiScore",
    "VendiBatchComputer",
    "CrossSetVendiScore",
    "VendiScoreStatistics",
    "NormalisedVendiScore",
    "FeatureSpaceVendiScore",
    # Data classes
    "EigenspectrumAnalysis",
    "KernelDiagnostics",
    "BandwidthSensitivity",
    "ItemContribution",
    "VendiComparison",
    # Convenience functions
    "vendi_score",
    "vendi_score_from_embeddings",
    "vendi_score_from_texts",
    "matrix_entropy",
    "compare_diversity",
]
