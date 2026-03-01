"""
Determinantal Point Process (DPP) for diverse subset selection.

Implements exact DPP sampling, k-DPP sampling, greedy approximate DPP,
MAP inference, quality-diversity DPP, conditional DPP, streaming DPP,
and DPP kernel learning. All algorithms use numpy only.
"""

import numpy as np
from typing import Optional, List, Tuple, Union


def compute_kernel(items: np.ndarray, kernel: str = 'rbf',
                   gamma: Optional[float] = None,
                   degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    """Compute kernel matrix from item feature vectors.

    Args:
        items: (n, d) array of feature vectors.
        kernel: 'rbf', 'cosine', or 'polynomial'.
        gamma: RBF bandwidth (default 1/d).
        degree: Polynomial degree.
        coef0: Polynomial constant term.

    Returns:
        (n, n) positive semi-definite kernel matrix.
    """
    items = np.asarray(items, dtype=np.float64)
    n, d = items.shape

    if kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / d
        sq_dists = np.sum(items ** 2, axis=1, keepdims=True) \
                   - 2.0 * items @ items.T \
                   + np.sum(items ** 2, axis=1, keepdims=False)
        sq_dists = np.maximum(sq_dists, 0.0)
        K = np.exp(-gamma * sq_dists)

    elif kernel == 'cosine':
        norms = np.linalg.norm(items, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = items / norms
        K = normed @ normed.T
        # Clip to valid range
        np.clip(K, -1.0, 1.0, out=K)

    elif kernel == 'polynomial':
        K = (items @ items.T + coef0) ** degree

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Symmetrize to correct floating-point drift
    K = 0.5 * (K + K.T)
    return K


class DPPSampler:
    """Determinantal Point Process sampler.

    A DPP defines a probability distribution over subsets of a ground set
    where the probability of a subset S is proportional to det(L_S),
    the determinant of the principal submatrix of the L-ensemble kernel L.

    This encourages diversity because the determinant is large when the
    selected items are dissimilar (rows/columns are linearly independent).
    """

    def __init__(self):
        self.L = None
        self.n = 0
        self._eigenvalues = None
        self._eigenvectors = None
        self._is_decomposed = False

    def fit(self, kernel_matrix: np.ndarray) -> 'DPPSampler':
        """Fit the DPP with an L-ensemble kernel matrix.

        Args:
            kernel_matrix: (n, n) positive semi-definite matrix.

        Returns:
            self
        """
        L = np.asarray(kernel_matrix, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("Kernel matrix must be square.")
        # Symmetrize
        L = 0.5 * (L + L.T)
        self.L = L
        self.n = L.shape[0]
        self._is_decomposed = False
        self._eigenvalues = None
        self._eigenvectors = None
        return self

    def _ensure_decomposed(self):
        """Eigendecompose L if not already done."""
        if self._is_decomposed:
            return
        eigenvalues, eigenvectors = np.linalg.eigh(self.L)
        # Clamp small negative eigenvalues from numerical error
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._is_decomposed = True

    # ------------------------------------------------------------------
    # Exact DPP sampling (L-ensemble)
    # ------------------------------------------------------------------

    def sample(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Draw an exact sample from the DPP.

        Algorithm (Hough-Krishnapur-Peres-Virag):
          1. Eigendecompose L = V diag(λ) V^T.
          2. Include eigenvector i independently with prob λ_i/(1+λ_i).
          3. Sample from the elementary DPP defined by the selected eigenvectors.

        Returns:
            Sorted array of selected indices.
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")
        if rng is None:
            rng = np.random.RandomState()

        self._ensure_decomposed()
        lam = self._eigenvalues
        V = self._eigenvectors

        # Phase 1: select eigenvectors
        probs = lam / (1.0 + lam)
        selected_eig = rng.rand(len(lam)) < probs
        if not np.any(selected_eig):
            return np.array([], dtype=np.intp)

        V_sel = V[:, selected_eig].copy()  # (n, k)
        return self._sample_elementary_dpp(V_sel, rng)

    def _sample_elementary_dpp(self, V: np.ndarray,
                               rng: np.random.RandomState) -> np.ndarray:
        """Sample from an elementary DPP defined by column space of V.

        Args:
            V: (n, k) orthonormal-ish basis.
            rng: Random state.

        Returns:
            Sorted array of k selected indices.
        """
        n, k = V.shape
        if k == 0:
            return np.array([], dtype=np.intp)

        # Work with a copy; we'll iteratively project out dimensions
        B = V.copy()
        selected = []
        remaining = np.arange(n)

        for i in range(k):
            # Compute marginal probabilities: Pr[j selected] ∝ ||B[j,:]||^2
            probs = np.sum(B ** 2, axis=1)
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total < 1e-15:
                break
            probs /= total

            # Select item
            idx = rng.choice(len(probs), p=probs)
            selected.append(idx)

            # Project out the selected direction
            if i < k - 1:
                b_idx = B[idx, :].copy()
                norm_sq = np.dot(b_idx, b_idx)
                if norm_sq > 1e-15:
                    b_idx /= np.sqrt(norm_sq)
                    B -= np.outer(B @ b_idx, b_idx)

        return np.sort(selected)

    # ------------------------------------------------------------------
    # k-DPP sampling: sample exactly k items
    # ------------------------------------------------------------------

    def sample_k(self, k: int,
                 rng: Optional[np.random.RandomState] = None,
                 max_rejections: int = 10000) -> np.ndarray:
        """Sample exactly k items from a k-DPP.

        Uses the eigenvalue-based k-DPP algorithm:
          1. Compute elementary symmetric polynomials of eigenvalues.
          2. Select exactly k eigenvectors with the correct marginals.
          3. Sample from the elementary DPP.

        Args:
            k: Number of items to select.
            rng: Random state.
            max_rejections: Safety limit for rejection-based fallback.

        Returns:
            Sorted array of k selected indices.
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")
        if k <= 0 or k > self.n:
            raise ValueError(f"k={k} must be in [1, {self.n}].")
        if rng is None:
            rng = np.random.RandomState()

        self._ensure_decomposed()
        lam = self._eigenvalues
        V = self._eigenvectors
        n = len(lam)

        # Compute elementary symmetric polynomials e_l(λ_1,...,λ_n) for l=0..k
        E = self._compute_elem_sym_poly(lam, k)

        # Phase 1: select exactly k eigenvectors
        selected_eig = []
        remaining_k = k
        for i in range(n - 1, -1, -1):
            if remaining_k == 0:
                break
            if remaining_k > i + 1:
                # Must include this eigenvector
                selected_eig.append(i)
                remaining_k -= 1
                continue
            # Probability of including eigenvector i
            if E[remaining_k, i] < 1e-30:
                prob = 1.0
            else:
                prob = lam[i] * E[remaining_k - 1, i] / E[remaining_k, i + 1]
            prob = min(max(prob, 0.0), 1.0)
            if rng.rand() < prob:
                selected_eig.append(i)
                remaining_k -= 1

        if len(selected_eig) != k:
            # Fallback: rejection sampling
            return self._sample_k_rejection(k, rng, max_rejections)

        V_sel = V[:, selected_eig].copy()
        return self._sample_elementary_dpp(V_sel, rng)

    def _compute_elem_sym_poly(self, lam: np.ndarray, k: int) -> np.ndarray:
        """Compute elementary symmetric polynomials.

        E[l, i] = e_l(λ_1, ..., λ_i)  for l = 0..k, i = 0..n.

        Uses the recursion:
            e_l({λ_1,...,λ_i}) = e_l({λ_1,...,λ_{i-1}}) + λ_i * e_{l-1}({λ_1,...,λ_{i-1}})
        """
        n = len(lam)
        E = np.zeros((k + 1, n + 1), dtype=np.float64)
        E[0, :] = 1.0  # e_0 = 1

        for i in range(1, n + 1):
            for l in range(1, min(k, i) + 1):
                E[l, i] = E[l, i - 1] + lam[i - 1] * E[l - 1, i - 1]

        return E

    def _sample_k_rejection(self, k: int,
                            rng: np.random.RandomState,
                            max_iter: int) -> np.ndarray:
        """Fallback k-DPP sampling via rejection."""
        for _ in range(max_iter):
            s = self.sample(rng=rng)
            if len(s) == k:
                return s
        raise RuntimeError(f"k-DPP rejection sampling failed after {max_iter} tries.")

    # ------------------------------------------------------------------
    # Greedy DPP (fast approximate via Cholesky updates)
    # ------------------------------------------------------------------

    def greedy_sample(self, k: int) -> np.ndarray:
        """Greedy approximate DPP sampling via sequential Cholesky updates.

        At each step, select the item that maximizes the conditional
        log-probability (equivalently, the Schur complement / gain in
        log-determinant).

        Runs in O(n k^2) time.

        Args:
            k: Number of items to select.

        Returns:
            Array of k selected indices (in selection order).
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")
        n = self.n
        k = min(k, n)

        L = self.L
        selected = []
        # c[i, j] tracks the Cholesky-style coefficients for item i
        # We maintain: d[i] = L[i,i] - sum_{j in selected} c[i,j]^2
        # This is the conditional variance (Schur complement diagonal).
        d = np.diag(L).copy()
        c = np.zeros((n, k), dtype=np.float64)

        mask = np.ones(n, dtype=bool)

        for t in range(k):
            # Select item with largest conditional variance
            scores = np.where(mask, d, -np.inf)
            best = np.argmax(scores)
            selected.append(best)
            mask[best] = False

            if t < k - 1:
                # Update Cholesky coefficients
                sqrt_d_best = np.sqrt(max(d[best], 1e-15))
                # e_j = (L[j, best] - sum_{s<t} c[j,s]*c[best,s]) / sqrt_d_best
                e = L[:, best].copy()
                for s_idx in range(t):
                    e -= c[:, s_idx] * c[best, s_idx]
                e /= sqrt_d_best
                c[:, t] = e
                # Update conditional variances
                d -= e ** 2
                d = np.maximum(d, 0.0)

        return np.array(selected, dtype=np.intp)

    # ------------------------------------------------------------------
    # MAP inference: find the most diverse set (greedy + lazy evaluations)
    # ------------------------------------------------------------------

    def map_inference(self, k: int) -> Tuple[np.ndarray, float]:
        """Find the subset S of size k maximizing det(L_S).

        Uses greedy with lazy evaluations (priority queue emulation).
        The greedy algorithm achieves at least (1/k!)^k of the optimal
        determinant but works well in practice.

        Args:
            k: Number of items to select.

        Returns:
            (selected_indices, log_det_value)
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")
        n = self.n
        k = min(k, n)

        # Use the greedy Cholesky approach (equivalent to MAP greedy)
        selected = self.greedy_sample(k)

        # Compute log-det of the selected submatrix
        L_S = self.L[np.ix_(selected, selected)]
        sign, logdet = np.linalg.slogdet(L_S)
        if sign <= 0:
            logdet = -np.inf

        return selected, logdet

    def map_inference_lazy(self, k: int) -> Tuple[np.ndarray, float]:
        """MAP inference with lazy evaluations for speedup.

        Maintains upper bounds on marginal gains; only recomputes
        when an item rises to the top of the priority list.

        Args:
            k: Subset size.

        Returns:
            (selected_indices, log_det_value)
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")
        n = self.n
        k = min(k, n)

        selected = []
        selected_set = set()
        # Upper bounds on marginal gains
        gains = np.diag(self.L).copy()
        current = np.arange(n)
        up_to_date = np.zeros(n, dtype=bool)

        for t in range(k):
            while True:
                # Pick candidate with highest upper bound
                candidates = np.array([i for i in range(n) if i not in selected_set])
                if len(candidates) == 0:
                    break
                best_cand = candidates[np.argmax(gains[candidates])]

                if up_to_date[best_cand]:
                    selected.append(best_cand)
                    selected_set.add(best_cand)
                    up_to_date[:] = False
                    break

                # Recompute exact marginal gain
                trial = list(selected) + [best_cand]
                L_trial = self.L[np.ix_(trial, trial)]
                if len(selected) > 0:
                    L_curr = self.L[np.ix_(selected, selected)]
                    _, ld_curr = np.linalg.slogdet(L_curr)
                    _, ld_trial = np.linalg.slogdet(L_trial)
                    gains[best_cand] = ld_trial - ld_curr
                else:
                    gains[best_cand] = np.log(max(self.L[best_cand, best_cand], 1e-30))
                up_to_date[best_cand] = True

        selected = np.array(selected, dtype=np.intp)
        L_S = self.L[np.ix_(selected, selected)]
        sign, logdet = np.linalg.slogdet(L_S)
        if sign <= 0:
            logdet = -np.inf
        return selected, logdet

    # ------------------------------------------------------------------
    # Log-probability of a subset
    # ------------------------------------------------------------------

    def log_probability(self, subset: Union[List[int], np.ndarray]) -> float:
        """Compute log P(S) = log det(L_S) - log det(L + I).

        Args:
            subset: Indices of items in the subset.

        Returns:
            Log-probability of the subset under the DPP.
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")

        subset = np.asarray(subset, dtype=np.intp)
        if len(subset) == 0:
            # P(empty) = det(I) / det(L + I) = 1/det(L+I)
            sign, logdet_norm = np.linalg.slogdet(self.L + np.eye(self.n))
            return -logdet_norm

        L_S = self.L[np.ix_(subset, subset)]
        sign_s, logdet_s = np.linalg.slogdet(L_S)

        LpI = self.L + np.eye(self.n)
        sign_n, logdet_n = np.linalg.slogdet(LpI)

        if sign_s <= 0 or sign_n <= 0:
            return -np.inf

        return logdet_s - logdet_n

    # ------------------------------------------------------------------
    # Quality-Diversity DPP: L = diag(q) S diag(q)
    # ------------------------------------------------------------------

    @staticmethod
    def quality_diversity_kernel(quality_scores: np.ndarray,
                                 similarity_matrix: np.ndarray) -> np.ndarray:
        """Build quality-diversity L-ensemble kernel.

        L_ij = q_i * S_ij * q_j

        This decomposes relevance (quality) from diversity (similarity).
        High-quality items are more likely to be selected, but similar
        items repel each other.

        Args:
            quality_scores: (n,) positive quality scores.
            similarity_matrix: (n, n) similarity matrix (PSD).

        Returns:
            (n, n) L-ensemble kernel.
        """
        q = np.asarray(quality_scores, dtype=np.float64)
        S = np.asarray(similarity_matrix, dtype=np.float64)
        Q = np.diag(q)
        L = Q @ S @ Q
        L = 0.5 * (L + L.T)
        return L

    def fit_quality_diversity(self, quality_scores: np.ndarray,
                              similarity_matrix: np.ndarray) -> 'DPPSampler':
        """Fit DPP with quality-diversity decomposition.

        Args:
            quality_scores: (n,) positive quality scores.
            similarity_matrix: (n, n) PSD similarity matrix.

        Returns:
            self
        """
        L = self.quality_diversity_kernel(quality_scores, similarity_matrix)
        return self.fit(L)

    # ------------------------------------------------------------------
    # Conditional DPP: sample given some items already selected
    # ------------------------------------------------------------------

    def sample_conditional(self, fixed_items: Union[List[int], np.ndarray],
                           k_additional: int,
                           rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample from DPP conditioned on fixed_items being in the set.

        Uses Schur complement: the conditional DPP on the remaining items
        has kernel L_cond = L_BB - L_BA L_AA^{-1} L_AB where A = fixed, B = rest.

        Args:
            fixed_items: Indices of items already selected.
            k_additional: Number of additional items to sample.
            rng: Random state.

        Returns:
            Array of k_additional new indices (sorted).
        """
        if self.L is None:
            raise RuntimeError("Call fit() first.")
        if rng is None:
            rng = np.random.RandomState()

        fixed = np.asarray(fixed_items, dtype=np.intp)
        all_idx = np.arange(self.n)
        remaining_mask = np.ones(self.n, dtype=bool)
        remaining_mask[fixed] = False
        remaining = all_idx[remaining_mask]

        if len(remaining) == 0 or k_additional == 0:
            return np.array([], dtype=np.intp)

        L_AA = self.L[np.ix_(fixed, fixed)]
        L_AB = self.L[np.ix_(fixed, remaining)]
        L_BB = self.L[np.ix_(remaining, remaining)]

        # Conditional kernel via Schur complement
        try:
            L_AA_inv = np.linalg.inv(L_AA + 1e-10 * np.eye(len(fixed)))
            L_cond = L_BB - L_AB.T @ L_AA_inv @ L_AB
        except np.linalg.LinAlgError:
            L_cond = L_BB.copy()

        # Ensure PSD
        L_cond = 0.5 * (L_cond + L_cond.T)
        eigvals = np.linalg.eigvalsh(L_cond)
        if np.min(eigvals) < -1e-6:
            L_cond -= (np.min(eigvals) - 1e-6) * np.eye(len(remaining))

        cond_sampler = DPPSampler()
        cond_sampler.fit(L_cond)

        k_additional = min(k_additional, len(remaining))
        local_indices = cond_sampler.greedy_sample(k_additional)

        return np.sort(remaining[local_indices])

    # ------------------------------------------------------------------
    # Streaming DPP: online updates
    # ------------------------------------------------------------------

    def streaming_update(self, new_item_features: np.ndarray,
                         selected_so_far: np.ndarray,
                         max_selected: int,
                         kernel_func: str = 'rbf',
                         gamma: Optional[float] = None) -> Tuple[bool, float]:
        """Decide whether to include a new streaming item.

        Uses the log-det gain criterion: include the new item if it
        increases log det(L_S) by more than a threshold.

        Args:
            new_item_features: (d,) feature vector of new item.
            selected_so_far: (m, d) features of already-selected items.
            max_selected: Maximum number of items to select.
            kernel_func: Kernel type for computing similarities.
            gamma: RBF gamma parameter.

        Returns:
            (should_include, gain): Whether to include and the log-det gain.
        """
        if len(selected_so_far) == 0:
            return True, float('inf')

        if len(selected_so_far) >= max_selected:
            # Check if swapping improves diversity
            all_items = np.vstack([selected_so_far, new_item_features.reshape(1, -1)])
            K = compute_kernel(all_items, kernel=kernel_func, gamma=gamma)
            n_sel = len(selected_so_far)

            K_current = K[:n_sel, :n_sel]
            _, ld_current = np.linalg.slogdet(K_current)

            best_gain = -np.inf
            for i in range(n_sel):
                trial_idx = [j for j in range(n_sel) if j != i] + [n_sel]
                K_trial = K[np.ix_(trial_idx, trial_idx)]
                _, ld_trial = np.linalg.slogdet(K_trial)
                gain = ld_trial - ld_current
                best_gain = max(best_gain, gain)

            return best_gain > 0.0, best_gain

        # Not at capacity: compute gain of adding
        all_items = np.vstack([selected_so_far, new_item_features.reshape(1, -1)])
        K = compute_kernel(all_items, kernel=kernel_func, gamma=gamma)
        n_sel = len(selected_so_far)
        K_current = K[:n_sel, :n_sel]
        K_new = K[:n_sel + 1, :n_sel + 1]

        _, ld_current = np.linalg.slogdet(K_current)
        _, ld_new = np.linalg.slogdet(K_new)
        gain = ld_new - ld_current

        return gain > 0.01, gain

    # ------------------------------------------------------------------
    # DPP Kernel Learning
    # ------------------------------------------------------------------

    @staticmethod
    def learn_kernel(observed_sets: List[List[int]], n_items: int,
                     n_features: int = 10, lr: float = 0.01,
                     max_iter: int = 200, reg: float = 1e-4) -> np.ndarray:
        """Learn DPP kernel from observed diverse sets via gradient ascent.

        Parameterize L = B B^T where B is (n, d). Optimize the
        log-likelihood of observed subsets.

        The log-likelihood of a set S under DPP(L) is:
            log P(S) = log det(L_S) - log det(L + I)

        We maximize sum of log-likelihoods over observed sets.

        Args:
            observed_sets: List of subsets (each a list of indices).
            n_items: Total number of items.
            n_features: Dimensionality of the learned embedding.
            lr: Learning rate.
            max_iter: Number of gradient steps.
            reg: L2 regularization on B.

        Returns:
            (n_items, n_items) learned kernel matrix L.
        """
        rng = np.random.RandomState(42)
        B = rng.randn(n_items, n_features) * 0.1

        for iteration in range(max_iter):
            L = B @ B.T
            LpI = L + np.eye(n_items)

            try:
                LpI_inv = np.linalg.inv(LpI)
            except np.linalg.LinAlgError:
                LpI_inv = np.linalg.pinv(LpI)

            grad_B = np.zeros_like(B)

            for S in observed_sets:
                S = list(S)
                if len(S) == 0:
                    continue

                L_S = L[np.ix_(S, S)]
                try:
                    L_S_inv = np.linalg.inv(L_S + 1e-10 * np.eye(len(S)))
                except np.linalg.LinAlgError:
                    continue

                # Gradient of log det(L_S) w.r.t. L
                # = L_S^{-1} restricted to S×S positions
                grad_logdet_S = np.zeros((n_items, n_items))
                for ii, si in enumerate(S):
                    for jj, sj in enumerate(S):
                        grad_logdet_S[si, sj] = L_S_inv[ii, jj]

                # Gradient of log det(L+I) w.r.t. L = (L+I)^{-1}
                grad_norm = LpI_inv

                # grad w.r.t. L
                grad_L = grad_logdet_S - grad_norm

                # grad w.r.t. B: d/dB (L) where L = B B^T
                # dL/dB = 2 grad_L @ B (since L = BB^T is symmetric)
                grad_B += 2.0 * grad_L @ B

            # Regularization
            grad_B -= 2.0 * reg * B

            # Gradient ascent
            B += lr * grad_B / max(len(observed_sets), 1)

            # Clip for stability
            B = np.clip(B, -10.0, 10.0)

        L = B @ B.T
        L = 0.5 * (L + L.T)
        return L


class DPPEnsemble:
    """Ensemble of DPP samplers for robust diversity.

    Combines multiple kernel functions and averages their selections.
    """

    def __init__(self, kernels: Optional[List[str]] = None,
                 gammas: Optional[List[float]] = None):
        self.kernels = kernels or ['rbf', 'cosine']
        self.gammas = gammas or [None]
        self.samplers = []

    def fit(self, items: np.ndarray) -> 'DPPEnsemble':
        """Fit multiple DPP samplers with different kernels.

        Args:
            items: (n, d) feature matrix.

        Returns:
            self
        """
        self.samplers = []
        for kernel in self.kernels:
            for gamma in self.gammas:
                K = compute_kernel(items, kernel=kernel, gamma=gamma)
                sampler = DPPSampler()
                sampler.fit(K)
                self.samplers.append(sampler)
        return self

    def sample(self, k: int,
               rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample k items by scoring across all DPP samplers.

        Each sampler does a greedy selection; items appearing most often
        across samplers are selected.

        Args:
            k: Number of items.
            rng: Random state.

        Returns:
            Array of k selected indices.
        """
        if not self.samplers:
            raise RuntimeError("Call fit() first.")
        if rng is None:
            rng = np.random.RandomState()

        n = self.samplers[0].n
        counts = np.zeros(n, dtype=np.float64)

        for sampler in self.samplers:
            sel = sampler.greedy_sample(k)
            counts[sel] += 1.0

        # Select top-k by frequency
        top_k = np.argsort(-counts)[:k]
        return np.sort(top_k)


class DualDPP:
    """Dual DPP representation for when n >> d.

    When the number of items n is much larger than the feature
    dimensionality d, we can work with the d×d dual kernel C = B^T B
    instead of the n×n L = B B^T.
    """

    def __init__(self):
        self.B = None
        self.C = None
        self.n = 0
        self.d = 0

    def fit(self, features: np.ndarray) -> 'DualDPP':
        """Fit dual DPP from features.

        Args:
            features: (n, d) feature matrix where typically n >> d.

        Returns:
            self
        """
        self.B = np.asarray(features, dtype=np.float64)
        self.n, self.d = self.B.shape
        self.C = self.B.T @ self.B  # (d, d)
        return self

    def sample_k(self, k: int,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample k items using the dual representation.

        1. Eigendecompose C = U diag(μ) U^T  (d×d, fast).
        2. Select k eigenvectors via k-DPP on eigenvalues.
        3. Map back to item space and do elementary DPP sampling.

        Args:
            k: Number of items.
            rng: Random state.

        Returns:
            Sorted array of k selected indices.
        """
        if self.B is None:
            raise RuntimeError("Call fit() first.")
        if rng is None:
            rng = np.random.RandomState()

        eigenvalues, U = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Build a DPP sampler on the dual eigenvalues
        dual_sampler = DPPSampler()
        dual_L = np.diag(eigenvalues)
        dual_sampler.fit(dual_L)

        # Use k-DPP to select eigenvectors
        E = dual_sampler._compute_elem_sym_poly(eigenvalues, k)

        selected_eig = []
        remaining_k = k
        d = len(eigenvalues)
        for i in range(d - 1, -1, -1):
            if remaining_k == 0:
                break
            if remaining_k > i + 1:
                selected_eig.append(i)
                remaining_k -= 1
                continue
            if E[remaining_k, i] < 1e-30:
                prob = 1.0
            else:
                prob = eigenvalues[i] * E[remaining_k - 1, i] / E[remaining_k, i + 1]
            prob = min(max(prob, 0.0), 1.0)
            if rng.rand() < prob:
                selected_eig.append(i)
                remaining_k -= 1

        if len(selected_eig) != k:
            # Fallback: greedy
            L = self.B @ self.B.T
            s = DPPSampler()
            s.fit(L)
            return s.greedy_sample(k)

        # Map selected dual eigenvectors back to item space
        # V_hat = B U_sel diag(1/sqrt(μ_sel))
        U_sel = U[:, selected_eig]
        mu_sel = eigenvalues[selected_eig]
        mu_sel_sqrt_inv = 1.0 / np.sqrt(np.maximum(mu_sel, 1e-15))
        V_hat = self.B @ U_sel @ np.diag(mu_sel_sqrt_inv)

        return DPPSampler()._sample_elementary_dpp(V_hat, rng)


class MixtureDPP:
    """Mixture of DPPs for multi-modal diversity.

    P(S) = Σ_m π_m * P_m(S)

    Useful when diversity requirements vary across contexts.
    """

    def __init__(self):
        self.samplers = []
        self.weights = None

    def fit(self, kernels: List[np.ndarray],
            weights: Optional[np.ndarray] = None) -> 'MixtureDPP':
        """Fit mixture of DPPs.

        Args:
            kernels: List of (n, n) kernel matrices.
            weights: Mixture weights (default: uniform).

        Returns:
            self
        """
        self.samplers = []
        for K in kernels:
            s = DPPSampler()
            s.fit(K)
            self.samplers.append(s)

        if weights is None:
            self.weights = np.ones(len(kernels)) / len(kernels)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights /= self.weights.sum()

        return self

    def sample(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample from the mixture DPP.

        First select a component, then sample from that DPP.

        Returns:
            Sorted array of selected indices.
        """
        if rng is None:
            rng = np.random.RandomState()

        comp = rng.choice(len(self.samplers), p=self.weights)
        return self.samplers[comp].sample(rng=rng)

    def sample_k(self, k: int,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample exactly k items from the mixture k-DPP."""
        if rng is None:
            rng = np.random.RandomState()

        comp = rng.choice(len(self.samplers), p=self.weights)
        return self.samplers[comp].sample_k(k, rng=rng)


def demo_dpp():
    """Quick demonstration of DPP sampling."""
    rng = np.random.RandomState(42)
    n, d = 50, 5
    items = rng.randn(n, d)
    K = compute_kernel(items, kernel='rbf', gamma=0.5)

    sampler = DPPSampler()
    sampler.fit(K)

    # Exact sample
    s1 = sampler.sample(rng=rng)
    print(f"Exact DPP sample size: {len(s1)}, indices: {s1[:10]}...")

    # k-DPP sample
    s2 = sampler.sample_k(5, rng=rng)
    print(f"k-DPP sample (k=5): {s2}")

    # Greedy sample
    s3 = sampler.greedy_sample(5)
    print(f"Greedy DPP sample (k=5): {s3}")

    # MAP inference
    s4, ld = sampler.map_inference(5)
    print(f"MAP inference (k=5): {s4}, log-det={ld:.4f}")

    # Log probability
    lp = sampler.log_probability(s2)
    print(f"Log probability of k-DPP sample: {lp:.4f}")

    # Quality-diversity
    quality = rng.rand(n) + 0.1
    L_qd = DPPSampler.quality_diversity_kernel(quality, K)
    sampler_qd = DPPSampler()
    sampler_qd.fit(L_qd)
    s5 = sampler_qd.greedy_sample(5)
    print(f"Quality-diversity sample: {s5}")

    # Conditional
    s6 = sampler.sample_conditional([0, 1], 3, rng=rng)
    print(f"Conditional sample (fixed [0,1], +3): {s6}")

    # Kernel learning
    observed = [rng.choice(n, size=5, replace=False).tolist() for _ in range(10)]
    L_learned = DPPSampler.learn_kernel(observed, n, n_features=5, max_iter=50)
    print(f"Learned kernel shape: {L_learned.shape}, max={L_learned.max():.4f}")


if __name__ == '__main__':
    demo_dpp()
