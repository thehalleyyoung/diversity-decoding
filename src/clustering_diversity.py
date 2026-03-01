"""
Cluster-based diversity selection.

Implements k-medoids (PAM), DBSCAN, hierarchical clustering,
spectral clustering, cluster-then-select, coverage maximization,
and facility location — all from scratch using numpy only.
"""

import numpy as np
from typing import Optional, List, Tuple, Union


def pairwise_distances(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Compute pairwise distance matrix.

    Args:
        X: (n, d) feature matrix.
        metric: 'euclidean' or 'cosine'.

    Returns:
        (n, n) distance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if metric == 'euclidean':
        sq = np.sum(X ** 2, axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        D = np.sqrt(np.maximum(D, 0.0))
    elif metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Xn = X / norms
        sim = Xn @ Xn.T
        np.clip(sim, -1.0, 1.0, out=sim)
        D = 1.0 - sim
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return D


# ======================================================================
# K-Medoids (PAM Algorithm)
# ======================================================================

class KMedoids:
    """Partitioning Around Medoids (PAM) clustering.

    Unlike k-means, k-medoids uses actual data points as cluster centers,
    making it more robust to outliers and applicable to arbitrary distance metrics.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 metric: str = 'euclidean', random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray) -> 'KMedoids':
        """Fit k-medoids clustering.

        BUILD phase: greedily select initial medoids.
        SWAP phase: iteratively try swapping medoid with non-medoid
        if it reduces total cost.

        Args:
            X: (n, d) data matrix.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        k = min(self.n_clusters, n)
        rng = np.random.RandomState(self.random_state)

        D = pairwise_distances(X, self.metric)

        # BUILD phase: greedy initialization
        medoids = []
        # First medoid: minimizes sum of distances to all points
        total_dists = D.sum(axis=1)
        medoids.append(int(np.argmin(total_dists)))

        for _ in range(1, k):
            # Select point that most reduces total distance
            current_d = D[:, medoids].min(axis=1)
            gains = np.zeros(n)
            for j in range(n):
                if j in medoids:
                    gains[j] = -np.inf
                    continue
                new_d = np.minimum(current_d, D[:, j])
                gains[j] = current_d.sum() - new_d.sum()
            medoids.append(int(np.argmax(gains)))

        medoids = np.array(medoids, dtype=np.intp)

        # SWAP phase
        for iteration in range(self.max_iter):
            # Assign points to nearest medoid
            D_medoids = D[:, medoids]
            labels = np.argmin(D_medoids, axis=1)
            cost = np.sum(D_medoids[np.arange(n), labels])

            improved = False
            for m_idx in range(k):
                for candidate in range(n):
                    if candidate in medoids:
                        continue
                    # Try swapping medoid m_idx with candidate
                    new_medoids = medoids.copy()
                    new_medoids[m_idx] = candidate
                    D_new = D[:, new_medoids]
                    new_labels = np.argmin(D_new, axis=1)
                    new_cost = np.sum(D_new[np.arange(n), new_labels])

                    if new_cost < cost - 1e-10:
                        medoids = new_medoids
                        cost = new_cost
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        # Final assignment
        D_medoids = D[:, medoids]
        self.labels_ = np.argmin(D_medoids, axis=1)
        self.medoid_indices_ = medoids
        self.inertia_ = np.sum(D_medoids[np.arange(n), self.labels_])
        return self

    def predict(self, X: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Assign new points to nearest medoid."""
        X = np.asarray(X, dtype=np.float64)
        X_train = np.asarray(X_train, dtype=np.float64)
        medoid_points = X_train[self.medoid_indices_]
        if self.metric == 'euclidean':
            dists = np.sqrt(np.sum((X[:, None, :] - medoid_points[None, :, :]) ** 2, axis=2))
        else:
            nX = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            nM = medoid_points / np.maximum(np.linalg.norm(medoid_points, axis=1, keepdims=True), 1e-12)
            dists = 1.0 - nX @ nM.T
        return np.argmin(dists, axis=1)


# ======================================================================
# DBSCAN
# ======================================================================

class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise.

    Finds clusters of arbitrary shape based on density connectivity.
    Points in low-density regions are classified as noise.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """Fit DBSCAN clustering.

        1. Find core points (>= min_samples neighbors within eps).
        2. Build clusters by connecting density-reachable core points.
        3. Assign border points to nearest core point's cluster.
        4. Remaining points are noise (label = -1).

        Args:
            X: (n, d) data matrix.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        D = pairwise_distances(X, self.metric)

        # Find neighbors within eps for each point
        neighborhoods = []
        for i in range(n):
            neighbors = np.where(D[i] <= self.eps)[0]
            neighborhoods.append(neighbors)

        # Identify core points
        is_core = np.array([len(neighborhoods[i]) >= self.min_samples for i in range(n)])
        core_indices = np.where(is_core)[0]
        self.core_sample_indices_ = core_indices

        labels = np.full(n, -1, dtype=np.intp)
        cluster_id = 0

        visited = np.zeros(n, dtype=bool)

        for i in core_indices:
            if visited[i]:
                continue

            # BFS from core point i
            queue = [i]
            visited[i] = True
            labels[i] = cluster_id

            head = 0
            while head < len(queue):
                current = queue[head]
                head += 1

                for neighbor in neighborhoods[current]:
                    if labels[neighbor] == -1 or not visited[neighbor]:
                        labels[neighbor] = cluster_id
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        if is_core[neighbor]:
                            queue.append(neighbor)

            cluster_id += 1

        self.labels_ = labels
        return self


# ======================================================================
# Hierarchical Clustering
# ======================================================================

class HierarchicalClustering:
    """Agglomerative hierarchical clustering.

    Implements single-link, complete-link, and average-link strategies.
    Builds a dendrogram by iteratively merging the closest clusters.
    """

    def __init__(self, n_clusters: int = 3, linkage: str = 'average',
                 metric: str = 'euclidean'):
        """
        Args:
            n_clusters: Number of clusters to produce.
            linkage: 'single', 'complete', or 'average'.
            metric: Distance metric.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None
        self.dendrogram_ = None

    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """Fit hierarchical clustering.

        Args:
            X: (n, d) data matrix.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        D = pairwise_distances(X, self.metric)

        # Each point starts as its own cluster
        clusters = {i: [i] for i in range(n)}
        # Distance between clusters
        cluster_dist = D.copy()
        np.fill_diagonal(cluster_dist, np.inf)

        # Track merges for dendrogram: (i, j, distance, new_size)
        merges = []
        active = set(range(n))
        next_id = n

        while len(active) > self.n_clusters:
            # Find closest pair
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            active_list = sorted(active)
            for ii, ci in enumerate(active_list):
                for cj in active_list[ii + 1:]:
                    if cluster_dist[ci, cj] < min_dist:
                        min_dist = cluster_dist[ci, cj]
                        merge_i, merge_j = ci, cj

            if merge_i < 0:
                break

            # Merge clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            clusters[next_id] = new_cluster
            merges.append((merge_i, merge_j, min_dist, len(new_cluster)))

            # Update distances
            new_row = np.full(cluster_dist.shape[0], np.inf)
            for ck in active:
                if ck == merge_i or ck == merge_j:
                    continue
                if self.linkage == 'single':
                    d = min(cluster_dist[merge_i, ck], cluster_dist[merge_j, ck])
                elif self.linkage == 'complete':
                    d = max(cluster_dist[merge_i, ck], cluster_dist[merge_j, ck])
                elif self.linkage == 'average':
                    ni = len(clusters[merge_i])
                    nj = len(clusters[merge_j])
                    d = (ni * cluster_dist[merge_i, ck] + nj * cluster_dist[merge_j, ck]) / (ni + nj)
                else:
                    d = min(cluster_dist[merge_i, ck], cluster_dist[merge_j, ck])
                new_row[ck] = d

            # Expand matrix if needed
            if next_id >= cluster_dist.shape[0]:
                new_size = next_id + 1
                new_mat = np.full((new_size, new_size), np.inf)
                old_size = cluster_dist.shape[0]
                new_mat[:old_size, :old_size] = cluster_dist
                cluster_dist = new_mat

            cluster_dist[next_id, :len(new_row)] = new_row
            cluster_dist[:len(new_row), next_id] = new_row
            cluster_dist[next_id, next_id] = np.inf

            active.discard(merge_i)
            active.discard(merge_j)
            active.add(next_id)
            next_id += 1

        # Assign labels
        labels = np.full(n, -1, dtype=np.intp)
        for cluster_label, cluster_id in enumerate(sorted(active)):
            for point_idx in clusters[cluster_id]:
                labels[point_idx] = cluster_label

        self.labels_ = labels
        self.dendrogram_ = merges
        return self


# ======================================================================
# Spectral Clustering
# ======================================================================

class SpectralClustering:
    """Spectral clustering via graph Laplacian eigendecomposition.

    1. Construct affinity/adjacency graph.
    2. Compute normalized graph Laplacian.
    3. Find bottom-k eigenvectors.
    4. Run k-means on the eigenvector embedding.
    """

    def __init__(self, n_clusters: int = 3, gamma: float = 1.0,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X: np.ndarray) -> 'SpectralClustering':
        """Fit spectral clustering.

        Args:
            X: (n, d) data matrix.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        k = min(self.n_clusters, n)

        # Construct RBF affinity matrix
        D_sq = pairwise_distances(X, 'euclidean') ** 2
        W = np.exp(-self.gamma * D_sq)
        np.fill_diagonal(W, 0.0)

        # Degree matrix
        d = W.sum(axis=1)
        d_inv_sqrt = np.zeros(n)
        mask = d > 1e-12
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)

        # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
        L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # Eigendecomposition (smallest k eigenvectors)
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        # Take first k eigenvectors (smallest eigenvalues)
        U = eigenvectors[:, :k]

        # Normalize rows
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        U = U / row_norms

        # K-means on U
        self.labels_ = self._kmeans(U, k)
        return self

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """Simple k-means implementation."""
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]

        # K-means++ initialization
        centers = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / (dists.sum() + 1e-12)
            centers.append(X[rng.choice(n, p=probs)])
        centers = np.array(centers)

        for _ in range(max_iter):
            # Assign
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            # Update
            new_centers = np.zeros_like(centers)
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)
                else:
                    new_centers[c] = centers[c]

            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers

        return labels


# ======================================================================
# ClusterDiversity: cluster-then-select
# ======================================================================

class ClusterDiversity:
    """Cluster-based diversity selection.

    Clusters items and selects representatives from each cluster
    to ensure diverse coverage.
    """

    def __init__(self, method: str = 'kmedoids', metric: str = 'euclidean',
                 random_state: int = 42):
        """
        Args:
            method: 'kmedoids', 'dbscan', 'hierarchical', or 'spectral'.
            metric: Distance metric.
            random_state: Random seed.
        """
        self.method = method
        self.metric = metric
        self.random_state = random_state

    def select(self, items: np.ndarray, k: int,
               strategy: str = 'nearest_centroid') -> np.ndarray:
        """Select k diverse items via clustering.

        Args:
            items: (n, d) feature matrix.
            k: Number of items to select.
            strategy: 'nearest_centroid' or 'random'.

        Returns:
            Array of k selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        k = min(k, n)

        # Cluster
        if self.method == 'kmedoids':
            model = KMedoids(n_clusters=k, metric=self.metric,
                             random_state=self.random_state)
            model.fit(items)
            labels = model.labels_
            if strategy == 'nearest_centroid':
                return model.medoid_indices_[:k]
        elif self.method == 'hierarchical':
            model = HierarchicalClustering(n_clusters=k, metric=self.metric)
            model.fit(items)
            labels = model.labels_
        elif self.method == 'spectral':
            model = SpectralClustering(n_clusters=k, random_state=self.random_state)
            model.fit(items)
            labels = model.labels_
        elif self.method == 'dbscan':
            model = DBSCAN(metric=self.metric)
            model.fit(items)
            labels = model.labels_
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Select one item per cluster
        rng = np.random.RandomState(self.random_state)
        unique_labels = np.unique(labels)
        # Remove noise label (-1)
        unique_labels = unique_labels[unique_labels >= 0]

        selected = []
        D = pairwise_distances(items, self.metric)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            if strategy == 'nearest_centroid':
                # Select the medoid of this cluster
                cluster_D = D[np.ix_(cluster_indices, cluster_indices)]
                medoid_local = np.argmin(cluster_D.sum(axis=1))
                selected.append(cluster_indices[medoid_local])
            elif strategy == 'random':
                selected.append(rng.choice(cluster_indices))
            else:
                selected.append(cluster_indices[0])

        selected = np.array(selected, dtype=np.intp)

        # If we have fewer than k, fill with most distant remaining points
        if len(selected) < k:
            remaining = np.array([i for i in range(n) if i not in selected])
            while len(selected) < k and len(remaining) > 0:
                min_dists = D[remaining][:, selected].min(axis=1)
                best = np.argmax(min_dists)
                selected = np.append(selected, remaining[best])
                remaining = np.delete(remaining, best)

        return selected[:k]


# ======================================================================
# Coverage Maximization
# ======================================================================

class CoverageMaximizer:
    """Greedily select items to maximize space coverage.

    Coverage is defined as the fraction of the space within some
    radius of at least one selected item.
    """

    def __init__(self, radius: Optional[float] = None, metric: str = 'euclidean'):
        self.radius = radius
        self.metric = metric

    def select(self, items: np.ndarray, k: int) -> np.ndarray:
        """Select k items maximizing coverage.

        Greedy algorithm: at each step, select the item that covers
        the most uncovered points.

        Args:
            items: (n, d) feature matrix.
            k: Number of items to select.

        Returns:
            Selected indices.
        """
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        k = min(k, n)

        D = pairwise_distances(items, self.metric)

        if self.radius is None:
            # Auto-set radius to median pairwise distance
            upper_tri = D[np.triu_indices(n, k=1)]
            self.radius = float(np.median(upper_tri))

        covered = np.zeros(n, dtype=bool)
        selected = []

        for _ in range(k):
            # Count newly covered points for each candidate
            best_idx = -1
            best_gain = -1

            for i in range(n):
                if i in selected:
                    continue
                # Points covered by i
                newly_covered = np.sum((~covered) & (D[i] <= self.radius))
                if newly_covered > best_gain:
                    best_gain = newly_covered
                    best_idx = i

            if best_idx < 0:
                break

            selected.append(best_idx)
            covered |= (D[best_idx] <= self.radius)

        # If fewer than k selected, add farthest points
        if len(selected) < k:
            remaining = [i for i in range(n) if i not in selected]
            while len(selected) < k and remaining:
                sel_arr = np.array(selected)
                min_dists = D[remaining][:, sel_arr].min(axis=1)
                best = np.argmax(min_dists)
                selected.append(remaining[best])
                remaining.pop(best)

        return np.array(selected[:k], dtype=np.intp)

    def coverage_score(self, items: np.ndarray, selected: np.ndarray) -> float:
        """Compute coverage fraction.

        Args:
            items: (n, d) all items.
            selected: Indices of selected items.

        Returns:
            Fraction of items within radius of at least one selected item.
        """
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        D = pairwise_distances(items, self.metric)

        if self.radius is None:
            upper_tri = D[np.triu_indices(n, k=1)]
            self.radius = float(np.median(upper_tri))

        covered = np.zeros(n, dtype=bool)
        for s in selected:
            covered |= (D[s] <= self.radius)
        return float(np.mean(covered))


# ======================================================================
# Facility Location
# ======================================================================

class FacilityLocation:
    """Facility location for diversity: maximize Σ_i max_{j∈S} sim(i,j).

    This is a monotone submodular function, so greedy achieves
    (1 - 1/e) approximation.
    """

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def select(self, items: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        """Greedy facility location selection.

        Args:
            items: (n, d) feature matrix.
            k: Number to select.

        Returns:
            (selected_indices, objective_value)
        """
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        k = min(k, n)

        # Convert distances to similarities
        D = pairwise_distances(items, self.metric)
        max_d = D.max() + 1e-10
        sim = max_d - D  # similarity: higher = more similar
        np.fill_diagonal(sim, 0.0)

        # Greedy with lazy evaluations
        selected = []
        max_sim_to_sel = np.zeros(n)  # max similarity to selected set
        obj_value = 0.0

        for _ in range(k):
            # Marginal gain of adding each candidate
            gains = np.zeros(n)
            for i in range(n):
                if i in selected:
                    gains[i] = -np.inf
                    continue
                # gain = Σ_j max(sim(j,i) - max_sim_to_sel[j], 0)
                improvements = np.maximum(sim[:, i] - max_sim_to_sel, 0.0)
                gains[i] = improvements.sum()

            best = np.argmax(gains)
            if gains[best] <= 0:
                break

            selected.append(best)
            obj_value += gains[best]
            np.maximum(max_sim_to_sel, sim[:, best], out=max_sim_to_sel)

        return np.array(selected, dtype=np.intp), obj_value

    def objective(self, items: np.ndarray, selected: np.ndarray) -> float:
        """Compute facility location objective.

        f(S) = Σ_i max_{j∈S} sim(i,j)
        """
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        D = pairwise_distances(items, self.metric)
        max_d = D.max() + 1e-10
        sim = max_d - D
        np.fill_diagonal(sim, 0.0)

        selected = np.asarray(selected, dtype=np.intp)
        return float(np.sum(np.max(sim[:, selected], axis=1)))


class DiversityDispersion:
    """Max-min dispersion: select items maximizing minimum pairwise distance.

    This is a classic approach to geometric diversity.
    """

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric

    def select(self, items: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        """Greedy max-min dispersion.

        Start with the farthest pair, then iteratively add the item
        farthest from all selected items.

        Args:
            items: (n, d) feature matrix.
            k: Number to select.

        Returns:
            (selected_indices, min_pairwise_distance)
        """
        items = np.asarray(items, dtype=np.float64)
        n = items.shape[0]
        k = min(k, n)

        D = pairwise_distances(items, self.metric)

        if k <= 1:
            return np.array([0], dtype=np.intp), 0.0

        # Start with farthest pair
        np.fill_diagonal(D, 0.0)
        i, j = np.unravel_index(np.argmax(D), D.shape)
        selected = [i, j]
        min_dist_to_sel = np.minimum(D[:, i], D[:, j])

        for _ in range(2, k):
            # Add point farthest from selected set
            min_dist_to_sel_copy = min_dist_to_sel.copy()
            for s in selected:
                min_dist_to_sel_copy[s] = -np.inf
            best = np.argmax(min_dist_to_sel_copy)
            selected.append(best)
            np.minimum(min_dist_to_sel, D[:, best], out=min_dist_to_sel)

        sel = np.array(selected, dtype=np.intp)
        D_sel = D[np.ix_(sel, sel)]
        np.fill_diagonal(D_sel, np.inf)
        min_pw = float(D_sel.min()) if k > 1 else 0.0

        return sel, min_pw


def demo_clustering():
    """Demonstrate clustering diversity methods."""
    rng = np.random.RandomState(42)

    # Create clustered data
    centers = np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float64)
    X = np.vstack([c + rng.randn(20, 2) * 0.5 for c in centers])

    # K-medoids
    km = KMedoids(n_clusters=3)
    km.fit(X)
    print(f"K-medoids labels: {np.unique(km.labels_, return_counts=True)}")
    print(f"Medoid indices: {km.medoid_indices_}")

    # DBSCAN
    db = DBSCAN(eps=1.5, min_samples=3)
    db.fit(X)
    print(f"DBSCAN labels: {np.unique(db.labels_, return_counts=True)}")

    # Hierarchical
    hc = HierarchicalClustering(n_clusters=3)
    hc.fit(X)
    print(f"Hierarchical labels: {np.unique(hc.labels_, return_counts=True)}")

    # Spectral
    sc = SpectralClustering(n_clusters=3, gamma=0.1)
    sc.fit(X)
    print(f"Spectral labels: {np.unique(sc.labels_, return_counts=True)}")

    # ClusterDiversity selection
    cd = ClusterDiversity(method='kmedoids')
    sel = cd.select(X, k=3)
    print(f"Cluster diversity selection: {sel}")

    # Coverage
    cov = CoverageMaximizer(radius=2.0)
    sel_cov = cov.select(X, k=5)
    score = cov.coverage_score(X, sel_cov)
    print(f"Coverage selection: {sel_cov}, coverage={score:.3f}")

    # Facility location
    fl = FacilityLocation()
    sel_fl, obj = fl.select(X, k=5)
    print(f"Facility location: {sel_fl}, objective={obj:.3f}")

    # Dispersion
    dd = DiversityDispersion()
    sel_dd, min_d = dd.select(X, k=5)
    print(f"Dispersion: {sel_dd}, min_dist={min_d:.3f}")


if __name__ == '__main__':
    demo_clustering()
