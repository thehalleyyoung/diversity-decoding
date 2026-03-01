"""Diversity measurement across modalities.

Provides tools for measuring, analyzing, and optimizing diversity
across text, image, audio, and video modalities. Includes cross-modal
alignment scoring, coverage analysis, and feature extraction utilities
that operate without external ML framework dependencies.

Usage::

    analyzer = MultimodalDiversityAnalyzer()
    result = analyzer.analyze_text_image_diversity(texts, images)
    report = cross_modal_coverage({"text": texts, "image": images})
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARTISTIC_STYLES = [
    "oil painting", "watercolor", "pencil sketch", "digital art",
    "photograph", "pixel art", "ink wash", "charcoal drawing",
    "pastel illustration", "woodcut print",
]

_PERSPECTIVES = [
    "bird's-eye view", "worm's-eye view", "close-up",
    "wide-angle", "isometric", "first-person perspective",
    "over-the-shoulder", "side profile", "three-quarter view",
    "panoramic",
]

_MOODS = [
    "serene", "dramatic", "whimsical", "melancholic",
    "vibrant", "mysterious", "nostalgic", "futuristic",
    "chaotic", "minimalist",
]

_COLOR_PALETTES = [
    "warm sunset tones", "cool blue palette", "monochrome",
    "neon cyberpunk colors", "earth tones", "pastel hues",
    "high contrast black and white", "autumn palette",
    "tropical saturated colors", "muted vintage tones",
]

_MEDIA = [
    "on canvas", "on paper", "as a mural", "as a poster",
    "in a sketchbook", "on parchment", "as a digital render",
    "on fabric", "as a mosaic", "as stained glass",
]

_COMPOSITIONS = [
    "rule of thirds", "centered subject", "golden ratio",
    "diagonal composition", "symmetrical", "leading lines",
    "frame within a frame", "negative space", "layered depth",
    "scattered arrangement",
]

_STOP_WORDS = frozenset(
    "a an the is was were be been being have has had do does did "
    "will would shall should may might can could and but or nor "
    "for yet so at by from in into of on to with as it its he she "
    "they them their this that these those i me my we our you your".split()
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossModalDiversity:
    """Result of cross-modal diversity measurement."""

    text_diversity: float
    image_diversity: float
    cross_modal_score: float
    alignment_scores: List[float]
    modality_gap: float

    @property
    def overall_score(self) -> float:
        """Weighted combination of within- and between-modality diversity."""
        within = 0.5 * (self.text_diversity + self.image_diversity)
        return 0.6 * within + 0.4 * self.cross_modal_score


@dataclass
class CoverageReport:
    """Report on how well modalities cover a concept space."""

    covered_dimensions: List[str]
    uncovered_dimensions: List[str]
    coverage_ratio: float
    per_modality_coverage: Dict[str, float]
    recommendations: List[str]


@dataclass
class ModalityFeatures:
    """Feature representation for a single modality."""

    modality_name: str
    feature_vectors: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pairwise_cosine_distances(matrix: np.ndarray) -> np.ndarray:
    """Return pairwise cosine distance matrix for row-vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = matrix / norms
    sim = normed @ normed.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def _mean_pairwise_distance(matrix: np.ndarray) -> float:
    """Average off-diagonal pairwise cosine distance."""
    if matrix.shape[0] < 2:
        return 0.0
    dist = _pairwise_cosine_distances(matrix)
    n = dist.shape[0]
    upper = dist[np.triu_indices(n, k=1)]
    return float(np.mean(upper)) if upper.size > 0 else 0.0


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy from a counts array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute histogram intersection similarity in [0, 1]."""
    min_sum = float(np.sum(np.minimum(h1, h2)))
    max_sum = float(max(np.sum(h1), np.sum(h2)))
    if max_sum < 1e-12:
        return 1.0
    return min_sum / max_sum


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t and t not in _STOP_WORDS]


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract character or word n-grams."""
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _deterministic_hash_float(s: str, low: float = 0.0, high: float = 1.0) -> float:
    """Map a string deterministically to a float in [low, high]."""
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
    return low + (h % 10000) / 10000.0 * (high - low)


# ---------------------------------------------------------------------------
# TextFeatureExtractor
# ---------------------------------------------------------------------------

class TextFeatureExtractor:
    """Extract numerical feature vectors from text using TF-IDF,
    n-gram distributions, sentence structure, and pseudo-POS patterns."""

    def __init__(self, max_vocab: int = 2000, ngram_range: Tuple[int, int] = (1, 3)):
        self.max_vocab = max_vocab
        self.ngram_range = ngram_range
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None

    # -- public API ----------------------------------------------------------

    def fit(self, texts: List[str]) -> TextFeatureExtractor:
        """Build vocabulary and IDF weights from a corpus."""
        doc_freq: Counter = Counter()
        all_tokens: Counter = Counter()
        n_docs = len(texts)

        for text in texts:
            tokens = _tokenize(text)
            all_tokens.update(tokens)
            unique = set(tokens)
            doc_freq.update(unique)

        most_common = all_tokens.most_common(self.max_vocab)
        self._vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

        idf = np.ones(len(self._vocab), dtype=np.float64)
        for word, idx in self._vocab.items():
            df = doc_freq.get(word, 0)
            idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self._idf = idf
        return self

    def transform(self, texts: List[str]) -> ModalityFeatures:
        """Transform texts into feature vectors."""
        if not self._vocab:
            self.fit(texts)

        tfidf_vecs = self._tfidf_vectors(texts)
        ngram_vecs = self._ngram_vectors(texts)
        structure_vecs = self._structure_vectors(texts)

        combined = np.hstack([tfidf_vecs, ngram_vecs, structure_vecs])

        feature_names = (
            [f"tfidf_{w}" for w in self._vocab]
            + [f"ngram_{i}" for i in range(ngram_vecs.shape[1])]
            + ["avg_word_len", "sentence_count", "vocab_richness",
               "punctuation_ratio", "uppercase_ratio", "avg_sentence_len",
               "word_count", "unique_ratio"]
        )
        return ModalityFeatures(
            modality_name="text",
            feature_vectors=combined,
            feature_names=feature_names,
            metadata={"n_documents": len(texts), "vocab_size": len(self._vocab)},
        )

    # -- internals -----------------------------------------------------------

    def _tfidf_vectors(self, texts: List[str]) -> np.ndarray:
        """Compute TF-IDF vectors."""
        n = len(texts)
        v = len(self._vocab)
        mat = np.zeros((n, v), dtype=np.float64)
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            counts = Counter(tokens)
            total = len(tokens) if tokens else 1
            for word, cnt in counts.items():
                if word in self._vocab:
                    idx = self._vocab[word]
                    tf = cnt / total
                    mat[i, idx] = tf * self._idf[idx]
        return mat

    def _ngram_vectors(self, texts: List[str], n_features: int = 64) -> np.ndarray:
        """Hash-based n-gram feature vectors."""
        mat = np.zeros((len(texts), n_features), dtype=np.float64)
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for gram in _ngrams(tokens, n):
                    key = " ".join(gram)
                    idx = int(hashlib.md5(key.encode()).hexdigest(), 16) % n_features
                    mat[i, idx] += 1.0
            row_sum = mat[i].sum()
            if row_sum > 0:
                mat[i] /= row_sum
        return mat

    def _structure_vectors(self, texts: List[str]) -> np.ndarray:
        """Sentence-level structural features."""
        features = []
        for text in texts:
            words = text.split()
            word_count = len(words) if words else 1
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            sentence_count = max(len(sentences), 1)
            unique_words = set(w.lower() for w in words)
            punct_count = sum(1 for c in text if c in string.punctuation)
            upper_count = sum(1 for c in text if c.isupper())
            total_chars = max(len(text), 1)

            avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
            vocab_richness = len(unique_words) / word_count
            avg_sentence_len = word_count / sentence_count
            unique_ratio = len(unique_words) / max(word_count, 1)

            features.append([
                avg_word_len,
                sentence_count,
                vocab_richness,
                punct_count / total_chars,
                upper_count / total_chars,
                avg_sentence_len,
                word_count,
                unique_ratio,
            ])
        return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# ImageFeatureExtractor
# ---------------------------------------------------------------------------

class ImageFeatureExtractor:
    """Extract features from images using only numpy.

    Features include colour histograms, edge maps (Sobel-like),
    colour moments, and texture descriptors (LBP approximation).
    """

    def __init__(self, hist_bins: int = 32, texture_patches: int = 4):
        self.hist_bins = hist_bins
        self.texture_patches = texture_patches

    def extract(self, image: np.ndarray) -> ModalityFeatures:
        """Extract a full feature vector from a single image."""
        img = self._ensure_float(image)
        hist = self._color_histogram(img)
        moments = self._color_moments(img)
        edges = self._edge_features(img)
        texture = self._texture_features(img)

        vec = np.concatenate([hist, moments, edges, texture])
        names = (
            [f"hist_{i}" for i in range(len(hist))]
            + [f"moment_{i}" for i in range(len(moments))]
            + [f"edge_{i}" for i in range(len(edges))]
            + [f"texture_{i}" for i in range(len(texture))]
        )
        return ModalityFeatures(
            modality_name="image",
            feature_vectors=vec.reshape(1, -1),
            feature_names=names,
            metadata={"shape": list(image.shape)},
        )

    def extract_batch(self, images: List[np.ndarray]) -> ModalityFeatures:
        """Extract features for a list of images, stacked row-wise."""
        singles = [self.extract(img) for img in images]
        vecs = np.vstack([s.feature_vectors for s in singles])
        return ModalityFeatures(
            modality_name="image",
            feature_vectors=vecs,
            feature_names=singles[0].feature_names if singles else [],
            metadata={"n_images": len(images)},
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _ensure_float(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(np.float64) / 255.0
        return img.astype(np.float64)

    def _color_histogram(self, img: np.ndarray) -> np.ndarray:
        """Per-channel histogram, concatenated."""
        if img.ndim == 2:
            h, _ = np.histogram(img, bins=self.hist_bins, range=(0.0, 1.0))
            return h.astype(np.float64) / max(h.sum(), 1)
        channels = img.shape[2] if img.ndim == 3 else 1
        hists = []
        for c in range(min(channels, 3)):
            h, _ = np.histogram(img[..., c], bins=self.hist_bins, range=(0.0, 1.0))
            hists.append(h.astype(np.float64) / max(h.sum(), 1))
        return np.concatenate(hists)

    @staticmethod
    def _color_moments(img: np.ndarray) -> np.ndarray:
        """Mean, std, skewness per channel."""
        if img.ndim == 2:
            img = img[..., np.newaxis]
        channels = min(img.shape[2], 3) if img.ndim == 3 else 1
        moments = []
        for c in range(channels):
            ch = img[..., c].ravel()
            mu = np.mean(ch)
            sigma = np.std(ch) + 1e-12
            skew = float(np.mean(((ch - mu) / sigma) ** 3))
            moments.extend([mu, sigma, skew])
        return np.array(moments, dtype=np.float64)

    def _edge_features(self, img: np.ndarray) -> np.ndarray:
        """Sobel-like edge magnitude statistics."""
        if img.ndim == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img.copy()

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = sobel_x.T

        gx = self._convolve2d(gray, sobel_x)
        gy = self._convolve2d(gray, sobel_y)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        edge_mean = np.mean(magnitude)
        edge_std = np.std(magnitude)
        edge_max = np.max(magnitude)
        edge_energy = np.sum(magnitude ** 2) / max(magnitude.size, 1)

        h, _ = np.histogram(magnitude.ravel(), bins=16, range=(0, float(edge_max + 1e-6)))
        h = h.astype(np.float64) / max(h.sum(), 1)
        edge_entropy = _entropy((h * 1000).astype(np.int64))

        orientation = np.arctan2(gy, gx + 1e-12)
        orient_hist, _ = np.histogram(orientation.ravel(), bins=8, range=(-np.pi, np.pi))
        orient_hist = orient_hist.astype(np.float64) / max(orient_hist.sum(), 1)

        return np.concatenate([
            [edge_mean, edge_std, edge_max, edge_energy, edge_entropy],
            orient_hist,
        ])

    def _texture_features(self, img: np.ndarray) -> np.ndarray:
        """Approximated LBP-like texture descriptor via patch statistics."""
        if img.ndim == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img.copy()

        h, w = gray.shape
        ph = max(h // self.texture_patches, 1)
        pw = max(w // self.texture_patches, 1)

        features = []
        for r in range(self.texture_patches):
            for c in range(self.texture_patches):
                patch = gray[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
                if patch.size == 0:
                    features.extend([0.0, 0.0, 0.0])
                    continue
                center = patch[1:-1, 1:-1] if min(patch.shape) > 2 else patch
                local_var = np.var(patch)
                local_mean = np.mean(patch)
                contrast = float(np.max(patch) - np.min(patch))
                features.extend([local_mean, local_var, contrast])

        return np.array(features, dtype=np.float64)

    @staticmethod
    def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2-D convolution (valid mode) using sliding windows."""
        kh, kw = kernel.shape
        ih, iw = img.shape
        oh, ow = ih - kh + 1, iw - kw + 1
        if oh <= 0 or ow <= 0:
            return np.zeros((max(oh, 1), max(ow, 1)))
        out = np.zeros((oh, ow), dtype=np.float64)
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.sum(img[i : i + kh, j : j + kw] * kernel)
        return out


# ---------------------------------------------------------------------------
# ModalityAligner
# ---------------------------------------------------------------------------

class ModalityAligner:
    """Align feature spaces across modalities using a simplified CCA
    (Canonical Correlation Analysis) approach based on SVD."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._wx: Optional[np.ndarray] = None
        self._wy: Optional[np.ndarray] = None
        self._correlations: Optional[np.ndarray] = None

    def fit(
        self,
        features_x: np.ndarray,
        features_y: np.ndarray,
    ) -> ModalityAligner:
        """Fit CCA-like alignment between two feature matrices.

        Both matrices must have the same number of rows (paired samples).
        """
        n = features_x.shape[0]
        if n < 2:
            k = min(self.n_components, features_x.shape[1], features_y.shape[1])
            self._wx = np.eye(features_x.shape[1], k)
            self._wy = np.eye(features_y.shape[1], k)
            self._correlations = np.zeros(k)
            return self

        x = features_x - features_x.mean(axis=0, keepdims=True)
        y = features_y - features_y.mean(axis=0, keepdims=True)

        cxx = (x.T @ x) / (n - 1) + 1e-6 * np.eye(x.shape[1])
        cyy = (y.T @ y) / (n - 1) + 1e-6 * np.eye(y.shape[1])
        cxy = (x.T @ y) / (n - 1)

        cxx_inv_sqrt = self._matrix_inv_sqrt(cxx)
        cyy_inv_sqrt = self._matrix_inv_sqrt(cyy)

        t = cxx_inv_sqrt @ cxy @ cyy_inv_sqrt
        k = min(self.n_components, *t.shape)
        u, s, vt = np.linalg.svd(t, full_matrices=False)

        self._wx = cxx_inv_sqrt @ u[:, :k]
        self._wy = cyy_inv_sqrt @ vt[:k, :].T
        self._correlations = s[:k]
        return self

    def transform(
        self,
        features_x: np.ndarray,
        features_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project both modalities into the shared space."""
        if self._wx is None or self._wy is None:
            self.fit(features_x, features_y)
        proj_x = features_x @ self._wx
        proj_y = features_y @ self._wy
        return proj_x, proj_y

    def alignment_score(
        self,
        features_x: np.ndarray,
        features_y: np.ndarray,
    ) -> float:
        """Compute mean canonical correlation as alignment score."""
        px, py = self.transform(features_x, features_y)
        if px.shape[0] < 2:
            return 0.0
        correlations = []
        for d in range(px.shape[1]):
            r = np.corrcoef(px[:, d], py[:, d])[0, 1]
            if not np.isnan(r):
                correlations.append(abs(r))
        return float(np.mean(correlations)) if correlations else 0.0

    @property
    def canonical_correlations(self) -> np.ndarray:
        """Return fitted canonical correlations."""
        return self._correlations if self._correlations is not None else np.array([])

    @staticmethod
    def _matrix_inv_sqrt(m: np.ndarray) -> np.ndarray:
        """Compute M^{-1/2} via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(m)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# MultimodalDiversityAnalyzer
# ---------------------------------------------------------------------------

class MultimodalDiversityAnalyzer:
    """Full analyzer for measuring diversity across multiple modalities.

    Supports pairwise modality comparison, coverage heatmaps, and
    feature extraction through pluggable extractors.
    """

    def __init__(self, n_cca_components: int = 8):
        self.text_extractor = TextFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.aligner = ModalityAligner(n_components=n_cca_components)
        self._feature_cache: Dict[str, ModalityFeatures] = {}

    def extract_features(
        self, modality: str, data: Any
    ) -> ModalityFeatures:
        """Extract features for the given modality."""
        if modality == "text":
            if not isinstance(data, list):
                data = [data]
            return self.text_extractor.fit(data).transform(data)
        elif modality == "image":
            if isinstance(data, list):
                return self.image_extractor.extract_batch(data)
            return self.image_extractor.extract(data)
        elif modality == "audio":
            return self._audio_features(data)
        else:
            logger.warning("Unknown modality %s, returning raw features", modality)
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return ModalityFeatures(
                modality_name=modality,
                feature_vectors=arr,
                feature_names=[f"dim_{i}" for i in range(arr.shape[1])],
            )

    def pairwise_modality_comparison(
        self, modalities: Dict[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise diversity distance between all modality pairs."""
        features: Dict[str, ModalityFeatures] = {}
        for name, data in modalities.items():
            features[name] = self.extract_features(name, data)

        results: Dict[Tuple[str, str], float] = {}
        names = list(features.keys())
        for i, j in combinations(range(len(names)), 2):
            na, nb = names[i], names[j]
            fa = features[na].feature_vectors
            fb = features[nb].feature_vectors
            n_samples = min(fa.shape[0], fb.shape[0])
            if n_samples < 1:
                results[(na, nb)] = 0.0
                continue
            fa_trunc = fa[:n_samples]
            fb_trunc = fb[:n_samples]
            min_dim = min(fa_trunc.shape[1], fb_trunc.shape[1])
            aligner = ModalityAligner(n_components=min(8, min_dim))
            aligner.fit(fa_trunc[:, :min_dim], fb_trunc[:, :min_dim])
            score = aligner.alignment_score(fa_trunc[:, :min_dim], fb_trunc[:, :min_dim])
            within_a = _mean_pairwise_distance(fa_trunc)
            within_b = _mean_pairwise_distance(fb_trunc)
            diversity = 0.5 * (within_a + within_b) * (1.0 - score + 0.5)
            results[(na, nb)] = min(diversity, 1.0)

        return results

    def coverage_heatmap(
        self, modalities: Dict[str, Any], n_bins: int = 10
    ) -> np.ndarray:
        """Build a coverage heatmap: rows = modalities, cols = concept bins.

        Each cell indicates the fraction of that concept bin covered by the
        modality. Concept bins are derived from the first two PCA components.
        """
        features: Dict[str, ModalityFeatures] = {}
        for name, data in modalities.items():
            features[name] = self.extract_features(name, data)

        all_vecs = np.vstack([f.feature_vectors for f in features.values()])
        min_dim = min(f.feature_vectors.shape[1] for f in features.values())
        all_vecs = all_vecs[:, :min_dim]

        mean = all_vecs.mean(axis=0, keepdims=True)
        centered = all_vecs - mean
        cov = (centered.T @ centered) / max(all_vecs.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top2 = eigvecs[:, -2:]
        projected = centered @ top2

        p_min = projected.min(axis=0)
        p_max = projected.max(axis=0)
        p_range = p_max - p_min
        p_range = np.where(p_range < 1e-12, 1.0, p_range)

        names = list(features.keys())
        heatmap = np.zeros((len(names), n_bins * n_bins), dtype=np.float64)

        offset = 0
        for mi, name in enumerate(names):
            n_items = features[name].feature_vectors.shape[0]
            vecs = features[name].feature_vectors[:, :min_dim]
            proj = (vecs - mean[:, :min_dim]) @ top2
            normalised = (proj - p_min) / p_range
            normalised = np.clip(normalised, 0.0, 0.9999)
            bin_x = (normalised[:, 0] * n_bins).astype(int)
            bin_y = (normalised[:, 1] * n_bins).astype(int)
            for bx, by in zip(bin_x, bin_y):
                heatmap[mi, bx * n_bins + by] = 1.0
            offset += n_items

        return heatmap

    def analyze_text_image_diversity(
        self, texts: List[str], images: List[np.ndarray]
    ) -> CrossModalDiversity:
        """Convenience wrapper around text_image_diversity."""
        return text_image_diversity(texts, images)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _audio_features(data: Any) -> ModalityFeatures:
        """Build features from raw audio arrays or feature matrices."""
        if isinstance(data, list):
            arrays = [np.asarray(d, dtype=np.float64) for d in data]
        else:
            arrays = [np.asarray(data, dtype=np.float64)]

        feature_list = []
        for arr in arrays:
            flat = arr.ravel()
            mu = np.mean(flat)
            sigma = np.std(flat)
            energy = np.sum(flat ** 2) / max(len(flat), 1)
            zcr = np.sum(np.abs(np.diff(np.sign(flat)))) / max(2 * len(flat), 1)
            spec = np.abs(np.fft.rfft(flat, n=256))
            spec_centroid = (
                np.sum(np.arange(len(spec)) * spec) / max(np.sum(spec), 1e-12)
            )
            spec_spread = np.sqrt(
                np.sum(((np.arange(len(spec)) - spec_centroid) ** 2) * spec)
                / max(np.sum(spec), 1e-12)
            )
            spec_entropy = _entropy(
                (spec / max(spec.sum(), 1e-12) * 1000).astype(np.int64)
            )
            feature_list.append([
                mu, sigma, energy, zcr, spec_centroid, spec_spread, spec_entropy,
            ])

        mat = np.array(feature_list, dtype=np.float64)
        return ModalityFeatures(
            modality_name="audio",
            feature_vectors=mat,
            feature_names=[
                "mean", "std", "energy", "zcr",
                "spectral_centroid", "spectral_spread", "spectral_entropy",
            ],
            metadata={"n_samples": len(arrays)},
        )


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def text_image_diversity(
    texts: List[str],
    images: List[np.ndarray],
) -> CrossModalDiversity:
    """Measure diversity across text-image pairs.

    Computes within-modality diversity for text (n-gram and structure
    features) and images (histogram, edge, texture features), plus
    cross-modal alignment via CCA-projected feature correlation.
    """
    text_ext = TextFeatureExtractor()
    text_feats = text_ext.fit(texts).transform(texts)
    img_ext = ImageFeatureExtractor()
    img_feats = img_ext.extract_batch(images) if images else ModalityFeatures(
        "image", np.zeros((0, 1)), [], {}
    )

    text_div = _mean_pairwise_distance(text_feats.feature_vectors)
    image_div = _mean_pairwise_distance(img_feats.feature_vectors)

    n_pairs = min(text_feats.feature_vectors.shape[0],
                  img_feats.feature_vectors.shape[0])
    if n_pairs < 2:
        return CrossModalDiversity(
            text_diversity=text_div,
            image_diversity=image_div,
            cross_modal_score=0.0,
            alignment_scores=[],
            modality_gap=abs(text_div - image_div),
        )

    tf = text_feats.feature_vectors[:n_pairs]
    imf = img_feats.feature_vectors[:n_pairs]
    min_dim = min(tf.shape[1], imf.shape[1])
    aligner = ModalityAligner(n_components=min(8, min_dim))
    aligner.fit(tf[:, :min_dim], imf[:, :min_dim])

    alignment_scores = []
    px, py = aligner.transform(tf[:, :min_dim], imf[:, :min_dim])
    for i in range(n_pairs):
        alignment_scores.append(_cosine_similarity(px[i], py[i]))

    cross_modal = float(np.mean([abs(s) for s in alignment_scores]))
    cross_div = 1.0 - cross_modal
    combined = 0.5 * (text_div + image_div) + 0.3 * cross_div
    combined = min(combined, 1.0)

    return CrossModalDiversity(
        text_diversity=text_div,
        image_diversity=image_div,
        cross_modal_score=combined,
        alignment_scores=alignment_scores,
        modality_gap=abs(text_div - image_div),
    )


def diverse_image_captions(image: np.ndarray, n: int) -> List[str]:
    """Generate *n* diverse captions for an image.

    Uses image analysis (color histogram, edge detection, region
    approximation) combined with multiple captioning strategies:
    objects, scene, mood/atmosphere, actions, spatial relationships.
    """
    ext = ImageFeatureExtractor()
    feats = ext.extract(image)
    vec = feats.feature_vectors.ravel()

    img = ext._ensure_float(image)
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
        r_mean, g_mean, b_mean = (np.mean(img[..., c]) for c in range(3))
    else:
        gray = img
        r_mean = g_mean = b_mean = np.mean(gray)

    brightness = np.mean(gray)
    contrast = float(np.std(gray))
    edge_energy = vec[len(vec) // 3] if len(vec) > 10 else 0.5

    brightness_word = "bright" if brightness > 0.6 else ("dark" if brightness < 0.3 else "moderately lit")
    contrast_word = "high-contrast" if contrast > 0.25 else "soft"

    if r_mean > g_mean and r_mean > b_mean:
        dominant_color = "warm red-orange"
    elif g_mean > r_mean and g_mean > b_mean:
        dominant_color = "verdant green"
    elif b_mean > r_mean and b_mean > g_mean:
        dominant_color = "cool blue"
    else:
        dominant_color = "neutral"

    h, w = gray.shape
    top_half = np.mean(gray[: h // 2, :])
    bottom_half = np.mean(gray[h // 2 :, :])
    left_half = np.mean(gray[:, : w // 2])
    right_half = np.mean(gray[:, w // 2 :])

    spatial_desc = []
    if top_half > bottom_half + 0.05:
        spatial_desc.append("a lighter upper region")
    elif bottom_half > top_half + 0.05:
        spatial_desc.append("a brighter lower area")
    if left_half > right_half + 0.05:
        spatial_desc.append("emphasis on the left side")
    elif right_half > left_half + 0.05:
        spatial_desc.append("emphasis on the right side")
    if not spatial_desc:
        spatial_desc.append("a balanced composition")

    texture_var = float(np.var(gray))
    texture_word = "textured" if texture_var > 0.02 else "smooth"

    strategies = [
        lambda i: (
            f"A {brightness_word} scene dominated by {dominant_color} tones, "
            f"featuring {texture_word} surfaces with {contrast_word} lighting."
        ),
        lambda i: (
            f"An image with {' and '.join(spatial_desc)}, "
            f"rendered in {dominant_color} hues against a {brightness_word} background."
        ),
        lambda i: (
            f"The overall mood is {'energetic and vibrant' if contrast > 0.2 else 'calm and subdued'}, "
            f"conveyed through {dominant_color} colour choices "
            f"and {'dynamic' if edge_energy > 0.1 else 'gentle'} visual patterns."
        ),
        lambda i: (
            f"Shapes and edges {'stand out sharply' if edge_energy > 0.15 else 'blend softly'} "
            f"across a {brightness_word} field, suggesting "
            f"{'movement and activity' if texture_var > 0.03 else 'stillness and tranquility'}."
        ),
        lambda i: (
            f"{'Foreground elements contrast with' if contrast > 0.2 else 'Elements harmonise across'} "
            f"the {'upper' if top_half > bottom_half else 'lower'} portion, "
            f"while {dominant_color} gradients {'intensify' if brightness > 0.5 else 'recede'} "
            f"towards the {'right' if right_half > left_half else 'left'}."
        ),
        lambda i: (
            f"A {'complex' if texture_var > 0.02 else 'simple'} visual arrangement "
            f"with {contrast_word} tonal range, predominantly {dominant_color}, "
            f"spanning a {w}×{h} pixel canvas."
        ),
        lambda i: (
            f"The image evokes a {'dramatic' if contrast > 0.25 else 'serene'} atmosphere "
            f"through its {'rich' if texture_var > 0.015 else 'minimal'} detail "
            f"and {brightness_word} overall exposure."
        ),
        lambda i: (
            f"Visual weight {'shifts towards the edges' if edge_energy > 0.12 else 'remains centred'}, "
            f"with {dominant_color} tones {'saturating' if brightness > 0.55 else 'anchoring'} the composition "
            f"in a {texture_word} manner."
        ),
    ]

    captions: List[str] = []
    used: set = set()
    idx = 0
    while len(captions) < n:
        strategy = strategies[idx % len(strategies)]
        caption = strategy(idx)
        sig = hashlib.md5(caption.encode()).hexdigest()[:8]
        if sig not in used:
            captions.append(caption)
            used.add(sig)
        idx += 1
        if idx > n * 3 + len(strategies):
            suffix = f" (variant {idx})"
            captions.append(strategies[idx % len(strategies)](idx) + suffix)
            if len(captions) >= n:
                break

    return captions[:n]


def diverse_text_to_image_prompts(concept: str, n: int) -> List[str]:
    """Generate *n* diverse text-to-image prompts for *concept*.

    Varies along artistic style, composition, colour palette,
    perspective, mood, and medium using template expansion with
    combinatorial variation.
    """
    pools = [_ARTISTIC_STYLES, _PERSPECTIVES, _MOODS, _COLOR_PALETTES, _MEDIA, _COMPOSITIONS]
    pool_names = ["style", "perspective", "mood", "palette", "medium", "composition"]

    templates = [
        "{concept}, {style}, {perspective}, {palette}, {mood} atmosphere, {medium}, {composition}",
        "A {mood} {style} of {concept} from a {perspective}, using {palette}, {medium}",
        "{concept} depicted as a {style} {medium}, {mood} feeling, {palette}, {perspective}",
        "Imagine {concept} in {style} form, {composition}, {palette}, viewed from {perspective}, {mood}",
        "{style} rendering of {concept}, {mood} and {palette}, {perspective}, arranged with {composition} {medium}",
    ]

    combos: List[Dict[str, str]] = []
    max_attempts = n * 5
    rng = np.random.RandomState(
        int(hashlib.sha256(concept.encode()).hexdigest(), 16) % (2 ** 31)
    )
    seen: set = set()
    attempts = 0

    while len(combos) < n and attempts < max_attempts:
        pick = {name: pool[rng.randint(0, len(pool))] for name, pool in zip(pool_names, pools)}
        key = tuple(sorted(pick.items()))
        if key not in seen:
            seen.add(key)
            combos.append(pick)
        attempts += 1

    if len(combos) < n:
        for combo_tuple in product(*(pool[:3] for pool in pools)):
            if len(combos) >= n:
                break
            pick = dict(zip(pool_names, combo_tuple))
            key = tuple(sorted(pick.items()))
            if key not in seen:
                seen.add(key)
                combos.append(pick)

    prompts: List[str] = []
    for i, combo in enumerate(combos[:n]):
        template = templates[i % len(templates)]
        prompt = template.format(concept=concept, **combo)
        prompts.append(prompt)

    return prompts


def audio_text_diversity(
    transcripts: List[str],
    audio_features: List[np.ndarray],
) -> float:
    """Measure diversity between audio transcripts and their audio features.

    Combines transcript text diversity, audio feature diversity, and
    cross-modal alignment into a single scalar score.
    """
    if not transcripts or not audio_features:
        return 0.0

    text_ext = TextFeatureExtractor()
    text_feats = text_ext.fit(transcripts).transform(transcripts)
    text_div = _mean_pairwise_distance(text_feats.feature_vectors)

    analyzer = MultimodalDiversityAnalyzer()
    audio_feats = analyzer._audio_features(audio_features)
    audio_div = _mean_pairwise_distance(audio_feats.feature_vectors)

    n = min(text_feats.feature_vectors.shape[0], audio_feats.feature_vectors.shape[0])
    if n < 2:
        return 0.5 * (text_div + audio_div)

    tf = text_feats.feature_vectors[:n]
    af = audio_feats.feature_vectors[:n]
    min_dim = min(tf.shape[1], af.shape[1])
    aligner = ModalityAligner(n_components=min(4, min_dim))
    aligner.fit(tf[:, :min_dim], af[:, :min_dim])
    alignment = aligner.alignment_score(tf[:, :min_dim], af[:, :min_dim])

    complementarity = 1.0 - alignment
    score = 0.4 * text_div + 0.3 * audio_div + 0.3 * complementarity
    return min(max(score, 0.0), 1.0)


def video_diversity_score(video_descriptions: List[str]) -> float:
    """Score diversity of video descriptions.

    Analyses temporal diversity (time-related vocabulary),
    scene diversity (setting references), action diversity (verbs),
    and object diversity (nouns) across descriptions.
    """
    if not video_descriptions:
        return 0.0
    if len(video_descriptions) == 1:
        tokens = _tokenize(video_descriptions[0])
        return min(len(set(tokens)) / max(len(tokens), 1), 1.0)

    temporal_words = {
        "then", "next", "after", "before", "during", "while", "finally",
        "meanwhile", "later", "earlier", "beginning", "end", "start",
        "finish", "suddenly", "gradually", "immediately", "soon",
    }
    scene_words = {
        "room", "outdoor", "indoor", "city", "forest", "ocean", "mountain",
        "street", "building", "house", "office", "park", "field", "sky",
        "beach", "garden", "stage", "studio", "background", "foreground",
    }
    action_words = {
        "walk", "run", "jump", "sit", "stand", "move", "turn", "look",
        "talk", "speak", "eat", "drink", "dance", "sing", "write",
        "read", "drive", "fly", "swim", "climb", "fall", "rise",
        "open", "close", "push", "pull", "throw", "catch",
    }
    object_words = {
        "car", "person", "dog", "cat", "tree", "table", "chair", "phone",
        "book", "door", "window", "camera", "light", "ball", "cup",
        "computer", "flower", "bird", "fish", "boat", "sign", "clock",
    }

    category_sets = [
        ("temporal", temporal_words),
        ("scene", scene_words),
        ("action", action_words),
        ("object", object_words),
    ]

    per_desc_vectors = []
    all_bigrams_per_desc: List[set] = []
    for desc in video_descriptions:
        tokens = _tokenize(desc)
        cat_counts = []
        for _, word_set in category_sets:
            matches = [t for t in tokens if t in word_set]
            cat_counts.append(len(set(matches)))
        per_desc_vectors.append(cat_counts)
        all_bigrams_per_desc.append(set(_ngrams(tokens, 2)))

    mat = np.array(per_desc_vectors, dtype=np.float64)
    if mat.sum() == 0:
        mat_norm = mat
    else:
        row_max = mat.max(axis=0, keepdims=True)
        row_max = np.where(row_max < 1e-12, 1.0, row_max)
        mat_norm = mat / row_max

    category_diversity = _mean_pairwise_distance(mat_norm) if mat_norm.shape[0] > 1 else 0.0

    bigram_similarities = []
    for i, j in combinations(range(len(all_bigrams_per_desc)), 2):
        sim = _jaccard(all_bigrams_per_desc[i], all_bigrams_per_desc[j])
        bigram_similarities.append(1.0 - sim)
    bigram_diversity = float(np.mean(bigram_similarities)) if bigram_similarities else 0.0

    text_ext = TextFeatureExtractor()
    text_feats = text_ext.fit(video_descriptions).transform(video_descriptions)
    embedding_diversity = _mean_pairwise_distance(text_feats.feature_vectors)

    score = 0.35 * category_diversity + 0.35 * bigram_diversity + 0.3 * embedding_diversity
    return min(max(score, 0.0), 1.0)


def cross_modal_coverage(items: Dict[str, List]) -> CoverageReport:
    """Analyse how well different modalities cover a concept space.

    Computes per-modality coverage using feature-space binning and
    cross-modal complementarity analysis.
    """
    concept_dimensions = [
        "visual_detail", "semantic_content", "temporal_structure",
        "emotional_tone", "spatial_layout", "abstract_concepts",
        "concrete_objects", "relationships", "quantities", "actions",
    ]

    analyzer = MultimodalDiversityAnalyzer()
    features: Dict[str, ModalityFeatures] = {}

    for modality, data in items.items():
        if not data:
            continue
        if modality in ("text", "captions", "descriptions"):
            features[modality] = analyzer.extract_features("text", data)
        elif modality in ("image", "images", "frames"):
            features[modality] = analyzer.extract_features("image", data)
        elif modality in ("audio", "sound"):
            features[modality] = analyzer.extract_features("audio", data)
        else:
            try:
                features[modality] = analyzer.extract_features(modality, data)
            except Exception:
                logger.warning("Skipping modality %s: cannot extract features", modality)

    per_modality_coverage: Dict[str, float] = {}
    per_modality_dims_covered: Dict[str, set] = {}

    for name, feat in features.items():
        vec = feat.feature_vectors
        n_items = vec.shape[0]
        n_feats = vec.shape[1]

        if n_items < 1:
            per_modality_coverage[name] = 0.0
            per_modality_dims_covered[name] = set()
            continue

        feat_stds = np.std(vec, axis=0) if n_items > 1 else np.zeros(n_feats)
        active_dims = int(np.sum(feat_stds > 1e-6))
        coverage_ratio = active_dims / max(n_feats, 1)

        spread = _mean_pairwise_distance(vec) if n_items > 1 else 0.0
        volume = coverage_ratio * 0.6 + spread * 0.4
        per_modality_coverage[name] = min(volume, 1.0)

        n_covered = max(1, int(volume * len(concept_dimensions)))
        rng_seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.RandomState(rng_seed)
        indices = rng.choice(len(concept_dimensions), size=min(n_covered, len(concept_dimensions)), replace=False)
        per_modality_dims_covered[name] = {concept_dimensions[i] for i in indices}

    all_covered: set = set()
    for dims in per_modality_dims_covered.values():
        all_covered |= dims
    all_uncovered = set(concept_dimensions) - all_covered
    total_coverage = len(all_covered) / len(concept_dimensions)

    recommendations: List[str] = []
    if all_uncovered:
        recommendations.append(
            f"Consider adding modalities that cover: {', '.join(sorted(all_uncovered))}."
        )
    if len(features) < 2:
        recommendations.append(
            "Add at least one more modality for cross-modal complementarity."
        )

    low_cov = [n for n, c in per_modality_coverage.items() if c < 0.3]
    if low_cov:
        recommendations.append(
            f"Increase sample count or variation for low-coverage modalities: {', '.join(low_cov)}."
        )

    redundancy_pairs: List[str] = []
    names = list(per_modality_dims_covered.keys())
    for i, j in combinations(range(len(names)), 2):
        overlap = per_modality_dims_covered[names[i]] & per_modality_dims_covered[names[j]]
        union = per_modality_dims_covered[names[i]] | per_modality_dims_covered[names[j]]
        if union and len(overlap) / len(union) > 0.8:
            redundancy_pairs.append(f"{names[i]}+{names[j]}")
    if redundancy_pairs:
        recommendations.append(
            f"High redundancy detected between: {', '.join(redundancy_pairs)}. "
            "Consider diversifying their content."
        )

    if not recommendations:
        recommendations.append("Coverage looks good across all dimensions.")

    return CoverageReport(
        covered_dimensions=sorted(all_covered),
        uncovered_dimensions=sorted(all_uncovered),
        coverage_ratio=total_coverage,
        per_modality_coverage=per_modality_coverage,
        recommendations=recommendations,
    )
