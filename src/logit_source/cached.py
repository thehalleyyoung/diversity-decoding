"""
CachedLogitSource — Content-addressed caching layer for logit distributions.

Wraps any LogitSource with a multi-tier caching system:
  - In-memory LRU/LFU/FIFO cache with size-based eviction
  - Optional disk-backed persistence with compression
  - Cache warming for predictable prefix sets
  - Thread-safe operation with per-key locking
  - Detailed statistics tracking (hits, misses, evictions, hit rate)

Usage::

    from src.logit_source.base import LogitSource
    from src.logit_source.cached import CachedLogitSource, CacheConfig

    config = CacheConfig(max_entries=5000, eviction_policy="lru")
    cached = CachedLogitSource(wrapped=live_source, cache_config=config)
    logits = cached.get_logits(input_ids)
    print(cached.cache_stats())
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import struct
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.logit_source.base import LogitSource, LogitSourceConfig, LogitBatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=False)
class CacheKey:
    """Content-addressed key for cached logit lookups.

    Built from the model name, quantization scheme, and the full
    input-token sequence so that identical prompts always resolve to
    the same cache slot.
    """

    input_ids: Tuple[int, ...]
    model_name: str
    quantization: Optional[str] = None

    def __post_init__(self) -> None:
        # Guarantee input_ids is a tuple for hashability.
        if not isinstance(self.input_ids, tuple):
            object.__setattr__(self, "input_ids", tuple(self.input_ids))

    def __hash__(self) -> int:
        return hash((self.input_ids, self.model_name, self.quantization))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CacheKey):
            return NotImplemented
        return (
            self.input_ids == other.input_ids
            and self.model_name == other.model_name
            and self.quantization == other.quantization
        )

    def content_hash(self) -> str:
        """Return a hex-digest suitable for file-system keys."""
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))
        if self.quantization:
            h.update(self.quantization.encode("utf-8"))
        h.update(struct.pack(f">{len(self.input_ids)}i", *self.input_ids))
        return h.hexdigest()


@dataclass
class CacheEntry:
    """A single cached logit distribution together with bookkeeping metadata."""

    logits: np.ndarray
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size_bytes == 0:
            self.size_bytes = self._compute_size()

    def _compute_size(self) -> int:
        """Estimate the in-memory footprint of this entry."""
        base = self.logits.nbytes
        # Overhead for the dataclass fields + dict.
        base += 128
        return base

    def touch(self) -> None:
        """Record an access (updates timestamp and counter)."""
        self.timestamp = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Tuning knobs for the caching layer."""

    max_entries: int = 10_000
    max_size_bytes: int = 1 * 1024 * 1024 * 1024  # 1 GB
    eviction_policy: str = "lru"  # "lru" | "lfu" | "fifo"
    persist_to_disk: bool = False
    disk_cache_dir: Optional[str] = None
    compression: bool = True
    ttl_seconds: Optional[int] = None  # None ⇒ no expiry

    def __post_init__(self) -> None:
        valid_policies = {"lru", "lfu", "fifo"}
        if self.eviction_policy not in valid_policies:
            raise ValueError(
                f"eviction_policy must be one of {valid_policies}, "
                f"got {self.eviction_policy!r}"
            )
        if self.max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if self.max_size_bytes < 1:
            raise ValueError("max_size_bytes must be >= 1")


# ---------------------------------------------------------------------------
# LRUCache — thread-safe, size-aware in-memory cache
# ---------------------------------------------------------------------------


class LRUCache:
    """Thread-safe in-memory cache with pluggable eviction policies.

    Supports three eviction strategies:

    * **lru** — least-recently-used (default).  Implemented with
      :class:`collections.OrderedDict` so that ``move_to_end`` marks
      an access and eviction pops from the front.
    * **lfu** — least-frequently-used.  On eviction the entry with the
      lowest ``access_count`` is removed (ties broken by oldest
      timestamp).
    * **fifo** — first-in, first-out.  Insertion order is preserved by
      the ``OrderedDict``; eviction always pops from the front
      regardless of access pattern.
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._data: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size_bytes: int = 0

        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._inserts: int = 0

    # -- public API ---------------------------------------------------------

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Retrieve a cache entry, or ``None`` on a miss."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None

            # TTL check.
            if self._is_expired(entry):
                self._remove(key)
                self._misses += 1
                return None

            entry.touch()

            # LRU bookkeeping: move accessed key to the *end*.
            if self._config.eviction_policy == "lru":
                self._data.move_to_end(key)

            self._hits += 1
            return entry

    def put(self, key: CacheKey, entry: CacheEntry) -> None:
        """Insert *entry* under *key*, evicting as needed."""
        with self._lock:
            # If the key already exists, remove the old entry first.
            if key in self._data:
                self._remove(key)

            # Evict until we have room.
            while self._needs_eviction(entry.size_bytes):
                if not self._data:
                    break
                self.evict()

            self._data[key] = entry
            self._current_size_bytes += entry.size_bytes
            self._inserts += 1

            # For LRU, newly inserted entries go to the *end* (most recent).
            if self._config.eviction_policy == "lru":
                self._data.move_to_end(key)

    def evict(self) -> Optional[CacheKey]:
        """Remove one entry according to the active eviction policy.

        Returns the evicted key, or ``None`` if the cache is empty.
        """
        with self._lock:
            if not self._data:
                return None

            victim_key = self._select_victim()
            self._remove(victim_key)
            self._evictions += 1
            return victim_key

    def remove(self, key: CacheKey) -> bool:
        """Explicitly remove *key*.  Returns ``True`` if it was present."""
        with self._lock:
            if key not in self._data:
                return False
            self._remove(key)
            return True

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._data.clear()
            self._current_size_bytes = 0

    # -- capacity helpers ---------------------------------------------------

    def size(self) -> int:
        """Current size in bytes."""
        with self._lock:
            return self._current_size_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: CacheKey) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            if self._is_expired(self._data[key]):
                self._remove(key)
                return False
            return True

    def keys(self) -> List[CacheKey]:
        with self._lock:
            return list(self._data.keys())

    def values(self) -> List[CacheEntry]:
        with self._lock:
            return list(self._data.values())

    def items(self) -> List[Tuple[CacheKey, CacheEntry]]:
        with self._lock:
            return list(self._data.items())

    # -- statistics ---------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
                "evictions": self._evictions,
                "inserts": self._inserts,
                "entries": len(self._data),
                "size_bytes": self._current_size_bytes,
                "max_entries": self._config.max_entries,
                "max_size_bytes": self._config.max_size_bytes,
                "eviction_policy": self._config.eviction_policy,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._inserts = 0

    # -- internals ----------------------------------------------------------

    def _needs_eviction(self, incoming_bytes: int) -> bool:
        if len(self._data) >= self._config.max_entries:
            return True
        if self._current_size_bytes + incoming_bytes > self._config.max_size_bytes:
            return True
        return False

    def _select_victim(self) -> CacheKey:
        """Pick the entry to evict based on the configured policy."""
        policy = self._config.eviction_policy

        if policy == "lru" or policy == "fifo":
            # Front of the OrderedDict is the oldest / least-recently-used.
            return next(iter(self._data))

        if policy == "lfu":
            # Least-frequently-used; ties broken by oldest timestamp.
            return min(
                self._data,
                key=lambda k: (
                    self._data[k].access_count,
                    self._data[k].timestamp,
                ),
            )

        # Fallback (should not happen after validation).
        return next(iter(self._data))

    def _remove(self, key: CacheKey) -> None:
        entry = self._data.pop(key, None)
        if entry is not None:
            self._current_size_bytes -= entry.size_bytes
            if self._current_size_bytes < 0:
                self._current_size_bytes = 0

    def _is_expired(self, entry: CacheEntry) -> bool:
        if self._config.ttl_seconds is None:
            return False
        return (time.time() - entry.timestamp) > self._config.ttl_seconds

    def expire_stale(self) -> int:
        """Remove all expired entries.  Returns the number removed."""
        if self._config.ttl_seconds is None:
            return 0
        with self._lock:
            stale = [k for k, v in self._data.items() if self._is_expired(v)]
            for k in stale:
                self._remove(k)
            return len(stale)


# ---------------------------------------------------------------------------
# DiskCache — content-addressed file-based persistence
# ---------------------------------------------------------------------------


class DiskCache:
    """Content-addressed file-based cache with optional zlib compression.

    Each entry is stored as a single file whose name is derived from
    :meth:`CacheKey.content_hash`.  An accompanying ``.meta`` JSON file
    holds the cache-entry metadata so that TTL and statistics survive
    across restarts.
    """

    _MAGIC = b"DDAC"  # Diversity-Decoding Arena Cache
    _VERSION = 1

    def __init__(
        self,
        cache_dir: Union[str, Path],
        compression: bool = True,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._compression = compression
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Load an entry from disk, or ``None`` if absent / expired."""
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)

        if not data_path.exists():
            return None

        with self._lock:
            try:
                raw = data_path.read_bytes()
                meta = self._read_meta(meta_path)

                # TTL check.
                if self._ttl_seconds is not None and meta:
                    age = time.time() - meta.get("timestamp", 0.0)
                    if age > self._ttl_seconds:
                        self.delete(key)
                        return None

                logits = self._deserialize(raw)

                entry = CacheEntry(
                    logits=logits,
                    timestamp=meta.get("timestamp", time.time()),
                    access_count=meta.get("access_count", 0),
                    size_bytes=meta.get("size_bytes", logits.nbytes),
                    metadata=meta.get("metadata", {}),
                )
                # Update access bookkeeping on disk.
                entry.touch()
                self._write_meta(meta_path, entry)
                return entry
            except Exception:
                logger.warning("Corrupt disk-cache entry for %s", key, exc_info=True)
                self.delete(key)
                return None

    def put(self, key: CacheKey, entry: CacheEntry) -> None:
        """Persist *entry* to disk."""
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)

        with self._lock:
            raw = self._serialize(entry.logits)
            data_path.write_bytes(raw)
            self._write_meta(meta_path, entry)

    def has(self, key: CacheKey) -> bool:
        """Check existence without deserialising."""
        data_path = self._data_path(key)
        if not data_path.exists():
            return False
        if self._ttl_seconds is not None:
            meta = self._read_meta(self._meta_path(key))
            if meta:
                age = time.time() - meta.get("timestamp", 0.0)
                if age > self._ttl_seconds:
                    self.delete(key)
                    return False
        return True

    def delete(self, key: CacheKey) -> bool:
        """Remove the entry for *key*.  Returns ``True`` if it existed."""
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)
        removed = False
        for p in (data_path, meta_path):
            try:
                p.unlink()
                removed = True
            except FileNotFoundError:
                pass
        return removed

    def cleanup(self, max_age_seconds: Optional[int] = None) -> int:
        """Remove entries older than *max_age_seconds*.

        If *max_age_seconds* is ``None``, uses the TTL configured at
        construction time.  Returns the number of entries removed.
        """
        ttl = max_age_seconds if max_age_seconds is not None else self._ttl_seconds
        if ttl is None:
            return 0

        now = time.time()
        removed = 0
        with self._lock:
            for meta_path in self._cache_dir.glob("*.meta"):
                try:
                    meta = self._read_meta(meta_path)
                    if meta and (now - meta.get("timestamp", 0.0)) > ttl:
                        data_path = meta_path.with_suffix(".dat")
                        for p in (data_path, meta_path):
                            try:
                                p.unlink()
                            except FileNotFoundError:
                                pass
                        removed += 1
                except Exception:
                    logger.debug("Skipping corrupt meta %s", meta_path, exc_info=True)
        return removed

    def total_size(self) -> int:
        """Aggregate byte-size of all ``.dat`` files."""
        total = 0
        for p in self._cache_dir.glob("*.dat"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def list_keys(self) -> List[str]:
        """Return the content-hash stems of every cached entry."""
        return [p.stem for p in self._cache_dir.glob("*.dat")]

    # -- serialization helpers ----------------------------------------------

    def _serialize(self, logits: np.ndarray) -> bytes:
        raw = pickle.dumps(logits, protocol=pickle.HIGHEST_PROTOCOL)
        if self._compression:
            raw = self._MAGIC + struct.pack(">B", self._VERSION) + zlib.compress(raw, level=6)
        else:
            raw = self._MAGIC + struct.pack(">B", self._VERSION) + raw
        return raw

    def _deserialize(self, data: bytes) -> np.ndarray:
        if data[:4] != self._MAGIC:
            raise ValueError("Invalid cache file magic bytes")
        version = struct.unpack(">B", data[4:5])[0]
        if version != self._VERSION:
            raise ValueError(f"Unsupported cache version {version}")
        payload = data[5:]
        if self._compression:
            payload = zlib.decompress(payload)
        return pickle.loads(payload)  # noqa: S301

    # -- path helpers -------------------------------------------------------

    def _data_path(self, key: CacheKey) -> Path:
        return self._cache_dir / f"{key.content_hash()}.dat"

    def _meta_path(self, key: CacheKey) -> Path:
        return self._cache_dir / f"{key.content_hash()}.meta"

    def _read_meta(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_meta(self, path: Path, entry: CacheEntry) -> None:
        meta = {
            "timestamp": entry.timestamp,
            "access_count": entry.access_count,
            "size_bytes": entry.size_bytes,
            "metadata": entry.metadata,
        }
        path.write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# CacheWarmer — proactively populate the cache
# ---------------------------------------------------------------------------


class CacheWarmer:
    """Pre-computes logit distributions for a set of known prefixes.

    Typical usage::

        warmer = CacheWarmer()
        warmer.warm(cached_source, prefixes=common_prompts, batch_size=32)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: int = 0
        self._completed: int = 0
        self._running: bool = False
        self._errors: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    # -- public API ---------------------------------------------------------

    def warm(
        self,
        source: "CachedLogitSource",
        prefixes: List[List[int]],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Warm the cache for every prefix in *prefixes*.

        Parameters
        ----------
        source:
            The :class:`CachedLogitSource` whose cache should be populated.
        prefixes:
            Token-ID sequences to pre-compute.
        batch_size:
            How many sequences to pass to ``get_logits_batch`` at once.

        Returns
        -------
        dict
            Summary statistics: total, completed, errors, elapsed_seconds.
        """
        with self._lock:
            self._total = len(prefixes)
            self._completed = 0
            self._errors = 0
            self._running = True
            self._start_time = time.time()
            self._end_time = None

        logger.info("Cache warming started: %d prefixes, batch_size=%d", len(prefixes), batch_size)

        try:
            for batch_start in range(0, len(prefixes), batch_size):
                batch = prefixes[batch_start : batch_start + batch_size]
                try:
                    # Force population by calling through the cached source.
                    source.get_logits_batch(batch)
                    with self._lock:
                        self._completed += len(batch)
                except Exception:
                    logger.warning(
                        "Cache-warm batch starting at %d failed", batch_start, exc_info=True
                    )
                    with self._lock:
                        self._errors += len(batch)

                logger.debug(
                    "Cache warming progress: %d/%d (errors: %d)",
                    self._completed,
                    self._total,
                    self._errors,
                )
        finally:
            with self._lock:
                self._running = False
                self._end_time = time.time()

        summary = self.progress()
        logger.info(
            "Cache warming complete: %d/%d succeeded (%.1f s)",
            summary["completed"],
            summary["total"],
            summary["elapsed_seconds"],
        )
        return summary

    def progress(self) -> Dict[str, Any]:
        """Return a snapshot of warming progress."""
        with self._lock:
            elapsed = 0.0
            if self._start_time is not None:
                end = self._end_time if self._end_time else time.time()
                elapsed = end - self._start_time
            return {
                "total": self._total,
                "completed": self._completed,
                "errors": self._errors,
                "running": self._running,
                "elapsed_seconds": round(elapsed, 3),
                "progress_pct": (
                    round(100.0 * self._completed / self._total, 1)
                    if self._total > 0
                    else 0.0
                ),
            }


# ---------------------------------------------------------------------------
# CachedLogitSource — main entry point
# ---------------------------------------------------------------------------


class CachedLogitSource(LogitSource):
    """A caching decorator around any :class:`LogitSource`.

    Caching is content-addressed: the cache key is derived from the
    model name, quantization setting, and the exact input-token
    sequence.  This means two requests with the same prompt will always
    share a cache entry regardless of ordering or batching.

    Parameters
    ----------
    wrapped:
        The underlying :class:`LogitSource` that actually computes
        logit distributions.
    cache_config:
        A :class:`CacheConfig` controlling capacity, eviction policy,
        TTL, disk persistence, etc.
    """

    def __init__(
        self,
        wrapped: LogitSource,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        self._wrapped = wrapped
        self._config = cache_config or CacheConfig()
        self._memory_cache = LRUCache(self._config)
        self._disk_cache: Optional[DiskCache] = None
        self._warmer = CacheWarmer()
        self._lock = threading.Lock()

        # Total counters (including batch calls).
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0

        # Set up disk cache if requested.
        if self._config.persist_to_disk:
            disk_dir = self._config.disk_cache_dir or ".cache/logit_source"
            self._disk_cache = DiskCache(
                cache_dir=disk_dir,
                compression=self._config.compression,
                ttl_seconds=self._config.ttl_seconds,
            )
            logger.info("Disk cache enabled at %s", disk_dir)

    # -- LogitSource interface ----------------------------------------------

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self._wrapped.model_name

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return self._wrapped.vocab_size

    @property
    def device(self) -> str:  # type: ignore[override]
        return self._wrapped.device

    def get_logits(
        self,
        input_ids: Union[List[int], Tuple[int, ...]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Return logits for *input_ids*, serving from cache when possible.

        On a cache miss the request is forwarded to the wrapped source
        and the result is stored before being returned.
        """
        key = self._make_key(input_ids)

        # 1. Memory cache.
        entry = self._memory_cache.get(key)
        if entry is not None:
            with self._lock:
                self._total_hits += 1
            return entry.logits

        # 2. Disk cache (if enabled).
        if self._disk_cache is not None:
            entry = self._disk_cache.get(key)
            if entry is not None:
                # Promote to memory cache.
                self._memory_cache.put(key, entry)
                with self._lock:
                    self._total_hits += 1
                return entry.logits

        # 3. Cache miss — compute.
        with self._lock:
            self._total_misses += 1

        logits = self._wrapped.get_logits(input_ids, **kwargs)

        entry = CacheEntry(
            logits=logits,
            metadata={"source": "computed", "model": self.model_name},
        )
        self._store(key, entry)
        return logits

    def get_logits_batch(
        self,
        input_ids_batch: List[List[int]],
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Batched logit retrieval with per-sequence cache lookups.

        Sequences that are already cached are served directly; only the
        cache-miss subset is forwarded to the wrapped source in a single
        batched call.
        """
        n = len(input_ids_batch)
        keys = [self._make_key(ids) for ids in input_ids_batch]
        results: List[Optional[np.ndarray]] = [None] * n
        miss_indices: List[int] = []

        # Phase 1: check caches.
        for i, key in enumerate(keys):
            entry = self._memory_cache.get(key)
            if entry is not None:
                results[i] = entry.logits
                with self._lock:
                    self._total_hits += 1
                continue

            if self._disk_cache is not None:
                entry = self._disk_cache.get(key)
                if entry is not None:
                    self._memory_cache.put(key, entry)
                    results[i] = entry.logits
                    with self._lock:
                        self._total_hits += 1
                    continue

            miss_indices.append(i)

        # Phase 2: forward misses.
        if miss_indices:
            miss_ids = [input_ids_batch[i] for i in miss_indices]
            with self._lock:
                self._total_misses += len(miss_indices)

            # Use batch call if the wrapped source supports it, else iterate.
            try:
                computed = self._wrapped.get_logits_batch(miss_ids, **kwargs)
            except (AttributeError, NotImplementedError):
                computed = [self._wrapped.get_logits(ids, **kwargs) for ids in miss_ids]

            for idx, logits in zip(miss_indices, computed):
                results[idx] = logits
                entry = CacheEntry(
                    logits=logits,
                    metadata={"source": "computed", "model": self.model_name},
                )
                self._store(keys[idx], entry)

        return results  # type: ignore[return-value]

    # -- cache management ---------------------------------------------------

    def cache_stats(self) -> Dict[str, Any]:
        """Return a dictionary of cache performance statistics."""
        mem_stats = self._memory_cache.stats
        total_requests = self._total_hits + self._total_misses
        stats: Dict[str, Any] = {
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "total_requests": total_requests,
            "hit_rate": self._total_hits / total_requests if total_requests else 0.0,
            "memory_cache": mem_stats,
            "eviction_policy": self._config.eviction_policy,
            "ttl_seconds": self._config.ttl_seconds,
        }
        if self._disk_cache is not None:
            stats["disk_cache"] = {
                "enabled": True,
                "total_size_bytes": self._disk_cache.total_size(),
                "entry_count": len(self._disk_cache.list_keys()),
            }
        else:
            stats["disk_cache"] = {"enabled": False}
        return stats

    def clear_cache(self) -> None:
        """Flush both memory and disk caches."""
        self._memory_cache.clear()
        if self._disk_cache is not None:
            for key_hash in self._disk_cache.list_keys():
                data_path = self._disk_cache._cache_dir / f"{key_hash}.dat"
                meta_path = self._disk_cache._cache_dir / f"{key_hash}.meta"
                for p in (data_path, meta_path):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
        with self._lock:
            self._total_hits = 0
            self._total_misses = 0
            self._total_evictions = 0
        logger.info("Cache cleared")

    def save_cache(self, path: Union[str, Path]) -> None:
        """Serialize the in-memory cache to *path*.

        The file format is a pickled dictionary of
        ``{CacheKey: CacheEntry}`` pairs, optionally zlib-compressed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        items = self._memory_cache.items()
        data = {
            "version": 1,
            "config": {
                "max_entries": self._config.max_entries,
                "max_size_bytes": self._config.max_size_bytes,
                "eviction_policy": self._config.eviction_policy,
                "compression": self._config.compression,
                "ttl_seconds": self._config.ttl_seconds,
            },
            "entries": {key: entry for key, entry in items},
            "stats": {
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
            },
        }

        raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if self._config.compression:
            raw = zlib.compress(raw, level=6)
            header = b"DDACZ"  # compressed
        else:
            header = b"DDACU"  # uncompressed

        path.write_bytes(header + raw)
        logger.info("Cache saved to %s (%d entries, %d bytes)", path, len(items), len(raw))

    def load_cache(self, path: Union[str, Path]) -> None:
        """Deserialize a cache snapshot from *path*.

        Entries from the file are merged into the current in-memory
        cache (existing entries with the same key are overwritten).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")

        raw = path.read_bytes()
        header = raw[:5]
        payload = raw[5:]

        if header == b"DDACZ":
            payload = zlib.decompress(payload)
        elif header == b"DDACU":
            pass
        else:
            raise ValueError(f"Unrecognised cache file header: {header!r}")

        data = pickle.loads(payload)  # noqa: S301

        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("Invalid cache file format")

        entries: Dict[CacheKey, CacheEntry] = data["entries"]
        loaded = 0
        for key, entry in entries.items():
            self._memory_cache.put(key, entry)
            loaded += 1

        # Restore aggregate counters if present.
        saved_stats = data.get("stats", {})
        with self._lock:
            self._total_hits += saved_stats.get("total_hits", 0)
            self._total_misses += saved_stats.get("total_misses", 0)

        logger.info("Cache loaded from %s (%d entries)", path, loaded)

    def warm_cache(
        self,
        prefixes: List[List[int]],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Pre-compute and cache logits for *prefixes*.

        Delegates to :class:`CacheWarmer` for progress tracking and
        batched execution.

        Parameters
        ----------
        prefixes:
            Token-ID sequences to pre-compute logits for.
        batch_size:
            Number of sequences per batch call.

        Returns
        -------
        dict
            Warming summary with keys ``total``, ``completed``,
            ``errors``, ``elapsed_seconds``.
        """
        return self._warmer.warm(self, prefixes, batch_size=batch_size)

    def warming_progress(self) -> Dict[str, Any]:
        """Return current cache-warming progress."""
        return self._warmer.progress()

    # -- helpers ------------------------------------------------------------

    def _make_key(self, input_ids: Union[List[int], Tuple[int, ...], Sequence[int]]) -> CacheKey:
        """Build a :class:`CacheKey` from the current model config and tokens."""
        quantization = getattr(self._wrapped, "quantization", None)
        return CacheKey(
            input_ids=tuple(input_ids),
            model_name=self.model_name,
            quantization=quantization,
        )

    def _store(self, key: CacheKey, entry: CacheEntry) -> None:
        """Insert *entry* into memory (and optionally disk) cache."""
        self._memory_cache.put(key, entry)
        if self._disk_cache is not None:
            try:
                self._disk_cache.put(key, entry)
            except Exception:
                logger.warning("Failed to persist entry to disk cache", exc_info=True)

    # -- context manager / cleanup ------------------------------------------

    def close(self) -> None:
        """Persist (if configured) and release resources."""
        if self._config.persist_to_disk and self._disk_cache is not None:
            logger.info("Flushing memory cache to disk on close")
            for key, entry in self._memory_cache.items():
                if not self._disk_cache.has(key):
                    self._disk_cache.put(key, entry)

    def __enter__(self) -> "CachedLogitSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.cache_stats()
        return (
            f"<CachedLogitSource wrapped={self._wrapped!r} "
            f"entries={stats['memory_cache']['entries']} "
            f"hit_rate={stats['hit_rate']:.1%}>"
        )
