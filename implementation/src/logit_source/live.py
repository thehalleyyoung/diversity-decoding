"""
LiveLogitSource — real-time model inference for the Diversity Decoding Arena.

Provides live logit computation from transformer language models via PyTorch
or ONNX Runtime backends, with KV-cache management, batched inference,
tokenizer utilities, and performance instrumentation.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from src.logit_source.base import LogitBatch, LogitSource, LogitSourceConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = os.environ.get(
    "LOGIT_SOURCE_CACHE_DIR",
    str(Path.home() / ".cache" / "diversity_decoding_arena"),
)
_WARMUP_SEQ_LEN = 16
_ONNX_PROVIDERS_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_ONNX_PROVIDERS_CPU = ["CPUExecutionProvider"]

# Lazy-import sentinels
_torch = None
_transformers = None
_ort = None


def _import_torch():
    """Lazily import PyTorch."""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for the LiveLogitSource PyTorch backend. "
                "Install it with: pip install torch"
            )
    return _torch


def _import_transformers():
    """Lazily import HuggingFace transformers."""
    global _transformers
    if _transformers is None:
        try:
            import transformers

            _transformers = transformers
        except ImportError:
            raise ImportError(
                "HuggingFace Transformers is required for the LiveLogitSource. "
                "Install it with: pip install transformers"
            )
    return _transformers


def _import_onnxruntime():
    """Lazily import ONNX Runtime."""
    global _ort
    if _ort is None:
        try:
            import onnxruntime

            _ort = onnxruntime
        except ImportError:
            raise ImportError(
                "ONNX Runtime is required for the ONNX backend. "
                "Install it with: pip install onnxruntime"
            )
    return _ort


# ---------------------------------------------------------------------------
# InferenceTimer — context manager for timing
# ---------------------------------------------------------------------------


class InferenceTimer:
    """Context manager that tracks wall-clock time, CPU time, and throughput.

    Usage::

        timer = InferenceTimer()
        with timer:
            # perform inference
            ...
        timer.tokens_generated = 128
        print(timer.throughput)  # tokens / sec

    Can also be used in a ``with`` statement as a one-shot or accumulated
    across multiple ``with`` blocks by calling :meth:`reset` between uses.
    """

    __slots__ = (
        "_wall_start",
        "_cpu_start",
        "wall_time",
        "cpu_time",
        "tokens_generated",
        "_running",
    )

    def __init__(self) -> None:
        self._wall_start: float = 0.0
        self._cpu_start: float = 0.0
        self.wall_time: float = 0.0
        self.cpu_time: float = 0.0
        self.tokens_generated: int = 0
        self._running: bool = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "InferenceTimer":
        if self._running:
            raise RuntimeError("InferenceTimer is already running")
        self._running = True
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.wall_time += time.perf_counter() - self._wall_start
        self.cpu_time += time.process_time() - self._cpu_start
        self._running = False

    # -- properties ---------------------------------------------------------

    @property
    def throughput(self) -> float:
        """Tokens generated per wall-clock second."""
        if self.wall_time <= 0:
            return 0.0
        return self.tokens_generated / self.wall_time

    @property
    def cpu_utilization(self) -> float:
        """Ratio of CPU time to wall time (1.0 = fully CPU-bound)."""
        if self.wall_time <= 0:
            return 0.0
        return min(self.cpu_time / self.wall_time, 1.0)

    # -- helpers ------------------------------------------------------------

    def reset(self) -> None:
        """Reset all accumulated timings."""
        self.wall_time = 0.0
        self.cpu_time = 0.0
        self.tokens_generated = 0
        self._running = False

    def as_dict(self) -> Dict[str, Any]:
        """Return a summary dictionary."""
        return {
            "wall_time_s": round(self.wall_time, 6),
            "cpu_time_s": round(self.cpu_time, 6),
            "tokens_generated": self.tokens_generated,
            "throughput_tok_s": round(self.throughput, 2),
            "cpu_utilization": round(self.cpu_utilization, 4),
        }

    def __repr__(self) -> str:
        return (
            f"InferenceTimer(wall={self.wall_time:.4f}s, "
            f"cpu={self.cpu_time:.4f}s, "
            f"tokens={self.tokens_generated}, "
            f"tok/s={self.throughput:.1f})"
        )


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------


class KVCacheManager:
    """LRU cache for key/value tensors produced during autoregressive decoding.

    Each entry is keyed by a tuple of token IDs that produced the KV state.
    The cache evicts least-recently-used entries when *max_entries* is exceeded.

    Parameters:
        max_entries: Maximum number of cached KV states.
        max_memory_bytes: Optional soft memory cap (bytes); entries are evicted
            when estimated usage exceeds this value.  Set to ``0`` to disable.
    """

    def __init__(
        self,
        max_entries: int = 64,
        max_memory_bytes: int = 0,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_bytes
        # OrderedDict used as an LRU map (move_to_end on access)
        self._cache: OrderedDict[Tuple[int, ...], Any] = OrderedDict()
        self._entry_sizes: Dict[Tuple[int, ...], int] = {}
        self._total_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0

    # -- public interface ---------------------------------------------------

    def store(self, key: Tuple[int, ...], past_key_values: Any) -> None:
        """Insert or update a KV-cache entry.

        Args:
            key: Tuple of token IDs representing the prefix.
            past_key_values: Opaque KV-cache object (PyTorch tuple-of-tensors,
                NumPy arrays, etc.).
        """
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple, got {type(key).__name__}")

        entry_bytes = self._estimate_size(past_key_values)

        # If the key already exists, remove old size accounting
        if key in self._cache:
            self._total_bytes -= self._entry_sizes.get(key, 0)
            self._cache.move_to_end(key)
        else:
            # Evict if necessary before inserting
            while len(self._cache) >= self._max_entries:
                self._evict_oldest()
            if (
                self._max_memory_bytes > 0
                and self._total_bytes + entry_bytes > self._max_memory_bytes
            ):
                while (
                    self._cache
                    and self._total_bytes + entry_bytes > self._max_memory_bytes
                ):
                    self._evict_oldest()

        self._cache[key] = past_key_values
        self._entry_sizes[key] = entry_bytes
        self._total_bytes += entry_bytes
        logger.debug(
            "KVCacheManager: stored key len=%d, entry=%d bytes, total=%d bytes",
            len(key),
            entry_bytes,
            self._total_bytes,
        )

    def lookup(self, key: Tuple[int, ...]) -> Optional[Any]:
        """Look up a KV-cache entry.

        Returns:
            The cached past_key_values if found, else ``None``.
        """
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple, got {type(key).__name__}")

        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            logger.debug("KVCacheManager: hit for key len=%d", len(key))
            return self._cache[key]
        self._misses += 1
        logger.debug("KVCacheManager: miss for key len=%d", len(key))
        return None

    def trim(self, max_entries: int) -> int:
        """Evict entries until at most *max_entries* remain.

        Returns:
            Number of entries evicted.
        """
        if max_entries < 0:
            raise ValueError(f"max_entries must be >= 0, got {max_entries}")
        evicted = 0
        while len(self._cache) > max_entries:
            self._evict_oldest()
            evicted += 1
        self._max_entries = max(max_entries, self._max_entries)
        logger.debug("KVCacheManager: trimmed %d entries", evicted)
        return evicted

    def clear(self) -> None:
        """Remove all cached entries and reset accounting."""
        n = len(self._cache)
        self._cache.clear()
        self._entry_sizes.clear()
        self._total_bytes = 0
        logger.debug("KVCacheManager: cleared %d entries", n)

    def memory_usage(self) -> int:
        """Return estimated total memory usage of cached entries in bytes."""
        return self._total_bytes

    @property
    def size(self) -> int:
        """Number of currently cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def stats(self) -> Dict[str, Any]:
        """Return a summary of cache statistics."""
        return {
            "size": self.size,
            "max_entries": self._max_entries,
            "memory_bytes": self._total_bytes,
            "max_memory_bytes": self._max_memory_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    # -- internal -----------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._cache:
            return
        key, _ = self._cache.popitem(last=False)
        evicted_bytes = self._entry_sizes.pop(key, 0)
        self._total_bytes -= evicted_bytes
        logger.debug(
            "KVCacheManager: evicted key len=%d, freed %d bytes",
            len(key),
            evicted_bytes,
        )

    @staticmethod
    def _estimate_size(obj: Any) -> int:
        """Estimate byte size of a KV-cache object.

        Supports PyTorch tensors, NumPy arrays, nested tuples/lists, and
        falls back to ``sys.getsizeof`` for unknown types.
        """
        if obj is None:
            return 0

        # NumPy array
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)

        # Try PyTorch tensor
        try:
            torch = _import_torch()
            if isinstance(obj, torch.Tensor):
                return int(obj.nelement() * obj.element_size())
        except ImportError:
            pass

        # Nested tuple / list (common for HuggingFace KV caches)
        if isinstance(obj, (tuple, list)):
            return sum(KVCacheManager._estimate_size(item) for item in obj)

        # Fallback
        return sys.getsizeof(obj)

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(size={self.size}/{self._max_entries}, "
            f"mem={self._total_bytes} B, "
            f"hit_rate={self.hit_rate:.1%})"
        )

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: Tuple[int, ...]) -> bool:
        return key in self._cache


# ---------------------------------------------------------------------------
# BatchManager — padding / collation utilities
# ---------------------------------------------------------------------------


class BatchManager:
    """Utilities for padding, masking, and splitting batches of sequences.

    All methods are stateless class/static methods so ``BatchManager`` can be
    used without instantiation, but an instance may be created for convenience.
    """

    def __init__(self, pad_value: int = 0, max_batch_size: int = 32) -> None:
        self.pad_value = pad_value
        self.max_batch_size = max_batch_size

    @staticmethod
    def pad_sequences(
        sequences: List[List[int]],
        pad_value: int = 0,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ) -> np.ndarray:
        """Pad variable-length sequences to a uniform length.

        Args:
            sequences: List of integer sequences.
            pad_value: Value used for padding tokens.
            max_length: If given, pad/truncate to this length; otherwise use
                the longest sequence in the batch.
            padding_side: ``"left"`` or ``"right"``.

        Returns:
            2-D ``int64`` array of shape ``[B, L]``.
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")

        lengths = [len(s) for s in sequences]
        target_len = max_length if max_length is not None else max(lengths)
        if target_len <= 0:
            raise ValueError(f"target length must be positive, got {target_len}")

        batch_size = len(sequences)
        padded = np.full((batch_size, target_len), pad_value, dtype=np.int64)

        for i, seq in enumerate(sequences):
            truncated = seq[:target_len]
            seq_len = len(truncated)
            if padding_side == "right":
                padded[i, :seq_len] = truncated
            elif padding_side == "left":
                padded[i, target_len - seq_len :] = truncated
            else:
                raise ValueError(
                    f"padding_side must be 'left' or 'right', got {padding_side!r}"
                )

        return padded

    @staticmethod
    def create_attention_mask(
        sequences: List[List[int]],
        pad_value: int = 0,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ) -> np.ndarray:
        """Create a binary attention mask for padded sequences.

        ``1`` indicates real tokens, ``0`` indicates padding.

        Args:
            sequences: List of integer sequences (pre-padding).
            pad_value: Padding token value (unused; mask is built from lengths).
            max_length: If given, use this length; otherwise derive from data.
            padding_side: ``"left"`` or ``"right"``.

        Returns:
            2-D ``int64`` array of shape ``[B, L]``.
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")

        lengths = [len(s) for s in sequences]
        target_len = max_length if max_length is not None else max(lengths)
        if target_len <= 0:
            raise ValueError(f"target length must be positive, got {target_len}")

        batch_size = len(sequences)
        mask = np.zeros((batch_size, target_len), dtype=np.int64)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), target_len)
            if padding_side == "right":
                mask[i, :seq_len] = 1
            elif padding_side == "left":
                mask[i, target_len - seq_len :] = 1
            else:
                raise ValueError(
                    f"padding_side must be 'left' or 'right', got {padding_side!r}"
                )

        return mask

    @staticmethod
    def collate(
        sequences: List[List[int]],
        pad_value: int = 0,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ) -> Dict[str, np.ndarray]:
        """Pad sequences and build attention mask in one call.

        Returns:
            Dict with ``"input_ids"`` and ``"attention_mask"`` arrays.
        """
        input_ids = BatchManager.pad_sequences(
            sequences,
            pad_value=pad_value,
            max_length=max_length,
            padding_side=padding_side,
        )
        attention_mask = BatchManager.create_attention_mask(
            sequences,
            pad_value=pad_value,
            max_length=max_length,
            padding_side=padding_side,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def split_batch(
        batch: Dict[str, np.ndarray], max_batch_size: int
    ) -> List[Dict[str, np.ndarray]]:
        """Split a collated batch into sub-batches of at most *max_batch_size*.

        Args:
            batch: Dict containing at least ``"input_ids"``; may also have
                ``"attention_mask"`` and other arrays with a leading batch dim.
            max_batch_size: Maximum number of sequences per sub-batch.

        Returns:
            List of dicts with the same keys, each with ≤ *max_batch_size* rows.
        """
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive, got {max_batch_size}"
            )
        if "input_ids" not in batch:
            raise KeyError("batch must contain 'input_ids'")

        total = batch["input_ids"].shape[0]
        if total <= max_batch_size:
            return [batch]

        sub_batches: List[Dict[str, np.ndarray]] = []
        for start in range(0, total, max_batch_size):
            end = min(start + max_batch_size, total)
            sub: Dict[str, np.ndarray] = {}
            for key, arr in batch.items():
                if isinstance(arr, np.ndarray) and arr.shape[0] == total:
                    sub[key] = arr[start:end]
                else:
                    sub[key] = arr  # non-batchable value — keep as-is
            sub_batches.append(sub)

        return sub_batches

    def __repr__(self) -> str:
        return (
            f"BatchManager(pad_value={self.pad_value}, "
            f"max_batch_size={self.max_batch_size})"
        )


# ---------------------------------------------------------------------------
# TokenizerWrapper
# ---------------------------------------------------------------------------


class TokenizerWrapper:
    """Thin wrapper around a HuggingFace tokenizer with explicit API.

    Adds consistent error handling, default arguments, and convenience
    properties for vocabulary metadata.

    Parameters:
        tokenizer: A ``PreTrainedTokenizerBase`` (or compatible) instance.
    """

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer must not be None")
        self._tokenizer = tokenizer

        # Ensure pad_token is set (many models lack one)
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info(
                    "TokenizerWrapper: pad_token not set; defaulting to eos_token (%r)",
                    self._tokenizer.eos_token,
                )
            else:
                logger.warning(
                    "TokenizerWrapper: tokenizer has no pad_token or eos_token; "
                    "padding may produce unexpected results"
                )

    # -- encoding -----------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tokenize *text* and return a dict of arrays / lists.

        Args:
            text: Input text.
            add_special_tokens: Whether to include BOS/EOS tokens.
            max_length: Maximum number of tokens.
            truncation: Whether to truncate at *max_length*.
            padding: Padding strategy (``False``, ``True``, ``"max_length"``).
            return_tensors: ``"np"`` for NumPy, ``"pt"`` for PyTorch, or
                ``None`` for plain Python lists.

        Returns:
            Dict with ``"input_ids"``, ``"attention_mask"``, and possibly
            ``"token_type_ids"``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        kwargs: Dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True  # max_length requires truncation
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors

        try:
            result = self._tokenizer(text, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Tokenizer encode failed: {exc}") from exc

        # Normalise HuggingFace BatchEncoding → plain dict
        if hasattr(result, "data"):
            return dict(result.data)
        return dict(result)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token IDs into text.

        Args:
            token_ids: Integer token IDs.
            skip_special_tokens: Whether to strip special tokens from output.

        Returns:
            Decoded string.
        """
        if not isinstance(token_ids, (list, tuple, np.ndarray)):
            raise TypeError(
                f"token_ids must be list/tuple/ndarray, got {type(token_ids).__name__}"
            )
        try:
            return self._tokenizer.decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
        except Exception as exc:
            raise RuntimeError(f"Tokenizer decode failed: {exc}") from exc

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Union[bool, str] = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tokenize a list of texts with automatic padding.

        Args:
            texts: List of input strings.
            add_special_tokens: Whether to include BOS/EOS tokens.
            max_length: Optional maximum sequence length.
            truncation: Whether to truncate at *max_length*.
            padding: Padding strategy (``True`` = pad to longest).
            return_tensors: ``"np"``, ``"pt"``, or ``None``.

        Returns:
            Dict with batched ``"input_ids"`` and ``"attention_mask"``.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"texts must be a list, got {type(texts).__name__}")
        if not texts:
            raise ValueError("texts must be non-empty")

        kwargs: Dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors

        try:
            result = self._tokenizer(texts, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Tokenizer batch_encode failed: {exc}") from exc

        if hasattr(result, "data"):
            return dict(result.data)
        return dict(result)

    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token-ID sequences.

        Args:
            token_ids_list: List of lists of integer token IDs.
            skip_special_tokens: Whether to strip special tokens.

        Returns:
            List of decoded strings.
        """
        if not isinstance(token_ids_list, (list, tuple)):
            raise TypeError(
                f"token_ids_list must be a list, got {type(token_ids_list).__name__}"
            )
        try:
            return self._tokenizer.batch_decode(
                token_ids_list, skip_special_tokens=skip_special_tokens
            )
        except Exception as exc:
            raise RuntimeError(f"Tokenizer batch_decode failed: {exc}") from exc

    # -- vocabulary properties ----------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (including added tokens)."""
        return len(self._tokenizer)

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id

    @property
    def pad_token(self) -> Optional[str]:
        return self._tokenizer.pad_token

    @property
    def eos_token(self) -> Optional[str]:
        return self._tokenizer.eos_token

    @property
    def bos_token(self) -> Optional[str]:
        return self._tokenizer.bos_token

    # -- token conversion ---------------------------------------------------

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert a list of token IDs to their string representations.

        Args:
            ids: Token IDs.

        Returns:
            List of token strings.
        """
        if not isinstance(ids, (list, tuple, np.ndarray)):
            raise TypeError(f"ids must be list/tuple/ndarray, got {type(ids).__name__}")
        try:
            return self._tokenizer.convert_ids_to_tokens(list(ids))
        except Exception as exc:
            raise RuntimeError(
                f"convert_ids_to_tokens failed: {exc}"
            ) from exc

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert a list of token strings to their integer IDs.

        Args:
            tokens: Token strings.

        Returns:
            List of integer IDs.
        """
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(
                f"tokens must be a list, got {type(tokens).__name__}"
            )
        try:
            return self._tokenizer.convert_tokens_to_ids(tokens)
        except Exception as exc:
            raise RuntimeError(
                f"convert_tokens_to_ids failed: {exc}"
            ) from exc

    def get_special_tokens(self) -> Dict[str, Optional[str]]:
        """Return a mapping of special-token roles to their string values.

        Returns:
            Dict with keys ``"bos_token"``, ``"eos_token"``, ``"pad_token"``,
            ``"unk_token"``, ``"sep_token"``, ``"cls_token"``, ``"mask_token"``.
        """
        return {
            "bos_token": getattr(self._tokenizer, "bos_token", None),
            "eos_token": getattr(self._tokenizer, "eos_token", None),
            "pad_token": getattr(self._tokenizer, "pad_token", None),
            "unk_token": getattr(self._tokenizer, "unk_token", None),
            "sep_token": getattr(self._tokenizer, "sep_token", None),
            "cls_token": getattr(self._tokenizer, "cls_token", None),
            "mask_token": getattr(self._tokenizer, "mask_token", None),
        }

    @property
    def inner(self) -> Any:
        """Access the underlying HuggingFace tokenizer object."""
        return self._tokenizer

    def __repr__(self) -> str:
        return (
            f"TokenizerWrapper(vocab_size={self.vocab_size}, "
            f"pad={self.pad_token_id}, "
            f"eos={self.eos_token_id}, "
            f"bos={self.bos_token_id})"
        )


# ---------------------------------------------------------------------------
# ModelLoader — utilities for loading and exporting models
# ---------------------------------------------------------------------------


class ModelLoader:
    """Utility class for loading, quantizing, and exporting language models.

    All methods are classmethods / staticmethods and do not require
    instantiation, but an instance can be created for convenience.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # -- PyTorch loading ----------------------------------------------------

    @staticmethod
    def load_pytorch_model(
        model_name: str,
        device: str = "cpu",
        dtype: str = "float32",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> Any:
        """Load a PyTorch causal-LM from HuggingFace Hub.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"gpt2"``).
            device: Target device (``"cpu"``, ``"cuda"``, …).
            dtype: Weight precision (``"float16"``, ``"float32"``, …).
            trust_remote_code: Whether to trust remote model code.
            cache_dir: Optional local cache directory.

        Returns:
            A ``PreTrainedModel`` in eval mode on the specified device.
        """
        torch = _import_torch()
        transformers = _import_transformers()

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        logger.info(
            "Loading PyTorch model %r (device=%s, dtype=%s)",
            model_name,
            device,
            dtype,
        )

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PyTorch model {model_name!r}: {exc}"
            ) from exc

        model = model.to(device)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Loaded %s: %s parameters on %s", model_name, f"{param_count:,}", device
        )
        return model

    # -- tokenizer loading --------------------------------------------------

    @staticmethod
    def load_tokenizer(
        model_name: str,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> Any:
        """Load a HuggingFace tokenizer.

        Args:
            model_name: HuggingFace model identifier.
            trust_remote_code: Whether to trust remote tokenizer code.
            cache_dir: Optional local cache directory.

        Returns:
            A ``PreTrainedTokenizer`` instance.
        """
        transformers = _import_transformers()

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        logger.info("Loading tokenizer for %r", model_name)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **load_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer for {model_name!r}: {exc}"
            ) from exc

        return tokenizer

    # -- ONNX loading -------------------------------------------------------

    @staticmethod
    def load_onnx_model(
        model_path: str,
        device: str = "cpu",
    ) -> Any:
        """Load an ONNX model via ONNX Runtime.

        Args:
            model_path: Path to the ``.onnx`` file.
            device: ``"cpu"`` or ``"cuda"``.

        Returns:
            An ``onnxruntime.InferenceSession``.
        """
        ort = _import_onnxruntime()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = (
            _ONNX_PROVIDERS_CUDA if "cuda" in device else _ONNX_PROVIDERS_CPU
        )

        logger.info("Loading ONNX model from %s (providers=%s)", model_path, providers)
        try:
            session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX model from {model_path}: {exc}"
            ) from exc

        logger.info(
            "ONNX model loaded: inputs=%s, outputs=%s",
            [inp.name for inp in session.get_inputs()],
            [out.name for out in session.get_outputs()],
        )
        return session

    # -- quantization -------------------------------------------------------

    @staticmethod
    def quantize_model(
        model: Any,
        quantization_type: str = "int8",
    ) -> Any:
        """Apply post-training quantization to a PyTorch model.

        Args:
            model: A PyTorch ``nn.Module``.
            quantization_type: ``"int8"`` for dynamic quantization,
                ``"int4"`` for GPTQ-style (requires ``auto_gptq``).

        Returns:
            The quantized model (may be a new object).
        """
        torch = _import_torch()

        if quantization_type == "int8":
            logger.info("Applying dynamic int8 quantization")
            try:
                quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info("int8 quantization complete")
                return quantized
            except Exception as exc:
                raise RuntimeError(
                    f"int8 quantization failed: {exc}"
                ) from exc

        elif quantization_type == "int4":
            logger.info("int4 quantization requested — attempting bitsandbytes")
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                # int4 is typically applied at load time; return config
                # to be passed to from_pretrained
                logger.warning(
                    "int4 quantization must be applied at model load time. "
                    "Returning BitsAndBytesConfig to pass to from_pretrained."
                )
                return bnb_config
            except ImportError:
                raise ImportError(
                    "int4 quantization requires bitsandbytes: "
                    "pip install bitsandbytes"
                )

        else:
            raise ValueError(
                f"Unsupported quantization_type: {quantization_type!r}; "
                f"supported: 'int8', 'int4'"
            )

    # -- model info ---------------------------------------------------------

    @staticmethod
    def get_model_info(model: Any) -> Dict[str, Any]:
        """Collect metadata about a PyTorch model.

        Args:
            model: A PyTorch ``nn.Module``.

        Returns:
            Dict with ``"parameter_count"``, ``"trainable_parameters"``,
            ``"memory_bytes"``, ``"memory_mb"``, ``"dtype"``, ``"device"``,
            and ``"model_type"``.
        """
        torch = _import_torch()

        total_params = 0
        trainable_params = 0
        total_bytes = 0

        first_dtype = None
        first_device = None

        for name, param in model.named_parameters():
            n = param.numel()
            total_params += n
            if param.requires_grad:
                trainable_params += n
            total_bytes += n * param.element_size()
            if first_dtype is None:
                first_dtype = str(param.dtype)
            if first_device is None:
                first_device = str(param.device)

        model_type = type(model).__name__
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            model_type = model.config.model_type

        return {
            "parameter_count": total_params,
            "trainable_parameters": trainable_params,
            "memory_bytes": total_bytes,
            "memory_mb": round(total_bytes / (1024 * 1024), 2),
            "dtype": first_dtype or "unknown",
            "device": first_device or "unknown",
            "model_type": model_type,
        }

    # -- download -----------------------------------------------------------

    @staticmethod
    def download_model(
        model_name: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Ensure a HuggingFace model is downloaded and cached locally.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Target cache directory; defaults to HuggingFace's
                default cache.

        Returns:
            Local path to the cached model snapshot.
        """
        transformers = _import_transformers()
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for download_model: "
                "pip install huggingface_hub"
            )

        logger.info("Downloading/verifying model %r to %s", model_name, cache_dir)
        try:
            local_path = snapshot_download(
                model_name,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model {model_name!r}: {exc}"
            ) from exc

        logger.info("Model cached at %s", local_path)
        return local_path

    # -- ONNX export --------------------------------------------------------

    @staticmethod
    def export_to_onnx(
        model: Any,
        output_path: str,
        max_seq_len: int = 512,
        opset_version: int = 14,
    ) -> str:
        """Export a PyTorch causal-LM to ONNX format.

        Args:
            model: A PyTorch ``nn.Module`` (causal LM).
            output_path: Destination ``.onnx`` file path.
            max_seq_len: Sequence length for dummy input.
            opset_version: ONNX opset version.

        Returns:
            The *output_path* on success.
        """
        torch = _import_torch()

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting model to ONNX: %s", output_path)

        # Create dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randint(
            0, 1000, (1, min(max_seq_len, 64)), device=device
        )
        attention_mask = torch.ones_like(dummy_input)

        model.eval()
        try:
            torch.onnx.export(
                model,
                (dummy_input, attention_mask),
                str(output_path_obj),
                opset_version=opset_version,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"ONNX export failed: {exc}"
            ) from exc

        file_size = output_path_obj.stat().st_size
        logger.info(
            "ONNX model exported to %s (%.1f MB)",
            output_path,
            file_size / (1024 * 1024),
        )
        return str(output_path_obj)

    def __repr__(self) -> str:
        return f"ModelLoader(cache_dir={self.cache_dir!r})"


# ---------------------------------------------------------------------------
# LiveLogitSource
# ---------------------------------------------------------------------------


class LiveLogitSource(LogitSource):
    """Live model inference backend for the Diversity Decoding Arena.

    Supports PyTorch and ONNX Runtime backends with optional KV-caching,
    batched inference, and automatic model/tokenizer lazy-loading.

    Parameters:
        config: A :class:`LogitSourceConfig` specifying model name, device,
            dtype, and other parameters.
        model: Optional pre-loaded model (PyTorch ``nn.Module`` or
            ``onnxruntime.InferenceSession``).  If *None*, the model will be
            loaded lazily on first use.
        tokenizer: Optional pre-loaded HuggingFace tokenizer.  If *None*,
            loaded lazily alongside the model.
        backend: ``"pytorch"`` (default) or ``"onnx"``.
        onnx_path: Path to an ONNX model file (required when
            ``backend="onnx"`` and *model* is not provided).
        kv_cache_max_entries: Maximum number of KV-cache entries.
        kv_cache_max_memory: Soft memory cap for KV cache in bytes.
        trust_remote_code: Whether to trust remote model/tokenizer code on
            HuggingFace Hub.

    Example::

        config = LogitSourceConfig(model_name="gpt2", device="cuda")
        with LiveLogitSource(config) as src:
            src.warmup()
            ids = src.encode("Hello, world!")
            logits = src.get_logits(ids)
    """

    def __init__(
        self,
        config: LogitSourceConfig,
        model: Any = None,
        tokenizer: Any = None,
        backend: str = "pytorch",
        onnx_path: Optional[str] = None,
        kv_cache_max_entries: int = 64,
        kv_cache_max_memory: int = 0,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(config)

        if backend not in ("pytorch", "onnx"):
            raise ValueError(
                f"backend must be 'pytorch' or 'onnx', got {backend!r}"
            )

        self._backend = backend
        self._onnx_path = onnx_path
        self._trust_remote_code = trust_remote_code

        # Model / tokenizer (may be None → lazy load)
        self._model: Any = model
        self._raw_tokenizer: Any = tokenizer
        self._tokenizer_wrapper: Optional[TokenizerWrapper] = None
        self._model_loaded: bool = model is not None
        self._tokenizer_loaded: bool = tokenizer is not None

        # KV-cache
        self._kv_cache = KVCacheManager(
            max_entries=kv_cache_max_entries,
            max_memory_bytes=kv_cache_max_memory,
        )

        # Batch manager
        self._batch_manager = BatchManager()

        # Model loader
        self._model_loader = ModelLoader()

        # Inference timer (reusable)
        self._timer = InferenceTimer()

        logger.info(
            "LiveLogitSource created: model=%s, backend=%s, device=%s",
            config.model_name,
            backend,
            config.device,
        )

    # -- lazy loading -------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load model and tokenizer if they haven't been loaded yet.

        Thread-safe from a correctness perspective (idempotent), though not
        locked — concurrent first calls may trigger redundant loads.
        """
        if self._model_loaded and self._tokenizer_loaded:
            return

        self._check_closed()

        if not self._tokenizer_loaded:
            logger.info(
                "Lazy-loading tokenizer for %s", self._config.model_name
            )
            self._raw_tokenizer = ModelLoader.load_tokenizer(
                self._config.model_name,
                trust_remote_code=self._trust_remote_code,
                cache_dir=self._model_loader.cache_dir,
            )
            self._tokenizer_loaded = True

        if not self._model_loaded:
            if self._backend == "pytorch":
                logger.info(
                    "Lazy-loading PyTorch model %s", self._config.model_name
                )
                self._model = ModelLoader.load_pytorch_model(
                    self._config.model_name,
                    device=self._config.device,
                    dtype=self._config.dtype,
                    trust_remote_code=self._trust_remote_code,
                    cache_dir=self._model_loader.cache_dir,
                )

                # Apply quantization if requested
                if self._config.quantization is not None:
                    logger.info(
                        "Applying %s quantization", self._config.quantization
                    )
                    self._model = ModelLoader.quantize_model(
                        self._model, self._config.quantization
                    )

            elif self._backend == "onnx":
                if self._onnx_path is None:
                    raise ValueError(
                        "onnx_path is required when backend='onnx' and no "
                        "model is provided"
                    )
                self._model = ModelLoader.load_onnx_model(
                    self._onnx_path,
                    device=self._config.device,
                )

            self._model_loaded = True

        # Wrap tokenizer
        if self._tokenizer_wrapper is None and self._raw_tokenizer is not None:
            self._tokenizer_wrapper = TokenizerWrapper(self._raw_tokenizer)

        logger.info("Model and tokenizer ready for %s", self._config.model_name)

    # -- abstract interface implementations ---------------------------------

    def get_logits(self, input_ids: List[int]) -> np.ndarray:
        """Return logits for every position in *input_ids*.

        Uses the configured backend (PyTorch or ONNX) and optionally
        leverages KV-cache for incremental computation.

        Args:
            input_ids: Token-ID sequence of length *S*.

        Returns:
            Array of shape ``[S, vocab_size]`` with raw logits.
        """
        self._check_closed()
        self._ensure_model_loaded()

        if not input_ids:
            raise ValueError("input_ids must be non-empty")

        if len(input_ids) > self._config.max_seq_len:
            warnings.warn(
                f"input_ids length {len(input_ids)} exceeds max_seq_len "
                f"{self._config.max_seq_len}; truncating from the left"
            )
            input_ids = input_ids[-self._config.max_seq_len :]

        if self._backend == "pytorch":
            return self._get_logits_pytorch(input_ids)
        elif self._backend == "onnx":
            return self._get_logits_onnx(input_ids)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def get_logits_batch(self, input_ids_batch: List[List[int]]) -> LogitBatch:
        """Batched logit computation.

        Pads sequences, builds attention masks, runs a single forward pass,
        and extracts the last-position logits for each sequence.

        Args:
            input_ids_batch: List of *B* token-ID sequences.

        Returns:
            A :class:`LogitBatch` with logits of shape ``[B, vocab_size]``
            (last position per sequence).
        """
        self._check_closed()
        self._ensure_model_loaded()

        if not input_ids_batch:
            raise ValueError("input_ids_batch must be non-empty")

        # Truncate sequences exceeding max_seq_len
        truncated: List[List[int]] = []
        for seq in input_ids_batch:
            if not seq:
                raise ValueError("Each sequence in the batch must be non-empty")
            if len(seq) > self._config.max_seq_len:
                truncated.append(seq[-self._config.max_seq_len :])
            else:
                truncated.append(seq)

        # Determine pad token
        pad_id = 0
        if self._tokenizer_wrapper is not None and self._tokenizer_wrapper.pad_token_id is not None:
            pad_id = self._tokenizer_wrapper.pad_token_id

        # Split into sub-batches if needed
        if len(truncated) > self._config.batch_size:
            sub_results: List[np.ndarray] = []
            for i in range(0, len(truncated), self._config.batch_size):
                sub = truncated[i : i + self._config.batch_size]
                sub_batch = self._run_batch(sub, pad_id)
                sub_results.append(sub_batch)
            all_logits = np.concatenate(sub_results, axis=0)
        else:
            all_logits = self._run_batch(truncated, pad_id)

        timer_info = self._timer.as_dict() if self._timer.wall_time > 0 else {}

        return LogitBatch(
            logits=all_logits,
            token_ids=truncated,
            metadata={
                "backend": self._backend,
                "batch_size": len(truncated),
                "timing": timer_info,
            },
        )

    def encode(self, text: str) -> List[int]:
        """Tokenize *text* into a list of integer token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        self._check_closed()
        self._ensure_model_loaded()

        if self._tokenizer_wrapper is None:
            raise RuntimeError("Tokenizer not available")

        result = self._tokenizer_wrapper.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        ids = result.get("input_ids", [])
        if isinstance(ids, np.ndarray):
            ids = ids.flatten().tolist()
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
        # Ensure we return plain ints
        return [int(x) for x in ids]

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            token_ids: Integer token IDs.

        Returns:
            Decoded text string.
        """
        self._check_closed()
        self._ensure_model_loaded()

        if self._tokenizer_wrapper is None:
            raise RuntimeError("Tokenizer not available")

        return self._tokenizer_wrapper.decode(
            token_ids, skip_special_tokens=True
        )

    # -- PyTorch inference --------------------------------------------------

    def _get_logits_pytorch(self, input_ids: List[int]) -> np.ndarray:
        """Run a PyTorch forward pass and return logits as a NumPy array.

        Optionally uses KV-cache when ``config.use_kv_cache`` is True.
        """
        torch = _import_torch()

        input_key = tuple(input_ids)
        past_kv = None
        new_ids = input_ids
        cache_prefix_len = 0

        # KV-cache lookup: find longest cached prefix
        if self._config.use_kv_cache:
            for end in range(len(input_ids), 0, -1):
                prefix_key = tuple(input_ids[:end])
                cached = self._kv_cache.lookup(prefix_key)
                if cached is not None:
                    past_kv = cached
                    new_ids = input_ids[end:]
                    cache_prefix_len = end
                    self._stats.cache_hits += 1
                    logger.debug(
                        "KV-cache hit: prefix_len=%d, new_tokens=%d",
                        end,
                        len(new_ids),
                    )
                    break
            else:
                self._stats.cache_misses += 1

        # If all tokens were cached and no new tokens, run full forward
        if not new_ids:
            new_ids = [input_ids[-1]]
            cache_prefix_len = len(input_ids) - 1
            if cache_prefix_len > 0:
                prefix_key = tuple(input_ids[:cache_prefix_len])
                past_kv = self._kv_cache.lookup(prefix_key)

        # Build tensors
        device = self._config.device
        ids_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)

        # Build position IDs for KV-cache continuation
        if past_kv is not None:
            position_offset = cache_prefix_len
            position_ids = torch.arange(
                position_offset,
                position_offset + len(new_ids),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
        else:
            position_ids = None

        # Attention mask covers both cached + new tokens
        total_len = cache_prefix_len + len(new_ids)
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

        # Forward pass
        with torch.no_grad():
            kwargs: Dict[str, Any] = {
                "input_ids": ids_tensor,
                "attention_mask": attention_mask,
            }
            if past_kv is not None:
                kwargs["past_key_values"] = past_kv
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            kwargs["use_cache"] = self._config.use_kv_cache

            try:
                outputs = self._model(**kwargs)
            except TypeError:
                # Some models don't support all kwargs; fall back
                minimal_kwargs = {"input_ids": ids_tensor}
                if past_kv is not None:
                    try:
                        outputs = self._model(
                            input_ids=ids_tensor,
                            past_key_values=past_kv,
                            attention_mask=attention_mask,
                        )
                    except TypeError:
                        outputs = self._model(input_ids=ids_tensor)
                else:
                    outputs = self._model(**minimal_kwargs)

        logits = outputs.logits  # [1, new_len, vocab]

        # Store updated KV cache
        if (
            self._config.use_kv_cache
            and hasattr(outputs, "past_key_values")
            and outputs.past_key_values is not None
        ):
            self._kv_cache.store(input_key, outputs.past_key_values)

        # Convert to NumPy
        logits_np = logits[0].cpu().float().numpy()  # [new_len, vocab]

        # If we used KV-cache, we only have logits for new tokens.
        # For the full-sequence interface, we need all S positions.
        # But re-computing the cached portion is wasteful; instead,
        # if the caller truly needs all S positions, do a full pass.
        if cache_prefix_len > 0 and logits_np.shape[0] < len(input_ids):
            # Return only the new logits — caller typically wants last anyway.
            # For full compatibility, re-run without cache.
            full_ids_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=device
            )
            full_mask = torch.ones(
                (1, len(input_ids)), dtype=torch.long, device=device
            )
            with torch.no_grad():
                try:
                    full_out = self._model(
                        input_ids=full_ids_tensor,
                        attention_mask=full_mask,
                        use_cache=False,
                    )
                except TypeError:
                    full_out = self._model(input_ids=full_ids_tensor)
            logits_np = full_out.logits[0].cpu().float().numpy()

        return logits_np.astype(self._config.numpy_dtype)

    def _get_logits_onnx(self, input_ids: List[int]) -> np.ndarray:
        """Run an ONNX Runtime forward pass and return logits."""
        ort = _import_onnxruntime()
        session = self._model

        input_array = np.array([input_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_array, dtype=np.int64)

        # Determine available input names
        input_names = {inp.name for inp in session.get_inputs()}

        feed: Dict[str, np.ndarray] = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_array
        else:
            # Use the first input name as fallback
            feed[session.get_inputs()[0].name] = input_array

        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask

        try:
            output_names = [out.name for out in session.get_outputs()]
            results = session.run(output_names, feed)
        except Exception as exc:
            raise RuntimeError(
                f"ONNX inference failed: {exc}"
            ) from exc

        logits = results[0]  # typically [1, seq_len, vocab]
        if logits.ndim == 3:
            logits = logits[0]  # [seq_len, vocab]

        return logits.astype(self._config.numpy_dtype)

    # -- batch inference ----------------------------------------------------

    def _run_batch(self, sequences: List[List[int]], pad_id: int) -> np.ndarray:
        """Run a single-batch forward pass and return last-position logits.

        Args:
            sequences: List of token-ID sequences (same batch).
            pad_id: Padding token ID.

        Returns:
            Array of shape ``[B, vocab_size]``.
        """
        collated = BatchManager.collate(
            sequences, pad_value=pad_id, padding_side="left"
        )
        input_ids_np = collated["input_ids"]
        attention_mask_np = collated["attention_mask"]
        seq_lengths = [len(s) for s in sequences]

        self._timer.reset()

        if self._backend == "pytorch":
            logits_np = self._run_batch_pytorch(input_ids_np, attention_mask_np)
        elif self._backend == "onnx":
            logits_np = self._run_batch_onnx(input_ids_np, attention_mask_np)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        # logits_np is [B, padded_len, vocab] — extract last real token
        batch_size = logits_np.shape[0]
        vocab_size = logits_np.shape[-1]
        last_logits = np.empty((batch_size, vocab_size), dtype=logits_np.dtype)

        for i in range(batch_size):
            # With left-padding, the last real token is always at position -1
            last_logits[i] = logits_np[i, -1, :]

        return last_logits

    def _run_batch_pytorch(
        self, input_ids_np: np.ndarray, attention_mask_np: np.ndarray
    ) -> np.ndarray:
        """PyTorch batched forward pass."""
        torch = _import_torch()
        device = self._config.device

        input_ids_t = torch.tensor(input_ids_np, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(
            attention_mask_np, dtype=torch.long, device=device
        )

        with self._timer:
            with torch.no_grad():
                try:
                    outputs = self._model(
                        input_ids=input_ids_t,
                        attention_mask=attention_mask_t,
                        use_cache=False,
                    )
                except TypeError:
                    outputs = self._model(
                        input_ids=input_ids_t,
                        attention_mask=attention_mask_t,
                    )
            logits = outputs.logits.cpu().float().numpy()

        self._timer.tokens_generated = int(np.sum(attention_mask_np))
        return logits

    def _run_batch_onnx(
        self, input_ids_np: np.ndarray, attention_mask_np: np.ndarray
    ) -> np.ndarray:
        """ONNX Runtime batched forward pass."""
        session = self._model

        input_names = {inp.name for inp in session.get_inputs()}
        feed: Dict[str, np.ndarray] = {}

        if "input_ids" in input_names:
            feed["input_ids"] = input_ids_np.astype(np.int64)
        else:
            feed[session.get_inputs()[0].name] = input_ids_np.astype(np.int64)

        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask_np.astype(np.int64)

        with self._timer:
            try:
                output_names = [out.name for out in session.get_outputs()]
                results = session.run(output_names, feed)
            except Exception as exc:
                raise RuntimeError(f"ONNX batch inference failed: {exc}") from exc

        self._timer.tokens_generated = int(np.sum(attention_mask_np))
        logits = results[0]  # [B, seq_len, vocab]
        return logits

    # -- convenience overrides ----------------------------------------------

    def get_next_token_logits(self, prefix: List[int]) -> np.ndarray:
        """Optimised next-token logits using KV-cache when available.

        For PyTorch with KV-cache enabled, this avoids recomputing the
        full sequence by feeding only new tokens.
        """
        self._check_closed()
        self._ensure_model_loaded()

        if not prefix:
            raise ValueError("prefix must be non-empty")

        if len(prefix) > self._config.max_seq_len:
            warnings.warn(
                f"prefix length {len(prefix)} exceeds max_seq_len "
                f"{self._config.max_seq_len}; truncating from the left"
            )
            prefix = prefix[-self._config.max_seq_len :]

        t0 = time.monotonic()

        if self._backend == "pytorch" and self._config.use_kv_cache:
            logits_last = self._next_token_logits_cached_pytorch(prefix)
        else:
            all_logits = self.get_logits(prefix)
            logits_last = all_logits[-1]

        latency = time.monotonic() - t0
        self._stats.record_call(latency, n_tokens=len(prefix))
        return logits_last

    def _next_token_logits_cached_pytorch(
        self, prefix: List[int]
    ) -> np.ndarray:
        """Get next-token logits for a prefix, leveraging KV-cache.

        Only feeds the minimal set of new tokens to the model.
        """
        torch = _import_torch()
        device = self._config.device

        prefix_key = tuple(prefix)

        # Find longest cached prefix
        past_kv = None
        cache_prefix_len = 0

        for end in range(len(prefix), 0, -1):
            sub_key = tuple(prefix[:end])
            cached = self._kv_cache.lookup(sub_key)
            if cached is not None:
                past_kv = cached
                cache_prefix_len = end
                self._stats.cache_hits += 1
                break
        else:
            self._stats.cache_misses += 1

        new_ids = prefix[cache_prefix_len:]
        if not new_ids:
            # All cached — feed last token anyway to get logits
            new_ids = [prefix[-1]]
            cache_prefix_len = len(prefix) - 1
            sub_key = tuple(prefix[:cache_prefix_len])
            past_kv = self._kv_cache.lookup(sub_key) if cache_prefix_len > 0 else None

        ids_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
        total_len = cache_prefix_len + len(new_ids)
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

        position_ids = None
        if past_kv is not None:
            position_ids = torch.arange(
                cache_prefix_len,
                cache_prefix_len + len(new_ids),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        with torch.no_grad():
            try:
                kwargs: Dict[str, Any] = {
                    "input_ids": ids_tensor,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                }
                if past_kv is not None:
                    kwargs["past_key_values"] = past_kv
                if position_ids is not None:
                    kwargs["position_ids"] = position_ids

                outputs = self._model(**kwargs)
            except TypeError:
                # Fallback for models that don't support all kwargs
                minimal = {"input_ids": ids_tensor}
                if past_kv is not None:
                    minimal["past_key_values"] = past_kv
                    minimal["attention_mask"] = attention_mask
                try:
                    outputs = self._model(**minimal)
                except TypeError:
                    outputs = self._model(input_ids=ids_tensor)

        # Store KV cache for the full prefix
        if (
            hasattr(outputs, "past_key_values")
            and outputs.past_key_values is not None
        ):
            self._kv_cache.store(prefix_key, outputs.past_key_values)

        last_logits = outputs.logits[0, -1, :].cpu().float().numpy()
        return last_logits.astype(self._config.numpy_dtype)

    # -- warmup -------------------------------------------------------------

    def warmup(self, n_steps: int = 3) -> None:
        """Run *n_steps* dummy forward passes to warm up the model.

        This triggers lazy model loading if needed, then runs short random
        sequences through the model to warm up JIT compilation, CUDA kernels,
        and any other one-time initialization.

        Args:
            n_steps: Number of warmup forward passes.
        """
        self._check_closed()
        self._ensure_model_loaded()

        logger.info(
            "Warming up %s (%s backend) for %d steps",
            self._config.model_name,
            self._backend,
            n_steps,
        )

        rng = np.random.RandomState(42)
        warm_len = min(_WARMUP_SEQ_LEN, self._config.max_seq_len)

        for i in range(n_steps):
            dummy_ids = rng.randint(
                0, self._config.vocab_size, size=warm_len
            ).tolist()
            try:
                _ = self.get_logits(dummy_ids)
            except Exception as exc:
                logger.warning("Warmup step %d failed: %s", i, exc)

        # Clear KV cache from warmup
        self._kv_cache.clear()
        # Reset stats so warmup doesn't pollute measurements
        self._stats.reset()

        logger.info("Warmup complete for %s", self._config.model_name)

    # -- benchmark ----------------------------------------------------------

    def benchmark(self, n_iterations: int = 50, seq_len: int = 64) -> Dict[str, Any]:
        """Measure throughput and latency over *n_iterations* forward passes.

        Includes backend-specific timings and KV-cache statistics.

        Args:
            n_iterations: Number of forward passes.
            seq_len: Length of the random input used in each pass.

        Returns:
            Dict with latency statistics, throughput, and cache info.
        """
        self._check_closed()
        self._ensure_model_loaded()

        rng = np.random.RandomState(0)
        seq_len = min(seq_len, self._config.max_seq_len)

        latencies: List[float] = []
        total_tokens = 0

        logger.info(
            "Benchmarking %s (%s): %d iterations, seq_len=%d",
            self._config.model_name,
            self._backend,
            n_iterations,
            seq_len,
        )

        # Clear caches for clean benchmark
        self._kv_cache.clear()
        self._stats.reset()

        overall_timer = InferenceTimer()

        with overall_timer:
            for _ in range(n_iterations):
                ids = rng.randint(
                    0, self._config.vocab_size, size=seq_len
                ).tolist()
                t0 = time.perf_counter()
                _ = self.get_next_token_logits(ids)
                latencies.append(time.perf_counter() - t0)
                total_tokens += seq_len

        overall_timer.tokens_generated = total_tokens
        lat = np.array(latencies)

        result: Dict[str, Any] = {
            "model_name": self._config.model_name,
            "backend": self._backend,
            "device": self._config.device,
            "dtype": self._config.dtype,
            "quantization": self._config.quantization,
            "n_iterations": n_iterations,
            "seq_len": seq_len,
            "total_time_s": round(overall_timer.wall_time, 4),
            "mean_latency_s": round(float(np.mean(lat)), 6),
            "std_latency_s": round(
                float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0, 6
            ),
            "min_latency_s": round(float(np.min(lat)), 6),
            "max_latency_s": round(float(np.max(lat)), 6),
            "p50_latency_s": round(float(np.percentile(lat, 50)), 6),
            "p95_latency_s": round(float(np.percentile(lat, 95)), 6),
            "p99_latency_s": round(float(np.percentile(lat, 99)), 6),
            "tokens_per_second": round(overall_timer.throughput, 2),
            "cpu_utilization": round(overall_timer.cpu_utilization, 4),
            "kv_cache": self._kv_cache.stats(),
        }

        logger.info("Benchmark results: %s", result)
        return result

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release model, tokenizer, and KV-cache resources."""
        if self._closed:
            return

        logger.info("Closing LiveLogitSource (%s)", self._config.model_name)

        # Clear KV cache
        self._kv_cache.clear()

        # Release model
        if self._model is not None:
            if self._backend == "pytorch":
                try:
                    torch = _import_torch()
                    del self._model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    del self._model
            else:
                del self._model
            self._model = None
            self._model_loaded = False

        # Release tokenizer
        self._raw_tokenizer = None
        self._tokenizer_wrapper = None
        self._tokenizer_loaded = False

        gc.collect()
        super().close()

    # -- properties ---------------------------------------------------------

    @property
    def tokenizer(self) -> Optional[TokenizerWrapper]:
        """The wrapped tokenizer (may trigger lazy loading)."""
        if self._tokenizer_wrapper is None and not self._closed:
            self._ensure_model_loaded()
        return self._tokenizer_wrapper

    @property
    def model(self) -> Any:
        """The underlying model object (may trigger lazy loading)."""
        if not self._model_loaded and not self._closed:
            self._ensure_model_loaded()
        return self._model

    @property
    def backend(self) -> str:
        """The inference backend in use (``"pytorch"`` or ``"onnx"``)."""
        return self._backend

    @property
    def kv_cache(self) -> KVCacheManager:
        """The KV-cache manager."""
        return self._kv_cache

    @property
    def batch_manager(self) -> BatchManager:
        """The batch padding/collation manager."""
        return self._batch_manager

    @property
    def is_loaded(self) -> bool:
        """Whether the model and tokenizer are loaded."""
        return self._model_loaded and self._tokenizer_loaded

    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model.

        Triggers lazy loading if the model is not yet loaded.
        """
        self._ensure_model_loaded()

        if self._backend == "pytorch":
            try:
                return ModelLoader.get_model_info(self._model)
            except Exception as exc:
                logger.warning("Could not get PyTorch model info: %s", exc)
                return {"backend": "pytorch", "error": str(exc)}

        elif self._backend == "onnx":
            try:
                session = self._model
                return {
                    "backend": "onnx",
                    "inputs": [
                        {"name": inp.name, "shape": inp.shape, "type": inp.type}
                        for inp in session.get_inputs()
                    ],
                    "outputs": [
                        {"name": out.name, "shape": out.shape, "type": out.type}
                        for out in session.get_outputs()
                    ],
                }
            except Exception as exc:
                logger.warning("Could not get ONNX model info: %s", exc)
                return {"backend": "onnx", "error": str(exc)}

        return {"backend": self._backend}

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "loaded" if self._model_loaded else "not loaded"
        return (
            f"LiveLogitSource("
            f"model={self._config.model_name!r}, "
            f"backend={self._backend!r}, "
            f"device={self._config.device!r}, "
            f"{loaded})"
        )
