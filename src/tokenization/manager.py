"""
Tokenization management module for the Diversity Decoding Arena.

Provides configurable tokenizer implementations including BPE, whitespace,
character-level, and regex-based tokenizers with caching, analysis, and
vocabulary management.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Pattern,
    Set,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TokenizerType(Enum):
    """Supported tokenizer back-ends."""

    BPE = auto()
    WORDPIECE = auto()
    SENTENCEPIECE = auto()
    CHARACTER = auto()
    WHITESPACE = auto()
    REGEX = auto()


class PaddingStrategy(Enum):
    """How sequences are padded when batching."""

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(Enum):
    """How sequences are truncated when they exceed *max_length*."""

    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpecialTokens:
    """Container for the standard special tokens used by most tokenizers."""

    bos: str = "<bos>"
    eos: str = "<eos>"
    pad: str = "<pad>"
    unk: str = "<unk>"
    sep: str = "<sep>"
    cls: str = "<cls>"
    mask: str = "<mask>"

    def as_list(self) -> List[str]:
        return [self.bos, self.eos, self.pad, self.unk, self.sep, self.cls, self.mask]

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class TokenizerConfig:
    """Full configuration for a :class:`TokenizerManager`."""

    tokenizer_type: TokenizerType = TokenizerType.BPE
    vocab_size: int = 32_000
    model_name: str = "default"
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    max_length: int = 512
    padding_strategy: PaddingStrategy = PaddingStrategy.LONGEST
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE
    lowercase: bool = False
    min_frequency: int = 1
    regex_pattern: Optional[str] = None
    cache_capacity: int = 10_000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tokenizer_type"] = self.tokenizer_type.name
        d["padding_strategy"] = self.padding_strategy.value
        d["truncation_strategy"] = self.truncation_strategy.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizerConfig":
        d = dict(d)
        d["tokenizer_type"] = TokenizerType[d["tokenizer_type"]]
        d["padding_strategy"] = PaddingStrategy(d["padding_strategy"])
        d["truncation_strategy"] = TruncationStrategy(d["truncation_strategy"])
        if isinstance(d.get("special_tokens"), dict):
            d["special_tokens"] = SpecialTokens(**d["special_tokens"])
        return cls(**d)


@dataclass
class VocabularyInfo:
    """Statistics about a tokenizer's vocabulary."""

    vocab_size: int = 0
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    token_frequencies: Dict[str, int] = field(default_factory=dict)
    coverage_statistics: Dict[str, float] = field(default_factory=dict)
    oov_rate: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Vocabulary size : {self.vocab_size}",
            f"Special tokens  : {len(self.special_token_ids)}",
            f"OOV rate        : {self.oov_rate:.4%}",
        ]
        for key, val in self.coverage_statistics.items():
            lines.append(f"  {key}: {val:.4f}")
        return "\n".join(lines)


@dataclass
class TokenizationResult:
    """Output produced by encoding a single text string."""

    input_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    token_type_ids: List[int] = field(default_factory=list)
    special_tokens_mask: List[int] = field(default_factory=list)
    offsets: List[Tuple[int, int]] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)

    # ---- dunder helpers ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.input_ids)

    def __repr__(self) -> str:
        return (
            f"TokenizationResult(num_tokens={len(self)}, "
            f"tokens={self.tokens[:8]}{'...' if len(self.tokens) > 8 else ''})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenizationResult):
            return NotImplemented
        return (
            self.input_ids == other.input_ids
            and self.attention_mask == other.attention_mask
            and self.token_type_ids == other.token_type_ids
        )

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids,
            "special_tokens_mask": self.special_tokens_mask,
            "offsets": [list(o) for o in self.offsets],
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizationResult":
        d = dict(d)
        if "offsets" in d:
            d["offsets"] = [tuple(o) for o in d["offsets"]]
        return cls(**d)

    # ---- numpy conversion -------------------------------------------------

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "input_ids": np.array(self.input_ids, dtype=np.int64),
            "attention_mask": np.array(self.attention_mask, dtype=np.int64),
            "token_type_ids": np.array(self.token_type_ids, dtype=np.int64),
        }

    # ---- padding / truncation helpers -------------------------------------

    def pad_to(self, length: int, pad_id: int = 0) -> "TokenizationResult":
        """Return a *new* result padded (or truncated) to *length*."""
        cur = len(self)
        if cur >= length:
            return TokenizationResult(
                input_ids=self.input_ids[:length],
                attention_mask=self.attention_mask[:length],
                token_type_ids=self.token_type_ids[:length],
                special_tokens_mask=self.special_tokens_mask[:length],
                offsets=self.offsets[:length],
                tokens=self.tokens[:length],
            )
        extra = length - cur
        return TokenizationResult(
            input_ids=self.input_ids + [pad_id] * extra,
            attention_mask=self.attention_mask + [0] * extra,
            token_type_ids=self.token_type_ids + [0] * extra,
            special_tokens_mask=self.special_tokens_mask + [1] * extra,
            offsets=self.offsets + [(0, 0)] * extra,
            tokens=self.tokens + ["<pad>"] * extra,
        )


# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

class TokenizerCache:
    """LRU cache for tokenization encode / decode results."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = max(1, capacity)
        self._encode_cache: OrderedDict[str, TokenizationResult] = OrderedDict()
        self._decode_cache: OrderedDict[str, str] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -- cache key helpers --------------------------------------------------

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _ids_key(ids: Sequence[int]) -> str:
        raw = ",".join(str(i) for i in ids)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -- encode cache -------------------------------------------------------

    def get_encode(self, text: str) -> Optional[TokenizationResult]:
        key = self._text_key(text)
        with self._lock:
            if key in self._encode_cache:
                self._encode_cache.move_to_end(key)
                self._hits += 1
                return copy.deepcopy(self._encode_cache[key])
            self._misses += 1
            return None

    def put_encode(self, text: str, result: TokenizationResult) -> None:
        key = self._text_key(text)
        with self._lock:
            if key in self._encode_cache:
                self._encode_cache.move_to_end(key)
                self._encode_cache[key] = copy.deepcopy(result)
                return
            if len(self._encode_cache) >= self._capacity:
                self._encode_cache.popitem(last=False)
                self._evictions += 1
            self._encode_cache[key] = copy.deepcopy(result)

    # -- decode cache -------------------------------------------------------

    def get_decode(self, ids: Sequence[int]) -> Optional[str]:
        key = self._ids_key(ids)
        with self._lock:
            if key in self._decode_cache:
                self._decode_cache.move_to_end(key)
                self._hits += 1
                return self._decode_cache[key]
            self._misses += 1
            return None

    def put_decode(self, ids: Sequence[int], text: str) -> None:
        key = self._ids_key(ids)
        with self._lock:
            if key in self._decode_cache:
                self._decode_cache.move_to_end(key)
                self._decode_cache[key] = text
                return
            if len(self._decode_cache) >= self._capacity:
                self._decode_cache.popitem(last=False)
                self._evictions += 1
            self._decode_cache[key] = text

    # -- batch helpers ------------------------------------------------------

    def batch_get_encode(
        self, texts: List[str]
    ) -> Tuple[List[Optional[TokenizationResult]], List[int]]:
        """Return cached results and indices of cache misses."""
        results: List[Optional[TokenizationResult]] = []
        miss_indices: List[int] = []
        for idx, t in enumerate(texts):
            cached = self.get_encode(t)
            results.append(cached)
            if cached is None:
                miss_indices.append(idx)
        return results, miss_indices

    def batch_put_encode(
        self, texts: List[str], results: List[TokenizationResult]
    ) -> None:
        for t, r in zip(texts, results):
            self.put_encode(t, r)

    # -- statistics ---------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / total if total else 0.0,
            "encode_cache_size": len(self._encode_cache),
            "decode_cache_size": len(self._decode_cache),
            "capacity": self._capacity,
        }

    def clear(self) -> None:
        with self._lock:
            self._encode_cache.clear()
            self._decode_cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0


# ---------------------------------------------------------------------------
# Base tokenizer
# ---------------------------------------------------------------------------

class _BaseTokenizer:
    """Shared helpers inherited by every concrete tokenizer."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._next_id: int = 0
        self._init_special_tokens()

    # -- special tokens -----------------------------------------------------

    def _init_special_tokens(self) -> None:
        for tok in self.config.special_tokens.as_list():
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token in self._vocab:
            return self._vocab[token]
        idx = self._next_id
        self._vocab[token] = idx
        self._id_to_token[idx] = token
        self._next_id += 1
        return idx

    # -- public vocab helpers -----------------------------------------------

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.config.special_tokens.unk, 0))

    def id_to_token(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, self.config.special_tokens.unk)

    def add_special_tokens(self, tokens: List[str]) -> int:
        added = 0
        for t in tokens:
            if t not in self._vocab:
                self._add_token(t)
                added += 1
        return added

    # -- serialization helpers ----------------------------------------------

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "vocab": self._vocab,
            "next_id": self._next_id,
        }

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        self._vocab = state["vocab"]
        self._next_id = state["next_id"]
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    # -- helpers for subclasses ---------------------------------------------

    def _unk_id(self) -> int:
        return self._vocab[self.config.special_tokens.unk]

    def _bos_id(self) -> int:
        return self._vocab[self.config.special_tokens.bos]

    def _eos_id(self) -> int:
        return self._vocab[self.config.special_tokens.eos]

    def _pad_id(self) -> int:
        return self._vocab[self.config.special_tokens.pad]

    def _is_special(self, token: str) -> bool:
        return token in self.config.special_tokens.as_list()

    def _preprocess(self, text: str) -> str:
        if self.config.lowercase:
            return text.lower()
        return text


# ---------------------------------------------------------------------------
# BPE tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer(_BaseTokenizer):
    """Byte-Pair Encoding tokenizer with training support.

    Training learns merge rules from a corpus.  Encoding applies the learned
    merges greedily.  A basic English character vocabulary is seeded by default.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)
        self._merges: List[Tuple[str, str]] = []
        self._merge_priority: Dict[Tuple[str, str], int] = {}
        self._bpe_cache: Dict[str, List[str]] = {}
        self._init_base_vocab()

    # -- seed vocabulary ----------------------------------------------------

    def _init_base_vocab(self) -> None:
        """Populate vocabulary with individual byte / ASCII characters."""
        for byte_val in range(256):
            ch = chr(byte_val)
            token = self._byte_token(byte_val)
            self._add_token(token)
        # common ASCII printable as raw characters
        for ch in (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:'\"-()[]{}/@#$%^&*+=<>~`|\\\n\t\r"
        ):
            self._add_token(ch)

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02X}>"

    # -- training -----------------------------------------------------------

    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        """Learn BPE merges from *texts* until *vocab_size* is reached."""
        target = vocab_size or self.config.vocab_size
        logger.info("BPE training – target vocab size %d", target)

        word_freqs: Counter = Counter()
        for text in texts:
            text = self._preprocess(text)
            words = text.strip().split()
            for w in words:
                spaced = " ".join(list(w)) + " </w>"
                word_freqs[spaced] += 1

        splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            splits[word] = word.split()

        # register all initial symbols
        for word in splits:
            for sym in splits[word]:
                self._add_token(sym)

        iteration = 0
        max_iters = target * 3  # safety bound

        while self.get_vocab_size() < target and iteration < max_iters:
            iteration += 1
            pair_freqs = self._count_pairs(splits, word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)  # type: ignore[arg-type]
            if pair_freqs[best_pair] < self.config.min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]
            self._merges.append(best_pair)
            self._merge_priority[best_pair] = len(self._merges) - 1
            self._add_token(merged_token)

            new_splits: Dict[str, List[str]] = {}
            for word, syms in splits.items():
                new_splits[word] = self._apply_merge(syms, best_pair)
            splits = new_splits

            if iteration % 500 == 0:
                logger.debug(
                    "BPE iteration %d – vocab %d – merged '%s'+'%s'",
                    iteration,
                    self.get_vocab_size(),
                    best_pair[0],
                    best_pair[1],
                )

        self._bpe_cache.clear()
        logger.info(
            "BPE training done – %d merges, vocab size %d",
            len(self._merges),
            self.get_vocab_size(),
        )

    @staticmethod
    def _count_pairs(
        splits: Dict[str, List[str]], word_freqs: Counter
    ) -> Dict[Tuple[str, str], int]:
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, syms in splits.items():
            freq = word_freqs[word]
            for i in range(len(syms) - 1):
                pair_freqs[(syms[i], syms[i + 1])] += freq
        return pair_freqs

    @staticmethod
    def _apply_merge(symbols: List[str], pair: Tuple[str, str]) -> List[str]:
        merged: List[str] = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    # -- encoding -----------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        text = self._preprocess(text)
        words = text.strip().split()
        ids: List[int] = []
        for w in words:
            tokens = self._bpe_word(w)
            for t in tokens:
                ids.append(self.token_to_id(t))
        return ids

    def encode_to_tokens(self, text: str) -> List[str]:
        text = self._preprocess(text)
        words = text.strip().split()
        tokens: List[str] = []
        for w in words:
            tokens.extend(self._bpe_word(w))
        return tokens

    def _bpe_word(self, word: str) -> List[str]:
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        symbols = list(word) + ["</w>"]

        for merge in self._merges:
            symbols = self._apply_merge(symbols, merge)
            if len(symbols) == 1:
                break

        self._bpe_cache[word] = symbols
        return symbols

    # -- decoding -----------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token(i) for i in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        # collapse multiple spaces
        text = re.sub(r" +", " ", text).strip()
        # remove special tokens
        for sp in self.config.special_tokens.as_list():
            text = text.replace(sp, "")
        return text.strip()

    # -- accessors ----------------------------------------------------------

    def get_merges(self) -> List[Tuple[str, str]]:
        return list(self._merges)

    # -- state --------------------------------------------------------------

    def _state_dict(self) -> Dict[str, Any]:
        base = super()._state_dict()
        base["merges"] = self._merges
        return base

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        super()._load_state_dict(state)
        self._merges = [tuple(m) for m in state.get("merges", [])]  # type: ignore[misc]
        self._merge_priority = {m: i for i, m in enumerate(self._merges)}
        self._bpe_cache.clear()


# ---------------------------------------------------------------------------
# Whitespace tokenizer
# ---------------------------------------------------------------------------

class WhitespaceTokenizer(_BaseTokenizer):
    """Trivial whitespace-splitting tokenizer with vocabulary building."""

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)

    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        target = vocab_size or self.config.vocab_size
        freq: Counter = Counter()
        for text in texts:
            text = self._preprocess(text)
            for w in text.strip().split():
                freq[w] += 1

        for word, count in freq.most_common(target - self.get_vocab_size()):
            if count < self.config.min_frequency:
                break
            self._add_token(word)
        logger.info("Whitespace tokenizer vocab size: %d", self.get_vocab_size())

    def encode(self, text: str) -> List[int]:
        text = self._preprocess(text)
        words = text.strip().split()
        return [self.token_to_id(w) for w in words]

    def encode_to_tokens(self, text: str) -> List[str]:
        text = self._preprocess(text)
        return text.strip().split()

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token(i) for i in ids]
        special = set(self.config.special_tokens.as_list())
        return " ".join(t for t in tokens if t not in special)


# ---------------------------------------------------------------------------
# Character tokenizer
# ---------------------------------------------------------------------------

class CharacterTokenizer(_BaseTokenizer):
    """One token per Unicode character."""

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)

    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        target = vocab_size or self.config.vocab_size
        freq: Counter = Counter()
        for text in texts:
            text = self._preprocess(text)
            for ch in text:
                freq[ch] += 1
        for ch, count in freq.most_common(target - self.get_vocab_size()):
            if count < self.config.min_frequency:
                break
            self._add_token(ch)
        logger.info("Character tokenizer vocab size: %d", self.get_vocab_size())

    def encode(self, text: str) -> List[int]:
        text = self._preprocess(text)
        return [self.token_to_id(ch) for ch in text]

    def encode_to_tokens(self, text: str) -> List[str]:
        text = self._preprocess(text)
        return list(text)

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token(i) for i in ids]
        special = set(self.config.special_tokens.as_list())
        return "".join(t for t in tokens if t not in special)


# ---------------------------------------------------------------------------
# Regex tokenizer
# ---------------------------------------------------------------------------

# GPT-2 style regex for pre-tokenization
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Simplified fallback that works with Python stdlib *re*
GPT2_PATTERN_FALLBACK = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z\u00C0-\u024F]+"
    r"| ?[0-9]+"
    r"| ?[^\s\w]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class RegexTokenizer(_BaseTokenizer):
    """Tokenizer driven by a configurable regex pattern."""

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)
        raw_pattern = config.regex_pattern or GPT2_PATTERN_FALLBACK
        try:
            self._pattern: Pattern = re.compile(raw_pattern)  # type: ignore[type-arg]
        except re.error:
            logger.warning("Regex pattern failed to compile, using fallback.")
            self._pattern = re.compile(GPT2_PATTERN_FALLBACK)

    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        target = vocab_size or self.config.vocab_size
        freq: Counter = Counter()
        for text in texts:
            text = self._preprocess(text)
            for match in self._pattern.findall(text):
                freq[match] += 1
        for tok, count in freq.most_common(target - self.get_vocab_size()):
            if count < self.config.min_frequency:
                break
            self._add_token(tok)
        logger.info("Regex tokenizer vocab size: %d", self.get_vocab_size())

    def encode(self, text: str) -> List[int]:
        text = self._preprocess(text)
        parts = self._pattern.findall(text)
        return [self.token_to_id(p) for p in parts]

    def encode_to_tokens(self, text: str) -> List[str]:
        text = self._preprocess(text)
        return self._pattern.findall(text)

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token(i) for i in ids]
        special = set(self.config.special_tokens.as_list())
        return "".join(t for t in tokens if t not in special)


# ---------------------------------------------------------------------------
# Tokenizer analyzer
# ---------------------------------------------------------------------------

class TokenizerAnalyzer:
    """Analytical utilities for inspecting tokenizer behaviour on a corpus."""

    def __init__(self, manager: "TokenizerManager") -> None:
        self._manager = manager

    # -- frequency analysis -------------------------------------------------

    def token_frequency_analysis(
        self, texts: List[str]
    ) -> Dict[str, int]:
        """Count token occurrences across *texts*."""
        freq: Counter = Counter()
        for text in texts:
            result = self._manager.encode(text)
            freq.update(result.tokens)
        return dict(freq.most_common())

    def id_frequency_analysis(self, texts: List[str]) -> Dict[int, int]:
        freq: Counter = Counter()
        for text in texts:
            result = self._manager.encode(text)
            freq.update(result.input_ids)
        return dict(freq.most_common())

    # -- vocabulary coverage ------------------------------------------------

    def vocabulary_coverage(
        self, texts: List[str]
    ) -> Dict[str, float]:
        """Measure how well the vocabulary covers the given corpus.

        Returns a dict with *covered_ratio*, *oov_ratio*, *unique_tokens*,
        *unique_oov*, and *total_tokens*.
        """
        total = 0
        oov = 0
        unique_tokens: Set[str] = set()
        oov_tokens: Set[str] = set()
        unk_id = self._manager._backend._unk_id()

        for text in texts:
            result = self._manager.encode(text)
            for tid, tok in zip(result.input_ids, result.tokens):
                total += 1
                unique_tokens.add(tok)
                if tid == unk_id:
                    oov += 1
                    oov_tokens.add(tok)

        covered = total - oov
        return {
            "covered_ratio": covered / total if total else 1.0,
            "oov_ratio": oov / total if total else 0.0,
            "unique_tokens": float(len(unique_tokens)),
            "unique_oov": float(len(oov_tokens)),
            "total_tokens": float(total),
        }

    # -- token length distribution ------------------------------------------

    def token_length_distribution(
        self, texts: List[str]
    ) -> Dict[str, float]:
        """Statistics of token string lengths."""
        lengths: List[int] = []
        for text in texts:
            result = self._manager.encode(text)
            for tok in result.tokens:
                if not self._manager._backend._is_special(tok):
                    lengths.append(len(tok))
        if not lengths:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        arr = np.array(lengths, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    # -- compression ratio --------------------------------------------------

    def compression_ratio(self, texts: List[str]) -> float:
        """Average ratio of characters to tokens (higher = more compression)."""
        total_chars = 0
        total_tokens = 0
        for text in texts:
            total_chars += len(text)
            result = self._manager.encode(text)
            total_tokens += len(result)
        if total_tokens == 0:
            return 0.0
        return total_chars / total_tokens

    # -- OOV analysis -------------------------------------------------------

    def oov_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Detailed out-of-vocabulary analysis."""
        unk_id = self._manager._backend._unk_id()
        oov_examples: List[str] = []
        oov_freq: Counter = Counter()
        total = 0
        oov_count = 0

        for text in texts:
            result = self._manager.encode(text)
            for tid, tok in zip(result.input_ids, result.tokens):
                total += 1
                if tid == unk_id:
                    oov_count += 1
                    oov_freq[tok] += 1
                    if len(oov_examples) < 50:
                        oov_examples.append(tok)

        return {
            "total_tokens": total,
            "oov_count": oov_count,
            "oov_rate": oov_count / total if total else 0.0,
            "unique_oov": len(oov_freq),
            "top_oov": oov_freq.most_common(20),
            "oov_examples": oov_examples[:20],
        }

    # -- fertility ----------------------------------------------------------

    def fertility(self, texts: List[str]) -> Dict[str, float]:
        """Tokens-per-word (fertility) statistics.

        A fertility of 1.0 means each whitespace-delimited word becomes
        exactly one token.
        """
        ratios: List[float] = []
        for text in texts:
            words = text.strip().split()
            if not words:
                continue
            result = self._manager.encode(text)
            # exclude special tokens injected by the manager
            n_content = sum(
                1 for m in result.special_tokens_mask if m == 0
            )
            ratios.append(n_content / len(words) if words else 0.0)
        if not ratios:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        arr = np.array(ratios, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    # -- combined report ----------------------------------------------------

    def full_report(self, texts: List[str]) -> Dict[str, Any]:
        return {
            "token_frequency": self.token_frequency_analysis(texts),
            "vocabulary_coverage": self.vocabulary_coverage(texts),
            "token_length_distribution": self.token_length_distribution(texts),
            "compression_ratio": self.compression_ratio(texts),
            "oov_analysis": self.oov_analysis(texts),
            "fertility": self.fertility(texts),
        }


# ---------------------------------------------------------------------------
# Tokenizer Manager (main public interface)
# ---------------------------------------------------------------------------

_BACKEND_MAP = {
    TokenizerType.BPE: BPETokenizer,
    TokenizerType.WHITESPACE: WhitespaceTokenizer,
    TokenizerType.CHARACTER: CharacterTokenizer,
    TokenizerType.REGEX: RegexTokenizer,
}


class TokenizerManager:
    """High-level tokenizer that wraps a concrete backend and provides
    caching, padding/truncation, special-token wrapping, and batch ops.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        backend_cls = _BACKEND_MAP.get(config.tokenizer_type)
        if backend_cls is None:
            raise ValueError(
                f"Unsupported tokenizer type: {config.tokenizer_type}. "
                f"Supported: {list(_BACKEND_MAP.keys())}"
            )
        self._backend: _BaseTokenizer = backend_cls(config)
        self._cache = TokenizerCache(capacity=config.cache_capacity)
        self._encode_count = 0
        self._decode_count = 0
        self._total_tokens_produced = 0

    # -- training -----------------------------------------------------------

    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        """Train the underlying backend on *texts*."""
        if hasattr(self._backend, "train"):
            self._backend.train(texts, vocab_size)  # type: ignore[attr-defined]
            self._cache.clear()
        else:
            raise NotImplementedError(
                f"{type(self._backend).__name__} does not support training."
            )

    # -- encode -------------------------------------------------------------

    def encode(self, text: str, add_special: bool = True) -> TokenizationResult:
        """Encode a single text into a :class:`TokenizationResult`."""
        cached = self._cache.get_encode(text)
        if cached is not None:
            return cached

        self._encode_count += 1
        preprocessed = self._backend._preprocess(text)

        raw_ids = self._backend.encode(preprocessed)
        raw_tokens = self._backend.encode_to_tokens(preprocessed)  # type: ignore[attr-defined]

        # build offsets by scanning through the preprocessed text
        offsets = self._compute_offsets(preprocessed, raw_tokens)

        # add BOS / EOS
        if add_special:
            raw_ids = [self._backend._bos_id()] + raw_ids + [self._backend._eos_id()]
            raw_tokens = [self.config.special_tokens.bos] + raw_tokens + [self.config.special_tokens.eos]
            offsets = [(0, 0)] + offsets + [(len(preprocessed), len(preprocessed))]

        # truncation
        raw_ids, raw_tokens, offsets = self._truncate(raw_ids, raw_tokens, offsets)

        n = len(raw_ids)
        attention_mask = [1] * n
        token_type_ids = [0] * n
        special_tokens_mask = [
            1 if self._backend._is_special(t) else 0 for t in raw_tokens
        ]

        result = TokenizationResult(
            input_ids=raw_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            special_tokens_mask=special_tokens_mask,
            offsets=offsets,
            tokens=raw_tokens,
        )
        self._total_tokens_produced += len(result)
        self._cache.put_encode(text, result)
        return result

    # -- decode -------------------------------------------------------------

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token ids back to a string."""
        cached = self._cache.get_decode(token_ids)
        if cached is not None:
            return cached

        self._decode_count += 1
        if skip_special:
            special_ids = {
                self._backend.token_to_id(t)
                for t in self.config.special_tokens.as_list()
            }
            ids = [i for i in token_ids if i not in special_ids]
        else:
            ids = list(token_ids)

        text = self._backend.decode(ids)
        self._cache.put_decode(token_ids, text)
        return text

    # -- batch encode / decode ----------------------------------------------

    def batch_encode(
        self,
        texts: List[str],
        add_special: bool = True,
        pad: bool = False,
    ) -> List[TokenizationResult]:
        """Encode a list of texts, optionally padding to equal length."""
        cached_results, miss_indices = self._cache.batch_get_encode(texts)

        for idx in miss_indices:
            cached_results[idx] = self.encode(texts[idx], add_special=add_special)

        results: List[TokenizationResult] = cached_results  # type: ignore[assignment]

        if pad and results:
            results = self._pad_batch(results)

        return results

    def batch_decode(
        self, token_id_lists: List[List[int]], skip_special: bool = True
    ) -> List[str]:
        return [self.decode(ids, skip_special=skip_special) for ids in token_id_lists]

    # -- vocabulary ---------------------------------------------------------

    def get_vocab(self) -> Dict[str, int]:
        return self._backend.get_vocab()

    def get_vocab_size(self) -> int:
        return self._backend.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self._backend.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self._backend.id_to_token(token_id)

    def add_special_tokens(self, tokens: List[str]) -> int:
        added = self._backend.add_special_tokens(tokens)
        if added:
            self._cache.clear()
        return added

    # -- vocabulary info ----------------------------------------------------

    def vocabulary_info(self, texts: Optional[List[str]] = None) -> VocabularyInfo:
        special_ids = {
            name: self._backend.token_to_id(tok)
            for name, tok in self.config.special_tokens.as_dict().items()
        }
        info = VocabularyInfo(
            vocab_size=self.get_vocab_size(),
            special_token_ids=special_ids,
        )
        if texts:
            analyzer = TokenizerAnalyzer(self)
            info.coverage_statistics = analyzer.vocabulary_coverage(texts)
            info.oov_rate = info.coverage_statistics.get("oov_ratio", 0.0)
            info.token_frequencies = analyzer.token_frequency_analysis(texts)
        return info

    # -- persistence --------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the full tokenizer state to *path* (directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / "config.json"
        state_path = path / "state.json"
        meta_path = path / "meta.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self._backend._state_dict(), f, indent=2, ensure_ascii=False)

        meta = {
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "total_tokens_produced": self._total_tokens_produced,
            "backend": type(self._backend).__name__,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Tokenizer saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TokenizerManager":
        """Load a tokenizer previously saved with :meth:`save`."""
        path = Path(path)
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = TokenizerConfig.from_dict(json.load(f))

        manager = cls(config)

        with open(path / "state.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        manager._backend._load_state_dict(state)

        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            manager._encode_count = meta.get("encode_count", 0)
            manager._decode_count = meta.get("decode_count", 0)
            manager._total_tokens_produced = meta.get("total_tokens_produced", 0)

        logger.info("Tokenizer loaded from %s", path)
        return manager

    # -- analysis -----------------------------------------------------------

    def analyzer(self) -> TokenizerAnalyzer:
        return TokenizerAnalyzer(self)

    # -- cache access -------------------------------------------------------

    def cache_statistics(self) -> Dict[str, Any]:
        return self._cache.statistics()

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- manager statistics -------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        return {
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "total_tokens_produced": self._total_tokens_produced,
            "vocab_size": self.get_vocab_size(),
            "tokenizer_type": self.config.tokenizer_type.name,
            "cache": self._cache.statistics(),
        }

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _compute_offsets(
        self, text: str, tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Best-effort character-level offsets for each token."""
        offsets: List[Tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            # try to find the token verbatim in remaining text
            clean = tok.replace("</w>", "")
            idx = text.find(clean, pos)
            if idx != -1:
                offsets.append((idx, idx + len(clean)))
                pos = idx + len(clean)
            else:
                offsets.append((pos, pos))
        return offsets

    def _truncate(
        self,
        ids: List[int],
        tokens: List[str],
        offsets: List[Tuple[int, int]],
    ) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
        strategy = self.config.truncation_strategy
        if strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            return ids, tokens, offsets
        max_len = self.config.max_length
        if len(ids) <= max_len:
            return ids, tokens, offsets
        return ids[:max_len], tokens[:max_len], offsets[:max_len]

    def _pad_batch(
        self, results: List[TokenizationResult]
    ) -> List[TokenizationResult]:
        strategy = self.config.padding_strategy
        if strategy == PaddingStrategy.DO_NOT_PAD:
            return results

        pad_id = self._backend._pad_id()

        if strategy == PaddingStrategy.MAX_LENGTH:
            target = self.config.max_length
        else:
            target = max(len(r) for r in results)

        return [r.pad_to(target, pad_id) for r in results]


# ---------------------------------------------------------------------------
# Convenience factory helpers
# ---------------------------------------------------------------------------

def create_bpe_tokenizer(
    vocab_size: int = 32_000,
    model_name: str = "bpe-default",
    **kwargs: Any,
) -> TokenizerManager:
    """Quick helper to build a BPE-backed :class:`TokenizerManager`."""
    config = TokenizerConfig(
        tokenizer_type=TokenizerType.BPE,
        vocab_size=vocab_size,
        model_name=model_name,
        **kwargs,
    )
    return TokenizerManager(config)


def create_whitespace_tokenizer(
    vocab_size: int = 50_000,
    model_name: str = "ws-default",
    lowercase: bool = True,
    **kwargs: Any,
) -> TokenizerManager:
    config = TokenizerConfig(
        tokenizer_type=TokenizerType.WHITESPACE,
        vocab_size=vocab_size,
        model_name=model_name,
        lowercase=lowercase,
        **kwargs,
    )
    return TokenizerManager(config)


def create_character_tokenizer(
    vocab_size: int = 1_000,
    model_name: str = "char-default",
    **kwargs: Any,
) -> TokenizerManager:
    config = TokenizerConfig(
        tokenizer_type=TokenizerType.CHARACTER,
        vocab_size=vocab_size,
        model_name=model_name,
        **kwargs,
    )
    return TokenizerManager(config)


def create_regex_tokenizer(
    vocab_size: int = 50_000,
    model_name: str = "regex-default",
    pattern: Optional[str] = None,
    **kwargs: Any,
) -> TokenizerManager:
    config = TokenizerConfig(
        tokenizer_type=TokenizerType.REGEX,
        vocab_size=vocab_size,
        model_name=model_name,
        regex_pattern=pattern,
        **kwargs,
    )
    return TokenizerManager(config)


# ---------------------------------------------------------------------------
# BatchTokenizer – high-level batching with numpy outputs
# ---------------------------------------------------------------------------

class BatchTokenizer:
    """Wraps :class:`TokenizerManager` for efficient bulk tokenization."""

    def __init__(self, manager: TokenizerManager) -> None:
        self._manager = manager

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = False,
    ) -> Union[List[TokenizationResult], Dict[str, np.ndarray]]:
        if isinstance(texts, str):
            texts = [texts]

        old_trunc = self._manager.config.truncation_strategy
        old_max = self._manager.config.max_length

        if truncation:
            self._manager.config.truncation_strategy = TruncationStrategy.LONGEST_FIRST
        if max_length is not None:
            self._manager.config.max_length = max_length

        try:
            results = self._manager.batch_encode(texts, pad=padding)
        finally:
            self._manager.config.truncation_strategy = old_trunc
            self._manager.config.max_length = old_max

        if return_numpy:
            return self._to_numpy(results)
        return results

    @staticmethod
    def _to_numpy(results: List[TokenizationResult]) -> Dict[str, np.ndarray]:
        return {
            "input_ids": np.array(
                [r.input_ids for r in results], dtype=np.int64
            ),
            "attention_mask": np.array(
                [r.attention_mask for r in results], dtype=np.int64
            ),
            "token_type_ids": np.array(
                [r.token_type_ids for r in results], dtype=np.int64
            ),
        }


# ---------------------------------------------------------------------------
# TokenizerComparator – compare two tokenizers side-by-side
# ---------------------------------------------------------------------------

class TokenizerComparator:
    """Compare tokenization behaviour between two managers."""

    def __init__(
        self, manager_a: TokenizerManager, manager_b: TokenizerManager
    ) -> None:
        self._a = manager_a
        self._b = manager_b

    def compare_text(self, text: str) -> Dict[str, Any]:
        ra = self._a.encode(text)
        rb = self._b.encode(text)
        return {
            "text": text,
            "a_tokens": ra.tokens,
            "b_tokens": rb.tokens,
            "a_num_tokens": len(ra),
            "b_num_tokens": len(rb),
            "a_ids": ra.input_ids,
            "b_ids": rb.input_ids,
            "token_count_diff": len(ra) - len(rb),
        }

    def compare_corpus(self, texts: List[str]) -> Dict[str, Any]:
        a_total = 0
        b_total = 0
        diffs: List[int] = []
        for text in texts:
            ra = self._a.encode(text)
            rb = self._b.encode(text)
            a_total += len(ra)
            b_total += len(rb)
            diffs.append(len(ra) - len(rb))

        arr = np.array(diffs, dtype=np.float64)
        total_chars = sum(len(t) for t in texts)
        return {
            "num_texts": len(texts),
            "a_total_tokens": a_total,
            "b_total_tokens": b_total,
            "a_compression": total_chars / a_total if a_total else 0.0,
            "b_compression": total_chars / b_total if b_total else 0.0,
            "mean_diff": float(np.mean(arr)),
            "std_diff": float(np.std(arr)),
            "a_vocab_size": self._a.get_vocab_size(),
            "b_vocab_size": self._b.get_vocab_size(),
        }

    def overlap_analysis(self) -> Dict[str, Any]:
        va = set(self._a.get_vocab().keys())
        vb = set(self._b.get_vocab().keys())
        inter = va & vb
        union = va | vb
        return {
            "a_vocab_size": len(va),
            "b_vocab_size": len(vb),
            "intersection_size": len(inter),
            "union_size": len(union),
            "jaccard_similarity": len(inter) / len(union) if union else 1.0,
            "a_only_count": len(va - vb),
            "b_only_count": len(vb - va),
            "a_only_sample": sorted(list(va - vb))[:20],
            "b_only_sample": sorted(list(vb - va))[:20],
        }


# ---------------------------------------------------------------------------
# TokenizerPipeline – chain pre/post-processing steps
# ---------------------------------------------------------------------------

class _PipelineStep:
    """Abstract base for pipeline steps."""

    def process(self, text: str) -> str:
        raise NotImplementedError

    def name(self) -> str:
        return type(self).__name__


class LowercaseStep(_PipelineStep):
    def process(self, text: str) -> str:
        return text.lower()

    def name(self) -> str:
        return "lowercase"


class StripStep(_PipelineStep):
    def process(self, text: str) -> str:
        return text.strip()

    def name(self) -> str:
        return "strip"


class NormalizeWhitespaceStep(_PipelineStep):
    def process(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def name(self) -> str:
        return "normalize_whitespace"


class RemovePunctuationStep(_PipelineStep):
    _PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

    def process(self, text: str) -> str:
        return self._PUNCT.sub("", text)

    def name(self) -> str:
        return "remove_punctuation"


class UnicodeNormalizeStep(_PipelineStep):
    """NFC normalization (requires ``unicodedata``)."""

    def __init__(self, form: str = "NFC") -> None:
        import unicodedata
        self._form = form
        self._unicodedata = unicodedata

    def process(self, text: str) -> str:
        return self._unicodedata.normalize(self._form, text)

    def name(self) -> str:
        return f"unicode_normalize({self._form})"


class RegexSubStep(_PipelineStep):
    def __init__(self, pattern: str, replacement: str) -> None:
        self._pattern = re.compile(pattern)
        self._replacement = replacement

    def process(self, text: str) -> str:
        return self._pattern.sub(self._replacement, text)

    def name(self) -> str:
        return f"regex_sub({self._pattern.pattern})"


class TokenizerPipeline:
    """Chain of text-processing steps applied before tokenization."""

    def __init__(self) -> None:
        self._steps: List[_PipelineStep] = []

    def add_step(self, step: _PipelineStep) -> "TokenizerPipeline":
        self._steps.append(step)
        return self

    def add_lowercase(self) -> "TokenizerPipeline":
        return self.add_step(LowercaseStep())

    def add_strip(self) -> "TokenizerPipeline":
        return self.add_step(StripStep())

    def add_normalize_whitespace(self) -> "TokenizerPipeline":
        return self.add_step(NormalizeWhitespaceStep())

    def add_remove_punctuation(self) -> "TokenizerPipeline":
        return self.add_step(RemovePunctuationStep())

    def add_unicode_normalize(self, form: str = "NFC") -> "TokenizerPipeline":
        return self.add_step(UnicodeNormalizeStep(form))

    def add_regex_sub(self, pattern: str, replacement: str) -> "TokenizerPipeline":
        return self.add_step(RegexSubStep(pattern, replacement))

    def process(self, text: str) -> str:
        for step in self._steps:
            text = step.process(text)
        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(t) for t in texts]

    def describe(self) -> List[str]:
        return [s.name() for s in self._steps]


# ---------------------------------------------------------------------------
# PipelinedTokenizerManager – manager with pipeline pre-processing
# ---------------------------------------------------------------------------

class PipelinedTokenizerManager:
    """Wraps a :class:`TokenizerManager` with an optional pre-processing
    :class:`TokenizerPipeline`.
    """

    def __init__(
        self,
        manager: TokenizerManager,
        pipeline: Optional[TokenizerPipeline] = None,
    ) -> None:
        self._manager = manager
        self._pipeline = pipeline or TokenizerPipeline()

    def encode(self, text: str, **kwargs: Any) -> TokenizationResult:
        text = self._pipeline.process(text)
        return self._manager.encode(text, **kwargs)

    def decode(self, ids: List[int], **kwargs: Any) -> str:
        return self._manager.decode(ids, **kwargs)

    def batch_encode(self, texts: List[str], **kwargs: Any) -> List[TokenizationResult]:
        texts = self._pipeline.process_batch(texts)
        return self._manager.batch_encode(texts, **kwargs)

    def batch_decode(self, id_lists: List[List[int]], **kwargs: Any) -> List[str]:
        return self._manager.batch_decode(id_lists, **kwargs)

    @property
    def manager(self) -> TokenizerManager:
        return self._manager

    @property
    def pipeline(self) -> TokenizerPipeline:
        return self._pipeline


# ---------------------------------------------------------------------------
# VocabularyBuilder – incremental vocabulary construction
# ---------------------------------------------------------------------------

class VocabularyBuilder:
    """Incrementally build a vocabulary from text chunks."""

    def __init__(
        self,
        max_size: int = 50_000,
        min_frequency: int = 1,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self._max_size = max_size
        self._min_frequency = min_frequency
        self._counter: Counter = Counter()
        self._special: List[str] = special_tokens or []
        self._finalized = False
        self._vocab: Dict[str, int] = {}

    def add_text(self, text: str) -> None:
        if self._finalized:
            raise RuntimeError("Vocabulary has been finalized.")
        for word in text.strip().split():
            self._counter[word] += 1

    def add_texts(self, texts: List[str]) -> None:
        for t in texts:
            self.add_text(t)

    def add_characters(self, text: str) -> None:
        if self._finalized:
            raise RuntimeError("Vocabulary has been finalized.")
        for ch in text:
            self._counter[ch] += 1

    def finalize(self) -> Dict[str, int]:
        """Build and return the final vocabulary mapping."""
        self._vocab = {}
        idx = 0
        for tok in self._special:
            self._vocab[tok] = idx
            idx += 1

        remaining = self._max_size - idx
        for word, count in self._counter.most_common(remaining):
            if count < self._min_frequency:
                break
            if word not in self._vocab:
                self._vocab[word] = idx
                idx += 1

        self._finalized = True
        return dict(self._vocab)

    @property
    def current_counts(self) -> Counter:
        return Counter(self._counter)

    @property
    def vocab(self) -> Dict[str, int]:
        if not self._finalized:
            raise RuntimeError("Call finalize() first.")
        return dict(self._vocab)

    def reset(self) -> None:
        self._counter.clear()
        self._vocab.clear()
        self._finalized = False


# ---------------------------------------------------------------------------
# TokenizerRegistry – global registry of tokenizer instances
# ---------------------------------------------------------------------------

class TokenizerRegistry:
    """Singleton registry for named :class:`TokenizerManager` instances."""

    _instance: ClassVar[Optional["TokenizerRegistry"]] = None
    _lock: ClassVar[Lock] = Lock()

    def __new__(cls) -> "TokenizerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._registry = {}  # type: ignore[attr-defined]
            return cls._instance

    def register(self, name: str, manager: TokenizerManager) -> None:
        self._registry[name] = manager

    def get(self, name: str) -> Optional[TokenizerManager]:
        return self._registry.get(name)

    def unregister(self, name: str) -> bool:
        return self._registry.pop(name, None) is not None

    def list_names(self) -> List[str]:
        return list(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# ---------------------------------------------------------------------------
# TokenAlignment – align tokens to source spans
# ---------------------------------------------------------------------------

class TokenAlignment:
    """Utilities for aligning token sequences back to source text."""

    @staticmethod
    def align_tokens_to_words(
        text: str, result: TokenizationResult
    ) -> List[List[int]]:
        """Map each whitespace-delimited word index to its token indices.

        Returns a list where ``out[word_idx]`` is the list of token indices
        that overlap with that word.
        """
        words = text.split()
        word_spans: List[Tuple[int, int]] = []
        pos = 0
        for w in words:
            start = text.find(w, pos)
            word_spans.append((start, start + len(w)))
            pos = start + len(w)

        word_to_tokens: List[List[int]] = [[] for _ in words]
        for tok_idx, (ts, te) in enumerate(result.offsets):
            if ts == te:
                continue
            for w_idx, (ws, we) in enumerate(word_spans):
                if ts < we and te > ws:
                    word_to_tokens[w_idx].append(tok_idx)
        return word_to_tokens

    @staticmethod
    def token_spans(result: TokenizationResult) -> List[Tuple[int, int]]:
        return list(result.offsets)

    @staticmethod
    def reconstruct_from_offsets(text: str, result: TokenizationResult) -> str:
        """Reconstruct text from token offsets (debug helper)."""
        parts: List[str] = []
        for s, e in result.offsets:
            if s < e <= len(text):
                parts.append(text[s:e])
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Encoding validation utilities
# ---------------------------------------------------------------------------

class EncodingValidator:
    """Validate encode/decode round-trip fidelity."""

    def __init__(self, manager: TokenizerManager) -> None:
        self._manager = manager

    def validate_roundtrip(self, text: str) -> Dict[str, Any]:
        encoded = self._manager.encode(text)
        decoded = self._manager.decode(encoded.input_ids)
        normalized_original = re.sub(r"\s+", " ", text.strip())
        normalized_decoded = re.sub(r"\s+", " ", decoded.strip())
        match = normalized_original == normalized_decoded

        # character-level edit distance (simple Levenshtein)
        dist = self._edit_distance(normalized_original, normalized_decoded)
        similarity = 1.0 - dist / max(len(normalized_original), len(normalized_decoded), 1)

        return {
            "original": text,
            "decoded": decoded,
            "exact_match": match,
            "edit_distance": dist,
            "similarity": similarity,
            "num_tokens": len(encoded),
        }

    def validate_batch(self, texts: List[str]) -> Dict[str, Any]:
        results = [self.validate_roundtrip(t) for t in texts]
        exact = sum(1 for r in results if r["exact_match"])
        sims = [r["similarity"] for r in results]
        return {
            "total": len(texts),
            "exact_matches": exact,
            "exact_match_rate": exact / len(texts) if texts else 0.0,
            "mean_similarity": float(np.mean(sims)) if sims else 0.0,
            "min_similarity": float(np.min(sims)) if sims else 0.0,
            "details": results,
        }

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Wagner-Fischer edit distance."""
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,    # insertion
                    prev[j - 1] + cost, # substitution
                )
            prev, curr = curr, prev
        return prev[n]


# ---------------------------------------------------------------------------
# SubwordRegularization – sampling multiple segmentations
# ---------------------------------------------------------------------------

class SubwordRegularization:
    """Sample alternative BPE segmentations for regularization.

    During training, stochastically dropping merges can improve model
    robustness.  This class wraps a :class:`BPETokenizer` and provides
    a ``dropout`` probability that randomly skips merges.
    """

    def __init__(
        self,
        bpe: BPETokenizer,
        dropout: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self._bpe = bpe
        self._dropout = dropout
        self._rng = np.random.RandomState(seed)

    def encode(self, text: str) -> List[int]:
        """Encode *text* with stochastic merge dropout."""
        text = self._bpe._preprocess(text)
        words = text.strip().split()
        ids: List[int] = []
        for w in words:
            tokens = self._bpe_word_dropout(w)
            for t in tokens:
                ids.append(self._bpe.token_to_id(t))
        return ids

    def encode_to_tokens(self, text: str) -> List[str]:
        text = self._bpe._preprocess(text)
        words = text.strip().split()
        tokens: List[str] = []
        for w in words:
            tokens.extend(self._bpe_word_dropout(w))
        return tokens

    def _bpe_word_dropout(self, word: str) -> List[str]:
        symbols = list(word) + ["</w>"]

        for merge in self._bpe._merges:
            if self._rng.random() < self._dropout:
                continue
            symbols = BPETokenizer._apply_merge(symbols, merge)
            if len(symbols) == 1:
                break
        return symbols

    def sample_segmentations(
        self, text: str, n: int = 5
    ) -> List[List[str]]:
        """Generate *n* random segmentations for *text*."""
        return [self.encode_to_tokens(text) for _ in range(n)]


# ---------------------------------------------------------------------------
# TokenizerBenchmark – measure throughput
# ---------------------------------------------------------------------------

class TokenizerBenchmark:
    """Quick throughput benchmark for a :class:`TokenizerManager`."""

    def __init__(self, manager: TokenizerManager) -> None:
        self._manager = manager

    def benchmark_encode(
        self,
        texts: List[str],
        warmup: int = 5,
        iterations: int = 20,
    ) -> Dict[str, float]:
        # warmup
        for text in texts[:warmup]:
            self._manager.encode(text)

        self._manager.clear_cache()

        start = time.perf_counter()
        total_tokens = 0
        for _ in range(iterations):
            for text in texts:
                r = self._manager.encode(text)
                total_tokens += len(r)
            self._manager.clear_cache()
        elapsed = time.perf_counter() - start

        total_chars = sum(len(t) for t in texts) * iterations
        return {
            "total_time_s": elapsed,
            "texts_per_second": (len(texts) * iterations) / elapsed,
            "tokens_per_second": total_tokens / elapsed,
            "chars_per_second": total_chars / elapsed,
            "avg_time_per_text_ms": (elapsed / (len(texts) * iterations)) * 1000,
        }

    def benchmark_decode(
        self,
        texts: List[str],
        warmup: int = 5,
        iterations: int = 20,
    ) -> Dict[str, float]:
        encoded = [self._manager.encode(t).input_ids for t in texts]

        for ids in encoded[:warmup]:
            self._manager.decode(ids)
        self._manager.clear_cache()

        start = time.perf_counter()
        for _ in range(iterations):
            for ids in encoded:
                self._manager.decode(ids)
            self._manager.clear_cache()
        elapsed = time.perf_counter() - start

        total = len(encoded) * iterations
        return {
            "total_time_s": elapsed,
            "sequences_per_second": total / elapsed,
            "avg_time_per_seq_ms": (elapsed / total) * 1000,
        }

    def benchmark_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        iterations: int = 10,
    ) -> Dict[str, float]:
        batches = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]

        self._manager.clear_cache()
        start = time.perf_counter()
        total_tokens = 0
        for _ in range(iterations):
            for batch in batches:
                results = self._manager.batch_encode(batch)
                total_tokens += sum(len(r) for r in results)
            self._manager.clear_cache()
        elapsed = time.perf_counter() - start

        total_texts = len(texts) * iterations
        return {
            "total_time_s": elapsed,
            "texts_per_second": total_texts / elapsed,
            "tokens_per_second": total_tokens / elapsed,
            "batch_size": batch_size,
            "num_batches": len(batches),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def quick_tokenize(
    text: str,
    tokenizer_type: TokenizerType = TokenizerType.WHITESPACE,
    lowercase: bool = True,
) -> TokenizationResult:
    """One-shot tokenization without explicit manager setup."""
    config = TokenizerConfig(
        tokenizer_type=tokenizer_type,
        lowercase=lowercase,
    )
    manager = TokenizerManager(config)
    return manager.encode(text)


def quick_batch_tokenize(
    texts: List[str],
    tokenizer_type: TokenizerType = TokenizerType.WHITESPACE,
    lowercase: bool = True,
    pad: bool = True,
) -> List[TokenizationResult]:
    config = TokenizerConfig(
        tokenizer_type=tokenizer_type,
        lowercase=lowercase,
    )
    manager = TokenizerManager(config)
    return manager.batch_encode(texts, pad=pad)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # enums
    "TokenizerType",
    "PaddingStrategy",
    "TruncationStrategy",
    # configs / results
    "SpecialTokens",
    "TokenizerConfig",
    "VocabularyInfo",
    "TokenizationResult",
    # backends
    "BPETokenizer",
    "WhitespaceTokenizer",
    "CharacterTokenizer",
    "RegexTokenizer",
    # manager
    "TokenizerManager",
    # analysis
    "TokenizerAnalyzer",
    "TokenizerCache",
    # utilities
    "BatchTokenizer",
    "TokenizerComparator",
    "TokenizerPipeline",
    "PipelinedTokenizerManager",
    "VocabularyBuilder",
    "TokenizerRegistry",
    "TokenAlignment",
    "EncodingValidator",
    "SubwordRegularization",
    "TokenizerBenchmark",
    # factories
    "create_bpe_tokenizer",
    "create_whitespace_tokenizer",
    "create_character_tokenizer",
    "create_regex_tokenizer",
    "quick_tokenize",
    "quick_batch_tokenize",
]
