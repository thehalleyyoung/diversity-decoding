"""
Data preprocessing module for the Diversity Decoding Arena.

Provides text cleaning, normalization, filtering, deduplication,
statistics computation, and prompt formatting utilities.
"""

import re
import math
import string
import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np


# ---------------------------------------------------------------------------
# 1. PreprocessingConfig
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing operations."""

    # Basic cleaning
    lowercase: bool = False
    strip_whitespace: bool = True
    remove_punctuation: bool = False
    normalize_unicode: bool = True

    # Length constraints
    min_length: int = 0
    max_length: int = 100_000
    min_words: int = 0
    max_words: int = 100_000

    # Deduplication
    remove_duplicates: bool = False
    deduplicate_threshold: float = 0.9

    # Language / encoding
    language_filter: Optional[str] = None
    encoding: str = "utf-8"

    # Additional toggles
    remove_urls: bool = False
    remove_emails: bool = False
    remove_html_tags: bool = False
    normalize_quotes: bool = False
    normalize_dashes: bool = False
    expand_contractions: bool = False

    # Tokenization
    tokenizer: str = "whitespace"
    sentence_splitter: str = "regex"

    # Pipeline preset
    preset: Optional[str] = None

    def validate(self) -> List[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors: List[str] = []
        if self.min_length < 0:
            errors.append("min_length must be >= 0")
        if self.max_length < self.min_length:
            errors.append("max_length must be >= min_length")
        if self.min_words < 0:
            errors.append("min_words must be >= 0")
        if self.max_words < self.min_words:
            errors.append("max_words must be >= min_words")
        if not 0.0 <= self.deduplicate_threshold <= 1.0:
            errors.append("deduplicate_threshold must be in [0, 1]")
        return errors

    @classmethod
    def minimal(cls) -> "PreprocessingConfig":
        """Preset that only strips whitespace."""
        return cls(
            strip_whitespace=True,
            normalize_unicode=False,
            preset="minimal",
        )

    @classmethod
    def standard(cls) -> "PreprocessingConfig":
        """Balanced preset suitable for most NLP tasks."""
        return cls(
            lowercase=True,
            strip_whitespace=True,
            normalize_unicode=True,
            remove_urls=True,
            remove_emails=True,
            remove_html_tags=True,
            normalize_quotes=True,
            normalize_dashes=True,
            remove_duplicates=True,
            preset="standard",
        )

    @classmethod
    def aggressive(cls) -> "PreprocessingConfig":
        """Heavy cleaning for noisy data."""
        return cls(
            lowercase=True,
            strip_whitespace=True,
            remove_punctuation=True,
            normalize_unicode=True,
            remove_urls=True,
            remove_emails=True,
            remove_html_tags=True,
            normalize_quotes=True,
            normalize_dashes=True,
            expand_contractions=True,
            remove_duplicates=True,
            deduplicate_threshold=0.8,
            min_words=3,
            preset="aggressive",
        )


# ---------------------------------------------------------------------------
# Compiled regex patterns used across the module
# ---------------------------------------------------------------------------

_RE_URL = re.compile(
    r"https?://(?:www\.)?[-\w]+(?:\.[-\w]+)+(?:/[^\s]*)?"
)
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z0-9\s]")
_RE_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"])'
)
_RE_SENTENCE_BOUNDARY_SIMPLE = re.compile(r"(?<=[.!?])\s+")
_RE_WORD_TOKEN = re.compile(r"\b\w+\b")
_RE_NUMBER = re.compile(r"\b\d[\d,]*\.?\d*\b")
_RE_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 2. TextPreprocessor
# ---------------------------------------------------------------------------


class TextPreprocessor:
    """Low-level text cleaning and transformation utilities."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def clean(self, text: str) -> str:
        """Apply basic cleaning according to the current config."""
        if not isinstance(text, str):
            text = str(text)

        if self.config.remove_html_tags:
            text = self.remove_html_tags(text)
        if self.config.remove_urls:
            text = self.remove_urls(text)
        if self.config.remove_emails:
            text = self.remove_emails(text)
        if self.config.strip_whitespace:
            text = text.strip()
        if self.config.normalize_unicode:
            text = self.normalize(text)
        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_punctuation:
            text = self.remove_special_characters(text)
        text = self.collapse_whitespace(text)
        return text

    def normalize(self, text: str) -> str:
        """Unicode NFC normalization and encoding round-trip."""
        text = unicodedata.normalize("NFC", text)
        try:
            text = text.encode(self.config.encoding, errors="replace").decode(
                self.config.encoding, errors="replace"
            )
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        replacements = {
            "\u00a0": " ",
            "\u200b": "",
            "\u200c": "",
            "\u200d": "",
            "\ufeff": "",
            "\u2028": "\n",
            "\u2029": "\n",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        if self.config.tokenizer == "regex":
            return _RE_WORD_TOKEN.findall(text)
        return text.split()

    def sentence_split(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text.strip():
            return []
        if self.config.sentence_splitter == "simple":
            parts = _RE_SENTENCE_BOUNDARY_SIMPLE.split(text)
        else:
            parts = _RE_SENTENCE_BOUNDARY.split(text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences

    def remove_special_characters(self, text: str) -> str:
        """Remove punctuation and special characters, keeping alphanumerics and spaces."""
        return _RE_NON_ALPHA.sub(" ", text)

    def collapse_whitespace(self, text: str) -> str:
        """Collapse runs of whitespace into single spaces / newlines."""
        text = _RE_MULTI_SPACE.sub(" ", text)
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)
        return text

    def truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate *text* to at most *max_length* characters.

        Tries to break at the last space before the limit so words are not
        split in the middle.
        """
        limit = max_length if max_length is not None else self.config.max_length
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        last_space = truncated.rfind(" ")
        if last_space > limit * 0.6:
            truncated = truncated[:last_space]
        return truncated.rstrip()

    def pad(self, text: str, target_length: int, pad_char: str = " ") -> str:
        """Right-pad *text* to *target_length* using *pad_char*."""
        if len(text) >= target_length:
            return text
        return text + pad_char * (target_length - len(text))

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return _RE_URL.sub("", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return _RE_EMAIL.sub("", text)

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags and decode common HTML entities."""
        text = _RE_HTML_TAG.sub("", text)
        html_entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
            "&#39;": "'",
            "&nbsp;": " ",
            "&ndash;": "-",
            "&mdash;": "\u2014",
            "&lsquo;": "\u2018",
            "&rsquo;": "\u2019",
            "&ldquo;": "\u201c",
            "&rdquo;": "\u201d",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        return text


# ---------------------------------------------------------------------------
# 3. PreprocessingPipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """Composable pipeline of named preprocessing steps.

    Example::

        pipe = PreprocessingPipeline()
        pipe.add_step("lower", str.lower)
        pipe.add_step("strip", str.strip)
        result = pipe.process("  HELLO  ")
        assert result == "hello"
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self._steps: List[Tuple[str, Callable[[str], str]]] = []
        self._statistics: Dict[str, Any] = {
            "total_processed": 0,
            "total_filtered": 0,
            "total_chars_before": 0,
            "total_chars_after": 0,
            "errors": 0,
            "step_times": defaultdict(float),
        }

    def add_step(
        self, name: str, func: Callable[[str], str]
    ) -> "PreprocessingPipeline":
        """Append a processing step. Returns *self* for chaining."""
        self._steps.append((name, func))
        return self

    def remove_step(self, name: str) -> "PreprocessingPipeline":
        """Remove the first step matching *name*."""
        self._steps = [(n, f) for n, f in self._steps if n != name]
        return self

    def insert_step(
        self, index: int, name: str, func: Callable[[str], str]
    ) -> "PreprocessingPipeline":
        """Insert a step at *index*."""
        self._steps.insert(index, (name, func))
        return self

    def clear_steps(self) -> "PreprocessingPipeline":
        """Remove all steps."""
        self._steps.clear()
        return self

    def list_steps(self) -> List[str]:
        """Return the names of all registered steps."""
        return [name for name, _ in self._steps]

    def process(self, text: str) -> str:
        """Run *text* through every registered step in order."""
        self._statistics["total_processed"] += 1
        self._statistics["total_chars_before"] += len(text)
        try:
            for _name, func in self._steps:
                text = func(text)
        except Exception:
            self._statistics["errors"] += 1
            raise
        self._statistics["total_chars_after"] += len(text)
        return text

    def process_batch(
        self,
        texts: List[str],
        filter_empty: bool = True,
    ) -> List[str]:
        """Process a list of texts, optionally dropping empty results."""
        results: List[str] = []
        for t in texts:
            try:
                processed = self.process(t)
                if filter_empty and not processed.strip():
                    self._statistics["total_filtered"] += 1
                    continue
                results.append(processed)
            except Exception:
                self._statistics["errors"] += 1
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Return accumulated processing statistics."""
        stats = dict(self._statistics)
        stats["step_times"] = dict(stats["step_times"])
        if stats["total_processed"] > 0:
            stats["avg_chars_before"] = (
                stats["total_chars_before"] / stats["total_processed"]
            )
            stats["avg_chars_after"] = (
                stats["total_chars_after"] / stats["total_processed"]
            )
            total_before = stats["total_chars_before"]
            if total_before > 0:
                stats["compression_ratio"] = (
                    stats["total_chars_after"] / total_before
                )
            else:
                stats["compression_ratio"] = 1.0
        else:
            stats["avg_chars_before"] = 0
            stats["avg_chars_after"] = 0
            stats["compression_ratio"] = 1.0
        return stats

    def reset_statistics(self) -> None:
        """Reset accumulated statistics to zero."""
        for key in list(self._statistics):
            if key == "step_times":
                self._statistics[key] = defaultdict(float)
            else:
                self._statistics[key] = 0

    @classmethod
    def minimal(cls) -> "PreprocessingPipeline":
        """Pipeline that only strips whitespace and collapses spaces."""
        cfg = PreprocessingConfig.minimal()
        pipe = cls(config=cfg)
        tp = TextPreprocessor(cfg)
        pipe.add_step("strip", str.strip)
        pipe.add_step("collapse_ws", tp.collapse_whitespace)
        return pipe

    @classmethod
    def standard(cls) -> "PreprocessingPipeline":
        """Balanced pipeline for general NLP tasks."""
        cfg = PreprocessingConfig.standard()
        pipe = cls(config=cfg)
        tp = TextPreprocessor(cfg)
        tn = TextNormalizer()
        pipe.add_step("clean", tp.clean)
        pipe.add_step("normalize_quotes", tn.normalize_quotes)
        pipe.add_step("normalize_dashes", tn.normalize_dashes)
        pipe.add_step("collapse_ws", tp.collapse_whitespace)
        pipe.add_step("strip", str.strip)
        return pipe

    @classmethod
    def aggressive(cls) -> "PreprocessingPipeline":
        """Heavy-duty cleaning pipeline for noisy web text."""
        cfg = PreprocessingConfig.aggressive()
        pipe = cls(config=cfg)
        tp = TextPreprocessor(cfg)
        tn = TextNormalizer()
        pipe.add_step("clean", tp.clean)
        pipe.add_step("normalize_quotes", tn.normalize_quotes)
        pipe.add_step("normalize_dashes", tn.normalize_dashes)
        pipe.add_step("expand_contractions", tn.expand_contractions)
        pipe.add_step("normalize_numbers", tn.normalize_numbers)
        pipe.add_step("collapse_ws", tp.collapse_whitespace)
        pipe.add_step("strip", str.strip)
        return pipe


# ---------------------------------------------------------------------------
# 4. TextFilter
# ---------------------------------------------------------------------------


class TextFilter:
    """Filtering utilities for lists of texts."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def filter_by_length(
        self,
        texts: List[str],
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
    ) -> List[str]:
        """Keep texts whose character length is within [min_len, max_len]."""
        lo = min_len if min_len is not None else self.config.min_length
        hi = max_len if max_len is not None else self.config.max_length
        return [t for t in texts if lo <= len(t) <= hi]

    def filter_by_word_count(
        self,
        texts: List[str],
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
    ) -> List[str]:
        """Keep texts whose word count is within [min_words, max_words]."""
        lo = min_words if min_words is not None else self.config.min_words
        hi = max_words if max_words is not None else self.config.max_words
        return [t for t in texts if lo <= len(t.split()) <= hi]

    def filter_by_language(
        self, texts: List[str], lang: str = "en"
    ) -> List[str]:
        """Heuristic language filter based on character-set statistics.

        Supports ``"en"`` (Latin), ``"zh"`` (CJK), ``"ar"`` (Arabic),
        ``"ru"`` (Cyrillic), ``"ja"`` (Japanese), ``"ko"`` (Korean).
        """
        range_map: Dict[str, Callable[[str], bool]] = {
            "en": self._is_latin_char,
            "zh": self._is_cjk_char,
            "ar": self._is_arabic_char,
            "ru": self._is_cyrillic_char,
            "ja": self._is_japanese_char,
            "ko": self._is_korean_char,
        }
        detector = range_map.get(lang, self._is_latin_char)
        results: List[str] = []
        for text in texts:
            alpha_chars = [
                c for c in text if not c.isspace() and not c.isdigit()
            ]
            if not alpha_chars:
                continue
            matching = sum(1 for c in alpha_chars if detector(c))
            ratio = matching / len(alpha_chars)
            if ratio >= 0.5:
                results.append(text)
        return results

    def filter_duplicates(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Remove near-duplicate texts using Jaccard similarity on word 3-grams."""
        thresh = (
            threshold
            if threshold is not None
            else self.config.deduplicate_threshold
        )
        if thresh >= 1.0:
            return self._filter_exact_duplicates(texts)

        keep_indices: List[int] = []
        fingerprints: List[Set[str]] = []
        for i, text in enumerate(texts):
            words = text.lower().split()
            ngrams: Set[str] = set()
            for j in range(len(words) - 2):
                ngrams.add(" ".join(words[j : j + 3]))
            if not ngrams:
                ngrams = {text.lower().strip()}

            is_dup = False
            for fp in fingerprints:
                if not fp and not ngrams:
                    is_dup = True
                    break
                intersection = len(ngrams & fp)
                union = len(ngrams | fp)
                if union > 0 and intersection / union >= thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
                fingerprints.append(ngrams)
        return [texts[i] for i in keep_indices]

    def filter_by_quality(
        self,
        texts: List[str],
        min_score: float = 0.3,
    ) -> List[str]:
        """Heuristic quality filter based on lexical diversity, length, etc."""
        results: List[str] = []
        for text in texts:
            score = self._quality_score(text)
            if score >= min_score:
                results.append(text)
        return results

    def filter_empty(self, texts: List[str]) -> List[str]:
        """Remove empty or whitespace-only texts."""
        return [t for t in texts if t.strip()]

    def filter_by_vocabulary(
        self,
        texts: List[str],
        min_vocab: int = 5,
    ) -> List[str]:
        """Keep texts that use at least *min_vocab* distinct words."""
        return [t for t in texts if len(set(t.lower().split())) >= min_vocab]

    def filter_by_regex(
        self, texts: List[str], pattern: str, keep_matching: bool = True
    ) -> List[str]:
        """Keep (or remove) texts matching a regex *pattern*."""
        compiled = re.compile(pattern)
        if keep_matching:
            return [t for t in texts if compiled.search(t)]
        return [t for t in texts if not compiled.search(t)]

    def filter_by_sentence_count(
        self,
        texts: List[str],
        min_sentences: int = 1,
        max_sentences: int = 1000,
    ) -> List[str]:
        """Keep texts with sentence count in [min_sentences, max_sentences]."""
        tp = TextPreprocessor()
        return [
            t
            for t in texts
            if min_sentences <= len(tp.sentence_split(t)) <= max_sentences
        ]

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _filter_exact_duplicates(texts: List[str]) -> List[str]:
        seen: Set[str] = set()
        results: List[str] = []
        for t in texts:
            key = t.strip().lower()
            if key not in seen:
                seen.add(key)
                results.append(t)
        return results

    def _quality_score(self, text: str) -> float:
        """Return a heuristic quality score in [0, 1]."""
        if not text.strip():
            return 0.0

        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return 0.0

        scores: List[float] = []

        # 1. Word count score
        if n_words < 3:
            scores.append(0.1)
        elif n_words < 10:
            scores.append(0.4)
        elif n_words <= 500:
            scores.append(1.0)
        else:
            scores.append(0.7)

        # 2. Type-token ratio
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / n_words if n_words else 0
        scores.append(min(ttr * 1.5, 1.0))

        # 3. Alphabetic character ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        total_non_space = sum(1 for c in text if not c.isspace())
        alpha_ratio = alpha_count / total_non_space if total_non_space else 0
        scores.append(alpha_ratio)

        # 4. Average word length
        avg_len = sum(len(w) for w in words) / n_words
        if 3 <= avg_len <= 10:
            scores.append(1.0)
        elif avg_len < 2:
            scores.append(0.2)
        else:
            scores.append(0.5)

        # 5. Sentence punctuation
        has_ending = any(text.rstrip().endswith(c) for c in ".!?")
        scores.append(1.0 if has_ending else 0.4)

        # 6. Repetition penalty
        word_counts = Counter(w.lower() for w in words)
        most_common_ratio = word_counts.most_common(1)[0][1] / n_words
        if most_common_ratio > 0.5:
            scores.append(0.1)
        elif most_common_ratio > 0.3:
            scores.append(0.5)
        else:
            scores.append(1.0)

        return sum(scores) / len(scores)

    # -- character-set detectors ---------------------------------------------

    @staticmethod
    def _is_latin_char(c: str) -> bool:
        try:
            name = unicodedata.name(c, "")
            return "LATIN" in name or c in string.ascii_letters
        except ValueError:
            return False

    @staticmethod
    def _is_cjk_char(c: str) -> bool:
        cp = ord(c)
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        )

    @staticmethod
    def _is_arabic_char(c: str) -> bool:
        cp = ord(c)
        return (
            (0x0600 <= cp <= 0x06FF)
            or (0x0750 <= cp <= 0x077F)
            or (0x08A0 <= cp <= 0x08FF)
            or (0xFB50 <= cp <= 0xFDFF)
            or (0xFE70 <= cp <= 0xFEFF)
        )

    @staticmethod
    def _is_cyrillic_char(c: str) -> bool:
        cp = ord(c)
        return (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F)

    @staticmethod
    def _is_japanese_char(c: str) -> bool:
        cp = ord(c)
        return (
            (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xFF65 <= cp <= 0xFF9F)
        )

    @staticmethod
    def _is_korean_char(c: str) -> bool:
        cp = ord(c)
        return (
            (0xAC00 <= cp <= 0xD7AF)
            or (0x1100 <= cp <= 0x11FF)
            or (0x3130 <= cp <= 0x318F)
        )


# ---------------------------------------------------------------------------
# 5. TextNormalizer
# ---------------------------------------------------------------------------


class TextNormalizer:
    """Higher-level normalization routines for text standardisation."""

    _CONTRACTIONS: Dict[str, str] = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'d": " would",
        "'ll": " will",
        "'m": " am",
        "let's": "let us",
        "it's": "it is",
        "i'm": "I am",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "what's": "what is",
        "there's": "there is",
        "here's": "here is",
        "who's": "who is",
        "how's": "how is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "isn't": "is not",
        "wasn't": "was not",
        "weren't": "were not",
        "aren't": "are not",
        "ain't": "am not",
        "shan't": "shall not",
        "mustn't": "must not",
        "needn't": "need not",
        "mightn't": "might not",
        "they're": "they are",
        "we're": "we are",
        "you're": "you are",
        "they've": "they have",
        "we've": "we have",
        "you've": "you have",
        "they'd": "they would",
        "we'd": "we would",
        "you'd": "you would",
        "they'll": "they will",
        "we'll": "we will",
        "you'll": "you will",
        "i've": "I have",
        "i'd": "I would",
        "i'll": "I will",
        "he'd": "he would",
        "she'd": "she would",
        "he'll": "he will",
        "she'll": "she will",
        "it'll": "it will",
        "it'd": "it would",
        "who'll": "who will",
        "who'd": "who would",
        "what'll": "what will",
        "what'd": "what did",
        "where'd": "where did",
        "when'd": "when did",
        "why'd": "why did",
        "how'd": "how did",
        "y'all": "you all",
        "ma'am": "madam",
        "o'clock": "of the clock",
        "'cause": "because",
        "'em": "them",
        "gonna": "going to",
        "gotta": "got to",
        "wanna": "want to",
        "kinda": "kind of",
        "sorta": "sort of",
        "lotta": "lot of",
        "lemme": "let me",
        "gimme": "give me",
        "dunno": "do not know",
        "c'mon": "come on",
    }

    _CONTRACTION_RE = re.compile(
        r"\b("
        + "|".join(
            re.escape(k)
            for k in sorted(_CONTRACTIONS, key=len, reverse=True)
        )
        + r")\b",
        re.IGNORECASE,
    )

    def normalize_unicode(self, text: str) -> str:
        """Apply NFC normalization and strip combining marks."""
        text = unicodedata.normalize("NFC", text)
        nfkd = unicodedata.normalize("NFKD", text)
        result: List[str] = []
        for ch in nfkd:
            cat = unicodedata.category(ch)
            if cat == "Mn":
                continue
            result.append(ch)
        return "".join(result)

    def normalize_quotes(self, text: str) -> str:
        """Replace curly / fancy quotes with straight ASCII quotes."""
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201A": "'",
            "\u201B": "'",
            "\u201C": '"',
            "\u201D": '"',
            "\u201E": '"',
            "\u201F": '"',
            "\u00AB": '"',
            "\u00BB": '"',
            "\u2039": "'",
            "\u203A": "'",
            "\u300C": '"',
            "\u300D": '"',
            "\u300E": '"',
            "\u300F": '"',
            "\u301D": '"',
            "\u301E": '"',
            "\u301F": '"',
            "\uFF02": '"',
            "\uFF07": "'",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def normalize_dashes(self, text: str) -> str:
        """Replace various dash characters with standard ASCII hyphens."""
        replacements = {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": " - ",
            "\u2015": " - ",
            "\u2212": "-",
            "\uFE58": " - ",
            "\uFE63": "-",
            "\uFF0D": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def normalize_numbers(self, text: str) -> str:
        """Light normalization of numbers.

        - Remove commas inside numbers (1,000 -> 1000)
        - Normalize ordinals to lowercase suffix (1ST -> 1st)
        - Replace fullwidth digits with ASCII
        """
        for i in range(10):
            text = text.replace(chr(0xFF10 + i), str(i))

        # Remove commas in numbers (repeat for multi-group)
        for _ in range(3):
            text = re.sub(r"(\d),(\d{3})", r"\1\2", text)

        text = _RE_ORDINAL.sub(
            lambda m: m.group(1) + m.group(2).lower(), text
        )
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Collapse all whitespace to single spaces and strip."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""

        def _replace(match: re.Match) -> str:
            word = match.group(0)
            lower = word.lower()
            expanded = self._CONTRACTIONS.get(lower, word)
            if word[0].isupper() and expanded[0].islower():
                expanded = expanded[0].upper() + expanded[1:]
            return expanded

        return self._CONTRACTION_RE.sub(_replace, text)

    def normalize_all(self, text: str) -> str:
        """Run all normalizations in sequence."""
        text = self.normalize_unicode(text)
        text = self.normalize_quotes(text)
        text = self.normalize_dashes(text)
        text = self.normalize_numbers(text)
        text = self.normalize_whitespace(text)
        return text


# ---------------------------------------------------------------------------
# 6. DuplicateDetector
# ---------------------------------------------------------------------------


class DuplicateDetector:
    """Detect exact and near-duplicate texts."""

    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 3,
        seed: int = 42,
    ) -> None:
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._max_hash = (1 << 32) - 1
        self._prime = 4294967311  # prime > 2^32
        self._a = self._rng.randint(
            1, self._prime, size=num_perm
        ).astype(np.uint64)
        self._b = self._rng.randint(
            0, self._prime, size=num_perm
        ).astype(np.uint64)

    def exact_duplicates(self, texts: List[str]) -> Set[int]:
        """Return indices of texts that are exact duplicates of an earlier text."""
        seen: Dict[str, int] = {}
        duplicates: Set[int] = set()
        for i, text in enumerate(texts):
            key = text.strip()
            if key in seen:
                duplicates.add(i)
            else:
                seen[key] = i
        return duplicates

    def near_duplicates(
        self,
        texts: List[str],
        threshold: float = 0.8,
    ) -> List[Set[int]]:
        """Group texts by Jaccard similarity on word n-grams.

        Returns a list of sets, each containing indices of texts in a
        near-duplicate cluster.
        """
        n = len(texts)
        shingle_sets: List[Set[str]] = []
        for text in texts:
            shingle_sets.append(self._text_to_shingles(text))

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._jaccard(shingle_sets[i], shingle_sets[j])
                if sim >= threshold:
                    union(i, j)

        clusters: Dict[int, Set[int]] = defaultdict(set)
        for i in range(n):
            clusters[find(i)].add(i)
        return [s for s in clusters.values() if len(s) > 1]

    def minhash_duplicates(
        self,
        texts: List[str],
        threshold: float = 0.8,
    ) -> List[Set[int]]:
        """Approximate near-duplicate detection using MinHash + LSH.

        Uses banding technique to efficiently find candidate pairs, then
        verifies with exact Jaccard on shingles.
        """
        n = len(texts)
        if n == 0:
            return []

        signatures = np.full(
            (n, self.num_perm), self._max_hash, dtype=np.uint64
        )
        shingle_sets: List[Set[str]] = []

        for idx, text in enumerate(texts):
            shingles = self._text_to_shingles(text)
            shingle_sets.append(shingles)
            for shingle in shingles:
                h = self._hash_shingle(shingle)
                hashes = (self._a * h + self._b) % self._prime
                signatures[idx] = np.minimum(signatures[idx], hashes)

        num_bands, rows_per_band = self._compute_lsh_params(threshold)
        candidates: Set[FrozenSet[int]] = set()

        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = min(start + rows_per_band, self.num_perm)
            buckets: Dict[bytes, List[int]] = defaultdict(list)

            for doc_idx in range(n):
                band_sig = signatures[doc_idx, start:end].tobytes()
                buckets[band_sig].append(doc_idx)

            for bucket_docs in buckets.values():
                if len(bucket_docs) > 1:
                    for i_pos in range(len(bucket_docs)):
                        for j_pos in range(i_pos + 1, len(bucket_docs)):
                            candidates.add(
                                frozenset(
                                    [bucket_docs[i_pos], bucket_docs[j_pos]]
                                )
                            )

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for pair in candidates:
            i, j = tuple(pair)
            sim = self._jaccard(shingle_sets[i], shingle_sets[j])
            if sim >= threshold:
                union(i, j)

        clusters: Dict[int, Set[int]] = defaultdict(set)
        for i in range(n):
            clusters[find(i)].add(i)
        return [s for s in clusters.values() if len(s) > 1]

    def ngram_fingerprint(self, text: str, n: Optional[int] = None) -> int:
        """Compute a deterministic hash-based fingerprint from character n-grams."""
        n = n if n is not None else self.ngram_size
        text = text.lower().strip()
        if len(text) < n:
            return (
                int(
                    hashlib.md5(text.encode("utf-8")).hexdigest(), 16
                )
                & self._max_hash
            )
        ngrams = sorted(
            set(text[i : i + n] for i in range(len(text) - n + 1))
        )
        fingerprint = hashlib.md5(
            "|".join(ngrams).encode("utf-8")
        ).hexdigest()
        return int(fingerprint, 16) & self._max_hash

    def simhash(self, text: str) -> int:
        """Compute a 64-bit SimHash fingerprint for *text*."""
        tokens = text.lower().split()
        v = [0] * 64
        for token in tokens:
            h = int(
                hashlib.md5(token.encode("utf-8")).hexdigest(), 16
            )
            for i in range(64):
                bit = (h >> i) & 1
                if bit:
                    v[i] += 1
                else:
                    v[i] -= 1
        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= 1 << i
        return fingerprint

    def simhash_distance(self, hash1: int, hash2: int) -> int:
        """Hamming distance between two SimHash fingerprints."""
        x = hash1 ^ hash2
        distance = 0
        while x:
            distance += 1
            x &= x - 1
        return distance

    # -- internal helpers ----------------------------------------------------

    def _text_to_shingles(self, text: str) -> Set[str]:
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {text.lower().strip()} if text.strip() else set()
        return {
            " ".join(words[i : i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        }

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def _hash_shingle(shingle: str) -> int:
        return (
            int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
            & 0xFFFFFFFF
        )

    def _compute_lsh_params(
        self, threshold: float
    ) -> Tuple[int, int]:
        """Choose number of bands / rows to approximate *threshold*."""
        best_bands = 1
        best_rows = self.num_perm
        best_diff = float("inf")
        for bands in range(1, self.num_perm + 1):
            rows = self.num_perm // bands
            if rows == 0:
                continue
            approx = (1.0 / bands) ** (1.0 / rows)
            diff = abs(approx - threshold)
            if diff < best_diff:
                best_diff = diff
                best_bands = bands
                best_rows = rows
        return best_bands, best_rows


# ---------------------------------------------------------------------------
# 7. TextStatistics
# ---------------------------------------------------------------------------


class TextStatistics:
    """Compute corpus-level and per-text statistics."""

    def compute_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Comprehensive statistics for a list of texts."""
        if not texts:
            return self._empty_stats()

        lengths = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        all_words: List[str] = []
        for t in texts:
            all_words.extend(t.lower().split())
        word_freq = Counter(all_words)
        vocab_size = len(word_freq)
        total_words = len(all_words)

        char_freq = Counter("".join(texts))
        total_chars = sum(char_freq.values())

        return {
            "num_texts": len(texts),
            "total_characters": sum(lengths),
            "total_words": total_words,
            "vocabulary_size": vocab_size,
            "length": {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "q25": float(np.percentile(lengths, 25)),
                "q75": float(np.percentile(lengths, 75)),
            },
            "word_count": {
                "mean": float(np.mean(word_counts)),
                "median": float(np.median(word_counts)),
                "std": float(np.std(word_counts)),
                "min": int(np.min(word_counts)),
                "max": int(np.max(word_counts)),
            },
            "type_token_ratio": self.type_token_ratio(texts),
            "hapax_legomena_ratio": self.hapax_legomena_ratio(texts),
            "vocabulary_coverage": self.vocabulary_coverage(texts),
            "yules_k": self.yules_k(texts),
            "char_entropy": self._entropy(char_freq, total_chars),
            "word_entropy": self._entropy(word_freq, total_words),
            "top_words": word_freq.most_common(20),
            "avg_word_length": (
                sum(len(w) for w in all_words) / total_words
                if total_words
                else 0
            ),
            "sentences_per_text": self._avg_sentences(texts),
        }

    def vocabulary_coverage(self, texts: List[str]) -> float:
        """Fraction of vocabulary used relative to total token count."""
        all_words: List[str] = []
        for t in texts:
            all_words.extend(t.lower().split())
        if not all_words:
            return 0.0
        vocab_size = len(set(all_words))
        return vocab_size / len(all_words)

    def type_token_ratio(self, texts: List[str]) -> float:
        """Classic type-token ratio (TTR) across all texts."""
        all_words: List[str] = []
        for t in texts:
            all_words.extend(t.lower().split())
        if not all_words:
            return 0.0
        return len(set(all_words)) / len(all_words)

    def hapax_legomena_ratio(self, texts: List[str]) -> float:
        """Ratio of words appearing exactly once (hapax legomena)."""
        all_words: List[str] = []
        for t in texts:
            all_words.extend(t.lower().split())
        if not all_words:
            return 0.0
        freq = Counter(all_words)
        hapax_count = sum(1 for _w, c in freq.items() if c == 1)
        return hapax_count / len(freq) if freq else 0.0

    def yules_k(self, texts: List[str]) -> float:
        """Yule's K measure of lexical richness.

        K = 10^4 * (M2 - N) / N^2
        where M2 = sum(i^2 * freq_of_freq(i)) and N = total tokens.
        Lower values indicate greater lexical diversity.
        """
        all_words: List[str] = []
        for t in texts:
            all_words.extend(t.lower().split())
        n = len(all_words)
        if n <= 1:
            return 0.0
        freq = Counter(all_words)
        freq_spectrum = Counter(freq.values())
        m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        k = 10_000 * (m2 - n) / (n * n)
        return k

    def per_text_statistics(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Return statistics for each individual text."""
        results: List[Dict[str, Any]] = []
        for text in texts:
            words = text.split()
            word_freq = Counter(w.lower() for w in words)
            n_words = len(words)
            vocab = len(word_freq)
            results.append(
                {
                    "length": len(text),
                    "word_count": n_words,
                    "vocabulary_size": vocab,
                    "type_token_ratio": (
                        vocab / n_words if n_words else 0
                    ),
                    "avg_word_length": (
                        sum(len(w) for w in words) / n_words
                        if n_words
                        else 0
                    ),
                    "sentence_count": (
                        len(_RE_SENTENCE_BOUNDARY.split(text))
                        if text.strip()
                        else 0
                    ),
                    "char_entropy": self._entropy(
                        Counter(text), len(text)
                    ),
                }
            )
        return results

    def corpus_diversity(self, texts: List[str]) -> Dict[str, float]:
        """Aggregate diversity measures for the entire corpus."""
        if not texts:
            return {
                "type_token_ratio": 0.0,
                "hapax_ratio": 0.0,
                "yules_k": 0.0,
                "vocabulary_coverage": 0.0,
                "mean_text_ttr": 0.0,
            }
        per_text = self.per_text_statistics(texts)
        mean_ttr = float(
            np.mean([s["type_token_ratio"] for s in per_text])
        )
        return {
            "type_token_ratio": self.type_token_ratio(texts),
            "hapax_ratio": self.hapax_legomena_ratio(texts),
            "yules_k": self.yules_k(texts),
            "vocabulary_coverage": self.vocabulary_coverage(texts),
            "mean_text_ttr": mean_ttr,
        }

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _entropy(freq: Counter, total: int) -> float:
        """Shannon entropy in bits."""
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _avg_sentences(texts: List[str]) -> float:
        counts: List[int] = []
        for t in texts:
            if t.strip():
                sents = _RE_SENTENCE_BOUNDARY.split(t)
                counts.append(len(sents))
            else:
                counts.append(0)
        return float(np.mean(counts)) if counts else 0.0

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        return {
            "num_texts": 0,
            "total_characters": 0,
            "total_words": 0,
            "vocabulary_size": 0,
            "length": {
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "q25": 0,
                "q75": 0,
            },
            "word_count": {
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0,
            },
            "type_token_ratio": 0.0,
            "hapax_legomena_ratio": 0.0,
            "vocabulary_coverage": 0.0,
            "yules_k": 0.0,
            "char_entropy": 0.0,
            "word_entropy": 0.0,
            "top_words": [],
            "avg_word_length": 0,
            "sentences_per_text": 0.0,
        }


# ---------------------------------------------------------------------------
# 8. PromptFormatter
# ---------------------------------------------------------------------------


class PromptFormatter:
    """Format prompts for various generation and evaluation tasks.

    Provides domain-specific templates and instruction helpers used by
    the Diversity Decoding Arena evaluation pipeline.
    """

    _TEMPLATES: Dict[str, str] = {
        "general": (
            "Below is a prompt. Please provide a thoughtful response.\n\n"
            "Prompt: {prompt}\n\n"
            "Response:"
        ),
        "creative_writing": (
            "You are a creative writer. Write a compelling piece based "
            "on the following prompt.\n\n"
            "Prompt: {prompt}\n\n"
            "Your writing:"
        ),
        "summarization": (
            "Please provide a concise summary of the following text.\n\n"
            "Text: {prompt}\n\n"
            "Summary:"
        ),
        "question_answering": (
            "Answer the following question accurately and concisely.\n\n"
            "Question: {prompt}\n\n"
            "Answer:"
        ),
        "translation": (
            "Translate the following text as accurately as possible.\n\n"
            "Source text: {prompt}\n\n"
            "Translation:"
        ),
        "code_generation": (
            "Write clean, well-documented code for the following task.\n\n"
            "Task: {prompt}\n\n"
            "Code:\n```"
        ),
        "dialogue": (
            "Continue the following conversation naturally.\n\n"
            "{prompt}\n\n"
            "Response:"
        ),
        "instruction_following": (
            "Follow the instruction below carefully.\n\n"
            "Instruction: {prompt}\n\n"
            "Output:"
        ),
        "reasoning": (
            "Think step by step to solve the following problem.\n\n"
            "Problem: {prompt}\n\n"
            "Solution:"
        ),
        "classification": (
            "Classify the following text into the appropriate category.\n\n"
            "Text: {prompt}\n\n"
            "Classification:"
        ),
        "paraphrase": (
            "Rewrite the following text in your own words while "
            "preserving the original meaning.\n\n"
            "Original: {prompt}\n\n"
            "Paraphrase:"
        ),
        "sentiment": (
            "Analyze the sentiment of the following text and explain "
            "your reasoning.\n\n"
            "Text: {prompt}\n\n"
            "Sentiment analysis:"
        ),
        "math": (
            "Solve the following mathematical problem. Show your work.\n\n"
            "Problem: {prompt}\n\n"
            "Solution:"
        ),
        "explanation": (
            "Explain the following concept clearly and thoroughly.\n\n"
            "Topic: {prompt}\n\n"
            "Explanation:"
        ),
        "debate": (
            "Present arguments for and against the following "
            "proposition.\n\n"
            "Proposition: {prompt}\n\n"
            "Analysis:"
        ),
        "story_continuation": (
            "Continue the following story in an engaging way.\n\n"
            "Story so far: {prompt}\n\n"
            "Continuation:"
        ),
        "brainstorming": (
            "Generate creative ideas related to the following topic.\n\n"
            "Topic: {prompt}\n\n"
            "Ideas:"
        ),
    }

    _INSTRUCTION_MODIFIERS: Dict[str, str] = {
        "be_concise": "Keep your response brief and to the point. ",
        "be_detailed": "Provide a detailed and thorough response. ",
        "be_creative": "Be creative and original in your response. ",
        "formal_tone": "Use a formal and professional tone. ",
        "casual_tone": "Use a casual and conversational tone. ",
        "academic_tone": "Use an academic and scholarly tone. ",
        "step_by_step": "Explain your reasoning step by step. ",
        "use_examples": "Include relevant examples in your response. ",
        "cite_sources": "Cite sources where applicable. ",
        "eli5": "Explain as if to a five-year-old. ",
        "expert_level": (
            "Respond at an expert level, assuming the reader has "
            "deep domain knowledge. "
        ),
        "compare_contrast": (
            "Compare and contrast different perspectives. "
        ),
        "pros_cons": "List the pros and cons. ",
        "bullet_points": "Format your response using bullet points. ",
        "numbered_list": "Format your response as a numbered list. ",
    }

    def __init__(self, default_domain: str = "general") -> None:
        self.default_domain = default_domain
        self._custom_templates: Dict[str, str] = {}

    def register_template(self, domain: str, template: str) -> None:
        """Register a custom template for a domain."""
        if "{prompt}" not in template:
            raise ValueError(
                "Template must contain '{prompt}' placeholder"
            )
        self._custom_templates[domain] = template

    def get_template(self, domain: str) -> str:
        """Retrieve the template string for *domain*."""
        if domain in self._custom_templates:
            return self._custom_templates[domain]
        if domain in self._TEMPLATES:
            return self._TEMPLATES[domain]
        return self._TEMPLATES["general"]

    def list_domains(self) -> List[str]:
        """Return all available domain names."""
        domains = set(self._TEMPLATES.keys()) | set(
            self._custom_templates.keys()
        )
        return sorted(domains)

    def format_for_generation(
        self,
        prompt: str,
        template: Optional[str] = None,
    ) -> str:
        """Format *prompt* using a raw template string.

        If *template* is ``None``, uses the default domain template.
        The template must contain a ``{prompt}`` placeholder.
        """
        if template is None:
            template = self.get_template(self.default_domain)
        prompt = prompt.strip()
        return template.format(prompt=prompt)

    def format_for_task(
        self,
        prompt: str,
        task_domain: Optional[str] = None,
    ) -> str:
        """Format *prompt* using the template for *task_domain*."""
        domain = task_domain or self.default_domain
        template = self.get_template(domain)
        return template.format(prompt=prompt.strip())

    def add_instructions(
        self,
        prompt: str,
        instructions: Union[str, List[str]],
    ) -> str:
        """Prepend instruction modifiers to *prompt*.

        *instructions* can be a list of modifier keys (looked up in
        ``_INSTRUCTION_MODIFIERS``), a single modifier key, or a raw
        instruction string.
        """
        if isinstance(instructions, str):
            instructions = [instructions]

        prefix_parts: List[str] = []
        for instr in instructions:
            if instr in self._INSTRUCTION_MODIFIERS:
                prefix_parts.append(self._INSTRUCTION_MODIFIERS[instr])
            else:
                prefix_parts.append(instr.strip() + " ")

        prefix = "".join(prefix_parts).strip()
        if prefix:
            return f"{prefix}\n\n{prompt}"
        return prompt

    def format_with_system(
        self,
        prompt: str,
        system_message: str,
        task_domain: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return a dict with ``system`` and ``user`` keys for chat APIs."""
        formatted_user = self.format_for_task(prompt, task_domain)
        return {
            "system": system_message.strip(),
            "user": formatted_user,
        }

    def format_few_shot(
        self,
        prompt: str,
        examples: List[Tuple[str, str]],
        task_domain: Optional[str] = None,
    ) -> str:
        """Format a few-shot prompt with input/output examples."""
        parts: List[str] = []
        for i, (inp, out) in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Input: {inp}")
            parts.append(f"Output: {out}")
            parts.append("")
        parts.append("Now do the same for:")
        parts.append(f"Input: {prompt.strip()}")
        parts.append("Output:")
        few_shot_block = "\n".join(parts)

        if task_domain:
            template = self.get_template(task_domain)
            return template.format(prompt=few_shot_block)
        return few_shot_block

    def format_chain_of_thought(
        self,
        prompt: str,
        task_domain: Optional[str] = None,
    ) -> str:
        """Wrap *prompt* in a chain-of-thought instruction."""
        cot_prompt = (
            f"{prompt.strip()}\n\n"
            "Let's think about this step by step:\n"
            "Step 1:"
        )
        if task_domain:
            template = self.get_template(task_domain)
            return template.format(prompt=cot_prompt)
        return cot_prompt

    def format_comparative(
        self,
        prompts: List[str],
        task_domain: Optional[str] = None,
    ) -> str:
        """Format multiple prompts for comparative evaluation."""
        parts = ["Compare the following items:\n"]
        for i, p in enumerate(prompts, 1):
            parts.append(f"Item {i}: {p.strip()}")
        parts.append("\nComparative analysis:")
        joined = "\n".join(parts)
        if task_domain:
            template = self.get_template(task_domain)
            return template.format(prompt=joined)
        return joined


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def build_pipeline(preset: str = "standard") -> PreprocessingPipeline:
    """Build a ``PreprocessingPipeline`` from a named preset.

    Supported presets: ``"minimal"``, ``"standard"``, ``"aggressive"``.
    """
    builders = {
        "minimal": PreprocessingPipeline.minimal,
        "standard": PreprocessingPipeline.standard,
        "aggressive": PreprocessingPipeline.aggressive,
    }
    builder = builders.get(preset)
    if builder is None:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {sorted(builders)}"
        )
    return builder()


def preprocess_texts(
    texts: List[str],
    config: Optional[PreprocessingConfig] = None,
    preset: Optional[str] = None,
) -> List[str]:
    """One-shot convenience function: preprocess a batch of texts.

    Either supply a ``config`` or a ``preset`` name.
    """
    if preset:
        pipe = build_pipeline(preset)
    else:
        cfg = config or PreprocessingConfig.standard()
        pipe = PreprocessingPipeline(cfg)
        tp = TextPreprocessor(cfg)
        pipe.add_step("clean", tp.clean)
        pipe.add_step("collapse_ws", tp.collapse_whitespace)
        pipe.add_step("strip", str.strip)
    return pipe.process_batch(texts)


def compute_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """One-shot convenience: compute statistics on *texts*."""
    return TextStatistics().compute_statistics(texts)


def detect_duplicates(
    texts: List[str],
    threshold: float = 0.8,
    method: str = "minhash",
) -> List[Set[int]]:
    """One-shot convenience: find near-duplicate clusters.

    *method* can be ``"exact"``, ``"jaccard"``, or ``"minhash"``.
    """
    detector = DuplicateDetector()
    if method == "exact":
        dup_indices = detector.exact_duplicates(texts)
        if dup_indices:
            return [dup_indices]
        return []
    if method == "jaccard":
        return detector.near_duplicates(texts, threshold=threshold)
    return detector.minhash_duplicates(texts, threshold=threshold)


def format_prompt(
    prompt: str,
    task_domain: str = "general",
    instructions: Optional[List[str]] = None,
) -> str:
    """One-shot convenience: format a prompt for a task domain."""
    formatter = PromptFormatter(default_domain=task_domain)
    formatted = formatter.format_for_task(prompt, task_domain)
    if instructions:
        formatted = formatter.add_instructions(formatted, instructions)
    return formatted


# ---------------------------------------------------------------------------
# BatchPreprocessor
# ---------------------------------------------------------------------------


class BatchPreprocessor:
    """Process large text collections with progress tracking."""

    def __init__(
        self,
        pipeline: Optional[PreprocessingPipeline] = None,
        text_filter: Optional[TextFilter] = None,
        dedup: Optional[DuplicateDetector] = None,
        config: Optional[PreprocessingConfig] = None,
    ) -> None:
        self.config = config or PreprocessingConfig.standard()
        self.pipeline = pipeline or PreprocessingPipeline.standard()
        self.text_filter = text_filter or TextFilter(self.config)
        self.dedup = dedup or DuplicateDetector()
        self._progress: Dict[str, Any] = {
            "total": 0,
            "processed": 0,
            "filtered": 0,
            "duplicates_removed": 0,
        }

    def process(
        self,
        texts: List[str],
        deduplicate: bool = True,
        filter_empty: bool = True,
        min_quality: Optional[float] = None,
    ) -> List[str]:
        """Full preprocessing pipeline for a batch of texts.

        1. Run the text pipeline on each text.
        2. Filter empty texts.
        3. Optionally filter by quality.
        4. Optionally deduplicate.
        """
        self._progress["total"] = len(texts)

        results = self.pipeline.process_batch(
            texts, filter_empty=filter_empty
        )
        self._progress["processed"] = len(results)

        if min_quality is not None:
            before = len(results)
            results = self.text_filter.filter_by_quality(
                results, min_quality
            )
            self._progress["filtered"] += before - len(results)

        before = len(results)
        results = self.text_filter.filter_by_length(results)
        self._progress["filtered"] += before - len(results)

        before = len(results)
        results = self.text_filter.filter_by_word_count(results)
        self._progress["filtered"] += before - len(results)

        if deduplicate:
            before = len(results)
            results = self.text_filter.filter_duplicates(
                results,
                threshold=self.config.deduplicate_threshold,
            )
            self._progress["duplicates_removed"] = (
                before - len(results)
            )

        return results

    def get_progress(self) -> Dict[str, Any]:
        """Return current progress information."""
        return dict(self._progress)

    def process_chunks(
        self,
        texts: List[str],
        chunk_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Process texts in chunks to manage memory usage."""
        all_results: List[str] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            processed = self.process(chunk, **kwargs)
            all_results.extend(processed)
        if kwargs.get("deduplicate", True):
            all_results = self.text_filter.filter_duplicates(
                all_results,
                threshold=self.config.deduplicate_threshold,
            )
        return all_results


# ---------------------------------------------------------------------------
# TextAugmenter
# ---------------------------------------------------------------------------


class TextAugmenter:
    """Lightweight text augmentation for diversity analysis.

    These augmentations intentionally preserve semantic meaning so that
    diversity metrics can be evaluated on controlled variations.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    def synonym_swap(
        self, text: str, swap_ratio: float = 0.1
    ) -> str:
        """Randomly swap words with casing-changed variants.

        This is a placeholder interface; a real implementation would
        use a thesaurus or word-vector neighbours.
        """
        words = text.split()
        if not words:
            return text
        n_swaps = max(1, int(len(words) * swap_ratio))
        indices = self._rng.choice(
            len(words),
            size=min(n_swaps, len(words)),
            replace=False,
        )
        for idx in indices:
            word = words[idx]
            if word.islower():
                words[idx] = word.capitalize()
            else:
                words[idx] = word.lower()
        return " ".join(words)

    def random_deletion(
        self, text: str, delete_ratio: float = 0.1
    ) -> str:
        """Randomly delete words from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
        n_delete = max(1, int(len(words) * delete_ratio))
        indices = set(
            self._rng.choice(
                len(words),
                size=min(n_delete, len(words) - 1),
                replace=False,
            )
        )
        return " ".join(
            w for i, w in enumerate(words) if i not in indices
        )

    def random_swap(self, text: str, n_swaps: int = 1) -> str:
        """Randomly swap adjacent word pairs."""
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n_swaps):
            idx = self._rng.randint(0, len(words) - 1)
            j = (idx + 1) % len(words)
            words[idx], words[j] = words[j], words[idx]
        return " ".join(words)

    def random_insertion(
        self, text: str, n_insertions: int = 1
    ) -> str:
        """Insert random copies of existing words at random positions."""
        words = text.split()
        if not words:
            return text
        for _ in range(n_insertions):
            word = words[self._rng.randint(0, len(words))]
            pos = self._rng.randint(0, len(words) + 1)
            words.insert(pos, word)
        return " ".join(words)

    def case_augment(self, text: str) -> str:
        """Randomly change casing of characters."""
        chars = list(text)
        for i in range(len(chars)):
            if self._rng.random() < 0.1 and chars[i].isalpha():
                chars[i] = (
                    chars[i].upper()
                    if chars[i].islower()
                    else chars[i].lower()
                )
        return "".join(chars)

    def generate_variations(
        self,
        text: str,
        n: int = 5,
        methods: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate *n* augmented variations of *text*."""
        available = {
            "synonym_swap": self.synonym_swap,
            "random_deletion": self.random_deletion,
            "random_swap": self.random_swap,
            "random_insertion": self.random_insertion,
            "case_augment": self.case_augment,
        }
        if methods is None:
            methods = list(available.keys())
        variations: List[str] = []
        for i in range(n):
            method_name = methods[i % len(methods)]
            func = available.get(method_name, self.synonym_swap)
            variations.append(func(text))
        return variations


# ---------------------------------------------------------------------------
# Module-level __all__
# ---------------------------------------------------------------------------

__all__ = [
    "PreprocessingConfig",
    "TextPreprocessor",
    "PreprocessingPipeline",
    "TextFilter",
    "TextNormalizer",
    "DuplicateDetector",
    "TextStatistics",
    "PromptFormatter",
    "BatchPreprocessor",
    "TextAugmenter",
    "build_pipeline",
    "preprocess_texts",
    "compute_text_statistics",
    "detect_duplicates",
    "format_prompt",
]
