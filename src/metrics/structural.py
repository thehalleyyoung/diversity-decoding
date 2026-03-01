"""
Structural diversity metrics for the Diversity Decoding Arena.

Provides metrics that measure syntactic, parse-tree, code-structural,
and text-structural diversity across sets of generated texts.  All
metrics are self-contained with no external NLP library dependencies —
heuristic POS tagging, lightweight constituency parsing, and rule-based
clause detection are implemented from scratch.
"""

from __future__ import annotations

import logging
import math
import re
import string
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
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
from scipy.spatial.distance import pdist, squareform

from .diversity import tokenize_simple, extract_ngrams, ngram_overlap

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants / regex helpers
# ──────────────────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'(])|(?<=[.!?])$'
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")

# Word-lists used by the heuristic POS tagger
_DETERMINERS = frozenset(
    ["the", "a", "an", "this", "that", "these", "those", "my", "your",
     "his", "her", "its", "our", "their", "some", "any", "no", "every",
     "each", "all", "both", "few", "several", "many", "much", "enough",
     "such", "what", "which", "whatever", "whichever"]
)

_PREPOSITIONS = frozenset(
    ["in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
     "into", "through", "during", "before", "after", "above", "below",
     "between", "under", "over", "without", "within", "along", "across",
     "behind", "beyond", "towards", "upon", "among", "against", "around",
     "beside", "beneath", "despite", "except", "near", "since", "until",
     "throughout", "past", "onto", "off", "up", "down", "out"]
)

_CONJUNCTIONS = frozenset(
    ["and", "but", "or", "nor", "for", "yet", "so", "because", "although",
     "though", "while", "whereas", "if", "unless", "until", "since",
     "whether", "however", "therefore", "moreover", "furthermore",
     "nevertheless", "otherwise", "hence", "thus", "besides", "meanwhile"]
)

_PRONOUNS = frozenset(
    ["i", "me", "my", "mine", "myself", "you", "your", "yours",
     "yourself", "yourselves", "he", "him", "his", "himself", "she",
     "her", "hers", "herself", "it", "its", "itself", "we", "us",
     "our", "ours", "ourselves", "they", "them", "their", "theirs",
     "themselves", "who", "whom", "whose", "which", "that", "what",
     "this", "these", "those", "one", "ones", "somebody", "someone",
     "something", "anybody", "anyone", "anything", "nobody", "nothing",
     "everybody", "everyone", "everything"]
)

_AUXILIARY_VERBS = frozenset(
    ["is", "am", "are", "was", "were", "be", "been", "being",
     "have", "has", "had", "having",
     "do", "does", "did",
     "will", "would", "shall", "should", "may", "might", "can",
     "could", "must", "need", "dare", "ought"]
)

_COMMON_ADVERBS = frozenset(
    ["very", "really", "quite", "rather", "too", "also", "just", "still",
     "already", "always", "never", "often", "sometimes", "usually",
     "here", "there", "now", "then", "today", "tomorrow", "yesterday",
     "soon", "well", "badly", "quickly", "slowly", "carefully",
     "hardly", "nearly", "almost", "perhaps", "probably", "certainly",
     "definitely", "possibly", "only", "even", "again", "ever",
     "not", "n't", "merely", "simply", "actually", "apparently",
     "basically", "clearly", "essentially", "exactly", "generally",
     "increasingly", "indeed", "largely", "literally", "mostly",
     "naturally", "obviously", "particularly", "precisely", "primarily",
     "recently", "significantly", "specifically", "substantially",
     "typically", "ultimately", "unfortunately", "virtually"]
)

_COMMON_ADJECTIVES_SUFFIXES = (
    "able", "ible", "al", "ful", "ive", "ous", "ious", "less",
    "ish", "ic", "ical", "ent", "ant", "ary", "ory",
)

_VERB_SUFFIXES = ("ing", "ed", "ize", "ise", "ify", "ate")

_NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ence", "ance",
    "er", "or", "ist", "ism", "ship", "dom", "hood",
)

_CLAUSE_MARKERS = frozenset(
    ["that", "which", "who", "whom", "whose", "where", "when",
     "while", "because", "since", "although", "though", "if",
     "unless", "until", "after", "before", "whether", "as",
     "once", "whereas", "wherever", "whenever", "however"]
)

_CODE_KEYWORDS = frozenset(
    ["if", "else", "elif", "for", "while", "do", "switch", "case",
     "try", "catch", "except", "finally", "return", "break",
     "continue", "pass", "yield", "raise", "throw", "import",
     "from", "class", "def", "function", "var", "let", "const",
     "new", "delete", "with", "as", "lambda", "async", "await",
     "public", "private", "protected", "static", "void", "int",
     "float", "double", "string", "bool", "boolean", "null",
     "none", "true", "false", "self", "this", "super", "extends",
     "implements", "interface", "abstract", "enum", "struct",
     "typedef", "namespace", "package", "module", "export",
     "default", "require", "include", "print", "println",
     "printf", "console", "log", "assert", "type", "fn",
     "match", "impl", "trait", "pub", "use", "mod", "crate",
     "extern", "unsafe", "mut", "ref", "val", "fun", "object",
     "companion", "data", "sealed", "open", "override", "init",
     "deinit", "guard", "defer", "select", "chan", "go",
     "goroutine", "range"]
)


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SyntacticPattern:
    """Represents a syntactic pattern found in text."""
    pattern_type: str
    tokens: Tuple[str, ...]
    frequency: int = 1
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "tokens": list(self.tokens),
            "frequency": self.frequency,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntacticPattern":
        return cls(
            pattern_type=d["pattern_type"],
            tokens=tuple(d["tokens"]),
            frequency=d.get("frequency", 1),
            depth=d.get("depth", 0),
        )

    def __hash__(self) -> int:
        return hash((self.pattern_type, self.tokens))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntacticPattern):
            return NotImplemented
        return (self.pattern_type == other.pattern_type
                and self.tokens == other.tokens)


@dataclass
class ParseNode:
    """A node in a lightweight parse tree."""
    label: str
    children: List["ParseNode"] = field(default_factory=list)
    word: Optional[str] = None
    head_index: int = -1
    depth: int = 0

    # ── tree queries ──────────────────────────────────────────────
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_preterminal(self) -> bool:
        return (len(self.children) == 1 and self.children[0].is_leaf())

    def height(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(c.height() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def leaf_count(self) -> int:
        if self.is_leaf():
            return 1
        return sum(c.leaf_count() for c in self.children)

    def leaves(self) -> List["ParseNode"]:
        if self.is_leaf():
            return [self]
        result: List[ParseNode] = []
        for c in self.children:
            result.extend(c.leaves())
        return result

    def all_labels(self) -> List[str]:
        result = [self.label]
        for c in self.children:
            result.extend(c.all_labels())
        return result

    def subtrees(self) -> List["ParseNode"]:
        result = [self]
        for c in self.children:
            result.extend(c.subtrees())
        return result

    def branching_factor(self) -> float:
        internal = [n for n in self.subtrees() if not n.is_leaf()]
        if not internal:
            return 0.0
        return np.mean([len(n.children) for n in internal])

    def to_bracket_string(self) -> str:
        if self.is_leaf():
            return self.word if self.word else self.label
        child_strs = " ".join(c.to_bracket_string() for c in self.children)
        return f"({self.label} {child_strs})"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"label": self.label}
        if self.word is not None:
            d["word"] = self.word
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.head_index >= 0:
            d["head_index"] = self.head_index
        d["depth"] = self.depth
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParseNode":
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            label=d["label"],
            children=children,
            word=d.get("word"),
            head_index=d.get("head_index", -1),
            depth=d.get("depth", 0),
        )

    def copy(self) -> "ParseNode":
        return ParseNode(
            label=self.label,
            children=[c.copy() for c in self.children],
            word=self.word,
            head_index=self.head_index,
            depth=self.depth,
        )


@dataclass
class StructuralConfig:
    """Configuration for structural diversity analysis."""
    max_ngram: int = 4
    min_pattern_frequency: int = 1
    max_tree_depth: int = 20
    tree_kernel_lambda: float = 0.5
    tree_kernel_sigma: float = 1.0
    use_subtree_kernel: bool = True
    pos_pattern_window: int = 5
    clause_boundary_markers: List[str] = field(
        default_factory=lambda: list(_CLAUSE_MARKERS)
    )
    code_indent_size: int = 4
    min_texts_for_diversity: int = 2
    sentence_length_bins: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 30, 50]
    )
    paragraph_length_bins: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 20]
    )
    discount_factor: float = 0.9
    complexity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "syntactic": 0.25,
            "parse_tree": 0.25,
            "text_structure": 0.25,
            "code_structure": 0.25,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructuralConfig":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────
# Helper / utility functions
# ──────────────────────────────────────────────────────────────────────

def tokenize_to_sentences(text: str) -> List[str]:
    """Split *text* into sentences using regex heuristics."""
    text = text.strip()
    if not text:
        return []
    abbreviations = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
                     "st.", "vs.", "etc.", "inc.", "ltd.", "co.", "e.g.",
                     "i.e.", "vol.", "dept.", "est.", "approx.", "fig.",
                     "no.", "gen.", "govt.", "corp."}
    placeholder_map: Dict[str, str] = {}
    modified = text
    for abbr in abbreviations:
        if abbr in modified.lower():
            placeholder = f"__ABBR{len(placeholder_map)}__"
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            placeholder_map[placeholder] = abbr
            modified = pattern.sub(placeholder.replace(".", ""), modified)

    raw_sentences = _SENTENCE_SPLIT_RE.split(modified)
    sentences: List[str] = []
    for raw in raw_sentences:
        s = raw.strip()
        if not s:
            continue
        for placeholder, original in placeholder_map.items():
            s = s.replace(placeholder.replace(".", ""), original)
        sentences.append(s)

    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """Split *text* into paragraphs on blank-line boundaries."""
    parts = _PARAGRAPH_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def compute_tree_height(node: ParseNode) -> int:
    """Return the height of a parse tree rooted at *node*."""
    return node.height()


def count_tree_nodes(node: ParseNode) -> int:
    """Return the total number of nodes in the tree."""
    return node.node_count()


def levenshtein_distance(seq_a: Sequence[str],
                         seq_b: Sequence[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    return prev[n]


def longest_common_subsequence(seq_a: Sequence[str],
                               seq_b: Sequence[str]) -> List[str]:
    """Return the longest common subsequence of two token sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return []
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    result: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq_a[i - 1] == seq_b[j - 1]:
            result.append(seq_a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    result.reverse()
    return result


def compute_compression_ratio_diversity(texts: List[str]) -> Dict[str, float]:
    """Measure diversity via compression-ratio variation across *texts*.

    The idea: structurally similar texts compress at similar ratios; if the
    ratios differ a lot the texts are structurally diverse.
    """
    if not texts:
        return {"mean_ratio": 0.0, "std_ratio": 0.0, "diversity": 0.0}
    ratios: List[float] = []
    for t in texts:
        raw = t.encode("utf-8")
        if len(raw) == 0:
            ratios.append(1.0)
            continue
        compressed = zlib.compress(raw, 9)
        ratios.append(len(compressed) / len(raw))

    arr = np.array(ratios, dtype=np.float64)
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr))
    diversity = std_r / (mean_r + 1e-12)

    if len(texts) >= 2:
        concat_all = "\n\n".join(texts).encode("utf-8")
        compressed_all = zlib.compress(concat_all, 9)
        sum_individual = sum(
            len(zlib.compress(t.encode("utf-8"), 9)) for t in texts
        )
        if sum_individual > 0:
            ncd = (len(compressed_all) - min(
                len(zlib.compress(t.encode("utf-8"), 9)) for t in texts
            )) / max(
                len(zlib.compress(t.encode("utf-8"), 9)) for t in texts
            )
            diversity = max(diversity, min(ncd, 1.0))

    return {
        "mean_ratio": mean_r,
        "std_ratio": std_r,
        "diversity": float(np.clip(diversity, 0.0, 1.0)),
    }


def extract_ngram_patterns(
    tokens: Sequence[str], n: int, pattern_type: str = "token"
) -> List[SyntacticPattern]:
    """Extract n-gram patterns from *tokens* and wrap in SyntacticPattern."""
    if len(tokens) < n:
        return []
    counter: Counter[Tuple[str, ...]] = Counter()
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i: i + n])
        counter[gram] += 1
    return [
        SyntacticPattern(
            pattern_type=pattern_type,
            tokens=gram,
            frequency=freq,
            depth=0,
        )
        for gram, freq in counter.most_common()
    ]


def _shannon_entropy(counter: Counter) -> float:
    """Shannon entropy in nats from a Counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return ent


def _normalised_entropy(counter: Counter) -> float:
    """Entropy normalised to [0, 1]."""
    n_types = len(counter)
    if n_types <= 1:
        return 0.0
    ent = _shannon_entropy(counter)
    return ent / math.log(n_types)


def _jaccard(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _cosine_similarity_counters(c1: Counter, c2: Counter) -> float:
    """Cosine similarity between two Counters treated as sparse vectors."""
    keys = set(c1) | set(c2)
    if not keys:
        return 1.0
    dot = sum(c1.get(k, 0) * c2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(v * v for v in c1.values()))
    mag2 = math.sqrt(sum(v * v for v in c2.values()))
    if mag1 < 1e-12 or mag2 < 1e-12:
        return 0.0
    return dot / (mag1 * mag2)


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Normalised histogram intersection similarity."""
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    if h1.sum() < 1e-12 or h2.sum() < 1e-12:
        return 0.0
    h1n = h1 / h1.sum()
    h2n = h2 / h2.sum()
    return float(np.minimum(h1n, h2n).sum())


def _bin_values(values: List[float], bins: List[int]) -> np.ndarray:
    """Bin a list of float values into a histogram with the given edges."""
    edges = [0] + sorted(bins) + [10**9]
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    for v in values:
        for bi in range(len(edges) - 1):
            if edges[bi] <= v < edges[bi + 1]:
                hist[bi] += 1
                break
    return hist


def _mean_pairwise(values: List[float]) -> float:
    """Mean of all pairwise absolute differences."""
    if len(values) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += abs(values[i] - values[j])
            count += 1
    return total / count if count else 0.0


# ──────────────────────────────────────────────────────────────────────
# 4. SyntacticDiversityAnalyzer
# ──────────────────────────────────────────────────────────────────────

class SyntacticDiversityAnalyzer:
    """Analyse syntactic diversity using heuristic POS tagging and
    rule-based clause detection — no external NLP libraries required."""

    def __init__(self, config: Optional[StructuralConfig] = None) -> None:
        self.config = config or StructuralConfig()

    # ── heuristic POS tagger ─────────────────────────────────────
    def heuristic_pos_tagger(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Rule-based POS tagger.  Returns list of (token, tag) pairs.

        Tag set (simplified Penn-style):
            DT  – determiner
            PRP – pronoun
            IN  – preposition / subordinating conjunction
            CC  – coordinating conjunction
            MD  – modal
            VB  – verb (base)
            VBG – verb gerund / present participle
            VBD – verb past tense
            VBN – verb past participle (used after auxiliaries)
            RB  – adverb
            JJ  – adjective
            NN  – noun
            NNP – proper noun
            CD  – cardinal number
            .   – punctuation
            SYM – symbol
            UH  – interjection
        """
        tagged: List[Tuple[str, str]] = []
        lower_tokens = [t.lower() for t in tokens]

        for idx, tok in enumerate(tokens):
            low = lower_tokens[idx]
            prev_tag = tagged[-1][1] if tagged else "START"
            prev2_tag = tagged[-2][1] if len(tagged) >= 2 else "START"

            # punctuation
            if _PUNCT_RE.fullmatch(tok):
                tagged.append((tok, "."))
                continue

            # numbers
            if re.fullmatch(r"-?\d+(\.\d+)?%?", tok):
                tagged.append((tok, "CD"))
                continue

            # determiners
            if low in _DETERMINERS:
                tagged.append((tok, "DT"))
                continue

            # pronouns
            if low in _PRONOUNS:
                tagged.append((tok, "PRP"))
                continue

            # prepositions / subordinating conjunctions
            if low in _PREPOSITIONS:
                tagged.append((tok, "IN"))
                continue

            # coordinating conjunctions
            if low in ("and", "but", "or", "nor"):
                tagged.append((tok, "CC"))
                continue

            # other conjunctions treated as IN (subordinating)
            if low in _CONJUNCTIONS:
                tagged.append((tok, "IN"))
                continue

            # modals / auxiliary verbs
            if low in _AUXILIARY_VERBS:
                if low in ("will", "would", "shall", "should", "may",
                           "might", "can", "could", "must"):
                    tagged.append((tok, "MD"))
                else:
                    tagged.append((tok, "VB"))
                continue

            # adverbs (known list)
            if low in _COMMON_ADVERBS:
                tagged.append((tok, "RB"))
                continue

            # ----- morphological rules -----

            # gerund / present participle
            if low.endswith("ing") and len(low) > 4:
                if prev_tag in ("VB", "MD", "RB", "IN", "START", "."):
                    tagged.append((tok, "VBG"))
                elif prev_tag in ("DT", "JJ", "PRP"):
                    tagged.append((tok, "NN"))
                else:
                    tagged.append((tok, "VBG"))
                continue

            # past tense / past participle
            if low.endswith("ed") and len(low) > 3:
                if prev_tag in ("VB", "MD", "have", "has", "had"):
                    tagged.append((tok, "VBN"))
                elif prev_tag in ("DT", "JJ", "RB"):
                    tagged.append((tok, "JJ"))
                else:
                    tagged.append((tok, "VBD"))
                continue

            # adverb -ly
            if low.endswith("ly") and len(low) > 3:
                tagged.append((tok, "RB"))
                continue

            # adjective suffixes
            if any(low.endswith(suf) for suf in _COMMON_ADJECTIVES_SUFFIXES):
                if prev_tag in ("DT", "RB", "JJ", "CC"):
                    tagged.append((tok, "JJ"))
                    continue

            # verb suffixes
            if any(low.endswith(suf) for suf in _VERB_SUFFIXES):
                if prev_tag in ("DT", "PRP", "NN", "NNP", "CD"):
                    tagged.append((tok, "VB"))
                    continue

            # noun suffixes
            if any(low.endswith(suf) for suf in _NOUN_SUFFIXES):
                tagged.append((tok, "NN"))
                continue

            # proper noun (capitalised, not sentence-start)
            if tok[0].isupper() and idx > 0 and prev_tag != ".":
                tagged.append((tok, "NNP"))
                continue

            # ----- context-based fallback -----

            # after determiner / adjective → noun
            if prev_tag in ("DT", "JJ", "CD"):
                tagged.append((tok, "NN"))
                continue

            # after modal / auxiliary → verb
            if prev_tag in ("MD", "VB", "RB") and prev2_tag != "DT":
                tagged.append((tok, "VB"))
                continue

            # after pronoun → verb
            if prev_tag == "PRP":
                tagged.append((tok, "VB"))
                continue

            # after noun → verb (simple SVO assumption)
            if prev_tag in ("NN", "NNP", "PRP") and not low.endswith("s"):
                tagged.append((tok, "VB"))
                continue

            # default: noun
            tagged.append((tok, "NN"))

        return tagged

    # ── clause boundaries ────────────────────────────────────────
    def extract_clause_boundaries(
        self, tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Identify clause spans as (start, end) index pairs.

        Uses punctuation, conjunctions, and subordinating markers.
        """
        if not tokens:
            return []
        lower = [t.lower() for t in tokens]
        boundaries: List[int] = [0]

        for i, tok in enumerate(lower):
            if tok in (",", ";", ":", "—", "--"):
                if i + 1 < len(lower) and lower[i + 1] in _CLAUSE_MARKERS:
                    boundaries.append(i + 1)
                elif i + 1 < len(lower) and lower[i + 1] in ("and", "but", "or"):
                    boundaries.append(i + 1)
            elif tok in _CLAUSE_MARKERS and i > 0:
                boundaries.append(i)
            elif tok in (".", "!", "?"):
                if i + 1 < len(lower):
                    boundaries.append(i + 1)

        boundaries = sorted(set(boundaries))
        if boundaries[-1] != len(tokens):
            boundaries.append(len(tokens))

        clauses: List[Tuple[int, int]] = []
        for k in range(len(boundaries) - 1):
            s, e = boundaries[k], boundaries[k + 1]
            span_tokens = tokens[s:e]
            non_punct = [t for t in span_tokens if not _PUNCT_RE.fullmatch(t)]
            if len(non_punct) >= 2:
                clauses.append((s, e))

        if not clauses and tokens:
            clauses = [(0, len(tokens))]
        return clauses

    # ── POS pattern diversity ────────────────────────────────────
    def compute_pos_pattern_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Measure the diversity of POS-tag n-gram patterns across *texts*."""
        if not texts:
            return {"diversity": 0.0, "unique_patterns": 0, "details": {}}

        window = self.config.pos_pattern_window
        all_patterns_per_text: List[Set[Tuple[str, ...]]] = []
        global_counter: Counter[Tuple[str, ...]] = Counter()

        for text in texts:
            tokens = tokenize_simple(text)
            tagged = self.heuristic_pos_tagger(tokens)
            tags = [t for _, t in tagged]

            patterns_this: Set[Tuple[str, ...]] = set()
            for n in range(2, min(window + 1, len(tags) + 1)):
                for gram in extract_ngrams(tags, n):
                    patterns_this.add(gram)
                    global_counter[gram] += 1
            all_patterns_per_text.append(patterns_this)

        if not all_patterns_per_text:
            return {"diversity": 0.0, "unique_patterns": 0, "details": {}}

        unique_total = len(global_counter)
        entropy = _normalised_entropy(global_counter)

        if len(all_patterns_per_text) >= 2:
            overlaps: List[float] = []
            for i in range(len(all_patterns_per_text)):
                for j in range(i + 1, len(all_patterns_per_text)):
                    overlaps.append(
                        1.0 - _jaccard(
                            all_patterns_per_text[i],
                            all_patterns_per_text[j],
                        )
                    )
            mean_dissimilarity = float(np.mean(overlaps)) if overlaps else 0.0
        else:
            mean_dissimilarity = 0.0

        diversity = 0.5 * entropy + 0.5 * mean_dissimilarity

        top_patterns = global_counter.most_common(20)
        details = {
            "entropy": entropy,
            "mean_pairwise_dissimilarity": mean_dissimilarity,
            "top_patterns": [
                {"pattern": list(p), "count": c} for p, c in top_patterns
            ],
        }
        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "unique_patterns": unique_total,
            "details": details,
        }

    # ── sentence structure diversity ─────────────────────────────
    def compute_sentence_structure_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Diversity of sentence lengths and clause patterns across texts."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        bins = self.config.sentence_length_bins
        per_text_hists: List[np.ndarray] = []
        all_lengths: List[List[int]] = []
        all_clause_counts: List[List[int]] = []

        for text in texts:
            sentences = tokenize_to_sentences(text)
            lengths = [len(tokenize_simple(s)) for s in sentences]
            all_lengths.append(lengths)
            per_text_hists.append(_bin_values(
                [float(x) for x in lengths], bins
            ))

            clause_counts: List[int] = []
            for s in sentences:
                toks = tokenize_simple(s)
                clauses = self.extract_clause_boundaries(toks)
                clause_counts.append(len(clauses))
            all_clause_counts.append(clause_counts)

        length_vars: List[float] = []
        for lengths in all_lengths:
            if lengths:
                length_vars.append(float(np.std(lengths)))
            else:
                length_vars.append(0.0)
        intra_var = float(np.mean(length_vars)) if length_vars else 0.0

        if len(per_text_hists) >= 2:
            inter_sims: List[float] = []
            for i in range(len(per_text_hists)):
                for j in range(i + 1, len(per_text_hists)):
                    inter_sims.append(
                        1.0 - _histogram_intersection(
                            per_text_hists[i], per_text_hists[j]
                        )
                    )
            inter_diversity = float(np.mean(inter_sims))
        else:
            inter_diversity = 0.0

        clause_diversity = 0.0
        flat_clauses: List[int] = []
        for cc in all_clause_counts:
            flat_clauses.extend(cc)
        if flat_clauses:
            clause_counter = Counter(flat_clauses)
            clause_diversity = _normalised_entropy(clause_counter)

        diversity = (0.3 * min(intra_var / 10.0, 1.0)
                     + 0.4 * inter_diversity
                     + 0.3 * clause_diversity)

        details = {
            "intra_length_variance": intra_var,
            "inter_length_diversity": inter_diversity,
            "clause_count_entropy": clause_diversity,
            "mean_sentence_count": float(np.mean(
                [len(l) for l in all_lengths]
            )) if all_lengths else 0.0,
        }
        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": details,
        }

    # ── dependency pattern diversity ─────────────────────────────
    def compute_dependency_pattern_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Approximate dependency-pattern diversity using POS bigrams
        that act as proxies for head–dependent relations."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        per_text: List[Counter] = []
        for text in texts:
            tokens = tokenize_simple(text)
            tagged = self.heuristic_pos_tagger(tokens)
            dep_counter: Counter[Tuple[str, str, str]] = Counter()
            for i in range(len(tagged) - 1):
                _, tag_i = tagged[i]
                word_j, tag_j = tagged[i + 1]
                relation = self._infer_dep_relation(tag_i, tag_j)
                dep_counter[(tag_i, relation, tag_j)] += 1
            per_text.append(dep_counter)

        if len(per_text) < 2:
            global_c: Counter = Counter()
            for c in per_text:
                global_c.update(c)
            return {
                "diversity": _normalised_entropy(global_c),
                "details": {"unique_dep_patterns": len(global_c)},
            }

        sims: List[float] = []
        for i in range(len(per_text)):
            for j in range(i + 1, len(per_text)):
                sims.append(
                    1.0 - _cosine_similarity_counters(per_text[i], per_text[j])
                )

        global_c = Counter()
        for c in per_text:
            global_c.update(c)

        diversity = float(np.mean(sims))
        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "unique_dep_patterns": len(global_c),
                "mean_pairwise_dissimilarity": diversity,
                "entropy": _normalised_entropy(global_c),
            },
        }

    @staticmethod
    def _infer_dep_relation(head_tag: str, dep_tag: str) -> str:
        """Heuristic dependency relation label from POS tags."""
        if head_tag in ("VB", "VBG", "VBD", "VBN", "MD"):
            if dep_tag in ("NN", "NNP", "PRP"):
                return "nsubj" if dep_tag != "PRP" else "nsubj"
            if dep_tag == "RB":
                return "advmod"
            if dep_tag == "IN":
                return "prep"
            if dep_tag in ("DT",):
                return "det"
            return "dep"
        if head_tag in ("NN", "NNP"):
            if dep_tag == "DT":
                return "det"
            if dep_tag == "JJ":
                return "amod"
            if dep_tag == "IN":
                return "prep"
            if dep_tag in ("NN", "NNP"):
                return "compound"
            return "dep"
        if head_tag == "IN":
            if dep_tag in ("NN", "NNP", "PRP"):
                return "pobj"
            return "dep"
        return "dep"

    # ── phrase structure diversity ───────────────────────────────
    def compute_phrase_structure_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Diversity of phrase-level POS patterns (NP, VP, PP chunks)."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        per_text_phrases: List[Counter] = []
        for text in texts:
            tokens = tokenize_simple(text)
            tagged = self.heuristic_pos_tagger(tokens)
            phrases = self._extract_phrases(tagged)
            phrase_counter: Counter[str] = Counter()
            for ptype, _ in phrases:
                phrase_counter[ptype] += 1
            per_text_phrases.append(phrase_counter)

        global_phrase = Counter()
        for c in per_text_phrases:
            global_phrase.update(c)
        entropy = _normalised_entropy(global_phrase)

        per_text_phrase_sets: List[Set[Tuple[str, Tuple[str, ...]]]] = []
        for text in texts:
            tokens = tokenize_simple(text)
            tagged = self.heuristic_pos_tagger(tokens)
            phrases = self._extract_phrases(tagged)
            pset: Set[Tuple[str, Tuple[str, ...]]] = set()
            for ptype, ptags in phrases:
                pset.add((ptype, tuple(ptags)))
            per_text_phrase_sets.append(pset)

        if len(per_text_phrase_sets) >= 2:
            dissims: List[float] = []
            for i in range(len(per_text_phrase_sets)):
                for j in range(i + 1, len(per_text_phrase_sets)):
                    dissims.append(
                        1.0 - _jaccard(
                            per_text_phrase_sets[i], per_text_phrase_sets[j]
                        )
                    )
            pair_div = float(np.mean(dissims))
        else:
            pair_div = 0.0

        diversity = 0.5 * entropy + 0.5 * pair_div
        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "phrase_type_entropy": entropy,
                "pairwise_dissimilarity": pair_div,
                "unique_phrase_types": len(global_phrase),
            },
        }

    def _extract_phrases(
        self, tagged: List[Tuple[str, str]]
    ) -> List[Tuple[str, List[str]]]:
        """Chunking: extract NP, VP, PP spans from tagged tokens."""
        phrases: List[Tuple[str, List[str]]] = []
        i = 0
        while i < len(tagged):
            _, tag = tagged[i]

            # NP: (DT)? (JJ)* (NN|NNP|PRP|CD)+
            if tag in ("DT", "JJ", "NN", "NNP", "PRP", "CD"):
                start = i
                chunk_tags: List[str] = []
                while i < len(tagged):
                    _, t = tagged[i]
                    if t in ("DT", "JJ", "NN", "NNP", "PRP", "CD"):
                        chunk_tags.append(t)
                        i += 1
                    else:
                        break
                has_noun = any(
                    t in ("NN", "NNP", "PRP") for t in chunk_tags
                )
                if has_noun:
                    phrases.append(("NP", chunk_tags))
                else:
                    i = start + 1
                continue

            # VP: (MD|RB)* (VB|VBG|VBD|VBN)+
            if tag in ("MD", "VB", "VBG", "VBD", "VBN"):
                chunk_tags = []
                while i < len(tagged):
                    _, t = tagged[i]
                    if t in ("MD", "VB", "VBG", "VBD", "VBN", "RB"):
                        chunk_tags.append(t)
                        i += 1
                    else:
                        break
                has_verb = any(
                    t in ("VB", "VBG", "VBD", "VBN") for t in chunk_tags
                )
                if has_verb:
                    phrases.append(("VP", chunk_tags))
                continue

            # PP: IN (DT)? (JJ)* (NN|NNP)+
            if tag == "IN":
                chunk_tags = ["IN"]
                j = i + 1
                while j < len(tagged):
                    _, t = tagged[j]
                    if t in ("DT", "JJ", "NN", "NNP", "PRP", "CD"):
                        chunk_tags.append(t)
                        j += 1
                    else:
                        break
                if len(chunk_tags) > 1:
                    phrases.append(("PP", chunk_tags))
                    i = j
                else:
                    i += 1
                continue

            i += 1

        return phrases

    # ── syntactic complexity ─────────────────────────────────────
    def compute_syntactic_complexity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Compute Yngve-depth approximation and branching-factor metrics.

        Yngve depth is estimated from clause nesting; branching factor
        from the phrase chunking.
        """
        if not texts:
            return {"diversity": 0.0, "details": {}}

        complexities: List[Dict[str, float]] = []
        for text in texts:
            sentences = tokenize_to_sentences(text)
            yngve_depths: List[float] = []
            branching_factors: List[float] = []
            clause_depths: List[int] = []

            for sent in sentences:
                tokens = tokenize_simple(sent)
                tagged = self.heuristic_pos_tagger(tokens)

                depth = self._estimate_yngve_depth(tokens)
                yngve_depths.append(depth)

                phrases = self._extract_phrases(tagged)
                if phrases:
                    bf = len(tokens) / len(phrases)
                    branching_factors.append(bf)
                else:
                    branching_factors.append(1.0)

                clauses = self.extract_clause_boundaries(tokens)
                clause_depths.append(len(clauses))

            avg_yngve = float(np.mean(yngve_depths)) if yngve_depths else 0.0
            avg_bf = float(np.mean(branching_factors)) if branching_factors else 1.0
            avg_cd = float(np.mean(clause_depths)) if clause_depths else 0.0

            complexities.append({
                "yngve_depth": avg_yngve,
                "branching_factor": avg_bf,
                "clause_depth": avg_cd,
            })

        yngve_vals = [c["yngve_depth"] for c in complexities]
        bf_vals = [c["branching_factor"] for c in complexities]
        cd_vals = [c["clause_depth"] for c in complexities]

        yngve_div = float(np.std(yngve_vals)) / (float(np.mean(yngve_vals)) + 1e-12)
        bf_div = float(np.std(bf_vals)) / (float(np.mean(bf_vals)) + 1e-12)
        cd_div = float(np.std(cd_vals)) / (float(np.mean(cd_vals)) + 1e-12)

        diversity = (yngve_div + bf_div + cd_div) / 3.0

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "mean_yngve_depth": float(np.mean(yngve_vals)),
                "std_yngve_depth": float(np.std(yngve_vals)),
                "mean_branching_factor": float(np.mean(bf_vals)),
                "std_branching_factor": float(np.std(bf_vals)),
                "mean_clause_depth": float(np.mean(cd_vals)),
                "std_clause_depth": float(np.std(cd_vals)),
                "yngve_diversity": yngve_div,
                "branching_diversity": bf_div,
                "clause_diversity": cd_div,
            },
        }

    def _estimate_yngve_depth(self, tokens: List[str]) -> float:
        """Estimate Yngve depth from embedding depth markers."""
        if not tokens:
            return 0.0
        lower = [t.lower() for t in tokens]
        depth = 0
        max_depth = 0
        total_depth = 0
        count = 0

        openers = {"(", "[", "{"}
        closers = {")", "]", "}"}

        for tok in lower:
            if tok in openers:
                depth += 1
                max_depth = max(max_depth, depth)
            elif tok in closers:
                depth = max(0, depth - 1)
            elif tok in _CLAUSE_MARKERS:
                depth += 1
                max_depth = max(max_depth, depth)
            elif tok in (",", ";"):
                total_depth += depth
                count += 1

        for tok in lower:
            if tok in _CLAUSE_MARKERS:
                total_depth += 1
                count += 1

        if count == 0:
            return float(max_depth)
        return total_depth / count

    # ── construction diversity ───────────────────────────────────
    def compute_construction_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Identify and count unique syntactic constructions.

        Constructions tracked:
        - passive voice patterns
        - relative clauses
        - conditional clauses
        - question forms
        - imperative forms
        - existential there-constructions
        - cleft sentences
        - topicalized constructions
        """
        if not texts:
            return {"diversity": 0.0, "details": {}}

        per_text: List[Counter] = []
        for text in texts:
            constructions: Counter[str] = Counter()
            sentences = tokenize_to_sentences(text)
            for sent in sentences:
                tokens = tokenize_simple(sent)
                tagged = self.heuristic_pos_tagger(tokens)
                lower = [t.lower() for t in tokens]

                # Passive voice: form of "be" + VBN/VBD
                for k in range(len(tagged) - 1):
                    if (tagged[k][0].lower() in
                            ("is", "are", "was", "were", "been", "being",
                             "be") and
                            tagged[k + 1][1] in ("VBN", "VBD")):
                        constructions["passive"] += 1
                        break

                # Relative clause: who/which/that + VP
                for k in range(len(lower)):
                    if lower[k] in ("who", "which", "that", "whom"):
                        if k + 1 < len(tagged) and tagged[k + 1][1] in (
                            "VB", "VBG", "VBD", "VBN", "MD"
                        ):
                            constructions["relative_clause"] += 1
                            break

                # Conditional
                if lower and lower[0] == "if":
                    constructions["conditional"] += 1
                elif "if" in lower:
                    idx = lower.index("if")
                    if idx > 0 and lower[idx - 1] in (",", ";", "—"):
                        constructions["conditional"] += 1

                # Question
                if tokens and tokens[-1] == "?":
                    constructions["question"] += 1
                elif lower and lower[0] in (
                    "is", "are", "was", "were", "do", "does", "did",
                    "can", "could", "will", "would", "shall", "should",
                    "may", "might", "have", "has", "had",
                    "what", "where", "when", "who", "whom", "whose",
                    "which", "how", "why",
                ):
                    constructions["question"] += 1

                # Imperative: first token is base-form verb
                if tagged and tagged[0][1] == "VB" and lower[0] not in _AUXILIARY_VERBS:
                    constructions["imperative"] += 1

                # Existential there
                if lower and lower[0] == "there" and len(tagged) > 1:
                    if tagged[1][0].lower() in ("is", "are", "was", "were",
                                                 "exist", "exists"):
                        constructions["existential"] += 1

                # Cleft: "it is/was ... that/who"
                if (len(lower) > 3 and lower[0] == "it"
                        and lower[1] in ("is", "was")):
                    if "that" in lower[2:] or "who" in lower[2:]:
                        constructions["cleft"] += 1

                # Topicalized (fronted object): NN/NNP at start
                # followed by comma then subject
                if (len(tagged) > 3 and tagged[0][1] in ("NN", "NNP")
                        and tagged[1][0] == ","):
                    constructions["topicalized"] += 1

                # Coordination
                if "and" in lower or "but" in lower or "or" in lower:
                    constructions["coordination"] += 1

                # Negation
                if "not" in lower or "n't" in lower or "never" in lower:
                    constructions["negation"] += 1

                # Comparative / superlative
                for w in lower:
                    if w.endswith("er") and len(w) > 3:
                        constructions["comparative"] += 1
                        break
                    if w.endswith("est") and len(w) > 4:
                        constructions["superlative"] += 1
                        break
                    if w in ("more", "most", "less", "least"):
                        constructions["comparative"] += 1
                        break

            per_text.append(constructions)

        all_types: Set[str] = set()
        for c in per_text:
            all_types.update(c.keys())

        if len(per_text) >= 2:
            vectors: List[np.ndarray] = []
            sorted_types = sorted(all_types)
            for c in per_text:
                vectors.append(np.array(
                    [c.get(t, 0) for t in sorted_types], dtype=np.float64
                ))
            vmat = np.stack(vectors)
            norms = np.linalg.norm(vmat, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            vmat_normed = vmat / norms
            dists = pdist(vmat_normed, metric="cosine")
            dists = np.nan_to_num(dists, nan=0.0)
            diversity = float(np.mean(dists))
        else:
            global_c: Counter = Counter()
            for c in per_text:
                global_c.update(c)
            diversity = _normalised_entropy(global_c)

        global_counter: Counter = Counter()
        for c in per_text:
            global_counter.update(c)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "unique_constructions": len(all_types),
                "construction_counts": dict(global_counter.most_common()),
                "entropy": _normalised_entropy(global_counter),
            },
        }


# ──────────────────────────────────────────────────────────────────────
# 5. ParseTreeSimilarity
# ──────────────────────────────────────────────────────────────────────

class ParseTreeSimilarity:
    """Lightweight constituency parsing and tree comparison metrics."""

    def __init__(self, config: Optional[StructuralConfig] = None) -> None:
        self.config = config or StructuralConfig()
        self._analyzer = SyntacticDiversityAnalyzer(self.config)

    # ── build constituency tree ──────────────────────────────────
    def build_constituency_tree(self, text: str) -> ParseNode:
        """Build a lightweight constituency tree using POS tags and
        phrase chunking — no external parser required.

        Structure:  S → (NP|VP|PP|ADVP|SBAR)+ for each sentence,
        then TOP → S+ for multi-sentence texts.
        """
        sentences = tokenize_to_sentences(text)
        if not sentences:
            return ParseNode(label="TOP", children=[
                ParseNode(label="S", children=[
                    ParseNode(label="EMPTY", word="")
                ])
            ])

        sentence_nodes: List[ParseNode] = []
        for sent in sentences:
            tokens = tokenize_simple(sent)
            if not tokens:
                continue
            tagged = self._analyzer.heuristic_pos_tagger(tokens)
            phrase_nodes = self._build_phrase_nodes(tagged)
            s_node = ParseNode(label="S", children=phrase_nodes)
            self._assign_depths(s_node, 0)
            sentence_nodes.append(s_node)

        if not sentence_nodes:
            sentence_nodes = [ParseNode(label="S", children=[
                ParseNode(label="EMPTY", word="")
            ])]

        top = ParseNode(label="TOP", children=sentence_nodes)
        self._assign_depths(top, 0)
        return top

    def _build_phrase_nodes(
        self, tagged: List[Tuple[str, str]]
    ) -> List[ParseNode]:
        """Convert tagged tokens into a flat list of phrase ParseNodes."""
        nodes: List[ParseNode] = []
        i = 0
        while i < len(tagged):
            word, tag = tagged[i]

            # NP chunk
            if tag in ("DT", "JJ", "NN", "NNP", "PRP", "CD"):
                chunk: List[Tuple[str, str]] = []
                while i < len(tagged) and tagged[i][1] in (
                    "DT", "JJ", "NN", "NNP", "PRP", "CD"
                ):
                    chunk.append(tagged[i])
                    i += 1
                has_noun = any(t in ("NN", "NNP", "PRP") for _, t in chunk)
                if has_noun:
                    children = [
                        ParseNode(label=t, children=[
                            ParseNode(label=w, word=w)
                        ]) for w, t in chunk
                    ]
                    nodes.append(ParseNode(label="NP", children=children))
                else:
                    for w, t in chunk:
                        nodes.append(ParseNode(label=t, children=[
                            ParseNode(label=w, word=w)
                        ]))
                continue

            # VP chunk
            if tag in ("MD", "VB", "VBG", "VBD", "VBN"):
                chunk = []
                while i < len(tagged) and tagged[i][1] in (
                    "MD", "VB", "VBG", "VBD", "VBN", "RB"
                ):
                    chunk.append(tagged[i])
                    i += 1
                children = [
                    ParseNode(label=t, children=[
                        ParseNode(label=w, word=w)
                    ]) for w, t in chunk
                ]
                nodes.append(ParseNode(label="VP", children=children))
                continue

            # PP chunk
            if tag == "IN":
                chunk = [tagged[i]]
                i += 1
                while i < len(tagged) and tagged[i][1] in (
                    "DT", "JJ", "NN", "NNP", "PRP", "CD"
                ):
                    chunk.append(tagged[i])
                    i += 1
                if len(chunk) > 1:
                    in_node = ParseNode(label="IN", children=[
                        ParseNode(label=chunk[0][0], word=chunk[0][0])
                    ])
                    np_children = [
                        ParseNode(label=t, children=[
                            ParseNode(label=w, word=w)
                        ]) for w, t in chunk[1:]
                    ]
                    np_node = ParseNode(label="NP", children=np_children)
                    nodes.append(ParseNode(
                        label="PP", children=[in_node, np_node]
                    ))
                else:
                    nodes.append(ParseNode(label="IN", children=[
                        ParseNode(label=word, word=word)
                    ]))
                continue

            # SBAR (subordinate clause marker)
            if word.lower() in _CLAUSE_MARKERS and tag != "IN":
                sbar_children: List[ParseNode] = [
                    ParseNode(label=tag, children=[
                        ParseNode(label=word, word=word)
                    ])
                ]
                i += 1
                # consume the rest as an embedded S
                sub_chunk: List[Tuple[str, str]] = []
                while i < len(tagged) and tagged[i][0] not in (",", ";", "."):
                    sub_chunk.append(tagged[i])
                    i += 1
                if sub_chunk:
                    sub_nodes = self._build_phrase_nodes(sub_chunk)
                    sbar_children.append(
                        ParseNode(label="S", children=sub_nodes)
                    )
                nodes.append(ParseNode(label="SBAR", children=sbar_children))
                continue

            # RB → ADVP
            if tag == "RB":
                rb_children = [ParseNode(label=tag, children=[
                    ParseNode(label=word, word=word)
                ])]
                i += 1
                while i < len(tagged) and tagged[i][1] == "RB":
                    rb_children.append(ParseNode(
                        label=tagged[i][1],
                        children=[ParseNode(
                            label=tagged[i][0], word=tagged[i][0]
                        )]
                    ))
                    i += 1
                nodes.append(ParseNode(label="ADVP", children=rb_children))
                continue

            # CC
            if tag == "CC":
                nodes.append(ParseNode(label="CC", children=[
                    ParseNode(label=word, word=word)
                ]))
                i += 1
                continue

            # punctuation / fallback
            nodes.append(ParseNode(label=tag, children=[
                ParseNode(label=word, word=word)
            ]))
            i += 1

        return nodes if nodes else [ParseNode(label="EMPTY", word="")]

    @staticmethod
    def _assign_depths(node: ParseNode, depth: int) -> None:
        node.depth = depth
        for c in node.children:
            ParseTreeSimilarity._assign_depths(c, depth + 1)

    # ── Zhang-Shasha tree edit distance ──────────────────────────
    def tree_edit_distance(
        self, tree_a: ParseNode, tree_b: ParseNode
    ) -> int:
        """Zhang-Shasha tree edit distance between two ParseNode trees.

        Operations: insert, delete, relabel — each costs 1.
        """
        nodes_a = self._post_order(tree_a)
        nodes_b = self._post_order(tree_b)
        if not nodes_a or not nodes_b:
            return max(len(nodes_a), len(nodes_b))

        lr_a = self._leftmost_leaf_descendants(tree_a, nodes_a)
        lr_b = self._leftmost_leaf_descendants(tree_b, nodes_b)

        kr_a = self._key_roots(lr_a)
        kr_b = self._key_roots(lr_b)

        size_a = len(nodes_a)
        size_b = len(nodes_b)

        td = np.full((size_a + 1, size_b + 1), 0, dtype=np.int64)
        fd = np.full((size_a + 1, size_b + 1), 0, dtype=np.int64)

        for x in kr_a:
            for y in kr_b:
                lx = lr_a[x]
                ly = lr_b[y]

                fd[lx][ly] = 0
                for i in range(lx, x + 1):
                    fd[i + 1][ly] = fd[i][ly] + 1
                for j in range(ly, y + 1):
                    fd[lx][j + 1] = fd[lx][j] + 1

                for i in range(lx, x + 1):
                    for j in range(ly, y + 1):
                        li = lr_a[i]
                        lj = lr_b[j]
                        if li == lx and lj == ly:
                            cost = (0 if nodes_a[i].label == nodes_b[j].label
                                    else 1)
                            fd[i + 1][j + 1] = min(
                                fd[i][j + 1] + 1,
                                fd[i + 1][j] + 1,
                                fd[i][j] + cost,
                            )
                            td[i + 1][j + 1] = fd[i + 1][j + 1]
                        else:
                            fd[i + 1][j + 1] = min(
                                fd[i][j + 1] + 1,
                                fd[i + 1][j] + 1,
                                fd[li][lj] + td[i + 1][j + 1],
                            )

        return int(td[size_a][size_b])

    @staticmethod
    def _post_order(node: ParseNode) -> List[ParseNode]:
        result: List[ParseNode] = []
        stack: List[Tuple[ParseNode, bool]] = [(node, False)]
        while stack:
            n, visited = stack.pop()
            if visited or n.is_leaf():
                result.append(n)
            else:
                stack.append((n, True))
                for c in reversed(n.children):
                    stack.append((c, False))
        return result

    @staticmethod
    def _leftmost_leaf_descendants(
        root: ParseNode, post_order_nodes: List[ParseNode]
    ) -> List[int]:
        """For each node in post-order, return the index of its
        leftmost leaf descendant (also in post-order)."""
        node_to_idx: Dict[int, int] = {}
        for idx, n in enumerate(post_order_nodes):
            node_to_idx[id(n)] = idx

        result = [0] * len(post_order_nodes)
        for idx, n in enumerate(post_order_nodes):
            if n.is_leaf():
                result[idx] = idx
            else:
                leftmost = n
                while not leftmost.is_leaf():
                    leftmost = leftmost.children[0]
                result[idx] = node_to_idx.get(id(leftmost), idx)
        return result

    @staticmethod
    def _key_roots(lr: List[int]) -> List[int]:
        """Key roots for Zhang-Shasha: nodes whose leftmost-leaf
        index is not shared by any node to the right."""
        visited: Set[int] = set()
        kr: List[int] = []
        for i in range(len(lr) - 1, -1, -1):
            if lr[i] not in visited:
                kr.append(i)
                visited.add(lr[i])
        kr.sort()
        return kr

    # ── tree kernel similarity ───────────────────────────────────
    def tree_kernel_similarity(
        self,
        tree_a: ParseNode,
        tree_b: ParseNode,
        kernel_type: str = "subtree",
    ) -> float:
        """Compute tree kernel similarity.

        kernel_type:
            'subtree'  – counts common subtrees (SST kernel)
            'subset'   – counts common subset trees (more general)
        """
        lam = self.config.tree_kernel_lambda

        if kernel_type == "subtree":
            k_ab = self._subtree_kernel(tree_a, tree_b, lam)
            k_aa = self._subtree_kernel(tree_a, tree_a, lam)
            k_bb = self._subtree_kernel(tree_b, tree_b, lam)
        else:
            k_ab = self._subset_tree_kernel(tree_a, tree_b, lam)
            k_aa = self._subset_tree_kernel(tree_a, tree_a, lam)
            k_bb = self._subset_tree_kernel(tree_b, tree_b, lam)

        denom = math.sqrt(k_aa * k_bb)
        if denom < 1e-12:
            return 0.0
        return k_ab / denom

    def _subtree_kernel(
        self, t1: ParseNode, t2: ParseNode, lam: float
    ) -> float:
        """Collins-Duffy subtree (SST) kernel."""
        nodes1 = t1.subtrees()
        nodes2 = t2.subtrees()
        total = 0.0
        cache: Dict[Tuple[int, int], float] = {}

        for n1 in nodes1:
            for n2 in nodes2:
                total += self._sst_delta(n1, n2, lam, cache)
        return total

    def _sst_delta(
        self,
        n1: ParseNode,
        n2: ParseNode,
        lam: float,
        cache: Dict[Tuple[int, int], float],
    ) -> float:
        key = (id(n1), id(n2))
        if key in cache:
            return cache[key]

        if n1.label != n2.label:
            cache[key] = 0.0
            return 0.0

        if n1.is_leaf() and n2.is_leaf():
            val = lam
            cache[key] = val
            return val

        if len(n1.children) != len(n2.children):
            cache[key] = 0.0
            return 0.0

        prod = lam
        for c1, c2 in zip(n1.children, n2.children):
            prod *= (1.0 + self._sst_delta(c1, c2, lam, cache))

        cache[key] = prod
        return prod

    def _subset_tree_kernel(
        self, t1: ParseNode, t2: ParseNode, lam: float
    ) -> float:
        """Subset tree kernel (more permissive than SST)."""
        nodes1 = t1.subtrees()
        nodes2 = t2.subtrees()
        total = 0.0
        cache: Dict[Tuple[int, int], float] = {}

        for n1 in nodes1:
            for n2 in nodes2:
                total += self._subset_delta(n1, n2, lam, cache)
        return total

    def _subset_delta(
        self,
        n1: ParseNode,
        n2: ParseNode,
        lam: float,
        cache: Dict[Tuple[int, int], float],
    ) -> float:
        key = (id(n1), id(n2))
        if key in cache:
            return cache[key]

        if n1.label != n2.label:
            cache[key] = 0.0
            return 0.0

        if n1.is_leaf() and n2.is_leaf():
            val = lam
            cache[key] = val
            return val

        if n1.is_leaf() or n2.is_leaf():
            cache[key] = 0.0
            return 0.0

        # For subset trees, we allow partial matching of children
        prod = lam
        matched = 0
        for c1 in n1.children:
            best = 0.0
            for c2 in n2.children:
                d = self._subset_delta(c1, c2, lam, cache)
                best = max(best, d)
            if best > 0:
                prod *= (1.0 + best)
                matched += 1

        if matched == 0:
            cache[key] = 0.0
            return 0.0

        cache[key] = prod
        return prod

    # ── depth / branching similarity ─────────────────────────────
    def compute_tree_depth_similarity(
        self, tree_a: ParseNode, tree_b: ParseNode
    ) -> float:
        """Similarity based on depth distribution of nodes."""
        depths_a = self._collect_depths(tree_a)
        depths_b = self._collect_depths(tree_b)
        if not depths_a or not depths_b:
            return 0.0

        max_d = max(max(depths_a), max(depths_b)) + 1
        hist_a = np.zeros(max_d, dtype=np.float64)
        hist_b = np.zeros(max_d, dtype=np.float64)
        for d in depths_a:
            hist_a[d] += 1
        for d in depths_b:
            hist_b[d] += 1

        return float(np.clip(_histogram_intersection(hist_a, hist_b), 0.0, 1.0))

    @staticmethod
    def _collect_depths(node: ParseNode) -> List[int]:
        depths: List[int] = []
        stack: List[Tuple[ParseNode, int]] = [(node, 0)]
        while stack:
            n, d = stack.pop()
            depths.append(d)
            for c in n.children:
                stack.append((c, d + 1))
        return depths

    def compute_branching_similarity(
        self, tree_a: ParseNode, tree_b: ParseNode
    ) -> float:
        """Similarity based on branching factor distribution."""
        bf_a = self._collect_branching(tree_a)
        bf_b = self._collect_branching(tree_b)
        if not bf_a or not bf_b:
            return 0.0

        max_bf = max(max(bf_a), max(bf_b)) + 1
        hist_a = np.zeros(max_bf, dtype=np.float64)
        hist_b = np.zeros(max_bf, dtype=np.float64)
        for b in bf_a:
            hist_a[b] += 1
        for b in bf_b:
            hist_b[b] += 1

        return float(np.clip(_histogram_intersection(hist_a, hist_b), 0.0, 1.0))

    @staticmethod
    def _collect_branching(node: ParseNode) -> List[int]:
        result: List[int] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if not n.is_leaf():
                result.append(len(n.children))
                for c in n.children:
                    stack.append(c)
        return result

    # ── pairwise distances ───────────────────────────────────────
    def pairwise_tree_distances(
        self, texts: List[str], metric: str = "edit"
    ) -> np.ndarray:
        """Compute a pairwise distance matrix over parse trees of *texts*.

        metric: 'edit' (tree edit distance) or 'kernel' (1 - kernel sim).
        """
        n = len(texts)
        trees = [self.build_constituency_tree(t) for t in texts]
        dist = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                if metric == "edit":
                    d = float(self.tree_edit_distance(trees[i], trees[j]))
                elif metric == "kernel":
                    sim = self.tree_kernel_similarity(trees[i], trees[j])
                    d = 1.0 - sim
                else:
                    d_depth = 1.0 - self.compute_tree_depth_similarity(
                        trees[i], trees[j]
                    )
                    d_branch = 1.0 - self.compute_branching_similarity(
                        trees[i], trees[j]
                    )
                    d = 0.5 * d_depth + 0.5 * d_branch
                dist[i, j] = d
                dist[j, i] = d

        return dist

    def average_tree_diversity(
        self, texts: List[str], metric: str = "edit"
    ) -> Dict[str, Any]:
        """Aggregate tree-based diversity over a set of texts."""
        if len(texts) < 2:
            return {"diversity": 0.0, "details": {}}

        dist_mat = self.pairwise_tree_distances(texts, metric=metric)
        upper = dist_mat[np.triu_indices_from(dist_mat, k=1)]

        if metric == "edit" and upper.max() > 0:
            upper = upper / upper.max()

        mean_d = float(np.mean(upper))
        std_d = float(np.std(upper))
        min_d = float(np.min(upper))
        max_d = float(np.max(upper))

        return {
            "diversity": float(np.clip(mean_d, 0.0, 1.0)),
            "details": {
                "mean_distance": mean_d,
                "std_distance": std_d,
                "min_distance": min_d,
                "max_distance": max_d,
                "num_texts": len(texts),
                "metric": metric,
            },
        }


# ──────────────────────────────────────────────────────────────────────
# 6. CodeStructureDiversity
# ──────────────────────────────────────────────────────────────────────

class CodeStructureDiversity:
    """Structural diversity analysis specialised for code generation."""

    def __init__(self, config: Optional[StructuralConfig] = None) -> None:
        self.config = config or StructuralConfig()

    # ── extract code structure ───────────────────────────────────
    def extract_code_structure(
        self, code: str
    ) -> Dict[str, Any]:
        """Extract a structural fingerprint from *code* based on
        indentation levels, line types, and block nesting."""
        lines = code.split("\n")
        indent_size = self.config.code_indent_size
        structure: Dict[str, Any] = {
            "total_lines": len(lines),
            "blank_lines": 0,
            "comment_lines": 0,
            "code_lines": 0,
            "max_indent": 0,
            "indent_histogram": Counter(),
            "block_depths": [],
            "line_types": [],
        }

        for line in lines:
            stripped = line.strip()
            if not stripped:
                structure["blank_lines"] += 1
                structure["line_types"].append("blank")
                continue

            is_comment = False
            if stripped.startswith("#"):
                is_comment = True
            elif stripped.startswith("//"):
                is_comment = True
            elif stripped.startswith("/*") or stripped.startswith("*"):
                is_comment = True
            elif stripped.startswith("'''") or stripped.startswith('"""'):
                is_comment = True

            if is_comment:
                structure["comment_lines"] += 1
                structure["line_types"].append("comment")
            else:
                structure["code_lines"] += 1
                structure["line_types"].append("code")

            leading = len(line) - len(line.lstrip())
            indent_level = leading // max(indent_size, 1)
            structure["indent_histogram"][indent_level] += 1
            structure["max_indent"] = max(
                structure["max_indent"], indent_level
            )
            structure["block_depths"].append(indent_level)

        return structure

    # ── control flow diversity ───────────────────────────────────
    def compute_control_flow_diversity(
        self, codes: List[str]
    ) -> Dict[str, Any]:
        """Measure diversity of control-flow constructs across code samples."""
        if not codes:
            return {"diversity": 0.0, "details": {}}

        cf_re = {
            "if": re.compile(r"\b(if)\b"),
            "else": re.compile(r"\b(else)\b"),
            "elif": re.compile(r"\b(elif|else\s+if)\b"),
            "for": re.compile(r"\b(for)\b"),
            "while": re.compile(r"\b(while)\b"),
            "do_while": re.compile(r"\b(do)\b"),
            "switch": re.compile(r"\b(switch|match)\b"),
            "try": re.compile(r"\b(try)\b"),
            "catch": re.compile(r"\b(catch|except)\b"),
            "finally": re.compile(r"\b(finally)\b"),
            "return": re.compile(r"\b(return)\b"),
            "break": re.compile(r"\b(break)\b"),
            "continue": re.compile(r"\b(continue)\b"),
            "yield": re.compile(r"\b(yield)\b"),
            "raise": re.compile(r"\b(raise|throw)\b"),
            "with": re.compile(r"\b(with)\b"),
            "assert": re.compile(r"\b(assert)\b"),
        }

        per_code: List[Counter] = []
        for code in codes:
            counts: Counter[str] = Counter()
            for name, pat in cf_re.items():
                matches = pat.findall(code)
                if matches:
                    counts[name] += len(matches)
            per_code.append(counts)

        # Build pattern sequences (order of control-flow statements)
        cf_sequences: List[List[str]] = []
        for code in codes:
            seq: List[str] = []
            for line in code.split("\n"):
                stripped = line.strip()
                for name, pat in cf_re.items():
                    if pat.match(stripped):
                        seq.append(name)
                        break
            cf_sequences.append(seq)

        all_types: Set[str] = set()
        for c in per_code:
            all_types.update(c.keys())
        sorted_types = sorted(all_types) if all_types else ["if"]

        if len(per_code) >= 2:
            vectors = []
            for c in per_code:
                vectors.append(
                    np.array([c.get(t, 0) for t in sorted_types],
                             dtype=np.float64)
                )
            mat = np.stack(vectors)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            mat_n = mat / norms
            dists = pdist(mat_n, metric="cosine")
            dists = np.nan_to_num(dists, nan=0.0)
            type_diversity = float(np.mean(dists))
        else:
            gc: Counter = Counter()
            for c in per_code:
                gc.update(c)
            type_diversity = _normalised_entropy(gc)

        # Sequence diversity via edit distance
        seq_diversity = 0.0
        if len(cf_sequences) >= 2:
            seq_dists: List[float] = []
            for i in range(len(cf_sequences)):
                for j in range(i + 1, len(cf_sequences)):
                    max_len = max(
                        len(cf_sequences[i]), len(cf_sequences[j]), 1
                    )
                    d = levenshtein_distance(
                        cf_sequences[i], cf_sequences[j]
                    )
                    seq_dists.append(d / max_len)
            seq_diversity = float(np.mean(seq_dists))

        diversity = 0.5 * type_diversity + 0.5 * seq_diversity
        global_counts: Counter = Counter()
        for c in per_code:
            global_counts.update(c)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "type_diversity": type_diversity,
                "sequence_diversity": seq_diversity,
                "global_counts": dict(global_counts.most_common()),
                "unique_types": len(all_types),
            },
        }

    # ── variable naming diversity ────────────────────────────────
    def compute_variable_naming_diversity(
        self, codes: List[str]
    ) -> Dict[str, Any]:
        """Measure diversity of variable naming conventions."""
        if not codes:
            return {"diversity": 0.0, "details": {}}

        var_patterns = [
            re.compile(r"\b(?:let|var|const|val)\s+(\w+)"),
            re.compile(r"\b(\w+)\s*=\s*"),
            re.compile(r"(?:int|float|double|string|bool|char|long|short|auto|var)\s+(\w+)"),
            re.compile(r"def\s+\w+\s*\(([^)]*)\)"),
            re.compile(r"for\s+(\w+)\s+in\b"),
            re.compile(r"for\s*\(\s*(?:let|var|const|int|auto)\s+(\w+)"),
        ]

        per_code_vars: List[Set[str]] = []
        per_code_styles: List[Counter] = []

        for code in codes:
            variables: Set[str] = set()
            for pat in var_patterns:
                for m in pat.finditer(code):
                    candidates = m.group(1).split(",")
                    for c in candidates:
                        c = c.strip().split(":")[0].split("=")[0].strip()
                        c = re.sub(r"[^a-zA-Z0-9_]", "", c)
                        if c and c not in _CODE_KEYWORDS and len(c) > 1:
                            variables.add(c)
            per_code_vars.append(variables)

            style_counter: Counter[str] = Counter()
            for v in variables:
                style = self._classify_naming_style(v)
                style_counter[style] += 1
            per_code_styles.append(style_counter)

        if len(per_code_vars) >= 2:
            name_overlaps: List[float] = []
            for i in range(len(per_code_vars)):
                for j in range(i + 1, len(per_code_vars)):
                    name_overlaps.append(
                        1.0 - _jaccard(per_code_vars[i], per_code_vars[j])
                    )
            name_div = float(np.mean(name_overlaps))
        else:
            all_vars: Set[str] = set()
            for vs in per_code_vars:
                all_vars.update(vs)
            name_div = min(len(all_vars) / 20.0, 1.0)

        global_styles: Counter = Counter()
        for sc in per_code_styles:
            global_styles.update(sc)
        style_ent = _normalised_entropy(global_styles)

        avg_len: List[float] = []
        for vs in per_code_vars:
            if vs:
                avg_len.append(float(np.mean([len(v) for v in vs])))
        len_div = float(np.std(avg_len)) / (float(np.mean(avg_len)) + 1e-12) if avg_len else 0.0

        diversity = 0.4 * name_div + 0.3 * style_ent + 0.3 * min(len_div, 1.0)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "name_diversity": name_div,
                "style_entropy": style_ent,
                "length_diversity": len_div,
                "styles": dict(global_styles),
                "total_unique_vars": sum(len(vs) for vs in per_code_vars),
            },
        }

    @staticmethod
    def _classify_naming_style(name: str) -> str:
        """Classify a variable name into a naming convention."""
        if "_" in name:
            if name == name.upper():
                return "UPPER_SNAKE"
            if name == name.lower():
                return "snake_case"
            return "mixed_snake"
        if name[0].isupper():
            return "PascalCase"
        has_upper = any(c.isupper() for c in name[1:])
        if has_upper:
            return "camelCase"
        if name == name.lower():
            return "lowercase"
        if name == name.upper():
            return "UPPERCASE"
        return "other"

    # ── function signature diversity ─────────────────────────────
    def compute_function_signature_diversity(
        self, codes: List[str]
    ) -> Dict[str, Any]:
        """Measure diversity of function signatures."""
        if not codes:
            return {"diversity": 0.0, "details": {}}

        func_patterns = [
            re.compile(r"def\s+(\w+)\s*\(([^)]*)\)"),
            re.compile(r"function\s+(\w+)\s*\(([^)]*)\)"),
            re.compile(r"(?:public|private|protected|static)?\s*(?:\w+)\s+(\w+)\s*\(([^)]*)\)"),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>"),
            re.compile(r"fn\s+(\w+)\s*\(([^)]*)\)"),
            re.compile(r"func\s+(\w+)\s*\(([^)]*)\)"),
        ]

        per_code_sigs: List[List[Dict[str, Any]]] = []
        for code in codes:
            sigs: List[Dict[str, Any]] = []
            for pat in func_patterns:
                for m in pat.finditer(code):
                    name = m.group(1)
                    params_str = m.group(2).strip()
                    params = [
                        p.strip() for p in params_str.split(",")
                        if p.strip()
                    ] if params_str else []
                    sigs.append({
                        "name": name,
                        "param_count": len(params),
                        "params": params,
                    })
            per_code_sigs.append(sigs)

        if len(per_code_sigs) >= 2:
            sig_sets = [
                set((s["name"], s["param_count"]) for s in sigs)
                for sigs in per_code_sigs
            ]
            overlaps: List[float] = []
            for i in range(len(sig_sets)):
                for j in range(i + 1, len(sig_sets)):
                    overlaps.append(1.0 - _jaccard(sig_sets[i], sig_sets[j]))
            name_div = float(np.mean(overlaps)) if overlaps else 0.0
        else:
            all_names = set()
            for sigs in per_code_sigs:
                for s in sigs:
                    all_names.add(s["name"])
            name_div = min(len(all_names) / 10.0, 1.0)

        param_counts: List[List[int]] = []
        for sigs in per_code_sigs:
            param_counts.append([s["param_count"] for s in sigs])

        pc_counter: Counter = Counter()
        for pcs in param_counts:
            for pc in pcs:
                pc_counter[pc] += 1
        pc_ent = _normalised_entropy(pc_counter)

        total_funcs = sum(len(s) for s in per_code_sigs)
        func_count_div = float(np.std([len(s) for s in per_code_sigs])) / (
            float(np.mean([len(s) for s in per_code_sigs])) + 1e-12
        ) if per_code_sigs else 0.0

        diversity = 0.4 * name_div + 0.3 * pc_ent + 0.3 * min(func_count_div, 1.0)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "name_diversity": name_div,
                "param_count_entropy": pc_ent,
                "func_count_diversity": func_count_div,
                "total_functions": total_funcs,
            },
        }

    # ── code complexity metrics ──────────────────────────────────
    def compute_code_complexity_metrics(
        self, codes: List[str]
    ) -> Dict[str, Any]:
        """Approximate cyclomatic complexity and other code metrics."""
        if not codes:
            return {"diversity": 0.0, "details": {}}

        branch_re = re.compile(
            r"\b(if|elif|else\s+if|case|catch|except|&&|\|\||and|or)\b"
        )
        loop_re = re.compile(r"\b(for|while|do)\b")

        complexities: List[Dict[str, float]] = []
        for code in codes:
            lines = code.split("\n")
            non_blank = [l for l in lines if l.strip()]
            loc = len(non_blank)

            branches = len(branch_re.findall(code))
            loops = len(loop_re.findall(code))
            cyclomatic = 1 + branches + loops

            nesting_depths: List[int] = []
            indent_size = self.config.code_indent_size
            for line in non_blank:
                leading = len(line) - len(line.lstrip())
                nesting_depths.append(leading // max(indent_size, 1))
            max_nesting = max(nesting_depths) if nesting_depths else 0
            avg_nesting = float(np.mean(nesting_depths)) if nesting_depths else 0.0

            unique_tokens = len(set(tokenize_simple(code)))
            total_tokens = len(tokenize_simple(code))
            halstead_volume = (
                total_tokens * math.log2(unique_tokens + 1)
                if unique_tokens > 0 else 0.0
            )

            complexities.append({
                "loc": float(loc),
                "cyclomatic": float(cyclomatic),
                "max_nesting": float(max_nesting),
                "avg_nesting": avg_nesting,
                "halstead_volume": halstead_volume,
                "branches": float(branches),
                "loops": float(loops),
            })

        metrics = list(complexities[0].keys())
        diversity_scores: Dict[str, float] = {}
        for m in metrics:
            vals = [c[m] for c in complexities]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            diversity_scores[f"{m}_cv"] = std_v / (mean_v + 1e-12)

        overall = float(np.mean(list(diversity_scores.values())))

        return {
            "diversity": float(np.clip(overall, 0.0, 1.0)),
            "details": {
                "per_metric_cv": diversity_scores,
                "mean_complexities": {
                    m: float(np.mean([c[m] for c in complexities]))
                    for m in metrics
                },
            },
        }

    # ── API usage diversity ──────────────────────────────────────
    def compute_api_usage_diversity(
        self, codes: List[str]
    ) -> Dict[str, Any]:
        """Measure diversity of API / library calls across code samples."""
        if not codes:
            return {"diversity": 0.0, "details": {}}

        import_re = [
            re.compile(r"^\s*import\s+(.+)", re.MULTILINE),
            re.compile(r"^\s*from\s+(\S+)\s+import", re.MULTILINE),
            re.compile(r'^\s*(?:const|let|var)\s+.*=\s*require\s*\(\s*["\'](.+?)["\']\s*\)',
                       re.MULTILINE),
            re.compile(r"^\s*#include\s*[<\"](.+?)[>\"]", re.MULTILINE),
            re.compile(r"^\s*using\s+(\S+)\s*;", re.MULTILINE),
        ]

        call_re = re.compile(r"\b(\w+(?:\.\w+)*)\s*\(")

        per_code_imports: List[Set[str]] = []
        per_code_calls: List[Counter] = []

        for code in codes:
            imports: Set[str] = set()
            for pat in import_re:
                for m in pat.finditer(code):
                    imp = m.group(1).strip().split(" as ")[0].strip()
                    imp = imp.split(",")[0].strip()
                    imports.add(imp)
            per_code_imports.append(imports)

            calls: Counter[str] = Counter()
            for m in call_re.finditer(code):
                func = m.group(1)
                if func.lower() not in _CODE_KEYWORDS and len(func) > 1:
                    calls[func] += 1
            per_code_calls.append(calls)

        if len(per_code_imports) >= 2:
            imp_divs: List[float] = []
            for i in range(len(per_code_imports)):
                for j in range(i + 1, len(per_code_imports)):
                    imp_divs.append(
                        1.0 - _jaccard(
                            per_code_imports[i], per_code_imports[j]
                        )
                    )
            import_diversity = float(np.mean(imp_divs))
        else:
            all_imports: Set[str] = set()
            for ims in per_code_imports:
                all_imports.update(ims)
            import_diversity = min(len(all_imports) / 10.0, 1.0)

        if len(per_code_calls) >= 2:
            call_divs: List[float] = []
            for i in range(len(per_code_calls)):
                for j in range(i + 1, len(per_code_calls)):
                    call_divs.append(
                        1.0 - _cosine_similarity_counters(
                            per_code_calls[i], per_code_calls[j]
                        )
                    )
            call_diversity = float(np.mean(call_divs))
        else:
            gc: Counter = Counter()
            for c in per_code_calls:
                gc.update(c)
            call_diversity = _normalised_entropy(gc)

        diversity = 0.5 * import_diversity + 0.5 * call_diversity

        all_imports_set: Set[str] = set()
        for ims in per_code_imports:
            all_imports_set.update(ims)
        all_calls: Counter = Counter()
        for c in per_code_calls:
            all_calls.update(c)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "import_diversity": import_diversity,
                "call_diversity": call_diversity,
                "unique_imports": len(all_imports_set),
                "unique_calls": len(all_calls),
                "top_calls": dict(all_calls.most_common(20)),
            },
        }

    # ── code pattern similarity ──────────────────────────────────
    def compute_code_pattern_similarity(
        self, codes: List[str]
    ) -> Dict[str, Any]:
        """Compare code samples using structural pattern fingerprints."""
        if len(codes) < 2:
            return {"diversity": 0.0, "similarity_matrix": [], "details": {}}

        fingerprints: List[Dict[str, Any]] = []
        for code in codes:
            struct = self.extract_code_structure(code)
            fp: Dict[str, float] = {
                "total_lines": float(struct["total_lines"]),
                "blank_ratio": (struct["blank_lines"] /
                                max(struct["total_lines"], 1)),
                "comment_ratio": (struct["comment_lines"] /
                                  max(struct["total_lines"], 1)),
                "max_indent": float(struct["max_indent"]),
            }

            depths = struct["block_depths"]
            if depths:
                fp["mean_depth"] = float(np.mean(depths))
                fp["std_depth"] = float(np.std(depths))
                fp["depth_range"] = float(max(depths) - min(depths))
            else:
                fp["mean_depth"] = 0.0
                fp["std_depth"] = 0.0
                fp["depth_range"] = 0.0

            line_types = struct["line_types"]
            lt_counter = Counter(line_types)
            fp["code_ratio"] = lt_counter.get("code", 0) / max(len(line_types), 1)

            tokens = tokenize_simple(code)
            kw_count = sum(1 for t in tokens if t in _CODE_KEYWORDS)
            fp["keyword_density"] = kw_count / max(len(tokens), 1)

            fingerprints.append(fp)

        keys = sorted(fingerprints[0].keys())
        mat = np.array(
            [[fp[k] for k in keys] for fp in fingerprints],
            dtype=np.float64,
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        mat_n = mat / norms

        n = len(codes)
        sim_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                dot = float(np.dot(mat_n[i], mat_n[j]))
                sim_mat[i, j] = max(0.0, min(dot, 1.0))

        upper = []
        for i in range(n):
            for j in range(i + 1, n):
                upper.append(1.0 - sim_mat[i, j])
        diversity = float(np.mean(upper)) if upper else 0.0

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "similarity_matrix": sim_mat.tolist(),
            "details": {
                "features": keys,
                "mean_similarity": float(np.mean(sim_mat[
                    np.triu_indices_from(sim_mat, k=1)
                ])) if n >= 2 else 0.0,
            },
        }


# ──────────────────────────────────────────────────────────────────────
# 7. TextStructureAnalyzer
# ──────────────────────────────────────────────────────────────────────

class TextStructureAnalyzer:
    """Analyse the structural diversity of natural-language text outputs
    at the paragraph, discourse, and formatting level."""

    def __init__(self, config: Optional[StructuralConfig] = None) -> None:
        self.config = config or StructuralConfig()

    # ── paragraph structure diversity ────────────────────────────
    def compute_paragraph_structure_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Diversity of paragraph count, lengths, and ordering."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        bins = self.config.paragraph_length_bins
        per_text_para_counts: List[int] = []
        per_text_hists: List[np.ndarray] = []
        per_text_lengths: List[List[int]] = []

        for text in texts:
            paras = split_paragraphs(text)
            per_text_para_counts.append(len(paras))
            lengths = [len(tokenize_to_sentences(p)) for p in paras]
            per_text_lengths.append(lengths)
            per_text_hists.append(
                _bin_values([float(x) for x in lengths], bins)
            )

        count_cv = (
            float(np.std(per_text_para_counts))
            / (float(np.mean(per_text_para_counts)) + 1e-12)
        )

        if len(per_text_hists) >= 2:
            hist_divs: List[float] = []
            for i in range(len(per_text_hists)):
                for j in range(i + 1, len(per_text_hists)):
                    hist_divs.append(
                        1.0 - _histogram_intersection(
                            per_text_hists[i], per_text_hists[j]
                        )
                    )
            hist_div = float(np.mean(hist_divs))
        else:
            hist_div = 0.0

        length_shapes: List[str] = []
        for lengths in per_text_lengths:
            if not lengths:
                length_shapes.append("empty")
            elif len(lengths) == 1:
                length_shapes.append("single")
            else:
                if lengths == sorted(lengths):
                    length_shapes.append("ascending")
                elif lengths == sorted(lengths, reverse=True):
                    length_shapes.append("descending")
                elif (lengths[0] < max(lengths)
                      and lengths[-1] < max(lengths)):
                    length_shapes.append("arch")
                else:
                    length_shapes.append("varied")
        shape_counter = Counter(length_shapes)
        shape_ent = _normalised_entropy(shape_counter)

        diversity = (0.3 * min(count_cv, 1.0)
                     + 0.4 * hist_div
                     + 0.3 * shape_ent)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "paragraph_count_cv": count_cv,
                "histogram_diversity": hist_div,
                "shape_entropy": shape_ent,
                "shapes": dict(shape_counter),
                "mean_paragraph_count": float(np.mean(per_text_para_counts)),
            },
        }

    # ── discourse pattern diversity ──────────────────────────────
    def compute_discourse_pattern_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Classify paragraphs into discourse roles (introduction, body,
        conclusion) and measure role-pattern diversity."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        intro_markers = re.compile(
            r"\b(introduction|overview|abstract|summary|background|"
            r"in this paper|we present|this paper|the purpose|"
            r"this article|first|to begin|let us consider|"
            r"we introduce|the goal|the aim|in this work)\b",
            re.IGNORECASE,
        )
        conclusion_markers = re.compile(
            r"\b(conclusion|finally|in summary|to summarize|"
            r"in conclusion|overall|to conclude|therefore|"
            r"thus|hence|as a result|consequently|"
            r"summing up|in closing|to sum up|"
            r"all in all|on the whole|in the end)\b",
            re.IGNORECASE,
        )
        transition_markers = re.compile(
            r"\b(however|moreover|furthermore|additionally|"
            r"on the other hand|in contrast|similarly|"
            r"likewise|consequently|nevertheless|"
            r"meanwhile|subsequently|accordingly|"
            r"in addition|besides|alternatively)\b",
            re.IGNORECASE,
        )
        example_markers = re.compile(
            r"\b(for example|for instance|such as|"
            r"e\.g\.|consider|specifically|"
            r"in particular|to illustrate|namely)\b",
            re.IGNORECASE,
        )

        per_text_patterns: List[List[str]] = []
        for text in texts:
            paras = split_paragraphs(text)
            if not paras:
                per_text_patterns.append(["empty"])
                continue

            roles: List[str] = []
            for idx, para in enumerate(paras):
                if idx == 0 and intro_markers.search(para):
                    roles.append("introduction")
                elif idx == len(paras) - 1 and conclusion_markers.search(para):
                    roles.append("conclusion")
                elif example_markers.search(para):
                    roles.append("example")
                elif transition_markers.search(para):
                    roles.append("transition")
                else:
                    roles.append("body")
            per_text_patterns.append(roles)

        # Pattern as tuple for set comparison
        pattern_tuples = [tuple(p) for p in per_text_patterns]
        unique_patterns = len(set(pattern_tuples))
        pattern_counter = Counter(pattern_tuples)
        pattern_ent = _normalised_entropy(pattern_counter)

        # Role distribution diversity
        per_text_role_counts: List[Counter] = []
        for roles in per_text_patterns:
            per_text_role_counts.append(Counter(roles))

        if len(per_text_role_counts) >= 2:
            role_divs: List[float] = []
            for i in range(len(per_text_role_counts)):
                for j in range(i + 1, len(per_text_role_counts)):
                    role_divs.append(
                        1.0 - _cosine_similarity_counters(
                            per_text_role_counts[i], per_text_role_counts[j]
                        )
                    )
            role_div = float(np.mean(role_divs))
        else:
            gc: Counter = Counter()
            for c in per_text_role_counts:
                gc.update(c)
            role_div = _normalised_entropy(gc)

        has_intro = sum(1 for p in per_text_patterns if "introduction" in p)
        has_concl = sum(1 for p in per_text_patterns if "conclusion" in p)
        has_example = sum(1 for p in per_text_patterns if "example" in p)
        has_transition = sum(1 for p in per_text_patterns if "transition" in p)

        diversity = 0.5 * pattern_ent + 0.5 * role_div

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "unique_discourse_patterns": unique_patterns,
                "pattern_entropy": pattern_ent,
                "role_distribution_diversity": role_div,
                "texts_with_intro": has_intro,
                "texts_with_conclusion": has_concl,
                "texts_with_examples": has_example,
                "texts_with_transitions": has_transition,
            },
        }

    # ── rhetorical structure diversity ───────────────────────────
    def compute_rhetorical_structure_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Detect rhetorical moves and measure their diversity."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        rhetorical_cues: Dict[str, re.Pattern] = {
            "claim": re.compile(
                r"\b(we argue|I believe|it is clear|"
                r"we claim|the evidence shows|undoubtedly|"
                r"it is evident|we contend|I assert)\b",
                re.IGNORECASE,
            ),
            "evidence": re.compile(
                r"\b(research shows|studies indicate|data suggests|"
                r"according to|evidence suggests|results show|"
                r"findings indicate|statistics reveal|experiments demonstrate)\b",
                re.IGNORECASE,
            ),
            "concession": re.compile(
                r"\b(although|admittedly|while it is true|"
                r"granted|it must be acknowledged|"
                r"despite|notwithstanding|even though)\b",
                re.IGNORECASE,
            ),
            "refutation": re.compile(
                r"\b(however|on the contrary|in contrast|"
                r"nevertheless|but|yet|opponents argue|"
                r"critics suggest|some might say)\b",
                re.IGNORECASE,
            ),
            "elaboration": re.compile(
                r"\b(in other words|that is|specifically|"
                r"more precisely|to clarify|put differently|"
                r"this means|in fact|indeed)\b",
                re.IGNORECASE,
            ),
            "cause_effect": re.compile(
                r"\b(because|therefore|thus|hence|consequently|"
                r"as a result|due to|owing to|leads to|"
                r"results in|causes|so that)\b",
                re.IGNORECASE,
            ),
            "comparison": re.compile(
                r"\b(similarly|likewise|in comparison|compared to|"
                r"just as|in the same way|analogous|"
                r"parallel|corresponds to)\b",
                re.IGNORECASE,
            ),
            "enumeration": re.compile(
                r"\b(first|second|third|finally|"
                r"firstly|secondly|thirdly|lastly|"
                r"next|then|additionally|moreover)\b",
                re.IGNORECASE,
            ),
            "question": re.compile(r"\?"),
            "definition": re.compile(
                r"\b(is defined as|refers to|means|"
                r"is known as|is called|denotes|"
                r"signifies|represents)\b",
                re.IGNORECASE,
            ),
        }

        per_text: List[Counter] = []
        per_text_sequences: List[List[str]] = []

        for text in texts:
            sentences = tokenize_to_sentences(text)
            counter: Counter[str] = Counter()
            sequence: List[str] = []

            for sent in sentences:
                found_cues: List[str] = []
                for move, pattern in rhetorical_cues.items():
                    if pattern.search(sent):
                        counter[move] += 1
                        found_cues.append(move)
                if found_cues:
                    sequence.extend(found_cues)
                else:
                    sequence.append("neutral")

            per_text.append(counter)
            per_text_sequences.append(sequence)

        if len(per_text) >= 2:
            move_divs: List[float] = []
            for i in range(len(per_text)):
                for j in range(i + 1, len(per_text)):
                    move_divs.append(
                        1.0 - _cosine_similarity_counters(
                            per_text[i], per_text[j]
                        )
                    )
            distribution_div = float(np.mean(move_divs))
        else:
            gc: Counter = Counter()
            for c in per_text:
                gc.update(c)
            distribution_div = _normalised_entropy(gc)

        if len(per_text_sequences) >= 2:
            seq_divs: List[float] = []
            for i in range(len(per_text_sequences)):
                for j in range(i + 1, len(per_text_sequences)):
                    max_l = max(
                        len(per_text_sequences[i]),
                        len(per_text_sequences[j]), 1,
                    )
                    d = levenshtein_distance(
                        per_text_sequences[i], per_text_sequences[j]
                    )
                    seq_divs.append(d / max_l)
            sequence_div = float(np.mean(seq_divs))
        else:
            sequence_div = 0.0

        diversity = 0.5 * distribution_div + 0.5 * sequence_div

        global_c: Counter = Counter()
        for c in per_text:
            global_c.update(c)

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "distribution_diversity": distribution_div,
                "sequence_diversity": sequence_div,
                "global_move_counts": dict(global_c.most_common()),
                "unique_moves_used": len(global_c),
            },
        }

    # ── formatting diversity ─────────────────────────────────────
    def compute_formatting_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Detect and compare formatting features (lists, headers, emphasis)."""
        if not texts:
            return {"diversity": 0.0, "details": {}}

        header_re = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)
        ul_re = re.compile(r"^\s*[-*+]\s+.+", re.MULTILINE)
        ol_re = re.compile(r"^\s*\d+[.)]\s+.+", re.MULTILINE)
        bold_re = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
        italic_re = re.compile(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)|(?<!_)_(?!_)[^_]+_(?!_)")
        code_inline_re = re.compile(r"`[^`]+`")
        code_block_re = re.compile(r"```[\s\S]*?```")
        link_re = re.compile(r"\[([^\]]+)\]\([^)]+\)")
        blockquote_re = re.compile(r"^>\s+.+", re.MULTILINE)
        table_re = re.compile(r"^\|.+\|$", re.MULTILINE)

        feature_names = [
            "headers", "unordered_lists", "ordered_lists",
            "bold", "italic", "inline_code", "code_blocks",
            "links", "blockquotes", "tables",
        ]
        patterns = [
            header_re, ul_re, ol_re, bold_re, italic_re,
            code_inline_re, code_block_re, link_re, blockquote_re, table_re,
        ]

        per_text_features: List[Dict[str, int]] = []
        for text in texts:
            features: Dict[str, int] = {}
            for name, pat in zip(feature_names, patterns):
                features[name] = len(pat.findall(text))
            per_text_features.append(features)

        per_text_binary: List[Set[str]] = []
        for feat in per_text_features:
            per_text_binary.append(
                {k for k, v in feat.items() if v > 0}
            )

        if len(per_text_binary) >= 2:
            feat_divs: List[float] = []
            for i in range(len(per_text_binary)):
                for j in range(i + 1, len(per_text_binary)):
                    feat_divs.append(
                        1.0 - _jaccard(per_text_binary[i], per_text_binary[j])
                    )
            presence_div = float(np.mean(feat_divs))
        else:
            all_present: Set[str] = set()
            for bs in per_text_binary:
                all_present.update(bs)
            presence_div = len(all_present) / len(feature_names)

        # Intensity diversity via vectors
        if len(per_text_features) >= 2:
            vecs = np.array(
                [[f[k] for k in feature_names] for f in per_text_features],
                dtype=np.float64,
            )
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            vecs_n = vecs / norms
            dists = pdist(vecs_n, metric="cosine")
            dists = np.nan_to_num(dists, nan=0.0)
            intensity_div = float(np.mean(dists))
        else:
            intensity_div = 0.0

        diversity = 0.5 * presence_div + 0.5 * intensity_div

        global_features: Dict[str, int] = defaultdict(int)
        for feat in per_text_features:
            for k, v in feat.items():
                global_features[k] += v

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "presence_diversity": presence_div,
                "intensity_diversity": intensity_div,
                "global_feature_counts": dict(global_features),
                "features_used_per_text": [
                    len(bs) for bs in per_text_binary
                ],
            },
        }

    # ── information flow diversity ───────────────────────────────
    def compute_information_flow_diversity(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Analyse topic progression patterns across sentences.

        Tracks how the dominant content words shift from sentence to sentence
        and compares these progression patterns across texts.
        """
        if not texts:
            return {"diversity": 0.0, "details": {}}

        stop_words = frozenset(
            list(_DETERMINERS) + list(_PREPOSITIONS)
            + list(_CONJUNCTIONS) + list(_PRONOUNS)
            + list(_AUXILIARY_VERBS)
            + ["is", "am", "are", "was", "were", "be", "been", "being",
               "have", "has", "had", "do", "does", "did", "a", "an",
               "the", "not", "no", "yes"]
        )

        per_text_flows: List[List[float]] = []
        per_text_topic_chains: List[List[Set[str]]] = []

        for text in texts:
            sentences = tokenize_to_sentences(text)
            if len(sentences) < 2:
                per_text_flows.append([])
                per_text_topic_chains.append([])
                continue

            sent_topics: List[Set[str]] = []
            for sent in sentences:
                tokens = tokenize_simple(sent)
                content_words = {
                    t for t in tokens
                    if t not in stop_words and len(t) > 2
                    and not _PUNCT_RE.fullmatch(t)
                }
                sent_topics.append(content_words)
            per_text_topic_chains.append(sent_topics)

            flow: List[float] = []
            for k in range(len(sent_topics) - 1):
                overlap = _jaccard(sent_topics[k], sent_topics[k + 1])
                flow.append(1.0 - overlap)
            per_text_flows.append(flow)

        if len(per_text_flows) >= 2:
            flow_summaries: List[Tuple[float, float]] = []
            for fl in per_text_flows:
                if fl:
                    flow_summaries.append(
                        (float(np.mean(fl)), float(np.std(fl)))
                    )
                else:
                    flow_summaries.append((0.0, 0.0))

            mean_vals = [s[0] for s in flow_summaries]
            std_vals = [s[1] for s in flow_summaries]
            mean_div = _mean_pairwise(mean_vals)
            std_div = _mean_pairwise(std_vals)
            flow_diversity = 0.5 * min(mean_div, 1.0) + 0.5 * min(std_div, 1.0)
        else:
            flow_diversity = 0.0

        # Topic vocabulary diversity
        if len(per_text_topic_chains) >= 2:
            all_topic_sets = [
                set().union(*chain) if chain else set()
                for chain in per_text_topic_chains
            ]
            topic_divs: List[float] = []
            for i in range(len(all_topic_sets)):
                for j in range(i + 1, len(all_topic_sets)):
                    topic_divs.append(
                        1.0 - _jaccard(all_topic_sets[i], all_topic_sets[j])
                    )
            vocab_div = float(np.mean(topic_divs))
        else:
            vocab_div = 0.0

        diversity = 0.5 * flow_diversity + 0.5 * vocab_div

        avg_flow = float(np.mean([
            np.mean(fl) if fl else 0.0 for fl in per_text_flows
        ]))

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "flow_pattern_diversity": flow_diversity,
                "topic_vocabulary_diversity": vocab_div,
                "average_topic_shift": avg_flow,
            },
        }

    # ── cohesion metrics ─────────────────────────────────────────
    def compute_cohesion_metrics(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Measure cohesion diversity via lexical chains and reference patterns.

        Lexical chain: sequence of semantically related content words
        (approximated by exact repetition across sentences).
        Reference pattern: pronoun / demonstrative use patterns.
        """
        if not texts:
            return {"diversity": 0.0, "details": {}}

        stop_words = frozenset(
            list(_DETERMINERS) + list(_PREPOSITIONS)
            + list(_CONJUNCTIONS) + list(_PRONOUNS)
            + list(_AUXILIARY_VERBS)
        )

        per_text_cohesion: List[Dict[str, float]] = []

        for text in texts:
            sentences = tokenize_to_sentences(text)
            if len(sentences) < 2:
                per_text_cohesion.append({
                    "lexical_overlap": 0.0,
                    "chain_length": 0.0,
                    "pronoun_ratio": 0.0,
                    "demonstrative_ratio": 0.0,
                    "connective_ratio": 0.0,
                })
                continue

            # Lexical chains (simple repetition-based)
            word_positions: Dict[str, List[int]] = defaultdict(list)
            all_tokens: List[List[str]] = []
            total_words = 0
            for s_idx, sent in enumerate(sentences):
                tokens = tokenize_simple(sent)
                content = [
                    t for t in tokens
                    if t not in stop_words and len(t) > 2
                    and not _PUNCT_RE.fullmatch(t)
                ]
                all_tokens.append(content)
                for w in content:
                    word_positions[w].append(s_idx)
                total_words += len(tokens)

            chains = {
                w: positions for w, positions in word_positions.items()
                if len(positions) >= 2
            }
            avg_chain_length = (
                float(np.mean([len(p) for p in chains.values()]))
                if chains else 0.0
            )

            # Adjacent sentence lexical overlap
            overlaps: List[float] = []
            for k in range(len(all_tokens) - 1):
                s1 = set(all_tokens[k])
                s2 = set(all_tokens[k + 1])
                overlaps.append(_jaccard(s1, s2))
            avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0

            # Pronoun and demonstrative ratios
            flat_tokens = tokenize_simple(text)
            pronoun_count = sum(
                1 for t in flat_tokens if t.lower() in _PRONOUNS
            )
            demonstrative = {"this", "that", "these", "those", "such"}
            demo_count = sum(
                1 for t in flat_tokens if t.lower() in demonstrative
            )
            connectives = {
                "however", "therefore", "moreover", "furthermore",
                "nevertheless", "consequently", "thus", "hence",
                "additionally", "meanwhile", "similarly", "likewise",
            }
            conn_count = sum(
                1 for t in flat_tokens if t.lower() in connectives
            )

            n_tokens = max(len(flat_tokens), 1)

            per_text_cohesion.append({
                "lexical_overlap": avg_overlap,
                "chain_length": avg_chain_length,
                "pronoun_ratio": pronoun_count / n_tokens,
                "demonstrative_ratio": demo_count / n_tokens,
                "connective_ratio": conn_count / n_tokens,
            })

        if not per_text_cohesion:
            return {"diversity": 0.0, "details": {}}

        # Diversity of cohesion profiles
        keys = sorted(per_text_cohesion[0].keys())
        if len(per_text_cohesion) >= 2:
            mat = np.array(
                [[c[k] for k in keys] for c in per_text_cohesion],
                dtype=np.float64,
            )
            cvs: List[float] = []
            for col in range(mat.shape[1]):
                mean_v = float(np.mean(mat[:, col]))
                std_v = float(np.std(mat[:, col]))
                cvs.append(std_v / (mean_v + 1e-12))
            diversity = float(np.mean([min(cv, 1.0) for cv in cvs]))
        else:
            diversity = 0.0

        means = {
            k: float(np.mean([c[k] for c in per_text_cohesion]))
            for k in keys
        }

        return {
            "diversity": float(np.clip(diversity, 0.0, 1.0)),
            "details": {
                "mean_cohesion_metrics": means,
                "num_texts": len(texts),
            },
        }


# ──────────────────────────────────────────────────────────────────────
# 8. StructuralMetricsSuite
# ──────────────────────────────────────────────────────────────────────

class StructuralMetricsSuite:
    """Unified interface for all structural diversity metrics."""

    def __init__(self, config: Optional[StructuralConfig] = None) -> None:
        self.config = config or StructuralConfig()
        self.syntactic = SyntacticDiversityAnalyzer(self.config)
        self.parse_tree = ParseTreeSimilarity(self.config)
        self.code = CodeStructureDiversity(self.config)
        self.text = TextStructureAnalyzer(self.config)

    def compute_all_structural_metrics(
        self,
        texts: List[str],
        include_code: bool = False,
        include_parse_tree: bool = True,
    ) -> Dict[str, Any]:
        """Compute every structural metric in one call.

        Parameters
        ----------
        texts : list of str
            The generated texts to evaluate.
        include_code : bool
            Whether to run code-specific metrics (slower).
        include_parse_tree : bool
            Whether to run parse-tree metrics (slower for many texts).

        Returns
        -------
        dict with keys for each metric group and an overall score.
        """
        logger.info("Computing all structural metrics for %d texts", len(texts))
        results: Dict[str, Any] = {}

        # Syntactic metrics
        results["pos_pattern_diversity"] = (
            self.syntactic.compute_pos_pattern_diversity(texts)
        )
        results["sentence_structure_diversity"] = (
            self.syntactic.compute_sentence_structure_diversity(texts)
        )
        results["dependency_pattern_diversity"] = (
            self.syntactic.compute_dependency_pattern_diversity(texts)
        )
        results["phrase_structure_diversity"] = (
            self.syntactic.compute_phrase_structure_diversity(texts)
        )
        results["syntactic_complexity"] = (
            self.syntactic.compute_syntactic_complexity(texts)
        )
        results["construction_diversity"] = (
            self.syntactic.compute_construction_diversity(texts)
        )

        # Parse-tree metrics
        if include_parse_tree and len(texts) <= 50:
            results["tree_edit_diversity"] = (
                self.parse_tree.average_tree_diversity(texts, metric="edit")
            )
            if len(texts) <= 20:
                results["tree_kernel_diversity"] = (
                    self.parse_tree.average_tree_diversity(
                        texts, metric="kernel"
                    )
                )

        # Text-structure metrics
        results["paragraph_structure_diversity"] = (
            self.text.compute_paragraph_structure_diversity(texts)
        )
        results["discourse_pattern_diversity"] = (
            self.text.compute_discourse_pattern_diversity(texts)
        )
        results["rhetorical_structure_diversity"] = (
            self.text.compute_rhetorical_structure_diversity(texts)
        )
        results["formatting_diversity"] = (
            self.text.compute_formatting_diversity(texts)
        )
        results["information_flow_diversity"] = (
            self.text.compute_information_flow_diversity(texts)
        )
        results["cohesion_diversity"] = (
            self.text.compute_cohesion_metrics(texts)
        )

        # Compression-based
        results["compression_ratio_diversity"] = (
            compute_compression_ratio_diversity(texts)
        )

        # Code metrics
        if include_code:
            results["control_flow_diversity"] = (
                self.code.compute_control_flow_diversity(texts)
            )
            results["variable_naming_diversity"] = (
                self.code.compute_variable_naming_diversity(texts)
            )
            results["function_signature_diversity"] = (
                self.code.compute_function_signature_diversity(texts)
            )
            results["code_complexity_diversity"] = (
                self.code.compute_code_complexity_metrics(texts)
            )
            results["api_usage_diversity"] = (
                self.code.compute_api_usage_diversity(texts)
            )
            results["code_pattern_similarity"] = (
                self.code.compute_code_pattern_similarity(texts)
            )

        return results

    def compute_aggregate_structural_diversity(
        self,
        texts: List[str],
        include_code: bool = False,
        include_parse_tree: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Single aggregate diversity score from all structural metrics.

        Parameters
        ----------
        weights : dict, optional
            Override per-metric-group weights.  Keys should match
            ``self.config.complexity_weights``.
        """
        all_metrics = self.compute_all_structural_metrics(
            texts,
            include_code=include_code,
            include_parse_tree=include_parse_tree,
        )

        # Group scores
        syntactic_keys = [
            "pos_pattern_diversity", "sentence_structure_diversity",
            "dependency_pattern_diversity", "phrase_structure_diversity",
            "syntactic_complexity", "construction_diversity",
        ]
        parse_tree_keys = [
            "tree_edit_diversity", "tree_kernel_diversity",
        ]
        text_keys = [
            "paragraph_structure_diversity", "discourse_pattern_diversity",
            "rhetorical_structure_diversity", "formatting_diversity",
            "information_flow_diversity", "cohesion_diversity",
            "compression_ratio_diversity",
        ]
        code_keys = [
            "control_flow_diversity", "variable_naming_diversity",
            "function_signature_diversity", "code_complexity_diversity",
            "api_usage_diversity", "code_pattern_similarity",
        ]

        def _group_score(keys: List[str]) -> float:
            scores: List[float] = []
            for k in keys:
                if k in all_metrics:
                    val = all_metrics[k]
                    if isinstance(val, dict) and "diversity" in val:
                        scores.append(val["diversity"])
            return float(np.mean(scores)) if scores else 0.0

        w = weights or self.config.complexity_weights
        syntactic_score = _group_score(syntactic_keys)
        parse_tree_score = _group_score(parse_tree_keys)
        text_score = _group_score(text_keys)
        code_score = _group_score(code_keys)

        groups_present: List[Tuple[str, float, float]] = [
            ("syntactic", syntactic_score, w.get("syntactic", 0.25)),
            ("text_structure", text_score, w.get("text_structure", 0.25)),
        ]
        if include_parse_tree:
            groups_present.append(
                ("parse_tree", parse_tree_score, w.get("parse_tree", 0.25))
            )
        if include_code:
            groups_present.append(
                ("code_structure", code_score, w.get("code_structure", 0.25))
            )

        total_weight = sum(g[2] for g in groups_present)
        if total_weight < 1e-12:
            total_weight = 1.0
        aggregate = sum(g[1] * g[2] for g in groups_present) / total_weight

        return {
            "aggregate_diversity": float(np.clip(aggregate, 0.0, 1.0)),
            "group_scores": {g[0]: g[1] for g in groups_present},
            "weights_used": {g[0]: g[2] for g in groups_present},
            "all_metrics": all_metrics,
        }

    def compare_structural_diversity(
        self,
        texts_a: List[str],
        texts_b: List[str],
        include_code: bool = False,
        include_parse_tree: bool = True,
    ) -> Dict[str, Any]:
        """Compare structural diversity between two sets of texts.

        Returns per-metric comparison and an overall delta.
        """
        agg_a = self.compute_aggregate_structural_diversity(
            texts_a,
            include_code=include_code,
            include_parse_tree=include_parse_tree,
        )
        agg_b = self.compute_aggregate_structural_diversity(
            texts_b,
            include_code=include_code,
            include_parse_tree=include_parse_tree,
        )

        delta_overall = (agg_a["aggregate_diversity"]
                         - agg_b["aggregate_diversity"])

        per_group_delta: Dict[str, float] = {}
        for group in agg_a["group_scores"]:
            score_a = agg_a["group_scores"].get(group, 0.0)
            score_b = agg_b["group_scores"].get(group, 0.0)
            per_group_delta[group] = score_a - score_b

        per_metric_delta: Dict[str, float] = {}
        all_keys = set(agg_a["all_metrics"]) | set(agg_b["all_metrics"])
        for k in all_keys:
            va = agg_a["all_metrics"].get(k, {})
            vb = agg_b["all_metrics"].get(k, {})
            da = va.get("diversity", 0.0) if isinstance(va, dict) else 0.0
            db = vb.get("diversity", 0.0) if isinstance(vb, dict) else 0.0
            per_metric_delta[k] = da - db

        winner = "A" if delta_overall > 0 else ("B" if delta_overall < 0 else "tie")

        return {
            "set_a_aggregate": agg_a["aggregate_diversity"],
            "set_b_aggregate": agg_b["aggregate_diversity"],
            "delta_overall": delta_overall,
            "winner": winner,
            "per_group_delta": per_group_delta,
            "per_metric_delta": per_metric_delta,
            "set_a_details": agg_a,
            "set_b_details": agg_b,
        }
