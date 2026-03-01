"""
Synthetic Data — Generate diverse synthetic data.

Tools for creating diverse synthetic datasets, augmenting existing data,
balancing class distributions, generating edge cases, and creating
adversarial examples. All functions operate on plain Python data structures
with numpy — no external ML framework required.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "was", "are", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "so", "if",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [t for t in _WS_RE.split(text) if t and t not in _STOPWORDS and len(t) > 1]


def _tfidf_matrix(texts: List[str]) -> np.ndarray:
    docs = [_tokenize(t) for t in texts]
    vocab: Dict[str, int] = {}
    for doc in docs:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    n_docs, n_vocab = len(docs), len(vocab)
    if n_vocab == 0:
        return np.zeros((n_docs, 1))
    tf = np.zeros((n_docs, n_vocab))
    for i, doc in enumerate(docs):
        for tok in doc:
            tf[i, vocab[tok]] += 1
    row_sums = tf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tf /= row_sums
    df = (tf > 0).sum(axis=0).astype(float)
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    return tf * idf


def _pairwise_distances(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = mat / norms
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def _avg_diversity(texts: List[str]) -> float:
    if len(texts) < 2:
        return 0.0
    mat = _tfidf_matrix(texts)
    dist = _pairwise_distances(mat)
    n = len(texts)
    total = sum(dist[i, j] for i, j in combinations(range(n), 2))
    return total / (n * (n - 1) / 2)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """A simple dataset container."""
    items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.items)

    def column(self, key: str) -> List[Any]:
        return [item[key] for item in self.items if key in item]

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "Dataset":
        return Dataset(items=[it for it in self.items if predicate(it)],
                       metadata=dict(self.metadata))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Schema-based diverse example generation
# ---------------------------------------------------------------------------

# Supported field types for schema-based generation
_TYPE_GENERATORS: Dict[str, Callable[..., Any]] = {}


def _register_generator(type_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _TYPE_GENERATORS[type_name] = fn
        return fn
    return decorator


@_register_generator("int")
def _gen_int(rng: np.random.Generator, **kwargs: Any) -> int:
    lo = kwargs.get("min", 0)
    hi = kwargs.get("max", 1000)
    return int(rng.integers(lo, hi + 1))


@_register_generator("float")
def _gen_float(rng: np.random.Generator, **kwargs: Any) -> float:
    lo = kwargs.get("min", 0.0)
    hi = kwargs.get("max", 1.0)
    return float(rng.uniform(lo, hi))


@_register_generator("bool")
def _gen_bool(rng: np.random.Generator, **kwargs: Any) -> bool:
    return bool(rng.integers(0, 2))


@_register_generator("str")
def _gen_str(rng: np.random.Generator, **kwargs: Any) -> str:
    length = kwargs.get("length", rng.integers(3, 20))
    chars = string.ascii_lowercase + " "
    return "".join(rng.choice(list(chars)) for _ in range(int(length)))


@_register_generator("choice")
def _gen_choice(rng: np.random.Generator, **kwargs: Any) -> Any:
    options = kwargs.get("options", ["a", "b", "c"])
    return options[rng.integers(0, len(options))]


@_register_generator("date")
def _gen_date(rng: np.random.Generator, **kwargs: Any) -> str:
    year = int(rng.integers(2000, 2025))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    return f"{year:04d}-{month:02d}-{day:02d}"


@_register_generator("email")
def _gen_email(rng: np.random.Generator, **kwargs: Any) -> str:
    name_len = int(rng.integers(4, 10))
    name = "".join(rng.choice(list(string.ascii_lowercase)) for _ in range(name_len))
    domains = ["example.com", "test.org", "demo.net", "sample.io"]
    return f"{name}@{domains[rng.integers(0, len(domains))]}"


def generate_diverse_examples(
    schema: Dict[str, Dict[str, Any]],
    n: int,
    diversity: float = 0.8,
    *,
    seed: int = 42,
    max_attempts: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate *n* diverse examples matching *schema*.

    Parameters
    ----------
    schema : dict mapping field_name -> {"type": ..., **kwargs}
        Supported types: int, float, bool, str, choice, date, email.
    n : int
        Number of examples to generate.
    diversity : float in [0, 1]
        Target diversity level. Higher values push for more diverse outputs.
    seed : int
    max_attempts : int
        Stochastic restarts to achieve diversity target.

    Example schema::

        {
            "age": {"type": "int", "min": 18, "max": 80},
            "score": {"type": "float", "min": 0, "max": 100},
            "category": {"type": "choice", "options": ["A", "B", "C"]},
        }
    """
    rng = np.random.default_rng(seed)
    best_batch: Optional[List[Dict[str, Any]]] = None
    best_div = -1.0

    for attempt in range(max_attempts):
        batch: List[Dict[str, Any]] = []
        for _ in range(n):
            example: Dict[str, Any] = {}
            for field_name, field_spec in schema.items():
                ftype = field_spec.get("type", "str")
                gen = _TYPE_GENERATORS.get(ftype, _gen_str)
                example[field_name] = gen(rng, **{k: v for k, v in field_spec.items() if k != "type"})
            batch.append(example)

        # measure diversity on string representations
        reps = [str(sorted(ex.items())) for ex in batch]
        div = _avg_diversity(reps)

        if div > best_div:
            best_div = div
            best_batch = batch

        if div >= diversity:
            break

        # adjust seed for next attempt
        rng = np.random.default_rng(seed + attempt + 1)

    logger.info("generate_diverse_examples: n=%d, diversity=%.3f", n, best_div)
    return best_batch or batch  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# Dataset augmentation
# ---------------------------------------------------------------------------


def augment_dataset(
    dataset: Union[Dataset, List[Dict[str, Any]]],
    target_diversity: float = 0.8,
    *,
    text_key: str = "text",
    augmentation_factor: float = 2.0,
    seed: int = 42,
) -> Dataset:
    """
    Augment *dataset* to increase diversity toward *target_diversity*.

    Augmentation strategies:
    - Token shuffling within sentences
    - Synonym-like perturbation (character-level)
    - Back-translation approximation (reordering)
    - Insertion of random context

    Parameters
    ----------
    dataset : Dataset or list of dicts
    target_diversity : float
    text_key : str
    augmentation_factor : float
        How many times to multiply the dataset size.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if isinstance(dataset, Dataset):
        items = list(dataset.items)
    else:
        items = list(dataset)

    original_texts = [item.get(text_key, str(item)) for item in items]
    current_div = _avg_diversity(original_texts)

    target_n = int(len(items) * augmentation_factor)
    augmented = list(items)

    strategies = [
        _aug_shuffle_tokens,
        _aug_char_perturb,
        _aug_reorder_sentences,
        _aug_insert_context,
    ]

    attempts = 0
    max_attempts = target_n * 3

    while len(augmented) < target_n and attempts < max_attempts:
        attempts += 1
        source = rng.choice(items)
        strategy = rng.choice(strategies)
        new_item = dict(source)
        text = source.get(text_key, str(source))
        new_text = strategy(text, rng)
        new_item[text_key] = new_text
        new_item["_augmented"] = True

        # check if adding improves diversity
        current_texts = [it.get(text_key, str(it)) for it in augmented[-20:]]
        trial_texts = current_texts + [new_text]
        trial_div = _avg_diversity(trial_texts)
        baseline_div = _avg_diversity(current_texts) if len(current_texts) >= 2 else 0.0

        if trial_div >= baseline_div * 0.9:
            augmented.append(new_item)

    result = Dataset(items=augmented, metadata={
        "original_size": len(items),
        "augmented_size": len(augmented),
        "target_diversity": target_diversity,
    })
    return result


def _aug_shuffle_tokens(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) > 2:
        i, j = rng.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def _aug_char_perturb(text: str, rng: random.Random) -> str:
    if not text:
        return text
    chars = list(text)
    n_perturb = max(1, len(chars) // 20)
    for _ in range(n_perturb):
        idx = rng.randint(0, len(chars) - 1)
        if chars[idx].isalpha():
            # swap case or adjacent char
            if rng.random() < 0.5:
                chars[idx] = chars[idx].swapcase()
            else:
                offset = rng.choice([-1, 1])
                new_char = chr(max(ord('a'), min(ord('z'), ord(chars[idx].lower()) + offset)))
                chars[idx] = new_char
    return "".join(chars)


def _aug_reorder_sentences(text: str, rng: random.Random) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        rng.shuffle(sentences)
    return " ".join(sentences)


def _aug_insert_context(text: str, rng: random.Random) -> str:
    fillers = [
        "Notably,", "In particular,", "Furthermore,", "Additionally,",
        "Interestingly,", "Moreover,", "Specifically,", "Importantly,",
    ]
    tokens = text.split()
    if tokens:
        pos = rng.randint(0, len(tokens))
        tokens.insert(pos, rng.choice(fillers))
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Dataset balancing
# ---------------------------------------------------------------------------


def balance_dataset(
    dataset: Union[Dataset, List[Dict[str, Any]]],
    protected_attrs: List[str],
    *,
    strategy: str = "oversample",
    seed: int = 42,
) -> Dataset:
    """
    Balance *dataset* across *protected_attrs* (e.g., gender, race).

    Parameters
    ----------
    dataset : Dataset or list of dicts
    protected_attrs : list of attribute names to balance across
    strategy : "oversample" | "undersample" | "hybrid"
    seed : int
    """
    rng = random.Random(seed)

    if isinstance(dataset, Dataset):
        items = list(dataset.items)
    else:
        items = list(dataset)

    if not protected_attrs:
        return Dataset(items=items)

    # group by protected attribute combinations
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = tuple(item.get(attr, None) for attr in protected_attrs)
        groups[key].append(item)

    group_sizes = {k: len(v) for k, v in groups.items()}
    max_size = max(group_sizes.values())
    min_size = min(group_sizes.values())

    if strategy == "undersample":
        target = min_size
    elif strategy == "hybrid":
        target = int((max_size + min_size) / 2)
    else:  # oversample
        target = max_size

    balanced: List[Dict[str, Any]] = []
    for key, group in groups.items():
        if len(group) >= target:
            balanced.extend(rng.sample(group, target))
        else:
            balanced.extend(group)
            # oversample
            deficit = target - len(group)
            for _ in range(deficit):
                source = rng.choice(group)
                new_item = dict(source)
                new_item["_balanced"] = True
                balanced.append(new_item)

    rng.shuffle(balanced)

    return Dataset(
        items=balanced,
        metadata={
            "strategy": strategy,
            "protected_attrs": protected_attrs,
            "original_size": len(items),
            "balanced_size": len(balanced),
            "group_sizes": {str(k): v for k, v in group_sizes.items()},
        },
    )


# ---------------------------------------------------------------------------
# Edge case generation
# ---------------------------------------------------------------------------


def generate_edge_cases(
    fn_signature: Dict[str, str],
    n: int = 100,
    *,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate edge-case inputs for a function described by *fn_signature*.

    Parameters
    ----------
    fn_signature : dict mapping param_name -> type_name
        E.g. {"x": "int", "name": "str", "flag": "bool"}
    n : int
        Number of edge cases to generate.

    Edge case strategies per type:
    - int: 0, -1, max_int, min_int, powers of 2, primes
    - float: 0.0, -0.0, inf, -inf, nan, very small, very large
    - str: empty, single char, very long, unicode, special chars
    - bool: True, False
    - list: empty, single element, duplicates, very long
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    edge_values: Dict[str, List[Any]] = {
        "int": [0, -1, 1, -2**31, 2**31 - 1, 2, 7, 13, 100, 999, -100, 2**16],
        "float": [0.0, -0.0, float("inf"), float("-inf"), float("nan"),
                  1e-10, -1e-10, 1e10, -1e10, 0.5, -0.5, 1.0, -1.0,
                  math.pi, math.e, 1e-300, 1e300],
        "str": ["", " ", "a", "ab", "a" * 1000, "\n", "\t", "\0",
                "hello world", "café", "日本語", "<script>", "'; DROP TABLE",
                "null", "None", "undefined", "true", "false", "NaN"],
        "bool": [True, False],
        "list": [[], [0], [1, 1], list(range(100)), [None], [""], [0, "a", True]],
        "dict": [{}, {"a": 1}, {"": ""}, {str(i): i for i in range(50)}],
        "None": [None],
    }

    cases: List[Dict[str, Any]] = []

    # Phase 1: systematic edge values (all combinations up to a limit)
    param_names = list(fn_signature.keys())
    param_types = [fn_signature[p] for p in param_names]
    param_edges = [edge_values.get(t, edge_values["str"]) for t in param_types]

    # add all single-param edge cases
    for p_idx, (pname, edges) in enumerate(zip(param_names, param_edges)):
        for edge_val in edges:
            case: Dict[str, Any] = {}
            for j, (other_name, other_type) in enumerate(zip(param_names, param_types)):
                if j == p_idx:
                    case[other_name] = edge_val
                else:
                    # default value
                    defaults = {"int": 1, "float": 1.0, "str": "test", "bool": True}
                    case[other_name] = defaults.get(other_type, None)
            cases.append(case)
            if len(cases) >= n:
                break
        if len(cases) >= n:
            break

    # Phase 2: random combinations of edge values
    while len(cases) < n:
        case = {}
        for pname, ptype in zip(param_names, param_types):
            edges = edge_values.get(ptype, edge_values["str"])
            case[pname] = rng.choice(edges)
        if case not in cases:
            cases.append(case)

    return cases[:n]


# ---------------------------------------------------------------------------
# Diverse test inputs
# ---------------------------------------------------------------------------


def diverse_test_inputs(
    fn: Callable,
    input_types: Dict[str, str],
    n: int = 50,
    *,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate *n* diverse test inputs for callable *fn*.

    Combines edge cases with random diverse inputs and validates them
    by actually calling *fn* to filter out inputs that cause crashes
    (unless they're interesting edge cases).

    Parameters
    ----------
    fn : callable
        The function to generate inputs for.
    input_types : dict mapping param_name -> type_name
    n : int
        Number of test inputs.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # generate edge cases
    edges = generate_edge_cases(input_types, n=n // 2, seed=seed)

    # generate random diverse inputs
    randoms: List[Dict[str, Any]] = []
    for _ in range(n):
        case: Dict[str, Any] = {}
        for pname, ptype in input_types.items():
            if ptype == "int":
                case[pname] = int(rng.integers(-1000, 1001))
            elif ptype == "float":
                case[pname] = float(rng.uniform(-1000, 1000))
            elif ptype == "str":
                length = int(rng.integers(1, 50))
                case[pname] = "".join(py_rng.choices(string.ascii_letters + " ", k=length))
            elif ptype == "bool":
                case[pname] = bool(rng.integers(0, 2))
            else:
                case[pname] = None
        randoms.append(case)

    all_inputs = edges + randoms

    # classify: valid (fn runs), crash (fn raises), interesting crash
    valid: List[Dict[str, Any]] = []
    crashes: List[Dict[str, Any]] = []

    for inp in all_inputs:
        try:
            fn(**inp)
            valid.append(inp)
        except Exception:
            crashes.append(inp)

    # mix: mostly valid + some crashes (edge cases)
    n_crash = min(len(crashes), n // 5)
    n_valid = min(len(valid), n - n_crash)

    # select diverse valid inputs
    if len(valid) > n_valid:
        reps = [str(sorted(v.items())) for v in valid]
        selected_idx = _greedy_diverse_select(reps, n_valid, seed=seed)
        valid = [valid[i] for i in selected_idx]
    else:
        valid = valid[:n_valid]

    result = valid + crashes[:n_crash]
    py_rng.shuffle(result)
    return result[:n]


def _greedy_diverse_select(texts: List[str], k: int, seed: int = 42) -> List[int]:
    """Greedy max-diversity selection returning indices."""
    if len(texts) <= k:
        return list(range(len(texts)))
    mat = _tfidf_matrix(texts)
    dist = _pairwise_distances(mat)
    n = len(texts)
    rng = np.random.default_rng(seed)

    selected = [int(rng.integers(n))]
    remaining = set(range(n)) - {selected[0]}

    while len(selected) < k and remaining:
        best_idx = -1
        best_min_dist = -1.0
        for idx in remaining:
            min_d = min(dist[idx, s] for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Adversarial dataset generation
# ---------------------------------------------------------------------------


def adversarial_dataset(
    model: Callable[[str], str],
    dataset: Union[Dataset, List[Dict[str, Any]]],
    n: int = 100,
    *,
    text_key: str = "text",
    label_key: str = "label",
    seed: int = 42,
) -> Dataset:
    """
    Generate adversarial examples that are likely to fool *model*.

    Strategies:
    1. Character-level perturbation (typos, swaps)
    2. Word-level insertion/deletion
    3. Semantic-preserving rewrites
    4. Boundary probing (minimal edits near decision boundary)

    Parameters
    ----------
    model : callable(text) -> label
        Model to attack.
    dataset : source dataset to perturb
    n : int
        Number of adversarial examples to generate.
    text_key, label_key : field names
    """
    rng = random.Random(seed)

    if isinstance(dataset, Dataset):
        items = list(dataset.items)
    else:
        items = list(dataset)

    perturbation_fns = [
        _adv_char_swap,
        _adv_char_insert,
        _adv_char_delete,
        _adv_word_swap,
        _adv_word_delete,
        _adv_homoglyph,
    ]

    adversarial: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = n * 10

    while len(adversarial) < n and attempts < max_attempts:
        attempts += 1
        source = rng.choice(items)
        text = source.get(text_key, "")
        original_label = source.get(label_key, "")

        if not text:
            continue

        # apply perturbation
        perturb_fn = rng.choice(perturbation_fns)
        perturbed_text = perturb_fn(text, rng)

        if perturbed_text == text:
            continue

        # check if model prediction changes
        try:
            new_label = model(perturbed_text)
        except Exception:
            continue

        # it's adversarial if the label changed
        if str(new_label) != str(original_label):
            adversarial.append({
                text_key: perturbed_text,
                label_key: original_label,
                "_original_text": text,
                "_predicted_label": new_label,
                "_perturbation": perturb_fn.__name__,
                "_adversarial": True,
            })

    if len(adversarial) < n:
        # fill with perturbations even if they don't fool the model
        for source in rng.choices(items, k=n - len(adversarial)):
            text = source.get(text_key, "")
            perturb_fn = rng.choice(perturbation_fns)
            perturbed = perturb_fn(text, rng)
            adversarial.append({
                text_key: perturbed,
                label_key: source.get(label_key, ""),
                "_original_text": text,
                "_perturbation": perturb_fn.__name__,
                "_adversarial": False,
            })

    return Dataset(
        items=adversarial[:n],
        metadata={
            "n_adversarial": sum(1 for a in adversarial if a.get("_adversarial")),
            "n_total": min(n, len(adversarial)),
            "attempts": attempts,
        },
    )


def _adv_char_swap(text: str, rng: random.Random) -> str:
    chars = list(text)
    if len(chars) < 2:
        return text
    idx = rng.randint(0, len(chars) - 2)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def _adv_char_insert(text: str, rng: random.Random) -> str:
    chars = list(text)
    idx = rng.randint(0, len(chars))
    char = rng.choice(list(string.ascii_lowercase))
    chars.insert(idx, char)
    return "".join(chars)


def _adv_char_delete(text: str, rng: random.Random) -> str:
    chars = list(text)
    if not chars:
        return text
    idx = rng.randint(0, len(chars) - 1)
    chars.pop(idx)
    return "".join(chars)


def _adv_word_swap(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    i, j = rng.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def _adv_word_delete(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    idx = rng.randint(0, len(words) - 1)
    words.pop(idx)
    return " ".join(words)


def _adv_homoglyph(text: str, rng: random.Random) -> str:
    """Replace a character with a visually similar one."""
    homoglyphs = {
        'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
        'x': 'х', 'y': 'у', 'i': 'і',
    }
    chars = list(text)
    replaceable = [i for i, c in enumerate(chars) if c.lower() in homoglyphs]
    if not replaceable:
        return text
    idx = rng.choice(replaceable)
    chars[idx] = homoglyphs[chars[idx].lower()]
    return "".join(chars)
