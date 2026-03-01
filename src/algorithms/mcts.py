"""
Monte Carlo Tree Search (MCTS) Decoding for the Diversity Decoding Arena.
=========================================================================

Implements MCTS-based decoding for diverse text generation.  The algorithm
builds a search tree over the token vocabulary, using the classic four phases
— selection, expansion, simulation (rollout), and backpropagation — to
discover high-quality, diverse sequences without exhaustive enumeration.

Key components
--------------
* **MCTSConfig** — dataclass extending ``DecodingConfig`` with MCTS
  hyper-parameters (simulations, exploration constant, rollout policy,
  progressive widening, virtual loss, etc.).
* **MCTSNode** — a single node in the search tree, storing visit counts,
  accumulated value, prior probability, children, and parent pointer.
* **MCTSTree** — manages the full search tree and exposes the four MCTS
  phases as methods, plus tree-level utilities (pruning, statistics,
  sequence extraction).
* **MCTSDecoding** — the main ``DecodingAlgorithm`` implementation.
  Overrides ``generate()`` to run MCTS simulations and extract diverse
  high-scoring sequences from the resulting tree.
* **RolloutPolicy** — strategy classes for the simulation/rollout phase
  (random, greedy, nucleus, temperature-based).
* **ValueFunction** — pluggable value estimators (perplexity, length-
  normalised log-prob, diversity-aware scoring).
* **MCTSAnalyzer** — post-hoc analysis and visualisation of the search
  tree, visit distributions, and diversity metrics.

References
----------
- Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods*.
  IEEE Transactions on Computational Intelligence and AI in Games.
- Silver, D. et al. (2017). *Mastering the Game of Go without Human
  Knowledge*. Nature.
- Chaffin, A. et al. (2022). *PPO-MCTS: Text Generation with Discriminator-
  Guided Monte Carlo Tree Search*. Findings of NAACL.
"""

from __future__ import annotations

import abc
import collections
import copy
import heapq
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Counter,
    DefaultDict,
    Deque,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    _log_softmax,
    _stable_softmax,
    _top_k_filter,
    sample_token,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEG_INF: float = float("-inf")
_LOG_EPS: float = 1e-10
_EPSILON: float = 1e-8
_DEFAULT_VOCAB_SIZE: int = 50_257  # GPT-2 default


# =========================================================================
# MCTSConfig
# =========================================================================


@dataclass
class MCTSConfig(DecodingConfig):
    """Configuration for MCTS-based decoding.

    Extends :class:`DecodingConfig` with all hyper-parameters needed to
    control the MCTS search: number of simulations, exploration/exploitation
    trade-off, rollout policy, progressive widening, virtual loss for
    parallelism, and diversity-aware value functions.
    """

    algorithm_name: str = "MCTSDecoding"

    # -- core MCTS parameters -----------------------------------------------
    n_simulations: int = 100
    exploration_constant: float = 1.4142135623730951  # sqrt(2)
    rollout_depth: int = 20
    rollout_policy: str = "random"  # 'random', 'greedy', 'nucleus', 'temperature'
    rollout_temperature: float = 1.0
    discount_factor: float = 0.99

    # -- expansion ----------------------------------------------------------
    max_children: int = 50
    top_k_expansion: int = 50

    # -- value function -----------------------------------------------------
    value_function: str = "perplexity"  # 'perplexity', 'length_normalized', 'diversity_aware'
    diversity_weight: float = 0.3

    # -- sequence extraction ------------------------------------------------
    n_sequences: int = 5
    reuse_tree: bool = False

    # -- progressive widening -----------------------------------------------
    progressive_widening: bool = False
    pw_alpha: float = 0.5

    # -- virtual loss (parallel MCTS) ---------------------------------------
    virtual_loss: float = 1.0

    # -- temperature for final selection ------------------------------------
    temperature: float = 1.0

    # -- nucleus rollout parameter ------------------------------------------
    rollout_top_p: float = 0.9

    # -- length penalty for value function ----------------------------------
    length_penalty_alpha: float = 0.6

    # -- n-gram diversity bonus parameters ----------------------------------
    diversity_ngram_size: int = 3
    diversity_bonus_scale: float = 0.1

    # -- tree management ----------------------------------------------------
    prune_threshold: int = 1
    max_tree_depth: int = 200

    def validate(self) -> List[str]:
        """Validate MCTS-specific configuration."""
        errors = super().validate()
        if self.n_simulations < 1:
            errors.append("n_simulations must be >= 1")
        if self.exploration_constant < 0:
            errors.append("exploration_constant must be >= 0")
        if self.rollout_depth < 1:
            errors.append("rollout_depth must be >= 1")
        if self.rollout_policy not in ("random", "greedy", "nucleus", "temperature"):
            errors.append(
                f"rollout_policy must be one of 'random', 'greedy', "
                f"'nucleus', 'temperature'; got '{self.rollout_policy}'"
            )
        if self.discount_factor <= 0 or self.discount_factor > 1.0:
            errors.append("discount_factor must be in (0, 1]")
        if self.max_children < 1:
            errors.append("max_children must be >= 1")
        if self.value_function not in ("perplexity", "length_normalized", "diversity_aware"):
            errors.append(
                f"value_function must be one of 'perplexity', "
                f"'length_normalized', 'diversity_aware'; "
                f"got '{self.value_function}'"
            )
        if self.n_sequences < 1:
            errors.append("n_sequences must be >= 1")
        if self.pw_alpha <= 0 or self.pw_alpha >= 1.0:
            errors.append("pw_alpha must be in (0, 1)")
        if self.virtual_loss < 0:
            errors.append("virtual_loss must be >= 0")
        if self.top_k_expansion < 1:
            errors.append("top_k_expansion must be >= 1")
        if self.rollout_temperature <= 0:
            errors.append("rollout_temperature must be > 0")
        if self.max_tree_depth < 1:
            errors.append("max_tree_depth must be >= 1")
        return errors


# =========================================================================
# MCTSNode
# =========================================================================


class MCTSNode:
    """A single node in the MCTS search tree.

    Each node represents a token in a partial sequence.  The root node
    corresponds to the end of the prompt; every child corresponds to a
    possible next token.

    Attributes
    ----------
    token_id : int
        The vocabulary index of the token this node represents.
        ``-1`` for the root node.
    parent : MCTSNode or None
        The parent node.  ``None`` only for the root.
    children : dict
        Mapping ``token_id -> MCTSNode`` for expanded children.
    visit_count : int
        Number of times this node has been visited during search.
    total_value : float
        Accumulated value from all rollouts passing through this node.
    prior_probability : float
        Prior probability from the language model softmax, used in PUCT.
    depth : int
        Depth in the tree (root is 0).
    is_terminal : bool
        Whether this node represents an EOS token or max-depth leaf.
    virtual_loss_count : int
        Number of virtual losses currently applied (for parallel MCTS).
    _expanded : bool
        Whether the node has been expanded at least once.
    """

    __slots__ = (
        "token_id",
        "parent",
        "children",
        "visit_count",
        "total_value",
        "prior_probability",
        "depth",
        "is_terminal",
        "virtual_loss_count",
        "_expanded",
        "_sum_squared_value",
    )

    def __init__(
        self,
        token_id: int = -1,
        parent: Optional["MCTSNode"] = None,
        prior_prob: float = 1.0,
    ) -> None:
        self.token_id = token_id
        self.parent = parent
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior_probability: float = prior_prob
        self.depth: int = (parent.depth + 1) if parent is not None else 0
        self.is_terminal: bool = False
        self.virtual_loss_count: int = 0
        self._expanded: bool = False
        self._sum_squared_value: float = 0.0

    # -- value properties ---------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s, a) = total_value / visit_count.

        Returns 0.0 for unvisited nodes to avoid division by zero.
        """
        effective_visits = self.visit_count + self.virtual_loss_count
        if effective_visits == 0:
            return 0.0
        effective_value = self.total_value - self.virtual_loss_count
        return effective_value / effective_visits

    @property
    def value_variance(self) -> float:
        """Variance of the value estimates at this node."""
        if self.visit_count < 2:
            return 0.0
        mean = self.total_value / self.visit_count
        return max(0.0, self._sum_squared_value / self.visit_count - mean * mean)

    @property
    def value_std(self) -> float:
        """Standard deviation of the value estimates."""
        return math.sqrt(self.value_variance)

    # -- selection scores ---------------------------------------------------

    def ucb_score(
        self,
        parent_visits: int,
        exploration_constant: float,
    ) -> float:
        """Compute UCB1 score for this node.

        UCB1 = Q(s, a) + c * sqrt(ln(N_parent) / N_child)

        Parameters
        ----------
        parent_visits : int
            Visit count of the parent node.
        exploration_constant : float
            The exploration constant *c* controlling explore/exploit balance.

        Returns
        -------
        float
            The UCB1 score.  Returns +inf for unvisited nodes.
        """
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visit_count
        )
        return exploitation + exploration

    def puct_score(
        self,
        parent_visits: int,
        c_puct: float,
    ) -> float:
        """Compute PUCT score (Predictor + UCB applied to Trees).

        Used in AlphaGo/AlphaZero-style search:

            PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N_child)

        Parameters
        ----------
        parent_visits : int
            Visit count of the parent node.
        c_puct : float
            Exploration constant scaling the prior term.

        Returns
        -------
        float
            The PUCT score.
        """
        exploitation = self.q_value
        exploration = (
            c_puct
            * self.prior_probability
            * math.sqrt(max(parent_visits, 1))
            / (1.0 + self.visit_count)
        )
        return exploitation + exploration

    def ucb_tuned_score(
        self,
        parent_visits: int,
        exploration_constant: float,
    ) -> float:
        """UCB1-Tuned variant that uses value variance.

        UCB1-Tuned = Q + sqrt( (ln N_parent / N_child) * min(1/4, V) )

        where V = Var + sqrt(2 * ln(N_parent) / N_child).
        """
        if self.visit_count == 0:
            return float("inf")
        ln_parent = math.log(max(parent_visits, 1))
        v_term = self.value_variance + math.sqrt(2.0 * ln_parent / self.visit_count)
        bound = math.sqrt(
            (ln_parent / self.visit_count) * min(0.25, v_term)
        )
        return self.q_value + exploration_constant * bound

    # -- expansion ----------------------------------------------------------

    def expand(
        self,
        logits: np.ndarray,
        top_k: int = 50,
    ) -> List["MCTSNode"]:
        """Expand this node by creating children from the top-*k* logits.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.
        top_k : int
            Number of highest-probability tokens to create children for.

        Returns
        -------
        list of MCTSNode
            Newly created child nodes.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        vocab_size = logits.shape[0]

        effective_k = min(top_k, vocab_size)

        # Get top-k token indices
        if effective_k < vocab_size:
            top_indices = np.argpartition(logits, -effective_k)[-effective_k:]
        else:
            top_indices = np.arange(vocab_size)

        # Compute prior probabilities from logits restricted to top-k
        top_logits = logits[top_indices]
        priors = _stable_softmax(top_logits)

        new_children: List[MCTSNode] = []
        for idx, (token_idx, prior) in enumerate(zip(top_indices, priors)):
            token_id = int(token_idx)
            if token_id in self.children:
                continue
            child = MCTSNode(
                token_id=token_id,
                parent=self,
                prior_prob=float(prior),
            )
            self.children[token_id] = child
            new_children.append(child)

        self._expanded = True
        return new_children

    def expand_single(
        self,
        token_id: int,
        prior_prob: float,
    ) -> "MCTSNode":
        """Expand a single child token.

        Parameters
        ----------
        token_id : int
            Token vocabulary index.
        prior_prob : float
            Prior probability for the token.

        Returns
        -------
        MCTSNode
            The new (or existing) child node.
        """
        if token_id in self.children:
            return self.children[token_id]
        child = MCTSNode(
            token_id=token_id,
            parent=self,
            prior_prob=prior_prob,
        )
        self.children[token_id] = child
        self._expanded = True
        return child

    # -- tree queries -------------------------------------------------------

    def is_leaf(self) -> bool:
        """True if the node has no children."""
        return len(self.children) == 0

    def is_fully_expanded(self, max_children: int = 50) -> bool:
        """True if the node has reached its maximum number of children."""
        return self._expanded and len(self.children) >= max_children

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        """Return the child with the highest UCB1 score.

        Parameters
        ----------
        exploration_constant : float
            UCB1 exploration constant.

        Returns
        -------
        MCTSNode
            The child maximising UCB1.

        Raises
        ------
        ValueError
            If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select best_child from a node with no children.")

        best: Optional[MCTSNode] = None
        best_score = _NEG_INF

        parent_visits = self.visit_count
        for child in self.children.values():
            score = child.ucb_score(parent_visits, exploration_constant)
            if score > best_score:
                best_score = score
                best = child

        assert best is not None
        return best

    def best_child_puct(self, c_puct: float) -> "MCTSNode":
        """Return the child with the highest PUCT score.

        Parameters
        ----------
        c_puct : float
            PUCT exploration constant.

        Returns
        -------
        MCTSNode
            The child maximising PUCT.

        Raises
        ------
        ValueError
            If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select best_child_puct from a node with no children.")

        best: Optional[MCTSNode] = None
        best_score = _NEG_INF

        parent_visits = self.visit_count
        for child in self.children.values():
            score = child.puct_score(parent_visits, c_puct)
            if score > best_score:
                best_score = score
                best = child

        assert best is not None
        return best

    def most_visited_child(self) -> "MCTSNode":
        """Return the child with the highest visit count.

        Returns
        -------
        MCTSNode
            The most-visited child node.

        Raises
        ------
        ValueError
            If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot get most_visited_child from a node with no children.")

        best: Optional[MCTSNode] = None
        best_visits = -1

        for child in self.children.values():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best = child

        assert best is not None
        return best

    def child_with_highest_value(self) -> "MCTSNode":
        """Return the child with the highest Q-value."""
        if not self.children:
            raise ValueError("No children to select from.")

        best: Optional[MCTSNode] = None
        best_q = _NEG_INF

        for child in self.children.values():
            if child.visit_count > 0 and child.q_value > best_q:
                best_q = child.q_value
                best = child

        if best is None:
            # Fall back to any child if none have been visited
            best = next(iter(self.children.values()))

        return best

    def get_sequence(self) -> List[int]:
        """Trace back from this node to the root, returning the token sequence.

        Returns
        -------
        list of int
            Token IDs from root to this node (excluding the root's -1 token).
        """
        tokens: List[int] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            if node.token_id >= 0:
                tokens.append(node.token_id)
            node = node.parent
        tokens.reverse()
        return tokens

    def get_ancestors(self) -> List["MCTSNode"]:
        """Return the list of ancestor nodes from root to this node."""
        ancestors: List[MCTSNode] = []
        node: Optional[MCTSNode] = self
        while node is not None:
            ancestors.append(node)
            node = node.parent
        ancestors.reverse()
        return ancestors

    def subtree_size(self) -> int:
        """Return the total number of nodes in the subtree rooted here."""
        count = 1
        stack: List[MCTSNode] = [self]
        while stack:
            current = stack.pop()
            for child in current.children.values():
                count += 1
                stack.append(child)
        return count

    def subtree_depth(self) -> int:
        """Return the maximum depth of the subtree rooted here."""
        if not self.children:
            return 0
        max_child_depth = 0
        stack: List[Tuple[MCTSNode, int]] = [(self, 0)]
        while stack:
            node, d = stack.pop()
            if not node.children:
                max_child_depth = max(max_child_depth, d)
            else:
                for child in node.children.values():
                    stack.append((child, d + 1))
        return max_child_depth

    def leaf_nodes(self) -> List["MCTSNode"]:
        """Collect all leaf nodes in the subtree."""
        leaves: List[MCTSNode] = []
        stack: List[MCTSNode] = [self]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def sorted_children(self, by: str = "visits") -> List["MCTSNode"]:
        """Return children sorted by a criterion.

        Parameters
        ----------
        by : str
            One of ``'visits'``, ``'value'``, ``'prior'``, ``'ucb'``.
        """
        children_list = list(self.children.values())
        if by == "visits":
            children_list.sort(key=lambda c: c.visit_count, reverse=True)
        elif by == "value":
            children_list.sort(key=lambda c: c.q_value, reverse=True)
        elif by == "prior":
            children_list.sort(key=lambda c: c.prior_probability, reverse=True)
        elif by == "ucb":
            pv = self.visit_count
            children_list.sort(
                key=lambda c: c.ucb_score(pv, 1.414), reverse=True
            )
        else:
            raise ValueError(f"Unknown sort criterion: {by}")
        return children_list

    def child_visit_distribution(self) -> Dict[int, int]:
        """Return a mapping of token_id -> visit_count for all children."""
        return {
            token_id: child.visit_count
            for token_id, child in self.children.items()
        }

    def child_value_distribution(self) -> Dict[int, float]:
        """Return a mapping of token_id -> q_value for all children."""
        return {
            token_id: child.q_value
            for token_id, child in self.children.items()
        }

    def child_prior_distribution(self) -> Dict[int, float]:
        """Return a mapping of token_id -> prior_probability for all children."""
        return {
            token_id: child.prior_probability
            for token_id, child in self.children.items()
        }

    def __repr__(self) -> str:
        return (
            f"MCTSNode(token={self.token_id}, depth={self.depth}, "
            f"visits={self.visit_count}, Q={self.q_value:.4f}, "
            f"prior={self.prior_probability:.4f}, "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# =========================================================================
# RolloutPolicy — strategy classes for the simulation phase
# =========================================================================


class RolloutPolicy(abc.ABC):
    """Abstract base class for MCTS rollout (simulation) policies."""

    @abc.abstractmethod
    def select_token(self, logits: np.ndarray) -> int:
        """Choose a token given raw logits.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.

        Returns
        -------
        int
            Selected token index.
        """
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this policy."""
        ...


class RandomRolloutPolicy(RolloutPolicy):
    """Uniform random sampling from the full vocabulary."""

    def select_token(self, logits: np.ndarray) -> int:
        probs = _stable_softmax(logits)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            return int(np.argmax(logits))
        probs /= total
        return int(np.random.choice(len(probs), p=probs))

    def name(self) -> str:
        return "random"


class GreedyRolloutPolicy(RolloutPolicy):
    """Always select the highest-probability token."""

    def select_token(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits))

    def name(self) -> str:
        return "greedy"


class NucleusRolloutPolicy(RolloutPolicy):
    """Top-p (nucleus) sampling rollout policy."""

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0) -> None:
        self.top_p = top_p
        self.temperature = temperature

    def select_token(self, logits: np.ndarray) -> int:
        return sample_token(
            logits, temperature=self.temperature, top_p=self.top_p
        )

    def name(self) -> str:
        return f"nucleus(p={self.top_p})"


class TemperatureRolloutPolicy(RolloutPolicy):
    """Temperature-scaled sampling rollout policy."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def select_token(self, logits: np.ndarray) -> int:
        return sample_token(logits, temperature=self.temperature)

    def name(self) -> str:
        return f"temperature(T={self.temperature})"


def _create_rollout_policy(config: MCTSConfig) -> RolloutPolicy:
    """Factory function: create a rollout policy from config."""
    policy_name = config.rollout_policy
    if policy_name == "random":
        return RandomRolloutPolicy()
    elif policy_name == "greedy":
        return GreedyRolloutPolicy()
    elif policy_name == "nucleus":
        return NucleusRolloutPolicy(
            top_p=config.rollout_top_p,
            temperature=config.rollout_temperature,
        )
    elif policy_name == "temperature":
        return TemperatureRolloutPolicy(temperature=config.rollout_temperature)
    else:
        logger.warning(
            "Unknown rollout policy '%s', falling back to random.", policy_name
        )
        return RandomRolloutPolicy()


# =========================================================================
# ValueFunction — pluggable value estimators
# =========================================================================


class ValueFunction(abc.ABC):
    """Abstract base class for sequence value estimation."""

    @abc.abstractmethod
    def evaluate(
        self,
        tokens: List[int],
        logit_source: LogitSource,
        **kwargs: Any,
    ) -> float:
        """Evaluate the quality of a (partial) token sequence.

        Parameters
        ----------
        tokens : list of int
            The token sequence to evaluate.
        logit_source : LogitSource
            Source of next-token logits.

        Returns
        -------
        float
            A scalar value (higher is better).
        """
        ...


class PerplexityValueFunction(ValueFunction):
    """Value based on negative perplexity (lower perplexity → higher value)."""

    def evaluate(
        self,
        tokens: List[int],
        logit_source: LogitSource,
        **kwargs: Any,
    ) -> float:
        if len(tokens) < 2:
            return 0.0

        total_log_prob = 0.0
        count = 0

        for i in range(len(tokens) - 1):
            prefix = tokens[: i + 1]
            logits = logit_source([prefix])
            if logits.ndim > 1:
                logits = logits[0]
            log_probs = _log_softmax(logits)
            next_token = tokens[i + 1]
            if next_token < len(log_probs):
                total_log_prob += log_probs[next_token]
                count += 1

        if count == 0:
            return 0.0

        avg_log_prob = total_log_prob / count
        # Negative perplexity as value (higher is better)
        # perplexity = exp(-avg_log_prob), so value = avg_log_prob
        return float(avg_log_prob)


class LengthNormalizedValueFunction(ValueFunction):
    """Log-probability normalised by sequence length with a penalty term."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha

    def evaluate(
        self,
        tokens: List[int],
        logit_source: LogitSource,
        **kwargs: Any,
    ) -> float:
        if len(tokens) < 2:
            return 0.0

        total_log_prob = 0.0
        count = 0

        for i in range(len(tokens) - 1):
            prefix = tokens[: i + 1]
            logits = logit_source([prefix])
            if logits.ndim > 1:
                logits = logits[0]
            log_probs = _log_softmax(logits)
            next_token = tokens[i + 1]
            if next_token < len(log_probs):
                total_log_prob += log_probs[next_token]
                count += 1

        if count == 0:
            return 0.0

        # Length penalty: ((5 + length) / 6) ^ alpha  (Wu et al., 2016)
        length_penalty = ((5.0 + count) / 6.0) ** self.alpha
        return float(total_log_prob / length_penalty)


class DiversityAwareValueFunction(ValueFunction):
    """Value that combines log-probability with a diversity bonus.

    The diversity bonus is computed relative to a set of existing sequences
    passed via kwargs (key ``existing_sequences``).
    """

    def __init__(
        self,
        diversity_weight: float = 0.3,
        ngram_size: int = 3,
        alpha: float = 0.6,
    ) -> None:
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self.alpha = alpha

    def evaluate(
        self,
        tokens: List[int],
        logit_source: LogitSource,
        **kwargs: Any,
    ) -> float:
        existing_sequences: List[List[int]] = kwargs.get("existing_sequences", [])

        # Base value: length-normalized log-prob
        base_value = self._compute_base_value(tokens, logit_source)

        # Diversity bonus
        diversity_bonus = 0.0
        if existing_sequences:
            diversity_bonus = self._compute_diversity_bonus(
                tokens, existing_sequences
            )

        return base_value + self.diversity_weight * diversity_bonus

    def _compute_base_value(
        self, tokens: List[int], logit_source: LogitSource
    ) -> float:
        if len(tokens) < 2:
            return 0.0

        total_log_prob = 0.0
        count = 0
        for i in range(len(tokens) - 1):
            prefix = tokens[: i + 1]
            logits = logit_source([prefix])
            if logits.ndim > 1:
                logits = logits[0]
            log_probs = _log_softmax(logits)
            next_token = tokens[i + 1]
            if next_token < len(log_probs):
                total_log_prob += log_probs[next_token]
                count += 1

        if count == 0:
            return 0.0

        length_penalty = ((5.0 + count) / 6.0) ** self.alpha
        return float(total_log_prob / length_penalty)

    def _compute_diversity_bonus(
        self,
        candidate: List[int],
        existing: List[List[int]],
    ) -> float:
        """N-gram overlap-based diversity bonus (higher = more diverse)."""
        if not existing or len(candidate) < self.ngram_size:
            return 0.0

        candidate_ngrams = self._extract_ngrams(candidate, self.ngram_size)
        if not candidate_ngrams:
            return 0.0

        total_jaccard_distance = 0.0
        for seq in existing:
            seq_ngrams = self._extract_ngrams(seq, self.ngram_size)
            if not seq_ngrams:
                total_jaccard_distance += 1.0
                continue
            intersection = candidate_ngrams & seq_ngrams
            union = candidate_ngrams | seq_ngrams
            if len(union) == 0:
                total_jaccard_distance += 1.0
            else:
                jaccard_sim = len(intersection) / len(union)
                total_jaccard_distance += 1.0 - jaccard_sim

        return total_jaccard_distance / len(existing)

    @staticmethod
    def _extract_ngrams(tokens: List[int], n: int) -> Set[Tuple[int, ...]]:
        """Extract n-grams from a token list."""
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def _create_value_function(config: MCTSConfig) -> ValueFunction:
    """Factory function: create a value function from config."""
    vf_name = config.value_function
    if vf_name == "perplexity":
        return PerplexityValueFunction()
    elif vf_name == "length_normalized":
        return LengthNormalizedValueFunction(alpha=config.length_penalty_alpha)
    elif vf_name == "diversity_aware":
        return DiversityAwareValueFunction(
            diversity_weight=config.diversity_weight,
            ngram_size=config.diversity_ngram_size,
            alpha=config.length_penalty_alpha,
        )
    else:
        logger.warning(
            "Unknown value function '%s', falling back to perplexity.", vf_name
        )
        return PerplexityValueFunction()


# =========================================================================
# SequenceExtractor — diverse sequence extraction from the tree
# =========================================================================


class SequenceExtractor:
    """Extract diverse sequences from an MCTS tree.

    Uses a combination of visit-count ranking and diversity filtering
    to select sequences that are both high-quality and diverse.
    """

    def __init__(
        self,
        diversity_weight: float = 0.3,
        ngram_size: int = 3,
        temperature: float = 1.0,
    ) -> None:
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self.temperature = temperature

    def extract_top_k(
        self,
        root: MCTSNode,
        k: int,
        min_length: int = 1,
        eos_token_id: Optional[int] = None,
    ) -> List[Tuple[List[int], float]]:
        """Extract the top-*k* diverse sequences from the tree.

        Parameters
        ----------
        root : MCTSNode
            Root of the MCTS tree.
        k : int
            Number of sequences to extract.
        min_length : int
            Minimum sequence length.
        eos_token_id : int, optional
            EOS token id (sequences ending in EOS are preferred).

        Returns
        -------
        list of (sequence, score) tuples
        """
        # Collect all candidate paths
        candidates = self._collect_candidates(root, min_length, eos_token_id)

        if not candidates:
            # Fallback: extract the single best path by visit count
            best_seq = self._extract_greedy_path(root)
            if best_seq:
                return [(best_seq, 0.0)]
            return []

        # Sort by score (visit-weighted Q-value)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Diversity-aware selection
        selected: List[Tuple[List[int], float]] = []
        used_ngrams: Set[Tuple[int, ...]] = set()

        for seq, score in candidates:
            if len(selected) >= k:
                break

            # Compute diversity penalty
            seq_ngrams = self._extract_ngrams(seq, self.ngram_size)
            if selected and seq_ngrams:
                overlap = len(seq_ngrams & used_ngrams)
                total = max(len(seq_ngrams), 1)
                diversity_penalty = self.diversity_weight * (overlap / total)
                adjusted_score = score - diversity_penalty
            else:
                adjusted_score = score

            selected.append((seq, adjusted_score))
            if seq_ngrams:
                used_ngrams.update(seq_ngrams)

        # Re-sort selected by adjusted score
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected

    def _collect_candidates(
        self,
        root: MCTSNode,
        min_length: int,
        eos_token_id: Optional[int],
    ) -> List[Tuple[List[int], float]]:
        """DFS to collect all sufficiently long paths from root."""
        candidates: List[Tuple[List[int], float]] = []
        stack: List[MCTSNode] = [root]

        while stack:
            node = stack.pop()

            is_candidate = False
            if node.is_terminal and node.depth >= min_length:
                is_candidate = True
            elif node.is_leaf() and node.depth >= min_length and node.visit_count > 0:
                is_candidate = True
            elif (
                eos_token_id is not None
                and node.token_id == eos_token_id
                and node.depth >= min_length
            ):
                is_candidate = True

            if is_candidate:
                seq = node.get_sequence()
                score = self._path_score(node)
                candidates.append((seq, score))

            for child in node.children.values():
                if child.visit_count > 0:
                    stack.append(child)

        return candidates

    def _extract_greedy_path(self, root: MCTSNode) -> List[int]:
        """Follow the most-visited children from root to a leaf."""
        path: List[int] = []
        node = root
        while node.children:
            node = node.most_visited_child()
            if node.token_id >= 0:
                path.append(node.token_id)
        return path

    def _path_score(self, node: MCTSNode) -> float:
        """Score a path ending at this node using visit count and Q-value."""
        if node.visit_count == 0:
            return _NEG_INF
        log_visits = math.log(max(node.visit_count, 1))
        return node.q_value + 0.1 * log_visits

    @staticmethod
    def _extract_ngrams(tokens: List[int], n: int) -> Set[Tuple[int, ...]]:
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


# =========================================================================
# MCTSTree — full search tree manager
# =========================================================================


class MCTSTree:
    """Manages the MCTS search tree and implements the four MCTS phases.

    Parameters
    ----------
    root_tokens : list of int
        Token IDs of the prompt (used as context for the root node).
    config : MCTSConfig
        MCTS configuration.
    """

    def __init__(self, root_tokens: List[int], config: MCTSConfig) -> None:
        self.root = MCTSNode(token_id=-1, parent=None, prior_prob=1.0)
        self.root_tokens = list(root_tokens)
        self.config = config
        self._rollout_policy = _create_rollout_policy(config)
        self._value_fn = _create_value_function(config)
        self._extractor = SequenceExtractor(
            diversity_weight=config.diversity_weight,
            ngram_size=config.diversity_ngram_size,
            temperature=config.temperature,
        )
        self._total_simulations: int = 0
        self._simulation_times: List[float] = []
        self._existing_sequences: List[List[int]] = []

    # -- Phase 1: Selection -------------------------------------------------

    def select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: descend the tree by UCB / PUCT until a leaf.

        Traverses from *node* downward, at each internal node choosing the
        child with the highest UCB1 (or PUCT) score, until a leaf or
        unexpanded node is reached.

        Parameters
        ----------
        node : MCTSNode
            Starting node (usually the root).

        Returns
        -------
        MCTSNode
            The selected leaf/unexpanded node.
        """
        current = node
        c_explore = self.config.exploration_constant

        while not current.is_leaf() and not current.is_terminal:
            # Check progressive widening
            if self.config.progressive_widening:
                if self._progressive_widen_check(current):
                    return current

            # If not fully expanded, return for expansion
            if not current._expanded:
                return current

            # Select best child by PUCT (uses prior) or UCB1
            current = current.best_child_puct(c_explore)

        return current

    # -- Phase 2: Expansion -------------------------------------------------

    def expand(
        self,
        node: MCTSNode,
        logit_source: LogitSource,
    ) -> MCTSNode:
        """Expansion phase: add children to a leaf node.

        Uses the logit source to get next-token probabilities and creates
        child nodes for the top-k tokens.

        Parameters
        ----------
        node : MCTSNode
            The leaf node to expand.
        logit_source : LogitSource
            Source of next-token logits.

        Returns
        -------
        MCTSNode
            A randomly chosen new child (or the node itself if terminal).
        """
        if node.is_terminal:
            return node

        if node.depth >= self.config.max_tree_depth:
            node.is_terminal = True
            return node

        # Build prefix from root tokens + path from root to this node
        prefix = self.root_tokens + node.get_sequence()

        # Get logits from the model
        logits = logit_source([prefix])
        if logits.ndim > 1:
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64).ravel()

        # Determine expansion width
        top_k = self.config.top_k_expansion
        if self.config.progressive_widening:
            pw_k = max(1, int(math.ceil(
                (node.visit_count + 1) ** self.config.pw_alpha
            )))
            top_k = min(pw_k, top_k)
        top_k = min(top_k, self.config.max_children - len(node.children))

        if top_k <= 0:
            return node

        # Mark EOS children as terminal
        new_children = node.expand(logits, top_k=top_k)
        eos_id = self.config.eos_token_id
        if eos_id is not None:
            for child in new_children:
                if child.token_id == eos_id:
                    child.is_terminal = True

        if not new_children and node.children:
            # All top-k already expanded; pick least-visited existing child
            children_list = list(node.children.values())
            children_list.sort(key=lambda c: c.visit_count)
            return children_list[0]
        elif new_children:
            # Select a child proportional to prior probability
            priors = np.array([c.prior_probability for c in new_children])
            priors = np.maximum(priors, _EPSILON)
            priors /= priors.sum()
            idx = int(np.random.choice(len(new_children), p=priors))
            return new_children[idx]

        return node

    # -- Phase 3: Simulation (Rollout) --------------------------------------

    def simulate(
        self,
        node: MCTSNode,
        logit_source: LogitSource,
    ) -> float:
        """Simulation phase: perform a rollout from *node* and return a value.

        Generates tokens using the configured rollout policy up to
        ``rollout_depth``, then evaluates the resulting sequence.

        Parameters
        ----------
        node : MCTSNode
            The node from which to start the rollout.
        logit_source : LogitSource
            Source of next-token logits.

        Returns
        -------
        float
            Estimated value of the sequence (higher is better).
        """
        if node.is_terminal:
            # Evaluate the sequence ending at this terminal node
            full_seq = self.root_tokens + node.get_sequence()
            return self._evaluate_terminal(full_seq, logit_source)

        # Build current prefix
        current_tokens = self.root_tokens + node.get_sequence()
        rollout_tokens = list(current_tokens)
        rollout_depth = self.config.rollout_depth
        eos_id = self.config.eos_token_id

        for step in range(rollout_depth):
            # Get logits
            logits = logit_source([rollout_tokens])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            # Select next token using rollout policy
            next_token = self._rollout_policy.select_token(logits)
            rollout_tokens.append(next_token)

            # Check EOS
            if eos_id is not None and next_token == eos_id:
                break

        # Evaluate the complete rollout sequence
        return self._evaluate_rollout(rollout_tokens, logit_source)

    def _evaluate_terminal(
        self,
        tokens: List[int],
        logit_source: LogitSource,
    ) -> float:
        """Evaluate a terminal sequence."""
        return self._value_fn.evaluate(
            tokens,
            logit_source,
            existing_sequences=self._existing_sequences,
        )

    def _evaluate_rollout(
        self,
        tokens: List[int],
        logit_source: LogitSource,
    ) -> float:
        """Evaluate a rollout sequence with discount factor applied."""
        base_value = self._value_fn.evaluate(
            tokens,
            logit_source,
            existing_sequences=self._existing_sequences,
        )
        # Apply discount factor based on rollout length beyond the prefix
        rollout_len = len(tokens) - len(self.root_tokens)
        discounted = base_value * (self.config.discount_factor ** max(rollout_len, 0))
        return discounted

    # -- Phase 4: Backpropagation -------------------------------------------

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation: update visit counts and values up to the root.

        Parameters
        ----------
        node : MCTSNode
            The leaf node where the rollout started.
        value : float
            The value returned by the simulation phase.
        """
        current: Optional[MCTSNode] = node
        depth_from_leaf = 0

        while current is not None:
            current.visit_count += 1
            discounted_value = value * (
                self.config.discount_factor ** depth_from_leaf
            )
            current.total_value += discounted_value
            current._sum_squared_value += discounted_value * discounted_value
            current = current.parent
            depth_from_leaf += 1

    # -- Virtual loss (parallel MCTS) ---------------------------------------

    def _apply_virtual_loss(self, node: MCTSNode) -> None:
        """Apply virtual loss along the path from *node* to root.

        Virtual loss discourages other threads from exploring the same path
        concurrently, encouraging diversity in parallel MCTS.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.virtual_loss_count += 1
            current = current.parent

    def _remove_virtual_loss(self, node: MCTSNode) -> None:
        """Remove virtual loss along the path from *node* to root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.virtual_loss_count = max(0, current.virtual_loss_count - 1)
            current = current.parent

    # -- Progressive widening -----------------------------------------------

    def _progressive_widen_check(self, node: MCTSNode) -> bool:
        """Check if a node should be further widened.

        Progressive widening allows the number of children to grow with
        the number of visits: |children| ≤ ceil(N^alpha).

        Returns True if expansion is allowed (node should be expanded further).
        """
        max_allowed = max(1, int(math.ceil(
            (node.visit_count + 1) ** self.config.pw_alpha
        )))
        return len(node.children) < max_allowed

    # -- Sequence extraction ------------------------------------------------

    def best_sequence(self) -> List[int]:
        """Extract the single best sequence by following most-visited children."""
        tokens: List[int] = []
        node = self.root
        while node.children:
            node = node.most_visited_child()
            if node.token_id >= 0:
                tokens.append(node.token_id)
        return tokens

    def top_k_sequences(self, k: int) -> List[List[int]]:
        """Extract *k* diverse sequences from the tree.

        Parameters
        ----------
        k : int
            Number of sequences to extract.

        Returns
        -------
        list of list of int
            Token sequences (without prompt tokens).
        """
        results = self._extractor.extract_top_k(
            self.root,
            k=k,
            min_length=max(1, self.config.min_new_tokens),
            eos_token_id=self.config.eos_token_id,
        )
        if not results:
            # Fallback: greedy path extraction for k paths
            return self._extract_diverse_greedy_paths(k)
        return [seq for seq, _score in results]

    def _extract_diverse_greedy_paths(self, k: int) -> List[List[int]]:
        """Fallback extraction: greedily follow top paths with diversity."""
        paths: List[List[int]] = []
        used_first_tokens: Set[int] = set()

        # Collect candidate first-step children sorted by visits
        if not self.root.children:
            return paths

        sorted_starts = sorted(
            self.root.children.values(),
            key=lambda c: c.visit_count,
            reverse=True,
        )

        for start_child in sorted_starts:
            if len(paths) >= k:
                break
            if start_child.token_id in used_first_tokens and len(paths) > 0:
                continue

            # Follow greedy path from this child
            path = [start_child.token_id]
            node = start_child
            while node.children:
                node = node.most_visited_child()
                if node.token_id >= 0:
                    path.append(node.token_id)
            paths.append(path)
            used_first_tokens.add(start_child.token_id)

        return paths

    # -- Tree statistics ----------------------------------------------------

    def tree_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the current search tree.

        Returns
        -------
        dict
            Statistics including node count, depth, width profile,
            visit distribution summary, and branching factor.
        """
        total_nodes = 0
        total_visits = 0
        max_depth = 0
        depth_counts: DefaultDict[int, int] = collections.defaultdict(int)
        depth_visits: DefaultDict[int, int] = collections.defaultdict(int)
        terminal_count = 0
        leaf_count = 0
        branching_factors: List[int] = []

        stack: List[MCTSNode] = [self.root]
        while stack:
            node = stack.pop()
            total_nodes += 1
            total_visits += node.visit_count
            max_depth = max(max_depth, node.depth)
            depth_counts[node.depth] += 1
            depth_visits[node.depth] += node.visit_count

            if node.is_terminal:
                terminal_count += 1
            if node.is_leaf():
                leaf_count += 1
            else:
                branching_factors.append(len(node.children))

            stack.extend(node.children.values())

        avg_branching = (
            float(np.mean(branching_factors)) if branching_factors else 0.0
        )
        max_branching = max(branching_factors) if branching_factors else 0

        # Visit distribution at root children
        root_child_visits = sorted(
            [c.visit_count for c in self.root.children.values()],
            reverse=True,
        )
        root_child_values = sorted(
            [c.q_value for c in self.root.children.values()],
            reverse=True,
        )

        # Width profile (nodes at each depth)
        width_profile = dict(sorted(depth_counts.items()))

        return {
            "total_nodes": total_nodes,
            "total_visits": total_visits,
            "max_depth": max_depth,
            "terminal_nodes": terminal_count,
            "leaf_nodes": leaf_count,
            "avg_branching_factor": round(avg_branching, 2),
            "max_branching_factor": max_branching,
            "width_profile": width_profile,
            "depth_visit_profile": dict(sorted(depth_visits.items())),
            "root_children_count": len(self.root.children),
            "root_child_visits_top5": root_child_visits[:5],
            "root_child_values_top5": [round(v, 4) for v in root_child_values[:5]],
            "total_simulations": self._total_simulations,
            "mean_simulation_time_ms": (
                round(float(np.mean(self._simulation_times)) * 1000, 2)
                if self._simulation_times
                else 0.0
            ),
        }

    def prune(self, min_visits: int = 1) -> None:
        """Prune nodes with fewer than *min_visits* visits.

        Parameters
        ----------
        min_visits : int
            Minimum visit count to keep a node.
        """
        self._prune_subtree(self.root, min_visits)

    def _prune_subtree(self, node: MCTSNode, min_visits: int) -> None:
        """Recursively prune children with insufficient visits."""
        tokens_to_remove: List[int] = []
        for token_id, child in node.children.items():
            if child.visit_count < min_visits:
                tokens_to_remove.append(token_id)
            else:
                self._prune_subtree(child, min_visits)

        for token_id in tokens_to_remove:
            del node.children[token_id]

    def node_count(self) -> int:
        """Count total nodes in the tree."""
        return self.root.subtree_size()

    def reset(self) -> None:
        """Reset the tree to a fresh root node."""
        self.root = MCTSNode(token_id=-1, parent=None, prior_prob=1.0)
        self._total_simulations = 0
        self._simulation_times.clear()
        self._existing_sequences.clear()

    def add_existing_sequence(self, seq: List[int]) -> None:
        """Register a previously extracted sequence for diversity-aware scoring."""
        self._existing_sequences.append(list(seq))


# =========================================================================
# MCTSDecoding — main algorithm
# =========================================================================


class MCTSDecoding(DecodingAlgorithm):
    """Monte Carlo Tree Search decoding for diverse text generation.

    Builds a search tree over the token vocabulary using MCTS simulations,
    then extracts diverse high-quality sequences from the tree.

    The algorithm proceeds as follows:
    1. Initialise a search tree rooted at the end of the prompt.
    2. Run ``n_simulations`` MCTS iterations (select → expand → simulate →
       backpropagate).
    3. Extract ``n_sequences`` diverse sequences from the tree.
    4. Optionally reuse the tree across sequence extractions for efficiency.

    Parameters
    ----------
    config : MCTSConfig
        Full configuration for the MCTS decoding run.
    """

    def __init__(self, config: MCTSConfig) -> None:
        super().__init__(config)
        self.mcts_config: MCTSConfig = config
        self._tree: Optional[MCTSTree] = None
        self._rollout_cache: Dict[str, float] = {}
        self._total_rollouts: int = 0
        self._cache_hits: int = 0
        self._generation_start_time: float = 0.0

    # -- Public API ---------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: Optional[int] = None,
    ) -> List[TokenSequence]:
        """Generate diverse sequences using MCTS.

        Parameters
        ----------
        logit_source : LogitSource
            Callable ``(List[List[int]]) -> np.ndarray`` returning logits.
        prompt_ids : list of int
            Token IDs of the prompt.
        n_sequences : int, optional
            Number of sequences to generate.  Defaults to ``config.n_sequences``.

        Returns
        -------
        list of TokenSequence
            Generated token sequences (without prompt).
        """
        self._generation_start_time = time.monotonic()

        if self.mcts_config.seed is not None:
            np.random.seed(self.mcts_config.seed)

        n_seq = n_sequences or self.mcts_config.n_sequences
        logger.info(
            "Starting MCTS decoding: n_simulations=%d, n_sequences=%d, "
            "rollout_policy=%s, value_fn=%s",
            self.mcts_config.n_simulations,
            n_seq,
            self.mcts_config.rollout_policy,
            self.mcts_config.value_function,
        )

        # Build or reuse the search tree
        tree = self._run_mcts(logit_source, prompt_ids)

        # Extract diverse sequences
        sequences = self._extract_diverse_sequences(tree, n_seq)

        elapsed = time.monotonic() - self._generation_start_time
        logger.info(
            "MCTS decoding complete: %d sequences in %.2fs, "
            "%d total rollouts, %d cache hits",
            len(sequences),
            elapsed,
            self._total_rollouts,
            self._cache_hits,
        )

        return sequences

    # -- Core MCTS loop -----------------------------------------------------

    def _run_mcts(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> MCTSTree:
        """Run the full MCTS search and return the resulting tree.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int

        Returns
        -------
        MCTSTree
        """
        # Create or reuse tree
        if self.mcts_config.reuse_tree and self._tree is not None:
            tree = self._tree
        else:
            tree = MCTSTree(prompt_ids, self.mcts_config)
            self._tree = tree

        n_sims = self.mcts_config.n_simulations

        for sim_idx in range(n_sims):
            sim_start = time.monotonic()

            # Phase 1: Selection
            selected = tree.select(tree.root)

            # Apply virtual loss during simulation
            if self.mcts_config.virtual_loss > 0:
                tree._apply_virtual_loss(selected)

            try:
                # Phase 2: Expansion
                if not selected.is_terminal and not selected._expanded:
                    expanded = tree.expand(selected, logit_source)
                else:
                    expanded = selected

                # Phase 3: Simulation (Rollout)
                value = self._run_simulation(
                    expanded, logit_source, tree
                )
                self._total_rollouts += 1

                # Phase 4: Backpropagation
                tree.backpropagate(expanded, value)

            finally:
                # Remove virtual loss
                if self.mcts_config.virtual_loss > 0:
                    tree._remove_virtual_loss(selected)

            sim_time = time.monotonic() - sim_start
            tree._simulation_times.append(sim_time)
            tree._total_simulations += 1

            # Periodic logging
            if (sim_idx + 1) % max(1, n_sims // 10) == 0:
                stats = tree.tree_statistics()
                self._log_search_progress(sim_idx + 1, stats)

        return tree

    def _run_simulation(
        self,
        node: MCTSNode,
        logit_source: LogitSource,
        tree: MCTSTree,
    ) -> float:
        """Run a single simulation (rollout) with optional caching.

        Parameters
        ----------
        node : MCTSNode
            Node from which to start the rollout.
        logit_source : LogitSource
        tree : MCTSTree

        Returns
        -------
        float
            Estimated value.
        """
        # Check cache
        seq = node.get_sequence()
        cache_key = self._make_cache_key(seq)
        if cache_key in self._rollout_cache:
            self._cache_hits += 1
            return self._rollout_cache[cache_key]

        # Dispatch to appropriate rollout method
        policy = self.mcts_config.rollout_policy
        prefix = tree.root_tokens + seq

        if policy == "random":
            value = self._rollout_random(
                logit_source, prefix, self.mcts_config.rollout_depth
            )
        elif policy == "greedy":
            value = self._rollout_greedy(
                logit_source, prefix, self.mcts_config.rollout_depth
            )
        elif policy == "nucleus":
            value = self._rollout_nucleus(
                logit_source,
                prefix,
                self.mcts_config.rollout_depth,
                p=self.mcts_config.rollout_top_p,
            )
        elif policy == "temperature":
            value = self._rollout_temperature(
                logit_source,
                prefix,
                self.mcts_config.rollout_depth,
                temperature=self.mcts_config.rollout_temperature,
            )
        else:
            value = tree.simulate(node, logit_source)

        # Add diversity bonus if applicable
        if self.mcts_config.value_function == "diversity_aware":
            div_bonus = self._compute_diversity_bonus(seq, tree)
            value += self.mcts_config.diversity_weight * div_bonus

        # Cache the result
        self._rollout_cache[cache_key] = value

        return value

    # -- Rollout methods ----------------------------------------------------

    def _rollout_random(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        depth: int,
    ) -> float:
        """Random rollout: sample tokens uniformly from the softmax distribution.

        Parameters
        ----------
        logit_source : LogitSource
        prefix : list of int
            Current token prefix (prompt + tree path).
        depth : int
            Maximum rollout depth.

        Returns
        -------
        float
            Estimated value of the completed sequence.
        """
        tokens = list(prefix)
        eos_id = self.mcts_config.eos_token_id
        total_log_prob = 0.0
        n_generated = 0

        for _step in range(depth):
            logits = logit_source([tokens])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            probs = _stable_softmax(logits)
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total <= 0 or not np.isfinite(total):
                break
            probs /= total

            next_token = int(np.random.choice(len(probs), p=probs))
            log_prob = np.log(max(probs[next_token], _LOG_EPS))
            total_log_prob += log_prob
            tokens.append(next_token)
            n_generated += 1

            if eos_id is not None and next_token == eos_id:
                break

        return self._evaluate_sequence(tokens, total_log_prob, n_generated)

    def _rollout_greedy(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        depth: int,
    ) -> float:
        """Greedy rollout: always pick the highest-probability token.

        Parameters
        ----------
        logit_source : LogitSource
        prefix : list of int
        depth : int

        Returns
        -------
        float
        """
        tokens = list(prefix)
        eos_id = self.mcts_config.eos_token_id
        total_log_prob = 0.0
        n_generated = 0

        for _step in range(depth):
            logits = logit_source([tokens])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            log_probs = _log_softmax(logits)
            next_token = int(np.argmax(logits))
            total_log_prob += log_probs[next_token]
            tokens.append(next_token)
            n_generated += 1

            if eos_id is not None and next_token == eos_id:
                break

        return self._evaluate_sequence(tokens, total_log_prob, n_generated)

    def _rollout_nucleus(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        depth: int,
        p: float = 0.9,
    ) -> float:
        """Nucleus (top-p) rollout.

        Parameters
        ----------
        logit_source : LogitSource
        prefix : list of int
        depth : int
        p : float
            Nucleus probability threshold.

        Returns
        -------
        float
        """
        tokens = list(prefix)
        eos_id = self.mcts_config.eos_token_id
        total_log_prob = 0.0
        n_generated = 0

        for _step in range(depth):
            logits = logit_source([tokens])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            next_token = sample_token(
                logits,
                temperature=self.mcts_config.rollout_temperature,
                top_p=p,
            )
            log_probs = _log_softmax(logits)
            total_log_prob += log_probs[next_token]
            tokens.append(next_token)
            n_generated += 1

            if eos_id is not None and next_token == eos_id:
                break

        return self._evaluate_sequence(tokens, total_log_prob, n_generated)

    def _rollout_temperature(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        depth: int,
        temperature: float = 1.0,
    ) -> float:
        """Temperature-scaled sampling rollout.

        Parameters
        ----------
        logit_source : LogitSource
        prefix : list of int
        depth : int
        temperature : float

        Returns
        -------
        float
        """
        tokens = list(prefix)
        eos_id = self.mcts_config.eos_token_id
        total_log_prob = 0.0
        n_generated = 0

        for _step in range(depth):
            logits = logit_source([tokens])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            next_token = sample_token(logits, temperature=temperature)
            log_probs = _log_softmax(logits)
            total_log_prob += log_probs[next_token]
            tokens.append(next_token)
            n_generated += 1

            if eos_id is not None and next_token == eos_id:
                break

        return self._evaluate_sequence(tokens, total_log_prob, n_generated)

    # -- Value / scoring functions ------------------------------------------

    def _evaluate_sequence(
        self,
        tokens: List[int],
        total_log_prob: float = 0.0,
        n_generated: int = 0,
    ) -> float:
        """Evaluate a sequence using the configured value function.

        Parameters
        ----------
        tokens : list of int
            Full token sequence (prompt + generated).
        total_log_prob : float
            Accumulated log-probability of generated tokens.
        n_generated : int
            Number of generated tokens.

        Returns
        -------
        float
            Scalar value (higher is better).
        """
        vf_name = self.mcts_config.value_function

        if vf_name == "perplexity":
            if n_generated == 0:
                return 0.0
            avg_log_prob = total_log_prob / max(n_generated, 1)
            return float(avg_log_prob)

        elif vf_name == "length_normalized":
            return self._length_normalized_score(total_log_prob, n_generated)

        elif vf_name == "diversity_aware":
            base = self._length_normalized_score(total_log_prob, n_generated)
            return base

        else:
            if n_generated == 0:
                return 0.0
            return float(total_log_prob / max(n_generated, 1))

    def _diversity_aware_value(
        self,
        sequence: List[int],
        existing_sequences: List[List[int]],
    ) -> float:
        """Compute diversity-aware value combining quality and novelty.

        Parameters
        ----------
        sequence : list of int
            Candidate token sequence.
        existing_sequences : list of list of int
            Previously selected sequences.

        Returns
        -------
        float
            Combined quality + diversity score.
        """
        if not existing_sequences:
            return 0.0

        ngram_size = self.mcts_config.diversity_ngram_size
        candidate_ngrams = _extract_ngrams_set(sequence, ngram_size)
        if not candidate_ngrams:
            return 0.0

        total_distance = 0.0
        for existing_seq in existing_sequences:
            existing_ngrams = _extract_ngrams_set(existing_seq, ngram_size)
            if not existing_ngrams:
                total_distance += 1.0
                continue
            intersection = candidate_ngrams & existing_ngrams
            union = candidate_ngrams | existing_ngrams
            if len(union) == 0:
                total_distance += 1.0
            else:
                jaccard = len(intersection) / len(union)
                total_distance += 1.0 - jaccard

        return total_distance / len(existing_sequences)

    def _sequence_log_probability(
        self,
        logit_source: LogitSource,
        tokens: List[int],
    ) -> float:
        """Compute the total log-probability of a token sequence.

        Parameters
        ----------
        logit_source : LogitSource
        tokens : list of int
            Full token sequence.

        Returns
        -------
        float
            Sum of log-probabilities for each token given its prefix.
        """
        if len(tokens) < 2:
            return 0.0

        total_log_prob = 0.0
        for i in range(len(tokens) - 1):
            prefix = tokens[: i + 1]
            logits = logit_source([prefix])
            if logits.ndim > 1:
                logits = logits[0]
            log_probs = _log_softmax(np.asarray(logits, dtype=np.float64).ravel())
            next_token = tokens[i + 1]
            if next_token < len(log_probs):
                total_log_prob += log_probs[next_token]

        return float(total_log_prob)

    def _length_normalized_score(
        self,
        log_prob: float,
        length: int,
    ) -> float:
        """Compute length-normalised score (Wu et al., 2016).

        score = log_prob / lp(length)
        lp(l) = ((5 + l) / 6) ^ alpha

        Parameters
        ----------
        log_prob : float
            Total log-probability.
        length : int
            Sequence length (number of generated tokens).

        Returns
        -------
        float
        """
        if length <= 0:
            return 0.0
        alpha = self.mcts_config.length_penalty_alpha
        length_penalty = ((5.0 + length) / 6.0) ** alpha
        return float(log_prob / max(length_penalty, _EPSILON))

    def _compute_diversity_bonus(
        self,
        sequence: List[int],
        tree: MCTSTree,
    ) -> float:
        """Compute a diversity bonus for a sequence relative to the tree.

        Encourages exploration of under-visited regions of the tree by
        rewarding sequences that differ from the current most-visited path.

        Parameters
        ----------
        sequence : list of int
            Candidate token sequence (tree path, without prompt).
        tree : MCTSTree

        Returns
        -------
        float
            Non-negative diversity bonus.
        """
        if not tree._existing_sequences:
            return 0.0

        bonus = self._diversity_aware_value(
            sequence, tree._existing_sequences
        )
        return float(bonus * self.mcts_config.diversity_bonus_scale)

    # -- Sequence extraction ------------------------------------------------

    def _extract_diverse_sequences(
        self,
        tree: MCTSTree,
        n: int,
    ) -> List[TokenSequence]:
        """Extract *n* diverse sequences from the MCTS tree.

        Uses a greedy iterative approach: extract the best sequence, add it
        to the "existing" set (for diversity-aware scoring), optionally re-run
        simulations, and repeat.

        Parameters
        ----------
        tree : MCTSTree
        n : int
            Number of sequences to extract.

        Returns
        -------
        list of TokenSequence
        """
        if n <= 0:
            return []

        # Simple extraction from the tree
        raw_sequences = tree.top_k_sequences(n)

        if len(raw_sequences) >= n:
            result = raw_sequences[:n]
        else:
            result = list(raw_sequences)

        # If we don't have enough, try to extract more via beam-style paths
        if len(result) < n:
            additional = self._extract_additional_sequences(
                tree, n - len(result), result
            )
            result.extend(additional)

        # Register extracted sequences for future diversity scoring
        for seq in result:
            tree.add_existing_sequence(seq)

        # De-duplicate
        seen: Set[Tuple[int, ...]] = set()
        unique: List[TokenSequence] = []
        for seq in result:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique.append(seq)

        return unique[:n]

    def _extract_additional_sequences(
        self,
        tree: MCTSTree,
        n_needed: int,
        existing: List[List[int]],
    ) -> List[List[int]]:
        """Extract additional sequences by sampling from the tree with diversity.

        Parameters
        ----------
        tree : MCTSTree
        n_needed : int
        existing : list of list of int

        Returns
        -------
        list of list of int
        """
        additional: List[List[int]] = []

        if not tree.root.children:
            return additional

        # Use temperature-weighted visit counts to sample starting children
        children = list(tree.root.children.values())
        visit_counts = np.array(
            [max(c.visit_count, 1) for c in children], dtype=np.float64
        )

        temp = self.mcts_config.temperature
        if temp > 0:
            weights = np.power(visit_counts, 1.0 / temp)
        else:
            weights = np.zeros_like(visit_counts)
            weights[np.argmax(visit_counts)] = 1.0

        weights_sum = weights.sum()
        if weights_sum <= 0:
            return additional
        probs = weights / weights_sum

        existing_set = {tuple(s) for s in existing}

        attempts = 0
        max_attempts = n_needed * 10

        while len(additional) < n_needed and attempts < max_attempts:
            attempts += 1

            # Sample a starting child
            start_idx = int(np.random.choice(len(children), p=probs))
            start_child = children[start_idx]

            # Follow a stochastic path
            path = [start_child.token_id]
            node = start_child

            while node.children:
                child_list = list(node.children.values())
                child_visits = np.array(
                    [max(c.visit_count, 1) for c in child_list],
                    dtype=np.float64,
                )
                if temp > 0:
                    cw = np.power(child_visits, 1.0 / temp)
                else:
                    cw = np.zeros_like(child_visits)
                    cw[np.argmax(child_visits)] = 1.0

                cw_sum = cw.sum()
                if cw_sum <= 0:
                    break
                cp = cw / cw_sum

                chosen_idx = int(np.random.choice(len(child_list), p=cp))
                node = child_list[chosen_idx]
                if node.token_id >= 0:
                    path.append(node.token_id)

            key = tuple(path)
            if key not in existing_set and len(path) > 0:
                additional.append(path)
                existing_set.add(key)

        return additional

    # -- Batch decoding -----------------------------------------------------

    def decode_batch(
        self,
        logit_source: LogitSource,
        prompt_batch: List[List[int]],
        n_sequences: Optional[int] = None,
    ) -> List[List[TokenSequence]]:
        """Decode a batch of prompts.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_batch : list of list of int
            Batch of prompt token IDs.
        n_sequences : int, optional
            Number of sequences per prompt.

        Returns
        -------
        list of list of TokenSequence
        """
        results: List[List[TokenSequence]] = []
        n_seq = n_sequences or self.mcts_config.n_sequences

        for prompt_idx, prompt_ids in enumerate(prompt_batch):
            logger.info(
                "Decoding prompt %d/%d", prompt_idx + 1, len(prompt_batch)
            )
            # Reset tree for each prompt (unless reuse is enabled AND same prompt)
            if not self.mcts_config.reuse_tree:
                self._tree = None
                self._rollout_cache.clear()

            sequences = self.generate(logit_source, prompt_ids, n_sequences=n_seq)
            results.append(sequences)

        return results

    # -- Hyperparameter grid ------------------------------------------------

    def get_hyperparameter_grid(self) -> Dict[str, Any]:
        """Return a dictionary describing the hyperparameter search space.

        Returns
        -------
        dict
            Maps parameter names to lists/ranges of candidate values.
        """
        return {
            "n_simulations": {
                "type": "int",
                "values": [25, 50, 100, 200, 500],
                "description": "Number of MCTS simulations per step",
            },
            "exploration_constant": {
                "type": "float",
                "values": [0.5, 1.0, 1.414, 2.0, 3.0],
                "description": "UCB1 / PUCT exploration constant",
            },
            "rollout_depth": {
                "type": "int",
                "values": [5, 10, 20, 50],
                "description": "Maximum rollout depth",
            },
            "rollout_policy": {
                "type": "categorical",
                "values": ["random", "greedy", "nucleus", "temperature"],
                "description": "Policy for rollout token selection",
            },
            "rollout_temperature": {
                "type": "float",
                "values": [0.3, 0.5, 0.7, 1.0, 1.5],
                "description": "Temperature for temperature/nucleus rollout",
            },
            "discount_factor": {
                "type": "float",
                "values": [0.9, 0.95, 0.99, 1.0],
                "description": "Discount factor for backpropagation",
            },
            "max_children": {
                "type": "int",
                "values": [10, 25, 50, 100],
                "description": "Maximum children per node",
            },
            "top_k_expansion": {
                "type": "int",
                "values": [10, 25, 50, 100],
                "description": "Top-k tokens for expansion",
            },
            "value_function": {
                "type": "categorical",
                "values": ["perplexity", "length_normalized", "diversity_aware"],
                "description": "Sequence value estimation method",
            },
            "diversity_weight": {
                "type": "float",
                "values": [0.0, 0.1, 0.3, 0.5, 1.0],
                "description": "Weight for diversity in value function",
            },
            "progressive_widening": {
                "type": "bool",
                "values": [True, False],
                "description": "Enable progressive widening",
            },
            "pw_alpha": {
                "type": "float",
                "values": [0.25, 0.5, 0.75],
                "description": "Progressive widening exponent",
            },
            "virtual_loss": {
                "type": "float",
                "values": [0.0, 0.5, 1.0, 3.0],
                "description": "Virtual loss for parallel exploration",
            },
            "temperature": {
                "type": "float",
                "values": [0.5, 0.7, 1.0, 1.5],
                "description": "Temperature for sequence extraction",
            },
        }

    # -- Description --------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of this algorithm.

        Returns
        -------
        str
        """
        cfg = self.mcts_config
        lines = [
            "Monte Carlo Tree Search (MCTS) Decoding",
            "=" * 42,
            "",
            "Uses MCTS to build a search tree over the token vocabulary",
            "and extract diverse, high-quality sequences.",
            "",
            "Configuration:",
            f"  Simulations per step:   {cfg.n_simulations}",
            f"  Exploration constant:   {cfg.exploration_constant:.4f}",
            f"  Rollout depth:          {cfg.rollout_depth}",
            f"  Rollout policy:         {cfg.rollout_policy}",
            f"  Rollout temperature:    {cfg.rollout_temperature}",
            f"  Discount factor:        {cfg.discount_factor}",
            f"  Max children per node:  {cfg.max_children}",
            f"  Top-k expansion:        {cfg.top_k_expansion}",
            f"  Value function:         {cfg.value_function}",
            f"  Diversity weight:       {cfg.diversity_weight}",
            f"  Sequences to extract:   {cfg.n_sequences}",
            f"  Reuse tree:             {cfg.reuse_tree}",
            f"  Progressive widening:   {cfg.progressive_widening}",
            f"  PW alpha:               {cfg.pw_alpha}",
            f"  Virtual loss:           {cfg.virtual_loss}",
            f"  Temperature:            {cfg.temperature}",
            "",
            "Phases:",
            "  1. Selection:    Descend tree via PUCT (prior-guided UCB)",
            "  2. Expansion:    Add children from top-k logits",
            "  3. Simulation:   Rollout to fixed depth using chosen policy",
            "  4. Backprop:     Update visit counts and values to root",
            "",
            "Sequence extraction uses diversity-aware greedy selection",
            "with n-gram overlap penalties.",
        ]
        return "\n".join(lines)

    # -- Internal: _step (required by DecodingAlgorithm) --------------------

    def _step(
        self,
        state: DecodingState,
        logit_source: LogitSource,
    ) -> DecodingState:
        """Single decoding step (not used directly; MCTS overrides generate).

        This is implemented to satisfy the abstract interface but MCTS
        operates via its own generate() method that runs full simulations.
        """
        for i in state.active_indices():
            seq = state.sequences[i]
            logits = logit_source([seq])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()
            logits = self._apply_constraints(logits, state)
            token = sample_token(logits, temperature=self.mcts_config.temperature)
            state.update_sequence(i, token)

            if self.mcts_config.eos_token_id is not None:
                if token == self.mcts_config.eos_token_id:
                    state.mark_finished(i)

        state.step += 1
        return state

    # -- Logging ------------------------------------------------------------

    def _log_search_progress(
        self,
        iteration: int,
        tree_stats: Dict[str, Any],
    ) -> None:
        """Log periodic search progress.

        Parameters
        ----------
        iteration : int
            Current simulation number.
        tree_stats : dict
            Tree statistics from ``tree.tree_statistics()``.
        """
        elapsed = time.monotonic() - self._generation_start_time
        logger.info(
            "MCTS progress: sim=%d/%d, nodes=%d, depth=%d, "
            "root_children=%d, elapsed=%.2fs",
            iteration,
            self.mcts_config.n_simulations,
            tree_stats.get("total_nodes", 0),
            tree_stats.get("max_depth", 0),
            tree_stats.get("root_children_count", 0),
            elapsed,
        )
        if tree_stats.get("root_child_visits_top5"):
            logger.debug(
                "  Top-5 root child visits: %s",
                tree_stats["root_child_visits_top5"],
            )
        if tree_stats.get("root_child_values_top5"):
            logger.debug(
                "  Top-5 root child values: %s",
                tree_stats["root_child_values_top5"],
            )

    # -- Utilities ----------------------------------------------------------

    @staticmethod
    def _make_cache_key(seq: List[int]) -> str:
        """Create a hashable cache key from a token sequence."""
        return ",".join(str(t) for t in seq)


# =========================================================================
# MCTSAnalyzer — post-hoc tree analysis
# =========================================================================


class MCTSAnalyzer:
    """Analyse an MCTS tree after search is complete.

    Provides methods for inspecting the tree structure, visit distributions,
    diversity of extracted sequences, and search efficiency.
    """

    def __init__(self, tree: MCTSTree) -> None:
        self.tree = tree

    # -- Tree structure analysis --------------------------------------------

    def depth_profile(self) -> Dict[int, int]:
        """Count nodes at each depth level.

        Returns
        -------
        dict
            Mapping depth -> number of nodes at that depth.
        """
        profile: DefaultDict[int, int] = collections.defaultdict(int)
        stack: List[MCTSNode] = [self.tree.root]
        while stack:
            node = stack.pop()
            profile[node.depth] += 1
            stack.extend(node.children.values())
        return dict(sorted(profile.items()))

    def branching_factor_profile(self) -> Dict[int, float]:
        """Average branching factor at each depth.

        Returns
        -------
        dict
            Mapping depth -> average number of children.
        """
        depth_children: DefaultDict[int, List[int]] = collections.defaultdict(list)
        stack: List[MCTSNode] = [self.tree.root]
        while stack:
            node = stack.pop()
            if node.children:
                depth_children[node.depth].append(len(node.children))
            stack.extend(node.children.values())

        return {
            d: round(float(np.mean(counts)), 2)
            for d, counts in sorted(depth_children.items())
        }

    def visit_entropy(self) -> float:
        """Entropy of the visit distribution at the root's children.

        Higher entropy indicates more uniform exploration.

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        if not self.tree.root.children:
            return 0.0

        visits = np.array(
            [c.visit_count for c in self.tree.root.children.values()],
            dtype=np.float64,
        )
        total = visits.sum()
        if total <= 0:
            return 0.0
        probs = visits / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def visit_concentration(self) -> float:
        """Fraction of total visits captured by the top child.

        Returns
        -------
        float
            Ratio in [0, 1].  Higher means more concentrated search.
        """
        if not self.tree.root.children:
            return 0.0

        visits = [c.visit_count for c in self.tree.root.children.values()]
        total = sum(visits)
        if total <= 0:
            return 0.0
        return float(max(visits) / total)

    def effective_branching_factor(self) -> float:
        """Effective branching factor: exp(entropy) of root visit distribution.

        Returns
        -------
        float
        """
        return float(math.exp(self.visit_entropy()))

    # -- Path analysis ------------------------------------------------------

    def top_paths(
        self,
        k: int = 5,
        by: str = "visits",
    ) -> List[Dict[str, Any]]:
        """Extract the top-*k* paths from the tree.

        Parameters
        ----------
        k : int
        by : str
            Ranking criterion: ``'visits'`` or ``'value'``.

        Returns
        -------
        list of dict
            Each dict has keys ``'tokens'``, ``'visits'``, ``'value'``,
            ``'length'``.
        """
        leaves = self.tree.root.leaf_nodes()
        if by == "visits":
            leaves.sort(key=lambda n: n.visit_count, reverse=True)
        else:
            leaves.sort(key=lambda n: n.q_value, reverse=True)

        results: List[Dict[str, Any]] = []
        for leaf in leaves[:k]:
            seq = leaf.get_sequence()
            results.append({
                "tokens": seq,
                "visits": leaf.visit_count,
                "value": round(leaf.q_value, 6),
                "length": len(seq),
                "depth": leaf.depth,
            })
        return results

    def path_diversity(
        self,
        paths: List[List[int]],
        ngram_size: int = 3,
    ) -> Dict[str, float]:
        """Compute diversity metrics for a set of paths.

        Parameters
        ----------
        paths : list of list of int
        ngram_size : int

        Returns
        -------
        dict
            Metrics including mean pairwise Jaccard distance, unique n-gram
            ratio, and self-BLEU approximation.
        """
        if len(paths) < 2:
            return {
                "mean_pairwise_jaccard_distance": 0.0,
                "unique_ngram_ratio": 1.0,
                "n_unique_paths": len(paths),
            }

        # Pairwise Jaccard distances
        ngrams_per_path = [
            _extract_ngrams_set(p, ngram_size) for p in paths
        ]
        distances: List[float] = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                s1, s2 = ngrams_per_path[i], ngrams_per_path[j]
                if not s1 and not s2:
                    distances.append(0.0)
                elif not s1 or not s2:
                    distances.append(1.0)
                else:
                    inter = len(s1 & s2)
                    union = len(s1 | s2)
                    distances.append(1.0 - inter / union if union > 0 else 0.0)

        # Unique n-grams
        all_ngrams: Set[Tuple[int, ...]] = set()
        total_ngrams = 0
        for ng_set in ngrams_per_path:
            all_ngrams.update(ng_set)
            total_ngrams += len(ng_set)

        unique_ratio = len(all_ngrams) / max(total_ngrams, 1)

        return {
            "mean_pairwise_jaccard_distance": round(
                float(np.mean(distances)) if distances else 0.0, 4
            ),
            "unique_ngram_ratio": round(unique_ratio, 4),
            "n_unique_paths": len(set(tuple(p) for p in paths)),
        }

    # -- Search efficiency --------------------------------------------------

    def search_efficiency(self) -> Dict[str, Any]:
        """Compute search efficiency metrics.

        Returns
        -------
        dict
            Metrics including simulation throughput, visit efficiency,
            tree utilisation, and timing statistics.
        """
        stats = self.tree.tree_statistics()
        total_nodes = stats["total_nodes"]
        total_sims = stats["total_simulations"]

        # Fraction of tree nodes with at least one visit
        visited_nodes = 0
        stack: List[MCTSNode] = [self.tree.root]
        while stack:
            node = stack.pop()
            if node.visit_count > 0:
                visited_nodes += 1
            stack.extend(node.children.values())

        tree_utilisation = visited_nodes / max(total_nodes, 1)

        return {
            "total_simulations": total_sims,
            "total_nodes": total_nodes,
            "visited_nodes": visited_nodes,
            "tree_utilisation": round(tree_utilisation, 4),
            "sims_per_node": round(total_sims / max(total_nodes, 1), 2),
            "mean_sim_time_ms": stats.get("mean_simulation_time_ms", 0.0),
            "visit_entropy": round(self.visit_entropy(), 4),
            "visit_concentration": round(self.visit_concentration(), 4),
            "effective_branching_factor": round(
                self.effective_branching_factor(), 2
            ),
        }

    def summary(self) -> str:
        """Return a human-readable summary of the tree analysis.

        Returns
        -------
        str
        """
        stats = self.tree.tree_statistics()
        efficiency = self.search_efficiency()

        lines = [
            "MCTS Tree Analysis",
            "=" * 40,
            f"Total nodes:              {stats['total_nodes']}",
            f"Max depth:                {stats['max_depth']}",
            f"Terminal nodes:           {stats['terminal_nodes']}",
            f"Leaf nodes:               {stats['leaf_nodes']}",
            f"Root children:            {stats['root_children_count']}",
            f"Avg branching factor:     {stats['avg_branching_factor']}",
            f"Total simulations:        {stats['total_simulations']}",
            f"Mean sim time (ms):       {stats['mean_simulation_time_ms']}",
            "",
            "Search Efficiency:",
            f"  Tree utilisation:       {efficiency['tree_utilisation']}",
            f"  Sims per node:          {efficiency['sims_per_node']}",
            f"  Visit entropy:          {efficiency['visit_entropy']}",
            f"  Visit concentration:    {efficiency['visit_concentration']}",
            f"  Eff. branching factor:  {efficiency['effective_branching_factor']}",
        ]

        top = self.top_paths(k=3)
        if top:
            lines.append("")
            lines.append("Top-3 paths by visits:")
            for i, p in enumerate(top, 1):
                lines.append(
                    f"  {i}. len={p['length']}, visits={p['visits']}, "
                    f"value={p['value']}"
                )

        return "\n".join(lines)


# =========================================================================
# MCTSWithRAVE — MCTS variant with Rapid Action Value Estimation
# =========================================================================


class RAVETable:
    """Rapid Action Value Estimation (RAVE / AMAF) table.

    RAVE maintains "all-moves-as-first" statistics: the value of a token
    is estimated not just from simulations where it was the immediate next
    token, but also from simulations where it appeared anywhere in the
    rollout.

    This can dramatically speed up initial exploration in large action spaces.
    """

    def __init__(self, equivalence_parameter: float = 1000.0) -> None:
        self.equivalence_parameter = equivalence_parameter
        self._rave_visits: DefaultDict[int, int] = collections.defaultdict(int)
        self._rave_values: DefaultDict[int, float] = collections.defaultdict(float)

    def update(self, token_id: int, value: float) -> None:
        """Update RAVE statistics for a token.

        Parameters
        ----------
        token_id : int
        value : float
        """
        self._rave_visits[token_id] += 1
        self._rave_values[token_id] += value

    def rave_value(self, token_id: int) -> float:
        """Get the RAVE value for a token.

        Returns
        -------
        float
            Average RAVE value, or 0.0 if unvisited.
        """
        visits = self._rave_visits.get(token_id, 0)
        if visits == 0:
            return 0.0
        return self._rave_values[token_id] / visits

    def beta(self, node_visits: int) -> float:
        """Compute the RAVE blending parameter beta.

        beta = sqrt(k / (3 * N + k))

        where k is the equivalence parameter and N is the node visit count.
        Higher beta means more weight on RAVE estimates.

        Parameters
        ----------
        node_visits : int

        Returns
        -------
        float
            Beta in [0, 1].
        """
        k = self.equivalence_parameter
        return math.sqrt(k / (3.0 * node_visits + k))

    def blended_value(
        self,
        node_q: float,
        token_id: int,
        node_visits: int,
    ) -> float:
        """Blend the node Q-value with the RAVE estimate.

        Q_blended = (1 - beta) * Q_node + beta * Q_rave

        Parameters
        ----------
        node_q : float
        token_id : int
        node_visits : int

        Returns
        -------
        float
        """
        b = self.beta(node_visits)
        rv = self.rave_value(token_id)
        return (1.0 - b) * node_q + b * rv

    def update_from_rollout(self, tokens: List[int], value: float) -> None:
        """Update RAVE stats for all tokens that appeared in a rollout.

        Parameters
        ----------
        tokens : list of int
        value : float
        """
        seen: Set[int] = set()
        for t in tokens:
            if t not in seen:
                self.update(t, value)
                seen.add(t)

    def reset(self) -> None:
        """Clear all RAVE statistics."""
        self._rave_visits.clear()
        self._rave_values.clear()


class MCTSWithRAVE(MCTSDecoding):
    """MCTS variant that uses RAVE (Rapid Action Value Estimation).

    RAVE blends standard Q-values with AMAF (all-moves-as-first) estimates
    to speed up initial tree exploration.
    """

    def __init__(self, config: MCTSConfig) -> None:
        super().__init__(config)
        self._rave_table = RAVETable(equivalence_parameter=1000.0)

    def _run_mcts(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> MCTSTree:
        """Run MCTS with RAVE-augmented selection."""
        if self.mcts_config.reuse_tree and self._tree is not None:
            tree = self._tree
        else:
            tree = MCTSTree(prompt_ids, self.mcts_config)
            self._tree = tree
            self._rave_table.reset()

        n_sims = self.mcts_config.n_simulations

        for sim_idx in range(n_sims):
            sim_start = time.monotonic()

            # Selection with RAVE-augmented scoring
            selected = self._select_with_rave(tree.root, tree)

            if self.mcts_config.virtual_loss > 0:
                tree._apply_virtual_loss(selected)

            try:
                # Expansion
                if not selected.is_terminal and not selected._expanded:
                    expanded = tree.expand(selected, logit_source)
                else:
                    expanded = selected

                # Simulation
                value = self._run_simulation(expanded, logit_source, tree)
                self._total_rollouts += 1

                # Update RAVE from the rollout path
                rollout_tokens = expanded.get_sequence()
                self._rave_table.update_from_rollout(rollout_tokens, value)

                # Backpropagation
                tree.backpropagate(expanded, value)

            finally:
                if self.mcts_config.virtual_loss > 0:
                    tree._remove_virtual_loss(selected)

            sim_time = time.monotonic() - sim_start
            tree._simulation_times.append(sim_time)
            tree._total_simulations += 1

            if (sim_idx + 1) % max(1, n_sims // 10) == 0:
                stats = tree.tree_statistics()
                self._log_search_progress(sim_idx + 1, stats)

        return tree

    def _select_with_rave(
        self,
        node: MCTSNode,
        tree: MCTSTree,
    ) -> MCTSNode:
        """Selection with RAVE-blended values."""
        current = node
        c_explore = self.mcts_config.exploration_constant

        while not current.is_leaf() and not current.is_terminal:
            if self.mcts_config.progressive_widening:
                if tree._progressive_widen_check(current):
                    return current

            if not current._expanded:
                return current

            # Select best child using RAVE-blended PUCT
            best: Optional[MCTSNode] = None
            best_score = _NEG_INF
            parent_visits = current.visit_count

            for child in current.children.values():
                blended_q = self._rave_table.blended_value(
                    child.q_value, child.token_id, child.visit_count
                )
                exploration = (
                    c_explore
                    * child.prior_probability
                    * math.sqrt(max(parent_visits, 1))
                    / (1.0 + child.visit_count)
                )
                score = blended_q + exploration
                if score > best_score:
                    best_score = score
                    best = child

            if best is None:
                return current
            current = best

        return current


# =========================================================================
# MCTSWithPolicyNetwork — MCTS guided by a learned policy prior
# =========================================================================


class PolicyNetworkAdapter:
    """Adapter to use a logit source as a policy network for MCTS priors.

    In AlphaGo-style MCTS, the policy network provides informed priors
    for which actions to explore first, while the value network estimates
    leaf values directly (avoiding rollouts).
    """

    def __init__(
        self,
        logit_source: LogitSource,
        temperature: float = 1.0,
    ) -> None:
        self.logit_source = logit_source
        self.temperature = temperature

    def get_priors(self, prefix: List[int]) -> np.ndarray:
        """Get policy priors (probability distribution over vocab).

        Parameters
        ----------
        prefix : list of int
            Current token prefix.

        Returns
        -------
        np.ndarray
            Probability distribution of shape ``(vocab_size,)``.
        """
        logits = self.logit_source([prefix])
        if logits.ndim > 1:
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64).ravel()

        if self.temperature != 1.0:
            logits = logits / self.temperature

        return _stable_softmax(logits)

    def get_value(self, prefix: List[int]) -> float:
        """Estimate value from the policy network (approximate).

        Uses the entropy of the predicted distribution as a proxy:
        lower entropy (more confident) → higher value.

        Parameters
        ----------
        prefix : list of int

        Returns
        -------
        float
        """
        probs = self.get_priors(prefix)
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(len(probs)) if len(probs) > 0 else 1.0
        # Normalise: 0 entropy → value 1, max entropy → value 0
        return max(0.0, 1.0 - entropy / max(max_entropy, _EPSILON))


# =========================================================================
# Utility: n-gram extraction
# =========================================================================


def _extract_ngrams_set(
    tokens: List[int],
    n: int,
) -> Set[Tuple[int, ...]]:
    """Extract a set of n-grams from a token list.

    Parameters
    ----------
    tokens : list of int
    n : int
        N-gram size.

    Returns
    -------
    set of tuple of int
    """
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard_distance(
    set_a: Set[Tuple[int, ...]],
    set_b: Set[Tuple[int, ...]],
) -> float:
    """Compute Jaccard distance between two sets.

    Parameters
    ----------
    set_a, set_b : set

    Returns
    -------
    float
        Jaccard distance in [0, 1].
    """
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


def _hamming_distance(seq_a: List[int], seq_b: List[int]) -> int:
    """Compute token-level Hamming distance between two sequences.

    Sequences are compared up to the length of the shorter one.

    Parameters
    ----------
    seq_a, seq_b : list of int

    Returns
    -------
    int
    """
    min_len = min(len(seq_a), len(seq_b))
    diff = sum(1 for i in range(min_len) if seq_a[i] != seq_b[i])
    diff += abs(len(seq_a) - len(seq_b))
    return diff


def _edit_distance(seq_a: List[int], seq_b: List[int]) -> int:
    """Compute Levenshtein edit distance between two token sequences.

    Parameters
    ----------
    seq_a, seq_b : list of int

    Returns
    -------
    int
    """
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two-row DP for memory efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


# =========================================================================
# TreeVisualizer — text-based tree rendering
# =========================================================================


class TreeVisualizer:
    """Render an MCTS tree as text for debugging and inspection."""

    def __init__(
        self,
        max_depth: int = 5,
        max_children: int = 5,
        show_stats: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.max_children = max_children
        self.show_stats = show_stats

    def render(self, root: MCTSNode) -> str:
        """Render the tree as an indented text string.

        Parameters
        ----------
        root : MCTSNode

        Returns
        -------
        str
        """
        lines: List[str] = []
        self._render_node(root, lines, prefix="", is_last=True, current_depth=0)
        return "\n".join(lines)

    def _render_node(
        self,
        node: MCTSNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
        current_depth: int,
    ) -> None:
        """Recursively render a node and its children."""
        connector = "└── " if is_last else "├── "
        if current_depth == 0:
            connector = ""
            label = "[ROOT]"
        else:
            label = f"tok={node.token_id}"

        stats_str = ""
        if self.show_stats:
            stats_str = (
                f" (N={node.visit_count}, Q={node.q_value:.3f}, "
                f"P={node.prior_probability:.3f})"
            )

        lines.append(f"{prefix}{connector}{label}{stats_str}")

        if current_depth >= self.max_depth:
            if node.children:
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{child_prefix}... ({len(node.children)} children)")
            return

        # Sort children by visit count and limit
        sorted_children = sorted(
            node.children.values(),
            key=lambda c: c.visit_count,
            reverse=True,
        )
        display_children = sorted_children[: self.max_children]
        remaining = len(sorted_children) - len(display_children)

        child_prefix = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(display_children):
            child_is_last = (i == len(display_children) - 1) and remaining == 0
            self._render_node(
                child, lines, child_prefix, child_is_last, current_depth + 1
            )

        if remaining > 0:
            lines.append(f"{child_prefix}└── ... (+{remaining} more children)")


# =========================================================================
# MCTSEnsemble — run multiple MCTS trees with different configs
# =========================================================================


class MCTSEnsemble:
    """Run multiple MCTS instances with different configurations and merge results.

    This is useful for combining the strengths of different rollout policies,
    exploration constants, or value functions.
    """

    def __init__(self, configs: List[MCTSConfig]) -> None:
        if not configs:
            raise ValueError("At least one config is required.")
        self.configs = configs
        self.decoders = [MCTSDecoding(cfg) for cfg in configs]

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: int = 5,
    ) -> List[TokenSequence]:
        """Generate sequences using all ensemble members and merge.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        n_sequences : int

        Returns
        -------
        list of TokenSequence
        """
        all_sequences: List[Tuple[List[int], float]] = []

        # Each decoder produces n_sequences candidates
        per_decoder = max(1, n_sequences)
        for decoder in self.decoders:
            seqs = decoder.generate(
                logit_source, prompt_ids, n_sequences=per_decoder
            )
            for seq in seqs:
                # Score with the decoder's own log-probability computation
                score = decoder._sequence_log_probability(
                    logit_source, prompt_ids + seq
                )
                all_sequences.append((seq, score))

        # Sort by score
        all_sequences.sort(key=lambda x: x[1], reverse=True)

        # Diversity-aware selection
        selected: List[TokenSequence] = []
        used_ngrams: Set[Tuple[int, ...]] = set()

        for seq, _score in all_sequences:
            if len(selected) >= n_sequences:
                break

            seq_ngrams = _extract_ngrams_set(seq, 3)
            if selected and seq_ngrams:
                overlap = len(seq_ngrams & used_ngrams) / max(len(seq_ngrams), 1)
                if overlap > 0.8:
                    continue

            selected.append(seq)
            if seq_ngrams:
                used_ngrams.update(seq_ngrams)

        return selected

    def describe(self) -> str:
        """Describe the ensemble configuration."""
        lines = [
            f"MCTS Ensemble with {len(self.configs)} members:",
        ]
        for i, cfg in enumerate(self.configs):
            lines.append(
                f"  {i + 1}. sims={cfg.n_simulations}, "
                f"c={cfg.exploration_constant:.2f}, "
                f"policy={cfg.rollout_policy}, "
                f"vf={cfg.value_function}"
            )
        return "\n".join(lines)


# =========================================================================
# Adaptive MCTS — dynamically adjust parameters during search
# =========================================================================


class AdaptiveMCTS(MCTSDecoding):
    """MCTS variant that adaptively adjusts exploration during search.

    Monitors search progress and adjusts the exploration constant,
    rollout depth, and expansion width to balance exploration and
    exploitation over time.
    """

    def __init__(self, config: MCTSConfig) -> None:
        super().__init__(config)
        self._initial_exploration = config.exploration_constant
        self._exploration_schedule: List[float] = []
        self._adaptation_window: int = 20
        self._min_exploration: float = 0.1
        self._max_exploration: float = 5.0

    def _run_mcts(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> MCTSTree:
        """Run MCTS with adaptive exploration constant."""
        if self.mcts_config.reuse_tree and self._tree is not None:
            tree = self._tree
        else:
            tree = MCTSTree(prompt_ids, self.mcts_config)
            self._tree = tree

        n_sims = self.mcts_config.n_simulations
        current_exploration = self._initial_exploration
        recent_values: Deque[float] = collections.deque(
            maxlen=self._adaptation_window
        )

        for sim_idx in range(n_sims):
            sim_start = time.monotonic()

            # Update exploration constant based on progress
            tree.config = copy.copy(self.mcts_config)
            tree.config.exploration_constant = current_exploration

            selected = tree.select(tree.root)

            if self.mcts_config.virtual_loss > 0:
                tree._apply_virtual_loss(selected)

            try:
                if not selected.is_terminal and not selected._expanded:
                    expanded = tree.expand(selected, logit_source)
                else:
                    expanded = selected

                value = self._run_simulation(expanded, logit_source, tree)
                self._total_rollouts += 1
                recent_values.append(value)

                tree.backpropagate(expanded, value)

            finally:
                if self.mcts_config.virtual_loss > 0:
                    tree._remove_virtual_loss(selected)

            sim_time = time.monotonic() - sim_start
            tree._simulation_times.append(sim_time)
            tree._total_simulations += 1

            # Adapt exploration constant
            if (sim_idx + 1) % self._adaptation_window == 0:
                current_exploration = self._adapt_exploration(
                    current_exploration, recent_values, tree
                )
                self._exploration_schedule.append(current_exploration)

            if (sim_idx + 1) % max(1, n_sims // 10) == 0:
                stats = tree.tree_statistics()
                self._log_search_progress(sim_idx + 1, stats)

        return tree

    def _adapt_exploration(
        self,
        current: float,
        recent_values: Deque[float],
        tree: MCTSTree,
    ) -> float:
        """Adapt the exploration constant based on search progress.

        Strategy:
        - If value variance is high → increase exploration (more to discover)
        - If value variance is low → decrease exploration (exploit best paths)
        - Also consider visit concentration at root

        Parameters
        ----------
        current : float
        recent_values : deque of float
        tree : MCTSTree

        Returns
        -------
        float
            Adapted exploration constant.
        """
        if len(recent_values) < 5:
            return current

        values = np.array(list(recent_values))
        value_std = float(np.std(values))
        value_mean = float(np.mean(values))

        # Coefficient of variation
        if abs(value_mean) > _EPSILON:
            cv = value_std / abs(value_mean)
        else:
            cv = 1.0

        # Visit concentration at root
        if tree.root.children:
            visits = [c.visit_count for c in tree.root.children.values()]
            total = sum(visits) or 1
            concentration = max(visits) / total
        else:
            concentration = 0.0

        # Adaptation logic
        if cv > 0.5 or concentration > 0.8:
            # High variance or too concentrated → explore more
            new_exploration = current * 1.2
        elif cv < 0.1 and concentration < 0.3:
            # Low variance and well-spread → exploit more
            new_exploration = current * 0.8
        else:
            new_exploration = current

        return float(np.clip(
            new_exploration, self._min_exploration, self._max_exploration
        ))


# =========================================================================
# Module-level helpers
# =========================================================================


def create_mcts_decoder(
    n_simulations: int = 100,
    exploration_constant: float = 1.414,
    rollout_policy: str = "random",
    value_function: str = "perplexity",
    n_sequences: int = 5,
    **kwargs: Any,
) -> MCTSDecoding:
    """Convenience factory for creating an MCTSDecoding instance.

    Parameters
    ----------
    n_simulations : int
    exploration_constant : float
    rollout_policy : str
    value_function : str
    n_sequences : int
    **kwargs
        Additional MCTSConfig parameters.

    Returns
    -------
    MCTSDecoding
    """
    config = MCTSConfig(
        n_simulations=n_simulations,
        exploration_constant=exploration_constant,
        rollout_policy=rollout_policy,
        value_function=value_function,
        n_sequences=n_sequences,
        **kwargs,
    )
    return MCTSDecoding(config)


def create_mcts_ensemble(
    n_members: int = 3,
    base_simulations: int = 50,
    **kwargs: Any,
) -> MCTSEnsemble:
    """Create an MCTS ensemble with diverse configurations.

    Parameters
    ----------
    n_members : int
    base_simulations : int
    **kwargs
        Shared parameters for all members.

    Returns
    -------
    MCTSEnsemble
    """
    policies = ["random", "greedy", "nucleus", "temperature"]
    exploration_values = [0.5, 1.0, 1.414, 2.0, 3.0]

    configs: List[MCTSConfig] = []
    for i in range(n_members):
        policy = policies[i % len(policies)]
        exploration = exploration_values[i % len(exploration_values)]
        cfg = MCTSConfig(
            n_simulations=base_simulations,
            exploration_constant=exploration,
            rollout_policy=policy,
            **kwargs,
        )
        configs.append(cfg)

    return MCTSEnsemble(configs)


# =========================================================================
# Self-test
# =========================================================================


def _self_test() -> None:
    """Minimal smoke test for the MCTS module."""

    # -- MCTSConfig --
    config = MCTSConfig()
    errors = config.validate()
    assert not errors, f"Default config should be valid: {errors}"

    bad_config = MCTSConfig(n_simulations=0)
    errors = bad_config.validate()
    assert any("n_simulations" in e for e in errors)

    # -- MCTSNode --
    root = MCTSNode(token_id=-1, parent=None, prior_prob=1.0)
    assert root.depth == 0
    assert root.is_leaf()
    assert root.q_value == 0.0
    assert root.get_sequence() == []

    # Expand with dummy logits
    logits = np.array([1.0, 3.0, 2.0, 0.5, 4.0])
    children = root.expand(logits, top_k=3)
    assert len(children) == 3
    assert not root.is_leaf()

    # UCB and PUCT scores
    for child in children:
        child.visit_count = 5
        child.total_value = 2.5
        ucb = child.ucb_score(parent_visits=15, exploration_constant=1.414)
        assert isinstance(ucb, float)
        puct = child.puct_score(parent_visits=15, c_puct=1.414)
        assert isinstance(puct, float)

    # Best child / most visited
    best = root.best_child(1.414)
    assert isinstance(best, MCTSNode)
    most = root.most_visited_child()
    assert isinstance(most, MCTSNode)

    # Sequence extraction
    child = list(root.children.values())[0]
    seq = child.get_sequence()
    assert len(seq) == 1

    # -- MCTSTree --
    dummy_source: LogitSource = lambda ids: np.random.randn(len(ids), 5)
    tree = MCTSTree([1, 2, 3], MCTSConfig(n_simulations=5, rollout_depth=3))

    # Expand root
    expanded = tree.expand(tree.root, dummy_source)
    assert not tree.root.is_leaf()

    stats = tree.tree_statistics()
    assert stats["total_nodes"] > 1

    # -- Pruning --
    tree.prune(min_visits=0)
    assert tree.node_count() > 0

    # -- MCTSDecoding --
    decoder = MCTSDecoding(MCTSConfig(
        n_simulations=5,
        rollout_depth=3,
        n_sequences=2,
        max_new_tokens=5,
    ))
    sequences = decoder.generate(dummy_source, [1, 2, 3], n_sequences=2)
    assert isinstance(sequences, list)

    # -- Describe --
    desc = decoder.describe()
    assert "Monte Carlo Tree Search" in desc

    # -- Hyperparameter grid --
    grid = decoder.get_hyperparameter_grid()
    assert "n_simulations" in grid
    assert "exploration_constant" in grid

    # -- MCTSAnalyzer --
    tree2 = MCTSTree([1, 2], MCTSConfig(n_simulations=10, rollout_depth=3))
    for _ in range(10):
        sel = tree2.select(tree2.root)
        if not sel._expanded:
            exp = tree2.expand(sel, dummy_source)
        else:
            exp = sel
        val = tree2.simulate(exp, dummy_source)
        tree2.backpropagate(exp, val)

    analyzer = MCTSAnalyzer(tree2)
    dp = analyzer.depth_profile()
    assert 0 in dp
    eff = analyzer.search_efficiency()
    assert "total_simulations" in eff
    summary = analyzer.summary()
    assert "MCTS Tree Analysis" in summary

    # -- TreeVisualizer --
    viz = TreeVisualizer(max_depth=3, max_children=3)
    text = viz.render(tree2.root)
    assert "[ROOT]" in text

    # -- RolloutPolicy --
    random_policy = RandomRolloutPolicy()
    token = random_policy.select_token(np.array([1.0, 2.0, 3.0]))
    assert 0 <= token <= 2

    greedy_policy = GreedyRolloutPolicy()
    token = greedy_policy.select_token(np.array([1.0, 2.0, 3.0]))
    assert token == 2

    # -- ValueFunction --
    ppl_vf = PerplexityValueFunction()
    val = ppl_vf.evaluate([1, 2, 3], dummy_source)
    assert isinstance(val, float)

    # -- Utility functions --
    ngrams = _extract_ngrams_set([1, 2, 3, 4, 5], 3)
    assert len(ngrams) == 3  # (1,2,3), (2,3,4), (3,4,5)

    jd = _jaccard_distance({(1, 2)}, {(2, 3)})
    assert 0 <= jd <= 1

    hd = _hamming_distance([1, 2, 3], [1, 3, 3])
    assert hd == 1

    ed = _edit_distance([1, 2, 3], [1, 3])
    assert ed > 0

    # -- Factory functions --
    decoder2 = create_mcts_decoder(n_simulations=5, rollout_policy="greedy")
    assert decoder2.mcts_config.rollout_policy == "greedy"

    ensemble = create_mcts_ensemble(n_members=2, base_simulations=5)
    assert len(ensemble.decoders) == 2

    print("mcts.py self-test passed ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()


# =========================================================================
# UCTDiversityBonus — UCT selection with explicit diversity bonus
# =========================================================================


class UCTDiversityBonus:
    """UCT selection policy with an explicit diversity bonus term.

    Modifies the standard UCT formula by adding a diversity bonus:

        UCT = Q/N + C * sqrt(ln(N_parent) / N) + D * diversity_bonus

    The diversity bonus is computed as the average n-gram Jaccard distance
    from the node's partial sequence to all other explored sequences at the
    same depth.  This encourages the search to explore parts of the tree
    that produce sequences dissimilar to those already found.

    Parameters
    ----------
    diversity_weight : float
        Weight *D* for the diversity bonus term.  Higher values push the
        search toward more diverse paths at the cost of raw quality.
    ngram_size : int
        Size of n-grams used for diversity computation.
    exploration_constant : float
        Standard UCT exploration constant *C*.
    """

    def __init__(
        self,
        diversity_weight: float = 0.5,
        ngram_size: int = 3,
        exploration_constant: float = 1.414,
    ) -> None:
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self.exploration_constant = exploration_constant
        self._explored_sequences: DefaultDict[int, List[List[int]]] = (
            collections.defaultdict(list)
        )
        logger.debug(
            "UCTDiversityBonus initialised: D=%.3f, ngram=%d, C=%.3f",
            diversity_weight,
            ngram_size,
            exploration_constant,
        )

    # -- diversity computation ----------------------------------------------

    def _compute_diversity_bonus(
        self,
        node: MCTSNode,
        depth_sequences: List[List[int]],
    ) -> float:
        """Compute diversity bonus for a node relative to peer sequences.

        The bonus is the mean Jaccard distance (in n-gram space) between
        this node's partial sequence and every other sequence explored at
        the same depth.

        Parameters
        ----------
        node : MCTSNode
            The node whose diversity bonus is being computed.
        depth_sequences : list of list of int
            Other sequences at the same depth in the tree.

        Returns
        -------
        float
            Diversity bonus in ``[0, 1]``.  Returns 1.0 when there are no
            peer sequences (maximum exploration incentive).
        """
        if not depth_sequences:
            return 1.0

        node_seq = node.get_sequence()
        node_ngrams = _extract_ngrams_set(node_seq, self.ngram_size)

        if not node_ngrams:
            return 1.0

        distances = np.array([
            _jaccard_distance(
                node_ngrams,
                _extract_ngrams_set(seq, self.ngram_size),
            )
            for seq in depth_sequences
        ], dtype=np.float64)

        return float(np.mean(distances)) if distances.size > 0 else 1.0

    def _uct_with_diversity(
        self,
        node: MCTSNode,
        parent_visits: int,
        depth_sequences: List[List[int]],
    ) -> float:
        """Compute UCT score augmented with the diversity bonus.

        Parameters
        ----------
        node : MCTSNode
        parent_visits : int
            Visit count of the parent node.
        depth_sequences : list of list of int
            Sequences at the same depth used for diversity computation.

        Returns
        -------
        float
            The diversity-augmented UCT score.
        """
        if node.visit_count == 0:
            return float("inf")

        exploitation = node.q_value
        exploration = self.exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / node.visit_count
        )
        diversity = self.diversity_weight * self._compute_diversity_bonus(
            node, depth_sequences
        )
        return exploitation + exploration + diversity

    # -- selection ----------------------------------------------------------

    def select(self, root: MCTSNode) -> MCTSNode:
        """Select a leaf node using diversity-augmented UCT.

        Descends from *root*, at each internal node choosing the child
        that maximises the diversity-augmented UCT score.

        Parameters
        ----------
        root : MCTSNode
            The root of the search tree.

        Returns
        -------
        MCTSNode
            The selected leaf or unexpanded node.
        """
        current = root

        while not current.is_leaf() and not current.is_terminal:
            if not current._expanded:
                return current

            depth_seqs = self._sequences_at_depth(current.depth + 1)
            best: Optional[MCTSNode] = None
            best_score = _NEG_INF
            parent_visits = current.visit_count

            for child in current.children.values():
                score = self._uct_with_diversity(
                    child, parent_visits, depth_seqs
                )
                if score > best_score:
                    best_score = score
                    best = child

            if best is None:
                return current
            current = best

        return current

    def _sequences_at_depth(self, depth: int) -> List[List[int]]:
        """Return all explored sequences recorded at a given depth.

        Parameters
        ----------
        depth : int
            Tree depth to query.

        Returns
        -------
        list of list of int
        """
        return self._explored_sequences.get(depth, [])

    def record_sequence(self, node: MCTSNode) -> None:
        """Record a node's partial sequence for future diversity computation.

        Parameters
        ----------
        node : MCTSNode
        """
        seq = node.get_sequence()
        if seq:
            self._explored_sequences[node.depth].append(seq)


# =========================================================================
# ProgressiveWideningQD — quality-diversity progressive widening
# =========================================================================


class ProgressiveWideningQD:
    """Progressive widening with quality-diversity balance.

    Controls how many children each node can have, gradually widening
    based on visit count.  When deciding which child to expand next,
    balances quality (expected value) with diversity (behavioural distance
    from existing siblings).

    The maximum number of children at a node with *N* visits is:

        max_children = floor(C_w * N^alpha)

    where *C_w* is the widening constant and *alpha* controls the growth
    rate.

    Parameters
    ----------
    config : MCTSConfig
        MCTS configuration providing base parameters.
    widening_constant : float
        Multiplier *C_w* for progressive widening threshold.
    widening_exponent : float
        Exponent *alpha* controlling growth rate.
    quality_weight : float
        Weight for quality (expected value) in child selection.
    diversity_weight : float
        Weight for diversity (distance from siblings) in child selection.
    ngram_size : int
        N-gram size for diversity computation.
    """

    def __init__(
        self,
        config: MCTSConfig,
        widening_constant: float = 1.0,
        widening_exponent: float = 0.5,
        quality_weight: float = 0.7,
        diversity_weight: float = 0.3,
        ngram_size: int = 3,
    ) -> None:
        self.config = config
        self.widening_constant = widening_constant
        self.widening_exponent = widening_exponent
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self._tree: Optional[MCTSTree] = None
        logger.debug(
            "ProgressiveWideningQD initialised: C_w=%.2f, alpha=%.2f, "
            "q_w=%.2f, d_w=%.2f",
            widening_constant,
            widening_exponent,
            quality_weight,
            diversity_weight,
        )

    def _max_children(self, visit_count: int) -> int:
        """Compute the maximum number of children allowed for a node.

        Parameters
        ----------
        visit_count : int
            Number of visits to the node.

        Returns
        -------
        int
            Maximum number of children, at least 1.
        """
        return max(1, int(math.floor(
            self.widening_constant * (visit_count + 1) ** self.widening_exponent
        )))

    def _should_widen(self, node: MCTSNode) -> bool:
        """Check whether a node should have more children added.

        Parameters
        ----------
        node : MCTSNode

        Returns
        -------
        bool
            ``True`` if the current child count is below the progressive
            widening threshold for the node's visit count.
        """
        max_c = self._max_children(node.visit_count)
        return len(node.children) < max_c

    def _select_diverse_expansion(
        self,
        node: MCTSNode,
        logits: np.ndarray,
        n_candidates: int = 10,
    ) -> int:
        """Select a token for expansion that balances quality and diversity.

        From the top candidate tokens (by logit score), selects the one
        that maximises:

            score = quality_weight * prior + diversity_weight * sibling_div

        Parameters
        ----------
        node : MCTSNode
            The parent node being expanded.
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.
        n_candidates : int
            Number of top candidates to consider.

        Returns
        -------
        int
            Token ID selected for expansion.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        vocab_size = logits.shape[0]
        effective_k = min(n_candidates, vocab_size)

        # Get top candidates not yet expanded
        sorted_indices = np.argsort(logits)[::-1]
        probs = _stable_softmax(logits)

        candidates: List[Tuple[int, float]] = []
        for idx in sorted_indices:
            token_id = int(idx)
            if token_id not in node.children:
                candidates.append((token_id, float(probs[idx])))
            if len(candidates) >= effective_k:
                break

        if not candidates:
            # All top candidates already expanded; pick the best remaining
            for idx in sorted_indices:
                token_id = int(idx)
                if token_id not in node.children:
                    return token_id
            # Fallback: return first sorted token
            return int(sorted_indices[0])

        # Score each candidate
        best_token = candidates[0][0]
        best_score = _NEG_INF

        for token_id, prior in candidates:
            div = self._sibling_diversity(node, token_id)
            score = self.quality_weight * prior + self.diversity_weight * div
            if score > best_score:
                best_score = score
                best_token = token_id

        return best_token

    def _sibling_diversity(self, parent: MCTSNode, token_id: int) -> float:
        """Compute how diverse a candidate token is from existing siblings.

        Returns the mean Jaccard distance (in n-gram space) between the
        hypothetical sequence ``parent_seq + [token_id]`` and the sequences
        of existing sibling nodes.

        Parameters
        ----------
        parent : MCTSNode
        token_id : int

        Returns
        -------
        float
            Mean diversity in ``[0, 1]``.  Returns 1.0 if there are no
            siblings.
        """
        if not parent.children:
            return 1.0

        parent_seq = parent.get_sequence()
        candidate_seq = parent_seq + [token_id]
        candidate_ngrams = _extract_ngrams_set(candidate_seq, self.ngram_size)

        if not candidate_ngrams:
            return 1.0

        distances: List[float] = []
        for child in parent.children.values():
            sibling_seq = child.get_sequence()
            sibling_ngrams = _extract_ngrams_set(sibling_seq, self.ngram_size)
            distances.append(_jaccard_distance(candidate_ngrams, sibling_ngrams))

        return float(np.mean(distances)) if distances else 1.0

    def expand_with_widening(
        self,
        node: MCTSNode,
        logit_source: LogitSource,
        root_tokens: List[int],
    ) -> MCTSNode:
        """Expand a node using progressive widening with QD selection.

        If the node is eligible for widening, selects a quality-diverse
        token and creates a new child.  Otherwise returns the node.

        Parameters
        ----------
        node : MCTSNode
            The node to potentially expand.
        logit_source : LogitSource
            Source of next-token logits.
        root_tokens : list of int
            Prompt token IDs for context.

        Returns
        -------
        MCTSNode
            The newly created child or the node itself.
        """
        if node.is_terminal:
            return node

        if not self._should_widen(node):
            return node

        prefix = root_tokens + node.get_sequence()
        logits = logit_source([prefix])
        if logits.ndim > 1:
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64).ravel()

        token_id = self._select_diverse_expansion(node, logits)
        probs = _stable_softmax(logits)
        prior = float(probs[token_id])

        child = node.expand_single(token_id, prior)

        eos_id = self.config.eos_token_id
        if eos_id is not None and token_id == eos_id:
            child.is_terminal = True

        logger.debug(
            "QD-widening expanded node depth=%d with token=%d (prior=%.4f)",
            node.depth,
            token_id,
            prior,
        )
        return child

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: int = 5,
        n_simulations: int = 100,
    ) -> List[TokenSequence]:
        """Generate diverse sequences using progressive widening with QD.

        Runs a full MCTS loop using quality-diversity progressive widening
        for the expansion phase.

        Parameters
        ----------
        logit_source : LogitSource
            Callable ``(List[List[int]]) -> np.ndarray`` returning logits.
        prompt_ids : list of int
            Token IDs of the prompt.
        n_sequences : int
            Number of sequences to generate.
        n_simulations : int
            Number of MCTS simulations to run.

        Returns
        -------
        list of TokenSequence
            Generated token sequences (without prompt).
        """
        tree = MCTSTree(prompt_ids, self.config)
        self._tree = tree

        rollout_policy = _create_rollout_policy(self.config)
        value_fn = _create_value_function(self.config)

        for sim_idx in range(n_simulations):
            # Selection
            selected = tree.select(tree.root)

            # Expansion with QD widening
            if not selected.is_terminal:
                expanded = self.expand_with_widening(
                    selected, logit_source, prompt_ids
                )
            else:
                expanded = selected

            # Simulation
            prefix = prompt_ids + expanded.get_sequence()
            value = value_fn.evaluate(prefix, logit_source)

            # Backpropagation
            tree.backpropagate(expanded, value)
            tree._total_simulations += 1

            if (sim_idx + 1) % max(1, n_simulations // 10) == 0:
                logger.debug(
                    "QD-widening simulation %d/%d complete",
                    sim_idx + 1,
                    n_simulations,
                )

        sequences = tree.top_k_sequences(n_sequences)
        logger.info(
            "ProgressiveWideningQD generated %d sequences from %d simulations",
            len(sequences),
            n_simulations,
        )
        return sequences


# =========================================================================
# DiversityRewardRollout — rollout with diversity-shaped rewards
# =========================================================================


class DiversityRewardRollout(RolloutPolicy):
    """Rollout policy that shapes rewards for diversity.

    During simulation, this policy adds a diversity reward based on:

    * **N-gram novelty** — fraction of n-grams in the rollout that have
      not been seen in previously completed sequences.
    * **Vocabulary coverage** — fraction of unique tokens in the rollout
      relative to a reference vocabulary size.
    * **Information content** — entropy-based measure of the token
      distribution in the rollout.

    The final shaped reward is a weighted combination of these components
    added to a base quality score.

    Parameters
    ----------
    novelty_weight : float
        Weight for the n-gram novelty component.
    coverage_weight : float
        Weight for the vocabulary coverage component.
    info_weight : float
        Weight for the information content component.
    ngram_size : int
        N-gram size used for novelty computation.
    vocab_size : int
        Reference vocabulary size for coverage computation.
    """

    def __init__(
        self,
        novelty_weight: float = 0.4,
        coverage_weight: float = 0.3,
        info_weight: float = 0.3,
        ngram_size: int = 3,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
    ) -> None:
        self.novelty_weight = novelty_weight
        self.coverage_weight = coverage_weight
        self.info_weight = info_weight
        self.ngram_size = ngram_size
        self.vocab_size = vocab_size
        self._seen_ngrams: Set[Tuple[int, ...]] = set()
        self._completed_sequences: List[List[int]] = []
        logger.debug(
            "DiversityRewardRollout initialised: novelty=%.2f, coverage=%.2f, "
            "info=%.2f, ngram=%d",
            novelty_weight,
            coverage_weight,
            info_weight,
            ngram_size,
        )

    def select_token(self, logits: np.ndarray) -> int:
        """Select the next token during rollout using nucleus sampling.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.

        Returns
        -------
        int
            Selected token index.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        probs = _stable_softmax(logits)
        return int(np.random.choice(len(probs), p=probs))

    def _novelty_reward(
        self,
        tokens: List[int],
    ) -> float:
        """Compute n-gram novelty reward.

        The novelty is the fraction of n-grams in *tokens* that do not
        appear in the set of previously seen n-grams.

        Parameters
        ----------
        tokens : list of int
            Rollout token sequence.

        Returns
        -------
        float
            Novelty score in ``[0, 1]``.
        """
        rollout_ngrams = _extract_ngrams_set(tokens, self.ngram_size)
        if not rollout_ngrams:
            return 1.0

        novel = rollout_ngrams - self._seen_ngrams
        return len(novel) / len(rollout_ngrams)

    def _coverage_reward(self, tokens: List[int]) -> float:
        """Compute vocabulary coverage reward.

        Fraction of unique tokens in the rollout relative to the reference
        vocabulary size, clamped to ``[0, 1]``.

        Parameters
        ----------
        tokens : list of int

        Returns
        -------
        float
        """
        if not tokens:
            return 0.0
        unique_tokens = len(set(tokens))
        return min(1.0, unique_tokens / max(self.vocab_size, 1))

    def _information_reward(self, tokens: List[int]) -> float:
        """Compute information content reward based on token entropy.

        Uses the empirical token distribution of the rollout to compute
        normalised Shannon entropy.

        Parameters
        ----------
        tokens : list of int

        Returns
        -------
        float
            Normalised entropy in ``[0, 1]``.
        """
        if not tokens:
            return 0.0

        counts = np.array(
            list(collections.Counter(tokens).values()),
            dtype=np.float64,
        )
        probs = counts / counts.sum()
        # Shannon entropy
        entropy = -float(np.sum(probs * np.log(probs + _LOG_EPS)))
        # Normalise by max possible entropy
        max_entropy = math.log(max(len(counts), 1) + _LOG_EPS)
        if max_entropy <= 0:
            return 0.0
        return min(1.0, entropy / max_entropy)

    def _shaped_reward(
        self,
        tokens: List[int],
        base_value: float,
    ) -> float:
        """Compute the diversity-shaped reward.

        Combines the base quality value with novelty, coverage, and
        information content rewards.

        Parameters
        ----------
        tokens : list of int
            Rollout token sequence.
        base_value : float
            Base quality value from the value function.

        Returns
        -------
        float
            Final shaped reward.
        """
        novelty = self._novelty_reward(tokens)
        coverage = self._coverage_reward(tokens)
        info = self._information_reward(tokens)

        diversity_bonus = (
            self.novelty_weight * novelty
            + self.coverage_weight * coverage
            + self.info_weight * info
        )

        return base_value + diversity_bonus

    def rollout(
        self,
        logit_source: LogitSource,
        prefix: List[int],
        max_steps: int = 20,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Perform a rollout (simulation) from a prefix.

        Parameters
        ----------
        logit_source : LogitSource
            Source of next-token logits.
        prefix : list of int
            Tokens up to the current point.
        max_steps : int
            Maximum rollout length.
        eos_token_id : int, optional
            End-of-sequence token ID.

        Returns
        -------
        list of int
            Tokens generated during the rollout.
        """
        rollout_tokens: List[int] = []
        current_prefix = list(prefix)

        for _ in range(max_steps):
            logits = logit_source([current_prefix])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            token = self.select_token(logits)
            rollout_tokens.append(token)
            current_prefix.append(token)

            if eos_token_id is not None and token == eos_token_id:
                break

        return rollout_tokens

    def evaluate(
        self,
        rollout_tokens: List[int],
        base_value: float,
    ) -> float:
        """Evaluate a rollout and return a diversity-shaped reward.

        Also records the rollout's n-grams for future novelty computation.

        Parameters
        ----------
        rollout_tokens : list of int
            Tokens produced during the rollout.
        base_value : float
            Base quality score from the value function.

        Returns
        -------
        float
            The diversity-shaped reward value.
        """
        shaped = self._shaped_reward(rollout_tokens, base_value)

        # Record n-grams for future novelty
        new_ngrams = _extract_ngrams_set(rollout_tokens, self.ngram_size)
        self._seen_ngrams.update(new_ngrams)
        self._completed_sequences.append(list(rollout_tokens))

        logger.debug(
            "DiversityRewardRollout: base=%.4f, shaped=%.4f, "
            "n_seen_ngrams=%d",
            base_value,
            shaped,
            len(self._seen_ngrams),
        )
        return shaped


# =========================================================================
# RAVEDiversity — RAVE extended for diversity-aware MCTS
# =========================================================================


class RAVEDiversity:
    """RAVE (Rapid Action Value Estimation) extended for diversity-aware MCTS.

    Maintains RAVE statistics that track not just value but also diversity
    contribution of each action (token).  Uses RAVE to quickly estimate
    which tokens lead to diverse sequences, even before those tokens have
    been fully explored at a given node.

    The combined RAVE value blends:

    * Standard RAVE value — average outcome when the token appeared in
      any rollout.
    * Diversity RAVE — average diversity contribution when the token
      appeared in rollouts.

    Parameters
    ----------
    equivalence_parameter : float
        Controls the RAVE blending schedule (higher = trust RAVE longer).
    diversity_weight : float
        Weight for the diversity component in the combined RAVE value.
    ngram_size : int
        N-gram size for diversity computation.
    """

    def __init__(
        self,
        equivalence_parameter: float = 1000.0,
        diversity_weight: float = 0.3,
        ngram_size: int = 3,
    ) -> None:
        self.equivalence_parameter = equivalence_parameter
        self.diversity_weight = diversity_weight
        self.ngram_size = ngram_size
        self._rave_visits: DefaultDict[int, int] = collections.defaultdict(int)
        self._rave_values: DefaultDict[int, float] = collections.defaultdict(float)
        self._rave_diversity: DefaultDict[int, float] = (
            collections.defaultdict(float)
        )
        self._rave_diversity_counts: DefaultDict[int, int] = (
            collections.defaultdict(int)
        )
        self._all_sequences: List[List[int]] = []
        logger.debug(
            "RAVEDiversity initialised: k=%.0f, d_w=%.2f, ngram=%d",
            equivalence_parameter,
            diversity_weight,
            ngram_size,
        )

    def _update_rave_diversity(
        self,
        token_id: int,
        sequence: List[int],
    ) -> None:
        """Update RAVE diversity statistics for a token.

        Computes the mean Jaccard distance of the sequence containing
        *token_id* to all previously recorded sequences, and updates
        the running average.

        Parameters
        ----------
        token_id : int
            The token whose diversity statistics are updated.
        sequence : list of int
            The full sequence from the rollout.
        """
        if not self._all_sequences:
            div_score = 1.0
        else:
            seq_ngrams = _extract_ngrams_set(sequence, self.ngram_size)
            if not seq_ngrams:
                div_score = 1.0
            else:
                distances = [
                    _jaccard_distance(
                        seq_ngrams,
                        _extract_ngrams_set(prev, self.ngram_size),
                    )
                    for prev in self._all_sequences
                ]
                div_score = float(np.mean(distances)) if distances else 1.0

        self._rave_diversity[token_id] += div_score
        self._rave_diversity_counts[token_id] += 1

    def _rave_diversity_score(self, token_id: int) -> float:
        """Get the average diversity RAVE score for a token.

        Parameters
        ----------
        token_id : int

        Returns
        -------
        float
            Average diversity score, or 1.0 if no data.
        """
        count = self._rave_diversity_counts.get(token_id, 0)
        if count == 0:
            return 1.0
        return self._rave_diversity[token_id] / count

    def _combined_rave_value(
        self,
        token_id: int,
        node_q: float,
        node_visits: int,
    ) -> float:
        """Compute the combined RAVE value blending quality and diversity.

        The blending uses the standard RAVE beta schedule:

            Q_combined = (1 - beta) * Q_node
                         + beta * ((1 - d_w) * Q_rave + d_w * D_rave)

        Parameters
        ----------
        token_id : int
        node_q : float
            Standard Q-value at the node.
        node_visits : int
            Visit count of the node.

        Returns
        -------
        float
        """
        k = self.equivalence_parameter
        beta = math.sqrt(k / (3.0 * node_visits + k))

        # Standard RAVE value
        rave_v = self._rave_visits.get(token_id, 0)
        if rave_v == 0:
            rave_q = 0.0
        else:
            rave_q = self._rave_values[token_id] / rave_v

        # Diversity RAVE
        div_rave = self._rave_diversity_score(token_id)

        blended_rave = (
            (1.0 - self.diversity_weight) * rave_q
            + self.diversity_weight * div_rave
        )

        return (1.0 - beta) * node_q + beta * blended_rave

    def select_with_rave(
        self,
        node: MCTSNode,
        exploration_constant: float,
    ) -> MCTSNode:
        """Select the best child of *node* using diversity-aware RAVE.

        Parameters
        ----------
        node : MCTSNode
            Parent node whose children are being evaluated.
        exploration_constant : float
            UCT exploration constant *C*.

        Returns
        -------
        MCTSNode
            The child with the highest RAVE-augmented score.

        Raises
        ------
        ValueError
            If *node* has no children.
        """
        if not node.children:
            raise ValueError("Cannot select from a node with no children.")

        best: Optional[MCTSNode] = None
        best_score = _NEG_INF
        parent_visits = node.visit_count

        for child in node.children.values():
            combined_q = self._combined_rave_value(
                child.token_id, child.q_value, child.visit_count
            )
            exploration = (
                exploration_constant
                * child.prior_probability
                * math.sqrt(max(parent_visits, 1))
                / (1.0 + child.visit_count)
            )
            score = combined_q + exploration
            if score > best_score:
                best_score = score
                best = child

        assert best is not None
        return best

    def backpropagate_with_rave(
        self,
        node: MCTSNode,
        value: float,
        rollout_tokens: List[int],
    ) -> None:
        """Backpropagate value and update RAVE diversity statistics.

        Updates standard RAVE values for all tokens in the rollout, then
        updates diversity RAVE statistics for each unique token.

        Parameters
        ----------
        node : MCTSNode
            The leaf node from which to backpropagate.
        value : float
            The simulation value.
        rollout_tokens : list of int
            Tokens generated during the rollout.
        """
        # Standard RAVE update for all rollout tokens
        seen: Set[int] = set()
        for t in rollout_tokens:
            if t not in seen:
                self._rave_visits[t] += 1
                self._rave_values[t] += value
                seen.add(t)

        # Diversity RAVE update
        full_seq = node.get_sequence() + rollout_tokens
        for t in seen:
            self._update_rave_diversity(t, full_seq)

        # Record sequence
        self._all_sequences.append(full_seq)

        # Standard backpropagation up the tree
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current._sum_squared_value += value * value
            current = current.parent

        logger.debug(
            "RAVEDiversity backpropagated value=%.4f with %d rollout tokens",
            value,
            len(rollout_tokens),
        )


# =========================================================================
# ParallelMCTS — parallel MCTS with multiple parallelisation modes
# =========================================================================


class ParallelMCTS:
    """Parallel MCTS with root parallelisation.

    Runs multiple independent MCTS trees from the same root, then
    aggregates results.  Supports three parallelisation modes:

    * **root** — run *n_workers* independent MCTS trees and merge their
      sequence recommendations.
    * **tree** — single tree with virtual loss to encourage divergent
      selection paths (simulates thread-level parallelism).
    * **leaf** — expand and evaluate multiple leaves per simulation step
      to amortise overhead.

    Parameters
    ----------
    config : MCTSConfig
        MCTS configuration shared across all workers.
    n_workers : int
        Number of parallel workers (trees in root mode).
    parallelization_mode : str
        One of ``'root'``, ``'tree'``, or ``'leaf'``.
    """

    def __init__(
        self,
        config: MCTSConfig,
        n_workers: int = 4,
        parallelization_mode: str = "root",
    ) -> None:
        if parallelization_mode not in ("root", "tree", "leaf"):
            raise ValueError(
                f"Unknown parallelization_mode: {parallelization_mode!r}; "
                "expected 'root', 'tree', or 'leaf'."
            )
        self.config = config
        self.n_workers = max(1, n_workers)
        self.parallelization_mode = parallelization_mode
        self._trees: List[MCTSTree] = []
        logger.debug(
            "ParallelMCTS initialised: n_workers=%d, mode=%s",
            self.n_workers,
            parallelization_mode,
        )

    # -- root parallelisation -----------------------------------------------

    def _root_parallel(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_simulations: int,
    ) -> List[MCTSTree]:
        """Run independent MCTS trees for root parallelisation.

        Each worker runs ``n_simulations // n_workers`` simulations on its
        own tree.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        n_simulations : int
            Total simulation budget.

        Returns
        -------
        list of MCTSTree
            One tree per worker.
        """
        sims_per_worker = max(1, n_simulations // self.n_workers)
        trees: List[MCTSTree] = []

        for worker_idx in range(self.n_workers):
            worker_config = copy.copy(self.config)
            # Vary exploration slightly for diversity across workers
            worker_config.exploration_constant = (
                self.config.exploration_constant
                * (0.8 + 0.4 * worker_idx / max(self.n_workers - 1, 1))
            )
            if self.config.seed is not None:
                np.random.seed(self.config.seed + worker_idx)

            tree = MCTSTree(prompt_ids, worker_config)
            decoder = MCTSDecoding(worker_config)
            decoder._tree = tree

            # Run simulations
            for sim_idx in range(sims_per_worker):
                selected = tree.select(tree.root)

                if not selected.is_terminal and not selected._expanded:
                    expanded = tree.expand(selected, logit_source)
                else:
                    expanded = selected

                prefix = prompt_ids + expanded.get_sequence()
                value_fn = _create_value_function(worker_config)
                value = value_fn.evaluate(prefix, logit_source)

                tree.backpropagate(expanded, value)
                tree._total_simulations += 1

            trees.append(tree)
            logger.debug(
                "Root-parallel worker %d/%d completed %d simulations",
                worker_idx + 1,
                self.n_workers,
                sims_per_worker,
            )

        return trees

    # -- tree parallelisation helpers ---------------------------------------

    def _tree_parallel_select(
        self,
        tree: MCTSTree,
        n_select: int,
    ) -> List[MCTSNode]:
        """Select multiple leaves with virtual loss for tree parallelism.

        After selecting each leaf, applies virtual loss so subsequent
        selections are pushed toward different paths.

        Parameters
        ----------
        tree : MCTSTree
        n_select : int
            Number of leaves to select.

        Returns
        -------
        list of MCTSNode
            Selected leaf nodes.
        """
        selected: List[MCTSNode] = []
        for _ in range(n_select):
            leaf = tree.select(tree.root)
            self._apply_virtual_loss(leaf, self.config.virtual_loss)
            selected.append(leaf)
        return selected

    def _apply_virtual_loss(
        self,
        node: MCTSNode,
        loss_value: float,
    ) -> None:
        """Apply virtual loss along the path from *node* to root.

        Parameters
        ----------
        node : MCTSNode
        loss_value : float
            Virtual loss magnitude (unused beyond counting; the MCTSNode
            tracks ``virtual_loss_count``).
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.virtual_loss_count += 1
            current = current.parent

    def _remove_virtual_loss(
        self,
        node: MCTSNode,
        loss_value: float,
    ) -> None:
        """Remove virtual loss along the path from *node* to root.

        Parameters
        ----------
        node : MCTSNode
        loss_value : float
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.virtual_loss_count = max(0, current.virtual_loss_count - 1)
            current = current.parent

    # -- aggregation --------------------------------------------------------

    def _aggregate_trees(
        self,
        trees: List[MCTSTree],
        n_sequences: int,
    ) -> List[TokenSequence]:
        """Aggregate sequences from multiple trees with diversity filtering.

        Collects top sequences from each tree, then applies diversity-aware
        selection to produce the final set.

        Parameters
        ----------
        trees : list of MCTSTree
        n_sequences : int

        Returns
        -------
        list of TokenSequence
        """
        all_sequences: List[Tuple[List[int], float]] = []

        for tree in trees:
            seqs = tree.top_k_sequences(n_sequences)
            for seq in seqs:
                # Estimate quality as normalised visit-path score
                score = 0.0
                node = tree.root
                for token_id in seq:
                    if token_id in node.children:
                        child = node.children[token_id]
                        score += child.q_value
                        node = child
                    else:
                        break
                if seq:
                    score /= len(seq)
                all_sequences.append((seq, score))

        # Sort by score
        all_sequences.sort(key=lambda x: x[1], reverse=True)

        # Diversity-aware selection
        selected: List[TokenSequence] = []
        used_ngrams: Set[Tuple[int, ...]] = set()

        for seq, _score in all_sequences:
            if len(selected) >= n_sequences:
                break

            seq_ngrams = _extract_ngrams_set(seq, 3)
            if selected and seq_ngrams:
                overlap = len(seq_ngrams & used_ngrams) / max(len(seq_ngrams), 1)
                if overlap > 0.8:
                    continue

            selected.append(seq)
            if seq_ngrams:
                used_ngrams.update(seq_ngrams)

        return selected

    def _merge_statistics(
        self,
        trees: List[MCTSTree],
    ) -> Dict[str, Any]:
        """Merge tree statistics from all parallel workers.

        Parameters
        ----------
        trees : list of MCTSTree

        Returns
        -------
        dict
            Merged statistics including total simulations, total nodes,
            per-tree details, and combined depth profile.
        """
        total_sims = 0
        total_nodes = 0
        depth_profile: DefaultDict[int, int] = collections.defaultdict(int)
        per_tree: List[Dict[str, Any]] = []

        for i, tree in enumerate(trees):
            stats = tree.tree_statistics()
            total_sims += stats.get("total_simulations", 0)
            total_nodes += stats.get("total_nodes", 0)
            per_tree.append({"worker": i, **stats})

            # Merge depth profiles
            for depth, count in stats.get("depth_profile", {}).items():
                depth_profile[depth] += count

        return {
            "total_simulations": total_sims,
            "total_nodes": total_nodes,
            "n_workers": len(trees),
            "depth_profile": dict(depth_profile),
            "per_tree": per_tree,
        }

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: int = 5,
        n_simulations: Optional[int] = None,
    ) -> List[TokenSequence]:
        """Generate diverse sequences using parallel MCTS.

        Parameters
        ----------
        logit_source : LogitSource
            Callable ``(List[List[int]]) -> np.ndarray`` returning logits.
        prompt_ids : list of int
            Token IDs of the prompt.
        n_sequences : int
            Number of sequences to generate.
        n_simulations : int, optional
            Total simulation budget.  Defaults to ``config.n_simulations``.

        Returns
        -------
        list of TokenSequence
            Generated token sequences (without prompt).
        """
        n_sims = n_simulations or self.config.n_simulations
        start_time = time.monotonic()

        if self.parallelization_mode == "root":
            trees = self._root_parallel(logit_source, prompt_ids, n_sims)
            self._trees = trees
            sequences = self._aggregate_trees(trees, n_sequences)

        elif self.parallelization_mode == "tree":
            tree = MCTSTree(prompt_ids, self.config)
            value_fn = _create_value_function(self.config)
            batches = max(1, n_sims // self.n_workers)

            for batch_idx in range(batches):
                # Select n_workers leaves with virtual loss
                leaves = self._tree_parallel_select(tree, self.n_workers)

                for leaf in leaves:
                    # Expand
                    if not leaf.is_terminal and not leaf._expanded:
                        expanded = tree.expand(leaf, logit_source)
                    else:
                        expanded = leaf

                    # Evaluate
                    prefix = prompt_ids + expanded.get_sequence()
                    value = value_fn.evaluate(prefix, logit_source)

                    # Backpropagate
                    tree.backpropagate(expanded, value)
                    tree._total_simulations += 1

                # Remove virtual losses
                for leaf in leaves:
                    self._remove_virtual_loss(leaf, self.config.virtual_loss)

            self._trees = [tree]
            sequences = tree.top_k_sequences(n_sequences)

        else:  # leaf
            tree = MCTSTree(prompt_ids, self.config)
            value_fn = _create_value_function(self.config)

            for sim_idx in range(n_sims):
                selected = tree.select(tree.root)

                if not selected.is_terminal and not selected._expanded:
                    expanded = tree.expand(selected, logit_source)
                else:
                    expanded = selected

                prefix = prompt_ids + expanded.get_sequence()
                value = value_fn.evaluate(prefix, logit_source)

                tree.backpropagate(expanded, value)
                tree._total_simulations += 1

            self._trees = [tree]
            sequences = tree.top_k_sequences(n_sequences)

        elapsed = time.monotonic() - start_time
        merged = self._merge_statistics(self._trees)
        logger.info(
            "ParallelMCTS (%s) generated %d sequences in %.2fs, "
            "total_sims=%d, total_nodes=%d",
            self.parallelization_mode,
            len(sequences),
            elapsed,
            merged["total_simulations"],
            merged["total_nodes"],
        )

        return sequences


# =========================================================================
# NeuralValueMCTS — MCTS with neural value network integration
# =========================================================================


class NeuralValueMCTS(MCTSDecoding):
    """MCTS with neural value network integration.

    Uses a learned value function (simulated here as a configurable
    scoring function) to evaluate leaf nodes instead of (or in addition
    to) random rollouts.  The final leaf value is a weighted blend of the
    neural value estimate and the rollout value.

    Parameters
    ----------
    config : MCTSConfig
        MCTS configuration.
    value_weight : float
        Weight for the neural value estimate in the blend.
    rollout_weight : float
        Weight for the rollout value in the blend.
    score_fn : callable, optional
        Custom scoring function ``(List[int]) -> float`` used as the
        neural value proxy.  If ``None``, a default length-normalised
        log-probability scorer is used.
    """

    def __init__(
        self,
        config: MCTSConfig,
        value_weight: float = 0.5,
        rollout_weight: float = 0.5,
        score_fn: Optional[Callable[[List[int]], float]] = None,
    ) -> None:
        super().__init__(config)
        self.value_weight = value_weight
        self.rollout_weight = rollout_weight
        self._score_fn = score_fn
        logger.debug(
            "NeuralValueMCTS initialised: v_w=%.2f, r_w=%.2f",
            value_weight,
            rollout_weight,
        )

    def _neural_evaluate(
        self,
        prefix: List[int],
        logit_source: LogitSource,
    ) -> float:
        """Evaluate a prefix using the neural value proxy.

        If a custom ``score_fn`` was provided, it is called directly.
        Otherwise, computes a length-normalised log-probability score.

        Parameters
        ----------
        prefix : list of int
            Token sequence to evaluate.
        logit_source : LogitSource
            Source of next-token logits.

        Returns
        -------
        float
            Neural value estimate.
        """
        if self._score_fn is not None:
            return self._score_fn(prefix)

        return self._score_sequence(prefix, logit_source)

    def _blended_value(
        self,
        neural_value: float,
        rollout_value: float,
    ) -> float:
        """Blend neural value and rollout value.

        Parameters
        ----------
        neural_value : float
        rollout_value : float

        Returns
        -------
        float
            Weighted combination:
            ``value_weight * neural + rollout_weight * rollout``.
        """
        total_weight = self.value_weight + self.rollout_weight
        if total_weight <= 0:
            return 0.0
        return (
            self.value_weight * neural_value
            + self.rollout_weight * rollout_value
        ) / total_weight

    def _feature_extract(self, prefix: List[int]) -> np.ndarray:
        """Extract simple features from a token prefix.

        Features include length, unique-token ratio, n-gram diversity,
        and basic distributional statistics.  These can be used by a
        learned value function.

        Parameters
        ----------
        prefix : list of int

        Returns
        -------
        np.ndarray
            Feature vector of shape ``(n_features,)``.
        """
        if not prefix:
            return np.zeros(5, dtype=np.float64)

        length = len(prefix)
        unique_ratio = len(set(prefix)) / max(length, 1)

        # N-gram diversity
        bigrams = _extract_ngrams_set(prefix, 2)
        trigrams = _extract_ngrams_set(prefix, 3)
        bigram_diversity = len(bigrams) / max(length - 1, 1)
        trigram_diversity = len(trigrams) / max(length - 2, 1)

        # Token distribution entropy
        counts = np.array(
            list(collections.Counter(prefix).values()),
            dtype=np.float64,
        )
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs + _LOG_EPS)))

        return np.array(
            [length, unique_ratio, bigram_diversity, trigram_diversity, entropy],
            dtype=np.float64,
        )

    def _score_sequence(
        self,
        prefix: List[int],
        logit_source: LogitSource,
    ) -> float:
        """Score a sequence using length-normalised log-probability.

        Used as the default neural value proxy when no custom ``score_fn``
        is provided.

        Parameters
        ----------
        prefix : list of int
        logit_source : LogitSource

        Returns
        -------
        float
            Normalised log-probability score in a reasonable range.
        """
        if len(prefix) < 2:
            return 0.0

        log_prob_sum = 0.0
        n_scored = 0

        # Score each token given its prefix
        for i in range(1, min(len(prefix), 50)):
            context = prefix[:i]
            logits = logit_source([context])
            if logits.ndim > 1:
                logits = logits[0]
            logits = np.asarray(logits, dtype=np.float64).ravel()

            log_probs = _log_softmax(logits)
            target = prefix[i] if i < len(prefix) else 0
            if target < len(log_probs):
                log_prob_sum += log_probs[target]
                n_scored += 1

        if n_scored == 0:
            return 0.0

        # Length-normalised, then sigmoid-like squashing to [0, 1]
        normalised = log_prob_sum / n_scored
        return 1.0 / (1.0 + math.exp(-normalised - 2.0))

    def simulate_with_neural(
        self,
        node: MCTSNode,
        logit_source: LogitSource,
        tree: MCTSTree,
    ) -> float:
        """Simulate using blended neural and rollout values.

        Performs a standard rollout and combines the result with a
        neural value estimate.

        Parameters
        ----------
        node : MCTSNode
            The leaf node to evaluate.
        logit_source : LogitSource
        tree : MCTSTree

        Returns
        -------
        float
            Blended value estimate.
        """
        prefix = tree.root_tokens + node.get_sequence()

        # Neural value
        neural_val = self._neural_evaluate(prefix, logit_source)

        # Rollout value
        rollout_val = self._run_simulation(node, logit_source, tree)

        blended = self._blended_value(neural_val, rollout_val)

        logger.debug(
            "NeuralValueMCTS: neural=%.4f, rollout=%.4f, blended=%.4f",
            neural_val,
            rollout_val,
            blended,
        )
        return blended

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: Optional[int] = None,
    ) -> List[TokenSequence]:
        """Generate diverse sequences using neural-value-augmented MCTS.

        Overrides the base ``generate`` to use ``simulate_with_neural``
        instead of pure rollout simulation.

        Parameters
        ----------
        logit_source : LogitSource
            Callable ``(List[List[int]]) -> np.ndarray`` returning logits.
        prompt_ids : list of int
            Token IDs of the prompt.
        n_sequences : int, optional
            Number of sequences to generate.

        Returns
        -------
        list of TokenSequence
            Generated token sequences (without prompt).
        """
        self._generation_start_time = time.monotonic()

        if self.mcts_config.seed is not None:
            np.random.seed(self.mcts_config.seed)

        n_seq = n_sequences or self.mcts_config.n_sequences

        logger.info(
            "Starting NeuralValueMCTS: n_simulations=%d, n_sequences=%d, "
            "value_weight=%.2f, rollout_weight=%.2f",
            self.mcts_config.n_simulations,
            n_seq,
            self.value_weight,
            self.rollout_weight,
        )

        # Build tree
        if self.mcts_config.reuse_tree and self._tree is not None:
            tree = self._tree
        else:
            tree = MCTSTree(prompt_ids, self.mcts_config)
            self._tree = tree

        n_sims = self.mcts_config.n_simulations

        for sim_idx in range(n_sims):
            sim_start = time.monotonic()

            selected = tree.select(tree.root)

            if self.mcts_config.virtual_loss > 0:
                tree._apply_virtual_loss(selected)

            try:
                if not selected.is_terminal and not selected._expanded:
                    expanded = tree.expand(selected, logit_source)
                else:
                    expanded = selected

                # Use neural + rollout blended value
                value = self.simulate_with_neural(
                    expanded, logit_source, tree
                )
                self._total_rollouts += 1

                tree.backpropagate(expanded, value)

            finally:
                if self.mcts_config.virtual_loss > 0:
                    tree._remove_virtual_loss(selected)

            sim_time = time.monotonic() - sim_start
            tree._simulation_times.append(sim_time)
            tree._total_simulations += 1

            if (sim_idx + 1) % max(1, n_sims // 10) == 0:
                stats = tree.tree_statistics()
                self._log_search_progress(sim_idx + 1, stats)

        # Extract diverse sequences
        sequences = self._extract_diverse_sequences(tree, n_seq)

        elapsed = time.monotonic() - self._generation_start_time
        logger.info(
            "NeuralValueMCTS complete: %d sequences in %.2fs, "
            "%d total rollouts",
            len(sequences),
            elapsed,
            self._total_rollouts,
        )

        return sequences


# =========================================================================
# Module-level helper functions for diversity-aware MCTS
# =========================================================================


def uct_diversity_score(
    node: MCTSNode,
    parent: MCTSNode,
    sequences: List[List[int]],
    diversity_weight: float = 0.5,
    ngram_size: int = 3,
) -> float:
    """Compute UCT score with a diversity bonus term.

    UCT_div = Q/N + C_default * sqrt(ln(N_parent)/N)
              + diversity_weight * mean_jaccard_distance

    Parameters
    ----------
    node : MCTSNode
        The child node being scored.
    parent : MCTSNode
        The parent node.
    sequences : list of list of int
        Reference sequences for diversity computation.
    diversity_weight : float
        Weight for the diversity term.
    ngram_size : int
        N-gram size for Jaccard distance.

    Returns
    -------
    float
        Diversity-augmented UCT score.
    """
    if node.visit_count == 0:
        return float("inf")

    exploitation = node.q_value
    parent_visits = parent.visit_count
    exploration = math.sqrt(
        math.log(max(parent_visits, 1)) / node.visit_count
    )

    # Diversity bonus
    if not sequences:
        div_bonus = 1.0
    else:
        node_seq = node.get_sequence()
        node_ngrams = _extract_ngrams_set(node_seq, ngram_size)
        if not node_ngrams:
            div_bonus = 1.0
        else:
            distances = [
                _jaccard_distance(
                    node_ngrams,
                    _extract_ngrams_set(s, ngram_size),
                )
                for s in sequences
            ]
            div_bonus = float(np.mean(distances)) if distances else 1.0

    return exploitation + exploration + diversity_weight * div_bonus


def progressive_widening_threshold(
    visit_count: int,
    widening_constant: float = 1.0,
    widening_exponent: float = 0.5,
) -> int:
    """Compute the maximum number of children under progressive widening.

    max_children = floor(C_w * (N + 1)^alpha)

    Parameters
    ----------
    visit_count : int
        Number of visits to the node.
    widening_constant : float
        Multiplier *C_w*.
    widening_exponent : float
        Exponent *alpha*.

    Returns
    -------
    int
        Maximum number of children, at least 1.
    """
    return max(1, int(math.floor(
        widening_constant * (visit_count + 1) ** widening_exponent
    )))


def diversity_shaped_reward(
    sequence: List[int],
    reference_sequences: List[List[int]],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a diversity-shaped reward for a sequence.

    Combines n-gram novelty, vocabulary coverage, and information content
    relative to a set of reference sequences.

    Parameters
    ----------
    sequence : list of int
        The sequence to score.
    reference_sequences : list of list of int
        Previously generated sequences for novelty computation.
    weights : dict, optional
        Weights for ``'novelty'``, ``'coverage'``, and ``'info'``.
        Defaults to equal weights.

    Returns
    -------
    float
        Diversity-shaped reward value.
    """
    if weights is None:
        weights = {"novelty": 0.4, "coverage": 0.3, "info": 0.3}

    # Novelty
    seq_ngrams = _extract_ngrams_set(sequence, 3)
    if reference_sequences and seq_ngrams:
        ref_ngrams: Set[Tuple[int, ...]] = set()
        for ref in reference_sequences:
            ref_ngrams.update(_extract_ngrams_set(ref, 3))
        novel = seq_ngrams - ref_ngrams
        novelty = len(novel) / max(len(seq_ngrams), 1)
    else:
        novelty = 1.0

    # Coverage
    if sequence:
        coverage = min(1.0, len(set(sequence)) / _DEFAULT_VOCAB_SIZE)
    else:
        coverage = 0.0

    # Information content
    if sequence:
        counts = np.array(
            list(collections.Counter(sequence).values()),
            dtype=np.float64,
        )
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs + _LOG_EPS)))
        max_entropy = math.log(max(len(counts), 1) + _LOG_EPS)
        info = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        info = 0.0

    return (
        weights.get("novelty", 0.4) * novelty
        + weights.get("coverage", 0.3) * coverage
        + weights.get("info", 0.3) * info
    )


def aggregate_mcts_trees(
    trees: List[MCTSTree],
    method: str = "visit_weighted",
) -> Dict[str, Any]:
    """Aggregate node statistics from multiple MCTS trees.

    Supports two aggregation methods:

    * ``'visit_weighted'`` — merge by summing visit counts and weighting
      Q-values proportionally.
    * ``'max'`` — take the maximum Q-value across trees for each token.

    Parameters
    ----------
    trees : list of MCTSTree
        Trees to aggregate.
    method : str
        Aggregation method: ``'visit_weighted'`` or ``'max'``.

    Returns
    -------
    dict
        Merged statistics keyed by token ID, with ``'visit_count'``,
        ``'q_value'``, and ``'source_trees'`` for each token.
    """
    if method not in ("visit_weighted", "max"):
        raise ValueError(
            f"Unknown aggregation method: {method!r}; "
            "expected 'visit_weighted' or 'max'."
        )

    merged: Dict[int, Dict[str, Any]] = {}

    for tree_idx, tree in enumerate(trees):
        for token_id, child in tree.root.children.items():
            if token_id not in merged:
                merged[token_id] = {
                    "visit_count": 0,
                    "total_value": 0.0,
                    "q_value": 0.0,
                    "source_trees": [],
                }

            entry = merged[token_id]
            entry["source_trees"].append(tree_idx)

            if method == "visit_weighted":
                entry["visit_count"] += child.visit_count
                entry["total_value"] += child.total_value
                if entry["visit_count"] > 0:
                    entry["q_value"] = (
                        entry["total_value"] / entry["visit_count"]
                    )
            else:  # max
                entry["visit_count"] += child.visit_count
                entry["q_value"] = max(entry["q_value"], child.q_value)

    return merged


def virtual_loss_update(
    node: MCTSNode,
    loss_value: float,
    add: bool = True,
) -> None:
    """Apply or remove virtual loss along the path from *node* to root.

    Parameters
    ----------
    node : MCTSNode
        The leaf node.
    loss_value : float
        Virtual loss magnitude (used for logging; the actual mechanism
        increments/decrements ``virtual_loss_count``).
    add : bool
        If ``True``, apply virtual loss; if ``False``, remove it.
    """
    current: Optional[MCTSNode] = node
    while current is not None:
        if add:
            current.virtual_loss_count += 1
        else:
            current.virtual_loss_count = max(0, current.virtual_loss_count - 1)
        current = current.parent

    logger.debug(
        "Virtual loss %s (value=%.2f) along path of length %d",
        "applied" if add else "removed",
        loss_value,
        node.depth + 1,
    )


def compute_rave_value(
    node: MCTSNode,
    action: int,
    rave_table: RAVETable,
    beta: Optional[float] = None,
) -> float:
    """Compute RAVE-blended value for a node-action pair.

    If *beta* is not provided, it is computed from the RAVE table's
    equivalence parameter and the node's visit count.

    Parameters
    ----------
    node : MCTSNode
        The node at which the action is being evaluated.
    action : int
        Token ID (action) to evaluate.
    rave_table : RAVETable
        RAVE statistics table.
    beta : float, optional
        RAVE blending parameter.  If ``None``, computed automatically.

    Returns
    -------
    float
        Blended value: ``(1 - beta) * Q_node + beta * Q_rave``.
    """
    if beta is None:
        beta = rave_table.beta(node.visit_count)
    return rave_table.blended_value(node.q_value, action, node.visit_count)
