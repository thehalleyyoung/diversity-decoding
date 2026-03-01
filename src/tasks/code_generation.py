"""
Code Generation task domain for the Diversity Decoding Arena.

Implements comprehensive code generation evaluation including syntax checking,
complexity analysis, style scoring, test execution, and diversity measurement
across multiple programming languages and task types.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import copy
import difflib
import enum
import hashlib
import io
import itertools
import keyword
import math
import multiprocessing
import operator
import os
import re
import signal
import statistics
import string
import sys
import textwrap
import threading
import time
import tokenize
import traceback
import types as builtin_types
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from src.tasks.base import (
    GenerationTask,
    PromptDataset,
    TaskConfig,
    TaskConstraint,
    TaskEvaluator,
    TaskPrompt,
)
from src.types import TaskDomain


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProgrammingLanguage(enum.Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    TYPESCRIPT = "typescript"


class CodeTaskType(enum.Enum):
    """Types of code-generation tasks."""
    FUNCTION_SYNTHESIS = "function_synthesis"
    TEST_GENERATION = "test_generation"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    DOCUMENTATION = "documentation"
    CODE_COMPLETION = "code_completion"
    API_DESIGN = "api_design"


class CodeComplexity(enum.Enum):
    """Difficulty / complexity tiers."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class CodeTestCase:
    """A single test-case for a code-generation prompt."""
    input_args: List[Any] = field(default_factory=list)
    expected_output: Any = None
    description: str = ""
    is_edge_case: bool = False


@dataclass
class CodePrompt(TaskPrompt):
    """Prompt specialised for code generation."""
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    function_signature: str = ""
    docstring: str = ""
    test_cases: List[CodeTestCase] = field(default_factory=list)
    expected_complexity: CodeComplexity = CodeComplexity.MEDIUM
    input_types: List[str] = field(default_factory=list)
    output_type: str = ""


@dataclass
class CodeGenerationConfig(TaskConfig):
    """Configuration for the code-generation task domain."""
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    task_type: CodeTaskType = CodeTaskType.FUNCTION_SYNTHESIS
    complexity: CodeComplexity = CodeComplexity.MEDIUM
    allow_imports: bool = True
    max_functions: int = 10
    require_docstrings: bool = True
    require_type_hints: bool = False
    test_framework: str = "pytest"


# ---------------------------------------------------------------------------
# Helpers – safe execution sandbox
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised when sandboxed execution exceeds its time budget."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _TimeoutError("Execution timed out")


_SANDBOX_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "vars": vars,
    "zip": zip,
    "__import__": __import__,
    "None": None,
    "True": True,
    "False": False,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "ArithmeticError": ArithmeticError,
}


def _safe_exec(code: str, timeout: int = 5) -> Tuple[Dict[str, Any], str, str]:
    """Execute *code* in a restricted sandbox with a timeout.

    Returns (namespace, stdout, stderr).
    """
    namespace: Dict[str, Any] = {"__builtins__": _SANDBOX_BUILTINS}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    old_alarm = None
    try:
        if hasattr(signal, "SIGALRM"):
            old_alarm = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
    except _TimeoutError:
        stderr_capture.write("TimeoutError: execution exceeded time limit\n")
    except Exception as exc:
        stderr_capture.write(f"{type(exc).__name__}: {exc}\n")
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)

    return namespace, stdout_capture.getvalue(), stderr_capture.getvalue()


def _safe_call(
    func: Callable[..., Any],
    args: List[Any],
    timeout: int = 3,
) -> Tuple[Any, Optional[str]]:
    """Call *func* with *args* inside the sandbox.  Returns (result, error_msg)."""
    result_holder: Dict[str, Any] = {"value": None, "error": None}

    def _target() -> None:
        try:
            result_holder["value"] = func(*args)
        except Exception as exc:
            result_holder["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return None, "TimeoutError: function call exceeded time limit"
    return result_holder["value"], result_holder["error"]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_python(code: str) -> Optional[ast.Module]:
    """Return the AST for *code* or ``None`` on parse failure."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _all_nodes(tree: ast.AST) -> List[ast.AST]:
    """Flat list of every node in *tree*."""
    return list(ast.walk(tree))


def _function_defs(tree: ast.Module) -> List[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _class_defs(tree: ast.Module) -> List[ast.ClassDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Return maximum nesting depth of control-flow statements."""
    nesting_types = (
        ast.If, ast.For, ast.While, ast.With, ast.Try,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    )
    max_d = depth if isinstance(node, nesting_types) else depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, nesting_types) else depth
        max_d = max(max_d, _max_nesting(child, child_depth))
    return max_d


def _count_branches(tree: ast.Module) -> int:
    """Count decision points (if/elif/for/while/except/with/and/or/ternary)."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            count += 1
        elif isinstance(node, ast.ExceptHandler):
            count += 1
        elif isinstance(node, ast.BoolOp):
            count += len(node.values) - 1
        elif isinstance(node, ast.IfExp):
            count += 1
    return count


def _lines_of_code(code: str) -> Tuple[int, int, int]:
    """Return (total_lines, code_lines, comment_lines)."""
    lines = code.splitlines()
    total = len(lines)
    comment = sum(1 for l in lines if l.strip().startswith("#"))
    blank = sum(1 for l in lines if not l.strip())
    code_lines = total - comment - blank
    return total, code_lines, comment


def _identifiers(tree: ast.Module) -> List[str]:
    """Return all user-defined identifier names."""
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
            names.extend(a.arg for a in node.args.args)
        elif isinstance(node, ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
    return names


def _ast_fingerprint(tree: ast.AST) -> str:
    """Produce a structural fingerprint by recording node types in pre-order."""
    parts: List[str] = []
    for node in ast.walk(tree):
        parts.append(type(node).__name__)
    return "|".join(parts)


def _node_type_counter(tree: ast.AST) -> Counter[str]:
    ctr: Counter[str] = collections.Counter()
    for node in ast.walk(tree):
        ctr[type(node).__name__] += 1
    return ctr


# ---------------------------------------------------------------------------
# Halstead helpers
# ---------------------------------------------------------------------------

_PYTHON_OPERATORS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,
    ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
}


def _halstead_raw(tree: ast.Module) -> Tuple[Counter[str], Counter[str]]:
    """Return (operator_counts, operand_counts)."""
    operators: Counter[str] = collections.Counter()
    operands: Counter[str] = collections.Counter()
    for node in ast.walk(tree):
        ntype = type(node)
        if ntype in _PYTHON_OPERATORS:
            operators[ntype.__name__] += 1
        elif isinstance(node, ast.Constant):
            operands[repr(node.value)] += 1
        elif isinstance(node, ast.Name):
            operands[node.id] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            operators["FunctionDef"] += 1
            operands[node.name] += 1
        elif isinstance(node, ast.Call):
            operators["Call"] += 1
        elif isinstance(node, ast.Assign):
            operators["Assign"] += 1
        elif isinstance(node, ast.AugAssign):
            operators["AugAssign"] += 1
        elif isinstance(node, ast.Return):
            operators["Return"] += 1
        elif isinstance(node, ast.If):
            operators["If"] += 1
        elif isinstance(node, ast.For):
            operators["For"] += 1
        elif isinstance(node, ast.While):
            operators["While"] += 1
    return operators, operands


# ---------------------------------------------------------------------------
# CodeGenerationTask
# ---------------------------------------------------------------------------

class CodeGenerationTask(GenerationTask):
    """Full-featured code generation task domain."""

    domain = TaskDomain.CODE_GENERATION

    def __init__(self, config: Optional[CodeGenerationConfig] = None) -> None:
        self.config: CodeGenerationConfig = config or CodeGenerationConfig()
        super().__init__(self.config)

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Return a :class:`PromptDataset` containing 60+ code prompts."""
        prompts: List[CodePrompt] = []
        prompts.extend(self._generate_function_synthesis_prompts())
        prompts.extend(self._generate_test_generation_prompts())
        prompts.extend(self._generate_refactoring_prompts())
        prompts.extend(self._generate_bug_fix_prompts())
        prompts.extend(self._generate_documentation_prompts())
        prompts.extend(self._generate_code_completion_prompts())
        prompts.extend(self._generate_api_design_prompts())
        return PromptDataset(prompts=prompts, name="code_generation", version="1.0")

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, prompt: CodePrompt) -> str:  # type: ignore[override]
        """Format *prompt* using language-specific conventions."""
        parts: List[str] = []

        lang = prompt.language
        lang_label = lang.value.capitalize()

        parts.append(f"# {lang_label} Code Generation Task")
        parts.append("")

        if prompt.docstring:
            parts.append(f"## Description\n{prompt.docstring}")
            parts.append("")

        if prompt.function_signature:
            comment_prefix = self._comment_prefix(lang)
            parts.append(f"## Function Signature")
            fence = self._fence_language(lang)
            parts.append(f"```{fence}")
            parts.append(prompt.function_signature)
            parts.append("```")
            parts.append("")

        if prompt.input_types:
            parts.append("## Input Types")
            for it in prompt.input_types:
                parts.append(f"- {it}")
            parts.append("")

        if prompt.output_type:
            parts.append(f"## Return Type\n`{prompt.output_type}`")
            parts.append("")

        if prompt.test_cases:
            parts.append("## Examples")
            for i, tc in enumerate(prompt.test_cases, 1):
                args_str = ", ".join(repr(a) for a in tc.input_args)
                parts.append(
                    f"{i}. `f({args_str})` → `{repr(tc.expected_output)}`"
                    + (f"  ({tc.description})" if tc.description else "")
                )
            parts.append("")

        if prompt.constraints:
            parts.append("## Constraints")
            for c in prompt.constraints:
                parts.append(f"- {c}")
            parts.append("")

        complexity_label = prompt.expected_complexity.value
        parts.append(f"**Difficulty:** {complexity_label}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generations: List[str],
        prompts: List[CodePrompt],  # type: ignore[override]
    ) -> Dict[str, Any]:
        """Score a batch of generated code snippets.

        Returns a dictionary mapping metric names to aggregate scores in [0, 1].
        """
        per_item: List[Dict[str, float]] = []

        for code, prompt in zip(generations, prompts):
            scores: Dict[str, float] = {}

            # --- Syntax ---
            valid, errors = self._syntax_check(code, prompt.language)
            scores["syntax_valid"] = 1.0 if valid else 0.0

            # --- Complexity metrics ---
            if valid and prompt.language == ProgrammingLanguage.PYTHON:
                cx = self._complexity_analysis(code)
                scores["cyclomatic_complexity_norm"] = cx.get("cyclomatic_norm", 0.0)
                scores["loc_norm"] = cx.get("loc_norm", 0.0)
                scores["nesting_depth_norm"] = cx.get("nesting_depth_norm", 0.0)
                scores["cognitive_complexity_norm"] = min(
                    1.0, self._cognitive_complexity(code) / 50.0
                )
            else:
                scores["cyclomatic_complexity_norm"] = 0.0
                scores["loc_norm"] = 0.0
                scores["nesting_depth_norm"] = 0.0
                scores["cognitive_complexity_norm"] = 0.0

            # --- Style ---
            scores["style"] = self._style_score(code, prompt.language)

            # --- Test execution ---
            if prompt.test_cases and prompt.language == ProgrammingLanguage.PYTHON:
                passed, total, _ = self._test_execution(code, prompt.test_cases)
                scores["test_pass_rate"] = passed / max(total, 1)
            else:
                scores["test_pass_rate"] = 0.0

            # --- Documentation quality ---
            scores["documentation"] = self._documentation_quality(code)

            # --- Identifier quality ---
            scores["identifier_quality"] = self._identifier_quality(code)

            # --- Error handling ---
            scores["error_handling"] = self._error_handling_score(code)

            # --- Type annotation coverage ---
            scores["type_annotation_coverage"] = self._type_annotation_coverage(code)

            # --- Function decomposition ---
            scores["function_decomposition"] = self._function_decomposition_score(code)

            # --- Dead code ---
            dead = self._dead_code_detection(code)
            scores["dead_code_penalty"] = max(0.0, 1.0 - len(dead) * 0.1)

            # --- Import analysis ---
            imp = self._import_analysis(code)
            scores["import_cleanliness"] = imp.get("cleanliness", 1.0)

            per_item.append(scores)

        # --- Cross-generation diversity ---
        if len(generations) > 1:
            diversity = self._diversity_of_approaches(generations)
        else:
            diversity = 0.0

        # Aggregate
        agg: Dict[str, Any] = {}
        if per_item:
            all_keys = per_item[0].keys()
            for k in all_keys:
                vals = [s[k] for s in per_item]
                agg[k] = statistics.mean(vals)
        agg["diversity"] = diversity
        agg["per_item"] = per_item
        return agg

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def get_constraints(self) -> List[TaskConstraint]:
        """Return constraints relevant to code generation."""
        constraints: List[TaskConstraint] = []

        constraints.append(TaskConstraint(
            name="syntax_valid",
            description="Generated code must be syntactically valid.",
            weight=1.0,
            required=True,
        ))

        constraints.append(TaskConstraint(
            name="max_length",
            description="Generated code must not exceed 5000 characters.",
            weight=0.5,
            required=False,
            metadata={"max_chars": 5000},
        ))

        constraints.append(TaskConstraint(
            name="max_functions",
            description=f"Code must define at most {self.config.max_functions} functions.",
            weight=0.3,
            required=False,
            metadata={"max_functions": self.config.max_functions},
        ))

        if self.config.require_docstrings:
            constraints.append(TaskConstraint(
                name="docstrings",
                description="Every function must have a docstring.",
                weight=0.4,
                required=False,
            ))

        if self.config.require_type_hints:
            constraints.append(TaskConstraint(
                name="type_hints",
                description="Functions must have type annotations.",
                weight=0.4,
                required=False,
            ))

        if not self.config.allow_imports:
            constraints.append(TaskConstraint(
                name="no_imports",
                description="Code must not import external modules.",
                weight=0.6,
                required=True,
            ))

        constraints.append(TaskConstraint(
            name="complexity_budget",
            description="Cyclomatic complexity should stay below 20.",
            weight=0.3,
            required=False,
            metadata={"max_cyclomatic": 20},
        ))

        return constraints

    # ------------------------------------------------------------------
    # Syntax checking
    # ------------------------------------------------------------------

    def _syntax_check(
        self, code: str, language: ProgrammingLanguage,
    ) -> Tuple[bool, List[str]]:
        """Check *code* for syntax errors.

        Full AST validation is performed for Python.  For other languages a
        set of simple heuristic checks (brace matching, semicolons, etc.) is
        applied.  Returns ``(is_valid, list_of_error_messages)``.
        """
        errors: List[str] = []
        if language == ProgrammingLanguage.PYTHON:
            try:
                ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
            return (len(errors) == 0, errors)

        if language in (ProgrammingLanguage.JAVA, ProgrammingLanguage.CPP,
                        ProgrammingLanguage.RUST, ProgrammingLanguage.GO,
                        ProgrammingLanguage.TYPESCRIPT,
                        ProgrammingLanguage.JAVASCRIPT):
            errors.extend(self._brace_balance_check(code))
            errors.extend(self._paren_balance_check(code))
            errors.extend(self._bracket_balance_check(code))
            if language in (ProgrammingLanguage.JAVA, ProgrammingLanguage.CPP):
                errors.extend(self._semicolon_heuristic(code))
            return (len(errors) == 0, errors)

        return (True, [])

    @staticmethod
    def _brace_balance_check(code: str) -> List[str]:
        depth = 0
        for i, ch in enumerate(code):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth < 0:
                return [f"Unmatched '}}' at position {i}"]
        if depth != 0:
            return [f"Unmatched '{{': {depth} unclosed"]
        return []

    @staticmethod
    def _paren_balance_check(code: str) -> List[str]:
        depth = 0
        for i, ch in enumerate(code):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return [f"Unmatched ')' at position {i}"]
        if depth != 0:
            return [f"Unmatched '(': {depth} unclosed"]
        return []

    @staticmethod
    def _bracket_balance_check(code: str) -> List[str]:
        depth = 0
        for i, ch in enumerate(code):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if depth < 0:
                return [f"Unmatched ']' at position {i}"]
        if depth != 0:
            return [f"Unmatched '[': {depth} unclosed"]
        return []

    @staticmethod
    def _semicolon_heuristic(code: str) -> List[str]:
        """Warn if lines that look like statements lack trailing semicolons."""
        errors: List[str] = []
        stmt_pattern = re.compile(
            r"^\s*(int|float|double|char|string|var|let|const|return|auto)\b"
        )
        for lineno, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue
            if stmt_pattern.match(stripped) and not stripped.endswith((";", "{", "}", ",")):
                errors.append(f"Line {lineno} may be missing a semicolon")
        return errors

    # ------------------------------------------------------------------
    # Complexity analysis
    # ------------------------------------------------------------------

    def _complexity_analysis(self, code: str) -> Dict[str, Any]:
        """Return complexity metrics for Python *code*.

        Keys:
        - cyclomatic: raw cyclomatic complexity
        - cyclomatic_norm: normalised to [0,1] via ``min(cc / 30, 1)``
        - loc, code_lines, comment_lines
        - loc_norm: ``min(code_lines / 200, 1)``
        - nesting_depth, nesting_depth_norm
        - num_functions, num_classes
        """
        tree = _parse_python(code)
        if tree is None:
            return {"cyclomatic": 0, "cyclomatic_norm": 0.0,
                    "loc": 0, "code_lines": 0, "comment_lines": 0,
                    "loc_norm": 0.0, "nesting_depth": 0,
                    "nesting_depth_norm": 0.0, "num_functions": 0,
                    "num_classes": 0}

        cc = 1 + _count_branches(tree)
        total, code_lines, comment_lines = _lines_of_code(code)
        nesting = _max_nesting(tree)
        funcs = _function_defs(tree)
        classes = _class_defs(tree)

        return {
            "cyclomatic": cc,
            "cyclomatic_norm": min(cc / 30.0, 1.0),
            "loc": total,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "loc_norm": min(code_lines / 200.0, 1.0),
            "nesting_depth": nesting,
            "nesting_depth_norm": min(nesting / 8.0, 1.0),
            "num_functions": len(funcs),
            "num_classes": len(classes),
        }

    # ------------------------------------------------------------------
    # Style scoring
    # ------------------------------------------------------------------

    def _style_score(self, code: str, language: ProgrammingLanguage) -> float:
        """Return a style score in [0, 1]."""
        if language == ProgrammingLanguage.PYTHON:
            return self._python_style_score(code)
        return self._generic_style_score(code)

    def _python_style_score(self, code: str) -> float:
        """PEP-8-inspired style heuristics for Python."""
        score = 1.0
        lines = code.splitlines()

        # Line length
        long_lines = sum(1 for l in lines if len(l) > 100)
        if lines:
            score -= 0.1 * min(long_lines / max(len(lines), 1), 1.0)

        # Trailing whitespace
        trailing = sum(1 for l in lines if l != l.rstrip())
        if lines:
            score -= 0.05 * min(trailing / max(len(lines), 1), 1.0)

        # Consistent indentation (spaces, not tabs)
        tab_lines = sum(1 for l in lines if "\t" in l)
        if lines:
            score -= 0.1 * min(tab_lines / max(len(lines), 1), 1.0)

        tree = _parse_python(code)
        if tree is None:
            return max(score * 0.5, 0.0)

        # Naming conventions
        funcs = _function_defs(tree)
        for f in funcs:
            if not re.match(r"^[a-z_][a-z0-9_]*$", f.name):
                score -= 0.02

        classes = _class_defs(tree)
        for c in classes:
            if not re.match(r"^[A-Z][a-zA-Z0-9]*$", c.name):
                score -= 0.02

        # Blank lines around functions
        for f in funcs:
            if hasattr(f, "lineno") and f.lineno > 2:
                prev_line = lines[f.lineno - 2] if f.lineno - 2 < len(lines) else ""
                if prev_line.strip():
                    score -= 0.01

        return max(min(score, 1.0), 0.0)

    def _generic_style_score(self, code: str) -> float:
        """Basic style heuristics applicable to any language."""
        score = 1.0
        lines = code.splitlines()
        if not lines:
            return 0.0
        long = sum(1 for l in lines if len(l) > 120)
        score -= 0.15 * min(long / max(len(lines), 1), 1.0)
        trailing = sum(1 for l in lines if l != l.rstrip())
        score -= 0.05 * min(trailing / max(len(lines), 1), 1.0)
        return max(min(score, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def _test_execution(
        self,
        code: str,
        test_cases: List[CodeTestCase],
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Execute Python *code* and run *test_cases* against it.

        Returns ``(pass_count, total, results_list)``.
        """
        ns, stdout, stderr = _safe_exec(code, timeout=5)
        if stderr and "Error" in stderr:
            return 0, len(test_cases), [
                {"passed": False, "error": stderr} for _ in test_cases
            ]

        # Find the first user-defined function
        callable_names = [
            k for k, v in ns.items()
            if callable(v) and not k.startswith("_") and k not in _SANDBOX_BUILTINS
        ]
        if not callable_names:
            return 0, len(test_cases), [
                {"passed": False, "error": "No callable function found"} for _ in test_cases
            ]

        func = ns[callable_names[0]]
        passed = 0
        results: List[Dict[str, Any]] = []

        for tc in test_cases:
            result_val, err = _safe_call(func, tc.input_args, timeout=3)
            if err is not None:
                results.append({"passed": False, "error": err, "description": tc.description})
                continue

            match = self._values_equal(result_val, tc.expected_output)
            if match:
                passed += 1
            results.append({
                "passed": match,
                "expected": tc.expected_output,
                "actual": result_val,
                "description": tc.description,
            })

        return passed, len(test_cases), results

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Flexible equality comparison tolerant of floats."""
        if a == b:
            return True
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(CodeGenerationTask._values_equal(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(CodeGenerationTask._values_equal(a[k], b[k]) for k in a)
        return False

    # ------------------------------------------------------------------
    # Code similarity (structural, AST-based)
    # ------------------------------------------------------------------

    def _code_similarity(self, code_a: str, code_b: str) -> float:
        """Return structural similarity of two Python snippets in [0,1]."""
        tree_a = _parse_python(code_a)
        tree_b = _parse_python(code_b)
        if tree_a is None or tree_b is None:
            return self._text_similarity(code_a, code_b)

        fp_a = _ast_fingerprint(tree_a)
        fp_b = _ast_fingerprint(tree_b)
        sm = difflib.SequenceMatcher(None, fp_a.split("|"), fp_b.split("|"))
        ast_sim = sm.ratio()

        cnt_a = _node_type_counter(tree_a)
        cnt_b = _node_type_counter(tree_b)
        all_types = set(cnt_a) | set(cnt_b)
        if not all_types:
            return 1.0
        dot = sum(cnt_a.get(t, 0) * cnt_b.get(t, 0) for t in all_types)
        mag_a = math.sqrt(sum(v * v for v in cnt_a.values())) or 1
        mag_b = math.sqrt(sum(v * v for v in cnt_b.values())) or 1
        cos_sim = dot / (mag_a * mag_b)

        return 0.6 * ast_sim + 0.4 * cos_sim

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    # ------------------------------------------------------------------
    # Diversity of approaches
    # ------------------------------------------------------------------

    def _diversity_of_approaches(self, codes: List[str]) -> float:
        """Measure how structurally diverse a set of implementations are.

        Returns a value in [0, 1] where 1 = maximally diverse.
        """
        if len(codes) < 2:
            return 0.0

        sims: List[float] = []
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                sims.append(self._code_similarity(codes[i], codes[j]))

        avg_sim = statistics.mean(sims) if sims else 1.0
        diversity = 1.0 - avg_sim

        # Bonus for different control-flow patterns
        patterns: List[FrozenSet[str]] = []
        for code in codes:
            tree = _parse_python(code)
            if tree:
                cf_nodes = frozenset(
                    type(n).__name__ for n in ast.walk(tree)
                    if isinstance(n, (ast.If, ast.For, ast.While,
                                     ast.ListComp, ast.DictComp,
                                     ast.SetComp, ast.GeneratorExp,
                                     ast.Try, ast.With))
                )
                patterns.append(cf_nodes)

        if len(patterns) >= 2:
            unique_patterns = len(set(patterns))
            pattern_diversity = (unique_patterns - 1) / max(len(patterns) - 1, 1)
            diversity = 0.7 * diversity + 0.3 * pattern_diversity

        return max(min(diversity, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Documentation quality
    # ------------------------------------------------------------------

    def _documentation_quality(self, code: str) -> float:
        """Score documentation quality in [0, 1]."""
        tree = _parse_python(code)
        if tree is None:
            # Fallback: count comment density
            lines = code.splitlines()
            if not lines:
                return 0.0
            comment_lines = sum(1 for l in lines if l.strip().startswith(("#", "//")))
            return min(comment_lines / max(len(lines) * 0.3, 1), 1.0)

        funcs = _function_defs(tree)
        if not funcs:
            # Module-level docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)
                    and isinstance(tree.body[0].value.value, str)):
                return 0.5
            return 0.1

        documented = 0
        quality_sum = 0.0
        for f in funcs:
            ds = ast.get_docstring(f)
            if ds:
                documented += 1
                q = self._docstring_quality(ds, f)
                quality_sum += q

        coverage = documented / len(funcs) if funcs else 0.0
        avg_quality = quality_sum / max(documented, 1)
        return 0.5 * coverage + 0.5 * avg_quality

    def _docstring_quality(self, docstring: str, func: ast.FunctionDef) -> float:
        """Rate the quality of a single docstring."""
        score = 0.0
        # Length adequacy
        words = docstring.split()
        if len(words) >= 3:
            score += 0.2
        if len(words) >= 10:
            score += 0.1

        # Mentions parameters
        params = [a.arg for a in func.args.args if a.arg != "self"]
        if params:
            mentioned = sum(1 for p in params if p in docstring)
            score += 0.3 * (mentioned / len(params))

        # Mentions return
        if any(kw in docstring.lower() for kw in ("return", "returns", "->", "result")):
            score += 0.2

        # Mentions exceptions
        if any(kw in docstring.lower() for kw in ("raise", "raises", "exception", "error")):
            score += 0.1

        # Has example
        if any(kw in docstring.lower() for kw in ("example", ">>>", "e.g.")):
            score += 0.1

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # API design score
    # ------------------------------------------------------------------

    def _api_design_score(self, code: str) -> float:
        """Score the quality of an API design in [0, 1]."""
        tree = _parse_python(code)
        if tree is None:
            return 0.0

        score = 0.0
        classes = _class_defs(tree)
        funcs = _function_defs(tree)
        total_checks = 0

        # Naming consistency
        if funcs:
            snake_count = sum(
                1 for f in funcs if re.match(r"^[a-z_][a-z0-9_]*$", f.name)
            )
            score += 0.15 * (snake_count / len(funcs))
            total_checks += 1

        if classes:
            pascal_count = sum(
                1 for c in classes if re.match(r"^[A-Z][a-zA-Z0-9]*$", c.name)
            )
            score += 0.15 * (pascal_count / len(classes))
            total_checks += 1

        # Method grouping in classes
        for cls in classes:
            methods = [
                n for n in ast.walk(cls)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                has_init = any(m.name == "__init__" for m in methods)
                has_repr = any(m.name in ("__repr__", "__str__") for m in methods)
                public = [m for m in methods if not m.name.startswith("_")]
                private = [m for m in methods if m.name.startswith("_") and m.name != "__init__"]
                score += 0.1 * (1.0 if has_init else 0.0)
                score += 0.05 * (1.0 if has_repr else 0.0)
                score += 0.05 * min(len(public) / max(len(methods), 1), 1.0)
                total_checks += 1

        # Parameter count sanity (functions with ≤5 params preferred)
        if funcs:
            sane = sum(
                1 for f in funcs if len(f.args.args) <= 6
            )
            score += 0.15 * (sane / len(funcs))
            total_checks += 1

        # Docstring coverage
        if funcs:
            doc_count = sum(1 for f in funcs if ast.get_docstring(f))
            score += 0.15 * (doc_count / len(funcs))
            total_checks += 1

        # Type annotation presence
        if funcs:
            annotated = sum(
                1 for f in funcs if f.returns is not None
            )
            score += 0.1 * (annotated / len(funcs))
            total_checks += 1

        # Avoid global state
        global_assigns = sum(
            1 for n in tree.body
            if isinstance(n, ast.Assign)
        )
        score += 0.1 * max(1.0 - global_assigns * 0.1, 0.0)
        total_checks += 1

        return max(min(score, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Refactoring quality
    # ------------------------------------------------------------------

    def _refactoring_quality(self, original: str, refactored: str) -> float:
        """Score how well *refactored* improves on *original* in [0, 1]."""
        orig_tree = _parse_python(original)
        ref_tree = _parse_python(refactored)
        if orig_tree is None or ref_tree is None:
            return 0.0

        score = 0.0

        # Complexity reduction
        orig_cc = 1 + _count_branches(orig_tree)
        ref_cc = 1 + _count_branches(ref_tree)
        if ref_cc < orig_cc:
            score += 0.2
        elif ref_cc == orig_cc:
            score += 0.1

        # Nesting reduction
        orig_nest = _max_nesting(orig_tree)
        ref_nest = _max_nesting(ref_tree)
        if ref_nest < orig_nest:
            score += 0.15
        elif ref_nest == orig_nest:
            score += 0.05

        # LOC reduction (or reasonable growth)
        _, orig_loc, _ = _lines_of_code(original)
        _, ref_loc, _ = _lines_of_code(refactored)
        if ref_loc <= orig_loc * 1.1:
            score += 0.1
        elif ref_loc <= orig_loc * 1.5:
            score += 0.05

        # Better function decomposition
        orig_funcs = len(_function_defs(orig_tree))
        ref_funcs = len(_function_defs(ref_tree))
        if ref_funcs > orig_funcs:
            score += 0.15

        # Improved naming
        orig_ids = _identifiers(orig_tree)
        ref_ids = _identifiers(ref_tree)
        orig_avg_len = statistics.mean(len(n) for n in orig_ids) if orig_ids else 0
        ref_avg_len = statistics.mean(len(n) for n in ref_ids) if ref_ids else 0
        if ref_avg_len > orig_avg_len:
            score += 0.1

        # Preserved docstrings
        orig_docs = sum(1 for f in _function_defs(orig_tree) if ast.get_docstring(f))
        ref_docs = sum(1 for f in _function_defs(ref_tree) if ast.get_docstring(f))
        if ref_docs >= orig_docs:
            score += 0.1

        # Added type hints
        orig_hints = sum(
            1 for f in _function_defs(orig_tree) if f.returns is not None
        )
        ref_hints = sum(
            1 for f in _function_defs(ref_tree) if f.returns is not None
        )
        if ref_hints > orig_hints:
            score += 0.1

        # Structural similarity preserved (shouldn't be entirely different code)
        sim = self._code_similarity(original, refactored)
        if 0.3 <= sim <= 0.85:
            score += 0.1

        return max(min(score, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Identifier quality
    # ------------------------------------------------------------------

    def _identifier_quality(self, code: str) -> float:
        """Score meaningfulness of variable and function names in [0, 1]."""
        tree = _parse_python(code)
        if tree is None:
            return 0.0

        ids = _identifiers(tree)
        if not ids:
            return 0.5

        scores: List[float] = []
        for name in ids:
            scores.append(self._single_identifier_score(name))

        return statistics.mean(scores) if scores else 0.0

    @staticmethod
    def _single_identifier_score(name: str) -> float:
        """Rate a single identifier name."""
        if name.startswith("_"):
            name = name.lstrip("_")
        if not name:
            return 0.3
        score = 0.0
        # Length: 1-char names are poor except conventional ones
        conventional_short = {"i", "j", "k", "n", "x", "y", "z", "e", "f", "v", "s", "c", "m"}
        if len(name) == 1:
            score += 0.4 if name in conventional_short else 0.1
        elif len(name) == 2:
            score += 0.5
        elif 3 <= len(name) <= 25:
            score += 0.7
        else:
            score += 0.4

        # Snake_case compliance
        if re.match(r"^[a-z][a-z0-9_]*$", name):
            score += 0.2
        elif re.match(r"^[A-Z][a-zA-Z0-9]*$", name):
            score += 0.2
        elif re.match(r"^[A-Z_][A-Z0-9_]*$", name):
            score += 0.15

        # Not a keyword / builtin shadow
        if name in dir(__builtins__) or keyword.iskeyword(name):
            score -= 0.1

        return max(min(score, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Dead code detection
    # ------------------------------------------------------------------

    def _dead_code_detection(self, code: str) -> List[str]:
        """Return a list of warnings about likely dead code."""
        tree = _parse_python(code)
        warnings: List[str] = []
        if tree is None:
            return warnings

        # Unreachable code after return
        for func in _function_defs(tree):
            for i, stmt in enumerate(func.body):
                if isinstance(stmt, ast.Return) and i < len(func.body) - 1:
                    warnings.append(
                        f"Unreachable code after return at line {stmt.lineno} in '{func.name}'"
                    )

        # Unused local variables (simple check)
        for func in _function_defs(tree):
            assigned: Set[str] = set()
            used: Set[str] = set()
            for node in ast.walk(func):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.add(node.id)
            params = {a.arg for a in func.args.args}
            unused = assigned - used - params - {"_"}
            for name in sorted(unused):
                warnings.append(f"Unused variable '{name}' in '{func.name}'")

        # Duplicate function definitions
        func_names: List[str] = [f.name for f in _function_defs(tree)]
        seen: Set[str] = set()
        for name in func_names:
            if name in seen:
                warnings.append(f"Duplicate function definition: '{name}'")
            seen.add(name)

        # pass in non-empty body
        for func in _function_defs(tree):
            if len(func.body) > 1:
                for stmt in func.body:
                    if isinstance(stmt, ast.Pass):
                        warnings.append(
                            f"Unnecessary 'pass' in '{func.name}' (body has other statements)"
                        )

        return warnings

    # ------------------------------------------------------------------
    # Import analysis
    # ------------------------------------------------------------------

    def _import_analysis(self, code: str) -> Dict[str, Any]:
        """Analyse imports in Python *code*.

        Returns dict with keys: modules, from_imports, total, unused_estimate,
        cleanliness (float in [0,1]).
        """
        tree = _parse_python(code)
        if tree is None:
            return {"modules": [], "from_imports": [], "total": 0,
                    "unused_estimate": [], "cleanliness": 1.0}

        modules: List[str] = []
        from_imports: List[Tuple[str, List[str]]] = []
        imported_names: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    modules.append(alias.name)
                    imported_names.add(name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names_list: List[str] = []
                for alias in node.names:
                    n = alias.asname or alias.name
                    names_list.append(n)
                    imported_names.add(n)
                from_imports.append((mod, names_list))

        # Check which imported names are used in the rest of the code
        all_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    all_names.add(node.value.id)

        unused = imported_names - all_names
        total = len(imported_names)
        cleanliness = 1.0 - (len(unused) / max(total, 1))
        cleanliness = max(min(cleanliness, 1.0), 0.0)

        # Star imports are bad
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(alias.name == "*" for alias in node.names):
                    cleanliness *= 0.5

        return {
            "modules": modules,
            "from_imports": from_imports,
            "total": total,
            "unused_estimate": sorted(unused),
            "cleanliness": cleanliness,
        }

    # ------------------------------------------------------------------
    # Function decomposition score
    # ------------------------------------------------------------------

    def _function_decomposition_score(self, code: str) -> float:
        """Score how well the code is decomposed into functions.

        Prefers multiple short, focused functions over a single long one.
        Returns a value in [0, 1].
        """
        tree = _parse_python(code)
        if tree is None:
            return 0.0

        funcs = _function_defs(tree)
        _, code_lines, _ = _lines_of_code(code)

        if code_lines == 0:
            return 0.0

        if not funcs:
            # All code at module level – penalise if it's substantial
            return max(1.0 - code_lines / 50.0, 0.0)

        score = 0.0

        # Number of functions relative to code size
        ratio = len(funcs) / max(code_lines / 15.0, 1.0)
        score += 0.3 * min(ratio, 1.0)

        # Average function length (prefer ≤20 lines)
        func_lengths: List[int] = []
        for f in funcs:
            if hasattr(f, "end_lineno") and f.end_lineno and f.lineno:
                func_lengths.append(f.end_lineno - f.lineno + 1)
        if func_lengths:
            avg_len = statistics.mean(func_lengths)
            score += 0.3 * max(1.0 - (avg_len - 10) / 30.0, 0.0)
        else:
            score += 0.15

        # Low parameter count
        param_counts = [len(f.args.args) for f in funcs]
        avg_params = statistics.mean(param_counts) if param_counts else 0
        score += 0.2 * max(1.0 - (avg_params - 2) / 6.0, 0.0)

        # Single-responsibility heuristic: low cyclomatic per function
        per_func_cc: List[int] = []
        for f in funcs:
            per_func_cc.append(1 + _count_branches(f))
        avg_cc = statistics.mean(per_func_cc) if per_func_cc else 0
        score += 0.2 * max(1.0 - (avg_cc - 3) / 10.0, 0.0)

        return max(min(score, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Error handling score
    # ------------------------------------------------------------------

    def _error_handling_score(self, code: str) -> float:
        """Score how well error handling is implemented in [0, 1]."""
        tree = _parse_python(code)
        if tree is None:
            return 0.0

        funcs = _function_defs(tree)
        if not funcs:
            return 0.5

        score = 0.0
        total_checks = 0

        for func in funcs:
            total_checks += 1
            func_score = 0.0

            # Has try/except?
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(func))
            if has_try:
                func_score += 0.3

            # Catches specific exceptions (not bare except)?
            for node in ast.walk(func):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is not None:
                        func_score += 0.2
                    else:
                        func_score += 0.05  # bare except

            # Raises exceptions with messages?
            for node in ast.walk(func):
                if isinstance(node, ast.Raise) and node.exc is not None:
                    func_score += 0.15

            # Guard clauses (early returns on invalid input)
            if func.body:
                first_stmts = func.body[:3]
                for stmt in first_stmts:
                    if isinstance(stmt, ast.If):
                        for sub in ast.walk(stmt):
                            if isinstance(sub, (ast.Raise, ast.Return)):
                                func_score += 0.1
                                break

            # Input validation (isinstance checks, assert)
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
                        func_score += 0.1
                elif isinstance(node, ast.Assert):
                    func_score += 0.05

            score += min(func_score, 1.0)

        return max(min(score / max(total_checks, 1), 1.0), 0.0)

    # ------------------------------------------------------------------
    # Type annotation coverage
    # ------------------------------------------------------------------

    def _type_annotation_coverage(self, code: str) -> float:
        """Fraction of function parameters and returns that have type annotations."""
        tree = _parse_python(code)
        if tree is None:
            return 0.0

        funcs = _function_defs(tree)
        if not funcs:
            return 0.0

        total_slots = 0
        annotated_slots = 0

        for func in funcs:
            args = [a for a in func.args.args if a.arg != "self"]
            for arg in args:
                total_slots += 1
                if arg.annotation is not None:
                    annotated_slots += 1
            # Return annotation
            total_slots += 1
            if func.returns is not None:
                annotated_slots += 1

        if total_slots == 0:
            return 0.0
        return annotated_slots / total_slots

    # ------------------------------------------------------------------
    # Cognitive complexity
    # ------------------------------------------------------------------

    def _cognitive_complexity(self, code: str) -> int:
        """Compute cognitive complexity (Sonar-style) for Python *code*.

        Higher numbers indicate harder-to-understand code.
        """
        tree = _parse_python(code)
        if tree is None:
            return 0

        total = 0
        for func in _function_defs(tree):
            total += self._cognitive_complexity_of_node(func, nesting=0)
        return total

    def _cognitive_complexity_of_node(self, node: ast.AST, nesting: int) -> int:
        """Recursive cognitive-complexity walker."""
        score = 0

        # Increments
        increment_types = (ast.If, ast.For, ast.While, ast.ExceptHandler)
        if isinstance(node, increment_types):
            score += 1 + nesting

        # Boolean operators add per extra operand
        if isinstance(node, ast.BoolOp):
            score += len(node.values) - 1

        # Nesting increments for structural nodes
        nesting_types = (ast.If, ast.For, ast.While, ast.With, ast.Try)
        child_nesting = nesting + 1 if isinstance(node, nesting_types) else nesting

        # Recursion: calling a function with the same name
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Heuristic: we can't easily know the enclosing func name here
                pass

        # Walk children
        for child in ast.iter_child_nodes(node):
            score += self._cognitive_complexity_of_node(child, child_nesting)

        return score

    # ------------------------------------------------------------------
    # Halstead metrics
    # ------------------------------------------------------------------

    def _halstead_metrics(self, code: str) -> Dict[str, float]:
        """Compute Halstead software science metrics.

        Returns dict with: n1, n2, N1, N2, vocabulary, length, volume,
        difficulty, effort, time_estimate, bugs_estimate.
        All values default to 0 on parse failure.
        """
        tree = _parse_python(code)
        if tree is None:
            return self._empty_halstead()

        operators, operands = _halstead_raw(tree)

        n1 = len(operators)   # distinct operators
        n2 = len(operands)    # distinct operands
        cap_n1 = sum(operators.values())  # total operators
        cap_n2 = sum(operands.values())   # total operands

        vocabulary = n1 + n2
        length = cap_n1 + cap_n2
        if vocabulary == 0:
            return self._empty_halstead()

        volume = length * math.log2(max(vocabulary, 1))
        difficulty = (n1 / 2.0) * (cap_n2 / max(n2, 1))
        effort = difficulty * volume
        time_estimate = effort / 18.0
        bugs_estimate = volume / 3000.0

        return {
            "n1": float(n1),
            "n2": float(n2),
            "N1": float(cap_n1),
            "N2": float(cap_n2),
            "vocabulary": float(vocabulary),
            "length": float(length),
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2),
            "time_estimate": round(time_estimate, 2),
            "bugs_estimate": round(bugs_estimate, 4),
        }

    @staticmethod
    def _empty_halstead() -> Dict[str, float]:
        return {
            "n1": 0, "n2": 0, "N1": 0, "N2": 0,
            "vocabulary": 0, "length": 0, "volume": 0,
            "difficulty": 0, "effort": 0, "time_estimate": 0,
            "bugs_estimate": 0,
        }

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self, code: str) -> str:
        """Clean and lightly format generated *code*."""
        # Strip markdown fences
        code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)

        # Remove leading/trailing blank lines
        lines = code.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Normalise trailing whitespace
        lines = [l.rstrip() for l in lines]

        # Ensure trailing newline
        result = "\n".join(lines)
        if result and not result.endswith("\n"):
            result += "\n"

        return result

    # ------------------------------------------------------------------
    # Language helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _comment_prefix(lang: ProgrammingLanguage) -> str:
        if lang == ProgrammingLanguage.PYTHON:
            return "#"
        return "//"

    @staticmethod
    def _fence_language(lang: ProgrammingLanguage) -> str:
        mapping = {
            ProgrammingLanguage.PYTHON: "python",
            ProgrammingLanguage.JAVASCRIPT: "javascript",
            ProgrammingLanguage.JAVA: "java",
            ProgrammingLanguage.CPP: "cpp",
            ProgrammingLanguage.RUST: "rust",
            ProgrammingLanguage.GO: "go",
            ProgrammingLanguage.TYPESCRIPT: "typescript",
        }
        return mapping.get(lang, "")

    # ------------------------------------------------------------------
    # Prompt generators
    # ------------------------------------------------------------------

    def _generate_function_synthesis_prompts(self) -> List[CodePrompt]:
        """Return function-synthesis prompts across difficulty tiers."""
        prompts: List[CodePrompt] = []

        # ---- TRIVIAL ----

        prompts.append(CodePrompt(
            id="fs-trivial-01",
            text="Write a function that adds two numbers.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def add(a: int, b: int) -> int:",
            docstring="Return the sum of a and b.",
            test_cases=[
                CodeTestCase([1, 2], 3, "basic addition"),
                CodeTestCase([0, 0], 0, "zeros"),
                CodeTestCase([-1, 1], 0, "negative + positive"),
                CodeTestCase([1000000, 2000000], 3000000, "large numbers"),
            ],
            expected_complexity=CodeComplexity.TRIVIAL,
            input_types=["int", "int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-trivial-02",
            text="Write a function that returns the absolute value of a number.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def absolute(n: int) -> int:",
            docstring="Return the absolute value of n.",
            test_cases=[
                CodeTestCase([5], 5, "positive"),
                CodeTestCase([-5], 5, "negative"),
                CodeTestCase([0], 0, "zero"),
            ],
            expected_complexity=CodeComplexity.TRIVIAL,
            input_types=["int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-trivial-03",
            text="Write a function that returns the maximum of two numbers.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def maximum(a: int, b: int) -> int:",
            docstring="Return the larger of a and b.",
            test_cases=[
                CodeTestCase([3, 5], 5, "b larger"),
                CodeTestCase([7, 2], 7, "a larger"),
                CodeTestCase([4, 4], 4, "equal"),
            ],
            expected_complexity=CodeComplexity.TRIVIAL,
            input_types=["int", "int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-trivial-04",
            text="Write a function that checks if a number is even.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def is_even(n: int) -> bool:",
            docstring="Return True if n is even, False otherwise.",
            test_cases=[
                CodeTestCase([2], True, "even"),
                CodeTestCase([3], False, "odd"),
                CodeTestCase([0], True, "zero"),
                CodeTestCase([-4], True, "negative even"),
            ],
            expected_complexity=CodeComplexity.TRIVIAL,
            input_types=["int"],
            output_type="bool",
        ))

        # ---- EASY ----

        prompts.append(CodePrompt(
            id="fs-easy-01",
            text="Write a function that reverses a string.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def reverse_string(s: str) -> str:",
            docstring="Return the reversed version of s.",
            test_cases=[
                CodeTestCase(["hello"], "olleh", "basic"),
                CodeTestCase([""], "", "empty"),
                CodeTestCase(["a"], "a", "single char"),
                CodeTestCase(["racecar"], "racecar", "palindrome"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["str"],
            output_type="str",
        ))

        prompts.append(CodePrompt(
            id="fs-easy-02",
            text="Write a function that computes the factorial of a non-negative integer.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def factorial(n: int) -> int:",
            docstring="Return n! for n >= 0.",
            test_cases=[
                CodeTestCase([0], 1, "base case"),
                CodeTestCase([1], 1, "one"),
                CodeTestCase([5], 120, "five"),
                CodeTestCase([10], 3628800, "ten"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-easy-03",
            text="Write a function that checks if a string is a palindrome.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def is_palindrome(s: str) -> bool:",
            docstring="Return True if s reads the same forwards and backwards.",
            test_cases=[
                CodeTestCase(["racecar"], True, "palindrome"),
                CodeTestCase(["hello"], False, "not palindrome"),
                CodeTestCase([""], True, "empty", True),
                CodeTestCase(["a"], True, "single char"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["str"],
            output_type="bool",
        ))

        prompts.append(CodePrompt(
            id="fs-easy-04",
            text="Write a function that returns the nth Fibonacci number.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def fibonacci(n: int) -> int:",
            docstring="Return the nth Fibonacci number (0-indexed).",
            test_cases=[
                CodeTestCase([0], 0, "zeroth"),
                CodeTestCase([1], 1, "first"),
                CodeTestCase([6], 8, "sixth"),
                CodeTestCase([10], 55, "tenth"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-easy-05",
            text="Write a function that counts vowels in a string.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def count_vowels(s: str) -> int:",
            docstring="Return the number of vowels (a, e, i, o, u) in s.",
            test_cases=[
                CodeTestCase(["hello"], 2, "basic"),
                CodeTestCase(["aeiou"], 5, "all vowels"),
                CodeTestCase(["xyz"], 0, "no vowels"),
                CodeTestCase([""], 0, "empty", True),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["str"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-easy-06",
            text="Write a function that flattens a nested list one level deep.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def flatten(lst: list) -> list:",
            docstring="Flatten a list of lists by one level.",
            test_cases=[
                CodeTestCase([[[1, 2], [3, 4]]], [1, 2, 3, 4], "basic"),
                CodeTestCase([[]], [], "empty"),
                CodeTestCase([[[1], [2], [3]]], [1, 2, 3], "single elements"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["list"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-easy-07",
            text="Write a function that removes duplicates from a list while preserving order.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def unique(lst: list) -> list:",
            docstring="Return a list with duplicates removed, preserving first occurrence order.",
            test_cases=[
                CodeTestCase([[1, 2, 2, 3, 1]], [1, 2, 3], "basic"),
                CodeTestCase([[]], [], "empty"),
                CodeTestCase([[1, 1, 1]], [1], "all same"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["list"],
            output_type="list",
        ))

        # ---- MEDIUM ----

        prompts.append(CodePrompt(
            id="fs-medium-01",
            text="Write a function that performs binary search on a sorted list.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def binary_search(arr: list, target: int) -> int:",
            docstring="Return the index of target in sorted arr, or -1 if not found.",
            test_cases=[
                CodeTestCase([[1, 3, 5, 7, 9], 5], 2, "found in middle"),
                CodeTestCase([[1, 3, 5, 7, 9], 1], 0, "found at start"),
                CodeTestCase([[1, 3, 5, 7, 9], 9], 4, "found at end"),
                CodeTestCase([[1, 3, 5, 7, 9], 4], -1, "not found"),
                CodeTestCase([[], 1], -1, "empty list", True),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["list", "int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-02",
            text="Write a function that merges two sorted lists into one sorted list.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def merge_sorted(a: list, b: list) -> list:",
            docstring="Merge two sorted lists into a single sorted list.",
            test_cases=[
                CodeTestCase([[1, 3, 5], [2, 4, 6]], [1, 2, 3, 4, 5, 6], "interleaved"),
                CodeTestCase([[], [1, 2]], [1, 2], "first empty"),
                CodeTestCase([[1, 2], []], [1, 2], "second empty"),
                CodeTestCase([[1, 1], [1, 1]], [1, 1, 1, 1], "duplicates"),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["list", "list"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-03",
            text="Write a function that groups anagrams from a list of strings.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def group_anagrams(words: list) -> list:",
            docstring="Group words that are anagrams of each other. Return a list of groups.",
            test_cases=[
                CodeTestCase(
                    [["eat", "tea", "tan", "ate", "nat", "bat"]],
                    [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
                    "basic grouping",
                ),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["list"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-04",
            text="Write a function that validates balanced parentheses including (), [], {}.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def is_balanced(s: str) -> bool:",
            docstring="Return True if all brackets in s are properly balanced.",
            test_cases=[
                CodeTestCase(["()[]{}"], True, "simple balanced"),
                CodeTestCase(["([{}])"], True, "nested"),
                CodeTestCase(["([)]"], False, "mismatched"),
                CodeTestCase([""], True, "empty", True),
                CodeTestCase(["("], False, "unclosed"),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["str"],
            output_type="bool",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-05",
            text="Write a function that computes the longest common subsequence length.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def lcs_length(a: str, b: str) -> int:",
            docstring="Return the length of the longest common subsequence of a and b.",
            test_cases=[
                CodeTestCase(["abcde", "ace"], 3, "basic"),
                CodeTestCase(["abc", "abc"], 3, "identical"),
                CodeTestCase(["abc", "def"], 0, "no common"),
                CodeTestCase(["", "abc"], 0, "empty", True),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["str", "str"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-06",
            text="Write a function that rotates a 2D matrix 90 degrees clockwise in-place.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def rotate_matrix(matrix: list) -> list:",
            docstring="Rotate an NxN matrix 90 degrees clockwise and return it.",
            test_cases=[
                CodeTestCase(
                    [[[1, 2], [3, 4]]],
                    [[3, 1], [4, 2]],
                    "2x2",
                ),
                CodeTestCase(
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[7, 4, 1], [8, 5, 2], [9, 6, 3]],
                    "3x3",
                ),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["list"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-07",
            text="Write a function that finds all prime factors of a number.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def prime_factors(n: int) -> list:",
            docstring="Return a sorted list of prime factors of n (with repetition).",
            test_cases=[
                CodeTestCase([12], [2, 2, 3], "twelve"),
                CodeTestCase([1], [], "one"),
                CodeTestCase([7], [7], "prime"),
                CodeTestCase([100], [2, 2, 5, 5], "hundred"),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["int"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-08",
            text="Write a function that implements a simple LRU cache.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def make_lru_cache(capacity: int) -> object:",
            docstring="Return an LRU cache object with get(key) and put(key, value) methods.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["int"],
            output_type="object",
        ))

        prompts.append(CodePrompt(
            id="fs-medium-09",
            text="Write a function that converts a Roman numeral string to an integer.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def roman_to_int(s: str) -> int:",
            docstring="Convert a Roman numeral string to its integer value.",
            test_cases=[
                CodeTestCase(["III"], 3, "three"),
                CodeTestCase(["IV"], 4, "four"),
                CodeTestCase(["IX"], 9, "nine"),
                CodeTestCase(["MCMXCIV"], 1994, "complex"),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["str"],
            output_type="int",
        ))

        # ---- HARD ----

        prompts.append(CodePrompt(
            id="fs-hard-01",
            text="Write a function that solves the N-Queens problem.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def solve_n_queens(n: int) -> list:",
            docstring="Return all distinct solutions to the N-Queens puzzle as lists of queen column positions per row.",
            test_cases=[
                CodeTestCase([1], [[0]], "1-queen"),
                CodeTestCase([4], [[1, 3, 0, 2], [2, 0, 3, 1]], "4-queens"),
            ],
            expected_complexity=CodeComplexity.HARD,
            input_types=["int"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-hard-02",
            text="Write a function that finds the longest increasing subsequence.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def longest_increasing_subsequence(arr: list) -> list:",
            docstring="Return one longest strictly increasing subsequence of arr.",
            test_cases=[
                CodeTestCase([[10, 9, 2, 5, 3, 7, 101, 18]], [2, 3, 7, 18], "basic"),
                CodeTestCase([[0, 1, 0, 3, 2, 3]], [0, 1, 2, 3], "with dups"),
            ],
            expected_complexity=CodeComplexity.HARD,
            input_types=["list"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-hard-03",
            text="Write a function to serialize and deserialize a binary tree.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def serialize(root) -> str:\ndef deserialize(data: str):",
            docstring="Convert a binary tree to a string and back. Nodes have val, left, right.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=["TreeNode"],
            output_type="str / TreeNode",
        ))

        prompts.append(CodePrompt(
            id="fs-hard-04",
            text="Write a function that implements Dijkstra's shortest path algorithm.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def dijkstra(graph: dict, start: str, end: str) -> tuple:",
            docstring="Return (distance, path) for the shortest path in a weighted graph.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=["dict", "str", "str"],
            output_type="tuple",
        ))

        prompts.append(CodePrompt(
            id="fs-hard-05",
            text="Write a function that evaluates a mathematical expression string.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def evaluate_expression(expr: str) -> float:",
            docstring="Parse and evaluate an arithmetic expression with +, -, *, /, and parentheses.",
            test_cases=[
                CodeTestCase(["2+3"], 5.0, "simple addition"),
                CodeTestCase(["(2+3)*4"], 20.0, "with parens"),
                CodeTestCase(["10/3"], 3.3333333333333335, "division"),
                CodeTestCase(["2+3*4"], 14.0, "precedence"),
            ],
            expected_complexity=CodeComplexity.HARD,
            input_types=["str"],
            output_type="float",
        ))

        prompts.append(CodePrompt(
            id="fs-hard-06",
            text="Write a function implementing the Knuth-Morris-Pratt string search.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def kmp_search(text: str, pattern: str) -> list:",
            docstring="Return all starting indices where pattern occurs in text.",
            test_cases=[
                CodeTestCase(["abcabcabc", "abc"], [0, 3, 6], "repeating"),
                CodeTestCase(["hello", "xyz"], [], "no match"),
                CodeTestCase(["aaaa", "aa"], [0, 1, 2], "overlapping"),
            ],
            expected_complexity=CodeComplexity.HARD,
            input_types=["str", "str"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="fs-hard-07",
            text="Write a function that topologically sorts a directed acyclic graph.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def topological_sort(graph: dict) -> list:",
            docstring="Return a topological ordering of nodes. graph maps node -> list of neighbours.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=["dict"],
            output_type="list",
        ))

        # ---- EXPERT ----

        prompts.append(CodePrompt(
            id="fs-expert-01",
            text="Write a function that implements a suffix array with LCP array construction.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def build_suffix_array(s: str) -> tuple:",
            docstring="Return (suffix_array, lcp_array) for the string s.",
            test_cases=[],
            expected_complexity=CodeComplexity.EXPERT,
            input_types=["str"],
            output_type="tuple",
        ))

        prompts.append(CodePrompt(
            id="fs-expert-02",
            text="Write an AVL tree implementation with insert, delete, and search.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="class AVLTree:\n    def insert(self, key):\n    def delete(self, key):\n    def search(self, key) -> bool:",
            docstring="Self-balancing binary search tree with O(log n) operations.",
            test_cases=[],
            expected_complexity=CodeComplexity.EXPERT,
            input_types=["int"],
            output_type="AVLTree",
        ))

        prompts.append(CodePrompt(
            id="fs-expert-03",
            text="Write a regex engine that supports '.', '*', '+', '?', and character classes.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def regex_match(pattern: str, text: str) -> bool:",
            docstring="Return True if the entire text matches the pattern.",
            test_cases=[
                CodeTestCase(["a.b", "acb"], True, "dot"),
                CodeTestCase(["a*", "aaa"], True, "star"),
                CodeTestCase(["a+", ""], False, "plus empty"),
            ],
            expected_complexity=CodeComplexity.EXPERT,
            input_types=["str", "str"],
            output_type="bool",
        ))

        return prompts

    def _generate_test_generation_prompts(self) -> List[CodePrompt]:
        """Return prompts asking the model to generate test cases."""
        prompts: List[CodePrompt] = []

        prompts.append(CodePrompt(
            id="tg-easy-01",
            text="Write comprehensive unit tests for a Stack class with push, pop, peek, and is_empty methods.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="class Stack:\n    def push(self, item): ...\n    def pop(self): ...\n    def peek(self): ...\n    def is_empty(self) -> bool: ...",
            docstring="Generate tests covering normal operations, edge cases, and error conditions.",
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="test suite",
        ))

        prompts.append(CodePrompt(
            id="tg-easy-02",
            text="Write unit tests for a function that validates email addresses.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def validate_email(email: str) -> bool:",
            docstring="Generate tests for valid emails, invalid formats, edge cases, and unicode.",
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=["str"],
            output_type="test suite",
        ))

        prompts.append(CodePrompt(
            id="tg-medium-01",
            text="Write property-based tests for a sorting function using hypothesis.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def sort(arr: list) -> list:",
            docstring="Generate property-based tests: idempotency, element preservation, ordering.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["list"],
            output_type="test suite",
        ))

        prompts.append(CodePrompt(
            id="tg-medium-02",
            text="Write integration tests for a simple key-value store with get, set, delete, and list_keys.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="class KVStore:\n    def get(self, key): ...\n    def set(self, key, value): ...\n    def delete(self, key): ...\n    def list_keys(self) -> list: ...",
            docstring="Test concurrency safety, persistence, and error handling.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="test suite",
        ))

        prompts.append(CodePrompt(
            id="tg-medium-03",
            text="Write tests for a rate limiter that allows N requests per time window.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="class RateLimiter:\n    def __init__(self, max_requests: int, window_seconds: float): ...\n    def allow(self) -> bool: ...",
            docstring="Test burst behaviour, window expiry, and concurrent access.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="test suite",
        ))

        prompts.append(CodePrompt(
            id="tg-hard-01",
            text="Write tests for a concurrent task scheduler with priorities and dependencies.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="class TaskScheduler:\n    def add_task(self, task, priority, deps): ...\n    def run(self): ...",
            docstring="Test dependency resolution, priority ordering, deadlock detection, cancellation.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=[],
            output_type="test suite",
        ))

        prompts.append(CodePrompt(
            id="tg-hard-02",
            text="Write fuzz tests for a JSON parser.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def parse_json(text: str) -> Any:",
            docstring="Generate adversarial inputs: deeply nested, large numbers, unicode, escape sequences.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=["str"],
            output_type="test suite",
        ))

        return prompts

    def _generate_refactoring_prompts(self) -> List[CodePrompt]:
        """Return prompts asking the model to refactor code."""
        prompts: List[CodePrompt] = []

        prompts.append(CodePrompt(
            id="rf-easy-01",
            text=(
                "Refactor the following code to remove duplication:\n\n"
                "def area_circle(r):\n"
                "    return 3.14159 * r * r\n\n"
                "def area_sphere(r):\n"
                "    return 4 * 3.14159 * r * r\n\n"
                "def volume_sphere(r):\n"
                "    return (4/3) * 3.14159 * r * r * r\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Extract the repeated constant and simplify.",
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="refactored code",
        ))

        prompts.append(CodePrompt(
            id="rf-easy-02",
            text=(
                "Refactor this function to use early returns instead of nested ifs:\n\n"
                "def process(data):\n"
                "    result = None\n"
                "    if data is not None:\n"
                "        if len(data) > 0:\n"
                "            if isinstance(data, list):\n"
                "                result = sum(data)\n"
                "            else:\n"
                "                result = -1\n"
                "        else:\n"
                "            result = 0\n"
                "    else:\n"
                "        result = -1\n"
                "    return result\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Flatten the nesting using guard clauses.",
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="refactored code",
        ))

        prompts.append(CodePrompt(
            id="rf-medium-01",
            text=(
                "Refactor this procedural code to use a class with proper encapsulation:\n\n"
                "students = []\n\n"
                "def add_student(name, grade):\n"
                "    students.append({'name': name, 'grade': grade})\n\n"
                "def get_average():\n"
                "    total = 0\n"
                "    for s in students:\n"
                "        total += s['grade']\n"
                "    return total / len(students)\n\n"
                "def get_top_student():\n"
                "    best = None\n"
                "    for s in students:\n"
                "        if best is None or s['grade'] > best['grade']:\n"
                "            best = s\n"
                "    return best['name']\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Convert to a GradeBook class with proper methods.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="refactored code",
        ))

        prompts.append(CodePrompt(
            id="rf-medium-02",
            text=(
                "Refactor this code to use the Strategy pattern:\n\n"
                "def calculate_price(base, discount_type):\n"
                "    if discount_type == 'percentage':\n"
                "        return base * 0.9\n"
                "    elif discount_type == 'fixed':\n"
                "        return base - 10\n"
                "    elif discount_type == 'bogo':\n"
                "        return base / 2\n"
                "    elif discount_type == 'loyalty':\n"
                "        return base * 0.85\n"
                "    elif discount_type == 'student':\n"
                "        return base * 0.8\n"
                "    else:\n"
                "        return base\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Use the Strategy design pattern with pluggable discount strategies.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="refactored code",
        ))

        prompts.append(CodePrompt(
            id="rf-medium-03",
            text=(
                "Refactor this function to improve readability and efficiency:\n\n"
                "def f(d):\n"
                "    r = {}\n"
                "    for k in d:\n"
                "        v = d[k]\n"
                "        if v not in r:\n"
                "            r[v] = []\n"
                "        r[v].append(k)\n"
                "    o = []\n"
                "    for k in r:\n"
                "        if len(r[k]) > 1:\n"
                "            o.append((k, r[k]))\n"
                "    o.sort(key=lambda x: -len(x[1]))\n"
                "    return o\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Improve naming, use collections.defaultdict, add docstring.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="refactored code",
        ))

        prompts.append(CodePrompt(
            id="rf-hard-01",
            text=(
                "Refactor this monolithic function into smaller composable functions:\n\n"
                "def process_csv(filename):\n"
                "    with open(filename) as f:\n"
                "        lines = f.readlines()\n"
                "    headers = lines[0].strip().split(',')\n"
                "    data = []\n"
                "    for line in lines[1:]:\n"
                "        values = line.strip().split(',')\n"
                "        row = {}\n"
                "        for i, h in enumerate(headers):\n"
                "            try:\n"
                "                row[h] = float(values[i])\n"
                "            except ValueError:\n"
                "                row[h] = values[i]\n"
                "        data.append(row)\n"
                "    # filter\n"
                "    filtered = []\n"
                "    for row in data:\n"
                "        if 'age' in row and row['age'] >= 18:\n"
                "            filtered.append(row)\n"
                "    # aggregate\n"
                "    total = 0\n"
                "    count = 0\n"
                "    for row in filtered:\n"
                "        if 'salary' in row:\n"
                "            total += row['salary']\n"
                "            count += 1\n"
                "    avg = total / count if count else 0\n"
                "    return {'average_salary': avg, 'count': count}\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Break into parse_csv, filter_rows, aggregate functions with clear interfaces.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=[],
            output_type="refactored code",
        ))

        prompts.append(CodePrompt(
            id="rf-hard-02",
            text=(
                "Refactor this deeply coupled code to follow SOLID principles:\n\n"
                "class OrderProcessor:\n"
                "    def process(self, order):\n"
                "        # validate\n"
                "        if not order.get('items'):\n"
                "            raise ValueError('No items')\n"
                "        for item in order['items']:\n"
                "            if item['qty'] <= 0:\n"
                "                raise ValueError('Bad qty')\n"
                "        # calculate total\n"
                "        total = sum(i['price'] * i['qty'] for i in order['items'])\n"
                "        if order.get('coupon'):\n"
                "            total *= 0.9\n"
                "        # charge payment\n"
                "        import stripe\n"
                "        stripe.Charge.create(amount=int(total*100), currency='usd',\n"
                "                             source=order['payment_token'])\n"
                "        # send email\n"
                "        import smtplib\n"
                "        server = smtplib.SMTP('localhost')\n"
                "        server.sendmail('shop@ex.com', order['email'], f'Total: {total}')\n"
                "        server.quit()\n"
                "        return {'status': 'ok', 'total': total}\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring="Separate validation, pricing, payment, and notification concerns.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=[],
            output_type="refactored code",
        ))

        return prompts

    def _generate_bug_fix_prompts(self) -> List[CodePrompt]:
        """Return prompts with buggy code to fix."""
        prompts: List[CodePrompt] = []

        prompts.append(CodePrompt(
            id="bf-easy-01",
            text=(
                "Fix the bug in this function:\n\n"
                "def find_max(lst):\n"
                "    max_val = 0\n"
                "    for item in lst:\n"
                "        if item > max_val:\n"
                "            max_val = item\n"
                "    return max_val\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Bug: initialising max_val to 0 fails for all-negative lists.",
            test_cases=[
                CodeTestCase([[-3, -1, -5]], -1, "all negative"),
                CodeTestCase([[1, 2, 3]], 3, "positive"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["list"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="bf-easy-02",
            text=(
                "Fix the bug in this function:\n\n"
                "def average(numbers):\n"
                "    return sum(numbers) / len(numbers)\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Bug: crashes with ZeroDivisionError on empty list.",
            test_cases=[
                CodeTestCase([[1, 2, 3]], 2.0, "normal"),
                CodeTestCase([[]], 0.0, "empty", True),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["list"],
            output_type="float",
        ))

        prompts.append(CodePrompt(
            id="bf-medium-01",
            text=(
                "Fix the off-by-one error in this binary search:\n\n"
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr)\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            lo = mid\n"
                "        else:\n"
                "            hi = mid\n"
                "    return -1\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Bug: lo = mid can cause infinite loop when lo + 1 == hi.",
            test_cases=[
                CodeTestCase([[1, 3, 5, 7], 3], 1, "found"),
                CodeTestCase([[1, 3, 5, 7], 4], -1, "not found"),
                CodeTestCase([[1, 2], 2], 1, "boundary"),
            ],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["list", "int"],
            output_type="int",
        ))

        prompts.append(CodePrompt(
            id="bf-medium-02",
            text=(
                "Fix the race condition in this thread-safe counter:\n\n"
                "import threading\n\n"
                "class Counter:\n"
                "    def __init__(self):\n"
                "        self.value = 0\n\n"
                "    def increment(self):\n"
                "        temp = self.value\n"
                "        self.value = temp + 1\n\n"
                "    def get(self):\n"
                "        return self.value\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Bug: increment is not atomic; add a threading.Lock.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="Counter",
        ))

        prompts.append(CodePrompt(
            id="bf-hard-01",
            text=(
                "Fix the memory leak in this cache implementation:\n\n"
                "class Cache:\n"
                "    _instances = {}\n\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "        self.data = {}\n"
                "        Cache._instances[name] = self\n\n"
                "    def set(self, key, value):\n"
                "        self.data[key] = value\n\n"
                "    def get(self, key):\n"
                "        return self.data.get(key)\n\n"
                "    @classmethod\n"
                "    def get_instance(cls, name):\n"
                "        return cls._instances.get(name)\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Bug: _instances dict holds strong references, preventing garbage collection.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=[],
            output_type="Cache",
        ))

        return prompts

    def _generate_documentation_prompts(self) -> List[CodePrompt]:
        """Return prompts asking for documentation of existing code."""
        prompts: List[CodePrompt] = []

        prompts.append(CodePrompt(
            id="doc-easy-01",
            text=(
                "Add comprehensive docstrings to this code:\n\n"
                "def gcd(a, b):\n"
                "    while b:\n"
                "        a, b = b, a % b\n"
                "    return a\n\n"
                "def lcm(a, b):\n"
                "    return a * b // gcd(a, b)\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Add Google-style docstrings with Args, Returns, and Examples sections.",
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="documented code",
        ))

        prompts.append(CodePrompt(
            id="doc-easy-02",
            text=(
                "Write a module-level docstring and function docstrings for:\n\n"
                "import re\n\n"
                "def tokenize(text):\n"
                "    return re.findall(r'\\w+', text.lower())\n\n"
                "def word_freq(tokens):\n"
                "    freq = {}\n"
                "    for t in tokens:\n"
                "        freq[t] = freq.get(t, 0) + 1\n"
                "    return freq\n\n"
                "def top_n(freq, n=10):\n"
                "    return sorted(freq.items(), key=lambda x: -x[1])[:n]\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Include type information, edge cases, and usage examples.",
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="documented code",
        ))

        prompts.append(CodePrompt(
            id="doc-medium-01",
            text=(
                "Write comprehensive API documentation for this class:\n\n"
                "class Matrix:\n"
                "    def __init__(self, data):\n"
                "        self.data = data\n"
                "        self.rows = len(data)\n"
                "        self.cols = len(data[0]) if data else 0\n\n"
                "    def __mul__(self, other):\n"
                "        result = [[0]*other.cols for _ in range(self.rows)]\n"
                "        for i in range(self.rows):\n"
                "            for j in range(other.cols):\n"
                "                for k in range(self.cols):\n"
                "                    result[i][j] += self.data[i][k] * other.data[k][j]\n"
                "        return Matrix(result)\n\n"
                "    def transpose(self):\n"
                "        return Matrix([[self.data[j][i] for j in range(self.rows)]\n"
                "                       for i in range(self.cols)])\n\n"
                "    def determinant(self):\n"
                "        if self.rows != self.cols:\n"
                "            raise ValueError('Must be square')\n"
                "        if self.rows == 1:\n"
                "            return self.data[0][0]\n"
                "        if self.rows == 2:\n"
                "            return self.data[0][0]*self.data[1][1] - self.data[0][1]*self.data[1][0]\n"
                "        det = 0\n"
                "        for j in range(self.cols):\n"
                "            minor = Matrix([row[:j]+row[j+1:] for row in self.data[1:]])\n"
                "            det += ((-1)**j) * self.data[0][j] * minor.determinant()\n"
                "        return det\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Include class docstring, method docs, complexity notes, and examples.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="documented code",
        ))

        prompts.append(CodePrompt(
            id="doc-medium-02",
            text=(
                "Write README-style documentation for this module:\n\n"
                "import hashlib\nimport time\n\n"
                "class BloomFilter:\n"
                "    def __init__(self, size, num_hashes):\n"
                "        self.size = size\n"
                "        self.num_hashes = num_hashes\n"
                "        self.bits = [False] * size\n\n"
                "    def _hashes(self, item):\n"
                "        results = []\n"
                "        for i in range(self.num_hashes):\n"
                "            h = hashlib.md5(f'{item}{i}'.encode()).hexdigest()\n"
                "            results.append(int(h, 16) % self.size)\n"
                "        return results\n\n"
                "    def add(self, item):\n"
                "        for pos in self._hashes(item):\n"
                "            self.bits[pos] = True\n\n"
                "    def __contains__(self, item):\n"
                "        return all(self.bits[pos] for pos in self._hashes(item))\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            docstring="Include usage examples, performance characteristics, and false positive rates.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="documented code",
        ))

        return prompts

    def _generate_code_completion_prompts(self) -> List[CodePrompt]:
        """Return prompts with partial code to complete."""
        prompts: List[CodePrompt] = []

        prompts.append(CodePrompt(
            id="cc-easy-01",
            text=(
                "Complete the following function:\n\n"
                "def caesar_cipher(text: str, shift: int) -> str:\n"
                "    \"\"\"Encrypt text using Caesar cipher with the given shift.\"\"\"\n"
                "    result = []\n"
                "    for char in text:\n"
                "        if char.isalpha():\n"
                "            # TODO: complete this\n"
                "            pass\n"
                "        else:\n"
                "            result.append(char)\n"
                "    return ''.join(result)\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="def caesar_cipher(text: str, shift: int) -> str:",
            docstring="Complete the Caesar cipher implementation.",
            test_cases=[
                CodeTestCase(["abc", 1], "bcd", "shift by 1"),
                CodeTestCase(["xyz", 3], "abc", "wrap around"),
                CodeTestCase(["Hello", 0], "Hello", "no shift"),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["str", "int"],
            output_type="str",
        ))

        prompts.append(CodePrompt(
            id="cc-easy-02",
            text=(
                "Complete the following function:\n\n"
                "def matrix_multiply(a: list, b: list) -> list:\n"
                "    \"\"\"Multiply two matrices a and b.\"\"\"\n"
                "    rows_a, cols_a = len(a), len(a[0])\n"
                "    rows_b, cols_b = len(b), len(b[0])\n"
                "    if cols_a != rows_b:\n"
                "        raise ValueError('Incompatible dimensions')\n"
                "    # TODO: complete the multiplication\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="def matrix_multiply(a: list, b: list) -> list:",
            docstring="Complete the matrix multiplication.",
            test_cases=[
                CodeTestCase(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[19, 22], [43, 50]],
                    "2x2 multiplication",
                ),
            ],
            expected_complexity=CodeComplexity.EASY,
            input_types=["list", "list"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="cc-medium-01",
            text=(
                "Complete this trie implementation:\n\n"
                "class TrieNode:\n"
                "    def __init__(self):\n"
                "        self.children = {}\n"
                "        self.is_end = False\n\n"
                "class Trie:\n"
                "    def __init__(self):\n"
                "        self.root = TrieNode()\n\n"
                "    def insert(self, word: str) -> None:\n"
                "        # TODO\n"
                "        pass\n\n"
                "    def search(self, word: str) -> bool:\n"
                "        # TODO\n"
                "        pass\n\n"
                "    def starts_with(self, prefix: str) -> bool:\n"
                "        # TODO\n"
                "        pass\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="class Trie",
            docstring="Complete insert, search, and starts_with methods.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["str"],
            output_type="Trie",
        ))

        prompts.append(CodePrompt(
            id="cc-medium-02",
            text=(
                "Complete this graph BFS implementation:\n\n"
                "from collections import deque\n\n"
                "def bfs_shortest_path(graph: dict, start, end) -> list:\n"
                "    \"\"\"Find shortest path in unweighted graph using BFS.\"\"\"\n"
                "    if start == end:\n"
                "        return [start]\n"
                "    visited = set()\n"
                "    queue = deque()\n"
                "    # TODO: complete BFS with path tracking\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="def bfs_shortest_path(graph: dict, start, end) -> list:",
            docstring="Complete the BFS with path reconstruction.",
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=["dict", "str", "str"],
            output_type="list",
        ))

        prompts.append(CodePrompt(
            id="cc-hard-01",
            text=(
                "Complete this parser combinator framework:\n\n"
                "class Parser:\n"
                "    def __init__(self, fn):\n"
                "        self.fn = fn\n\n"
                "    def parse(self, text, pos=0):\n"
                "        return self.fn(text, pos)\n\n"
                "    def __or__(self, other):\n"
                "        # TODO: alternative combinator\n"
                "        pass\n\n"
                "    def __rshift__(self, other):\n"
                "        # TODO: sequence combinator\n"
                "        pass\n\n"
                "def literal(s):\n"
                "    # TODO: match exact string\n"
                "    pass\n\n"
                "def regex(pattern):\n"
                "    # TODO: match regex\n"
                "    pass\n"
            ),
            language=ProgrammingLanguage.PYTHON,
            function_signature="class Parser",
            docstring="Complete the parser combinator with alt, seq, literal, and regex.",
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=["str"],
            output_type="Parser",
        ))

        return prompts

    def _generate_api_design_prompts(self) -> List[CodePrompt]:
        """Return prompts asking the model to design APIs."""
        prompts: List[CodePrompt] = []

        prompts.append(CodePrompt(
            id="api-easy-01",
            text="Design a Python API for a simple task/todo list manager.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Design classes/functions for creating, listing, updating, and deleting tasks. "
                "Include priorities, due dates, and tags. Provide type hints and docstrings."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-easy-02",
            text="Design a Python API for a temperature converter.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support Celsius, Fahrenheit, and Kelvin. Include a Temperature class "
                "with conversion methods and comparison operators."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.EASY,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-medium-01",
            text="Design a Python API for an event bus / pub-sub system.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support subscribing to topics, publishing events, wildcard subscriptions, "
                "event history, and unsubscribing. Thread-safe."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-medium-02",
            text="Design a Python API for a simple in-memory database with SQL-like queries.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support create_table, insert, select with where/order_by/limit, "
                "update, delete. Type-safe column definitions."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-medium-03",
            text="Design a Python API for a configuration management system.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support hierarchical configs, environment variable overrides, "
                "type validation, defaults, and config file formats (JSON, YAML, TOML)."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.MEDIUM,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-hard-01",
            text="Design a Python API for a workflow/pipeline orchestration engine.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support defining DAG-based workflows, task dependencies, retries, "
                "timeouts, parallel execution, and status monitoring."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-hard-02",
            text="Design a Python API for a plugin system with dependency injection.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support plugin registration, lifecycle hooks, dependency resolution, "
                "versioning, and conflict detection."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.HARD,
            input_types=[],
            output_type="module",
        ))

        prompts.append(CodePrompt(
            id="api-expert-01",
            text="Design a Python API for a distributed task queue with exactly-once semantics.",
            language=ProgrammingLanguage.PYTHON,
            function_signature="",
            docstring=(
                "Support task submission, worker pools, result backends, dead-letter queues, "
                "rate limiting, priorities, and task chaining."
            ),
            test_cases=[],
            expected_complexity=CodeComplexity.EXPERT,
            input_types=[],
            output_type="module",
        ))

        return prompts


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_code_generation_task(
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    task_type: CodeTaskType = CodeTaskType.FUNCTION_SYNTHESIS,
    complexity: CodeComplexity = CodeComplexity.MEDIUM,
    **kwargs: Any,
) -> CodeGenerationTask:
    """Convenience factory for :class:`CodeGenerationTask`."""
    config = CodeGenerationConfig(
        language=language,
        task_type=task_type,
        complexity=complexity,
        **kwargs,
    )
    return CodeGenerationTask(config=config)


# ---------------------------------------------------------------------------
# Diversity analysis utilities
# ---------------------------------------------------------------------------

@dataclass
class _DiversityScores:
    """Container for a set of diversity measurements."""
    functional: float = 0.0
    style: float = 0.0
    algorithmic: float = 0.0
    complexity: float = 0.0
    overall: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """Return scores as a plain dictionary."""
        return {
            "functional": self.functional,
            "style": self.style,
            "algorithmic": self.algorithmic,
            "complexity": self.complexity,
            "overall": self.overall,
        }


# ---------------------------------------------------------------------------
# CodeDiversityAnalyzer
# ---------------------------------------------------------------------------

class CodeDiversityAnalyzer:
    """Analyse diversity across a collection of generated code samples.

    The analyser examines *functional*, *stylistic*, *algorithmic*, and
    *complexity* diversity.  All public ``measure_*`` methods return a
    float in [0, 1] where higher values indicate greater diversity.
    """

    # Timeout (seconds) when executing code for functional equivalence.
    _EXEC_TIMEOUT: int = 5

    # ---- naming convention regexes ----------------------------------------
    _RE_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
    _RE_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
    _RE_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
    _RE_UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")

    # ---- control-flow node types we care about ----------------------------
    _CF_NODE_TYPES: Tuple[type, ...] = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
        *,
        exec_timeout: int = 5,
        max_test_inputs: int = 50,
    ) -> None:
        self.language = language
        self._EXEC_TIMEOUT = exec_timeout
        self._max_test_inputs = max_test_inputs

    # -----------------------------------------------------------------------
    # Public — high-level diversity scores
    # -----------------------------------------------------------------------

    def full_analysis(
        self,
        code_samples: List[str],
        test_inputs: Optional[List[Any]] = None,
    ) -> _DiversityScores:
        """Run every diversity metric and return an aggregate score object."""
        if len(code_samples) < 2:
            return _DiversityScores()

        func_div = self.measure_functional_diversity(code_samples, test_inputs)
        style_div = self.measure_style_diversity(code_samples)
        algo_div = self.measure_algorithm_diversity(code_samples)
        cplx_div = self.measure_complexity_diversity(code_samples)

        overall = (
            0.30 * func_div
            + 0.25 * style_div
            + 0.30 * algo_div
            + 0.15 * cplx_div
        )
        return _DiversityScores(
            functional=func_div,
            style=style_div,
            algorithmic=algo_div,
            complexity=cplx_div,
            overall=overall,
        )

    # -----------------------------------------------------------------------
    # measure_functional_diversity
    # -----------------------------------------------------------------------

    def measure_functional_diversity(
        self,
        code_samples: List[str],
        test_inputs: Optional[List[Any]] = None,
    ) -> float:
        """Measure the fraction of *functionally distinct* implementations.

        Two samples are functionally equivalent when they produce identical
        outputs for every input in *test_inputs*.  The returned score is

            1 − (|equivalence-classes| / |samples|)

        clamped to [0, 1].  A score of 0 means all samples are equivalent;
        a score approaching 1 means almost every sample behaves differently.
        """
        if len(code_samples) < 2:
            return 0.0

        if test_inputs is None:
            # Attempt to auto-generate inputs from the first parseable sample.
            for sample in code_samples:
                generated = self.generate_test_cases(sample, num_tests=10)
                if generated:
                    test_inputs = generated
                    break
            if not test_inputs:
                test_inputs = [(), (0,), (1,), (-1,), ("",), ([],)]

        # Build output fingerprints ----------------------------------------
        fingerprints: List[Optional[Tuple[Any, ...]]] = []
        for code in code_samples:
            fp = self._execute_and_fingerprint(code, test_inputs)
            fingerprints.append(fp)

        # Group into equivalence classes -----------------------------------
        classes: List[List[int]] = []
        for idx, fp in enumerate(fingerprints):
            if fp is None:
                # Non-runnable samples each count as their own class.
                classes.append([idx])
                continue
            placed = False
            for cls in classes:
                rep = fingerprints[cls[0]]
                if rep == fp:
                    cls.append(idx)
                    placed = True
                    break
            if not placed:
                classes.append([idx])

        n = len(code_samples)
        return 1.0 - (len(classes) / n) if len(classes) < n else 0.0

    # -----------------------------------------------------------------------
    # measure_style_diversity
    # -----------------------------------------------------------------------

    def measure_style_diversity(self, code_samples: List[str]) -> float:
        """Quantify stylistic diversity (naming conventions, formatting).

        The returned score is in [0, 1].  Higher values mean the samples
        exhibit a wider variety of coding styles.
        """
        if len(code_samples) < 2:
            return 0.0

        naming_patterns: List[Dict[str, float]] = []
        cf_patterns: List[Tuple[str, ...]] = []

        for code in code_samples:
            naming_patterns.append(self._extract_naming_convention(code))
            cf_patterns.append(self._extract_control_flow_pattern(code))

        # Naming diversity — average pairwise Jensen-Shannon style distance.
        naming_div = self._naming_distribution_diversity(naming_patterns)

        # Control-flow diversity — normalised number of unique patterns.
        unique_cf = len(set(cf_patterns))
        cf_div = (unique_cf - 1) / max(len(code_samples) - 1, 1)

        # Whitespace / formatting features ---------------------------------
        format_features: List[Tuple[int, int, bool]] = []
        for code in code_samples:
            lines = code.splitlines()
            avg_len = statistics.mean(len(l) for l in lines) if lines else 0
            indent = self._dominant_indent_width(code)
            uses_trailing = any(l.rstrip() != l for l in lines)
            format_features.append((int(avg_len), indent, uses_trailing))
        unique_fmt = len(set(format_features))
        fmt_div = (unique_fmt - 1) / max(len(code_samples) - 1, 1)

        return min(1.0, 0.45 * naming_div + 0.35 * cf_div + 0.20 * fmt_div)

    # -----------------------------------------------------------------------
    # measure_algorithm_diversity
    # -----------------------------------------------------------------------

    def measure_algorithm_diversity(self, code_samples: List[str]) -> float:
        """Score algorithmic diversity using AST structural comparison.

        The score is the *mean pairwise dissimilarity* (1 − similarity) over
        all sample pairs, using :meth:`_ast_structural_similarity`.
        """
        if len(code_samples) < 2:
            return 0.0

        dissimilarities: List[float] = []
        n = len(code_samples)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._ast_structural_similarity(
                    code_samples[i], code_samples[j]
                )
                dissimilarities.append(1.0 - sim)

        return statistics.mean(dissimilarities) if dissimilarities else 0.0

    # -----------------------------------------------------------------------
    # compute_cyclomatic_complexity
    # -----------------------------------------------------------------------

    def compute_cyclomatic_complexity(self, code: str) -> int:
        """Return the cyclomatic complexity of *code*.

        Cyclomatic complexity is defined as *E − N + 2P* but for a single
        connected component this simplifies to counting decision-points
        (``if``, ``elif``, ``for``, ``while``, ``except``, ``with``,
        ``and``, ``or``, ``assert``) + 1.
        """
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return -1

        complexity = 1  # base path
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, (ast.While,)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each `and` / `or` adds one decision point.
                complexity += len(node.values) - 1
        return complexity

    # -----------------------------------------------------------------------
    # measure_complexity_diversity
    # -----------------------------------------------------------------------

    def measure_complexity_diversity(self, code_samples: List[str]) -> float:
        """Measure variation in cyclomatic complexity across samples.

        Returns a normalised coefficient of variation (CV) clamped to [0, 1].
        A value of 0 indicates all samples share the same complexity; values
        near 1 indicate wide spread.
        """
        if len(code_samples) < 2:
            return 0.0

        complexities = [
            self.compute_cyclomatic_complexity(c) for c in code_samples
        ]
        # Filter out unparseable samples.
        valid = [c for c in complexities if c >= 0]
        if len(valid) < 2:
            return 0.0

        mean_c = statistics.mean(valid)
        if mean_c == 0:
            return 0.0

        std_c = statistics.pstdev(valid)
        cv = std_c / mean_c
        # Normalise: CV > 1.5 is effectively maximum diversity.
        return min(cv / 1.5, 1.0)

    # -----------------------------------------------------------------------
    # generate_test_cases
    # -----------------------------------------------------------------------

    def generate_test_cases(
        self,
        code: str,
        num_tests: int = 10,
    ) -> List[Tuple[Any, ...]]:
        """Auto-generate plausible test inputs by inspecting *code*.

        Heuristics:
        1. Parse the first function definition and inspect its parameters.
        2. For each parameter, produce a small pool of representative values
           based on annotation or name.
        3. Return up to *num_tests* combinations drawn from the Cartesian
           product of per-parameter pools.
        """
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return []

        func_defs: List[ast.FunctionDef] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not func_defs:
            return []

        func = func_defs[0]
        params = [
            a.arg for a in func.args.args if a.arg != "self"
        ]
        if not params:
            return [()]  # no-arg function

        # Build value pools per parameter ----------------------------------
        annotations: Dict[str, Optional[str]] = {}
        for arg in func.args.args:
            ann_str: Optional[str] = None
            if arg.annotation is not None:
                try:
                    ann_str = ast.unparse(arg.annotation)
                except Exception:
                    pass
            annotations[arg.arg] = ann_str

        pools: List[List[Any]] = []
        for pname in params:
            pools.append(self._value_pool(pname, annotations.get(pname)))

        # Cartesian-product, capped ----------------------------------------
        product = list(itertools.islice(itertools.product(*pools), num_tests))
        return product

    # -----------------------------------------------------------------------
    # check_semantic_equivalence
    # -----------------------------------------------------------------------

    def check_semantic_equivalence(
        self,
        code_a: str,
        code_b: str,
        test_inputs: Optional[List[Any]] = None,
    ) -> bool:
        """Return ``True`` if *code_a* and *code_b* are semantically equivalent.

        Equivalence is determined empirically: both snippets must produce
        identical outputs for every element of *test_inputs*.  If no inputs
        are supplied the method attempts to auto-generate them from *code_a*.
        """
        if test_inputs is None:
            test_inputs = self.generate_test_cases(code_a, num_tests=20)
            if not test_inputs:
                test_inputs = [(), (0,), (1,)]

        fp_a = self._execute_and_fingerprint(code_a, test_inputs)
        fp_b = self._execute_and_fingerprint(code_b, test_inputs)

        if fp_a is None or fp_b is None:
            return False
        return fp_a == fp_b

    # -----------------------------------------------------------------------
    # Private helpers — control-flow pattern extraction
    # -----------------------------------------------------------------------

    def _extract_control_flow_pattern(self, code: str) -> Tuple[str, ...]:
        """Return a tuple of control-flow node type names in source order.

        The tuple acts as a compact *fingerprint* of the high-level control
        structure of the code.
        """
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return ()

        pattern: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, self._CF_NODE_TYPES):
                pattern.append(type(node).__name__)
        return tuple(pattern)

    # -----------------------------------------------------------------------
    # Private helpers — naming convention extraction
    # -----------------------------------------------------------------------

    def _extract_naming_convention(self, code: str) -> Dict[str, float]:
        """Return a distribution over naming conventions found in *code*.

        Keys are ``"snake"``, ``"camel"``, ``"pascal"``, ``"upper_snake"``,
        and ``"other"``.  Values are relative frequencies summing to 1.
        """
        dist: Dict[str, float] = {
            "snake": 0.0,
            "camel": 0.0,
            "pascal": 0.0,
            "upper_snake": 0.0,
            "other": 0.0,
        }
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return dist

        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.FunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.arg):
                names.append(node.arg)

        # Filter out Python builtins / keywords / dunder names.
        _builtin_names: Set[str] = set()
        if isinstance(__builtins__, dict):
            _builtin_names = set(__builtins__.keys())
        else:
            _builtin_names = set(dir(__builtins__))
        names = [
            n for n in names
            if not keyword.iskeyword(n)
            and not n.startswith("__")
            and n not in _builtin_names
        ]

        if not names:
            return dist

        for name in names:
            if self._RE_UPPER_SNAKE.match(name):
                dist["upper_snake"] += 1
            elif self._RE_SNAKE_CASE.match(name):
                dist["snake"] += 1
            elif self._RE_PASCAL_CASE.match(name):
                dist["pascal"] += 1
            elif self._RE_CAMEL_CASE.match(name):
                dist["camel"] += 1
            else:
                dist["other"] += 1

        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        return dist

    # -----------------------------------------------------------------------
    # Private helpers — AST structural similarity
    # -----------------------------------------------------------------------

    def _ast_structural_similarity(
        self, code_a: str, code_b: str
    ) -> float:
        """Compute a structural similarity score in [0, 1] between two ASTs.

        The algorithm serialises each AST into a multiset of *node-type*
        strings, then computes the *Jaccard* similarity of the two multisets.
        A score of 1 means the ASTs share exactly the same node-type
        distribution; 0 means complete disjointness.
        """
        bag_a = self._ast_node_bag(code_a)
        bag_b = self._ast_node_bag(code_b)

        if not bag_a and not bag_b:
            return 1.0
        if not bag_a or not bag_b:
            return 0.0

        # Multiset (Counter) Jaccard similarity.
        all_keys = set(bag_a.keys()) | set(bag_b.keys())
        intersection = sum(min(bag_a.get(k, 0), bag_b.get(k, 0)) for k in all_keys)
        union = sum(max(bag_a.get(k, 0), bag_b.get(k, 0)) for k in all_keys)

        return intersection / union if union > 0 else 1.0

    # -----------------------------------------------------------------------
    # Private helpers — execution & fingerprinting
    # -----------------------------------------------------------------------

    def _execute_and_fingerprint(
        self,
        code: str,
        test_inputs: List[Any],
    ) -> Optional[Tuple[Any, ...]]:
        """Execute *code* with each element of *test_inputs* and return a
        tuple of ``repr``-ed outputs, or ``None`` on failure.
        """
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return None

        # Locate the first function definition to call.
        func_name: Optional[str] = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
        if func_name is None:
            return None

        namespace: Dict[str, Any] = {}
        try:
            exec(compile(tree, "<diversity>", "exec"), namespace)  # noqa: S102
        except Exception:
            return None

        target = namespace.get(func_name)
        if not callable(target):
            return None

        results: List[str] = []
        for inp in test_inputs:
            try:
                if isinstance(inp, tuple):
                    out = target(*inp)
                else:
                    out = target(inp)
                results.append(repr(out))
            except Exception:
                results.append("<error>")
        return tuple(results)

    # -----------------------------------------------------------------------
    # Private helpers — miscellaneous
    # -----------------------------------------------------------------------

    def _ast_node_bag(self, code: str) -> Counter[str]:
        """Return a :class:`~collections.Counter` of AST node type names."""
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return collections.Counter()
        return collections.Counter(
            type(node).__name__ for node in ast.walk(tree)
        )

    @staticmethod
    def _dominant_indent_width(code: str) -> int:
        """Detect the most common indentation width in *code*."""
        widths: Counter[int] = collections.Counter()
        for line in code.splitlines():
            stripped = line.lstrip(" ")
            if stripped and len(line) != len(stripped):
                widths[len(line) - len(stripped)] += 1
        if not widths:
            return 4
        return widths.most_common(1)[0][0]

    @staticmethod
    def _value_pool(
        param_name: str, annotation: Optional[str]
    ) -> List[Any]:
        """Return a small pool of representative values for a parameter."""
        # Annotation-based heuristics.
        if annotation is not None:
            ann_lower = annotation.lower()
            if "int" in ann_lower:
                return [0, 1, -1, 42, 100]
            if "float" in ann_lower:
                return [0.0, 1.0, -1.0, 3.14, 1e-6]
            if "str" in ann_lower:
                return ["", "a", "hello", "Hello World", "12345"]
            if "bool" in ann_lower:
                return [True, False]
            if "list" in ann_lower or "List" in annotation:
                return [[], [1], [1, 2, 3], [0, -1, 5]]
            if "dict" in ann_lower or "Dict" in annotation:
                return [{}, {"a": 1}, {"x": 0, "y": 0}]

        # Name-based heuristics.
        name = param_name.lower()
        if name in ("n", "num", "count", "size", "length", "k", "limit"):
            return [0, 1, 5, 10, 100]
        if name in ("s", "text", "word", "string", "msg", "name"):
            return ["", "a", "hello", "Hello World"]
        if name in ("flag", "enabled", "verbose", "debug"):
            return [True, False]
        if name in ("items", "values", "data", "arr", "lst", "numbers"):
            return [[], [1], [1, 2, 3], [3, 1, 2]]
        if name in ("x", "y", "z", "val", "value"):
            return [0, 1, -1, 42]

        # Fallback mix.
        return [0, 1, "", "a", True, None, [], [1]]

    def _naming_distribution_diversity(
        self, distributions: List[Dict[str, float]]
    ) -> float:
        """Compute diversity across a list of naming-convention distributions.

        Uses mean pairwise Hellinger distance normalised to [0, 1].
        """
        if len(distributions) < 2:
            return 0.0

        keys = sorted(distributions[0].keys())
        distances: List[float] = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                d = self._hellinger(distributions[i], distributions[j], keys)
                distances.append(d)
        return statistics.mean(distances) if distances else 0.0

    @staticmethod
    def _hellinger(
        p: Dict[str, float], q: Dict[str, float], keys: List[str]
    ) -> float:
        """Hellinger distance between two discrete distributions."""
        sum_sq = 0.0
        for k in keys:
            sum_sq += (math.sqrt(p.get(k, 0.0)) - math.sqrt(q.get(k, 0.0))) ** 2
        return math.sqrt(sum_sq / 2.0)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def analyze_code_diversity(
    code_samples: List[str],
    test_inputs: Optional[List[Any]] = None,
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
) -> Dict[str, float]:
    """One-shot helper that returns a diversity-score dictionary.

    Example::

        scores = analyze_code_diversity([
            "def f(n): return n * 2",
            "def f(n):\\n    result = n + n\\n    return result",
        ])
        print(scores)
        # {'functional': ..., 'style': ..., 'algorithmic': ...,
        #  'complexity': ..., 'overall': ...}
    """
    analyzer = CodeDiversityAnalyzer(language=language)
    result = analyzer.full_analysis(code_samples, test_inputs=test_inputs)
    return result.as_dict()
