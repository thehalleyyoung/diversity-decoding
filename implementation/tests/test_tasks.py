"""Comprehensive tests for the Diversity Decoding Arena task system.

Covers configuration, prompts, constraints, datasets, evaluation, domain-specific
tasks (creative writing, code generation, summarization, QA, translation,
brainstorming), task registry, serialization, and edge cases.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import string
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.tasks.base import (
    ConstraintType,
    GenerationTask,
    PromptDataset,
    TaskConfig,
    TaskConstraint,
    TaskDomain,
    TaskEvaluator,
    TaskPrompt,
    _TASK_REGISTRY,
    get_registered_tasks,
)
from src.tasks.creative_writing import (
    CreativeWritingConfig,
    CreativeWritingTask,
    DialoguePrompt,
    PoetryPrompt,
    StoryPrompt,
    WritingGenre,
    WritingStyle,
)
from src.tasks.code_generation import (
    CodeComplexity,
    CodeGenerationConfig,
    CodeGenerationTask,
    CodePrompt,
    CodeTaskType,
    CodeTestCase,
    ProgrammingLanguage,
)
from src.tasks.summarization import (
    SummarizationConfig,
    SummarizationPrompt,
    SummarizationTask,
    SummaryLength,
    SummaryType,
)
from src.tasks.qa import (
    AnswerFormat,
    QAConfig,
    QAPrompt,
    QuestionAnsweringTask,
    QuestionType,
)
from src.tasks.translation import (
    LanguagePair,
    TranslationConfig,
    TranslationDifficulty,
    TranslationPrompt,
    TranslationTask,
)
from src.tasks.brainstorming import (
    BrainstormCategory,
    BrainstormingConfig,
    BrainstormingTask,
    BrainstormPrompt,
    IdeaComplexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt(text: str = "Write a story about a cat.",
                 domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION,
                 context: str = "",
                 metadata: Optional[Dict[str, Any]] = None,
                 reference_outputs: Optional[List[str]] = None,
                 constraints: Optional[List[TaskConstraint]] = None) -> TaskPrompt:
    """Create a TaskPrompt with sensible defaults for testing."""
    return TaskPrompt(
        text=text,
        domain=domain,
        context=context,
        metadata=metadata or {},
        reference_outputs=reference_outputs or [],
        constraints=constraints or [],
    )


def _make_length_constraint(min_w: int = 10, max_w: int = 200,
                            required: bool = True) -> TaskConstraint:
    return TaskConstraint(
        constraint_type=ConstraintType.LENGTH,
        parameters={"min": min_w, "max": max_w, "unit": "words"},
        required=required,
    )


def _make_keyword_constraint(keywords: List[str],
                             mode: str = "all") -> TaskConstraint:
    return TaskConstraint(
        constraint_type=ConstraintType.KEYWORD,
        parameters={"keywords": keywords, "mode": mode},
    )


def _make_content_constraint(**kwargs: Any) -> TaskConstraint:
    return TaskConstraint(
        constraint_type=ConstraintType.CONTENT,
        parameters=kwargs,
    )


def _make_dataset(n: int = 10,
                  domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION,
                  name: str = "test") -> PromptDataset:
    """Build a small PromptDataset with n prompts."""
    prompts = [
        _make_prompt(text=f"Prompt number {i} about topic {i % 3}.", domain=domain)
        for i in range(n)
    ]
    return PromptDataset(prompts=prompts, name=name, domain=domain)


# ===================================================================
# 1. TestTaskConfig
# ===================================================================

class TestTaskConfig:
    """Configuration validation and serialization."""

    def test_default_values(self):
        cfg = TaskConfig()
        assert cfg.name == "default"
        assert cfg.domain == TaskDomain.OPEN_ENDED_GENERATION
        assert cfg.num_prompts == 100
        assert cfg.max_length == 512
        assert cfg.min_length == 10
        assert cfg.temperature == 1.0
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = TaskConfig(
            name="custom",
            domain=TaskDomain.CODE_GENERATION,
            num_prompts=50,
            max_length=1024,
            min_length=5,
            temperature=0.7,
            seed=123,
        )
        assert cfg.name == "custom"
        assert cfg.domain == TaskDomain.CODE_GENERATION
        assert cfg.num_prompts == 50
        assert cfg.max_length == 1024
        assert cfg.temperature == 0.7

    def test_validate_valid_config(self):
        cfg = TaskConfig()
        errors = cfg.validate()
        assert errors == []

    def test_validate_negative_max_length(self):
        cfg = TaskConfig(max_length=-1)
        errors = cfg.validate()
        assert len(errors) > 0
        assert any("max_length" in e.lower() or "length" in e.lower() for e in errors)

    def test_validate_negative_min_length(self):
        cfg = TaskConfig(min_length=-5)
        errors = cfg.validate()
        assert len(errors) > 0

    def test_validate_min_greater_than_max(self):
        cfg = TaskConfig(min_length=1000, max_length=100)
        errors = cfg.validate()
        assert len(errors) > 0

    def test_validate_zero_temperature(self):
        cfg = TaskConfig(temperature=0.0)
        errors = cfg.validate()
        assert len(errors) > 0

    def test_validate_negative_temperature(self):
        cfg = TaskConfig(temperature=-0.5)
        errors = cfg.validate()
        assert len(errors) > 0

    def test_validate_zero_num_prompts(self):
        cfg = TaskConfig(num_prompts=0)
        errors = cfg.validate()
        assert len(errors) > 0

    def test_to_dict_keys(self):
        cfg = TaskConfig()
        d = cfg.to_dict()
        expected_keys = {
            "name", "domain", "num_prompts", "max_length", "min_length",
            "temperature", "constraints", "evaluation_metrics",
            "prompt_template", "seed",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_domain_is_string(self):
        cfg = TaskConfig(domain=TaskDomain.SUMMARIZATION)
        d = cfg.to_dict()
        assert d["domain"] == "SUMMARIZATION"

    def test_to_dict_constraints_serialized(self):
        c = _make_length_constraint(10, 100)
        cfg = TaskConfig(constraints=[c])
        d = cfg.to_dict()
        assert len(d["constraints"]) == 1
        assert d["constraints"][0]["constraint_type"] == "LENGTH"

    def test_from_dict_roundtrip(self):
        cfg = TaskConfig(
            name="rt", domain=TaskDomain.TRANSLATION,
            num_prompts=25, max_length=256,
            min_length=20, temperature=0.8, seed=99,
        )
        d = cfg.to_dict()
        cfg2 = TaskConfig.from_dict(d)
        assert cfg2.name == cfg.name
        assert cfg2.domain == cfg.domain
        assert cfg2.num_prompts == cfg.num_prompts
        assert cfg2.max_length == cfg.max_length
        assert cfg2.temperature == cfg.temperature
        assert cfg2.seed == cfg.seed

    def test_evaluation_metrics_default(self):
        cfg = TaskConfig()
        assert "fluency" in cfg.evaluation_metrics
        assert "relevance" in cfg.evaluation_metrics
        assert "diversity" in cfg.evaluation_metrics

    def test_prompt_template_default(self):
        cfg = TaskConfig()
        assert "{text}" in cfg.prompt_template

    @pytest.mark.parametrize("domain", list(TaskDomain))
    def test_all_domains_valid(self, domain):
        cfg = TaskConfig(domain=domain)
        assert cfg.validate() == []


# ===================================================================
# 2. TestTaskPrompt
# ===================================================================

class TestTaskPrompt:
    """Prompt creation, metadata, and formatting."""

    def test_default_prompt(self):
        p = TaskPrompt()
        assert p.text == ""
        assert p.domain == TaskDomain.OPEN_ENDED_GENERATION
        assert p.metadata == {}
        assert p.reference_outputs == []
        assert p.max_gen_length == 512

    def test_custom_prompt(self):
        p = _make_prompt("Hello world", domain=TaskDomain.CODE_GENERATION)
        assert p.text == "Hello world"
        assert p.domain == TaskDomain.CODE_GENERATION

    def test_prompt_id_generated(self):
        p = _make_prompt("Some text")
        pid = p._generate_id()
        assert len(pid) == 16
        expected = hashlib.sha256("Some text".encode("utf-8")).hexdigest()[:16]
        assert pid == expected

    def test_prompt_id_deterministic(self):
        p1 = _make_prompt("Same text")
        p2 = _make_prompt("Same text")
        assert p1._generate_id() == p2._generate_id()

    def test_prompt_id_different_text(self):
        p1 = _make_prompt("Text A")
        p2 = _make_prompt("Text B")
        assert p1._generate_id() != p2._generate_id()

    def test_prompt_id_includes_context(self):
        p1 = _make_prompt("Hello", context="ctx1")
        p2 = _make_prompt("Hello", context="ctx2")
        assert p1._generate_id() != p2._generate_id()

    def test_word_count(self):
        p = _make_prompt("one two three four five")
        assert p.word_count == 5

    def test_char_count(self):
        p = _make_prompt("abc")
        assert p.char_count == 3

    def test_has_reference_true(self):
        p = _make_prompt(reference_outputs=["ref1"])
        assert p.has_reference() is True

    def test_has_reference_false(self):
        p = _make_prompt()
        assert p.has_reference() is False

    def test_metadata_dict(self):
        p = _make_prompt(metadata={"difficulty": "hard", "source": "manual"})
        assert p.metadata["difficulty"] == "hard"
        assert p.metadata["source"] == "manual"

    def test_to_dict_complete(self):
        p = _make_prompt(
            text="Test prompt",
            domain=TaskDomain.SUMMARIZATION,
            context="Some context",
            metadata={"key": "value"},
            reference_outputs=["ref"],
        )
        d = p.to_dict()
        assert d["text"] == "Test prompt"
        assert d["domain"] == "SUMMARIZATION"
        assert d["context"] == "Some context"
        assert d["metadata"] == {"key": "value"}
        assert d["reference_outputs"] == ["ref"]

    def test_from_dict_roundtrip(self):
        p = _make_prompt(
            text="Roundtrip test",
            domain=TaskDomain.QUESTION_ANSWERING,
            context="ctx",
            reference_outputs=["answer"],
        )
        d = p.to_dict()
        p2 = TaskPrompt.from_dict(d)
        assert p2.text == p.text
        assert p2.domain == p.domain
        assert p2.context == p.context
        assert p2.reference_outputs == p.reference_outputs

    def test_all_constraints_merges(self):
        local = [_make_length_constraint(10, 50)]
        global_c = [_make_keyword_constraint(["hello"])]
        p = _make_prompt(constraints=local)
        merged = p.all_constraints(global_c)
        assert len(merged) == 2


# ===================================================================
# 3. TestTaskConstraint
# ===================================================================

class TestTaskConstraint:
    """Constraint definition, validation, and checking."""

    def test_length_constraint_within_range(self):
        c = _make_length_constraint(5, 20)
        text = "This is a short text with about ten words here total."
        assert c.check(text) is True

    def test_length_constraint_too_short(self):
        c = _make_length_constraint(100, 200)
        text = "Short."
        assert c.check(text) is False

    def test_length_constraint_too_long(self):
        c = _make_length_constraint(1, 3)
        text = "This text has way more than three words in it."
        assert c.check(text) is False

    def test_keyword_constraint_all_present(self):
        c = _make_keyword_constraint(["cat", "dog"], mode="all")
        text = "The cat and the dog played together."
        assert c.check(text) is True

    def test_keyword_constraint_all_missing_one(self):
        c = _make_keyword_constraint(["cat", "elephant"], mode="all")
        text = "The cat sat on the mat."
        assert c.check(text) is False

    def test_keyword_constraint_any_present(self):
        c = _make_keyword_constraint(["cat", "elephant"], mode="any")
        text = "The cat sat on the mat."
        assert c.check(text) is True

    def test_keyword_constraint_none_present(self):
        c = _make_keyword_constraint(["xyz", "abc"], mode="none")
        text = "The cat sat on the mat."
        assert c.check(text) is True

    def test_keyword_constraint_none_violated(self):
        c = _make_keyword_constraint(["cat"], mode="none")
        text = "The cat sat on the mat."
        assert c.check(text) is False

    def test_content_constraint_banned_words(self):
        c = _make_content_constraint(banned_words=["hate", "violence"])
        text = "This is a peaceful and loving text."
        assert c.check(text) is True

    def test_content_constraint_banned_words_violated(self):
        c = _make_content_constraint(banned_words=["hate", "violence"])
        text = "This text contains hate speech."
        assert c.check(text) is False

    def test_format_constraint_regex(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.FORMAT,
            parameters={"pattern": r"^\d{4}-\d{2}-\d{2}"},
        )
        assert c.check("2024-01-15 is the date") is True
        assert c.check("January 15, 2024") is False

    def test_style_constraint_sentence_length(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.STYLE,
            parameters={"max_sentence_length": 10},
        )
        short_text = "Hello. World. Short."
        assert c.check(short_text) is True

    def test_to_dict_structure(self):
        c = _make_length_constraint(10, 100)
        d = c.to_dict()
        assert d["constraint_type"] == "LENGTH"
        assert d["parameters"]["min"] == 10
        assert d["parameters"]["max"] == 100
        assert d["required"] is True
        assert d["weight"] == 1.0

    def test_from_dict_roundtrip(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.KEYWORD,
            parameters={"keywords": ["hello"], "mode": "all"},
            required=False,
            weight=0.5,
        )
        d = c.to_dict()
        c2 = TaskConstraint.from_dict(d)
        assert c2.constraint_type == c.constraint_type
        assert c2.parameters == c.parameters
        assert c2.required == c.required
        assert c2.weight == c.weight

    def test_describe_returns_string(self):
        c = _make_length_constraint(10, 100)
        desc = c.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_weight_default(self):
        c = TaskConstraint(constraint_type=ConstraintType.LENGTH, parameters={})
        assert c.weight == 1.0

    def test_required_default(self):
        c = TaskConstraint(constraint_type=ConstraintType.LENGTH, parameters={})
        assert c.required is True


# ===================================================================
# 4. TestPromptDataset
# ===================================================================

class TestPromptDataset:
    """Dataset iteration, filtering, and sampling."""

    def test_len(self):
        ds = _make_dataset(15)
        assert len(ds) == 15

    def test_getitem_single(self):
        ds = _make_dataset(5)
        p = ds[0]
        assert isinstance(p, TaskPrompt)

    def test_getitem_slice(self):
        ds = _make_dataset(10)
        sub = ds[2:5]
        assert isinstance(sub, PromptDataset)
        assert len(sub) == 3

    def test_iteration(self):
        ds = _make_dataset(5)
        items = list(ds)
        assert len(items) == 5
        assert all(isinstance(p, TaskPrompt) for p in items)

    def test_repr(self):
        ds = _make_dataset(5, name="myds")
        r = repr(ds)
        assert "myds" in r or "PromptDataset" in r

    def test_add_datasets(self):
        ds1 = _make_dataset(5, name="a")
        ds2 = _make_dataset(3, name="b")
        combined = ds1 + ds2
        assert len(combined) == 8
        assert "a" in combined.name and "b" in combined.name

    def test_sample_returns_correct_count(self):
        ds = _make_dataset(20)
        sampled = ds.sample(5, seed=42)
        assert len(sampled) == 5

    def test_sample_deterministic(self):
        ds = _make_dataset(20)
        s1 = ds.sample(5, seed=42)
        s2 = ds.sample(5, seed=42)
        assert [p.text for p in s1] == [p.text for p in s2]

    def test_sample_different_seeds(self):
        ds = _make_dataset(20)
        s1 = ds.sample(5, seed=42)
        s2 = ds.sample(5, seed=99)
        texts1 = [p.text for p in s1]
        texts2 = [p.text for p in s2]
        assert texts1 != texts2

    def test_sample_dataset(self):
        ds = _make_dataset(20)
        sub = ds.sample_dataset(5, seed=42)
        assert isinstance(sub, PromptDataset)
        assert len(sub) == 5

    def test_filter_by_predicate(self):
        ds = _make_dataset(10)
        filtered = ds.filter_by(lambda p: "0" in p.text)
        assert len(filtered) > 0
        assert all("0" in p.text for p in filtered)

    def test_filter_by_domain(self):
        prompts = [
            _make_prompt(f"P{i}", domain=TaskDomain.CODE_GENERATION if i % 2 == 0
                         else TaskDomain.SUMMARIZATION)
            for i in range(10)
        ]
        ds = PromptDataset(prompts, name="mixed")
        code_ds = ds.filter_by_domain(TaskDomain.CODE_GENERATION)
        assert len(code_ds) == 5
        assert all(p.domain == TaskDomain.CODE_GENERATION for p in code_ds)

    def test_filter_by_length(self):
        prompts = [
            _make_prompt(" ".join(["word"] * (i + 1)))
            for i in range(20)
        ]
        ds = PromptDataset(prompts, name="lengths")
        filtered = ds.filter_by_length(min_words=5, max_words=10)
        for p in filtered:
            wc = p.word_count
            assert 5 <= wc <= 10

    def test_split_fractions(self):
        ds = _make_dataset(100)
        train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == 100

    def test_split_invalid_fractions(self):
        ds = _make_dataset(100)
        with pytest.raises(ValueError, match="sum to 1"):
            ds.split(0.5, 0.1, 0.1)

    def test_split_names(self):
        ds = _make_dataset(50, name="base")
        train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
        assert "train" in train.name
        assert "val" in val.name
        assert "test" in test.name

    def test_statistics_keys(self):
        ds = _make_dataset(10)
        stats = ds.statistics()
        assert stats["count"] == 10
        assert "length_mean" in stats
        assert "vocab_size" in stats
        assert "has_reference_frac" in stats

    def test_statistics_empty_dataset(self):
        ds = PromptDataset(prompts=[], name="empty")
        stats = ds.statistics()
        assert stats["count"] == 0

    def test_empty_dataset_len(self):
        ds = PromptDataset(prompts=[], name="empty")
        assert len(ds) == 0
        assert list(ds) == []

    def test_to_json_from_json_roundtrip(self):
        ds = _make_dataset(5, name="jsontest", domain=TaskDomain.SUMMARIZATION)
        json_str = ds.to_json()
        ds2 = PromptDataset.from_json(json_str)
        assert len(ds2) == len(ds)
        assert ds2.name == ds.name
        for p1, p2 in zip(ds, ds2):
            assert p1.text == p2.text


# ===================================================================
# 5. TestCreativeWriting
# ===================================================================

class TestCreativeWriting:
    """Creative writing task: prompt generation, evaluation, constraints."""

    def test_default_config(self):
        cfg = CreativeWritingConfig()
        assert cfg.genre == WritingGenre.FICTION
        assert cfg.style == WritingStyle.LITERARY

    def test_custom_config(self):
        cfg = CreativeWritingConfig(
            genre=WritingGenre.POETRY,
            style=WritingStyle.MINIMALIST,
            min_words=50,
            max_words=500,
        )
        assert cfg.genre == WritingGenre.POETRY
        assert cfg.min_words == 50

    def test_task_instantiation(self):
        task = CreativeWritingTask()
        assert isinstance(task, GenerationTask)
        assert isinstance(task, CreativeWritingTask)

    def test_task_with_config(self):
        cfg = CreativeWritingConfig(genre=WritingGenre.DIALOGUE)
        task = CreativeWritingTask(config=cfg)
        assert task.config.genre == WritingGenre.DIALOGUE

    def test_load_prompts(self):
        task = CreativeWritingTask()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    def test_format_prompt_returns_string(self):
        task = CreativeWritingTask()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_get_constraints(self):
        task = CreativeWritingTask()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)
        assert all(isinstance(c, TaskConstraint) for c in constraints)

    def test_evaluate_returns_dict(self):
        task = CreativeWritingTask()
        ds = task.load_prompts()
        prompts = [ds[0], ds[1]] if len(ds) >= 2 else [ds[0]]
        generations = [
            "Once upon a time, in a land far away, there lived a brave knight. "
            "He traveled through forests and mountains, seeking adventure. "
            "The sun set over the horizon as he reached a mysterious castle.",
        ] * len(prompts)
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_story_prompt_dataclass(self):
        sp = StoryPrompt(
            text="Write a mystery story",
            genre=WritingGenre.FICTION,
            setting="Victorian London",
            characters=["detective", "suspect"],
            theme="justice",
        )
        assert sp.genre == WritingGenre.FICTION
        assert sp.setting == "Victorian London"
        assert len(sp.characters) == 2

    def test_poetry_prompt_dataclass(self):
        pp = PoetryPrompt(
            text="Write a sonnet about nature",
            form="sonnet",
            rhyme_scheme="ABAB CDCD EFEF GG",
        )
        assert pp.form == "sonnet"
        assert "ABAB" in pp.rhyme_scheme

    def test_dialogue_prompt_dataclass(self):
        dp = DialoguePrompt(
            text="Write a dialogue between friends",
            num_speakers=3,
            setting="coffee shop",
            topic="weekend plans",
        )
        assert dp.num_speakers == 3
        assert dp.setting == "coffee shop"

    def test_story_prompt_as_text(self):
        sp = StoryPrompt(
            text="Write a story",
            setting="Space station",
            characters=["Captain"],
            theme="isolation",
        )
        text = sp.as_text()
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.parametrize("genre", list(WritingGenre))
    def test_all_genres_instantiate(self, genre):
        cfg = CreativeWritingConfig(genre=genre)
        task = CreativeWritingTask(config=cfg)
        assert task.config.genre == genre

    @pytest.mark.parametrize("style", list(WritingStyle))
    def test_all_styles_instantiate(self, style):
        cfg = CreativeWritingConfig(style=style)
        task = CreativeWritingTask(config=cfg)
        assert task.config.style == style

    def test_validate_generation(self):
        task = CreativeWritingTask()
        ds = task.load_prompts()
        text = (
            "The old lighthouse keeper watched the storm approach. "
            "Waves crashed against the rocky shore. "
            "He lit the lamp, its beam cutting through the darkness."
        )
        valid, errors = task.validate_generation(text, ds[0])
        assert isinstance(valid, bool)
        assert isinstance(errors, list)

    def test_post_process_strips(self):
        task = CreativeWritingTask()
        result = task.post_process("  Hello world.  ")
        assert result == "Hello world."


# ===================================================================
# 6. TestCodeGeneration
# ===================================================================

class TestCodeGeneration:
    """Code generation task: prompts, syntax, evaluation."""

    def test_default_config(self):
        cfg = CodeGenerationConfig()
        assert cfg.language == ProgrammingLanguage.PYTHON
        assert cfg.task_type == CodeTaskType.FUNCTION_SYNTHESIS
        assert cfg.complexity == CodeComplexity.MEDIUM

    def test_custom_config(self):
        cfg = CodeGenerationConfig(
            language=ProgrammingLanguage.JAVASCRIPT,
            task_type=CodeTaskType.BUG_FIX,
            complexity=CodeComplexity.HARD,
            require_docstrings=False,
        )
        assert cfg.language == ProgrammingLanguage.JAVASCRIPT
        assert cfg.require_docstrings is False

    def test_task_instantiation(self):
        task = CodeGenerationTask()
        assert isinstance(task, GenerationTask)

    def test_load_prompts(self):
        task = CodeGenerationTask()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    def test_format_prompt_returns_string(self):
        task = CodeGenerationTask()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_get_constraints(self):
        task = CodeGenerationTask()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)

    def test_evaluate_returns_dict(self):
        task = CodeGenerationTask()
        ds = task.load_prompts()
        prompts = [ds[0]]
        generations = [
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n"
        ]
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)

    def test_code_prompt_dataclass(self):
        cp = CodePrompt(
            text="Implement binary search",
            language=ProgrammingLanguage.PYTHON,
            function_signature="def binary_search(arr: List[int], target: int) -> int",
            docstring="Find target in sorted array, return index or -1.",
            expected_complexity=CodeComplexity.EASY,
            input_types=["List[int]", "int"],
            output_type="int",
        )
        assert cp.language == ProgrammingLanguage.PYTHON
        assert "binary_search" in cp.function_signature
        assert cp.expected_complexity == CodeComplexity.EASY

    def test_code_test_case(self):
        tc = CodeTestCase(
            input_args=[[1, 2, 3, 4, 5], 3],
            expected_output=2,
            description="Find element in sorted array",
            is_edge_case=False,
        )
        assert tc.expected_output == 2
        assert tc.is_edge_case is False

    def test_code_prompt_with_test_cases(self):
        tc1 = CodeTestCase(input_args=[[1, 2, 3], 2], expected_output=1)
        tc2 = CodeTestCase(input_args=[[], 1], expected_output=-1, is_edge_case=True)
        cp = CodePrompt(
            text="Binary search",
            test_cases=[tc1, tc2],
        )
        assert len(cp.test_cases) == 2
        assert cp.test_cases[1].is_edge_case is True

    @pytest.mark.parametrize("lang", list(ProgrammingLanguage))
    def test_all_languages_instantiate(self, lang):
        cfg = CodeGenerationConfig(language=lang)
        assert cfg.language == lang

    @pytest.mark.parametrize("complexity", list(CodeComplexity))
    def test_all_complexities_valid(self, complexity):
        cfg = CodeGenerationConfig(complexity=complexity)
        assert cfg.complexity == complexity

    def test_validate_generation(self):
        task = CodeGenerationTask()
        ds = task.load_prompts()
        code = "def hello():\n    return 'hello'\n"
        valid, errors = task.validate_generation(code, ds[0])
        assert isinstance(valid, bool)
        assert isinstance(errors, list)

    def test_post_process_code(self):
        task = CodeGenerationTask()
        code = "  def foo():\n      return 1  "
        result = task.post_process(code)
        assert isinstance(result, str)


# ===================================================================
# 7. TestSummarization
# ===================================================================

class TestSummarization:
    """Summarization: source handling, compression, evaluation."""

    def test_default_config(self):
        cfg = SummarizationConfig()
        assert cfg.summary_type == SummaryType.ABSTRACTIVE
        assert cfg.target_length == SummaryLength.MEDIUM

    def test_custom_config(self):
        cfg = SummarizationConfig(
            summary_type=SummaryType.EXTRACTIVE,
            target_length=SummaryLength.SHORT,
            compression_ratio=0.1,
            preserve_entities=True,
        )
        assert cfg.summary_type == SummaryType.EXTRACTIVE
        assert cfg.compression_ratio == 0.1

    def test_task_instantiation(self):
        task = SummarizationTask()
        assert isinstance(task, GenerationTask)

    def test_load_prompts(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    def test_format_prompt_returns_string(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_get_constraints(self):
        task = SummarizationTask()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)

    def test_evaluate_returns_dict(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        prompts = [ds[0]]
        generations = [
            "Machine learning models learn patterns from data to make predictions. "
            "Deep learning uses neural networks with multiple layers."
        ]
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)

    def test_summarization_prompt_dataclass(self):
        sp = SummarizationPrompt(
            text="Summarize the following article.",
            source_document="A long article about climate change and its effects on ecosystems worldwide.",
            summary_type=SummaryType.ABSTRACTIVE,
            target_length=SummaryLength.SHORT,
            key_points=["climate change", "ecosystems", "global effects"],
        )
        assert sp.summary_type == SummaryType.ABSTRACTIVE
        assert len(sp.key_points) == 3
        assert "climate" in sp.source_document

    @pytest.mark.parametrize("stype", list(SummaryType))
    def test_all_summary_types_instantiate(self, stype):
        cfg = SummarizationConfig(summary_type=stype)
        assert cfg.summary_type == stype

    @pytest.mark.parametrize("slength", list(SummaryLength))
    def test_all_summary_lengths_valid(self, slength):
        cfg = SummarizationConfig(target_length=slength)
        assert cfg.target_length == slength

    def test_summary_length_word_range(self):
        for sl in SummaryLength:
            wr = sl.word_range
            assert isinstance(wr, tuple)
            assert len(wr) == 2
            assert wr[0] <= wr[1]

    def test_compression_ratio_range(self):
        cfg = SummarizationConfig(compression_ratio=0.3)
        assert 0.0 < cfg.compression_ratio <= 1.0

    def test_summarization_prompt_with_references(self):
        sp = SummarizationPrompt(
            text="Summarize this",
            source_document="Long source text here.",
            reference_summaries=["Short summary one.", "Short summary two."],
        )
        assert len(sp.reference_summaries) == 2

    def test_validate_generation(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        summary = "This is a brief summary of the key points discussed in the text."
        valid, errors = task.validate_generation(summary, ds[0])
        assert isinstance(valid, bool)


# ===================================================================
# 8. TestQuestionAnswering
# ===================================================================

class TestQuestionAnswering:
    """QA task: question format, answer evaluation."""

    def test_default_config(self):
        cfg = QAConfig()
        assert cfg.question_type == QuestionType.FACTUAL
        assert cfg.answer_format == AnswerFormat.PARAGRAPH

    def test_custom_config(self):
        cfg = QAConfig(
            question_type=QuestionType.ANALYTICAL,
            answer_format=AnswerFormat.STEP_BY_STEP,
            require_evidence=True,
            max_answer_length=256,
        )
        assert cfg.question_type == QuestionType.ANALYTICAL
        assert cfg.require_evidence is True

    def test_task_instantiation(self):
        task = QuestionAnsweringTask()
        assert isinstance(task, GenerationTask)

    def test_load_prompts(self):
        task = QuestionAnsweringTask()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    def test_format_prompt_returns_string(self):
        task = QuestionAnsweringTask()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)

    def test_get_constraints(self):
        task = QuestionAnsweringTask()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)

    def test_evaluate_returns_dict(self):
        task = QuestionAnsweringTask()
        ds = task.load_prompts()
        prompts = [ds[0]]
        generations = [
            "Photosynthesis is the process by which plants convert sunlight, "
            "water, and carbon dioxide into glucose and oxygen. This process "
            "occurs in the chloroplasts of plant cells."
        ]
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)

    def test_qa_prompt_dataclass(self):
        qp = QAPrompt(
            text="What is photosynthesis?",
            question="What is photosynthesis?",
            question_type=QuestionType.FACTUAL,
            context_passage="Plants use sunlight to make food.",
            reference_answers=["Process of converting light to energy."],
            difficulty=0.3,
        )
        assert qp.question_type == QuestionType.FACTUAL
        assert qp.difficulty == 0.3
        assert len(qp.reference_answers) == 1

    def test_qa_prompt_with_evidence(self):
        qp = QAPrompt(
            text="Explain gravity",
            question="How does gravity work?",
            evidence_passages=["Newton's law of gravitation states..."],
        )
        assert len(qp.evidence_passages) == 1

    @pytest.mark.parametrize("qtype", list(QuestionType))
    def test_all_question_types_valid(self, qtype):
        cfg = QAConfig(question_type=qtype)
        assert cfg.question_type == qtype

    @pytest.mark.parametrize("afmt", list(AnswerFormat))
    def test_all_answer_formats_valid(self, afmt):
        cfg = QAConfig(answer_format=afmt)
        assert cfg.answer_format == afmt

    def test_qa_prompt_difficulty_range(self):
        for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
            qp = QAPrompt(text="Q", difficulty=d)
            assert 0.0 <= qp.difficulty <= 1.0

    def test_validate_generation(self):
        task = QuestionAnsweringTask()
        ds = task.load_prompts()
        answer = "Gravity is a fundamental force that attracts objects with mass toward each other."
        valid, errors = task.validate_generation(answer, ds[0])
        assert isinstance(valid, bool)

    def test_qa_prompt_multiple_perspectives(self):
        qp = QAPrompt(
            text="Is social media beneficial?",
            question_type=QuestionType.OPINION,
            required_perspectives=["positive", "negative", "neutral"],
        )
        assert len(qp.required_perspectives) == 3


# ===================================================================
# 9. TestTranslation
# ===================================================================

class TestTranslation:
    """Translation task: language pairs, evaluation."""

    def test_default_config(self):
        cfg = TranslationConfig()
        assert cfg.difficulty == TranslationDifficulty.LITERAL

    def test_custom_config(self):
        cfg = TranslationConfig(
            difficulty=TranslationDifficulty.IDIOMATIC,
            preserve_register=True,
            formality_level="formal",
        )
        assert cfg.difficulty == TranslationDifficulty.IDIOMATIC
        assert cfg.formality_level == "formal"

    def test_language_pair_creation(self):
        lp = LanguagePair(source_lang="en", target_lang="de")
        assert lp.source_lang == "en"
        assert lp.target_lang == "de"

    def test_language_pair_reversed(self):
        lp = LanguagePair(source_lang="en", target_lang="fr")
        rev = lp.reversed()
        assert rev.source_lang == "fr"
        assert rev.target_lang == "en"

    def test_language_pair_str(self):
        lp = LanguagePair(source_lang="en", target_lang="es")
        s = str(lp)
        assert "en" in s and "es" in s

    def test_task_instantiation(self):
        task = TranslationTask()
        assert isinstance(task, GenerationTask)

    def test_load_prompts(self):
        task = TranslationTask()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    def test_format_prompt_returns_string(self):
        task = TranslationTask()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)

    def test_get_constraints(self):
        task = TranslationTask()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)

    def test_evaluate_returns_dict(self):
        task = TranslationTask()
        ds = task.load_prompts()
        prompts = [ds[0]]
        generations = ["Bonjour le monde, comment allez-vous aujourd'hui?"]
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)

    def test_translation_prompt_dataclass(self):
        tp = TranslationPrompt(
            text="Translate to French",
            source_text="Hello, how are you?",
            source_lang="en",
            target_lang="fr",
            reference_translations=["Bonjour, comment allez-vous?"],
            difficulty=TranslationDifficulty.LITERAL,
        )
        assert tp.source_lang == "en"
        assert tp.target_lang == "fr"
        assert len(tp.reference_translations) == 1

    def test_translation_prompt_with_glossary(self):
        tp = TranslationPrompt(
            text="Translate",
            source_text="The machine learning model is trained on data.",
            glossary={"machine learning": "apprentissage automatique"},
        )
        assert "machine learning" in tp.glossary

    @pytest.mark.parametrize("diff", list(TranslationDifficulty))
    def test_all_difficulties_valid(self, diff):
        cfg = TranslationConfig(difficulty=diff)
        assert cfg.difficulty == diff

    def test_translation_prompt_notes(self):
        tp = TranslationPrompt(
            text="Translate",
            source_text="It's raining cats and dogs.",
            notes="This is an idiom meaning heavy rain.",
        )
        assert "idiom" in tp.notes

    def test_validate_generation(self):
        task = TranslationTask()
        ds = task.load_prompts()
        translation = "Ceci est une traduction de test."
        valid, errors = task.validate_generation(translation, ds[0])
        assert isinstance(valid, bool)


# ===================================================================
# 10. TestBrainstorming
# ===================================================================

class TestBrainstorming:
    """Brainstorming task: idea generation, uniqueness."""

    def test_default_config(self):
        cfg = BrainstormingConfig()
        assert cfg.category == BrainstormCategory.PRODUCT_IDEAS
        assert cfg.min_ideas == 5
        assert cfg.max_ideas == 20

    def test_custom_config(self):
        cfg = BrainstormingConfig(
            category=BrainstormCategory.SOLUTIONS,
            min_ideas=3,
            max_ideas=10,
            require_explanations=True,
            novelty_threshold=0.5,
        )
        assert cfg.category == BrainstormCategory.SOLUTIONS
        assert cfg.novelty_threshold == 0.5

    def test_task_instantiation(self):
        task = BrainstormingTask()
        assert isinstance(task, GenerationTask)

    def test_load_prompts(self):
        task = BrainstormingTask()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    def test_format_prompt_returns_string(self):
        task = BrainstormingTask()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)

    def test_get_constraints(self):
        task = BrainstormingTask()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)

    def test_evaluate_returns_dict(self):
        task = BrainstormingTask()
        ds = task.load_prompts()
        prompts = [ds[0]]
        generations = [
            "1. Solar-powered phone charger\n"
            "2. Biodegradable packaging materials\n"
            "3. Vertical garden for small apartments\n"
            "4. Rainwater collection system\n"
            "5. Composting mobile app tracker\n"
        ]
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)

    def test_brainstorm_prompt_dataclass(self):
        bp = BrainstormPrompt(
            text="Generate product ideas",
            category=BrainstormCategory.PRODUCT_IDEAS,
            complexity_level=IdeaComplexity.MODERATE,
            topic="sustainable living",
            num_ideas_requested=10,
        )
        assert bp.category == BrainstormCategory.PRODUCT_IDEAS
        assert bp.num_ideas_requested == 10

    @pytest.mark.parametrize("cat", list(BrainstormCategory))
    def test_all_categories_valid(self, cat):
        cfg = BrainstormingConfig(category=cat)
        assert cfg.category == cat

    @pytest.mark.parametrize("complexity", list(IdeaComplexity))
    def test_all_idea_complexities_valid(self, complexity):
        bp = BrainstormPrompt(text="Ideas", complexity_level=complexity)
        assert bp.complexity_level == complexity

    def test_brainstorm_category_description(self):
        for cat in BrainstormCategory:
            desc = cat.description
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_idea_complexity_min_word_count(self):
        for ic in IdeaComplexity:
            mwc = ic.min_word_count
            assert isinstance(mwc, int)
            assert mwc >= 0

    def test_idea_complexity_elaboration(self):
        for ic in IdeaComplexity:
            elab = ic.elaboration_expectation
            assert isinstance(elab, str)

    def test_brainstorm_prompt_with_constraints_text(self):
        bp = BrainstormPrompt(
            text="Ideas for reducing waste",
            topic="waste reduction",
            constraints="Must be implementable in under $100",
        )
        assert bp.constraints is not None
        assert "$100" in bp.constraints

    def test_brainstorm_prompt_reference_ideas(self):
        bp = BrainstormPrompt(
            text="More ideas",
            reference_ideas=["idea 1", "idea 2"],
        )
        assert len(bp.reference_ideas) == 2

    def test_validate_generation(self):
        task = BrainstormingTask()
        ds = task.load_prompts()
        ideas = "1. First idea\n2. Second idea\n3. Third idea"
        valid, errors = task.validate_generation(ideas, ds[0])
        assert isinstance(valid, bool)


# ===================================================================
# 11. TestTaskEvaluator
# ===================================================================

class TestTaskEvaluator:
    """Quality scoring and constraint checking."""

    def test_default_instantiation(self):
        ev = TaskEvaluator()
        assert isinstance(ev, TaskEvaluator)

    def test_custom_metrics_config(self):
        ev = TaskEvaluator(metrics_config={"fluency_weight": 0.5})
        assert isinstance(ev, TaskEvaluator)

    def test_fluency_score_normal_text(self):
        ev = TaskEvaluator()
        text = "The quick brown fox jumps over the lazy dog. It was a beautiful day."
        score = ev.fluency_score(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # reasonable English text should score well

    def test_fluency_score_empty(self):
        ev = TaskEvaluator()
        assert ev.fluency_score("") == 0.0

    def test_fluency_score_whitespace_only(self):
        ev = TaskEvaluator()
        assert ev.fluency_score("   ") == 0.0

    def test_fluency_score_gibberish(self):
        ev = TaskEvaluator()
        score = ev.fluency_score("xyzq qwrt plmk jhnb")
        # gibberish should score lower than real text
        normal_score = ev.fluency_score("The cat sat on the mat.")
        assert score <= normal_score

    def test_relevance_score_high_overlap(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("Write about cats and dogs playing together")
        text = "The cats and dogs played together in the park happily."
        score = ev.relevance_score(text, prompt)
        assert 0.0 <= score <= 1.0
        assert score > 0.1

    def test_relevance_score_no_overlap(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("Write about quantum physics")
        text = "Basketball games are exciting entertainment for fans."
        score = ev.relevance_score(text, prompt)
        assert 0.0 <= score <= 1.0

    def test_relevance_score_empty_text(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("Something")
        assert ev.relevance_score("", prompt) == 0.0

    def test_relevance_score_empty_prompt(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("")
        score = ev.relevance_score("Hello world", prompt)
        assert 0.0 <= score <= 1.0

    def test_evaluate_single_keys(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("Write about nature")
        gen = "Nature is beautiful. Trees grow tall and rivers flow gently."
        scores = ev.evaluate_single(gen, prompt)
        assert "fluency" in scores
        assert "relevance" in scores
        assert "length_ratio" in scores
        assert "lexical_diversity" in scores

    def test_evaluate_single_with_reference(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("What is water?")
        gen = "Water is a clear liquid essential for life."
        ref = "Water is a transparent chemical substance essential for all forms of life."
        scores = ev.evaluate_single(gen, prompt, reference=ref)
        assert "reference_overlap" in scores
        assert "bleu_1gram" in scores

    def test_evaluate_set(self):
        ev = TaskEvaluator()
        prompts = [_make_prompt(f"Topic {i}") for i in range(3)]
        generations = [
            "First generation about the topic at hand.",
            "Second generation discusses different aspects.",
            "Third generation provides another perspective entirely.",
        ]
        scores = ev.evaluate_set(generations, prompts)
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_check_constraints(self):
        ev = TaskEvaluator()
        constraints = [_make_length_constraint(3, 50)]
        text = "This is a valid text with enough words."
        results = ev.check_constraints(text, constraints)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0] is True

    def test_weighted_constraint_score(self):
        ev = TaskEvaluator()
        constraints = [
            _make_length_constraint(3, 50),
            _make_keyword_constraint(["hello"], mode="all"),
        ]
        text = "This text says hello to the whole world."
        score = ev.weighted_constraint_score(text, constraints)
        assert 0.0 <= score <= 1.0

    def test_diversity_within_set(self):
        ev = TaskEvaluator()
        generations = [
            "The cat sat on the mat.",
            "Dogs love to play in the park.",
            "Birds fly high in the clear blue sky.",
        ]
        div = ev.diversity_within_set(generations)
        assert 0.0 <= div <= 1.0

    def test_diversity_identical_texts(self):
        ev = TaskEvaluator()
        generations = ["Same text here."] * 5
        div = ev.diversity_within_set(generations)
        assert div == 0.0

    def test_ngram_diversity(self):
        ev = TaskEvaluator()
        generations = [
            "The quick brown fox jumps over the lazy dog.",
            "A slow red cat crawls under the active hamster.",
            "One bright green parrot flies above the sleepy turtle.",
        ]
        score = ev.ngram_diversity(generations, n=2)
        assert 0.0 <= score <= 1.0

    def test_self_bleu(self):
        ev = TaskEvaluator()
        generations = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing handles text and speech.",
        ]
        sb = ev.self_bleu(generations, n=4)
        assert 0.0 <= sb <= 1.0

    def test_lexical_diversity_via_ttr(self):
        ev = TaskEvaluator()
        ttr = ev._type_token_ratio("The cat sat on the mat near the hat.")
        assert 0.0 <= ttr <= 1.0


# ===================================================================
# 12. TestTaskRegistry
# ===================================================================

class TestTaskRegistry:
    """Task registration, lookup, and listing."""

    def test_list_registered_returns_list(self):
        result = GenerationTask.list_registered()
        assert isinstance(result, list)

    def test_list_registered_sorted(self):
        result = GenerationTask.list_registered()
        assert result == sorted(result)

    def test_register_new_task(self):
        @GenerationTask.register("test-dummy-task-12345")
        class DummyTask(GenerationTask):
            def load_prompts(self) -> PromptDataset:
                return _make_dataset(1)

            def format_prompt(self, prompt: TaskPrompt) -> str:
                return prompt.text

            def evaluate(self, generations, prompts):
                return {"score": 1.0}

            def get_constraints(self):
                return []

        assert "test-dummy-task-12345" in GenerationTask.list_registered()
        # Cleanup
        _TASK_REGISTRY.pop("test-dummy-task-12345", None)

    def test_from_registry_valid(self):
        @GenerationTask.register("test-registry-lookup-789")
        class LookupTask(GenerationTask):
            def load_prompts(self):
                return _make_dataset(1)

            def format_prompt(self, prompt):
                return prompt.text

            def evaluate(self, generations, prompts):
                return {}

            def get_constraints(self):
                return []

        task = GenerationTask.from_registry("test-registry-lookup-789")
        assert isinstance(task, LookupTask)
        _TASK_REGISTRY.pop("test-registry-lookup-789", None)

    def test_from_registry_invalid(self):
        with pytest.raises(KeyError, match="Unknown task"):
            GenerationTask.from_registry("nonexistent-task-xyz-999")

    def test_from_registry_with_config(self):
        @GenerationTask.register("test-config-task-456")
        class ConfigTask(GenerationTask):
            def load_prompts(self):
                return _make_dataset(1)

            def format_prompt(self, prompt):
                return prompt.text

            def evaluate(self, generations, prompts):
                return {}

            def get_constraints(self):
                return []

        cfg = TaskConfig(name="custom-cfg", num_prompts=5)
        task = GenerationTask.from_registry("test-config-task-456", config=cfg)
        assert task.config.name == "custom-cfg"
        _TASK_REGISTRY.pop("test-config-task-456", None)

    def test_get_registered_tasks_function(self):
        result = get_registered_tasks()
        assert isinstance(result, dict)

    def test_get_registered_tasks_is_copy(self):
        r1 = get_registered_tasks()
        r2 = get_registered_tasks()
        assert r1 is not r2

    def test_register_duplicate_overwrites(self):
        @GenerationTask.register("test-dup-task-111")
        class DupTask1(GenerationTask):
            def load_prompts(self):
                return _make_dataset(1)
            def format_prompt(self, prompt):
                return "v1"
            def evaluate(self, generations, prompts):
                return {}
            def get_constraints(self):
                return []

        @GenerationTask.register("test-dup-task-111")
        class DupTask2(GenerationTask):
            def load_prompts(self):
                return _make_dataset(1)
            def format_prompt(self, prompt):
                return "v2"
            def evaluate(self, generations, prompts):
                return {}
            def get_constraints(self):
                return []

        task = GenerationTask.from_registry("test-dup-task-111")
        assert isinstance(task, DupTask2)
        _TASK_REGISTRY.pop("test-dup-task-111", None)

    def test_register_returns_class_unchanged(self):
        @GenerationTask.register("test-unchanged-222")
        class OriginalTask(GenerationTask):
            custom_attr = "hello"
            def load_prompts(self):
                return _make_dataset(1)
            def format_prompt(self, prompt):
                return prompt.text
            def evaluate(self, generations, prompts):
                return {}
            def get_constraints(self):
                return []

        assert OriginalTask.custom_attr == "hello"
        _TASK_REGISTRY.pop("test-unchanged-222", None)

    def test_registry_isolation(self):
        """Registered tasks don't affect other task classes."""
        before = set(GenerationTask.list_registered())

        @GenerationTask.register("test-isolation-333")
        class IsolationTask(GenerationTask):
            def load_prompts(self):
                return _make_dataset(1)
            def format_prompt(self, prompt):
                return prompt.text
            def evaluate(self, generations, prompts):
                return {}
            def get_constraints(self):
                return []

        after = set(GenerationTask.list_registered())
        assert after - before == {"test-isolation-333"}
        _TASK_REGISTRY.pop("test-isolation-333", None)


# ===================================================================
# 13. TestTaskSerialization
# ===================================================================

class TestTaskSerialization:
    """JSON round-trip and config persistence."""

    def test_task_prompt_json_roundtrip(self):
        p = _make_prompt(
            text="Serialize me",
            domain=TaskDomain.CODE_GENERATION,
            context="some context",
            metadata={"score": 0.9},
            reference_outputs=["ref1", "ref2"],
        )
        d = p.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        p2 = TaskPrompt.from_dict(loaded)
        assert p2.text == p.text
        assert p2.domain == p.domain
        assert p2.context == p.context
        assert p2.reference_outputs == p.reference_outputs

    def test_task_config_json_roundtrip(self):
        cfg = TaskConfig(
            name="serialize-test",
            domain=TaskDomain.TRANSLATION,
            num_prompts=42,
            max_length=256,
            min_length=5,
            temperature=0.7,
            seed=777,
        )
        d = cfg.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        cfg2 = TaskConfig.from_dict(loaded)
        assert cfg2.name == cfg.name
        assert cfg2.domain == cfg.domain
        assert cfg2.num_prompts == cfg.num_prompts
        assert cfg2.seed == cfg.seed

    def test_constraint_json_roundtrip(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.KEYWORD,
            parameters={"keywords": ["alpha", "beta"], "mode": "any"},
            required=False,
            weight=0.75,
        )
        d = c.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        c2 = TaskConstraint.from_dict(loaded)
        assert c2.constraint_type == c.constraint_type
        assert c2.parameters == c.parameters
        assert c2.required == c.required
        assert c2.weight == c.weight

    def test_dataset_json_roundtrip(self):
        ds = _make_dataset(8, name="ser-ds", domain=TaskDomain.SUMMARIZATION)
        json_str = ds.to_json()
        ds2 = PromptDataset.from_json(json_str)
        assert len(ds2) == 8
        assert ds2.name == "ser-ds"
        for p1, p2 in zip(ds, ds2):
            assert p1.text == p2.text

    def test_dataset_json_valid_json(self):
        ds = _make_dataset(3)
        json_str = ds.to_json()
        parsed = json.loads(json_str)
        assert "name" in parsed
        assert "domain" in parsed
        assert "prompts" in parsed
        assert isinstance(parsed["prompts"], list)

    def test_config_with_constraints_roundtrip(self):
        constraints = [
            _make_length_constraint(10, 200),
            _make_keyword_constraint(["must", "include"]),
        ]
        cfg = TaskConfig(name="constrained", constraints=constraints)
        d = cfg.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        cfg2 = TaskConfig.from_dict(loaded)
        assert len(cfg2.constraints) == 2
        assert cfg2.constraints[0].constraint_type == ConstraintType.LENGTH
        assert cfg2.constraints[1].constraint_type == ConstraintType.KEYWORD

    def test_prompt_with_constraints_roundtrip(self):
        c = _make_length_constraint(5, 100)
        p = _make_prompt("Constrained prompt", constraints=[c])
        d = p.to_dict()
        p2 = TaskPrompt.from_dict(d)
        assert len(p2.constraints) == 1
        assert p2.constraints[0].constraint_type == ConstraintType.LENGTH

    def test_nested_metadata_serialization(self):
        p = _make_prompt(
            text="Nested",
            metadata={
                "scores": {"fluency": 0.9, "relevance": 0.8},
                "tags": ["important", "verified"],
                "count": 42,
            },
        )
        d = p.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        p2 = TaskPrompt.from_dict(loaded)
        assert p2.metadata["scores"]["fluency"] == 0.9
        assert p2.metadata["tags"] == ["important", "verified"]
        assert p2.metadata["count"] == 42

    def test_empty_dataset_json_roundtrip(self):
        ds = PromptDataset(prompts=[], name="empty")
        json_str = ds.to_json()
        ds2 = PromptDataset.from_json(json_str)
        assert len(ds2) == 0

    def test_constraint_type_serialized_as_string(self):
        for ct in ConstraintType:
            c = TaskConstraint(constraint_type=ct, parameters={})
            d = c.to_dict()
            assert isinstance(d["constraint_type"], str)
            assert d["constraint_type"] == ct.name

    def test_domain_serialized_as_string(self):
        for td in TaskDomain:
            cfg = TaskConfig(domain=td)
            d = cfg.to_dict()
            assert isinstance(d["domain"], str)
            assert d["domain"] == td.name

    def test_large_dataset_json_roundtrip(self):
        ds = _make_dataset(200, name="large")
        json_str = ds.to_json()
        ds2 = PromptDataset.from_json(json_str)
        assert len(ds2) == 200

    def test_special_chars_in_text_roundtrip(self):
        p = _make_prompt(text='Text with "quotes" and <brackets> & ampersands')
        d = p.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        p2 = TaskPrompt.from_dict(loaded)
        assert p2.text == p.text

    def test_unicode_text_roundtrip(self):
        p = _make_prompt(text="日本語テスト 🎉 café résumé naïve")
        d = p.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        loaded = json.loads(json_str)
        p2 = TaskPrompt.from_dict(loaded)
        assert p2.text == p.text


# ===================================================================
# 14. TestTaskEdgeCases
# ===================================================================

class TestTaskEdgeCases:
    """Edge cases: empty prompts, long text, special characters."""

    def test_empty_prompt_text(self):
        p = _make_prompt(text="")
        assert p.word_count == 0
        assert p.char_count == 0

    def test_whitespace_only_prompt(self):
        p = _make_prompt(text="   \t\n  ")
        assert p.char_count > 0  # whitespace is counted

    def test_very_long_text_prompt(self):
        long_text = "word " * 10000
        p = _make_prompt(text=long_text.strip())
        assert p.word_count == 10000

    def test_single_word_prompt(self):
        p = _make_prompt(text="Hello")
        assert p.word_count == 1
        assert p.char_count == 5

    def test_special_characters_prompt(self):
        p = _make_prompt(text="!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert p.char_count > 0

    def test_newlines_in_prompt(self):
        p = _make_prompt(text="Line 1\nLine 2\nLine 3")
        assert "\n" in p.text

    def test_unicode_prompt(self):
        p = _make_prompt(text="こんにちは世界 🌍")
        assert p.char_count > 0

    def test_empty_dataset_statistics(self):
        ds = PromptDataset(prompts=[], name="empty")
        stats = ds.statistics()
        assert stats["count"] == 0

    def test_single_prompt_dataset(self):
        ds = _make_dataset(1)
        assert len(ds) == 1
        sampled = ds.sample(1, seed=42)
        assert len(sampled) == 1

    def test_constraint_on_empty_text(self):
        c = _make_length_constraint(1, 100)
        assert c.check("") is False

    def test_keyword_constraint_empty_keywords(self):
        c = _make_keyword_constraint([], mode="all")
        result = c.check("Any text here")
        assert isinstance(result, bool)

    def test_evaluator_empty_text(self):
        ev = TaskEvaluator()
        score = ev.fluency_score("")
        assert score == 0.0

    def test_evaluator_single_char_text(self):
        ev = TaskEvaluator()
        score = ev.fluency_score("a")
        assert 0.0 <= score <= 1.0

    def test_evaluator_repetitive_text(self):
        ev = TaskEvaluator()
        text = "the the the the the the the the the the"
        score = ev.fluency_score(text)
        assert 0.0 <= score <= 1.0

    def test_diversity_single_generation(self):
        ev = TaskEvaluator()
        div = ev.diversity_within_set(["Only one text."])
        assert 0.0 <= div <= 1.0

    def test_prompt_id_empty_text(self):
        p = _make_prompt(text="")
        pid = p._generate_id()
        assert isinstance(pid, str)
        assert len(pid) == 16

    def test_config_very_large_max_length(self):
        cfg = TaskConfig(max_length=1_000_000)
        errors = cfg.validate()
        assert errors == []

    def test_config_min_equals_max(self):
        cfg = TaskConfig(min_length=100, max_length=100)
        errors = cfg.validate()
        assert errors == []

    def test_dataset_filter_returns_empty(self):
        ds = _make_dataset(10)
        filtered = ds.filter_by(lambda p: False)
        assert len(filtered) == 0

    def test_dataset_split_small(self):
        ds = _make_dataset(3)
        train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == 3

    def test_post_process_multiple_blanks(self):
        task = CreativeWritingTask()
        text = "Line one.\n\n\n\nLine two."
        result = task.post_process(text)
        assert "\n\n\n" not in result

    def test_post_process_multiple_spaces(self):
        task = CreativeWritingTask()
        text = "Word   with    many     spaces."
        result = task.post_process(text)
        assert "  " not in result

    def test_constraint_describe_all_types(self):
        for ct in ConstraintType:
            c = TaskConstraint(constraint_type=ct, parameters={})
            desc = c.describe()
            assert isinstance(desc, str)

    def test_dataset_negative_index(self):
        ds = _make_dataset(5)
        p = ds[-1]
        assert isinstance(p, TaskPrompt)

    def test_evaluator_very_long_text(self):
        ev = TaskEvaluator()
        long_text = "This is a sentence. " * 500
        score = ev.fluency_score(long_text)
        assert 0.0 <= score <= 1.0

    def test_mixed_domain_dataset_statistics(self):
        prompts = []
        for i, domain in enumerate([
            TaskDomain.CODE_GENERATION,
            TaskDomain.SUMMARIZATION,
            TaskDomain.QUESTION_ANSWERING,
            TaskDomain.TRANSLATION,
            TaskDomain.OPEN_ENDED_GENERATION,
        ]):
            prompts.append(_make_prompt(f"Prompt {i} for testing", domain=domain))
        ds = PromptDataset(prompts, name="mixed")
        stats = ds.statistics()
        assert len(stats["domain_distribution"]) == 5


# ===================================================================
# Additional parametrized cross-task tests
# ===================================================================

class TestCrossTaskParametrized:
    """Parametrized tests that apply across all task domains."""

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_is_generation_task(self, TaskClass):
        task = TaskClass()
        assert isinstance(task, GenerationTask)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_load_prompts_returns_dataset(self, TaskClass):
        task = TaskClass()
        ds = task.load_prompts()
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_format_prompt_returns_str(self, TaskClass):
        task = TaskClass()
        ds = task.load_prompts()
        formatted = task.format_prompt(ds[0])
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_get_constraints_returns_list(self, TaskClass):
        task = TaskClass()
        constraints = task.get_constraints()
        assert isinstance(constraints, list)
        assert all(isinstance(c, TaskConstraint) for c in constraints)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_describe_returns_str(self, TaskClass):
        task = TaskClass()
        desc = task.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_get_metric_names(self, TaskClass):
        task = TaskClass()
        names = task.get_metric_names()
        assert isinstance(names, list)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_summary(self, TaskClass):
        task = TaskClass()
        s = task.summary()
        assert isinstance(s, dict)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_get_dataset(self, TaskClass):
        task = TaskClass()
        ds = task.get_dataset()
        assert isinstance(ds, PromptDataset)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_format_all_prompts(self, TaskClass):
        task = TaskClass()
        ds = task.load_prompts()
        small_ds = ds[:min(3, len(ds))]
        formatted = task.format_all_prompts(small_ds)
        assert isinstance(formatted, list)
        assert all(isinstance(f, str) for f in formatted)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_repr(self, TaskClass):
        task = TaskClass()
        r = repr(task)
        assert isinstance(r, str)

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_post_process(self, TaskClass):
        task = TaskClass()
        result = task.post_process("  Hello world.  \n\n\n\nMore text.  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    @pytest.mark.parametrize("TaskClass", [
        CreativeWritingTask,
        CodeGenerationTask,
        SummarizationTask,
        QuestionAnsweringTask,
        TranslationTask,
        BrainstormingTask,
    ])
    def test_task_validate_batch(self, TaskClass):
        task = TaskClass()
        ds = task.load_prompts()
        prompts = list(ds[:min(2, len(ds))])
        generations = ["Sample generation text."] * len(prompts)
        result = task.validate_batch(generations, prompts)
        assert isinstance(result, dict)


# ===================================================================
# Integration-style tests using conftest fixtures
# ===================================================================

class TestWithConfTestFixtures:
    """Tests that use fixtures from conftest.py."""

    def test_sample_prompts_are_strings(self, sample_prompts):
        assert isinstance(sample_prompts, list)
        assert all(isinstance(p, str) for p in sample_prompts)
        assert len(sample_prompts) > 0

    def test_creative_writing_prompts_nonempty(self, creative_writing_prompts):
        assert len(creative_writing_prompts) > 0
        assert all(isinstance(p, str) for p in creative_writing_prompts)

    def test_code_generation_prompts_nonempty(self, code_generation_prompts):
        assert len(code_generation_prompts) > 0
        assert all(isinstance(p, str) for p in code_generation_prompts)

    def test_summarization_prompts_nonempty(self, summarization_prompts):
        assert len(summarization_prompts) > 0
        assert all(isinstance(p, str) for p in summarization_prompts)

    def test_qa_prompts_nonempty(self, qa_prompts):
        assert len(qa_prompts) > 0
        assert all(isinstance(p, str) for p in qa_prompts)

    def test_translation_prompts_nonempty(self, translation_prompts):
        assert len(translation_prompts) > 0
        assert all(isinstance(p, str) for p in translation_prompts)

    def test_brainstorming_prompts_nonempty(self, brainstorming_prompts):
        assert len(brainstorming_prompts) > 0
        assert all(isinstance(p, str) for p in brainstorming_prompts)

    def test_evaluate_creative_prompts(self, creative_writing_prompts):
        ev = TaskEvaluator()
        for prompt_text in creative_writing_prompts[:3]:
            prompt = _make_prompt(prompt_text)
            gen = "A creative response that addresses the prompt with style and flair."
            scores = ev.evaluate_single(gen, prompt)
            assert "fluency" in scores
            assert all(0.0 <= v <= 2.0 for v in scores.values())

    def test_evaluate_code_prompts(self, code_generation_prompts):
        ev = TaskEvaluator()
        for prompt_text in code_generation_prompts[:3]:
            prompt = _make_prompt(prompt_text, domain=TaskDomain.CODE_GENERATION)
            gen = "def solution(arr):\n    return sorted(arr)\n"
            scores = ev.evaluate_single(gen, prompt)
            assert "fluency" in scores

    def test_create_dataset_from_fixture_prompts(self, sample_prompts):
        prompts = [_make_prompt(text=t) for t in sample_prompts]
        ds = PromptDataset(prompts=prompts, name="fixture-ds")
        assert len(ds) == len(sample_prompts)
        stats = ds.statistics()
        assert stats["count"] == len(sample_prompts)

    def test_dataset_filter_fixture_prompts(self, sample_prompts):
        prompts = [_make_prompt(text=t) for t in sample_prompts]
        ds = PromptDataset(prompts=prompts, name="filter-test")
        long_prompts = ds.filter_by_length(min_words=5)
        assert len(long_prompts) <= len(ds)

    def test_constraints_on_fixture_prompts(self, creative_writing_prompts):
        c = _make_length_constraint(3, 1000)
        for prompt_text in creative_writing_prompts:
            result = c.check(prompt_text)
            assert isinstance(result, bool)

    def test_diversity_of_fixture_generations(self, sample_prompts):
        ev = TaskEvaluator()
        generations = [
            f"Response to: {p}. This is a unique answer." for p in sample_prompts
        ]
        if len(generations) >= 2:
            div = ev.diversity_within_set(generations)
            assert 0.0 <= div <= 1.0


# ===================================================================
# Additional depth tests for better coverage
# ===================================================================

class TestTaskConfigAdvanced:
    """Additional TaskConfig tests for depth."""

    def test_evaluation_metrics_custom(self):
        cfg = TaskConfig(evaluation_metrics=["bleu", "rouge", "meteor"])
        assert cfg.evaluation_metrics == ["bleu", "rouge", "meteor"]

    def test_prompt_template_custom(self):
        cfg = TaskConfig(prompt_template="[INST] {text} [/INST]")
        assert "[INST]" in cfg.prompt_template

    def test_from_dict_missing_optional_fields(self):
        d = {"domain": "OPEN_ENDED_GENERATION"}
        cfg = TaskConfig.from_dict(d)
        assert cfg.name == "default"
        assert cfg.num_prompts == 100

    def test_from_dict_all_domains(self):
        for domain in TaskDomain:
            d = {"domain": domain.name}
            cfg = TaskConfig.from_dict(d)
            assert cfg.domain == domain

    def test_multiple_constraints_validate(self):
        constraints = [
            _make_length_constraint(10, 100),
            _make_keyword_constraint(["test"]),
            _make_content_constraint(banned_words=["bad"]),
        ]
        cfg = TaskConfig(constraints=constraints)
        errors = cfg.validate()
        assert errors == []

    def test_config_copy_independence(self):
        cfg1 = TaskConfig(name="original", num_prompts=50)
        d = cfg1.to_dict()
        cfg2 = TaskConfig.from_dict(d)
        cfg2.name = "modified"
        assert cfg1.name == "original"

    def test_seed_reproducibility(self):
        cfg1 = TaskConfig(seed=42)
        cfg2 = TaskConfig(seed=42)
        assert cfg1.seed == cfg2.seed

    def test_temperature_float_precision(self):
        cfg = TaskConfig(temperature=0.123456789)
        d = cfg.to_dict()
        cfg2 = TaskConfig.from_dict(d)
        assert abs(cfg2.temperature - 0.123456789) < 1e-9


class TestPromptDatasetAdvanced:
    """Additional PromptDataset tests."""

    def test_sample_without_replacement(self):
        ds = _make_dataset(20)
        sampled = ds.sample(10, seed=42)
        texts = [p.text for p in sampled]
        assert len(texts) == len(set(texts))

    def test_sample_all(self):
        ds = _make_dataset(5)
        sampled = ds.sample(5, seed=42)
        assert len(sampled) == 5

    def test_filter_by_preserves_type(self):
        ds = _make_dataset(10)
        filtered = ds.filter_by(lambda p: True)
        assert isinstance(filtered, PromptDataset)

    def test_filter_by_length_no_match(self):
        ds = _make_dataset(10)
        filtered = ds.filter_by_length(min_words=10000)
        assert len(filtered) == 0

    def test_getitem_out_of_range(self):
        ds = _make_dataset(5)
        with pytest.raises(IndexError):
            _ = ds[100]

    def test_split_reproducible(self):
        ds = _make_dataset(50)
        t1, v1, te1 = ds.split(0.8, 0.1, 0.1, seed=42)
        t2, v2, te2 = ds.split(0.8, 0.1, 0.1, seed=42)
        assert [p.text for p in t1] == [p.text for p in t2]

    def test_split_no_overlap(self):
        ds = _make_dataset(50)
        train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
        train_texts = {p.text for p in train}
        val_texts = {p.text for p in val}
        test_texts = {p.text for p in test}
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_add_preserves_order(self):
        ds1 = _make_dataset(3, name="first")
        ds2 = _make_dataset(2, name="second")
        combined = ds1 + ds2
        for i in range(3):
            assert combined[i].text == ds1[i].text
        for i in range(2):
            assert combined[3 + i].text == ds2[i].text

    def test_statistics_word_length_stats(self):
        ds = _make_dataset(10)
        stats = ds.statistics()
        assert stats["length_min"] <= stats["length_mean"] <= stats["length_max"]
        assert stats["length_p25"] <= stats["length_median"] <= stats["length_p75"]

    def test_statistics_vocab_richness(self):
        ds = _make_dataset(10)
        stats = ds.statistics()
        assert 0.0 <= stats["vocab_richness"] <= 1.0

    def test_statistics_reference_fraction(self):
        prompts = [
            _make_prompt(f"P{i}", reference_outputs=["ref"] if i % 2 == 0 else [])
            for i in range(10)
        ]
        ds = PromptDataset(prompts=prompts, name="mixed-ref")
        stats = ds.statistics()
        assert stats["has_reference_frac"] == 0.5


class TestTaskConstraintAdvanced:
    """Additional constraint validation tests."""

    def test_length_constraint_characters(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.LENGTH,
            parameters={"min": 5, "max": 20, "unit": "characters"},
        )
        assert c.check("Hello World") is True  # 11 chars

    def test_format_constraint_starts_with(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.FORMAT,
            parameters={"starts_with": "Dear"},
        )
        assert c.check("Dear Sir, I write to you...") is True
        assert c.check("Hello there") is False

    def test_format_constraint_ends_with(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.FORMAT,
            parameters={"ends_with": "."},
        )
        assert c.check("This is a sentence.") is True
        assert c.check("No period here") is False

    def test_content_required_topics(self):
        c = _make_content_constraint(required_topics=["science", "technology"])
        text = "Advances in science and technology have changed our world."
        assert c.check(text) is True

    def test_content_max_repetition_ratio(self):
        c = _make_content_constraint(max_repetition_ratio=0.3)
        text = "unique words in this text make it diverse and varied"
        result = c.check(text)
        assert isinstance(result, bool)

    def test_content_min_unique_words(self):
        c = _make_content_constraint(min_unique_words=5)
        text = "cat dog bird fish elephant"
        assert c.check(text) is True

    def test_style_prohibited_patterns(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.STYLE,
            parameters={"prohibited_patterns": [r"\b(very|really)\b"]},
        )
        clean = "The sunset was beautiful."
        assert c.check(clean) is True

    def test_constraint_weight_zero(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.LENGTH,
            parameters={"min": 1, "max": 5},
            weight=0.0,
        )
        assert c.weight == 0.0

    def test_constraint_optional(self):
        c = TaskConstraint(
            constraint_type=ConstraintType.LENGTH,
            parameters={"min": 100, "max": 200},
            required=False,
        )
        assert c.required is False


class TestTaskEvaluatorAdvanced:
    """Additional evaluator edge cases."""

    def test_self_bleu_identical(self):
        ev = TaskEvaluator()
        generations = ["The exact same sentence."] * 5
        sb = ev.self_bleu(generations, n=4)
        assert sb > 0.5  # identical should have high self-BLEU

    def test_self_bleu_diverse(self):
        ev = TaskEvaluator()
        generations = [
            "Apples grow on trees in orchards.",
            "Quantum mechanics describes atomic behavior.",
            "Basketball requires agility and teamwork.",
            "Cooking pasta requires boiling water.",
        ]
        sb = ev.self_bleu(generations, n=4)
        assert sb < 1.0

    def test_ngram_diversity_unigrams(self):
        ev = TaskEvaluator()
        generations = [
            "Alpha beta gamma delta epsilon.",
            "Zeta eta theta iota kappa.",
        ]
        score = ev.ngram_diversity(generations, n=1)
        assert 0.0 <= score <= 1.0

    def test_evaluate_set_with_references(self):
        ev = TaskEvaluator()
        prompts = [_make_prompt("Question 1"), _make_prompt("Question 2")]
        generations = [
            "Answer to question one with details.",
            "Answer to question two with info.",
        ]
        references = ["Reference answer one.", "Reference answer two."]
        scores = ev.evaluate_set(generations, prompts, references)
        assert isinstance(scores, dict)

    def test_check_constraints_multiple(self):
        ev = TaskEvaluator()
        constraints = [
            _make_length_constraint(3, 100),
            _make_keyword_constraint(["world"], mode="all"),
        ]
        text = "Hello world, this is a test."
        results = ev.check_constraints(text, constraints)
        assert len(results) == 2
        assert all(r is True for r in results)

    def test_weighted_constraint_all_pass(self):
        ev = TaskEvaluator()
        constraints = [
            _make_length_constraint(1, 100),
        ]
        score = ev.weighted_constraint_score("Hello there.", constraints)
        assert score == 1.0

    def test_weighted_constraint_all_fail(self):
        ev = TaskEvaluator()
        constraints = [
            _make_length_constraint(1000, 2000),
        ]
        score = ev.weighted_constraint_score("Short.", constraints)
        assert score == 0.0

    def test_length_ratio_computation(self):
        ev = TaskEvaluator()
        prompt = _make_prompt("Short prompt")
        ratio = ev._length_ratio("A much longer generation text here.", prompt)
        assert isinstance(ratio, float)
        assert ratio > 0

    def test_reference_overlap_high(self):
        ev = TaskEvaluator()
        gen = "The cat sat on the mat."
        ref = "The cat sat on the mat."
        overlap = ev._reference_overlap(gen, ref)
        assert overlap > 0.9

    def test_reference_overlap_low(self):
        ev = TaskEvaluator()
        gen = "Completely different text here."
        ref = "Nothing in common at all."
        overlap = ev._reference_overlap(gen, ref)
        assert overlap < 0.5

    def test_bleu_ngram_identical(self):
        ev = TaskEvaluator()
        text = "The quick brown fox."
        score = ev._bleu_ngram(text, text, n=1)
        assert score > 0.9

    def test_bleu_ngram_different(self):
        ev = TaskEvaluator()
        gen = "Alpha beta gamma delta."
        ref = "One two three four."
        score = ev._bleu_ngram(gen, ref, n=1)
        assert score < 0.5


class TestCreativeWritingAdvanced:
    """Additional creative writing tests."""

    def test_config_pov_options(self):
        for pov in ["first", "second", "third"]:
            cfg = CreativeWritingConfig(pov=pov)
            assert cfg.pov == pov

    def test_config_tense_options(self):
        for tense in ["past", "present", "future"]:
            cfg = CreativeWritingConfig(tense=tense)
            assert cfg.tense == tense

    def test_config_required_elements(self):
        cfg = CreativeWritingConfig(required_elements=["dialogue", "conflict"])
        assert "dialogue" in cfg.required_elements

    def test_config_tone(self):
        cfg = CreativeWritingConfig(tone="melancholic")
        assert cfg.tone == "melancholic"

    def test_story_prompt_conflict_type(self):
        sp = StoryPrompt(
            text="Write a story",
            conflict_type="person vs nature",
        )
        assert sp.conflict_type == "person vs nature"

    def test_dialogue_prompt_relationship(self):
        dp = DialoguePrompt(
            text="Dialogue",
            relationship="siblings",
            topic="inheritance",
        )
        assert dp.relationship == "siblings"

    def test_poetry_prompt_meter(self):
        pp = PoetryPrompt(
            text="Write a poem",
            meter="iambic pentameter",
        )
        assert pp.meter == "iambic pentameter"

    def test_evaluate_multiple_generations(self):
        task = CreativeWritingTask()
        ds = task.load_prompts()
        n = min(3, len(ds))
        prompts = list(ds[:n])
        generations = [
            "The wind howled through the abandoned town. Dust swirled in empty streets. "
            "A lone figure emerged from the shadows, clutching a worn photograph.",
            "Moonlight danced on the surface of the still lake. Crickets sang their evening "
            "chorus as the old man cast his fishing line into the water.",
            "She opened the letter with trembling hands. The words blurred before her eyes "
            "as tears began to fall. Nothing would ever be the same again.",
        ][:n]
        result = task.evaluate(generations, prompts)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestCodeGenerationAdvanced:
    """Additional code generation tests."""

    def test_config_allow_imports(self):
        cfg = CodeGenerationConfig(allow_imports=False)
        assert cfg.allow_imports is False

    def test_config_max_functions(self):
        cfg = CodeGenerationConfig(max_functions=5)
        assert cfg.max_functions == 5

    def test_config_require_type_hints(self):
        cfg = CodeGenerationConfig(require_type_hints=True)
        assert cfg.require_type_hints is True

    def test_config_test_framework(self):
        cfg = CodeGenerationConfig(test_framework="unittest")
        assert cfg.test_framework == "unittest"

    def test_code_prompt_input_output_types(self):
        cp = CodePrompt(
            text="Sort an array",
            input_types=["List[int]"],
            output_type="List[int]",
        )
        assert cp.input_types == ["List[int]"]
        assert cp.output_type == "List[int]"

    def test_code_test_case_edge(self):
        tc = CodeTestCase(
            input_args=[[]],
            expected_output=[],
            description="Empty input",
            is_edge_case=True,
        )
        assert tc.is_edge_case is True
        assert tc.expected_output == []

    @pytest.mark.parametrize("task_type", list(CodeTaskType))
    def test_all_task_types_valid(self, task_type):
        cfg = CodeGenerationConfig(task_type=task_type)
        assert cfg.task_type == task_type

    def test_evaluate_python_code(self):
        task = CodeGenerationTask()
        ds = task.load_prompts()
        code = textwrap.dedent("""\
            def add(a, b):
                \"\"\"Add two numbers.\"\"\"
                return a + b
        """)
        result = task.evaluate([code], [ds[0]])
        assert isinstance(result, dict)


class TestSummarizationAdvanced:
    """Additional summarization tests."""

    def test_config_focus_aspects(self):
        cfg = SummarizationConfig(focus_aspects=["key findings", "methodology"])
        assert len(cfg.focus_aspects) == 2

    def test_config_audience_level(self):
        cfg = SummarizationConfig(audience_level="expert")
        assert cfg.audience_level == "expert"

    def test_summarization_prompt_key_points(self):
        sp = SummarizationPrompt(
            text="Summarize",
            key_points=["point1", "point2", "point3"],
        )
        assert len(sp.key_points) == 3

    def test_summarization_prompt_source_document_length(self):
        long_doc = "This is a sentence about an important topic. " * 100
        sp = SummarizationPrompt(
            text="Summarize the following",
            source_document=long_doc,
        )
        assert len(sp.source_document) > 1000

    def test_evaluate_short_summary(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        summary = "Key point one. Key point two. Conclusion."
        result = task.evaluate([summary], [ds[0]])
        assert isinstance(result, dict)

    def test_evaluate_longer_summary(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        summary = (
            "The article discusses several important findings. First, the researchers "
            "discovered a novel approach to the problem. Second, they validated their "
            "results through extensive experimentation. Finally, they outlined future "
            "directions for research in the field."
        )
        result = task.evaluate([summary], [ds[0]])
        assert isinstance(result, dict)


class TestTranslationAdvanced:
    """Additional translation tests."""

    def test_language_pair_name_auto(self):
        lp = LanguagePair(source_lang="en", target_lang="ja")
        assert lp.source_lang == "en"
        assert lp.target_lang == "ja"

    def test_config_domain_specific_terms(self):
        cfg = TranslationConfig(
            domain_specific_terms={"neural network": "réseau neuronal"}
        )
        assert "neural network" in cfg.domain_specific_terms

    def test_translation_prompt_domain_field(self):
        tp = TranslationPrompt(
            text="Translate",
            source_text="The patient presented with symptoms.",
            domain="medical",
        )
        assert tp.domain == "medical"

    def test_translation_prompt_multiple_references(self):
        tp = TranslationPrompt(
            text="Translate",
            source_text="Hello",
            reference_translations=[
                "Bonjour",
                "Salut",
                "Coucou",
            ],
        )
        assert len(tp.reference_translations) == 3

    def test_evaluate_multiple_translations(self):
        task = TranslationTask()
        ds = task.load_prompts()
        n = min(2, len(ds))
        prompts = list(ds[:n])
        translations = [
            "Ceci est une traduction.",
            "Voici une autre traduction possible.",
        ][:n]
        result = task.evaluate(translations, prompts)
        assert isinstance(result, dict)

    def test_config_preserve_register(self):
        cfg = TranslationConfig(preserve_register=False)
        assert cfg.preserve_register is False


class TestQAAdvanced:
    """Additional QA tests."""

    def test_config_allow_uncertainty(self):
        cfg = QAConfig(allow_uncertainty=False)
        assert cfg.allow_uncertainty is False

    def test_config_perspective_count(self):
        cfg = QAConfig(perspective_count=3)
        assert cfg.perspective_count == 3

    def test_qa_prompt_context_passage(self):
        qp = QAPrompt(
            text="Based on the passage, explain photosynthesis.",
            question="What is photosynthesis?",
            context_passage=(
                "Photosynthesis is a process used by plants to convert light energy "
                "into chemical energy that can be stored and later released."
            ),
        )
        assert qp.context_passage is not None
        assert "Photosynthesis" in qp.context_passage

    def test_qa_prompt_no_context(self):
        qp = QAPrompt(
            text="What is gravity?",
            question="What is gravity?",
        )
        assert qp.context_passage is None

    def test_evaluate_multiple_answers(self):
        task = QuestionAnsweringTask()
        ds = task.load_prompts()
        n = min(3, len(ds))
        prompts = list(ds[:n])
        answers = [
            "Gravity is the force of attraction between objects with mass.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The Earth revolves around the Sun once every 365.25 days.",
        ][:n]
        result = task.evaluate(answers, prompts)
        assert isinstance(result, dict)

    def test_qa_prompt_reference_answers(self):
        qp = QAPrompt(
            text="Question",
            question="What color is the sky?",
            reference_answers=["Blue", "The sky appears blue due to Rayleigh scattering."],
        )
        assert len(qp.reference_answers) == 2


class TestBrainstormingAdvanced:
    """Additional brainstorming tests."""

    def test_config_scoring_weights(self):
        cfg = BrainstormingConfig(
            novelty_weight=0.3,
            specificity_weight=0.2,
            diversity_weight=0.2,
        )
        assert cfg.novelty_weight == 0.3
        assert cfg.specificity_weight == 0.2

    def test_config_topic_constraint(self):
        cfg = BrainstormingConfig(topic_constraint="sustainable energy")
        assert cfg.topic_constraint == "sustainable energy"

    def test_config_feasibility_weight(self):
        cfg = BrainstormingConfig(feasibility_weight=0.5)
        assert cfg.feasibility_weight == 0.5

    def test_brainstorm_prompt_topic(self):
        bp = BrainstormPrompt(
            text="Generate ideas",
            topic="urban farming",
        )
        assert bp.topic == "urban farming"

    def test_evaluate_structured_ideas(self):
        task = BrainstormingTask()
        ds = task.load_prompts()
        ideas = (
            "1. Rooftop gardens for apartment buildings\n"
            "2. Community composting stations\n"
            "3. Vertical hydroponic farms\n"
            "4. Solar-powered irrigation systems\n"
            "5. Insect protein farming kits\n"
            "6. Aquaponics starter systems\n"
            "7. Seed library exchange programs\n"
        )
        result = task.evaluate([ideas], [ds[0]])
        assert isinstance(result, dict)

    def test_brainstorm_prompt_num_ideas(self):
        bp = BrainstormPrompt(text="Ideas", num_ideas_requested=15)
        assert bp.num_ideas_requested == 15


class TestGenerationTaskBase:
    """Tests for GenerationTask base class behavior."""

    def test_get_default_config(self):
        cfg = CreativeWritingTask.get_default_config()
        assert isinstance(cfg, TaskConfig)

    def test_describe_nonempty(self):
        task = CreativeWritingTask()
        desc = task.describe()
        assert isinstance(desc, str)
        assert len(desc) > 5

    def test_validate_generation_returns_tuple(self):
        task = CodeGenerationTask()
        ds = task.load_prompts()
        valid, errors = task.validate_generation("x = 1", ds[0])
        assert isinstance(valid, bool)
        assert isinstance(errors, list)

    def test_evaluate_with_references(self):
        task = SummarizationTask()
        ds = task.load_prompts()
        prompts = [ds[0]]
        gens = ["This is a summary of the main points."]
        result = task.evaluate_with_references(gens, prompts)
        assert isinstance(result, dict)

    def test_validate_batch_stats(self):
        task = QuestionAnsweringTask()
        ds = task.load_prompts()
        n = min(3, len(ds))
        prompts = list(ds[:n])
        gens = ["An answer to the question."] * n
        batch_result = task.validate_batch(gens, prompts)
        assert isinstance(batch_result, dict)

    def test_lazy_dataset_caching(self):
        task = TranslationTask()
        ds1 = task.get_dataset()
        ds2 = task.get_dataset()
        assert ds1 is ds2  # same object, cached

    def test_format_all_prompts_length(self):
        task = BrainstormingTask()
        ds = task.load_prompts()
        small = ds[:min(5, len(ds))]
        formatted = task.format_all_prompts(small)
        assert len(formatted) == len(small)

    def test_summary_has_required_keys(self):
        task = CreativeWritingTask()
        s = task.summary()
        assert isinstance(s, dict)
