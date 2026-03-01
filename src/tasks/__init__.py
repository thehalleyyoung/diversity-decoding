"""
Task domain definitions for the Diversity Decoding Arena.

Provides base classes and concrete task implementations for evaluating
diversity-promoting decoding algorithms across different generation tasks.
"""

from __future__ import annotations

from src.tasks.base import (
    TaskDomain,
    TaskConfig,
    TaskPrompt,
    TaskConstraint,
    TaskEvaluator,
    PromptDataset,
    GenerationTask,
)
from src.tasks.creative_writing import CreativeWritingTask
from src.tasks.code_generation import CodeGenerationTask
from src.tasks.brainstorming import BrainstormingTask
from src.tasks.translation import TranslationTask
from src.tasks.summarization import SummarizationTask
from src.tasks.qa import QuestionAnsweringTask

__all__ = [
    "TaskDomain",
    "TaskConfig",
    "TaskPrompt",
    "TaskConstraint",
    "TaskEvaluator",
    "PromptDataset",
    "GenerationTask",
    "CreativeWritingTask",
    "CodeGenerationTask",
    "BrainstormingTask",
    "TranslationTask",
    "SummarizationTask",
    "QuestionAnsweringTask",
]
