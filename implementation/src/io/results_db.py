"""
Results database module for the Diversity Decoding Arena.

Provides SQLite-backed storage for experiments, runs, metrics, and generations
with query building, aggregation, indexing, and migration support.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import sqlite3
import statistics
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version history
# ---------------------------------------------------------------------------
CURRENT_SCHEMA_VERSION = 3

_MIGRATIONS: Dict[int, List[str]] = {
    1: [
        # v0 -> v1: initial schema (handled by create_tables)
    ],
    2: [
        "ALTER TABLE runs ADD COLUMN error_message TEXT DEFAULT NULL;",
        "ALTER TABLE metrics ADD COLUMN percentile_rank REAL DEFAULT NULL;",
    ],
    3: [
        "ALTER TABLE experiments ADD COLUMN parent_experiment_id TEXT DEFAULT NULL;",
        "ALTER TABLE generations ADD COLUMN scores TEXT DEFAULT '[]';",
    ],
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(str, Enum):
    DIVERSITY = "diversity"
    QUALITY = "quality"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass
class ExperimentRecord:
    """Represents a single experiment containing multiple runs."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = ExperimentStatus.PENDING.value
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config,
            "status": self.status,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=data.get("experiment_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            updated_at=data.get(
                "updated_at", datetime.now(timezone.utc).isoformat()
            ),
            config=data.get("config", {}),
            status=data.get("status", ExperimentStatus.PENDING.value),
            tags=data.get("tags", []),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.experiment_id:
            errors.append("experiment_id is required")
        if not self.name:
            errors.append("name is required")
        if self.status not in {s.value for s in ExperimentStatus}:
            errors.append(f"Invalid status: {self.status}")
        return errors

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)
            self.touch()

    def remove_tag(self, tag: str) -> None:
        if tag in self.tags:
            self.tags.remove(tag)
            self.touch()

    def merge_config(self, extra: Dict[str, Any]) -> None:
        self.config.update(extra)
        self.touch()


@dataclass
class RunRecord:
    """Represents a single algorithm run within an experiment."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    algorithm_name: str = ""
    task_domain: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = RunStatus.PENDING.value
    elapsed_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "algorithm_name": self.algorithm_name,
            "task_domain": self.task_domain,
            "config": self.config,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "status": self.status,
            "elapsed_time": self.elapsed_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            experiment_id=data.get("experiment_id", ""),
            algorithm_name=data.get("algorithm_name", ""),
            task_domain=data.get("task_domain", ""),
            config=data.get("config", {}),
            seed=data.get("seed", 0),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            status=data.get("status", RunStatus.PENDING.value),
            elapsed_time=data.get("elapsed_time", 0.0),
            metadata=data.get("metadata", {}),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.run_id:
            errors.append("run_id is required")
        if not self.experiment_id:
            errors.append("experiment_id is required")
        if not self.algorithm_name:
            errors.append("algorithm_name is required")
        if self.status not in {s.value for s in RunStatus}:
            errors.append(f"Invalid status: {self.status}")
        if self.elapsed_time < 0:
            errors.append("elapsed_time must be non-negative")
        return errors

    def mark_running(self) -> None:
        self.status = RunStatus.RUNNING.value
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, elapsed: float) -> None:
        self.status = RunStatus.COMPLETED.value
        self.elapsed_time = elapsed

    def mark_failed(self, error_msg: str = "") -> None:
        self.status = RunStatus.FAILED.value
        if error_msg:
            self.metadata["error"] = error_msg


@dataclass
class MetricRecord:
    """Stores a single metric measurement for a run."""

    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    metric_name: str = ""
    metric_type: str = MetricType.DIVERSITY.value
    value: float = 0.0
    values: List[float] = field(default_factory=list)
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "value": self.value,
            "values": self.values,
            "computed_at": self.computed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricRecord":
        return cls(
            metric_id=data.get("metric_id", str(uuid.uuid4())),
            run_id=data.get("run_id", ""),
            metric_name=data.get("metric_name", ""),
            metric_type=data.get("metric_type", MetricType.DIVERSITY.value),
            value=float(data.get("value", 0.0)),
            values=data.get("values", []),
            computed_at=data.get(
                "computed_at", datetime.now(timezone.utc).isoformat()
            ),
            metadata=data.get("metadata", {}),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.metric_id:
            errors.append("metric_id is required")
        if not self.run_id:
            errors.append("run_id is required")
        if not self.metric_name:
            errors.append("metric_name is required")
        if self.metric_type not in {t.value for t in MetricType}:
            errors.append(f"Invalid metric_type: {self.metric_type}")
        if not isinstance(self.value, (int, float)):
            errors.append("value must be numeric")
        return errors

    @property
    def mean_per_prompt(self) -> float:
        if not self.values:
            return self.value
        return float(np.mean(self.values))

    @property
    def std_per_prompt(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))

    @property
    def min_per_prompt(self) -> float:
        if not self.values:
            return self.value
        return float(np.min(self.values))

    @property
    def max_per_prompt(self) -> float:
        if not self.values:
            return self.value
        return float(np.max(self.values))


@dataclass
class GenerationRecord:
    """Stores generated text outputs for a single prompt within a run."""

    generation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    prompt_text: str = ""
    prompt_id: str = ""
    generated_texts: List[str] = field(default_factory=list)
    token_ids: List[List[int]] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "run_id": self.run_id,
            "prompt_text": self.prompt_text,
            "prompt_id": self.prompt_id,
            "generated_texts": self.generated_texts,
            "token_ids": self.token_ids,
            "log_probs": self.log_probs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationRecord":
        return cls(
            generation_id=data.get("generation_id", str(uuid.uuid4())),
            run_id=data.get("run_id", ""),
            prompt_text=data.get("prompt_text", ""),
            prompt_id=data.get("prompt_id", ""),
            generated_texts=data.get("generated_texts", []),
            token_ids=data.get("token_ids", []),
            log_probs=data.get("log_probs", []),
            metadata=data.get("metadata", {}),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.generation_id:
            errors.append("generation_id is required")
        if not self.run_id:
            errors.append("run_id is required")
        if not self.prompt_text:
            errors.append("prompt_text is required")
        return errors

    @property
    def num_generations(self) -> int:
        return len(self.generated_texts)

    @property
    def mean_log_prob(self) -> float:
        if not self.log_probs:
            return 0.0
        return float(np.mean(self.log_probs))

    @property
    def total_tokens(self) -> int:
        return sum(len(ids) for ids in self.token_ids)

    def get_text(self, index: int) -> str:
        if 0 <= index < len(self.generated_texts):
            return self.generated_texts[index]
        raise IndexError(f"Generation index {index} out of range")


# ---------------------------------------------------------------------------
# QueryBuilder
# ---------------------------------------------------------------------------
class QueryBuilder:
    """Fluent SQL query builder with parameterised queries."""

    _VALID_OPS: Set[str] = {
        "=", "!=", "<", ">", "<=", ">=",
        "IN", "LIKE", "BETWEEN", "IS NULL", "IS NOT NULL",
        "NOT IN", "NOT LIKE",
    }

    def __init__(self) -> None:
        self._table: str = ""
        self._select_cols: List[str] = ["*"]
        self._where_clauses: List[Tuple[str, str, Any]] = []
        self._order_clauses: List[Tuple[str, str]] = []
        self._limit_val: Optional[int] = None
        self._offset_val: Optional[int] = None
        self._joins: List[Tuple[str, str]] = []
        self._group_by_cols: List[str] = []
        self._having_clauses: List[Tuple[str, str, Any]] = []
        self._distinct: bool = False
        self._count_mode: bool = False
        self._raw_where: List[Tuple[str, list]] = []

    # -- builder methods ---------------------------------------------------

    def select(self, table: str, columns: Optional[List[str]] = None) -> "QueryBuilder":
        self._table = table
        if columns:
            self._select_cols = list(columns)
        return self

    def distinct(self) -> "QueryBuilder":
        self._distinct = True
        return self

    def count(self) -> "QueryBuilder":
        self._count_mode = True
        return self

    def where(self, field_name: str, op: str, value: Any = None) -> "QueryBuilder":
        op_upper = op.upper()
        if op_upper not in self._VALID_OPS:
            raise ValueError(f"Unsupported operator: {op}")
        self._where_clauses.append((field_name, op_upper, value))
        return self

    def where_raw(self, clause: str, params: Optional[list] = None) -> "QueryBuilder":
        self._raw_where.append((clause, params or []))
        return self

    def order_by(self, field_name: str, direction: str = "ASC") -> "QueryBuilder":
        direction = direction.upper()
        if direction not in ("ASC", "DESC"):
            raise ValueError(f"Invalid direction: {direction}")
        self._order_clauses.append((field_name, direction))
        return self

    def limit(self, n: int) -> "QueryBuilder":
        if n < 0:
            raise ValueError("limit must be non-negative")
        self._limit_val = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        if n < 0:
            raise ValueError("offset must be non-negative")
        self._offset_val = n
        return self

    def join(self, table: str, on: str) -> "QueryBuilder":
        self._joins.append((table, on))
        return self

    def left_join(self, table: str, on: str) -> "QueryBuilder":
        self._joins.append((f"LEFT JOIN {table}", on))
        return self

    def group_by(self, *fields: str) -> "QueryBuilder":
        self._group_by_cols.extend(fields)
        return self

    def having(self, field_name: str, op: str, value: Any) -> "QueryBuilder":
        op_upper = op.upper()
        if op_upper not in self._VALID_OPS:
            raise ValueError(f"Unsupported operator in HAVING: {op}")
        self._having_clauses.append((field_name, op_upper, value))
        return self

    # -- build -------------------------------------------------------------

    def build(self) -> Tuple[str, List[Any]]:
        if not self._table:
            raise ValueError("No table specified. Call .select(table) first.")

        params: List[Any] = []

        # SELECT clause
        if self._count_mode:
            select_part = "SELECT COUNT(*)"
        elif self._distinct:
            select_part = f"SELECT DISTINCT {', '.join(self._select_cols)}"
        else:
            select_part = f"SELECT {', '.join(self._select_cols)}"

        sql = f"{select_part} FROM {self._table}"

        # JOINs
        for join_table, join_on in self._joins:
            if join_table.startswith("LEFT JOIN"):
                sql += f" {join_table} ON {join_on}"
            else:
                sql += f" JOIN {join_table} ON {join_on}"

        # WHERE
        where_parts: List[str] = []
        for field_name, op, value in self._where_clauses:
            clause, clause_params = self._build_where_clause(
                field_name, op, value
            )
            where_parts.append(clause)
            params.extend(clause_params)

        for raw_clause, raw_params in self._raw_where:
            where_parts.append(raw_clause)
            params.extend(raw_params)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        # GROUP BY
        if self._group_by_cols:
            sql += f" GROUP BY {', '.join(self._group_by_cols)}"

        # HAVING
        if self._having_clauses:
            having_parts: List[str] = []
            for field_name, op, value in self._having_clauses:
                clause, clause_params = self._build_where_clause(
                    field_name, op, value
                )
                having_parts.append(clause)
                params.extend(clause_params)
            sql += " HAVING " + " AND ".join(having_parts)

        # ORDER BY
        if self._order_clauses:
            order_strs = [f"{f} {d}" for f, d in self._order_clauses]
            sql += f" ORDER BY {', '.join(order_strs)}"

        # LIMIT / OFFSET
        if self._limit_val is not None:
            sql += " LIMIT ?"
            params.append(self._limit_val)

        if self._offset_val is not None:
            sql += " OFFSET ?"
            params.append(self._offset_val)

        return sql, params

    def execute(self, connection: sqlite3.Connection) -> List[Dict[str, Any]]:
        sql, params = self.build()
        cursor = connection.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def execute_scalar(self, connection: sqlite3.Connection) -> Any:
        sql, params = self.build()
        cursor = connection.execute(sql, params)
        row = cursor.fetchone()
        if row is None:
            return None
        return row[0]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_where_clause(
        field_name: str, op: str, value: Any
    ) -> Tuple[str, List[Any]]:
        if op == "IS NULL":
            return f"{field_name} IS NULL", []
        if op == "IS NOT NULL":
            return f"{field_name} IS NOT NULL", []
        if op == "IN":
            if not isinstance(value, (list, tuple, set)):
                raise ValueError("IN operator requires a list/tuple/set value")
            placeholders = ", ".join("?" for _ in value)
            return f"{field_name} IN ({placeholders})", list(value)
        if op == "NOT IN":
            if not isinstance(value, (list, tuple, set)):
                raise ValueError("NOT IN operator requires a list/tuple/set value")
            placeholders = ", ".join("?" for _ in value)
            return f"{field_name} NOT IN ({placeholders})", list(value)
        if op == "BETWEEN":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("BETWEEN requires a 2-element list [lo, hi]")
            return f"{field_name} BETWEEN ? AND ?", [value[0], value[1]]
        if op in ("LIKE", "NOT LIKE"):
            return f"{field_name} {op} ?", [value]
        # Standard comparison operators
        return f"{field_name} {op} ?", [value]


# ---------------------------------------------------------------------------
# DatabaseIndex
# ---------------------------------------------------------------------------
class DatabaseIndex:
    """Creates and manages SQLite indices for query performance."""

    _INDEX_DEFINITIONS: List[Tuple[str, str, List[str]]] = [
        ("idx_runs_experiment_id", "runs", ["experiment_id"]),
        ("idx_runs_algorithm_name", "runs", ["algorithm_name"]),
        ("idx_runs_status", "runs", ["status"]),
        ("idx_runs_task_domain", "runs", ["task_domain"]),
        ("idx_runs_experiment_algorithm", "runs", ["experiment_id", "algorithm_name"]),
        ("idx_runs_experiment_status", "runs", ["experiment_id", "status"]),
        ("idx_metrics_run_id", "metrics", ["run_id"]),
        ("idx_metrics_metric_name", "metrics", ["metric_name"]),
        ("idx_metrics_metric_type", "metrics", ["metric_type"]),
        ("idx_metrics_run_name", "metrics", ["run_id", "metric_name"]),
        ("idx_metrics_run_type", "metrics", ["run_id", "metric_type"]),
        ("idx_generations_run_id", "generations", ["run_id"]),
        ("idx_generations_prompt_id", "generations", ["prompt_id"]),
        ("idx_generations_run_prompt", "generations", ["run_id", "prompt_id"]),
        ("idx_experiment_tags_experiment_id", "experiment_tags", ["experiment_id"]),
        ("idx_experiment_tags_tag", "experiment_tags", ["tag"]),
    ]

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def create_indices(self) -> None:
        for idx_name, table, columns in self._INDEX_DEFINITIONS:
            col_str = ", ".join(columns)
            sql = (
                f"CREATE INDEX IF NOT EXISTS {idx_name} "
                f"ON {table} ({col_str})"
            )
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError as exc:
                logger.warning("Could not create index %s: %s", idx_name, exc)
        self._conn.commit()
        logger.info("Created %d indices", len(self._INDEX_DEFINITIONS))

    def drop_indices(self) -> None:
        for idx_name, _, _ in self._INDEX_DEFINITIONS:
            self._conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
        self._conn.commit()
        logger.info("Dropped all custom indices")

    def analyze(self) -> None:
        self._conn.execute("ANALYZE")
        self._conn.commit()
        logger.info("ANALYZE complete")

    def reindex(self) -> None:
        self._conn.execute("REINDEX")
        self._conn.commit()
        logger.info("REINDEX complete")

    def list_indices(self) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master "
            "WHERE type = 'index' AND sql IS NOT NULL "
            "ORDER BY tbl_name, name"
        )
        return [
            {"name": row[0], "table": row[1], "sql": row[2]}
            for row in cursor.fetchall()
        ]

    def get_index_stats(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            cursor = self._conn.execute(
                "SELECT idx, stat FROM sqlite_stat1 ORDER BY idx"
            )
            for row in cursor.fetchall():
                results.append({"index": row[0], "stat": row[1]})
        except sqlite3.OperationalError:
            logger.debug("sqlite_stat1 not available; run ANALYZE first")
        return results


# ---------------------------------------------------------------------------
# ResultsAggregator
# ---------------------------------------------------------------------------
class ResultsAggregator:
    """Aggregation utilities operating on collections of MetricRecord."""

    @staticmethod
    def aggregate_by_algorithm(
        metrics: List[MetricRecord],
    ) -> Dict[str, Dict[str, Any]]:
        algo_map: Dict[str, List[MetricRecord]] = {}
        for m in metrics:
            algo = m.metadata.get("algorithm_name", "unknown")
            algo_map.setdefault(algo, []).append(m)

        result: Dict[str, Dict[str, Any]] = {}
        for algo, records in algo_map.items():
            values = [r.value for r in records]
            result[algo] = {
                "count": len(records),
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "median": float(np.median(values)) if values else 0.0,
                "metrics": {r.metric_name for r in records},
            }
        return result

    @staticmethod
    def aggregate_by_task(
        metrics: List[MetricRecord],
    ) -> Dict[str, Dict[str, Any]]:
        task_map: Dict[str, List[MetricRecord]] = {}
        for m in metrics:
            task = m.metadata.get("task_domain", "unknown")
            task_map.setdefault(task, []).append(m)

        result: Dict[str, Dict[str, Any]] = {}
        for task, records in task_map.items():
            values = [r.value for r in records]
            result[task] = {
                "count": len(records),
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "median": float(np.median(values)) if values else 0.0,
                "algorithms": {
                    r.metadata.get("algorithm_name", "unknown") for r in records
                },
            }
        return result

    @staticmethod
    def aggregate_by_metric(
        metrics: List[MetricRecord],
    ) -> Dict[str, Dict[str, Any]]:
        metric_map: Dict[str, List[MetricRecord]] = {}
        for m in metrics:
            metric_map.setdefault(m.metric_name, []).append(m)

        result: Dict[str, Dict[str, Any]] = {}
        for name, records in metric_map.items():
            values = [r.value for r in records]
            result[name] = {
                "count": len(records),
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "median": float(np.median(values)) if values else 0.0,
                "metric_type": records[0].metric_type if records else "",
                "algorithms": {
                    r.metadata.get("algorithm_name", "unknown") for r in records
                },
            }
        return result

    @staticmethod
    def rank_algorithms(
        metrics: List[MetricRecord],
        metric_name: str,
        higher_is_better: bool = True,
    ) -> List[Dict[str, Any]]:
        filtered = [m for m in metrics if m.metric_name == metric_name]
        algo_values: Dict[str, List[float]] = {}
        for m in filtered:
            algo = m.metadata.get("algorithm_name", "unknown")
            algo_values.setdefault(algo, []).append(m.value)

        algo_means = [
            {"algorithm": algo, "mean": float(np.mean(vals)), "n": len(vals)}
            for algo, vals in algo_values.items()
        ]
        algo_means.sort(key=lambda x: x["mean"], reverse=higher_is_better)

        for rank, entry in enumerate(algo_means, start=1):
            entry["rank"] = rank

        return algo_means

    @staticmethod
    def compute_summary_statistics(
        metrics: List[MetricRecord],
    ) -> Dict[str, Any]:
        if not metrics:
            return {
                "total_metrics": 0,
                "unique_runs": 0,
                "unique_metric_names": 0,
                "by_type": {},
            }

        values = [m.value for m in metrics]
        run_ids = {m.run_id for m in metrics}
        metric_names = {m.metric_name for m in metrics}

        by_type: Dict[str, Dict[str, Any]] = {}
        for mt in MetricType:
            typed = [m for m in metrics if m.metric_type == mt.value]
            if typed:
                tv = [m.value for m in typed]
                by_type[mt.value] = {
                    "count": len(typed),
                    "mean": float(np.mean(tv)),
                    "std": float(np.std(tv, ddof=1)) if len(tv) > 1 else 0.0,
                    "min": float(np.min(tv)),
                    "max": float(np.max(tv)),
                    "median": float(np.median(tv)),
                    "metric_names": list({m.metric_name for m in typed}),
                }

        return {
            "total_metrics": len(metrics),
            "unique_runs": len(run_ids),
            "unique_metric_names": len(metric_names),
            "global_mean": float(np.mean(values)),
            "global_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "global_min": float(np.min(values)),
            "global_max": float(np.max(values)),
            "global_median": float(np.median(values)),
            "by_type": by_type,
        }

    @staticmethod
    def compute_pairwise_comparison(
        metrics: List[MetricRecord],
        metric_name: str,
    ) -> Dict[str, Dict[str, float]]:
        filtered = [m for m in metrics if m.metric_name == metric_name]
        algo_values: Dict[str, List[float]] = {}
        for m in filtered:
            algo = m.metadata.get("algorithm_name", "unknown")
            algo_values.setdefault(algo, []).append(m.value)

        algo_means = {a: float(np.mean(v)) for a, v in algo_values.items()}
        algos = sorted(algo_means.keys())

        result: Dict[str, Dict[str, float]] = {}
        for a in algos:
            result[a] = {}
            for b in algos:
                result[a][b] = algo_means[a] - algo_means[b]
        return result

    @staticmethod
    def compute_effect_sizes(
        metrics: List[MetricRecord],
        metric_name: str,
        baseline_algo: str,
    ) -> Dict[str, Dict[str, float]]:
        filtered = [m for m in metrics if m.metric_name == metric_name]
        algo_values: Dict[str, List[float]] = {}
        for m in filtered:
            algo = m.metadata.get("algorithm_name", "unknown")
            algo_values.setdefault(algo, []).append(m.value)

        if baseline_algo not in algo_values:
            return {}

        baseline = algo_values[baseline_algo]
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 1e-10

        result: Dict[str, Dict[str, float]] = {}
        for algo, vals in algo_values.items():
            if algo == baseline_algo:
                continue
            algo_mean = float(np.mean(vals))
            algo_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1e-10
            pooled_std = float(
                np.sqrt((baseline_std ** 2 + algo_std ** 2) / 2)
            )
            if pooled_std < 1e-15:
                pooled_std = 1e-15
            cohens_d = (algo_mean - baseline_mean) / pooled_std
            result[algo] = {
                "mean": algo_mean,
                "std": algo_std,
                "cohens_d": cohens_d,
                "diff": algo_mean - baseline_mean,
                "relative_improvement": (
                    (algo_mean - baseline_mean) / abs(baseline_mean)
                    if abs(baseline_mean) > 1e-15
                    else 0.0
                ),
            }
        return result


# ---------------------------------------------------------------------------
# ResultsDatabase
# ---------------------------------------------------------------------------
class ResultsDatabase:
    """SQLite-backed storage for experiment results."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        conn = self._get_connection()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64 MB
        self.create_tables()
        self._ensure_schema_version()
        self._indexer = DatabaseIndex(conn)

    # -- connection --------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @property
    def connection(self) -> sqlite3.Connection:
        return self._get_connection()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # -- schema ------------------------------------------------------------

    def create_tables(self) -> None:
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config TEXT DEFAULT '{}',
                status TEXT DEFAULT 'pending'
                    CHECK (status IN ('pending','running','completed','failed')),
                parent_experiment_id TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS experiment_tags (
                experiment_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (experiment_id, tag),
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                algorithm_name TEXT NOT NULL,
                task_domain TEXT DEFAULT '',
                config TEXT DEFAULT '{}',
                seed INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
                    CHECK (status IN
                        ('pending','running','completed','failed','cancelled')),
                elapsed_time REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}',
                error_message TEXT DEFAULT NULL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_type TEXT DEFAULT 'diversity'
                    CHECK (metric_type IN ('diversity','quality')),
                value REAL DEFAULT 0.0,
                values_json TEXT DEFAULT '[]',
                computed_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                percentile_rank REAL DEFAULT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS generations (
                generation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                prompt_text TEXT DEFAULT '',
                prompt_id TEXT DEFAULT '',
                generated_texts TEXT DEFAULT '[]',
                token_ids TEXT DEFAULT '[]',
                log_probs TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                scores TEXT DEFAULT '[]',
                FOREIGN KEY (run_id)
                    REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()

    def _ensure_schema_version(self) -> None:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT version FROM schema_version WHERE id = 1"
        )
        row = cursor.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (id, version, updated_at) VALUES (1, ?, ?)",
                (CURRENT_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def get_schema_version(self) -> int:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT version FROM schema_version WHERE id = 1"
        )
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def migrate(self, target_version: int = CURRENT_SCHEMA_VERSION) -> None:
        current = self.get_schema_version()
        if current >= target_version:
            logger.info(
                "Already at schema version %d (target %d)",
                current,
                target_version,
            )
            return

        conn = self._get_connection()
        for version in range(current + 1, target_version + 1):
            statements = _MIGRATIONS.get(version, [])
            for stmt in statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" in str(exc).lower():
                        logger.debug("Column already exists, skipping: %s", exc)
                    else:
                        raise
            conn.execute(
                "UPDATE schema_version SET version = ?, updated_at = ? WHERE id = 1",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            logger.info("Migrated to schema version %d", version)

    # -- experiments -------------------------------------------------------

    def insert_experiment(self, record: ExperimentRecord) -> str:
        errors = record.validate()
        if errors:
            raise ValueError(f"Invalid ExperimentRecord: {errors}")

        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                INSERT INTO experiments
                    (experiment_id, name, description, created_at, updated_at,
                     config, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.name,
                    record.description,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.config),
                    record.status,
                ),
            )
            for tag in record.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO experiment_tags (experiment_id, tag) VALUES (?, ?)",
                    (record.experiment_id, tag),
                )
            conn.commit()
        return record.experiment_id

    def update_experiment(self, record: ExperimentRecord) -> None:
        record.touch()
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                UPDATE experiments
                SET name = ?, description = ?, updated_at = ?,
                    config = ?, status = ?
                WHERE experiment_id = ?
                """,
                (
                    record.name,
                    record.description,
                    record.updated_at,
                    json.dumps(record.config),
                    record.status,
                    record.experiment_id,
                ),
            )
            conn.execute(
                "DELETE FROM experiment_tags WHERE experiment_id = ?",
                (record.experiment_id,),
            )
            for tag in record.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO experiment_tags (experiment_id, tag) VALUES (?, ?)",
                    (record.experiment_id, tag),
                )
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        tags_cursor = conn.execute(
            "SELECT tag FROM experiment_tags WHERE experiment_id = ?",
            (experiment_id,),
        )
        tags = [r[0] for r in tags_cursor.fetchall()]

        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            config=json.loads(row["config"]),
            status=row["status"],
            tags=tags,
        )

    def list_experiments(
        self,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExperimentRecord]:
        conn = self._get_connection()
        qb = QueryBuilder().select("experiments")
        if status:
            qb.where("status", "=", status)
        if tag:
            qb.join(
                "experiment_tags",
                "experiments.experiment_id = experiment_tags.experiment_id",
            ).where("experiment_tags.tag", "=", tag)
        qb.order_by("created_at", "DESC").limit(limit).offset(offset)

        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        experiments: List[ExperimentRecord] = []
        for row in rows:
            eid = row["experiment_id"]
            tags_cursor = conn.execute(
                "SELECT tag FROM experiment_tags WHERE experiment_id = ?",
                (eid,),
            )
            tags = [t[0] for t in tags_cursor.fetchall()]
            experiments.append(
                ExperimentRecord(
                    experiment_id=eid,
                    name=row["name"],
                    description=row["description"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    config=json.loads(row["config"]),
                    status=row["status"],
                    tags=tags,
                )
            )
        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        conn = self._get_connection()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            conn.commit()
        return cursor.rowcount > 0

    def count_experiments(self, status: Optional[str] = None) -> int:
        conn = self._get_connection()
        if status:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = ?", (status,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
        return cursor.fetchone()[0]

    # -- runs --------------------------------------------------------------

    def insert_run(self, record: RunRecord) -> str:
        errors = record.validate()
        if errors:
            raise ValueError(f"Invalid RunRecord: {errors}")

        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                INSERT INTO runs
                    (run_id, experiment_id, algorithm_name, task_domain,
                     config, seed, timestamp, status, elapsed_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.experiment_id,
                    record.algorithm_name,
                    record.task_domain,
                    json.dumps(record.config),
                    record.seed,
                    record.timestamp,
                    record.status,
                    record.elapsed_time,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        return record.run_id

    def update_run(self, record: RunRecord) -> None:
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                UPDATE runs
                SET algorithm_name = ?, task_domain = ?, config = ?,
                    seed = ?, timestamp = ?, status = ?,
                    elapsed_time = ?, metadata = ?
                WHERE run_id = ?
                """,
                (
                    record.algorithm_name,
                    record.task_domain,
                    json.dumps(record.config),
                    record.seed,
                    record.timestamp,
                    record.status,
                    record.elapsed_time,
                    json.dumps(record.metadata),
                    record.run_id,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def get_runs(
        self,
        experiment_id: str,
        algorithm_name: Optional[str] = None,
        status: Optional[str] = None,
        task_domain: Optional[str] = None,
    ) -> List[RunRecord]:
        conn = self._get_connection()
        qb = QueryBuilder().select("runs").where("experiment_id", "=", experiment_id)
        if algorithm_name:
            qb.where("algorithm_name", "=", algorithm_name)
        if status:
            qb.where("status", "=", status)
        if task_domain:
            qb.where("task_domain", "=", task_domain)
        qb.order_by("timestamp", "ASC")

        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return [self._row_to_run(row) for row in cursor.fetchall()]

    def delete_run(self, run_id: str) -> bool:
        conn = self._get_connection()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM runs WHERE run_id = ?", (run_id,)
            )
            conn.commit()
        return cursor.rowcount > 0

    def count_runs(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        conn = self._get_connection()
        qb = QueryBuilder().select("runs").count()
        if experiment_id:
            qb.where("experiment_id", "=", experiment_id)
        if status:
            qb.where("status", "=", status)
        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return cursor.fetchone()[0]

    def get_run_algorithms(self, experiment_id: str) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT DISTINCT algorithm_name FROM runs WHERE experiment_id = ? "
            "ORDER BY algorithm_name",
            (experiment_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_run_task_domains(self, experiment_id: str) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT DISTINCT task_domain FROM runs WHERE experiment_id = ? "
            "ORDER BY task_domain",
            (experiment_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            algorithm_name=row["algorithm_name"],
            task_domain=row["task_domain"],
            config=json.loads(row["config"]),
            seed=row["seed"],
            timestamp=row["timestamp"],
            status=row["status"],
            elapsed_time=row["elapsed_time"],
            metadata=json.loads(row["metadata"]),
        )

    # -- metrics -----------------------------------------------------------

    def insert_metric(self, record: MetricRecord) -> str:
        errors = record.validate()
        if errors:
            raise ValueError(f"Invalid MetricRecord: {errors}")

        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                INSERT INTO metrics
                    (metric_id, run_id, metric_name, metric_type,
                     value, values_json, computed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.metric_id,
                    record.run_id,
                    record.metric_name,
                    record.metric_type,
                    record.value,
                    json.dumps(record.values),
                    record.computed_at,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        return record.metric_id

    def batch_insert_metrics(self, records: List[MetricRecord]) -> List[str]:
        ids: List[str] = []
        conn = self._get_connection()
        with self._lock:
            for record in records:
                errors = record.validate()
                if errors:
                    raise ValueError(
                        f"Invalid MetricRecord ({record.metric_id}): {errors}"
                    )
                conn.execute(
                    """
                    INSERT INTO metrics
                        (metric_id, run_id, metric_name, metric_type,
                         value, values_json, computed_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.metric_id,
                        record.run_id,
                        record.metric_name,
                        record.metric_type,
                        record.value,
                        json.dumps(record.values),
                        record.computed_at,
                        json.dumps(record.metadata),
                    ),
                )
                ids.append(record.metric_id)
            conn.commit()
        return ids

    def get_metric(self, metric_id: str) -> Optional[MetricRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM metrics WHERE metric_id = ?", (metric_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_metric(row)

    def get_metrics(
        self,
        run_id: str,
        metric_name: Optional[str] = None,
        metric_type: Optional[str] = None,
    ) -> List[MetricRecord]:
        conn = self._get_connection()
        qb = QueryBuilder().select("metrics").where("run_id", "=", run_id)
        if metric_name:
            qb.where("metric_name", "=", metric_name)
        if metric_type:
            qb.where("metric_type", "=", metric_type)
        qb.order_by("computed_at", "ASC")

        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return [self._row_to_metric(row) for row in cursor.fetchall()]

    def query_metrics(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order_by: str = "computed_at",
        direction: str = "ASC",
        limit: int = 1000,
        offset: int = 0,
    ) -> List[MetricRecord]:
        conn = self._get_connection()
        qb = QueryBuilder().select("metrics")

        if filters:
            if "run_id" in filters:
                qb.where("run_id", "=", filters["run_id"])
            if "run_ids" in filters:
                qb.where("run_id", "IN", filters["run_ids"])
            if "metric_name" in filters:
                qb.where("metric_name", "=", filters["metric_name"])
            if "metric_names" in filters:
                qb.where("metric_name", "IN", filters["metric_names"])
            if "metric_type" in filters:
                qb.where("metric_type", "=", filters["metric_type"])
            if "min_value" in filters:
                qb.where("value", ">=", filters["min_value"])
            if "max_value" in filters:
                qb.where("value", "<=", filters["max_value"])
            if "value_between" in filters:
                qb.where("value", "BETWEEN", filters["value_between"])
            if "computed_after" in filters:
                qb.where("computed_at", ">=", filters["computed_after"])
            if "computed_before" in filters:
                qb.where("computed_at", "<=", filters["computed_before"])

        qb.order_by(order_by, direction).limit(limit).offset(offset)
        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return [self._row_to_metric(row) for row in cursor.fetchall()]

    def delete_metrics(self, run_id: str) -> int:
        conn = self._get_connection()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM metrics WHERE run_id = ?", (run_id,)
            )
            conn.commit()
        return cursor.rowcount

    def count_metrics(
        self,
        run_id: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> int:
        conn = self._get_connection()
        qb = QueryBuilder().select("metrics").count()
        if run_id:
            qb.where("run_id", "=", run_id)
        if metric_name:
            qb.where("metric_name", "=", metric_name)
        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return cursor.fetchone()[0]

    def get_unique_metric_names(
        self, experiment_id: Optional[str] = None
    ) -> List[str]:
        conn = self._get_connection()
        if experiment_id:
            cursor = conn.execute(
                """
                SELECT DISTINCT m.metric_name
                FROM metrics m
                JOIN runs r ON m.run_id = r.run_id
                WHERE r.experiment_id = ?
                ORDER BY m.metric_name
                """,
                (experiment_id,),
            )
        else:
            cursor = conn.execute(
                "SELECT DISTINCT metric_name FROM metrics ORDER BY metric_name"
            )
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _row_to_metric(row: sqlite3.Row) -> MetricRecord:
        return MetricRecord(
            metric_id=row["metric_id"],
            run_id=row["run_id"],
            metric_name=row["metric_name"],
            metric_type=row["metric_type"],
            value=row["value"],
            values=json.loads(row["values_json"]),
            computed_at=row["computed_at"],
            metadata=json.loads(row["metadata"]),
        )

    # -- generations -------------------------------------------------------

    def insert_generation(self, record: GenerationRecord) -> str:
        errors = record.validate()
        if errors:
            raise ValueError(f"Invalid GenerationRecord: {errors}")

        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                INSERT INTO generations
                    (generation_id, run_id, prompt_text, prompt_id,
                     generated_texts, token_ids, log_probs, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.generation_id,
                    record.run_id,
                    record.prompt_text,
                    record.prompt_id,
                    json.dumps(record.generated_texts),
                    json.dumps(record.token_ids),
                    json.dumps(record.log_probs),
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        return record.generation_id

    def batch_insert_generations(
        self, records: List[GenerationRecord]
    ) -> List[str]:
        ids: List[str] = []
        conn = self._get_connection()
        with self._lock:
            for record in records:
                errors = record.validate()
                if errors:
                    raise ValueError(
                        f"Invalid GenerationRecord ({record.generation_id}): {errors}"
                    )
                conn.execute(
                    """
                    INSERT INTO generations
                        (generation_id, run_id, prompt_text, prompt_id,
                         generated_texts, token_ids, log_probs, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.generation_id,
                        record.run_id,
                        record.prompt_text,
                        record.prompt_id,
                        json.dumps(record.generated_texts),
                        json.dumps(record.token_ids),
                        json.dumps(record.log_probs),
                        json.dumps(record.metadata),
                    ),
                )
                ids.append(record.generation_id)
            conn.commit()
        return ids

    def get_generation(self, generation_id: str) -> Optional[GenerationRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM generations WHERE generation_id = ?",
            (generation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_generation(row)

    def get_generations(
        self,
        run_id: str,
        prompt_id: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[GenerationRecord]:
        conn = self._get_connection()
        qb = QueryBuilder().select("generations").where("run_id", "=", run_id)
        if prompt_id:
            qb.where("prompt_id", "=", prompt_id)
        qb.limit(limit).offset(offset)

        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return [self._row_to_generation(row) for row in cursor.fetchall()]

    def delete_generations(self, run_id: str) -> int:
        conn = self._get_connection()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM generations WHERE run_id = ?", (run_id,)
            )
            conn.commit()
        return cursor.rowcount

    def count_generations(self, run_id: Optional[str] = None) -> int:
        conn = self._get_connection()
        if run_id:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM generations WHERE run_id = ?", (run_id,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM generations")
        return cursor.fetchone()[0]

    def search_generations(
        self, query: str, run_id: Optional[str] = None, limit: int = 100
    ) -> List[GenerationRecord]:
        conn = self._get_connection()
        qb = QueryBuilder().select("generations")
        qb.where("prompt_text", "LIKE", f"%{query}%")
        if run_id:
            qb.where("run_id", "=", run_id)
        qb.limit(limit)

        sql, params = qb.build()
        cursor = conn.execute(sql, params)
        return [self._row_to_generation(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_generation(row: sqlite3.Row) -> GenerationRecord:
        return GenerationRecord(
            generation_id=row["generation_id"],
            run_id=row["run_id"],
            prompt_text=row["prompt_text"],
            prompt_id=row["prompt_id"],
            generated_texts=json.loads(row["generated_texts"]),
            token_ids=json.loads(row["token_ids"]),
            log_probs=json.loads(row["log_probs"]),
            metadata=json.loads(row["metadata"]),
        )

    # -- high-level queries ------------------------------------------------

    def get_algorithm_comparison(
        self, experiment_id: str
    ) -> Dict[str, Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                r.algorithm_name,
                m.metric_name,
                m.metric_type,
                COUNT(m.metric_id)         AS num_measurements,
                AVG(m.value)               AS mean_value,
                MIN(m.value)               AS min_value,
                MAX(m.value)               AS max_value,
                GROUP_CONCAT(m.value, ',') AS all_values
            FROM runs r
            JOIN metrics m ON r.run_id = m.run_id
            WHERE r.experiment_id = ?
            GROUP BY r.algorithm_name, m.metric_name, m.metric_type
            ORDER BY r.algorithm_name, m.metric_name
            """,
            (experiment_id,),
        )

        comparison: Dict[str, Dict[str, Any]] = {}
        for row in cursor.fetchall():
            algo = row["algorithm_name"]
            metric = row["metric_name"]

            if algo not in comparison:
                comparison[algo] = {"metrics": {}}

            raw_values = [
                float(v) for v in row["all_values"].split(",") if v
            ]
            std_val = float(np.std(raw_values, ddof=1)) if len(raw_values) > 1 else 0.0

            comparison[algo]["metrics"][metric] = {
                "metric_type": row["metric_type"],
                "count": row["num_measurements"],
                "mean": row["mean_value"],
                "std": std_val,
                "min": row["min_value"],
                "max": row["max_value"],
                "median": float(np.median(raw_values)) if raw_values else 0.0,
                "values": raw_values,
            }

        # Add run-level summary
        for algo in comparison:
            run_cursor = conn.execute(
                """
                SELECT COUNT(*) as n_runs,
                       AVG(elapsed_time) as avg_time,
                       SUM(elapsed_time) as total_time
                FROM runs
                WHERE experiment_id = ? AND algorithm_name = ?
                """,
                (experiment_id, algo),
            )
            run_row = run_cursor.fetchone()
            comparison[algo]["n_runs"] = run_row["n_runs"]
            comparison[algo]["avg_elapsed_time"] = run_row["avg_time"]
            comparison[algo]["total_elapsed_time"] = run_row["total_time"]

        return comparison

    def get_metric_statistics(
        self, experiment_id: str, metric_name: str
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                m.value,
                r.algorithm_name,
                r.task_domain,
                r.seed,
                m.run_id
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE r.experiment_id = ?
              AND m.metric_name = ?
            ORDER BY m.value
            """,
            (experiment_id, metric_name),
        )
        rows = cursor.fetchall()

        if not rows:
            return {
                "metric_name": metric_name,
                "experiment_id": experiment_id,
                "count": 0,
                "by_algorithm": {},
                "by_task_domain": {},
            }

        all_values = [row["value"] for row in rows]
        arr = np.array(all_values)

        by_algorithm: Dict[str, Dict[str, Any]] = {}
        by_task: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            algo = row["algorithm_name"]
            task = row["task_domain"]
            val = row["value"]

            by_algorithm.setdefault(algo, {"values": []})["values"].append(val)
            if task:
                by_task.setdefault(task, {"values": []})["values"].append(val)

        for algo, data in by_algorithm.items():
            v = np.array(data["values"])
            data.update({
                "count": len(v),
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)),
                "q75": float(np.percentile(v, 75)),
                "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            })

        for task, data in by_task.items():
            v = np.array(data["values"])
            data.update({
                "count": len(v),
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "median": float(np.median(v)),
            })

        return {
            "metric_name": metric_name,
            "experiment_id": experiment_id,
            "count": len(all_values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "skewness": float(_safe_skewness(arr)),
            "kurtosis": float(_safe_kurtosis(arr)),
            "by_algorithm": by_algorithm,
            "by_task_domain": by_task,
        }

    def get_pareto_data(
        self,
        experiment_id: str,
        objectives: List[str],
    ) -> np.ndarray:
        if len(objectives) < 2:
            raise ValueError("Need at least 2 objectives for Pareto analysis")

        conn = self._get_connection()
        # Collect per-run mean values for each objective
        run_ids_cursor = conn.execute(
            "SELECT run_id FROM runs WHERE experiment_id = ?",
            (experiment_id,),
        )
        run_ids = [r[0] for r in run_ids_cursor.fetchall()]

        if not run_ids:
            return np.empty((0, len(objectives)))

        data_rows: List[List[float]] = []
        valid_run_ids: List[str] = []

        for rid in run_ids:
            obj_values: List[Optional[float]] = []
            for obj in objectives:
                cursor = conn.execute(
                    """
                    SELECT AVG(value) as avg_val
                    FROM metrics
                    WHERE run_id = ? AND metric_name = ?
                    """,
                    (rid, obj),
                )
                row = cursor.fetchone()
                if row is None or row["avg_val"] is None:
                    obj_values.append(None)
                else:
                    obj_values.append(float(row["avg_val"]))

            if all(v is not None for v in obj_values):
                data_rows.append([v for v in obj_values])  # type: ignore
                valid_run_ids.append(rid)

        if not data_rows:
            return np.empty((0, len(objectives)))

        return np.array(data_rows)

    def get_pareto_front(
        self,
        experiment_id: str,
        objectives: List[str],
        maximize: Optional[List[bool]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        data = self.get_pareto_data(experiment_id, objectives)
        if data.size == 0:
            return np.empty((0, len(objectives))), []

        if maximize is None:
            maximize = [True] * len(objectives)

        signs = np.array([1.0 if m else -1.0 for m in maximize])
        signed = data * signs

        n = len(signed)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(signed[j] >= signed[i]) and np.any(signed[j] > signed[i]):
                    is_dominated[i] = True
                    break

        front_indices = [i for i in range(n) if not is_dominated[i]]
        return data[front_indices], front_indices

    def get_metrics_for_experiment(
        self, experiment_id: str, metric_name: Optional[str] = None
    ) -> List[MetricRecord]:
        conn = self._get_connection()
        if metric_name:
            cursor = conn.execute(
                """
                SELECT m.*
                FROM metrics m
                JOIN runs r ON m.run_id = r.run_id
                WHERE r.experiment_id = ? AND m.metric_name = ?
                ORDER BY m.computed_at
                """,
                (experiment_id, metric_name),
            )
        else:
            cursor = conn.execute(
                """
                SELECT m.*
                FROM metrics m
                JOIN runs r ON m.run_id = r.run_id
                WHERE r.experiment_id = ?
                ORDER BY m.computed_at
                """,
                (experiment_id,),
            )
        return [self._row_to_metric(row) for row in cursor.fetchall()]

    def get_metrics_with_run_info(
        self, experiment_id: str
    ) -> List[MetricRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT m.*, r.algorithm_name, r.task_domain
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE r.experiment_id = ?
            ORDER BY r.algorithm_name, m.metric_name
            """,
            (experiment_id,),
        )
        records: List[MetricRecord] = []
        for row in cursor.fetchall():
            rec = self._row_to_metric(row)
            rec.metadata["algorithm_name"] = row["algorithm_name"]
            rec.metadata["task_domain"] = row["task_domain"]
            records.append(rec)
        return records

    def get_best_run(
        self,
        experiment_id: str,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> Optional[RunRecord]:
        conn = self._get_connection()
        direction = "DESC" if higher_is_better else "ASC"
        cursor = conn.execute(
            f"""
            SELECT r.*
            FROM runs r
            JOIN metrics m ON r.run_id = m.run_id
            WHERE r.experiment_id = ? AND m.metric_name = ?
            ORDER BY m.value {direction}
            LIMIT 1
            """,
            (experiment_id, metric_name),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def get_experiment_timeline(
        self, experiment_id: str
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                r.run_id,
                r.algorithm_name,
                r.timestamp,
                r.elapsed_time,
                r.status,
                COUNT(m.metric_id) as n_metrics,
                COUNT(g.generation_id) as n_generations
            FROM runs r
            LEFT JOIN metrics m ON r.run_id = m.run_id
            LEFT JOIN generations g ON r.run_id = g.run_id
            WHERE r.experiment_id = ?
            GROUP BY r.run_id
            ORDER BY r.timestamp ASC
            """,
            (experiment_id,),
        )
        return [
            {
                "run_id": row["run_id"],
                "algorithm_name": row["algorithm_name"],
                "timestamp": row["timestamp"],
                "elapsed_time": row["elapsed_time"],
                "status": row["status"],
                "n_metrics": row["n_metrics"],
                "n_generations": row["n_generations"],
            }
            for row in cursor.fetchall()
        ]

    def get_cross_experiment_comparison(
        self,
        experiment_ids: List[str],
        metric_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for eid in experiment_ids:
            stats = self.get_metric_statistics(eid, metric_name)
            exp = self.get_experiment(eid)
            result[eid] = {
                "experiment_name": exp.name if exp else eid,
                "statistics": stats,
            }
        return result

    # -- export / import ---------------------------------------------------

    def export_to_dict(self, experiment_id: str) -> Dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        runs = self.get_runs(experiment_id)
        runs_data: List[Dict[str, Any]] = []
        for run in runs:
            metrics = self.get_metrics(run.run_id)
            generations = self.get_generations(run.run_id)
            runs_data.append({
                "run": run.to_dict(),
                "metrics": [m.to_dict() for m in metrics],
                "generations": [g.to_dict() for g in generations],
            })

        return {
            "version": CURRENT_SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment.to_dict(),
            "runs": runs_data,
        }

    def import_from_dict(self, data: Dict[str, Any]) -> str:
        version = data.get("version", 1)
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Data version {version} is newer than supported {CURRENT_SCHEMA_VERSION}"
            )

        exp_data = data["experiment"]
        experiment = ExperimentRecord.from_dict(exp_data)

        existing = self.get_experiment(experiment.experiment_id)
        if existing is not None:
            experiment.experiment_id = str(uuid.uuid4())

        self.insert_experiment(experiment)

        for run_data in data.get("runs", []):
            run = RunRecord.from_dict(run_data["run"])
            run.experiment_id = experiment.experiment_id
            old_run_id = run.run_id
            run.run_id = str(uuid.uuid4())
            self.insert_run(run)

            for metric_data in run_data.get("metrics", []):
                metric = MetricRecord.from_dict(metric_data)
                metric.metric_id = str(uuid.uuid4())
                metric.run_id = run.run_id
                self.insert_metric(metric)

            for gen_data in run_data.get("generations", []):
                gen = GenerationRecord.from_dict(gen_data)
                gen.generation_id = str(uuid.uuid4())
                gen.run_id = run.run_id
                self.insert_generation(gen)

        return experiment.experiment_id

    def export_to_json(self, experiment_id: str, path: str) -> None:
        data = self.export_to_dict(experiment_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported experiment %s to %s", experiment_id, path)

    def import_from_json(self, path: str) -> str:
        with open(path, "r") as f:
            data = json.load(f)
        eid = self.import_from_dict(data)
        logger.info("Imported experiment %s from %s", eid, path)
        return eid

    def export_metrics_csv(
        self,
        experiment_id: str,
        path: str,
        metric_names: Optional[List[str]] = None,
    ) -> None:
        conn = self._get_connection()
        if metric_names:
            placeholders = ", ".join("?" for _ in metric_names)
            cursor = conn.execute(
                f"""
                SELECT r.algorithm_name, r.task_domain, r.seed,
                       m.metric_name, m.metric_type, m.value, m.computed_at
                FROM metrics m
                JOIN runs r ON m.run_id = r.run_id
                WHERE r.experiment_id = ?
                  AND m.metric_name IN ({placeholders})
                ORDER BY r.algorithm_name, m.metric_name
                """,
                [experiment_id] + metric_names,
            )
        else:
            cursor = conn.execute(
                """
                SELECT r.algorithm_name, r.task_domain, r.seed,
                       m.metric_name, m.metric_type, m.value, m.computed_at
                FROM metrics m
                JOIN runs r ON m.run_id = r.run_id
                WHERE r.experiment_id = ?
                ORDER BY r.algorithm_name, m.metric_name
                """,
                (experiment_id,),
            )

        rows = cursor.fetchall()
        with open(path, "w") as f:
            f.write(
                "algorithm_name,task_domain,seed,metric_name,"
                "metric_type,value,computed_at\n"
            )
            for row in rows:
                f.write(
                    f"{row['algorithm_name']},{row['task_domain']},"
                    f"{row['seed']},{row['metric_name']},"
                    f"{row['metric_type']},{row['value']},"
                    f"{row['computed_at']}\n"
                )
        logger.info("Exported metrics CSV for %s to %s", experiment_id, path)

    # -- maintenance -------------------------------------------------------

    def vacuum(self) -> None:
        conn = self._get_connection()
        conn.execute("VACUUM")
        logger.info("VACUUM complete")

    def backup(self, path: str) -> None:
        if self._db_path == ":memory:":
            dest = sqlite3.connect(path)
            self._get_connection().backup(dest)
            dest.close()
        else:
            self.close()
            shutil.copy2(self._db_path, path)
            self._get_connection()  # reconnect
        logger.info("Database backed up to %s", path)

    def get_database_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        stats: Dict[str, Any] = {}
        for table in ("experiments", "runs", "metrics", "generations", "experiment_tags"):
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]

        if self._db_path != ":memory:" and os.path.exists(self._db_path):
            stats["file_size_bytes"] = os.path.getsize(self._db_path)
            stats["file_size_mb"] = round(
                stats["file_size_bytes"] / (1024 * 1024), 2
            )

        stats["schema_version"] = self.get_schema_version()

        cursor = conn.execute("PRAGMA page_count")
        stats["page_count"] = cursor.fetchone()[0]
        cursor = conn.execute("PRAGMA page_size")
        stats["page_size"] = cursor.fetchone()[0]

        return stats

    def integrity_check(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute("PRAGMA integrity_check")
        results = [row[0] for row in cursor.fetchall()]
        return results

    def optimize(self) -> None:
        conn = self._get_connection()
        self._indexer.create_indices()
        self._indexer.analyze()
        conn.execute("PRAGMA optimize")
        logger.info("Database optimized")

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ResultsDatabase":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -- transaction helper ------------------------------------------------

    class _Transaction:
        def __init__(self, db: "ResultsDatabase") -> None:
            self._db = db
            self._conn = db._get_connection()

        def __enter__(self) -> sqlite3.Connection:
            self._conn.execute("BEGIN IMMEDIATE")
            return self._conn

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()

    def transaction(self) -> "_Transaction":
        return self._Transaction(self)

    # -- bulk operations ---------------------------------------------------

    def bulk_update_run_status(
        self, run_ids: List[str], status: str
    ) -> int:
        conn = self._get_connection()
        total = 0
        with self._lock:
            for rid in run_ids:
                cursor = conn.execute(
                    "UPDATE runs SET status = ? WHERE run_id = ?",
                    (status, rid),
                )
                total += cursor.rowcount
            conn.commit()
        return total

    def clone_experiment(
        self,
        experiment_id: str,
        new_name: Optional[str] = None,
        include_metrics: bool = True,
        include_generations: bool = False,
    ) -> str:
        source = self.get_experiment(experiment_id)
        if source is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        clone = ExperimentRecord(
            name=new_name or f"{source.name} (clone)",
            description=source.description,
            config=copy.deepcopy(source.config),
            status=ExperimentStatus.PENDING.value,
            tags=list(source.tags),
        )
        self.insert_experiment(clone)

        runs = self.get_runs(experiment_id)
        for run in runs:
            new_run = RunRecord(
                experiment_id=clone.experiment_id,
                algorithm_name=run.algorithm_name,
                task_domain=run.task_domain,
                config=copy.deepcopy(run.config),
                seed=run.seed,
                status=run.status,
                elapsed_time=run.elapsed_time,
                metadata=copy.deepcopy(run.metadata),
            )
            self.insert_run(new_run)

            if include_metrics:
                metrics = self.get_metrics(run.run_id)
                for m in metrics:
                    new_m = MetricRecord(
                        run_id=new_run.run_id,
                        metric_name=m.metric_name,
                        metric_type=m.metric_type,
                        value=m.value,
                        values=list(m.values),
                        metadata=copy.deepcopy(m.metadata),
                    )
                    self.insert_metric(new_m)

            if include_generations:
                generations = self.get_generations(run.run_id)
                for g in generations:
                    new_g = GenerationRecord(
                        run_id=new_run.run_id,
                        prompt_text=g.prompt_text,
                        prompt_id=g.prompt_id,
                        generated_texts=list(g.generated_texts),
                        token_ids=[list(t) for t in g.token_ids],
                        log_probs=list(g.log_probs),
                        metadata=copy.deepcopy(g.metadata),
                    )
                    self.insert_generation(new_g)

        return clone.experiment_id

    def merge_experiments(
        self,
        experiment_ids: List[str],
        merged_name: str = "Merged Experiment",
    ) -> str:
        merged = ExperimentRecord(
            name=merged_name,
            description=f"Merged from: {', '.join(experiment_ids)}",
            tags=["merged"],
        )
        self.insert_experiment(merged)

        for eid in experiment_ids:
            runs = self.get_runs(eid)
            for run in runs:
                new_run = RunRecord(
                    experiment_id=merged.experiment_id,
                    algorithm_name=run.algorithm_name,
                    task_domain=run.task_domain,
                    config=copy.deepcopy(run.config),
                    seed=run.seed,
                    status=run.status,
                    elapsed_time=run.elapsed_time,
                    metadata={
                        **copy.deepcopy(run.metadata),
                        "source_experiment_id": eid,
                        "source_run_id": run.run_id,
                    },
                )
                self.insert_run(new_run)

                metrics = self.get_metrics(run.run_id)
                for m in metrics:
                    new_m = MetricRecord(
                        run_id=new_run.run_id,
                        metric_name=m.metric_name,
                        metric_type=m.metric_type,
                        value=m.value,
                        values=list(m.values),
                        metadata=copy.deepcopy(m.metadata),
                    )
                    self.insert_metric(new_m)

        return merged.experiment_id

    # -- percentile computation -------------------------------------------

    def compute_percentile_ranks(self, experiment_id: str) -> None:
        conn = self._get_connection()
        metric_names = self.get_unique_metric_names(experiment_id)

        with self._lock:
            for mname in metric_names:
                cursor = conn.execute(
                    """
                    SELECT m.metric_id, m.value
                    FROM metrics m
                    JOIN runs r ON m.run_id = r.run_id
                    WHERE r.experiment_id = ? AND m.metric_name = ?
                    ORDER BY m.value ASC
                    """,
                    (experiment_id, mname),
                )
                rows = cursor.fetchall()
                if not rows:
                    continue

                n = len(rows)
                for rank, row in enumerate(rows):
                    pct = (rank / max(n - 1, 1)) * 100.0
                    conn.execute(
                        "UPDATE metrics SET percentile_rank = ? WHERE metric_id = ?",
                        (pct, row["metric_id"]),
                    )
            conn.commit()

    # -- correlation analysis ---------------------------------------------

    def compute_metric_correlations(
        self, experiment_id: str
    ) -> Dict[str, Dict[str, float]]:
        conn = self._get_connection()
        metric_names = self.get_unique_metric_names(experiment_id)

        run_ids_cursor = conn.execute(
            "SELECT run_id FROM runs WHERE experiment_id = ?",
            (experiment_id,),
        )
        run_ids = [r[0] for r in run_ids_cursor.fetchall()]

        if not run_ids or len(metric_names) < 2:
            return {}

        # Build run_id -> metric_name -> avg_value mapping
        metric_matrix: Dict[str, Dict[str, float]] = {}
        for rid in run_ids:
            metric_matrix[rid] = {}
            for mname in metric_names:
                cursor = conn.execute(
                    "SELECT AVG(value) as avg_val FROM metrics "
                    "WHERE run_id = ? AND metric_name = ?",
                    (rid, mname),
                )
                row = cursor.fetchone()
                if row and row["avg_val"] is not None:
                    metric_matrix[rid][mname] = float(row["avg_val"])

        # Compute pairwise correlations
        correlations: Dict[str, Dict[str, float]] = {}
        for m1 in metric_names:
            correlations[m1] = {}
            for m2 in metric_names:
                vals1: List[float] = []
                vals2: List[float] = []
                for rid in run_ids:
                    if m1 in metric_matrix.get(rid, {}) and m2 in metric_matrix.get(rid, {}):
                        vals1.append(metric_matrix[rid][m1])
                        vals2.append(metric_matrix[rid][m2])

                if len(vals1) < 2:
                    correlations[m1][m2] = float("nan")
                else:
                    arr1 = np.array(vals1)
                    arr2 = np.array(vals2)
                    if np.std(arr1) < 1e-15 or np.std(arr2) < 1e-15:
                        correlations[m1][m2] = float("nan")
                    else:
                        corr = np.corrcoef(arr1, arr2)[0, 1]
                        correlations[m1][m2] = float(corr)

        return correlations

    # -- search & filter ---------------------------------------------------

    def search_experiments(
        self,
        query: str,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[ExperimentRecord]:
        conn = self._get_connection()
        sql = """
            SELECT DISTINCT e.*
            FROM experiments e
        """
        params: List[Any] = []
        conditions: List[str] = []

        if tags:
            sql += " JOIN experiment_tags et ON e.experiment_id = et.experiment_id"
            placeholders = ", ".join("?" for _ in tags)
            conditions.append(f"et.tag IN ({placeholders})")
            params.extend(tags)

        conditions.append("(e.name LIKE ? OR e.description LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])

        if status:
            conditions.append("e.status = ?")
            params.append(status)

        sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY e.updated_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        experiments: List[ExperimentRecord] = []
        for row in cursor.fetchall():
            eid = row["experiment_id"]
            tags_cursor = conn.execute(
                "SELECT tag FROM experiment_tags WHERE experiment_id = ?",
                (eid,),
            )
            etags = [t[0] for t in tags_cursor.fetchall()]
            experiments.append(
                ExperimentRecord(
                    experiment_id=eid,
                    name=row["name"],
                    description=row["description"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    config=json.loads(row["config"]),
                    status=row["status"],
                    tags=etags,
                )
            )
        return experiments

    def get_runs_by_seed(
        self, experiment_id: str, seed: int
    ) -> List[RunRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? AND seed = ? "
            "ORDER BY algorithm_name",
            (experiment_id, seed),
        )
        return [self._row_to_run(row) for row in cursor.fetchall()]

    def get_seed_analysis(
        self, experiment_id: str, metric_name: str
    ) -> Dict[int, Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT r.seed, r.algorithm_name, m.value
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE r.experiment_id = ? AND m.metric_name = ?
            ORDER BY r.seed, r.algorithm_name
            """,
            (experiment_id, metric_name),
        )

        seed_data: Dict[int, Dict[str, List[float]]] = {}
        for row in cursor.fetchall():
            seed = row["seed"]
            algo = row["algorithm_name"]
            seed_data.setdefault(seed, {}).setdefault(algo, []).append(row["value"])

        result: Dict[int, Dict[str, Any]] = {}
        for seed, algos in seed_data.items():
            result[seed] = {}
            for algo, vals in algos.items():
                result[seed][algo] = {
                    "mean": float(np.mean(vals)),
                    "values": vals,
                    "count": len(vals),
                }
        return result

    # -- leaderboard -------------------------------------------------------

    def get_leaderboard(
        self,
        experiment_id: str,
        metric_name: str,
        higher_is_better: bool = True,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        direction = "DESC" if higher_is_better else "ASC"
        cursor = conn.execute(
            f"""
            SELECT
                r.algorithm_name,
                r.task_domain,
                r.seed,
                r.run_id,
                AVG(m.value)   AS mean_value,
                MIN(m.value)   AS min_value,
                MAX(m.value)   AS max_value,
                COUNT(m.value) AS n
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE r.experiment_id = ? AND m.metric_name = ?
            GROUP BY r.run_id
            ORDER BY mean_value {direction}
            LIMIT ?
            """,
            (experiment_id, metric_name, top_k),
        )

        leaderboard: List[Dict[str, Any]] = []
        for rank, row in enumerate(cursor.fetchall(), start=1):
            leaderboard.append({
                "rank": rank,
                "algorithm_name": row["algorithm_name"],
                "task_domain": row["task_domain"],
                "seed": row["seed"],
                "run_id": row["run_id"],
                "mean_value": row["mean_value"],
                "min_value": row["min_value"],
                "max_value": row["max_value"],
                "n_measurements": row["n"],
            })
        return leaderboard

    # -- generation statistics ---------------------------------------------

    def get_generation_statistics(
        self, run_id: str
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                COUNT(*)                          AS n_prompts,
                AVG(LENGTH(prompt_text))           AS avg_prompt_len,
                AVG(LENGTH(generated_texts))       AS avg_gen_data_len
            FROM generations
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None or row["n_prompts"] == 0:
            return {"n_prompts": 0}

        generations = self.get_generations(run_id)
        n_gens_per_prompt = [len(g.generated_texts) for g in generations]
        total_tokens = sum(g.total_tokens for g in generations)
        avg_log_probs = [
            g.mean_log_prob for g in generations if g.log_probs
        ]
        gen_lengths = [
            len(text)
            for g in generations
            for text in g.generated_texts
        ]

        return {
            "n_prompts": row["n_prompts"],
            "avg_prompt_length": float(row["avg_prompt_len"] or 0),
            "total_generations": sum(n_gens_per_prompt),
            "avg_generations_per_prompt": (
                float(np.mean(n_gens_per_prompt)) if n_gens_per_prompt else 0.0
            ),
            "total_tokens": total_tokens,
            "avg_generation_length": (
                float(np.mean(gen_lengths)) if gen_lengths else 0.0
            ),
            "avg_log_prob": (
                float(np.mean(avg_log_probs)) if avg_log_probs else 0.0
            ),
        }

    # -- time series -------------------------------------------------------

    def get_metric_time_series(
        self,
        experiment_id: str,
        metric_name: str,
        algorithm_name: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        conn = self._get_connection()
        qb = QueryBuilder().select(
            "metrics m",
            [
                "m.metric_id", "m.value", "m.computed_at",
                "r.algorithm_name", "r.run_id",
            ],
        ).join("runs r", "m.run_id = r.run_id")
        qb.where_raw("r.experiment_id = ?", [experiment_id])
        qb.where("m.metric_name", "=", metric_name)
        if algorithm_name:
            qb.where("r.algorithm_name", "=", algorithm_name)
        qb.order_by("m.computed_at", "ASC")

        sql, params = qb.build()
        cursor = conn.execute(sql, params)

        series: Dict[str, List[Dict[str, Any]]] = {}
        for row in cursor.fetchall():
            algo = row["algorithm_name"]
            series.setdefault(algo, []).append({
                "metric_id": row["metric_id"],
                "value": row["value"],
                "computed_at": row["computed_at"],
                "run_id": row["run_id"],
            })
        return series

    # -- cleanup -----------------------------------------------------------

    def remove_orphaned_records(self) -> Dict[str, int]:
        conn = self._get_connection()
        removed: Dict[str, int] = {}

        with self._lock:
            cursor = conn.execute(
                """
                DELETE FROM metrics
                WHERE run_id NOT IN (SELECT run_id FROM runs)
                """
            )
            removed["orphaned_metrics"] = cursor.rowcount

            cursor = conn.execute(
                """
                DELETE FROM generations
                WHERE run_id NOT IN (SELECT run_id FROM runs)
                """
            )
            removed["orphaned_generations"] = cursor.rowcount

            cursor = conn.execute(
                """
                DELETE FROM runs
                WHERE experiment_id NOT IN
                    (SELECT experiment_id FROM experiments)
                """
            )
            removed["orphaned_runs"] = cursor.rowcount

            cursor = conn.execute(
                """
                DELETE FROM experiment_tags
                WHERE experiment_id NOT IN
                    (SELECT experiment_id FROM experiments)
                """
            )
            removed["orphaned_tags"] = cursor.rowcount

            conn.commit()
        return removed

    def purge_failed_runs(self, experiment_id: str) -> int:
        conn = self._get_connection()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM runs WHERE experiment_id = ? AND status = 'failed'",
                (experiment_id,),
            )
            conn.commit()
        return cursor.rowcount

    # -- duplicate detection -----------------------------------------------

    def find_duplicate_runs(
        self, experiment_id: str
    ) -> List[List[str]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT algorithm_name, task_domain, seed, config,
                   GROUP_CONCAT(run_id) AS run_ids,
                   COUNT(*) AS cnt
            FROM runs
            WHERE experiment_id = ?
            GROUP BY algorithm_name, task_domain, seed, config
            HAVING cnt > 1
            """,
            (experiment_id,),
        )
        duplicates: List[List[str]] = []
        for row in cursor.fetchall():
            duplicates.append(row["run_ids"].split(","))
        return duplicates

    # -- tagging helpers ---------------------------------------------------

    def add_experiment_tag(self, experiment_id: str, tag: str) -> None:
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                "INSERT OR IGNORE INTO experiment_tags (experiment_id, tag) VALUES (?, ?)",
                (experiment_id, tag),
            )
            conn.execute(
                "UPDATE experiments SET updated_at = ? WHERE experiment_id = ?",
                (datetime.now(timezone.utc).isoformat(), experiment_id),
            )
            conn.commit()

    def remove_experiment_tag(self, experiment_id: str, tag: str) -> None:
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                "DELETE FROM experiment_tags WHERE experiment_id = ? AND tag = ?",
                (experiment_id, tag),
            )
            conn.execute(
                "UPDATE experiments SET updated_at = ? WHERE experiment_id = ?",
                (datetime.now(timezone.utc).isoformat(), experiment_id),
            )
            conn.commit()

    def get_experiment_tags(self, experiment_id: str) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT tag FROM experiment_tags WHERE experiment_id = ? ORDER BY tag",
            (experiment_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_all_tags(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT DISTINCT tag FROM experiment_tags ORDER BY tag"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_experiments_by_tag(self, tag: str) -> List[ExperimentRecord]:
        return self.list_experiments(tag=tag)

    # -- run progress tracking --------------------------------------------

    def update_run_progress(
        self,
        run_id: str,
        progress: float,
        message: str = "",
    ) -> None:
        conn = self._get_connection()
        with self._lock:
            run = self.get_run(run_id)
            if run is None:
                raise ValueError(f"Run {run_id} not found")
            run.metadata["progress"] = min(max(progress, 0.0), 1.0)
            if message:
                run.metadata["progress_message"] = message
            conn.execute(
                "UPDATE runs SET metadata = ? WHERE run_id = ?",
                (json.dumps(run.metadata), run_id),
            )
            conn.commit()

    def get_experiment_progress(
        self, experiment_id: str
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT status, COUNT(*) as cnt
            FROM runs
            WHERE experiment_id = ?
            GROUP BY status
            """,
            (experiment_id,),
        )
        status_counts: Dict[str, int] = {}
        total = 0
        for row in cursor.fetchall():
            status_counts[row["status"]] = row["cnt"]
            total += row["cnt"]

        completed = status_counts.get("completed", 0)
        return {
            "total_runs": total,
            "status_counts": status_counts,
            "completion_rate": completed / total if total > 0 else 0.0,
            "is_complete": completed == total and total > 0,
        }

    # -- metric comparison helpers -----------------------------------------

    def get_metric_pivot_table(
        self,
        experiment_id: str,
        metric_name: str,
    ) -> Dict[str, Dict[str, float]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT r.algorithm_name, r.task_domain, AVG(m.value) as avg_val
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE r.experiment_id = ? AND m.metric_name = ?
            GROUP BY r.algorithm_name, r.task_domain
            ORDER BY r.algorithm_name, r.task_domain
            """,
            (experiment_id, metric_name),
        )

        pivot: Dict[str, Dict[str, float]] = {}
        for row in cursor.fetchall():
            algo = row["algorithm_name"]
            task = row["task_domain"] or "default"
            pivot.setdefault(algo, {})[task] = float(row["avg_val"])
        return pivot

    def get_metric_distribution(
        self,
        experiment_id: str,
        metric_name: str,
        n_bins: int = 20,
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT m.value
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE r.experiment_id = ? AND m.metric_name = ?
            ORDER BY m.value
            """,
            (experiment_id, metric_name),
        )
        values = [row["value"] for row in cursor.fetchall()]
        if not values:
            return {"metric_name": metric_name, "count": 0, "bins": [], "counts": []}

        arr = np.array(values)
        counts, bin_edges = np.histogram(arr, bins=n_bins)

        return {
            "metric_name": metric_name,
            "count": len(values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
        }

    # -- raw SQL escape hatch ---------------------------------------------

    def execute_raw(
        self, sql: str, params: Optional[Sequence[Any]] = None
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(sql, params or [])
        if cursor.description is None:
            conn.commit()
            return []
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute_script(self, script: str) -> None:
        conn = self._get_connection()
        conn.executescript(script)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-15:
        return 0.0
    m3 = np.mean((arr - mean) ** 3)
    return float(m3 / (std ** 3)) * (n * (n - 1)) ** 0.5 / (n - 2) if n > 2 else 0.0


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-15:
        return 0.0
    m4 = np.mean((arr - mean) ** 4)
    kurt = m4 / (std ** 4) - 3.0
    return float(kurt)


def create_results_database(
    db_path: str = ":memory:",
    create_indices: bool = True,
) -> ResultsDatabase:
    db = ResultsDatabase(db_path)
    if create_indices:
        indexer = DatabaseIndex(db.connection)
        indexer.create_indices()
        indexer.analyze()
    return db


def open_results_database(db_path: str) -> ResultsDatabase:
    if not os.path.exists(db_path) and db_path != ":memory:":
        raise FileNotFoundError(f"Database not found: {db_path}")
    return ResultsDatabase(db_path)
