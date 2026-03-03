"""Load texts from CSV and Parquet formats."""

import csv
import os

# Column names commonly used for text in LLM evaluation outputs
DEFAULT_TEXT_COLUMNS = ("text", "output", "response", "content", "generation", "completion", "sentence")

# Column names that indicate a grouping/model column
GROUP_COLUMNS = ("group", "config", "model")


def _detect_text_column(headers, text_column=None):
    """Return the text column name, auto-detecting if needed."""
    if text_column:
        if text_column not in headers:
            raise ValueError(
                f"Column '{text_column}' not found (available: {headers})"
            )
        return text_column
    for col in DEFAULT_TEXT_COLUMNS:
        if col in headers:
            return col
    raise ValueError(
        f"No recognised text column (tried {DEFAULT_TEXT_COLUMNS}, "
        f"available: {headers})"
    )


def _detect_group_column(headers):
    """Return group column name if one exists, else None."""
    for col in GROUP_COLUMNS:
        if col in headers:
            return col
    return None


def load_csv(path, text_column="text", delimiter=","):
    """Load texts from a CSV file.

    Auto-detects common text column names if the explicit *text_column*
    is not present.  If a ``group``, ``config``, or ``model`` column
    exists, returns ``dict[str, list[str]]`` keyed by group name;
    otherwise returns ``list[str]``.

    Args:
        path: Path to .csv file.
        text_column: Preferred text column name (default ``'text'``).
        delimiter: CSV delimiter (default ``','``).

    Returns:
        ``list[str]`` or ``dict[str, list[str]]``.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = list(reader.fieldnames or [])
        text_col = _detect_text_column(headers, text_column if text_column in headers else None)
        group_col = _detect_group_column(headers)

        if group_col:
            groups = {}
            for row in reader:
                g = row[group_col]
                groups.setdefault(g, []).append(str(row[text_col]))
            return groups

        return [str(row[text_col]) for row in reader]


def load_parquet(path, text_column="text"):
    """Load texts from a Parquet file.

    Requires ``pyarrow`` or ``pandas``.  Same auto-detection and
    grouping logic as :func:`load_csv`.

    Args:
        path: Path to .parquet file.
        text_column: Preferred text column name (default ``'text'``).

    Returns:
        ``list[str]`` or ``dict[str, list[str]]``.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        df = table.to_pandas()
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError(
                "Parquet support requires pyarrow or pandas. "
                "Install with: pip install pyarrow  (or)  pip install pandas"
            )

    headers = list(df.columns)
    text_col = _detect_text_column(headers, text_column if text_column in headers else None)
    group_col = _detect_group_column(headers)

    if group_col:
        groups = {}
        for g, sub in df.groupby(group_col):
            groups[str(g)] = [str(v) for v in sub[text_col]]
        return groups

    return [str(v) for v in df[text_col]]
