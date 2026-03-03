"""Load texts from JSONL and HuggingFace JSON formats."""

import json


# Field names commonly used for text in LLM evaluation outputs
DEFAULT_TEXT_FIELDS = ("text", "output", "response", "content", "generation", "completion")


def load_texts_jsonl(path, text_field=None):
    """Load texts from a JSONL file (one JSON object per line).

    Args:
        path: Path to .jsonl file.
        text_field: Explicit field name. If None, auto-detects from
                    DEFAULT_TEXT_FIELDS.

    Returns:
        List of text strings.

    Raises:
        ValueError: If no recognised text field is found.
    """
    texts = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            field = _resolve_field(obj, text_field, lineno)
            texts.append(str(obj[field]))
    return texts


def load_texts_hf_json(path, text_field=None):
    """Load texts from HuggingFace-style JSON.

    Accepts either ``{"data": [{"text": ...}, ...]}`` or
    ``[{"text": ...}, ...]``.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            "HuggingFace JSON must be a list or an object with a 'data' key"
        )
    texts = []
    for i, obj in enumerate(items, 1):
        if isinstance(obj, str):
            texts.append(obj)
            continue
        field = _resolve_field(obj, text_field, i)
        texts.append(str(obj[field]))
    return texts


def load_texts_auto(path, text_field=None):
    """Auto-detect format by extension and load texts.

    * ``.jsonl``   → JSONL
    * ``.json``    → tries HuggingFace JSON (list / ``{data:[...]}``)
    * ``.csv``     → CSV (returns flat list; groups ignored)
    * ``.parquet`` → Parquet (returns flat list; groups ignored)
    * anything else → plain text (one text per line)
    """
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return load_texts_jsonl(path, text_field=text_field)
    if lower.endswith(".json"):
        return load_texts_hf_json(path, text_field=text_field)
    if lower.endswith(".csv"):
        from src.io.csv_loader import load_csv
        result = load_csv(path, text_column=text_field or "text")
        if isinstance(result, dict):
            return [t for texts in result.values() for t in texts]
        return result
    if lower.endswith(".parquet"):
        from src.io.csv_loader import load_parquet
        result = load_parquet(path, text_column=text_field or "text")
        if isinstance(result, dict):
            return [t for texts in result.values() for t in texts]
        return result
    # Plain text fallback
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_field(obj, explicit_field, lineno):
    """Return the field name to use for extracting text from *obj*."""
    if explicit_field:
        if explicit_field not in obj:
            raise ValueError(
                f"Line {lineno}: field '{explicit_field}' not found "
                f"(keys: {list(obj.keys())})"
            )
        return explicit_field
    for f in DEFAULT_TEXT_FIELDS:
        if f in obj:
            return f
    raise ValueError(
        f"Line {lineno}: no recognised text field "
        f"(tried {DEFAULT_TEXT_FIELDS}, got {list(obj.keys())})"
    )
