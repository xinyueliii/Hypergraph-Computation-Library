"""JSON serialization helpers for task contracts and run artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


def json_ready(value: Any) -> Any:
    """Convert task values into JSON-compatible Python objects."""

    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> Path:
    """Write a UTF-8 JSON artifact with deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
