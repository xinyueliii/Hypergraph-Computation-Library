"""Stable request and result contracts shared by native and isolated tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TaskRequest:
    run_id: str
    component_id: str
    action: str
    task_id: str | None = None
    inputs: tuple[Path, ...] = ()
    checkpoint: Path | None = None
    output_dir: Path = Path("artifacts")
    device: str = "auto"
    seed: int = 2026
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskResult:
    run_id: str
    component_id: str
    status: str
    task_id: str | None = None
    predictions: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    timing: Mapping[str, float] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    message: str = ""
