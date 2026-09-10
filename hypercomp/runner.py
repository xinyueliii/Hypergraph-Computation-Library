"""Component dispatch and failure capture."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Callable

from .contracts import TaskRequest, TaskResult
from .registry import ComponentRegistry
from .serialization import write_json


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COMPONENTS = ComponentRegistry(_PROJECT_ROOT / "components")


def _load_entry_point(specification: str) -> Callable[[TaskRequest], TaskResult]:
    module_name, function_name = specification.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, function_name)


def run_component(request: TaskRequest) -> TaskResult:
    """Run a registered component and preserve actionable failure evidence."""

    try:
        component = _COMPONENTS.get(request.component_id)
        if request.task_id is not None and component.task != request.task_id:
            raise ValueError(
                f"Model '{component.id}' belongs to task '{component.task}', "
                f"not '{request.task_id}'."
            )
        if component.entry_point is None:
            raise RuntimeError(f"Component '{component.id}' has no runnable entry point.")
        if request.component_id in {"hgnn", "hgnnp"}:
            os.environ.setdefault(
                "DHG_CACHE_ROOT",
                str((request.output_dir / ".dhg-cache").resolve()),
            )
        return _load_entry_point(component.entry_point)(request)
    except Exception as error:  # The result contract must also cover import/runtime failures.
        run_dir = request.output_dir / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = write_json(run_dir / "request.json", request)
        error_path = run_dir / "error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        result_path = run_dir / "result.json"
        result = TaskResult(
            run_id=request.run_id,
            component_id=request.component_id,
            status="failed",
            task_id=request.task_id,
            artifacts=(request_path, error_path, result_path),
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            errors=(f"{type(error).__name__}: {error}",),
            message="Component execution failed; inspect error.txt.",
        )
        write_json(result_path, result)
        return result
