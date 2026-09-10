"""Common subprocess boundary for isolated research-code workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.serialization import json_ready, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    component_id: str
    task_id: str
    source_directory: str
    worker_script: str
    python_environment_variable: str


def _artifact_path(value: str, run_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def run_isolated_worker(request: TaskRequest, specification: WorkerSpec) -> TaskResult:
    """Execute a component worker without importing its dependencies into core."""

    if request.component_id != specification.component_id:
        raise ValueError(
            f"Worker '{specification.component_id}' cannot run model '{request.component_id}'."
        )
    if request.task_id not in {None, specification.task_id}:
        raise ValueError(
            f"Worker '{specification.component_id}' belongs to task "
            f"'{specification.task_id}', not '{request.task_id}'."
        )

    started = time.perf_counter()
    run_dir = request.output_dir / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = write_json(run_dir / "request.json", request)
    worker_request_path = run_dir / "worker-request.json"
    response_path = run_dir / "worker-response.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    source_root = Path(
        str(request.parameters.get("source_root", PROJECT_ROOT / specification.source_directory))
    ).resolve()
    worker_script = (PROJECT_ROOT / specification.worker_script).resolve()
    python_executable = str(
        request.parameters.get("python_executable")
        or os.environ.get(specification.python_environment_variable)
        or sys.executable
    )

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Component source is missing: {source_root}. Run scripts/fetch_components.ps1 first."
        )
    if not worker_script.is_file():
        raise FileNotFoundError(f"Worker script is missing: {worker_script}")

    worker_request = json_ready(request)
    worker_request["output_dir"] = str(request.output_dir.resolve())
    worker_request["inputs"] = [str(path.resolve()) for path in request.inputs]
    if request.checkpoint is not None:
        worker_request["checkpoint"] = str(request.checkpoint.resolve())
    write_json(worker_request_path, worker_request)

    command = [
        python_executable,
        str(worker_script),
        "--request",
        str(worker_request_path.resolve()),
        "--response",
        str(response_path.resolve()),
        "--source-root",
        str(source_root),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        result_path = run_dir / "result.json"
        result = TaskResult(
            run_id=request.run_id,
            component_id=request.component_id,
            status="failed",
            task_id=request.task_id or specification.task_id,
            artifacts=(request_path, worker_request_path, stdout_path, stderr_path, result_path),
            timing={"platform_total_seconds": time.perf_counter() - started},
            environment={
                "python_executable": python_executable,
                "source_root": str(source_root),
            },
            errors=(
                f"WorkerProcessError: {specification.component_id} exited with code "
                f"{completed.returncode}.",
            ),
            message=f"Isolated worker failed; inspect {stderr_path}.",
        )
        write_json(result_path, result)
        return result
    if not response_path.is_file():
        raise RuntimeError(f"Worker did not produce its response: {response_path}")

    payload: dict[str, Any] = json.loads(response_path.read_text(encoding="utf-8"))
    worker_artifacts = tuple(
        _artifact_path(str(item), run_dir) for item in payload.get("artifacts", [])
    )
    result_path = run_dir / "result.json"
    result = TaskResult(
        run_id=request.run_id,
        component_id=request.component_id,
        status=str(payload.get("status", "succeeded")),
        task_id=request.task_id or specification.task_id,
        predictions=payload.get("predictions", {}),
        artifacts=(
            request_path,
            worker_request_path,
            response_path,
            stdout_path,
            stderr_path,
            *worker_artifacts,
            result_path,
        ),
        metrics=payload.get("metrics", {}),
        timing={
            **payload.get("timing", {}),
            "platform_total_seconds": time.perf_counter() - started,
        },
        environment={
            **payload.get("environment", {}),
            "python_executable": python_executable,
            "source_root": str(source_root),
        },
        warnings=tuple(payload.get("warnings", [])),
        errors=tuple(payload.get("errors", [])),
        message=str(payload.get("message", "Isolated worker completed.")),
    )
    write_json(result_path, result)
    return result
