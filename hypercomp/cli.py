"""Command-line interface for HyperComp component execution."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

from .contracts import TaskRequest
from .registry import ComponentRegistry, TaskRegistry
from .runner import run_component
from .serialization import json_ready


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _registries() -> tuple[ComponentRegistry, TaskRegistry]:
    components = ComponentRegistry(PROJECT_ROOT / "components")
    return components, TaskRegistry(PROJECT_ROOT / "tasks", components)


def _parameter_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parameters(values: Sequence[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Parameter must use KEY=VALUE syntax: {value}")
        key, raw_value = value.split("=", maxsplit=1)
        if not key:
            raise ValueError("Parameter key cannot be empty.")
        parsed[key] = _parameter_value(raw_value)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hypercomp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tasks", help="List registered tasks and default models")
    models_parser = subparsers.add_parser("models", help="List models available for a task")
    models_parser.add_argument("--task", required=True)

    run_parser = subparsers.add_parser("run", help="Run a task with a selected model")
    run_parser.add_argument("--task")
    run_parser.add_argument(
        "--model",
        "--component",
        dest="model",
        help="Model/component ID; defaults to the task's configured model",
    )
    run_parser.add_argument(
        "--action",
        choices=("smoke", "train", "evaluate", "predict"),
        default="smoke",
    )
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--seed", type=int, default=2026)
    run_parser.add_argument("--checkpoint", type=Path)
    run_parser.add_argument("--input", dest="inputs", action="append", type=Path, default=[])
    run_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    run_parser.add_argument("--epochs", type=int)
    run_parser.add_argument("--hidden-dim", type=int)
    run_parser.add_argument("--learning-rate", type=float)
    run_parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task/model-specific option; repeat as needed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    components, tasks = _registries()

    if args.command == "tasks":
        payload = [
            {
                "task": item.id,
                "name": item.name,
                "default_model": item.default_model,
                "models": [model.id for model in tasks.models(item.id)],
            }
            for item in tasks.manifests()
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "models":
        try:
            payload = [json_ready(item) for item in tasks.models(args.task)]
        except (KeyError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.task is None and args.model is None:
        parser.error("run requires --task or --model")
    try:
        if args.task is not None:
            component = tasks.resolve(args.task, args.model)
            task_id = args.task
        else:
            component = components.get(args.model)
            task_id = component.task
    except (KeyError, ValueError) as error:
        parser.error(str(error))

    run_id = args.run_id or f"{component.id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        parameters = _parameters(args.parameter)
    except ValueError as error:
        parser.error(str(error))
    if args.epochs is not None:
        parameters["epochs"] = args.epochs
    if args.hidden_dim is not None:
        parameters["hidden_dim"] = args.hidden_dim
    if args.learning_rate is not None:
        parameters["learning_rate"] = args.learning_rate
    request = TaskRequest(
        run_id=run_id,
        component_id=component.id,
        action=args.action,
        task_id=task_id,
        inputs=tuple(args.inputs),
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        parameters=parameters,
    )
    result = run_component(request)
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "succeeded" else 1
