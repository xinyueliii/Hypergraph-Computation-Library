"""Manifest loading and component discovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    id: str
    name: str
    task: str
    integration_mode: str
    source_url: str
    paper_url: str
    checkpoint: str
    upstream_environment: str
    entry_point: str | None
    status: str
    source_path: str | None = None
    source_distribution: str | None = None
    upstream_version: str | None = None
    upstream_commit: str | None = None
    upstream_git_linked: bool | None = None
    license: str | None = None
    evidence_path: str | None = None
    environment_id: str | None = None
    local_feasibility: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> "ComponentManifest":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(**payload)


class ComponentRegistry:
    def __init__(self, manifest_root: Path) -> None:
        self.manifest_root = manifest_root

    def manifests(self) -> Iterator[ComponentManifest]:
        for path in sorted(self.manifest_root.rglob("*.json")):
            yield ComponentManifest.from_path(path)

    def get(self, component_id: str) -> ComponentManifest:
        matches = [item for item in self.manifests() if item.id == component_id]
        if not matches:
            raise KeyError(f"Unknown component: {component_id}")
        if len(matches) > 1:
            raise ValueError(f"Duplicate component id: {component_id}")
        return matches[0]

    def for_task(self, task_id: str) -> tuple[ComponentManifest, ...]:
        return tuple(item for item in self.manifests() if item.task == task_id)


@dataclass(frozen=True, slots=True)
class TaskManifest:
    id: str
    name: str
    description: str
    default_model: str

    @classmethod
    def from_path(cls, path: Path) -> "TaskManifest":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(**payload)


class TaskRegistry:
    def __init__(self, manifest_root: Path, components: ComponentRegistry) -> None:
        self.manifest_root = manifest_root
        self.components = components

    def manifests(self) -> Iterator[TaskManifest]:
        for path in sorted(self.manifest_root.rglob("*.json")):
            yield TaskManifest.from_path(path)

    def get(self, task_id: str) -> TaskManifest:
        matches = [item for item in self.manifests() if item.id == task_id]
        if not matches:
            raise KeyError(f"Unknown task: {task_id}")
        if len(matches) > 1:
            raise ValueError(f"Duplicate task id: {task_id}")
        return matches[0]

    def models(self, task_id: str) -> tuple[ComponentManifest, ...]:
        self.get(task_id)
        return self.components.for_task(task_id)

    def resolve(self, task_id: str, model_id: str | None = None) -> ComponentManifest:
        task = self.get(task_id)
        selected_model = model_id or task.default_model
        component = self.components.get(selected_model)
        if component.task != task_id:
            raise ValueError(
                f"Model '{selected_model}' belongs to task '{component.task}', "
                f"not '{task_id}'."
            )
        return component
