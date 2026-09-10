"""HyperComp shared platform package."""

from .contracts import TaskRequest, TaskResult
from .registry import ComponentManifest, ComponentRegistry, TaskManifest, TaskRegistry
from .runner import run_component

__all__ = [
    "ComponentManifest",
    "ComponentRegistry",
    "TaskRequest",
    "TaskResult",
    "TaskManifest",
    "TaskRegistry",
    "run_component",
]

__version__ = "0.1.0"
