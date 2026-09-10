"""Core-environment adapter for the Hyper-YOLO worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="hyper-yolo",
    task_id="object_detection",
    source_directory="third_party/Hyper-YOLO",
    worker_script="hypercomp/plugins/ultralytics_detection_worker.py",
    python_environment_variable="HYPERCOMP_HYPER_YOLO_PYTHON",
)


def run_hyper_yolo(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
