"""Core-environment adapter for the YOLOv13 worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="yolov13",
    task_id="object_detection",
    source_directory="third_party/YOLOv13",
    worker_script="hypercomp/plugins/ultralytics_detection_worker.py",
    python_environment_variable="HYPERCOMP_YOLOV13_PYTHON",
)


def run_yolov13(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
