"""Core-environment adapter for SoftHGNN object detection."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="soft-hgnn-detection",
    task_id="object_detection",
    source_directory="third_party/SoftHGNN",
    worker_script="hypercomp/plugins/soft_hgnn_detection/worker.py",
    python_environment_variable="HYPERCOMP_SOFT_HGNN_DETECTION_PYTHON",
)


def run_soft_hgnn_detection(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
