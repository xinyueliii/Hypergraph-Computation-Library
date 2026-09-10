"""Core-environment adapter for the SoftHGNN image-classification worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="soft-hgnn",
    task_id="image_classification",
    source_directory="third_party/SoftHGNN",
    worker_script="hypercomp/plugins/soft_hgnn/worker.py",
    python_environment_variable="HYPERCOMP_SOFT_HGNN_PYTHON",
)


def run_soft_hgnn(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
