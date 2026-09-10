"""Core-environment adapter for the MeshNet worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="meshnet",
    task_id="mesh_shape_recognition",
    source_directory="third_party/MeshNet",
    worker_script="hypercomp/plugins/meshnet/worker.py",
    python_environment_variable="HYPERCOMP_MESHNET_PYTHON",
)


def run_meshnet(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
