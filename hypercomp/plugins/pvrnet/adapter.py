"""Core-environment adapter for the PVRNet worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="pvrnet",
    task_id="multimodal_3d_shape_recognition",
    source_directory="third_party/PVRNet",
    worker_script="hypercomp/plugins/pvrnet/worker.py",
    python_environment_variable="HYPERCOMP_PVRNET_PYTHON",
)


def run_pvrnet(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
