"""Core-environment adapter for the E-3DTrack worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="e-3dtrack",
    task_id="event_based_3d_feature_tracking",
    source_directory="third_party/E-3DTrack",
    worker_script="hypercomp/plugins/e_3dtrack/worker.py",
    python_environment_variable="HYPERCOMP_E_3DTRACK_PYTHON",
)


def run_e_3dtrack(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
