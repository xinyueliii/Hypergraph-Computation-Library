"""Core-environment adapter for the E-HRSAI worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="e-hrsai",
    task_id="event_frame_synthetic_aperture_imaging",
    source_directory="third_party/E-HRSAI",
    worker_script="hypercomp/plugins/e_hrsai/worker.py",
    python_environment_variable="HYPERCOMP_E_HRSAI_PYTHON",
)


def run_e_hrsai(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
