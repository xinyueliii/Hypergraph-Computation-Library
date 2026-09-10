"""Core-environment adapter for the SuperFast worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="superfast",
    task_id="event_based_video_frame_interpolation",
    source_directory="third_party/SuperFast",
    worker_script="hypercomp/plugins/superfast/worker.py",
    python_environment_variable="HYPERCOMP_SUPERFAST_PYTHON",
)


def run_superfast(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
