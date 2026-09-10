"""Core-environment adapter for the EvLight V11 worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="evlight-v11",
    task_id="event_guided_low_light_enhancement",
    source_directory="third_party/EvLight",
    worker_script="hypercomp/plugins/evlight_v11/worker.py",
    python_environment_variable="HYPERCOMP_EVLIGHT_V11_PYTHON",
)


def run_evlight_v11(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
