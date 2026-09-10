"""Core-environment adapter for the Hyper-PCN worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="hyper-pcn",
    task_id="point_cloud_completion",
    source_directory="third_party/Hyper-PCN",
    worker_script="hypercomp/plugins/hyper_pcn/worker.py",
    python_environment_variable="HYPERCOMP_HYPER_PCN_PYTHON",
)


def run_hyper_pcn(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
