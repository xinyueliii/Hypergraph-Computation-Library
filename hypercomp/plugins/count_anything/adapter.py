"""Core-environment adapter for the Count Anything worker."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="count-anything",
    task_id="text_guided_object_counting",
    source_directory="third_party/Count-Anything",
    worker_script="hypercomp/plugins/count_anything/worker.py",
    python_environment_variable="HYPERCOMP_COUNT_ANYTHING_PYTHON",
)


def run_count_anything(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
