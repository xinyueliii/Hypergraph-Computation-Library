"""Core-environment adapter for HGM2R."""

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.plugins.process import WorkerSpec, run_isolated_worker


SPECIFICATION = WorkerSpec(
    component_id="hgm2r",
    task_id="multimodal_3d_object_retrieval",
    source_directory="third_party/HGM2R",
    worker_script="hypercomp/plugins/hgm2r/worker.py",
    python_environment_variable="HYPERCOMP_HGM2R_PYTHON",
)


def run_hgm2r(request: TaskRequest) -> TaskResult:
    return run_isolated_worker(request, SPECIFICATION)
