from pathlib import Path
import os
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / ".test-artifacts"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DHG_CACHE_ROOT", str(ARTIFACT_ROOT / "dhg-cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from hypercomp import TaskRequest, run_component  # noqa: E402


class NativeVertexClassificationTest(unittest.TestCase):
    def test_hgnn_family_cpu_smoke_and_checkpoint_restore(self) -> None:
        for component_id in ("hgnn", "hgnnp"):
            smoke = run_component(
                TaskRequest(
                    run_id=f"{component_id}-cpu-smoke",
                    component_id=component_id,
                    action="smoke",
                    output_dir=ARTIFACT_ROOT,
                    device="cpu",
                    seed=7,
                    parameters={"epochs": 2, "hidden_dim": 5},
                )
            )
            self.assertEqual(smoke.status, "succeeded", smoke.errors)
            self.assertEqual(smoke.environment["device"], "cpu")
            self.assertEqual(len(smoke.predictions["classes"]), 12)
            self.assertTrue(all(path.exists() for path in smoke.artifacts))
            checkpoint = next(path for path in smoke.artifacts if path.name == "checkpoint.pt")

            evaluation = run_component(
                TaskRequest(
                    run_id=f"{component_id}-cpu-evaluate",
                    component_id=component_id,
                    action="evaluate",
                    output_dir=ARTIFACT_ROOT,
                    checkpoint=checkpoint,
                    device="cpu",
                    seed=7,
                )
            )
            self.assertEqual(evaluation.status, "succeeded", evaluation.errors)
            self.assertEqual(
                smoke.predictions["classes"],
                evaluation.predictions["classes"],
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA PyTorch environment required")
    def test_hgnn_family_gpu_smoke(self) -> None:
        for component_id in ("hgnn", "hgnnp"):
            result = run_component(
                TaskRequest(
                    run_id=f"{component_id}-gpu-smoke",
                    component_id=component_id,
                    action="smoke",
                    output_dir=ARTIFACT_ROOT,
                    device="cuda:0",
                    seed=11,
                    parameters={"epochs": 1},
                )
            )
            self.assertEqual(result.status, "succeeded", result.errors)
            self.assertEqual(result.environment["device"], "cuda:0")
            self.assertEqual(result.environment["gpu"], torch.cuda.get_device_name(0))

    def test_unknown_component_returns_failure_artifact(self) -> None:
        result = run_component(
            TaskRequest(
                run_id="unknown-component",
                component_id="missing",
                action="smoke",
                output_dir=ARTIFACT_ROOT,
            )
        )
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.errors)
        self.assertTrue(all(path.exists() for path in result.artifacts))


if __name__ == "__main__":
    unittest.main()
