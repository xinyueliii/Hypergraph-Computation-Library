from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hypercomp import ComponentRegistry  # noqa: E402


class ComponentRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ComponentRegistry(PROJECT_ROOT / "components")

    def test_registered_components_are_discoverable(self) -> None:
        component_ids = {item.id for item in self.registry.manifests()}
        self.assertEqual(
            component_ids,
            {
                "count-anything",
                "deep-hypergraph",
                "e-3dtrack",
                "e-hrsai",
                "evlight-v11",
                "hgnn",
                "hgnnp",
                "hyper-pcn",
                "hyper-yolo",
                "hgm2r",
                "meshnet",
                "pvrnet",
                "soft-hgnn",
                "soft-hgnn-detection",
                "superfast",
                "yolov13",
            },
        )

    def test_manifest_lookup(self) -> None:
        manifest = self.registry.get("deep-hypergraph")
        self.assertEqual(manifest.integration_mode, "native")
        self.assertEqual(manifest.status, "registered")
        self.assertEqual(manifest.source_path, "dhg")
        self.assertEqual(manifest.upstream_version, "v0.9.7")
        self.assertFalse(manifest.upstream_git_linked)

    def test_native_hgnn_entry_points_are_registered(self) -> None:
        for component_id in ("hgnn", "hgnnp"):
            manifest = self.registry.get(component_id)
            self.assertEqual(manifest.integration_mode, "native")
            self.assertEqual(manifest.task, "vertex_classification")
            self.assertIsNotNone(manifest.entry_point)
            self.assertEqual(manifest.status, "adapted")
            self.assertTrue((PROJECT_ROOT / manifest.evidence_path).is_file())

    def test_external_environment_assignments_are_recorded(self) -> None:
        expected = {
            "hyper-yolo": ("shared-win-cu121", "local_safe", "adapted"),
            "soft-hgnn": ("shared-win-cu121", "local_safe", "adapted"),
            "soft-hgnn-detection": (
                "visual-remote-cu121",
                "remote_recommended",
                "adapted",
            ),
            "superfast": (
                "superfast-remote-cu113",
                "remote_recommended",
                "adapted",
            ),
            "yolov13": ("shared-win-cu121", "local_safe", "adapted"),
            "e-hrsai": (
                "event-remote-cu113",
                "remote_recommended",
                "adapted",
            ),
            "evlight-v11": (
                "visual-remote-cu121",
                "remote_recommended",
                "adapted",
            ),
            "e-3dtrack": (
                "event-remote-cu113",
                "remote_recommended",
                "adapted",
            ),
            "meshnet": (
                "event-remote-cu113",
                "remote_recommended",
                "adapted",
            ),
            "hyper-pcn": (
                "visual-remote-cu121",
                "remote_recommended",
                "adapted",
            ),
            "count-anything": (
                "count-anything-remote-cu126",
                "remote_recommended",
                "adapted",
            ),
            "pvrnet": (
                "visual-remote-cu121",
                "remote_recommended",
                "adapted",
            ),
            "hgm2r": (
                "visual-remote-cu121",
                "remote_recommended",
                "adapted",
            ),
        }
        for component_id, values in expected.items():
            manifest = self.registry.get(component_id)
            self.assertEqual(
                (manifest.environment_id, manifest.local_feasibility, manifest.status),
                values,
            )
            self.assertTrue((PROJECT_ROOT / manifest.evidence_path).is_file())

    def test_native_components_use_the_shared_runtime(self) -> None:
        for component_id in ("deep-hypergraph", "hgnn", "hgnnp"):
            manifest = self.registry.get(component_id)
            self.assertEqual(manifest.environment_id, "shared-win-cu121")
            self.assertEqual(manifest.local_feasibility, "local_safe")


if __name__ == "__main__":
    unittest.main()
