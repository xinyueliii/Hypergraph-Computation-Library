"""Create a deterministic point-cloud plus 12-view sample from an OBJ mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    if not vertices or not faces:
        raise ValueError(f"No triangular mesh data found in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), size=count, p=probabilities)]
    uv = rng.random((count, 2), dtype=np.float32)
    reflect = uv.sum(axis=1) > 1
    uv[reflect] = 1 - uv[reflect]
    return chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--retain-x-quantile", type=float)
    args = parser.parse_args()

    vertices, faces = load_obj(args.mesh)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    vertices = (vertices - center) / max(float(scale), 1e-8)
    points = sample_surface(vertices, faces, args.points, args.seed)
    if args.retain_x_quantile is not None:
        if not 0 < args.retain_x_quantile <= 1:
            raise ValueError("--retain-x-quantile must be in (0, 1].")
        points = points[points[:, 0] <= np.quantile(points[:, 0], args.retain_x_quantile)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "points.npy", points.astype(np.float32))
    view_dir = args.output_dir / "views"
    view_dir.mkdir(exist_ok=True)
    for index, azimuth in enumerate(range(0, 360, 30)):
        figure = plt.figure(figsize=(2.24, 2.24), dpi=100)
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                          color="#7895b2", edgecolor="none", shade=True)
        axis.view_init(elev=25, azim=azimuth)
        axis.set_axis_off()
        axis.set_box_aspect((1, 1, 1))
        figure.subplots_adjust(0, 0, 1, 1)
        figure.savefig(view_dir / f"view_{index:02d}.png", facecolor="white")
        plt.close(figure)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
