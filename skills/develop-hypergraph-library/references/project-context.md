# Project context

## Contents

- Current repository facts
- Product story
- Target architecture
- Integration boundary
- Current prioritization
- Source documents

## Current repository facts

- Repository: `xinyueliii/Hypergraph-Computation-Library`.
- Product name: HyperComp, a unified hypergraph computation toolkit for
  intelligent multimedia processing and streaming.
- Core foundation: DeepHypergraph (DHG) v0.9.7, imported from upstream commit
  `bb88281a8fdf7333c46312691027ac80c2b282ef` under Apache-2.0.
- Git relationship to upstream DHG: none. The imported code contains no nested
  `.git`, submodule, or subtree connection.
- Current foundation package: `dhg/` at repository root.
- Current extension package: `hypercomp/` at repository root.
- Component records: `components/`.
- DHG examples and tests: `examples/` and `tests/upstream_dhg/`.
- Architecture and provenance: `docs/` and `docs/upstream/`.

The repository now has a component registry, common request/result contracts,
a CLI runner, and 16 component manifests: DHG, HGNN, HGNN+, Hyper-YOLO,
YOLOv13, SoftHGNN image classification, SuperFast, E-HRSAI, E-3DTrack,
MeshNet, Hyper-PCN, Count Anything, PVRNet, SoftHGNN object detection, HGM2R,
and EvLight V11. Count Anything, PVRNet, and EvLight V11 now
have checkpoint-backed remote unified-entry evidence. SoftHGNN crowd
counting is not registered. SoftHGNN detection and HGM2R are registered and
have untrained remote integration smoke evidence because neither publishes a
trained checkpoint.

HGNN, HGNN+, YOLOv13, Hyper-YOLO, SoftHGNN, SoftHGNN object detection,
SuperFast, E-HRSAI, E-3DTrack, MeshNet, Hyper-PCN, Count Anything, PVRNet,
HGM2R, and EvLight V11
are `adapted`. SoftHGNN has only a deterministic untrained GPU smoke because no
public trained checkpoint is available. SuperFast and E-HRSAI used official
checkpoints with synthetic structured inputs; Hyper-PCN now uses its official PCN
checkpoint with a real mesh-derived partial point cloud and visible completion. These runs prove the unified adapter
paths, not paper benchmark reproduction, official-data reproduction, or
`platform_verified`. MeshNet used a maintainer replacement checkpoint and a
single Manifold40 mesh to reach `adapted`. E-3DTrack used its official
checkpoint and the first official test sequence (`airplane_1/20`) to produce
finite 3D tracks through the unified entry, so it is also `adapted`. The clean core environment and
complete preserved upstream test suite still require explicit verification.

EvLight V11 uses an internal soft-hypergraph research snapshot. Its epoch-19
best checkpoint from a completed 20-epoch run was trained on a 16-sequence
SDSD-in fast subset, compacted by
removing optimizer and duplicate CLIP state, and exercised through HyperComp on
one fixed `pair23/0003` sample. This is checkpoint-backed adapter evidence, not
the CVPR 2024 paper protocol or full-dataset reproduction.

On 2026-08-11, remote execution was consolidated into two shared Python
runtimes. `/root/autodl-tmp/hypercomp-data/envs/visual-cu121` (Python 3.11,
PyTorch 2.2.2+cu121) serves DHG/HGNN, YOLOv13, Hyper-YOLO, SoftHGNN,
SoftHGNN object detection, Hyper-PCN, PVRNet, HGM2R, and EvLight V11.
`/root/autodl-tmp/hypercomp-data/envs/event-cu113` (Python 3.8,
PyTorch 1.10+cu113) serves compatible legacy event and mesh workers, currently
including E-HRSAI, E-3DTrack, and MeshNet. MeshNet and E-3DTrack both have
checkpoint-backed unified-entry sample runs.
SuperFast retains the server's already-validated Python 3.8/PyTorch 1.10 root
runtime. Hyper-PCN reused the visual runtime plus the shared CUDA 12.1 compiler
toolchain; no component-specific Python environment was retained.
Count Anything first reproduced a missing `torch.nn.attention` API in the
shared runtime, then retained `/root/autodl-tmp/hypercomp-data/envs/count-anything-py311`
with PyTorch 2.7.1+cu126. Its official checkpoint and fixed `bus.jpg`/`person`
query produced four point predictions. PVRNet reused the shared visual runtime;
its official final checkpoint correctly classified the deterministic
Manifold40-derived point/12-view sample as `piano` and exported a retrieval feature.

## Product story

Multimedia tasks contain relations among groups of regions, frames, events,
views, entities, points, or voxels. HyperComp uses hypergraphs to represent
these high-order relations through a common computation pattern instead of
reducing every relation to unrelated pairs.

The project goal is not to claim that one model solves every task. The goal is
to reuse DHG as a common high-order computation foundation and reduce the
engineering cost of turning heterogeneous research code into traceable,
runnable components.

## Target architecture

Organize the system into five conceptual layers:

1. Data and structure: ingest multimedia inputs and construct graph,
   bipartite-graph, or hypergraph objects with metadata.
2. DHG computation: incidence and degree matrices, Laplacians, structure
   construction, and vertex-to-hyperedge-to-vertex message passing.
3. Models and plugins: native DHG models plus application worker processes.
4. Tasks: classification, detection, counting, tracking, retrieval, point-cloud
   understanding, and 3D generation as implementation becomes available.
5. Orchestration and presentation: registry, CLI/API, environment selection,
   execution records, result collection, and visualization.

This architecture is a target. Check the repository before describing a layer
or task as implemented.

## Integration boundary

Default to a shared model registry and the two established remote runtimes:
`visual-cu121` for compatible modern components and `event-cu113` for
compatible Python 3.8 legacy event/mesh components. Extend them incrementally
when dependencies are compatible. Organize dataset loading,
preprocessing, inference orchestration, postprocessing, and artifacts by task,
so different tasks can reuse registered models without pretending their raw
data pipelines are identical.

Use native DHG extension for general operators and models that can share the
core environment and work with a boundary such as `forward(X, hg)`. Reuse
common structure construction, splits, training, early stopping, metrics, and
checkpoint handling.

Use worker processes for import isolation, custom CUDA operators, large
modality-specific pipelines, or distinct license obligations. A worker process
may and should reuse a compatible shared Python environment. Create a new
environment only after a reproduced compatibility conflict. Exchange
JSON-compatible requests/results and file paths; do not require workers to
share DHG's internal data loader.

## Current prioritization

- Keep DHG as the base rather than an external component.
- Establish a few working demonstrations before expanding breadth.
- Prioritize visual, event, multi-view, point-cloud, retrieval, and 3D tasks
  with available source code.
- Do not prioritize large-model/RAG or attack/defense components unless the
  user changes scope.
- Checkpoints are optional for cataloging and source integration, but required
  for any inference path that depends on pretrained weights.

## Source documents

This context distills implementation-relevant content from `Core Project.docx`,
`HyperComp参赛作品中文模拟完成稿.docx`, and the matching English submission
draft. Those documents include simulated completed-state prose. Treat their
architecture and SOP as design guidance, not proof of current implementation.
