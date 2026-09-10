# HGM2R source audit

Status: `adapted`; runtime evidence is recorded in
`docs/validation/hgm2r-remote-adaptation-2026-08-12.md`.

- Source: https://github.com/iMoonLab/HGM2R
- Pinned commit: `94d88e5fc727a0abcf730c0528f67484aa1def7f`
- Paper: Hypergraph-Based Multi-Modal Representation for Open-Set 3D Object Retrieval
- Paper URL: https://ieeexplore.ieee.org/document/10319392
- License: Apache-2.0
- Upstream entry: `train_hgm2r.py`
- Published checkpoint: none
- Entry point: `hypercomp.plugins.hgm2r.adapter:run_hgm2r`

The repository includes three aligned ESB feature files rather than only source
code: PointNet point-cloud features `(670, 512)`, VoxNet voxel features
`(670, 6912)`, and ResNet18 multi-view features `(670, 512)`, with shared train,
query, and target masks. More features are linked through Baidu Drive.

The unified `smoke` action accepts these three task-specific `.npy` inputs and
exercises one CMAE optimization step, multimodal feature fusion, hypergraph
construction, one HGNN optimization step, and retrieval ranking. Because no
trained checkpoint is published, the output verifies the real-data execution
chain but is not presented as a paper metric or meaningful trained retrieval
result. Full training remains in the upstream project-specific script.
