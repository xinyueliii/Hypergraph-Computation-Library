# HGM2R remote adaptation record

## Result

HGM2R completed a unified-entry real-feature GPU smoke on the AutoDL RTX 3090.
The component is `adapted`, not `platform_verified` and not a paper-result
reproduction because upstream publishes no trained checkpoint.

## Fixed source and inputs

- Source commit: `94d88e5fc727a0abcf730c0528f67484aa1def7f`
- Run ID: `hgm2r-20260812-01`
- Point-cloud feature: bundled ESB PointNet `(670, 512)`
- Voxel feature: bundled ESB VoxNet `(670, 6912)`
- Multi-view feature: bundled ESB ResNet18 `(670, 512)`
- All modalities share the upstream train/query/target masks and labels.

## Exercised path

The worker performed one CMAE optimization step, fused the three modalities,
constructed the k-nearest-neighbor hypergraph and normalized propagation
matrix, performed one HGM2R HGNN optimization step, exported 256-dimensional
retrieval embeddings, and returned top-5 target rankings for three queries.

## Environment and result

- Environment: shared `visual-remote-cu121`; no new environment was created
- Python: 3.11.15
- PyTorch: 2.2.2+cu121
- SciPy: 1.17.1
- GPU: NVIDIA GeForce RTX 3090
- Samples/modalities: 670 / 3
- CMAE one-step loss: 0.09168294
- HGNN one-step loss: 0.55368960
- Retrieval forward: 0.000632 seconds
- Platform total: 3.2253 seconds
- Peak allocated GPU memory: 466.92 MiB
- Embedding artifact: `/root/autodl-tmp/hypercomp-data/runs/hgm2r-20260812-01/hgm2r_smoke_embeddings.npy`

The first two smoke queries returned same-class examples as their nearest
neighbors, but this is only an execution sanity check. With one training step
and no published checkpoint, these rankings must not be reported as trained
retrieval accuracy or as the paper's benchmark result.
