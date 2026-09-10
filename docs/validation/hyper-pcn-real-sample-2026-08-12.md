# Hyper-PCN mesh-derived sample validation

Status: `adapted`

- Runtime: shared `visual-remote-cu121` on RTX 3090
- Checkpoint: official PCN checkpoint already recorded in the component manifest
- Input: deterministic half-surface sample derived from Manifold40 `piano_0303.obj`
- Run: `hyper-pcn-20260812-124234`
- Output stages: 1,024, 2,048, and 16,384 points
- Timing: 0.48 seconds worker inference; 4.91 seconds platform total; 300.60 MiB peak allocated GPU memory
- Artifacts: completed point cloud and side-by-side input/completion PNG

This improves the earlier random-point smoke by accepting a real mesh-derived
partial point cloud through the common `predict` action. It validates direct
inference and visualization, not PCN benchmark metrics or official-dataset
accuracy.
