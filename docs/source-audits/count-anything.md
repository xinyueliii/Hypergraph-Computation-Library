# Count Anything source audit

Status: `adapted` (runtime evidence: `docs/validation/count-anything-remote-adaptation-2026-08-12.md`)

- Source: https://github.com/Mengqi-Lei/count-anything
- Pinned commit: `0fffaefb3bbdcd930c135dcc02e2e359a038f2cb`
- License: Apache-2.0 for code
- Task: single-image text-guided cross-domain object counting and point localization
- Public API: `CountAnything(checkpoint)(image_path, text_query)`
- Input: one image and one non-empty natural-language query
- Output: count, point coordinates and scores, prediction JSON, and rendered image
- Checkpoint: official `count_anything.pt`, 3,565,313,633 bytes, SHA256
  `2f84be3c3af5508388b732827e725cb6d6c4dd826ffda7f80de234ebd1ef0f92`
- Full-data paths: CLOC annotations and source images are required for benchmark
  validation, but not for single-image inference
- Declared dependencies: Python 3.12; PyTorch `>=2.0`; torchvision `>=0.15`;
  NumPy `>=1.26,<2`; Hydra, Triton, timm, pycocotools, and supporting packages
- HyperComp mode: isolated worker in `count-anything-remote-cu126`, retained
  after a reproduced shared-runtime API conflict

The pinned repository includes a purpose-built single-image API that constructs
a one-record inference annotation and invokes the upstream evaluation stack.
HyperComp passes image, query, checkpoint, output directory, and one-GPU
selection through the common request boundary and returns the count, points,
JSON, visualization, timing, and environment. The official checkpoint and a
fixed bundled image produced output on the remote RTX 3090, so the component
is now `adapted`; this does not imply CLOC benchmark reproduction.
