# Durable user requirements

Last updated: 2026-08-15

Read this file before every Hypergraph-Computation-Library task. Update it
automatically when the user gives a clear recurring project rule. Keep one-off
requests and secrets out of this file.

## Contents

- Working method
- Architecture and integration
- Completion and evidence
- Current priority and scope
- Documentation and source lists
- Git workflow

## Working method

- Read the project skill and this ledger before answering or acting on every
  request in this repository.
- Record new durable requirements about naming, scope, architecture,
  integration, acceptance, documentation, and Git workflow as part of the
  current task.
- Use Chinese by default for project discussion, planning, and internal working
  documents unless the requested deliverable is English.
- Use Markdown as the main format for ongoing organization and revision.
- Maintain `项目进度与后续计划.md`; mark only completed, evidence-backed work
  with `[x]` and keep unfinished work as `[ ]`.
- Keep a concise integrated-method overview at the beginning of the main
  progress Markdown. Separate the native foundation, actually executed models,
  and code-first source integrations, and update it whenever component status
  changes.
- Prefer making a reasonable in-scope implementation decision over repeatedly
  asking the user about small details. Ask when a choice would materially
  change project scope or the final deliverable.

## Architecture and integration

- Treat root-level `dhg/` as the native project foundation, not as a
  `third_party` dependency. It must not retain a nested Git link to the
  original DHG repository.
- Extend reusable hypergraph structures, operators, and compatible models on
  top of DHG. Put cross-task registration, execution, provenance, and plugin
  behavior in `hypercomp/`.
- Prefer one incrementally extended shared runtime, using the current
  YOLOv13-capable environment as the first-stage base. Add compatible model
  dependencies to it instead of creating a new environment by default; split
  out a worker environment only after a concrete Python, PyTorch, CUDA, custom
  operator, or system-library conflict is confirmed.
- Keep models in one common registration and loading layer. Keep each task's
  dataset reading, preprocessing, inference procedure, postprocessing, and
  task-specific artifacts in its own task module, while exposing them through
  the same request/result contract.
- Standardize the user-visible boundary: component registration, task/model
  selection, inputs, checkpoint, device, outputs, metrics, logs, environment,
  artifacts, and errors.
- Preserve task-specific preprocessing and model internals when rewriting them
  would increase risk without improving the common platform boundary.
- Use native integration for DHG-compatible reusable components; use a worker
  process for import isolation, custom operators, or strongly modality-specific
  pipelines. Assign that worker to a shared environment unless a reproduced
  conflict requires a split.
- If the user explicitly requests local execution, evaluate whether the current
  machine can safely perform its intended inference using the GPU model and
  VRAM, system memory, free disk, checkpoint size, input resolution, batch
  size, precision, and compiled-operator requirements. Record the conclusion
  as local-safe, local-conditional, or remote-recommended with the evidence.
- Maintain a clear model-to-environment mapping. Record which models share the
  base environment, which require a separate environment, the exact reason for
  each split, and the Python, PyTorch, CUDA runtime, checkpoint, and validation
  state for every environment.
- Use the AutoDL Linux RTX 3090 server as the primary runtime and acceptance
  platform for new integrations. The Windows laptop is only a Git working copy
  and may perform dependency-free syntax or manifest checks unless the user
  explicitly reopens local execution.
- Store remote environments, checkpoints, samples, caches, and run artifacts
  below `/root/autodl-tmp/hypercomp-data`; keep them out of Git. Check free
  remote space before large downloads and do not download full training
  datasets when a minimal inference sample is sufficient.
- Run remote training jobs inside a clearly named `tmux` session so they
  survive SSH or desktop interruptions. Save the exact launch command, log
  path, checkpoint directory, and resume command before leaving the job
  unattended; inspect for an existing session/process before relaunching to
  avoid duplicate GPU training.
- Minimize the number of remote environments. Reuse `visual-cu121` for modern
  PyTorch components and `event-cu113` for compatible Python 3.8/legacy event
  and mesh components. Add small dependencies or a compiler toolchain to a
  shared environment before cloning it. Create another full environment only
  after a reproduced version or binary conflict proves sharing unsafe.
- A worker process is an interface and import-isolation mechanism, not a reason
  to create a separate environment. Multiple workers should share one runtime
  when their actual dependencies are compatible.
- After adding dependencies to a shared environment, run `pip check` and
  focused regression checks for the components already assigned to it. Record
  the added packages and evidence in the environment map.

## Completion and evidence

- Use a code-first integration stage when speed matters: pin and download the
  source, implement the common task/model entry and adapters, and perform
  syntax and manifest checks before spending time on environment recreation or
  model execution.
- During this code-first stage, defer dependency installation, checkpoint
  download, and runtime tests. Keep component status at `source_audited` until
  actual execution evidence justifies a later status.
- Provide one entry that selects a task and a model. Add runtime environments
  and fixed-sample tests through that same entry in a later milestone.
- A small number of genuinely runnable demonstrations is sufficient for the
  first milestone; do not delay progress by trying to integrate every model.
- A component can be cataloged or source-adapted without a public checkpoint.
  Do not claim checkpoint-dependent pretrained inference is runnable when the
  required weights are unavailable.
- Keep `registered`, `source_audited`, `upstream_verified`, `adapted`, and
  `platform_verified` distinct and advance them only with matching evidence.
- Do not infer runtime success from a repository, README, paper, or checkpoint
  link. Inspect code and run the relevant path before calling it runnable.
- Distinguish paper results, original-repository results, HyperComp local
  measurements, and simulated competition prose.
- Keep generated datasets, caches, checkpoints, and run outputs out of Git
  unless redistribution is explicitly approved and legally permitted.

## Current priority and scope

- First complete a few visual, frame-event, multi-view, point-cloud, retrieval,
  or 3D demonstrations with available source code.
- Aim for one or two representative integrated methods in each selected
  technical direction. Once a direction already has adequate runnable
  coverage, prioritize filling an uncovered direction over adding another
  near-duplicate model.
- YOLOv13 and Hyper-YOLO are the current checkpoint-backed visual detectors;
  SoftHGNN remains an untrained smoke path. SuperFast, E-HRSAI, and Hyper-PCN
  have unified-entry remote RTX 3090 execution evidence. MeshNet also has a
  checkpoint-backed single-mesh unified-entry result from the replacement
  assets documented in official issues. E-3DTrack also has an official-
  checkpoint unified-entry result on one official test sequence.
- Defer large-model/RAG and attack/defense integrations unless the user
  explicitly reopens those directions.
- When choosing another component, favor complete source, a usable checkpoint,
  a clear inference entry point, legal sample data, visible demo output, and
  feasibility on the remote RTX 3090 with 24 GB memory.
- Count Anything, PVRNet, and Hyper-PCN real-sample work was completed on the
  remote RTX 3090 on 2026-08-12. Preserve their checkpoint hashes, fixed sample
  provenance, environment split rationale, and evidence boundaries in future
  edits; do not downgrade them to `source_audited` without contrary evidence.
- SoftHGNN detection and HGM2R were selected and adapted on the remote RTX 3090
  on 2026-08-12. Both reuse `visual-cu121` and publish no trained checkpoint:
  keep SoftHGNN detection labeled as a random-weight architecture smoke and
  HGM2R as a bundled-real-feature one-step integration smoke, never as paper
  reproduction or trained inference.

## Documentation and source lists

- For concise method lists, include the method name, English fine-grained task
  direction, code link, paper link, checkpoint availability, venue, and year
  when those facts have been verified.
- Keep environment conflicts, interface differences, missing weights, sample
  data, inference-entry availability, integration suitability, and difficulty
  visible in detailed research-code tables.
- Treat competition documents that describe an imagined completed system as
  drafts and design guidance, not as proof of implementation.
- Keep the competition manuscript synchronized with the integration registry:
  every model at `adapted` status must be named and explained in the technical
  solution and must have its actual HyperComp evidence boundary recorded.
- Organize scientific performance comparisons by task rather than mixing
  heterogeneous datasets or metrics in one table. Copy values only from the
  corresponding original paper or its official project page, identify the
  dataset/protocol and metric direction, and never present those literature
  values as HyperComp reproductions. If an exact table is inaccessible, leave
  an explicit source gap instead of estimating or inventing a number.

## Git workflow

- Keep each component or coherent platform milestone in a reviewable commit.
- Run proportionate tests and inspect the diff before committing.
- On this Windows machine, perform GitHub HTTPS network operations once through
  the approved external-network path with PowerShell login disabled; do not
  make repeated sandboxed attempts that trigger `git-remote-https.exe` dialogs.
- Push completed milestones to
  `xinyueliii/Hypergraph-Computation-Library` so local and remote history stay
  aligned.
- On the remote server, do not repeatedly retry GitHub HTTPS authentication or
  save credentials. Prefer `git pull --ff-only` when authentication works;
  otherwise transfer a Git bundle from the already-pushed local repository and
  fast-forward the remote working copy from that bundle.
- Preserve unrelated user changes and never use destructive Git cleanup to
  hide or discard them.
