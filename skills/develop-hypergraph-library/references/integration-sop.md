# Component integration SOP

## Contents

- Intake and evidence record
- Original repository gate
- Remote hardware preflight and shared-environment assignment
- Integration-mode decision
- Native-extension procedure
- Worker procedure
- Common runtime contract
- Verification gates
- Delivery and reporting

## 1. Intake and evidence record

Create or update a component manifest before implementation. Record:

- component ID, task family, and proposed integration mode;
- upstream repository and pinned commit or release;
- paper link and method role;
- code, checkpoint, and dataset licenses;
- Python, PyTorch, CUDA, system library, and custom-operator requirements;
- training, evaluation, and inference entry points;
- checkpoint location, expected keys, size, and SHA256 when available;
- minimum legal sample input and expected outputs;
- hard-coded paths, GPU IDs, network services, and secrets;
- current status and supporting evidence path.

If license or redistribution permission is unknown, register the source but do
not copy it into the repository. If a checkpoint is missing, allow source-level
integration but mark pretrained inference unavailable.

## 2. Original repository gate

When the user explicitly requests a code-first milestone, the static source
audit and adapter implementation may precede environment recreation and this
runtime gate. In that case, pin the source, implement the common boundary, run
syntax and manifest checks only, and keep the status at `source_audited`. The
steps below remain mandatory before any `upstream_verified`, `adapted`, or
`platform_verified` claim.

Before adapting internals:

1. Reconstruct the documented environment or a justified compatible variant.
2. Test imports and compiled operators.
3. Run the smallest official example with fixed input.
4. Save the command, environment, logs, timing, output, and failure reason.
5. Retry only after changing a specific diagnosed condition.

Do not advance to `upstream_verified` merely because installation succeeds.
The original task path must produce a valid output.

## 2A. Remote hardware preflight and shared-environment assignment

Before installing dependencies, downloading a large checkpoint, or allocating
the GPU:

1. Use the AutoDL Linux RTX 3090 as the default execution and acceptance
   target. Use Windows only for dependency-free checks unless the user
   explicitly requests local execution.
2. Record the GPU model, total and currently free VRAM, driver version, system
   memory, free disk on the environment/checkpoint/artifact volumes, and
   whether a system CUDA compiler is required.
3. Estimate feasibility from model scale, checkpoint size, input shape,
   batch size, precision, task intermediates, and custom CUDA operators. Start
   inference with batch size one and the smallest official model unless the
   component requires another setting.
4. Assign the component to an existing runtime before installing anything:
   use `visual-cu121` for compatible modern PyTorch visual/DHG/point-cloud
   components, or `event-cu113` for compatible Python 3.8 legacy event/mesh
   components.
5. Add small compatible dependencies to that shared runtime. Do not create a
   full environment merely because the upstream README pins different package
   versions or because the adapter uses a worker process.
6. If only CUDA compilation support is missing, reuse or extend the shared
   toolchain below `/root/autodl-tmp/hypercomp-data/toolchains`; do not clone
   the Python environment for `nvcc` alone.
7. Create a separate named environment only after reproducing a concrete
   Python, PyTorch/CUDA, ABI, binary-extension, or system-library conflict.
   Record the failed compatibility attempt and exact reason for the split.
8. After changing a shared runtime, run `pip check` and focused regression
   checks for the components already assigned to it.
9. Update the version-controlled environment map with environment ID, remote
   path or specification, package versions, assigned models, checkpoint
   policy, dependencies added, and latest verified action.
10. Store environments, checkpoints, samples, caches, and artifacts below
    `/root/autodl-tmp/hypercomp-data`. Check free space before large downloads
    and prefer a minimal legal inference sample to a full training dataset.

Do not classify a component as remote-runnable from dependency inspection
alone; actual output through the declared entry is still required.

## 3. Integration-mode decision

Choose a native extension when all are true:

- the method has reusable hypergraph semantics rather than only a task shell;
- the dependency range is compatible with the core environment;
- inputs can be represented through DHG structures or stable converters;
- licensing permits direct inclusion and modification.

Choose a worker process when any are true:

- upstream imports or package names must be isolated from another component;
- the project relies on fragile compiled extensions or a legacy runtime;
- preprocessing and outputs are strongly modality-specific;
- a process boundary is needed for license, service, or resource isolation.

## 4. Native-extension procedure

1. Place general structures and operators in the appropriate `dhg/` module.
2. Place cross-task registry, execution, and provenance behavior in
   `hypercomp/`; avoid coupling DHG primitives to competition-only concerns.
3. Preserve upstream mathematical behavior and add a focused compatibility
   layer before refactoring model internals.
4. Register a stable model or task ID and typed configuration.
5. Reuse common training and metric code only when semantics match.
6. Add unit tests for shapes, devices, gradients, serialization, and edge
   cases, followed by one minimal task test.

## 5. Worker procedure

1. Assign the worker to an existing compatible shared environment first. Keep
   a separate environment specification only when the conflict evidence in
   section 2A requires it.
2. Replace hard-coded data, checkpoint, output, and GPU paths with parameters
   or environment variables without redesigning the network first.
3. Accept a request through a small worker entry point.
4. Validate inputs before allocating the model or GPU.
5. Return only JSON-compatible values plus artifact paths.
6. Capture stdout/stderr, exit code, runtime, environment, and generated files.
7. Convert native predictions into the common result schema without discarding
   task-specific artifacts.

## 6. Common runtime contract

Minimum `TaskRequest` fields:

| Field | Purpose |
| --- | --- |
| `run_id` | Unique traceable execution ID |
| `task` / `model` / `action` | Registry routing |
| `inputs` | Files and typed input metadata |
| `checkpoint` | Optional weight reference |
| `device` | CPU/GPU selection |
| `seed` | Reproducibility |
| `parameters` | Task-specific options |

Minimum `TaskResult` fields:

| Field | Purpose |
| --- | --- |
| `status` | Success, failure, or skipped state |
| `predictions` | JSON-compatible primary output |
| `metrics` | Named measured values |
| `artifacts` | Files such as images, point clouds, or checkpoints |
| `timing` | Load and execution timing |
| `environment` | Runtime and source provenance |
| `warnings` / `errors` | Actionable diagnostics |

## 7. Verification gates

Use these statuses in order:

| Status | Required evidence |
| --- | --- |
| `registered` | Manifest exists; no runtime claim |
| `source_audited` | Source, license, entry points, dependencies, and assets inspected |
| `upstream_verified` | Pinned original code produces a valid minimal output |
| `adapted` | Common request/result boundary works |
| `platform_verified` | Fixed sample passes acceptance and artifacts are preserved |

At platform acceptance, verify:

- a clean or declared environment can install the component;
- the fixed sample and checkpoint hashes match the record;
- the expected prediction, metric, and artifact files exist;
- environment and source revisions are saved;
- missing dependencies, weights, input fields, or GPU produce clear errors;
- repeated runs are deterministic where the method permits it.

## 8. Delivery and reporting

Commit each component as reviewable milestones, for example source audit,
upstream smoke, adapter, then acceptance. Do not combine several unrelated
imports into one opaque commit.

In reports, label values as upstream-paper results or HyperComp measurements.
Do not combine results from different datasets or protocols into a single
leaderboard. Only call a component runnable or integrated at the evidence level
actually reached.
