---
name: develop-hypergraph-library
description: Develop and review Hypergraph-Computation-Library with DHG as its native foundation and preserve the user's durable project requirements. Use for every task in this repository, including code, documentation, planning, research-code integration, manifests, environments, checkpoints, tests, reproducibility evidence, and competition-facing status claims.
---

# Develop Hypergraph Computation Library

## Start every repository task with a requirements pass

Before planning, answering, editing, or running commands for this repository:

1. Read this `SKILL.md` completely.
2. Read `references/user-requirements.md` completely.
3. Inspect the current worktree and the files relevant to the request.
4. Read `references/project-context.md` before changing architecture, package
   layout, public interfaces, component status, or competition-facing claims.
5. Read `references/integration-sop.md` before importing, adapting, testing,
   or promoting any model or task component.
6. Read `references/git-workflow.md` before any GitHub network operation such
   as fetch, pull, push, or `ls-remote` in this Windows desktop environment.

Do not rely on an older conversation summary when the repository or these
references can provide the current state.

## Capture durable user requirements

Treat a user instruction as durable when it defines a recurring project rule,
such as naming, scope, architecture, integration boundaries, priority,
acceptance evidence, documentation format, or Git workflow. When the rule is
clear and reusable, update `references/user-requirements.md` in the same task
and tell the user that the skill caused the update.

Do not record one-off content requests, temporary experiments, speculative
ideas, passwords, tokens, account credentials, or other secrets. If a new
instruction conflicts with an existing requirement, follow the latest explicit
instruction and update the ledger so it contains one current rule rather than
two contradictory rules. Ask only when the conflict would materially change
the requested result and cannot be resolved from context.

## Apply the project boundary

Treat `dhg/` as the main computation foundation. Extend it directly when an
operator, structure, convolution, model, metric, or dataset is broadly useful
and compatible with the core environment. Treat `hypercomp/` as the common
task, registry, execution, provenance, and plugin layer.

## Preserve the central design

Build one toolchain around the flow:

`input -> high-order structure -> hypergraph propagation -> task model -> verified result`

Do not present the project as a folder containing unrelated repositories.
Standardize the boundary that users see while preserving task-specific model
internals where necessary.

Use two integration modes:

1. Use a native extension when the component can operate on DHG structures,
   share the core Python environment, and expose a stable Python API.
2. Use a worker process when upstream import paths, compiled operators,
   modality-specific preprocessing, or license boundaries make in-process
   reuse unsafe or fragile. A worker process does not by itself justify a new
   Python environment; compatible workers must still share an existing
   runtime.

Keep task-specific datasets and preprocessing separate. Share Python/PyTorch/
CUDA runtimes whenever compatibility is demonstrated, and split a runtime only
after a reproduced version, binary, or system-library conflict.

## Use the remote shared-runtime policy

Treat the AutoDL Linux RTX 3090 server as the default execution and acceptance
platform. Treat Windows as the Git working copy and use it only for
dependency-free syntax, manifest, and documentation checks unless the user
explicitly reopens local execution.

Before installing any model dependency, read the current environment map and
assign the component in this order:

1. Reuse `/root/autodl-tmp/hypercomp-data/envs/visual-cu121` for compatible
   modern PyTorch visual, DHG, and point-cloud components.
2. Reuse `/root/autodl-tmp/hypercomp-data/envs/event-cu113` for compatible
   Python 3.8 legacy event and mesh components.
3. Add small compatible dependencies to the selected shared environment and
   run `pip check` plus focused regression checks for its existing models.
4. Reuse a shared compiler toolchain under
   `/root/autodl-tmp/hypercomp-data/toolchains` when only `nvcc` or headers are
   missing.
5. Create another full environment only after recording and reproducing the
   exact conflict that prevents safe sharing.

Store environments, checkpoints, samples, caches, and run artifacts below
`/root/autodl-tmp/hypercomp-data` and keep them out of Git. Check remote free
space before large downloads; prefer a legal minimal inference sample over a
full training dataset.

## Follow the implementation workflow

1. Inspect the current worktree, relevant manifests, tests, and documentation.
   Preserve unrelated user changes.
2. Pin the candidate source revision and record code, weight, and data license
   evidence before copying or adapting code.
3. Inspect the actual entry points, imports, configuration, input schema,
   output schema, checkpoint loading, device assumptions, and sample data.
4. Run the original minimal path before adaptation when the required
   environment and assets are available. Record failures instead of masking
   them.
5. Choose native extension or worker process using the criteria above, then
   assign the worker to the smallest compatible shared environment.
6. Implement the smallest stable adapter and avoid rewriting the research
   model before its original behavior is understood.
7. Add or update the component manifest, minimal sample contract, tests, and
   reproducibility record in the same change.
8. Advance status only after its evidence gate passes. Keep source audit,
   original-repository execution, adaptation, and platform acceptance distinct.
9. Commit one coherent milestone at a time so each imported or integrated
   component has a reviewable history.

## Use the repository Git workflow

For local Git inspection and commits, avoid loading the Windows PowerShell
login profile. For GitHub network operations, follow
`references/git-workflow.md` exactly so the known `git-remote-https.exe`
application-error dialog is not triggered by repeated sandboxed attempts.

## Enforce common contracts

Represent each run with a request containing at least:

- run identifier, task, model, and action;
- input manifest and task parameters;
- checkpoint reference, device, and random seed.

Return a result containing at least:

- status, predictions, and metrics;
- artifact paths and timing;
- environment details, warnings, and errors.

Record each component's source revision, licenses, environment identifier,
entry point, checkpoint URL and hash when applicable, sample input, and last
acceptance date. Do not require every component to have a checkpoint, but do
not call checkpoint-dependent inference runnable without one.

## Verify proportionally

Run focused tests first, then broader regression checks. Cover these layers as
the component matures:

- before dependency installation or GPU inference, assess the target remote
  GPU and VRAM, available system memory and disk, driver/runtime support,
  expected checkpoint size, input resolution, batch size, compiled-operator
  needs, and likely peak VRAM; record the feasibility basis and safeguards;
- structure and metadata conversion;
- imports, forward computation, and checkpoint save/restore;
- one minimal valid task input and its predictions, metrics, and artifacts;
- Python, PyTorch, CUDA, GPU, dependency, and source revision capture;
- clean-environment regression of the first-release demonstration;
- actionable failures for missing weights, credentials, GPU, or input fields.

Keep generated data, checkpoints, caches, and run artifacts out of Git unless
the user explicitly approves redistribution and the license permits it.

Maintain a version-controlled model-to-environment map. Prefer the two declared
shared remote environments, record every compatible dependency added to them,
and name a separate environment only when a verified conflict requires one.
Each mapping
must state the environment path or reproducible specification, Python,
PyTorch, CUDA runtime, supported components, checkpoint location policy, and
verification status.

## Report evidence honestly

Separate these categories in code comments, manifests, documents, and status
reports:

- upstream paper result;
- source-level static audit;
- original repository smoke result;
- adapted component result;
- platform acceptance result.

Never infer runtime success from the existence of source code, a README, or a
checkpoint link. Never turn a simulated competition draft into an implementation
claim. Prefer a smaller set of verified demonstrations over many unverified
entries.

## Respect current scope

Prioritize the DHG foundation and a small number of visual, event, multi-view,
and 3D integrations. Treat large-model/RAG and attack/defense directions as
out of the current implementation scope unless the user explicitly reopens
them.
