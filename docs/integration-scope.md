# Integration Scope

## What “integrated” means

HyperComp does not require every upstream project to share one data pipeline or one CUDA environment. Integration is defined at three levels:

1. **Registration:** source, paper, license, environment, checkpoint, inputs, outputs, and entry point are recorded.
2. **Runtime adaptation:** the component accepts a HyperComp request and returns a HyperComp result while preserving its own internal preprocessing when necessary.
3. **Platform verification:** a fixed sample produces recorded logs, outputs, environment metadata, and reproducible status evidence.

## Shared platform boundary

The platform standardizes:

- component discovery and task registration;
- model and checkpoint selection;
- input and output path exchange;
- run identifiers and status reporting;
- metrics, logs, artifacts, and error summaries.

The platform does not initially standardize:

- raw dataset preprocessing across unrelated modalities;
- training loops from every upstream repository;
- a single Python, PyTorch, or CUDA environment for all plugins;
- redistribution of datasets or checkpoints;
- automatic synchronization with the original repositories.

DHG is the native foundation of this repository and lives directly under
`dhg/`. It retains its Apache-2.0 license and upstream provenance, but contains
no nested `.git` directory and has no automatic connection to the original
repository. Future shared hypergraph operators and models should extend this
foundation directly. The `third_party/` area is reserved only for task projects
that genuinely need an isolated runtime or cannot yet be adapted natively.

## Initial priority

| Priority | Component | Integration mode | Current status |
| --- | --- | --- | --- |
| P0 | DeepHypergraph | Native foundation | Registered |
| P0 | HGNN/HGNN+ | Native task | Adapted; local CPU/GPU smoke passed |
| P0 | Hyper-YOLO | Isolated plugin | Source audited; adapter staged; not run |
| P0 | YOLOv13 | Isolated plugin | Adapted; upstream and unified-entry single-image GPU runs passed |
| P1 | SoftHGNN classification | Isolated plugin | Source audited; prediction adapter staged; not run |
| P1 | SuperFast | Isolated plugin | Source audited; adapter staged; not run |

## Code-first external milestone

Hyper-YOLO, YOLOv13, SoftHGNN, and SuperFast are selectable through the common
task and model registry while retaining task-specific preprocessing and
isolated Python environments. Hyper-YOLO and YOLOv13 share only the platform's
Ultralytics-family detection worker. YOLOv13 has crossed the runtime-adaptation
gate with a fixed single-image GPU run; the other three external components
remain code-first `source_audited` integrations until matching runtime evidence
is collected.
