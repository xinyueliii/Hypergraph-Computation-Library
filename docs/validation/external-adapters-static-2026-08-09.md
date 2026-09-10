# Hyper-YOLO and SuperFast static integration record

Date: 2026-08-09

Scope: code-first integration only. This record does not contain environment,
checkpoint, sample-input, import, inference, accuracy, or performance evidence.

## Source revisions

- Hyper-YOLO: local ignored checkout at
  `9bfdabd8b97b5ee5da04e5df30d140a9d15557c5` (`v0.1`).
- SuperFast: local ignored checkout at
  `b4d29efc75c4a6ca07de4c5f56f555de48b65741`.

Both revisions were downloaded with `scripts/fetch_components.ps1` and checked
with `git rev-parse HEAD`.

## Checks completed

- Python syntax: `python -m compileall -q hypercomp` passed.
- Component and task JSON files parsed successfully.
- `scripts/fetch_components.ps1` passed the PowerShell parser.
- The staged Git diff passed whitespace/error checking.
- Worker calls were compared statically with the pinned upstream source APIs.

## Deliberately deferred

- dependency installation and isolated-environment creation;
- checkpoint and sample-data download;
- upstream smoke tests and HyperComp end-to-end runs;
- numerical output, GPU-memory, timing, and reproducibility validation.

The two component manifests must remain `source_audited` until later evidence
supports a status change.
