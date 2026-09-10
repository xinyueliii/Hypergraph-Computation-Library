# DeepHypergraph upstream snapshot

The root `dhg/` package is based on this imported source snapshot and is
managed directly by Hypergraph-Computation-Library.

- Upstream project: DeepHypergraph (DHG)
- Upstream repository: https://github.com/iMoonLab/DeepHypergraph
- Imported version: v0.9.7
- Upstream commit: `bb88281a8fdf7333c46312691027ac80c2b282ef`
- Imported on: 2026-08-09
- License: Apache-2.0 (retained in `LICENSE`)
- Local Git relationship: none

## Import verification

- Nested `.git` directories: 0
- Python source compilation: passed
- HyperComp registry tests: passed
- DHG runtime smoke test: pending creation of the DHG dependency environment
  (`optuna` is not installed in the current platform environment)

The upstream repository's `.git` directory was intentionally removed after
download. The code was promoted into the root `dhg/` package because DHG is
the computation foundation rather than a third-party plugin. It is not a
submodule or subtree. Future local changes and integration adapters are
recorded only in Hypergraph-Computation-Library.

To update this snapshot later, download a reviewed upstream release into a
temporary directory, compare it with the root `dhg/` package, and import the selected
changes in a dedicated parent-repository commit. Do not restore a nested
`.git` directory.
