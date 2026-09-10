# E-3DTrack and MeshNet asset retry

Date: 2026-08-12

## E-3DTrack

- The AutoDL server still timed out connecting to the official Google Drive
  checkpoint URL.
- The same URL worked from Windows after disabling curl's unavailable
  certificate-revocation check. Google Drive identified the file as
  `ckpt.pth` (133 MB).
- Downloaded file size: `138960387` bytes.
- SHA256: `862ff7db5e31beb93716c325b91c0120c2f61825ebfbaef28f951f053fb29cb2`.
- External server path:
  `/root/autodl-tmp/hypercomp-data/checkpoints/e-3dtrack/ckpt.pth`.
- Server and Windows hashes matched.
- In shared `event-cu113`, `torch.load` returned a `state_dict`, and
  `TrackerNetEval(feature_dim=384, hgnn=True)` accepted every key with no
  missing or unexpected entries.
- The official dataset archive was downloaded on Windows with eight parallel
  HTTP range requests, transferred to the server, and passed `unzip -tq`.
- Dataset archive size: `3524467933` bytes.
- Dataset SHA256:
  `e729e69c6e868edf1f738a2b5302de9054488d692f41cfd48668fb6d5bdbd4e0`.
- External server path:
  `/root/autodl-tmp/hypercomp-data/datasets/e-3dtrack/E-3DTrack.zip`.
- The first entry in the official test list, `airplane_1/20`, was extracted as
  a minimal evaluation input.
- HyperComp unified-entry result:
  - run ID: `e-3dtrack-20260812-105115`;
  - status: `succeeded`;
  - worker inference: `9.097064673900604` seconds;
  - platform total: `15.426771499216557` seconds;
  - prediction and ground-truth shapes: `(167, 7, 3)`;
  - both arrays contain only finite values;
  - artifacts: `pos_3d_pred.npy` and `pos_3d_gt.npy` under
    `/root/autodl-tmp/hypercomp-data/runs/e-3dtrack-minimal-20260812/`.
- A supplementary pipeline-health calculation produced MAE
  `297.62079277654294`, RMSE `408.60667589648205`, and mean Euclidean error
  `690.6718841354377`. These are raw-array diagnostics, not the paper's
  benchmark metrics and should not be compared with published results.

Conclusion: the assets were blocked by the AutoDL-to-Google route, not
deleted. Windows download plus SCP resolved the route problem, and E-3DTrack
now reaches `adapted` with one official checkpoint-backed test sequence. This
is not a complete-dataset benchmark or `platform_verified` acceptance run.

## MeshNet

- All three original README Tsinghua Cloud links returned HTTP 200 HTML pages
  whose body states `Link does not exist`:
  - pretrained checkpoint;
  - processed ModelNet40;
  - Manifold40.
- Official GitHub issue 27 contains a maintainer-provided replacement Google
  Drive checkpoint link. The link downloaded a valid PyTorch zip-format state
  dictionary.
- Replacement checkpoint size: `17094421` bytes.
- SHA256: `cff4252d94057d44c33cbab7e549756f7ace1330309ca17b19a55a62b56f4613`.
- External server path:
  `/root/autodl-tmp/hypercomp-data/checkpoints/meshnet-pretrained.pkl`.
- Official issue 26 contains a community replacement data source from the
  Tsinghua SubdivNet site. `Manifold40.zip` remained directly downloadable.
- Archive size: `132455392` bytes.
- SHA256: `fba848d46a6650c6b182d47a81dd200167e98986203efff2c658a8fa54eea952`.
- External server path:
  `/root/autodl-tmp/hypercomp-data/datasets/meshnet/Manifold40.zip`.
- Shared `event-cu113` gained `pymeshlab==2021.10` and `rich==13.9.4`;
  `pip check` reported no broken requirements.
- The upstream preprocessor converted
  `Manifold40/piano/test/piano_0303.obj` into
  `ModelNet40_processed/piano/test/piano_0303.npz`.
- HyperComp unified-entry result:
  - status: `succeeded`;
  - predicted class index: `6`;
  - worker inference: `0.04018247127532959` seconds;
  - platform total: `5.688501417636871` seconds;
  - 256-dimensional retrieval feature saved under
    `/root/autodl-tmp/hypercomp-data/runs/meshnet-retry-20260812/`.

Conclusion: the README links are truly deleted, but the official issue history
provides a usable maintainer checkpoint and a compatible data source. MeshNet
now reaches `adapted` for one checkpoint-backed unified-entry sample. This is
not a full benchmark reproduction.
