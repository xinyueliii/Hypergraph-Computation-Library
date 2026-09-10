# Component Manifests

Every external model or platform component must have one JSON manifest. A
manifest records provenance and integration status. Isolated upstream source
is downloaded into the ignored `third_party/` runtime area by a pinned fetch
script, rather than being committed as native HyperComp code.

Required fields:

- `id`, `name`, `task`, and `integration_mode`;
- source and paper URLs;
- checkpoint availability;
- upstream environment notes;
- adapter command or native entry point;
- evidence-based status.

External repositories belong in the ignored `third_party/` directory. Their license and pinned commit must be checked before integration work begins.
