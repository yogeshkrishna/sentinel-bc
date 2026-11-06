# Architecture

```mermaid
flowchart LR
    A[Bundled WDBC or validated labeled CSV] --> B[Disjoint stratified partitions]
    B --> C[Fold-local classical and Keras training]
    C --> D[Development selection]
    D --> E[Held-out sigmoid calibration]
    E --> F[Independent MAPIE conformalization]
    F --> G[Development safety policy]
    G --> H[Versioned hashed artifact bundle]
    H --> I[One locked test evaluation]
    H --> J[Offline inference engine]
    K[30 numerical measurements] --> J
    J --> L[Validity and OOD screens]
    J --> M[Calibrated probability and uncertainty]
    L --> N[Safety gate]
    M --> N
    N --> O[Gradio console]
    O --> P[SQLite human review and audit]
```

`train.py` and Gradio start the same local worker through `runs.py`.
`training.py` coordinates data roles;
`models.py` owns fresh Keras weights and classical candidates. `calibration.py`,
`conformal.py`, `uncertainty.py` and `safety_gate.py` provide small independent
interfaces. `reporting.py` performs one locked evaluation and writes cached
predictions and figures. `evaluate.py` reads that cache.

`Engine` validates the manifest, runtime versions, hashes, schema, required
components and class order before loading a trusted local bundle. It validates
input columns, reorders them canonically, and keeps invalid rows away from the
models. A lock serializes access to Keras inference. Single and batch inference
share the same path. Batch limits are 100 rows and 1 MiB.

`storage.py` preserves original output separately from human annotations.
SQLite transactions commit a review and its audit entry atomically. SQL triggers
prevent ordinary updates to original predictions and audit history. Exports omit
free text and measurements. The UI binds only to 127.0.0.1 by default and never
opens a public tunnel. It has no deployment authentication or clinical record
capabilities.

`configs/default.yaml` supplies defaults. Every run saves its effective settings,
provenance, fingerprint, status and version under an isolated registry directory.
A project lock allows one worker at a time. The worker records phases and checks
for cancellation at model and epoch boundaries. Activation requires completed
outputs, verified hashes and a matching model and evaluation version. The
historical v1.0.0 bundle and its evaluation occupy separate directories.
Runtime files, uploaded data and review databases stay out of Git.
