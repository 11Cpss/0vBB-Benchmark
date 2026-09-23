# GitHub publication snapshot

This branch publishes the code-only EnergyBench reproducibility bundle
prepared on 23 September 2026. The existing detector workflows are retained;
the complete paper reproduction snapshot is under this directory.

The source archive SHA-256 is
`235420ea3dfca9e4a7d62f285a6b4878055959b843c453146a4142331118792b`.
All 974 files from that archive are included unchanged. This publication note
and the local `.gitignore` are Git-distribution additions. The repository
homepage links the benchmark implementation and the English instructions.
The snapshot README's statement about the original local packaging operation
describes that earlier operation; this branch is the subsequent Git publication.

## What can be reproduced from this checkout

From this directory, install `environments/requirements-core.txt` in a Python
3.11 virtual environment as described in [README.md](README.md). Then run:

```bash
.venv-reproduce/bin/python -B reproduction/verify_bundle.py
.venv-reproduce/bin/python -B benchmark/tests/test_metrics.py
.venv-reproduce/bin/python -B reproduction/reproduce.py render --output outputs/render
```

These commands verify the code-only snapshot, run the benchmark's 14 numerical
checks, and regenerate the available final tables and figures. Manuscript
compilation additionally requires a TeX engine; model training and checkpoint
inference require the external inputs and environments documented in the model
READMEs. No training or inference is started by the commands above.

## Event-level replay and external inputs

The separate `EnergyBench_event_data_20260923.zip` companion was deliberately
left out of this Git branch. Its SHA-256 is
`60482d03e5f424379b92f8fa1093801f364920db413aae12d182bee2b1f6abea`.
It is required to replay the 61 audited records and 12 historical Transformer
records. No public download URL is supplied by this publication. If that
companion is obtained separately, place its `event_data/` directory directly
under `reproducibility/`, then use the benchmark commands in the README.

The source archive includes selected waveform snippets and derived geometry
descriptors needed by the figures, along with a synthetic numerical test NPZ.
It does not include the bulk event-prediction companion, full raw detector
datasets, or trained checkpoint tensors. Original machine paths in provenance
records document the source audit; portable evaluation and rendering commands
use paths within this checkout.

Missing original sources and the two pre-existing unresolved manuscript
references remain documented in the snapshot README. This publication does
not claim to resolve those gaps or grant additional rights to third-party
material. Existing source notices and licenses are retained.
