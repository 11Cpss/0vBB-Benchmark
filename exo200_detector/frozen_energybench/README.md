# Frozen EnergyBench snapshot

This directory contains the exact EnergyBench Python package used to evaluate
the EXO-200 Transformer runs. It was copied from the collaborator-maintained
benchmark on 2026-08-31 with permission to publish. Keeping this snapshot beside
the adapter prevents later changes to another checkout from silently changing
reported metrics.

The public package is under `energybench/` and reports version `0.1.0`. The
complete source fingerprint and per-file SHA-256 hashes are recorded in
`SOURCE.json`. Do not edit this snapshot in place. If the shared evaluator
changes, add a new versioned snapshot and report the resulting protocol and code
fingerprints as a new benchmark version.

The EXO adapter imports this directory explicitly. Because the repository also
contains an older NEXT-facing package named `energybench`, start the EXO
energy-aware notebook in a fresh Python kernel so a different package is not
already present in `sys.modules`.

This directory contains evaluation software only. It does not contain detector
data, predictions, checkpoints, or benchmark results.
