# Event provenance and input standardization

The runnable, portable final evaluator is `../../reproduction/evaluate.py`. The optional `../../event_data/` companion already supplies exactly the standardized score/label/physical-energy/event-ID/group/weight arrays used by the audited results. It can be evaluated without the original account tree.

`reference_snapshots/` contains byte-identical extraction and audit programs actually used to recover these inputs. They are provenance snapshots, **not portable entry points**: several use the original account's detector paths, original checkpoints, original audit-tree layout, or frozen intermediate evaluation versions. Do not execute them as a reproduction shortcut. In particular `collect_results.py` historically writes into its configured paper tree. The recommended commands never call those programs. No original or historical incorrect code was changed.

## Required semantics when exporting a new checkpoint

- NEXT: positive = 0nubb, negative = Bi214; retain the native scalar logit. The conditioning energy is physical summed hit energy in MeV, explicitly converted to keV. Keep the exact campaign split and event IDs. The 115,499-event PointMamba run is a separate referenced population from the main 116,549-event NEXT run.
- MJD: positive = clean, requiring all four reference PSD flags. Binary models/Transformers provide a clean-oriented logit. Classic GINE uses its archived four-PSD-output aggregation; do not replace it with one channel. Recover original float64 `energy_label` from the official Test shards in verified loader order. Cached clipped energies cannot support strict range exclusions. The extraction snapshots verify labels, cached energies, and physical shard/row/ID identities before attaching original energies.
- EXO-200: stored class 0 (one charge cluster) is evaluation-positive, so standardized label is `1 - stored_label` and score is the negative native background-oriented logit. Conditioning energy is `Rotated_energy` in keV. Preserve run-based splits and original event IDs.
- SuperNEMO: the trained task is 2nu vs Bi214; the separate illustration is 0nu vs Bi214 with score equal to physical calorimeter energy E1+E2. These tasks and inputs must never be interchanged.

Do not add a sigmoid, clip physical energy, replace the checkpoint/test split, or reconstruct events from aggregate metrics. Where inference exports lack event IDs, the archived loader-order and raw-metadata checks must be repeated; matching array length alone is insufficient. Training/inference launchers and per-run configurations are documented in `../classic/README.md` and `../transformers/README.md`.
