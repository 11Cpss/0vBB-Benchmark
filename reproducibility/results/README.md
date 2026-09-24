# Final result inputs

These files describe the computations reported in the paper:

- `event_inputs.json`: 61 event-result records with relative input paths,
  input hashes, explicit evaluation profiles and full-precision expected I,
  matched AUC, inclusive AUC and sample count. Includes appendix comparisons
  and the independent SuperNEMO illustration.
- `transformer_inputs.json`: the 12 NEXT/SuperNEMO Transformer prediction sets,
  their evaluation configurations and expected metrics.
- `transformer_models.json`: selected training configurations and checkpoint
  hashes for the Transformer launchers; missing run sources are explicit.

The final display inputs are under
[`paper/wing_contribution/figures/data/`](../paper/wing_contribution/figures/data/).
In particular, `main_table_results.json` contains the 52 final dataset/model
records, and `unified_results.json` contains the extended evaluated results.
The table renderer reads these final values directly.

Raw detector data, checkpoint tensors and the bulk per-event companion are
external. Paths in the event manifests are relative to the `--data-root`
provided to the evaluator. See the [reproduction guide](../README.md) for
commands and availability limits.
