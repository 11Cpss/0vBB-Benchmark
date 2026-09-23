# Assembly and validation summary

Prepared 2026-09-23 in a new directory. Original manuscript, experiment code, historical/incorrect evaluators, checkpoints and detector files were not edited or moved. The original paper's uncommitted changes were preserved in the snapshot and patch.

- 879 immutable bundle-file hashes recorded before archive assembly, including optional event inputs.
- 755 copied-source comparisons checked against originals by an independent reviewer; original paper Git HEAD and working-tree status preserved.
- 61 shared-profile event replays pass; 12 additional historical Transformer event replays pass. These cover 43 of 52 main-table dataset/model entries plus extended and illustrative records.
- All 73 event archives match their recorded input hashes.
- 14 numerical metric tests pass.
- All six generated numerical tables match the final manuscript after the three documented presentation-only substitutions.
- All six generated active Wing PDFs match the reference PDFs byte for byte in the validated environment. Extent histograms, medians and complete CDF reconstruction also pass.
- Classic source checks: 331 original copies, 153 Python AST checks, 40 CLI/import checks and four independent guarded model constructors pass. No training, forward inference or raw-event loading was performed.
- Transformer source checks: 201 original copies, 102 Python files, 81 notebook code cells, five imports/constructors and convenience CLI help/description checks pass. New checkpoint wrappers have not been tested on full raw-data inference.
- Relocated render/evaluation/compilation pass from a random temporary directory with original `/home` sources blocked. Standard virtual-environment symlinks do not invalidate the package integrity checker.
- Compilation succeeds to 25 pages. The original source's two unresolved references (`app`, `fig:energybench-auc`) remain documented and unchanged; no overfull boxes were found.

Outstanding provenance: nine NEXT/MJD/SuperNEMO RoPE entries lack recovered original code/checkpoints/predictions; eight supplied dataset-example PDFs lack recovered original generation/selection scripts. Raw training datasets and checkpoint weights remain external. The package does not mislabel those gaps as successful training reproduction.

Detailed evidence is in the adjacent JSON/log files, the independent-bundle review, and the original source/input manifests. Archives contain code plus compact validation evidence and a separate optional event-data companion. No GitHub push or external publication was performed.
