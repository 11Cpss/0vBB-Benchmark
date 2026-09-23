# MJD / EXO-200 strict 600-bin reevaluation

Final scope: 23 configurations (8 classic, 9 EXO-200 Transformers, 6 MJD MLP/Fourier Transformers). NEXT, SuperNEMO and the three MJD RoPE results are retained unchanged.

The selected-results JSON contains only these 23 evaluations. The larger `results_full_precision.json` in the local audit retains 77 mixed-profile records so the full paper can be rendered without changing other datasets. `old_*` denotes the original historical evaluation; `previous_paper_*` and `delta_from_previous_paper_*` in the comparison CSV distinguish the immediately preceding manuscript values.

Input manifests identify original event archives and immutable hashes. Aggregates are display/audit outputs, never recomputation inputs. Raw NPZ, HDF5 and checkpoint files are external dependencies. `source_provenance.json` records historical traces, including historical scores; current results are in `selected_results_full_precision.json`.

See `../../README.md` in the evaluation directory for portable evaluation and rendering, and `EXECUTION_REPORT.md` for source and methodological limitations. The audit-workspace `reproduce.sh` runs the numerical checks, recomputes the selected 23 archived event sets, collects full precision, and regenerates tables without modifying the retained figures.

Final independent checks: `final_independent_review.md`, `final_paper_review.json`, and `final_pdf_review.json`. The low-priority caption suggestion in `exo_text_crosscheck.md` was applied before final PDF review.
