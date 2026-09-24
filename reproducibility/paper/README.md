# Manuscript source

`iclr2027_conference.tex` is the entry point. This directory contains the active
manuscript include tree, bibliography/style files, the included figure assets,
and the code/data needed to render the available numerical tables and figures.
Superseded drafts and internal review records are excluded.

Build a copy without writing generated files into this source directory:

```bash
python reproduction/reproduce.py compile --engine latexmk --output outputs/compiled
```

Run this command from the parent `reproducibility/` directory. The manuscript text
and final numerical values are preserved. Two unresolved references (`app` and
`fig:energybench-auc`) are present in the supplied manuscript. Eight detector-example
PDFs lack their original generators, as documented in `code/figure_sources/README.md`.
