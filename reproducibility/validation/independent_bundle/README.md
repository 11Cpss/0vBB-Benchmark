# Independent package portability and preservation review

The runtime review copied the package's manuscript, code, environments, reproduction entry points, provenance and results into a randomly named directory under `/tmp`. Only one standardized event archive was needed for the two-profile CLI test. Every command ran from `/tmp`, outside the package root.

A Python audit hook rejected reads under `/home` except the installed scientific Python interpreter and its packages. A deliberate attempt to open the original manuscript failed, demonstrating that the guard was active. The relocated entry points nevertheless completed successfully. This verifies that replay uses bundle-relative sources and plotting data rather than original account-specific code/data paths. The existing Python installation is an explicit environment dependency, not a hidden experiment-data dependency.

## Executed results

- `reproduction/reproduce.py render`: all six generated numerical tables are byte-identical to the current paper snapshot after the three documented presentation substitutions. All six active Wing figure PDFs are byte-identical. No table metric or source paper was modified.
- `reproduction/reproduce.py check`: numerical unit/boundary/tie/sparsity/reporting tests pass in the relocated copy. The final relocated check also passes integrity verification for all 805 non-event static files, with a normal newly created virtual environment present.
- `reproduction/evaluate.py --input ... --profile strict600` and `--profile overflow601`: both complete on the same copied EXO-200 CNN archive. Input SHA256, all 140,383 original events, and inclusive AUC 0.8949408988858205 agree. Strict600 excludes 1,426 above-range events; overflow601 retains them in index 600. The protocol fingerprints and distinct resulting metrics are correct.
- `reproduction/reproduce.py compile`: succeeds with a copied Tectonic executable, relocated prepared TeX cache, and `--only-cached`. The compiled copy has 25 pages and no overfull boxes. It retains exactly two existing unresolved reference names, `app` and `fig:energybench-auc`; both lack labels in the frozen active include tree. These source issues were preserved as requested and are disclosed in the root README.
- Immutable-source checks: all 755 copied source files match both their declared SHA256 values and their original files. All 73 event files (61 profile-registry and 12 historical-native archives) match their input hashes; the added historical files also match their original sources. No symlinks were found in the published package tree. Original Overleaf Git HEAD and uncommitted status remain identical to the initial snapshot. The complete final assembly check covers 879 manifest-listed files including the optional event companion.

`portability_review.json`, `source_integrity.json`, copied command logs, and the two single-archive metric JSONs contain the evidence. Successful replay does not claim retraining or checkpoint inference.

## Issue found and resolved in the new package

The first integrity verifier rejected every symlink anywhere below the package root. A normal virtual environment created by the documented quick-start contains Python symlinks, so this would have rejected a valid installation. The maintainer changed the new verifier to apply immutable-file checks only to the assembly manifest. A fresh `/tmp` virtual environment now coexists with a passing verifier. `venv_coexistence_regression.json` records that focused regression test. No historical source implementation was changed.

A second review finding was that the new classic/Transformer validation helpers wrote rerun environment metadata into manifest-listed `environments/` snapshots. Both new helpers now write those observations under mutable `validation/`, while preserving the frozen original validation-environment records. The final source review confirms no remaining immutable-output writes in these helpers; the maintainer separately reran their process checks.

## Explicit limits

Nine reported RoPE configurations on NEXT, MJD and SuperNEMO lack recovered original implementations/checkpoints. Eight dataset-description panels lack recovered original figure generators and exact selected-event records. Their supplied values/assets are retained with these limitations stated; another detector's implementation or replacement event was not substituted. Raw detector datasets and checkpoints remain external. The two unresolved manuscript references are original snapshot issues, not packaging repairs.

## Final assembly status

**PASS.** The final manifest SHA256 is `b9f8b31f02880496e861e3ef999787eddabe4a515ba3917e0c16998c9212c8b1`. All 879 listed files pass in the complete bundle, and all 805 non-event static files pass in the randomly relocated code-only tree. Its 14 numerical tests also pass with the original-account read guard enabled. Root/model/figure/input README commands agree with the delivered scope and disclose the known gaps. `assembly_review.json` records the final checks, and `portability_review.json` has `all_pass: true`. No unresolved entry-point issue remains. The documented provenance gaps and original manuscript references remain limitations, not silently repaired results.
