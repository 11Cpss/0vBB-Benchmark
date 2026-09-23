# Transformer environments

Use a dedicated virtual environment. `requirements.txt` collects the dependencies needed for the copied classification code and notebook syntax tooling; its bounds are not an original lockfile. Select an appropriate PyTorch CPU/CUDA wheel for the target machine. `source_pyproject.toml`, `source_next_requirements.txt`, and `source_published_supernemo_requirements.txt` are unmodified original declarations.

`validation_environment.json` records the actual Python and installed-package versions used for syntax checks, isolated imports/model construction, and all twelve historical NEXT/SuperNEMO metric replays. Training and model inference were not executed while preparing this archive. No global environment was modified. Distinct historical packages both use the import name `energybench`; isolate them in separate processes as `code/transformers/replay_historical_transformers.py` does.
