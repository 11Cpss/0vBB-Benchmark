# Classic-model environments

Create and activate a separate Python 3.11 environment before running training or checkpoint inference. The source snapshots do not include a virtual environment.

```bash
python3.11 -m venv .venv-classic
source .venv-classic/bin/activate
python -m pip install -r environments/classic/MJD/requirements.txt
python -m pip install -r environments/classic/EXO200/requirements.txt
python -m pip install -r environments/classic/SuperNEMO/requirements.txt
python -m pip install -r environments/classic/NEXT/nontransformer.txt
python -m pip install -e code/classic/NEXT
```

These are the original dependency declarations. The NEXT project additionally includes its original `uv.lock`, Python version declaration and `requirements/next-cnn-cu128.txt`; the latter explicitly selects the PyTorch CUDA 12.8 wheel index and torch 2.11.0. Use an installation appropriate for the available hardware. Original requirements containing lower bounds are not a claim of a fully pinned historical environment. Read the exact copied lockfile and run metadata when reproducing training precision and selected checkpoints.

`validation_environment.json` records the existing environment actually used for this bundle's source-only smoke checks: Python 3.11.15, torch 2.11.0+cu128, NumPy 2.4.6, h5py 3.16.0 and XGBoost 3.0.5, among other dependencies. It is an observation, not a new dependency resolution or a claim that every historical experiment used the same installed versions. No installation or training was performed while collecting these snapshots.

The old package's `energybench` CLI provides historical inference/evaluation utilities. Final paper metric reproduction uses `reproduction/evaluate.py` at the bundle root, with the profile assigned in the event-input manifest. This separation is necessary even when both commands run in the same environment.
