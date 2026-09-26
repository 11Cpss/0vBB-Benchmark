# Transformer environment

Use Python 3.11 and an isolated environment, created from the `reproducibility/` directory:

```bash
python3.11 -m venv .venv-transformers
.venv-transformers/bin/python -m pip install -r environments/transformers/requirements.txt
```

Choose a PyTorch wheel compatible with the target CPU/GPU. These runtime bounds are not a reconstruction of every original training environment. Training uses the exact hyperparameters in `code/transformers/configs/`; hardware and nondeterministic operations can affect retraining results.

Metric-only reproduction uses `environments/requirements-core.txt` and does not need PyTorch. See `code/transformers/README.md` for the training, inference and metric commands.
