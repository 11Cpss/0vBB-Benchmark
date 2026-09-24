# Classic-model environment

Use a separate Python 3.11 environment for model training and checkpoint inference.
From the `reproducibility/` directory:

```bash
python3.11 -m venv .venv-classic
source .venv-classic/bin/activate
python -m pip install -r environments/classic/MJD/requirements.txt
python -m pip install -r environments/classic/EXO200/requirements.txt
python -m pip install -r environments/classic/SuperNEMO/requirements.txt
python -m pip install -r environments/classic/NEXT/nontransformer.txt
python -m pip install -e code/classic/NEXT
```

Choose a PyTorch wheel compatible with the available accelerator. The optional
`NEXT/next-cnn-cu128.txt` specifies the CUDA 12.8 installation used for the release
checks. Core tested versions: Python 3.11.15, torch 2.11.0+cu128, NumPy 2.4.6,
h5py 3.16.0, XGBoost 3.0.5. Requirements containing lower bounds do not pin the
complete training environment. Match each effective model configuration's AMP
and device settings when comparing externally supplied checkpoint predictions.

For final metric recomputation alone, use the lighter environment in
[`../requirements-core.txt`](../requirements-core.txt) and the central
[`../../benchmark/README.md`](../../benchmark/README.md). The NEXT installation
provides models and training support; it does not install a second EnergyBench CLI.
