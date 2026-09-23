# 23-model Simple EnergyBench workflow

All maintained architecture entry points now use the standalone workflow in
`evalutaions_workflow`.

## Formal contract

- The complete NEXT inventory is split by event count with seed 42 and
  stratified 80/10/10 train/validation/test targets.
- PyTorch jobs use `TrainingConfig()` unchanged: effective batch 64, up to 50
  epochs, AdamW at `5e-4`, weight decay `1e-4`, cosine scheduling, gradient
  clipping at 1.0, and early stopping after 5 non-improving epochs.
- Architecture-specific microbatches limit device memory without changing the
  effective loss batch of 64.
- The best validation-AUC classifier (or validation-RMSE regressor) is restored
  before the held-out test evaluation.
- Classification and regression use `EvaluationConfig()` and the canonical
  0--3000 keV grid in exact 5 keV bins.
- Existing artifacts are never overwritten. Every campaign owns a new run ID.

The 3-D, point, graph, sequence, topology, sparse, and hybrid models keep their
native representations. Their custom loaders consume the same event-count
manifest and expose the same outer EnergyBench batch metadata. No model
reconstructs 3-D input from the three 2-D projections.

## One model

```bash
/home/wenyu/summer/.venv/bin/python \
  01_code/architectures/gnn_003_egnn/train_classification.py \
  --output-dir 03_training_runs/energybench_manual/gnn_003_egnn \
  --data /home/klz/Data/zeronu_benchmark/NEXT \
  --manifest 03_training_runs/energybench_manual/event_split.json
```

CNN-001 through CNN-003 also expose `train_energy_regression.py`. Regression
predictions are physical MeV.

## Full parallel campaign

```bash
/home/wenyu/summer/.venv/bin/python \
  01_code/architectures/run_energybench_campaign.py \
  --run-id RUN_ID \
  --parallel-capacity 3 \
  --include-regression
```

The scheduler may run several light models together, while dense 3-D and
geometric graph jobs reserve the full GPU capacity. XGBoost remains CPU-only
and can overlap a GPU job. Resume an interrupted campaign with the same run ID
and `--resume`.

## Live tmux windows

For a new campaign, launch the actual supervisor inside tmux and create one
live window per task in a single command:

```bash
/home/wenyu/summer/.venv/bin/python \
  01_code/architectures/energybench_campaign_tmux.py RUN_ID \
  --launch \
  --session energybench-RUN_ID \
  --parallel-capacity 3 \
  --include-regression

tmux attach -t energybench-RUN_ID
```

Without `--launch`, the same command adds viewer windows to an already-running
campaign.

The weighted concurrency can be changed while a campaign is running:

```bash
/home/wenyu/summer/.venv/bin/python \
  01_code/architectures/energybench_campaign_tmux.py RUN_ID \
  --update-capacity 10
```

`energybench_campaign_autorecover.py` can run in an additional tmux window. It
waits for the active supervisor, archives incomplete failed-task outputs, and
retries only failed jobs while preserving completed tasks and the split.

The session contains a dashboard, the campaign scheduler log, and one fixed
window per classification or regression task. Use `Ctrl-b w` for the window
list, `Ctrl-b n`/`Ctrl-b p` to move between windows, and `Ctrl-b d` to detach.
Closing these viewer windows does not stop training. For pending jobs, output
from an older failed attempt is hidden until the scheduler starts a new one.

## Outputs

Each task directory contains:

```text
training/
  best_model.pt
  last_model.pt
  history.json
  training_history.png
evaluation/
  metrics.json
  results.csv
  predictions.npz
  ...standard task plots...
run_summary.json
```

The campaign root also retains `event_split.json`, `manifest.json`, one log per
job, and `campaign.log`.
