# Training the alternative NEXT models with tmux

The alternative models are full training programs; they do not expose or run a
smoke-training mode.  A single GPU should run one model at a time.

## Start the sequential queue

The queue script trains every alternative model in a fixed order, starting with
the lighter baselines and leaving dense 3D for last. Start it in one detached
tmux window:

```bash
cd /home/wenyu/summer

tmux new-session -d \
  -s next-alt-models \
  -n training \
  "cd /home/wenyu/summer && source .venv/bin/activate && bash 01_code/architectures/run_alternative_training_queue.sh"
```

The queue stops on the first failed program. A successful model leaves both a
best and last checkpoint before the next model begins. Existing artifacts are
preserved by refusing to start that model when `allow_overwrite: false`; the
runner never silently invents a continuation or mixes two runs.

## Inspect progress

```bash
tmux attach-session -t next-alt-models
```

Detach without stopping training by pressing `Ctrl-b`, then `d`.

From another shell, list the session and inspect the latest persisted metrics:

```bash
tmux list-sessions
tr '\r' '\n' < 03_training_runs/logs/alternative_training_queue_<timestamp>.log \
  | tail -n 20
```

Architecture-specific epoch CSV files under `03_training_runs/logs/` remain
the authoritative progress records.

## Stop deliberately

Inside the attached window, press `Ctrl-c` once and allow the current Python
process to exit.  To terminate the entire session from another shell:

```bash
tmux kill-session -t next-alt-models
```

Killing a session does not delete checkpoints or logs already written.

## Train one model instead

```bash
tmux new-session -d \
  -s next-particle-net \
  -n training \
  "cd /home/wenyu/summer && source .venv/bin/activate && python 01_code/architectures/gnn_002_particlenet_edgeconv/train_classification.py"
```

After training, use the existing `energybench next` command documented in
[`USAGE_GUIDE.md`](USAGE_GUIDE.md) and the model's README.
