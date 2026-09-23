# CNN-001 waveform baseline

The two entry points instantiate the same `WaveformCNNBackbone`:

- `train_classification.py`: four raw logits, BCE-with-logits;
- `train_regression.py`: one scalar energy prediction in keV, MSE.

The models are trained independently. No weights, optimizer state, checkpoint,
or test result is shared across tasks. Classification amplitude-normalizes each
baseline-subtracted waveform by default so total pulse height cannot become an
easy PSD shortcut. Regression preserves pulse amplitude because it carries the
energy information.

