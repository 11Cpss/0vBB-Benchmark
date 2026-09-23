# MJD Transformer

This package adds a waveform Transformer without modifying the shared
`mjdbench` data split, preprocessing, training, checkpoint, or evaluation
code. The shared loader supplies the same baseline-subtracted waveform to the
CNN and Transformer. The Transformer tokenizes that dense batch internally.

## Tokenization strategies

The default settings produce 500 tokens from each 10,000-sample waveform, so
the three representations have comparable sequence length and attention cost:

- `raw_patches`: 500 non-overlapping 20-sample patches; the 20 ordered raw
  amplitudes are the token content;
- `segment_summary`: 500 temporal regions summarized by mean and RMS
  amplitude;
- `pulse_entities`: 250 uniformly spaced time points plus 250 points selected
  by normalized amplitude and local change. Tokens contain raw amplitude and
  local RMS, and selection never uses labels or target energy.

Every strategy returns three coordinates: normalized time, normalized
amplitude/mean, and local/first difference. Every generated token is valid, so
the current masks are all true; the common mask interface supports future
variable-length representations.

The representation supports two positional encodings:

- `coordinate_mlp`;
- `fourier_coordinates`.

The same backbone supports the shared clean/non-clean classification task and
clean-event energy regression task.

## Shared-workflow usage

Run from a directory containing both `mjdbench/` and `mjd_transformer/`:

```python
from mjdbench import DataConfig, TrainingConfig, prepare_dataset
from mjdbench import evaluate_model, train_model
from mjd_transformer import MJDTransformer, TokenizationConfig

data_config = DataConfig(seed=42, validation_fraction=0.10)
training_config = TrainingConfig(seed=42)

data = prepare_dataset(
    task="classification",
    data_config=data_config,
    batch_size=training_config.batch_size,
    num_workers=training_config.num_workers,
)

model = MJDTransformer(
    task="classification",
    tokenization_config=TokenizationConfig(
        tokenization="segment_summary",
        token_count=500,
    ),
    position_encoding="coordinate_mlp",
)

train_model(
    model,
    data.train_loader,
    data.validation_loader,
    task="classification",
    config=training_config,
    output_dir="outputs/transformer_segment_coordinate",
)

metrics = evaluate_model(
    model,
    data.test_loader,
    task="classification",
    device=training_config.device,
    output_dir="outputs/transformer_segment_coordinate",
)
```

For regression, change both `task` arguments to `"regression"`. The shared
loader will then retain only clean events before constructing the seeded
development split, matching the CNN workflow.

## Design boundary

- Keep `mjdbench/config.py`, `data.py`, and `training.py` unchanged.
- Keep the partner CNN in `mjdbench/models.py` unchanged.
- Add future tokenizers through `TokenizationConfig` and `build_tokenizer`.
- Add future positional encodings through `build_position_encoder`.
- Use a notebook only for experiment configuration and orchestration; call
  the shared `train_model` and `evaluate_model` functions directly.

The coordinate vectors include waveform-derived amplitude and slope values,
so this first representation is a learned waveform-geometry encoding rather
than pure time-only positional encoding. Hold token content constant when
comparing its two positional encoders.
