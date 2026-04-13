# Scalper

A multi-resolution transformer that predicts future Forex price highs from 5-minute and hourly OHLC data.

## What it does

Given the last 64 five-minute candles and the last 64 hourly candles for a currency pair, the model predicts the **highest price reached over the next 12 five-minute candles** (1 hour lookahead). Prices are normalised pointwise against a sliding window so the model generalises across instruments and market regimes.

The design allows "marrying" different resolution data: the hourly branch provides macro context while the 5-minute branch captures recent momentum. A third "partial hour" input gives the model access to the current incomplete hour as it forms.

## Architecture

```
5-min OHLC (64 × 4)  ──────► Transformer branch ─┐
Hourly OHLC (64 × 4) ──────► Transformer branch ─┤
Partial hour OHLC (1 × 4) ──► Embedding          ─┤─► Dense head ─► target_high
Minutes into hour (scalar) ──► Embedding          ─┤
Partial hour length (scalar) ► Embedding          ─┘
```

Custom components:
- `LearnablePositionalEncoding` — learned positional embeddings
- `StochasticGatedTransformerBlock` — gated transformer with stochastic depth
- `AddTypeEmbedding` — distinguishes 5-min vs hourly tokens
- `AttentionPooling` — soft aggregation over sequence dimension
- `WarmupCosineDecay` — learning rate schedule with linear warmup

## Normalisation

All prices are divided by the mean close of the preceding 144 candles (6 days of hourly data) before being fed to the model. This window-based pointwise normalisation means:
- The model sees dimensionless ratios, not raw prices
- The same model works across different instruments and price levels
- Labels (target highs) are computed in the same normalised space, then optionally converted back to real values for evaluation

## Data layout

```
data/
  final_data/          Raw OHLC CSVs per instrument (input)
  experimenting/       Preprocessed chunks for multi-instrument training
    {INSTRUMENT}/
      hour.csv
      training/
      validation/
      testing/
  regression_final/    Preprocessed data for instrument fine-tuning
```

Each instrument directory under `experimenting/` and `regression_final/` is produced by `regression_preprocess.py`.

## Quickstart

```bash
# Install dependencies
pip install tensorflow polars pandas numpy scikit-learn ta-lib

# 1. Prepare data (raw CSVs → normalised chunk CSVs)
python regression_preprocess.py

# 2. Train base model across all instruments
python regressive_trainer.py

# 3. (Optional) Fine-tune on a single instrument
#    Edit `instrument` variable at top of the file, then:
python regressive_fine_tuner.py

# 4. Evaluate (normalised + denormalised MAE/MSE)
python regression_test.py

# 5. Compare against random-walk baseline
python regression_get_mae.py

# 6. Tune noise augmentation parameters
python best_noise.py
```

## Key hyperparameters

All in `constants/global_constants.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NORMALIZING_WINDOW_SIZE` | 144 | Candles used for price normalisation |
| `REGRESSION_LABELING_WINDOW_SIZE` | 12 | Future 5-min candles for label |
| `NUM_TOKENS` | 64 | 5-min lookback length |
| `OTHER_TOKENS` | 64 | Hourly lookback length |
| `R_D_MODEL` | 256 | Model hidden dimension |
| `R_NUM_HEADS` | ~10 | Attention heads |
| `BATCH_SIZE` | 64 | Training batch size |
| `LR` | 4e-5 | Peak learning rate |

## File reference

| File | Purpose |
|------|---------|
| `regression_preprocess.py` | Preprocess raw CSVs → chunked datasets |
| `working_data.py` | Core data utilities (normalisation, labelling, cleaning) |
| `generators/regression_multi_instrument_data_generator.py` | TF dataset with noise augmentation |
| `modeler.py` | Keras model definition (`create_regression_model`) |
| `transformer_builder.py` | Custom Keras layers and LR schedule |
| `regression_losses.py` | Asymmetric Huber loss + trading profit metrics |
| `regressive_trainer.py` | Multi-instrument training entry point |
| `regressive_fine_tuner.py` | Single-instrument fine-tuning |
| `gradient_monitor.py` | Training callbacks (gradient/weight monitoring) |
| `regression_test.py` | Evaluation with denormalised metrics |
| `regression_get_mae.py` | Random-walk baseline |
| `regression_predictor.py` | Inference on historical data |
| `best_noise.py` | Grid search over noise augmentation params |
| `constants/global_constants.py` | All hyperparameters |

## Suggested clean structure

The repo also contains a legacy classification pipeline (buy/sell signal prediction, pre-August 2025). If cleaning up, these are the files to archive or remove:

**Legacy files:** `trainer.py`, `preprocess.py`, `predictor.py`, `losses.py`, `datasets.py`, `time_training.py`, `instrument_fine_tuning.py`, `recent_data_training.py`, `comparison.py`, `extract_weights.py`, `generators/multi_instrument_data_generator.py`, `generators/dataset_generator.py`, `memd/`

**Proposed clean tree:**
```
scalper/
├── constants/
│   └── global_constants.py
├── generators/
│   └── regression_multi_instrument_data_generator.py
├── modeler.py
├── transformer_builder.py
├── working_data.py
├── regression_preprocess.py
├── regression_losses.py
├── gradient_monitor.py
├── regressive_trainer.py
├── regressive_fine_tuner.py
├── regression_test.py
├── regression_get_mae.py
├── regression_predictor.py
└── best_noise.py
```
