# Legacy — classification pipeline

This is the project's **first formulation**, superseded in August 2025. It is kept
because it explains how the current design was arrived at, not because it is
maintained. Nothing in `regression/` imports it.

## What it did

Instead of predicting a price range, it framed the problem as **binary
classification**: label each candle 1 if a long entry there would have reached a
profit target of `LABEL_MEAN_MULTIPLIER × mean_candle_size` before hitting a
stop-loss of `DRAWDOWN × mean_candle_size`, within `LABELING_WINDOW_SIZE`
candles. Everything else is 0.

## Why it was abandoned

The labelling embeds a fixed profit target, a fixed stop, and a fixed holding
window. Every one of those is a trading decision baked into the ground truth, so
the model could only ever learn *one* strategy's entry signal — and changing the
strategy meant relabelling and retraining from scratch. The classes were also
severely imbalanced, which is why `generators/data_generator.py` carries SMOTE
resampling that the regression generator has no need for.

Predicting the high/low range directly moves those decisions out of the label and
into the consumer of the prediction. The same model output can then serve
different position-sizing or risk rules without retraining.

## Contents

| File | Role |
|---|---|
| `classification/trainer.py` | Training loop with time-window scheduling |
| `classification/preprocess.py` | Preprocessing into binary-labelled chunks |
| `classification/datasets.py` | Dataset assembly |
| `classification/losses.py` | Precision-focused and balanced trading losses |
| `classification/predictor.py` | Inference |
| `classification/fine_tuner.py` | Per-instrument fine-tuning |
| `classification/recent_data_training.py` | Continuation training on recent data |
| `classification/generators/data_generator.py` | Generator with SMOTE resampling |
| `classification/generators/dataset_generator.py` | Earlier generator implementation |
| `classification/comparison.py` | One-off loading-method comparison |
| `classification/extract_weights.py` | Weight extraction utility |
| `classification/time_training.py` | Non-functional — imports a module that no longer exists |

## `memd.py`

Third-party code, not written for this project: a Python translation of the
MEMD (Multivariate Empirical Mode Decomposition) Matlab implementation, by
Mario de Souza e Silva, from the original by Rehman & Mandic. It was explored as
a signal-decomposition front-end for the price series and never wired into
either pipeline. Kept with its original attribution header intact.

## Caveats

These scripts still use `from constants.global_constants import *`, hardcoded
paths, and in several cases execute work at import time. `preprocess.py` has been
put behind a `main()` guard because importing it silently overwrote a data
directory; the others have not been touched. Treat this directory as read-only
history.
