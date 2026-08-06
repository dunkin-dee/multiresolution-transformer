"""Run a trained model over historical CSVs and print high-conviction signals.

Usage::

    python -m regression.predictor --instrument EURUSD# --data-dir data/checking_data

Expects ``{data-dir}/{instrument}/five_minutes.csv`` and ``hours.csv`` in the same
raw format as ``data/final_data``. Applies the preprocessing pipeline live rather
than reading preprocessed chunks, which is what a real-time caller would do.

The entry filters below (bullish structure, mid-range position, session hours)
mirror the conditions the labelling stage assumed. They are a demonstration of
how the prediction would be consumed, not a validated trading strategy.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from constants.global_constants import (
    FEATURES,
    NORMALIZING_WINDOW_SIZE,
    NUM_TOKENS,
    OTHER_TOKENS,
    R_D_MODEL,
    R_FF_DIM,
    R_NUM_HEADS,
)
from core.modeler import create_regression_model
from core.working_data import (
    add_partial_hour_ohlc,
    add_timing,
    normalize_by_window,
    normalize_partial_hour,
)

OHLC = ['open', 'high', 'low', 'close']


def create_inference_batch(main_df, hourly_df, main_lookback_tokens,
                           hourly_lookback_tokens, feature_columns):
    """Build a single batch-of-one matching the training tensor layout.

    Returns the five tensors the model expects, in input order:
    ``(minute, hourly, partial_hour, minutes_into_hour, partial_hour_length)``.
    """
    main_features = main_df[feature_columns].iloc[-main_lookback_tokens:].values
    hourly_features = hourly_df[feature_columns].iloc[-hourly_lookback_tokens:].values

    partial_hour_cols = ['partial_open_normalized', 'partial_high_normalized',
                         'partial_low_normalized', 'partial_close_normalized']
    partial_hour_data = main_df[partial_hour_cols].iloc[-1:].values

    current_row = main_df.iloc[-1]

    return (
        tf.convert_to_tensor(np.expand_dims(main_features, 0).astype(np.float32)),
        tf.convert_to_tensor(np.expand_dims(hourly_features, 0).astype(np.float32)),
        tf.convert_to_tensor(np.expand_dims(partial_hour_data, 0).astype(np.float32)),
        tf.convert_to_tensor([[current_row['position_in_hour']]], dtype=tf.float32),
        tf.convert_to_tensor([[current_row['partial_hour_length']]], dtype=tf.float32),
    )


def prepare_frames(main_path, hourly_path):
    """Apply the same normalisation pipeline used in preprocessing.

    This must stay in step with ``regression.preprocess`` — any divergence means
    the model sees a different distribution at inference than it trained on.
    """
    df = pd.read_csv(main_path)
    df = add_timing(df)
    df = add_partial_hour_ohlc(df)
    df = normalize_by_window(
        df, window_size=NORMALIZING_WINDOW_SIZE, low_col='low', high_col='high',
        normalizing_cols=OHLC, label_cols=['open', 'close'],
    )

    hour_df = pd.read_csv(hourly_path)
    hour_df = add_timing(hour_df)
    hour_df = normalize_by_window(
        hour_df, window_size=NORMALIZING_WINDOW_SIZE, low_col='low', high_col='high',
        normalizing_cols=OHLC, label_cols=['open', 'close'], add_partial_hour=True,
    )

    df = normalize_partial_hour(df, hour_df)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    hour_df['time'] = pd.to_datetime(hour_df['time'], unit='s')

    return (df.sort_values('time').reset_index(drop=True),
            hour_df.sort_values('time').reset_index(drop=True))


def iter_inference_points(df, hour_df, main_lookback_tokens, hourly_lookback_tokens,
                          feature_columns):
    """Yield one inference point per 5-minute row that has sufficient history.

    Hourly context is truncated to hours *strictly before* the current hour, so
    the partial-hour input is the only view the model gets of the hour in
    progress. This is the leakage guard that makes the walk-forward honest.
    """
    for idx in range(max(main_lookback_tokens - 1, 0), len(df)):
        mini_df = df.iloc[: idx + 1]
        hour_floor = mini_df.iloc[-1]['time'].replace(minute=0, second=0, microsecond=0)
        mini_hour_df = hour_df[hour_df['time'] < hour_floor]

        if len(mini_df) < main_lookback_tokens or len(mini_hour_df) < hourly_lookback_tokens:
            continue

        row, prev = mini_df.iloc[-1], mini_df.iloc[-2]
        yield {
            'inputs': create_inference_batch(
                mini_df, mini_hour_df, main_lookback_tokens,
                hourly_lookback_tokens, feature_columns,
            ),
            'time': row['time'],
            'close': row['close'], 'open': row['open'],
            'prev_close': prev['close'], 'prev_open': prev['open'],
            'close_normalized': row['close_normalized'],
        }


def passes_entry_filters(point, low=0.3, high=0.7, start_hour=10, end_hour=18):
    """Bullish structure, mid-range price, and inside the active session."""
    bullish_pair = (point['close'] > point['open']
                    and point['prev_close'] > point['prev_open'])
    above_prev_open = point['close'] > point['prev_open']
    if not (bullish_pair or above_prev_open):
        return False
    if not low <= point['close_normalized'] <= high:
        return False
    return start_hour <= point['time'].hour <= end_hour


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True, help="e.g. 'EURUSD#'")
    parser.add_argument("--data-dir", default="data/checking_data",
                        help="Directory holding {instrument}/five_minutes.csv and hours.csv.")
    parser.add_argument("--model-path", default=None,
                        help="Defaults to models/regressor_{instrument}.keras.")
    parser.add_argument("--threshold", type=float, default=9.0,
                        help="Only report predictions above this target_high value.")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable mixed_float16, as used during training.")
    args = parser.parse_args()

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled: mixed_float16")

    model_path = args.model_path or f"models/regressor_{args.instrument}.keras"
    main_path = os.path.join(args.data_dir, args.instrument, 'five_minutes.csv')
    hourly_path = os.path.join(args.data_dir, args.instrument, 'hours.csv')

    for path in (main_path, hourly_path, model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    model = create_regression_model(
        feature_cols=FEATURES, d_model=R_D_MODEL, num_heads=R_NUM_HEADS,
        ff_dim=R_FF_DIM, num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS,
    )
    model.load_weights(model_path)

    print(f"Scanning {args.instrument} with {model_path}...")
    df, hour_df = prepare_frames(main_path, hourly_path)

    hits = 0
    for point in iter_inference_points(df, hour_df, NUM_TOKENS, OTHER_TOKENS, FEATURES):
        if not passes_entry_filters(point):
            continue
        prediction = model.predict(point['inputs'], verbose=0)
        predicted_high = float(prediction['target_high'][0][0])
        if predicted_high > args.threshold:
            hits += 1
            print(f"  {point['time']}  target_high={predicted_high:.3f}  "
                  f"close={point['close']:.5f}")

    print(f"\n{hits} signal(s) above threshold {args.threshold}.")


if __name__ == "__main__":
    main()
