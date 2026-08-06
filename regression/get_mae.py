"""Random-walk baseline: how well does "tomorrow looks like today" do?

Usage::

    python -m regression.get_mae [--instruments GBPUSD#]

Predicts each candle's normalised close as the previous candle's normalised
close, and reports MAE/MSE. This is the number the trained model has to beat to
be worth anything — reported alongside the model's own error by
``regression.test``.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from constants.global_constants import NORMALIZING_WINDOW_SIZE
from core.working_data import clean_five_minute_data, normalize_by_window

DEFAULT_SOURCE_DIR = "data/final_data"
OHLC = ['open', 'high', 'low', 'close']


def random_walk_baseline(instrument, source_dir):
    """Compute MAE/MSE for the previous-close predictor on one instrument."""
    df = pd.read_csv(f"{source_dir}/{instrument}/five_minutes.csv")
    print(f"\nProcessing {instrument}...")
    print(f"  raw: {len(df)} rows | "
          f"{datetime.fromtimestamp(df['time'].min())} to "
          f"{datetime.fromtimestamp(df['time'].max())}")

    df = clean_five_minute_data(df)
    print(f"  cleaned: {len(df)} rows")

    df = normalize_by_window(
        df,
        window_size=NORMALIZING_WINDOW_SIZE,
        low_col='low', high_col='high',
        normalizing_cols=OHLC,
        label_cols=['open', 'close'],
    )

    df['prediction'] = df['close_normalized_for_label'].shift(1)
    df = df.dropna(subset=['prediction'])

    y_true, y_pred = df['close_normalized_for_label'], df['prediction']
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"  Random-walk MAE: {mae:.6f}")
    print(f"  Random-walk MSE: {mse:.6f}")
    return {'instrument': instrument, 'mae': mae, 'mse': mse}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--instruments", nargs="*", default=None,
                        help="Defaults to all instruments under --source-dir.")
    args = parser.parse_args()

    instruments = args.instruments or sorted(
        d for d in os.listdir(args.source_dir)
        if os.path.isdir(os.path.join(args.source_dir, d))
    )

    for instrument in instruments:
        random_walk_baseline(instrument, args.source_dir)


if __name__ == "__main__":
    main()
