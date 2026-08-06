"""Turn raw OHLC CSVs into normalised, labelled, leakage-safe chunk files.

Usage::

    python -m regression.preprocess                      # every instrument found
    python -m regression.preprocess --instruments GBPUSD#

Input   ``data/final_data/{instrument}/{five_minutes,hours}.csv``
Output  ``data/experimenting/{instrument}/`` containing ``hour.csv`` plus
        ``training/``, ``validation/`` and ``testing/`` chunk CSVs.

The five stages, and why each exists:

1. **Clean** — broker exports contain gaps, weekend stubs and off-grid rows.
   Keep only candles that sit on a true 5-minute (or hourly) grid.
2. **Partial hour** — build a running OHLC of the *incomplete* current hour so
   the model can see the hour it is standing inside, not only closed hours.
3. **Normalise** — rescale prices against a trailing window (see README). This
   is what makes GOLD# and EURUSD# comparable to the same network.
4. **Label** — ``target_high``/``target_low`` are the max high and min low over
   the next ``REGRESSION_LABELING_WINDOW_SIZE`` candles, in normalised space.
5. **Chunk & split** — split into train/val/test *within* each chunk, with a
   margin that stops the hourly lookback of one chunk reaching into another.
"""

import argparse
import os
from datetime import datetime

import pandas as pd

from constants.global_constants import (
    NORMALIZING_WINDOW_SIZE,
    NUM_TOKENS,
    OTHER_TOKENS,
    REGRESSION_LABELING_WINDOW_SIZE,
)
from core.working_data import (
    add_partial_hour_ohlc,
    add_timing,
    clean_five_minute_data,
    clean_hour_data,
    normalize_by_window,
    normalize_partial_hour,
    regression_label_df_next,
    split_multiresolution_chunks,
)

DEFAULT_SOURCE_DIR = "data/final_data"
DEFAULT_OUTPUT_DIR = "data/experimenting"
OHLC = ['open', 'high', 'low', 'close']

#: Columns written to each chunk CSV. Raw `close` and the normalisation bounds
#: are carried through so evaluation can convert predictions back to real prices.
CHUNK_COLUMNS = [
    'time', 'position_in_hour', 'partial_hour_length',
    'open_normalized', 'high_normalized', 'low_normalized', 'close_normalized',
    'partial_open_normalized', 'partial_high_normalized',
    'partial_low_normalized', 'partial_close_normalized',
    'include', 'target_high', 'target_low',
    'norm_window_min', 'norm_window_max', 'close',
]


def describe(label, df):
    span = (f"{datetime.fromtimestamp(df['time'].min())} to "
            f"{datetime.fromtimestamp(df['time'].max())}")
    print(f"  {label}: {len(df)} rows | {span}")


def preprocess_instrument(instrument, source_dir, output_dir, chunk_size=20000):
    """Run the full pipeline for one instrument and write its chunk files."""
    print(f"\nProcessing {instrument}...")

    df = pd.read_csv(f"{source_dir}/{instrument}/five_minutes.csv")
    describe("5-minute raw", df)
    df = clean_five_minute_data(df)
    print(f"  5-minute cleaned: {len(df)} rows")

    df = add_timing(df)
    df = add_partial_hour_ohlc(df)
    df = normalize_by_window(
        df,
        window_size=NORMALIZING_WINDOW_SIZE,
        low_col='low', high_col='high',
        normalizing_cols=OHLC,
        label_cols=OHLC,
        keep_normalization_params=True,
    )

    hour_df = pd.read_csv(f"{source_dir}/{instrument}/hours.csv")
    describe("hourly raw", hour_df)
    hour_df = clean_hour_data(hour_df)
    print(f"  hourly cleaned: {len(hour_df)} rows")

    hour_df = add_timing(hour_df)
    hour_df = normalize_by_window(
        hour_df,
        window_size=NORMALIZING_WINDOW_SIZE,
        low_col='low', high_col='high',
        normalizing_cols=OHLC,
        label_cols=['open', 'close'],
        add_partial_hour=True,
    )

    df = normalize_partial_hour(df, hour_df)

    print("  labelling...")
    df = regression_label_df_next(df, window_size=REGRESSION_LABELING_WINDOW_SIZE)

    instrument_dir = f"{output_dir}/{instrument}"
    os.makedirs(instrument_dir, exist_ok=True)
    hour_df.to_csv(f"{instrument_dir}/hour.csv", index=False)

    summary = split_multiresolution_chunks(
        df_5min=df,
        df_hour=hour_df,
        dump_path=instrument_dir,
        chunk_size=chunk_size,
        hour_lookback=OTHER_TOKENS,
        lookback=NUM_TOKENS,
        cols=CHUNK_COLUMNS,
    )
    print(f"  wrote {summary['total_chunks']} chunks to {instrument_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--instruments", nargs="*", default=None,
                        help="Instrument directory names. Defaults to all found "
                             "under --source-dir.")
    parser.add_argument("--chunk-size", type=int, default=20000)
    args = parser.parse_args()

    instruments = args.instruments or sorted(
        d for d in os.listdir(args.source_dir)
        if os.path.isdir(os.path.join(args.source_dir, d))
    )
    print(f"Preprocessing {len(instruments)} instrument(s) into {args.output_dir}")

    for instrument in instruments:
        preprocess_instrument(instrument, args.source_dir, args.output_dir,
                              chunk_size=args.chunk_size)

    print("\nDone. Next: python -m regression.trainer")


if __name__ == "__main__":
    main()
