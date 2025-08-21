import os
import pandas as pd
from datetime import datetime
from working_data import clean_five_minute_data, add_partial_hour_ohlc, add_timing, normalize_by_window, clean_hour_data, split_multiresolution_chunks, regression_label_df, normalize_partial_hour, regression_label_df_next
from constants.global_constants import *

starting_dir = "data/final_data"
working_path = "data/experimenting"
instruments = os.listdir(starting_dir)
instruments = ['GBPUSD#']

for instrument in instruments:
    df = pd.read_csv(f"{starting_dir}/{instrument}/five_minutes.csv")
    print(f"Processing {instrument}...")
    print(f"  5 minute Original data: {len(df)} rows")
    print(f"  5 minute Time range: {datetime.fromtimestamp(df['time'].min())} to {datetime.fromtimestamp(df['time'].max())}")
    df = clean_five_minute_data(df)
    print(f"  Cleaned data: {len(df)} rows")
    df = add_timing(df)
    df = add_partial_hour_ohlc(df)
    df = normalize_by_window(
        df, 
        window_size=NORMALIZING_WINDOW_SIZE, 
        low_col='low',
        high_col='high',
        normalizing_cols=[
            'open',
            'high',
            'low',
            'close'
        ],
        label_cols=['open', 'close'])

    hour_df = pd.read_csv(f"{starting_dir}/{instrument}/hours.csv")
    print(f"  Hour Original data: {len(hour_df)} rows")
    print(f"  Hour Time range: {datetime.fromtimestamp(hour_df['time'].min())} to {datetime.fromtimestamp(hour_df['time'].max())}")
    hour_df = clean_hour_data(hour_df)
    print(f"  Cleaned data: {len(hour_df)} rows")
    hour_df = add_timing(hour_df)
    hour_df = normalize_by_window(
        hour_df, 
        window_size=NORMALIZING_WINDOW_SIZE, 
        low_col='low',
        high_col='high',
        normalizing_cols=[
            'open',
            'high',
            'low',
            'close'
        ],
        label_cols=['open', 'close'],
        add_partial_hour=True)

    df = normalize_partial_hour(df, hour_df)

    print(f"Labeling...\n\n\n")
    # df = regression_label_df(df, window_size=REGRESSION_LABELING_WINDOW_SIZE, 
    #                 positive_slope=POSITIVE_SLOPE, 
    #                 negative_slope=NEGATIVE_SLOPE,
    #                 starting_hour=9,
    #                 ending_hour=18,
    #                 lookback_window=LABEL_LOOKBACK)
    df = regression_label_df_next(df)


    os.makedirs(f"{working_path}/{instrument}", exist_ok=True)

    hour_df.to_csv(f"{working_path}/{instrument}/hour.csv", index=False)
    split_multiresolution_chunks(df_5min=df,
                                df_hour=hour_df,
                                dump_path=f"{working_path}/{instrument}",
                                chunk_size=20000,
                                hour_lookback=OTHER_TOKENS,
                                lookback=NUM_TOKENS,
                                cols=[
                                    'time',
                                    'position_in_hour',
                                    'partial_hour_length',
                                    'open_normalized',
                                    'high_normalized',
                                    'low_normalized',
                                    'close_normalized',
                                    'partial_open_normalized',
                                    'partial_high_normalized',
                                    'partial_low_normalized',
                                    'partial_close_normalized',
                                    'include',
                                    'target_high',
                                    'target_low'
                                ])