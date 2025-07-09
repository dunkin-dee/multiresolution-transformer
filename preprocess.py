import os
import pandas as pd
from datetime import datetime
from working_data import clean_five_minute_data, add_timing, normalize_by_window, clean_hour_data, split_multiresolution_chunks, alt_label_df
from constants.global_constants import *


starting_dir = "data/finer_data"
working_path = "data/split_finer_data"
instruments = os.listdir(starting_dir)

for instrument in instruments:
    df = pd.read_csv(f"{starting_dir}/{instrument}/five_minutes.csv")
    print(f"Processing {instrument}...")
    print(f"  5 minute Original data: {len(df)} rows")
    print(f"  5 minute Time range: {datetime.fromtimestamp(df['time'].min())} to {datetime.fromtimestamp(df['time'].max())}")
    df = clean_five_minute_data(df)
    print(f"  Cleaned data: {len(df)} rows")
    df = add_timing(df)
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
        label_cols=['open', 'close'])
    
    print(f"Labeling...\n\n\n")
    df = alt_label_df(df, window_size=LABELING_WINDOW_SIZE, 
                  mean_multiplier=LABEL_MEAN_MULTIPLIER, 
                  cur_candle_multiplier=LABEL_CUR_CANDLE_MULTIPLIER, 
                  positive_slope=POSITIVE_SLOPE, 
                  negative_slope=NEGATIVE_SLOPE,
                  starting_hour=STARTING_HOUR,
                  ending_hour=ENDING_HOUR,
                  drawdown=DRAWDOWN,
                  lookback_window=LABEL_LOOKBACK)
    
    
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
                                    'open_normalized',
                                    'high_normalized',
                                    'low_normalized',
                                    'close_normalized',
                                    'include',
                                    'target'
                                ])

