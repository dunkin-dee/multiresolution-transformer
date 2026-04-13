import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.working_data import clean_five_minute_data, normalize_by_window
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
    
    df['label'] = df['close_normalized_for_label'].shift(1)

    df = df.dropna(subset=['label'])


    y_true = df['close_normalized_for_label']
    y_pred = df['label']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"Random Walk MAE: {mae:.6f}")
    print(f"Random Walk MSE: {mse:.6f}")