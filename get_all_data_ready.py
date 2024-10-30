from data_preparation import label_df, normalize_by_window, clean_cols, clean_non_minute_rows, break_by_time, split_df
from datetime import datetime
import pandas as pd
import os

NORMALIZING_WINDOW_SIZE = 60*24
LABELING_WINDOW_SIZE = 20
POSITIVE_SLOPE = 0.3
LABEL_CUR_CANDLE_MULTIPLIER = 0
LABEL_MEAN_MULTIPLIER = 6
DUMP_PATH = "ready_data"


dt = datetime(2016, 1, 1)
epoch_seconds = int(dt.timestamp())

file_paths = [
    "data/GBPUSD/minutes.csv",
    "data/EURUSD/minutes.csv",
    "data/USDJPY/minutes.csv",
    "data/USDCHF/minutes.csv",
    "data/USDCAD/minutes.csv",
    "data/EURGBP/minutes.csv",
    "data/GOLD/minutes.csv",
    "data/BTCUSD/minutes.csv",
    "data/ETHUSD/minutes.csv",
    "data/AUDUSD/minutes.csv",
    "data/USDCAD/minutes.csv"
]


train_files = []
val_files = []
test_files = []

for source_csv in file_paths:
    print(f"Working on {source_csv}...")
    if os.path.exists(f"ready_data/train/{source_csv.split('/')[1]}.csv"):
        continue
    base_name = source_csv.split('/')[1]
    df = pd.read_csv(source_csv)
    df = clean_non_minute_rows(df)
    df = clean_cols(df)
    print(f"Breaking at time {epoch_seconds}")
    df = break_by_time(df, epoch_seconds)
    print("Normalizing...")
    df = normalize_by_window(
        df, 
        window_size=NORMALIZING_WINDOW_SIZE, 
        normalizing_cols=[
            'open',
            'high',
            'low',
            'close',
        ])
    print("Labeling...")
    df = label_df(
        df, 
        window_size=LABELING_WINDOW_SIZE, 
        mean_multiplier=LABEL_MEAN_MULTIPLIER, 
        cur_candle_multiplier=LABEL_CUR_CANDLE_MULTIPLIER,
        positive_slope=POSITIVE_SLOPE)
    split_df(
        df=df, 
        dump_path=DUMP_PATH,
        base_name=base_name,
        cols=[
            'time',
            'open',
            'high',
            'low',
            'close',
            'open_normalized',
            'high_normalized',
            'low_normalized',
            'close_normalized',
            'target'
        ])