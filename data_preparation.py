import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_df(
        df, 
        dump_path,
        base_name,
        train_size=0.7, 
        cols=[
    'open_normalized',
    'high_normalized',
    'low_normalized',
    'close_normalized',
    'target'
    ]):

    df = df[cols]

    train_path = os.path.join(dump_path, 'train/')
    val_path = os.path.join(dump_path, 'val/')
    test_path = os.path.join(dump_path, 'test/')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    train_data, temp_data = train_test_split(df, test_size=1-train_size, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, shuffle=False)
    train_data.to_csv(os.path.join(train_path, f'{base_name}.csv'), index=False)
    val_data.to_csv(os.path.join(val_path, f'{base_name}.csv'), index=False)
    test_data.to_csv(os.path.join(test_path, f'{base_name}.csv'), index=False)


def clean_cols(df):
    clear_cols = ['tick_volume', 'spread', 'real_volume']
    df.drop(columns=clear_cols, inplace=True)
    return df


def clean_non_minute_rows(df):
    cut_off = df.index[df['time'].shift(-1) - df['time'] == 60].min()
    if cut_off == len(df):
        return df
    return df[cut_off:]

def normalize_by_window(
        df, 
        window_size=1440, 
        chunk_size=20, 
        normalizing_cols=[
            'open',
            'high',
            'low',
            'close']):
    col_numbers = range(1, window_size+1)
    col_split = [col_numbers[i:i + chunk_size] for i in range(0, len(col_numbers), chunk_size)]
    df['window_min'] = df['low']
    df['window_max'] = df['high']

    for col_range in col_split:
        high_cols = []
        low_cols = []
        
        for x in col_range:
            high_col_name = f"high-{x}"
            df[high_col_name] = df['high'].shift(x)
            high_cols.append(high_col_name)
        high_cols_inclusive = high_cols + ['window_max']
        df['window_max'] = df[high_cols_inclusive].max(axis=1)
        df.drop(columns=high_cols, inplace=True)

        for x in col_range:
            low_col_name = f"low-{x}"
            df[low_col_name] = df['low'].shift(x)
            low_cols.append(low_col_name)
        low_cols_inclusive = low_cols + ['window_min']
        df['window_min'] = df[low_cols_inclusive].min(axis=1)
        df.drop(columns=low_cols, inplace=True)

    for normalizing_col in normalizing_cols:
        df[f"{normalizing_col}_normalized"] = (df[normalizing_col] - df['window_min'])/(df['window_max'] - df['window_min'])


    df['window_max_prev'] = df['window_max'].shift(1)
    df['window_min_prev'] = df['window_min'].shift(1)

    df['open_normalized_for_label'] = (df['open'] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
    df['close_normalized_for_label'] = (df['close'] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
    df.drop(columns=['window_max', 'window_min', 'window_max_prev', 'window_min_prev'], inplace=True)
    return df[window_size:]

def label_df(df, window_size=60, mean_multiplier=4, positive_slope=0.4, cur_candle_multiplier=2):
    df = df.reset_index(drop=True)  # Reset the index
    df['candle'] = df['close_normalized_for_label'] - df['open_normalized_for_label']
    df['target'] = 0
    mean_candle = df['candle'].abs().mean()
    acceptable_candle = mean_candle * cur_candle_multiplier

    for index in range(0, len(df) - window_size):
        row = df.iloc[index]
        if row['candle'] < acceptable_candle:
            continue
        if row['close_normalized_for_label'] < positive_slope:
            continue
        target_signal = row['close_normalized'] + mean_candle * mean_multiplier
        end_signal = row['close_normalized'] - (mean_candle * 2)
        cur_close = row['close_normalized']

        for mini_index in range(index + 1, index + window_size):
            mini_row = df.iloc[mini_index]
            if mini_row['close_normalized'] < end_signal:
                break
            cur_close += mini_row['candle']
            if cur_close > target_signal: # or mini_row['close_normalized_for_label'] >= 1:
                df.at[index, 'target'] = 1
                break

    return df[:len(df) - window_size]


def break_by_time(df, time_break):
    return df[df['time'] > time_break]
