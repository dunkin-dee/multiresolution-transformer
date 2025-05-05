import pandas as pd
import numpy as np
import os
import talib
from sklearn.model_selection import train_test_split


def clean_cols(df):
    clear_cols = ['tick_volume', 'spread', 'real_volume']
    df.drop(columns=clear_cols, inplace=True)
    return df

def clean_non_minute_rows(df):
    cut_off = df.index[df['time'].shift(-1) - df['time'] == 60].min()
    if cut_off == len(df):
        return df
    return df[cut_off:]

def add_rsi(df):
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi'] = df['rsi']/100
    print(df.tail())
    return df


def add_hour_position(df):
    """
    Adds hour position features to the dataframe.
    Assumes 'time' column contains Unix timestamps.
    
    Returns:
    - DataFrame with added hour position features:
      - hour_position: Normalized position in hour (0 to 1)
      - hour_position_sin: Sine component of the cyclic hour position
      - hour_position_cos: Cosine component of the cyclic hour position
    """
    # Convert Unix timestamp to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate position in the hour (0-11 for 5-minute intervals)
    df['minute_in_hour'] = df['datetime'].dt.minute
    df['interval_in_hour'] = df['minute_in_hour'] // 5  # 0 to 11
    
    # Create normalized position (0 to 1)
    df['hour_position'] = df['interval_in_hour'] / 12
    
    # Create sine and cosine components to capture cyclic nature
    df['hour_position_sin'] = np.sin(2 * np.pi * df['hour_position'])
    df['hour_position_cos'] = np.cos(2 * np.pi * df['hour_position'])
    
    # Drop intermediate columns
    df.drop(columns=['minute_in_hour', 'interval_in_hour'], inplace=True)
    
    return df


def add_bollinger_bands(df):
    df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'])
    return df

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


def label_df(df, window_size=20, mean_multiplier=4, positive_slope=0.4, cur_candle_multiplier=2):

    df['candle'] = df['close_normalized_for_label'] - df['open_normalized_for_label']
    mean_candle = df['candle'].abs().mean()
    future_cols = []
    sum_cols = []
    sum_cum = 1
    for x in range(-1, -window_size-1, -1):
        col_name = f'candle-{abs(x)}'
        sum_name = f'sum-{sum_cum}'
        df[col_name] = df['candle'].shift(x)
        future_cols.append(col_name)
        df[sum_name] = df[future_cols].sum(axis=1)
        sum_cols.append(sum_name)
        sum_cum += 1

    df['target'] = 0

    df['prev_candle'] = df['candle'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)

    mask = (df[sum_cols].max(axis=1) > mean_candle*mean_multiplier) & (df['close_normalized']>positive_slope) & ((df['candle']>mean_candle*cur_candle_multiplier) | (df['prev_candle'] > 0) | (df['close'] > df[['prev_close', 'prev_open']].max(axis=1))) & (df['candle'] > 0)
    df.loc[mask, 'target'] = 1

    drop_cols = future_cols + sum_cols + ['candle', 'prev_candle', 'prev_close', 'prev_open', 'close_normalized_for_label','open_normalized_for_label', 'open', 'high', 'low', 'close']
    df.drop(columns=drop_cols, inplace=True)

    return df[1:-window_size]


def alt_label_df(df, window_size=60, mean_multiplier=4, positive_slope=0.4, cur_candle_multiplier=2):
    df = df.reset_index(drop=True)  # Reset the index
    df['candle'] = df['close_normalized_for_label'] - df['open_normalized_for_label']
    df['target'] = 0
    mean_candle = df['candle'].abs().mean()
    acceptable_candle = mean_candle * cur_candle_multiplier

    for index in range(0, len(df) - window_size):
        row = df.iloc[index]
        if row['candle'] < acceptable_candle:
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




def split_df(
        df, 
        dump_path, 
        train_size=0.7,
        lookback=30,
        cols=[
    'open_normalized',
    'high_normalized',
    'low_normalized',
    'close_normalized',
    'target'
    ]):

    df = df[cols]
    train_data, temp_data = train_test_split(df, test_size=1-train_size, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, shuffle=False)
    val_data = val_data[lookback:]
    test_data = test_data[lookback:]
    train_data.to_csv(os.path.join(dump_path, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(dump_path, 'val.csv'), index=False)
    test_data.to_csv(os.path.join(dump_path, 'test.csv'), index=False)


def chunky_split_df(
        df, 
        dump_path, 
        train_size=0.7,
        chunk_size=1000,
        lookback=30,
        cols=[
            'open_normalized',
            'high_normalized',
            'low_normalized',
            'close_normalized',
            'target'
        ]):
    # Select only desired columns
    df = df[cols]
    
    # Initialize empty dataframes to collect the splits
    super_train_df = pd.DataFrame(columns=cols)
    super_val_df = pd.DataFrame(columns=cols)
    super_test_df = pd.DataFrame(columns=cols)
    
    # Process the dataframe in chunks
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        
        # Split chunk into training and temporary (val + test) sets
        train_chunk, temp_chunk = train_test_split(chunk, test_size=1 - train_size, shuffle=False)
        
        # Split temporary set equally into validation and test sets
        val_chunk, test_chunk = train_test_split(temp_chunk, test_size=0.5, shuffle=False)
        
        # Apply lookback slice to validation and test sets
        if len(val_chunk) > lookback:
            val_chunk = val_chunk.iloc[lookback:]
        else:
            val_chunk = pd.DataFrame(columns=cols)
        if len(test_chunk) > lookback:
            test_chunk = test_chunk.iloc[lookback:]
        else:
            test_chunk = pd.DataFrame(columns=cols)
        
        # Append the results to the super dataframes
        super_train_df = pd.concat([super_train_df, train_chunk], ignore_index=True)
        super_val_df = pd.concat([super_val_df, val_chunk], ignore_index=True)
        super_test_df = pd.concat([super_test_df, test_chunk], ignore_index=True)
    
    # Save the combined datasets to CSV files
    super_train_df.to_csv(os.path.join(dump_path, 'train.csv'), index=False)
    super_val_df.to_csv(os.path.join(dump_path, 'val.csv'), index=False)
    super_test_df.to_csv(os.path.join(dump_path, 'test.csv'), index=False)