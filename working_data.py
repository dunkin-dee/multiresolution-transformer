import pandas as pd
import numpy as np
import os
import talib
from sklearn.model_selection import train_test_split


def clean_cols(df):
    clear_cols = ['spread', 'tick_volume', 'real_volume']
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
    return df

def add_timing(df):
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['hour_of_day'] = df['datetime'].dt.hour
    df['position_in_hour'] = (df['datetime'].dt.minute // 55)
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

# def normalize_by_window(
#         df, 
#         window_size=1440, 
#         chunk_size=20, 
#         normalizing_cols=[
#             'open',
#             'high',
#             'low',
#             'close']):
#     col_numbers = range(1, window_size+1)
#     col_split = [col_numbers[i:i + chunk_size] for i in range(0, len(col_numbers), chunk_size)]
#     df['window_min'] = df['low']
#     df['window_max'] = df['high']

#     for col_range in col_split:
#         high_cols = []
#         low_cols = []
        
#         for x in col_range:
#             high_col_name = f"high-{x}"
#             df[high_col_name] = df['high'].shift(x)
#             high_cols.append(high_col_name)
#         high_cols_inclusive = high_cols + ['window_max']
#         df['window_max'] = df[high_cols_inclusive].max(axis=1)
#         df.drop(columns=high_cols, inplace=True)

#         for x in col_range:
#             low_col_name = f"low-{x}"
#             df[low_col_name] = df['low'].shift(x)
#             low_cols.append(low_col_name)
#         low_cols_inclusive = low_cols + ['window_min']
#         df['window_min'] = df[low_cols_inclusive].min(axis=1)
#         df.drop(columns=low_cols, inplace=True)

#     for normalizing_col in normalizing_cols:
#         df[f"{normalizing_col}_normalized"] = (df[normalizing_col] - df['window_min'])/(df['window_max'] - df['window_min'])


#     df['window_max_prev'] = df['window_max'].shift(1)
#     df['window_min_prev'] = df['window_min'].shift(1)

#     df['open_normalized_for_label'] = (df['open'] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
#     df['close_normalized_for_label'] = (df['close'] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
#     df.drop(columns=['window_max', 'window_min', 'window_max_prev', 'window_min_prev'], inplace=True)
#     return df[window_size:]

def normalize_by_window(
        df, 
        high_col='high',
        low_col='low',
        window_size=1440, 
        chunk_size=20, 
        normalizing_cols=[
            'open',
            'high',
            'low',
            'close'],
        label_cols=[]):
    """
    Normalize columns by rolling window min/max values.
    
    Parameters:
    - df: DataFrame to normalize
    - high_col: Column name containing high values
    - low_col: Column name containing low values  
    - window_size: Size of rolling window for min/max calculation
    - chunk_size: Size of chunks to process at once (for memory efficiency)
    - normalizing_cols: List of columns to normalize using current window
    - label_cols: List of columns to normalize using previous window (for labels)
    
    Returns:
    - DataFrame with normalized columns, trimmed to remove first window_size rows
    """
    col_numbers = range(1, window_size+1)
    col_split = [col_numbers[i:i + chunk_size] for i in range(0, len(col_numbers), chunk_size)]
    df['window_min'] = df[low_col]
    df['window_max'] = df[high_col]

    for col_range in col_split:
        high_cols = []
        low_cols = []
        
        for x in col_range:
            high_col_name = f"{high_col}-{x}"
            df[high_col_name] = df[high_col].shift(x)
            high_cols.append(high_col_name)
        high_cols_inclusive = high_cols + ['window_max']
        df['window_max'] = df[high_cols_inclusive].max(axis=1)
        df.drop(columns=high_cols, inplace=True)

        for x in col_range:
            low_col_name = f"{low_col}-{x}"
            df[low_col_name] = df[low_col].shift(x)
            low_cols.append(low_col_name)
        low_cols_inclusive = low_cols + ['window_min']
        df['window_min'] = df[low_cols_inclusive].min(axis=1)
        df.drop(columns=low_cols, inplace=True)

    # Normalize the main columns using current window
    for normalizing_col in normalizing_cols:
        df[f"{normalizing_col}_normalized"] = (df[normalizing_col] - df['window_min'])/(df['window_max'] - df['window_min'])

    # Handle label normalization only if label_cols is not empty
    if label_cols:
        df['window_max_prev'] = df['window_max'].shift(1)
        df['window_min_prev'] = df['window_min'].shift(1)

        for label_col in label_cols:
            df[f"{label_col}_normalized_for_label"] = (df[label_col] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
        
        df.drop(columns=['window_max_prev', 'window_min_prev'], inplace=True)

    # df.drop(columns=['window_max', 'window_min'], inplace=True)
    return df[window_size:]

def normalize_by_window_v2(
        df, 
        window_size=1440, 
        chunk_size=20, 
        normalizing_cols=[
            'open',
            'high',
            'low',
            'close']):
    """
    Alternative implementation using pandas rolling functions for better performance
    and clearer exclusion of current values.
    """
    # Calculate rolling min/max excluding current row
    df['window_max'] = df['high'].shift(1).rolling(window=window_size, min_periods=1).max()
    df['window_min'] = df['low'].shift(1).rolling(window=window_size, min_periods=1).min()
    
    # Normalize columns using historical min/max
    for normalizing_col in normalizing_cols:
        df[f"{normalizing_col}_normalized"] = (df[normalizing_col] - df['window_min'])/(df['window_max'] - df['window_min'])

    # For labels, use the previous period's normalization range
    df['window_max_prev'] = df['window_max'].shift(1)
    df['window_min_prev'] = df['window_min'].shift(1)

    df['open_normalized_for_label'] = (df['open'] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
    df['close_normalized_for_label'] = (df['close'] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
    
    # Clean up temporary columns
    df.drop(columns=['window_max_prev', 'window_min_prev'], inplace=True)
    
    return df[window_size:]


def label_df(df, window_size=20, mean_multiplier=4, positive_slope=0.4, negative_slope=0.8, cur_candle_multiplier=2):

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

    mask = (df[sum_cols].max(axis=1) >= mean_candle*mean_multiplier) & (df['close_normalized']<negative_slope) & (df['close_normalized']>positive_slope) & ((df['candle']>mean_candle*cur_candle_multiplier) | (df['prev_candle'] > 0) | (df['close'] > df[['prev_close', 'prev_open']].max(axis=1))) & (df['candle'] > 0)
    df.loc[mask, 'target'] = 1

    drop_cols = future_cols + sum_cols + ['candle', 'prev_candle', 'prev_close', 'prev_open', 'close_normalized_for_label','open_normalized_for_label', 'open', 'high', 'low', 'close']
    df.drop(columns=drop_cols, inplace=True)

    return df[1:-window_size]


# def alt_label_df(df, 
#                  window_size=60, 
#                  mean_multiplier=4, 
#                  positive_slope=0.4, 
#                  negative_slope=0.8,
#                  cur_candle_multiplier=2,
#                  starting_hour=0,
#                  ending_hour=24):
#     df = df.reset_index(drop=True)  # Reset the index
#     df['candle'] = df['close_normalized_for_label'] - df['open_normalized_for_label']
#     df['target'] = 0
#     df['include'] = 0
#     mean_candle = df['candle'].abs().mean()
#     acceptable_candle = mean_candle * cur_candle_multiplier

#     for index in range(0, len(df) - window_size):
#         row = df.iloc[index]
#         if row['candle'] < acceptable_candle:
#             continue
#         if row['close_normalized'] > negative_slope or row['close_normalized'] < positive_slope:
#             continue
#         if row['hour_of_day'] < starting_hour or row['hour_of_day'] >= ending_hour:
#             continue
#         df.at[index, 'include'] = 1
#         target_signal = row['close_normalized'] + mean_candle * mean_multiplier
#         end_signal = row['close_normalized'] - (mean_candle * 1)
#         cur_close = row['close_normalized']

#         for mini_index in range(index + 1, index + window_size):
#             mini_row = df.iloc[mini_index]
#             if mini_row['close_normalized'] < end_signal:
#                 break
#             cur_close += mini_row['candle']
#             if cur_close > target_signal: # or mini_row['close_normalized_for_label'] >= 1:
#                 df.at[index, 'target'] = 1
#                 break

#     return df[:len(df) - window_size]


def alt_label_df(df, 
                 window_size=60, 
                 mean_multiplier=4, 
                 positive_slope=0.4, 
                 negative_slope=0.8,
                 cur_candle_multiplier=2,
                 starting_hour=0,
                 ending_hour=24,
                 lookback_window=64,
                 drawdown=1):
    """
    Label buy signals in normalized OHLC data.
    
    Parameters:
    - df: DataFrame with normalized OHLC data
    - window_size: Lookforward window to check for target achievement
    - mean_multiplier: Multiplier for mean candle size to set target
    - positive_slope: Lower bound for close_normalized (buy zone)
    - negative_slope: Upper bound for close_normalized (buy zone)  
    - cur_candle_multiplier: Multiplier for acceptable current candle size
    - starting_hour: Start of trading window
    - ending_hour: End of trading window
    
    Returns:
    - DataFrame with 'target' column (1 for buy signals, 0 otherwise)
    """
    df = df.copy()  # Avoid modifying original DataFrame
    df = df.reset_index(drop=True)
    
    # Calculate candle body size
    df['candle'] = df['close_normalized_for_label'] - df['open_normalized_for_label']
    df['target'] = 0
    df['include'] = 0
    
    # Calculate mean candle size and acceptable threshold
    # mean_candle = df['candle'].abs().mean()
    # acceptable_candle = mean_candle * cur_candle_multiplier
    
    # Ensure we have required columns
    required_cols = ['close_normalized', 'open_normalized', 'low_normalized', 
                    'close_normalized_for_label', 'open_normalized_for_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if hour_of_day column exists
    has_hour_filter = 'hour_of_day' in df.columns
    
    for index in range(len(df) - window_size):
        lookback_start = max(0, index - lookback_window)
        mean_candle = df['candle'].iloc[lookback_start:index].abs().mean()
        acceptable_candle = mean_candle * cur_candle_multiplier
        row = df.iloc[index]
        
        # Filter 1: Check candle size
        if abs(row['candle']) < acceptable_candle:
            continue
            
        # Filter 2: Check if price is in acceptable range (buy zone)
        if row['close_normalized'] > negative_slope or row['close_normalized'] < positive_slope:
            continue
            
        # Filter 3: Check trading hours (if column exists)
        if has_hour_filter:
            if row['hour_of_day'] < starting_hour or row['hour_of_day'] >= ending_hour:
                continue

        df.at[index, 'include'] = 1
        
        # Set target and stop-loss levels
        target_signal = row['close_normalized'] + (mean_candle * mean_multiplier)
        # Use low_normalized for stop-loss instead of close_normalized
        stop_loss = row['close_normalized'] - (mean_candle * drawdown)
        
        # Look forward to see if target is hit before stop-loss
        target_hit = False
        for mini_index in range(index + 1, min(index + window_size + 1, len(df))):
            mini_row = df.iloc[mini_index]
            
            # Check stop-loss using low_normalized (more realistic)
            if mini_row['low_normalized'] < stop_loss:
                break
                
            # Check if target is hit using high_normalized (more realistic)
            if mini_row['high_normalized'] >= target_signal:
                target_hit = True
                break
        
        if target_hit:
            df.at[index, 'target'] = 1
    
    # Remove the temporary candle column and return
    df = df.drop(columns=['candle'])
    return df.iloc[:len(df) - window_size].copy()

# def alt_label_df_raw(df, window_size=60, mean_multiplier=4, positive_slope=0.4, negative_slope=0.8, cur_candle_multiplier=2):
#     """
#     Label buying opportunities using raw price values instead of normalized values
#     to avoid missing opportunities at the top of normalization curves.
#     """
#     df = df.reset_index(drop=True)
    
#     # Calculate raw candle size using actual OHLC values
#     df['candle_raw'] = df['close'] - df['open']
#     df['target'] = 0
    
#     # Use absolute candle size for mean calculation
#     mean_candle_raw = df['candle_raw'].abs().mean()
#     acceptable_candle_raw = mean_candle_raw * cur_candle_multiplier
    
#     for index in range(0, len(df) - window_size):
#         row = df.iloc[index]
        
#         # Filter based on raw candle size
#         if abs(row['candle_raw']) < acceptable_candle_raw:
#             continue

#         if row['close_normalized'] < positive_slope or row['close_normalized'] > negative_slope:
#             continue
            
#         # Set target and end signals based on raw close price
#         target_signal = row['close'] + (mean_candle_raw * mean_multiplier)
#         end_signal = row['close'] - (mean_candle_raw * 1)
        
#         # Track cumulative price movement using raw values
#         cur_close = row['close']
        
#         for mini_index in range(index + 1, index + window_size):
#             mini_row = df.iloc[mini_index]
            
#             # Exit if price drops below end signal
#             if mini_row['close'] < end_signal:
#                 break
                
#             # Add raw candle movement to current price
#             cur_close += mini_row['candle_raw']
            
#             # Check if target is reached using raw price
#             if cur_close > target_signal:
#                 df.at[index, 'target'] = 1
#                 break
    
#     return df[:len(df) - window_size]

def alt_label_df_raw(df, window_size=60, mean_multiplier=4, 
                     positive_slope=0.4, negative_slope=0.8,
                     starting_hour=0, ending_hour=24,
                     cur_candle_multiplier=2, lookback_window=64, drawdown=1):
    """
    Label buying opportunities using raw price values instead of normalized values
    to avoid missing opportunities at the top of normalization curves.
    Now calculates mean_candle_raw as a column using the absolute mean of the last N rows.
    """
    df = df.reset_index(drop=True)
    
    # Calculate raw candle size using actual OHLC values
    df['candle_raw'] = df['close'] - df['open']
    df['mean_candle_raw'] = 0.0
    df['target'] = 0
    df['include'] = 0
    
    # Calculate mean_candle_raw for each row
    for i in range(lookback_window, len(df)):
        start_idx = max(0, i - lookback_window)
        df.at[i, 'mean_candle_raw'] = df.iloc[start_idx:i]['candle_raw'].abs().mean()
    
    for index in range(lookback_window, len(df) - window_size):
        row = df.iloc[index]
        
        # Use the pre-calculated mean_candle_raw from the column
        mean_candle_raw = row['mean_candle_raw']
        acceptable_candle_raw = mean_candle_raw * cur_candle_multiplier
        
        # Filter based on raw candle size

        if row['close_normalized'] < positive_slope or row['close_normalized'] > negative_slope:
            continue
        
        if row['hour_of_day'] < starting_hour or row['hour_of_day'] >= ending_hour:
            continue

        if abs(row['candle_raw']) < acceptable_candle_raw:
            continue
        df.at[index, 'include'] = 1
        # Set target and end signals based on raw close price using the column value
        target_signal = row['close'] + (mean_candle_raw * mean_multiplier)
        end_signal = row['close'] - (mean_candle_raw * drawdown)
        
        for mini_index in range(index + 1, index + window_size):
            mini_row = df.iloc[mini_index]
            
            # Exit if price drops below end signal
            if mini_row['low'] < end_signal:
                break
                
            # Check if target is reached using raw price
            if mini_row['close'] > target_signal:
                df.at[index, 'target'] = 1
                break
    
    return df[lookback_window:len(df) - window_size]


def enhanced_labeling(df, base_window=90, min_window=30, max_window=180,
                     gain_multiplier=2.5, drawdown_multiplier=0.3,
                     volatility_threshold=0.7, volatility_window=64,
                     starting_hour=8, ending_hour=16):
    """
    Enhanced labeling with:
    - Volatility-adaptive thresholds using True Range
    - Probabilistic labeling (0-1)
    - Dynamic holding periods
    - Volatility regime filters
    - Multi-scale volatility adjustment
    """
    df = df.copy().reset_index(drop=True)
    
    # 1. Calculate True Range (superior to candle size)
    df['prev_close'] = df['close'].shift(1)
    df['TR'] = df.apply(lambda x: max(x['high'] - x['low'], 
                                     abs(x['high'] - x['prev_close']),
                                     abs(x['low'] - x['prev_close'])), axis=1)
    
    # 2. Multi-scale volatility calculation
    df['ATR_short'] = df['TR'].rolling(window=16, min_periods=1).mean()
    df['ATR_medium'] = df['TR'].rolling(window=64, min_periods=1).mean()
    df['vol_ratio'] = df['ATR_short'] / df['ATR_medium']
    
    # 3. Initialize output columns
    df['target'] = np.nan
    df['include'] = 0
    
    # 4. Main processing loop
    for i in range(volatility_window, len(df) - max_window):
        # Market hours filter
        if not (starting_hour <= df.at[i, 'hour_of_day'] < ending_hour):
            continue
            
        # Volatility regime filter (low-volatility conditions)
        if df.at[i, 'vol_ratio'] > volatility_threshold:
            continue
            
        # Dynamic holding period based on volatility
        vol_factor = np.sqrt(df.at[i, 'vol_ratio'])
        holding_period = int(np.clip(base_window / max(vol_factor, 0.01), 
                             min_window, max_window))
        
        # Adaptive thresholds using current volatility
        atr = df.at[i, 'ATR_short']
        base_price = df.at[i, 'close']
        target_price = base_price + atr * gain_multiplier
        stop_price = base_price - atr * drawdown_multiplier
        
        # Path simulation with time decay weights
        max_return = 0
        time_weights = np.linspace(1.0, 0.1, holding_period)
        
        for j in range(1, holding_period + 1):
            idx = i + j
            current_high = df.at[idx, 'high']
            current_low = df.at[idx, 'low']
            
            # Path validation - momentum preservation
            if current_low < stop_price:
                break
                
            # Calculate weighted return
            candle_return = (current_high - base_price) * time_weights[j-1]
            if candle_return > max_return:
                max_return = candle_return
        
        # 5. Probabilistic labeling
        target_return = (target_price - base_price) * np.mean(time_weights)
        label_value = min(1.0, max(0.0, max_return / target_return))
        
        df.at[i, 'include'] = 1
        df.at[i, 'target'] = label_value
    
    return df[volatility_window:-max_window]

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