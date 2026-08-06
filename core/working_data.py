"""Data preparation utilities shared by preprocessing and inference.

The pipeline these functions implement, in order:

1. ``clean_five_minute_data`` / ``clean_hour_data`` — discard rows that do not sit
   on a regular grid (broker feeds contain gaps, weekend stubs and duplicates).
2. ``add_timing`` — derive the calendar features the model conditions on.
3. ``add_partial_hour_ohlc`` — build the *incomplete current hour* OHLC, so the
   model can see the hour it is standing inside rather than only closed hours.
4. ``normalize_by_window`` — window min/max normalisation (see README).
5. ``normalize_partial_hour`` — put the partial hour on the hourly scale.
6. ``regression_label_df_next`` — forward-looking high/low regression labels.
7. ``split_multiresolution_chunks`` — leakage-safe chunked train/val/test split.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_five_minute_data(df):
    """
    Clean OHLC data to keep only entries that follow 5-minute intervals.
    
    Args:
        df: DataFrame with 'time' column containing Unix timestamps
        
    Returns:
        DataFrame with only 5-minute interval entries
    """
    # Convert time to datetime for easier analysis
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    
    # Sort by time to ensure proper order
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate time differences between consecutive entries (in seconds)
    df['time_diff'] = df['time'].diff()
    
    # 5 minutes = 300 seconds
    five_minutes = 300
    
    # Find the first entry that starts a consistent 5-minute pattern
    start_idx = None
    
    # Look for at least 3 consecutive 5-minute intervals to confirm the pattern
    for i in range(1, len(df) - 2):
        if (df.loc[i, 'time_diff'] == five_minutes and 
            df.loc[i+1, 'time_diff'] == five_minutes and 
            df.loc[i+2, 'time_diff'] == five_minutes):
            start_idx = i
            break
    
    if start_idx is None:
        # If no clear 5-minute pattern found, try a more flexible approach
        # Find the most common time difference that's close to 300 seconds
        time_diffs = df['time_diff'].dropna()
        # Filter for differences between 250-350 seconds (allowing some tolerance)
        valid_diffs = time_diffs[(time_diffs >= 250) & (time_diffs <= 350)]
        
        if len(valid_diffs) > 0:
            # Use the most common interval as our target
            target_interval = valid_diffs.mode().iloc[0] if len(valid_diffs.mode()) > 0 else five_minutes
            
            # Find first occurrence of this interval
            for i, diff in enumerate(time_diffs):
                if abs(diff - target_interval) <= 50:  # 50 second tolerance
                    start_idx = i + 1  # +1 because diff() shifts indices
                    break
    
    if start_idx is None:
        print("Warning: Could not find a clear 5-minute pattern. Returning original data.")
        return df.drop(['datetime', 'time_diff'], axis=1)
    
    # Keep only data from the start of the 5-minute pattern
    cleaned_df = df.iloc[start_idx:].copy()
    
    # Optional: Further filter to keep only entries with proper 5-minute intervals
    # This removes any outliers within the main data
    cleaned_df['time_diff_clean'] = cleaned_df['time'].diff()
    
    # Keep first row and rows with proper 5-minute intervals (with some tolerance)
    mask = (cleaned_df['time_diff_clean'].isna()) | (abs(cleaned_df['time_diff_clean'] - five_minutes) <= 60)
    cleaned_df = cleaned_df[mask]
    
    # Clean up temporary columns
    cleaned_df = cleaned_df.drop(['datetime', 'time_diff', 'time_diff_clean'], axis=1)
    
    return cleaned_df.reset_index(drop=True)

def clean_hour_data(df):
    """
    Clean OHLC data to keep only entries that follow hour intervals.
    
    Args:
        df: DataFrame with 'time' column containing Unix timestamps
        
    Returns:
        DataFrame with only hour interval entries
    """
    # Convert time to datetime for easier analysis
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    
    # Sort by time to ensure proper order
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate time differences between consecutive entries (in seconds)
    df['time_diff'] = df['time'].diff()
    
    hour = 3600
    
    # Find the first entry that starts a consistent 5-minute pattern
    start_idx = None
    
    # Look for at least 3 consecutive hour intervals to confirm the pattern
    for i in range(1, len(df) - 2):
        if (df.loc[i, 'time_diff'] == hour and 
            df.loc[i+1, 'time_diff'] == hour and 
            df.loc[i+2, 'time_diff'] == hour):
            start_idx = i
            break
    
    if start_idx is None:
        # If no clear hour pattern found, try a more flexible approach
        # Find the most common time difference that's close to 300 seconds
        time_diffs = df['time_diff'].dropna()
        # Filter for differences between 250-350 seconds (allowing some tolerance)
        valid_diffs = time_diffs[(time_diffs >= 3500) & (time_diffs <= 3700)]
        
        if len(valid_diffs) > 0:
            # Use the most common interval as our target
            target_interval = valid_diffs.mode().iloc[0] if len(valid_diffs.mode()) > 0 else hour
            
            # Find first occurrence of this interval
            for i, diff in enumerate(time_diffs):
                if abs(diff - target_interval) <= 50:  # 50 second tolerance
                    start_idx = i + 1  # +1 because diff() shifts indices
                    break
    
    if start_idx is None:
        print("Warning: Could not find a clear hour pattern. Returning original data.")
        return df.drop(['datetime', 'time_diff'], axis=1)
    
    cleaned_df = df.iloc[start_idx:].copy()
    
    # Optional: Further filter to keep only entries with proper hour intervals
    # This removes any outliers within the main data
    cleaned_df['time_diff_clean'] = cleaned_df['time'].diff()
    
    # Keep first row and rows with proper hour intervals (with some tolerance)
    mask = (cleaned_df['time_diff_clean'].isna()) | (abs(cleaned_df['time_diff_clean'] - hour) <= 60)
    cleaned_df = cleaned_df[mask]
    
    # Clean up temporary columns
    cleaned_df = cleaned_df.drop(['datetime', 'time_diff', 'time_diff_clean'], axis=1)
    
    return cleaned_df.reset_index(drop=True)

def add_timing(df):
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['hour_of_day'] = df['datetime'].dt.hour
    df['position_in_hour'] = df['datetime'].dt.minute
    df['partial_hour_length'] = df['datetime'].dt.minute * 0.2
    return df

def add_partial_hour_ohlc(df):
    # make sure 'datetime' exists
    if 'datetime' not in df:
        df = add_timing(df)

    # create an "hour bucket"
    df['hour_start'] = df['datetime'].dt.floor('H')

    # first open of the hour
    df['partial_open'] = df.groupby('hour_start')['open'].transform('first')
    # running high
    df['partial_high'] = df.groupby('hour_start')['high'].cummax()
    # running low
    df['partial_low'] = df.groupby('hour_start')['low'].cummin()
    # current close is already the "partial close"
    df['partial_close'] = df['close']

    # if you don't need the helper column any more
    df.drop(columns=['hour_start'], inplace=True)

    return df

def normalize_partial_hour(df,
                           hour_df, 
                           features=[
                               'partial_open', 
                               'partial_high', 
                               'partial_low', 
                               'partial_close'
                               ],
                            high_col='partial_high',
                            low_col='partial_low',
                            hour_high='for_partial_hour_window_max',
                            hour_low='for_partial_hour_window_min'):
    # 1. Make a copy so we don’t clobber the originals
    df = df.copy()
    hour_df = hour_df.copy()

    # 2. Create the “base hour” column on each
    df['base_hour'] = df['datetime'].dt.floor('H')
    hour_df['hour_datetime'] = hour_df['datetime']

    merged = pd.merge_asof(
        left=df,
        right=hour_df[['hour_datetime', hour_high, hour_low]],
        left_on='base_hour',
        right_on='hour_datetime',
        direction='backward'
    )

    merged['partial_max'] = merged[[high_col, hour_high]].max(axis=1)
    merged['partial_min'] = merged[[low_col, hour_low]].min(axis=1)

    for feature in features:
        merged[f"{feature}_normalized"] = merged[feature] = (merged[feature] - merged['partial_min']) / (merged['partial_max'] - merged['partial_min'])
    drop_cols = ['base_hour', 'hour_datetime', 'partial_max', 'partial_min']

    merged.drop(columns=drop_cols, inplace=True)
    return merged



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
        label_cols=[],
        add_partial_hour=False,
        keep_normalization_params=True):
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
    if add_partial_hour:
        df['for_partial_hour_window_min'] = df[low_col]
        df['for_partial_hour_window_max'] = df[high_col]
        partial_exclude = col_numbers[-1]

    for col_range in col_split:
        high_cols = []
        low_cols = []
        partial_high_cols = []
        partial_low_cols = []

        for x in col_range:
            high_col_name = f"{high_col}-{x}"
            df[high_col_name] = df[high_col].shift(x)
            high_cols.append(high_col_name)
            if add_partial_hour and x != partial_exclude:
                partial_high_cols.append(high_col_name)
        high_cols_inclusive = high_cols + ['window_max']
        df['window_max'] = df[high_cols_inclusive].max(axis=1)
        if add_partial_hour:
            partial_high_cols_inclusive = partial_high_cols + ['for_partial_hour_window_max']
            df['for_partial_hour_window_max'] = df[partial_high_cols_inclusive].max(axis=1)
        df.drop(columns=high_cols, inplace=True)

        for x in col_range:
            low_col_name = f"{low_col}-{x}"
            df[low_col_name] = df[low_col].shift(x)
            low_cols.append(low_col_name)
            if add_partial_hour and x != partial_exclude:
                partial_low_cols.append(low_col_name)
        low_cols_inclusive = low_cols + ['window_min']
        df['window_min'] = df[low_cols_inclusive].min(axis=1)
        if add_partial_hour:
            partial_low_cols_inclusive = partial_low_cols + ['for_partial_hour_window_min']
            df['for_partial_hour_window_min'] = df[partial_low_cols_inclusive].min(axis=1)
        df.drop(columns=low_cols, inplace=True)

    # Normalize the main columns using current window
    for normalizing_col in normalizing_cols:
        df[f"{normalizing_col}_normalized"] = (df[normalizing_col] - df['window_min'])/(df['window_max'] - df['window_min'])

    if keep_normalization_params:
        # Keep these for denormalization later
        df['norm_window_min'] = df['window_min']
        df['norm_window_max'] = df['window_max']

    # Handle label normalization only if label_cols is not empty
    if label_cols:
        df['window_max_prev'] = df['window_max'].shift(1)
        df['window_min_prev'] = df['window_min'].shift(1)

        for label_col in label_cols:
            df[f"{label_col}_normalized_for_label"] = (df[label_col] - df['window_min_prev'])/(df['window_max_prev'] - df['window_min_prev'])
        
        df.drop(columns=['window_max_prev', 'window_min_prev'], inplace=True)

    return df[window_size:]

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
            if mini_row['close_normalized'] >= target_signal:
                target_hit = True
                break
        
        if target_hit:
            df.at[index, 'target'] = 1
    
    # Remove the temporary candle column and return
    df = df.drop(columns=['candle'])
    return df.iloc[:len(df) - window_size].copy()

def regression_label_df_next(df, window_size=12):
    """
    Compute regression labels over a forward-looking window.

    target_high = max(high_normalized_for_label) over the next window_size candles
    target_low  = min(low_normalized_for_label)  over the next window_size candles

    Rows where the full forward window is unavailable (last window_size rows) are
    marked include=0 so the generator skips them.
    """
    df = df.copy()
    df = df.reset_index(drop=True)

    shifted_highs = pd.concat(
        [df['high_normalized_for_label'].shift(-i) for i in range(1, window_size + 1)],
        axis=1
    )
    shifted_lows = pd.concat(
        [df['low_normalized_for_label'].shift(-i) for i in range(1, window_size + 1)],
        axis=1
    )

    df['target_high'] = shifted_highs.max(axis=1)
    df['target_low'] = shifted_lows.min(axis=1)
    df['include'] = 1
    df.iloc[-window_size:, df.columns.get_loc('include')] = 0

    return df


def split_multiresolution_chunks(
    df_5min, 
    df_hour, 
    dump_path, 
    chunk_size=10000,
    hour_lookback=30,  # other_tokens parameter from your prepare_data function
    train_size=0.7,
    lookback=30,  # 5-minute lookback for validation/test splits
    cols=['time', 'open', 'high', 'low', 'close']
):
    """
    Split 5-minute dataframe into chunks while preventing data leakage with hourly data.
    
    Parameters:
    -----------
    df_5min : pd.DataFrame
        Main 5-minute OHLC dataframe to be split
    df_hour : pd.DataFrame  
        Hourly OHLC dataframe used for reference to prevent data leakage
    dump_path : str
        Base path where chunk folders will be created
    chunk_size : int
        Maximum size of each chunk
    hour_lookback : int
        Number of hourly periods to look back (other_tokens from prepare_data)
    train_size : float
        Proportion of data for training (0.7 = 70%)
    lookback : int
        Lookback window for validation/test adjustments
    cols : list
        Columns to include in the output CSV files
    
    Returns:
    --------
    dict: Summary of chunks created
    """
    
    # Create base directories
    os.makedirs(dump_path, exist_ok=True)
    for split_type in ['training', 'validation', 'testing']:
        os.makedirs(os.path.join(dump_path, split_type), exist_ok=True)
    
    # Filter columns
    df_5min_filtered = df_5min[cols].copy()
    df_hour_filtered = df_hour[['time']].copy()  # Only need time for reference
    
    # Sort both dataframes by time to ensure proper ordering
    df_5min_filtered = df_5min_filtered.sort_values('time').reset_index(drop=True)
    df_hour_filtered = df_hour_filtered.sort_values('time').reset_index(drop=True)
    
    # Convert time columns to numpy arrays for efficient searching
    hour_times = df_hour_filtered['time'].values
    min5_times = df_5min_filtered['time'].values
    
    total_rows = len(df_5min_filtered)
    chunk_info = []
    
    # Calculate chunks
    start_idx = 0
    chunk_num = 0
    
    while start_idx < total_rows:
        # Calculate end index for this chunk
        end_idx = min(start_idx + chunk_size, total_rows)
        
        # Get the time range for this chunk
        chunk_start_time = min5_times[start_idx]
        chunk_end_time = min5_times[end_idx - 1]
        
        # Find the corresponding hour index for the chunk start time
        # Use searchsorted to find where chunk_start_time would fit in hour_times
        hour_start_pos = np.searchsorted(hour_times, chunk_start_time, side='right') - 1
        
        # Calculate how much 5-minute data we need to discard from the start
        # to ensure no bleeding with previous chunks when using hour_lookback
        if chunk_num > 0:  # Skip adjustment for first chunk
            # We need to ensure that when we look back hour_lookback periods
            # from any point in this chunk, we don't access data from previous chunks
            
            # Find the earliest hour time that would be accessed by hour_lookback
            earliest_hour_idx = max(0, hour_start_pos - hour_lookback + 1)
            earliest_hour_time = hour_times[earliest_hour_idx]
            
            # Find the first 5-minute index that is >= earliest_hour_time
            safe_start_idx = np.searchsorted(min5_times[start_idx:end_idx], 
                                           earliest_hour_time, side='left') + start_idx
            
            # Adjust start_idx to ensure safety margin
            start_idx = max(start_idx, safe_start_idx)
            
            # Recalculate end_idx if start_idx changed
            end_idx = min(start_idx + chunk_size, total_rows)
        
        # Skip if chunk is too small after adjustments
        if end_idx - start_idx < lookback * 3:  # Need minimum data for train/val/test
            print(f"Skipping chunk {chunk_num}: too small after safety adjustments")
            break
            
        # Extract chunk data
        chunk_data = df_5min_filtered.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        
        # Split chunk into train/val/test
        train_data, temp_data = train_test_split(
            chunk_data, 
            test_size=1-train_size, 
            shuffle=False
        )
        
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=0.50, 
            shuffle=False
        )
        
        # Apply lookback adjustments to val and test data
        val_data = val_data.iloc[lookback:].reset_index(drop=True)
        test_data = test_data.iloc[lookback:].reset_index(drop=True)
        
        # Save chunks to CSV files
        chunk_folder_name = f'chunk_{chunk_num:03d}'
        
        # Save training data
        train_path = os.path.join(dump_path, 'training', f'{chunk_folder_name}_train.csv')
        train_data.to_csv(train_path, index=False)
        
        # Save validation data (if not empty)
        if len(val_data) > 0:
            val_path = os.path.join(dump_path, 'validation', f'{chunk_folder_name}_val.csv')
            val_data.to_csv(val_path, index=False)
        
        # Save test data (if not empty)
        if len(test_data) > 0:
            test_path = os.path.join(dump_path, 'testing', f'{chunk_folder_name}_test.csv')
            test_data.to_csv(test_path, index=False)
        
        # Store chunk information
        chunk_info.append({
            'chunk_num': chunk_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_rows': end_idx - start_idx,
            'train_rows': len(train_data),
            'val_rows': len(val_data),
            'test_rows': len(test_data),
            'start_time': chunk_start_time,
            'end_time': min5_times[end_idx - 1],
            'hour_start_pos': hour_start_pos
        })

        # Move to next chunk
        start_idx = end_idx
        chunk_num += 1
    
    # Create summary
    summary = {
        'total_chunks': len(chunk_info),
        'total_original_rows': total_rows,
        'chunks_info': chunk_info,
        'dump_path': dump_path
    }
    
    return summary
