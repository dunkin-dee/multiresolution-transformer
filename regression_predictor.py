
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from constants.global_constants import NORMALIZING_WINDOW_SIZE, FEATURES, NUM_TOKENS, OTHER_TOKENS, R_D_MODEL, R_FF_DIM, R_NUM_HEADS
from working_data import normalize_by_window, add_timing, add_partial_hour_ohlc, normalize_partial_hour
import json
from modeler import create_regression_model
from datetime import datetime, timedelta


try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set: {policy.name}")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

base_win_path = '/mnt/c/Users/dimad/projects/trader'

instruments_path = os.path.join(base_win_path, 'instruments.json')

# with open(instruments_path, 'r') as f:
#     instruments_dict = json.load(f)
# instruments = list(instruments_dict.keys())
instruments = ['EURUSD#']

MAIN_LOOKBACK_TOKENS = NUM_TOKENS  # Update this to match your training config
HOURLY_LOOKBACK_TOKENS = OTHER_TOKENS  # Update this to match your training config

def create_inference_batch(main_df, hourly_df, main_lookback_tokens, hourly_lookback_tokens, feature_columns):
    """
    Create a batch of size 1 for inference, matching the training data format.
    
    Args:
        main_df: Main timeframe dataframe (5-minute data)
        hourly_df: Hourly timeframe dataframe
        main_lookback_tokens: Number of main sequence tokens (from training config)
        hourly_lookback_tokens: Number of hourly sequence tokens (from training config)
        feature_columns: List of feature column names
    
    Returns:
        Tuple of (main_input, hourly_input, partial_hour_data, minutes_into_hour, partial_hour_length) ready for model inference
    """
    # Extract main sequence features (most recent main_lookback_tokens)
    main_features = main_df[feature_columns].iloc[-main_lookback_tokens:].values
    
    # Extract hourly sequence features (most recent hourly_lookback_tokens)
    hourly_features = hourly_df[feature_columns].iloc[-hourly_lookback_tokens:].values
    
    # Extract partial hour data (current values only, matching training format)
    partial_hour_cols = ['partial_open_normalized', 'partial_high_normalized', 
                        'partial_low_normalized', 'partial_close_normalized']
    partial_hour_data = main_df[partial_hour_cols].iloc[-1:].values  # Shape: (1, 4)
    
    # Extract temporal context from the most recent row
    current_row = main_df.iloc[-1]
    minutes_into_hour = np.array([[current_row['position_in_hour']]], dtype=np.float32)  # Shape: (1, 1)
    partial_hour_length = np.array([[current_row['partial_hour_length']]], dtype=np.float32)  # Shape: (1, 1)

    # Add batch dimension (batch_size = 1)
    main_batch = np.expand_dims(main_features, axis=0).astype(np.float32)  # Shape: (1, main_lookback_tokens, features)
    hourly_batch = np.expand_dims(hourly_features, axis=0).astype(np.float32)  # Shape: (1, hourly_lookback_tokens, features)
    
    # Fix: Add sequence dimension to partial_hour_data to match expected shape (None, 1, 4)
    partial_hour_batch = np.expand_dims(partial_hour_data, axis=1).astype(np.float32)  # Shape: (1, 1, 4)
    
    # Fix: Add sequence dimension to temporal features to match expected shapes
    minutes_batch = np.expand_dims(minutes_into_hour, axis=1).astype(np.float32)  # Shape: (1, 1, 1)
    length_batch = np.expand_dims(partial_hour_length, axis=1).astype(np.float32)  # Shape: (1, 1, 1)
    
    # Convert to TensorFlow tensors
    main_tensor = tf.convert_to_tensor(main_batch, dtype=tf.float32)
    hourly_tensor = tf.convert_to_tensor(hourly_batch, dtype=tf.float32)
    partial_hour_tensor = tf.convert_to_tensor(partial_hour_batch, dtype=tf.float32)
    minutes_tensor = tf.convert_to_tensor(minutes_batch, dtype=tf.float32)
    length_tensor = tf.convert_to_tensor(length_batch, dtype=tf.float32)
    
    return main_tensor, hourly_tensor, partial_hour_tensor, minutes_tensor, length_tensor

def get_inference_data(main_path, hourly_path, main_lookback_tokens, hourly_lookback_tokens, feature_columns):

    df = pd.read_csv(main_path)

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
    hour_df = pd.read_csv(hourly_path)
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
    hour_df['time'] = pd.to_datetime(hour_df['time'], unit='s')
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    break_off_time = df.iloc[-1]['time'].replace(minute=0)
    hour_df = hour_df[hour_df['time'] < break_off_time]


    ##### checking past 
    df = df.sort_values('time').reset_index(drop=True)
    hour_df = hour_df.sort_values('time').reset_index(drop=True)

    # earliest index where we can form a full main lookback
    start_idx = max(main_lookback_tokens - 1, 0)
    end_idx = len(df) - 1

    main_inputs = []
    hourly_inputs = []
    partial_hour_inputs = []
    minutes_inputs = []
    length_inputs = []
    times = []
    prev_closes = []
    closes = []
    opens = []
    prev_opens = []
    cns = []
    candles = []

    for idx in range(start_idx, end_idx + 1):
        # current timestamp is the df row at idx
        current_time = df.iloc[idx]['time']

        # all rows up to and including idx
        mini_df = df.iloc[: idx + 1]

        # form hourly break_off_time (hour floor of the latest mini_df time)
        break_off_time = mini_df.iloc[-1]['time'].replace(minute=0, second=0, microsecond=0)

        # hourly history strictly before the hour containing latest mini_df time
        mini_hour_df = hour_df[hour_df['time'] < break_off_time]

        # defensive checks: ensure we have enough rows to create inputs
        if len(mini_df) < main_lookback_tokens:
            # not enough main history yet
            continue
        if len(mini_hour_df) < hourly_lookback_tokens:
            # not enough hourly history yet
            continue

        main_input, hourly_input, partial_hour_input, minutes_input, length_input = create_inference_batch(
            main_df=mini_df,
            hourly_df=mini_hour_df,
            main_lookback_tokens=main_lookback_tokens,
            hourly_lookback_tokens=hourly_lookback_tokens,
            feature_columns=feature_columns
        )

        main_inputs.append(main_input)
        hourly_inputs.append(hourly_input)
        partial_hour_inputs.append(partial_hour_input)
        minutes_inputs.append(minutes_input)
        length_inputs.append(length_input)
        times.append(current_time)

        # gather debug / metrics consistent with your original code
        closes.append(mini_df.iloc[-1]['close'])
        prev_closes.append(mini_df.iloc[-2]['close'])
        opens.append(mini_df.iloc[-1]['open'])
        prev_opens.append(mini_df.iloc[-2]['open'])
        cns.append(mini_df.iloc[-1]['close_normalized'])
        candles.append(mini_df.iloc[-1]['close_normalized'] - mini_df.iloc[-1]['open_normalized'])

    return main_inputs, hourly_inputs, partial_hour_inputs, minutes_inputs, length_inputs, times, closes, prev_closes, opens, prev_opens, cns, candles
        
    

    
    # Check if we have enough data
    if len(df) < main_lookback_tokens:
        print(f"Warning: {instrument} has insufficient main data ({len(df)} < {main_lookback_tokens})")
        return
    
    if len(hour_df) < hourly_lookback_tokens:
        print(f"Warning: {instrument} has insufficient hourly data ({len(hour_df)} < {hourly_lookback_tokens})")
        return
    
    main_input, hourly_input = create_inference_batch(
        main_df=df,
        hourly_df=hour_df,
        main_lookback_tokens=main_lookback_tokens,
        hourly_lookback_tokens=hourly_lookback_tokens,
        feature_columns=feature_columns
    )




model = create_regression_model(feature_cols=FEATURES, d_model=R_D_MODEL, num_heads=R_NUM_HEADS, ff_dim=R_FF_DIM,
                                num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS, training=False)


for instrument in instruments:
    print(f"Processing {instrument}...")

    main_path = os.path.join(base_win_path, 'checking_data', instrument, 'five_minutes.csv')
    hourly_path = os.path.join(base_win_path, 'checking_data', instrument, 'hours.csv')
    
    # Updated to handle all 5 returned components
    main_inputs, hourly_inputs, partial_hour_inputs, minutes_inputs, length_inputs, times, closes, prev_closes, opens, prev_opens, cns, candles = get_inference_data(
        main_path=main_path,
        hourly_path=hourly_path,
        main_lookback_tokens=MAIN_LOOKBACK_TOKENS,
        hourly_lookback_tokens=HOURLY_LOOKBACK_TOKENS,
        feature_columns=FEATURES
    )
    
    model.load_weights(f"models/regressor_{instrument}.keras")
    
    for i in range(len(main_inputs)):
        main_input = main_inputs[i]
        hourly_input = hourly_inputs[i]
        partial_hour_input = partial_hour_inputs[i]
        minutes_input = minutes_inputs[i]
        length_input = length_inputs[i]
        
        # Apply your filtering conditions
        condition1 = (closes[i] > opens[i]) and (prev_closes[i] > prev_opens[i])
        
        # Condition 2: Current close is above previous open
        condition2 = closes[i] > prev_opens[i]
        if not (condition1 or condition2):
            continue
        if cns[i] > 0.7 or cns[i] < 0.3:
            continue
        if times[i].hour < 10 or times[i].hour > 18:
            continue
            
        # Model prediction with all 5 inputs
        prediction = model.predict([
            main_input, 
            hourly_input, 
            partial_hour_input, 
            minutes_input, 
            length_input
        ], verbose=0)
        
        if prediction['target_high'][0][0] > 9:
            print(prediction['target_high'][0][0], instrument)
            print('At:', times[i])
            print('='*10, '\n\n')