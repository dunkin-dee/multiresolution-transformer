
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from constants.global_constants import NORMALIZING_WINDOW_SIZE, FEATURES, NUM_TOKENS, OTHER_TOKENS
from working_data import normalize_by_window, add_timing
import json
# from modeler import create_model
from datetime import datetime, timedelta


try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set: {policy.name}")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

base_win_path = '/mnt/c/Users/dimad/projects/trader'

instruments_path = os.path.join(base_win_path, 'instruments.json')

with open(instruments_path, 'r') as f:
    instruments_dict = json.load(f)
instruments = list(instruments_dict.keys())

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
        Tuple of (main_input, hourly_input) ready for model inference
    """
    # Extract main sequence features (most recent main_lookback_tokens)
    main_features = main_df[feature_columns].iloc[-main_lookback_tokens:].values
    
    # Extract hourly sequence features (most recent hourly_lookback_tokens)
    hourly_features = hourly_df[feature_columns].iloc[-hourly_lookback_tokens:].values
    
    # Add batch dimension (batch_size = 1)
    main_batch = np.expand_dims(main_features, axis=0).astype(np.float32)
    hourly_batch = np.expand_dims(hourly_features, axis=0).astype(np.float32)
    
    # Convert to TensorFlow tensors
    main_tensor = tf.convert_to_tensor(main_batch, dtype=tf.float32)
    hourly_tensor = tf.convert_to_tensor(hourly_batch, dtype=tf.float32)
    
    return main_tensor, hourly_tensor

def get_inference_data(main_path, hourly_path, main_lookback_tokens, hourly_lookback_tokens, feature_columns):
    df = pd.read_csv(main_path)

    df['time'] = pd.to_datetime(df['time'], unit='s')

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
    hour_df['time'] = pd.to_datetime(hour_df['time'], unit='s')
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
    
    break_off_time = df.iloc[-1]['time'].replace(minute=0)
    hour_df = hour_df[hour_df['time'] < break_off_time]


    ##### checking past 
    start_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0) - timedelta(days=20)
    end_time = datetime.now().replace(hour=18) - timedelta(days=1)

    current_time = start_time

    main_inputs = []
    hourly_inputs = []
    times = []
    closes = []
    cns = []
    candles = []
    while current_time <= end_time:

        mini_df = df[df['time'] <= current_time]
        break_off_time = mini_df.iloc[-1]['time'].replace(minute=0)
        mini_hour_df = hour_df[hour_df['time'] < break_off_time]

        main_input, hourly_input = create_inference_batch(
            main_df=mini_df,
            hourly_df=mini_hour_df,
            main_lookback_tokens=main_lookback_tokens,
            hourly_lookback_tokens=hourly_lookback_tokens,
            feature_columns=feature_columns
        )

        main_inputs.append(main_input)
        hourly_inputs.append(hourly_input)
        times.append(current_time)
        current_time += timedelta(minutes=5)
        closes.append(mini_df.iloc[-1]['close'])
        cns.append(mini_df.iloc[-1]['close_normalized'])
        candles.append(mini_df.iloc[-1]['close_normalized'] - mini_df.iloc[-1]['open_normalized'])

    return main_inputs, hourly_inputs, times, closes, cns, candles
        
    

    
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

    return main_input, hourly_input


# model = create_model(training=False)

for instrument in instruments:
    print(f"Processing {instrument}...")

    main_path = os.path.join(base_win_path, 'checking_data', instrument, 'five_minutes.csv')
    hourly_path = os.path.join(base_win_path, 'checking_data', instrument, 'hours.csv')
    

    # main_input, hourly_input = get_inference_data(
    #     main_path=main_path,
    #     hourly_path=hourly_path,
    #     main_lookback_tokens=MAIN_LOOKBACK_TOKENS,
    #     hourly_lookback_tokens=HOURLY_LOOKBACK_TOKENS,
    #     feature_columns=FEATURES
    # )

    main_inputs, hourly_inputs, times, closes, cns, candles = get_inference_data(
        main_path=main_path,
        hourly_path=hourly_path,
        main_lookback_tokens=MAIN_LOOKBACK_TOKENS,
        hourly_lookback_tokens=HOURLY_LOOKBACK_TOKENS,
        feature_columns=FEATURES
    )

    # print(main_inputs)

    # if main_input is None or hourly_input is None:
    #     continue
    
    # model.load_weights(f"models/{instrument}/middle/up/best_model.weights.h5")
    for i in range(len(main_inputs)):

        print(cns[i])
        main_input = main_inputs[i]
        hourly_input = hourly_inputs[i]
        if candles[i] < 0:
            continue
        if cns[i] > 0.7 or cns[i] < 0.3:
            continue
        if times[i].hour < 10 or times[i].hour > 18:
            continue
        continue
        prediction = model.predict([main_input, hourly_input], verbose=0)
        if prediction[0][0] > 0.5:
            print(prediction, instrument)
            print('At:', times[i])
            print('='*10, '\n\n')