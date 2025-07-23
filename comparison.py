import os
import pandas as pd
import numpy as np
import tensorflow as tf
from constants.global_constants import NORMALIZING_WINDOW_SIZE, FEATURES, NUM_TOKENS, OTHER_TOKENS
from working_data import normalize_by_window, add_timing
import json
import time
from transformer_builder import (
    LearnablePositionalEncoding, 
    StochasticGatedTransformerBlock, 
    AddTypeEmbedding, 
    AttentionPooling
)
from losses import recommended_trading_loss
from modeler import create_model

custom_objects = {
        'LearnablePositionalEncoding': LearnablePositionalEncoding,
        'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
        'AddTypeEmbedding': AddTypeEmbedding,
        'AttentionPooling': AttentionPooling,
        'recommended_trading_loss': recommended_trading_loss
    }

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

# Configuration - these should match your training configuration

# Process each instrument and prepare inference data
inference_data = {}

print("Preparing inference data...")
for instrument in instruments:
    print(f"Processing {instrument}...")
    
    # Load and prepare main timeframe data (5-minute)
    df = pd.read_csv(f"{base_win_path}/data/{instrument}/five_minutes.csv")
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
    
    # Load and prepare hourly data
    hour_df = pd.read_csv(f"{base_win_path}/data/{instrument}/hours.csv")
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
    
    # Check if we have enough data
    if len(df) < MAIN_LOOKBACK_TOKENS:
        print(f"Warning: {instrument} has insufficient main data ({len(df)} < {MAIN_LOOKBACK_TOKENS})")
        continue
    
    if len(hour_df) < HOURLY_LOOKBACK_TOKENS:
        print(f"Warning: {instrument} has insufficient hourly data ({len(hour_df)} < {HOURLY_LOOKBACK_TOKENS})")
        continue
    
    # Create inference batch
    main_input, hourly_input = create_inference_batch(
        main_df=df,
        hourly_df=hour_df,
        main_lookback_tokens=MAIN_LOOKBACK_TOKENS,
        hourly_lookback_tokens=HOURLY_LOOKBACK_TOKENS,
        feature_columns=FEATURES
    )
    
    inference_data[instrument] = {
        'main_input': main_input,
        'hourly_input': hourly_input
    }

print(f"Prepared inference data for {len(inference_data)} instruments")

# Method 1: Loading complete model files (.keras)
print("\n" + "="*60)
print("METHOD 1: Loading complete model files (.keras)")
print("="*60)

method1_times = {}
method1_total_start = time.time()

for instrument in inference_data.keys():
    print(f"Processing {instrument} with method 1...")
    
    start_time = time.time()
    
    # Load complete model
    model = tf.keras.models.load_model(f"models/{instrument}/middle/up/best_model.keras", custom_objects=custom_objects)
    
    # Make prediction
    prediction = model.predict([inference_data[instrument]['main_input'], inference_data[instrument]['hourly_input']])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    method1_times[instrument] = elapsed_time
    
    print(f"  Time: {elapsed_time:.4f}s")
    print(f"  Prediction: {prediction}")

method1_total_time = time.time() - method1_total_start

# Method 2: Create model once, then load weights
print("\n" + "="*60)
print("METHOD 2: Create model once, then load weights")
print("="*60)

method2_times = {}
method2_total_start = time.time()

# Create model once before the loop
print("Creating model architecture...")
model_creation_start = time.time()
model = create_model(training=False)
model_creation_time = time.time() - model_creation_start
print(f"Model creation time: {model_creation_time:.4f}s")

for instrument in inference_data.keys():
    print(f"Processing {instrument} with method 2...")
    
    start_time = time.time()
    
    # Load weights only
    model.load_weights(f"models/{instrument}/middle/up/best_model.weights.h5")
    
    # Make prediction
    prediction = model.predict([inference_data[instrument]['main_input'], inference_data[instrument]['hourly_input']])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    method2_times[instrument] = elapsed_time
    
    print(f"  Time: {elapsed_time:.4f}s")
    print(f"  Prediction: {prediction}")

method2_total_time = time.time() - method2_total_start

# Performance comparison
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

print(f"Method 1 (load .keras) total time: {method1_total_time:.4f}s")
print(f"Method 2 (create + load weights) total time: {method2_total_time:.4f}s")
print(f"Method 2 model creation overhead: {model_creation_time:.4f}s")
print(f"Method 2 net inference time: {method2_total_time - model_creation_time:.4f}s")

print(f"\nSpeedup: {method1_total_time / method2_total_time:.2f}x")
if method2_total_time < method1_total_time:
    print(f"Method 2 is {((method1_total_time - method2_total_time) / method1_total_time) * 100:.1f}% faster")
else:
    print(f"Method 1 is {((method2_total_time - method1_total_time) / method1_total_time) * 100:.1f}% faster")

print(f"\nPer-instrument timing comparison:")
print(f"{'Instrument':<15} {'Method 1 (s)':<12} {'Method 2 (s)':<12} {'Speedup':<8}")
print("-" * 55)

for instrument in inference_data.keys():
    m1_time = method1_times[instrument]
    m2_time = method2_times[instrument]
    speedup = m1_time / m2_time if m2_time > 0 else float('inf')
    print(f"{instrument:<15} {m1_time:<12.4f} {m2_time:<12.4f} {speedup:<8.2f}x")

avg_m1_time = sum(method1_times.values()) / len(method1_times)
avg_m2_time = sum(method2_times.values()) / len(method2_times)
print("-" * 55)
print(f"{'Average':<15} {avg_m1_time:<12.4f} {avg_m2_time:<12.4f} {avg_m1_time/avg_m2_time:<8.2f}x")