import os
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
from constants.global_constants import *
from modeler import create_regression_model
from transformer_builder import WarmupCosineDecay
from regression_losses import asymmetric_huber_loss_single, profit_precision_metric, profit_recall_metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gradient_monitor import GradientAndWeightMonitor, BranchScalingMonitor


starting_dir = "data/final_data"
working_path = "data/experimenting"

import os
from generators.regression_multi_instrument_data_generator import InstrumentConfig, MultiInstrumentDatasetConfig, create_multi_instrument_dataset
from constants.global_constants import FEATURES, NUM_TOKENS, OTHER_TOKENS, BATCH_SIZE, LOOKBACK_WINDOW


instruments = os.listdir(working_path)
feature_cols = FEATURES

def get_datasets_and_steps(instruments=instruments, working_path=working_path, feature_cols=feature_cols):
    train_instrument_configs = []
    val_instrument_configs = []
    test_instrument_configs = []

    for instrument in instruments:
        train_instrument_configs.append(
            InstrumentConfig(
                name=instrument,
                hourly_data_path=f"{working_path}/{instrument}/hour.csv",
                chunked_data_dir=f"{working_path}/{instrument}/training"
            )
        )
        val_instrument_configs.append(
            InstrumentConfig(
                name=instrument,
                hourly_data_path=f"{working_path}/{instrument}/hour.csv",
                chunked_data_dir=f"{working_path}/{instrument}/validation"
            )
        )
        test_instrument_configs.append(
            InstrumentConfig(
                name=instrument,
                hourly_data_path=f"{working_path}/{instrument}/hour.csv",
                chunked_data_dir=f"{working_path}/{instrument}/testing"
            )
        )

    train_config = MultiInstrumentDatasetConfig(
        instruments=train_instrument_configs,
        main_lookback_tokens=NUM_TOKENS,
        hourly_lookback_tokens=OTHER_TOKENS,
        lookback_window=LOOKBACK_WINDOW,
        batch_size=BATCH_SIZE,
        shuffle_data=True,
        feature_columns=feature_cols,
        max_chunks_per_instrument=25,
        add_noise_5min=True,        
        add_noise_hourly=True,     
        noise_std_5min=0.01,  
        noise_std_hourly=0.015,    
        noise_probability_5min=0.8,
        noise_probability_hourly=0.8
    )

    val_config = MultiInstrumentDatasetConfig(
        instruments=val_instrument_configs,
        main_lookback_tokens=NUM_TOKENS,
        hourly_lookback_tokens=OTHER_TOKENS,
        lookback_window=LOOKBACK_WINDOW,
        batch_size=BATCH_SIZE,
        shuffle_data=False,
        feature_columns=feature_cols,
        max_chunks_per_instrument=25
    )

    test_config = MultiInstrumentDatasetConfig(
        instruments=test_instrument_configs,
        main_lookback_tokens=NUM_TOKENS,
        hourly_lookback_tokens=OTHER_TOKENS,
        lookback_window=LOOKBACK_WINDOW,
        batch_size=BATCH_SIZE,
        shuffle_data=False,
        feature_columns=feature_cols,
        max_chunks_per_instrument=25
    )


    train_dataset, train_rows = create_multi_instrument_dataset(
        config=train_config,
        repeat_dataset=True
    )
    val_dataset, val_rows = create_multi_instrument_dataset(
        config=val_config,
        repeat_dataset=True
    )
    test_dataset, test_rows = create_multi_instrument_dataset(
        config=test_config,
        repeat_dataset=True
    )

    train_steps = train_rows//BATCH_SIZE
    val_steps = val_rows//BATCH_SIZE
    test_steps = test_rows//BATCH_SIZE

    return (
        (train_dataset, val_dataset, test_dataset),
        (train_steps, val_steps, test_steps)
    )


(train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps()

def filter_targets_for_training(inputs, targets):
    """Filter targets to only include what the model expects for training"""
    filtered_targets = {
        'target_high': targets['target_high']
    }
    return inputs, filtered_targets


def compile_model_lightweight(model, updelta, downdelta):
    """
    Streamlined compilation with only the most important metrics
    Mixed precision compatible.
    """
    lr_schedule = WarmupCosineDecay(initial_lr=1e-5, warmup_steps=train_steps*2, decay_steps=train_steps*40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss={
            'target_high': 'mse'
        },
        metrics={
            'target_high': [
                'mae',
                'mse',
            ]
        }
    )
    return model




model = create_regression_model(feature_cols=feature_cols, d_model=R_D_MODEL, num_heads=R_NUM_HEADS, ff_dim=R_FF_DIM,
                                num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS, training=False)
model = compile_model_lightweight(model=model, updelta=6.0, downdelta=-1.0)
model.load_weights('models/regressor.keras')

def evaluate_with_denormalization(model, test_dataset, test_steps):
    """Custom evaluation with both normalized and denormalized comparison (MAE + MSE)."""
    
    total_denorm_mae = 0.0
    total_random_walk_mae = 0.0
    total_norm_mae = 0.0
    total_norm_random_walk_mae = 0.0

    # NEW: MSE accumulators
    total_denorm_mse = 0.0
    total_random_walk_mse = 0.0
    total_norm_mse = 0.0
    total_norm_random_walk_mse = 0.0

    batch_count = 0

    for batch in test_dataset.take(test_steps):
        (main_input, hourly_input, partial_input, minutes_input, length_input), targets = batch

        # Model predictions - match model.evaluate behavior
        predictions = model.predict((main_input, hourly_input, partial_input, minutes_input, length_input))

        # Extract prediction and target tensors
        pred_high = predictions if not isinstance(predictions, dict) else predictions['target_high']
        target_high = targets['target_high']
        norm_min = targets['norm_min']
        norm_max = targets['norm_max']
        original_close = targets['original_close']

        # Debug/reshape as you had it
        if batch_count == 0:
            print("=== DEBUGGING BATCH SHAPES AND VALUES ===")
            print(f"pred_high shape: {pred_high.shape}, first 3 values: {pred_high[:3].numpy()}")
            print(f"target_high shape: {target_high.shape}, first 3 values: {target_high[:3].numpy()}")
            print(f"pred_high dtype: {pred_high.dtype}, target_high dtype: {target_high.dtype}")
            if len(pred_high.shape) != len(target_high.shape):
                print("WARNING: Shape mismatch between predictions and targets!")
                if len(pred_high.shape) == 2 and pred_high.shape[-1] == 1:
                    print("Reshaping predictions from (batch, 1) to (batch,)")
                    pred_high = tf.squeeze(pred_high, axis=-1)
                    print(f"New pred_high shape: {pred_high.shape}")

        if len(pred_high.shape) == 2 and pred_high.shape[-1] == 1:
            pred_high = tf.squeeze(pred_high, axis=-1)

        # Normalized values (model output is normalized)
        norm_pred = pred_high
        norm_target = target_high

        # Random-walk baseline in normalized space
        norm_random_walk = (original_close - norm_min) / (norm_max - norm_min)
        norm_random_walk = tf.clip_by_value(norm_random_walk, 0.0, 1.0)

        # Denormalize
        denorm_pred = norm_pred * (norm_max - norm_min) + norm_min
        denorm_target = norm_target * (norm_max - norm_min) + norm_min
        random_walk_pred = original_close  # original_close is already denormalized

        # --- MAE ---
        batch_norm_mae = tf.reduce_mean(tf.abs(norm_pred - norm_target))
        batch_norm_random_walk_mae = tf.reduce_mean(tf.abs(norm_random_walk - norm_target))

        batch_denorm_mae = tf.reduce_mean(tf.abs(denorm_pred - denorm_target))
        batch_random_walk_mae = tf.reduce_mean(tf.abs(random_walk_pred - denorm_target))

        # --- MSE (NEW) ---
        batch_norm_mse = tf.reduce_mean(tf.square(norm_pred - norm_target))
        batch_norm_random_walk_mse = tf.reduce_mean(tf.square(norm_random_walk - norm_target))

        batch_denorm_mse = tf.reduce_mean(tf.square(denorm_pred - denorm_target))
        batch_random_walk_mse = tf.reduce_mean(tf.square(random_walk_pred - denorm_target))

        if batch_count == 0:
            print(f"First batch norm MAE: {batch_norm_mae.numpy():.6f}, norm MSE: {batch_norm_mse.numpy():.6f}")
            print(f"Model.evaluate should show similar MAE around: {batch_norm_mae.numpy():.6f}")

        # Accumulate (convert to python floats)
        total_norm_mae += batch_norm_mae.numpy()
        total_norm_random_walk_mae += batch_norm_random_walk_mae.numpy()
        total_denorm_mae += batch_denorm_mae.numpy()
        total_random_walk_mae += batch_random_walk_mae.numpy()

        total_norm_mse += batch_norm_mse.numpy()
        total_norm_random_walk_mse += batch_norm_random_walk_mse.numpy()
        total_denorm_mse += batch_denorm_mse.numpy()
        total_random_walk_mse += batch_random_walk_mse.numpy()

        batch_count += 1

    # Averages
    avg_norm_mae = total_norm_mae / batch_count
    avg_norm_random_walk_mae = total_norm_random_walk_mae / batch_count
    avg_denorm_mae = total_denorm_mae / batch_count
    avg_random_walk_mae = total_random_walk_mae / batch_count

    avg_norm_mse = total_norm_mse / batch_count
    avg_norm_random_walk_mse = total_norm_random_walk_mse / batch_count
    avg_denorm_mse = total_denorm_mse / batch_count
    avg_random_walk_mse = total_random_walk_mse / batch_count

    # Print results
    print("===== Normalized comparison =====")
    print(f"Model MAE (normalized): {avg_norm_mae:.6f}")
    print(f"Random Walk MAE (normalized): {avg_norm_random_walk_mae:.6f}")
    print(f"Improvement (MAE): {((avg_norm_random_walk_mae - avg_norm_mae) / avg_norm_random_walk_mae * 100):.2f}%")

    print(f"Model MSE (normalized): {avg_norm_mse:.6f}")
    print(f"Random Walk MSE (normalized): {avg_norm_random_walk_mse:.6f}")
    print(f"Improvement (MSE): {((avg_norm_random_walk_mse - avg_norm_mse) / avg_norm_random_walk_mse * 100):.2f}%")

    print("\n===== Denormalized comparison =====")
    print(f"Model MAE (denormalized): {avg_denorm_mae:.6f}")
    print(f"Random Walk MAE (denormalized): {avg_random_walk_mae:.6f}")
    print(f"Improvement (MAE): {((avg_random_walk_mae - avg_denorm_mae) / avg_random_walk_mae * 100):.2f}%")

    print(f"Model MSE (denormalized): {avg_denorm_mse:.6f}")
    print(f"Random Walk MSE (denormalized): {avg_random_walk_mse:.6f}")
    print(f"Improvement (MSE): {((avg_random_walk_mse - avg_denorm_mse) / avg_random_walk_mse * 100):.2f}%")

    print(f"\n===== Consistency Check =====")
    print(f"model.evaluate MAE should be around: {avg_norm_mae:.6f}")

    return {
        "normalized": {
            "mae": (avg_norm_mae, avg_norm_random_walk_mae),
            "mse": (avg_norm_mse, avg_norm_random_walk_mse)
        },
        "denormalized": {
            "mae": (avg_denorm_mae, avg_random_walk_mae),
            "mse": (avg_denorm_mse, avg_random_walk_mse)
        }
    }


def evaluate_with_denormalization_matching_keras(model, test_dataset, test_steps):
    mae_norm_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_norm_metric = tf.keras.metrics.MeanSquaredError()
    mae_denorm_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_denorm_metric = tf.keras.metrics.MeanSquaredError()

    # optional: also keep random-walk baselines
    mae_norm_rw = tf.keras.metrics.MeanAbsoluteError()
    mse_norm_rw = tf.keras.metrics.MeanSquaredError()
    mae_denorm_rw = tf.keras.metrics.MeanAbsoluteError()
    mse_denorm_rw = tf.keras.metrics.MeanSquaredError()

    batch_count = 0
    for batch in test_dataset.take(test_steps):
        (main_input, hourly_input, partial_input, minutes_input, length_input), targets = batch

        preds = model.predict((main_input, hourly_input, partial_input, minutes_input, length_input))
        pred_high = preds if not isinstance(preds, dict) else preds['target_high']
        target_high = targets['target_high']
        norm_min = targets['norm_min']
        norm_max = targets['norm_max']
        original_close = targets['original_close']

        if len(pred_high.shape) == 2 and pred_high.shape[-1] == 1:
            pred_high = tf.squeeze(pred_high, axis=-1)

        # normalized
        norm_pred = pred_high
        norm_target = target_high

        # random-walk (normalized)
        norm_random_walk = (original_close - norm_min) / (norm_max - norm_min)
        norm_random_walk = tf.clip_by_value(norm_random_walk, 0.0, 1.0)

        # denormalize
        denorm_pred = norm_pred * (norm_max - norm_min) + norm_min
        denorm_target = norm_target * (norm_max - norm_min) + norm_min
        denorm_random_walk = original_close  # already denorm

        # Update TF metrics with per-sample vectors (metrics handle sample weighting properly)
        mae_norm_metric.update_state(norm_target, norm_pred)
        mse_norm_metric.update_state(norm_target, norm_pred)
        mae_denorm_metric.update_state(denorm_target, denorm_pred)
        mse_denorm_metric.update_state(denorm_target, denorm_pred)

        mae_norm_rw.update_state(norm_target, norm_random_walk)
        mse_norm_rw.update_state(norm_target, norm_random_walk)
        mae_denorm_rw.update_state(denorm_target, denorm_random_walk)
        mse_denorm_rw.update_state(denorm_target, denorm_random_walk)

        batch_count += 1

    results = {
        "normalized": {
            "mae": (mae_norm_metric.result().numpy(), mae_norm_rw.result().numpy()),
            "mse": (mse_norm_metric.result().numpy(), mse_norm_rw.result().numpy())
        },
        "denormalized": {
            "mae": (mae_denorm_metric.result().numpy(), mae_denorm_rw.result().numpy()),
            "mse": (mse_denorm_metric.result().numpy(), mse_denorm_rw.result().numpy())
        }
    }

    print("Normalized MAE (model, RW):", results["normalized"]["mae"])
    print("Denormalized MAE (model, RW):", results["denormalized"]["mae"])
    return results


# clean_test_dataset = val_dataset.map(filter_targets_for_training)


# model.evaluate(clean_test_dataset, steps=val_steps)

(train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps()

evaluate_with_denormalization_matching_keras(model, val_dataset, val_steps)

# (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps()
# model.evaluate(clean_test_dataset, steps=val_steps)
