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
working_path = "data/regression"

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

def compile_model_lightweight(model, updelta, downdelta):
    """
    Streamlined compilation with only the most important metrics
    Mixed precision compatible.
    """
    lr_schedule = WarmupCosineDecay(initial_lr=5e-5, warmup_steps=train_steps*2, decay_steps=train_steps*40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5, weight_decay=1e-4),
        loss={
            'target_high': asymmetric_huber_loss_single(
                delta=2.5, 
                underestimate_weight=3.2, 
                overestimate_weight=0.6
            )
        },
        metrics={
            'target_high': [
                'mae',
                'mse',
                profit_precision_metric(threshold=6.0),
                profit_recall_metric(threshold=6.0),
            ]
        }
    )
    return model


model = create_regression_model(feature_cols=feature_cols, d_model=R_D_MODEL, num_heads=R_NUM_HEADS, ff_dim=R_FF_DIM,
                                num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS)
model = compile_model_lightweight(model=model, updelta=6.0, downdelta=-1.0)


early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=10,
                               mode='min', 
                               verbose=1)

model_checkpoint = ModelCheckpoint('models/regressor.keras', 
                                   monitor='val_loss', 
                                   save_best_only=True, 
                                   mode='min', 
                                   verbose=1)

gradient_monitor = GradientAndWeightMonitor(
    log_frequency=1,           # Check every epoch
    gradient_threshold=10.0,   # Alert if gradient norm > 10
    weight_threshold=100.0     # Alert if any layer weight norm > 100
)

branch_monitor = BranchScalingMonitor(
    log_frequency=1,    # Print weights every 5 epochs
    save_history=True   # Save history for potential plotting
)

def get_naive_baseline_metrics(val_dataset, val_steps):
    """
    Calculate naive baseline metrics for both target_high and target_low predictions.
    Uses mean prediction as the naive baseline for each target.
    
    Args:
        val_dataset: TensorFlow dataset from create_multi_instrument_dataset
        val_steps: Number of validation steps/batches to process
        
    Returns:
        dict: Contains metrics for both target_high and target_low
              Each target has: baseline_value, mae, mse, rmse
    """
    # Collect all target values
    all_target_highs = []
    # all_target_lows = []
    
    for i, batch in enumerate(val_dataset):
        if i >= val_steps:
            break
        (main_input, hourly_input, partial, position, hourly_position), targets = batch
        
        target_highs = targets['target_high'].numpy()        
        all_target_highs.extend(target_highs)
    
    all_target_highs = np.array(all_target_highs)
    baseline_high = np.mean(all_target_highs)
    predictions_high = np.full_like(all_target_highs, baseline_high)
    mae_high = np.mean(np.abs(predictions_high - all_target_highs))
    mse_high = np.mean((predictions_high - all_target_highs) ** 2)
    rmse_high = np.sqrt(mse_high)
    
    
    return {
        'target_high': {
            'baseline_value': baseline_high,
            'mae': mae_high,
            'mse': mse_high,
            'rmse': rmse_high
        },
        'total_samples': len(all_target_highs)
    }

print(get_naive_baseline_metrics(val_dataset, val_steps))

history = model.fit(
    train_dataset,
    epochs=50,
    steps_per_epoch=train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=[early_stopping, model_checkpoint, gradient_monitor, branch_monitor]
)