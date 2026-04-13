import os
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
from constants.global_constants import *
from core.modeler import create_regression_model
from core.transformer_builder import WarmupCosineDecay
from regression.losses import asymmetric_huber_loss_single, profit_precision_metric, profit_recall_metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from core.gradient_monitor import GradientAndWeightMonitor, BranchScalingMonitor


starting_dir = "data/final_data"
working_path = "data/experimenting"

import os
from core.data_generator import InstrumentConfig, MultiInstrumentDatasetConfig, create_multi_instrument_dataset
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


if __name__ == "__main__":
    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps()

    def compile_model_lightweight(model, updelta, downdelta):
        """
        Streamlined compilation with only the most important metrics
        Mixed precision compatible.
        """
        lr_schedule = WarmupCosineDecay(initial_lr=1e-5, warmup_steps=train_steps*2, decay_steps=train_steps*40)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'target_high': 'mse',
                'target_low': 'mse',
            },
            loss_weights={
                'target_high': 1.0,
                'target_low': 1.0,
            },
            metrics={
                'target_high': ['mae', 'mse'],
                'target_low': ['mae', 'mse'],
            }
        )
        return model

    model = create_regression_model(feature_cols=feature_cols, d_model=R_D_MODEL, num_heads=R_NUM_HEADS, ff_dim=R_FF_DIM,
                                    num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS, training=False)
    model = compile_model_lightweight(model=model, updelta=6.0, downdelta=-1.0)
    model.load_weights('models/regressor.keras')

    def evaluate_with_denormalization(model, test_dataset, test_steps):
        """Custom evaluation with both normalized and denormalized comparison (MAE + MSE)."""

        accum = {k: 0.0 for k in [
            'high_norm_mae', 'high_norm_rw_mae', 'high_denorm_mae', 'high_rw_mae',
            'high_norm_mse', 'high_norm_rw_mse', 'high_denorm_mse', 'high_rw_mse',
            'low_norm_mae',  'low_norm_rw_mae',  'low_denorm_mae',  'low_rw_mae',
            'low_norm_mse',  'low_norm_rw_mse',  'low_denorm_mse',  'low_rw_mse',
        ]}
        batch_count = 0

        for batch in test_dataset.take(test_steps):
            (main_input, hourly_input, partial_input, minutes_input, length_input), targets = batch
            predictions = model.predict((main_input, hourly_input, partial_input, minutes_input, length_input))

            pred_high = predictions['target_high'] if isinstance(predictions, dict) else predictions
            pred_low  = predictions['target_low']  if isinstance(predictions, dict) else predictions
            target_high = targets['target_high']
            target_low  = targets['target_low']
            norm_min = targets['norm_min']
            norm_max = targets['norm_max']
            original_close = targets['original_close']

            if len(pred_high.shape) == 2 and pred_high.shape[-1] == 1:
                pred_high = tf.squeeze(pred_high, axis=-1)
            if len(pred_low.shape) == 2 and pred_low.shape[-1] == 1:
                pred_low = tf.squeeze(pred_low, axis=-1)

            norm_rw = tf.clip_by_value((original_close - norm_min) / (norm_max - norm_min), 0.0, 1.0)
            scale = norm_max - norm_min

            denorm_pred_high   = pred_high  * scale + norm_min
            denorm_target_high = target_high * scale + norm_min
            denorm_pred_low    = pred_low   * scale + norm_min
            denorm_target_low  = target_low  * scale + norm_min
            denorm_rw          = original_close

            def mae(a, b): return tf.reduce_mean(tf.abs(a - b)).numpy()
            def mse(a, b): return tf.reduce_mean(tf.square(a - b)).numpy()

            accum['high_norm_mae']    += mae(pred_high,  target_high)
            accum['high_norm_rw_mae'] += mae(norm_rw,    target_high)
            accum['high_denorm_mae']  += mae(denorm_pred_high, denorm_target_high)
            accum['high_rw_mae']      += mae(denorm_rw,  denorm_target_high)
            accum['high_norm_mse']    += mse(pred_high,  target_high)
            accum['high_norm_rw_mse'] += mse(norm_rw,    target_high)
            accum['high_denorm_mse']  += mse(denorm_pred_high, denorm_target_high)
            accum['high_rw_mse']      += mse(denorm_rw,  denorm_target_high)

            accum['low_norm_mae']    += mae(pred_low,  target_low)
            accum['low_norm_rw_mae'] += mae(norm_rw,   target_low)
            accum['low_denorm_mae']  += mae(denorm_pred_low, denorm_target_low)
            accum['low_rw_mae']      += mae(denorm_rw, denorm_target_low)
            accum['low_norm_mse']    += mse(pred_low,  target_low)
            accum['low_norm_rw_mse'] += mse(norm_rw,   target_low)
            accum['low_denorm_mse']  += mse(denorm_pred_low, denorm_target_low)
            accum['low_rw_mse']      += mse(denorm_rw, denorm_target_low)

            batch_count += 1

        avg = {k: v / batch_count for k, v in accum.items()}

        def pct(rw, model): return (rw - model) / rw * 100

        print("===== target_high — Normalized =====")
        print(f"  MAE  model={avg['high_norm_mae']:.6f}  rw={avg['high_norm_rw_mae']:.6f}  improvement={pct(avg['high_norm_rw_mae'], avg['high_norm_mae']):.2f}%")
        print(f"  MSE  model={avg['high_norm_mse']:.6f}  rw={avg['high_norm_rw_mse']:.6f}  improvement={pct(avg['high_norm_rw_mse'], avg['high_norm_mse']):.2f}%")
        print("===== target_high — Denormalized =====")
        print(f"  MAE  model={avg['high_denorm_mae']:.6f}  rw={avg['high_rw_mae']:.6f}  improvement={pct(avg['high_rw_mae'], avg['high_denorm_mae']):.2f}%")
        print(f"  MSE  model={avg['high_denorm_mse']:.6f}  rw={avg['high_rw_mse']:.6f}  improvement={pct(avg['high_rw_mse'], avg['high_denorm_mse']):.2f}%")

        print("===== target_low — Normalized =====")
        print(f"  MAE  model={avg['low_norm_mae']:.6f}  rw={avg['low_norm_rw_mae']:.6f}  improvement={pct(avg['low_norm_rw_mae'], avg['low_norm_mae']):.2f}%")
        print(f"  MSE  model={avg['low_norm_mse']:.6f}  rw={avg['low_norm_rw_mse']:.6f}  improvement={pct(avg['low_norm_rw_mse'], avg['low_norm_mse']):.2f}%")
        print("===== target_low — Denormalized =====")
        print(f"  MAE  model={avg['low_denorm_mae']:.6f}  rw={avg['low_rw_mae']:.6f}  improvement={pct(avg['low_rw_mae'], avg['low_denorm_mae']):.2f}%")
        print(f"  MSE  model={avg['low_denorm_mse']:.6f}  rw={avg['low_rw_mse']:.6f}  improvement={pct(avg['low_rw_mse'], avg['low_denorm_mse']):.2f}%")

        return avg

    def evaluate_with_denormalization_matching_keras(model, test_dataset, test_steps):
        metrics = {
            'high': {
                'norm':   {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
                'denorm': {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
                'rw_norm':   {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
                'rw_denorm': {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
            },
            'low': {
                'norm':   {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
                'denorm': {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
                'rw_norm':   {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
                'rw_denorm': {'mae': tf.keras.metrics.MeanAbsoluteError(), 'mse': tf.keras.metrics.MeanSquaredError()},
            },
        }

        for batch in test_dataset.take(test_steps):
            (main_input, hourly_input, partial_input, minutes_input, length_input), targets = batch
            preds = model.predict((main_input, hourly_input, partial_input, minutes_input, length_input))

            pred_high = preds['target_high'] if isinstance(preds, dict) else preds
            pred_low  = preds['target_low']  if isinstance(preds, dict) else preds
            target_high = targets['target_high']
            target_low  = targets['target_low']
            norm_min = targets['norm_min']
            norm_max = targets['norm_max']
            original_close = targets['original_close']

            if len(pred_high.shape) == 2 and pred_high.shape[-1] == 1:
                pred_high = tf.squeeze(pred_high, axis=-1)
            if len(pred_low.shape) == 2 and pred_low.shape[-1] == 1:
                pred_low = tf.squeeze(pred_low, axis=-1)

            scale = norm_max - norm_min
            norm_rw = tf.clip_by_value((original_close - norm_min) / scale, 0.0, 1.0)

            denorm_pred_high   = pred_high   * scale + norm_min
            denorm_target_high = target_high * scale + norm_min
            denorm_pred_low    = pred_low    * scale + norm_min
            denorm_target_low  = target_low  * scale + norm_min
            denorm_rw          = original_close

            for m in metrics['high']['norm'].values():   m.update_state(target_high, pred_high)
            for m in metrics['high']['denorm'].values(): m.update_state(denorm_target_high, denorm_pred_high)
            for m in metrics['high']['rw_norm'].values():   m.update_state(target_high, norm_rw)
            for m in metrics['high']['rw_denorm'].values(): m.update_state(denorm_target_high, denorm_rw)

            for m in metrics['low']['norm'].values():    m.update_state(target_low, pred_low)
            for m in metrics['low']['denorm'].values():  m.update_state(denorm_target_low, denorm_pred_low)
            for m in metrics['low']['rw_norm'].values():    m.update_state(target_low, norm_rw)
            for m in metrics['low']['rw_denorm'].values():  m.update_state(denorm_target_low, denorm_rw)

        def r(m): return m.result().numpy()

        results = {
            'target_high': {
                'normalized':   {'mae': (r(metrics['high']['norm']['mae']),   r(metrics['high']['rw_norm']['mae'])),
                                 'mse': (r(metrics['high']['norm']['mse']),   r(metrics['high']['rw_norm']['mse']))},
                'denormalized': {'mae': (r(metrics['high']['denorm']['mae']), r(metrics['high']['rw_denorm']['mae'])),
                                 'mse': (r(metrics['high']['denorm']['mse']), r(metrics['high']['rw_denorm']['mse']))},
            },
            'target_low': {
                'normalized':   {'mae': (r(metrics['low']['norm']['mae']),   r(metrics['low']['rw_norm']['mae'])),
                                 'mse': (r(metrics['low']['norm']['mse']),   r(metrics['low']['rw_norm']['mse']))},
                'denormalized': {'mae': (r(metrics['low']['denorm']['mae']), r(metrics['low']['rw_denorm']['mae'])),
                                 'mse': (r(metrics['low']['denorm']['mse']), r(metrics['low']['rw_denorm']['mse']))},
            },
        }

        for target, data in results.items():
            print(f"===== {target} =====")
            print(f"  Normalized   MAE (model, RW): {data['normalized']['mae']}")
            print(f"  Normalized   MSE (model, RW): {data['normalized']['mse']}")
            print(f"  Denormalized MAE (model, RW): {data['denormalized']['mae']}")
            print(f"  Denormalized MSE (model, RW): {data['denormalized']['mse']}")

        return results

    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps()

    evaluate_with_denormalization_matching_keras(model, val_dataset, val_steps)
