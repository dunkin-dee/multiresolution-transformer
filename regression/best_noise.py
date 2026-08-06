"""Resumable grid search over the Gaussian noise augmentation parameters.

Usage::

    python -m regression.best_noise --instruments SILVER#

Trains a short run per (noise_std, noise_probability) combination and records the
best validation loss. Progress is checkpointed after every combination, so the
search survives interruption — rerun the same command and it picks up where it
stopped.

Results from the run committed to this repo are in ``results/noise_search.csv``.

Note: an earlier version of this script hardcoded the *hourly* noise values while
only sweeping the 5-minute ones, so its recorded results describe a narrower
search than the parameter names suggest. Both resolutions are now swept together.
"""

import argparse
import json
import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from constants.global_constants import FEATURES, NUM_TOKENS, OTHER_TOKENS, R_D_MODEL, R_FF_DIM, R_NUM_HEADS
from core.modeler import create_regression_model
from core.transformer_builder import WarmupCosineDecay
from regression.datasets import NoiseConfig, get_datasets_and_steps
from regression.losses import (
    asymmetric_huber_loss_single,
    profit_precision_metric,
    profit_recall_metric,
)

DEFAULT_WORKING_PATH = "data/regression"
CHECKPOINT_FILE = 'search_checkpoint.json'
RESULTS_FILE = 'search_results_incremental.csv'

NOISE_STD_RANGE = [0.001, 0.005, 0.01, 0.02]
NOISE_PROB_RANGE = [0.2, 0.5, 0.7, 0.9]


def save_checkpoint(completed, results, best_params, best_val_loss):
    """Persist search progress so an interrupted run can resume."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'completed_combinations': [list(c) for c in completed],
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    if results:
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
    print(f"  checkpoint saved ({len(completed)} combinations complete)")


def load_checkpoint():
    """Restore prior progress, or start fresh if no checkpoint exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), [], None, float('inf')
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        completed = {tuple(c) for c in data['completed_combinations']}
        results = (pd.read_csv(RESULTS_FILE).to_dict('records')
                   if os.path.exists(RESULTS_FILE) else [])
        best_params = data.get('best_params')
        best_val_loss = data.get('best_val_loss', float('inf'))
        print(f"Resuming: {len(completed)} combinations already done, "
              f"best so far {best_params} @ {best_val_loss:.6f}")
        return completed, results, best_params, best_val_loss
    except Exception as e:
        print(f"Could not read checkpoint ({e}) — starting fresh.")
        return set(), [], None, float('inf')


def compile_for_search(model, train_steps):
    """Compile with the asymmetric high-head loss used during the search."""
    lr_schedule = WarmupCosineDecay(
        initial_lr=1e-4, warmup_steps=train_steps * 2, decay_steps=train_steps * 40
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, clipnorm=0.5, weight_decay=1e-4
        ),
        loss={
            'target_high': asymmetric_huber_loss_single(
                delta=2.5, underestimate_weight=3.2, overestimate_weight=0.6
            ),
            'target_low': 'mse',
        },
        metrics={
            'target_high': [
                'mae', 'mse',
                profit_precision_metric(threshold=6.0),
                profit_recall_metric(threshold=6.0),
            ],
            'target_low': ['mae'],
        },
    )
    return model


def _history_value(history, epoch, *candidate_keys):
    """Read a metric at ``epoch``, tolerating Keras' output-prefixed metric names."""
    for key in candidate_keys:
        if key in history:
            return history[key][epoch]
    return float('nan')


def train_and_evaluate(noise_std, noise_prob, working_path, instruments, max_epochs):
    """Train one configuration and return its result row."""
    print(f"\n{'=' * 60}\nnoise_std={noise_std}, noise_probability={noise_prob}\n{'=' * 60}")

    checkpoint_name = f'models/regressor_std{noise_std}_prob{noise_prob}.keras'
    try:
        tf.keras.backend.clear_session()

        noise = (None if noise_std == 0 and noise_prob == 0 else
                 NoiseConfig(std_5min=noise_std, std_hourly=noise_std,
                             probability_5min=noise_prob, probability_hourly=noise_prob))

        (train_dataset, val_dataset, _), (train_steps, val_steps, _) = (
            get_datasets_and_steps(working_path=working_path, instruments=instruments,
                                   train_noise=noise)
        )

        model = create_regression_model(
            feature_cols=FEATURES, d_model=R_D_MODEL, num_heads=R_NUM_HEADS,
            ff_dim=R_FF_DIM, num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS,
        )
        model = compile_for_search(model, train_steps)

        history = model.fit(
            train_dataset,
            epochs=max_epochs,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=8, mode='min',
                              verbose=1, restore_best_weights=True),
                ModelCheckpoint(checkpoint_name, monitor='val_loss',
                                save_best_only=True, mode='min', verbose=0),
            ],
            verbose=1,
        ).history

        best_epoch = int(np.argmin(history['val_loss']))
        best_val_loss = float(history['val_loss'][best_epoch])
        print(f"  best val_loss {best_val_loss:.6f} at epoch {best_epoch + 1}")

        return {
            'noise_std_min': noise_std,
            'noise_probability_min': noise_prob,
            'best_val_loss': best_val_loss,
            'final_val_loss': float(history['val_loss'][-1]),
            'best_epoch': best_epoch + 1,
            'total_epochs': len(history['val_loss']),
            'best_val_mae': _history_value(history, best_epoch,
                                           'val_target_high_mae', 'val_mae'),
            'best_val_mse': _history_value(history, best_epoch,
                                           'val_target_high_mse', 'val_mse'),
            'checkpoint_path': checkpoint_name,
            'status': 'completed',
            'error': '',
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return {
            'noise_std_min': noise_std, 'noise_probability_min': noise_prob,
            'best_val_loss': float('inf'), 'final_val_loss': float('inf'),
            'best_epoch': 0, 'total_epochs': 0,
            'best_val_mae': float('inf'), 'best_val_mse': float('inf'),
            'checkpoint_path': '', 'status': 'failed', 'error': str(e),
        }


def save_heatmap(results_df, timestamp):
    """Render the search surface, marking the best configuration."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping heatmap.")
        return

    pivot = results_df.pivot(index='noise_probability_min',
                            columns='noise_std_min', values='best_val_loss')
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Best validation loss')
    plt.xticks(range(len(pivot.columns)), [f'{x:.3f}' for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f'{x:.1f}' for x in pivot.index])
    plt.xlabel('Noise standard deviation')
    plt.ylabel('Noise probability')
    plt.title('Validation loss — noise parameter search')

    best = results_df.loc[results_df['best_val_loss'].idxmin()]
    plt.scatter(list(pivot.columns).index(best['noise_std_min']),
                list(pivot.index).index(best['noise_probability_min']),
                color='red', s=200, marker='*', label='Best')
    plt.legend()
    plt.tight_layout()

    path = f'results/noise_search_heatmap_{timestamp}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-path", default=DEFAULT_WORKING_PATH)
    parser.add_argument("--instruments", nargs="*", default=None,
                        help="Defaults to every instrument under --working-path.")
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    completed, results, best_params, best_val_loss = load_checkpoint()

    combinations = list(product(NOISE_STD_RANGE, NOISE_PROB_RANGE))
    if (0, 0) not in combinations:
        combinations.insert(0, (0, 0))  # no-augmentation baseline

    print(f"Grid search: {len(combinations)} combinations, "
          f"{len(combinations) - len(completed)} remaining")

    for i, (noise_std, noise_prob) in enumerate(combinations, 1):
        if (noise_std, noise_prob) in completed:
            continue
        print(f"\nProgress: {i}/{len(combinations)}")

        result = train_and_evaluate(noise_std, noise_prob, args.working_path,
                                    args.instruments, args.max_epochs)
        results.append(result)
        completed.add((noise_std, noise_prob))

        if result['best_val_loss'] < best_val_loss:
            best_val_loss = result['best_val_loss']
            best_params = {'noise_std_min': noise_std,
                           'noise_probability_min': noise_prob}
            print(f"  *** new best: {best_val_loss:.6f} ***")

        save_checkpoint(completed, results, best_params, best_val_loss)

    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    final_path = f'results/noise_search_results_{timestamp}.csv'
    results_df.to_csv(final_path, index=False)

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_val_loss:.6f}\n")
    print("Top 5 configurations:")
    for rank, (_, row) in enumerate(results_df.nsmallest(5, 'best_val_loss').iterrows(), 1):
        print(f"  {rank}. std={row['noise_std_min']}, prob={row['noise_probability_min']}, "
              f"val_loss={row['best_val_loss']:.6f}, epoch={row['best_epoch']}")
    print(f"\nResults saved to {final_path}")

    save_heatmap(results_df, timestamp)

    for path in (CHECKPOINT_FILE, RESULTS_FILE):
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
