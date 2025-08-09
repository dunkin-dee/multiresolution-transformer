import os
import logging
import warnings
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
from itertools import product
import json
from constants.global_constants import *
from modeler import create_regression_model
from transformer_builder import WarmupCosineDecay
from regression_losses import asymmetric_huber_loss_single, profit_precision_metric, profit_recall_metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gradient_monitor import GradientAndWeightMonitor, BranchScalingMonitor


starting_dir = "data/final_data"
working_path = "data/regression"

from generators.regression_multi_instrument_data_generator import InstrumentConfig, MultiInstrumentDatasetConfig, create_multi_instrument_dataset
from constants.global_constants import FEATURES, NUM_TOKENS, OTHER_TOKENS, BATCH_SIZE, LOOKBACK_WINDOW


instruments = os.listdir(working_path)
instruments = ['SILVER#']
feature_cols = FEATURES

# Define search ranges
min_noise_std_range = [0.001, 0.005, 0.01, 0.02]
min_noise_prob_range = [0.2, 0.5, 0.7, 0.9]

# Checkpoint file names
CHECKPOINT_FILE = 'search_checkpoint.json'
RESULTS_FILE = 'search_results_incremental.csv'

def save_checkpoint(completed_combinations, current_results, best_params, best_val_loss):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'completed_combinations': completed_combinations,
        'best_params': best_params,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Also save results incrementally
    if current_results:
        pd.DataFrame(current_results).to_csv(RESULTS_FILE, index=False)
    
    print(f"✓ Checkpoint saved: {len(completed_combinations)} combinations completed")

def load_checkpoint():
    """Load previous progress from checkpoint file"""
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), [], None, float('inf')
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        
        completed = set(tuple(combo) for combo in data['completed_combinations'])
        best_params = data.get('best_params')
        best_val_loss = data.get('best_val_loss', float('inf'))
        
        # Load existing results
        results = []
        if os.path.exists(RESULTS_FILE):
            df = pd.read_csv(RESULTS_FILE)
            results = df.to_dict('records')
        
        print(f"✓ Loaded checkpoint: {len(completed)} combinations already completed")
        print(f"✓ Current best: {best_params} with val_loss={best_val_loss:.6f}")
        
        return completed, results, best_params, best_val_loss
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return set(), [], None, float('inf')

# Results tracking
completed_combinations, results, best_params, best_val_loss = load_checkpoint()

def get_datasets_and_steps(instruments=instruments, working_path=working_path, feature_cols=feature_cols,
                          noise_std_min=0.01, noise_probability_min=0.2):
    """
    Modified to accept noise parameters as arguments
    """
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
        noise_std_5min=noise_std_min,
        noise_std_hourly=0.015,  # Use parameter
        noise_probability_5min=noise_probability_min,
        noise_probability_hourly=0.07  # Use parameter
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


def compile_model_lightweight(model, train_steps, updelta=6.0, downdelta=-1.0):
    """
    Streamlined compilation with only the most important metrics
    Mixed precision compatible.
    """
    lr_schedule = WarmupCosineDecay(initial_lr=1e-4, warmup_steps=train_steps*2, decay_steps=train_steps*40)
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


def train_and_evaluate(noise_std_min, noise_probability_min, max_epochs=30):
    """
    Train a model with given noise parameters and return the best validation loss
    """
    print(f"\n{'='*60}")
    print(f"Testing: noise_std_min={noise_std_min}, noise_probability_min={noise_probability_min}")
    print(f"{'='*60}")
    
    try:
        # Clear any existing models from memory
        tf.keras.backend.clear_session()
        
        # Get datasets with specific noise parameters
        (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(
            noise_std_min=noise_std_min,
            noise_probability_min=noise_probability_min
        )
        
        # Create and compile model
        model = create_regression_model(feature_cols=feature_cols, d_model=R_D_MODEL, num_heads=R_NUM_HEADS, ff_dim=R_FF_DIM,
                                        num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS)
        model = compile_model_lightweight(model=model, train_steps=train_steps, updelta=6.0, downdelta=-1.0)
        
        # Setup callbacks
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=8,  # Reduced patience for faster search
                                       mode='min', 
                                       verbose=1,
                                       restore_best_weights=True)
        
        # Create unique checkpoint filename for this run
        checkpoint_name = f'models/regressor_std{noise_std_min}_prob{noise_probability_min}.keras'
        model_checkpoint = ModelCheckpoint(checkpoint_name, 
                                           monitor='val_loss', 
                                           save_best_only=True, 
                                           mode='min', 
                                           verbose=0)  # Reduced verbosity
        
        # Train model
        history = model.fit(
            train_dataset,
            epochs=max_epochs,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Get best validation loss
        best_val_loss = min(history.history['val_loss'])
        final_val_loss = history.history['val_loss'][-1]
        
        # Get corresponding metrics at best epoch
        best_epoch = np.argmin(history.history['val_loss'])
        best_val_mae = history.history['val_mae'][best_epoch]
        best_val_mse = history.history['val_mse'][best_epoch]
        
        print(f"✓ SUCCESS: Best val_loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
        print(f"  Best val_mae: {best_val_mae:.6f}")
        print(f"  Best val_mse: {best_val_mse:.6f}")
        
        return {
            'noise_std_min': noise_std_min,
            'noise_probability_min': noise_probability_min,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'best_epoch': best_epoch + 1,
            'total_epochs': len(history.history['val_loss']),
            'best_val_mae': best_val_mae,
            'best_val_mse': best_val_mse,
            'checkpoint_path': checkpoint_name,
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return {
            'noise_std_min': noise_std_min,
            'noise_probability_min': noise_probability_min,
            'best_val_loss': float('inf'),
            'final_val_loss': float('inf'),
            'best_epoch': 0,
            'total_epochs': 0,
            'best_val_mae': float('inf'),
            'best_val_mse': float('inf'),
            'checkpoint_path': '',
            'status': 'failed',
            'error': str(e)
        }


# Main hyperparameter search loop
print("Starting hyperparameter search for noise parameters...")

# Generate all combinations
all_combinations = list(product(min_noise_std_range, min_noise_prob_range))
# Add the baseline case if not already included
baseline = (0, 0)
if baseline not in all_combinations:
    all_combinations.insert(0, baseline)

total_combinations = len(all_combinations)
print(f"Total combinations to test: {total_combinations}")
print(f"Already completed: {len(completed_combinations)}")
print(f"Remaining: {total_combinations - len(completed_combinations)}")

# Test baseline case first if not already done
if baseline not in completed_combinations:
    print(f"\nTesting baseline case: {baseline}")
    result = train_and_evaluate(0, 0)
    results.append(result)
    completed_combinations.add(baseline)
    
    # Track best overall performance
    if result['best_val_loss'] < best_val_loss:
        best_val_loss = result['best_val_loss']
        best_params = {
            'noise_std_min': 0,
            'noise_probability_min': 0
        }
        print(f"*** NEW BEST! Val Loss: {best_val_loss:.6f} ***")
    
    # Save checkpoint after baseline
    save_checkpoint(list(completed_combinations), results, best_params, best_val_loss)

# Continue with remaining combinations
current_combination = len(completed_combinations)

for noise_std, noise_prob in all_combinations:
    combination = (noise_std, noise_prob)
    
    # Skip if already completed
    if combination in completed_combinations:
        continue
        
    current_combination += 1
    print(f"\nProgress: {current_combination}/{total_combinations}")
    
    result = train_and_evaluate(noise_std, noise_prob)
    results.append(result)
    completed_combinations.add(combination)
    
    # Track best overall performance
    if result['best_val_loss'] < best_val_loss:
        best_val_loss = result['best_val_loss']
        best_params = {
            'noise_std_min': noise_std,
            'noise_probability_min': noise_prob
        }
        print(f"*** NEW BEST! Val Loss: {best_val_loss:.6f} ***")
    
    # Save checkpoint after each combination
    save_checkpoint(list(completed_combinations), results, best_params, best_val_loss)

# Save final results
results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_results_file = f'noise_search_results_final_{timestamp}.csv'
results_df.to_csv(final_results_file, index=False)

# Clean up checkpoint files
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)

# Print summary
print("\n" + "="*80)
print("HYPERPARAMETER SEARCH COMPLETE")
print("="*80)

print(f"\nBest parameters:")
print(f"  noise_std_min: {best_params['noise_std_min']}")
print(f"  noise_probability_min: {best_params['noise_probability_min']}")
print(f"  Best validation loss: {best_val_loss:.6f}")

print(f"\nTop 5 configurations:")
top_5 = results_df.nsmallest(5, 'best_val_loss')
for idx, row in top_5.iterrows():
    print(f"  Rank {idx+1}: std={row['noise_std_min']}, prob={row['noise_probability_min']}, "
          f"val_loss={row['best_val_loss']:.6f}, epoch={row['best_epoch']}")

print(f"\nResults saved to: {final_results_file}")

# Optional: Create a simple visualization of results
try:
    import matplotlib.pyplot as plt
    
    # Create pivot table for heatmap
    pivot_table = results_df.pivot(index='noise_probability_min', 
                                   columns='noise_std_min', 
                                   values='best_val_loss')
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot_table.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Best Validation Loss')
    plt.xticks(range(len(pivot_table.columns)), [f'{x:.3f}' for x in pivot_table.columns])
    plt.yticks(range(len(pivot_table.index)), [f'{x:.1f}' for x in pivot_table.index])
    plt.xlabel('Noise Standard Deviation (min)')
    plt.ylabel('Noise Probability (min)')
    plt.title('Validation Loss Heatmap - Noise Parameter Search')
    
    # Mark the best point
    best_row = results_df.loc[results_df['best_val_loss'].idxmin()]
    std_idx = list(pivot_table.columns).index(best_row['noise_std_min'])
    prob_idx = list(pivot_table.index).index(best_row['noise_probability_min'])
    plt.scatter(std_idx, prob_idx, color='red', s=200, marker='*', label='Best')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'noise_search_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: noise_search_heatmap_{timestamp}.png")
    
except ImportError:
    print("matplotlib not available - skipping visualization")
except Exception as e:
    print(f"Error creating visualization: {str(e)}")

print(f"\nSearch complete! Best model checkpoint: {results_df.loc[results_df['best_val_loss'].idxmin(), 'checkpoint_path']}")