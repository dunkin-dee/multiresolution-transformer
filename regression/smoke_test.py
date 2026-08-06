"""End-to-end smoke test: data generator → model → two short training epochs.

Usage::

    python -m regression.smoke_test [--instrument GBPUSD#]

Verifies that preprocessed data loads, batch shapes match the model's inputs, and
gradients flow — in about a minute, rather than the hours a real run takes. Run
this after changing anything in ``core/`` before starting a long training job.
"""

import argparse
import sys

import tensorflow as tf

from constants.global_constants import FEATURES, NUM_TOKENS, OTHER_TOKENS, R_D_MODEL, R_FF_DIM, R_NUM_HEADS
from core.modeler import create_regression_model
from regression.datasets import DEFAULT_WORKING_PATH, discover_instruments, get_datasets_and_steps

EXPECTED_INPUT_NAMES = [
    'minute_input', 'hourly_input', 'partial_hour_input',
    'minutes_into_hour', 'partial_hour_length',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-path", default=DEFAULT_WORKING_PATH)
    parser.add_argument("--instrument", default=None,
                        help="Defaults to the first instrument found.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    instrument = args.instrument or discover_instruments(args.working_path)[0]
    print(f"\n=== Smoke test: {instrument}, {args.epochs} epochs x {args.steps} steps ===\n")

    # strip_for_fit=False so we can inspect the eval-only keys, then strip below.
    (train_dataset, val_dataset, _), (train_rows, val_rows, _) = get_datasets_and_steps(
        working_path=args.working_path,
        instruments=[instrument],
        max_chunks_per_instrument=5,
        strip_for_fit=False,
    )

    expected = {
        'minute_input': (NUM_TOKENS, len(FEATURES)),
        'hourly_input': (OTHER_TOKENS, len(FEATURES)),
        'partial_hour_input': (1, len(FEATURES)),
        'minutes_into_hour': (1,),
        'partial_hour_length': (1,),
    }
    for inputs, targets in train_dataset.take(1):
        for name, tensor in zip(EXPECTED_INPUT_NAMES, inputs):
            actual = tuple(tensor.shape[1:])
            status = "ok" if actual == expected[name] else f"MISMATCH, want {expected[name]}"
            print(f"  {name:<22} {str(tensor.shape):<22} {status}")
            if actual != expected[name]:
                sys.exit(f"Batch shape does not match model input '{name}'.")
        for key in ('target_high', 'target_low', 'norm_min', 'norm_max', 'original_close'):
            print(f"  target[{key}]{'':<10} {targets[key].shape}")

    from core.data_generator import strip_eval_targets
    train_dataset = strip_eval_targets(train_dataset)
    val_dataset = strip_eval_targets(val_dataset)

    model = create_regression_model(
        feature_cols=FEATURES, d_model=R_D_MODEL, num_heads=R_NUM_HEADS,
        ff_dim=R_FF_DIM, num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={'target_high': 'mse', 'target_low': 'mse'},
        loss_weights={'target_high': 1.0, 'target_low': 1.0},
        metrics={'target_high': ['mae'], 'target_low': ['mae']},
    )

    print(f"\n  Model parameters: {model.count_params():,}")
    print(f"  Training batches available: {train_rows}  |  validation: {val_rows}\n")

    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=val_dataset,
        validation_steps=args.steps,
    )

    losses = history.history['loss']
    print(f"\n=== Smoke test passed. Loss {losses[0]:.4f} -> {losses[-1]:.4f} ===\n")


if __name__ == "__main__":
    main()
