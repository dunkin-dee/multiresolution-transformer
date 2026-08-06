"""Fine-tune the shared base model on a single instrument.

Usage::

    python -m regression.fine_tuner --instrument USDJPY#

Starts from ``models/regressor.keras`` (produced by ``regression.trainer``) and
writes ``models/regressor_{instrument}.keras``. Uses a much lower learning rate
than training from scratch, on the assumption the base model already carries the
cross-instrument structure and only the instrument-specific scale needs to move.
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from constants.global_constants import FEATURES, NUM_TOKENS, OTHER_TOKENS, R_D_MODEL, R_FF_DIM, R_NUM_HEADS
from core.gradient_monitor import GradientAndWeightMonitor
from core.data_generator import strip_eval_targets
from core.modeler import create_regression_model
from core.transformer_builder import WarmupCosineDecay
from regression.datasets import get_datasets_and_steps
from regression.losses import (
    asymmetric_huber_loss_single,
    profit_precision_metric,
    profit_recall_metric,
)

DEFAULT_WORKING_PATH = "data/regression_final"
BASE_MODEL_PATH = "models/regressor.keras"


def compile_for_finetuning(model, train_steps, initial_lr=1e-8):
    """Compile with the asymmetric Huber loss, weighted to punish underestimating highs.

    Unlike ``regression.trainer`` (plain MSE on both heads), fine-tuning optimises
    ``target_high`` asymmetrically: missing an upside move costs more than
    overshooting one. ``target_low`` keeps MSE so the head does not drift while
    the high head is being reshaped.
    """
    lr_schedule = WarmupCosineDecay(
        initial_lr=initial_lr,
        warmup_steps=train_steps * 2,
        decay_steps=train_steps * 40,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule, weight_decay=1e-10),
        loss={
            'target_high': asymmetric_huber_loss_single(
                delta=2.5, underestimate_weight=3.2, overestimate_weight=0.6
            ),
            'target_low': 'mse',
        },
        loss_weights={'target_high': 1.0, 'target_low': 1.0},
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


def naive_baseline(val_dataset, val_steps):
    """Mean-prediction baseline for ``target_high`` — the floor any model must clear."""
    values = []
    for i, (_inputs, targets) in enumerate(val_dataset):
        if i >= val_steps:
            break
        values.extend(targets['target_high'].numpy())

    values = np.array(values)
    baseline = float(np.mean(values))
    mse = float(np.mean((baseline - values) ** 2))
    return {
        'baseline_value': baseline,
        'mae': float(np.mean(np.abs(baseline - values))),
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'total_samples': len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True,
                        help="Instrument directory name, e.g. 'USDJPY#'.")
    parser.add_argument("--working-path", default=DEFAULT_WORKING_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    output_path = f"models/regressor_{args.instrument}.keras"

    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = (
        get_datasets_and_steps(working_path=args.working_path,
                               instruments=[args.instrument])
    )

    model = create_regression_model(
        feature_cols=FEATURES,
        d_model=R_D_MODEL,
        num_heads=R_NUM_HEADS,
        ff_dim=R_FF_DIM,
        num_tokens=NUM_TOKENS,
        other_tokens=OTHER_TOKENS,
    )
    model.load_weights(args.base_model)
    model = compile_for_finetuning(model, train_steps)

    print(f"Naive baseline on validation: {naive_baseline(val_dataset, val_steps)}")

    callbacks = [
        # Monitors val_loss: the previous 'val_metric' matched no compiled metric,
        # so EarlyStopping and ModelCheckpoint silently never fired.
        EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=1),
        ModelCheckpoint(output_path, monitor='val_loss',
                        save_best_only=True, mode='min', verbose=1),
        GradientAndWeightMonitor(log_frequency=1, gradient_threshold=10.0,
                                 weight_threshold=100.0),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    model.load_weights(output_path)
    # get_datasets_and_steps always leaves the eval-only target keys on the test
    # split; Keras rejects target keys with no matching output, so strip them.
    model.evaluate(strip_eval_targets(test_dataset), steps=test_steps)
    print(f"\nFine-tuned model saved to {output_path}")


if __name__ == "__main__":
    main()
