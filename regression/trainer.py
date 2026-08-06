"""Train the multi-resolution regression model across all preprocessed instruments.

Usage::

    python -m regression.trainer

Reads chunks from ``data/experimenting`` (see ``regression.preprocess``) and
writes the best checkpoint to ``models/regressor.keras``.
"""

import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from constants.global_constants import R_D_MODEL, R_FF_DIM, R_NUM_HEADS, NUM_TOKENS, OTHER_TOKENS, FEATURES
from core.gradient_monitor import BranchScalingMonitor, GradientAndWeightMonitor
from core.modeler import create_regression_model
from core.transformer_builder import WarmupCosineDecay
from regression.datasets import DEFAULT_WORKING_PATH, get_datasets_and_steps

MODEL_PATH = "models/regressor.keras"


def compile_model(model, train_steps, initial_lr=1e-5):
    """Compile with plain MSE on both heads and a warmup-then-cosine LR schedule.

    The asymmetric losses in ``regression.losses`` were tried earlier in the
    project; plain MSE on both heads is what the current results were produced
    with. See ``regression/losses.py`` for the alternatives.
    """
    lr_schedule = WarmupCosineDecay(
        initial_lr=initial_lr,
        warmup_steps=train_steps * 2,
        decay_steps=train_steps * 40,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss={'target_high': 'mse', 'target_low': 'mse'},
        loss_weights={'target_high': 1.0, 'target_low': 1.0},
        metrics={'target_high': ['mae', 'mse'], 'target_low': ['mae', 'mse']},
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-path", default=DEFAULT_WORKING_PATH,
                        help="Directory of preprocessed chunks.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Where to save the best checkpoint.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience in epochs.")
    args = parser.parse_args()

    (train_dataset, val_dataset, _test_dataset), (train_steps, val_steps, _) = (
        get_datasets_and_steps(working_path=args.working_path)
    )

    model = create_regression_model(
        feature_cols=FEATURES,
        d_model=R_D_MODEL,
        num_heads=R_NUM_HEADS,
        ff_dim=R_FF_DIM,
        num_tokens=NUM_TOKENS,
        other_tokens=OTHER_TOKENS,
    )
    model = compile_model(model, train_steps)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=1),
        ModelCheckpoint(args.model_path, monitor='val_loss',
                        save_best_only=True, mode='min', verbose=1),
        GradientAndWeightMonitor(log_frequency=1, gradient_threshold=10.0,
                                 weight_threshold=100.0),
        BranchScalingMonitor(log_frequency=1, save_history=True),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    print(f"\nBest checkpoint saved to {args.model_path}")
    print("Evaluate it with: python -m regression.test")


if __name__ == "__main__":
    main()
