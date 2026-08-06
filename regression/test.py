"""Evaluate a trained checkpoint against a random-walk baseline.

Usage::

    python -m regression.test [--split validation|testing]

Reports MAE and MSE for both heads, in normalised space and denormalised back to
real price units, alongside the naive "price stays where it is" baseline. The
denormalised numbers are the ones worth reading — normalised error is not
comparable across instruments or volatility regimes.
"""

import argparse

import tensorflow as tf

from constants.global_constants import FEATURES, NUM_TOKENS, OTHER_TOKENS, R_D_MODEL, R_FF_DIM, R_NUM_HEADS
from core.modeler import create_regression_model
from regression.datasets import DEFAULT_WORKING_PATH, get_datasets_and_steps

MODEL_PATH = "models/regressor.keras"


def evaluate_with_denormalization(model, dataset, steps):
    """Accumulate model-vs-baseline MAE/MSE using streaming Keras metrics.

    The random-walk baseline predicts that the future high (and low) equals the
    current close. Beating it is the bar this project set itself.
    """
    def metric_pair():
        return {'mae': tf.keras.metrics.MeanAbsoluteError(),
                'mse': tf.keras.metrics.MeanSquaredError()}

    metrics = {
        head: {space: metric_pair()
               for space in ('norm', 'denorm', 'rw_norm', 'rw_denorm')}
        for head in ('high', 'low')
    }

    for batch in dataset.take(steps):
        inputs, targets = batch
        preds = model.predict(inputs, verbose=0)

        pred_high, pred_low = preds['target_high'], preds['target_low']
        target_high, target_low = targets['target_high'], targets['target_low']
        norm_min, norm_max = targets['norm_min'], targets['norm_max']
        original_close = targets['original_close']

        if len(pred_high.shape) == 2 and pred_high.shape[-1] == 1:
            pred_high = tf.squeeze(pred_high, axis=-1)
        if len(pred_low.shape) == 2 and pred_low.shape[-1] == 1:
            pred_low = tf.squeeze(pred_low, axis=-1)

        scale = norm_max - norm_min
        norm_rw = tf.clip_by_value((original_close - norm_min) / scale, 0.0, 1.0)

        pairs = {
            ('high', 'norm'): (target_high, pred_high),
            ('high', 'denorm'): (target_high * scale + norm_min, pred_high * scale + norm_min),
            ('high', 'rw_norm'): (target_high, norm_rw),
            ('high', 'rw_denorm'): (target_high * scale + norm_min, original_close),
            ('low', 'norm'): (target_low, pred_low),
            ('low', 'denorm'): (target_low * scale + norm_min, pred_low * scale + norm_min),
            ('low', 'rw_norm'): (target_low, norm_rw),
            ('low', 'rw_denorm'): (target_low * scale + norm_min, original_close),
        }
        for (head, space), (y_true, y_pred) in pairs.items():
            for m in metrics[head][space].values():
                m.update_state(y_true, y_pred)

    def value(head, space, kind):
        return float(metrics[head][space][kind].result().numpy())

    results = {}
    for head in ('high', 'low'):
        print(f"\n===== target_{head} =====")
        results[f'target_{head}'] = {}
        for space, rw_space, label in (('norm', 'rw_norm', 'Normalized'),
                                       ('denorm', 'rw_denorm', 'Denormalized')):
            entry = {}
            for kind in ('mae', 'mse'):
                model_v, rw_v = value(head, space, kind), value(head, rw_space, kind)
                improvement = (rw_v - model_v) / rw_v * 100 if rw_v else float('nan')
                entry[kind] = {'model': model_v, 'random_walk': rw_v,
                               'improvement_pct': improvement}
                print(f"  {label:<13} {kind.upper()}  model={model_v:.6f}  "
                      f"random_walk={rw_v:.6f}  improvement={improvement:+.2f}%")
            results[f'target_{head}'][space] = entry

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-path", default=DEFAULT_WORKING_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--split", choices=("validation", "testing"), default="validation",
                        help="Which split to evaluate.")
    args = parser.parse_args()

    # strip_for_fit=False keeps norm_min/norm_max/original_close, which the
    # denormalisation below needs.
    (_train, val_dataset, test_dataset), (_ts, val_steps, test_steps) = (
        get_datasets_and_steps(working_path=args.working_path, strip_for_fit=False)
    )
    dataset, steps = ((val_dataset, val_steps) if args.split == "validation"
                      else (test_dataset, test_steps))

    model = create_regression_model(
        feature_cols=FEATURES,
        d_model=R_D_MODEL,
        num_heads=R_NUM_HEADS,
        ff_dim=R_FF_DIM,
        num_tokens=NUM_TOKENS,
        other_tokens=OTHER_TOKENS,
    )
    model.load_weights(args.model_path)

    print(f"Evaluating {args.model_path} on the '{args.split}' split "
          f"({steps} batches)...")
    evaluate_with_denormalization(model, dataset, steps)


if __name__ == "__main__":
    main()
