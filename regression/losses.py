"""Loss functions and trading-oriented metrics explored during development.

Only three of these are wired into the pipeline:

- ``asymmetric_huber_loss_single`` — used by ``regression.fine_tuner`` and
  ``regression.best_noise`` to punish underestimating the high more than
  overestimating it.
- ``profit_precision_metric`` / ``profit_recall_metric`` — treat "predicted value
  above threshold" as a signal and score it like a classifier, which is closer to
  how a prediction would actually be consumed than MAE is.

``regression.trainer`` uses plain MSE on both heads; that is what the committed
results were produced with. The remaining losses and metrics are kept as a record
of what was tried. Each entry point defines its own ``compile`` step rather than
importing one from here.
"""

import tensorflow as tf


def asymmetric_huber_loss_single(delta=1.0, underestimate_weight=1.5, overestimate_weight=0.8, max_error=50.0):
    """
    Robust asymmetric Huber loss that prevents gradient explosion
    - Clips extreme errors to prevent wild predictions from breaking training
    - Mixed precision compatible
    """
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        delta_f32 = tf.cast(delta, tf.float32)
        underestimate_weight_f32 = tf.cast(underestimate_weight, tf.float32)
        overestimate_weight_f32 = tf.cast(overestimate_weight, tf.float32)
        max_error_f32 = tf.cast(max_error, tf.float32)
        
        error = y_true - y_pred
        
        # CRITICAL FIX: Clip extreme errors
        # This prevents exploding gradients when model makes wild predictions
        error = tf.clip_by_value(error, -max_error_f32, max_error_f32)
        
        pos_mask = tf.cast(error >= 0, tf.float32)
        neg_mask = tf.cast(error < 0, tf.float32)
        
        huber_loss = tf.where(
            tf.abs(error) <= delta_f32,
            0.5 * tf.square(error),
            delta_f32 * (tf.abs(error) - 0.5 * delta_f32)
        )
        
        weighted_loss = (pos_mask * underestimate_weight_f32 + neg_mask * overestimate_weight_f32) * huber_loss
        
        return tf.cast(tf.reduce_mean(weighted_loss), tf.float32)
    
    return loss

# Trading-specific loss for individual outputs
def trading_signal_loss_single(missed_opportunity_penalty=2.0, false_signal_penalty=1.5, base_weight=1.0):
    """
    Trading-specific loss for individual outputs
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cast penalty values to float32
        missed_penalty_f32 = tf.cast(missed_opportunity_penalty, tf.float32)
        false_penalty_f32 = tf.cast(false_signal_penalty, tf.float32)
        base_weight_f32 = tf.cast(base_weight, tf.float32)
        
        # Base MSE
        mse = tf.square(y_true - y_pred)
        
        # Missed opportunities (underestimating positive values)
        missed_penalty = tf.maximum(0.0, y_true - y_pred) * missed_penalty_f32
        
        # False signals (overestimating when shouldn't)
        false_penalty = tf.maximum(0.0, y_pred - y_true) * false_penalty_f32
        
        # Magnitude scaling for outliers
        threshold = tf.constant(3.0, dtype=tf.float32)
        magnitude_scale = tf.where(tf.abs(y_true) > threshold, 1.5, 1.0)
        
        # Combine losses
        total_loss = (mse + missed_penalty + false_penalty) * magnitude_scale
        
        return tf.cast(tf.reduce_mean(base_weight_f32 * total_loss), tf.float32)
    
    return loss

# Focal loss for individual outputs
def focal_regression_loss_single(alpha=0.25, gamma=2.0):
    """
    Focal loss adapted for regression on individual outputs
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cast parameters to float32
        alpha_f32 = tf.cast(alpha, tf.float32)
        gamma_f32 = tf.cast(gamma, tf.float32)
        
        mse = tf.square(y_true - y_pred)
        
        # Compute focal weight based on prediction difficulty
        abs_error = tf.abs(y_true - y_pred)
        # Normalize error to [0,1] range for focal weighting
        max_error = tf.reduce_max(abs_error) + tf.constant(1e-8, dtype=tf.float32)
        normalized_error = abs_error / max_error
        focal_weight = alpha_f32 * tf.pow(normalized_error, gamma_f32)
        
        return tf.cast(tf.reduce_mean(focal_weight * mse), tf.float32)
    
    return loss

# Updated metrics for individual outputs
def profit_accuracy_metric(threshold=0.5):
    """
    Measures accuracy in identifying profitable opportunities for target_high
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        # Cast to float32 for computation
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        # True positives: correctly identifying profitable opportunities
        profitable_true = tf.cast(y_true > threshold_f32, tf.float32)
        profitable_pred = tf.cast(y_pred > threshold_f32, tf.float32)
        
        correct_predictions = tf.cast(
            tf.equal(profitable_true, profitable_pred), tf.float32
        )
        
        return tf.cast(tf.reduce_mean(correct_predictions), tf.float32)
    
    return metric

def risk_accuracy_metric(threshold=-0.5):
    """
    Measures accuracy in identifying significant downside risk for target_low
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        # Cast to float32 for computation
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        risky_true = tf.cast(y_true < threshold_f32, tf.float32)
        risky_pred = tf.cast(y_pred < threshold_f32, tf.float32)
        
        correct_predictions = tf.cast(
            tf.equal(risky_true, risky_pred), tf.float32
        )
        
        return tf.cast(tf.reduce_mean(correct_predictions), tf.float32)
    
    return metric

def profit_precision_metric(threshold=0.5):
    """
    Precision for profit predictions - of predicted profitable trades, how many actually are?
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        # Cast to float32 for computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        profitable_true = tf.cast(y_true > threshold_f32, tf.float32)
        profitable_pred = tf.cast(y_pred > threshold_f32, tf.float32)
        
        true_positives = tf.reduce_sum(profitable_true * profitable_pred)
        predicted_positives = tf.reduce_sum(profitable_pred)
        
        precision = tf.where(
            predicted_positives > 0,
            true_positives / predicted_positives,
            tf.constant(0.0, dtype=tf.float32)
        )
        
        return tf.cast(precision, tf.float32)
    
    return metric

def profit_recall_metric(threshold=0.5):
    """
    Recall for profit predictions - of actual profitable trades, how many did we catch?
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        # Cast to float32 for computation
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        profitable_true = tf.cast(y_true > threshold_f32, tf.float32)
        profitable_pred = tf.cast(y_pred > threshold_f32, tf.float32)
        
        true_positives = tf.reduce_sum(profitable_true * profitable_pred)
        actual_positives = tf.reduce_sum(profitable_true)
        
        recall = tf.where(
            actual_positives > 0,
            true_positives / actual_positives,
            tf.constant(0.0, dtype=tf.float32)
        )
        
        return tf.cast(recall, tf.float32)
    
    return metric

def mean_absolute_percentage_error_custom(y_true, y_pred):
    """
    MAPE that handles zero values gracefully
    Mixed precision compatible.
    """
    # Cast to float32 for computation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))
    return tf.cast(tf.reduce_mean(diff) * 100, tf.float32)

def directional_accuracy_metric():
    """
    Measures if the model correctly predicts the direction (positive/negative)
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        # Cast to float32 for computation
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        true_direction = tf.sign(y_true)
        pred_direction = tf.sign(y_pred)
        
        correct_directions = tf.cast(
            tf.equal(true_direction, pred_direction), tf.float32
        )
        
        return tf.cast(tf.reduce_mean(correct_directions), tf.float32)
    
    return metric

def outlier_handling_metric(threshold=3.0):
    """
    Measures how well the model handles extreme values
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        # Cast to float32 for computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        is_outlier = tf.cast(tf.abs(y_true) > threshold_f32, tf.float32)
        error = tf.abs(y_true - y_pred)
        
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        
        # Average error on outliers vs non-outliers
        outlier_error = tf.reduce_sum(error * is_outlier) / (tf.reduce_sum(is_outlier) + epsilon)
        normal_error = tf.reduce_sum(error * (1 - is_outlier)) / (tf.reduce_sum(1 - is_outlier) + epsilon)
        
        # Return ratio - lower is better
        return tf.cast(outlier_error / (normal_error + epsilon), tf.float32)
    
    return metric
