import tensorflow as tf
from tensorflow.keras import backend as K

# Option 1: Asymmetric Huber Loss (RECOMMENDED) - Mixed Precision Compatible
def asymmetric_huber_loss(delta=1.0, high_weight=1.2, low_weight=0.8):
    """
    Asymmetric Huber loss that penalizes underestimating highs and overestimating lows more heavily.
    This is crucial for trading where missing upside or false downside signals are costly.
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cast delta to float32 first
        delta_f32 = tf.cast(delta, tf.float32)
        
        error = y_true - y_pred
        
        # Separate high and low predictions
        high_error = error[:, 0]  # target_high errors
        low_error = error[:, 1]   # target_low errors
        
        # For high predictions: penalize underestimation more (missing profit)
        high_pos_mask = tf.cast(high_error >= 0, tf.float32)  # underestimation
        high_neg_mask = tf.cast(high_error < 0, tf.float32)   # overestimation
        
        # For low predictions: penalize overestimation more (false risk signals)
        low_pos_mask = tf.cast(low_error >= 0, tf.float32)    # overestimation of loss
        low_neg_mask = tf.cast(low_error < 0, tf.float32)     # underestimation of loss
        
        # Huber loss components
        high_huber = tf.where(
            tf.abs(high_error) <= delta_f32,
            0.5 * tf.square(high_error),
            delta_f32 * (tf.abs(high_error) - 0.5 * delta_f32)
        )
        
        low_huber = tf.where(
            tf.abs(low_error) <= delta_f32,
            0.5 * tf.square(low_error),
            delta_f32 * (tf.abs(low_error) - 0.5 * delta_f32)
        )
        
        # Apply asymmetric weights (cast to float32)
        high_weight_f32 = tf.cast(high_weight, tf.float32)
        low_weight_f32 = tf.cast(low_weight, tf.float32)
        
        weighted_high = (high_pos_mask * high_weight_f32 + high_neg_mask) * high_huber
        weighted_low = (low_pos_mask * low_weight_f32 + low_neg_mask) * low_huber
        
        # Ensure loss is returned as float32
        return tf.cast(tf.reduce_mean(weighted_high + weighted_low), tf.float32)
    
    return loss

# Option 2: Quantile Loss for Risk-Aware Predictions - Mixed Precision Compatible
def quantile_loss(quantiles=[0.1, 0.5, 0.9]):
    """
    Quantile loss for capturing uncertainty in predictions.
    Useful for understanding confidence intervals of your targets.
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        total_loss = tf.constant(0.0, dtype=tf.float32)
        
        for i, q in enumerate(quantiles):
            q_f32 = tf.cast(q, tf.float32)
            error = y_true - y_pred
            loss_q = tf.maximum(q_f32 * error, (q_f32 - 1) * error)
            total_loss += tf.reduce_mean(loss_q)
            
        return tf.cast(total_loss / len(quantiles), tf.float32)
    
    return loss

# Option 3: Trading-Specific Custom Loss - Mixed Precision Compatible
def trading_signal_loss(profit_penalty=2.0, risk_penalty=1.5, base_weight=1.0):
    """
    Custom loss designed specifically for trading signals:
    - Heavily penalizes missing profitable opportunities (underestimating target_high)
    - Penalizes overestimating risk (overestimating negative target_low)
    - Includes magnitude awareness for outliers
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cast penalty values to float32 first
        profit_penalty_f32 = tf.cast(profit_penalty, tf.float32)
        risk_penalty_f32 = tf.cast(risk_penalty, tf.float32)
        base_weight_f32 = tf.cast(base_weight, tf.float32)
        
        high_true, low_true = y_true[:, 0], y_true[:, 1]
        high_pred, low_pred = y_pred[:, 0], y_pred[:, 1]
        
        # Base MSE components
        high_mse = tf.square(high_true - high_pred)
        low_mse = tf.square(low_true - low_pred)
        
        # Trading-specific penalties
        # Penalty for underestimating profit potential (missing buy signals)
        missed_profit_penalty = tf.maximum(0.0, high_true - high_pred) * profit_penalty_f32
        
        # Penalty for overestimating risk (false negative signals)
        false_risk_penalty = tf.maximum(0.0, low_pred - low_true) * risk_penalty_f32
        
        # Magnitude scaling for outliers (values > 3.0)
        threshold = tf.constant(3.0, dtype=tf.float32)
        high_magnitude_scale = tf.where(tf.abs(high_true) > threshold, 1.5, 1.0)
        low_magnitude_scale = tf.where(tf.abs(low_true) > threshold, 1.5, 1.0)
        
        # Combine losses
        high_loss = (high_mse + missed_profit_penalty) * high_magnitude_scale
        low_loss = (low_mse + false_risk_penalty) * low_magnitude_scale
        
        return tf.cast(tf.reduce_mean(base_weight_f32 * (high_loss + low_loss)), tf.float32)
    
    return loss

# Option 4: Focal Loss Adaptation for Regression - Mixed Precision Compatible
def focal_regression_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss adapted for regression - focuses learning on hard examples.
    Good for handling the imbalanced nature of extreme values (up to 11x).
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cast parameters to float32 first
        alpha_f32 = tf.cast(alpha, tf.float32)
        gamma_f32 = tf.cast(gamma, tf.float32)
        
        mse = tf.square(y_true - y_pred)
        
        # Compute focal weight based on prediction difficulty
        abs_error = tf.abs(y_true - y_pred)
        # Normalize error to [0,1] range for focal weighting
        max_error = tf.reduce_max(abs_error)
        normalized_error = abs_error / (max_error + tf.constant(1e-8, dtype=tf.float32))
        focal_weight = alpha_f32 * tf.pow(normalized_error, gamma_f32)
        
        return tf.cast(tf.reduce_mean(focal_weight * mse), tf.float32)
    
    return loss

# Alternative: Single combined loss for both outputs - Mixed Precision Compatible
def combined_trading_loss(high_weight=1.2, low_weight=1.0, profit_focus=1.5):
    """
    Single loss function that handles both outputs with trading-aware weighting
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        high_true, low_true = y_true[:, 0], y_true[:, 1]
        high_pred, low_pred = y_pred[:, 0], y_pred[:, 1]
        
        # Huber loss for robustness to outliers
        high_error = high_true - high_pred
        low_error = low_true - low_pred
        
        delta_f32 = tf.constant(1.0, dtype=tf.float32)
        
        high_huber = tf.where(
            tf.abs(high_error) <= delta_f32,
            0.5 * tf.square(high_error),
            tf.abs(high_error) - 0.5
        )
        
        low_huber = tf.where(
            tf.abs(low_error) <= delta_f32,
            0.5 * tf.square(low_error),
            tf.abs(low_error) - 0.5
        )
        
        # Cast weights to float32 first
        high_weight_f32 = tf.cast(high_weight, tf.float32)
        low_weight_f32 = tf.cast(low_weight, tf.float32)
        profit_focus_f32 = tf.cast(profit_focus, tf.float32)
        
        # Asymmetric penalty for missing profits (underestimating highs)
        profit_penalty = tf.maximum(0.0, high_error) * profit_focus_f32
        
        # Combined loss
        total_loss = (high_weight_f32 * (high_huber + profit_penalty) + 
                     low_weight_f32 * low_huber)
        
        return tf.cast(tf.reduce_mean(total_loss), tf.float32)
    
    return loss

# Custom metrics for comprehensive trading evaluation - Mixed Precision Compatible
def profit_accuracy_metric(threshold=0.5):
    """
    Custom metric to measure how well the model identifies profitable opportunities
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        # Cast to float32 for computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        high_true = y_true if len(y_true.shape) == 1 else y_true
        high_pred = y_pred if len(y_pred.shape) == 1 else y_pred
        
        # True positives: correctly identifying profitable opportunities
        profitable_true = tf.cast(high_true > threshold_f32, tf.float32)
        profitable_pred = tf.cast(high_pred > threshold_f32, tf.float32)
        
        correct_predictions = tf.cast(
            tf.equal(profitable_true, profitable_pred), tf.float32
        )
        
        return tf.cast(tf.reduce_mean(correct_predictions), tf.float32)
    
    return metric

def risk_accuracy_metric(threshold=-0.5):
    """
    Measures accuracy in identifying significant downside risk
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
        # Cast to float32 for computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        low_true = y_true if len(y_true.shape) == 1 else y_true
        low_pred = y_pred if len(y_pred.shape) == 1 else y_pred
        
        risky_true = tf.cast(low_true < threshold_f32, tf.float32)
        risky_pred = tf.cast(low_pred < threshold_f32, tf.float32)
        
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
        # Cast to float32 for computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        high_true = y_true if len(y_true.shape) == 1 else y_true
        high_pred = y_pred if len(y_pred.shape) == 1 else y_pred
        
        profitable_true = tf.cast(high_true > threshold_f32, tf.float32)
        profitable_pred = tf.cast(high_pred > threshold_f32, tf.float32)
        
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
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        threshold_f32 = tf.cast(threshold, tf.float32)
        
        high_true = y_true if len(y_true.shape) == 1 else y_true
        high_pred = y_pred if len(y_pred.shape) == 1 else y_pred
        
        profitable_true = tf.cast(high_true > threshold_f32, tf.float32)
        profitable_pred = tf.cast(high_pred > threshold_f32, tf.float32)
        
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
    Measures how well the model handles extreme values (>3x average candle size)
    Mixed precision compatible.
    """
    def metric(y_true, y_pred):
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
        
        # Return ratio - lower is better (outliers handled as well as normal cases)
        return tf.cast(outlier_error / (normal_error + epsilon), tf.float32)
    
    return metric

# Recommended compilation for your use case - Mixed Precision Compatible
def compile_model_recommended(model):
    """
    Comprehensive compilation setup for trading signal model with key metrics
    Mixed precision compatible.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={
            'target_high': asymmetric_huber_loss(delta=1.0, high_weight=1.5, low_weight=0.8),
            'target_low': asymmetric_huber_loss(delta=1.0, high_weight=0.8, low_weight=1.3)
        },
        loss_weights={
            'target_high': 1.2,  # Slightly prioritize profit prediction
            'target_low': 1.0
        },
        metrics={
            'target_high': [
                'mae', 
                'mse',
                profit_accuracy_metric(threshold=0.5),
                profit_precision_metric(threshold=0.5),
                profit_recall_metric(threshold=0.5),
                mean_absolute_percentage_error_custom,
                directional_accuracy_metric(),
                outlier_handling_metric(threshold=3.0)
            ],
            'target_low': [
                'mae',
                'mse', 
                risk_accuracy_metric(threshold=-0.5),
                mean_absolute_percentage_error_custom,
                directional_accuracy_metric(),
                outlier_handling_metric(threshold=3.0)
            ]
        }
    )
    return model

# Lightweight compilation for faster training - Mixed Precision Compatible
def compile_model_lightweight(model):
    """
    Streamlined compilation with only the most important metrics for quick iteration
    Mixed precision compatible.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={
            'target_high': asymmetric_huber_loss(delta=1.0, high_weight=1.5, low_weight=0.8),
            'target_low': asymmetric_huber_loss(delta=1.0, high_weight=0.8, low_weight=1.3)
        },
        loss_weights={
            'target_high': 1.2,
            'target_low': 1.0
        },
        metrics={
            'target_high': [
                'mae',
                profit_accuracy_metric(threshold=0.5),
                profit_precision_metric(threshold=0.5)
            ],
            'target_low': [
                'mae',
                risk_accuracy_metric(threshold=-0.5)
            ]
        }
    )
    return model

# Additional mixed precision optimization tip
def setup_mixed_precision_optimally():
    """
    Optimal mixed precision setup for your trading model
    """
    try:
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set: {policy.name}")
        
        # Verify it's working
        print(f"Compute dtype: {policy.compute_dtype}")  # Should be float16
        print(f"Variable dtype: {policy.variable_dtype}")  # Should be float32
        
        # Important: Make sure your model's final layer uses float32 for numerical stability
        print("Remember to add dtype='float32' to your final Dense layer:")
        print("Dense(2, activation='linear', dtype='float32', name='predictions')")
        
        return True
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
        return False