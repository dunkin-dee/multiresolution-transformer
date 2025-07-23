import tensorflow as tf
from tensorflow.keras import backend as K

# Updated losses for individual outputs (target_high or target_low separately)
def asymmetric_huber_loss_single(delta=1.0, underestimate_weight=1.5, overestimate_weight=0.8):
    """
    Asymmetric Huber loss for individual outputs (either target_high or target_low)
    - For target_high: penalize underestimation more (missing profit opportunities)
    - For target_low: penalize overestimation more (false risk signals)
    Mixed precision compatible.
    """
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])  # Force 1D
        y_pred = tf.reshape(y_pred, [-1])
        # Ensure float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cast delta to float32 first
        delta_f32 = tf.cast(delta, tf.float32)
        underestimate_weight_f32 = tf.cast(underestimate_weight, tf.float32)
        overestimate_weight_f32 = tf.cast(overestimate_weight, tf.float32)
        
        error = y_true - y_pred
        
        # Separate positive and negative errors
        pos_mask = tf.cast(error >= 0, tf.float32)  # underestimation (y_true > y_pred)
        neg_mask = tf.cast(error < 0, tf.float32)   # overestimation (y_true < y_pred)
        
        # Huber loss components
        huber_loss = tf.where(
            tf.abs(error) <= delta_f32,
            0.5 * tf.square(error),
            delta_f32 * (tf.abs(error) - 0.5 * delta_f32)
        )
        
        # Apply asymmetric weights
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

# Updated compilation functions for dictionary-based outputs
def compile_model_recommended(model):
    """
    Comprehensive compilation setup for trading signal model with dictionary outputs
    Mixed precision compatible.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={
            # For target_high: penalize underestimating profit opportunities more
            'target_high': asymmetric_huber_loss_single(
                delta=1.0, 
                underestimate_weight=1.8,  # Heavy penalty for missing profits
                overestimate_weight=0.7    # Light penalty for overestimating profits
            ),
            # For target_low: penalize overestimating losses more (false risk signals)
            'target_low': asymmetric_huber_loss_single(
                delta=1.0,
                underestimate_weight=0.8,  # Light penalty for underestimating losses
                overestimate_weight=1.5    # Heavy penalty for overestimating losses
            )
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

def compile_model_lightweight(model):
    """
    Streamlined compilation with only the most important metrics
    Mixed precision compatible.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={
            'target_high': asymmetric_huber_loss_single(
                delta=1.5, 
                underestimate_weight=2.2, 
                overestimate_weight=0.6
            ),
            'target_low': asymmetric_huber_loss_single(
                delta=1.8,
                underestimate_weight=0.8,
                overestimate_weight=2.5
            )
        },
        loss_weights={
            'target_high': 1.3,
            'target_low': 1.0
        },
        metrics={
            'target_high': [
                'mae',
                'mse',
                profit_accuracy_metric(threshold=6.0),
                profit_precision_metric(threshold=6.0),
                profit_recall_metric(threshold=6.0),
            ],
            'target_low': [
                'mae',
                risk_accuracy_metric(threshold=-1.0)
            ]
        }
    )
    return model

def compile_model_trading_specific(model):
    """
    Trading-focused compilation using trading-specific loss functions
    Mixed precision compatible.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={
            'target_high': trading_signal_loss_single(
                missed_opportunity_penalty=2.5,  # Heavy penalty for missing profits
                false_signal_penalty=1.0,        # Light penalty for false profits
                base_weight=1.0
            ),
            'target_low': trading_signal_loss_single(
                missed_opportunity_penalty=1.0,  # Light penalty for missing losses
                false_signal_penalty=2.0,        # Heavy penalty for false risk signals
                base_weight=1.0
            )
        },
        loss_weights={
            'target_high': 1.3,
            'target_low': 1.0
        },
        metrics={
            'target_high': [
                'mae',
                profit_accuracy_metric(threshold=0.5),
                profit_precision_metric(threshold=0.5),
                profit_recall_metric(threshold=0.5),
                directional_accuracy_metric()
            ],
            'target_low': [
                'mae',
                risk_accuracy_metric(threshold=-0.5),
                directional_accuracy_metric()
            ]
        }
    )
    return model

# Example usage with your model
def setup_and_compile_model(model, compilation_type='recommended'):
    """
    Setup mixed precision and compile model
    """
    # Enable mixed precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: {policy.name}")
    except:
        print("Mixed precision not available, using float32")
    
    # Compile based on type
    if compilation_type == 'recommended':
        return compile_model_recommended(model)
    elif compilation_type == 'lightweight':
        return compile_model_lightweight(model)
    elif compilation_type == 'trading':
        return compile_model_trading_specific(model)
    else:
        raise ValueError("compilation_type must be 'recommended', 'lightweight', or 'trading'")

# Make sure your final model layers have the correct dtype for mixed precision
def ensure_model_final_layer_dtype(model):
    """
    Ensure the final layers use float32 for numerical stability with mixed precision
    """
    for layer in model.layers:
        if 'target_high' in layer.name or 'target_low' in layer.name:
            if hasattr(layer, 'dtype'):
                print(f"Layer {layer.name} dtype: {layer.dtype}")
                if layer.dtype != 'float32':
                    print(f"Warning: {layer.name} should use dtype='float32' for mixed precision stability")
    return model