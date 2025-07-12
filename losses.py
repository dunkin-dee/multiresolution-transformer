import tensorflow as tf

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss to handle class imbalance in trading signals
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Compute focal loss
    ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
    
    return focal_weight * ce_loss

def smooth_precision_loss(y_true, y_pred, epsilon=1e-7):
    """
    Differentiable approximation of precision loss
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Smooth approximations of TP, FP
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    
    # Smooth precision
    precision = tp / (tp + fp + epsilon)
    
    return 1.0 - precision

def smooth_recall_loss(y_true, y_pred, epsilon=1e-7):
    """
    Differentiable approximation of recall loss
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Smooth approximations of TP, FN
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Smooth recall
    recall = tp / (tp + fn + epsilon)
    
    return 1.0 - recall

def smooth_f1_loss(y_true, y_pred, epsilon=1e-7):
    """
    Differentiable approximation of F1 score loss
    Balances precision and recall automatically
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Smooth approximations of TP, FP, FN
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Smooth precision and recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return 1.0 - f1

def smooth_f_beta_loss(y_true, y_pred, beta=1.0, epsilon=1e-7):
    """
    Differentiable F-beta score loss
    beta > 1 emphasizes recall, beta < 1 emphasizes precision
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Smooth approximations of TP, FP, FN
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Smooth precision and recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    # F-beta score
    beta_squared = beta ** 2
    f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + epsilon)
    
    return 1.0 - f_beta

def balanced_asymmetric_loss(y_true, y_pred, false_positive_cost=2.0, false_negative_cost=1.5):
    """
    Asymmetric loss with more balanced FP/FN penalties
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # False positive loss (predicting buy when should sell/hold)
    fp_loss = (1 - y_true) * y_pred * false_positive_cost
    
    # False negative loss (missing buy opportunities) - increased weight
    fn_loss = y_true * (1 - y_pred) * false_negative_cost
    
    return tf.reduce_mean(fp_loss + fn_loss)

def precision_recall_balanced_loss(y_true, y_pred, 
                                  precision_weight=0.6, 
                                  recall_weight=0.4, 
                                  epsilon=1e-7):
    """
    Directly optimizes precision and recall with custom weights
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Smooth approximations
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Precision and recall losses
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    precision_loss = 1.0 - precision
    recall_loss = 1.0 - recall
    
    return precision_weight * precision_loss + recall_weight * recall_loss

def trading_precision_recall_loss(y_true, y_pred, 
                                 focal_weight=0.3, 
                                 precision_weight=0.35, 
                                 recall_weight=0.25,
                                 ranking_weight=0.1):
    """
    Balanced trading loss that considers both precision and recall
    """
    # Focal loss for handling class imbalance
    focal = focal_loss(y_true, y_pred, alpha=0.5)  # More balanced alpha
    
    # Precision and recall losses
    precision = smooth_precision_loss(y_true, y_pred)
    recall = smooth_recall_loss(y_true, y_pred)
    
    # Ranking loss for better signal ordering
    ranking = ranking_loss(y_true, y_pred)
    
    # Combine losses
    total_loss = (focal_weight * focal + 
                  precision_weight * precision + 
                  recall_weight * recall +
                  ranking_weight * ranking)
    
    return total_loss

def ranking_loss(y_true, y_pred, margin=0.1):
    """
    Efficient ranking loss for trading signals
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Get positive and negative samples
    pos_mask = tf.cast(tf.equal(y_true, 1), tf.float32)
    neg_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    
    # Compute mean predictions for positive and negative samples
    pos_pred = tf.reduce_sum(y_pred * pos_mask) / (tf.reduce_sum(pos_mask) + 1e-8)
    neg_pred = tf.reduce_sum(y_pred * neg_mask) / (tf.reduce_sum(neg_mask) + 1e-8)
    
    # Ranking loss: positive samples should have higher predictions
    loss = tf.maximum(0.0, margin - (pos_pred - neg_pred))
    
    return loss

def adaptive_threshold_loss(y_true, y_pred, target_precision=0.7, target_recall=0.6):
    """
    Loss that adapts to maintain target precision and recall levels
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Compute current precision and recall
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    # Penalties for falling below targets
    precision_penalty = tf.maximum(0.0, target_precision - precision)
    recall_penalty = tf.maximum(0.0, target_recall - recall)
    
    # Base cross-entropy loss
    base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    return base_loss + 2.0 * precision_penalty + 2.0 * recall_penalty

# Recommended balanced loss functions

def recommended_balanced_loss(y_true, y_pred):
    """
    F1-optimized loss with focal weighting for class imbalance
    Good balance between precision and recall
    """
    focal = focal_loss(y_true, y_pred, alpha=0.5, gamma=1.5)
    f1 = smooth_f1_loss(y_true, y_pred)
    
    return 0.4 * focal + 0.6 * f1

def precision_focused_balanced_loss(y_true, y_pred):
    """
    Still emphasizes precision but gives recall meaningful weight
    """
    return trading_precision_recall_loss(y_true, y_pred,
                                       focal_weight=0.2,
                                       precision_weight=0.4,
                                       recall_weight=0.3,
                                       ranking_weight=0.1)

def recall_boosted_loss(y_true, y_pred):
    """
    For when you want to catch more opportunities (higher recall)
    while maintaining reasonable precision
    """
    f_beta = smooth_f_beta_loss(y_true, y_pred, beta=1.5)  # beta > 1 emphasizes recall
    focal = focal_loss(y_true, y_pred, alpha=0.6)  # Higher alpha for minority class
    
    return 0.6 * f_beta + 0.4 * focal

def dynamic_balanced_loss(y_true, y_pred):
    """
    Adaptive loss that maintains target precision/recall ratios
    """
    return adaptive_threshold_loss(y_true, y_pred, 
                                 target_precision=0.75, 
                                 target_recall=0.65)


def trading_combined_loss(y_true, y_pred, 
                         focal_weight=0.4, 
                         precision_weight=0.3, 
                         ranking_weight=0.3):
    """
    Combined loss specifically designed for trading signal prediction
    """
    # Focal loss for handling class imbalance
    focal = focal_loss(y_true, y_pred)
    
    # Smooth precision loss
    precision = smooth_precision_loss(y_true, y_pred)
    
    # Ranking loss for better signal ordering
    ranking = ranking_loss(y_true, y_pred)
    
    # Combine losses
    total_loss = (focal_weight * focal + 
                  precision_weight * precision + 
                  ranking_weight * ranking)
    
    return total_loss

def recommended_trading_loss(y_true, y_pred):
    """
    Optimized loss function for trading signal prediction
    Balances precision, handles class imbalance, and maintains smooth gradients
    """
    return trading_combined_loss(y_true, y_pred, 
                               focal_weight=0.4,
                               precision_weight=0.3, 
                               ranking_weight=0.3)

