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

def ranking_loss(y_true, y_pred, margin=0.1):
    """
    Efficient ranking loss for trading signals
    Focus on relative ordering rather than exact probabilities
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

def asymmetric_loss(y_true, y_pred, false_positive_cost=2.0, false_negative_cost=1.0):
    """
    Asymmetric loss that penalizes false positives more heavily
    Useful for trading where false buy signals can be more costly
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # False positive loss (predicting buy when should sell/hold)
    fp_loss = (1 - y_true) * y_pred * false_positive_cost
    
    # False negative loss (missing buy opportunities)
    fn_loss = y_true * (1 - y_pred) * false_negative_cost
    
    return tf.reduce_mean(fp_loss + fn_loss)

def contrastive_loss(y_true, y_pred, temperature=0.1):
    """
    Contrastive loss to better separate positive and negative samples
    """
    y_true = tf.cast(y_true, tf.float32)
    
    # Normalize predictions
    y_pred_norm = tf.nn.l2_normalize(tf.expand_dims(y_pred, -1), axis=-1)
    
    # Compute similarity matrix
    similarity = tf.matmul(y_pred_norm, y_pred_norm, transpose_b=True) / temperature
    
    # Create mask for positive pairs
    labels_equal = tf.equal(tf.expand_dims(y_true, 0), tf.expand_dims(y_true, 1))
    labels_equal = tf.cast(labels_equal, tf.float32)
    
    # Remove diagonal
    mask = tf.ones_like(similarity) - tf.eye(tf.shape(similarity)[0])
    labels_equal = labels_equal * mask
    
    # Compute contrastive loss
    exp_sim = tf.exp(similarity) * mask
    log_prob = similarity - tf.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8)
    
    loss = -tf.reduce_sum(labels_equal * log_prob) / (tf.reduce_sum(labels_equal) + 1e-8)
    
    return loss

# Recommended loss function for your trading model
def recommended_trading_loss(y_true, y_pred):
    """
    Optimized loss function for trading signal prediction
    Balances precision, handles class imbalance, and maintains smooth gradients
    """
    return trading_combined_loss(y_true, y_pred, 
                               focal_weight=0.4,
                               precision_weight=0.3, 
                               ranking_weight=0.3)

# Alternative: If you want to penalize false positives more heavily
def conservative_trading_loss(y_true, y_pred):
    """
    More conservative loss that heavily penalizes false buy signals
    """
    focal = focal_loss(y_true, y_pred, alpha=0.9)  # Higher alpha for more focus on minority class
    asymmetric = asymmetric_loss(y_true, y_pred, false_positive_cost=5.0)
    precision = smooth_precision_loss(y_true, y_pred)
    
    return 0.3 * focal + 0.4 * asymmetric + 0.3 * precision