import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# Positional Encoding remains the same
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def positional_encoding(self, maxlen, d_model):
        positions = np.arange(maxlen)[:, np.newaxis]
        angles = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (angles // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def auc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    
    # Approximate ranking using pairwise comparisons
    n = tf.shape(y_true)[0]
    pair_wise_diff = y_pred[:, None] - y_pred[None, :]
    
    # Use sigmoid to approximate step function
    pair_wise_rank = tf.sigmoid(pair_wise_diff)
    
    # Compute AUC
    pos_mask = y_true[:, None] * (1 - y_true[None, :])
    neg_mask = (1 - y_true[:, None]) * y_true[None, :]
    auc = tf.reduce_sum(pair_wise_rank * pos_mask) / (tf.reduce_sum(pos_mask) + 1e-8)
    
    
    # Return negative AUC as we want to maximize it
    return 1.0 - auc

precision_metric = tf.keras.metrics.Precision()

def precision_loss(y_true, y_pred):
    # Reset the metric's state
    precision_metric.reset_state()
    
    # Update the state with current batch's y_true and y_pred
    precision_metric.update_state(y_true, y_pred)
    
    # Return 1 - precision to minimize the loss
    return 1 - precision_metric.result()

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    precision = precision_loss(y_true, y_pred)
    auc = auc_loss(y_true, y_pred)
    # return bce
    return 0.7*bce + 0.3*auc + 0.5*precision

# Positional Encoding remains the same

def build_transformer_model(input_shape, d_model=64, num_heads=2, ff_dim=128, num_layers=4, training=True):
    inputs = Input(shape=input_shape)

    print(inputs.shape)
    
    # Project the input (4 features per token) to the embedding dimension (d_model)
    x = Dense(d_model)(inputs)

    print(x.shape)
    
    # Positional Encoding
    x = PositionalEncoding(input_shape[0], d_model)(x)
    
    # Transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x, training=training)
    
    x = Dense(num_layers*64, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(num_layers*32, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_combined_bb_model(transformer_input_shape, bb_input_shape, d_model=64, num_heads=2, ff_dim=128, num_layers=4, training=True):
    # Transformer branch
    transformer_inputs = Input(shape=transformer_input_shape)
    x_transformer = Dense(d_model)(transformer_inputs)
    x_transformer = PositionalEncoding(transformer_input_shape[0], d_model)(x_transformer)
    for _ in range(num_layers):
        x_transformer = TransformerBlock(d_model, num_heads, ff_dim)(x_transformer, training=training)
    x_transformer = Dense(num_layers*16, activation='relu')(x_transformer)
    x_transformer = tf.keras.layers.GlobalAveragePooling1D()(x_transformer)
    x_transformer = Dropout(0.1)(x_transformer)
    x_transformer = Dense(num_layers*8, activation='relu')(x_transformer)

    # CNN branch
    bb_inputs = Input(shape=bb_input_shape)
    b_transformer = Dense(d_model)(bb_inputs)
    b_transformer = PositionalEncoding(bb_input_shape[0], d_model)(b_transformer)
    for _ in range(num_layers):
        b_transformer = TransformerBlock(d_model, num_heads, ff_dim)(b_transformer, training=training)
    b_transformer = Dense(num_layers*16, activation='relu')(b_transformer)
    b_transformer = tf.keras.layers.GlobalAveragePooling1D()(b_transformer)
    b_transformer = Dropout(0.1)(b_transformer)
    b_transformer = Dense(num_layers*8, activation='relu')(b_transformer)


    # Merge both branches
    combined = Concatenate()([x_transformer, b_transformer])
    combined = Dense(num_layers*16, activation='relu')(combined)
    combined = Dropout(0.1)(combined)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[transformer_inputs, bb_inputs], outputs=outputs)
    return model


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