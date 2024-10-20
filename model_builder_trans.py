import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, Reshape
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
    return bce + 0.3*auc + 0.2*precision

# Positional Encoding remains the same

def build_transformer_model(input_shape, d_model=64, num_heads=2, ff_dim=128, num_layers=4, training=True):
    inputs = Input(shape=input_shape)
    
    # Project the input (4 features per token) to the embedding dimension (d_model)
    x = Dense(d_model)(inputs)
    
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
