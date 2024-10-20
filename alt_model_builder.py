import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate
from tensorflow.keras.models import Model
import tensorflow_models as tfm  # Make sure you have TensorFlow Models installed


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
    return bce + 0.2 * auc + 0.5 * precision

def build_combined_model(transformer_input_shape, cnn_input_shape, d_model=64, num_heads=2, ff_dim=128, num_layers=4):
    # Transformer branch
    transformer_inputs = Input(shape=transformer_input_shape)
    
    # Add a Dense layer to transform input to (batch_size, seq_length, d_model)
    x_transformer = Dense(d_model)(transformer_inputs)  # This will give (batch_size, seq_length, d_model)

    print("Shape after Dense layer:", x_transformer.shape)

    # TransformerDecoderBlock (uses tfm.nlp.layers.TransformerDecoderBlock)
    for _ in range(num_layers):
        transformer_block = tfm.nlp.layers.TransformerDecoderBlock(
            num_attention_heads=num_heads, 
            intermediate_size=ff_dim, 
            intermediate_activation='relu',
            dropout_rate=0.1
        )
        x_transformer = transformer_block(x_transformer, training=True)

    x_transformer = Dense(32, activation='relu')(x_transformer)
    x_transformer = tf.keras.layers.GlobalAveragePooling1D()(x_transformer)
    x_transformer = Dropout(0.1)(x_transformer)
    x_transformer = Dense(16, activation='relu')(x_transformer)

    # CNN branch
    cnn_inputs = Input(shape=cnn_input_shape)
    # First Block: Two convolutional layers followed by a max pooling layer
    x_cnn = Conv2D(16, (1, 4), activation='relu', padding='same', strides=4)(cnn_inputs)
    x_cnn = AveragePooling2D((1, 4), padding='same', strides=4)(x_cnn)

    # Second Block: Two convolutional layers followed by a max pooling layer
    x_cnn = Conv2D(8, (1, 4), activation='relu', padding='same')(x_cnn)
    x_cnn = Conv2D(8, (1, 4), activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling2D((1, 2), padding='same')(x_cnn)

    # Flatten the output to feed into a Dense layer
    x_cnn = Flatten()(x_cnn)

    # Dense layers with Dropout
    x_cnn = Dense(128, activation='relu')(x_cnn)
    x_cnn = Dropout(0.1)(x_cnn)  # Add dropout for regularization

    # Merge both branches
    combined = Concatenate()([x_transformer, x_cnn])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.1)(combined)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[transformer_inputs, cnn_inputs], outputs=outputs)
    return model

