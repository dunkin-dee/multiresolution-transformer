import polars as pl
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


NUM_OF_TOKENS = 16
NUM_OF_CNN_TOKENS = 8


def prepare_data_lazy(file_path, num_prev=NUM_OF_TOKENS-1, cnn_tokens=NUM_OF_CNN_TOKENS, batch_size=32, shuffle=False):
    """
    Prepare data for the transformer model using Polars and lazy loading,
    with shuffling done by randomly selecting indices, and processing in batches.
    
    Args:
        file_path (str): Path to the CSV file.
        num_prev (int): Number of previous time frames to include.
        shuffle (bool): Whether to shuffle the data.
        batch_size (int): Number of rows to process in each batch.
    
    Yields:
        input_tensor (tf.Tensor): Tensor with shape (batch_size, num_prev + 1, 4).
        target_tensor (tf.Tensor): Tensor with shape (batch_size,).
    """
    # Load CSV lazily with Polars
    df_lazy = pl.scan_csv(file_path).select(['open_normalized', 'high_normalized', 'low_normalized', 'close_normalized', 'target'])

    # Collect the dataframe and determine total number of rows
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]

    # Create an array of indices to use for shuffling
    indices = list(range(num_prev, total_rows))

    # Shuffle indices if required
    if shuffle:
        np.random.shuffle(indices)

    input_list = []
    cnn_input_list = []
    target_list = []

    # Process the data in batches
    while indices:
        batch_indices = [indices.pop(0) for _ in range(min(batch_size, len(indices)))]

        for idx in batch_indices:
            # Fetch the previous `num_prev + 1` rows for the input based on the current index
            input_rows = df_collected[idx - num_prev:idx + 1, :-1].to_numpy()  # Exclude 'target' for input
            target_value = df_collected[idx, -1]  # Get 'target' for the target

            # Convert target to 0 if it's not 1
            target_value = 1 if target_value == 1 else 0

            input_list.append(input_rows)
            cnn_input_list.append(input_rows[:cnn_tokens])  # Select the first `cnn_tokens` rows for the CNN input
            target_list.append(target_value)

        # Convert lists to NumPy arrays
        input_array = np.array(input_list)
        cnn_input_array = np.array(cnn_input_list)
        cnn_input_array = np.expand_dims(cnn_input_array, axis=-1)
        target_array = np.array(target_list)

        # Convert NumPy arrays to TensorFlow tensors
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
        cnn_tensor = tf.convert_to_tensor(cnn_input_array, dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(target_array, dtype=tf.int32)

        yield (input_tensor, cnn_tensor), target_tensor

        # Reset lists for the next batch
        input_list.clear()
        cnn_input_list.clear()
        target_list.clear()




# Rest of your model building and training code remains unchanged
def create_dataset_generator(file_path, batch_size, shuffle=False):
    return tf.data.Dataset.from_generator(
        lambda: prepare_data_lazy(file_path, batch_size=batch_size, shuffle=shuffle),
        output_signature=(
            (tf.TensorSpec(shape=(None, NUM_OF_TOKENS, 4), dtype=tf.float32),
             tf.TensorSpec(shape=(None, NUM_OF_CNN_TOKENS, 4, 1), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )


# Custom loss function influenced by precision
def precision_influenced_loss(y_true, y_pred):
    # Ensure both y_true and y_pred are float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # Binary Crossentropy loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Precision metric
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    # Introduce a penalty term based on precision
    precision_penalty = 1.0 - precision

    # Combine binary cross-entropy with precision penalty
    loss = bce_loss + 0.2 * precision_penalty  # Adjust 0.5 as a weighting factor

    return loss

# Define a precision metric separately (you already defined it within F1, but it's good to have it standalone too)
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Positional Encoding remains the same
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def positional_encoding(self, maxlen, d_model):
        import numpy as np
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


def build_transformer_model(input_shape, d_model=64, num_heads=2, ff_dim=128, num_layers=4):
    inputs = Input(shape=input_shape)
    
    # Project the input (4 features per token) to the embedding dimension (d_model)
    x = Dense(d_model)(inputs)
    
    # Positional Encoding
    x = PositionalEncoding(input_shape[0], d_model)(x)
    
    # Transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x)
    
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_combined_model(transformer_input_shape, cnn_input_shape, d_model=64, num_heads=2, ff_dim=128, num_layers=4):
    # Transformer branch
    transformer_inputs = Input(shape=transformer_input_shape)
    x_transformer = Dense(d_model)(transformer_inputs)
    x_transformer = PositionalEncoding(transformer_input_shape[0], d_model)(x_transformer)
    for _ in range(num_layers):
        x_transformer = TransformerBlock(d_model, num_heads, ff_dim)(x_transformer)
    x_transformer = Dense(128, activation='relu')(x_transformer)
    x_transformer = tf.keras.layers.GlobalAveragePooling1D()(x_transformer)
    x_transformer = Dropout(0.1)(x_transformer)
    x_transformer = Dense(64, activation='relu')(x_transformer)

    # CNN branch
    cnn_inputs = Input(shape=cnn_input_shape)
    x_cnn = Conv2D(32, (3, 3), activation='relu')(cnn_inputs)
    x_cnn = MaxPooling2D((2, 2))(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(64, activation='relu')(x_cnn)

    print(x_transformer.shape)
    print(x_cnn.shape)


    # Merge both branches
    combined = Concatenate()([x_transformer, x_cnn])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.1)(combined)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[transformer_inputs, cnn_inputs], outputs=outputs)
    return model

# Define input shapes
transformer_input_shape = (NUM_OF_TOKENS, 4)  # 11 tokens for transformer
cnn_input_shape = (NUM_OF_CNN_TOKENS, 4, 1)  # 11 tokens reshaped for CNN

model = build_combined_model(transformer_input_shape, cnn_input_shape, d_model=16, ff_dim=32, num_heads=2, num_layers=2)

# Compile the combined model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


train_dataset = create_dataset_generator('recent_data/BTCUSD/train.csv', batch_size=32, shuffle=True).prefetch(tf.data.AUTOTUNE)
val_dataset = create_dataset_generator('recent_data/BTCUSD/val.csv', batch_size=32).prefetch(tf.data.AUTOTUNE)
test_dataset = create_dataset_generator('recent_data/BTCUSD/test.csv', batch_size=32).prefetch(tf.data.AUTOTUNE)

class_weight = {
    0:2,
    1:5
}
early_stopping = EarlyStopping(monitor='val_precision_m', 
                               patience=5, # Stops if there's no improvement in precision for 5 epochs
                               mode='max', 
                               verbose=1)

model_checkpoint = ModelCheckpoint('best_comb_model.h5', 
                                   monitor='val_precision_m', 
                                   save_best_only=True, 
                                   mode='max', 
                                   verbose=1)

# Train the model using the train and validation datasets
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    class_weight=class_weight,
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best model after training
model.load_weights('best_comb_model.h5')