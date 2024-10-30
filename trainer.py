import numpy as np
import polars as pl
import pandas as pd
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, LayerNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, Add, Embedding
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from working_data import clean_cols, clean_non_minute_rows, alt_label_df as label_df, normalize_by_window, split_df
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_builder_trans import combined_loss
from scipy.signal import savgol_filter

NORMALIZING_WINDOW_SIZE = 60*24
LABELING_WINDOW_SIZE = 20
POSITIVE_SLOPE = 0.3
LABEL_CUR_CANDLE_MULTIPLIER = 0
LABEL_MEAN_MULTIPLIER = 6
BATCH_SIZE = 32
NUM_TOKENS = 128
LOOKBACK_WINDOW = NUM_TOKENS + 1
D_MODEL = 128
FF_DIM = 256
NUM_HEADS = D_MODEL//16

source_csv = "data/GBPUSD/minutes.csv"
working_path = "working"

df = pd.read_csv(source_csv)
df = clean_non_minute_rows(df)
df = clean_cols(df)
break_point = len(df) - len(df)//2
# cols = ['open', 'high', 'low', 'close']
# signal = np.array(df[cols])
# signal = np.apply_along_axis(lambda x: savgol_filter(x, window_length=5, polyorder=4), axis=0, arr=signal)
# df[cols] = signal
df = df[break_point:]
df = normalize_by_window(
    df, 
    window_size=NORMALIZING_WINDOW_SIZE, 
    normalizing_cols=[
        'open',
        'high',
        'low',
        'close',
    ])
print("labeling")
df = label_df(df, window_size=LABELING_WINDOW_SIZE, mean_multiplier=LABEL_MEAN_MULTIPLIER, cur_candle_multiplier=LABEL_CUR_CANDLE_MULTIPLIER)
print(df['target'].value_counts())
split_df(
    df=df, 
    dump_path=working_path, 
    cols=[
        'open',
        'high',
        'low',
        'close',
        'open_normalized',
        'high_normalized',
        'low_normalized',
        'close_normalized',
        'target'
    ])


def prepare_data(file_path, num_tokens, window_size=1440, batch_size=32, smote=False, shuffle=False, cols=['open', 'high', 'low', 'close']):
    scaler = MinMaxScaler()
    stand_scaler = StandardScaler()
    emd_range = window_size
    # Load CSV lazily with Polars
    collect_cols = cols + ['target']
    df_lazy = pl.scan_csv(file_path).select(collect_cols)
    
    # Collect the dataframe and determine total number of rows
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]
    if smote:
        df_collected = df_collected.with_columns(pl.arange(0, total_rows).alias("index"))
        indices_target_1 = df_collected.filter(
            (pl.col("target") == 1) & (pl.col("index") >= emd_range)
        ).select("index").to_series().to_list()

        # Get indices where target is 0 and >= num_tokens
        indices_target_0 = df_collected.filter(
            (pl.col("target") == 0) & (pl.col("index") >= emd_range)
        ).select("index").to_series().to_list()

        df_collected = df_collected.drop('index')
    
    while True:  # Loop to reshuffle and restart at each epoch
        # Create an array of indices to use for shuffling
        indices = list(range(emd_range, total_rows))
        if smote:
            indices_target_1_complete = []
            while len(indices_target_1_complete) < len(indices_target_0):
                indices_target_1_complete += indices_target_1

            indices_target_1_complete = indices_target_1_complete[:len(indices_target_0)]

            indices = indices_target_0 + indices_target_1_complete

        # Shuffle indices if required
        if shuffle:
            np.random.shuffle(indices)

        input_list = []
        target_list = []

        for idx in indices:
            #create imfs
            signal = np.array(df_collected[cols][idx - emd_range + 1:idx + 1])
            
            # signal = scaler.fit_transform(signal)

            # Fetch the previous `num_prev + 1` rows for the input based on the current index
            input_rows = signal[-num_tokens:]
            target_value = df_collected[idx, -1]  # Get 'target' for the target

            # Append the input rows to the input list
            input_list.append(input_rows)
            target_list.append(target_value)

            # Yield once we have enough for a batch
            if len(input_list) == batch_size:
                # Convert lists to NumPy arrays
                input_array = np.array(input_list)
                target_array = np.array(target_list)

                # Convert NumPy arrays to TensorFlow tensors
                input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
                target_tensor = tf.convert_to_tensor(target_array, dtype=tf.int32)

                yield input_tensor, target_tensor

                # Reset lists for the next batch

                input_list.clear()
                target_list.clear()

        break

def create_dataset_generator(file_path, batch_size, num_tokens, window_size=LOOKBACK_WINDOW, shuffle=False, repeat=False, smote=False, cols=['open', 'high', 'low', 'close']):
    dataset = tf.data.Dataset.from_generator(
        lambda: prepare_data(file_path, window_size=window_size, batch_size=batch_size, num_tokens=num_tokens, shuffle=shuffle, smote=smote, cols=cols),
        output_signature=(
            tf.TensorSpec(shape=(None, num_tokens, len(cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset 

def get_total_rows(file_path, num_tokens, smote=False):
    # Count the total number of rows in the CSV file
    df_lazy = pl.scan_csv(file_path)
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]
    if not smote:
        return total_rows
    df_collected = df_collected.with_columns(pl.arange(0, total_rows).alias("index"))
    indices_target_0 = df_collected.filter(
            (pl.col("target") == 0) & (pl.col("index") >= num_tokens)
    )
    return len(indices_target_0) * 2

train_path = 'working/train.csv'
val_path = 'working/val.csv'
test_path = 'working/test.csv'

cols = [
    'open_normalized',
    'high_normalized',
    'low_normalized',
    'close_normalized'
]

train_dataset = create_dataset_generator(train_path, batch_size=BATCH_SIZE, num_tokens=NUM_TOKENS, repeat=True, shuffle=True, smote=True, cols=cols)
val_dataset = create_dataset_generator(val_path, batch_size=BATCH_SIZE, num_tokens=NUM_TOKENS, repeat=True, cols=cols)
test_dataset = create_dataset_generator(test_path, batch_size=BATCH_SIZE, num_tokens=NUM_TOKENS, cols=cols)

train_steps = get_total_rows(train_path, num_tokens=LOOKBACK_WINDOW, smote=True)//BATCH_SIZE
val_steps = get_total_rows(val_path, num_tokens=LOOKBACK_WINDOW)//BATCH_SIZE
test_steps = get_total_rows(test_path, num_tokens=LOOKBACK_WINDOW)//BATCH_SIZE


input_length = NUM_TOKENS 

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

auc =  tf.keras.metrics.AUC()
auc.reset_state()
prec = tf.keras.metrics.Precision()
prec.reset_state()

input_shape = (NUM_TOKENS, 4)

input_layer = Input(shape=input_shape)
x = Conv1D(filters=D_MODEL//2, kernel_size=3, activation='relu', padding="same")(input_layer)
x = Conv1D(filters=D_MODEL, kernel_size=3, activation='relu', padding="same")(x)
x = MaxPooling1D(pool_size=2, padding="same")(x)
x = Dropout(0.1)(x)
x = PositionalEncoding(x.shape[1], d_model=D_MODEL)(x)
x = TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM)(x, training=True)
x = TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM)(x, training=True)
x = TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM)(x, training=True)
x = TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM)(x, training=True)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=outputs)

model.load_weights("best_cnn_trans_model.keras", skip_mismatch=True)

learning_rate = 1e-3  # Adjust this value as needed
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# Compile the model
model.compile(
    optimizer=optimizer, 
    loss=combined_loss, 
    metrics=['accuracy',auc, prec])

model.summary()

early_stopping = EarlyStopping(monitor='val_precision_1', 
                               patience=10, # Stops if there's no improvement in precision for 5 epochs
                               mode='max', 
                               verbose=1)

model_checkpoint = ModelCheckpoint('best_cnn_trans_model.keras', 
                                   monitor='val_precision_1', 
                                   save_best_only=True, 
                                   mode='max', 
                                   verbose=1)

# model.load_weights('best_cnn_trans_model.keras')

# Train the model using the train and validation datasets
history = model.fit(
    train_dataset,
    epochs=50,
    steps_per_epoch = train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=[early_stopping, model_checkpoint]
)


# Load the best model after training
model.load_weights('best_cnn_trans_model.keras')


