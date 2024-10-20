import pandas as pd
import tensorflow as tf
import os
from working_data import clean_cols, clean_non_minute_rows, label_df, normalize_by_window, split_df
from data_builder import create_dataset_generator, get_total_rows
from model_builder import build_combined_model, combined_loss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

NORMALIZING_WINDOW_SIZE = 180
LABELING_WINDOW_SIZE = 20
POSITIVE_SLOPE = 0.4
LABEL_CANDLE_MULTIPLIER = 4
LABEL_MULTIPLIER = 2
BATCH_SIZE = 32
NUM_TRANSFORMER_TOKENS = 64
NUM_CNN_TOKENS = 8
D_MODEL = 128
FF_DIM = 256
NUM_HEADS = 2
NUM_TRANSFORMER_LAYERS = 2

source_csv = "data/GBPUSD/minutes.csv"
working_path = "working"

df = pd.read_csv(source_csv)
df = clean_non_minute_rows(df)
break_point = len(df) - len(df)//4
df = df[break_point:]
df = clean_cols(df) 
df = normalize_by_window(df, window_size=NORMALIZING_WINDOW_SIZE)
df = label_df(df, window_size=LABELING_WINDOW_SIZE, multiplier=LABEL_MULTIPLIER, candle_multiplier=LABEL_CANDLE_MULTIPLIER)
split_df(df=df, dump_path=working_path)

transformer_input_shape = (NUM_TRANSFORMER_TOKENS, 4)
cnn_input_shape = (NUM_CNN_TOKENS, 4, 1)
model = build_combined_model(transformer_input_shape, cnn_input_shape, d_model=D_MODEL, ff_dim=FF_DIM, num_heads=NUM_HEADS, num_layers=NUM_TRANSFORMER_LAYERS)
auc =  tf.keras.metrics.AUC()
auc.reset_state()
prec = tf.keras.metrics.Precision()
prec.reset_state()
model.compile(optimizer='adam', 
              loss=combined_loss, 
              metrics=['accuracy',auc, prec])

train_file = os.path.join(working_path, 'train.csv')
val_file = os.path.join(working_path, 'val.csv')
test_file = os.path.join(working_path, 'test.csv')

train_dataset= create_dataset_generator(train_file, batch_size=BATCH_SIZE, num_transformer_tokens=NUM_TRANSFORMER_TOKENS, num_cnn_tokens=NUM_CNN_TOKENS, shuffle=True, repeat=True).prefetch(tf.data.AUTOTUNE)
val_dataset = create_dataset_generator(val_file, batch_size=BATCH_SIZE, num_transformer_tokens=NUM_TRANSFORMER_TOKENS, num_cnn_tokens=NUM_CNN_TOKENS, repeat=True).prefetch(tf.data.AUTOTUNE)
test_dataset = create_dataset_generator(test_file, batch_size=BATCH_SIZE, num_transformer_tokens=NUM_TRANSFORMER_TOKENS, num_cnn_tokens=NUM_CNN_TOKENS).prefetch(tf.data.AUTOTUNE)

train_steps_per_epoch = get_total_rows(train_file)//BATCH_SIZE
val_steps_per_epoch = get_total_rows(val_file)//BATCH_SIZE
test_steps_per_epoch = get_total_rows(test_file)//BATCH_SIZE

class_weight = {
    0:1,
    1:6
}
early_stopping = EarlyStopping(monitor='val_auc', 
                               patience=5, # Stops if there's no improvement in precision for 5 epochs
                               mode='max', 
                               verbose=1)

model_checkpoint = ModelCheckpoint('best_comb_model.keras', 
                                   monitor='val_auc', 
                                   save_best_only=True, 
                                   mode='max', 
                                   verbose=1)

# Train the model using the train and validation datasets
history = model.fit(
    train_dataset,
    epochs=20,
    steps_per_epoch = train_steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps_per_epoch,
    class_weight=class_weight,
    callbacks=[early_stopping, model_checkpoint]
)


# Load the best model after training
model.load_weights('best_comb_model.keras')

predictions = model.predict(test_dataset)

# Convert probabilities to binary class (0 or 1) using a threshold of 0.5
predicted_class = (predictions > 0.5).astype(int)

y_true = []
y_pred = []

for input_tensor, target_tensor in test_dataset:
    y_true.extend(target_tensor.numpy()) 

conf_matrix = confusion_matrix(y_true, predicted_class)

print("Confusion Matrix:\n", conf_matrix)