import datetime
import os
import tensorflow as tf
import numpy as np
from model_builder_trans import recommended_trading_loss
from tensorflow.keras.models import save_model, load_model, Model
from tensorflow.keras.layers import Input, Conv1D,  Dense, concatenate, Dropout, Layer, GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization, Add, Embedding
from global_constants import NUM_TOKENS, OTHER_TOKENS, BATCH_SIZE, LOOKBACK_WINDOW, NUM_HEADS, FF_DIM, D_MODEL
from multi_instrument_data_generator import InstrumentConfig, MultiInstrumentDatasetConfig, create_multi_instrument_dataset
from transformer_builder import LearnablePositionalEncoding, StochasticGatedTransformerBlock, WarmupCosineDecay
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint



auc =  tf.keras.metrics.AUC()
auc.reset_state()
prec = tf.keras.metrics.Precision()
prec.reset_state()

# State preservation paths
CHECKPOINT_DIR = "./training_state"
BEST_MODEL_PATH = "best_multiresolution_model.keras"
TRAINING_STATE_PATH = f"{CHECKPOINT_DIR}/training_state.ckpt"

feature_cols = [
    'open_normalized',
    'high_normalized',
    'low_normalized',
    'close_normalized'
]

working_path = "split_data"

starting_dir = "final_data"

instruments = os.listdir(working_path)

train_instrument_configs = []
val_instrument_configs = []
test_instrument_configs = []

for instrument in instruments:
    train_instrument_configs.append(
        InstrumentConfig(
            name=instrument,
            hourly_data_path=f"{working_path}/{instrument}/hour.csv",
            chunked_data_dir=f"{working_path}/{instrument}/training"
        )
    )
    val_instrument_configs.append(
        InstrumentConfig(
            name=instrument,
            hourly_data_path=f"{working_path}/{instrument}/hour.csv",
            chunked_data_dir=f"{working_path}/{instrument}/validation"
        )
    )
    test_instrument_configs.append(
        InstrumentConfig(
            name=instrument,
            hourly_data_path=f"{working_path}/{instrument}/hour.csv",
            chunked_data_dir=f"{working_path}/{instrument}/testing"
        )
    )

train_config = MultiInstrumentDatasetConfig(
    instruments=train_instrument_configs,
    main_lookback_tokens=NUM_TOKENS,
    hourly_lookback_tokens=OTHER_TOKENS,
    lookback_window=LOOKBACK_WINDOW,
    batch_size=BATCH_SIZE,
    apply_smote=True,
    shuffle_data=True,
    feature_columns=feature_cols,
    max_chunks_per_instrument=25
)

val_config = MultiInstrumentDatasetConfig(
    instruments=val_instrument_configs,
    main_lookback_tokens=NUM_TOKENS,
    hourly_lookback_tokens=OTHER_TOKENS,
    lookback_window=LOOKBACK_WINDOW,
    batch_size=BATCH_SIZE,
    apply_smote=False,
    shuffle_data=False,
    feature_columns=feature_cols,
    max_chunks_per_instrument=25
)

test_config = MultiInstrumentDatasetConfig(
    instruments=test_instrument_configs,
    main_lookback_tokens=NUM_TOKENS,
    hourly_lookback_tokens=OTHER_TOKENS,
    lookback_window=LOOKBACK_WINDOW,
    batch_size=BATCH_SIZE,
    apply_smote=False,
    shuffle_data=False,
    feature_columns=feature_cols,
    max_chunks_per_instrument=25
)


train_dataset, train_rows = create_multi_instrument_dataset(
    config=train_config,
    repeat_dataset=True
)
val_dataset, val_rows = create_multi_instrument_dataset(
    config=val_config,
    repeat_dataset=True
)
test_dataset, test_rows = create_multi_instrument_dataset(
    config=test_config,
    repeat_dataset=True
)

train_steps = train_rows//BATCH_SIZE
val_steps = val_rows//BATCH_SIZE
test_steps = test_rows//BATCH_SIZE

# Create checkpoint structure
training_state = tf.train.Checkpoint(
    model=tf.Variable(0),  # Placeholder
    optimizer=tf.Variable(0),
    current_epoch=tf.Variable(0, dtype=tf.int64),
    best_val_loss=tf.Variable(float('inf'), dtype=tf.float32),
    wait_counter=tf.Variable(0, dtype=tf.int64),
    lr_schedule_step=tf.Variable(0, dtype=tf.int64)
)


def create_model(training=True):
    # Main (5-minute) data input
    input_shape = (NUM_TOKENS, len(feature_cols))
    input_layer = Input(shape=input_shape, name='minute_input')
    
    # Hourly data input
    other_input_shape = (OTHER_TOKENS, len(feature_cols))
    other_input_layer = Input(shape=other_input_shape, name='hourly_input')
    
    # Custom layer for adding type embeddings
    class AddTypeEmbedding(Layer):
        def __init__(self, type_id, embed_dim=16, **kwargs):
            super().__init__(**kwargs)
            self.type_id = type_id
            self.embed_dim = embed_dim
            self.embedding_layer = None
            
        def build(self, input_shape):
            self.embedding_layer = Embedding(input_dim=2, output_dim=self.embed_dim, name=f'{self.name}_embed')
            super().build(input_shape)
            
        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            type_ids = tf.fill((batch_size, seq_len), self.type_id)
            type_embed = self.embedding_layer(type_ids)
            return tf.concat([inputs, type_embed], axis=-1)
            
        def get_config(self):
            config = super().get_config()
            config.update({
                "type_id": self.type_id,
                "embed_dim": self.embed_dim
            })
            return config
        
    
    
    # Improved CNN block function
    def cnn_feature_extractor(inputs, name_prefix, max_filter_output=D_MODEL-16):
        # Multi-scale feature extraction
        x = Conv1D(filters=max_filter_output//2, kernel_size=3, activation='relu', padding="same", name=f'{name_prefix}_conv1')(inputs)      
        x = Conv1D(filters=max_filter_output, kernel_size=5, activation='relu', padding="same", name=f'{name_prefix}_conv3')(x)
   
        return x
    
    # Process 5-minute data branch
    x = cnn_feature_extractor(input_layer, '5min', D_MODEL-16)
    x = Dense(D_MODEL - 16, name='5min_projection')(x)  # Reserve 16 dims for type embedding
    x = LearnablePositionalEncoding(maxlen=NUM_TOKENS, d_model=D_MODEL - 16)(x)
    
    # Add type embedding for 5-minute data (type 0)
    x = AddTypeEmbedding(type_id=0, embed_dim=16, name='5min_type_embed')(x)

    # Process hourly data branch
    h = cnn_feature_extractor(other_input_layer, 'hourly', D_MODEL-16)
    h = Dense(D_MODEL - 16, name='hourly_projection')(h)  # Reserve 16 dims for type embedding
    h = LearnablePositionalEncoding(maxlen=OTHER_TOKENS, d_model=D_MODEL - 16)(h)
    
    # Add type embedding for hourly data (type 1)
    h = AddTypeEmbedding(type_id=1, embed_dim=16, name='hourly_type_embed')(h)

    x = concatenate([h, x], axis=1, name='timeframe_concat')  # Shape: (batch, time_combined, d_model)
    x = Dropout(0.1)(x)

    # Deeper transformer stack with increasing stochastic depth
    stochastic_rates = [0.05, 0.1, 0.15, 0.2]
    for i in range(len(stochastic_rates)):  # or 6 if you expand later
        # Pre-norm the input
        x_norm = LayerNormalization(name=f'ln_pre_{i}')(x)

        # Pass through the transformer block
        x_block_out = StochasticGatedTransformerBlock(
            D_MODEL,
            NUM_HEADS,
            FF_DIM,
            rate=0.1,
            attention_dropout=stochastic_rates[i],
            stochastic_depth_rate=stochastic_rates[i]
        )(x_norm, training=training)

        # Apply dropout after the block
        x_block_out = Dropout(0.05, name=f'post_dropout_{i}')(x_block_out)

        # Residual connection: input + block output
        x = Add(name=f'residual_add_{i}')([x, x_block_out])


    # Global pooling with layer norm
    avg_pool = GlobalAveragePooling1D(name='avg_pool')(x)
    max_pool = GlobalMaxPooling1D(name='max_pool')(x)
    
    # Additional feature: attention-weighted pooling
    class AttentionPooling(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.attention_dense = Dense(1)  # Remove activation here
            
        def call(self, inputs):
            # inputs shape: (batch_size, seq_len, hidden_dim)
            attention_logits = self.attention_dense(inputs)  # (batch_size, seq_len, 1)
            attention_weights = tf.nn.softmax(attention_logits, axis=1)  # Softmax across sequence
            return tf.reduce_sum(inputs * attention_weights, axis=1)
            
        def get_config(self):
            config = super().get_config()
            return config
    
    attention_pool = AttentionPooling(name='attention_pooling')(x)
    
    combined_features = concatenate([avg_pool, max_pool, attention_pool], name='combined_features')
    combined_features = Dropout(0.2)(combined_features)

    dense_out = Dense(D_MODEL, activation='relu', name='pre_output_dense')(combined_features)
    dense_out = Dropout(0.2)(dense_out)
    
    outputs = Dense(1, activation='sigmoid', name='output')(dense_out)

    # Create model
    model = Model(inputs=[input_layer, other_input_layer], outputs=outputs)
    
    return model

class StatePreservingCallback(Callback):
    """Custom callback to save training state after each epoch"""
    def __init__(self, checkpoint, state_path):
        super().__init__()
        self.checkpoint = checkpoint
        self.state_path = state_path
        
    def on_epoch_end(self, epoch, logs=None):
        # Save training state after each epoch
        self.checkpoint.save(self.state_path)
        print(f"Training state saved at epoch {epoch+1}")

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Stateful LR schedule with proper restoration support"""
    def __init__(self, initial_lr, warmup_steps, decay_steps, offset_steps=0):
        super().__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.offset_steps = tf.cast(offset_steps, tf.float32)
    
    def __call__(self, step):
        # Add the offset to account for previously completed steps
        adjusted_step = tf.cast(step, tf.float32) + self.offset_steps
        
        # Warmup phase calculation
        warmup_lr = self.initial_lr * (adjusted_step / self.warmup_steps)
        
        # Cosine decay phase calculation
        angle = tf.constant(np.pi, dtype=tf.float32) * (adjusted_step - self.warmup_steps) / self.decay_steps
        decayed_lr = self.initial_lr * 0.5 * (1 + tf.cos(angle))

        return tf.cond(
            adjusted_step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decayed_lr
        )
    
    def get_config(self):
        return {
            "initial_lr": float(self.initial_lr.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "decay_steps": float(self.decay_steps.numpy()),
            "offset_steps": float(self.offset_steps.numpy())
        }
    
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=10, # Stops if there's no improvement for 3 epochs
                               mode='min', 
                               verbose=1)

model_checkpoint = ModelCheckpoint('best_multiresolution_model.keras', 
                                   monitor='val_loss', 
                                   save_best_only=True, 
                                   mode='min', 
                                   verbose=1)

def time_aware_training():
    # 1. Initialize time tracking
    current_time = datetime.datetime.now()
    daily_stop = current_time.replace(hour=21, minute=0, second=0)
    
    # 2. Initialize training state variables
    current_epoch = tf.Variable(0, dtype=tf.int64)
    best_val_loss = tf.Variable(float('inf'), dtype=tf.float32)
    wait_counter = tf.Variable(0, dtype=tf.int64)
    
    # 3. Create initial LR schedule (will be recreated after restoration)
    lr_schedule = WarmupCosineDecay(
        initial_lr=2e-5,
        warmup_steps=600000,
        decay_steps=6000000,
        offset_steps=0  # Will be updated after restoration
    )
    
    # 4. Create model and optimizer
    model = create_model(training=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    # 5. Create checkpoint BEFORE compiling
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        current_epoch=current_epoch,
        best_val_loss=best_val_loss,
        wait_counter=wait_counter
    )
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)
    
    # 6. Restore previous state if exists
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Resuming training from epoch {current_epoch.numpy()}")
        print(f"Restored optimizer iterations: {optimizer.iterations.numpy()}")
        
        # CRITICAL: Recreate LR schedule with proper offset
        restored_iterations = optimizer.iterations.numpy()
        lr_schedule = WarmupCosineDecay(
            initial_lr=2e-5,
            warmup_steps=600000,
            decay_steps=6000000,
            offset_steps=restored_iterations
        )
        
        # Recreate optimizer with corrected LR schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        
        # Update the checkpoint reference
        checkpoint.optimizer = optimizer
        
        print(f"Learning rate schedule restored with offset: {restored_iterations}")
        print(f"Best validation loss: {best_val_loss.numpy():.4f}")
    else:
        print("Starting new training")
    
    # 7. COMPILE THE MODEL after restoration
    model.compile(
        optimizer=optimizer, 
        loss=recommended_trading_loss, 
        metrics=['accuracy', auc, prec]
    )
    
    # 8. Create custom callback for state preservation
    state_callback = StatePreservingCallback(
        checkpoint=checkpoint,
        state_path=manager.latest_checkpoint or TRAINING_STATE_PATH
    )
    
    # 9. Training loop with time constraints
    while current_epoch.numpy() < 50:
        epoch_start = datetime.datetime.now()
        
        # Check if we have enough time for a full epoch
        time_left = (daily_stop - epoch_start).total_seconds() / 3600
        if time_left < 6.5:
            print(f"Stopping: Only {time_left:.1f}h left, insufficient for 6h epoch")
            break

        # Train single epoch
        print(f"\nStarting epoch {current_epoch.numpy() + 1}")
        current_lr = lr_schedule(optimizer.iterations)
        print(f"Current LR: {current_lr.numpy():.8f}")
        
        history = model.fit(
            train_dataset,
            epochs=current_epoch.numpy() + 1,
            initial_epoch=current_epoch.numpy(),
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[state_callback, early_stopping, model_checkpoint]
        )
        
        # 10. Update training state
        current_val_loss = history.history['val_loss'][-1]
        
        # Early stopping logic
        if current_val_loss < best_val_loss.numpy():
            best_val_loss.assign(current_val_loss)
            wait_counter.assign(0)
            model.save(BEST_MODEL_PATH)
            print(f"New best model saved (val_loss: {current_val_loss:.4f})")
        else:
            wait_counter.assign_add(1)
            print(f"No improvement: {wait_counter.numpy()}/10 patience")
        
        # Update epoch counter
        current_epoch.assign_add(1)
        
        # Save state after each epoch
        manager.save()
        
        # Check early stopping
        if wait_counter.numpy() >= 10:
            print("Early stopping triggered")
            break

    print("Training session completed")

if __name__ == "__main__":
    time_aware_training()