from constants.global_constants import *
import tensorflow as tf
from transformer_builder import LearnablePositionalEncoding, StochasticGatedTransformerBlock, AddTypeEmbedding, AttentionPooling
from tensorflow.keras.layers import Input, Conv1D,  Dense, concatenate, Dropout, Layer, GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization, Add, Lambda
from tensorflow.keras.models import Model
from keras.ops import sin, cos, concatenate as keras_concat, expand_dims
from tensorflow.keras.regularizers import l2



class ScalarScale(tf.keras.layers.Layer):
    """Trainable scalar scaling layer for branch weighting"""
    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True
        )
        
    def call(self, inputs):
        return inputs * self.scale
    
class TemporalPreservingDropout(Layer):
    def __init__(self, rate, preserve_last=1, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.keep = 1.0 - rate
        self.preserve_last = preserve_last

    def call(self, inputs, training=None):
        if not training or self.rate == 0.0:
            return inputs

        # inputs shape: (batch, time, features)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        
        # Create a mask instead of slicing
        # Mask will be 1.0 for tokens we want to preserve, random for others
        preserve_positions = tf.maximum(0, seq_len - self.preserve_last)
        
        # Create position indices
        positions = tf.range(seq_len, dtype=tf.int32)
        
        # Create binary mask for positions to preserve (last preserve_last tokens)
        preserve_mask = tf.cast(positions >= preserve_positions, tf.float32)
        
        # Create dropout mask for non-preserved positions
        dropout_shape = [batch_size, seq_len, feature_dim]
        random_tensor = tf.random.uniform(dropout_shape, dtype=inputs.dtype)
        dropout_mask = tf.cast(random_tensor < self.keep, inputs.dtype) / tf.cast(self.keep, inputs.dtype)
        
        # Combine masks: use 1.0 for preserved positions, dropout_mask for others
        final_mask = preserve_mask[None, :, None] + (1.0 - preserve_mask[None, :, None]) * dropout_mask
        
        return inputs * final_mask

    def get_config(self):
        return {"rate": self.rate, "preserve_last": self.preserve_last}


def create_regression_model(input_shape=(64, 5), other_input_shape=(64, 5), 
                           partial_hour_shape=(1, 5),  # Single aggregated hour token
                           training=True, stochastic_rates=[0.05, 0.1, 0.15, 0.2],
                           d_model=D_MODEL, num_heads=NUM_HEADS, ff_dim=FF_DIM,
                           num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS,
                           feature_cols=[
                               'open_normalized',
                               'high_normalized', 
                               'low_normalized',
                               'close_normalized']):
    
    # Main (5-minute) data input - normalized on 5-min lookback
    input_shape = (NUM_TOKENS, len(feature_cols))
    input_layer = Input(shape=input_shape, name='minute_input')
    
    # Complete hourly data input - normalized on hourly lookback
    other_input_shape = (OTHER_TOKENS, len(feature_cols))
    other_input_layer = Input(shape=other_input_shape, name='hourly_input')
    
    # Partial hour data - single aggregated token normalized using SAME scheme as hourly data
    # This provides hourly-compatible context for the current incomplete hour
    partial_hour_shape = (1, len(feature_cols))
    partial_hour_layer = Input(shape=partial_hour_shape, name='partial_hour_input')
    
    # Temporal context inputs for the partial hour data
    minutes_into_hour = Input(shape=(1,), name='minutes_into_hour')  # 0-55
    partial_hour_length = Input(shape=(1,), name='partial_hour_length')  # 0-12
    
    # CNN feature extractor (reduced embedding space for temporal info)
    def cnn_feature_extractor(inputs, name_prefix, max_filter_output=d_model-24):
        x = Conv1D(filters=max_filter_output//2, kernel_size=3, activation='relu', 
                padding="same", name=f'{name_prefix}_conv1')(inputs)
        x = TemporalPreservingDropout(0.05, preserve_last=8)(x)
        x = Conv1D(filters=max_filter_output, kernel_size=5, activation='relu', 
                padding="same", name=f'{name_prefix}_conv3')(x)
        x = TemporalPreservingDropout(0.05, preserve_last=8)(x)
        return x
        
    # Enhanced temporal context embedding for partial hour metadata
    def create_partial_hour_context(minutes_val, length_val, embed_dim=8):
        # Cyclical encoding for temporal position within hour
        minutes_norm = Lambda(lambda x: x / 55.0)(minutes_val)
        minutes_sin = Lambda(lambda x: tf.sin(2 * 3.14159 * x))(minutes_norm)
        minutes_cos = Lambda(lambda x: tf.cos(2 * 3.14159 * x))(minutes_norm)
        
        # Linear encoding for data availability  
        length_norm = Lambda(lambda x: x / 12.0)(length_val)
        
        # Combine all temporal features using Keras concatenate layer
        temporal_features = concatenate([
            minutes_sin, minutes_cos, minutes_norm, length_norm
        ], name='temporal_features')
        
        # Project to embedding dimension
        temporal_embed = Dense(embed_dim, activation='relu', name='temporal_embed')(temporal_features)
        return temporal_embed
    
    # Create temporal context for partial hour
    partial_temporal_context = create_partial_hour_context(
        minutes_into_hour, partial_hour_length, embed_dim=8
    )
    
    # Process 5-minute data branch (high-resolution, 5-min normalized)
    x = cnn_feature_extractor(input_layer, '5min', d_model-16)
    x = Dense(d_model - 16, name='5min_projection')(x)
    x = LearnablePositionalEncoding(max_seq_len=num_tokens, embed_dim=d_model-16)(x)
    x = AddTypeEmbedding(type_id=0, embed_dim=16, name='5min_type_embed')(x)

    # Process complete hourly data branch (hourly normalized)
    h = cnn_feature_extractor(other_input_layer, 'hourly', d_model-16)
    h = Dense(d_model - 16, name='hourly_projection')(h)
    h = LearnablePositionalEncoding(max_seq_len=other_tokens, embed_dim=d_model-16)(h)
    h = AddTypeEmbedding(type_id=1, embed_dim=16, name='hourly_type_embed')(h)
    # Process partial hour data branch (single token, hourly normalized, with temporal context)
    # Since it's a single token, we don't need CNN feature extraction
    p = Dense(d_model - 24, name='partial_projection')(partial_hour_layer)
    # No positional encoding needed for single token
    p = AddTypeEmbedding(type_id=2, embed_dim=16, name='partial_type_embed')(p)
    
    # Add temporal context to the single partial hour token
    # Expand temporal context to match sequence dimension (1) using Lambda layer
    partial_temporal_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1), 
                                     name='expand_temporal')(partial_temporal_context)  # (batch, 1, 8)
    p = concatenate([p, partial_temporal_expanded], axis=-1, name='partial_with_context')

    # ====== BRANCH SCALING WEIGHTS - NEW ADDITION ======
    # Create learnable scaling parameters for each branch using Dense layers
    # This ensures they're properly tracked as trainable variables
    
    # Create scaling layers that will learn a single scaling factor per branch
    hourly_scaled = ScalarScale(initial_value=1.0, name='hourly_scaler')(h)
    partial_scaled = ScalarScale(initial_value=1.0, name='partial_scaler')(p)
    minute_scaled = ScalarScale(initial_value=1.0, name='minute_scaler')(x)
    # Strategic concatenation with scaled branches:
    # 1. Hourly data first (longest-term context, now scaled)
    # 2. Partial hour data second (current hour context, hourly-normalized, now scaled)
    # 3. 5-minute data last (immediate high-resolution context, now scaled)
    x = concatenate([hourly_scaled, partial_scaled, minute_scaled], axis=1, name='multi_resolution_concat')
    # x = concatenate([h, p, x], axis=1, name='multi_resolution_concat')

    # Multi-layer transformer with progressive stochastic depth
    for i in range(len(stochastic_rates)):
        x_norm = LayerNormalization(name=f'ln_pre_{i}')(x)
        
        x_block_out = StochasticGatedTransformerBlock(
            d_model,  # Should match the final concatenated dimension
            num_heads,
            ff_dim,
            rate=0.1,
            stochastic_depth_rate=stochastic_rates[i]
        )(x_norm, training=training)

        x_block_out = TemporalPreservingDropout(stochastic_rates[i], preserve_last=8)(x_block_out)
        
        x = Add(name=f'residual_add_{i}')([x, x_block_out])

    # Multi-scale feature aggregation
    avg_pool = GlobalAveragePooling1D(name='avg_pool')(x)
    max_pool = GlobalMaxPooling1D(name='max_pool')(x)
    attention_pool = AttentionPooling(name='attention_pooling')(x)
    
    combined_features = concatenate([avg_pool, max_pool, attention_pool], name='combined_features')

    # Shared representation learning
    shared_dense = Dense(d_model, activation='relu', name='shared_dense')(combined_features)
    
    # Task-specific prediction heads
    high_dense = Dense(d_model // 2, activation='relu', name='high_dense')(shared_dense)
    target_high = Dense(1, activation='linear', name='target_high')(high_dense)
    
    
    model = Model(
        inputs=[
            input_layer,           # 5-minute data (5-min normalized)
            other_input_layer,     # Complete hourly data (hourly normalized)
            partial_hour_layer,    # Partial hour data (hourly normalized)
            minutes_into_hour,     # Temporal context: minutes into current hour
            partial_hour_length    # Temporal context: valid data points in partial hour
        ], 
        outputs={'target_high': target_high}
    )
    
    return model