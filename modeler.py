from constants.global_constants import *
import tensorflow as tf
from transformer_builder import LearnablePositionalEncoding, StochasticGatedTransformerBlock, AddTypeEmbedding, AttentionPooling
from tensorflow.keras.layers import Input, Conv1D,  Dense, concatenate, Dropout, Layer, GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization, Add, Lambda
from tensorflow.keras.models import Model
from keras.ops import sin, cos, concatenate as keras_concat, expand_dims
from tensorflow.keras.regularizers import l2



def create_model(input_shape=(64, 4), other_input_shape=(64, 4), 
                 training=True, stochastic_rates = [0.05, 0.1, 0.15, 0.2],
                 d_model=D_MODEL, num_heads=NUM_HEADS, ff_dim=FF_DIM,
                 num_tokens=NUM_TOKENS, other_tokens=OTHER_TOKENS,
                 feature_cols=[
                     'open_normalized',
                     'high_normalized',
                     'low_normalized',
                     'close_normalized']):
    # Main (5-minute) data input
    input_shape = (NUM_TOKENS, len(feature_cols))
    input_layer = Input(shape=input_shape, name='minute_input')
    
    # Hourly data input
    other_input_layer = Input(shape=other_input_shape, name='hourly_input')
    other_input_shape = (OTHER_TOKENS, len(feature_cols))
    
    # Improved CNN block function
    def cnn_feature_extractor(inputs, name_prefix, max_filter_output=d_model-16):
        # Multi-scale feature extraction
        x = Conv1D(filters=max_filter_output//2, kernel_size=3, activation='relu', padding="same", name=f'{name_prefix}_conv1')(inputs)      
        x = Conv1D(filters=max_filter_output, kernel_size=5, activation='relu', padding="same", name=f'{name_prefix}_conv3')(x)
   
        return x
    
    # Process 5-minute data branch
    x = cnn_feature_extractor(input_layer, '5min', d_model-16)
    x = Dense(d_model - 16, name='5min_projection')(x)  # Reserve 16 dims for type embedding
    x = LearnablePositionalEncoding(max_seq_len=num_tokens, embed_dim=d_model - 16)(x)
    
    # Add type embedding for 5-minute data (type 0)
    x = AddTypeEmbedding(type_id=0, embed_dim=16, name='5min_type_embed')(x)

    # Process hourly data branch
    h = cnn_feature_extractor(other_input_layer, 'hourly', d_model-16)
    h = Dense(d_model - 16, name='hourly_projection')(h)  # Reserve 16 dims for type embedding
    h = LearnablePositionalEncoding(max_seq_len=other_tokens, embed_dim=d_model-16)(h)
    
    # Add type embedding for hourly data (type 1)
    h = AddTypeEmbedding(type_id=1, embed_dim=16, name='hourly_type_embed')(h)

    x = concatenate([h, x], axis=1, name='timeframe_concat')
    x = Dropout(0.1)(x)

    # Deeper transformer stack with increasing stochastic depth
    for i in range(len(stochastic_rates)):  # or 6 if you expand later
        # Pre-norm the input
        x_norm = LayerNormalization(name=f'ln_pre_{i}')(x)

        # Pass through the transformer block
        x_block_out = StochasticGatedTransformerBlock(
            d_model,
            num_heads,
            ff_dim,
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
    attention_pool = AttentionPooling(name='attention_pooling')(x)
    
    combined_features = concatenate([avg_pool, max_pool, attention_pool], name='combined_features')
    combined_features = Dropout(0.2)(combined_features)

    dense_out = Dense(d_model, activation='relu', name='pre_output_dense')(combined_features)
    dense_out = Dropout(0.2)(dense_out)
    
    outputs = Dense(1, activation='sigmoid', name='output')(dense_out)

    # Create model
    model = Model(inputs=[input_layer, other_input_layer], outputs=outputs)
    
    return model


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
        x = Dropout(0.15)(x)  # Add dropout after first conv
        x = Conv1D(filters=max_filter_output, kernel_size=5, activation='relu', 
                padding="same", name=f'{name_prefix}_conv3')(x)
        x = Dropout(0.15)(x)  # Add dropout after second conv
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
    x = Dense(d_model - 16, kernel_regularizer=l2(5e-4), name='5min_projection')(x)
    x = LearnablePositionalEncoding(max_seq_len=num_tokens, embed_dim=d_model-16)(x)
    x = AddTypeEmbedding(type_id=0, embed_dim=16, name='5min_type_embed')(x)

    # Process complete hourly data branch (hourly normalized)
    h = cnn_feature_extractor(other_input_layer, 'hourly', d_model-16)
    h = Dense(d_model - 16, kernel_regularizer=l2(5e-4), name='hourly_projection')(h)
    h = LearnablePositionalEncoding(max_seq_len=other_tokens, embed_dim=d_model-16)(h)
    h = AddTypeEmbedding(type_id=1, embed_dim=16, name='hourly_type_embed')(h)
    h = Dropout(0.2)(h)
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
    x = Dropout(0.1)(x)

    # Multi-layer transformer with progressive stochastic depth
    for i in range(len(stochastic_rates)):
        x_norm = LayerNormalization(name=f'ln_pre_{i}')(x)
        
        x_block_out = StochasticGatedTransformerBlock(
            d_model,  # Should match the final concatenated dimension
            num_heads,
            ff_dim,
            rate=0.1,
            attention_dropout=stochastic_rates[i],
            stochastic_depth_rate=stochastic_rates[i]
        )(x_norm, training=training)
        
        x_block_out = Dropout(0.1, name=f'post_dropout_{i}')(x_block_out)
        x = Add(name=f'residual_add_{i}')([x, x_block_out])

    # Multi-scale feature aggregation
    avg_pool = GlobalAveragePooling1D(name='avg_pool')(x)
    max_pool = GlobalMaxPooling1D(name='max_pool')(x)
    attention_pool = AttentionPooling(name='attention_pooling')(x)
    
    combined_features = concatenate([avg_pool, max_pool, attention_pool], name='combined_features')
    combined_features = Dropout(0.2)(combined_features)

    # Shared representation learning
    shared_dense = Dense(d_model, activation='relu', kernel_regularizer=l2(1e-3), name='shared_dense')(combined_features)
    shared_dense = Dropout(0.2)(shared_dense)
    
    # Task-specific prediction heads
    high_dense = Dense(d_model // 2, activation='relu', name='high_dense')(shared_dense)
    high_dense = Dropout(0.2)(high_dense)
    target_high = Dense(1, activation='linear', kernel_regularizer=l2(1e-3), name='target_high')(high_dense)
    
    
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