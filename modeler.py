from constants.global_constants import *
from transformer_builder import LearnablePositionalEncoding, StochasticGatedTransformerBlock, AddTypeEmbedding, AttentionPooling
from tensorflow.keras.layers import Input, Conv1D,  Dense, concatenate, Dropout, Layer, GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization, Add
from tensorflow.keras.models import Model


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