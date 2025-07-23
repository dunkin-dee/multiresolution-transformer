import os
import json
from transformer_builder import (
    LearnablePositionalEncoding, 
    StochasticGatedTransformerBlock, 
    AddTypeEmbedding, 
    AttentionPooling
)
import tensorflow as tf
from losses import recommended_trading_loss

# Load instruments
instruments = os.listdir('models')

try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set: {policy.name}")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

def extract_weights_for_all_instruments():
    custom_objects = {
        'LearnablePositionalEncoding': LearnablePositionalEncoding,
        'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
        'AddTypeEmbedding': AddTypeEmbedding,
        'AttentionPooling': AttentionPooling,
        'recommended_trading_loss': recommended_trading_loss
    }
    
    for instrument in instruments:
        model_path = f"models/{instrument}/middle/up/best_model.keras"
        weights_path = f"models/{instrument}/middle/up/best_model.weights.h5"
        
        if os.path.exists(model_path):
            print(f"Processing {instrument}...")
            
            # Load model
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            # Save weights
            model.save_weights(weights_path)
            
            print(f"✓ Weights extracted for {instrument}")
        else:
            print(f"✗ Model not found for {instrument}")

# Run the extraction
extract_weights_for_all_instruments()