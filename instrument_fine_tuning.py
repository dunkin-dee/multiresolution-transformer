# =============================================================================
# INSTRUMENT CONFIGURATION - CHANGE THIS TO TRAIN ON DIFFERENT INSTRUMENTS
# =============================================================================
TARGET_INSTRUMENT = 'GBPUSD#'  # Change this to: 'EURUSD#', 'GBPUSD#', 'USDJPY#', 'AUDUSD#'
# =============================================================================

import tensorflow as tf
import os
import json
from datetime import datetime
from trainer import (
    TrainingStateManager,
    TimeBasedTrainingManager,
    ResumableTrainingCallback,
    restore_optimizer_state,
    create_fresh_training_setup
)
from datasets import get_datasets_and_steps
from transformer_builder import WarmupCosineDecay
from modeler import create_model
from losses import recommended_trading_loss
from constants.global_constants import LR, WARMUP, DECAY


class InstrumentFineTuningPipeline:
    """Fine-tuning pipeline for specific instrument"""
    
    def __init__(self, target_instrument, recent_model_path='checkpoints_recent/best_model.keras'):
        self.target_instrument = target_instrument
        self.recent_model_path = recent_model_path
        self.recent_data_path = 'data/split_finer_data'
        
        # Initialize managers
        self.instrument_manager = TrainingStateManager(f'checkpoints_{target_instrument}')
        self.time_manager = TimeBasedTrainingManager(
            start_time_hour=7,
            end_time_hour=23,
            end_time_minute=0,
            buffer_minutes=5
        )
    
    def load_recent_model(self):
        """Load the best trained model from recent data training"""
        print(f"Loading recent model from: {self.recent_model_path}")
        
        if not os.path.exists(self.recent_model_path):
            raise FileNotFoundError(f"Recent model not found at {self.recent_model_path}")
        
        # Import all custom components for loading
        from transformer_builder import (
            LearnablePositionalEncoding, 
            StochasticGatedTransformerBlock, 
            AddTypeEmbedding, 
            AttentionPooling,
            WarmupCosineDecay
        )
        
        # Import both loss functions for loading
        try:
            from losses import precision_focused_balanced_loss
        except ImportError:
            print("Warning: Could not import precision_focused_balanced_loss")
            precision_focused_balanced_loss = None
        
        custom_objects = {
            'recommended_trading_loss': recommended_trading_loss,
            'LearnablePositionalEncoding': LearnablePositionalEncoding,
            'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
            'AddTypeEmbedding': AddTypeEmbedding,
            'AttentionPooling': AttentionPooling,
            'WarmupCosineDecay': WarmupCosineDecay
        }
        
        # Add the original loss function if available
        if precision_focused_balanced_loss is not None:
            custom_objects['precision_focused_balanced_loss'] = precision_focused_balanced_loss
        
        model = tf.keras.models.load_model(self.recent_model_path, custom_objects=custom_objects)
        print("Recent model loaded successfully")
        return model
    
    def setup_model_for_fine_tuning(self, base_model, learning_rate_factor=0.05):
        """Setup model for instrument-specific fine-tuning with very low learning rate"""
        # Create new optimizer with very low learning rate for fine-tuning
        reduced_lr = LR * learning_rate_factor
        lr_schedule = WarmupCosineDecay(
            initial_lr=reduced_lr, 
            warmup_steps=WARMUP // 40,  # Very short warmup for fine-tuning
            decay_steps=DECAY // 40     # Very short decay for fine-tuning
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Create metrics
        auc = tf.keras.metrics.AUC()
        prec = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()
        
        # Compile model
        base_model.compile(
            optimizer=optimizer,
            loss=recommended_trading_loss,
            metrics=['accuracy', auc, prec, recall]
        )
        
        return base_model, optimizer
    
    def fine_tune_instrument_specific(self):
        """Fine-tune for specific instrument"""
        print(f"\n" + "="*80)
        print(f"FINE-TUNING FOR {self.target_instrument}")
        print("="*80)
        
        # Check if we can start training
        if not self.time_manager.start_training_session():
            print("Cannot start training - outside of allowed training window")
            return None
        
        # Check if we should resume or start fresh for this instrument
        if self.instrument_manager.has_checkpoint():
            print(f"Found existing {self.target_instrument} checkpoint - resuming...")
            try:
                model, optimizer_state, current_epoch, global_step, best_val_loss, patience_count, history, epoch_durations = self.instrument_manager.load_training_state()
                
                # Load instrument-specific dataset
                (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps([self.target_instrument], working_path=self.recent_data_path)
                
                # Setup optimizer
                _, optimizer = self.setup_model_for_fine_tuning(model)
                restore_optimizer_state(optimizer, optimizer_state, model, global_step)
                
                # Load epoch durations
                self.time_manager.load_epoch_durations(epoch_durations)
                
                initial_epoch = current_epoch
                print(f"Resumed {self.target_instrument} training from epoch {current_epoch}")
                
            except Exception as e:
                print(f"Failed to resume {self.target_instrument} training: {e}")
                print(f"Starting fresh {self.target_instrument} training...")
                recent_model = self.load_recent_model()
                model = tf.keras.models.clone_model(recent_model)
                model.set_weights(recent_model.get_weights())
                model, optimizer = self.setup_model_for_fine_tuning(model)
                
                # Load instrument-specific dataset
                (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps([self.target_instrument], working_path=self.recent_data_path)
                
                initial_epoch = 0
                best_val_loss = float('inf')
                patience_count = 0
                global_step = 0
        else:
            print(f"Starting fresh {self.target_instrument} fine-tuning...")
            recent_model = self.load_recent_model()
            model = tf.keras.models.clone_model(recent_model)
            model.set_weights(recent_model.get_weights())
            model, optimizer = self.setup_model_for_fine_tuning(model)
            
            # Load instrument-specific dataset
            print(f"Loading {self.target_instrument} data from {self.recent_data_path}")
            (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps([self.target_instrument], working_path=self.recent_data_path)
            
            initial_epoch = 0
            best_val_loss = float('inf')
            patience_count = 0
            global_step = 0
        
        # Check if we can start an epoch
        if not self.time_manager.can_start_epoch():
            print(f"Cannot start {self.target_instrument} training - insufficient time remaining")
            return model
        
        # Print training status
        print(f"\n{self.target_instrument} FINE-TUNING STATUS:")
        print(f"Starting from epoch: {initial_epoch}")
        print(f"Global step: {global_step}")
        print(f"Current best validation loss: {best_val_loss:.6f}")
        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps per epoch: {val_steps}")
        
        # Create callback
        callback = ResumableTrainingCallback(
            self.instrument_manager, 
            self.time_manager, 
            save_freq=1, 
            initial_epoch=initial_epoch
        )
        callback.set_initial_state(best_val_loss, patience_count, global_step)
        
        # Train the model
        total_epochs = 50  # Fewer epochs for instrument-specific fine-tuning
        remaining_epochs = total_epochs - initial_epoch
        
        if remaining_epochs > 0:
            print(f"\nStarting {self.target_instrument} fine-tuning for up to {remaining_epochs} more epochs...")
            
            history = model.fit(
                train_dataset,
                epochs=total_epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=train_steps,
                validation_data=val_dataset,
                validation_steps=val_steps,
                callbacks=[callback],
                verbose=1
            )
            
            print(f"{self.target_instrument} fine-tuning completed!")
            
            # Final save
            final_history = {}
            if hasattr(history, 'history'):
                final_history = {key: [float(v) for v in values] for key, values in history.history.items()}
            
            epochs_completed = len(history.history.get('loss', [])) if hasattr(history, 'history') else 0
            final_epoch = initial_epoch + epochs_completed
            
            callback.state_manager.save_training_state(
                model=model,
                optimizer=model.optimizer,
                epoch=final_epoch,
                best_val_loss=callback.best_val_loss,
                early_stopping_patience_count=callback.early_stopping_patience_count,
                global_step=callback.global_step,
                history=final_history,
                epoch_durations=self.time_manager.epoch_durations
            )
        
        # Load and return the best instrument-specific model
        best_instrument_path = os.path.join(self.instrument_manager.checkpoint_dir, 'best_model.keras')
        if os.path.exists(best_instrument_path):
            print(f"Loading best {self.target_instrument} model from: {best_instrument_path}")
            from transformer_builder import (
                LearnablePositionalEncoding, 
                StochasticGatedTransformerBlock, 
                AddTypeEmbedding, 
                AttentionPooling,
                WarmupCosineDecay
            )
            
            # Import both loss functions for loading
            try:
                from losses import precision_focused_balanced_loss
            except ImportError:
                print("Warning: Could not import precision_focused_balanced_loss")
                precision_focused_balanced_loss = None
            
            custom_objects = {
                'recommended_trading_loss': recommended_trading_loss,
                'LearnablePositionalEncoding': LearnablePositionalEncoding,
                'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
                'AddTypeEmbedding': AddTypeEmbedding,
                'AttentionPooling': AttentionPooling,
                'WarmupCosineDecay': WarmupCosineDecay
            }
            
            # Add the original loss function if available
            if precision_focused_balanced_loss is not None:
                custom_objects['precision_focused_balanced_loss'] = precision_focused_balanced_loss
            
            best_model = tf.keras.models.load_model(best_instrument_path, custom_objects=custom_objects)
            return best_model
        else:
            print(f"No best {self.target_instrument} model found, returning current model")
            return model
    
    def run_pipeline(self):
        """Run the instrument-specific fine-tuning pipeline"""
        print("\n" + "="*80)
        print(f"STARTING {self.target_instrument} FINE-TUNING PIPELINE")
        print("="*80)
        print("Pipeline Overview:")
        print(f"Fine-tuning specialized model for {self.target_instrument}")
        print(f"Using recent model from: {self.recent_model_path}")
        print("="*80)
        
        # Fine-tune for the specific instrument
        instrument_model = self.fine_tune_instrument_specific()
        if instrument_model is None:
            print("Fine-tuning failed - cannot complete pipeline")
            return None
        
        print(f"\n{self.target_instrument} fine-tuning completed successfully!")
        print(f"Best {self.target_instrument} model saved to: {self.instrument_manager.checkpoint_dir}/best_model.keras")
        print(f"\nThis specialized model can now be used for trading {self.target_instrument}.")
        print("="*80)
        
        return instrument_model


def main():
    """Main function to run the instrument-specific fine-tuning pipeline"""
    # Set up mixed precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set: {policy.name}")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
    
    # Display current configuration
    print("\n" + "="*80)
    print("INSTRUMENT FINE-TUNING CONFIGURATION")
    print("="*80)
    print(f"Target Instrument: {TARGET_INSTRUMENT}")
    print(f"To change instrument, edit the TARGET_INSTRUMENT variable at the top of this file")
    print("="*80)
    
    # Create and run the pipeline
    pipeline = InstrumentFineTuningPipeline(TARGET_INSTRUMENT)
    result = pipeline.run_pipeline()
    
    if result:
        print(f"\n{TARGET_INSTRUMENT} fine-tuning pipeline completed successfully!")
        print(f"You can now use the specialized model for trading {TARGET_INSTRUMENT}.")
    else:
        print("\nPipeline could not be completed due to time constraints.")
        print("Please run again during training hours (7 AM - 9 PM).")


if __name__ == "__main__":
    main()