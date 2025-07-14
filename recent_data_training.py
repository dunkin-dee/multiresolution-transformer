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
from losses import precision_focused_balanced_loss
from constants.global_constants import LR, WARMUP, DECAY


class RecentDataTrainingPipeline:
    """Training pipeline for recent data with all instruments"""
    
    def __init__(self, base_model_path='checkpoints/best_model.keras'):
        self.base_model_path = base_model_path
        self.recent_data_path = 'data/split_finer_data'
        
        # Initialize managers
        self.recent_training_manager = TrainingStateManager('checkpoints_recent')
        self.time_manager = TimeBasedTrainingManager(
            start_time_hour=0,
            end_time_hour=23,
            end_time_minute=59,
            buffer_minutes=0.1
        )
    
    def load_base_model(self):
        """Load the best trained model from initial training"""
        print(f"Loading base model from: {self.base_model_path}")
        
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Base model not found at {self.base_model_path}")
        
        # Import all custom components for loading
        from transformer_builder import (
            LearnablePositionalEncoding, 
            StochasticGatedTransformerBlock, 
            AddTypeEmbedding, 
            AttentionPooling,
            WarmupCosineDecay
        )
        
        # Import the original loss function that was used to train the base model
        try:
            from losses import recommended_trading_loss
        except ImportError:
            print("Warning: Could not import recommended_trading_loss")
            recommended_trading_loss = None
        
        custom_objects = {
            'precision_focused_balanced_loss': precision_focused_balanced_loss,
            'LearnablePositionalEncoding': LearnablePositionalEncoding,
            'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
            'AddTypeEmbedding': AddTypeEmbedding,
            'AttentionPooling': AttentionPooling,
            'WarmupCosineDecay': WarmupCosineDecay
        }
        
        # Add the original loss function if available
        if recommended_trading_loss is not None:
            custom_objects['recommended_trading_loss'] = recommended_trading_loss
        
        model = tf.keras.models.load_model(self.base_model_path, custom_objects=custom_objects)
        print("Base model loaded successfully")
        return model
    
    def setup_model_for_training(self, base_model, learning_rate_factor=0.05):
        """Setup model for continued training with reduced learning rate"""
        # Create new optimizer with reduced learning rate for fine-tuning
        reduced_lr = LR * learning_rate_factor
        lr_schedule = WarmupCosineDecay(
            initial_lr=reduced_lr, 
            warmup_steps=WARMUP // 5,  # Shorter warmup for fine-tuning
            decay_steps=DECAY // 5     # Shorter decay for fine-tuning
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
            loss=precision_focused_balanced_loss,
            metrics=['accuracy', auc, prec, recall]
        )
        
        return base_model, optimizer
    
    def train_on_recent_data(self):
        """Train on recent data (data/split_finer_data) with all instruments"""
        print("\n" + "="*80)
        print("TRAINING ON RECENT DATA (split_finer_data)")
        print("="*80)
        
        # Check if we can start training
        if not self.time_manager.start_training_session():
            print("Cannot start training - outside of allowed training window")
            return None
        
        # Check if we should resume recent training or start fresh
        if self.recent_training_manager.has_checkpoint():
            print("Found existing recent training checkpoint - resuming...")
            try:
                model, optimizer_state, current_epoch, global_step, best_val_loss, patience_count, history, epoch_durations = self.recent_training_manager.load_training_state()
                
                # Load datasets from recent data
                instruments = [inst for inst in os.listdir(self.recent_data_path) if inst.endswith('#')]
                (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments, working_path=self.recent_data_path)
                
                # Setup optimizer
                _, optimizer = self.setup_model_for_training(model)
                restore_optimizer_state(optimizer, optimizer_state, model, global_step)
                
                # Load epoch durations
                self.time_manager.load_epoch_durations(epoch_durations)
                
                initial_epoch = current_epoch
                print(f"Resumed recent training from epoch {current_epoch}")
                
            except Exception as e:
                print(f"Failed to resume recent training: {e}")
                print("Starting fresh recent training...")
                model = self.load_base_model()
                model, optimizer = self.setup_model_for_training(model)
                
                # Load datasets from recent data
                instruments = [inst for inst in os.listdir(self.recent_data_path) if inst.endswith('#')]
                (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments, working_path=self.recent_data_path)
                
                initial_epoch = 0
                best_val_loss = float('inf')
                patience_count = 0
                global_step = 0
        else:
            print("Starting fresh training on recent data...")
            model = self.load_base_model()
            model, optimizer = self.setup_model_for_training(model)
            
            # Load datasets from recent data
            instruments = [inst for inst in os.listdir(self.recent_data_path) if inst.endswith('#')]
            print(f"Training on recent data with instruments: {instruments}")
            (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments, working_path=self.recent_data_path)
            
            initial_epoch = 0
            best_val_loss = float('inf')
            patience_count = 0
            global_step = 0
        
        # Check if we can start an epoch
        if not self.time_manager.can_start_epoch():
            print("Cannot start any epochs today - insufficient time remaining")
            return model
        
        # Print training status
        print(f"\nRECENT DATA TRAINING STATUS:")
        print(f"Starting from epoch: {initial_epoch}")
        print(f"Global step: {global_step}")
        print(f"Current best validation loss: {best_val_loss:.6f}")
        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps per epoch: {val_steps}")
        
        # Create callback
        callback = ResumableTrainingCallback(
            self.recent_training_manager, 
            self.time_manager, 
            save_freq=1, 
            initial_epoch=initial_epoch
        )
        callback.set_initial_state(best_val_loss, patience_count, global_step)
        
        # Train the model
        total_epochs = 25  # Fewer epochs for fine-tuning
        remaining_epochs = total_epochs - initial_epoch
        
        if remaining_epochs > 0:
            print(f"\nStarting recent data training for up to {remaining_epochs} more epochs...")
            
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
            
            print("Recent data training completed!")
            
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
        
        # Load and return the best recent model
        best_recent_path = os.path.join(self.recent_training_manager.checkpoint_dir, 'best_model.keras')
        if os.path.exists(best_recent_path):
            print(f"Loading best recent model from: {best_recent_path}")
            from transformer_builder import (
                LearnablePositionalEncoding, 
                StochasticGatedTransformerBlock, 
                AddTypeEmbedding, 
                AttentionPooling,
                WarmupCosineDecay
            )
            
            # Import both loss functions for loading
            try:
                from losses import recommended_trading_loss
            except ImportError:
                print("Warning: Could not import recommended_trading_loss")
                recommended_trading_loss = None
            
            custom_objects = {
                'precision_focused_balanced_loss': precision_focused_balanced_loss,
                'LearnablePositionalEncoding': LearnablePositionalEncoding,
                'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
                'AddTypeEmbedding': AddTypeEmbedding,
                'AttentionPooling': AttentionPooling,
                'WarmupCosineDecay': WarmupCosineDecay
            }
            
            # Add the original loss function if available
            if recommended_trading_loss is not None:
                custom_objects['recommended_trading_loss'] = recommended_trading_loss
            
            best_model = tf.keras.models.load_model(best_recent_path, custom_objects=custom_objects)
            return best_model
        else:
            print("No best recent model found, returning current model")
            return model
    
    def run_pipeline(self):
        """Run the recent data training pipeline"""
        print("\n" + "="*80)
        print("STARTING RECENT DATA TRAINING PIPELINE")
        print("="*80)
        print("Pipeline Overview:")
        print("Training on recent data (split_finer_data) with all instruments")
        print("="*80)
        
        # Train on recent data
        recent_best_model = self.train_on_recent_data()
        if recent_best_model is None:
            print("Training failed - cannot complete pipeline")
            return None
        
        print(f"\nRecent data training completed successfully!")
        print(f"Best recent model saved to: {self.recent_training_manager.checkpoint_dir}/best_model.keras")
        print("\nThis model can now be used for instrument-specific fine-tuning.")
        print("="*80)
        
        return recent_best_model


def main():
    """Main function to run the recent data training pipeline"""
    # Set up mixed precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set: {policy.name}")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
    
    # Create and run the pipeline
    pipeline = RecentDataTrainingPipeline()
    result = pipeline.run_pipeline()
    
    if result:
        print("\nRecent data training pipeline completed successfully!")
        print("You can now run the instrument-specific fine-tuning script.")
    else:
        print("\nPipeline could not be completed due to time constraints.")
        print("Please run again during training hours (7 AM - 9 PM).")


if __name__ == "__main__":
    main()