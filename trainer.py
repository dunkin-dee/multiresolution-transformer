import tensorflow as tf
import os
import json
import pickle
from datetime import datetime, time
from tensorflow.keras.callbacks import Callback
from losses import precision_focused_balanced_loss
from datasets import get_datasets_and_steps
from transformer_builder import WarmupCosineDecay
from modeler import create_model
from constants.global_constants import LR, WARMUP, DECAY

class TimeBasedTrainingManager:
    """Manages training time windows to ensure stopping before 9 PM"""
    
    def __init__(self, start_time_hour=7, end_time_hour=21, end_time_minute=0, buffer_minutes=30):
        """
        Initialize time manager
        
        Args:
            start_time_hour: Hour to allow training start (24-hour format, default 7 = 7 AM)
            end_time_hour: Hour to stop training (24-hour format, default 21 = 9 PM)
            end_time_minute: Minute to stop training (default 0)
            buffer_minutes: Safety buffer before end time (default 30 minutes)
        """
        self.start_time = time(start_time_hour, 0)
        self.end_time = time(end_time_hour, end_time_minute)
        self.buffer_minutes = buffer_minutes
        self.training_start_time = None
        self.epoch_durations = []  # Track epoch durations for estimation
        
    def start_training_session(self):
        """Mark the start of training session"""
        current_time = datetime.now()
        current_time_only = current_time.time()
        
        # Check if we're in valid training window
        if current_time_only < self.start_time:
            print(f"Too early to start training. Current: {current_time_only}, Earliest start: {self.start_time}")
            return False
            
        if current_time_only >= self.end_time:
            print(f"Too late to start training. Current: {current_time_only}, Latest end: {self.end_time}")
            return False
            
        self.training_start_time = current_time
        print(f"Training session started at: {self.training_start_time.strftime('%H:%M:%S')}")
        return True
        
    def load_epoch_durations(self, epoch_durations):
        """Load previous epoch durations from saved state"""
        self.epoch_durations = epoch_durations
        print(f"Loaded {len(self.epoch_durations)} previous epoch durations for estimation")
        
    def can_start_epoch(self):
        """Check if we can safely start another epoch"""
        if not self.training_start_time:
            return False
            
        current_time = datetime.now()
        current_time_only = current_time.time()
        
        # Calculate time until cutoff (with buffer)
        today = current_time.date()
        cutoff_datetime = datetime.combine(today, self.end_time)
        buffer_cutoff = cutoff_datetime.timestamp() - (self.buffer_minutes * 60)
        buffer_cutoff_datetime = datetime.fromtimestamp(buffer_cutoff)
        
        # If it's already past buffer cutoff, don't start
        if current_time >= buffer_cutoff_datetime:
            print(f"Current time {current_time_only} is past buffer cutoff time (9 PM - {self.buffer_minutes} min buffer)")
            return False
            
        # Estimate time needed for next epoch
        estimated_epoch_duration = self.estimate_epoch_duration()
        
        # Check if we have enough time
        time_remaining = (buffer_cutoff_datetime - current_time).total_seconds()
        
        if time_remaining < estimated_epoch_duration:
            print(f"Insufficient time remaining. Need ~{estimated_epoch_duration/60:.1f} minutes, have {time_remaining/60:.1f} minutes")
            return False
            
        print(f"Safe to start epoch. Estimated duration: {estimated_epoch_duration/60:.1f} minutes, Time remaining: {time_remaining/60:.1f} minutes")
        return True
        
    def record_epoch_duration(self, duration_seconds):
        """Record the duration of completed epoch for future estimation"""
        self.epoch_durations.append(duration_seconds)
        # Keep only last 3 epochs for moving average
        if len(self.epoch_durations) > 3:
            self.epoch_durations.pop(0)
            
    def estimate_epoch_duration(self):
        """Estimate duration of next epoch based on historical data"""
        if not self.epoch_durations:
            # Conservative estimate: 7.5 hours = 27,000 seconds
            return 6
        
        # Use average of recent epochs with 15% safety margin
        avg_duration = sum(self.epoch_durations) / len(self.epoch_durations)
        return avg_duration * 1.15  # 15% safety margin
        
    def get_training_summary(self):
        """Get summary of training time management"""
        if not self.training_start_time:
            return "Training not started"
            
        current_time = datetime.now()
        total_training_time = (current_time - self.training_start_time).total_seconds()
        
        return f"""
Training Time Summary:
- Started: {self.training_start_time.strftime('%H:%M:%S')}
- Current: {current_time.strftime('%H:%M:%S')}
- Total training time: {total_training_time/3600:.1f} hours
- Epochs completed: {len(self.epoch_durations)}
- Average epoch duration: {sum(self.epoch_durations)/len(self.epoch_durations)/3600:.1f} hours (if any)
- Training window: {self.start_time} - {self.end_time}
"""

class TrainingStateManager:
    """Manages training state persistence and resumption"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # File paths for different state components
        self.model_path = os.path.join(checkpoint_dir, 'current_model.keras')
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.keras')
        self.optimizer_path = os.path.join(checkpoint_dir, 'optimizer_state.pkl')
        self.training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
        self.history_path = os.path.join(checkpoint_dir, 'training_history.json')
        
    def save_training_state(self, model, optimizer, epoch, best_val_loss, 
                          early_stopping_patience_count, global_step, history=None,
                          epoch_durations=None):
        """Save complete training state"""
        print(f"Saving training state at epoch {epoch}, global step {global_step}...")
        
        # Save current model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Save optimizer state (configuration and iteration count)
        optimizer_state = {
            'config': optimizer.get_config(),
            'global_step': int(global_step),  # Save the global step for LR schedule
            'learning_rate': float(optimizer.learning_rate.numpy()) if hasattr(optimizer.learning_rate, 'numpy') else float(optimizer.learning_rate),
            'iterations': int(optimizer.iterations.numpy()) if hasattr(optimizer.iterations, 'numpy') else int(optimizer.iterations)
        }
        
        # Try to get variable values for momentum etc. (this is the proper way in newer TF versions)
        try:
            # Get optimizer variables (includes momentum, etc.)
            optimizer_variables = {}
            for var in optimizer.variables:
                if hasattr(var, 'name') and hasattr(var, 'numpy'):
                    optimizer_variables[var.name] = var.numpy().tolist()
            optimizer_state['variables'] = optimizer_variables
            print(f"Saved {len(optimizer_variables)} optimizer variables")
        except Exception as e:
            print(f"Warning: Could not save optimizer variables: {e}")
            optimizer_state['variables'] = {}
        
        with open(self.optimizer_path, 'wb') as f:
            pickle.dump(optimizer_state, f)
        print(f"Optimizer state saved to {self.optimizer_path}")
        
        # Save training metadata including epoch durations
        training_state = {
            'current_epoch': epoch,
            'global_step': int(global_step),  # Critical for LR schedule continuity
            'best_val_loss': float(best_val_loss),
            'early_stopping_patience_count': early_stopping_patience_count,
            'epoch_durations': epoch_durations or [],  # Save epoch durations for time estimation
            'timestamp': datetime.now().isoformat()
        }
        with open(self.training_state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        print(f"Training state saved to {self.training_state_path}")
        
        # Save training history if provided
        if history:
            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Training history saved to {self.history_path}")
    
    def load_training_state(self):
        """Load complete training state"""
        if not os.path.exists(self.training_state_path):
            return None, None, 0, 0, float('inf'), 0, {}, []
        
        print("Loading previous training state...")
        
        # Load training metadata
        with open(self.training_state_path, 'r') as f:
            training_state = json.load(f)
        
        current_epoch = training_state['current_epoch']
        global_step = training_state.get('global_step', 0)
        best_val_loss = training_state['best_val_loss']
        early_stopping_patience_count = training_state['early_stopping_patience_count']
        epoch_durations = training_state.get('epoch_durations', [])
        
        print(f"Resuming from epoch {current_epoch}, global step {global_step}, best_val_loss: {best_val_loss}")
        print(f"Loaded {len(epoch_durations)} previous epoch durations")
        
        # Load model with ALL custom objects
        model = None
        if os.path.exists(self.model_path):
            # Import all custom components
            from transformer_builder import (
                LearnablePositionalEncoding, 
                StochasticGatedTransformerBlock, 
                AddTypeEmbedding, 
                AttentionPooling,
                WarmupCosineDecay  # Add this if it's also custom
            )
            
            custom_objects = {
                'precision_focused_balanced_loss': precision_focused_balanced_loss,
                'LearnablePositionalEncoding': LearnablePositionalEncoding,
                'StochasticGatedTransformerBlock': StochasticGatedTransformerBlock,
                'AddTypeEmbedding': AddTypeEmbedding,
                'AttentionPooling': AttentionPooling,
                'WarmupCosineDecay': WarmupCosineDecay
            }
            
            model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects=custom_objects
            )
            print(f"Model loaded from {self.model_path}")
        
        # Load optimizer state
        optimizer_state = None
        if os.path.exists(self.optimizer_path):
            with open(self.optimizer_path, 'rb') as f:
                optimizer_state = pickle.load(f)
            print(f"Optimizer state loaded from {self.optimizer_path}")
        
        # Load training history
        history = {}
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                history = json.load(f)
            print(f"Training history loaded from {self.history_path}")
        
        return model, optimizer_state, current_epoch, global_step, best_val_loss, early_stopping_patience_count, history, epoch_durations
    
    def save_best_model(self, model):
        """Save the best model separately"""
        model.save(self.best_model_path)
        print(f"Best model saved to {self.best_model_path}")
    
    def has_checkpoint(self):
        """Check if checkpoint exists"""
        return os.path.exists(self.training_state_path)

class ResumableTrainingCallback(Callback):
    """Custom callback to handle state saving, best model tracking, and time management"""
    
    def __init__(self, state_manager, time_manager, save_freq=1, initial_epoch=0):
        super().__init__()
        self.state_manager = state_manager
        self.time_manager = time_manager
        self.save_freq = save_freq  # Save every N epochs
        self.best_val_loss = float('inf')
        self.early_stopping_patience_count = 0
        self.early_stopping_patience = 10
        self.global_step = 0  # Track global steps for LR schedule
        self.epoch_start_time = None
        self.initial_epoch = initial_epoch  # Track the starting epoch for proper numbering
        
    def set_initial_state(self, best_val_loss, patience_count, global_step=0):
        """Set initial state when resuming"""
        self.best_val_loss = best_val_loss
        self.early_stopping_patience_count = patience_count
        self.global_step = global_step
        print(f"Initial state set - Best val loss: {best_val_loss}, Patience: {patience_count}, Global step: {global_step}")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Check if we can safely start this epoch"""
        self.epoch_start_time = datetime.now()
        
        # Adjust epoch number for display (epoch is 0-based within current session)
        actual_epoch = epoch + self.initial_epoch + 1
        
        # Get current learning rate and log it
        current_lr = self.model.optimizer.learning_rate
        if hasattr(current_lr, 'numpy'):
            lr_value = current_lr.numpy()
        elif hasattr(current_lr, '__call__'):
            # For custom LR schedules, call with current step
            lr_value = current_lr(self.global_step).numpy()
        else:
            lr_value = float(current_lr)
        
        print(f"Epoch {actual_epoch} - Learning Rate: {lr_value:.2e} (Global Step: {self.global_step})")
        
        if not self.time_manager.can_start_epoch():
            print(f"Time cutoff reached. Stopping training before epoch {actual_epoch}")
            self.model.stop_training = True
            return
            
        print(f"Starting epoch {actual_epoch} at {self.epoch_start_time.strftime('%H:%M:%S')}")
    
    def on_batch_end(self, batch, logs=None):
        """Update global step counter after each batch"""
        self.global_step += 1
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_loss = logs.get('val_loss', float('inf'))
        
        # Adjust epoch number for display and saving
        actual_epoch = epoch + self.initial_epoch + 1
        
        # Record epoch duration for time estimation
        if self.epoch_start_time:
            epoch_duration = (datetime.now() - self.epoch_start_time).total_seconds()
            self.time_manager.record_epoch_duration(epoch_duration)
            print(f"Epoch {actual_epoch} completed in {epoch_duration/3600:.2f} hours")
        
        # Check if this is the best model
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.early_stopping_patience_count = 0
            # Save best model
            self.state_manager.save_best_model(self.model)
            print(f"New best model! Val loss: {current_val_loss:.6f}")
        else:
            self.early_stopping_patience_count += 1
            print(f"No improvement. Patience count: {self.early_stopping_patience_count}/{self.early_stopping_patience}")
        
        # Save training state periodically
        if (epoch + 1) % self.save_freq == 0:
            # Prepare history for saving
            history_dict = {}
            if hasattr(self.model, 'history') and hasattr(self.model.history, 'history'):
                for key, values in self.model.history.history.items():
                    history_dict[key] = [float(v) for v in values]
            
            self.state_manager.save_training_state(
                model=self.model,
                optimizer=self.model.optimizer,
                epoch=actual_epoch,  # Save actual epoch number
                best_val_loss=self.best_val_loss,
                early_stopping_patience_count=self.early_stopping_patience_count,
                global_step=self.global_step,
                history=history_dict,
                epoch_durations=self.time_manager.epoch_durations  # Save epoch durations
            )
        
        # Check early stopping
        if self.early_stopping_patience_count >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
            self.model.stop_training = True
            
        # Print time summary
        print(self.time_manager.get_training_summary())

def restore_optimizer_state(optimizer, optimizer_state, model, global_step):
    """Restore optimizer state including iteration count"""
    if optimizer_state:
        # Run one step to initialize optimizer variables
        dummy_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        optimizer.apply_gradients(zip(dummy_gradients, model.trainable_variables))
        
        # Restore the iteration counter to maintain LR schedule
        # This is CRITICAL - use the saved global_step, not iterations from optimizer_state
        if hasattr(optimizer, 'iterations'):
            optimizer.iterations.assign(global_step)
            print(f"Optimizer iteration counter set to: {global_step}")
        
        # Try to restore optimizer variables (momentum, etc.)
        if 'variables' in optimizer_state and optimizer_state['variables']:
            try:
                # Map saved variables back to optimizer variables
                for opt_var in optimizer.variables:
                    if hasattr(opt_var, 'name') and opt_var.name in optimizer_state['variables']:
                        saved_value = optimizer_state['variables'][opt_var.name]
                        opt_var.assign(tf.constant(saved_value, dtype=opt_var.dtype))
                        print(f"Restored optimizer variable: {opt_var.name}")
            except Exception as e:
                print(f"Warning: Could not restore optimizer variables: {e}")
        
        print(f"Optimizer state restored successfully with global step: {global_step}")
        
        # Verify the learning rate is correct after restoration
        current_lr = optimizer.learning_rate
        if hasattr(current_lr, 'numpy'):
            lr_value = current_lr.numpy()
        elif hasattr(current_lr, '__call__'):
            # For custom LR schedules, call with current step
            lr_value = current_lr(global_step).numpy()
        else:
            lr_value = float(current_lr)
        
        print(f"Current learning rate after restoration: {lr_value:.2e}")
    else:
        print("No optimizer state to restore - starting fresh")


def create_fresh_training_setup():
    """Create fresh training setup"""
    print("Starting fresh training...")
    
    # Load datasets
    working_path = 'data/split_data'
    instruments = os.listdir(working_path)
    # For testing with single instrument - uncomment next line if needed
    # instruments = ['SILVER#']
    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments)
    
    # Create model
    model = create_model(training=True)
    
    # Create optimizer and learning rate schedule
    lr_schedule = WarmupCosineDecay(initial_lr=LR, warmup_steps=WARMUP, decay_steps=DECAY)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Create metrics
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=precision_focused_balanced_loss,
        metrics=['accuracy', auc, prec, recall]
    )
    
    return model, optimizer, train_dataset, val_dataset, train_steps, val_steps

def resume_training_setup(state_manager):
    """Resume training from checkpoint"""
    print("Resuming training from checkpoint...")
    
    # Load training state - now includes epoch_durations
    model, optimizer_state, current_epoch, global_step, best_val_loss, patience_count, history, epoch_durations = state_manager.load_training_state()
    
    if model is None:
        raise ValueError("Could not load model from checkpoint")
    
    # Recreate datasets (make sure they're the same as original training)
    working_path = 'data/split_data'
    instruments = os.listdir(working_path)
    # For testing with single instrument - make sure this matches your original setup
    # instruments = ['SILVER#']
    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments)
    
    # Recreate optimizer with same configuration and global step for LR schedule
    lr_schedule = WarmupCosineDecay(initial_lr=LR, warmup_steps=WARMUP, decay_steps=DECAY)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Recompile model (this creates new optimizer instance)
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    model.compile(
        optimizer=optimizer,
        loss=precision_focused_balanced_loss,
        metrics=['accuracy', auc, prec, recall]
    )
    
    # Restore optimizer state with correct global step
    restore_optimizer_state(optimizer, optimizer_state, model, global_step)
    
    return model, optimizer, train_dataset, val_dataset, train_steps, val_steps, current_epoch, best_val_loss, patience_count, global_step, history, epoch_durations

def main():
    """Main training function with resumable capability and time management"""
    
    # Initialize managers
    state_manager = TrainingStateManager()
    time_manager = TimeBasedTrainingManager(
        start_time_hour=7,   # 7 AM
        end_time_hour=23,    # 9 PM
        end_time_minute=0,
        buffer_minutes=5    # 30 minute safety buffer
    )
    
    # Check if we can start training at all today
    if not time_manager.start_training_session():
        print("Cannot start training - outside of allowed training window (7 AM - 9 PM)")
        return None
    
    # Check if we should resume or start fresh
    if state_manager.has_checkpoint():
        try:
            model, optimizer, train_dataset, val_dataset, train_steps, val_steps, \
            current_epoch, best_val_loss, patience_count, global_step, history, epoch_durations = resume_training_setup(state_manager)
            
            initial_epoch = current_epoch
            # Load epoch durations into time manager for accurate time estimation
            time_manager.load_epoch_durations(epoch_durations)
            print(f"Successfully resumed from checkpoint")
        except Exception as e:
            print(f"Failed to resume training: {e}")
            print("Starting fresh training instead...")
            model, optimizer, train_dataset, val_dataset, train_steps, val_steps = create_fresh_training_setup()
            initial_epoch = 0
            best_val_loss = float('inf')
            patience_count = 0
            global_step = 0
    else:
        model, optimizer, train_dataset, val_dataset, train_steps, val_steps = create_fresh_training_setup()
        initial_epoch = 0
        best_val_loss = float('inf')
        patience_count = 0
        global_step = 0
    
    # Check if we can even start an epoch today
    if not time_manager.can_start_epoch():
        print("Cannot start any epochs today - insufficient time remaining")
        print("The training state has been saved. Please run again tomorrow between 7 AM and 9 PM")
        return model  # Return current model
    
    # Print current training status
    print(f"\n{'='*60}")
    print(f"TRAINING STATUS")
    print(f"{'='*60}")
    print(f"Starting from epoch: {initial_epoch}")
    print(f"Global step (for LR schedule): {global_step}")
    print(f"Current best validation loss: {best_val_loss:.6f}")
    print(f"Early stopping patience count: {patience_count}")
    print(f"Training window: 7 AM - 9 PM with 30-minute buffer")
    
    # Show current learning rate
    current_lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate
    print(f"Current learning rate: {current_lr:.2e}")
    
    print(f"Previous epoch durations: {len(time_manager.epoch_durations)}")
    if time_manager.epoch_durations:
        avg_duration = sum(time_manager.epoch_durations) / len(time_manager.epoch_durations)
        print(f"Average epoch duration: {avg_duration/3600:.2f} hours")
    
    print(f"{'='*60}")
    model.summary()
    
    # Create resumable callback with time management
    resumable_callback = ResumableTrainingCallback(state_manager, time_manager, save_freq=1, initial_epoch=initial_epoch)
    resumable_callback.set_initial_state(best_val_loss, patience_count, global_step)
    
    # Train the model
    total_epochs = 50
    remaining_epochs = total_epochs - initial_epoch
    
    if remaining_epochs > 0:
        print(f"\nStarting training for up to {remaining_epochs} more epochs...")
        print(f"Training will automatically stop before 9 PM with 30-minute buffer")
        
        history = model.fit(
            train_dataset,
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[resumable_callback],
            verbose=1
        )
        
        print("Training session completed!")
        
        # Final save
        final_history = {}
        if hasattr(history, 'history'):
            final_history = {key: [float(v) for v in values] for key, values in history.history.items()}
        
        # Calculate final epoch number correctly
        epochs_completed = len(history.history.get('loss', [])) if hasattr(history, 'history') else 0
        final_epoch = initial_epoch + epochs_completed
        
        resumable_callback.state_manager.save_training_state(
            model=model,
            optimizer=model.optimizer,
            epoch=final_epoch,
            best_val_loss=resumable_callback.best_val_loss,
            early_stopping_patience_count=resumable_callback.early_stopping_patience_count,
            global_step=resumable_callback.global_step,
            history=final_history,
            epoch_durations=time_manager.epoch_durations  # Save updated epoch durations
        )
    else:
        print("Training already completed!")
    
    # Load and return the best model for inference
    if os.path.exists(state_manager.best_model_path):
        best_model = tf.keras.models.load_model(
            state_manager.best_model_path,
            custom_objects={'precision_focused_balanced_loss': precision_focused_balanced_loss}
        )
        print(f"Best model loaded from {state_manager.best_model_path}")
        return best_model
    else:
        print("No best model found, returning current model")
        return model

if __name__ == "__main__":
    # Instead of auto_mixed_precision, use explicit policy
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set: {policy.name}")
        
        # Verify it's working
        print(f"Compute dtype: {policy.compute_dtype}")  # Should be float16
        print(f"Variable dtype: {policy.variable_dtype}")  # Should be float32
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
    
    
    # Run the training
    print("Starting resumable training pipeline...")
    final_model = main()
    
    if final_model is not None:
        # Create inference model
        print("Creating inference model...")
        inference_model = create_model(training=False)
        inference_model.set_weights(final_model.get_weights())
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Models available:")
        print("- Best model: checkpoints/best_model.keras")
        print("- Current model: checkpoints/current_model.keras")
        print("- Training can be resumed by running this script again")
        print("="*60)
    else:
        print("Training could not be started due to time constraints.")
        print("Please run again during training hours (7 AM - 9 PM).")