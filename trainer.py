import tensorflow as tf
import os
import json
import pickle
from datetime import datetime, time
from tensorflow.keras.callbacks import Callback
from losses import recommended_trading_loss
from datasets import get_datasets_and_steps
from transformer_builder import WarmupCosineDecay
from modeler import create_model

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
            return 27000
        
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
                          early_stopping_patience_count, global_step, history=None):
        """Save complete training state"""
        print(f"Saving training state at epoch {epoch}, global step {global_step}...")
        
        # Save current model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Save optimizer state (weights and momentum)
        optimizer_state = {
            'config': optimizer.get_config(),
            'weights': optimizer.get_weights() if len(optimizer.get_weights()) > 0 else None,
            'global_step': int(global_step)  # Save the global step for LR schedule
        }
        with open(self.optimizer_path, 'wb') as f:
            pickle.dump(optimizer_state, f)
        print(f"Optimizer state saved to {self.optimizer_path}")
        
        # Save training metadata
        training_state = {
            'current_epoch': epoch,
            'global_step': int(global_step),  # Critical for LR schedule continuity
            'best_val_loss': float(best_val_loss),
            'early_stopping_patience_count': early_stopping_patience_count,
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
            return None, None, 0, 0, float('inf'), 0, {}
        
        print("Loading previous training state...")
        
        # Load training metadata
        with open(self.training_state_path, 'r') as f:
            training_state = json.load(f)
        
        current_epoch = training_state['current_epoch']
        global_step = training_state.get('global_step', 0)  # Global step for LR schedule
        best_val_loss = training_state['best_val_loss']
        early_stopping_patience_count = training_state['early_stopping_patience_count']
        
        print(f"Resuming from epoch {current_epoch}, global step {global_step}, best_val_loss: {best_val_loss}")
        
        # Load model
        model = None
        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects={'recommended_trading_loss': recommended_trading_loss}
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
        
        return model, optimizer_state, current_epoch, global_step, best_val_loss, early_stopping_patience_count, history
    
    def save_best_model(self, model):
        """Save the best model separately"""
        model.save(self.best_model_path)
        print(f"Best model saved to {self.best_model_path}")
    
    def has_checkpoint(self):
        """Check if checkpoint exists"""
        return os.path.exists(self.training_state_path)

class ResumableTrainingCallback(Callback):
    """Custom callback to handle state saving, best model tracking, and time management"""
    
    def __init__(self, state_manager, time_manager, save_freq=1):
        super().__init__()
        self.state_manager = state_manager
        self.time_manager = time_manager
        self.save_freq = save_freq  # Save every N epochs
        self.best_val_loss = float('inf')
        self.early_stopping_patience_count = 0
        self.early_stopping_patience = 10
        self.global_step = 0  # Track global steps for LR schedule
        self.epoch_start_time = None
        
    def set_initial_state(self, best_val_loss, patience_count, global_step=0):
        """Set initial state when resuming"""
        self.best_val_loss = best_val_loss
        self.early_stopping_patience_count = patience_count
        self.global_step = global_step
        print(f"Initial state set - Best val loss: {best_val_loss}, Patience: {patience_count}, Global step: {global_step}")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Check if we can safely start this epoch"""
        self.epoch_start_time = datetime.now()
        
        if not self.time_manager.can_start_epoch():
            print(f"Time cutoff reached. Stopping training before epoch {epoch + 1}")
            self.model.stop_training = True
            return
            
        print(f"Starting epoch {epoch + 1} at {self.epoch_start_time.strftime('%H:%M:%S')}")
    
    def on_batch_end(self, batch, logs=None):
        """Update global step counter after each batch"""
        self.global_step += 1
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_loss = logs.get('val_loss', float('inf'))
        
        # Record epoch duration for time estimation
        if self.epoch_start_time:
            epoch_duration = (datetime.now() - self.epoch_start_time).total_seconds()
            self.time_manager.record_epoch_duration(epoch_duration)
            print(f"Epoch {epoch + 1} completed in {epoch_duration/3600:.2f} hours")
        
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
                epoch=epoch + 1,
                best_val_loss=self.best_val_loss,
                early_stopping_patience_count=self.early_stopping_patience_count,
                global_step=self.global_step,
                history=history_dict
            )
        
        # Check early stopping
        if self.early_stopping_patience_count >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
            self.model.stop_training = True
            
        # Print time summary
        print(self.time_manager.get_training_summary())

def restore_optimizer_state(optimizer, optimizer_state, model, global_step):
    """Restore optimizer state including momentum"""
    if optimizer_state and optimizer_state['weights'] is not None:
        # Need to run one step to initialize optimizer variables
        dummy_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        optimizer.apply_gradients(zip(dummy_gradients, model.trainable_variables))
        
        # Now set the saved weights (this includes the step counter and momentum)
        try:
            optimizer.set_weights(optimizer_state['weights'])
            print(f"Optimizer weights restored successfully")
        except Exception as e:
            print(f"Warning: Could not restore optimizer weights: {e}")
        
        # Manually set the optimizer's iteration counter to maintain LR schedule
        if hasattr(optimizer, 'iterations'):
            optimizer.iterations.assign(global_step)
            print(f"Optimizer iteration counter set to: {global_step}")
        
        print(f"Optimizer state restored successfully with global step: {global_step}")
    else:
        print("No optimizer state to restore - starting fresh")

def create_fresh_training_setup():
    """Create fresh training setup"""
    print("Starting fresh training...")
    
    # Load datasets
    working_path = 'data/split_data'
    instruments = os.listdir(working_path)
    # For testing with single instrument - uncomment next line if needed
    instruments = ['SILVER#']
    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments)
    
    # Create model
    model = create_model(training=True)
    
    # Create optimizer and learning rate schedule
    lr_schedule = WarmupCosineDecay(initial_lr=2e-5, warmup_steps=600000, decay_steps=6000000)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    # Create metrics
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=recommended_trading_loss,
        metrics=['accuracy', auc, prec]
    )
    
    return model, optimizer, train_dataset, val_dataset, train_steps, val_steps

def resume_training_setup(state_manager):
    """Resume training from checkpoint"""
    print("Resuming training from checkpoint...")
    
    # Load training state
    model, optimizer_state, current_epoch, global_step, best_val_loss, patience_count, history = state_manager.load_training_state()
    
    if model is None:
        raise ValueError("Could not load model from checkpoint")
    
    # Recreate datasets (make sure they're the same as original training)
    working_path = 'data/split_data'
    instruments = os.listdir(working_path)
    # For testing with single instrument - make sure this matches your original setup
    # instruments = ['SILVER#']
    (train_dataset, val_dataset, test_dataset), (train_steps, val_steps, test_steps) = get_datasets_and_steps(instruments)
    
    # Recreate optimizer with same configuration and global step for LR schedule
    lr_schedule = WarmupCosineDecay(initial_lr=2e-5, warmup_steps=600000, decay_steps=6000000)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    # Recompile model (this creates new optimizer instance)
    auc = tf.keras.metrics.AUC()
    prec = tf.keras.metrics.Precision()
    model.compile(
        optimizer=optimizer,
        loss=recommended_trading_loss,
        metrics=['accuracy', auc, prec]
    )
    
    # Restore optimizer state with correct global step
    restore_optimizer_state(optimizer, optimizer_state, model, global_step)
    
    return model, optimizer, train_dataset, val_dataset, train_steps, val_steps, current_epoch, best_val_loss, patience_count, global_step, history

def main():
    """Main training function with resumable capability and time management"""
    
    # Initialize managers
    state_manager = TrainingStateManager()
    time_manager = TimeBasedTrainingManager(
        start_time_hour=7,   # 7 AM
        end_time_hour=21,    # 9 PM
        end_time_minute=0,
        buffer_minutes=30    # 30 minute safety buffer
    )
    
    # Check if we can start training at all today
    if not time_manager.start_training_session():
        print("Cannot start training - outside of allowed training window (7 AM - 9 PM)")
        return None
    
    # Check if we should resume or start fresh
    if state_manager.has_checkpoint():
        try:
            model, optimizer, train_dataset, val_dataset, train_steps, val_steps, \
            current_epoch, best_val_loss, patience_count, global_step, history = resume_training_setup(state_manager)
            
            initial_epoch = current_epoch
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
    print(f"{'='*60}")
    model.summary()
    
    # Create resumable callback with time management
    resumable_callback = ResumableTrainingCallback(state_manager, time_manager, save_freq=1)
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
        
        resumable_callback.state_manager.save_training_state(
            model=model,
            optimizer=model.optimizer,
            epoch=initial_epoch + len(history.history.get('loss', [])) if hasattr(history, 'history') else initial_epoch,
            best_val_loss=resumable_callback.best_val_loss,
            early_stopping_patience_count=resumable_callback.early_stopping_patience_count,
            global_step=resumable_callback.global_step,
            history=final_history
        )
    else:
        print("Training already completed!")
    
    # Load and return the best model for inference
    if os.path.exists(state_manager.best_model_path):
        best_model = tf.keras.models.load_model(
            state_manager.best_model_path,
            custom_objects={'recommended_trading_loss': recommended_trading_loss}
        )
        print(f"Best model loaded from {state_manager.best_model_path}")
        return best_model
    else:
        print("No best model found, returning current model")
        return model

if __name__ == "__main__":
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