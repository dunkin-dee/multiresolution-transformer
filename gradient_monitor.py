import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from modeler import ScalarScale

class GradientAndWeightMonitor(Callback):
    """
    Monitors gradient norms, weight norms, and NaN values to detect training instabilities
    """
    
    def __init__(self, log_frequency=1, gradient_threshold=10.0, weight_threshold=100.0):
        super().__init__()
        self.log_frequency = log_frequency  # Log every N epochs
        self.gradient_threshold = gradient_threshold  # Alert if gradient norm exceeds this
        self.weight_threshold = weight_threshold  # Alert if weight norm exceeds this
        self.gradient_history = []
        self.weight_history = []
        self.nan_detected = False
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            self._monitor_gradients_and_weights(epoch, logs)
    
    def on_batch_end(self, batch, logs=None):
        # Quick NaN check every batch (lightweight)
        if logs:
            loss = logs.get('loss', 0)
            if np.isnan(loss) or np.isinf(loss):
                print(f"\n🚨 NaN/Inf detected in loss at batch {batch}: {loss}")
                self.nan_detected = True
    
    def _monitor_gradients_and_weights(self, epoch, logs):
        """Monitor gradients and weights for this epoch"""
        
        print(f"\n📊 Gradient & Weight Monitor - Epoch {epoch + 1}")
        print("=" * 60)
        
        # Get a sample batch to compute gradients
        try:
            sample_batch = self._get_sample_batch()
            if sample_batch is None:
                print("⚠️  Could not get sample batch for gradient monitoring")
                return
                
            # Monitor gradients
            gradient_norm = self._compute_gradient_norm(sample_batch)
            self.gradient_history.append((epoch, gradient_norm))
            
            # Monitor weights
            weight_stats = self._compute_weight_stats()
            self.weight_history.append((epoch, weight_stats))
            
            # Report findings
            self._report_gradient_stats(gradient_norm)
            self._report_weight_stats(weight_stats)
            self._check_for_anomalies(gradient_norm, weight_stats)
            
        except Exception as e:
            print(f"⚠️  Error during monitoring: {str(e)}")
    
    def _get_sample_batch(self):
        try:
            if hasattr(self, 'validation_data') and self.validation_data:
                for batch in self.validation_data.take(1):
                    return batch
            
            # Create dictionary-formatted dummy targets
            dummy_targets = {'target_high': tf.random.normal((8, 1))}  # KEY FIX
            
            return (
                [
                    tf.random.normal((8, 64, 4)),  # minute_input
                    tf.random.normal((8, 64, 4)),  # hourly_input
                    tf.random.normal((8, 1, 4)),   # partial_hour_input
                    tf.random.uniform((8, 1), 0, 55),  # minutes_into_hour
                    tf.random.uniform((8, 1), 0, 12)   # partial_hour_length
                ],
                dummy_targets  # Now a dictionary
            )
        except:
            return None
    
    def _compute_gradient_norm(self, sample_batch):
        try:
            inputs, targets = sample_batch
            
            # Handle input formatting
            if isinstance(inputs, dict):
                input_list = [
                    inputs['minute_input'],
                    inputs['hourly_input'], 
                    inputs['partial_hour_input'],
                    inputs['minutes_into_hour'],
                    inputs['partial_hour_length']
                ]
            else:
                input_list = inputs
                
            with tf.GradientTape() as tape:
                predictions = self.model(input_list, training=True)
                
                # Ensure targets are dictionary-formatted
                if not isinstance(targets, dict):
                    targets = {'target_high': targets}
                    
                # Use the actual compiled loss function directly
                loss = self.model.compiled_loss(targets['target_high'], predictions['target_high'])
                
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Handle None gradients
            if any(g is None for g in gradients):
                print("⚠️  Some gradients are None!")
                gradients = [g if g is not None else tf.zeros_like(v) 
                            for g, v in zip(gradients, self.model.trainable_variables)]
            
            gradient_norm = tf.linalg.global_norm(gradients).numpy()
            return gradient_norm
            
        except Exception as e:
            print(f"⚠️  Error computing gradient norm: {str(e)}")
            return 0.0
        
    def _compute_weight_stats(self):
        """Compute statistics about model weights"""
        weight_stats = {
            'max_norm': 0.0,
            'mean_norm': 0.0,
            'num_layers': 0,
            'problematic_layers': []
        }
        
        layer_norms = []
        
        for layer in self.model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weight_norm = tf.norm(layer.kernel).numpy()
                layer_norms.append(weight_norm)
                weight_stats['num_layers'] += 1
                
                # Check for problematic weights
                if weight_norm > self.weight_threshold:
                    weight_stats['problematic_layers'].append((layer.name, weight_norm))
                
                # Check for NaN/Inf in weights
                if np.isnan(weight_norm) or np.isinf(weight_norm):
                    weight_stats['problematic_layers'].append((layer.name, f"NaN/Inf: {weight_norm}"))
        
        if layer_norms:
            weight_stats['max_norm'] = max(layer_norms)
            weight_stats['mean_norm'] = np.mean(layer_norms)
        
        return weight_stats
    
    def _report_gradient_stats(self, gradient_norm):
        """Report gradient statistics"""
        if gradient_norm is None:
            print("🔍 Gradient Norm: None (calculation failed)")
            return
        
        print(f"🔍 Gradient Norm: {gradient_norm:.6f}")
        
        if gradient_norm > self.gradient_threshold:
            print(f"🚨 HIGH GRADIENT NORM! ({gradient_norm:.3f} > {self.gradient_threshold})")
        elif gradient_norm < 1e-8:
            print(f"⚠️  Very small gradient norm ({gradient_norm:.2e}) - possible vanishing gradients")
        else:
            print(f"✅ Gradient norm looks healthy")
    
    def _report_weight_stats(self, weight_stats):
        """Report weight statistics"""
        print(f"⚖️  Weight Stats:")
        print(f"   Max layer norm: {weight_stats['max_norm']:.6f}")
        print(f"   Mean layer norm: {weight_stats['mean_norm']:.6f}")
        print(f"   Layers monitored: {weight_stats['num_layers']}")
        
        if weight_stats['problematic_layers']:
            print(f"🚨 Problematic layers detected:")
            for layer_name, norm in weight_stats['problematic_layers']:
                print(f"   {layer_name}: {norm}")
    
    def _check_for_anomalies(self, gradient_norm, weight_stats):
        """Check for training anomalies and provide recommendations"""
        
        issues = []
        recommendations = []
        
        # Check gradient explosion
        if gradient_norm > self.gradient_threshold:
            issues.append("Gradient explosion detected")
            recommendations.append("Consider reducing learning rate or increasing clipnorm")
        
        # Check weight explosion  
        if weight_stats['max_norm'] > self.weight_threshold:
            issues.append("Weight explosion detected")
            recommendations.append("Consider adding more L2 regularization")
        
        # Check gradient vanishing
        if gradient_norm < 1e-8:
            issues.append("Possible gradient vanishing")
            recommendations.append("Consider increasing learning rate or checking architecture")
        
        # Check for NaN
        if self.nan_detected:
            issues.append("NaN values detected in training")
            recommendations.append("Stop training immediately and check data/model")
        
        # Report issues and recommendations
        if issues:
            print(f"\n🚨 ISSUES DETECTED:")
            for issue in issues:
                print(f"   • {issue}")
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        else:
            print(f"\n✅ No anomalies detected - training looks stable")
    
    def on_train_end(self, logs=None):
        """Summary report at end of training"""
        print(f"\n📈 TRAINING SUMMARY - Gradient & Weight Monitor")
        print("=" * 60)
        
        if self.gradient_history:
            grad_norms = [norm for _, norm in self.gradient_history]
            print(f"Gradient norm range: {min(grad_norms):.6f} - {max(grad_norms):.6f}")
            print(f"Mean gradient norm: {np.mean(grad_norms):.6f}")
        
        if self.weight_history:
            max_norms = [stats['max_norm'] for _, stats in self.weight_history]
            print(f"Max weight norm range: {min(max_norms):.6f} - {max(max_norms):.6f}")
        
        if self.nan_detected:
            print(f"🚨 NaN values were detected during training!")
        else:
            print(f"✅ No NaN values detected during training")


class BranchScalingMonitor(Callback):
    """
    Callback to monitor and log the learnable branch scaling weights during training.
    Prints the weights at specified intervals and tracks their evolution.
    """
    
    def __init__(self, log_frequency=5, save_history=True):
        """
        Args:
            log_frequency (int): Print weights every N epochs
            save_history (bool): Whether to save weight history for plotting
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.save_history = save_history
        self.weight_history = {
            'epochs': [],
            'hourly_scaler': [],
            'partial_scaler': [],
            'minute_scaler': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        # Get current scaling weights
        weights = self._get_branch_weights()
        
        # Save to history if requested
        if self.save_history:
            self.weight_history['epochs'].append(epoch + 1)
            self.weight_history['hourly_scaler'].append(weights.get('hourly_scaler', 1.0))
            self.weight_history['partial_scaler'].append(weights.get('partial_scaler', 1.0))
            self.weight_history['minute_scaler'].append(weights.get('minute_scaler', 1.0))
        
        # Print weights at specified frequency
        if (epoch + 1) % self.log_frequency == 0:
            self._print_weights(epoch + 1, weights, logs)
    
    def on_train_end(self, logs=None):
        """Called at the end of training - print final weights"""
        weights = self._get_branch_weights()
        print("\n" + "="*60)
        print("FINAL BRANCH SCALING WEIGHTS:")
        self._print_weights("Final", weights, logs, detailed=True)
        print("="*60)
    
    def _get_branch_weights(self):
        """Extract branch scaling weights from ScalarScale layers"""
        weights = {}
        for layer in self.model.layers:
            if isinstance(layer, ScalarScale):
                weight_value = float(layer.scale.numpy())
                weights[layer.name] = weight_value
        return weights
    
    def _print_weights(self, epoch, weights, logs=None, detailed=False):
        """Print the current branch weights in a formatted way"""
        print(f"\n📊 Branch Scaling Weights - Epoch {epoch}:")
        print("-" * 50)
        
        # Print each weight with interpretation
        for name, value in weights.items():
            branch_type = name.replace('_scaler', '').capitalize()
            status = self._interpret_weight(value)
            print(f"  {branch_type:8} scale: {value:.4f} {status}")
        
        # Calculate relative influences
        if len(weights) == 3:
            total = sum(weights.values())
            print(f"\n  Relative Influence:")
            for name, value in weights.items():
                branch_type = name.replace('_scaler', '').capitalize()
                percentage = (value / total) * 100
                print(f"    {branch_type:8}: {percentage:.1f}%")
        
        if detailed and self.save_history and len(self.weight_history['epochs']) > 1:
            self._print_weight_evolution()
    
    def _interpret_weight(self, weight):
        """Provide interpretation of weight values"""
        if weight > 1.2:
            return "🔥 (High influence)"
        elif weight > 1.05:
            return "📈 (Increased influence)"
        elif weight < 0.8:
            return "📉 (Reduced influence)"
        elif weight < 0.95:
            return "⬇️  (Slightly reduced)"
        else:
            return "➡️  (Balanced)"
    
    def _print_weight_evolution(self):
        """Print how weights have changed over training"""
        if len(self.weight_history['epochs']) < 2:
            return
            
        print(f"\n  Weight Evolution (from start to end):")
        initial_epoch = self.weight_history['epochs'][0]
        final_epoch = self.weight_history['epochs'][-1]
        
        for weight_name in ['hourly_scaler', 'partial_scaler', 'minute_scaler']:
            if weight_name in self.weight_history:
                initial = self.weight_history[weight_name][0]
                final = self.weight_history[weight_name][-1]
                change = final - initial
                change_percent = (change / initial) * 100
                
                branch_type = weight_name.replace('_scaler', '').capitalize()
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"    {branch_type:8}: {initial:.3f} → {final:.3f} "
                      f"({change:+.3f}, {change_percent:+.1f}%) {direction}")
    
    def plot_weight_evolution(self, save_path=None):
        """Plot the evolution of branch weights over training"""
        if not self.save_history or len(self.weight_history['epochs']) < 2:
            print("No weight history available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot weight evolution
            epochs = self.weight_history['epochs']
            plt.plot(epochs, self.weight_history['hourly_scaler'], 'b-', label='Hourly Scale', linewidth=2)
            plt.plot(epochs, self.weight_history['partial_scaler'], 'g-', label='Partial Scale', linewidth=2)
            plt.plot(epochs, self.weight_history['minute_scaler'], 'r-', label='Minute Scale', linewidth=2)
            
            plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Weight (1.0)')
            
            plt.xlabel('Epoch')
            plt.ylabel('Scaling Weight')
            plt.title('Branch Scaling Weight Evolution During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Weight evolution plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting. Install with: pip install matplotlib")