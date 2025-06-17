import polars as pl
import numpy as np
import tensorflow as tf
import os
import logging
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration class for dataset parameters"""
    main_data_dir: str
    hourly_data_path: str
    main_lookback_tokens: int
    hourly_lookback_tokens: int
    lookback_window: int = 1440
    batch_size: int = 32
    apply_smote: bool = False
    shuffle_data: bool = False
    feature_columns: List[str] = None
    max_chunks_in_memory: int = 100  # Memory management
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ['open', 'high', 'low', 'close']
        
        # Validation
        if not os.path.exists(self.main_data_dir):
            raise FileNotFoundError(f"Main data directory not found: {self.main_data_dir}")
        if not os.path.exists(self.hourly_data_path):
            raise FileNotFoundError(f"Hourly data file not found: {self.hourly_data_path}")
        if self.main_lookback_tokens <= 0 or self.hourly_lookback_tokens <= 0:
            raise ValueError("Lookback tokens must be positive integers")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


class ChunkManager:
    """Manages loading and unloading of data chunks to control memory usage"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.chunk_files = self._discover_chunk_files()
        self.chunk_cache: Dict[int, pl.DataFrame] = {}
        self.cache_order: List[int] = []  # LRU tracking
        
    def _discover_chunk_files(self) -> List[str]:
        """Discover and sort chunk files"""
        chunk_files = sorted([
            f for f in os.listdir(self.config.main_data_dir) 
            if f.endswith('.csv')
        ])
        if not chunk_files:
            raise ValueError(f"No CSV files found in {self.config.main_data_dir}")
        
        logger.info(f"Discovered {len(chunk_files)} chunk files")
        return chunk_files
    
    def get_chunk(self, chunk_idx: int) -> pl.DataFrame:
        """Get chunk with LRU caching"""
        if chunk_idx in self.chunk_cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.chunk_cache[chunk_idx]
        
        # Load chunk
        chunk_path = os.path.join(self.config.main_data_dir, self.chunk_files[chunk_idx])
        required_cols = self.config.feature_columns + ['include', 'time', 'target']
        
        try:
            chunk_df = pl.scan_csv(chunk_path).select(required_cols).collect()
        except Exception as e:
            raise RuntimeError(f"Failed to load chunk {chunk_idx} ({self.chunk_files[chunk_idx]}): {e}")
        
        # Cache management
        if len(self.chunk_cache) >= self.config.max_chunks_in_memory:
            # Remove least recently used
            lru_chunk = self.cache_order.pop(0)
            del self.chunk_cache[lru_chunk]
        
        self.chunk_cache[chunk_idx] = chunk_df
        self.cache_order.append(chunk_idx)
        return chunk_df
    
    def get_chunk_count(self) -> int:
        return len(self.chunk_files)


class IndexManager:
    """Manages global indexing and SMOTE operations"""
    
    def __init__(self, config: DatasetConfig, chunk_manager: ChunkManager, time_threshold: Optional[float] = None):
        self.config = config
        self.chunk_manager = chunk_manager
        self.time_threshold = time_threshold
        self.global_indices_positive: List[Tuple[int, int]] = []
        self.global_indices_negative: List[Tuple[int, int]] = []
        self._build_indices()
    
    def _build_indices(self):
        """Build global indices for all valid samples"""
        logger.info("Building global indices...")
        
        total_valid = 0
        for chunk_idx in range(self.chunk_manager.get_chunk_count()):
            chunk_df = self.chunk_manager.get_chunk(chunk_idx)
            
            # Apply time filtering if threshold exists
            if self.time_threshold is not None:
                chunk_df = chunk_df.filter(pl.col('time') >= self.time_threshold)
            
            if chunk_df.shape[0] == 0:
                continue
            
            # Find valid indices within this chunk
            for row_idx in range(self.config.lookback_window, chunk_df.shape[0]):
                if chunk_df[row_idx, 'include'] == 1:
                    index_tuple = (chunk_idx, row_idx)
                    target_value = chunk_df[row_idx, 'target']
                    
                    if target_value == 0:
                        self.global_indices_negative.append(index_tuple)
                    else:
                        self.global_indices_positive.append(index_tuple)
                    
                    total_valid += 1
        
        logger.info(f"Built indices: {len(self.global_indices_positive)} positive, "
                   f"{len(self.global_indices_negative)} negative samples")
        
        if total_valid == 0:
            raise ValueError("No valid samples found after filtering")
    
    def get_balanced_indices(self) -> List[Tuple[int, int]]:
        """Get indices with SMOTE balancing applied"""
        if not self.config.apply_smote:
            return self.global_indices_positive + self.global_indices_negative
        
        if not self.global_indices_positive or not self.global_indices_negative:
            logger.warning("SMOTE requested but one class is empty, returning all indices")
            return self.global_indices_positive + self.global_indices_negative
        
        # Balance classes efficiently
        majority_size = max(len(self.global_indices_positive), len(self.global_indices_negative))
        
        balanced_positive = self._replicate_to_size(self.global_indices_positive, majority_size)
        balanced_negative = self._replicate_to_size(self.global_indices_negative, majority_size)
        
        return balanced_positive + balanced_negative
    
    @staticmethod
    def _replicate_to_size(indices: List[Tuple[int, int]], target_size: int) -> List[Tuple[int, int]]:
        """Efficiently replicate indices to target size"""
        if not indices:
            return []
        
        full_cycles = target_size // len(indices)
        remainder = target_size % len(indices)
        
        result = indices * full_cycles
        if remainder:
            result.extend(indices[:remainder])
        
        return result


class MultiResolutionDataGenerator:
    """Main data generator class with improved architecture"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.chunk_manager = ChunkManager(config)
        self.hourly_df = self._load_hourly_data()
        self.hourly_time_array = self.hourly_df['time'].to_numpy()
        self.time_threshold = self._calculate_time_threshold()
        self.index_manager = IndexManager(config, self.chunk_manager, self.time_threshold)

    def get_total_training_samples(self) -> int:
        """
        Calculate total number of training samples that will be generated.
        
        Returns:
            Total number of samples after filtering and SMOTE balancing
        """
        if not self.config.apply_smote:
            # Without SMOTE, return total valid samples
            return len(self.index_manager.global_indices_positive) + len(self.index_manager.global_indices_negative)
        
        # With SMOTE, return balanced total (2 * majority class size)
        if not self.index_manager.global_indices_positive or not self.index_manager.global_indices_negative:
            # One class is empty, return what we have
            return len(self.index_manager.global_indices_positive) + len(self.index_manager.global_indices_negative)
        
        # Return 2 * majority class size (balanced dataset)
        majority_size = max(
            len(self.index_manager.global_indices_positive),
            len(self.index_manager.global_indices_negative)
        )
        return majority_size * 2
    
    def _load_hourly_data(self) -> pl.DataFrame:
        """Load and validate hourly data"""
        logger.info("Loading hourly data...")
        
        try:
            hourly_cols = self.config.feature_columns + ['time']
            hourly_df = pl.scan_csv(self.config.hourly_data_path).select(hourly_cols).collect()
            
            if hourly_df.shape[0] < self.config.hourly_lookback_tokens + 1:
                raise ValueError(f"Insufficient hourly data: need at least {self.config.hourly_lookback_tokens + 1} rows")
            
            logger.info(f"Loaded hourly data: {hourly_df.shape[0]} rows")
            return hourly_df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load hourly data: {e}")
    
    def _calculate_time_threshold(self) -> Optional[float]:
        """Calculate time threshold based on hourly data"""
        if self.hourly_df.shape[0] >= self.config.hourly_lookback_tokens + 1:
            threshold = self.hourly_df[self.config.hourly_lookback_tokens, 'time']
            logger.info(f"Applied time threshold: {threshold}")
            return threshold
        return None
    
    def generate_batches(self) -> Generator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], None, None]:
        """Generate batches of data"""
        while True:  # Infinite generator for epochs
            working_indices = self.index_manager.get_balanced_indices()
            
            if self.config.shuffle_data:
                np.random.shuffle(working_indices)
            
            # Pre-allocate batch arrays for better performance
            batch_main = np.empty((self.config.batch_size, self.config.main_lookback_tokens, 
                                 len(self.config.feature_columns)), dtype=np.float32)
            batch_hourly = np.empty((self.config.batch_size, self.config.hourly_lookback_tokens, 
                                   len(self.config.feature_columns)), dtype=np.float32)
            batch_targets = np.empty(self.config.batch_size, dtype=np.int32)
            
            batch_idx = 0
            samples_processed = 0
            
            for chunk_idx, row_idx in working_indices:
                try:
                    sample_data = self._extract_sample(chunk_idx, row_idx)
                    if sample_data is None:
                        continue
                    
                    main_seq, hourly_seq, target = sample_data
                    
                    batch_main[batch_idx] = main_seq
                    batch_hourly[batch_idx] = hourly_seq
                    batch_targets[batch_idx] = target
                    batch_idx += 1
                    samples_processed += 1
                    
                    if batch_idx == self.config.batch_size:
                        # Yield full batch
                        yield (
                            tf.convert_to_tensor(batch_main, dtype=tf.float32),
                            tf.convert_to_tensor(batch_hourly, dtype=tf.float32),
                            tf.convert_to_tensor(batch_targets, dtype=tf.int32)
                        )
                        batch_idx = 0
                
                except Exception as e:
                    logger.warning(f"Skipping sample ({chunk_idx}, {row_idx}): {e}")
                    continue
            
            # Yield remaining samples
            if batch_idx > 0:
                yield (
                    tf.convert_to_tensor(batch_main[:batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(batch_hourly[:batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(batch_targets[:batch_idx], dtype=tf.int32)
                )
            
            break  # Remove for infinite epochs
    
    def _extract_sample(self, chunk_idx: int, row_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Extract a single sample with error handling"""
        try:
            chunk_df = self.chunk_manager.get_chunk(chunk_idx)
            
            # Extract main sequence
            sequence_start = row_idx - self.config.lookback_window + 1
            sequence_end = row_idx + 1
            main_sequence = chunk_df[sequence_start:sequence_end, self.config.feature_columns].to_numpy()
            
            # Get timestamp and find hourly position
            current_timestamp = chunk_df[row_idx, 'time']
            hourly_position = np.searchsorted(self.hourly_time_array, current_timestamp, side='right') - 1
            
            if hourly_position < self.config.hourly_lookback_tokens:
                return None
            
            # Extract hourly sequence
            hourly_start = hourly_position - self.config.hourly_lookback_tokens
            hourly_end = hourly_position
            hourly_sequence = self.hourly_df[hourly_start:hourly_end, self.config.feature_columns].to_numpy()
            
            # Get inputs and target
            main_input = main_sequence[-self.config.main_lookback_tokens:]
            target_value = chunk_df[row_idx, 'target']
            
            return main_input, hourly_sequence, target_value
            
        except Exception as e:
            logger.error(f"Error extracting sample ({chunk_idx}, {row_idx}): {e}")
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            'total_chunks': self.chunk_manager.get_chunk_count(),
            'positive_samples': len(self.index_manager.global_indices_positive),
            'negative_samples': len(self.index_manager.global_indices_negative),
            'hourly_data_rows': self.hourly_df.shape[0],
            'time_threshold': self.time_threshold,
            'feature_columns': self.config.feature_columns,
        }


def create_chunked_dataset_generator(config: DatasetConfig, repeat_dataset: bool = False) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from chunked data generator with improved error handling.
    
    Args:
        config: Dataset configuration object
        repeat_dataset: Whether to repeat dataset infinitely
    
    Returns:
        TensorFlow dataset
    """
    generator = MultiResolutionDataGenerator(config)
    
    # Print dataset info
    info = generator.get_dataset_info()
    logger.info(f"Dataset info: {info}")
    
    dataset = tf.data.Dataset.from_generator(
        generator.generate_batches,
        output_signature=(
            tf.TensorSpec(shape=(None, config.main_lookback_tokens, len(config.feature_columns)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, config.hourly_lookback_tokens, len(config.feature_columns)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    
    # Map to expected format ((main_input, hourly_input), target)
    dataset = dataset.map(lambda main, hourly, target: ((main, hourly), target))
    
    if repeat_dataset:
        dataset = dataset.repeat()
    
    return dataset, generator.get_total_training_samples()

def get_dataset_sample_count(config: DatasetConfig) -> int:
    """
    Get total sample count without creating the full generator.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Total number of training samples
    """
    generator = MultiResolutionDataGenerator(config)
    return generator.get_total_training_samples()


# Example usage with improved configuration
if __name__ == "__main__":
    # Create configuration
    config = DatasetConfig(
        main_data_dir="/path/to/chunked/csvs",
        hourly_data_path="/path/to/hourly_data.csv",
        main_lookback_tokens=100,
        hourly_lookback_tokens=24,
        lookback_window=1440,
        batch_size=32,
        apply_smote=True,
        shuffle_data=True,
        feature_columns=['open', 'high', 'low', 'close'],
        max_chunks_in_memory=5  # Memory management
    )
    
    # Create dataset
    try:
        train_dataset = create_chunked_dataset_generator(
            config=config,
            repeat_dataset=True
        )
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise