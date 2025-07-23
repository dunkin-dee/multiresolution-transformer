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
class InstrumentConfig:
    """Configuration for a single instrument"""
    name: str  # e.g., 'GBPUSD', 'EURUSD'
    hourly_data_path: str
    chunked_data_dir: str
    
    def __post_init__(self):
        if not os.path.exists(self.hourly_data_path):
            raise FileNotFoundError(f"Hourly data not found for {self.name}: {self.hourly_data_path}")
        if not os.path.exists(self.chunked_data_dir):
            raise FileNotFoundError(f"Chunked data directory not found for {self.name}: {self.chunked_data_dir}")

@dataclass
class MultiInstrumentDatasetConfig:
    """Configuration class for multi-instrument dataset parameters"""
    instruments: List[InstrumentConfig]
    main_lookback_tokens: int
    hourly_lookback_tokens: int
    lookback_window: int = 1440
    batch_size: int = 32
    shuffle_data: bool = False
    feature_columns: List[str] = None
    max_chunks_per_instrument: int = 20  # Memory management per instrument
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ['open', 'high', 'low', 'close']
        
        # Validation
        if not self.instruments:
            raise ValueError("At least one instrument must be provided")
        if self.main_lookback_tokens <= 0 or self.hourly_lookback_tokens <= 0:
            raise ValueError("Lookback tokens must be positive integers")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


class InstrumentChunkManager:
    """Manages loading and unloading of data chunks for a single instrument"""
    
    def __init__(self, instrument_config: InstrumentConfig, feature_columns: List[str], max_chunks: int):
        self.instrument_config = instrument_config
        self.feature_columns = feature_columns
        self.max_chunks = max_chunks
        self.chunk_files = self._discover_chunk_files()
        self.chunk_cache: Dict[int, pl.DataFrame] = {}
        self.cache_order: List[int] = []  # LRU tracking
        
    def _discover_chunk_files(self) -> List[str]:
        """Discover and sort chunk files"""
        chunk_files = sorted([
            f for f in os.listdir(self.instrument_config.chunked_data_dir) 
            if f.endswith('.csv')
        ])
        if not chunk_files:
            raise ValueError(f"No CSV files found in {self.instrument_config.chunked_data_dir}")
        
        logger.info(f"Discovered {len(chunk_files)} chunk files for {self.instrument_config.name}")
        return chunk_files
    
    def get_chunk(self, chunk_idx: int) -> pl.DataFrame:
        """Get chunk with LRU caching"""
        if chunk_idx in self.chunk_cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.chunk_cache[chunk_idx]
        
        # Load chunk
        chunk_path = os.path.join(self.instrument_config.chunked_data_dir, self.chunk_files[chunk_idx])
        required_cols = list(set(self.feature_columns + ['include', 'time', 'target_high', 'target_low']))
        
        try:
            chunk_df = pl.scan_csv(chunk_path).select(required_cols).collect()
        except Exception as e:
            raise RuntimeError(f"Failed to load chunk {chunk_idx} for {self.instrument_config.name} ({self.chunk_files[chunk_idx]}): {e}")
        
        # Cache management (unchanged)
        if len(self.chunk_cache) >= self.max_chunks:
            lru_chunk = self.cache_order.pop(0)
            del self.chunk_cache[lru_chunk]
            logger.debug(f"Evicted chunk {lru_chunk} from cache for {self.instrument_config.name}")
        
        self.chunk_cache[chunk_idx] = chunk_df
        self.cache_order.append(chunk_idx)
        return chunk_df
        
    def get_chunk_count(self) -> int:
        return len(self.chunk_files)


class SingleInstrumentProcessor:
    """Processes data for a single instrument - encapsulates all instrument-specific logic"""
    
    def __init__(self, instrument_config: InstrumentConfig, config: MultiInstrumentDatasetConfig):
        self.instrument_config = instrument_config
        self.config = config
        self.chunk_manager = InstrumentChunkManager(
            instrument_config, 
            config.feature_columns, 
            config.max_chunks_per_instrument
        )
        self.hourly_df = self._load_hourly_data()
        self.hourly_time_array = self.hourly_df['time'].to_numpy()
        self.time_threshold = self._calculate_time_threshold()
        self.valid_indices: List[Tuple[int, int]] = []  # Only need valid indices now
        self._build_indices()
    
    def _load_hourly_data(self) -> pl.DataFrame:
        """Load and validate hourly data for this instrument"""
        logger.info(f"Loading hourly data for {self.instrument_config.name}...")
        
        try:
            hourly_cols = list(set(self.config.feature_columns + ['time']))
            hourly_df = pl.scan_csv(self.instrument_config.hourly_data_path).select(hourly_cols).collect()
            
            if hourly_df.shape[0] < self.config.hourly_lookback_tokens + 1:
                raise ValueError(f"Insufficient hourly data for {self.instrument_config.name}: need at least {self.config.hourly_lookback_tokens + 1} rows")
            
            logger.info(f"Loaded hourly data for {self.instrument_config.name}: {hourly_df.shape[0]} rows")
            return hourly_df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load hourly data for {self.instrument_config.name}: {e}")
    
    def _calculate_time_threshold(self) -> Optional[float]:
        """Calculate time threshold based on hourly data"""
        if self.hourly_df.shape[0] >= self.config.hourly_lookback_tokens + 1:
            threshold = self.hourly_df[self.config.hourly_lookback_tokens, 'time']
            logger.info(f"Applied time threshold for {self.instrument_config.name}: {threshold}")
            return threshold
        return None
    
    def _build_indices(self):
        """Build indices for all valid samples for this instrument"""
        logger.info(f"Building indices for {self.instrument_config.name}...")
        
        for chunk_idx in range(self.chunk_manager.get_chunk_count()):
            chunk_df = self.chunk_manager.get_chunk(chunk_idx)
            
            # Apply time filtering if threshold exists
            if self.time_threshold is not None:
                chunk_df = chunk_df.filter(pl.col('time') >= self.time_threshold)
            
            if chunk_df.shape[0] == 0:
                continue
            
            # Find valid indices within this chunk - simplified for regression
            for row_idx in range(self.config.lookback_window, chunk_df.shape[0]):
                if chunk_df[row_idx, 'include'] == 1:
                    self.valid_indices.append((chunk_idx, row_idx))
        
        logger.info(f"Built indices for {self.instrument_config.name}: {len(self.valid_indices)} valid samples")
    
    def extract_sample(self, chunk_idx: int, row_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract a single sample with error handling - returns dual targets"""
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
            
            # Get inputs and dual targets
            main_input = main_sequence[-self.config.main_lookback_tokens:]
            target_values = np.array([
                chunk_df[row_idx, 'target_high'], 
                chunk_df[row_idx, 'target_low']
            ], dtype=np.float32)  # ← CHANGED: dual regression targets
            
            return main_input, hourly_sequence, target_values
        
        except Exception as e:
            logger.error(f"Error extracting sample for {self.instrument_config.name} ({chunk_idx}, {row_idx}): {e}")
            return None
        

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this instrument"""
        return {
            'name': self.instrument_config.name,
            'total_chunks': self.chunk_manager.get_chunk_count(),
            'valid_samples': len(self.valid_indices),  # ← SIMPLIFIED
            'hourly_data_rows': self.hourly_df.shape[0],
            'time_threshold': self.time_threshold,
        }



class MultiInstrumentDataGenerator:
    """Main multi-instrument data generator class"""
    
    def __init__(self, config: MultiInstrumentDatasetConfig):
        self.config = config
        # Create processors for each instrument
        self.processors = [
            SingleInstrumentProcessor(instrument_config, config)
            for instrument_config in config.instruments
        ]
        # Build simple global indices: (instrument_idx, chunk_idx, row_idx)
        self.global_indices: List[Tuple[int, int, int]] = []
        self._build_global_indices()

    def _build_global_indices(self):
        """Build global indices across all instruments"""
        logger.info("Building global indices across all instruments...")
        
        for instrument_idx, processor in enumerate(self.processors):
            for chunk_idx, row_idx in processor.valid_indices:
                self.global_indices.append((instrument_idx, chunk_idx, row_idx))
        
        logger.info(f"Global indices built: {len(self.global_indices)} total samples across {len(self.processors)} instruments")
        
        if len(self.global_indices) == 0:
            raise ValueError("No valid samples found across all instruments")
    
    def get_total_training_samples(self) -> int:
        """Calculate total number of training samples across all instruments"""
        return len(self.global_indices)  # ← SIMPLIFIED
    

    def generate_batches(self) -> Generator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], None, None]:
        """Generate batches of data from all instruments"""
        while True:  # Infinite generator for epochs
            working_indices = self.global_indices.copy()
            
            if self.config.shuffle_data:
                np.random.shuffle(working_indices)
            
            # Pre-allocate batch arrays for better performance
            batch_main = np.empty((self.config.batch_size, self.config.main_lookback_tokens, 
                                len(self.config.feature_columns)), dtype=np.float32)
            batch_hourly = np.empty((self.config.batch_size, self.config.hourly_lookback_tokens, 
                                len(self.config.feature_columns)), dtype=np.float32)
            batch_targets = np.empty((self.config.batch_size, 2), dtype=np.float32)  # ← CHANGED: shape (batch_size, 2)
            
            batch_idx = 0
            samples_processed = 0
            
            for instrument_idx, chunk_idx, row_idx in working_indices:
                try:
                    processor = self.processors[instrument_idx]
                    sample_data = processor.extract_sample(chunk_idx, row_idx)
                    
                    if sample_data is None:
                        continue
                    
                    main_seq, hourly_seq, targets = sample_data
                    
                    batch_main[batch_idx] = main_seq
                    batch_hourly[batch_idx] = hourly_seq
                    batch_targets[batch_idx] = targets  # ← targets is now [target_high, target_low]
                    batch_idx += 1
                    samples_processed += 1
                    
                    if batch_idx == self.config.batch_size:
                        # Yield full batch
                        yield (
                            tf.convert_to_tensor(batch_main, dtype=tf.float32),
                            tf.convert_to_tensor(batch_hourly, dtype=tf.float32),
                            tf.convert_to_tensor(batch_targets, dtype=tf.float32)  # ← CHANGED: float32 for regression
                        )
                        batch_idx = 0
                
                except Exception as e:
                    instrument_name = self.config.instruments[instrument_idx].name
                    logger.warning(f"Skipping sample from {instrument_name} ({chunk_idx}, {row_idx}): {e}")
                    continue
            
            # Yield remaining samples
            if batch_idx > 0:
                yield (
                    tf.convert_to_tensor(batch_main[:batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(batch_hourly[:batch_idx], dtype=tf.float32),
                    tf.convert_to_tensor(batch_targets[:batch_idx], dtype=tf.float32)  # ← CHANGED: float32
                )
            
            break  # Remove for infinite epochs
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the multi-instrument dataset"""
        instrument_stats = [processor.get_stats() for processor in self.processors]
        
        total_samples = sum(stats['valid_samples'] for stats in instrument_stats)
        
        return {
            'total_instruments': len(self.processors),
            'instruments': instrument_stats,
            'total_samples': total_samples,
            'feature_columns': self.config.feature_columns,
            'target_columns': ['target_high', 'target_low'],
            'shuffle_enabled': self.config.shuffle_data,
        }

def create_multi_instrument_dataset(config: MultiInstrumentDatasetConfig, repeat_dataset: bool = False) -> Tuple[tf.data.Dataset, int]:
    """Create TensorFlow dataset from multi-instrument data generator."""
    generator = MultiInstrumentDataGenerator(config)
    
    # Print comprehensive dataset info
    info = generator.get_dataset_info()
    logger.info("=== Multi-Instrument Regression Dataset Info ===")
    logger.info(f"Total instruments: {info['total_instruments']}")
    for instrument_info in info['instruments']:
        logger.info(f"  {instrument_info['name']}: {instrument_info['valid_samples']} samples, "
                   f"{instrument_info['total_chunks']} chunks")
    logger.info(f"Global total: {info['total_samples']} samples")
    logger.info(f"Target columns: {info['target_columns']}")
    logger.info(f"Shuffle: {info['shuffle_enabled']}")
    logger.info("=====================================")
    
    dataset = tf.data.Dataset.from_generator(
        generator.generate_batches,
        output_signature=(
            tf.TensorSpec(shape=(None, config.main_lookback_tokens, len(config.feature_columns)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, config.hourly_lookback_tokens, len(config.feature_columns)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)  # ← CHANGED: shape (None, 2) for dual targets
        )
    )
    
    # Map to expected format ((main_input, hourly_input), target)
    dataset = dataset.map(lambda main, hourly, target: ((main, hourly), target))
    
    if repeat_dataset:
        dataset = dataset.repeat()
    
    return dataset, generator.get_total_training_samples()


def get_multi_instrument_sample_count(config: MultiInstrumentDatasetConfig) -> int:
    """
    Get total sample count across all instruments without creating the full generator.
    
    Args:
        config: Multi-instrument dataset configuration
        
    Returns:
        Total number of training samples
    """
    generator = MultiInstrumentDataGenerator(config)
    return generator.get_total_training_samples()


# Example usage
if __name__ == "__main__":
    # Define instruments
    instruments = [
        InstrumentConfig(
            name="GBPUSD",
            hourly_data_path="/path/to/GBPUSD_hour.csv",
            chunked_data_dir="/path/to/chunked_GBPUSD_five_minute_dir/"
        ),
        InstrumentConfig(
            name="EURUSD", 
            hourly_data_path="/path/to/EURUSD_hour.csv",
            chunked_data_dir="/path/to/chunked_EURUSD_five_minute_dir/"
        ),
        # Add more instruments as needed
    ]
    
    # Create configuration
    config = MultiInstrumentDatasetConfig(
        instruments=instruments,
        main_lookback_tokens=100,
        hourly_lookback_tokens=24,
        lookback_window=1440,
        batch_size=32,
        apply_smote=True,
        shuffle_data=True,
        feature_columns=['open', 'high', 'low', 'close'],
        max_chunks_per_instrument=10  # Memory management per instrument
    )
    
    # Create dataset
    try:
        train_dataset, total_samples = create_multi_instrument_dataset(
            config=config,
            repeat_dataset=True
        )
        
        logger.info(f"Dataset created successfully with {total_samples} total samples")
        
        # Test the dataset
        for batch in train_dataset.take(1):
            (main_input, hourly_input), targets = batch
            logger.info(f"Batch shapes - Main: {main_input.shape}, Hourly: {hourly_input.shape}, Targets: {targets.shape}")
            break
            
    except Exception as e:
        logger.error(f"Failed to create multi-instrument dataset: {e}")
        raise