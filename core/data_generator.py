import polars as pl
import numpy as np
import tensorflow as tf
import os
import logging
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SecondaryResolution:
    """Describes a single secondary (coarser) timeframe data source."""
    name: str                        # e.g. 'hourly', 'four_hour', 'daily'
    data_path: str                   # path to CSV (same column format as current hour.csv)
    lookback_tokens: int             # how many candles to extract per sample
    add_noise: bool = False
    noise_std: float = 0.01
    noise_probability: float = 1.0

    def __post_init__(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Secondary resolution '{self.name}' data not found: {self.data_path}"
            )
        if self.lookback_tokens <= 0:
            raise ValueError(f"lookback_tokens must be positive for resolution '{self.name}'")


@dataclass
class InstrumentConfig:
    """Configuration for a single instrument.

    Two APIs are supported:
    - Legacy (backward-compatible): pass hourly_data_path; secondary_resolutions will be
      auto-constructed in MultiInstrumentDatasetConfig.__post_init__ once lookback_tokens
      is known from the dataset config.
    - New: pass secondary_resolutions directly (a list of SecondaryResolution objects).
    """
    name: str
    chunked_data_dir: str
    secondary_resolutions: Optional[List[SecondaryResolution]] = None  # new API
    hourly_data_path: Optional[str] = None                             # legacy API

    def __post_init__(self):
        if not os.path.exists(self.chunked_data_dir):
            raise FileNotFoundError(
                f"Chunked data directory not found for {self.name}: {self.chunked_data_dir}"
            )
        if self.hourly_data_path is not None and not os.path.exists(self.hourly_data_path):
            raise FileNotFoundError(
                f"Hourly data not found for {self.name}: {self.hourly_data_path}"
            )
        if self.secondary_resolutions is None and self.hourly_data_path is None:
            raise ValueError(
                f"InstrumentConfig '{self.name}' requires either "
                "'secondary_resolutions' or 'hourly_data_path'"
            )


@dataclass
class MultiInstrumentDatasetConfig:
    """Configuration class for multi-instrument dataset parameters"""
    instruments: List[InstrumentConfig]
    main_lookback_tokens: int
    # Legacy field kept for backward compat. Ignored when secondary_resolutions is used.
    hourly_lookback_tokens: int = 64
    lookback_window: int = 1440
    batch_size: int = 32
    shuffle_data: bool = False
    feature_columns: List[str] = None
    max_chunks_per_instrument: int = 20

    add_noise_5min: bool = False
    # Legacy noise flags for hourly resolution. Ignored when secondary_resolutions is used.
    add_noise_hourly: bool = False
    noise_std_5min: float = 0.01
    noise_std_hourly: float = 0.01
    noise_probability_5min: float = 1.0
    noise_probability_hourly: float = 1.0

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ['open', 'high', 'low', 'close']

        if not self.instruments:
            raise ValueError("At least one instrument must be provided")
        if self.main_lookback_tokens <= 0:
            raise ValueError("main_lookback_tokens must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.noise_std_5min < 0 or self.noise_std_hourly < 0:
            raise ValueError("Noise standard deviations must be non-negative")
        if not 0.0 <= self.noise_probability_5min <= 1.0:
            raise ValueError("5-minute noise probability must be between 0.0 and 1.0")
        if not 0.0 <= self.noise_probability_hourly <= 1.0:
            raise ValueError("Hourly noise probability must be between 0.0 and 1.0")

        # Bridge legacy API: auto-construct SecondaryResolution objects here where
        # hourly_lookback_tokens is already known. Each call produces fresh objects,
        # so sharing InstrumentConfig across multiple dataset configs is safe.
        for instrument in self.instruments:
            if instrument.secondary_resolutions is None:
                instrument.secondary_resolutions = [
                    SecondaryResolution(
                        name='hourly',
                        data_path=instrument.hourly_data_path,
                        lookback_tokens=self.hourly_lookback_tokens,
                        add_noise=self.add_noise_hourly,
                        noise_std=self.noise_std_hourly,
                        noise_probability=self.noise_probability_hourly,
                    )
                ]
            elif self.add_noise_hourly:
                logger.warning(
                    "add_noise_hourly/noise_std_hourly are ignored when secondary_resolutions "
                    "is provided explicitly on InstrumentConfig. Set noise on each "
                    "SecondaryResolution object instead."
                )

        # Validate: all instruments must have the same resolution names in the same order
        # so that batch tensor positions are consistent across instruments.
        ref_names = [r.name for r in self.instruments[0].secondary_resolutions]
        for instr in self.instruments[1:]:
            names = [r.name for r in instr.secondary_resolutions]
            if names != ref_names:
                raise ValueError(
                    f"All instruments must have matching secondary_resolutions. "
                    f"Expected {ref_names}, got {names} for '{instr.name}'"
                )


def add_gaussian_noise(data: np.ndarray, noise_std: float, probability: float = 1.0,
                      clip_range: Tuple[float, float] = (0.0, 1.0),
                      gradient_axis: int = 0, min_noise_factor: float = 0.0) -> np.ndarray:
    """
    Add Gaussian noise to data with gradient intensity and optional probability and clipping.

    Args:
        data: Input data array
        noise_std: Standard deviation of Gaussian noise at maximum intensity
        probability: Probability of adding noise (0.0 to 1.0)
        clip_range: Range to clip values after adding noise
        gradient_axis: Axis along which to apply the noise gradient (0 for rows, 1 for columns)
        min_noise_factor: Minimum noise factor (0.0 = no noise at end, 1.0 = full noise at end)

    Returns:
        Data with gradient noise added (if applied)
    """
    if noise_std <= 0 or np.random.random() > probability:
        return data

    axis_length = data.shape[gradient_axis]
    gradient = np.linspace(1.0, min_noise_factor, axis_length)
    gradient_shape = [1] * data.ndim
    gradient_shape[gradient_axis] = axis_length
    gradient = gradient.reshape(gradient_shape)

    base_noise = np.random.normal(0, noise_std, data.shape).astype(data.dtype)
    scaled_noise = base_noise * gradient
    noisy_data = data + scaled_noise
    return np.clip(noisy_data, clip_range[0], clip_range[1])


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
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.chunk_cache[chunk_idx]

        chunk_path = os.path.join(self.instrument_config.chunked_data_dir, self.chunk_files[chunk_idx])
        required_cols = list(set(self.feature_columns + [
            'include', 'time', 'target_high', 'target_low',
            'partial_open_normalized', 'partial_high_normalized',
            'partial_low_normalized', 'partial_close_normalized',
            'position_in_hour', 'partial_hour_length', 'norm_window_min',
            'norm_window_max', 'close'
        ]))

        try:
            chunk_df = pl.scan_csv(chunk_path).select(required_cols).collect()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load chunk {chunk_idx} for {self.instrument_config.name} "
                f"({self.chunk_files[chunk_idx]}): {e}"
            )

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
        # N secondary DataFrames and time arrays, indexed in the same order as
        # instrument_config.secondary_resolutions.
        self.secondary_dfs: List[pl.DataFrame] = []
        self.secondary_time_arrays: List[np.ndarray] = []
        self._load_secondary_data()

        self.time_threshold = self._calculate_time_threshold()
        self.valid_indices: List[Tuple[int, int]] = []
        self._build_indices()

    def _load_secondary_data(self):
        """Load all secondary resolution DataFrames into memory."""
        for res in self.instrument_config.secondary_resolutions:
            logger.info(f"Loading '{res.name}' data for {self.instrument_config.name}...")
            try:
                cols = list(set(self.config.feature_columns + ['time']))
                df = pl.scan_csv(res.data_path).select(cols).collect()
                if df.shape[0] < res.lookback_tokens + 1:
                    raise ValueError(
                        f"Insufficient '{res.name}' data for {self.instrument_config.name}: "
                        f"need at least {res.lookback_tokens + 1} rows, got {df.shape[0]}"
                    )
                self.secondary_dfs.append(df)
                self.secondary_time_arrays.append(df['time'].to_numpy())
                logger.info(
                    f"Loaded '{res.name}' data for {self.instrument_config.name}: {df.shape[0]} rows"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load '{res.name}' data for {self.instrument_config.name}: {e}"
                )

    def _calculate_time_threshold(self) -> Optional[float]:
        """Calculate the earliest timestamp at which ALL secondary resolutions have a full
        lookback window. This is the maximum threshold across all resolutions."""
        threshold = None
        for i, res in enumerate(self.instrument_config.secondary_resolutions):
            df = self.secondary_dfs[i]
            if df.shape[0] > res.lookback_tokens:
                candidate = float(df[res.lookback_tokens, 'time'])
                threshold = candidate if threshold is None else max(threshold, candidate)
        if threshold is not None:
            logger.info(
                f"Applied time threshold for {self.instrument_config.name}: {threshold}"
            )
        return threshold

    def _build_indices(self):
        """Build indices for all valid samples for this instrument"""
        logger.info(f"Building indices for {self.instrument_config.name}...")

        for chunk_idx in range(self.chunk_manager.get_chunk_count()):
            chunk_df = self.chunk_manager.get_chunk(chunk_idx)

            start_row = self.config.lookback_window
            if self.time_threshold is not None:
                chunk_times = chunk_df['time'].to_numpy()
                threshold_idx = int(np.searchsorted(chunk_times, self.time_threshold, side='left'))
                start_row = max(start_row, threshold_idx)

            if start_row >= chunk_df.shape[0]:
                continue

            for row_idx in range(start_row, chunk_df.shape[0]):
                if chunk_df[row_idx, 'include'] == 1:
                    self.valid_indices.append((chunk_idx, row_idx))

        logger.info(
            f"Built indices for {self.instrument_config.name}: {len(self.valid_indices)} valid samples"
        )

    def extract_sample(self, chunk_idx: int, row_idx: int) -> Optional[Tuple]:
        """Extract a single sample.

        Returns:
            (main_input, secondary_sequences, target_values, partial_hour_data,
             minutes_into_hour, partial_hour_length, norm_min, norm_max, original_close)
            where secondary_sequences is a list of np.ndarray, one per secondary resolution.
        """
        try:
            chunk_df = self.chunk_manager.get_chunk(chunk_idx)

            # Extract main (primary) sequence
            sequence_start = row_idx - self.config.lookback_window + 1
            sequence_end = row_idx + 1
            main_sequence = chunk_df[sequence_start:sequence_end, self.config.feature_columns].to_numpy()
            main_input = main_sequence[-self.config.main_lookback_tokens:]

            if self.config.add_noise_5min:
                main_input = add_gaussian_noise(
                    main_input,
                    self.config.noise_std_5min,
                    self.config.noise_probability_5min
                )

            # Extract all secondary resolution sequences via binary search on timestamp
            current_timestamp = chunk_df[row_idx, 'time']
            secondary_sequences = []
            for i, res in enumerate(self.instrument_config.secondary_resolutions):
                position = (
                    np.searchsorted(self.secondary_time_arrays[i], current_timestamp, side='right') - 1
                )
                if position < res.lookback_tokens:
                    return None
                seq = self.secondary_dfs[i][
                    position - res.lookback_tokens : position,
                    self.config.feature_columns
                ].to_numpy()
                if res.add_noise:
                    seq = add_gaussian_noise(seq, res.noise_std, res.noise_probability)
                secondary_sequences.append(seq)

            # Targets
            target_values = np.array([
                chunk_df[row_idx, 'target_high'],
                chunk_df[row_idx, 'target_low']
            ], dtype=np.float32)

            # Partial hour data (semantically tied to 5min→hourly relationship)
            partial_hour_cols = [
                'partial_open_normalized', 'partial_high_normalized',
                'partial_low_normalized', 'partial_close_normalized'
            ]
            partial_hour_data = chunk_df[row_idx, partial_hour_cols].to_numpy().reshape(1, -1)

            minutes_into_hour = np.array([chunk_df[row_idx, 'position_in_hour']], dtype=np.float32)
            partial_hour_length = np.array([chunk_df[row_idx, 'partial_hour_length']], dtype=np.float32)
            norm_min = np.array([chunk_df[row_idx, 'norm_window_min']], dtype=np.float32)
            norm_max = np.array([chunk_df[row_idx, 'norm_window_max']], dtype=np.float32)
            original_close = np.array([chunk_df[row_idx, 'close']], dtype=np.float32)

            return (main_input, secondary_sequences, target_values, partial_hour_data,
                    minutes_into_hour, partial_hour_length, norm_min, norm_max, original_close)

        except Exception as e:
            logger.error(
                f"Error extracting sample for {self.instrument_config.name} "
                f"({chunk_idx}, {row_idx}): {e}"
            )
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this instrument"""
        return {
            'name': self.instrument_config.name,
            'total_chunks': self.chunk_manager.get_chunk_count(),
            'valid_samples': len(self.valid_indices),
            'secondary_data_rows': {
                res.name: self.secondary_dfs[i].shape[0]
                for i, res in enumerate(self.instrument_config.secondary_resolutions)
            },
            'time_threshold': self.time_threshold,
        }


class MultiInstrumentDataGenerator:
    """Main multi-instrument data generator class"""

    def __init__(self, config: MultiInstrumentDatasetConfig):
        self.config = config
        self.processors = [
            SingleInstrumentProcessor(instrument_config, config)
            for instrument_config in config.instruments
        ]
        self.global_indices: List[Tuple[int, int, int]] = []
        self._build_global_indices()
        self.loop_counter = 0

    def _build_global_indices(self):
        """Build global indices across all instruments"""
        logger.info("Building global indices across all instruments...")

        for instrument_idx, processor in enumerate(self.processors):
            for chunk_idx, row_idx in processor.valid_indices:
                self.global_indices.append((instrument_idx, chunk_idx, row_idx))

        logger.info(
            f"Global indices built: {len(self.global_indices)} total samples "
            f"across {len(self.processors)} instruments"
        )

        if len(self.global_indices) == 0:
            raise ValueError("No valid samples found across all instruments")

    def get_total_training_samples(self) -> int:
        return len(self.global_indices)

    def _make_batch(
        self,
        batch_main: np.ndarray,
        batch_secondary: List[np.ndarray],
        batch_partial: np.ndarray,
        batch_minutes: np.ndarray,
        batch_length: np.ndarray,
        batch_targets: np.ndarray,
        batch_norm_min: np.ndarray,
        batch_norm_max: np.ndarray,
        batch_original_close: np.ndarray,
        size: int,
    ) -> Tuple:
        """Convert pre-allocated numpy arrays into a batch tuple of TF tensors.

        Returns a tuple:
            (main, (sec_0, ..., sec_N), partial, minutes, length, targets_dict)
        where secondary resolutions are packed as a tuple.
        """
        secondary_tensors = tuple(
            tf.convert_to_tensor(b[:size], dtype=tf.float32)
            for b in batch_secondary
        )
        return (
            tf.convert_to_tensor(batch_main[:size], dtype=tf.float32),
            secondary_tensors,
            tf.convert_to_tensor(batch_partial[:size], dtype=tf.float32),
            tf.convert_to_tensor(batch_minutes[:size], dtype=tf.float32),
            tf.convert_to_tensor(batch_length[:size], dtype=tf.float32),
            {
                'target_high': tf.convert_to_tensor(batch_targets[:size, 0], dtype=tf.float32),
                'target_low': tf.convert_to_tensor(batch_targets[:size, 1], dtype=tf.float32),
                'norm_min': tf.convert_to_tensor(batch_norm_min[:size, 0], dtype=tf.float32),
                'norm_max': tf.convert_to_tensor(batch_norm_max[:size, 0], dtype=tf.float32),
                'original_close': tf.convert_to_tensor(batch_original_close[:size, 0], dtype=tf.float32),
            }
        )

    def generate_batches(self) -> Generator[Tuple, None, None]:
        """Generate batches of data from all instruments"""
        # Resolution list is the same for all instruments (validated at config construction).
        resolutions = self.config.instruments[0].secondary_resolutions
        n_features = len(self.config.feature_columns)
        batch_size = self.config.batch_size

        while True:
            working_indices = self.global_indices.copy()

            if self.config.shuffle_data:
                np.random.shuffle(working_indices)

            # Pre-allocate batch arrays. Shapes are fully known at this point.
            batch_main = np.empty(
                (batch_size, self.config.main_lookback_tokens, n_features), dtype=np.float32
            )
            batch_secondary = [
                np.empty((batch_size, res.lookback_tokens, n_features), dtype=np.float32)
                for res in resolutions
            ]
            batch_targets = np.empty((batch_size, 2), dtype=np.float32)
            batch_partial = np.empty((batch_size, 1, 4), dtype=np.float32)
            batch_minutes = np.empty((batch_size, 1), dtype=np.float32)
            batch_length = np.empty((batch_size, 1), dtype=np.float32)
            batch_norm_min = np.empty((batch_size, 1), dtype=np.float32)
            batch_norm_max = np.empty((batch_size, 1), dtype=np.float32)
            batch_original_close = np.empty((batch_size, 1), dtype=np.float32)

            batch_idx = 0

            for instrument_idx, chunk_idx, row_idx in working_indices:
                try:
                    processor = self.processors[instrument_idx]
                    sample_data = processor.extract_sample(chunk_idx, row_idx)

                    if sample_data is None:
                        continue

                    (main_seq, secondary_seqs, targets, partial_data,
                     minutes, length, norm_min, norm_max, original_close) = sample_data

                    batch_main[batch_idx] = main_seq
                    for i, seq in enumerate(secondary_seqs):
                        batch_secondary[i][batch_idx] = seq
                    batch_targets[batch_idx] = targets
                    batch_partial[batch_idx] = partial_data
                    batch_minutes[batch_idx] = minutes
                    batch_length[batch_idx] = length
                    batch_norm_min[batch_idx] = norm_min
                    batch_norm_max[batch_idx] = norm_max
                    batch_original_close[batch_idx] = original_close
                    batch_idx += 1

                    if batch_idx == batch_size:
                        yield self._make_batch(
                            batch_main, batch_secondary, batch_partial,
                            batch_minutes, batch_length, batch_targets,
                            batch_norm_min, batch_norm_max, batch_original_close,
                            size=batch_size,
                        )
                        batch_idx = 0

                except Exception as e:
                    instrument_name = self.config.instruments[instrument_idx].name
                    logger.warning(
                        f"Skipping sample from {instrument_name} ({chunk_idx}, {row_idx}): {e}"
                    )
                    continue

            if batch_idx > 0:
                yield self._make_batch(
                    batch_main, batch_secondary, batch_partial,
                    batch_minutes, batch_length, batch_targets,
                    batch_norm_min, batch_norm_max, batch_original_close,
                    size=batch_idx,
                )

            self.loop_counter += 1
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


def create_multi_instrument_dataset(
    config: MultiInstrumentDatasetConfig,
    repeat_dataset: bool = False
) -> Tuple[tf.data.Dataset, int]:
    """Create TensorFlow dataset from multi-instrument data generator."""
    generator = MultiInstrumentDataGenerator(config)

    info = generator.get_dataset_info()
    logger.info("=== Multi-Instrument Regression Dataset Info ===")
    logger.info(f"Total instruments: {info['total_instruments']}")
    for instrument_info in info['instruments']:
        logger.info(
            f"  {instrument_info['name']}: {instrument_info['valid_samples']} samples, "
            f"{instrument_info['total_chunks']} chunks"
        )
    logger.info(f"Global total: {info['total_samples']} samples")
    logger.info(f"Target columns: {info['target_columns']}")
    logger.info(f"Shuffle: {info['shuffle_enabled']}")
    logger.info("=====================================")

    resolutions = config.instruments[0].secondary_resolutions
    n_features = len(config.feature_columns)

    # output_signature uses tuples (not lists) for secondary — TF tf.nest requires tuples
    # for variable-length structured sequences.
    secondary_signatures = tuple(
        tf.TensorSpec(shape=(None, res.lookback_tokens, n_features), dtype=tf.float32)
        for res in resolutions
    )

    dataset = tf.data.Dataset.from_generator(
        generator.generate_batches,
        output_signature=(
            tf.TensorSpec(shape=(None, config.main_lookback_tokens, n_features), dtype=tf.float32),
            secondary_signatures,
            tf.TensorSpec(shape=(None, 1, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            {
                'target_high': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'target_low': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'norm_min': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'norm_max': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'original_close': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            }
        )
    )

    n_resolutions = len(resolutions)

    if n_resolutions == 1:
        # Legacy path: produce the exact same 5-tuple the model already expects.
        # secondary is a 1-element tuple, so secondary[0] is the hourly tensor.
        dataset = dataset.map(
            lambda main, secondary, partial, minutes, length, targets:
                ((main, secondary[0], partial, minutes, length), targets)
        )
    else:
        # N-resolution path: flatten into (main, sec_0, ..., sec_N-1, partial, minutes, length).
        # build a concrete fixed-arity lambda via a closure so TF can trace it correctly.
        # range(n) is evaluated at Python time; secondary[i] indexing is TF-safe.
        def _make_map_fn(n: int):
            def map_fn(main, secondary, partial, minutes, length, targets):
                inputs = (main,) + tuple(secondary[i] for i in range(n)) + (partial, minutes, length)
                return inputs, targets
            return map_fn

        dataset = dataset.map(_make_map_fn(n_resolutions))

    if repeat_dataset:
        dataset = dataset.repeat()

    return dataset, generator.get_total_training_samples()


def get_multi_instrument_sample_count(config: MultiInstrumentDatasetConfig) -> int:
    """Get total sample count across all instruments."""
    generator = MultiInstrumentDataGenerator(config)
    return generator.get_total_training_samples()


# Example usage
if __name__ == "__main__":
    # Legacy API (unchanged call site):
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
    ]

    config = MultiInstrumentDatasetConfig(
        instruments=instruments,
        main_lookback_tokens=64,
        hourly_lookback_tokens=64,
        lookback_window=65,
        batch_size=64,
        shuffle_data=True,
        feature_columns=['open_normalized', 'high_normalized', 'low_normalized', 'close_normalized'],
        max_chunks_per_instrument=25,
    )

    try:
        train_dataset, total_samples = create_multi_instrument_dataset(
            config=config,
            repeat_dataset=True
        )
        logger.info(f"Dataset created successfully with {total_samples} total samples")
    except Exception as e:
        logger.error(f"Failed to create multi-instrument dataset: {e}")
        raise
