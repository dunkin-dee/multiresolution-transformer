"""Single source of truth for building the regression train/val/test datasets.

Every regression entry point (``trainer``, ``test``, ``fine_tuner``, ``best_noise``)
used to carry its own ~90-line copy of this logic. They now all call
:func:`get_datasets_and_steps`.

Typical use::

    from regression.datasets import get_datasets_and_steps

    (train, val, test), (train_steps, val_steps, test_steps) = get_datasets_and_steps(
        working_path="data/experimenting"
    )
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import tensorflow as tf

from constants.global_constants import (
    BATCH_SIZE,
    FEATURES,
    LOOKBACK_WINDOW,
    NUM_TOKENS,
    OTHER_TOKENS,
)
from core.data_generator import (
    InstrumentConfig,
    MultiInstrumentDatasetConfig,
    create_multi_instrument_dataset,
    strip_eval_targets,
)

#: Default location of preprocessed chunks produced by ``regression.preprocess``.
DEFAULT_WORKING_PATH = "data/experimenting"


@dataclass
class NoiseConfig:
    """Gaussian augmentation applied to the training split only.

    Noise is scaled by a linear gradient across the sequence: the oldest candle
    receives the full ``std``, the most recent receives ``std * min_noise_factor``
    (0 by default). The intent is to blur distant history while leaving the
    immediate past — the part the label actually depends on — intact.
    """

    std_5min: float = 0.01
    std_hourly: float = 0.015
    probability_5min: float = 0.8
    probability_hourly: float = 0.8


#: Augmentation used by ``regression.trainer``. The grid search in
#: ``regression.best_noise`` explored this space; see ``results/noise_search.csv``.
DEFAULT_TRAIN_NOISE = NoiseConfig()


def _instrument_configs(
    instruments: Sequence[str], working_path: str, split: str
) -> List[InstrumentConfig]:
    """Build one :class:`InstrumentConfig` per instrument for a given split."""
    return [
        InstrumentConfig(
            name=instrument,
            hourly_data_path=f"{working_path}/{instrument}/hour.csv",
            chunked_data_dir=f"{working_path}/{instrument}/{split}",
        )
        for instrument in instruments
    ]


def _dataset_config(
    instruments: Sequence[str],
    working_path: str,
    split: str,
    feature_cols: Sequence[str],
    batch_size: int,
    max_chunks_per_instrument: int,
    shuffle: bool,
    noise: Optional[NoiseConfig],
) -> MultiInstrumentDatasetConfig:
    """Assemble the generator config for a single split."""
    kwargs = dict(
        instruments=_instrument_configs(instruments, working_path, split),
        main_lookback_tokens=NUM_TOKENS,
        hourly_lookback_tokens=OTHER_TOKENS,
        lookback_window=LOOKBACK_WINDOW,
        batch_size=batch_size,
        shuffle_data=shuffle,
        feature_columns=list(feature_cols),
        max_chunks_per_instrument=max_chunks_per_instrument,
    )

    if noise is not None:
        kwargs.update(
            add_noise_5min=True,
            add_noise_hourly=True,
            noise_std_5min=noise.std_5min,
            noise_std_hourly=noise.std_hourly,
            noise_probability_5min=noise.probability_5min,
            noise_probability_hourly=noise.probability_hourly,
        )

    return MultiInstrumentDatasetConfig(**kwargs)


def discover_instruments(working_path: str = DEFAULT_WORKING_PATH) -> List[str]:
    """List instrument directories under ``working_path``.

    Raises:
        FileNotFoundError: if ``working_path`` does not exist — usually means
            ``python -m regression.preprocess`` has not been run yet.
    """
    if not os.path.isdir(working_path):
        raise FileNotFoundError(
            f"No preprocessed data at '{working_path}'. "
            f"Run 'python -m regression.preprocess' first."
        )
    instruments = sorted(
        d for d in os.listdir(working_path)
        if os.path.isdir(os.path.join(working_path, d))
    )
    if not instruments:
        raise FileNotFoundError(f"'{working_path}' contains no instrument directories.")
    return instruments


def get_datasets_and_steps(
    working_path: str = DEFAULT_WORKING_PATH,
    instruments: Optional[Sequence[str]] = None,
    feature_cols: Sequence[str] = FEATURES,
    batch_size: int = BATCH_SIZE,
    max_chunks_per_instrument: int = 25,
    train_noise: Optional[NoiseConfig] = DEFAULT_TRAIN_NOISE,
    strip_for_fit: bool = True,
) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], Tuple[int, int, int]]:
    """Build the train/validation/test datasets and their steps-per-epoch.

    Args:
        working_path: Directory of preprocessed chunks, one subdirectory per
            instrument, each containing ``hour.csv`` plus ``training/``,
            ``validation/`` and ``testing/``.
        instruments: Instrument names to include. Defaults to every instrument
            directory found under ``working_path``.
        feature_cols: Normalised OHLC column names fed to the model.
        batch_size: Samples per batch.
        max_chunks_per_instrument: LRU cache size for chunk CSVs, per instrument.
        train_noise: Augmentation for the training split. Pass ``None`` to disable.
            Validation and test are never augmented.
        strip_for_fit: When True, drop the evaluation-only target keys
            (``norm_min``, ``norm_max``, ``original_close``) from the train and
            validation datasets. Keras raises ``KeyError`` on target keys that do
            not correspond to a model output, so this must stay True for
            ``model.fit``. Set it False when you need those keys to denormalise
            predictions, as ``regression.test`` does.

    Returns:
        ``((train, val, test), (train_steps, val_steps, test_steps))``. The test
        dataset always retains the evaluation-only keys.
    """
    if instruments is None:
        instruments = discover_instruments(working_path)

    configs = {
        "training": _dataset_config(
            instruments, working_path, "training", feature_cols, batch_size,
            max_chunks_per_instrument, shuffle=True, noise=train_noise,
        ),
        "validation": _dataset_config(
            instruments, working_path, "validation", feature_cols, batch_size,
            max_chunks_per_instrument, shuffle=False, noise=None,
        ),
        "testing": _dataset_config(
            instruments, working_path, "testing", feature_cols, batch_size,
            max_chunks_per_instrument, shuffle=False, noise=None,
        ),
    }

    datasets, rows = {}, {}
    for split, config in configs.items():
        datasets[split], rows[split] = create_multi_instrument_dataset(
            config=config, repeat_dataset=True
        )

    if strip_for_fit:
        datasets["training"] = strip_eval_targets(datasets["training"])
        datasets["validation"] = strip_eval_targets(datasets["validation"])

    return (
        (datasets["training"], datasets["validation"], datasets["testing"]),
        tuple(rows[s] // batch_size for s in ("training", "validation", "testing")),
    )
