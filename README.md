# Multi-Resolution Transformer

A transformer that forecasts short-horizon price ranges by attending across
multiple timeframes in a single token stream.

Given the last 64 five-minute candles and the last 64 hourly candles for an
instrument, the model predicts two numbers: the highest and the lowest price
reached over the next 12 five-minute candles. Together those bound the expected
**price range of the next hour**.

Financial data is the application here, not the point. The transferable parts —
multi-resolution fusion, window normalisation, leakage-safe splitting of
overlapping windows, and baselining against a naive predictor — apply to any
irregular, non-stationary, multi-scale time series.

> **Status: research project, negative result.** The model trains stably and
> learns real structure, but it does not reliably beat a random-walk baseline on
> out-of-sample data. It is published as a worked example of building a
> multi-resolution time-series pipeline end to end — normalisation, leakage-safe
> splitting, augmentation, and honest baselining — not as a trading system. See
> [Results](#results).

---

## The problem this is built around

Financial time series break most of the assumptions that make sequence modelling
easy, and nearly all of the design here is a response to one of them:

| Problem | Response |
|---|---|
| Price levels are not comparable — GBPUSD trades near 1.27, GOLD near 2000 | Window normalisation to dimensionless ratios |
| A 5-minute chart alone has no macro context | Two resolutions fed to one attention stack |
| The current hour is incomplete at prediction time | An explicit "partial hour" input |
| Adjacent windows overlap almost entirely, so naive splits leak | Chunked splitting with a lookback margin |
| Non-stationarity makes models memorise regimes | Gradient-scaled Gaussian augmentation |
| "Good MAE" is easy to fake on normalised targets | Every metric reported against a random-walk baseline |

---

## Core idea: marrying resolutions in one attention stack

The obvious way to combine timeframes is to encode each separately and
concatenate the summaries. This does something different: **all resolutions are
concatenated along the sequence axis into a single token stream**, so one
transformer stack attends *across* timeframes — a 5-minute token can attend
directly to an hourly token from two days ago.

```
                                    each token tagged with a learned
                                    type embedding (5min / hourly / partial)
                                              │
5-min OHLC   (64 × 4) ─► CNN ─► +position ─► [type 0] ─┐
hourly OHLC  (64 × 4) ─► CNN ─► +position ─► [type 1] ─┤
partial hour (1 × 4)  ─► Dense ───────────► [type 2] ─┤
minutes-into-hour ─┐                                   │
                   ├─► cyclical encoding ──────────────┘
partial-hour-len ──┘                                   │
                                                       ▼
                              concatenate along sequence axis (129 tokens)
                                                       │
                                    4 × gated transformer blocks
                                       (stochastic depth 0.05→0.2)
                                                       │
                              avg-pool ‖ max-pool ‖ attention-pool
                                                       │
                                              shared dense (256)
                                                   ╱        ╲
                                          target_high      target_low
```

Two details make the merge work:

- **Type embeddings** — without them the stack cannot tell a 5-minute candle from
  an hourly one, since both are just 4 normalised floats. A learned 16-dim
  embedding is concatenated onto every token identifying its resolution.
- **Learned branch scalars** (`ScalarScale`) — one trainable scalar per branch, so
  the model decides for itself how loudly each resolution speaks rather than
  having the ratio fixed by architecture.

~4.4M parameters at the default `R_D_MODEL = 256`.

### The partial hour

At 14:25, the 14:00 hourly candle does not exist yet. Most setups either wait for
the hour to close (throwing away 55 minutes of information) or leak the completed
candle backwards (invalidating the result).

Instead, a third input carries the **incomplete current hour** — its running
open/high/low/close so far — normalised on the *hourly* scale so it is
directly comparable to the closed hourly candles beside it. Two scalars tell the
model how much to trust it: `minutes_into_hour` (cyclically encoded) and
`partial_hour_length`. A partial hour that is 5 minutes old and one that is 55
minutes old are very different objects, and the model is told which it is looking at.

### Normalisation

Every price is rescaled against the high/low range of the preceding
`NORMALIZING_WINDOW_SIZE = 144` candles (six days of hourly data):

```
x_normalized = (x - window_min) / (window_max - window_min)
```

This is what lets one model see every instrument. The raw `close` and the window
bounds are carried through preprocessing into the chunk CSVs, so predictions can
be converted back to real prices at evaluation time — **normalised error is not
a meaningful number on its own**, because its scale depends on how volatile the
window happened to be.

Labels use the *previous* row's window bounds (`_normalized_for_label`), so a
label never depends on a window that includes the future it is describing.

### Leakage-safe splitting

Consecutive samples share 63 of 64 candles, so a shuffled split puts near-copies
of training rows into the test set and produces beautiful, meaningless scores.

`split_multiresolution_chunks` splits the series into large contiguous chunks and
splits train/val/test *within* each chunk in time order. It additionally walks
back through the hourly index to find the earliest hour any sample in a chunk
could reach via its 64-hour lookback, and trims the chunk start so that lookback
cannot cross into the previous chunk. That second step is the one that is easy to
forget when two resolutions are involved.

### Augmentation

`add_gaussian_noise` applies noise on a **linear gradient across the sequence**:
the oldest candle gets the full standard deviation, the most recent gets none.
The reasoning is that distant history should be a blurry regime signal while the
recent candles — the ones the label actually depends on — must stay sharp.

`regression/best_noise.py` grid-searches the noise parameters with checkpointing
after every configuration, so an interrupted search resumes where it stopped.

---

## Results

Honest reporting is the point of this section.

The augmentation search (`results/noise_search.csv`, SILVER#, validation loss):

| noise_std | noise_prob | best val_loss | val MAE |
|---|---|---|---|
| 0.0 (none) | 0.0 | 9.826 | 4.211 |
| **0.001** | **0.2** | **9.712** | **3.940** |
| 0.001 | 0.7 | 9.838 | 3.975 |
| 0.001 | 0.9 | 9.862 | 3.592 |

Augmentation helps, but by ~1%. That is the honest size of the effect — not
nothing, not a breakthrough. The search was cut short of the full 20-cell grid.

**Against the random-walk baseline**, the model does not establish a durable
edge. `python -m regression.test` reports MAE and MSE for both heads in
normalised *and* denormalised space, next to the naive "the future high equals
the current close" predictor. The denormalised comparison is the one to read.
Improvements that appear in normalised space frequently vanish once converted
back to real prices, which is exactly why both are printed.

What did work: training is stable (no NaN/divergence across long runs, with
gradient monitoring to confirm), the multi-resolution merge trains without one
branch collapsing, and the pipeline handles 19 instruments of tick-derived data
without leakage. What did not: turning any of that into out-of-sample predictive
power good enough to act on.

---

## Repository layout

```
constants/global_constants.py   All hyperparameters, documented inline
core/
  working_data.py               Cleaning, normalisation, labelling, chunk splitting
  data_generator.py             Multi-instrument TF dataset with N-resolution support
  modeler.py                    Keras model definition
  transformer_builder.py        Custom layers + LR schedule
  gradient_monitor.py           NaN / gradient-norm / branch-scale callbacks
regression/
  preprocess.py                 Raw CSVs → normalised, labelled chunks
  datasets.py                   Single source of truth for dataset construction
  trainer.py                    Multi-instrument training
  fine_tuner.py                 Per-instrument fine-tuning from a base checkpoint
  test.py                       Evaluation vs random-walk baseline
  get_mae.py                    Standalone random-walk baseline
  best_noise.py                 Resumable augmentation grid search
  predictor.py                  Walk-forward inference on historical CSVs
  smoke_test.py                 One-minute end-to-end check
  losses.py                     Asymmetric losses and trading metrics
legacy/                         Superseded classification pipeline — see legacy/README.md
results/                        Committed experiment outputs
```

### Custom components worth a look

| Component | File | What it does |
|---|---|---|
| `StochasticGatedTransformerBlock` | `core/transformer_builder.py` | Transformer block with sigmoid gates on both attention and FFN outputs, stochastic depth, and training-time noise injected into gates and activations |
| `AddTypeEmbedding` | `core/transformer_builder.py` | Tags each token with its source resolution |
| `AttentionPooling` | `core/transformer_builder.py` | Learned soft aggregation over the sequence, pooled alongside avg and max |
| `TemporalPreservingDropout` | `core/modeler.py` | Dropout that never masks the most recent N timesteps |
| `ScalarScale` | `core/modeler.py` | One trainable scalar per branch for resolution weighting |
| `WarmupCosineDecay` | `core/transformer_builder.py` | Linear warmup into cosine decay |
| `InstrumentChunkManager` | `core/data_generator.py` | LRU-cached chunk loading so multi-instrument data need not fit in memory |

---

## Setup

Requires Python ≥ 3.10 and TensorFlow 2.16.

```bash
git clone https://github.com/deetailed/multiresolution-transformer
cd multiresolution-transformer
python -m venv .venv && source .venv/bin/activate
pip install -e .          # add [viz] for the search heatmap
```

Developed on Python 3.10 / TF 2.16.1 / Keras 3.5 with CUDA. It runs on CPU, slowly.

**Run every script from the project root** — they resolve `data/`, `models/` and
`results/` relative to the working directory.

### Input data

Not included (tens of GB, broker-licensed). Supply your own OHLC exports as:

```
data/final_data/{INSTRUMENT}/five_minutes.csv
data/final_data/{INSTRUMENT}/hours.csv
```

Both need the columns MetaTrader 5 exports:

```csv
time,open,high,low,close,tick_volume,spread,real_volume
978309300,1.4931,1.4931,1.4931,1.4931,1,0,0
```

`time` is a Unix timestamp; only `time` and OHLC are used. Any instrument works —
the normalisation makes the scale irrelevant. Gaps, duplicates and off-grid rows
are handled by the cleaning stage.

---

## Usage

```bash
# 1. Preprocess raw CSVs into normalised, labelled, leakage-safe chunks.
#    Slow — this is the expensive step. Writes data/experimenting/.
python -m regression.preprocess
python -m regression.preprocess --instruments GBPUSD# EURUSD#   # or a subset

# 2. Confirm the pipeline is wired up (~1 minute; do this before a long run).
python -m regression.smoke_test

# 3. Train across all preprocessed instruments. Saves models/regressor.keras.
python -m regression.trainer
python -m regression.trainer --epochs 20 --patience 5

# 4. Evaluate against the random-walk baseline.
python -m regression.test --split testing

# 5. Optional: fine-tune the base model on one instrument.
python -m regression.fine_tuner --instrument USDJPY#

# 6. Optional: re-tune augmentation (resumable).
python -m regression.best_noise --instruments SILVER#

# 7. Walk-forward inference over historical CSVs.
python -m regression.predictor --instrument EURUSD# --data-dir data/checking_data
```

Standalone baseline, no model required:

```bash
python -m regression.get_mae --instruments GBPUSD#
```

### Key hyperparameters

All in `constants/global_constants.py`, documented inline.

| Constant | Default | Meaning |
|---|---|---|
| `NORMALIZING_WINDOW_SIZE` | 144 | Candles in the normalisation window (6 days hourly) |
| `REGRESSION_LABELING_WINDOW_SIZE` | 12 | Forward candles defining the label (1 hour) |
| `NUM_TOKENS` | 64 | 5-minute lookback |
| `OTHER_TOKENS` | 64 | Hourly lookback |
| `R_D_MODEL` | 256 | Hidden width |
| `R_NUM_HEADS` | 8 | Attention heads |
| `R_FF_DIM` | 1024 | Feed-forward width |
| `BATCH_SIZE` | 64 | Batch size |

---

## Extending to more resolutions

`core/data_generator.py` already generalises past two timeframes. Any number of
coarser resolutions can be declared per instrument:

```python
from core.data_generator import InstrumentConfig, SecondaryResolution

InstrumentConfig(
    name="GBPUSD#",
    chunked_data_dir="data/experimenting/GBPUSD#/training",
    secondary_resolutions=[
        SecondaryResolution(name="hourly", data_path=".../hour.csv", lookback_tokens=64),
        SecondaryResolution(name="four_hour", data_path=".../four_hour.csv", lookback_tokens=32),
        SecondaryResolution(name="daily", data_path=".../daily.csv", lookback_tokens=16),
    ],
)
```

Each resolution is matched to the 5-minute timestamp by binary search, with a
per-instrument time threshold ensuring every resolution has a full lookback
window before a sample is emitted. Passing `hourly_data_path=` instead keeps the
original single-resolution behaviour.

**This is where the work stopped.** The generator emits N resolutions, but
`create_regression_model` still builds exactly one secondary branch, so anything
beyond hourly currently has no consumer. Adding one means looping the
CNN → positional → type-embedding branch over the resolution list and giving each
its own `type_id`. That is the natural next step for anyone picking this up.

---

## Known limitations

- The model does not beat a random walk out-of-sample. Treat any published number
  as a description of the pipeline, not of predictive power.
- Preprocessing is single-threaded pandas and slow on multi-year datasets.
- `regression/losses.py` contains several loss and metric variants tried during
  development. `trainer.py` uses plain MSE; `fine_tuner.py` and `best_noise.py`
  use the asymmetric Huber. The rest are kept as a record of what was explored.
- `regression/predictor.py` recomputes normalisation over the whole history for
  every prediction point, which is correct but O(n²)-ish and unsuited to live use.
- The entry filters in `predictor.py` (session hours, bullish structure) are
  inherited from the classification-era labelling assumptions and are illustrative
  only.

## License

MIT — see `pyproject.toml`. Note that `legacy/memd.py` is third-party code with
its own attribution header.
