"""Hyperparameters for the whole pipeline.

Constants prefixed ``R_`` belong to the active regression model. Unprefixed
model constants (``D_MODEL``, ``FF_DIM``, ``NUM_HEADS``) and the labelling
thresholds below them are used only by the legacy classification pipeline in
``legacy/classification`` — they are kept because that code still imports them.
"""

# --- Normalisation -----------------------------------------------------------

#: Candles of trailing history used to compute the min/max normalisation window.
#: 6*24 = 144, i.e. six days of hourly candles. Every price is rescaled against
#: the high/low range of this window, which is what lets one model see GOLD# and
#: EURUSD# as the same kind of object.
NORMALIZING_WINDOW_SIZE = 6 * 24

# --- Labelling ---------------------------------------------------------------

#: Forward 5-minute candles used to build regression labels. 12 candles = 1 hour,
#: so ``target_high``/``target_low`` describe the price range of the next hour.
REGRESSION_LABELING_WINDOW_SIZE = 12

#: Forward window for the legacy classification labeller.
LABELING_WINDOW_SIZE = 18

# --- Sequence shape ----------------------------------------------------------

NUM_TOKENS = 64        #: 5-minute candles per sample (~5.3 hours of history).
OTHER_TOKENS = 64      #: Hourly candles per sample (~2.7 days of history).
LOOKBACK_WINDOW = NUM_TOKENS + 1  #: Rows to slice per sample; +1 for the partial hour.

#: Normalised OHLC columns fed to the model, in channel order.
FEATURES = ['open_normalized', 'high_normalized', 'low_normalized', 'close_normalized']

# --- Training ----------------------------------------------------------------

BATCH_SIZE = 64
LR = 4e-5          #: Peak learning rate for training from scratch.
WARMUP = 800000    #: Warmup steps (used when a step-based schedule is set manually).
DECAY = 400000 * 40

FTLR = 2e-5        #: Fine-tuning peak learning rate.
FTWARMUP = 3000 * 5
FTDECAY = 3000 * 40

# --- Regression model geometry -----------------------------------------------

R_D_MODEL = 256                    #: Hidden width.
R_FF_DIM = R_D_MODEL * 4           #: Feed-forward width inside each block.
R_NUM_HEADS = R_D_MODEL // 32      #: 8 attention heads.

# --- Legacy classification model geometry ------------------------------------

D_MODEL = 768
FF_DIM = D_MODEL * 4
NUM_HEADS = D_MODEL // 24

# --- Legacy classification labelling thresholds ------------------------------

POSITIVE_SLOPE = 0.3               #: Lower bound of the normalised "buy zone".
NEGATIVE_SLOPE = 0.7               #: Upper bound of the normalised "buy zone".
STARTING_HOUR = 0
ENDING_HOUR = 25
LABEL_CUR_CANDLE_MULTIPLIER = 0
LABEL_MEAN_MULTIPLIER = 6
DRAWDOWN = 0.5                     #: Stop-loss distance in mean-candle units.
LABEL_LOOKBACK = 64                #: Candles used to estimate mean candle size.
