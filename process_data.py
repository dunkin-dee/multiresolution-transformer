import pandas as pd

def clean_cols(df):
    clear_cols = ['tick_volume', 'spread', 'real_volume']
    df.drop(columns=clear_cols, inplace=True)
    return df

def clean_non_minute_rows(df):
    cut_off = df.index[df['time'].shift(-1) - df['time'] == 60].min()
    return df[cut_off:]

def normalize_by_window(df, window_size=1440, chunk_size=20):
    col_numbers = range(1, window_size+1)
    col_split = [col_numbers[i:i + chunk_size] for i in range(0, len(col_numbers), chunk_size)]
    df['window_min'] = df['low']
    df['window_max'] = df['high']

    for col_range in col_split:
        high_cols = []
        low_cols = []
        
        for x in col_range:
            high_col_name = f"high-{x}"
            df[high_col_name] = df['high'].shift(x)
            high_cols.append(high_col_name)
        high_cols_inclusive = high_cols + ['window_max']
        df['window_max'] = df[high_cols_inclusive].max(axis=1)
        df.drop(columns=high_cols, inplace=True)

        for x in col_range:
            low_col_name = f"low-{x}"
            df[low_col_name] = df['low'].shift(x)
            low_cols.append(low_col_name)
        low_cols_inclusive = low_cols + ['window_min']
        df['window_min'] = df[low_cols_inclusive].min(axis=1)
        df.drop(columns=low_cols, inplace=True)


    df['open_normalized'] = (df['open'] - df['window_min'])/(df['window_max'] - df['window_min'])
    df['high_normalized'] = (df['high'] - df['window_min'])/(df['window_max'] - df['window_min'])
    df['low_normalized'] = (df['low'] - df['window_min'])/(df['window_max'] - df['window_min'])
    df['close_normalized'] = (df['close'] - df['window_min'])/(df['window_max'] - df['window_min'])

    df.drop(columns=['window_max', 'window_min'], inplace=True)
    return df[window_size:]

def label_df(df, window_size=20, multiplier=4, positive_slope=0.4, candle_multiplier=2):
    df['candle'] = df['close_normalized'] - df['open_normalized']
    mean_candle = df['candle'].abs().mean()
    future_cols = []
    sum_cols = []
    sum_cum = 1
    for x in range(-1, -window_size-1, -1):
        col_name = f'candle-{abs(x)}'
        sum_name = f'sum-{sum_cum}'
        df[col_name] = df['candle'].shift(x)
        future_cols.append(col_name)
        df[sum_name] = df[future_cols].sum(axis=1)
        sum_cols.append(sum_name)
        sum_cum += 1

    df['target'] = 0

    df['prev_candle'] = df['candle'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)

    mask = (df[sum_cols].max(axis=1) > mean_candle*multiplier) & (df['close_normalized']>positive_slope) & ((df['candle']>mean_candle*candle_multiplier) | (df['prev_candle'] > 0) | (df['close'] > df[['prev_close', 'prev_open']].max(axis=1))) & (df['candle'] > 0)
    df.loc[mask, 'target'] = 1


    mask_down = (df[sum_cols].min(axis=1) < -mean_candle*multiplier) & (df['close_normalized']<1-positive_slope) & ((df['candle']<-mean_candle*candle_multiplier) | (df['prev_candle'] < 0) | (df['close'] < df[['prev_close', 'prev_open']].min(axis=1))) & (df['candle'] < 0)
    df.loc[mask_down, 'target'] = 2

    drop_cols = future_cols + sum_cols + ['candle', 'prev_candle', 'prev_close', 'prev_open']
    df.drop(columns=drop_cols, inplace=True)

    return df[1:-window_size]