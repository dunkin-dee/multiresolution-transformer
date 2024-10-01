import os
import pandas as pd
from process_data import clean_cols, clean_non_minute_rows, normalize_by_window, label_df


if __name__ == "__main__":

    if not os.path.exists(f'training'):
        os.mkdir('training')
    
    counter = 1

    dirs = os.listdir('data')
    for dir in dirs:
        if not os.path.exists(f'training/{dir}'):
            os.mkdir(f'training/{dir}')
        read_path = f'data/{dir}/minutes.csv'
        write_path = f'training/{dir}/minutes.csv'
        
        if os.path.exists(write_path):
            continue

        print(f'Working on {dir} - {counter} of {len(dirs)}')
        df = pd.read_csv(read_path)

        df = clean_cols(df)
        df = clean_non_minute_rows(df)

        window_size = 60*3

        print("Normalizing...")
        df = normalize_by_window(df, window_size=window_size)

        print("Labeling...")
        df = label_df(df, multiplier=4)

        df.to_csv(write_path, index=False)

        print("\n")

        counter += 1

