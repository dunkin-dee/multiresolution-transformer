import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Directory containing folders with CSV files
data_folder = 'training'

cols = [
    'open_normalized',
    'high_normalized',
    'low_normalized',
    'close_normalized',
    'target'
    ]

cols = cols[:5]

# Iterate through each subfolder and CSV file
for folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder)
    
    if os.path.isdir(folder_path):
        # Assume only one CSV file per folder
       
        csv_path = os.path.join(folder_path, 'minutes.csv')
        if os.path.exists(csv_path):
            print(csv_path)

            # Load the CSV file
            df = pd.read_csv(csv_path, usecols=cols)
            
            # Split into training (70%), validation (15%), and testing (15%) sets
            train_data, temp_data = train_test_split(df, test_size=0.30, shuffle=False)  # No shuffling for time-series data
            val_data, test_data = train_test_split(temp_data, test_size=0.50, shuffle=False)
            
            # Save the splits
            train_data.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
            val_data.to_csv(os.path.join(folder_path, 'val.csv'), index=False)
            test_data.to_csv(os.path.join(folder_path, 'test.csv'), index=False)
