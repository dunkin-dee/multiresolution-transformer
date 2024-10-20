import polars as pl
import tensorflow as tf
import numpy as np


def get_total_rows(file_path, num_tokens, smote=False):
    # Count the total number of rows in the CSV file
    df_lazy = pl.scan_csv(file_path)
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]
    if not smote:
        return total_rows
    df_collected = df_collected.with_columns(pl.arange(0, total_rows).alias("index"))
    indices_target_0 = df_collected.filter(
            (pl.col("target") == 0) & (pl.col("index") >= num_tokens)
    )
    return len(indices_target_0) * 2


def prepare_data_lazy(file_path, num_tokens, batch_size=32, shuffle=False, smote=False):
    """
    Prepare data for the transformer model using Polars and lazy loading,
    with shuffling done by randomly selecting indices, and processing in batches.
    
    Args:
        file_path (str): Path to the CSV file.
        num_prev (int): Number of previous time frames to include.
        batch_size (int): Number of rows to process in each batch.
        shuffle (bool): Whether to shuffle the data.
    
    Yields:
        input_tensor (tf.Tensor): Tensor with shape (batch_size, num_prev + 1, 4).
        target_tensor (tf.Tensor): Tensor with shape (batch_size,).
    """
    num_prev = num_tokens - 1
    # Load CSV lazily with Polars
    df_lazy = pl.scan_csv(file_path).select(['open_normalized', 'high_normalized', 'low_normalized', 'close_normalized', 'target'])
    
    # Collect the dataframe and determine total number of rows
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]
    if smote:
        df_collected = df_collected.with_columns(pl.arange(0, total_rows).alias("index"))
        indices_target_1 = df_collected.filter(
            (pl.col("target") == 1) & (pl.col("index") >= num_tokens)
        ).select("index").to_series().to_list()

        # Get indices where target is 0 and >= num_tokens
        indices_target_0 = df_collected.filter(
            (pl.col("target") == 0) & (pl.col("index") >= num_tokens)
        ).select("index").to_series().to_list()

        df_collected = df_collected.drop('index')


    while True:  # Loop to reshuffle and restart at each epoch
        # Create an array of indices to use for shuffling
        indices = list(range(num_prev, total_rows))
        if smote:
            indices_target_1_complete = []
            while len(indices_target_1_complete) < len(indices_target_0):
                indices_target_1_complete += indices_target_1

            indices_target_1_complete = indices_target_1_complete[:len(indices_target_0)]

            indices = indices_target_0 + indices_target_1_complete

        # Shuffle indices if required
        if shuffle:
            np.random.shuffle(indices)

        input_list = []
        target_list = []

        for idx in indices:
            # Fetch the previous `num_prev + 1` rows for the input based on the current index
            input_rows = df_collected[idx - num_prev:idx + 1, :-1].to_numpy()  # Exclude 'target' for input
            target_value = df_collected[idx, -1]  # Get 'target' for the target

            # Convert target to 0 if it's not 1
            target_value = 1 if target_value == 1 else 0

            input_list.append(input_rows)
            target_list.append(target_value)

            # Yield once we have enough for a batch
            if len(input_list) == batch_size:
                # Convert lists to NumPy arrays
                input_array = np.array(input_list)
                target_array = np.array(target_list)

                # Convert NumPy arrays to TensorFlow tensors
                input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
                target_tensor = tf.convert_to_tensor(target_array, dtype=tf.int32)

                yield input_tensor, target_tensor

                # Reset lists for the next batch
                input_list.clear()
                target_list.clear()
        
        # Optional: You can break after processing all indices if you want
        break  # Uncomment if you want to stop after one full pass

def bb_prepare_data_lazy(file_path, num_tokens, batch_size=32, shuffle=False, smote=False):
    """
    Prepare data for the transformer model using Polars and lazy loading,
    with shuffling done by randomly selecting indices, and processing in batches.
    
    Args:
        file_path (str): Path to the CSV file.
        num_prev (int): Number of previous time frames to include.
        batch_size (int): Number of rows to process in each batch.
        shuffle (bool): Whether to shuffle the data.
    
    Yields:
        input_tensor (tf.Tensor): Tensor with shape (batch_size, num_prev + 1, 4).
        target_tensor (tf.Tensor): Tensor with shape (batch_size,).
    """
    num_prev = num_tokens - 1
    # Load CSV lazily with Polars
    df_lazy = pl.scan_csv(file_path).select(['open_normalized', 'close_normalized', 'target'])
    
    # Collect the dataframe and determine total number of rows
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]
    if smote:
        df_collected = df_collected.with_columns(pl.arange(0, total_rows).alias("index"))
        indices_target_1 = df_collected.filter(
            (pl.col("target") == 1) & (pl.col("index") >= num_tokens)
        ).select("index").to_series().to_list()

        # Get indices where target is 0 and >= num_tokens
        indices_target_0 = df_collected.filter(
            (pl.col("target") == 0) & (pl.col("index") >= num_tokens)
        ).select("index").to_series().to_list()

        df_collected = df_collected.drop('index')


    while True:  # Loop to reshuffle and restart at each epoch
        # Create an array of indices to use for shuffling
        indices = list(range(num_prev, total_rows))
        if smote:
            indices_target_1_complete = []
            while len(indices_target_1_complete) < len(indices_target_0):
                indices_target_1_complete += indices_target_1

            indices_target_1_complete = indices_target_1_complete[:len(indices_target_0)]

            indices = indices_target_0 + indices_target_1_complete

        # Shuffle indices if required
        if shuffle:
            np.random.shuffle(indices)

        input_list = []
        target_list = []

        for idx in indices:
            # Fetch the previous `num_prev + 1` rows for the input based on the current index
            input_rows = df_collected[idx - num_prev:idx + 1, :-1].to_numpy()  # Exclude 'target' for input
            target_value = df_collected[idx, -1]  # Get 'target' for the target

            # Convert target to 0 if it's not 1
            target_value = 1 if target_value == 1 else 0

            input_list.append(input_rows)
            target_list.append(target_value)

            # Yield once we have enough for a batch
            if len(input_list) == batch_size:
                # Convert lists to NumPy arrays
                input_array = np.array(input_list)
                target_array = np.array(target_list)

                # Convert NumPy arrays to TensorFlow tensors
                input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
                target_tensor = tf.convert_to_tensor(target_array, dtype=tf.int32)

                yield input_tensor, target_tensor

                # Reset lists for the next batch
                input_list.clear()
                target_list.clear()
        
        # Optional: You can break after processing all indices if you want
        break  # Uncomment if you want to stop after one full pass


def bb_combined_prepare_data_lazy(file_path, num_tokens, batch_size=32, shuffle=False, smote=False):
    """
    Prepare data for the transformer model using Polars and lazy loading,
    with shuffling done by randomly selecting indices, and processing in batches.
    
    Args:
        file_path (str): Path to the CSV file.
        num_prev (int): Number of previous time frames to include.
        batch_size (int): Number of rows to process in each batch.
        shuffle (bool): Whether to shuffle the data.
    
    Yields:
        input_tensor (tf.Tensor): Tensor with shape (batch_size, num_prev + 1, 4).
        target_tensor (tf.Tensor): Tensor with shape (batch_size,).
    """
    num_prev = num_tokens - 1
    # Load CSV lazily with Polars
    df_lazy = pl.scan_csv(file_path).select([
        'open_normalized', 
        'high_normalized', 
        'low_normalized',
        'close_normalized', 
        'upper_normalized',
        'middle_normalized',
        'lower_normalized',
        'target'])
    
    # Collect the dataframe and determine total number of rows
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]
    if smote:
        df_collected = df_collected.with_columns(pl.arange(0, total_rows).alias("index"))
        indices_target_1 = df_collected.filter(
            (pl.col("target") == 1) & (pl.col("index") >= num_tokens)
        ).select("index").to_series().to_list()

        # Get indices where target is 0 and >= num_tokens
        indices_target_0 = df_collected.filter(
            (pl.col("target") == 0) & (pl.col("index") >= num_tokens)
        ).select("index").to_series().to_list()

        df_collected = df_collected.drop('index')


    while True:  # Loop to reshuffle and restart at each epoch
        # Create an array of indices to use for shuffling
        indices = list(range(num_prev, total_rows))
        if smote:
            indices_target_1_complete = []
            while len(indices_target_1_complete) < len(indices_target_0):
                indices_target_1_complete += indices_target_1

            indices_target_1_complete = indices_target_1_complete[:len(indices_target_0)]

            indices = indices_target_0 + indices_target_1_complete

        # Shuffle indices if required
        if shuffle:
            np.random.shuffle(indices)

        input_list = []
        bb_list = []
        target_list = []

        for idx in indices:
            # Fetch the previous `num_prev + 1` rows for the input based on the current index
            input_rows = df_collected[idx - num_prev:idx + 1, :-1].to_numpy()  # Exclude 'target' for input
            target_value = df_collected[idx, -1]  # Get 'target' for the target


            # Convert target to 0 if it's not 1
            target_value = 1 if target_value == 1 else 0

            input_list.append(input_rows[:, :4])
            bb_list.append(input_rows[:, 4:])
            target_list.append(target_value)

            # Yield once we have enough for a batch
            if len(input_list) == batch_size:
                # Convert lists to NumPy arrays
                input_array = np.array(input_list)
                bb_array = np.array(bb_list)
                target_array = np.array(target_list)

                # Convert NumPy arrays to TensorFlow tensors
                input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
                bb_tensor = tf.convert_to_tensor(bb_array, dtype=tf.float32)
                target_tensor = tf.convert_to_tensor(target_array, dtype=tf.int32)

                yield (input_tensor, bb_tensor), target_tensor

                # Reset lists for the next batch
                input_list.clear()
                bb_list.clear()
                target_list.clear()
        
        # Optional: You can break after processing all indices if you want
        break  # Uncomment if you want to stop after one full pass


# Create dataset generators
def create_dataset_generator(file_path, batch_size, num_tokens, shuffle=False, repeat=False, smote=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: prepare_data_lazy(file_path, batch_size=batch_size, num_tokens=num_tokens, shuffle=shuffle, smote=smote),
        output_signature=(  
            tf.TensorSpec(shape=(None, num_tokens, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset
 
def bb_create_dataset_generator(file_path, batch_size, num_tokens, shuffle=False, repeat=False, smote=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: bb_prepare_data_lazy(file_path, batch_size=batch_size, num_tokens=num_tokens, shuffle=shuffle, smote=smote),
        output_signature=(  
            tf.TensorSpec(shape=(None, num_tokens, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset 


def bb_combined_create_dataset_generator(file_path, batch_size, num_tokens, shuffle=False, repeat=False, smote=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: bb_combined_prepare_data_lazy(file_path, batch_size=batch_size, num_tokens=num_tokens, shuffle=shuffle, smote=smote),
        output_signature=(
            (tf.TensorSpec(shape=(None, num_tokens, 4), dtype=tf.float32),
             tf.TensorSpec(shape=(None, num_tokens, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset 
