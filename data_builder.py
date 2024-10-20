import polars as pl
import tensorflow as tf
import numpy as np


def get_total_rows(file_path):
    # Count the total number of rows in the CSV file
    df_lazy = pl.scan_csv(file_path)
    return df_lazy.collect().shape[0]


def prepare_data_lazy(file_path, num_transformer_tokens, num_cnn_tokens, batch_size, shuffle=False):
    """
    Prepare data for the transformer model using Polars and lazy loading,
    with shuffling done by randomly selecting indices, and processing in batches.
    
    Args:
        file_path (str): Path to the CSV file.
        num_prev (int): Number of previous time frames to include.
        shuffle (bool): Whether to shuffle the data.
        batch_size (int): Number of rows to process in each batch.
    
    Yields:
        input_tensor (tf.Tensor): Tensor with shape (batch_size, num_prev + 1, 4).
        target_tensor (tf.Tensor): Tensor with shape (batch_size,).
    """

    num_prev = num_transformer_tokens - 1
    # Load CSV lazily with Polars
    df_lazy = pl.scan_csv(file_path).select(['open_normalized', 'high_normalized', 'low_normalized', 'close_normalized', 'target'])

    # Collect the dataframe and determine total number of rows
    df_collected = df_lazy.collect()
    total_rows = df_collected.shape[0]

    # Create an array of indices to use for shuffling
    indices = list(range(num_prev, total_rows))
    # Shuffle indices if required
    if shuffle:
        np.random.shuffle(indices)

    input_list = []
    cnn_input_list = []
    target_list = []

    # Process the data in batches
    while indices:
        batch_indices = [indices.pop(0) for _ in range(min(batch_size, len(indices)))]

        for idx in batch_indices:
            # Fetch the previous `num_prev + 1` rows for the input based on the current index
            input_rows = df_collected[idx - num_prev:idx + 1, :-1].to_numpy()  # Exclude 'target' for input
            target_value = df_collected[idx, -1]  # Get 'target' for the target

            # Convert target to 0 if it's not 1
            target_value = 1 if target_value == 1 else 0

            input_list.append(input_rows)
            cnn_input_list.append(input_rows[:num_cnn_tokens])  # Select the first `cnn_tokens` rows for the CNN input
            target_list.append(target_value)

        # Convert lists to NumPy arrays
        input_array = np.array(input_list)
        cnn_input_array = np.array(cnn_input_list)
        cnn_input_array = np.expand_dims(cnn_input_array, axis=-1)
        target_array = np.array(target_list)

        # Convert NumPy arrays to TensorFlow tensors
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
        cnn_tensor = tf.convert_to_tensor(cnn_input_array, dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(target_array, dtype=tf.int32)

        yield (input_tensor, cnn_tensor), target_tensor

        # Reset lists for the next batch
        input_list.clear()
        cnn_input_list.clear()
        target_list.clear()




# Rest of your model building and training code remains unchanged
def create_dataset_generator(file_path, batch_size, num_transformer_tokens, num_cnn_tokens, shuffle=False, repeat=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: prepare_data_lazy(file_path, batch_size=batch_size, num_transformer_tokens=num_transformer_tokens, num_cnn_tokens=num_cnn_tokens, shuffle=shuffle),
        output_signature=(
            (tf.TensorSpec(shape=(None, num_transformer_tokens, 4), dtype=tf.float32),
             tf.TensorSpec(shape=(None, num_cnn_tokens, 4, 1), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset

