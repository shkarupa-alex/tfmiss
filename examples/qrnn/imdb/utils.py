import tensorflow as tf
from keras.src.datasets import imdb


def data_generator(batch_size):
    """
    Args:
        dataset: Dataset name
        seq_length: Length of sequence
        batch_size: Size of batch
    """
    vocab_size = 20000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train, y_train, x_test, y_test = (
        tf.ragged.constant(x_train),
        tf.constant(y_train[..., None]),
        tf.ragged.constant(x_test),
        tf.constant(y_test[..., None]),
    )

    # Shuffle only train dataset
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
        batch_size
    )

    return train_dataset, test_dataset, vocab_size
