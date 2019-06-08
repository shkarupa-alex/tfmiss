from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import observations


def data_generator(dataset, seq_length, batch_size):
    """
    Args:
        dataset: Dataset name
        seq_length: Length of sequence
        batch_size: Size of batch
    """
    train_chars, test_chars, valid_chars = getattr(observations, dataset)('data/')
    test_chars += ' ' + valid_chars
    del valid_chars

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts([train_chars, test_chars])

    train_ids, test_ids = tokenizer.texts_to_sequences([train_chars, test_chars])
    vocab_size = max(train_ids + test_ids) + 1

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        # target_text = tf.expand_dims(target_text, axis=-1)  # required to match expected labels shape

        return input_text, target_text

    # Shuffle only train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_ids) \
        .batch(seq_length + 1, drop_remainder=True) \
        .map(split_input_target) \
        .shuffle(batch_size * 100) \
        .batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_ids) \
        .batch(seq_length + 1, drop_remainder=True) \
        .map(split_input_target) \
        .batch(batch_size)

    return train_dataset, test_dataset, vocab_size
