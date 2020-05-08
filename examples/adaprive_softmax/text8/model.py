from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.keras.layers import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax


class Text8Model(tf.keras.Model):
    OUT_ASM = 'ASS'
    OUT_NCE = 'NCE'
    OUT_SS = 'SS'
    OUT_SM = 'SM'

    def __init__(self, seq_len, vocab_size, embed_size, units, core, dropout, cutoff, negatives):
        inputs = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='inputs')
        targets = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='targets')

        encoder = tf.keras.layers.Embedding(vocab_size, embed_size)
        sequence = tf.keras.layers.LSTM(units=units, dropout=dropout, return_sequences=True)

        if self.OUT_ASM == core:
            decoder = AdaptiveSoftmax(vocab_size, cutoff=cutoff)
        elif self.OUT_NCE == core:
            decoder = NoiseContrastiveEstimation(vocab_size, negatives=negatives)
        elif self.OUT_SS == core:
            decoder = SampledSofmax(vocab_size, negatives=negatives)
        else:
            if not self.OUT_SM == core:
                raise ValueError('Wrong "core" value')
            decoder = tf.keras.layers.Dense(vocab_size, activation='softmax')

        outputs = encoder(inputs)
        outputs = sequence(outputs)

        if self.OUT_SM == core:
            outputs = decoder(outputs)
        else:
            outputs = decoder([outputs, targets])

        super(Text8Model, self).__init__(inputs=[inputs, targets], outputs=outputs)
