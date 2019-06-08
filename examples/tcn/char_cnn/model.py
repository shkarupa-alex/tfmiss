from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import TemporalConvNet


class CharModel(keras.Model):
    CORE_GRU = 'GRU'
    CORE_LSTM = 'LSTM'
    CORE_TCN = 'TCN'

    def __init__(self, seq_len, vocab_size, embed_size, core, kernels, kernel_size, dropout, embed_dropout):
        inputs = keras.layers.Input(shape=(seq_len,))

        drop = keras.layers.Dropout(embed_dropout)
        encoder = keras.layers.Embedding(vocab_size, embed_size)

        if self.CORE_GRU == core:
            sequence = keras.layers.GRU(units=kernels[0], dropout=dropout, return_sequences=True)
        elif self.CORE_LSTM == core:
            sequence = keras.layers.LSTM(units=kernels[0], dropout=dropout, return_sequences=True)
        else:
            assert self.CORE_TCN == core
            sequence = TemporalConvNet(kernels=kernels, kernel_size=kernel_size, dropout=dropout)

        decoder = keras.layers.TimeDistributed(
            keras.layers.Dense(vocab_size, activation='softmax')
        )

        outputs = drop(encoder(inputs))
        outputs = sequence(outputs)
        outputs = decoder(outputs)
        print(outputs)

        super(CharModel, self).__init__(inputs=inputs, outputs=outputs)
