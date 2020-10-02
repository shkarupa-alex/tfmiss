from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import QRNN, ToDense


class ImdbModel(keras.Model):
    CORE_LSTM = 'LSTM'
    CORE_QRNN = 'QRNN'

    def __init__(self, vocab_size, embed_size, core, units, dropout, embed_dropout):
        inputs = keras.layers.Input(shape=(None,), ragged=True)

        drop = keras.layers.Dropout(embed_dropout)
        encoder = keras.layers.Embedding(vocab_size, embed_size)

        if self.CORE_LSTM == core:
            sequence = keras.layers.LSTM(units=units, dropout=dropout)
        else:
            if not self.CORE_QRNN == core:
                raise ValueError('Wrong "core" value')
            sequence = QRNN(units=units, window=2, zoneout=dropout)
        sequence = keras.layers.Bidirectional(sequence)

        decoder = keras.layers.Dense(1, activation='sigmoid')

        outputs = ToDense(mask=True)(inputs)
        outputs = encoder(outputs)
        outputs = drop(outputs)
        outputs = sequence(outputs)
        outputs = decoder(outputs)

        super(ImdbModel, self).__init__(inputs=inputs, outputs=outputs)
