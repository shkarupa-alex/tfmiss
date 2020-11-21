from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import QRNN, ToDense


class ImdbModel(keras.Model):
    CORE_LSTM = 'LSTM'
    CORE_QRNN = 'QRNN'

    def __init__(self, vocab_size, embed_size, core, layers, units, dropout, embed_dropout):
        inputs = keras.layers.Input(shape=(None,), ragged=True)
        outputs = ToDense(mask=True)(inputs)

        encoder = keras.layers.Embedding(vocab_size, embed_size)
        outputs = encoder(outputs)

        drop = keras.layers.Dropout(embed_dropout)
        outputs = drop(outputs)

        for i in range(layers):
            not_last = i != layers - 1
            if self.CORE_LSTM == core:
                sequence = keras.layers.LSTM(units=units, dropout=dropout, return_sequences=not_last)
            else:
                if not self.CORE_QRNN == core:
                    raise ValueError('Wrong "core" value')
                sequence = QRNN(units=units, window=2, zoneout=dropout, return_sequences=not_last)
            sequence = keras.layers.Bidirectional(sequence)
            outputs = sequence(outputs)

        decoder = keras.layers.Dense(1, activation='sigmoid')
        outputs = decoder(outputs)

        super(ImdbModel, self).__init__(inputs=inputs, outputs=outputs)
