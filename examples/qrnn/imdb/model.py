from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers, models
from tfmiss.keras.layers import QRNN, ToDense


class ImdbModel(models.Model):
    CORE_LSTM = 'LSTM'
    CORE_QRNN = 'QRNN'

    def __init__(self, vocab_size, embed_size, core, n_layers, units, dropout, layer_dropout):
        inputs = layers.Input(shape=(None,), ragged=True)

        outputs = layers.Embedding(vocab_size, embed_size)(inputs)
        outputs = ToDense(pad_value=0, mask=True)(outputs)

        for i in range(n_layers):
            not_last = i != n_layers - 1
            if self.CORE_LSTM == core:
                sequence = layers.LSTM(units=units, dropout=dropout, return_sequences=not_last)
            else:
                if not self.CORE_QRNN == core:
                    raise ValueError('Wrong "core" value')
                sequence = QRNN(units=units, window=2, zoneout=dropout, return_sequences=not_last)
            sequence = layers.Bidirectional(sequence)
            outputs = sequence(outputs)
            if not_last:
                outputs = layers.Dropout(layer_dropout)(outputs)

        decoder = layers.Dense(1, activation='sigmoid')
        outputs = decoder(outputs)

        super(ImdbModel, self).__init__(inputs=inputs, outputs=outputs)
