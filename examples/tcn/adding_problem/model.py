from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import TemporalConvNet


class AddingModel(keras.Model):
    CORE_GRU = 'GRU'
    CORE_LSTM = 'LSTM'
    CORE_TCN = 'TCN'

    def __init__(self, core, filters, kernel_size, dropout):
        inputs = keras.layers.Input(shape=(None, 2))

        if self.CORE_GRU == core:
            sequence = keras.layers.GRU(units=filters[0], dropout=dropout, return_sequences=True)
        elif self.CORE_LSTM == core:
            sequence = keras.layers.LSTM(units=filters[0], dropout=dropout, return_sequences=True)
        else:
            assert self.CORE_TCN == core
            sequence = TemporalConvNet(filters=filters, kernel_size=kernel_size, dropout=dropout)

        last = keras.layers.Lambda(lambda x: x[:, -1, :])
        predict = keras.layers.Dense(1)

        outputs = sequence(inputs)
        outputs = last(outputs)
        outputs = predict(outputs)

        super(AddingModel, self).__init__(inputs=inputs, outputs=outputs)
