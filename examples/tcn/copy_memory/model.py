from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import TemporalConvNet


class CopyModel(keras.Model):
    CORE_GRU = 'GRU'
    CORE_LSTM = 'LSTM'
    CORE_TCN = 'TCN'

    def __init__(self, core, kernels, kernel_size, dropout):
        inputs = keras.layers.Input(shape=(None, 1))

        if self.CORE_GRU == core:
            sequence = keras.layers.GRU(units=kernels[0], dropout=dropout, return_sequences=True)
        elif self.CORE_LSTM == core:
            sequence = keras.layers.LSTM(units=kernels[0], dropout=dropout, return_sequences=True)
        else:
            assert self.CORE_TCN == core
            sequence = TemporalConvNet(kernels=kernels, kernel_size=kernel_size, dropout=dropout)

        predict = keras.layers.TimeDistributed(
            keras.layers.Dense(10, activation='softmax')  # Digits 0 - 9
        )

        outputs = sequence(inputs)
        outputs = predict(outputs)

        super(CopyModel, self).__init__(inputs=inputs, outputs=outputs)
