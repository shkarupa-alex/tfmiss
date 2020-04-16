from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import TemporalConvNet


class CopyModel(keras.Model):
    CORE_GRU = 'GRU'
    CORE_LSTM = 'LSTM'
    CORE_TCN = 'TCN'
    CORE_TCN_HE = 'TCN_HE'

    def __init__(self, core, filters, kernel_size, dropout):
        inputs = keras.layers.Input(shape=(None, 1))

        if self.CORE_GRU == core:
            sequence = keras.layers.GRU(units=filters[0], dropout=dropout, return_sequences=True)
        elif self.CORE_LSTM == core:
            sequence = keras.layers.LSTM(units=filters[0], dropout=dropout, return_sequences=True)
        elif self.CORE_TCN == core:
            sequence = TemporalConvNet(filters=filters, kernel_size=kernel_size, dropout=dropout)
        else:
            if not self.CORE_TCN_HE == core:
                raise ValueError('Wrong "core" value')
            sequence = TemporalConvNet(
                filters=filters, kernel_size=kernel_size, dropout=dropout, kernel_initializer='he_uniform')

        predict = keras.layers.Dense(10, activation='softmax')  # Digits 0 - 9

        outputs = sequence(inputs)
        outputs = predict(outputs)

        super(CopyModel, self).__init__(inputs=inputs, outputs=outputs)
