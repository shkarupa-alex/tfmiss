from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers, models
from tfmiss.keras.layers import TemporalConvNet


class AddingModel(models.Model):
    CORE_GRU = 'GRU'
    CORE_LSTM = 'LSTM'
    CORE_TCN = 'TCN'
    CORE_TCN_HE = 'TCN_HE'

    def __init__(self, core, filters, kernel_size, dropout):
        inputs = layers.Input(shape=(None, 2))

        if self.CORE_GRU == core:
            sequence = layers.GRU(units=filters[0], dropout=dropout, return_sequences=True)
        elif self.CORE_LSTM == core:
            sequence = layers.LSTM(units=filters[0], dropout=dropout, return_sequences=True)
        elif self.CORE_TCN == core:
            sequence = TemporalConvNet(filters=filters, kernel_size=kernel_size, dropout=dropout)
        else:
            if not self.CORE_TCN_HE == core:
                raise ValueError('Wrong "core" value')
            sequence = TemporalConvNet(
                filters=filters, kernel_size=kernel_size, dropout=dropout, kernel_initializer='he_uniform')

        last = layers.Lambda(lambda x: x[:, -1, :])
        predict = layers.Dense(1)

        outputs = sequence(inputs)
        outputs = last(outputs)
        outputs = predict(outputs)

        super(AddingModel, self).__init__(inputs=inputs, outputs=outputs)
