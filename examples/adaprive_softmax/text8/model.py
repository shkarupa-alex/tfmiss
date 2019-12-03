from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tfmiss.keras.layers import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax


class Text8Model(keras.Model):
    OUT_SM = 'SM'
    OUT_NCE = 'NCE'
    OUT_SS = 'SS'
    OUT_ASM = 'SS'

    def __init__(self, seq_len, vocab_size, embed_size, units, core, dropout, negatives):
        inputs = keras.layers.Input(shape=(seq_len,))

        encoder = keras.layers.Embedding(vocab_size, embed_size)
        sequence = keras.layers.LSTM(units=units, dropout=dropout, return_sequences=True)

        if self.OUT_ASM == core:
            decoder = AdaptiveSoftmax(vocab_size, cutoff=[1, 2, 3])
        elif self.OUT_NCE == core:
            decoder = NoiseContrastiveEstimation(vocab_size, negatives=negatives)
        elif self.OUT_SS == core:
            decoder = SampledSofmax(vocab_size, negatives=negatives)
        else:
            assert self.OUT_ASM == core
            decoder = keras.layers.Dense(vocab_size, activation='softmax')

        decoder = keras.layers.TimeDistributed(decoder)

        outputs = encoder(inputs)
        outputs = sequence(outputs)
        outputs = decoder(outputs)

        super(Text8Model, self).__init__(inputs=inputs, outputs=outputs)
