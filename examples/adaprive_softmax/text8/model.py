import tensorflow as tf
from keras.src import layers
from keras.src import models

from tfmiss.keras.layers import AdaptiveSoftmax
from tfmiss.keras.layers import SampledSofmax


class Text8Model(models.Model):
    OUT_ASM = "ASS"
    OUT_SS = "SS"
    OUT_SM = "SM"

    def __init__(
        self,
        seq_len,
        vocab_size,
        embed_size,
        units,
        core,
        dropout,
        cutoff,
        negatives,
        return_probs,
    ):
        inputs = layers.Input(shape=(seq_len,), dtype=tf.int32, name="inputs")
        targets = layers.Input(shape=(seq_len,), dtype=tf.int32, name="targets")

        encoder = layers.Embedding(vocab_size, embed_size)
        sequence = layers.LSTM(
            units=units, dropout=dropout, return_sequences=True
        )

        if self.OUT_ASM == core:
            decoder = AdaptiveSoftmax(
                vocab_size, cutoff=cutoff, return_probs=return_probs
            )
        elif self.OUT_SS == core:
            decoder = SampledSofmax(
                vocab_size, negatives=negatives, return_probs=return_probs
            )
        else:
            if not self.OUT_SM == core:
                raise ValueError('Wrong "core" value')
            decoder = layers.Dense(vocab_size, activation="softmax")

        outputs = encoder(inputs)
        outputs = sequence(outputs)

        if self.OUT_SM == core:
            outputs = decoder(outputs)
        else:
            outputs = decoder([outputs, targets])

        super(Text8Model, self).__init__(
            inputs=[inputs, targets], outputs=outputs
        )
