from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow_addons.layers import WeightNormalization


class Cifar10Model(keras.Model):
    def __init__(self, weight_norm=False):
        super(Cifar10Model, self).__init__()

        # ZCA whitening should be applied to input
        self.noise = keras.layers.GaussianNoise(0.15, input_shape=(32, 32, 3))

        conv1 = keras.layers.Conv2D(96, 3, strides=1, activation='leaky_relu', padding='same')
        conv2 = keras.layers.Conv2D(96, 3, strides=1, activation='leaky_relu', padding='same')
        conv3 = keras.layers.Conv2D(96, 3, strides=1, activation='leaky_relu', padding='same')
        self.conv1 = WeightNormalization(conv1) if weight_norm else conv1
        self.conv2 = WeightNormalization(conv2) if weight_norm else conv2
        self.conv3 = WeightNormalization(conv3) if weight_norm else conv3

        self.pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.drop1 = keras.layers.Dropout(0.5)

        conv4 = keras.layers.Conv2D(192, 3, strides=1, activation='leaky_relu', padding='same')
        conv5 = keras.layers.Conv2D(192, 3, strides=1, activation='leaky_relu', padding='same')
        conv6 = keras.layers.Conv2D(192, 3, strides=1, activation='leaky_relu', padding='same')
        self.conv4 = WeightNormalization(conv4) if weight_norm else conv4
        self.conv5 = WeightNormalization(conv5) if weight_norm else conv5
        self.conv6 = WeightNormalization(conv6) if weight_norm else conv6

        self.pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.drop2 = keras.layers.Dropout(0.5)

        conv7 = keras.layers.Conv2D(192, 3, strides=2, activation='leaky_relu', padding='same')
        conv8 = keras.layers.Conv2D(192, 1, strides=1, activation='leaky_relu', padding='same')
        conv9 = keras.layers.Conv2D(192, 1, strides=1, activation='leaky_relu', padding='same')
        self.conv7 = WeightNormalization(conv7) if weight_norm else conv7
        self.conv8 = WeightNormalization(conv8) if weight_norm else conv8
        self.conv9 = WeightNormalization(conv9) if weight_norm else conv9

        self.pool3 = keras.layers.GlobalAveragePooling2D()

        dense = keras.layers.Dense(10, activation='softmax')
        self.dense = WeightNormalization(dense) if weight_norm else dense

    def call(self, inputs, training=None, mask=None):
        outputs = inputs
        if training:
            outputs = self.noise(outputs)

        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        outputs = self.pool1(outputs)
        outputs = self.drop1(outputs)

        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.conv6(outputs)

        outputs = self.pool2(outputs)
        outputs = self.drop2(outputs)

        outputs = self.conv7(outputs)
        outputs = self.conv8(outputs)
        outputs = self.conv9(outputs)

        outputs = self.pool3(outputs)
        outputs = self.dense(outputs)

        return outputs
