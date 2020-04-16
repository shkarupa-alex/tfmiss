from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.python.keras import backend as K
from model import MnistModel
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout applied to layers')
    parser.add_argument('--clip', type=float, default=5.0, help='Gradient clip')
    parser.add_argument('--epochs', type=int, default=20, help='Epoch count')
    parser.add_argument('--ksize', type=int, default=7, help='Kernel size (default: 7)')
    parser.add_argument('--levels', type=int, default=8, help='Number of levels')
    parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate')
    parser.add_argument('--nhid', type=int, default=25, help='Number of hidden units per layer')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed (default: 1111)')
    parser.add_argument('--permute', action='store_true', help='Use permuted MNIST')
    parser.add_argument('--gru_units', type=int, default=150, help='Number of hidden units for GRU layer')
    parser.add_argument('--lstm_units', type=int, default=130, help='Number of hidden units for LSTM layer')
    argv, _ = parser.parse_known_args()

    np.random.seed(argv.seed)
    K.random_ops.random_seed.set_random_seed(argv.seed)

    train_dataset, test_dataset = data_generator(argv.permute, argv.batch_size)

    filters = {
        MnistModel.CORE_GRU: [argv.gru_units],
        MnistModel.CORE_LSTM: [argv.lstm_units],
        MnistModel.CORE_TCN: [argv.nhid] * argv.levels,
        MnistModel.CORE_TCN_HE: [argv.nhid] * argv.levels,
    }

    histories = {}
    for core in [MnistModel.CORE_GRU, MnistModel.CORE_LSTM, MnistModel.CORE_TCN, MnistModel.CORE_TCN_HE]:
        model = MnistModel(
            core=core,
            filters=filters[core],
            kernel_size=argv.ksize,
            dropout=argv.dropout,
        )
        model.compile(
            run_eagerly=False,
            optimizer=keras.optimizers.Adam(
                lr=argv.lr,
                clipnorm=argv.clip
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        histories[core] = model.fit(
            train_dataset,
            epochs=argv.epochs,
            validation_data=test_dataset,
        ).history

    interval = np.linspace(0, argv.epochs, argv.epochs)
    cmap = pyplot.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(histories.items())))

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['loss'], color=color, label=core)

    pyplot.title('MNIST{} pixel'.format('(permuted)' if argv.permute else ''))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss{}.png'.format('_permuted' if argv.permute else ''))
    pyplot.close()

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['val_accuracy'], color=color, label=core)

    pyplot.title('MNIST{} pixel'.format('(permuted)' if argv.permute else ''))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.savefig('accuracy{}.png'.format('_permuted' if argv.permute else ''))
    pyplot.close()
