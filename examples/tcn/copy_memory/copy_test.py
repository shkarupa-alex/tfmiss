from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from matplotlib import pyplot
from tf_keras import optimizers
from model import CopyModel
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Copying Memory Task')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers')
    parser.add_argument('--clip', type=float, default=5., help='Gradient clip')
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--ksize', type=int, default=8, help='kernel size')
    parser.add_argument('--levels', type=int, default=8, help='Number of levels')
    parser.add_argument('--blank_len', type=int, default=1000, help='The size of the blank (i.e. T)')
    parser.add_argument('--seq_len', type=int, default=10, help='Initial history size')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--nhid', type=int, default=10, help='number of hidden units per layer')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gru_units', type=int, default=75, help='Number of hidden units for GRU layer')
    parser.add_argument('--lstm_units', type=int, default=60, help='Number of hidden units for LSTM layer')
    argv, _ = parser.parse_known_args()

    np.random.seed(argv.seed)

    train_dataset = data_generator(argv.blank_len, argv.seq_len, 10000, argv.batch_size)
    test_dataset = data_generator(argv.blank_len, argv.seq_len, 1000, argv.batch_size)

    filters = {
        CopyModel.CORE_GRU: [argv.gru_units],
        CopyModel.CORE_LSTM: [argv.lstm_units],
        CopyModel.CORE_TCN: [argv.nhid] * argv.levels,
        CopyModel.CORE_TCN_HE: [argv.nhid] * argv.levels,
    }

    histories = {}
    for core in [CopyModel.CORE_LSTM, CopyModel.CORE_GRU, CopyModel.CORE_TCN, CopyModel.CORE_TCN_HE]:
        model = CopyModel(
            core=core,
            filters=filters[core],
            kernel_size=argv.ksize,
            dropout=argv.dropout,
        )
        model.compile(
            run_eagerly=False,
            optimizer=optimizers.RMSprop(argv.lr, clipnorm=argv.clip),
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

    pyplot.title('Copying problem')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss.png')
    pyplot.close()

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['val_accuracy'], color=color, label=core)

    pyplot.title('Copying problem')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.savefig('accuracy.png')
    pyplot.close()
