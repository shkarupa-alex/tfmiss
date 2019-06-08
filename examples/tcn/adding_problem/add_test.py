from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.python.keras import backend as K
from model import AddingModel
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
    parser.add_argument('--seq_len', type=int, default=600, help='Sequence length')
    parser.add_argument('--nhid', type=int, default=24, help='Number of hidden units per layer')
    parser.add_argument('--levels', type=int, default=8, help='Number of levels')
    parser.add_argument('--ksize', type=int, default=8, help='Kernel size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout applied to layers')
    parser.add_argument('--lr', type=float, default=4e-3, help='Initial learning rate')
    parser.add_argument('--clip', type=float, default=5., help='Gradient clip')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch count')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--gru_units', type=int, default=150, help='Number of hidden units for GRU layer')
    parser.add_argument('--lstm_units', type=int, default=130, help='Number of hidden units for LSTM layer')
    argv, _ = parser.parse_known_args()

    np.random.seed(argv.seed)
    K.random_ops.random_seed.set_random_seed(argv.seed)

    train_dataset = data_generator(200000, argv.seq_len, argv.batch_size)
    test_dataset = data_generator(40000, argv.seq_len, argv.batch_size)

    kernels = {
        AddingModel.CORE_GRU: [argv.gru_units],
        AddingModel.CORE_LSTM: [argv.lstm_units],
        AddingModel.CORE_TCN: [argv.nhid] * argv.levels,

    }

    histories = {}
    for core in [AddingModel.CORE_GRU, AddingModel.CORE_LSTM, AddingModel.CORE_TCN]:
        model = AddingModel(
            core=core,
            kernels=kernels[core],
            kernel_size=argv.ksize,
            dropout=argv.dropout,
        )
        model.compile(
            run_eagerly=False,
            optimizer=keras.optimizers.Adam(
                lr=argv.lr,
                clipnorm=argv.clip
            ),
            loss='mse',
        )
        model.summary()
        histories[core] = model.fit_generator(
            generator=train_dataset,
            epochs=argv.epochs,
            validation_data=test_dataset,
        ).history

    interval = np.linspace(0, argv.epochs, argv.epochs)
    cmap = pyplot.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(histories.items())))

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['loss'], color=color, label=core)

    pyplot.title('Adding problem')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss.png')
    pyplot.close()
