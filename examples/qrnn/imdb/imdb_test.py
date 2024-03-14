from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
from matplotlib import pyplot
from tf_keras import optimizers
from model import ImdbModel
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout applied to layers')
    parser.add_argument('--layer_dropout', type=float, default=0.3, help='Dropout applied between layers')
    parser.add_argument('--epochs', type=int, default=15, help='Upper epoch limit')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--embed_size', type=int, default=300, help='Dimension of character embeddings')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--qrnn_layers', type=int, default=4, help='Number of QRNN layers')
    parser.add_argument('--qrnn_units', type=int, default=256, help='Number of hidden units for QRNN layer')
    parser.add_argument('--lstm_layers', type=int, default=4, help='Number of LSTM layers')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of hidden units for LSTM layer')
    argv, _ = parser.parse_known_args()

    np.random.seed(argv.seed)

    train_dataset, test_dataset, vocab_size = data_generator(argv.batch_size)

    layers = {
        ImdbModel.CORE_QRNN: argv.qrnn_layers,
        ImdbModel.CORE_LSTM: argv.lstm_layers,
    }
    filters = {
        ImdbModel.CORE_QRNN: argv.qrnn_units,
        ImdbModel.CORE_LSTM: argv.lstm_units,
    }

    histories = {}
    for core in [ImdbModel.CORE_LSTM, ImdbModel.CORE_QRNN]:
        model = ImdbModel(
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            core=core,
            n_layers=layers[core],
            units=filters[core],
            dropout=argv.dropout,
            layer_dropout=argv.layer_dropout
        )
        model.compile(
            run_eagerly=False,
            optimizer=optimizers.RMSprop(argv.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        start_time = time.time()
        histories[core] = model.fit(
            train_dataset,
            epochs=argv.epochs,
            validation_data=test_dataset,
        ).history
        histories[core]['time'] = int((time.time() - start_time) / 60)

    interval = np.linspace(0, argv.epochs, argv.epochs)
    cmap = pyplot.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(histories.items())))

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['loss'], color=color, dashes=[6, 2], label='{} ({} min)'.format(core, m['time']))
        pyplot.plot(interval, m['val_loss'], color=color, label='{} val ({} min)'.format(core, m['time']))

    pyplot.title('IMBD')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss.png')
    pyplot.close()

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['accuracy'], color=color, dashes=[6, 2], label='{} ({} min)'.format(core, m['time']))
        pyplot.plot(interval, m['val_accuracy'], color=color, label='{} val ({} min)'.format(core, m['time']))

    pyplot.title('IMBD')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.savefig('accuracy.png')
    pyplot.close()
