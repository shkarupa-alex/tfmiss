from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.python.keras import backend as K
from model import Text8Model
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout applied to layers')
    parser.add_argument('--epochs', type=int, default=100, help='Upper epoch limit')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--embed_size', type=int, default=100, help='Dimension of character embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='Number of hidden units per layer')
    parser.add_argument('--seq_len', type=int, default=400, help='Total sequence length, including effective history')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--cutoff', type=str, default='2000,10000', help='Adaptive softmax cutoffs')
    parser.add_argument('--negatives', type=int, default=10, help='Number of negative examples for sampled softmaxes')
    argv, _ = parser.parse_known_args()

    argv_cutoff = [int(cf) for cf in argv.cutoff.split(',')]
    if len(argv_cutoff) < 2:
        raise ValueError('Wrong adaptive softmax cutoff specified')

    np.random.seed(argv.seed)
    K.random_ops.random_seed.set_random_seed(argv.seed)

    train_dataset, test_dataset, vocab_size = data_generator(argv.seq_len, argv.batch_size)

    titles = ['AdaptiveSoftmax (default)', 'AdaptiveSoftmax (accurate)', 'AdaptiveSoftmax (fast)',
              'NoiseContrastiveEstimation', 'SampledSoftmax', 'Softmax']
    models = [
        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_ASM,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives
        ),
        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_ASM,
            dropout=argv.dropout,
            cutoff=[2030, 6774],
            negatives=argv.negatives
        ),
        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_ASM,
            dropout=argv.dropout,
            cutoff=[454, 6590],
            negatives=argv.negatives
        ),

        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_NCE,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives
        ),

        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_SS,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives
        ),

        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_SM,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives
        ),
    ]

    histories = {}
    for model, title in zip(models, titles):
        loss = 'sparse_categorical_crossentropy' if 'Softmax' == title else None
        model.compile(
            run_eagerly=False,
            optimizer=keras.optimizers.SGD(lr=argv.lr),
            loss='sparse_categorical_crossentropy',
        )
        model.summary()
        histories[title] = model.fit(
            train_dataset,
            epochs=argv.epochs,
            validation_data=test_dataset,
        ).history

    interval = np.linspace(0, argv.epochs, argv.epochs)
    cmap = pyplot.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(histories.items())))

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['loss'], color=color, label=core)

    pyplot.title('Character LM on {}'.format(argv.dataset.upper()))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss_{}.png'.format(argv.dataset))
    pyplot.close()

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, np.exp(m['loss']), color=color, label=core)
